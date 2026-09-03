# -*- coding: utf-8 -*-
"""
rtable_worker.py — parent-side handle for the RapidTable (SLANet-plus) child
process living in .venv_rtable. SLANet-plus (7.4 MB) is the primary table
structure recognizer (validated +29 TEDS on a stratified 60-table A/B,
catastrophic tables -0.01 -> 0.57). The OCR worker's coordinate heuristic is
the fallback when the child is unavailable or returns empty HTML.

Enable/disable with PRISM_RTABLE (default on). The child runs its own
PP-OCRv6-small det/rec on the table crop, so cell content no longer depends on
token-to-grid assignment.
"""
import io
import json
import os
import re
import struct
import subprocess
import threading

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_CHILD_SCRIPT = os.path.join(ROOT_DIR, "pipeline", "rtable_child.py")

# Preferred location is venvs/rtable; the legacy root path is the fallback so
# the venv can be relocated without breaking a running deployment. Windows
# layouts (Scripts/python.exe) are listed first because that is the only
# configuration this has been run on; the POSIX bin/python paths are there so a
# Linux/macOS venv is at least found rather than silently ignored.
_VENV_CANDIDATES = (
    os.path.join(ROOT_DIR, "venvs", "rtable", "Scripts", "python.exe"),
    os.path.join(ROOT_DIR, "venvs", "rtable", "bin", "python"),
    os.path.join(ROOT_DIR, ".venv_rtable", "Scripts", "python.exe"),
    os.path.join(ROOT_DIR, ".venv_rtable", "bin", "python"),
)
_VENV_PYTHON = next((p for p in _VENV_CANDIDATES if os.path.exists(p)),
                    _VENV_CANDIDATES[0])

_REQUEST_TIMEOUT_S = 30.0


class RapidTableUnavailable(RuntimeError):
    """RapidTable is enabled but its child interpreter is missing."""


def available() -> bool:
    """True if the RapidTable child can run.

    SLANet-plus is the primary table structure recognizer (+29 TEDS on a
    stratified 60-table A/B). It runs in a separate venv, which is not part of
    the repo and has to be built once. Previously a missing venv made this
    return False and the pipeline quietly dropped to the coordinate heuristic —
    no error, no warning, just materially worse table scores that were very hard
    to attribute. Now an unbuilt venv is a hard failure and opting out is
    explicit.
    """
    if os.environ.get("PRISM_RTABLE", "1") == "0":
        return False          # explicit opt-out: coordinate heuristic, no error

    if not os.path.exists(_CHILD_SCRIPT):
        raise RapidTableUnavailable(
            f"RapidTable child script missing: {_CHILD_SCRIPT}\n"
            "This means the checkout is incomplete — pipeline/rtable_child.py "
            "ships with the repo."
        )

    if not os.path.exists(_VENV_PYTHON):
        raise RapidTableUnavailable(
            "RapidTable is enabled (PRISM_RTABLE is not 0) but its interpreter "
            "was not found.\n"
            "Looked for:\n"
            + "".join(f"  {p}\n" for p in _VENV_CANDIDATES)
            + "\nThis venv is not part of the repo; build it once (SETUP.md, "
            '"RapidTable child venv"):\n'
            "  python -m venv venvs/rtable\n"
            "  venvs\\rtable\\Scripts\\python -m pip install rapid_table rapidocr\n"
            "\nOr run without table structure recognition - tables fall back to "
            "the coordinate heuristic and TEDS drops substantially:\n"
            "  set PRISM_RTABLE=0\n"
        )

    return True


class RapidTableWorker:
    """Persistent RapidTable child; call build_table_html(crop) -> HTML or ''."""

    def __init__(self):
        self._proc = None
        self._lock = threading.Lock()

    def start(self):
        if self._proc is not None:
            return
        self._proc = subprocess.Popen(
            [_VENV_PYTHON, _CHILD_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=ROOT_DIR,
        )
        # Wait for the 'ready' handshake (model load, first run downloads none —
        # weights are cached inside the venv site-packages).
        ready = self._read_msg(timeout=120.0)
        if ready != b"ready":
            self.stop()
            raise RuntimeError("rtable child failed to start")

    def stop(self):
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            proc.stdin.write(struct.pack(">I", 0))
            proc.stdin.flush()
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

    def _read_msg(self, timeout: float):
        """Read one length-prefixed message with a watchdog kill on timeout."""
        result = {}

        def _reader():
            try:
                header = self._proc.stdout.read(4)
                if len(header) < 4:
                    return
                (length,) = struct.unpack(">I", header)
                buf = b""
                while len(buf) < length:
                    chunk = self._proc.stdout.read(length - len(buf))
                    if not chunk:
                        return
                    buf += chunk
                result["data"] = buf
            except Exception:
                pass

        t = threading.Thread(target=_reader, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            try:
                self._proc.kill()
            except Exception:
                pass
            self._proc = None
            return None
        return result.get("data")

    def build_table_html(self, crop, ocr_tokens=None) -> str:
        """crop: PIL Image. Optional ocr_tokens = [(x1,y1,x2,y2,text,score),...]
        in crop coordinates — when given, SLANet uses them for cell content
        instead of its own (older) internal OCR. Returns '<table>...' or ''."""
        with self._lock:
            if self._proc is None:
                try:
                    self.start()
                except Exception:
                    return ""
            buf = io.BytesIO()
            crop.convert("RGB").save(buf, format="PNG")
            png = buf.getvalue()
            if ocr_tokens:
                import base64
                png = json.dumps({
                    "png": base64.b64encode(png).decode("ascii"),
                    "ocr": [[float(t[0]), float(t[1]), float(t[2]), float(t[3]),
                             str(t[4]), float(t[5])] for t in ocr_tokens],
                }).encode("utf-8")
            try:
                self._proc.stdin.write(struct.pack(">I", len(png)) + png)
                self._proc.stdin.flush()
            except Exception:
                self._proc = None
                return ""
            data = self._read_msg(timeout=_REQUEST_TIMEOUT_S)
            if not data:
                return ""
            try:
                html = json.loads(data.decode("utf-8")).get("html", "") or ""
            except Exception:
                return ""
            # RapidTable wraps output in <html><body>...</body></html>;
            # downstream wants the bare <table> element.
            m = re.search(r"<table\b.*</table>", html, re.DOTALL | re.IGNORECASE)
            return m.group(0) if m else ""
