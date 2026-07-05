# -*- coding: utf-8 -*-
"""
rtable_worker.py — parent-side handle for the RapidTable (SLANet-plus) child
process living in .venv_rtable. SLANet-plus replaces TATR as the primary table
structure recognizer (7.4 MB vs 30 MB; validated +29 TEDS on a stratified
60-table A/B, catastrophic tables -0.01 -> 0.57). TATR remains the fallback
when the child is unavailable or returns empty HTML.

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
_VENV_PYTHON = os.path.join(ROOT_DIR, ".venv_rtable", "Scripts", "python.exe")

_REQUEST_TIMEOUT_S = 30.0


def available() -> bool:
    return (
        os.environ.get("PRISM_RTABLE", "1") != "0"
        and os.path.exists(_VENV_PYTHON)
        and os.path.exists(_CHILD_SCRIPT)
    )


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

    def build_table_html(self, crop) -> str:
        """crop: PIL Image. Returns '<table>...</table>' or '' on failure."""
        with self._lock:
            if self._proc is None:
                try:
                    self.start()
                except Exception:
                    return ""
            buf = io.BytesIO()
            crop.convert("RGB").save(buf, format="PNG")
            png = buf.getvalue()
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
