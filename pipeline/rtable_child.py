# -*- coding: utf-8 -*-
"""
rtable_child.py — child process for RapidTable (SLANet-plus) table structure
recognition. Runs inside .venv_rtable (rapid_table + rapidocr 3.x), spoken to
by pipeline/rtable_worker.py over a length-prefixed stdio protocol:

    parent -> child:  4-byte big-endian length, then PNG bytes of the crop
    child  -> parent: 4-byte big-endian length, then UTF-8 JSON {"html": ...}

An empty html means recognition failed and the caller should fall back to TATR.
"""
import sys
import os
import json
import struct

# Keep ORT quiet and modest on threads: the parent pipeline already saturates cores.
os.environ.setdefault("OMP_NUM_THREADS", "4")


def _read_exact(stream, n):
    buf = b""
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def main():
    import numpy as np
    import cv2
    from rapid_table import RapidTable, RapidTableInput
    from rapid_table.utils.typings import ModelType

    engine = RapidTable(RapidTableInput(model_type=ModelType.SLANETPLUS, use_ocr=True))

    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    # Handshake so the parent can wait for model load.
    stdout.write(struct.pack(">I", 5) + b"ready")
    stdout.flush()

    while True:
        header = _read_exact(stdin, 4)
        if header is None:
            break
        (length,) = struct.unpack(">I", header)
        if length == 0:  # shutdown sentinel
            break
        png = _read_exact(stdin, length)
        if png is None:
            break
        html = ""
        try:
            img = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is not None and img.size > 0:
                out = engine(img)
                if out.pred_htmls:
                    html = out.pred_htmls[0] or ""
        except Exception as exc:  # never crash the loop; parent falls back
            sys.stderr.write(f"[rtable_child] {exc}\n")
            html = ""
        payload = json.dumps({"html": html}).encode("utf-8")
        stdout.write(struct.pack(">I", len(payload)) + payload)
        stdout.flush()


if __name__ == "__main__":
    main()
