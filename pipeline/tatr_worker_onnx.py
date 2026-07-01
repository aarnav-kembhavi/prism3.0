"""
tatr_worker_onnx.py
--------------------
Persistent subprocess for TATR table structure recognition via ONNX Runtime.

Drop-in replacement for tatr_worker.py that uses ONNX instead of PyTorch+transformers.
The main process never imports torch.

Benefits:
  - No torch/transformers in main process
  - Model size: 30 MB INT8 vs 115 MB safetensors
  - ~1.5x faster inference vs PyTorch CPU

Public API (identical to tatr_worker.py):
    from tatr_worker_onnx import TATROnnxWorker
    worker = TATROnnxWorker()
    worker.start()
    html = worker.build_table_html(crop_pil, tokens, img_w)
    worker.stop()
"""

import os
import sys
import multiprocessing as mp
import numpy as np
from pathlib import Path
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_PATH = Path(ROOT_DIR) / "models" / "tatr_structure_int8.onnx"

_MEAN    = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD     = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_MAX_SIDE = 800

# TATR id2label for structure-recognition-v1.1-all
_ID2LABEL = {
    0: "table",
    1: "table column",
    2: "table row",
    3: "table column header",
    4: "table projected row header",
    5: "table spanning cell",
    6: "no object",
}


# ── preprocessing (no torch) ──────────────────────────────────────────────────

def _preprocess(pil_img: Image.Image):
    """Resize longest-edge to <=800, normalize with ImageNet stats. Returns (array, orig_hw)."""
    w, h = pil_img.size
    scale = _MAX_SIDE / max(w, h)
    nw, nh = int(w * scale), int(h * scale)
    img = pil_img.convert("RGB").resize((nw, nh), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0          # [H, W, 3]
    arr = (arr - _MEAN) / _STD                             # normalize
    arr = arr.transpose(2, 0, 1)[np.newaxis]               # [1, 3, H, W]
    return arr, (h, w)


def _decode(logits, boxes, orig_hw, threshold=0.5):
    """DETR cxcywh -> xyxy in original pixel coords, filtered by score threshold."""
    probs  = _softmax(logits[0])[:, :-1]   # drop no-object class
    scores = probs.max(axis=1)
    labels = probs.argmax(axis=1)
    keep   = scores > threshold

    H, W = orig_hw
    b = boxes[0][keep]                     # [K, 4] cxcywh normalized
    cx, cy, bw, bh = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    xyxy = np.stack([
        (cx - bw / 2) * W,
        (cy - bh / 2) * H,
        (cx + bw / 2) * W,
        (cy + bh / 2) * H,
    ], axis=1)
    return labels[keep].tolist(), xyxy.tolist()


def _softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _overlap_1d(a1, a2, b1, b2):
    inter   = max(0.0, min(a2, b2) - max(a1, b1))
    shorter = min(a2 - a1, b2 - b1)
    return inter / shorter if shorter > 0 else 0.0


# ── worker process ────────────────────────────────────────────────────────────

def _worker_main(conn):
    import onnxruntime as ort
    sess_opts = ort.SessionOptions()
    sess_opts.inter_op_num_threads = 4
    sess_opts.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(_MODEL_PATH), sess_opts,
                                providers=["CPUExecutionProvider"])
    print(f"  [TATR-ONNX] ready ({_MODEL_PATH.name})", flush=True)
    conn.send("ready")

    while True:
        msg = conn.recv()
        if msg == "stop":
            break
        task, payload = msg
        try:
            if task == "detect":
                crop_arr, orig_hw = payload
                outputs = sess.run(None, {"pixel_values": crop_arr})
                logits, boxes = outputs[0], outputs[1]  # first two: logits, pred_boxes
                labels, xyxy = _decode(logits, boxes, orig_hw)
                conn.send(("ok", (labels, xyxy)))
            else:
                conn.send(("err", f"unknown task {task}"))
        except Exception as e:
            conn.send(("err", str(e)))


# ── public worker class ───────────────────────────────────────────────────────

class TATROnnxWorker:
    def __init__(self):
        self._proc = None
        self._conn = None

    def start(self):
        parent_conn, child_conn = mp.Pipe()
        self._conn = parent_conn
        self._proc = mp.Process(target=_worker_main, args=(child_conn,), daemon=True)
        self._proc.start()
        child_conn.close()
        msg = self._conn.recv()
        assert msg == "ready"

    def stop(self):
        if self._conn:
            try:
                self._conn.send("stop")
            except Exception:
                pass
        if self._proc and self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=5)

    def _detect_structure(self, crop: Image.Image, conf: float = 0.5):
        arr, orig_hw = _preprocess(crop)
        self._conn.send(("detect", (arr, orig_hw)))
        status, result = self._conn.recv()
        if status == "err":
            raise RuntimeError(result)
        labels, xyxy = result

        rows, cols = [], []
        for lbl, box in zip(labels, xyxy):
            cls = _ID2LABEL.get(lbl, "").lower()
            if "row" in cls and "header" not in cls:
                rows.append(box)
            elif "column" in cls:
                cols.append(box)

        rows.sort(key=lambda b: b[1])
        cols.sort(key=lambda b: b[0])

        crop_w = crop.size[0]
        cols = [c for c in cols if (c[2] - c[0]) < 0.85 * crop_w]
        return rows, cols

    def build_table_html(self, crop: Image.Image, tokens: list, img_w: int):
        """Same interface as tatr_worker.build_table_html."""
        from pipeline.models_interface import escape_latex_chars
        try:
            rows, cols = self._detect_structure(crop)
        except Exception as e:
            print(f"  [TATR-ONNX] error: {e}")
            return None

        if not rows or not cols:
            return None

        n_rows, n_cols = len(rows), len(cols)
        grid = [[[] for _ in range(n_cols)] for _ in range(n_rows)]

        for tok in tokens:
            best_r = max(range(n_rows),
                         key=lambda r: _overlap_1d(tok["y1"], tok["y2"],
                                                   rows[r][1], rows[r][3]))
            best_c = max(range(n_cols),
                         key=lambda c: _overlap_1d(tok["x1"], tok["x2"],
                                                   cols[c][0], cols[c][2]))
            grid[best_r][best_c].append(tok)

        for row in grid:
            for cell in row:
                cell.sort(key=lambda t: t["x1"])

        cell_grid = []
        for r in range(n_rows):
            row = [escape_latex_chars(" ".join(t["text"] for t in grid[r][c]))
                   for c in range(n_cols)]
            if any(c.strip() for c in row):
                cell_grid.append(row)

        if not cell_grid:
            return None

        lines = [f"\\begin{{tabular}}{{{'l' * n_cols}}}", "\\toprule"]
        for i, row in enumerate(cell_grid):
            lines.append(" & ".join(row) + " \\\\")
            if i == 0:
                lines.append("\\midrule")
        lines += ["\\bottomrule", "\\end{tabular}"]
        return "\n".join(lines)
