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
    from pipeline.onnx_config import apply_session_threads
    sess_opts = ort.SessionOptions()
    apply_session_threads(sess_opts)
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

        rows, cols, spans = [], [], []
        for lbl, box in zip(labels, xyxy):
            cls = _ID2LABEL.get(lbl, "").lower()
            if "row" in cls and "header" not in cls:
                rows.append(box)
            elif "column" in cls:
                cols.append(box)
            elif "spanning" in cls:
                spans.append(box)

        rows.sort(key=lambda b: b[1])
        cols.sort(key=lambda b: b[0])

        crop_w = crop.size[0]
        if len(cols) > 1:
            # single-column tables ARE nearly full-width; only filter when the
            # near-full-width box duplicates a multi-column layout
            cols = [c for c in cols if (c[2] - c[0]) < 0.85 * crop_w]
        return rows, cols, spans

    @staticmethod
    def _split_token_by_cols(tok, cols):
        """Split an OCR token that straddles column boundaries.

        RapidOCR det boxes are line-level: tightly spaced adjacent cells come
        back as ONE token, which the old best-overlap assignment dumped whole
        into a single cell. Characters are apportioned by linear interpolation
        of the token's x-extent across the column edges it crosses.
        """
        hit = [c for c in range(len(cols))
               if _overlap_1d(tok["x1"], tok["x2"], cols[c][0], cols[c][2]) > 0
               and min(tok["x2"], cols[c][2]) - max(tok["x1"], cols[c][0])
               > 0.15 * (tok["x2"] - tok["x1"])]
        if len(hit) <= 1:
            return [tok]
        text = tok["text"]
        w = tok["x2"] - tok["x1"]
        if not text or w <= 0:
            return [tok]
        pieces = []
        for c in hit:
            lo = max(tok["x1"], cols[c][0])
            hi = min(tok["x2"], cols[c][2])
            i0 = int(round((lo - tok["x1"]) / w * len(text)))
            i1 = int(round((hi - tok["x1"]) / w * len(text)))
            part = text[i0:i1].strip()
            if part:
                pieces.append({"text": part, "x1": lo, "x2": hi,
                               "y1": tok["y1"], "y2": tok["y2"]})
        return pieces if pieces else [tok]

    def build_table_html(self, crop: Image.Image, tokens: list, img_w: int):
        """Same interface as tatr_worker.build_table_html. Returns a LaTeX
        tabular; with PRISM_TBL_V2=1 spanning cells become \\multicolumn /
        \\multirow (tex_to_md converts those to colspan/rowspan HTML)."""
        import os
        from pipeline.models_interface import escape_latex_chars
        tbl_v2 = os.environ.get('PRISM_TBL_V2', '1') != '0'
        try:
            rows, cols, spans = self._detect_structure(crop)
        except Exception as e:
            print(f"  [TATR-ONNX] error: {e}")
            return None

        if not rows or not cols:
            return None

        n_rows, n_cols = len(rows), len(cols)
        grid = [[[] for _ in range(n_cols)] for _ in range(n_rows)]

        if tbl_v2:
            split_tokens = []
            for tok in tokens:
                split_tokens.extend(self._split_token_by_cols(tok, cols))
            tokens = split_tokens

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

        # spanning cells -> (r0, r1, c0, c1) anchor map
        span_of = {}       # (r, c) anchor -> (rspan, cspan)
        consumed = set()   # (r, c) covered by an anchor's span
        if tbl_v2 and spans:
            def _covered(cells_boxes, s, axis):
                out = []
                for i, b in enumerate(cells_boxes):
                    lo, hi = (b[1], b[3]) if axis == 'y' else (b[0], b[2])
                    if _overlap_1d(lo, hi, s[1] if axis == 'y' else s[0],
                                   s[3] if axis == 'y' else s[2]) >= 0.5:
                        out.append(i)
                return out
            for s in spans:
                rr = _covered(rows, s, 'y')
                cc = _covered(cols, s, 'x')
                if not rr or not cc or (len(rr) == 1 and len(cc) == 1):
                    continue
                r0, r1, c0, c1 = min(rr), max(rr), min(cc), max(cc)
                if (r0, c0) in consumed or (r0, c0) in span_of:
                    continue
                span_of[(r0, c0)] = (r1 - r0 + 1, c1 - c0 + 1)
                for r in range(r0, r1 + 1):
                    for c in range(c0, c1 + 1):
                        if (r, c) != (r0, c0):
                            consumed.add((r, c))
            # pool consumed cells' tokens into their anchor
            for (r0, c0), (rs, cs) in span_of.items():
                pooled = []
                for r in range(r0, r0 + rs):
                    for c in range(c0, c0 + cs):
                        pooled.extend(grid[r][c])
                        if (r, c) != (r0, c0):
                            grid[r][c] = []
                pooled.sort(key=lambda t: (t["y1"], t["x1"]))
                grid[r0][c0] = pooled

        def cell_text(r, c):
            return escape_latex_chars(" ".join(t["text"] for t in grid[r][c]))

        lines = [f"\\begin{{tabular}}{{{'l' * n_cols}}}", "\\toprule"]
        emitted = 0
        for r in range(n_rows):
            parts = []
            row_has_content = any(grid[r][c] for c in range(n_cols))
            if not tbl_v2 and not row_has_content:
                continue   # legacy behaviour: skip empty rows
            c = 0
            while c < n_cols:
                if (r, c) in span_of:
                    rs, cs = span_of[(r, c)]
                    inner = cell_text(r, c)
                    if rs > 1:
                        inner = f"\\multirow{{{rs}}}{{*}}{{{inner}}}"
                    if cs > 1:
                        parts.append(f"\\multicolumn{{{cs}}}{{l}}{{{inner}}}")
                    else:
                        parts.append(inner)
                    c += cs
                elif (r, c) in consumed:
                    # covered by a rowspan anchor above: placeholder
                    anchor = next(((ar, ac) for (ar, ac), (ars, acs) in span_of.items()
                                   if ar <= r < ar + ars and ac <= c < ac + acs), None)
                    cs = span_of[anchor][1] if anchor else 1
                    if cs > 1:
                        parts.append(f"\\multicolumn{{{cs}}}{{l}}{{}}")
                    else:
                        parts.append("")
                    c += cs
                else:
                    parts.append(cell_text(r, c))
                    c += 1
            lines.append(" & ".join(parts) + " \\\\")
            emitted += 1
            if emitted == 1:
                lines.append("\\midrule")
        if emitted == 0:
            return None
        lines += ["\\bottomrule", "\\end{tabular}"]
        return "\n".join(lines)
