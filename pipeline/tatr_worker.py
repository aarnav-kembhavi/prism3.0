"""
Table Transformer (TATR) structure recognition.
Model: microsoft/table-transformer-structure-recognition-v1.1-all

Uses manual torchvision preprocessing — AutoImageProcessor breaks on
transformers 4.42.x with the longest_edge size config this model uses.

Public API: build_table_html(crop, tokens, img_w) -> str | None
Returns None on failure; caller falls back to the coordinate heuristic.
"""
import torch
from PIL import Image

_tatr_model = None
_MODEL_ID   = "microsoft/table-transformer-structure-recognition-v1.1-all"
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]
_MAX_SIDE = 800


def _get_tatr():
    global _tatr_model
    if _tatr_model is None:
        from transformers import AutoModelForObjectDetection
        print(f"  [TATR] loading {_MODEL_ID} ...")
        _tatr_model = AutoModelForObjectDetection.from_pretrained(_MODEL_ID)
        _tatr_model.eval()
        print("  [TATR] ready")
    return _tatr_model


def _preprocess(pil_img: Image.Image):
    """Resize longest-edge to ≤800, normalize with ImageNet stats."""
    from torchvision import transforms as T
    w, h = pil_img.size
    scale = _MAX_SIDE / max(w, h)
    img = pil_img.convert("RGB").resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    t = T.Compose([T.ToTensor(), T.Normalize(_MEAN, _STD)])
    return t(img).unsqueeze(0), (h, w)   # orig_size as (H, W)


def _decode(outputs, orig_hw, threshold=0.5):
    """DETR cxcywh → xyxy in original pixel coords, filtered by threshold."""
    logits = outputs.logits[0]       # [N, C+1]
    boxes  = outputs.pred_boxes[0]   # [N, 4] cxcywh normalized

    probs  = logits.softmax(-1)[:, :-1]  # drop no-object
    scores, labels = probs.max(-1)
    keep   = scores > threshold

    H, W = orig_hw
    cx, cy, bw, bh = boxes[keep].unbind(-1)
    xyxy = torch.stack([
        (cx - bw / 2) * W,
        (cy - bh / 2) * H,
        (cx + bw / 2) * W,
        (cy + bh / 2) * H,
    ], dim=-1)
    return labels[keep].tolist(), xyxy.tolist()


def _overlap_1d(a1, a2, b1, b2):
    inter   = max(0.0, min(a2, b2) - max(a1, b1))
    shorter = min(a2 - a1, b2 - b1)
    return inter / shorter if shorter > 0 else 0.0


def _detect_structure(crop: Image.Image, conf: float = 0.5):
    """Return (row_boxes, col_boxes) in crop pixel coords, sorted spatially."""
    model = _get_tatr()
    tensor, orig_hw = _preprocess(crop)
    with torch.no_grad():
        outputs = model(pixel_values=tensor)
    labels, boxes = _decode(outputs, orig_hw, threshold=conf)

    id2label = model.config.id2label
    rows, cols = [], []
    for lbl, box in zip(labels, boxes):
        cls = id2label[lbl].lower()
        if "row" in cls and "header" not in cls:
            rows.append(box)
        elif "column" in cls:
            cols.append(box)

    rows.sort(key=lambda b: b[1])   # top → bottom
    cols.sort(key=lambda b: b[0])   # left → right

    # Drop columns that span nearly the full crop width — likely misdetections
    # from header cells or spanning rows being classified as columns.
    crop_w = crop.size[0]
    cols = [c for c in cols if (c[2] - c[0]) < 0.85 * crop_w]

    return rows, cols


def build_table_html(crop: Image.Image, tokens: list, img_w: int) -> str | None:
    """
    Build a LaTeX tabular string using TATR row/col structure + RapidOCR tokens.
    tokens: list of dicts with x1,x2,y1,y2,text (from _tokens_from_ocr_result).
    Returns None if TATR gives unusable results.
    """
    from pipeline.models_interface import escape_latex_chars

    try:
        rows, cols = _detect_structure(crop)
    except Exception as e:
        print(f"  [TATR] error: {e}")
        return None

    if not rows or not cols:
        return None

    n_rows, n_cols = len(rows), len(cols)
    grid = [[[] for _ in range(n_cols)] for _ in range(n_rows)]

    for tok in tokens:
        # assign to best row by 1-D overlap on Y axis
        best_r = max(range(n_rows),
                     key=lambda r: _overlap_1d(tok["y1"], tok["y2"], rows[r][1], rows[r][3]))
        # assign to best col by 1-D overlap on X axis
        best_c = max(range(n_cols),
                     key=lambda c: _overlap_1d(tok["x1"], tok["x2"], cols[c][0], cols[c][2]))
        grid[best_r][best_c].append(tok)

    # sort tokens within cells left-to-right
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
