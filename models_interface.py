"""
models_interface.py
-------------------
Interfaces for downstream specialist models.

Unified OCR Logic (v3.13):
  - Backend: RapidOCR
  - Fix: Quiet Zone padding + max 1500px downscale for OCR (no min upscale).
  - Fix: Non-ASCII artifact filter on OCR output.
  - Fix: Texo max_new_tokens + repetition_penalty to reduce hallucination.
  - Fix: Formula-specific preprocessing: Otsu binarization + aspect-ratio padding.
  - Fix: Moved all deferred stdlib imports to module level.
"""

import io
import sys
import os
import re
import time
import statistics
import torch
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from concurrent.futures import ThreadPoolExecutor

# ----------------------------------------------------------------
# Path handling and Windows DLL fix
# ----------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'Texo', 'src'))
sys.path.append(os.path.join(ROOT_DIR, 'text-table-latex'))

if sys.platform == 'win32':
    try:
        lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
        if os.path.exists(lib_path) and hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(lib_path)
    except Exception:
        pass


def escape_latex_chars(text: str) -> str:
    if not text:
        return text
    replacements = {
        '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_',
        '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'
    }
    for char, replacement in replacements.items():
        if char in text:
            text = text.replace(char, replacement)
    return text


def _filter_nonascii(text: str) -> str:
    """Replace non-ASCII characters with a space; they are OCR hallucinations in LaTeX context."""
    if all(ord(c) < 128 for c in text):
        return text
    return ''.join(c if ord(c) < 128 else ' ' for c in text)


# ----------------------------------------------------------------
# Singletons
# ----------------------------------------------------------------
_rapid_ocr       = None
_texo_model      = None
_texo_tokenizer  = None
_texo_processor  = None
_table_solver    = None
_yolo_model      = None
_yolo_model_path = None
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _get_rapidocr():
    global _rapid_ocr
    if _rapid_ocr is None:
        from rapidocr_onnxruntime import RapidOCR
        import importlib.util

        kwargs = dict(
            det_limit_type='max',
            det_limit_side_len=1280,
        )

        # Use English PP-OCRv4 rec model if available — smaller char set,
        # better accuracy on English-only academic text than the default
        # Chinese-dominant model.
        en_rec  = os.path.join(ROOT_DIR, 'en_PP-OCRv4_rec.onnx')
        paddle_spec = importlib.util.find_spec('paddleocr')
        if os.path.exists(en_rec) and paddle_spec is not None:
            import paddleocr as _poc
            en_dict = os.path.join(
                os.path.dirname(_poc.__file__), 'ppocr', 'utils', 'en_dict.txt'
            )
            if os.path.exists(en_dict):
                kwargs['rec_model_path'] = en_rec
                kwargs['rec_keys_path']  = en_dict
                print('[*] RapidOCR: using English PP-OCRv4 rec model')

        _rapid_ocr = RapidOCR(**kwargs)
    return _rapid_ocr


# Profiling stores
math_latencies       = []
math_batch_latencies = []
text_latencies       = []
table_latencies      = []
text_batch_latencies = []

def get_math_latencies():       return math_latencies
def get_math_batch_latencies(): return math_batch_latencies
def get_text_latencies():       return text_latencies
def get_table_latencies():      return table_latencies
def get_text_batch_latencies(): return text_batch_latencies


def _get_texo():
    global _texo_model, _texo_tokenizer, _texo_processor
    if _texo_model is None:
        from texo.data.processor import EvalMERImageProcessor
        from texo.model.formulanet import FormulaNet
        import texo.utils.config  # registers 'my_hgnetv2'
        from transformers import AutoTokenizer
        MODEL_PATH = os.path.join(ROOT_DIR, "Texo", "model")
        _texo_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        _texo_model = FormulaNet.from_pretrained(MODEL_PATH)
        _texo_model.eval().to(_device)
        _texo_processor = EvalMERImageProcessor(image_size={'width': 384, 'height': 384})
    return _texo_model, _texo_tokenizer, _texo_processor


def get_yolo_model(model_path: str):
    global _yolo_model, _yolo_model_path
    if _yolo_model is None or _yolo_model_path != model_path:
        from ultralytics import YOLO
        print(f"[*] Loading YOLO model: {model_path}")
        try:
            _yolo_model = YOLO(model_path, task='detect')
        except Exception as e:
            print(f"[!] Falling back to .pt: {e}")
            _yolo_model = YOLO("yolov11n-doclaynet.pt")
        _yolo_model_path = model_path
    return _yolo_model


def unload_texo():
    global _texo_model, _texo_tokenizer, _texo_processor
    if _texo_model is not None:
        del _texo_model, _texo_tokenizer, _texo_processor
        _texo_model     = None
        _texo_tokenizer = None
        _texo_processor = None
        import gc as _gc
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[*] Texo unloaded")


def _table_heuristic(tokens: list, img_w: int) -> str:
    """Coordinate-based table grid builder using RapidOCR token bboxes."""
    # 1. Group rows by Y centroid proximity
    tokens.sort(key=lambda t: t['cy'])
    rows = []
    current_row: list = []
    current_y = None
    for tok in tokens:
        if current_y is None:
            current_y = tok['cy']
            current_row.append(tok)
        elif abs(tok['cy'] - current_y) < max(tok['h'], 10) * 0.6:
            current_row.append(tok)
            current_y = sum(t['cy'] for t in current_row) / len(current_row)
        else:
            rows.append(current_row)
            current_row = [tok]
            current_y = tok['cy']
    if current_row:
        rows.append(current_row)

    # 2. X-axis projection to find column boundary gaps
    proj = np.zeros(img_w, dtype=int)
    for tok in tokens:
        if tok['w'] > 0.4 * img_w:
            continue
        proj[max(0, int(tok['x1'])):min(img_w, int(tok['x2']))] += 1

    in_col = False
    col_segments: list = []
    start = 0
    for i in range(img_w):
        if proj[i] > 0 and not in_col:
            in_col = True
            start = i
        elif proj[i] == 0 and in_col:
            in_col = False
            col_segments.append((start, i))
    if in_col:
        col_segments.append((start, img_w))
    if not col_segments:
        col_segments = [(0, img_w)]

    # 3. Assign tokens to column slots, build grid
    grid = []
    for row_tokens in rows:
        row_cells: list = [[] for _ in range(len(col_segments))]
        for tok in row_tokens:
            best_col, max_overlap, min_dist = 0, -1, float('inf')
            for i, (cs, ce) in enumerate(col_segments):
                overlap = max(0, min(tok['x2'], ce) - max(tok['x1'], cs))
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_col = i
                dist = (0 if cs <= tok['cx'] <= ce
                        else min(abs(tok['cx'] - cs), abs(tok['cx'] - ce)))
                if overlap == 0 and dist < min_dist and max_overlap <= 0:
                    min_dist = dist
                    best_col = i
            row_cells[best_col].append(tok)

        final_row = []
        for cell_tokens in row_cells:
            cell_tokens.sort(key=lambda t: t['x1'])
            final_row.append(escape_latex_chars(" ".join(t['text'] for t in cell_tokens)))
        if any(c.strip() for c in final_row):
            grid.append(final_row)

    # 4. Emit booktabs table
    if not grid:
        return ""
    col_count = max(len(row) for row in grid)
    lines = [f"\\begin{{tabular}}{{{'l' * col_count}}}", "\\toprule"]
    for i, row in enumerate(grid):
        padded = row + [""] * (col_count - len(row))
        lines.append(" & ".join(padded) + " \\\\")
        if i == 0:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────
# OCR ENGINE: RAPIDOCR
# ────────────────────────────────────────────────────────────────

def _preprocess_crop_for_ocr(crop: Image.Image, is_screenshot: bool = False) -> Image.Image:
    """
    Sharpen, contrast-boost, and optionally binarize a crop before RapidOCR.

    Screenshots skip all enhancement — they are already clean and high-contrast.
    Phone photos get: autocontrast (if low contrast) → unsharp mask → adaptive
    binarization (if still low contrast after sharpening).
    """
    if is_screenshot:
        return crop.convert("RGB")

    import cv2
    grey = crop.convert("L")
    arr  = np.array(grey, dtype=np.float32)
    rms_contrast = float(arr.std())

    result = crop.convert("RGB")

    if rms_contrast < 40:
        grey_eq = ImageOps.autocontrast(grey, cutoff=1)
        result  = Image.merge("RGB", [grey_eq, grey_eq, grey_eq])
        grey    = grey_eq

    result = result.filter(ImageFilter.UnsharpMask(radius=1.5, percent=180, threshold=3))

    grey_sharp = result.convert("L")
    rms_after  = float(np.array(grey_sharp, dtype=np.float32).std())
    if rms_after < 40:
        grey_np   = np.array(grey_sharp, dtype=np.uint8)
        binary_np = cv2.adaptiveThreshold(
            grey_np, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=25,
            C=12,
        )
        result = Image.fromarray(binary_np).convert("RGB")

    return result


def _reconstruct_lines(res: list) -> str:
    """
    Reconstruct word-spaced, newline-delimited text from RapidOCR results.

    RapidOCR returns: list of [bbox_polygon, text, confidence].
    bbox_polygon = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]] (4 corners, pixels).

    Groups detections into visual lines, inserts spaces where inter-detection
    gaps indicate a genuine word boundary, and joins lines with newlines.

    SPACE_GAP_FACTOR = 0.35: insert space when gap >= 35% of median char width.
    LINE_MERGE_FRAC = 0.6: detections within 60% of min-height of each other
      belong to the same line.
    """
    if not res:
        return ""

    SPACE_GAP_FACTOR = 0.35
    LINE_MERGE_FRAC  = 0.6

    items = []
    for entry in res:
        bbox, text, _ = entry[0], entry[1], entry[2]
        try:
            xs = [pt[0] for pt in bbox]
            ys = [pt[1] for pt in bbox]
        except (TypeError, IndexError):
            xs = [bbox[0], bbox[2]]
            ys = [bbox[1], bbox[3]]
        x_left  = min(xs);  x_right = max(xs)
        y_top   = min(ys);  y_bot   = max(ys)
        height  = max(y_bot - y_top, 1)
        items.append({
            'text': text, 'x_left': x_left, 'x_right': x_right,
            'y_ctr': (y_top + y_bot) / 2.0, 'height': height,
        })

    if not items:
        return ""

    # Group into visual lines
    items.sort(key=lambda d: d['y_ctr'])
    lines        = []
    current_line = [items[0]]

    for item in items[1:]:
        threshold = min(current_line[-1]['height'], item['height']) * LINE_MERGE_FRAC
        if abs(item['y_ctr'] - current_line[-1]['y_ctr']) <= threshold:
            current_line.append(item)
        else:
            lines.append(current_line)
            current_line = [item]
    lines.append(current_line)

    # Reconstruct spacing within each line
    output_lines = []
    for line in lines:
        line.sort(key=lambda d: d['x_left'])

        char_widths = [
            (d['x_right'] - d['x_left']) / max(len(d['text']), 1)
            for d in line
        ]
        median_cw   = statistics.median(char_widths) if char_widths else 8.0
        space_thresh = SPACE_GAP_FACTOR * median_cw

        parts = [line[0]['text']]
        for j in range(1, len(line)):
            gap = line[j]['x_left'] - line[j-1]['x_right']
            if gap >= space_thresh:
                parts.append(' ')
            parts.append(line[j]['text'])

        output_lines.append(''.join(parts))

    return '\n'.join(output_lines)


def _stitch_and_run(engine, processed_nps: list) -> list[list]:
    """
    Stitch processed numpy crops into one tall image and run one DBNet pass.
    Returns per-crop list of matching OCR entries (already y-shifted to local coords).
    """
    SEP   = 20
    max_w = max(p.shape[1] for p in processed_nps)
    strips    = []
    y_ranges  = []
    y = 0
    for p in processed_nps:
        h, w = p.shape[:2]
        canvas = np.full((h, max_w, 3), 255, dtype=np.uint8)
        canvas[:h, :w] = p
        y_ranges.append((y, y + h))
        strips.append(canvas)
        y += h + SEP
        strips.append(np.full((SEP, max_w, 3), 255, dtype=np.uint8))
    stitched = np.vstack(strips[:-1])

    res, _ = engine(stitched)
    per_crop = [[] for _ in processed_nps]
    if res:
        for entry in res:
            bbox_poly, text, conf = entry[0], entry[1], entry[2]
            try:
                cy = sum(pt[1] for pt in bbox_poly) / len(bbox_poly)
            except (TypeError, IndexError):
                cy = (bbox_poly[1] + bbox_poly[3]) / 2
            for crop_idx, (y_start, y_end) in enumerate(y_ranges):
                if y_start <= cy < y_end:
                    adj = [[pt[0], pt[1] - y_start] for pt in bbox_poly]
                    per_crop[crop_idx].append([adj, text, conf])
                    break
    return per_crop


def run_text_ocr_batched(
    crops: list[Image.Image],
    chunk_size: int = 10,
    is_screenshot: bool = False,
) -> list[str]:
    """
    Stitch crops into chunks of chunk_size, one DBNet pass per chunk.
    Chunking keeps stitched image height bounded so DBNet doesn't process
    an excessively tall image (det_limit_side_len=1280 max-type already
    helps, but chunking gives a second layer of protection).
    """
    global text_batch_latencies
    if not crops:
        return []
    try:
        engine        = _get_rapidocr()
        final_results = [""] * len(crops)

        # Screenshots are clean — smaller quiet zone and tighter size cap
        border   = 10 if is_screenshot else 30
        max_side = 960 if is_screenshot else 1500

        def process_single_crop(crop):
            sharpened = _preprocess_crop_for_ocr(crop, is_screenshot=is_screenshot)
            padded    = ImageOps.expand(sharpened, border=border, fill='white')
            max_dim   = max(padded.width, padded.height)
            if max_dim > max_side:
                scale  = max_side / max_dim
                padded = padded.resize(
                    (int(padded.width * scale), int(padded.height * scale)),
                    Image.Resampling.LANCZOS,
                )
            return np.array(padded.convert("RGB"))

        processed_nps = [process_single_crop(crop) for crop in crops]

        t_start = time.perf_counter()

        # Process in chunks to keep each stitched image height manageable.
        for chunk_start in range(0, len(crops), chunk_size):
            chunk_nps = processed_nps[chunk_start:chunk_start + chunk_size]
            per_crop  = _stitch_and_run(engine, chunk_nps)
            for ci, matches in enumerate(per_crop):
                if matches:
                    txt = _reconstruct_lines(matches)
                    txt = _filter_nonascii(txt)
                    final_results[chunk_start + ci] = escape_latex_chars(txt)

        t_end      = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000
        text_batch_latencies.append(latency_ms)
        n_chunks = (len(crops) + chunk_size - 1) // chunk_size
        print(f"    [text] RapidOCR {len(crops)} regions ({n_chunks} chunk(s)): {latency_ms:.2f} ms")

        return final_results
    except Exception as e:
        print(f"[RAPID ERROR] {e}")
        return [""] * len(crops)


def run_text_ocr(crop: Image.Image, is_screenshot: bool = False) -> str:
    return run_text_ocr_batched([crop], is_screenshot=is_screenshot)[0]


def run_text_ocr_full_page(
    image: Image.Image,
    text_detections: list,
    is_screenshot: bool = False,
) -> list[str]:
    """
    Run RapidOCR once on the full page, then assign each word to its enclosing
    YOLO detection bbox by center-point lookup.

    One full-page inference call instead of N per-crop calls — much faster when
    there are many text regions (e.g. 14 crops → 1 call).
    """
    if not text_detections:
        return []

    engine = _get_rapidocr()

    img = image.convert("RGB")
    # Screenshots are clean; phone photos get a light contrast boost
    if not is_screenshot:
        img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=180, threshold=3))
    img_np = np.array(img)

    t_start = time.perf_counter()
    res, _ = engine(img_np)
    latency_ms = (time.perf_counter() - t_start) * 1000
    text_batch_latencies.append(latency_ms)
    print(f"    [text] RapidOCR full-page ({len(text_detections)} regions): {latency_ms:.2f} ms")

    if not res:
        return [""] * len(text_detections)

    results = [""] * len(text_detections)
    for det_idx, det in enumerate(text_detections):
        x1, y1, x2, y2 = det["bbox"]
        matching = []
        for entry in res:
            bbox_poly, text, conf = entry[0], entry[1], entry[2]
            try:
                cx = sum(pt[0] for pt in bbox_poly) / len(bbox_poly)
                cy = sum(pt[1] for pt in bbox_poly) / len(bbox_poly)
            except (TypeError, IndexError):
                cx = (bbox_poly[0] + bbox_poly[2]) / 2
                cy = (bbox_poly[1] + bbox_poly[3]) / 2
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                matching.append(entry)
        if matching:
            text_out = _reconstruct_lines(matching)
            text_out = _filter_nonascii(text_out)
            results[det_idx] = escape_latex_chars(text_out)

    return results


# ────────────────────────────────────────────────────────────────
# MATH & TABLES
# ────────────────────────────────────────────────────────────────

# Maximum tokens Texo may generate per formula.
_TEXO_MAX_NEW_TOKENS  = 150
_TEXO_REPEAT_PENALTY  = 1.15

# Repeated filler sequences Texo hallucinates on bad crops (e.g. 200 tildes)
_REPEAT_TILDE_RE = re.compile(r'(~\s*){10,}')
_REPEAT_NEG_RE   = re.compile(r'(\\![\s\\!]*){10,}')
_REPEAT_QQUAD_RE = re.compile(r'(\\q{0,1}quad\s*){5,}')
# HTML-style arrows Texo occasionally emits
_ARROW_SUBS = [
    (re.compile(r'\\[Rr]arr\b'),  r'\\rightarrow'),
    (re.compile(r'\\[Ll]arr\b'),  r'\\leftarrow'),
    (re.compile(r'\\[Uu]arr\b'),  r'\\uparrow'),
    (re.compile(r'\\[Dd]arr\b'),  r'\\downarrow'),
]


def _sanitize_math_output(text: str) -> str:
    """Fix known Texo output artifacts that break LaTeX compilation."""
    if not text:
        return text
    # Arrow synonyms
    for pat, rep in _ARROW_SUBS:
        text = pat.sub(rep, text)
    # Collapse long repetitive filler sequences
    text = _REPEAT_TILDE_RE.sub('~ ', text)
    text = _REPEAT_NEG_RE.sub(r'\\! ', text)
    text = _REPEAT_QQUAD_RE.sub(r'\\qquad ', text)
    # Balance braces — trim trailing chars until { count == } count
    open_c  = text.count('{')
    close_c = text.count('}')
    if open_c > close_c:
        text = text + '}' * (open_c - close_c)
    elif close_c > open_c:
        excess = close_c - open_c
        for _ in range(excess):
            idx = text.rfind('}')
            if idx != -1:
                text = text[:idx] + text[idx + 1:]
    # Balance \begin{array} / \end{array} pairs (Texo hits token limit mid-formula)
    begin_count = text.count(r'\begin{array}')
    end_count   = text.count(r'\end{array}')
    if begin_count > end_count:
        text = text.rstrip() + r'\end{array}' * (begin_count - end_count)
    # Balance \left / \right pairs — imbalance causes cascade errors in equation env
    left_count  = text.count(r'\left')
    right_count = text.count(r'\right')
    if left_count > right_count:
        text = text.rstrip() + r'\right.' * (left_count - right_count)
    return text.strip()

def _preprocess_formula_crop(crop: Image.Image) -> Image.Image:
    """
    Prepare a formula crop for Texo's EvalMERImageProcessor.

    Otsu-binarize to pure black-on-white, matching Texo's training distribution
    (UNIMERNET_MEAN=0.7931 = near-white background).

    Note on aspect ratio: EvalMERImageProcessor.crop_margin() crops to the ink
    bounding box before resize, so any white padding we add is stripped. Black
    padding is preserved but dilutes the formula height proportionally — it does
    not actually improve character size in the 384×384 canvas. For very wide thin
    crops (aspect > 8:1), Texo will generate tilde garbage regardless; this is a
    fundamental limitation of 384×384 single-formula inference on multi-equation
    display rows that YOLO detects as one region.
    """
    import cv2
    arr = np.array(crop.convert("L"), dtype=np.uint8)
    _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binary).convert("RGB")


def run_math_recognition_batched(
    crops: list[Image.Image],
    fallback_figures_dir: str = None,
    fallback_counter: list = None,
) -> list[str]:
    global math_batch_latencies
    if not crops:
        return []
    try:
        model, tokenizer, processor = _get_texo()

        processed_list = [processor(_preprocess_formula_crop(c)) for c in crops]
        processed_images = torch.stack(processed_list).to(_device)

        t_start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                pixel_values=processed_images,
                max_new_tokens=_TEXO_MAX_NEW_TOKENS,
                repetition_penalty=_TEXO_REPEAT_PENALTY,
            )
        t_end      = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000
        math_batch_latencies.append(latency_ms)
        print(f"    [math] Texo batch ({len(crops)} eq): {latency_ms:.2f} ms")

        results_raw = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        final_results = []
        for i, result in enumerate(results_raw):
            clean_res = result.strip()
            for delim in ("$$", "$", r"\[", r"\]", r"\(", r"\)"):
                if clean_res.startswith(delim): clean_res = clean_res[len(delim):]
                if clean_res.endswith(delim):   clean_res = clean_res[:-len(delim)]
            clean_res = _sanitize_math_output(clean_res)
            if clean_res:
                final_results.append(clean_res)
            else:
                final_results.append(_math_fallback(crops[i], fallback_figures_dir, fallback_counter))
        return final_results
    except Exception as e:
        print(f"    [math] Texo batch failed: {e}")
        return [_math_fallback(c, fallback_figures_dir, fallback_counter) for c in crops]


def _math_fallback(
    crop: Image.Image,
    fallback_figures_dir: str,
    fallback_counter: list,
) -> str:
    if fallback_figures_dir and fallback_counter is not None:
        fallback_counter[0] += 1
        fname = f"formula_{fallback_counter[0]:03d}.png"
        fpath = os.path.join(fallback_figures_dir, fname)
        try:
            crop.save(fpath)
            return f"\\includegraphics[width=0.5\\linewidth]{{{fname}}}"
        except Exception:
            pass
    return ""


def run_math_recognition(
    crop: Image.Image,
    fallback_figures_dir: str = None,
    fallback_counter: list = None,
) -> str:
    return run_math_recognition_batched([crop], fallback_figures_dir, fallback_counter)[0]


def run_table_extraction(crop: Image.Image) -> str:
    """Extract table structure using RapidOCR (already-warm) + coordinate heuristic.

    Replaces EasyOCR + PaddleOCR TextAndTableSolver — eliminates 20-40s cold load.
    Same heuristic grid logic, zero extra model weight.
    """
    global table_latencies
    try:
        t_start = time.perf_counter()
        engine  = _get_rapidocr()
        img_np  = np.array(crop.convert("RGB"))
        img_w   = img_np.shape[1]

        result, _ = engine(img_np)

        tokens = []
        if result:
            for entry in result:
                try:
                    bbox_poly, text, _conf = entry[0], entry[1], entry[2]
                    text = text.strip()
                    if not text or re.match(r'^[\-\_\=\.]+$', text):
                        continue
                    xs = [pt[0] for pt in bbox_poly]
                    ys = [pt[1] for pt in bbox_poly]
                    x1, x2 = min(xs), max(xs)
                    y1, y2 = min(ys), max(ys)
                    if (x2 - x1) > 0.8 * img_w:
                        continue
                    tokens.append({
                        'text': text,
                        'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2,
                        'cx': (x1 + x2) / 2, 'cy': (y1 + y2) / 2,
                        'h': y2 - y1, 'w': x2 - x1,
                    })
                except (TypeError, IndexError, ValueError):
                    continue

        latex = _table_heuristic(tokens, img_w) if tokens else ""
        latency_ms = (time.perf_counter() - t_start) * 1000
        table_latencies.append(latency_ms)
        print(f"    [table] Table (RapidOCR): {latency_ms:.2f} ms")
        return latex
    except Exception as e:
        print(f"    [table] Table extraction failed: {e}")
        return ""
