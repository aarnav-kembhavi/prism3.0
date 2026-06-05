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
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _get_rapidocr():
    global _rapid_ocr
    if _rapid_ocr is None:
        from rapidocr_onnxruntime import RapidOCR
        _rapid_ocr = RapidOCR()
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
        _texo_model     = FormulaNet.from_pretrained(MODEL_PATH)
        _texo_model.eval().to(_device)
        _texo_processor = EvalMERImageProcessor(image_size={'width': 384, 'height': 384})
    return _texo_model, _texo_tokenizer, _texo_processor


def _get_table_solver():
    global _table_solver
    if _table_solver is None:
        from solver import TextAndTableSolver
        _table_solver = TextAndTableSolver()
    return _table_solver


# ────────────────────────────────────────────────────────────────
# OCR ENGINE: RAPIDOCR
# ────────────────────────────────────────────────────────────────

def _preprocess_crop_for_ocr(crop: Image.Image, is_screenshot: bool = False) -> Image.Image:
    """
    Sharpen, contrast-boost, and optionally binarize a crop before RapidOCR.

    Steps:
      1. Convert to greyscale, measure RMS contrast.
      2. If contrast is low (< 40), apply autocontrast to recover faded ink.
      3. Apply unsharp-mask to restore crisp letter edges.
      4. Phone photos only — adaptive binarization when contrast is still low
         after step 2/3 (RMS < 40).
      5. Return as RGB (RapidOCR expects colour input).

    The binarization (step 4) uses OpenCV adaptiveThreshold with a Gaussian
    neighbourhood (block=25, C=12) to handle illumination gradients robustly.
    It is skipped for screenshots (already clean) and high-contrast phone crops.
    """
    import cv2
    grey = crop.convert("L")
    arr  = np.array(grey, dtype=np.float32)
    rms_contrast = float(arr.std())

    result = crop.convert("RGB")

    # Step 2: auto-contrast on low-contrast crops
    if rms_contrast < 40:
        grey_eq = ImageOps.autocontrast(grey, cutoff=1)
        result  = Image.merge("RGB", [grey_eq, grey_eq, grey_eq])
        grey    = grey_eq

    # Step 3: unsharp mask
    result = result.filter(ImageFilter.UnsharpMask(radius=1.5, percent=180, threshold=3))

    # Step 4: adaptive binarization for low-contrast phone photos
    if not is_screenshot:
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
      Slightly lower than the previous 0.45 to catch tighter inter-word gaps
      on compressed or small-font crops without introducing spurious splits.
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


def run_text_ocr_batched(
    crops: list[Image.Image],
    chunk_size: int = 10,
    is_screenshot: bool = False,
) -> list[str]:
    global text_batch_latencies
    if not crops:
        return []
    try:
        engine        = _get_rapidocr()
        final_results = [""] * len(crops)

        t_start = time.perf_counter()

        def process_single_crop(crop):
            sharpened = _preprocess_crop_for_ocr(crop, is_screenshot=is_screenshot)

            # Quiet Zone: white border helps DBNet segment edge characters
            padded = ImageOps.expand(sharpened, border=30, fill='white')

            # Clamp oversized crops to ≤1500px (no min upscale — upscaling
            # many small crops simultaneously spikes RAM by 10x+)
            max_dim = max(padded.width, padded.height)
            if max_dim > 1500:
                scale = 1500 / max_dim
                padded = padded.resize(
                    (int(padded.width * scale), int(padded.height * scale)),
                    Image.Resampling.LANCZOS,
                )
            return np.array(padded.convert("RGB"))

        # Sequential preprocessing — parallel holds all enlarged arrays in RAM simultaneously
        processed_nps = [process_single_crop(crop) for crop in crops]

        # Sequential OCR calls (RapidOCR ONNX runtime is not thread-safe)
        for i, img_np in enumerate(processed_nps):
            res, _ = engine(img_np)
            if res:
                text = _reconstruct_lines(res)
                text = _filter_nonascii(text)
                final_results[i] = escape_latex_chars(text)

        t_end      = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000
        text_batch_latencies.append(latency_ms)
        print(f"    [text] RapidOCR {len(crops)} regions: {latency_ms:.2f} ms")

        return final_results
    except Exception as e:
        print(f"[RAPID ERROR] {e}")
        return [""] * len(crops)


def run_text_ocr(crop: Image.Image, is_screenshot: bool = False) -> str:
    return run_text_ocr_batched([crop], is_screenshot=is_screenshot)[0]


# ────────────────────────────────────────────────────────────────
# MATH & TABLES
# ────────────────────────────────────────────────────────────────

# Maximum tokens Texo may generate per formula.
_TEXO_MAX_NEW_TOKENS  = 300
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
    global table_latencies
    try:
        t_start = time.perf_counter()
        solver  = _get_table_solver()
        region  = {"type": "Table", "image": np.array(crop.convert("RGB")), "region_id": "0"}
        result  = solver.solve(region)
        latency_ms = (time.perf_counter() - t_start) * 1000
        table_latencies.append(latency_ms)
        print(f"    [table] Table Solver: {latency_ms:.2f} ms")
        return result.get("latex", "")
    except Exception as e:
        print(f"    [table] Table Solver failed: {e}")
        return ""
