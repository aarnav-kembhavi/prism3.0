"""
models_interface.py
-------------------
Interfaces for downstream specialist models.

Unified OCR Logic (v3.11):
  - Backend: RapidOCR 
  - Fix: Added White-Space Padding (Quiet Zone) and Anti-Downscale safety limits.
"""

import io
import sys
import os
import torch
import numpy as np
from PIL import Image, ImageOps

# ----------------------------------------------------------------
# Path handling and Windows DLL fix
# ----------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'Texo', 'src'))
sys.path.append(os.path.join(ROOT_DIR, 'text-table-latex'))

if sys.platform == 'win32':
    try:
        import torch
        lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
        if os.path.exists(lib_path) and hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(lib_path)
    except:
        pass

def escape_latex_chars(text: str) -> str:
    if not text: return text
    replacements = {
        '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_',
        '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'
    }
    for char, replacement in replacements.items():
        if char in text: text = text.replace(char, replacement)
    return text


# ----------------------------------------------------------------
# Singletons
# ----------------------------------------------------------------
_rapid_ocr = None
_texo_model = None
_texo_tokenizer = None
_texo_processor = None
_table_solver = None
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _get_rapidocr():
    global _rapid_ocr
    if _rapid_ocr is None:
        from rapidocr_onnxruntime import RapidOCR
        _rapid_ocr = RapidOCR()
    return _rapid_ocr

# Profiling stores
math_latencies = []
math_batch_latencies = []
text_latencies = []
table_latencies = []
text_batch_latencies = []

def get_math_latencies(): return math_latencies
def get_math_batch_latencies(): return math_batch_latencies
def get_text_latencies(): return text_latencies
def get_table_latencies(): return table_latencies
def get_text_batch_latencies(): return text_batch_latencies

def _get_texo():
    global _texo_model, _texo_tokenizer, _texo_processor
    if _texo_model is None:
        from texo.data.processor import EvalMERImageProcessor
        from texo.model.formulanet import FormulaNet 
        from transformers import AutoTokenizer, VisionEncoderDecoderModel
        MODEL_PATH = os.path.join(ROOT_DIR, "Texo", "model") 
        _texo_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        _texo_model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)
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
         after step 2/3 (RMS < 55).

         Why binarization fixes word fusion:
         DBNet (RapidOCR's text detector) was designed for near-binary document
         images. On glare-damaged phone-photo crops, residual mesh artefacts and
         illumination gradients make inter-word white space appear grey (~180-220)
         rather than white (255). DBNet treats grey gaps as part of the ink region
         and returns the whole paragraph as one detection bbox — _reconstruct_lines
         then has nothing to split because there is only one box.

         OpenCV adaptiveThreshold (Gaussian, block=31, C=10) computes a local
         threshold for each pixel from its 31×31 neighbourhood. This is robust to
         illumination gradients: even if the left edge of the crop is darker than
         the right edge, each local neighbourhood is thresholded independently.
         The result is a clean black-ink-on-white image where inter-word gaps are
         genuinely white (255), which DBNet can segment into individual word boxes.

         Block size 31 and C=10 are calibrated for body-text at 15-30px cap height
         (typical after the 2000px floor resize). Smaller blocks cause salt-and-
         pepper noise in large ink strokes; larger blocks miss narrow inter-word gaps.

         Binarization is skipped for screenshots (already clean binary-like pixels)
         and for high-contrast phone crops (RMS >= 55 after sharpening) to avoid
         destroying anti-aliasing information the recognition model uses.
      5. Return as RGB (RapidOCR expects colour input).
    """
    import cv2
    from PIL import ImageFilter
    grey = crop.convert("L")
    arr = np.array(grey, dtype=np.float32)
    rms_contrast = float(arr.std())

    result = crop.convert("RGB")

    # Step 2: auto-contrast on low-contrast crops
    if rms_contrast < 40:
        grey_eq = ImageOps.autocontrast(grey, cutoff=1)
        result = Image.merge("RGB", [grey_eq, grey_eq, grey_eq])
        # Recompute grey for step 4
        grey = grey_eq

    # Step 3: unsharp mask
    result = result.filter(ImageFilter.UnsharpMask(radius=1.5, percent=180, threshold=3))

    # Step 4: adaptive binarization for phone photos with residual contrast issues
    if not is_screenshot:
        # Re-measure contrast after sharpening
        grey_sharp = result.convert("L")
        arr_sharp = np.array(grey_sharp, dtype=np.float32)
        rms_after = float(arr_sharp.std())

        if rms_after < 60:
            # Adaptive threshold: local Gaussian neighbourhood.
            # blockSize=25 works better than 31 for narrow column crops
            # (31 can be wider than a character at small font sizes).
            # C=12 gives a slightly more aggressive local threshold that
            # pushes inter-word grey gaps to white more reliably.
            grey_np = np.array(grey_sharp, dtype=np.uint8)
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

    RapidOCR returns: list of [bbox_polygon, text, confidence]
    where bbox_polygon = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]] (4 corners, pixels).

    Problem: on glare-damaged crops DBNet merges adjacent words into one
    detection region. The recognition model then returns the merged region
    as a single string with no internal spaces ("WeusetheUCI", "from30vol").

    Algorithm:
      1. Extract left-x, right-x, vertical-centre for each detection.
      2. Group detections into visual lines: detections whose vertical centres
         are within LINE_MERGE_FRAC of the crop height of each other belong
         to the same line. Sort lines top-to-bottom, detections left-to-right.
      3. Within each line, compute the median character width as:
             median_char_w = median(region_width / max(len(text), 1))
         Insert a space between consecutive regions when the horizontal gap
         (left edge of next − right edge of prev) exceeds SPACE_GAP_FACTOR
         times the median character width. A gap that large means a genuine
         inter-word space was present in the original but DBNet merged over it.
      4. Join lines with newline so _split_bullet_items() can work on them.

    SPACE_GAP_FACTOR = 0.45:  a space in typical fonts is ~0.25–0.5 em.
    Using 0.45× median char width catches real word gaps without inserting
    spurious spaces inside ligatures or tight letter pairs.

    LINE_MERGE_FRAC = 0.6:  two detections belong to the same line if their
    y-centres differ by less than 60% of the shorter detection's height.
    Robust to mixed-cap/body-text rows without merging adjacent text lines.
    """
    if not res:
        return ""

    SPACE_GAP_FACTOR = 0.45
    LINE_MERGE_FRAC  = 0.6

    # --- Step 1: parse bbox info ---
    items = []
    for entry in res:
        bbox, text, conf = entry[0], entry[1], entry[2]
        # bbox is [[x1,y1],[x2,y1],[x2,y2],[x1,y2]] (may be list or np.array)
        try:
            xs = [pt[0] for pt in bbox]
            ys = [pt[1] for pt in bbox]
        except (TypeError, IndexError):
            # Fallback: treat as flat [x1,y1,x2,y2]
            xs = [bbox[0], bbox[2]]
            ys = [bbox[1], bbox[3]]
        x_left  = min(xs)
        x_right = max(xs)
        y_top   = min(ys)
        y_bot   = max(ys)
        y_ctr   = (y_top + y_bot) / 2.0
        height  = max(y_bot - y_top, 1)
        items.append({
            'text': text, 'x_left': x_left, 'x_right': x_right,
            'y_ctr': y_ctr, 'height': height,
        })

    if not items:
        return ""

    # --- Step 2: group into lines ---
    items.sort(key=lambda d: d['y_ctr'])
    lines = []
    current_line = [items[0]]

    for item in items[1:]:
        prev_h = current_line[-1]['height']
        this_h = item['height']
        threshold = min(prev_h, this_h) * LINE_MERGE_FRAC
        if abs(item['y_ctr'] - current_line[-1]['y_ctr']) <= threshold:
            current_line.append(item)
        else:
            lines.append(current_line)
            current_line = [item]
    lines.append(current_line)

    # --- Step 3: reconstruct spacing within each line ---
    output_lines = []
    for line in lines:
        line.sort(key=lambda d: d['x_left'])

        # Median character width for this line
        char_widths = []
        for d in line:
            n_chars = max(len(d['text']), 1)
            char_widths.append((d['x_right'] - d['x_left']) / n_chars)
        import statistics
        median_cw = statistics.median(char_widths) if char_widths else 8.0
        space_thresh = SPACE_GAP_FACTOR * median_cw

        parts = [line[0]['text']]
        for j in range(1, len(line)):
            gap = line[j]['x_left'] - line[j-1]['x_right']
            if gap >= space_thresh:
                parts.append(' ')
            parts.append(line[j]['text'])

        output_lines.append(''.join(parts))

    return '\n'.join(output_lines)


def run_text_ocr_batched(crops: list[Image.Image], chunk_size: int = 10, is_screenshot: bool = False) -> list[str]:
    import time
    global text_batch_latencies
    if not crops: return []
    try:
        engine = _get_rapidocr()
        final_results = [""] * len(crops)
        
        t_start = time.perf_counter()
        
        for i, crop in enumerate(crops):
            # FIX 1: Sharpen and boost contrast before padding.
            # Glare-degraded crops have blurred edges that RapidOCR's DBNet
            # misreads as merged glyphs ("muluple", "Scripuutilling", etc.).
            sharpened = _preprocess_crop_for_ocr(crop, is_screenshot=is_screenshot)

            # FIX 2: Add a 30-pixel white border (Quiet Zone).
            # YOLO crops cut exactly on the letters. RapidOCR fails to recognise
            # edge characters (like brackets) without surrounding white space.
            padded_crop = ImageOps.expand(sharpened, border=30, fill='white')
            
            # FIX 3: Anti-Downscale Safety Limit.
            # RapidOCR's internal DBNet has a strict size limit. If a crop is too
            # large it violently shrinks it, destroying readability. Lanczos keeps
            # it within limits while preserving sharpness.
            # Raised cap from 1000 → 1500 so narrow single-column crops (which are
            # only ~400 px wide) are never upscaled past the safe zone.
            max_dim = max(padded_crop.width, padded_crop.height)
            if max_dim > 1500:
                scale = 1500 / max_dim
                new_w = int(padded_crop.width * scale)
                new_h = int(padded_crop.height * scale)
                padded_crop = padded_crop.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            img_np = np.array(padded_crop.convert("RGB"))
            res, _ = engine(img_np)
            
            if res:
                # FIX 4: Reconstruct word spacing from bbox coordinates, then
                # join lines with newline.
                #
                # RapidOCR returns one entry per detected text region as
                # [bbox_polygon, text, confidence]. On glare-damaged crops,
                # DBNet merges adjacent words into one detection region and
                # the recognition model returns them without internal spaces
                # ("WeusetheUCI" instead of "We use the UCI").
                #
                # Fix: group detections into visual lines by their vertical
                # centre, then within each line sort by x and insert a space
                # between consecutive regions whose horizontal gap exceeds a
                # threshold derived from the median character width of that
                # line. This reconstructs word boundaries that DBNet lost.
                final_results[i] = escape_latex_chars(
                    _reconstruct_lines(res)
                )
                
        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000
        text_batch_latencies.append(latency_ms)
        print(f"    [text] RapidOCR Processed {len(crops)} regions (Padded+Sharpened): {latency_ms:.2f} ms")
        
        return final_results
    except Exception as e:
        print(f"[RAPID ERROR] {e}"); return [""] * len(crops)

def run_text_ocr(crop: Image.Image, is_screenshot: bool = False) -> str:
    return run_text_ocr_batched([crop], is_screenshot=is_screenshot)[0]


# ────────────────────────────────────────────────────────────────
# MATH & TABLES
# ────────────────────────────────────────────────────────────────

def run_math_recognition_batched(crops: list[Image.Image], fallback_figures_dir: str = None,
                                 fallback_counter: list = None) -> list[str]:
    import time
    global math_batch_latencies
    if not crops: return []
    try:
        model, tokenizer, processor = _get_texo()
        t_start = time.perf_counter()
        image_list = [c.convert("RGB") for c in crops]
        processed_images = processor(image_list).to(_device)
        with torch.no_grad():
            outputs = model.generate(pixel_values=processed_images)
        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000
        math_batch_latencies.append(latency_ms)
        print(f"    [math] Texo Batch ({len(crops)} equations): {latency_ms:.2f} ms")
        results_raw = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        final_results = []
        for i, result in enumerate(results_raw):
            clean_res = ""
            if result:
                clean_res = result.strip()
                for delim in ("$$", "$", r"\[", r"\]", r"\(", r"\)"):
                    if clean_res.startswith(delim): clean_res = clean_res[len(delim):]
                    if clean_res.endswith(delim): clean_res = clean_res[:-len(delim)]
                clean_res = clean_res.strip()
            if clean_res: final_results.append(clean_res)
            else: final_results.append(_math_fallback(crops[i], fallback_figures_dir, fallback_counter))
        return final_results
    except Exception as e:
        print(f"    [math] Texo batch failed: {e}")
        return [_math_fallback(c, fallback_figures_dir, fallback_counter) for c in crops]

def _math_fallback(crop: Image.Image, fallback_figures_dir: str, fallback_counter: list) -> str:
    if fallback_figures_dir and fallback_counter is not None:
        import os
        fallback_counter[0] += 1
        fname = f"formula_{fallback_counter[0]:03d}.png"
        fpath = os.path.join(fallback_figures_dir, fname)
        try:
            crop.save(fpath); return f"\\includegraphics[width=0.5\\linewidth]{{{fname}}}"
        except: pass
    return ""

def run_math_recognition(crop: Image.Image, fallback_figures_dir: str = None,
                          fallback_counter: list = None) -> str:
    return run_math_recognition_batched([crop], fallback_figures_dir, fallback_counter)[0]

def run_table_extraction(crop: Image.Image) -> str:
    import time
    global table_latencies
    try:
        t_start = time.perf_counter()
        solver = _get_table_solver()
        image_np = np.array(crop.convert("RGB"))
        region = {"type": "Table", "image": image_np, "region_id": "0"}
        result = solver.solve(region)
        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000
        table_latencies.append(latency_ms)
        print(f"    [table] Table Solver latency: {latency_ms:.2f} ms")
        return result.get("latex", "")
    except Exception as e:
        print(f"    [table] Table Solver failed: {e}")
        return ""