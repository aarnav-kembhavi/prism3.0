"""
models_interface.py
-------------------
Interfaces for downstream specialist models.

Interface contract (DO NOT change function signatures):
  - All functions receive: image (PIL.Image.Image), a cropped
    region already extracted by the orchestrator
  - All functions return: str (the recognized content)
  - On failure: return empty string "", never raise
"""

import io
import sys
import os
import torch
import numpy as np
from PIL import Image

# ----------------------------------------------------------------
# Dynamically add custom module paths for Texo and text-table-latex
# ----------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'Texo', 'src'))
sys.path.append(os.path.join(ROOT_DIR, 'text-table-latex'))

def escape_latex_chars(text: str) -> str:
    """Escape special characters in plaintext OCR to prevent LaTeX compilation errors."""
    if not text:
        return text
    
    # Define replacements for common LaTeX special characters.
    replacements = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}'
    }
    
    # Replace characters one by one
    for char, replacement in replacements.items():
        if char in text:
            text = text.replace(char, replacement)
            
    return text


# ----------------------------------------------------------------
# Lazy-loaded singletons — models are heavy, load once on first call
# ----------------------------------------------------------------
_easyocr_reader = None
_texo_model = None
_texo_tokenizer = None
_texo_processor = None
_table_solver = None
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _get_easyocr():
    """Load EasyOCR reader once and cache it."""
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(["en"], gpu=False)
    return _easyocr_reader

# Store execution times for profiling
math_latencies = []
math_batch_latencies = []
text_latencies = []
table_latencies = []
text_batch_latencies = []

def get_math_latencies():
    return math_latencies

def get_math_batch_latencies():
    return math_batch_latencies

def get_text_latencies():
    return text_latencies

def get_table_latencies():
    return table_latencies

def get_text_batch_latencies():
    return text_batch_latencies

def _get_texo():
    """Load Texo Math OCR model, tokenizer, and processor."""
    global _texo_model, _texo_tokenizer, _texo_processor
    if _texo_model is None:
        from texo.data.processor import EvalMERImageProcessor
        from texo.model.formulanet import FormulaNet 
        from transformers import AutoTokenizer, VisionEncoderDecoderModel
        
        # Path to pre-downloaded weights in workspace
        MODEL_PATH = os.path.join(ROOT_DIR, "Texo", "model") 
        
        _texo_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        _texo_model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)
        _texo_model.eval().to(_device)
        
        _texo_processor = EvalMERImageProcessor(image_size={'width': 384, 'height': 384})
    return _texo_model, _texo_tokenizer, _texo_processor

def _get_table_solver():
    """Load the hybrid TATR + PaddleOCR Table Solver."""
    global _table_solver
    if _table_solver is None:
        from solver import TextAndTableSolver
        _table_solver = TextAndTableSolver() 
    return _table_solver


def run_text_ocr(crop: Image.Image) -> str:
    """
    Text recognition using EasyOCR (Sequential/Fallback).
    Input:  PIL crop of a text/header/caption/list region
    Output: recognized plaintext string
    """
    import time
    global text_latencies
    try:
        t_start = time.perf_counter()
        import numpy as np
        reader = _get_easyocr()
        img_array = np.array(crop)
        results = reader.readtext(img_array, detail=0)
        text = " ".join(results)
        t_end = time.perf_counter()
        
        latency_ms = (t_end - t_start) * 1000
        text_latencies.append(latency_ms)
        print(f"    [text] EasyOCR latency (seq): {latency_ms:.2f} ms")
        
        return escape_latex_chars(text)
    except Exception as e:
        print(f"[OCR ERROR] {type(e).__name__}: {e}")
        return ""


def run_text_ocr_batched(crops: list[Image.Image], chunk_size: int = 12) -> list[str]:
    """
    Batched text recognition using a Horizontal Montage strategy.
    """
    import time
    global text_batch_latencies
    if not crops:
        return []

    try:
        reader = _get_easyocr()
        
        final_results = [""] * len(crops)
        
        # Process in chunks to manage memory
        for chunk_idx in range(0, len(crops), chunk_size):
            chunk = crops[chunk_idx : chunk_idx + chunk_size]
            
            t_start = time.perf_counter()
            
            # 1. Build Horizontal Montage
            max_h = max(c.height for c in chunk)
            gap = 150
            
            total_w = sum(c.width for c in chunk) + (len(chunk) * gap)
            montage = Image.new("RGB", (total_w, max_h), (255, 255, 255))
            
            x_offsets = []
            current_x = 0
            for c in chunk:
                y_off = (max_h - c.height) // 2
                montage.paste(c, (current_x, y_off))
                x_offsets.append((current_x, current_x + c.width))
                current_x += c.width + gap
            
            # 2. Run OCR on Montage
            img_array = np.array(montage)
            results = reader.readtext(img_array, detail=1)
            
            # Logging detections
            char_count = sum(len(r[1]) for r in results)
            print(f"    [text] Montage OCR: detected {len(results)} boxes, {char_count} chars")
            
            # 3. Map Results back to original crops
            grouped_texts = [[] for _ in range(len(chunk))]
            mapped_count = 0
            for bbox, text, conf in results:
                xs = [p[0] for p in bbox]
                cx = sum(xs) / len(xs)
                for i, (x_start, x_end) in enumerate(x_offsets):
                    if x_start <= cx <= x_end:
                        grouped_texts[i].append(text)
                        mapped_count += 1
                        break
            
            if mapped_count < len(results):
                print(f"    [text] WARNING: {len(results) - mapped_count} boxes lost in mapping")
            
            t_end = time.perf_counter()
            latency_ms = (t_end - t_start) * 1000
            text_batch_latencies.append(latency_ms)
            print(f"    [text] EasyOCR Montage Batch ({len(chunk)} regions): {latency_ms:.2f} ms")

            # 4. Finalize strings
            for i, parts in enumerate(grouped_texts):
                combined = " ".join(parts)
                final_results[chunk_idx + i] = escape_latex_chars(combined)
                
        return final_results
        
    except Exception as e:
        print(f"[MONTAGE OCR ERROR] {type(e).__name__}: {e}")
        return [run_text_ocr(c) for c in crops]


def run_math_recognition_batched(crops: list[Image.Image], fallback_figures_dir: str = None,
                                 fallback_counter: list = None) -> list[str]:
    """
    Batched math recognition using Texo.
    """
    import time
    global math_batch_latencies
    if not crops:
        return []

    try:
        model, tokenizer, processor = _get_texo()
        
        t_start = time.perf_counter()
        
        # Prepare batch
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
            
            if clean_res:
                final_results.append(clean_res)
            else:
                final_results.append(_math_fallback(crops[i], fallback_figures_dir, fallback_counter))
                
        return final_results

    except Exception as e:
        print(f"    [math] Texo batch failed: {e}")
        return [_math_fallback(c, fallback_figures_dir, fallback_counter) for c in crops]

def _math_fallback(crop: Image.Image, fallback_figures_dir: str, fallback_counter: list) -> str:
    """Helper for saving formula crop on failure."""
    if fallback_figures_dir and fallback_counter is not None:
        import os
        fallback_counter[0] += 1
        fname = f"formula_{fallback_counter[0]:03d}.png"
        fpath = os.path.join(fallback_figures_dir, fname)
        try:
            crop.save(fpath)
            return f"\\includegraphics[width=0.5\\linewidth]{{{fname}}}"
        except: pass
    return ""

def run_math_recognition(crop: Image.Image, fallback_figures_dir: str = None,
                          fallback_counter: list = None) -> str:
    """Sequential wrapper for run_math_recognition_batched."""
    return run_math_recognition_batched([crop], fallback_figures_dir, fallback_counter)[0]


def run_table_extraction(crop: Image.Image) -> str:
    """
    Table recognition using TATR Table Transformer + OCR fallback in text-table-latex.
    """
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
        print(f"    [table] TATR latency: {latency_ms:.2f} ms")
        return result.get("latex", "")
    except Exception as e:
        print(f"    [table] Table Transformer solver failed: {e}")
        return ""
