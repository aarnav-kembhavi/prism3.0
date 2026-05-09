"""
models_interface.py
-------------------
Interfaces for downstream specialist models.

Unified OCR Logic (v3.8 Stable):
  - Backend: RapidOCR (ONNX + Big Montage)
  - Memory: ~1.0 GB peak
  - Optimization: Windows DLL fix and Parallel Routing active.
"""

import io
import sys
import os
import torch
import numpy as np
from PIL import Image

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
        # Stable v3.8 settings
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
# OCR ENGINE: RAPIDOCR v3.8 (Stable Big Montage)
# ────────────────────────────────────────────────────────────────
def run_text_ocr_batched(crops: list[Image.Image], chunk_size: int = 10) -> list[str]:
    import time
    global text_batch_latencies
    if not crops: return []
    try:
        engine = _get_rapidocr()
        final_results = [""] * len(crops)
        for chunk_idx in range(0, len(crops), chunk_size):
            chunk = crops[chunk_idx : chunk_idx + chunk_size]
            t_start = time.perf_counter()
            max_h = max(c.height for c in chunk); gap = 200
            total_w = sum(c.width for c in chunk) + (len(chunk) * gap)
            montage = Image.new("RGB", (total_w, max_h), (255, 255, 255))
            current_x = 0; x_offsets = []
            for c in chunk:
                y_off = (max_h - c.height) // 2
                montage.paste(c, (current_x, y_off)); x_offsets.append((current_x, current_x + c.width))
                current_x += c.width + gap
            img_np = np.array(montage); res, _ = engine(img_np)
            grouped_texts = [[] for _ in range(len(chunk))]
            if res:
                for bbox, text, conf in res:
                    xs = [p[0] for p in bbox]; cx = sum(xs) / len(xs)
                    for i, (x_start, x_end) in enumerate(x_offsets):
                        if x_start <= cx <= x_end: grouped_texts[i].append(text); break
            t_end = time.perf_counter()
            latency_ms = (t_end - t_start) * 1000
            text_batch_latencies.append(latency_ms)
            print(f"    [text] RapidOCR Montage Batch ({len(chunk)} regions): {latency_ms:.2f} ms")
            for i, parts in enumerate(grouped_texts):
                final_results[chunk_idx + i] = escape_latex_chars(" ".join(parts))
        return final_results
    except Exception as e:
        print(f"[RAPID ERROR] {e}"); return [""] * len(crops)

def run_text_ocr(crop: Image.Image) -> str:
    return run_text_ocr_batched([crop])[0]


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
        print(f"    [table] TATR latency: {latency_ms:.2f} ms")
        return result.get("latex", "")
    except Exception as e:
        print(f"    [table] Table Transformer solver failed: {e}")
        return ""
