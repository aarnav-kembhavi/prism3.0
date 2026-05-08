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
    # We don't escape backslash here to avoid double-escaping if we add logic later,
    # but the usual suspects from OCR are _, &, %, $, #.
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

def get_math_latencies():
    return math_latencies

def _get_texo():
    """Load Texo Math OCR model, tokenizer, and processor."""
    global _texo_model, _texo_tokenizer, _texo_processor
    if _texo_model is None:
        from texo.data.processor import EvalMERImageProcessor
        # Critical: formulanet MUST be imported to register the custom 'my_hgnetv2' architecture
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
    Text recognition using EasyOCR.
    Input:  PIL crop of a text/header/caption/list region
    Output: recognized plaintext string
    """
    try:
        import numpy as np
        reader = _get_easyocr()
        img_array = np.array(crop)
        results = reader.readtext(img_array, detail=0)
        text = " ".join(results)
        return escape_latex_chars(text)
    except Exception as e:
        print(f"[OCR ERROR] {type(e).__name__}: {e}")
        return ""


def run_math_recognition(crop: Image.Image, fallback_figures_dir: str = None,
                          fallback_counter: list = None) -> str:
    """
    Math/formula recognition using Texo.
    Input:  PIL crop of a formula/equation region
    Output: LaTeX math string WITHOUT delimiters
            e.g.  "E = mc^{2}"  not  "$E = mc^{2}$"

    Fix D — fallback: if Texo returns an empty string (failed recognition),
    save the crop as a PNG and return an \\includegraphics reference so the
    formula is not silently lost from the output document.  The caller must
    pass fallback_figures_dir (the output folder path) and fallback_counter
    (a one-element list [n] so we can mutate the counter across calls).
    """
    import time
    global math_latencies
    try:
        model, tokenizer, processor = _get_texo()

        image_rgb = crop.convert("RGB")
        processed_image = processor(image_rgb).unsqueeze(0).to(_device)

        t_start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(pixel_values=processed_image)
        t_end = time.perf_counter()

        latency_ms = (t_end - t_start) * 1000
        math_latencies.append(latency_ms)
        print(f"    [math] Texo latency: {latency_ms:.2f} ms")

        result = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        if result:
            result = result.strip()
            for delim in ("$$", "$", r"\[", r"\]", r"\(", r"\)"):
                if result.startswith(delim):
                    result = result[len(delim):]
                if result.endswith(delim):
                    result = result[:-len(delim)]
            result = result.strip()

        if result:
            return result

        # ---- Fallback: save crop as image --------------------------------
        raise ValueError("Texo returned empty string")

    except Exception as e:
        print(f"    [math] Texo failed: {e}")
        # Fix D fallback: save formula crop as image so it is NOT lost
        if fallback_figures_dir and fallback_counter is not None:
            import os
            fallback_counter[0] += 1
            fname = f"formula_{fallback_counter[0]:03d}.png"
            fpath = os.path.join(fallback_figures_dir, fname)
            try:
                crop.save(fpath)
                print(f"    [math] Saved formula crop → {fname}")
                return f"\\includegraphics[width=0.5\\linewidth]{{{fname}}}"
            except Exception as save_err:
                print(f"    [math] Could not save formula crop: {save_err}")
        return ""


def run_table_extraction(crop: Image.Image) -> str:
    """
    Table recognition using TATR Table Transformer + OCR fallback in text-table-latex.
    Input:  PIL crop of a table region
    Output: LaTeX tabular environment string, complete,
            e.g. "\\begin{tabular}{cc}\\hline A & B \\\\ \\hline \\end{tabular}"
    """
    try:
        solver = _get_table_solver()
        
        # The solver expects an OpenCV/NumPy format and a region dictionary
        image_np = np.array(crop.convert("RGB"))
        region = {
            "type": "Table",
            "image": image_np,
            "region_id": "0" 
        }
        
        # .solve() executes TATR -> OCR -> postprocessing cascade
        result = solver.solve(region)
        
        return result.get("latex", "")
    except Exception as e:
        print(f"    [table] Table Transformer solver failed: {e}")
        return ""