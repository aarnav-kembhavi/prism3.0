import os
import sys
import json
import re

import numpy as np
import torch

# ----------------------------------------------------------------
# Windows DLL Search Path Fix (Required for PaddleOCR on Win)
# ----------------------------------------------------------------
if sys.platform == 'win32':
    try:
        import torch
        lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
        if os.path.exists(lib_path) and hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(lib_path)
    except:
        pass

from PIL import Image
from transformers import AutoImageProcessor, TableTransformerForObjectDetection

from config import CONFIG, TATR_MODEL
from utils import (
    latex_escape,
    strip_bullet,
    html_table_to_grid,
    grid_to_tabular,
    clean_table_grid,
    clean_latex_table,
)
from preprocess import preprocess_image


# ─────────────────────────────────────────────
# Shared Table Formatting (Booktabs) & Cleaning
# ─────────────────────────────────────────────

_I_FIX = [
    (re.compile(r'\bI(\d)'),          r'1\1'),
    (re.compile(r'(\d)I(\d)'),         r'\g<1>1\2'),
    (re.compile(r'(\d)I\b'),           r'\g<1>1'),
    (re.compile(r'\bIS([A-Z])'),       r'18\1'),
    (re.compile(r'(\d)S([KMGkm%\b])'), r'\g<1>8\2'),
    (re.compile(r'\bt0\b'),            'to'),
    (re.compile(r'(?<=\d)O\b'),        '0'),
    (re.compile(r'(?<=\d)O(?=\d)'),    '0'),
]

def _clean_table_cell(cell: str) -> str:
    """Applies OCR typo fixes and escapes LaTeX characters for table cells."""
    if not cell: return ""
    for pat, rep in _I_FIX:
        cell = pat.sub(rep, cell)
    return latex_escape(cell)


def _apply_booktabs_format(grid: list) -> str:
    """Converts a 2D list grid into a clean IEEE-style booktabs LaTeX table."""
    if not grid: return ""
    col_count = max(len(row) for row in grid)
    col_spec  = "l" * col_count  
    
    lines = [f"\\begin{{tabular}}{{{col_spec}}}", "\\toprule"] 
    for i, row in enumerate(grid):
        padded = row + [""] * (col_count - len(row))
        lines.append(" & ".join(padded) + " \\\\")
        if i == 0:
            lines.append("\\midrule")
            
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# NEW: Coordinate-Based Heuristic Solver
# ─────────────────────────────────────────────
class HeuristicTableSolver:
    """
    Replaces ML-based layout parsers (TATR/SLANet). Forces word-level OCR, 
    then uses X-Y axis projection to mathematically guarantee column alignment.
    """
    def __init__(self):
        import easyocr
        self.reader = easyocr.Reader(['en'], gpu=CONFIG.get("gpu", False), verbose=False)
        
    def parse(self, img: np.ndarray) -> dict:
        # width_ths=0.1 explicitly forces EasyOCR to STOP merging words across horizontal gaps
        raw_results = self.reader.readtext(img, detail=1, paragraph=False, width_ths=0.1)
        
        tokens = []
        img_w = img.shape[1]
        
        for bbox, text, conf in raw_results:
            text = text.strip()
            if not text: continue
            # Ignore horizontal table rules misread as text
            if re.match(r'^[\-\_\=\.]+$', text): continue
            
            xs = [p[0] for p in bbox]; ys = [p[1] for p in bbox]
            x1, x2 = min(xs), max(xs); y1, y2 = min(ys), max(ys)
            
            # Ignore false-positive bounding boxes that span the entire image width
            if (x2 - x1) > 0.8 * img_w: continue
            
            tokens.append({
                'text': text, 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2,
                'cx': (x1 + x2) / 2, 'cy': (y1 + y2) / 2,
                'h': y2 - y1, 'w': x2 - x1
            })
            
        if not tokens: return {"latex": "", "text": ""}
            
        # 1. GROUP ROWS (Y-Axis)
        tokens.sort(key=lambda t: t['cy'])
        rows = []
        current_row = []
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
            
        # 2. GROUP COLUMNS (X-Axis Projection)
        proj = np.zeros(img_w, dtype=int)
        for tok in tokens:
            # Skip very wide headers from projection so they don't destroy column gaps
            if tok['w'] > 0.4 * img_w: continue 
            px1 = max(0, int(tok['x1']))
            px2 = min(img_w, int(tok['x2']))
            proj[px1:px2] += 1
            
        in_col = False
        col_segments = []
        start = 0
        for i in range(img_w):
            if proj[i] > 0 and not in_col:
                in_col = True; start = i
            elif proj[i] == 0 and in_col:
                in_col = False; col_segments.append((start, i))
        if in_col: col_segments.append((start, img_w))
            
        if not col_segments: col_segments = [(0, img_w)] # Fallback
            
        # 3. BUILD GRID
        grid = []
        for row_tokens in rows:
            row_cells = [[] for _ in range(len(col_segments))]
            for tok in row_tokens:
                best_col, max_overlap, min_dist = 0, -1, float('inf')
                for i, (cs, ce) in enumerate(col_segments):
                    overlap = max(0, min(tok['x2'], ce) - max(tok['x1'], cs))
                    if overlap > max_overlap:
                        max_overlap = overlap; best_col = i
                    dist = 0 if cs <= tok['cx'] <= ce else min(abs(tok['cx'] - cs), abs(tok['cx'] - ce))
                    if overlap == 0 and dist < min_dist and max_overlap <= 0:
                        min_dist = dist; best_col = i
                row_cells[best_col].append(tok)
                
            final_row = []
            for cell_tokens in row_cells:
                cell_tokens.sort(key=lambda t: t['x1']) # Read left-to-right inside the cell
                text = " ".join(t['text'] for t in cell_tokens)
                final_row.append(_clean_table_cell(text))
            
            # Prevent pushing entirely empty rows caused by layout noise
            if any(cell.strip() for cell in final_row):
                grid.append(final_row)
            
        return {
            "latex": _apply_booktabs_format(grid),
            "table_grid": grid,
            "text": " | ".join(c for row in grid for c in row)
        }


# ─────────────────────────────────────────────
# OCR Backend Abstraction
# ─────────────────────────────────────────────
class _PaddleOCRBackend:
    def __init__(self):
        from paddleocr import PaddleOCR
        self._ocr = PaddleOCR(use_angle_cls=True, enable_mkldnn=False, lang="en")
        
    def readtext(self, img: np.ndarray) -> list:
        import cv2
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        result = self._ocr.ocr(bgr)
        out = []
        if not result or result[0] is None: return out
        for line in result[0]:
            if not line or len(line) < 2: continue
            try:
                bbox = line[0]
                if not isinstance(bbox, (list, tuple)) or len(bbox) == 0: continue
                text_data = line[1]
                if isinstance(text_data, (tuple, list)) and len(text_data) > 0:
                    text, conf = str(text_data[0]), float(text_data[1]) if len(text_data) > 1 else 1.0
                else:
                    text, conf = str(text_data), 1.0
                if text.strip(): out.append((bbox, text.strip(), conf))
            except Exception: continue
        return out


class _EasyOCRBackend:
    def __init__(self):
        import easyocr
        self._reader = easyocr.Reader(CONFIG["lang"], gpu=CONFIG["gpu"], verbose=False)

    def readtext(self, img: np.ndarray) -> list:
        return self._reader.readtext(img, detail=1, paragraph=False)


def _build_ocr_backend():
    backend = CONFIG.get("ocr_backend", "easyocr")
    if backend == "paddle":
        try:
            b = _PaddleOCRBackend()
            print("[Stage 2] OCR backend: PaddleOCR v4")
            return b
        except Exception as e:
            print(f"[WARN] PaddleOCR unavailable ({e}), falling back to EasyOCR")
    b = _EasyOCRBackend()
    print("[Stage 2] OCR backend: EasyOCR")
    return b


# ─────────────────────────────────────────────
# MAIN SOLVER CLASS
# ─────────────────────────────────────────────
class TextAndTableSolver:
    def __init__(self):
        print("[Stage 2] Loading OCR backend ...")
        self.ocr = _build_ocr_backend()
        self.heuristic_solver = HeuristicTableSolver()

    @staticmethod
    def _correct_orientation(img: np.ndarray) -> np.ndarray:
        import cv2
        try:
            import subprocess, shutil
            if shutil.which("tesseract"):
                import tempfile, os
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf: tmp = tf.name
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(tmp, bgr)
                result = subprocess.run(["tesseract", tmp, "stdout", "--psm", "0", "-l", "eng"], capture_output=True, text=True, timeout=5)
                os.unlink(tmp)
                for line in result.stdout.splitlines():
                    if "Rotate:" in line:
                        if int(line.split(":")[1].strip()) == 180:
                            print("  [ocr] Correcting 180° rotation (Tesseract OSD)")
                            return cv2.rotate(img, cv2.ROTATE_180)
                        break
                return img
        except Exception: pass

        try:
            import easyocr
            _reader = easyocr.Reader(["en"], gpu=CONFIG["gpu"], verbose=False)
            def _avg_conf(image):
                res = _reader.readtext(image, detail=1, paragraph=False)
                return (sum(r[2] for r in res) / len(res)) if res else 0.0

            img_180 = cv2.rotate(img, cv2.ROTATE_180)
            conf_orig, conf_flip = _avg_conf(img), _avg_conf(img_180)
            if conf_flip > conf_orig + 0.15:   
                print(f"  [ocr] Correcting 180° rotation (confidence {conf_orig:.2f} → {conf_flip:.2f})")
                return img_180
        except Exception: pass
        return img

    def solve(self, region: dict) -> dict:
        if region.get("type") == "Table":
            return self._solve_table(region)
        img = preprocess_image(region["image"])
        img = self._correct_orientation(img)
        lines = self._run_ocr_with_coords(img)
        text = " ".join(l["text"] for l in lines)
        region["text"] = text
        region["latex"] = latex_escape(text)
        return region

    def _solve_table(self, region: dict) -> dict:
        img = region["image"]
        # Route directly to the Coordinate-Based Heuristic Solver
        parsed = self.heuristic_solver.parse(img)
        region.update(parsed)
        return region

    def _run_ocr_with_coords(self, img: np.ndarray) -> list:
        results = self.ocr.readtext(img)
        return [{"text": r[1], "x": r[0][0][0], "y": r[0][0][1]} for r in results]

    def _passthrough(self, region: dict) -> dict:
        region.setdefault("latex", "")
        return region