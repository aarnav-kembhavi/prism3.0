import os
import sys
import json

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
# OCR backend abstraction
# ─────────────────────────────────────────────

class _PaddleOCRBackend:
    """PaddleOCR v4 text recognition."""

    def __init__(self):
        from paddleocr import PaddleOCR
        self._ocr = PaddleOCR(
            use_angle_cls=True, 
            enable_mkldnn=False,
            lang="en"
        )
        
    def readtext(self, img: np.ndarray) -> list:
        import cv2
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        result = self._ocr.ocr(bgr)
        out = []
        if not result or result[0] is None:
            return out
            
        for line in result[0]:
            if not line or len(line) < 2: 
                continue
            
            try:
                bbox = line[0]
                if not isinstance(bbox, (list, tuple)) or len(bbox) == 0:
                    continue
                
                text_data = line[1]
                if isinstance(text_data, (tuple, list)) and len(text_data) > 0:
                    text = str(text_data[0])
                    conf = float(text_data[1]) if len(text_data) > 1 else 1.0
                else:
                    text = str(text_data)
                    conf = 1.0
                    
                if text.strip():  
                    out.append((bbox, text.strip(), conf))
            except Exception:
                continue
                
        return out


class _EasyOCRBackend:
    """EasyOCR fallback backend."""

    def __init__(self):
        import easyocr
        self._reader = easyocr.Reader(
            CONFIG["lang"], gpu=CONFIG["gpu"], verbose=False
        )

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
# IMPROVEMENT 2 — TATR Table Solver
# ─────────────────────────────────────────────

class TATRTableSolver:
    def __init__(self, ocr_backend):
        device = "cuda" if CONFIG["gpu"] and torch.cuda.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained(TATR_MODEL)
        self.model     = TableTransformerForObjectDetection.from_pretrained(
            TATR_MODEL
        ).to(device)
        self.model.eval()
        self.ocr    = ocr_backend
        self.device = device

    def parse(self, img: np.ndarray) -> dict:
        pil_img = Image.fromarray(img)
        inputs  = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_object_detection(
            outputs,
            threshold=0.15,  # Max recall. Let NMS handle duplicates.
            target_sizes=torch.tensor([pil_img.size[::-1]]),
        )[0]

        rows, cols = self._extract_rows_cols(results)
        if not rows or not cols:
            text = self._ocr_full(img)
            return {"latex": f"% Table (no grid detected)\n{latex_escape(text)}", "text": text}

        tokens = self._ocr_tokens(img)
        grid = self._assign_tokens_to_grid(tokens, rows, cols, img.shape, img=img)
        
        # Heuristic cleanup: Merges stray columns, deletes empty ones, attaches currency symbols
        grid = clean_table_grid(grid)
        
        grid = [[latex_escape(cell) for cell in row] for row in grid]
        return {
            "latex":      self._grid_to_tabular(grid),
            "table_grid": grid,
            "text":       " | ".join(c for row in grid for c in row),
        }

    def _extract_rows_cols(self, results) -> tuple:
        id2label = self.model.config.id2label
        row_boxes = []
        col_boxes = []
        
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            name = id2label[label.item()].lower()
            # Strictly rely on 'table row' and 'table column'. Spanning headers confuse grids.
            if name == "table row": 
                row_boxes.append((box.tolist(), score.item()))
            elif name == "table column": 
                col_boxes.append((box.tolist(), score.item()))
                
        # 1D Non-Maximum Suppression (NMS) using proper IoU
        def nms_1d(items, i1, i2, thresh=0.25):
            if not items: return []
            # Sort by confidence score so we keep the best boxes
            items.sort(key=lambda x: x[1], reverse=True)
            keep = []
            for box, score in items:
                discard = False
                for k_box in keep:
                    overlap = max(0, min(box[i2], k_box[i2]) - max(box[i1], k_box[i1]))
                    union = (box[i2] - box[i1]) + (k_box[i2] - k_box[i1]) - overlap
                    iou = overlap / union if union > 0 else 0
                    
                    if iou > thresh:
                        discard = True
                        break
                if not discard:
                    keep.append(box)
                    
            # Sort back by spatial coordinate for top-to-bottom/left-to-right grid order
            keep.sort(key=lambda b: b[i1])
            return keep

        # Apply strict IoU NMS to prevent row snowballing
        rows = nms_1d(row_boxes, 1, 3, thresh=0.3)
        cols = nms_1d(col_boxes, 0, 2, thresh=0.3)
        return rows, cols

    def _ocr_tokens(self, img: np.ndarray) -> list:
        results = self.ocr.readtext(img)
        
        if not results and isinstance(self.ocr, _PaddleOCRBackend):
            print("  [table] PaddleOCR found no text, falling back to EasyOCR...")
            fallback_ocr = _EasyOCRBackend()
            results = fallback_ocr.readtext(img)
            
        tokens = []
        for bbox, text, conf in results:
            pts = bbox
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            tokens.append({
                "text": text.strip(),
                "cx": (min(xs) + max(xs)) / 2,
                "cy": (min(ys) + max(ys)) / 2,
                "conf": conf,
            })
        return tokens

    def _ocr_full(self, img: np.ndarray) -> str:
        tokens = self._ocr_tokens(img)
        tokens.sort(key=lambda t: t["cy"])
        return " ".join(t["text"] for t in tokens)

    def _assign_tokens_to_grid(self, tokens: list, rows: list, cols: list, shape: tuple, img: np.ndarray = None) -> list:
        grid = [[[] for _ in cols] for _ in rows]
        
        for tok in tokens:
            cx, cy = tok["cx"], tok["cy"]
            
            best_row = None
            min_row_dist = float('inf')
            for ri, rb in enumerate(rows):
                dist = 0 if (rb[1] <= cy <= rb[3]) else min(abs(cy - rb[1]), abs(cy - rb[3]))
                if dist < min_row_dist:
                    min_row_dist = dist
                    best_row = ri
                    
            best_col = None
            min_col_dist = float('inf')
            for ci, cb in enumerate(cols):
                dist = 0 if (cb[0] <= cx <= cb[2]) else min(abs(cx - cb[0]), abs(cx - cb[2]))
                if dist < min_col_dist:
                    min_col_dist = dist
                    best_col = ci

            if best_row is not None and best_col is not None:
                grid[best_row][best_col].append(tok)
                
        result = []
        for ri, row in enumerate(grid):
            cells = []
            for ci, cell_tokens in enumerate(row):
                cell_tokens.sort(key=lambda t: t["cx"])
                cells.append(" ".join(t["text"] for t in cell_tokens))
            result.append(cells)
        return result

    def _grid_to_tabular(self, grid: list) -> str:
        if not grid: return ""
        col_count = max(len(row) for row in grid)
        col_spec  = "|" + "l|" * col_count
        lines     = [f"\\begin{{tabular}}{{{col_spec}}}", "\\hline"]
        for i, row in enumerate(grid):
            padded = row + [""] * (col_count - len(row))
            lines.append(" & ".join(padded) + " \\\\")
            lines.append("\\hline")
        lines.append("\\end{tabular}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# SLANet (PaddleOCR PP-Structure) Table Solver
# ─────────────────────────────────────────────

class SLANetTableSolver:
    """
    PaddleOCR PP-Structure table recognition using SLANet.
    """

    def __init__(self):
        try:
            from paddleocr import PPStructure
            self.engine = PPStructure(show_log=False, lang="en")
        except ImportError:
            try:
                from paddleocr.paddleocr import PPStructure
                self.engine = PPStructure(show_log=False, lang="en")
            except ImportError:
                try:
                    from paddleocr.ppstructure.predict_system import PPStructure
                    self.engine = PPStructure(show_log=False, lang="en")
                except ImportError as e:
                    raise ImportError(f"PaddleOCR PPStructure failed to load from any known path: {e}")

    def parse(self, img: np.ndarray) -> dict:
        import cv2
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        result = self.engine(bgr)
        if not result or not result[0].get("res"): return {}
        
        html = result[0]["res"]["html"]
        grid = html_table_to_grid(html)
        return {
            "latex":      grid_to_tabular(grid),
            "table_grid": grid,
            "text":       " | ".join(c for row in grid for c in row),
        }


# ─────────────────────────────────────────────
# MAIN SOLVER CLASS
# ─────────────────────────────────────────────

class TextAndTableSolver:
    def __init__(self):
        print("[Stage 2] Loading OCR backend ...")
        self.ocr = _build_ocr_backend()
        self.slanet = None
        if CONFIG.get("use_slanet"):
            try:
                print("[Stage 2] Loading SLANet (PP-Structure) ...")
                self.slanet = SLANetTableSolver()
            except Exception as e:
                print(f"[WARN] SLANet failed: {e}")
        self.tatr = TATRTableSolver(self.ocr)

    def solve(self, region: dict) -> dict:
        if region.get("type") == "Table":
            return self._solve_table(region)
        img = preprocess_image(region["image"])
        lines = self._run_ocr_with_coords(img)
        text = " ".join(l["text"] for l in lines)
        region["text"] = text
        region["latex"] = latex_escape(text)
        return region

    def _solve_table(self, region: dict) -> dict:
        # Pass the raw image directly. Aggressive contrast destroys thin text/single characters.
        img = region["image"]
        
        if self.slanet:
            parsed = self.slanet.parse(img)
            if parsed.get("latex"):
                region.update(parsed)
                return region
        parsed = self.tatr.parse(img)
        region.update(parsed)
        return region

    def _run_ocr_with_coords(self, img: np.ndarray) -> list:
        results = self.ocr.readtext(img)
        return [{"text": r[1], "x": r[0][0][0], "y": r[0][0][1]} for r in results]

    def _passthrough(self, region: dict) -> dict:
        region.setdefault("latex", "")
        return region