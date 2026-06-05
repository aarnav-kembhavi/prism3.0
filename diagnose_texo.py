"""
diagnose_texo.py
----------------
For specified benchmark pages, runs YOLO detection, saves every formula crop
that would be sent to Texo, runs Texo on each, and shows a side-by-side of
crop dimensions / Texo output / GT math.

Usage:
    python diagnose_texo.py 1 6 10 26
"""

import sys
import os
import json
from pathlib import Path
from PIL import Image
import numpy as np
import torch

# Bootstrap paths
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Texo" / "src"))

from normalization import normalize_image_pil
from layout_utils import xyxy_to_pil_crop
from detection_postprocess import postprocess_detections
from evaluation.normalizer import normalize_latex, split_math_and_text
from ultralytics import YOLO

IMAGES_DIR   = ROOT / "benchmark_results" / "temp_images"
GT_DIR       = ROOT / "pdf2latex_dataset" / "dataset"
DIAG_OUT     = ROOT / "benchmark_results" / "texo_diagnosis"
YOLO_PATH    = str(ROOT / "yolov11n-doclaynet.onnx")

FORMULA_PAD = 12

def load_texo():
    from texo.data.processor import EvalMERImageProcessor
    from texo.model.formulanet import FormulaNet
    import texo.utils.config
    from transformers import AutoTokenizer
    MODEL_PATH = str(ROOT / "Texo" / "model")
    tok   = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = FormulaNet.from_pretrained(MODEL_PATH)
    model.eval()
    proc  = EvalMERImageProcessor(image_size={'width': 384, 'height': 384})
    return model, tok, proc


def binarize(crop: Image.Image) -> Image.Image:
    import cv2
    arr = np.array(crop.convert("L"), dtype=np.uint8)
    _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binary).convert("RGB")


MAX_FORMULA_ASPECT = 3.5

def split_crop(crop: Image.Image) -> list:
    import math
    w, h = crop.size
    if h == 0: return [crop]
    aspect = w / h
    if aspect <= MAX_FORMULA_ASPECT: return [crop]
    n = math.ceil(aspect / MAX_FORMULA_ASPECT)
    seg_w = w // n
    return [crop.crop((i*seg_w, 0, (i+1)*seg_w if i<n-1 else w, h)) for i in range(n)]


def run_texo_on_crop(crop, model, tok, proc):
    segs = split_crop(crop)
    parts = []
    for seg in segs:
        prepared = binarize(seg)
        pixel_values = proc(prepared).unsqueeze(0)
        with torch.no_grad():
            out = model.generate(pixel_values, max_new_tokens=300, repetition_penalty=1.15)
        raw = tok.decode(out[0], skip_special_tokens=True).strip()
        for delim in ("$$","$",r"\[",r"\]",r"\(",r"\)"):
            if raw.startswith(delim): raw = raw[len(delim):]
            if raw.endswith(delim):   raw = raw[:-len(delim)]
        raw = raw.strip()
        if raw:
            parts.append(raw)
    return ' '.join(parts)


def diagnose_page(page_id: str, model, tok, proc):
    img_path = IMAGES_DIR / f"{page_id}.png"
    gt_path  = GT_DIR / f"{page_id}_gt.tex"
    if not img_path.exists():
        print(f"  [!] Image not found: {img_path}"); return
    if not gt_path.exists():
        print(f"  [!] GT not found: {gt_path}"); return

    out_dir = DIAG_OUT / page_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Normalise image
    image_norm, image_fidelity, modality = normalize_image_pil(str(img_path))
    img_w, img_h = image_norm.width, image_norm.height

    # YOLO
    yolo = YOLO(YOLO_PATH, task='detect')
    norm_png = str(out_dir / "normalized.png")
    image_norm.save(norm_png)
    results = yolo(norm_png, verbose=False)
    del yolo

    detections = []
    for box in results[0].boxes:
        x1,y1,x2,y2 = box.xyxy[0].tolist()
        cls = results[0].names[int(box.cls[0])]
        conf = float(box.conf[0])
        detections.append({"bbox":[x1,y1,x2,y2],"class_name":cls,"confidence":conf})
    detections = postprocess_detections(detections, img_w, img_h)

    formula_dets = [d for d in detections if d["class_name"] == "Formula"]
    all_classes  = [d["class_name"] for d in detections]

    print(f"\n  Page {page_id}: {img_w}x{img_h}px")
    print(f"  All YOLO classes: {sorted(set(all_classes))}")
    print(f"  Total detections: {len(detections)}  |  Formula regions: {len(formula_dets)}")

    if not formula_dets:
        print("  [!] NO formula regions detected — Texo never called on this page")
        report = {"page_id": page_id, "formula_regions": 0, "crops": []}
        with open(out_dir / "report.json", "w") as f: json.dump(report, f, indent=2)
        return

    # GT math for comparison
    gt_tex = gt_path.read_text(encoding='utf-8', errors='replace')
    gt_norm = normalize_latex(gt_tex, remove_spaces=True)
    gt_math, _ = split_math_and_text(gt_norm)

    crops_info = []
    for i, det in enumerate(formula_dets):
        x1,y1,x2,y2 = det["bbox"]
        # Apply formula bbox expansion (same as pipeline)
        x1e = max(0, x1 - FORMULA_PAD)
        y1e = max(0, y1 - FORMULA_PAD)
        x2e = min(img_w, x2 + FORMULA_PAD)
        y2e = min(img_h, y2 + FORMULA_PAD)

        crop_raw  = xyxy_to_pil_crop(image_norm, [x1e, y1e, x2e, y2e])
        crop_proc = binarize(crop_raw)

        # Save both raw crop and preprocessed crop
        crop_raw.save(out_dir  / f"crop_{i+1:02d}_raw.png")
        crop_proc.save(out_dir / f"crop_{i+1:02d}_proc.png")

        aspect = crop_raw.width / max(crop_raw.height, 1)
        texo_out = run_texo_on_crop(crop_raw, model, tok, proc)

        info = {
            "crop_idx":    i + 1,
            "bbox_orig":   [round(x1), round(y1), round(x2), round(y2)],
            "bbox_expand": [round(x1e), round(y1e), round(x2e), round(y2e)],
            "size_px":     f"{crop_raw.width}x{crop_raw.height}",
            "aspect_ratio": round(aspect, 2),
            "yolo_conf":   round(det["confidence"], 3),
            "texo_output": texo_out,
            "texo_chars":  len(texo_out),
        }
        crops_info.append(info)

        print(f"\n  Crop {i+1}: {info['size_px']}  aspect={aspect:.2f}  conf={det['confidence']:.2f}")
        print(f"    Texo ({len(texo_out)} chars): {texo_out[:200]}")

    report = {
        "page_id":        page_id,
        "image_size":     f"{img_w}x{img_h}",
        "formula_regions": len(formula_dets),
        "gt_math_chars":  len(gt_math),
        "crops":          crops_info,
    }
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  GT math total: {len(gt_math)} chars")
    pred_total = sum(c["texo_chars"] for c in crops_info)
    print(f"  Texo total:   {pred_total} chars across {len(formula_dets)} crops")
    print(f"  Crops saved -> {out_dir}")


def main():
    page_ids = sys.argv[1:] if len(sys.argv) > 1 else ["1","6","10","26"]
    print(f"[*] Loading Texo model...")
    model, tok, proc = load_texo()
    print(f"[*] Diagnosing pages: {page_ids}\n")
    for pid in page_ids:
        print(f"{'='*60}")
        diagnose_page(pid, model, tok, proc)
    print(f"\n[OK] Diagnosis complete. Crops saved in {DIAG_OUT}/")


if __name__ == "__main__":
    main()
