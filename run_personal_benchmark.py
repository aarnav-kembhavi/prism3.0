"""
run_personal_benchmark.py
--------------------------
Run the updated PRISM pipeline on the 4 personal test images and report
latency + peak RAM per stage. No ground truth exists so EDR is not computed;
output character count is shown as a proxy for coverage.
"""

import gc, os, shutil, time
from pathlib import Path

import numpy as np
import psutil

IMAGES = [
    Path("image.png"),
    Path("image2.png"),
    Path("image3.jpeg"),
    Path("image4.png"),
]

YOLO_MODEL_PATH = "yolov11n-doclaynet.onnx"

from normalization import normalize_image_pil
from normalization.modality import CaptureModality
from models_interface import (
    run_text_ocr_batched, run_math_recognition_batched,
    run_table_extraction_batched, get_yolo_model, unload_texo,
)
from layout_utils import (
    apply_semantic_reading_order, xyxy_to_pil_crop,
    detect_column_count, split_detections_by_column,
)
from latex_builder import wrap_content, assemble_document, save_tex
from detection_postprocess import postprocess_detections

TEXT_CLASSES    = {"Text", "Title", "Section-header", "Caption",
                   "Footnote", "Page-footer", "Page-header", "List-item"}
MATH_CLASSES    = {"Formula"}
TABLE_CLASSES   = {"Table"}
IMAGE_CLASSES   = {"Picture"}
LIST_ITEM_CLASS = "List-item"

process = psutil.Process(os.getpid())


def peak_mb():
    return process.memory_info().rss / 1024 / 1024


def _is_likely_logo(crop):
    arr = np.array(crop.convert("RGB"), dtype=np.float32)
    gray = arr.mean(axis=2)
    return float(np.mean(gray < 230)) < 0.15 and float(arr.std()) > 8.0


def run_image(img_path: Path, yolo_model) -> dict:
    name = img_path.stem
    out_dir = Path(f"{name}_output")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    assets_dir  = out_dir / "assets"
    figures_dir = assets_dir / "figures"
    for d in [out_dir, assets_dir, figures_dir]:
        d.mkdir(parents=True, exist_ok=True)

    t_total = time.perf_counter()

    # Stage 1: Normalization
    t0 = time.perf_counter()
    image_norm, image_fidelity, modality = normalize_image_pil(str(img_path))
    is_screenshot = (modality.modality == CaptureModality.SCREENSHOT)
    norm_path = str(assets_dir / "normalized.png")
    image_norm.save(norm_path)
    t_norm = time.perf_counter() - t0
    mb_norm = peak_mb()

    # Stage 2: YOLO layout detection
    t0 = time.perf_counter()
    results = yolo_model(norm_path, verbose=False)
    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cn = results[0].names[int(box.cls[0].item())]
        crop = xyxy_to_pil_crop(image_fidelity if cn in IMAGE_CLASSES else image_norm,
                                [x1, y1, x2, y2])
        if cn == "Page-header" and _is_likely_logo(
                xyxy_to_pil_crop(image_fidelity, [x1, y1, x2, y2])):
            cn = "Picture"
        detections.append({"bbox": [x1, y1, x2, y2], "class_name": cn,
                            "confidence": float(box.conf[0].item()), "crop": crop})
    img_w, img_h = image_norm.width, image_norm.height
    detections = postprocess_detections(detections, img_w, img_h)
    t_yolo = time.perf_counter() - t0
    mb_yolo = peak_mb()

    # Re-crop with formula padding
    FORMULA_PAD = 12
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        if det["class_name"] in MATH_CLASSES:
            x1 = max(0, x1 - FORMULA_PAD); y1 = max(0, y1 - FORMULA_PAD)
            x2 = min(img_w, x2 + FORMULA_PAD); y2 = min(img_h, y2 + FORMULA_PAD)
        det["crop"] = xyxy_to_pil_crop(
            image_fidelity if det["class_name"] in IMAGE_CLASSES else image_norm,
            [x1, y1, x2, y2])

    del image_norm, image_fidelity
    gc.collect()

    # Stage 3: Content extraction
    text_indices  = [i for i, d in enumerate(detections) if d["class_name"] in TEXT_CLASSES]
    math_indices  = [i for i, d in enumerate(detections) if d["class_name"] in MATH_CLASSES]
    table_indices = [i for i, d in enumerate(detections) if d["class_name"] in TABLE_CLASSES]

    t0 = time.perf_counter()
    if math_indices:
        math_results = run_math_recognition_batched(
            [detections[i]["crop"] for i in math_indices], str(figures_dir), [0])
        for idx, raw in zip(math_indices, math_results):
            detections[idx]["raw_content"] = raw
    t_math = time.perf_counter() - t0
    mb_math = peak_mb()

    t0 = time.perf_counter()
    if text_indices:
        texts = run_text_ocr_batched(
            [detections[i]["crop"] for i in text_indices], is_screenshot=is_screenshot)
        for idx, txt in zip(text_indices, texts):
            detections[idx]["raw_content"] = txt
    t_text = time.perf_counter() - t0
    mb_text = peak_mb()

    t0 = time.perf_counter()
    if table_indices:
        table_results = run_table_extraction_batched(
            [detections[i]["crop"] for i in table_indices])
        for idx, raw in zip(table_indices, table_results):
            detections[idx]["raw_content"] = raw
    t_table = time.perf_counter() - t0
    mb_table = peak_mb()

    # Assembly
    body_parts, list_idx = [], set()
    figure_counter = 0
    body_sorted = apply_semantic_reading_order(
        [d for d in detections if not d.get("is_header_logo")], img_w, img_h)
    for det in body_sorted:
        cn = det["class_name"]
        if cn in TEXT_CLASSES or cn in MATH_CLASSES:
            raw = det.get("raw_content", "")
            wrapped = wrap_content(cn, raw)
            if cn == LIST_ITEM_CLASS:
                list_idx.add(len(body_parts))
            body_parts.append(wrapped)
        elif cn in TABLE_CLASSES:
            raw = det.get("raw_content", "")
            if raw:
                body_parts.append(wrap_content(cn, raw))
        elif cn in IMAGE_CLASSES:
            figure_counter += 1
            fname = f"figure_{figure_counter:03d}.png"
            det["crop"].save(str(figures_dir / fname))
            body_parts.append(wrap_content("Picture", fname))

    doc = assemble_document(body_parts, list_idx, False)
    tex_path = out_dir / "main.tex"
    save_tex(doc, str(tex_path))

    total_sec = time.perf_counter() - t_total
    mb_peak = peak_mb()

    n_text  = len(text_indices)
    n_math  = len(math_indices)
    n_table = len(table_indices)
    char_count = len(tex_path.read_text(encoding="utf-8", errors="ignore"))

    return {
        "name": name,
        "modality": "screenshot" if is_screenshot else "phone",
        "n_text": n_text, "n_math": n_math, "n_table": n_table,
        "t_norm": t_norm, "mb_norm": mb_norm,
        "t_yolo": t_yolo, "mb_yolo": mb_yolo,
        "t_math": t_math, "mb_math": mb_math,
        "t_text": t_text, "mb_text": mb_text,
        "t_table": t_table, "mb_table": mb_table,
        "total_sec": total_sec, "mb_peak": mb_peak,
        "char_count": char_count,
        "tex_path": tex_path,
    }


# ── Run ───────────────────────────────────────────────────────────────────────
yolo_model = get_yolo_model(YOLO_MODEL_PATH)
rows = []

for img_path in IMAGES:
    if not img_path.exists():
        print(f"[skip] {img_path} not found")
        continue
    print(f"\n{'='*60}")
    print(f"[*] {img_path.name}")
    print('='*60)
    r = run_image(img_path, yolo_model)
    rows.append(r)
    print(f"  Detections: {r['n_text']} text / {r['n_math']} math / {r['n_table']} table")
    print(f"  Modality:   {r['modality']}")
    print(f"  Output:     {r['char_count']} chars  ->  {r['tex_path']}")
    gc.collect()

unload_texo()

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "="*90)
print("  PRISM Personal Images  --  Post-optimisation metrics")
print("="*90)
print(f"  {'Image':<10}  {'Mode':<10}  {'Regions':>9}  {'Norm':>6}  {'YOLO':>6}  {'Math':>6}  {'Text':>6}  {'Table':>6}  {'Total':>7}  {'Peak RAM':>9}  {'Output':>8}")
print(f"  {'':10}  {'':10}  {'t/m/tbl':>9}  {'s':>6}  {'s':>6}  {'s':>6}  {'s':>6}  {'s':>6}  {'s':>7}  {'MB':>9}  {'chars':>8}")
print("  " + "-"*86)

for r in rows:
    regions = f"{r['n_text']}/{r['n_math']}/{r['n_table']}"
    print(f"  {r['name']:<10}  {r['modality']:<10}  {regions:>9}"
          f"  {r['t_norm']:>5.1f}s  {r['t_yolo']:>5.1f}s"
          f"  {r['t_math']:>5.1f}s  {r['t_text']:>5.1f}s  {r['t_table']:>5.1f}s"
          f"  {r['total_sec']:>6.1f}s  {r['mb_peak']:>8.0f}MB"
          f"  {r['char_count']:>8}")

print("  " + "-"*86)
if rows:
    print(f"  {'AVERAGE':<10}  {'':10}  {'':9}"
          f"  {sum(r['t_norm'] for r in rows)/len(rows):>5.1f}s"
          f"  {sum(r['t_yolo'] for r in rows)/len(rows):>5.1f}s"
          f"  {sum(r['t_math'] for r in rows)/len(rows):>5.1f}s"
          f"  {sum(r['t_text'] for r in rows)/len(rows):>5.1f}s"
          f"  {sum(r['t_table'] for r in rows)/len(rows):>5.1f}s"
          f"  {sum(r['total_sec'] for r in rows)/len(rows):>6.1f}s"
          f"  {sum(r['mb_peak'] for r in rows)/len(rows):>8.0f}MB"
          f"  {sum(r['char_count'] for r in rows)//len(rows):>8}")

print("="*90)
print("\n  Note: EDR requires ground truth -- not available for personal images.")
print("  Output char count is a proxy for content coverage.")
print("  Peak RAM reflects post-optimisation pipeline (Texo on CPU, no gray/binary copies,")
print("  batched table OCR).")
