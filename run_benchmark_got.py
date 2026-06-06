"""
run_benchmark_got.py
--------------------
In-process benchmark runner with GOT-OCR integration.

Loads all models ONCE then processes all 26 benchmark pages sequentially.
Avoids per-page model-reload overhead that would make the subprocess-based
runner take ~30 min with GOT-OCR on board.

Outputs:
  benchmark_results/prism_tex/<N>_prism.tex   (rebuilt .tex files)
  benchmark_results/latency_log_got.csv        (per-page timing)
  benchmark_results/benchmark_summary_got.txt  (final metric report)
"""

import csv
import gc
import os
import shutil
import statistics
import sys
import time
from pathlib import Path

import psutil
import torch
from PIL import Image
from ultralytics import YOLO

from normalization import normalize_image_pil
from normalization.modality import CaptureModality
from models_interface import (
    run_text_ocr_batched,
    run_text_ocr_full_page,
    run_math_recognition_batched,
    run_table_extraction,
    get_math_batch_latencies,
    get_text_batch_latencies,
    get_table_latencies,
)
from layout_utils import (
    apply_semantic_reading_order,
    detect_column_count,
    split_detections_by_column,
    xyxy_to_pil_crop,
)
from latex_builder import wrap_content, assemble_document, save_tex
from detection_postprocess import postprocess_detections
from orchestrate import (
    _is_likely_logo,
    _adjust_figure_paths,
    TEXT_CLASSES, MATH_CLASSES, TABLE_CLASSES, IMAGE_CLASSES,
    LIST_ITEM_CLASS,
    route_and_extract,
)

YOLO_MODEL_PATH = "yolov11n-doclaynet.onnx"
IMAGES_DIR      = Path("benchmark_results/temp_images")
GT_DIR          = Path("pdf2latex_dataset/dataset")
PRISM_TEX_DIR   = Path("benchmark_results/prism_tex")
LATENCY_LOG     = Path("benchmark_results/latency_log_got.csv")

PRISM_TEX_DIR.mkdir(parents=True, exist_ok=True)

image_paths = sorted(
    [p for p in IMAGES_DIR.glob("*.png") if (GT_DIR / f"{p.stem}_gt.tex").exists()],
    key=lambda p: int(p.stem),
)
print(f"[*] {len(image_paths)} pages to process (GT matched)\n")

# ── Load YOLO once ─────────────────────────────────────────────────────────────
print("[*] Loading YOLO model (kept warm for all pages)...")
yolo_model = YOLO(YOLO_MODEL_PATH, task="detect")
print("[*] YOLO ready.\n")

process    = psutil.Process(os.getpid())
rows       = []


def _detect(image_path_str, image_norm, image_fidelity):
    """Run YOLO on saved normalized image, return raw detections."""
    results     = yolo_model(image_path_str, verbose=False)
    detections  = []
    result      = results[0]
    class_names = result.names
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        class_id   = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        class_name = class_names[class_id]
        if class_name == "Picture":
            crop = xyxy_to_pil_crop(image_fidelity, [x1, y1, x2, y2])
        else:
            crop = xyxy_to_pil_crop(image_norm, [x1, y1, x2, y2])
        if class_name == "Page-header":
            fid_crop = xyxy_to_pil_crop(image_fidelity, [x1, y1, x2, y2])
            if _is_likely_logo(fid_crop):
                class_name = "Picture"
                crop       = fid_crop
        detections.append({
            "bbox": [x1, y1, x2, y2],
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
            "crop": crop,
        })
    return detections


# ── Main loop ─────────────────────────────────────────────────────────────────
for idx, img_path in enumerate(image_paths, 1):
    page_id = img_path.stem
    out_dir = Path(f"{page_id}_output")
    out_dir.mkdir(parents=True, exist_ok=True)
    assets_dir  = out_dir / "assets"
    figures_dir = assets_dir / "figures"
    logs_dir    = out_dir / "logs"
    for d in [out_dir, assets_dir, figures_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"[{idx:>2}/{len(image_paths)}] {img_path.name}", flush=True)
    t_page_start = time.perf_counter()

    # ── Stage 1: Normalization ────────────────────────────────────
    t0 = time.perf_counter()
    image_norm, image_fidelity, modality_result = normalize_image_pil(str(img_path))
    is_screenshot = (modality_result.modality == CaptureModality.SCREENSHOT)
    norm_png = str(assets_dir / "normalized.png")
    image_norm.save(norm_png)
    t_norm = time.perf_counter() - t0
    mem_after_norm = process.memory_info().rss / 1024 / 1024

    # ── Stage 2: Detection ────────────────────────────────────────
    t0 = time.perf_counter()
    img_w, img_h = image_norm.width, image_norm.height
    detections   = _detect(norm_png, image_norm, image_fidelity)
    detections   = postprocess_detections(detections, img_w, img_h)
    t_yolo = time.perf_counter() - t0
    mem_after_yolo = process.memory_info().rss / 1024 / 1024

    # ── Stage 1.5: Header / re-crop ───────────────────────────────
    HEADER_SUPPRESS_H_FRAC = 0.12
    header_suppress_y = img_h * HEADER_SUPPRESS_H_FRAC
    detections = [
        d for d in detections
        if not (
            d["class_name"] in {"Section-header", "Page-header"}
            and d["bbox"][3] <= header_suppress_y
        )
    ]
    HEADER_H_FRAC, HEADER_W_FRAC = 0.065, 0.25
    header_right_box = [img_w * (1 - HEADER_W_FRAC), 0, img_w, img_h * HEADER_H_FRAC]
    if not any(
        d["class_name"] == "Picture"
        and d["bbox"][0] >= header_right_box[0]
        and d["bbox"][3] <= header_right_box[3]
        for d in detections
    ):
        hx1, hy1, hx2, hy2 = [int(v) for v in header_right_box]
        header_crop = xyxy_to_pil_crop(image_fidelity, [hx1, hy1, hx2, hy2])
        if header_crop.width > 20:
            detections.insert(0, {
                "bbox": [hx1, hy1, hx2, hy2], "class_id": -1,
                "class_name": "Picture", "crop": header_crop,
                "is_header_logo": True,
            })

    FORMULA_PAD = 12
    for det in detections:
        bbox = det["bbox"]
        if det["class_name"] in MATH_CLASSES:
            x1, y1, x2, y2 = bbox
            bbox = [
                max(0, x1 - FORMULA_PAD), max(0, y1 - FORMULA_PAD),
                min(img_w, x2 + FORMULA_PAD), min(img_h, y2 + FORMULA_PAD),
            ]
        if det["class_name"] in IMAGE_CLASSES:
            det["crop"] = xyxy_to_pil_crop(image_fidelity, bbox)
        else:
            det["crop"] = xyxy_to_pil_crop(image_norm, bbox)

    # Full-page OCR before freeing image_norm (one call instead of N per-crop calls)
    text_dets = [d for d in detections if d["class_name"] in TEXT_CLASSES]
    if text_dets:
        page_texts = run_text_ocr_full_page(image_norm, text_dets, is_screenshot=is_screenshot)
        for det, txt in zip(text_dets, page_texts):
            det["_ocr_text"] = txt

    del image_norm, image_fidelity
    gc.collect()

    # ── Stage 3: Extraction ───────────────────────────────────────
    t0 = time.perf_counter()
    n_math_before  = len(get_math_batch_latencies())
    n_text_before  = len(get_text_batch_latencies())
    n_table_before = len(get_table_latencies())

    col_count        = detect_column_count(detections, img_w)
    header_logo_dets = [d for d in detections if d.get("is_header_logo")]
    body_detections  = [d for d in detections if not d.get("is_header_logo")]
    header_logo_fname = "assets/figure_header_logo.png" if header_logo_dets else None
    if header_logo_fname:
        header_logo_dets[0]["crop"].save(out_dir / header_logo_fname)

    if col_count == 2:
        full_dets, left_dets, right_dets = split_detections_by_column(
            body_detections, img_w, img_h, use_dag=True
        )
        full_parts, full_idx, f_cnt, m_cnt = route_and_extract(
            full_dets,  str(figures_dir), 0,     is_screenshot=is_screenshot, math_start=0
        )
        left_parts, left_idx, f_cnt, m_cnt = route_and_extract(
            left_dets,  str(figures_dir), f_cnt, is_screenshot=is_screenshot, math_start=m_cnt
        )
        right_parts, right_idx, f_cnt, m_cnt = route_and_extract(
            right_dets, str(figures_dir), f_cnt, is_screenshot=is_screenshot, math_start=m_cnt
        )
        full_parts  = _adjust_figure_paths(full_parts)
        left_parts  = _adjust_figure_paths(left_parts)
        right_parts = _adjust_figure_paths(right_parts)
        document = assemble_document(
            full_parts, full_idx, True,
            left_parts, left_idx, right_parts, right_idx,
            header_logo_fname,
        )
    else:
        body_sorted = apply_semantic_reading_order(body_detections, img_w, img_h)
        body_parts, list_idx, _, _ = route_and_extract(
            body_sorted, str(figures_dir), is_screenshot=is_screenshot
        )
        body_parts = _adjust_figure_paths(body_parts)
        document   = assemble_document(body_parts, list_idx, False, header_logo=header_logo_fname)

    t_extract = time.perf_counter() - t0

    # Component sub-timings from singletons
    math_lats  = get_math_batch_latencies() [n_math_before:]
    text_lats  = get_text_batch_latencies() [n_text_before:]
    table_lats = get_table_latencies()      [n_table_before:]

    t_math  = sum(math_lats)  / 1000.0
    t_text  = sum(text_lats)  / 1000.0
    t_table = sum(table_lats) / 1000.0

    # ── Stage 4: Save ─────────────────────────────────────────────
    tex_out = out_dir / "main.tex"
    save_tex(document, str(tex_out))

    # ── Totals ────────────────────────────────────────────────────
    t_total   = time.perf_counter() - t_page_start
    peak_mb   = process.memory_info().rss / 1024 / 1024

    # Copy to prism_tex
    dest = PRISM_TEX_DIR / f"{page_id}_prism.tex"
    shutil.copy(tex_out, dest)
    shutil.rmtree(out_dir, ignore_errors=True)

    row = {
        "page_id":   page_id,
        "total_sec": round(t_total,   2),
        "norm_sec":  round(t_norm,    2),
        "yolo_sec":  round(t_yolo,    2),
        "rapid_sec": round(t_text,    2),
        "math_sec":  round(t_math,    2),
        "table_sec": round(t_table,   2),
        "peak_mb":   round(peak_mb,   1),
        "n_math_crops":  len(math_lats),
        "n_rapid_crops": len(text_lats),
    }
    rows.append(row)

    print(
        f"  total={t_total:.1f}s  "
        f"texo={t_math:.1f}s({len(math_lats)} eq)  "
        f"rapid={t_text:.1f}s(1 full-page call)  "
        f"peak={peak_mb:.0f}MB"
    )

# ── Write latency CSV ──────────────────────────────────────────────────────────
fields = ["page_id","total_sec","norm_sec","yolo_sec","rapid_sec",
          "math_sec","table_sec","peak_mb","n_math_crops","n_rapid_crops"]
with open(LATENCY_LOG, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print(f"\n[OK] Latency log -> {LATENCY_LOG}  ({len(rows)} pages)")

# ── Print detailed latency table ───────────────────────────────────────────────
totals    = [r["total_sec"] for r in rows]
math_secs = [r["math_sec"]  for r in rows]
rapid_secs= [r["rapid_sec"] for r in rows]
ram_vals  = [r["peak_mb"]   for r in rows]

print(f"\n{'='*72}")
print(f"  Detailed Latency Breakdown (full-page OCR pipeline, {len(rows)} pages)")
print(f"{'='*72}")
print(f"  {'Page':>4}  {'Total':>7}  {'Rapid':>7}  {'Texo':>6}  {'YOLO':>6}  {'RAM':>8}  {'Equations':>9}")
print(f"  {'-'*62}")
for r in rows:
    print(
        f"  {r['page_id']:>4}  "
        f"{r['total_sec']:>6.1f}s  "
        f"{r['rapid_sec']:>6.1f}s  "
        f"{r['math_sec']:>5.1f}s  "
        f"{r['yolo_sec']:>5.1f}s  "
        f"{r['peak_mb']:>7.0f}MB  "
        f"{r['n_math_crops']:>6} eq"
    )
print(f"  {'-'*62}")
print(
    f"  {'AVG':>4}  "
    f"{statistics.mean(totals):>6.1f}s  "
    f"{statistics.mean(rapid_secs):>6.1f}s  "
    f"{statistics.mean(math_secs):>5.1f}s  "
    f"       "
    f"{statistics.mean(ram_vals):>7.0f}MB"
)
print(
    f"  {'MED':>4}  "
    f"{statistics.median(totals):>6.1f}s  "
    f"{statistics.median(rapid_secs):>6.1f}s  "
    f"{statistics.median(math_secs):>5.1f}s"
)
print(
    f"  {'P95':>4}  "
    f"{sorted(totals)[max(0,int(len(totals)*0.95)-1)]:>6.1f}s"
)
print(f"  {'MAX':>4}  {max(totals):>6.1f}s")

# Previous baseline (RapidOCR pipeline)
print(f"\n  vs PREVIOUS (RapidOCR, no GOT):")
print(f"    Avg latency  16.35s  →  {statistics.mean(totals):.1f}s")
print(f"    Avg RAM     1275 MB  →  {statistics.mean(ram_vals):.0f} MB")
print(f"{'='*72}\n")

# ── Update latency_log.csv for run_full_benchmark.py compatibility ─────────────
compat_log = Path("benchmark_results/latency_log.csv")
with open(compat_log, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["page_id","total_sec","peak_mb"])
    w.writeheader()
    for r in rows:
        w.writerow({"page_id": r["page_id"], "total_sec": r["total_sec"], "peak_mb": r["peak_mb"]})

# ── Run full metric evaluation ─────────────────────────────────────────────────
print("="*60)
print("  Running full metric evaluation...")
print("="*60 + "\n")
import subprocess
subprocess.run([sys.executable, "run_full_benchmark.py"])
