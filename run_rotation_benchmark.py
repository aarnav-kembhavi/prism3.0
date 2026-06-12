"""
run_rotation_benchmark.py
─────────────────────────
Run PRISM on the first 26 PDF2LaTeX images at 0°, 5°, 15°, 25°, and 45°
rotations to measure pipeline robustness under skew/tilt degradation.

Rotated images are saved to:
    robustness_results/rotated_pdf2latex/<page_id>/<angle>deg.png

Summary table is printed to stdout and written to:
    robustness_results/rotation_benchmark.csv
"""

import csv, gc, os, shutil, statistics, sys, time
from pathlib import Path

import cv2
import numpy as np
import psutil
from PIL import Image

# ── Config ───────────────────────────────────────────────────────────────────
IMAGES_DIR  = Path("benchmark_results/temp_images")
GT_DIR      = Path("pdf2latex_dataset/dataset")
OUT_BASE    = Path("robustness_results")
ROTATED_DIR = OUT_BASE / "rotated_pdf2latex"
CSV_OUT     = OUT_BASE / "rotation_benchmark.csv"
ANGLES      = [0, 5, 15, 25, 45]
YOLO_MODEL_PATH = "yolov11n-doclaynet.onnx"

OUT_BASE.mkdir(parents=True, exist_ok=True)
ROTATED_DIR.mkdir(parents=True, exist_ok=True)

# ── Pipeline imports (models loaded once) ────────────────────────────────────
from normalization import normalize_image_pil
from normalization.modality import CaptureModality
from models_interface import (
    run_text_ocr_batched, run_math_recognition_batched,
    run_table_extraction_batched, get_yolo_model, unload_texo,
    get_math_batch_latencies, get_text_batch_latencies, get_table_latencies,
)
from layout_utils import (
    apply_semantic_reading_order, sort_detections_geometric,
    xyxy_to_pil_crop, detect_column_count, split_detections_by_column,
)
from latex_builder import wrap_content, assemble_document, save_tex
from detection_postprocess import postprocess_detections
from evaluation.normalizer import normalize_latex, split_math_and_text
from run_full_benchmark import compute_edr

TEXT_CLASSES    = {"Text", "Title", "Section-header", "Caption",
                   "Footnote", "Page-footer", "Page-header", "List-item"}
MATH_CLASSES    = {"Formula"}
TABLE_CLASSES   = {"Table"}
IMAGE_CLASSES   = {"Picture"}
LIST_ITEM_CLASS = "List-item"


# ── Rotation helper ───────────────────────────────────────────────────────────
def _rotate_image(src_path: Path, angle: int, dst_path: Path) -> Path:
    """Rotate image by `angle` degrees with dark padding, save to dst_path."""
    if angle == 0:
        if not dst_path.exists():
            shutil.copy(src_path, dst_path)
        return dst_path

    img = cv2.imread(str(src_path))
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    pad = 150
    M[0, 2] += pad
    M[1, 2] += pad

    rotated = cv2.warpAffine(img, M, (new_w + pad * 2, new_h + pad * 2),
                             borderValue=(40, 40, 40))
    cv2.imwrite(str(dst_path), rotated)
    return dst_path


# ── Pipeline helpers (copied from run_bench_26) ───────────────────────────────
def _is_likely_logo(crop):
    arr = np.array(crop.convert("RGB"), dtype=np.float32)
    gray = arr.mean(axis=2)
    return float(np.mean(gray < 230)) < 0.15 and float(arr.std()) > 8.0


def _adjust_figure_paths(parts):
    return [
        p.replace("{figure_", "{assets/figures/figure_")
        if "includegraphics" in p else p
        for p in parts
    ]


def run_detection(model, image_norm, image_fidelity, image_path):
    results = model(image_path, verbose=False)
    detections = []
    result = results[0]
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        class_id   = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        class_name = result.names[class_id]
        if class_name in IMAGE_CLASSES:
            crop = xyxy_to_pil_crop(image_fidelity, [x1, y1, x2, y2])
        else:
            crop = xyxy_to_pil_crop(image_norm, [x1, y1, x2, y2])
        if class_name == "Page-header":
            fc = xyxy_to_pil_crop(image_fidelity, [x1, y1, x2, y2])
            if _is_likely_logo(fc):
                class_name = "Picture"
                crop = fc
        detections.append({"bbox": [x1, y1, x2, y2], "class_id": class_id,
                            "class_name": class_name, "confidence": confidence,
                            "crop": crop})
    return detections


def route_and_extract(detections, figures_dir, figure_start=0,
                      is_screenshot=False, math_start=0):
    os.makedirs(figures_dir, exist_ok=True)
    body_parts, list_indices = [], set()
    figure_counter = figure_start
    math_counter   = [math_start]
    text_indices = [i for i, d in enumerate(detections) if d["class_name"] in TEXT_CLASSES]
    math_indices = [i for i, d in enumerate(detections) if d["class_name"] in MATH_CLASSES]
    if math_indices:
        crops = [detections[i]["crop"] for i in math_indices]
        results = run_math_recognition_batched(crops, figures_dir, math_counter)
        for idx, raw in zip(math_indices, results):
            detections[idx]["raw_content"] = raw
    if text_indices:
        texts = run_text_ocr_batched(
            [detections[i]["crop"] for i in text_indices],
            is_screenshot=is_screenshot)
        for idx, txt in zip(text_indices, texts):
            detections[idx]["raw_content"] = txt
    table_indices = [i for i, d in enumerate(detections) if d["class_name"] in TABLE_CLASSES]
    if table_indices:
        table_results = run_table_extraction_batched([detections[i]["crop"] for i in table_indices])
        for idx, raw in zip(table_indices, table_results):
            detections[idx]["raw_content"] = raw
    for i, det in enumerate(detections):
        cn, crop = det["class_name"], det["crop"]
        if cn in TEXT_CLASSES or cn in MATH_CLASSES:
            raw = det.get("raw_content", "")
            wrapped = wrap_content(cn, raw)
            if cn == LIST_ITEM_CLASS:
                list_indices.add(len(body_parts))
            body_parts.append(wrapped)
        elif cn in TABLE_CLASSES:
            raw = det.get("raw_content", "")
            if raw:
                body_parts.append(wrap_content(cn, raw))
        elif cn in IMAGE_CLASSES:
            figure_counter += 1
            fname = f"figure_{figure_counter:03d}.png"
            crop.save(os.path.join(figures_dir, fname))
            body_parts.append(wrap_content("Picture", fname))
    return body_parts, list_indices, figure_counter, math_counter[0]


def process_page(img_path: Path, output_dir: Path, yolo_model) -> dict:
    process = psutil.Process(os.getpid())
    t0 = time.perf_counter()

    assets_dir  = output_dir / "assets"
    figures_dir = assets_dir / "figures"
    for d in [output_dir, assets_dir, figures_dir, output_dir / "logs"]:
        d.mkdir(parents=True, exist_ok=True)

    image_norm, image_fidelity, modality = normalize_image_pil(str(img_path))
    is_screenshot = (modality.modality == CaptureModality.SCREENSHOT)
    image_norm.save(assets_dir / "normalized.png")

    yolo_input = str(assets_dir / "normalized.png")
    detections = run_detection(yolo_model, image_norm, image_fidelity, yolo_input)
    img_w, img_h = image_norm.width, image_norm.height
    detections = postprocess_detections(detections, img_w, img_h)

    HEADER_SUPPRESS_H_FRAC = 0.12
    header_suppress_y = img_h * HEADER_SUPPRESS_H_FRAC
    detections = [d for d in detections if not (
        d["class_name"] in {"Section-header", "Page-header"}
        and d["bbox"][3] <= header_suppress_y)]

    HEADER_H_FRAC, HEADER_W_FRAC = 0.065, 0.25
    header_right_box = [img_w * (1 - HEADER_W_FRAC), 0, img_w, img_h * HEADER_H_FRAC]
    if not any(d["class_name"] == "Picture" and d["bbox"][0] >= header_right_box[0]
               and d["bbox"][3] <= header_right_box[3] for d in detections):
        hx1, hy1, hx2, hy2 = [int(v) for v in header_right_box]
        header_crop = xyxy_to_pil_crop(image_fidelity, [hx1, hy1, hx2, hy2])
        if header_crop.width > 20:
            detections.insert(0, {"bbox": [hx1, hy1, hx2, hy2], "class_id": -1,
                                  "class_name": "Picture", "crop": header_crop,
                                  "is_header_logo": True})

    FORMULA_PAD = 12
    for det in detections:
        bbox = det["bbox"]
        if det["class_name"] in MATH_CLASSES:
            x1, y1, x2, y2 = bbox
            bbox = [max(0, x1 - FORMULA_PAD), max(0, y1 - FORMULA_PAD),
                    min(img_w, x2 + FORMULA_PAD), min(img_h, y2 + FORMULA_PAD)]
        det["crop"] = xyxy_to_pil_crop(
            image_fidelity if det["class_name"] in IMAGE_CLASSES else image_norm, bbox)

    del image_norm, image_fidelity
    gc.collect()

    col_count        = detect_column_count(detections, img_w)
    header_logo_dets = [d for d in detections if d.get("is_header_logo")]
    body_detections  = [d for d in detections if not d.get("is_header_logo")]
    header_logo_fname = "assets/figure_header_logo.png" if header_logo_dets else None
    if header_logo_fname:
        header_logo_dets[0]["crop"].save(output_dir / header_logo_fname)

    if col_count == 2:
        full_dets, left_dets, right_dets = split_detections_by_column(
            body_detections, img_w, img_h, use_dag=True)
        full_parts, full_idx, f_cnt, m_cnt = route_and_extract(
            full_dets, str(figures_dir), 0, is_screenshot=is_screenshot)
        left_parts, left_idx, f_cnt, m_cnt = route_and_extract(
            left_dets, str(figures_dir), f_cnt, is_screenshot=is_screenshot, math_start=m_cnt)
        right_parts, right_idx, f_cnt, m_cnt = route_and_extract(
            right_dets, str(figures_dir), f_cnt, is_screenshot=is_screenshot, math_start=m_cnt)
        full_parts  = _adjust_figure_paths(full_parts)
        left_parts  = _adjust_figure_paths(left_parts)
        right_parts = _adjust_figure_paths(right_parts)
        document = assemble_document(full_parts, full_idx, True,
                                     left_parts, left_idx, right_parts, right_idx,
                                     header_logo_fname)
    else:
        body_sorted = apply_semantic_reading_order(body_detections, img_w, img_h)
        body_parts, list_idx, _, _ = route_and_extract(
            body_sorted, str(figures_dir), is_screenshot=is_screenshot)
        body_parts = _adjust_figure_paths(body_parts)
        document   = assemble_document(body_parts, list_idx, False,
                                       header_logo=header_logo_fname)

    tex_path = output_dir / "main.tex"
    save_tex(document, str(tex_path))
    t_total = time.perf_counter() - t0
    peak_mb = process.memory_info().rss / 1024 / 1024
    return {"tex_path": tex_path, "total_sec": t_total, "peak_mb": peak_mb}


# ── Main ──────────────────────────────────────────────────────────────────────
all_imgs = sorted(
    [p for p in IMAGES_DIR.glob("*.png")
     if (GT_DIR / f"{p.stem}_gt.tex").exists()],
    key=lambda p: int(p.stem),
)[:26]

print(f"[*] Rotation robustness benchmark — {len(all_imgs)} pages × {len(ANGLES)} angles "
      f"= {len(all_imgs) * len(ANGLES)} total runs\n")

yolo_model = get_yolo_model(YOLO_MODEL_PATH)

rows = []   # one row per (page, angle)

for idx, src_img in enumerate(all_imgs, 1):
    page_id = src_img.stem
    page_dir = ROTATED_DIR / page_id
    page_dir.mkdir(exist_ok=True)

    # Pre-generate all rotated images for this page
    rotated_paths = {}
    for angle in ANGLES:
        dst = page_dir / f"{angle}deg.png"
        rotated_paths[angle] = _rotate_image(src_img, angle, dst)

    # Load GT once per page
    gt_raw = (GT_DIR / f"{page_id}_gt.tex").read_text(encoding="utf-8", errors="ignore")
    gt_norm = normalize_latex(gt_raw, remove_spaces=True)
    _, gt_text = split_math_and_text(gt_norm)
    gt_math, _ = split_math_and_text(gt_norm)

    angle_edrs = {}
    for angle in ANGLES:
        work_dir = OUT_BASE / "tmp" / f"{page_id}_{angle}deg"
        if work_dir.exists():
            shutil.rmtree(work_dir)

        try:
            result = process_page(rotated_paths[angle], work_dir, yolo_model)
        except Exception as e:
            print(f"  [!] page {page_id} @ {angle}°: {e}")
            angle_edrs[angle] = None
            continue

        pred = result["tex_path"].read_text(encoding="utf-8", errors="ignore")
        pred_norm = normalize_latex(pred, remove_spaces=True)
        pred_math, pred_text = split_math_and_text(pred_norm)

        overall = compute_edr(pred_norm, gt_norm)
        math    = compute_edr(pred_math, gt_math)
        text    = compute_edr(pred_text, gt_text)
        angle_edrs[angle] = {"overall": overall, "math": math, "text": text,
                              "sec": result["total_sec"]}

        shutil.rmtree(work_dir, ignore_errors=True)

    # Print per-page summary row
    row_str = f"[{idx:>2}/26] page {page_id:>3}  |"
    for angle in ANGLES:
        v = angle_edrs[angle]
        row_str += f"  {angle:>2}°: {v['overall']:.1%}" if v else f"  {angle:>2}°:   ERR "
    print(row_str)

    for angle in ANGLES:
        v = angle_edrs[angle]
        if v:
            rows.append({
                "page_id": page_id, "angle": angle,
                "overall_edr": v["overall"], "math_edr": v["math"],
                "text_edr": v["text"], "total_sec": v["sec"],
            })

unload_texo()

# ── Write CSV ────────────────────────────────────────────────────────────────
fields = ["page_id", "angle", "overall_edr", "math_edr", "text_edr", "total_sec"]
with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

# ── Summary table ─────────────────────────────────────────────────────────────
def _avg_by_angle(key, angle):
    vals = [r[key] for r in rows if r["angle"] == angle and r[key] is not None]
    return statistics.mean(vals) if vals else float("nan")

print("\n" + "=" * 72)
print("  PRISM Rotation Robustness  —  First 26 PDF2LaTeX pages")
print("=" * 72)
header = f"  {'Metric':<14}" + "".join(f"  {a:>2}°     " for a in ANGLES)
print(header)
print("  " + "-" * 68)
for key, label in [("overall_edr", "Overall EDR"), ("text_edr", "Text EDR"),
                   ("math_edr", "Math EDR")]:
    row = f"  {label:<14}"
    for angle in ANGLES:
        row += f"  {_avg_by_angle(key, angle):>6.1%}  "
    print(row)
print("  " + "-" * 68)
lat_row = f"  {'Avg latency':<14}"
for angle in ANGLES:
    vals = [r["total_sec"] for r in rows if r["angle"] == angle]
    lat_row += f"  {statistics.mean(vals):>5.1f}s  " if vals else "     N/A  "
print(lat_row)
print("=" * 72)
print(f"\n  CSV saved → {CSV_OUT.resolve()}")
print(f"  Rotated images → {ROTATED_DIR.resolve()}")
