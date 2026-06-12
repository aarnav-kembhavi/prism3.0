"""
run_robustness_benchmark.py
───────────────────────────
Run PRISM on all images in the robustness_dataset and report EDR metrics
organised by document type, variant, and degradation level.

Dataset structure:
  robustness_dataset/<doc_type>/<variant>/<degradation>.png + ground_truth.tex

Degradation levels tested:
  screenshot, glared, glared_rotated_5deg, glared_rotated_15deg,
  glared_rotated_25deg, glared_rotated_45deg
"""

import csv, gc, os, shutil, statistics, time
from pathlib import Path

import numpy as np
import psutil

DATASET_DIR = Path("robustness_dataset")
OUT_BASE    = Path("robustness_results/custom_dataset")
CSV_OUT     = OUT_BASE / "robustness_metrics.csv"
OUT_BASE.mkdir(parents=True, exist_ok=True)

DEGRADATIONS = [
    "screenshot",
    "glared",
    "glared_rotated_5deg",
    "glared_rotated_15deg",
    "glared_rotated_25deg",
    "glared_rotated_45deg",
]

YOLO_MODEL_PATH = "yolov11n-doclaynet.onnx"

# ── Pipeline imports ──────────────────────────────────────────────────────────
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
from evaluation.normalizer import normalize_latex, split_math_and_text
from run_full_benchmark import compute_edr

TEXT_CLASSES    = {"Text", "Title", "Section-header", "Caption",
                   "Footnote", "Page-footer", "Page-header", "List-item"}
MATH_CLASSES    = {"Formula"}
TABLE_CLASSES   = {"Table"}
IMAGE_CLASSES   = {"Picture"}
LIST_ITEM_CLASS = "List-item"


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
        crop = xyxy_to_pil_crop(image_fidelity if class_name in IMAGE_CLASSES else image_norm,
                                [x1, y1, x2, y2])
        if class_name == "Page-header":
            fc = xyxy_to_pil_crop(image_fidelity, [x1, y1, x2, y2])
            if _is_likely_logo(fc):
                class_name = "Picture"
                crop = fc
        detections.append({"bbox": [x1, y1, x2, y2], "class_id": class_id,
                            "class_name": class_name, "confidence": confidence, "crop": crop})
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
        document = assemble_document(
            _adjust_figure_paths(full_parts), full_idx, True,
            _adjust_figure_paths(left_parts), left_idx,
            _adjust_figure_paths(right_parts), right_idx,
            header_logo_fname)
    else:
        body_sorted = apply_semantic_reading_order(body_detections, img_w, img_h)
        body_parts, list_idx, _, _ = route_and_extract(
            body_sorted, str(figures_dir), is_screenshot=is_screenshot)
        document = assemble_document(
            _adjust_figure_paths(body_parts), list_idx, False,
            header_logo=header_logo_fname)

    tex_path = output_dir / "main.tex"
    save_tex(document, str(tex_path))
    return {
        "tex_path":  tex_path,
        "total_sec": time.perf_counter() - t0,
        "peak_mb":   process.memory_info().rss / 1024 / 1024,
    }


# ── Collect all test cases ────────────────────────────────────────────────────
test_cases = []
for doc_type in sorted(DATASET_DIR.iterdir()):
    if not doc_type.is_dir(): continue
    for variant in sorted(doc_type.iterdir()):
        if not variant.is_dir(): continue
        gt_path = variant / "ground_truth.tex"
        if not gt_path.exists(): continue
        for deg in DEGRADATIONS:
            img = variant / f"{deg}.png"
            if img.exists():
                test_cases.append({
                    "doc_type": doc_type.name,
                    "variant":  variant.name,
                    "degradation": deg,
                    "img_path": img,
                    "gt_path":  gt_path,
                })

total = len(test_cases)
print(f"[*] Robustness benchmark: {total} images "
      f"({len(list(DATASET_DIR.iterdir()))} doc types x 4 variants x {len(DEGRADATIONS)} degradations)\n")

yolo_model = get_yolo_model(YOLO_MODEL_PATH)
rows = []

for i, tc in enumerate(test_cases, 1):
    label = f"{tc['doc_type']}/{tc['variant']}/{tc['degradation']}"
    print(f"[{i:>2}/{total}] {label}")

    work_dir = OUT_BASE / "tmp" / f"{tc['doc_type']}_{tc['variant']}_{tc['degradation']}"
    if work_dir.exists():
        shutil.rmtree(work_dir)

    try:
        result = process_page(tc["img_path"], work_dir, yolo_model)
    except Exception as e:
        print(f"  [!] ERROR: {e}")
        import traceback; traceback.print_exc()
        continue

    pred = result["tex_path"].read_text(encoding="utf-8", errors="ignore")
    gt   = tc["gt_path"].read_text(encoding="utf-8", errors="ignore")
    pred_norm = normalize_latex(pred,  remove_spaces=True)
    gt_norm   = normalize_latex(gt,    remove_spaces=True)
    pred_math, pred_text = split_math_and_text(pred_norm)
    gt_math,   gt_text   = split_math_and_text(gt_norm)

    overall = compute_edr(pred_norm, gt_norm)
    math    = compute_edr(pred_math, gt_math)
    text    = compute_edr(pred_text, gt_text)

    shutil.rmtree(work_dir, ignore_errors=True)

    rows.append({**tc, "overall_edr": overall, "math_edr": math,
                 "text_edr": text, "total_sec": result["total_sec"]})
    print(f"  {result['total_sec']:.1f}s  EDR={overall:.1%}  text={text:.1%}  math={math:.1%}")

unload_texo()

# ── Write CSV ─────────────────────────────────────────────────────────────────
fields = ["doc_type","variant","degradation","overall_edr","math_edr","text_edr","total_sec"]
with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader(); w.writerows(rows)

# ── Summary table ─────────────────────────────────────────────────────────────
DEG_SHORT = {
    "screenshot":           "clean",
    "glared":               "glare",
    "glared_rotated_5deg":  "g+5°",
    "glared_rotated_15deg": "g+15°",
    "glared_rotated_25deg": "g+25°",
    "glared_rotated_45deg": "g+45°",
}
COL_W = 8

def get(doc, var, deg, key="overall_edr"):
    for r in rows:
        if r["doc_type"]==doc and r["variant"]==var and r["degradation"]==deg:
            return r[key]
    return None

def fmt(v):
    return f"{v:.1%}" if v is not None else "  N/A "

doc_types = sorted(set(r["doc_type"] for r in rows))
variants  = sorted(set(r["variant"]  for r in rows))
degs      = DEGRADATIONS

print("\n" + "=" * 90)
print("  PRISM Custom Robustness Dataset  —  Overall EDR")
print("=" * 90)
hdr = f"  {'Document / Variant':<36}" + "".join(f"{DEG_SHORT[d]:>{COL_W}}" for d in degs)
print(hdr)
print("  " + "-" * 86)

for doc in doc_types:
    doc_short = doc.split("_",1)[1].replace("_"," ")
    print(f"  {doc_short.upper()}")
    for var in variants:
        var_short = var.replace("variant_","")
        row_label = f"    {var_short:<32}"
        vals = [get(doc, var, d) for d in degs]
        print(f"  {row_label}" + "".join(f"{fmt(v):>{COL_W}}" for v in vals))

print("  " + "-" * 86)
print("  AVERAGES")
for var in variants:
    var_short = var.replace("variant_","")
    row_label = f"    {var_short:<32}"
    avgs = []
    for d in degs:
        vals = [r["overall_edr"] for r in rows if r["variant"]==var and r["degradation"]==d]
        avgs.append(statistics.mean(vals) if vals else None)
    print(f"  {row_label}" + "".join(f"{fmt(v):>{COL_W}}" for v in avgs))

print("  " + "-" * 86)
all_by_deg = {d: [r["overall_edr"] for r in rows if r["degradation"]==d] for d in degs}
overall_row = f"  {'  ALL':<36}"
print(overall_row + "".join(
    f"{statistics.mean(v):.1%}".rjust(COL_W) if v else "  N/A ".rjust(COL_W)
    for v in all_by_deg.values()))
print("=" * 90)
print(f"\n  CSV saved -> {CSV_OUT.resolve()}")
