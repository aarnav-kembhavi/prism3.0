"""
run_bench_26.py  —  Run PRISM on first 26 PDF2LaTeX benchmark images in-process.
Models are loaded once and reused across all pages.
"""

import csv, gc, os, shutil, statistics, sys, time
from pathlib import Path

import psutil, torch
import numpy as np
from PIL import Image

IMAGES_DIR    = Path("benchmark_results/temp_images")
GT_DIR        = Path("pdf2latex_dataset/dataset")
PRISM_TEX_DIR = Path("benchmark_results/prism_tex_26")
LATENCY_LOG   = Path("benchmark_results/latency_log_26.csv")
PRISM_TEX_DIR.mkdir(parents=True, exist_ok=True)

# ── Pipeline imports (done once) ─────────────────────────────────
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
from evaluation.eval import levenshtein_distance
from run_full_benchmark import compute_edr

YOLO_MODEL_PATH = "yolov11n-doclaynet.onnx"
TEXT_CLASSES    = {"Text","Title","Section-header","Caption",
                   "Footnote","Page-footer","Page-header","List-item"}
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
                            "class_name": class_name, "confidence": confidence, "crop": crop})
    return detections


def route_and_extract(detections, figures_dir, figure_start=0, is_screenshot=False, math_start=0):
    os.makedirs(figures_dir, exist_ok=True)
    body_parts, list_indices = [], set()
    figure_counter = figure_start
    math_counter   = [math_start]
    text_indices  = [i for i, d in enumerate(detections) if d["class_name"] in TEXT_CLASSES]
    math_indices  = [i for i, d in enumerate(detections) if d["class_name"] in MATH_CLASSES]
    table_indices = [i for i, d in enumerate(detections) if d["class_name"] in TABLE_CLASSES]
    if math_indices:
        crops = [detections[i]["crop"] for i in math_indices]
        results = run_math_recognition_batched(crops, figures_dir, math_counter)
        for idx, raw in zip(math_indices, results):
            detections[idx]["raw_content"] = raw
    if text_indices:
        texts = run_text_ocr_batched(
            [detections[i]["crop"] for i in text_indices], is_screenshot=is_screenshot)
        for idx, txt in zip(text_indices, texts):
            detections[idx]["raw_content"] = txt
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
            if raw: body_parts.append(wrap_content(cn, raw))
        elif cn in IMAGE_CLASSES:
            figure_counter += 1
            fname = f"figure_{figure_counter:03d}.png"
            crop.save(os.path.join(figures_dir, fname))
            body_parts.append(wrap_content("Picture", fname))
    return body_parts, list_indices, figure_counter, math_counter[0]


def process_page(img_path: Path, output_dir: Path, yolo_model) -> dict:
    """Run the full PRISM pipeline on one page. Returns timing + content dict."""
    process = psutil.Process(os.getpid())
    t0 = time.perf_counter()

    assets_dir  = output_dir / "assets"
    figures_dir = assets_dir / "figures"
    for d in [output_dir, assets_dir, figures_dir, output_dir / "logs"]:
        d.mkdir(parents=True, exist_ok=True)

    # Stage 1: Normalization
    t1 = time.perf_counter()
    image_norm, image_fidelity, modality = normalize_image_pil(str(img_path))
    is_screenshot = (modality.modality == CaptureModality.SCREENSHOT)
    image_norm.save(assets_dir / "normalized.png")
    t_norm = time.perf_counter() - t1

    # Stage 2: YOLO detection
    t2 = time.perf_counter()
    yolo_input = str(assets_dir / "normalized.png")
    detections = run_detection(yolo_model, image_norm, image_fidelity, yolo_input)
    img_w, img_h = image_norm.width, image_norm.height
    detections = postprocess_detections(detections, img_w, img_h)
    t_yolo = time.perf_counter() - t2

    # Header suppression + re-crop
    HEADER_SUPPRESS_H_FRAC = 0.12
    header_suppress_y = img_h * HEADER_SUPPRESS_H_FRAC
    detections = [d for d in detections if not (
        d["class_name"] in {"Section-header","Page-header"} and d["bbox"][3] <= header_suppress_y)]

    HEADER_H_FRAC, HEADER_W_FRAC = 0.065, 0.25
    header_right_box = [img_w*(1-HEADER_W_FRAC), 0, img_w, img_h*HEADER_H_FRAC]
    if not any(d["class_name"]=="Picture" and d["bbox"][0]>=header_right_box[0]
               and d["bbox"][3]<=header_right_box[3] for d in detections):
        hx1,hy1,hx2,hy2 = [int(v) for v in header_right_box]
        header_crop = xyxy_to_pil_crop(image_fidelity, [hx1,hy1,hx2,hy2])
        if header_crop.width > 20:
            detections.insert(0, {"bbox":[hx1,hy1,hx2,hy2],"class_id":-1,
                                   "class_name":"Picture","crop":header_crop,"is_header_logo":True})

    FORMULA_PAD = 12
    for det in detections:
        bbox = det["bbox"]
        if det["class_name"] in MATH_CLASSES:
            x1,y1,x2,y2 = bbox
            bbox = [max(0,x1-FORMULA_PAD),max(0,y1-FORMULA_PAD),
                    min(img_w,x2+FORMULA_PAD),min(img_h,y2+FORMULA_PAD)]
        det["crop"] = xyxy_to_pil_crop(image_fidelity if det["class_name"] in IMAGE_CLASSES
                                        else image_norm, bbox)

    del image_norm, image_fidelity
    gc.collect()

    # Stage 3: Extraction
    t3 = time.perf_counter()
    n_math_before = len(get_math_batch_latencies())
    n_text_before = len(get_text_batch_latencies())

    col_count       = detect_column_count(detections, img_w)
    header_logo_dets= [d for d in detections if d.get("is_header_logo")]
    body_detections = [d for d in detections if not d.get("is_header_logo")]
    header_logo_fname = "assets/figure_header_logo.png" if header_logo_dets else None
    if header_logo_fname:
        header_logo_dets[0]["crop"].save(output_dir / header_logo_fname)

    if col_count == 2:
        full_dets, left_dets, right_dets = split_detections_by_column(
            body_detections, img_w, img_h, use_dag=True)
        full_parts, full_idx, f_cnt, m_cnt = route_and_extract(
            full_dets, str(figures_dir), 0, is_screenshot=is_screenshot, math_start=0)
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
        document   = assemble_document(body_parts, list_idx, False, header_logo=header_logo_fname)

    math_lats = get_math_batch_latencies()[n_math_before:]
    text_lats = get_text_batch_latencies()[n_text_before:]
    t_math = sum(math_lats) / 1000.0
    t_text = sum(text_lats) / 1000.0
    t_extract = time.perf_counter() - t3

    # Stage 4: Save
    tex_path = output_dir / "main.tex"
    save_tex(document, str(tex_path))

    peak_mb  = process.memory_info().rss / 1024 / 1024
    t_total  = time.perf_counter() - t0

    return {
        "norm_sec":  t_norm,
        "yolo_sec":  t_yolo,
        "ocr_sec":   t_text,
        "math_sec":  t_math,
        "table_sec": sum(get_table_latencies()) / 1000.0,
        "total_sec": t_total,
        "peak_mb":   peak_mb,
        "tex_path":  tex_path,
    }


# ── Main ─────────────────────────────────────────────────────────
all_imgs = sorted(
    [p for p in IMAGES_DIR.glob("*.png") if (GT_DIR / f"{p.stem}_gt.tex").exists()],
    key=lambda p: int(p.stem),
)[:26]

print(f"[*] Running {len(all_imgs)} pages (in-process, models shared)\n")

# Pre-load YOLO once
yolo_model = get_yolo_model(YOLO_MODEL_PATH)

rows = []
for idx, img_path in enumerate(all_imgs, 1):
    page_id = img_path.stem
    out_dir = Path(f"bench_tmp/{page_id}_output")
    if out_dir.exists():
        shutil.rmtree(out_dir)

    print(f"[{idx:>2}/26] {img_path.name}", flush=True)
    try:
        result = process_page(img_path, out_dir, yolo_model)
    except Exception as e:
        print(f"  [!] ERROR: {e}")
        import traceback; traceback.print_exc()
        continue

    tex_path = result["tex_path"]
    dest = PRISM_TEX_DIR / f"{page_id}_prism.tex"
    shutil.copy(tex_path, dest)
    shutil.rmtree(out_dir, ignore_errors=True)

    # Metrics
    pred = dest.read_text(encoding="utf-8", errors="ignore")
    gt   = (GT_DIR / f"{page_id}_gt.tex").read_text(encoding="utf-8", errors="ignore")
    pred_norm = normalize_latex(pred, remove_spaces=True)
    gt_norm   = normalize_latex(gt,   remove_spaces=True)
    pred_math, pred_text = split_math_and_text(pred_norm)
    gt_math,   gt_text   = split_math_and_text(gt_norm)
    overall_edr = compute_edr(pred_norm, gt_norm)
    math_edr    = compute_edr(pred_math, gt_math)
    text_edr    = compute_edr(pred_text, gt_text)

    row = {**{k: result[k] for k in
              ["norm_sec","yolo_sec","ocr_sec","math_sec","table_sec","total_sec","peak_mb"]},
           "page_id": page_id, "overall_edr": overall_edr,
           "math_edr": math_edr, "text_edr": text_edr}
    rows.append(row)
    print(f"  {result['total_sec']:.1f}s  {result['peak_mb']:.0f}MB  "
          f"EDR={overall_edr:.1%}  math={math_edr:.1%}  text={text_edr:.1%}")

# Unload math model if loaded
unload_texo()

# Write CSV
fields = ["page_id","norm_sec","yolo_sec","ocr_sec","math_sec","table_sec",
          "total_sec","peak_mb","overall_edr","math_edr","text_edr"]
with open(LATENCY_LOG, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader(); w.writerows(rows)

if rows:
    def avg(k): return statistics.mean(r[k] for r in rows)
    def med(k): return statistics.median(r[k] for r in rows)
    print("\n" + "="*56)
    print("  PRISM  —  First 26 PDF2LaTeX pages")
    print("="*56)
    print(f"  Pages       : {len(rows)}/26")
    print(f"  Overall EDR : {avg('overall_edr'):.1%}  (median {med('overall_edr'):.1%})")
    print(f"  Text EDR    : {avg('text_edr'):.1%}")
    print(f"  Math EDR    : {avg('math_edr'):.1%}")
    print(f"  Avg latency : {avg('total_sec'):.1f}s  (median {med('total_sec'):.1f}s)")
    print(f"  Avg peak RAM: {avg('peak_mb'):.0f} MB")
    print("="*56)
