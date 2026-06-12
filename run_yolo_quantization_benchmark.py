"""
run_yolo_quantization_benchmark.py
------------------------------------
1. Quantize yolov11n-doclaynet.onnx → yolov11n-doclaynet-int8.onnx
2. Run the full PRISM pipeline on the first 26 PDF2LaTeX pages twice:
     - Pass A: float32 YOLO (baseline)
     - Pass B: INT8 YOLO
3. Compare per-page EDR, latency, and peak RAM.
"""

import csv, gc, os, shutil, statistics, time
from pathlib import Path

import numpy as np
import psutil

# ── Quantize ─────────────────────────────────────────────────────────────────
SRC_MODEL  = "yolov11n-doclaynet.onnx"
INT8_MODEL = "yolov11n-doclaynet-int8.onnx"

if not Path(INT8_MODEL).exists():
    print("[*] Quantizing YOLO model to INT8 (dynamic quantization)...")
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(
        SRC_MODEL,
        INT8_MODEL,
        weight_type=QuantType.QUInt8,
    )
    print(f"    float32: {Path(SRC_MODEL).stat().st_size / 1024**2:.1f} MB")
    print(f"    INT8:    {Path(INT8_MODEL).stat().st_size / 1024**2:.1f} MB")
else:
    print(f"[*] INT8 model already exists ({Path(INT8_MODEL).stat().st_size/1024**2:.1f} MB)")

print(f"    float32: {Path(SRC_MODEL).stat().st_size/1024**2:.1f} MB")

# ── Pipeline imports ──────────────────────────────────────────────────────────
from normalization import normalize_image_pil
from normalization.modality import CaptureModality
from models_interface import (
    run_text_ocr_batched, run_math_recognition_batched,
    run_table_extraction_batched, get_yolo_model, unload_yolo, unload_texo,
)
from layout_utils import (
    apply_semantic_reading_order, sort_detections_geometric,
    xyxy_to_pil_crop, detect_column_count, split_detections_by_column,
)
from latex_builder import wrap_content, assemble_document, save_tex
from detection_postprocess import postprocess_detections
from evaluation.normalizer import normalize_latex, split_math_and_text
from run_full_benchmark import compute_edr

GT_DIR     = Path("pdf2latex_dataset/dataset")
IMAGES_DIR = Path("benchmark_results/temp_images")
OUT_BASE   = Path("benchmark_results/quant_comparison")
OUT_BASE.mkdir(parents=True, exist_ok=True)

TEXT_CLASSES    = {"Text", "Title", "Section-header", "Caption",
                   "Footnote", "Page-footer", "Page-header", "List-item"}
MATH_CLASSES    = {"Formula"}
TABLE_CLASSES   = {"Table"}
IMAGE_CLASSES   = {"Picture"}
LIST_ITEM_CLASS = "List-item"

all_imgs = sorted(
    [p for p in IMAGES_DIR.glob("*.png")
     if (GT_DIR / f"{p.stem}_gt.tex").exists()],
    key=lambda p: int(p.stem),
)[:26]

process = psutil.Process(os.getpid())


def _is_likely_logo(crop):
    arr = np.array(crop.convert("RGB"), dtype=np.float32)
    gray = arr.mean(axis=2)
    return float(np.mean(gray < 230)) < 0.15 and float(arr.std()) > 8.0


def run_page(img_path: Path, out_dir: Path, yolo_model) -> dict:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    assets_dir  = out_dir / "assets"
    figures_dir = assets_dir / "figures"
    for d in [out_dir, assets_dir, figures_dir]:
        d.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    image_norm, image_fidelity, modality = normalize_image_pil(str(img_path))
    is_screenshot = (modality.modality == CaptureModality.SCREENSHOT)
    norm_path = str(assets_dir / "normalized.png")
    image_norm.save(norm_path)

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

    text_indices  = [i for i, d in enumerate(detections) if d["class_name"] in TEXT_CLASSES]
    math_indices  = [i for i, d in enumerate(detections) if d["class_name"] in MATH_CLASSES]
    table_indices = [i for i, d in enumerate(detections) if d["class_name"] in TABLE_CLASSES]

    if math_indices:
        res = run_math_recognition_batched(
            [detections[i]["crop"] for i in math_indices], str(figures_dir), [0])
        for idx, raw in zip(math_indices, res):
            detections[idx]["raw_content"] = raw

    if text_indices:
        texts = run_text_ocr_batched(
            [detections[i]["crop"] for i in text_indices], is_screenshot=is_screenshot)
        for idx, txt in zip(text_indices, texts):
            detections[idx]["raw_content"] = txt

    if table_indices:
        t_res = run_table_extraction_batched([detections[i]["crop"] for i in table_indices])
        for idx, raw in zip(table_indices, t_res):
            detections[idx]["raw_content"] = raw

    body_parts, list_idx = [], set()
    figure_counter = 0
    col_count = detect_column_count(detections, img_w)
    if col_count == 2:
        from layout_utils import split_detections_by_column
        full_dets, left_dets, right_dets = split_detections_by_column(
            detections, img_w, img_h, use_dag=True)
        def _make_parts(dets, fc=0, mc=0):
            parts, li = [], set()
            for det in dets:
                cn = det["class_name"]
                if cn in TEXT_CLASSES or cn in MATH_CLASSES:
                    raw = det.get("raw_content", "")
                    w = wrap_content(cn, raw)
                    if cn == LIST_ITEM_CLASS: li.add(len(parts))
                    parts.append(w)
                elif cn in TABLE_CLASSES:
                    raw = det.get("raw_content", "")
                    if raw: parts.append(wrap_content(cn, raw))
                elif cn in IMAGE_CLASSES:
                    fc += 1
                    fname = f"figure_{fc:03d}.png"
                    det["crop"].save(str(figures_dir / fname))
                    parts.append(wrap_content("Picture", fname))
            return parts, li
        fp, fi = _make_parts(full_dets)
        lp, li2 = _make_parts(left_dets)
        rp, ri = _make_parts(right_dets)
        document = assemble_document(fp, fi, True, lp, li2, rp, ri)
    else:
        body_sorted = apply_semantic_reading_order(detections, img_w, img_h)
        for det in body_sorted:
            cn = det["class_name"]
            if cn in TEXT_CLASSES or cn in MATH_CLASSES:
                raw = det.get("raw_content", "")
                w = wrap_content(cn, raw)
                if cn == LIST_ITEM_CLASS: list_idx.add(len(body_parts))
                body_parts.append(w)
            elif cn in TABLE_CLASSES:
                raw = det.get("raw_content", "")
                if raw: body_parts.append(wrap_content(cn, raw))
            elif cn in IMAGE_CLASSES:
                figure_counter += 1
                fname = f"figure_{figure_counter:03d}.png"
                det["crop"].save(str(figures_dir / fname))
                body_parts.append(wrap_content("Picture", fname))
        document = assemble_document(body_parts, list_idx, False)

    tex_path = out_dir / "main.tex"
    save_tex(document, str(tex_path))

    elapsed = time.perf_counter() - t0
    peak_mb = process.memory_info().rss / 1024 / 1024

    gt_path = GT_DIR / f"{img_path.stem}_gt.tex"
    pred = normalize_latex(tex_path.read_text(encoding="utf-8", errors="ignore"), remove_spaces=True)
    gt   = normalize_latex(gt_path.read_text(encoding="utf-8", errors="ignore"), remove_spaces=True)
    pm, pt = split_math_and_text(pred)
    gm, gt_ = split_math_and_text(gt)

    return {
        "pid": img_path.stem,
        "overall": compute_edr(pred, gt),
        "math":    compute_edr(pm, gm),
        "text":    compute_edr(pt, gt_),
        "sec":     elapsed,
        "mb":      peak_mb,
        "n_math":  len(math_indices),
        "n_text":  len(text_indices),
        "n_table": len(table_indices),
    }


# ── Run both passes ───────────────────────────────────────────────────────────
def run_pass(model_path: str, label: str) -> list:
    print(f"\n{'='*70}")
    print(f"  Pass: {label}  ({model_path})")
    print('='*70)
    yolo = get_yolo_model(model_path)
    rows = []
    out_dir_base = OUT_BASE / label.replace(" ", "_").lower()
    for i, img_path in enumerate(all_imgs, 1):
        pid = img_path.stem
        print(f"[{i:>2}/26] page {pid}", flush=True)
        r = run_page(img_path, out_dir_base / pid, yolo)
        rows.append(r)
        print(f"  {r['sec']:.1f}s  {r['mb']:.0f}MB  "
              f"EDR={r['overall']:.1%}  text={r['text']:.1%}  math={r['math']:.1%}")
        gc.collect()
    unload_texo()
    unload_yolo()
    gc.collect()
    return rows


rows_f32  = run_pass(SRC_MODEL,  "float32 YOLO")
rows_int8 = run_pass(INT8_MODEL, "INT8 YOLO")

# ── Save CSVs ─────────────────────────────────────────────────────────────────
for label, rows in [("float32", rows_f32), ("int8", rows_int8)]:
    csv_path = OUT_BASE / f"{label}_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pid","overall","math","text","sec","mb"])
        w.writeheader()
        w.writerows({k: r[k] for k in ["pid","overall","math","text","sec","mb"]} for r in rows)

# ── Comparison table ──────────────────────────────────────────────────────────
print("\n" + "="*100)
print("  YOLO Quantization Comparison  --  First 26 PDF2LaTeX pages")
print("="*100)
print(f"  {'Pg':>3}  | {'-- float32 --':^28}  | {'-- INT8 --':^28}  | Delta EDR")
print(f"  {'':>3}  | {'EDR':>7} {'text':>7} {'math':>7} {'s':>4}  | {'EDR':>7} {'text':>7} {'math':>7} {'s':>4}  |")
print("  " + "-"*94)

for r32, r8 in zip(rows_f32, rows_int8):
    delta = r8["overall"] - r32["overall"]
    marker = f"{delta:+.1%}"
    print(f"  {r32['pid']:>3}  | {r32['overall']:>7.1%} {r32['text']:>7.1%} {r32['math']:>7.1%} {r32['sec']:>3.1f}s"
          f"  | {r8['overall']:>7.1%} {r8['text']:>7.1%} {r8['math']:>7.1%} {r8['sec']:>3.1f}s"
          f"  | {marker}")

print("  " + "-"*94)

def avg(rows, k): return statistics.mean(r[k] for r in rows)
def med(rows, k): return statistics.median(r[k] for r in rows)

print(f"  {'AVG':>3}  | {avg(rows_f32,'overall'):>7.1%} {avg(rows_f32,'text'):>7.1%} {avg(rows_f32,'math'):>7.1%} {avg(rows_f32,'sec'):>3.1f}s"
      f"  | {avg(rows_int8,'overall'):>7.1%} {avg(rows_int8,'text'):>7.1%} {avg(rows_int8,'math'):>7.1%} {avg(rows_int8,'sec'):>3.1f}s"
      f"  | {avg(rows_int8,'overall')-avg(rows_f32,'overall'):+.1%}")
print(f"  {'MED':>3}  | {med(rows_f32,'overall'):>7.1%} {med(rows_f32,'text'):>7.1%} {med(rows_f32,'math'):>7.1%}"
      f"  {'':>4}  | {med(rows_int8,'overall'):>7.1%} {med(rows_int8,'text'):>7.1%} {med(rows_int8,'math'):>7.1%}"
      f"  {'':>4}  |")
print(f"  {'RAM':>3}  | {'avg':>5} {avg(rows_f32,'mb'):>5.0f}MB {'':>11}  | {'avg':>5} {avg(rows_int8,'mb'):>5.0f}MB {'':>11}  |")
print("="*100)

print(f"\n  Model sizes:")
print(f"    float32  {Path(SRC_MODEL).stat().st_size/1024**2:>6.1f} MB")
print(f"    INT8     {Path(INT8_MODEL).stat().st_size/1024**2:>6.1f} MB")
print(f"    Reduction: {(1 - Path(INT8_MODEL).stat().st_size/Path(SRC_MODEL).stat().st_size)*100:.0f}%")
