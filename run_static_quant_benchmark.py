"""
run_static_quant_benchmark.py
------------------------------
Static INT8 quantization of yolov11n-doclaynet.onnx using the 26 benchmark
normalized images as calibration data. Then benchmarks float32 vs dynamic INT8
vs static INT8 across 26 pages, comparing EDR, latency, and peak RAM.

Static quantization quantizes both weights AND activation tensors, so the ONNX
Runtime session allocates INT8 buffers for intermediate feature maps instead of
float32 — giving genuine runtime RAM savings (expected ~3-4x reduction for
activation memory, translating to a meaningful drop in the YOLO session RSS).
"""

import csv, gc, os, shutil, statistics, time
from pathlib import Path

import cv2
import numpy as np
import psutil

# ── Calibration data reader ───────────────────────────────────────────────────
from onnxruntime.quantization import (
    quantize_static, CalibrationDataReader,
    QuantType, QuantFormat, CalibrationMethod,
)

CALIB_IMAGES = sorted(
    Path("benchmark_results/quant_comparison/float32_yolo").glob("*/assets/normalized.png")
)

INPUT_NAME  = "images"
INPUT_SHAPE = (640, 640)   # YOLO fixed input size


class YOLOCalibReader(CalibrationDataReader):
    """Feeds letterboxed benchmark images to the YOLO model for calibration."""

    def __init__(self, image_paths):
        self._paths = list(image_paths)
        self._idx   = 0

    def _preprocess(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(path)
        # Letterbox to 640×640
        h, w  = img.shape[:2]
        th, tw = INPUT_SHAPE
        scale  = min(tw / w, th / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas  = np.full((th, tw, 3), 114, dtype=np.uint8)
        top  = (th - nh) // 2
        left = (tw - nw) // 2
        canvas[top:top+nh, left:left+nw] = resized
        # BGR → RGB, HWC → CHW, [0,255] → [0,1]
        rgb  = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        chw  = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        return chw[np.newaxis]   # (1, 3, 640, 640)

    def get_next(self):
        if self._idx >= len(self._paths):
            return None
        data = self._preprocess(self._paths[self._idx])
        self._idx += 1
        return {INPUT_NAME: data}

    def rewind(self):
        self._idx = 0


# ── Quantize ─────────────────────────────────────────────────────────────────
SRC_MODEL      = "yolov11n-doclaynet.onnx"
DYN_INT8_MODEL = "yolov11n-doclaynet-int8.onnx"        # from previous run
STAT_INT8_MODEL = "yolov11n-doclaynet-static-v3.onnx"  # v3: QOperator, Entropy, backbone-only

print(f"[*] Using pre-built static INT8 v3 ({Path(STAT_INT8_MODEL).stat().st_size/1024**2:.1f} MB)")

# ── Pipeline imports ──────────────────────────────────────────────────────────
from normalization import normalize_image_pil
from normalization.modality import CaptureModality
from models_interface import (
    run_text_ocr_batched, run_math_recognition_batched,
    run_table_extraction_batched, get_yolo_model, unload_yolo, unload_texo,
)
from layout_utils import (
    apply_semantic_reading_order, xyxy_to_pil_crop,
    detect_column_count, split_detections_by_column,
)
from latex_builder import wrap_content, assemble_document, save_tex
from detection_postprocess import postprocess_detections
from evaluation.normalizer import normalize_latex, split_math_and_text
from run_full_benchmark import compute_edr

GT_DIR     = Path("pdf2latex_dataset/dataset")
IMAGES_DIR = Path("benchmark_results/temp_images")
OUT_BASE   = Path("benchmark_results/quant_comparison")

TEXT_CLASSES    = {"Text","Title","Section-header","Caption",
                   "Footnote","Page-footer","Page-header","List-item"}
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
            x1 = max(0, x1-FORMULA_PAD); y1 = max(0, y1-FORMULA_PAD)
            x2 = min(img_w, x2+FORMULA_PAD); y2 = min(img_h, y2+FORMULA_PAD)
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
        full_dets, left_dets, right_dets = split_detections_by_column(
            detections, img_w, img_h, use_dag=True)
        def _parts(dets):
            p, li, fc = [], set(), [0]
            for det in dets:
                cn = det["class_name"]
                if cn in TEXT_CLASSES or cn in MATH_CLASSES:
                    w = wrap_content(cn, det.get("raw_content", ""))
                    if cn == LIST_ITEM_CLASS: li.add(len(p))
                    p.append(w)
                elif cn in TABLE_CLASSES:
                    raw = det.get("raw_content", "")
                    if raw: p.append(wrap_content(cn, raw))
                elif cn in IMAGE_CLASSES:
                    fc[0] += 1; fname = f"figure_{fc[0]:03d}.png"
                    det["crop"].save(str(figures_dir/fname))
                    p.append(wrap_content("Picture", fname))
            return p, li
        fp, fi = _parts(full_dets)
        lp, li2 = _parts(left_dets)
        rp, ri  = _parts(right_dets)
        document = assemble_document(fp, fi, True, lp, li2, rp, ri)
    else:
        for det in apply_semantic_reading_order(detections, img_w, img_h):
            cn = det["class_name"]
            if cn in TEXT_CLASSES or cn in MATH_CLASSES:
                w = wrap_content(cn, det.get("raw_content", ""))
                if cn == LIST_ITEM_CLASS: list_idx.add(len(body_parts))
                body_parts.append(w)
            elif cn in TABLE_CLASSES:
                raw = det.get("raw_content", "")
                if raw: body_parts.append(wrap_content(cn, raw))
            elif cn in IMAGE_CLASSES:
                figure_counter += 1; fname = f"figure_{figure_counter:03d}.png"
                det["crop"].save(str(figures_dir/fname))
                body_parts.append(wrap_content("Picture", fname))
        document = assemble_document(body_parts, list_idx, False)

    tex_path = out_dir / "main.tex"
    save_tex(document, str(tex_path))

    elapsed = time.perf_counter() - t0
    peak_mb = process.memory_info().rss / 1024 / 1024

    gt = GT_DIR / f"{img_path.stem}_gt.tex"
    pred_n = normalize_latex(tex_path.read_text(encoding="utf-8", errors="ignore"), remove_spaces=True)
    gt_n   = normalize_latex(gt.read_text(encoding="utf-8", errors="ignore"),        remove_spaces=True)
    pm, pt = split_math_and_text(pred_n)
    gm, gt_ = split_math_and_text(gt_n)
    return {
        "pid":     img_path.stem,
        "overall": compute_edr(pred_n, gt_n),
        "math":    compute_edr(pm, gm),
        "text":    compute_edr(pt, gt_),
        "sec":     elapsed,
        "mb":      peak_mb,
    }


def run_pass(model_path: str, label: str) -> list:
    print(f"\n{'='*70}\n  {label}  ({model_path})\n{'='*70}")
    yolo = get_yolo_model(model_path)
    rows, out_base = [], OUT_BASE / label.replace(" ","_").lower()
    for i, img_path in enumerate(all_imgs, 1):
        print(f"[{i:>2}/26] page {img_path.stem}", flush=True)
        r = run_page(img_path, out_base / img_path.stem, yolo)
        rows.append(r)
        print(f"  {r['sec']:.1f}s  {r['mb']:.0f}MB  "
              f"EDR={r['overall']:.1%}  text={r['text']:.1%}  math={r['math']:.1%}")
        gc.collect()
    unload_texo()
    unload_yolo()
    gc.collect()
    return rows


# ── Load float32 results from previous run ───────────────────────────────────
def load_csv(path: Path) -> list:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "pid": r["pid"],
                "overall": float(r["overall"]),
                "math":    float(r["math"]),
                "text":    float(r["text"]),
                "sec":     float(r["sec"]),
                "mb":      float(r["mb"]),
            })
    return rows

f32_csv = OUT_BASE / "float32_results.csv"
if f32_csv.exists():
    print(f"\n[*] Reusing float32 results from previous run ({f32_csv})")
    rows_f32 = load_csv(f32_csv)
else:
    rows_f32 = run_pass(SRC_MODEL, "float32 YOLO")
    with open(f32_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pid","overall","math","text","sec","mb"])
        w.writeheader(); w.writerows({k: r[k] for k in ["pid","overall","math","text","sec","mb"]} for r in rows_f32)

# ── Run static INT8 pass ──────────────────────────────────────────────────────
rows_static = run_pass(STAT_INT8_MODEL, "static INT8 YOLO")

stat_csv = OUT_BASE / "static_int8_v3_results.csv"
with open(stat_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["pid","overall","math","text","sec","mb"])
    w.writeheader(); w.writerows({k: r[k] for k in ["pid","overall","math","text","sec","mb"]} for r in rows_static)

# ── Load dynamic INT8 for 3-way comparison ───────────────────────────────────
dyn_csv = OUT_BASE / "int8_results.csv"
rows_dyn = load_csv(dyn_csv) if dyn_csv.exists() else []

# ── 3-way comparison table ────────────────────────────────────────────────────
def avg(rows, k): return statistics.mean(r[k] for r in rows)
def med(rows, k): return statistics.median(r[k] for r in rows)

print("\n" + "="*110)
print("  YOLO Quantization 3-Way Comparison  --  26 PDF2LaTeX pages")
print("="*110)

if rows_dyn:
    hdr_cols = "  {'Pg':>3}  | {'f32 EDR':>8} {'s':>4}  | {'dyn8 EDR':>9} {'s':>4}  | {'stat8 EDR':>10} {'s':>4} {'MB':>6}  | d(stat-f32)"
    print(f"  {'Pg':>3}  | {'f32 EDR':>8} {'s':>4}  | {'dyn8 EDR':>9} {'s':>4}  | {'stat8 EDR':>10} {'s':>4} {'MB':>6}  | d(stat-f32)")
    print("  " + "-"*104)
    for rf, rd, rs in zip(rows_f32, rows_dyn, rows_static):
        delta = rs["overall"] - rf["overall"]
        print(f"  {rf['pid']:>3}  | {rf['overall']:>8.1%} {rf['sec']:>3.1f}s  |"
              f" {rd['overall']:>9.1%} {rd['sec']:>3.1f}s  |"
              f" {rs['overall']:>10.1%} {rs['sec']:>3.1f}s {rs['mb']:>5.0f}MB  | {delta:+.1%}")
else:
    print(f"  {'Pg':>3}  | {'f32 EDR':>8} {'s':>4} {'MB':>6}  | {'stat8 EDR':>10} {'s':>4} {'MB':>6}  | delta")
    print("  " + "-"*70)
    for rf, rs in zip(rows_f32, rows_static):
        delta = rs["overall"] - rf["overall"]
        print(f"  {rf['pid']:>3}  | {rf['overall']:>8.1%} {rf['sec']:>3.1f}s {rf['mb']:>5.0f}MB  |"
              f" {rs['overall']:>10.1%} {rs['sec']:>3.1f}s {rs['mb']:>5.0f}MB  | {delta:+.1%}")

print("  " + "-"*70)
print(f"  {'AVG':>3}  | {avg(rows_f32,'overall'):>8.1%} {avg(rows_f32,'sec'):>3.1f}s {avg(rows_f32,'mb'):>5.0f}MB  |"
      f" {avg(rows_static,'overall'):>10.1%} {avg(rows_static,'sec'):>3.1f}s {avg(rows_static,'mb'):>5.0f}MB"
      f"  | {avg(rows_static,'overall')-avg(rows_f32,'overall'):+.1%}")
print(f"  {'MED':>3}  | {med(rows_f32,'overall'):>8.1%}{'':>5}{'':>7}  |"
      f" {med(rows_static,'overall'):>10.1%}{'':>5}{'':>7}  |"
      f" {med(rows_static,'overall')-med(rows_f32,'overall'):+.1%}")
print("="*110)

print(f"\n  Model file sizes:")
for label, path in [("float32", SRC_MODEL), ("dynamic INT8", DYN_INT8_MODEL), ("static INT8", STAT_INT8_MODEL)]:
    p = Path(path)
    if p.exists():
        print(f"    {label:<14}  {p.stat().st_size/1024**2:>6.1f} MB")
