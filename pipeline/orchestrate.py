"""
orchestrate.py
--------------
CLI entry point for the Screen2LaTeX orchestration pipeline.

v3.11 performance optimizations:
  1. Background worker startup — workers load in a thread while YOLO runs,
     overlapping ~3s Texo load with normalization+YOLO inference.
  2. Parallel math+text dispatch — math and text workers run concurrently
     via ThreadPoolExecutor, cutting per-page extraction by up to 1s.
  3. Multi-image / daemon mode — pass multiple images as positional args;
     workers start once and stay alive for all images (saves 3.3s per image
     after the first).
"""

import sys
import os

os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')

# UTF-8 stdout/stderr — the pipeline prints Unicode (→, CJK) and the main
# process is now torch-free, so nothing else reconfigures the Windows console.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from pathlib import Path

# Add repo root to path so pipeline.* and normalization are importable
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_pypath = os.environ.get('PYTHONPATH', '')
if _ROOT not in _pypath.split(os.pathsep):
    os.environ['PYTHONPATH'] = _ROOT + (os.pathsep + _pypath if _pypath else '')

# Cap OpenMP/BLAS threads before ultralytics/torch import so the main-process
# YOLO + DocLayout don't grab every core while worker subprocesses run.
from pipeline.onnx_config import apply_thread_env
apply_thread_env()

import time
import argparse
import gc
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

from normalization import normalize_image_pil, normalize_image_pil_skip_stage1
from pipeline.models_interface import (
    run_text_ocr_batched, run_math_recognition_batched,
    run_page_got,
    run_table_extraction, run_table_extraction_batched, get_yolo_model,
    unload_yolo, unload_texo, unload_got, unload_rapidocr,
    get_math_latencies, get_math_batch_latencies,
    get_text_latencies, get_table_latencies, get_text_batch_latencies,
)
from pipeline.text_worker import TextOCRWorker
from pipeline.math_worker_onnx import MathOCRWorkerOnnx as MathOCRWorker
from pipeline.tatr_worker_onnx import TATROnnxWorker

# Subprocess workers — populated by main(); None means in-process fallback.
_ocr_worker:  "TextOCRWorker | None" = None
_math_worker: "MathOCRWorker | None" = None
_tatr_worker: "TATROnnxWorker | None" = None
from pipeline.layout_utils import xyxy_to_pil_crop, detect_column_count
from pipeline.latex_builder import save_tex
from pipeline.detection_postprocess import postprocess_detections
from pipeline.page_core import Workers, build_document

try:
    from evaluation.profiler import BackgroundProfiler
    HAS_PROFILER = True
except ImportError:
    HAS_PROFILER = False


YOLO_MODEL_PATH      = str(Path(__file__).resolve().parent.parent / 'weights' / 'yolov11n-doclaynet.onnx')
DOCLAYOUT_MODEL_PATH = str(Path(__file__).resolve().parent.parent / 'models' / 'doclayout_yolo_docstructbench_imgsz1024.onnx')

# Backend switch: raw onnxruntime detectors (no torch) by default; set
# PRISM_RAW_YOLO=0 to fall back to ultralytics.
_USE_RAW_YOLO = os.environ.get('PRISM_RAW_YOLO', '1') != '0'

_doclayout_model = None

def _get_doclayout_model():
    global _doclayout_model
    if _doclayout_model is None:
        from ultralytics import YOLO as _YOLO
        _doclayout_model = _YOLO(DOCLAYOUT_MODEL_PATH, task='detect')
    return _doclayout_model


def _doclayout_detect(norm_path, conf=0.15):
    """Uniform DocLayout detection → list of {bbox, class_name, confidence}."""
    if _USE_RAW_YOLO:
        from pipeline.models_interface import get_doclayout_detector
        return get_doclayout_detector(DOCLAYOUT_MODEL_PATH, imgsz=1024).detect(norm_path, conf=conf)
    r = _get_doclayout_model()(norm_path, conf=conf, verbose=False)[0]
    return [{'bbox': b.xyxy[0].tolist(), 'class_name': r.names[int(b.cls[0])],
             'confidence': float(b.conf[0])} for b in r.boxes]


def _iou(a, b):
    ix1=max(a[0],b[0]); iy1=max(a[1],b[1]); ix2=min(a[2],b[2]); iy2=min(a[3],b[3])
    iw=max(0,ix2-ix1); ih=max(0,iy2-iy1); inter=iw*ih
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0.0

TEXT_CLASSES   = {"Text", "Title", "Section-header", "Caption",
                  "Footnote", "Page-footer", "Page-header", "List-item"}
MATH_CLASSES   = {"Formula"}
TABLE_CLASSES  = {"Table"}
IMAGE_CLASSES  = {"Picture"}
LIST_ITEM_CLASS = "List-item"



def _is_likely_logo(crop_pil: Image.Image) -> bool:
    arr      = np.array(crop_pil.convert("RGB"), dtype=np.float32)
    gray     = arr.mean(axis=2)
    non_white = float(np.mean(gray < 230))
    color_std = float(arr.std())
    return non_white < 0.15 and color_std > 8.0


def run_detection(model, image_norm: Image.Image, image_fidelity: Image.Image, image_path: str):
    if _USE_RAW_YOLO:
        from pipeline.models_interface import get_yolo_detector
        raw = get_yolo_detector(YOLO_MODEL_PATH, imgsz=640).detect(image_path, conf=0.25, iou=0.7)
    else:
        result = model(image_path, verbose=False)[0]
        raw = [{'bbox': b.xyxy[0].tolist(), 'class_name': result.names[int(b.cls[0])],
                'confidence': float(b.conf[0])} for b in result.boxes]

    detections = []
    for d in raw:
        x1, y1, x2, y2 = d['bbox']
        class_name = d['class_name']
        confidence = d['confidence']

        if class_name in IMAGE_CLASSES:
            crop = xyxy_to_pil_crop(image_fidelity, [x1, y1, x2, y2])
        else:
            crop = xyxy_to_pil_crop(image_norm, [x1, y1, x2, y2])

        if class_name == "Page-header":
            fidelity_crop = xyxy_to_pil_crop(image_fidelity, [x1, y1, x2, y2])
            if _is_likely_logo(fidelity_crop):
                class_name = "Picture"
                crop       = fidelity_crop

        detections.append({
            "bbox": [x1, y1, x2, y2],
            "class_id": 0,
            "class_name": class_name,
            "confidence": confidence,
            "crop": crop,
        })
    return detections


class _InProcessOCR:
    """Adapter exposing the OCR-worker interface backed by in-process RapidOCR
    (used only for --no-ocr-worker; no CJK/mixed distinction in-process)."""
    def run_text_batch(self, crops, is_screenshot=False):
        return run_text_ocr_batched(crops, is_screenshot=is_screenshot)
    run_text_batch_cjk   = run_text_batch
    run_text_batch_mixed = run_text_batch
    def run_table_batch(self, crops):
        return run_table_extraction_batched(crops)
    def run_table_tokens_batch(self, crops):
        # No token export in-process; empty forces the OCR-worker heuristic path.
        return [[] for _ in crops]


class _InProcessMath:
    """Adapter exposing MathOCRWorker.run_math_batch via in-process Texo."""
    def run_math_batch(self, crops, figures_dir, counter):
        c = [counter]
        res = run_math_recognition_batched(crops, figures_dir, c)
        unload_texo()
        return res, c[0]


def _build_workers_bundle() -> Workers:
    """Bundle the active workers for page_core. Falls back to in-process
    adapters when subprocess workers weren't launched (--no-ocr-worker)."""
    if _ocr_worker is not None:
        return Workers(ocr=_ocr_worker, math=_math_worker, tatr=_tatr_worker)
    return Workers(ocr=_InProcessOCR(), math=_InProcessMath(), tatr=None)


def _launch_workers():
    """Start all subprocess workers (called in a background thread)."""
    global _ocr_worker, _math_worker, _tatr_worker
    _ocr_worker = TextOCRWorker()
    _ocr_worker.start()
    _math_worker = MathOCRWorker()
    _math_worker.start()
    _tatr_worker = TATROnnxWorker()
    _tatr_worker.start()


def main():
    global _ocr_worker, _math_worker
    parser = argparse.ArgumentParser(description="Screen2LaTeX Orchestrator")
    parser.add_argument("image_paths", type=str, nargs="+",
                        help="One or more images to process (workers shared across all)")
    parser.add_argument("--profile",          action="store_true")
    parser.add_argument("--high-quality",     action="store_true",
                        help="Use GOT-OCR2 for full-page LaTeX (slower, higher quality)")
    parser.add_argument("--no-ocr-worker",    action="store_true",
                        help="Run RapidOCR in-process instead of a subprocess worker")
    parser.add_argument("--skip-stage1",      action="store_true",
                        help="Ablation: skip Stage 1 corrections (deskew+modality only, no CLAHE/shadow/glare/moire/resize)")
    args = parser.parse_args()

    # Optimization 1: start workers in a background thread so Texo loads
    # concurrently with normalization and YOLO inference (~3s saved per run).
    _worker_thread = None
    if not args.no_ocr_worker and not args.high_quality:
        _worker_thread = threading.Thread(target=_launch_workers, daemon=True)
        _worker_thread.start()

    # Process each image with the shared worker set.
    for image_path_str in args.image_paths:
        _process_one(image_path_str, args, _worker_thread)
        _worker_thread = None  # already joined on first image; workers stay alive

    if _ocr_worker is not None:
        _ocr_worker.stop()
    if _math_worker is not None:
        _math_worker.stop()
    if _tatr_worker is not None:
        _tatr_worker.stop()


def _process_one(image_path_str: str, args, worker_thread):
    """Run the full pipeline on a single image, joining worker thread if needed."""
    image_stem = Path(image_path_str).stem
    output_dir = Path(_ROOT) / "outputs" / f"{image_stem}_output"
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)

    assets_dir  = output_dir / "assets"
    figures_dir = assets_dir / "figures"
    logs_dir    = output_dir / "logs"
    for d in [output_dir, assets_dir, figures_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    tex_path = output_dir / "main.tex"

    profiler = None
    if args.profile and HAS_PROFILER:
        profiler = BackgroundProfiler(interval=0.1)
        profiler.start()

    import psutil
    process = psutil.Process(os.getpid())

    # ── High-quality mode: GOT-OCR2 full-page ───────────────────
    if args.high_quality:
        print("[*] High-quality mode: GOT-OCR2 full-page OCR")
        t0 = time.perf_counter()
        image_norm, _, _ = normalize_image_pil(image_path_str)
        norm_path = str(assets_dir / "normalized.png")
        image_norm.save(norm_path)
        del image_norm

        raw_latex = run_page_got(norm_path)
        unload_got()

        if not raw_latex.strip().startswith("\\documentclass"):
            from pipeline.latex_builder import assemble_document
            document = assemble_document([raw_latex], set(), False)
        else:
            document = raw_latex

        save_tex(document, str(tex_path))
        t_total = time.perf_counter() - t0
        peak_mb = process.memory_info().rss / 1024 / 1024
        if args.profile:
            print(f"\n    TOTAL           | {t_total:6.2f}s | {peak_mb:7.1f} MB")
        print("\n[OK] Done (high-quality mode).")
        return

    # ── Stage 1: Normalization ───────────────────────────────────
    t_stage1_start = time.perf_counter()
    if args.skip_stage1:
        print("[*] Stage 1: SKIPPED (deskew+modality only)")
        image_norm, image_fidelity, modality_result = normalize_image_pil_skip_stage1(image_path_str)
    else:
        print("[*] Stage 1: Image Normalization")
        image_norm, image_fidelity, modality_result = normalize_image_pil(image_path_str)
    from normalization.modality import CaptureModality
    is_screenshot = (modality_result.modality == CaptureModality.SCREENSHOT)
    print(f"[*] Modality: {'screenshot' if is_screenshot else 'phone_photo'}")
    image_norm.save(assets_dir / "normalized.png")
    t_stage1_end  = time.perf_counter()
    mem_stage1_end = process.memory_info().rss / 1024 / 1024

    # ── Stage 2: Layout Detection ────────────────────────────────
    t_stage2_start = time.perf_counter()
    # Raw-ONNX detector needs no torch/ultralytics model object; pass None.
    model      = None if _USE_RAW_YOLO else get_yolo_model(YOLO_MODEL_PATH)
    yolo_input = str(assets_dir / "normalized.png")
    detections = run_detection(model, image_norm, image_fidelity, yolo_input)
    img_width, img_height = image_norm.width, image_norm.height
    detections = postprocess_detections(detections, img_width, img_height)

    # DocLayout YOLO boost: supplement nano YOLO for formulas AND tables.
    if Path(DOCLAYOUT_MODEL_PATH).exists():
        try:
            existing_fml = [d['bbox'] for d in detections if d['class_name'] == 'Formula']
            existing_tbl = [d['bbox'] for d in detections if d['class_name'] == 'Table']
            n_fml = n_tbl = 0
            for box in _doclayout_detect(yolo_input, conf=0.15):
                    cls   = box['class_name']
                    conf_ = box['confidence']
                    bbox  = box['bbox']
                    if cls == 'isolate_formula':
                        # 0.20 floor: drop the lowest-confidence formula boxes
                        # that overlap text and feed Texo garbage.
                        if conf_ < 0.20:
                            continue
                        if any(_iou(bbox, ef) > 0.4 for ef in existing_fml):
                            continue
                        detections.append({
                            'bbox': bbox, 'class_id': -2,
                            'class_name': 'Formula', 'confidence': conf_,
                        })
                        existing_fml.append(bbox); n_fml += 1
                    elif cls == 'table' and conf_ >= 0.30:
                        if any(_iou(bbox, et) > 0.4 for et in existing_tbl):
                            continue
                        detections.append({
                            'bbox': bbox, 'class_id': -3,
                            'class_name': 'Table', 'confidence': conf_,
                        })
                        existing_tbl.append(bbox); n_tbl += 1
            if n_fml or n_tbl:
                print(f'  [DL] +{n_fml} formula(s), +{n_tbl} table(s)')
        except Exception as _e:
            print(f'  [DL] skipped: {_e}')

    t_stage2_end  = time.perf_counter()
    mem_stage2_end = process.memory_info().rss / 1024 / 1024

    # ── Stage 1.5: Header suppression + bbox re-crop ─────────────
    t_stage15_start = time.perf_counter()

    HEADER_SUPPRESS_H_FRAC = 0.12
    header_suppress_y = img_height * HEADER_SUPPRESS_H_FRAC
    detections = [
        d for d in detections
        if not (
            d["class_name"] in {"Section-header", "Page-header"}
            and d["bbox"][3] <= header_suppress_y
        )
    ]

    HEADER_H_FRAC, HEADER_W_FRAC = 0.065, 0.25
    header_right_box = [
        img_width * (1 - HEADER_W_FRAC), 0,
        img_width, img_height * HEADER_H_FRAC,
    ]
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

    # Re-crop all regions using final refined bboxes.
    # 4px is optimal per hyperparameter sweep on 224 EN formula crops;
    # larger padding bleeds neighboring content into the crop and hurts accuracy.
    FORMULA_PAD = 4
    for det in detections:
        bbox = det['bbox']
        if det["class_name"] in MATH_CLASSES:
            x1, y1, x2, y2 = bbox
            bbox = [
                max(0,          x1 - FORMULA_PAD),
                max(0,          y1 - FORMULA_PAD),
                min(img_width,  x2 + FORMULA_PAD),
                min(img_height, y2 + FORMULA_PAD),
            ]
        if det["class_name"] in IMAGE_CLASSES:
            det['crop'] = xyxy_to_pil_crop(image_fidelity, bbox)
        else:
            det['crop'] = xyxy_to_pil_crop(image_norm, bbox)

    # Free full-resolution images — all crops are now extracted into det['crop']
    del image_norm, image_fidelity
    gc.collect()
    # Change A: YOLO not needed again for this image; free its session (~520 MB)
    unload_yolo()

    t_stage15_end  = time.perf_counter()
    mem_stage15_end = process.memory_info().rss / 1024 / 1024

    # ── Stage 3: Content Extraction ──────────────────────────────
    # Optimization 1: join the background worker thread before we need workers.
    # By now normalization + YOLO + prep have run (~3s), Texo should be loaded.
    if worker_thread is not None:
        worker_thread.join()

    print("\n[*] Stage 3: Content Extraction")
    t_stage3_start = time.perf_counter()

    col_count       = detect_column_count(detections, img_width)
    header_logo_dets = [d for d in detections if d.get("is_header_logo")]
    body_detections  = [d for d in detections if not d.get("is_header_logo")]
    header_logo_fname = "assets/figure_header_logo.png" if header_logo_dets else None
    if header_logo_fname:
        header_logo_dets[0]['crop'].save(output_dir / header_logo_fname)

    # Language detection: run CJK engine on sample crops and count real CJK
    # Unicode codepoints. English OCR returns garbage ASCII on Chinese text so
    # char-count alone is unreliable; codepoint presence is the right signal.
    is_cjk = is_mixed = False
    if _ocr_worker is not None:
        sample_crops = [
            d["crop"] for d in body_detections
            if d["class_name"] in TEXT_CLASSES
        ][:4]
        if sample_crops:
            cjk_chars = _ocr_worker.run_cjk_probe(sample_crops, is_screenshot)
            if cjk_chars > 0:
                en_chars = _ocr_worker.run_language_probe(sample_crops, is_screenshot)
                if en_chars < 10 or cjk_chars > en_chars:
                    is_cjk = True
                    print(f"  [lang] CJK page detected ({cjk_chars} CJK chars) → PP-OCRv4 CJK engine")
                else:
                    is_mixed = True
                    print(f"  [lang] Mixed page detected ({cjk_chars} CJK, {en_chars} EN) → dual-engine")

    if col_count >= 3:
        print(f"    [layout] N-column layout detected: {col_count} columns")

    # Shared extraction + assembly (same core as the benchmark, page_core.py).
    workers = _build_workers_bundle()
    document = build_document(
        body_detections, img_width, img_height, workers, str(figures_dir),
        is_screenshot=is_screenshot, is_cjk=is_cjk, is_mixed=is_mixed,
        header_logo_fname=header_logo_fname,
    )

    if _math_worker is None:
        unload_texo()

    t_stage3_end  = time.perf_counter()
    mem_stage3_end = process.memory_info().rss / 1024 / 1024

    # ── Stage 4: Assembly ────────────────────────────────────────
    t_stage4_start = time.perf_counter()
    save_tex(document, str(tex_path))
    t_stage4_end  = time.perf_counter()
    mem_stage4_end = process.memory_info().rss / 1024 / 1024

    if profiler:
        metrics = profiler.stop()

        # Wall-clock extraction time (includes parallel math+text IPC roundtrip)
        t_extraction = t_stage3_end - t_stage3_start

        summary = [
            f"\n[*] Component Profiling ({image_stem}):",
            f"    {'Component':<15} | {'Latency':<8} | {'RAM (Peak)':<10}",
            f"    {'-'*15}-|-{'-'*8}-|-{'-'*10}",
            f"    {'Normalization':<15} | {t_stage1_end  - t_stage1_start:6.2f}s | {mem_stage1_end:7.1f} MB",
            f"    {'YOLO (ONNX)':<15} | {t_stage2_end  - t_stage2_start:6.2f}s | {mem_stage2_end:7.1f} MB",
            f"    {'Adaptive Prep':<15} | {t_stage15_end - t_stage15_start:6.2f}s | {mem_stage15_end:7.1f} MB",
            f"    {'Extraction':<15} | {t_extraction:6.2f}s | {mem_stage3_end:7.1f} MB",
            f"    {'Assembly':<15} | {t_stage4_end  - t_stage4_start:6.2f}s | {mem_stage4_end:7.1f} MB",
            f"    {'-'*40}",
            f"    {'TOTAL':<15} | {metrics['latency_sec']:6.2f}s | {metrics['mem_peak_mb']:7.1f} MB",
        ]
        for line in summary:
            print(line)
        with open(logs_dir / "profiling.txt", "w") as f:
            f.write("\n".join(summary))

    print(f"\n[OK] Done — {image_stem}.")


if __name__ == "__main__":
    main()
