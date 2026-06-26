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

from pathlib import Path

# Add repo root to path so pipeline.* and normalization are importable
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_pypath = os.environ.get('PYTHONPATH', '')
if _ROOT not in _pypath.split(os.pathsep):
    os.environ['PYTHONPATH'] = _ROOT + (os.pathsep + _pypath if _pypath else '')

import time
import argparse
import gc
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

from normalization import normalize_image_pil
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

# Subprocess workers — populated by main(); None means in-process fallback.
_ocr_worker:  "TextOCRWorker | None" = None
_math_worker: "MathOCRWorker | None" = None
from pipeline.layout_utils import (
    apply_semantic_reading_order, sort_detections_geometric,
    xyxy_to_pil_crop, detect_column_count, split_detections_by_column,
    split_detections_n_columns,
)
from pipeline.latex_builder import wrap_content, assemble_document, save_tex
from pipeline.detection_postprocess import postprocess_detections

try:
    from evaluation.profiler import BackgroundProfiler
    HAS_PROFILER = True
except ImportError:
    HAS_PROFILER = False


YOLO_MODEL_PATH      = str(Path(__file__).resolve().parent.parent / 'weights' / 'yolov11n-doclaynet.onnx')
DOCLAYOUT_MODEL_PATH = str(Path(__file__).resolve().parent.parent / 'models' / 'doclayout_yolo_docstructbench_imgsz1024.onnx')

_doclayout_model = None

def _get_doclayout_model():
    global _doclayout_model
    if _doclayout_model is None:
        from ultralytics import YOLO as _YOLO
        _doclayout_model = _YOLO(DOCLAYOUT_MODEL_PATH, task='detect')
    return _doclayout_model


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


def _adjust_figure_paths(parts: list[str]) -> list[str]:
    """Prefix bare figure_NNN filenames with the assets/figures/ subdirectory."""
    return [
        p.replace("{figure_", "{assets/figures/figure_")
        if "includegraphics" in p else p
        for p in parts
    ]


def run_detection(model, image_norm: Image.Image, image_fidelity: Image.Image, image_path: str):
    results    = model(image_path, verbose=False)
    detections = []
    result     = results[0]
    class_names = result.names

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        class_id   = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        class_name = class_names[class_id]

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
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
            "crop": crop,
        })
    return detections


def route_and_extract(
    detections,
    figures_dir: str,
    figure_start: int = 0,
    is_screenshot: bool = False,
    math_start: int = 0,
):
    """
    Route detections to specialist models and return wrapped LaTeX parts.
    Returns (body_parts, list_indices, figure_counter, math_counter).
    """
    os.makedirs(figures_dir, exist_ok=True)
    body_parts     = []
    list_indices   = set()
    figure_counter = figure_start
    math_counter   = [math_start]

    text_indices  = [i for i, d in enumerate(detections) if d["class_name"] in TEXT_CLASSES]
    math_indices  = [i for i, d in enumerate(detections) if d["class_name"] in MATH_CLASSES]
    table_indices = [i for i, d in enumerate(detections) if d["class_name"] in TABLE_CLASSES]

    # Optimization 2: dispatch math and text to their workers in parallel.
    # Each worker has its own independent Pipe connection so concurrent calls
    # are safe. Table must wait for the text worker to be free first.
    use_workers = (_math_worker is not None and _ocr_worker is not None)

    if use_workers and math_indices and text_indices:
        math_crops = [detections[i]["crop"] for i in math_indices]
        text_crops = [detections[i]["crop"] for i in text_indices]
        with ThreadPoolExecutor(max_workers=2) as exe:
            math_fut = exe.submit(
                _math_worker.run_math_batch,
                math_crops, figures_dir, math_counter[0],
            )
            text_fut = exe.submit(
                _ocr_worker.run_text_batch, text_crops, is_screenshot,
            )
            math_results, math_counter[0] = math_fut.result()
            texts = text_fut.result()
        for idx, raw in zip(math_indices, math_results):
            detections[idx]["raw_content"] = raw
        for idx, txt in zip(text_indices, texts):
            detections[idx]["raw_content"] = txt
    else:
        if math_indices:
            crops = [detections[i]["crop"] for i in math_indices]
            if _math_worker is not None:
                results, math_counter[0] = _math_worker.run_math_batch(
                    crops, figures_dir, math_counter[0]
                )
            else:
                results = run_math_recognition_batched(crops, figures_dir, math_counter)
                unload_texo()
            for idx, raw in zip(math_indices, results):
                detections[idx]["raw_content"] = raw

        if text_indices:
            crops = [detections[i]["crop"] for i in text_indices]
            if _ocr_worker is not None:
                texts = _ocr_worker.run_text_batch(crops, is_screenshot=is_screenshot)
            else:
                texts = run_text_ocr_batched(crops, is_screenshot=is_screenshot)
            for idx, txt in zip(text_indices, texts):
                detections[idx]["raw_content"] = txt

    if table_indices:
        table_crops = [detections[i]["crop"] for i in table_indices]
        if _ocr_worker is not None:
            table_results = _ocr_worker.run_table_batch(table_crops)
        else:
            table_results = run_table_extraction_batched(table_crops)
        for idx, raw in zip(table_indices, table_results):
            detections[idx]["raw_content"] = raw

    # Change E: when not using subprocess, free RapidOCR sessions after all OCR
    if _ocr_worker is None:
        unload_rapidocr()

    for i, det in enumerate(detections):
        class_name = det["class_name"]
        crop       = det["crop"]

        if class_name in TEXT_CLASSES or class_name in MATH_CLASSES:
            raw     = det.get("raw_content", "")
            wrapped = wrap_content(class_name, raw)
            if class_name == LIST_ITEM_CLASS:
                list_indices.add(len(body_parts))
            body_parts.append(wrapped)

        elif class_name in TABLE_CLASSES:
            raw = det.get("raw_content", "")
            if raw:
                body_parts.append(wrap_content(class_name, raw))
            else:
                print("  [table] WARNING: Extraction returned empty.")

        elif class_name in IMAGE_CLASSES:
            figure_counter += 1
            fname = f"figure_{figure_counter:03d}.png"
            crop.save(os.path.join(figures_dir, fname))
            body_parts.append(wrap_content("Picture", fname))

    return body_parts, list_indices, figure_counter, math_counter[0]


def _launch_workers():
    """Start both subprocess workers (called in a background thread)."""
    global _ocr_worker, _math_worker
    _ocr_worker = TextOCRWorker()
    _ocr_worker.start()
    _math_worker = MathOCRWorker()
    _math_worker.start()


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


def _process_one(image_path_str: str, args, worker_thread):
    """Run the full pipeline on a single image, joining worker thread if needed."""
    image_stem = Path(image_path_str).stem
    output_dir = Path(f"{image_stem}_output")
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
    model      = get_yolo_model(YOLO_MODEL_PATH)
    yolo_input = str(assets_dir / "normalized.png")
    detections = run_detection(model, image_norm, image_fidelity, yolo_input)
    img_width, img_height = image_norm.width, image_norm.height
    detections = postprocess_detections(detections, img_width, img_height)

    # DocLayout YOLO boost: supplement nano YOLO for formulas AND tables.
    if Path(DOCLAYOUT_MODEL_PATH).exists():
        try:
            dl_model = _get_doclayout_model()
            dl_res = dl_model(yolo_input, conf=0.15, verbose=False)
            existing_fml = [d['bbox'] for d in detections if d['class_name'] == 'Formula']
            existing_tbl = [d['bbox'] for d in detections if d['class_name'] == 'Table']
            n_fml = n_tbl = 0
            for r in dl_res:
                for box in r.boxes:
                    cls   = r.names[int(box.cls[0])]
                    conf_ = float(box.conf[0])
                    bbox  = box.xyxy[0].tolist()
                    if cls == 'isolate_formula':
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
    # Formula bboxes get a 12px expansion so crop_margin() has headroom to
    # detect ink near the edge (YOLO tight-fits the formula region).
    FORMULA_PAD = 12
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

    if col_count == 2:
        full_dets, left_dets, right_dets = split_detections_by_column(
            body_detections, img_width, img_height, use_dag=True
        )
        full_parts,  full_idx,  f_cnt, m_cnt = route_and_extract(
            full_dets,  str(figures_dir), 0,     is_screenshot=is_screenshot, math_start=0
        )
        left_parts,  left_idx,  f_cnt, m_cnt = route_and_extract(
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
    elif col_count >= 3:
        print(f"    [layout] N-column layout detected: {col_count} columns")
        full_dets, col_lists = split_detections_n_columns(
            body_detections, img_width, img_height, use_dag=True
        )
        all_parts: list = []
        all_list_idx: set = set()
        offset = 0
        f_cnt, m_cnt = 0, 0
        for group in [full_dets] + col_lists:
            parts, list_idx, f_cnt, m_cnt = route_and_extract(
                group, str(figures_dir), f_cnt, is_screenshot=is_screenshot, math_start=m_cnt
            )
            parts = _adjust_figure_paths(parts)
            all_parts.extend(parts)
            all_list_idx.update(i + offset for i in list_idx)
            offset += len(parts)
        document = assemble_document(all_parts, all_list_idx, False, header_logo=header_logo_fname)
    else:
        body_sorted = apply_semantic_reading_order(body_detections, img_width, img_height)
        body_parts, list_idx, _, _ = route_and_extract(
            body_sorted, str(figures_dir), is_screenshot=is_screenshot
        )
        body_parts = _adjust_figure_paths(body_parts)
        document   = assemble_document(body_parts, list_idx, False, header_logo=header_logo_fname)

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
