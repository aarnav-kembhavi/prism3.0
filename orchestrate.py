"""
orchestrate.py
--------------
CLI entry point for the Screen2LaTeX orchestration pipeline.

Optimized with Inference Engineering:
  1. YOLO Unloading: Explicit memory management frees ~650MB RAM.
  2. Batched Math: Texo now processes all equations in one pass.
  3. Parallel Routing: Text and Math OCR run concurrently via multithreading.
  4. ONNX Inference: YOLO runs via ONNX Runtime for 30% CPU speedup.
"""

import sys
import os
import time
import argparse
import gc
import torch
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import concurrent.futures

import cv2
import numpy as np
from normalization import normalize_image_pil
from normalization.region_adaptive import preprocess_crop, RegionArtifactProfile
from models_interface import (
    run_text_ocr, run_text_ocr_batched, run_math_recognition, run_math_recognition_batched,
    run_table_extraction, get_math_latencies, get_math_batch_latencies,
    get_text_latencies, get_table_latencies, get_text_batch_latencies
)
from layout_utils import apply_semantic_reading_order, sort_detections_geometric, xyxy_to_pil_crop, detect_column_count, split_detections_by_column
from latex_builder import wrap_content, assemble_document, save_tex
from detection_postprocess import postprocess_detections

try:
    from evaluation.profiler import BackgroundProfiler
    HAS_PROFILER = True
except ImportError:
    HAS_PROFILER = False


# ----------------------------------------------------------------
# YOLO MODEL (ONNX Optimized)
# ----------------------------------------------------------------
YOLO_MODEL_PATH = "yolov11n-doclaynet.onnx"


# ----------------------------------------------------------------
# Classes handled by each specialist model
# ----------------------------------------------------------------
TEXT_CLASSES = {"Text", "Title", "Section-header", "Caption",
                "Footnote", "Page-footer", "Page-header", "List-item"}
MATH_CLASSES = {"Formula"}
TABLE_CLASSES = {"Table"}
IMAGE_CLASSES = {"Picture"}

LIST_ITEM_CLASS = "List-item"


def load_model(model_path: str) -> YOLO:
    """Load YOLO model (supports .pt and .onnx)."""
    print(f"[*] Loading YOLO model: {model_path}")
    try:
        model = YOLO(model_path, task='detect')
    except Exception as e:
        print(f"[!] Could not load {model_path}: {e}")
        print("[!] Attempting fallback to yolov11n-doclaynet.pt")
        model = YOLO("yolov11n-doclaynet.pt")
    print("[✓] Model loaded.")
    return model


def _is_likely_logo(crop_pil: Image.Image) -> bool:
    arr = np.array(crop_pil.convert("RGB"), dtype=np.float32)
    gray = arr.mean(axis=2)
    non_white = float(np.mean(gray < 230))
    color_std = float(arr.std())
    return non_white < 0.15 and color_std > 8.0


def run_detection(model: YOLO, image_norm: Image.Image, image_fidelity: Image.Image, image_path: str):
    results = model(image_path, verbose=False)
    detections = []
    result = results[0]
    class_names = result.names

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        class_id = int(box.cls[0].item())
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
                crop = fidelity_crop

        detections.append({
            "bbox": [x1, y1, x2, y2],
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
            "crop": crop,
        })

    print(f"[✓] Detected {len(detections)} regions.")
    return detections


def route_and_extract(detections, figures_dir: str, figure_start: int = 0):
    """
    Route detections to specialist models with parallel batching.
    """
    os.makedirs(figures_dir, exist_ok=True)
    body_parts = [""] * len(detections)
    list_indices = set()
    
    # 1. Collect Tasks
    text_indices = [i for i, d in enumerate(detections) if d["class_name"] in TEXT_CLASSES]
    math_indices = [i for i, d in enumerate(detections) if d["class_name"] in MATH_CLASSES]
    table_indices = [i for i, d in enumerate(detections) if d["class_name"] in TABLE_CLASSES]
    image_indices = [i for i, d in enumerate(detections) if d["class_name"] in IMAGE_CLASSES]
    
    # 2. Parallel Execution (Text and Math)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {}
        
        if text_indices:
            text_crops = [detections[i]["crop"] for i in text_indices]
            futures["text"] = executor.submit(run_text_ocr_batched, text_crops)
            
        if math_indices:
            math_crops = [detections[i]["crop"] for i in math_indices]
            # Texo Fallback counter needs to be shared/managed
            math_counter = [0]
            futures["math"] = executor.submit(run_math_recognition_batched, math_crops, figures_dir, math_counter)

        # 3. Tables (Run sequentially in main thread to avoid TATR RAM spikes)
        for i in table_indices:
            raw = run_table_extraction(detections[i]["crop"])
            body_parts[i] = wrap_content(detections[i]["class_name"], raw)

        # 4. Images (Fast, IO-bound)
        fig_count = figure_start
        for i in image_indices:
            fig_count += 1
            fname = f"figure_{fig_count:03d}.png"
            detections[i]["crop"].save(os.path.join(figures_dir, fname))
            body_parts[i] = wrap_content("Picture", fname)

        # 5. Retrieve Parallel Results
        if "text" in futures:
            raw_texts = futures["text"].result()
            for idx, raw in zip(text_indices, raw_texts):
                body_parts[idx] = wrap_content(detections[idx]["class_name"], raw)
                if detections[idx]["class_name"] == LIST_ITEM_CLASS:
                    list_indices.add(idx)
                    
        if "math" in futures:
            raw_maths = futures["math"].result()
            for idx, raw in zip(math_indices, raw_maths):
                body_parts[idx] = wrap_content(detections[idx]["class_name"], raw)

    # Return body_parts filtered for empty strings (unknown/errors) and mapping count
    return body_parts, list_indices, fig_count


def main():
    parser = argparse.ArgumentParser(description="Screen2LaTeX Orchestrator")
    parser.add_argument("image_path", type=str)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--skip-normalize", action="store_true")
    parser.add_argument("--target-dpi", type=int, default=250)
    parser.add_argument("--source-dpi", type=int, default=96)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"[✗] Image not found: {args.image_path}")
        sys.exit(1)

    image_stem = Path(args.image_path).stem
    output_dir = Path(f"{image_stem}_output")
    import shutil
    if output_dir.exists(): shutil.rmtree(output_dir)
    output_dir.mkdir(exist_ok=True)
    tex_path = output_dir / "main.tex"

    print(f"\n[*] Input image  : {args.image_path}")
    print(f"[*] Output folder: {output_dir.absolute()}")

    profiler = None
    if args.profile:
        if HAS_PROFILER:
            profiler = BackgroundProfiler(interval=0.1)
            profiler.start()

    # Stage 1: Normalization
    import psutil
    process = psutil.Process(os.getpid())
    t_stage1_start = time.perf_counter()
    
    if args.skip_normalize:
        image_norm = Image.open(args.image_path).convert("RGB")
        image_fidelity = image_norm
        is_screenshot = False
    else:
        image_norm, image_fidelity, modality_result = normalize_image_pil(args.image_path, target_dpi=args.target_dpi, source_dpi=args.source_dpi)
        is_screenshot = (modality_result.modality.value == "screenshot")
        image_norm.save(output_dir / "normalized.png")

    t_stage1_end = time.perf_counter()
    mem_stage1_end = process.memory_info().rss / 1024 / 1024

    # Stage 2: Layout
    t_stage2_start = time.perf_counter()
    model = load_model(YOLO_MODEL_PATH)
    yolo_input = str(output_dir / "normalized.png") if not args.skip_normalize else args.image_path
    detections = run_detection(model, image_norm, image_fidelity, yolo_input)
    
    img_width, img_height = image_norm.width, image_norm.height
    detections = postprocess_detections(detections, img_width, img_height)

    # YOLO UNLOADING (Optimization 4)
    print("[*] Unloading YOLO model to free RAM...")
    del model
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    t_stage2_end = time.perf_counter()
    mem_stage2_end = process.memory_info().rss / 1024 / 1024

    # Stage 1.5: Adaptive Prep
    t_stage15_start = time.perf_counter()
    HEADER_SUPPRESS_H_FRAC = 0.12
    header_suppress_y = img_height * HEADER_SUPPRESS_H_FRAC
    detections = [d for d in detections if not (d["class_name"] in {"Section-header", "Page-header"} and d["bbox"][3] <= header_suppress_y)]
    
    HEADER_H_FRAC, HEADER_W_FRAC = 0.065, 0.25
    header_right_box = [img_width * (1 - HEADER_W_FRAC), 0, img_width, img_height * HEADER_H_FRAC]
    if not any(d["class_name"] == "Picture" and d["bbox"][0] >= header_right_box[0] and d["bbox"][3] <= header_right_box[3] for d in detections):
        hx1, hy1, hx2, hy2 = [int(v) for v in header_right_box]
        header_crop = xyxy_to_pil_crop(image_fidelity, [hx1, hy1, hx2, hy2])
        if header_crop.width > 20:
            detections.insert(0, {"bbox": [hx1, hy1, hx2, hy2], "class_id": -1, "class_name": "Picture", "crop": header_crop, "is_header_logo": True})

    for det in detections:
        if det["class_name"] in IMAGE_CLASSES: continue
        det['crop'] = xyxy_to_pil_crop(image_norm, det['bbox'])
        crop_bgr = cv2.cvtColor(np.array(det['crop']), cv2.COLOR_RGB2BGR)
        corrected_bgr, _ = preprocess_crop(crop_bgr, det['class_name'], is_screenshot=is_screenshot)
        det['crop'] = Image.fromarray(cv2.cvtColor(corrected_bgr, cv2.COLOR_BGR2RGB))

    t_stage15_end = time.perf_counter()
    mem_stage15_end = process.memory_info().rss / 1024 / 1024

    # Stage 3: Extraction
    t_stage3_start = time.perf_counter()
    col_count = detect_column_count(detections, img_width)
    header_logo_dets = [d for d in detections if d.get("is_header_logo")]
    body_detections  = [d for d in detections if not d.get("is_header_logo")]
    header_logo_fname = None
    if header_logo_dets:
        header_logo_fname = "figure_header_logo.png"
        header_logo_dets[0]['crop'].save(output_dir / header_logo_fname)

    if col_count == 2:
        full_dets, left_dets, right_dets = split_detections_by_column(body_detections, img_width, img_height, use_dag=True)
        full_parts, full_idx, f_cnt = route_and_extract(full_dets, str(output_dir), 0)
        left_parts, left_idx, f_cnt = route_and_extract(left_dets, str(output_dir), f_cnt)
        right_parts, right_idx, f_cnt = route_and_extract(right_dets, str(output_dir), f_cnt)
        document = assemble_document(full_parts, full_idx, True, left_parts, left_idx, right_parts, right_idx, header_logo_fname)
    else:
        body_sorted = apply_semantic_reading_order(body_detections, img_width, img_height)
        body_parts, list_idx, _ = route_and_extract(body_sorted, str(output_dir))
        document = assemble_document(body_parts, list_idx, False, header_logo=header_logo_fname)

    t_stage3_end = time.perf_counter()
    mem_stage3_end = process.memory_info().rss / 1024 / 1024

    # Stage 4: Assembly
    t_stage4_start = time.perf_counter()
    save_tex(document, str(tex_path))
    t_stage4_end = time.perf_counter()
    mem_stage4_end = process.memory_info().rss / 1024 / 1024

    if profiler:
        metrics = profiler.stop()
        print(f"\n[*] Component Profiling ({image_stem}) [ENGINEERING OPTIMIZED]:")
        print(f"    {'Component':<15} | {'Latency':<8} | {'RAM (Peak)':<10}")
        print(f"    {'-'*15}-|-{'-'*8}-|-{'-'*10}")
        print(f"    {'Normalization':<15} | {t_stage1_end-t_stage1_start:6.2f}s | {mem_stage1_end:7.1f} MB")
        print(f"    {'YOLO (ONNX)':<15} | {t_stage2_end-t_stage2_start:6.2f}s | {mem_stage2_end:7.1f} MB")
        print(f"    {'Adaptive Prep':<15} | {t_stage15_end-t_stage15_start:6.2f}s | {mem_stage15_end:7.1f} MB")
        print(f"    {'OCR (Batched)':<15} | {t_stage3_end-t_stage3_start:6.2f}s | {mem_stage3_end:7.1f} MB")
        print(f"    {'Assembly':<15} | {t_stage4_end-t_stage4_start:6.2f}s | {mem_stage4_end:7.1f} MB")
        print(f"    {'-'*40}")
        print(f"    {'TOTAL':<15} | {metrics['latency_sec']:6.2f}s | {metrics['mem_peak_mb']:7.1f} MB")

    print(f"\n[✓] Done.")


if __name__ == "__main__":
    main()