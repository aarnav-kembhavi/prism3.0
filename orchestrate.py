"""
orchestrate.py
--------------
CLI entry point for the Screen2LaTeX orchestration pipeline.

Restored to Initial Methodology (v3.9):
  1. Detailed LaTeX Wrappers (Detailed Titles, resizebox Tables).
  2. Sequential Specialist Routing (Robust Table Handling).
  3. RapidOCR Unified Backend (RAM Safe).
  4. YOLO ONNX + Unloading (Speed Optimized).
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

import cv2
import numpy as np
from normalization import normalize_image_pil
from normalization.region_adaptive import preprocess_crop, RegionArtifactProfile
from models_interface import (
    run_text_ocr_batched, run_math_recognition_batched,
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


YOLO_MODEL_PATH = "yolov11n-doclaynet.onnx"

TEXT_CLASSES = {"Text", "Title", "Section-header", "Caption",
                "Footnote", "Page-footer", "Page-header", "List-item"}
MATH_CLASSES = {"Formula"}
TABLE_CLASSES = {"Table"}
IMAGE_CLASSES = {"Picture"}

LIST_ITEM_CLASS = "List-item"


def load_model(model_path: str) -> YOLO:
    print(f"[*] Loading YOLO model: {model_path}")
    try:
        model = YOLO(model_path, task='detect')
    except Exception as e:
        print(f"[!] Falling back to .pt: {e}")
        model = YOLO("yolov11n-doclaynet.pt")
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
    return detections


def route_and_extract(detections, figures_dir: str, figure_start: int = 0):
    """
    Route detections to specialist models.
    Sequential execution restored to prevent threading-related table failures.
    """
    os.makedirs(figures_dir, exist_ok=True)
    body_parts = []
    list_indices = set()
    figure_counter = figure_start

    # 1. Batch Text and Math for efficiency
    text_indices = [i for i, d in enumerate(detections) if d["class_name"] in TEXT_CLASSES]
    math_indices = [i for i, d in enumerate(detections) if d["class_name"] in MATH_CLASSES]
    
    if text_indices:
        text_crops = [detections[i]["crop"] for i in text_indices]
        raw_texts = run_text_ocr_batched(text_crops)
        for idx, raw in zip(text_indices, raw_texts):
            detections[idx]["raw_content"] = raw
            
    if math_indices:
        math_crops = [detections[i]["crop"] for i in math_indices]
        # Texo Fallback counter needs to be shared/managed
        math_counter = [0]
        raw_maths = run_math_recognition_batched(math_crops, figures_dir, math_counter)
        for idx, raw in zip(math_indices, raw_maths):
            detections[idx]["raw_content"] = raw

    # 2. Sequential Wrap and Table/Picture processing
    for i, det in enumerate(detections):
        class_name = det["class_name"]
        crop = det["crop"]

        if class_name in TEXT_CLASSES or class_name in MATH_CLASSES:
            raw = det.get("raw_content", "")
            wrapped = wrap_content(class_name, raw)
            if class_name == LIST_ITEM_CLASS:
                list_indices.add(len(body_parts))
            body_parts.append(wrapped)

        elif class_name in TABLE_CLASSES:
            print(f"  [table] Extracting table structure...")
            raw = run_table_extraction(crop)
            if raw:
                wrapped = wrap_content(class_name, raw)
                body_parts.append(wrapped)
            else:
                print(f"  [table] WARNING: Extraction returned empty.")

        elif class_name in IMAGE_CLASSES:
            figure_counter += 1
            fname = f"figure_{figure_counter:03d}.png"
            crop.save(os.path.join(figures_dir, fname))
            body_parts.append(wrap_content("Picture", fname))

    return body_parts, list_indices, figure_counter


def main():
    parser = argparse.ArgumentParser(description="Screen2LaTeX Orchestrator")
    parser.add_argument("image_path", type=str)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    image_stem = Path(args.image_path).stem
    output_dir = Path(f"{image_stem}_output")
    if output_dir.exists(): import shutil; shutil.rmtree(output_dir)
    output_dir.mkdir(exist_ok=True)
    tex_path = output_dir / "main.tex"

    profiler = None
    if args.profile and HAS_PROFILER:
        profiler = BackgroundProfiler(interval=0.1); profiler.start()

    # Stage 1: Normalization
    import psutil
    process = psutil.Process(os.getpid())
    t_stage1_start = time.perf_counter()
    
    print("[*] Stage 1: Image Normalization")
    image_norm, image_fidelity, modality_result = normalize_image_pil(args.image_path)
    from normalization.modality import CaptureModality
    is_screenshot = (modality_result.modality == CaptureModality.SCREENSHOT)
    print(f"[*] Modality resolved in orchestrate: {'screenshot' if is_screenshot else 'phone_photo'}")

    # For phone photos: run a whole-image moiré removal pass BEFORE saving
    # normalized.png and before YOLO detection.
    #
    # Why here and not only in region_adaptive per-crop:
    # A fine screen-mesh or crosshatch glare pattern covers the ENTIRE image,
    # including the inter-column gutter.  detect_column_count() uses the YOLO
    # detection layout, but YOLO itself is confused by the mesh filling what
    # should be empty gutter space — it detects fewer boxes in the right column
    # or misses the column boundary entirely, causing detect_column_count() to
    # return 1 and collapsing both columns into one stream.
    # Removing the mesh from the full image before YOLO runs lets the gutter
    # appear as the expected empty vertical stripe, restoring 2-column detection.
    #
    # The per-crop moiré removal in region_adaptive.py is still needed for any
    # residual mesh that survives in individual crops.
    if not is_screenshot:
        from normalization.region_adaptive import detect_moire
        from normalization.frequency_filter import remove_moire as _remove_moire_full
        import cv2 as _cv2
        _norm_bgr = _cv2.cvtColor(np.array(image_norm), _cv2.COLOR_RGB2BGR)
        _moire_hit, _moire_sev = detect_moire(_norm_bgr, is_screenshot=False)
        if _moire_hit:
            print(f"[*] Whole-image moiré detected (severity={_moire_sev:.2f}), removing before YOLO...")
            _clean_bgr = _remove_moire_full(_norm_bgr)
            image_norm = Image.fromarray(_cv2.cvtColor(_clean_bgr, _cv2.COLOR_BGR2RGB))
        del _norm_bgr

    image_norm.save(output_dir / "normalized.png")

    t_stage1_end = time.perf_counter(); mem_stage1_end = process.memory_info().rss / 1024 / 1024

    # Stage 2: Layout
    t_stage2_start = time.perf_counter()
    model = load_model(YOLO_MODEL_PATH)
    yolo_input = str(output_dir / "normalized.png")
    detections = run_detection(model, image_norm, image_fidelity, yolo_input)
    img_width, img_height = image_norm.width, image_norm.height
    detections = postprocess_detections(detections, img_width, img_height)

    # YOLO UNLOADING
    print("[*] Unloading YOLO model to free RAM...")
    del model; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    t_stage2_end = time.perf_counter(); mem_stage2_end = process.memory_info().rss / 1024 / 1024

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
        corrected_bgr, prof = preprocess_crop(crop_bgr, det['class_name'], is_screenshot=is_screenshot)
        # For phone photos: if glare OR moiré was severe, run a final CLAHE
        # pass to restore local contrast in corrected regions before OCR.
        if not is_screenshot and (
            (prof.glare_detected and prof.glare_severity > 0.10) or
            (prof.moire_detected and prof.moire_severity > 0.30)
        ):
            from normalization.frequency_filter import normalize_contrast
            corrected_bgr = normalize_contrast(corrected_bgr)
        det['crop'] = Image.fromarray(cv2.cvtColor(corrected_bgr, cv2.COLOR_BGR2RGB))

    t_stage15_end = time.perf_counter(); mem_stage15_end = process.memory_info().rss / 1024 / 1024

    # Stage 3: Extraction
    print(f"\n[*] Stage 3: Content Extraction")
    t_stage3_start = time.perf_counter()
    col_count = detect_column_count(detections, img_width)
    header_logo_dets = [d for d in detections if d.get("is_header_logo")]
    body_detections  = [d for d in detections if not d.get("is_header_logo")]
    header_logo_fname = "figure_header_logo.png" if header_logo_dets else None
    if header_logo_fname: header_logo_dets[0]['crop'].save(output_dir / header_logo_fname)

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

    t_stage3_end = time.perf_counter(); mem_stage3_end = process.memory_info().rss / 1024 / 1024

    # Stage 4: Assembly
    t_stage4_start = time.perf_counter()
    save_tex(document, str(tex_path))
    t_stage4_end = time.perf_counter(); mem_stage4_end = process.memory_info().rss / 1024 / 1024

    if profiler:
        metrics = profiler.stop()
        print(f"\n[*] Component Profiling ({image_stem}) [RESTORED v3.9]:")
        print(f"    {'Component':<15} | {'Latency':<8} | {'RAM (Peak)':<10}")
        print(f"    {'-'*15}-|-{'-'*8}-|-{'-'*10}")
        print(f"    {'Normalization':<15} | {t_stage1_end-t_stage1_start:6.2f}s | {mem_stage1_end:7.1f} MB")
        print(f"    {'YOLO (ONNX)':<15} | {t_stage2_end-t_stage2_start:6.2f}s | {mem_stage2_end:7.1f} MB")
        print(f"    {'Adaptive Prep':<15} | {t_stage15_end-t_stage15_start:6.2f}s | {mem_stage15_end:7.1f} MB")
        print(f"    {'OCR (Unified)':<15} | {t_stage3_end-t_stage3_start:6.2f}s | {mem_stage3_end:7.1f} MB")
        print(f"    {'Assembly':<15} | {t_stage4_end-t_stage4_start:6.2f}s | {mem_stage4_end:7.1f} MB")
        print(f"    {'-'*40}")
        print(f"    {'TOTAL':<15} | {metrics['latency_sec']:6.2f}s | {metrics['mem_peak_mb']:7.1f} MB")

    print(f"\n[✓] Done.")


if __name__ == "__main__":
    main()