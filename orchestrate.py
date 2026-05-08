"""
orchestrate.py
--------------
CLI entry point for the Screen2LaTeX orchestration pipeline.

Usage:
    python orchestrate.py path/to/image.png
    python orchestrate.py path/to/image.jpg
    python orchestrate.py path/to/image.png --skip-normalize   (bypass Stage 1)

Output:
    A single folder named  <input_stem>_output/  is created in the
    current working directory. Example: input is page1.png →

        page1_output/
            main.tex          ← compilable LaTeX, figures referenced by name only
            figure_001.png    ← YOLO-cropped Picture regions
            figure_002.png
            ...

    Upload the ENTIRE folder to Overleaf. Compiles with images
    intact because \\includegraphics paths are filename-only (no
    subdirectory), so Overleaf resolves them in the same folder.
"""

import sys
import os
import argparse
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

import cv2
import numpy as np
from normalization import normalize_image_pil
from normalization.region_adaptive import preprocess_crop, RegionArtifactProfile
from models_interface import (
    run_text_ocr, run_text_ocr_batched, run_math_recognition, run_table_extraction,
    get_math_latencies, get_text_latencies, get_table_latencies, get_text_batch_latencies
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
# YOLO MODEL
# ----------------------------------------------------------------
# Pretrained YOLOv8n fine-tuned on DocLayNet (11 classes).
# Source: https://huggingface.co/hantian/yolo-doclaynet
# This model outputs all 11 original DocLayNet class names directly.
# No custom training required. Runs on CPU within latency budget.
YOLO_MODEL_PATH = "yolov11n-doclaynet.pt"


# ----------------------------------------------------------------
# Classes handled by each specialist model
# ----------------------------------------------------------------
TEXT_CLASSES = {"Text", "Title", "Section-header", "Caption",
                "Footnote", "Page-footer", "Page-header", "List-item"}
MATH_CLASSES = {"Formula"}
TABLE_CLASSES = {"Table"}
IMAGE_CLASSES = {"Picture"}

# Class names that produce List-item LaTeX (for itemize grouping)
LIST_ITEM_CLASS = "List-item"


def load_model(model_path: str) -> YOLO:
    """Load YOLO model. Downloads automatically on first run if not cached."""
    print(f"[*] Loading YOLO model: {model_path}")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"[!] Could not load {model_path}: {e}")
        print("[!] Attempting fallback to yolov8n.pt (no DocLayNet fine-tune).")
        model = YOLO("yolov8n.pt")
    print("[✓] Model loaded.")
    return model


def _is_likely_logo(crop_pil: Image.Image) -> bool:
    """
    Heuristic: is a page-header crop more likely a logo/image than text?

    Logos have low text-pixel density — few pure-black pixels relative to
    their bounding box.  A text header line is mostly black ink on white.
    We convert to grayscale and measure the fraction of non-white pixels;
    if it is very low AND the crop contains colour variance (not just B&W
    ink), we classify it as a logo/picture region.

    Criteria (both must hold):
      - non-white pixel fraction < 15%  (sparse content, not dense text)
      - colour std-dev > 8 across RGB channels  (has colour, not just ink)
    """
    import numpy as np
    arr = np.array(crop_pil.convert("RGB"), dtype=np.float32)
    gray = arr.mean(axis=2)
    non_white = float(np.mean(gray < 230))
    color_std = float(arr.std())
    return non_white < 0.15 and color_std > 8.0


def run_detection(model: YOLO, image_norm: Image.Image, image_fidelity: Image.Image, image_path: str):
    """
    Run YOLO inference on the image.
    Returns list of detection dicts with keys:
        bbox, class_id, class_name, confidence, crop

    Logo reclassification: Page-header regions that pass the _is_likely_logo
    heuristic are re-labelled as "Picture" so the logo is saved as an image
    crop rather than being passed through OCR (which produces garbage text).
    """
    results = model(image_path, verbose=False)

    detections = []
    result = results[0]

    class_names = result.names  # {0: 'Caption', 1: 'Footnote', ...}

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        class_name = class_names[class_id]

        if class_name in IMAGE_CLASSES:
            crop = xyxy_to_pil_crop(image_fidelity, [x1, y1, x2, y2])
        else:
            crop = xyxy_to_pil_crop(image_norm, [x1, y1, x2, y2])

        # Logo reclassification: page-header crops that look like images
        # (sparse non-white pixels + colour) are saved as figures rather
        # than sent to OCR, which would produce garbled output.
        if class_name == "Page-header":
            fidelity_crop = xyxy_to_pil_crop(image_fidelity, [x1, y1, x2, y2])
            if _is_likely_logo(fidelity_crop):
                print(f"  [detect] Page-header at ({int(x1)},{int(y1)}) reclassified "
                      f"as Picture (logo heuristic)")
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
    Route each detection to the correct specialist model.
    Optimized: Batches text regions for a single OCR pass to save CPU latency.
    
    Returns:
        body_parts      : List[str]  — LaTeX-wrapped content in reading order
        list_indices    : set of int — indices in body_parts that are list items
        figure_counter  : int — final figure counter (for chaining calls)
    """
    os.makedirs(figures_dir, exist_ok=True)
    body_parts = []
    list_indices = set()
    figure_counter = figure_start

    # Stage 3a: Identify text regions and pre-process in batch
    text_indices = [i for i, d in enumerate(detections) if d["class_name"] in TEXT_CLASSES]
    if text_indices:
        text_crops = [detections[i]["crop"] for i in text_indices]
        # Perform batched OCR pass
        raw_texts = run_text_ocr_batched(text_crops)
        # Store back into detection objects
        for idx, raw in zip(text_indices, raw_texts):
            detections[idx]["raw_text"] = raw

    # Stage 3b: Build body parts sequentially (preserving reading order)
    for det in detections:
        class_name = det["class_name"]
        crop = det["crop"]

        if class_name in TEXT_CLASSES:
            # Use pre-computed raw text from batch
            raw = det.get("raw_text", "")
            wrapped = wrap_content(class_name, raw)
            if class_name == LIST_ITEM_CLASS:
                list_indices.add(len(body_parts))

        elif class_name in MATH_CLASSES:
            raw = run_math_recognition(crop)
            wrapped = wrap_content(class_name, raw)

        elif class_name in TABLE_CLASSES:
            raw = run_table_extraction(crop)
            wrapped = wrap_content(class_name, raw)

        elif class_name in IMAGE_CLASSES:
            # Save figure crop, reference it in LaTeX
            figure_counter += 1
            fig_filename = f"figure_{figure_counter:03d}.png"
            fig_path = os.path.join(figures_dir, fig_filename)
            crop.save(fig_path)
            # Filename-only for LaTeX — Overleaf resolves in same folder
            wrapped = wrap_content("Picture", fig_filename)

        else:
            # Unknown class — treat as plain text, log it
            print(f"[!] Unknown class '{class_name}', treating as plain text.")
            raw = run_text_ocr(crop)
            wrapped = wrap_content("Text", raw)

        body_parts.append(wrapped)

    return body_parts, list_indices, figure_counter


def main():
    parser = argparse.ArgumentParser(
        description="Screen2LaTeX Orchestrator — converts document image to LaTeX"
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the input document image (PNG, JPG, etc.)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="YOLO confidence threshold (default: 0.25)"
    )
    parser.add_argument(
        "--skip-normalize",
        action="store_true",
        help="Skip Stage 1 image normalization (useful for clean PDF inputs)"
    )
    parser.add_argument(
        "--target-dpi",
        type=int,
        default=250,
        help="Target DPI for normalization upscale (default: 250)"
    )
    parser.add_argument(
        "--source-dpi",
        type=int,
        default=96,
        help="Assumed source DPI of input image (default: 96 for screen captures)"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run performance metrics (CPU, RAM, Latency) during execution"
    )
    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.image_path):
        print(f"[✗] Image not found: {args.image_path}")
        sys.exit(1)

    # ----------------------------------------------------------------
    # ALL outputs go into one folder: <image_stem>_output/
    # Upload this entire folder to Overleaf — main.tex + all figures
    # are in the same directory, so \includegraphics resolves correctly.
    # ----------------------------------------------------------------
    image_stem = Path(args.image_path).stem
    output_dir = Path(f"{image_stem}_output")
    # Clear stale files from previous runs
    import shutil
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(exist_ok=True)
    tex_path = output_dir / "main.tex"

    print(f"\n[*] Input image  : {args.image_path}")
    print(f"[*] Output folder: {output_dir.absolute()}")
    print(f"[*] LaTeX file   : {tex_path}\n")

    profiler = None
    if args.profile:
        if not HAS_PROFILER:
            print("[!] Warning: --profile requested but 'psutil' is not installed or profiler.py is missing.")
        else:
            print("[*] Performance profiling started...")
            profiler = BackgroundProfiler(interval=0.1)
            profiler.start()

    # ================================================================
    # Stage 1: Image Normalization (NEW)
    # Geometric rectification → moiré removal → glare suppression
    # → DPI scaling. Produces a clean RGB PIL Image for downstream.
    # ================================================================
    if args.skip_normalize:
        print("[*] Stage 1: SKIPPED (--skip-normalize)")
        image_norm = Image.open(args.image_path).convert("RGB")
        image_fidelity = image_norm
        modality_result = None
        is_screenshot = False
    else:
        print("[*] Stage 1: Image Normalization")
        image_norm, image_fidelity, modality_result = normalize_image_pil(
            args.image_path,
            target_dpi=args.target_dpi,
            source_dpi=args.source_dpi,
        )
        is_screenshot = (modality_result.modality.value == "screenshot")
        print(f"[✓] Modality: {modality_result}")
        # Save normalized image for inspection / YOLO input
        normalized_path = output_dir / "normalized.png"
        image_norm.save(str(normalized_path))
        print(f"[✓] Normalized image saved: {normalized_path}")

    # ================================================================
    # Stage 2: Layout Analysis
    # ================================================================
    print("\n[*] Stage 2: Layout Analysis")

    # Step 2a: Load model
    model = load_model(YOLO_MODEL_PATH)

    # For YOLO inference: save normalized image to a temp path if we
    # normalized it (YOLO expects a file path). If skipped, use original.
    if args.skip_normalize:
        yolo_input_path = args.image_path
    else:
        yolo_input_path = str(output_dir / "normalized.png")

    # Step 2b: Detect regions
    detections = run_detection(model, image_norm, image_fidelity, yolo_input_path)

    if not detections:
        print("[!] No regions detected. Check image quality or YOLO model.")
        sys.exit(1)

    # Post-process detections (NMS, overlap resolution, refinement)
    img_width = image_norm.width
    img_height = image_norm.height
    detections = postprocess_detections(detections, img_width, img_height)

    # Re-crop after bbox refinement
    for det in detections:
        if det["class_name"] in IMAGE_CLASSES:
            det['crop'] = xyxy_to_pil_crop(image_fidelity, det['bbox'])
        else:
            det['crop'] = xyxy_to_pil_crop(image_norm, det['bbox'])

    # ----------------------------------------------------------------
    # Stage 1.5: Per-Region Adaptive Preprocessing
    # Run artifact detection on each crop independently. Only apply
    # corrections that are actually needed for that specific region.
    # Picture crops are skipped — their fidelity must be preserved.
    # ----------------------------------------------------------------
    print("\n[*] Stage 1.5: Per-Region Adaptive Preprocessing")
    _region_profiles: list[RegionArtifactProfile] = []

    # ----------------------------------------------------------------
    # Header zone suppression + logo heuristic
    # ----------------------------------------------------------------
    HEADER_SUPPRESS_H_FRAC = 0.12  # suppress any Section/Page-header
                                    # whose bottom (y2) is above this line

    header_suppress_y = img_height * HEADER_SUPPRESS_H_FRAC
    before = len(detections)
    detections = [
        d for d in detections
        if not (
            d["class_name"] in {"Section-header", "Page-header"}
            and d["bbox"][3] <= header_suppress_y
        )
    ]
    suppressed = before - len(detections)
    if suppressed:
        print(f"  [header-suppress] Removed {suppressed} header-zone "
              f"Section-header/Page-header detection(s)")

    # ----------------------------------------------------------------
    # Logo Injection Heuristic
    # ----------------------------------------------------------------
    HEADER_H_FRAC = 0.065
    HEADER_W_FRAC = 0.25
    header_right_box = [
        img_width * (1 - HEADER_W_FRAC), 0,
        img_width,                         img_height * HEADER_H_FRAC,
    ]
    has_picture_in_header = any(
        d["class_name"] == "Picture"
        and d["bbox"][0] >= header_right_box[0]
        and d["bbox"][3] <= header_right_box[3]
        for d in detections
    )
    if not has_picture_in_header:
        hx1, hy1, hx2, hy2 = [int(v) for v in header_right_box]
        header_crop = xyxy_to_pil_crop(image_fidelity, [hx1, hy1, hx2, hy2])
        if header_crop.width > 20 and header_crop.height > 10:
            print(f"  [logo-heuristic] No Picture found in header zone — injecting logo crop "
                  f"({header_crop.width}x{header_crop.height}px)")
            detections.insert(0, {
                "bbox": [hx1, hy1, hx2, hy2],
                "class_id": -1,
                "class_name": "Picture",
                "confidence": 1.0,
                "crop": header_crop,
                "is_header_logo": True,
            })

    for det in detections:
        if det["class_name"] in IMAGE_CLASSES:
            continue  # preserve fidelity crops untouched

        crop_bgr = cv2.cvtColor(np.array(det['crop']), cv2.COLOR_RGB2BGR)
        corrected_bgr, profile = preprocess_crop(crop_bgr, det['class_name'],
                                                  is_screenshot=is_screenshot)
        det['crop'] = Image.fromarray(cv2.cvtColor(corrected_bgr, cv2.COLOR_BGR2RGB))
        _region_profiles.append(profile)

        if profile.any_detected():
            print(f"  [region-prep] {profile.summary()}")

    # Detect layout type (single vs two-column)
    col_count = detect_column_count(detections, img_width)
    print(f"[*] Detected layout: {col_count}-column")

    # ================================================================
    # Stage 3 & 4: Extraction + Assembly
    # ================================================================
    print(f"\n[*] Stage 3: Content Extraction")

    # Re-enable DAG everywhere as per user request
    use_dag = True

    # Fix: Extract header logos to pass as a separate parameter
    header_logo_dets = [d for d in detections if d.get("is_header_logo")]
    body_detections  = [d for d in detections if not d.get("is_header_logo")]
    
    header_logo_fname = None
    if header_logo_dets:
        logo_det = header_logo_dets[0]
        header_logo_fname = "figure_header_logo.png"
        logo_det['crop'].save(output_dir / header_logo_fname)
        print(f"  [logo] Header logo extracted: {header_logo_fname}")

    if col_count == 2:
        full_width_dets, left_dets, right_dets = split_detections_by_column(
            body_detections, img_width, img_height, use_dag=use_dag
        )
        # Process each group separately, chaining figure counter
        full_parts, full_indices, fig_count = route_and_extract(
            full_width_dets, str(output_dir), figure_start=0
        )
        left_parts, left_indices, fig_count = route_and_extract(
            left_dets, str(output_dir), figure_start=fig_count
        )
        right_parts, right_indices, fig_count = route_and_extract(
            right_dets, str(output_dir), figure_start=fig_count
        )
        document = assemble_document(
            body_parts=full_parts,
            list_regions=full_indices,
            is_two_column=True,
            left_parts=left_parts,
            left_list_regions=left_indices,
            right_parts=right_parts,
            right_list_regions=right_indices,
            header_logo=header_logo_fname
        )
    else:
        if use_dag:
            body_detections = apply_semantic_reading_order(
                body_detections, image_width=img_width, image_height=img_height
            )
        else:
            body_detections = sort_detections_geometric(body_detections)
            
        body_parts, list_indices, _ = route_and_extract(
            body_detections, str(output_dir)
        )
        document = assemble_document(
            body_parts=body_parts,
            list_regions=list_indices,
            is_two_column=False,
            header_logo=header_logo_fname
        )

    # Save main.tex inside the output folder
    print(f"\n[*] Stage 4: LaTeX Assembly")
    save_tex(document, str(tex_path))

    if profiler:
        metrics = profiler.stop()
        math_lats = get_math_latencies()
        text_lats = get_text_latencies()
        table_lats = get_table_latencies()
        text_batch_lats = get_text_batch_latencies()
        
        math_mean = round(sum(math_lats)/len(math_lats), 2) if math_lats else 0.0
        text_mean = round(sum(text_lats)/len(text_lats), 2) if text_lats else 0.0
        table_mean = round(sum(table_lats)/len(table_lats), 2) if table_lats else 0.0
        batch_total = round(sum(text_batch_lats), 2) if text_batch_lats else 0.0

        print(f"\n[*] Profiling Results:")
        print(f"    Latency : {metrics['latency_sec']}s")
        print(f"    CPU     : Mean {metrics['cpu_mean_pct']}%, Peak {metrics['cpu_peak_pct']}%")
        print(f"    Memory  : Mean {metrics['mem_mean_mb']} MB, Peak {metrics['mem_peak_mb']} MB")
        if text_batch_lats:
            print(f"    OCR Bat: Total {batch_total} ms (over {len(text_batch_lats)} batches)")
        if text_lats:
            print(f"    Text Seq: Mean {text_mean} ms (over {len(text_lats)} regions)")
        if math_lats:
            print(f"    Math Lat: Mean {math_mean} ms (over {len(math_lats)} equations)")
        if table_lats:
            print(f"    Table Lat: Mean {table_mean} ms (over {len(table_lats)} tables)")
        
        # Log to CSV
        log_file = "profiling_report.csv"
        file_exists = os.path.exists(log_file)
        with open(log_file, "a") as f:
            if not file_exists:
                f.write("Image_Name,Latency_s,CPU_Mean_%,CPU_Peak_%,Mem_Mean_MB,Mem_Peak_MB,OCR_Batch_ms,Math_Mean_ms,Table_Mean_ms\n")
            f.write(f"{image_stem},{metrics['latency_sec']},{metrics['cpu_mean_pct']},{metrics['cpu_peak_pct']},{metrics['mem_mean_mb']},{metrics['mem_peak_mb']},{batch_total},{math_mean},{table_mean}\n")

    print(f"\n[✓] Done.")
    print(f"    Upload folder '{output_dir}/' to Overleaf and compile main.tex")
    print(f"    Or locally: pdflatex {tex_path}\n")


if __name__ == "__main__":
    main()