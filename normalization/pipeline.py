# normalization/pipeline.py
"""
Full normalization pipeline for Stage 1.

Pipeline order (phone photo — full path):
  1. Load image
  0. Capture modality detection (histogram bin analysis)
  2. White balance correction (gray world)
  3. Geometric rectification (multi-strategy)
  4. Shadow removal (difference-of-Gaussians)
  5. Glare inpainting (LAB threshold + Telea inpaint)
  6. Moiré removal (FFT notch filter)
  7. Contrast normalization (CLAHE)
  8. Smart DPI resize

Screenshot path (steps 1–5 skipped):
  Digital renders have no perspective distortion, physical shadows,
  specular glare, or moiré patterns. Applying these corrections to
  clean digital input wastes computation and can degrade quality
  (e.g., FFT notch filtering introduces ringing on sharp text edges).
  Screenshots only receive CLAHE contrast polish and DPI resize.

Two entry points:
  normalize_image(path, ...)     — returns (color, gray, binary, fidelity, modality)
  normalize_image_pil(path, ...) — returns (PIL.Image, PIL.Image, ModalityResult)
"""

import cv2
import numpy as np
from PIL import Image

from .geometric import detect_and_rectify
from .frequency_filter import (
    white_balance_gray_world,
    remove_shadows,
    remove_glare,
    remove_moire,
    normalize_contrast,
)
from .modality import detect_capture_modality, CaptureModality


def _smart_dpi_resize(img, target_dpi, source_dpi):
    """
    Resize image to target DPI, but skip if the image is already
    large enough or if upscaling would be excessive.

    Phone cameras typically shoot at 3000-4000px wide, which is
    already ~300+ effective DPI for a document page. Upscaling
    those further just wastes memory and hurts YOLO performance.
    """
    h, w = img.shape[:2]
    scale_factor = target_dpi / source_dpi

    # If the image is already large enough (likely a high-res phone photo),
    # don't upscale — only downscale if needed to keep compute bounded.
    LARGE_THRESHOLD = 1000  # lowered from 2000
    shorter_side = min(h, w)

    if shorter_side >= LARGE_THRESHOLD and scale_factor > 1.0:
        # Image is already at workable resolution, cap the scale to avoid bloating
        # and magnifying physical artifacts/sensor noise.
        desired_shorter = 1500 # lowered from 2500
        if shorter_side > desired_shorter:
            scale_factor = desired_shorter / shorter_side
            print(f"  [norm] High-res input detected ({w}x{h}), "
                  f"scaling to {scale_factor:.2f}x instead of upscaling")
        else:
            print(f"  [norm] Image already at good resolution ({w}x{h}), skipping resize")
            return img
    elif scale_factor < 0.5:
        # Don't downscale too aggressively
        scale_factor = 0.5
        print(f"  [norm] Capping downscale at 0.5x")

    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)

    if abs(scale_factor - 1.0) < 0.05:
        # Scale factor ~1.0, skip resize
        return img

    interpolation = cv2.INTER_LANCZOS4 if scale_factor > 1.0 else cv2.INTER_AREA
    img = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
    print(f"  [norm] Resized: {w}x{h} → {new_w}x{new_h} (scale={scale_factor:.2f}x)")
    return img


def normalize_image(input_path, target_dpi=250, source_dpi=96):
    """
    Full normalization pipeline for a single screenshot / phone photo.

    Automatically detects capture modality (screenshot vs. phone photo)
    via grayscale histogram analysis and adapts the pipeline accordingly:
      - Phone photos: full 7-step pipeline
      - Screenshots:  CLAHE + DPI resize only (steps 1–5 skipped)

    Returns:
        color    — BGR uint8 numpy array (cleaned, full-color, DPI-scaled)
        gray     — single-channel grayscale
        binary   — Otsu-thresholded black/white
        fidelity — BGR uint8 numpy array (geometrically rectified, natural colors)
        modality — ModalityResult with classification and diagnostic metrics
    """
    print(f"  [norm] Normalizing: {input_path}")

    # Step 0: Load and basic validation
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")

    h, w = img.shape[:2]
    print(f"  [norm] Input size: {w}x{h}")

    # ================================================================
    # Capture Modality Detection
    # Analyze grayscale histogram to classify input as screenshot
    # (sparse bins, discrete pixel values) or phone photo (dense bins,
    # continuous sensor noise across all 256 intensity levels).
    # ================================================================
    modality_result = detect_capture_modality(img)
    is_screenshot = modality_result.modality == CaptureModality.SCREENSHOT
    print(f"  [norm] Capture modality: {modality_result}")

    if is_screenshot:
        # =============================================================
        # SCREENSHOT PATH — skip physical-artifact corrections
        # Digital renders have no perspective distortion, desk-lamp
        # shadows, specular glare, or screen-on-screen moiré.
        # Applying these filters to clean digital input can actually
        # degrade quality (DoG reduces contrast, FFT causes ringing).
        # =============================================================
        print("  [norm] Screenshot path: skipping Steps 1–5 (no physical artifacts)")
        fidelity_img = img.copy()

        # Step 6: Contrast normalization (CLAHE)
        # Still useful for low-contrast or dark-theme screenshots
        print("  [norm] Step 6: Contrast normalization (CLAHE)")
        img = normalize_contrast(img)

        # Step 7: Smart DPI resize
        print("  [norm] Step 7: DPI resize")
        img = _smart_dpi_resize(img, target_dpi, source_dpi)
        fidelity_img = _smart_dpi_resize(fidelity_img, target_dpi, source_dpi)

    else:
        # =============================================================
        # PHONE PHOTO PATH — full pipeline
        # Camera captures suffer from color casts, perspective
        # distortion, uneven shadows, specular glare, and moiré.
        # All correction steps are applied.
        # =============================================================
        print("  [norm] Phone photo path: full pipeline")

        # Step 1: White balance correction
        # Fix color casts from indoor / artificial lighting
        print("  [norm] Step 1: White balance correction")
        img = white_balance_gray_world(img)

        # Step 2: Geometric rectification
        # Multi-strategy: morph gradient → Hough lines → Canny → fallback
        print("  [norm] Step 2: Geometric rectification")
        img = detect_and_rectify(input_path, img_override=img)

        # Save fidelity copy before destructive steps (DoG, Glare Inpainting, FFT, CLAHE)
        # This preserves natural colors and textures while maintaining the exact
        # geometric coordinates (perspective warp) of the normalized image.
        fidelity_img = img.copy()

        # Step 3: Shadow removal
        # Even out illumination (desk lamp shadows, window gradients)
        print("  [norm] Step 3: Shadow removal (DoG)")
        img = remove_shadows(img)

        # Step 4: Glare inpainting
        # Detect bright spots → fill with surrounding data
        print("  [norm] Step 4: Glare removal (inpainting)")
        img = remove_glare(img)

        # Step 5: Moiré removal via FFT
        print("  [norm] Step 5: Moiré removal (FFT)")
        img = remove_moire(img)

        # Step 6: Contrast normalization (CLAHE)
        print("  [norm] Step 6: Contrast normalization (CLAHE)")
        img = normalize_contrast(img)

        # Step 7: Smart DPI resize
        print("  [norm] Step 7: DPI resize")
        img = _smart_dpi_resize(img, target_dpi, source_dpi)
        fidelity_img = _smart_dpi_resize(fidelity_img, target_dpi, source_dpi)

    # Keep color version
    color = img.copy()

    # Grayscale + binarize (for OCR models that prefer it)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    final_h, final_w = img.shape[:2]
    print(f"  [norm] Done. Output: {final_w}x{final_h}")
    return color, gray, binary, fidelity_img, modality_result


def normalize_image_pil(input_path, target_dpi=250, source_dpi=96):
    """
    Convenience wrapper: returns PIL RGB Images and modality classification.

    Returns:
        normalized_img — PIL.Image (RGB), aggressively cleaned for YOLO/OCR
        fidelity_img   — PIL.Image (RGB), natural colors for figure cropping
        modality       — ModalityResult with classification and metrics
    """
    color, _, _, fidelity, modality_result = normalize_image(
        input_path, target_dpi, source_dpi
    )
    rgb_color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    rgb_fidelity = cv2.cvtColor(fidelity, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_color), Image.fromarray(rgb_fidelity), modality_result

