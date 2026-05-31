# normalization/pipeline.py
"""
Full normalization pipeline for Stage 1.

Pipeline order (phone photo — full path):
  1. Load image
  0. Capture modality detection (histogram bin analysis)
  2. White balance correction (gray world)
  3. Geometric rectification (multi-strategy)
  4. Moiré removal (FFT notch filter)    ← must be first: mesh fakes glare+shadow signals
  5. Glare inpainting (LAB threshold + Telea inpaint)
  6. Shadow removal (difference-of-Gaussians)
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

    Floor raised from 1500 → 2000 (shorter side):
    At 1500px the inter-column gutter on a two-column IEEE page is
    only ~75px wide. YOLO bbox over-expansion (5–15% on glared crops)
    causes boxes to bleed across a 75px gutter, triggering the
    crosses_gutter branch in detect_column_count and collapsing both
    columns into one. At 2000px the gutter is ~100px, which gives
    enough margin for the gutter detection to remain stable.
    """
    h, w = img.shape[:2]
    scale_factor = target_dpi / source_dpi

    LARGE_THRESHOLD = 1000
    shorter_side = min(h, w)

    if shorter_side >= LARGE_THRESHOLD and scale_factor > 1.0:
        desired_shorter = 2000  # raised from 1500 — keeps gutter wide enough for column detection
        if shorter_side > desired_shorter:
            scale_factor = desired_shorter / shorter_side
            print(f"  [norm] High-res input detected ({w}x{h}), "
                  f"scaling to {scale_factor:.2f}x instead of upscaling")
        else:
            print(f"  [norm] Image already at good resolution ({w}x{h}), skipping resize")
            return img
    elif scale_factor < 0.5:
        scale_factor = 0.5
        print(f"  [norm] Capping downscale at 0.5x")

    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)

    if abs(scale_factor - 1.0) < 0.05:
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
    #
    # GLARE OVERRIDE: Heavy glare creates large blown-out regions with
    # constant pixel values that mimic the sparse histogram of a screenshot.
    # If initial classification is "screenshot" but we detect significant
    # glare (>8% of pixels with LAB L > 230), override to phone photo so
    # the full correction pipeline fires.
    # ================================================================
    modality_result = detect_capture_modality(img)
    is_screenshot = modality_result.modality == CaptureModality.SCREENSHOT
    print(f"  [norm] Capture modality: {modality_result}")

    if is_screenshot:
        # Check for glare signature that may have caused misclassification.
        #
        # White-background documents (IEEE pages, papers) are ~85-90% white,
        # which means L > 230 in LAB for almost all pixels. The original
        # threshold of 0.08 (8%) fired on every clean white-background screenshot,
        # overriding it to phone_photo and applying destructive binarization.
        #
        # Fix: use L > 248 (genuinely blown-out / specular) and require > 25%
        # of pixels at that extreme level. Normal white paper sits at L ≈ 230-242;
        # specular glare from a phone flash sits at L ≈ 248-255.
        # This only fires when a large fraction of the image is completely saturated
        # (true phone-photo glare), not on normal white document backgrounds.
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel = img_lab[:, :, 0]
        glare_pixel_fraction = float(np.mean(l_channel > 248))
        print(f"  [norm] Glare pixel fraction (pre-check, L>248): {glare_pixel_fraction:.3f}")
        if glare_pixel_fraction > 0.25:
            print("  [norm] WARNING: Heavy specular glare on classified screenshot — "
                  "likely a misclassified phone photo. Overriding to PHONE_PHOTO path.")
            is_screenshot = False

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

        # Step 7: DPI resize — capped for screenshots.
        # Screenshots are pixel-perfect digital renders; source_dpi=96 and
        # target_dpi=250 would apply a 2.6× upscale to a 729px-wide image,
        # producing a 1898px image with bicubic interpolation artefacts.
        # For screenshots we only upscale if the image is genuinely small
        # (shorter side < 800px), and cap the scale at 1.5× to avoid
        # introducing interpolation blur on clean crisp pixels.
        h_sc, w_sc = img.shape[:2]
        shorter_sc = min(h_sc, w_sc)
        if shorter_sc < 800:
            sc_scale = min(1.5, 800 / shorter_sc)
            new_w_sc = int(w_sc * sc_scale)
            new_h_sc = int(h_sc * sc_scale)
            img = cv2.resize(img, (new_w_sc, new_h_sc), interpolation=cv2.INTER_LANCZOS4)
            fidelity_img = cv2.resize(fidelity_img, (new_w_sc, new_h_sc), interpolation=cv2.INTER_LANCZOS4)
            print(f"  [norm] Step 7: Screenshot small-upscale {w_sc}x{h_sc} → {new_w_sc}x{new_h_sc}")
        else:
            print(f"  [norm] Step 7: Screenshot already adequate resolution ({w_sc}x{h_sc}), skipping resize")

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

        # Save fidelity copy after geometric rectification but before
        # any destructive corrections (moiré/glare/shadow/CLAHE).
        fidelity_img = img.copy()

        # Step 3: Moiré removal via FFT  ← FIRST destructive step
        #
        # Correct pipeline order: moiré → glare → shadow → contrast.
        # Reason: the mesh/grid pattern creates bright intersection spots
        # that look like glare to detect_glare_mask (LAB L > threshold),
        # and look like illumination gradients to DoG shadow detection.
        # If glare or shadow run first they bake the mesh into inpainted
        # regions where FFT can no longer remove it, fusing word boundaries.
        # Running FFT first gives both subsequent stages a clean image.
        print("  [norm] Step 3: Moiré removal (FFT) — first destructive step")
        img = remove_moire(img)

        # Step 4: Glare inpainting
        # Operates on a mesh-free image — glare mask is now clean.
        img_lab_check = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        glare_fraction = float(np.mean(img_lab_check[:, :, 0] > 225))
        print(f"  [norm] Step 4: Glare removal (inpainting) — glare fraction: {glare_fraction:.3f}")
        if glare_fraction > 0.05:
            print("  [norm]   Heavy glare detected — using aggressive two-pass inpainting")
            img = remove_glare(img, lightness_threshold=210)   # pass 1: aggressive
            img = remove_glare(img, lightness_threshold=228)   # pass 2: halos
        else:
            img = remove_glare(img)   # standard single pass

        # Step 5: Shadow removal
        # Even out illumination (desk lamp shadows, window gradients).
        # Runs after moiré+glare so DoG doesn't mistake mesh intersection
        # gradients or inpainted halo edges for illumination shadows.
        print("  [norm] Step 5: Shadow removal (DoG)")
        img = remove_shadows(img)

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