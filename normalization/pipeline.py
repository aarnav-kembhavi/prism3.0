# normalization/pipeline.py
"""
Stage 1 normalization pipeline — adaptive gating by modality and defect detection.

Decision tree:
  1. Deskew + modality detection (always)
  2a. Screenshot → skip all Stage 1 (v6 path). Clean digital renders need no
      corrections; CLAHE degrades already-crisp text.
  2b. Photo, no defects detected → skip Stage 1 (v6 path). Saves formula OCR
      quality on clean captures (+11.6pp formula English vs full Stage 1).
  2c. Photo, defects detected (shadow ≥ 0.20, glare ≥ 0.5%, moiré ≥ 4.0) →
      full Stage 1 with only the triggered correction steps applied.

Full Stage 1 order (defective photo path):
  1. White balance correction (gray world)
  2. Geometric rectification (multi-strategy)
  3. Shadow removal (difference-of-Gaussians)     — if has_shadow
  4. Glare inpainting (LAB threshold + Telea)     — if has_glare
  5. Moiré removal (FFT notch filter)              — if has_moire
  6. Contrast normalization (CLAHE)
  7. Smart DPI resize
"""

import cv2
import numpy as np
from PIL import Image

from .geometric import detect_and_rectify, deskew
from .frequency_filter import (
    white_balance_gray_world,
    remove_shadows,
    remove_glare,
    remove_moire,
    normalize_contrast,
    detect_glare_mask,
    measure_shadow_gradient,
    measure_moire_score,
)
from .modality import detect_capture_modality, CaptureModality

# Defect detection thresholds for photo gating
_SHADOW_THRESHOLD = 0.20   # illumination ratio std-dev
_MOIRE_THRESHOLD  = 4.0    # high-freq FFT peak/mean ratio
_GLARE_THRESHOLD  = 0.005  # fraction of image pixels flagged as glare


def _measure_glare_coverage(image_bgr):
    mask = detect_glare_mask(image_bgr)
    return float(np.count_nonzero(mask)) / (image_bgr.shape[0] * image_bgr.shape[1])


def _smart_dpi_resize(img, target_dpi, source_dpi):
    h, w = img.shape[:2]
    scale_factor = target_dpi / source_dpi

    # LARGE_THRESHOLD=1200: any phone photo with shorter side >= 1200px is
    # already at workable resolution for YOLO. Upscaling a 1599px image by
    # 2.6x to 4164px causes YOLO to rescale it back down internally, losing
    # detection quality and wasting memory. Cap at 1800px shorter side.
    LARGE_THRESHOLD = 1200
    shorter_side = min(h, w)

    if shorter_side >= LARGE_THRESHOLD and scale_factor > 1.0:
        desired_shorter = 1800
        if shorter_side > desired_shorter:
            scale_factor = desired_shorter / shorter_side
            print(f"  [norm] High-res input ({w}x{h}), scaling to {scale_factor:.2f}x")
        else:
            print(f"  [norm] Already good resolution ({w}x{h}), skipping resize")
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
    print(f"  [norm] Resized: {w}x{h} -> {new_w}x{new_h} (scale={scale_factor:.2f}x)")
    return img


def normalize_image(input_path, target_dpi=250, source_dpi=96):
    print(f"  [norm] Normalizing: {input_path}")

    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")

    h, w = img.shape[:2]
    print(f"  [norm] Input size: {w}x{h}")

    img = deskew(img)

    modality_result = detect_capture_modality(img)
    is_screenshot = modality_result.modality == CaptureModality.SCREENSHOT
    print(f"  [norm] Capture modality: {modality_result}")

    if is_screenshot:
        # Clean digital render — no corrections needed (v6 path).
        # Screenshots have no perspective distortion, shadows, glare, moiré,
        # and are already well-contrasted (CLAHE degrades clean digital text).
        print("  [norm] Screenshot: skipping all Stage 1 corrections")
        fidelity_img = img.copy()

        # Only downscale if unusually large — never upscale.
        SCREENSHOT_MAX_SIDE = 1280
        h, w = img.shape[:2]
        if max(h, w) > SCREENSHOT_MAX_SIDE:
            scale = SCREENSHOT_MAX_SIDE / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img          = cv2.resize(img,          (new_w, new_h), interpolation=cv2.INTER_AREA)
            fidelity_img = cv2.resize(fidelity_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"  [norm] Screenshot: downsampled to {new_w}x{new_h}")
        else:
            print(f"  [norm] Screenshot: keeping original {w}x{h}")

    else:
        # Phone photo — run defect detection to decide whether Stage 1 is needed.
        shadow_score = measure_shadow_gradient(img)
        moire_score  = measure_moire_score(img)
        glare_cov    = _measure_glare_coverage(img)

        has_shadow = shadow_score >= _SHADOW_THRESHOLD
        has_moire  = moire_score  >= _MOIRE_THRESHOLD
        has_glare  = glare_cov    >= _GLARE_THRESHOLD
        has_defects = has_shadow or has_moire or has_glare

        print(
            f"  [norm] Defect scores — shadow: {shadow_score:.3f} "
            f"(thr {_SHADOW_THRESHOLD}), moiré: {moire_score:.1f} "
            f"(thr {_MOIRE_THRESHOLD}), glare: {glare_cov:.4f} "
            f"(thr {_GLARE_THRESHOLD})"
        )
        print(f"  [norm] Defects: shadow={has_shadow}, moiré={has_moire}, glare={has_glare}")

        if not has_defects:
            # Clean photo — skip Stage 1 corrections (v6 path).
            # White balance, CLAHE, and DPI resize all hurt formula OCR
            # on already-clean captures (benchmark v6 vs v4: +11.6pp formula).
            print("  [norm] Photo clean — skipping Stage 1 corrections")
            fidelity_img = img.copy()

            # Light cap: phone photos can be 12MP+; cap shorter side at 1800px
            # for YOLO memory management without the full DPI-based resize.
            PHOTO_MAX_SHORTER = 1800
            shorter = min(img.shape[:2])
            if shorter > PHOTO_MAX_SHORTER:
                scale = PHOTO_MAX_SHORTER / shorter
                new_w = int(img.shape[1] * scale)
                new_h = int(img.shape[0] * scale)
                img          = cv2.resize(img,          (new_w, new_h), interpolation=cv2.INTER_AREA)
                fidelity_img = cv2.resize(fidelity_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                print(f"  [norm] Photo: capped to {new_w}x{new_h} (shorter side {PHOTO_MAX_SHORTER}px)")
            else:
                print(f"  [norm] Photo: keeping original {w}x{h}")

        else:
            # Defective photo — run full Stage 1.
            print("  [norm] Photo defective — running full Stage 1")

            print("  [norm] Step 1: White balance correction")
            img = white_balance_gray_world(img)

            print("  [norm] Step 2: Geometric rectification")
            img = detect_and_rectify(input_path, img_override=img)

            import gc as _gc; _gc.collect()
            fidelity_img = img.copy()

            if has_shadow:
                print("  [norm] Step 3: Shadow removal")
                img = remove_shadows(img)

            if has_glare:
                print("  [norm] Step 4: Glare removal (inpainting)")
                img = remove_glare(img)

            if has_moire:
                print("  [norm] Step 5: Moiré removal (FFT notch)")
                img = remove_moire(img)

            import gc as _gc; _gc.collect()

            print("  [norm] Step 6: Contrast normalization (CLAHE)")
            img = normalize_contrast(img)

            print("  [norm] Step 7: DPI resize")
            img = _smart_dpi_resize(img, target_dpi, source_dpi)
            fidelity_img = _smart_dpi_resize(fidelity_img, target_dpi, source_dpi)

    final_h, final_w = img.shape[:2]
    print(f"  [norm] Done. Output: {final_w}x{final_h}")
    return img, fidelity_img, modality_result


def normalize_image_pil(input_path, target_dpi=250, source_dpi=96):
    color, fidelity, modality_result = normalize_image(input_path, target_dpi, source_dpi)
    rgb_color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    rgb_fidelity = cv2.cvtColor(fidelity, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_color), Image.fromarray(rgb_fidelity), modality_result


def normalize_image_pil_skip_stage1(input_path):
    """
    Stage 1.5-only baseline: deskew + modality detection only.
    No white balance, rectification, shadow/glare/moiré/CLAHE, or DPI resize.
    Returns the deskewed raw image as both norm and fidelity so YOLO and OCR
    both see uncorrected pixels. Used for ablation comparisons.
    """
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")
    img = deskew(img)
    modality_result = detect_capture_modality(img)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return pil, pil.copy(), modality_result