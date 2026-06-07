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

Screenshot path (steps 1-5 skipped):
  Digital renders have no perspective distortion, physical shadows,
  specular glare, or moire patterns. Only CLAHE + DPI resize.
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

    modality_result = detect_capture_modality(img)
    is_screenshot = modality_result.modality == CaptureModality.SCREENSHOT
    print(f"  [norm] Capture modality: {modality_result}")

    if is_screenshot:
        print("  [norm] Screenshot path: skipping Steps 1-5")
        fidelity_img = img.copy()

        print("  [norm] Step 6: Contrast normalization (CLAHE)")
        img = normalize_contrast(img)

        # Screenshots need no DPI upscale — they're already clean, and our
        # OCR backend (DBNet) caps internally at 1280px anyway. Upscaling
        # 700px → 1815px wastes ~6x RAM and YOLO/crop processing time.
        # Only downscale if the screenshot is unusually large (>1280px).
        SCREENSHOT_MAX_SIDE = 1280
        h, w = img.shape[:2]
        if max(h, w) > SCREENSHOT_MAX_SIDE:
            scale    = SCREENSHOT_MAX_SIDE / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img          = cv2.resize(img,          (new_w, new_h), interpolation=cv2.INTER_AREA)
            fidelity_img = cv2.resize(fidelity_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"  [norm] Screenshot: downsampled to {new_w}x{new_h}")
        else:
            print(f"  [norm] Screenshot: keeping original {w}x{h} (no upscale)")

    else:
        print("  [norm] Phone photo path: full pipeline")

        print("  [norm] Step 1: White balance correction")
        img = white_balance_gray_world(img)

        print("  [norm] Step 2: Geometric rectification")
        img = detect_and_rectify(input_path, img_override=img)

        # Fidelity copy: post-rectification, pre-destructive corrections
        fidelity_img = img.copy()

        print("  [norm] Step 3: Shadow removal (DoG)")
        img = remove_shadows(img)

        print("  [norm] Step 4: Glare removal (inpainting)")
        img = remove_glare(img)

        print("  [norm] Step 5: Moire removal (FFT)")
        img = remove_moire(img)

        print("  [norm] Step 6: Contrast normalization (CLAHE)")
        img = normalize_contrast(img)

        print("  [norm] Step 7: DPI resize")
        img = _smart_dpi_resize(img, target_dpi, source_dpi)
        fidelity_img = _smart_dpi_resize(fidelity_img, target_dpi, source_dpi)

    color = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    final_h, final_w = img.shape[:2]
    print(f"  [norm] Done. Output: {final_w}x{final_h}")
    return color, gray, binary, fidelity_img, modality_result


def normalize_image_pil(input_path, target_dpi=250, source_dpi=96):
    color, _, _, fidelity, modality_result = normalize_image(
        input_path, target_dpi, source_dpi
    )
    rgb_color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    rgb_fidelity = cv2.cvtColor(fidelity, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_color), Image.fromarray(rgb_fidelity), modality_result