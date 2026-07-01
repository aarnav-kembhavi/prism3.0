# normalization/pipeline.py
"""
Stage 1 normalization pipeline — adaptive gating by modality and defect detection.

Decision tree:
  1. Deskew + modality detection (always)
  2a. Screenshot → skip all Stage 1 (v6 path). Clean digital renders need no
      corrections; CLAHE degrades already-crisp text.
  2b. Phone-photo but clean digital doc/scan (pure-white background present,
      white_frac ≥ 0.02) → skip Stage 1 (v6 path). The entropy modality detector
      mislabels content-rich digital docs as photos; the white test rescues them.
  2c. Genuine camera capture (white_frac < 0.02 — no pure white, as real
      sensors/lighting produce) → full Stage 1.

The white-fraction gate replaced an earlier moiré/glare/shadow gate that fired
on ~100% of pages (moiré ~300 and white paper both trip those metrics — they
don't discriminate). See the _CAMERA_WHITE_THRESHOLD note below.

Full Stage 1 order (camera-capture path):
  1. White balance correction (gray world)
  2. Geometric rectification (multi-strategy)
  3. Shadow removal (difference-of-Gaussians)
  4. Glare inpainting (LAB threshold + Telea)
  5. Moiré removal (FFT notch filter)
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
)
from .modality import detect_capture_modality, CaptureModality

# Camera-capture gate. The entropy modality detector labels both genuine phone
# captures AND clean digital docs / scans as "phone_photo". The reliable way to
# tell them apart is pure-white presence: a real camera capture has almost no
# near-white (L>250) pixels — sensor noise and real lighting never produce pure
# white, so paper reads ~240 off-white — whereas digital renders and clean scans
# have large pure-white backgrounds. A page below this white fraction is treated
# as a genuine capture and gets full Stage 1; above it, it's a clean doc and
# skips (the benchmark-best v6 path).
#
# Chosen by a labeled gate sweep (22-60 clean phone-photo pages vs 6 real defect
# photos): white<0.02 gives 100% defect recall at ~10% false-positive, versus the
# old moiré/glare thresholds which fired on ~100% of pages (moiré ~300 and white
# paper both trip them — they don't discriminate). Adding skew/noise signals
# either raised false positives or put the threshold dangerously close to real
# defects, so a single robust test wins.
_CAMERA_WHITE_THRESHOLD = 0.02


def _white_fraction(image_bgr):
    """Fraction of near-pure-white (LAB L > 250) pixels."""
    L = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
    return float(np.mean(L > 250))


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

        # Cap the SHORTER side at 1800px (matches the clean-photo path) instead
        # of the longer side at 1280. The old 1280-longer cap shrank a 2667x1500
        # screenshot to 1280x720, and formula crops taken from it lost the detail
        # Texo needs — costing several points of formula accuracy vs the
        # full-resolution skip-stage1 path. 1800-shorter preserves detail while
        # still bounding memory on unusually large captures. Never upscale.
        SCREENSHOT_MAX_SHORTER = 1800
        h, w = img.shape[:2]
        if min(h, w) > SCREENSHOT_MAX_SHORTER:
            scale = SCREENSHOT_MAX_SHORTER / min(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img          = cv2.resize(img,          (new_w, new_h), interpolation=cv2.INTER_AREA)
            fidelity_img = cv2.resize(fidelity_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"  [norm] Screenshot: downsampled to {new_w}x{new_h}")
        else:
            print(f"  [norm] Screenshot: keeping original {w}x{h}")

    else:
        # Phone-photo modality — decide genuine camera capture vs a clean digital
        # doc / scan the entropy detector mislabeled, using pure-white presence.
        white_frac = _white_fraction(img)
        is_camera_photo = white_frac < _CAMERA_WHITE_THRESHOLD
        print(f"  [norm] Phone-photo white_frac={white_frac:.4f} "
              f"(thr {_CAMERA_WHITE_THRESHOLD}) -> "
              f"{'camera capture' if is_camera_photo else 'clean digital'}")

        if not is_camera_photo:
            # Clean digital doc / scan misclassified as phone_photo — skip Stage 1
            # (benchmark-best v6 path). CLAHE/white-balance/DPI-resize hurt formula
            # OCR on already-clean pages (+11.6pp formula, v6 vs v4).
            print("  [norm] Clean digital — skipping Stage 1 corrections")
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
            # Genuine camera capture — run full Stage 1 (a real photo benefits from
            # white balance, rectification, shadow/glare/moiré removal, and CLAHE).
            print("  [norm] Camera capture — running full Stage 1")

            print("  [norm] Step 1: White balance correction")
            img = white_balance_gray_world(img)

            print("  [norm] Step 2: Geometric rectification")
            img = detect_and_rectify(input_path, img_override=img)

            import gc as _gc; _gc.collect()
            fidelity_img = img.copy()

            print("  [norm] Step 3: Shadow removal")
            img = remove_shadows(img)

            print("  [norm] Step 4: Glare removal (inpainting)")
            img = remove_glare(img)

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