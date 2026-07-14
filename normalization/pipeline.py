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

import os

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


def _paper_shadow_dim(image_bgr):
    """How much the local 'paper white' dims across the page (0 = uniform).

    4x4 grid; per cell take the p95 grayscale as local paper white, keep only
    cells whose bright pixels are low-saturation (excludes colored design
    backgrounds), and compare the darkest to the brightest paper cells.
    Real illumination shadows dim PAPER; page design does not.
    """
    h, w = image_bgr.shape[:2]
    scale = 640 / max(h, w)
    img = cv2.resize(image_bgr, (int(w * scale), int(h * scale)),
                     interpolation=cv2.INTER_AREA) if scale < 1 else image_bgr
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]
    H, W = gray.shape
    whites = []
    for gy in range(4):
        for gx in range(4):
            cg = gray[gy * H // 4:(gy + 1) * H // 4, gx * W // 4:(gx + 1) * W // 4]
            cs = sat[gy * H // 4:(gy + 1) * H // 4, gx * W // 4:(gx + 1) * W // 4]
            p95 = np.percentile(cg, 95)
            bright = cg >= p95 * 0.97
            if bright.sum() < 30 or np.median(cs[bright]) > 45:
                continue
            whites.append(p95)
    if len(whites) < 6:
        return 0.0
    whites = np.array(whites)
    top = np.percentile(whites, 90)
    return float(1.0 - np.percentile(whites, 10) / max(top, 1e-6))


def normalize_image(input_path, target_dpi=250, source_dpi=96):
    print(f"  [norm] Normalizing: {input_path}")

    # Unicode-safe read: cv2.imread returns None on non-ASCII Windows paths
    # (e.g. Chinese filenames with full-width parens), so decode via numpy.
    img = cv2.imdecode(np.fromfile(input_path, dtype=np.uint8), cv2.IMREAD_COLOR)
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
        SCREENSHOT_MAX_SHORTER = int(os.environ.get('PRISM_MAX_SHORTER', '1800'))
        # Resolution FLOOR: low-DPI renders (Fox pages are ~857x1109) starve
        # detection and OCR of stroke detail — dense two-column pages lost
        # half their text (pred 1397 chars vs GT 5807 on the worst page).
        # 2x-upscale anything whose shorter side is below 1100. OmniDocBench
        # pages are all >1300px, so benchmark output is unchanged.
        SCREENSHOT_MIN_SHORTER = 1100
        h, w = img.shape[:2]
        if min(h, w) > SCREENSHOT_MAX_SHORTER:
            scale = SCREENSHOT_MAX_SHORTER / min(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img          = cv2.resize(img,          (new_w, new_h), interpolation=cv2.INTER_AREA)
            fidelity_img = cv2.resize(fidelity_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"  [norm] Screenshot: downsampled to {new_w}x{new_h}")
        elif (min(h, w) < SCREENSHOT_MIN_SHORTER
              and os.environ.get('PRISM_RES_FLOOR', '1') != '0'):
            new_w, new_h = w * 2, h * 2
            img          = cv2.resize(img,          (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            fidelity_img = cv2.resize(fidelity_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            print(f"  [norm] Screenshot: low-DPI input, upscaled to {new_w}x{new_h}")
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

        # Product-mode rescue: a shadowed capture can still contain >2% pure
        # white (bright paper next to the shadow), which the white-frac test
        # mistakes for a clean digital page and skips all corrections
        # (observed on the shadowed defect samples). If the local paper white
        # dims noticeably across the page, it IS a lit capture — run the
        # camera stack. PRISM_NORM_STRICT=1 (set by the benchmark runner)
        # pins the original routing: OmniDocBench design pages (PPT
        # gradients) can mimic this signature, and benchmark behavior must
        # stay byte-identical.
        if (not is_camera_photo
                and os.environ.get('PRISM_NORM_STRICT', '0') != '1'):
            dim = _paper_shadow_dim(img)
            if dim > 0.05:
                print(f"  [norm] Paper white dims {dim:.2f} across page -> "
                      f"treating as shadowed capture")
                is_camera_photo = True

        if not is_camera_photo:
            # Clean digital doc / scan misclassified as phone_photo — skip Stage 1
            # (benchmark-best v6 path). CLAHE/white-balance/DPI-resize hurt formula
            # OCR on already-clean pages (+11.6pp formula, v6 vs v4).
            print("  [norm] Clean digital — skipping Stage 1 corrections")
            fidelity_img = img.copy()

            # Light cap: phone photos can be 12MP+; cap shorter side at 1800px
            # for YOLO memory management without the full DPI-based resize.
            PHOTO_MAX_SHORTER = int(os.environ.get('PRISM_MAX_SHORTER', '1800'))
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
            # (All fixes below are CAMERA-BRANCH ONLY — the benchmark-validated
            # screenshot / clean-digital paths above are untouched.)
            print("  [norm] Camera capture — running full Stage 1")

            # (EXIF orientation needs no handling here: cv2.imdecode applies
            # the JPEG orientation tag itself since OpenCV 3.4 — verified on a
            # tag-6 portrait receipt that loads upright.)

            # Pre-cap BEFORE the correction stack: glare inpainting, FFT moiré
            # and Hough rectification on a 9 MP capture cost 20-30 s/page for
            # no quality gain — the end of this branch capped to 1800px shorter
            # side anyway. Capping first makes every step pay on ≤1800px.
            _cap = int(os.environ.get('PRISM_MAX_SHORTER', '1800'))
            _shorter = min(img.shape[:2])
            if _shorter > _cap:
                _scale = _cap / _shorter
                img = cv2.resize(img, (int(img.shape[1] * _scale),
                                       int(img.shape[0] * _scale)),
                                 interpolation=cv2.INTER_AREA)
                print(f"  [norm] Pre-capped to {img.shape[1]}x{img.shape[0]} "
                      f"(shorter side 1800px)")

            # Recognition-verified mode (product default): every correction
            # below is a PROPOSAL, accepted only if the text-detection probe
            # does not get worse (normalization/verified.py). Open-loop mode
            # (PRISM_NORM_VERIFY=0, or probe model absent) applies them
            # unconditionally as before.
            from normalization import verified as _nv
            _verify = (_nv.available()
                       and os.environ.get('PRISM_NORM_STRICT', '0') != '1')
            _score = None
            if _verify:
                _score = _nv.probe_score(img)
                print(f"  [norm] Verified mode: base probe score {_score:.4f}")

            print("  [norm] Step 1: White balance correction")
            if _verify:
                img, _score, _ = _nv.verified_apply(
                    'white_balance', white_balance_gray_world, img, _score)
            else:
                img = white_balance_gray_world(img)

            print("  [norm] Step 2: Geometric rectification")
            if _verify:
                img, _score, _ = _nv.verified_apply(
                    'rectify',
                    lambda im: detect_and_rectify(input_path, img_override=im),
                    img, _score)
            else:
                img = detect_and_rectify(input_path, img_override=img)

            import gc as _gc; _gc.collect()
            fidelity_img = img.copy()

            # Glare BEFORE shadow removal: the shadow divide brightens paper
            # to ~255, after which the L>230 glare detector saw 99.7% of the
            # page as "glare" and Telea-inpainted the entire document into
            # mush. On the pre-flattened image real glare is still a distinct
            # local highlight.
            print("  [norm] Step 3: Glare removal (inpainting)")
            if _verify:
                img, _score, _ = _nv.verified_apply('glare', remove_glare, img, _score)
            else:
                img = remove_glare(img)

            print("  [norm] Step 4: Shadow removal")
            if _verify:
                img, _score, _ = _nv.verified_apply('shadow', remove_shadows, img, _score)
            else:
                img = remove_shadows(img)

            # Moiré is a SCREEN-photo artifact; on paper photos the strongest
            # high-frequency FFT spikes are the text-line periodicity itself,
            # so the notch filter GHOSTS the text (measured on the receipt
            # samples: text visibly faded). Opt-in for screen captures only.
            if os.environ.get('PRISM_MOIRE', '0') != '0':
                print("  [norm] Step 5: Moiré removal (FFT notch)")
                if _verify:
                    img, _score, _ = _nv.verified_apply('moire', remove_moire, img, _score)
                else:
                    img = remove_moire(img)
            else:
                print("  [norm] Step 5: Moiré removal skipped (paper photo; PRISM_MOIRE=1 to enable)")

            import gc as _gc; _gc.collect()

            # CLAHE only helps LOW-contrast captures; after the shadow divide
            # the page is already well-separated on most photos and CLAHE just
            # re-amplifies background texture (wood grain, paper noise).
            _std = float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).std())
            if _std < 45.0:
                print(f"  [norm] Step 6: Contrast normalization (CLAHE, std={_std:.0f})")
                if _verify:
                    img, _score, _ = _nv.verified_apply('clahe', normalize_contrast, img, _score)
                else:
                    img = normalize_contrast(img)
            else:
                print(f"  [norm] Step 6: CLAHE skipped (contrast already good, std={_std:.0f})")

            print("  [norm] Step 7: resolution floor")
            # The old 250-DPI resize UPSCALED small captures 2.6x (971px ->
            # 2528px) — pure waste: layout detection runs at 800px and the OCR
            # worker caps crops at 1500px, so the upscale was immediately
            # undone downstream. Keep only a mild floor for tiny captures.
            _shorter = min(img.shape[:2])
            if _shorter < 900:
                _scale = min(1.5, 900.0 / _shorter)
                _new = (int(img.shape[1] * _scale), int(img.shape[0] * _scale))
                img = cv2.resize(img, _new, interpolation=cv2.INTER_CUBIC)
                fidelity_img = cv2.resize(fidelity_img, _new, interpolation=cv2.INTER_CUBIC)
                print(f"  [norm] Upscaled small capture {_scale:.2f}x -> {_new[0]}x{_new[1]}")

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
    img = cv2.imdecode(np.fromfile(input_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")
    img = deskew(img)
    modality_result = detect_capture_modality(img)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return pil, pil.copy(), modality_result