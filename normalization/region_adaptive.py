# normalization/region_adaptive.py
"""
Per-Region Adaptive Preprocessing — Stage 1.5

MOTIVATION
----------
The global normalization pipeline (pipeline.py) applies corrections to the
entire image before layout detection. This is a blunt instrument: a page
that has one glared paragraph and seven clean paragraphs will have the
inpainting algorithm run over all eight regions regardless.

This module moves preprocessing *downstream* — after YOLO has partitioned
the page into individual region crops — and applies only the corrections
that are actually needed for each crop, independently.

DESIGN
------
Three-layer architecture:

  1. Artifact Detectors
     Lightweight per-artifact probes that inspect a single crop and return
     a boolean + a severity float in [0, 1].  They are designed to be cheap
     relative to the corrections they gate:
       - detect_glare    : LAB L-channel threshold (reuses detect_glare_mask)
       - detect_shadow   : illumination gradient via DoG background ratio
       - detect_moire    : FFT spike energy outside DC radius
       - detect_low_contrast : RMS contrast of grayscale crop

  2. RegionArtifactProfile  (dataclass)
     Records which artifacts were detected and at what severity, plus which
     corrections were actually applied.  Returned alongside the corrected
     crop so orchestrate.py can log or aggregate it.

  3. preprocess_crop(crop_bgr, class_name) → (corrected_bgr, profile)
     The public API.  Runs the detectors that are relevant for the given
     YOLO class, applies corrections that fired, and returns the result.

CLASS-AWARE GATING
------------------
Different region types have different preprocessing needs and tolerances:

  Picture   — skip everything.  Natural color and texture must be
               preserved for \includegraphics.  Any correction risks
               altering visual content.

  Formula   — skip shadow removal.  DoG normalisation darkens thin
               strokes and can break MER encoder attention.  Glare
               and contrast correction are still applied.

  Table     — full treatment.  Tables often sit near page edges where
               glare and shadow are worst, and OCR accuracy on cell text
               is highly sensitive to contrast uniformity.

  Text / Title / Section-header / Caption / Footnote / List-item
            — full treatment.

  Page-header / Page-footer
            — contrast only.  These are narrow strips; geometric
               artefacts are rare, and shadow removal on thin bands
               produces ringing.

THRESHOLDS
----------
All thresholds are empirically chosen and documented.  They are module-level
constants so a researcher can tune them from a single place.

  GLARE_LIGHTNESS_THRESH   = 230   (LAB L, same as global pipeline)
  GLARE_AREA_THRESH        = 0.005 (0.5 % of crop pixels)
  SHADOW_GRADIENT_THRESH   = 0.25  (DoG background ratio std-dev)
  MOIRE_SPIKE_RATIO_THRESH = 1.8   (peak / mean outside DC zone)
  CONTRAST_RMS_THRESH      = 18.0  (grayscale RMS contrast, 0–127.5)

USAGE
-----
    from normalization.region_adaptive import preprocess_crop

    corrected_bgr, profile = preprocess_crop(crop_bgr, class_name="Text")
    if profile.glare_corrected:
        print(f"  glare severity {profile.glare_severity:.2f} → inpainted")

INTEGRATION WITH orchestrate.py
--------------------------------
Call preprocess_crop immediately after re-cropping, before handing the
crop to models_interface:

    # After det['crop'] is assigned (post-NMS recrop loop):
    if det['class_name'] not in IMAGE_CLASSES:
        crop_bgr = cv2.cvtColor(np.array(det['crop']), cv2.COLOR_RGB2BGR)
        corrected_bgr, profile = preprocess_crop(crop_bgr, det['class_name'])
        det['crop'] = Image.fromarray(cv2.cvtColor(corrected_bgr, cv2.COLOR_BGR2RGB))
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Set

from .frequency_filter import (
    detect_glare_mask,
    remove_glare,
    remove_shadows,
    remove_moire,
    normalize_contrast,
)


# ================================================================
# Thresholds — tune from here
# ================================================================

GLARE_LIGHTNESS_THRESH: int   = 230    # LAB L threshold for glare pixels
GLARE_AREA_THRESH: float      = 0.005  # min fraction of crop area to trigger (phone photos)
GLARE_AREA_THRESH_SCREENSHOT: float = 0.25  # stricter threshold for screenshots:
                                             # white page backgrounds easily have >0.5% of
                                             # pixels at L>230, so we need 25% to be truly
                                             # glare-bright before inpainting (which destroys text).
SHADOW_GRADIENT_THRESH: float = 0.25   # DoG background ratio std-dev trigger
MOIRE_SPIKE_RATIO_THRESH: float = 1.8  # FFT peak/mean ratio trigger (phone photos)
MOIRE_SPIKE_RATIO_THRESH_SCREENSHOT: float = 5.0  # screenshots have no sensor moiré;
                                                    # only trigger on extreme FFT spikes
CONTRAST_RMS_THRESH: float    = 18.0   # grayscale RMS contrast trigger


# ================================================================
# Class-aware gating tables
# ================================================================

# Classes that receive NO preprocessing (preserve natural appearance)
_SKIP_ALL: Set[str] = {"Picture"}

# Classes that skip shadow removal (DoG hurts thin strokes)
_SKIP_SHADOW: Set[str] = {"Formula"}

# Classes that receive contrast correction only (narrow strips)
_CONTRAST_ONLY: Set[str] = {"Page-header", "Page-footer"}


# ================================================================
# Result dataclass
# ================================================================

@dataclass
class RegionArtifactProfile:
    """
    Records the artifact detection results and which corrections ran
    for a single region crop.

    Severity scores are in [0, 1] where 0 = clean, 1 = worst case.
    Boolean flags record what was actually applied (detection may fire
    but correction could still be skipped due to class gating).
    """
    class_name: str = ""

    # Detection results
    glare_detected: bool = False
    glare_severity: float = 0.0       # fraction of crop pixels that are glare

    shadow_detected: bool = False
    shadow_severity: float = 0.0      # std-dev of DoG background ratio

    moire_detected: bool = False
    moire_severity: float = 0.0       # FFT spike ratio (peak / mean)

    low_contrast_detected: bool = False
    contrast_rms: float = 0.0         # RMS contrast of grayscale crop

    # Applied corrections
    glare_corrected: bool = False
    shadow_corrected: bool = False
    moire_corrected: bool = False
    contrast_corrected: bool = False

    # Corrections skipped due to class gating (even though detected)
    skipped_corrections: list = field(default_factory=list)

    def any_detected(self) -> bool:
        return (self.glare_detected or self.shadow_detected
                or self.moire_detected or self.low_contrast_detected)

    def any_applied(self) -> bool:
        return (self.glare_corrected or self.shadow_corrected
                or self.moire_corrected or self.contrast_corrected)

    def summary(self) -> str:
        """One-line human-readable summary for logging."""
        detected = []
        if self.glare_detected:
            detected.append(f"glare({self.glare_severity:.2f})")
        if self.shadow_detected:
            detected.append(f"shadow({self.shadow_severity:.2f})")
        if self.moire_detected:
            detected.append(f"moire({self.moire_severity:.2f})")
        if self.low_contrast_detected:
            detected.append(f"contrast(rms={self.contrast_rms:.1f})")

        applied = []
        if self.glare_corrected:
            applied.append("glare")
        if self.shadow_corrected:
            applied.append("shadow")
        if self.moire_corrected:
            applied.append("moire")
        if self.contrast_corrected:
            applied.append("contrast")

        if not detected:
            return f"[{self.class_name}] clean — no corrections needed"

        skipped_str = ""
        if self.skipped_corrections:
            skipped_str = f" | skipped(class-gated): {','.join(self.skipped_corrections)}"

        return (f"[{self.class_name}] "
                f"detected: {', '.join(detected)} | "
                f"applied: {', '.join(applied) if applied else 'none'}"
                + skipped_str)


# ================================================================
# Artifact Detectors
# ================================================================

def detect_glare(crop_bgr: np.ndarray, is_screenshot: bool = False) -> tuple[bool, float]:
    """
    Detect glare in a crop using the existing LAB L-channel mask.

    Returns (detected: bool, severity: float)
    severity = fraction of crop pixels identified as glare [0, 1].

    Trigger threshold differs by modality:
      - Phone photo : GLARE_AREA_THRESH (0.5%) — sensor glare is localised
      - Screenshot  : GLARE_AREA_THRESH_SCREENSHOT (25%) — white page backgrounds
                      have many pixels at L>230 but that is NOT glare; we only
                      fire when a very large fraction is glare-bright, indicating
                      a true overexposed region rather than a white margin.

    Reuses detect_glare_mask from frequency_filter.py so the detection
    logic is identical to what remove_glare uses — no threshold mismatch.
    """
    total_pixels = crop_bgr.shape[0] * crop_bgr.shape[1]
    if total_pixels == 0:
        return False, 0.0

    mask = detect_glare_mask(crop_bgr, lightness_threshold=GLARE_LIGHTNESS_THRESH)
    glare_pixels = int(cv2.countNonZero(mask))
    severity = glare_pixels / total_pixels

    threshold = GLARE_AREA_THRESH_SCREENSHOT if is_screenshot else GLARE_AREA_THRESH
    return severity >= threshold, severity


def detect_shadow(crop_bgr: np.ndarray) -> tuple[bool, float]:
    """
    Detect uneven shadows using DoG background ratio analysis.

    We compute the same "background illumination estimate" that
    remove_shadows uses (heavy Gaussian blur), then measure the
    spatial std-dev of the per-pixel ratio (original / background).
    A uniform image → ratio ~constant → low std-dev.
    A shadowed image → ratio varies spatially → high std-dev.

    Returns (detected: bool, severity: float)
    severity = std-dev of the illumination ratio, normalised to [0, 1]
               via tanh(severity / 0.5) so the value is interpretable.

    Trigger: raw std-dev >= SHADOW_GRADIENT_THRESH (default 0.25).

    Note: works in grayscale to keep the detector cheap (no per-channel
    processing).  Shadow is a luminance phenomenon so this is sufficient.
    """
    if crop_bgr.shape[0] < 16 or crop_bgr.shape[1] < 16:
        # Too small for meaningful shadow analysis
        return False, 0.0

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Match the blur kernel from remove_shadows
    blur_large = 51 | 1   # must be odd
    bg = cv2.GaussianBlur(gray, (blur_large, blur_large), 0)

    # Per-pixel illumination ratio
    ratio = gray / (bg + 1e-6)

    # Spatial std-dev of the ratio measures illumination non-uniformity
    std_dev = float(np.std(ratio))

    # Normalise for the severity field (still use raw for threshold)
    severity_normalised = float(np.tanh(std_dev / 0.5))

    return std_dev >= SHADOW_GRADIENT_THRESH, severity_normalised


def detect_moire(crop_bgr: np.ndarray, is_screenshot: bool = False) -> tuple[bool, float]:
    """
    Detect moiré patterns via FFT spike analysis.

    Moiré appears in the frequency domain as one or more sharp, high-
    amplitude spikes outside the DC component.  We measure the ratio of
    the maximum magnitude spike to the mean magnitude in the high-frequency
    zone.  A clean image has a smooth roll-off → ratio near 1.  A moiré
    image has isolated energy peaks → ratio >> 1.

    Returns (detected: bool, severity: float)
    severity = (peak_magnitude / mean_magnitude) in the analysis zone,
               clipped to [0, 10] and rescaled to [0, 1].

    Trigger: raw ratio >= MOIRE_SPIKE_RATIO_THRESH (default 1.8).

    We analyse only the green channel (highest SNR in Bayer sensors) to
    keep the detector fast.  The notch filter in remove_moire still runs
    per-channel when a correction is needed.
    """
    h, w = crop_bgr.shape[:2]
    if h < 32 or w < 32:
        # FFT is uninformative on tiny patches
        return False, 0.0

    green = crop_bgr[:, :, 1].astype(np.float32)
    fshift = np.fft.fftshift(np.fft.fft2(green))
    magnitude = np.abs(fshift)

    crow, ccol = h // 2, w // 2
    notch_radius = min(crow, ccol) // 4   # DC exclusion radius, scales with crop
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - ccol)**2 + (Y - crow)**2)

    high_freq_zone = dist > notch_radius
    if not np.any(high_freq_zone):
        return False, 0.0

    zone_magnitudes = magnitude[high_freq_zone]
    mean_mag = float(np.mean(zone_magnitudes))
    peak_mag = float(np.max(zone_magnitudes))

    if mean_mag < 1e-6:
        return False, 0.0

    ratio = peak_mag / mean_mag
    severity = float(min(ratio / 10.0, 1.0))   # normalise to [0, 1]

    threshold = MOIRE_SPIKE_RATIO_THRESH_SCREENSHOT if is_screenshot else MOIRE_SPIKE_RATIO_THRESH
    return ratio >= threshold, severity


def detect_low_contrast(crop_bgr: np.ndarray) -> tuple[bool, float]:
    """
    Detect low contrast via RMS contrast of the grayscale crop.

    RMS contrast = std-dev of pixel intensities in [0, 255] grayscale.
    A white page with clean black text: RMS ≈ 80–120.
    A faded, washed-out crop:           RMS < CONTRAST_RMS_THRESH (18).

    Returns (detected: bool, rms: float)

    CLAHE is beneficial when the crop is low-contrast, but it can
    flatten fine texture in already well-contrasted regions.  We only
    apply it when the RMS is genuinely low.
    """
    if crop_bgr.shape[0] == 0 or crop_bgr.shape[1] == 0:
        return False, 0.0

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    rms = float(np.std(gray))

    return rms < CONTRAST_RMS_THRESH, rms


# ================================================================
# Public API
# ================================================================

def preprocess_crop(
    crop_bgr: np.ndarray,
    class_name: str,
    is_screenshot: bool = False,
) -> tuple[np.ndarray, RegionArtifactProfile]:
    """
    Apply per-region adaptive preprocessing to a single YOLO crop.

    Parameters
    ----------
    crop_bgr      : BGR uint8 numpy array (from xyxy_to_pil_crop → cv2.cvtColor)
    class_name    : YOLO class name string, e.g. "Text", "Formula", "Table"
    is_screenshot : Pass True when the source image was classified as a screenshot
                    by detect_capture_modality.  This raises the glare and moiré
                    thresholds to avoid false-positive corrections on white page
                    backgrounds and JPEG compression artefacts that mimic sensor noise.

    Returns
    -------
    corrected_bgr : BGR uint8 numpy array, with only necessary corrections
    profile       : RegionArtifactProfile recording what was detected/applied

    Pipeline (per crop, in this order):
      1. Detect glare   → conditionally inpaint
      2. Detect shadow  → conditionally apply DoG normalisation
      3. Detect moiré   → conditionally apply FFT notch filter
      4. Detect contrast → conditionally apply CLAHE
    """
    profile = RegionArtifactProfile(class_name=class_name)
    result = crop_bgr.copy()

    # --- Gate 1: Skip-all classes (Picture) -----------------------
    if class_name in _SKIP_ALL:
        return result, profile

    # --- Gate 2: Contrast-only classes (page strips) --------------
    if class_name in _CONTRAST_ONLY:
        low_c, rms = detect_low_contrast(result)
        profile.low_contrast_detected = low_c
        profile.contrast_rms = rms
        if low_c:
            result = normalize_contrast(result)
            profile.contrast_corrected = True
        return result, profile

    # --- Full detection pass for all other classes ----------------

    # Step 1: Glare
    glare_det, glare_sev = detect_glare(result, is_screenshot=is_screenshot)
    profile.glare_detected = glare_det
    profile.glare_severity = glare_sev
    if glare_det:
        result = remove_glare(result, lightness_threshold=GLARE_LIGHTNESS_THRESH)
        profile.glare_corrected = True

    # Step 2: Shadow  (skipped for Formula — DoG degrades thin strokes)
    # Also skip for screenshots — DoG flattens JPEG compression gradients
    # that look like shadows but are actually compression artifacts
    shadow_det, shadow_sev = detect_shadow(result)
    profile.shadow_detected = shadow_det
    profile.shadow_severity = shadow_sev
    if shadow_det:
        if class_name in _SKIP_SHADOW or is_screenshot:
            profile.skipped_corrections.append("shadow")
        else:
            result = remove_shadows(result)
            profile.shadow_corrected = True

    # Step 3: Moiré
    moire_det, moire_sev = detect_moire(result, is_screenshot=is_screenshot)
    profile.moire_detected = moire_det
    profile.moire_severity = moire_sev
    if moire_det:
        result = remove_moire(result)
        profile.moire_corrected = True

    # Step 4: Contrast  (run last — polish after artifact removal)
    low_c, rms = detect_low_contrast(result)
    profile.low_contrast_detected = low_c
    profile.contrast_rms = rms
    if low_c:
        result = normalize_contrast(result)
        profile.contrast_corrected = True

    return result, profile