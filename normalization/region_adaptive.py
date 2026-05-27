# normalization/region_adaptive.py
"""
Per-Region Adaptive Preprocessing — Stage 1.5
Stable Version (v3.3) — Reverted aggressive binarization and dilation.
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
# Thresholds
# ================================================================

GLARE_LIGHTNESS_THRESH: int   = 245    
GLARE_LIGHTNESS_THRESH_PHONE: int = 225  # Lowered: 240→225 to catch glare halos on phone photos
GLARE_AREA_THRESH: float      = 0.02   # Lowered: 0.05→0.02 — fire glare correction on 2% of crop pixels
GLARE_AREA_THRESH_SCREENSHOT: float = 0.25  
SHADOW_GRADIENT_THRESH: float = 0.20   # Lowered: 0.25→0.20 — more sensitive shadow detection on glared crops
MOIRE_SPIKE_RATIO_THRESH: float = 3.5  
MOIRE_SPIKE_RATIO_THRESH_SCREENSHOT: float = 8.0  
CONTRAST_RMS_THRESH: float    = 18.0   


# ================================================================
# Class-aware gating tables
# ================================================================

_SKIP_ALL: Set[str] = {"Picture"}
_SKIP_SHADOW: Set[str] = {"Formula"}
_CONTRAST_ONLY: Set[str] = {"Page-header", "Page-footer"}

# ================================================================
# Result dataclass
# ================================================================

@dataclass
class RegionArtifactProfile:
    class_name: str = ""
    glare_detected: bool = False
    glare_severity: float = 0.0
    shadow_detected: bool = False
    shadow_severity: float = 0.0
    moire_detected: bool = False
    moire_severity: float = 0.0
    low_contrast_detected: bool = False
    contrast_rms: float = 0.0
    glare_corrected: bool = False
    shadow_corrected: bool = False
    moire_corrected: bool = False
    contrast_corrected: bool = False
    skipped_corrections: list = field(default_factory=list)

    def any_detected(self) -> bool:
        return (self.glare_detected or self.shadow_detected or self.moire_detected or self.low_contrast_detected)

    def any_applied(self) -> bool:
        return (self.glare_corrected or self.shadow_corrected or self.moire_corrected or self.contrast_corrected)

    def summary(self) -> str:
        detected = []
        if self.glare_detected: detected.append(f"glare({self.glare_severity:.2f})")
        if self.shadow_detected: detected.append(f"shadow({self.shadow_severity:.2f})")
        if self.moire_detected: detected.append(f"moire({self.moire_severity:.2f})")
        if self.low_contrast_detected: detected.append(f"contrast(rms={self.contrast_rms:.1f})")
        applied = []
        if self.glare_corrected: applied.append("glare")
        if self.shadow_corrected: applied.append("shadow")
        if self.moire_corrected: applied.append("moire")
        if self.contrast_corrected: applied.append("contrast")
        if not detected: return f"[{self.class_name}] clean"
        return f"[{self.class_name}] detected: {', '.join(detected)} | applied: {', '.join(applied) if applied else 'none'}"


# ================================================================
# Artifact Detectors
# ================================================================

def detect_glare(crop_bgr: np.ndarray, is_screenshot: bool = False) -> tuple[bool, float]:
    total_pixels = crop_bgr.shape[0] * crop_bgr.shape[1]
    if total_pixels == 0: return False, 0.0
    l_thresh = GLARE_LIGHTNESS_THRESH if is_screenshot else GLARE_LIGHTNESS_THRESH_PHONE
    mask = detect_glare_mask(crop_bgr, lightness_threshold=l_thresh)
    glare_pixels = int(cv2.countNonZero(mask))
    severity = glare_pixels / total_pixels
    threshold = GLARE_AREA_THRESH_SCREENSHOT if is_screenshot else GLARE_AREA_THRESH
    return severity >= threshold, severity

def detect_shadow(crop_bgr: np.ndarray) -> tuple[bool, float]:
    if crop_bgr.shape[0] < 16 or crop_bgr.shape[1] < 16: return False, 0.0
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg = cv2.GaussianBlur(gray, (51, 51), 0)
    ratio = gray / (bg + 1e-6)
    std_dev = float(np.std(ratio))
    return std_dev >= SHADOW_GRADIENT_THRESH, float(np.tanh(std_dev / 0.5))

def detect_moire(crop_bgr: np.ndarray, is_screenshot: bool = False) -> tuple[bool, float]:
    h, w = crop_bgr.shape[:2]
    if h < 32 or w < 32: return False, 0.0
    green = crop_bgr[:, :, 1].astype(np.float32)
    fshift = np.fft.fftshift(np.fft.fft2(green))
    magnitude = np.abs(fshift)
    crow, ccol = h // 2, w // 2
    notch_radius = min(crow, ccol) // 4
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - ccol)**2 + (Y - crow)**2)
    high_freq_zone = dist > notch_radius
    if not np.any(high_freq_zone): return False, 0.0
    mean_mag = float(np.mean(magnitude[high_freq_zone]))
    peak_mag = float(np.max(magnitude[high_freq_zone]))
    ratio = peak_mag / (mean_mag + 1e-6)
    threshold = MOIRE_SPIKE_RATIO_THRESH_SCREENSHOT if is_screenshot else MOIRE_SPIKE_RATIO_THRESH
    return ratio >= threshold, float(min(ratio / 10.0, 1.0))

def detect_low_contrast(crop_bgr: np.ndarray) -> tuple[bool, float]:
    if crop_bgr.size == 0: return False, 0.0
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    rms = float(np.std(gray))
    return rms < CONTRAST_RMS_THRESH, rms


# ================================================================
# Public API
# ================================================================

def preprocess_crop(crop_bgr: np.ndarray, class_name: str, is_screenshot: bool = False) -> tuple[np.ndarray, RegionArtifactProfile]:
    profile = RegionArtifactProfile(class_name=class_name)
    result = crop_bgr.copy()
    if class_name in _SKIP_ALL: return result, profile
    
    if class_name in _CONTRAST_ONLY:
        low_c, rms = detect_low_contrast(result)
        profile.low_contrast_detected, profile.contrast_rms = low_c, rms
        if low_c:
            result = normalize_contrast(result)
            profile.contrast_corrected = True
        return result, profile

    # Step 0: Moiré / mesh-glare removal for phone photos — MUST run first.
    #
    # Phone photos taken of a screen exhibit a fine grid/crosshatch pattern
    # from the screen mesh or reflection. This pattern:
    #   a) confuses YOLO's column-gutter detector (fills the gutter with noise)
    #   b) gets bicubic-interpolated into letter strokes during the 2.6× upscale
    #   c) is then sharpened by the UnsharpMask in models_interface, making
    #      word-boundary destruction worse.
    #
    # For screenshots this is a non-issue (no physical mesh) — the higher
    # MOIRE_SPIKE_RATIO_THRESH_SCREENSHOT = 8.0 prevents false fires.
    # For phone photos MOIRE_SPIKE_RATIO_THRESH = 3.5 catches real screen-mesh
    # patterns (typical ratio 4–15×) while ignoring JPEG ring artefacts (2–3×).
    moire_det, moire_sev = detect_moire(result, is_screenshot=is_screenshot)
    profile.moire_detected, profile.moire_severity = moire_det, moire_sev
    if moire_det:
        result = remove_moire(result)
        profile.moire_corrected = True

    # Step 1: Glare
    glare_det, glare_sev = detect_glare(result, is_screenshot=is_screenshot)
    profile.glare_detected, profile.glare_severity = glare_det, glare_sev
    if glare_det:
        l_thresh = GLARE_LIGHTNESS_THRESH if is_screenshot else GLARE_LIGHTNESS_THRESH_PHONE
        result = remove_glare(result, lightness_threshold=l_thresh)
        profile.glare_corrected = True
        # Heavy glare (>15% of crop) co-occurs with illumination halos.
        # Run shadow removal on the inpainted result to even out the halo
        # gradient left behind after inpainting, but skip for screenshots.
        if not is_screenshot and glare_sev > 0.15 and class_name not in _SKIP_SHADOW:
            shadow_det, shadow_sev = detect_shadow(result)
            profile.shadow_detected, profile.shadow_severity = shadow_det, shadow_sev
            if shadow_det:
                result = remove_shadows(result)
                profile.shadow_corrected = True

    # Step 2: Shadow (Stage 1 handles it globally for normal photos)
    # Step 3: Moiré — handled above as Step 0 (per-crop, before glare)

    # Step 4: Contrast (Safer CLAHE only)
    low_c, rms = detect_low_contrast(result)
    profile.low_contrast_detected, profile.contrast_rms = low_c, rms
    if low_c:
        result = normalize_contrast(result)
        profile.contrast_corrected = True

    return result, profile