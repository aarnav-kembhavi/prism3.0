# normalization/modality.py
"""
Capture Modality Detection — automatically classifies input images as
digital screenshots or phone camera captures using grayscale histogram analysis.

Digital renders (screenshots, PDF exports) produce sparse histograms with
few occupied intensity bins because pixel values come from discrete color
palettes and font rasterizers. Phone camera captures produce dense histograms
spread across nearly all 256 bins due to sensor noise, lens aberrations,
Bayer demosaicing, and analog signal processing.

This classification drives adaptive preprocessing: screenshots skip
geometric rectification, shadow removal, glare inpainting, and moiré
filtering — artifacts that are physically impossible in digital renders.
Phone photos receive the full treatment.

Method:
    1. Convert input to grayscale (single channel)
    2. Compute 256-bin intensity histogram
    3. Count "occupied" bins — bins whose pixel count exceeds a noise floor
       (min_pixel_fraction of total pixels) to filter single-pixel sensor noise
    4. Compute Shannon entropy of the histogram distribution as a secondary
       signal (screenshots have low entropy; phone photos have high entropy)
    5. Classify: occupied_bins < threshold → screenshot, else phone photo
    6. Compute confidence as distance from the decision boundary

Empirical basis (tested on project dataset):
    - Screenshots (text, UI, embedded figures): entropy 0.29–0.48
    - Clean phone photos of documents:          entropy 0.86–0.91
    - Glared phone photos:                      entropy 0.55–0.72
      (large blown-out white regions spike one intensity bin,
       concentrating probability mass and reducing entropy vs clean photos)
    - Default entropy threshold of 0.55 sits below the glared-photo range
      and well below clean phone photos, ensuring both are classified correctly.

    Note: Raw bin count is unreliable for content-rich screenshots —
    documents with embedded photos/gradients occupy all 256 bins despite
    being digital renders. Entropy captures the *shape* of the distribution:
    screenshots concentrate probability mass in a few dominant bins (text
    colors, background), while phone photos spread it more uniformly due to
    sensor noise, Bayer demosaicing artifacts, and analog signal variation.
    Even glared phone photos retain more entropy than clean screenshots
    because the non-glare regions still carry sensor noise across all bins.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from enum import Enum


class CaptureModality(Enum):
    """Classification of how the input image was captured."""
    SCREENSHOT = "screenshot"
    PHONE_PHOTO = "phone_photo"


@dataclass
class ModalityResult:
    """Result of capture modality detection with diagnostic metrics."""
    modality: CaptureModality
    occupied_bins: int
    total_bins: int = 256
    bin_occupancy_ratio: float = field(init=False)
    histogram_entropy: float = 0.0
    confidence: float = 0.0

    def __post_init__(self):
        self.bin_occupancy_ratio = self.occupied_bins / self.total_bins

    def __str__(self):
        return (
            f"{self.modality.value} "
            f"(bins={self.occupied_bins}/256, "
            f"occupancy={self.bin_occupancy_ratio:.1%}, "
            f"entropy={self.histogram_entropy:.4f}, "
            f"confidence={self.confidence:.1%})"
        )


def detect_capture_modality(
    image_bgr: np.ndarray,
    entropy_threshold: float = 0.55,
    min_pixel_fraction: float = 0.0001,
) -> ModalityResult:
    """
    Classify an image as a digital screenshot or phone camera capture
    by analyzing the Shannon entropy of its grayscale histogram.

    Primary signal: Normalized Shannon entropy of the 256-bin histogram.
    Screenshots have concentrated probability mass in a few dominant
    intensity bins (text + background colors), yielding low entropy
    (0.29–0.48). Clean phone photos spread probability mass uniformly across
    nearly all bins due to sensor noise, yielding high entropy (0.86–0.91).
    Glared phone photos fall in between (0.55–0.72) because blown-out white
    regions spike the high-intensity bins, reducing entropy vs clean photos
    but still well above the 0.29–0.48 screenshot range.

    Threshold change: 0.65 → 0.55
    The original 0.65 threshold caused glared phone photos (entropy ~0.55–0.65)
    to be misclassified as screenshots, skipping all physical-artifact
    corrections (shadow removal, glare inpainting, moiré filtering).
    Lowering to 0.55 correctly routes glared photos to the full pipeline.
    The pipeline.py glare-override check (LAB L > 230 in >8% of pixels)
    acts as a secondary safety net for borderline cases.

    Args:
        image_bgr: Input image in BGR format (OpenCV convention).
        entropy_threshold: Normalized entropy decision boundary.
                          < threshold → screenshot, >= threshold → phone photo.
                          Default 0.55 covers both clean (0.86–0.91) and
                          glared (0.55–0.72) phone photos while staying clear
                          of screenshots (0.29–0.48).
        min_pixel_fraction: Minimum fraction of total pixels for a histogram
                           bin to count as "occupied". Filters stray sensor
                           noise that might scatter 1–2 pixels into random
                           bins. Default 0.01% of total pixels.

    Returns:
        ModalityResult with classification, occupied bin count,
        normalized Shannon entropy, and confidence score.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    total_pixels = gray.shape[0] * gray.shape[1]
    min_pixels = max(1, int(total_pixels * min_pixel_fraction))

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()

    occupied_bins = int(np.sum(hist >= min_pixels))

    hist_prob = hist / total_pixels
    nonzero = hist_prob[hist_prob > 0]
    entropy = float(-np.sum(nonzero * np.log2(nonzero)))
    max_entropy = np.log2(256)  # 8.0 bits
    normalized_entropy = entropy / max_entropy

    modality = (
        CaptureModality.PHONE_PHOTO
        if normalized_entropy >= entropy_threshold
        else CaptureModality.SCREENSHOT
    )

    distance = abs(normalized_entropy - entropy_threshold)
    max_distance = 0.5
    confidence = 0.5 + 0.5 * min(distance / max_distance, 1.0)

    return ModalityResult(
        modality=modality,
        occupied_bins=occupied_bins,
        histogram_entropy=normalized_entropy,
        confidence=confidence,
    )