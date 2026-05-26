# =============================================================================
# preprocess.py — Screenshot preprocessing pipeline
#
# UPDATED: Removed destructive denoising and harsh binarization.
# Modern OCR engines (EasyOCR/PaddleOCR) perform their own internal thresholding.
# Pre-binarizing or aggressive denoising shreds thin text and causes hallucinations.
# =============================================================================

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from config import CONFIG


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Applies safe, non-destructive preprocessing steps to a crop
    before passing it to EasyOCR or TATR.
    """
    result = img.copy()

    # 1. Deskew (Safe: fixes slight camera rotations)
    if CONFIG.get("preprocess_deskew", False):
        result = _deskew(result)

    # 2. Gentle Contrast Boost (Safe: helps OCR separate text from background)
    if CONFIG.get("preprocess_contrast", False):
        pil      = Image.fromarray(result)
        # Lowered factor to 1.2 to prevent blowing out the text
        factor   = float(CONFIG.get("preprocess_contrast_factor", 1.2))
        pil      = ImageEnhance.Contrast(pil).enhance(factor)
        result   = np.array(pil)

    # 3. DISABLED: Denoise, Binarize, Sharpen
    # We explicitly skip cv2.fastNlMeansDenoisingColored and cv2.adaptiveThreshold
    # because they destroy the continuous-tone gradients that OCR models rely on.

    return result


def _deskew(img: np.ndarray) -> np.ndarray:
    """
    Detects skew angle using Hough line transform and corrects it.
    Only corrects angles between 0.5° and 10° to avoid over-rotation
    on naturally tilted content like italics.
    """
    gray   = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges  = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines  = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    if lines is None:
        return img

    angles = []
    for line in lines[:20]:
        rho, theta = line[0]
        angle = (theta * 180 / np.pi) - 90
        if 0.5 < abs(angle) < 10:
            angles.append(angle)

    if not angles:
        return img

    median_angle = float(np.median(angles))
    h, w         = img.shape[:2]
    M            = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
    
    # Use border replicate so we don't introduce hard black triangles at the edges
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )