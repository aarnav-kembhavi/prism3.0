"""
normalization — Stage 1: Intelligent Image Normalization
Geometric rectification, moiré removal, glare suppression, DPI scaling.
Includes automatic capture modality detection (screenshot vs. phone photo).
"""

from .pipeline import normalize_image, normalize_image_pil
from .modality import detect_capture_modality, CaptureModality, ModalityResult
