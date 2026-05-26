# normalization/frequency_filter.py
"""
Image artifact removal: glare inpainting, moiré suppression, shadow removal,
and white balance correction.

All functions operate on BGR uint8 numpy arrays (OpenCV convention).
"""

import cv2
import numpy as np
from scipy import ndimage


# ----------------------------------------------------------------
# Glare removal — LAB thresholding + inpainting
# ----------------------------------------------------------------

def detect_glare_mask(image_bgr, lightness_threshold=245, min_area=100):
    """
    Detect glare/specular highlight regions using LAB L-channel.

    Glare spots are saturated-white patches with very high luminance.
    We threshold the L-channel, then clean up with morphological ops.

    Returns: binary mask (uint8, 255=glare, 0=normal)
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]

    # Threshold: L > threshold → glare. 
    # Use 245 to only target pure blown-out white where text is unrecoverable.
    _, glare_mask = cv2.threshold(
        l_channel, lightness_threshold, 255, cv2.THRESH_BINARY
    )

    # FIX: Shrink the kernel to 3x3 so we don't swallow thin text strokes!
    # Remove MORPH_CLOSE as it destroys text structure inside the glare.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    glare_mask = cv2.dilate(glare_mask, kernel, iterations=1)

    # Remove tiny detections (noise) — keep only substantial glare regions
    contours, _ = cv2.findContours(
        glare_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cleaned = np.zeros_like(glare_mask)
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(cleaned, [c], -1, 255, -1)

    return cleaned


def remove_glare(image_bgr, lightness_threshold=245, inpaint_radius=3):
    """
    Remove glare by detecting bright spots and inpainting them.

    Two-phase approach:
    1. Detect glare mask via LAB L-channel thresholding
    2. Inpaint the masked regions using Telea's algorithm
    """
    glare_mask = detect_glare_mask(image_bgr, lightness_threshold)

    glare_pixels = cv2.countNonZero(glare_mask)
    total_pixels = image_bgr.shape[0] * image_bgr.shape[1]
    glare_pct = glare_pixels / total_pixels * 100

    if glare_pct < 0.1:
        # No significant glare detected
        print(f"  [norm] No significant glare detected (<0.1%)")
        return image_bgr

    print(f"  [norm] Glare detected: {glare_pct:.1f}% of image, inpainting...")

    # FIX: Reduce inpaint radius to 3 to prevent wide smearing over text
    result = cv2.inpaint(image_bgr, glare_mask, inpaint_radius, cv2.INPAINT_TELEA)

    return result


# ----------------------------------------------------------------
# Contrast normalization — CLAHE (post-glare-removal polish)
# ----------------------------------------------------------------

def normalize_contrast(image_bgr, clip_limit=2.0, tile_grid=(8, 8)):
    """
    Normalize contrast using CLAHE on the L-channel of LAB space.
    Applied AFTER glare removal as a final polish step.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_eq = clahe.apply(l)

    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


# ----------------------------------------------------------------
# Shadow removal — difference-of-Gaussians
# ----------------------------------------------------------------

def remove_shadows(image_bgr, blur_large=51, blur_small=5):
    """
    Remove uneven shadows using difference-of-Gaussians (DoG).

    Idea: a heavily blurred version of the image captures the
    low-frequency lighting pattern (shadows). Dividing the original
    by this estimate normalizes the illumination.

    This is very effective for phone photos with desk lamp shadows,
    window light gradients, etc.
    """
    # Work in float to avoid overflow
    img_float = image_bgr.astype(np.float32)

    # Ensure kernel sizes are odd
    blur_large = blur_large | 1
    blur_small = blur_small | 1

    # Heavy blur captures the illumination pattern
    bg = cv2.GaussianBlur(img_float, (blur_large, blur_large), 0)

    # Divide original by background illumination, rescale to [0, 255]
    # Add small epsilon to avoid division by zero
    normalized = (img_float / (bg + 1e-6)) * 128.0

    # Light blur to smooth any artifacts
    normalized = cv2.GaussianBlur(normalized, (blur_small, blur_small), 0)

    # Clip and convert back
    result = np.clip(normalized, 0, 255).astype(np.uint8)

    return result


# ----------------------------------------------------------------
# White balance — gray world algorithm
# ----------------------------------------------------------------

def white_balance_gray_world(image_bgr):
    """
    Apply gray world white balance correction.

    Assumption: the average color of a scene should be gray.
    Phone cameras often produce warm/cool color casts under
    indoor lighting; this corrects for that.
    """
    img_float = image_bgr.astype(np.float32)

    # Compute mean of each channel
    mean_b = np.mean(img_float[:, :, 0])
    mean_g = np.mean(img_float[:, :, 1])
    mean_r = np.mean(img_float[:, :, 2])

    # Compute global mean
    global_mean = (mean_b + mean_g + mean_r) / 3.0

    # Scale each channel to match the global mean
    if mean_b > 0:
        img_float[:, :, 0] *= global_mean / mean_b
    if mean_g > 0:
        img_float[:, :, 1] *= global_mean / mean_g
    if mean_r > 0:
        img_float[:, :, 2] *= global_mean / mean_r

    return np.clip(img_float, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------
# Moiré removal — FFT notch filtering (improved)
# ----------------------------------------------------------------

def remove_moire(image_bgr, notch_radius=30, threshold_percentile=97):
    """
    Remove moiré patterns using FFT notch filtering.
    Works channel-by-channel on BGR image.

    Improved: uses a gentler Gaussian-smoothed mask to avoid
    ringing artifacts from hard frequency cutoffs.
    """
    result = np.zeros_like(image_bgr, dtype=np.float32)

    for ch in range(3):
        channel = image_bgr[:, :, ch].astype(np.float32)

        f = np.fft.fft2(channel)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)

        rows, cols = channel.shape
        crow, ccol = rows // 2, cols // 2

        # Distance grid from DC center
        Y, X = np.ogrid[:rows, :cols]
        dist_from_center = np.sqrt((X - ccol)**2 + (Y - crow)**2)

        # Only look at frequencies beyond the notch radius
        mask_region = dist_from_center > notch_radius
        if not np.any(mask_region):
            result[:, :, ch] = channel
            continue

        threshold = np.percentile(magnitude[mask_region], threshold_percentile)

        # Create notch mask
        notch_mask = np.ones((rows, cols), dtype=np.float32)
        spike_locs = (magnitude > threshold) & mask_region
        notch_mask[spike_locs] = 0

        # Smooth mask to reduce ringing
        notch_mask = ndimage.gaussian_filter(notch_mask, sigma=3)

        # Apply and transform back
        fshift_filtered = fshift * notch_mask
        f_back = np.fft.ifftshift(fshift_filtered)
        img_back = np.fft.ifft2(f_back)
        result[:, :, ch] = np.real(img_back)

    return np.clip(result, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------
# Legacy aliases for backward compatibility
# ----------------------------------------------------------------

def suppress_glare(image_bgr, clip_limit=2.0, tile_grid=(8, 8)):
    """Legacy alias — calls remove_glare + normalize_contrast."""
    result = remove_glare(image_bgr)
    return normalize_contrast(result, clip_limit, tile_grid)