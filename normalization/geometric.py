# normalization/geometric.py
"""
Multi-strategy document boundary detection and perspective rectification.

Strategies tried in order (first success wins):
  1. Morphological gradient + adaptive threshold + largest quadrilateral contour
  2. Hough line detection → intersect to find 4 corners
  3. Classic Canny + contour (original approach)
  4. Fallback: return the original image unchanged

A detection is accepted only if the quadrilateral area covers ≥15% of the
image — this prevents tiny false-positive contours from triggering a warp.
"""

import cv2
import numpy as np


# ----------------------------------------------------------------
# Low-level helpers
# ----------------------------------------------------------------

def order_points(pts):
    """Order 4 corner points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left has smallest sum
    rect[2] = pts[np.argmax(s)]   # bottom-right has largest sum
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def four_point_transform(image, pts):
    """Apply perspective warp using 4 corner points."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def _quad_area(pts):
    """Compute area of a quadrilateral given 4 points (shoelace formula)."""
    ordered = order_points(pts)
    n = len(ordered)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += ordered[i][0] * ordered[j][1]
        area -= ordered[j][0] * ordered[i][1]
    return abs(area) / 2.0


def _is_valid_quad(pts, img_area, min_ratio=0.15):
    """Check if a quadrilateral is large enough relative to the image."""
    return _quad_area(pts) >= img_area * min_ratio


def _find_largest_quad_contour(binary_img, img_area):
    """
    Find the largest quadrilateral contour in a binary image.
    Returns the 4 corner points or None.
    """
    contours, _ = cv2.findContours(
        binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # Sort by area descending
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype("float32")
            if _is_valid_quad(pts, img_area):
                return pts
    return None


# ----------------------------------------------------------------
# Strategy 1: Morphological gradient approach
# Better under uneven lighting / glare because adaptive threshold
# handles local brightness variations.
# ----------------------------------------------------------------

def _strategy_morph_gradient(gray, img_area):
    """Morphological gradient + adaptive threshold + contour."""
    # Morphological gradient highlights edges regardless of lighting
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

    # Adaptive threshold handles uneven illumination
    binary = cv2.adaptiveThreshold(
        gradient, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11, C=-2
    )

    # Close gaps in edges
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

    return _find_largest_quad_contour(binary, img_area)


# ----------------------------------------------------------------
# Strategy 2: Hough lines → intersections
# Works when contour detection fails because edges are broken
# by glare / shadows, but individual line segments are detectable.
# ----------------------------------------------------------------

def _line_intersection(line1, line2):
    """Compute intersection of two lines in (rho, theta) form."""
    rho1, theta1 = line1
    rho2, theta2 = line2
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([rho1, rho2])
    det = np.linalg.det(A)
    if abs(det) < 1e-6:
        return None  # parallel lines
    x, y = np.linalg.solve(A, b)
    return (x, y)


def _cluster_lines(lines, angle_threshold=15, dist_threshold=50):
    """
    Cluster similar Hough lines.
    Returns representative (rho, theta) for each cluster.
    """
    if lines is None or len(lines) == 0:
        return []

    angle_thresh_rad = np.radians(angle_threshold)
    used = [False] * len(lines)
    clusters = []

    for i in range(len(lines)):
        if used[i]:
            continue
        rho_i, theta_i = lines[i][0]
        cluster = [(rho_i, theta_i)]
        used[i] = True

        for j in range(i + 1, len(lines)):
            if used[j]:
                continue
            rho_j, theta_j = lines[j][0]
            # Check angular and distance similarity
            angle_diff = abs(theta_i - theta_j)
            dist_diff = abs(rho_i - rho_j)
            if angle_diff < angle_thresh_rad and dist_diff < dist_threshold:
                cluster.append((rho_j, theta_j))
                used[j] = True

        # Average the cluster
        avg_rho = np.mean([c[0] for c in cluster])
        avg_theta = np.mean([c[1] for c in cluster])
        clusters.append((avg_rho, avg_theta))

    return clusters


def _strategy_hough_lines(gray, img_area):
    """Hough line detection → intersect to find 4 corners."""
    h, w = gray.shape[:2]

    # Edge detection with multiple thresholds for robustness
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Dilate edges to connect broken segments
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Detect lines
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
    if lines is None or len(lines) < 4:
        return None

    # Cluster similar lines
    clustered = _cluster_lines(lines)
    if len(clustered) < 4:
        return None

    # Separate into roughly horizontal and roughly vertical lines
    horizontal = []
    vertical = []
    for rho, theta in clustered:
        angle_deg = np.degrees(theta)
        if 45 < angle_deg < 135:
            horizontal.append((rho, theta))
        else:
            vertical.append((rho, theta))

    if len(horizontal) < 2 or len(vertical) < 2:
        return None

    # Take the two most extreme lines in each direction
    horizontal.sort(key=lambda l: l[0])
    vertical.sort(key=lambda l: l[0])
    h_lines = [horizontal[0], horizontal[-1]]
    v_lines = [vertical[0], vertical[-1]]

    # Find 4 intersections
    corners = []
    for hl in h_lines:
        for vl in v_lines:
            pt = _line_intersection(hl, vl)
            if pt is not None:
                x, y = pt
                # Must be within image bounds (with margin)
                margin = -50  # allow slightly outside
                if margin <= x <= w - margin and margin <= y <= h - margin:
                    corners.append(pt)

    if len(corners) != 4:
        return None

    pts = np.array(corners, dtype="float32")
    if _is_valid_quad(pts, img_area):
        return pts
    return None


# ----------------------------------------------------------------
# Strategy 3: Classic Canny + contour (original approach)
# Works well on clean images with clear document edges.
# ----------------------------------------------------------------

def _strategy_canny_contour(gray, img_area):
    """Classic Canny edge detection + contour finding."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    contours, _ = cv2.findContours(
        edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype("float32")
            if _is_valid_quad(pts, img_area):
                return pts
    return None


# ----------------------------------------------------------------
# Skew detection and correction (projection profile method)
# ----------------------------------------------------------------

def detect_skew_angle(gray, max_angle: float = 15.0, angle_step: float = 0.5) -> float:
    """
    Estimate in-plane rotation angle using the projection profile method.

    Rotates a downsampled binary image across candidate angles and picks
    the angle whose horizontal row-sum histogram has the highest variance
    (text lines are sharpest when perfectly horizontal).

    Handles dark-border padding (from augmentation) by cropping to the
    bright document region before thresholding, so border pixels don't
    swamp the text signal.

    Returns the detected skew angle in degrees (positive = CCW in OpenCV).
    Returns 0.0 if the image appears straight or detection is unreliable.
    """
    # Downsample for speed — 600px on the longer side is plenty
    h, w = gray.shape[:2]
    scale = min(1.0, 600.0 / max(h, w))
    small = cv2.resize(gray, (int(w * scale), int(h * scale)),
                       interpolation=cv2.INTER_AREA) if scale < 1.0 else gray.copy()

    # Crop to the bright document area to exclude dark border padding.
    # Padding like (40,40,40) would swamp Otsu; the document background is > 150.
    bright = (small > 150).astype(np.uint8)
    ys, xs = np.where(bright)
    if len(ys) < 500:
        return 0.0
    margin = 5
    y1 = max(int(ys.min()) + margin, 0)
    y2 = min(int(ys.max()) - margin, small.shape[0] - 1)
    x1 = max(int(xs.min()) + margin, 0)
    x2 = min(int(xs.max()) - margin, small.shape[1] - 1)
    crop = small[y1:y2 + 1, x1:x2 + 1]
    if crop.size < 1000:
        return 0.0

    # Otsu on the clean document region: text → 255, white background → 0
    _, binary = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if binary.astype(bool).mean() > 0.5:
        # Inverted image (dark background): flip so text = 255
        binary = cv2.bitwise_not(binary)

    sh, sw = binary.shape[:2]
    center = (sw // 2, sh // 2)
    best_angle, best_var = 0.0, -1.0

    for angle in np.arange(-max_angle, max_angle + angle_step, angle_step):
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(binary, M, (sw, sh),
                                 flags=cv2.INTER_NEAREST, borderValue=0)
        row_sums = rotated.sum(axis=1).astype(np.float64)
        # Derivative variance: measures sharpness of text-line transitions.
        # More robust than raw row-sum variance which is dominated by margins.
        var = float(np.diff(row_sums).var())
        if var > best_var:
            best_var, best_angle = var, float(angle)

    return best_angle


def deskew(img: np.ndarray, max_angle: float = 15.0, angle_step: float = 0.5,
           min_correction: float = 0.5) -> np.ndarray:
    """
    Detect and correct document skew using projection profile analysis.

    Corrects in-plane rotation up to `max_angle` degrees.
    Skips correction if the detected angle is below `min_correction`.
    Fills the background with white (255) — appropriate for scanned documents.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    angle = detect_skew_angle(gray, max_angle=max_angle, angle_step=angle_step)

    if abs(angle) < min_correction:
        return img

    print(f"  [norm] Deskew: detected {angle:+.1f}° skew, correcting")
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Expand canvas so corners aren't clipped
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    corrected = cv2.warpAffine(img, M, (new_w, new_h),
                               flags=cv2.INTER_LINEAR,
                               borderValue=(255, 255, 255))
    return corrected


# ----------------------------------------------------------------
# Public API
# ----------------------------------------------------------------

def detect_and_rectify(image_path, img_override=None):
    """
    Multi-strategy auto-rectification pipeline.
    Tries 3 strategies in order and falls back to returning the
    original image if none succeed.

    Args:
        image_path: Path to the image file (used if img_override is None)
        img_override: Optional pre-loaded BGR numpy array. If provided,
                      this image is used instead of reading from image_path.
                      Useful when upstream steps (e.g. white balance) have
                      already modified the image.
    """
    if img_override is not None:
        img = img_override
    else:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    img_area = h * w

    strategies = [
        ("morphological gradient", _strategy_morph_gradient),
        ("Hough line intersection", _strategy_hough_lines),
        ("Canny contour",          _strategy_canny_contour),
    ]

    for name, strategy_fn in strategies:
        try:
            pts = strategy_fn(gray, img_area)
            if pts is not None:
                print(f"  [norm] Document detected via: {name}")
                warped = four_point_transform(img, pts)
                # Sanity check: warped should not be drastically smaller
                wh, ww = warped.shape[:2]
                if wh * ww >= img_area * 0.10:
                    return warped
                else:
                    print(f"  [norm] Warning: {name} produced too-small result, trying next...")
        except Exception as e:
            print(f"  [norm] Warning: {name} failed ({e}), trying next...")
            continue

    print("  [norm] No document boundary found by any strategy. Using original image.")
    return img
