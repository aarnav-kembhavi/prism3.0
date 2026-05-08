"""
layout_utils.py
---------------
Bounding box utilities: sorting and reading order.
Supports single-column, multi-column, and full-width elements using line-clustering and gutter detection.
"""

from typing import List, Dict, Any


def sort_detections_reading_order(
    detections: List[Dict[str, Any]],
    image_width: int = 0
) -> List[Dict[str, Any]]:
    """
    Sort YOLO detections into correct reading order using Y-axis tolerance (line clustering).
    
    Groups bounding boxes into horizontal lines based on Y-overlap. Within each line, 
    sorts boxes left-to-right. This handles cases like resumes where a left-aligned 
    title and a right-aligned date exist on the same line.
    """
    if not detections:
        return []

    # Sort primarily by the top Y coordinate
    sorted_dets = sorted(detections, key=lambda d: d['bbox'][1])

    lines = []
    current_line = []
    current_line_y_max = -1
    current_line_y_min = float('inf')

    for det in sorted_dets:
        x1, y1, x2, y2 = det['bbox']
        
        if not current_line:
            current_line.append(det)
            current_line_y_max = y2
            current_line_y_min = y1
        else:
            cy = (y1 + y2) / 2
            # Check overlap. We consider it the same line if the Y-center of the box
            # falls within the Y-bounds of the current line, or if there is significant overlap.
            if current_line_y_min <= cy <= current_line_y_max:
                current_line.append(det)
                current_line_y_max = max(current_line_y_max, y2)
                current_line_y_min = min(current_line_y_min, y1)
            else:
                lines.append(current_line)
                current_line = [det]
                current_line_y_max = y2
                current_line_y_min = y1

    if current_line:
        lines.append(current_line)

    final_sorted = []
    for line in lines:
        # Sort left-to-right within the line
        line.sort(key=lambda d: d['bbox'][0])
        final_sorted.extend(line)

    return final_sorted


def xyxy_to_pil_crop(image, bbox):
    """
    Crop a PIL image using xyxy bbox coordinates.
    Clamps to image boundaries to prevent out-of-bounds errors.
    """
    from PIL import Image
    w, h = image.size
    x1, y1, x2, y2 = bbox
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))
    return image.crop((x1, y1, x2, y2))


def detect_column_count(detections: List[Dict[str, Any]], image_width: int) -> int:
    """
    Heuristic: detect if page is single or double column using Gutter Detection.

    Checks for a vertical corridor (gutter) near the middle of the page that is 
    completely empty of bounding boxes.
    Returns 2 if a clear gutter exists, otherwise 1.
    """
    if image_width == 0 or not detections:
        return 1

    full_width_threshold = image_width * 0.60
    
    # Filter out full-width elements and get X-spans of the rest
    non_full_spans = []
    for d in detections:
        x1, _, x2, _ = d['bbox']
        if (x2 - x1) < full_width_threshold:
            non_full_spans.append((x1, x2))

    if not non_full_spans:
        return 1

    # Define a target gutter zone in the middle of the page (e.g. 40% to 60%)
    gutter_center = image_width / 2
    gutter_min = gutter_center - (image_width * 0.05)
    gutter_max = gutter_center + (image_width * 0.05)

    # Check if any bounding box prominently crosses this true middle gutter area
    crosses_gutter_count = 0
    left_items = 0
    right_items = 0

    for x1, x2 in non_full_spans:
        if x1 < gutter_min and x2 > gutter_max:
            crosses_gutter_count += 1
        elif x2 <= gutter_center:
            left_items += 1
        elif x1 >= gutter_center:
            right_items += 1

    # A two-column document should have almost no elements crossing the exact center,
    # and should have substantial items on both left and right sides.
    # Allowing up to 2 exceptions (like small misaligned blocks) to cross the gutter.
    if crosses_gutter_count <= 2 and left_items >= 3 and right_items >= 3:
        return 2

    return 1


def split_detections_by_column(
    detections: List[Dict[str, Any]],
    image_width: int
) -> tuple:
    """
    Split detections into full_width, left_col, right_col lists.
    Returns (full_width, left_col, right_col) each sorted by Y (line-clustered).
    """
    if image_width == 0:
        return [], sort_detections_reading_order(detections, image_width), []

    full_width_threshold = image_width * 0.60
    midpoint = image_width / 2

    full_width, left_col, right_col = [], [], []

    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        w = x2 - x1
        
        # Consider an element full_width if it spans significantly across the middle
        if w >= full_width_threshold or (x1 < midpoint - image_width*0.1 and x2 > midpoint + image_width*0.1):
            full_width.append(det)
        elif (x1 + x2) / 2 < midpoint:
            left_col.append(det)
        else:
            right_col.append(det)

    # Sort each zone properly
    full_width = sort_detections_reading_order(full_width, image_width)
    left_col = sort_detections_reading_order(left_col, image_width)
    right_col = sort_detections_reading_order(right_col, image_width)

    return full_width, left_col, right_col
