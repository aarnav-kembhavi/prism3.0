"""
layout_utils.py
---------------
Bounding box utilities: sorting and reading order.
Supports single-column, multi-column, and full-width elements using
line-clustering and gutter detection.
"""

from typing import List, Dict, Any


def apply_semantic_reading_order(
    detections: List[Dict[str, Any]],
    image_width: int,
    image_height: int
) -> List[Dict[str, Any]]:
    """
    Build a semantic DAG to determine reading order.

    Rules:
    1. Baseline: Geometric order (Top-to-Bottom, Left-to-Right) is used as tie-breaker.
    2. Caption Pairing: Captions are tied to the nearest Picture or Table.
    3. Footnote Sinking: Footnotes always follow body text.
    """
    if not detections:
        return []

    sorted_dets = sort_detections_geometric(detections)

    adj = {i: set() for i in range(len(sorted_dets))}
    in_degree = {i: 0 for i in range(len(sorted_dets))}

    def add_edge(u, v):
        if v not in adj[u]:
            adj[u].add(v)
            in_degree[v] += 1

    # Rule B: Caption Pairing
    for i, det in enumerate(sorted_dets):
        if det['class_name'] == 'Caption':
            cx1, cy1, cx2, cy2 = det['bbox']
            best_parent = None
            min_dist = float('inf')

            for j, potential in enumerate(sorted_dets):
                if potential['class_name'] in {'Picture', 'Table'}:
                    px1, py1, px2, py2 = potential['bbox']
                    dist = min(abs(cy1 - py2), abs(cy2 - py1))
                    h_overlap = max(0, min(cx2, px2) - max(cx1, px1))

                    if dist < 150 and (h_overlap > 0 or dist < 50):
                        if dist < min_dist:
                            min_dist = dist
                            best_parent = j

            if best_parent is not None:
                add_edge(best_parent, i)

    # Rule C: Footnote Sinking
    footnotes = [i for i, d in enumerate(sorted_dets) if d['class_name'] == 'Footnote']
    non_footnotes = [i for i, d in enumerate(sorted_dets) if d['class_name'] != 'Footnote']

    for fn in footnotes:
        for nfn in non_footnotes:
            add_edge(nfn, fn)

    # Topological Sort (Kahn's Algorithm)
    queue = [i for i in range(len(sorted_dets)) if in_degree[i] == 0]
    result_indices = []

    while queue:
        queue.sort()
        u = queue.pop(0)
        result_indices.append(u)

        for v in list(adj[u]):
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    if len(result_indices) < len(sorted_dets):
        missing = [i for i in range(len(sorted_dets)) if i not in result_indices]
        result_indices.extend(missing)

    return [sorted_dets[i] for i in result_indices]


def sort_detections_geometric(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Robust geometric reading order (T-B, L-R) with line clustering.

    Fix: changed line membership test from center-Y-in-range to
    bbox-overlap-fraction. The original test:
        current_line_y_min <= cy <= current_line_y_max
    failed when a tall element (e.g. a figure) sat next to a short element
    (e.g. a section header) at the same vertical position — the tall element's
    center Y fell below the short element's y_max, incorrectly starting a new
    line and breaking the row grouping.

    New test: an element joins the current line if its bbox overlaps the
    line's accumulated y-range by more than 30% of the element's own height.
    This is robust to mixed-height elements in the same visual row.
    """
    if not detections:
        return []

    sorted_dets = sorted(detections, key=lambda d: d['bbox'][1])

    lines = []
    current_line = []
    current_line_y_max = -1
    current_line_y_min = float('inf')

    for det in sorted_dets:
        x1, y1, x2, y2 = det['bbox']
        det_height = max(y2 - y1, 1)

        if not current_line:
            current_line.append(det)
            current_line_y_max = y2
            current_line_y_min = y1
        else:
            # Overlap between this element and the current line's y-range
            overlap_top = max(y1, current_line_y_min)
            overlap_bot = min(y2, current_line_y_max)
            overlap = max(0, overlap_bot - overlap_top)
            overlap_fraction = overlap / det_height

            if overlap_fraction >= 0.30:
                # Enough vertical overlap → same line
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
        line.sort(key=lambda d: d['bbox'][0])
        final_sorted.extend(line)

    return final_sorted


def xyxy_to_pil_crop(image, bbox):
    """
    Crop a PIL image using xyxy bbox coordinates.
    """
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

    Changes vs original:
    - full_width_threshold raised 0.60 → 0.70: items that span 60–70% of
      the page are typically wide single-column elements, not full-width spans.
      Using 0.60 caused left-column figures (~45% wide) that YOLO over-expanded
      to be counted as full-width, reducing non_full_spans and masking the
      two-column signal.
    - Gutter zone widened from ±5% to ±8% of image width (16% total):
      phone photos with slight perspective warp produce YOLO bboxes that
      don't align perfectly to the optical center. The narrow ±5% zone caused
      gutter-adjacent items to be counted as "crosses_gutter" instead of
      left_items / right_items, suppressing the two-column detection.
    """
    if image_width == 0 or not detections:
        return 1

    full_width_threshold = image_width * 0.70   # raised from 0.60

    non_full_spans = []
    for d in detections:
        x1, _, x2, _ = d['bbox']
        if (x2 - x1) < full_width_threshold:
            non_full_spans.append((x1, x2))

    if not non_full_spans:
        return 1

    gutter_center = image_width / 2
    gutter_min = gutter_center - (image_width * 0.08)  # widened from 0.05
    gutter_max = gutter_center + (image_width * 0.08)  # widened from 0.05

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

    if crosses_gutter_count <= 2 and left_items >= 3 and right_items >= 3:
        return 2

    return 1


def _is_y_monotonic(
    dets: List[Dict[str, Any]],
    back_jump_threshold: float = 0.25,
    min_back_jump_px: float = 50.0,
) -> bool:
    """Return True if dets are approximately top-to-bottom (≤25% backward y jumps)."""
    if len(dets) < 3:
        return True
    centers = [(d['bbox'][1] + d['bbox'][3]) / 2 for d in dets]
    back = sum(
        1 for i in range(1, len(centers))
        if centers[i] < centers[i - 1] - min_back_jump_px
    )
    return back / (len(centers) - 1) <= back_jump_threshold


def split_detections_by_column(
    detections: List[Dict[str, Any]],
    image_width: int,
    image_height: int = 0,
    use_dag: bool = True
) -> tuple:
    """
    Split detections into full_width, left_col, right_col lists.
    Returns (full_width, left_col, right_col) sorted by specified strategy.

    Changes vs original:
    - full_width_threshold raised 0.60 → 0.70 (matches detect_column_count).
    - Cross-gutter margin widened from 10% to 20% per side:
      The original second condition routed an item to full_width if its bbox
      crossed ±10% of the midpoint. On phone photos, YOLO over-expands bboxes
      by 5–15% on glared/blurry text, so right-column items whose left edge
      barely crossed the 10% margin were pulled into full_width and rendered
      BEFORE the column block — this is what caused right-column text to appear
      above left-column text in the output. With ±20%, only items that genuinely
      span both columns (like page-width tables or titles) are routed full_width.
    """
    if image_width == 0:
        if use_dag:
            sorted_all = apply_semantic_reading_order(detections, image_width, image_height)
        else:
            sorted_all = sort_detections_geometric(detections)
        return [], sorted_all, []

    full_width_threshold = image_width * 0.70   # raised from 0.60
    midpoint = image_width / 2

    full_width, left_col, right_col = [], [], []

    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        w = x2 - x1

        if w >= full_width_threshold or (
            x1 < midpoint - image_width * 0.20 and   # widened from 0.10
            x2 > midpoint + image_width * 0.20        # widened from 0.10
        ):
            full_width.append(det)
        elif (x1 + x2) / 2 < midpoint:
            left_col.append(det)
        else:
            right_col.append(det)

    def _sort(group):
        if not use_dag:
            return sort_detections_geometric(group)
        dag_sorted = apply_semantic_reading_order(group, image_width, image_height)
        if _is_y_monotonic(dag_sorted):
            return dag_sorted
        # DAG produced a non-monotonic order (cycle or bad edge) — fall back
        print("    [layout] DAG order non-monotonic; falling back to geometric sort")
        return sort_detections_geometric(group)

    full_width = _sort(full_width)
    left_col   = _sort(left_col)
    right_col  = _sort(right_col)

    return full_width, left_col, right_col