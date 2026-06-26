"""
layout_utils.py
---------------
Bounding box utilities: sorting and reading order.
Supports single-column, multi-column, and full-width elements using
line-clustering and gutter detection.
"""

from typing import List, Dict, Any
import numpy as np


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

    # Move footnotes to end (O(n), preserving relative order within each group)
    non_fn = [i for i in result_indices if sorted_dets[i]['class_name'] != 'Footnote']
    fn_    = [i for i in result_indices if sorted_dets[i]['class_name'] == 'Footnote']
    result_indices = non_fn + fn_

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


def _find_gutters(detections: List[Dict[str, Any]], image_width: int):
    """
    Build a horizontal coverage histogram and return (non_full, gutter_list).

    A gutter is a contiguous run of completely-empty bins (zero detections
    cover that x-slice) that is at least 1.5% of the page width wide and
    lies within the inner 90% of the page (5% margin each side).

    Returns (non_full_dets, [(g_start_bin, g_end_bin), ...]) where bin
    coordinates are in [0, BIN_COUNT).
    """
    BIN_COUNT = 200  # 0.5% resolution
    full_width_threshold = image_width * 0.70

    non_full = [d for d in detections if (d['bbox'][2] - d['bbox'][0]) < full_width_threshold]
    if not non_full:
        return non_full, []

    coverage = np.zeros(BIN_COUNT, dtype=int)
    for d in non_full:
        x1, _, x2, _ = d['bbox']
        l = max(0, int(x1 / image_width * BIN_COUNT))
        r = min(BIN_COUNT - 1, int(x2 / image_width * BIN_COUNT))
        if l <= r:
            coverage[l:r + 1] += 1

    margin = int(BIN_COUNT * 0.05)
    is_gutter = np.zeros(BIN_COUNT, dtype=bool)
    is_gutter[margin:BIN_COUNT - margin] = (coverage[margin:BIN_COUNT - margin] == 0)

    gutters = []
    in_gutter = False
    g_start = 0
    for i in range(BIN_COUNT):
        if is_gutter[i] and not in_gutter:
            in_gutter = True
            g_start = i
        elif not is_gutter[i] and in_gutter:
            in_gutter = False
            if (i - g_start) / BIN_COUNT >= 0.015:  # min 1.5% width
                gutters.append((g_start, i - 1))
    if in_gutter and (BIN_COUNT - g_start) / BIN_COUNT >= 0.015:
        gutters.append((g_start, BIN_COUNT - 1))

    return non_full, gutters


def detect_column_count(detections: List[Dict[str, Any]], image_width: int) -> int:
    """
    Detect the number of columns on the page (1–8).

    Strategy:
    1. Try N ≥ 3 via horizontal coverage histogram: find completely-empty
       vertical slices (gutters).  Each gutter meeting min-width and
       each resulting column having ≥ 2 detections → return N.
    2. Fall back to the existing 2-column gutter heuristic (tuned for
       phone photos of academic papers with YOLO bbox expansion).

    Why two strategies:
    - The histogram approach is conservative (requires ZERO coverage in a
      bin) and would miss 2-column academic papers where sharpened YOLO
      bboxes slightly overlap the gutter.  The original ±8% tolerance
      heuristic handles that better.
    - Newspapers/magazines have much wider, cleaner gutters (3–8 columns)
      so the histogram finds them reliably without the ±8% fudge.
    """
    if image_width == 0 or not detections:
        return 1

    non_full, gutters = _find_gutters(detections, image_width)
    if not non_full:
        return 1

    BIN_COUNT = 200
    full_width_threshold = image_width * 0.70

    # --- Try N ≥ 3 ---
    n_potential = len(gutters) + 1
    if n_potential >= 3 and gutters:
        boundaries = [0] + [g[1] + 1 for g in gutters] + [BIN_COUNT]
        valid = True
        for i in range(n_potential):
            col_min_x = boundaries[i] / BIN_COUNT * image_width
            col_max_x = boundaries[i + 1] / BIN_COUNT * image_width
            in_col = [
                d for d in non_full
                if col_min_x <= (d['bbox'][0] + d['bbox'][2]) / 2 < col_max_x
            ]
            if len(in_col) < 2:
                valid = False
                break
        if valid:
            return min(n_potential, 8)

    # --- Fall back to 2-column heuristic (original logic) ---
    non_full_spans = [(d['bbox'][0], d['bbox'][2]) for d in non_full]

    gutter_center = image_width / 2
    gutter_min = gutter_center - image_width * 0.08
    gutter_max = gutter_center + image_width * 0.08

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


def split_detections_n_columns(
    detections: List[Dict[str, Any]],
    image_width: int,
    image_height: int = 0,
    use_dag: bool = True,
) -> tuple:
    """
    Split detections for N ≥ 3 column layouts (newspapers / magazines).
    Returns (full_width_list, [col0_list, col1_list, ..., col{N-1}_list]).

    Uses the same coverage histogram as detect_column_count() to find the
    gutter centers, then assigns each non-full-width detection to the
    column whose x-range contains its centroid.
    """
    full_width_threshold = image_width * 0.70
    full_dets = [d for d in detections if (d['bbox'][2] - d['bbox'][0]) >= full_width_threshold]
    non_full,  gutters   = _find_gutters(detections, image_width)

    BIN_COUNT = 200

    if not gutters:
        # No gutters found — fall back to single column
        def _sort(group):
            if not group:
                return group
            if not use_dag:
                return sort_detections_geometric(group)
            dag_sorted = apply_semantic_reading_order(group, image_width, image_height)
            return dag_sorted if _is_y_monotonic(dag_sorted) else sort_detections_geometric(group)
        return _sort(full_dets), [_sort(non_full)]

    # Gutter center → pixel boundary between columns
    boundaries_px = [
        (g[0] + g[1]) / 2 / BIN_COUNT * image_width
        for g in gutters
    ]
    N = len(boundaries_px) + 1
    columns: List[List[Dict]] = [[] for _ in range(N)]

    for d in non_full:
        x_ctr = (d['bbox'][0] + d['bbox'][2]) / 2
        col_idx = sum(1 for b in boundaries_px if x_ctr > b)
        col_idx = min(col_idx, N - 1)
        columns[col_idx].append(d)

    def _sort(group):
        if not group:
            return group
        if not use_dag:
            return sort_detections_geometric(group)
        dag_sorted = apply_semantic_reading_order(group, image_width, image_height)
        if _is_y_monotonic(dag_sorted):
            return dag_sorted
        print("    [layout] DAG order non-monotonic in N-col; falling back to geometric sort")
        return sort_detections_geometric(group)

    full_dets = _sort(full_dets)
    columns   = [_sort(c) for c in columns]
    return full_dets, columns