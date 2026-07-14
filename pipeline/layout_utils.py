"""
layout_utils.py
---------------
Bounding box utilities: sorting and reading order.
Supports single-column, multi-column, and full-width elements using
line-clustering and gutter detection.
"""

from typing import List, Dict, Any
import numpy as np


def xycut_order(detections: List[Dict[str, Any]], img_width: int,
                img_height: int) -> List[Dict[str, Any]]:
    """Recursive XY-cut reading order for complex (3+ column) layouts.

    The flat "full-width elements, then column 1..N top-to-bottom" model
    scrambles newspapers: their structure is vertical REGIONS (masthead,
    article blocks separated by banners/photos), each with its own columns.
    XY-cut expresses that: recursively split at the widest empty horizontal
    band (top block first), else at empty vertical gutters (left to right),
    else fall back to geometric order.

    Boxes spanning nearly the whole page (background pictures) are excluded
    from cut computation — a full-page box would otherwise block every cut —
    and are emitted first within their group.
    """
    if not detections:
        return []
    min_gap_y = max(8.0, 0.006 * img_height)
    min_gap_x = max(8.0, 0.006 * img_width)

    def gaps(intervals, min_gap):
        """Empty gaps >= min_gap between merged [lo, hi] intervals."""
        ivs = sorted(intervals)
        merged = [list(ivs[0])]
        for lo, hi in ivs[1:]:
            if lo <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        return [(merged[i][1], merged[i + 1][0]) for i in range(len(merged) - 1)
                if merged[i + 1][0] - merged[i][1] >= min_gap]

    def rec(dets, depth):
        if len(dets) <= 1 or depth > 12:
            return sort_detections_geometric(dets)
        # background boxes don't participate in cuts
        cuttable = [d for d in dets
                    if not ((d['bbox'][2] - d['bbox'][0]) > 0.9 * img_width
                            and (d['bbox'][3] - d['bbox'][1]) > 0.9 * img_height)]
        _cut_ids = {id(d) for d in cuttable}
        bg = [d for d in dets if id(d) not in _cut_ids]
        if not cuttable:
            return sort_detections_geometric(dets)

        ycuts = gaps([(d['bbox'][1], d['bbox'][3]) for d in cuttable], min_gap_y)
        if ycuts:
            bands: List[list] = [[] for _ in range(len(ycuts) + 1)]
            for d in cuttable:
                cy = (d['bbox'][1] + d['bbox'][3]) / 2
                k = sum(1 for g0, g1 in ycuts if cy > (g0 + g1) / 2)
                bands[k].append(d)
            out = list(bg)
            for band in bands:
                out.extend(rec(band, depth + 1))
            return out

        xcuts = gaps([(d['bbox'][0], d['bbox'][2]) for d in cuttable], min_gap_x)
        if xcuts:
            cols: List[list] = [[] for _ in range(len(xcuts) + 1)]
            for d in cuttable:
                cx = (d['bbox'][0] + d['bbox'][2]) / 2
                k = sum(1 for g0, g1 in xcuts if cx > (g0 + g1) / 2)
                cols[k].append(d)
            out = list(bg)
            for col in cols:
                out.extend(rec(col, depth + 1))
            return out

        return sort_detections_geometric(dets)

    return rec(list(detections), 0)


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


def _find_gutters(detections: List[Dict[str, Any]], image_width: int,
                  max_coverage: int = 0):
    """
    Build a horizontal coverage histogram and return (non_full, gutter_list).

    A gutter is a contiguous run of bins with coverage <= max_coverage that is
    at least 1.5% of the page width wide and lies within the inner 90% of the
    page (5% margin each side).

    max_coverage=0 (default): strict zero-coverage gutters only.
    max_coverage=1: allows one stray detection per bin — catches 3-column
    pages where a single YOLO bbox slightly overhangs the column gutter.

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
    is_gutter[margin:BIN_COUNT - margin] = (coverage[margin:BIN_COUNT - margin] <= max_coverage)

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


def _centroid_gaps(non_full, image_width, min_gap_frac=0.15, min_per_col=2):
    """
    Detect column boundaries via X-centroid gap detection.

    Useful for dense-text layouts (newspapers) where column text blocks cover
    the narrow inter-column gutter, so the coverage histogram only finds
    margin gutters. This approach works on *where elements are* rather than
    *where whitespace is*.

    Uses the top-(N-1) largest inter-centroid gaps as column boundaries, so a
    single cross-column headline element does not break 3-column detection by
    splitting one large gap into two smaller ones.

    Returns (n_cols, boundaries_px) if ≥ 3 valid columns found, else (0, []).
    boundaries_px = pixel x-positions of column separators (sorted).
    """
    if len(non_full) < 3 * min_per_col:
        return 0, []

    centroids = sorted((d['bbox'][0] + d['bbox'][2]) / (2 * image_width) for d in non_full)
    if len(centroids) < 6:
        return 0, []

    all_gaps = [(centroids[i + 1] - centroids[i], i) for i in range(len(centroids) - 1)]

    for n_cols in range(3, min(9, len(centroids))):
        # Take the n_cols-1 largest gaps as column boundaries
        top = sorted(all_gaps, key=lambda x: -x[0])[:n_cols - 1]
        if min(g for g, _ in top) < min_gap_frac:
            break  # remaining gaps are too small for any reasonable column count

        gap_indices = sorted(i for _, i in top)
        splits = [0] + [i + 1 for i in gap_indices] + [len(centroids)]
        col_sizes = [splits[j + 1] - splits[j] for j in range(n_cols)]

        if all(s >= min_per_col for s in col_sizes):
            boundaries_px = [
                (centroids[i] + centroids[i + 1]) / 2 * image_width
                for i in gap_indices
            ]
            return min(n_cols, 8), boundaries_px

    return 0, []


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

    def _validate_n_columns(n_potential, gutters_list, nf):
        """
        Count effective content columns, trimming empty edge columns (page margins).
        Returns the effective column count if ≥3 real columns all have ≥2 dets, else 0.
        """
        if not gutters_list:
            return 0
        boundaries = [0] + [g[1] + 1 for g in gutters_list] + [BIN_COUNT]
        counts = []
        for i in range(n_potential):
            col_min_x = boundaries[i] / BIN_COUNT * image_width
            col_max_x = boundaries[i + 1] / BIN_COUNT * image_width
            in_col = [d for d in nf
                      if col_min_x <= (d['bbox'][0] + d['bbox'][2]) / 2 < col_max_x]
            counts.append(len(in_col))

        # Trim empty edge columns — these are page margins, not real content columns.
        # A single YOLO bbox whose left/right edge slightly overshoots the text area
        # creates a phantom gutter at the margin, splitting off a zero-detection
        # "column" that breaks the ≥2-detection requirement for real columns.
        first = next((i for i, c in enumerate(counts) if c >= 1), None)
        last  = next((i for i, c in enumerate(reversed(counts)) if c >= 1), None)
        if first is None:
            return 0
        last = n_potential - 1 - last
        effective_n = last - first + 1
        if effective_n < 3:
            return 0
        if all(c >= 2 for c in counts[first:last + 1]):
            return min(effective_n, 8)
        return 0

    # --- Try N ≥ 3 (strict: zero-coverage gutters) ---
    n = _validate_n_columns(len(gutters) + 1, gutters, non_full)
    if n:
        return n

    # --- Retry with relaxed gutters (≤1 stray detection per bin) ---
    # Catches 3-column pages where a single YOLO bbox slightly overhangs the gutter.
    _, gutters_relaxed = _find_gutters(detections, image_width, max_coverage=1)
    n = _validate_n_columns(len(gutters_relaxed) + 1, gutters_relaxed, non_full)
    if n:
        return n

    # --- Centroid gap fallback (catches dense-text newspapers) ---
    # When every column's text blocks cover the narrow inter-column gutter,
    # coverage-histogram gutters only appear at the page margins.  In that
    # case, look for large gaps (≥20% of page width) between sorted X-centroids
    # of non-full-width elements — newspaper columns produce gaps of ~25-30%.
    n, _ = _centroid_gaps(non_full, image_width)
    if n >= 3:
        return n

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


_FIDELITY_BODY_CLASSES = {"Text", "List-item", "Section-header", "Caption"}


def detect_columns_fidelity(
    detections: List[Dict[str, Any]],
    image_width: int,
    image_height: int = 0,
) -> tuple:
    """Robust column detector for VISUAL-FIDELITY output.

    Unlike detect_column_count() (whitespace-gutter + centroid heuristics tuned
    to fire conservatively for benchmark reading-order), this clusters the LEFT
    EDGES of body-text boxes. Column starts align almost perfectly (a 3-column
    page has left edges tightly grouped at e.g. x≈77/437/796), so left-edge
    clustering is far more reliable than centroids (which a non-full-width
    title/headline pollutes) or gutters (which dense text fills).

    Method: find vertical whitespace gutters over the NARROW body boxes only
    (excluding full-width titles/banners/figures). This is the key: a spanning
    title fills the gutter and hides it (which is exactly why the benchmark
    detector reports "1 column" on a 3-column IRS form), so those boxes must be
    removed before looking for the gutter. A genuine 2/3-column page then shows
    clean empty vertical corridors; a single-column page whose paragraphs merely
    start at different x has overlapping x-spans (once unioned) and NO empty
    corridor, so it correctly stays 1 column.

    Returns (n_cols, col_lefts) where col_lefts is the sorted list of the
    representative left-edge x-position (pixels) of each detected column.
    n_cols == 1 (col_lefts == []) means "not multi-column — use the normal path".
    """
    if image_width == 0 or not detections:
        return 1, []
    W = image_width
    # Body boxes narrower than half the page.  The lower bound (5% width) drops
    # margin artefacts — line-number bars, rotated side labels, stray slivers —
    # that would otherwise seed a phantom column at the page edge.
    narrow = [d for d in detections
              if d['class_name'] in _FIDELITY_BODY_CLASSES
              and 0.05 * W <= (d['bbox'][2] - d['bbox'][0]) < 0.5 * W]
    if len(narrow) < 3:
        return 1, []

    # Coverage histogram over the narrow body boxes only.  A genuine multi-column
    # page shows tall content plateaus separated by sharp corridors that dip to
    # (near) zero coverage.  Those corridors can be razor-thin — a single 0.5%
    # bin — because the detector's bboxes fill nearly the whole column width, so
    # the fixed 1.5% floor in _find_gutters() misses them.  Here we instead look
    # for INTERIOR corridors: runs of (near-)zero coverage that lie strictly
    # between the first and last covered bin.  Margin whitespace falls outside
    # that content span and is ignored automatically, and a single-column page
    # (whose unioned x-spans leave no interior corridor) correctly stays 1.
    BIN_COUNT = 200
    cov = np.zeros(BIN_COUNT, dtype=int)
    for d in narrow:
        x1, _, x2, _ = d['bbox']
        l = max(0, int(x1 / W * BIN_COUNT))
        r = min(BIN_COUNT - 1, int(x2 / W * BIN_COUNT))
        if l <= r:
            cov[l:r + 1] += 1

    covered = np.nonzero(cov > 0)[0]
    if covered.size == 0:
        return 1, []
    c_lo, c_hi = int(covered[0]), int(covered[-1])

    def _interior_boundaries(max_cov: int) -> List[float]:
        bnds: List[float] = []
        in_run = False
        r_start = 0
        for i in range(c_lo + 1, c_hi):        # strictly interior to content
            if cov[i] <= max_cov and not in_run:
                in_run, r_start = True, i
            elif cov[i] > max_cov and in_run:
                in_run = False
                bnds.append((r_start + i - 1) / 2 / BIN_COUNT * W)
        return bnds

    boundaries = _interior_boundaries(0) or _interior_boundaries(1)
    if not boundaries:
        return 1, []

    n = len(boundaries) + 1
    cols: List[List[Dict[str, Any]]] = [[] for _ in range(n)]
    for d in narrow:
        xc = (d['bbox'][0] + d['bbox'][2]) / 2
        ci = sum(1 for b in boundaries if xc > b)
        cols[min(ci, n - 1)].append(d)

    # Keep non-empty columns.  A real multi-column page may leave one side with a
    # single body box (the other side being a figure/table/full-width caption),
    # so we do not demand >= 2 per column; instead we require >= 2 columns AND >= 3
    # body boxes total, and rely on the band-overlap check below to reject a
    # single-column page that happened to split on a spurious corridor.
    kept = [c for c in cols if len(c) >= 1]
    if len(kept) < 2 or sum(len(c) for c in kept) < 3:
        return 1, []

    # Validation: adjacent columns must occupy roughly non-overlapping x-bands.
    # A single-column page split on a spurious corridor would produce columns
    # whose bands interleave heavily — reject those.
    kept.sort(key=lambda c: np.median([d['bbox'][0] for d in c]))
    tol = 0.04 * W
    for a, b in zip(kept, kept[1:]):
        a_right = float(np.median([d['bbox'][2] for d in a]))
        b_left  = float(np.median([d['bbox'][0] for d in b]))
        if a_right > b_left + tol:
            return 1, []

    col_lefts = sorted(min(d['bbox'][0] for d in c) for c in kept)
    return min(len(kept), 6), col_lefts


def split_detections_fidelity(
    detections: List[Dict[str, Any]],
    n_cols: int,
    col_lefts: List[float],
    image_width: int,
    image_height: int = 0,
) -> tuple:
    """Assign detections to (top_full, columns, bottom_full) for fidelity render.

    - A box wider than ~1.7× the typical column width (or ≥55% page width) spans
      columns → it is "full width" (title, abstract banner, page-wide figure).
    - Every other box joins the column whose representative left edge is nearest
      its own left edge (left edges align tightly, so this is unambiguous).
    - Full-width boxes above the first column line render as a full-width header
      block; the rest render full-width after the columns.
    """
    W = image_width
    cand_w = [d['bbox'][2] - d['bbox'][0] for d in detections
              if d['class_name'] in _FIDELITY_BODY_CLASSES
              and (d['bbox'][2] - d['bbox'][0]) < 0.55 * W]
    typ = float(np.median(cand_w)) if cand_w else 0.30 * W
    span_thr = max(0.55 * W, 1.7 * typ)

    full: List[Dict] = []
    columns: List[List[Dict]] = [[] for _ in range(n_cols)]
    for d in detections:
        x1, _, x2, _ = d['bbox']
        if (x2 - x1) >= span_thr:
            full.append(d)
        else:
            ci = min(range(n_cols), key=lambda i: abs(x1 - col_lefts[i]))
            columns[ci].append(d)

    columns = [sort_detections_geometric(c) if c else c for c in columns]

    col_top = min((d['bbox'][1] for c in columns for d in c), default=0.0)
    band = 0.02 * image_height
    top = sort_detections_geometric([d for d in full if d['bbox'][3] <= col_top + band])
    bottom = sort_detections_geometric([d for d in full if d['bbox'][3] > col_top + band])
    return top, columns, bottom


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

    # If gutters were only margin gutters (≥2 consecutive empty edge columns), the
    # gutter-based split collapses to a single effective column.  Fall back to
    # centroid gap detection which works on element positions instead of whitespace.
    non_empty = sum(1 for col in columns if col)
    if non_empty < 3:
        _, centroid_bpx = _centroid_gaps(non_full, image_width)
        if centroid_bpx:
            boundaries_px = centroid_bpx
            N = len(boundaries_px) + 1
            columns = [[] for _ in range(N)]
            for d in non_full:
                x_ctr = (d['bbox'][0] + d['bbox'][2]) / 2
                col_idx = sum(1 for b in boundaries_px if x_ctr > b)
                columns[min(col_idx, N - 1)].append(d)

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