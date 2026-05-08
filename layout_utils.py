"""
layout_utils.py
---------------
Bounding box utilities: sorting and reading order.
Supports single-column, multi-column, and full-width elements using line-clustering and gutter detection.
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

    # 1. Initial Geometric Sort (Baseline tie-breaker)
    sorted_dets = sort_detections_geometric(detections)
    
    # 2. Build Adjacency List (DAG)
    # Nodes are indices in sorted_dets.
    adj = {i: set() for i in range(len(sorted_dets))}
    in_degree = {i: 0 for i in range(len(sorted_dets))}

    def add_edge(u, v):
        if v not in adj[u]:
            adj[u].add(v)
            in_degree[v] += 1

    # 3. Apply Semantic Rules (No hard geometric edges here to avoid cycles)
    
    # Rule B: Caption Pairing
    for i, det in enumerate(sorted_dets):
        if det['class_name'] == 'Caption':
            cx1, cy1, cx2, cy2 = det['bbox']
            best_parent = None
            min_dist = float('inf')
            
            for j, potential in enumerate(sorted_dets):
                if potential['class_name'] in {'Picture', 'Table'}:
                    px1, py1, px2, py2 = potential['bbox']
                    # Vertical distance
                    dist = min(abs(cy1 - py2), abs(cy2 - py1))
                    # Horizontal overlap
                    h_overlap = max(0, min(cx2, px2) - max(cx1, px1))
                    
                    if dist < 150 and (h_overlap > 0 or dist < 50):
                        if dist < min_dist:
                            min_dist = dist
                            best_parent = j
            
            if best_parent is not None:
                # Force Parent -> Caption
                add_edge(best_parent, i)

    # Rule C: Footnote Sinking
    # Every non-footnote node must point to every footnote node.
    footnotes = [i for i, d in enumerate(sorted_dets) if d['class_name'] == 'Footnote']
    non_footnotes = [i for i, d in enumerate(sorted_dets) if d['class_name'] != 'Footnote']
    
    for fn in footnotes:
        for nfn in non_footnotes:
            add_edge(nfn, fn)

    # 4. Topological Sort (Kahn's Algorithm)
    # The queue stores indices that have no remaining dependencies.
    queue = [i for i in range(len(sorted_dets)) if in_degree[i] == 0]
    result_indices = []
    
    while queue:
        # Sort queue to maintain GEOMETRIC preference when multiple nodes are available.
        queue.sort() 
        u = queue.pop(0)
        result_indices.append(u)
        
        for v in list(adj[u]):
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    # Fallback: if there's still a cycle (should be impossible now),
    # append any missing indices in their original geometric order.
    if len(result_indices) < len(sorted_dets):
        missing = [i for i in range(len(sorted_dets)) if i not in result_indices]
        result_indices.extend(missing)

    return [sorted_dets[i] for i in result_indices]


def sort_detections_geometric(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Robust geometric reading order (T-B, L-R) with line clustering.
    Use this for noisy phone photos where semantic rules might fail.
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
    """
    if image_width == 0 or not detections:
        return 1

    full_width_threshold = image_width * 0.60
    non_full_spans = []
    for d in detections:
        x1, _, x2, _ = d['bbox']
        if (x2 - x1) < full_width_threshold:
            non_full_spans.append((x1, x2))

    if not non_full_spans:
        return 1

    gutter_center = image_width / 2
    gutter_min = gutter_center - (image_width * 0.05)
    gutter_max = gutter_center + (image_width * 0.05)

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


def split_detections_by_column(
    detections: List[Dict[str, Any]],
    image_width: int,
    image_height: int = 0,
    use_dag: bool = True
) -> tuple:
    """
    Split detections into full_width, left_col, right_col lists.
    Returns (full_width, left_col, right_col) sorted by specified strategy.
    """
    if image_width == 0:
        if use_dag:
            sorted_all = apply_semantic_reading_order(detections, image_width, image_height)
        else:
            sorted_all = sort_detections_geometric(detections)
        return [], sorted_all, []

    full_width_threshold = image_width * 0.60
    midpoint = image_width / 2

    full_width, left_col, right_col = [], [], []

    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        w = x2 - x1
        
        if w >= full_width_threshold or (x1 < midpoint - image_width*0.1 and x2 > midpoint + image_width*0.1):
            full_width.append(det)
        elif (x1 + x2) / 2 < midpoint:
            left_col.append(det)
        else:
            right_col.append(det)

    # Sort each zone
    if use_dag:
        full_width = apply_semantic_reading_order(full_width, image_width, image_height)
        left_col = apply_semantic_reading_order(left_col, image_width, image_height)
        right_col = apply_semantic_reading_order(right_col, image_width, image_height)
    else:
        full_width = sort_detections_geometric(full_width)
        left_col = sort_detections_geometric(left_col)
        right_col = sort_detections_geometric(right_col)

    return full_width, left_col, right_col
