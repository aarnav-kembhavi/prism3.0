"""
detection_postprocess.py
------------------------
Multi-stage post-processing pipeline for YOLO layout detections.
Converts raw overlapping detections into clean, non-overlapping blocks.

Pipeline: raw → confidence filter → class-aware NMS → overlap resolution
          → box refinement → reading order
"""

from typing import List, Dict, Any, Tuple


# ----------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------

def compute_iou(bbox1: list, bbox2: list) -> float:
    """Compute Intersection over Union between two xyxy bboxes."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area


def compute_containment(inner_bbox: list, outer_bbox: list) -> float:
    """
    Compute what fraction of inner_bbox's area is inside outer_bbox.
    Returns 0.0 to 1.0. If 1.0, inner is fully contained in outer.
    """
    x1 = max(inner_bbox[0], outer_bbox[0])
    y1 = max(inner_bbox[1], outer_bbox[1])
    x2 = min(inner_bbox[2], outer_bbox[2])
    y2 = min(inner_bbox[3], outer_bbox[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    inner_area = (inner_bbox[2] - inner_bbox[0]) * (inner_bbox[3] - inner_bbox[1])
    if inner_area == 0:
        return 0.0
    return inter_area / inner_area


def bbox_area(bbox: list) -> float:
    """Compute area of an xyxy bbox."""
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


# ----------------------------------------------------------------
# Stage 1: Confidence Filtering
# ----------------------------------------------------------------

def filter_by_confidence(
    detections: List[Dict[str, Any]],
    threshold: float = 0.3
) -> List[Dict[str, Any]]:
    """Remove detections below confidence threshold."""
    return [d for d in detections if d['confidence'] >= threshold]


# ----------------------------------------------------------------
# Stage 2: Class-Aware NMS
# ----------------------------------------------------------------

def class_aware_nms(
    detections: List[Dict[str, Any]],
    iou_threshold: float = 0.4
) -> List[Dict[str, Any]]:
    """
    Non-Maximum Suppression applied per class.
    Only suppresses boxes of the SAME class that overlap above iou_threshold.
    Keeps the higher-confidence box.
    """
    if not detections:
        return []

    # Group by class
    by_class = {}
    for det in detections:
        cls = det['class_name']
        by_class.setdefault(cls, []).append(det)

    result = []
    for cls, dets in by_class.items():
        # Sort by confidence descending
        dets_sorted = sorted(dets, key=lambda d: d['confidence'], reverse=True)
        keep = []

        while dets_sorted:
            best = dets_sorted.pop(0)
            keep.append(best)

            # Remove boxes that overlap too much with the kept box
            remaining = []
            for det in dets_sorted:
                iou = compute_iou(best['bbox'], det['bbox'])
                if iou < iou_threshold:
                    remaining.append(det)
            dets_sorted = remaining

        result.extend(keep)

    return result


# ----------------------------------------------------------------
# Stage 3: Hierarchical Overlap Resolution
# ----------------------------------------------------------------

def resolve_overlaps(
    detections: List[Dict[str, Any]],
    containment_threshold: float = 0.80,
    partial_iou_threshold: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Resolve remaining cross-class overlaps.

    Rules:
    1. If box A is mostly contained inside box B (>80% of A inside B),
       remove A (it's a duplicate sub-detection of the same region).
       EXCEPTION: never remove a Section-header or Page-header that is
       "contained" in a Text block — YOLO frequently detects a bold header
       line as both a Section-header and part of a neighbouring Text region.
       Dropping the Section-header causes it to vanish from the output
       (observed: "B. FRONTEND LAYER" missing entirely).
    2. If two boxes partially overlap (IoU > 0.3, different classes),
       clip the lower-confidence box to remove the overlapping region.
    """
    if not detections:
        return []

    # Classes that should NEVER be silently consumed by a containing box.
    _PROTECTED_CLASSES = {"Section-header", "Page-header", "Title", "Caption"}

    # Sort by area descending (larger boxes first)
    dets = sorted(detections, key=lambda d: bbox_area(d['bbox']), reverse=True)

    to_remove = set()

    for i in range(len(dets)):
        if i in to_remove:
            continue
        for j in range(i + 1, len(dets)):
            if j in to_remove:
                continue

            bbox_i = dets[i]['bbox']
            bbox_j = dets[j]['bbox']

            # Check if smaller box (j) is mostly contained in larger box (i)
            containment = compute_containment(bbox_j, bbox_i)
            if containment >= containment_threshold:
                # Never drop protected classes regardless of containment
                if dets[j]['class_name'] in _PROTECTED_CLASSES:
                    continue
                # j is inside i — remove the contained box
                to_remove.add(j)
                continue

            # Check partial overlap between different classes
            iou = compute_iou(bbox_i, bbox_j)
            if iou > partial_iou_threshold:
                # Clip the lower-confidence box
                if dets[j]['confidence'] <= dets[i]['confidence']:
                    dets[j] = _clip_box(dets[j], dets[i])
                else:
                    dets[i] = _clip_box(dets[i], dets[j])

    return [d for idx, d in enumerate(dets) if idx not in to_remove]


def _clip_box(
    box_to_clip: Dict[str, Any],
    dominant_box: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Clip box_to_clip to remove the region overlapping with dominant_box.
    Shrinks box_to_clip from the side that produces the least area loss.
    Returns a new detection dict with modified bbox.
    """
    ax1, ay1, ax2, ay2 = box_to_clip['bbox']
    bx1, by1, bx2, by2 = dominant_box['bbox']

    # Compute how much we'd need to shrink from each side
    candidates = []

    # Clip from top (move ay1 down to by2)
    if ay1 < by2 and ay2 > by2:
        new_bbox = [ax1, by2, ax2, ay2]
        if bbox_area(new_bbox) > 0:
            candidates.append(new_bbox)

    # Clip from bottom (move ay2 up to by1)
    if ay2 > by1 and ay1 < by1:
        new_bbox = [ax1, ay1, ax2, by1]
        if bbox_area(new_bbox) > 0:
            candidates.append(new_bbox)

    # Clip from left (move ax1 right to bx2)
    if ax1 < bx2 and ax2 > bx2:
        new_bbox = [bx2, ay1, ax2, ay2]
        if bbox_area(new_bbox) > 0:
            candidates.append(new_bbox)

    # Clip from right (move ax2 left to bx1)
    if ax2 > bx1 and ax1 < bx1:
        new_bbox = [ax1, ay1, bx1, ay2]
        if bbox_area(new_bbox) > 0:
            candidates.append(new_bbox)

    if not candidates:
        return box_to_clip

    # Keep the candidate that preserves the most area
    best = max(candidates, key=lambda b: bbox_area(b))

    clipped = dict(box_to_clip)
    clipped['bbox'] = best
    # Re-crop the image if crop exists
    if 'crop' in clipped and hasattr(clipped['crop'], 'crop'):
        from layout_utils import xyxy_to_pil_crop
        # We need the original image to re-crop — skip if not available
        # The crop will be slightly inaccurate but acceptable
    return clipped


# ----------------------------------------------------------------
# Stage 4: Box Refinement
# ----------------------------------------------------------------

def refine_boxes(
    detections: List[Dict[str, Any]],
    image_width: int,
    image_height: int,
    merge_gap: int = 8,
    padding: int = 2
) -> List[Dict[str, Any]]:
    """
    Merge and refine bounding boxes.

    1. Merge nearby same-class boxes (vertical gap < merge_gap pixels)
       that are horizontally aligned (overlap in X by > 50%).
    2. Add small padding to each box for OCR readability.
    3. Clamp all boxes to image boundaries.
    """
    if not detections:
        return []

    # --- Merge nearby same-class boxes ---
    merged = _merge_nearby_boxes(detections, merge_gap)

    # --- Add padding and clamp ---
    refined = []
    for det in merged:
        x1, y1, x2, y2 = det['bbox']

        # Add padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image_width, x2 + padding)
        y2 = min(image_height, y2 + padding)

        det_copy = dict(det)
        det_copy['bbox'] = [x1, y1, x2, y2]
        refined.append(det_copy)

    return refined


def _merge_nearby_boxes(
    detections: List[Dict[str, Any]],
    merge_gap: int
) -> List[Dict[str, Any]]:
    """
    Merge vertically adjacent boxes of the same class if:
    - Vertical gap between them is < merge_gap pixels
    - They overlap horizontally by > 50%

    EXCEPTION: List-item detections are NEVER merged.
    Each YOLO List-item box corresponds to a single bullet point; merging them
    into one crop sends all bullets to OCR as one blob, which then collapses
    into a run-on string that _split_bullet_items() cannot reliably re-split.
    Keeping them separate means each bullet is cropped and read individually,
    producing clean per-item OCR output.
    """
    if not detections:
        return []

    # Classes that must never be merged with their neighbours
    _NO_MERGE_CLASSES = {"List-item", "Section-header", "Title", "Caption"}

    # Group by class
    by_class = {}
    for det in detections:
        cls = det['class_name']
        by_class.setdefault(cls, []).append(det)

    result = []
    for cls, dets in by_class.items():
        # Never merge these classes — emit them as-is
        if cls in _NO_MERGE_CLASSES:
            result.extend(dets)
            continue

        # Sort by Y center
        dets_sorted = sorted(dets, key=lambda d: (d['bbox'][1] + d['bbox'][3]) / 2)

        merged_group = [dets_sorted[0]]

        for i in range(1, len(dets_sorted)):
            prev = merged_group[-1]
            curr = dets_sorted[i]

            px1, py1, px2, py2 = prev['bbox']
            cx1, cy1, cx2, cy2 = curr['bbox']

            # Vertical gap
            v_gap = cy1 - py2

            # Horizontal overlap ratio
            x_overlap_start = max(px1, cx1)
            x_overlap_end = min(px2, cx2)
            x_overlap = max(0, x_overlap_end - x_overlap_start)
            min_width = min(px2 - px1, cx2 - cx1)
            h_overlap_ratio = x_overlap / min_width if min_width > 0 else 0

            if 0 <= v_gap < merge_gap and h_overlap_ratio > 0.5:
                # Merge: expand prev bbox to include curr
                new_bbox = [
                    min(px1, cx1),
                    min(py1, cy1),
                    max(px2, cx2),
                    max(py2, cy2),
                ]
                merged_det = dict(prev)
                merged_det['bbox'] = new_bbox
                merged_det['confidence'] = max(prev['confidence'], curr['confidence'])
                merged_group[-1] = merged_det
            else:
                merged_group.append(curr)

        result.extend(merged_group)

    return result


# ----------------------------------------------------------------
# Stage 5: Full Pipeline
# ----------------------------------------------------------------

def postprocess_detections(
    detections: List[Dict[str, Any]],
    image_width: int,
    image_height: int,
    conf_threshold: float = 0.3,
    nms_iou_threshold: float = 0.4
) -> List[Dict[str, Any]]:
    """
    Full post-processing pipeline.

    raw_detections → confidence filter → class-aware NMS
    → overlap resolution → box refinement

    Returns clean, non-overlapping detections ready for rendering.
    """
    count_raw = len(detections)

    # Stage 1: Confidence filter
    dets = filter_by_confidence(detections, threshold=conf_threshold)

    # Stage 2: Class-aware NMS
    dets = class_aware_nms(dets, iou_threshold=nms_iou_threshold)

    # Stage 3: Resolve remaining overlaps (cross-class)
    dets = resolve_overlaps(dets)

    # Stage 4: Refine boxes (merge, pad, clamp)
    dets = refine_boxes(dets, image_width, image_height)

    count_final = len(dets)
    print(f"[*] Post-processing: {count_raw} → {count_final} detections")

    return dets