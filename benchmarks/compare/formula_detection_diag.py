"""Diagnose formula DETECTION quality vs GT on English pages.
Runs PRISM's raw YOLO + DocLayout formula detectors on each page, matches
detected formula boxes to GT equation_isolated boxes, reports recall / merge /
split / IoU-tightness — isolating where the end-to-end formula loss comes from.
"""
import os, sys, json, ast
sys.path.insert(0, r"C:\PROJECTS\s2l2\testprism")
os.chdir(r"C:\PROJECTS\s2l2\testprism")

from pipeline.models_interface import get_yolo_detector, get_doclayout_detector

YOLO = "weights/yolov11n-doclaynet.onnx"
DL = "models/doclayout_yolo_docstructbench_imgsz1024.onnx"
GTJ = "data/omnidocbench/OmniDocBench_available.json"
IMG = "data/omnidocbench/images"


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0

def contains(gt, det):  # fraction of gt box inside det box
    ix1, iy1 = max(gt[0], det[0]), max(gt[1], det[1])
    ix2, iy2 = min(gt[2], det[2]), min(gt[3], det[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    ga = (gt[2]-gt[0])*(gt[3]-gt[1])
    return inter/ga if ga > 0 else 0.0


def main():
    max_pages = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    ydet = get_yolo_detector(YOLO, 640)
    ddet = get_doclayout_detector(DL, 1024)
    gt = json.load(open(GTJ, encoding='utf-8'))

    tot_gt = 0; detected = 0; matched_ious = []
    merged_boxes = 0; total_det = 0
    pages = 0
    for page in gt:
        a = page['page_info']['page_attribute']
        if isinstance(a, str): a = ast.literal_eval(a)
        if a.get('language') != 'english':
            continue
        gtf = [ann['poly'] for ann in page.get('layout_dets', [])
               if ann.get('category_type') == 'equation_isolated' and not ann.get('ignore')]
        gtf = [[p[0], p[1], p[4], p[5]] for p in gtf]
        if not gtf:
            continue
        ip = os.path.join(IMG, page['page_info']['image_path'])
        if not os.path.exists(ip):
            continue
        # detect formula boxes (YOLO 'Formula' + DocLayout 'isolate_formula')
        det = [d['bbox'] for d in ydet.detect(ip, conf=0.25, iou=0.7) if d['class_name'] == 'Formula']
        det += [d['bbox'] for d in ddet.detect(ip, conf=0.20) if d['class_name'] == 'isolate_formula']
        tot_gt += len(gtf); total_det += len(det); pages += 1
        # recall + tightness
        for g in gtf:
            best = max((iou(g, d) for d in det), default=0.0)
            if best > 0.5:
                detected += 1; matched_ious.append(best)
        # merge: a detected box that contains >=2 GT formula boxes
        for d in det:
            n_inside = sum(1 for g in gtf if contains(g, d) > 0.6)
            if n_inside >= 2:
                merged_boxes += 1
        if pages >= max_pages:
            break

    import statistics
    print(f"pages: {pages} | GT formulas: {tot_gt} | detected boxes: {total_det}")
    print(f"RECALL (GT formula matched @IoU>0.5): {detected}/{tot_gt} = {100*detected/max(tot_gt,1):.1f}%")
    print(f"mean IoU of matches (tightness): {statistics.mean(matched_ious):.3f}" if matched_ious else "no matches")
    print(f"MERGED boxes (>=2 GT formulas in one det box): {merged_boxes}/{total_det} = {100*merged_boxes/max(total_det,1):.1f}%")


if __name__ == '__main__':
    main()
