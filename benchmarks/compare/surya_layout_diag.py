"""Display-formula DETECTION recall for Surya layout (torch SegFormer, 0.13.x).
Same methodology as formula_detection_diag.py so results are comparable to the
29% baseline and PP-DocLayout_plus-L. Run in .venv_smol (surya==0.13.1).
Surya layout label for display math is 'Equation'.
"""
import os, sys, json, ast, time, statistics
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault('TORCH_DEVICE', 'cpu')
from PIL import Image

GTJ = "data/omnidocbench/OmniDocBench_available.json"
IMG = "data/omnidocbench/images"


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    max_pages = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    from surya.layout import LayoutPredictor
    pred = LayoutPredictor()
    gt = json.load(open(GTJ, encoding='utf-8'))

    tot_gt = 0; detected = 0; matched_ious = []
    total_det = 0; pages = 0; lat = []; labels_seen = set()
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
        img = Image.open(ip).convert('RGB')
        t = time.perf_counter()
        res = pred([img])
        lat.append(time.perf_counter() - t)
        r = res[0]
        for b in r.bboxes:
            labels_seen.add(b.label)
        det = [b.bbox for b in r.bboxes if b.label in ('Equation',)]
        tot_gt += len(gtf); total_det += len(det); pages += 1
        for g in gtf:
            best = max((iou(g, d) for d in det), default=0.0)
            if best > 0.5:
                detected += 1; matched_ious.append(best)
        if pages >= max_pages:
            break

    print("=== Surya layout (torch) ===")
    print(f"labels seen: {sorted(labels_seen)}")
    print(f"pages: {pages} | GT formulas: {tot_gt} | detected 'Equation' boxes: {total_det}")
    print(f"RECALL (IoU>0.5): {detected}/{tot_gt} = {100*detected/max(tot_gt,1):.1f}%")
    if matched_ious:
        print(f"mean IoU of matches (tightness): {statistics.mean(matched_ious):.3f}")
    if lat:
        print(f"latency: median {statistics.median(lat):.2f}s/page, mean {statistics.mean(lat):.2f}s")


if __name__ == '__main__':
    main()
