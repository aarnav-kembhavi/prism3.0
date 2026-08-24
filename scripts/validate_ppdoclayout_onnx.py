"""Validate the onnxruntime PP-DocLayout wrapper against the paddle model's
cached boxes (ppdl_full_cache.json, in PRISM class vocab) on the same normalized
images. Reports per-page box-count match + mean IoU of best matches.
"""
import os, sys, json, glob
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from pipeline.ppdoclayout_onnx import PPDocLayoutOnnxDetector

CACHE = "benchmarks/compare/formula_eval/ppdl_full_cache.json"
PREDS = "benchmarks/compare/formula_eval/preds_ab_before"
MODEL = "models/ppdoclayout/ppdoclayout_plus_l.onnx"


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0


det = PPDocLayoutOnnxDetector(MODEL)
paddle_cache = json.load(open(CACHE, encoding='utf-8'))

norm_imgs = sorted(glob.glob(os.path.join(PREDS, "_tmp_*", "assets", "normalized.png")))
tot_p = tot_o = matched = 0
ious = []
cls_match = 0
for ip in norm_imgs:
    tmp = os.path.basename(os.path.dirname(os.path.dirname(ip)))
    stem = tmp[len("_tmp_"):]
    pboxes = paddle_cache.get(stem, [])
    oboxes = det.detect(ip, conf=0.5)
    tot_p += len(pboxes); tot_o += len(oboxes)
    # match each paddle box to best onnx box
    for pb in pboxes:
        best = 0.0; best_cls = False
        for ob in oboxes:
            v = iou(pb['bbox'], ob['bbox'])
            if v > best:
                best = v; best_cls = (ob['class_name'] == pb['class_name'])
        if best > 0.5:
            matched += 1; ious.append(best)
            if best_cls: cls_match += 1

import statistics
print(f"pages: {len(norm_imgs)}")
print(f"paddle boxes: {tot_p} | onnx boxes: {tot_o}")
print(f"matched (IoU>0.5): {matched}/{tot_p} = {100*matched/max(tot_p,1):.1f}%")
print(f"  of matched, same class: {cls_match}/{matched} = {100*cls_match/max(matched,1):.1f}%")
print(f"mean IoU of matches: {statistics.mean(ious):.4f}" if ious else "no matches")
print("VERDICT:", "PASS" if (matched/max(tot_p,1) > 0.95 and statistics.mean(ious) > 0.95) else "CHECK")
