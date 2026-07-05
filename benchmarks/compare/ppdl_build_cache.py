"""Run PP-DocLayout_plus-L on the persisted NORMALIZED images from a benchmark
run and cache 'formula' boxes keyed by image stem. Coordinates are in the
normalized-image space, so the benchmark can add them directly. Run in .venv_ppocr.

usage: python ppdl_build_cache.py <preds_dir_with_tmp> <out_cache.json>
"""
import os, sys, json, glob
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
sys.path.insert(0, r"C:\PROJECTS\s2l2\testprism")
os.chdir(r"C:\PROJECTS\s2l2\testprism")

preds_dir = sys.argv[1]
out_path = sys.argv[2]

import paddle; paddle.set_device('cpu')
from paddleocr import LayoutDetection
m = LayoutDetection(model_name='PP-DocLayout_plus-L')

norm_imgs = sorted(glob.glob(os.path.join(preds_dir, "_tmp_*", "assets", "normalized.png")))
cache = {}
total = 0
for ip in norm_imgs:
    # stem = the _tmp_<stem> folder name minus the _tmp_ prefix
    tmp_dir = os.path.basename(os.path.dirname(os.path.dirname(ip)))
    stem = tmp_dir[len("_tmp_"):]
    res = list(m.predict(ip))
    boxes = res[0]['boxes'] if res else []
    fboxes = [{'bbox': [float(c) for c in b['coordinate']],
               'confidence': float(b['score'])}
              for b in boxes if b['label'] == 'formula']
    cache[stem] = fboxes
    total += len(fboxes)

json.dump(cache, open(out_path, "w", encoding='utf-8'), ensure_ascii=False)
print(f"cached formula boxes for {len(cache)} pages, {total} boxes total -> {out_path}")
