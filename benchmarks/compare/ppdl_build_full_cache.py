"""Run PP-DocLayout_plus-L on persisted NORMALIZED images and cache ALL layout
boxes (mapped to PRISM class names) keyed by image stem — for validating the
'replace both detectors' path. Run in .venv_ppocr.

usage: python ppdl_build_full_cache.py <preds_dir_with_tmp> <out_cache.json>
"""
import os, sys, json, glob
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
sys.path.insert(0, r"C:\PROJECTS\s2l2\testprism")
os.chdir(r"C:\PROJECTS\s2l2\testprism")

# PP-DocLayout_plus-L (20 labels) -> PRISM class vocab
PPDL2PRISM = {
    'text': 'Text', 'abstract': 'Text', 'content': 'Text', 'reference': 'Text',
    'reference_content': 'Text', 'aside_text': 'Text', 'algorithm': 'Text',
    'footnote': 'Footnote',
    'paragraph_title': 'Section-header', 'doc_title': 'Title', 'figure_title': 'Caption',
    'header': 'Page-header', 'footer': 'Page-footer', 'number': 'Page-header',
    'formula': 'Formula',
    'table': 'Table',
    'image': 'Picture', 'chart': 'Picture',
    # dropped: formula_number (eq numbers, GT ignores), seal (stamps)
}

preds_dir = sys.argv[1]
out_path = sys.argv[2]

import paddle; paddle.set_device('cpu')
from paddleocr import LayoutDetection
m = LayoutDetection(model_name='PP-DocLayout_plus-L')

norm_imgs = sorted(glob.glob(os.path.join(preds_dir, "_tmp_*", "assets", "normalized.png")))
cache = {}
total = 0
dropped = 0
for ip in norm_imgs:
    tmp_dir = os.path.basename(os.path.dirname(os.path.dirname(ip)))
    stem = tmp_dir[len("_tmp_"):]
    res = list(m.predict(ip))
    boxes = res[0]['boxes'] if res else []
    out = []
    for b in boxes:
        cls = PPDL2PRISM.get(b['label'])
        if cls is None:
            dropped += 1
            continue
        out.append({'bbox': [float(c) for c in b['coordinate']],
                    'class_name': cls,
                    'confidence': float(b['score'])})
    cache[stem] = out
    total += len(out)

json.dump(cache, open(out_path, "w", encoding='utf-8'), ensure_ascii=False)
print(f"cached {total} boxes ({dropped} dropped) across {len(cache)} pages -> {out_path}")
