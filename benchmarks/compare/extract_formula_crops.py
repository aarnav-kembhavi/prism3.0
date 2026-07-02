"""Extract isolated-formula crops (English pages) from OmniDocBench with GT LaTeX.
Writes crops/*.png + gt.json {crop_id: latex}."""
import os, sys, json, ast
from PIL import Image

ROOT = r"C:\PROJECTS\s2l2\testprism"
GT = os.path.join(ROOT, "data/omnidocbench/OmniDocBench_available.json")
IMGDIR = os.path.join(ROOT, "data/omnidocbench/images")
OUT = os.path.join(ROOT, "benchmarks/compare/formula_eval")
CROPS = os.path.join(OUT, "crops")
os.makedirs(CROPS, exist_ok=True)

MAX = int(sys.argv[1]) if len(sys.argv) > 1 else 120

gt = json.load(open(GT, encoding='utf-8'))
records = {}
n = 0
for page in gt:
    attrs = page['page_info']['page_attribute']
    if isinstance(attrs, str):
        attrs = ast.literal_eval(attrs)
    if attrs.get('language') != 'english':
        continue
    ipath = os.path.join(IMGDIR, page['page_info']['image_path'])
    if not os.path.exists(ipath):
        continue
    im = None
    for ann in page.get('layout_dets', []):
        if ann.get('category_type') != 'equation_isolated':
            continue
        latex = ann.get('latex', '')
        if not latex or ann.get('ignore'):
            continue
        p = ann['poly']
        x1, y1, x2, y2 = p[0], p[1], p[4], p[5]
        if x2 - x1 < 16 or y2 - y1 < 8:
            continue
        if im is None:
            im = Image.open(ipath).convert("RGB")
        crop = im.crop((x1, y1, x2, y2))
        cid = f"f{n:04d}"
        crop.save(os.path.join(CROPS, cid + ".png"))
        records[cid] = latex
        n += 1
        if n >= MAX:
            break
    if n >= MAX:
        break

json.dump(records, open(os.path.join(OUT, "gt.json"), "w", encoding='utf-8'), ensure_ascii=False, indent=0)
print(f"extracted {n} English formula crops -> {CROPS}")
