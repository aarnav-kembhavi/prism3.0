"""Run PP-FormulaNet-S on the extracted formula crops, save predictions + latency."""
import os, sys, json, time, glob, statistics
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

ROOT = r"C:\PROJECTS\s2l2\testprism"
OUT = os.path.join(ROOT, "benchmarks/compare/formula_eval")

import paddle
paddle.set_device('cpu')
from paddleocr import FormulaRecognition

model_name = sys.argv[1] if len(sys.argv) > 1 else 'PP-FormulaNet-S'
model = FormulaRecognition(model_name=model_name)

def extract_latex(res):
    # paddleocr 3.x result objects: try common fields
    for attr in ('rec_formula', 'rec_text', 'formula'):
        v = getattr(res, attr, None)
        if isinstance(v, str) and v:
            return v
    d = res if isinstance(res, dict) else getattr(res, 'json', None) or getattr(res, '__dict__', {})
    if isinstance(d, dict):
        for k in ('rec_formula', 'rec_text', 'formula', 'text'):
            if isinstance(d.get(k), str) and d[k]:
                return d[k]
    return str(res)

crops = sorted(glob.glob(os.path.join(OUT, "crops", "*.png")))
preds = {}
lat = []
for cp in crops:
    cid = os.path.splitext(os.path.basename(cp))[0]
    t = time.perf_counter()
    results = list(model.predict(cp))
    lat.append(time.perf_counter() - t)
    preds[cid] = extract_latex(results[0]) if results else ''

tag = model_name.replace('/', '_')
json.dump(preds, open(os.path.join(OUT, f"pred_{tag}.json"), "w", encoding='utf-8'), ensure_ascii=False)
print(f"{model_name}: {len(preds)} crops, median {statistics.median(lat)*1000:.0f}ms/crop, mean {statistics.mean(lat)*1000:.0f}ms")
# print one sample to verify extraction worked
first = next(iter(preds.values()))
print("sample pred:", (first[:80] if first else '(EMPTY - check extract_latex)'))
