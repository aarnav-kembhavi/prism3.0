"""
Pick ~100 held-out calibration pages for static quantization.

Hard constraint: zero overlap with the frozen 30-page eval set. Spread over
data_source x language so the activation ranges the calibrator sees cover the
same variety the eval set does (a calibration set of only clean English books
gives clipped ranges on CJK and camera pages).
"""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
N = 100

full = json.load(open(ROOT / "data/omnidocbench_full/OmniDocBench.json", encoding="utf-8"))
eval_imgs = {p["image"] for p in json.load(
    open(ROOT / "quant/eval_pages.json", encoding="utf-8"))["pages"]}

by_strata = defaultdict(list)
for r in full:
    img = r["page_info"]["image_path"]
    if img in eval_imgs:
        continue
    a = r["page_info"]["page_attribute"]
    by_strata[(a.get("data_source"), a.get("language"))].append(img)

for k in by_strata:
    by_strata[k].sort()

picked, strata = [], sorted(by_strata)
while len(picked) < N and any(by_strata.values()):
    for s in strata:
        if by_strata[s] and len(picked) < N:
            picked.append(by_strata[s].pop(0))

assert not (set(picked) & eval_imgs), "calibration set overlaps eval set"
assert len(picked) == len(set(picked)) == N

(ROOT / "quant/calib_pages.json").write_text(json.dumps({
    "description": f"{N} held-out calibration pages for static quantization. "
                   "Disjoint from quant/eval_pages.json by construction.",
    "image_dir": "data/omnidocbench_full/images",
    "n": N,
    "overlap_with_eval_set": 0,
    "pages": picked,
}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"wrote quant/calib_pages.json: {len(picked)} pages, "
      f"{len({s for s in strata if s})} strata, overlap with eval = 0")
