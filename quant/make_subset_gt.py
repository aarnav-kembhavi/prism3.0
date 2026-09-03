"""Write the OmniDocBench GT subset containing exactly the 30 frozen eval pages."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
full = json.load(open(ROOT / "data/omnidocbench_full/OmniDocBench.json", encoding="utf-8"))
spec = json.load(open(ROOT / "quant/eval_pages.json", encoding="utf-8"))
want = {p["image"] for p in spec["pages"]}

subset = [r for r in full if r["page_info"]["image_path"] in want]
assert len(subset) == len(want), f"{len(subset)} != {len(want)}"

out = ROOT / "quant/eval_gt.json"
out.write_text(json.dumps(subset, ensure_ascii=False), encoding="utf-8")
print(f"wrote {out.relative_to(ROOT)}: {len(subset)} pages")
