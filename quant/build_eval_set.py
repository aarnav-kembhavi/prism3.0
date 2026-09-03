"""
Build the fixed 30-page quantization eval set from OmniDocBench.

Selection is deterministic (no RNG): candidates are filtered by rule, then
round-robined over data_source so each bucket spans as many document types as
possible. Chosen for DIVERSITY, not difficulty -- the 'hard' subsets
(equation_hard / table_hard / layout_hard) are allowed but never allowed to
dominate a bucket.

Buckets are disjoint and assigned most-constrained-first:
    degraded -> formula -> table -> text_multicol -> cjk

Output: quant/eval_pages.json  (frozen; never regenerate)
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC  = ROOT / "data" / "omnidocbench_full" / "OmniDocBench.json"
OUT  = ROOT / "quant" / "eval_pages.json"

PER_BUCKET = 6
DEGRADED_ISSUES = {
    "fuzzy_scan", "geometric_deformation", "watermark", "with_watermark",
    "fuzzy_content", "transparent_pages", "handwriting",
}
CJK_LANGS = {"simplified_chinese", "traditional_chinese", "en_ch_mixed"}
MULTICOL  = {"double_column", "three_column", "1andmore_column"}


def summarize(rec):
    attr = rec["page_info"]["page_attribute"]
    cats = Counter(d.get("category_type") for d in rec["layout_dets"])
    return {
        "image": rec["page_info"]["image_path"],
        "source": attr.get("data_source"),
        "language": attr.get("language"),
        "layout": attr.get("layout"),
        "subset": attr.get("subset"),
        "issues": set(attr.get("special_issue") or []),
        "n_eq": cats.get("equation_isolated", 0) + cats.get("equation_semantic", 0),
        "n_table": cats.get("table", 0),
        "n_text": cats.get("text_block", 0),
        "n_figure": cats.get("figure", 0),
    }


def pick(cands, n, taken):
    """Round-robin over data_source for spread; deterministic tie-break by image name."""
    by_src = defaultdict(list)
    for c in sorted(cands, key=lambda c: c["image"]):
        if c["image"] not in taken:
            by_src[c["source"]].append(c)
    out, srcs = [], sorted(by_src)
    while len(out) < n and any(by_src.values()):
        for s in srcs:
            if by_src[s] and len(out) < n:
                out.append(by_src[s].pop(0))
    for c in out:
        taken.add(c["image"])
    return out


def main():
    recs = [summarize(r) for r in json.load(open(SRC, encoding="utf-8"))]
    taken = set()
    buckets = {}

    # 1. degraded / camera-capture -- rarest signal, assign first
    buckets["degraded"] = pick(
        [r for r in recs if r["issues"] & DEGRADED_ISSUES and r["n_text"] >= 2],
        PER_BUCKET, taken)

    # 2. formula-heavy -- >=3 equations, not dominated by tables
    buckets["formula"] = pick(
        [r for r in recs if r["n_eq"] >= 3 and r["n_table"] == 0],
        PER_BUCKET, taken)

    # 3. table-heavy -- at least one real table, some surrounding text
    buckets["table"] = pick(
        [r for r in recs if r["n_table"] >= 1 and r["n_text"] >= 2],
        PER_BUCKET, taken)

    # 4. plain text / multi-column -- latin script, no tables, few equations
    buckets["text_multicol"] = pick(
        [r for r in recs
         if r["layout"] in MULTICOL and r["language"] == "english"
         and r["n_table"] == 0 and r["n_eq"] <= 1 and r["n_text"] >= 4],
        PER_BUCKET, taken)

    # 5. CJK
    buckets["cjk"] = pick(
        [r for r in recs if r["language"] in CJK_LANGS and r["n_text"] >= 3],
        PER_BUCKET, taken)

    pages = []
    for bucket, items in buckets.items():
        for r in items:
            pages.append({
                "id": r["image"].rsplit(".", 1)[0],
                "image": r["image"],
                "bucket": bucket,
                "source": r["source"],
                "language": r["language"],
                "layout": r["layout"],
                "subset": r["subset"],
                "special_issue": sorted(r["issues"]),
                "n_equations": r["n_eq"],
                "n_tables": r["n_table"],
                "n_text_blocks": r["n_text"],
            })

    doc = {
        "description": "Frozen 30-page eval set for the PRISM quantization sweep. "
                       "Chosen for diversity, not difficulty. Do not regenerate.",
        "source_json": str(SRC.relative_to(ROOT)).replace("\\", "/"),
        "image_dir": "data/omnidocbench_full/images",
        "buckets": {k: len(v) for k, v in buckets.items()},
        "pages": pages,
    }
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    for b, items in buckets.items():
        print(f"\n=== {b} ({len(items)}) ===")
        for r in items:
            print(f"  [{r['source']:<20}] {r['language']:<18} {r['layout']:<15} "
                  f"eq={r['n_eq']:<3} tbl={r['n_table']:<3} {sorted(r['issues'])}")
    print(f"\ntotal: {len(pages)}  unique: {len({p['image'] for p in pages})}")


if __name__ == "__main__":
    main()
