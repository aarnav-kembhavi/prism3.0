"""Assemble quant/baseline_meta.json from the fp32 baseline run."""
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pipeline.quant_select import GRAPHS  # noqa: E402

BASE = ROOT / "quant" / "baseline"
perf = json.loads((BASE / "perf.json").read_text(encoding="utf-8"))
spec = json.loads((ROOT / "quant" / "eval_pages.json").read_text(encoding="utf-8"))

graphs = {}
for key, rel in GRAPHS.items():
    p = ROOT / rel
    graphs[key] = {"path": rel, "bytes": p.stat().st_size,
                   "mb": round(p.stat().st_size / 1e6, 2), "exists": p.exists()}

per_page = perf["per_page"]
bucket_of = {p["id"]: p["bucket"] for p in spec["pages"]}
by_bucket = {}
for stem, sec in per_page.items():
    by_bucket.setdefault(bucket_of.get(stem, "?"), []).append(sec)

md_chars = {p.stem: len(p.read_text(encoding="utf-8").strip())
            for p in sorted(BASE.glob("*.md"))}

meta = {
    "description": "fp32 baseline for the PRISM quantization sweep. All later "
                   "variants are compared against quant/baseline/*.md.",
    "git_tag": "fp32-baseline",
    "eval_set": "quant/eval_pages.json",
    "n_pages": perf["n_pages"],
    "graphs": graphs,
    "total_stack_bytes": sum(g["bytes"] for g in graphs.values()),
    "total_stack_mb": round(sum(g["bytes"] for g in graphs.values()) / 1e6, 2),
    "wall_s": perf["wall_s"],
    "latency_s_per_page": perf["latency_s_per_page"],
    "peak_ram_mb_process_tree": perf["peak_ram_mb_process_tree"],
    "ram_mb_process_tree": perf["ram_mb_process_tree"],
    "per_page_latency_s": per_page,
    "latency_by_bucket_s": {b: {"n": len(v), "median": round(statistics.median(v), 3),
                                "mean": round(statistics.fmean(v), 3)}
                            for b, v in sorted(by_bucket.items())},
    "per_page_output_chars": md_chars,
    "n_empty_outputs": sum(1 for v in md_chars.values() if v == 0),
    "notes": [
        "PRISM_NORM_STRICT=1 is set by benchmarks/run_omnidocbench.py, which "
        "disables the recognition-verified normalization path "
        "(normalization/pipeline.py:234). The 640px PP-OCRv6 det verification "
        "probe therefore does NOT run on this eval path; probe acceptance is "
        "measured separately by quant/probe_rate.py.",
    ],
}
out = ROOT / "quant" / "baseline_meta.json"
out.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
print("wrote", out.relative_to(ROOT))
print("  pages           :", meta["n_pages"], "| empty:", meta["n_empty_outputs"])
print("  stack           : %.1f MB across %d graphs" % (meta["total_stack_mb"], len(graphs)))
print("  median latency  : %ss/page" % meta["latency_s_per_page"]["median"])
print("  peak RAM        : %s MB" % meta["peak_ram_mb_process_tree"])
for b, v in meta["latency_by_bucket_s"].items():
    print("    %-14s n=%d median=%ss" % (b, v["n"], v["median"]))
