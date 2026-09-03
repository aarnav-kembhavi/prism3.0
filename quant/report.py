"""
Build the per-variant results table for the quantization sweep.

Reads quant/baseline_meta.json plus every quant/variants/*/{meta.json,compare.json}
and emits a markdown table plus a machine-readable roll-up.

    python quant/report.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VAR = ROOT / "quant" / "variants"

# graph a variant is *about*, for the "graph size before/after" column
VARIANT_GRAPH = {
    "texo_decoder": "texo_decoder",
    "texo_encoder": "texo_encoder",
    "ppocr_rec": "ppocr_rec",
    "ppocr_det": "ppocr_det",
    "ppdoclayout_v3": "ppdoclayout_v3",
    "ppdoclayout_v3_heads_fp32": "ppdoclayout_v3",
    "slanet_plus": "slanet_plus",
}

ORDER = ["texo_decoder", "texo_encoder", "ppocr_rec", "ppocr_det",
         "ppdoclayout_v3", "ppdoclayout_v3_heads_fp32", "slanet_plus",
         "combo_conservative", "combo_greedy"]


def load(name):
    d = VAR / name
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8")) if (d / "meta.json").exists() else None
    comp = json.loads((d / "compare.json").read_text(encoding="utf-8")) if (d / "compare.json").exists() else None
    return meta, comp


def main():
    base = json.loads((ROOT / "quant" / "baseline_meta.json").read_text(encoding="utf-8"))
    base_graphs = base["graphs"]
    base_stack = base["total_stack_mb"]

    names = [n for n in ORDER if (VAR / n).is_dir()]
    names += sorted(p.name for p in VAR.iterdir()
                    if p.is_dir() and p.name not in names)

    rows, roll = [], {}
    for n in names:
        meta, comp = load(n)
        if not meta:
            continue
        gkey = VARIANT_GRAPH.get(n)
        if gkey:
            before = base_graphs[gkey]["mb"]
            after = meta["graphs"][gkey]["mb"]
            gcol = "%.1f -> %.1f" % (before, after)
            saved = before - after
        else:
            int8 = [k for k, v in meta["graphs"].items() if v["int8"]]
            gcol = "%d graphs INT8" % len(int8)
            saved = base_stack - meta["total_stack_mb"]

        row = {
            "variant": n,
            "graph_mb": gcol,
            "stack_mb": meta["total_stack_mb"],
            "stack_saved_mb": round(base_stack - meta["total_stack_mb"], 1),
            "median_lat_s": meta["median_latency_s"],
            "peak_ram_mb": meta["peak_ram_mb_process_tree"],
            "mean_sim": comp["mean_similarity"] if comp else None,
            "min_sim": comp["min_similarity"] if comp else None,
            "n_below_095": comp["n_below_threshold"] if comp else None,
            "n_struct_fail": comp["n_structural_failures"] if comp else None,
            "saved_mb": round(saved, 1),
        }
        if comp and saved > 0:
            # quality cost per MB saved: how much mean similarity fell per MB
            row["cost_per_mb"] = round((1.0 - comp["mean_similarity"]) / saved, 6)
        rows.append(row)
        roll[n] = row

    hdr = ("| variant | graph MB (before -> after) | stack MB | saved MB | median s/pg "
           "| peak RAM MB | mean sim | min sim | pages <0.95 | struct fails |")
    sep = "|" + "---|" * 11
    lines = [hdr, sep]
    lines.append("| **fp32 baseline** | %.1f (all) | %.1f | 0.0 | %s | %s | 1.00000 | 1.00000 | 0 | 0 |"
                 % (base_stack, base_stack, base["latency_s_per_page"]["median"],
                    base["peak_ram_mb_process_tree"]))
    for r in rows:
        lines.append("| %s | %s | %.1f | %.1f | %s | %s | %s | %s | %s | %s |" % (
            r["variant"], r["graph_mb"], r["stack_mb"], r["stack_saved_mb"],
            r["median_lat_s"], r["peak_ram_mb"],
            "%.5f" % r["mean_sim"] if r["mean_sim"] is not None else "-",
            "%.5f" % r["min_sim"] if r["min_sim"] is not None else "-",
            r["n_below_095"] if r["n_below_095"] is not None else "-",
            r["n_struct_fail"] if r["n_struct_fail"] is not None else "-"))

    table = "\n".join(lines)
    (ROOT / "quant" / "results_table.md").write_text(table + "\n", encoding="utf-8")
    (ROOT / "quant" / "results.json").write_text(
        json.dumps({"baseline": {"stack_mb": base_stack,
                                 "median_lat_s": base["latency_s_per_page"]["median"],
                                 "peak_ram_mb": base["peak_ram_mb_process_tree"]},
                    "variants": roll}, indent=2), encoding="utf-8")
    print(table)

    scored = [r for r in rows if r.get("cost_per_mb") is not None
              and r["variant"] in VARIANT_GRAPH]
    if scored:
        worst = max(scored, key=lambda r: r["cost_per_mb"])
        print("\nworst quality per MB saved: %s "
              "(mean sim %.5f, %.1f MB saved, %.2e similarity lost per MB)"
              % (worst["variant"], worst["mean_sim"], worst["saved_mb"],
                 worst["cost_per_mb"]))


if __name__ == "__main__":
    sys.exit(main())
