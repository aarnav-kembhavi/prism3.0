"""
Run the frozen 30-page eval set under a given quantization selection.

    python quant/run_variant.py --name texo_decoder --quant texo_decoder
    python quant/run_variant.py --name baseline_fp32 --quant ""
    python quant/run_variant.py --name ppdl_heads_fp32 \
        --env PRISM_QUANT_PPDOCLAYOUT_V3=models/ppdoclayoutv3/PP-DocLayoutV3_int8_heads.onnx

Writes quant/variants/<name>/{*.md, perf.json, meta.json}. meta.json records
which graph file every stage actually resolved to, each graph's on-disk size,
the total stack size, and the latency/RAM numbers -- so a variant's numbers can
never drift from the graphs that produced them.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pipeline.quant_select import GRAPHS  # noqa: E402


def resolve_graphs(env):
    """Resolve every registered graph under `env`, in a clean subprocess."""
    code = (
        "import json,sys; sys.path.insert(0, r'%s');"
        "from pipeline.quant_select import graph_path, GRAPHS;"
        "print(json.dumps({k: graph_path(k) for k in GRAPHS}))" % str(ROOT)
    )
    out = subprocess.run([sys.executable, "-c", code], env=env, cwd=str(ROOT),
                         capture_output=True, text=True)
    line = [l for l in out.stdout.splitlines() if l.startswith("{")]
    if not line:
        raise RuntimeError("could not resolve graphs: " + out.stdout + out.stderr)
    return json.loads(line[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--quant", default="", help="value for PRISM_QUANT (INT8)")
    ap.add_argument("--fp16", default="", help="value for PRISM_FP16")
    ap.add_argument("--fp16-runtime", default="fp32", choices=["fp32", "fp16"],
                    help="fp32 = Variant A (back-convert at load); fp16 = Variant B (native)")
    ap.add_argument("--env", action="append", default=[], help="extra KEY=VALUE")
    ap.add_argument("--images-dir", default="data/omnidocbench_full/images")
    a = ap.parse_args()

    outdir = ROOT / "quant" / "variants" / a.name
    outdir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["PRISM_QUANT"] = a.quant
    env["PRISM_FP16"] = a.fp16
    env["PRISM_FP16_RUNTIME"] = a.fp16_runtime
    env["PRISM_PROBE_LOG"] = str(outdir / "probe.jsonl")
    env["PRISM_QUANT_MANIFEST"] = str(outdir / "graph_manifest.jsonl")
    for kv in a.env:
        k, _, v = kv.partition("=")
        env[k] = v

    resolved = resolve_graphs(env)
    # An explicit PRISM_QUANT_<KEY> override may be given as a repo-relative
    # path; normalise so downstream size lookups and relative_to() work.
    resolved = {k: (p if os.path.isabs(p) else str(ROOT / p)) for k, p in resolved.items()}
    print("[*] graphs for '%s':" % a.name)
    for k, p in resolved.items():
        print("      %-16s %s" % (k, Path(p).name))

    probe_log = outdir / "probe.jsonl"
    if probe_log.exists():
        probe_log.unlink()

    t0 = time.perf_counter()
    with open(outdir / "run.log", "w", encoding="utf-8") as lf:
        rc = subprocess.run(
            [sys.executable, str(ROOT / "benchmarks" / "run_omnidocbench.py"),
             "--gt-json", str(ROOT / "quant" / "eval_gt.json"),
             "--images-dir", str(ROOT / a.images_dir),
             "--pred-dir", str(outdir),
             "--skip-eval"],
            cwd=str(ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
    wall = time.perf_counter() - t0

    perf = {}
    pf = outdir / "perf.json"
    if pf.exists():
        perf = json.loads(pf.read_text(encoding="utf-8"))

    sizes = {k: (os.path.getsize(p) if os.path.exists(p) else None)
             for k, p in resolved.items()}
    lat = sorted((perf.get("per_page") or {}).values())

    # verification-probe acceptance rate (normalization/verified.py proposals)
    accepted = total = 0
    if probe_log.exists():
        for line in probe_log.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            accepted += bool(rec.get("accepted"))

    manifest = outdir / "graph_manifest.jsonl"
    loaded = {}
    back_converts = []
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("event") == "fp16_to_fp32":
                back_converts.append(r["file"])
                continue
            loaded.setdefault(r["key"], set()).add(
                (r["file"], bool(r.get("int8")) or bool(r.get("fp16")),
                 r.get("caller", "?")))
    def _wanted(spec):
        if spec.strip().lower() == "all":
            return set(GRAPHS)
        return {t.strip() for t in spec.split(",") if t.strip()}

    want_int8 = _wanted(a.quant)
    want_fp16 = _wanted(a.fp16)
    # A reduced-precision graph that no process actually opened is the failure
    # mode this sweep already hit once: it looks exactly like "the conversion
    # cost us nothing". Check per requested graph, not per run.
    unproven = sorted(k for k in (want_int8 | want_fp16)
                      if not any(r for _, r, _c in loaded.get(k, ())))
    if unproven:
        print("[!] WARNING: no process recorded loading a reduced-precision "
              "graph for: %s" % ", ".join(unproven))

    meta = {
        "variant": a.name,
        "PRISM_QUANT": a.quant,
        "PRISM_FP16": a.fp16,
        "PRISM_FP16_RUNTIME": a.fp16_runtime,
        "extra_env": a.env,
        "returncode": rc,
        "n_pages": len(list(outdir.glob("*.md"))),
        "graphs": {k: {"path": str(Path(p).relative_to(ROOT)).replace("\\", "/"),
                       "bytes": sizes[k],
                       "mb": round(sizes[k] / 1e6, 2) if sizes[k] else None,
                       "int8": "_int8" in Path(p).name,
                       "fp16": "_fp16" in Path(p).name}
                   for k, p in resolved.items()},
        "total_stack_bytes": sum(v for v in sizes.values() if v),
        "total_stack_mb": round(sum(v for v in sizes.values() if v) / 1e6, 2),
        "wall_s": round(wall, 1),
        "median_latency_s": round(statistics.median(lat), 3) if lat else None,
        "mean_latency_s": round(statistics.fmean(lat), 3) if lat else None,
        "peak_ram_mb_process_tree": perf.get("peak_ram_mb_process_tree"),
        "per_page_latency_s": (perf.get("per_page") or {}),
        "graphs_actually_loaded": {
            k: sorted(f"{c}: {f}" + (" (int8)" if i else "") for f, i, c in v)
            for k, v in sorted(loaded.items())},
        "requested_int8": sorted(want_int8),
        "requested_fp16": sorted(want_fp16),
        "load_unproven": unproven,
        "int8_load_unproven": [k for k in unproven if k in want_int8],
        "fp16_back_conversions": sorted(set(back_converts)),
        "n_fp16_back_conversions": len(back_converts),
        "probe": {"proposals": total, "accepted": accepted,
                  "acceptance_rate": round(accepted / total, 4) if total else None},
    }
    (outdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print("[*] %s: %d pages, rc=%d, median %ss/pg, peak RAM %s MB, stack %.1f MB"
          % (a.name, meta["n_pages"], rc, meta["median_latency_s"],
             meta["peak_ram_mb_process_tree"], meta["total_stack_mb"]))
    print("    probe: %d proposals, %d accepted (%s)"
          % (total, accepted, meta["probe"]["acceptance_rate"]))
    return 0 if rc == 0 else rc


if __name__ == "__main__":
    sys.exit(main())
