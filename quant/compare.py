"""
Degradation check for one quantized variant against the fp32 baseline.

Comparison is against quant/baseline/ -- the fp32 OUTPUT -- not ground truth.
The bar is "structurally correct and readable versus fp32", so this reports
two independent things:

  1. normalized edit-distance similarity per page (rapidfuzz Levenshtein)
  2. structural sanity that edit distance cannot see:
       - LaTeX environments / delimiters balanced
       - table and table-cell counts unchanged vs fp32
       - output non-empty where fp32 was non-empty
       - no runaway repetition in formula decode

The last one matters because a degenerate INT8 decoder that starts looping
produces a long repeated tail; on a short formula inside a long page, edit
distance barely moves while the formula itself is complete garbage.

    python quant/compare.py --variant texo_decoder
"""
import argparse
import difflib
import json
import re
import unicodedata
from pathlib import Path

from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "quant" / "baseline"

MATH_BLOCK = re.compile(r"\\\[(.+?)\\\]|\$\$(.+?)\$\$|\$([^$\n]+?)\$", re.S)
ENV_BEGIN = re.compile(r"\\begin\{([A-Za-z*]+)\}")
ENV_END = re.compile(r"\\end\{([A-Za-z*]+)\}")


def norm_text(s):
    """NFKC + whitespace collapse: ignore cosmetic-only differences."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s)).strip()


def similarity(a, b):
    a, b = norm_text(a), norm_text(b)
    if not a and not b:
        return 1.0
    return Levenshtein.normalized_similarity(a, b)


def math_spans(s):
    out = []
    for m in MATH_BLOCK.finditer(s):
        for g in m.groups():
            if g:
                out.append(g)
                break
    return out


def has_runaway(expr, min_reps=8, max_unit=24):
    """
    Detect a degenerate decode loop: some unit of 1..max_unit chars repeated
    >= min_reps times back to back. Returns a description, or None.
    """
    e = re.sub(r"\s+", "", expr)
    if len(e) < min_reps * 2:
        return None
    for unit in range(1, max_unit + 1):
        if len(e) < unit * min_reps:
            break
        limit = min(len(e) - unit * min_reps, 200)
        for start in range(0, max(limit, 0) + 1):
            seg = e[start:start + unit]
            if not seg.strip():
                continue
            reps = 1
            while e[start + reps * unit: start + (reps + 1) * unit] == seg:
                reps += 1
            if reps >= min_reps:
                return "%dx%dch %r" % (reps, unit, seg)
    return None


def structure(md):
    return {
        "n_chars": len(md.strip()),
        "n_tables": md.count("<table"),
        "n_cells": len(re.findall(r"<td[ >]", md)),
        "n_rows": len(re.findall(r"<tr[ >]", md)),
        "n_math": len(math_spans(md)),
        "env_open": sorted(ENV_BEGIN.findall(md)),
        "env_close": sorted(ENV_END.findall(md)),
        "bracket_open": len(re.findall(r"\\\[", md)),
        "bracket_close": len(re.findall(r"\\\]", md)),
        "dollar_parity_ok": md.count("$") % 2 == 0,
    }


def structural_failures(md, base_md):
    f = []
    s, b = structure(md), structure(base_md)

    if b["n_chars"] > 0 and s["n_chars"] == 0:
        f.append("EMPTY: fp32 produced output, variant produced none")
    if s["env_open"] != s["env_close"]:
        f.append("UNBALANCED env: begin=%s end=%s" % (s["env_open"], s["env_close"]))
    if s["bracket_open"] != s["bracket_close"]:
        f.append("UNBALANCED display math: %d open, %d close"
                 % (s["bracket_open"], s["bracket_close"]))
    if not s["dollar_parity_ok"]:
        f.append("UNBALANCED $: odd number of $ delimiters")
    if s["n_tables"] != b["n_tables"]:
        f.append("TABLE COUNT: %d -> %d" % (b["n_tables"], s["n_tables"]))
    if s["n_cells"] != b["n_cells"]:
        f.append("CELL COUNT: %d -> %d" % (b["n_cells"], s["n_cells"]))
    if s["n_rows"] != b["n_rows"]:
        f.append("ROW COUNT: %d -> %d" % (b["n_rows"], s["n_rows"]))

    base_runaway = set()
    for e in math_spans(base_md):
        r = has_runaway(e)
        if r:
            base_runaway.add(r)
    for e in math_spans(md):
        r = has_runaway(e)
        if r and r not in base_runaway:
            f.append("RUNAWAY formula decode: %s in %r" % (r, e[:70]))
            break
    return f


def side_by_side(name, base_md, var_md, width=78):
    a, b = base_md.splitlines(), var_md.splitlines()
    bar = "=" * (width * 2 + 7)
    out = [bar, " " + name, bar,
           " %-*s | %-*s" % (width, "--- fp32 baseline", width, "+++ variant"),
           "-" * (width * 2 + 7)]
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            n = i2 - i1
            if n > 4:
                for k in list(range(i1, i1 + 2)):
                    out.append(" %-*s | %-*s" % (width, a[k][:width], width, a[k][:width]))
                out.append(" %-*s | ..." % (width, "  ... %d identical lines ..." % (n - 4)))
                for k in list(range(i2 - 2, i2)):
                    out.append(" %-*s | %-*s" % (width, a[k][:width], width, a[k][:width]))
            else:
                for k in range(i1, i2):
                    out.append(" %-*s | %-*s" % (width, a[k][:width], width, a[k][:width]))
        else:
            for k in range(max(i2 - i1, j2 - j1)):
                l = a[i1 + k][:width] if i1 + k < i2 else ""
                r = b[j1 + k][:width] if j1 + k < j2 else ""
                out.append("-%-*s |+%-*s" % (width, l, width, r))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, help="name of dir under quant/variants/")
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--worst", type=int, default=3)
    a = ap.parse_args()

    vdir = ROOT / "quant" / "variants" / a.variant
    pages = sorted(p.stem for p in BASE.glob("*.md"))
    rows = []
    for stem in pages:
        bmd = (BASE / (stem + ".md")).read_text(encoding="utf-8")
        vp = vdir / (stem + ".md")
        if not vp.exists():
            rows.append({"page": stem, "similarity": 0.0,
                         "failures": ["MISSING: variant produced no output file"]})
            continue
        vmd = vp.read_text(encoding="utf-8")
        rows.append({"page": stem,
                     "similarity": round(similarity(bmd, vmd), 5),
                     "failures": structural_failures(vmd, bmd)})

    sims = [r["similarity"] for r in rows]
    below = [r for r in rows if r["similarity"] < a.threshold]
    failed = [r for r in rows if r["failures"]]
    summary = {
        "variant": a.variant,
        "n_pages": len(rows),
        "mean_similarity": round(sum(sims) / len(sims), 5),
        "min_similarity": round(min(sims), 5),
        "median_similarity": round(sorted(sims)[len(sims) // 2], 5),
        "threshold": a.threshold,
        "n_below_threshold": len(below),
        "pages_below": [r["page"] for r in below],
        "n_structural_failures": len(failed),
        "structural_failures": {r["page"]: r["failures"] for r in failed},
        "per_page": {r["page"]: r["similarity"] for r in rows},
    }
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / "compare.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    worst = sorted(rows, key=lambda r: r["similarity"])[:a.worst]
    chunks = []
    for r in worst:
        vp = vdir / (r["page"] + ".md")
        chunks.append(side_by_side(
            "%s   similarity=%s" % (r["page"], r["similarity"]),
            (BASE / (r["page"] + ".md")).read_text(encoding="utf-8"),
            vp.read_text(encoding="utf-8") if vp.exists() else ""))
    (vdir / "worst_diffs.txt").write_text("\n\n".join(chunks), encoding="utf-8")

    print("[%s] mean=%.4f min=%.4f below_%.2f=%d structural_failures=%d"
          % (a.variant, summary["mean_similarity"], summary["min_similarity"],
             a.threshold, len(below), len(failed)))
    for r in failed:
        print("  ! %s: %s" % (r["page"], r["failures"][0]))


if __name__ == "__main__":
    main()
