"""
Verification-probe acceptance rate, fp32 det vs INT8 det.

PP-OCRv6 det is used twice: once for OCR text boxes, and again at 640px inside
normalization/verified.py as the probe that accepts or rejects every
normalization proposal (white balance, rectify, glare, shadow, moire, CLAHE).
Quantizing it therefore moves a decision gate, not just box coordinates -- a
shifted probe score changes which corrections get applied to the page.

That gate cannot be measured on the normal eval path: benchmarks/run_omnidocbench.py
sets PRISM_NORM_STRICT=1, and normalization/pipeline.py:234 disables the whole
verified path when it is set. So this harness runs normalization ONLY (no OCR,
no layout) with PRISM_NORM_STRICT=0, once per det variant, and counts
proposals accepted.

It also cannot be measured on the 30 eval pages themselves: those are scans and
digital renders, and detect_capture_modality routes them past the whole Stage-1
correction stack, so no proposal is ever made (measured: 0 in both arms). The
probe is a camera-capture feature, so it is measured on the repo's real
camera/defect captures.

    python quant/probe_rate.py                 # runs both arms, writes report
"""
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CHILD = r'''
import json, os, sys
sys.path.insert(0, r"{root}")
from normalization import normalize_image_pil
pages = json.load(open(r"{pageset}", encoding="utf-8"))
from normalization import verified as nv
for p in pages:
    nv.log_context = {{"page": p["id"], "bucket": p["bucket"]}}
    try:
        normalize_image_pil(p["path"])
    except Exception as e:
        print("ERR", p["id"], type(e).__name__, e, file=sys.stderr)
'''


def run_arm(name, quant_value, pageset):
    log = ROOT / "quant" / f"probe_{name}.jsonl"
    if log.exists():
        log.unlink()
    env = dict(os.environ)
    env["PRISM_NORM_STRICT"] = "0"          # enable the verified path
    env["PRISM_NORM_VERIFY"] = "1"
    env["PRISM_PROBE_LOG"] = str(log)
    env["PRISM_QUANT"] = quant_value
    print(f"[*] probe arm '{name}' (PRISM_QUANT={quant_value!r}) ...")
    r = subprocess.run([sys.executable, "-c", CHILD.format(root=str(ROOT), pageset=str(pageset))],
                       cwd=str(ROOT), env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise SystemExit(f"probe arm {name} failed rc={r.returncode}")

    recs = []
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return recs


def summarize(recs):
    total = len(recs)
    acc = sum(1 for r in recs if r.get("accepted"))
    by_step = {}
    for r in recs:
        k = r.get("correction", "?")
        s = by_step.setdefault(k, {"proposals": 0, "accepted": 0})
        s["proposals"] += 1
        s["accepted"] += bool(r.get("accepted"))
    per_page = {}
    for r in recs:
        p = r.get("page", "?")
        s = per_page.setdefault(p, {"proposals": 0, "accepted": 0})
        s["proposals"] += 1
        s["accepted"] += bool(r.get("accepted"))
    return {
        "proposals": total,
        "accepted": acc,
        "acceptance_rate": round(acc / total, 4) if total else None,
        "by_step": {k: {**v, "rate": round(v["accepted"] / v["proposals"], 4)}
                    for k, v in sorted(by_step.items())},
        "per_page": per_page,
        "outcomes": dict(Counter(r.get("outcome", "?") for r in recs)),
    }


def build_pageset():
    """
    The 30 eval pages are scans/digital renders: detect_capture_modality routes
    them past the whole Stage-1 correction stack, so the probe never fires and
    the gate cannot be observed there (measured: 0 proposals, both arms). The
    probe is a CAMERA-capture feature, so the gate is measured on the repo's
    real camera/defect captures instead.
    """
    out, seen = [], set()
    for d in ["test_images/real/defects", "test_images/real/misc",
              "test_images/real/handwritten", "test_images/real/clean"]:
        base = ROOT / d
        if not base.exists():
            continue
        for f in sorted(base.rglob("*")):
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"} and f.name not in seen:
                seen.add(f.name)
                out.append({"id": f.stem, "bucket": d.rsplit("/", 1)[-1],
                            "path": str(f)})
    ps = ROOT / "quant" / "probe_pageset.json"
    ps.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[*] probe page set: %d images" % len(out))
    return ps


def main():
    pageset = build_pageset()
    fp32 = summarize(run_arm("fp32", "", pageset))
    int8 = summarize(run_arm("det_int8", "ppocr_det", pageset))

    delta = None
    if fp32["acceptance_rate"] is not None and int8["acceptance_rate"] is not None:
        delta = round((int8["acceptance_rate"] - fp32["acceptance_rate"]) * 100, 2)

    flips = []
    for page, a in fp32["per_page"].items():
        b = int8["per_page"].get(page, {"accepted": None})
        if b["accepted"] != a["accepted"]:
            flips.append({"page": page, "fp32_accepted": a["accepted"],
                          "int8_accepted": b["accepted"]})

    report = {
        "description": "PP-OCRv6 det 640px verification-probe acceptance, fp32 vs INT8, "
                       "on camera/defect captures (the 30 OmniDocBench eval pages are "
                       "scans/digital renders, where the probe never fires at all). "
                       "PRISM_NORM_STRICT=0 so the verified path is active.",
        "fp32": fp32, "det_int8": int8,
        "acceptance_rate_delta_pp": delta,
        "n_pages_with_different_accept_count": len(flips),
        "pages_changed": flips,
    }
    out = ROOT / "quant" / "probe_rate.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== verification probe acceptance (camera/defect captures) ===")
    print("  fp32 det : %4d proposals, %4d accepted (%s)"
          % (fp32["proposals"], fp32["accepted"], fp32["acceptance_rate"]))
    print("  int8 det : %4d proposals, %4d accepted (%s)"
          % (int8["proposals"], int8["accepted"], int8["acceptance_rate"]))
    print("  delta    : %s pp" % delta)
    print("  pages whose accept count changed: %d" % len(flips))
    print("\n  by normalization step:")
    for k in sorted(set(fp32["by_step"]) | set(int8["by_step"])):
        a = fp32["by_step"].get(k, {"accepted": 0, "proposals": 0, "rate": None})
        b = int8["by_step"].get(k, {"accepted": 0, "proposals": 0, "rate": None})
        print("    %-14s fp32 %2d/%-2d (%s)   int8 %2d/%-2d (%s)"
              % (k, a["accepted"], a["proposals"], a["rate"],
                 b["accepted"], b["proposals"], b["rate"]))
    print("\nwrote", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
