"""
run_personal_got.py
-------------------
Run GOT-OCR2 on image.png and image2.png, measure latency and RAM.
Then print a full comparison table for all 4 personal images:
  image, image2, glared, glare2
  (glared/glare2 GOT outputs already exist; re-read latency from saved file if present)
"""

import gc, os, time
from pathlib import Path

import psutil

from models_interface import run_page_got, unload_got

process = psutil.Process(os.getpid())


def wrap_got_output(raw: str) -> str:
    preamble = (
        "\\documentclass{article}\n"
        "\\usepackage[margin=2cm]{geometry}\n"
        "\\usepackage{amsmath}\n"
        "\\usepackage{graphicx}\n"
        "\\usepackage{booktabs}\n"
        "\\usepackage[utf8]{inputenc}\n"
        "\\begin{document}\n"
    )
    return preamble + raw + "\n\\end{document}\n"


IMAGES = [
    ("image",  Path("image.png")),
    ("image2", Path("image2.png")),
]

timing = {}

for name, img_path in IMAGES:
    out_dir = Path(f"{name}_output")
    out_dir.mkdir(exist_ok=True)
    got_out = out_dir / "got_main.tex"

    if not img_path.exists():
        print(f"[skip] {img_path} not found")
        continue

    print(f"[GOT] {img_path.name} ...", flush=True)
    t0 = time.perf_counter()
    raw = run_page_got(str(img_path))
    elapsed = time.perf_counter() - t0
    peak_mb = process.memory_info().rss / 1024 / 1024

    doc = wrap_got_output(raw)
    got_out.write_text(doc, encoding="utf-8")
    timing[name] = {"sec": elapsed, "mb": peak_mb}
    print(f"  {elapsed:.1f}s  {peak_mb:.0f}MB  -> {got_out}  ({got_out.stat().st_size} bytes)")

unload_got()
gc.collect()

# ── Summary table ─────────────────────────────────────────────────────────────
# Known GOT latencies from previous session (glared/glare2)
KNOWN_TIMING = {
    "glared": {"sec": 9.0,  "mb": None},
    "glare2": {"sec": 17.9, "mb": None},
}
timing.update(KNOWN_TIMING)

ALL_IMAGES = ["image", "image2", "glared", "glare2"]

print("\n" + "=" * 90)
print("  Personal Images  --  PRISM vs GOT-OCR2")
print("  (No ground truth available -- EDR cannot be computed; output size shown as proxy)")
print("=" * 90)
print(f"  {'Image':<10}  {'PRISM tex':>12}  {'GOT tex':>10}  {'GOT lat':>8}  {'GOT RAM':>9}")
print("  " + "-" * 56)

for name in ALL_IMAGES:
    prism_tex = Path(f"{name}_output/main.tex")
    got_tex   = Path(f"{name}_output/got_main.tex")

    p_size = prism_tex.stat().st_size if prism_tex.exists() else 0
    g_size = got_tex.stat().st_size   if got_tex.exists()   else 0

    t = timing.get(name, {})
    lat_str = f"{t['sec']:.1f}s" if t.get("sec") else "  N/A"
    ram_str = f"{t['mb']:.0f}MB" if t.get("mb") else "  N/A"

    print(f"  {name:<10}  {p_size:>10} B  {g_size:>8} B  {lat_str:>8}  {ram_str:>9}")

print("=" * 90)
print("\n  Outputs:")
for name in ALL_IMAGES:
    p = Path(f"{name}_output/main.tex")
    g = Path(f"{name}_output/got_main.tex")
    print(f"  {name}: PRISM -> {p}  |  GOT -> {g}")
