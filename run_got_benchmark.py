"""
run_got_benchmark.py
────────────────────
Run GOT-OCR2 on:
  1. First 26 PDF2LaTeX benchmark images
  2. glared.jpeg and glare2.jpeg

Then print a full comparison table:
  InftyReader vs PRISM vs GOT-OCR2
  Metrics: overall EDR, text EDR, math EDR, latency, peak RAM
"""

import csv, gc, os, statistics, time
from pathlib import Path

import psutil, torch
from PIL import Image

GT_DIR        = Path("pdf2latex_dataset/dataset")
IMAGES_DIR    = Path("benchmark_results/temp_images")
PRISM_TEX_DIR = Path("benchmark_results/prism_tex_26")
PRISM_LAT_CSV = Path("benchmark_results/latency_log_26.csv")
GOT_OUT_DIR   = Path("benchmark_results/got_tex_26")
GOT_OUT_DIR.mkdir(parents=True, exist_ok=True)

from normalization import normalize_image_pil
from models_interface import run_page_got, unload_got
from latex_builder import assemble_document, save_tex
from evaluation.normalizer import normalize_latex, split_math_and_text
from run_full_benchmark import compute_edr

# ── Helpers ───────────────────────────────────────────────────────────────────
def eval_pair(pred_path, gt_path):
    pred = normalize_latex(Path(pred_path).read_text(encoding="utf-8", errors="ignore"),
                           remove_spaces=True)
    gt   = normalize_latex(Path(gt_path).read_text(  encoding="utf-8", errors="ignore"),
                           remove_spaces=True)
    pm, pt = split_math_and_text(pred)
    gm, gt_= split_math_and_text(gt)
    return compute_edr(pred, gt), compute_edr(pm, gm), compute_edr(pt, gt_)


def wrap_got_output(raw: str) -> str:
    """Wrap raw GOT LaTeX output in a minimal document."""
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


# ── Load PRISM latency CSV ────────────────────────────────────────────────────
prism_lat = {}
prism_ram = {}
if PRISM_LAT_CSV.exists():
    for row in csv.DictReader(open(PRISM_LAT_CSV)):
        prism_lat[row["page_id"]] = float(row["total_sec"])
        prism_ram[row["page_id"]] = float(row["peak_mb"])

# ── Collect 26 benchmark images ───────────────────────────────────────────────
all_imgs = sorted(
    [p for p in IMAGES_DIR.glob("*.png")
     if (GT_DIR / f"{p.stem}_gt.tex").exists()],
    key=lambda p: int(p.stem),
)[:26]

print(f"[*] Running GOT-OCR2 on {len(all_imgs)} benchmark pages\n")
process = psutil.Process(os.getpid())

got_rows = []
for i, img_path in enumerate(all_imgs, 1):
    pid = img_path.stem
    out_path = GOT_OUT_DIR / f"{pid}_got.tex"

    print(f"[{i:>2}/26] page {pid}", flush=True)
    t0 = time.perf_counter()
    raw = run_page_got(str(img_path))
    elapsed = time.perf_counter() - t0
    peak_mb = process.memory_info().rss / 1024 / 1024

    doc = wrap_got_output(raw)
    out_path.write_text(doc, encoding="utf-8")

    ov, ma, tx = eval_pair(out_path, GT_DIR / f"{pid}_gt.tex")
    got_rows.append({"pid": pid, "overall": ov, "math": ma, "text": tx,
                     "sec": elapsed, "mb": peak_mb})
    print(f"  {elapsed:.1f}s  {peak_mb:.0f}MB  EDR={ov:.1%}  text={tx:.1%}  math={ma:.1%}")

unload_got()
gc.collect()

# ── Evaluate InftyReader & PRISM on same 26 pages ─────────────────────────────
infty_rows, prism_rows = [], []
for img_path in all_imgs:
    pid = img_path.stem
    gt  = GT_DIR / f"{pid}_gt.tex"

    ov, ma, tx = eval_pair(GT_DIR / f"{pid}_infty.tex", gt)
    infty_rows.append({"pid": pid, "overall": ov, "math": ma, "text": tx,
                       "sec": None, "mb": None})

    ov, ma, tx = eval_pair(PRISM_TEX_DIR / f"{pid}_prism.tex", gt)
    prism_rows.append({"pid": pid, "overall": ov, "math": ma, "text": tx,
                       "sec": prism_lat.get(pid), "mb": prism_ram.get(pid)})

# ── Per-page table ─────────────────────────────────────────────────────────────
print("\n" + "="*100)
print("  InftyReader vs PRISM vs GOT-OCR2  --  First 26 PDF2LaTeX pages  (Overall EDR)")
print("="*100)
print(f"  {'Pg':>3}  | {'Infty':>7}  | {'PRISM':>7} {'lat':>5} {'RAM':>6}  | {'GOT-OCR2':>8} {'lat':>6} {'RAM':>6}")
print(f"  {'-'*95}")

for ir, pr, gr in zip(infty_rows, prism_rows, got_rows):
    p_lat = f"{pr['sec']:.1f}s" if pr['sec'] else "  N/A"
    p_ram = f"{pr['mb']:.0f}MB" if pr['mb'] else "  N/A"
    print(f"  {ir['pid']:>3}  | {ir['overall']:>7.1%}  |"
          f" {pr['overall']:>7.1%} {p_lat:>5} {p_ram:>6}  |"
          f" {gr['overall']:>8.1%} {gr['sec']:>5.1f}s {gr['mb']:>5.0f}MB")

print(f"  {'-'*95}")

def avg(rows, k): return statistics.mean(r[k] for r in rows)
def med(rows, k): return statistics.median(r[k] for r in rows)

i_avg = avg(infty_rows, "overall")
p_avg = avg(prism_rows, "overall")
g_avg = avg(got_rows,   "overall")
print(f"  {'AVG':>3}  | {i_avg:>7.1%}  |"
      f" {p_avg:>7.1%} {avg(prism_rows,'sec'):>4.1f}s {avg(prism_rows,'mb'):>5.0f}MB  |"
      f" {g_avg:>8.1%} {avg(got_rows,'sec'):>5.1f}s {avg(got_rows,'mb'):>5.0f}MB")
print(f"  {'MED':>3}  | {med(infty_rows,'overall'):>7.1%}  |"
      f" {med(prism_rows,'overall'):>7.1%}                |"
      f" {med(got_rows,'overall'):>8.1%}")

print("="*100)

# ── Summary by metric ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  Summary  --  Averages across 26 pages")
print("="*60)
print(f"  {'Metric':<16}  {'InftyReader':>11}  {'PRISM':>8}  {'GOT-OCR2':>10}")
print(f"  {'-'*56}")
for key, label in [("overall","Overall EDR"), ("text","Text EDR"), ("math","Math EDR")]:
    print(f"  {label:<16}  {avg(infty_rows,key):>11.1%}  {avg(prism_rows,key):>8.1%}  {avg(got_rows,key):>10.1%}")
print(f"  {'-'*56}")
print(f"  {'Avg latency':<16}  {'N/A':>11}  {avg(prism_rows,'sec'):>7.1f}s  {avg(got_rows,'sec'):>9.1f}s")
print(f"  {'Avg peak RAM':<16}  {'N/A':>11}  {avg(prism_rows,'mb'):>7.0f}MB  {avg(got_rows,'mb'):>9.0f}MB")
print("="*60)

# ── Glare images ──────────────────────────────────────────────────────────────
print("\n\n[*] Running GOT-OCR2 on glared.jpeg and glare2.jpeg\n")

GLARE_IMGS = [Path("glared.jpeg"), Path("glare2.jpeg")]
glare_results = {}

from models_interface import run_page_got
_got_loaded = False

for img_path in GLARE_IMGS:
    if not img_path.exists():
        print(f"  [skip] {img_path} not found"); continue

    name = img_path.stem
    out_dir = Path(f"{name}_output")
    prism_tex = out_dir / "main.tex"
    got_out   = out_dir / "got_main.tex"

    # GOT-OCR2
    print(f"[GOT] {img_path.name}", flush=True)
    t0 = time.perf_counter()
    raw = run_page_got(str(img_path))
    got_sec = time.perf_counter() - t0
    got_mb  = process.memory_info().rss / 1024 / 1024
    doc = wrap_got_output(raw)
    got_out.write_text(doc, encoding="utf-8")
    print(f"  GOT: {got_sec:.1f}s  {got_mb:.0f}MB  -> {got_out}")

    glare_results[name] = {
        "prism_tex": prism_tex,
        "got_tex":   got_out,
        "got_sec":   got_sec,
        "got_mb":    got_mb,
    }

unload_got()

# PRISM latency from orchestrate output (re-measure from file timestamps isn't reliable;
# use stored values from earlier run if available, else mark N/A)
PRISM_GLARE_LAT = {"glared": None, "glare2": None}

print("\n" + "="*72)
print("  Glare images  --  PRISM vs GOT-OCR2")
print("="*72)
print(f"  {'Image':<10}  {'PRISM output':>18}  {'GOT-OCR2 output':>18}  {'GOT lat':>8}  {'GOT RAM':>8}")
print(f"  {'-'*68}")
for name, res in glare_results.items():
    p_size = res["prism_tex"].stat().st_size if res["prism_tex"].exists() else 0
    g_size = res["got_tex"].stat().st_size   if res["got_tex"].exists()   else 0
    print(f"  {name:<10}  {p_size:>14} bytes  {g_size:>14} bytes  "
          f"{res['got_sec']:>7.1f}s  {res['got_mb']:>6.0f}MB")
print("="*72)
print("\n  Note: No GT available for glare images -- output size shown as proxy.")
print("  Review glared_output/got_main.tex and glare2_output/got_main.tex manually.")
