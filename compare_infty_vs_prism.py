"""
compare_infty_vs_prism.py
─────────────────────────
Compare PRISM vs InftyReader on the first 26 PDF2LaTeX pages.
No pipeline runs needed — predictions already exist on disk.
"""

import statistics
from pathlib import Path

from evaluation.normalizer import normalize_latex, split_math_and_text
from run_full_benchmark import compute_edr

GT_DIR      = Path("pdf2latex_dataset/dataset")
PRISM_DIR   = Path("benchmark_results/prism_tex_26")

all_imgs = sorted(
    [p for p in Path("benchmark_results/temp_images").glob("*.png")
     if (GT_DIR / f"{p.stem}_gt.tex").exists()],
    key=lambda p: int(p.stem),
)[:26]

rows = []
for img in all_imgs:
    pid = img.stem
    gt_path     = GT_DIR   / f"{pid}_gt.tex"
    infty_path  = GT_DIR   / f"{pid}_infty.tex"
    prism_path  = PRISM_DIR / f"{pid}_prism.tex"

    if not gt_path.exists() or not infty_path.exists() or not prism_path.exists():
        print(f"  [skip] page {pid} — missing file")
        continue

    gt    = normalize_latex(gt_path.read_text(   encoding="utf-8", errors="ignore"), remove_spaces=True)
    infty = normalize_latex(infty_path.read_text( encoding="utf-8", errors="ignore"), remove_spaces=True)
    prism = normalize_latex(prism_path.read_text( encoding="utf-8", errors="ignore"), remove_spaces=True)

    def scores(pred, ref):
        pm, pt = split_math_and_text(pred)
        rm, rt = split_math_and_text(ref)
        return compute_edr(pred, ref), compute_edr(pm, rm), compute_edr(pt, rt)

    i_ov, i_ma, i_tx = scores(infty, gt)
    p_ov, p_ma, p_tx = scores(prism, gt)
    rows.append(dict(pid=pid,
                     i_ov=i_ov, i_ma=i_ma, i_tx=i_tx,
                     p_ov=p_ov, p_ma=p_ma, p_tx=p_tx))

# ── Per-page table ────────────────────────────────────────────────────────────
print(f"\n{'='*74}")
print(f"  PRISM vs InftyReader  —  First 26 PDF2LaTeX pages")
print(f"{'='*74}")
print(f"  {'Page':>4}  |  {'-- InftyReader --':^22}  |  {'---- PRISM ----':^22}  |  Delta")
print(f"  {'':>4}  |  {'Overall':>7}  {'Text':>7}  {'Math':>7}  |  {'Overall':>7}  {'Text':>7}  {'Math':>7}  |  Overall")
print(f"  {'-'*70}")

for r in rows:
    delta = r['p_ov'] - r['i_ov']
    marker = f"{delta:+.1%}"
    print(f"  {r['pid']:>4}  |  {r['i_ov']:>7.1%}  {r['i_tx']:>7.1%}  {r['i_ma']:>7.1%}  |"
          f"  {r['p_ov']:>7.1%}  {r['p_tx']:>7.1%}  {r['p_ma']:>7.1%}  |  {marker}")

print(f"  {'-'*70}")

def avg(key): return statistics.mean(r[key] for r in rows)

print(f"  {'AVG':>4}  |  {avg('i_ov'):>7.1%}  {avg('i_tx'):>7.1%}  {avg('i_ma'):>7.1%}  |"
      f"  {avg('p_ov'):>7.1%}  {avg('p_tx'):>7.1%}  {avg('p_ma'):>7.1%}  |"
      f"  {avg('p_ov')-avg('i_ov'):>+.1%}")
print(f"{'='*74}")

wins_p  = sum(1 for r in rows if r['p_ov'] > r['i_ov'])
wins_i  = sum(1 for r in rows if r['i_ov'] > r['p_ov'])
ties    = len(rows) - wins_p - wins_i
print(f"\n  PRISM wins: {wins_p}/26   InftyReader wins: {wins_i}/26   Ties: {ties}/26")
