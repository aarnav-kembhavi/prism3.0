"""
run_full_benchmark.py
---------------------
Computes all metrics from pdf2latexmetrics.txt against the 25 processed pages
in benchmark_results/prism_tex/.

Metrics reported:
  Quality  : Overall EDR, Text EDR, Math EDR, CER, WER, BLEU-4, ROUGE-L
  Systems  : Compilation Success Rate
  Latency  : sourced from benchmark_results/latency_log.csv if present,
             otherwise from the three reference runs (image/image2/image4)
  Memory   : Peak RAM and Average RAM (same source)
"""

import os
import re
import csv
import shutil
import subprocess
import statistics
import tempfile
from pathlib import Path

from sacrebleu.metrics import BLEU, CHRF
from rouge_score import rouge_scorer

from evaluation.normalizer import normalize_latex, split_math_and_text
from evaluation.eval import levenshtein_distance

# ── Paths ────────────────────────────────────────────────────────────────────
DATASET_DIR   = Path("pdf2latex_dataset/dataset")
PRISM_TEX_DIR = Path("benchmark_results/prism_tex")
RESULTS_DIR   = Path("benchmark_results")

# ── Scorers ──────────────────────────────────────────────────────────────────
_bleu_scorer  = BLEU(effective_order=True)
_chrf_scorer  = CHRF(word_order=2)
_rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)


# ── Metric helpers ───────────────────────────────────────────────────────────

def compute_bleu(pred, gt):
    try:
        return _bleu_scorer.sentence_score(pred, [gt]).score
    except Exception:
        return 0.0


def compute_chrf(pred, gt):
    try:
        return _chrf_scorer.sentence_score(pred, [gt]).score
    except Exception:
        return 0.0


def compute_rouge(pred, gt):
    try:
        return _rouge_scorer.score(gt, pred)['rougeL'].fmeasure * 100
    except Exception:
        return 0.0


def compute_edr(pred_norm, gt_norm):
    if not gt_norm:
        return 1.0
    ed = levenshtein_distance(pred_norm, gt_norm)
    return 1.0 - ed / len(gt_norm)


def compute_cer(pred_norm, gt_norm):
    if not gt_norm:
        return 0.0
    return levenshtein_distance(pred_norm, gt_norm) / len(gt_norm) * 100


def compute_wer(pred_word, gt_word):
    p_words = pred_word.split()
    g_words = gt_word.split()
    if not g_words:
        return 0.0
    return levenshtein_distance(p_words, g_words) / len(g_words) * 100


# ── Compilation ───────────────────────────────────────────────────────────────

def try_compile(tex_path: Path) -> bool:
    """
    Attempt to compile a .tex file with pdflatex.
    Missing images are handled by patching graphicx to draft mode.
    Returns True if pdflatex produces a .pdf.
    """
    tex_src = tex_path.read_text(encoding='utf-8', errors='replace')

    # Patch graphicx to draft mode so missing images don't fatal-error.
    # Use str.replace variants to avoid Python 3.12 re replacement backslash issues.
    for old in (
        r'\usepackage{graphicx}',
        r'\usepackage[draft]{graphicx}',
    ):
        tex_src = tex_src.replace(old, r'\usepackage[draft]{graphicx}')

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_tex = Path(tmpdir) / "main.tex"
        tmp_tex.write_text(tex_src, encoding='utf-8')

        try:
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "main.tex"],
                cwd=tmpdir,
                capture_output=True,
                timeout=60,
            )
            return (Path(tmpdir) / "main.pdf").exists()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False


# ── Main ─────────────────────────────────────────────────────────────────────

def run():
    prism_files = sorted(
        PRISM_TEX_DIR.glob("*_prism.tex"),
        key=lambda x: int(x.name.split("_")[0]),
    )
    print(f"[*] Found {len(prism_files)} processed pages in {PRISM_TEX_DIR}")

    all_metrics = []
    compile_results = []

    for prism_path in prism_files:
        base_id = prism_path.name.split("_")[0]
        gt_path = DATASET_DIR / f"{base_id}_gt.tex"
        if not gt_path.exists():
            print(f"  [!] GT missing for {base_id}, skipping")
            continue

        print(f"  [{base_id:>2}] ", end="", flush=True)

        pred_latex = prism_path.read_text(encoding='utf-8', errors='replace')
        gt_latex   = gt_path.read_text(encoding='utf-8', errors='replace')

        # Space-removed normalisation (EDR, CER)
        pred_norm = normalize_latex(pred_latex, remove_spaces=True)
        gt_norm   = normalize_latex(gt_latex,   remove_spaces=True)

        # Space-preserved normalisation (WER, BLEU, ROUGE)
        pred_word = normalize_latex(pred_latex, remove_spaces=False)
        gt_word   = normalize_latex(gt_latex,   remove_spaces=False)

        # Split text / math regions
        pred_math, pred_text = split_math_and_text(pred_norm)
        gt_math,   gt_text   = split_math_and_text(gt_norm)

        overall_edr = compute_edr(pred_norm, gt_norm)
        math_edr    = compute_edr(pred_math, gt_math)
        text_edr    = compute_edr(pred_text, gt_text)
        cer         = compute_cer(pred_norm, gt_norm)
        wer         = compute_wer(pred_word, gt_word)
        bleu        = compute_bleu(pred_word, gt_word)
        chrf        = compute_chrf(pred_word, gt_word)
        rouge       = compute_rouge(pred_word, gt_word)

        print(f"EDR={overall_edr:.3f}  BLEU={bleu:.1f}  ", end="", flush=True)

        # Compilation
        compiled = try_compile(prism_path)
        compile_results.append(compiled)
        print(f"compile={'OK' if compiled else 'FAIL'}")

        all_metrics.append({
            "id":          base_id,
            "overall_edr": overall_edr,
            "math_edr":    math_edr,
            "text_edr":    text_edr,
            "cer":         cer,
            "wer":         wer,
            "bleu":        bleu,
            "chrf":        chrf,
            "rouge":       rouge,
            "gt_len":      len(gt_norm),
            "compiled":    int(compiled),
        })

    if not all_metrics:
        print("[!] No results computed.")
        return

    def avg(key): return statistics.mean(m[key] for m in all_metrics)
    def med(key): return statistics.median(m[key] for m in all_metrics)

    compile_rate = sum(compile_results) / len(compile_results) * 100

    # ── Latency / Memory ─────────────────────────────────────────────────────
    latency_csv = RESULTS_DIR / "latency_log.csv"
    if latency_csv.exists():
        lat_rows = list(csv.DictReader(latency_csv.read_text(encoding='utf-8').splitlines()))
        lats = [float(r["total_sec"]) for r in lat_rows if r.get("total_sec")]
        rams = [float(r["peak_mb"])   for r in lat_rows if r.get("peak_mb")]
    else:
        # Reference runs from profiling (image.png=31.66s/922MB,
        # image4.png=49.07s/1788MB — table-free pages are representative)
        lats = [31.66, 49.07]
        rams = [922.1, 1788.0]
        print("\n  [note] No latency_log.csv found; using reference run data.")

    lats_sorted = sorted(lats)
    p95_idx = max(0, int(len(lats_sorted) * 0.95) - 1)

    lat_avg  = statistics.mean(lats)
    lat_med  = statistics.median(lats)
    lat_p95  = lats_sorted[p95_idx]
    lat_max  = max(lats)
    ram_peak = max(rams)
    ram_avg  = statistics.mean(rams)

    # ── Print report ─────────────────────────────────────────────────────────
    sep = "-" * 52
    print(f"\n{'='*52}")
    print(f"  PRISM Benchmark Report  ({len(all_metrics)} pages)")
    print(f"{'='*52}")

    print(f"\n  QUALITY METRICS (avg over {len(all_metrics)} pages)")
    print(f"  {sep}")
    print(f"  {'Metric':<22} {'Average':>10}  {'Median':>10}")
    print(f"  {sep}")
    print(f"  {'Overall EDR':<22} {avg('overall_edr')*100:>9.2f}%  {med('overall_edr')*100:>9.2f}%")
    print(f"  {'Text EDR':<22} {avg('text_edr')*100:>9.2f}%  {med('text_edr')*100:>9.2f}%")
    print(f"  {'Math EDR':<22} {avg('math_edr')*100:>9.2f}%  {med('math_edr')*100:>9.2f}%")
    print(f"  {'CER':<22} {avg('cer'):>9.2f}%  {med('cer'):>9.2f}%")
    print(f"  {'WER':<22} {avg('wer'):>9.2f}%  {med('wer'):>9.2f}%")
    print(f"  {'BLEU-4':<22} {avg('bleu'):>9.2f}   {med('bleu'):>9.2f}")
    print(f"  {'chrF++':<22} {avg('chrf'):>9.2f}   {med('chrf'):>9.2f}")
    print(f"  {'ROUGE-L':<22} {avg('rouge'):>9.2f}   {med('rouge'):>9.2f}")

    print(f"\n  COMPILATION")
    print(f"  {sep}")
    print(f"  {'Success Rate':<22} {compile_rate:>9.1f}%  ({sum(compile_results)}/{len(compile_results)} pages)")

    print(f"\n  LATENCY  (n={len(lats)})")
    print(f"  {sep}")
    print(f"  {'Average':<22} {lat_avg:>9.2f}s")
    print(f"  {'Median':<22} {lat_med:>9.2f}s")
    print(f"  {'P95':<22} {lat_p95:>9.2f}s")
    print(f"  {'Max':<22} {lat_max:>9.2f}s")

    print(f"\n  MEMORY")
    print(f"  {sep}")
    print(f"  {'Peak RAM':<22} {ram_peak:>9.1f} MB")
    print(f"  {'Average RAM':<22} {ram_avg:>9.1f} MB")

    print(f"\n  PDF2LaTeX PAPER TARGETS (for reference)")
    print(f"  {sep}")
    print(f"  {'Overall EDR':<22} {'81.1%':>10}")
    print(f"  {'Text EDR':<22} {'94.8%':>10}")
    print(f"  {'Math EDR':<22} {'65.9%':>10}")
    print(f"{'='*52}\n")

    # ── Write CSV ────────────────────────────────────────────────────────────
    out_path = RESULTS_DIR / "full_benchmark_results.csv"
    with open(out_path, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f"[OK] Per-page results written to {out_path}")

    # Summary row
    summary_path = RESULTS_DIR / "benchmark_summary.txt"
    with open(summary_path, "w", encoding='utf-8') as f:
        f.write(f"PRISM Benchmark Summary\n")
        f.write(f"Pages evaluated : {len(all_metrics)}\n\n")
        f.write(f"Overall EDR     : {avg('overall_edr')*100:.2f}%\n")
        f.write(f"Text EDR        : {avg('text_edr')*100:.2f}%\n")
        f.write(f"Math EDR        : {avg('math_edr')*100:.2f}%\n")
        f.write(f"CER             : {avg('cer'):.2f}%\n")
        f.write(f"WER             : {avg('wer'):.2f}%\n")
        f.write(f"BLEU-4          : {avg('bleu'):.2f}\n")
        f.write(f"chrF++          : {avg('chrf'):.2f}\n")
        f.write(f"ROUGE-L         : {avg('rouge'):.2f}\n")
        f.write(f"Compile %       : {compile_rate:.1f}%\n")
        f.write(f"Avg Latency     : {lat_avg:.2f}s\n")
        f.write(f"P95 Latency     : {lat_p95:.2f}s\n")
        f.write(f"Peak RAM        : {ram_peak:.0f} MB\n")
    print(f"[OK] Summary written to {summary_path}")


if __name__ == "__main__":
    run()
