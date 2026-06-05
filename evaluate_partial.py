import os
import re
import csv
import statistics
from pathlib import Path
from sacrebleu.metrics import BLEU, CHRF
from rouge_score import rouge_scorer
from evaluation.normalizer import normalize_latex, split_math_and_text
from evaluation.eval import levenshtein_distance

# --- CONFIGURATION ---
DATASET_DIR   = Path("pdf2latex_dataset/dataset")
PRISM_TEX_DIR = Path("benchmark_results/prism_tex")
RESULTS_DIR   = Path("benchmark_results")

_bleu_scorer  = BLEU(effective_order=True)
_chrf_scorer  = CHRF(word_order=2)        # chrF++ (character + word n-grams)
_rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)


def compute_bleu(pred, gt):
    try:
        return _bleu_scorer.sentence_score(pred, [gt]).score
    except Exception:
        return 0.0


def compute_chrf(pred, gt):
    """chrF++ — more appropriate than BLEU for LaTeX: gives partial credit for
    near-correct formulas and is robust to word-fusion OCR errors."""
    try:
        return _chrf_scorer.sentence_score(pred, [gt]).score
    except Exception:
        return 0.0


def compute_rouge(pred, gt):
    try:
        return _rouge_scorer.score(gt, pred)['rougeL'].fmeasure * 100
    except Exception:
        return 0.0


def compute_cer_wer(pred, gt):
    dist = levenshtein_distance(pred, gt)
    cer  = (dist / len(gt)) * 100 if gt else 0
    p_words, g_words = pred.split(), gt.split()
    dist_w = levenshtein_distance(p_words, g_words)
    wer    = (dist_w / len(g_words)) * 100 if g_words else 0
    return cer, wer


def run_partial_eval():
    all_metrics = []
    prism_files = sorted(
        PRISM_TEX_DIR.glob("*_prism.tex"),
        key=lambda x: int(x.name.split("_")[0])
    )

    print(f"[*] Evaluating {len(prism_files)} processed pages...")

    for prism_path in prism_files:
        base_id = prism_path.name.split("_")[0]
        gt_path = DATASET_DIR / f"{base_id}_gt.tex"
        if not gt_path.exists():
            continue

        pred_latex = prism_path.read_text(encoding='utf-8')
        gt_latex   = gt_path.read_text(encoding='utf-8')

        # Normalise
        pred_norm = normalize_latex(pred_latex, remove_spaces=True)
        gt_norm   = normalize_latex(gt_latex,   remove_spaces=True)
        pred_word = normalize_latex(pred_latex, remove_spaces=False)
        gt_word   = normalize_latex(gt_latex,   remove_spaces=False)

        # Math / text split (on space-removed version for EDR)
        pred_math, pred_text = split_math_and_text(pred_norm)
        gt_math,   gt_text   = split_math_and_text(gt_norm)

        # EDR metrics
        ed_total  = levenshtein_distance(pred_norm,  gt_norm)
        edr_total = 1.0 - (ed_total / len(gt_norm))  if gt_norm  else 1.0

        ed_math   = levenshtein_distance(pred_math,  gt_math)
        edr_math  = 1.0 - (ed_math  / len(gt_math))  if gt_math  else 1.0

        ed_text   = levenshtein_distance(pred_text,  gt_text)
        edr_text  = 1.0 - (ed_text  / len(gt_text))  if gt_text  else 1.0

        cer, _    = compute_cer_wer(pred_norm,  gt_norm)
        _, wer    = compute_cer_wer(pred_word,  gt_word)

        bleu  = compute_bleu(pred_word,  gt_word)
        chrf  = compute_chrf(pred_word,  gt_word)
        rouge = compute_rouge(pred_word, gt_word)

        all_metrics.append({
            "id":          base_id,
            "overall_edr": edr_total,
            "math_edr":    edr_math,
            "text_edr":    edr_text,
            "bleu":        bleu,
            "chrf":        chrf,
            "rouge":       rouge,
            "cer":         cer,
            "wer":         wer,
            "gt_len":      len(gt_norm),
        })

    if not all_metrics:
        return

    def avg(key):
        return statistics.mean(m[key] for m in all_metrics)

    summary = {
        "overall_edr": avg("overall_edr"),
        "math_edr":    avg("math_edr"),
        "text_edr":    avg("text_edr"),
        "avg_bleu":    avg("bleu"),
        "avg_chrf":    avg("chrf"),
        "avg_rouge":   avg("rouge"),
        "avg_cer":     avg("cer"),
        "avg_wer":     avg("wer"),
    }

    print("\n--- Summary (First 25 Pages) ---")
    for k, v in summary.items():
        print(f"{k.upper():<15}: {v:.4f}")

    out_path = RESULTS_DIR / "partial_results.csv"
    with open(out_path, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f"\n[OK] Results written to {out_path}")


if __name__ == "__main__":
    run_partial_eval()
