# evaluation/eval.py
"""
Evaluation framework for Screen2LaTeX output quality.
Compares predicted .tex files against ground-truth using
Levenshtein edit distance with PDF2LaTeX normalization.
"""
import os
import json
from .normalizer import normalize_latex, split_math_and_text


def levenshtein_distance(s1, s2):
    """
    Compute Levenshtein edit distance between two strings.
    Uses dynamic programming — O(m*n) time, O(n) space.
    """
    m, n = len(s1), len(s2)
    # Use two rows to save memory
    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                curr[j] = prev[j-1]
            else:
                curr[j] = 1 + min(prev[j], curr[j-1], prev[j-1])
        prev, curr = curr, prev

    return prev[n]


def edit_distance_rate(predicted, ground_truth):
    """
    Compute edit distance rate = 1 - edit_dist / len(ground_truth).
    Higher is better (1.0 = perfect match).
    """
    ed = levenshtein_distance(predicted, ground_truth)
    if len(ground_truth) == 0:
        return 1.0
    return 1.0 - (ed / len(ground_truth))


def evaluate_page(predicted_latex, groundtruth_latex):
    """
    Evaluate one page: returns dict with overall, math, and text metrics.
    """
    pred_norm = normalize_latex(predicted_latex)
    gt_norm   = normalize_latex(groundtruth_latex)

    # Overall
    overall_ed   = levenshtein_distance(pred_norm, gt_norm)
    overall_rate = edit_distance_rate(pred_norm, gt_norm)

    # Split into math and text
    pred_math, pred_text = split_math_and_text(pred_norm)
    gt_math,   gt_text   = split_math_and_text(gt_norm)

    math_rate = edit_distance_rate(pred_math, gt_math)
    text_rate = edit_distance_rate(pred_text, gt_text)

    return {
        "edit_distance":       overall_ed,
        "edit_distance_rate":  round(overall_rate, 4),
        "math_edit_dist_rate": round(math_rate, 4),
        "text_edit_dist_rate": round(text_rate, 4),
        "gt_char_count":       len(gt_norm),
    }


def evaluate_dataset(predictions_dir, groundtruth_dir, output_json="results.json"):
    """
    Evaluate all pages in a dataset.
    Both dirs should contain .tex files with matching filenames.
    """
    results = []
    total_ed = 0
    total_chars = 0

    for fname in sorted(os.listdir(groundtruth_dir)):
        if not fname.endswith('.tex'):
            continue
        gt_path   = os.path.join(groundtruth_dir, fname)
        pred_path = os.path.join(predictions_dir, fname)

        if not os.path.exists(pred_path):
            print(f"Warning: No prediction found for {fname}")
            continue

        with open(gt_path,   'r', encoding='utf-8') as f: gt_latex   = f.read()
        with open(pred_path, 'r', encoding='utf-8') as f: pred_latex = f.read()

        page_result = evaluate_page(pred_latex, gt_latex)
        page_result['file'] = fname
        results.append(page_result)

        total_ed    += page_result['edit_distance']
        total_chars += page_result['gt_char_count']
        print(f"  {fname}: rate={page_result['edit_distance_rate']:.3f}, "
              f"math={page_result['math_edit_dist_rate']:.3f}, "
              f"text={page_result['text_edit_dist_rate']:.3f}")

    # Aggregate (same as PDF2LaTeX: sum all edit distances, divide by total chars)
    overall_rate = 1.0 - (total_ed / total_chars) if total_chars > 0 else 0.0

    summary = {
        "total_pages":         len(results),
        "total_edit_distance": total_ed,
        "total_gt_chars":      total_chars,
        "overall_edit_distance_rate": round(overall_rate, 4),
        "per_page": results
    }

    with open(output_json, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== SUMMARY ===")
    print(f"Pages evaluated:      {summary['total_pages']}")
    print(f"Total edit distance:  {summary['total_edit_distance']}")
    print(f"Overall EDR:          {summary['overall_edit_distance_rate']:.1%}")

    return summary
