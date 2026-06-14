"""
run_ocrbench.py
---------------
Evaluate PRISM on OCRBench (1000-question document/OCR benchmark).

For each image, PRISM extracts the full text. For each question we check
whether the ground-truth answer can be found (approximately) in that text,
using a sliding-window NED match. This adapts the extraction pipeline to
OCRBench's QA format without requiring a separate LLM for answer selection.

Usage:
    python run_ocrbench.py [--pred-dir ocrbench_preds] [--eval-only]
"""
import argparse, io, json, os, re, sys, unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from qa_extract import extract_answer, anls as compute_anls

os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
ROOT    = Path(__file__).parent
PARQUET = ROOT / 'data' / 'ocrbench' / 'data' / 'test-00000-of-00001.parquet'

# Question types to evaluate (skip scene-text-only tasks that PRISM isn't for)
EVAL_TYPES = {
    'Doc-oriented VQA',
    'Key Information Extraction',
    'Handwritten Mathematical Expression Recognition',
    'Handwriting Recognition',
}


pass  # helpers moved to qa_extract.py


def run_prism_ocrbench(df: pd.DataFrame, pred_dir: Path) -> None:
    """Run PRISM on each unique image in the benchmark."""
    sys.path.insert(0, str(ROOT))
    from run_omnidocbench import _run_prism_on_images

    pred_dir.mkdir(parents=True, exist_ok=True)
    seen, image_paths = set(), []
    tmp_imgs = pred_dir / '_imgs'
    tmp_imgs.mkdir(exist_ok=True)

    for idx, row in df.iterrows():
        img_key = str(idx)
        img_path = tmp_imgs / f'{img_key}.png'
        if img_key not in seen:
            img_bytes = row['image']['bytes']
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            img.save(str(img_path))
            image_paths.append(str(img_path))
            seen.add(img_key)

    print(f'[*] Running PRISM on {len(image_paths)} images...')
    _run_prism_on_images(image_paths, str(pred_dir))


def evaluate(df: pd.DataFrame, pred_dir: Path) -> None:
    results_by_type = {}

    for idx, row in df.iterrows():
        qtype = row['question_type']
        if qtype not in EVAL_TYPES:
            continue
        img_key = str(idx)
        md_path = pred_dir / f'{img_key}.md'
        if not md_path.exists():
            continue

        pred_text = md_path.read_text(encoding='utf-8')
        gt_answers = list(row['answer'])
        question = row['question']
        pred_answer = extract_answer(question, pred_text)
        score = compute_anls(pred_answer, gt_answers)

        if qtype not in results_by_type:
            results_by_type[qtype] = []
        results_by_type[qtype].append(score)

    print('\n========== OCRBench Results (PRISM) ==========')
    all_scores = []
    for qtype, scores in sorted(results_by_type.items()):
        avg = sum(scores) / len(scores) if scores else 0
        print(f'  {qtype:<45s} n={len(scores):3d}  score={avg*100:.1f}%')
        all_scores.extend(scores)

    if all_scores:
        overall = sum(all_scores) / len(all_scores)
        print(f'\n  {"OVERALL (eval types)":<45s} n={len(all_scores):3d}  score={overall*100:.1f}%')

    out = {qt: {'n': len(s), 'score': sum(s)/len(s) if s else 0}
           for qt, s in results_by_type.items()}
    out_path = pred_dir / 'ocrbench_results.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved to {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred-dir', default=str(ROOT / 'preds' / 'ocrbench'))
    ap.add_argument('--eval-only', action='store_true')
    args = ap.parse_args()
    pred_dir = Path(args.pred_dir)

    df = pd.read_parquet(str(PARQUET))
    print(f'[*] Loaded OCRBench: {len(df)} questions')

    eval_df = df[df['question_type'].isin(EVAL_TYPES)]
    print(f'[*] Evaluating {len(eval_df)} questions across {len(EVAL_TYPES)} task types')

    if not args.eval_only:
        run_prism_ocrbench(eval_df, pred_dir)

    evaluate(eval_df, pred_dir)


if __name__ == '__main__':
    main()
