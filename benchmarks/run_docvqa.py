"""
run_docvqa.py
-------------
Evaluate PRISM on DocVQA validation set using ANLS metric.

PRISM extracts full-page text; a sliding-window NED search finds the best
matching answer span. This tests whether DocVQA answers are present and
recoverable from PRISM's text output.

ANLS (Average Normalized Levenshtein Similarity) is the official metric:
  NLS(pred, gt) = 1 - NED(pred, gt)  if NED < 0.5, else 0
  ANLS = mean of max(NLS over all GT answers) per question.

Usage:
    python run_docvqa.py [--pred-dir docvqa_preds] [--eval-only] [--limit N]
"""
import argparse, glob, io, json, os, re, sys, unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
ROOT    = Path(__file__).parent.parent
VAL_DIR = ROOT / 'data' / 'docvqa' / 'data'


def _load_validation() -> pd.DataFrame:
    shards = sorted(VAL_DIR.glob('validation-*.parquet'))
    return pd.concat([pd.read_parquet(str(s)) for s in shards], ignore_index=True)




def run_prism(df: pd.DataFrame, pred_dir: Path) -> None:
    sys.path.insert(0, str(ROOT))
    from benchmarks.run_omnidocbench import _run_prism_on_images

    pred_dir.mkdir(parents=True, exist_ok=True)
    img_dir = pred_dir / '_imgs'
    img_dir.mkdir(exist_ok=True)

    # Deduplicate by docId — only run PRISM once per unique page
    seen_docs = {}
    for _, row in df.iterrows():
        doc_id = str(row['docId'])
        if doc_id not in seen_docs:
            img_path = img_dir / f'{doc_id}.png'
            if not img_path.exists():
                img = Image.open(io.BytesIO(row['image']['bytes'])).convert('RGB')
                img.save(str(img_path))
            seen_docs[doc_id] = str(img_path)

    # Skip docs already processed (resume support)
    image_paths = [p for doc_id, p in seen_docs.items()
                   if not (pred_dir / f'{doc_id}.md').exists()]
    skipped = len(seen_docs) - len(image_paths)
    if skipped:
        print(f'[*] Skipping {skipped} already-processed docs, {len(image_paths)} remaining...')
    else:
        print(f'[*] Running PRISM on {len(image_paths)} unique document images...')
    if image_paths:
        _run_prism_on_images(image_paths, str(pred_dir))


def _sliding_anls(text: str, gt_answers: list) -> float:
    """Find best approximate match for any GT answer in the extracted text."""
    import Levenshtein
    text_n = re.sub(r'\s+', ' ', text.lower()).strip()
    best = 0.0
    for ans in gt_answers:
        ans_n = re.sub(r'\s+', ' ', str(ans).lower()).strip()
        if not ans_n:
            continue
        w = len(ans_n)
        if w == 0:
            continue
        # Exact match fast path
        if ans_n in text_n:
            best = 1.0
            break
        # Slide window of same length over text
        min_ned = 1.0
        step = max(1, w // 4)
        for i in range(0, max(1, len(text_n) - w + 1), step):
            window = text_n[i:i + w]
            ned = Levenshtein.distance(window, ans_n) / w
            if ned < min_ned:
                min_ned = ned
        nls = 1.0 - min_ned if min_ned < 0.5 else 0.0
        if nls > best:
            best = nls
    return best


def evaluate(df: pd.DataFrame, pred_dir: Path) -> None:
    scores = []
    missing = 0

    for _, row in df.iterrows():
        doc_id = str(row['docId'])
        md_path = pred_dir / f'{doc_id}.md'
        if not md_path.exists():
            missing += 1
            continue
        pred_text = md_path.read_text(encoding='utf-8')
        gt_answers = list(row['answers'])
        score = _sliding_anls(pred_text, gt_answers)
        scores.append(score)

    anls = sum(scores) / len(scores) if scores else 0.0
    print('\n========== DocVQA Results (PRISM) ==========')
    print(f'Questions evaluated : {len(scores)}')
    print(f'Missing predictions : {missing}')
    print(f'ANLS                : {anls:.4f}  ({anls*100:.1f}%)')

    out = {'anls': anls, 'n': len(scores), 'missing': missing}
    out_path = pred_dir / 'docvqa_results.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Saved to {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred-dir', default=str(ROOT / 'preds' / 'docvqa'))
    ap.add_argument('--eval-only', action='store_true')
    ap.add_argument('--limit', type=int, default=0, help='Limit to N unique docs (0=all)')
    args = ap.parse_args()
    pred_dir = Path(args.pred_dir)

    print('[*] Loading DocVQA validation set...')
    df = _load_validation()
    print(f'[*] {len(df)} questions, {df["docId"].nunique()} unique documents')

    if args.limit:
        keep_docs = df['docId'].unique()[:args.limit]
        df = df[df['docId'].isin(keep_docs)]
        print(f'[*] Limited to {args.limit} docs → {len(df)} questions')

    if not args.eval_only:
        run_prism(df, pred_dir)

    evaluate(df, pred_dir)


if __name__ == '__main__':
    main()
