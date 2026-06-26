"""
run_fox.py
----------
Run PRISM on the Fox benchmark (English + Chinese page OCR).
GT: fox_benchmark/focus_benchmark_test/{en,cn}_page_ocr.json
Images: fox_benchmark/focus_benchmark_test/{en,cn}_pdf_png/

Usage:
    python run_fox.py [--pred-dir fox_preds] [--eval-only]
"""
import argparse, json, os, re, sys, time
from pathlib import Path

os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
ROOT = Path(__file__).parent.parent
FOX  = ROOT / 'data' / 'fox' / 'focus_benchmark_test'


def _strip_markdown(text: str) -> str:
    """Strip markdown/LaTeX formatting to get plain text for comparison."""
    # Remove HTML tables — keep cell text
    text = re.sub(r'<table[^>]*>', '', text)
    text = re.sub(r'</table>', '', text)
    text = re.sub(r'<tr>', '', text)
    text = re.sub(r'</tr>', '\n', text)
    text = re.sub(r'<td>(.*?)</td>', r'\1 ', text, flags=re.DOTALL)
    # Remove display math delimiters
    text = re.sub(r'\\\[(.*?)\\\]', r'\1', text, flags=re.DOTALL)
    # Remove markdown heading markers
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Remove bold/italic
    text = re.sub(r'\*{1,2}(.*?)\*{1,2}', r'\1', text)
    # Collapse whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{2,}', '\n', text)
    return text.strip()


def _ned(pred: str, gt: str) -> float:
    """Normalised edit distance (0=identical, 1=totally different)."""
    import Levenshtein
    if not gt:
        return 0.0 if not pred else 1.0
    dist = Levenshtein.distance(pred, gt)
    return dist / max(len(pred), len(gt))


def _load_gt(json_path: Path) -> dict[str, str]:
    """Load Fox GT JSON → {image_filename: gt_text}."""
    with open(json_path, encoding='utf-8') as f:
        items = json.load(f)
    return {item['image']: item['conversations'][1]['value'].strip()
            for item in items}


def run_prism(image_paths: list[str], pred_dir: Path, cjk_stems: set) -> None:
    sys.path.insert(0, str(ROOT))
    from benchmarks.run_omnidocbench import _run_prism_on_images
    pred_dir.mkdir(parents=True, exist_ok=True)
    _run_prism_on_images(image_paths, str(pred_dir), cjk_pages=cjk_stems)


def evaluate(gt_en: dict, gt_cn: dict, pred_dir: Path) -> None:
    results = {'en': [], 'cn': []}

    for lang, gt in [('en', gt_en), ('cn', gt_cn)]:
        for img_name, gt_text in gt.items():
            stem = Path(img_name).stem
            md_path = pred_dir / f'{stem}.md'
            if not md_path.exists():
                print(f'  [!] missing pred: {stem}')
                continue
            pred_raw  = md_path.read_text(encoding='utf-8')
            pred_text = _strip_markdown(pred_raw)
            ned = _ned(pred_text, gt_text)
            results[lang].append((stem, ned))

    print('\n========== Fox Benchmark Results ==========')
    for lang in ('en', 'cn'):
        scores = [s for _, s in results[lang]]
        if not scores:
            print(f'{lang.upper()}: no results')
            continue
        avg = sum(scores) / len(scores)
        acc = 1 - avg
        print(f'{lang.upper()} ({len(scores)} pages): NED={avg:.4f}  Accuracy={acc*100:.1f}%')

    all_scores = [s for lang in ('en','cn') for _, s in results[lang]]
    if all_scores:
        avg = sum(all_scores) / len(all_scores)
        print(f'OVERALL ({len(all_scores)} pages): NED={avg:.4f}  Accuracy={(1-avg)*100:.1f}%')

    # Save JSON
    out = {lang: {s: float(v) for s,v in results[lang]} for lang in ('en','cn')}
    out_path = pred_dir / 'fox_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f'\nDetailed results saved to {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred-dir', default=str(ROOT / 'preds' / 'fox'))
    ap.add_argument('--eval-only', action='store_true')
    args = ap.parse_args()
    pred_dir = Path(args.pred_dir)

    gt_en = _load_gt(FOX / 'en_page_ocr.json')
    gt_cn = _load_gt(FOX / 'cn_page_ocr.json')
    print(f'[*] Fox GT loaded: {len(gt_en)} EN, {len(gt_cn)} CN pages')

    if not args.eval_only:
        en_dir = FOX / 'en_pdf_png'
        cn_dir = FOX / 'cn_pdf_png'
        image_paths, cjk_stems = [], set()
        for img_name in gt_en:
            p = en_dir / img_name
            if p.exists():
                image_paths.append(str(p))
        for img_name in gt_cn:
            p = cn_dir / img_name
            if p.exists():
                image_paths.append(str(p))
                cjk_stems.add(Path(img_name).stem)
        print(f'[*] Running PRISM on {len(image_paths)} images ({len(cjk_stems)} CJK)...')
        run_prism(image_paths, pred_dir, cjk_stems)

    evaluate(gt_en, gt_cn, pred_dir)


if __name__ == '__main__':
    main()
