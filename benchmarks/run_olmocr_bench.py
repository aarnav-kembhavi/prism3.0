"""
run_olmocr_bench.py
-------------------
Run PRISM on olmOCR-Bench (AllenAI unit-test benchmark) and lay predictions out
in the candidate format the official harness expects.

Pipeline:
  1. For the requested splits, collect the unique single-page PDFs they reference.
  2. Rasterize each PDF -> PNG (MiKTeX pdftoppm, 200 DPI) into a cache dir.
  3. Run PRISM (shared workers) over the PNGs -> {stem}.md in a staging dir.
  4. Copy each into  bench_data/prism/{stem}_pg1_repeat1.md  (harness naming:
     ^{md_base}_pg\\d+_repeat\\d+\\.md$ , recursive-globbed under the candidate dir).

Then evaluate with the harness venv:
  <olmobench-venv>/python -m olmocr.bench.benchmark --dir <bench_data> --candidate prism

Usage:
  python run_olmocr_bench.py --splits arxiv_math old_scans_math table_tests multi_column
  python run_olmocr_bench.py --all
  python run_olmocr_bench.py --splits table_tests --rasterize-only
"""
import argparse, json, os, subprocess, sys, time, shutil
from pathlib import Path

os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
ROOT   = Path(__file__).parent.parent
BENCH  = ROOT / 'data' / 'olmocr_bench' / 'bench_data'
PDFDIR = BENCH / 'pdfs'
PNGDIR = ROOT / 'data' / 'olmocr_bench' / '_png'
CAND   = BENCH / 'prism'                       # candidate name = "prism"

ALL_SPLITS = ['arxiv_math', 'old_scans_math', 'table_tests', 'multi_column',
              'headers_footers', 'long_tiny_text', 'old_scans']


def unique_pdfs(splits) -> dict:
    """{stem: pdf_rel_path} across the given splits (deduped). The rel path
    keeps the pdf's parent subfolder, which the harness mirrors when it looks
    up candidate predictions (e.g. table_tests.jsonl -> tables/xxx.pdf ->
    candidate at prism/tables/xxx_pg1_repeat1.md)."""
    out = {}
    for s in splits:
        jf = BENCH / f'{s}.jsonl'
        if not jf.exists():
            print(f'  [!] missing split file: {jf.name}')
            continue
        for line in jf.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rel = rec['pdf']                    # e.g. arxiv_math/xxx.pdf
            out[Path(rel).stem] = rel
    return out


def layout_candidates(pdfs: dict, stage: Path) -> int:
    """Copy staged {stem}.md into the harness candidate layout, mirroring each
    pdf's parent subfolder:  prism/<parent>/<stem>_pg1_repeat1.md ."""
    n = 0
    for stem, rel in pdfs.items():
        md = stage / f'{stem}.md'
        if not md.exists():
            continue
        sub = CAND / Path(rel).parent            # prism/<parent>
        sub.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(md, sub / f'{stem}_pg1_repeat1.md')
        n += 1
    return n


def rasterize(pdf_rel: str, dpi: int = 200) -> Path | None:
    """PDF -> single PNG via pdftoppm. Returns png path (cached)."""
    stem = Path(pdf_rel).stem
    png = PNGDIR / f'{stem}.png'
    if png.exists() and png.stat().st_size > 0:
        return png
    src = PDFDIR / pdf_rel
    if not src.exists():
        return None
    PNGDIR.mkdir(parents=True, exist_ok=True)
    prefix = PNGDIR / stem
    try:
        subprocess.run(['pdftoppm', '-png', '-r', str(dpi), '-singlefile',
                        str(src), str(prefix)],
                       check=True, capture_output=True)
    except Exception as e:
        print(f'  [!] rasterize failed {stem}: {e}')
        return None
    return png if png.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', nargs='+', default=None)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--dpi', type=int, default=200)
    ap.add_argument('--rasterize-only', action='store_true')
    ap.add_argument('--relayout-only', action='store_true',
                    help='re-copy staged md into candidate layout, no recompute')
    ap.add_argument('--limit', type=int, default=0, help='cap #PDFs (debug)')
    args = ap.parse_args()

    splits = ALL_SPLITS if args.all else (args.splits or
             ['arxiv_math', 'old_scans_math', 'table_tests', 'multi_column'])

    pdfs = unique_pdfs(splits)
    print(f'[*] {len(splits)} splits -> {len(pdfs)} unique PDFs')

    if args.relayout_only:
        stage = ROOT / 'preds' / 'olmocr_stage'
        shutil.rmtree(CAND, ignore_errors=True)
        n = layout_candidates(pdfs, stage)
        print(f'[*] re-laid {n} candidate files -> {CAND}')
        return

    # 1+2. rasterize
    t0 = time.time()
    img_paths, missing = [], []
    for i, (stem, rel) in enumerate(sorted(pdfs.items())):
        if args.limit and i >= args.limit:
            break
        png = rasterize(rel, args.dpi)
        if png:
            img_paths.append(str(png))
        else:
            missing.append(stem)
    print(f'[*] rasterized {len(img_paths)} PNGs in {time.time()-t0:.0f}s '
          f'({len(missing)} missing)')
    if missing:
        print('    missing:', ', '.join(missing[:10]), '...' if len(missing) > 10 else '')
    if args.rasterize_only:
        return

    # 3. run PRISM
    sys.path.insert(0, str(ROOT))
    from benchmarks.run_omnidocbench import _run_prism_on_images
    stage = ROOT / 'preds' / 'olmocr_stage'
    stage.mkdir(parents=True, exist_ok=True)
    print(f'[*] Running PRISM on {len(img_paths)} pages...')
    _run_prism_on_images(img_paths, str(stage))          # writes {stem}.md

    # 4. lay out in candidate format (mirrors each pdf's parent subfolder)
    shutil.rmtree(CAND, ignore_errors=True)
    n = layout_candidates(pdfs, stage)
    print(f'[*] wrote {n} candidate files -> {CAND}')
    print(f'[*] evaluate:  python -m olmocr.bench.benchmark --dir "{BENCH}" --candidate prism')


if __name__ == '__main__':
    main()
