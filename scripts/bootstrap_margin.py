# -*- coding: utf-8 -*-
"""TASK 3 — paired page-level bootstrap CI on the PRISM vs MinerU margin.

PRISM (odb_ablA_full = v21, 87.29) vs MinerU-pipeline CPU (mineru_cpu_pipeline,
86.08), OmniDocBench v1.6, official harness (commit 0b6e8b3), same 1651 pages.

Composite replicates the paper's `overall.py` EXACTLY (verified against the
reported aggregates before bootstrapping):
  Overall = ((1 - text) + CDM + TEDS)/3 * 100 (each term already 0..1), where
    text = text_block.all.Edit_dist.ALL_page_avg
         = mean over text pages of the per-page (upper-len-weighted) edit
           [dumped per page in *_text_block_per_page_edit.json]
    CDM  = display_formula.all.CDM.all
         = POOLED mean over all matched formula samples
           [*_display_formula_per_sample_CDM.json], NOT a page-mean
    TEDS = table.page.TEDS.ALL
         = mean over GT-table pages of the per-page mean table TEDS, where
           GT-table pages with no matched table contribute 0.0
           (harness _append_missing_page_rows), NOT a pooled table-mean

Paired page bootstrap: resample the 1651 page ids with replacement (SAME indices
for both systems), recompute each metric the harness way over the resampled
pages, 10,000 iterations. Report mean Overall diff, 95% percentile CI, and the
fraction of resamples PRISM leads; same for CDM.
"""
import os, io, sys, json
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
ROOT = str(Path(__file__).resolve().parents[1])
RES = os.path.join(ROOT, 'omnidocbench_eval', 'result')
GT = os.path.join(ROOT, 'data', 'omnidocbench_full', 'OmniDocBench.json')

SYSTEMS = {'PRISM': 'odb_ablA_full_quick_match',
           'MinerU': 'mineru_cpu_pipeline_quick_match'}
REPORTED = {'PRISM': 87.29, 'MinerU': 86.08}


def load_gt_pages():
    import ast
    gt = json.load(open(GT, encoding='utf-8'))
    all_pages, table_pages = [], set()
    for p in gt:
        name = p['page_info']['image_path']
        all_pages.append(name)
        cats = {e.get('category_type') for e in p.get('layout_dets', [])}
        if 'table' in cats:
            table_pages.add(name)
    return all_pages, table_pages


def page_of(sample_key):
    # 'page-xxx.png_[3]' -> 'page-xxx.png' ; 'name.jpg_[0]' -> 'name.jpg'
    return sample_key.rsplit('_[', 1)[0]


def load_system(tag, table_pages):
    j = lambda s: json.load(open(os.path.join(RES, f'{tag}_{s}.json'), encoding='utf-8'))
    txt = j('text_block_per_page_edit')                    # {page: edit}
    cdm_raw = j('display_formula_per_sample_CDM')          # {page_[i]: cdm}
    teds_raw = j('table_per_table_TEDS')                   # {page_[i]: {TEDS,...}}

    cdm_by_page = defaultdict(list)
    for k, v in cdm_raw.items():
        cdm_by_page[page_of(k)].append(float(v))

    teds_tables_by_page = defaultdict(list)
    for k, v in teds_raw.items():
        teds_tables_by_page[page_of(k)].append(float(v['TEDS']))
    # harness: every GT-table page with no matched table contributes a 0.0 row
    teds_page = {}
    for pg in table_pages:
        vals = teds_tables_by_page.get(pg, [0.0])
        teds_page[pg] = float(np.mean(vals))
    return {'txt': txt, 'cdm_by_page': dict(cdm_by_page), 'teds_page': teds_page}


def metrics_over_pages(sysd, pages):
    """Recompute (text, cdm, teds) the harness way over a page multiset."""
    tv = [sysd['txt'][p] for p in pages if p in sysd['txt']]
    text = float(np.mean(tv)) if tv else 0.0
    cs = []
    for p in pages:
        cs.extend(sysd['cdm_by_page'].get(p, ()))
    cdm = float(np.mean(cs)) if cs else 0.0
    tp = [sysd['teds_page'][p] for p in pages if p in sysd['teds_page']]
    teds = float(np.mean(tp)) if tp else 0.0
    overall = ((1 - text) + cdm + teds) / 3 * 100
    return overall, cdm * 100, text, teds * 100


def main():
    all_pages, table_pages = load_gt_pages()
    print(f'[gt] {len(all_pages)} pages, {len(table_pages)} with GT tables')

    data = {name: load_system(tag, table_pages) for name, tag in SYSTEMS.items()}

    # ---- page matching ----
    print('\n=== page matching ===')
    for name in SYSTEMS:
        for m in ('txt', 'cdm_by_page', 'teds_page'):
            pass
    # union of pages that appear in ANY per-page file, per system
    def sys_pages(sysd):
        return set(sysd['txt']) | set(sysd['cdm_by_page']) | set(sysd['teds_page'])
    pp, mp = sys_pages(data['PRISM']), sys_pages(data['MinerU'])
    gtset = set(all_pages)
    matched = gtset  # bootstrap universe = the 1651 GT pages
    print(f'  PRISM per-page pages: {len(pp)} | MinerU: {len(mp)} | GT: {len(gtset)}')
    print(f'  pages in PRISM not GT: {len(pp - gtset)} | in MinerU not GT: {len(mp - gtset)}')
    assert len(all_pages) == 1651, f'expected 1651 GT pages, got {len(all_pages)}'
    print(f'  bootstrap universe (GT pages) = {len(all_pages)}  [assert 1651 OK]')

    # ---- reproduction check on the full set ----
    print('\n=== reproduction check (full set must match reported) ===')
    ok = True
    for name in SYSTEMS:
        ov, cdm, text, teds = metrics_over_pages(data[name], all_pages)
        d = ov - REPORTED[name]
        print(f'  {name}: Overall={ov:.2f} (reported {REPORTED[name]}, Δ{d:+.3f}) '
              f'| text={text:.4f} CDM={cdm:.2f} TEDS={teds:.2f}')
        if abs(d) > 0.05:
            ok = False
    if not ok:
        print('  !! reproduction off by >0.05 — aggregation not exact, do NOT trust bootstrap')
        # continue anyway to show the discrepancy

    # ---- paired page bootstrap ----
    N = len(all_pages)
    pages_arr = np.array(all_pages, dtype=object)
    rng = np.random.default_rng(20260718)
    B = 10000
    d_overall = np.empty(B); d_cdm = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, N, N)
        pages = list(pages_arr[idx])
        ov_p, cdm_p, _, _ = metrics_over_pages(data['PRISM'], pages)
        ov_m, cdm_m, _, _ = metrics_over_pages(data['MinerU'], pages)
        d_overall[b] = ov_p - ov_m
        d_cdm[b] = cdm_p - cdm_m

    def report(name, d, point):
        lo, hi = np.percentile(d, [2.5, 97.5])
        frac = float(np.mean(d > 0)) * 100
        print(f'\n{name}: point {point:+.2f} | bootstrap mean {d.mean():+.3f} '
              f'| 95% CI [{lo:+.3f}, {hi:+.3f}] | PRISM ahead in {frac:.1f}% of resamples')
        return lo, hi, frac

    print('\n=== 10,000 paired bootstrap resamples ===')
    ov_pt = REPORTED['PRISM'] - REPORTED['MinerU']
    lo_o, hi_o, fr_o = report('Overall margin', d_overall, ov_pt)
    # CDM point margin from reported
    cdm_pt = metrics_over_pages(data['PRISM'], all_pages)[1] - metrics_over_pages(data['MinerU'], all_pages)[1]
    lo_c, hi_c, fr_c = report('CDM margin', d_cdm, cdm_pt)

    print('\n=== LaTeX-ready sentence ===')
    print(f'the margin is +{ov_pt:.2f} Overall (95\\% paired bootstrap CI '
          f'[{lo_o:.2f}, {hi_o:.2f}], PRISM ahead in {fr_o:.0f}\\% of resamples)')


if __name__ == '__main__':
    main()
