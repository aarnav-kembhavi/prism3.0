# -*- coding: utf-8 -*-
"""Phase 4 — Docling row for Table 3 (20-page EN-heavy subset, CPU, 8-thread).

Measurement mirrors the Table-3 harness (0h): affinity pin 0-7, 8-thread budget,
isolated, warmup page (idx 0) excluded, mean of two runs, peak process-tree RAM
sampled at 0.3s (reusing benchmarks.benchmark_glare.PeakRAMTracker, the project's
RAM primitive). Docling runs under DEFAULT configuration.

Conversion -> harness .md: text as markdown, tables as HTML <table>, formulas as
$$...$$ IF Docling emits LaTeX. Every decision logged to
results/rebuttal/docling_conversion_notes.md.

Run: venvs/docling_rebuttal/Scripts/python.exe scripts/rebuttal/phase4_docling.py
"""
import os, sys, io, json, time, threading
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
from pathlib import Path; ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, ROOT)
import psutil
try:
    import torch
    torch.set_num_threads(8)
except Exception:
    pass

GT = os.path.join(ROOT, 'benchmarks', 'compare', 'compare20_subset.json')
IMGDIR = os.path.join(ROOT, 'benchmarks', 'compare', 'mineru_in_full')
PRED = os.path.join(ROOT, 'preds', 'docling_table3')
PERF = os.path.join(PRED, 'perf.json')
NOTES = os.path.join(ROOT, 'results', 'rebuttal', 'docling_conversion_notes.md')
os.makedirs(PRED, exist_ok=True)

# affinity pin (8-thread budget), like run_omnidocbench.py
try:
    psutil.Process().cpu_affinity(list(range(8)))
    print('[*] affinity pinned to cores 0-7')
except Exception as e:
    print('[!] affinity pin failed:', e)

from benchmarks.benchmark_glare import PeakRAMTracker  # process-tree RSS primitive
from docling.document_converter import DocumentConverter

# stem list from GT (matches image files)
gt = json.load(open(GT, encoding='utf-8'))
stems = [os.path.splitext(os.path.basename(r['page_info']['image_path']))[0] for r in gt]
images = []
for s in stems:
    for ext in ('.jpg', '.png', '.jpeg'):
        p = os.path.join(IMGDIR, s + ext)
        if os.path.exists(p):
            images.append((s, p)); break
print(f'{len(images)}/{len(stems)} images found')

conv = DocumentConverter()
conv_stats = {'tables': 0, 'formula_items': 0, 'formula_with_latex': 0, 'pictures': 0,
              'text': 0, 'headers': 0, 'other': {}}

def table_html(item, doc):
    for call in (lambda: item.export_to_html(doc), lambda: item.export_to_html()):
        try:
            h = call()
            if h:
                return h
        except Exception:
            continue
    return None

def to_md(doc):
    from docling_core.types.doc import DocItemLabel
    parts = []
    for item, _lvl in doc.iterate_items():
        cls = type(item).__name__
        label = str(getattr(item, 'label', '')).lower()
        if cls == 'TableItem':
            h = table_html(item, doc)
            if h:
                parts.append(h); conv_stats['tables'] += 1
        elif 'formula' in label:
            conv_stats['formula_items'] += 1
            txt = (getattr(item, 'text', '') or '').strip()
            # heuristic: default Docling formula text is often empty / non-LaTeX
            if txt:
                conv_stats['formula_with_latex'] += 1
                parts.append(f'$$ {txt} $$')
        elif cls == 'PictureItem':
            conv_stats['pictures'] += 1
        elif cls == 'SectionHeaderItem' or 'header' in label:
            t = (getattr(item, 'text', '') or '').strip()
            if t:
                parts.append('## ' + t); conv_stats['headers'] += 1
        elif hasattr(item, 'text'):
            t = (item.text or '').strip()
            if t:
                parts.append(t); conv_stats['text'] += 1
        else:
            conv_stats['other'][cls] = conv_stats['other'].get(cls, 0) + 1
    return '\n\n'.join(parts)

def run_once(write_preds):
    tracker = PeakRAMTracker(interval=0.3)
    tracker.start()
    times = []
    for i, (stem, path) in enumerate(images):
        t = time.perf_counter()
        doc = conv.convert(path).document
        md = to_md(doc)
        dt = time.perf_counter() - t
        times.append(dt)
        if write_preds:
            with open(os.path.join(PRED, stem + '.md'), 'w', encoding='utf-8') as f:
                f.write(md)
        print(f'  page {i+1}/{len(images)} {dt:.1f}s  {stem[:40]}', flush=True)
    peak = tracker.stop()
    return times, peak

t0 = time.time()
print('=== RUN 1 (writes preds) ===')
times1, peak1 = run_once(write_preds=True)
print('=== RUN 2 ===')
times2, peak2 = run_once(write_preds=False)

# warmup (idx 0) excluded from latency; mean of two runs
def spg(times):
    body = times[1:]
    return sum(body) / max(len(body), 1)
lat = (spg(times1) + spg(times2)) / 2
ram_gb = (peak1 + peak2) / 2 / 1024.0
perf = {'system': 'Docling', 'n_pages': len(images), 'warmup_excluded': True,
        'runs': 2, 'latency_s_per_page': {'mean': lat, 'run1': spg(times1), 'run2': spg(times2)},
        'peak_ram_mb_process_tree': (peak1 + peak2) / 2, 'peak_ram_gb': ram_gb,
        'ram_sample_interval_s': 0.3, 'affinity': '0-7',
        'per_page_times_run1': times1, 'per_page_times_run2': times2}
json.dump(perf, open(PERF, 'w', encoding='utf-8'), indent=1)

# conversion notes
import docling
notes = f"""# Docling conversion notes (Phase 4)

- Docling version: {docling.__version__}; default DocumentConverter (layout +
  TableFormer + RapidOCR-torch, CPU). Formula enrichment is OFF by default.
- Input: {len(images)} images from compare20_subset (Table 3, 20-page EN-heavy).
- Measurement: affinity 0-7, warmup page excluded, mean of 2 runs, peak
  process-tree RAM @0.3s (benchmarks.benchmark_glare.PeakRAMTracker).

## Conversion decisions (item -> .md)
- TextItem / ListItem  -> markdown text
- SectionHeaderItem    -> '## ' heading
- TableItem            -> HTML <table> via TableItem.export_to_html (TEDS-scoreable)
- FORMULA-labeled item -> '$$ ... $$' ONLY if the item carries non-empty text
- PictureItem          -> dropped (no scoreable text)

## Emission audit (summed over run 1, {len(images)} pages)
- tables emitted (HTML):      {conv_stats['tables']}
- formula items detected:     {conv_stats['formula_items']}
- formula items WITH LaTeX:   {conv_stats['formula_with_latex']}
- text blocks:                {conv_stats['text']}
- section headers:            {conv_stats['headers']}
- pictures dropped:           {conv_stats['pictures']}
- other item classes:         {conv_stats['other']}

## Scoreability
- TextEN: scoreable (markdown text).
- TEDS: {'scoreable — Docling emitted HTML tables' if conv_stats['tables'] else 'NO tables emitted in this subset'}.
- FormulaEN: {'formula items carried LaTeX in %d cases' % conv_stats['formula_with_latex'] if conv_stats['formula_with_latex'] else 'NOT scoreable under default config — Docling detected %d formula regions but emitted NO LaTeX (formula enrichment is opt-in). Report as exclusion, do NOT zero.' % conv_stats['formula_items']}
"""
open(NOTES, 'w', encoding='utf-8').write(notes)
print(f'\nDONE {time.time()-t0:.0f}s')
print(f'latency {lat:.2f} s/pg, peak RAM {ram_gb:.2f} GB')
print(f'conv_stats: {conv_stats}')
print(f'preds -> {PRED}\nperf -> {PERF}\nnotes -> {NOTES}')
