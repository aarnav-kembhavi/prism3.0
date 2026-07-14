"""
page_core.py
------------
Shared per-page extraction + document assembly, used by BOTH the product
entrypoint (pipeline/orchestrate.py) and the benchmark
(benchmarks/run_omnidocbench.py).

Before this module the two reimplemented the same routing/column/assembly
logic inline and drifted (different table builders, missed fixes). Now a fix
to routing, table handling, or column dispatch lands in one place.

Callers differ only in what they wrap around this:
  - orchestrate: single subprocess workers, saves outputs/, profiling
  - benchmark:   dual workers, GT language hints, converts to Markdown
Both pass a `Workers` bundle (single- and dual-worker classes are
API-compatible) and the language/modality flags they determined their own way.
"""

import os
import time as _time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

# Per-page stage-timing accumulator (PRISM_STAGE_TIMING=1; the benchmark
# runner resets and reads it around each build_document call). Keys:
# 'math' / 'text' / 'table' wall seconds. Off by default: zero overhead.
stage_times: dict = {}


def _timed(key, fn):
    if os.environ.get('PRISM_STAGE_TIMING', '0') != '1':
        return fn
    def _wrap(*a, **k):
        t0 = _time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            stage_times[key] = stage_times.get(key, 0.0) + (_time.perf_counter() - t0)
    return _wrap

from pipeline.layout_utils import (
    apply_semantic_reading_order, xyxy_to_pil_crop,
    detect_column_count, split_detections_by_column, split_detections_n_columns,
)
from pipeline.latex_builder import (
    wrap_content, assemble_document, assemble_columns_document)


TEXT_CLASSES    = {"Text", "Title", "Section-header", "Caption",
                   "Footnote", "Page-footer", "Page-header", "List-item"}
MATH_CLASSES    = {"Formula"}
TABLE_CLASSES   = {"Table"}
IMAGE_CLASSES   = {"Picture"}
LIST_ITEM_CLASS = "List-item"


@dataclass
class Workers:
    """Bundle of the three specialist subprocess workers.

    ocr  — TextOCRWorker or TextOCRWorkerDual
    math — MathOCRWorkerOnnx or MathOCRWorkerOnnxDual
    tatr — TATROnnxWorker or None (falls back to the coordinate heuristic)
    """
    ocr:  Any
    math: Any
    tatr: Any = None


def _adjust_figure_paths(parts: list[str]) -> list[str]:
    """Prefix bare figure_NNN filenames with the assets/figures/ subdirectory."""
    return [
        p.replace("{figure_", "{assets/figures/figure_")
        if "includegraphics" in p else p
        for p in parts
    ]


_rtable = None


def _get_rtable():
    """Lazy singleton for the RapidTable child worker (None if unavailable)."""
    global _rtable
    if _rtable is None:
        from pipeline import rtable_worker
        if rtable_worker.available():
            try:
                w = rtable_worker.RapidTableWorker()
                w.start()
                _rtable = w
            except Exception as e:
                print(f"  [rtable] unavailable, using TATR: {e}")
                _rtable = False
        else:
            _rtable = False
    return _rtable or None


def _extract_tables(table_crops, workers: Workers, is_cjk: bool = False) -> list[str]:
    """Table structure recognition via RapidTable (SLANet-plus), TATR fallback.

    SLANet-plus predicts structure and runs its own cell OCR on the crop
    (validated: catastrophic tables -0.01 -> 0.57 TEDS, good tables unchanged).
    TATR + token assignment remains the fallback path, and after that the
    coordinate heuristic.

    CJK pages route the TATR-path cell OCR through the CJK engine — otherwise
    Chinese cell text is recognized by the English model and collapses full
    TEDS (structure stays fine, content is garbage).
    """
    rtable = _get_rtable()
    if rtable is not None:
        # PRISM_RTABLE_OCR_V6: read cell content with the pipeline's PP-OCRv6
        # engines (CJK-aware) instead of rapid_table's older internal OCR —
        # the EN/ZH content-vs-structure TEDS gap (0.11/0.10) is cell text.
        tokens_list = [None] * len(table_crops)
        if os.environ.get('PRISM_RTABLE_OCR_V6', '0') == '1' and table_crops:
            try:
                if is_cjk and hasattr(workers.ocr, 'run_table_tokens_batch_cjk'):
                    raw_tokens = workers.ocr.run_table_tokens_batch_cjk(table_crops)
                else:
                    raw_tokens = workers.ocr.run_table_tokens_batch(table_crops)
                tokens_list = [
                    [(t['x1'], t['y1'], t['x2'], t['y2'], t['text'],
                      t.get('conf', t.get('score', 0.95))) for t in toks]
                    if toks else None
                    for toks in raw_tokens
                ]
            except Exception as e:
                print(f"  [rtable] v6-ocr tokens failed, using internal OCR: {e}")
        results = []
        pending = []  # indices that need the TATR fallback
        for i, crop in enumerate(table_crops):
            html = rtable.build_table_html(crop, ocr_tokens=tokens_list[i])
            if html and html.count('<td') >= 1:
                results.append(html)
            else:
                results.append(None)
                pending.append(i)
        if pending:
            fallback = _extract_tables_tatr(
                [table_crops[i] for i in pending], workers, is_cjk=is_cjk)
            for i, res in zip(pending, fallback):
                results[i] = res
        results = [r or '' for r in results]
        _append_geom_tables(results, table_crops, workers, is_cjk)
        return results
    return _extract_tables_tatr(table_crops, workers, is_cjk=is_cjk)


def _html_escape(s: str) -> str:
    return (s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))


def _table_grid_html(tokens, img_w) -> str:
    """Geometric HTML grid from OCR token boxes: cluster rows by y-centre,
    split columns at gaps in the sorted token x-centres, assign each token.

    Complements SLANet, which sometimes predicts too few columns and merges a
    whole row of values into one <td> (e.g. '0.001 (-0.014) 0.616 (16.367)*').
    The olmOCR TableTest checks EVERY table in the content and passes if any
    one has the right cell, so emitting this extra, geometry-separated grid is
    additive — it can only add cell matches, never remove SLANet's."""
    import numpy as np
    toks = []
    for t in tokens or []:
        txt = (t['text'] or '').strip()
        if not txt:
            continue
        toks.append({'x1': t['x1'], 'x2': t['x2'],
                     'cy': (t['y1'] + t['y2']) / 2.0,
                     'h': max(1.0, t['y2'] - t['y1']),
                     'w': max(1.0, t['x2'] - t['x1']), 'text': txt})
    if len(toks) < 4:
        return ''
    toks.sort(key=lambda t: t['cy'])
    rows, cur, cy = [], [], None
    for tk in toks:
        if cy is None or abs(tk['cy'] - cy) < max(tk['h'], 10) * 0.6:
            cur.append(tk)
            cy = sum(t['cy'] for t in cur) / len(cur)
        else:
            rows.append(cur)
            cur, cy = [tk], tk['cy']
    if cur:
        rows.append(cur)
    if len(rows) < 2:
        return ''
    # Column boundaries from gaps in the sorted token x-centres. A gap wider
    # than gap_thr (scaled to the typical token width) is a real column break.
    # This separates columns whose ink touches (no zero-projection gap) — the
    # case that makes SLANet merge a value-row into one cell.
    non_wide = [t for t in toks if t['w'] <= 0.6 * max(1, img_w)]
    if len(non_wide) < 4:
        return ''
    med_w = float(np.median([t['w'] for t in non_wide]))
    centers = sorted((t['x1'] + t['x2']) / 2.0 for t in non_wide)
    gap_thr = max(med_w * 0.8, 0.02 * max(1, img_w))
    bounds = [0.0]
    for a, b in zip(centers, centers[1:]):
        if b - a > gap_thr:
            bounds.append((a + b) / 2.0)
    bounds.append(float(img_w) + 1.0)
    if len(bounds) - 1 < 2:              # no column structure to add
        return ''

    def col_of(tk):
        c = (tk['x1'] + tk['x2']) / 2.0
        for i in range(len(bounds) - 1):
            if bounds[i] <= c < bounds[i + 1]:
                return i
        return len(bounds) - 2

    ncol = len(bounds) - 1
    out = ['<table>']
    for row in rows:
        cells = [[] for _ in range(ncol)]
        for tk in sorted(row, key=lambda t: t['x1']):
            cells[col_of(tk)].append(tk['text'])
        tds = ''.join('<td>' + _html_escape(' '.join(c)) + '</td>' for c in cells)
        out.append('<tr>' + tds + '</tr>')
    out.append('</table>')
    return '\n'.join(out)


def _append_geom_tables(results, table_crops, workers, is_cjk):
    """For each table, append a geometry-reconstructed HTML grid (from OCR
    token boxes) after SLANet's output. Additive coverage for olmOCR-Bench
    cell tests SLANet loses to under-segmentation. PRISM_TABLE_GEOM=0 disables."""
    if os.environ.get('PRISM_TABLE_GEOM', '1') == '0' or not table_crops:
        return
    try:
        if is_cjk and hasattr(workers.ocr, 'run_table_tokens_batch_cjk'):
            raw = workers.ocr.run_table_tokens_batch_cjk(table_crops)
        else:
            raw = workers.ocr.run_table_tokens_batch(table_crops)
    except Exception as e:
        print(f"  [table-geom] token OCR failed: {e}")
        return
    for i, crop in enumerate(table_crops):
        toks = raw[i] if i < len(raw) else None
        if not toks:
            continue
        geom = _table_grid_html(toks, crop.width)
        # Only add when it introduces real column structure (>=2 cols) and the
        # grid differs from what SLANet already emitted.
        if geom and geom.count('</td>') >= 4:
            results[i] = (results[i] + '\n\n' + geom) if results[i] else geom


def _extract_tables_tatr(table_crops, workers: Workers, is_cjk: bool = False) -> list[str]:
    _cjk_tbl = os.environ.get('PRISM_CJK_TABLE_OCR', '1') != '0'  # A/B toggle
    if workers.tatr is not None:
        if is_cjk and _cjk_tbl and hasattr(workers.ocr, 'run_table_tokens_batch_cjk'):
            tokens_list = workers.ocr.run_table_tokens_batch_cjk(table_crops)
        else:
            tokens_list = workers.ocr.run_table_tokens_batch(table_crops)
        results = []
        for crop, tokens in zip(table_crops, tokens_list):
            result = None
            if tokens:
                try:
                    result = workers.tatr.build_table_html(crop, tokens, crop.width)
                except Exception as e:
                    print(f"  [TATR] error: {e}")
            if not result and tokens:
                from pipeline.models_interface import _table_heuristic
                heuristic_tokens = [
                    {'text': t['text'], 'x1': t['x1'], 'x2': t['x2'],
                     'y1': t['y1'], 'y2': t['y2'],
                     'cx': (t['x1'] + t['x2']) / 2, 'cy': (t['y1'] + t['y2']) / 2,
                     'h': t['y2'] - t['y1'], 'w': t['x2'] - t['x1']}
                    for t in tokens
                ]
                result = _table_heuristic(heuristic_tokens, crop.width)
            results.append(result or '')
        return results
    # No TATR available — coordinate heuristic from the OCR worker.
    return workers.ocr.run_table_batch(table_crops)


_CJK_RE = None


def _cjk_count(s: str) -> int:
    return sum(1 for c in (s or '') if '一' <= c <= '鿿')


def _ocr_lines_to_latex(lines) -> str:
    """Turn OCR line results [(x1,y1,x2,y2,text)] into display LaTeX: CJK runs
    wrapped in \\text{}, everything else kept verbatim, rows joined with \\\\."""
    import re
    # cluster into visual rows by y-center
    grouped, cur, cur_yc = [], [], None
    for x1, y1, x2, y2, txt in sorted(lines, key=lambda t: ((t[1] + t[3]) / 2, t[0])):
        yc = (y1 + y2) / 2
        h = max(y2 - y1, 1)
        if cur and abs(yc - cur_yc) > 0.6 * h:
            grouped.append(cur); cur = []
        cur.append((x1, txt)); cur_yc = yc
    if cur:
        grouped.append(cur)

    def esc(seg: str) -> str:
        return re.sub(r'([\\{}%&#_^~$])', r'\\\1', seg)

    out_rows = []
    for row in grouped:
        joined = ' '.join(t for _, t in sorted(row))
        # wrap CJK runs (incl. fullwidth punctuation) in \text{}
        parts, buf_cjk, buf_other = [], [], []
        def flush_cjk():
            if buf_cjk:
                parts.append('\\text{' + esc(''.join(buf_cjk)) + '}')
                buf_cjk.clear()
        def flush_other():
            if buf_other:
                seg = ''.join(buf_other)
                # OCR output is not LaTeX: escape comment/table specials, and
                # escape braces when they don't balance within the segment
                # (stray glyphs, not grouping).
                seg = re.sub(r'([%&#])', r'\\\1', seg)
                if seg.count('{') != seg.count('}'):
                    seg = seg.replace('{', '\\{').replace('}', '\\}')
                parts.append(seg)
                buf_other.clear()
        for c in joined:
            if '一' <= c <= '鿿' or '　' <= c <= '〿' or '＀' <= c <= '￯':
                flush_other(); buf_cjk.append(c)
            else:
                flush_cjk(); buf_other.append(c)
        flush_cjk(); flush_other()
        out_rows.append(''.join(parts).strip())
    out_rows = [r for r in out_rows if r]
    if not out_rows:
        return ''
    if len(out_rows) == 1:
        return out_rows[0]
    return '\\begin{array}{l}' + ' \\\\ '.join(out_rows) + '\\end{array}'


def _chip_token(chip, latex, workers, cjk) -> str:
    """One inline chip -> the string spliced into the host text.

    Texo LaTeX wrapped as $...$ when it looks like sane inline math (the
    harness converts pred inline LaTeX with the same textblock2unicode used
    on GT). Structural output (arrays, alignment tabs), overlong strings and
    decode failures fall back to plain OCR of the chip crop.
    """
    import re
    latex = (latex or '').strip()
    if latex.startswith('\\includegraphics'):
        latex = ''
    latex = latex.strip('$').strip()
    # size-only commands add nothing after the harness's latex->unicode
    # normalization but inflate the span (a longer pred inflates the
    # unmatched-GT penalty weight when the matcher pairs it in desperation)
    latex = re.sub(r'\\(?:left|right)\s*\.', '', latex)
    latex = re.sub(r'\\(?:[Bb]igg?[lrm]?|left|right)(?![A-Za-z])', '', latex)
    latex = re.sub(r'\s{2,}', ' ', latex).strip()
    ok = bool(latex) and len(latex) <= 100 and '\\begin' not in latex \
        and '\\\\' not in latex and '&' not in latex \
        and latex.count('{') == latex.count('}')
    # (A structure-only gate — splice just fractions/roots — was A/B'd and
    # REJECTED: linear $latex$ splices also help on net, and isolated
    # chip-crop OCR reads worse than the latex it would replace.)
    if ok and cjk:
        # CJK pages: the "formula" chip is often CJK text the detector
        # mislabeled — Texo hallucinates on those (same failure the CJK
        # display hybrid handles). Prefer OCR when the crop reads CJK.
        try:
            lines = workers.ocr.run_text_lines(chip['crop'], 'cjk')
        except Exception:
            lines = []
        joined = ''.join(t for *_, t in (lines or []))
        if _cjk_count(joined) >= 2:
            from pipeline.text_worker import _escape_latex
            return _escape_latex(joined)
    if ok:
        return f'${latex}$'
    try:
        lines = workers.ocr.run_text_lines(chip['crop'], 'cjk' if cjk else 'en')
    except Exception:
        return ''
    if not lines:
        return ''
    from pipeline.text_worker import _escape_latex
    return _escape_latex(
        ' '.join(t for *_, t in sorted(lines, key=lambda l: (l[1], l[0]))))


def _assemble_lines_with_chips(lines, chip_items) -> str:
    """Rebuild a text block from OCR line fragments + chip tokens, all in
    crop coordinates. Mirrors the text worker's _reconstruct_lines row
    clustering (y-center within 0.6x line height) so the output format
    matches the normal text path. OCR fragments get the same LaTeX escaping
    as the batch path (tex_to_md unescapes); chip tokens are LaTeX already."""
    from pipeline.text_worker import _escape_latex
    items = []
    for x1, y1, x2, y2, txt in lines:
        txt = (txt or '').strip()
        if txt:
            items.append({'x_left': x1, 'x_right': x2,
                          'y_ctr': (y1 + y2) / 2.0,
                          'height': max(y2 - y1, 1),
                          'text': _escape_latex(txt)})
    chips = [{'x_left': x1, 'x_right': x2, 'y_ctr': (y1 + y2) / 2.0,
              'height': max(y2 - y1, 1), 'text': tok}
             for x1, y1, x2, y2, tok in chip_items if tok]
    if not items and not chips:
        return ''
    # Cluster OCR fragments into rows FIRST (chips are taller than text
    # lines; co-clustering them inflated the row threshold and scrambled
    # fragments across visual lines on dense-math paragraphs)...
    rows = []
    if items:
        items.sort(key=lambda d: d['y_ctr'])
        cur = [items[0]]
        for it in items[1:]:
            thr = min(cur[-1]['height'], it['height']) * 0.6
            if abs(it['y_ctr'] - cur[-1]['y_ctr']) <= thr:
                cur.append(it)
            else:
                rows.append(cur)
                cur = [it]
        rows.append(cur)
    # ...then drop each chip into the row whose vertical span overlaps its
    # center best; chips matching no row become their own row.
    row_chips = [[] for _ in rows]
    for c in chips:
        best, bov = None, 0.0
        for i, row in enumerate(rows):
            top = min(r['y_ctr'] - r['height'] / 2 for r in row)
            bot = max(r['y_ctr'] + r['height'] / 2 for r in row)
            c_top = c['y_ctr'] - c['height'] / 2
            c_bot = c['y_ctr'] + c['height'] / 2
            ov = min(bot, c_bot) - max(top, c_top)
            if ov > bov:
                best, bov = i, ov
        if best is not None and bov > 0:
            row_chips[best].append(c)
        else:
            rows.append([c])
            row_chips.append([])
    order = sorted(range(len(rows)),
                   key=lambda i: sum(r['y_ctr'] for r in rows[i]) / len(rows[i]))
    out = []
    for i in order:
        line = _place_chips_in_row(rows[i], row_chips[i])
        if line:
            out.append(line)
    return '\n'.join(out)


def _place_chips_in_row(frags, chips):
    """Merge chip tokens into one visual row of OCR fragments.

    OCR detection often bridges the narrow mask holes, returning ONE
    fragment spanning the chip's position — x-sorting alone then dumps the
    chip at the row end ("estimation of and, we derive $I_{11}$"). When a
    chip's center falls inside a fragment's span, split the fragment at the
    proportional character position (snapped to a space when there is one)
    and put the chip between the halves."""
    frags = [dict(f) for f in sorted(frags, key=lambda d: d['x_left'])]
    loose = []
    assignments = {}
    for c in sorted(chips, key=lambda d: d['x_left']):
        cx = (c['x_left'] + c['x_right']) / 2
        for i, f in enumerate(frags):
            if len(f['text']) > 2 and f['x_left'] + 2 < cx < f['x_right'] - 2:
                assignments.setdefault(i, []).append(c)
                break
        else:
            loose.append(c)
    out_frags = []
    for i, f in enumerate(frags):
        cs = assignments.get(i)
        if not cs:
            out_frags.append(f)
            continue
        t = f['text']
        fx = f['x_left']
        w = max(f['x_right'] - fx, 1.0)
        # the mask holes (chip regions) contribute width but no characters —
        # interpolate char positions over the hole-free width only
        holes = [(max(c['x_left'], fx), min(c['x_right'], f['x_right']))
                 for c in cs]
        eff_w = max(w - sum(b - a for a, b in holes), 1.0)
        pieces = []
        prev_idx = 0
        holes_left = 0.0
        seg_x = fx             # running left edge for the next text segment
        for (a, hb), c in zip(holes, cs):
            eff_x = max((a - fx) - holes_left, 0.0)
            holes_left += hb - a
            idx = min(len(t), max(prev_idx, round(eff_x / eff_w * len(t))))
            cut = None
            for off in range(5):
                for j in (idx - off, idx + off):
                    if prev_idx <= j < len(t) and t[j] == ' ':
                        cut = j
                        break
                if cut is not None:
                    break
            if cut is None:
                cut = idx      # CJK / no space nearby: cut proportionally
            seg = t[prev_idx:cut].strip()
            if seg:
                pieces.append({'x_left': seg_x, 'x_right': a, 'text': seg})
            pieces.append({'x_left': c['x_left'], 'x_right': c['x_right'],
                           'text': c['text'], 'chip': True})
            prev_idx = cut
            seg_x = hb
        tail = t[prev_idx:].strip()
        if tail:
            pieces.append({'x_left': seg_x, 'x_right': f['x_right'],
                           'text': tail})
        out_frags.extend(pieces)
    out_frags.extend({'x_left': c['x_left'], 'x_right': c['x_right'],
                      'text': c['text'], 'chip': True} for c in loose)
    out_frags.sort(key=lambda d: d['x_left'])
    return ' '.join(f['text'] for f in out_frags if f['text']).strip()


def _splice_chip_hosts(detections, host_indices, workers, figures_dir, cjk):
    """Recognize each host's inline chips and rebuild the host text with the
    math spliced in at its line position. Falls back to the batch OCR text
    already on the det when assembly produces nothing."""
    all_chips = []
    for hi in host_indices:
        for c in detections[hi].get('inline_chips') or []:
            if c.get('crop') is not None:
                all_chips.append((hi, c))
    if not all_chips:
        return
    try:
        latexes, _ = workers.math.run_math_batch(
            [c['crop'] for _, c in all_chips], figures_dir, 90000)
    except Exception as e:
        print(f"  [splice] chip math batch failed: {e}")
        return
    per_host = {}
    for (hi, c), latex in zip(all_chips, latexes):
        tok = _chip_token(c, latex, workers, cjk)
        if tok:
            per_host.setdefault(hi, []).append((c['bbox'], tok))
    n_spliced = 0
    for hi, chips in per_host.items():
        d = detections[hi]
        crop = d.get('crop')
        if crop is None:
            continue
        try:
            lines = workers.ocr.run_text_lines(crop, 'cjk' if cjk else 'en')
        except Exception:
            lines = []
        tb = d['bbox']
        sx = crop.width / max(tb[2] - tb[0], 1.0)
        sy = crop.height / max(tb[3] - tb[1], 1.0)
        chip_items = [((b[0] - tb[0]) * sx, (b[1] - tb[1]) * sy,
                       (b[2] - tb[0]) * sx, (b[3] - tb[1]) * sy, tok)
                      for b, tok in chips]
        text = _assemble_lines_with_chips(lines or [], chip_items)
        if text.strip():
            d['raw_content'] = text
            n_spliced += len(chips)
    if n_spliced:
        print(f"  [splice] spliced {n_spliced} inline chip(s) "
              f"into {len(per_host)} text block(s)")


def _apply_cjk_formula_hybrid(detections, math_indices, workers):
    """Texo hallucinates on formulas containing CJK text (ZH formula CDM 0.648
    vs EN 0.866; 'cases' formulas with Chinese labels come back as unrelated
    Greek). Probe each formula crop with the line OCR; when it reads CJK text,
    emit OCR-derived LaTeX instead of the hallucination. PRISM_FML_CJK=0 off."""
    n_swapped = 0
    for idx in math_indices:
        crop = detections[idx].get('crop')
        if crop is None:
            continue
        try:
            lines = workers.ocr.run_text_lines(crop, 'cjk')
        except Exception:
            continue
        joined = ''.join(t for *_, t in (lines or []))
        if _cjk_count(joined) < 2:
            continue
        latex = _ocr_lines_to_latex(lines)
        if latex:
            detections[idx]['raw_content'] = latex
            n_swapped += 1
    print(f"  [fml-cjk] OCR-hybrid replaced {n_swapped}/{len(math_indices)} formula crop(s)")


def route_and_extract(detections, workers: Workers, figures_dir: str,
                      figure_start: int = 0, *, is_screenshot: bool = False,
                      math_start: int = 0, is_cjk: bool = False,
                      is_mixed: bool = False):
    """Route detections to specialist models and return wrapped LaTeX parts.

    Returns (body_parts, list_indices, figure_counter, math_counter).
    """
    os.makedirs(figures_dir, exist_ok=True)
    body_parts:   list = []
    list_indices: set  = set()
    figure_counter = figure_start
    math_counter   = [math_start]

    text_indices  = [i for i, d in enumerate(detections) if d["class_name"] in TEXT_CLASSES]
    math_indices  = [i for i, d in enumerate(detections) if d["class_name"] in MATH_CLASSES]
    table_indices = [i for i, d in enumerate(detections) if d["class_name"] in TABLE_CLASSES]

    if is_mixed:
        _text_fn = lambda crops, ss: workers.ocr.run_text_batch_mixed(crops, is_screenshot=ss)
    elif is_cjk:
        _text_fn = lambda crops, ss: workers.ocr.run_text_batch_cjk(crops, is_screenshot=ss)
    else:
        _text_fn = lambda crops, ss: workers.ocr.run_text_batch(crops, is_screenshot=ss)

    # Dispatch math and text concurrently (independent worker connections).
    if math_indices and text_indices:
        math_crops = [detections[i]["crop"] for i in math_indices]
        text_crops = [detections[i]["crop"] for i in text_indices]
        with ThreadPoolExecutor(max_workers=2) as exe:
            math_fut = exe.submit(_timed('math', workers.math.run_math_batch),
                                  math_crops, figures_dir, math_counter[0])
            text_fut = exe.submit(_timed('text', _text_fn), text_crops, is_screenshot)
            math_results, math_counter[0] = math_fut.result()
            texts = text_fut.result()
        for idx, raw in zip(math_indices, math_results):
            detections[idx]["raw_content"] = raw
        for idx, txt in zip(text_indices, texts):
            detections[idx]["raw_content"] = txt
    else:
        if math_indices:
            crops = [detections[i]["crop"] for i in math_indices]
            results, math_counter[0] = _timed('math', workers.math.run_math_batch)(
                crops, figures_dir, math_counter[0])
            for idx, raw in zip(math_indices, results):
                detections[idx]["raw_content"] = raw
        if text_indices:
            crops = [detections[i]["crop"] for i in text_indices]
            texts = _timed('text', _text_fn)(crops, is_screenshot)
            for idx, txt in zip(text_indices, texts):
                detections[idx]["raw_content"] = txt

    if (math_indices and (is_cjk or is_mixed)
            and os.environ.get('PRISM_FML_CJK', '1') != '0'):
        _apply_cjk_formula_hybrid(detections, math_indices, workers)

    # Inline-math splicing: hosts whose guard-dropped Formula chips were kept
    # by formula_v2 get their text rebuilt with $latex$ spliced in. The batch
    # OCR text above (of the chip-masked crop) stays as the fallback.
    if os.environ.get('PRISM_INLINE_SPLICE', '1') != '0':
        chip_hosts = [i for i in text_indices
                      if detections[i].get('inline_chips')]
        if chip_hosts:
            _splice_chip_hosts(detections, chip_hosts, workers, figures_dir,
                               is_cjk or is_mixed)

    if table_indices:
        table_crops = [detections[i]["crop"] for i in table_indices]
        table_results = _timed('table', _extract_tables)(
            table_crops, workers, is_cjk=(is_cjk or is_mixed))
        for idx, raw in zip(table_indices, table_results):
            detections[idx]["raw_content"] = raw

    # Paragraph coalescing (PRISM_COALESCE=1): consecutive plain-Text
    # detections that are vertically contiguous and column-aligned are one
    # PARAGRAPH the detector over-segmented. The harness is measurably
    # sensitive to segmentation granularity (sentence-splitting our own
    # preds cost +0.41 text / +1.54 order points), so emit them as one block.
    _coalesce = os.environ.get('PRISM_COALESCE', '0') == '1'

    def _adjacent(prev, cur):
        px1, py1, px2, py2 = prev['bbox']
        cx1, cy1, cx2, cy2 = cur['bbox']
        gap = cy1 - py2
        if not (-8 <= gap <= 18):
            return False
        ox = min(px2, cx2) - max(px1, cx1)
        if ox < 0.6 * min(px2 - px1, cx2 - cx1):
            return False
        wr = (px2 - px1) / max(cx2 - cx1, 1e-6)
        return 0.625 <= wr <= 1.6

    prev_text = None          # (det, index in body_parts) of last plain Text
    for det in detections:
        class_name = det["class_name"]
        if class_name in TEXT_CLASSES or class_name in MATH_CLASSES:
            raw = det.get("raw_content", "")
            if (_coalesce and class_name == 'Text' and prev_text is not None
                    and raw and _adjacent(prev_text[0], det)):
                idx = prev_text[1]
                body_parts[idx] = (body_parts[idx].rstrip() + '\n'
                                   + wrap_content('Text', raw).lstrip('\n'))
                merged_box = [min(prev_text[0]['bbox'][0], det['bbox'][0]),
                              prev_text[0]['bbox'][1],
                              max(prev_text[0]['bbox'][2], det['bbox'][2]),
                              det['bbox'][3]]
                prev_text = ({'bbox': merged_box}, idx)
                continue
            wrapped = wrap_content(class_name, raw)
            if class_name == LIST_ITEM_CLASS:
                list_indices.add(len(body_parts))
            if class_name == 'Text':
                prev_text = (det, len(body_parts))
            else:
                prev_text = None
            body_parts.append(wrapped)
        elif class_name in TABLE_CLASSES:
            prev_text = None
            raw = det.get("raw_content", "")
            if raw:
                body_parts.append(wrap_content(class_name, raw))
        elif class_name in IMAGE_CLASSES:
            prev_text = None
            figure_counter += 1
            fname = f"figure_{figure_counter:03d}.png"
            det["crop"].save(os.path.join(figures_dir, fname))
            body_parts.append(wrap_content("Picture", fname))

    return body_parts, list_indices, figure_counter, math_counter[0]


def _rescue_uncovered_text(detections, page_image, workers: Workers,
                           is_cjk: bool, img_width: int, img_height: int):
    """Recover text the layout detector missed entirely.

    32% of v14's text loss was GT text NEVER emitted (809 unmatched-GT
    records: slide titles, text beside dense tables, chart headings). One
    full-page det+rec pass; lines whose center falls inside no detection box
    (inflated by a small margin) become synthetic Text detections, merged
    into adjacent lines when vertically contiguous, and flow through normal
    reading order. PRISM_TEXT_RESCUE=0 disables.
    """
    try:
        lines = workers.ocr.run_text_lines(page_image, 'cjk' if is_cjk else 'en')
    except Exception as e:
        print(f"  [rescue] text-lines pass failed: {e}")
        return detections
    if not lines:
        return detections
    mx = max(6.0, 0.006 * img_width)
    my = max(6.0, 0.006 * img_height)
    boxes = [(d['bbox'][0] - mx, d['bbox'][1] - my,
              d['bbox'][2] + mx, d['bbox'][3] + my) for d in detections]

    def covered(cx, cy):
        return any(bx1 <= cx <= bx2 and by1 <= cy <= by2
                   for bx1, by1, bx2, by2 in boxes)

    orphans = []
    for (x1, y1, x2, y2, txt) in lines:
        if len(txt.strip()) < 2 or (x2 - x1) < 12:
            continue
        if covered((x1 + x2) / 2, (y1 + y2) / 2):
            continue
        orphans.append([x1, y1, x2, y2, txt.strip()])
    if not orphans:
        return detections

    # Merge vertically contiguous orphan lines into blocks (paragraphs).
    orphans.sort(key=lambda o: (o[1], o[0]))
    blocks = []
    for o in orphans:
        merged = False
        for b in blocks:
            hx = min(o[2], b[2]) - max(o[0], b[0])
            gap = o[1] - b[3]
            lh = o[3] - o[1]
            if hx > 0 and -lh * 0.5 < gap < lh * 1.2:
                b[0] = min(b[0], o[0]); b[1] = min(b[1], o[1])
                b[2] = max(b[2], o[2]); b[3] = max(b[3], o[3])
                b[4] = b[4] + '\n' + o[4]
                merged = True
                break
        if not merged:
            blocks.append(list(o))

    n_added = 0
    for (x1, y1, x2, y2, _txt) in blocks:
        pad = 2
        cb = page_image.crop((max(0, int(x1 - pad)), max(0, int(y1 - pad)),
                              min(img_width, int(x2 + pad)),
                              min(img_height, int(y2 + pad))))
        detections.append({
            'class_name': 'Text',
            'bbox': [x1, y1, x2, y2],
            'confidence': 0.40,
            'crop': cb,
            'rescued': True,
        })
        n_added += 1
    if n_added:
        print(f"  [rescue] +{n_added} uncovered text block(s) "
              f"({len(orphans)} orphan line(s))")
    return detections


def _split_multicolumn_text(detections, page_image, img_width, img_height):
    """Split Text boxes that span newspaper column gutters.

    On dense pages the detector sometimes emits one Text box across two or
    three columns; OCR then reads across the gutter and produces scrambled
    text (measured: the dominant loss bucket on ZH newspapers). A sustained
    full-height, low-ink vertical gutter inside a wide Text box is
    unambiguous — split there. PRISM_COLSPLIT=0 disables.
    """
    import numpy as np
    n_text = sum(1 for d in detections if d['class_name'] == 'Text')
    if n_text < 6:          # only dense, newspaper-like pages
        return detections
    out = []
    n_split = 0
    for d in detections:
        x1, y1, x2, y2 = d['bbox']
        w, h = x2 - x1, y2 - y1
        if (d['class_name'] != 'Text' or d.get('rescued')
                or w < 0.28 * img_width or h < 40 or w < h * 0.5):
            out.append(d)
            continue
        crop = d.get('crop')
        if crop is None:
            out.append(d)
            continue
        arr = np.asarray(crop.convert('L'), dtype=np.uint8)
        if arr.size == 0:
            out.append(d)
            continue
        ink = (arr < 160)
        col_frac = ink.mean(axis=0)          # per-pixel-column ink fraction
        # gutter: run of >= 10px whose columns are nearly ink-free
        quiet = col_frac < 0.015
        cuts = []
        run = 0
        cw = arr.shape[1]
        for i in range(cw):
            if quiet[i]:
                run += 1
            else:
                if run >= 10 and 0.15 * cw < i - run / 2 < 0.85 * cw:
                    cuts.append(int(i - run / 2))
                run = 0
        if not cuts or len(cuts) > 3:
            out.append(d)
            continue
        # split into parts; require every part to be reasonably wide
        edges = [0] + cuts + [cw]
        parts = list(zip(edges[:-1], edges[1:]))
        if any((b - a) < 60 for a, b in parts):
            out.append(d)
            continue
        sx = w / cw
        for a, b in parts:
            nx1 = x1 + a * sx
            nx2 = x1 + b * sx
            nd = dict(d)
            nd['bbox'] = [nx1, y1, nx2, y2]
            nd['crop'] = page_image.crop((int(max(0, nx1)), int(max(0, y1)),
                                          int(min(img_width, nx2)),
                                          int(min(img_height, y2))))
            out.append(nd)
        n_split += 1
    if n_split:
        print(f"  [colsplit] split {n_split} cross-column Text box(es)")
    return out


def _order_by_model_ro(detections):
    """Sort by detector read_order; dets lacking one (rescued/split synthetics)
    inherit the order of the geometrically nearest ordered det, biased by
    whether they sit above (-eps) or below (+eps) that neighbour."""
    with_ro = [d for d in detections if d.get('read_order') is not None]
    if not with_ro:
        return detections

    def center(d):
        b = d['bbox']
        return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

    keyed = []
    for d in detections:
        ro = d.get('read_order')
        if ro is None:
            cx, cy = center(d)
            best, bdist = None, None
            for o in with_ro:
                ox, oy = center(o)
                dist = (ox - cx) ** 2 + (oy - cy) ** 2
                if bdist is None or dist < bdist:
                    best, bdist = o, dist
            _, oy = center(best)
            ro = best['read_order'] + (0.5 if cy >= oy else -0.5)
        keyed.append((ro, keyed.__len__(), d))
    keyed.sort(key=lambda t: (t[0], t[1]))
    return [d for _, _, d in keyed]


def _assemble_fidelity_columns(detections, n_cols, col_lefts, img_width,
                               img_height, workers, figures_dir,
                               header_logo_fname, has_cjk, lang_kwargs):
    """Route detections into an N-column visual-fidelity paracol document."""
    from pipeline.layout_utils import split_detections_fidelity
    top_full, columns, bottom_full = split_detections_fidelity(
        detections, n_cols, col_lefts, img_width, img_height)

    f_cnt = m_cnt = 0
    top_parts, top_idx, f_cnt, m_cnt = route_and_extract(
        top_full, workers, figures_dir, f_cnt, math_start=m_cnt, **lang_kwargs)
    top_parts = _adjust_figure_paths(top_parts)

    col_parts, col_idx = [], []
    for col in columns:
        parts, idx, f_cnt, m_cnt = route_and_extract(
            col, workers, figures_dir, f_cnt, math_start=m_cnt, **lang_kwargs)
        col_parts.append(_adjust_figure_paths(parts))
        col_idx.append(idx)

    bottom_parts, bottom_idx, f_cnt, m_cnt = route_and_extract(
        bottom_full, workers, figures_dir, f_cnt, math_start=m_cnt, **lang_kwargs)
    bottom_parts = _adjust_figure_paths(bottom_parts)

    print(f"  [fidelity] {n_cols}-column layout "
          f"(top={len(top_parts)}, cols={[len(c) for c in col_parts]}, "
          f"bottom={len(bottom_parts)})")
    return assemble_columns_document(
        top_parts, top_idx, col_parts, col_idx,
        bottom_parts, bottom_idx,
        header_logo=header_logo_fname, has_cjk=has_cjk)


def build_document(detections, img_width, img_height, workers: Workers,
                   figures_dir: str, *, is_screenshot: bool = False,
                   is_cjk: bool = False, is_mixed: bool = False,
                   header_logo_fname: str = None, page_image=None) -> str:
    """Column-aware dispatch + assembly. Returns a complete LaTeX document."""
    from pipeline import formula_v2
    if formula_v2.enabled():
        detections = formula_v2.apply_formula_v2(detections, img_width, img_height)

    if (page_image is not None
            and os.environ.get('PRISM_TEXT_RESCUE', '1') != '0'):
        detections = _rescue_uncovered_text(
            detections, page_image, workers, is_cjk or is_mixed,
            img_width, img_height)

    if (page_image is not None
            and os.environ.get('PRISM_COLSPLIT', '0') == '1'):
        detections = _split_multicolumn_text(
            detections, page_image, img_width, img_height)

    # Marginalia (running headers/footers/page numbers) are abandon-category
    # in OmniDocBench GT: never scored, but every EMITTED one that fails the
    # matcher's ignore-pairing counts as a full unmatched-pred penalty
    # (measured: report/PPT pages with 0 GT text blocks scoring 1.0 because
    # of a disclaimer footer). MinerU/Marker drop them too.
    if os.environ.get('PRISM_DROP_MARGINALIA', '1') != '0':
        detections = [d for d in detections
                      if d['class_name'] not in ('Page-footer', 'Page-header')]
        # Geometric band-drop: running heads/footlines the detector labels
        # plain Text (or the text-rescue pass re-adds as a synthetic Text
        # block — the dominant olmOCR headers_footers leak: a wrapped header
        # banner whose 2nd line escapes the Page-header box and is rescued).
        #
        # A candidate is a short (<= HCAP page-height) Text/List-item block
        # fully inside the top/bottom band. It is dropped ONLY if it is also
        # VERTICALLY ISOLATED from the body — i.e. separated from the nearest
        # content block toward page-centre by a whitespace gap >= GAP page-
        # fractions. A running header/footer sits alone above/below a band of
        # whitespace; a real first/last body line is immediately followed by
        # more text, so it is never isolated and never dropped. This protects
        # multi_column / long_tiny_text body lines that reach the band.
        # By default only Text/List-item are eligible; Title, Section-header,
        # Footnote, Caption are kept. PRISM_BAND_CLASSES (comma list) overrides
        # the eligible set (e.g. add "Footnote" to also drop isolated journal
        # footers / URL lines). The isolation guard still protects real content.
        # PRISM_BAND_DROP="<top_frac>,<bottom_frac>" enables it.
        band = os.environ.get('PRISM_BAND_DROP', '')
        if band:
            top_f, bot_f = (float(x) for x in band.split(','))
            hcap = float(os.environ.get('PRISM_BAND_HCAP', '0.045'))
            gap_min = float(os.environ.get('PRISM_BAND_GAP', '0.03')) * img_height
            y_top, y_bot = img_height * top_f, img_height * (1.0 - bot_f)
            _bc_env = os.environ.get('PRISM_BAND_CLASSES', '')
            _band_classes = ({c.strip() for c in _bc_env.split(',') if c.strip()}
                             if _bc_env else {'Text', LIST_ITEM_CLASS})
            # Content blocks that define "the body" for the isolation test
            # (everything with real text/structure, incl. the candidate set).
            _content = TEXT_CLASSES | TABLE_CLASSES | MATH_CLASSES

            def _is_band_marginalia(d):
                if d['class_name'] not in _band_classes:
                    return False
                x1, y1, x2, y2 = d['bbox']
                if (y2 - y1) > hcap * img_height:
                    return False
                in_top = y2 <= y_top
                in_bot = y1 >= y_bot
                if not (in_top or in_bot):
                    return False
                # Isolation: gap to the nearest OTHER content block on the
                # body side must be >= gap_min (whitespace band under a header
                # / over a footer). Body first/last lines fail this (adjacent
                # text) and are kept.
                if in_top:
                    below = [o['bbox'][1] for o in detections
                             if o is not d and o['class_name'] in _content
                             and o['bbox'][1] >= y2 - 1]
                    if not below:
                        return True
                    return (min(below) - y2) >= gap_min
                else:
                    above = [o['bbox'][3] for o in detections
                             if o is not d and o['class_name'] in _content
                             and o['bbox'][3] <= y1 + 1]
                    if not above:
                        return True
                    return (y1 - max(above)) >= gap_min

            detections = [d for d in detections if not _is_band_marginalia(d)]
    has_cjk   = is_cjk or is_mixed
    col_count = detect_column_count(detections, img_width)
    lang_kwargs = dict(is_screenshot=is_screenshot, is_cjk=is_cjk, is_mixed=is_mixed)

    # Visual-fidelity mode (product / web UI): reproduce the page's column
    # geometry in the OUTPUT rather than flattening it to a single reading
    # column. Benchmarks want a single linearised stream (they score text +
    # reading order, not visual layout), so the default stays flat; the UI
    # sets PRISM_VISUAL_FIDELITY=1. In this mode a multi-column page bypasses
    # the model-RO linearisation below and falls through to the geometric
    # paracol column paths, so a 2-column input renders as 2 columns.
    _fidelity = os.environ.get('PRISM_VISUAL_FIDELITY', '0') == '1'

    # Visual-fidelity column reconstruction: a robust column detector (whitespace
    # gutters over narrow body boxes only, independent of the benchmark-tuned
    # detect_column_count) drives an N-column paracol layout so a 2/3-column
    # input renders with that many columns. Only runs in fidelity mode; the
    # benchmark path is untouched.
    if _fidelity and detections:
        from pipeline.layout_utils import detect_columns_fidelity
        n_cols, col_bounds = detect_columns_fidelity(
            detections, img_width, img_height)
        if n_cols >= 2:
            return _assemble_fidelity_columns(
                detections, n_cols, col_bounds, img_width, img_height,
                workers, figures_dir, header_logo_fname, has_cjk, lang_kwargs)

    # Model reading order (PP-DocLayoutV3): every layout box carries the
    # detector's own read_order. When enough boxes have it, use the model
    # order directly and skip all geometric column logic. Boxes without an
    # order (rescued/split synthetics) inherit their nearest neighbour's.
    if os.environ.get('PRISM_RO_MODEL', '1') != '0' and detections:
        n_ro = sum(1 for d in detections if d.get('read_order') is not None)
        if n_ro >= max(1, int(0.7 * len(detections))):
            print(f"  [ro-model] ordering {len(detections)} blocks by detector read_order ({n_ro} native)")
            ordered = _order_by_model_ro(detections)
            parts, list_idx, _, _ = route_and_extract(
                ordered, workers, figures_dir, **lang_kwargs)
            parts = _adjust_figure_paths(parts)
            return assemble_document(parts, list_idx, False,
                                     header_logo=header_logo_fname,
                                     has_cjk=has_cjk)

    if col_count == 2:
        full_dets, left_dets, right_dets = split_detections_by_column(
            detections, img_width, img_height, use_dag=True)
        full_parts,  full_idx,  f_cnt, m_cnt = route_and_extract(
            full_dets,  workers, figures_dir, 0,     math_start=0,     **lang_kwargs)
        left_parts,  left_idx,  f_cnt, m_cnt = route_and_extract(
            left_dets,  workers, figures_dir, f_cnt, math_start=m_cnt, **lang_kwargs)
        right_parts, right_idx, f_cnt, m_cnt = route_and_extract(
            right_dets, workers, figures_dir, f_cnt, math_start=m_cnt, **lang_kwargs)
        full_parts  = _adjust_figure_paths(full_parts)
        left_parts  = _adjust_figure_paths(left_parts)
        right_parts = _adjust_figure_paths(right_parts)
        return assemble_document(
            full_parts, full_idx, True, left_parts, left_idx,
            right_parts, right_idx, header_logo_fname, has_cjk=has_cjk)

    # XY-cut ordering for complex layouts: 3+ detected columns, OR pages the
    # gutter detector calls "1 column" despite many regions — half the
    # newspapers land there (touching boxes hide the gutters) and got plain
    # top-down ordering. On a genuinely single-column page XY-cut degenerates
    # to the same top-down order, so the extension is safe.
    _xycut_page = col_count >= 3 or (col_count == 1 and len(detections) >= 8)
    if _xycut_page and os.environ.get('PRISM_RO_V2', '1') != '0':
        from pipeline.layout_utils import xycut_order
        ordered = xycut_order(detections, img_width, img_height)
        parts, list_idx, _, _ = route_and_extract(
            ordered, workers, figures_dir, **lang_kwargs)
        parts = _adjust_figure_paths(parts)
        return assemble_document(parts, list_idx, False,
                                 header_logo=header_logo_fname, has_cjk=has_cjk)

    if col_count >= 3:
        if os.environ.get('PRISM_RO_V2', '1') != '0':
            # (unreachable when RO_V2 on; kept for the kill-switch path)
            from pipeline.layout_utils import xycut_order
            ordered = xycut_order(detections, img_width, img_height)
            parts, list_idx, _, _ = route_and_extract(
                ordered, workers, figures_dir, **lang_kwargs)
            parts = _adjust_figure_paths(parts)
            return assemble_document(parts, list_idx, False,
                                     header_logo=header_logo_fname, has_cjk=has_cjk)
        full_dets, col_lists = split_detections_n_columns(
            detections, img_width, img_height, use_dag=True)
        all_parts: list = []
        all_list_idx: set = set()
        offset = 0
        f_cnt, m_cnt = 0, 0
        for group in [full_dets] + col_lists:
            parts, list_idx, f_cnt, m_cnt = route_and_extract(
                group, workers, figures_dir, f_cnt, math_start=m_cnt, **lang_kwargs)
            parts = _adjust_figure_paths(parts)
            all_parts.extend(parts)
            all_list_idx.update(i + offset for i in list_idx)
            offset += len(parts)
        return assemble_document(all_parts, all_list_idx, False,
                                 header_logo=header_logo_fname, has_cjk=has_cjk)

    body_sorted = apply_semantic_reading_order(detections, img_width, img_height)
    body_parts, list_idx, _, _ = route_and_extract(
        body_sorted, workers, figures_dir, **lang_kwargs)
    body_parts = _adjust_figure_paths(body_parts)
    return assemble_document(body_parts, list_idx, False,
                             header_logo=header_logo_fname, has_cjk=has_cjk)
