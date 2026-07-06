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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from pipeline.layout_utils import (
    apply_semantic_reading_order, xyxy_to_pil_crop,
    detect_column_count, split_detections_by_column, split_detections_n_columns,
)
from pipeline.latex_builder import wrap_content, assemble_document


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
        results = []
        pending = []  # indices that need the TATR fallback
        for i, crop in enumerate(table_crops):
            html = rtable.build_table_html(crop)
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
        return [r or '' for r in results]
    return _extract_tables_tatr(table_crops, workers, is_cjk=is_cjk)


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
    if n_swapped:
        print(f"  [fml-cjk] OCR-hybrid replaced {n_swapped} CJK formula(s)")


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
            math_fut = exe.submit(workers.math.run_math_batch, math_crops, figures_dir, math_counter[0])
            text_fut = exe.submit(_text_fn, text_crops, is_screenshot)
            math_results, math_counter[0] = math_fut.result()
            texts = text_fut.result()
        for idx, raw in zip(math_indices, math_results):
            detections[idx]["raw_content"] = raw
        for idx, txt in zip(text_indices, texts):
            detections[idx]["raw_content"] = txt
    else:
        if math_indices:
            crops = [detections[i]["crop"] for i in math_indices]
            results, math_counter[0] = workers.math.run_math_batch(crops, figures_dir, math_counter[0])
            for idx, raw in zip(math_indices, results):
                detections[idx]["raw_content"] = raw
        if text_indices:
            crops = [detections[i]["crop"] for i in text_indices]
            texts = _text_fn(crops, is_screenshot)
            for idx, txt in zip(text_indices, texts):
                detections[idx]["raw_content"] = txt

    if (math_indices and (is_cjk or is_mixed)
            and os.environ.get('PRISM_FML_CJK', '1') != '0'):
        _apply_cjk_formula_hybrid(detections, math_indices, workers)

    if table_indices:
        table_crops = [detections[i]["crop"] for i in table_indices]
        table_results = _extract_tables(table_crops, workers, is_cjk=(is_cjk or is_mixed))
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
    has_cjk   = is_cjk or is_mixed
    col_count = detect_column_count(detections, img_width)
    lang_kwargs = dict(is_screenshot=is_screenshot, is_cjk=is_cjk, is_mixed=is_mixed)

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
