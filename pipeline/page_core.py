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

    if table_indices:
        table_crops = [detections[i]["crop"] for i in table_indices]
        table_results = _extract_tables(table_crops, workers, is_cjk=(is_cjk or is_mixed))
        for idx, raw in zip(table_indices, table_results):
            detections[idx]["raw_content"] = raw

    for det in detections:
        class_name = det["class_name"]
        if class_name in TEXT_CLASSES or class_name in MATH_CLASSES:
            raw = det.get("raw_content", "")
            wrapped = wrap_content(class_name, raw)
            if class_name == LIST_ITEM_CLASS:
                list_indices.add(len(body_parts))
            body_parts.append(wrapped)
        elif class_name in TABLE_CLASSES:
            raw = det.get("raw_content", "")
            if raw:
                body_parts.append(wrap_content(class_name, raw))
        elif class_name in IMAGE_CLASSES:
            figure_counter += 1
            fname = f"figure_{figure_counter:03d}.png"
            det["crop"].save(os.path.join(figures_dir, fname))
            body_parts.append(wrap_content("Picture", fname))

    return body_parts, list_indices, figure_counter, math_counter[0]


def build_document(detections, img_width, img_height, workers: Workers,
                   figures_dir: str, *, is_screenshot: bool = False,
                   is_cjk: bool = False, is_mixed: bool = False,
                   header_logo_fname: str = None) -> str:
    """Column-aware dispatch + assembly. Returns a complete LaTeX document."""
    from pipeline import formula_v2
    if formula_v2.enabled():
        detections = formula_v2.apply_formula_v2(detections, img_width, img_height)

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
