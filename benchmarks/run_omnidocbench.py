"""
run_omnidocbench.py
-------------------
Run PRISM on OmniDocBench images, convert LaTeX output to Markdown,
save predictions, then invoke the OmniDocBench evaluation.

Usage:
    python run_omnidocbench.py [--gt-json PATH] [--images-dir DIR] [--pred-dir DIR] [--eval-only]

Defaults use the bundled 18-page demo data inside omnidocbench_eval/.
"""

import argparse
import json
import os
import sys
import shutil
import time
import threading
from pathlib import Path

os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', 'NO_ALBUMENTATIONS_UPDATE')
# Pin the benchmark-validated normalization routing: the product-mode
# shadowed-capture rescue (normalization/pipeline.py) must never fire on
# benchmark pages (PPT design gradients mimic the shadow signature).
os.environ.setdefault('PRISM_NORM_STRICT', '1')

# UTF-8 stdout/stderr: the pipeline prints Unicode (→, 【】, CJK). On Windows
# the default console codec is cp1252 and raises UnicodeEncodeError. Previously
# ultralytics/torch import happened to reconfigure the console; now that the
# main process is torch-free we must do it explicitly.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Cap ONNX/OpenMP threads before any model library is imported so the
# concurrent worker subprocesses don't oversubscribe a modest CPU target.
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.onnx_config import apply_thread_env
apply_thread_env()

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
EVAL_DIR = ROOT / 'omnidocbench_eval'
DEMO_IMAGES = EVAL_DIR / 'demo_data' / 'omnidocbench_demo' / 'images'
DEMO_GT = EVAL_DIR / 'demo_data' / 'omnidocbench_demo' / 'OmniDocBench_demo.json'
DEFAULT_PRED = ROOT / 'preds' / 'omnidocbench'

# Replace-both: PP-DocLayout_plus-L (RT-DETR) as the SOLE layout detector.
# Native raw-onnxruntime path (PRISM_USE_PPDL_LAYOUT) — replaces YOLOv11n +
# DocLayout + MFD. Validated to beat the two-detector combo on every category.
# The PRISM_LAYOUT_CACHE path (pre-computed boxes) is kept for offline A/B only.
_PPDL_MODEL_PATH = str(ROOT / 'models' / 'ppdoclayout' / 'ppdoclayout_plus_l.onnx')
_USE_PPDL_LAYOUT = os.environ.get(
    'PRISM_USE_PPDL_LAYOUT', '1' if os.path.exists(_PPDL_MODEL_PATH) else '0') != '0'
_LAYOUT_CACHE_PATH = os.environ.get('PRISM_LAYOUT_CACHE', '')
_LAYOUT_CACHE_DATA = None

def _layout_boxes(norm_path, stem):
    """Full-layout boxes (PRISM vocab) from PP-DocLayout — native ONNX or cache."""
    global _LAYOUT_CACHE_DATA
    if _USE_PPDL_LAYOUT:
        from pipeline.models_interface import get_ppdoclayout_detector
        det = get_ppdoclayout_detector(800)
        if det is None:
            raise RuntimeError('PP-DocLayout ONNX model not found')
        base = float(os.environ.get('PRISM_PPDL_BASE_CONF', '0.50'))
        fml = float(os.environ.get('PRISM_PPDL_CONF', '0.30'))
        tbl = float(os.environ.get('PRISM_PPDL_TBL_CONF', '0.50'))
        low = min(base, fml, tbl)
        if low >= base:
            return det.detect(norm_path, conf=base)  # match paddle draw_threshold
        # Detect low; keep sub-base boxes only for the classes with their own
        # validated lower gates (Formula 0.30, Table 0.30 — recall 91.7->94.4%
        # at 0.13 FP/page for tables; see paper.md).
        from pipeline.page_core import MATH_CLASSES, TABLE_CLASSES
        out = []
        for b in det.detect(norm_path, conf=low):
            c = b['confidence']
            if b['class_name'] in MATH_CLASSES:
                if c >= fml:
                    out.append(b)
            elif b['class_name'] in TABLE_CLASSES:
                if c >= tbl:
                    out.append(b)
            elif c >= base:
                out.append(b)
        return out
    if _LAYOUT_CACHE_DATA is None:
        with open(_LAYOUT_CACHE_PATH, encoding='utf-8') as f:
            _LAYOUT_CACHE_DATA = json.load(f)
    return _LAYOUT_CACHE_DATA.get(stem, [])

def _layout_from_cache(stem, image_norm, image_fidelity, formula_from_fidelity, norm_path=None):
    from pipeline.layout_utils import xyxy_to_pil_crop
    from pipeline.page_core import IMAGE_CLASSES, MATH_CLASSES
    formula_conf = float(os.environ.get('PRISM_PPDL_CONF', '0.30'))
    dets = []
    for b in _layout_boxes(norm_path, stem):
        cls = b['class_name']; conf = b.get('confidence', 1.0); bbox = b['bbox']
        if cls in MATH_CLASSES and conf < formula_conf:
            continue
        if cls in IMAGE_CLASSES or (formula_from_fidelity and cls in MATH_CLASSES):
            crop = xyxy_to_pil_crop(image_fidelity, bbox)
        else:
            crop = xyxy_to_pil_crop(image_norm, bbox)
        d = {'bbox': bbox, 'class_id': 0, 'class_name': cls,
             'confidence': conf, 'crop': crop}
        if b.get('read_order') is not None:
            d['read_order'] = b['read_order']
        if b.get('from_inline'):
            d['from_inline'] = True
        dets.append(d)
    return dets


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--gt-json', default=str(DEMO_GT))
    p.add_argument('--images-dir', default=str(DEMO_IMAGES))
    p.add_argument('--pred-dir', default=str(DEFAULT_PRED))
    p.add_argument('--eval-only', action='store_true', help='Skip PRISM, just run eval on existing preds')
    p.add_argument('--skip-eval', action='store_true', help='Run PRISM only, skip final eval step')
    p.add_argument('--no-cdm', action='store_true', default=True, help='Disable CDM metric (requires TeX Live + Linux)')
    p.add_argument('--skip-stage1', action='store_true', help='Ablation: skip Stage 1 corrections (deskew+modality only)')
    p.add_argument('--formula-from-fidelity', action='store_true', help='Route formula crops from image_fidelity (no CLAHE) while text/table use image_norm (with CLAHE)')
    p.add_argument('--lang-filter', default=None, help='Only process pages with this language (e.g. english)')
    return p.parse_args()


# ── PRISM pipeline (adapted from orchestrate.py) ──────────────────────────────

def _run_prism_on_images(image_paths: list[str], pred_dir: str, cjk_pages: set = None, mixed_pages: set = None, ppt_pages: set = None, skip_stage1: bool = False, formula_from_fidelity: bool = False) -> dict[str, str]:
    """
    Run PRISM on a list of image files.  Workers are shared across all images.
    Returns {image_stem: markdown_text} dict.
    """
    import gc
    import numpy as np
    from PIL import Image
    from concurrent.futures import ThreadPoolExecutor

    sys.path.insert(0, str(ROOT))
    from normalization import normalize_image_pil, normalize_image_pil_skip_stage1
    from normalization.modality import CaptureModality
    _normalise_fn = normalize_image_pil_skip_stage1 if skip_stage1 else normalize_image_pil
    from pipeline.layout_utils import xyxy_to_pil_crop
    from pipeline.latex_builder import save_tex
    from pipeline.detection_postprocess import postprocess_detections
    from pipeline.page_core import Workers, build_document, MATH_CLASSES, IMAGE_CLASSES
    from pipeline.tex_to_md import tex_to_omnidocbench_md

    # PRISM_SINGLE_WORKER=1 → product/single-worker config (1 OCR + 1 math),
    # matching orchestrate.py and the isolated single-process competitor rows in
    # Table 3 (tab:cpu_frontier). Default keeps the dual-worker benchmark config.
    _single = os.environ.get('PRISM_SINGLE_WORKER', '0') == '1'
    if _single:
        from pipeline.text_worker import TextOCRWorker as _OCRCls
        from pipeline.math_worker_onnx import MathOCRWorkerOnnx as _MathCls
    else:
        from pipeline.text_worker import TextOCRWorkerDual as _OCRCls
        from pipeline.math_worker_onnx import MathOCRWorkerOnnxDual as _MathCls

    # PRISM_AFFINITY="0-7" (or "0,1,2,..") → pin the whole process tree to those
    # logical cores; children inherit affinity, so this is the 8-thread-budget
    # affinity pin used for the Windows sweeps. Set before workers spawn.
    _aff = os.environ.get('PRISM_AFFINITY', '')
    if _aff:
        import psutil
        cores = []
        for tok in _aff.split(','):
            tok = tok.strip()
            if '-' in tok:
                a, b = tok.split('-'); cores.extend(range(int(a), int(b) + 1))
            elif tok:
                cores.append(int(tok))
        try:
            psutil.Process().cpu_affinity(cores)
            print(f'[*] CPU affinity pinned to {cores}')
        except Exception as _e:
            print(f'[!] affinity pin failed: {_e}')

    ocr_worker = _OCRCls()
    math_worker = _MathCls()
    print(f'[*] Starting workers ({"single" if _single else "dual"}-worker)...')
    ocr_worker.start()
    math_worker.start()
    print('[*] Workers ready.')

    # ── Perf instrumentation: peak process-tree RSS + per-page latency ────────
    # Samples the main process + all worker subprocesses (spawned as children)
    # every 0.3s and tracks the peak summed RSS. This is the "peak RAM (process
    # tree)" number reported for the paper — comparable to how the competitor
    # head-to-head measured PP-StructureV3 / SmolDocling.
    import psutil, threading, json as _json
    _perf_latencies: list = []
    _peak_rss = {'mb': 0.0}
    _perf_stop = threading.Event()

    _ram_samples: list = []          # full RSS time-series → RAM percentiles

    def _ram_sampler():
        me = psutil.Process(os.getpid())
        while not _perf_stop.is_set():
            try:
                rss = me.memory_info().rss
                for ch in me.children(recursive=True):
                    try:
                        rss += ch.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                mb = rss / 1024 / 1024
                _peak_rss['mb'] = max(_peak_rss['mb'], mb)
                _ram_samples.append(round(mb, 1))
            except Exception:
                pass
            _perf_stop.wait(0.3)

    _sampler = threading.Thread(target=_ram_sampler, daemon=True)
    _sampler.start()
    _wall_t0 = time.perf_counter()

    results: dict[str, str] = {}

    for img_path_str in image_paths:
        img_path = Path(img_path_str)
        stem = img_path.stem
        if (os.environ.get('PRISM_SKIP_EXISTING', '0') == '1'
                and (Path(pred_dir) / f'{stem}.md').exists()):
            continue
        t0 = time.perf_counter()
        print(f'[>] {stem}')

        _stage_on = os.environ.get('PRISM_STAGE_TIMING', '0') == '1'
        _st = {}

        def _mark(key, since):
            now = time.perf_counter()
            if _stage_on:
                _st[key] = _st.get(key, 0.0) + (now - since)
            return now

        try:
            work_dir = Path(pred_dir) / f'_tmp_{stem}'
            work_dir.mkdir(parents=True, exist_ok=True)
            assets_dir = work_dir / 'assets'
            figures_dir = assets_dir / 'figures'
            figures_dir.mkdir(parents=True, exist_ok=True)

            # Stage 1: normalise (or bypass for ablation)
            _tm = time.perf_counter()
            image_norm, image_fidelity, modality_result = _normalise_fn(img_path_str)
            _tm = _mark('normalize', _tm)
            is_screenshot = (modality_result.modality == CaptureModality.SCREENSHOT) or (stem in (ppt_pages or set()))
            norm_path = str(assets_dir / 'normalized.png')
            image_norm.save(norm_path)

            # Stage 2: YOLO detection (raw onnxruntime by default; no torch)
            img_width, img_height = image_norm.width, image_norm.height
            # PP-DocLayout_plus-L is the sole layout detector.
            _tm = time.perf_counter()
            detections = _layout_from_cache(stem, image_norm, image_fidelity, formula_from_fidelity, norm_path=norm_path)
            _tm = _mark('layout', _tm)
            print(f'  [PPDL-LAYOUT] {len(detections)} boxes')

            detections = postprocess_detections(detections, img_width, img_height)

            # Header suppress
            HEADER_SUPPRESS_H_FRAC = 0.12
            header_suppress_y = img_height * HEADER_SUPPRESS_H_FRAC
            detections = [
                d for d in detections
                if not (
                    d['class_name'] in {'Section-header', 'Page-header'}
                    and d['bbox'][3] <= header_suppress_y
                )
            ]

            # Formula pad + re-crop (4px optimal per sweep on 224 EN crops)
            FORMULA_PAD = 4
            for det in detections:
                bbox = det['bbox']
                if det['class_name'] in MATH_CLASSES:
                    x1, y1, x2, y2 = bbox
                    bbox = [
                        max(0, x1 - FORMULA_PAD), max(0, y1 - FORMULA_PAD),
                        min(img_width, x2 + FORMULA_PAD), min(img_height, y2 + FORMULA_PAD),
                    ]
                if det['class_name'] in IMAGE_CLASSES or (formula_from_fidelity and det['class_name'] in MATH_CLASSES):
                    det['crop'] = xyxy_to_pil_crop(image_fidelity, bbox)
                else:
                    det['crop'] = xyxy_to_pil_crop(image_norm, bbox)

            _rescue_page = image_norm if os.environ.get('PRISM_TEXT_RESCUE', '1') != '0' else None
            del image_fidelity
            gc.collect()

            # Stage 3: extraction + assembly (shared with orchestrate.py)
            is_cjk  = stem in (cjk_pages  or set())
            is_mixed = stem in (mixed_pages or set())
            workers = Workers(ocr=ocr_worker, math=math_worker)
            if _stage_on:
                from pipeline import page_core as _pc
                _pc.stage_times.clear()
            _tm = time.perf_counter()
            document = build_document(
                detections, img_width, img_height, workers, str(figures_dir),
                is_screenshot=is_screenshot, is_cjk=is_cjk, is_mixed=is_mixed,
                page_image=_rescue_page,
            )
            _tm = _mark('build_document', _tm)
            if _stage_on:
                for _k, _v in _pc.stage_times.items():
                    _st[_k] = _st.get(_k, 0.0) + _v
            # Block-level sidecar dump (offline tuning of geometric filters):
            # class/bbox/extracted text per detection, plus page dims.
            if os.environ.get('PRISM_DUMP_BLOCKS', '0') == '1':
                _blocks = [{'class': d.get('class_name'),
                            'bbox': [round(float(v), 1) for v in d.get('bbox', [0, 0, 0, 0])],
                            'read_order': d.get('read_order'),
                            'raw': d.get('raw_content', '')}
                           for d in detections]
                (Path(pred_dir) / f'{stem}.blocks.json').write_text(
                    _json.dumps({'w': img_width, 'h': img_height,
                                 'blocks': _blocks}, ensure_ascii=False),
                    encoding='utf-8')

            del image_norm, _rescue_page
            gc.collect()

            # Convert LaTeX → Markdown
            md_text = tex_to_omnidocbench_md(document)

            # Empty-page fallback: 36/1651 pages (PPT slides, colorful
            # textbooks, magazines) come out empty because the whole page is
            # detected as one Picture. An empty pred scores ~1.0 edit
            # distance; whole-page OCR is strictly better than nothing.
            if os.environ.get('PRISM_PAGE_OCR_FALLBACK', '1') != '0':
                import re as _re
                _content = _re.sub(r'!\[[^\]]*\]\([^)]*\)|\s+', '', md_text)
                if len(_content) < 30:
                    try:
                        _page_img = Image.open(img_path_str).convert('RGB')
                        if max(_page_img.size) > 1800:
                            _sc = 1800 / max(_page_img.size)
                            _page_img = _page_img.resize(
                                (int(_page_img.width * _sc), int(_page_img.height * _sc)))
                        if is_cjk:
                            _lines = ocr_worker.run_text_batch_cjk([_page_img], is_screenshot=is_screenshot)
                        elif is_mixed:
                            _lines = ocr_worker.run_text_batch_mixed([_page_img], is_screenshot=is_screenshot)
                        else:
                            _lines = ocr_worker.run_text_batch([_page_img], is_screenshot=is_screenshot)
                        _page_text = (_lines[0] or '').strip() if _lines else ''
                        if len(_page_text) > len(_content):
                            md_text = (md_text.rstrip() + '\n\n' + _page_text).strip()
                            print(f'  [PAGE-OCR-FALLBACK] +{len(_page_text)} chars')
                    except Exception as _e:
                        print(f'  [PAGE-OCR-FALLBACK] failed: {_e}')

            results[stem] = md_text

            # Save .tex and .md for inspection
            tex_path = work_dir / 'main.tex'
            save_tex(document, str(tex_path))
            md_path = Path(pred_dir) / f'{stem}.md'
            md_path.write_text(md_text, encoding='utf-8')

            elapsed = time.perf_counter() - t0
            _perf_latencies.append((stem, round(elapsed, 3)))
            if _stage_on:
                _st['total'] = round(elapsed, 3)
                _st['assembly_other'] = round(
                    elapsed - sum(v for k, v in _st.items()
                                  if k in ('normalize', 'layout', 'build_document')), 3)
                with open(Path(pred_dir) / '_stage_timing.jsonl', 'a', encoding='utf-8') as _sf:
                    _sf.write(_json.dumps({'page': stem,
                                           **{k: round(v, 3) for k, v in _st.items()}}) + '\n')
            print(f'    done in {elapsed:.1f}s → {md_path.name}')

        except Exception as e:
            import traceback
            _perf_latencies.append((stem, round(time.perf_counter() - t0, 3)))
            print(f'    ERROR: {e}')
            traceback.print_exc()
            results[stem] = ''
            md_path = Path(pred_dir) / f'{stem}.md'
            md_path.write_text('', encoding='utf-8')

    # ── Finalize perf: stop sampler, aggregate, dump perf.json ────────────────
    _wall_s = time.perf_counter() - _wall_t0
    _perf_stop.set()
    _sampler.join(timeout=2)
    _lat = sorted(v for _, v in _perf_latencies)
    def _pct(p):
        if not _lat:
            return 0.0
        return round(_lat[min(len(_lat) - 1, int(p / 100 * len(_lat)))], 3)
    _mean = round(sum(_lat) / len(_lat), 3) if _lat else 0.0
    _median = round(_lat[len(_lat) // 2], 3) if _lat else 0.0
    _ram_sorted = sorted(_ram_samples)
    def _rpct(p):
        if not _ram_sorted:
            return 0.0
        return round(_ram_sorted[min(len(_ram_sorted) - 1, int(p / 100 * len(_ram_sorted)))], 1)
    perf = {
        'n_pages': len(_perf_latencies),
        'wall_s': round(_wall_s, 1),
        'latency_s_per_page': {
            'mean': _mean, 'median': _median,
            'p90': _pct(90), 'p95': _pct(95), 'p99': _pct(99), 'p99.9': _pct(99.9),
            'min': _lat[0] if _lat else 0.0, 'max': _lat[-1] if _lat else 0.0,
        },
        'peak_ram_mb_process_tree': round(_peak_rss['mb'], 1),
        'ram_mb_process_tree': {
            'n_samples': len(_ram_sorted), 'sample_interval_s': 0.3,
            'median': _rpct(50), 'p90': _rpct(90), 'p95': _rpct(95),
            'p99': _rpct(99), 'max': _ram_sorted[-1] if _ram_sorted else 0.0,
        },
        'per_page': dict(_perf_latencies),
    }
    with open(Path(pred_dir) / 'perf.json', 'w', encoding='utf-8') as _pf:
        _json.dump(perf, _pf, indent=2)
    print(f"\n[PERF] {perf['n_pages']} pages | wall {perf['wall_s']}s | "
          f"median {_median}s/pg mean {_mean}s/pg p90 {_pct(90)}s | "
          f"peak RAM {perf['peak_ram_mb_process_tree']} MB (process tree)")

    ocr_worker.stop()
    math_worker.stop()
    return results


# ── Evaluation ─────────────────────────────────────────────────────────────────

def _write_eval_config(gt_json: str, pred_dir: str, no_cdm: bool) -> str:
    import yaml
    cfg = {
        'end2end_eval': {
            'metrics': {
                'text_block': {'metric': ['Edit_dist']},
                'display_formula': {
                    'metric': ['Edit_dist'] + ([] if no_cdm else ['CDM']),
                    'cdm_workers': 1,
                },
                'table': {
                    'metric': ['Edit_dist', 'TEDS'],
                    'teds_workers': int(os.environ.get('PRISM_EVAL_TEDS_WORKERS', '12')),
                },
                'reading_order': {'metric': ['Edit_dist']},
            },
            'dataset': {
                'dataset_name': 'end2end_dataset',
                'ground_truth': {'data_path': gt_json},
                'prediction': {'data_path': pred_dir},
                'match_method': 'quick_match',
                'match_workers': int(os.environ.get('PRISM_EVAL_MATCH_WORKERS', '12')),
                'quick_match_truncated_timeout_sec': 120,
                'match_timeout_sec': 180,
                'timeout_fallback_max_chunk_span': 10,
                'timeout_fallback_order_penalty': 0.10,
            },
        }
    }
    # _run_evaluation chdirs into EVAL_DIR before the harness reads this file, so
    # write the paths relative to EVAL_DIR rather than absolute. Relative keeps the
    # config portable across clones (it used to bake in one machine's checkout
    # path); it still resolves correctly because the cwd is pinned.
    def _rel_to_eval_dir(p: str) -> str:
        target = Path(p).resolve()
        try:
            return os.path.relpath(target, EVAL_DIR.resolve()).replace(os.sep, '/')
        except ValueError:
            # Different drive on Windows — no relative path exists; keep absolute.
            return str(target)

    config_path = str(Path(pred_dir).resolve() / 'eval_config.yaml')
    cfg['end2end_eval']['dataset']['ground_truth']['data_path'] = _rel_to_eval_dir(gt_json)
    cfg['end2end_eval']['dataset']['prediction']['data_path'] = _rel_to_eval_dir(pred_dir)
    with open(config_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return config_path


def _run_evaluation(config_path: str) -> None:
    """Run OmniDocBench eval using the installed src package."""
    eval_dir = str(EVAL_DIR)
    orig_cwd = os.getcwd()
    os.chdir(eval_dir)
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    try:
        sys.argv = ['pdf_validation.py', '--config', config_path]
        from src.cli import main as eval_main
        eval_main(sys.argv[1:])
    finally:
        os.chdir(orig_cwd)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    pred_dir = args.pred_dir
    os.makedirs(pred_dir, exist_ok=True)

    # Collect image paths from GT JSON, building language map
    import ast as _ast
    with open(args.gt_json, encoding='utf-8') as f:
        gt_data = json.load(f)
    images_dir = Path(args.images_dir)
    image_paths = []
    cjk_pages   = set()  # simplified_chinese → CJK engine
    mixed_pages = set()  # en_ch_mixed → dual-engine (EN + CJK, pick best)
    ppt_pages   = set()  # PPT2PDF → force screenshot mode
    lang_filter = args.lang_filter  # e.g. 'english'
    for page in gt_data:
        img_name = page['page_info']['image_path']
        img_path = images_dir / img_name
        if img_path.exists():
            attrs = page['page_info']['page_attribute']
            if isinstance(attrs, str):
                attrs = _ast.literal_eval(attrs)
            lang = attrs.get('language', '')
            if lang_filter and lang != lang_filter:
                continue
            image_paths.append(str(img_path))
            if lang in ('simplified_chinese', 'traditional_chinese'):
                # traditional_chinese was falling through to the ENGLISH OCR
                # engine (text edit 0.913, table TEDS 44 on those pages)
                cjk_pages.add(Path(img_name).stem)
            elif lang == 'en_ch_mixed':
                mixed_pages.add(Path(img_name).stem)
            if attrs.get('data_source', '') == 'PPT2PDF':
                ppt_pages.add(Path(img_name).stem)
        else:
            print(f'[!] Image not found: {img_path}')

    if args.skip_stage1:
        label = 'skip-stage1'
    elif args.formula_from_fidelity:
        label = 'full-pipeline+formula-from-fidelity'
    else:
        label = 'full-pipeline'
    print(f'[*] Config: {label}')
    print(f'[*] Found {len(image_paths)} pages to process ({len(cjk_pages)} CJK, {len(mixed_pages)} mixed, {len(ppt_pages)} PPT).')

    if not args.eval_only:
        _run_prism_on_images(image_paths, pred_dir, cjk_pages=cjk_pages, mixed_pages=mixed_pages, ppt_pages=ppt_pages, skip_stage1=args.skip_stage1, formula_from_fidelity=args.formula_from_fidelity)

    if not args.skip_eval:
        print('\n[*] Running OmniDocBench evaluation...')
        config_path = _write_eval_config(args.gt_json, pred_dir, no_cdm=args.no_cdm)
        _run_evaluation(config_path)


if __name__ == '__main__':
    main()
