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

YOLO_MODEL_PATH      = str(ROOT / 'weights' / 'yolov11n-doclaynet.onnx')
DOCLAYOUT_MODEL_PATH = str(ROOT / 'models' / 'doclayout_yolo_docstructbench_imgsz1024.onnx')

# Backend switch: raw onnxruntime detectors (no torch) by default; set
# PRISM_RAW_YOLO=0 to fall back to ultralytics for A/B comparison.
_USE_RAW_YOLO = os.environ.get('PRISM_RAW_YOLO', '1') != '0'

# DocLayout YOLO singleton — loaded once, shared across all pages
_doclayout_model = None

def _get_doclayout_model():
    global _doclayout_model
    if _doclayout_model is None:
        from ultralytics import YOLO as _YOLO
        _doclayout_model = _YOLO(DOCLAYOUT_MODEL_PATH, task='detect')
    return _doclayout_model


def _yolo_detect(norm_path, conf=0.25, iou=0.7):
    """Uniform layout detection → list of {bbox, class_name, confidence}."""
    if _USE_RAW_YOLO:
        from pipeline.models_interface import get_yolo_detector
        return get_yolo_detector(YOLO_MODEL_PATH, imgsz=640).detect(norm_path, conf=conf, iou=iou)
    from pipeline.models_interface import get_yolo_model
    r = get_yolo_model(YOLO_MODEL_PATH)(norm_path, verbose=False)[0]
    return [{'bbox': b.xyxy[0].tolist(), 'class_name': r.names[int(b.cls[0])],
             'confidence': float(b.conf[0])} for b in r.boxes]


def _doclayout_detect(norm_path, conf=0.15):
    """Uniform DocLayout detection → list of {bbox, class_name, confidence}."""
    if _USE_RAW_YOLO:
        from pipeline.models_interface import get_doclayout_detector
        return get_doclayout_detector(DOCLAYOUT_MODEL_PATH, imgsz=1024).detect(norm_path, conf=conf)
    r = _get_doclayout_model()(norm_path, conf=conf, verbose=False)[0]
    return [{'bbox': b.xyxy[0].tolist(), 'class_name': r.names[int(b.cls[0])],
             'confidence': float(b.conf[0])} for b in r.boxes]


def _iou(a, b):
    ix1=max(a[0],b[0]); iy1=max(a[1],b[1]); ix2=min(a[2],b[2]); iy2=min(a[3],b[3])
    iw=max(0,ix2-ix1); ih=max(0,iy2-iy1); inter=iw*ih
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0.0


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
    from pipeline.text_worker import TextOCRWorkerDual
    from pipeline.math_worker_onnx import MathOCRWorkerOnnxDual
    from pipeline.tatr_worker_onnx import TATROnnxWorker
    from pipeline.models_interface import unload_yolo
    from pipeline.page_core import Workers, build_document, MATH_CLASSES, IMAGE_CLASSES
    from pipeline.tex_to_md import tex_to_omnidocbench_md

    ocr_worker = TextOCRWorkerDual()
    math_worker = MathOCRWorkerOnnxDual()
    tatr_worker = TATROnnxWorker()
    print('[*] Starting workers...')
    ocr_worker.start()
    math_worker.start()
    tatr_worker.start()
    print('[*] Workers ready.')

    results: dict[str, str] = {}

    for img_path_str in image_paths:
        img_path = Path(img_path_str)
        stem = img_path.stem
        t0 = time.perf_counter()
        print(f'[>] {stem}')

        try:
            work_dir = Path(pred_dir) / f'_tmp_{stem}'
            work_dir.mkdir(parents=True, exist_ok=True)
            assets_dir = work_dir / 'assets'
            figures_dir = assets_dir / 'figures'
            figures_dir.mkdir(parents=True, exist_ok=True)

            # Stage 1: normalise (or bypass for ablation)
            image_norm, image_fidelity, modality_result = _normalise_fn(img_path_str)
            is_screenshot = (modality_result.modality == CaptureModality.SCREENSHOT) or (stem in (ppt_pages or set()))
            norm_path = str(assets_dir / 'normalized.png')
            image_norm.save(norm_path)

            # Stage 2: YOLO detection (raw onnxruntime by default; no torch)
            img_width, img_height = image_norm.width, image_norm.height
            detections = []
            for d in _yolo_detect(norm_path, conf=0.25, iou=0.7):
                x1, y1, x2, y2 = d['bbox']
                class_name = d['class_name']
                if class_name in IMAGE_CLASSES or (formula_from_fidelity and class_name in MATH_CLASSES):
                    crop = xyxy_to_pil_crop(image_fidelity, [x1, y1, x2, y2])
                else:
                    crop = xyxy_to_pil_crop(image_norm, [x1, y1, x2, y2])
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'class_id': 0,
                    'class_name': class_name,
                    'confidence': d['confidence'],
                    'crop': crop,
                })

            detections = postprocess_detections(detections, img_width, img_height)

            # DocLayout YOLO boost: supplement nano YOLO for formulas AND tables.
            # conf=0.15 for broader formula recall; tables use 0.30 (higher precision needed).
            try:
                existing_fml = [d['bbox'] for d in detections if d['class_name'] == 'Formula']
                existing_tbl = [d['bbox'] for d in detections if d['class_name'] == 'Table']
                n_fml = n_tbl = 0
                for box in _doclayout_detect(norm_path, conf=0.15):
                        cls = box['class_name']
                        conf_ = box['confidence']
                        bbox = box['bbox']
                        if cls == 'isolate_formula':
                            # 0.20 floor: the raw 0.15 recall level adds very
                            # low-confidence formula boxes that overlap text and
                            # feed Texo garbage the quality gate must discard.
                            if conf_ < 0.20:
                                continue
                            if any(_iou(bbox, ef) > 0.4 for ef in existing_fml):
                                continue
                            crop = xyxy_to_pil_crop(image_fidelity if formula_from_fidelity else image_norm, bbox)
                            detections.append({
                                'bbox': bbox, 'class_id': -2,
                                'class_name': 'Formula', 'confidence': conf_, 'crop': crop,
                            })
                            existing_fml.append(bbox); n_fml += 1
                        elif cls == 'table' and conf_ >= 0.30:
                            if any(_iou(bbox, et) > 0.4 for et in existing_tbl):
                                continue
                            crop = xyxy_to_pil_crop(image_norm, bbox)
                            detections.append({
                                'bbox': bbox, 'class_id': -3,
                                'class_name': 'Table', 'confidence': conf_, 'crop': crop,
                            })
                            existing_tbl.append(bbox); n_tbl += 1
                if n_fml or n_tbl:
                    print(f'  [DL] +{n_fml} formula(s), +{n_tbl} table(s)')
            except Exception as _e:
                print(f'  [DL] skipped: {_e}')

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

            del image_norm, image_fidelity
            gc.collect()

            # Stage 3: extraction + assembly (shared with orchestrate.py)
            is_cjk  = stem in (cjk_pages  or set())
            is_mixed = stem in (mixed_pages or set())
            workers = Workers(ocr=ocr_worker, math=math_worker, tatr=tatr_worker)
            document = build_document(
                detections, img_width, img_height, workers, str(figures_dir),
                is_screenshot=is_screenshot, is_cjk=is_cjk, is_mixed=is_mixed,
            )

            # Convert LaTeX → Markdown
            md_text = tex_to_omnidocbench_md(document)
            results[stem] = md_text

            # Save .tex and .md for inspection
            tex_path = work_dir / 'main.tex'
            save_tex(document, str(tex_path))
            md_path = Path(pred_dir) / f'{stem}.md'
            md_path.write_text(md_text, encoding='utf-8')

            elapsed = time.perf_counter() - t0
            print(f'    done in {elapsed:.1f}s → {md_path.name}')

        except Exception as e:
            import traceback
            print(f'    ERROR: {e}')
            traceback.print_exc()
            results[stem] = ''
            md_path = Path(pred_dir) / f'{stem}.md'
            md_path.write_text('', encoding='utf-8')

    ocr_worker.stop()
    math_worker.stop()
    tatr_worker.stop()
    unload_yolo()
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
                    'teds_workers': 2,
                },
                'reading_order': {'metric': ['Edit_dist']},
            },
            'dataset': {
                'dataset_name': 'end2end_dataset',
                'ground_truth': {'data_path': gt_json},
                'prediction': {'data_path': pred_dir},
                'match_method': 'quick_match',
                'match_workers': 4,
                'quick_match_truncated_timeout_sec': 120,
                'match_timeout_sec': 180,
                'timeout_fallback_max_chunk_span': 10,
                'timeout_fallback_order_penalty': 0.10,
            },
        }
    }
    # Use absolute paths so _run_evaluation's os.chdir() doesn't break resolution
    config_path = str(Path(pred_dir).resolve() / 'eval_config.yaml')
    cfg['end2end_eval']['dataset']['ground_truth']['data_path'] = str(Path(gt_json).resolve())
    cfg['end2end_eval']['dataset']['prediction']['data_path'] = str(Path(pred_dir).resolve())
    with open(config_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return config_path


def _run_evaluation(config_path: str) -> None:
    """Run OmniDocBench eval using the installed src package."""
    eval_dir = str(EVAL_DIR)
    orig_cwd = os.getcwd()
    os.chdir(eval_dir)
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
            if lang == 'simplified_chinese':
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
