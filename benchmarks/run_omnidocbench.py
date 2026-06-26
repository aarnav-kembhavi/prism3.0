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

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
EVAL_DIR = ROOT / 'omnidocbench_eval'
DEMO_IMAGES = EVAL_DIR / 'demo_data' / 'omnidocbench_demo' / 'images'
DEMO_GT = EVAL_DIR / 'demo_data' / 'omnidocbench_demo' / 'OmniDocBench_demo.json'
DEFAULT_PRED = ROOT / 'preds' / 'omnidocbench'

YOLO_MODEL_PATH      = str(ROOT / 'weights' / 'yolov11n-doclaynet.onnx')
DOCLAYOUT_MODEL_PATH = str(ROOT / 'models' / 'doclayout_yolo_docstructbench_imgsz1024.onnx')

# DocLayout YOLO singleton — loaded once, shared across all pages
_doclayout_model = None

def _get_doclayout_model():
    global _doclayout_model
    if _doclayout_model is None:
        from ultralytics import YOLO as _YOLO
        _doclayout_model = _YOLO(DOCLAYOUT_MODEL_PATH, task='detect')
    return _doclayout_model


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
    return p.parse_args()


# ── PRISM pipeline (adapted from orchestrate.py) ──────────────────────────────

def _run_prism_on_images(image_paths: list[str], pred_dir: str, cjk_pages: set = None, mixed_pages: set = None, ppt_pages: set = None) -> dict[str, str]:
    """
    Run PRISM on a list of image files.  Workers are shared across all images.
    Returns {image_stem: markdown_text} dict.
    """
    import gc
    import numpy as np
    from PIL import Image
    from concurrent.futures import ThreadPoolExecutor

    sys.path.insert(0, str(ROOT))
    from normalization import normalize_image_pil
    from normalization.modality import CaptureModality
    from pipeline.models_interface import get_yolo_model, unload_yolo
    from pipeline.layout_utils import (
        apply_semantic_reading_order, sort_detections_geometric,
        xyxy_to_pil_crop, detect_column_count, split_detections_by_column,
        split_detections_n_columns,
    )
    from pipeline.latex_builder import wrap_content, assemble_document, save_tex
    from pipeline.detection_postprocess import postprocess_detections
    from pipeline.text_worker import TextOCRWorkerDual
    from pipeline.math_worker_onnx import MathOCRWorkerOnnxDual
    from pipeline.tex_to_md import tex_to_omnidocbench_md

    TEXT_CLASSES  = {"Text", "Title", "Section-header", "Caption",
                     "Footnote", "Page-footer", "Page-header", "List-item"}
    MATH_CLASSES  = {"Formula"}
    TABLE_CLASSES = {"Table"}
    IMAGE_CLASSES = {"Picture"}
    LIST_ITEM_CLASS = "List-item"

    ocr_worker = TextOCRWorkerDual()
    math_worker = MathOCRWorkerOnnxDual()
    print('[*] Starting workers...')
    ocr_worker.start()
    math_worker.start()
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

            # Stage 1: normalise
            image_norm, image_fidelity, modality_result = normalize_image_pil(img_path_str)
            is_screenshot = (modality_result.modality == CaptureModality.SCREENSHOT) or (stem in (ppt_pages or set()))
            norm_path = str(assets_dir / 'normalized.png')
            image_norm.save(norm_path)

            # Stage 2: YOLO detection
            model = get_yolo_model(YOLO_MODEL_PATH)
            results_yolo = model(norm_path, verbose=False)
            detections = []
            result_yolo = results_yolo[0]
            class_names = result_yolo.names
            img_width, img_height = image_norm.width, image_norm.height

            for box in result_yolo.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                class_name = class_names[class_id]
                if class_name in IMAGE_CLASSES:
                    crop = xyxy_to_pil_crop(image_fidelity, [x1, y1, x2, y2])
                else:
                    crop = xyxy_to_pil_crop(image_norm, [x1, y1, x2, y2])
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence,
                    'crop': crop,
                })

            detections = postprocess_detections(detections, img_width, img_height)

            # DocLayout YOLO formula boost: run a second model specialized for
            # isolate_formula to recover formulas the nano YOLO misses.
            try:
                dl_model = _get_doclayout_model()
                dl_results = dl_model(norm_path, conf=0.25, verbose=False)
                existing_fml = [d['bbox'] for d in detections if d['class_name'] == 'Formula']
                n_added = 0
                for r in dl_results:
                    for box in r.boxes:
                        if r.names[int(box.cls[0])] != 'isolate_formula':
                            continue
                        bbox = box.xyxy[0].tolist()
                        # skip if heavily overlapping an already-detected formula
                        if any(_iou(bbox, ef) > 0.4 for ef in existing_fml):
                            continue
                        crop = xyxy_to_pil_crop(image_norm, bbox)
                        detections.append({
                            'bbox': bbox,
                            'class_id': -2,
                            'class_name': 'Formula',
                            'confidence': float(box.conf[0]),
                            'crop': crop,
                        })
                        existing_fml.append(bbox)
                        n_added += 1
                if n_added:
                    print(f'  [DL] added {n_added} formula(s)')
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

            # Formula pad + re-crop
            FORMULA_PAD = 12
            for det in detections:
                bbox = det['bbox']
                if det['class_name'] in MATH_CLASSES:
                    x1, y1, x2, y2 = bbox
                    bbox = [
                        max(0, x1 - FORMULA_PAD), max(0, y1 - FORMULA_PAD),
                        min(img_width, x2 + FORMULA_PAD), min(img_height, y2 + FORMULA_PAD),
                    ]
                if det['class_name'] in IMAGE_CLASSES:
                    det['crop'] = xyxy_to_pil_crop(image_fidelity, bbox)
                else:
                    det['crop'] = xyxy_to_pil_crop(image_norm, bbox)

            del image_norm, image_fidelity
            gc.collect()

            # Stage 3: extraction
            col_count = detect_column_count(detections, img_width)
            body_detections = detections
            is_cjk  = stem in (cjk_pages  or set())
            is_mixed = stem in (mixed_pages or set())

            def _route(dets, f_start=0, m_start=0):
                text_idx  = [i for i, d in enumerate(dets) if d['class_name'] in TEXT_CLASSES]
                math_idx  = [i for i, d in enumerate(dets) if d['class_name'] in MATH_CLASSES]
                table_idx = [i for i, d in enumerate(dets) if d['class_name'] in TABLE_CLASSES]
                math_ctr  = [m_start]

                if is_mixed:
                    _run_text = ocr_worker.run_text_batch_mixed
                elif is_cjk:
                    _run_text = ocr_worker.run_text_batch_cjk
                else:
                    _run_text = ocr_worker.run_text_batch

                if math_idx and text_idx:
                    math_crops = [dets[i]['crop'] for i in math_idx]
                    text_crops = [dets[i]['crop'] for i in text_idx]
                    with ThreadPoolExecutor(max_workers=2) as exe:
                        mf = exe.submit(math_worker.run_math_batch, math_crops, str(figures_dir), math_ctr[0])
                        tf = exe.submit(_run_text, text_crops, is_screenshot)
                        math_results, math_ctr[0] = mf.result()
                        texts = tf.result()
                    for idx, raw in zip(math_idx, math_results):
                        dets[idx]['raw_content'] = raw
                    for idx, txt in zip(text_idx, texts):
                        dets[idx]['raw_content'] = txt
                else:
                    if math_idx:
                        crops = [dets[i]['crop'] for i in math_idx]
                        mres, math_ctr[0] = math_worker.run_math_batch(crops, str(figures_dir), math_ctr[0])
                        for idx, raw in zip(math_idx, mres):
                            dets[idx]['raw_content'] = raw
                    if text_idx:
                        crops = [dets[i]['crop'] for i in text_idx]
                        texts = _run_text(crops, is_screenshot)
                        for idx, txt in zip(text_idx, texts):
                            dets[idx]['raw_content'] = txt

                if table_idx:
                    table_crops = [dets[i]['crop'] for i in table_idx]
                    table_results = ocr_worker.run_table_batch(table_crops)
                    for idx, raw in zip(table_idx, table_results):
                        dets[idx]['raw_content'] = raw

                body_parts = []
                list_indices = set()
                f_ctr = f_start
                for i, det in enumerate(dets):
                    cn = det['class_name']
                    if cn in TEXT_CLASSES or cn in MATH_CLASSES:
                        raw = det.get('raw_content', '')
                        wrapped = wrap_content(cn, raw)
                        if cn == LIST_ITEM_CLASS:
                            list_indices.add(len(body_parts))
                        body_parts.append(wrapped)
                    elif cn in TABLE_CLASSES:
                        raw = det.get('raw_content', '')
                        if raw:
                            body_parts.append(wrap_content(cn, raw))
                    elif cn in IMAGE_CLASSES:
                        f_ctr += 1
                        fname = f'figure_{f_ctr:03d}.png'
                        det['crop'].save(str(figures_dir / fname))
                        body_parts.append(wrap_content('Picture', fname))
                return body_parts, list_indices, f_ctr, math_ctr[0]

            if col_count == 2:
                full_dets, left_dets, right_dets = split_detections_by_column(
                    body_detections, img_width, img_height, use_dag=True
                )
                fp, fi, fc, mc = _route(full_dets)
                lp, li, fc, mc = _route(left_dets,  f_start=fc, m_start=mc)
                rp, ri, fc, mc = _route(right_dets, f_start=fc, m_start=mc)
                document = assemble_document(fp, fi, True, lp, li, rp, ri)
            elif col_count >= 3:
                full_dets, col_lists = split_detections_n_columns(
                    body_detections, img_width, img_height, use_dag=True
                )
                all_parts: list = []
                all_list_idx: set = set()
                offset = 0
                fc, mc = 0, 0
                for group in [full_dets] + col_lists:
                    gp, gi, fc, mc = _route(group, f_start=fc, m_start=mc)
                    all_parts.extend(gp)
                    all_list_idx.update(i + offset for i in gi)
                    offset += len(gp)
                document = assemble_document(all_parts, all_list_idx, False)
            else:
                body_sorted = apply_semantic_reading_order(body_detections, img_width, img_height)
                bp, bi, _, _ = _route(body_sorted)
                document = assemble_document(bp, bi, False)

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
    with open(args.gt_json, encoding='utf-8') as f:
        gt_data = json.load(f)
    images_dir = Path(args.images_dir)
    image_paths = []
    cjk_pages   = set()  # simplified_chinese → CJK engine
    mixed_pages = set()  # en_ch_mixed → dual-engine (EN + CJK, pick best)
    ppt_pages   = set()  # PPT2PDF → force screenshot mode
    for page in gt_data:
        img_name = page['page_info']['image_path']
        img_path = images_dir / img_name
        if img_path.exists():
            image_paths.append(str(img_path))
            attrs = page['page_info']['page_attribute']
            lang = attrs.get('language', '')
            if lang == 'simplified_chinese':
                cjk_pages.add(Path(img_name).stem)
            elif lang == 'en_ch_mixed':
                mixed_pages.add(Path(img_name).stem)
            if attrs.get('data_source', '') == 'PPT2PDF':
                ppt_pages.add(Path(img_name).stem)
        else:
            print(f'[!] Image not found: {img_path}')

    print(f'[*] Found {len(image_paths)} pages to process ({len(cjk_pages)} CJK, {len(mixed_pages)} mixed, {len(ppt_pages)} PPT).')

    if not args.eval_only:
        _run_prism_on_images(image_paths, pred_dir, cjk_pages=cjk_pages, mixed_pages=mixed_pages, ppt_pages=ppt_pages)

    if not args.skip_eval:
        print('\n[*] Running OmniDocBench evaluation...')
        config_path = _write_eval_config(args.gt_json, pred_dir, no_cdm=args.no_cdm)
        _run_evaluation(config_path)


if __name__ == '__main__':
    main()
