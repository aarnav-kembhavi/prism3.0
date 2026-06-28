"""
Hyperparameter sweep for formula recognition.

Tests rep_penalty and FORMULA_PAD on GT formula crops from OmniDocBench (EN).
Uses GT bboxes to crop from page images, so FORMULA_PAD has real context to work with.

Usage:
    python scripts/sweep_formula_hypers.py
"""

import json
import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps
from Levenshtein import distance as edit_distance

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'Texo' / 'src'))

from pipeline.math_worker_onnx import (
    _preprocess_to_tensor, _sanitize, _quality_gate, _onnx_generate
)

# ── load GT ──────────────────────────────────────────────────────────────────

GT_JSON   = ROOT / 'data' / 'omnidocbench' / 'OmniDocBench_available.json'
IMAGES_DIR = ROOT / 'data' / 'omnidocbench' / 'images'

def load_en_formulas():
    data = json.loads(GT_JSON.read_bytes().decode('utf-8'))
    samples = []
    for page in data:
        pi = page['page_info']
        if pi.get('page_attribute', {}).get('language') != 'english':
            continue
        img_path = IMAGES_DIR / pi['image_path']
        if not img_path.exists():
            continue
        pw, ph = pi['width'], pi['height']
        for det in page['layout_dets']:
            if det.get('category_type') != 'equation_isolated':
                continue
            if det.get('ignore'):
                continue
            latex = det.get('latex', '').strip()
            if not latex:
                continue
            poly = det['poly']  # [x1,y1, x2,y2, x3,y3, x4,y4]
            x_vals = poly[0::2]
            y_vals = poly[1::2]
            x1, y1 = min(x_vals), min(y_vals)
            x2, y2 = max(x_vals), max(y_vals)
            samples.append({
                'img_path': img_path,
                'bbox': [x1, y1, x2, y2],
                'gt': latex,
                'pw': pw, 'ph': ph,
            })
    return samples


def strip_delims(text: str) -> str:
    for d in ('$$', '$', r'\[', r'\]', r'\(', r'\)'):
        if text.startswith(d): text = text[len(d):]
        if text.endswith(d):   text = text[:-len(d)]
    return text.strip()


def norm_latex(text: str) -> str:
    """Minimal normalization: strip outer delimiters and extra whitespace."""
    text = strip_delims(text)
    return ' '.join(text.split())


def edr(pred: str, gt: str) -> float:
    p, g = norm_latex(pred), norm_latex(gt)
    if not g:
        return 0.0
    if not p:
        return 1.0
    max_len = max(len(p), len(g))
    return edit_distance(p, g) / max_len


# ── inference ────────────────────────────────────────────────────────────────

def load_model():
    import onnxruntime as ort
    from tokenizers import Tokenizer

    ort.set_default_logger_severity(3)
    MODEL_DIR = ROOT / 'Texo' / 'model'
    ONNX_DIR  = MODEL_DIR / 'onnx'

    tok = Tokenizer.from_file(str(MODEL_DIR / 'tokenizer.json'))
    opts = ort.SessionOptions()
    opts.enable_cpu_mem_arena = False
    enc = ort.InferenceSession(str(ONNX_DIR / 'encoder_model.onnx'),
                                sess_options=opts, providers=['CPUExecutionProvider'])
    dec = ort.InferenceSession(str(ONNX_DIR / 'decoder_model_merged.onnx'),
                                sess_options=opts, providers=['CPUExecutionProvider'])
    return enc, dec, tok


def run_crop(enc, dec, tok, crop: Image.Image, rep_penalty: float) -> str:
    pixel = _preprocess_to_tensor(crop)
    ids = _onnx_generate(enc, dec, pixel, tok, max_new_tokens=384,
                          rep_penalty=rep_penalty)
    raw = tok.decode(ids).strip()
    text = _sanitize(strip_delims(raw))
    if text and not _quality_gate(text, crop.width, crop.height):
        text = ''
    return text


def eval_samples(samples, enc, dec, tok, pad: int, rep_penalty: float) -> dict:
    edrs = []
    empty = 0
    for s in samples:
        img = Image.open(s['img_path']).convert('RGB')
        x1, y1, x2, y2 = s['bbox']
        pw, ph = s['pw'], s['ph']
        # Scale bbox to actual image size
        iw, ih = img.size
        sx, sy = iw / pw, ih / ph
        x1, x2 = x1 * sx, x2 * sx
        y1, y2 = y1 * sy, y2 * sy
        # Apply pad
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(iw, x2 + pad)
        y2 = min(ih, y2 + pad)
        crop = img.crop((x1, y1, x2, y2))
        pred = run_crop(enc, dec, tok, crop, rep_penalty)
        e = edr(pred, s['gt'])
        edrs.append(e)
        if not pred:
            empty += 1
    mean_edr = float(np.mean(edrs))
    return {
        'edr': mean_edr,
        'acc': 1 - mean_edr,
        'empty': empty,
        'n': len(edrs),
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print('Loading GT formula samples...')
    samples = load_en_formulas()
    print(f'  EN formula samples: {len(samples)}')

    print('Loading Texo ONNX model...')
    enc, dec, tok = load_model()
    print('  Ready.\n')

    # Baseline values
    BASE_PAD = 12
    BASE_REP = 1.15

    # Sweep 1: rep_penalty (fixed pad=12)
    rep_values = [1.0, 1.15, 1.3, 1.5, 2.0]
    print('=== Sweep: rep_penalty (pad=12) ===')
    rep_results = {}
    for rep in rep_values:
        r = eval_samples(samples, enc, dec, tok, pad=BASE_PAD, rep_penalty=rep)
        rep_results[rep] = r
        tag = ' <-- current' if rep == BASE_REP else ''
        print(f'  rep={rep:.2f}  acc={r["acc"]*100:.1f}%  edr={r["edr"]:.4f}  '
              f'empty={r["empty"]}/{r["n"]}{tag}')

    print()

    # Sweep 2: FORMULA_PAD (fixed rep=1.15)
    pad_values = [4, 8, 12, 16, 20, 28]
    print('=== Sweep: FORMULA_PAD (rep=1.15) ===')
    pad_results = {}
    for pad in pad_values:
        r = eval_samples(samples, enc, dec, tok, pad=pad, rep_penalty=BASE_REP)
        pad_results[pad] = r
        tag = ' <-- current' if pad == BASE_PAD else ''
        print(f'  pad={pad:2d}px  acc={r["acc"]*100:.1f}%  edr={r["edr"]:.4f}  '
              f'empty={r["empty"]}/{r["n"]}{tag}')

    print()

    # Find best combo
    best_rep = min(rep_results, key=lambda k: rep_results[k]['edr'])
    best_pad = min(pad_results, key=lambda k: pad_results[k]['edr'])
    print(f'Best rep_penalty: {best_rep}  ({rep_results[best_rep]["acc"]*100:.1f}%)')
    print(f'Best FORMULA_PAD: {best_pad}px  ({pad_results[best_pad]["acc"]*100:.1f}%)')

    # If best values differ from current, run that combo too
    if best_rep != BASE_REP or best_pad != BASE_PAD:
        print(f'\n=== Best combo: pad={best_pad}, rep={best_rep} ===')
        r = eval_samples(samples, enc, dec, tok, pad=best_pad, rep_penalty=best_rep)
        print(f'  acc={r["acc"]*100:.1f}%  edr={r["edr"]:.4f}  empty={r["empty"]}/{r["n"]}')


if __name__ == '__main__':
    main()
