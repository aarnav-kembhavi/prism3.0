"""
Extended hyperparameter sweep for formula recognition.

Tests both Texo-transfer and Texo-distill models across:
  - FORMULA_PAD: [2, 4, 6, 8, 12]
  - rep_penalty: [1.0, 1.15, 1.3, 1.5]
  - max_new_tokens: [192, 256, 384]
  - tilde_threshold: [6, 8, 10]
  - overgen_multiplier (quality gate): [5, 10, 20]

Uses GT bboxes from OmniDocBench EN pages to crop with real surrounding context.

Usage:
    python scripts/sweep_formula_hypers2.py
    python scripts/sweep_formula_hypers2.py --model transfer   # swap model first
"""

import argparse
import json
import re
import sys
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps
from Levenshtein import distance as edit_distance

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'Texo' / 'src'))

import onnxruntime as ort
from tokenizers import Tokenizer

# ── GT loading ────────────────────────────────────────────────────────────────

GT_JSON    = ROOT / 'data' / 'omnidocbench' / 'OmniDocBench_available.json'
IMAGES_DIR = ROOT / 'data' / 'omnidocbench' / 'images'
MODEL_DIR  = ROOT / 'Texo' / 'model'
ONNX_DIR   = MODEL_DIR / 'onnx'
DATA_DIR   = ROOT / 'Texo' / 'data'


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
            if det.get('category_type') != 'equation_isolated' or det.get('ignore'):
                continue
            latex = det.get('latex', '').strip()
            if not latex:
                continue
            poly = det['poly']
            x1, y1 = min(poly[0::2]), min(poly[1::2])
            x2, y2 = max(poly[0::2]), max(poly[1::2])
            samples.append({'img_path': img_path, 'bbox': [x1, y1, x2, y2],
                             'gt': latex, 'pw': pw, 'ph': ph})
    return samples


# ── model swap helpers ────────────────────────────────────────────────────────

def install_transfer():
    """Swap in Texo-transfer model files (vocab=687)."""
    import shutil
    # Texo/data/tokenizer/ is the original WordLevel vocab=687 transfer tokenizer
    transfer_tok = DATA_DIR / 'tokenizer'
    if not transfer_tok.exists():
        raise FileNotFoundError(f'Transfer tokenizer not found: {transfer_tok}')
    for fname in ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']:
        src = transfer_tok / fname
        if src.exists():
            shutil.copy(str(src), str(MODEL_DIR / fname))
    # Update config vocab_size
    cfg = json.loads((MODEL_DIR / 'config.json').read_bytes().decode('utf-8'))
    cfg['decoder']['vocab_size'] = 687
    (MODEL_DIR / 'config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    print(f'[swap] Installed transfer tokenizer (vocab=687)')


def install_distill():
    """Swap in Texo-distill model files (vocab=1264)."""
    import shutil
    distill_tok = DATA_DIR / 'unimernet_tokenizer_distill'
    if not distill_tok.exists():
        raise FileNotFoundError(f'Distill tokenizer not found: {distill_tok}')
    for fname in ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']:
        src = distill_tok / fname
        if src.exists():
            shutil.copy(str(src), str(MODEL_DIR / fname))
    cfg = json.loads((MODEL_DIR / 'config.json').read_bytes().decode('utf-8'))
    cfg['decoder']['vocab_size'] = 1264
    (MODEL_DIR / 'config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    print(f'[swap] Installed distill tokenizer (vocab=1264)')


def get_vocab_size():
    tok = Tokenizer.from_file(str(MODEL_DIR / 'tokenizer.json'))
    return tok.get_vocab_size()


# ── inference primitives ──────────────────────────────────────────────────────

import cv2

_UNIMERNET_MEAN = 0.7931
_UNIMERNET_STD  = 0.1738
_IMG_SIZE       = 384

_REPEAT_TILDE_RE = re.compile(r'(~\s*){10,}')
_REPEAT_NEG_RE   = re.compile(r'(\\![\s\\!]*){10,}')
_REPEAT_QQUAD_RE = re.compile(r'(\\q{0,1}quad\s*){5,}')
_ARROW_SUBS = [
    (re.compile(r'\\[Rr]arr\b'), r'\\rightarrow'),
    (re.compile(r'\\[Ll]arr\b'), r'\\leftarrow'),
]
_BAD_PATTERNS = [
    re.compile(r'(\\hline\s*){5,}'),
    re.compile(r'(\\cdot\s*){8,}'),
    re.compile(r'(\\quad\s*){8,}'),
    re.compile(r'(&\s*\{\s*\}\s*){5,}'),
    re.compile(r'\\mathrm\s*\{[^}]*(~\s*){5,}'),
]


def _crop_margin(img):
    data = np.array(img.convert('L'), dtype=np.uint8)
    mx, mn = int(data.max()), int(data.min())
    if mx == mn:
        return img
    norm = (data.astype(np.float32) - mn) / (mx - mn) * 255
    gray = (255 * (norm < 200)).astype(np.uint8)
    coords = cv2.findNonZero(gray)
    if coords is None:
        return img
    a, b, w, h = cv2.boundingRect(coords)
    return img.crop((a, b, w + a, h + b))


def _preprocess(crop):
    arr = np.array(crop.convert('L'), dtype=np.uint8)
    _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img = Image.fromarray(binary).convert('RGB')
    img = _crop_margin(img)
    w, h = img.size
    if w > 0 and h > 0:
        scale = _IMG_SIZE / max(w, h)
        img = img.resize((max(1, round(w*scale)), max(1, round(h*scale))), Image.LANCZOS)
    dw, dh = _IMG_SIZE - img.width, _IMG_SIZE - img.height
    img = ImageOps.expand(img, (dw//2, dh//2, dw - dw//2, dh - dh//2))
    gray = np.array(img.convert('L'), dtype=np.float32) / 255.0
    gray = (gray - _UNIMERNET_MEAN) / _UNIMERNET_STD
    return np.stack([gray]*3)[np.newaxis].astype(np.float32)


def _generate(enc, dec, tok, pixel, max_new_tokens=384, rep_penalty=1.15):
    LAYERS, HEADS, HEAD_DIM = 2, 16, 24
    BOS = tok.token_to_id('<s>')
    EOS = tok.token_to_id('</s>')
    (enc_h,) = enc.run(['last_hidden_state'], {'pixel_values': pixel})
    empty = np.zeros((1, HEADS, 0, HEAD_DIM), dtype=np.float32)
    pdk = [empty]*LAYERS; pdv = [empty]*LAYERS
    pek = [empty]*LAYERS; pev = [empty]*LAYERS
    inp = np.array([[BOS]], dtype=np.int64)
    generated = []
    for step in range(max_new_tokens):
        use_cache = step > 0
        feed = {'input_ids': inp, 'encoder_hidden_states': enc_h,
                'use_cache_branch': np.array([use_cache])}
        for l in range(LAYERS):
            feed[f'past_key_values.{l}.decoder.key']   = pdk[l]
            feed[f'past_key_values.{l}.decoder.value'] = pdv[l]
            feed[f'past_key_values.{l}.encoder.key']   = pek[l]
            feed[f'past_key_values.{l}.encoder.value'] = pev[l]
        out = dec.run(None, feed)
        logits = out[0][0, -1].copy()
        if rep_penalty != 1.0 and generated:
            for t in set(generated):
                logits[t] = logits[t] / rep_penalty if logits[t] > 0 else logits[t] * rep_penalty
        next_tok = int(np.argmax(logits))
        pdk = [out[1], out[5]]; pdv = [out[2], out[6]]
        if step == 0:
            pek = [out[3], out[7]]; pev = [out[4], out[8]]
        inp = np.array([[next_tok]], dtype=np.int64)
        if next_tok == EOS:
            break
        generated.append(next_tok)
    return generated


def _sanitize(text):
    if not text:
        return text
    for pat, rep in _ARROW_SUBS:
        text = pat.sub(rep, text)
    text = _REPEAT_TILDE_RE.sub('~ ', text)
    text = _REPEAT_NEG_RE.sub(r'\\! ', text)
    text = _REPEAT_QQUAD_RE.sub(r'\\qquad ', text)
    oc, cc = text.count('{'), text.count('}')
    if oc > cc:
        text += '}' * (oc - cc)
    elif cc > oc:
        for _ in range(cc - oc):
            idx = text.rfind('}')
            if idx != -1:
                text = text[:idx] + text[idx+1:]
    return text.strip()


def _quality_gate(text, w, h, tilde_thresh=10, overgen_mult=10):
    if not text:
        return False
    for pat in _BAD_PATTERNS:
        if pat.search(text):
            return False
    if text.count('~') >= tilde_thresh:
        return False
    if len(text) > overgen_mult * max(80, w * h / 50):
        return False
    return True


def strip_delims(text):
    for d in ('$$', '$', r'\[', r'\]', r'\(', r'\)'):
        if text.startswith(d): text = text[len(d):]
        if text.endswith(d):   text = text[:-len(d)]
    return text.strip()


def norm_latex(text):
    return ' '.join(strip_delims(text).split())


def edr(pred, gt):
    p, g = norm_latex(pred), norm_latex(gt)
    if not g:
        return 0.0
    if not p:
        return 1.0
    return edit_distance(p, g) / max(len(p), len(g))


# ── evaluation ────────────────────────────────────────────────────────────────

def load_sessions(onnx_dir=None):
    """Load ONNX sessions and tokenizer. onnx_dir overrides default ONNX_DIR;
    tokenizer.json is read from onnx_dir (so each model dir is self-contained)."""
    ort.set_default_logger_severity(3)
    opts = ort.SessionOptions()
    opts.enable_cpu_mem_arena = False
    odir = Path(onnx_dir) if onnx_dir else ONNX_DIR
    tok = Tokenizer.from_file(str(odir / 'tokenizer.json'))
    enc = ort.InferenceSession(str(odir / 'encoder_model.onnx'),
                                sess_options=opts, providers=['CPUExecutionProvider'])
    dec = ort.InferenceSession(str(odir / 'decoder_model_merged.onnx'),
                                sess_options=opts, providers=['CPUExecutionProvider'])
    return enc, dec, tok


def eval_all(samples, enc, dec, tok,
             pad=4, rep_penalty=1.15, max_new_tokens=384,
             tilde_thresh=10, overgen_mult=10):
    edrs, empty = [], 0
    for s in samples:
        img = Image.open(s['img_path']).convert('RGB')
        iw, ih = img.size
        sx, sy = iw / s['pw'], ih / s['ph']
        x1, y1, x2, y2 = s['bbox']
        x1 = max(0, x1*sx - pad); y1 = max(0, y1*sy - pad)
        x2 = min(iw, x2*sx + pad); y2 = min(ih, y2*sy + pad)
        crop = img.crop((x1, y1, x2, y2))
        pixel = _preprocess(crop)
        ids = _generate(enc, dec, tok, pixel, max_new_tokens=max_new_tokens,
                        rep_penalty=rep_penalty)
        raw = tok.decode(ids).strip()
        text = _sanitize(strip_delims(raw))
        if text and not _quality_gate(text, crop.width, crop.height,
                                       tilde_thresh=tilde_thresh,
                                       overgen_mult=overgen_mult):
            text = ''
        edrs.append(edr(text, s['gt']))
        if not text:
            empty += 1
    return {'acc': (1 - np.mean(edrs)) * 100, 'edr': float(np.mean(edrs)),
            'empty': empty, 'n': len(edrs)}


def sweep(label, samples, enc, dec, tok, base):
    """Run sweeps around the given base config, print results."""
    def run(desc, **kwargs):
        cfg = {**base, **kwargs}
        r = eval_all(samples, enc, dec, tok, **cfg)
        tag = ' <-- base' if kwargs == {} else ''
        print(f'  {desc:<40} acc={r["acc"]:5.1f}%  edr={r["edr"]:.4f}  empty={r["empty"]}/{r["n"]}{tag}')
        return r

    print(f'\n=== {label} ===')
    print(f'    Base: {base}')

    print('\n  -- FORMULA_PAD --')
    for pad in [2, 4, 6, 8, 12]:
        run(f'pad={pad}px', pad=pad)

    print('\n  -- rep_penalty --')
    for rep in [1.0, 1.15, 1.3, 1.5]:
        run(f'rep={rep}', rep_penalty=rep)

    print('\n  -- max_new_tokens --')
    for mnt in [192, 256, 384]:
        run(f'max_new_tokens={mnt}', max_new_tokens=mnt)

    print('\n  -- tilde_threshold (quality gate) --')
    for tt in [5, 6, 8, 10]:
        run(f'tilde_thresh={tt}', tilde_thresh=tt)

    print('\n  -- overgen_multiplier (quality gate) --')
    for om in [5, 10, 20]:
        run(f'overgen_mult={om}', overgen_mult=om)

    print('\n  -- best combo search --')
    best = base.copy()
    # Try best individual values together
    combos = [
        ('pad=4, rep=1.15, mnt=256, tt=8',
         dict(pad=4, rep_penalty=1.15, max_new_tokens=256, tilde_thresh=8)),
        ('pad=4, rep=1.15, mnt=384, tt=8',
         dict(pad=4, rep_penalty=1.15, max_new_tokens=384, tilde_thresh=8)),
        ('pad=2, rep=1.15, mnt=256, tt=8',
         dict(pad=2, rep_penalty=1.15, max_new_tokens=256, tilde_thresh=8)),
        ('pad=4, rep=1.15, mnt=256, tt=6',
         dict(pad=4, rep_penalty=1.15, max_new_tokens=256, tilde_thresh=6)),
    ]
    for desc, kwargs in combos:
        run(desc, **kwargs)


# ── main ─────────────────────────────────────────────────────────────────────

TRANSFER_ONNX_DIR = MODEL_DIR / 'onnx_transfer'
DISTILL_ONNX_DIR  = MODEL_DIR / 'onnx'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['distill', 'transfer', 'both'],
                        default='both')
    parser.add_argument('--onnx-dir', default=None,
                        help='Override ONNX dir (tokenizer.json must be in same dir). '
                             'Skips install_* file swapping — use for parallel runs.')
    args = parser.parse_args()

    print('Loading GT formula samples...')
    samples = load_en_formulas()
    print(f'  {len(samples)} EN formula samples\n')

    base = dict(pad=4, rep_penalty=1.15, max_new_tokens=384,
                tilde_thresh=10, overgen_mult=10)

    if args.onnx_dir:
        # Direct path mode: no file swapping, fully parallel-safe
        odir = Path(args.onnx_dir)
        enc, dec, tok = load_sessions(odir)
        label = f'MODEL ({odir.name}, vocab={tok.get_vocab_size()})'
        print(f'  Using onnx-dir: {odir}  vocab={tok.get_vocab_size()}')
        sweep(label, samples, enc, dec, tok, base)
        return

    if args.model in ('distill', 'both'):
        enc, dec, tok = load_sessions(DISTILL_ONNX_DIR)
        print(f'  Model vocab: {tok.get_vocab_size()} (distill)')
        sweep('DISTILL MODEL', samples, enc, dec, tok, base)

    if args.model in ('transfer', 'both'):
        if not TRANSFER_ONNX_DIR.exists():
            raise FileNotFoundError(
                f'Transfer ONNX not found at {TRANSFER_ONNX_DIR}. '
                'Download with: python -c "from huggingface_hub import hf_hub_download; ..."')
        enc, dec, tok = load_sessions(TRANSFER_ONNX_DIR)
        print(f'  Model vocab: {tok.get_vocab_size()} (transfer)')
        sweep('TRANSFER MODEL', samples, enc, dec, tok, base)

    if args.model == 'both':
        print('\n\n=== HEAD-TO-HEAD at best known config (pad=4, rep=1.15) ===')
        for name, odir in [('distill', DISTILL_ONNX_DIR), ('transfer', TRANSFER_ONNX_DIR)]:
            enc, dec, tok = load_sessions(odir)
            r = eval_all(samples, enc, dec, tok, pad=4, rep_penalty=1.15)
            print(f'  {name:<10}  acc={r["acc"]:5.1f}%  edr={r["edr"]:.4f}  empty={r["empty"]}/{r["n"]}')


if __name__ == '__main__':
    main()
