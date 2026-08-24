# -*- coding: utf-8 -*-
"""Phase 3 — probe acceptance-threshold sweep, in-process, NO source edits.

_ACCEPT_GAIN is a module constant in normalization/verified.py (verified.py:38),
read as a module GLOBAL at call time (verified.py:151). Consumers access it via
`from normalization import verified as _nv` (pipeline.py:232) / `as nv`
(study scripts) and call `nv.verified_apply(...)`, which reads `nv._ACCEPT_GAIN`
live. Therefore setting `nv._ACCEPT_GAIN = <value>` on the imported module BEFORE
each arm patches the gate for real (no frozen-import problem — nobody does
`from normalization.verified import _ACCEPT_GAIN`). We import verified FIRST here
and verify the patch by logging the value the gate sees at runtime on the first
page of each sweep arm.

Sweep: {1.00, 1.01, 1.02, 1.05, 1.10}.
(a) 40-page synth-defect study (v2h, 5 conditions); none/open run once (threshold-
    independent), verified rerun per threshold (dedup identical outputs).
(b) 44-capture forced-stack; raw/open char counts reused (threshold-independent),
    verified regenerated per threshold; damage/prevented per 0c (>10% char loss
    vs raw, raw>200).
Paired bootstrap 95% CI over 40 pages for (verified-none) per (threshold,cond).
Raw per-page numbers saved.

Run:  venvs/gpu/Scripts/python.exe scripts/rebuttal/sweep_threshold.py [--smoke]
"""
import sys, io, os, json, time, random, hashlib, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from pathlib import Path; ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, ROOT)

# ---- import the gate module FIRST, before any consumer ----
from normalization import verified as nv
import numpy as np
import cv2
import Levenshtein
from normalization.frequency_filter import (
    remove_glare, remove_shadows, normalize_contrast, white_balance_gray_world)
from normalization.geometric import detect_and_rectify, deskew

THRESHOLDS = [1.00, 1.01, 1.02, 1.05, 1.10]
GT = os.path.join(ROOT, 'data', 'omnidocbench_full', 'OmniDocBench.json')
IMG = os.path.join(ROOT, 'data', 'omnidocbench_full', 'images')
DEFECTS = os.path.join(ROOT, 'test_images', 'real', 'defects', 'defects-images')
CAPSTATS = os.path.join(ROOT, 'scratchpad_runs', 'probe_gate', 'captures_ocr_stats_v2.json')
OUTDIR = os.path.join(ROOT, 'results', 'rebuttal')
os.makedirs(OUTDIR, exist_ok=True)

# ---- one-shot gate-value logger (verifies the monkeypatch took effect) ----
_orig_va = nv.verified_apply
_probe_first = {'tag': None, 'acc': 0, 'off': 0}
def _wrapped_va(name, fn, image_bgr, score_before=None):
    if _probe_first['tag'] is not None:
        print(f"   [gate-check] first verified_apply of arm '{_probe_first['tag']}' "
              f"sees nv._ACCEPT_GAIN = {nv._ACCEPT_GAIN}", flush=True)
        _probe_first['tag'] = None
    res = _orig_va(name, fn, image_bgr, score_before)
    # tally acceptance (res[2] True/False); count only calls that actually scored
    _probe_first['off'] += 1
    if res[2]:
        _probe_first['acc'] += 1
    return res
nv.verified_apply = _wrapped_va

# ---------- degradations (v2h harsh, verbatim from synth_defect_v2.py) ----------
def v2_shadow(img):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w * 0.95, h * 0.95
    r = np.hypot(xx - cx, yy - cy) / np.hypot(w, h)
    corner = 0.18 + 0.82 * np.clip((r - 0.15) / 0.55, 0, 1)
    lin = 0.35 + 0.65 * np.clip(xx / (w * 0.6), 0, 1)
    field = np.minimum(corner, lin)
    field = cv2.GaussianBlur(field, (0, 0), min(h, w) * 0.04)
    out = img.astype(np.float32) * field[:, :, None]
    noise = np.random.default_rng(0).normal(0, 6, img.shape).astype(np.float32)
    out = out + noise * (1.2 - field[:, :, None])
    return np.clip(out, 0, 255).astype(np.uint8)

def v2_glare(img):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w * 0.55, h * 0.3
    sx, sy = w * 0.22, h * 0.12
    blob = np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))
    out = img.astype(np.float32) * (1 - 0.65 * blob[:, :, None]) + 255.0 * blob[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)

def v2_cast(img):
    mult = np.array([0.55, 0.82, 1.0], dtype=np.float32)
    out = img.astype(np.float32) * mult[None, None, :] * 0.85
    return np.clip(out, 0, 255).astype(np.uint8)

def v2_combo(img):
    return v2_shadow(v2_cast(img))

CONDS = {'clean': lambda im: im, 'shadow': v2_shadow, 'glare': v2_glare,
         'cast': v2_cast, 'combo': v2_combo}

# ---------- correction stacks ----------
def stack_open(img):  # synth: wb->glare->shadow->clahe(if std<45), no rectify
    img = white_balance_gray_world(img)
    img = remove_glare(img)
    img = remove_shadows(img)
    if float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).std()) < 45.0:
        img = normalize_contrast(img)
    return img

def stack_verified(img):  # reads nv._ACCEPT_GAIN live via nv.verified_apply
    s = nv.probe_score(img)
    img, s, _ = nv.verified_apply('wb', white_balance_gray_world, img, s)
    img, s, _ = nv.verified_apply('glare', remove_glare, img, s)
    img, s, _ = nv.verified_apply('shadow', remove_shadows, img, s)
    if float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).std()) < 45.0:
        img, s, _ = nv.verified_apply('clahe', normalize_contrast, img, s)
    return img

def cap_stack_verified(img, path):  # captures: wb->rectify->glare->shadow->clahe
    s = nv.probe_score(img)
    v = img
    v, s, _ = nv.verified_apply('white_balance', white_balance_gray_world, v, s)
    v, s, _ = nv.verified_apply('rectify',
                                lambda im: detect_and_rectify(path, img_override=im), v, s)
    v, s, _ = nv.verified_apply('glare', remove_glare, v, s)
    v, s, _ = nv.verified_apply('shadow', remove_shadows, v, s)
    if float(cv2.cvtColor(v, cv2.COLOR_BGR2GRAY).std()) < 45.0:
        v, s, _ = nv.verified_apply('clahe', normalize_contrast, v, s)
    return v

# ---------- OCR ----------
from rapidocr_onnxruntime import RapidOCR
W = os.path.join(ROOT, 'weights')
engine = RapidOCR(det_limit_type='max', det_limit_side_len=1280, det_box_thresh=0.3,
                  det_model_path=os.path.join(W, 'PP-OCRv6_det_small.onnx'),
                  rec_model_path=os.path.join(W, 'PP-OCRv6_rec_small.onnx'))

def ocr_lines(img):
    res, _ = engine(img)
    return res or []

def ocr_text(img):
    return '\n'.join(r[1] for r in ocr_lines(img))

def _norm(s):
    return ' '.join(s.split())

def block_edits(page_img, blocks):
    out = []
    Hh, Ww = page_img.shape[:2]
    for b in blocks:
        x0, y0, x1, y1 = [int(round(c)) for c in b['bbox']]
        x0 = max(0, x0 - 2); y0 = max(0, y0 - 2)
        x1 = min(Ww, x1 + 2); y1 = min(Hh, y1 + 2)
        crop = page_img[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        pred = _norm(ocr_text(crop)); gt = _norm(b['text'])
        out.append(round(Levenshtein.distance(pred, gt) / max(len(pred), len(gt), 1), 4))
    return out

def imread(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)

def imhash(img):
    return hashlib.md5(np.ascontiguousarray(img)).hexdigest()

# ---------- seed-7 40-page sample (identical to synth_defect_v2.py) ----------
def build_pages():
    gt = json.load(open(GT, encoding='utf-8'))
    random.seed(7)
    cands = {'english': [], 'simplified_chinese': []}
    for page in gt:
        pi = page['page_info']; a = pi.get('page_attribute', {})
        lang = a.get('language')
        if lang not in cands or a.get('data_source') not in ('book', 'academic_literature'):
            continue
        blocks = []
        for det in page.get('layout_dets', []):
            if det.get('category_type') == 'text_block' and det.get('text') and not det.get('ignore'):
                poly = det.get('poly')
                if not poly:
                    continue
                xs, ys = poly[0::2], poly[1::2]
                if (max(xs) - min(xs)) < 30 or (max(ys) - min(ys)) < 12:
                    continue
                blocks.append({'bbox': [min(xs), min(ys), max(xs), max(ys)], 'text': det['text']})
        if len(blocks) >= 5:
            cands[lang].append({'img': os.path.basename(pi['image_path']), 'blocks': blocks})
    random.shuffle(cands['english']); random.shuffle(cands['simplified_chinese'])
    return cands['english'][:20] + cands['simplified_chinese'][:20]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    t0 = time.time()
    pages = build_pages()
    if args.smoke:
        pages = pages[:2]
    print(f'{len(pages)} synth pages; thresholds {THRESHOLDS}', flush=True)

    # ---------- (a) SYNTH: none/open once, verified per threshold ----------
    # results[cond][arm_or_thr][page_img] = {blocks, page_mean}
    synth = {c: {'none': {}, 'open': {}} for c in CONDS}
    for c in CONDS:
        for t in THRESHOLDS:
            synth[c][f'verified@{t}'] = {}

    for pi_, page in enumerate(pages):
        raw = imread(os.path.join(IMG, page['img']))
        if raw is None:
            print('MISSING', page['img']); continue
        sh = min(raw.shape[:2]); blocks = [dict(b) for b in page['blocks']]
        if sh > 1800:
            sc = 1800 / sh
            raw = cv2.resize(raw, (int(raw.shape[1]*sc), int(raw.shape[0]*sc)),
                             interpolation=cv2.INTER_AREA)
            for b in blocks:
                b['bbox'] = [c*sc for c in b['bbox']]
        for cond, dfn in CONDS.items():
            vimg = dfn(raw.copy())
            # none
            ed = block_edits(vimg, blocks)
            synth[cond]['none'][page['img']] = {'blocks': ed, 'page_mean': round(sum(ed)/max(len(ed),1),4)}
            # open
            oimg = stack_open(vimg.copy())
            ed = block_edits(oimg, blocks)
            synth[cond]['open'][page['img']] = {'blocks': ed, 'page_mean': round(sum(ed)/max(len(ed),1),4)}
            # verified per threshold (dedup identical outputs)
            seen = {}
            for t in THRESHOLDS:
                nv._ACCEPT_GAIN = t
                if pi_ == 0:
                    _probe_first['tag'] = f'synth verified@{t}'
                vout = stack_verified(vimg.copy())
                h = imhash(vout)
                if h in seen:
                    cell = seen[h]
                else:
                    ed = block_edits(vout, blocks)
                    cell = {'blocks': ed, 'page_mean': round(sum(ed)/max(len(ed),1),4)}
                    seen[h] = cell
                synth[cond][f'verified@{t}'][page['img']] = cell
        print(f'  synth {pi_+1}/{len(pages)}  {time.time()-t0:.0f}s', flush=True)
        json.dump(synth, open(os.path.join(OUTDIR, 'phase3_synth_sweep.json'), 'w', encoding='utf-8'), ensure_ascii=False)

    # ---------- (b) CAPTURES: raw/open reused, verified per threshold ----------
    capstats = json.load(open(CAPSTATS, encoding='utf-8'))
    caps = sorted(c for c in os.listdir(DEFECTS)
                  if c.lower().endswith(('.png', '.jpg', '.jpeg')))
    if args.smoke:
        caps = caps[:2]
    cap_res = {}
    for ci, name in enumerate(caps):
        img0 = imread(os.path.join(DEFECTS, name))
        if img0 is None:
            print('UNREADABLE', name); continue
        img0 = deskew(img0)
        sh = min(img0.shape[:2])
        if sh > 1800:
            sc = 1800/sh
            img0 = cv2.resize(img0, (int(img0.shape[1]*sc), int(img0.shape[0]*sc)),
                              interpolation=cv2.INTER_AREA)
        idx = os.path.splitext(name)[0]
        arms = capstats.get(idx, {})
        raw_chars = arms.get('raw', {}).get('chars')
        open_chars = arms.get('open', {}).get('chars')
        rec = {'raw_chars': raw_chars, 'open_chars': open_chars, 'verified': {}}
        seen = {}
        path = os.path.join(DEFECTS, name)
        for t in THRESHOLDS:
            nv._ACCEPT_GAIN = t
            if ci == 0:
                _probe_first['tag'] = f'capture verified@{t}'
            vout = cap_stack_verified(img0.copy(), path)
            h = imhash(vout)
            if h in seen:
                vchars = seen[h]
            else:
                vchars = sum(len(r[1]) for r in ocr_lines(vout))
                seen[h] = vchars
            rec['verified'][str(t)] = vchars
        cap_res[idx] = rec
        print(f'  cap {ci+1}/{len(caps)} {name}: raw={raw_chars} open={open_chars} '
              f"ver={ {t:rec['verified'][str(t)] for t in THRESHOLDS} }  {time.time()-t0:.0f}s", flush=True)
        json.dump(cap_res, open(os.path.join(OUTDIR, 'phase3_capture_sweep.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    print(f'\nDONE compute {time.time()-t0:.0f}s. Acceptance tally (all arms): '
          f"{_probe_first['acc']}/{_probe_first['off']}", flush=True)


if __name__ == '__main__':
    main()
