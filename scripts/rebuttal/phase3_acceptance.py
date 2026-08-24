# -*- coding: utf-8 -*-
"""Phase 3 helper — acceptance rate per threshold (probe-only, no OCR).

The gate sequence is threshold-dependent (each step's score_before depends on
which earlier steps were accepted), so acceptance rate cannot be read off the
fixed-build probe log; we re-run the forced stacks per threshold and count
accept/reject. Probe-only -> fast. Mirrors sweep_threshold.py stacks exactly.
"""
import sys, io, os, json, time, random
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from pathlib import Path; ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, ROOT)
from normalization import verified as nv
import numpy as np, cv2
from normalization.frequency_filter import (
    remove_glare, remove_shadows, normalize_contrast, white_balance_gray_world)
from normalization.geometric import detect_and_rectify, deskew

THRESHOLDS = [1.00, 1.01, 1.02, 1.05, 1.10]
GT = os.path.join(ROOT, 'data', 'omnidocbench_full', 'OmniDocBench.json')
IMG = os.path.join(ROOT, 'data', 'omnidocbench_full', 'images')
DEFECTS = os.path.join(ROOT, 'test_images', 'real', 'defects', 'defects-images')
OUT = os.path.join(ROOT, 'results', 'rebuttal', 'phase3_acceptance.json')

# self-contained (no sweep_threshold import — avoids stdout/engine side effects)
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
    return np.clip(img.astype(np.float32) * mult[None, None, :] * 0.85, 0, 255).astype(np.uint8)

def v2_combo(img):
    return v2_shadow(v2_cast(img))

CONDS = {'clean': lambda im: im, 'shadow': v2_shadow, 'glare': v2_glare,
         'cast': v2_cast, 'combo': v2_combo}

def build_pages():
    import random
    gt = json.load(open(GT, encoding='utf-8'))
    random.seed(7)
    cands = {'english': [], 'simplified_chinese': []}
    for page in gt:
        pi = page['page_info']; a = pi.get('page_attribute', {})
        lang = a.get('language')
        if lang not in cands or a.get('data_source') not in ('book', 'academic_literature'):
            continue
        nb = 0
        for det in page.get('layout_dets', []):
            if det.get('category_type') == 'text_block' and det.get('text') and not det.get('ignore'):
                poly = det.get('poly')
                if not poly:
                    continue
                xs, ys = poly[0::2], poly[1::2]
                if (max(xs) - min(xs)) < 30 or (max(ys) - min(ys)) < 12:
                    continue
                nb += 1
        if nb >= 5:
            cands[lang].append({'img': os.path.basename(pi['image_path'])})
    random.shuffle(cands['english']); random.shuffle(cands['simplified_chinese'])
    return cands['english'][:20] + cands['simplified_chinese'][:20]

class _S:  # namespace shim
    build_pages = staticmethod(build_pages)
    CONDS = CONDS
swp = _S

def imread(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)

def count_synth_verified(img):
    acc = off = 0
    s = nv.probe_score(img)
    for name, fn in (('wb', white_balance_gray_world), ('glare', remove_glare),
                     ('shadow', remove_shadows)):
        img, s, a = nv.verified_apply(name, fn, img, s); off += 1; acc += int(a)
    if float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).std()) < 45.0:
        img, s, a = nv.verified_apply('clahe', normalize_contrast, img, s); off += 1; acc += int(a)
    return acc, off

def count_cap_verified(img, path):
    acc = off = 0
    s = nv.probe_score(img)
    steps = [('white_balance', white_balance_gray_world),
             ('rectify', lambda im: detect_and_rectify(path, img_override=im)),
             ('glare', remove_glare), ('shadow', remove_shadows)]
    for name, fn in steps:
        img, s, a = nv.verified_apply(name, fn, img, s); off += 1; acc += int(a)
    if float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).std()) < 45.0:
        img, s, a = nv.verified_apply('clahe', normalize_contrast, img, s); off += 1; acc += int(a)
    return acc, off

def main():
    pages = swp.build_pages()
    conds = swp.CONDS
    caps = sorted(c for c in os.listdir(DEFECTS)
                  if c.lower().endswith(('.png', '.jpg', '.jpeg')))
    # cache degraded synth images + precapped captures
    out = {str(t): {'synth': {'acc': 0, 'off': 0}, 'capture': {'acc': 0, 'off': 0},
                    'synth_by_cond': {c: {'acc': 0, 'off': 0} for c in conds}}
           for t in THRESHOLDS}
    t0 = time.time()
    # synth
    for pi_, page in enumerate(pages):
        raw = imread(os.path.join(IMG, page['img']))
        if raw is None: continue
        sh = min(raw.shape[:2])
        if sh > 1800:
            sc = 1800/sh
            raw = cv2.resize(raw, (int(raw.shape[1]*sc), int(raw.shape[0]*sc)), interpolation=cv2.INTER_AREA)
        for cond, dfn in conds.items():
            vimg = dfn(raw.copy())
            for t in THRESHOLDS:
                nv._ACCEPT_GAIN = t
                a, o = count_synth_verified(vimg.copy())
                out[str(t)]['synth']['acc'] += a; out[str(t)]['synth']['off'] += o
                out[str(t)]['synth_by_cond'][cond]['acc'] += a
                out[str(t)]['synth_by_cond'][cond]['off'] += o
        print(f'synth {pi_+1}/{len(pages)} {time.time()-t0:.0f}s', flush=True)
    # captures
    for ci, name in enumerate(caps):
        img0 = imread(os.path.join(DEFECTS, name))
        if img0 is None: continue
        img0 = deskew(img0)
        sh = min(img0.shape[:2])
        if sh > 1800:
            sc = 1800/sh
            img0 = cv2.resize(img0, (int(img0.shape[1]*sc), int(img0.shape[0]*sc)), interpolation=cv2.INTER_AREA)
        path = os.path.join(DEFECTS, name)
        for t in THRESHOLDS:
            nv._ACCEPT_GAIN = t
            a, o = count_cap_verified(img0.copy(), path)
            out[str(t)]['capture']['acc'] += a; out[str(t)]['capture']['off'] += o
        print(f'cap {ci+1}/{len(caps)} {time.time()-t0:.0f}s', flush=True)
    for t in THRESHOLDS:
        for grp in ('synth', 'capture'):
            d = out[str(t)][grp]
            d['rate'] = round(d['acc']/max(d['off'],1), 4)
    json.dump(out, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n=== acceptance rate ===')
    print(f"{'thr':>5s} {'synth':>10s} {'capture':>10s}")
    for t in THRESHOLDS:
        print(f"{t:5.2f} {out[str(t)]['synth']['rate']*100:9.1f}% {out[str(t)]['capture']['rate']*100:9.1f}%")
    print(f'saved {OUT}')

if __name__ == '__main__':
    main()
