# -*- coding: utf-8 -*-
"""Task 2 — three-arm run over the 44 uncontrolled real captures.

Arms (132 page runs total, 44 x 3):
  none     : corrections disabled. deskew + 1800px shorter-side cap ONLY --
             i.e. exactly the image the other two arms receive as input, with
             the photometric/geometric correction stack switched off.
  open     : forced open-loop camera stack, no gate
             (white_balance -> rectify -> glare -> shadow -> clahe if std<45)
  verified : identical stack, every step routed through the probe gate
             (normalization.verified.verified_apply, _ACCEPT_GAIN = 1.02 default)

For each arm we persist the FULL OCR output (text + per-line boxes/conf), not
just aggregate counts, so any downstream metric (incl. a GT-based block edit
distance, should ground truth ever be transcribed) can be computed offline
without re-running the stack.

NOTE ON THE METRIC: these 44 captures have NO ground-truth transcription in
this repo (verified 2026-08-02). This script therefore computes NO edit
distance against GT. It emits raw OCR per arm; scoring is a separate step.

Run: venvs/gpu/Scripts/python.exe scripts/rebuttal/task2_capture_arms.py
"""
import sys, io, os, json, time, hashlib
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from pathlib import Path; ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, ROOT)

from normalization import verified as nv  # import gate module first
import numpy as np
import cv2
from normalization.frequency_filter import (
    remove_glare, remove_shadows, normalize_contrast, white_balance_gray_world)
from normalization.geometric import detect_and_rectify, deskew

DEFECTS = os.path.join(ROOT, 'test_images', 'real', 'defects', 'defects-images')
OUTDIR = os.path.join(ROOT, 'results', 'rebuttal')
os.makedirs(OUTDIR, exist_ok=True)
OUT = os.path.join(OUTDIR, 'task2_captures_3arm.json')

from rapidocr_onnxruntime import RapidOCR
W = os.path.join(ROOT, 'weights')
engine = RapidOCR(det_limit_type='max', det_limit_side_len=1280, det_box_thresh=0.3,
                  det_model_path=os.path.join(W, 'PP-OCRv6_det_small.onnx'),
                  rec_model_path=os.path.join(W, 'PP-OCRv6_rec_small.onnx'))


def imread(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


def imhash(img):
    return hashlib.md5(np.ascontiguousarray(img)).hexdigest()


def stack_open(img, path):
    for fn in (white_balance_gray_world,
               lambda im: detect_and_rectify(path, img_override=im),
               remove_glare, remove_shadows):
        try:
            o = fn(img)
            if o is not None:
                img = o
        except Exception as e:
            print('   open-stack err', e, flush=True)
    if float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).std()) < 45.0:
        try:
            img = normalize_contrast(img)
        except Exception:
            pass
    return img


def stack_verified(img, path):
    s = nv.probe_score(img)
    decisions = []
    for name, fn in (('white_balance', white_balance_gray_world),
                     ('rectify', lambda im: detect_and_rectify(path, img_override=im)),
                     ('glare', remove_glare),
                     ('shadow', remove_shadows)):
        img, s, acc = nv.verified_apply(name, fn, img, s)
        decisions.append((name, bool(acc)))
    if float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).std()) < 45.0:
        img, s, acc = nv.verified_apply('clahe', normalize_contrast, img, s)
        decisions.append(('clahe', bool(acc)))
    return img, decisions


def ocr_record(img):
    res, _ = engine(img)
    res = res or []
    lines = [{'box': [[round(float(c), 1) for c in pt] for pt in r[0]],
              'text': r[1], 'conf': round(float(r[2]), 4)} for r in res]
    text = '\n'.join(r[1] for r in res)
    return {'n_lines': len(lines),
            'n_chars': sum(len(r[1]) for r in res),
            'conf_mean': round(float(np.mean([r[2] for r in res])), 4) if res else 0.0,
            'text': text,
            'lines': lines}


def main():
    caps = sorted(c for c in os.listdir(DEFECTS)
                  if c.lower().endswith(('.png', '.jpg', '.jpeg')))
    print(f'captures: {len(caps)}   accept_gain={nv._ACCEPT_GAIN}', flush=True)
    out = {'_meta': {'n_captures': len(caps),
                     'accept_gain': float(nv._ACCEPT_GAIN),
                     'arms': ['none', 'open', 'verified'],
                     'none_arm': 'deskew + 1800px shorter-side cap, no corrections',
                     'ocr': 'RapidOCR PP-OCRv6_det_small/rec_small, det_limit_side_len=1280, box_thresh=0.3',
                     'gt_available': False},
           'pages': {}}
    t0 = time.time()
    for ci, name in enumerate(caps):
        img0 = imread(os.path.join(DEFECTS, name))
        if img0 is None:
            print('UNREADABLE', name, flush=True)
            continue
        path = os.path.join(DEFECTS, name)
        img0 = deskew(img0)
        sh = min(img0.shape[:2])
        if sh > 1800:
            sc = 1800 / sh
            img0 = cv2.resize(img0, (int(img0.shape[1] * sc), int(img0.shape[0] * sc)),
                              interpolation=cv2.INTER_AREA)
        pid = os.path.splitext(name)[0]

        imgs = {'none': img0}
        imgs['open'] = stack_open(img0.copy(), path)
        vimg, decisions = stack_verified(img0.copy(), path)
        imgs['verified'] = vimg

        rec = {'file': name, 'h': int(img0.shape[0]), 'w': int(img0.shape[1]),
               'gate_decisions': decisions, 'arms': {}}
        hashes = {a: imhash(im) for a, im in imgs.items()}
        rec['identical_to_none'] = {a: (hashes[a] == hashes['none']) for a in imgs}
        for arm, im in imgs.items():
            rec['arms'][arm] = ocr_record(im)
        out['pages'][pid] = rec
        c = {a: rec['arms'][a]['n_chars'] for a in ('none', 'open', 'verified')}
        print(f"  {ci+1:2d}/{len(caps)} {name:10s} none={c['none']:5d} "
              f"open={c['open']:5d} verified={c['verified']:5d}  "
              f"{time.time()-t0:.0f}s", flush=True)
        json.dump(out, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False)

    json.dump(out, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False)
    print(f'DONE {time.time()-t0:.0f}s -> {OUT}', flush=True)


if __name__ == '__main__':
    main()
