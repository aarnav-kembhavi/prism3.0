# -*- coding: utf-8 -*-
"""Phase 5 — DocUNet three-arm evaluation through PRISM's normalization path.

Arms (instruction 5.3): none (raw distorted crop), open-loop (full stack always
applied), verified (probe-gated stack @ submitted threshold 1.02). Forced stack =
same as the 44-capture study: deskew + precap 1800 -> wb -> rectify -> glare ->
shadow -> clahe(if std<45).

OCR metric: CANONICAL protocol is pytesseract 0.3.8 / Tesseract 5.0.1.20220118
(Setting 1, 60 images). Tesseract is NOT installable in this sandbox (no admin;
UB-Mannheim host unreachable) -> canonical CER/ED is BLOCKED and reported as such.
We compute (a) the 0c damage-filter analysis via PP-OCRv6 char counts (the same
engine and >10%-char-loss criterion as the capture study) and (b) a PP-OCRv6-
substituted CER/ED proxy vs the flat scan GT (LABELED non-canonical; engine
swapped only because Tesseract is unavailable). No protocol is invented; the
Tesseract cells are left explicitly blocked.

Saves per-image raw numbers so CIs/canonical rescoring can be recomputed later.
Run: venvs/gpu/Scripts/python.exe scripts/rebuttal/phase5_docunet.py [--smoke]
"""
import sys, io, os, json, time, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from pathlib import Path; ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, ROOT)

from normalization import verified as nv
import numpy as np, cv2, Levenshtein
from normalization.frequency_filter import (
    remove_glare, remove_shadows, normalize_contrast, white_balance_gray_world)
from normalization.geometric import detect_and_rectify, deskew

DU = os.path.join(ROOT, 'results', 'rebuttal', 'docunet')
CROP = os.path.join(DU, 'crop')
SCAN = os.path.join(DU, 'scan')
ARMDIR = os.path.join(DU, 'arms')
OUT = os.path.join(ROOT, 'results', 'rebuttal', 'phase5_docunet_perimage.json')
THRESH = 1.02

# Setting 1 (DocTr ocr_img.txt) — 30 documents / 60 images
SUBSET1 = {1,2,3,4,5,6,7,9,10,21,22,23,24,27,30,31,32,36,38,40,41,44,45,46,47,48,50,51,52,53}

def imread(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)

def precap(img):
    img = deskew(img)
    sh = min(img.shape[:2])
    if sh > 1800:
        sc = 1800 / sh
        img = cv2.resize(img, (int(img.shape[1]*sc), int(img.shape[0]*sc)),
                         interpolation=cv2.INTER_AREA)
    return img

def stack_open(img, path):
    img = white_balance_gray_world(img)
    img = detect_and_rectify(path, img_override=img)
    img = remove_glare(img)
    img = remove_shadows(img)
    if float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).std()) < 45.0:
        img = normalize_contrast(img)
    return img

def stack_verified(img, path):
    nv._ACCEPT_GAIN = THRESH
    s = nv.probe_score(img)
    img, s, _ = nv.verified_apply('white_balance', white_balance_gray_world, img, s)
    img, s, _ = nv.verified_apply('rectify',
                                  lambda im: detect_and_rectify(path, img_override=im), img, s)
    img, s, _ = nv.verified_apply('glare', remove_glare, img, s)
    img, s, _ = nv.verified_apply('shadow', remove_shadows, img, s)
    if float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).std()) < 45.0:
        img, s, _ = nv.verified_apply('clahe', normalize_contrast, img, s)
    return img

from rapidocr_onnxruntime import RapidOCR
W = os.path.join(ROOT, 'weights')
engine = RapidOCR(det_limit_type='max', det_limit_side_len=1280, det_box_thresh=0.3,
                  det_model_path=os.path.join(W, 'PP-OCRv6_det_small.onnx'),
                  rec_model_path=os.path.join(W, 'PP-OCRv6_rec_small.onnx'))

def ocr(img):
    res, _ = engine(img)
    res = res or []
    text = ' '.join(r[1] for r in res)
    return text, len(''.join(r[1] for r in res))   # (text, char count)

def norm_txt(s):
    return ' '.join(s.split())

def cer(pred, gt):
    p, g = norm_txt(pred), norm_txt(gt)
    if not g:
        return None
    return round(Levenshtein.distance(p, g) / len(g), 4)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    for a in ('none', 'open', 'verified'):
        os.makedirs(os.path.join(ARMDIR, a), exist_ok=True)
    crops = sorted(f for f in os.listdir(CROP) if f.lower().endswith('.png'))
    if args.smoke:
        crops = [c for c in crops if c.split('_')[0] in ('1', '10')][:4]
    print(f'{len(crops)} crop images', flush=True)

    gt_cache = {}
    def gt_text(doc):
        if doc not in gt_cache:
            p = os.path.join(SCAN, f'{doc}.png')
            gt_cache[doc] = ocr(imread(p))[0] if os.path.exists(p) else ''
        return gt_cache[doc]

    res = {}
    t0 = time.time()
    for i, name in enumerate(crops):
        stem = name[:-4]
        doc = name.split('_')[0]
        try:
            docn = int(doc)
        except ValueError:
            docn = -1
        raw = imread(os.path.join(CROP, name))
        if raw is None:
            print('UNREADABLE', name); continue
        base = precap(raw)
        path = os.path.join(CROP, name)
        arms_img = {'none': base,
                    'open': stack_open(base.copy(), path),
                    'verified': stack_verified(base.copy(), path)}
        gt = gt_text(doc)
        rec = {'doc': docn, 'in_subset1': docn in SUBSET1}
        for a, im in arms_img.items():
            cv2.imwrite(os.path.join(ARMDIR, a, stem + '.png'), im)
            txt, ch = ocr(im)
            rec[a] = {'chars': ch, 'cer_ppocr': cer(txt, gt),
                      'ed_ppocr': Levenshtein.distance(norm_txt(txt), norm_txt(gt))}
        res[stem] = rec
        if i % 5 == 0:
            json.dump(res, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
            print(f'  {i+1}/{len(crops)} {stem}  none={rec["none"]["chars"]} '
                  f'open={rec["open"]["chars"]} ver={rec["verified"]["chars"]}  '
                  f'{time.time()-t0:.0f}s', flush=True)
    json.dump(res, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'DONE {time.time()-t0:.0f}s -> {OUT}', flush=True)

if __name__ == '__main__':
    main()
