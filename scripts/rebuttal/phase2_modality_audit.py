# -*- coding: utf-8 -*-
"""Phase 2 — modality classifier accuracy over 1651 OmniDocBench pages + 44 captures.

Replicates the submitted build's routing decision path (normalize_image_pil under
PRISM_NORM_STRICT=1):
    img = deskew(imread(path))
    modality = detect_capture_modality(img)          # entropy-only decision
    if PHONE_PHOTO: is_camera = _white_fraction(img) < 0.02   # separate refinement
    correction_path = PHONE_PHOTO and is_camera      # (shadow-dim rescue OFF under STRICT=1)

Ground truth: every benchmark page = born-digital/scan (NOT camera);
every defect capture = camera.
  FP = benchmark page routed into the correction path.
  FN = capture routed PAST the correction path (screenshot OR white>=0.02).

Usage: python scripts/rebuttal/phase2_modality_audit.py [--limit N]
"""
import sys, io, os, json, time, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from pathlib import Path; ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, ROOT)
os.environ.setdefault('PRISM_NORM_STRICT', '1')   # benchmark setting

import numpy as np
import cv2
from normalization.geometric import deskew
from normalization.modality import detect_capture_modality, CaptureModality
from normalization.pipeline import _white_fraction, _CAMERA_WHITE_THRESHOLD

GT = os.path.join(ROOT, 'data', 'omnidocbench_full', 'OmniDocBench.json')
IMG = os.path.join(ROOT, 'data', 'omnidocbench_full', 'images')
DEFECTS = os.path.join(ROOT, 'test_images', 'real', 'defects', 'defects-images')
CAPSTATS = os.path.join(ROOT, 'scratchpad_runs', 'probe_gate', 'captures_ocr_stats_v2.json')
OUT = os.path.join(ROOT, 'results', 'rebuttal', 'modality_audit.json')


def imread(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


def classify(path, do_deskew=True):
    img = imread(path)
    if img is None:
        return None
    h0, w0 = img.shape[:2]
    if do_deskew:
        img = deskew(img)
    mod = detect_capture_modality(img)
    is_screenshot = mod.modality == CaptureModality.SCREENSHOT
    rec = {
        'entropy': round(mod.histogram_entropy, 4),
        'occupied_bins': mod.occupied_bins,
        'modality': 'screenshot' if is_screenshot else 'phone_photo',
        'w': w0, 'h': h0,
    }
    if is_screenshot:
        rec['white_frac'] = None
        rec['route'] = 'screenshot_skip'
    else:
        wf = _white_fraction(img)
        rec['white_frac'] = round(wf, 5)
        rec['route'] = 'camera' if wf < _CAMERA_WHITE_THRESHOLD else 'clean_digital_skip'
    rec['correction_path'] = (rec['route'] == 'camera')
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0, help='cap benchmark pages (smoke)')
    args = ap.parse_args()

    gt = json.load(open(GT, encoding='utf-8'))
    names = [os.path.basename(p['page_info']['image_path']) for p in gt]
    attr = {os.path.basename(p['page_info']['image_path']):
            p['page_info'].get('page_attribute', {}) for p in gt}
    if args.limit:
        names = names[:args.limit]

    t0 = time.time()
    bench = {}
    for i, name in enumerate(names):
        rec = classify(os.path.join(IMG, name))
        if rec is None:
            print('MISSING', name); continue
        a = attr.get(name, {})
        rec['data_source'] = a.get('data_source', '?')
        rec['language'] = a.get('language', '?')
        bench[name] = rec
        if i % 200 == 0:
            print(f'bench {i+1}/{len(names)}  {time.time()-t0:.0f}s', flush=True)
    print(f'bench done {len(bench)} pages {time.time()-t0:.0f}s', flush=True)

    # captures
    capstats = json.load(open(CAPSTATS, encoding='utf-8')) if os.path.exists(CAPSTATS) else {}
    caps = sorted(c for c in os.listdir(DEFECTS)
                  if c.lower().endswith(('.png', '.jpg', '.jpeg')))
    cap_recs = {}
    for name in caps:
        rec = classify(os.path.join(DEFECTS, name), do_deskew=True)
        if rec is None:
            print('UNREADABLE', name); continue
        idx = os.path.splitext(name)[0]
        rec['study_arms'] = capstats.get(idx)   # {raw,open,verified: {chars,...}}
        cap_recs[name] = rec

    # ---- FP / FN ----
    fp = {n: r for n, r in bench.items() if r['correction_path']}          # benchmark -> corrections
    phone_labeled = {n: r for n, r in bench.items() if r['modality'] == 'phone_photo'}
    fn = {n: r for n, r in cap_recs.items() if not r['correction_path']}   # capture -> skipped

    # bit-identity for FP pages: does the camera branch actually alter pixels?
    # (route into camera under STRICT=1 => open-loop corrections => not skip path)
    for n, r in fp.items():
        r['bit_identical_to_no_correction'] = False  # entered correction path
    # phone_labeled pages that were RESCUED by the white gate (skip) are bit-identical
    for n, r in phone_labeled.items():
        if not r['correction_path']:
            r['rescued_by_white_gate'] = True

    summary = {
        'benchmark_pages': len(bench),
        'benchmark_phone_photo_labeled': len(phone_labeled),
        'benchmark_FP_correction_path': len(fp),
        'benchmark_rescued_by_white_gate': sum(
            1 for r in phone_labeled.values() if not r['correction_path']),
        'captures_total': len(cap_recs),
        'captures_camera_route': sum(1 for r in cap_recs.values() if r['correction_path']),
        'captures_FN_routed_past': len(fn),
        'fn_capture_names': sorted(fn.keys()),
        'fp_page_names': sorted(fp.keys()),
        'entropy_threshold': 0.55,
        'white_threshold': _CAMERA_WHITE_THRESHOLD,
        'strict_mode': os.environ.get('PRISM_NORM_STRICT'),
        'note': 'Under PRISM_NORM_STRICT=1 the verified probe is OFF; white-fraction '
                'gate is the only correction-path guard on the benchmark.',
    }

    out = {'summary': summary, 'benchmark': bench, 'captures': cap_recs}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n=== SUMMARY ===')
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f'\nsaved {OUT}  ({time.time()-t0:.0f}s total)')


if __name__ == '__main__':
    main()
