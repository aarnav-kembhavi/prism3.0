# -*- coding: utf-8 -*-
"""Phase 2b — for the benchmark FP pages (routed into the correction path),
determine whether the submitted build's normalized image is bit-identical to the
no-correction (skip) path. Under PRISM_NORM_STRICT=1 the verified probe is OFF, so
any camera-routed page receives OPEN-LOOP corrections. This measures whether those
corrections actually changed pixels (=> emitted output not bit-identical).
"""
import sys, io, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from pathlib import Path; ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, ROOT)
os.environ['PRISM_NORM_STRICT'] = '1'

import numpy as np
import cv2
from normalization.pipeline import normalize_image
from normalization.geometric import deskew

IMG = os.path.join(ROOT, 'data', 'omnidocbench_full', 'images')
aud = json.load(open(os.path.join(ROOT, 'results', 'rebuttal', 'modality_audit.json'),
                     encoding='utf-8'))
fp_names = aud['summary']['fp_page_names']
print(f'{len(fp_names)} FP pages\n')

def imread(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)

rows = []
import contextlib
for name in fp_names:
    path = os.path.join(IMG, name)
    with contextlib.redirect_stdout(io.StringIO()):   # silence [norm] chatter
        norm, _fid, _mod = normalize_image(path)       # submitted path, STRICT=1
    norm = cv2.cvtColor(np.array(norm), cv2.COLOR_RGB2BGR) if not isinstance(norm, np.ndarray) else norm
    base = deskew(imread(path))
    if base.shape != norm.shape:
        base = cv2.resize(base, (norm.shape[1], norm.shape[0]), interpolation=cv2.INTER_AREA)
    diff = np.abs(norm.astype(np.int16) - base.astype(np.int16))
    mad = float(diff.mean())
    changed = float((diff.max(axis=2) > 0).mean())
    identical = bool(np.array_equal(norm, base))
    rows.append({'page': name, 'bit_identical': identical,
                 'mean_abs_diff': round(mad, 3),
                 'frac_pixels_changed': round(changed, 4),
                 'data_source': aud['benchmark'][name].get('data_source'),
                 'white_frac': aud['benchmark'][name].get('white_frac')})
    print(f"{'IDENT' if identical else 'CORRECTED':10s} mad={mad:6.2f} "
          f"chg={changed*100:5.1f}%  {aud['benchmark'][name].get('data_source'):20s} {name[:50]}")

n_ident = sum(r['bit_identical'] for r in rows)
print(f"\n{n_ident}/{len(rows)} FP pages bit-identical to no-correction path; "
      f"{len(rows)-n_ident} were open-loop CORRECTED on the submitted benchmark.")
json.dump(rows, open(os.path.join(ROOT, 'results', 'rebuttal', 'phase2_fp_bitidentity.json'),
                     'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('saved results/rebuttal/phase2_fp_bitidentity.json')
