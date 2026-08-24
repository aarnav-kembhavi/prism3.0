# -*- coding: utf-8 -*-
"""Task 2 scoring — 44 uncontrolled captures, three arms.

BLOCKED: the requested metric (mean per-block OCR edit distance vs ground truth,
and "pages damaged = edit distance worsened by >0.02 vs raw") CANNOT be computed.
The 44 captures in test_images/real/defects/defects-images/ have no ground-truth
transcription and no GT block geometry anywhere in this repo. No substitute
reference is fabricated here.

What IS computed, from results/rebuttal/task2_captures_3arm.json:
  - per-arm OCR char yield, line count, mean confidence
  - per-page yield ratio vs the none arm (none == 1.0 by construction)
  - paired bootstrap 95% CIs over the 44 pages (10k resamples, seed 0; the SAME
    resampled page indices across all three arms), matching the protocol in
    scratchpad_runs/probe_gate/bootstrap_cis.py
  - damage counts at the criterion already used in Sec 4.6 / phase3
    (>10% char loss vs none, none > 200 chars), plus a stricter >2% variant
    reported ONLY as the nearest available analog to the requested >0.02 rule --
    it is a yield criterion, NOT an edit-distance criterion.
"""
import sys, io, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np

from pathlib import Path; ROOT = str(Path(__file__).resolve().parents[2])
D = json.load(open(os.path.join(ROOT, 'results', 'rebuttal', 'task2_captures_3arm.json'),
                   encoding='utf-8'))
pages = D['pages']
ids = sorted(pages, key=lambda s: (len(s), s))
n = len(ids)
ARMS = ('none', 'open', 'verified')
print(f'pages: {n}   accept_gain={D["_meta"]["accept_gain"]}   gt_available={D["_meta"]["gt_available"]}')
assert n == 44, f'expected 44 captures, got {n}'

chars = {a: np.array([pages[p]['arms'][a]['n_chars'] for p in ids], float) for a in ARMS}
lines = {a: np.array([pages[p]['arms'][a]['n_lines'] for p in ids], float) for a in ARMS}
conf = {a: np.array([pages[p]['arms'][a]['conf_mean'] for p in ids], float) for a in ARMS}

base = chars['none']
ratio = {a: np.divide(chars[a], base, out=np.ones_like(base), where=base > 0) for a in ARMS}

rng = np.random.default_rng(0)
idx = rng.integers(0, n, size=(10000, n))


def ci(x):
    m = x[idx].mean(axis=1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi)


print('\n=== Per-arm aggregates (44 captures) ===')
print(f'{"arm":9s} {"mean chars":>26s} {"mean lines":>10s} {"conf":>7s} {"mean yield ratio vs none":>34s}')
rows = {}
for a in ARMS:
    mc, lc, hc = ci(chars[a])
    mr, lr, hr = ci(ratio[a])
    rows[a] = {'chars': (mc, lc, hc), 'ratio': (mr, lr, hr),
               'lines': float(lines[a].mean()), 'conf': float(conf[a].mean()),
               'total_chars': int(chars[a].sum())}
    print(f'{a:9s} {mc:8.1f} [{lc:7.1f}, {hc:7.1f}] {lines[a].mean():10.1f} '
          f'{conf[a].mean():7.4f} {mr:14.4f} [{lr:.4f}, {hr:.4f}]')

print('\n=== Paired differences, 95% paired-bootstrap CI (10k, seed 0) ===')
pairs = [('open', 'none'), ('verified', 'none'), ('verified', 'open')]
diffs = {}
for a, b in pairs:
    for label, src in (('chars', chars), ('ratio', ratio)):
        d = src[a] - src[b]
        m, lo, hi = ci(d)
        star = '' if lo <= 0 <= hi else ' *'
        diffs[f'{a}-{b}|{label}'] = {'mean': m, 'lo': lo, 'hi': hi,
                                     'excludes_zero': not (lo <= 0 <= hi)}
        unit = 'chars' if label == 'chars' else 'ratio'
        print(f'{a:9s} - {b:9s} [{unit:5s}] {m:+9.4f} [{lo:+9.4f}, {hi:+9.4f}]{star}')

print('\n=== Damage counts (yield criterion; NOT edit distance) ===')
dmg = {}
for thr, name in ((0.10, '>10% char loss (Sec 4.6 / phase3 criterion)'),
                  (0.02, '>2% char loss (nearest analog to the requested >0.02 rule)')):
    print(f'\n-- {name}, none > 200 chars --')
    dmg[thr] = {}
    for a in ARMS:
        hits = [ids[i] for i in range(n)
                if base[i] > 200 and chars[a][i] < (1 - thr) * base[i]]
        dmg[thr][a] = hits
        print(f'  {a:9s} damaged {len(hits):2d}/44  {sorted(hits, key=lambda s:(len(s),s))}')
    prevented = [p for p in dmg[thr]['open'] if p not in dmg[thr]['verified']]
    induced = [p for p in dmg[thr]['verified'] if p not in dmg[thr]['open']]
    dmg[thr]['prevented'] = prevented
    dmg[thr]['induced'] = induced
    print(f'  prevented by gate: {len(prevented)}/{len(dmg[thr]["open"])}  {prevented}')
    print(f'  induced by gate  : {len(induced)}  {induced}')

print('\n=== Gate behaviour ===')
ident = {a: sum(1 for p in ids if pages[p]['identical_to_none'][a]) for a in ARMS}
print(f'  outputs bit-identical to none:  open {ident["open"]}/44   verified {ident["verified"]}/44')
tally = {}
for p in ids:
    for name, acc in pages[p]['gate_decisions']:
        t = tally.setdefault(name, [0, 0])
        t[0 if acc else 1] += 1
acc_tot = sum(v[0] for v in tally.values()); rej_tot = sum(v[1] for v in tally.values())
print(f'  {"step":15s} {"accepted":>9s} {"rejected":>9s}')
for k, v in tally.items():
    print(f'  {k:15s} {v[0]:9d} {v[1]:9d}')
print(f'  {"TOTAL":15s} {acc_tot:9d} {rej_tot:9d}   '
      f'rejection rate {rej_tot/max(acc_tot+rej_tot,1)*100:.1f}%')

print('\n=== Per-page table (chars) ===')
print(f'{"id":6s} {"none":>7s} {"open":>7s} {"verified":>9s} {"open/none":>10s} {"ver/none":>9s}')
per_page = {}
for i, p in enumerate(ids):
    per_page[p] = {a: int(chars[a][i]) for a in ARMS}
    per_page[p].update({'ratio_open': round(float(ratio['open'][i]), 4),
                        'ratio_verified': round(float(ratio['verified'][i]), 4)})
    print(f'{p:6s} {chars["none"][i]:7.0f} {chars["open"][i]:7.0f} {chars["verified"][i]:9.0f} '
          f'{ratio["open"][i]:10.3f} {ratio["verified"][i]:9.3f}')

json.dump({'n_pages': n, 'arms': rows, 'diffs': diffs,
           'damage': {str(k): {a: v for a, v in d.items()} for k, d in dmg.items()},
           'identical_to_none': ident, 'gate_tally': tally,
           'per_page': per_page,
           'BLOCKED': 'per-block edit distance vs GT: no ground truth for these 44 captures'},
          open(os.path.join(ROOT, 'results', 'rebuttal', 'task2_summary.json'), 'w'), indent=1)
print('\nwrote results/rebuttal/task2_summary.json')
