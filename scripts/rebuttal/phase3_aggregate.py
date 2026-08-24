# -*- coding: utf-8 -*-
"""Phase 3 aggregation — per-threshold tables + paired bootstrap CIs.

Reads phase3_synth_sweep.json, phase3_capture_sweep.json, phase3_acceptance.json.
Outputs phase3_summary.json and prints tables. Damage criterion (0c):
>10% char loss vs raw, only pages with raw>200 chars.
"""
import sys, io, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
from pathlib import Path; ROOT = str(Path(__file__).resolve().parents[2])
R = os.path.join(ROOT, 'results', 'rebuttal')
THRESHOLDS = [1.00, 1.01, 1.02, 1.05, 1.10]
CONDS = ['clean', 'shadow', 'glare', 'cast', 'combo']

synth = json.load(open(os.path.join(R, 'phase3_synth_sweep.json'), encoding='utf-8'))
caps = json.load(open(os.path.join(R, 'phase3_capture_sweep.json'), encoding='utf-8'))
acc = None
p = os.path.join(R, 'phase3_acceptance.json')
if os.path.exists(p):
    acc = json.load(open(p, encoding='utf-8'))

def arr(cond, arm):
    pages = sorted(synth[cond]['none'].keys())
    return np.array([synth[cond][arm][pg]['page_mean'] for pg in pages]), pages

out = {'thresholds': THRESHOLDS, 'synth': {}, 'captures': {}, 'acceptance': acc}

# ---- synth: mean per-block edit + paired bootstrap (verified@t - none/open) ----
print('=== SYNTH: mean per-page block edit distance (lower=better) ===')
hdr = f"{'cond':8s} {'none':>7s} {'open':>7s}" + ''.join(f'  ver@{t:<4}' for t in THRESHOLDS)
print(hdr)
for cond in CONDS:
    if cond not in synth:
        continue
    none, pages = arr(cond, 'none')
    n = len(pages)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n, size=(10000, n))
    openv, _ = arr(cond, 'open')
    row = {'none': float(none.mean()), 'open': float(openv.mean()), 'verified': {}, 'diffs': {}}
    line = f"{cond:8s} {none.mean():7.4f} {openv.mean():7.4f}"
    for t in THRESHOLDS:
        v, _ = arr(cond, f'verified@{t}')
        row['verified'][str(t)] = float(v.mean())
        line += f"  {v.mean():7.4f}"
        # paired bootstrap (verified - none) and (verified - open)
        d_vn = v - none; d_vo = v - openv
        vn = d_vn[idx].mean(axis=1); vo = d_vo[idx].mean(axis=1)
        lo_vn, hi_vn = np.percentile(vn, [2.5, 97.5])
        lo_vo, hi_vo = np.percentile(vo, [2.5, 97.5])
        row['diffs'][str(t)] = {
            'verified-none': {'mean': float(d_vn.mean()), 'lo': float(lo_vn), 'hi': float(hi_vn),
                              'straddles_zero': bool(lo_vn <= 0 <= hi_vn)},
            'verified-open': {'mean': float(d_vo.mean()), 'lo': float(lo_vo), 'hi': float(hi_vo),
                              'straddles_zero': bool(lo_vo <= 0 <= hi_vo)}}
    out['synth'][cond] = row
    print(line)

print('\n=== SYNTH: (verified@t - none) paired 95% CI [* = excludes 0] ===')
print(f"{'cond':8s}" + ''.join(f"  ver@{t:<20}" for t in THRESHOLDS))
for cond in CONDS:
    if cond not in synth:
        continue
    cells = []
    for t in THRESHOLDS:
        d = out['synth'][cond]['diffs'][str(t)]['verified-none']
        star = '' if d['straddles_zero'] else '*'
        cells.append(f"{d['mean']:+.4f}[{d['lo']:+.3f},{d['hi']:+.3f}]{star}")
    print(f"{cond:8s} " + '  '.join(cells))

# ---- captures: damage counts per threshold (0c) ----
print('\n=== CAPTURES: damage counts (0c: >10% char loss vs raw, raw>200) ===')
def damaged(chars, raw):
    return raw is not None and raw > 200 and chars is not None and chars < 0.9 * raw
open_dmg = set()
for cid, r in caps.items():
    if damaged(r.get('open_chars'), r.get('raw_chars')):
        open_dmg.add(cid)
print(f"open-loop damages: {len(open_dmg)} captures {sorted(open_dmg, key=lambda x:int(x))}")
print(f"{'thr':>5s} {'ver_damages':>12s} {'prevented(of open)':>20s}")
cap_out = {'open_damaged': sorted(open_dmg, key=lambda x: int(x)), 'per_threshold': {}}
for t in THRESHOLDS:
    ver_dmg = set()
    for cid, r in caps.items():
        vc = r.get('verified', {}).get(str(t))
        if damaged(vc, r.get('raw_chars')):
            ver_dmg.add(cid)
    prevented = open_dmg - ver_dmg
    cap_out['per_threshold'][str(t)] = {
        'verified_damaged': sorted(ver_dmg, key=lambda x: int(x)),
        'verified_damage_count': len(ver_dmg),
        'prevented_count': len(prevented),
        'prevented_of_open': f'{len(prevented)}/{len(open_dmg)}'}
    print(f"{t:5.2f} {len(ver_dmg):12d} {len(prevented)}/{len(open_dmg):>2d}")
out['captures'] = cap_out

if acc:
    print('\n=== acceptance rate per threshold ===')
    print(f"{'thr':>5s} {'synth':>8s} {'capture':>8s}")
    for t in THRESHOLDS:
        s = acc[str(t)]['synth']['rate']*100; c = acc[str(t)]['capture']['rate']*100
        print(f"{t:5.2f} {s:7.1f}% {c:7.1f}%")

json.dump(out, open(os.path.join(R, 'phase3_summary.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print(f"\nsaved {os.path.join(R, 'phase3_summary.json')}")
