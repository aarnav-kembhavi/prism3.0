# -*- coding: utf-8 -*-
"""Phase 5 aggregation — mean/median CER/ED per arm, paired bootstrap CIs,
damage counts (0c), LaTeX S8 table + verdict paragraph.

CER/ED here use PP-OCRv6 (Tesseract unavailable). Absolute values are NOT the
published DocUNet numbers; only the arm comparison (the damage-filter claim) is
interpreted. Canonical Tesseract cells are reported BLOCKED.
"""
import sys, io, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
from pathlib import Path; ROOT = str(Path(__file__).resolve().parents[2])
R = os.path.join(ROOT, 'results', 'rebuttal')
d = json.load(open(os.path.join(R, 'phase5_docunet_perimage.json'), encoding='utf-8'))
ARMS = ['none', 'open', 'verified']

def subset(items, only_sub1):
    return {k: v for k, v in items.items() if (v['in_subset1'] or not only_sub1)}

def boot_diff(a, b, n=10000):
    a = np.asarray(a, float); b = np.asarray(b, float)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    dm = (a - b)[idx].mean(axis=1)
    lo, hi = np.percentile(dm, [2.5, 97.5])
    return float((a - b).mean()), float(lo), float(hi), bool(lo <= 0 <= hi)

def report(items, label):
    print(f"\n================ {label}  (n={len(items)} images) ================")
    # per-arm CER/ED (drop None CER)
    cer = {a: [] for a in ARMS}; ed = {a: [] for a in ARMS}; ch = {a: [] for a in ARMS}
    paired_keys = [k for k, v in items.items()
                   if all(v[a]['cer_ppocr'] is not None for a in ARMS)]
    for k in sorted(paired_keys):
        v = items[k]
        for a in ARMS:
            cer[a].append(v[a]['cer_ppocr']); ed[a].append(v[a]['ed_ppocr']); ch[a].append(v[a]['chars'])
    stat = {}
    print(f"{'arm':10s} {'CER_mean':>9s} {'CER_med':>9s} {'ED_mean':>9s} {'ED_med':>9s} {'chars_mean':>11s}")
    for a in ARMS:
        c = np.array(cer[a]); e = np.array(ed[a]); cc = np.array(ch[a])
        stat[a] = {'cer_mean': float(c.mean()), 'cer_median': float(np.median(c)),
                   'ed_mean': float(e.mean()), 'ed_median': float(np.median(e)),
                   'chars_mean': float(cc.mean())}
        print(f"{a:10s} {c.mean():9.4f} {np.median(c):9.4f} {e.mean():9.1f} {np.median(e):9.1f} {cc.mean():11.1f}")
    print("\npaired bootstrap 95% CI on CER (negative = arm better than baseline):")
    diffs = {}
    for a, b in (('verified', 'none'), ('open', 'none'), ('verified', 'open')):
        m, lo, hi, sz = boot_diff([items[k][a]['cer_ppocr'] for k in sorted(paired_keys)],
                                  [items[k][b]['cer_ppocr'] for k in sorted(paired_keys)])
        diffs[f'{a}-{b}'] = {'mean': m, 'lo': lo, 'hi': hi, 'straddles_zero': sz}
        print(f"  {a}-{b}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}]{'' if sz else ' *'}")
    # damage (0c): char-loss vs none, none>200
    none_ch = {k: items[k]['none']['chars'] for k in items}
    def dmg(k, a):
        r = none_ch[k]
        return r > 200 and items[k][a]['chars'] < 0.9 * r
    open_d = [k for k in items if dmg(k, 'open')]
    ver_d = [k for k in items if dmg(k, 'verified')]
    prevented = [k for k in open_d if k not in ver_d]
    print(f"\n0c damage (char-loss vs none, none>200): open damages {len(open_d)}, "
          f"verified damages {len(ver_d)}, verified prevents {len(prevented)}/{len(open_d)}")
    return {'n': len(items), 'n_paired': len(paired_keys), 'per_arm': stat, 'cer_diffs': diffs,
            'damage': {'open_damaged': open_d, 'verified_damaged': ver_d,
                       'prevented': prevented, 'open_count': len(open_d),
                       'verified_count': len(ver_d), 'prevented_count': len(prevented)}}

full = report(d, 'FULL n=130')
sub = report(subset(d, True), 'SETTING-1 subset (60 img)')

# LaTeX S8-style table (PP-OCRv6 proxy)
lat = []
lat.append(r'\begin{table}[t]\centering')
lat.append(r'\caption{DocUNet three-arm normalization audit (n=130). CER/ED via '
           r'\textbf{PP-OCRv6} (Tesseract 5.0.1 unavailable in this environment; '
           r'canonical CER/ED pending). Lower is better.}')
lat.append(r'\label{tab:s8}\begin{tabular}{lcccc}\toprule')
lat.append(r'Arm & CER (mean) & CER (med) & ED (mean) & chars \\\midrule')
for a in ARMS:
    s = full['per_arm'][a]
    lat.append(f"{a} & {s['cer_mean']:.3f} & {s['cer_median']:.3f} & "
               f"{s['ed_mean']:.1f} & {s['chars_mean']:.0f} \\\\")
lat.append(r'\bottomrule\end{tabular}\end{table}')
latex = '\n'.join(lat)
print('\n' + latex)

out = {'full_130': full, 'subset1_60': sub, 'latex_s8': latex,
       'engine': 'PP-OCRv6 (RapidOCR det1280); Tesseract 5.0.1 BLOCKED',
       'canonical_protocol': 'Setting 1 DocTr: 60 images, pytesseract 0.3.8, '
                             'Tesseract 5.0.1.20220118, CER+ED'}
json.dump(out, open(os.path.join(R, 'phase5_summary.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print(f"\nsaved {os.path.join(R, 'phase5_summary.json')}")
