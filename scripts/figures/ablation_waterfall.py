# -*- coding: utf-8 -*-
"""Ablation waterfall (single column 3.25in, 8pt): cumulative v1.6 progression
from tab:ablation, split deterministic (solid) vs model-swap (hatched).
Deterministic sums to +11.58, model swaps to +5.34 (70.37 -> 87.29, v21).

HORIZONTAL layout: bars run left-to-right, stage names upright at 8pt on the
y axis — a 3.25in column cannot fit 10 rotated x labels legibly."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[2])
HERE = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
    'ytick.labelsize': 8, 'pdf.fonttype': 42, 'axes.linewidth': 0.6})

BLUE, ORANGE = '#0072B2', '#E69F00'

# (label, overall_after, deterministic?)  -- numbers = tab:ablation, v21 final
STAGES = [
    ('baseline (v9)',                 70.37, None),
    ('formula ink geometry+',         78.46, True),
    ('reading rules, marginalia',     80.43, True),
    ('SLANet-plus, PP-OCRv6',         83.55, False),
    ('DocLayoutV3, native order',     85.77, False),
    ('inline-formula recovery',       86.35, True),
    ('text guard, render repair',     86.37, True),
    ('inline-math splicing',          86.62, True),
    ('column assembly, table repair', 87.29, True),
]
XMIN = 68.0
NROW = len(STAGES) + 1          # + final total row

fig, ax = plt.subplots(figsize=(3.25, 2.95))

prev = None
det_sum = swap_sum = 0.0
for i, (lab, val, det) in enumerate(STAGES):
    if det is None:                      # baseline: full bar from axis floor
        ax.barh(i, val - XMIN, left=XMIN, height=0.62, facecolor='0.82',
                edgecolor='black', linewidth=0.6)
        ax.text(val + 0.3, i, f'{val:.2f}', va='center', fontsize=8)
    else:
        d = val - prev
        (det_sum, swap_sum) = (det_sum + d, swap_sum) if det else \
                              (det_sum, swap_sum + d)
        ax.barh(i, d, left=prev, height=0.62,
                facecolor=BLUE if det else ORANGE,
                hatch=None if det else '////',
                edgecolor='black', linewidth=0.6)
        # connector from previous bar end, vertical
        ax.plot([prev, prev], [i - 1 + 0.31, i - 0.31], color='0.45',
                linewidth=0.6, linestyle=(0, (1.5, 1.5)))
        ax.text(val + 0.3, i, f'+{d:.2f}', va='center', fontsize=8)
    prev = val

# final total row
i = NROW - 1
ax.barh(i, prev - XMIN, left=XMIN, height=0.62, facecolor='0.82',
        edgecolor='black', linewidth=0.9)
ax.plot([prev, prev], [i - 1 + 0.31, i - 0.31], color='0.45', linewidth=0.6,
        linestyle=(0, (1.5, 1.5)))
ax.text(prev + 0.3, i, f'{prev:.2f}', va='center', fontsize=8,
        fontweight='bold')

labels = [s[0] for s in STAGES] + ['final (v21)']
ax.set_yticks(range(NROW))
ax.set_yticklabels(labels, fontsize=8)
ax.invert_yaxis()                        # baseline on top, final at bottom
ax.set_xlim(XMIN, 91.2)
ax.set_ylim(NROW - 0.55, -0.55)
ax.set_xticks([68, 72, 76, 80, 84, 88])
ax.set_xlabel('OmniDocBench v1.6 Overall\n(axis starts at 68, not 0)')
ax.tick_params(length=2.5, width=0.6)
ax.spines[['top', 'right']].set_visible(False)

# callout: the paper's thesis in two numbers. The empty region of an
# ascending horizontal waterfall is the lower-left hole (late rows only have
# ink at x >= 85.8), so callout + legend live there.
ax.text(0.03, 0.115,
        'deterministic: $\\bf{+11.58}$\nmodel swaps: $\\bf{+5.34}$',
        transform=ax.transAxes, ha='left', va='bottom', fontsize=8,
        bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='0.4', lw=0.7))
assert abs(det_sum - 11.58) < 0.005 and abs(swap_sum - 5.34) < 0.005, \
    (det_sum, swap_sum)

legend = [Patch(facecolor=BLUE, edgecolor='black', lw=0.6,
                label='deterministic\n(no new weights)'),
          Patch(facecolor=ORANGE, edgecolor='black', lw=0.6, hatch='////',
                label='model swap')]
ax.legend(handles=legend, loc='lower left', frameon=False, fontsize=8,
          handlelength=1.2, borderaxespad=0.15, labelspacing=0.45,
          bbox_to_anchor=(0.015, 0.30))

fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(ROOT, 'figures', 'ablation_waterfall.pdf'))
fig.savefig(os.path.join(HERE, 'waterfall_check_300dpi.png'), dpi=300)
print('det=%.2f swap=%.2f saved figures/ablation_waterfall.pdf' % (det_sum, swap_sum))
