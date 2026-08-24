# -*- coding: utf-8 -*-
"""Efficiency frontier figure (Task 5): OmniDocBench v1.6 Overall vs weights.

Data = paper Table 1 (tab:v16). API-only rows excluded (no weight figure).
Sub-300M CPU VLMs (SmolDocling/GraniteDocling) omitted: no OmniDocBench-
comparable Overall exists (our tab:cpu_frontier numbers are a 20-page subset
with per-task metrics, not the composite).
Latency NOT encoded: no same-machine latency for the leaderboard rows.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[2])
OUT = os.path.join(ROOT, 'figures', 'efficiency_frontier.pdf')
os.makedirs(os.path.dirname(OUT), exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'Nimbus Roman', 'DejaVu Serif'],
    'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7,
    'ytick.labelsize': 7, 'legend.fontsize': 6.5,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.linewidth': 0.6,
})

# name, weights_GB, overall, family(VLM/pipeline), cpu_only
DATA = [
    ('MinerU2.5-Pro',   2.4, 95.75, 'VLM',      False),
    ('PaddleOCR-VL',    1.8, 94.18, 'VLM',      False),
    ('dots.ocr',        6.0, 90.77, 'VLM',      False),
    ('PRISM (ours)',    0.28, 87.29, 'pipeline', True),
    ('MinerU-Pipeline', 2.0, 86.47, 'pipeline', False),
    ('olmOCR',          14.0, 85.74, 'VLM',     False),
    ('Nanonets-OCR-s',  6.0, 83.61, 'VLM',      False),
    ('POINTS-Reader',   6.0, 83.37, 'VLM',      False),
    ('Marker',          3.0, 78.44, 'pipeline', False),
]

# Pareto frontier (min weights, max overall)
pts = sorted([(w, o, n) for n, w, o, _, _ in DATA])
frontier = []
best = -1
for w, o, n in pts:
    if o > best:
        frontier.append((w, o, n))
        best = o
assert any(n.startswith('PRISM') for _, _, n in frontier), 'PRISM dominated!'
assert pts[0][2].startswith('PRISM'), 'PRISM is not the leftmost point!'
assert not any(w <= 0.28 and o >= 87.29 for w, o, n in pts
               if not n.startswith('PRISM')), 'PRISM dominated by another point!'

# colorblind-safe (Okabe-Ito): VLM = blue, pipeline = vermillion-ish orange
C_VLM = '#0072B2'
C_PIPE = '#D55E00'
C_PRISM = '#D55E00'

fig, ax = plt.subplots(figsize=(3.25, 2.55))

# NOTE: the former "within 10x of PRISM weights" shaded band is deliberately
# removed — 10x of 251MB reaches 2.5GB, which contains PaddleOCR-VL and
# MinerU2.5-Pro (both ~8pts above PRISM). The claim is hardware-categorical:
# PRISM is the leftmost Pareto point and the only CPU-only system (square).

# stepped Pareto frontier
fx, fy = [], []
for i, (w, o, n) in enumerate(frontier):
    if i > 0:
        fx.append(w); fy.append(frontier[i - 1][1])
    fx.append(w); fy.append(o)
# extend right edge
fx.append(20); fy.append(frontier[-1][1])
ax.plot(fx, fy, color='0.55', lw=0.8, ls='-', zorder=2, dashes=(4, 2))

for name, w, o, fam, cpu in DATA:
    is_prism = name.startswith('PRISM')
    marker = 's' if cpu else 'o'
    color = C_PRISM if is_prism else (C_VLM if fam == 'VLM' else C_PIPE)
    # shape carries the CPU/GPU claim: the lone square must dominate visually
    # in greyscale, so it is large with a heavy black edge.
    size = 95 if cpu else 24
    ax.scatter(w, o, s=size, marker=marker,
               facecolor=color, edgecolor='black',
               linewidth=1.4 if cpu else 0.5, zorder=4 if is_prism else 3)

# manual label offsets: (dx points, dy points, ha)
OFFS = {
    'MinerU2.5-Pro':   (4, 1, 'left'),
    'PaddleOCR-VL':    (-4, 3, 'right'),
    'dots.ocr':        (4, 1, 'left'),
    'PRISM (ours)':    (0, -10, 'center'),
    'MinerU-Pipeline': (5, -1.5, 'left'),
    'olmOCR':          (4, 2, 'left'),
    'Nanonets-OCR-s':  (4, 2.5, 'left'),
    'POINTS-Reader':   (4, -5.5, 'left'),
    'Marker':          (4, -1, 'left'),
}
for name, w, o, fam, cpu in DATA:
    dx, dy, ha = OFFS[name]
    is_prism = name.startswith('PRISM')
    ax.annotate(name, (w, o), xytext=(dx, dy), textcoords='offset points',
                fontsize=7.5 if is_prism else 6.5,
                fontweight='bold' if is_prism else 'normal',
                color=C_PRISM if is_prism else 'black', ha=ha, va='center')

ax.set_xscale('log')
ax.set_xlim(0.1, 20)
ax.set_ylim(76.5, 97.5)
ax.set_xlabel('Model weights (GB, log scale)')
ax.set_ylabel('OmniDocBench v1.6 Overall')
ax.set_xticks([0.1, 0.3, 1, 3, 10])
ax.set_xticklabels(['0.1', '0.3', '1', '3', '10'])
ax.tick_params(length=2.5, width=0.6)
ax.spines[['top', 'right']].set_visible(False)

legend = [
    Line2D([], [], marker='s', ls='none', mfc='0.75', mec='black', mew=1.2,
           ms=6.5, label='CPU-only'),
    Line2D([], [], marker='o', ls='none', mfc='0.75', mec='black', mew=0.5,
           ms=5, label='GPU required'),
    Line2D([], [], marker='o', ls='none', mfc=C_VLM, mec='black', mew=0.5,
           ms=5, label='VLM'),
    Line2D([], [], marker='o', ls='none', mfc=C_PIPE, mec='black', mew=0.5,
           ms=5, label='Pipeline'),
    Line2D([], [], color='0.55', lw=0.8, dashes=(4, 2), label='Pareto frontier'),
]
ax.legend(handles=legend, loc='upper left', frameon=False,
          borderaxespad=0.2, handletextpad=0.5, labelspacing=0.35,
          bbox_to_anchor=(0.005, 0.84))

fig.tight_layout(pad=0.3)
fig.savefig(OUT)
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'frontier_check_300dpi.png'), dpi=300)
print('frontier points:', frontier)
print('saved', OUT)
