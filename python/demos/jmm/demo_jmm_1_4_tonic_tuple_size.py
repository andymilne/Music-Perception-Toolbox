"""demo_jmm_1_4_tonic_tuple_size.py — Analysis 1.4 (Online Supplement,
Section 7): cadence-tonic matching at increasing tuple size.

A demo of the Music Perception Toolbox reproducing the analysis from the
JMM article; lightly edited from the article's own script. Data come
from jmm_data (BWV 347 read from the bundled MusicXML) or piano_phase
(the rendered Piano Phase voices); the figures stay on screen unless
SAVE_FIGURES is set.


Analysis 1.4: structural matching of the four cadence tonics of BWV 347
at increasing tuple size r, on a single chord (no nesting).

Each cadence tonic (the final chord of cadences C1-C4) is one event with
a single pitch attribute, the unordered chord multiset (exch = 1, K = 4).
The four tonics are compared pairwise under the cross of absolute vs
relative mode and non-periodic vs periodic, each swept over r in {1,2,3}.
Because the comparison is on one attribute (not a role product), raising
r tightens the match informatively rather than annihilating it: pitch
content (r = 1) -> dyad/interval content (r = 2) -> triad content (r = 3).

The relative, periodic row is the informative one. At r = 2 the interval-class
content cannot separate a major triad from a minor one (they are
inversionally related, and the unordered relative pair content is
inversion-invariant): the three major tonics and the minor tonic all read
as near-identical. At r = 3 the triadic structure separates them: the
three majors stay mutually 1 (transposition-equivalent) and the minor
isolates. Relative mode at r = 1 is a constant (degenerate) density and is
shown only for completeness.

These are the same four tonics compared as whole nested cadences in
Analysis 1.3, in the same panel layout, so the two figures read together.

Data: ``jmm_data.bwv347_notes`` (the bundled MusicXML read with
``read_score``, repeats expanded). Toolbox: ``grid_attr_table``,
``pre_maet_from_attr_table``, ``select_pre_maet``, ``build_maet``,
``sim_maet``. Runtime: a second or two.
"""
import os
import warnings
import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

import mpt

# Set True to write the figures to a figures/ folder beside this script;
# False shows them instead.
SAVE_FIGURES = False
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
_prev_defaults = mpt.set_default(show_hints=False)
from mpt import build_maet, sim_maet

from jmm_data import GRID_STEP_QN, bwv347_notes


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
SIGMA_PITCH = 0.15
PERIOD = 12.0
R_VALUES = [1, 2, 3]
TONIC_TIMES = {1: 7.0, 2: 15.0, 3: 45.0, 4: 65.0}   # cadence finals
CADENCE_NAMES = ['C1', 'C2', 'C3', 'C4']

ROWS = [
    (False, False, 'absolute, non-periodic'),
    (False, True,  'absolute, periodic'),
    (True,  False, 'relative, non-periodic'),
    (True,  True,  'relative, periodic'),
]


# ---------------------------------------------------------------------------
# Extract the cadence tonics
# ---------------------------------------------------------------------------
print('Loading BWV 347 and extracting cadence tonics...')
# The chorale as a pre-MAET: each grid point one event, holding its chord
# as an unordered pitch multiset, alongside the point's own time. The
# onset attribute locates a cadence tonic and is dropped before any
# density is built, so its width never enters; the tuple size, mode, and
# periodicity are the sweep's, which build_maet overrides per call.
pm = mpt.pre_maet_from_attr_table(
    mpt.grid_attr_table(bwv347_notes(), GRID_STEP_QN),
    attributes=(dict(column='pitch', sigma=SIGMA_PITCH, r=1, exch=True,
                     is_per=True, period=PERIOD),
                dict(column='onset', sigma=1.0)),
    time='beats', weights='ones')
p_attr, _, specs = mpt.unpack_pre_maet(pm)
onsets = p_attr[[spec['name'] for spec in specs].index('onset')][0]
tonic_events = {}                       # the event of each cadence final
for cid, t in sorted(TONIC_TIMES.items()):
    tonic_events[cid] = int(np.argmin(np.abs(onsets - t)))
    chord = sorted(p_attr[0][:, tonic_events[cid]].tolist(), reverse=True)
    print(f'  tonic C{cid}: {chord}')

# 2 x 6 landscape: rows are absolute / relative; the left half
# (cols 0-2) is non-periodic and the right half (cols 3-5) periodic,
# with r ascending 1 -> 3 within each half. The two relative r = 1
# cells are degenerate (a single pitch has no relative content) and
# are omitted. Cadence names sit on each row's leftmost visible cell
# and each column's lowest visible cell; r headers sit on the top row.
mode_rows = [(False, 'Absolute'), (True, 'Relative')]
halves = [False, True]                          # non-periodic, periodic
if plt is None:
    print('matplotlib not available; skipping the figure.')
    mpt.set_default(**_prev_defaults)
    raise SystemExit(0)

fig, axes = plt.subplots(2, 6, figsize=(12.9, 5.4))
for row, (is_rel, row_lab) in enumerate(mode_rows):
    for half, is_per in enumerate(halves):
        for ri, r in enumerate(R_VALUES):
            col = half * 3 + ri
            ax = axes[row, col]
            if is_rel and r == 1:               # degenerate: omit
                ax.axis('off')
                continue
            # All pairwise tonic similarities: each tonic is a
            # single-event density (one pitch attribute, exch = 1), and
            # the 4 x 4 matrix (unit diagonal, symmetric) comes from one
            # cartesian density-list call.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')   # relative r = 1: constant
                dens = [build_maet(
                    mpt.select_pre_maet(pm, attributes=['pitch'],
                                        events=[tonic_events[c]]),
                    sigma=[SIGMA_PITCH], r=[r], rel=[is_rel],
                    is_per=[is_per], period=[PERIOD], verbose=False)
                    for c in sorted(tonic_events)]
            M = sim_maet(dens, dens, mode='cartesian', verbose=False)
            ax.imshow(M, cmap='viridis', vmin=0, vmax=1, aspect='equal')
            for i in range(4):
                for j in range(4):
                    ax.text(j, i, f'{M[i, j]:.2f}', ha='center',
                            va='center', fontsize=10,
                            color='white' if M[i, j] < 0.5 else 'black')
            ax.set_xticks(range(4)); ax.set_yticks(range(4))
            # x labels on each column's lowest visible cell (cols 0 and
            # 3 have their relative r = 1 cell omitted, so they carry
            # the cadence names on the absolute row)
            x_edge = (row == 1) or (col in (0, 3))
            ax.set_xticklabels(CADENCE_NAMES if x_edge else [], fontsize=12)
            # y labels on each row's leftmost visible cell
            left_col = 1 if is_rel else 0
            ax.set_yticklabels(CADENCE_NAMES if col == left_col else [],
                               fontsize=12)
            if row == 0:
                ax.set_title(f'$r$ = {r}', fontsize=14)
# half-headers spanning each block of three columns
fig.text(0.30, 0.885, 'Non-periodic', ha='center', fontsize=15)
fig.text(0.745, 0.885, 'Periodic', ha='center', fontsize=15)
fig.suptitle(
    'BWV 347 cadence-tonic pair similarity (single chords), '
    '$\\sigma_{\\mathrm{pitch}} = 15$ cents\n'
    'single pitch attribute, exch 1 (unordered chord); cosine in $[0,1]$, '
    'diagonals $1$, symmetric; relative $r = 1$ omitted (degenerate); '
    'major/minor separation appears at $r = 3$',
    fontsize=13, y=0.995,
)
fig.subplots_adjust(left=0.065, right=0.99, top=0.80, bottom=0.085,
                    wspace=0.10, hspace=0.10)
# row labels at the left, vertically centred on each row
for row, (_, row_lab) in enumerate(mode_rows):
    pos = axes[row, 1].get_position()
    fig.text(0.014, pos.y0 + pos.height / 2, row_lab, rotation=90,
             va='center', ha='left', fontsize=15)
if SAVE_FIGURES:
    os.makedirs(FIG_DIR, exist_ok=True)
    out_png = os.path.join(FIG_DIR, 'demo_jmm_1_4_tonic_tuple_size.png')
    fig.savefig(out_png, dpi=140, bbox_inches='tight')
    print('Saved figures/demo_jmm_1_4_tonic_tuple_size.png')
else:
    plt.show()


# The demo leaves the toolbox as it found it: the defaults it set at the
# top are restored here.
mpt.set_default(**_prev_defaults)
