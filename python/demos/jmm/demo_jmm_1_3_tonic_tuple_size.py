"""demo_jmm_1_3_tonic_tuple_size.py — Analysis 1.3 (Section 4.1.3 of the JMM article; the article calls it Analysis 1.2 in the reduced version).

A demo of the Music Perception Toolbox reproducing the analysis from the
JMM article; lightly edited from the article's own script. Data come
from jmm_data (BWV 347 read from the bundled MusicXML) or piano_phase
(the rendered Piano Phase voices); a figure is written to figures/ when
matplotlib is available.


Analysis 1.3: structural matching of the four cadence tonics of BWV 347
at increasing tuple size r, on a single chord (no nesting).

Each cadence tonic (the final chord of cadences C1-C4) is one event with
a single pitch attribute, the unordered chord multiset (sym = 1, K = 4).
The four tonics are compared pairwise under the cross of absolute vs
relative mode and non-periodic vs periodic, each swept over r in {1,2,3}.
Because the comparison is on one attribute (not a role product), raising
r tightens the match informatively rather than annihilating it: pitch
content (r = 1) -> dyad/interval content (r = 2) -> triad content (r = 3).

The payoff is the relative, periodic row. At r = 2 the interval-class
content cannot separate a major triad from a minor one (they are
inversionally related, and the unordered relative pair content is
inversion-invariant): the three major tonics and the minor tonic all read
as near-identical. At r = 3 the triadic structure separates them: the
three majors stay mutually 1 (transposition-equivalent) and the minor
isolates. Relative mode at r = 1 is a constant (degenerate) density and is
shown only for completeness.

These are the same four tonics compared as whole nested cadences in
Analysis 1.4, in the same panel layout, so the two figures read together.

Toolbox-dependency notes
------------------------
Uses
    build_exp_tens, cos_sim_exp_tens; bwv347_encoding.parse_bwv347.
"""
import os
import warnings
import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

import mpt
mpt.set_default(show_hints=False)
from mpt import build_exp_tens, cos_sim_exp_tens

from jmm_data import bwv347_grid


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


def extract_tonics():
    """Return dict cad_id -> (4,) pitch vector for each cadence tonic."""
    times, pitches_satb, _ = bwv347_grid()
    return {cid: pitches_satb[int(np.argmin(np.abs(times - t)))].astype(float)
            for cid, t in TONIC_TIMES.items()}


def tonic_density(chord, r, is_rel, is_per):
    """Single-event density for one chord: one pitch attribute, sym = 1."""
    p = [chord.reshape(len(chord), 1)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')   # relative r=1 -> constant density
        return build_exp_tens(p, None, [SIGMA_PITCH], [r], [is_rel],
                              [is_per], [PERIOD], verbose=False)


def similarity_matrix(tonics, r, is_rel, is_per):
    """All pairwise tonic similarities in one batched call."""
    dens = [tonic_density(tonics[c], r, is_rel, is_per) for c in sorted(tonics)]
    # List-vs-list cartesian mode returns the full 4 x 4 matrix (unit
    # diagonal, symmetric) in a single call.
    return cos_sim_exp_tens(dens, dens, mode='cartesian', verbose=False)


def main():
    print('Loading BWV 347 and extracting cadence tonics...')
    tonics = extract_tonics()
    for cid in sorted(tonics):
        print(f'  tonic C{cid}: {tonics[cid].tolist()}')

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
        return

    fig, axes = plt.subplots(2, 6, figsize=(12.9, 5.4))
    for row, (is_rel, row_lab) in enumerate(mode_rows):
        for half, is_per in enumerate(halves):
            for ri, r in enumerate(R_VALUES):
                col = half * 3 + ri
                ax = axes[row, col]
                if is_rel and r == 1:               # degenerate: omit
                    ax.axis('off')
                    continue
                M = similarity_matrix(tonics, r, is_rel, is_per)
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
        'single pitch attribute, sym 1 (unordered chord); cosine in $[0,1]$, '
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
    os.makedirs('figures', exist_ok=True)
    out_png = 'figures/demo_jmm_1_3_tonic_tuple_size.png'
    fig.savefig(out_png, dpi=140, bbox_inches='tight')
    print(f'Saved {out_png}')


if __name__ == '__main__':
    main()
