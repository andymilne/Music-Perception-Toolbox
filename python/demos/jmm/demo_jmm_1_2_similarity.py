"""demo_jmm_1_2_similarity.py — Analysis 1.2: voice-aware versus voice-agnostic similarity across the pitch–pitch-class blend.

A demo of the Music Perception Toolbox reproducing the analysis from the
JMM article (Section 4.1.2, "Voice-aware versus voice-agnostic across
the pitch–pitch-class blend"), lightly edited from the article's own
scripts. Data come from jmm_data (BWV 347 read from the bundled
MusicXML); figures are written to figures/ when matplotlib is available.

What the analysis asks. When do two chords count as alike? Six chord
pairs from BWV 347 — an identical voicing, a bass octave shift, a full
re-voicing, root position against first inversion, and two
different-chord baselines — are compared under three encodings of
voice information, each embodying a different answer, across the
pitch–pitch-class continuum of Shepard's helix stretched or compressed.

How it is computed. Every pitch is routed through two attributes at
once: a periodic pitch-class attribute (sigma_pc = 50 cents, P = 1200)
and a non-periodic pitch-height attribute whose width sigma_ph is swept
from one semitone to several octaves — narrow, and pitches must agree
in octave to count as similar; wide, and pitch-class equivalence
dominates. Voice information enters in one of three ways:
  (i)   Voice-aware: one event per chord; each attribute holds the
        ordered (S, A, T, B) voicing — K = 4, [sym] = 0, r = 4 — so
        matching is voice by voice, a multiplicative AND across voices.
  (ii)  Simplex-voice: one event per note (N = 4 single-pitch events);
        pitch class and pitch height at r = 1, plus a voice attribute
        holding each note's vertex of a regular tetrahedron
        (``simplex_vertices(4)``, its three coordinates taken in order:
        [sym] = 0, r = 3, sigma_voice = 0.2), so matching accrues
        additive partial credit, voice by voice.
  (iii) Voice-agnostic: one event per chord; the four pitches as an
        unordered multiset (K = 4, r = 1) on both attributes; voice
        identity is not encoded.
Each chord's density is built once per encoding and sigma_ph
(``build_exp_tens``), and the six pair similarities come from one
batched ``cos_sim_exp_tens`` call on density lists (mode='pairwise').

An appendix figure (``--heatmaps``) extends the same three encodings to
every event of the chorale: N x N cosine-similarity matrices over the
272 sixteenth-note grid points, one ``cos_sim_exp_tens`` call in
mode='cartesian' per encoding, at three pitch-height widths.

Data: ``jmm_data.bwv347_grid``. Toolbox: ``build_exp_tens``,
``cos_sim_exp_tens``, ``simplex_vertices``. Runtime: seconds for the
sweep; a few minutes more for the heat maps.
"""
from __future__ import annotations
import os
import sys

import numpy as np
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

import mpt
mpt.set_default(show_hints=False)
from mpt import (build_exp_tens, cos_sim_exp_tens, show_pre_maet,
                 simplex_vertices)

from jmm_data import bwv347_grid, GRID_STEP_QN


# ---------------------------------------------------------------------------
# Parameters (cents; the article's tables quote the same values in semitones)
# ---------------------------------------------------------------------------
SIGMA_PC = 50.0
SIGMA_VOICE = 0.2
SIGMA_PHS = np.logspace(np.log10(100), np.log10(8000), 50)
SIGMA_PH_SNAPSHOTS = [200.0, 600.0, 3000.0]      # heat-map widths


# ---------------------------------------------------------------------------
# Load chorale, identify reference chord pairs by score time
# ---------------------------------------------------------------------------
times, pitches_satb, _ = bwv347_grid()
pitches_cents = mpt.transform_attributes(pitches_satb, None,
                                         ('midi', 'cents'))
N = len(times)


def event_at(t):
    return int(round(t / GRID_STEP_QN))


# Six reference chord pairs (label, t_i, t_j), times in the played-through
# chorale (bars 1–4 repeat, so events from bar 5 on sit 16 QN later than in
# the unexpanded score). Chord locations (bar.beat):
#   (a) A maj b3.b1 vs b16.b1            (d) A maj b3.b1 vs b3.b3
#   (b) E maj cad1 b2.b3 vs cad2 b4.b3   (e) E maj cad1 b2.b3 vs B min cad3 b12.b1
#   (c) E maj b2.b3 vs b2.b4             (f) E maj cad1 b2.b3 vs A maj cad4 b17.b1
REFERENCE_PAIRS = [
    ('(a) identical voicing', 9.0, 61.0),
    ('(b) bass octave shift', 7.0, 15.0),
    ('(c) full re-voicing',   7.0,  8.0),
    ('(d) root vs first inv', 9.0, 11.0),
    ('(e) E maj vs B min',    7.0, 45.0),
    ('(f) E maj vs A maj',    7.0, 65.0),
]
PAIR_COLOURS = ['#1f4eb8', '#2b8a3e', '#c25008', '#b03060', '#666666', '#aaaaaa']

V_SIMPLEX = simplex_vertices(4)


# ---------------------------------------------------------------------------
# Three encoding builders
# ---------------------------------------------------------------------------
def build_voice_aware(satb_cents, sigma_pc, sigma_ph):
    """Two attributes (PC, pitch height), each the ordered (S, A, T, B)
    voicing: K = 4 per event, [sym] = 0 (ordered), r = 4. Numerically
    identical to four r = 1 per-voice attributes."""
    voicing = np.asarray(satb_cents, dtype=float).reshape(4, 1)   # K=4, N=1
    p_attr = [voicing, voicing]
    return build_exp_tens(
        p_attr, None,
        [sigma_pc, sigma_ph],     # sigma per attribute
        [4, 4],                    # r per attribute
        [False, False],            # is_rel
        [True, False],             # is_per: PC periodic, pitch height not
        [1200.0, 0.0],             # period
        [False, False],            # is_sym = 0 -> ordered (voice-aware)
        verbose=False,
    )


def build_simplex_voice(satb_cents, sigma_pc, sigma_ph, sigma_voice=SIGMA_VOICE):
    """One event per voice (N = 4): PC and pitch-height attributes (r = 1)
    plus a voice attribute carrying the simplex vertex as an ordered
    categorical attribute (K = 3 coordinates, [sym] = 0, r = 3)."""
    s, a, t, b = satb_cents
    pitches = np.array([[s, a, t, b]], dtype=float)        # (1,4): K=1, N=4
    voice = np.asarray(V_SIMPLEX, dtype=float).T           # (3,4): 3 coords x 4 voices
    p_attr = [pitches, pitches, voice]
    return build_exp_tens(
        p_attr, None,
        [sigma_pc, sigma_ph, sigma_voice],
        [1, 1, 3],
        [False, False, False],
        [True, False, False],
        [1200.0, 0.0, 0.0],
        [True, True, False],       # voice attribute ordered ([sym] = 0)
        verbose=False,
    )


def build_voice_agnostic(satb_cents, sigma_pc, sigma_ph):
    """Two attributes (PC, pitch height): the four sounding pitches as an
    unordered multiset (K = 4) at r = 1; voice identity is not encoded."""
    p4 = np.asarray(satb_cents, dtype=float).reshape(4, 1)
    return build_exp_tens(
        [p4, p4], None,
        [sigma_pc, sigma_ph],
        [1, 1],
        [False, False],
        [True, False],
        [1200.0, 0.0],
        [True, True],              # unordered (moot at r = 1)
        verbose=False,
    )


PC_PH = ['pitch class', 'pitch height']
BUILDERS = [('Voice-aware encoding', build_voice_aware, PC_PH),
            (f'Simplex-voice encoding (σ$_{{voice}}$ = {SIGMA_VOICE})',
             build_simplex_voice, PC_PH + ['voice']),
            ('Voice-agnostic encoding', build_voice_agnostic, PC_PH)]


# The three encodings carry the same chord differently, so each is shown
# as the pre-MAET the cosine actually receives, on the cadence-1 tonic.
for _label, _builder, _names in BUILDERS:
    show_pre_maet(_builder(pitches_cents[event_at(7.0)], SIGMA_PC,
                           SIGMA_PHS[0]), names=_names, title=_label)
    print()


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
def sweep():
    """(3 encodings, 6 pairs, len(SIGMA_PHS)) cosine similarities."""
    # The six pairs draw on eight distinct chords, several shared between
    # pairs (the cadence-1 tonic at t = 7 QN appears in four of them), so
    # each chord's density is built once per encoding and sigma_ph and the
    # pair similarities are read from that cache.
    chord_times = sorted({t for _, ti, tj in REFERENCE_PAIRS for t in (ti, tj)})
    chords = {t: pitches_cents[event_at(t)] for t in chord_times}
    sims = np.zeros((len(BUILDERS), len(REFERENCE_PAIRS), len(SIGMA_PHS)))
    for sp_idx, sigma_ph in enumerate(SIGMA_PHS):
        for b_idx, (_, build, _) in enumerate(BUILDERS):
            dens = {t: build(chords[t], SIGMA_PC, sigma_ph) for t in chord_times}
            # One batched call per encoding: list-vs-list pairwise mode
            # returns all six pair similarities at once.
            sims[b_idx, :, sp_idx] = cos_sim_exp_tens(
                [dens[ti] for _, ti, _ in REFERENCE_PAIRS],
                [dens[tj] for _, _, tj in REFERENCE_PAIRS],
                mode='pairwise', verbose=False)
    return sims


def report_sweep(sims):
    def at(sigma):
        return int(np.argmin(np.abs(SIGMA_PHS - sigma)))
    cols = [100.0, 1200.0, 8000.0]
    print('cosine similarity at σ_ph = ' + ', '.join(f'{c:g}' for c in cols)
          + ' cents (σ_pc = 50 cents):')
    for b_idx, (title, _, _) in enumerate(BUILDERS):
        print(f'  {title.split(" (")[0]}')
        for p_idx, (label, _, _) in enumerate(REFERENCE_PAIRS):
            vals = '  '.join(f'{sims[b_idx, p_idx, at(c)]:5.3f}' for c in cols)
            print(f'    {label:24s} {vals}')


def plot_sweep(sims):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for ax, (title, _, _), S in zip(axes, BUILDERS, sims):
        for p_idx, ((label, _, _), colour) in enumerate(zip(REFERENCE_PAIRS, PAIR_COLOURS)):
            ax.plot(SIGMA_PHS, S[p_idx], color=colour, linewidth=2, label=label)
        ax.set_xscale('log')
        ax.set_xlim(SIGMA_PHS[0], SIGMA_PHS[-1])
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel('σ$_{ph}$ (cents)', fontsize=21)
        ax.set_title(title, fontsize=23)
        ax.tick_params(labelsize=18)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    axes[0].set_ylabel('cosine similarity', fontsize=21)
    axes[0].legend(loc='lower right', fontsize=13, framealpha=0.0)
    fig.suptitle(f'BWV 347 chord-pair similarity vs σ$_{{ph}}$ '
                 f'(σ$_{{pc}}$ = {SIGMA_PC:g} cents fixed)', fontsize=27, y=0.995)
    fig.tight_layout()
    os.makedirs('figures', exist_ok=True)
    fig.savefig('figures/demo_jmm_1_2_sweep.png', dpi=140, bbox_inches='tight')
    fig.savefig('figures/demo_jmm_1_2_sweep.pdf', bbox_inches='tight')
    plt.close(fig)
    print('Saved figures/demo_jmm_1_2_sweep.png')


# ---------------------------------------------------------------------------
# Appendix: N x N event-pair heat maps
# ---------------------------------------------------------------------------
def heatmaps():
    """{(sigma_ph, encoding index): (N, N) cosine-similarity matrix}."""
    out = {}
    for sigma_ph in SIGMA_PH_SNAPSHOTS:
        print(f'  σ_ph = {sigma_ph:g}: building {N} densities x 3 encodings ...')
        for b_idx, (_, build, _) in enumerate(BUILDERS):
            dens = [build(pitches_cents[i], SIGMA_PC, sigma_ph) for i in range(N)]
            # One cartesian-mode call per encoding returns the full N x N
            # matrix (unit diagonal, symmetric) directly.
            out[(sigma_ph, b_idx)] = cos_sim_exp_tens(dens, dens, mode='cartesian',
                                                      verbose=False)
    return out


def plot_heatmaps(maps):
    ref = [(lab[:3], event_at(ti), event_at(tj), c)
           for (lab, ti, tj), c in zip(REFERENCE_PAIRS, PAIR_COLOURS)]
    bar_downbeats = [1, 9, 17, 25, 33, 41, 49, 57, 65]
    tick_positions = [event_at(t) for t in bar_downbeats]
    tick_labels = [str(int((t - 1) / 4) + 1) for t in bar_downbeats]

    fig, axes = plt.subplots(3, 3, figsize=(16, 16), constrained_layout=True)
    for row, sigma_ph in enumerate(SIGMA_PH_SNAPSHOTS):
        for col, (title, _, _) in enumerate(BUILDERS):
            ax = axes[row, col]
            S = maps[(sigma_ph, col)]
            im = ax.imshow(S, cmap='magma', vmin=0, vmax=1, origin='lower',
                           aspect='equal')
            if row == 0:
                ax.set_title(f'{title}\n(σ$_{{ph}}$ = {sigma_ph:g} cents)', fontsize=21)
            else:
                ax.set_title(f'σ$_{{ph}}$ = {sigma_ph:g} cents', fontsize=21)
            for _, i, j, colour in ref:
                for (x, y) in ((j, i), (i, j)):
                    ax.plot(x, y, marker='s', color=colour, markersize=12,
                            fillstyle='none', markeredgewidth=2.0)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, fontsize=16)
            ax.set_yticks(tick_positions)
            if col == 0:
                ax.set_yticklabels(tick_labels, fontsize=16)
                ax.set_ylabel('event index $i$\n(ticks = bar number)', fontsize=19)
            else:
                ax.set_yticklabels([])
            if row == 2:
                ax.set_xlabel('event index $j$\n(ticks = bar number)', fontsize=19)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label('similarity', fontsize=19)
    cbar.ax.tick_params(labelsize=16)
    fig.suptitle(f'BWV 347 $N \\times N$ event similarity heat maps '
                 f'(N = {N} at $\\Delta = {GRID_STEP_QN:g}$ QN; '
                 f'σ$_{{pc}}$ = {SIGMA_PC:g} cents fixed).\n'
                 f'Rows: σ$_{{ph}}$ snapshots. Columns: encodings.', fontsize=24)
    os.makedirs('figures', exist_ok=True)
    fig.savefig('figures/demo_jmm_1_2_heatmaps.png', dpi=140, bbox_inches='tight')
    fig.savefig('figures/demo_jmm_1_2_heatmaps.pdf', bbox_inches='tight')
    plt.close(fig)
    print('Saved figures/demo_jmm_1_2_heatmaps.png')


if __name__ == '__main__':
    sims = sweep()
    report_sweep(sims)
    if plt is not None:
        plot_sweep(sims)
    if '--heatmaps' in sys.argv:
        maps = heatmaps()
        if plt is not None:
            plot_heatmaps(maps)
    if plt is None:
        print('matplotlib not available; figures skipped.')
