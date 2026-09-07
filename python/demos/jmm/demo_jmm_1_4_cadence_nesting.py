"""demo_jmm_1_4_cadence_nesting.py — Analysis 1.4: cadence localization with nested multisets.

A demo of the Music Perception Toolbox reproducing the analysis from the
JMM article ("Cadence localization using nested multisets"; Analysis 1.4
in the preprint's numbering), lightly edited from the article's own
scripts. Data come from jmm_data (BWV 347 read from the bundled
MusicXML); figures are written to figures/ when matplotlib is available.

What the analysis asks. Where in the chorale do cadences, and
cadence-like progressions, occur — of which type, and in any key? A
prototypical cadential progression (a short ordered run of chords) is
slid across the chorale, and its one-sided similarity to the local
harmony is read at every candidate resolution moment.

How it is computed. Query and context alike are single events whose
pitch attribute is a nested multiset two levels deep: the inner level is
each chord's pitch multiset ([sym] = 1, at inner tuple size r = 1, 2, or
3, the parameter this analysis varies), the outer level the chords in
order ([sym] = 0, r = the progression length). The comparison is taken
relative at the outer level alone ([rel] = (0, 1)), removing one common
transposition of the whole progression while leaving each chord's own
pitch classes absolute, and periodic at the octave (sigma = 0.15
semitones, P = 12). The chorale is reduced beat by beat to weighted
pitch aggregates — its two eighth-note events weighted by metrical
position (1 on the beat, 0.5 off it, times 1.5 under a fermata) and by
the fraction of the eighth each note sounds, merged and normalized by
the 1.5 a beat carries — and the aligned span of L consecutive beats,
the resolution on the last, is bound into one nested super-event
(``bind_events``) and compared with the query under the one-sided
similarity (``cos_sim_exp_tens(..., normalize='oneSidedDenom')``), so a
peak of 1 is one isolated exact match. An optional inversion flag — a
second, simplex-coded attribute at +/-0.5 with sigma_flag = 0.1 — marks
whether a chosen chord is a root-position triad (the dyad skeleton's
resolution) or a second-inversion triad (the six-four's antepenult); the
context's flag value is derived from the pitch content of the sonority
at that beat, no harmonic labels being consulted.

Eight queries are swept at inner r = 1, 2, 3: the tritone-to-major-third
dyad skeleton B–F -> C–E, plain and with the root-position flag on its
resolution; the maximal prototypes ii7–V7–I and its minor counterpart;
and the major and minor cadential six-fours, each plain and flagged.
Three figures are written from the one computation: the article's
five major-mode rows, the three minor-mode rows of the Online
Supplement, and all eight together. The dyad rows are blank at r = 3,
where a two-pitch chord has no inner triple.

The encodings live in bwv_window.py: the windowed chorale events, the
nested context and query builders, and the pitch-derived flags.
Toolbox: ``bind_events``, ``flat_specs``, ``build_exp_tens``,
``cos_sim_exp_tens``. Runtime: a minute or two (eight queries, three
inner tuple sizes, some sixty resolution moments each, every comparison a
nested inner product).
"""
from __future__ import annotations
import os
from collections import OrderedDict

import numpy as np
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from bwv_window import (cos_sim_exp_tens, win_events, aggregate, build_pair,
                        bound_density, query, son_at, is_root_position,
                        is_six_four, b2bar, T0, T1, ROOT_YES, ROOT_NO)

# Similarity normalization for all sweeps: 'oneSidedDenom' (query
# self-overlap alone) or 'cosine' (symmetric; penalizes context content
# the query does not match).
NORMALIZE = 'oneSidedDenom'

CAD = {7: 'C1', 15: 'C2', 23: "C1'", 31: "C2'", 45: 'C3', 65: 'C4'}
COL = '#1f4e79'
RS = (1, 2, 3)

# ---------------------------------------------------------------------------
# The six three-chord prototype queries (MIDI semitones, as in the article's
# tables): each chord enters as its distinct pitch classes, with no doubling.
# Doubling is material at inner r >= 2 (two unison-weighted entries of a
# pitch class are not one double-weighted entry: only the former contributes
# repeated-PC tuples), so entering doubled voicings would change the sweep
# values substantially. The flagged six-fours share their unflagged twins'
# pitch content — the inversion flag is the only difference.
# ---------------------------------------------------------------------------
QUERIES = OrderedDict([
    ('ii7-V7-I',   dict(chords=([62, 65, 69, 72], [55, 59, 62, 65],
                                [60, 64, 67]), flagged=False)),
    ('iio7-V7-i',  dict(chords=([62, 65, 68, 72], [55, 59, 62, 65],
                                [60, 63, 67]), flagged=False)),
    ('I-V-I',      dict(chords=([60, 64, 67], [62, 67, 71],
                                [60, 64, 67]), flagged=False)),
    ('Ic-V-I',     dict(chords=([60, 64, 67], [62, 67, 71],
                                [60, 64, 67]), flagged=True)),
    ('i-V-i',      dict(chords=([60, 63, 67], [62, 67, 71],
                                [60, 63, 67]), flagged=False)),
    ('ic-V-i',     dict(chords=([60, 63, 67], [62, 67, 71],
                                [60, 63, 67]), flagged=True)),
])

STEP = 1.0                                       # sweep step, QN (one beat)
MUS = np.arange(2.0, 67.0 + 1e-9, STEP)          # candidate resolution moments
WIN_BAR = b2bar(MUS)


def _query_density(spec, r_inner):
    aggs = [(np.asarray(c, float), np.ones(len(c))) for c in spec['chords']]
    flag = ROOT_YES if spec['flagged'] else None
    return bound_density(aggs, flag=flag, r_inner=r_inner)


def prototype_sweep(r_inner: int, normalize: str = NORMALIZE):
    """One-sided similarity profiles of all six three-chord queries at one
    inner r: {query name: (len(MUS),) profile}. Positions whose windows lack
    events score 0. The chorale's aligned span at resolution moment mu is
    the three beat aggregates [mu-2, mu-1), [mu-1, mu), [mu, mu+1). One
    batched density-list call per query."""
    ctx_plain, ctx_flag, idxs = [], [], []
    for k, mu in enumerate(MUS):
        if mu - 2.0 < T0 - 1e-9 or mu + 1.0 > T1 + 1e-9:
            continue
        wins = [win_events(mu - 2.0, mu - 1.0), win_events(mu - 1.0, mu),
                win_events(mu, mu + 1.0)]
        if any(len(w[0]) < 2 for w in wins):
            continue
        aggs = [aggregate(w) for w in wins]
        # The inversion flag is pitch-derived: a predicate on the sonority
        # at the antepenult beat (no harmonic labels are consulted).
        flag = ROOT_YES if is_six_four(son_at(mu - 2.0)) else ROOT_NO
        ctx_plain.append(bound_density(aggs, flag=None, r_inner=r_inner))
        ctx_flag.append(bound_density(aggs, flag=flag, r_inner=r_inner))
        idxs.append(k)
    out = {}
    for name, spec in QUERIES.items():
        qd = _query_density(spec, r_inner)
        ctx = ctx_flag if spec['flagged'] else ctx_plain
        vals = np.atleast_1d(cos_sim_exp_tens(ctx, qd, normalize=normalize,
                                              verbose=False))
        prof = np.zeros(len(MUS))
        prof[np.asarray(idxs, int)] = vals
        out[name] = prof
    return out


def dyad_sweep(r_inner: int, use_flag: bool):
    """One-sided similarity of the dyad-skeleton query against the chorale,
    swept over candidate resolution moments mu (every beat)."""
    qd = query(flag=(ROOT_YES if use_flag else None), r_inner=r_inner)
    so = np.full(len(MUS), np.nan)
    wins, idxs = [], []
    for k, mu in enumerate(MUS):
        if mu - 1.0 < T0 - 1e-9 or mu + 1.0 > T1 + 1e-9:
            continue
        # The two 1-QN halves either side of mu: the approach chord(s) in
        # [mu-1, mu) and the resolution chord(s) in [mu, mu+1).
        c1 = win_events(mu - 1.0, mu)
        c2 = win_events(mu, mu + 1.0)
        if len(c1[0]) < 2 or len(c2[0]) < 2:
            so[k] = 0.0
            continue
        # The optional inversion attribute is pitch-derived: a predicate on
        # the sonority sounding at mu (no harmonic labels are consulted).
        flag = None
        if use_flag:
            flag = ROOT_YES if is_root_position(son_at(mu)) else ROOT_NO
        wins.append(build_pair(c1, c2, flag=flag, r_inner=r_inner))
        idxs.append(k)
    # One batched call: density list vs single query, one-sided
    # (query-normalized) similarity.
    vals = np.atleast_1d(cos_sim_exp_tens(wins, qd, normalize=NORMALIZE,
                                          verbose=False))
    for k, v in zip(idxs, vals):
        so[k] = float(v)
    return WIN_BAR, so


qk = list(QUERIES.keys())
ROWS = [('d5/A4–M3/m6', 'dyad', False),
        ('d5/A4–*M3', 'dyad', True),
        (r'$\mathrm{ii}^7$–$\mathrm{V}^7$–$\mathrm{I}$', 'proto', qk[0]),
        (r'$\mathrm{ii}^{\varnothing 7}$–$\mathrm{V}^7$–$\mathrm{i}$', 'proto', qk[1]),
        (r'$\mathrm{I}$–$\mathrm{V}$–$\mathrm{I}$', 'proto', qk[2]),
        (r'*$\mathrm{I_c}$–$\mathrm{V}$–$\mathrm{I}$', 'proto', qk[3]),
        (r'$\mathrm{i}$–$\mathrm{V}$–$\mathrm{i}$', 'proto', qk[4]),
        (r'*$\mathrm{i_c}$–$\mathrm{V}$–$\mathrm{i}$', 'proto', qk[5])]

VARIANTS = [
    ('figures/demo_jmm_1_4_cadence_sweeps.pdf', [0, 1, 2, 4, 5]),    # the article's figure
    ('figures/demo_jmm_1_4_cadence_sweeps_minor.pdf', [3, 6, 7]),    # Online Supplement
    ('figures/demo_jmm_1_4_cadence_sweeps_all8.pdf', list(range(len(ROWS)))),
]


def compute():
    proto_prof = {r: prototype_sweep(r) for r in RS}
    data = {}
    for ri, (_, kind, spec) in enumerate(ROWS):
        for r in RS:
            if kind == 'dyad':
                data[(ri, r)] = None if r == 3 else dyad_sweep(r, spec)
            else:
                data[(ri, r)] = (WIN_BAR, proto_prof[r][spec])
    return data


def report(data):
    print('panel maxima (one-sided similarity; 1 = one isolated exact match),')
    print('at inner r = 1, 2, 3:')
    for ri, (lab, _, _) in enumerate(ROWS):
        ms = []
        for r in RS:
            d = data[(ri, r)]
            ms.append('   --  ' if d is None else f'{np.nanmax(d[1]):6.3f}')
        print(f'  {lab:48s}  ' + '  '.join(ms))
    # Where each query peaks at inner r = 2, the discriminating size.
    print('\nlocation of the r = 2 maximum (played-through bar):')
    for ri, (lab, _, _) in enumerate(ROWS):
        x, y = data[(ri, 2)]
        k = int(np.nanargmax(y))
        print(f'  {lab:48s}  bar {x[k]:5.2f}')


def plot(data):
    os.makedirs('figures', exist_ok=True)
    for out, row_ids in VARIANTS:
        n = len(row_ids)
        fig, axes = plt.subplots(n, 3, figsize=(11.5, 1.275 * n + 0.4), sharex=True)
        axes = np.atleast_2d(axes)
        for vi, ri in enumerate(row_ids):
            lab, kind, _ = ROWS[ri]
            for ci, r in enumerate(RS):
                ax = axes[vi, ci]
                d = data[(ri, r)]
                if d is None:                                # dyad at r = 3: blank cell
                    for s in ax.spines.values():
                        s.set_visible(False)
                    ax.xaxis.set_visible(False)
                    ax.yaxis.set_visible(False)
                    if vi == 0:
                        ax.set_title(f'Inner $r = {r}$', fontsize=15, pad=26)
                    continue
                x, y = d
                m = float(np.nanmax(y))
                top = m * 1.10 if m > 1e-3 else 1.0          # per-panel scale
                ax.set_ylim(0, top)
                ax.set_axisbelow(True)
                ax.grid(axis='y', color='0.90', lw=0.5)
                for bar in range(1, 19):
                    ax.axvline(bar, color='0.86', lw=0.6, zorder=0)
                for bt, nm in CAD.items():
                    xc = float(b2bar(bt))
                    ax.axvline(xc, color='0.4', lw=1.0, ls=(0, (4, 2)), zorder=1)
                    if vi == 0:
                        ax.text(xc, top * 1.02, nm, fontsize=11, ha='center',
                                va='bottom', color='0.3')
                ax.plot(x, y, lw=1.2, color=COL, zorder=3)
                ax.fill_between(x, np.nan_to_num(y), alpha=0.10, color=COL, zorder=2)
                ax.set_xticks([1, 5, 9, 13, 17])
                ax.set_xlim(1, 17.6)
                ax.tick_params(labelsize=12)
                if ci == 0:
                    ax.set_ylabel(lab, fontsize=13)
                if vi == 0:
                    ax.set_title(f'Inner $r = {r}$', fontsize=15, pad=26)
                if vi == n - 1:
                    ax.set_xlabel('bar', fontsize=13)
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.13)
        fig.savefig(out)
        fig.savefig(out.replace('.pdf', '.png'), dpi=150)
        plt.close(fig)
        print('Saved', out)


if __name__ == '__main__':
    data = compute()
    report(data)
    if plt is not None:
        plot(data)
    else:
        print('matplotlib not available; figures skipped.')
