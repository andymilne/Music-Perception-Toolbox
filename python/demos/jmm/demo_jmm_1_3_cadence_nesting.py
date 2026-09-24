"""demo_jmm_1_3_cadence_nesting.py — Analysis 1.3 (JMM article, Section
4.1.3; its minor-mode rows Online Supplement, Section 6): cadence
localization with nested multisets.

A demo of the Music Perception Toolbox reproducing the analysis from the
JMM article ("Cadence localization using nested multisets"), lightly
edited from the article's own
scripts. Data come from jmm_data (BWV 347 read from the bundled
MusicXML); the figures stay on screen unless SAVE_FIGURES is set.

What the analysis asks. Where in the chorale do cadences, and
cadence-like progressions, occur — of which type, and in any key? A
prototypical cadential progression (a short ordered run of chords) is
slid across the chorale, and its one-sided similarity to the local
harmony is read at every candidate resolution moment.

How it is computed. Query and context alike are single events whose
pitch attribute is a nested multiset two levels deep: the inner level is
each chord's pitch multiset ([exch] = 1, at inner tuple size r = 1, 2, or
3, the parameter this analysis varies), the outer level the chords in
order ([exch] = 0, r = the progression length). The comparison is taken
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
similarity (``windowed_similarity(..., normalize='oneSidedDenom')``), so a
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

The encodings live in bwv_window.py: the beat aggregates, the nested
context and query builders, and the pitch-derived flags.

Data: ``jmm_data.bwv347_notes``. Toolbox: ``grid_attr_table`` (twice, the
second regridding the first), ``pre_maet_from_attr_table``,
``bind_events`` (with per-attribute orders, so the window's time and flag
stay flat), ``windowed_similarity``, ``flat_specs``, ``select_pre_maet``.
Runtime: a few seconds (eight queries at three inner tuple sizes, each
sweep one call).
"""
from __future__ import annotations
import os
from collections import OrderedDict

import numpy as np

import mpt
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# The defaults this demo runs under, captured so that they can be put
# back at the end. They are set before bwv_window is imported, since
# that module builds the chorale's beat aggregates on import.
_prev_defaults = mpt.set_default(show_hints=False)

from bwv_window import (show_pre_maet, as_compared, bound_context, query,
                        dyad_query, b2bar, T0, T1, ROOT_YES)

# Set True to write the figures to a figures/ folder beside this script;
# False shows them instead.
SAVE_FIGURES = False
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')

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


def _prototype_query(spec, r_inner):
    """A prototype spec as a bound query; the one thing this adds over
    ``query`` is the mapping from the spec's flagged to the flag value."""
    return query(spec['chords'], flag=ROOT_YES if spec['flagged'] else None,
                 r_inner=r_inner)


def _window_starts(lead):
    """The window start times the sweep visits, and the MUS positions they
    belong to: a window of L beats resolving at mu starts lead beats before
    it, and must lie inside the piece."""
    keep = [(k, mu - lead) for k, mu in enumerate(MUS)
            if mu - lead >= T0 - 1e-9 and mu + 1.0 <= T1 + 1e-9]
    return np.array([k for k, _ in keep], int), np.array([t for _, t in keep])


def prototype_sweep(r_inner: int, normalize: str = NORMALIZE,
                    show_input: bool = False):
    """One-sided similarity profiles of all six three-chord queries at one
    inner r: {query name: (len(MUS),) profile}. Positions whose windows lack
    events score 0. The chorale's aligned span at resolution moment mu is
    the three beat aggregates [mu-2, mu-1), [mu-1, mu), [mu, mu+1). One
    windowed_similarity call per query."""
    # One bind_events call nests every window of three beats across the
    # whole chorale, carrying the window's own start time and the
    # inversion flag flat alongside the nested pitch. The sweep is then
    # one call: a rectangle of one beat admits exactly one window at each
    # centre. The flag is pitch-derived --- a predicate on the sonority at
    # the antepenult beat, which is the window's first, and no harmonic
    # labels are consulted.
    ctx_plain = bound_context(3, r_inner)
    ctx_flag = bound_context(3, r_inner, flag='six_four')
    idxs, centres = _window_starts(2.0)
    out = {}
    for name, spec in QUERIES.items():
        qd = _prototype_query(spec, r_inner)
        if show_input:
            show_pre_maet(as_compared(qd),
                          title=f'  query: {name} (r_inner = {r_inner})')
            print()
        # A rectangle of full support one beat, centred on each window's
        # own start time, admits exactly that window and no other --- its
        # neighbours sit exactly a beat away. The time axis (attribute 1)
        # is dropped from the comparison, having done its work in placing
        # the window.
        prof = np.zeros(len(MUS))
        prof[idxs] = np.asarray(mpt.windowed_similarity(
            ctx_flag if spec['flagged'] else ctx_plain, qd, centres,
            context_window=(1.0, 1.0), window_attr=1, drop_window_attr=True,
            normalize=normalize, verbose=False)).ravel()
        out[name] = prof
    return out


def dyad_sweep(r_inner: int, use_flag: bool):
    """One-sided similarity of the dyad-skeleton query against the chorale,
    swept over candidate resolution moments mu (every beat)."""
    qd = dyad_query(flag=(ROOT_YES if use_flag else None), r_inner=r_inner)
    # One bind_events call nests every pair of adjacent beats: the approach
    # beat [mu-1, mu) and the resolution beat [mu, mu+1). The optional
    # inversion attribute is pitch-derived --- a predicate on the sonority
    # sounding at mu, which is the window's second beat --- and no harmonic
    # labels are consulted.
    ctx = bound_context(2, r_inner,
                        flag='root_position_next' if use_flag else None)
    idxs, centres = _window_starts(1.0)
    so = np.full(len(MUS), np.nan)
    so[idxs] = np.asarray(mpt.windowed_similarity(
        ctx, qd, centres, context_window=(1.0, 1.0), window_attr=1,
        drop_window_attr=True, normalize=NORMALIZE, verbose=False)).ravel()
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
    ('demo_jmm_1_3_cadence_sweeps.pdf', [0, 1, 2, 4, 5]),    # the article's figure
    ('demo_jmm_1_3_cadence_sweeps_minor.pdf', [3, 6, 7]),    # Online Supplement
    ('demo_jmm_1_3_cadence_sweeps_all8.pdf', list(range(len(ROWS)))),
]


def compute():
    proto_prof = {r: prototype_sweep(r, show_input=(i == 0))
                  for i, r in enumerate(RS)}
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
    if SAVE_FIGURES:
        os.makedirs(FIG_DIR, exist_ok=True)
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
        if SAVE_FIGURES:
            path = os.path.join(FIG_DIR, out)
            fig.savefig(path)
            fig.savefig(path.replace('.pdf', '.png'), dpi=150)
            plt.close(fig)
            print('Saved figures/' + out)
        else:
            plt.show()


if __name__ == '__main__':
    data = compute()
    report(data)
    if plt is not None:
        plot(data)
    else:
        print('matplotlib not available; figures skipped.')

# The demo leaves the toolbox as it found it: the defaults it set at the
# top are restored here.
mpt.set_default(**_prev_defaults)
