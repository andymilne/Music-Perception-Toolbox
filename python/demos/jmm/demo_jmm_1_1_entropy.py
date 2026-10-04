"""demo_jmm_1_1_entropy.py — Analysis 1.1 (JMM article, Section 4.1.1):
windowed pitch entropy across BWV 347.

Reproduces Analysis 1.1 of the JMM article (Section 4.1.1): the temporal
evolution of pitch entropy across Bach's chorale BWV 347, read through a
window that slides along the piece.

What the analysis asks. Where in the chorale is the sounding pitch
content most and least concentrated? A cadence resolves onto a triad
whose partials cohere, so the spectral pitch density there is peaked and
its differential entropy low; passing sonorities between cadences spread
the density and raise the entropy. Read at every grid point and grouped
by metric class, the profile bears on the article's prediction that
spectral entropy (a model for dissonance) is on average higher at times
of lower metrical weight, which a permutation test then tests directly.

How it is computed. The chorale is gridded (``grid_attr_table``) and
converted to a two-attribute pre-MAET (pitch, time) in one call
(``pre_maet_from_attr_table``): each grid point is an event holding its
chord as an unordered pitch multiset in MIDI semitones (sigma = 0.1),
alongside the point's own time. Every chord is then spectrally enriched
(``add_spectra`` on that attribute: twelve harmonics, partial h at
p + 12 log2 h semitones weighted h^-0.67), so the pitch attribute carries
48 partials per event. ``swept_entropy`` sweeps a window along the
time attribute: at each sweep value the events are reweighted by the window
(event weighting, as ``weight_events`` does, the window factor
multiplied into the pitch weights), the time attribute is dropped, and
the differential entropy of the remaining pitch density is returned
(``method='differential'``: adaptive grid with Richardson extrapolation,
in bits, with pitch in semitones). Two windows are compared: a tight
rectangle of one sixteenth note (one event per window, so the profile is
the per-event entropy) and a Gaussian of standard deviation one quarter
note. The metric-class panels give each class's mean over its grid
points; their error bars are cluster-robust standard errors with the
sonority as the cluster, so that the grid points of a chord held across
several of them are not treated as independent observations.

The test. The prediction is tested on the rectangular-window profile,
one observation per sonority: its entropy and the metric class of its
onset, ranked downbeat > medium > weak > offbeat. The repeat of bars 1-4
is omitted, so that it does not count twice. The statistic is Kendall's
tau-b between rank and entropy, which the prediction makes negative; its
one-sided p-value comes from permuting the entropies among the
sonorities of each phrase (the stretches closed by the score's
fermatas), which leaves any difference between phrases intact.

Data: ``jmm_data.bwv347_notes`` (the bundled MusicXML read with
``read_score``, repeats expanded). Toolbox: ``grid_attr_table``,
``pre_maet_from_attr_table``, ``add_spectra``, ``swept_entropy``.
Runtime: under a minute (the differential estimator refines its grid at
every sweep value). The figures stay on screen unless SAVE_FIGURES is set.
"""
import os
import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:          # the numbers print without a figure
    plt = None
import time as _time

import mpt
_prev_defaults = mpt.set_default(show_hints=False)
from mpt import add_spectra, show_pre_maet, swept_entropy

# Set True to write the figures (and the checkpoint data) to a figures/ folder beside this
# script; False shows them instead.
SAVE_FIGURES = False
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')

from jmm_data import GRID_STEP_QN, bwv347_notes


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
SIGMA_PITCH = 0.1           # semitones (10 cents)
H_PARTIALS = 12
ROLLOFF = 0.67             # partial h weighted h^-0.67 (Milne et al. 2015)
SPECTRUM = ['harmonic', H_PARTIALS, 'powerlaw', ROLLOFF]

# Window specifications, as weight_events and swept_entropy take them.
# Each window is specified through one of two interchangeable parameters:
# `sd` (the window's standard deviation) or `width` (the full support of
# the rectangle at shape = 1). Across the shape family the standard
# deviation is held constant whichever parameter is supplied.
WINDOWS = [
    {'kind': 'width', 'value': 0.25,  # rect: full support 0.25 QN
     'shape': 1.0,
     'label': 'Rect, support 0.25 QN'},
    {'kind': 'sd', 'value': 1.0,      # Gaussian: sd = 1 QN
     'shape': 0.0,
     'label': 'Gaussian sigma = 1 QN'},
]


# ---------------------------------------------------------------------------
# Load chorale, spectral enrichment
# ---------------------------------------------------------------------------
print('Loading BWV 347 and enriching it spectrally...')
# The chorale on the sixteenth-note grid, converted to a two-attribute
# pre-MAET: the grid point's chord as the pitch attribute, read in MIDI
# semitones as an unordered multiset, and the grid point's own time as the
# attribute the window will slide along. Each chord's four pitches then take
# their partials (units=12: twelve units to the octave), which multiplies
# the pitch attribute's K by twelve and leaves the events alone.
grid = mpt.grid_attr_table(bwv347_notes(), GRID_STEP_QN)
pm = mpt.pre_maet_from_attr_table(
    grid,
    specs=(dict(column='pitch', sigma=SIGMA_PITCH, r=1, exch=True),
                dict(column='onset', name='time', sigma=1.0)),
    time='beats', pitch='midi', weights='ones')
# Each grid point's chord (its pitches sorted), read before enrichment: a
# run of consecutive grid points holding the same chord is one sonority.
chords = np.sort(mpt.unpack_pre_maet(pm)[0][0], axis=0)
new_chord = np.any(chords[:, 1:] != chords[:, :-1], axis=0)
sonority = np.concatenate([[0], np.cumsum(new_chord)])
pm = add_spectra(pm, *SPECTRUM, attribute='pitch', units=12.0)

times = mpt.unpack_pre_maet(pm)[0][1][0]
N = len(times)
t_end = times[-1] + GRID_STEP_QN

show_pre_maet(pm, max_events=4, max_elements=4, decimals=2)


# ---------------------------------------------------------------------------
# Compute differential entropy at each event time
# ---------------------------------------------------------------------------

print(f'Computing windowed differential entropy at {N} sweep values '
      f'over {len(WINDOWS)} window(s)...')

H = {wi: np.zeros(N) for wi in range(len(WINDOWS))}

# Each window is a single swept_entropy sweep over all sweep values.
# The time attribute (attribute 1) is the window attribute: it supplies
# the window and is dropped from the entropy density (drop=[1]; for an
# r = 1 absolute attribute, dropping it equals marginalizing it out),
# leaving the pitch density whose differential entropy is returned. The
# window is given as the specification above, by its width or by its
# standard deviation. (The placeholder time sigma is unused: the time
# attribute is dropped before any density is built.)
t0 = _time.time()
for wi, window in enumerate(WINDOWS):
    print(f'Window {wi + 1}/{len(WINDOWS)}: {window["label"]}')
    H[wi] = swept_entropy(
        pm, sweep={1: times}, drop=[1],
        window={1: {'shape': window['shape'],
                    window['kind']: window['value']}},
        method='differential', verbose=False)
    print(f'  done ({_time.time() - t0:.0f}s elapsed)')

for wi, window in enumerate(WINDOWS):
    arr = H[wi]
    finite = arr[np.isfinite(arr)]
    if finite.size:
        print(f'  {window["label"]}: '
              f'range [{finite.min():.4f}, {finite.max():.4f}] bits')

# Checkpoint: save H so the figure can be rebuilt without re-computing.
if SAVE_FIGURES:
    os.makedirs(FIG_DIR, exist_ok=True)
    np.savez(os.path.join(FIG_DIR, 'demo_jmm_1_1_H.npz'),
             times=times,
             H_rect=H[0],
             H_gauss=H[1],
             window_labels=np.array([w['label'] for w in WINDOWS]))


# ---------------------------------------------------------------------------
# Metric-class classification
# ---------------------------------------------------------------------------
def metric_class(t):
    if abs(t - round(t)) > 1e-6:
        return 'offbeat'
    beat_in_bar = ((int(round(t)) - 1) % 4) + 1
    return {1: 'downbeat', 3: 'medium'}.get(beat_in_bar, 'weak')


classes_per_event = np.array([metric_class(t) for t in times])
class_names = ['downbeat', 'medium', 'weak', 'offbeat']
class_labels = ['down\n(b1)', 'med\n(b3)', 'weak\n(b2,4)', 'off-\nbeat']
class_colours = ['#1f4eb8', '#5a8acb', '#a8b7d6', '#cccccc']


def class_stats(h, mask):
    """Mean of h over the grid points of one metric class, with its
    cluster-robust standard error (CR1), the sonority being the cluster:
    the residuals of the grid points holding one sonority are summed
    before squaring, so that a chord held across grid points is not
    counted as independent observations. With one grid point per
    sonority this is the ordinary standard error of the mean."""
    x = h[mask]
    _, g = np.unique(sonority[mask], return_inverse=True)
    n_clusters = int(g.max()) + 1
    if n_clusters < 2:
        return x.mean(), 0.0, n_clusters
    score = np.bincount(g, weights=x - x.mean())
    se = np.sqrt(n_clusters / (n_clusters - 1) * np.sum(score ** 2)) / x.size
    return x.mean(), se, n_clusters


print('Mean differential entropy (bits) by metric class, '
      '+/- cluster-robust SE (number of sonorities):')
stats = {}
for wi, window in enumerate(WINDOWS):
    print(f'  {window["label"]}')
    for cls in class_names:
        stats[wi, cls] = class_stats(H[wi], classes_per_event == cls)
        m, se, n = stats[wi, cls]
        print(f'    {cls:9s} {m:.4f} +/- {se:.4f} ({n})')


# ---------------------------------------------------------------------------
# Test of the prediction
# ---------------------------------------------------------------------------
# One observation per sonority, from the rectangular-window profile: its
# entropy (the same at every grid point it holds) and the metric class of
# its onset, ranked downbeat 4 > medium 3 > weak 2 > offbeat 1. The grid
# points from 16 to 32 QN repeat those from 0 to 16 QN (bars 1-4 and
# their upbeat), so they are omitted rather than counted twice.
REPEAT_SPAN = (16.0, 32.0)      # QN: the expanded repeat of bars 1-4
N_PERM = 20000
CLASS_RANK = {'downbeat': 4, 'medium': 3, 'weak': 2, 'offbeat': 1}

# Phrases: each closes with the last grid point of a fermata.
fermata_pts = np.zeros(N, dtype=bool)
fermata_pts[grid.loc[grid['fermata'].fillna(False).astype(bool),
                     'grid_index'].unique()] = True
phrase = np.concatenate([[0], np.cumsum(fermata_pts[:-1] & ~fermata_pts[1:])])

first = np.concatenate([[True], sonority[1:] != sonority[:-1]])
in_repeat = (times >= REPEAT_SPAN[0]) & (times < REPEAT_SPAN[1])
obs = np.flatnonzero(first & ~in_repeat)
rank = np.array([CLASS_RANK[c] for c in classes_per_event[obs]], dtype=float)
h_obs = H[0][obs]
phrase_obs = phrase[obs]

# Kendall's tau-b from the pairwise sign matrices. Permuting the
# entropies changes only the numerator: the tie counts, and so the
# denominator, stay fixed.
n_obs = obs.size
s_rank = np.sign(rank[:, None] - rank[None, :])
s_h = np.sign(h_obs[:, None] - h_obs[None, :])
n_pairs = n_obs * (n_obs - 1) / 2
untied_rank = n_pairs - (np.count_nonzero(s_rank == 0) - n_obs) / 2
untied_h = n_pairs - (np.count_nonzero(s_h == 0) - n_obs) / 2
concord = np.sum(s_rank * s_h) / 2
tau_b = concord / np.sqrt(untied_rank * untied_h)

# One-sided p-value: entropies permuted among the sonorities of each
# phrase, so that differences between phrases cannot produce the result.
rng = np.random.default_rng(347)
members = [np.flatnonzero(phrase_obs == k) for k in np.unique(phrase_obs)]
n_as_low = 0
for _ in range(N_PERM):
    perm = np.arange(n_obs)
    for m in members:
        perm[m] = m[rng.permutation(m.size)]
    n_as_low += np.sum(s_rank * s_h[np.ix_(perm, perm)]) / 2 <= concord
p_perm = (1 + n_as_low) / (1 + N_PERM)
print(f'Test (rectangular window, {n_obs} sonorities, repeat omitted): '
      f'Kendall tau-b = {tau_b:.3f}, one-sided permutation p = {p_perm:.2g} '
      f'({N_PERM} permutations within {len(members)} phrases)')


if plt is None:
    print('matplotlib not available; skipping the figure.')
    raise SystemExit(0)

# ---------------------------------------------------------------------------
# Plot: one row per window
# ---------------------------------------------------------------------------
n_rows = len(WINDOWS)
fig = plt.figure(figsize=(15, 4.0 * n_rows))
gs = fig.add_gridspec(n_rows, 2, width_ratios=[4, 1.0],
                      wspace=0.25, hspace=0.25)

cadence_spans = [
    (7.0,  8.0,  'tonic 1\n(E maj)'),
    (15.0, 16.0, 'tonic 2\n(E maj)'),
    (23.0, 24.0, "tonic 1'\n(E maj)"),
    (31.0, 32.0, "tonic 2'\n(E maj)"),
    (45.0, 48.0, 'tonic 3\n(B min)'),
    (65.0, 68.0, 'tonic 4\n(A maj)'),
]
bar_downbeats = np.arange(1, int(t_end) + 1, 4)

for wi, window in enumerate(WINDOWS):
    H_row = H[wi]
    # Window-edge band (Gaussian only; tight rect has no useful edge band).
    edge = 2 * window['value'] if window['shape'] == 0.0 else 0.0

    ax = fig.add_subplot(gs[wi, 0])
    ax_bar = fig.add_subplot(gs[wi, 1])
    for a in (ax, ax_bar):
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)

    for t in bar_downbeats:
        ax.axvspan(t - 0.06, t + 0.06, color='black', alpha=0.15, zorder=0)
        ax.axvspan(t + 1.94, t + 2.06, color='black', alpha=0.07, zorder=0)
    if edge > 0.0:
        ax.axvspan(0.0, edge, facecolor='#999999', alpha=0.22, zorder=0.5)
        ax.axvspan(t_end - edge, t_end,
                   facecolor='#999999', alpha=0.22, zorder=0.5)
    for t0_, t1_, _ in cadence_spans:
        ax.axvspan(t0_, t1_, color='#c25008', alpha=0.18, zorder=1)
    # Step plot for the rectangular per-event case, smooth plot for Gaussian.
    if window['shape'] == 1.0:
        ax.step(times, H_row, where='post', color='#1f4eb8',
                linewidth=1.2, zorder=2)
    else:
        ax.plot(times, H_row, color='#1f4eb8', linewidth=1.4, zorder=2)
    ax.set_xlim(0, t_end)
    ax.set_xticks(np.arange(1, int(t_end) + 1, 8))
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=15)
    ax.set_ylabel(f'{window["label"]}\n\ndifferential entropy (bits)',
                  fontsize=17)
    if wi == n_rows - 1:
        ax.set_xlabel('time (quarter notes)', fontsize=17)

    if wi == 0:
        ymin, ymax = ax.get_ylim()
        for t0_, t1_, lab in cadence_spans:
            ax.text((t0_ + t1_) / 2, ymin + (ymax - ymin) * 0.02, lab,
                    color='#c25008', fontsize=14, ha='center', va='bottom',
                    fontweight='bold')

    # Bar panel: metric-class means +/- cluster-robust SE
    means = [stats[wi, cls][0] for cls in class_names]
    sems = [stats[wi, cls][1] for cls in class_names]
    positions = np.arange(len(class_names))
    ax_bar.bar(positions, means, yerr=sems, color=class_colours,
               edgecolor='black', linewidth=0.7, capsize=4)
    ax_bar.set_xticks(positions)
    ax_bar.set_xticklabels(class_labels, fontsize=11)
    ax_bar.set_ylabel('mean ± SE', fontsize=17)
    span = max(means) - min(means)
    pad = max(sems) * 2 + span * 0.1 + 0.001
    ax_bar.set_ylim(min(means) - pad, max(means) + pad)
    ax_bar.grid(True, axis='y', alpha=0.3)
    ax_bar.tick_params(axis='y', labelsize=15)
    if wi == 0:
        ax_bar.set_title('By metric class', fontsize=19)

fig.suptitle(f'BWV 347 windowed differential pitch entropy '
             f'($\\sigma_{{pitch}}$ = {SIGMA_PITCH:g} semitones, '
             f'harmonic × {H_PARTIALS}, weight $h^{{-{ROLLOFF}}}$)',
             fontsize=20, y=0.995)
if SAVE_FIGURES:
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, 'demo_jmm_1_1_entropy.png'),
                dpi=140, bbox_inches='tight')
    print('Saved figures/demo_jmm_1_1_entropy.png.')
else:
    plt.show()

# The demo leaves the toolbox as it found it: the defaults it set at the
# top are restored here.
mpt.set_default(**_prev_defaults)
