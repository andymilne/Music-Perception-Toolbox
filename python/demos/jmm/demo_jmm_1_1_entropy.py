"""demo_jmm_1_1_entropy.py — Analysis 1.1: windowed pitch entropy across BWV 347.

Reproduces Analysis 1.1 of the JMM article (Section 4.1.1): the temporal
evolution of pitch entropy across Bach's chorale BWV 347, read through a
window that slides along the piece.

What the analysis asks. Where in the chorale is the sounding pitch
content most and least concentrated? A cadence resolves onto a triad
whose partials cohere, so the spectral pitch density there is peaked and
its differential entropy low; passing sonorities between cadences spread
the density and raise the entropy. Read at every grid point and grouped
by metric class, the profile also shows the on-beat / off-beat contrast
that the Online Supplement tests across a corpus of chorales.

How it is computed. Every grid-point chord is spectrally augmented
(``add_spectra``: twelve harmonics with 1/n roll-off), so the pitch
attribute carries 48 partials per event. The chorale is then a two-
attribute pre-MAET (pitch, time). ``windowed_entropy`` sweeps a
window along the time attribute: at each centre the events are
reweighted by the window (``weight_events`` under the hood, the window
factor multiplied into the pitch weights), the time axis is dropped, and
the differential entropy of the remaining pitch density is returned
(``method='differential'``: adaptive grid with Richardson extrapolation,
in nats). Two windows are compared: a tight rectangle of one sixteenth
note (one event per window, so the profile is the per-event entropy) and
a Gaussian of one quarter note.

Data: ``jmm_data.bwv347_grid`` (the score sampled on the sixteenth-note
grid, repeats expanded). Toolbox: ``add_spectra``, ``windowed_entropy``.
Runtime: a few minutes (the differential estimator refines its grid at
every centre). A figure is written to ``figures/`` when matplotlib is
available.
"""
import os
import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:          # the numbers print without a figure
    plt = None
import time as _time

import mpt
mpt.set_default(
    show_hints=False,
    truncation_sigmas=3.0,          # truncate Gaussian tails at 3 sigma
    kernel_precision='single',      # 32-bit kernel arithmetic
)
from mpt import add_spectra, show_pre_maet, windowed_entropy

from jmm_data import bwv347_grid


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
SIGMA_PITCH = 10.0          # cents
H_PARTIALS = 12
ROLLOFF = 1.0
SPECTRUM = ['harmonic', H_PARTIALS, 'powerlaw', ROLLOFF]

# Window specifications for weight_events. Each window is specified through
# one of two interchangeable name-value arguments: `sd` (the window's
# standard deviation) or `width` (the full support of the rectangle at
# shape = 1). Across the shape family the SD is held constant regardless of
# which parameter the caller supplies.
WINDOWS = [
    {'kind': 'width', 'value': 0.25,  # rect: full support 0.25 QN
     'shape': 1.0,
     'label': 'Rect, support 0.25 QN'},
    {'kind': 'sd', 'value': 1.0,      # Gaussian: sd = 1 QN
     'shape': 0.0,
     'label': 'Gaussian sigma = 1 QN'},
]


# ---------------------------------------------------------------------------
# Load chorale, spectrally expand partials
# ---------------------------------------------------------------------------
print('Loading BWV 347 and expanding partials...')
times, pitches_satb, _ = bwv347_grid()
N = len(times)
t_end = times[-1] + 0.25
pitches_cents = mpt.transform_attributes(pitches_satb, None,
                                         ('midi', 'cents'))          # (N, 4)
K = 4 * H_PARTIALS                            # 48 partials per event

# add_spectra operates on one weighted multiset (one event) at a time,
# so the expansion runs as a per-event loop.
p_partials = np.zeros((N, K))
w_partials = np.zeros((N, K))
for n in range(N):
    p_aug, w_aug = add_spectra(pitches_cents[n], None, *SPECTRUM)
    p_partials[n] = p_aug
    w_partials[n] = w_aug

# Pre-MAET inputs: 2 attributes (pitch K=48 partials, time K=1 events).
# Pitch is attribute 0, time is attribute 1.
p_attr_pre = [p_partials.T, times.reshape(1, N)]
w_pre = [w_partials.T, np.ones((1, N))]


show_pre_maet(p_attr_pre, w_pre, names=['pitch', 'time'],
              sigma=[SIGMA_PITCH, 1.0], is_per=[False, False],
              max_events=4, max_elements=4, decimals=2)


# ---------------------------------------------------------------------------
# Compute differential entropy at each event time
# ---------------------------------------------------------------------------
sweep_centres = times.copy()
n_sweep = len(sweep_centres)

print(f'Computing windowed differential entropy at {n_sweep} sweep centres '
      f'over {len(WINDOWS)} window(s)...')

H = {wi: np.zeros(n_sweep) for wi in range(len(WINDOWS))}

# Each window is a single windowed_entropy sweep over all centres. The
# time axis (attribute 1) supplies the window and is dropped from the
# entropy density (drop_window_attr=True; for an r = 1 absolute axis,
# dropping the axis equals marginalising it out), leaving the pitch
# density whose differential entropy is returned. The
# window width is the rectangular full support; a Gaussian window given by
# its standard deviation s maps to the variance-matched width 2*sqrt(3)*s.
# (The placeholder time sigma is unused: the time axis is dropped
# before any density is built.)
RT3 = 2.0 * np.sqrt(3.0)
t0 = _time.time()
for wi, window in enumerate(WINDOWS):
    print(f'Window {wi + 1}/{len(WINDOWS)}: {window["label"]}')
    width = window['value'] if window['kind'] == 'width' else window['value'] * RT3
    H[wi] = windowed_entropy(
        p_attr_pre, w_pre,
        [SIGMA_PITCH, 1.0], [1, 1],
        [False, False], [False, False], [0.0, 0.0],
        sweep_centres,
        context_window=(window['shape'], width),
        method='differential',
        window_attr=1, drop_window_attr=True,
        verbose=False,
    )
    print(f'  done ({_time.time() - t0:.0f}s elapsed)')

for wi, window in enumerate(WINDOWS):
    arr = H[wi]
    finite = arr[np.isfinite(arr)]
    if finite.size:
        print(f'  {window["label"]}: '
              f'range [{finite.min():.4f}, {finite.max():.4f}] nats')

# Checkpoint: save H so the figure can be rebuilt without re-computing.
os.makedirs('figures', exist_ok=True)
np.savez('figures/demo_jmm_1_1_H.npz',
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
    ax.set_ylabel(f'{window["label"]}\n\ndifferential entropy (nats)',
                  fontsize=17)
    if wi == n_rows - 1:
        ax.set_xlabel('time (quarter notes)', fontsize=17)

    if wi == 0:
        ymin, ymax = ax.get_ylim()
        for t0_, t1_, lab in cadence_spans:
            ax.text((t0_ + t1_) / 2, ymin + (ymax - ymin) * 0.02, lab,
                    color='#c25008', fontsize=14, ha='center', va='bottom',
                    fontweight='bold')

    # Bar panel: metric-class means +/- SE
    means, sems = [], []
    for cls in class_names:
        mask = (classes_per_event == cls)
        n = int(mask.sum())
        means.append(H_row[mask].mean())
        sems.append(H_row[mask].std(ddof=1) / np.sqrt(n) if n > 1 else 0.0)
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
             f'($\\sigma_{{pitch}}$ = {SIGMA_PITCH:.0f} cents, '
             f'harmonic × {H_PARTIALS} with 1/n rolloff)',
             fontsize=20, y=0.995)
fig.savefig('figures/demo_jmm_1_1_entropy.png', dpi=140, bbox_inches='tight')
print('Saved figures/demo_jmm_1_1_entropy.png.')
