"""demo_jmm_3_1_texture.py — Analysis 3.1 (JMM article, Section 4.3.1):
phase as local texture in Piano Phase.

A demo of the Music Perception Toolbox reproducing the analysis from the
JMM article; lightly edited from the article's own script. Data come
from piano_phase (the rendered Piano Phase voices); the figures stay on
screen unless SAVE_FIGURES is set.

Analysis 3.1: phase as local texture in Reich's *Piano Phase*.

The two pianos are pooled into a single (pitch, time) event stream --- the
voice label is *not* an attribute, so the measure reads the combined sounding
texture rather than either line on its own. A broad Gaussian localization
window (s.d. 3 s) is swept over the piece; at each sweep value the windowed
Renyi-2 entropy of the joint (pitch, time) density is read off. Because the
window is smooth and wide, there is no rectangular-edge artefact, and away
from the ends of the piece (shaded in the figure) every window has ample
mass.

The single controlling parameter is the time kernel sigma_t:

  * sigma_t = 15 ms  --- within the precedence-effect fusion window (~5-30 ms),
    far narrower than the 138 ms pulse, so the
    density resolves every event. Only *exact vertical coincidences* (the two
    pianos striking the same pitch at the same instant) merge. This is the
    coincidence reading: a high plateau cut by dips wherever the canonical
    near-period-6 structure forces pitches to align (phase k = 4, 6, 8).

  * sigma_t = 100 ms --- a kernel about 0.7 of a pulse wide, so a pitch the two
    pianos play one pulse apart now overlaps and merges. This is the redundancy
    reading: the one-pulse canonic echo is counted as repetition, giving a
    smooth two-humped profile (peaks at the maximally de-correlated phases
    k ~ 2-3 and k ~ 9-10, a valley at the half-cycle k ~ 6, troughs at the
    unisons k = 0, 12).

Same surface, same window, same estimator; only the kernel width differs, and
that single change carries the reading from coincidence to redundancy.

Pre-MAET structure (both panels)::

    attribute   sigma            rel    per
    ---------   ---------------  -----  -----
    pitch       0.15 semitone    no     no
    time        sigma_t (s)      no     no

    r = (1, 1);  voices pooled (voice is not an attribute);
    window: Gaussian (shape 0) on the time attribute, s.d. 3 s, aligned at
    each sweep value, time retained (not dropped); estimator: Renyi-2.

Data: ``piano_phase`` (the rendered Piano Phase voices). Toolbox:
``pre_maet_from_attr_table``, ``swept_entropy``. Runtime: a few
seconds.
"""

import os
import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
if plt is not None:
    plt.rcParams.update({'font.size': 15, 'axes.titlesize': 17, 'axes.labelsize': 15,
                         'xtick.labelsize': 13, 'ytick.labelsize': 13, 'figure.titlesize': 20,
                         'font.family': 'DejaVu Sans'})

import mpt
_prev_defaults = mpt.set_default(show_hints=False)
from mpt import show_pre_maet, swept_entropy

import piano_phase as pe

# Set True to write the figures to a figures/ folder beside this script;
# False shows them instead.
SAVE_FIGURES = False
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')

# --- fixed parameters ------------------------------------------------------
SIGMA_PITCH = 0.15          # semitone (= 15 cents)
WIN_SD      = 3.0           # localization-window s.d. (s)
N_SWEEP     = 300
SIGMAS_T    = [0.015, 0.100]   # coincidence (precedence/fusion window), redundancy

# --- two-voice surface, voices pooled --------------------------------------
# Both pianos pooled, as an attribute table, converted to a pitch and a
# time attribute with one event per note. The time width is the sweep's,
# which swept_entropy overrides per call.
piece = mpt.pre_maet_from_attr_table(
    pe.piece_table(),
    attributes=(dict(column='pitch', sigma=SIGMA_PITCH),
                dict(column='onset', name='time', sigma=SIGMAS_T[0])),
    time='seconds', chords='separate', weights='ones')
onset = mpt.unpack_pre_maet(piece)[0][1].ravel()
t_lo, t_hi = onset.min(), onset.max()
edge = 2 * WIN_SD                                   # unreliable near the ends
sweep_values = np.linspace(t_lo, t_hi, N_SWEEP)
phase_at = pe.lag_at(sweep_values / (pe.NC * pe.BASE_IOI))      # continuous lag


def sweep(sigma_t, show_input=False):
    """Windowed joint (pitch, time) Renyi-2 entropy at each sweep value.

    A single swept_entropy sweep: a Gaussian localization window, with
    standard deviation WIN_SD, on the time attribute (attribute index 1)
    modulates the event weights, with the time attribute retained (not
    dropped) so the joint (pitch, time) density is built and its Renyi-2
    entropy returned. The window is truncated at
    truncation_sigmas standard deviations (the toolbox default, 6), so
    events far from the sweep value carry zero weight and need no separate
    pruning.
    """
    if show_input:
        show_pre_maet(piece, sigma=[SIGMA_PITCH, sigma_t], max_events=4)
    return swept_entropy(
        piece, sweep={1: sweep_values}, sigma=[SIGMA_PITCH, sigma_t],
        window={1: {'shape': 'gaussian', 'sd': WIN_SD}},
        method='renyi2', verbose=False,
    )


H = {st: sweep(st, show_input=(i == 0))
     for i, st in enumerate(SIGMAS_T)}

# --- figure ----------------------------------------------------------------
if plt is None:
    print('matplotlib not available; skipping the figure.')
    mpt.set_default(**_prev_defaults)
    raise SystemExit(0)

fig, axes = plt.subplots(2, 1, figsize=(13, 5.2), sharex=True)
labels = {0.015: r'$\sigma_t$ = 15 ms', 0.100: r'$\sigma_t$ = 100 ms'}
shifts = pe.shift_centre_times()

for ax, st in zip(axes, SIGMAS_T):
    lab = labels[st]
    ax.axvspan(t_lo, t_lo + edge, color='#999', alpha=0.2)
    ax.axvspan(t_hi - edge, t_hi, color='#999', alpha=0.2)
    for s in shifts:
        ax.axvline(s, color='#c25008', lw=0.6, alpha=0.5)
    ax.plot(sweep_values, H[st], color='#1f4eb8', lw=1.7)
    ax2 = ax.twinx()
    ax2.plot(sweep_values, phase_at, color='#bbb', lw=0.9)
    ax2.set_yticks(range(0, 13, 3)); ax2.set_ylim(-1, 13)
    ax2.tick_params(colors='#999')
    interior = (sweep_values > t_lo + edge) & (sweep_values < t_hi - edge)
    ax.set_ylim(H[st][interior].min() - 0.1, H[st][interior].max() + 0.1)
    ax.set_ylabel(f'{lab}\nRenyi-2 (bits)')
    ax.grid(True, alpha=0.3); ax.spines['top'].set_visible(False)

axes[-1].set_xlabel('window time (s); grey = phase $k$')
axes[0].set_xlim(t_lo, t_hi)
fig.suptitle('Texture entropy (voices pooled, Gaussian 3 s window): '
             'coincidence vs redundancy', y=1.0)
fig.tight_layout()
fig.subplots_adjust(hspace=0.10)
if SAVE_FIGURES:
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, 'demo_jmm_3_1_texture.png'),
                dpi=140, bbox_inches='tight')
    print('Saved figures/demo_jmm_3_1_texture.png')
else:
    plt.show()

# The demo leaves the toolbox as it found it: the defaults it set at the
# top are restored here.
mpt.set_default(**_prev_defaults)
