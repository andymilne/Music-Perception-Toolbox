"""demo_jmm_3_2_texture.py — Analysis 3.2 (Section 4.3.2 of the JMM article).

A demo of the Music Perception Toolbox reproducing the analysis from the
JMM article; lightly edited from the article's own script. Data come
from jmm_data (BWV 347 read from the bundled MusicXML) or piano_phase
(the rendered Piano Phase voices); a figure is written to figures/ when
matplotlib is available.

Analysis 3.2: phase as local texture in Reich's *Piano Phase*.

The two pianos are pooled into a single (pitch, time) event stream --- the
voice label is *not* an attribute, so the measure reads the combined sounding
texture rather than either line on its own. A broad Gaussian localisation
window (s.d. 3 s) is swept over the piece; at each sweep centre the windowed
Renyi-2 entropy of the joint (pitch, time) density is read off. Because the
window is smooth and wide, there is no rectangular-edge artefact and every
window has ample mass.

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
    window: Gaussian (shape 0) on the time attribute, s.d. 3 s, centred at the
    sweep offset, time retained (drop_window_attr=False); estimator: Renyi-2.
"""

import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
plt.rcParams.update({'font.size': 15, 'axes.titlesize': 17, 'axes.labelsize': 15,
                     'xtick.labelsize': 13, 'ytick.labelsize': 13, 'figure.titlesize': 20,
                     'font.family': 'DejaVu Sans'})

import mpt
mpt.set_default(show_hints=False, truncation_sigmas=3.0, kernel_precision='double')
from mpt import show_pre_maet, windowed_entropy

import piano_phase as pe

# --- fixed parameters ------------------------------------------------------
SIGMA_PITCH = 0.15          # semitone (= 15 cents)
WIN_SD      = 3.0           # localisation-window s.d. (s)
PRUNE       = 4.0           # keep events within PRUNE * WIN_SD of the centre
N_SWEEP     = 300
SIGMAS_T    = [0.015, 0.100]   # coincidence (precedence/fusion window), redundancy
IOI         = pe.BASE_IOI

# --- two-voice surface, voices pooled --------------------------------------
pitch, onset, _voice = pe.render_piece()
t_lo, t_hi = onset.min(), onset.max()
edge = 2 * WIN_SD                                   # unreliable near the ends
centres = np.linspace(t_lo, t_hi, N_SWEEP)
phase_at = np.array([pe.lag_at(c / (pe.NC * IOI)) for c in centres])   # continuous lag


def sweep(sigma_t, show_input=False):
    """Windowed joint (pitch, time) Renyi-2 entropy at each sweep centre.

    A single windowed_entropy sweep: a Gaussian localisation window
    (shape 0) on the time axis (attribute index 1) modulates the event
    weights, with the time axis retained (drop_window_attr=False) so the joint
    (pitch, time) density is built and its Renyi-2 entropy returned. The
    window standard deviation WIN_SD maps to the variance-matched
    rectangular width 2*sqrt(3)*sd. The previous explicit prune to
    +/- PRUNE * WIN_SD is unnecessary here: the global truncation_sigmas
    (set to 3.0 above, tighter than PRUNE = 4.0) already zeros every event
    the prune would have removed, so the result is identical.
    """
    if show_input:
        show_pre_maet([pitch.reshape(1, -1), onset.reshape(1, -1)], None,
                      names=['pitch', 'onset'],
                      sigma=[SIGMA_PITCH, sigma_t], is_per=[False, False],
                      max_events=4)
    return windowed_entropy(
        [pitch.reshape(1, -1), onset.reshape(1, -1)], None,
        [SIGMA_PITCH, sigma_t], [1, 1],
        [False, False], [False, False], [0.0, 0.0],
        centres,
        context_window=(0.0, WIN_SD * 2.0 * np.sqrt(3.0)),
        method='renyi2',
        window_attr=1, drop_window_attr=False,
        verbose=False,
    )


H = {st: sweep(st, show_input=(i == 0))
     for i, st in enumerate(SIGMAS_T)}

# --- figure ----------------------------------------------------------------
if plt is None:
    print('matplotlib not available; skipping the figure.')
    raise SystemExit(0)

fig, axes = plt.subplots(2, 1, figsize=(13, 5.2), sharex=True)
notes = {0.015: ('sigma_t = 15 ms', 'coincidence: dips at k = 4, 6, 8'),
         0.100: ('sigma_t = 100 ms',
                 'redundancy: humps at k ~ 2-3, 9-10; valley at k ~ 6')}
shifts = pe.shift_centre_times()

for ax, st in zip(axes, SIGMAS_T):
    lab, note = notes[st]
    ax.axvspan(t_lo, t_lo + edge, color='#999', alpha=0.2)
    ax.axvspan(t_hi - edge, t_hi, color='#999', alpha=0.2)
    for s in shifts:
        ax.axvline(s, color='#c25008', lw=0.6, alpha=0.5)
    ax.plot(centres, H[st], color='#1f4eb8', lw=1.7)
    ax2 = ax.twinx()
    ax2.plot(centres, phase_at, color='#bbb', lw=0.9)
    ax2.set_yticks(range(0, 13, 3)); ax2.set_ylim(-1, 13)
    ax2.tick_params(colors='#999')
    interior = (centres > t_lo + edge) & (centres < t_hi - edge)
    ax.set_ylim(H[st][interior].min() - 0.1, H[st][interior].max() + 0.1)
    ax.set_ylabel(f'{lab}\nRenyi-2')
    ax.grid(True, alpha=0.3); ax.spines['top'].set_visible(False)

axes[-1].set_xlabel('window-centre offset (s); grey = phase $k$')
axes[0].set_xlim(t_lo, t_hi)
fig.suptitle('Texture entropy (voices pooled, Gaussian 3 s window): '
             'coincidence vs redundancy', y=1.0)
fig.tight_layout()
fig.subplots_adjust(hspace=0.10)
fig.savefig('figures/demo_jmm_3_2_texture.png', dpi=140, bbox_inches='tight')
print('saved figures/demo_jmm_3_2_texture.png')
