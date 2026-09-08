"""demo_jmm_3_1_diff.py — Analysis 3.1 (Section 4.3.1 of the JMM article).

A demo of the Music Perception Toolbox reproducing the analysis from the
JMM article; lightly edited from the article's own script. Data come
from jmm_data (BWV 347 read from the bundled MusicXML) or piano_phase
(the rendered Piano Phase voices); a figure is written to figures/ when
matplotlib is available.

Analysis 3.1: joint differencing on pitch and time in Reich's *Piano Phase*.

The phasing voice (Piano 2) is differenced jointly on pitch and time via
`difference_events` with per-attribute orders [1, 1, 0]: the pitch and onset
attributes are first-differenced to (dp, dt), while a third copy of the onset
attribute is passed through (order 0) to carry absolute time for the windowing
sweep (after alignment all three share the N-1 grid). A broad Gaussian window
is swept over the piece and the windowed Renyi-2 entropy of the (dp, dt)
density is read at each centre.

The analysis is run at two values of the time-difference kernel width:

  * sigma_t = 6 ms --- the just-noticeable difference for inter-onset
    intervals in an isochronous sequence (Friberg & Sundberg 1995). The
    accelerandi shift the IOI over a 135.4-137.8 ms range (a 2.4 ms
    excursion, below the JND), so at this width the (dp, dt) fingerprint is
    indistinguishable everywhere and the entropy is flat while the phase
    staircase climbs 0 -> 12 pulses --- the foil that motivates Analyses 3.2
    (phase as texture) and 3.3 (phase as lag).

  * sigma_t = 0.1 ms --- far below the JND. At this super-human resolution
    the sub-JND IOI excursion is resolved: the entropy fluctuates strongly,
    rising where Piano 2's tempo is modulating (the accelerandi by which it
    advances its phase). This is voice 2's own tempo change becoming visible,
    not the inter-voice phase (which single-voice differencing quotients out).

Plotting both on a shared scale shows that matching sigma_t to the perceptual
JND is what aligns the analysis with what a listener hears.

Pre-MAET structure (after differencing)::

    attribute   order  sigma            rel    per
    ---------   -----  ---------------  -----  -----
    dp          1      0.5 semitone     no     no
    dt          1      6 ms / 0.1 ms    no     no
    abs onset   0      --- (window axis, deleted after weighting)

    r = (1, 1); estimator: windowed Renyi-2 (Gaussian window, s.d. 6 s).
"""
import os
import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
plt.rcParams.update({'font.size': 17, 'axes.titlesize': 19, 'axes.labelsize': 17,
                     'xtick.labelsize': 15, 'ytick.labelsize': 15, 'figure.titlesize': 22,
                     'font.family': 'DejaVu Sans'})

import mpt
from mpt import unpack_pre_maet
mpt.set_default(show_hints=False, truncation_sigmas=4.0, kernel_precision='double')
from mpt import (difference_events, build_exp_tens, eval_exp_tens,
                 show_pre_maet,
                 windowed_entropy)

import piano_phase as pe

SIGMA_DP    = 0.5            # semitones
SIGMA_JND   = 0.006         # seconds: IOI JND in an isochronous sequence
SIGMA_FINE  = 0.0001        # seconds: 0.1 ms, far below the JND
WINDOW_SD   = 6.0           # seconds; Gaussian time window for the entropy sweep
N_SWEEP     = 200
IOI         = pe.BASE_IOI

C_JND  = '#1f4eb8'          # blue  --- perceptual (JND-matched) line
C_FINE = '#8e2f9e'          # purple --- sub-JND (super-human) line

# --- joint differencing of the phasing voice ------------------------------
pitch, onset = pe.render_voice(2)
N = len(pitch)
p_attr = [pitch.reshape(1, N), onset.reshape(1, N), onset.reshape(1, N)]
# Per-attribute difference orders: pitch and onset first-differenced, the
# third (onset copy) passed through at order 0 as the windowing axis.
pd, wd, _ = unpack_pre_maet(difference_events(p_attr, None, [1, 1, 0]))
dp, dt, t_abs = pd[0].ravel(), pd[1].ravel(), pd[2].ravel()
print(f'Differenced events: {len(dp)}; dp distinct: '
      f'{sorted(set(np.round(dp).astype(int).tolist()))}')
print(f'IOI (=dt) min/max: {dt.min()*1000:.2f} / {dt.max()*1000:.2f} ms  '
      f'(excursion {(dt.max()-dt.min())*1000:.2f} ms)')

show_pre_maet([pd[0], pd[1]], None, names=['dp', 'dt'],
              sigma=[SIGMA_DP, SIGMA_JND], is_per=[False, False],
              max_events=4, decimals=3)

# --- (a) static (dp, dt) density over the whole voice (at the JND width) ---
static = build_exp_tens([pd[0], pd[1]], None, [SIGMA_DP, SIGMA_JND], [1, 1],
                        [False, False], [False, False], [0.0, 0.0],
                        verbose=False)
dp_grid = np.linspace(dp.min() - 2, dp.max() + 2, 200)
dt_grid = np.linspace(dt.min() - 0.04, dt.max() + 0.04, 120)
DP, DT = np.meshgrid(dp_grid, dt_grid)
Z = eval_exp_tens(static, np.column_stack([DP.ravel(), DT.ravel()]).T,
                  verbose=False).reshape(DP.shape)
print('static density evaluated')

# --- (b) windowed (dp, dt) Renyi-2 entropy across the piece, two widths ---
centres = np.linspace(t_abs.min(), t_abs.max(), N_SWEEP)
phase_at = np.array([pe.lag_at(c / (pe.NC * IOI)) for c in centres])   # continuous lag

def sweep(sig):
    """Windowed (dp, dt) Renyi-2 entropy at each sweep centre.

    A single windowed_entropy sweep: a Gaussian window (shape 0) on the
    absolute-onset axis (attribute index 2) modulates the event weights,
    and that onset axis is dropped from the entropy density
    (drop_window_attr=True), leaving the two-attribute (dp, dt) density
    whose Renyi-2 entropy is returned. The window standard deviation
    WINDOW_SD maps to the variance-matched rectangular width 2*sqrt(3)*sd.
    (The placeholder onset sigma is unused: that axis is dropped.)
    """
    return windowed_entropy(
        pd, wd,
        [SIGMA_DP, sig, 1.0], [1, 1, 1],
        [False, False, False], [False, False, False], [0.0, 0.0, 0.0],
        centres,
        context_window=(0.0, WINDOW_SD * 2.0 * np.sqrt(3.0)),
        method='renyi2',
        window_attr=2, drop_window_attr=True,
        verbose=False,
    )

H_jnd  = sweep(SIGMA_JND)
H_fine = sweep(SIGMA_FINE)
for tag, h in [('6 ms (JND)', H_jnd), ('0.1 ms', H_fine)]:
    print(f'sigma_t = {tag:>10}: entropy {np.nanmin(h):.3f}..{np.nanmax(h):.3f} '
          f'nats (range {np.nanmax(h)-np.nanmin(h):.4f})')

# --- figure ---------------------------------------------------------------
if plt is None:
    print('matplotlib not available; skipping the figure.')
    raise SystemExit(0)

fig, (axD, axH) = plt.subplots(1, 2, figsize=(15, 4.4),
                               gridspec_kw={'width_ratios': [2, 3]})

pcm = axD.contourf(DP, DT * 1000, Z, levels=24, cmap='magma')
axD.scatter(dp, dt * 1000, s=4, color='white', alpha=0.30, zorder=3)
axD.set_xlabel(r'$\Delta p$ (semitones)')
axD.set_xticks(np.arange(-10, 11, 2))
axD.set_ylabel(r'$\Delta t$ (ms)')
axD.set_title(r'Static $(\Delta p,\ \Delta t)$ density (whole voice)')
fig.colorbar(pcm, ax=axD, fraction=0.046, pad=0.04, label='density')

for s in pe.shift_centre_times():
    axH.axvline(s, color='#c25008', lw=0.7, alpha=0.5)
axH.plot(centres, H_jnd,  color=C_JND,  lw=1.9,
         label=r'$\sigma_t = 6$ ms (IOI JND): flat')
axH.plot(centres, H_fine, color=C_FINE, lw=1.6,
         label=r'$\sigma_t = 0.1$ ms: resolves tempo modulation')
ax2 = axH.twinx()
ax2.plot(centres, phase_at, color='#bbb', lw=1.0, zorder=0)
ax2.set_ylabel('phase $k$ (pulses)', color='#999')
ax2.set_yticks(range(0, 13, 3)); ax2.set_ylim(-1, 13); ax2.tick_params(colors='#999')
lo = min(np.nanmin(H_jnd), np.nanmin(H_fine))
hi = max(np.nanmax(H_jnd), np.nanmax(H_fine))
pad = 0.12 * (hi - lo)
axH.set_ylim(lo - pad, hi + pad)
axH.set_xlabel('window-centre offset (s); accelerandi marked orange, phase grey')
axH.set_ylabel('Renyi-2 entropy (nats)')
axH.set_title(r'Windowed $(\Delta p,\ \Delta t)$ entropy at two kernel widths')
axH.legend(loc='center left', framealpha=0.9, fontsize=14)
axH.grid(True, alpha=0.3)
axH.spines['top'].set_visible(False)
axD.spines[['top', 'right']].set_visible(False)

fig.tight_layout()
os.makedirs('figures', exist_ok=True)
fig.savefig('figures/demo_jmm_3_1_diff.png', dpi=140, bbox_inches='tight')
print('saved figures/demo_jmm_3_1_diff.png')
