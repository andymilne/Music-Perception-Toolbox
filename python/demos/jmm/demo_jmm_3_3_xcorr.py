"""demo_jmm_3_3_xcorr.py — Analysis 3.3 (Section 4.3.3 of the JMM article).

A demo of the Music Perception Toolbox reproducing the analysis from the
JMM article; lightly edited from the article's own script. Data come
from jmm_data (BWV 347 read from the bundled MusicXML) or piano_phase
(the rendered Piano Phase voices); a figure is written to figures/ when
matplotlib is available.

Analysis 3.3: phase as lag in Reich's *Piano Phase*.

The query is a single canonical cell (Piano 1's repeating pattern); the context
is Piano 2's event stream. At each anchor (a Piano-1 cell boundary) the query is
translated in time by a lag tau spanning one whole cell, and the one-sided
matched-filter response against Piano 2 is read off (cos_sim_exp_tens with
normalize='oneSidedDenom', the Analysis-1.4 idiom). Collecting these rows gives
a cross-correlogram R(anchor, tau); its bright ridge tracks the running phase
offset between the two pianos, climbing the staircase 0 -> 12 pulses across the
piece. Secondary ridges a half-cell away are expected from the cell's
near-period-6 internal self-similarity.

Pre-MAET structure::

    attribute   sigma           rel   per
    ---------   --------------  ----  ----
    pitch       0.15 semitone   no    no
    time        0.015 s         no    no

    r = (1, 1); query = 12-event canonical cell; context = Piano 2;
    one-sided normalisation (divide by the query self-overlap).
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
mpt.set_default(show_hints=False, truncation_sigmas=4.0, kernel_precision='double')
from mpt import show_pre_maet, windowed_similarity

import piano_phase as pe

# --- parameters ------------------------------------------------------------
SIGMA_PITCH = 0.15
SIGMA_TIME  = 0.015                 # 15 ms
IOI         = pe.BASE_IOI
CELL_DUR    = pe.NC * IOI           # one cell in seconds
N_TAU       = 241                   # lag samples over one cell (resolves ~33 ms ridge)
ASTEP       = 2                     # anchor every ASTEP Piano-1 cells

is_rel   = [False, False]
is_per   = [False, False]
periods  = [0.0, 0.0]
sigma    = [SIGMA_PITCH, SIGMA_TIME]
r_vec    = [1, 1]

# --- query: one canonical cell (Piano 1's pattern), cell starting at t = 0 --
q_pitch = pe.CELL.astype(float).reshape(1, pe.NC)
q_time  = (np.arange(pe.NC) * IOI).reshape(1, pe.NC)
query_pattr = [q_pitch, q_time]

# --- context: Piano 2 -------------------------------------------------------
p2, t2 = pe.render_voice(2)

# --- anchors and lag grid ---------------------------------------------------
n_cells = pe.N_REPS_V1
anchors = np.arange(0, n_cells, ASTEP) * CELL_DUR
tau_grid = np.linspace(0.0, CELL_DUR, N_TAU, endpoint=False)

# Single windowed_similarity call producing the whole (anchor, lag)
# correlogram. The context (Piano 2) is localised by a rectangular window
# of full support 2 * HALF at each anchor; the query is translated to each
# lag. The output shape follows query_centres: an (anchors x N_TAU) matrix
# of absolute query centres gives an (anchors x N_TAU) response, with the
# context broadcast along the lag axis.
#
# Query placement. windowed_similarity translates the query so its mean on
# the time axis lands at query_centres[i, j]; the original loop translated
# the query by (a - tau) directly, i.e. landed its mean at (a - tau) + mu_q.
# So query_centres = (a - tau) + mu_q reproduces the original placement.
# Piano 2 leads, so retarding the query by tau makes the ridge read the
# running phase k directly.
HALF = CELL_DUR + 2 * IOI
mu_q = float(query_pattr[1].mean())
query_centres = (anchors[:, None] - tau_grid[None, :]) + mu_q   # (anchors, N_TAU)
show_pre_maet([p2.reshape(1, -1), t2.reshape(1, -1)], None,
              names=['pitch', 'onset'], sigma=sigma, is_rel=is_rel,
              is_per=is_per, period=periods, max_events=4)
show_pre_maet(query_pattr, None, names=['pitch', 'onset'], sigma=sigma,
              is_rel=is_rel, is_per=is_per, period=periods, max_events=4)

R = windowed_similarity(
    [p2.reshape(1, -1), t2.reshape(1, -1)], None,
    query_pattr, None,
    sigma, r_vec, is_rel, is_per, periods,
    anchors,
    query_centres=query_centres,
    context_window=(1.0, 2 * HALF),     # rectangle, full support 2 * HALF
    normalize='oneSidedDenom',
    window_attr=1,
    drop_window_attr=False,
    verbose=False,
)
# Reproduce the original sparse-context skip: blank anchors whose
# localisation window holds fewer than one full cell of context events.
ctx_counts = np.array([((t2 >= a - HALF) & (t2 <= a + HALF)).sum()
                       for a in anchors])
R[ctx_counts < pe.NC, :] = np.nan

# --- true lag staircase for overlay ----------------------------------------
true_lag = np.array([pe.lag_at(a / CELL_DUR) for a in anchors])

# --- figure ----------------------------------------------------------------
if plt is None:
    print('matplotlib not available; skipping the figure.')
    raise SystemExit(0)

fig, ax = plt.subplots(figsize=(13, 4.6))
extent = [anchors[0], anchors[-1], 0, pe.NC]
im = ax.imshow(R.T, origin='lower', aspect='auto', extent=extent,
               cmap='magma', vmin=0.0)
ax.plot(anchors, true_lag, color='#39d3ff', lw=1.1, alpha=0.8,
        label='true phase $k$')
for s in pe.shift_centre_times():
    ax.axvline(s, color='w', lw=0.4, alpha=0.3)
ax.set_xlabel(r'anchor time $\alpha$ (s)')
ax.set_ylabel('lag $\\tau$ (pulses)')
ax.set_yticks(range(0, pe.NC + 1, 2))
ax.set_title('Lag cross-correlation: canonical cell (Piano 1) vs Piano 2 '
             '(one-sided matched filter)')
ax.legend(loc='upper left', fontsize=12, framealpha=0.6)
fig.colorbar(im, ax=ax, label='matched-filter response', pad=0.06)
fig.tight_layout()
fig.savefig('figures/demo_jmm_3_3_xcorr.png', dpi=140, bbox_inches='tight')
print(f'saved; R range {np.nanmin(R):.3f}-{np.nanmax(R):.3f}, '
      f'{len(anchors)} anchors x {N_TAU} lags')
