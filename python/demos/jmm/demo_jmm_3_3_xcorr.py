"""demo_jmm_3_3_xcorr.py — Analysis 3.3 (Online Supplement, Section 10):
phase as lag, a cross-correlogram.

A demo of the Music Perception Toolbox reproducing the analysis from the
JMM article; lightly edited from the article's own script. Data come
from piano_phase (the rendered Piano Phase voices); the figures stay on
screen unless SAVE_FIGURES is set.

Analysis 3.3: phase as lag in Reich's *Piano Phase*.

The query is a single canonical cell (Piano 1's repeating pattern); the context
is Piano 2's event stream. At each anchor alpha (a Piano-1 cell boundary) the
query is translated in time by a lag tau spanning one whole cell, and its
one-sided similarity against Piano 2 is read off (normalize='oneSidedDenom', as
in Analysis 1.3). Collecting these rows gives the supplement's cross-correlogram
R(alpha, tau) = <f_X, f_Y^(alpha - tau)> / <f_Y, f_Y>: a translation sweep of
the query by alpha - tau, which sweep_sim_maet computes in one pass. R = 1
where Piano 2 holds one cell-aligned copy of the query. The bright ridge
(R = 1) tracks the running phase k between the two pianos, climbing the
staircase 0 -> 12 pulses across the piece. Three fainter ridges (R = 0.5) run
at lags k + 4, k + 6, and k + 8 pulses (mod 12): rotated by 4, 6, or 8
positions, the cell agrees with itself at 6 of its 12 positions, and at
every other nonzero rotation at none of them. The maximum therefore fixes the
lag uniquely.

Pre-MAET structure::

    attribute   sigma           rel   per
    ---------   --------------  ----  ----
    pitch       0.15 semitone   no    no
    time        0.015 s         no    no

    r = (1, 1); query = 12-event canonical cell; context = Piano 2;
    one-sided normalization (divide by the query's self inner product).

Data: ``piano_phase`` (the rendered Piano Phase voices). Toolbox:
``pre_maet_from_attr_table``, ``pack_pre_maet``, ``build_maet``,
``sweep_sim_maet``. Runtime: a few seconds.
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
_prev_defaults = mpt.set_default(show_hints=False, truncation_sigmas=4.0)
from mpt import show_pre_maet, build_maet, sweep_sim_maet

import piano_phase as pe

# Set True to write the figures to a figures/ folder beside this script;
# False shows them instead.
SAVE_FIGURES = False
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')

# --- parameters ------------------------------------------------------------
SIGMA_PITCH = 0.15
SIGMA_TIME  = 0.015                 # 15 ms
IOI         = pe.BASE_IOI
CELL_DUR    = pe.NC * IOI           # one cell in seconds
N_TAU       = 241                   # lag samples over one cell (resolves ~33 ms ridge)
ASTEP       = 2                     # anchor every ASTEP Piano-1 cells

# --- query: one canonical cell (Piano 1's pattern), cell starting at t = 0 --
q_pitch = pe.CELL.astype(float).reshape(1, pe.NC)
q_time  = (np.arange(pe.NC) * IOI).reshape(1, pe.NC)
query_pattr = [q_pitch, q_time]   # its specs are the context's, below

# --- context: Piano 2 -------------------------------------------------------
context = mpt.pre_maet_from_attr_table(
    pe.voice_table(2),
    specs=(dict(column='pitch', sigma=SIGMA_PITCH),
                dict(column='onset', name='time', sigma=SIGMA_TIME)),
    time='seconds', chords='separate', weights='ones')

# --- anchors and lag grid ---------------------------------------------------
n_cells = pe.N_REPS_V1
anchors = np.arange(0, n_cells, ASTEP) * CELL_DUR
tau_grid = np.linspace(0.0, CELL_DUR, N_TAU, endpoint=False)

# The correlogram is a translation sweep: at anchor alpha and lag tau the
# query (the cell, written from t = 0) is translated in time by
# alpha - tau and compared with the whole of Piano 2. Piano 2 leads, so
# retarding the query by tau makes the ridge read the running phase k
# directly. sweep_sim_maet takes every (anchor, lag) offset in one pass:
# the pitch row of the offsets is zero (pitch is not translated), the time
# row holds alpha - tau, and the result is reshaped to (anchors, N_TAU).
offsets = anchors[:, None] - tau_grid[None, :]                   # (anchors, N_TAU)
show_pre_maet(context, max_events=4)
# The query is read under the context's geometry, so it carries the same
# specs.
query = mpt.pack_pre_maet(query_pattr, None,
                          mpt.unpack_pre_maet(context)[2])
show_pre_maet(query, max_events=4)

R = np.asarray(sweep_sim_maet(
    build_maet(context, verbose=False), build_maet(query, verbose=False),
    np.vstack([np.zeros(offsets.size), offsets.ravel()]),
    normalize='oneSidedDenom', verbose=False)).reshape(offsets.shape)
# Blank the anchors near the ends of the piece, where the span a cell can
# reach from the anchor (one cell and two pulses either side) holds fewer
# than one full cell of Piano 2's events.
HALF = CELL_DUR + 2 * IOI
ctx_onsets = mpt.unpack_pre_maet(context)[0][1].ravel()
ctx_counts = np.array([((ctx_onsets >= a - HALF)
                        & (ctx_onsets <= a + HALF)).sum()
                       for a in anchors])
R[ctx_counts < pe.NC, :] = np.nan

# --- true lag staircase for overlay ----------------------------------------
true_lag = pe.lag_at(anchors / CELL_DUR)

print(f'R range {np.nanmin(R):.3f}-{np.nanmax(R):.3f}, '
      f'{len(anchors)} anchors x {N_TAU} lags')

# --- figure ----------------------------------------------------------------
if plt is None:
    print('matplotlib not available; skipping the figure.')
    mpt.set_default(**_prev_defaults)
    raise SystemExit(0)

fig, ax = plt.subplots(figsize=(13, 4.6))
extent = [anchors[0], anchors[-1], 0, pe.NC]
im = ax.imshow(R.T, origin='lower', aspect='auto', extent=extent,
               cmap='magma', vmin=0.0)
ax.plot(anchors, true_lag, color='#39d3ff', lw=1.1, alpha=0.8,
        label='known phase $k$')
for s in pe.shift_centre_times():
    ax.axvline(s, color='w', lw=0.4, alpha=0.3)
ax.set_xlabel(r'anchor time $\alpha$ (s)')
ax.set_ylabel('lag $\\tau$ (pulses)')
ax.set_yticks(range(0, pe.NC + 1, 2))
# The title is raised clear of the tick labels and the colour bar, so a
# crop can remove it cleanly.
ax.set_title('Lag cross-correlation: canonical cell (Piano 1) vs Piano 2 '
             '(one-sided similarity)', pad=28)
ax.legend(loc='upper left', fontsize=12, framealpha=0.6)
fig.colorbar(im, ax=ax, label=r'one-sided similarity $R(\alpha, \tau)$', pad=0.06)
fig.tight_layout()
if SAVE_FIGURES:
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, 'demo_jmm_3_3_xcorr.png'),
                dpi=140, bbox_inches='tight')
    print('Saved figures/demo_jmm_3_3_xcorr.png')
else:
    plt.show()

# The demo leaves the toolbox as it found it: the defaults it set at the
# top are restored here.
mpt.set_default(**_prev_defaults)
