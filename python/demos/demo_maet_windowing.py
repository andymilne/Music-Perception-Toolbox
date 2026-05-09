"""demo_maet_windowing.py

MAET post-tensor windowing over time and pitch.

Pedagogical demonstration of four time-windowed pipelines for locating
a three-event motif in a longer context, plus one pitch-windowed
pipeline that interrogates harmonic and subharmonic relationships at
a fixed time offset:

  raw               Absolute per-event pitch. Only literal-pitch
                    repetitions of the motif register.

  differenced       Inter-event pitch intervals (via
                    ``difference_events``). Transposition-invariant;
                    time order matters.

  spectrum          Spectrum-enriched pitch (via ``add_spectra``,
                    harmonic partials). Literal matches plus
                    partial-shared transpositions (P4, P5).

  spectrum + decay  Spectrum enrichment with an exponential decay
                    applied to the context's per-event weights (query
                    weights left uniform), modelling memory fade for
                    older events. Late-motif matches gain relative
                    weight at the expense of early literal matches.

  pitch @ M4        Pitch-windowed scan at the time offset where the
                    query lands on M4 (A-B-C, the P5 transposition of
                    the D-E-F query). Both operands are spectrum-
                    enriched, so peaks in the pitch profile reveal the
                    intervallic structure of partial-sharing: the
                    principal peak sits at +700 cents (the P5 up to
                    M4's centroid) with smaller secondary peaks at
                    octave and subharmonic offsets.

  pitch x time      2-D version of the pitch row: sweep both time and
                    pitch offsets simultaneously. Row 5 is the
                    horizontal slice of this heatmap at t = M4.

For the first four rows, sweeps use the cross-correlation offset API
with the time group windowed and the pitch group unwindowed. Row 5
windows both groups simultaneously, fixing the time offset at the M4
centroid and sweeping pitch. Offsets are measured from each query's
unweighted centroid to the window centre (the default reference used
by ``windowed_similarity``), so offset zero sits on the query's own
geometric centre.

Reference-point choice. ``windowed_similarity`` offers two reference-
point options for the cross-correlation: the default (unweighted
column mean of the query's tuple centres) and a user-supplied fixed
reference. This demo uses the default throughout. Which method is
appropriate depends on how queries vary between runs: see USER_GUIDE.md
§3.1 "Post-tensor windowing" for a summary, and the demo
``demo_windowing_reference`` for a full analysis across slot-count,
slot-value, and slot-weight sweeps of both harmonic and non-harmonic
queries.

Note on ``differenceEvents`` and ``addSpectra``. The demo deliberately
does not show a combined "diff + spectrum" pipeline, because diff
renders spectral enrichment numerically invisible to the windowed
similarity. For consecutive events, the harmonic offset
``1200 * log2(k)`` is identical, so it cancels in the difference:
every partial slot carries the same pitch-interval value per
differenced event. The remaining slot-weight factor H_K(2*rho)^2
multiplies numerator and denominator of the similarity ratio equally
and cancels. The same argument holds for any per-event pre-weighting
applied before ``add_spectra`` -- an exponential time decay, a
``seqWeights`` recency profile, or anything else; that pre-weighting
propagates through ``difference_events`` via its rolling-product of
weights, but its interaction with the spectrum weights always
factorises slot-wise and cancels. So
    ``seqWeights -> add_spectra -> difference_events``
gives the same windowed-similarity profile as
    ``seqWeights -> difference_events``
and the spectrum step is a no-op in any diff pipeline.

The context is a monophonic sequence of 18 events over 8 s, carrying
four instances of an ascending two-step motif (intervals {+200, +100}
cents) on four different roots:

  M1  D4 E4 F4    (query)
  M2  E4 F4 G4    (same interval multiset, reversed time order)
  M3  G4 A4 Bb4   (up P4)
  M4  A4 B4 C5    (up P5)

Uses:
    build_exp_tens, eval_exp_tens, cos_sim_exp_tens, windowed_similarity,
    difference_events, add_spectra, convert_pitch.

Run from the repo root (or with the installed package):

    python python/demos/demo_maet_windowing.py

Requires: numpy, matplotlib, mpt.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# Allow running from the source tree without `pip install`.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import mpt


# ======================================================================
#  User-tweakable parameters
# ======================================================================

# --- Perceptual uncertainties -----------------------------------------
PITCH_SIGMA_CENTS = 12.0        # pitch perceptual uncertainty
TIME_SIGMA_SEC    = 0.030       # time perceptual uncertainty

# --- Window shape (see window_tensor / windowed_similarity) --------------
WIN_SIZE_TIME = 25              # time window effective sd, in TIME_SIGMA_SEC units  (≈ 0.75 s)
WIN_MIX       = 0.5             # 0 = Gaussian, 1 = rectangular, 0.5 = flat-topped

# --- Geometry ---------------------------------------------------------
PITCH_PERIODIC = False          # True  -> pitch-class wrapping (period 1200)
TIME_PERIODIC  = False          # True  -> metric-position wrapping (period BAR_SEC)
BAR_SEC        = 2.0

# --- Visualisation sweep parameters -----------------------------------
# Time step for the 1-D profile sweeps (rows 1-4). 10 ms is one third
# of sigma_t = 30 ms, fine enough to resolve the peak shape without
# excessive cost, and lands exactly on every motif time (0, 2.0, 4.0,
# 6.5) when anchored at 0.
TIME_STEP_1D = 0.01
OFFSET_LO    = -0.8
OFFSET_HI    =  7.8

# --- Spectrum enrichment (rows 3 and 4) ------------------------------
N_PARTIALS  = 12                # harmonic spectrum partials
ROLLOFF_EXP = 0.5               # power-law rolloff exponent  (partial n -> weight 1/n^rho)

# --- Context exponential decay (row 4) -------------------------------
# Per-event context weight = exp(-CONTEXT_DECAY_RATE * (t_end - t)).
# At rate 0.3 s^-1 over the 8-s piece, M1 events sit at ≈ 0.10 weight,
# M4 at ≈ 0.74, so the recency bias is visible without being extreme.
# The query is left with uniform weights.
CONTEXT_DECAY_RATE = 0.3

# --- Pitch windowing at M4 (row 5) -----------------------------------
# Both groups windowed simultaneously: time fixed at the M4 centroid
# offset, pitch swept across PITCH_OFFSET_RANGE. Query and context are
# both spectrum-enriched. The profile reveals the family of harmonic /
# subharmonic relationships between query and windowed-context pitches.
M4_TIME_OFFSET      = 6.5       # time offset from query centroid to M4 centroid
PITCH_OFFSET_RANGE  = 3600.0    # sweep +/- this (cents)
PITCH_STEP_1D       = 10.0      # cents per sample -- half the peak HWHM (~20 c)
WIN_SIZE_PITCH_EFF  = 400.0     # effective sd of pitch window (cents)

# --- 2-D pitch x time windowing (row 6) ------------------------------
# Same configuration as row 5, but sweep time as well as pitch. Row 5
# is the horizontal slice at t = M4_TIME_OFFSET.
# To line up the grid with the motif coordinates used in this example,
# TIME_STEP_2D must be a subdivision of 0.5 s (the GCD of motif times
# {0, 2, 4, 6.5}) and PITCH_STEP_2D a subdivision of 100 cents (GCD of
# motif pitch offsets {0, ±200, ±500, ±700, ±1200, ±2400}). To resolve
# peak shape as well as location, steps smaller than half the relevant
# sigma are needed (e.g. 0.01 s < TIME_SIGMA_SEC/2, 5 cents <
# PITCH_SIGMA_CENTS/2); finer grids are slower to compute.
PITCH_STEP_2D    = 100.0         # cents per pitch grid cell
TIME_STEP_2D     = 0.10          # seconds per time grid cell

# --- Inharmonic query spectrum (second 2-D plot) ---------------------
# The second 2-D heatmap uses the same harmonic context but a query
# with log-stretched partials (n-th partial at frequency ratio
# n**STRETCH_BETA, so partial spacing on a log-frequency axis is
# uniformly scaled by STRETCH_BETA). beta = 1.05 = 5% stretch makes
# every partial audibly inharmonic (partial 12 is 215 cents sharp of
# its harmonic value) without being extreme.
STRETCH_BETA     = 1.05

# --- Output filenames ------------------------------------------------
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FN_OUT          = os.path.join(OUT_DIR, 'demo_maet_windowing.png')
FN_OUT_PT       = os.path.join(OUT_DIR, 'demo_maet_pitch_time.png')


# ======================================================================
#  Musical example
# ======================================================================

# (t_sec, MIDI, role)
events = [
    (0.00, 62, 'bar 1 beat 1'),
    (0.50, 64, 'bar 1 beat 2'),
    (1.00, 65, 'M1 ends (D-E-F)'),
    (1.50, 64, 'passing tone'),
    (2.00, 64, 'bar 2 beat 1'),
    (2.50, 65, 'bar 2 beat 2'),
    (3.00, 67, 'M2 ends (E-F-G, reversed intervals)'),
    (3.25, 65, '1st of eighth pair'),
    (3.50, 64, '2nd of eighth pair'),
    (4.00, 67, 'bar 3 beat 1'),
    (4.50, 69, 'bar 3 beat 2'),
    (5.00, 70, 'M3 ends (G-A-Bb, up P4)'),
    (6.00, 72, 'bar 4 (after 1 s rest)'),
    (6.25, 70, '1st of eighth pair'),
    (6.50, 69, 'M4 start'),
    (7.00, 71, 'M4 middle'),
    (7.50, 72, 'M4 ends (A-B-C, up P5)'),
    (8.00, 74, 'resolution'),
]

context_time  = np.array([e[0] for e in events], dtype=np.float64)
context_midi  = np.array([e[1] for e in events], dtype=np.int64)
context_cents = mpt.convert_pitch(context_midi.astype(np.float64),
                                  'midi', 'cents')

# Query: first three events (D4, E4, F4)
query_time  = context_time[:3].copy()     # [0.0, 0.5, 1.0]
query_cents = context_cents[:3].copy()    # [6200, 6400, 6500]

# Motif annotations in OFFSET coordinates (shared across all three rows).
motif_offsets = [0.0, 2.0, 4.0, 6.5]
motif_labels  = ['M1', 'M2', 'M3', 'M4']
motif_roots   = ['D',  'E',  'G',  'A']

# Periods (zero means non-periodic inside buildExpTens / differenceEvents)
pitch_period = 1200.0 if PITCH_PERIODIC else 0.0
time_period  = BAR_SEC if TIME_PERIODIC  else 0.0


# ======================================================================
#  Density builders
# ======================================================================

def build_raw_maet(pitch_cents, time_sec):
    """Raw MAET with K=1, r=1 per attribute; each attribute its own group."""
    p_attr = [pitch_cents[np.newaxis, :], time_sec[np.newaxis, :]]
    return mpt.build_exp_tens(
        p_attr,
        None,
        [PITCH_SIGMA_CENTS, TIME_SIGMA_SEC],
        [1, 1],
        None,
        [False, False],
        [PITCH_PERIODIC, TIME_PERIODIC],
        [pitch_period, time_period],
        verbose=False,
    )


def build_diff_maet(pitch_cents, time_sec):
    """Differenced MAET: pitch -> inter-event intervals, time unchanged.

    Returns (dens, p_diff_attr), where p_diff_attr is the post-difference
    cell, useful for sanity checks and plotting.
    """
    p_attr = [pitch_cents[np.newaxis, :], time_sec[np.newaxis, :]]
    p_diff, w_diff = mpt.difference_events(
        p_attr, None, None,
        [1, 0],
        [pitch_period, time_period],
    )
    dens = mpt.build_exp_tens(
        p_diff,
        w_diff,
        [PITCH_SIGMA_CENTS, TIME_SIGMA_SEC],
        [1, 1],
        None,
        [False, False],
        [PITCH_PERIODIC, TIME_PERIODIC],
        [pitch_period, time_period],
        verbose=False,
    )
    return dens, p_diff


def build_spec_maet(pitch_cents, time_sec, event_weights=None,
                    stretch_beta=1.0):
    """Spectrum-enriched raw MAET: each event carries N_PARTIALS slots
    on the pitch attribute. Optional per-event weights are multiplied
    into the spectrum weights via add_spectra (all ones if None).

    stretch_beta controls the partial spacing: 1.0 = harmonic (default);
    values > 1 stretch partial spacings uniformly on a log-frequency
    axis (inharmonic). Implemented via add_spectra's 'stretched' mode.
    """
    n_events = pitch_cents.size
    if stretch_beta == 1.0:
        p_flat, w_flat = mpt.add_spectra(
            pitch_cents, event_weights,
            'harmonic', N_PARTIALS, 'powerlaw', ROLLOFF_EXP,
        )
    else:
        p_flat, w_flat = mpt.add_spectra(
            pitch_cents, event_weights,
            'stretched', N_PARTIALS, stretch_beta, 'powerlaw', ROLLOFF_EXP,
        )
    # Python add_spectra uses C-order ravel, so the flat output is
    # [event 0 partials..., event 1 partials..., ...]. Reshape to
    # (n_events, N_PARTIALS) then transpose to (N_PARTIALS, n_events)
    # for K=N_PARTIALS slots per event on the pitch attribute.
    p_mat = p_flat.reshape(n_events, N_PARTIALS).T
    w_mat = w_flat.reshape(n_events, N_PARTIALS).T
    dens = mpt.build_exp_tens(
        [p_mat, time_sec[np.newaxis, :]],
        [w_mat, None],
        [PITCH_SIGMA_CENTS, TIME_SIGMA_SEC],
        [1, 1],
        None,
        [False, False],
        [PITCH_PERIODIC, TIME_PERIODIC],
        [pitch_period, time_period],
        verbose=False,
    )
    # Also return the (K, N) pitch and weight matrices for plotting.
    return dens, p_mat, w_mat


# ======================================================================
#  Build densities
# ======================================================================

print('Building densities ...')

# Context exponential-decay weights (used for row 4).
# exp(-rate * (t_end - t)): oldest event -> small weight, newest -> 1.
context_decay_w = np.exp(
    -CONTEXT_DECAY_RATE * (context_time.max() - context_time)
)
# Query weights stay uniform throughout.

# --- Row 1: raw ------------------------------------------------------
dens_ctx_raw = build_raw_maet(context_cents, context_time)
dens_q_raw   = build_raw_maet(query_cents,   query_time)

# --- Row 2: differenced ---------------------------------------------
dens_ctx_diff, p_diff_ctx = build_diff_maet(context_cents, context_time)
dens_q_diff,   p_diff_q   = build_diff_maet(query_cents,   query_time)

# --- Row 3: spectrum (all ones) ------------------------------
dens_ctx_spec, p_spec_ctx, _ = build_spec_maet(
    context_cents, context_time, event_weights=None
)
dens_q_spec,   p_spec_q,   _ = build_spec_maet(
    query_cents, query_time, event_weights=None
)

# --- Row 4: spectrum + context decay --------------------------------
dens_ctx_spec_d, p_specd_ctx, _ = build_spec_maet(
    context_cents, context_time, event_weights=context_decay_w
)
# Query stays uniform-weighted (identical to row-3 query).
dens_q_spec_d = dens_q_spec


# ======================================================================
#  Sweep (same offset vector for all four pipelines)
# ======================================================================

# Time-only window spec: pitch group not windowed (size = Inf), time group
# raised-rectangular of effective sd WIN_SIZE_TIME * sigma_t, mix = WIN_MIX.
spec_time_only = {'size': [np.inf, WIN_SIZE_TIME], 'mix': [0.0, WIN_MIX]}

# Offset sweep. Offset 0 places the window on the query's unweighted
# centroid; because the four motifs occupy the same absolute positions
# in every pipeline, their offsets are identical across pipelines (at
# 0, 2.0, 4.0, 6.5) -- one offset vector serves all rows.
offset_sweep = np.arange(OFFSET_LO, OFFSET_HI + TIME_STEP_1D/2, TIME_STEP_1D)
offsets_2d = np.vstack([np.zeros_like(offset_sweep), offset_sweep])

print('Computing windowed similarity profiles ...')

profile_raw    = mpt.windowed_similarity(dens_q_raw,    dens_ctx_raw,    spec_time_only, offsets_2d, verbose=False)
profile_diff   = mpt.windowed_similarity(dens_q_diff,   dens_ctx_diff,   spec_time_only, offsets_2d, verbose=False)
profile_spec   = mpt.windowed_similarity(dens_q_spec,   dens_ctx_spec,   spec_time_only, offsets_2d, verbose=False)
profile_spec_d = mpt.windowed_similarity(dens_q_spec_d, dens_ctx_spec_d, spec_time_only, offsets_2d, verbose=False)

# Unwindowed baselines (one scalar each), for the reference line.
nowin_raw    = float(mpt.cos_sim_exp_tens(dens_q_raw,    dens_ctx_raw,    verbose=False))
nowin_diff   = float(mpt.cos_sim_exp_tens(dens_q_diff,   dens_ctx_diff,   verbose=False))
nowin_spec   = float(mpt.cos_sim_exp_tens(dens_q_spec,   dens_ctx_spec,   verbose=False))
nowin_spec_d = float(mpt.cos_sim_exp_tens(dens_q_spec_d, dens_ctx_spec_d, verbose=False))


# ======================================================================
#  Row 5: pitch windowing at a fixed time offset (M4)
# ======================================================================

# Both pitch and time groups windowed simultaneously. Time offset is
# fixed at the M4 centroid; pitch offset sweeps across +/-3 octaves.
# Using the spectrum-enriched densities from row 3 (all ones).
# The pitch window's effective sd is WIN_SIZE_PITCH_EFF cents, expressed
# in PITCH_SIGMA_CENTS units for the `size` field.
win_size_pitch = WIN_SIZE_PITCH_EFF / PITCH_SIGMA_CENTS
spec_both = {'size': [win_size_pitch, WIN_SIZE_TIME],
             'mix':  [WIN_MIX,        WIN_MIX]}

pitch_offset_sweep = np.arange(-PITCH_OFFSET_RANGE,
                                PITCH_OFFSET_RANGE + PITCH_STEP_1D/2,
                                PITCH_STEP_1D)
offsets_pitch_2d = np.vstack([
    pitch_offset_sweep,
    np.full_like(pitch_offset_sweep, M4_TIME_OFFSET),
])

print('Computing pitch-windowed profile at M4 ...')

profile_pitch = mpt.windowed_similarity(
    dens_q_spec, dens_ctx_spec, spec_both, offsets_pitch_2d, verbose=False
)


# ======================================================================
#  Row 6: 2-D pitch x time windowing
# ======================================================================

# Same window spec as row 5; sweep both pitch and time offsets.
pitch_offset_2d = np.arange(-PITCH_OFFSET_RANGE,
                             PITCH_OFFSET_RANGE + PITCH_STEP_2D/2,
                             PITCH_STEP_2D)
N_PITCH_SWEEP_2D = pitch_offset_2d.size
# Time grid at TIME_STEP_2D seconds per cell, spanning OFFSET_LO to
# OFFSET_HI. Alignment: OFFSET_LO = -0.8 and TIME_STEP_2D = 0.05
# together put a sample exactly at each motif time (0, 2.0, 4.0, 6.5).
time_offset_2d  = np.arange(OFFSET_LO, OFFSET_HI + TIME_STEP_2D/2,
                             TIME_STEP_2D)
N_TIME_SWEEP_2D = time_offset_2d.size
# Flatten to a (2, M) offsets matrix. Order: inner-loop on pitch, outer
# on time, so reshape back to (N_TIME, N_PITCH) with the time axis first.
PP_2D, TT_2D = np.meshgrid(pitch_offset_2d, time_offset_2d, indexing='xy')
offsets_pt_2d = np.vstack([PP_2D.ravel(), TT_2D.ravel()])

print(f'Computing 2-D pitch x time sweep '
      f'({N_PITCH_SWEEP_2D} x {N_TIME_SWEEP_2D} = '
      f'{offsets_pt_2d.shape[1]} points, harmonic query) ...')

import time as _time
_t0 = _time.perf_counter()
profile_pt_flat = mpt.windowed_similarity(
    dens_q_spec, dens_ctx_spec, spec_both, offsets_pt_2d, verbose=False
)
print(f'  done in {_time.perf_counter() - _t0:.1f} s')
profile_pt = profile_pt_flat.reshape(N_TIME_SWEEP_2D, N_PITCH_SWEEP_2D)

# -- Inharmonic query: same fundamentals, but stretched partials. Context
#    is left harmonic (dens_ctx_spec reused). The same offsets grid is
#    used: offset is a relative transposition of the context window
#    relative to the query's unweighted tuple-centre mean (mu_q).
#    mu_q differs between harmonic and stretched queries (by ~144 cents
#    here), so offset = 0 refers to a different absolute pitch in the
#    two plots. The peak *positions* are still musically interpretable
#    because they arise from partial alignment, not from mu_q itself.
dens_q_stretch, p_qstretch, _ = build_spec_maet(
    query_cents, query_time, event_weights=None,
    stretch_beta=STRETCH_BETA,
)

print(f'Computing 2-D pitch x time sweep '
      f'(query partials stretched, beta = {STRETCH_BETA:g}) ...')
_t0 = _time.perf_counter()
profile_pt_stretch_flat = mpt.windowed_similarity(
    dens_q_stretch, dens_ctx_spec, spec_both, offsets_pt_2d, verbose=False,
)
print(f'  done in {_time.perf_counter() - _t0:.1f} s')
profile_pt_stretch = profile_pt_stretch_flat.reshape(
    N_TIME_SWEEP_2D, N_PITCH_SWEEP_2D
)


# ======================================================================
#  Main figure  (5 rows; column 1 = context density, column 2 = query
#  density, column 3 = similarity profile)
# ======================================================================

print('Drawing figure ...')


def eval_density_grid(dens, p_grid, t_grid):
    """Evaluate a MAET density on a pitch x time grid."""
    PP, TT = np.meshgrid(p_grid, t_grid, indexing='xy')
    X = np.vstack([PP.ravel(), TT.ravel()])
    vals = mpt.eval_exp_tens(dens, X)
    return vals.reshape(TT.shape)


def plot_time_profile_panel(ax, profile, with_roots=False):
    """Time-offset similarity profile."""
    ymax = float(profile.max()) * 1.15

    ax.plot(offset_sweep, profile, '-', lw=1.6, color='C0')
    ax.set_xlabel('window offset from query centroid (s)')
    ax.set_ylabel('windowed similarity')
    ax.set_xlim(offset_sweep[0], offset_sweep[-1])
    ax.set_ylim(0.0, ymax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Motif annotations: light vertical guide lines, labels hanging from
    # the top inside the axes so the title above stays uncluttered.
    for off, name, root in zip(motif_offsets, motif_labels, motif_roots):
        ax.axvline(off, color='lightgrey', lw=0.8, zorder=0)
        label = f'{name} {root}' if with_roots else name
        ax.text(off, ymax, label, ha='center', va='top',
                fontsize=8, color=(0.25, 0.25, 0.25))
    ax.set_title('similarity profile')


def plot_pitch_profile_panel(ax, profile):
    """Pitch-offset similarity profile for row 5.

    Annotates musically interesting harmonic and subharmonic intervals
    with vertical ticks and top labels.
    """
    ymax = float(profile.max()) * 1.15

    ax.plot(pitch_offset_sweep, profile, '-', lw=1.6, color='C0')
    ax.set_xlabel('window offset from query centroid (cents)')
    ax.set_ylabel('windowed similarity')
    ax.set_xlim(pitch_offset_sweep[0], pitch_offset_sweep[-1])
    ax.set_ylim(0.0, ymax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Pitch ticks every 600 cents (every tritone; matches whole-tone,
    # minor third, P4, P5, m6, M6, m7, M7, octave spacings by
    # multiples of 600 or close to them).
    tick_span_cents = 600
    tick_max = int(np.floor(pitch_offset_sweep[-1] / tick_span_cents)) * tick_span_cents
    ticks = np.arange(-tick_max, tick_max + 1, tick_span_cents)
    ax.set_xticks(ticks)

    # Reference intervals (cents) with short labels. These are the
    # offsets at which partial-sharing is expected between a
    # harmonic-spectrum query and context.
    refs = [
        (-2400, '-2 oct'),
        (-1200, '-1 oct'),
        (  -700, '-P5'),
        (     0, '0'),
        (  +700, '+P5'),
        ( +1200, '+1 oct'),
        ( +2400, '+2 oct'),
    ]
    for off, lbl in refs:
        ax.axvline(off, color='lightgrey', lw=0.8, zorder=0)
        ax.text(off, ymax, lbl, ha='center', va='top',
                fontsize=8, color=(0.25, 0.25, 0.25))
    ax.set_title(f'pitch-windowed profile at t = {M4_TIME_OFFSET:g} s (M4)')


# --- Build figure using GridSpec ------------------------------------

from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

N_ROWS = 5
fig = plt.figure(figsize=(14, 3.2 * N_ROWS))
# Outer split: density-block | profile.
# The density-block holds the context and query panels glued together
# with a thin gap; the inner split is computed per row so that the
# query panel's horizontal scaling (seconds-per-inch) matches the
# context's.
gs_outer = GridSpec(N_ROWS, 2, figure=fig,
                    width_ratios=[6, 6],
                    hspace=0.6, wspace=0.15,
                    left=0.095, right=0.985, top=0.955, bottom=0.03)


def make_row(row_idx, dens_ctx, dens_q, pitch_for_range_ctx,
             pitch_for_range_q, pitch_axis_label,
             profile_fn, profile_args, extra_title=''):
    """Build one row: context density, query density, profile.

    The pitch y-range is shared between the two density panels so the
    query panel sits visually in the same coordinate system as the
    context; context and query heatmaps use a shared colour scale.
    The inner GridSpec splits the density block so context and query
    share the same seconds-per-inch (horizontal scaling).
    """
    # Shared pitch y-range (union of context and query, padded).
    p_vals_all = np.concatenate([
        np.asarray(pitch_for_range_ctx).ravel(),
        np.asarray(pitch_for_range_q).ravel(),
    ])
    p_lo = float(p_vals_all.min()) - 100.0
    p_hi = float(p_vals_all.max()) + 100.0

    # Context time range.
    t_lo_c = -0.3
    t_hi_c = float(context_time.max()) + 0.3

    # Query time range: pad the query's own extent.
    q_t_vals = query_time if 'diff' not in pitch_axis_label else p_diff_q[1].ravel()
    t_lo_q = float(q_t_vals.min()) - 0.3
    t_hi_q = float(q_t_vals.max()) + 0.3

    # Split the density-block cell so the two panels' widths are
    # proportional to their time extents. Tight wspace draws the two
    # panels together with only a thin visual separation.
    ctx_extent = t_hi_c - t_lo_c
    q_extent   = t_hi_q - t_lo_q
    gs_inner = GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_outer[row_idx, 0],
        width_ratios=[ctx_extent, q_extent],
        wspace=0.04,
    )

    # Grid steps. For rows whose events sit at multiples of 100 cents
    # and 0.25 s (raw, differenced), a 20-cent x 0.05-s grid anchored
    # at 0 samples every event at its exact centre, giving uniform
    # brightness across events. Spectrum-enriched rows have partials
    # at irrational cent offsets (1200 * log2(k)); we cannot align a
    # uniform grid to every partial, so we use a fine 4-cent step so
    # the worst-case mis-alignment is ~2 c ~ 0.17 sigma (under 5% of
    # peak).
    #
    # "has partials" is inferred from whether the density's pitch
    # attribute has K > 1 slots.
    has_partials = (dens_ctx.centres[0].shape[0] > 1)
    dp = 4.0 if has_partials else 20.0
    dt = 0.03 if has_partials else 0.05

    # Anchor pitch and time grids at 0, so multiples of dp / dt include
    # every event exactly (for the aligned rows), and the grid is
    # reproducible across rows.
    def _anchored_arange(lo, hi, step):
        i_lo = int(np.ceil(lo / step))
        i_hi = int(np.floor(hi / step))
        return np.arange(i_lo, i_hi + 1) * step

    p_grid   = _anchored_arange(p_lo,   p_hi,   dp)
    t_grid_c = _anchored_arange(t_lo_c, t_hi_c, dt)
    t_grid_q = _anchored_arange(t_lo_q, t_hi_q, dt)

    Z_c = eval_density_grid(dens_ctx, p_grid, t_grid_c)
    Z_q = eval_density_grid(dens_q,   p_grid, t_grid_q)
    vmax = max(float(Z_c.max()), float(Z_q.max()))

    ax_ctx = fig.add_subplot(gs_inner[0, 0])
    ax_q   = fig.add_subplot(gs_inner[0, 1], sharey=ax_ctx)
    ax_pr  = fig.add_subplot(gs_outer[row_idx, 1])

    # Context (time x pitch).
    # imshow's `extent` should match the grid's actual span, not the
    # bounding box requested; otherwise the pixel cells stretch.
    ax_ctx.imshow(Z_c.T, extent=[t_grid_c[0], t_grid_c[-1],
                                   p_grid[0],   p_grid[-1]],
                  aspect='auto', cmap='viridis', origin='lower',
                  vmin=0.0, vmax=vmax)
    ax_ctx.set_xlabel('time (s)')
    ax_ctx.set_ylabel(pitch_axis_label)
    ax_ctx.set_title(f'context density{extra_title}')

    # Query: same pitch y-range (shared), same time scaling (enforced
    # by the inner GridSpec width ratios).
    ax_q.imshow(Z_q.T, extent=[t_grid_q[0], t_grid_q[-1],
                                 p_grid[0],   p_grid[-1]],
                aspect='auto', cmap='viridis', origin='lower',
                vmin=0.0, vmax=vmax)
    ax_q.set_xlabel('time (s)')
    ax_q.tick_params(axis='y', labelleft=False)
    ax_q.set_title('query')

    profile_fn(ax_pr, *profile_args)
    return ax_ctx


def plot_pt_heatmap_panel(ax, profile_pt):
    """2-D pitch x time similarity heatmap for row 6.

    Rows are time offsets, columns are pitch offsets. Displayed with
    time on the x-axis and pitch offset on the y-axis.
    """
    # profile_pt has shape (N_TIME, N_PITCH). For imshow with time on
    # x and pitch on y, transpose so rows are pitch (y), cols are time (x).
    Z = profile_pt.T
    extent = [time_offset_2d[0], time_offset_2d[-1],
              pitch_offset_2d[0], pitch_offset_2d[-1]]
    im = ax.imshow(Z, extent=extent, aspect='auto', cmap='viridis',
                   origin='lower')
    ax.set_xlabel('time offset from query centroid (s)')
    ax.set_ylabel('pitch offset (cents)')
    ax.set_title('2-D pitch x time similarity')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Pitch ticks every 600 cents on y.
    tick_span = 600
    tick_max = int(np.floor(pitch_offset_2d[-1] / tick_span)) * tick_span
    ax.set_yticks(np.arange(-tick_max, tick_max + 1, tick_span))

    # Motif labels hanging from the top inside the axes. Light colour
    # so they read against the dark end of the viridis colormap.
    _, y1 = ax.get_ylim()
    for off, name in zip(motif_offsets, motif_labels):
        ax.text(off, y1, name, ha='center', va='top',
                fontsize=8, color='white')


# --- Rows ------------------------------------------------------------

q_diff_t = p_diff_q[1].ravel()

# Row 1: raw
ax_row1 = make_row(
    0, dens_ctx_raw, dens_q_raw,
    context_cents, query_cents, 'pitch (cents)',
    plot_time_profile_panel, (profile_raw, False),
)

# Row 2: differenced
ax_row2 = make_row(
    1, dens_ctx_diff, dens_q_diff,
    p_diff_ctx[0].ravel(), p_diff_q[0].ravel(), 'pitch interval (cents)',
    plot_time_profile_panel, (profile_diff, False),
)

# Row 3: spectrum (all ones)
ax_row3 = make_row(
    2, dens_ctx_spec, dens_q_spec,
    p_spec_ctx.ravel(), p_spec_q.ravel(), 'pitch (cents)',
    plot_time_profile_panel, (profile_spec, False),
)

# Row 4: spectrum + context decay
ax_row4 = make_row(
    3, dens_ctx_spec_d, dens_q_spec_d,
    p_specd_ctx.ravel(), p_spec_q.ravel(), 'pitch (cents)',
    plot_time_profile_panel, (profile_spec_d, True),
)

# Row 5: pitch windowing at M4 (same densities as row 3).
ax_row5 = make_row(
    4, dens_ctx_spec, dens_q_spec,
    p_spec_ctx.ravel(), p_spec_q.ravel(), 'pitch (cents)',
    plot_pitch_profile_panel, (profile_pitch,),
    extra_title='  (pitch sweep at M4)',
)


# --- Suptitle, row labels --------------------------------------------

fig.suptitle(
    f'MAET windowing over time and pitch  '
    f'[pitch σ = {PITCH_SIGMA_CENTS:g} c, time σ = {TIME_SIGMA_SEC*1000:g} ms, '
    f'win size (t) = {WIN_SIZE_TIME}σ, mix = {WIN_MIX:g}]',
    fontsize=12,
)

row_descriptions = [
    'raw\n(per-event pitch)',
    'differenced\n(inter-event intervals)',
    f'spectrum\n({N_PARTIALS} partials, ρ = {ROLLOFF_EXP:g})',
    f'spectrum + decay\n(context rate = {CONTEXT_DECAY_RATE:g}/s)',
    f'pitch @ M4\n(win size (p) ≈ {WIN_SIZE_PITCH_EFF:g} c)',
]
for row_idx, label in enumerate(row_descriptions):
    ax_ref = fig.axes[row_idx * 3]  # first axis of each row
    bbox = ax_ref.get_position()
    y_centre = 0.5 * (bbox.y0 + bbox.y1)
    fig.text(0.018, y_centre, label, rotation=90,
             ha='center', va='center', fontsize=11, fontweight='bold')

fig.savefig(FN_OUT, dpi=120)
print(f'  wrote {FN_OUT}')


# ======================================================================
#  Second figure: 2-D pitch x time similarity, two query variants
# ======================================================================

print('Drawing pitch x time figure ...')

N_ROWS_PT = 2
fig2 = plt.figure(figsize=(14, 6.0 * N_ROWS_PT))   # tall rows for pitch axis
gs2_outer = GridSpec(N_ROWS_PT, 2, figure=fig2,
                     width_ratios=[6, 6],
                     hspace=0.25, wspace=0.15,
                     left=0.095, right=0.985, top=0.95, bottom=0.04)


def make_pt_row(fig_, gs_, row_idx, dens_ctx, dens_q, pitch_for_range_ctx,
                pitch_for_range_q, profile_pt):
    """Same structure as make_row, but with a 2-D heatmap profile.

    The spectrum configuration is not annotated per panel; it is
    described once in the row label on the left margin of the figure.
    """
    p_vals_all = np.concatenate([
        np.asarray(pitch_for_range_ctx).ravel(),
        np.asarray(pitch_for_range_q).ravel(),
    ])
    p_lo = float(p_vals_all.min()) - 100.0
    p_hi = float(p_vals_all.max()) + 100.0

    t_lo_c = -0.3
    t_hi_c = float(context_time.max()) + 0.3
    t_lo_q = float(query_time.min()) - 0.3
    t_hi_q = float(query_time.max()) + 0.3

    ctx_extent = t_hi_c - t_lo_c
    q_extent   = t_hi_q - t_lo_q
    gs_inner = GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_[row_idx, 0],
        width_ratios=[ctx_extent, q_extent],
        wspace=0.04,
    )

    dp = 4.0
    dt = 0.03

    def _anchored_arange(lo, hi, step):
        i_lo = int(np.ceil(lo / step))
        i_hi = int(np.floor(hi / step))
        return np.arange(i_lo, i_hi + 1) * step

    p_grid   = _anchored_arange(p_lo,   p_hi,   dp)
    t_grid_c = _anchored_arange(t_lo_c, t_hi_c, dt)
    t_grid_q = _anchored_arange(t_lo_q, t_hi_q, dt)

    Z_c = eval_density_grid(dens_ctx, p_grid, t_grid_c)
    Z_q = eval_density_grid(dens_q,   p_grid, t_grid_q)
    vmax = max(float(Z_c.max()), float(Z_q.max()))

    ax_ctx = fig_.add_subplot(gs_inner[0, 0])
    ax_q   = fig_.add_subplot(gs_inner[0, 1], sharey=ax_ctx)
    ax_pr  = fig_.add_subplot(gs_[row_idx, 1])

    ax_ctx.imshow(Z_c.T, extent=[t_grid_c[0], t_grid_c[-1],
                                  p_grid[0],   p_grid[-1]],
                  aspect='auto', cmap='viridis', origin='lower',
                  vmin=0.0, vmax=vmax)
    ax_ctx.set_xlabel('time (s)')
    ax_ctx.set_ylabel('pitch (cents)')
    ax_ctx.set_title('context density')

    ax_q.imshow(Z_q.T, extent=[t_grid_q[0], t_grid_q[-1],
                                 p_grid[0],   p_grid[-1]],
                aspect='auto', cmap='viridis', origin='lower',
                vmin=0.0, vmax=vmax)
    ax_q.set_xlabel('time (s)')
    ax_q.tick_params(axis='y', labelleft=False)
    ax_q.set_title('query')

    plot_pt_heatmap_panel(ax_pr, profile_pt)
    return ax_ctx


# Row A: harmonic query, harmonic context (reference).
make_pt_row(
    fig2, gs2_outer, 0,
    dens_ctx_spec, dens_q_spec,
    p_spec_ctx.ravel(), p_spec_q.ravel(),
    profile_pt,
)

# Row B: stretched query, harmonic context.
make_pt_row(
    fig2, gs2_outer, 1,
    dens_ctx_spec, dens_q_stretch,
    p_spec_ctx.ravel(), p_qstretch.ravel(),
    profile_pt_stretch,
)

fig2.suptitle(
    f'2-D pitch × time similarity: harmonic vs stretched query spectrum  '
    f'(context harmonic, {N_PARTIALS} partials, ρ = {ROLLOFF_EXP:g})',
    fontsize=12,
)

row_descriptions_pt = [
    'harmonic query\n(β = 1.0)',
    f'stretched query\n(β = {STRETCH_BETA:g})',
]
for row_idx, label in enumerate(row_descriptions_pt):
    ax_ref = fig2.axes[row_idx * 3]
    bbox = ax_ref.get_position()
    y_centre = 0.5 * (bbox.y0 + bbox.y1)
    fig2.text(0.018, y_centre, label, rotation=90,
              ha='center', va='center', fontsize=11, fontweight='bold')

fig2.savefig(FN_OUT_PT, dpi=120)
print(f'  wrote {FN_OUT_PT}')


# ======================================================================
#  Peak-location summary
# ======================================================================

def peak_near(offs, profile, off_target, radius):
    mask = np.abs(offs - off_target) <= radius
    return float(profile[mask].max()) if np.any(mask) else float('nan')


print()
print('Time-windowed peak similarity at motif offsets:')
hdr = ('name', 'root', 'offset', 'raw', 'diff', 'spec', 'spec+d')
print('  {:>4s}  {:>4s}  {:>7s}  {:>7s}  {:>7s}  {:>7s}  {:>7s}'
      .format(*hdr))
for i, name in enumerate(motif_labels):
    off = motif_offsets[i]
    print('  {:>4s}  {:>4s}  {:7.2f}  {:7.3f}  {:7.3f}  {:7.3f}  {:7.3f}'
          .format(name, motif_roots[i], off,
                  peak_near(offset_sweep, profile_raw,    off, 0.3),
                  peak_near(offset_sweep, profile_diff,   off, 0.3),
                  peak_near(offset_sweep, profile_spec,   off, 0.3),
                  peak_near(offset_sweep, profile_spec_d, off, 0.3)))

print()
print(f'Pitch-windowed peak similarity at reference intervals '
      f'(time fixed at M4, t = {M4_TIME_OFFSET:g} s):')
print('  {:>10s}  {:>7s}  {:>7s}'.format('interval', 'cents', 'sim'))
ref_intervals = [
    ('-2 oct', -2400), ('-1 oct', -1200), ('-P5', -700), ('-P4', -500),
    ('unison', 0), ('+P4', 500), ('+P5 (M4)', 700),
    ('+1 oct', 1200), ('+2 oct', 2400),
]
for lbl, cents in ref_intervals:
    print('  {:>10s}  {:+7d}  {:7.3f}'
          .format(lbl, cents,
                  peak_near(pitch_offset_sweep, profile_pitch, cents, 60.0)))

# Also report the single global max and its location.
i_max = int(np.argmax(profile_pitch))
print(f'\n  Global max at offset = {pitch_offset_sweep[i_max]:+.1f} cents  '
      f'(sim = {profile_pitch[i_max]:.3f})')

print('\nDone.')
