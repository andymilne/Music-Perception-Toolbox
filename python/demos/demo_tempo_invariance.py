"""demo_tempo_invariance.py

Anisotropic kernels for tempo tolerance and tempo invariance:
searching for a rhythmic motif in an onset stream with
`interval_kernel_cov` and `windowed_similarity`.

A matrix-valued kernel covariance (accepted wherever `sigma` is, on an
ordered, absolute, non-periodic, non-nested attribute whose tuple is
its whole multiset, r == K) lets one Gaussian kernel express several
independent sources of perceptual tolerance at once. The constructor
`interval_kernel_cov(r, sd_position, sd_interval, sd_shift)` builds the
covariance for an ordered tuple of r consecutive differences
(intervals) of r + 1 underlying positions:

    Sigma = sd_position**2 * D D^T      (tridiagonal: 2 / -1 / -1)
          + sd_interval**2 * I          (diagonal)
          + sd_shift**2   * ones((r,r)) (rank-one ridge)

Here the intervals are LOG inter-onset intervals (log-IOIs), for one
reason: a tempo change t -> a*t multiplies every IOI by a, which in log
coordinates is a common ADDITIVE shift of the whole tuple, log(a), along
the all-ones diagonal. That makes the three constructor terms three
musically distinct tolerances:

  sd_position   Uncertainty on the underlying positions whose
                consecutive differences are the tuple's intervals.
                Shared endpoints propagate it to the tridiagonal
                sd_position**2 * D D^T: displacing one interior
                position lengthens one interval and shortens its
                neighbour by the same amount. On log-IOIs this models
                onset-level timing jitter that scales with the local
                inter-onset interval (Weber-like motor noise); the
                shared-endpoint reading is exact when the adjacent
                intervals are equal and holds to first order otherwise.
  sd_interval   Independent uncertainty on each interval itself
                (central-timekeeper variance in the Wing &
                Kristofferson 1973 reading; on log-IOIs, proportional
                per-interval noise).
  sd_shift      Graded tolerance for a common shift of the whole
                tuple -- on log-IOIs, a TEMPO change. One sd is a
                tempo factor of exp(sd_shift). As sd_shift grows the
                kernel's precision tends to the relative-mode
                projector, so `is_rel=True` is the exact
                (infinite-sd_shift) limit: graded tempo TOLERANCE
                tends to exact tempo INVARIANCE.

The demo searches a monophonic onset stream for a long-short-short
motif. The stream contains variations of the motif at different tempos,
some with small onset-timing perturbations as well, plus two foils.
Each candidate rhythm occupies a 3-interval cell; the stream is scanned
by its overlapping log-IOI trigrams, each an ordered K = 3 value
multiset read at r = 3 (the matrix covariance requires r == K), with
an onset time as a second attribute that only places the sliding
window (`window_attr`, `drop_window_attr=True`); each trigram is timed
at the onset that completes its first interval, the stamp the
difference/bind pipeline gives it. The trigram
attribute must be ORDERED: one foil is the motif reversed, which has
the same interval multiset as the motif and is separated from it only
by slot order. The trigrams are built with the toolbox's cross-event
preprocessing -- `difference_events` (onsets to IOIs), then
`bind_events` (overlapping windows of three consecutive log-IOIs per
event). The rel kernel reads each trigram relative to a common shift;
a second `bind_events` call with `rel_outer=True` produces that
reading of the same trigrams.

Four sections:

  1. Material     The motif, the candidate cells, and the stream.
  2. Constructor  The three covariance terms, printed, and the price
                  each pure kernel puts on three canonical
                  perturbations of the motif.
  3. The search   `windowed_similarity` sweeps over every trigram
                  under six kernels; the candidate table contrasts
                  positional (timing) tolerance with tempo tolerance,
                  and both with exact tempo invariance.
  4. The limit    sd_shift -> infinity converges to `is_rel=True`.

Similarities throughout are `normalize='oneSidedDenom'`, which is 1 on
a self-match; for these single-trigram comparisons with a shared
covariance it equals exp(-delta^T Sigma^{-1} delta / 4), where delta is
the difference between the two trigrams' points.

Two figures are written next to this script: the onset stream with the
query and the six similarity profiles aligned beneath it
(demo_tempo_invariance.png), and the sd_shift sweep converging to the
is_rel limit (demo_tempo_invariance_limit.png).

Requires: numpy, matplotlib, mpt.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

import mpt
from mpt import interval_kernel_cov

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FN_FIG_MAIN = os.path.join(OUT_DIR, 'demo_tempo_invariance.png')
FN_FIG_LIMIT = os.path.join(OUT_DIR, 'demo_tempo_invariance_limit.png')

# Keep the one-shot kernel-controls tip out of the printed tables; the
# tip and the controls it points to are covered in
# demo_dispatch_and_kernel_controls.
_prev_defaults = mpt.set_default(show_hints=False)


# ===== 1. Material =====

# The motif: long-short-short, three IOIs (four onsets), in seconds at
# the reference tempo. All densities are built on natural-log IOIs, so
# a tempo factor a appears as a common shift of log(a).
D_MOTIF = np.array([0.50, 0.25, 0.25])
X_MOTIF = np.log(D_MOTIF)

# Candidate cells. The jittered cells displace the onset shared by the
# two short intervals by +25 ms (at the reference tempo), lengthening
# one short interval and shortening the other -- the signature of
# onset-level jitter. In the jittered-and-faster cell the displacement
# scales with the tempo, consistent with the proportional (log-space)
# jitter model.
CELLS = [
    ("exact",         D_MOTIF),
    ("20% faster",    D_MOTIF / 1.2),
    ("double speed",  D_MOTIF / 2.0),
    ("jittered",      np.array([0.50, 0.275, 0.225])),
    ("jit + faster",  np.array([0.50, 0.275, 0.225]) / 1.2),
    ("reversed",      np.array([0.25, 0.25, 0.50])),   # foil: same multiset
    ("isochronous",   np.array([1, 1, 1]) / 3.0),      # foil: same total
]

# The stream: the seven cells in order, separated by 1 s of silence.
# Because the gaps exceed every within-cell interval, any trigram that
# straddles a cell boundary reads that 1 s gap as one of its intervals
# -- realistic near-miss material for the search.
GAP = 1.0
onsets = []
t = 0.0
for _, iois in CELLS:
    cell_onsets = t + np.concatenate([[0.0], np.cumsum(iois)])
    onsets.extend(cell_onsets)
    t = cell_onsets[-1] + GAP
onsets = np.asarray(onsets)

# The input to difference_events is a two-attribute pre-MAET both of
# which are the onset times. difference_events is told to difference
# the first attribute once and leave the second untouched (the [1, 0]
# orders): the first becomes the inter-onset intervals (each interval
# timed at the onset that completes it), while the second stays as the
# raw onset times. bind_events then lays a sliding window of three
# consecutive log-IOIs across the event axis (the [3, 1] orders bind
# the interval attribute in threes and keep the time attribute as a
# singleton), one bound event per window, each trigram carrying the
# time of its first interval. The rel kernel needs its trigrams read
# relative to a common shift; the second bind, with rel_outer=True,
# produces that reading of the same trigrams.
p_diff, w_diff, sp_diff = mpt.difference_events(
    [onsets[None, :], onsets[None, :]], None, [1, 0])
p_diff[0] = np.log(p_diff[0])
p_bound, w_bound, sp_bound = mpt.bind_events(p_diff, w_diff, [3, 1],
                                             specs=sp_diff)
_, _, sp_bound_rel = mpt.bind_events(p_diff, w_diff, [3, 1],
                                     specs=sp_diff, rel_outer=True)
# Two quantities read off the bound carrier feed the search below:
N_TRI = p_bound[0].shape[1]      # number of trigrams (windows to place)
tri_times = p_bound[1].ravel()   # window-placing times (the sweep
                                 # centres); trigram i is timed at
                                 # onsets[i + 1]

# Presentation only -- no effect on the search, which is blind to cell
# boundaries. CELL_STARTS is the trigram index at which each cell
# begins, used to pick each cell's own trigram for the results table,
# the figure markers, and the near-miss annotation.
CELL_STARTS = [4 * i for i in range(len(CELLS))]

print("\n=== 1. Material ===\n")
print(f"  Motif IOIs (s): {D_MOTIF}  (long-short-short)")
print(f"  Stream: {len(CELLS)} cells x 4 onsets, 1 s gaps -> "
      f"{onsets.size} onsets, {N_TRI} overlapping log-IOI trigrams.")
print("  Each trigram is one event: an ordered K = 3 value multiset")
print("  read at r = 3, timed at the onset that completes its first")
print("  interval (the window-placing attribute).")


# ===== 2. interval_kernel_cov: shaping the kernel covariance =====

# interval_kernel_cov builds the covariance of the trigram attribute's
# Gaussian kernel -- the object that sets how much of each kind of
# departure from the motif the search treats as small. The covariance
# is a sum of three independently scaled terms, one per source of
# uncertainty being smoothed over:
#   sd_position -- jitter in the underlying onset times. Neighbouring
#     intervals share an onset, so this uncertainty couples them: it
#     enters as a tridiagonal term (2 sd^2 on the diagonal, -sd^2
#     between neighbours).
#   sd_interval -- noise on each interval on its own, uncorrelated
#     across intervals: a diagonal term.
#   sd_shift -- a common shift of all three intervals at once, i.e. a
#     tempo change in these log-IOI coordinates: a rank-one ridge
#     (sd_shift^2 added to every entry). The larger this ridge, the
#     freer the common shift becomes, and in the limit the kernel
#     approaches the relative-mode reading (is_rel=True), which quotients
#     the shift out exactly. Section 4 traces this convergence.
# The three cases below turn on one term at a time so each contribution
# to the covariance is visible on its own.
print("\n=== 2. interval_kernel_cov: the kernel covariance ===\n")
print("  interval_kernel_cov builds the covariance of the trigram")
print("  attribute's Gaussian kernel from three terms, one per source")
print("  of uncertainty the search smooths over: onset-time jitter")
print("  (sd_position), per-interval noise (sd_interval), and a common")
print("  tempo shift (sd_shift). Each case below turns on one term:\n")

S_POS = interval_kernel_cov(3, sd_position=0.05)
S_INT = interval_kernel_cov(3, sd_interval=0.05 * np.sqrt(2))
S_RDG = interval_kernel_cov(3, sd_position=0.05, sd_shift=0.25)

np.set_printoptions(precision=4, suppress=True)
print("  sd_position = 0.05 alone -- onset-time jitter, coupled across")
print("  shared onsets (tridiagonal, 2 sd^2 diagonal, -sd^2 off):")
print("   ", str(S_POS).replace("\n", "\n    "))
print("  sd_interval = 0.05*sqrt(2) alone -- independent per-interval")
print("  noise (diagonal; chosen to match the tridiagonal's per-interval")
print("  marginal variance of 0.005):")
print("   ", str(S_INT).replace("\n", "\n    "))
print("  sd_position = 0.05 with sd_shift = 0.25 -- onset jitter plus a")
print("  common-shift ridge (rank-one, added to every entry):")
print("   ", str(S_RDG).replace("\n", "\n    "))

# Price table: perturb the motif along three canonical directions and
# read the similarity of the perturbed trigram to the original under
# each kernel. The directions, in log-IOI units (eps = 0.10):
#   displaced onset:  (0, +eps, -eps)  one interior onset displaced
#                     (the shared endpoint of the two short intervals)
#   one interval:     (0, +eps,  0)    one interval stretched alone
#   tempo shift:      (+eps, +eps, +eps)  every interval scaled by
#                     exp(eps), i.e. a 10.5% tempo change
EPS = 0.10
PERTS = [
    ("displaced onset (0,+e,-e)", np.array([0.0,  EPS, -EPS])),
    ("one interval    (0,+e, 0)", np.array([0.0,  EPS,  0.0])),
    ("tempo shift     (e, e, e)", np.array([EPS,  EPS,  EPS])),
]
PURE = [
    ("position", S_POS),
    ("interval", S_INT),
    ("pos+shift", S_RDG),
]
W3 = np.ones(3)

print(f"\n  Similarity of motif + perturbation to motif (eps = {EPS}):\n")
print("  " + f"{'perturbation':<28}{'|d|^2':>7}"
      + "".join(f"{nm:>12}" for nm, _ in PURE))
print("  " + "-" * (35 + 12 * len(PURE)))
for pname, delta in PERTS:
    row = []
    for _, S in PURE:
        v = mpt.cos_sim_exp_tens(X_MOTIF, W3, X_MOTIF + delta, W3,
                                 S, 3, False, False, 0.0, False,
                                 normalize="oneSidedDenom", verbose=False)
        row.append(float(v))
    print("  " + f"{pname:<28}{np.sum(delta**2):>7.2f}"
          + "".join(f"{v:>12.3f}" for v in row))

print("""
  Reading the columns:
  - position: the displaced onset is CHEAPER than the single stretched
    interval despite having twice its squared norm -- anticorrelated
    perturbation of adjacent intervals is exactly what shared-endpoint
    noise generates, and the -sd^2 off-diagonals price it accordingly.
    The tempo shift is all but forbidden: the sum of the r intervals
    equals the difference of the two endpoint positions, so its
    variance under position noise is 2 sd^2 regardless of r -- a
    common drift of all intervals is highly atypical of position
    noise.
  - interval: pricing is by Euclidean norm alone (the two marginals
    are matched to the position column), so the ordering of the first
    two rows reverses.
  - pos+shift: the ridge makes the tempo shift the cheapest direction
    while leaving the within-shape prices essentially unchanged.""")


# ===== 3. The search: positional sigma vs tempo sigma =====

print("\n=== 3. Searching the stream for the motif ===\n")

# Six kernels. sd values are in natural-log units: sd_position = 0.10
# tolerates onset jitter of roughly 10% of the local inter-onset
# interval; sd_shift = 0.25 makes one sd a tempo factor of
# exp(0.25) ~ 1.28 (or its reciprocal). The rel entry is exact tempo
# invariance; its scalar sigma = 0.10*sqrt(2) matches the timing
# kernel's per-interval marginal (2 * sd_position^2).
KERNELS = [
    ("strict", "sd_position = 0.02",
     dict(sigma=interval_kernel_cov(3, sd_position=0.02),
          spec=sp_bound[0])),
    ("timing", "sd_position = 0.10",
     dict(sigma=interval_kernel_cov(3, sd_position=0.10),
          spec=sp_bound[0])),
    ("tempo", "sd_position = 0.02, sd_shift = 0.25",
     dict(sigma=interval_kernel_cov(3, sd_position=0.02, sd_shift=0.25),
          spec=sp_bound[0])),
    ("timing+tempo", "sd_position = 0.10, sd_shift = 0.25",
     dict(sigma=interval_kernel_cov(3, sd_position=0.10, sd_shift=0.25),
          spec=sp_bound[0])),
    ("large-shift", "sd_interval = 0.10*sqrt(2), sd_shift = 100",
     dict(sigma=interval_kernel_cov(3, sd_interval=0.10 * np.sqrt(2),
                                    sd_shift=100.0),
          spec=sp_bound[0])),
    ("rel", "rel_outer = True, sigma = 0.10*sqrt(2)",
     dict(sigma=0.10 * np.sqrt(2), spec=sp_bound_rel[0])),
]
print("  strict       : interval_kernel_cov(3, sd_position=0.02)")
print("  timing       : interval_kernel_cov(3, sd_position=0.10)")
print("  tempo        : interval_kernel_cov(3, sd_position=0.02, "
      "sd_shift=0.25)")
print("  timing+tempo : interval_kernel_cov(3, sd_position=0.10, "
      "sd_shift=0.25)")
print("  large-shift  : interval_kernel_cov(3, sd_interval=0.10*sqrt(2), "
      "sd_shift=100)")
print("  rel          : rel_outer=True bind spec, sigma = 0.10*sqrt(2)  "
      "(exact tempo invariance)")
print("  The large-shift kernel's within-shape term is matched to the")
print("  rel kernel's sigma, so its enormous ridge should reproduce the")
print("  rel column almost exactly (Section 4 gives the limit argument).")

# One windowed_similarity sweep per kernel. The rect window (full
# width 0.1 s, narrower than the smallest trigram spacing of 0.125 s)
# restricts each comparison to the single trigram at its centre; the
# time attribute only places the window and is dropped from the
# comparison, so each profile value is the plain similarity of that
# trigram to the query under the kernel. The search itself uses no
# knowledge of where the cells sit: every event of the stream starts a
# candidate trigram and receives a window, boundary-straddling
# trigrams included. Window centres are anchored
# to the trigram times rather than laid on a uniform grid, so
# their spacing follows the stream's own inter-onset intervals --
# denser where the music is faster, widest across the silences. A
# uniform grid (available via start/stop/step) would add nothing at
# this window width: a window containing one trigram returns that
# trigram's similarity wherever within its span the window is placed,
# and a window containing none returns zero, its context density
# being empty.
p_context = p_bound                    # [trigrams, times] as bound
w_context = [np.ones((3, N_TRI)), np.ones((1, N_TRI))]
p_query = [X_MOTIF[:, None], np.array([[0.0]])]
w_query = [np.ones((3, 1)), np.ones((1, 1))]

profiles = {}
for kname, _, kw in KERNELS:
    profiles[kname] = mpt.windowed_similarity(
        p_context, w_context, p_query, w_query,
        [kw["sigma"], 0.25], [3, 1], [False, False],
        [False, False], [0.0, 0.0],
        specs=[kw["spec"], sp_bound[1]], centres=tri_times, window_attr=1,
        drop_window_attr=True, context_window=("rect", 0.1),
        normalize="oneSidedDenom", verbose=False)

print("\n  Profile at each cell's own trigram:\n")
print("  " + f"{'candidate':<14}"
      + "".join(f"{k:>14}" for k, _, _ in KERNELS))
print("  " + "-" * (14 + 14 * len(KERNELS)))
for i, (cname, _) in enumerate(CELLS):
    print("  " + f"{cname:<14}"
          + "".join(f"{profiles[k][CELL_STARTS[i]]:>14.3f}"
                    for k, _, _ in KERNELS))

print("""
  Reading the rows:
  - 20% faster / double speed: pure tempo changes. Positional sigma
    alone barely admits them at any tolerable width ('timing' gives
    0.016 at sd_position = 0.10); sd_shift admits the moderate change
    and GRADES the large one ('tempo' gives 0.876 and 0.147); rel
    admits both exactly.
  - jittered: a same-tempo timing perturbation. Tempo sigma alone does
    not help ('tempo' gives 0.011); positional sigma does ('timing'
    gives 0.832). The two tolerances are separate currencies: in
    log-IOI space a tempo change moves the trigram's point ALONG the
    all-ones diagonal, timing jitter moves it off that line, and the
    kernel prices the two components independently.
  - jit + faster: needs both currencies at once -- only the kernels
    carrying both, 'timing+tempo' (0.742), 'large-shift', and 'rel'
    (0.777 each), admit it. Under 'rel' the two jittered rows are
    identical: the jittered-and-faster cell is the jittered cell under
    a pure tempo change (its displacement scales with the tempo), and
    rel quotients tempo out.
  - reversed: same interval multiset as the motif; the ordered outer
    read (sym_outer = False, the bind default) keeps it at zero under
    every kernel.
  - isochronous: a genuinely different shape; near zero throughout.
  - large-shift vs rel: the two columns agree at this precision.
    Graded tolerance with a sufficiently large ridge is numerically
    indistinguishable from the exact quotient -- the limit Section 4
    approaches from below, effectively reached.
""")
print("  Max |large-shift - rel| across all "
      f"{N_TRI} trigram positions: "
      f"{np.max(np.abs(profiles['large-shift'] - profiles['rel'])):.1e}\n")

# The full profile also sweeps the boundary-straddling trigrams. One
# is instructive: the trigram reading (last interval of the reversed
# cell, the 1 s gap, first isochronous interval) = (0.50, 1.0, 0.33) s
# -- the silence itself parses as the 'long' of a long-short-short
# figure with ratio 3:1:1, close in shape to the motif's 2:1:1 but at
# a remote tempo. That trigram is the one immediately before the
# isochronous cell's own.
i_straddle = CELL_STARTS[-1] - 1
print(f"  A boundary near-miss: the trigram at t = "
      f"{tri_times[i_straddle]:.2f} s reads the rest")
print("  after the reversed cell as a 'long', giving a")
print("  long-short-short of ratio 3:1:1 at a remote tempo. Graded")
print("  tempo tolerance suppresses what exact invariance admits:")
for kname in ("tempo", "timing+tempo", "large-shift", "rel"):
    print(f"    {kname:<13}: {profiles[kname][i_straddle]:.3f}")

# --- Main figure: the stream, the query, and the six profiles ------
# Top panel: the onset stream as an event raster, with each cell's
# span shaded and named, and the query drawn on a second y-level at
# t = 0..1 s -- the same time scale, sitting directly above the exact
# copy for visual comparison. Below: one panel per kernel, sharing the
# time axis, with the cell spans repeated so each peak reads off
# against its cell. Each profile is drawn as end-aligned stair steps:
# a trigram's tread spans its first inter-onset interval and its value
# sits at the right edge (the trigram's stamp), with a dot marking
# each stamp. Panel titles carry each kernel's constructor parameters.
cell_spans = [(onsets[4 * i], onsets[4 * i + 3])
              for i in range(len(CELLS))]
query_onsets = np.concatenate([[0.0], np.cumsum(D_MOTIF)])

fig, axes = plt.subplots(
    len(KERNELS) + 1, 1, figsize=(12, 2.4 + 1.35 * len(KERNELS)),
    sharex=True,
    gridspec_kw=dict(height_ratios=[1.7] + [1] * len(KERNELS),
                     hspace=0.12, left=0.09, right=0.98,
                     top=0.96, bottom=0.06))

ax = axes[0]
for lo, hi in cell_spans:
    ax.axvspan(lo, hi, color='0.55', alpha=0.15, lw=0)
ax.vlines(onsets, 0.0, 1.0, color='k', lw=1.2)
ax.vlines(query_onsets, 1.55, 2.45, color='C3', lw=1.8)
ax.text(query_onsets[-1] + 0.15, 2.0, 'query', color='C3',
        fontsize=9, va='center')
for (cname, _), (lo, hi) in zip(CELLS, cell_spans):
    ax.text(0.5 * (lo + hi), 1.12, cname, fontsize=8,
            rotation=30, ha='left', va='bottom')
ax.set_ylim(0, 3.3)
ax.set_yticks([])
ax.set_ylabel('events', fontsize=9)

for ax, (kname, plabel, _) in zip(axes[1:], KERNELS):
    for lo, hi in cell_spans:
        ax.axvspan(lo, hi, color='0.55', alpha=0.15, lw=0)
    prof = profiles[kname]
    prof = profiles[kname]
    # End-aligned stair steps: the tread for a trigram spans its first
    # inter-onset interval -- from the trigram's opening onset to the
    # onset that stamps it (onsets[i] to onsets[i+1]) -- so the tread
    # width shows that interval and the value sits at its right edge.
    # For the boundary trigram this first interval is the 1 s gap, so
    # the tread widens to cover it. Small dots mark the stamps; the
    # leading edge is carried back to the opening onset.
    ax.step(np.insert(tri_times, 0, onsets[0]), np.insert(prof, 0, prof[0]),
            where='pre', color='C0', lw=1.0)
    ax.plot(tri_times, prof, '.', color='C0', ms=4)
    ax.text(0.008, 0.97, f"{kname} ({plabel})", transform=ax.transAxes,
            fontsize=9, fontweight='bold', va='top')
    ax.set_ylim(-0.07, 1.30)
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel('similarity', fontsize=9)
# Annotate the boundary near-miss wherever tempo invariance admits it:
# the wide gap tread reads as a 'long', so both the large-shift and rel
# panels score it. Text sits up and to the right of the point so the
# arrow stays short.
for kname in ("large-shift", "rel"):
    ax = axes[[k for k, _, _ in KERNELS].index(kname) + 1]
    ax.annotate("gap parses as 'long'",
                xy=(tri_times[i_straddle], profiles[kname][i_straddle]),
                xytext=(tri_times[i_straddle] - 0.05, 0.66),
                ha='left', fontsize=8,
                arrowprops=dict(arrowstyle='->', lw=0.8))
axes[-1].set_xlabel('time (s)')
# Span the whole stream: the last onsets carry no trigram (a trigram
# needs three IOIs ahead of it), so the stair stamps stop before the
# stream does; set the limit from the onsets, not the stamps, so the
# raster's tail is not clipped.
axes[-1].set_xlim(-0.6, onsets[-1] + 0.6)

fig.savefig(FN_FIG_MAIN, dpi=120)
print(f"\n  wrote {FN_FIG_MAIN}")


# ===== 4. From tolerance to invariance: the is_rel limit =====

print("\n=== 4. sd_shift -> infinity is is_rel=True ===\n")
print("  As sd_shift grows, the kernel's precision tends to the")
print("  relative-mode projector: the shift direction becomes free")
print("  while the within-shape metric is left behind. With the")
print("  within-shape term supplied by sd_interval = 0.08, the limit")
print("  is EXACTLY is_rel=True at sigma = 0.08, because rel mode's")
print("  isotropic within-shape kernel is the limit of the diagonal")
print("  (sd_interval) family. (The sd_position family also has a")
print("  shift-invariant limit, but its within-shape metric is the")
print("  tridiagonal restricted to the zero-sum subspace, which is")
print("  not isotropic, so no scalar rel sigma reproduces it.)\n")

SD_INT = 0.08
targets = [("double speed", np.log(D_MOTIF / 2.0)),
           ("jit + faster", np.log(np.array([0.50, 0.275, 0.225]) / 1.2))]

print("  " + f"{'sd_shift':<12}"
      + "".join(f"{nm:>15}" for nm, _ in targets))
print("  " + "-" * (12 + 15 * len(targets)))
for ss in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 20.0):
    S = interval_kernel_cov(3, sd_interval=SD_INT, sd_shift=ss)
    row = [float(mpt.cos_sim_exp_tens(X_MOTIF, W3, x, W3, S, 3,
                                      False, False, 0.0, False,
                                      normalize="oneSidedDenom",
                                      verbose=False))
           for _, x in targets]
    print("  " + f"{ss:<12g}" + "".join(f"{v:>15.4f}" for v in row))
row = [float(mpt.cos_sim_exp_tens(X_MOTIF, W3, x, W3, SD_INT, 3,
                                  True, False, 0.0, False,
                                  normalize="oneSidedDenom",
                                  verbose=False))
       for _, x in targets]
print("  " + f"{'is_rel=True':<12}"
      + "".join(f"{v:>15.4f}" for v in row))

# --- Limit figure: similarity vs sd_shift, with the rel asymptotes --
ss_dense = np.logspace(np.log10(0.05), np.log10(20.0), 60)
fig2, ax2 = plt.subplots(figsize=(6.5, 4.2))
for j, ((tname, x), rel_v) in enumerate(zip(targets, row)):
    curve = [float(mpt.cos_sim_exp_tens(
                 X_MOTIF, W3, x, W3,
                 interval_kernel_cov(3, sd_interval=SD_INT, sd_shift=ss),
                 3, False, False, 0.0, False,
                 normalize="oneSidedDenom", verbose=False))
             for ss in ss_dense]
    ax2.semilogx(ss_dense, curve, '-', color=f'C{j}', label=tname)
    ax2.axhline(rel_v, color=f'C{j}', ls='--', lw=0.9)
    ax2.text(0.055, rel_v + 0.02, f'is_rel=True: {rel_v:.3f}',
             color=f'C{j}', fontsize=8)
ax2.set_xlabel('sd_shift (log scale)')
ax2.set_ylabel('similarity')
ax2.set_ylim(-0.03, 1.08)
ax2.set_title(f'Tempo tolerance -> tempo invariance '
              f'(sd_interval = {SD_INT})', fontsize=10)
ax2.legend(loc='center right', fontsize=9)
fig2.tight_layout()
fig2.savefig(FN_FIG_LIMIT, dpi=120)
print(f"\n  wrote {FN_FIG_LIMIT}")

print("""
  The double-speed copy converges to 1 (a pure shift is fully
  absorbed); the jittered-and-faster copy converges to the rel value
  set by its within-shape (jitter) component alone. Tempo tolerance
  varies continuously with sd_shift; tempo invariance is its limit.""")

mpt.set_default(**_prev_defaults)
