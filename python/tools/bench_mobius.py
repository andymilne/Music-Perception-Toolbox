"""bench_mobius.py

Wall-time comparison of the toolbox's alternative routes, across tuple
size r, multiset size K, and the four modes.

Two tasks, selected with --task:

  ip    the inner product (cos_sim_exp_tens): 'centres' (unrestricted
        enumeration of the tuple centres, the O(K^(2r)) baseline),
        'bulger' (the within-r-ad decomposition), and 'mobius' (the
        Moebius-orbit decomposition).
  eval  point evaluation (eval_exp_tens): 'centres' (the centres array)
        and 'mobius' (the Moebius point evaluator). Bulger's identity
        has no analogue here, there being no two-sided pairing to
        exploit, so evaluation has two routes rather than three.

The point of the benchmark is the RELATIVE timings and their scaling,
not the absolute ones: absolute times establish only that the regimes
reported are feasible. Accordingly the script reports, per cell, the
ratio of each method to the fastest, and fits the exponent of K for
each method so that the measured scaling can be checked against the
predicted degrees (2r for direct and Bulger, 2 for orbit).

Protocol notes (these matter for comparability):

* Correctness is checked BEFORE any timing, with truncation disabled.
  'centres' is the accuracy reference for both tasks: it enumerates the
  definition directly and involves no alternating sum, so it cannot
  suffer the cancellation the Moebius routes can. Each route's relative
  deviation from it is recorded per cell (column `rel_dev`) and
  reported, so the cancellation regimes -- K close to r, and
  relative-periodic at large sigma/P -- show up as loss of agreement
  rather than being hidden by a pass/fail threshold. Deviations beyond
  `TOL` are flagged but the cell is still timed.
* Timing uses truncation_sigmas=inf so that the three methods do the
  same arithmetic. A separate pass (--truncation) re-times the routed
  default (6 sigma) to quantify what truncation buys.
* Densities are built once per cell, outside the timed region: the timed
  unit is the similarity call alone, so construction is charged to no
  method. The memoised self inner products are cleared before each timed
  call, so every call pays for the full triple <X,Y>, <X,X>, <Y,Y> that
  a similarity requires, rather than for the cross term alone once the
  self terms have been cached (see clear_self_ip).
* Repetitions are auto-scaled so each timed unit takes ~TARGET_MS,
  after a warm-up call that absorbs first-call costs (orbit-table load,
  einsum path setup, BLAS thread spin-up).
* Every route timed is a toolbox route, called through the public API on
  the same density objects, so the comparison is like-for-like: the
  routes differ in their decomposition, not in how much of the toolbox
  they bypass. 'centres' is the unrestricted enumeration and so both the
  O(K^(2r)) baseline and the accuracy reference.
* The relative modes' orbit route is pinned with --rel-route. The toolbox
  otherwise chooses between a materialised tuple-centres route and a
  translation-grid route on a cost estimate, and the two have very
  different profiles: leaving it on 'auto' makes the 'mobius' column a
  mixture of two implementations, which shows up as non-monotonic timings
  in K. Run once per route and report them separately.
* Each method is forced explicitly. The aim is the real-world cost and
  feasibility of the three routes in each regime, so the toolbox's
  automatic dispatch is deliberately bypassed: timing 'auto' would
  measure the routing policy, not the methods it routes between.

Usage:
    python3 bench_mobius.py                    # standard sweep
    python3 bench_mobius.py --quick            # small sweep, for a smoke test
    python3 bench_mobius.py --truncation       # add the 6-sigma pass
    python3 bench_mobius.py --out results.csv

Writes a tidy CSV (one row per cell x method) and prints a summary.
"""
from __future__ import annotations

import argparse
import csv
import os
import platform
import sys
import time

import numpy as np

import mpt

# Silence one-time hints, and switch off the post-hoc guards: with guards
# on, a route that diverts pays for both routes, so the measured cost
# would not be the cost of the route being forced.
try:
    mpt.set_default(show_hints=False)
except Exception:
    pass
try:
    mpt.set_default(post_hoc_guards=False)
except Exception:
    pass                      # older builds lack the knob; forcing still holds

# --- sweep definition -------------------------------------------------------

# K grid: a geometric progression at ratio 1.5, six points spanning a
# factor of eight (6, 9, 14, 21, 32, 48). Equal spacing on a log axis is
# what a power-law slope fit wants, since the fit is a straight line in
# log K against log t and evenly spaced points weight it evenly; the
# earlier 6, 12, 20, 34 had ratios 2.0, 1.67, 1.70 and no stated basis.
# The lower end starts at 6 so that r = 4 still has K > r; the upper end
# stops at 48 because the enumerating routes are already far past
# feasibility there and the budget will skip them.
RS = (2, 3, 4)
KS = (6, 9, 14, 21, 32, 48)

#: Extension grid, Moebius only. The enumerating routes are infeasible
#: beyond K ~ 48, but the Moebius route is nearly flat there and its
#: predicted O(|Omega_r| K^2) has not begun to bite: measured exponents
#: on 6..48 come out at 0.1-0.5 against a predicted 2, because the fixed
#: per-call cost still dominates. It does bite further out -- the fitted
#: exponent reaches 1.96 by K = 1200 at r = 2 -- so the asymptote is
#: reached by extending K for that route alone, at a few hundred
#: milliseconds per cell.
#:
#: This continues the main grid's geometric progression at ratio 1.5,
#: rounded to two significant figures, rather than starting a new one:
#: 6, 9, 14, 21, 32, 48 | 72, 110, 160, 240, 360, 540, 820, 1200. One
#: progression across the whole range keeps the points evenly spaced on
#: the log axis a slope fit reads, with no discontinuity at the join.
EXT_KS = (72, 110, 160, 240, 360, 540, 820, 1200)
#: The extension pass runs every route, not the Moebius one alone. Which
#: routes can reach a given K is a question for the per-cell budget, not
#: for a hard-coded list: in the relative modes at low r the centres
#: array is the cheaper route and was previously cut off at K = 48 while
#: still below the Moebius evaluator, so the crossing went unobserved.
#: Cells beyond reach are skipped by the budget and the block abandoned,
#: as everywhere else. None means 'use the task's full route list'.
EXT_METHODS = None
MODES = (                      # (is_rel, is_per, label)
    (False, False, 'absolute non-periodic'),
    (False, True,  'absolute periodic'),
    (True,  False, 'relative non-periodic'),
    (True,  True,  'relative periodic'),
)
IP_METHODS = ('centres', 'bulger', 'mobius')
EVAL_METHODS = ('centres', 'mobius')
#: All three are toolbox routes, so every timing is like-for-like: the
#: same density objects, the same dispatch and truncation machinery,
#: differing only in the decomposition. 'centres' is the unrestricted
#: enumeration and so the O(K^(2r)) baseline.
N_QUERIES = 64

#: Kernel width, in cents, with the octave as period. The default follows
#: the manuscript's worked examples (sigma_pitch = 10 cents; equivalently
#: sigma = 0.1 with P = 12 in semitone units), giving sigma/P = 0.0083.
#: This is not a neutral choice: sigma/P governs every threshold in the
#: toolbox, so it decides which regime is being timed. Below sigma/P
#: ~0.03 the relative-periodic routes agree to floating point and the
#: periodic kernel sums a single image; above it they diverge and the
#: image count grows. Override with --sigma to time the other regime.
SIGMA = 10.0
PERIOD = 1200.0
TOL = 1e-9            # agreement tolerance between methods (untruncated)
TARGET_MS = 150.0     # per timed unit
SEED = 20260822

# --- work limits ------------------------------------------------------------
# Both enumerating methods cost roughly one unit per tuple PAIR, and the pair
# count explodes: at r = 5, K = 48 it is 3.5e14, which no budget accommodates.
# Cells are therefore skipped when their predicted cost exceeds CELL_BUDGET_S.
# The estimate uses a measured rate (about 1e-8 s per pair on the machine this
# was calibrated on, re-measured at startup) times the pair count:
#
#     direct:  (r! C(K,r))^2          both sides enumerated
#     bulger:  C(K,r) * r! C(K,r)     one side restricted (Eq. S8)
#     mobius:  |Omega_r| contractions, effectively independent of K
#
# Raise CELL_BUDGET_S to reach further into the enumerating methods' range;
# the orbit method is unaffected either way.
CELL_BUDGET_S = 20.0      # per timed cell, per method, predicted
PAIR_RATE_S = 1.0e-8      # seconds per tuple pair; re-calibrated at startup

#: Per-pair cost multipliers, relative to absolute non-periodic Bulger,
#: fitted from measured runs rather than guessed. The earlier guesses
#: (40x for periodic wrapping, 4x for the relative quadrature) were an
#: order of magnitude too pessimistic and rejected cells that complete in
#: well under a second: the slowest cell that survived a 2 s budget
#: actually took 347 ms.
MODE_SCALE = {
    #  (is_rel, is_per): (bulger/centres-side multiplier)
    (False, False): 1.0,
    (False, True):  3.2,
    (True,  False): 1.5,
    (True,  True):  5.5,
}
#: The unrestricted route carries a further factor over Bulger at the
#: same pair count, largest in the relative periodic mode.
CENTRES_SCALE = {(False, False): 1.8, (False, True): 1.3,
                 (True, False): 1.5, (True, True): 2.9}
OMEGA = {2: 4, 3: 10, 4: 33, 5: 92, 6: 306, 7: 948, 8: 3210}


class LCG:
    """Numerical Recipes ranqd1. Used in place of NumPy's generator so the
    MATLAB twin can draw byte-identical inputs and the two languages
    benchmark the same problem, not merely the same problem sizes."""

    def __init__(self, seed):
        self.state = int(seed)

    def uniform01(self, n):
        out = np.empty(n)
        s = self.state
        for i in range(n):
            s = (1664525 * s + 1013904223) % (1 << 32)
            out[i] = s / float(1 << 32)
        self.state = s
        return out


def draw(K, rng):
    u = rng.uniform01(2 * K)
    p = np.sort(u[:K] * 3 * PERIOD)
    w = 0.4 + 0.6 * u[K:]
    return p, w


def make_pair(K, rng):
    """Two weighted multisets of K atoms, in cents over three octaves."""
    p, wp = draw(K, rng)
    q, wq = draw(K, rng)
    return (p, wp), (q, wq)


def build(pair, r, is_rel, is_per, sigma=None):
    (p, wp), (q, wq) = pair
    sigma = SIGMA if sigma is None else sigma
    A = mpt.build_exp_tens(p, wp, sigma, r, is_rel, is_per, PERIOD, verbose=False)
    B = mpt.build_exp_tens(q, wq, sigma, r, is_rel, is_per, PERIOD, verbose=False)
    return A, B


def eval_points(r, is_rel, rng):
    """Query points for an evaluation cell, shaped (dim, n_queries)."""
    dim = max(r - (1 if is_rel else 0), 1)
    u = np.asarray(rng.uniform01(dim * N_QUERIES)).reshape(dim, N_QUERIES)
    return u * PERIOD


def call_eval(A, pts, method, trunc):
    kw = dict(method=method, verbose=False)
    if trunc is not None:
        kw['truncation_sigmas'] = trunc
    return float(np.sum(mpt.eval_exp_tens(A, pts, **kw)))


def clear_self_ip(*densities):
    """Discard memoised self inner products on these densities.

    Timings are taken with the process warm -- orbit tables built, einsum
    paths cached -- but with <X,X> and <Y,Y> NOT carried over from the
    previous call, so each timed call pays for the full triple a
    similarity actually requires.

    Without this the first call on a pair computes all three products and
    memoises the two self terms, and every later call computes only the
    cross term: measured times fall by a factor of about three, and the
    figure reported is no longer the cost of a similarity. It also breaks
    comparability between the languages, since a MATLAB density is a
    value struct whose cache is discarded unless the caller captures the
    returned density, while a Python density is an object whose cache
    persists.
    """
    for d in densities:
        cache = getattr(d, '_self_ip_cache', None)
        if cache is not None:
            cache.clear()


def call(A, B, method, trunc):
    kw = dict(method=method, verbose=False)
    if trunc is not None:
        kw['truncation_sigmas'] = trunc
    return float(mpt.cos_sim_exp_tens(A, B, **kw))


def timed(fn, target_ms=TARGET_MS, budget_s=None):
    """Median of repeated batches, each batch auto-sized to ~target_ms.

    Returns (seconds_per_call, n_reps). One warm-up call first, so that
    orbit-table loading and einsum path setup are not charged to the
    measurement.

    The warm-up is timed, and if it alone exceeds ``budget_s`` the cell is
    reported from that single call with the repetitions skipped. A
    predicted-cost budget can only skip what its model prices correctly;
    this guard is model-free, so a cell the predictor underestimates
    cannot run away. Such cells carry reps = 1 in the CSV, marking them
    as single-shot and therefore noisier than the rest.
    """
    t_warm = time.perf_counter()
    fn()                                    # warm-up, timed
    warm = time.perf_counter() - t_warm
    if budget_s is not None and warm > budget_s:
        return warm, 1, True
    n, elapsed = 1, 0.0
    while True:                             # size the batch
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        elapsed = time.perf_counter() - t0
        if elapsed * 1e3 >= target_ms or n >= 1 << 20:
            break
        n = max(2 * n, int(n * target_ms / max(elapsed * 1e3, 1e-6)))
    per_call = []
    for _ in range(5):                      # five batches, take the median
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        per_call.append((time.perf_counter() - t0) / n)
    return float(np.median(per_call)), n, False


def pair_count(method, r, K):
    """Tuple pairs a method must visit; None when K-independent."""
    from math import comb, factorial
    if K < r:
        return 0
    ordered = factorial(r) * comb(K, r)
    if method == 'centres':
        return ordered * ordered        # unrestricted: both sides enumerated
    if method == 'bulger':
        # In the single-multiset path 'direct' routes through the same core
        # as 'bulger' (the distinction surfaces only in the windowed paths),
        # so both are priced at the restricted count. Pricing 'direct' at
        # the unrestricted ordered^2 put it over budget almost everywhere
        # and it dropped out of the grid entirely.
        return comb(K, r) * ordered
    return None                       # mobius: |Omega_r| contractions


#: Measured per-call cost of the orbit method, in seconds, by mode and r.
#: The relative modes carry a one-dimensional quadrature over the common
#: shift, which the absolute modes do not, and it dominates: at r = 5 the
#: relative non-periodic call is ~30x the absolute one. These are order-of-
#: magnitude figures used only for budgeting.
ORBIT_S = {
    (False, False): {2: 6e-4, 3: 1.3e-3, 4: 3.8e-3, 5: 1.2e-2, 6: 4e-2},
    (False, True):  {2: 6e-4, 3: 1.2e-3, 4: 3.6e-3, 5: 1.2e-2, 6: 4e-2},
    (True,  False): {2: 1.3e-3, 3: 1.6e-2, 4: 8.5e-2, 5: 3.4e-1, 6: 1.4e0},
    (True,  True):  {2: 4e-4, 3: 9e-4, 4: 2.5e-2, 5: 9.1e-2, 6: 3.6e-1},
}


def predict_s(method, r, K, rate, is_rel=False, is_per=False):
    """Predicted seconds for ONE call."""
    n = pair_count(method, r, K)
    if n is None:
        return ORBIT_S.get((is_rel, is_per), ORBIT_S[(False, False)]).get(r, 1.0)
    # Measured multipliers on the per-pair rate. Periodic wrapping is the
    # large one: at r = 3, K = 20 the periodic Bulger call is ~18x the
    # non-periodic, and at K = 34 nearer 40x, the wrap being applied per
    # coordinate of every tuple pair. The relative modes add a quadrature
    # over the common shift. Both are deliberate over-estimates, so the
    # budget errs toward skipping a cell rather than running for minutes.
    scale = MODE_SCALE.get((bool(is_rel), bool(is_per)), 1.0)
    if method == 'centres':
        scale *= CENTRES_SCALE.get((bool(is_rel), bool(is_per)), 1.5)
    return n * rate * scale


def calibrate_rate():
    """Measure seconds per tuple pair on this machine, on a small safe cell."""
    rng = LCG(SEED)
    p, w = draw(12, rng)
    q, wq = draw(12, rng)
    A = mpt.build_exp_tens(p, w, SIGMA, 3, False, False, PERIOD, verbose=False)
    B = mpt.build_exp_tens(q, wq, SIGMA, 3, False, False, PERIOD, verbose=False)
    fn = lambda: mpt.cos_sim_exp_tens(A, B, method='bulger',
                                      truncation_sigmas=float('inf'), verbose=False)
    t, _, _ = timed(fn, 60.0)
    return t / pair_count('bulger', 3, 12)


def fit_exponent(Ks, ts, tail=3):
    """Slope of log t against log K over the largest `tail` values of K.

    Fitting the whole sweep understates the exponent badly: at small K
    the per-call overhead (argument handling, object access, BLAS entry)
    dominates the arithmetic, so the curve is flat there for every
    method. The asymptotic regime is the large-K tail, which is what the
    predicted degrees describe.
    """
    Ks, ts = np.asarray(Ks, float), np.asarray(ts, float)
    ok = (Ks > 0) & (ts > 0)
    Ks, ts = Ks[ok], ts[ok]
    # Distinct K only: a repeated K makes the fit singular and numpy
    # returns a meaningless slope with a conditioning warning.
    if np.unique(Ks).size < 3:
        return float('nan')
    order = np.argsort(Ks)
    Ks, ts = Ks[order][-tail:], ts[order][-tail:]
    if Ks.size < 3:
        return float('nan')
    return float(np.polyfit(np.log(Ks), np.log(ts), 1)[0])


def load_done(paths):
    """Cells already measured, as (task, mode, r, K, method).

    A run given --resume skips these, so it fills only what is missing.
    The identity deliberately excludes the language and the run's
    settings: the files named are this machine's own earlier output, and
    a cell measured there needs no second measurement here.
    """
    done = set()
    for path in paths or []:
        try:
            with open(path, newline='') as fh:
                for row in csv.DictReader(fh):
                    done.add((row['task'], row['mode'], int(row['r']),
                              int(row['K']), row['method']))
        except (OSError, KeyError, ValueError) as exc:
            print(f'warning: could not read {path}: {exc}')
    return done


def run_ip(args, modes, rs, ks, rng, rate, rows, failures, skipped,
           abandoned, done, methods=IP_METHODS):
    """Inner-product sweep over the toolbox routes."""
    done_ip = {(m, r, K, meth) for t, m, r, K, meth in done if t == "ip"}
    header = (f"{'mode':<22}{'r':>3}{'K':>5}  "
              + ''.join(f'{m:>12}' for m in methods) + '   fastest')
    print(header)
    print('-' * len(header))
    for is_rel, is_per, mlabel in modes:
        for r in rs:
            # Cost rises monotonically with K for each route separately,
            # so a route that has blown the budget will blow it at every
            # larger K and is dropped from the rest of the block. The
            # bookkeeping is per route, not per block: a single flag for
            # the whole block let one expensive route abandon the others,
            # which is how the cheaper route in a panel came to stop
            # early while still below the route it was being compared
            # against.
            done_methods = set()
            for K in ks:
                if len(done_methods) == len(methods):
                    abandoned.append((mlabel, r, K))
                    continue
                if K < r + 1:
                    continue
                pair = make_pair(K, rng)
                A, B = build(pair, r, is_rel, is_per, args.sigma)

                vals = {}
                for m in methods:
                    if m in done_methods:
                        continue
                    if (mlabel, r, K, m) in done_ip:
                        continue
                    if predict_s(m, r, K, rate, is_rel, is_per) > args.budget:
                        skipped.append((mlabel, r, K, m,
                                        predict_s(m, r, K, rate, is_rel, is_per)))
                        done_methods.add(m)
                        continue
                    try:
                        vals[m] = call(A, B, m, float('inf'))
                    except Exception as exc:
                        failures.append((mlabel, r, K, m,
                                         f'{type(exc).__name__}: {exc}'))

                # centres is the accuracy reference: it enumerates the
                # definition with no restriction and no alternating sum.
                ref = vals.get('centres', float('nan'))
                devs = {}
                for m, v in vals.items():
                    if m == 'centres' or not np.isfinite(ref) or not np.isfinite(v):
                        devs[m] = float('nan')
                        continue
                    devs[m] = abs(v - ref) / max(1.0, abs(ref))
                    if devs[m] > TOL:
                        failures.append((mlabel, r, K, m,
                                         f'deviates from centres by {devs[m]:.2e}'))

                times = {}
                for m in vals:
                    A2, B2 = build(pair, r, is_rel, is_per)
                    try:
                        def fn(m=m, A2=A2, B2=B2):
                            clear_self_ip(A2, B2)
                            return call(A2, B2, m, float('inf'))
                        t, n, guarded = timed(fn, budget_s=args.budget)
                        if guarded:
                            done_methods.add(m)
                        times[m] = t
                        rows.append(dict(language='python', task='ip', sigma=args.sigma,
                                         period=PERIOD,
                                         mode=mlabel, is_rel=is_rel,
                                         is_per=is_per, r=r, K=K,
                                         n_queries='', method=m,
                                         rel_route=('forced-by-method' if args.rel_route == 'auto'
                                                    else args.rel_route),
                                         truncation='inf', seconds=t, reps=n, guarded=guarded,
                                         value=vals[m],
                                         rel_dev=devs.get(m, float('nan'))))
                    except Exception as exc:
                        failures.append((mlabel, r, K, m, f'timing: {exc}'))

                # Nothing measured here -- every route resumed, over
                # budget, or abandoned -- so print no line: a table of
                # dashes buries the rows that do carry a measurement.
                if not times:
                    continue
                comp = [m for m in methods if m in times]
                # 'fastest' ranks only what this run measured. With
                # --resume the absent routes may well be quicker, so the
                # column is marked rather than left to imply otherwise.
                best = min(comp, key=lambda m: times[m]) if comp else '-'
                if len(comp) < len(methods):
                    best += '*'
                cells = ''.join(
                    (f'{times[m] * 1e3:>11.3f}m' if m in times else f'{"-":>12}')
                    for m in methods)
                print(f'{mlabel:<22}{r:>3}{K:>5}  {cells}   {best}')


def run_eval(args, modes, rs, ks, rng, rows, failures, abandoned, done,
             methods=EVAL_METHODS):
    """Evaluation sweep: centres / mobius, at a fixed query count."""
    done_eval = {(m, r, K, meth) for t, m, r, K, meth in done if t == "eval"}
    header = (f"{'mode':<22}{'r':>3}{'K':>5}  "
              + ''.join(f'{m:>12}' for m in methods) + '   fastest')
    print(f'point evaluation, {N_QUERIES} query points per cell\n')
    print(header)
    print('-' * len(header))
    for is_rel, is_per, mlabel in modes:
        for r in rs:
            done_methods = set()        # see the note in run_ip
            for K in ks:
                if len(done_methods) == len(methods):
                    abandoned.append((mlabel, r, K))
                    continue
                if K < r + 1:
                    continue
                (p, w), _ = make_pair(K, rng)
                A = mpt.build_exp_tens(p, w, args.sigma, r, is_rel, is_per,
                                       PERIOD, verbose=False)
                pts = eval_points(r, is_rel, rng)

                vals, times = {}, {}
                for m in methods:
                    if m in done_methods:
                        continue
                    if (mlabel, r, K, m) in done_eval:
                        continue
                    try:
                        vals[m] = call_eval(A, pts, m, float('inf'))
                    except Exception as exc:
                        failures.append((mlabel, r, K, m,
                                         f'{type(exc).__name__}: {exc}'))
                # centres is the reference for evaluation. The comparison
                # needs an absolute floor as well as a relative one: at
                # query points far from every centre the centres route
                # returns exactly 0 while the alternating Moebius sum
                # returns dust of order 1e-16, and a purely relative
                # measure then divides by zero.
                ref = vals.get('centres', float('nan'))
                scale = max(abs(ref), 1.0)
                devs = {}
                for m, v in vals.items():
                    if m == 'centres' or not np.isfinite(ref) or not np.isfinite(v):
                        devs[m] = float('nan')
                        continue
                    devs[m] = abs(v - ref) / scale
                    if devs[m] > TOL:
                        failures.append((mlabel, r, K, m,
                                         f'deviates from centres by {devs[m]:.2e}'))
                for m in vals:
                    A2 = mpt.build_exp_tens(p, w, args.sigma, r, is_rel, is_per,
                                            PERIOD, verbose=False)
                    try:
                        def fn_eval(m=m, A2=A2):
                            clear_self_ip(A2)
                            return call_eval(A2, pts, m, float('inf'))
                        t, n, guarded = timed(fn_eval, budget_s=args.budget)
                        if guarded:
                            done_methods.add(m)
                        times[m] = t
                        rows.append(dict(language='python', task='eval', sigma=args.sigma,
                                         period=PERIOD,
                                         mode=mlabel, is_rel=is_rel,
                                         is_per=is_per, r=r, K=K,
                                         n_queries=N_QUERIES, method=m,
                                         rel_route=('forced-by-method' if args.rel_route == 'auto'
                                                    else args.rel_route),
                                         truncation='inf', seconds=t, reps=n, guarded=guarded,
                                         value=vals[m],
                                         rel_dev=devs.get(m, float('nan'))))
                    except Exception as exc:
                        failures.append((mlabel, r, K, m, f'timing: {exc}'))

                if not times:
                    continue                      # see the note in run_ip
                best = min(times, key=times.get)
                if len(times) < len(methods):
                    best += '*'
                cells = ''.join(
                    (f'{times[m] * 1e3:>11.3f}m' if m in times else f'{"-":>12}')
                    for m in methods)
                print(f'{mlabel:<22}{r:>3}{K:>5}  {cells}   {best}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=('ip', 'eval', 'both'), default='both',
                    help='which comparison to run')
    ap.add_argument('--quick', action='store_true', help='small sweep')
    ap.add_argument('--rs', type=str, default=None,
                    help='comma-separated tuple sizes, e.g. 2,3,4')
    ap.add_argument('--ks', type=str, default=None,
                    help='comma-separated multiset sizes, e.g. 6,12,20,34')
    ap.add_argument('--rel-route',
                    choices=('auto', 'centres', 'mobius', 'grid'),
                    default='auto', dest='rel_route',
                    help="relative-attribute route inside the orbit method. "
                         "Has no effect on these timings: every method here "
                         "is forced, and an explicit method='mobius' pins "
                         "the sub-route ahead of this lever. Retained only "
                         "to measure the lever itself")
    ap.add_argument('--extend', action='store_true',
                    help='append a Moebius-only pass at large K '
                         '(80..1200), where its K^2 asymptote appears')
    ap.add_argument('--extend-only', action='store_true', dest='extend_only',
                    help='run that pass alone, for topping up a run whose '
                         'main grid is already in hand')
    ap.add_argument('--resume', nargs='*', default=None, metavar='CSV',
                    help='skip any (task, mode, r, K, method) cell already '
                         'present in these CSVs, so the run fills only what '
                         'is missing')
    ap.add_argument('--ext-ks', type=str, default=None, dest='ext_ks',
                    help='comma-separated K for the extension pass')
    ap.add_argument('--spectral-gate', action='store_true',
                    dest='spectral_gate',
                    help="leave the spectral branch's cost gate in force; "
                         'by default the benchmark bypasses it, because '
                         'the gate misroutes outside its calibration range '
                         'and the figures are about the decompositions, '
                         'not the routing')
    ap.add_argument('--sigma', type=float, default=SIGMA,
                    help='kernel width in cents (default 10; sigma/P '
                         'governs which regime is timed)')
    ap.add_argument('--budget', type=float, default=CELL_BUDGET_S,
                    help='predicted seconds per cell per method; cells above '
                         'are skipped (inner product only)')
    ap.add_argument('--out', default=None,
                    help='CSV path; defaults to '
                         'bench_mobius_python_<route>.csv')
    args = ap.parse_args()
    if args.out is None:
        # Never overwrite an existing results file. A run given --resume
        # reads the earlier output and then writes only what it measured;
        # writing back to the same name would delete the very cells it
        # skipped, which is how an extension run once destroyed the data
        # it had just been told to preserve.
        tag = '_ext' if args.extend_only else ''
        stem = f'bench_mobius_python_{args.rel_route}_sig{args.sigma:g}{tag}'
        args.out = f'{stem}.csv'
        n = 2
        while os.path.exists(args.out):
            args.out = f'{stem}_{n}.csv'
            n += 1

    rs = tuple(int(x) for x in args.rs.split(',')) if args.rs else \
        ((2, 3) if args.quick else RS)
    ks = tuple(int(x) for x in args.ks.split(',')) if args.ks else \
        ((6, 9, 14) if args.quick else KS)
    modes = MODES[:1] if args.quick else MODES

    print(f'python {platform.python_version()} | numpy {np.__version__} | '
          f'mpt {getattr(mpt, "__version__", "?")}')
    print(f'{platform.platform()}')
    try:
        import mpt._tensor._mobius_inner as _mi
        _mi._SPECTRAL_IP_FORCE = not args.spectral_gate
    except Exception:
        print('warning: could not set the spectral-branch lever')
    try:
        mpt.set_default(rel_attr_route=args.rel_route)
    except Exception:
        if args.rel_route != 'auto':
            print(f"warning: could not pin rel_attr_route={args.rel_route}")

    rng = LCG(SEED)
    rows, failures, skipped, abandoned = [], [], [], []
    done = load_done(args.resume)
    if done:
        print(f'resuming: {len(done)} cells already measured will be '
              f'skipped. Rows with nothing left to measure are not '
              f'printed, and a starred "fastest" ranks only the routes '
              f'this run timed.')
    rate = calibrate_rate()
    print(f'calibrated at {rate * 1e9:.1f} ns per tuple pair; '
          f'cell budget {args.budget:g} s; rel route {args.rel_route}')
    print(f'sigma {args.sigma:g} cents, period {PERIOD:g} '
          f'(sigma/P = {args.sigma / PERIOD:.4f})')
    print("spectral branch cost gate: "
          + ('in force (shipped routing)' if args.spectral_gate
             else 'bypassed (measuring the decomposition)'))
    print("relative-attribute sub-route: pinned by the forced method "
          "(rel_attr_route has no effect here)")
    print()

    if not args.extend_only:
        if args.task in ('ip', 'both'):
            run_ip(args, modes, rs, ks, rng, rate, rows, failures, skipped,
                   abandoned, done)
        if args.task in ('eval', 'both'):
            if args.task == 'both':
                print()
            run_eval(args, modes, rs, ks, rng, rows, failures, abandoned, done)

    if args.extend or args.extend_only:
        ext_ks = (tuple(int(x) for x in args.ext_ks.split(','))
                  if args.ext_ks else EXT_KS)
        print(f'\nextension pass, K = '
              f'{", ".join(str(k) for k in ext_ks)}\n')
        if args.task in ('ip', 'both'):
            run_ip(args, modes, rs, ext_ks, rng, rate, rows, failures,
                   skipped, abandoned, done,
                   methods=EXT_METHODS or IP_METHODS)
        if args.task in ('eval', 'both'):
            print()
            run_eval(args, modes, rs, ext_ks, rng, rows, failures,
                     abandoned, done, methods=EXT_METHODS or EVAL_METHODS)

    if abandoned:
        print(f'\nabandoned ({len(abandoned)}): a smaller K in the same '
              f'(mode, r) block already exceeded the {args.budget:g} s '
              f'budget, and cost rises with K:')
        blocks = {}
        for mlabel, r, K in abandoned:
            blocks.setdefault((mlabel, r), []).append(K)
        for (mlabel, r), Ks in sorted(blocks.items()):
            print(f"   {mlabel:<22} r={r}   K = "
                  f"{', '.join(str(k) for k in Ks)}")

    single = [rw for rw in rows if rw.get('guarded')]
    if single:
        print(f'\nguarded cells ({len(single)}): one call already '
              f'exceeded the {args.budget:g} s budget, so repetitions were '
              f'skipped; these timings are noisier than the rest:')
        for rw in sorted(single, key=lambda r: -r['seconds'])[:8]:
            print(f"   {rw['task']:<5} {rw['mode']:<22} r={rw['r']} "
                  f"K={rw['K']:>4} {rw['method']:<8} {rw['seconds']:8.2f} s")
        if len(single) > 8:
            print(f'   ... and {len(single) - 8} more')

    print('\nagreement with the centres route (relative deviation, '
          'untruncated):')
    worst = sorted((rw for rw in rows
                    if rw['method'] == 'mobius' and np.isfinite(rw['rel_dev'])),
                   key=lambda rw: -rw['rel_dev'])[:8]
    for rw in worst:
        print(f"   {rw['task']:<5} {rw['mode']:<22} r={rw['r']}  "
              f"K={rw['K']:>3}   {rw['rel_dev']:.2e}")

    if skipped:
        print(f'\nskipped as over budget ({len(skipped)}); '
              f'raise --budget to include:')
        for sk in sorted(skipped, key=lambda s: s[4])[:6]:
            print(f'   {sk[0]:<22} r={sk[1]}  K={sk[2]:>3}  {sk[3]:<12} '
                  f'predicted {sk[4]:.1f} s')
        if len(skipped) > 6:
            print(f'   ... and {len(skipped) - 6} more')

    if failures:
        print(f'\nFAILURES ({len(failures)}):')
        for f in failures[:25]:
            print('  ', f)
    else:
        print(f'\nall routes agreed to within {TOL:g} on every cell')

    with open(args.out, 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f'\nwrote {args.out}  ({len(rows)} rows)')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
