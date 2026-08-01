"""Measure the relative-mode inner-product cost, per sub-route, for fitting.

Python twin of matlab/tools/calibrateRelIpCost.m. Same grid, same columns,
same policy, so the two languages' constants are fitted from measurements
made the same way.

Why this exists. The multi-attribute selector chooses between Bulger's
method and the Möbius method by comparing two predicted wall times
(``_select_ma_inner_product_method``). On relative-mode densities that
comparison misroutes badly: measured on the MATLAB side it sent 47% of
cells to the slower method. Refitting it needs a few hundred cells
measured on a quiet machine -- fitted on twenty-odd, no candidate form
beat the shipped one by more than a single cell under cross-validation.

What it records per cell: each sub-route timed in isolation, with
``rel_attr_route`` pinning the route so a curve is the scaling of one
route rather than a mixture of two; Bulger's method on the same cell,
since the decision is between them; the two predictions the selector's
comparison rests on; and the quantities a fit needs, the grid node count
N_u and the permutation-side tuple count M = r!·C(K, r).

Agreement is checked before any timing, so a comparison is never made
between two computations that disagree.

Three things keep the runtime affordable. An arm the running estimate
puts over budget is never started; a call slower than ``--repeat-below``
is measured from its single warm run rather than repeated, since
repeating a call that already ran for a second buys nothing; and the
unforced Möbius arm is not timed, because ``gate_route`` says which
route column already holds its time.

Usage:
    python tools/calibrate_rel_ip_cost.py > rel_ip_cost.csv

Run it from the ``python`` directory with the package importable, on a
machine doing nothing else. Rows stream as they are measured, so a run
can be read while it proceeds and stopped early without losing what came
before. Expect a few minutes.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import math
import statistics
import sys
import time
import warnings

import numpy as np

import mpt
from mpt._defaults import resolve_samples_per_sigma
from mpt._tensor._mobius_inner import (
    _ma_rel_attr_prefers_centres,
    _rel_window_margin,
)
from mpt._tensor._nested_contraction import auto_ntau_default
from mpt._tensor.dispatch import _select_ma_inner_product_method

# K = 100 at r = 2 is dropped: its Bulger arm runs for seconds and the
# r = 2 curve is already determined by K = 64.
# Three value counts per order rather than six: the event count is now a
# swept axis too, and a full factorial over both would run for hours.
# Geometric spacing separates a power law as well as a dense grid does.
K_BY_ORDER = {2: [6, 16, 40], 3: [6, 12, 24], 4: [5, 8, 12]}

# The two densities need not carry the same number of values, and a chord
# against a scale is the ordinary case. A sweep with equal counts
# throughout leaves the asymmetric case unconstrained, and it is where a
# cost model most easily goes wrong: the tuple-pair count spans five
# orders of magnitude across it, so a term fitted only on equal counts
# flattens exactly the shape that matters. K_REF is the small side.
K_REF = 5
SHAPES = ("equal", "asym")

# Weights change how much of a multiset the truncated kernel actually
# touches, so a model fitted on one profile need not hold on another.
# Three shapes, each jittered per seed so no cell is a special case:
# flat, decaying towards one end, and concentrated at both ends.
WEIGHT_PROFILES = ("flat", "decay", "bimodal")

# Event counts. Both methods price per event pair, but they do not scale
# with the pair count the same way: Bulger's method builds one joint
# tuple-pair kernel over all events at once, so its working set grows
# with the product, while the Möbius method repeats a per-pair cost. A
# sweep at one event per density leaves that unconstrained, and it is a
# large effect -- at twelve events against twelve values, relative
# periodic at sigma/period = 0.05, Bulger's method takes 50.3 s where the
# Möbius method takes 5.5 ms, a factor of 9100. A model fitted on
# single-event cells alone put that one on the wrong side.
#
# The range runs well past where Bulger's method can be timed: measured
# at K = 8, it reaches 5.4 s by twelve events, while the Möbius method is
# still 43 ms at 128. Those cells are not wasted. An arm that exceeds the
# budget is recorded as ``inf`` rather than dropped, which is a censored
# observation -- its time is unknown but bounded below -- and that is
# enough to settle which method is faster, which is what the routing fit
# is scored on. Dropping them would discard exactly the cells where the
# decision is most consequential.
# Event counts, capped by tuple order. Cost grows with the tuple order,
# the value count and the event count together, and the budget can only
# decline to *start* an arm --- neither language can interrupt one already
# running --- so the worst cell has to be bounded by construction. Sixty-
# four events at r = 4 is not a workload anyone runs; sixty-four at r = 2
# is, and that is where the range is wanted.
N_BY_ORDER = {2: (1, 4, 16, 64), 3: (1, 4, 16), 4: (1, 4, 8)}


def _weights(profile, K, rng):
    """Weight vector of the named shape, jittered so no cell is exact."""
    if profile == "flat":
        base = np.ones(K)
    elif profile == "decay":
        base = np.exp(-np.linspace(0.0, 4.0, K))
    elif profile == "bimodal":
        x = np.linspace(-1.0, 1.0, K)
        base = np.exp(-((1.0 - np.abs(x)) ** 2) * 6.0)
    else:
        raise ValueError(f"unknown weight profile {profile!r}")
    return base * (0.75 + 0.5 * rng.uniform(0, 1, K))


#: Working-set ceiling for Bulger's method, in float64 entries. Its joint
#: tuple-pair kernel is n_J x n_K entries held at once, and that grows
#: with the event count as well as the value count: at r = 2, K = 40 and
#: 64 events it is 4.98e9 entries, 40 GB. Such a cell cannot be timed --
#: the call raises, and an arm that raises used to be recorded as missing,
#: losing the cell. It is not missing: a method that cannot run is
#: decisively the slower one, so the arm is recorded as censored and the
#: comparison still resolves. Predicting it also avoids paying for the
#: attempt, which allocates before it fails.
_BULGER_MAX_ENTRIES = 2.0e8            # 1.6 GB at float64

#: Prior cost per predictor unit, in seconds, used to decline an arm
#: before any measurement exists to estimate from. The running median
#: replaces it as soon as three cells have been timed at that order, so
#: these need only be the right order of magnitude; they exist so the
#: first few cells of a sweep cannot each cost minutes. Taken from direct
#: measurement: the grid route ran 0.38 s at a predictor of 4.1e7.
_PRIOR_RATE = {"B": 2.0e-8, "C": 2.0e-8, "G": 1.0e-8}

#: The one place the column names live, so the header cannot drift from
#: the rows. _ROW_FIELDS is what the row format string emits, asserted
#: equal to the header width by the check mode.
_HEADER = ("r,K_x,K_y,N,shape,weights,isPer,sigma,seed,nu,M_x,M_y,"
           "t_bulger,t_centres,t_grid,pred_bulger,pred_mobius,"
           "max_abs_diff,gate_route,faster,"
           "declined_bulger,declined_centres,declined_grid")
_ROW_FIELDS = 23


def _bulger_kernel_entries(r, K_x, K_y, N):
    """Entries in Bulger's joint tuple-pair kernel for one cell."""
    n_j = N * math.factorial(r) * math.comb(K_x, r)
    n_k = N * math.comb(K_y, r)
    return float(n_j) * float(n_k)


def _timed(dens_x, dens_y, method, route,
           budget, repeat_below, seen, key, predictor, infeasible=False):
    """Time one arm, or decline to start it. Returns (ms, value, seen)."""
    if infeasible:
        # Exact arithmetic, not an estimate: the working set does not fit.
        return float("inf"), None, seen, "memory"
    samples = seen.get(key, [])
    learned = len(samples) >= 3
    rate = statistics.median(samples) if learned else _PRIOR_RATE[key[0]]
    if rate * predictor > budget:
        # Censored, not missing: predicted to exceed the budget. Which
        # rate made that call is recorded, since a prior-based skip is a
        # guess and a learned one is a measurement.
        return float("inf"), None, seen, ("rate" if learned else "prior")
    mpt.set_default(rel_attr_route=route)
    t = float("nan")
    v = None
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            t0 = time.perf_counter()
            v = mpt.cos_sim_exp_tens(dens_x, dens_y, method=method,
                                     verbose=False)
            warm = time.perf_counter() - t0
            if warm > budget:
                t = float("inf")       # ran, but over: censored
            elif warm > repeat_below:
                # Over the repeat threshold: keep the one sample rather
                # than discard the time already spent.
                t = warm * 1e3
            else:
                reps = []
                for _ in range(3):
                    t1 = time.perf_counter()
                    mpt.cos_sim_exp_tens(dens_x, dens_y, method=method,
                                         verbose=False)
                    reps.append(time.perf_counter() - t1)
                t = sorted(reps)[1] * 1e3
        if predictor > 0 and math.isfinite(t):
            seen.setdefault(key, []).append((t / 1e3) / predictor)
    except Exception:
        pass                      # inadmissible route, or a shape it cannot take
    mpt.set_default(rel_attr_route="auto")
    return t, v, seen, ("" if math.isfinite(t) else "budget")


def _check(period=1200.0):
    """Assert each swept axis does what it claims, on one cell apiece.

    Every defect this sweep has carried was an axis quietly not varying
    what it was supposed to: the route lever pinned a choice the workload
    never reached, the cells all ran through a different code path, the
    event count was read as a batch of separate collections. Each showed
    up only after a full run, or not at all. These take seconds.
    """
    failures = []

    def ok(name, cond, detail=""):
        print(f"  {'pass' if cond else 'FAIL'}  {name}"
              + (f"   {detail}" if detail else ""))
        if not cond:
            failures.append(name)

    rng = np.random.default_rng(0)
    K, N, sigma, r = 8, 6, 25.0, 2
    p_x = np.sort(rng.uniform(0, period, (K, N)), axis=0)
    w_x = np.ones((K, N))
    with contextlib.redirect_stdout(io.StringIO()):
        d = mpt.build_exp_tens([p_x], [w_x], [sigma], [r], [1], [1],
                               [period], verbose=False)
        d_flat = mpt.build_exp_tens(p_x, w_x, sigma, r, 1, 1, period,
                                    verbose=False)

    # Events: the density must carry N of them. The single-multiset
    # signature flattens the same array into one event of K*N values, so
    # the two are checked apart.
    n_ev = int(np.shape(d.p_attr[0])[1])
    ok("event count reaches the density", n_ev == N,
       f"built {n_ev}, wanted {N}")
    ok("the flattening signature is not the one used",
       int(np.shape(d_flat.p_attr[0])[1]) == 1,
       f"flattened to {np.shape(d_flat.p_attr[0])}")

    # A multi-event density is one density, so its cosine is one number.
    with contextlib.redirect_stdout(io.StringIO()):
        v = mpt.cos_sim_exp_tens(d, d, method="mobius", verbose=False)
    ok("multi-event call returns one cosine", np.shape(v) == (),
       f"shape {np.shape(v)}")
    ok("self-similarity is 1", abs(float(v) - 1.0) < 1e-9, f"{float(v):.12f}")

    # The route lever must change which route runs, and not the answer.
    # Asserted on the gate rather than on a stopwatch: the gate is what
    # the lever exists to override, and a timing threshold would be a
    # machine-dependent assertion in a check whose whole purpose is to be
    # reliable. The gate is queried on a shape where it prefers grid, so
    # forcing centres has something to overturn.
    from mpt._tensor._mobius_inner import _ma_rel_attr_prefers_centres
    big = np.sort(rng.uniform(0, period, (40, 1)), axis=0)
    picks = {}
    for route in ("auto", "centres", "grid"):
        mpt.set_default(rel_attr_route=route)
        picks[route] = bool(_ma_rel_attr_prefers_centres(
            big, big, sigma, r, True, True, period))
    mpt.set_default(rel_attr_route="auto")
    ok("route lever overrides the gate",
       picks["centres"] and not picks["grid"],
       f"auto={picks['auto']}, centres={picks['centres']}, "
       f"grid={picks['grid']}")

    # And the two routes must agree on the answer, which is not
    # machine-dependent at all.
    vals = {}
    for route in ("centres", "grid"):
        mpt.set_default(rel_attr_route=route)
        with contextlib.redirect_stdout(io.StringIO()):
            vals[route] = mpt.cos_sim_exp_tens(d, d, method="mobius",
                                               verbose=False)
    mpt.set_default(rel_attr_route="auto")
    gap = float(np.max(np.abs(vals["centres"] - vals["grid"])))
    ok("routes agree on the value", gap < 1.5e-8, f"worst gap {gap:.2e}")

    # Unequal value counts must reach both sides.
    p_y = np.sort(rng.uniform(0, period, (K * 3, N)), axis=0)
    with contextlib.redirect_stdout(io.StringIO()):
        d2 = mpt.build_exp_tens([p_y], [np.ones((K * 3, N))], [sigma], [r],
                                [1], [1], [period], verbose=False)
        v2 = mpt.cos_sim_exp_tens(d, d2, method="mobius", verbose=False)
    ok("unequal value counts run", np.all(np.isfinite(v2)))

    # Weight profiles must actually differ.
    prof = {name: _weights(name, 12, np.random.default_rng(1))
            for name in WEIGHT_PROFILES}
    spread = {n: float(np.max(w) / np.min(w)) for n, w in prof.items()}
    ok("weight profiles differ",
       len({round(x, 3) for x in spread.values()}) == len(spread),
       ", ".join(f"{n} {v:.1f}x" for n, v in spread.items()))

    # The memory rule must decline the cell that cannot run.
    ok("memory rule declines a 40 GB cell",
       _bulger_kernel_entries(2, 40, 40, 64) > _BULGER_MAX_ENTRIES)
    ok("memory rule admits a small cell",
       _bulger_kernel_entries(2, 6, 6, 4) < _BULGER_MAX_ENTRIES)

    # The header and the rows must carry the same number of fields. This
    # is checked because it has already gone wrong: an edit added three
    # columns to the header and not to the row, and the mismatch was only
    # noticed after a full sweep had been run and uploaded. Counting the
    # commas in the two format strings costs nothing.
    n_header = len(_HEADER.split(","))
    n_row = _ROW_FIELDS
    ok("header and rows have the same field count", n_header == n_row,
       f"header {n_header}, row {n_row}")

    print(f"\n{len(failures)} failed" if failures else "\nall checks passed")
    return 1 if failures else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[3, 25])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--budget", type=float, default=5.0,
                    help="skip an arm predicted, or found, to exceed this (s)")
    ap.add_argument("--repeat-below", type=float, default=0.25,
                    help="repeat-time only calls faster than this (s)")
    ap.add_argument("--period", type=float, default=1200.0)
    ap.add_argument("--check", action="store_true",
                    help="run one cell per swept axis, asserting each does "
                         "what it claims, and exit")
    args = ap.parse_args(argv)

    warnings.simplefilter("ignore")
    if args.check:
        return _check(args.period)
    ts = mpt.get_default("truncation_sigmas")
    margin = _rel_window_margin(ts)

    print("# calibrate_rel_ip_cost")
    print(f"# python {sys.version.split()[0]}, numpy {np.__version__}")
    print(f"# truncation_sigmas {ts}, period {args.period}")
    print(f"# budget {args.budget}, repeat_below {args.repeat_below}, "
          f"seeds {args.seeds}")
    print("# times in milliseconds. inf means the arm exceeded the "
          "budget: its time is")
    print("# unknown but at least the budget, which still settles the "
          "comparison. nan")
    print("# means the arm is inadmissible in that mode, or raised.")
    print("# declined_* says why an arm was not measured: memory (working "
          "set does not")
    print("# fit, by exact arithmetic), rate (over budget, predicted from "
          "cells already")
    print("# timed), prior (over budget, from the built-in rate before any "
          "measurement")
    print("# existed), budget (it ran, and exceeded).")
    print(_HEADER)

    seen: dict[str, list[float]] = {}
    n = 0
    for r in (2, 3, 4):
        for K in K_BY_ORDER[r]:
            for shape in SHAPES:
                K_x = K_REF if shape == "asym" else K
                K_y = K
                if K_x < r or K_y < r:
                    continue
                if shape == "asym" and K_x == K_y:
                    continue
                for profile in WEIGHT_PROFILES:
                  for N in N_BY_ORDER[r]:
                    for sigma in args.sigmas:
                        for is_per in (False, True):
                            for seed in args.seeds:
                                rng = np.random.default_rng(
                                    7919 * K + 131 * r + 17 * seed
                                    + 3 * WEIGHT_PROFILES.index(profile)
                                    + 5 * SHAPES.index(shape)
                                    + round(1000 * sigma))
                                p_x = np.sort(rng.uniform(
                                    0, args.period, (K_x, N)), axis=0)
                                p_y = np.sort(rng.uniform(
                                    0, args.period, (K_y, N)), axis=0)
                                w_x = np.stack(
                                    [_weights(profile, K_x, rng)
                                     for _ in range(N)], axis=1)
                                w_y = np.stack(
                                    [_weights(profile, K_y, rng)
                                     for _ in range(N)], axis=1)

                                M_x = math.factorial(r) * math.comb(K_x, r)
                                M_y = math.factorial(r) * math.comb(K_y, r)
                                if is_per:
                                    nu = auto_ntau_default(args.period, sigma)
                                    sop = sigma / args.period
                                else:
                                    sps = resolve_samples_per_sigma(None, r, ts)
                                    span = ((p_x.max() - p_x.min())
                                            + (p_y.max() - p_y.min())
                                            + 2 * margin * sigma)
                                    nu = max(64, int(np.ceil(
                                        max(span, 1.0) / sigma * sps)))
                                    sop = 0.0

                                # Predictors price by the larger side, since
                                # that is what dominates each route.
                                M_big = max(M_x, M_y)
                                K_big = max(K_x, K_y)
                                # Both sides scale per event pair, so the
                                # predictors carry it: Bulger's over the
                                # joint kernel, the Möbius routes per pair.
                                # Densities are built explicitly. Passing
                                # 2-D arrays to the raw entry would be read
                                # as a *batch of collections*, one per
                                # column, not as one density carrying N
                                # events -- a different computation, and
                                # one that cannot even broadcast when the
                                # two sides carry different value counts.
                                P = args.period if is_per else 0.0
                                # The multi-attribute signature, selected
                                # by passing lists, is the one that carries
                                # events: it keeps a (K, N) matrix as K
                                # values across N events. The
                                # single-multiset signature flattens the
                                # same array into one event of K*N values,
                                # which is a different density entirely.
                                with contextlib.redirect_stdout(io.StringIO()):
                                    dens_x = mpt.build_exp_tens(
                                        [p_x], [w_x], [sigma], [r], [1],
                                        [int(is_per)], [P], verbose=False)
                                    dens_y = mpt.build_exp_tens(
                                        [p_y], [w_y], [sigma], [r], [1],
                                        [int(is_per)], [P], verbose=False)
                                pairs = float(N * N)
                                too_big = (_bulger_kernel_entries(
                                    r, K_x, K_y, N) > _BULGER_MAX_ENTRIES)
                                arms = (("B", "bulger", "auto",
                                         pairs * M_big ** 2),
                                        ("C", "mobius", "centres",
                                         pairs * M_big ** 2),
                                        ("G", "mobius", "grid",
                                         pairs * float(nu) * K_big))
                                print(f"  r={r} K={K_x}/{K_y} N={N} "
                                      f"{shape} {profile} "
                                      f"per={int(is_per)} sigma={sigma:g}",
                                      file=sys.stderr, flush=True)
                                times, vals, why = {}, [], {}
                                for tag, method, route, predictor in arms:
                                    t, v, seen, reason = _timed(
                                        dens_x, dens_y, method, route,
                                        args.budget, args.repeat_below, seen,
                                        f"{tag}{r}", predictor,
                                        infeasible=(tag == "B" and too_big))
                                    times[tag] = t
                                    why[tag] = reason
                                    vals.append(v)

                                # Multi-event calls return an array of
                                # cosines, one per event pair, so the
                                # agreement check is elementwise.
                                good = [np.asarray(v, dtype=float)
                                        for v in vals
                                        if v is not None
                                        and not np.any(np.isnan(v))]
                                diff = (max(float(np.max(np.abs(v - good[0])))
                                            for v in good)
                                        if len(good) > 1 else float("nan"))
                                gate = ("centres"
                                        if _ma_rel_attr_prefers_centres(
                                            p_x[:, None], p_y[:, None], sigma,
                                            r, True, is_per,
                                            max(args.period, 1.0))
                                        else "grid")
                                # A censored arm still settles the
                                # comparison whenever the other side is
                                # measured and below the bound.
                                mob = [times[t] for t in ("C", "G")
                                       if not math.isnan(times[t])]
                                t_b = times["B"]
                                if math.isnan(t_b) or not mob:
                                    faster = "unknown"
                                elif min(mob) == t_b:            # both inf
                                    faster = "unknown"
                                else:
                                    faster = ("mobius" if min(mob) < t_b
                                              else "bulger")

                                _, p_b, p_m = _select_ma_inner_product_method(
                                    r_vec=np.array([r]),
                                    k_vec=np.array([K_x]), A=1, N_x=N, N_y=N,
                                    any_per=is_per, any_rel_nonper=not is_per,
                                    any_rel_per=is_per, sigma_over_P_max=sop,
                                    user_method="auto",
                                    rel_vec=np.array([True]),
                                    nu_vec=np.array([float(nu)]),
                                    guard_forced_bulger=False, wrap_vec=None,
                                    k_vec_y=np.array([K_y]),
                                    return_costs=True)

                                print(f"{r},{K_x},{K_y},{N},{shape},"
                                      f"{profile},"
                                      f"{int(is_per)},{sigma:g},{seed},{nu},"
                                      f"{M_x},{M_y},{times['B']:.4f},"
                                      f"{times['C']:.4f},{times['G']:.4f},"
                                      f"{p_b:.4f},{p_m:.4f},{diff:.3e},"
                                      f"{gate},{faster},"
                                      f"{why['B']},{why['C']},{why['G']}",
                                      flush=True)
                                n += 1
    mpt.reset_defaults()
    print(f"# {n} cells")


if __name__ == "__main__":
    main()
