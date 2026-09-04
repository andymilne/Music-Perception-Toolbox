"""Timing grid for calibrating the multi-attribute eval cost model.

Twin of matlab/tests/bench_ma_eval_calibration.m. The cost model in
``mpt._tensor.dispatch`` routes ``eval_exp_tens`` between the
joint-centres accumulator and the factored Moebius evaluator, and its
constants absorb per-language constant factors -- BLAS, interpreter
overhead, array layout -- so each language must be calibrated on its own
measurements. This script MEASURES the grid; it does not fit the
constants.

HOW TO RUN
----------
From the ``python`` directory, with the package importable::

    PYTHONPATH=. python3 tools/calibrate_ma_eval_cost.py

The run turns ``show_hints`` off and restores it afterwards. Dispatch
announcements are gated by that default rather than by the per-call
``verbose`` argument, so passing ``verbose=False`` does not silence
them; left on, each announcement costs about 0.13 ms, which on the
cheapest cells is a third of the work being timed and would be
measured as if it were part of it.

Run it on a quiet machine. Timing is by the median of several repeats
after discarding warm-up runs, which removes scatter but not a loaded
machine's systematic slowdown.

WHAT TO SEND BACK
-----------------
Everything between the BEGIN_CSV and END_CSV markers, inclusive of the
header row. That block alone is sufficient to fit the constants.

FIVE SECTIONS
-------------
Section A sweeps shape -- mode, tuple order, value count, query count --
at one reference geometry.

Section B sweeps the geometry, sigma and the value span, over a reduced
shape set. The centres per-query cost is culled by a factor set by sigma
against the source spread, so a grid at one geometry cannot constrain
that factor: it fixes the quantity the term is a function of.

Section C traces the relative-mode node count alone. The Moebius
relative evaluator integrates over a grid whose node count is set by the
period (or the source span) over sigma, and the cost model prices its
per-query work linearly in that count. Measurement on the MATLAB side
puts the exponent nearer 0.55, so this section holds the shape fixed and
walks sigma over a wide range, which is what separates the node-count
exponent from everything else that moves with sigma.

Section D exercises the spectral strategy inside the Moebius relative
evaluator, whose per-mode constants are per tuple order. The branch
engages only above its query and value-count thresholds and only where
the mode grid fits under the memory guard -- which at r = 4 needs sigma
at least P/78 periodically, or a short span with a wide kernel
otherwise. Sections A and B satisfy that in one cell per periodicity at
r = 4, so its constant would rest on two measurements; these geometries
are chosen so every tuple order gets a proper sample.

Section E walks the relative-periodic r = 3 family through the K = 34
crossover at sigma/P = 0.0083, the cell the September 2026 audit found
misrouted; it is what constrains the rel-per per-query exponent.

Single-attribute cells only: the multi-attribute contrast is
product-against-sum, structurally Moebius-dominated and insensitive to
the constants. The crossover the constants must place lives in the
single-attribute grid.
"""
import math
import time

import numpy as np

import mpt

PERIOD = 1200.0
SIGMA_REF = 15.0
SPAN_REF = 3600.0

# Centres is not timed where it would cost too much to be worth the
# wall time. Two bounds, whichever bites first: the joint tuple count,
# beyond which centres is decisively the wrong pick; and the work the
# centres arm actually does, which is the joint count times the query
# count. Measured on the spectral-branch shapes, one centres call runs
# 35 ms at 1e5 units of work, 1.6 s at 5e6 and 10.6 s at 4e7, so the
# work bound is what keeps the run finite where the joint bound alone
# would not. Cells not timed carry cen_ms = -1.
JOINT_SKIP = 3e5
WORK_SKIP = 4e6

N_DISCARD = 3
N_TIMED = 7
BUDGET_SEC = 2.0


def _time_ms(fn):
    """Median of ``N_TIMED`` runs after warm-ups.

    Stops early once the budget is spent and at least three timed runs
    are in hand, mirroring internal.timeRepeated on the MATLAB side.
    The warm-up count is cut to one where a single call already costs
    more than the budget: on those cells the repeats buy nothing --- the
    scatter they remove is a fixed overhead whose share of a call that
    long is negligible --- and paying three of them dominates the run.
    """
    t0 = time.perf_counter()
    fn()
    n_discard = 1 if (time.perf_counter() - t0) > BUDGET_SEC else N_DISCARD
    for _ in range(n_discard - 1):
        fn()
    samples = []
    spent = 0.0
    for _ in range(N_TIMED):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        samples.append(dt * 1000.0)
        spent += dt
        if spent > BUDGET_SEC and len(samples) >= 3:
            break
    return sorted(samples)[len(samples) // 2]


def _cell(rows, rng, is_rel, is_per, r, K, n_q_vals, sigma, span):
    """Time both arms for one (mode, shape, geometry) cell."""
    period = PERIOD if is_per else 0.0
    extent = PERIOD if is_per else span
    joint = math.factorial(r) * math.comb(K, r)
    p = np.sort(rng.uniform(0.0, extent, K))
    w = 0.2 + 0.8 * rng.random(K)
    dens = mpt.build_exp_tens(p, w, sigma, r, is_rel, is_per, period,
                              verbose=False)
    dim = r - 1 if is_rel else r
    for n_q in n_q_vals:
        x = rng.uniform(0.0, extent, (dim, n_q))
        if joint <= JOINT_SKIP and joint * n_q <= WORK_SKIP:
            cen = _time_ms(lambda: mpt.eval_exp_tens(
                dens, x, method="centres", verbose=False))
        else:
            cen = -1.0
        mob = _time_ms(lambda: mpt.eval_exp_tens(
            dens, x, method="mobius", verbose=False))
        # Where centres was not timed there is no comparison to report:
        # the cell says which arm was faster only when both were run.
        faster = "-" if cen < 0 else ("centres" if cen < mob else "mobius")
        print(f"{int(is_rel):<4d} {int(is_per):<4d} {r:<3d} {K:<4d} {n_q:<5d} "
              f"{sigma:<6.0f} {extent:<6.0f} {joint:>10d} {cen:>10.4f} "
              f"{mob:>10.4f} {faster:<8s}", flush=True)
        rows.append(f"{int(is_rel)},{int(is_per)},{r},{K},{n_q},"
                    f"{sigma:g},{extent:g},{joint},{cen:.4f},{mob:.4f}")


def main():
    # Dispatch announcements are gated by show_hints, not by the
    # per-call verbose argument (see the note in _tensor/eval.py), and
    # printing one per call would be timed along with the work.
    prev_hints = mpt.get_default("show_hints")
    mpt.set_default(show_hints=False)
    try:
        _run()
    finally:
        mpt.set_default(show_hints=prev_hints)


def _run():
    rng = np.random.default_rng(0)
    rows = []
    modes = [(False, False), (True, False), (False, True), (True, True)]
    n_q_vals = [1, 200]

    header = (f"{'rel':<4s} {'per':<4s} {'r':<3s} {'K':<4s} {'nQ':<5s} "
              f"{'sigma':<6s} {'span':<6s} {'joint':>10s} {'cen_ms':>10s} "
              f"{'mob_ms':>10s} {'faster':<8s}")

    print(f"\n--- Section A: shape sweep at sigma = {SIGMA_REF:g} "
          f"over a span of {SPAN_REF:g} ---")
    print(header)
    for is_rel, is_per in modes:
        for r in (2, 3, 4):
            for K in (6, 12, 24, 48):
                _cell(rows, rng, is_rel, is_per, r, K, n_q_vals,
                      SIGMA_REF, SPAN_REF)

    print("\n--- Section B: geometry sweep (sigma, span) ---")
    print(header)
    for is_rel, is_per in modes:
        for sigma in (5.0, 15.0, 60.0):
            spans = (PERIOD,) if is_per else (1200.0, 3600.0, 9600.0)
            for span in spans:
                if is_per and sigma == SIGMA_REF:
                    continue
                if not is_per and sigma == SIGMA_REF and span == SPAN_REF:
                    continue
                for r in (3, 4):
                    for K in (12, 24):
                        _cell(rows, rng, is_rel, is_per, r, K, n_q_vals,
                              sigma, span)

    print("\n--- Section C: node-count trace (relative modes) ---")
    print(header)
    for is_rel, is_per in ((True, True), (True, False)):
        # Avoids Section B's 5, 15 and 60, so no cell is measured twice.
        for sigma in (3.0, 7.0, 10.0, 20.0, 40.0, 80.0):
            span = PERIOD if is_per else SPAN_REF
            for r in (3, 4):
                for K in (12, 24):
                    _cell(rows, rng, is_rel, is_per, r, K, [200],
                          sigma, span)

    print("\n--- Section D: spectral branch (relative modes) ---")
    print(header)
    for sigma in (20.0, 40.0, 80.0):
        for r in (2, 3, 4):
            for K in (16, 24, 48):
                _cell(rows, rng, True, True, r, K, [100, 400],
                      sigma, PERIOD)
    for sigma, span in ((40.0, 600.0), (80.0, 600.0),
                        (60.0, 1200.0), (120.0, 1200.0)):
        for r in (2, 3, 4):
            for K in (16, 24, 48):
                _cell(rows, rng, True, False, r, K, [100, 400],
                      sigma, span)

    print("\n--- Section E: relative-periodic r = 3 family around K = 34 ---")
    print(header)
    # The September 2026 audit found auto picking centres at rel-per
    # r = 3, K = 34, sigma/P = 0.0083 where Moebius was faster: the
    # rel-per centres per-query cost grows superlinearly in the joint
    # tuple count (~T^1.3) and the shared linear slope could not express
    # it. This family walks K through the crossover at that geometry so
    # a refit sees it.
    for K in (12, 20, 28, 34, 40, 48):
        _cell(rows, rng, True, True, 3, K, [24, 200], 10.0, PERIOD)

    print("\nBEGIN_CSV")
    print("rel,per,r,K,nQ,sigma,span,joint,cen_ms,mob_ms")
    for row in rows:
        print(row)
    print("END_CSV")
    print(f"\ncen_ms = -1 marks a cell where centres was not timed: "
          f"joint > {JOINT_SKIP:g}, or joint x nQ > {WORK_SKIP:g}. "
          f"The faster column reads '-' there, since only one arm ran.")
    print(f"{len(rows)} cells.")


if __name__ == "__main__":
    main()
