"""Audit of the multi-attribute eval dispatcher: is the pick the faster arm?

Twin of matlab/tests/bench_ma_eval_dispatch.m. ``_select_ma_eval``
routes ``eval_maet`` between the factored Möbius evaluator and the
centres routes (the joint-centres path where an attribute is at r = 1,
the factored centres route otherwise); this times both arms on a shape
grid spanning the crossover, single and multi-attribute, one event and
many, and reports where the pick disagrees with the measurement.

It does NOT fit anything: ``tools/calibrate_ma_eval_cost.py`` measures
the grid the ``_MA_COST_*`` constants are fitted from, and
``tools/fit_ma_eval_cost.py`` fits them. This is the quick check that
the fit still holds on this machine, including on shapes the fit's
single-multiset grid does not contain.

Read the report by magnitude, not by the mispick count alone: a model
out by 40x that still orders two routes correctly scores no mispick,
while one out by 1.01x near a crossover scores one. ``regret`` is the
time of the picked arm over the faster one.

HOW TO RUN
----------
From the ``python`` directory, with the package importable::

    PYTHONPATH=. python3 tools/bench_ma_eval_dispatch.py

Each arm is the minimum of five runs after a warm-up.
"""
import time
from math import comb, factorial

import numpy as np

import mpt
from mpt._tensor.dispatch import _eval_costs_ms, _select_ma_eval

N_Q = 200
BELL = [1, 2, 5, 15, 52, 203, 877, 4140, 21147, 115975]

# (label, sigma, r, rel, per, period, K, N); K is one value for every
# attribute or one per attribute; values are uniform on [0, 100]. The
# cells with an attribute at r = 1 take the joint-centres path; the cells
# with N > 1 check the pricing of the routes that take a density event
# by event. Same grid as the MATLAB twin.
GRID = [
    ("A1 r2 K6", [30], [2], [0], [0], [0], 6, 1),
    ("A1 r2 K10", [30], [2], [0], [0], [0], 10, 1),
    ("A1 r2 K20", [30], [2], [0], [0], [0], 20, 1),
    ("A1 r3 K6", [30], [3], [0], [0], [0], 6, 1),
    ("A1 r3 K8", [30], [3], [0], [0], [0], 8, 1),
    ("A1 r3 K12", [30], [3], [0], [0], [0], 12, 1),
    ("A2 r2 K5", [30, 25], [2, 2], [0, 0], [0, 0], [0, 0], 5, 1),
    ("A2 r2 K8", [30, 25], [2, 2], [0, 0], [0, 0], [0, 0], 8, 1),
    ("A2 r2 K12", [30, 25], [2, 2], [0, 0], [0, 0], [0, 0], 12, 1),
    ("A2 r3 K6", [30, 25], [3, 3], [0, 0], [0, 0], [0, 0], 6, 1),
    ("A2 r3 K10", [30, 25], [3, 3], [0, 0], [0, 0], [0, 0], 10, 1),
    ("A1 rel r2 K8", [30], [2], [1], [0], [0], 8, 1),
    ("A1 rel r3 K8", [30], [3], [1], [0], [0], 8, 1),
    ("A2 r2 K6 N20", [30, 25], [2, 2], [0, 0], [0, 0], [0, 0], 6, 20),
    ("A2 r2 K12 N20", [30, 25], [2, 2], [0, 0], [0, 0], [0, 0], 12, 20),
    ("A2 r3 K8 N20", [30, 25], [3, 3], [0, 0], [0, 0], [0, 0], 8, 20),
    ("A2 rel r2 K8 N10", [30, 25], [2, 2], [1, 1], [0, 0], [0, 0], 8, 10),
    ("r1+r2 N30", [0.3, 15], [1, 2], [0, 0], [0, 0], [0, 0], [1, 3], 30),
    ("r1+r2 N300", [0.3, 15], [1, 2], [0, 0], [0, 0], [0, 0], [1, 3], 300),
    ("r1+r2 N3000", [0.3, 15], [1, 2], [0, 0], [0, 0], [0, 0], [1, 3], 3000),
    ("r1+r2 wide N300", [5, 15], [1, 2], [0, 0], [0, 0], [0, 0], [1, 3], 300),
    ("r1+rel r3 N300", [0.3, 15], [1, 3], [0, 1], [0, 0], [0, 0], [1, 4], 300),
    ("r1+r3 K10 N20", [15, 15], [1, 3], [0, 0], [0, 0], [0, 0], [10, 10], 20),
    ("r1+r2 K12 N50", [15, 15], [1, 2], [0, 0], [0, 0], [0, 0], [12, 12], 50),
    ("r1x2 K4 N200", [3, 3], [1, 1], [0, 0], [0, 0], [0, 0], [4, 4], 200),
    ("r1x2 K8 N200", [3, 3], [1, 1], [0, 0], [0, 0], [0, 0], [8, 8], 200),
    ("r1x2 K20 N200", [3, 3], [1, 1], [0, 0], [0, 0], [0, 0], [20, 20], 200),
    ("r1x2 K20 N200 narrow", [0.3, 0.3], [1, 1], [0, 0], [0, 0], [0, 0],
     [20, 20], 200),
    ("r1x2 K50 N200", [3, 3], [1, 1], [0, 0], [0, 0], [0, 0], [50, 50], 200),
    ("r1x3 K4 N200", [3, 3, 3], [1, 1, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0],
     [4, 4, 4], 200),
]


def _time_ms(fn, repeats=5):
    fn()
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def main():
    prev = mpt.get_default("show_hints")
    mpt.set_default(show_hints=False)
    try:
        rng = np.random.default_rng(0)
        print(f"{'cell':22s} {'orbit':>8s} {'joint':>9s} {'cen_ms':>9s} "
              f"{'mob_ms':>9s} {'pred_cen':>9s} {'pred_mob':>9s} "
              f"{'faster':>8s} {'predict':>8s} {'regret':>7s}")
        print("-" * 104)
        mispicks = 0
        regrets = []
        for label, sig, rv, rel, per, period, K, N in GRID:
            A = len(sig)
            Ks = K if isinstance(K, list) else [K] * A
            ps = [rng.uniform(0.0, 100.0, (Ks[a], N)) for a in range(A)]
            dens = mpt.build_maet(ps, None, sig, rv, [bool(v) for v in rel],
                                  [bool(v) for v in per], period,
                                  verbose=False)
            xq = rng.uniform(0.0, 100.0, (dens.dim, N_Q))
            pred, _ = _select_ma_eval(dens, N_Q, method="auto")
            pc, pm = _eval_costs_ms(dens, N_Q)
            joint = N
            orbit = 0
            for a in range(A):
                joint *= factorial(rv[a]) * comb(Ks[a], rv[a])
                orbit += N * BELL[rv[a] - 1] * rv[a] * Ks[a]
            t_c = _time_ms(lambda: mpt.eval_maet(dens, xq, method="centres",
                                                 verbose=False))
            t_m = _time_ms(lambda: mpt.eval_maet(dens, xq, method="mobius",
                                                 verbose=False))
            faster = "centres" if t_c < t_m else "mobius"
            regret = (t_c if pred == "centres" else t_m) / min(t_c, t_m)
            regrets.append(regret)
            within = abs(t_c - t_m) / max(min(t_c, t_m), 1e-9) < 0.25
            ok = pred == faster or within
            mispicks += 0 if ok else 1
            print(f"{label:22s} {orbit:8d} {joint:9d} {t_c:9.2f} {t_m:9.2f} "
                  f"{pc:9.2f} {pm:9.2f} {faster:>8s} {pred:>8s} "
                  f"{regret:7.2f}{'' if ok else '  <-- MISPICK'}",
                  flush=True)
        print("-" * 104)
        print(f"mispicks outside 25% noise band: {mispicks} / {len(GRID)}; "
              f"regret geometric mean {np.exp(np.mean(np.log(regrets))):.3f}, "
              f"worst {max(regrets):.2f}")
    finally:
        mpt.set_default(show_hints=prev)


if __name__ == "__main__":
    main()
