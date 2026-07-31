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
K_BY_ORDER = {2: [6, 10, 16, 24, 40, 64], 3: [6, 8, 12, 16, 24],
              4: [5, 6, 8, 10, 12]}


def _timed(p_x, w_x, p_y, w_y, sigma, r, is_per, period, method, route,
           budget, repeat_below, seen, key, predictor):
    """Time one arm, or decline to start it. Returns (ms, value, seen)."""
    if key in seen and len(seen[key]) >= 3:
        if statistics.median(seen[key]) * predictor > budget:
            return float("nan"), float("nan"), seen
    P = period if is_per else 0.0
    mpt.set_default(rel_attr_route=route)
    t = v = float("nan")
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            t0 = time.perf_counter()
            v = mpt.cos_sim_exp_tens(p_x, w_x, p_y, w_y, sigma, r, 1,
                                     int(is_per), P, method=method,
                                     verbose=False)
            warm = time.perf_counter() - t0
            if warm > repeat_below:
                # Over the repeat threshold: keep the one sample rather
                # than discard the time already spent.
                t = warm * 1e3
            else:
                reps = []
                for _ in range(3):
                    t1 = time.perf_counter()
                    mpt.cos_sim_exp_tens(p_x, w_x, p_y, w_y, sigma, r, 1,
                                         int(is_per), P, method=method,
                                         verbose=False)
                    reps.append(time.perf_counter() - t1)
                t = sorted(reps)[1] * 1e3
        if predictor > 0:
            seen.setdefault(key, []).append((t / 1e3) / predictor)
    except Exception:
        pass                      # inadmissible route, or a shape it cannot take
    mpt.set_default(rel_attr_route="auto")
    return t, v, seen


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sigmas", type=float, nargs="+",
                    default=[2, 3, 6, 12, 25, 50])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--budget", type=float, default=5.0,
                    help="skip an arm predicted, or found, to exceed this (s)")
    ap.add_argument("--repeat-below", type=float, default=0.25,
                    help="repeat-time only calls faster than this (s)")
    ap.add_argument("--period", type=float, default=1200.0)
    args = ap.parse_args(argv)

    warnings.simplefilter("ignore")
    ts = mpt.get_default("truncation_sigmas")
    margin = _rel_window_margin(ts)

    print("# calibrate_rel_ip_cost")
    print(f"# python {sys.version.split()[0]}, numpy {np.__version__}")
    print(f"# truncation_sigmas {ts}, period {args.period}")
    print(f"# budget {args.budget}, repeat_below {args.repeat_below}, "
          f"seeds {args.seeds}")
    print("# times in milliseconds; nan means the arm was skipped as over "
          "budget")
    print("# or is inadmissible in that mode")
    print("r,K,isPer,sigma,seed,nu,M,t_bulger,t_centres,t_grid,"
          "pred_bulger,pred_mobius,max_abs_diff,gate_route,faster")

    seen: dict[str, list[float]] = {}
    n = 0
    for r in (2, 3, 4):
        for K in K_BY_ORDER[r]:
            for sigma in args.sigmas:
                for is_per in (False, True):
                    for seed in args.seeds:
                        rng = np.random.default_rng(
                            7919 * K + 131 * r + 17 * seed
                            + round(1000 * sigma))
                        p_x = np.sort(rng.uniform(0, args.period, K))
                        p_y = np.sort(rng.uniform(0, args.period, K))
                        w_x = 0.5 + rng.uniform(0, 1, K)
                        w_y = 0.5 + rng.uniform(0, 1, K)

                        M = math.factorial(r) * math.comb(K, r)
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

                        arms = (("B", "bulger", "auto", M ** 2),
                                ("C", "mobius", "centres", M ** 2),
                                ("G", "mobius", "grid", float(nu)))
                        times, vals = {}, []
                        for tag, method, route, predictor in arms:
                            t, v, seen = _timed(
                                p_x, w_x, p_y, w_y, sigma, r, is_per,
                                args.period, method, route, args.budget,
                                args.repeat_below, seen, f"{tag}{r}",
                                predictor)
                            times[tag] = t
                            vals.append(v)

                        good = [v for v in vals if not math.isnan(v)]
                        diff = (max(abs(v - good[0]) for v in good)
                                if len(good) > 1 else float("nan"))

                        gate = ("centres" if _ma_rel_attr_prefers_centres(
                            p_x[:, None], p_y[:, None], sigma, r, True,
                            is_per, max(args.period, 1.0)) else "grid")
                        mob = [times[t] for t in ("C", "G")
                               if not math.isnan(times[t])]
                        if math.isnan(times["B"]) or not mob:
                            faster = "unknown"
                        else:
                            faster = ("mobius" if min(mob) < times["B"]
                                      else "bulger")

                        _, p_b, p_m = _select_ma_inner_product_method(
                            r_vec=np.array([r]), k_vec=np.array([K]), A=1,
                            N_x=1, N_y=1, any_per=is_per,
                            any_rel_nonper=not is_per, any_rel_per=is_per,
                            sigma_over_P_max=sop, user_method="auto",
                            rel_vec=np.array([True]),
                            nu_vec=np.array([float(nu)]),
                            guard_forced_bulger=False, wrap_vec=None,
                            k_vec_y=np.array([K]), return_costs=True)

                        print(f"{r},{K},{int(is_per)},{sigma:g},{seed},"
                              f"{nu},{M},{times['B']:.4f},{times['C']:.4f},"
                              f"{times['G']:.4f},{p_b:.4f},{p_m:.4f},"
                              f"{diff:.3e},{gate},{faster}", flush=True)
                        n += 1
    mpt.reset_defaults()
    print(f"# {n} cells")


if __name__ == "__main__":
    main()
