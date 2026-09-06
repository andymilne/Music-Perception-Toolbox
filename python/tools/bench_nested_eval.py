"""Timing grid for pricing the per-level Möbius evaluator on nested densities.

Twin of matlab/tools/benchNestedEval.m. ``eval_exp_tens`` on a nested
density has two routes: the tuple-centres route (materialise every
nested tuple, ``M_perm`` per event, and sum a Gaussian per centre) and
the per-level Möbius evaluator (``_tensor/_nested_mobius_eval.py``), which
touches no tuple. The eval selector keeps the centres route under
``'auto'`` until the Möbius route has a fitted cost row; this script
MEASURES the grid on which that row is to be fitted. It does not fit.

HOW TO RUN
----------
From the ``python`` directory::

    PYTHONPATH=. python3 tools/bench_nested_eval.py > nested_eval_python.csv

Run it on a quiet machine. Each cell reports the median of repeats after
a warm-up.

COLUMNS
-------
The structural quantities the two routes work over, in the units their
laws would use: ``m_perm`` (nested tuple centres per event), ``d``
(leaf slots), ``k`` (values per event), ``n`` (events), ``n_q``
(queries), ``bell_sum`` (Σ over symmetric levels of the Bell number of
the level's r, the partition count the per-level sum forms),
``n_u`` (translation-grid nodes of the co-transposition unit, 1 for
absolute), ``rel_unit`` (-1 absolute, else the 0-based level), ``per``,
``sym`` (per-level flags as a string), and the two medians in ms.
"""
from __future__ import annotations

import math
import sys
import time

import numpy as np

import mpt
from mpt import build_exp_tens, eval_exp_tens
from mpt._mobius import get_set_partitions_with_mobius

P = 12.0


def _median_ms(fn, repeats=5):
    fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


def _grid_nodes(d, rel_unit, r_levels, sigma, per, ts):
    if rel_unit is None:
        return 1
    from mpt._defaults import resolve_samples_per_sigma
    s = int(np.prod(r_levels[:rel_unit + 1]))
    spp = resolve_samples_per_sigma(None, max(2, s), ts)
    if per:
        return max(64, int(math.ceil(P / sigma * spp)))
    span = float(np.nanmax(d.p_attr[0]) - np.nanmin(d.p_attr[0])) + 16 * sigma
    return max(64, int(math.ceil(max(span, 1.0) / sigma * spp)))


def main():
    prev_hints = mpt.get_default('show_hints')
    mpt.set_default(show_hints=False)
    ts = mpt.get_default('truncation_sigmas')
    rng = np.random.default_rng(0)
    print("BEGIN_CSV")
    print("groups,group_size,r_levels,sym,rel_unit,per,k,n,n_q,d,m_perm,"
          "bell_sum,n_u,centres_ms,mobius_ms")
    cells = []
    for groups, gsize in [(2, 3), (3, 3), (4, 3), (3, 4), (4, 4), (2, 5)]:
        for r_levels in [(1, 2), (2, 2), (2, 3), (3, 2), (3, 3), (1, 3), (2, 4)]:
            if r_levels[0] > gsize or r_levels[1] > groups:
                continue
            for sym in [(True, True), (True, False), (False, True)]:
                for rel_unit in [None, 1, 0]:
                    if rel_unit == 0 and r_levels[0] < 2:
                        continue
                    for per in [False, True]:
                        cells.append((groups, gsize, r_levels, sym, rel_unit, per))
    for groups, gsize, r_levels, sym, rel_unit, per in cells:
        K = groups * gsize
        tags = np.repeat(np.arange(groups), gsize)
        rel = [0, 0]
        if rel_unit is not None:
            rel[rel_unit] = 1
        for N in (4, 32):
            p = np.sort(rng.uniform(0.0, P, size=(K, N)), axis=0)
            spec = {"tags": tags, "r": list(r_levels), "sym": list(sym),
                    "rel": rel}
            sigma = 0.6
            d = build_exp_tens([p], None, specs=[spec], sigma=[sigma],
                               is_per=[per], period=[P], verbose=False)
            if d.dim == 0:
                continue
            m_perm = int(d.n_j // N)
            bell_sum = sum(len(get_set_partitions_with_mobius(int(r)))
                           for r, s in zip(r_levels, sym) if s and r >= 2)
            n_u = _grid_nodes(d, rel_unit, r_levels, sigma, per, ts)
            for n_q in (1, 20, 200):
                X = rng.uniform(0.0, P, size=(d.dim, n_q))
                if m_perm * N * n_q > 4e7:
                    t_c = float('nan')
                else:
                    t_c = _median_ms(lambda: eval_exp_tens(
                        d, X, method="centres", verbose=False))
                t_m = _median_ms(lambda: eval_exp_tens(
                    d, X, method="mobius", verbose=False))
                print(f"{groups},{gsize},{'x'.join(map(str, r_levels))},"
                      f"{''.join('1' if s else '0' for s in sym)},"
                      f"{-1 if rel_unit is None else rel_unit},{int(per)},"
                      f"{K},{N},{n_q},{d.dim},{m_perm},{bell_sum},{n_u},"
                      f"{t_c:.4f},{t_m:.4f}")
                sys.stdout.flush()
    print("END_CSV")
    mpt.set_default(show_hints=prev_hints)


if __name__ == "__main__":
    main()
