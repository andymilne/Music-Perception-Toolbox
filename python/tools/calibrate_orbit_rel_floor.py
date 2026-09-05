"""Measure the Möbius route's per-attribute setup floor (_ORBIT_REL_FLOOR_MS).

The two relative-route cost laws in ``mpt._tensor.dispatch``
(``_REL_COST_LAW``) are multiplicative in their term, so they carry no
fixed setup cost and extrapolate below the route's wall time at small
value counts. ``_ORBIT_REL_FLOOR_MS`` guards against that: for each tuple
order it holds ``(fixed, per_matrix)`` in milliseconds, applied with
``max`` to the per-attribute price of a call computing ``n_matrices`` of
the three inner matrices.

This script measures the pair of coefficients for r = 2, 3, 4 on the
machine it runs on. For each order it times the cosine call alone (the
densities are built outside the timer) on a *cold* pair, where all three
matrices are computed, and on a *warm* pair whose self inner products are
memoised, where only the cross matrix is; the median over repeats is
minimised over the smallest feasible K, where the route is flat in K, and
the two coefficients solve from

    cold = fixed + 3 * per_matrix,     warm = fixed + per_matrix.

Relative-periodic at sigma/P = 0.025 is used because it measured the same
as or lower than relative-non-periodic at every order, so it is the
conservative row.

HOW TO RUN
----------
    python tools/calibrate_orbit_rel_floor.py

and paste the printed ``_ORBIT_REL_FLOOR_MS`` row into
``mpt/_tensor/dispatch.py``. The MATLAB counterpart is
``matlab/tests/bench_orbit_rel_floor.m``, which fills
``ORBIT_REL_FLOOR_MS`` in ``+internal/predictOrbitCostMs.m``. Expect a
factor of two or three of spread between machines: this is a guard, not
a fitted term, and it only decides routing where the fitted laws predict
below it (r = 2 at small K).
"""
import time

import numpy as np

import mpt
from mpt import build_exp_tens, cos_sim_exp_tens

P = 12.0
SIGMA_OVER_P = 0.025
COLD_REPS = 15
WARM_REPS = 25


def _dens(seed, K, r):
    rng = np.random.default_rng(seed)
    p = np.sort(rng.uniform(0.0, P, K))
    w = 0.2 + 0.8 * rng.random(K)
    return build_exp_tens(p, w, SIGMA_OVER_P * P, r, True, True, P,
                          verbose=False)


def _cos(x, y):
    return cos_sim_exp_tens(x, y, method='mobius', verbose=False)


def main():
    mpt.set_default(show_hints=False)
    row = {}
    for r in (2, 3, 4):
        cold, warm = [], []
        for K in range(r, r + 5):
            ts = []
            for _ in range(COLD_REPS):
                x, y = _dens(1, K, r), _dens(2, K, r)
                t0 = time.perf_counter()
                _cos(x, y)
                ts.append(time.perf_counter() - t0)
            cold.append(1e3 * float(np.median(ts)))
            x, y = _dens(1, K, r), _dens(2, K, r)
            _cos(x, y)                      # memoise both self products
            ts = []
            for _ in range(WARM_REPS):
                t0 = time.perf_counter()
                _cos(x, y)
                ts.append(time.perf_counter() - t0)
            warm.append(1e3 * float(np.median(ts)))
        t3, t1 = min(cold), min(warm)
        per_matrix = max((t3 - t1) / 2.0, 0.0)
        fixed = max(t1 - per_matrix, 0.0)
        row[r] = (round(fixed, 3), round(per_matrix, 3))
        print(f"r={r}: cold (3 matrices) min {t3:.3f} ms, "
              f"warm (1 matrix) min {t1:.3f} ms -> fixed {fixed:.3f}, "
              f"per_matrix {per_matrix:.3f}; cold by K: "
              f"{[round(c, 2) for c in cold]}")
    print("\n_ORBIT_REL_FLOOR_MS = {"
          + ", ".join(f"{r}: ({f}, {pm})" for r, (f, pm) in row.items())
          + "}")


if __name__ == "__main__":
    main()
