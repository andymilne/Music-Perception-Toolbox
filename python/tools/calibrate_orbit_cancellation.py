"""Calibrate the Moebius route's accuracy estimate against enumeration.

The Moebius reduction sums signed terms that largely cancel, so it
carries fewer digits than the terms it was built from. Enumeration sums
only non-negative terms and loses nothing, so it is the reference. The
guard in ``_combine_pair`` decides whether the Moebius route is accurate
enough by comparing an estimate of that rounding error against
``truncation_floor(truncation_sigmas)``, the accuracy the caller asked
for.

WHAT IS MEASURED
----------------
Everything here is an absolute error on the value scale --- the scale
``truncation_sigmas`` is stated on --- and never a ratio. A ratio has a
denominator that can legitimately approach zero: a node whose true value
is zero scores an enormous relative error while being exactly right in
absolute terms. The previous version of this script normalised by the
largest value in the block for that reason, and that normalisation is
what made its output unusable; it is also why the guard it was
calibrating was replaced.

Three quantities per cell:

  true      max |Moebius - enumeration|, the error that actually occurs.
  estimate  what the shipped guard uses, ``eps * sum|term| / r!``,
            returned by ``_combine_orbit(..., return_bound=True)``.
  ratio     estimate / true, the conservatism.

TWO QUESTIONS
-------------
Safety: does the guard ever *admit* a route whose true error exceeds
the floor? That is the decision the guard makes, and it is the thing
that would be a defect. It is not the same as the estimate falling
below the true error: an estimate can understate an error that is
itself far inside the budget, which changes no decision. Testing the
raw comparison rather than the decision reports failures that are not
failures. Both are shown, but only the second is a defect.

Cost: how far above the true error does the estimate sit? Too
conservative and the guard refuses a route that was fine, which costs
speed but not correctness. The estimate is a heuristic, not a guaranteed
limit --- sequential summation of n terms admits
``(n-1) * eps * sum|term|`` in the worst case, and this sits below that
--- so its conservatism is the evidence for it, and this script is that
evidence.

The weight profile dominates: uniform weights are safe across the
shipped range, while a steeply peaked profile breaches far earlier. A
sweep that omits the peaked profiles reports a clean result that
ordinary music does not obey.

Run from the python directory::

    PYTHONPATH=. python3 tools/calibrate_orbit_cancellation.py
"""
from __future__ import annotations

import math
import warnings

import numpy as np

import mpt._tensor._nested_contraction as nc
from mpt._defaults import truncation_floor
from mpt._tensor.dispatch import _ORBIT_R_MAX_SHIPPED as R_MAX

SIGMA = 0.30
N_DRAW = 12
SPREADS = (0.0, 0.3, 1.0, 3.0, 12.0, 48.0)
MARGINS = (1, 2, 3)

# Enumerating both sides costs C(K, r) * r! kernel products per draw, so
# cap it rather than let a large cell run for minutes. Cells beyond it
# have no reference and are reported as such rather than scored.
GOLDEN_MAX_TUPLES = 2_000_000

FLOORS = (("truncation_sigmas = 6", 6.0),
          ("truncation_sigmas = 8", 8.0),
          ("truncation_sigmas = inf", math.inf))


def _weight_profiles(K):
    """Weight vectors spanning flat to steeply peaked."""
    idx = np.arange(K, dtype=float)
    return {
        "uniform": np.ones(K),
        "linear decay": 1.0 - 0.9 * idx / max(K - 1, 1),
        "harmonic": 1.0 / (idx + 1.0),
        "one dominant": np.concatenate(([1.0], np.full(K - 1, 1e-3))),
    }


def _blocks(rng, K, w, spread):
    """Weighted Gaussian kernel blocks, shape (N_DRAW, K, K)."""
    if spread == 0.0:
        v = np.repeat(rng.uniform(0.0, 12.0, size=(N_DRAW, 1)), K, axis=1)
    else:
        v = np.sort(rng.uniform(0.0, spread, size=(N_DRAW, K)), axis=1)
    d = v[:, :, None] - v[:, None, :]
    return np.exp(-(d ** 2) / (4.0 * SIGMA ** 2)) * np.outer(w, w)[None, :, :]


def _golden_feasible(K, r):
    xt, yt = nc._tuple_sides(K, r, True)
    return xt.shape[0] * yt.shape[0] <= GOLDEN_MAX_TUPLES


def _cell(rng, r, K, w, spread):
    """(true error, estimate) as absolute quantities on the value scale."""
    m = _blocks(rng, K, w, spread)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        orb, estimate = nc._combine_orbit(m, r, return_bound=True)
    if not _golden_feasible(K, r):
        return float("nan"), float(estimate)
    xt, yt = nc._tuple_sides(K, r, True)
    gold = nc._combine(m, xt, yt)
    return float(np.abs(orb - gold).max()), float(estimate)


def main():
    rng = np.random.default_rng(20260728)
    profiles = list(_weight_profiles(4).keys())
    print()
    print("Moebius rounding error against enumeration, absolute, on the "
          "value scale")
    print(f"sigma = {SIGMA}, {N_DRAW} draws per cell, worst case over "
          f"margins K-r = {MARGINS} and spreads {SPREADS}")
    print("estimate is the shipped guard's eps * sum|term| / r!")
    print()

    table = {}
    under = []          # estimate below the true error, decision aside
    admitted = []       # admitted although the true error breaches: a defect
    for name in profiles:
        print(f"  {name}")
        print(f"    {'r':>3} {'true':>12} {'estimate':>12} "
              f"{'estimate/true':>14}")
        for r in range(2, R_MAX + 1):
            worst_true = 0.0
            worst_est = 0.0
            referenced = False
            for margin in MARGINS:
                K = r + margin
                w = _weight_profiles(K)[name]
                for spread in SPREADS:
                    t, e = _cell(rng, r, K, w, spread)
                    worst_est = max(worst_est, e)
                    if not math.isnan(t):
                        referenced = True
                        worst_true = max(worst_true, t)
                        if e < t:
                            under.append((name, r, K, spread, t, e))
                        for _, ts_val in FLOORS:
                            fl = truncation_floor(ts_val)
                            if e <= fl < t:
                                admitted.append((name, r, K, spread,
                                                 ts_val, t, e, fl))
            table[(r, name)] = (worst_true if referenced else float("nan"),
                                worst_est)
            shown = f"{worst_true:.2e}" if referenced else "unreferenced"
            ratio = (f"{worst_est / worst_true:.0f}x"
                     if referenced and worst_true > 0 else "--")
            print(f"    {r:>3} {shown:>12} {worst_est:>12.2e} {ratio:>14}")
        print()

    print("Defect test --- the guard admitted a route whose true error "
          "breaches the floor:")
    if not admitted:
        print("    none at any of the floors swept")
    else:
        for name, r, K, spread, ts_val, t, e, fl in admitted:
            print(f"    {name}, r={r}, K={K}, spread={spread}, ts={ts_val}: "
                  f"true {t:.2e} > floor {fl:.2e} >= estimate {e:.2e}")
    print()
    print(f"Cells where the estimate understated the true error: "
          f"{len(under)}")
    if under:
        worst = max(under, key=lambda c: c[4] / c[5])
        print(f"    worst by {worst[4] / worst[5]:.1f}x at {worst[0]}, "
              f"r={worst[1]}, K={worst[2]}, spread={worst[3]} "
              f"(true {worst[4]:.2e}, estimate {worst[5]:.2e})")
        print("    None of these changes a decision unless it appears "
              "above as well.")
    print()

    print("Largest r whose true error stays inside each accuracy floor:")
    for label, ts in FLOORS:
        floor = truncation_floor(ts)
        print(f"  {label}  (floor {floor:.2e})")
        for name in profiles:
            top = None
            unref = False
            for r in range(2, R_MAX + 1):
                true = table[(r, name)][0]
                if math.isnan(true):
                    unref = True
                    break
                if true >= floor:
                    break
                top = r
            if top is None:
                verdict = "none"
            else:
                verdict = f"r <= {top}" + (" (above unreferenced)"
                                           if unref else "")
            print(f"      {name:>14}: {verdict}")
    print()


if __name__ == "__main__":
    main()
