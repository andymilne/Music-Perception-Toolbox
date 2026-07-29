"""Calibrate the orbit route's cancellation error against the truncation budget.

The Moebius (orbit) reduction computes a sum of signed terms that largely
cancel, so its answer carries fewer significant digits than the terms it was
built from. Enumeration sums only non-negative terms and loses nothing. The
dispatcher must therefore know when orbit's cancellation error would exceed
the accuracy the caller has asked for.

Error model. For a sum whose terms reach magnitude ``max|term|`` and whose
total is ``|S|``, floating-point cancellation leaves an absolute error of
order ``eps * max|term|``, so the error relative to the result is

    err  ~  eps * max|term| / max|value|

normalised to the largest value in the block, because a node whose sum lands
on a true zero has a large relative error and no consequence. The orbit
routines report ``max|term|`` directly as the term mass.

Sweep. Tuple size ``r``, the margin ``K - r``, the spread of the values
relative to sigma (clustered values cancel worse than dispersed ones), and
the weight profile (uniform, decaying, harmonic-like, one dominant). Value
draws are batched along the leading axis the orbit routines already accept,
so each cell is one call rather than one call per draw.

Output. Worst-case relative error per (r, K - r), and the largest r that
stays inside each truncation budget:

    truncationSigmas = 6    ->  ~1.5e-8   (the shipped default)
    truncationSigmas = Inf  ->  ~1e-12    (the accuracy floor)

Run from the python directory:
    PYTHONPATH=. python3 tools/calibrate_orbit_cancellation.py
"""
from __future__ import annotations

import math
import warnings

import numpy as np

import mpt._tensor._nested_contraction as nc
from mpt._mobius import inner_product_orbit_grid as orbit_grid
from mpt._tensor.dispatch import _ORBIT_R_MAX_SHIPPED as R_MAX

EPS = float(np.finfo(float).eps)
SIGMA = 0.30
N_DRAW = 12
TOL_DEFAULT = 1.5e-8    # truncationSigmas = 6, the shipped default
TOL_FLOOR = 1e-12       # truncationSigmas = Inf, the accuracy floor

# Enumerating both sides costs C(K, r) * r! kernel products per draw, so
# cap it rather than let a large cell run for minutes.
GOLDEN_MAX_TUPLES = 2_000_000


def _weight_profiles(K):
    """Weight vectors spanning flat to steeply peaked."""
    idx = np.arange(K, dtype=float)
    return {
        "uniform": np.ones(K),
        "linear decay": 1.0 - 0.9 * idx / max(K - 1, 1),
        "harmonic": 1.0 / (idx + 1.0),
        "one dominant": np.concatenate(([1.0], np.full(K - 1, 1e-3))),
    }


def _blocks(rng, r, K, w, spread):
    """Weighted Gaussian kernel blocks, shape (N_DRAW, K, K)."""
    if spread == 0.0:
        V = np.repeat(rng.uniform(0.0, 12.0, size=(N_DRAW, 1)), K, axis=1)
    else:
        V = np.sort(rng.uniform(0.0, spread, size=(N_DRAW, K)), axis=1)
    d = V[:, :, None] - V[:, None, :]
    return np.exp(-(d ** 2) / (4.0 * SIGMA ** 2)) * np.outer(w, w)[None, :, :]


def _golden_feasible(K, r):
    xt, yt = nc._tuple_sides(K, r, True)
    return xt.shape[0] * yt.shape[0] <= GOLDEN_MAX_TUPLES


def _errors(rng, r, K, w, spread):
    """(true error, proxy) on the value scale.

    Enumeration sums only non-negative terms, so it carries no
    cancellation and serves as the reference. The proxy is what the orbit
    call already reports -- eps times the worst term, normalised the same
    way -- and is what a runtime guard could use where enumeration is out
    of reach.
    """
    M = _blocks(rng, r, K, w, spread)
    empty = np.empty((0, r), dtype=np.intp)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        orb = nc._combine_orbit(M, r, empty, empty)
    _, _, mass = orbit_grid(M, np.ones(K), np.ones(K), r, prefactor=1.0,
                            return_cancellation_ratio=True,
                            return_term_mass=True)
    if _golden_feasible(K, r):
        xt, yt = nc._tuple_sides(K, r, True)
        gold = nc._combine(M, xt, yt)
        scale = float(np.abs(gold).max())
        true = float(np.abs(orb - gold).max()) / scale if scale > 0 else 0.0
    else:
        scale = float(np.abs(orb).max())
        true = float("nan")
    proxy = EPS * float(np.atleast_1d(mass).max()) / scale if scale > 0 else 0.0
    return true, proxy


def main():
    rng = np.random.default_rng(20260728)
    spreads = [0.0, 0.3, 1.0, 3.0, 12.0, 48.0]
    names = list(_weight_profiles(4).keys())

    print()
    print("Orbit cancellation error against enumeration (the golden value)")
    print(f"sigma = {SIGMA}, {N_DRAW} draws per cell, worst case over "
          "margins K-r = 1..3 and all spreads")
    print("'proxy' is eps * worst-term / scale, the quantity a runtime guard")
    print("could use; 'nan' means enumeration was too large to reference.")
    print()

    table = {}
    for wname in names:
        print(f"  {wname}")
        print(f"    {'r':>3} {'true error':>13} {'proxy':>13} {'proxy/true':>12}")
        for r in range(2, R_MAX + 1):
            wt = wp = 0.0
            have_golden = False
            for d in (1, 2, 3):
                K = r + d
                w = _weight_profiles(K)[wname]
                for spread in spreads:
                    t, p = _errors(rng, r, K, w, spread)
                    if not math.isnan(t):
                        wt = max(wt, t)
                        have_golden = True
                    wp = max(wp, p)
            table[(r, wname)] = (wt if have_golden else float("nan"), wp)
            ratio = wp / wt if (have_golden and wt > 0) else float("nan")
            tt = "unreferenced" if not have_golden else f"{wt:.1e}"
            print(f"    {r:>3} {tt:>13} {wp:>13.1e} {ratio:>12.0f}")
        print()

    print("Largest r inside each budget, by true error:")
    for label, tol in (("truncationSigmas = 6   (~1.5e-8)", TOL_DEFAULT),
                       ("truncationSigmas = Inf (~1e-12)", TOL_FLOOR)):
        print(f"  {label}")
        for wname in names:
            top = None
            for r in range(2, R_MAX + 1):
                tr = table[(r, wname)][0]
                if math.isnan(tr):
                    break            # no reference above here
                if tr >= tol:
                    break            # breached: the safe run ends
                top = r
            if top is None:
                verdict = "none"
            else:
                nxt = table[(top + 1, wname)][0] if top + 1 <= R_MAX else 0.0
                tail = " (above unreferenced)" if math.isnan(nxt) else ""
                verdict = f"r <= {top}{tail}"
            print(f"      {wname:>14}: {verdict}")
    print()


if __name__ == "__main__":
    main()
