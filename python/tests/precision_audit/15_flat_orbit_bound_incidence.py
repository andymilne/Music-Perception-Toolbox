"""Incidence of a value-scale bound on the flat multi-attribute orbit route.

The flat path currently refuses the Moebius method on the size margin
``K_a >= r_a + 2``. This script measures what a value-scale guard would
do in its place: it wraps the three orbit entry points, requests the
per-call term mass, and forms the same bound the nested path uses,

    bound = |Omega_r| * eps * max|term|

on the scale the orbit values are returned on, then compares it against
``truncation_floor(truncationSigmas)``.

Reported per configuration: the largest bound seen across all orbit
calls, and whether it exceeds the floor. Also reported is the measured
disagreement between the orbit and enumerated routes, so the bound can
be checked against the error it is supposed to predict.
"""
import warnings

import numpy as np

import mpt._mobius as _mob
from mpt._defaults import truncation_floor
from mpt._tensor._nested_contraction import _n_orbits
from mpt.tensor import (
    build_exp_tens, _cos_sim_exp_tens_ma_orbit, _cos_sim_exp_tens_ma_pairwise,
)

EPS = np.finfo(float).eps
P = 1200.0
N = 2
SEEDS = 8
FLOOR = truncation_floor(6)

_seen = {"bound": 0.0}


def _instrument():
    """Wrap the orbit entry points to record the worst bound seen."""
    for name in ("inner_product_orbit_grid", "inner_product_orbit_pw_batched",
                 "inner_product_orbit_sparse"):
        orig = getattr(_mob, name)

        def wrapper(*a, _orig=orig, **kw):
            r = a[3] if len(a) > 3 else kw.get("r")
            want_ratio = kw.pop("return_cancellation_ratio", False)
            want_mass = kw.pop("return_term_mass", False)
            out = _orig(*a, return_cancellation_ratio=True,
                        return_term_mass=True, **kw)
            vals, ratios, mass = out
            m = float(np.max(np.abs(np.atleast_1d(mass)))) if np.size(mass) \
                else 0.0
            b = _n_orbits(int(r)) * EPS * m
            if b > _seen["bound"]:
                _seen["bound"] = b
            if want_ratio and want_mass:
                return vals, ratios, mass
            if want_ratio:
                return vals, ratios
            return vals

        setattr(_mob, name, wrapper)
        # The flat inner-product module imports these by name at call
        # time from .._mobius, so patching the module attribute suffices.


def config(r, K, is_rel, is_per, w_hi):
    """Worst bound and worst orbit/enumeration disagreement over seeds."""
    bounds, errs = [], []
    for seed in range(SEEDS):
        rng = np.random.default_rng(seed)
        geom = ([0.025 * P], [r], [is_rel], [is_per], [P])
        d1 = build_exp_tens([rng.uniform(0, P, (K, N))],
                            [rng.uniform(0.1, w_hi, (K, N))], *geom,
                            verbose=False)
        d2 = build_exp_tens([rng.uniform(0, P, (K, N))],
                            [rng.uniform(0.1, w_hi, (K, N))], *geom,
                            verbose=False)
        _seen["bound"] = 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            oxy, oxx, oyy = _cos_sim_exp_tens_ma_orbit(d1, d2)
            pxy, pxx, pyy = _cos_sim_exp_tens_ma_pairwise(d1, d2, verbose=False)
        bounds.append(_seen["bound"])
        errs.append(abs(oxy / np.sqrt(oxx * oyy) - pxy / np.sqrt(pxx * pyy)))
    return max(bounds), max(errs)


def main():
    _instrument()
    print(f"Flat orbit bound vs truncation_floor(6) = {FLOOR:.3e}. "
          f"{SEEDS} seeds, A=1, N={N}, sigma/P=0.025.")
    print("'gate' is the shipped size margin: adm = admits Moebius "
          "(K-r>=2), ref = refuses.\n")
    header = (f"{'mode':<11} {'r':>2} {'K':>2} {'w_hi':>5} {'gate':>4} | "
              f"{'bound':>10} {'err':>10} {'bound>floor':>12}")
    print(header)
    print("-" * len(header))
    for name, is_rel, is_per in [("abs_nonper", False, False),
                                 ("rel_nonper", True, False),
                                 ("rel_per", True, True)]:
        for r in (2, 3, 4, 5):
            for dk in (0, 1, 2, 3):
                for w_hi in (1.0, 100.0):
                    K = r + dk
                    b, e = config(r, K, is_rel, is_per, w_hi)
                    gate = "adm" if dk >= 2 else "ref"
                    print(f"{name:<11} {r:>2} {K:>2} {w_hi:>5.0f} {gate:>4} | "
                          f"{b:>10.2e} {e:>10.2e} "
                          f"{'YES' if b > FLOOR else '':>12}")
            print()


if __name__ == "__main__":
    main()
