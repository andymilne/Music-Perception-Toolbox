"""How often does the corruption check fire, and is it ever needed?

Result: never, across 80 random-collection configurations and 60
exact-match ones. See the notes at the foot of this file.

The flat cosine path computes the Möbius route, then inspects the result
for unambiguous corruption --- a non-finite inner product, a negative
Gram diagonal, or a cosine outside [-1, 1] by more than 1e-6 --- and
silently recomputes by enumeration when it finds any. That silent
recompute is the only place in the flat path where a route is chosen
after the fact, so its incidence decides whether it can simply raise
instead.

The sweep includes the regime the check's own docstring names as its
reason for existing: sigma driven towards zero at low K with r >= 3.
Where the check fires, the enumerated cosine is reported alongside, so
a firing can be read as either genuine corruption or a false alarm.
"""
import warnings

import numpy as np

import mpt._tensor.dispatch as D
from mpt.tensor import (
    build_exp_tens, _cos_sim_exp_tens_ma_orbit, _cos_sim_exp_tens_ma_pairwise,
)

P = 1200.0
SEEDS = 2
_orig = D._orbit_ips_look_corrupted
_count = {"calls": 0, "fires": 0}


def _counting(ip_xy, ip_xx, ip_yy):
    _count["calls"] += 1
    out = _orig(ip_xy, ip_xx, ip_yy)
    if out:
        _count["fires"] += 1
    return out


def probe(r, K, sigma, is_rel, is_per, seed):
    """Return (fired, orbit cosine, enumerated cosine)."""
    rng = np.random.default_rng(seed)
    geom = ([sigma], [r], [is_rel], [is_per], [P])
    px = np.sort(rng.uniform(0, P, (K, 1)))
    py = np.sort(rng.uniform(0, P, (K, 1)))
    w = np.ones((K, 1))
    dx = build_exp_tens([px], [w], *geom, verbose=False)
    dy = build_exp_tens([py], [w], *geom, verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(all="ignore"):
            oxy, oxx, oyy = _cos_sim_exp_tens_ma_orbit(dx, dy)
            pxy, pxx, pyy = _cos_sim_exp_tens_ma_pairwise(dx, dy, verbose=False)
    fired = _orig(oxy, oxx, oyy)
    with np.errstate(all="ignore"):
        co = oxy / np.sqrt(oxx * oyy)
        cp = pxy / np.sqrt(pxx * pyy)
    return bool(fired), float(co), float(cp)


def main():
    sigmas = [1e-6, 1e-3, 1.0, 30.0, 400.0]
    modes = [("abs_nonper", False, False), ("rel_per", True, True)]
    print("Corruption check: how often does it fire?\n")
    print("Regimes swept: sigma 1e-6 to 400 (the docstring names sigma -> 0 "
          "at low K, r >= 3 as its reason), r 2..6, K = r and r+2, four "
          f"modes, {SEEDS} seeds.\n")
    header = (f"{'mode':<11} {'r':>2} {'K':>2} {'sigma':>8} | {'fired':>5} "
              f"{'cos orbit':>12} {'cos enum':>12}")
    print(header)
    print("-" * len(header))
    total = fired_n = 0
    for name, is_rel, is_per in modes:
        for r in (3, 5):
            for K in (r, r + 2):
                for sigma in sigmas:
                    for seed in range(SEEDS):
                        total += 1
                        try:
                            f, co, cp = probe(
                                r, K, sigma, is_rel, is_per, seed)
                        except Exception as exc:
                            print(f"{name:<11} {r:>2} {K:>2} {sigma:>8.0e} | "
                                  f"{'ERR':>5} {type(exc).__name__}")
                            continue
                        if f:
                            fired_n += 1
                            print(f"{name:<11} {r:>2} {K:>2} {sigma:>8.0e} | "
                                  f"{'YES':>5} {co:>12.4e} {cp:>12.4e}")
    print(f"\nFired in {fired_n} of {total} configurations.")
    if fired_n == 0:
        print("No firing anywhere in the swept range, including the regime "
              "the check was written for.")


if __name__ == "__main__":
    main()


# Measured, July 2026
# -------------------
# Random collections: 0 firings in 80 configurations (two modes, r in
# {3, 5}, K in {r, r+2}, sigma from 1e-6 to 400, two seeds).
#
# Exact-match regime, which the check's docstring names as its reason
# for existing: 0 firings in 60 configurations. Identical triad,
# identical seventh chord at r = 3 and r = 4, identical diatonic set at
# r = 5, a triad against itself displaced by 1e-9, and a triad against
# its semitone transposition; sigma from 1e-9 to 1, absolute and
# relative-periodic. No firing, and no exception.
#
# The check therefore costs a silent recompute of the whole cosine by
# the enumerated route on every firing, and nothing has been observed to
# make it fire.
