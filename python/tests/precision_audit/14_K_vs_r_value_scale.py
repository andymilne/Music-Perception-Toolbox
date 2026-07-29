"""K-vs-r boundary, re-measured as absolute error on the value scale.

Script 10 reported ``|c_orbit - c_enum| / |c_enum|``. The quantity is a
cosine similarity, bounded by 1 in magnitude, so its absolute error can
never exceed 2; every figure script 10 reports above that came from the
denominator alone. This script reports ``|c_orbit - c_enum|`` and judges
it against ``truncation_floor(truncationSigmas)``, the largest normalised
kernel value truncation discards.

Configurations are those of script 10: r in {2..5}, K - r in {0..3}, all
four modes, sigma/P = 0.025, N = 2, A = 1, 30 seeds.
"""
import warnings

import numpy as np

from mpt._defaults import truncation_floor
from mpt.tensor import (
    build_exp_tens, _cos_sim_exp_tens_ma_orbit, _cos_sim_exp_tens_ma_pairwise,
)

MODES = [
    ("abs_nonper", False, False),
    ("abs_per", False, True),
    ("rel_nonper", True, False),
    ("rel_per", True, True),
]
P = 1200.0
SIGMA = 0.025 * P
R_VALUES = [2, 3, 4, 5]
DELTA_K_VALUES = [0, 1, 2, 3]
N = 2
SEEDS = 30
FLOOR = truncation_floor(6)


def both_cosines(p1, w1, p2, w2, r, is_rel, is_per):
    """Return the orbit and enumerated cosine similarities."""
    geom = ([SIGMA], [r], [is_rel], [is_per], [P])
    d1 = build_exp_tens([p1], [w1], *geom, verbose=False)
    d2 = build_exp_tens([p2], [w2], *geom, verbose=False)
    o_xy, o_xx, o_yy = _cos_sim_exp_tens_ma_orbit(d1, d2)
    p_xy, p_xx, p_yy = _cos_sim_exp_tens_ma_pairwise(d1, d2, verbose=False)
    return (o_xy / np.sqrt(o_xx * o_yy), p_xy / np.sqrt(p_xx * p_yy))


def main():
    print(f"K-vs-r boundary on the value scale — {SEEDS} seeds, A=1, N={N}, "
          f"sigma={SIGMA}, P={P} (sigma/P={SIGMA / P:.3f}).")
    print(f"Reported: |c_orbit - c_enum|, against truncation_floor(6) = "
          f"{FLOOR:.3e}.\n")
    header = (f"{'mode':<11} {'r':>2} {'K':>2} {'K-r':>3} | {'med':>10} "
              f"{'p90':>10} {'max':>10} {'n>floor':>8}")
    print(header)
    print("-" * len(header))
    for mode_name, is_rel, is_per in MODES:
        for r in R_VALUES:
            for dk in DELTA_K_VALUES:
                K = r + dk
                errs = []
                for seed in range(SEEDS):
                    rng = np.random.default_rng(seed)
                    p1 = rng.uniform(0, P, (K, N))
                    w1 = rng.uniform(0.1, 1.0, (K, N))
                    p2 = rng.uniform(0, P, (K, N))
                    w2 = rng.uniform(0.1, 1.0, (K, N))
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        co, cp = both_cosines(
                            p1, w1, p2, w2, r, is_rel, is_per)
                    errs.append(abs(co - cp))
                errs = np.array(errs)
                n_over = int((errs > FLOOR).sum())
                flag = "BREACH" if n_over else ""
                print(f"{mode_name:<11} {r:>2} {K:>2} {dk:>3} | "
                      f"{np.median(errs):>10.2e} "
                      f"{np.percentile(errs, 90):>10.2e} "
                      f"{errs.max():>10.2e} {n_over:>8} {flag}")
            print()


if __name__ == "__main__":
    main()
