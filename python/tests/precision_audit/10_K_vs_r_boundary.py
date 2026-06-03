"""K-vs-r boundary determination — refined post-Issue-1.

With Issue 1 closed (8σ window in `_orbit_inner_rel`) and at σ/P safely
below the rel_per definitional threshold, sweep K ∈ {r, r+1, r+2, r+3}
at r ∈ {2, 3, 4, 5}, all four modes, 30 seeds each.

Goal: pin down the smallest safe K-r gap. The dispatcher currently uses
`K_a >= r_a + 2` in `_orbit_safe_for_precision`; this script tests
whether `K_a >= r_a + 1` would suffice, and how steep the precision
cliff at K = r is.

σ/P = 0.025 (below 0.03 rel_per threshold, no Issue 4 sharp-Gaussian
amplification).
"""
import numpy as np
import warnings
from mpt.tensor import (
    build_exp_tens, _cos_sim_exp_tens_ma_orbit, _cos_sim_exp_tens_ma_pairwise,
)


MODES = [
    ("abs_nonper", False, False),
    ("abs_per",    False, True),
    ("rel_nonper", True,  False),
    ("rel_per",    True,  True),
]
P = 1200.0
SIGMA = 0.025 * P     # σ = 30, σ/P = 0.025
R_VALUES = [2, 3, 4, 5]
DELTA_K_VALUES = [0, 1, 2, 3]      # K - r
N = 2
A = 1
SEEDS = 30


def err(p1, w1, p2, w2, sigma, P, r, is_rel, is_per):
    sigmas = [sigma]
    rs = [r]
    groups = [0]
    is_rels = [is_rel]
    is_pers = [is_per]
    periods = [P]
    d1 = build_exp_tens(
        [p1], [w1], sigmas, rs, is_rels, is_pers, periods, verbose=False,
    )
    d2 = build_exp_tens(
        [p2], [w2], sigmas, rs, is_rels, is_pers, periods, verbose=False,
    )
    o_xy, o_xx, o_yy = _cos_sim_exp_tens_ma_orbit(d1, d2)
    p_xy, p_xx, p_yy = _cos_sim_exp_tens_ma_pairwise(d1, d2, verbose=False)
    co = o_xy / np.sqrt(o_xx * o_yy)
    cp = p_xy / np.sqrt(p_xx * p_yy)
    return abs(co - cp) / max(abs(cp), 1e-300)


def main():
    print(
        f"K-vs-r boundary — {SEEDS} seeds at A={A}, N={N}, σ={SIGMA}, "
        f"P={P} (σ/P={SIGMA/P:.3f}).\n"
        f"Probe: how big a K-r margin is needed for FP precision?\n"
    )
    header = (
        f"{'mode':<11} {'r':>2} {'K':>2} {'K-r':>3} | "
        f"{'med':>10} {'p90':>10} {'p95':>10} {'max':>10}"
    )
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
                        errs.append(err(
                            p1, w1, p2, w2, SIGMA, P, r, is_rel, is_per,
                        ))
                errs = np.array(errs)
                flag = "    "
                if errs.max() > 1e-4:
                    flag = "BAD!"
                elif errs.max() > 1e-9:
                    flag = "BAD "
                elif errs.max() > 1e-12:
                    flag = "WARN"
                print(
                    f"{mode_name:<11} {r:>2} {K:>2} {dk:>3} | "
                    f"{np.median(errs):>10.2e} "
                    f"{np.percentile(errs, 90):>10.2e} "
                    f"{np.percentile(errs, 95):>10.2e} "
                    f"{errs.max():>10.2e} {flag}"
                )
            print()


if __name__ == "__main__":
    main()
