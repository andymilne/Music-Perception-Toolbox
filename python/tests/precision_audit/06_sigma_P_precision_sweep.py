"""σ/P precision sweep: orbit vs pairwise across (σ, P) at A=1, N=2.

Crosses σ/P ∈ {0.01, 0.05, 0.2, 0.5, 1.0} with P ∈ {1200, 12000} for the four
modes, r ∈ {2, 3, 4}, K = r+2 (clear of K~r cancellation but keeps grid
sizes manageable).

Expected behaviour:
  abs_nonper:    FP at all (σ, P) — period unused.
  abs_per:       FP at all σ/P — wrapping is exact for both formulas.
  rel_nonper:    FP at all (σ, P) — period unused.
  rel_per:       FP at σ/P ≲ 0.03; defined gap above (different formulas).
                 Orbit dispatcher routes away from orbit above 0.03; we
                 force orbit here to characterise the gap directly.
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
SIGMA_OVER_P_VALUES = [0.01, 0.05, 0.2, 0.5, 1.0]
PERIODS = [1200.0, 12000.0]
R_VALUES = [2, 3, 4]
N = 2
A = 1
SEEDS = 5


def err_orbit_vs_pairwise(p1, w1, p2, w2, sigma, P, r, is_rel, is_per):
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
        f"σ/P precision sweep — {SEEDS} seeds at A={A}, N={N}.\n"
        f"All cells K = r+2 (clear of K~r cancellation).\n"
    )
    header = f"{'mode':<11} {'r':>2} {'P':>6} {'σ/P':>7} {'σ':>7} | "
    header += f"{'med':>10} {'max':>10}"
    print(header)
    print("-" * len(header))
    for mode_name, is_rel, is_per in MODES:
        for r in R_VALUES:
            K = r + 2
            for P in PERIODS:
                # for non-periodic modes, P is unused but we still run both
                # to confirm nothing scales with P
                for sp in SIGMA_OVER_P_VALUES:
                    sigma = sp * P
                    errs = []
                    for seed in range(SEEDS):
                        rng = np.random.default_rng(seed)
                        p1 = rng.uniform(0, P, (K, N))
                        w1 = rng.uniform(0.1, 1.0, (K, N))
                        p2 = rng.uniform(0, P, (K, N))
                        w2 = rng.uniform(0.1, 1.0, (K, N))
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            errs.append(err_orbit_vs_pairwise(
                                p1, w1, p2, w2, sigma, P, r, is_rel, is_per,
                            ))
                    errs = np.array(errs)
                    flag = "    "
                    if errs.max() > 1e-4:
                        flag = "BAD!"
                    elif errs.max() > 1e-9:
                        flag = "DEFN"  # rel_per high σ/P expected divergence
                    elif errs.max() > 1e-12:
                        flag = "WARN"
                    print(
                        f"{mode_name:<11} {r:>2} {P:>6.0f} {sp:>7.3f} "
                        f"{sigma:>7.1f} | {np.median(errs):>10.2e} "
                        f"{errs.max():>10.2e} {flag}"
                    )
            print()
    print("Legend: BAD = >1e-4, DEFN = expected definitional gap (rel_per σ/P>0.03),")
    print("        WARN = >1e-12 worth investigating, blank = FP precision.")


if __name__ == "__main__":
    main()
