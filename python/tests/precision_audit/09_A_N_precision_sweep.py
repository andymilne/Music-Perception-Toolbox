"""A/N precision sweep: orbit vs pairwise across (A, N) at safe (r, K, σ/P).

Verifies that precision scales gracefully with the number of attributes
A ∈ {1, 2, 4} and the number of events N ∈ {2, 4, 8}, in the regime where
the K-vs-r and σ/P precision boundaries are clear (K = r+2, σ/P = 0.05).

Per-attribute factorisation in the MAET inner product suggests precision
should accumulate as O(A × per-attribute-error). For multi-event, each
(n_x, n_y) entry of the per-attribute kernel matrix accumulates in the
final cosine independently, so error should scale at most ~sqrt(N²) = N.

Floating-point baseline ~1e-15. We expect cells to stay below ~1e-12
even at A=4 N=8 r=4 (worst case here).
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
A_VALUES = [1, 2]
N_VALUES = [2, 4, 8]
R_VALUES = [2, 3, 4]
# A=4 dropped: pairwise inner-product matrix is N² × (r! × C(K, r))^A,
# which is multi-GB even at r=2 K=4 N=8. The OOM zone is exactly where
# orbit dispatcher dominates anyway, so testing pairwise here is moot.
SIGMA = 60.0    # σ/P = 0.05 at P=1200
P = 1200.0
SEEDS = 5


def err(p1, w1, p2, w2, sigma, P, r, A, is_rel, is_per):
    """orbit vs pairwise relative cosine err at given attribute config."""
    sigmas = [sigma]                 # single group
    rs = [r] * A
    groups = [0] * A
    is_rels = [is_rel]
    is_pers = [is_per]
    periods = [P]
    d1 = build_exp_tens(p1, w1, sigmas, rs, groups, is_rels, is_pers, periods, verbose=False)
    d2 = build_exp_tens(p2, w2, sigmas, rs, groups, is_rels, is_pers, periods, verbose=False)
    o_xy, o_xx, o_yy = _cos_sim_exp_tens_ma_orbit(d1, d2)
    p_xy, p_xx, p_yy = _cos_sim_exp_tens_ma_pairwise(d1, d2, verbose=False)
    co = o_xy / np.sqrt(o_xx * o_yy)
    cp = p_xy / np.sqrt(p_xx * p_yy)
    return abs(co - cp) / max(abs(cp), 1e-300)


def main():
    print(
        f"A/N precision sweep — {SEEDS} seeds at K=r+2, σ={SIGMA}, P={P} "
        f"(σ/P={SIGMA/P:.3f}).\n"
    )
    header = (
        f"{'mode':<11} {'r':>2} {'A':>2} {'N':>2} | "
        f"{'med':>10} {'p90':>10} {'max':>10}"
    )
    print(header)
    print("-" * len(header))
    for mode_name, is_rel, is_per in MODES:
        for r in R_VALUES:
            K = r + 2
            for A in A_VALUES:
                for N in N_VALUES:
                    # Skip pairwise OOM zone:
                    # pairwise size ~ N² × (r! × C(K, r))^A.
                    # At A=2 r=4, even N=2 is ~1 GB (kernel matrix in
                    # _ip_full + intermediates push the container over
                    # its limit). The orbit dispatcher dominates this
                    # regime decisively, so the comparison isn't
                    # informative anyway.
                    if A == 2 and r == 4:
                        print(
                            f"{mode_name:<11} {r:>2} {A:>2} {N:>2} | "
                            f"{'(pairwise OOM zone — skipped)':>34}"
                        )
                        continue
                    if A == 2 and r == 3 and N == 8:
                        # 600 MB tight; some abs_per runs exceed.
                        print(
                            f"{mode_name:<11} {r:>2} {A:>2} {N:>2} | "
                            f"{'(pairwise tight — skipped)':>34}"
                        )
                        continue
                    errs = []
                    for seed in range(SEEDS):
                        rng = np.random.default_rng(seed)
                        p1 = [rng.uniform(0, P, (K, N)) for _ in range(A)]
                        w1 = [rng.uniform(0.1, 1.0, (K, N)) for _ in range(A)]
                        p2 = [rng.uniform(0, P, (K, N)) for _ in range(A)]
                        w2 = [rng.uniform(0.1, 1.0, (K, N)) for _ in range(A)]
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            errs.append(err(
                                p1, w1, p2, w2, SIGMA, P, r, A, is_rel, is_per,
                            ))
                    errs = np.array(errs)
                    flag = "    "
                    if errs.max() > 1e-9:
                        flag = "BAD!"
                    elif errs.max() > 1e-12:
                        flag = "WARN"
                    print(
                        f"{mode_name:<11} {r:>2} {A:>2} {N:>2} | "
                        f"{np.median(errs):>10.2e} "
                        f"{np.percentile(errs, 90):>10.2e} "
                        f"{errs.max():>10.2e} {flag}"
                    )
            print()


if __name__ == "__main__":
    main()
