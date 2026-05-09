"""σ→0 sweep at low K — music-theoretical regime characterisation.

In music-theoretical analyses, σ→0 (or σ very small) gives delta-like
kernels and exact-match semantics. This is a legitimate use case, not
an edge case. Pairwise handles σ→0 cleanly (kernel entries become
{0, 1}, inner product becomes a count of shared tuples, cosine reflects
proportion of overlap).

Orbit's behaviour at σ→0 is the question. The Möbius alternating sum
involves products of kernel entries; when those entries become tiny
the alternating sum can suffer catastrophic cancellation or even
overflow.

Sweep: σ/P ∈ {1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7}, P=1200,
all four modes, r ∈ {2, 3, 4, 5}, K ∈ {r+1, r+2, r+4}, A=1, N=2,
5 seeds. P=12000 not needed — earlier sweeps confirmed σ/P is the
controlling parameter.

To make the cosine non-trivial, we share half the K slots between the
two events (so the cosine sits in (0, 1) rather than 0 or 1). This
makes the comparison meaningful even at extreme σ.
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
SIGMA_OVER_P_VALUES = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
P = 1200.0
R_VALUES = [2, 3, 4, 5]
DELTA_K_VALUES = [1, 2, 4]
N = 2
A = 1
SEEDS = 5


def err_and_cosine(p1, w1, p2, w2, sigma, P, r, is_rel, is_per):
    """Returns (rel_err, pairwise_cosine, orbit_cosine_or_nan)."""
    sigmas = [sigma]
    rs = [r]
    groups = [0]
    is_rels = [is_rel]
    is_pers = [is_per]
    periods = [P]
    d1 = build_exp_tens(
        [p1], [w1], sigmas, rs, groups, is_rels, is_pers, periods, verbose=False,
    )
    d2 = build_exp_tens(
        [p2], [w2], sigmas, rs, groups, is_rels, is_pers, periods, verbose=False,
    )
    o_xy, o_xx, o_yy = _cos_sim_exp_tens_ma_orbit(d1, d2)
    p_xy, p_xx, p_yy = _cos_sim_exp_tens_ma_pairwise(d1, d2, verbose=False)
    co = (o_xy / np.sqrt(o_xx * o_yy)
          if o_xx > 0 and o_yy > 0 and np.isfinite(o_xx * o_yy)
          else float('nan'))
    cp = (p_xy / np.sqrt(p_xx * p_yy)
          if p_xx > 0 and p_yy > 0
          else float('nan'))
    if np.isnan(cp) or np.isnan(co):
        return float('nan'), cp, co
    return abs(co - cp) / max(abs(cp), 1e-300), cp, co


def main():
    print(
        f"σ→0 sweep — {SEEDS} seeds at A={A}, N={N}, P={P}.\n"
        f"Half of K slots shared between p1 and p2 to give a non-trivial "
        f"cosine.\n"
        f"Note: rel-mode orbit uses trapezoidal grid of period/σ × sps "
        f"points,\n"
        f"infeasible below σ/P ≈ 1e-3 (grid > 100k points). Cells skipped.\n"
    )
    header = (
        f"{'mode':<11} {'r':>2} {'K':>3} {'σ/P':>8} | "
        f"{'med err':>10} {'max err':>10} {'med cos':>9}"
    )
    print(header)
    print("-" * len(header))
    for mode_name, is_rel, is_per in MODES:
        for r in R_VALUES:
            for dk in DELTA_K_VALUES:
                K = r + dk
                for sp in SIGMA_OVER_P_VALUES:
                    sigma = sp * P
                    # Skip rel modes at extreme σ — orbit grid infeasible
                    if is_rel and sp < 1e-3:
                        print(
                            f"{mode_name:<11} {r:>2} {K:>3} {sp:>8.0e} | "
                            f"{'(orbit grid infeasible — skipped)':>40}"
                        )
                        continue
                    errs = []
                    cosines = []
                    for seed in range(SEEDS):
                        rng = np.random.default_rng(seed)
                        # Half of K slots shared, half random — so cosine
                        # is non-degenerate at any σ.
                        K_shared = K // 2
                        shared_p = rng.uniform(0, P, (K_shared, N))
                        p1 = np.vstack([
                            shared_p,
                            rng.uniform(0, P, (K - K_shared, N)),
                        ])
                        p2 = np.vstack([
                            shared_p,
                            rng.uniform(0, P, (K - K_shared, N)),
                        ])
                        w1 = rng.uniform(0.1, 1.0, (K, N))
                        w2 = rng.uniform(0.1, 1.0, (K, N))
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            try:
                                e, cp, co = err_and_cosine(
                                    p1, w1, p2, w2, sigma, P, r,
                                    is_rel, is_per,
                                )
                            except Exception:
                                e, cp, co = float('nan'), float('nan'), float('nan')
                        errs.append(e)
                        cosines.append(cp)
                    errs = np.array(errs)
                    cosines = np.array(cosines)
                    if np.isnan(errs).any():
                        flag = "NAN!"
                    elif errs.max() > 1.0:
                        flag = "BLOW"   # orbit overflowed / huge err
                    elif errs.max() > 1e-4:
                        flag = "BAD!"
                    elif errs.max() > 1e-9:
                        flag = "BAD "
                    elif errs.max() > 1e-12:
                        flag = "WARN"
                    else:
                        flag = "    "
                    print(
                        f"{mode_name:<11} {r:>2} {K:>3} {sp:>8.0e} | "
                        f"{np.nanmedian(errs):>10.2e} "
                        f"{np.nanmax(errs):>10.2e} "
                        f"{np.nanmedian(cosines):>9.3f} {flag}"
                    )
                print()


if __name__ == "__main__":
    main()
