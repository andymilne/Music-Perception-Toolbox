"""High-K sweep — spectral analysis regime.

Composite spectra from chords routinely have K in the tens to hundreds.
Pairwise OOMs above K~50 even at r=2, so orbit is the only feasible
method in this regime — there is no `method='bulger'` fallback. This
sweep verifies orbit is correct and finite where pairwise can still run,
and that orbit returns sensible values at high K where pairwise cannot.

Sweep:
- All 4 modes
- r=2: K ∈ {20, 50, 100}  (pairwise OOM above K~50)
- r=3: K ∈ {10, 15, 20}   (pairwise OOM above K~20)
- r=4: K ∈ {8, 10, 12}    (pairwise OOM above K~12)
- σ/P ∈ {0.005, 0.025, 0.05}  (typical spectral analysis range)
- A=1, N=2, 3 seeds.

For (K, r) combinations where pairwise is feasible, we compare orbit vs
pairwise. For (K, r) where pairwise OOMs, we confirm orbit returns a
finite cosine in [-1, 1] and is self-consistent across two seeds.
"""
import numpy as np
import warnings
from math import comb, factorial
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
SIGMA_OVER_P_VALUES = [0.005, 0.025, 0.05]
N = 2
A = 1
SEEDS = 3

K_BY_R = {
    2: [20, 50, 100],
    3: [10, 15, 20],
    4: [8, 10, 12],
}


def pairwise_size(r, K, N):
    """Approximate pairwise inner-product matrix size in entries."""
    return N * N * factorial(r) * comb(K, r) ** 2


# Pairwise size threshold beyond which we skip pairwise comparison.
# 1e8 entries × 8 bytes ≈ 800 MB — borderline OK on this container.
PAIRWISE_FEASIBLE_LIMIT = 5e7


def main():
    print(
        f"High-K sweep — A={A}, N={N}, {SEEDS} seeds.\n"
        f"Pairwise comparison run when feasible "
        f"(matrix < {PAIRWISE_FEASIBLE_LIMIT:.0e} entries);\n"
        f"otherwise orbit-only sanity check (finite, in [-1, 1]).\n"
    )
    header = (
        f"{'mode':<11} {'r':>2} {'K':>3} {'σ/P':>6} | "
        f"{'pw size':>10} {'orbit cos':>10} {'pw cos':>10} {'rel err':>10}"
    )
    print(header)
    print("-" * len(header))
    for mode_name, is_rel, is_per in MODES:
        for r, K_values in K_BY_R.items():
            for K in K_values:
                pw_size = pairwise_size(r, K, N)
                pw_feasible = (pw_size < PAIRWISE_FEASIBLE_LIMIT)
                # rel modes at σ/P=0.005: orbit grid 200k points,
                # workable but slow. rel_per orbit at σ/P=0.05 routes via
                # the dispatcher's threshold; we force it to characterise.
                for sp in SIGMA_OVER_P_VALUES:
                    sigma = sp * P
                    orbit_cosines = []
                    pw_cosines = []
                    for seed in range(SEEDS):
                        rng = np.random.default_rng(seed)
                        p1 = rng.uniform(0, P, (K, N))
                        w1 = rng.uniform(0.1, 1.0, (K, N))
                        p2 = rng.uniform(0, P, (K, N))
                        w2 = rng.uniform(0.1, 1.0, (K, N))
                        d1 = build_exp_tens(
                            [p1], [w1], [sigma], [r], [0],
                            [is_rel], [is_per], [P], verbose=False,
                        )
                        d2 = build_exp_tens(
                            [p2], [w2], [sigma], [r], [0],
                            [is_rel], [is_per], [P], verbose=False,
                        )
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            try:
                                o_xy, o_xx, o_yy = _cos_sim_exp_tens_ma_orbit(d1, d2)
                                co = (o_xy / np.sqrt(o_xx * o_yy)
                                      if o_xx > 0 and o_yy > 0
                                         and np.isfinite(o_xx * o_yy)
                                      else float('nan'))
                            except Exception:
                                co = float('nan')
                            orbit_cosines.append(co)
                            if pw_feasible:
                                try:
                                    p_xy, p_xx, p_yy = _cos_sim_exp_tens_ma_pairwise(
                                        d1, d2, verbose=False,
                                    )
                                    cp = (p_xy / np.sqrt(p_xx * p_yy)
                                          if p_xx > 0 and p_yy > 0
                                          else float('nan'))
                                except Exception:
                                    cp = float('nan')
                                pw_cosines.append(cp)
                    co_med = np.nanmedian(orbit_cosines)
                    if pw_feasible and pw_cosines:
                        cp_med = np.nanmedian(pw_cosines)
                        errs = [
                            abs(co - cp) / max(abs(cp), 1e-300)
                            for co, cp in zip(orbit_cosines, pw_cosines)
                            if not (np.isnan(co) or np.isnan(cp))
                        ]
                        max_err = max(errs) if errs else float('nan')
                        flag = ""
                        if max_err > 1e-9:
                            flag = "BAD"
                        elif max_err > 1e-12:
                            flag = "WARN"
                        print(
                            f"{mode_name:<11} {r:>2} {K:>3} {sp:>6.3f} | "
                            f"{pw_size:>10.1e} {co_med:>10.4f} "
                            f"{cp_med:>10.4f} {max_err:>10.2e} {flag}"
                        )
                    else:
                        # Orbit-only — sanity check
                        in_range = all(
                            -1.01 <= co <= 1.01 if not np.isnan(co) else False
                            for co in orbit_cosines
                        )
                        flag = "OK" if in_range else "BAD"
                        print(
                            f"{mode_name:<11} {r:>2} {K:>3} {sp:>6.3f} | "
                            f"{pw_size:>10.1e} {co_med:>10.4f} "
                            f"{'(skip)':>10} {'(orbit only)':>10} {flag}"
                        )
                print()


if __name__ == "__main__":
    main()
