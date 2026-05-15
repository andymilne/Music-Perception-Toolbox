"""σ/P performance sweep: orbit vs pairwise wall time across (σ, P).

The current cost model in `_predict_orbit_cost_ms` has no σ/P term.
For periodic modes, orbit's trapezoidal grid has N_u ≈ period/σ × sps
points; for non-periodic rel mode, N_u ≈ (data_range + 16σ)/σ × sps.
So orbit cost in rel modes scales with 1/σ at fixed P (or fixed data
range). The cost-model constants were fit at σ=50, P=1200 (σ/P=0.042),
so off-axis predictions may be wrong.

This sweep measures actual orbit/pairwise wall-time ratios across
σ/P ∈ {0.01, 0.05, 0.2} (sharp, default, broad) at P=1200, A=1, N=2,
for r ∈ {2, 3, 4}, K = r+2, all four modes, 5 reps to dampen jitter.

We exclude rel_per σ/P > 0.03 — the dispatcher routes those to pairwise
unconditionally (Issue 3 definitional gap).

Output: measured wall time (ms) for each method, ratio orbit/pairwise,
and predicted ratio from cost model. Discrepancy flag at >50% off.
"""
import time
import numpy as np
import warnings
from mpt.tensor import (
    build_exp_tens,
    _cos_sim_exp_tens_ma_orbit,
    _cos_sim_exp_tens_ma_pairwise,
    _predict_orbit_cost_ms,
    _predict_pairwise_kernel_size,
    _pw_per_entry_ms,
)


MODES = [
    ("abs_nonper", False, False),
    ("abs_per",    False, True),
    ("rel_nonper", True,  False),
    ("rel_per",    True,  True),
]
SIGMA_OVER_P_VALUES = [0.01, 0.05, 0.2]
P = 1200.0
R_VALUES = [2, 3, 4]
N = 2
A = 1
REPS = 5  # repetitions per cell for timing
WARMUP = 1


def time_call(fn, reps):
    """Best-of-reps wall time in ms."""
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return min(times)  # best-of, dampens jitter


def main():
    print(
        f"σ/P performance sweep — best-of-{REPS} ms at A={A}, N={N}, K=r+2.\n"
        f"P = {P}; σ/P ∈ {SIGMA_OVER_P_VALUES}.\n"
    )
    header = (
        f"{'mode':<11} {'r':>2} {'σ/P':>5} | "
        f"{'orbit ms':>9} {'pairwise ms':>11} {'meas O/P':>9} "
        f"{'pred O/P':>9} {'fold off':>9}"
    )
    print(header)
    print("-" * len(header))
    for mode_name, is_rel, is_per in MODES:
        for r in R_VALUES:
            K = r + 2
            for sp in SIGMA_OVER_P_VALUES:
                # rel_per σ/P > 0.03: dispatcher always pairwise — but
                # we still measure to characterise.
                sigma = sp * P
                rng = np.random.default_rng(seed=42)
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

                # Warmup
                for _ in range(WARMUP):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        _cos_sim_exp_tens_ma_orbit(d1, d2)
                        _cos_sim_exp_tens_ma_pairwise(d1, d2, verbose=False)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    t_orbit = time_call(
                        lambda: _cos_sim_exp_tens_ma_orbit(d1, d2), REPS,
                    )
                    t_pw = time_call(
                        lambda: _cos_sim_exp_tens_ma_pairwise(
                            d1, d2, verbose=False,
                        ), REPS,
                    )

                # Predicted orbit cost
                pred_orbit = _predict_orbit_cost_ms(
                    r_max=r, A=A, N_x=N, N_y=N, k_vec=[K],
                    any_rel_nonper=(is_rel and not is_per),
                    any_rel_per=(is_rel and is_per),
                )
                pred_pw_size = _predict_pairwise_kernel_size(
                    r_vec=[r], k_vec=[K], A=A, N_x=N, N_y=N,
                )
                pred_pw = pred_pw_size * _pw_per_entry_ms(any_per=is_per)

                meas_ratio = t_orbit / t_pw if t_pw > 0 else float('inf')
                pred_ratio = pred_orbit / pred_pw if pred_pw > 0 else float('inf')
                fold_off = (meas_ratio / pred_ratio
                            if pred_ratio > 0 else float('inf'))
                flag = "    "
                if abs(np.log10(fold_off)) > np.log10(2.0):
                    flag = "MISS"

                print(
                    f"{mode_name:<11} {r:>2} {sp:>5.2f} | "
                    f"{t_orbit:>9.2f} {t_pw:>11.2f} {meas_ratio:>9.2f} "
                    f"{pred_ratio:>9.2f} {fold_off:>8.1f}× {flag}"
                )
            print()
    print("MISS = predicted ratio off by more than 2× from measured.")
    print("orbit/pairwise > 1: pairwise faster (dispatcher should route there).")


if __name__ == "__main__":
    main()
