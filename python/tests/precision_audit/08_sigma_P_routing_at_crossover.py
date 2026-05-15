"""σ/P routing decision test: does σ/P affect dispatcher correctness?

The cost-model has no σ/P term, but for periodic and rel-non-per modes
orbit's actual cost has a 1/σ factor (trapezoidal grid resolution).
This raises the question: at the K where orbit and pairwise have similar
cost (the routing boundary), does varying σ/P flip the optimal choice?

Design: for each mode that uses a u-grid (rel-nonper, rel-per), find the
K at which orbit ≈ pairwise at the default σ/P=0.042. Then sweep σ/P at
that K and measure whether the optimal choice (orbit vs pairwise) flips.

For abs modes there's no u-grid, so σ/P doesn't enter orbit cost (only
the pairwise-via-wrap cost via _PW_PER_ENTRY_MS_PER, which is σ/P-
independent already). Skip abs modes here.
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


P = 1200.0
SIGMA_OVER_P_VALUES = [0.01, 0.03, 0.042, 0.1, 0.2, 0.5]
N = 2
A = 1
REPS = 5

# Crossover K determined approximately by inspecting the cost model.
# At A=1, N=2, default σ/P=0.042:
#   rel_nonper:   r=2 K=10, r=3 K=8, r=4 K=8 — orbit BASE ~5 ms,
#                 pairwise scales as binom(K,r)².
#   rel_per:      r=2 K=14, r=3 K=10, r=4 K=10 — pairwise wraps so
#                 per-entry is 7e-4 vs 1e-4, lower K threshold.
CROSSOVER_K = {
    ("rel_nonper", 2): 10,
    ("rel_nonper", 3): 8,
    ("rel_nonper", 4): 8,
    ("rel_per",    2): 14,
    ("rel_per",    3): 10,
    ("rel_per",    4): 10,
}


def time_call(fn, reps):
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return min(times)


def measure_cell(is_rel, is_per, r, K, sigma):
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # warmup
        _cos_sim_exp_tens_ma_orbit(d1, d2)
        _cos_sim_exp_tens_ma_pairwise(d1, d2, verbose=False)
        t_orbit = time_call(
            lambda: _cos_sim_exp_tens_ma_orbit(d1, d2), REPS,
        )
        t_pw = time_call(
            lambda: _cos_sim_exp_tens_ma_pairwise(
                d1, d2, verbose=False,
            ), REPS,
        )
    pred_orbit = _predict_orbit_cost_ms(
        r_max=r, A=A, N_x=N, N_y=N, k_vec=[K],
        any_rel_nonper=(is_rel and not is_per),
        any_rel_per=(is_rel and is_per),
    )
    pred_pw_size = _predict_pairwise_kernel_size(
        r_vec=[r], k_vec=[K], A=A, N_x=N, N_y=N,
    )
    pred_pw = pred_pw_size * _pw_per_entry_ms(any_per=is_per)
    return t_orbit, t_pw, pred_orbit, pred_pw


def main():
    print(
        f"σ/P routing test at crossover K, P={P}, A={A}, N={N}, "
        f"best-of-{REPS} ms.\n"
    )
    header = (
        f"{'mode':<11} {'r':>2} {'K':>2} {'σ/P':>5} | "
        f"{'orbit ms':>9} {'pw ms':>9} {'measured':>9} "
        f"{'predicted':>9} {'choice':>9}"
    )
    print(header)
    print("-" * len(header))
    for (mode_name, r), K in CROSSOVER_K.items():
        is_rel = True
        is_per = (mode_name == "rel_per")
        for sp in SIGMA_OVER_P_VALUES:
            if mode_name == "rel_per" and sp > 0.03:
                # dispatcher routes away regardless; skip
                continue
            sigma = sp * P
            t_orbit, t_pw, pred_orbit, pred_pw = measure_cell(
                is_rel, is_per, r, K, sigma,
            )
            measured_choice = "mobius" if t_orbit < t_pw else "bulger"
            predicted_choice = "mobius" if pred_orbit < pred_pw else "bulger"
            agree = "" if measured_choice == predicted_choice else "DISAGREE"
            print(
                f"{mode_name:<11} {r:>2} {K:>2} {sp:>5.3f} | "
                f"{t_orbit:>9.2f} {t_pw:>9.2f} {measured_choice:>9} "
                f"{predicted_choice:>9} {agree}"
            )
        print()
    print(
        "DISAGREE = the cost model would route the wrong way for this cell."
    )


if __name__ == "__main__":
    main()
