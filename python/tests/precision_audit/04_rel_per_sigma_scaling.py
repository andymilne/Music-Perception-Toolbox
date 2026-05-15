"""Test rel_per orbit-pw discrepancy as σ/P decreases.

If the discrepancy is σ/P-induced approximation in pairwise (the "wrap-on-diffs"
isn't exact at non-tiny σ/P), it should vanish as σ shrinks.
"""
import numpy as np
from mpt.tensor import (
    build_exp_tens, _cos_sim_exp_tens_sa_orbit, _cos_sim_exp_tens_sa_pairwise,
)


def test_case(seed, r, K, sigma, period=1200.0):
    rng = np.random.default_rng(seed)
    p_a = rng.uniform(0, period, size=K)
    w_a = rng.uniform(0.1, 1.0, size=K)
    p_b = rng.uniform(0, period, size=K)
    w_b = rng.uniform(0.1, 1.0, size=K)
    da = build_exp_tens(p_a, w_a, sigma, r, True, True, period, verbose=False)
    db = build_exp_tens(p_b, w_b, sigma, r, True, True, period, verbose=False)
    o_xy, o_xx, o_yy, _ = _cos_sim_exp_tens_sa_orbit(da, db)
    p_xy, p_xx, p_yy = _cos_sim_exp_tens_sa_pairwise(da, db, verbose=False)
    cos_o = o_xy / np.sqrt(o_xx * o_yy)
    cos_p = p_xy / np.sqrt(p_xx * p_yy)
    return abs(cos_o - cos_p) / abs(cos_p)


print("SA rel_per: σ/P scaling test (r=3, K=6, 5 seeds)")
print("If discrepancy vanishes at small σ, it's σ/P-induced pairwise approximation.")
print(f"{'σ':>6} {'σ/P':>6} | {'med':>9} {'p90':>9} {'max':>9}")
for sigma in [50, 25, 12, 6, 3, 1.5]:
    rels = []
    for seed in range(5):
        rels.append(test_case(seed, 3, 6, sigma))
    sp = sigma / 1200.0
    print(f"{sigma:>6.1f} {sp:>6.4f} | {np.median(rels):>9.2e} {np.quantile(rels, 0.9):>9.2e} {np.max(rels):>9.2e}")
