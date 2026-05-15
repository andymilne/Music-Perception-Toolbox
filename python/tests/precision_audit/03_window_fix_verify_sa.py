"""Test that widening _orbit_inner_rel's window fixes the SA bug."""
import numpy as np
from mpt._mobius import inner_product_orbit_grid
from mpt.tensor import build_exp_tens, _cos_sim_exp_tens_sa_pairwise


def _orbit_inner_rel_widened(p_a, w_a, p_b, w_b, sigma, r, samples_per_sigma=10,
                                window_sigma=8.0):
    u_min = p_b.min() - p_a.max() - window_sigma * sigma
    u_max = p_b.max() - p_a.min() + window_sigma * sigma
    N_u = max(64, int(np.ceil((u_max - u_min) / sigma * samples_per_sigma)))
    u_grid = np.linspace(u_min, u_max, N_u)
    diffs = (p_a[None, :, None] - p_b[None, None, :] + u_grid[:, None, None])
    K_u = np.exp(-(diffs ** 2) / (4 * sigma ** 2))
    F = inner_product_orbit_grid(K_u, w_a, w_b, r)
    integral = float(np.trapezoid(F, u_grid))
    c = sigma * np.sqrt(2 * np.pi / r)
    return (sigma * np.sqrt(np.pi)) ** r * integral / c ** 2


def sa_orbit_widened_cosine(p_a, w_a, p_b, w_b, sigma, r, window_sigma):
    ip_xy = _orbit_inner_rel_widened(p_a, w_a, p_b, w_b, sigma, r, window_sigma=window_sigma)
    ip_xx = _orbit_inner_rel_widened(p_a, w_a, p_a, w_a, sigma, r, window_sigma=window_sigma)
    ip_yy = _orbit_inner_rel_widened(p_b, w_b, p_b, w_b, sigma, r, window_sigma=window_sigma)
    return ip_xy / np.sqrt(ip_xx * ip_yy)


def sa_pairwise_cosine(p_a, w_a, p_b, w_b, sigma, r):
    da = build_exp_tens(p_a, w_a, sigma, r, True, False, 0.0, verbose=False)
    db = build_exp_tens(p_b, w_b, sigma, r, True, False, 0.0, verbose=False)
    ip_xy, ip_xx, ip_yy = _cos_sim_exp_tens_sa_pairwise(da, db, verbose=False)
    return ip_xy / np.sqrt(ip_xx * ip_yy)


print("SA rel_nonper precision: orbit (varied window) vs pairwise reference")
print(f"{'r':>2} {'K':>2} {'win':>5} | {'med_rel':>10} {'p90':>10} {'max_rel':>10}")

sigma = 50.0
for r in [2, 3, 4]:
    for K in [4, 6]:
        for window in [4.0, 6.0, 8.0, 12.0]:
            rels = []
            for seed in range(30):
                rng = np.random.default_rng(seed)
                p_a = rng.uniform(0, 1200, size=K)
                w_a = rng.uniform(0.1, 1.0, size=K)
                p_b = rng.uniform(0, 1200, size=K)
                w_b = rng.uniform(0.1, 1.0, size=K)
                cos_o = sa_orbit_widened_cosine(p_a, w_a, p_b, w_b, sigma, r, window)
                cos_p = sa_pairwise_cosine(p_a, w_a, p_b, w_b, sigma, r)
                rels.append(abs(cos_o - cos_p) / abs(cos_p))
            print(f"{r:>2} {K:>2} {window:>4.1f}σ | {np.median(rels):>10.2e} {np.quantile(rels, 0.9):>10.2e} {np.max(rels):>10.2e}")
        print()
