"""Check different window sizes for rel_nonper r=2 precision."""
import numpy as np
from mpt._mobius import inner_product_orbit_grid

def _orbit_inner_rel_custom(p_a, w_a, p_b, w_b, sigma, r, samples_per_sigma=10,
                              window_sigma=4.0):
    """Configurable-window version of _orbit_inner_rel for non-periodic rel mode."""
    u_min = p_b.min() - p_a.max() - window_sigma * sigma
    u_max = p_b.max() - p_a.min() + window_sigma * sigma
    N_u = max(64, int(np.ceil((u_max - u_min) / sigma * samples_per_sigma)))
    u_grid = np.linspace(u_min, u_max, N_u)
    diffs = (p_a[None, :, None] - p_b[None, None, :]
             + u_grid[:, None, None])
    K_u = np.exp(-(diffs ** 2) / (4 * sigma ** 2))
    F = inner_product_orbit_grid(K_u, w_a, w_b, r)
    integral = float(np.trapezoid(F, u_grid))
    c = sigma * np.sqrt(2 * np.pi / r)
    return (sigma * np.sqrt(np.pi)) ** r * integral / c ** 2


def cosine_via_window(seed, A, r, K, N, window, samples_per_sigma=10, sigma=50.0):
    """Compute orbit MA cosine using custom window."""
    rng = np.random.default_rng(seed)
    p_x = [rng.uniform(0, 1200, size=(K, N)) for _ in range(A)]
    p_y = [rng.uniform(0, 1200, size=(K, N)) for _ in range(A)]
    w_x = [rng.uniform(0.1, 1.0, size=(K, N)) for _ in range(A)]
    w_y = [rng.uniform(0.1, 1.0, size=(K, N)) for _ in range(A)]

    # Per-attribute orbit MA inner matrices via custom-window per-pair calls
    P_xy = np.ones((N, N))
    P_xx = np.ones((N, N))
    P_yy = np.ones((N, N))
    for a in range(A):
        Px, Py, Wx, Wy = p_x[a], p_y[a], w_x[a], w_y[a]
        I_xy = np.empty((N, N))
        I_xx = np.empty((N, N))
        I_yy = np.empty((N, N))
        for i in range(N):
            for j in range(N):
                I_xy[i, j] = _orbit_inner_rel_custom(
                    Px[:, i], Wx[:, i], Py[:, j], Wy[:, j], sigma, r,
                    samples_per_sigma, window)
                I_xx[i, j] = _orbit_inner_rel_custom(
                    Px[:, i], Wx[:, i], Px[:, j], Wx[:, j], sigma, r,
                    samples_per_sigma, window)
                I_yy[i, j] = _orbit_inner_rel_custom(
                    Py[:, i], Wy[:, i], Py[:, j], Wy[:, j], sigma, r,
                    samples_per_sigma, window)
        P_xy *= I_xy
        P_xx *= I_xx
        P_yy *= I_yy
    ip_xy = float(P_xy.sum())
    ip_xx = float(P_xx.sum())
    ip_yy = float(P_yy.sum())
    return ip_xy / np.sqrt(ip_xx * ip_yy)


def cosine_pairwise(seed, A, r, K, N, sigma=50.0):
    """Pairwise cosine for reference."""
    from mpt.tensor import build_exp_tens, _cos_sim_exp_tens_ma_pairwise
    rng = np.random.default_rng(seed)
    p_x = [rng.uniform(0, 1200, size=(K, N)) for _ in range(A)]
    p_y = [rng.uniform(0, 1200, size=(K, N)) for _ in range(A)]
    w_x = [rng.uniform(0.1, 1.0, size=(K, N)) for _ in range(A)]
    w_y = [rng.uniform(0.1, 1.0, size=(K, N)) for _ in range(A)]
    sigma_vec = [sigma] * A
    r_vec = [r] * A
    groups = list(range(A))
    is_rel_vec = [True] * A
    is_per_vec = [False] * A
    period_vec = [0.0] * A
    dx = build_exp_tens(p_x, w_x, sigma_vec, r_vec, groups, is_rel_vec, is_per_vec, period_vec, verbose=False)
    dy = build_exp_tens(p_y, w_y, sigma_vec, r_vec, groups, is_rel_vec, is_per_vec, period_vec, verbose=False)
    ip_xy, ip_xx, ip_yy = _cos_sim_exp_tens_ma_pairwise(dx, dy, verbose=False)
    return ip_xy / np.sqrt(ip_xx * ip_yy)


# Sweep window sizes on a tough case
print("MA rel_nonper r=2 K=4 N=4 A=2 (10 seeds, comparing windows)")
print(f"{'window':>8}  {'samples/σ':>10}  {'med_rel':>10}  {'max_rel':>10}")
for window, sps in [(4.0, 10), (6.0, 10), (8.0, 10), (10.0, 10), (12.0, 10), (16.0, 10),
                     (8.0, 5), (12.0, 5)]:
    rels = []
    for seed in range(10):
        cos_ref = cosine_pairwise(seed, 2, 2, 4, 4)
        cos_o = cosine_via_window(seed, 2, 2, 4, 4, window=window, samples_per_sigma=sps)
        rels.append(abs(cos_o - cos_ref) / abs(cos_ref))
    print(f"  {window:>5.1f}σ  {sps:>10}  {np.median(rels):>10.2e}  {np.max(rels):>10.2e}")

print()
print("Same but at r=3 K=4 N=4 A=2 (sanity, expect FP at any window):")
for window, sps in [(4.0, 10), (6.0, 10), (8.0, 10)]:
    rels = []
    for seed in range(10):
        cos_ref = cosine_pairwise(seed, 2, 3, 4, 4)
        cos_o = cosine_via_window(seed, 2, 3, 4, 4, window=window, samples_per_sigma=sps)
        rels.append(abs(cos_o - cos_ref) / abs(cos_ref))
    print(f"  {window:>5.1f}σ  {sps:>10}  {np.median(rels):>10.2e}  {np.max(rels):>10.2e}")
