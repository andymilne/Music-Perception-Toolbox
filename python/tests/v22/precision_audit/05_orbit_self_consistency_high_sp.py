"""High σ/P — does orbit stay self-consistent? Compare two orbit configs."""
import numpy as np
from mpt._mobius import inner_product_orbit_grid


def orbit_rel_per(p_a, w_a, p_b, w_b, sigma, r, period, sps):
    N_u = max(64, int(np.ceil(period / sigma * sps)))
    u_grid = np.linspace(0.0, period, N_u, endpoint=False)
    du = period / N_u
    diffs = (p_a[None, :, None] - p_b[None, None, :] + u_grid[:, None, None])
    diffs = diffs - period * np.floor(diffs / period + 0.5)
    K_u = np.exp(-(diffs ** 2) / (4 * sigma ** 2))
    F = inner_product_orbit_grid(K_u, w_a, w_b, r)
    integral = float(F.sum() * du)
    c = sigma * np.sqrt(2 * np.pi / r)
    return (sigma * np.sqrt(np.pi)) ** r * integral / c ** 2


print("rel_per orbit self-consistency at varied σ/P")
print("(orbit at sps=10 vs orbit at sps=50 — if trapezoid converges, agreement is FP)")
print(f"{'σ':>6} {'σ/P':>6} | {'sps=10':>13} {'sps=50':>13} {'rel diff':>10}")
period = 1200.0
r = 2
K = 6
rng = np.random.default_rng(0)
p_a = rng.uniform(0, period, K)
w_a = rng.uniform(0.1, 1.0, K)
p_b = rng.uniform(0, period, K)
w_b = rng.uniform(0.1, 1.0, K)
for sigma in [50, 100, 200, 400, 800, 1200]:
    v10 = orbit_rel_per(p_a, w_a, p_b, w_b, sigma, r, period, 10)
    v50 = orbit_rel_per(p_a, w_a, p_b, w_b, sigma, r, period, 50)
    diff = abs(v10 - v50) / abs(v50)
    print(f"{sigma:>6.0f} {sigma/period:>6.3f} | {v10:>13.6e} {v50:>13.6e} {diff:>10.2e}")
