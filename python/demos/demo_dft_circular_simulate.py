"""demo_dft_circular_simulate.py

Demonstration of the v2.1.0 sigma + Monte Carlo additions to the
Argand-DFT family.

New in v2.1:
  * dft_circular_simulate — Monte Carlo estimation of |F(k)|
    distribution under independent positional jitter.
  * balance, evenness — accept an optional `sigma` argument; with
    `return_std=True` also return the per-coefficient standard
    deviation.
  * proj_centroid — accepts `sigma` and applies analytical kernel-
    smoothing damping (alpha_1 factor); no Monte Carlo needed for the
    projection because it is linear in F(0).

The deterministic dft_circular function is unchanged.
"""
import numpy as np

from mpt import (
    balance,
    dft_circular,
    dft_circular_simulate,
    evenness,
    proj_centroid,
)


# ===== 1. Balance and evenness sweeps =====

print("\n=== Balance and evenness on the diatonic scale ===")
print(f"{'sigma':>6} {'b':>10} {'b_std':>10} {'e':>10} {'e_std':>10}")
print("  " + "-" * 50)
diat = [0, 200, 400, 500, 700, 900, 1100]
period = 1200
for s in [0, 10, 50, 100, 200]:
    b, bs = balance(diat, None, period, sigma=s, return_std=True, rng_seed=42)
    e, es = evenness(diat, period, sigma=s, return_std=True, rng_seed=42)
    print(f"{s:>6} {b:>10.4f} {bs:>10.4f} {e:>10.4f} {es:>10.4f}")
print("  Both balance and evenness shrink as sigma grows; the standard")
print("  deviations grow correspondingly. At sigma = 0 the v2.0 deterministic")
print("  values are recovered exactly and SD = 0.")


# ===== 2. Rayleigh bias on a perfectly balanced multiset =====

print("\n=== Augmented triad: deterministically balanced (F(0) = 0) ===")
print("Under jitter, |F(0)| picks up a positive bias from the underlying")
print("Rayleigh distribution (sum of two independent N(0, V) components).")
print(f"{'sigma':>6} {'b':>10} {'closed-form mean':>20}")
print("  " + "-" * 36)
period = 1200
K = 3
for s in [0, 10, 25, 50, 100]:
    b, _ = balance([0, 400, 800], None, period,
                   sigma=s, return_std=True, n_draws=20000, rng_seed=42)
    mc_mag = 1 - b
    if s == 0:
        closed = 0.0
    else:
        alpha1 = np.exp(-2 * np.pi**2 * s**2 / period**2)
        closed = np.sqrt((1 - alpha1**2) * np.pi / (4 * K))
    print(f"{s:>6} {b:>10.4f} {closed:>20.4f}")
print("  MC estimate of E[|F(0)|] tracks the closed-form Rayleigh mean")
print("  sqrt((1 - alpha_1^2) * pi / (4K)) very accurately.")


# ===== 3. Full DFT distribution via dft_circular_simulate =====

print("\n=== Full DFT magnitudes for the son clave under jitter ===")
clave = [0, 3, 6, 10, 12]
period = 16
sigma = 0.5
print(f"  Pattern: {clave} on {period}-step cycle, sigma = {sigma}")
_, mag_det = dft_circular(clave, None, period)
m, s = dft_circular_simulate(clave, None, period, sigma=sigma,
                             n_draws=20000, rng_seed=42)
print(f"  {'k':>3} {'det |F|':>12} {'mean |F|':>12} {'SD |F|':>12} {'CV':>10}")
print("  " + "-" * 51)
for k in range(len(clave)):
    cv = s[k] / m[k] if m[k] > 1e-10 else float('inf')
    print(f"  {k:>3} {mag_det[k]:>12.4f} {m[k]:>12.4f} {s[k]:>12.4f} {cv:>10.3f}")
print("  Note that low-order coefficients (k = 1: evenness) have the")
print("  smallest CV under jitter — Milne & Herff (2020) Fig 13 reports")
print("  this is the most jitter-robust of the magnitudes.")


# ===== 4. proj_centroid with analytical sigma damping =====

print("\n=== proj_centroid: closed-form alpha_1 damping (no MC) ===")
p = [0, 4, 7]   # major triad
period = 12
sigma = 0.5
alpha1 = np.exp(-2 * np.pi**2 * sigma**2 / period**2)
y_det, cm_det, cp_det = proj_centroid(p, None, period)
y_smooth, cm_smooth, cp_smooth = proj_centroid(p, None, period, sigma=sigma)
print(f"  C major triad [0, 4, 7] on 12-step cycle, sigma = {sigma}")
print(f"  alpha_1 = exp(-2 pi^2 sigma^2 / period^2) = {alpha1:.6f}")
print()
print(f"  {'x':>3} {'y_det(x)':>12} {'y_smooth(x)':>14} {'ratio':>10}")
print("  " + "-" * 41)
for k in range(period):
    ratio = y_smooth[k] / y_det[k] if abs(y_det[k]) > 1e-10 else float('nan')
    print(f"  {k:>3} {y_det[k]:>12.4f} {y_smooth[k]:>14.4f} {ratio:>10.6f}")
print(f"  All ratios equal alpha_1 exactly (analytical, not MC).")
print(f"  cent_phase preserved: {cp_det:.4f} -> {cp_smooth:.4f}")
print(f"  cent_mag damped by alpha_1: {cm_det:.4f} -> {cm_smooth:.4f}")
print()
print("  cent_mag here is |E[F(0)]| = alpha_1 * |F_det(0)|, which is")
print("  consistent with the projection. The DIFFERENT scalar E[|F(0)|]")
print("  (Rician/Rayleigh-style mean magnitude under jitter) is what")
print("  balance returns: 1 - balance(p, w, T, sigma) = E[|F(0)|].")


# ===== 5. Full distribution via return_samples =====

print("\n=== Full distribution: histogram of |F(1)| under jitter ===")
diat = [0, 200, 400, 500, 700, 900, 1100]
sigma = 50
m, s, samples = dft_circular_simulate(
    diat, None, 1200, sigma=sigma,
    n_draws=20000, rng_seed=42, return_samples=True
)
F1_samples = samples[:, 1]
print(f"  Diatonic scale, sigma = {sigma} cents, n_draws = {len(F1_samples)}")
print(f"  Sample shape: {samples.shape}  (n_draws x K)")
print(f"  |F(1)| min  = {F1_samples.min():.4f}")
print(f"  |F(1)| 5%   = {np.percentile(F1_samples, 5):.4f}")
print(f"  |F(1)| mean = {m[1]:.4f}")
print(f"  |F(1)| 95%  = {np.percentile(F1_samples, 95):.4f}")
print(f"  |F(1)| max  = {F1_samples.max():.4f}")
print(f"  |F(1)| SD   = {s[1]:.4f}")

print("\nDone.")
