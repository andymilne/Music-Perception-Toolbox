"""Benchmark: three implementations of the full-image (all-image)
measure on the periodic domain, for relative-periodic mode.

Motivating question. The current toolbox has three candidate routes
to the all-image relative-periodic overlap:

(1) direct image-sum : truncated sum over the periodic image lattice
    Z^r / Z*1, using the projected quadratic form Q on 1^perp;
(2) spectral         : Poisson-summed Fourier expansion of the same
    kernel, constrained to sum-zero multi-indices;
(3) corrected tau-grid : discretisation of the tau-averaging identity
    k_rel_wrap(delta) = (1/P) int_0^P prod_m theta(delta_m + tau) dtau,
    with the wrapped (absolute) kernel inside, evaluated at K_tau
    uniform tau-nodes on [0, P).

The current toolbox's existing tau-grid route is (3) with the
*minimum-image* kernel inside, which computes a *different*
mathematical object (call it (B), the transposition average of the
minimum-image kernel). Route (3) above -- (B) with the wrapped kernel
inside -- computes (C), the same object as (1) and (2). This is
the direct empirical test of whether the "corrected tau-grid" ever
beats direct image-sum or spectral for (C).

Analytical expectation. The wrapped rel kernel is diagonally-shift
invariant (because M 1 = 0), so tau-averaging it with the wrapped
rel kernel inside is redundant. Equivalently, the tau-averaging
identity connecting (3) to (1) is analytical: doing the integral in
closed form collapses to a direct all-image sum on the (r-1)-dim
relative lattice. Route (3) is therefore K_tau evaluations of an
r-dim abs-per image sum, versus (1)'s single evaluation of an
(r-1)-dim rel-per image sum -- larger per-node cost, K_tau times
more nodes.

Empirical grid. sigma/P in {0.01, 0.05, 0.1, 0.2, 0.3, 0.5},
r in {2, 3, 4}, K_x = K_y = 6, weights in [0.3, 1.0]. Timing per
call reported over ``NREPEATS = 5`` repeats after 1 warm-up.

Correctness gate. Every timed route must agree with the reference
(the reference module's lattice sum) to within ``TOL = 1e-10``
relative. If it doesn't, timings are still reported but flagged.

Output. Wall time per call (median across repeats), plus tau-grid
node count K_tau for context.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

# Repo import path -- this script lives at python/tests/precision_audit/
# and imports the reference module at python/tests/references/.
_HERE = Path(__file__).resolve().parent
_PYTHON_DIR = _HERE.parent.parent
sys.path.insert(0, str(_PYTHON_DIR))

from tests.references.ref_general import (  # noqa: E402
    overlap_all_image_rel, overlap_spectral_rel,
    _truncation_L, _truncation_L_rel, _spectral_M,
)


# ---------------------------------------------------------------------------
# Corrected tau-grid: (C) via tau-average of the wrapped absolute kernel
# ---------------------------------------------------------------------------


def overlap_taugrid_wrapped(
    centres_x: np.ndarray, weights_x: np.ndarray,
    centres_y: np.ndarray, weights_y: np.ndarray,
    sigma: float, period: float, tol: float = 1e-15,
    n_tau: int | None = None,
) -> tuple[float, int]:
    """Return ``(overlap, n_tau)`` in the raw kernel-sum convention
    matching :func:`overlap_all_image_rel`.

    Implements the identity
    ``k_rel_wrap(delta) = (1/P) int_0^P prod_m theta(delta_m + tau) dtau``,
    with ``theta(x) = sum_{n=-L..L} exp(-(x + n P)^2 / (4 sigma^2))``
    the 1-D wrapped-Gaussian overlap. Trapezoid with ``n_tau`` uniform
    nodes on ``[0, P)``. Multiplied by
    ``P sqrt(r) / (2 sigma sqrt(pi))`` to match the raw kernel-sum
    convention.

    When ``n_tau`` is ``None``, use the toolbox's default
    ``ceil(2 pi P/sigma * margin)``, ``margin ~ 1.6`` at ``tol=1e-15``,
    floored at 64 -- matching :func:`mpt._tensor._nested_contraction.auto_ntau`.
    """
    centres_x = np.asarray(centres_x, dtype=np.float64)
    centres_y = np.asarray(centres_y, dtype=np.float64)
    weights_x = np.asarray(weights_x, dtype=np.float64)
    weights_y = np.asarray(weights_y, dtype=np.float64)
    Kx, r = centres_x.shape

    if n_tau is None:
        base = 2.0 * math.pi * period / sigma
        margin = 1.0 + 0.5 * (-math.log10(max(tol, 1e-16))) / 12.0
        n_tau = int(max(64, math.ceil(base * margin)))

    L = _truncation_L(sigma, period, tol)
    coef = -1.0 / (4.0 * sigma * sigma)

    # delta[j, k, m] = c^X_{j, m} - c^Y_{k, m}
    delta = centres_x[:, None, :] - centres_y[None, :, :]  # (Kx, Ky, r)

    taus = np.linspace(0.0, period, n_tau, endpoint=False)  # (T,)

    # For each pair (j, k), each component m, each tau_i, evaluate
    # theta(delta_{jkm} + tau_i). Shape (Kx, Ky, r, T).
    # We then image-sum over n and product across r.
    # Reduce (delta + tau) to [-P/2, P/2] per component so that the
    # n=0 term is the dominant one and the L truncation applies -- the
    # same reduction the reference makes. Without this, at small
    # sigma/P where L=0, the sum can miss the entire near-boundary
    # image.
    n_grid = np.arange(-L, L + 1, dtype=np.float64)  # (2L+1,)
    # d[j, k, m, i, n] = (delta[j, k, m] + tau[i]) reduced + n * P
    x = delta[:, :, :, None] + taus[None, None, None, :]  # (Kx, Ky, r, T)
    x = (x + 0.5 * period) % period - 0.5 * period
    d = x[:, :, :, :, None] + period * n_grid[None, None, None, None, :]
    # Sum over n gives theta at each (j, k, m, i)
    theta = np.exp(coef * d * d).sum(axis=-1)  # (Kx, Ky, r, T)
    # Product across r gives prod_m theta(delta_{jkm} + tau_i)
    integrand = theta.prod(axis=2)  # (Kx, Ky, T)
    # Weighted sum over pairs, mean over tau
    inner = (weights_x[:, None, None]
             * weights_y[None, :, None]
             * integrand).sum(axis=(0, 1))  # (T,)
    tau_mean = float(inner.mean())  # (1/n_tau) sum_i integrand

    prefactor = period * math.sqrt(r) / (2.0 * sigma * math.sqrt(math.pi))
    return prefactor * tau_mean, n_tau


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_data(K, r, period, seed):
    rng = np.random.default_rng(seed)
    Cx = rng.uniform(0.0, period, size=(K, r))
    Wx = rng.uniform(0.3, 1.0, size=K)
    Cy = rng.uniform(0.0, period, size=(K, r))
    Wy = rng.uniform(0.3, 1.0, size=K)
    return Cx, Wx, Cy, Wy


def _time_call(func, *args, n_repeats=5, warmup=1, **kwargs):
    """Return (median time in seconds, result of last call)."""
    for _ in range(warmup):
        _ = func(*args, **kwargs)
    times = []
    result = None
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return float(np.median(times)), result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


NREPEATS = 5
TOL_AGREE = 1e-10


def _run():
    period = 1.0
    K = 6
    print("== Benchmark: three routes to the (C) full-image rel-per overlap")
    print("   K = K_x = K_y =", K, "; N_REPEATS =", NREPEATS)
    print()
    print(f"{'sig/P':>6} {'r':>3} "
          f"{'L_abs':>5} {'L_rel':>5} {'M_spec':>7} {'n_tau':>6} "
          f"{'t_imgsum(s)':>12} {'t_spec(s)':>12} {'t_taugrid(s)':>14} "
          f"{'agree':>7} {'err_spc':>9} {'err_tau':>9}")

    for sigma_over_P in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]:
        sigma = sigma_over_P * period
        L_abs = _truncation_L(sigma, period, 1e-15)
        L_rel = _truncation_L_rel(sigma, period, 1e-15)
        M_spec = _spectral_M(sigma, period, 1e-15)
        for r in [2, 3, 4]:
            Cx, Wx, Cy, Wy = _random_data(K, r, period, seed=1000 + r)

            t_lat, o_lat = _time_call(
                overlap_all_image_rel, Cx, Wx, Cy, Wy, sigma, period,
                n_repeats=NREPEATS)
            t_spc, o_spc = _time_call(
                overlap_spectral_rel, Cx, Wx, Cy, Wy, sigma, period,
                n_repeats=NREPEATS)
            t_tau, (o_tau, n_tau) = _time_call(
                overlap_taugrid_wrapped, Cx, Wx, Cy, Wy, sigma, period,
                n_repeats=NREPEATS)

            # Agreement check
            scale = max(abs(o_lat), 1e-30)
            err_spec = abs(o_spc - o_lat) / scale
            err_tau = abs(o_tau - o_lat) / scale
            agree_spec = err_spec < TOL_AGREE
            agree_tau = err_tau < TOL_AGREE
            agree = "OK" if (agree_spec and agree_tau) else "FAIL"

            print(f"{sigma_over_P:>6.3f} {r:>3d} "
                  f"{L_abs:>5d} {L_rel:>5d} {M_spec:>7d} {n_tau:>6d} "
                  f"{t_lat:>12.4e} {t_spc:>12.4e} {t_tau:>14.4e} "
                  f"{agree:>7s} "
                  f"{err_spec:>9.1e} {err_tau:>9.1e}")

    print()
    print("Legend:")
    print("  L_abs   -- abs-per per-component image truncation (used by tau-grid inner)")
    print("  L_rel   -- rel-per lattice truncation (used by image-sum route)")
    print("  M_spec  -- Fourier mode truncation per free index")
    print("  n_tau   -- number of tau nodes for the tau-average")
    print("  t_imgsum   -- direct all-image lattice sum, route (1)")
    print("  t_spec     -- spectral (Poisson-Fourier), route (2)")
    print("  t_taugrid  -- corrected tau-grid with wrapped kernel inside, route (3)")


if __name__ == "__main__":
    _run()
