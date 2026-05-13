"""Integration tests: Möbius–orbit cosine similarity vs existing toolbox.

These are the regression tests that v2.2's drop-in replacement must pass.
The orbit-path cosine values must match the v2.1 ``cos_sim_exp_tens`` to
floating-point precision in all four mode combinations and across a range
of r and n.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/home/claude/Music-Perception-Toolbox/python")

import numpy as np
import pytest

from mpt._mobius import (
    get_orbit_table,
    inner_product_orbit,
    inner_product_orbit_grid,
    total_mass_abs,
)
from mpt import build_exp_tens, cos_sim_exp_tens


def wrap(d, P):
    return d - P * np.floor(d / P + 0.5)


def cos_sim_orbit_abs(p_A, w_A, p_B, w_B, sigma, r, is_per, P):
    """Cosine similarity in absolute mode using the Möbius method's approach."""
    diffs = p_A[:, None] - p_B[None, :]
    if is_per:
        diffs = wrap(diffs, P)
    K_AB = np.exp(-(diffs ** 2) / (4 * sigma ** 2))

    diffs_AA = p_A[:, None] - p_A[None, :]
    if is_per:
        diffs_AA = wrap(diffs_AA, P)
    K_AA = np.exp(-(diffs_AA ** 2) / (4 * sigma ** 2))

    diffs_BB = p_B[:, None] - p_B[None, :]
    if is_per:
        diffs_BB = wrap(diffs_BB, P)
    K_BB = np.exp(-(diffs_BB ** 2) / (4 * sigma ** 2))

    AB = inner_product_orbit(K_AB, w_A, w_B, r, prefactor=(sigma * np.sqrt(np.pi)) ** r)
    AA = inner_product_orbit(K_AA, w_A, w_A, r, prefactor=(sigma * np.sqrt(np.pi)) ** r)
    BB = inner_product_orbit(K_BB, w_B, w_B, r, prefactor=(sigma * np.sqrt(np.pi)) ** r)
    return AB / np.sqrt(AA * BB)


def cos_sim_orbit_rel(p_A, w_A, p_B, w_B, sigma, r, is_per, P, N_u=600):
    """Cosine similarity in relative mode using orbit Möbius + integration."""
    def _inner(pA, wA, pB, wB):
        if is_per:
            u_grid = np.linspace(0, P, N_u, endpoint=False)
            du = P / N_u
            diffs = (pA[None, :, None] - pB[None, None, :]) + u_grid[:, None, None]
            diffs = wrap(diffs, P)
        else:
            u_min = pB.min() - pA.max() - 4 * sigma
            u_max = pB.max() - pA.min() + 4 * sigma
            u_grid = np.linspace(u_min, u_max, N_u)
            diffs = (pA[None, :, None] - pB[None, None, :]) + u_grid[:, None, None]
        K_u = np.exp(-(diffs ** 2) / (4 * sigma ** 2))
        F = inner_product_orbit_grid(K_u, wA, wB, r)
        if is_per:
            integral = F.sum() * du
        else:
            integral = np.trapezoid(F, u_grid)
        c = sigma * np.sqrt(2 * np.pi / r)
        return (sigma * np.sqrt(np.pi)) ** r * integral / c ** 2

    AB = _inner(p_A, w_A, p_B, w_B)
    AA = _inner(p_A, w_A, p_A, w_A)
    BB = _inner(p_B, w_B, p_B, w_B)
    return AB / np.sqrt(AA * BB)


# ----------------------------------------------------------------------
# Match toolbox cos_sim_exp_tens across all four modes
# ----------------------------------------------------------------------


@pytest.mark.parametrize("r", [2, 3])
@pytest.mark.parametrize("n", [8, 12])
@pytest.mark.parametrize("is_per", [False, True])
@pytest.mark.parametrize("is_rel", [False, True])
def test_cos_sim_orbit_matches_toolbox(r, n, is_per, is_rel):
    """Orbit-path cosine matches toolbox to floating-point precision."""
    P = 1200.0
    sigma = 12.0  # sigma/P = 0.01, well within the integration regime
    rng = np.random.default_rng(seed=hash((r, n, is_per, is_rel)) & 0xFFFF)

    if is_per:
        p_A = np.sort(rng.uniform(50, P - 50, n))
        p_B = np.sort(rng.uniform(50, P - 50, n))
    else:
        p_A = np.sort(rng.uniform(100, 2000, n))
        p_B = np.sort(rng.uniform(100, 2000, n))
    w_A = rng.uniform(0.5, 1.5, n)
    w_B = rng.uniform(0.5, 1.5, n)

    T_A = build_exp_tens(p_A, w_A, sigma, r, is_rel, is_per, P, verbose=False)
    T_B = build_exp_tens(p_B, w_B, sigma, r, is_rel, is_per, P, verbose=False)
    cos_toolbox = float(cos_sim_exp_tens(T_A, T_B))

    if is_rel:
        cos_orbit = cos_sim_orbit_rel(p_A, w_A, p_B, w_B, sigma, r, is_per, P, N_u=2000 if is_per else 800)
    else:
        cos_orbit = cos_sim_orbit_abs(p_A, w_A, p_B, w_B, sigma, r, is_per, P)

    abs_err = abs(cos_toolbox - cos_orbit)
    rel_err = abs_err / max(abs(cos_toolbox), abs(cos_orbit), 1e-300)

    # Cosine is itself bounded in [-1, 1], so absolute error is the meaningful
    # measure here — relative error explodes when cos is near zero (cancellation
    # regime). We accept either tight relative error OR tight absolute error.
    assert rel_err < 1e-9 or abs_err < 1e-12, (
        f"r={r}, n={n}, is_per={is_per}, is_rel={is_rel}: "
        f"toolbox={cos_toolbox:.10e}, orbit={cos_orbit:.10e}, "
        f"abs_err={abs_err:.2e}, rel_err={rel_err:.2e}"
    )


# ----------------------------------------------------------------------
# Higher r, where toolbox direct path becomes infeasible
# ----------------------------------------------------------------------


@pytest.mark.parametrize("r", [4])
def test_cos_sim_orbit_at_r4(r):
    """Orbit cosine at r=4 matches toolbox where the latter still completes."""
    sigma = 12.0
    P = 1200.0
    rng = np.random.default_rng(seed=4242)
    n = 12  # n=12 so toolbox r=4 still completes in seconds
    p_A = np.sort(rng.uniform(100, 5000, n))
    p_B = np.sort(rng.uniform(100, 5000, n))
    w_A = rng.uniform(0.5, 1.5, n)
    w_B = rng.uniform(0.5, 1.5, n)

    T_A = build_exp_tens(p_A, w_A, sigma, r, False, False, P, verbose=False)
    T_B = build_exp_tens(p_B, w_B, sigma, r, False, False, P, verbose=False)
    cos_toolbox = float(cos_sim_exp_tens(T_A, T_B))

    cos_orbit = cos_sim_orbit_abs(p_A, w_A, p_B, w_B, sigma, r, False, P)
    abs_err = abs(cos_toolbox - cos_orbit)
    rel_err = abs_err / max(abs(cos_toolbox), abs(cos_orbit), 1e-300)
    assert rel_err < 1e-8 or abs_err < 1e-12, (
        f"r={r}: toolbox={cos_toolbox:.10e}, orbit={cos_orbit:.10e}, "
        f"abs_err={abs_err:.2e}, rel_err={rel_err:.2e}"
    )


# ----------------------------------------------------------------------
# Performance: Möbius method completes in reasonable time at full n=64
# ----------------------------------------------------------------------


@pytest.mark.parametrize("r", [2, 3, 4])
def test_cos_sim_orbit_performance_n64(r):
    """At n=64, orbit cosine completes in under a few seconds."""
    import time
    sigma = 12.0
    P = 1200.0
    rng = np.random.default_rng(seed=64 + r)
    n = 64
    p_A = np.sort(rng.uniform(0, 7000, n))
    p_B = np.sort(rng.uniform(0, 7000, n))
    w_A = rng.uniform(0.3, 1.5, n)
    w_B = rng.uniform(0.3, 1.5, n)
    # Warmup (orbit table + numpy)
    cos_sim_orbit_abs(p_A, w_A, p_B, w_B, sigma, r, False, P)
    t0 = time.perf_counter()
    cos_sim_orbit_abs(p_A, w_A, p_B, w_B, sigma, r, False, P)
    elapsed = time.perf_counter() - t0
    # Generous threshold: r=4 should be well under 1 sec; r=3 well under 100ms
    threshold = {2: 0.1, 3: 0.3, 4: 1.0}
    assert elapsed < threshold[r], (
        f"r={r}, n={n}: Möbius method took {elapsed*1000:.1f} ms, "
        f"expected < {threshold[r]*1000:.0f} ms"
    )


# ----------------------------------------------------------------------
# Deterministic regression: a fixed input produces a known output
# ----------------------------------------------------------------------


def test_regression_diatonic_triad_pair():
    """Diatonic triads fixed input — output must match a frozen value."""
    # C major triad vs D minor triad in 12-TET
    p_A = np.array([0., 400., 700.])
    w_A = np.array([1., 1., 1.])
    p_B = np.array([200., 500., 900.])
    w_B = np.array([1., 1., 1.])
    sigma = 12.0
    r = 2
    cos = cos_sim_orbit_abs(p_A, w_A, p_B, w_B, sigma, r, False, 1200)
    # The frozen value is computed on the toolbox v2.1 path.
    T_A = build_exp_tens(p_A, w_A, sigma, r, False, False, 1200, verbose=False)
    T_B = build_exp_tens(p_B, w_B, sigma, r, False, False, 1200, verbose=False)
    cos_toolbox = float(cos_sim_exp_tens(T_A, T_B))
    assert abs(cos - cos_toolbox) < 1e-12


if __name__ == "__main__":
    import subprocess
    subprocess.run(["pytest", __file__, "-v", "--tb=short"], check=False)
