"""Tests for the factored strategy in *periodic* relative-mode evaluation.

The circular variance/mean block reduction survives wrapping only when the
truncation window (and each query's position span) fits within half the
period; there the factored strategy tabulates the circular S_1..S_r once
and reads them back with a wrapped quintic stencil, removing the K factor
from the per-node cost. Below that boundary (window > P/2) the reduction
wraps and the direct strategy is used instead.

These tests pin: (a) factored-periodic equals the direct strategy to the
truncation floor in the valid regime, across r and query spread; (b) the
wrapped read-back is genuinely periodic; (c) forcing factored on a
too-small circle is rejected; (d) the auto gate stays exact.
"""
import numpy as np
import pytest

import mpt
from mpt._mobius import _lagrange6_circular, _lagrange6_uniform, eval_orbit_rel

mpt.set_default(show_hints=False)


def _sources(P, K=400, seed=0):
    rng = np.random.default_rng(seed)
    p = np.sort(rng.uniform(0.0, P, K))
    w = rng.uniform(0.5, 1.5, K)
    return p, w


@pytest.mark.parametrize("r", [2, 3, 4])
@pytest.mark.parametrize("P_over_sigma", [20.0, 40.0, 100.0])
def test_factored_periodic_matches_direct(r, P_over_sigma):
    sigma = 2.0
    P = P_over_sigma * sigma
    p, w = _sources(P, K=400, seed=r)
    rng = np.random.default_rng(100 + r)
    x_rel = rng.uniform(-6.0, 6.0, (r - 1, 5))
    vf = eval_orbit_rel(p, w, sigma, r, x_rel, is_per=True, period=P,
                        factored=True, truncation_sigmas=6.0)
    vd = eval_orbit_rel(p, w, sigma, r, x_rel, is_per=True, period=P,
                        factored=False, truncation_sigmas=6.0)
    rel = np.max(np.abs(vf - vd) / np.maximum(np.abs(vd), 1e-300))
    assert rel < 1e-10


def test_factored_periodic_large_query_spread():
    # Query span up to ~0.4 P is fine once the window fits in half the circle.
    sigma, P, r = 2.0, 400.0, 3
    p, w = _sources(P, K=500, seed=11)
    x_rel = np.vstack([np.full(4, 0.20 * P), np.full(4, -0.15 * P)])  # span 0.35P
    vf = eval_orbit_rel(p, w, sigma, r, x_rel, is_per=True, period=P,
                        factored=True, truncation_sigmas=6.0)
    vd = eval_orbit_rel(p, w, sigma, r, x_rel, is_per=True, period=P,
                        factored=False, truncation_sigmas=6.0)
    assert np.max(np.abs(vf - vd) / np.maximum(np.abs(vd), 1e-300)) < 1e-9


def test_factored_true_rejected_when_circle_too_small():
    # Window (sqrt(2)*6*sigma) exceeds P/2: reduction would wrap.
    sigma, r = 2.0, 3
    P = 10.0 * sigma  # P/2 = 10 < sqrt(2)*6*2 ~ 17
    p, w = _sources(P, K=200, seed=5)
    x_rel = np.zeros((r - 1, 3))
    with pytest.raises(ValueError, match="periodic"):
        eval_orbit_rel(p, w, sigma, r, x_rel, is_per=True, period=P,
                       factored=True, truncation_sigmas=6.0)


def test_auto_gate_periodic_matches_direct():
    sigma, P, r = 2.0, 300.0, 3
    p, w = _sources(P, K=900, seed=9)
    rng = np.random.default_rng(21)
    x_rel = rng.uniform(-5.0, 5.0, (r - 1, 25))   # batch + large K -> factored
    va = eval_orbit_rel(p, w, sigma, r, x_rel, is_per=True, period=P,
                        truncation_sigmas=6.0)
    vd = eval_orbit_rel(p, w, sigma, r, x_rel, is_per=True, period=P,
                        factored=False, truncation_sigmas=6.0)
    assert np.max(np.abs(va - vd) / np.maximum(np.abs(vd), 1e-300)) < 1e-9


def test_lagrange6_circular_is_periodic():
    # A band-limited periodic function is interpolated near-exactly, and the
    # wrapped stencil makes the read-back continuous across the seam.
    n = 60
    P = 2.0 * np.pi
    h = P / n
    grid = h * np.arange(n)
    y = np.sin(grid) + 0.3 * np.cos(2 * grid)
    q = np.array([0.01, P - 0.01, -0.02, P + 0.02, 1.234])
    got = _lagrange6_circular(y, 0.0, h, q)
    exact = np.sin(q) + 0.3 * np.cos(2 * q)
    assert np.max(np.abs(got - exact)) < 1e-6
    # seam continuity: value just below P equals value just above 0 (mod P)
    a = _lagrange6_circular(y, 0.0, h, np.array([P - 1e-9]))
    b = _lagrange6_circular(y, 0.0, h, np.array([1e-9]))
    assert abs(a - b) < 1e-6
