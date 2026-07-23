"""Tests for the r = 2 cross-correlation strategy of eval_orbit_rel.

At r = 2 the relative evaluation collapses to one tabulated
autocorrelation plus an analytic diagonal term (see
``_eval_orbit_rel_corr_r2``). These tests pin the strategy's agreement
with the direct and factored strategies to the truncation floor, its
engagement conditions, and its edge behaviour.
"""

import numpy as np
import pytest

import mpt
import mpt._mobius as M
from mpt._mobius import eval_orbit_rel, _eval_orbit_rel_corr_r2


def _floor(ts):
    from mpt._defaults import truncation_floor
    return truncation_floor(ts)


def test_corr_matches_direct_and_factored():
    """Auto (correlation) agrees with both forced strategies to within
    an order of the truncation floor, in both modes at both accuracy
    settings, at sizes where the worthwhile gate engages."""
    rng = np.random.default_rng(8)
    for trial in range(8):
        K = int(rng.integers(15, 40))
        per = bool(trial % 2)
        P = 1200.0 if per else 0.0
        sigma = float(rng.uniform(8, 30))
        ts = [6.0, np.inf][trial // 2 % 2]
        p = np.sort(rng.uniform(0, 1200, K))
        w = 0.5 + rng.random(K)
        x = rng.uniform(-200, 200, (1, 60))
        kw = dict(is_per=per, period=P, truncation_sigmas=ts)
        va = np.asarray(eval_orbit_rel(p, w, sigma, 2, x, **kw)).ravel()
        vd = np.asarray(eval_orbit_rel(
            p, w, sigma, 2, x, factored=False, **kw)).ravel()
        peak = float(np.max(np.abs(vd)))
        tol = 10.0 * _floor(ts) * peak
        assert np.max(np.abs(va - vd)) <= tol
        vf = np.asarray(eval_orbit_rel(
            p, w, sigma, 2, x, factored=True, **kw)).ravel()
        assert np.max(np.abs(va - vf)) <= tol


def test_corr_engagement_conditions():
    """The strategy engages only under auto selection without a
    cancellation-ratio request; forced strategies and ratio requests
    bypass it, and the module flag disables it."""
    rng = np.random.default_rng(3)
    p = np.sort(rng.uniform(0, 1200, 20))
    w = np.ones(20)
    x = rng.uniform(-100, 100, (1, 50))
    kw = dict(is_per=False, period=0.0, truncation_sigmas=6.0)

    orig = M._eval_orbit_rel_corr_r2
    hits = {"n": 0}

    def spy(*a, **k):
        hits["n"] += 1
        return orig(*a, **k)

    M._eval_orbit_rel_corr_r2 = spy
    try:
        eval_orbit_rel(p, w, 15.0, 2, x, **kw)
        assert hits["n"] == 1
        eval_orbit_rel(p, w, 15.0, 2, x, factored=True, **kw)
        eval_orbit_rel(p, w, 15.0, 2, x, factored=False, **kw)
        eval_orbit_rel(p, w, 15.0, 2, x,
                       return_cancellation_ratio=True, **kw)
        assert hits["n"] == 1
        M._CORR_R2_ENABLED = False
        v_off = np.asarray(eval_orbit_rel(p, w, 15.0, 2, x, **kw)).ravel()
        assert hits["n"] == 1
    finally:
        M._eval_orbit_rel_corr_r2 = orig
        M._CORR_R2_ENABLED = True
    # With the flag off, auto falls through to the normal gate (which
    # may pick either remaining strategy); the worker stays uncalled
    # (asserted above) and the values remain correct to the floor.
    v_dir = np.asarray(eval_orbit_rel(
        p, w, 15.0, 2, x, factored=False, **kw)).ravel()
    peak = float(np.max(np.abs(v_dir)))
    assert np.max(np.abs(v_off - v_dir)) <= 10.0 * _floor(6.0) * peak


def test_corr_worker_edges():
    """K = 1 (no distinct pairs) evaluates to zero at floating point,
    and queries far outside the source span return floor-level values,
    matching the Möbius cancellation character of the other
    strategies."""
    v1 = np.asarray(eval_orbit_rel(
        np.array([600.0]), np.array([1.0]), 15.0, 2,
        np.array([[0.0, 30.0]]), is_per=False, period=0.0)).ravel()
    assert np.max(np.abs(v1)) < 1e-12
    rng = np.random.default_rng(5)
    p = np.sort(rng.uniform(0, 1200, 15))
    vfar = np.asarray(eval_orbit_rel(
        p, np.ones(15), 15.0, 2, np.array([[5000.0, -5000.0]]),
        is_per=False, period=0.0)).ravel()
    assert np.max(np.abs(vfar)) < 1e-12


def test_corr_periodic_translation_invariance():
    """Periodic correlation values are invariant under shifting the
    queries by whole periods (the circular read-back wraps)."""
    rng = np.random.default_rng(11)
    K = 20
    P = 1200.0
    p = np.sort(rng.uniform(0, P, K))
    w = 0.5 + rng.random(K)
    x = rng.uniform(-200, 200, (1, 40))
    v1 = np.asarray(eval_orbit_rel(
        p, w, 20.0, 2, x, is_per=True, period=P)).ravel()
    v2 = np.asarray(eval_orbit_rel(
        p, w, 20.0, 2, x + 3.0 * P, is_per=True, period=P)).ravel()
    np.testing.assert_allclose(v1, v2, rtol=0.0,
                               atol=1e-12 * np.max(np.abs(v1)))


def test_corr_worker_direct_call():
    """The worker itself (bypassing the gate) matches the direct
    strategy at small K, where the gate would decline."""
    from mpt._mobius import _factored_target_eps, _factored_spp
    rng = np.random.default_rng(7)
    K = 6
    P = 1200.0
    sigma = 12.0
    p = np.sort(rng.uniform(0, P, K))
    w = 0.5 + rng.random(K)
    x = rng.uniform(-150, 150, (1, 25))
    eps = _factored_target_eps(6.0, None)
    spp = _factored_spp(eps)
    va = np.asarray(_eval_orbit_rel_corr_r2(
        p, w, sigma, x, True, P, 6.0, None, spp, 0.0, 0.0)).ravel()
    vd = np.asarray(eval_orbit_rel(
        p, w, sigma, 2, x, is_per=True, period=P,
        truncation_sigmas=6.0, factored=False)).ravel()
    peak = float(np.max(np.abs(vd)))
    assert np.max(np.abs(va - vd)) <= 10.0 * _floor(6.0) * peak
