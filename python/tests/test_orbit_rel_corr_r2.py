"""Tests for the spectral (Fourier) strategy of eval_orbit_rel.

The worker evaluates every partition of the Möbius sum as the
total-mode-zero coefficient of a product of closed-form per-block
spectra (see ``_eval_orbit_rel_fourier``). These tests pin its
agreement with the direct strategy to the truncation floor across
r = 2..5, chunk-invariance to machine precision, the gate's exclusion
rules, and edge behaviour. Engagement economics are intentionally not
pinned here: the strategy is under active cost calibration.
"""

import numpy as np

import mpt
import mpt._mobius as M
from mpt._mobius import eval_orbit_rel, _eval_orbit_rel_fourier


def _floor(ts):
    from mpt._defaults import truncation_floor
    return truncation_floor(ts)


def test_fourier_worker_matches_direct_r2_to_r5():
    rng = np.random.default_rng(31)
    kmin = {2: 12, 3: 16, 4: 28, 5: 34}
    for r in (2, 3, 4, 5):
        for ts in (6.0, np.inf):
            for trial in range(2):
                K = int(rng.integers(kmin[r], kmin[r] + 10))
                per = bool(trial % 2)
                P = 1200.0 if per else 0.0
                sigma = float(rng.uniform(8, 14))
                p = np.sort(rng.uniform(0, 1200, K))
                w = 0.5 + rng.random(K)
                x = rng.uniform(-120, 120, (r - 1, 15))
                va = np.asarray(_eval_orbit_rel_fourier(
                    p, w, sigma, r, x, per, P, ts)).ravel()
                vd = np.asarray(eval_orbit_rel(
                    p, w, sigma, r, x, is_per=per, period=P,
                    truncation_sigmas=ts, factored=False)).ravel()
                peak = float(np.max(np.abs(vd)))
                assert np.max(np.abs(va - vd)) <= 10.0 * _floor(ts) * peak


def test_fourier_chunk_invariance():
    rng = np.random.default_rng(7)
    p = np.sort(rng.uniform(0, 1200, 20))
    w = 0.5 + rng.random(20)
    x = rng.uniform(-120, 120, (3, 40))
    v1 = np.asarray(_eval_orbit_rel_fourier(
        p, w, 12.0, 4, x, False, 0.0, 6.0))
    mpt.set_default(kernel_chunk_bytes=2_000_000)
    try:
        v2 = np.asarray(_eval_orbit_rel_fourier(
            p, w, 12.0, 4, x, False, 0.0, 6.0))
    finally:
        mpt.set_default(kernel_chunk_bytes='auto')
    # Batched-FFT internals differ by batch shape at ULP level, so the
    # contract is machine-precision equivalence, not bitwise identity.
    peak = float(np.max(np.abs(v1)))
    assert np.max(np.abs(v1 - v2)) <= 1e-13 * peak


def test_fourier_gate_exclusions():
    """Forced strategies, ratio requests, and the periodic
    principal-image window keep the spectral worker out."""
    rng = np.random.default_rng(3)
    p = np.sort(rng.uniform(0, 1200, 20))
    w = np.ones(20)
    x = rng.uniform(-100, 100, (2, 50))
    kw = dict(is_per=False, period=0.0, truncation_sigmas=6.0)
    orig = M._eval_orbit_rel_fourier
    hits = {"n": 0}

    def spy(*a, **k):
        hits["n"] += 1
        return orig(*a, **k)

    M._eval_orbit_rel_fourier = spy
    try:
        eval_orbit_rel(p, w, 15.0, 3, x, factored=True, **kw)
        eval_orbit_rel(p, w, 15.0, 3, x, factored=False, **kw)
        eval_orbit_rel(p, w, 15.0, 3, x,
                       return_cancellation_ratio=True, **kw)
        eval_orbit_rel(np.sort(rng.uniform(0, 240, 15)), np.ones(15),
                       30.0, 2, rng.uniform(-40, 40, (1, 30)),
                       is_per=True, period=240.0, truncation_sigmas=6.0)
    finally:
        M._eval_orbit_rel_fourier = orig
    assert hits["n"] == 0


def test_fourier_worker_edges():
    """K = 1 (no distinct pairs) evaluates to ~0; periodic values are
    invariant under whole-period query shifts."""
    v1 = np.asarray(_eval_orbit_rel_fourier(
        np.array([600.0]), np.array([1.0]), 15.0, 2,
        np.array([[0.0, 30.0]]), False, 0.0, 6.0)).ravel()
    assert np.max(np.abs(v1)) < 1e-12
    rng = np.random.default_rng(11)
    p = np.sort(rng.uniform(0, 1200, 18))
    w = 0.5 + rng.random(18)
    x = rng.uniform(-100, 100, (1, 30))
    a = np.asarray(_eval_orbit_rel_fourier(
        p, w, 20.0, 2, x, True, 1200.0, 6.0)).ravel()
    b = np.asarray(_eval_orbit_rel_fourier(
        p, w, 20.0, 2, x + 3 * 1200.0, True, 1200.0, 6.0)).ravel()
    np.testing.assert_allclose(a, b, rtol=0.0,
                               atol=1e-12 * np.max(np.abs(a)))
