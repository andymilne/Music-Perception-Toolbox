"""Tests for the spectral (Fourier) branch of the relative-mode
per-attribute inner matrix.

The branch replaces the translation grid with a mode sum, building one
spectrum per event and forming their Gram matrix (see
``_spectral_rel_inner_matrix``). These tests pin its agreement with the
grid path it replaces, its stand-down conditions, and its edge
behaviour. Engagement economics are not pinned: the branch is under
cost calibration.
"""

import numpy as np

import mpt
import mpt._tensor.cosine as CO
import mpt._tensor._mobius_inner as _mobius_inner
from mpt._tensor._mobius_inner import (_rel_inner_batched, _spectral_rel_inner_matrix)


def _both(Px, Wx, Py, Wy, sigma, r, is_per, period):
    args = (Px, Wx, Py, Wy, sigma, r, is_per, period)
    got = np.asarray(_rel_inner_batched(
        *args, truncation_sigmas=np.inf))
    _mobius_inner._SPECTRAL_IP_ENABLED = False
    try:
        ref = np.asarray(_rel_inner_batched(
            *args, truncation_sigmas=np.inf))
    finally:
        _mobius_inner._SPECTRAL_IP_ENABLED = True
    return got, ref


def test_spectral_matches_grid_path_r2_to_r4():
    """Both modes, both orders: the spectral branch reproduces the
    grid path's matrix to well within the accuracy floor."""
    rng = np.random.default_rng(401)
    for r in (2, 3, 4):
        for is_per in (True, False):
            K, NX, NY, P, sigma = 10, 3, 3, 1200.0, 40.0
            Px = np.sort(rng.uniform(0, P, (K, NX)), axis=0)
            Wx = 0.5 + rng.random((K, NX))
            Py = np.sort(rng.uniform(0, P, (K, NY)), axis=0)
            Wy = 0.5 + rng.random((K, NY))
            got, ref = _both(Px, Wx, Py, Wy, sigma, r, is_per,
                             P if is_per else 0.0)
            assert (np.max(np.abs(got - ref))
                    <= 1e-10 * np.max(np.abs(ref)))


def test_spectral_stands_down_for_cancellation_ratio():
    """A cancellation-ratio request bypasses the branch, since the
    spectral form has no per-node terms matching that diagnostic."""
    rng = np.random.default_rng(7)
    K, P, sigma = 10, 1200.0, 40.0
    Px = np.sort(rng.uniform(0, P, (K, 3)), axis=0)
    Wx = 0.5 + rng.random((K, 3))
    calls = {"n": 0}
    orig = _mobius_inner._spectral_rel_inner_matrix

    def spy(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    _mobius_inner._spectral_rel_inner_matrix = spy
    try:
        _rel_inner_batched(
            Px, Wx, Px, Wx, sigma, 3, True, P,
            return_cancellation_ratio=True, truncation_sigmas=np.inf)
        assert calls["n"] == 0
        _rel_inner_batched(
            Px, Wx, Px, Wx, sigma, 3, True, P,
            truncation_sigmas=np.inf)
        assert calls["n"] == 1
    finally:
        _mobius_inner._spectral_rel_inner_matrix = orig


def test_spectral_declines_oversized_grid():
    """The mode grid is (r-1)-dimensional; when it would exceed the
    point cap the helper returns None so the caller falls through."""
    rng = np.random.default_rng(11)
    K, P = 8, 1200.0
    Px = np.sort(rng.uniform(0, P, (K, 2)), axis=0)
    Wx = np.ones((K, 2))
    # A small sigma at r = 4 makes the mode count large enough to trip
    # the cap; the helper must decline rather than allocate.
    out = _spectral_rel_inner_matrix(Px, Wx, Px, Wx, 1.0, 4, True, P)
    assert out is None
    # The public path still returns a finite matrix for that case.
    full = np.asarray(_rel_inner_batched(
        Px, Wx, Px, Wx, 1.0, 4, True, P, truncation_sigmas=np.inf))
    assert np.all(np.isfinite(full))


def test_spectral_zero_weight_events():
    """Zero-weight (padded) events contribute nothing, matching the
    grid path."""
    rng = np.random.default_rng(19)
    K, P, sigma = 10, 1200.0, 40.0
    Px = np.sort(rng.uniform(0, P, (K, 3)), axis=0)
    Wx = 0.5 + rng.random((K, 3))
    Wx[:, 2] = 0.0
    got, ref = _both(Px, Wx, Px, Wx, sigma, 3, True, P)
    assert np.max(np.abs(got[2, :])) < 1e-8 * np.max(np.abs(ref))
    assert np.max(np.abs(got - ref)) <= 1e-10 * np.max(np.abs(ref))


def test_spectral_symmetry_and_positivity():
    """Self-matrices are symmetric with positive diagonal."""
    rng = np.random.default_rng(23)
    K, P, sigma = 12, 1200.0, 40.0
    Px = np.sort(rng.uniform(0, P, (K, 4)), axis=0)
    Wx = 0.5 + rng.random((K, 4))
    G = np.asarray(_rel_inner_batched(
        Px, Wx, Px, Wx, sigma, 3, True, P, truncation_sigmas=np.inf))
    assert np.max(np.abs(G - G.T)) <= 1e-10 * np.max(np.abs(G))
    assert np.all(np.diag(G) > 0.0)


def test_spectral_cost_gate_declines_unprofitable_shapes():
    """The mode grid is (r-1)-dimensional while the grid path costs
    K^2 per event pair, so at small sigma/P with few values and few
    events the branch must decline. Calibrated against measured wall
    times; here we pin the decision, not the timing."""
    rng = np.random.default_rng(29)
    P = 1200.0
    # Small sigma/P, 4 values, one event: measured ~12x slower if taken.
    Px = np.sort(rng.uniform(0, P, (4, 1)), axis=0)
    Wx = np.ones((4, 1))
    assert _spectral_rel_inner_matrix(
        Px, Wx, Px, Wx, 3.0, 3, True, P) is None
    # The same sigma with many values is profitable and must be taken.
    Px2 = np.sort(rng.uniform(0, P, (40, 1)), axis=0)
    Wx2 = np.ones((40, 1))
    assert _spectral_rel_inner_matrix(
        Px2, Wx2, Px2, Wx2, 3.0, 3, True, P) is not None
    # Values stay correct either way.
    got, ref = _both(Px, Wx, Px, Wx, 3.0, 3, True, P)
    assert np.max(np.abs(got - ref)) <= 1e-10 * np.max(np.abs(ref))
