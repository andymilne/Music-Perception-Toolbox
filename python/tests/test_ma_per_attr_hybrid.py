"""Tests for ragged-event handling in ``_ma_per_attr_inner_matrix``.

Mirror of MATLAB ``tests/test_ma_per_attr_hybrid.m``. Every event flows
through the vectorised batched orbit with zero-weight padding for NaN
entries; accuracy is governed by ``truncationSigmas`` rather than by how
close K_eff is to r, so no size-based partition is applied. The
direct-enumeration reference
``tests/references/mobius_ip_reference.inner_product_direct_abs`` is the
comparison point.

Tests cover:

- ``inner_product_direct_abs`` standalone correctness against a
  hand-rolled centres-array IP.
- NaN-tolerance: NaN-padded input gives the same result as NaN-stripped
  input.
- ``K_eff < r`` returns 0.
- Ragged events at every K_eff, including K_eff = r: the batched matrix
  matches a per-pair direct-enum reference.
- Mixed K_eff via cos_sim_exp_tens: Möbius and Bulger agree.
- r=1 ragged: still matches pairwise (regression check on the
  unchanged path).
"""

import numpy as np

from mpt._defaults import accuracy_floor_context
import pytest

from mpt.tensor import (
    build_exp_tens,
    cos_sim_exp_tens,
    _ma_per_attr_inner_matrix,
)
from tests.references.mobius_ip_reference import (
    build_ordered_r_tuples,
    inner_product_direct_abs,
)


# ----------------------------------------------------------------------
# inner_product_direct_abs standalone
# ----------------------------------------------------------------------


def test_inner_product_direct_matches_centres_array_ip():
    """Direct enumeration matches a hand-rolled centres-array IP."""
    rng = np.random.default_rng(81)
    p_x = np.sort(2000 * rng.random(5))
    w_x = np.ones(5)
    p_y = np.sort(2000 * rng.random(5))
    w_y = np.ones(5)
    sigma = 30.0
    r = 3

    ip_direct = inner_product_direct_abs(
        p_x, w_x, p_y, w_y, sigma, r, False, 0.0,
    )

    # Hand-rolled reference via the ordered-tuple builder.
    U_x, wJ_x = build_ordered_r_tuples(p_x, w_x, r)
    U_y, wJ_y = build_ordered_r_tuples(p_y, w_y, r)
    diffs = U_x[:, :, None] - U_y[:, None, :]
    Q = np.sum(diffs ** 2, axis=0)
    K_mat = np.exp(-Q / (4 * sigma ** 2))
    ip_ref = float(
        (sigma * np.sqrt(np.pi)) ** r
        * np.einsum('i,ij,j->', wJ_x, K_mat, wJ_y)
    )

    assert abs(ip_direct - ip_ref) < 1e-12 * abs(ip_ref)


def test_inner_product_direct_drops_nan_per_side():
    """NaN-padded input is dropped per side before enumeration."""
    rng = np.random.default_rng(81)
    p_x = np.sort(2000 * rng.random(5))
    w_x = np.ones(5)
    p_y = np.sort(2000 * rng.random(5))
    w_y = np.ones(5)
    sigma = 30.0
    r = 3

    ip_clean = inner_product_direct_abs(
        p_x, w_x, p_y, w_y, sigma, r, False, 0.0,
    )
    p_x_nan = np.concatenate([p_x, [np.nan, np.nan]])
    w_x_nan = np.concatenate([w_x, [np.nan, np.nan]])
    ip_nan = inner_product_direct_abs(
        p_x_nan, w_x_nan, p_y, w_y, sigma, r, False, 0.0,
    )
    assert abs(ip_nan - ip_clean) < 1e-12 * abs(ip_clean)


def test_inner_product_direct_returns_zero_when_keff_below_r():
    """K_eff < r returns 0 (no r-tuple can be formed)."""
    ip = inner_product_direct_abs(
        np.array([0.0, 1.0]), np.array([1.0, 1.0]),
        np.array([0.0, 1.0]), np.array([1.0, 1.0]),
        30.0, 3, False, 0.0,
    )
    assert ip == 0.0


# ----------------------------------------------------------------------
# Hybrid coverage cases
# ----------------------------------------------------------------------


def test_ragged_matches_direct_enum_reference():
    """Ragged events well above r: the batched matrix equals a per-pair
    direct-enum reference (within FP)."""
    rng = np.random.default_rng(83)
    N = 4
    K = 6
    P = np.sort(rng.uniform(0, 2000, (K, N)), axis=0)
    W = np.ones_like(P)
    sigma = 30.0
    r = 3

    I_batched = _ma_per_attr_inner_matrix(
        P, W, P, W, sigma, r, False, False, 0.0,
    )
    I_ref = np.empty((N, N))
    for nx in range(N):
        for ny in range(N):
            I_ref[nx, ny] = inner_product_direct_abs(
                P[:, nx], W[:, nx], P[:, ny], W[:, ny],
                sigma, r, False, 0.0,
            )
    assert np.max(np.abs(I_batched - I_ref)) < 1e-10 * np.max(np.abs(I_ref))


def test_k_equal_r_matches_direct_enum():
    """K_eff = r on every event: the batched matrix equals direct-enum on
    every pair (the alternating sum at its shortest)."""
    P = np.array([[0.0, 100.0],
                  [4.0, 200.0],
                  [7.0, 300.0]])         # (3, 2), every K_eff = 3
    W = np.ones_like(P)
    # Under the default (inf) truncation -- which resolves to the 1e-12
    # accuracy floor -- the batched matrix and the direct-enum reference
    # can drop marginally different far-tail contributions and diverge at
    # ~1e-12, above this tolerance. Widen the floor so the two are
    # compared exhaustively, as the MATLAB twin does.
    with accuracy_floor_context(1e-300):
        sigma = 30.0
        r = 3

        I_batched = _ma_per_attr_inner_matrix(
            P, W, P, W, sigma, r, False, False, 0.0,
        )
        I_ref = np.empty((2, 2))
        for nx in range(2):
            for ny in range(2):
                I_ref[nx, ny] = inner_product_direct_abs(
                    P[:, nx], W[:, nx], P[:, ny], W[:, ny],
                    sigma, r, False, 0.0,
                )
        # Judge the agreement by absolute error on the scale the inner
        # product lives on. Entries of this matrix span sixteen orders
        # of magnitude: a relative tolerance would be dominated by the
        # near-zero cross terms, which contribute nothing at the scale
        # of the matrix.
        np.testing.assert_allclose(
            I_batched, I_ref, rtol=0.0,
            atol=1e-13 * float(np.max(np.abs(I_ref))))


def test_mixed_k_eff_cossim_orbit_matches_pairwise():
    """Mixed K_eff: cos_sim_exp_tens with method='mobius' agrees
    with method='bulger' on a deliberately mixed case (K_eff = 3,
    6, 4 across three events; r = 3).
    """
    P = np.array([[10.0, 100.0, 500.0],
                  [30.0, 200.0, 600.0],
                  [50.0, 300.0, 700.0],
                  [np.nan, 400.0, 800.0],
                  [np.nan, 500.0, np.nan],
                  [np.nan, 600.0, np.nan]])  # (6, 3)
    W = np.where(np.isnan(P), np.nan, 1.0)
    sigma = 30.0
    r = 3

    dx = build_exp_tens([P], [W], [sigma], [r], 
                        [False], [False], [0.0], verbose=False)
    dy = build_exp_tens([P], [W], [sigma], [r], 
                        [False], [False], [0.0], verbose=False)
    s_orbit = cos_sim_exp_tens(dx, dy, method='mobius', verbose=False)
    s_pw = cos_sim_exp_tens(dx, dy, method='bulger', verbose=False)
    assert abs(s_orbit - s_pw) < 1e-8


def test_r1_ragged_orbit_matches_pairwise():
    """r=1 ragged still routes through the unchanged zero-pad path."""
    P = np.array([[10.0, 100.0, 200.0],
                  [30.0, np.nan, 400.0],
                  [50.0, np.nan, np.nan]])
    W = np.where(np.isnan(P), np.nan, 1.0)
    sigma = 30.0

    dx = build_exp_tens([P], [W], [sigma], [1], 
                        [False], [False], [0.0], verbose=False)
    dy = build_exp_tens([P], [W], [sigma], [1], 
                        [False], [False], [0.0], verbose=False)
    s_orbit = cos_sim_exp_tens(dx, dy, method='mobius', verbose=False)
    s_pw = cos_sim_exp_tens(dx, dy, method='bulger', verbose=False)
    assert abs(s_orbit - s_pw) < 1e-10


# ----------------------------------------------------------------------
# Ragged K_eff through the public API
# ----------------------------------------------------------------------


class TestRaggedDispatch:
    """Variable-K_eff workloads through the public API."""

    def test_mixed_K_eff_matches_pairwise_at_machine_precision(self):
        """The Möbius method on a ragged density must match Bulger's
        method (which enumerates each event's own tuples) to numerical
        precision."""
        rng = np.random.default_rng(7)
        N = 12
        K_max = 6
        r = 3
        Px = np.full((K_max, N), np.nan)
        Wx = np.full((K_max, N), np.nan)
        # 4 events at K_eff=3 (= r), 4 at K_eff=4, 4 at K_eff=6.
        K_distribution = [3, 3, 3, 3, 4, 4, 4, 4, 6, 6, 6, 6]
        for n, K_eff in enumerate(K_distribution):
            Px[:K_eff, n] = rng.uniform(0, 1200, K_eff)
            Wx[:K_eff, n] = 1.0

        sigma = 30.0
        dens = build_exp_tens([Px], [Wx], [sigma], [r], 
                              [False], [False], [0.0], verbose=False)
        s_orbit = cos_sim_exp_tens(dens, dens, method='mobius',
                                    verbose=False)
        s_pw = cos_sim_exp_tens(dens, dens, method='bulger',
                                 verbose=False)
        assert abs(s_orbit - s_pw) < 1e-12
