"""Tests for the safe/unsafe hybrid in ``_ma_per_attr_inner_matrix``.

Mirror of MATLAB ``tests/v22/test_ma_per_attr_hybrid.m``. Strategy:

- Each event is classified as "safe" if ``K_eff - r >= 2`` (matches
  ``_ORBIT_K_MINUS_R_MIN``), "unsafe" otherwise.
- Safe-vs-safe pairs flow through the vectorised batched orbit with
  zero-pad within the safe group.
- Pairs involving any unsafe event flow through
  :func:`mpt.tensor._inner_product_direct_abs_sa` (direct r-tuple
  enumeration; no Möbius alternating sum, so no cancellation).

Tests cover:

- ``_inner_product_direct_abs_sa`` standalone correctness against a
  hand-rolled centres-array IP.
- NaN-tolerance: NaN-padded input gives the same result as NaN-stripped
  input.
- ``K_eff < r`` returns 0.
- All-safe ragged: hybrid output matches a per-pair direct-enum reference.
- All-unsafe: hybrid output matches per-pair direct-enum exactly.
- Mixed safe/unsafe via cos_sim_exp_tens: orbit and pairwise agree
  to 1e-8 (the killer test for the hybrid claim).
- r=1 ragged: still matches pairwise (regression check on the
  unchanged path).
"""

import numpy as np
import pytest

from mpt.tensor import (
    build_exp_tens,
    cos_sim_exp_tens,
    _inner_product_direct_abs_sa,
    _build_ordered_r_tuples,
    _ma_per_attr_inner_matrix,
)


# ----------------------------------------------------------------------
# _inner_product_direct_abs_sa standalone
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

    ip_direct = _inner_product_direct_abs_sa(
        p_x, w_x, p_y, w_y, sigma, r, False, 0.0,
    )

    # Hand-rolled reference via the ordered-tuple builder.
    U_x, wJ_x = _build_ordered_r_tuples(p_x, w_x, r)
    U_y, wJ_y = _build_ordered_r_tuples(p_y, w_y, r)
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

    ip_clean = _inner_product_direct_abs_sa(
        p_x, w_x, p_y, w_y, sigma, r, False, 0.0,
    )
    p_x_nan = np.concatenate([p_x, [np.nan, np.nan]])
    w_x_nan = np.concatenate([w_x, [np.nan, np.nan]])
    ip_nan = _inner_product_direct_abs_sa(
        p_x_nan, w_x_nan, p_y, w_y, sigma, r, False, 0.0,
    )
    assert abs(ip_nan - ip_clean) < 1e-12 * abs(ip_clean)


def test_inner_product_direct_returns_zero_when_keff_below_r():
    """K_eff < r returns 0 (no r-tuple can be formed)."""
    ip = _inner_product_direct_abs_sa(
        np.array([0.0, 1.0]), np.array([1.0, 1.0]),
        np.array([0.0, 1.0]), np.array([1.0, 1.0]),
        30.0, 3, False, 0.0,
    )
    assert ip == 0.0


# ----------------------------------------------------------------------
# Hybrid coverage cases
# ----------------------------------------------------------------------


def test_all_safe_ragged_matches_direct_enum_reference():
    """All-safe: hybrid output equals a per-pair direct-enum reference.

    With every event having ``K_eff - r >= 2``, the hybrid uses the
    vectorised orbit branch on the entire matrix. Comparing against
    a per-pair direct-enum reference proves the safe-orbit branch
    is mathematically equivalent (within FP).
    """
    rng = np.random.default_rng(83)
    N = 4
    K = 6
    P = np.sort(rng.uniform(0, 2000, (K, N)), axis=0)
    W = np.ones_like(P)
    sigma = 30.0
    r = 3

    I_hybrid = _ma_per_attr_inner_matrix(
        P, W, P, W, sigma, r, False, False, 0.0,
    )
    I_ref = np.empty((N, N))
    for nx in range(N):
        for ny in range(N):
            I_ref[nx, ny] = _inner_product_direct_abs_sa(
                P[:, nx], W[:, nx], P[:, ny], W[:, ny],
                sigma, r, False, 0.0,
            )
    assert np.max(np.abs(I_hybrid - I_ref)) < 1e-10 * np.max(np.abs(I_ref))


def test_all_unsafe_matches_direct_enum():
    """All-unsafe: hybrid output equals direct-enum on every pair."""
    P = np.array([[0.0, 100.0],
                  [4.0, 200.0],
                  [7.0, 300.0]])         # (3, 2), every K_eff = 3
    W = np.ones_like(P)
    sigma = 30.0
    r = 3

    I_hybrid = _ma_per_attr_inner_matrix(
        P, W, P, W, sigma, r, False, False, 0.0,
    )
    I_ref = np.empty((2, 2))
    for nx in range(2):
        for ny in range(2):
            I_ref[nx, ny] = _inner_product_direct_abs_sa(
                P[:, nx], W[:, nx], P[:, ny], W[:, ny],
                sigma, r, False, 0.0,
            )
    np.testing.assert_array_equal(I_hybrid, I_ref)


def test_mixed_safe_unsafe_cossim_orbit_matches_pairwise():
    """Mixed safe/unsafe: cos_sim_exp_tens with method='orbit' agrees
    with method='pairwise' on a deliberately mixed case (K_eff = 3,
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

    dx = build_exp_tens([P], [W], [sigma], [r], None,
                        [False], [False], [0.0], verbose=False)
    dy = build_exp_tens([P], [W], [sigma], [r], None,
                        [False], [False], [0.0], verbose=False)
    s_orbit = cos_sim_exp_tens(dx, dy, method='orbit', verbose=False)
    s_pw = cos_sim_exp_tens(dx, dy, method='pairwise', verbose=False)
    assert abs(s_orbit - s_pw) < 1e-8


def test_r1_ragged_orbit_matches_pairwise():
    """r=1 ragged still routes through the unchanged zero-pad path."""
    P = np.array([[10.0, 100.0, 200.0],
                  [30.0, np.nan, 400.0],
                  [50.0, np.nan, np.nan]])
    W = np.where(np.isnan(P), np.nan, 1.0)
    sigma = 30.0

    dx = build_exp_tens([P], [W], [sigma], [1], None,
                        [False], [False], [0.0], verbose=False)
    dy = build_exp_tens([P], [W], [sigma], [1], None,
                        [False], [False], [0.0], verbose=False)
    s_orbit = cos_sim_exp_tens(dx, dy, method='orbit', verbose=False)
    s_pw = cos_sim_exp_tens(dx, dy, method='pairwise', verbose=False)
    assert abs(s_orbit - s_pw) < 1e-10
