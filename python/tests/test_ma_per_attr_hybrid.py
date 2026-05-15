"""Tests for the safe/unsafe hybrid in ``_ma_per_attr_inner_matrix``.

Mirror of MATLAB ``tests/test_ma_per_attr_hybrid.m``. Strategy:

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
    _batched_direct_enum_abs_sa,
    _pack_nan_top,
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
    # v2.2.0 used a Python double-loop over pairs (one direct-enum
    # call per (nx, ny)); v2.2.x replaces it with a single vectorised
    # tensor contraction per (K_eff_x, K_eff_y) sub-block. The two
    # produce mathematically identical results but accumulate Q sums
    # in a different order, so individual entries can differ by ~1 ULP.
    np.testing.assert_allclose(I_hybrid, I_ref, atol=0.0, rtol=1e-13)


def test_mixed_safe_unsafe_cossim_orbit_matches_pairwise():
    """Mixed safe/unsafe: cos_sim_exp_tens with method='mobius' agrees
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

    dx = build_exp_tens([P], [W], [sigma], [r], None,
                        [False], [False], [0.0], verbose=False)
    dy = build_exp_tens([P], [W], [sigma], [r], None,
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

    dx = build_exp_tens([P], [W], [sigma], [1], None,
                        [False], [False], [0.0], verbose=False)
    dy = build_exp_tens([P], [W], [sigma], [1], None,
                        [False], [False], [0.0], verbose=False)
    s_orbit = cos_sim_exp_tens(dx, dy, method='mobius', verbose=False)
    s_pw = cos_sim_exp_tens(dx, dy, method='bulger', verbose=False)
    assert abs(s_orbit - s_pw) < 1e-10


# ----------------------------------------------------------------------
# v2.2.x: K-grouped batched direct enumeration (Item 3a)
# ----------------------------------------------------------------------


class TestBatchedDirectEnum:
    """The batched primitive must match a per-pair direct-enum reference
    to numerical precision, across r, K_x, K_y, periodic / non-periodic,
    and various event-count combinations.

    Operation ordering differs (einsum vs per-pair einsum + accumulator),
    so bit-equality is too strict; rtol=1e-13 is the appropriate band
    for r-tuple direct enumeration in float64.
    """

    @pytest.mark.parametrize("r", [1, 2, 3])
    @pytest.mark.parametrize("K_x,K_y", [(3, 3), (4, 4), (3, 5), (5, 3)])
    def test_matches_per_pair_direct_enum_nonper(self, r, K_x, K_y):
        if K_x < r or K_y < r:
            return
        rng = np.random.default_rng(0)
        N_x, N_y = 5, 7
        Px = rng.uniform(0, 1000, (K_x, N_x))
        Wx = np.ones((K_x, N_x))
        Py = rng.uniform(0, 1000, (K_y, N_y))
        Wy = np.ones((K_y, N_y))
        sigma = 25.0

        I_batched = _batched_direct_enum_abs_sa(
            Px, Wx, Py, Wy, sigma, r, False, 0.0,
        )
        I_ref = np.empty((N_x, N_y))
        for nx in range(N_x):
            for ny in range(N_y):
                I_ref[nx, ny] = _inner_product_direct_abs_sa(
                    Px[:, nx], Wx[:, nx], Py[:, ny], Wy[:, ny],
                    sigma, r, False, 0.0,
                )
        np.testing.assert_allclose(I_batched, I_ref, atol=0.0, rtol=1e-13)

    @pytest.mark.parametrize("r", [1, 2, 3])
    def test_matches_per_pair_direct_enum_periodic(self, r):
        K = max(3, r)
        rng = np.random.default_rng(1)
        N_x, N_y = 4, 6
        Px = rng.uniform(0, 1200, (K, N_x))
        Wx = np.ones((K, N_x))
        Py = rng.uniform(0, 1200, (K, N_y))
        Wy = np.ones((K, N_y))
        sigma = 80.0
        period = 1200.0

        I_batched = _batched_direct_enum_abs_sa(
            Px, Wx, Py, Wy, sigma, r, True, period,
        )
        I_ref = np.empty((N_x, N_y))
        for nx in range(N_x):
            for ny in range(N_y):
                I_ref[nx, ny] = _inner_product_direct_abs_sa(
                    Px[:, nx], Wx[:, nx], Py[:, ny], Wy[:, ny],
                    sigma, r, True, period,
                )
        np.testing.assert_allclose(I_batched, I_ref, atol=0.0, rtol=1e-13)

    def test_zero_when_K_below_r(self):
        """K_x < r => IP = 0 (no r-tuples to enumerate)."""
        Px = np.array([[0.0], [10.0]])
        Wx = np.ones_like(Px)
        Py = np.zeros((3, 1))
        Wy = np.ones_like(Py)
        I = _batched_direct_enum_abs_sa(
            Px, Wx, Py, Wy, 20.0, r=3, is_per=False, period=0.0,
        )
        assert I.shape == (1, 1)
        assert I[0, 0] == 0.0


class TestKGroupedDispatch:
    """Variable-K_eff workloads — the optimization target."""

    def test_mixed_K_eff_matches_v21_pairwise_at_machine_precision(self):
        """Hybrid Möbius method with K-grouped direct enum must match the
        v2.1 Bulger's method (which has no NaN issues at any K_eff) to
        numerical precision."""
        rng = np.random.default_rng(7)
        N = 12
        K_max = 6
        r = 3
        Px = np.full((K_max, N), np.nan)
        Wx = np.full((K_max, N), np.nan)
        # 4 events at K_eff=3 (unsafe at r=3), 4 at K_eff=4 (unsafe),
        # 4 at K_eff=6 (safe at r=3).
        K_distribution = [3, 3, 3, 3, 4, 4, 4, 4, 6, 6, 6, 6]
        for n, K_eff in enumerate(K_distribution):
            Px[:K_eff, n] = rng.uniform(0, 1200, K_eff)
            Wx[:K_eff, n] = 1.0

        sigma = 30.0
        dens = build_exp_tens([Px], [Wx], [sigma], [r], None,
                              [False], [False], [0.0], verbose=False)
        s_orbit = cos_sim_exp_tens(dens, dens, method='mobius',
                                    verbose=False)
        s_pw = cos_sim_exp_tens(dens, dens, method='bulger',
                                 verbose=False)
        assert abs(s_orbit - s_pw) < 1e-12


class TestPackNanTop:
    """The packing helper must place valid slots at the top of each
    column regardless of the user's NaN pattern."""

    def test_already_top_packed_is_preserved(self):
        P = np.array([[1.0, 2.0],
                      [3.0, 4.0],
                      [np.nan, 5.0]])
        W = np.where(np.isnan(P), np.nan, 1.0)
        Pp, Wp = _pack_nan_top(P, W)
        np.testing.assert_array_equal(Pp[:2, 0], [1.0, 3.0])
        np.testing.assert_array_equal(Pp[:3, 1], [2.0, 4.0, 5.0])

    def test_interleaved_nan_gets_packed(self):
        P = np.array([[1.0, 2.0],
                      [np.nan, 4.0],
                      [3.0, np.nan]])
        W = np.where(np.isnan(P), np.nan, 1.0)
        Pp, Wp = _pack_nan_top(P, W)
        np.testing.assert_array_equal(Pp[:2, 0], [1.0, 3.0])
        np.testing.assert_array_equal(Pp[:2, 1], [2.0, 4.0])
        assert np.isnan(Pp[2, 0]) and np.isnan(Pp[2, 1])

