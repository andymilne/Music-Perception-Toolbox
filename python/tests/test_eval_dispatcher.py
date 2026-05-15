"""Tests for the v2.2 eval_exp_tens dispatcher (SA path).

The dispatcher routes between the v2.0 centres-array body and the
v2.2 orbit-Möbius point evaluator according to ``method`` and the
cost rules in :func:`mpt.tensor._select_sa_eval_method`. These tests
verify (a) that auto routing picks the right path under each rule,
(b) that the three explicit methods produce mutually consistent
output, and (c) that the precision and convention guards behave as
documented.
"""
import numpy as np
import pytest

from mpt.tensor import (
    build_exp_tens,
    eval_exp_tens,
    _select_sa_eval_method,
)


P = 1200.0
ATOL = 1e-12
RTOL = 1e-10


# -------------------------------------------------------------------
#  Routing rules
# -------------------------------------------------------------------


@pytest.mark.parametrize(
    "r,K,is_rel,is_per,expected",
    [
        # r=1: always centres.
        (1, 5, False, True, 'centres'),
        (1, 5, True, True, 'centres'),
        # r=2 with K<=8: centres (small-K rule).
        (2, 4, False, True, 'centres'),
        (2, 8, False, True, 'centres'),
        # r=2 with K>=9: orbit.
        (2, 9, False, True, 'mobius'),
        (2, 20, False, True, 'mobius'),
        # r=3 abs: orbit when K-r >= 2.
        (3, 5, False, True, 'mobius'),
        (3, 5, False, False, 'mobius'),
        # r=3 abs with K-r=1: centres (precision guard).
        (3, 4, False, True, 'centres'),
        # r=4 abs: orbit when K-r >= 2.
        (4, 6, False, True, 'mobius'),
        # r=9 (beyond shipped tables): centres.
        (9, 12, False, True, 'centres'),
        # Rel mode at any r: centres (cost rule).
        (3, 6, True, True, 'centres'),
        (4, 7, True, False, 'centres'),
    ],
)
def test_select_sa_eval_method(r, K, is_rel, is_per, expected):
    chosen = _select_sa_eval_method(
        r=r, K=K, n_q=10, is_rel=is_rel, is_per=is_per,
        sigma_over_P=0.025, user_method='auto',
    )
    assert chosen == expected


def test_select_sa_eval_method_user_override():
    """Explicit method= bypasses cost rules."""
    # Force orbit even at r=2 K=5 (would auto to centres).
    assert _select_sa_eval_method(
        r=2, K=5, n_q=10, is_rel=False, is_per=True,
        sigma_over_P=0.025, user_method='mobius',
    ) == 'mobius'
    # Force centres at r=4 K=10 (would auto to orbit).
    assert _select_sa_eval_method(
        r=4, K=10, n_q=10, is_rel=False, is_per=True,
        sigma_over_P=0.025, user_method='centres',
    ) == 'centres'
    # 'direct' is treated as a synonym for 'centres'.
    assert _select_sa_eval_method(
        r=3, K=8, n_q=10, is_rel=False, is_per=True,
        sigma_over_P=0.025, user_method='direct',
    ) == 'centres'


def test_select_sa_eval_method_rejects_unknown():
    with pytest.raises(ValueError, match="method must be"):
        _select_sa_eval_method(
            r=3, K=8, n_q=10, is_rel=False, is_per=True,
            sigma_over_P=0.025, user_method='magic',
        )


# -------------------------------------------------------------------
#  Numerical agreement of explicit methods
# -------------------------------------------------------------------


@pytest.mark.parametrize(
    "r,K,is_rel,is_per",
    [
        (2, 9, False, True),
        (2, 9, False, False),
        (3, 8, False, True),
        (3, 8, False, False),
        (4, 10, False, True),
        # Rel mode: still tests that explicit method='mobius' produces
        # a value consistent with centres at moderate σ/P.
        (3, 8, True, True),
        (3, 8, True, False),
    ],
)
def test_eval_methods_agree(r, K, is_rel, is_per):
    """At well-conditioned cells, centres / orbit / auto agree to
    floating-point precision (with abs floor for near-zero density)."""
    rng = np.random.default_rng(0)
    sigma = 33.0 if is_per else 30.0
    period = P if is_per else 0.0
    if is_per:
        p = rng.uniform(0, P, K)
    else:
        p = rng.uniform(-300, 300, K)
    w = rng.uniform(0.5, 1.5, K)
    T = build_exp_tens(p, w, sigma, r, is_rel, is_per, period, verbose=False)

    dim = r - 1 if is_rel else r
    if is_per:
        x = rng.uniform(0, P, (dim, 30))
    else:
        x = rng.uniform(-300, 300, (dim, 30))

    v_auto = eval_exp_tens(T, x, method='auto', verbose=False)
    v_centres = eval_exp_tens(T, x, method='centres', verbose=False)
    v_orbit = eval_exp_tens(T, x, method='mobius', verbose=False)

    assert np.allclose(v_auto, v_centres, atol=ATOL, rtol=RTOL)
    assert np.allclose(v_centres, v_orbit, atol=ATOL, rtol=RTOL)


# -------------------------------------------------------------------
#  K-r precision guard
# -------------------------------------------------------------------


def test_eval_orbit_at_K_minus_r_below_guard_via_explicit():
    """User-forced method='mobius' at K-r < 2 still runs (no guard
    on explicit override). Auto would have picked centres."""
    rng = np.random.default_rng(0)
    K, r = 4, 3  # K - r = 1, below guard
    p = rng.uniform(0, P, K)
    w = rng.uniform(0.5, 1.5, K)
    T = build_exp_tens(p, w, 33.0, r, False, True, P, verbose=False)
    x = rng.uniform(0, P, (r, 5))
    # Auto routes to centres.
    v_auto = eval_exp_tens(T, x, method='auto', verbose=False)
    v_centres = eval_exp_tens(T, x, method='centres', verbose=False)
    assert np.allclose(v_auto, v_centres, atol=ATOL, rtol=RTOL)
    # Explicit orbit runs without error (precision may be reduced).
    v_orbit = eval_exp_tens(T, x, method='mobius', verbose=False)
    assert np.all(np.isfinite(v_orbit))


# -------------------------------------------------------------------
#  Memory-saving regime
# -------------------------------------------------------------------


def test_eval_orbit_memory_efficient_at_high_r():
    """At r=4 K=20 the centres tensor has ~116k distinct 4-tuples
    × dim × n_q. Orbit path materialises only O(K) per partition.
    Both must agree to FP precision on a small query batch."""
    rng = np.random.default_rng(0)
    r, K = 4, 20
    p = rng.uniform(0, P, K)
    w = rng.uniform(0.5, 1.5, K)
    T = build_exp_tens(p, w, 50.0, r, False, True, P, verbose=False)
    x = rng.uniform(0, P, (r, 5))
    v_centres = eval_exp_tens(T, x, method='centres', verbose=False)
    v_orbit = eval_exp_tens(T, x, method='mobius', verbose=False)
    # At low cancellation, orbit should match centres at FP.
    assert np.allclose(v_centres, v_orbit, atol=ATOL, rtol=1e-9)
