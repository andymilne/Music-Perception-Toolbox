"""Regression tests for the Möbius route's fallback to Bulger's method.

Two regimes are pinned, and they are pinned for opposite reasons.

1. **Catastrophic-overflow regime** (σ → 0, low K, r ≥ 3 in absolute
   modes). The alternating partition sum cancels to near machine
   epsilon and the ratio then overflows, so the Möbius route can
   return a non-finite inner product or one of impossible sign.
   ``_orbit_ips_impossible`` detects this and the dispatcher reroutes;
   the tests here assert that ``method='auto'`` agrees with Bulger's
   method throughout.

2. **Per-cell cancellation regime** (σ sharp relative to the data
   range, small K − r margin in absolute modes). Individual Möbius
   cells lose most of their significant digits, but the cosine
   consumes only the sums Σ_{n,m} P[n,m], to which those cells
   contribute negligibly. The tests here assert that the Möbius route
   is nonetheless correct to floating-point precision, which is why
   the per-cell diagnostic that used to divert on this signal is no
   longer in the code.

Accuracy, as opposed to impossibility, is governed by
``truncationSigmas`` and is tested elsewhere.
"""
import numpy as np
import pytest

from mpt.tensor import (
    build_exp_tens, cos_sim_exp_tens, _orbit_ips_impossible,
    _cos_sim_exp_tens_ma_orbit,
)


P = 1200.0
N = 2
SEED = 0


def _make_pair(r, K, sigma, is_per, share_half=True, seed=SEED):
    rng = np.random.default_rng(seed)
    if share_half:
        shared = rng.uniform(0, P, (K // 2, N))
        p1 = np.vstack([shared, rng.uniform(0, P, (K - K // 2, N))])
        p2 = np.vstack([shared, rng.uniform(0, P, (K - K // 2, N))])
    else:
        p1 = rng.uniform(0, P, (K, N))
        p2 = rng.uniform(0, P, (K, N))
    w1 = rng.uniform(0.1, 1.0, (K, N))
    w2 = rng.uniform(0.1, 1.0, (K, N))
    d1 = build_exp_tens(
        [p1], [w1], [sigma], [r], 
        [False], [is_per], [P], verbose=False,
    )
    d2 = build_exp_tens(
        [p2], [w2], [sigma], [r], 
        [False], [is_per], [P], verbose=False,
    )
    return d1, d2


# -------------------------------------------------------------------
# Regime 1: catastrophic-overflow at σ→0
# -------------------------------------------------------------------


@pytest.mark.parametrize("is_per", [False, True])
@pytest.mark.parametrize("sigma_over_P", [1e-3, 1e-4, 1e-5, 1e-7])
def test_sigma_to_zero_auto_matches_pairwise(is_per, sigma_over_P):
    """At σ→0 with r=4 K=6 the Möbius route returns an impossible
    value; ``method='auto'`` must fall back to Bulger's method."""
    sigma = sigma_over_P * P
    d1, d2 = _make_pair(r=4, K=6, sigma=sigma, is_per=is_per)
    c_auto = cos_sim_exp_tens(d1, d2, method="auto", verbose=False)
    c_pw = cos_sim_exp_tens(d1, d2, method="bulger", verbose=False)
    assert np.isfinite(c_auto)
    assert np.isfinite(c_pw)
    assert abs(c_auto - c_pw) <= 1e-12 + 1e-12 * abs(c_pw)


def test_orbit_ips_impossible_signals():
    """Direct test of the impossible-value predicate."""
    # Healthy case: positive auto-IPs, cross IP within bound
    assert not _orbit_ips_impossible(0.5, 1.0, 1.0)
    # NaN
    assert _orbit_ips_impossible(float("nan"), 1.0, 1.0)
    assert _orbit_ips_impossible(0.5, float("nan"), 1.0)
    # Inf
    assert _orbit_ips_impossible(float("inf"), 1.0, 1.0)
    # Negative auto-IP (a self inner product cannot be negative)
    assert _orbit_ips_impossible(0.0, -1.0, 1.0)
    assert _orbit_ips_impossible(0.0, 1.0, -1.0)
    # Cosine > 1 (impossible for a real cosine)
    assert _orbit_ips_impossible(2.0, 1.0, 1.0)
    # Cosine just slightly > 1 (within FP tolerance) is OK
    assert not _orbit_ips_impossible(1.0 + 1e-10, 1.0, 1.0)


def test_single_multiset_sigma_to_zero_auto_matches_pairwise():
    """single-multiset-path version of the σ→0 fallback test (r=4 K=6 abs_per)."""
    sigma = 1e-5 * P
    r = 4
    K = 6
    rng = np.random.default_rng(SEED)
    shared = rng.uniform(0, P, K // 2)
    p1 = np.concatenate([shared, rng.uniform(0, P, K - K // 2)])
    p2 = np.concatenate([shared, rng.uniform(0, P, K - K // 2)])
    w1 = rng.uniform(0.1, 1.0, K)
    w2 = rng.uniform(0.1, 1.0, K)
    d1 = build_exp_tens(
        p1, w1, sigma, r, False, True, P, verbose=False,
    )
    d2 = build_exp_tens(
        p2, w2, sigma, r, False, True, P, verbose=False,
    )
    c_auto = cos_sim_exp_tens(d1, d2, method="auto", verbose=False)
    c_pw = cos_sim_exp_tens(d1, d2, method="bulger", verbose=False)
    assert np.isfinite(c_auto)
    assert np.isfinite(c_pw)
    assert abs(c_auto - c_pw) <= 1e-12 + 1e-12 * abs(c_pw)


# -------------------------------------------------------------------
# Regime 2: per-cell cancellation (sharp σ, low K-r margin)
# -------------------------------------------------------------------


@pytest.mark.parametrize("is_per", [False, True])
@pytest.mark.parametrize("seed", range(10))
def test_per_cell_cancellation_regime_orbit_actually_clean(is_per, seed):
    """At r=4 K=6 σ/P=0.01 the per-attribute Möbius computation has
    near-zero per-(n,m) cancellation ratios: the alternating sum has
    lost most of its digits at the cell level. The per-cell ratio is
    nonetheless misleading, because cells with poor ratios contribute
    negligibly to Σ_{n,m} P[n,m], and the cosine consumes only that
    sum.

    This test pins that finding: the Möbius and Bulger routes agree to
    floating-point precision at r=4 K=6 σ/P=0.01, a configuration the
    per-cell diagnostic would have flagged. The dispatcher does not
    divert here, and does not need to.
    """
    sigma = 0.01 * P
    d1, d2 = _make_pair(r=4, K=6, sigma=sigma, is_per=is_per,
                        share_half=False, seed=seed)
    c_orbit = cos_sim_exp_tens(d1, d2, method="mobius", verbose=False)
    c_pw = cos_sim_exp_tens(d1, d2, method="bulger", verbose=False)
    assert abs(c_orbit - c_pw) <= 1e-12 + 1e-12 * abs(c_pw), (
        f"seed={seed}: orbit {c_orbit:.6e} vs pairwise {c_pw:.6e}"
    )


def test_orbit_returns_3tuple():
    """``_cos_sim_exp_tens_ma_orbit`` returns (xy, xx, yy). It carries
    no per-cell cancellation diagnostic: sweeps found that one flagged
    self inner products in essentially every typical musical regime
    while never catching a value that was actually wrong. See the
    CHANGELOG.
    """
    d1, d2 = _make_pair(r=3, K=8, sigma=50.0, is_per=False)
    result = _cos_sim_exp_tens_ma_orbit(d1, d2)
    assert len(result) == 3
    xy, xx, yy = result
    assert all(np.isfinite([xy, xx, yy]))
    assert xx > 0 and yy > 0


def test_orbit_clean_regime_agrees_with_pairwise():
    """In a clean regime (moderate σ, large K-r margin) the orbit and
    pairwise paths must agree to FP precision."""
    sigma = 50.0
    d1, d2 = _make_pair(r=2, K=8, sigma=sigma, is_per=True)
    c_orbit = cos_sim_exp_tens(d1, d2, method="mobius", verbose=False)
    c_pw = cos_sim_exp_tens(d1, d2, method="bulger", verbose=False)
    assert abs(c_orbit - c_pw) <= 1e-13 + 1e-13 * abs(c_pw)
