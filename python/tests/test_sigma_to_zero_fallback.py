"""Regression tests for the orbit-corruption fallback machinery.

Two regimes can corrupt orbit results:

1. **Catastrophic-overflow regime** (σ → 0, low K, r ≥ 3 in abs modes).
   Orbit returns non-finite or sign-corrupt inner products. Caught by
   the post-hoc sanity check ``_orbit_ips_look_corrupted``.

2. **Quiet-corruption regime** (sharp σ relative to data range, low
   K-r margin in abs modes). Orbit returns finite-looking values that
   are wrong by 1e-4 to 1e-2 relative. Caught by the runtime
   cancellation-ratio diagnostic in the alternating Möbius sum.

Both regimes are documented in V22_DEV_LOG.md. This file pins the
fallback behaviour so it doesn't regress.
"""
import numpy as np
import pytest

from mpt.tensor import (
    build_exp_tens, cos_sim_exp_tens, _orbit_ips_look_corrupted,
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
        [p1], [w1], [sigma], [r], [0],
        [False], [is_per], [P], verbose=False,
    )
    d2 = build_exp_tens(
        [p2], [w2], [sigma], [r], [0],
        [False], [is_per], [P], verbose=False,
    )
    return d1, d2


# -------------------------------------------------------------------
# Regime 1: catastrophic-overflow at σ→0
# -------------------------------------------------------------------


@pytest.mark.parametrize("is_per", [False, True])
@pytest.mark.parametrize("sigma_over_P", [1e-3, 1e-4, 1e-5, 1e-7])
def test_sigma_to_zero_auto_matches_pairwise(is_per, sigma_over_P):
    """At σ→0 with r=4 K=6, orbit path corrupts; auto must fall back."""
    sigma = sigma_over_P * P
    d1, d2 = _make_pair(r=4, K=6, sigma=sigma, is_per=is_per)
    c_auto = cos_sim_exp_tens(d1, d2, method="auto", verbose=False)
    c_pw = cos_sim_exp_tens(d1, d2, method="bulger", verbose=False)
    assert np.isfinite(c_auto)
    assert np.isfinite(c_pw)
    assert abs(c_auto - c_pw) <= 1e-12 + 1e-12 * abs(c_pw)


def test_orbit_ips_look_corrupted_signals():
    """Direct test of the catastrophic-overflow predicate."""
    # Healthy case: positive auto-IPs, cross IP within bound
    assert not _orbit_ips_look_corrupted(0.5, 1.0, 1.0)
    # NaN
    assert _orbit_ips_look_corrupted(float("nan"), 1.0, 1.0)
    assert _orbit_ips_look_corrupted(0.5, float("nan"), 1.0)
    # Inf
    assert _orbit_ips_look_corrupted(float("inf"), 1.0, 1.0)
    # Negative auto-IP (sign corruption)
    assert _orbit_ips_look_corrupted(0.0, -1.0, 1.0)
    assert _orbit_ips_look_corrupted(0.0, 1.0, -1.0)
    # Cosine > 1 (impossible for a real cosine)
    assert _orbit_ips_look_corrupted(2.0, 1.0, 1.0)
    # Cosine just slightly > 1 (within FP tolerance) is OK
    assert not _orbit_ips_look_corrupted(1.0 + 1e-10, 1.0, 1.0)


def test_sa_sigma_to_zero_auto_matches_pairwise():
    """SA-path version of the σ→0 fallback test (r=4 K=6 abs_per)."""
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
# Regime 2: quiet-corruption (sharp σ, low K-r margin)
# -------------------------------------------------------------------


@pytest.mark.parametrize("is_per", [False, True])
@pytest.mark.parametrize("seed", range(10))
def test_quiet_corruption_regime_orbit_actually_clean(is_per, seed):
    """At r=4 K=6 σ/P=0.01 the per-attribute orbit Möbius computation
    has near-zero per-(n,m) cancellation ratios — the alternating sum
    has lost most digits at the cell level. Earlier versions used this
    as a fallback trigger ("quiet corruption"). v2.2 sweep showed,
    however, that the per-cell ratio is misleading: cells with bad
    ratios contribute negligibly to Σ_{n,m} P[n,m], so the cosine
    consumes a sum that is nonetheless correct to FP precision.

    This test pins that empirical finding: orbit and pairwise must
    agree to FP precision at r=4 K=6 σ/P=0.01 even though the per-cell
    diagnostic would have flagged the configuration. The dispatcher
    no longer falls back here, and shouldn't need to.
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
    """``_cos_sim_exp_tens_ma_orbit`` returns (xy, xx, yy). The
    per-cell cancellation diagnostic was removed in v2.2.0 after
    empirical sweeps showed it was over-conservative for self-IPs
    (false alarms in 100% of typical musical regimes) without ever
    catching a genuinely corrupt result; see CHANGELOG and
    V22_DEV_LOG.md for the full investigation.
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
