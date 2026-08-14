"""Memoised self inner products, the oneSidedDenom <X,X> skip, and the
all-r = 1 direct inner-product route.

The contract under test: a broadcast (one context density against many
query densities) returns exactly the values that fresh scalar calls
return, under both normalisations and on both the Bulger and Möbius
routes; the r = 1 direct route computes the identical quantity to the
generic MA path; and the memoised values never leak across settings
that change them.
"""
import numpy as np
import pytest

import mpt
from mpt._defaults import resolve_truncation_sigmas
from mpt._tensor import cosine as C


RNG = np.random.default_rng(7)


def _r1_density(pitches, times, sigma=(0.25, 0.08), is_per=(False, False),
                period=(0.0, 0.0)):
    return mpt.build_exp_tens(
        [np.asarray(pitches, dtype=float).reshape(1, -1),
         np.asarray(times, dtype=float).reshape(1, -1)],
        None, list(sigma), [1, 1], [False, False], list(is_per),
        list(period), verbose=False,
    )


# ------------------------------------------------------------------
#  Broadcast == fresh scalar calls (cache transparency), Bulger route
# ------------------------------------------------------------------

@pytest.mark.parametrize("norm", ["cosine", "oneSidedDenom"])
def test_broadcast_matches_fresh_scalars_r1(norm):
    N = 150
    P = RNG.uniform(40, 90, N)
    T = np.sort(RNG.uniform(0, 40, N))
    qP = np.array([60.0, 63.0, 68.0])
    qT = np.array([0.0, 1.0, 3.0])

    dX = _r1_density(P, T)
    dYs = [_r1_density(qP + k * 0.7, qT + k * 1.3) for k in range(6)]
    batch = mpt.cos_sim_exp_tens(dX, dYs, normalize=norm, verbose=False)

    fresh = np.array([
        mpt.cos_sim_exp_tens(
            _r1_density(P, T), _r1_density(qP + k * 0.7, qT + k * 1.3),
            normalize=norm, verbose=False)
        for k in range(6)
    ])
    np.testing.assert_allclose(batch, fresh, rtol=0, atol=1e-12)


# ------------------------------------------------------------------
#  Broadcast == fresh scalar calls on the Möbius (orbit) route
# ------------------------------------------------------------------

def _r2_density(vals, sigma=1.0, n_events=None):
    vals = np.asarray(vals, dtype=float)
    return mpt.build_exp_tens(
        [vals], None, [sigma], [2], [True], [False], [0.0],
        verbose=False,
    )


@pytest.mark.parametrize("norm", ["cosine", "oneSidedDenom"])
def test_broadcast_matches_fresh_scalars_mobius(norm):
    # r = 2, relative, K large enough that the dispatcher's Möbius
    # route is exercised when forced; forcing keeps the test route-
    # deterministic while the equality contract stays the point.
    X = RNG.uniform(0, 24, (2, 30))
    Ys = [RNG.uniform(0, 24, (2, 6)) for _ in range(4)]

    def build(v):
        return mpt.build_exp_tens([v], None, [1.0], [2], [True], [False],
                                  [0.0], verbose=False)

    dX = build(X)
    dYs = [build(Y) for Y in Ys]
    batch = np.array([
        mpt.cos_sim_exp_tens(dX, dY, method="mobius", normalize=norm,
                             verbose=False)
        for dY in dYs
    ])
    fresh = np.array([
        mpt.cos_sim_exp_tens(build(X), build(Y), method="mobius",
                             normalize=norm, verbose=False)
        for Y in Ys
    ])
    np.testing.assert_allclose(batch, fresh, rtol=0, atol=1e-12)


# ------------------------------------------------------------------
#  r = 1 direct route == generic MA path, all abs mode combinations
# ------------------------------------------------------------------

@pytest.mark.parametrize("is_per,period", [
    ((False, False), (0.0, 0.0)),
    ((True, False), (12.0, 0.0)),
    ((True, True), (12.0, 4.0)),
])
def test_r1_direct_matches_generic(is_per, period):
    d1 = _r1_density(RNG.uniform(0, 24, 31), RNG.uniform(0, 8, 31),
                     sigma=(0.4, 0.1), is_per=is_per, period=period)
    d2 = _r1_density(RNG.uniform(0, 24, 17), RNG.uniform(0, 8, 17),
                     sigma=(0.4, 0.1), is_per=is_per, period=period)
    ts = resolve_truncation_sigmas(None)
    fast = C._ip_r1_direct(
        d1.u_perm, d1.w_j, d1.n_j, d2.v_comb, d2.wv_comb, d2.n_k,
        2, d1.sigma, d1.is_rel, d1.is_per, d1.period,
        truncation_sigmas=ts, wrap=list(d1.wrap),
    )
    slow = C._ip_full_ma(
        d1.u_perm, d1.w_j, d1.n_j, d2.v_comb, d2.wv_comb, d2.n_k,
        2, d1.r, d1.sigma, d1.is_rel, d1.is_per, d1.period,
        truncation_sigmas=ts, inner_r=None, wrap=list(d1.wrap),
    )
    assert fast == pytest.approx(slow, rel=1e-14, abs=0.0)


# ------------------------------------------------------------------
#  oneSidedDenom skips <X,X>; cosine refuses a skipped <X,X>
# ------------------------------------------------------------------

def test_pairwise_need_xx_false_returns_none_xx():
    dX = _r1_density(RNG.uniform(40, 90, 50),
                     np.sort(RNG.uniform(0, 20, 50))).pruned()
    dY = _r1_density([60.0, 64.0, 67.0], [0.0, 1.0, 2.0]).pruned()
    ip_xy, ip_xx, ip_yy = C._cos_sim_exp_tens_ma_pairwise(
        dX, dY, verbose=False, need_xx=False)
    assert ip_xx is None
    assert np.isfinite(ip_xy) and np.isfinite(ip_yy)
    # The value oneSidedDenom produces from the reduced triple matches
    # the full-triple computation.
    full = C._cos_sim_exp_tens_ma_pairwise(dX, dY, verbose=False)
    assert ip_xy == full[0] and ip_yy == full[2]
    with pytest.raises(ValueError):
        C._finalise_normalisation(ip_xy, None, ip_yy, "cosine")


def test_cached_self_ip_is_reused_and_keyed():
    dX = _r1_density(RNG.uniform(40, 90, 40),
                     np.sort(RNG.uniform(0, 20, 40))).pruned()
    dY = _r1_density([60.0, 64.0], [0.0, 1.0]).pruned()
    C._cos_sim_exp_tens_ma_pairwise(dX, dY, verbose=False)
    key = C._self_ip_cache_key("bulger", None, None)
    assert key in dX._self_ip_cache and key in dY._self_ip_cache
    cached = dX._self_ip_cache[key]
    # Second call reuses the cached value (identity of the float, and
    # the triple slot equals it exactly).
    trip = C._cos_sim_exp_tens_ma_pairwise(dX, dY, verbose=False)
    assert trip[1] == cached
    # A different truncation budget is a different key: no leak.
    C._cos_sim_exp_tens_ma_pairwise(dX, dY, verbose=False,
                                    truncation_sigmas=6.0)
    key6 = C._self_ip_cache_key("bulger", 6.0, None)
    assert key6 in dX._self_ip_cache
    assert key6 != key


def test_pruned_is_memoised():
    d = _r1_density(RNG.uniform(40, 90, 10), np.sort(RNG.uniform(0, 5, 10)))
    assert d.pruned() is d.pruned()
