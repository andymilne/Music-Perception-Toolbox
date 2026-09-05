"""Tests for the v3 multi-attribute orbit dispatcher.

The MA inner product factorises per attribute and per event pair (JMM
Eq. 3.4 with the per-attribute integral separation), so orbit Möbius
applies attribute-by-attribute. The integration tests below check
that orbit-MA matches pairwise-MA at the cosine level on representative
MAET configurations (single-multiset equivalence, pitch+time, mixed
r_a, NaN fallback, etc.).
"""
from __future__ import annotations

import warnings
from math import comb, factorial

import numpy as np
import pytest

import mpt
from mpt import build_exp_tens, cos_sim_exp_tens
from mpt._tensor.dispatch import _predict_pairwise_kernel_size
from mpt.tensor import (
    _ORBIT_R_MAX_SHIPPED,
    _cos_sim_exp_tens_ma_orbit,
    _cos_sim_exp_tens_ma_pairwise,
    _select_ma_inner_product_method,
)


# ----------------------------------------------------------------------
# Dispatcher decisions
# ----------------------------------------------------------------------


def _disp_kwargs(r_max=3, A=1, K=8, N_x=8, N_y=8,
                 any_per=False,
                 any_rel_nonper=False, any_rel_per=False,
                 sigma_over_P_max=0.0, user_method='auto'):
    """Helper to build kwargs for the cost-model dispatcher.

    The new dispatcher needs the per-attribute (r_a, K_a) vectors and
    event counts, not just r_max, because the cost-model crossover
    depends on N and K. It also needs ``any_per`` separately from
    ``any_rel_per`` because the pairwise wrap on δ tensors costs ~2×
    even in absolute mode (verified by side-by-side bench across the
    four (rel, per) combinations).

    Ragged K_{a,n} (NaN-padded events) is handled inside the per-attr
    IP wrapper by zero-weight padding; the dispatcher does not receive
    a ``has_nan`` flag.
    """
    r_vec = np.array([r_max] * A, dtype=np.intp)
    k_vec = np.array([K] * A, dtype=np.intp)
    return dict(
        r_vec=r_vec, k_vec=k_vec, A=A,
        N_x=N_x, N_y=N_y,
        any_per=any_per,
        any_rel_nonper=any_rel_nonper, any_rel_per=any_rel_per,
        sigma_over_P_max=sigma_over_P_max, user_method=user_method,
    )


def test_ma_dispatcher_routes_orbit_for_ragged_k():
    """Ragged K_{a,n} (NaN-padded events) no longer routes to pairwise.

    The MA dispatcher used to gate on a ``has_nan`` flag, but the
    per-attribute IP wrapper now handles NaN-padded events via a
    safe/unsafe partition (events with K_eff - r >= 2 go through
    orbit; pairs involving any K_eff - r < 2 event go through direct
    enumeration). The dispatcher therefore picks orbit on otherwise-
    healthy parameters even when NaN entries are present in p_attr.
    """
    # Same kwargs as test_ma_dispatcher_routes_orbit_when_clean[3]:
    # r_max=3, K=8, N=8, A=1 — auto picks orbit.
    assert _select_ma_inner_product_method(
        **_disp_kwargs(),
    ) == 'mobius'


def test_ma_dispatcher_routes_pairwise_at_r1():
    """r_max=1: no within-tuple distinct-index structure to exploit."""
    assert _select_ma_inner_product_method(
        **_disp_kwargs(r_max=1, K=2),
    ) == 'bulger'


# ----------------------------------------------------------------------
# Unequal value counts between the two densities
# ----------------------------------------------------------------------

# A chord compared against a scale, or a reference tuning against an
# equal division, gives the two densities different numbers of values.
# Bulger's method builds three matrices -- the cross matrix and one self
# matrix per density -- so its size is dominated by the larger density's
# self matrix, and the estimate has to read both counts to see that.


def test_pairwise_size_reduces_to_three_cross_terms_at_equal_counts():
    # Each density contributes n_J = N.prod r!C and n_K = N.prod C; at
    # equal counts the three matrices coincide.
    for r, K, N in ((2, 8, 3), (3, 10, 1), (4, 9, 2)):
        r_vec = np.array([r], dtype=np.intp)
        k_vec = np.array([K], dtype=np.intp)
        cross = (N * factorial(r) * comb(K, r)) * (N * comb(K, r))
        assert _predict_pairwise_kernel_size(
            r_vec, k_vec, 1, N, N) == pytest.approx(3.0 * cross, rel=1e-12)


def test_pairwise_size_grows_with_the_second_count():
    r_vec = np.array([2], dtype=np.intp)
    k_x = np.array([5], dtype=np.intp)
    base = _predict_pairwise_kernel_size(
        r_vec, k_x, 1, 1, 1, k_vec_y=np.array([5], dtype=np.intp))
    prev = base
    for K_y in (10, 20, 40, 80):
        got = _predict_pairwise_kernel_size(
            r_vec, k_x, 1, 1, 1, k_vec_y=np.array([K_y], dtype=np.intp))
        assert got > prev
        prev = got
    # At five values against eighty the second self matrix carries the
    # work: it holds 19971200 of the 20034600 entries, so the total is
    # 317 times the cross matrix and 33391 times the equal-count total.
    assert prev / base == pytest.approx(33391.0, rel=1e-9)


def test_second_count_vector_omitted_means_the_counts_agree():
    r_vec = np.array([3], dtype=np.intp)
    k_vec = np.array([9], dtype=np.intp)
    assert (_predict_pairwise_kernel_size(r_vec, k_vec, 1, 4, 4)
            == _predict_pairwise_kernel_size(
                r_vec, k_vec, 1, 4, 4, k_vec_y=k_vec))


def test_dispatcher_routes_mobius_once_the_second_density_is_large():
    """Five values against a large equal division: Bulger's three
    matrices grow as the fourth power of the second count while the
    Möbius side stays near its base, so the route must switch."""
    kw = _disp_kwargs(r_max=2, K=5, N_x=1, N_y=1,
                      any_per=True, any_rel_per=True,
                      sigma_over_P_max=0.005)
    kw['rel_vec'] = np.array([True])
    kw['nu_vec'] = np.array([1666.0])
    large = dict(kw, k_vec_y=np.array([80], dtype=np.intp))
    assert _select_ma_inner_product_method(**large) == 'mobius'

    # Five values against eight is not asserted. Measured, Bulger's
    # method takes 0.42 ms there and the Mobius method 0.63 ms, a ratio
    # of 1.5, and the cost model puts it on the wrong side of that. Both
    # are sub-millisecond and the call is nearly all overhead, so pinning
    # a 1.5-fold preference would be pinning noise. The case this test
    # exists for is the one above, where the ratio is 184.


def test_dispatcher_is_unchanged_when_the_counts_agree():
    # The corrected estimates reduce exactly to the previous ones at
    # equal counts, so no equal-count decision may move.
    for r in (2, 3):
        for K in (5, 8, 12, 20):
            kw = _disp_kwargs(r_max=r, K=K, N_x=4, N_y=4, any_per=True)
            with_y = dict(kw, k_vec_y=np.array([K], dtype=np.intp))
            assert (_select_ma_inner_product_method(**kw)
                    == _select_ma_inner_product_method(**with_y))


@pytest.mark.parametrize("r_max", [2, 3, 4, _ORBIT_R_MAX_SHIPPED])
def test_ma_dispatcher_routes_orbit_when_clean(r_max):
    """Healthy parameters with K and N large enough: dispatcher picks orbit."""
    # Pick K, N comfortably above the cost-model crossover for each r.
    K = max(12, r_max + 6)
    assert _select_ma_inner_product_method(
        **_disp_kwargs(r_max=r_max, K=K, N_x=12, N_y=12),
    ) == 'mobius'


def test_ma_dispatcher_raises_when_r_too_large_and_bulger_infeasible():
    """r_max above the shipped-table cutoff leaves no viable route.

    Above the shipped orbit order the Möbius method is unavailable, so
    the single-image (Bulger) route is the only fallback -- but its
    tuple-pair kernel is quadratic in the ordered-tuple count, which at
    r = 9 exceeds any practical budget for every K >= r (at K = 9 the
    kernel alone is ~1e12 bytes). The feasibility guard therefore raises
    with a sizing message rather than returning a route that cannot be
    executed.
    """
    from mpt._tensor.dispatch import SingleImageInfeasibleError
    with pytest.raises(SingleImageInfeasibleError, match="single-image"):
        _select_ma_inner_product_method(
            **_disp_kwargs(r_max=_ORBIT_R_MAX_SHIPPED + 1, K=12),
        )


def test_ma_dispatcher_routes_orbit_for_rel_nonper_at_A1():
    """rel + nonper at A=1, r=3, K=12, N=12: the Möbius method wins, and
    not narrowly.

    An earlier version of this test asserted the opposite, on the
    reasoning that a single attribute gives the Möbius per-attribute
    factorisation nothing to factor across, leaving one joint kernel
    against three per-attribute matrices --- and quoting a measurement
    of roughly 6 s against 12 s. Measured on this shape, Bulger's method
    takes 24.5 s and the Möbius method 173 ms, a factor of 142. The
    factorisation is not what decides it: Bulger's joint kernel grows
    with the event count as well as the value count, and at 12 events
    that is 144 event pairs in one array."""
    assert _select_ma_inner_product_method(
        **_disp_kwargs(K=12, N_x=12, N_y=12, A=1, any_rel_nonper=True),
    ) == 'mobius'


def test_ma_dispatcher_routes_orbit_for_rel_nonper_at_A2_heavy_K():
    """rel + nonper at A≥2 with moderate K: pairwise's ∏_a C(K_a,r_a)²
    compounding overtakes orbit's additive A·K_max² growth, so orbit
    wins. Without this branch the dispatcher would route to pairwise
    and OOM at larger K."""
    # A=2 r=3 K=8 N=4 rel_nonper:
    #   pw_pred = 4·(3!·C(8,3))²·... ≈ 4·336·56·1e-4·... = ~283 s
    #   orbit_pred = 2·(5 + 16·64·1.05) = ~2160 ms
    assert _select_ma_inner_product_method(
        **_disp_kwargs(r_max=3, K=8, N_x=4, N_y=4, A=2,
                       any_rel_nonper=True),
    ) == 'mobius'


def test_ma_dispatcher_warns_above_perrel_threshold():
    """Retired in v3: rel-per full-image is now the default measure and
    the dispatch no longer warns about the "substitution" (there is none
    to warn about; the user's ``wrap`` choice determines the measure).

    This test survives as a positive check that the previously-warning
    call path is now silent, and that the cost model still routes to
    Möbius on large problems and Bulger on small ones.
    """
    # Large problem: orbit/all-image is the faster path → Möbius, silent.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        chosen = _select_ma_inner_product_method(
            **_disp_kwargs(K=12, N_x=12, N_y=12, any_per=True,
                           any_rel_per=True, sigma_over_P_max=0.05),
        )
    assert chosen == 'mobius'
    # Small problem: single-wrap pairwise is the faster path → Bulger, silent.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        chosen = _select_ma_inner_product_method(
            **_disp_kwargs(K=6, N_x=6, N_y=6, any_per=True,
                           any_rel_per=True, sigma_over_P_max=0.05),
        )
    assert chosen == 'bulger'


def test_ma_dispatcher_routes_pairwise_at_small_problem():
    """Small N and K: pairwise wins (cost-model picks pairwise)."""
    # N=2, K=4, r=2, A=1, abs: pw_size = 4·2·6² = 144. Tiny — pairwise.
    assert _select_ma_inner_product_method(
        **_disp_kwargs(r_max=2, K=4, N_x=2, N_y=2),
    ) == 'bulger'


@pytest.mark.parametrize("mode_flags,A,K,N,expected", [
    # (any_per, any_rel_nonper, any_rel_per), A, K, N → expected route.
    # Each mode is tested at a point where the dispatcher decision is
    # informative (well above or below the cost-model crossover).
    ((False, False, False), 1, 12, 12, 'mobius'),     # abs + nonper, A=1 large
    ((True,  False, False), 1, 12, 12, 'mobius'),     # abs + per, A=1 large
    # rel + nonper, A=1: measured at r=3, K=12, N=12, Bulger's method
    # takes 24.5 s against the Möbius method's 173 ms. The comment this
    # replaces said the orbit path was catastrophic here; it is the
    # other way round, because Bulger's joint kernel grows with the
    # event count as well as the value count.
    ((False, True,  False), 1, 12, 12, 'mobius'),
    ((False, True,  False), 2,  8,  4, 'mobius'),     # rel + nonper, A=2 K=8: pw OOMs first
    ((True,  False, True),  1, 12, 12, 'mobius'),     # rel + per, A=1 large
])
def test_ma_dispatcher_routes_correctly_in_each_mode(
        mode_flags, A, K, N, expected):
    """Verify the dispatcher routes each of the four (rel, per) modes
    appropriately at representative points. rel + nonper is tested at
    both A=1 (where pairwise wins) and A=2 with moderate K (where
    pairwise's multiplicative blow-up makes orbit the only feasible
    choice)."""
    any_per, any_rel_nonper, any_rel_per = mode_flags
    chosen = _select_ma_inner_product_method(
        **_disp_kwargs(r_max=3, K=K, N_x=N, N_y=N, A=A,
                       any_per=any_per,
                       any_rel_nonper=any_rel_nonper,
                       any_rel_per=any_rel_per),
    )
    assert chosen == expected


@pytest.mark.parametrize("forced", ['bulger', 'centres', 'mobius'])
def test_ma_dispatcher_user_overrides_bypass_logic(forced):
    """Explicit method bypasses everything (e.g. the σ/P guard)."""
    chosen = _select_ma_inner_product_method(
        **_disp_kwargs(sigma_over_P_max=0.5, user_method=forced),
    )
    assert chosen == forced


# ----------------------------------------------------------------------
# Helper: build an MA density with the given parameters
# ----------------------------------------------------------------------


def _build_ma_pitch_time(rng, N, K_pitch=3, sigma_pitch=10.0, sigma_time=0.05,
                         r_pitch=2, r_time=1, P=1200.0,
                         pitch_rel=True, pitch_per=True):
    """Pitch + time MA density with N events, K_pitch partials per event."""
    pitch = rng.uniform(0, P, (K_pitch, N))
    time = np.atleast_2d(rng.uniform(0, N * 0.5, N))  # 1 × N
    return build_exp_tens(
        [pitch, time], None,
        [sigma_pitch, sigma_time], [r_pitch, r_time], 
        [pitch_rel, False], [pitch_per, False], [P, 0.0],
        verbose=False,
    )


# ----------------------------------------------------------------------
# Orbit MA matches pairwise MA at the cosine level
# ----------------------------------------------------------------------


@pytest.mark.parametrize("r_pitch", [2, 3])
@pytest.mark.parametrize("K_pitch", [3, 5])
@pytest.mark.parametrize("pitch_rel", [False, True])
def test_orbit_ma_matches_pairwise_pitch_time(r_pitch, K_pitch, pitch_rel):
    """Pitch + time MAET: orbit-MA cosine matches pairwise-MA cosine.

    Spans the typical MAET use case (one multi-value attribute with
    r_a >= 2, one single-value attribute with r_a = 1) across rel/abs
    pitch and several K_pitch values.
    """
    rng = np.random.default_rng(seed=hash((r_pitch, K_pitch, pitch_rel)) & 0xFFFF)
    N = 6
    if K_pitch < r_pitch:
        pytest.skip("Insufficient K_pitch for r_pitch.")

    dens_x = _build_ma_pitch_time(
        rng, N, K_pitch=K_pitch, r_pitch=r_pitch, pitch_rel=pitch_rel,
    )
    dens_y = _build_ma_pitch_time(
        rng, N, K_pitch=K_pitch, r_pitch=r_pitch, pitch_rel=pitch_rel,
    )

    cos_orbit = cos_sim_exp_tens(dens_x, dens_y, method='auto', verbose=False)
    cos_pw = cos_sim_exp_tens(dens_x, dens_y, method='bulger', verbose=False)
    abs_err = abs(cos_orbit - cos_pw)
    rel_err = abs_err / max(abs(cos_orbit), abs(cos_pw), 1e-300)
    assert rel_err < 1e-9 or abs_err < 1e-12, (
        f"r_pitch={r_pitch}, K_pitch={K_pitch}, pitch_rel={pitch_rel}: "
        f"orbit={cos_orbit:.10e}, pairwise={cos_pw:.10e}, "
        f"abs_err={abs_err:.2e}, rel_err={rel_err:.2e}"
    )


def test_orbit_ma_matches_pairwise_single_attr():
    """A=1, K_a > 1, r_a >= 2: orbit-MA matches pairwise-MA.

    The simplest MA configuration that exercises the per-event-pair
    factorisation. With A=1 there's only one attribute factor in the
    product and the cross-event sum becomes the dominant operation.
    """
    rng = np.random.default_rng(seed=99)
    N = 8
    K = 5
    pitch = rng.uniform(0, 1200, (K, N))
    dens_x = build_exp_tens(
        [pitch], None,
        [12.0], [3], 
        [True], [True], [1200.0],
        verbose=False,
    )
    pitch2 = rng.uniform(0, 1200, (K, N))
    dens_y = build_exp_tens(
        [pitch2], None,
        [12.0], [3], 
        [True], [True], [1200.0],
        verbose=False,
    )
    cos_orbit = cos_sim_exp_tens(dens_x, dens_y, method='auto', verbose=False)
    cos_pw = cos_sim_exp_tens(dens_x, dens_y, method='bulger', verbose=False)
    assert abs(cos_orbit - cos_pw) < 1e-10


@pytest.mark.parametrize("r_a,K_a,N", [
    (2, 4, 4),  # A=2 r=2 K=4 N=4: pw_pred ≈ 11ms, orbit_pred ≈ 36ms → pw,
                # but verify orbit is correct when forced.
    (2, 6, 4),  # A=2 r=2 K=6 N=4: dispatcher routes to orbit.
    (3, 6, 2),  # A=2 r=3 K=6 N=2: dispatcher routes to orbit (pw blows up).
])
def test_orbit_ma_matches_pairwise_rel_nonper(r_a, K_a, N):
    """rel + nonper at A=2: orbit-MA cosine matches pairwise-MA cosine.

    The dispatcher now routes to orbit for these configurations (the
    rel_nonper hard fallback was removed once it was verified that
    pairwise compounds multiplicatively across attributes whereas orbit
    adds linearly, so for A ≥ 2 with moderate K orbit becomes the only
    feasible path). This test confirms numerical agreement to FP
    precision when the Möbius method is taken — a regression here would
    indicate a bug in the per-pair Python loop in
    `inner_product_orbit_pw_batched`.
    """
    rng = np.random.default_rng(seed=hash((r_a, K_a, N)) & 0xFFFF)
    A = 2
    p_attr_X = [np.sort(rng.uniform(0, 1000, (K_a, N)), axis=0)
                for _ in range(A)]
    p_attr_Y = [np.sort(rng.uniform(0, 1000, (K_a, N)), axis=0)
                for _ in range(A)]
    dens_x = build_exp_tens(
        p_attr_X, None,
        [12.0]*A, [r_a]*A,
        [True]*A, [False]*A, [1200.0]*A,   # rel + nonper, shared geometry
        verbose=False,
    )
    dens_y = build_exp_tens(
        p_attr_Y, None,
        [12.0]*A, [r_a]*A,
        [True]*A, [False]*A, [1200.0]*A,
        verbose=False,
    )
    cos_orbit = cos_sim_exp_tens(
        dens_x, dens_y, method='mobius', verbose=False,
    )
    cos_pw = cos_sim_exp_tens(
        dens_x, dens_y, method='bulger', verbose=False,
    )
    abs_err = abs(cos_orbit - cos_pw)
    rel_err = abs_err / max(abs(cos_orbit), abs(cos_pw), 1e-300)
    assert rel_err < 1e-9 or abs_err < 1e-12, (
        f"r_a={r_a}, K_a={K_a}, N={N}: "
        f"orbit={cos_orbit:.12e}, pairwise={cos_pw:.12e}, "
        f"abs_err={abs_err:.2e}, rel_err={rel_err:.2e}"
    )


def test_orbit_ma_self_cosine_is_one():
    """<X, X> / sqrt(<X,X><X,X>) = 1 under orbit MA."""
    rng = np.random.default_rng(seed=11)
    dens = _build_ma_pitch_time(rng, N=5, K_pitch=4, r_pitch=2)
    cos_self = cos_sim_exp_tens(dens, dens, method='auto', verbose=False)
    assert abs(cos_self - 1.0) < 1e-12


# ----------------------------------------------------------------------
# r_a = 1 in any attribute exercises the einsum branch of
# _ma_per_attr_inner_matrix (which falls outside the orbit machinery)
# ----------------------------------------------------------------------


def test_orbit_ma_handles_r1_attribute():
    """The pitch + time test already exercises r_time = 1, but check
    explicitly that the Möbius method works when ALL attributes have r_a = 1.

    With r_max = 1 the dispatcher routes to pairwise, but if the user
    forces orbit explicitly the einsum branch must still produce the
    correct result (via the per-attribute matrix path).
    """
    rng = np.random.default_rng(seed=33)
    N = 6
    a1 = np.atleast_2d(rng.uniform(0, 1200, N))
    a2 = np.atleast_2d(rng.uniform(0, 1200, N))
    dens_x = build_exp_tens(
        [a1, a2], None,
        [10.0, 10.0], [1, 1], 
        [False, False], [True, True], [1200.0, 1200.0],
        verbose=False,
    )
    a1y = np.atleast_2d(rng.uniform(0, 1200, N))
    a2y = np.atleast_2d(rng.uniform(0, 1200, N))
    dens_y = build_exp_tens(
        [a1y, a2y], None,
        [10.0, 10.0], [1, 1], 
        [False, False], [True, True], [1200.0, 1200.0],
        verbose=False,
    )
    # Force orbit: dispatcher would route to pairwise at r_max=1 by default.
    ip_xy_o, ip_xx_o, ip_yy_o = _cos_sim_exp_tens_ma_orbit(dens_x, dens_y)
    cos_orbit = ip_xy_o / np.sqrt(ip_xx_o * ip_yy_o)
    cos_pw = cos_sim_exp_tens(dens_x, dens_y, method='bulger', verbose=False)
    assert abs(cos_orbit - cos_pw) < 1e-10


# ----------------------------------------------------------------------
# Ragged K_{a,n} (NaN-padded p_attr) still produces correct results
# ----------------------------------------------------------------------


def test_nan_in_p_attr_low_k_margin_routes_pairwise():
    """At K_a close to r, the dispatcher routes to pairwise via the
    K-vs-r precision guard (independent of NaN); default and explicit
    pairwise must agree.
    """
    rng = np.random.default_rng(seed=42)
    N = 5
    K = 3
    # Build pitch matrix with one event having only K-1 valid values.
    pitch_x = rng.uniform(0, 1200, (K, N))
    pitch_x[2, 1] = np.nan  # Event 1 has only 2 valid values
    pitch_y = rng.uniform(0, 1200, (K, N))
    weights_x = np.ones_like(pitch_x)
    weights_x[2, 1] = np.nan
    weights_y = np.ones_like(pitch_y)
    dens_x = build_exp_tens(
        [pitch_x], [weights_x],
        [12.0], [2], 
        [True], [True], [1200.0],
        verbose=False,
    )
    dens_y = build_exp_tens(
        [pitch_y], [weights_y],
        [12.0], [2], 
        [True], [True], [1200.0],
        verbose=False,
    )
    # Sanity: at least one density carries NaN in p_attr.
    assert any(np.isnan(M).any() for M in dens_x.p_attr)
    cos_default = cos_sim_exp_tens(dens_x, dens_y, verbose=False)
    cos_pw = cos_sim_exp_tens(dens_x, dens_y, method='bulger', verbose=False)
    assert cos_default == cos_pw


def test_nan_in_p_attr_orbit_matches_pairwise_via_hybrid():
    """At healthy K_a (K - r >= 2), ragged events trigger the per-pair
    safe/unsafe partition inside the orbit wrapper. Orbit and pairwise
    must agree to high precision on the ragged input.

    This is the killer test for the hybrid: it picks K=8, r=3 so the
    slab-level K-vs-r guard passes, then injects NaN entries to make
    individual events unsafe (K_eff - r < 2). The Möbius method then
    routes those events to direct enumeration while keeping safe
    events on the vectorised orbit; both submatrices stitch into a
    correct full IP matrix.
    """
    rng = np.random.default_rng(seed=2026)
    N = 6
    K = 8
    pitch_x = np.sort(rng.uniform(0, 2000, (K, N)), axis=0)
    pitch_y = np.sort(rng.uniform(0, 2000, (K, N)), axis=0)
    weights_x = np.ones_like(pitch_x)
    weights_y = np.ones_like(pitch_y)
    # Two unsafe events on each side: K_eff = 4 (-> K_eff-r=1 < 2).
    pitch_x[5:, 0] = np.nan; weights_x[5:, 0] = np.nan
    pitch_x[5:, 3] = np.nan; weights_x[5:, 3] = np.nan
    pitch_y[5:, 1] = np.nan; weights_y[5:, 1] = np.nan
    pitch_y[5:, 4] = np.nan; weights_y[5:, 4] = np.nan

    dens_x = build_exp_tens(
        [pitch_x], [weights_x],
        [25.0], [3], 
        [False], [False], [0.0],
        verbose=False,
    )
    dens_y = build_exp_tens(
        [pitch_y], [weights_y],
        [25.0], [3], 
        [False], [False], [0.0],
        verbose=False,
    )
    assert all(any(np.isnan(M).any() for M in d.p_attr)
               for d in (dens_x, dens_y))
    cos_orbit = cos_sim_exp_tens(dens_x, dens_y, method='mobius', verbose=False)
    cos_pw = cos_sim_exp_tens(dens_x, dens_y, method='bulger', verbose=False)
    assert abs(cos_orbit - cos_pw) < 1e-8


# ----------------------------------------------------------------------
# MA orbit triple has the documented relationship to pairwise triple
# ----------------------------------------------------------------------


def test_ma_orbit_pairwise_bare_ratio_is_consistent():
    """Although the bare orbit and pairwise triples differ by a
    per-attribute prefactor and an r_a! perm/comb factor, the SAME
    constant of proportionality applies to all three triples (xy, xx,
    yy). So orbit_xy / pairwise_xy == orbit_xx / pairwise_xx ==
    orbit_yy / pairwise_yy, modulo floating point.
    """
    rng = np.random.default_rng(seed=2026)
    dens_x = _build_ma_pitch_time(rng, N=5, K_pitch=4, r_pitch=2)
    dens_y = _build_ma_pitch_time(rng, N=5, K_pitch=4, r_pitch=2)
    ip_xy_o, ip_xx_o, ip_yy_o = _cos_sim_exp_tens_ma_orbit(dens_x, dens_y)
    ip_xy_p, ip_xx_p, ip_yy_p = _cos_sim_exp_tens_ma_pairwise(
        dens_x, dens_y, verbose=False,
    )
    r_xy = ip_xy_o / ip_xy_p
    r_xx = ip_xx_o / ip_xx_p
    r_yy = ip_yy_o / ip_yy_p
    # All three ratios must agree.
    assert abs(r_xy - r_xx) / abs(r_xy) < 1e-10
    assert abs(r_xy - r_yy) / abs(r_xy) < 1e-10
    # And the cosine must agree to floating point.
    cos_o = ip_xy_o / np.sqrt(ip_xx_o * ip_yy_o)
    cos_p = ip_xy_p / np.sqrt(ip_xx_p * ip_yy_p)
    assert abs(cos_o - cos_p) < 1e-12


# -------------------------------------------------------------------
#  Ordered attributes and the forced-Bulger feasibility guard
# -------------------------------------------------------------------


def test_ma_dispatcher_ordered_attrs_not_spuriously_infeasible():
    """Ordered attributes taken whole are one tuple each, not K!.

    An ordered ([sym] = 0) attribute's enumerated tuple set is its
    C(K, r) combinations (the perm side equals the comb side; see
    _enum_flat_attr), so at r = K it holds exactly one tuple. The
    forced-Bulger feasibility guard must therefore not raise for
    ordered attributes above the shipped orbit order, however many of
    them there are. The returned 'bulger' token names the tuple-pair
    path, which on ordered attributes is direct evaluation of the
    admitted tuples -- Bulger's combinations-vs-permutations
    organisation has no permutation expansion to exploit there.
    """
    kw = _disp_kwargs(r_max=_ORBIT_R_MAX_SHIPPED + 1,
                      K=_ORBIT_R_MAX_SHIPPED + 1, A=3)
    kw["sym_vec"] = [False, False, False]
    assert _select_ma_inner_product_method(**kw) == "bulger"


def test_ma_dispatcher_mixed_sym_still_guards_the_unordered_factor():
    """One unordered attribute at large K restores the K!/(K-r)! factor,
    so the guard fires even when the remaining attributes are ordered
    (their factors are the modest C(K, r))."""
    from mpt._tensor.dispatch import SingleImageInfeasibleError
    kw = _disp_kwargs(r_max=_ORBIT_R_MAX_SHIPPED + 1, K=12, A=3)
    kw["sym_vec"] = [True, False, False]
    with pytest.raises(SingleImageInfeasibleError, match="single-image"):
        _select_ma_inner_product_method(**kw)


def test_ordered_bound_r9_cos_sim_end_to_end():
    """A bound ordered 9-tuple density (three attributes, one read
    relative) passes auto-dispatch and matches the closed-form
    single-pair cosine: exp(-sum_a Q_a(delta_a) / (4 sigma_a^2)), with
    the relative attribute's delta projected off the common shift.
    Before the sym-aware guard this shape raised
    SingleImageInfeasibleError from a K!-per-attribute overcount.
    """
    from mpt import bind_events, build_exp_tens, cos_sim_exp_tens
    rng = np.random.default_rng(3)
    L = 9
    sig = [0.3, 0.4, 0.5]
    x = [rng.normal(0, 1, L) for _ in range(3)]
    y = [xi + rng.normal(0, 0.2, L) for xi in x]

    def dens(vals):
        p_attr = [v[None, :] for v in vals]
        p_b, w_b, sp_b = bind_events(p_attr, None, L,
                                     rel_outer=[False, True, False])
        return build_exp_tens(p_b, w_b, specs=sp_b, sigma=sig,
                              is_per=[False] * 3, period=[None] * 3,
                              verbose=False)

    got = cos_sim_exp_tens(dens(x), dens(y), verbose=False)

    q = 0.0
    for a, (xa, ya) in enumerate(zip(x, y)):
        d = xa - ya
        if a == 1:                      # relative attribute
            d = d - d.mean()
        q += float(d @ d) / (4.0 * sig[a] ** 2)
    want = np.exp(-q)
    assert abs(got - want) < 1e-10
