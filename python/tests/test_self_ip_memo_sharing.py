"""What the routes' self inner products may and may not share.

The multi-attribute inner product is computed by several routes -- Bulger's
joint-tuple enumeration, the unrestricted tuple-centres enumeration, the
per-attribute Möbius matrices (themselves either a translation grid or the
tuple-centres closed form), and, for a nested attribute, the per-level
contraction -- and each memoises the self inner products ``<X,X>`` and
``<Y,Y>`` on the densities so a repeated call or a sweep pays for the cross
term alone.

Two questions follow, and they have different answers.

*Are the routes' bare triples on the same scale?* Yes, and by an exactly
known constant per attribute; :func:`test_route_scale_identities` pins each
one. *May a memo written by one route therefore be read by another, rescaled?*
No: each route applies the truncation budget to its own arrays, so after the
exact rescaling the routes hold different numbers rather than the same number
in different units. :func:`test_routes_do_not_agree_to_working_precision`
measures that gap, and the cache keys stay route-specific because of it --- a
call's answer must not depend on which route happened to warm the memo.

What *is* shared is the pricing. A selector comparing two routes must price
both against the same statement of what a self inner product costs this call,
or the first route to run is priced at one matrix and its rival at three, and
that first choice locks in however cheap the rival becomes once warm. The
lock-in tests below pin the fix and the value-invariance it preserves.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import mpt
from mpt import build_exp_tens, cos_sim_exp_tens
from mpt._tensor.cosine import (
    _SELF_IP_ROUTES,
    _cos_sim_exp_tens_ma_centres,
    _cos_sim_exp_tens_ma_pairwise,
    _nested_attr_matrix,
    _nested_attr_plan,
    _nested_self_ip_skip_flags,
    _self_ip_cache_key,
    _self_ip_memoised,
)
from mpt._tensor._mobius_inner import (
    _closed_form_attr_centres,
    _closed_form_attr_matrix_from,
    _ma_per_attr_inner_matrix,
    _nested_orbit_mult,
)

P = 12.0


def _flat(K, N, r, is_rel, is_per, sigma, seed, wrap="full-image",
          span=20.0):
    rng = np.random.default_rng(seed)
    hi = P if is_per else span
    p = np.sort(rng.uniform(0.0, hi, size=(K, N)), axis=0)
    return build_exp_tens([p], [np.ones((K, N))], [sigma], [r],
                          [is_rel], [is_per], [P], verbose=False,
                          wrap=[wrap])


def _nested(n_values, sigma, is_rel, is_per, seed, r_levels=(2, 2),
            wrap="full-image"):
    rng = np.random.default_rng(seed)
    hi = P if is_per else 24.0
    v = np.sort(rng.uniform(0.0, hi, n_values))
    tags = np.repeat(np.arange(n_values // r_levels[-1]), r_levels[-1])
    spec = dict(r=list(r_levels), sym=[True] * len(r_levels), tags=tags,
                rel=[0] * (len(r_levels) - 1) + [1 if is_rel else 0])
    return build_exp_tens([v.reshape(-1, 1)], None, specs=[spec],
                          sigma=[sigma], is_per=[is_per], period=[P],
                          wrap=[wrap], verbose=False)


def _bulger_ip(dens):
    """``<X,X>`` on Bulger's scale, the canonical one for these identities."""
    return _cos_sim_exp_tens_ma_pairwise(dens, dens, verbose=False)[1]


# ----------------------------------------------------------------------
# 1. The scale identities
# ----------------------------------------------------------------------


@pytest.mark.parametrize("is_rel,is_per", [(False, False), (False, True),
                                           (True, False)])
@pytest.mark.parametrize("r", [2, 3])
@pytest.mark.parametrize("sigma", [0.3, 1.0])
def test_route_scale_identities(is_rel, is_per, r, sigma):
    """Each route's bare self inner product, over Bulger's, is its constant.

    Per attribute, against Bulger's perm-versus-comb enumeration:

    * the Möbius per-attribute matrix carries the single-multiset Gaussian
      prefactor and the full symmetric orbit, giving
      ``r! (sigma sqrt(pi))^r`` in an absolute mode; in relative-non-periodic
      the rigid translation is marginalised, which trades one Gaussian factor
      for the quotient's Jacobian: ``r! (sigma sqrt(pi))^(r-1) sqrt(r)``.
    * the tuple-centres closed form drops the Gaussian prefactor entirely and
      keeps only the orbit, giving ``r!``.
    * the unrestricted centres enumeration is perm against perm where Bulger
      is perm against comb: ``r!`` again.

    Checked at the suite's ``truncation_sigmas=inf`` baseline, where the
    routes' truncation treatments do not separate them (see
    :func:`test_routes_do_not_agree_to_working_precision` for what happens at
    the shipped default).
    """
    d = _flat(6, 3, r, is_rel, is_per, sigma, seed=11)
    base = _bulger_ip(d)

    grid = _ma_per_attr_inner_matrix(
        d.p_attr[0], d.w[0], d.p_attr[0], d.w[0], sigma, r,
        is_rel, is_per, P, wrap="full-image").sum()
    if is_rel:
        expect = (math.factorial(r)
                  * (sigma * math.sqrt(math.pi)) ** (r - 1)
                  * math.sqrt(r))
    else:
        expect = math.factorial(r) * (sigma * math.sqrt(math.pi)) ** r
    assert grid / base == pytest.approx(expect, rel=1e-12)

    cx = _closed_form_attr_centres(d, 0)
    closed = _closed_form_attr_matrix_from(cx, cx, None,
                                           "full-image").sum()
    assert closed / base == pytest.approx(math.factorial(r), rel=1e-12)

    d2 = _flat(6, 3, r, is_rel, is_per, sigma, seed=11)
    centres = _cos_sim_exp_tens_ma_centres(d2, d2, verbose=False)[1]
    assert centres / base == pytest.approx(math.factorial(r), rel=1e-12)


@pytest.mark.parametrize("is_rel,is_per,expect_route", [
    (False, False, "contract"),
    (False, True, "contract"),
    (True, False, "centres"),
])
def test_nested_route_scale_identities(is_rel, is_per, expect_route):
    """The nested per-level contraction is on Bulger's own scale; the nested
    centres route is on it times the wreath-product orbit order."""
    d = _nested(6, 0.8, is_rel, is_per, seed=3)
    base = _bulger_ip(d)
    route, taus = _nested_attr_plan(d, d, 0)
    assert route == expect_route
    got = _nested_attr_matrix(d, d, 0, route, taus).sum()
    mult = (1.0 if route == "contract"
            else float(_nested_orbit_mult(np.array([2, 2]),
                                          np.array([True, True]))))
    assert got / base == pytest.approx(mult, rel=1e-12)


def test_relative_periodic_grid_is_a_different_measure():
    """The tau-grid's reading is not the enumeration's, rescaled.

    The grid computes the all-image transposition average (C); the
    enumeration and the closed form compute the minimum-image reading (A).
    Below the sigma/period threshold the two sit inside the truncation floor
    of each other, but they are not the same number, and the gap grows with
    sigma/P --- so no constant relates them and no memo may cross.
    """
    gaps = []
    for sigma in (0.7, 1.5):
        d = _flat(6, 3, 2, True, True, sigma, seed=5)
        base = _bulger_ip(d)
        grid = _ma_per_attr_inner_matrix(
            d.p_attr[0], d.w[0], d.p_attr[0], d.w[0], sigma, 2,
            True, True, P, wrap="full-image").sum()
        expect = (2.0 * (sigma * math.sqrt(math.pi)) ** 1
                  * math.sqrt(2.0))
        gaps.append(abs(grid / base / expect - 1.0))
    assert gaps[0] > 1e-6           # already outside working precision
    assert gaps[1] > 20 * gaps[0]   # and growing with sigma/P


def test_routes_do_not_agree_to_working_precision():
    """Why the rescaling is not enough to share a memo.

    At the shipped default the routes' self inner products, put on one scale,
    still differ by orders more than working precision --- each applies the
    truncation budget to its own arrays. A shared memo would carry that
    difference into the returned cosine.
    """
    mpt.set_default(truncation_sigmas=6.0)   # the shipped default
    d = _flat(9, 3, 2, False, False, 0.5, seed=7, span=10.0)
    base = _bulger_ip(d)
    grid = _ma_per_attr_inner_matrix(
        d.p_attr[0], d.w[0], d.p_attr[0], d.w[0], 0.5, 2, False, False, P,
        wrap="full-image").sum()
    rescaled = grid / (2.0 * (0.5 * math.sqrt(math.pi)) ** 2)
    assert abs(rescaled / base - 1.0) > 1e-10


# ----------------------------------------------------------------------
# 2. Values stay route-keyed; the pricing flag is shared
# ----------------------------------------------------------------------


def test_memo_keys_stay_route_specific():
    """Two routes on one pair leave two entries, not one shared entry."""
    dx = _flat(6, 3, 2, False, False, 0.5, seed=1)
    dy = _flat(6, 4, 2, False, False, 0.5, seed=2)
    cos_sim_exp_tens(dx, dy, method="bulger", verbose=False)
    cos_sim_exp_tens(dx, dy, method="centres", verbose=False)
    routes = {k[0] for k in dx._self_ip_cache}
    assert routes == {"bulger", "centres"}
    b = dx._self_ip_cache[_self_ip_cache_key("bulger", None, None)]
    c = dx._self_ip_cache[_self_ip_cache_key("centres", None, None)]
    # Same quantity in different units, and deliberately stored that way.
    assert c / b == pytest.approx(2.0, rel=1e-9)


def test_self_ip_memoised_reports_any_route():
    """The pricing flag asks whether *any* route has paid, not which one."""
    dx = _flat(6, 3, 2, True, True, 0.25, seed=1)
    dy = _flat(6, 4, 2, True, True, 0.25, seed=2)
    assert not _self_ip_memoised(dx)
    cos_sim_exp_tens(dx, dy, method="mobius", verbose=False)
    assert _self_ip_memoised(dx) and _self_ip_memoised(dy)
    assert {k[0] for k in dx._self_ip_cache} <= set(_SELF_IP_ROUTES)


def test_sweep_memo_does_not_count_as_a_route_memo():
    """The sweep's own memo spares neither inner-product route any work."""
    dx = _flat(6, 3, 2, False, False, 0.5, seed=1)
    dx._self_ip_cache[("sweep", 6.0, "None")] = 1.0
    assert not _self_ip_memoised(dx)


def test_nested_skip_flags_are_shared_by_plan_and_enumeration():
    dx = _nested(6, 0.8, True, False, seed=3)
    dy = _nested(6, 0.8, True, False, seed=4)
    assert _nested_self_ip_skip_flags(dx, dy, "cosine") == (False, False)
    cos_sim_exp_tens(dx, dy, method="bulger", verbose=False)
    # The enumeration's memo now prices the contraction plan as warm too.
    assert _nested_self_ip_skip_flags(dx, dy, "cosine") == (True, True)


def test_one_sided_denominator_still_skips_xx():
    dx = _flat(6, 3, 2, False, False, 0.5, seed=1)
    dy = _flat(6, 4, 2, False, False, 0.5, seed=2)
    assert _nested_self_ip_skip_flags(dx, dy, "oneSidedDenom") == (True, False)


# ----------------------------------------------------------------------
# 3. The lock-in, and the value invariance it must not cost
# ----------------------------------------------------------------------


#: Two relative-periodic cells, chosen so that the cold cost model prefers a
#: different route on each --- a small one, whose tuple-pair count stays under
#: the Möbius side's per-attribute setup, and a larger one, where it does not.
#: *Which* route each cell draws depends on the fitted cost constants, and
#: those differ between the two languages; the tests below therefore read the
#: cold route off the call rather than naming it, and assert only the
#: invariance. Two cells are used so that a memo on the cheaper route and a
#: memo on the dearer one are both exercised in whichever language runs them.
#: The values are formula-based rather than drawn, so the MATLAB twin tests
#: the same numbers.
_LOCKIN_CELLS = ((5, 4), (9, 4))


def _lockin_values(a, b, K, N):
    i = np.arange(1, K + 1).reshape(-1, 1)
    j = np.arange(1, N + 1).reshape(1, -1)
    return np.mod(a * i ** 2 + b * j, P)


def _lockin_pair(K=5, N=4):
    def one(a, b):
        v = _lockin_values(a, b, K, N)
        return build_exp_tens([v], [np.ones_like(v)], [0.25], [2],
                              [True], [True], [P], verbose=False)
    return one(1.9, 0.4), one(2.6, 0.7)


def _routes_of(dens):
    return sorted({k[0] for k in dens._self_ip_cache})


@pytest.mark.parametrize("K,N", _LOCKIN_CELLS)
def test_cold_auto_takes_one_definite_route_on_the_lockin_cell(K, N):
    """A cold ``auto`` call leaves exactly one route's memo behind.

    The route it picks is a matter of the fitted constants and is not named
    here; what the lock-in test needs from the cell is only that the cold
    choice is definite, so that a later call can be checked against it.
    """
    dx, dy = _lockin_pair(K, N)
    cos_sim_exp_tens(dx, dy, verbose=False)
    assert len(_routes_of(dx)) == 1


@pytest.mark.parametrize("K,N", _LOCKIN_CELLS)
def test_a_memo_on_another_route_no_longer_locks_the_route_in(K, N):
    """A memo left by one route does not divert ``auto`` from its own.

    Before the pricing flags were shared, a memo made the route that wrote it
    free and left its rival priced at three matrices, so whichever route ran
    first locked itself in and the route the cold model prefers could never be
    reached again on that pair. What shared pricing guarantees is that a warm
    ``auto`` call prices both routes against the same flags, so its choice
    cannot depend on *which* route warmed the pair. The warm choice may
    legitimately differ from the cold one --- a call with both self products
    memoised is a different workload, and the Möbius side's setup floor does
    not scale with the matrix count, so a near-tie can fall the other way once
    warm --- so the reference is ``auto`` warmed by ``auto`` itself, and every
    rival-warmed call must take that route and return the cold value.
    """
    dx, dy = _lockin_pair(K, N)
    cold = cos_sim_exp_tens(dx, dy, verbose=False)
    cold_route, = _routes_of(dx)
    cos_sim_exp_tens(dx, dy, verbose=False)
    warm_routes = _routes_of(dx)
    # The reference warm route: the cold route's memo is present either way,
    # so it is the route that appears in the second call, or the cold route
    # again if none did.
    warm_route = next((r for r in warm_routes if r != cold_route), cold_route)

    others = [m for m in ("bulger", "centres", "mobius") if m != cold_route]
    assert others                      # the cell must have a rival at all
    for other in others:
        ax, ay = _lockin_pair(K, N)
        cos_sim_exp_tens(ax, ay, method=other, verbose=False)
        assert _routes_of(ax) == [other]
        got = cos_sim_exp_tens(ax, ay, verbose=False)
        # Exactly the rival's memo plus the reference route's (one entry
        # when they coincide): auto took the reference route, no other.
        assert set(_routes_of(ax)) == {other, warm_route}
        assert np.allclose(got, cold, rtol=0.0, atol=1e-13)


def test_call_order_does_not_change_the_value():
    """Whatever ran first, ``auto`` returns the number its route returns.

    Route-keyed memos are what buys this: a value computed under one route's
    truncation treatment is never consumed by another.
    """
    dx, dy = _lockin_pair()
    reference = cos_sim_exp_tens(dx, dy, verbose=False)

    orders = (
        ("mobius",),
        ("bulger",),
        ("centres",),
        ("mobius", "bulger"),
        ("bulger", "mobius"),
        ("centres", "mobius"),
    )
    for pre in orders:
        ax, ay = _lockin_pair()
        for m in pre:
            cos_sim_exp_tens(ax, ay, method=m, verbose=False)
        got = cos_sim_exp_tens(ax, ay, verbose=False)
        assert np.allclose(got, reference, rtol=0.0, atol=1e-13)


@pytest.mark.parametrize("method", ["bulger", "centres", "mobius"])
def test_forced_method_is_reproducible_whatever_warmed_the_memo(method):
    dx, dy = _lockin_pair()
    reference = cos_sim_exp_tens(dx, dy, method=method, verbose=False)
    for other in ("bulger", "centres", "mobius"):
        ax, ay = _lockin_pair()
        cos_sim_exp_tens(ax, ay, method=other, verbose=False)
        got = cos_sim_exp_tens(ax, ay, method=method, verbose=False)
        # Bit-identical: the forced route reads only its own memo.
        assert np.array_equal(got, reference)


def test_nested_call_order_does_not_change_the_value():
    dx = _nested(6, 0.8, True, False, seed=3)
    dy = _nested(6, 0.8, True, False, seed=4)
    reference = cos_sim_exp_tens(dx, dy, verbose=False)
    for pre in ("bulger", "contract"):
        ax = _nested(6, 0.8, True, False, seed=3)
        ay = _nested(6, 0.8, True, False, seed=4)
        cos_sim_exp_tens(ax, ay, method=pre, verbose=False)
        got = cos_sim_exp_tens(ax, ay, verbose=False)
        assert np.allclose(got, reference, rtol=0.0, atol=1e-13)
