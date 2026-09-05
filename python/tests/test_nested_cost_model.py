"""The nested inner product is routed by a priced cost model, not by counts.

The nested path used to choose its per-attribute route by comparing raw
analytic operation counts --- materialised kernel entries against quadrature
nodes times contraction work --- as though a kernel entry and a unit of
contraction work cost the same, and it never considered the joint-tuple
enumeration on price at all. It now prices every candidate in milliseconds
from the fitted laws of :mod:`mpt._tensor._nested_cost`, guards the
materialising route with the same working-set budget the flat selector uses,
and compares the whole contraction plan against the enumeration the way
:func:`mpt._tensor.dispatch._select_ma_inner_product_method` compares the
Möbius method against Bulger's.

What these tests pin is the *cost model*. The measure rule is pinned by
``tests/test_nested_measure_rule.py`` and is deliberately upstream of
everything here: the prices are stubbed to absurd values throughout, and no
stub is allowed to move a route that carries a different measure.
"""
import numpy as np
import pytest

import mpt
from mpt import build_exp_tens, cos_sim_exp_tens
from mpt._tensor import _nested_cost as _nc
from mpt._tensor import cosine as _cos
from mpt._tensor.cosine import (
    _LAST_NESTED_COSTS,
    _nested_admissible_routes,
    _nested_attr_route,
)
from mpt._tensor.dispatch import (
    _CENTRES_WORKING_SET_SOFT_BUDGET,
    _orbit_sigma_over_p_threshold,
)
from mpt._tensor.explain import explain_dispatch

P = 12.0


@pytest.fixture(autouse=True)
def _quiet():
    mpt.set_default(show_hints=False)
    yield
    mpt.reset_defaults()


def _dens(values, sigma, *, r_levels=(2, 2), sym=(1, 1), chord=3,
          is_rel=True, is_per=True, wrap='full-image', extra=None,
          extra_r=1):
    v = np.asarray(values, float)
    if v.ndim == 1:
        v = v.reshape(-1, 1)
    ngroup = v.shape[0] // chord
    tags = np.repeat(np.arange(ngroup), chord)
    spec = dict(r=list(r_levels), sym=[bool(s) for s in sym], tags=tags,
                rel=([0] * (len(r_levels) - 1) + [1] if is_rel else None))
    p = [v]
    specs = [spec]
    sig = [sigma]
    per = [bool(is_per)]
    period = [P if is_per else 0.0]
    wraps = [wrap]
    if extra is not None:
        ex = np.asarray(extra, float)
        p.append(ex.reshape(1, -1) if ex.ndim == 1 else ex)
        specs.append(dict(r=extra_r, rel=False, sym=True))
        sig.append(1.0)
        per.append(False)
        period.append(0.0)
        wraps.append('full-image')
    return build_exp_tens(p, None, specs=specs, sigma=sig, is_per=per,
                          period=period, wrap=wraps, verbose=False)


_RNG = np.random.default_rng(7)
_VX = np.sort(_RNG.uniform(0.0, P, 9))
_VY = np.sort(_RNG.uniform(0.0, P, 9))


def _stub_prices(monkeypatch, prices):
    """Replace the fitted laws by a fixed price per route."""
    monkeypatch.setattr(
        _nc, "nested_route_cost_ms",
        lambda route, order, term, n_matrices=3: float(prices[route]))


# --- the working-set guard ------------------------------------------


def _big_pair(N):
    """A relative-non-periodic nested pair whose centres bundle is huge.

    Twelve values per chord, ten chords, read two-at-two: the perm side is
    ``2!^10 * 2! * e_2(C(12, 2), ...)`` tuples per event, which at four
    events puts the materialised centres array well past the soft budget
    while every count this test reads stays analytic --- nothing here
    enumerates a tuple.
    """
    rng = np.random.default_rng(11)
    vx = np.sort(rng.uniform(0.0, 40.0, (120, N)), axis=0)
    vy = np.sort(rng.uniform(0.0, 40.0, (120, N)), axis=0)
    return (_dens(vx, 0.5, chord=12, is_per=False),
            _dens(vy, 0.5, chord=12, is_per=False))


def test_the_memory_guard_diverts_a_huge_shape(monkeypatch):
    """Above the soft budget the materialising route is not taken, however
    cheaply it is priced."""
    dx, dy = _big_pair(4)
    ws = _nc.nested_centres_working_set_bytes(dx, dy, 0)
    assert ws > _CENTRES_WORKING_SET_SOFT_BUDGET
    # Price centres as free and the contraction as ruinous: only the guard
    # can move the route now.
    _stub_prices(monkeypatch, {"centres": 1e-6, "contract_relnonper": 1e9})
    route, cost, prices, _info = _nc.price_nested_attr(
        dx, dy, 0, ["centres", "contract_relnonper"])
    assert prices["centres"] == float("inf")
    assert route == "contract_relnonper"
    assert _nested_attr_route(dx, dy, 0) == "contract_relnonper"


def test_a_small_shape_is_under_the_budget_and_stays_priced(monkeypatch):
    """The guard is a guard, not a policy: below the budget the price
    decides."""
    dx, dy = _big_pair(1)
    small_x, small_y = _dens(_VX, 0.1, is_per=False), _dens(_VY, 0.1,
                                                            is_per=False)
    assert (_nc.nested_centres_working_set_bytes(small_x, small_y, 0)
            < _CENTRES_WORKING_SET_SOFT_BUDGET)
    _stub_prices(monkeypatch, {"centres": 1e-6, "contract_relnonper": 1e9})
    assert _nested_attr_route(small_x, small_y, 0) == "centres"


# --- the per-attribute race, below the threshold ---------------------


def _below_threshold_sigma():
    return 0.5 * _orbit_sigma_over_p_threshold(
        mpt.get_default("truncation_sigmas")) * P


def test_full_image_below_the_threshold_races_the_two_routes(monkeypatch):
    """Both routes carry the full-image measure inside the truncation floor
    there, so the cheaper one is taken --- either way round."""
    sigma = _below_threshold_sigma()
    dx, dy = _dens(_VX, sigma), _dens(_VY, sigma)
    assert _nested_admissible_routes(dx, dy, 0) == ["centres", "taugrid"]
    _stub_prices(monkeypatch, {"centres": 100.0, "taugrid": 1.0})
    assert _nested_attr_route(dx, dy, 0) == "taugrid"
    _stub_prices(monkeypatch, {"centres": 1.0, "taugrid": 100.0})
    assert _nested_attr_route(dx, dy, 0) == "centres"


def test_above_the_threshold_the_price_cannot_reach_the_centres_route(
        monkeypatch):
    """Above the threshold the full-image measure admits the tau-grid alone,
    so pricing centres at nothing changes nothing."""
    sigma = 0.3 * P
    assert 0.3 > _orbit_sigma_over_p_threshold(
        mpt.get_default("truncation_sigmas"))
    dx, dy = _dens(_VX, sigma), _dens(_VY, sigma)
    assert _nested_admissible_routes(dx, dy, 0) == ["taugrid"]
    _stub_prices(monkeypatch, {"centres": 1e-9, "taugrid": 1e9})
    assert _nested_attr_route(dx, dy, 0) == "taugrid"


@pytest.mark.parametrize("sop", [0.05, 0.2, 0.4])
def test_single_image_above_the_threshold_is_centres_at_any_price(
        monkeypatch, sop):
    """Above the threshold the minimum-image measure has one carrier, so
    pricing the tau-grid at nothing changes nothing."""
    dx = _dens(_VX, sop * P, wrap='single-image')
    dy = _dens(_VY, sop * P, wrap='single-image')
    assert _nested_admissible_routes(dx, dy, 0) == ["centres"]
    _stub_prices(monkeypatch, {"centres": 1e9, "taugrid": 1e-9})
    assert _nested_attr_route(dx, dy, 0) == "centres"


def test_single_image_below_the_threshold_is_raced(monkeypatch):
    """Below the threshold the two readings agree inside the floor, so
    ``single-image`` is priced exactly as ``full-image`` is."""
    sigma = _below_threshold_sigma()
    dx = _dens(_VX, sigma, wrap='single-image')
    dy = _dens(_VY, sigma, wrap='single-image')
    assert _nested_admissible_routes(dx, dy, 0) == ["centres", "taugrid"]
    _stub_prices(monkeypatch, {"centres": 100.0, "taugrid": 1.0})
    assert _nested_attr_route(dx, dy, 0) == "taugrid"
    _stub_prices(monkeypatch, {"centres": 1.0, "taugrid": 100.0})
    assert _nested_attr_route(dx, dy, 0) == "centres"


def test_the_guard_never_moves_single_image_above_the_threshold():
    """The soft budget diverts the centres route only where another route
    carries the same measure; above the threshold under ``single-image``
    there is none, so the guard stands down."""
    # The same shape as the guard test's, declared relative-periodic
    # single-image above the threshold.
    rng = np.random.default_rng(11)
    vx = np.sort(rng.uniform(0.0, P, (120, 4)), axis=0)
    vy = np.sort(rng.uniform(0.0, P, (120, 4)), axis=0)
    bx = _dens(vx, 0.3 * P, chord=12, wrap='single-image')
    by = _dens(vy, 0.3 * P, chord=12, wrap='single-image')
    assert (_nc.nested_centres_working_set_bytes(bx, by, 0)
            > _CENTRES_WORKING_SET_SOFT_BUDGET)
    assert _nested_admissible_routes(bx, by, 0) == ["centres"]
    route, _c, prices, _i = _nc.price_nested_attr(bx, by, 0, ["centres"])
    assert route == "centres" and prices["centres"] < float("inf")


def test_an_absolute_attribute_stays_on_the_contraction(monkeypatch):
    dx = _dens(_VX, 0.2, is_rel=False, is_per=False)
    dy = _dens(_VY, 0.2, is_rel=False, is_per=False)
    assert _nested_admissible_routes(dx, dy, 0) == ["contract"]
    _stub_prices(monkeypatch, {"contract": 1e9, "centres": 1e-9})
    assert _nested_attr_route(dx, dy, 0) == "contract"


# --- the plan against the enumeration --------------------------------


def _cheap_enumeration(monkeypatch, factor=1e-6):
    """Price the enumeration at ``factor`` of everything else."""
    monkeypatch.setattr(
        _nc, "nested_route_cost_ms",
        lambda route, order, term, n_matrices=3:
            (factor if route == "bulger" else 1.0))


def test_the_enumeration_is_taken_where_it_is_priced_cheaper(monkeypatch):
    sigma = _below_threshold_sigma()
    dx, dy = _dens(_VX, sigma), _dens(_VY, sigma)
    reference = cos_sim_exp_tens(_dens(_VX, sigma), _dens(_VY, sigma),
                                 method='bulger', verbose=False)
    _cheap_enumeration(monkeypatch)
    got = cos_sim_exp_tens(dx, dy, verbose=False)
    assert _LAST_NESTED_COSTS["chosen"] == "bulger"
    assert _LAST_NESTED_COSTS["enum_ms"] < _LAST_NESTED_COSTS["plan_ms"]
    assert got == pytest.approx(reference, rel=1e-9, abs=1e-12)


def test_the_plan_is_kept_where_the_enumeration_is_priced_dearer(monkeypatch):
    sigma = _below_threshold_sigma()
    dx, dy = _dens(_VX, sigma), _dens(_VY, sigma)
    monkeypatch.setattr(
        _nc, "nested_route_cost_ms",
        lambda route, order, term, n_matrices=3:
            (1e9 if route == "bulger" else 1.0))
    cos_sim_exp_tens(dx, dy, verbose=False)
    assert _LAST_NESTED_COSTS["chosen"] == "contract"
    assert _cos._LAST_NESTED_ROUTES == ["centres"] or \
        _cos._LAST_NESTED_ROUTES == ["taugrid"]


def test_a_near_tie_keeps_the_plan(monkeypatch):
    """The enumeration has to be decisively cheaper, not marginally: see
    ``_NESTED_ENUM_SAFETY``."""
    sigma = _below_threshold_sigma()
    dx, dy = _dens(_VX, sigma), _dens(_VY, sigma)
    _cheap_enumeration(monkeypatch, factor=1.0 / _nc._NESTED_ENUM_SAFETY)
    cos_sim_exp_tens(dx, dy, verbose=False)
    assert _LAST_NESTED_COSTS["chosen"] == "contract"


def test_the_enumeration_is_inadmissible_above_the_threshold(monkeypatch):
    """It computes the minimum-image reading, so no price buys it there."""
    sigma = 0.3 * P
    dx, dy = _dens(_VX, sigma), _dens(_VY, sigma)
    assert not _cos._nested_enumeration_admissible(dx, dy)
    _cheap_enumeration(monkeypatch)
    cos_sim_exp_tens(dx, dy, verbose=False)
    assert _LAST_NESTED_COSTS["chosen"] == "contract"
    assert _LAST_NESTED_COSTS["enum_ms"] == float("inf")


def test_a_forced_method_is_never_diverted(monkeypatch):
    sigma = _below_threshold_sigma()
    _cheap_enumeration(monkeypatch)
    cos_sim_exp_tens(_dens(_VX, sigma), _dens(_VY, sigma),
                     method='contract', verbose=False)
    assert _cos._LAST_NESTED_ROUTES != []


def vx0():
    return np.tile(_VX.reshape(-1, 1), (1, 3))


def vy0():
    return np.tile(_VY.reshape(-1, 1), (1, 3))


def test_the_multi_attribute_plan_is_priced_too(monkeypatch):
    sigma = _below_threshold_sigma()
    # A flat *symmetric* r = 2 companion: the flat model absorbs an r = 1
    # attribute into its base and prices it at nothing, so an r = 1
    # companion could not show that companions are priced at all.
    ex = np.array([[0.3, 1.1, 2.0], [0.7, 1.5, 2.6], [1.2, 2.1, 3.3]])
    ey = ex + 0.2
    kw = dict(extra_r=2)
    reference = cos_sim_exp_tens(
        _dens(vx0(), sigma, extra=ex, **kw),
        _dens(vy0(), sigma, extra=ey, **kw),
        method='bulger', verbose=False)
    _cheap_enumeration(monkeypatch)
    got = cos_sim_exp_tens(_dens(vx0(), sigma, extra=ex, **kw),
                           _dens(vy0(), sigma, extra=ey, **kw),
                           verbose=False)
    assert _LAST_NESTED_COSTS["chosen"] == "bulger"
    # The flat companion is priced alongside the nested attribute.
    assert _LAST_NESTED_COSTS["detail"]["flat"] > 0.0
    assert got == pytest.approx(reference, rel=1e-9, abs=1e-12)


# --- the terms and the skip flags ------------------------------------


def test_the_centres_term_counts_the_restricted_side():
    """The centres route forms one ``(m_comb_X, m_perm_Y)`` array per event
    pair where Bulger's restriction holds, and the term says so."""
    sigma = _below_threshold_sigma()
    dx, dy = _dens(_VX, sigma), _dens(_VY, sigma)
    terms, info = _nc.nested_attr_terms(dx, dy, 0)
    assert info["restricted_x"] and info["restricted_y"]
    expected = 3.0 * info["m_comb_x"] * info["m_perm_y"]
    assert terms["centres"] == pytest.approx(expected)


def test_a_skipped_self_inner_product_is_not_priced():
    sigma = _below_threshold_sigma()
    dx, dy = _dens(_VX, sigma), _dens(_VY, sigma)
    full, _ = _nc.nested_attr_terms(dx, dy, 0)
    cross, _ = _nc.nested_attr_terms(dx, dy, 0, skip_xx=True, skip_yy=True)
    for route in ("centres", "taugrid"):
        assert cross[route] < full[route]


# --- the report -------------------------------------------------------


def test_explain_dispatch_shows_the_prices():
    sigma = _below_threshold_sigma()
    dx, dy = _dens(_VX, sigma), _dens(_VY, sigma)
    exp = explain_dispatch(dx, dy)
    text = str(exp)
    assert "nested cost model" in text
    # Both candidates carry a predicted time, and the per-attribute prices
    # of every admissible route are quoted.
    by_name = {r.name: r for r in exp.routes}
    assert by_name["contract"].predicted_ms > 0.0
    assert by_name["bulger"].predicted_ms > 0.0
    assert "centres" in by_name["contract"].reason
    assert "taugrid" in by_name["contract"].reason
    assert "ms" in by_name["contract"].reason


def test_explain_dispatch_reports_an_inadmissible_enumeration():
    dx, dy = _dens(_VX, 0.3 * P), _dens(_VY, 0.3 * P)
    text = str(explain_dispatch(dx, dy))
    assert "minimum-image measure" in text
