"""The nested relative-periodic measure is declared by ``wrap``, not chosen
by the speed dispatch.

A relative-periodic attribute admits two readings of "periodic": (A) the
minimum-image pairwise-wrap form the materialised-centres route evaluates, and
(C) the all-image transposition average the tau-grid contraction evaluates.
They agree for sigma << period and diverge as sigma approaches it. The flat
path has always let ``wrap`` decide between them; the nested path used to let
the *cost race* decide, so the same input returned one number or the other
depending on how many values or levels it happened to carry.

These tests pin the rule the nested path now follows, the same one
:func:`mpt._tensor.dispatch._select_ma_inner_product_method` enforces on the
flat path:

* ``wrap='full-image'`` (the default) declares (C); ``wrap='single-image'``
  declares (A).
* Above ``sigma/P = _orbit_sigma_over_p_threshold(ts)`` each declaration has
  exactly one carrier and that route is taken whatever it costs: the tau-grid
  under (C), the centres route under (A).
* At or below the threshold the two agree inside the truncation floor, so both
  routes serve either declaration and the cost model decides between them ---
  the same rule the flat path follows, whose ``wrap`` override is likewise
  reached only above the threshold.

They also pin what the ``method`` keyword now means on a nested density
(``'centres'`` and ``'mobius'`` no longer fall through to ``'bulger'``), the
memoisation of the multi-attribute self inner products, and the agreement of
``method='bulger'`` with the contraction that retires the claim that the
joint-tuple enumeration mis-shapes a nested attribute.
"""
import numpy as np
import pytest

import mpt
from mpt import build_exp_tens, cos_sim_exp_tens
from mpt._tensor import cosine as _cos
from mpt._tensor.cosine import _LAST_NESTED_ROUTES, _nested_attr_route
from mpt._tensor.dispatch import _orbit_sigma_over_p_threshold

P = 12.0
#: Grouping of nine values into three units of three: large enough that the
#: cost race sends it to the tau-grid on its own at some shapes, small enough
#: to run the materialised centres for comparison at every point.
TAGS = np.repeat(np.arange(3), 3)
SPEC = dict(r=[2, 2], sym=[True, True], tags=TAGS, rel=[0, 1])
#: A smaller nested attribute the cost race prefers to keep on the centres
#: route, so the measure rule is what moves it, not the price.
TAGS_S = np.repeat(np.arange(2), 2)
SPEC_S = dict(r=[2, 2], sym=[True, True], tags=TAGS_S, rel=[0, 1])

_RNG = np.random.default_rng(20240904)
_VX = np.sort(_RNG.uniform(0.0, P, 9))
_VY = np.sort(_RNG.uniform(0.0, P, 9))
_VX_S = np.sort(_RNG.uniform(0.0, P, 4))
_VY_S = np.sort(_RNG.uniform(0.0, P, 4))

SOPS = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3]


def _dens(values, sigma, spec=SPEC, wrap=None, extra=None):
    """A nested relative-periodic density, optionally tensored with a flat
    absolute attribute (``extra``) to exercise the multi-attribute path."""
    v = np.asarray(values, float)
    p = [v.reshape(-1, 1) if v.ndim == 1 else v]
    specs = [dict(spec)]
    sig = [sigma]
    per = [True]
    period = [P]
    wraps = [wrap or 'full-image']
    if extra is not None:
        p.append(np.asarray(extra, float).reshape(1, -1))
        specs.append(dict(r=1, rel=False, sym=True))
        sig.append(1.0)
        per.append(False)
        period.append(0.0)
        wraps.append('full-image')
    return build_exp_tens(p, None, specs=specs, sigma=sig, is_per=per,
                          period=period, wrap=wraps, verbose=False)


@pytest.fixture(autouse=True)
def _quiet():
    mpt.set_default(show_hints=False)
    yield
    mpt.reset_defaults()


def _forced(monkeypatch, route):
    """Pin every nested attribute to ``route``, bypassing the measure rule
    and the cost race, so a test can name a reference measure."""
    monkeypatch.setattr(_cos, "_nested_attr_route",
                        lambda dx, dy, a, **kw: route)


# --- the measure rule -------------------------------------------------


@pytest.mark.parametrize("ts", [6.0, float('inf')])
@pytest.mark.parametrize("spec,vx,vy", [
    (SPEC, _VX, _VY), (SPEC_S, _VX_S, _VY_S),
])
def test_auto_is_the_all_image_measure_above_the_threshold(monkeypatch, ts,
                                                           spec, vx, vy):
    """Above the sigma/P threshold ``auto`` equals the tau-grid reference
    exactly --- the centres route is not admissible there at any price."""
    mpt.set_default(truncation_sigmas=ts)
    limit = _orbit_sigma_over_p_threshold(ts)
    for sop in [s for s in SOPS if s > limit]:
        auto = cos_sim_exp_tens(_dens(vx, sop * P, spec),
                                _dens(vy, sop * P, spec), verbose=False)
        with pytest.MonkeyPatch.context() as mp:
            _forced(mp, "taugrid")
            ref = cos_sim_exp_tens(_dens(vx, sop * P, spec),
                                   _dens(vy, sop * P, spec), verbose=False)
        assert auto == pytest.approx(ref, abs=1e-12), f"sigma/P = {sop}"


@pytest.mark.parametrize("ts", [6.0, float('inf')])
def test_below_the_threshold_the_two_routes_agree_within_the_floor(ts):
    """Below the threshold either route may serve the full-image measure,
    which is admissible exactly because they agree inside the floor."""
    from mpt._defaults import truncation_floor
    mpt.set_default(truncation_sigmas=ts)
    floor = truncation_floor(ts)
    limit = _orbit_sigma_over_p_threshold(ts)
    for sop in [s for s in SOPS if s <= limit]:
        vals = {}
        for route in ("centres", "taugrid"):
            with pytest.MonkeyPatch.context() as mp:
                _forced(mp, route)
                vals[route] = cos_sim_exp_tens(
                    _dens(_VX, sop * P), _dens(_VY, sop * P), verbose=False)
        assert abs(vals["centres"] - vals["taugrid"]) <= max(floor, 1e-9), \
            f"sigma/P = {sop}: {vals}"


@pytest.mark.parametrize("method", ["auto", "contract"])
@pytest.mark.parametrize("extra", [None, [0.3, 1.1, 2.0]])
def test_sigma_over_p_sweep_tracks_one_measure(method, extra):
    """The swept value follows the all-image measure at every point.

    Testing the step sizes directly would not separate a route flip from the
    genuine curvature of the cosine, which is steep at small sigma/P. What
    the rule guarantees is stronger and exactly checkable: the value equals
    the tau-grid (all-image) reference throughout --- exactly above the
    threshold, and to within the truncation floor below it, where the
    minimum-image centres route is admitted. A cost-driven flip broke that by
    ~1e-3 to ~1e-2 wherever the shape crossed the price crossover.

    ``extra`` runs the same sweep through the multi-attribute path.
    """
    from mpt._defaults import truncation_floor
    ex_x = extra
    ex_y = None if extra is None else [v + 0.2 for v in extra]
    if extra is None:
        vx, vy = _VX, _VY
    else:
        vx = np.tile(_VX.reshape(-1, 1), (1, len(extra)))
        vy = np.tile(_VY.reshape(-1, 1), (1, len(extra)))
    tol = max(truncation_floor(None), 1e-9)
    for s in np.linspace(0.01, 0.30, 25):
        got = cos_sim_exp_tens(_dens(vx, s * P, extra=ex_x),
                               _dens(vy, s * P, extra=ex_y),
                               method=method, verbose=False)
        with pytest.MonkeyPatch.context() as mp:
            _forced(mp, "taugrid")
            ref = cos_sim_exp_tens(_dens(vx, s * P, extra=ex_x),
                                   _dens(vy, s * P, extra=ex_y),
                                   verbose=False)
        assert abs(got - ref) <= tol, f"sigma/P = {s:.4f}: {got} vs {ref}"


@pytest.mark.parametrize("sop", [0.05, 0.2, 0.4])
def test_single_image_above_the_threshold_takes_the_centres_route(sop):
    """Above the threshold (A) has exactly one carrier, so ``single-image``
    pins the centres route however the cost model would price it."""
    assert sop > _orbit_sigma_over_p_threshold(None)
    dx = _dens(_VX, sop * P, wrap='single-image')
    dy = _dens(_VY, sop * P, wrap='single-image')
    assert _nested_attr_route(dx, dy, 0) == "centres"


@pytest.mark.parametrize("sop", [0.005, 0.01, 0.02])
def test_single_image_below_the_threshold_admits_either_route(sop):
    """Below the threshold the two readings agree inside the floor, so
    ``single-image`` is raced exactly as ``full-image`` is --- the same rule
    the flat path follows, whose ``wrap`` override is reached only above the
    threshold.

    What is pinned is that the choice is free and cannot change the answer:
    whichever route the price picks, the value is the minimum-image (centres)
    value to within the truncation floor.
    """
    from mpt._defaults import truncation_floor
    assert sop <= _orbit_sigma_over_p_threshold(None)
    dx = _dens(_VX, sop * P, wrap='single-image')
    dy = _dens(_VY, sop * P, wrap='single-image')
    from mpt._tensor.cosine import _nested_admissible_routes
    assert _nested_admissible_routes(dx, dy, 0) == ["centres", "taugrid"]
    assert _nested_attr_route(dx, dy, 0) in ("centres", "taugrid")

    auto = cos_sim_exp_tens(_dens(_VX, sop * P, wrap='single-image'),
                            _dens(_VY, sop * P, wrap='single-image'),
                            verbose=False)
    with pytest.MonkeyPatch.context() as mp:
        _forced(mp, "centres")
        ref = cos_sim_exp_tens(_dens(_VX, sop * P, wrap='single-image'),
                               _dens(_VY, sop * P, wrap='single-image'),
                               verbose=False)
    assert abs(auto - ref) <= max(truncation_floor(None), 1e-9)


def test_forced_centres_above_the_threshold_raises():
    sop = 0.2
    assert sop > _orbit_sigma_over_p_threshold(6.0)
    with pytest.raises(ValueError, match="single-image"):
        cos_sim_exp_tens(_dens(_VX, sop * P), _dens(_VY, sop * P),
                         method='centres', verbose=False)


def test_forced_centres_below_the_threshold_is_honoured():
    sop = 0.01
    cos_sim_exp_tens(_dens(_VX, sop * P), _dens(_VY, sop * P),
                     method='centres', verbose=False)
    assert _LAST_NESTED_ROUTES == ["centres"]


# --- method keywords on a nested density ------------------------------


@pytest.mark.parametrize("method,expect", [
    ('auto', None), ('contract', None), ('mobius', None),
    ('centres', 'centres'),
])
def test_centres_and_mobius_no_longer_fall_through_to_bulger(method, expect):
    """Both used to reach ``chosen = 'bulger'`` silently. They now name
    routes of the contraction plan, which the route hook records."""
    sop = 0.01
    cos_sim_exp_tens(_dens(_VX, sop * P), _dens(_VY, sop * P),
                     method=method, verbose=False)
    assert len(_LAST_NESTED_ROUTES) == 1
    if expect is not None:
        assert _LAST_NESTED_ROUTES == [expect]


def test_the_nested_route_is_announced(capsys):
    mpt.set_default(show_hints=True)
    cos_sim_exp_tens(_dens(_VX, 0.2 * P), _dens(_VY, 0.2 * P), verbose=False)
    assert "chose 'contract' path" in capsys.readouterr().out


def test_contract_is_rejected_on_a_flat_density():
    d = build_exp_tens(np.array([0.0, 4.0, 7.0]), None, 1.0, 2, False, False,
                       0.0, verbose=False)
    with pytest.raises(ValueError, match="nested"):
        cos_sim_exp_tens(d, d, method='contract', verbose=False)


# --- memoisation ------------------------------------------------------


def test_ma_nested_self_inner_products_are_memoised():
    """The multi-attribute nested path memoises <X,X> and <Y,Y> on their
    densities, as the one-attribute nested plan does: a second call against the
    same pair (or a sweep against one prototype) pays for the cross term
    alone."""
    extra_x, extra_y = [0.3, 1.1, 2.0], [0.5, 1.3, 2.2]
    vx = np.tile(_VX.reshape(-1, 1), (1, 3))
    vy = np.tile(_VY.reshape(-1, 1), (1, 3))
    dx = _dens(vx, 0.01 * P, extra=extra_x)
    dy = _dens(vy, 0.01 * P, extra=extra_y)
    assert not dx._self_ip_cache and not dy._self_ip_cache
    first = cos_sim_exp_tens(dx, dy, verbose=False)
    keys_x = [k for k in dx._self_ip_cache if k[0] == "contract_ma"]
    keys_y = [k for k in dy._self_ip_cache if k[0] == "contract_ma"]
    assert len(keys_x) == 1 and len(keys_y) == 1
    # The cached values are consumed, not merely stored: a second call with
    # the cache warm returns the same number.
    assert cos_sim_exp_tens(dx, dy, verbose=False) == pytest.approx(first,
                                                                   abs=0.0)


def test_ma_memo_key_separates_the_routes():
    """A value taken under one route (hence one measure) is never reused
    under another."""
    extra = [0.3, 1.1, 2.0]
    vx = np.tile(_VX.reshape(-1, 1), (1, 3))
    vy = np.tile(_VY.reshape(-1, 1), (1, 3))
    dx = _dens(vx, 0.01 * P, extra=extra)
    dy = _dens(vy, 0.01 * P, extra=extra)
    with pytest.MonkeyPatch.context() as mp:
        _forced(mp, "centres")
        cos_sim_exp_tens(dx, dy, verbose=False)
    with pytest.MonkeyPatch.context() as mp:
        _forced(mp, "taugrid")
        cos_sim_exp_tens(dx, dy, verbose=False)
    routes = {k[3][0][2] for k in dx._self_ip_cache if k[0] == "contract_ma"}
    assert routes == {"centres", "taugrid"}


# --- the retired "bulger mis-shapes a nested attribute" claim ---------


_BULGER_CASES = [
    # label, rel on the outer level, is_per, period, sigma
    ("abs non-periodic", 0, False, 0.0, 2.0),
    ("rel non-periodic", 1, False, 0.0, 2.0),
    ("abs periodic", 0, True, P, 0.24),
    ("rel periodic", 1, True, P, 0.24),
]


@pytest.mark.parametrize("label,rel,is_per,period,sigma", _BULGER_CASES)
def test_bulger_agrees_with_the_contraction_on_an_ma_nested_density(
        label, rel, is_per, period, sigma):
    """``method='bulger'`` on a multi-attribute nested density agrees with
    the contraction to floating point, in every mode, at a sigma/P where the
    two measures coincide.

    This retires the claim that the joint-tuple enumeration mis-shapes a
    nested attribute's per-event tuples in the multi-attribute tensor build.
    The fallback policy is unchanged --- the contraction still always returns
    its triple rather than deferring to the enumeration --- but the reason is
    cost and per-attribute measure control, not a shape bug.
    """
    mpt.set_default(truncation_sigmas=float('inf'))
    rng = np.random.default_rng(5)
    n = 2
    v1x = np.sort(rng.uniform(0.0, P, (4, n)), axis=0)
    v1y = v1x + 0.4
    v2x = rng.uniform(0.0, 5.0, (1, n))
    v2y = v2x + 0.3
    specs = [dict(r=[2, 2], sym=[True, True],
                  tags=np.array([0, 0, 1, 1]), rel=[0, rel]),
             dict(r=1, rel=False, sym=True)]
    kw = dict(specs=specs, sigma=[sigma, 1.0], is_per=[is_per, False],
              period=[period, 0.0], verbose=False)

    def pair():
        return (build_exp_tens([v1x, v2x], None, **kw),
                build_exp_tens([v1y, v2y], None, **kw))

    dx, dy = pair()
    contracted = cos_sim_exp_tens(dx, dy, verbose=False)
    dx, dy = pair()
    enumerated = cos_sim_exp_tens(dx, dy, method='bulger', verbose=False)
    assert contracted == pytest.approx(enumerated, rel=1e-9, abs=1e-12)
