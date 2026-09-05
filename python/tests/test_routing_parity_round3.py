"""Regressions from the routing-parity audit, round 3 (September 2026).

Mirror of ``tests/test_routing_parity_round3.m``. The round removed the
dead code the audit found and made ``explain_dispatch`` report the route
the call takes; each block pins one of those outcomes so the two
languages keep agreeing:

* ``explain_dispatch`` on a flat cosine builds the selector's inputs by
  the same function the call uses (the wrap vector, the per-attribute
  grid node counts, the memo flags), applies the empty-operand rule
  before the selector and the ordered-attribute rule after it --- so on
  a rel-per pair above the sigma/P threshold the report follows the
  declared wrap as the call does (it used to name the cost model's
  pick), and on an ordered pair it names Bulger's method under any
  ``method``;
* the eval cost model prices the spectral branch of the Möbius
  relative evaluator wherever that branch engages: the evaluator has no
  mode-grid size decline, so the model no longer prices one (D-6);
* the Möbius per-attribute matrix has one abs r >= 2 route: the
  safe/unsafe partition and its direct-enumeration fill are gone, and
  the batched route agrees with the enumerated reference at K = r;
* the tuple-centres closed form refuses an inner ``[rel]`` unit rather
  than carrying a block-Q form no caller can reach;
* the deleted names stay deleted (no re-export brings them back).
"""
import math
import warnings

import numpy as np
import pytest

import mpt
from mpt import build_exp_tens, cos_sim_exp_tens, explain_dispatch
from mpt._tensor import cosine as _cos
from mpt._tensor import dispatch as _disp
from mpt._tensor._mobius_inner import (_closed_form_attr_matrix_from,
                                       _ma_per_attr_inner_matrix)
from tests.references.mobius_ip_reference import inner_product_direct_abs


@pytest.fixture(autouse=True)
def _quiet():
    mpt.set_default(show_hints=False)
    warnings.simplefilter("ignore", UserWarning)
    yield
    warnings.resetwarnings()
    mpt.reset_defaults()


P = 12.0


def _flat(seed, sigma, r=2, *, is_rel=False, is_per=False,
          wrap='full-image', K=5, N=2, is_sym=True, weight=None):
    rng = np.random.default_rng(seed)
    p = np.sort(rng.uniform(0.0, P, size=(K, N)), axis=0)
    w = None if weight is None else [np.full((K, N), float(weight))]
    return build_exp_tens([p], w, [sigma], [r], [is_rel], [is_per],
                          [P if is_per else 0.0], [is_sym], wrap=[wrap],
                          verbose=False)


def _route_taken(monkeypatch):
    """Record which flat inner-product route ``cos_sim_exp_tens`` runs."""
    taken = []
    orbit = _cos._cos_sim_exp_tens_ma_orbit
    pairwise = _cos._cos_sim_exp_tens_ma_pairwise

    def _orbit(*a, **k):
        taken.append("mobius")
        return orbit(*a, **k)

    def _pairwise(*a, **k):
        taken.append("bulger")
        return pairwise(*a, **k)

    monkeypatch.setattr(_cos, "_cos_sim_exp_tens_ma_orbit", _orbit)
    monkeypatch.setattr(_cos, "_cos_sim_exp_tens_ma_pairwise", _pairwise)
    return taken


# ---------------------------------------------------------------------
# explain_dispatch reports the route the call takes
# ---------------------------------------------------------------------

@pytest.mark.parametrize("wrap, expected", [("full-image", "mobius"),
                                            ("single-image", "bulger")])
def test_explain_follows_the_wrap_rule_above_the_threshold(
        monkeypatch, wrap, expected):
    """r = 3, K = 4 vs 5, rel-per at sigma/P = 0.5: above the threshold
    the declared wrap decides without pricing. The report used to omit
    the wrap vector and so named the cost model's pick."""
    x = _flat(1, 0.5 * P, r=3, is_rel=True, is_per=True, K=4, wrap=wrap)
    y = _flat(2, 0.5 * P, r=3, is_rel=True, is_per=True, K=5, wrap=wrap)
    exp = explain_dispatch(x, y)
    assert exp.chosen == expected
    assert exp.decided_by == "structural rule"
    taken = _route_taken(monkeypatch)
    cos_sim_exp_tens(x, y, verbose=False)
    assert taken == [expected]
    # The measure line follows the same reading.
    assert ("transposition average" in exp.measure) == (expected == "mobius")


@pytest.mark.parametrize("method", ["auto", "mobius", "centres"])
def test_explain_applies_the_ordered_attribute_rule(monkeypatch, method):
    """An ordered ([sym]=0) attribute at r > 1 has no orbit: the call
    takes Bulger's method whatever ``method`` asked for, and so does the
    report."""
    x = _flat(3, 1.0, r=2, is_sym=False, K=5)
    y = _flat(4, 1.0, r=2, is_sym=False, K=6)
    exp = explain_dispatch(x, y, method=method)
    assert exp.chosen == "bulger"
    assert "ordered" in exp.decided_by
    taken = _route_taken(monkeypatch)
    cos_sim_exp_tens(x, y, method=method, verbose=False)
    assert taken == ["bulger"]


def test_explain_uses_the_memo_flags_the_call_uses():
    """After a call has memoised both self inner products, the report
    prices the cross matrix alone, as the call does: its Bulger price
    drops below the cold report's."""
    x = _flat(5, 1.0, r=3, is_rel=True, is_per=False, K=6, N=3)
    y = _flat(6, 1.0, r=3, is_rel=True, is_per=False, K=6, N=3)
    cold = explain_dispatch(x, y)
    cos_sim_exp_tens(x, y, method="bulger", verbose=False)
    warm = explain_dispatch(x, y)
    pw_cold = next(r.predicted_ms for r in cold.routes if r.name == "bulger")
    pw_warm = next(r.predicted_ms for r in warm.routes if r.name == "bulger")
    assert pw_warm < pw_cold


def test_explain_applies_the_empty_operand_rule():
    """An operand that prunes to no events returns 0.0 before any
    selector runs; the report says so and names no route."""
    x = _flat(7, 1.0, r=2, weight=0.0)
    y = _flat(8, 1.0, r=2)
    exp = explain_dispatch(x, y)
    assert exp.chosen is None
    assert "empty operand" in exp.decided_by
    assert cos_sim_exp_tens(x, y, verbose=False) == 0.0


def test_selector_inputs_are_shared_with_the_call():
    """The report and the call build the selector's inputs by one
    function, so they cannot drift: its wrap vector, node counts and
    memo flags are those of the densities."""
    x = _flat(9, 0.5 * P, r=3, is_rel=True, is_per=True, K=4,
              wrap="single-image")
    y = _flat(10, 0.5 * P, r=3, is_rel=True, is_per=True, K=5,
              wrap="single-image")
    kw, ordered_any, nested_any = _cos._flat_selector_inputs(
        x.pruned(), y.pruned(), normalize="cosine", truncation_sigmas=None)
    assert kw["wrap_vec"] == ["single-image"]
    assert kw["per_vec"] == [True]
    assert kw["nu_vec"][0] > 1 and not ordered_any and not nested_any
    assert kw["skip_xx"] is False and kw["skip_yy"] is False
    cos_sim_exp_tens(x, y, verbose=False)
    kw2, _o, _n = _cos._flat_selector_inputs(
        x.pruned(), y.pruned(), normalize="cosine", truncation_sigmas=None)
    assert kw2["skip_xx"] is True and kw2["skip_yy"] is True


# ---------------------------------------------------------------------
# D-6: the eval cost model prices the spectral branch that runs
# ---------------------------------------------------------------------

def test_eval_cost_model_has_no_mode_grid_decline():
    """A periodic r = 4 relative attribute at small sigma/P passes the
    evaluator's spectral gate; the model prices that branch (its cost
    is the K-free per-mode slope, so it does not grow with K as the
    node path's tabulation term does)."""
    sigma = 0.004 * P          # a mode grid that the old decline refused
    small = _flat(11, sigma, r=4, is_rel=True, is_per=True, K=16, N=1)
    large = _flat(12, sigma, r=4, is_rel=True, is_per=True, K=48, N=1)
    n_q = 256
    _c1, m_small = _disp._ma_eval_costs_ms(small, n_q)
    _c2, m_large = _disp._ma_eval_costs_ms(large, n_q)
    assert math.isfinite(m_small) and math.isfinite(m_large)
    # The r = 4 periodic K term is zero, so the spectral price is the
    # same for both value counts; the node path it used to fall back to
    # carries a K-proportional tabulation term.
    assert m_large == pytest.approx(m_small, rel=1e-12)


# ---------------------------------------------------------------------
# B-16: one abs r >= 2 route in the per-attribute matrix
# ---------------------------------------------------------------------

def test_per_attr_matrix_batched_route_agrees_with_enumeration_at_k_equal_r():
    """With every event at K_eff = r the former partition sent every
    pair to direct enumeration; the one batched route agrees with that
    enumeration on the value scale."""
    rng = np.random.default_rng(13)
    Px = np.sort(rng.uniform(0.0, 40.0, size=(3, 4)), axis=0)
    Wx = np.ones_like(Px)
    Py = np.sort(rng.uniform(0.0, 40.0, size=(3, 5)), axis=0)
    Wy = np.ones_like(Py)
    sigma, r = 3.0, 3
    # Compare exhaustively: at the 1e-12 accuracy floor the two drop
    # marginally different far-tail terms, which the alternating sum
    # amplifies on the near-zero cross entries.
    from mpt._defaults import accuracy_floor_context
    with accuracy_floor_context(1e-300):
        I = _ma_per_attr_inner_matrix(Px, Wx, Py, Wy, sigma, r, False, False,
                                      0.0, truncation_sigmas=float("inf"))
    ref = np.array([[inner_product_direct_abs(Px[:, i], Wx[:, i],
                                              Py[:, j], Wy[:, j],
                                              sigma, r, False, 0.0)
                     for j in range(5)] for i in range(4)])
    np.testing.assert_allclose(I, ref, rtol=0.0,
                               atol=1e-12 * float(np.max(np.abs(ref))))


def test_per_attr_matrix_has_no_partition_helpers():
    from mpt._tensor import _mobius_inner as _mi
    for name in ("_batched_direct_enum_abs", "_pack_nan_top"):
        assert not hasattr(_mi, name)
        assert not hasattr(mpt.tensor, name)


# ---------------------------------------------------------------------
# B-16: the closed form refuses an inner [rel] unit
# ---------------------------------------------------------------------

def test_closed_form_refuses_an_inner_unit_bundle():
    """A bundle whose inner block size is >= 2 cannot reach the closed
    form from any shipped plan; handed one directly, it refuses rather
    than computing a flat quadratic form over a block-diagonal metric."""
    x = _flat(14, 1.0, r=2, is_rel=True, K=4, N=1)
    from mpt._tensor._mobius_inner import _closed_form_attr_centres
    cx = list(_closed_form_attr_centres(x.pruned(), 0))
    ok = _closed_form_attr_matrix_from(tuple(cx), tuple(cx))
    assert ok.shape == (1, 1)
    cx[3] = 2                        # pretend an inner unit of size 2
    with pytest.raises(ValueError, match="inner \\[rel\\] unit"):
        _closed_form_attr_matrix_from(tuple(cx), tuple(cx))


# ---------------------------------------------------------------------
# D: the orphans are gone
# ---------------------------------------------------------------------

def test_deleted_names_stay_deleted():
    gone = {
        _cos: ("_rel_contract_cheaper", "_attr_value_range", "_ma_has_nan",
               "_orbit_inner_abs", "_orbit_inner_rel",
               "_inner_product_direct_abs", "_build_ordered_r_tuples"),
        _disp: ("_orbit_ips_impossible", "_estimate_centres_array_bytes",
                "_pw_per_entry_ms", "_PW_PER_ENTRY_MS_NONPER",
                "_PW_PER_ENTRY_MS_PER", "_warn_rel_per_all_image",
                "_format_time", "_PROBE_MIN_N_Q", "_PROBE_K_IP_TARGET",
                "_PROBE_TIME_CACHE", "_PROBE_IP_MOBIUS_DECISION_MARGIN",
                "_PRESCREEN_IP_DOMINANCE", "_ORBIT_IP_FIXED_OVERHEAD",
                "_CENTRES_PROBE_MEM_BUDGET"),
        mpt.tensor: ("_orbit_ips_impossible", "_estimate_centres_array_bytes",
                     "_PROBE_MIN_N_Q", "_PROBE_K_IP_TARGET",
                     "_PRESCREEN_IP_DOMINANCE", "_orbit_inner_abs",
                     "_orbit_inner_rel", "_ma_has_nan"),
    }
    for mod, names in gone.items():
        for name in names:
            assert not hasattr(mod, name), f"{mod.__name__}.{name}"
    from mpt._tensor import sweep as _sweep
    assert not hasattr(_sweep, "SweepNotEligible")
    from mpt._tensor import _nested_contraction as _nc
    for name in ("nested_ip", "make_quadrature", "_ip_absolute",
                 "_ip_rel_nonper", "_theta_truncation_L"):
        assert not hasattr(_nc, name)
    import importlib
    with pytest.raises(ImportError):
        importlib.import_module("mpt._tensor._centres_inner")
    # The hard budget kept its value under its honest name.
    assert _disp._DISPATCH_MEM_BUDGET == 4 * 1024 ** 3


# ---------------------------------------------------------------------
# E: the retired cancellation_threshold keyword is rejected
# ---------------------------------------------------------------------

def test_cancellation_threshold_keyword_is_rejected():
    """``cancellation_threshold`` is gone; passing it raises the normal
    unknown-keyword error rather than being accepted and ignored."""
    x = _flat(21, 1.0, r=2, K=4, N=1)
    y = _flat(22, 1.0, r=2, K=4, N=1)
    with pytest.raises(TypeError, match="cancellation_threshold"):
        cos_sim_exp_tens(x, y, cancellation_threshold=1e-12, verbose=False)

