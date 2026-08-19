"""Tests for the dispatch explanation.

The report must agree with the dispatch it describes: if it says a call
takes a route, that is the route the selector picks. These tests assert
that agreement rather than the wording, so they survive rephrasing but
fail if the report and the toolbox part company.
"""
import math

import numpy as np
import pytest

import mpt
from mpt._defaults import truncation_floor
from mpt._tensor.dispatch import (
    _orbit_sigma_over_p_threshold,
    _select_ma_eval,
)

PERIOD = 1200.0


def _dens(K=12, r=3, sigma=60.0, is_rel=True, is_per=True, seed=0):
    rng = np.random.default_rng(seed)
    p = np.sort(rng.uniform(0.0, PERIOD, K))
    return mpt.build_exp_tens(
        p, None, sigma, r, is_rel, is_per,
        PERIOD if is_per else 0.0, verbose=False,
    )


class TestAgreesWithTheDispatch:
    @pytest.mark.parametrize("r,K,is_rel,is_per,sigma", [
        (2, 12, False, False, 15.0),
        (3, 12, True, True, 60.0),
        (3, 24, True, False, 15.0),
        (4, 12, False, True, 15.0),
    ])
    def test_reported_route_is_the_selected_route(self, r, K, is_rel,
                                                  is_per, sigma):
        d = _dens(K=K, r=r, sigma=sigma, is_rel=is_rel, is_per=is_per)
        chosen, _ = _select_ma_eval(d, 200, method="auto")
        assert mpt.explain_dispatch(d, n_q=200).chosen == chosen

    def test_user_override_is_reported(self):
        d = _dens()
        assert mpt.explain_dispatch(
            d, n_q=200, method="centres").chosen == "centres"
        assert mpt.explain_dispatch(
            d, n_q=200, method="mobius").chosen == "mobius"


class TestReportsTheQuantities:
    def test_floor_matches_the_accuracy_asked_for(self):
        d = _dens()
        for ts in (4.0, 6.0, 8.0, math.inf):
            e = mpt.explain_dispatch(d, n_q=200, truncation_sigmas=ts)
            assert e.floor == pytest.approx(truncation_floor(ts))

    def test_sigma_over_p_limit_tracks_the_accuracy_asked_for(self):
        # The limit is a function of truncation_sigmas, so the report
        # must not cache one value: a tighter setting gives a smaller
        # limit, which is the whole reason the constant became a
        # function.
        d = _dens()
        loose = mpt.explain_dispatch(d, n_q=200, truncation_sigmas=4.0)
        tight = mpt.explain_dispatch(d, n_q=200, truncation_sigmas=8.0)
        assert loose.sigma_over_p_limit > tight.sigma_over_p_limit
        assert tight.sigma_over_p_limit == _orbit_sigma_over_p_threshold(8.0)

    def test_sigma_over_p_is_none_when_not_relative_periodic(self):
        assert mpt.explain_dispatch(
            _dens(is_rel=False, is_per=False), n_q=200).sigma_over_p is None

    def test_both_routes_are_priced_when_the_cost_model_decides(self):
        # Non-periodic, so no measure rule pre-empts the pricing.
        e = mpt.explain_dispatch(_dens(is_per=False, sigma=15.0), n_q=200)
        assert all(r.predicted_ms is not None and r.predicted_ms > 0
                   for r in e.routes)

    def test_exactly_one_route_is_chosen(self):
        e = mpt.explain_dispatch(_dens(), n_q=200)
        assert sum(bool(r.chosen) for r in e.routes) == 1


class TestCosine:
    def test_reports_a_route_for_a_density_pair(self):
        e = mpt.explain_dispatch(_dens(seed=1), _dens(seed=2))
        assert e.call == "cos_sim_exp_tens"
        assert e.chosen in ("bulger", "mobius")


class TestRendering:
    def test_prints_the_chosen_route_and_the_floor(self):
        text = str(mpt.explain_dispatch(_dens(), n_q=200))
        assert "chosen" in text
        assert "truncation_sigmas" in text
        assert "sigma/P" in text

    def test_does_not_run_the_call(self, monkeypatch):
        # A diagnostic must not be expensive or have side effects: it
        # reports what would happen without evaluating anything.
        d = _dens()
        called = {"n": 0}
        real = mpt.eval_exp_tens

        def counting(*a, **k):
            called["n"] += 1
            return real(*a, **k)

        monkeypatch.setattr(mpt, "eval_exp_tens", counting)
        mpt.explain_dispatch(d, n_q=200)
        assert called["n"] == 0


def test_explain_dispatch_ordered_r9_does_not_raise():
    """explain_dispatch on a bound ordered 9-tuple density must reach
    the same routing the real call does: the selector's forced-Bulger
    guard receives the density's [sym] flags, so the ordered
    C(K, r) = 1 tuple count passes where the unordered K! count would
    spuriously raise."""
    import numpy as np
    from mpt import bind_events, build_exp_tens, explain_dispatch
    x = np.arange(9, dtype=float)
    p_b, w_b, sp_b = bind_events([x[None, :], x[None, :]], None, 9,
                                 rel_outer=[False, True])
    dens = build_exp_tens(p_b, w_b, specs=sp_b, sigma=[0.3, 0.3],
                          is_per=[False] * 2, period=[None] * 2,
                          verbose=False)
    report = explain_dispatch(dens, dens)
    assert report is not None
