"""Tests for the self-calibrated up-front evaluation time estimate.

Covers the machine-scale calibration, the culling correction on the
single-multiset centres cost (gated away from the multi-attribute
path), the once-per-top-level-call emission latch, and its gating by
``show_hints`` and the threshold. Timing-based assertions use generous
bounds --- the estimate is a cancel prompt, not a precise figure, and
wall-clock timing is load-sensitive.
"""

import numpy as np
import pytest

import mpt
from mpt._defaults import _dispatch_scope, _TIME_WARN_EMITTED
from mpt._tensor import _timeest
from mpt._tensor._timeest import (
    _estimate_eval_seconds,
    _maybe_warn_eval_time,
    _session_time_scale,
)
from mpt._tensor.dispatch import (
    _MA_COST_CENTRES_CALL_PER_JOINT_MS,
    _MA_COST_CENTRES_CULL_C,
    _MA_COST_CENTRES_FACTORED_QUERY_BASE_MS,
    _MA_COST_CENTRES_QUERY_BASE_MS,
    _MA_COST_CENTRES_QUERY_BASE_PER_MS,
    _MA_COST_CENTRES_QUERY_PER_JOINT_MS,
    _MA_COST_CENTRES_QUERY_JOINT_EXP_PER,
    _MA_COST_CENTRES_QUERY_JOINT_EXP_REL_PER,
    _MA_COST_CENTRES_QUERY_PER_JOINT_PER_MS,
    _MA_COST_CENTRES_QUERY_PER_JOINT_REL_PER_MS,
    _MA_COST_CENTRES_SETUP_MS,
    _ma_eval_costs_ms,
    _predict_ma_eval_cost_ms,
    _select_ma_eval,
)


def _single(K, spread, sigma, is_rel=True):
    p = [np.linspace(0.0, spread, K).reshape(-1, 1)]
    return mpt.build_exp_tens(
        p, None, [sigma], [2], [is_rel], [False], [0.0], verbose=False,
    )


def _two_attr(K, spread, sigma):
    p = [np.linspace(0.0, spread, K).reshape(-1, 1)] * 2
    return mpt.build_exp_tens(
        p, None, [sigma, sigma], [2, 2], [True, True], [False, False],
        [0.0, 0.0], verbose=False,
    )


class TestSessionCalibration:
    def test_scale_is_positive_finite(self):
        s = _session_time_scale()
        assert np.isfinite(s) and s > 0

    def test_scale_is_cached(self):
        s1 = _session_time_scale()
        s2 = _session_time_scale()
        assert s1 == s2


class TestCullingCorrection:
    def test_single_multiset_query_term_scales_with_sigma(self):
        # More culling at small sigma => smaller per-query centres cost.
        # Compare the query-only contribution (total minus the
        # sigma-independent setup and materialisation terms).
        K, spread, nq = 50, 1150.0, 40000
        d_lo = _single(K, spread, 10.0)
        d_hi = _single(K, spread, 40.0)
        joint = 2 * (K * (K - 1) // 2)
        base = (_MA_COST_CENTRES_SETUP_MS
                + _MA_COST_CENTRES_CALL_PER_JOINT_MS * joint)
        q_lo = _ma_eval_costs_ms(d_lo, nq)[0] - base
        q_hi = _ma_eval_costs_ms(d_hi, nq)[0] - base
        assert 0 < q_lo < q_hi  # culling grows the query cost with sigma

    def test_multi_attribute_centres_pricing(self):
        # The MA factored centres route (all r_a >= 2, scalar sigma)
        # is priced per attribute: SUM of per-attribute tuple counts
        # through culled kernels plus a per-attribute per-query base,
        # never the joint product.
        K, nq = 12, 500
        spread, sigma = 1150.0, 20.0
        d = _two_attr(K, spread, sigma)
        T_a = 2 * (K * (K - 1) // 2)
        # r = 2 relative, so each attribute's centres occupy one
        # dimension and the culled share is the ratio to the first power.
        cull = min(1.0, (_MA_COST_CENTRES_CULL_C * sigma / spread) ** 1)
        expected = _MA_COST_CENTRES_SETUP_MS + 2 * (
            _MA_COST_CENTRES_CALL_PER_JOINT_MS * T_a
            + nq * (_MA_COST_CENTRES_FACTORED_QUERY_BASE_MS
                    + _MA_COST_CENTRES_QUERY_BASE_MS
                    + _MA_COST_CENTRES_QUERY_PER_JOINT_MS * T_a * cull)
        )
        centres_ms, _ = _ma_eval_costs_ms(d, nq)
        assert centres_ms == pytest.approx(expected, rel=1e-12)

    def test_multi_attribute_joint_fallback_culls_per_attribute(self):
        # With an r = 1 attribute the factored route is unsupported and
        # the joint-materialisation fallback runs. It evaluates through
        # the same truncated kernel, so it is culled too: a query
        # reaches a joint centre only if it reaches that centre in every
        # attribute, making the joint share the product of the
        # per-attribute shares.
        K, nq = 12, 500
        spread, sigma = 1150.0, 20.0
        p = [np.linspace(0.0, spread, K).reshape(-1, 1)] * 2
        d = mpt.build_exp_tens(
            p, None, [sigma, sigma], [2, 1], [True, False], [False, False],
            [0.0, 0.0], verbose=False,
        )
        joint = (2 * (K * (K - 1) // 2)) * K
        # Both attributes occupy one dimension: r = 2 relative reduces to
        # one, and r = 1 absolute is one already.
        cull_a = min(1.0, (_MA_COST_CENTRES_CULL_C * sigma / spread) ** 1)
        expected = (
            _MA_COST_CENTRES_SETUP_MS
            + _MA_COST_CENTRES_CALL_PER_JOINT_MS * joint
            + nq * (_MA_COST_CENTRES_QUERY_BASE_MS
                    + _MA_COST_CENTRES_QUERY_PER_JOINT_MS
                    * joint * cull_a * cull_a)
        )
        centres_ms, _ = _ma_eval_costs_ms(d, nq)
        assert centres_ms == pytest.approx(expected, rel=1e-12)

    def test_predict_returns_chosen_path_cost(self):
        d = _single(30, 1150.0, 20.0)
        centres_ms, mobius_ms = _ma_eval_costs_ms(d, 2000)
        assert _predict_ma_eval_cost_ms(d, 2000, "centres") == centres_ms
        assert _predict_ma_eval_cost_ms(d, 2000, "mobius") == mobius_ms

    @pytest.mark.parametrize("is_rel", [True, False])
    def test_periodic_single_multiset_is_unculled(self, is_rel):
        # Periodic single-multiset runs dense (the pairwise wrap is not a
        # tail-truncatable ball), so it takes no culling discount: its
        # centres cost must equal the plain unculled form. The per-query
        # slope and the exponent on the tuple count are the periodic
        # kernel's own, and relative and absolute carry different ones ---
        # the relative kernel forms wrapped differences first and its
        # measured cost is superlinear in the tuple count where the
        # absolute kernel's is not.
        K, nq = 30, 4000
        r = 2 if is_rel else 1
        p = [np.linspace(0.0, 1150.0, K).reshape(-1, 1)]
        d = mpt.build_exp_tens(
            p, None, [22.0], [r], [is_rel], [True], [1200.0], verbose=False,
        )
        joint = 2 * (K * (K - 1) // 2) if is_rel else K
        slope = (_MA_COST_CENTRES_QUERY_PER_JOINT_REL_PER_MS if is_rel
                 else _MA_COST_CENTRES_QUERY_PER_JOINT_PER_MS)
        exp = (_MA_COST_CENTRES_QUERY_JOINT_EXP_REL_PER if is_rel
               else _MA_COST_CENTRES_QUERY_JOINT_EXP_PER)
        expected = (
            _MA_COST_CENTRES_SETUP_MS
            + _MA_COST_CENTRES_CALL_PER_JOINT_MS * joint
            + nq * (_MA_COST_CENTRES_QUERY_BASE_PER_MS
                    + slope * joint ** exp)
        )
        centres_ms, _ = _ma_eval_costs_ms(d, nq)
        assert centres_ms == pytest.approx(expected, rel=1e-12)


class TestEstimateShape:
    def test_positive(self):
        d = _single(40, 1150.0, 20.0)
        chosen, _ = _select_ma_eval(d, 5000, method="auto")
        assert _estimate_eval_seconds(d, 5000, chosen) > 0

    def test_scales_with_query_count(self):
        d = _single(40, 1150.0, 20.0)
        chosen, _ = _select_ma_eval(d, 10000, method="auto")
        e1 = _estimate_eval_seconds(d, 10000, chosen)
        e2 = _estimate_eval_seconds(d, 40000, chosen)
        # query term dominates at this size; ~4x the queries => ~4x time
        assert 3.0 < e2 / e1 < 5.0


class TestEmission:
    @pytest.fixture(autouse=True)
    def _low_threshold(self, monkeypatch):
        # Force every eval over threshold so emission logic is exercised
        # without needing a genuinely multi-second computation.
        monkeypatch.setattr(_timeest, "_EVAL_WARN_THRESHOLD_SEC", 1e-12)
        mpt.set_default(show_hints=True)
        yield
        mpt.reset_defaults()

    def test_emits_once_per_scope(self, capsys):
        d = _single(20, 1150.0, 20.0)
        chosen, _ = _select_ma_eval(d, 50, method="auto")
        with _dispatch_scope():
            _maybe_warn_eval_time("eval_exp_tens (MAET)", d, 50, chosen)
            _maybe_warn_eval_time("eval_exp_tens (MAET)", d, 50, chosen)
        out = capsys.readouterr().out
        assert out.count("estimated") == 1

    def test_latch_resets_between_scopes(self, capsys):
        d = _single(20, 1150.0, 20.0)
        chosen, _ = _select_ma_eval(d, 50, method="auto")
        for _ in range(2):
            with _dispatch_scope():
                _maybe_warn_eval_time("eval_exp_tens (MAET)", d, 50, chosen)
        assert capsys.readouterr().out.count("estimated") == 2

    def test_respects_show_hints(self, capsys):
        d = _single(20, 1150.0, 20.0)
        chosen, _ = _select_ma_eval(d, 50, method="auto")
        mpt.set_default(show_hints=False)
        with _dispatch_scope():
            _maybe_warn_eval_time("eval_exp_tens (MAET)", d, 50, chosen)
        assert "estimated" not in capsys.readouterr().out

    def test_threshold_suppresses_fast_evals(self, capsys, monkeypatch):
        monkeypatch.setattr(_timeest, "_EVAL_WARN_THRESHOLD_SEC", 1e9)
        d = _single(20, 1150.0, 20.0)
        chosen, _ = _select_ma_eval(d, 50, method="auto")
        with _dispatch_scope():
            _maybe_warn_eval_time("eval_exp_tens (MAET)", d, 50, chosen)
        assert "estimated" not in capsys.readouterr().out

    def test_harmony_wrapper_emits_at_most_once(self, capsys):
        # template_harmonicity makes two inner evals; the latch must
        # collapse them to a single warning.
        p = np.array([0.0, 400.0, 700.0, 1100.0])
        w = np.ones(4)
        mpt.template_harmonicity(p, w, sigma=30.0, verbose=False)
        assert capsys.readouterr().out.count("estimated") == 1


class TestAccuracySmoke:
    """Generous regression guard against gross mis-calibration on the
    culled centres path. Not a tight timing assertion."""

    def test_estimate_within_generous_band(self):
        import time

        _session_time_scale()
        # With the spectral (Fourier) strategy inside the mobius
        # relative evaluator, auto routes this shape to mobius
        # (measured ~10x faster than centres here); the estimate for
        # the chosen route must sit within a generous band of the
        # measured wall time.
        d = _single(60, 1150.0, 20.0)
        x = np.random.default_rng(7).uniform(0.0, 1150.0, (d.dim, 8000))
        chosen, _ = _select_ma_eval(d, 8000, method="auto")
        assert chosen == "mobius"
        est = _estimate_eval_seconds(d, 8000, chosen)
        mpt.eval_exp_tens(d, x, verbose=False)  # warm
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            mpt.eval_exp_tens(d, x, verbose=False)
            ts.append(time.perf_counter() - t0)
        act = sorted(ts)[1]
        assert 0.2 < est / act < 5.0
