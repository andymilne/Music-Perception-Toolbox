"""v2.2.x — unified dispatcher + probe-based estimator.

Verifies:
    * Hard rules decide without probing (user override, r<=1,
      K-r<2, tiny n_q, centres-memory budget).
    * Probing fires for non-trivial workloads and produces an estimate.
    * The dispatch-decision message prints when ``verbose=True`` and a
      probe ran, and is silent otherwise.
    * Probe-based path selection produces FP-identical results to
      explicit method= overrides (the dispatcher only picks a path; it
      does not mutate the answer).
"""
from __future__ import annotations

import numpy as np
import pytest

import mpt
from mpt import add_spectra, build_exp_tens, eval_exp_tens
from mpt.tensor import (
    _select_and_estimate_sa,
    _estimate_centres_array_bytes,
    _CENTRES_PROBE_MEM_BUDGET,
    _PROBE_MIN_N_Q,
)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _make_dens(K=12, r=3, is_rel=True, is_per=False, sigma=12.0):
    """Build a small SA density for probing."""
    if K <= 3:
        p = np.linspace(0, 1200, K, endpoint=False)
    else:
        # Use a harmonic-like spectrum
        tp, tw = add_spectra(
            np.array([0., 0., 0.]), np.array([1., 1., 1.]),
            'harmonic', K // 3, 'powerlaw', 1,
        )
        p, w = tp, tw
        return build_exp_tens(
            p, w, sigma, r, is_rel, is_per,
            1200.0 if is_per else 0.0,
        )
    return build_exp_tens(
        p, np.ones(K), sigma, r, is_rel, is_per,
        1200.0 if is_per else 0.0,
    )


# -----------------------------------------------------------------------
# Hard rules (no probe)
# -----------------------------------------------------------------------


class TestHardRules:

    def test_user_override_centres(self):
        dens = _make_dens(K=12, r=3, is_rel=True)
        x = np.random.uniform(0, 1200, (2, 1000))
        chosen, probed, est, _ = _select_and_estimate_sa(
            dens, x, 1000, method='centres',
            truncation_sigmas=None, kernel_precision=None,
            verbose=False,
        )
        assert chosen == 'centres'
        assert probed is False
        assert est == 0.0

    def test_user_override_orbit(self):
        dens = _make_dens(K=12, r=3, is_rel=True)
        x = np.random.uniform(0, 1200, (2, 1000))
        chosen, probed, est, _ = _select_and_estimate_sa(
            dens, x, 1000, method='mobius',
            truncation_sigmas=None, kernel_precision=None,
            verbose=False,
        )
        assert chosen == 'mobius'
        assert probed is False

    def test_user_invalid_method_raises(self):
        dens = _make_dens()
        x = np.zeros((2, 100))
        with pytest.raises(ValueError, match="method must be"):
            _select_and_estimate_sa(
                dens, x, 100, method='nonsense',
                truncation_sigmas=None, kernel_precision=None,
                verbose=False,
            )

    def test_r_one_routes_to_centres(self):
        dens = build_exp_tens(np.array([0., 400., 700.]), np.ones(3),
                              12.0, 1, False, False, 0.0)
        x = np.array([[100., 200., 300.]])
        chosen, probed, est, _ = _select_and_estimate_sa(
            dens, x, 3, method='auto',
            truncation_sigmas=None, kernel_precision=None,
            verbose=False,
        )
        assert chosen == 'centres'
        assert probed is False

    def test_cancellation_guard(self):
        """K - r < 2 → centres (orbit Möbius cancellation)."""
        # K=3, r=3 → K-r = 0, cancellation guard triggers
        dens = build_exp_tens(np.array([0., 400., 700.]), np.ones(3),
                              12.0, 3, True, False, 0.0)
        x = np.random.uniform(0, 1200, (2, 1000))
        chosen, probed, est, _ = _select_and_estimate_sa(
            dens, x, 1000, method='auto',
            truncation_sigmas=None, kernel_precision=None,
            verbose=False,
        )
        assert chosen == 'centres'
        assert probed is False

    def test_tiny_workload_skips_probe(self):
        dens = _make_dens(K=12, r=3, is_rel=True)
        x = np.random.uniform(0, 1200, (2, _PROBE_MIN_N_Q - 1))
        chosen, probed, est, _ = _select_and_estimate_sa(
            dens, x, _PROBE_MIN_N_Q - 1, method='auto',
            truncation_sigmas=None, kernel_precision=None,
            verbose=False,
        )
        assert chosen == 'centres'
        assert probed is False


class TestRelPreScreen:
    """Pre-screen catches the typical rel-mode case (centres dominates)
    so the dispatcher avoids the expensive orbit-rel probe.
    """

    def test_typical_rel_skips_probe(self):
        # K=36, r=3, rel mode, typical sigma/period → centres dominates
        # by ~50× on this workload; pre-screen should fire.
        dens = _make_dens(K=12, r=3, is_rel=True)
        x = np.random.uniform(0, 1200, (2, 5000))
        chosen, probed, est, _ = _select_and_estimate_sa(
            dens, x, 5000, method='auto',
            truncation_sigmas=6.0, kernel_precision=None,
            verbose=False,
        )
        assert chosen == 'centres'
        assert probed is False  # pre-screen caught it

    def test_huge_K_rel_routes_to_mobius_on_memory(self):
        # At very large K in rel mode the centres working set
        # (j_idx + u_perm + centres, each O(n_j = K!/(K-r)!)) runs to
        # ~1 GB here — far over the soft working-set budget. The
        # memory-aware guard (Rule 4b) routes to Möbius WITHOUT probing:
        # probing the centres path would itself materialise the full
        # n_j working set just to time it, defeating the purpose. This
        # supersedes the earlier time-only policy (which let the probe
        # run at huge K); the Möbius point evaluator is n_j-free, so it
        # keeps peak memory bounded on constrained machines.
        K = 250
        p = np.linspace(0, 1200, K, endpoint=False)
        dens = build_exp_tens(p, np.ones(K), 12.0, 3, True, False, 0.0)
        x = np.random.uniform(0, 1200, (2, 500))
        chosen, probed, est, reason = _select_and_estimate_sa(
            dens, x, 500, method='auto',
            truncation_sigmas=6.0, kernel_precision=None,
            verbose=False,
        )
        assert chosen == 'mobius'
        assert probed is False
        assert 'working-set' in reason

    def test_moderate_K_rel_still_probes(self):
        # Just under the soft working-set budget (K=140, r=3, rel →
        # ~180 MB working set), the memory guard does NOT fire, so the
        # dispatcher falls through to the time-based rel pre-screen /
        # probe as before. Guards the boundary: the memory guard must
        # not swallow the whole large-K rel regime, only the part that
        # would actually blow memory.
        K = 140
        p = np.linspace(0, 1200, K, endpoint=False)
        dens = build_exp_tens(p, np.ones(K), 12.0, 3, True, False, 0.0)
        x = np.random.uniform(0, 1200, (2, 500))
        chosen, probed, est, reason = _select_and_estimate_sa(
            dens, x, 500, method='auto',
            truncation_sigmas=6.0, kernel_precision=None,
            verbose=False,
        )
        # Memory guard did not fire; decision came from the time-based
        # pre-screen or the probe, not the working-set rule.
        assert 'working-set' not in reason


class TestAbsPreScreen:
    """Pre-screen catches the typical abs-mode large-K case (orbit
    dominates) so the dispatcher routes to orbit even at tiny n_q.

    For abs mode, centres cost per query is K^r and orbit cost is
    B_r * r * K; the ratio is K^(r-1) / (B_r * r). At K=72 r=3 this
    is ~1000x, so the tiny-workload shortcut would force centres
    at any n_q<200 without this pre-screen — wasting orders of
    magnitude of compute on workloads where orbit is clearly faster
    (e.g. pattern-finding at typical harmonic-template K).
    """

    def test_large_K_abs_routes_to_orbit_at_any_n_q(self):
        # K=72 r=3 abs: ratio = 72^2 / (5*3) = 345.6 >> 10 (margin).
        # Pre-screen should fire for ANY n_q.
        K = 72
        p = np.linspace(0, 1200, K, endpoint=False)
        dens = build_exp_tens(p, np.ones(K), 12.0, 3, False, False, 0.0)
        # Try multiple n_q values, including below the tiny-shortcut threshold.
        for n_q in [50, 100, _PROBE_MIN_N_Q - 1, 5000]:
            x = np.random.uniform(0, 1200, (3, n_q))
            chosen, probed, est, _ = _select_and_estimate_sa(
                dens, x, n_q, method='auto',
                truncation_sigmas=None, kernel_precision=None,
                verbose=False,
            )
            assert chosen == 'mobius', (
                f"K={K} r=3 abs n_q={n_q} should pre-screen to orbit "
                f"(ratio K^(r-1)/(B_r*r) = {K**2/(5*3):.0f}); got {chosen!r}"
            )
            assert probed is False, (
                f"n_q={n_q}: pre-screen should fire before any probe"
            )

    def test_small_K_abs_does_not_force_orbit(self):
        # K=4 r=2 abs: K^(r-1) = 4, B_r*r = 4. Ratio = 1, well below
        # margin 10. Pre-screen should NOT fire; falls through to
        # tiny-workload shortcut (centres) for small n_q, probe for
        # large n_q.
        K = 4
        p = np.linspace(0, 1200, K, endpoint=False)
        dens = build_exp_tens(p, np.ones(K), 12.0, 2, False, False, 0.0)
        x = np.random.uniform(0, 1200, (2, 50))  # tiny
        chosen, probed, est, _ = _select_and_estimate_sa(
            dens, x, 50, method='auto',
            truncation_sigmas=None, kernel_precision=None,
            verbose=False,
        )
        # Tiny shortcut should win: chosen='centres', no probe.
        assert chosen == 'centres'
        assert probed is False

    def test_borderline_K_abs_falls_through_to_probe(self):
        # K=6 r=3 abs: ratio = K^(r-1) / (B_r * r) = 36/15 = 2.4,
        # below the 3.0 dominance margin. Pre-screen does NOT fire;
        # probe runs for non-tiny n_q.
        K = 6
        p = np.linspace(0, 1200, K, endpoint=False)
        dens = build_exp_tens(p, np.ones(K), 12.0, 3, False, False, 0.0)
        x = np.random.uniform(0, 1200, (3, 1000))
        chosen, probed, est, _ = _select_and_estimate_sa(
            dens, x, 1000, method='auto',
            truncation_sigmas=None, kernel_precision=None,
            verbose=False,
        )
        # Probe MUST have fired (pre-screen didn't catch).
        assert probed is True

    def test_rel_mode_unaffected_by_abs_prescreen(self):
        # At K=36 r=3 rel non-per sigma=12: centres_cost = 1296 on the
        # per-K basis, versus a Möbius direct-strategy cost of ~32400
        # and a factored-strategy cost of ~9100 (at the measured
        # read-back weight). The rel-mode pre-screen clears its margin
        # and fires to centres unprobed. The
        # regression under test is the ABS-mode pre-screen spuriously
        # redirecting rel workloads to Möbius just because the
        # abs-mode cost ratio favours it at this K; its signature is
        # an unprobed 'mobius' choice with the abs-mode reason.
        K = 36
        p = np.linspace(0, 1200, K, endpoint=False)
        dens = build_exp_tens(p, np.ones(K), 12.0, 3, True, False, 0.0)
        x = np.random.uniform(0, 1200, (2, 5000))
        chosen, probed, est, reason = _select_and_estimate_sa(
            dens, x, 5000, method='auto',
            truncation_sigmas=None, kernel_precision=None,
            verbose=False,
        )
        assert reason != 'abs-mode pre-screen'
        assert chosen == 'centres'


# -----------------------------------------------------------------------
# Memory budget rule
# -----------------------------------------------------------------------


class TestCentresMemoryBudget:

    def test_estimate_centres_array_bytes_simple(self):
        # K=10, r=3, abs → 10*9*8 = 720 tuples × 3 dim × 8 bytes
        n_bytes = _estimate_centres_array_bytes(10, 3, False)
        assert n_bytes == 10 * 9 * 8 * 3 * 8

    def test_estimate_centres_array_bytes_rel(self):
        # K=10, r=3, rel → dim=r-1=2
        n_bytes = _estimate_centres_array_bytes(10, 3, True)
        assert n_bytes == 10 * 9 * 8 * 2 * 8

    def test_k_lt_r_returns_zero(self):
        assert _estimate_centres_array_bytes(2, 3, False) == 0

    def test_huge_centres_array_forces_orbit(self, monkeypatch):
        """If centres array would exceed budget, dispatcher picks orbit
        without probing."""
        # K=200, r=5, abs → ~200^5 = 3.2e11 tuples, way over budget
        K = 200
        p = np.linspace(0, 1200, K, endpoint=False)
        dens = build_exp_tens(p, np.ones(K), 12.0, 5, False, False, 0.0)
        x = np.random.uniform(0, 1200, (5, 1000))
        chosen, probed, est, _ = _select_and_estimate_sa(
            dens, x, 1000, method='auto',
            truncation_sigmas=None, kernel_precision=None,
            verbose=False,
        )
        assert chosen == 'mobius'
        assert probed is False


# -----------------------------------------------------------------------
# Probing
# -----------------------------------------------------------------------


class TestProbing:

    def test_probe_fires_for_non_trivial_workload(self):
        # K=6 r=3 abs: ratio K^(r-1)/(B_r * r) = 36/15 = 2.4 < 3.0
        # dominance margin, so the abs pre-screen does NOT fire and
        # the probe is reached. (K=7 r=3 abs would pre-screen to
        # orbit — ratio ≈ 3.3 — so it doesn't probe.)
        p = np.linspace(0, 1200, 6, endpoint=False)
        dens = build_exp_tens(p, np.ones(6), 12.0, 3, False, False, 0.0)
        x = np.random.uniform(0, 1200, (3, 500))
        chosen, probed, est, _ = _select_and_estimate_sa(
            dens, x, 500, method='auto',
            truncation_sigmas=6.0, kernel_precision=None,
            verbose=False,
        )
        assert probed is True
        assert chosen in ('centres', 'mobius')
        assert est > 0

    def test_estimate_scales_with_n_q(self):
        # K=6 r=3 abs falls through the abs pre-screen (ratio 2.4 < margin 3.0).
        p = np.linspace(0, 1200, 6, endpoint=False)
        dens = build_exp_tens(p, np.ones(6), 12.0, 3, False, False, 0.0)
        rng = np.random.default_rng(42)
        x1 = rng.uniform(0, 1200, (3, 500))
        x2 = rng.uniform(0, 1200, (3, 1000))
        _, _, est1, _ = _select_and_estimate_sa(
            dens, x1, 500, method='auto',
            truncation_sigmas=6.0, kernel_precision=None,
            verbose=False,
        )
        _, _, est2, _ = _select_and_estimate_sa(
            dens, x2, 1000, method='auto',
            truncation_sigmas=6.0, kernel_precision=None,
            verbose=False,
        )
        assert est1 > 0
        assert est2 > 0
        ratio = est2 / est1
        assert 0.2 < ratio < 5.0, f"ratio {ratio} out of plausible range"


class TestVerboseDispatchMessage:

    def test_message_prints_when_probed(self, capsys):
        # K=6 r=3 abs falls through to probe (see TestProbing notes).
        p = np.linspace(0, 1200, 6, endpoint=False)
        dens = build_exp_tens(p, np.ones(6), 12.0, 3, False, False, 0.0)
        x = np.random.uniform(0, 1200, (3, 500))
        mpt.reset_defaults()
        eval_exp_tens(dens, x, truncation_sigmas=6.0, verbose=True)
        captured = capsys.readouterr()
        assert "chose" in captured.out
        assert "path" in captured.out
        assert "estimated" in captured.out
        assert "Ctrl+C to cancel" in captured.out

    def test_message_appears_when_tiny(self, capsys):
        """Tiny workload skips probing but the dispatch message still
        fires (unprobed format: path only, no time estimate)."""
        dens = _make_dens(K=12, r=3, is_rel=True)
        x = np.random.uniform(0, 1200, (2, 50))
        mpt.reset_defaults()
        eval_exp_tens(dens, x, verbose=True)
        captured = capsys.readouterr()
        # Unprobed format: "eval_exp_tens: chose 'centres' path." (no
        # parenthetical, no time estimate).
        assert "eval_exp_tens: chose 'centres' path." in captured.out
        assert "estimated" not in captured.out

    def test_message_throttled_within_a_top_level_call(self, capsys):
        """Within one top-level call, repeated internal dispatch decisions
        emit at most one message per (func, chosen) pair. Two different
        routing reasons leading to the same chosen path collapse to a
        single announce — the visible distinction that matters is which
        path ran, not why. Across top-level calls, each call re-announces
        (see :meth:`test_message_re_announces_across_top_level_calls`).
        """
        dens = _make_dens(K=12, r=3, is_rel=True)
        # Many internal eval_exp_tens decisions inside one top-level
        # call via a 2-D query in a single invocation. (Repeated
        # top-level calls would each re-announce; here we exercise
        # the within-call throttle.)
        x = np.random.uniform(0, 1200, (2, 500))
        mpt.reset_defaults()
        eval_exp_tens(dens, x, verbose=True)
        captured = capsys.readouterr()
        # One "chose" line for this single top-level call.
        assert captured.out.count("chose") == 1

    def test_message_re_announces_across_top_level_calls(self, capsys):
        """Each top-level user call resets the dispatch seen-set, so
        repeated identical top-level calls each emit a fresh message.
        """
        dens = _make_dens(K=12, r=3, is_rel=True)
        x = np.random.uniform(0, 1200, (2, 50))
        mpt.reset_defaults()
        eval_exp_tens(dens, x, verbose=True)
        eval_exp_tens(dens, x, verbose=True)
        captured = capsys.readouterr()
        assert captured.out.count("chose") == 2

    def test_message_reappears_after_reset(self, capsys):
        """mpt.reset_defaults() clears the throttle."""
        dens = _make_dens(K=12, r=3, is_rel=True)
        x = np.random.uniform(0, 1200, (2, 50))
        mpt.reset_defaults()
        eval_exp_tens(dens, x, verbose=True)
        capsys.readouterr()                    # drain
        mpt.reset_defaults()
        eval_exp_tens(dens, x, verbose=True)  # fresh: prints again
        captured = capsys.readouterr()
        assert "eval_exp_tens: chose 'centres' path" in captured.out

    def test_message_fires_even_when_verbose_false(self, capsys):
        """v2.2.x: dispatch messages bypass per-call verbose; they're
        gated by mpt.get_default('show_hints'), not by verbose. This
        ensures users see the routing decision even when called from
        internal code paths that defensively pass verbose=False."""
        dens = _make_dens(K=12, r=3, is_rel=True)
        x = np.random.uniform(0, 1200, (2, 500))
        mpt.reset_defaults()
        eval_exp_tens(dens, x, truncation_sigmas=6.0, verbose=False)
        captured = capsys.readouterr()
        assert "eval_exp_tens: chose" in captured.out

    def test_silenced_by_show_hints_false(self, capsys):
        """Dispatch messages are silenced by show_hints=False."""
        dens = _make_dens(K=12, r=3, is_rel=True)
        x = np.random.uniform(0, 1200, (2, 500))
        mpt.reset_defaults()
        mpt.set_default(show_hints=False)
        eval_exp_tens(dens, x, truncation_sigmas=6.0, verbose=True)
        captured = capsys.readouterr()
        assert "chose" not in captured.out
        mpt.reset_defaults()


# -----------------------------------------------------------------------
# Path-selection doesn't change the answer
# -----------------------------------------------------------------------


class TestDispatcherDoesNotMutateAnswer:

    def test_auto_matches_centres_for_rel(self):
        dens = _make_dens(K=12, r=3, is_rel=True)
        rng = np.random.default_rng(0)
        x = rng.uniform(0, 1200, (2, 300))
        mpt.reset_defaults()
        v_auto = eval_exp_tens(dens, x, method='auto', verbose=False)
        v_centres = eval_exp_tens(dens, x, method='centres', verbose=False)
        np.testing.assert_allclose(v_auto, v_centres, rtol=0, atol=0)

    def test_auto_with_truncation_matches_centres(self):
        dens = _make_dens(K=12, r=3, is_rel=True)
        rng = np.random.default_rng(1)
        x = rng.uniform(0, 1200, (2, 300))
        mpt.reset_defaults()
        v_auto = eval_exp_tens(dens, x, truncation_sigmas=6.0,
                               method='auto', verbose=False)
        v_centres = eval_exp_tens(dens, x, truncation_sigmas=6.0,
                                  method='centres', verbose=False)
        np.testing.assert_allclose(v_auto, v_centres, rtol=0, atol=0)
