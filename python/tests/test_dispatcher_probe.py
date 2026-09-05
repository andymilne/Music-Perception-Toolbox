"""Multi-attribute eval dispatcher: the dispatch message and the
FP-identity of the auto-selected path with the explicit overrides (the
selector only picks a path; it does not mutate the answer)."""
from __future__ import annotations

import numpy as np
import pytest

import mpt
from mpt import add_spectra, build_exp_tens, eval_exp_tens


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _make_dens(K=12, r=3, is_rel=True, is_per=False, sigma=12.0):
    """Build a small single-multiset density."""
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
# Dispatch message
# -----------------------------------------------------------------------
class TestVerboseDispatchMessage:

    def test_message_prints_when_cost_model_decides(self, capsys):
        # K=6 r=3 abs reaches the cost model (no structural rule fires).
        p = np.linspace(0, 1200, 6, endpoint=False)
        dens = build_exp_tens(p, np.ones(6), 12.0, 3, False, False, 0.0)
        x = np.random.uniform(0, 1200, (3, 500))
        mpt.reset_defaults()
        eval_exp_tens(dens, x, truncation_sigmas=6.0, verbose=True)
        captured = capsys.readouterr()
        assert "chose" in captured.out
        assert "path" in captured.out
        # The message announces the routing DECISION only; the time
        # estimate ("estimated ... Ctrl+C to cancel") is a separate
        # concern emitted by the executing path.

    def test_message_appears_when_tiny(self, capsys):
        """Tiny workload: the dispatch message still fires (path only,
        no time estimate)."""
        dens = _make_dens(K=12, r=3, is_rel=True)
        x = np.random.uniform(0, 1200, (2, 50))
        mpt.reset_defaults()
        eval_exp_tens(dens, x, verbose=True)
        captured = capsys.readouterr()
        # Format: "eval_exp_tens (MAET): chose 'centres' path." (no
        # parenthetical, no time estimate).
        assert "eval_exp_tens (MAET): chose 'centres' path." in captured.out
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
        assert "eval_exp_tens (MAET): chose 'centres' path" in captured.out

    def test_message_fires_even_when_verbose_false(self, capsys):
        """v3: dispatch messages bypass per-call verbose; they're
        gated by mpt.get_default('show_hints'), not by verbose. This
        ensures users see the routing decision even when called from
        internal code paths that defensively pass verbose=False."""
        dens = _make_dens(K=12, r=3, is_rel=True)
        x = np.random.uniform(0, 1200, (2, 500))
        mpt.reset_defaults()
        eval_exp_tens(dens, x, truncation_sigmas=6.0, verbose=False)
        captured = capsys.readouterr()
        assert "eval_exp_tens (MAET): chose" in captured.out

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
