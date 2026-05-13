"""Tests for the v2.2.x probe-based dispatcher for SA cos_sim_exp_tens.

The dispatcher (``_select_and_estimate_sa_ip``) decides between the
Möbius method and Bulger's method. Hard rules (correctness /
feasibility) decide first; then an analytical pre-screen catches
clear-winner cases without paying probe overhead; otherwise both
paths are timed on a small subset and the faster is picked.
"""
from __future__ import annotations

import io
import warnings
from contextlib import redirect_stdout

import numpy as np
import pytest

import mpt
from mpt import build_exp_tens, cos_sim_exp_tens
from mpt.tensor import (
    _PROBE_K_IP_TARGET,
    _PRESCREEN_IP_DOMINANCE,
    _select_and_estimate_sa_ip,
)


def _dens(K: int, r: int, sigma: float = 1.0,
          is_rel: bool = False, is_per: bool = False, seed: int = 0):
    rng = np.random.default_rng(seed)
    p = np.sort(rng.uniform(0, 100, K))
    w = np.ones(K)
    return build_exp_tens(p, w, sigma, r, is_rel, is_per, 1200.0,
                          verbose=False)


# ----------------------------------------------------------------------
# Hard rules
# ----------------------------------------------------------------------


class TestHardRules:
    def test_user_method_pairwise(self):
        dens_x, dens_y = _dens(20, 3), _dens(20, 3, seed=1)
        chosen, probed, est = _select_and_estimate_sa_ip(
            dens_x, dens_y, method="bulger",
            truncation_sigmas=None, kernel_precision=None, verbose=False,
        )
        assert chosen == "bulger"
        assert probed is False

    def test_user_method_orbit(self):
        dens_x, dens_y = _dens(20, 3), _dens(20, 3, seed=1)
        chosen, probed, est = _select_and_estimate_sa_ip(
            dens_x, dens_y, method="mobius",
            truncation_sigmas=None, kernel_precision=None, verbose=False,
        )
        assert chosen == "mobius"
        assert probed is False

    def test_r1_routes_pairwise_no_probe(self):
        dens_x, dens_y = _dens(20, 1), _dens(20, 1, seed=1)
        chosen, probed, est = _select_and_estimate_sa_ip(
            dens_x, dens_y, method="auto",
            truncation_sigmas=None, kernel_precision=None, verbose=False,
        )
        assert chosen == "bulger"
        assert probed is False

    def test_n_min_too_small_routes_pairwise_no_probe(self):
        # K_y = 4, r = 3 -> n_min - r = 1 < 2: orbit precision guard.
        dens_x, dens_y = _dens(20, 3), _dens(4, 3, seed=1)
        chosen, probed, est = _select_and_estimate_sa_ip(
            dens_x, dens_y, method="auto",
            truncation_sigmas=None, kernel_precision=None, verbose=False,
        )
        assert chosen == "bulger"
        assert probed is False


# ----------------------------------------------------------------------
# Analytical pre-screen
# ----------------------------------------------------------------------


class TestPreScreen:
    def test_large_K_routes_orbit_via_prescreen(self):
        """Large K with moderate r: pairwise cost (K^r * K^r) dwarfs
        orbit cost (B_r * K^2). Pre-screen routes to orbit without
        probing."""
        dens_x, dens_y = _dens(40, 3), _dens(40, 3, seed=1)
        chosen, probed, est = _select_and_estimate_sa_ip(
            dens_x, dens_y, method="auto",
            truncation_sigmas=None, kernel_precision=None, verbose=False,
        )
        assert chosen == "mobius"
        assert probed is False

    def test_small_K_at_r_routes_pairwise_via_prescreen_or_probe(self):
        """r=2 with K=5: pairwise cost (K*(K-1))^2 = 400 vs orbit cost
        B_2 * K^2 = 50. Ratio = 8.0, just under the 10x pre-screen
        threshold, so probe fires. Either decision is acceptable;
        verify probed is True."""
        dens_x, dens_y = _dens(5, 2), _dens(5, 2, seed=1)
        chosen, probed, est = _select_and_estimate_sa_ip(
            dens_x, dens_y, method="auto",
            truncation_sigmas=None, kernel_precision=None, verbose=False,
        )
        assert chosen in ("mobius", "bulger")
        # Probe fires because pre-screen ratio is < 10x dominance.
        assert probed is True


# ----------------------------------------------------------------------
# Probe execution
# ----------------------------------------------------------------------


class TestProbe:
    def test_probe_returns_positive_estimate(self):
        """When probe fires, est_sec > 0."""
        dens_x, dens_y = _dens(8, 3), _dens(8, 3, seed=1)
        chosen, probed, est = _select_and_estimate_sa_ip(
            dens_x, dens_y, method="auto",
            truncation_sigmas=None, kernel_precision=None, verbose=False,
        )
        if probed:
            assert est > 0


# ----------------------------------------------------------------------
# Semantic equivalence with the analytical-heuristic decision
# ----------------------------------------------------------------------


class TestSemanticEquivalence:
    """The probe-based dispatcher must produce the same cos_sim_exp_tens
    output as the analytical heuristic for any reasonable workload —
    the dispatcher only chooses which mathematically-equivalent path to
    use, not what to compute."""

    def test_explicit_orbit_matches_explicit_pairwise(self):
        dens_x, dens_y = _dens(20, 3, seed=0), _dens(20, 3, seed=1)
        sim_orbit = cos_sim_exp_tens(
            dens_x, dens_y, method="mobius", verbose=False,
        )
        sim_pair = cos_sim_exp_tens(
            dens_x, dens_y, method="bulger", verbose=False,
        )
        # Orbit and pairwise should agree to numerical precision.
        np.testing.assert_allclose(sim_orbit, sim_pair, atol=1e-10)

    def test_auto_matches_explicit_paths(self):
        dens_x, dens_y = _dens(20, 3, seed=0), _dens(20, 3, seed=1)
        sim_auto = cos_sim_exp_tens(dens_x, dens_y, verbose=False)
        sim_orbit = cos_sim_exp_tens(
            dens_x, dens_y, method="mobius", verbose=False,
        )
        np.testing.assert_allclose(sim_auto, sim_orbit, atol=1e-10)


# ----------------------------------------------------------------------
# Verbose dispatch message
# ----------------------------------------------------------------------


class TestVerboseDispatchMessage:
    def test_message_prints_when_probe_fires(self):
        """Probe-firing region (r=2 K=5, ratio just under dominance) →
        message printed."""
        dens_x, dens_y = _dens(5, 2, seed=0), _dens(5, 2, seed=1)
        buf = io.StringIO()
        with redirect_stdout(buf):
            cos_sim_exp_tens(dens_x, dens_y, method="auto", verbose=True)
        out = buf.getvalue()
        # Probe should fire here, giving the dispatch message.
        assert "cos_sim_exp_tens" in out and "chose" in out

    def test_message_absent_when_hard_rule_decides(self):
        """r=1 → hard rule → no probe → no message."""
        dens_x, dens_y = _dens(8, 1, seed=0), _dens(8, 1, seed=1)
        buf = io.StringIO()
        with redirect_stdout(buf):
            cos_sim_exp_tens(dens_x, dens_y, method="auto", verbose=True)
        out = buf.getvalue()
        assert "cos_sim_exp_tens" not in out

    def test_message_absent_when_user_method_set(self):
        """Explicit method → hard rule → no probe → no message."""
        dens_x, dens_y = _dens(20, 3, seed=0), _dens(20, 3, seed=1)
        buf = io.StringIO()
        with redirect_stdout(buf):
            cos_sim_exp_tens(
                dens_x, dens_y, method="bulger", verbose=True,
            )
        out = buf.getvalue()
        assert "cos_sim_exp_tens" not in out
