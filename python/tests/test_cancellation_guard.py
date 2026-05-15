"""Tests for the v2.2 cancellation guard on the Möbius method.

The guard fires when ``|<A,B>_orbit| < cancellation_threshold *
sqrt(<A,A>_orbit * <B,B>_orbit)`` and reroutes the calculation
through Bulger's method. With the default threshold (1e-12) it
never fires on healthy inputs; pushing the threshold up to 1.0
forces it to fire so we can verify the wiring.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from mpt import build_exp_tens, cos_sim_exp_tens
from mpt.tensor import _cos_sim_exp_tens_sa_orbit, _cos_sim_exp_tens_sa_pairwise


# ----------------------------------------------------------------------
# In normal use the guard does not change the result
# ----------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_default_threshold_does_not_perturb_result(seed):
    """At the default threshold (1e-12), auto and pairwise agree."""
    rng = np.random.default_rng(seed=seed)
    n = 12
    p_a = np.sort(rng.uniform(0, 5000, n))
    p_b = np.sort(rng.uniform(0, 5000, n))
    w_a = rng.uniform(0.5, 1.5, n)
    w_b = rng.uniform(0.5, 1.5, n)
    sigma = 30.0
    T_a = build_exp_tens(p_a, w_a, sigma, 3, False, False, 1200.0, verbose=False)
    T_b = build_exp_tens(p_b, w_b, sigma, 3, False, False, 1200.0, verbose=False)

    cos_default = cos_sim_exp_tens(T_a, T_b, verbose=False)
    cos_pw = cos_sim_exp_tens(T_a, T_b, method='bulger', verbose=False)
    abs_err = abs(cos_default - cos_pw)
    rel_err = abs_err / max(abs(cos_default), abs(cos_pw), 1e-300)
    assert rel_err < 1e-10 or abs_err < 1e-12


# ----------------------------------------------------------------------
# A high threshold forces the guard to fire and route to pairwise
# ----------------------------------------------------------------------


def test_high_threshold_forces_fallback():
    """With cancellation_threshold=1.0 the guard always trips and routes
    through Bulger's method; the result must equal method='bulger'.

    This verifies the wiring: when the guard fires, it really does
    recompute via Bulger's method rather than returning the orbit
    estimate.
    """
    rng = np.random.default_rng(seed=20260505)
    n = 12
    p_a = np.sort(rng.uniform(0, 5000, n))
    p_b = np.sort(rng.uniform(0, 5000, n))
    w_a = rng.uniform(0.5, 1.5, n)
    w_b = rng.uniform(0.5, 1.5, n)
    sigma = 30.0
    T_a = build_exp_tens(p_a, w_a, sigma, 3, False, False, 1200.0, verbose=False)
    T_b = build_exp_tens(p_b, w_b, sigma, 3, False, False, 1200.0, verbose=False)

    # Default auto path with default threshold: orbit (no fallback)
    cos_orbit = cos_sim_exp_tens(T_a, T_b, verbose=False)
    # Auto path with absurdly high threshold: every call falls through
    cos_forced = cos_sim_exp_tens(
        T_a, T_b, cancellation_threshold=1.0, verbose=False,
    )
    cos_pw = cos_sim_exp_tens(T_a, T_b, method='bulger', verbose=False)

    # The orbit estimate at this n, r already matches pairwise to FP, so
    # the test of "guard fired and rerouted" hinges on:
    # (a) cos_forced equals cos_pw bit-for-bit (the pairwise call branch
    #     must have run, not the orbit branch), and
    # (b) cos_forced does not depend on the orbit estimate at all.
    assert cos_forced == cos_pw, (
        f"Guard fallback did not reach Bulger's method: "
        f"forced={cos_forced!r}, pairwise={cos_pw!r}"
    )
    # And in healthy regime the Möbius method is also correct, so
    # cos_orbit and cos_forced should still agree to FP.
    assert abs(cos_orbit - cos_forced) < 1e-12


def test_threshold_zero_disables_guard():
    """cancellation_threshold=0.0 means no fallback ever triggers; the
    orbit estimate is used as-is.
    """
    rng = np.random.default_rng(seed=7)
    n = 16
    p_a = np.sort(rng.uniform(0, 4000, n))
    p_b = np.sort(rng.uniform(0, 4000, n))
    w_a = rng.uniform(0.5, 1.5, n)
    w_b = rng.uniform(0.5, 1.5, n)
    sigma = 25.0
    T_a = build_exp_tens(p_a, w_a, sigma, 3, False, False, 1200.0, verbose=False)
    T_b = build_exp_tens(p_b, w_b, sigma, 3, False, False, 1200.0, verbose=False)

    # With threshold=0 the orbit estimate is final; with the default
    # threshold the same input does not trigger the guard either; both
    # values should agree.
    cos_no_guard = cos_sim_exp_tens(
        T_a, T_b, cancellation_threshold=0.0, verbose=False,
    )
    cos_default = cos_sim_exp_tens(T_a, T_b, verbose=False)
    assert cos_no_guard == cos_default


# ----------------------------------------------------------------------
# The guard does not fire spuriously on identical operands
# ----------------------------------------------------------------------


def test_self_cosine_is_one_under_orbit_path():
    """<A,A>/sqrt(<A,A><A,A>) = 1 must hold exactly under the Möbius method."""
    rng = np.random.default_rng(seed=11)
    n = 12
    p_a = np.sort(rng.uniform(0, 5000, n))
    w_a = rng.uniform(0.5, 1.5, n)
    sigma = 30.0
    T_a = build_exp_tens(p_a, w_a, sigma, 3, False, False, 1200.0, verbose=False)
    cos_self = cos_sim_exp_tens(T_a, T_a, verbose=False)
    assert abs(cos_self - 1.0) < 1e-12


# ----------------------------------------------------------------------
# Inspect the orbit and pairwise inner-product triples on a healthy input
# ----------------------------------------------------------------------


def test_orbit_and_pairwise_triples_give_same_cosine():
    """The cosines computed from the orbit and pairwise triples agree to
    floating point.

    Note: the two paths use different normalisation conventions for the
    bare inner product. The Möbius method applies the ``(σ√π)^r`` prefactor
    and sums over ordered tuples on both sides; Bulger's method
    omits the prefactor and uses (ordered × unordered), differing by
    ``(σ√π)^r · r!``. Both cancel in the cosine ratio, so agreement at
    the cosine level — but not at the bare-value level — is the meaningful
    invariant.
    """
    rng = np.random.default_rng(seed=12345)
    n = 12
    p_a = np.sort(rng.uniform(0, 4000, n))
    p_b = np.sort(rng.uniform(0, 4000, n))
    w_a = rng.uniform(0.5, 1.5, n)
    w_b = rng.uniform(0.5, 1.5, n)
    sigma = 20.0
    T_a = build_exp_tens(p_a, w_a, sigma, 3, False, False, 1200.0, verbose=False)
    T_b = build_exp_tens(p_b, w_b, sigma, 3, False, False, 1200.0, verbose=False)

    ip_xy_o, ip_xx_o, ip_yy_o, _ = _cos_sim_exp_tens_sa_orbit(T_a, T_b)
    ip_xy_p, ip_xx_p, ip_yy_p = _cos_sim_exp_tens_sa_pairwise(T_a, T_b, verbose=False)

    cos_orbit = ip_xy_o / np.sqrt(ip_xx_o * ip_yy_o)
    cos_pw = ip_xy_p / np.sqrt(ip_xx_p * ip_yy_p)
    abs_err = abs(cos_orbit - cos_pw)
    rel_err = abs_err / max(abs(cos_orbit), abs(cos_pw), 1e-300)
    assert rel_err < 1e-10 or abs_err < 1e-12, (
        f"orbit cos={cos_orbit:.12e}, pairwise cos={cos_pw:.12e}, "
        f"abs_err={abs_err:.2e}, rel_err={rel_err:.2e}"
    )

    # The bare-value ratio between paths is fixed, mode by mode. In
    # absolute mode, the constant of proportionality is (σ√π)^r · r!.
    # We verify it here so any future drift in either path's
    # convention is caught. (For fixed inputs the same ratio applies
    # to all three triples.)
    from math import factorial
    expected_ratio = (sigma * np.sqrt(np.pi)) ** 3 * factorial(3)
    for label, vo, vp in [("xy", ip_xy_o, ip_xy_p),
                          ("xx", ip_xx_o, ip_xx_p),
                          ("yy", ip_yy_o, ip_yy_p)]:
        ratio = vo / vp
        assert abs(ratio - expected_ratio) / expected_ratio < 1e-10, (
            f"{label}: orbit/pairwise = {ratio:.6e}, "
            f"expected (σ√π)^r · r! = {expected_ratio:.6e}"
        )
