"""Tests for the v2.2 ``method`` keyword on ``cos_sim_exp_tens``.

Each documented value of ``method`` must produce the documented
behaviour. With perceptually typical parameters all three values
(``'auto'``, ``'bulger'``, ``'direct'``) must agree to floating-point
precision; the keyword exists as an escape hatch, not as a knob.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from mpt import build_exp_tens, cos_sim_exp_tens, cos_sim_exp_tens_raw


# Parameter grids for the comparison tests
_R_VALUES = [2, 3, 4]
_N_VALUES = [10, 12]  # r=4 with n>12 spikes _ip_full memory into the GBs


def _make_pair(rng, n, periodic=True, period=1200.0):
    if periodic:
        p_a = np.sort(rng.uniform(50, period - 50, n))
        p_b = np.sort(rng.uniform(50, period - 50, n))
    else:
        p_a = np.sort(rng.uniform(0, 5000, n))
        p_b = np.sort(rng.uniform(0, 5000, n))
    w_a = rng.uniform(0.5, 1.5, n)
    w_b = rng.uniform(0.5, 1.5, n)
    return p_a, w_a, p_b, w_b


# ----------------------------------------------------------------------
# auto vs pairwise vs direct must agree in normal use
# ----------------------------------------------------------------------


@pytest.mark.parametrize("r", _R_VALUES)
@pytest.mark.parametrize("n", _N_VALUES)
@pytest.mark.parametrize("is_per", [False, True])
@pytest.mark.parametrize("is_rel", [False, True])
def test_method_values_agree_in_normal_use(r, n, is_per, is_rel):
    """auto, pairwise, and direct produce the same cosine to ~1e-10 relative.

    With σ = 12 cents and P = 1200 cents (σ/P = 0.01), all three paths
    must agree. Any larger discrepancy indicates the orbit dispatcher
    or one of the documented escape hatches has drifted.
    """
    rng = np.random.default_rng(seed=hash((r, n, is_per, is_rel)) & 0xFFFF)
    p_a, w_a, p_b, w_b = _make_pair(rng, n, periodic=is_per)
    sigma, P = 12.0, 1200.0
    T_a = build_exp_tens(p_a, w_a, sigma, r, is_rel, is_per, P, verbose=False)
    T_b = build_exp_tens(p_b, w_b, sigma, r, is_rel, is_per, P, verbose=False)

    cos_auto = cos_sim_exp_tens(T_a, T_b, method='auto', verbose=False)
    cos_pw = cos_sim_exp_tens(T_a, T_b, method='bulger', verbose=False)
    cos_dir = cos_sim_exp_tens(T_a, T_b, method='direct', verbose=False)

    # Pairwise and direct route through _ip_core in SA mode and so are
    # bit-identical. Auto routes through orbit at r >= 3 or large n.
    assert cos_pw == cos_dir, (
        f"r={r}, n={n}: pairwise={cos_pw}, direct={cos_dir} differ"
    )
    abs_err = abs(cos_auto - cos_pw)
    rel_err = abs_err / max(abs(cos_auto), abs(cos_pw), 1e-300)
    # The absolute escape hatch sits at the translation-grid quadrature
    # noise floor (~1e-12 at 10 nodes per sigma): for near-orthogonal
    # pairs (cosine ~1e-3 and below) that dust dominates the relative
    # error, and its phase depends on grid placement, so the relative
    # criterion alone would fail on quadrature phase rather than
    # dispatcher drift.
    assert rel_err < 1e-10 or abs_err < 1e-11, (
        f"r={r}, n={n}, rel={is_rel}, per={is_per}: "
        f"auto={cos_auto:.10e}, pairwise={cos_pw:.10e}, "
        f"abs_err={abs_err:.2e}, rel_err={rel_err:.2e}"
    )


# ----------------------------------------------------------------------
# Default keyword is 'auto'
# ----------------------------------------------------------------------


def test_default_method_is_auto():
    """Calling without method keyword equals method='auto'."""
    rng = np.random.default_rng(seed=2026)
    p_a, w_a, p_b, w_b = _make_pair(rng, 12, periodic=True)
    sigma, P = 12.0, 1200.0
    T_a = build_exp_tens(p_a, w_a, sigma, 3, False, True, P, verbose=False)
    T_b = build_exp_tens(p_b, w_b, sigma, 3, False, True, P, verbose=False)
    cos_default = cos_sim_exp_tens(T_a, T_b, verbose=False)
    cos_auto = cos_sim_exp_tens(T_a, T_b, method='auto', verbose=False)
    assert cos_default == cos_auto


# ----------------------------------------------------------------------
# Invalid method values are rejected
# ----------------------------------------------------------------------


@pytest.mark.parametrize("bad", ['fast', 'naive', '', 'AUTO', 'Pairwise', None])
def test_invalid_method_raises(bad):
    """Any unrecognised method value must raise ValueError."""
    rng = np.random.default_rng(seed=3)
    p_a, w_a, p_b, w_b = _make_pair(rng, 8)
    sigma, P = 12.0, 1200.0
    T_a = build_exp_tens(p_a, w_a, sigma, 3, False, True, P, verbose=False)
    T_b = build_exp_tens(p_b, w_b, sigma, 3, False, True, P, verbose=False)
    with pytest.raises(ValueError, match="method must be one of"):
        cos_sim_exp_tens(T_a, T_b, method=bad, verbose=False)


# ----------------------------------------------------------------------
# Forced pairwise silences the periodic-relative warning
# ----------------------------------------------------------------------


def test_forced_pairwise_silences_perrel_warning():
    """method='bulger' must not emit the σ/P fallback warning."""
    rng = np.random.default_rng(seed=42)
    n = 12
    P = 1200.0
    sigma = 80.0  # σ/P ≈ 0.067, well above 0.03
    p_a, w_a, p_b, w_b = _make_pair(rng, n, periodic=True, period=P)
    T_a = build_exp_tens(p_a, w_a, sigma, 3, True, True, P, verbose=False)
    T_b = build_exp_tens(p_b, w_b, sigma, 3, True, True, P, verbose=False)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cos_sim_exp_tens(T_a, T_b, method='bulger', verbose=False)


def test_auto_warns_in_perrel_high_sigma_over_P():
    """method='auto' issues a warning when σ/P exceeds the orbit threshold."""
    rng = np.random.default_rng(seed=42)
    n = 12
    P = 1200.0
    sigma = 80.0
    p_a, w_a, p_b, w_b = _make_pair(rng, n, periodic=True, period=P)
    T_a = build_exp_tens(p_a, w_a, sigma, 3, True, True, P, verbose=False)
    T_b = build_exp_tens(p_b, w_b, sigma, 3, True, True, P, verbose=False)

    with pytest.warns(UserWarning, match=r"σ/P = .* exceeds"):
        cos_sim_exp_tens(T_a, T_b, method='auto', verbose=False)


# ----------------------------------------------------------------------
# cos_sim_exp_tens_raw forwards the keywords
# ----------------------------------------------------------------------


def test_raw_function_forwards_method():
    """cos_sim_exp_tens_raw must forward method= to cos_sim_exp_tens."""
    rng = np.random.default_rng(seed=99)
    p_a, w_a, p_b, w_b = _make_pair(rng, 12, periodic=False)
    sigma = 30.0
    cos_auto = cos_sim_exp_tens_raw(
        p_a, w_a, p_b, w_b, sigma, 3, False, False, 1200.0,
        method='auto', verbose=False,
    )
    cos_pw = cos_sim_exp_tens_raw(
        p_a, w_a, p_b, w_b, sigma, 3, False, False, 1200.0,
        method='bulger', verbose=False,
    )
    abs_err = abs(cos_auto - cos_pw)
    assert abs_err < 1e-12, (
        f"raw forward broken: auto={cos_auto}, pw={cos_pw}, |diff|={abs_err}"
    )
