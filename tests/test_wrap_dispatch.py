"""Regression tests for the abs-per full-image spectral dispatch (v3+).

Verifies that the shared wrapped-Gaussian helper's automatic
image-sum vs Fourier dispatch produces indistinguishable numerical
answers at each side of the crossover, and that the wrapped-Gaussian
value itself matches an independent reference (the reference module's
lattice sum) to machine precision within the accuracy floor.
"""
import warnings

import numpy as np
import pytest

import mpt
from mpt._wrapped_kernel import (
    wrapped_gaussian_1d,
    _image_count_L,
    _fourier_count_M,
    _prefer_fourier,
)


PERIOD = 1200.0


# ---------------------------------------------------------------------
# 1D helper: image-sum and Fourier agree within the accuracy floor
# ---------------------------------------------------------------------

@pytest.mark.parametrize("sigma_over_P", [0.02, 0.05, 0.10, 0.15,
                                          0.20, 0.30, 0.50])
def test_image_sum_vs_fourier_agree_within_floor(sigma_over_P):
    """Force each 1D route and check they agree within the accuracy
    floor (default truncation_sigmas = 6, so floor ~ 10^-8)."""
    import mpt._wrapped_kernel as wk

    sigma = sigma_over_P * PERIOD
    d = np.linspace(-PERIOD, PERIOD, 41)
    ts = 6.0

    # Force image-sum
    orig = wk._prefer_fourier
    wk._prefer_fourier = lambda *a, **kw: False
    theta_img = wrapped_gaussian_1d(d, sigma, PERIOD, ts,
                                    exponent_denominator=4)
    # Force Fourier
    wk._prefer_fourier = lambda *a, **kw: True
    theta_fou = wrapped_gaussian_1d(d, sigma, PERIOD, ts,
                                    exponent_denominator=4)
    wk._prefer_fourier = orig

    # Both must be within the accuracy floor of each other
    from mpt._defaults import truncation_floor
    floor = truncation_floor(ts)
    max_err = float(np.max(np.abs(theta_img - theta_fou)))
    assert max_err < floor * 100  # generous margin


# ---------------------------------------------------------------------
# Crossover is where derived (~ sigma/P = 0.2)
# ---------------------------------------------------------------------

def test_dispatch_prefers_image_sum_at_small_sigma():
    """At sigma/P well below the derived 0.24 crossover, image-sum has
    the smaller grid width."""
    sigma = 0.05 * PERIOD
    assert not _prefer_fourier(sigma, PERIOD, 6.0,
                               exponent_denominator=4)
    L = _image_count_L(sigma, PERIOD, 6.0, exponent_denominator=4)
    M = _fourier_count_M(sigma, PERIOD, 6.0, exponent_denominator=4)
    assert (2 * L + 1) <= M


def test_dispatch_prefers_fourier_at_large_sigma():
    """At sigma/P above the crossover, Fourier has the smaller grid."""
    sigma = 0.30 * PERIOD
    assert _prefer_fourier(sigma, PERIOD, 6.0,
                           exponent_denominator=4)
    L = _image_count_L(sigma, PERIOD, 6.0, exponent_denominator=4)
    M = _fourier_count_M(sigma, PERIOD, 6.0, exponent_denominator=4)
    assert M < (2 * L + 1)


# ---------------------------------------------------------------------
# End-to-end: cosine values are dispatch-invariant
# ---------------------------------------------------------------------

@pytest.mark.parametrize("sigma_over_P", [0.10, 0.20, 0.30])
def test_cosine_invariant_under_dispatch(sigma_over_P):
    """Force each dispatch and confirm cos_sim gives the same value."""
    import mpt._wrapped_kernel as wk

    p1 = np.array([[0.0, 100.0, 300.0, 700.0]]).T
    p2 = np.array([[50.0, 250.0, 500.0, 900.0]]).T
    w = np.ones(4).reshape(-1, 1)
    sigma = sigma_over_P * PERIOD
    d1 = mpt.build_exp_tens(
        [p1], [w], [sigma], [2], [False], [True], [PERIOD], verbose=False,
    )
    d2 = mpt.build_exp_tens(
        [p2], [w], [sigma], [2], [False], [True], [PERIOD], verbose=False,
    )

    # Force image-sum
    orig = wk._prefer_fourier
    wk._prefer_fourier = lambda *a, **kw: False
    c_img = mpt.cos_sim_exp_tens(d1, d2, verbose=False)
    # Force Fourier
    wk._prefer_fourier = lambda *a, **kw: True
    c_fou = mpt.cos_sim_exp_tens(d1, d2, verbose=False)
    wk._prefer_fourier = orig

    # Same cosine either way, within accuracy floor
    assert abs(c_img - c_fou) < 1e-6


# ---------------------------------------------------------------------
# Independent reference cross-check
# ---------------------------------------------------------------------

def test_wrapped_gaussian_matches_reference_lattice():
    """The helper's 1D wrapped Gaussian matches the reference module's
    lattice-sum implementation to within the accuracy floor, at both
    sides of the crossover."""
    from tests.references.ref_general import kernel_all_image_abs

    for sigma_over_P in [0.05, 0.30]:
        sigma = sigma_over_P * PERIOD
        d = np.linspace(-PERIOD/2, PERIOD/2, 21)
        # 1D reference: kernel_all_image_abs takes shape (..., r); for
        # r=1 the return is a product over r=1, i.e. the 1D theta.
        d_2d = d[:, None]  # (21, 1)
        ref = kernel_all_image_abs(d_2d, sigma, PERIOD, tol=1e-15)
        got = wrapped_gaussian_1d(d, sigma, PERIOD, 6.0,
                                  exponent_denominator=4)
        max_err = float(np.max(np.abs(got - ref)))
        assert max_err < 1e-7, (
            f"sigma/P={sigma_over_P}: max_err={max_err:.3e}"
        )
