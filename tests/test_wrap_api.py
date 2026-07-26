"""Regression tests for the ``wrap=`` API (v3+).

Covers:
- Default is ``'full-image'`` on every attribute of every density.
- ``'single-image'`` opt-in is honoured — the numerics reproduce
  pre-v3 single-image behaviour (which is what the abs-per fast path
  used to give unconditionally at large sigma/P).
- Full-image cosine is bounded by 1 at large sigma/P, where the
  single-image kernel is no longer positive definite.
- ``wrap`` is validated: unknown strings error at build time, and
  per-attribute arrays must match the attribute count.
"""
import warnings

import numpy as np
import pytest

import mpt


PERIOD = 1200.0


# ---------------------------------------------------------------------
# Default behaviour
# ---------------------------------------------------------------------

def test_default_wrap_is_full_image_scalar():
    p = np.array([[0.0, 100.0, 300.0, 700.0]]).T
    w = np.ones(4).reshape(-1, 1)
    d = mpt.build_exp_tens(
        [p], [w], [60.0], [2], [False], [True], [PERIOD], verbose=False,
    )
    assert list(d.wrap) == ['full-image']


def test_default_wrap_is_full_image_multi_attr():
    p1 = np.array([[0.0, 100.0, 300.0]]).T
    p2 = np.array([[50.0, 200.0, 400.0]]).T
    w = np.ones(3).reshape(-1, 1)
    d = mpt.build_exp_tens(
        [p1, p2], [w, w],
        [60.0, 60.0], [2, 2],
        [False, False], [True, True], [PERIOD, PERIOD],
        verbose=False,
    )
    assert list(d.wrap) == ['full-image', 'full-image']


def test_wrap_scalar_broadcast():
    p1 = np.array([[0.0, 100.0, 300.0]]).T
    p2 = np.array([[50.0, 200.0, 400.0]]).T
    w = np.ones(3).reshape(-1, 1)
    d = mpt.build_exp_tens(
        [p1, p2], [w, w],
        [60.0, 60.0], [2, 2],
        [False, False], [True, True], [PERIOD, PERIOD],
        wrap='single-image', verbose=False,
    )
    assert list(d.wrap) == ['single-image', 'single-image']


def test_wrap_per_attribute():
    p1 = np.array([[0.0, 100.0, 300.0]]).T
    p2 = np.array([[50.0, 200.0, 400.0]]).T
    w = np.ones(3).reshape(-1, 1)
    d = mpt.build_exp_tens(
        [p1, p2], [w, w],
        [60.0, 60.0], [2, 2],
        [False, False], [True, True], [PERIOD, PERIOD],
        wrap=['full-image', 'single-image'], verbose=False,
    )
    assert list(d.wrap) == ['full-image', 'single-image']


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def test_wrap_unknown_string_raises():
    p = np.array([[0.0, 100.0, 300.0]]).T
    w = np.ones(3).reshape(-1, 1)
    with pytest.raises(ValueError):
        mpt.build_exp_tens(
            [p], [w], [60.0], [2], [False], [True], [PERIOD],
            wrap='torus', verbose=False,
        )


def test_wrap_array_wrong_length_raises():
    p1 = np.array([[0.0, 100.0, 300.0]]).T
    p2 = np.array([[50.0, 200.0, 400.0]]).T
    w = np.ones(3).reshape(-1, 1)
    with pytest.raises(ValueError):
        mpt.build_exp_tens(
            [p1, p2], [w, w],
            [60.0, 60.0], [2, 2],
            [False, False], [True, True], [PERIOD, PERIOD],
            wrap=['full-image'],  # too short
            verbose=False,
        )


# ---------------------------------------------------------------------
# Full-image is positive definite: cosine <= 1 at large sigma/P
# ---------------------------------------------------------------------

@pytest.mark.parametrize("sigma_over_P", [0.20, 0.30, 0.50])
def test_full_image_cosine_bounded_at_high_sigma_over_p(sigma_over_P):
    """The single-image (nearest-image only) kernel is not PD above
    sigma/P ~ 0.15, so its cosine can exceed 1. Full-image is a genuine
    torus overlap and PD; the cosine of any density with itself is
    exactly 1, and the cosine between two densities is <= 1.
    """
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
    c12 = mpt.cos_sim_exp_tens(d1, d2, verbose=False)
    c11 = mpt.cos_sim_exp_tens(d1, d1, verbose=False)
    c22 = mpt.cos_sim_exp_tens(d2, d2, verbose=False)
    assert c12 <= 1.0 + 1e-10
    assert abs(c11 - 1.0) < 1e-10
    assert abs(c22 - 1.0) < 1e-10


# ---------------------------------------------------------------------
# Single-image opt-in reproduces pre-v3 numerics at high sigma/P
# ---------------------------------------------------------------------

def test_single_image_matches_below_threshold():
    """At sigma/P below the accuracy-floor threshold, L = 0 in the
    full-image path so single-image and full-image are byte-identical.
    """
    p1 = np.array([[0.0, 100.0, 300.0, 700.0]]).T
    p2 = np.array([[50.0, 250.0, 500.0, 900.0]]).T
    w = np.ones(4).reshape(-1, 1)
    sigma = 0.03 * PERIOD  # well below threshold
    d_full = mpt.build_exp_tens(
        [p1], [w], [sigma], [2], [False], [True], [PERIOD], verbose=False,
    )
    d_single = mpt.build_exp_tens(
        [p1], [w], [sigma], [2], [False], [True], [PERIOD],
        wrap='single-image', verbose=False,
    )
    e_full = mpt.build_exp_tens(
        [p2], [w], [sigma], [2], [False], [True], [PERIOD], verbose=False,
    )
    e_single = mpt.build_exp_tens(
        [p2], [w], [sigma], [2], [False], [True], [PERIOD],
        wrap='single-image', verbose=False,
    )
    c_full = mpt.cos_sim_exp_tens(d_full, e_full, verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        c_single = mpt.cos_sim_exp_tens(d_single, e_single, verbose=False)
    assert abs(c_full - c_single) < 1e-12


def test_single_image_differs_from_full_at_high_sigma():
    """Above the threshold single-image and full-image are genuinely
    different measures (the whole point of the wrap axis).
    """
    p1 = np.array([[0.0, 100.0, 300.0, 700.0]]).T
    p2 = np.array([[50.0, 250.0, 500.0, 900.0]]).T
    w = np.ones(4).reshape(-1, 1)
    sigma = 0.20 * PERIOD  # well above threshold
    d_full = mpt.build_exp_tens(
        [p1], [w], [sigma], [2], [False], [True], [PERIOD], verbose=False,
    )
    d_single = mpt.build_exp_tens(
        [p1], [w], [sigma], [2], [False], [True], [PERIOD],
        wrap='single-image', verbose=False,
    )
    e_full = mpt.build_exp_tens(
        [p2], [w], [sigma], [2], [False], [True], [PERIOD], verbose=False,
    )
    e_single = mpt.build_exp_tens(
        [p2], [w], [sigma], [2], [False], [True], [PERIOD],
        wrap='single-image', verbose=False,
    )
    c_full = mpt.cos_sim_exp_tens(d_full, e_full, verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        c_single = mpt.cos_sim_exp_tens(d_single, e_single, verbose=False)
    # Genuinely different at high sigma/P
    assert abs(c_full - c_single) > 1e-6


# ---------------------------------------------------------------------
# Wrap axis is quiet for non-periodic and relative-periodic attributes
# ---------------------------------------------------------------------

def test_wrap_irrelevant_for_non_periodic():
    """A non-periodic attribute has no periodic images to sum; wrap is
    accepted but has no numerical effect.
    """
    p = np.array([[0.0, 100.0, 300.0, 700.0]]).T
    w = np.ones(4).reshape(-1, 1)
    d_full = mpt.build_exp_tens(
        [p], [w], [50.0], [2], [False], [False], [0.0], verbose=False,
    )
    d_single = mpt.build_exp_tens(
        [p], [w], [50.0], [2], [False], [False], [0.0],
        wrap='single-image', verbose=False,
    )
    c_full = mpt.cos_sim_exp_tens(d_full, d_full, verbose=False)
    c_single = mpt.cos_sim_exp_tens(d_single, d_single, verbose=False)
    assert abs(c_full - c_single) < 1e-12
    assert abs(c_full - 1.0) < 1e-12


def test_wrap_affects_rel_per_at_high_sigma():
    """v3+: rel-per honours ``wrap=`` at high σ/P. Full-image (default)
    gives (C); single-image forces dispatch through the centres/pairwise
    route, giving (A). The two measures diverge above the threshold.
    """
    p1 = np.array([[0.0, 100.0, 300.0, 700.0]]).T
    p2 = np.array([[50.0, 250.0, 500.0, 900.0]]).T
    w = np.ones(4).reshape(-1, 1)
    sigma = 0.20 * PERIOD
    d1a = mpt.build_exp_tens(
        [p1], [w], [sigma], [2], [True], [True], [PERIOD], verbose=False,
    )
    d1b = mpt.build_exp_tens(
        [p1], [w], [sigma], [2], [True], [True], [PERIOD],
        wrap='single-image', verbose=False,
    )
    d2 = mpt.build_exp_tens(
        [p2], [w], [sigma], [2], [True], [True], [PERIOD], verbose=False,
    )
    d2b = mpt.build_exp_tens(
        [p2], [w], [sigma], [2], [True], [True], [PERIOD],
        wrap='single-image', verbose=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        c_full = mpt.cos_sim_exp_tens(d1a, d2, verbose=False)
        c_single = mpt.cos_sim_exp_tens(d1b, d2b, verbose=False)
    # Full-image (C) and single-image (A) differ above the threshold.
    assert abs(c_full - c_single) > 1e-6


def test_wrap_matches_between_rel_per_measures_below_threshold():
    """Below the sigma/P threshold, (A) and (C) numerically agree, so
    the wrap choice is invisible in rel-per just as in abs-per."""
    p1 = np.array([[0.0, 100.0, 300.0, 700.0]]).T
    p2 = np.array([[50.0, 250.0, 500.0, 900.0]]).T
    w = np.ones(4).reshape(-1, 1)
    sigma = 0.02 * PERIOD  # well below threshold
    d1a = mpt.build_exp_tens(
        [p1], [w], [sigma], [2], [True], [True], [PERIOD], verbose=False,
    )
    d1b = mpt.build_exp_tens(
        [p1], [w], [sigma], [2], [True], [True], [PERIOD],
        wrap='single-image', verbose=False,
    )
    d2 = mpt.build_exp_tens(
        [p2], [w], [sigma], [2], [True], [True], [PERIOD], verbose=False,
    )
    d2b = mpt.build_exp_tens(
        [p2], [w], [sigma], [2], [True], [True], [PERIOD],
        wrap='single-image', verbose=False,
    )
    c_full = mpt.cos_sim_exp_tens(d1a, d2, verbose=False)
    c_single = mpt.cos_sim_exp_tens(d1b, d2b, verbose=False)
    assert abs(c_full - c_single) < 1e-8
