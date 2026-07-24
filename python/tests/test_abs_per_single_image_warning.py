"""Tests for the absolute-periodic single-image warning.

The absolute-periodic kernel wraps each difference to its nearest image.
Below ``_ABS_PER_SIGMA_OVER_P_THRESHOLD`` that agrees with the
full-image measure to within the toolbox's own accuracy floor; above it
the two diverge, and above roughly sigma/P = 0.15 the single-image
kernel also stops being positive definite. The warning marks the
crossing.

It is raised at density construction, because the choice is a property
of the density rather than of any one operation.
"""
import warnings

import numpy as np
import pytest

import mpt
from mpt._tensor.dispatch import _ABS_PER_SIGMA_OVER_P_THRESHOLD

PERIOD = 1200.0
_MARKER = "absolute-periodic"


def _build(sigma, is_rel, is_per, period=PERIOD, r=2):
    p = np.array([0.0, 100.0, 300.0, 700.0]).reshape(-1, 1)
    w = np.ones(4).reshape(-1, 1)
    return mpt.build_exp_tens(
        [p], [w], [sigma], [r], [is_rel], [is_per], [period], verbose=False
    )


def _warns(sigma, is_rel, is_per, period=PERIOD, r=2):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _build(sigma, is_rel, is_per, period, r)
    return [w for w in caught if _MARKER in str(w.message)]


# ---------------------------------------------------------------------
# Fires where it should
# ---------------------------------------------------------------------

@pytest.mark.parametrize("sigma_over_P", [0.06, 0.10, 0.20, 0.30, 0.50])
def test_warns_above_threshold(sigma_over_P):
    assert _warns(sigma_over_P * PERIOD, False, True)


@pytest.mark.parametrize("sigma_over_P", [0.001, 0.0125, 0.02, 0.03, 0.04])
def test_silent_below_threshold(sigma_over_P):
    assert not _warns(sigma_over_P * PERIOD, False, True)


def test_silent_exactly_at_threshold():
    # Strict inequality: the threshold value itself does not warn.
    assert not _warns(_ABS_PER_SIGMA_OVER_P_THRESHOLD * PERIOD, False, True)


# ---------------------------------------------------------------------
# Does not fire where it should not
# ---------------------------------------------------------------------

def test_silent_for_relative_periodic():
    # The relative-periodic case has its own warning and its own
    # threshold; this one must not double up on it.
    assert not _warns(0.30 * PERIOD, True, True)


def test_silent_for_absolute_non_periodic():
    assert not _warns(0.30 * PERIOD, False, False, period=0.0)


def test_silent_for_relative_non_periodic():
    assert not _warns(0.30 * PERIOD, True, False, period=0.0)


# ---------------------------------------------------------------------
# Message content
# ---------------------------------------------------------------------

def test_message_reports_the_ratio_and_threshold():
    hits = _warns(0.20 * PERIOD, False, True)
    assert hits
    msg = str(hits[0].message)
    assert "0.200" in msg
    assert str(_ABS_PER_SIGMA_OVER_P_THRESHOLD) in msg


def test_message_names_the_positive_definiteness_consequence():
    # The character of the departure matters as much as its size: above
    # roughly sigma/P = 0.15 the cosine is no longer bounded by 1.
    msg = str(_warns(0.20 * PERIOD, False, True)[0].message)
    assert "positive definite" in msg
    assert "bounded by 1" in msg


def test_message_offers_no_alternative_method():
    # Unlike the relative-periodic warning, there is no full-image route
    # to point at, so the message must not imply one exists.
    msg = str(_warns(0.20 * PERIOD, False, True)[0].message)
    assert "method=" not in msg


# ---------------------------------------------------------------------
# Robustness: a diagnostic must never break construction
# ---------------------------------------------------------------------

@pytest.mark.parametrize("r", [1, 2, 3])
def test_construction_succeeds_at_every_r(r):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dens = _build(0.20 * PERIOD, False, True, r=r)
    assert dens is not None


def test_multi_attribute_warns_once_per_offending_attribute():
    p = np.array([0.0, 100.0, 300.0, 700.0]).reshape(-1, 1)
    w = np.ones(4).reshape(-1, 1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mpt.build_exp_tens(
            [p, p], [w, w],
            [0.20 * PERIOD, 0.01 * PERIOD],   # first offends, second does not
            [2, 2], [False, False], [True, True], [PERIOD, PERIOD],
            verbose=False,
        )
    hits = [w for w in caught if _MARKER in str(w.message)]
    assert len(hits) == 1
