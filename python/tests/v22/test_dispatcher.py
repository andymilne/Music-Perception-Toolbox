"""Tests for the v2.2 single-attribute inner-product dispatcher.

The dispatcher (``_select_sa_inner_product_method``) decides between the
orbit path and the pairwise path based on (r, n, mode, σ/P) when the
user passes ``method='auto'``. These tests verify that each branch of
the decision tree fires as documented.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from mpt.tensor import (
    _ORBIT_R_MAX_SHIPPED,
    _ORBIT_SIGMA_OVER_P_THRESHOLD,
    _select_sa_inner_product_method,
)


# ----------------------------------------------------------------------
# r-based decisions
# ----------------------------------------------------------------------


def test_r1_routes_pairwise():
    """r=1 has no distinct-index structure; orbit machinery is undefined."""
    assert _select_sa_inner_product_method(
        r=1, n_max=20, is_rel=False, is_per=False,
        sigma_over_P=0.0, user_method='auto',
    ) == 'pairwise'


@pytest.mark.parametrize("n_max", [2, 4, 6, 8])
def test_r2_small_n_routes_pairwise(n_max):
    """r=2 with small n: pairwise dominates (orbit overhead exceeds benefit)."""
    assert _select_sa_inner_product_method(
        r=2, n_max=n_max, is_rel=False, is_per=False,
        sigma_over_P=0.0, user_method='auto',
    ) == 'pairwise'


@pytest.mark.parametrize("n_max", [9, 12, 32, 64])
def test_r2_large_n_routes_orbit(n_max):
    """r=2 with n>8: orbit path is selected."""
    assert _select_sa_inner_product_method(
        r=2, n_max=n_max, is_rel=False, is_per=False,
        sigma_over_P=0.0, user_method='auto',
    ) == 'orbit'


@pytest.mark.parametrize("r", [3, 4, 5, 6])
def test_r3_to_r6_routes_orbit(r):
    """At r=3..6 with shipped orbit tables, auto picks orbit."""
    assert _select_sa_inner_product_method(
        r=r, n_max=12, is_rel=False, is_per=False,
        sigma_over_P=0.0, user_method='auto',
    ) == 'orbit'


@pytest.mark.parametrize("r", [_ORBIT_R_MAX_SHIPPED + 1, _ORBIT_R_MAX_SHIPPED + 2])
def test_r_beyond_shipped_routes_pairwise(r):
    """Above the shipped-table cutoff, auto falls back to pairwise.

    Phase 5 will extend shipped tables to r=7, 8; until then, auto avoids
    the on-demand build cost surprise (~16 s at r=7, ~3 min at r=8).
    """
    assert _select_sa_inner_product_method(
        r=r, n_max=12, is_rel=False, is_per=False,
        sigma_over_P=0.0, user_method='auto',
    ) == 'pairwise'


# ----------------------------------------------------------------------
# Periodic-relative σ/P branch
# ----------------------------------------------------------------------


@pytest.mark.parametrize("sop", [0.001, 0.01, 0.02, _ORBIT_SIGMA_OVER_P_THRESHOLD])
def test_perrel_below_threshold_routes_orbit(sop):
    """Periodic-relative with σ/P at or below the threshold uses orbit."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would be a regression
        chosen = _select_sa_inner_product_method(
            r=3, n_max=12, is_rel=True, is_per=True,
            sigma_over_P=sop, user_method='auto',
        )
    assert chosen == 'orbit'


@pytest.mark.parametrize("sop", [0.031, 0.05, 0.1, 0.2])
def test_perrel_above_threshold_routes_pairwise_with_warning(sop):
    """Periodic-relative beyond σ/P threshold falls back to pairwise + warns."""
    with pytest.warns(UserWarning, match=r"σ/P = .* exceeds the orbit-path threshold"):
        chosen = _select_sa_inner_product_method(
            r=3, n_max=12, is_rel=True, is_per=True,
            sigma_over_P=sop, user_method='auto',
        )
    assert chosen == 'pairwise'


def test_perrel_threshold_warning_only_when_relevant():
    """Only periodic-relative mode triggers the σ/P warning, not other modes."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # is_rel=False, is_per=True with high σ/P: no warning, route is orbit
        chosen = _select_sa_inner_product_method(
            r=3, n_max=12, is_rel=False, is_per=True,
            sigma_over_P=0.5, user_method='auto',
        )
    assert chosen == 'orbit'


# ----------------------------------------------------------------------
# User overrides bypass the dispatcher
# ----------------------------------------------------------------------


@pytest.mark.parametrize("forced", ['pairwise', 'direct', 'orbit'])
def test_user_method_bypasses_dispatcher(forced):
    """When the user passes an explicit method, the dispatcher returns it
    unchanged (irrespective of r, n, mode, or σ/P).
    """
    chosen = _select_sa_inner_product_method(
        r=3, n_max=12, is_rel=False, is_per=False,
        sigma_over_P=0.0, user_method=forced,
    )
    assert chosen == forced


def test_user_method_pairwise_avoids_perrel_warning():
    """Passing method='pairwise' explicitly silences the σ/P warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        chosen = _select_sa_inner_product_method(
            r=3, n_max=12, is_rel=True, is_per=True,
            sigma_over_P=0.5, user_method='pairwise',
        )
    assert chosen == 'pairwise'
