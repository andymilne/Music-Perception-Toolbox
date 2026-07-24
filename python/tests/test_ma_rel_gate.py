"""Regression tests for the centres-vs-grid gate on the multi-attribute
relative path (``_ma_rel_attr_prefers_centres``).

The gate compares predicted wall time on the two paths. Its calibration
notes (see the constants ``_CENTRES_NS_BASE`` etc. in
``mpt._tensor.cosine``) explain the empirical fit. The cells below pin
its decisions on the labelled truth set from that fit, including the
r=2 band where the previous raw-op-count gate over-selected centres by
one to two orders of magnitude in wall time.

Each case is one (K, r, sigma, span_or_period, is_per, expect_centres)
tuple. ``expect_centres`` is what the calibrated model predicts and
also what timing measurements agree with (in the cells verified to
within the crossover neighbourhood).
"""
import math

import numpy as np
import pytest

from mpt._tensor.cosine import (
    _ma_rel_attr_prefers_centres,
    _predicted_centres_wall_ns,
    _predicted_grid_wall_ns,
)


def _make_events(K, span, seed=0):
    """Non-periodic side: pitches uniform on [0, span). Column of Px."""
    rng = np.random.default_rng(seed)
    return np.sort(rng.uniform(0.0, span, size=K)).reshape(K, 1)


# ---------------------------------------------------------------------------
# Non-periodic cells — the r=2 band where the previous gate misfired
# ---------------------------------------------------------------------------

# The old gate said centres for every one of these; the new gate must
# say grid because the closed-form path is 10x-100x slower there.
_NONPER_R2_GRID = [
    # (K, sigma, span_per_side)
    (20, 15.0, 1800.0),
    (30, 15.0, 1800.0),
    (50, 15.0, 1800.0),
    (70, 15.0, 1800.0),
    (30, 30.0, 1800.0),
    (40, 5.0, 1800.0),
    (50, 5.0, 1800.0),
]


@pytest.mark.parametrize("K, sigma, span_per_side", _NONPER_R2_GRID)
def test_nonper_r2_prefers_grid(K, sigma, span_per_side):
    Px = _make_events(K, span_per_side, seed=1)
    Py = _make_events(K, span_per_side, seed=2)
    assert not _ma_rel_attr_prefers_centres(
        Px, Py, sigma, 2, True, False, 0.0
    ), f"K={K} sigma={sigma}: centres path is much slower than grid here"


# Non-periodic cells where centres is genuinely faster.
_NONPER_CENTRES = [
    # (K, r, sigma, span_per_side)
    (5, 2, 15.0, 1800.0),
    (8, 2, 15.0, 1800.0),
    (12, 2, 15.0, 1800.0),
    (6, 3, 15.0, 1800.0),
    (8, 3, 15.0, 1800.0),
    (5, 4, 15.0, 1800.0),
    (6, 4, 15.0, 1800.0),
]


@pytest.mark.parametrize("K, r, sigma, span_per_side", _NONPER_CENTRES)
def test_nonper_prefers_centres(K, r, sigma, span_per_side):
    Px = _make_events(K, span_per_side, seed=1)
    Py = _make_events(K, span_per_side, seed=2)
    assert _ma_rel_attr_prefers_centres(
        Px, Py, sigma, r, True, False, 0.0
    ), f"K={K} r={r} sigma={sigma}: centres path is cheaper here"


# ---------------------------------------------------------------------------
# Periodic cells
# ---------------------------------------------------------------------------

# Periodic cells where the gate must pick grid.
_PER_GRID = [
    # (K, r, sigma, period)
    (30, 2, 15.0, 3600.0),
    (50, 2, 15.0, 3600.0),
    (12, 3, 15.0, 3600.0),
    (10, 3, 5.0, 1200.0),
    (8, 4, 15.0, 3600.0),
]


@pytest.mark.parametrize("K, r, sigma, period", _PER_GRID)
def test_per_prefers_grid(K, r, sigma, period):
    Px = _make_events(K, period, seed=1)
    Py = _make_events(K, period, seed=2)
    assert not _ma_rel_attr_prefers_centres(
        Px, Py, sigma, r, True, True, period
    ), f"K={K} r={r} sigma={sigma} P={period}: grid is cheaper here"


# Periodic cells where centres is cheaper.
_PER_CENTRES = [
    # (K, r, sigma, period)
    (5, 2, 15.0, 3600.0),
    (8, 2, 15.0, 3600.0),
    (6, 3, 15.0, 3600.0),
    (8, 3, 15.0, 3600.0),
    (5, 4, 15.0, 3600.0),
    (6, 4, 15.0, 3600.0),
]


@pytest.mark.parametrize("K, r, sigma, period", _PER_CENTRES)
def test_per_prefers_centres(K, r, sigma, period):
    Px = _make_events(K, period, seed=1)
    Py = _make_events(K, period, seed=2)
    assert _ma_rel_attr_prefers_centres(
        Px, Py, sigma, r, True, True, period
    ), f"K={K} r={r} sigma={sigma} P={period}: centres is cheaper here"


# ---------------------------------------------------------------------------
# Trivial-early-exit cases (invariant under any calibration)
# ---------------------------------------------------------------------------

def test_gate_returns_false_for_absolute_attribute():
    Px = _make_events(20, 1800.0)
    Py = _make_events(20, 1800.0)
    assert _ma_rel_attr_prefers_centres(Px, Py, 15.0, 2, False, False, 0.0) is False


def test_gate_returns_false_for_r_below_two():
    Px = _make_events(20, 1800.0)
    Py = _make_events(20, 1800.0)
    assert _ma_rel_attr_prefers_centres(Px, Py, 15.0, 1, True, False, 0.0) is False


def test_gate_returns_false_when_K_below_r():
    # r=3 needs at least K=3, r=4 at least K=4.
    Px = _make_events(2, 1800.0)
    Py = _make_events(2, 1800.0)
    assert _ma_rel_attr_prefers_centres(Px, Py, 15.0, 3, True, False, 0.0) is False
    Px = _make_events(3, 1800.0)
    Py = _make_events(3, 1800.0)
    assert _ma_rel_attr_prefers_centres(Px, Py, 15.0, 4, True, False, 0.0) is False


def test_gate_returns_false_above_sigma_over_P_threshold():
    # Above σ/P = 0.03 the centres route reads the minimum image while
    # the grid route reads the all-image torus, so the gate must not
    # flip in that band regardless of cost.
    Px = _make_events(8, 100.0)
    Py = _make_events(8, 100.0)
    # sigma/period = 0.10 -> above threshold
    assert not _ma_rel_attr_prefers_centres(
        Px, Py, 10.0, 2, True, True, 100.0
    )


# ---------------------------------------------------------------------------
# Cost model sanity: predictions must be finite and positive.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("K, r, is_per", [
    (5, 2, False), (15, 2, False), (60, 2, False),
    (6, 3, False), (12, 3, False),
    (5, 4, False), (8, 4, False),
    (5, 2, True), (15, 2, True), (60, 2, True),
    (5, 5, False), (6, 6, False),  # extrapolation regime
])
def test_predicted_walls_finite_and_positive(K, r, is_per):
    c = _predicted_centres_wall_ns(K, r, is_per)
    g = _predicted_grid_wall_ns(K, r, 15.0, 3600.0, is_per)
    assert np.isfinite(c) and c > 0
    assert np.isfinite(g) and g > 0
