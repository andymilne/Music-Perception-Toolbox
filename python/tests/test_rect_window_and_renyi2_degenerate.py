"""Regression tests for two v3 out-of-band fixes.

#20  weight_events' rectangular window uses a half-open support
     [c - W/2, c + W/2): a regular pulse grid yields exactly N pulses for
     full support N*IOI at every N (a closed interval over-counts even
     widths and can leave a between-pulse centre empty).

#21  renyi2 returns NaN for a zero-mass density (e.g. an out-of-support
     windowed sweep centre) rather than raising.
"""
import numpy as np
import pytest

import mpt
from mpt._tensor.preprocessing import weight_events


def _pulse_count(centre, width, n=9):
    """Number of pulses retained by a rectangular window on a unit grid."""
    pitch = np.arange(n, dtype=float).reshape(1, n)
    time = np.arange(n, dtype=float).reshape(1, n)
    _, w_out, _ = weight_events(
        [pitch, time], None, input_attr=1, target_attr=0,
        centre=centre, shape=1.0, width=width, drop_input_attr=False,
    )
    return int(np.count_nonzero(w_out[0][0]))


class TestRectWindowHalfOpen:
    """#20: half-open [c - W/2, c + W/2) rectangular support."""

    @pytest.mark.parametrize("width,expected",
                             [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)])
    def test_on_pulse_width_to_count(self, width, expected):
        # Centre on a pulse (IOI = 1). A closed interval would collapse
        # widths 1..5 to pulse counts 1, 3, 3, 5, 5.
        assert _pulse_count(4.0, float(width)) == expected

    def test_between_pulse_centre_not_empty(self):
        # Midway between pulses 3 and 4; width 1 keeps the lower-edge
        # pulse (count 1), not zero (the closed/FP degeneracy) or two.
        assert _pulse_count(3.5, 1.0) == 1

    def test_lower_edge_in_upper_edge_out(self):
        # width 2 at centre 4 -> [3, 5): pulses 3 and 4; pulse 5 excluded.
        pitch = np.arange(9, dtype=float).reshape(1, 9)
        time = np.arange(9, dtype=float).reshape(1, 9)
        _, w_out, _ = weight_events(
            [pitch, time], None, input_attr=1, target_attr=0,
            centre=4.0, shape=1.0, width=2.0, drop_input_attr=False,
        )
        factor = w_out[0][0]
        assert factor[3] > 0 and factor[4] > 0   # lower edge + interior
        assert factor[5] == 0.0                  # upper edge excluded

    def test_fractional_grid_mapping(self):
        # Non-integer IOI: pulses spaced 0.25, width 4*IOI = 1.0 -> 4 pulses.
        n = 9
        pitch = np.arange(n, dtype=float).reshape(1, n)
        time = (0.25 * np.arange(n, dtype=float)).reshape(1, n)
        _, w_out, _ = weight_events(
            [pitch, time], None, input_attr=1, target_attr=0,
            centre=time[0, 4], shape=1.0, width=1.0, drop_input_attr=False,
        )
        assert int(np.count_nonzero(w_out[0][0])) == 4

    def test_gaussian_shape_unaffected(self):
        pitch = np.arange(5, dtype=float).reshape(1, 5)
        time = np.arange(5, dtype=float).reshape(1, 5)
        _, w_out, _ = weight_events(
            [pitch, time], None, input_attr=1, target_attr=0,
            centre=2.0, shape=0.0, sd=1.0, drop_input_attr=False,
        )
        factor = w_out[0][0]
        assert factor[2] == pytest.approx(1.0)   # peak at centre
        assert np.all(factor > 0)                # no hard edge


class TestRenyi2ZeroMassNaN:
    """#21: zero-mass density -> NaN from renyi2, not a raise."""

    def test_single_multiset_zero_weight_returns_nan(self):
        v = mpt.entropy_exp_tens([0., 4., 7.], [0., 0., 0.], 1.0, 1,
                                 False, False, 0.0, method="renyi2",
                                 verbose=False)
        assert np.isnan(v)

    def test_ma_out_of_support_window_returns_nan(self):
        pitch = np.array([[60., 62., 64.]])
        time = np.array([[0., 1., 2.]])
        # Rectangular window centred far from every event -> zero mass.
        pa, wa, sp = weight_events(
            [pitch, time], None, input_attr=1, target_attr=0,
            centre=100.0, shape=1.0, width=1.0, drop_input_attr=False,
        )
        dens = mpt.build_exp_tens(
            pa, wa, specs=sp, sigma=[1.0, 1.0],
            is_per=[False, False], period=[0., 0.], verbose=False,
        )
        v = mpt.entropy_exp_tens(dens, method="renyi2", verbose=False)
        assert np.isnan(v)
