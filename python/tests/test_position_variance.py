"""Tests for position_variance helper.

Mirror of MATLAB tests/test_position_variance.m.
"""
import numpy as np
import pytest
import warnings

import mpt
from mpt._utils import position_variance


class TestPositionVariance:
    def test_disjoint_endpoints(self):
        V = position_variance([1, 2, 3, 4], [+1, -1, -1, +1], 1.0)
        assert V == pytest.approx(4.0, abs=1e-12)

    def test_shared_cancelling(self):
        V = position_variance([1, 2, 1, 3], [+1, -1, -1, +1], 1.0)
        assert V == pytest.approx(2.0, abs=1e-12)

    def test_shared_reinforcing(self):
        # Tritone-style configuration: both endpoints shared,
        # signs reinforce -> 8 sigma^2
        V = position_variance([2, 1, 1, 2], [+1, -1, -1, +1], 1.0)
        assert V == pytest.approx(8.0, abs=1e-12)

    def test_scales_with_sigma_squared(self):
        V = position_variance([1, 2], [+1, -1], 0.5)
        assert V == pytest.approx(0.5, abs=1e-12)


# -------------------------------------------------------------------
#  sameness with sigma_space
# -------------------------------------------------------------------
