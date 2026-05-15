"""Tests for sameness with sigma_space.

Mirror of MATLAB tests/test_sameness_sigma_space.m.
"""
import numpy as np
import pytest
import warnings

import mpt
from mpt._utils import position_variance


class TestSamenessSigmaSpace:
    DIATONIC = [0, 2, 4, 5, 7, 9, 11]

    def test_sigma0_matches_default(self):
        sq0, nd0 = mpt.sameness(self.DIATONIC, 12, 0)
        sq1, nd1 = mpt.sameness(self.DIATONIC, 12)
        assert sq0 == sq1
        assert nd0 == nd1

    def test_sigma0_flags_coincide(self):
        sqP, _ = mpt.sameness(self.DIATONIC, 12, 0, sigma_space="position")
        sqI, _ = mpt.sameness(self.DIATONIC, 12, 0, sigma_space="interval")
        assert sqP == pytest.approx(sqI, abs=1e-12)

    def test_diatonic_sigma05_position_regression(self):
        sq, _ = mpt.sameness(self.DIATONIC, 12, 0.5, sigma_space="position")
        assert sq == pytest.approx(0.422420, abs=1e-4)

    def test_diatonic_sigma05_interval_regression(self):
        sq, _ = mpt.sameness(self.DIATONIC, 12, 0.5, sigma_space="interval")
        assert sq == pytest.approx(0.714369, abs=1e-4)

    def test_position_more_aggressive_than_interval(self):
        # Position model has wider effective kernel for typical
        # disjoint-endpoint pairs (V = 4 sigma^2 vs interval's 2 sigma^2)
        sqP, _ = mpt.sameness(self.DIATONIC, 12, 0.5, sigma_space="position")
        sqI, _ = mpt.sameness(self.DIATONIC, 12, 0.5, sigma_space="interval")
        assert sqP < sqI

    def test_float_positions_accepted_when_sigma_positive(self):
        ji = [0, 203.91, 386.31, 498.04, 701.96, 884.36, 1088.27]
        sq, _ = mpt.sameness(ji, 1200, 25)
        assert np.isfinite(sq)
        assert 0 < sq <= 1.0

    def test_float_positions_rejected_when_sigma_zero(self):
        with pytest.raises(ValueError, match="integer"):
            mpt.sameness([0.5, 2, 4, 7], 12, 0)

    def test_invalid_sigma_space_errors(self):
        with pytest.raises(ValueError, match="sigma_space"):
            mpt.sameness(self.DIATONIC, 12, 0.5, sigma_space="bogus")


# -------------------------------------------------------------------
#  coherence with sigma_space
# -------------------------------------------------------------------
