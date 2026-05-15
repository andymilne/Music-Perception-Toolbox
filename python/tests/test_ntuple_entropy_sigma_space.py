"""Tests for n_tuple_entropy with sigma_space.

Mirror of MATLAB tests/test_ntuple_entropy_sigma_space.m.
"""
import numpy as np
import pytest
import warnings

import mpt
from mpt._utils import position_variance


class TestNTupleEntropySigmaSpace:
    DIATONIC = [0, 2, 4, 5, 7, 9, 11]

    def test_sigma0_matches_default(self):
        H0, _ = mpt.n_tuple_entropy(self.DIATONIC, 12, 1, sigma=0)
        H1, _ = mpt.n_tuple_entropy(self.DIATONIC, 12, 1)
        assert H0 == pytest.approx(H1, abs=1e-12)

    def test_n1_position_equals_interval_with_sqrt2_scaling(self):
        """At n = 1 the two models agree exactly when the interval
        sigma matches the position sigma * sqrt(2). This is the
        'marginal-matched' relationship the n >= 2 approximation
        is named after."""
        sigma = 0.5
        H_pos, _ = mpt.n_tuple_entropy(
            self.DIATONIC, 12, 1, sigma=sigma, sigma_space="position"
        )
        H_int, _ = mpt.n_tuple_entropy(
            self.DIATONIC, 12, 1,
            sigma=sigma * np.sqrt(2), sigma_space="interval",
        )
        assert H_pos == pytest.approx(H_int, abs=1e-10)

    def test_smoothing_increases_entropy(self):
        H_raw, _ = mpt.n_tuple_entropy(self.DIATONIC, 12, 1)
        H_smooth, _ = mpt.n_tuple_entropy(self.DIATONIC, 12, 1, sigma=0.2)
        assert H_smooth > H_raw

    def test_position_smoother_than_interval_at_same_sigma(self):
        Hpos, _ = mpt.n_tuple_entropy(
            self.DIATONIC, 12, 1, sigma=0.3, sigma_space="position"
        )
        Hint, _ = mpt.n_tuple_entropy(
            self.DIATONIC, 12, 1, sigma=0.3, sigma_space="interval"
        )
        # Position's effective sigma is sigma*sqrt(2), so wider kernel
        assert Hpos > Hint

    def test_float_positions_accepted_when_sigma_positive(self):
        ji = [0, 203.91, 386.31, 498.04, 701.96, 884.36, 1088.27]
        H, _ = mpt.n_tuple_entropy(ji, 1200, 1, sigma=25)
        assert np.isfinite(H) and H > 0

    def test_float_positions_rejected_when_sigma_zero(self):
        with pytest.raises(ValueError, match="integer"):
            mpt.n_tuple_entropy([0.5, 2, 4, 5, 7, 9, 11], 12, 1)

    def test_n2_position_emits_approximation_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            H, _ = mpt.n_tuple_entropy(
                self.DIATONIC, 12, 2, sigma=0.3, sigma_space="position"
            )
        # The function should still return a finite value
        assert np.isfinite(H) and H > 0
        # And exactly one UserWarning of the expected kind should fire
        approx_warns = [
            ww for ww in w
            if issubclass(ww.category, UserWarning)
            and "marginal-matched" in str(ww.message)
        ]
        assert len(approx_warns) == 1

    def test_n2_interval_does_not_warn(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mpt.n_tuple_entropy(
                self.DIATONIC, 12, 2, sigma=0.3, sigma_space="interval"
            )
        approx_warns = [
            ww for ww in w
            if issubclass(ww.category, UserWarning)
            and "marginal-matched" in str(ww.message)
        ]
        assert approx_warns == []

    def test_invalid_sigma_space_errors(self):
        with pytest.raises(ValueError, match="sigma_space"):
            mpt.n_tuple_entropy(
                self.DIATONIC, 12, 1, sigma=0.3, sigma_space="bogus"
            )


# -------------------------------------------------------------------
#  dft_circular_simulate
# -------------------------------------------------------------------
