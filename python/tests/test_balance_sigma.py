"""Tests for balance_circular with sigma.

Mirror of MATLAB tests/test_balance_sigma.m.
"""
import numpy as np
import pytest
import warnings

import mpt
from mpt._utils import position_variance


class TestBalanceSigma:
    def test_sigma_zero_returns_scalar_backward_compat(self):
        b = mpt.balance([0, 400, 800], None, 1200)
        assert isinstance(b, float)
        assert b == pytest.approx(1.0, abs=1e-10)

    def test_sigma_positive_default_returns_scalar(self):
        b = mpt.balance([0, 400, 800], None, 1200, sigma=25, rng_seed=42)
        assert isinstance(b, float)
        assert 0 <= b <= 1

    def test_return_std_yields_tuple(self):
        result = mpt.balance(
            [0, 400, 800], None, 1200,
            sigma=25, return_std=True, rng_seed=42
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        b, b_std = result
        assert isinstance(b, float)
        assert isinstance(b_std, float)
        assert b_std > 0

    def test_return_std_at_sigma_zero(self):
        b, b_std = mpt.balance(
            [0, 400, 800], None, 1200, sigma=0, return_std=True
        )
        assert b == pytest.approx(1.0)
        assert b_std == 0.0

    def test_augmented_triad_rayleigh_bias(self):
        """Augmented triad is perfectly balanced (F_det(0) = 0). Under
        jitter, |F(0)| is Rayleigh-distributed with positive mean. So
        b = 1 - E[|F(0)|] < 1 even though F_det(0) = 0."""
        sigma = 50
        period = 1200
        K = 3
        alpha1 = np.exp(-2 * np.pi**2 * sigma**2 / period**2)
        # Rayleigh mean = sigma_R * sqrt(pi/2), with sigma_R^2 = (1-alpha1^2)/(2*K)
        expected_mag_mean = np.sqrt((1 - alpha1**2) * np.pi / (4 * K))
        b, b_std = mpt.balance(
            [0, 400, 800], None, 1200,
            sigma=sigma, return_std=True,
            n_draws=50000, rng_seed=42,
        )
        mc_mag_mean = 1 - b
        assert mc_mag_mean == pytest.approx(expected_mag_mean, abs=5e-3)
        assert b < 1   # Rayleigh bias is strictly positive
        assert b_std > 0


# -------------------------------------------------------------------
#  evenness with sigma
# -------------------------------------------------------------------
