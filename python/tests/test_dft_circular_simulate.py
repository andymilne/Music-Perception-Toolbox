"""Tests for dft_circular_simulate.

Mirror of MATLAB tests/test_dft_circular_simulate.m.
"""
import numpy as np
import pytest
import warnings

import mpt
from mpt._utils import position_variance


class TestDftCircularSimulate:
    def test_small_sigma_recovers_deterministic_mean(self):
        """At sigma very close to 0, MC mean magnitudes should match
        the deterministic dftCircular magnitudes within tight MC error."""
        p = [0, 200, 400, 500, 700, 900, 1100]
        period = 1200
        _, mag_det = mpt.dft_circular(p, None, period)
        m, s = mpt.dft_circular_simulate(
            p, None, period, sigma=1e-3, n_draws=2000, rng_seed=42
        )
        np.testing.assert_allclose(m, mag_det, atol=1e-4)
        np.testing.assert_allclose(s, 0, atol=1e-4)

    def test_sigma_zero_exactly(self):
        """sigma == 0 should give zero variance and exact deterministic mean."""
        p = [0, 200, 400, 500, 700, 900, 1100]
        period = 1200
        _, mag_det = mpt.dft_circular(p, None, period)
        m, s = mpt.dft_circular_simulate(
            p, None, period, sigma=0.0, n_draws=100, rng_seed=42
        )
        np.testing.assert_allclose(m, mag_det, atol=1e-12)
        np.testing.assert_allclose(s, 0, atol=1e-12)

    def test_closed_form_F0_squared_mean(self):
        """E[|F(0)|^2] = alpha_1^2 * |F_det(0)|^2 + (1 - alpha_1^2) * sum(w^2) / sum(w)^2.
        Augmented triad has F_det(0) = 0, simplifying to the second term."""
        p = [0, 400, 800]
        period = 1200
        sigma = 50
        K = len(p)
        alpha1 = np.exp(-2 * np.pi**2 * sigma**2 / period**2)
        expected_F0_sq = (1 - alpha1**2) * 1.0 / K   # sum(w^2)/sum(w)^2 = K/K^2 = 1/K
        m, s, samples = mpt.dft_circular_simulate(
            p, None, period, sigma=sigma,
            n_draws=50000, rng_seed=42, return_samples=True
        )
        mc_F0_sq = np.mean(samples[:, 0]**2)
        assert mc_F0_sq == pytest.approx(expected_F0_sq, abs=2e-3)

    def test_rng_seed_reproducibility(self):
        p = [0, 200, 400, 500, 700, 900, 1100]
        m1, s1 = mpt.dft_circular_simulate(
            p, None, 1200, sigma=50, n_draws=1000, rng_seed=42
        )
        m2, s2 = mpt.dft_circular_simulate(
            p, None, 1200, sigma=50, n_draws=1000, rng_seed=42
        )
        np.testing.assert_array_equal(m1, m2)
        np.testing.assert_array_equal(s1, s2)

    def test_return_samples_shape(self):
        p = [0, 200, 400, 500, 700, 900, 1100]
        m, s, samples = mpt.dft_circular_simulate(
            p, None, 1200, sigma=50,
            n_draws=500, rng_seed=42, return_samples=True
        )
        assert samples.shape == (500, len(p))

    def test_F0_permutation_invariance(self):
        """F(0) is the sum z_k = exp(2*pi*i*p_k/T), permutation-invariant.
        This means the resort step does not affect F(0). At sigma > 0 we should
        therefore find E[|F(0)|^2] matches the closed form exactly (up to MC
        tolerance), regardless of how aggressively events swap."""
        p = [0, 100, 110, 200]   # close events => high swap probability
        period = 1200
        sigma = 50
        K = 4
        alpha1 = np.exp(-2 * np.pi**2 * sigma**2 / period**2)
        # F_det(0) = (1/K) * sum(exp(2*pi*i*p/T))
        F_det = np.fft.fft(np.exp(2j * np.pi * np.array(p) / period)) / K
        expected_F0_sq = alpha1**2 * abs(F_det[0])**2 + (1 - alpha1**2) / K
        _, _, samples = mpt.dft_circular_simulate(
            p, None, period, sigma=sigma,
            n_draws=50000, rng_seed=42, return_samples=True
        )
        mc_F0_sq = np.mean(samples[:, 0]**2)
        assert mc_F0_sq == pytest.approx(expected_F0_sq, abs=2e-3)


# -------------------------------------------------------------------
#  balance with sigma
# -------------------------------------------------------------------
