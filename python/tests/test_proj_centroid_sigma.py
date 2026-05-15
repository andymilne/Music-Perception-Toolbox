"""Tests for proj_centroid with sigma.

Mirror of MATLAB tests/test_proj_centroid_sigma.m.
"""
import numpy as np
import pytest
import warnings

import mpt
from mpt._utils import position_variance


class TestProjCentroidSigma:
    def test_alpha_1_damping(self):
        """y_smoothed(x) / y_deterministic(x) should equal alpha_1
        for every query point x — closed-form linear damping."""
        p = [0, 4, 7]
        period = 12
        sigma = 0.5
        alpha1 = np.exp(-2 * np.pi**2 * sigma**2 / period**2)
        x = np.arange(period)
        y_det, _, _ = mpt.proj_centroid(p, None, period, x)
        y_smooth, _, _ = mpt.proj_centroid(p, None, period, x, sigma=sigma)
        # Avoid division by tiny y_det values; multiplication form
        np.testing.assert_allclose(y_smooth, alpha1 * y_det, atol=1e-12)

    def test_phase_unchanged(self):
        """Centroid phase is preserved in expectation (arg of E[F(0)]
        equals arg of F(0)_det) — so cent_phase doesn't move with sigma."""
        p = [0, 4, 7]
        period = 12
        _, _, phase_det = mpt.proj_centroid(p, None, period)
        _, _, phase_smooth = mpt.proj_centroid(p, None, period, sigma=1.0)
        assert phase_smooth == pytest.approx(phase_det, abs=1e-12)

    def test_cent_mag_damped_by_alpha_1(self):
        """cent_mag in proj_centroid returns alpha_1 * |F_det(0)|, the
        magnitude of E[F(0)], NOT E[|F(0)|]. (For E[|F(0)|], use balance.)"""
        p = [0, 200, 400, 500, 700, 900, 1100]
        period = 1200
        sigma = 100
        alpha1 = np.exp(-2 * np.pi**2 * sigma**2 / period**2)
        _, cm_det, _ = mpt.proj_centroid(p, None, period)
        _, cm_smooth, _ = mpt.proj_centroid(p, None, period, sigma=sigma)
        assert cm_smooth == pytest.approx(alpha1 * cm_det, abs=1e-12)

    def test_sigma_zero_recovers_v2(self):
        p = [0, 4, 7]
        y0, cm0, cp0 = mpt.proj_centroid(p, None, 12, sigma=0)
        y1, cm1, cp1 = mpt.proj_centroid(p, None, 12)
        np.testing.assert_array_equal(y0, y1)
        assert cm0 == cm1
        assert cp0 == cp1


# ===================================================================
#  Harmony
# ===================================================================
