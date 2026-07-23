"""Tests for entropy_exp_tens — single-multiset Shannon and basic.

Mirror of MATLAB tests/test_entropy.m.
"""
import numpy as np
import pytest
import warnings

import mpt
from mpt._utils import position_variance


class TestEntropy:
    def test_n_tuple_entropy_whole_tone(self):
        # Whole-tone scale: single step size → entropy = 0
        H, _ = mpt.n_tuple_entropy([0, 2, 4, 6, 8, 10], 12)
        assert H == pytest.approx(0.0, abs=1e-10)

    def test_n_tuple_entropy_2tuple(self):
        # Diatonic 2-tuple entropy ~1.56 bits (Milne & Dean 2016)
        H, _ = mpt.n_tuple_entropy(
            [0, 2, 4, 5, 7, 9, 11], 12, 2, method='shannon'
        )
        assert H == pytest.approx(1.56, abs=0.01)

    def test_n_tuple_smoothed(self):
        H_raw, _ = mpt.n_tuple_entropy([0, 2, 4, 5, 7, 9, 11], 12, 1)
        H_smooth, _ = mpt.n_tuple_entropy(
            [0, 2, 4, 5, 7, 9, 11], 12, 1, sigma=0.2
        )
        # Smoothing should increase entropy (spread mass)
        assert H_smooth > H_raw

    def test_entropy_exp_tens_uniform(self):
        # Chromatic scale with wide sigma → nearly uniform → H ≈ 1
        H = mpt.entropy_exp_tens(
            np.arange(12), np.ones(12), 100, 1, False, True, 12,
            n_points_per_dim=1200,
        )
        assert H > 0.95


# -------------------------------------------------------------------
#  positionVariance helper
# -------------------------------------------------------------------
