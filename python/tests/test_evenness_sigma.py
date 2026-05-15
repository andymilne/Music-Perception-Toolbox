"""Tests for evenness_circular with sigma.

Mirror of MATLAB tests/test_evenness_sigma.m.
"""
import numpy as np
import pytest
import warnings

import mpt
from mpt._utils import position_variance


class TestEvennessSigma:
    def test_sigma_zero_returns_scalar_backward_compat(self):
        e = mpt.evenness([0, 200, 400, 600, 800, 1000], 1200)
        assert isinstance(e, float)
        assert e == pytest.approx(1.0, abs=1e-10)

    def test_sigma_positive_default_returns_scalar(self):
        e = mpt.evenness([0, 200, 400, 600, 800, 1000], 1200, sigma=25, rng_seed=42)
        assert isinstance(e, float)
        assert 0 <= e <= 1

    def test_return_std_yields_tuple(self):
        e, e_std = mpt.evenness(
            [0, 200, 400, 600, 800, 1000], 1200,
            sigma=25, return_std=True, rng_seed=42,
        )
        assert isinstance(e, float)
        assert e_std > 0

    def test_smoothing_reduces_evenness_for_irregular_pattern(self):
        """For a pattern that isn't maximally even, jitter on average
        reduces |F(1)| (the evenness coefficient) because the deterministic
        signal gets damped while incoherent noise contributes equally."""
        diatonic = [0, 200, 400, 500, 700, 900, 1100]
        e_det = mpt.evenness(diatonic, 1200)
        e_smooth = mpt.evenness(diatonic, 1200, sigma=100,
                                n_draws=20000, rng_seed=42)
        assert e_smooth < e_det


# -------------------------------------------------------------------
#  proj_centroid with sigma
# -------------------------------------------------------------------
