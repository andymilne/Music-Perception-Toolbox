"""Tests for add_spectra spectrum-generation tests.

Mirror of MATLAB tests/test_add_spectra.m.
"""
import numpy as np
import pytest
import warnings

import mpt
from mpt._utils import position_variance


class TestAddSpectra:
    def test_harmonic_count(self):
        p, w = mpt.add_spectra(np.array([0, 400, 700]), None, "harmonic", 8, "powerlaw", 1)
        assert len(p) == 24  # 3 pitches x 8 harmonics

    def test_harmonic_fundamental(self):
        p, w = mpt.add_spectra(np.array([0.0]), np.array([1.0]), "harmonic", 4, "powerlaw", 0)
        # With rho=0 (flat), all weights = 1
        np.testing.assert_allclose(w, 1.0)
        # Offsets: 1200*log2(1), 1200*log2(2), 1200*log2(3), 1200*log2(4)
        expected = 1200 * np.log2([1, 2, 3, 4])
        np.testing.assert_allclose(p, expected, atol=1e-10)

    def test_stretched(self):
        p, w = mpt.add_spectra(np.array([0.0]), np.array([1.0]), "stretched", 3, 1.02, "powerlaw", 1)
        assert len(p) == 3
        # beta=1.02 should give slightly wider spacing than harmonic
        p_harm, _ = mpt.add_spectra(np.array([0.0]), np.array([1.0]), "harmonic", 3, "powerlaw", 1)
        assert p[2] > p_harm[2]

    def test_stiff(self):
        p, _ = mpt.add_spectra(np.array([0.0]), np.array([1.0]), "stiff", 4, 0.0003, "powerlaw", 1)
        # With B > 0, higher partials should be sharper than harmonic
        p_harm, _ = mpt.add_spectra(np.array([0.0]), np.array([1.0]), "harmonic", 4, "powerlaw", 1)
        assert p[3] > p_harm[3]

    def test_custom(self):
        p, w = mpt.add_spectra(np.array([0, 700]), None, "custom", [0, 1200], [1, 0.5])
        np.testing.assert_allclose(p, [0, 1200, 700, 1900])
        np.testing.assert_allclose(w, [1, 0.5, 1, 0.5])

    def test_geometric_weights(self):
        _, w = mpt.add_spectra(np.array([0.0]), np.array([1.0]), "harmonic", 4, "geometric", 0.5)
        np.testing.assert_allclose(w, [1, 0.5, 0.25, 0.125])

    def test_freqlinear_alpha_zero_equals_harmonic(self):
        """At alpha=0, ratio(n) = n/(0+1) = n, identical to the harmonic series."""
        p_lin, w_lin = mpt.add_spectra(
            np.array([0.0]), np.array([1.0]),
            "freqlinear", 4, 0.0, "powerlaw", 0,
        )
        p_har, w_har = mpt.add_spectra(
            np.array([0.0]), np.array([1.0]),
            "harmonic", 4, "powerlaw", 0,
        )
        np.testing.assert_allclose(p_lin, p_har, atol=1e-10)
        np.testing.assert_allclose(w_lin, w_har, atol=1e-10)

    def test_freqlinear_alpha_one_partial_ratios(self):
        """At alpha=1, ratio(n) = (1+n)/2, giving partials at 1, 1.5, 2, 2.5, ...
        which is 1200 * log2 of those ratios in cents."""
        p, _ = mpt.add_spectra(
            np.array([0.0]), np.array([1.0]),
            "freqlinear", 4, 1.0, "powerlaw", 0,
        )
        expected = 1200 * np.log2(np.array([1.0, 1.5, 2.0, 2.5]))
        np.testing.assert_allclose(p, expected, atol=1e-10)

    def test_freqlinear_alpha_le_minus_one_errors(self):
        """alpha <= -1 makes ratio(n) non-positive for some n; must raise."""
        with pytest.raises(ValueError, match="alpha"):
            mpt.add_spectra(
                np.array([0.0]), np.array([1.0]),
                "freqlinear", 4, -1.0, "powerlaw", 0,
            )


# ===================================================================
#  Circular measures
# ===================================================================
