"""Tests for cross-cutting input-validation paths.

Mirror of MATLAB tests/test_input_validation.m.
"""
import numpy as np
import pytest
import warnings

import mpt
from mpt._utils import position_variance


class TestValidation:
    def test_weights_broadcast_scalar(self):
        p, w = mpt.add_spectra(np.array([0, 400, 700]), np.array([0.5]),
                               "harmonic", 2, "powerlaw", 0)
        assert np.all(w == 0.5)

    def test_weights_none_gives_uniform(self):
        p, w = mpt.add_spectra(np.array([0, 400, 700]), None,
                               "harmonic", 1, "powerlaw", 0)
        np.testing.assert_allclose(w, 1.0)

    def test_r_too_large(self):
        with pytest.raises(ValueError, match="must not exceed"):
            mpt.build_exp_tens([0, 4], None, 10, 3, False, True, 12, verbose=False)

    def test_is_rel_r_1(self):
        with pytest.raises(ValueError, match="at least 2"):
            mpt.build_exp_tens([0, 4, 7], None, 10, 1, True, True, 12, verbose=False)

    def test_coherence_duplicates(self):
        with pytest.raises(ValueError, match="duplicate"):
            mpt.coherence([0, 0, 4, 7], 12)

    def test_n_tuple_entropy_n_too_large(self):
        with pytest.raises(ValueError, match="must not exceed"):
            mpt.n_tuple_entropy([0, 2, 4], 12, 3)


# ===================================================================
#  continuity
# ===================================================================
