"""Tests for coherence with sigma_space.

Mirror of MATLAB tests/test_coherence_sigma_space.m.
"""
import numpy as np
import pytest
import warnings

import mpt
from mpt._utils import position_variance


class TestCoherenceSigmaSpace:
    DIATONIC = [0, 2, 4, 5, 7, 9, 11]

    def test_sigma0_matches_default(self):
        c0, nc0 = mpt.coherence(self.DIATONIC, 12, 0)
        c1, nc1 = mpt.coherence(self.DIATONIC, 12)
        assert c0 == c1
        assert nc0 == nc1

    def test_sigma0_strict_false(self):
        c, nc = mpt.coherence(self.DIATONIC, 12, 0, strict=False)
        # Diatonic with non-strict propriety: tritone tie does not count
        assert nc == 0
        assert c == pytest.approx(1.0, abs=1e-12)

    def test_diatonic_sigma05_position_regression(self):
        c, _ = mpt.coherence(self.DIATONIC, 12, 0.5, sigma_space="position")
        assert c == pytest.approx(0.873485, abs=1e-4)

    def test_diatonic_sigma05_interval_regression(self):
        c, _ = mpt.coherence(self.DIATONIC, 12, 0.5, sigma_space="interval")
        assert c == pytest.approx(0.944610, abs=1e-4)

    def test_tritone_tie_at_any_sigma(self):
        """The diatonic tritone (F-B as fourth, B-F as fifth) shares
        endpoints with reinforcing signs. Var(D2 - D1) = 8 sigma^2 and
        the means coincide exactly. Soft contribution is 0.5 at every
        sigma > 0, so the sigma -> 0+ limit of nc is 0.5 (one tritone,
        half a failure), giving c -> 1 - 0.5/140."""
        c, nc = mpt.coherence(self.DIATONIC, 12, 1e-6)
        assert c == pytest.approx(1 - 0.5 / 140, abs=1e-3)
        assert nc == pytest.approx(0.5, abs=1e-3)

    def test_invalid_sigma_space_errors(self):
        with pytest.raises(ValueError, match="sigma_space"):
            mpt.coherence(self.DIATONIC, 12, 0.5, sigma_space="bogus")


# -------------------------------------------------------------------
#  n_tuple_entropy with sigma_space
# -------------------------------------------------------------------
