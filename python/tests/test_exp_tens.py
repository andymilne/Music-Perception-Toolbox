"""Tests for build_exp_tens / eval_exp_tens / cos_sim_exp_tens — SA core.

Mirror of MATLAB tests/test_exp_tens.m.
"""
import numpy as np
import pytest
import warnings

import mpt
from mpt._utils import position_variance


class TestTensor:
    def test_build_eval_roundtrip(self):
        dens = mpt.build_exp_tens([0, 4, 7], None, 0.5, 1, False, True, 12, verbose=False)
        vals = mpt.eval_exp_tens(dens, np.arange(12), verbose=False)
        # Peaks should be at pitch classes 0, 4, 7
        peaks = np.where(vals > 0.5)[0]
        np.testing.assert_array_equal(peaks, [0, 4, 7])

    def test_cos_sim_identical(self):
        s = mpt.cos_sim_exp_tens_raw(
            [0, 4, 7], None, [0, 4, 7], None,
            10, 1, False, True, 1200, verbose=False
        )
        assert s == pytest.approx(1.0, abs=1e-10)

    def test_cos_sim_range(self):
        s = mpt.cos_sim_exp_tens_raw(
            [0, 200, 400, 500, 700, 900, 1100], None,
            [0, 400, 700], None,
            10, 1, False, True, 1200, verbose=False
        )
        assert 0 < s < 1

    def test_relative_tensor(self):
        dens = mpt.build_exp_tens([0, 4, 7], None, 0.5, 2, True, True, 12, verbose=False)
        assert dens.dim == 1  # r=2, is_rel=True → dim=1

    def test_normalize_pdf(self):
        # Use non-periodic case: the pdf should integrate to ~1 over R
        dens = mpt.build_exp_tens([0, 400, 700], None, 20, 1, False, False, 1200, verbose=False)
        x = np.arange(-100, 900, 1.0)
        vals = mpt.eval_exp_tens(dens, x, "pdf", verbose=False)
        integral = np.sum(vals) * 1.0  # dx = 1
        assert integral == pytest.approx(1.0, rel=0.01)

    def test_batch_cos_sim(self):
        A = np.array([
            [0, 200, 400, 500, 700, 900, 1100],
            [0, 200, 400, 500, 700, 900, 1100],
        ])
        B = np.array([
            [0, 400, 700, np.nan, np.nan, np.nan, np.nan],
            [0, 300, 700, np.nan, np.nan, np.nan, np.nan],
        ])
        s = mpt.batch_cos_sim_exp_tens(
            A, B, 10, 1, False, True, 1200, verbose=False
        )
        assert s.shape == (2,)
        assert np.all(~np.isnan(s))
        assert s[0] > s[1]  # major triad fits diatonic better than minor

    # --- Transposition invariance tests (cosSimExpTens fix) ----------

    def test_isrel_transposition_invariance_nonperiodic(self):
        """isRel should give exact transposition invariance (non-periodic)."""
        B = [0, 200, 400, 500, 700, 900, 1100]
        s0 = mpt.cos_sim_exp_tens_raw(
            [0, 400, 700], None, B, None,
            10, 2, True, False, 1200, verbose=False
        )
        s1 = mpt.cos_sim_exp_tens_raw(
            [100, 500, 800], None, B, None,
            10, 2, True, False, 1200, verbose=False
        )
        assert s0 == pytest.approx(s1, abs=1e-14)

    def test_isrel_transposition_invariance_periodic(self):
        """isPer + isRel should give exact transposition invariance."""
        B = [0, 200, 400, 500, 700, 900, 1100]
        s0 = mpt.cos_sim_exp_tens_raw(
            [0, 400, 700], None, B, None,
            10, 2, True, True, 1200, verbose=False
        )
        s1 = mpt.cos_sim_exp_tens_raw(
            [100, 500, 800], None, B, None,
            10, 2, True, True, 1200, verbose=False
        )
        assert s0 == pytest.approx(s1, abs=1e-14)

    def test_isrel_transposition_invariance_periodic_r3(self):
        """isPer + isRel transposition invariance should hold for r=3."""
        B = [0, 200, 400, 500, 700, 900, 1100]
        s0 = mpt.cos_sim_exp_tens_raw(
            [0, 400, 700], None, B, None,
            10, 3, True, True, 1200, verbose=False
        )
        s1 = mpt.cos_sim_exp_tens_raw(
            [500, 900, 1200], None, B, None,
            10, 3, True, True, 1200, verbose=False
        )
        assert s0 == pytest.approx(s1, abs=1e-14)

    @pytest.mark.parametrize("shift", [100, 300, 500, 700, 1100])
    def test_isrel_transposition_all_shifts(self, shift):
        """isPer + isRel should be invariant across many transpositions."""
        A = np.array([0.0, 400, 700])
        B = np.array([0, 200, 400, 500, 700, 900, 1100], dtype=float)
        s_ref = mpt.cos_sim_exp_tens_raw(
            A, None, B, None,
            10, 2, True, True, 1200, verbose=False
        )
        s_shift = mpt.cos_sim_exp_tens_raw(
            A + shift, None, B, None,
            10, 2, True, True, 1200, verbose=False
        )
        assert s_ref == pytest.approx(s_shift, abs=1e-14)

    def test_isper_octave_equivalence(self):
        """isPer should treat octave-displaced pitches as equivalent."""
        B = [0, 200, 400, 500, 700, 900, 1100]
        s0 = mpt.cos_sim_exp_tens_raw(
            [0, 400, 700], None, B, None,
            10, 1, False, True, 1200, verbose=False
        )
        s1 = mpt.cos_sim_exp_tens_raw(
            [1200, 1600, 1900], None, B, None,
            10, 1, False, True, 1200, verbose=False
        )
        assert s0 == pytest.approx(s1, abs=1e-14)

    def test_isrel_with_weights_periodic(self):
        """isPer + isRel transposition invariance should hold with weights."""
        B = [0, 200, 400, 500, 700, 900, 1100]
        w = [1.0, 0.8, 0.6]
        s0 = mpt.cos_sim_exp_tens_raw(
            [0, 400, 700], w, B, None,
            10, 2, True, True, 1200, verbose=False
        )
        s1 = mpt.cos_sim_exp_tens_raw(
            [100, 500, 800], w, B, None,
            10, 2, True, True, 1200, verbose=False
        )
        assert s0 == pytest.approx(s1, abs=1e-14)

    def test_batch_deduplication_octave(self):
        """Batch should deduplicate octave-displaced sets under isPer."""
        A = np.array([
            [0, 400, 700],
            [1200, 1600, 1900],  # octave displaced
            [0, 400, 700],       # exact duplicate
        ])
        B = np.tile([0, 200, 400, 500, 700, 900, 1100], (3, 1))
        s = mpt.batch_cos_sim_exp_tens(
            A, B, 10, 1, False, True, 1200, verbose=False
        )
        # All three rows should produce the same value
        np.testing.assert_allclose(s[0], s[1], atol=1e-14)
        np.testing.assert_allclose(s[0], s[2], atol=1e-14)


# ===================================================================
#  Entropy
# ===================================================================
