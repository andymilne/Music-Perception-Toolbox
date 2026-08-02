"""Tests for circular measures (r_phase, edges, proj_centroid, mean_offset, etc.).

Mirror of MATLAB tests/test_circular.m.
"""
import numpy as np
import pytest
import warnings

import mpt
from mpt._utils import position_variance


class TestCircular:
    def test_balance_augmented(self):
        b = mpt.balance([0, 400, 800], None, 1200)
        assert b == pytest.approx(1.0, abs=1e-10)

    def test_balance_cluster(self):
        b = mpt.balance([0, 100, 200], None, 1200)
        assert b < 0.5  # unbalanced

    def test_evenness_whole_tone(self):
        e = mpt.evenness([0, 200, 400, 600, 800, 1000], 1200)
        assert e == pytest.approx(1.0, abs=1e-10)

    def test_coherence_diatonic(self):
        c, nc = mpt.coherence([0, 2, 4, 5, 7, 9, 11], 12)
        assert nc == 1  # one failure (tritone)
        assert c > 0.99

    def test_coherence_whole_tone(self):
        c, nc = mpt.coherence([0, 2, 4, 6, 8, 10], 12)
        assert c == pytest.approx(1.0)
        assert nc == 0

    def test_sameness_diatonic(self):
        sq, nd = mpt.sameness([0, 2, 4, 5, 7, 9, 11], 12)
        assert nd == 1
        assert sq > 0.99

    def test_sameness_whole_tone(self):
        sq, nd = mpt.sameness([0, 2, 4, 6, 8, 10], 12)
        assert sq == pytest.approx(1.0)
        assert nd == 0

    def test_edges_output_shape(self):
        e, e_signed = mpt.edges([0, 2, 4, 5, 7, 9, 11], None, 12)
        assert e.shape == (12,)
        assert np.all(e >= 0)

    def test_edges_zero_at_events_of_even_scale(self):
        """Perfect rotational symmetry ⇒ no edges at the event positions
        of a perfectly even multiset (von Mises convolution of a symmetric
        density has no derivative at the symmetry points)."""
        e, _ = mpt.edges([0, 200, 400, 600, 800, 1000], None, 1200)
        e_at_events = e[[0, 200, 400, 600, 800, 1000]]
        np.testing.assert_allclose(e_at_events, 0, atol=1e-10)

    def test_edges_signed_antisymmetric_for_block(self):
        """For a contiguous block of events, the rising-edge boundary has
        positive sign and the falling-edge boundary has negative sign."""
        # Six events filling positions 0..5 of a 12-position circle.
        _, e_signed = mpt.edges([0, 1, 2, 3, 4, 5], None, 12)
        # Rising edge expected just before position 0 (i.e., at position 11).
        # Falling edge expected just after position 5 (i.e., at position 6).
        assert e_signed[11] > 0
        assert e_signed[6] < 0

    def test_proj_centroid_balanced(self):
        y, cm, cp = mpt.proj_centroid([0, 400, 800], None, 1200)
        assert cm == pytest.approx(0.0, abs=1e-10)
        np.testing.assert_allclose(y, 0, atol=1e-10)

    def test_mean_offset_shape(self):
        h = mpt.mean_offset([0, 2, 4, 5, 7, 9, 11], None, 12)
        assert h.shape == (12,)

    def test_mean_offset_zero_at_event_positions_of_even_scale(self):
        """For an evenly spaced multiset, mean_offset is zero at each
        event position by full rotational symmetry: at any event of
        the scale, the remaining events are arranged symmetrically
        around it."""
        h = mpt.mean_offset([0, 200, 400, 600, 800, 1000], None, 1200)
        np.testing.assert_allclose(
            h[[0, 200, 400, 600, 800, 1000]], 0, atol=1e-10,
        )

    def test_circ_apm_shape(self):
        R, rp, rl = mpt.circ_apm([0, 3, 6, 10, 12], None, 16)
        assert R.shape == (16, 16)
        assert rp.shape == (16,)
        assert rl.shape == (16,)

    def test_circ_apm_autocorrelation_symmetric(self):
        """Circular autocorrelation r_lag is symmetric about lag 0:
        r_lag[k] == r_lag[period - k] for k = 1, ..., period//2."""
        _, _, r_lag = mpt.circ_apm([0, 3, 6, 10, 12], None, 16)
        for k in range(1, 9):
            assert r_lag[k] == pytest.approx(r_lag[16 - k], abs=1e-10)

    def test_markov_s_shape(self):
        y = mpt.markov_s([0, 3, 6, 10, 12], None, 16)
        assert y.shape == (16,)
        # Event positions should have positive predictions
        assert y[0] > 0
        assert y[3] > 0

    def test_markov_s_periodic_pattern_equal_at_events(self):
        """For a perfectly 4-periodic pattern in period 16, the four
        event positions share the same S-step look-ahead context, so
        the predicted weights are equal there. Non-event positions
        also share a context (with each other) and have lower weight."""
        y = mpt.markov_s([0, 4, 8, 12], None, 16)
        np.testing.assert_allclose(y[[0, 4, 8, 12]], y[0], atol=1e-10)
        # Non-events 1, 2, 3 (and their period-4 copies) form a single
        # equivalence class as well; verify they're uniform and < event y.
        np.testing.assert_allclose(y[[1, 5, 9, 13]], y[1], atol=1e-10)
        assert y[0] > y[1]

    def test_dft_circular_unison_unit_F0(self):
        """All weight at one location ⇒ |F[0]| = 1."""
        _, mag = mpt.dft_circular([100, 100, 100], None, 1200)
        assert mag[0] == pytest.approx(1.0, abs=1e-10)

    def test_dft_circular_augmented_zero_F0(self):
        """Augmented triad (cube roots of unity, scaled to the period)
        sums to zero ⇒ |F[0]| = 0."""
        _, mag = mpt.dft_circular([0, 400, 800], None, 1200)
        assert mag[0] == pytest.approx(0.0, abs=1e-10)

    def test_markov_s_shape(self):
        y = mpt.markov_s([0, 3, 6, 10, 12], None, 16)
        assert y.shape == (16,)
        # Event positions should have positive predictions
        assert y[0] > 0
        assert y[3] > 0


# ===================================================================
#  Expectation tensors
# ===================================================================
