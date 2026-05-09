"""Test suite for the Music Perception Toolbox (mpt).

Run with: pytest tests/test_mpt.py -v
"""

import numpy as np
import pytest
import warnings

import mpt
from mpt._utils import position_variance


# ===================================================================
#  convert_pitch
# ===================================================================


class TestConvertPitch:
    def test_hz_to_midi(self):
        assert mpt.convert_pitch(440, "hz", "midi") == pytest.approx(69)

    def test_midi_to_hz(self):
        assert mpt.convert_pitch(60, "midi", "hz") == pytest.approx(261.6256, rel=1e-4)

    def test_hz_to_cents(self):
        assert mpt.convert_pitch(440, "hz", "cents") == pytest.approx(6900)

    def test_identity(self):
        arr = np.array([100, 200, 300])
        np.testing.assert_array_equal(mpt.convert_pitch(arr, "hz", "hz"), arr)

    @pytest.mark.parametrize("scale", ["midi", "cents", "mel", "bark", "erb", "greenwood"])
    def test_roundtrip(self, scale):
        val = 440.0
        rt = mpt.convert_pitch(mpt.convert_pitch(val, "hz", scale), scale, "hz")
        assert rt == pytest.approx(val, rel=1e-8)

    def test_vectorised(self):
        out = mpt.convert_pitch([261.63, 440, 880], "hz", "midi")
        np.testing.assert_allclose(out, [60, 69, 81], atol=0.01)

    def test_unknown_scale(self):
        with pytest.raises(ValueError, match="Unknown"):
            mpt.convert_pitch(440, "hz", "bogus")


# ===================================================================
#  add_spectra
# ===================================================================


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
        # Six events filling positions 0..5 of a 12-slot circle.
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


class TestEntropy:
    def test_n_tuple_entropy_whole_tone(self):
        # Whole-tone scale: single step size → entropy = 0
        H, _ = mpt.n_tuple_entropy([0, 2, 4, 6, 8, 10], 12)
        assert H == pytest.approx(0.0, abs=1e-10)

    def test_n_tuple_entropy_2tuple(self):
        # Diatonic 2-tuple entropy ~1.56 bits (Milne & Dean 2016)
        H, _ = mpt.n_tuple_entropy(
            [0, 2, 4, 5, 7, 9, 11], 12, 2, normalize=False
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
            np.arange(12), np.ones(12), 100, 1, False, True, 12
        )
        assert H > 0.95


# -------------------------------------------------------------------
#  positionVariance helper
# -------------------------------------------------------------------


class TestPositionVariance:
    def test_disjoint_endpoints(self):
        V = position_variance([1, 2, 3, 4], [+1, -1, -1, +1], 1.0)
        assert V == pytest.approx(4.0, abs=1e-12)

    def test_shared_cancelling(self):
        V = position_variance([1, 2, 1, 3], [+1, -1, -1, +1], 1.0)
        assert V == pytest.approx(2.0, abs=1e-12)

    def test_shared_reinforcing(self):
        # Tritone-style configuration: both endpoints shared,
        # signs reinforce -> 8 sigma^2
        V = position_variance([2, 1, 1, 2], [+1, -1, -1, +1], 1.0)
        assert V == pytest.approx(8.0, abs=1e-12)

    def test_scales_with_sigma_squared(self):
        V = position_variance([1, 2], [+1, -1], 0.5)
        assert V == pytest.approx(0.5, abs=1e-12)


# -------------------------------------------------------------------
#  sameness with sigma_space
# -------------------------------------------------------------------


class TestSamenessSigmaSpace:
    DIATONIC = [0, 2, 4, 5, 7, 9, 11]

    def test_sigma0_matches_default(self):
        sq0, nd0 = mpt.sameness(self.DIATONIC, 12, 0)
        sq1, nd1 = mpt.sameness(self.DIATONIC, 12)
        assert sq0 == sq1
        assert nd0 == nd1

    def test_sigma0_flags_coincide(self):
        sqP, _ = mpt.sameness(self.DIATONIC, 12, 0, sigma_space="position")
        sqI, _ = mpt.sameness(self.DIATONIC, 12, 0, sigma_space="interval")
        assert sqP == pytest.approx(sqI, abs=1e-12)

    def test_diatonic_sigma05_position_regression(self):
        sq, _ = mpt.sameness(self.DIATONIC, 12, 0.5, sigma_space="position")
        assert sq == pytest.approx(0.422420, abs=1e-4)

    def test_diatonic_sigma05_interval_regression(self):
        sq, _ = mpt.sameness(self.DIATONIC, 12, 0.5, sigma_space="interval")
        assert sq == pytest.approx(0.714369, abs=1e-4)

    def test_position_more_aggressive_than_interval(self):
        # Position model has wider effective kernel for typical
        # disjoint-endpoint pairs (V = 4 sigma^2 vs interval's 2 sigma^2)
        sqP, _ = mpt.sameness(self.DIATONIC, 12, 0.5, sigma_space="position")
        sqI, _ = mpt.sameness(self.DIATONIC, 12, 0.5, sigma_space="interval")
        assert sqP < sqI

    def test_float_positions_accepted_when_sigma_positive(self):
        ji = [0, 203.91, 386.31, 498.04, 701.96, 884.36, 1088.27]
        sq, _ = mpt.sameness(ji, 1200, 25)
        assert np.isfinite(sq)
        assert 0 < sq <= 1.0

    def test_float_positions_rejected_when_sigma_zero(self):
        with pytest.raises(ValueError, match="integer"):
            mpt.sameness([0.5, 2, 4, 7], 12, 0)

    def test_invalid_sigma_space_errors(self):
        with pytest.raises(ValueError, match="sigma_space"):
            mpt.sameness(self.DIATONIC, 12, 0.5, sigma_space="bogus")


# -------------------------------------------------------------------
#  coherence with sigma_space
# -------------------------------------------------------------------


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


class TestNTupleEntropySigmaSpace:
    DIATONIC = [0, 2, 4, 5, 7, 9, 11]

    def test_sigma0_matches_default(self):
        H0, _ = mpt.n_tuple_entropy(self.DIATONIC, 12, 1, sigma=0)
        H1, _ = mpt.n_tuple_entropy(self.DIATONIC, 12, 1)
        assert H0 == pytest.approx(H1, abs=1e-12)

    def test_n1_position_equals_interval_with_sqrt2_scaling(self):
        """At n = 1 the two models agree exactly when the interval
        sigma matches the position sigma * sqrt(2). This is the
        'marginal-matched' relationship the n >= 2 approximation
        is named after."""
        sigma = 0.5
        H_pos, _ = mpt.n_tuple_entropy(
            self.DIATONIC, 12, 1, sigma=sigma, sigma_space="position"
        )
        H_int, _ = mpt.n_tuple_entropy(
            self.DIATONIC, 12, 1,
            sigma=sigma * np.sqrt(2), sigma_space="interval",
        )
        assert H_pos == pytest.approx(H_int, abs=1e-10)

    def test_smoothing_increases_entropy(self):
        H_raw, _ = mpt.n_tuple_entropy(self.DIATONIC, 12, 1)
        H_smooth, _ = mpt.n_tuple_entropy(self.DIATONIC, 12, 1, sigma=0.2)
        assert H_smooth > H_raw

    def test_position_smoother_than_interval_at_same_sigma(self):
        Hpos, _ = mpt.n_tuple_entropy(
            self.DIATONIC, 12, 1, sigma=0.3, sigma_space="position"
        )
        Hint, _ = mpt.n_tuple_entropy(
            self.DIATONIC, 12, 1, sigma=0.3, sigma_space="interval"
        )
        # Position's effective sigma is sigma*sqrt(2), so wider kernel
        assert Hpos > Hint

    def test_float_positions_accepted_when_sigma_positive(self):
        ji = [0, 203.91, 386.31, 498.04, 701.96, 884.36, 1088.27]
        H, _ = mpt.n_tuple_entropy(ji, 1200, 1, sigma=25)
        assert np.isfinite(H) and H > 0

    def test_float_positions_rejected_when_sigma_zero(self):
        with pytest.raises(ValueError, match="integer"):
            mpt.n_tuple_entropy([0.5, 2, 4, 5, 7, 9, 11], 12, 1)

    def test_n2_position_emits_approximation_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            H, _ = mpt.n_tuple_entropy(
                self.DIATONIC, 12, 2, sigma=0.3, sigma_space="position"
            )
        # The function should still return a finite value
        assert np.isfinite(H) and H > 0
        # And exactly one UserWarning of the expected kind should fire
        approx_warns = [
            ww for ww in w
            if issubclass(ww.category, UserWarning)
            and "marginal-matched" in str(ww.message)
        ]
        assert len(approx_warns) == 1

    def test_n2_interval_does_not_warn(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mpt.n_tuple_entropy(
                self.DIATONIC, 12, 2, sigma=0.3, sigma_space="interval"
            )
        approx_warns = [
            ww for ww in w
            if issubclass(ww.category, UserWarning)
            and "marginal-matched" in str(ww.message)
        ]
        assert approx_warns == []

    def test_invalid_sigma_space_errors(self):
        with pytest.raises(ValueError, match="sigma_space"):
            mpt.n_tuple_entropy(
                self.DIATONIC, 12, 1, sigma=0.3, sigma_space="bogus"
            )


# -------------------------------------------------------------------
#  dft_circular_simulate
# -------------------------------------------------------------------


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


class TestHarmony:
    def test_roughness_zero_for_unison(self):
        # Single frequency: no pairs → roughness = 0
        r = mpt.roughness([440], [1])
        assert r == pytest.approx(0.0)

    def test_roughness_positive(self):
        r = mpt.roughness([300, 330], [1, 1])
        assert r > 0

    def test_spectral_entropy_ji_vs_edo(self):
        spec = ["harmonic", 24, "powerlaw", 1]
        H_ji = mpt.spectral_entropy([0, 386.31, 701.96], None, 12, spectrum=spec)
        H_edo = mpt.spectral_entropy([0, 400, 700], None, 12, spectrum=spec)
        # JI triad should have lower spectral entropy (more consonant)
        assert H_ji < H_edo

    def test_template_harmonicity_returns_two(self):
        h_max, h_ent = mpt.template_harmonicity([0, 400, 700], None, 12)
        assert 0 < h_max <= 1
        assert 0 < h_ent <= 1

    def test_template_harmonicity_hEntropy_octave_below_cluster(self):
        """hEntropy is lower (more peaked cross-correlation = more
        harmonic) for an octave dyad than for a semitone cluster.

        This is the cleanest cardinality-controlled-after-the-fact
        contrast: a maximally consonant interval (octave) versus a
        densely dissonant cluster, both 2-3 notes."""
        _, h_ent_oct = mpt.template_harmonicity([0, 1200], None, 12)
        _, h_ent_clu = mpt.template_harmonicity([0, 100, 200], None, 12)
        assert h_ent_oct < h_ent_clu

    def test_template_harmonicity_hEntropy_major_below_cluster(self):
        """Cardinality-controlled comparison: a major triad and a
        three-tone semitone cluster are both 3-note chords, but the
        major triad's intervals approximate 4:5:6 of a shared
        fundamental, so its harmonic-template cross-correlation is
        more peaked (lower hEntropy) than the cluster's."""
        _, h_ent_maj = mpt.template_harmonicity([0, 400, 700], None, 12)
        _, h_ent_clu = mpt.template_harmonicity([0, 100, 200], None, 12)
        assert h_ent_maj < h_ent_clu

    def test_tensor_harmonicity_unison_high(self):
        spec = ["harmonic", 12, "powerlaw", 1]
        h_uni = mpt.tensor_harmonicity([0, 0], None, 12, spectrum=spec)
        h_tri = mpt.tensor_harmonicity([0, 600], None, 12, spectrum=spec)
        assert h_uni > h_tri

    def test_tensor_harmonicity_ranking_octave_p5_major_minor(self):
        """Ordered ranking against a harmonic spectrum:
        octave > perfect fifth > major triad > minor triad.

        Each step of this chain is musically motivated: the octave
        2:1 is the most harmonic interval; the fifth 3:2 the next;
        a major triad approximates 4:5:6; a minor triad's third
        (6:5) sits on a higher harmonic, so the chord matches the
        local harmonic-series r-ad density less strongly."""
        spec = ["harmonic", 12, "powerlaw", 1]
        h_oct = mpt.tensor_harmonicity([0, 1200],      None, 12, spectrum=spec)
        h_p5  = mpt.tensor_harmonicity([0, 700],       None, 12, spectrum=spec)
        h_maj = mpt.tensor_harmonicity([0, 400, 700],  None, 12, spectrum=spec)
        h_min = mpt.tensor_harmonicity([0, 300, 700],  None, 12, spectrum=spec)
        assert h_oct > h_p5 > h_maj > h_min

    def test_virtual_pitches_shape(self):
        vp_p, vp_w = mpt.virtual_pitches([0, 400, 700], None, 12)
        assert len(vp_p) == len(vp_w)
        assert len(vp_p) > 0

    def test_virtual_pitches_single_pitch_peak_at_pitch(self):
        """For a single pitch x, the strongest virtual pitch is at x:
        partial 1 of the harmonic template aligns with the chord's
        sole pitch."""
        vp_p, vp_w = mpt.virtual_pitches([400.0], None, 12)
        i_max = int(np.argmax(vp_w))
        assert abs(vp_p[i_max] - 400.0) < 5.0

    def test_virtual_pitches_octave_peak_at_lower_note(self):
        """For an octave dyad [0, 1200], partials 1 and 2 of a
        template rooted at 0 align with both chord notes
        simultaneously, producing the strongest virtual-pitch peak
        at 0 (the lower note)."""
        vp_p, vp_w = mpt.virtual_pitches([0.0, 1200.0], None, 12)
        i_max = int(np.argmax(vp_w))
        assert abs(vp_p[i_max] - 0.0) < 5.0


# ===================================================================
#  Input validation
# ===================================================================


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


class TestContinuity:
    def test_example_1_strict(self):
        c, m = mpt.continuity([3, 5, 7, 7, 9], [11], 0, mode="strict")
        assert c[0] == pytest.approx(1.0)
        assert m[0] == pytest.approx(2.0)

    def test_example_1_lenient(self):
        c, m = mpt.continuity([3, 5, 7, 7, 9], [11], 0, mode="lenient")
        assert c[0] == pytest.approx(3.0)
        assert m[0] == pytest.approx(6.0)

    def test_query_equals_last(self):
        c, m = mpt.continuity([3, 5, 7, 7, 9], [9], 0)
        assert c[0] == pytest.approx(0.0)
        assert m[0] == pytest.approx(0.0)

    def test_multi_query_shape(self):
        c, m = mpt.continuity([3, 5, 7, 7, 9], [10, 11, 12], 0,
                              mode="lenient")
        assert c.shape == (3,)
        assert m.shape == (3,)

    def test_seq_too_short(self):
        c, m = mpt.continuity([5], [11, 12, 13], 0)
        np.testing.assert_allclose(c, 0.0)
        np.testing.assert_allclose(m, 0.0)

    def test_descending_query_on_ascending_seq(self):
        c, m = mpt.continuity([1, 2, 3, 4, 5], [4], 0, mode="lenient")
        assert c[0] == pytest.approx(0.0)

    def test_magnitude_slope_on_arithmetic(self):
        c, m = mpt.continuity([1, 4, 7, 10, 13], [16], 0, mode="lenient")
        assert c[0] == pytest.approx(4.0)
        assert m[0] == pytest.approx(12.0)
        assert (m[0] / c[0]) == pytest.approx(3.0)

    def test_magnitude_signed_descending(self):
        c, m = mpt.continuity([10, 8, 6, 4], [2], 0, mode="strict")
        assert c[0] == pytest.approx(3.0)
        assert m[0] == pytest.approx(-6.0)

    def test_smoothing_converges(self):
        vals = [
            mpt.continuity([3, 5, 7, 7, 9], [11], s, mode="lenient")[0][0]
            for s in (5.0, 1.0, 0.1, 0.0)
        ]
        assert all(vals[i] <= vals[i + 1] + 1e-10
                   for i in range(len(vals) - 1))
        assert vals[-1] == pytest.approx(3.0)

    def test_explicit_theta_overrides_mode(self):
        c, _ = mpt.continuity([3, 5, 7, 7, 9], [11], 0, theta=-1.0)
        assert c[0] == pytest.approx(3.0)
        c, _ = mpt.continuity([3, 5, 7, 7, 9], [11], 0, theta=0.0)
        assert c[0] == pytest.approx(1.0)

    def test_theta_out_of_range(self):
        with pytest.raises(ValueError, match="theta"):
            mpt.continuity([3, 5, 7], [8], 0, theta=2.0)

    def test_bad_mode(self):
        with pytest.raises(ValueError, match="mode"):
            mpt.continuity([3, 5, 7], [8], 0, mode="wibble")

    # --- Weight-argument tests (v2.1.0) ---

    def test_w_none_equals_default(self):
        c1, m1 = mpt.continuity([3, 5, 7, 7, 9], [11], 0,
                                mode="lenient")
        c2, m2 = mpt.continuity([3, 5, 7, 7, 9], [11], 0,
                                w=None, mode="lenient")
        c3, m3 = mpt.continuity([3, 5, 7, 7, 9], [11], 0,
                                w=[], mode="lenient")
        np.testing.assert_allclose(c1, c2)
        np.testing.assert_allclose(m1, m2)
        np.testing.assert_allclose(c1, c3)
        np.testing.assert_allclose(m1, m3)

    def test_w_scalar_one_equals_unweighted(self):
        c_un, m_un = mpt.continuity([3, 5, 7, 7, 9], [11], 0,
                                    mode="lenient")
        c_w, m_w = mpt.continuity([3, 5, 7, 7, 9], [11], 0,
                                  w=1.0, mode="lenient")
        np.testing.assert_allclose(c_un, c_w)
        np.testing.assert_allclose(m_un, m_w)

    def test_w_scalar_scales_by_square(self):
        # Scalar c -> every difference event has salience c**2,
        # so count and magnitude both scale by c**2.
        c_un, m_un = mpt.continuity([3, 5, 7, 7, 9], [11], 0,
                                    mode="lenient")
        c_w, m_w = mpt.continuity([3, 5, 7, 7, 9], [11], 0,
                                  w=0.5, mode="lenient")
        np.testing.assert_allclose(c_w, 0.25 * c_un)
        np.testing.assert_allclose(m_w, 0.25 * m_un)

    def test_w_vector_recency_truncation(self):
        # Zero weights on the three oldest events zero out every
        # difference event except the most recent pair.
        c, m = mpt.continuity([3, 5, 7, 7, 9], [11], 0,
                              w=[0, 0, 0, 1, 1], mode="lenient")
        assert c[0] == pytest.approx(1.0)
        assert m[0] == pytest.approx(2.0)

    def test_w_vector_matches_rolling_product(self):
        # Explicit per-event weights — verify against hand calculation
        # using the rolling-product rule.
        seq = [3, 5, 7, 7, 9]
        w = [1, 1, 0.5, 1, 1]
        # Difference events:  (3->5)=2, (5->7)=2, (7->7)=0, (7->9)=2
        # Diff-event weights: 1*1=1,   1*0.5=0.5, 0.5*1=0.5, 1*1=1
        # Backward lenient walk from query 11:
        #   (7->9): a=1, contrib 1 * 1 = 1; c+=1, m+=2
        #   (7->7): a=0, contrib 0 * 0.5 = 0
        #   (5->7): a=1, contrib 1 * 0.5 = 0.5; c+=0.5, m+=1
        #   (3->5): a=1, contrib 1 * 1 = 1; c+=1, m+=2
        c, m = mpt.continuity(seq, [11], 0, w=w, mode="lenient")
        assert c[0] == pytest.approx(2.5)
        assert m[0] == pytest.approx(5.0)

    def test_w_matches_difference_events_directly(self):
        # Parity check: compute the rolling-product weights via
        # difference_events and verify continuity's internal scaling
        # agrees with the same weights applied outside.
        seq = [3, 5, 7, 7, 9]
        w = [0.8, 1.0, 0.7, 1.0, 0.9]
        p_d, w_d = mpt.difference_events(
            [np.asarray(seq).reshape(1, -1)],
            [np.asarray(w).reshape(1, -1)],
            None, [1], [0],
        )
        diff_weights = np.asarray(w_d[0]).reshape(-1)   # length N-1
        # Unweighted continuity for the same query, then apply the
        # rolling-product weights term by term — this should match
        # the weighted call because the break condition and sign
        # products are identical.
        ctx = np.asarray(p_d[0]).reshape(-1)
        # For σ = 0, sign-products give a_k ∈ {-1, 0, +1}. Step
        # through the backward walk manually.
        N = len(seq)
        a = np.sign(ctx) * np.sign(11 - seq[-1])
        c_expected = 0.0
        m_expected = 0.0
        for k in range(N - 2, -1, -1):
            if a[k] <= -1.0:
                break
            contrib = max(a[k], 0.0) * diff_weights[k]
            c_expected += contrib
            m_expected += contrib * ctx[k]
        c, m = mpt.continuity(seq, [11], 0, w=w, mode="lenient")
        assert c[0] == pytest.approx(c_expected)
        assert m[0] == pytest.approx(m_expected)

    def test_w_wrong_length_raises(self):
        with pytest.raises(ValueError, match="length N"):
            mpt.continuity([3, 5, 7, 7, 9], [11], 0, w=[1, 1, 1])

    def test_w_negative_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            mpt.continuity([3, 5, 7, 7, 9], [11], 0,
                           w=[1, 1, -0.1, 1, 1])

    def test_w_negative_scalar_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            mpt.continuity([3, 5, 7, 7, 9], [11], 0, w=-0.5)

    def test_w_with_smoothing(self):
        # Weights should compose with Gaussian smoothing: uniform
        # scalar w just scales both outputs by w**2 regardless of σ.
        seq = [3, 5, 7, 7, 9]
        c_un, m_un = mpt.continuity(seq, [11], 0.3, mode="lenient")
        c_w, m_w = mpt.continuity(seq, [11], 0.3, w=0.7, mode="lenient")
        np.testing.assert_allclose(c_w, 0.49 * c_un)
        np.testing.assert_allclose(m_w, 0.49 * m_un)


# ===================================================================
#  seq_weights
# ===================================================================


class TestSeqWeights:
    def test_primacy(self):
        v = mpt.seq_weights(None, "primacy", n=5)
        assert v.shape == (5,) and v[0] == 1.0 and np.all(v[1:] == 0.0)

    def test_recency(self):
        v = mpt.seq_weights(None, "recency", n=5)
        assert v.shape == (5,) and v[-1] == 1.0 and np.all(v[:-1] == 0.0)

    def test_exp_zero_decay_uniform(self):
        v = mpt.seq_weights(None, "exponentialFromEnd", n=5, decay_rate=0.0)
        np.testing.assert_allclose(v, np.ones(5))

    def test_exp_from_start_shape(self):
        v = mpt.seq_weights(None, "exponentialFromStart", n=5, decay_rate=1.0)
        np.testing.assert_allclose(v, np.exp(-np.arange(5, dtype=float)))

    def test_exp_from_end_shape(self):
        v = mpt.seq_weights(None, "exponentialFromEnd", n=5, decay_rate=1.0)
        np.testing.assert_allclose(v, np.exp(-np.arange(4, -1, -1, dtype=float)))

    def test_ushape_symmetric(self):
        v = mpt.seq_weights(None, "uShape", n=7, decay_rate=0.5, alpha=0.5)
        np.testing.assert_allclose(v, v[::-1])

    def test_ushape_alpha_limits(self):
        v_s = mpt.seq_weights(None, "exponentialFromStart", n=5, decay_rate=0.5)
        v_e = mpt.seq_weights(None, "exponentialFromEnd", n=5, decay_rate=0.5)
        np.testing.assert_allclose(
            mpt.seq_weights(None, "uShape", n=5, decay_rate=0.5, alpha=1.0), v_s
        )
        np.testing.assert_allclose(
            mpt.seq_weights(None, "uShape", n=5, decay_rate=0.5, alpha=0.0), v_e
        )

    def test_explicit_passthrough(self):
        profile = [0.1, 0.2, 0.4, 0.2, 0.1]
        np.testing.assert_allclose(mpt.seq_weights(None, profile, n=5), profile)

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="length"):
            mpt.seq_weights(None, [0.1, 0.2, 0.3], n=5)

    def test_unit_t_matches_unit_spacing(self):
        v1 = mpt.seq_weights(None, "exponentialFromEnd", n=5, decay_rate=1.0)
        v2 = mpt.seq_weights(None, "exponentialFromEnd", n=5, decay_rate=1.0,
                              t=[1, 2, 3, 4, 5])
        np.testing.assert_allclose(v1, v2)

    def test_non_increasing_t_errors(self):
        with pytest.raises(ValueError, match="increasing"):
            mpt.seq_weights(None, "exponentialFromEnd", n=5, t=[1, 2, 2, 3, 4])

    def test_unknown_spec(self):
        with pytest.raises(ValueError, match="Unknown"):
            mpt.seq_weights(None, "wibble", n=5)

    def test_negative_decay(self):
        with pytest.raises(ValueError, match="decay_rate"):
            mpt.seq_weights(None, "exponentialFromEnd", n=5, decay_rate=-1.0)

    def test_alpha_out_of_range(self):
        with pytest.raises(ValueError, match="alpha"):
            mpt.seq_weights(None, "uShape", n=5, decay_rate=1.0, alpha=1.5)

    def test_w_uniform_via_none(self):
        # None for w is equivalent to all ones
        v_none = mpt.seq_weights(None, "exponentialFromEnd", n=5,
                                  decay_rate=0.5)
        v_ones = mpt.seq_weights(np.ones(5), "exponentialFromEnd",
                                  decay_rate=0.5)
        np.testing.assert_allclose(v_none, v_ones)

    def test_w_multiplies_profile(self):
        w = np.array([0.8, 0.5, 1.0, 0.3, 0.9])
        v = mpt.seq_weights(w, "recency")
        # Recency profile is [0, 0, 0, 0, 1]; product picks w[-1]
        expected = np.array([0.0, 0.0, 0.0, 0.0, 0.9])
        np.testing.assert_allclose(v, expected)

    def test_w_multiplies_explicit_profile(self):
        w = np.array([2.0, 2.0, 2.0])
        profile = np.array([0.1, 0.5, 0.4])
        v = mpt.seq_weights(w, profile)
        np.testing.assert_allclose(v, 2.0 * profile)

    def test_w_length_mismatch_errors(self):
        # Explicit n conflicts with length of w
        with pytest.raises(ValueError, match="does not match"):
            mpt.seq_weights(np.array([1.0, 2.0, 3.0]), "flat", n=5)

    def test_missing_n_with_none_w(self):
        with pytest.raises(ValueError, match="n must be supplied"):
            mpt.seq_weights(None, "flat")

    def test_missing_n_with_scalar_w(self):
        with pytest.raises(ValueError, match="n must be supplied"):
            mpt.seq_weights(0.5, "flat")

    def test_scalar_w_broadcasts(self):
        v = mpt.seq_weights(0.5, "flat", n=4)
        np.testing.assert_allclose(v, 0.5 * np.ones(4))

    def test_n_inferred_from_w_matches_explicit_n(self):
        w = np.array([0.2, 0.8, 0.5])
        v_inferred = mpt.seq_weights(w, "recency")
        v_explicit = mpt.seq_weights(w, "recency", n=3)
        np.testing.assert_allclose(v_inferred, v_explicit)


# ===================================================================
#  multi-attribute expectation tensor (MAET)
# ===================================================================


class TestMAET:
    """Multi-attribute expectation tensor tests.

    The v2.1.0 extension to ``build_exp_tens``. The single-attribute
    legacy path is covered by ``TestTensor`` above; these tests focus on
    the MAET-specific behaviours: SA-equivalence under degenerate mapping,
    per-attribute perm/comb enumeration, weight broadcasting, group
    canonicalisation, NaN handling for variable-size events, and the
    error paths introduced by the per-attribute / per-group parameter
    structure.
    """

    # --- SA-equivalence: (N=1, A=1, K_a x 1 column weight) == SA -------

    def test_ma_matches_sa_abs(self):
        """MA with one event and one attribute reproduces SA bit-for-bit."""
        p = [0.0, 400.0, 700.0]
        w = [1.0, 0.7, 0.5]
        sigma, r, is_rel, is_per, period = 10.0, 2, False, True, 1200.0

        dens_sa = mpt.build_exp_tens(
            p, w, sigma, r, is_rel, is_per, period, verbose=False
        )

        p_attr = [np.array(p, dtype=float).reshape(3, 1)]
        w_ma = [np.array(w, dtype=float).reshape(3, 1)]
        dens_ma = mpt.build_exp_tens(
            p_attr, w_ma, [sigma], [r], None,
            [is_rel], [is_per], [period], verbose=False,
        )

        assert dens_ma.tag == "MaetDensity"
        assert dens_ma.n_j == dens_sa.n_j
        assert dens_ma.n_k == dens_sa.n_k
        np.testing.assert_array_equal(dens_ma.u_perm[0], dens_sa.u_perm)
        np.testing.assert_array_equal(dens_ma.v_comb[0], dens_sa.v_comb)
        np.testing.assert_array_equal(dens_ma.centres[0], dens_sa.centres)
        np.testing.assert_array_almost_equal(dens_ma.w_j, dens_sa.w_j)
        np.testing.assert_array_almost_equal(dens_ma.wv_comb, dens_sa.wv_comb)

    def test_ma_matches_sa_rel(self):
        """Centres reduction: is_rel=True collapses r_a dims to r_a-1."""
        p = [0.0, 400.0, 700.0]
        w = [1.0, 0.7, 0.5]
        sigma, r, is_rel, is_per, period = 10.0, 2, True, True, 1200.0

        dens_sa = mpt.build_exp_tens(
            p, w, sigma, r, is_rel, is_per, period, verbose=False
        )
        p_attr = [np.array(p, dtype=float).reshape(3, 1)]
        w_ma = [np.array(w, dtype=float).reshape(3, 1)]
        dens_ma = mpt.build_exp_tens(
            p_attr, w_ma, [sigma], [r], None,
            [is_rel], [is_per], [period], verbose=False,
        )
        assert dens_ma.centres[0].shape == (r - 1, dens_ma.n_j)
        np.testing.assert_array_equal(dens_ma.centres[0], dens_sa.centres)

    # --- Struct basics ------------------------------------------------

    def test_ma_struct_fields(self):
        """Essential fields for a pitch + time two-attribute build."""
        pitch = np.array([[0, 12], [4, 15], [7, 19]], dtype=float)  # 3 x 2
        time = np.array([[0.0, 1.0]])                               # 1 x 2
        dens = mpt.build_exp_tens(
            [pitch, time], None,
            [10.0, 0.1], [3, 1], None,
            [True, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )
        assert dens.tag == "MaetDensity"
        assert dens.n_attrs == 2
        assert dens.n_groups == 2
        assert dens.n == 2
        np.testing.assert_array_equal(dens.group_of_attr, [0, 1])
        np.testing.assert_array_equal(dens.r, [3, 1])
        np.testing.assert_array_equal(dens.k, [3, 1])
        # pitch contributes r_p - 1 = 2, time contributes r_t = 1
        assert dens.dim == 3
        np.testing.assert_array_equal(dens.dim_per_attr, [2, 1])

    def test_ma_cartesian_product_count(self):
        """n_j = sum_n (product of per-attr per-event perm counts)."""
        pitch = np.array([[0, 12, 5], [4, 15, 9], [7, 19, 12]], dtype=float)
        time = np.array([[0.0, 1.0, 2.0]])
        dens = mpt.build_exp_tens(
            [pitch, time], None,
            [10.0, 0.1], [3, 1], None,
            [True, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )
        # Per event: pitch r=3, K=3 -> P(3,3)=6 perms, C(3,3)=1 comb.
        # Time r=1, K=1 -> 1 each. Cartesian per event: 6 perms, 1 comb.
        # Three events: 18 perms, 3 combs.
        assert dens.n_j == 18
        assert dens.n_k == 3

    def test_ma_event_bookkeeping(self):
        pitch = np.array([[0, 12, 5], [4, 15, 9], [7, 19, 12]], dtype=float)
        time = np.array([[0.0, 1.0, 2.0]])
        dens = mpt.build_exp_tens(
            [pitch, time], None,
            [10.0, 0.1], [3, 1], None,
            [True, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )
        np.testing.assert_array_equal(dens.event_of_j, np.repeat([0, 1, 2], 6))
        np.testing.assert_array_equal(dens.event_of_k, [0, 1, 2])

    # --- Weight broadcasting -----------------------------------------

    def test_ma_weight_none(self):
        pitch = np.array([[0, 4], [4, 8]], dtype=float)
        dens = mpt.build_exp_tens(
            [pitch], None, [10.0], [2], None,
            [False], [True], [1200.0], verbose=False,
        )
        np.testing.assert_array_equal(dens.w[0], np.ones((2, 2)))

    def test_ma_weight_scalar_top_level(self):
        pitch = np.array([[0, 4], [4, 8]], dtype=float)
        dens = mpt.build_exp_tens(
            [pitch], 0.5, [10.0], [2], None,
            [False], [True], [1200.0], verbose=False,
        )
        np.testing.assert_array_equal(dens.w[0], np.full((2, 2), 0.5))

    def test_ma_weight_2d_per_event_row(self):
        pitch = np.array([[0, 4, 5], [4, 8, 6]], dtype=float)  # K=2, N=3
        w_row = np.array([[0.5, 1.0, 2.0]])                    # (1, 3)
        dens = mpt.build_exp_tens(
            [pitch], [w_row], [10.0], [2], None,
            [False], [True], [1200.0], verbose=False,
        )
        np.testing.assert_array_equal(
            dens.w[0], np.array([[0.5, 1.0, 2.0], [0.5, 1.0, 2.0]])
        )

    def test_ma_weight_2d_per_slot_column(self):
        pitch = np.array(
            [[0, 4, 5], [4, 8, 6], [7, 10, 9]], dtype=float
        )  # K=3, N=3
        w_col = np.array([[0.5], [1.0], [2.0]])                # (3, 1)
        dens = mpt.build_exp_tens(
            [pitch], [w_col], [10.0], [2], None,
            [False], [True], [1200.0], verbose=False,
        )
        np.testing.assert_array_equal(
            dens.w[0], np.tile([[0.5], [1.0], [2.0]], (1, 3))
        )

    def test_ma_weight_2d_full_matrix(self):
        pitch = np.array([[0, 4], [4, 8]], dtype=float)
        W = np.array([[0.1, 0.2], [0.3, 0.4]])
        dens = mpt.build_exp_tens(
            [pitch], [W], [10.0], [2], None,
            [False], [True], [1200.0], verbose=False,
        )
        np.testing.assert_array_equal(dens.w[0], W)

    def test_ma_weight_1d_per_event_disambiguated(self):
        pitch = np.array([[0, 4], [4, 8], [7, 9]], dtype=float)  # K=3, N=2
        w_1d = np.array([0.5, 1.0])  # length N=2 (not K=3) -> per-event
        dens = mpt.build_exp_tens(
            [pitch], [w_1d], [10.0], [2], None,
            [False], [True], [1200.0], verbose=False,
        )
        expected = np.array([[0.5, 1.0], [0.5, 1.0], [0.5, 1.0]])
        np.testing.assert_array_equal(dens.w[0], expected)

    def test_ma_weight_1d_per_slot_disambiguated(self):
        pitch = np.array([[0, 4], [4, 8], [7, 9]], dtype=float)  # K=3, N=2
        w_1d = np.array([0.5, 1.0, 2.0])  # length K=3 (not N=2) -> per-slot
        dens = mpt.build_exp_tens(
            [pitch], [w_1d], [10.0], [2], None,
            [False], [True], [1200.0], verbose=False,
        )
        expected = np.tile([[0.5], [1.0], [2.0]], (1, 2))
        np.testing.assert_array_equal(dens.w[0], expected)

    def test_ma_weight_1d_ambiguous_when_K_equals_N(self):
        pitch = np.array(
            [[0, 4, 5], [4, 8, 6], [7, 10, 9]], dtype=float
        )  # K=3, N=3
        w_1d = np.array([0.5, 1.0, 2.0])
        with pytest.raises(ValueError, match="ambiguous"):
            mpt.build_exp_tens(
                [pitch], [w_1d], [10.0], [2], None,
                [False], [True], [1200.0], verbose=False,
            )

    # --- Groups -------------------------------------------------------

    def test_ma_groups_default_singleton(self):
        pitch = np.array([[0, 4]], dtype=float)
        time = np.array([[0.0, 1.0]])
        dens = mpt.build_exp_tens(
            [pitch, time], None, [10.0, 0.1], [1, 1], None,
            [False, False], [True, False], [1200.0, 0.0], verbose=False,
        )
        assert dens.n_groups == 2
        np.testing.assert_array_equal(dens.group_of_attr, [0, 1])

    def test_ma_groups_vector_and_cell_agree(self):
        pitch = np.array([[0, 4]], dtype=float)
        time = np.array([[0.0, 1.0]])
        x = np.array([[0.0, 0.5]])
        y = np.array([[0.0, 0.5]])
        z = np.array([[0.0, 0.5]])

        args_rest = (
            [10.0, 0.1, 0.2], [1, 1, 1, 1, 1],   # sigma_vec, r_vec
        )
        flags = ([False, False, False], [True, False, False],
                 [1200.0, 0.0, 0.0])

        dens_v = mpt.build_exp_tens(
            [pitch, time, x, y, z], None,
            *args_rest, [0, 1, 2, 2, 2], *flags, verbose=False,
        )
        dens_c = mpt.build_exp_tens(
            [pitch, time, x, y, z], None,
            *args_rest, [[0], [1], [2, 3, 4]], *flags, verbose=False,
        )
        np.testing.assert_array_equal(dens_v.group_of_attr, dens_c.group_of_attr)
        assert dens_v.n_groups == dens_c.n_groups
        for g in range(dens_v.n_groups):
            np.testing.assert_array_equal(
                dens_v.attrs_of_group[g], dens_c.attrs_of_group[g]
            )

    def test_ma_groups_noncontiguous_errors(self):
        pitch = np.array([[0, 4]], dtype=float)
        with pytest.raises(ValueError, match="contiguous"):
            mpt.build_exp_tens(
                [pitch, pitch], None, [10.0, 10.0], [1, 1], [0, 2],
                [False, False], [True, True], [1200.0, 1200.0], verbose=False,
            )

    def test_ma_groups_cell_duplicate_attr_errors(self):
        pitch = np.array([[0, 4]], dtype=float)
        with pytest.raises(ValueError, match="more than one group"):
            mpt.build_exp_tens(
                [pitch, pitch], None, [10.0], [1, 1], [[0, 1], [1]],
                [False], [True], [1200.0], verbose=False,
            )

    # --- NaN-padded variable-size events -----------------------------

    def test_ma_nan_padding(self):
        # Event 0: 3 pitches. Event 1: 2 pitches (third slot NaN).
        pitch = np.array([[0, 0], [4, 4], [7, np.nan]], dtype=float)
        time = np.array([[0.0, 1.0]])
        dens = mpt.build_exp_tens(
            [pitch, time], None, [10.0, 0.1], [2, 1], None,
            [False, False], [True, False], [1200.0, 0.0], verbose=False,
        )
        # Event 0: P(3,2)=6 perms, C(3,2)=3 combs. Event 1: P(2,2)=2, C(2,2)=1.
        # Cartesian x 1 time = same. Totals: n_j = 8, n_k = 4.
        assert dens.n_j == 8
        assert dens.n_k == 4

    # --- Per-tuple weight factorisation ------------------------------

    def test_ma_per_tuple_weight_product(self):
        # One event, K_p=2 with slot weights 2 and 3; r_p=2.
        # Time K=1, slot weight 5; r_t=1.
        # Each perm tuple weight = (w_i * w_j for pitch) * 5 for time.
        pitch = np.array([[0.0], [4.0]])
        time = np.array([[1.5]])
        w_pitch = np.array([[2.0], [3.0]])
        w_time = np.array([[5.0]])
        dens = mpt.build_exp_tens(
            [pitch, time], [w_pitch, w_time],
            [10.0, 0.1], [2, 1], None,
            [False, False], [True, False], [1200.0, 0.0], verbose=False,
        )
        # 2 pitch perms, each with weight 2*3*5 = 30.
        np.testing.assert_array_almost_equal(dens.w_j, [30.0, 30.0])
        # 1 pitch comb, weight 2*3*5 = 30.
        np.testing.assert_array_almost_equal(dens.wv_comb, [30.0])

    # --- Error paths -------------------------------------------------

    def test_ma_insufficient_slots_errors(self):
        # Event 1 has 1 valid slot, r=2 -> error
        pitch = np.array(
            [[0, 0], [4, np.nan], [np.nan, np.nan]], dtype=float
        )
        with pytest.raises(ValueError, match="non-NaN slot"):
            mpt.build_exp_tens(
                [pitch], None, [10.0], [2], None,
                [False], [True], [1200.0], verbose=False,
            )

    def test_ma_wrong_r_vec_length(self):
        pitch = np.array([[0, 4]], dtype=float)
        with pytest.raises(ValueError, match="r_vec"):
            mpt.build_exp_tens(
                [pitch, pitch], None, [10.0, 10.0], [1], None,
                [False, False], [True, True], [1200.0, 1200.0], verbose=False,
            )

    def test_ma_wrong_sigma_length(self):
        pitch = np.array([[0, 4]], dtype=float)
        with pytest.raises(ValueError, match="sigma_vec"):
            mpt.build_exp_tens(
                [pitch, pitch], None, [10.0], [1, 1], None,
                [False, False], [True, True], [1200.0, 1200.0], verbose=False,
            )

    def test_ma_mismatched_event_counts(self):
        pitch = np.array([[0, 4]], dtype=float)       # N=2
        time = np.array([[0.0, 1.0, 2.0]])             # N=3
        with pytest.raises(ValueError, match="share N"):
            mpt.build_exp_tens(
                [pitch, time], None, [10.0, 0.1], [1, 1], None,
                [False, False], [True, False], [1200.0, 0.0], verbose=False,
            )

    def test_ma_isrel_r1_warns(self):
        pitch = np.array([[0, 4]], dtype=float)
        with pytest.warns(UserWarning, match="degenerate"):
            mpt.build_exp_tens(
                [pitch], None, [10.0], [1], None,
                [True], [True], [1200.0], verbose=False,
            )

    def test_ma_wrong_positional_count(self):
        pitch = np.array([[0, 4]], dtype=float)
        # 7 positional args for MA is wrong (should be 8)
        with pytest.raises(ValueError, match="8 positional"):
            mpt.build_exp_tens(
                [pitch], None, [10.0], [1],
                [False], [True], [1200.0], verbose=False,
            )

    # --- evalExpTens MA path ------------------------------------------

    def test_ma_eval_matches_sa_abs(self):
        """MA eval matches SA at the same query points (is_rel=False)."""
        p = [0.0, 400.0, 700.0]
        w = [1.0, 0.7, 0.5]
        sigma, r, is_rel, is_per, period = 10.0, 2, False, True, 1200.0
        x_sa = np.array([[100, 500], [300, 600]], dtype=float)  # 2 x 2 (SA)

        dens_sa = mpt.build_exp_tens(
            p, w, sigma, r, is_rel, is_per, period, verbose=False
        )
        vals_sa = mpt.eval_exp_tens(dens_sa, x_sa, verbose=False)

        p_attr = [np.array(p, dtype=float).reshape(3, 1)]
        w_ma = [np.array(w, dtype=float).reshape(3, 1)]
        dens_ma = mpt.build_exp_tens(
            p_attr, w_ma, [sigma], [r], None,
            [is_rel], [is_per], [period], verbose=False,
        )
        vals_ma_cell = mpt.eval_exp_tens(dens_ma, [x_sa], verbose=False)
        vals_ma_mat = mpt.eval_exp_tens(dens_ma, x_sa, verbose=False)

        np.testing.assert_allclose(vals_ma_cell, vals_sa, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(vals_ma_mat, vals_sa, rtol=1e-12, atol=1e-12)

    def test_ma_eval_matches_sa_rel(self):
        """Same with is_rel=True: reduced-dim query points."""
        p = [0.0, 400.0, 700.0]
        w = [1.0, 0.7, 0.5]
        sigma, r, is_rel, is_per, period = 10.0, 3, True, True, 1200.0
        x_sa = np.array([[400, 200], [700, 500]], dtype=float)  # (r-1) x nQ

        dens_sa = mpt.build_exp_tens(
            p, w, sigma, r, is_rel, is_per, period, verbose=False
        )
        p_attr = [np.array(p, dtype=float).reshape(3, 1)]
        w_ma = [np.array(w, dtype=float).reshape(3, 1)]
        dens_ma = mpt.build_exp_tens(
            p_attr, w_ma, [sigma], [r], None,
            [is_rel], [is_per], [period], verbose=False,
        )

        vals_sa = mpt.eval_exp_tens(dens_sa, x_sa, verbose=False)
        vals_ma = mpt.eval_exp_tens(dens_ma, [x_sa], verbose=False)
        np.testing.assert_allclose(vals_ma, vals_sa, rtol=1e-12, atol=1e-12)

    def test_ma_eval_normalisation_matches_sa(self):
        """'gaussian' and 'pdf' normalisation modes match SA."""
        p = [0.0, 400.0, 700.0]
        w = [1.0, 0.7, 0.5]
        sigma, r, is_rel, is_per, period = 10.0, 3, True, True, 1200.0
        x_sa = np.array([[400, 200], [700, 500]], dtype=float)

        dens_sa = mpt.build_exp_tens(
            p, w, sigma, r, is_rel, is_per, period, verbose=False
        )
        p_attr = [np.array(p, dtype=float).reshape(3, 1)]
        w_ma = [np.array(w, dtype=float).reshape(3, 1)]
        dens_ma = mpt.build_exp_tens(
            p_attr, w_ma, [sigma], [r], None,
            [is_rel], [is_per], [period], verbose=False,
        )
        for mode in ("gaussian", "pdf"):
            vals_sa = mpt.eval_exp_tens(dens_sa, x_sa, mode, verbose=False)
            vals_ma = mpt.eval_exp_tens(dens_ma, [x_sa], mode, verbose=False)
            np.testing.assert_allclose(vals_ma, vals_sa, rtol=1e-12, atol=1e-12)

    def test_ma_eval_cell_vs_matrix_forms_agree(self):
        """Cell form and single-matrix form give identical results."""
        pitch = np.array([[0.0, 4.0, 7.0]]).T        # K=3, N=1
        time  = np.array([[1.0]])                     # K=1, N=1
        dens = mpt.build_exp_tens(
            [pitch, time], None,
            [10.0, 0.1], [2, 1], None,
            [False, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )
        # dim_per_attr = [2, 1], total dim = 3
        x_pitch = np.array([[0.0, 4.0], [4.0, 7.0]])  # 2 x 2
        x_time  = np.array([[1.0, 2.0]])               # 1 x 2
        vals_cell = mpt.eval_exp_tens(dens, [x_pitch, x_time], verbose=False)
        x_mat = np.vstack([x_pitch, x_time])           # 3 x 2
        vals_mat = mpt.eval_exp_tens(dens, x_mat, verbose=False)
        np.testing.assert_array_equal(vals_cell, vals_mat)

    def test_ma_eval_per_group_isper(self):
        """Periodic pitch wraps; nonperiodic time does not."""
        pitch = np.array([[0.0]])
        time  = np.array([[0.0]])
        dens = mpt.build_exp_tens(
            [pitch, time], None,
            [20.0, 20.0], [1, 1], None,
            [False, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )
        # Pitch at 0 vs 1200 under periodic pitch => equal density
        v_pitch_0    = mpt.eval_exp_tens(
            dens, [np.array([[0.0]]),    np.array([[0.0]])], verbose=False
        )[0]
        v_pitch_1200 = mpt.eval_exp_tens(
            dens, [np.array([[1200.0]]), np.array([[0.0]])], verbose=False
        )[0]
        np.testing.assert_allclose(v_pitch_0, v_pitch_1200, rtol=1e-12)

        # Time at 0 vs 1200 under nonperiodic time => strictly lower at 1200
        v_time_0    = mpt.eval_exp_tens(
            dens, [np.array([[0.0]]), np.array([[0.0]])], verbose=False
        )[0]
        v_time_1200 = mpt.eval_exp_tens(
            dens, [np.array([[0.0]]), np.array([[1200.0]])], verbose=False
        )[0]
        assert v_time_1200 < v_time_0

    def test_ma_eval_positive_at_tuple_centre(self):
        """Density at a tuple centre is positive and at least as high as
        at a point far from every tuple."""
        pitch = np.array([[0.0, 4.0, 7.0]]).T   # K=3, N=1
        time  = np.array([[1.0]])
        dens = mpt.build_exp_tens(
            [pitch, time], None,
            [10.0, 0.1], [2, 1], None,
            [False, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )
        # At tuple centre (pitch=(0,4), time=1.0): one of the perm tuples
        x_centre = [np.array([[0.0], [4.0]]), np.array([[1.0]])]
        x_far    = [np.array([[600.0], [800.0]]), np.array([[50.0]])]
        v_centre = mpt.eval_exp_tens(dens, x_centre, verbose=False)[0]
        v_far    = mpt.eval_exp_tens(dens, x_far,    verbose=False)[0]
        assert v_centre > 0
        assert v_centre > v_far

    def test_ma_eval_wrong_cell_length_errors(self):
        pitch = np.array([[0.0, 4.0]]).T
        time  = np.array([[1.0]])
        dens = mpt.build_exp_tens(
            [pitch, time], None,
            [10.0, 0.1], [2, 1], None,
            [False, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )
        # Only one cell provided, expected 2
        with pytest.raises(ValueError, match="length 2"):
            mpt.eval_exp_tens(
                dens, [np.array([[0.0], [4.0]])], verbose=False
            )

    def test_ma_eval_wrong_per_attr_rows_errors(self):
        pitch = np.array([[0.0, 4.0]]).T
        time  = np.array([[1.0]])
        dens = mpt.build_exp_tens(
            [pitch, time], None,
            [10.0, 0.1], [2, 1], None,
            [False, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )
        # pitch query has 3 rows instead of 2
        with pytest.raises(ValueError, match="attribute 0"):
            mpt.eval_exp_tens(
                dens, [np.zeros((3, 1)), np.zeros((1, 1))], verbose=False
            )

    def test_ma_eval_wrong_total_rows_errors(self):
        pitch = np.array([[0.0, 4.0]]).T
        time  = np.array([[1.0]])
        dens = mpt.build_exp_tens(
            [pitch, time], None,
            [10.0, 0.1], [2, 1], None,
            [False, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )
        # Single-matrix form: dim=3 expected, we pass 5 rows
        with pytest.raises(ValueError, match="total dim"):
            mpt.eval_exp_tens(dens, np.zeros((5, 1)), verbose=False)

    def test_ma_eval_empty_query(self):
        pitch = np.array([[0.0, 4.0]]).T
        dens = mpt.build_exp_tens(
            [pitch], None, [10.0], [2], None,
            [False], [True], [1200.0], verbose=False,
        )
        vals = mpt.eval_exp_tens(dens, np.zeros((2, 0)), verbose=False)
        assert vals.shape == (0,)

    def test_ma_eval_dispatch_on_type(self):
        """Public eval_exp_tens dispatches on dens type."""
        # ExpTensDensity -> SA path
        dens_sa = mpt.build_exp_tens(
            [0.0, 4.0], None, 10.0, 2, False, True, 1200.0, verbose=False
        )
        assert isinstance(dens_sa, mpt.ExpTensDensity)
        # MaetDensity -> MA path
        dens_ma = mpt.build_exp_tens(
            [np.array([[0.0, 4.0]]).T], None, [10.0], [2], None,
            [False], [True], [1200.0], verbose=False,
        )
        assert isinstance(dens_ma, mpt.MaetDensity)
        # Both evaluate successfully
        x = np.array([[0.0], [4.0]])
        mpt.eval_exp_tens(dens_sa, x, verbose=False)
        mpt.eval_exp_tens(dens_ma, x, verbose=False)

    # --- cosSimExpTens MA path ---------------------------------------

    def test_ma_cossim_matches_sa_abs(self):
        """MA cos-sim matches SA at the SA-equivalence mapping (is_rel=False)."""
        p_a = [0.0, 400.0, 700.0]
        p_b = [0.0, 300.0, 700.0]
        w_a = [1.0, 0.7, 0.5]
        w_b = [1.0, 0.6, 0.8]
        sigma, r, is_rel, is_per, period = 10.0, 2, False, True, 1200.0

        s_sa = mpt.cos_sim_exp_tens_raw(
            p_a, w_a, p_b, w_b, sigma, r, is_rel, is_per, period,
            verbose=False,
        )

        # MA form: one attribute, one event, column weight
        da = mpt.build_exp_tens(
            [np.array(p_a).reshape(3, 1)], [np.array(w_a).reshape(3, 1)],
            [sigma], [r], None, [is_rel], [is_per], [period], verbose=False,
        )
        db = mpt.build_exp_tens(
            [np.array(p_b).reshape(3, 1)], [np.array(w_b).reshape(3, 1)],
            [sigma], [r], None, [is_rel], [is_per], [period], verbose=False,
        )
        s_ma = mpt.cos_sim_exp_tens(da, db, verbose=False)

        np.testing.assert_allclose(s_ma, s_sa, rtol=1e-12, atol=1e-12)

    def test_ma_cossim_matches_sa_rel(self):
        """SA-equivalence with is_rel=True (uses pairwise-diff formula periodically)."""
        p_a = [0.0, 400.0, 700.0]
        p_b = [0.0, 300.0, 700.0]
        w_a = [1.0, 0.7, 0.5]
        w_b = [1.0, 0.6, 0.8]
        for r, is_per, period in [(2, True, 1200.0),
                                   (3, True, 1200.0),
                                   (3, False, 0.0)]:
            s_sa = mpt.cos_sim_exp_tens_raw(
                p_a, w_a, p_b, w_b, 10.0, r, True, is_per, period, verbose=False
            )
            da = mpt.build_exp_tens(
                [np.array(p_a).reshape(3, 1)], [np.array(w_a).reshape(3, 1)],
                [10.0], [r], None, [True], [is_per], [period], verbose=False,
            )
            db = mpt.build_exp_tens(
                [np.array(p_b).reshape(3, 1)], [np.array(w_b).reshape(3, 1)],
                [10.0], [r], None, [True], [is_per], [period], verbose=False,
            )
            s_ma = mpt.cos_sim_exp_tens(da, db, verbose=False)
            np.testing.assert_allclose(
                s_ma, s_sa, rtol=1e-12, atol=1e-12,
                err_msg=f"r={r}, is_per={is_per}, period={period}",
            )

    def test_ma_cossim_self_is_one(self):
        """cos_sim(d, d) == 1 for a non-degenerate MA density."""
        pitch = np.array([[0.0, 12.0], [4.0, 15.0], [7.0, 19.0]])
        time  = np.array([[0.0, 1.0]])
        d = mpt.build_exp_tens(
            [pitch, time], None,
            [10.0, 0.1], [3, 1], None,
            [True, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )
        s = mpt.cos_sim_exp_tens(d, d, verbose=False)
        np.testing.assert_allclose(s, 1.0, rtol=1e-12, atol=1e-12)

    def test_ma_cossim_symmetry(self):
        """cos_sim(a, b) == cos_sim(b, a)."""
        pitchA = np.array([[0.0, 12.0], [4.0, 15.0], [7.0, 19.0]])
        timeA  = np.array([[0.0, 1.0]])
        pitchB = np.array([[0.0, 10.0], [4.0, 13.0], [7.0, 17.0]])
        timeB  = np.array([[0.0, 1.2]])

        da = mpt.build_exp_tens(
            [pitchA, timeA], None,
            [10.0, 0.1], [3, 1], None,
            [True, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )
        db = mpt.build_exp_tens(
            [pitchB, timeB], None,
            [10.0, 0.1], [3, 1], None,
            [True, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )
        s_ab = mpt.cos_sim_exp_tens(da, db, verbose=False)
        s_ba = mpt.cos_sim_exp_tens(db, da, verbose=False)
        np.testing.assert_allclose(s_ab, s_ba, rtol=1e-12, atol=1e-12)

    def test_ma_cossim_isrel_transposition_invariance(self):
        """Shifting all pitches by a constant preserves cos_sim when
        the pitch group has is_rel=True (one event, so the shift affects
        every pitch slot equally)."""
        pitch = np.array([[0.0], [400.0], [700.0]])  # K=3, N=1
        time  = np.array([[1.0]])
        pitch_shifted = pitch + 137.0

        d1 = mpt.build_exp_tens(
            [pitch, time], None,
            [10.0, 0.1], [3, 1], None,
            [True, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )
        d2 = mpt.build_exp_tens(
            [pitch_shifted, time], None,
            [10.0, 0.1], [3, 1], None,
            [True, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )
        s = mpt.cos_sim_exp_tens(d1, d2, verbose=False)
        np.testing.assert_allclose(s, 1.0, rtol=1e-10, atol=1e-10)

    def test_ma_cossim_raw_matches_struct(self):
        """cos_sim_exp_tens_raw (MA) == build + cos_sim_exp_tens (MA)."""
        pitchA = np.array([[0.0, 12.0], [4.0, 15.0], [7.0, 19.0]])
        timeA  = np.array([[0.0, 1.0]])
        pitchB = np.array([[0.0, 10.0], [4.0, 13.0], [7.0, 17.0]])
        timeB  = np.array([[0.0, 1.2]])

        # Raw MA form: 10 positional args
        s_raw = mpt.cos_sim_exp_tens_raw(
            [pitchA, timeA], None, [pitchB, timeB], None,
            [10.0, 0.1], [3, 1], None,
            [True, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )
        da = mpt.build_exp_tens(
            [pitchA, timeA], None,
            [10.0, 0.1], [3, 1], None,
            [True, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )
        db = mpt.build_exp_tens(
            [pitchB, timeB], None,
            [10.0, 0.1], [3, 1], None,
            [True, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )
        s_struct = mpt.cos_sim_exp_tens(da, db, verbose=False)
        np.testing.assert_allclose(s_raw, s_struct, rtol=1e-12, atol=1e-12)

    def test_ma_cossim_raw_sa_still_works(self):
        """SA raw-args call unchanged from v2.0.0 behaviour."""
        s = mpt.cos_sim_exp_tens_raw(
            [0.0, 4.0, 7.0], None, [0.0, 4.0, 7.0], None,
            10.0, 2, True, True, 1200.0, verbose=False,
        )
        np.testing.assert_allclose(s, 1.0, rtol=1e-12, atol=1e-12)

    def test_ma_cossim_raw_mismatched_types_errors(self):
        """p1 is MA but p2 is SA -> TypeError."""
        pitch_ma = [np.array([[0.0, 4.0]]).T]
        pitch_sa = [0.0, 4.0]
        with pytest.raises(TypeError, match="same kind"):
            mpt.cos_sim_exp_tens_raw(
                pitch_ma, None, pitch_sa, None,
                10.0, 2, False, True, 1200.0, verbose=False,
            )

    def test_ma_cossim_mixed_struct_types_errors(self):
        """MaetDensity paired with ExpTensDensity -> TypeError."""
        d_sa = mpt.build_exp_tens(
            [0.0, 4.0, 7.0], None, 10.0, 2, False, True, 1200.0, verbose=False
        )
        d_ma = mpt.build_exp_tens(
            [np.array([[0.0, 4.0, 7.0]]).T], None,
            [10.0], [2], None, [False], [True], [1200.0], verbose=False,
        )
        with pytest.raises(TypeError, match="same type"):
            mpt.cos_sim_exp_tens(d_sa, d_ma, verbose=False)
        with pytest.raises(TypeError, match="same type"):
            mpt.cos_sim_exp_tens(d_ma, d_sa, verbose=False)

    def test_ma_cossim_parameter_mismatch_errors(self):
        """Mismatched MA densities raise specific ValueErrors."""
        pitch = np.array([[0.0, 4.0, 7.0]]).T
        base_kwargs = dict(
            p_attr=[pitch], w=None,
            sigma_vec=[10.0], r_vec=[2], groups=None,
            is_rel_vec=[False], is_per_vec=[True], period_vec=[1200.0],
        )
        d_ref = mpt.build_exp_tens(
            base_kwargs["p_attr"], base_kwargs["w"],
            base_kwargs["sigma_vec"], base_kwargs["r_vec"], base_kwargs["groups"],
            base_kwargs["is_rel_vec"], base_kwargs["is_per_vec"],
            base_kwargs["period_vec"], verbose=False,
        )
        # Different r_vec
        d_r = mpt.build_exp_tens(
            [pitch], None, [10.0], [3], None,
            [False], [True], [1200.0], verbose=False,
        )
        with pytest.raises(ValueError, match="r"):
            mpt.cos_sim_exp_tens(d_ref, d_r, verbose=False)
        # Different sigma
        d_s = mpt.build_exp_tens(
            [pitch], None, [20.0], [2], None,
            [False], [True], [1200.0], verbose=False,
        )
        with pytest.raises(ValueError, match="sigma"):
            mpt.cos_sim_exp_tens(d_ref, d_s, verbose=False)
        # Different is_rel
        d_rel = mpt.build_exp_tens(
            [pitch], None, [10.0], [2], None,
            [True], [True], [1200.0], verbose=False,
        )
        with pytest.raises(ValueError, match="is_rel"):
            mpt.cos_sim_exp_tens(d_ref, d_rel, verbose=False)
        # Different period on periodic group
        d_p = mpt.build_exp_tens(
            [pitch], None, [10.0], [2], None,
            [False], [True], [2400.0], verbose=False,
        )
        with pytest.raises(ValueError, match="period"):
            mpt.cos_sim_exp_tens(d_ref, d_p, verbose=False)

    # --- entropyExpTens MA path -------------------------------------

    def test_ma_entropy_sa_equivalence_periodic(self):
        """MA entropy matches SA entropy at the SA-equivalence mapping
        (single periodic group, is_rel=False)."""
        p = np.array([0.0, 4.0, 7.0])
        w = np.array([1.0, 1.0, 1.0])
        H_sa = mpt.entropy_exp_tens(
            p, w, 10.0, 1, False, True, 12.0,
            n_points_per_dim=400,
        )
        H_ma = mpt.entropy_exp_tens(
            [p.reshape(3, 1)], [w.reshape(3, 1)],
            [10.0], [1], None, [False], [True], [12.0],
            n_points_per_dim=400,
        )
        np.testing.assert_allclose(H_ma, H_sa, rtol=1e-10, atol=1e-10)

    def test_ma_entropy_sa_equivalence_nonperiodic(self):
        """MA entropy matches SA entropy for a non-periodic group with
        explicit bounds."""
        p = np.array([0.0, 4.0, 7.0])
        w = np.array([1.0, 1.0, 1.0])
        H_sa = mpt.entropy_exp_tens(
            p, w, 10.0, 1, False, False, 0.0,
            x_min=-3.0, x_max=10.0, n_points_per_dim=400,
        )
        H_ma = mpt.entropy_exp_tens(
            [p.reshape(3, 1)], [w.reshape(3, 1)],
            [10.0], [1], None, [False], [False], [0.0],
            x_min=-3.0, x_max=10.0, n_points_per_dim=400,
        )
        np.testing.assert_allclose(H_ma, H_sa, rtol=1e-10, atol=1e-10)

    def test_ma_entropy_uniform_pitch_high(self):
        """Chromatic scale with wide sigma gives near-uniform pmf,
        so normalised entropy is close to 1."""
        p = np.arange(12, dtype=np.float64)
        H = mpt.entropy_exp_tens(
            [p.reshape(12, 1)], None,
            [100.0], [1], None, [False], [True], [12.0],
            n_points_per_dim=400,
        )
        assert H > 0.95

    def test_ma_entropy_concentrated_below_uniform(self):
        """A single pitch is more concentrated than the chromatic
        scale, so gives lower normalised entropy."""
        p_one = np.array([5.0])
        H_one = mpt.entropy_exp_tens(
            [p_one.reshape(1, 1)], None,
            [20.0], [1], None, [False], [True], [12.0],
            n_points_per_dim=400,
        )
        p_all = np.arange(12, dtype=np.float64)
        H_all = mpt.entropy_exp_tens(
            [p_all.reshape(12, 1)], None,
            [20.0], [1], None, [False], [True], [12.0],
            n_points_per_dim=400,
        )
        assert H_one < H_all

    def test_ma_entropy_pitch_plus_time_runs(self):
        """A 3-attribute density with pitch (periodic) + time
        (non-periodic) produces a finite entropy and the grid
        dimensionality is handled correctly."""
        pitch = np.array([[0.0, 12.0], [4.0, 15.0], [7.0, 19.0]])  # 3 x 2
        time = np.array([[0.0, 1.0]])                               # 1 x 2
        dens = mpt.build_exp_tens(
            [pitch, time], None,
            [20.0, 0.1], [2, 1], None,
            [True, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )
        # dim = (2-1) + 1 = 2
        assert dens.dim == 2
        H = mpt.entropy_exp_tens(
            dens,
            x_min=-0.5, x_max=1.5,
            n_points_per_dim=80,
        )
        assert 0.0 < H < 1.0

    def test_ma_entropy_grid_limit_errors(self):
        """Excessively large grid requests error with a suggestion."""
        pitch = np.array([[0.0, 12.0], [4.0, 15.0], [7.0, 19.0]])
        time = np.array([[0.0, 1.0]])
        dens = mpt.build_exp_tens(
            [pitch, time], None,
            [20.0, 0.1], [2, 1], None,
            [True, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )
        with pytest.raises(ValueError, match="grid_limit"):
            mpt.entropy_exp_tens(
                dens,
                x_min=0.0, x_max=2.0,
                n_points_per_dim=20000,
                grid_limit=int(1e6),
            )

    def test_ma_entropy_missing_bounds_errors(self):
        """Non-periodic group without bounds raises ValueError."""
        p = np.array([0.0, 4.0, 7.0])
        with pytest.raises(ValueError, match="non-periodic"):
            mpt.entropy_exp_tens(
                [p.reshape(3, 1)], None,
                [10.0], [1], None, [False], [False], [0.0],
                n_points_per_dim=100,
            )

    def test_ma_entropy_per_group_bounds(self):
        """Length-G x_min/x_max vectors are accepted, with periodic
        group entries ignored."""
        pitch = np.array([[0.0, 12.0], [4.0, 15.0], [7.0, 19.0]])
        time = np.array([[0.0, 1.0]])
        H_scalar = mpt.entropy_exp_tens(
            [pitch, time], None,
            [20.0, 0.1], [2, 1], None,
            [True, False], [True, False], [1200.0, 0.0],
            x_min=-0.5, x_max=1.5,
            n_points_per_dim=60,
        )
        # Length-G vector with NaN for the periodic group — same result.
        H_vec = mpt.entropy_exp_tens(
            [pitch, time], None,
            [20.0, 0.1], [2, 1], None,
            [True, False], [True, False], [1200.0, 0.0],
            x_min=[float("nan"), -0.5],
            x_max=[float("nan"),  1.5],
            n_points_per_dim=60,
        )
        np.testing.assert_allclose(H_scalar, H_vec, rtol=1e-12, atol=1e-12)

    # --- differenceEvents -------------------------------------------

    def test_diff_order_0_identity(self):
        """Order 0 returns the input unchanged."""
        p = [np.array([[0.0, 2.0, 5.0, 7.0]])]
        pd, wd = mpt.difference_events(p, None, None, [0], [12.0])
        np.testing.assert_array_equal(pd[0], p[0])
        assert wd is None

    def test_diff_order_1_nonperiodic(self):
        """Order-1 differencing produces pairwise inter-event differences."""
        p = [np.array([[0.0, 2.0, 5.0, 7.0]])]
        pd, _ = mpt.difference_events(p, None, None, [1], [0.0])
        np.testing.assert_allclose(pd[0], [[2.0, 3.0, 2.0]])

    def test_diff_order_1_periodic_wrap(self):
        """Periodic wrapping maps large positive differences to negative
        shortest-arc values."""
        p = [np.array([[0.0, 11.0]])]  # diff = 11, wraps to -1 under P=12
        pd, _ = mpt.difference_events(p, None, None, [1], [12.0])
        np.testing.assert_allclose(pd[0], [[-1.0]])

    def test_diff_order_2(self):
        """Order 2 applies first-order differencing twice."""
        p = [np.array([[0.0, 2.0, 5.0, 7.0]])]
        pd, _ = mpt.difference_events(p, None, None, [2], [0.0])
        # 1st: [2, 3, 2]; 2nd: [1, -1]
        np.testing.assert_allclose(pd[0], [[1.0, -1.0]])

    def test_diff_weight_rolling_product_order_1(self):
        """Order-1 weights are a rolling product of width 2."""
        p = [np.array([[0.0, 2.0, 5.0, 7.0]])]
        w = [np.array([[0.5, 0.8, 1.0, 0.2]])]
        _, wd = mpt.difference_events(p, w, None, [1], [0.0])
        np.testing.assert_allclose(wd[0], [[0.5*0.8, 0.8*1.0, 1.0*0.2]])

    def test_diff_weight_rolling_product_order_2(self):
        """Order-2 weights are a rolling product of width 3 (each
        constituent weight appears exactly once per output column)."""
        p = [np.array([[0.0, 1.0, 3.0, 6.0]])]
        w = [np.array([[0.5, 0.8, 1.0, 0.2]])]
        _, wd = mpt.difference_events(p, w, None, [2], [0.0])
        np.testing.assert_allclose(
            wd[0], [[0.5*0.8*1.0, 0.8*1.0*0.2]]
        )

    def test_diff_weight_scalar_raised_to_power(self):
        """A top-level scalar c with uniform order k returns
        scalar c**(k+1), so scalar and vector-of-c inputs produce
        equivalent downstream densities."""
        p = [np.array([[0.0, 2.0, 5.0]])]
        _, wd = mpt.difference_events(p, 0.7, None, [1], [0.0])
        assert wd == pytest.approx(0.7 ** 2)

    def test_diff_mixed_orders_align(self):
        """Lower-order groups have leading events dropped to align with
        the highest-order group."""
        p = [np.array([[0.0, 2.0, 5.0, 7.0]]),
             np.array([[0.0, 1.0, 2.0, 3.5]])]
        pd, _ = mpt.difference_events(p, None, None, [0, 1], [0.0, 0.0])
        # Group 0 (k=0): max_order - k = 1 leading event dropped.
        np.testing.assert_allclose(pd[0], [[2.0, 5.0, 7.0]])
        # Group 1 (k=1): differenced, no further drop.
        np.testing.assert_allclose(pd[1], [[1.0, 1.0, 1.5]])
        # Both outputs have N' = 3.
        assert pd[0].shape[1] == 3 and pd[1].shape[1] == 3

    def test_diff_grouped_attrs_share_order(self):
        """Two attributes in one group receive the same differencing."""
        p = [np.array([[0.0, 1.0, 3.0]]), np.array([[10.0, 12.0, 16.0]])]
        groups = [0, 0]  # both in group 0 (0-indexed, Python)
        pd, _ = mpt.difference_events(p, None, groups, [1], [0.0])
        np.testing.assert_allclose(pd[0], [[1.0, 2.0]])
        np.testing.assert_allclose(pd[1], [[2.0, 4.0]])

    def test_diff_feeds_build_exp_tens(self):
        """Output of differenceEvents feeds directly into build_exp_tens."""
        p = [np.array([[0.0, 2.0, 5.0, 7.0]]),
             np.array([[0.0, 0.5, 1.2, 1.7]])]
        pd, wd = mpt.difference_events(
            p, None, None, [0, 1], [1200.0, 0.0],
        )
        dens = mpt.build_exp_tens(
            pd, wd, [10.0, 0.05], [1, 1], None,
            [False, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )
        assert dens.tag == "MaetDensity"
        assert dens.n == 3  # N' after max_order=1 drop

    def test_diff_too_high_order_errors(self):
        """Differencing order > N - 1 raises."""
        p = [np.array([[0.0, 2.0, 5.0]])]
        with pytest.raises(ValueError, match="too high"):
            mpt.difference_events(p, None, None, [3], [0.0])

    def test_diff_negative_order_errors(self):
        """Negative differencing order raises."""
        p = [np.array([[0.0, 2.0, 5.0]])]
        with pytest.raises(ValueError, match="non-negative"):
            mpt.difference_events(p, None, None, [-1], [0.0])

    def test_diff_wrong_length_diff_orders(self):
        """diff_orders length mismatch raises."""
        p = [np.array([[0.0, 2.0, 5.0]])]
        with pytest.raises(ValueError, match="diff_orders"):
            mpt.difference_events(p, None, None, [1, 1], [0.0, 0.0])

    def test_diff_mismatched_event_counts(self):
        """Attributes with different N raise."""
        p = [np.array([[0.0, 2.0, 5.0]]), np.array([[0.0, 1.0]])]
        with pytest.raises(ValueError, match="event count"):
            mpt.difference_events(p, None, None, [0, 0], [0.0, 0.0])

    def test_diff_multi_slot_attribute_errors(self):
        """A K_a = 2 attribute must raise. Column-wise differencing would
        impose a cross-event slot correspondence that within-event slot
        exchangeability does not license."""
        p = [np.array([[60.0, 62.0, 64.0], [67.0, 69.0, 71.0]])]  # K_a = 2
        with pytest.raises(ValueError, match=r"K_a\s*=\s*2"):
            mpt.difference_events(p, None, None, [1], [0.0])

    def test_diff_empty_attribute_errors(self):
        """K_a = 0 (empty attribute) is likewise rejected by the
        K_a = 1 check; there is nothing to difference."""
        p = [np.zeros((0, 3))]
        with pytest.raises(ValueError, match=r"K_a\s*=\s*0"):
            mpt.difference_events(p, None, None, [1], [0.0])

    def test_diff_multi_slot_error_names_offending_attribute(self):
        """With a mix of K_a = 1 and K_a > 1 attributes, the error must
        fire and its message must name the offending attribute index."""
        p = [
            np.array([[60.0, 62.0, 64.0]]),                         # K_a = 1
            np.array([[60.0, 62.0, 64.0], [67.0, 69.0, 71.0]]),     # K_a = 2
        ]
        with pytest.raises(ValueError, match=r"Attribute 1"):
            mpt.difference_events(p, None, None, [1, 1], [0.0, 0.0])

    def test_diff_voices_as_attrs_pipeline(self):
        """Round-trip test for the voices-as-attributes pipeline: encode
        each voice as its own K_a = 1 attribute, difference each, then
        stack the differenced attributes into a single multi-slot
        attribute before build_exp_tens. Verify the resulting density
        has the expected shape and that eval_exp_tens returns finite
        non-negative values at a few query points."""
        pS = np.array([[72.0, 74.0, 76.0, 77.0]])   # soprano
        pA = np.array([[67.0, 69.0, 71.0, 72.0]])   # alto
        pT = np.array([[60.0, 62.0, 64.0, 65.0]])   # tenor
        pB = np.array([[48.0, 50.0, 52.0, 53.0]])   # bass
        p_attr = [pS, pA, pT, pB]
        groups = [0, 0, 0, 0]   # shared group (0-indexed in Python)
        p_diff, _ = mpt.difference_events(p_attr, None, groups, [1], [0.0])
        # Each differenced attribute should be 1 x 3.
        assert all(M.shape == (1, 3) for M in p_diff)
        # Stack into a single K_a = 4, N' = 3 multi-slot attribute.
        p_bundled = [np.vstack(p_diff)]
        assert p_bundled[0].shape == (4, 3)
        dens = mpt.build_exp_tens(
            p_bundled, None, [10.0], [1], None,
            [False], [False], [0.0],
            verbose=False,
        )
        assert dens.tag == "MaetDensity"
        # Finite non-negative at a few query points.
        x_query = np.array([-3.0, 0.0, 2.0, 4.0, 7.0])
        vals = mpt.eval_exp_tens(dens, x_query)
        assert np.all(np.isfinite(vals))
        assert np.all(vals >= 0.0)

    # --- windowTensor / windowedSimilarity ------------------------------

    def _make_time_pitch_dens(self, events):
        """Build a pitch+time density from a list of (pitch, time) tuples."""
        pitches = np.array([[p for (p, _) in events]])
        times   = np.array([[t for (_, t) in events]])
        return mpt.build_exp_tens(
            [pitches, times], None,
            [10.0, 0.1], [1, 1], None,
            [False, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )

    def test_window_tensor_returns_tagged_object(self):
        """window_tensor returns a WindowedMaetDensity with the right tag."""
        dens = self._make_time_pitch_dens([(60, 0), (62, 1)])
        spec = {"size": [np.inf, 1.0], "mix": [0.0, 0.0],
                "centre": [np.zeros(1), np.array([0.5])]}
        wmd = mpt.window_tensor(dens, spec)
        assert isinstance(wmd, mpt.WindowedMaetDensity)
        assert wmd.tag == "WindowedMaetDensity"

    def test_window_infinite_size_equals_unwindowed(self):
        """A sufficiently wide window, centred at the context's mean
        time, gives cos_sim ~= 1.0 vs the unwindowed self.

        Under cross-correlation semantics the query is translated so
        that its effective-space mean moves onto the window centre, so
        a centred window at the context's mean position is the correct
        analogue of the no-window case.
        """
        events = [(60, 0), (62, 1), (64, 2), (65, 3)]
        dens = self._make_time_pitch_dens(events)
        s_self = mpt.cos_sim_exp_tens(dens, dens, verbose=False)
        t_mean = float(np.mean([t for (_, t) in events]))
        spec = {"size": [np.inf, 1e6], "mix": [0.0, 0.0],
                "centre": [np.zeros(1), np.array([t_mean])]}
        wmd = mpt.window_tensor(dens, spec)
        s_wide = mpt.cos_sim_exp_tens(dens, wmd, verbose=False)
        np.testing.assert_allclose(s_wide, s_self, rtol=1e-3, atol=1e-3)

    def test_window_no_groups_spec_means_identity(self):
        """All groups with size=inf means no window is effectively applied."""
        dens = self._make_time_pitch_dens([(60, 0), (62, 1)])
        spec = {"size": [np.inf, np.inf], "mix": [0.0, 0.0]}
        wmd = mpt.window_tensor(dens, spec)
        s = mpt.cos_sim_exp_tens(dens, wmd, verbose=False)
        # Should equal self-similarity (no windowing means identity).
        np.testing.assert_allclose(s, 1.0, atol=1e-6)

    def test_window_narrow_reduces_cos_sim(self):
        """A very narrow window gives cos_sim much less than 1.0."""
        dens = self._make_time_pitch_dens(
            [(60, 0), (62, 1), (64, 2), (65, 3)]
        )
        spec = {"size": [np.inf, 0.2], "mix": [0.0, 0.0],
                "centre": [np.zeros(1), np.array([0.0])]}
        wmd = mpt.window_tensor(dens, spec)
        s = mpt.cos_sim_exp_tens(dens, wmd, verbose=False)
        # Should be small — only ~1 event in window out of 4.
        assert s < 0.5

    def test_window_rectangular_1d_time(self):
        """Rectangular window (mix=1) on 1-D time works."""
        dens = self._make_time_pitch_dens([(60, 0), (62, 1), (64, 2)])
        spec = {"size": [np.inf, 0.5], "mix": [0.0, 1.0],
                "centre": [np.zeros(1), np.array([1.0])]}
        wmd = mpt.window_tensor(dens, spec)
        # Should run without error and give a finite value.
        s = mpt.cos_sim_exp_tens(dens, wmd, verbose=False)
        assert np.isfinite(s)
        assert 0 < s < 1

    def test_window_raised_rect_1d_time(self):
        """Raised-rectangular window (0 < mix < 1) on 1-D time works."""
        dens = self._make_time_pitch_dens([(60, 0), (62, 1), (64, 2)])
        spec = {"size": [np.inf, 0.5], "mix": [0.0, 0.5],
                "centre": [np.zeros(1), np.array([1.0])]}
        wmd = mpt.window_tensor(dens, spec)
        s = mpt.cos_sim_exp_tens(dens, wmd, verbose=False)
        assert np.isfinite(s)
        assert 0 < s < 1

    def test_window_multi_d_rel_gaussian_works(self):
        """Multi-D relative group with Gaussian window (mix=0) works."""
        pitch = np.array([[60.0, 62.0], [64.0, 65.0], [67.0, 69.0]])
        dens = mpt.build_exp_tens(
            [pitch], None,
            [10.0], [3], None,
            [True], [True], [1200.0],
            verbose=False,
        )
        spec = {"size": [1.0], "mix": [0.0],
                "centre": [np.array([50.0, 100.0])]}
        wmd = mpt.window_tensor(dens, spec)
        s = mpt.cos_sim_exp_tens(dens, wmd, verbose=False)
        assert np.isfinite(s)
        assert 0 <= s <= 1

    def test_window_multi_d_rel_rect_raises(self):
        """Multi-D relative group with rect (mix=1) raises
        NotImplementedError."""
        pitch = np.array([[60.0, 62.0], [64.0, 65.0], [67.0, 69.0]])
        dens = mpt.build_exp_tens(
            [pitch], None,
            [10.0], [3], None,
            [True], [True], [1200.0],
            verbose=False,
        )
        spec = {"size": [1.0], "mix": [1.0],
                "centre": [np.array([50.0, 100.0])]}
        wmd = mpt.window_tensor(dens, spec)
        with pytest.raises(NotImplementedError, match="Multi-D relative"):
            mpt.cos_sim_exp_tens(dens, wmd, verbose=False)

    def test_window_multi_d_rel_raised_rect_raises(self):
        """Multi-D relative with raised-rect also raises."""
        pitch = np.array([[60.0, 62.0], [64.0, 65.0], [67.0, 69.0]])
        dens = mpt.build_exp_tens(
            [pitch], None,
            [10.0], [3], None,
            [True], [True], [1200.0],
            verbose=False,
        )
        spec = {"size": [1.0], "mix": [0.5],
                "centre": [np.array([50.0, 100.0])]}
        wmd = mpt.window_tensor(dens, spec)
        with pytest.raises(NotImplementedError, match="Multi-D relative"):
            mpt.cos_sim_exp_tens(dens, wmd, verbose=False)

    def test_window_entropy_works_on_rect(self):
        """entropy_exp_tens works on a rectangular-windowed density,
        even on a multi-D relative group (evaluation is pointwise)."""
        pitch = np.array([[60.0, 62.0], [64.0, 65.0], [67.0, 69.0]])
        dens = mpt.build_exp_tens(
            [pitch], None,
            [10.0], [3], None,
            [True], [True], [1200.0],
            verbose=False,
        )
        spec = {"size": [1.0], "mix": [1.0],
                "centre": [np.array([50.0, 100.0])]}
        wmd = mpt.window_tensor(dens, spec)
        # dim = 2, grid 20x20 = 400 points
        H = mpt.entropy_exp_tens(wmd, n_points_per_dim=20)
        assert np.isfinite(H)

    def test_window_entropy_narrower_lower(self):
        """Narrower window gives lower entropy on a time-windowed density."""
        dens = self._make_time_pitch_dens(
            [(60, 0), (62, 1), (64, 2), (65, 3)]
        )
        H_base = mpt.entropy_exp_tens(
            dens, x_min=[0.0, -1.0], x_max=[1200.0, 4.0],
            n_points_per_dim=60,
        )
        spec_narrow = {"size": [np.inf, 0.3], "mix": [0.0, 0.0],
                       "centre": [np.zeros(1), np.array([1.0])]}
        wmd = mpt.window_tensor(dens, spec_narrow)
        H_narrow = mpt.entropy_exp_tens(
            wmd, x_min=[0.0, -1.0], x_max=[1200.0, 4.0],
            n_points_per_dim=60,
        )
        assert H_narrow < H_base

    def test_windowed_similarity_profile(self):
        """windowed_similarity returns a length-M profile that peaks at the
        offset where the context has a pitch matching the query."""
        ctx = mpt.build_exp_tens(
            [np.array([[60.0, 62.0, 64.0, 65.0]]),
             np.array([[0.0, 1.0, 2.0, 3.0]])],
            None, [0.5, 0.1], [1, 1], None,
            [False, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )
        # Query: single event at pitch 62, time 0 (fixed, not swept).
        q = mpt.build_exp_tens(
            [np.array([[62.0]]), np.array([[0.0]])], None,
            [0.5, 0.1], [1, 1], None,
            [False, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )
        # Sweep offsets in time. Query-centroid is at t=0, so an offset
        # of t means the window sits at absolute context time t. The
        # pitch-62 context event lives at t=1.
        M = 21
        offs = np.linspace(-0.5, 3.5, M)
        offsets = np.zeros((2, M))
        offsets[1, :] = offs
        spec = {"size": [np.inf, 0.3], "mix": [0.0, 0.0]}
        profile = mpt.windowed_similarity(q, ctx, spec, offsets, verbose=False)
        peak_idx = np.argmax(profile)
        assert abs(offs[peak_idx] - 1.0) < 0.3

    def test_windowed_similarity_vector_length(self):
        """windowed_similarity output has length M."""
        dens = self._make_time_pitch_dens([(60, 0), (62, 1)])
        offsets = np.zeros((2, 7))
        offsets[1, :] = np.linspace(0, 1, 7)
        spec = {"size": [np.inf, 0.5], "mix": [0.0, 0.0]}
        profile = mpt.windowed_similarity(dens, dens, spec, offsets,
                                         verbose=False)
        assert profile.shape == (7,)

    def test_windowed_similarity_reference_none_equals_default(self):
        """reference=None reproduces the default unweighted-centroid path.

        Explicit None must give identical numerical output to the call
        without the keyword. This guards backward compatibility when
        the reference= keyword is added to the API.
        """
        ctx = self._make_time_pitch_dens(
            [(60, 0), (62, 1), (64, 2), (65, 3)])
        q = self._make_time_pitch_dens([(62, 0)])
        M = 11
        offsets = np.zeros((2, M))
        offsets[1, :] = np.linspace(-0.5, 3.5, M)
        spec = {"size": [np.inf, 0.3], "mix": [0.0, 0.0]}
        prof_default  = mpt.windowed_similarity(q, ctx, spec, offsets,
                                             verbose=False)
        prof_explicit = mpt.windowed_similarity(q, ctx, spec, offsets,
                                             reference=None,
                                             verbose=False)
        assert np.allclose(prof_default, prof_explicit)

    def test_windowed_similarity_reference_shifts_profile(self):
        """A user-supplied reference shifts the profile by exactly the
        offset between the new reference and the default (unweighted
        centroid), over the span of the sweep.

        Concretely: calling windowed_similarity with reference = mu_default
        + shift should produce the same profile values at every sweep
        column as the default call shifted by -shift in offset space.
        """
        ctx = self._make_time_pitch_dens(
            [(60, 0), (62, 1), (64, 2), (65, 3)])
        q = self._make_time_pitch_dens([(62, 0), (64, 0.5)])
        # Default reference (unweighted centroid per attribute).
        mu_default = [q.centres[a].mean(axis=1) for a in range(q.n_attrs)]
        shift = np.array([0.0])    # time-attribute shift only
        # Shift the time reference by +0.2 s. (Pitch reference unchanged.)
        ref_shifted = [mu_default[0].copy(),
                       mu_default[1] + 0.2]
        # Sweep only in time.
        M = 21
        offs_time = np.linspace(-1.0, 3.0, M)
        offsets = np.zeros((2, M))
        offsets[1, :] = offs_time
        spec = {"size": [np.inf, 0.3], "mix": [0.0, 0.0]}
        prof_default = mpt.windowed_similarity(q, ctx, spec, offsets,
                                            verbose=False)
        prof_shifted = mpt.windowed_similarity(q, ctx, spec, offsets,
                                            reference=ref_shifted,
                                            verbose=False)
        # At sweep column m (offset o), the default places the window
        # at mu_default + o; the shifted call places it at mu_default +
        # 0.2 + o. So prof_shifted at offset o equals prof_default at
        # offset o + 0.2. Check this for interior indices.
        for m in range(M):
            target_off = offs_time[m] + 0.2
            # Find the nearest default column to target_off
            j = int(np.argmin(np.abs(offs_time - target_off)))
            if abs(offs_time[j] - target_off) < 1e-9:
                assert abs(prof_shifted[m] - prof_default[j]) < 1e-10

    def test_windowed_similarity_reference_wrong_length_raises(self):
        """reference with wrong per-attribute length raises ValueError."""
        q = self._make_time_pitch_dens([(62, 0)])
        ctx = self._make_time_pitch_dens([(60, 0), (62, 1)])
        offsets = np.zeros((2, 3))
        offsets[1, :] = np.linspace(0, 1, 3)
        spec = {"size": [np.inf, 0.3], "mix": [0.0, 0.0]}
        # Too few entries (1 instead of 2)
        with pytest.raises(ValueError, match="reference"):
            mpt.windowed_similarity(q, ctx, spec, offsets,
                                  reference=[np.array([0.0])],
                                  verbose=False)
        # Correct number of entries but wrong inner length
        with pytest.raises(ValueError, match="reference"):
            mpt.windowed_similarity(q, ctx, spec, offsets,
                                  reference=[np.array([0.0, 0.0]),
                                             np.array([0.0])],
                                  verbose=False)

    # ---- Periodic-window warning ------------------------------------
    #
    # The line-case closed form used downstream is exact for non-periodic
    # groups and only approximate for periodic groups (it retains only
    # the leading periodic image of the window). A single warning,
    # WindowedSimilarityPeriodicApproxWarning, is emitted on every call
    # involving a windowed periodic group, with two message forms:
    #
    #   - Within the recommended bound (lambda*sigma <= P/(2*sqrt(3))):
    #     a brief informational form.
    #   - Past the bound: a stronger form with phi reported and
    #     per-mix behaviour described.
    #
    # The pitch group of _make_time_pitch_dens has sigma = 10 cents and
    # period = 1200 cents, so the bound lambda*sigma > P/(2*sqrt(3)) ~=
    # 346.4 cents corresponds to size > 34.64. See manuscript §5.2
    # Remark 5.2 and User Guide §3.1 "Post-tensor windowing".

    def test_windowed_similarity_periodic_warning_always_fires(self):
        """The warning fires on every call involving a windowed
        periodic group, regardless of window size. Tested with a tiny
        window (size = 5) well within the bound."""
        q   = self._make_time_pitch_dens([(60, 0)])
        ctx = self._make_time_pitch_dens([(60, 0), (62, 1)])
        offsets = np.zeros((2, 3))
        offsets[1, :] = np.linspace(0, 1, 3)
        spec_small = {"size": [5.0, 0.3], "mix": [0.0, 0.0]}
        with pytest.warns(mpt.WindowedSimilarityPeriodicApproxWarning,
                          match="periodic"):
            _ = mpt.windowed_similarity(q, ctx, spec_small, offsets,
                                        verbose=False)

    def test_windowed_similarity_periodic_warning_within_bound_message(
            self):
        """Within the bound, the message takes the brief informational
        form. Detected by absence of the past-bound marker phrase."""
        q   = self._make_time_pitch_dens([(60, 0)])
        ctx = self._make_time_pitch_dens([(60, 0), (62, 1)])
        offsets = np.zeros((2, 3))
        offsets[1, :] = np.linspace(0, 1, 3)
        spec_small = {"size": [5.0, 0.3], "mix": [0.0, 0.0]}
        with warnings.catch_warnings(record=True) as caught:
            warnings.filterwarnings(
                "always",
                category=mpt.WindowedSimilarityPeriodicApproxWarning,
            )
            _ = mpt.windowed_similarity(q, ctx, spec_small, offsets,
                                        verbose=False)
        msgs = [str(w.message) for w in caught
                if issubclass(
                    w.category,
                    mpt.WindowedSimilarityPeriodicApproxWarning)]
        assert len(msgs) == 1, f"Expected 1 warning, got {len(msgs)}."
        assert "exceeds the recommended bound" not in msgs[0]
        assert "approximation is sub-percent" in msgs[0]

    def test_windowed_similarity_periodic_warning_past_bound_message(
            self):
        """Past the bound, the message takes the stronger form with
        phi and per-mix behaviour. Detected by the past-bound marker
        phrase. With sigma = 10 and P = 1200, the bound is at
        size ~= 34.64; size = 40 is past it."""
        q   = self._make_time_pitch_dens([(60, 0)])
        ctx = self._make_time_pitch_dens([(60, 0), (62, 1)])
        offsets = np.zeros((2, 3))
        offsets[1, :] = np.linspace(0, 1, 3)
        spec_offending = {"size": [40.0, 0.3], "mix": [0.0, 0.0]}
        with warnings.catch_warnings(record=True) as caught:
            warnings.filterwarnings(
                "always",
                category=mpt.WindowedSimilarityPeriodicApproxWarning,
            )
            _ = mpt.windowed_similarity(q, ctx, spec_offending, offsets,
                                        verbose=False)
        msgs = [str(w.message) for w in caught
                if issubclass(
                    w.category,
                    mpt.WindowedSimilarityPeriodicApproxWarning)]
        assert len(msgs) == 1, f"Expected 1 warning, got {len(msgs)}."
        assert "exceeds the recommended bound" in msgs[0]
        assert "phi (rect half-width)" in msgs[0]

    def test_windowed_similarity_periodic_warning_silent_for_aperiodic_group(
            self):
        """A non-periodic group never triggers the warning, even when
        the windowed group's lambda*sigma is huge.

        The time group has is_per=False, so applying a very wide
        time-only window must not trigger the periodic warning.
        (The pitch group is left unwindowed, size=inf.)
        """
        q   = self._make_time_pitch_dens([(60, 0)])
        ctx = self._make_time_pitch_dens([(60, 0), (62, 1)])
        offsets = np.zeros((2, 3))
        offsets[1, :] = np.linspace(0, 1, 3)
        spec = {"size": [np.inf, 1e6], "mix": [0.0, 0.0]}
        with warnings.catch_warnings():
            warnings.simplefilter(
                "error",
                category=mpt.WindowedSimilarityPeriodicApproxWarning,
            )
            _ = mpt.windowed_similarity(q, ctx, spec, offsets,
                                        verbose=False)

    def test_windowed_similarity_periodic_warning_fires_every_call(self):
        """The warning is registered with an 'always' filter so it
        fires on every call, not just the first.

        Matches the MATLAB warning(id, ...) per-call behaviour.
        """
        q   = self._make_time_pitch_dens([(60, 0)])
        ctx = self._make_time_pitch_dens([(60, 0), (62, 1)])
        offsets = np.zeros((2, 3))
        offsets[1, :] = np.linspace(0, 1, 3)
        spec_offending = {"size": [40.0, 0.3], "mix": [0.0, 0.0]}
        with warnings.catch_warnings(record=True) as caught:
            warnings.filterwarnings(
                "always",
                category=mpt.WindowedSimilarityPeriodicApproxWarning,
            )
            _ = mpt.windowed_similarity(q, ctx, spec_offending, offsets,
                                        verbose=False)
            _ = mpt.windowed_similarity(q, ctx, spec_offending, offsets,
                                        verbose=False)
            n_warns = sum(
                1 for w in caught
                if issubclass(
                    w.category,
                    mpt.WindowedSimilarityPeriodicApproxWarning,
                )
            )
        assert n_warns >= 2, (
            f"Expected >= 2 periodic warnings across two calls; "
            f"got {n_warns}."
        )

    def test_window_tensor_validates_shapes(self):
        """window_tensor rejects invalid size / mix / centre shapes."""
        dens = self._make_time_pitch_dens([(60, 0), (62, 1)])
        # Wrong size length
        with pytest.raises(ValueError, match="size"):
            mpt.window_tensor(dens, {"size": [1.0, 1.0, 1.0],
                                     "mix": [0.0, 0.0]})
        # Mix out of range
        with pytest.raises(ValueError, match="mix"):
            mpt.window_tensor(dens, {"size": [1.0, 1.0],
                                     "mix": [0.0, 1.5]})
        # Wrong centre length in list form
        with pytest.raises(ValueError, match="centre"):
            mpt.window_tensor(dens, {
                "size": [1.0, 1.0], "mix": [0.0, 0.0],
                "centre": [np.array([0.0, 0.0])],  # length 1 list, need A=2
            })


class TestWindowedCrossCorrelation:
    """Cross-correlation semantics of ``windowed_similarity`` (offset API).

    These tests exercise the coordinate substitution that places the
    query's effective-space centroid onto the window centre at each
    sweep position, so that a peak at offset ``o`` means the query
    pattern is present in the context displaced by ``o`` from its own
    centroid. The unwindowed path (``cos_sim_exp_tens``) is also
    covered, to confirm the patch does not perturb norms.
    """

    # ---- helpers ----------------------------------------------------

    @staticmethod
    def _build(events, sigma_pitch=1.0, sigma_time=0.2,
               pitch_is_rel=False, pitch_is_per=False,
               pitch_period=0.0, r_pitch=1, r_time=1):
        """Build a pitch/time MAET from a list of (pitch, time)."""
        pitches = np.array([[e[0] for e in events]], dtype=np.float64)
        times   = np.array([[e[1] for e in events]], dtype=np.float64)
        return mpt.build_exp_tens(
            [pitches, times], None,
            [sigma_pitch, sigma_time],
            [r_pitch, r_time], None,
            [pitch_is_rel, False],
            [pitch_is_per, False],
            [pitch_period, 0.0],
            verbose=False,
        )

    @staticmethod
    def _sweep(dens_q, dens_c, size_time=2.0, mix_time=0.0,
                 off_min=-2.0, off_max=12.0, n=29):
        """Run a time-offset sweep and return (offsets, profile)."""
        offs = np.linspace(off_min, off_max, n)
        offsets = np.zeros((2, n))
        offsets[1, :] = offs          # pitch offset = 0 (unwindowed anyway)
        spec = {
            "size": [np.inf, size_time],
            "mix":  [0.0, mix_time],
        }
        profile = mpt.windowed_similarity(dens_q, dens_c, spec, offsets,
                                          verbose=False)
        return offs, np.asarray(profile)

    # ---- 1. peak at the offset between query and context events ----

    def test_peak_at_single_event_context_time(self):
        """Query at t=0, context at t=5: profile peaks at offset 5."""
        q = self._build([(60.0, 0.0)])
        c = self._build([(60.0, 5.0)])
        offs, prof = self._sweep(q, c, off_min=0.0, off_max=10.0, n=41)
        peak_off = offs[np.argmax(prof)]
        assert abs(peak_off - 5.0) < 0.3
        # Also: profile must not be effectively zero everywhere (the
        # pre-fix symptom of unshifted localised cosine).
        assert prof.max() > 0.5

    # ---- 2. peak invariance over window size (Gaussian) -------------

    @pytest.mark.parametrize("size_time", [1.0, 2.0, 4.0, 8.0])
    def test_peak_invariance_over_size_gaussian(self, size_time):
        """Peak offset stays at 5 regardless of pure-Gaussian window size."""
        q = self._build([(60.0, 0.0)])
        c = self._build([(60.0, 5.0)])
        offs, prof = self._sweep(q, c, size_time=size_time, mix_time=0.0,
                                    off_min=0.0, off_max=10.0, n=41)
        peak_off = offs[np.argmax(prof)]
        assert abs(peak_off - 5.0) < 0.3

    # ---- 3. peak invariance over window mix (rect ⊛ Gaussian) -------

    @pytest.mark.parametrize("mix_time", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_peak_invariance_over_mix(self, mix_time):
        """Peak offset stays at 5 across the full mix range, from pure
        Gaussian (0) to pure rectangular (1)."""
        q = self._build([(60.0, 0.0)])
        c = self._build([(60.0, 5.0)])
        offs, prof = self._sweep(q, c, size_time=2.0, mix_time=mix_time,
                                    off_min=0.0, off_max=10.0, n=41)
        peak_off = offs[np.argmax(prof)]
        assert abs(peak_off - 5.0) < 0.3

    # ---- 4. multi-event query peak at centroid offset ---------------

    def test_multi_event_query_peak_at_centroid_offset(self):
        """Query centroid at t=0.5; context motif centroid at t=5.5.
        The expected peak offset is 5.5 - 0.5 = 5.0."""
        q = self._build([(60.0, 0.0), (64.0, 1.0)])
        c = self._build([(60.0, 5.0), (64.0, 6.0)])
        offs, prof = self._sweep(q, c, off_min=0.0, off_max=10.0, n=101)
        peak_off = offs[np.argmax(prof)]
        assert abs(peak_off - 5.0) < 0.2

    # ---- 5. two motif recurrences → two equal peaks -----------------

    def test_two_recurrences_give_two_equal_peaks(self):
        """Two copies of the query motif in context produce two peaks of
        equal height. Motif centroids at t=2.5 and t=5.5; query centroid
        at t=0.5; expected peak offsets 2.0 and 5.0."""
        q = self._build([(60.0, 0.0), (64.0, 1.0)])
        c = self._build([(60.0, 2.0), (64.0, 3.0),
                          (60.0, 5.0), (64.0, 6.0)])
        offs, prof = self._sweep(q, c, off_min=0.0, off_max=10.0, n=201)
        # Local maxima above half the global max.
        peaks = [i for i in range(1, len(offs) - 1)
                 if prof[i] > prof[i - 1]
                 and prof[i] > prof[i + 1]
                 and prof[i] > 0.5 * prof.max()]
        assert len(peaks) == 2
        peak_offs = sorted(offs[p] for p in peaks)
        assert abs(peak_offs[0] - 2.0) < 0.15
        assert abs(peak_offs[1] - 5.0) < 0.15
        # Equal height by symmetry.
        heights = [float(prof[p]) for p in peaks]
        assert abs(heights[0] - heights[1]) < 1e-3

    # ---- 6. isRel=true dyad: peak location invariant under pitch
    #         translation of the context ----------------------------

    def test_isrel_dyad_peak_invariant_under_pitch_translation(self):
        """Concurrent dyad query with isRel=True on pitch (r=2, K_a=2):
        translating the context's pitches uniformly leaves the time-peak
        location unchanged."""
        # One concurrent dyad at time 0 (query), at time 5 (context).
        p_q   = [np.array([[60.0], [64.0]]), np.array([[0.0]])]
        p_c1  = [np.array([[60.0], [64.0]]), np.array([[5.0]])]
        p_c2  = [np.array([[70.0], [74.0]]), np.array([[5.0]])]

        def _build_dyad(p_attr):
            return mpt.build_exp_tens(
                p_attr, None,
                [1.0, 0.2], [2, 1], None,
                [True, False], [False, False], [0.0, 0.0],
                verbose=False,
            )
        q  = _build_dyad(p_q)
        c1 = _build_dyad(p_c1)
        c2 = _build_dyad(p_c2)

        # Offset has dim = (r_pitch - isRel_pitch) + (r_time - isRel_time)
        # = (2 - 1) + (1 - 0) = 2 rows: one effective-pitch offset, one
        # time offset. Pitch offset is ignored (pitch group unwindowed).
        offs = np.linspace(0.0, 10.0, 101)
        offsets = np.zeros((2, len(offs)))
        offsets[1, :] = offs
        spec = {
            "size": [np.inf, 2.0],
            "mix":  [0.0, 0.0],
        }
        p1 = mpt.windowed_similarity(q, c1, spec, offsets, verbose=False)
        p2 = mpt.windowed_similarity(q, c2, spec, offsets, verbose=False)
        assert np.argmax(p1) == np.argmax(p2)
        assert abs(offs[np.argmax(p1)] - 5.0) < 0.3

    # ---- 7. unwindowed cos_sim on identical densities is 1 ---------

    def test_unwindowed_similarity_identical_is_one(self):
        """The unwindowed path is untouched; self-similarity is 1."""
        d = self._build([(60.0, 0.0), (64.0, 1.0), (67.0, 2.0)])
        s = mpt.cos_sim_exp_tens(d, d, verbose=False)
        assert abs(s - 1.0) < 1e-10

    # ---- 8. unwindowed cos_sim of distinct MA densities stays in the
    #         expected range ---------------------------------------

    def test_unwindowed_similarity_distinct_densities(self):
        """Two similar but not identical densities give cos_sim ∈ (0, 1)."""
        d1 = self._build([(60.0, 0.0), (64.0, 1.0), (67.0, 2.0)])
        d2 = self._build([(60.0, 0.0), (65.0, 1.0), (67.0, 2.0)])
        s = mpt.cos_sim_exp_tens(d1, d2, verbose=False)
        assert 0.0 < s < 1.0

    # ---- 9. pitch mismatch suppresses the matched-time peak ---------

    def test_pitch_mismatch_suppresses_peak(self):
        """Cross-correlation is selective: a pitch-mismatched event at
        the same offset produces a far weaker peak than a matching
        pitch."""
        q       = self._build([(60.0, 0.0)])
        c_match = self._build([(60.0, 5.0)])
        c_miss  = self._build([(72.0, 5.0)])
        _, p_match = self._sweep(q, c_match, off_min=0.0, off_max=10.0, n=41)
        _, p_miss  = self._sweep(q, c_miss,  off_min=0.0, off_max=10.0, n=41)
        assert p_miss.max() < 0.01 * p_match.max()

    # ---- 10. larger window size → larger peak height --------------

    def test_peak_height_increases_with_window_size(self):
        """Under a pure-Gaussian window, the peak-height factor b/σ_t is
        monotonically increasing in window size and approaches 1 as
        size → ∞ (recovering the unwindowed self-similarity value)."""
        q = self._build([(60.0, 0.0)])
        c = self._build([(60.0, 5.0)])
        peaks = []
        for s in [1.0, 2.0, 8.0, 100.0]:
            _, prof = self._sweep(q, c, size_time=s, mix_time=0.0,
                                    off_min=4.0, off_max=6.0, n=201)
            peaks.append(float(prof.max()))
        # Strict monotone increase.
        assert peaks[0] < peaks[1] < peaks[2] < peaks[3]
        # Unwindowed self-similarity recovered at very wide window.
        assert peaks[3] > 0.99


# ===================================================================
#  simplex_vertices
# ===================================================================


class TestSimplexVertices:
    def test_shapes(self):
        for N in [2, 3, 4, 5, 10]:
            V = mpt.simplex_vertices(N)
            assert V.shape == (N, N - 1)

    def test_centroid_at_origin(self):
        for N in [2, 3, 4, 5, 7]:
            V = mpt.simplex_vertices(N)
            np.testing.assert_allclose(
                V.mean(axis=0), np.zeros(N - 1), atol=1e-12
            )

    def test_edge_length_default_unit(self):
        for N in [2, 3, 4, 5, 7]:
            V = mpt.simplex_vertices(N)
            # All pairwise distances should equal 1.
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.linalg.norm(V[i] - V[j])
                    assert d == pytest.approx(1.0, abs=1e-12)

    def test_edge_length_custom(self):
        for L in [0.5, 2.0, 100.0]:
            V = mpt.simplex_vertices(4, edge_length=L)
            d = np.linalg.norm(V[0] - V[1])
            assert d == pytest.approx(L, abs=1e-12)

    def test_n_2_collapses_to_1d(self):
        V = mpt.simplex_vertices(2)
        # Two points at +/- 0.5 (so distance 1).
        assert V.shape == (2, 1)
        assert abs(V[0, 0] - V[1, 0]) == pytest.approx(1.0, abs=1e-12)
        assert V.mean() == pytest.approx(0.0, abs=1e-12)

    def test_n_too_small(self):
        with pytest.raises(ValueError, match="N must be"):
            mpt.simplex_vertices(1)
        with pytest.raises(ValueError, match="N must be"):
            mpt.simplex_vertices(0)

    def test_negative_edge_length(self):
        with pytest.raises(ValueError, match="edge_length"):
            mpt.simplex_vertices(3, edge_length=-1)

    def test_zero_edge_length(self):
        with pytest.raises(ValueError, match="edge_length"):
            mpt.simplex_vertices(3, edge_length=0)
