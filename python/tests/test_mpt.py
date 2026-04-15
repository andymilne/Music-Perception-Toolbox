"""Test suite for the Music Perception Toolbox (mpt).

Run with: pytest tests/test_mpt.py -v
"""

import numpy as np
import pytest

import mpt


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

    def test_proj_centroid_balanced(self):
        y, cm, cp = mpt.proj_centroid([0, 400, 800], None, 1200)
        assert cm == pytest.approx(0.0, abs=1e-10)
        np.testing.assert_allclose(y, 0, atol=1e-10)

    def test_mean_offset_shape(self):
        h = mpt.mean_offset([0, 2, 4, 5, 7, 9, 11], None, 12)
        assert h.shape == (12,)

    def test_circ_apm_shape(self):
        R, rp, rl = mpt.circ_apm([0, 3, 6, 10, 12], None, 16)
        assert R.shape == (16, 16)
        assert rp.shape == (16,)
        assert rl.shape == (16,)

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

    def test_tensor_harmonicity_unison_high(self):
        spec = ["harmonic", 12, "powerlaw", 1]
        h_uni = mpt.tensor_harmonicity([0, 0], None, 12, spectrum=spec)
        h_tri = mpt.tensor_harmonicity([0, 600], None, 12, spectrum=spec)
        assert h_uni > h_tri

    def test_virtual_pitches_shape(self):
        vp_p, vp_w = mpt.virtual_pitches([0, 400, 700], None, 12)
        assert len(vp_p) == len(vp_w)
        assert len(vp_p) > 0


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
