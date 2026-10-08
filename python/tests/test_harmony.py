"""Tests for tensor_harmonicity, template_harmonicity, virtual_pitches, spectral_entropy.

Mirror of MATLAB tests/test_harmony.m.
"""
import numpy as np
import pytest
import warnings

import mpt
from mpt._utils import position_variance


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
        assert np.isfinite(h_ent)

    def test_template_harmonicity_entropy_methods(self):
        """'differential' (default) is the discrete entropy plus
        log_b(resolution); 'normalized' divides the discrete entropy by
        log_b(N) and, on pitch, warns that N depends on the span."""
        chord = [0, 400, 700]
        _, h_d = mpt.template_harmonicity(chord, None, 12, resolution=2.0,
                                          verbose=False)
        _, h_s = mpt.template_harmonicity(chord, None, 12, resolution=2.0,
                                          method="shannon", verbose=False)
        assert h_d == pytest.approx(h_s + np.log2(2.0), abs=1e-12)
        with pytest.warns(UserWarning, match="normalized"):
            _, h_n = mpt.template_harmonicity(chord, None, 12,
                                              method="normalized",
                                              verbose=False)
        assert 0 < h_n <= 1
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mpt.template_harmonicity(chord, None, 12, method="normalized",
                                     per=True, verbose=False)
        with pytest.raises(TypeError, match="normalize"):
            mpt.template_harmonicity(chord, None, 12, normalize=True)

    def test_template_harmonicity_periodic_is_octave_invariant(self):
        """With per=True, moving a note by an octave changes nothing."""
        spec = ["harmonic", 12, "powerlaw", 1]
        a = mpt.template_harmonicity([0, 400, 700], None, 12, per=True,
                                     chord_spectrum=spec, verbose=False)
        b = mpt.template_harmonicity([0, 1600, 700], None, 12, per=True,
                                     chord_spectrum=spec, verbose=False)
        np.testing.assert_allclose(a, b, rtol=1e-9)
        c = mpt.template_harmonicity([0, 1600, 700], None, 12,
                                     chord_spectrum=spec, verbose=False)
        assert abs(c[0] - a[0]) > 1e-3

    def test_virtual_pitches_periodic_peaks_on_pitch_classes(self):
        """With per=True the profile covers one period in ascending
        pitch class, and a pure-tone chord's three strongest virtual
        pitch classes lie at its notes' pitch classes (within a few
        cents: on the circle, other partials' near-coincidences pull
        each peak slightly)."""
        chord = np.array([6000.0, 6400.0, 6700.0])
        vp_p, vp_w = mpt.virtual_pitches(chord, None, 12, per=True,
                                         verbose=False)
        assert len(vp_p) == 1200
        assert np.all(np.diff(vp_p) > 0) and vp_p[0] == 0.0
        n = len(vp_w)
        peaks = [j for j in range(n)
                 if vp_w[j] >= vp_w[j - 1] and vp_w[j] > vp_w[(j + 1) % n]]
        top = sorted(peaks, key=lambda j: -vp_w[j])[:3]
        np.testing.assert_allclose(np.sort(vp_p[top]), [0.0, 400.0, 700.0],
                                   atol=3.0)

    def test_periodic_batched_matches_scalar(self):
        """With per=True, batched rows equal scalar calls, including two
        rotations of one pitch-class set (which share a periodic
        canonical key but not a lowest pitch)."""
        spec = ["harmonic", 12, "powerlaw", 1]
        P = np.array([[0, 400, 700], [400, 700, 1200], [0, 300, 700]],
                     dtype=float)
        hb = mpt.template_harmonicity(P, None, 12, per=True,
                                      chord_spectrum=spec, verbose=False)
        vp_pb, vp_wb = mpt.virtual_pitches(P, None, 12, per=True,
                                           verbose=False)
        for i in range(3):
            hs = mpt.template_harmonicity(P[i], None, 12, per=True,
                                          chord_spectrum=spec,
                                          verbose=False)
            assert hb[0][i] == pytest.approx(hs[0], abs=1e-9)
            assert hb[1][i] == pytest.approx(hs[1], abs=1e-9)
            vp_ps, vp_ws = mpt.virtual_pitches(P[i], None, 12, per=True,
                                               verbose=False)
            np.testing.assert_array_equal(vp_pb[i], vp_ps)
            np.testing.assert_allclose(vp_wb[i], vp_ws, atol=1e-12)

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

    def test_template_harmonicity_single_tone_matches_closed_form(self):
        """A single tone against the default template (36 harmonics,
        weights 1/n) has hMax equal to the cosine of a Gaussian with
        the template's density, w_1 / sqrt(w' G w), where G holds the
        Gaussian inner products exp(-(h_i - h_j)^2 / (4 sigma^2)) of
        the partials. Both densities' lowest Gaussians must be
        evaluated whole for this to hold."""
        sigma = 12.0
        n = np.arange(1, 37)
        h = 1200.0 * np.log2(n)
        w = 1.0 / n
        G = np.exp(-(h[:, None] - h[None, :]) ** 2 / (4.0 * sigma ** 2))
        expected = w[0] / np.sqrt(w @ G @ w)
        h_max, _ = mpt.template_harmonicity([400.0], None, sigma,
                                            verbose=False)
        assert h_max == pytest.approx(expected, rel=1e-6)

    def test_virtual_pitches_peaks_fall_on_chord_notes(self):
        """For pure tones narrower than an octave, each note can be the
        template's fundamental, with no other note on a partial, so
        the three strongest peaks lie exactly on the notes and are
        equal in height, the lowest note's included."""
        chord = np.array([6000.0, 6400.0, 6700.0])
        vp_p, vp_w = mpt.virtual_pitches(chord, None, 12, verbose=False)
        vp_p = np.asarray(vp_p)
        vp_w = np.asarray(vp_w)
        peaks = [j for j in range(1, len(vp_w) - 1)
                 if vp_w[j] >= vp_w[j - 1] and vp_w[j] > vp_w[j + 1]]
        top = sorted(peaks, key=lambda j: -vp_w[j])[:3]
        np.testing.assert_allclose(np.sort(vp_p[top]), chord, atol=0.5)
        np.testing.assert_allclose(vp_w[top], vp_w[top[0]], rtol=1e-6)


# ===================================================================
#  Input validation
# ===================================================================
