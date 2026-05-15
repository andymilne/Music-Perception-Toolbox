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
