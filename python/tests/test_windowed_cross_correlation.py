"""Tests for windowed_similarity cross-correlation semantics (offset API).

Mirror of MATLAB tests/test_windowed_cross_correlation.m.
"""
import numpy as np
import pytest
import warnings

import mpt
from mpt._utils import position_variance


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
            [r_pitch, r_time], 
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
        profile = mpt.windowed_similarity(dens_c, dens_q, spec, offsets,
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
                [1.0, 0.2], [2, 1], 
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
        p1 = mpt.windowed_similarity(c1, q, spec, offsets, verbose=False)
        p2 = mpt.windowed_similarity(c2, q, spec, offsets, verbose=False)
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
