"""Tests for the batched forms of roughness, continuity, and markov_s.

Each batched form must agree, row by row, with the single form, and
must merge the rows its key says are equivalent: repeated rows (in any
order) for roughness, jointly transposed rows for continuity, and
rotations for markov_s. The MATLAB twin is
``tests/test_batched_roughness_continuity_markov.m``.
"""

import numpy as np
import pytest

import mpt


SPEC = ['harmonic', 8, 'powerlaw', 1]


def _hz(cents, f0=261.63):
    return f0 * 2.0 ** (np.asarray(cents, dtype=float) / 1200.0)


# ---------------------------------------------------------------------
#  roughness
# ---------------------------------------------------------------------

class TestRoughnessBatched:

    F = np.array([_hz([0, 400, 700, np.nan]),
                  _hz([0, 300, 700, np.nan]),
                  _hz([0, 400, 700, 1000]),
                  _hz([700, 0, 400, np.nan]),      # row 0, reordered
                  [np.nan] * 4])

    def _single(self, row, **kw):
        row = row[~np.isnan(row)]
        return mpt.roughness(row, None, **kw)

    def test_matches_single_form(self):
        r = mpt.roughness(self.F)
        assert r.shape == (5,)
        for i in range(4):
            assert r[i] == pytest.approx(self._single(self.F[i]), rel=1e-12)
        assert np.isnan(r[4])

    def test_matches_single_form_with_spectrum(self):
        r = mpt.roughness(self.F, spectrum=SPEC)
        for i in range(4):
            assert r[i] == pytest.approx(
                self._single(self.F[i], spectrum=SPEC), rel=1e-12)

    def test_spectrum_equals_adding_partials_by_hand(self):
        f = _hz([0, 400, 700])
        cents = 1200 * np.log2(f)
        p, w = mpt.add_spectra(cents, None, *SPEC)
        by_hand = mpt.roughness(2 ** (np.asarray(p) / 1200), w)
        assert mpt.roughness(f, spectrum=SPEC) == pytest.approx(
            by_hand, rel=1e-12)

    def test_reordered_row_equals_original(self):
        r = mpt.roughness(self.F, spectrum=SPEC)
        assert r[3] == pytest.approx(r[0], rel=1e-12)

    def test_transpositions_differ(self):
        F = np.vstack([_hz([0, 400, 700]), _hz([500, 900, 1200])])
        r = mpt.roughness(F, spectrum=SPEC)
        assert abs(r[0] - r[1]) > 1e-6

    def test_weights_matrix_and_broadcast(self):
        F = np.vstack([_hz([0, 400, 700]), _hz([0, 300, 700])])
        w = np.array([1.0, 0.5, 0.8])
        r_b = mpt.roughness(F, w)
        r_m = mpt.roughness(F, np.tile(w, (2, 1)))
        np.testing.assert_allclose(r_b, r_m, rtol=1e-12)
        assert r_b[0] == pytest.approx(mpt.roughness(F[0], w), rel=1e-12)

    def test_average_and_p_norm_pass_through(self):
        F = np.vstack([_hz([0, 400, 700]), _hz([0, 300, 700])])
        r = mpt.roughness(F, spectrum=SPEC, average=True, p_norm=2)
        assert r[1] == pytest.approx(mpt.roughness(
            F[1], spectrum=SPEC, average=True, p_norm=2), rel=1e-12)

    def test_nonpositive_row_raises(self):
        with pytest.raises(ValueError):
            mpt.roughness(np.array([[100.0, 200.0], [0.0, 300.0]]))

    def test_bad_weight_shape_raises(self):
        with pytest.raises(ValueError):
            mpt.roughness(self.F, np.ones(3))


# ---------------------------------------------------------------------
#  continuity
# ---------------------------------------------------------------------

class TestContinuityBatched:

    SEQ = np.array([[60, 62, 64, 65, 67],
                    [60, 64, 62, 60, np.nan],
                    [65, 67, 69, 70, 72],          # row 0 up a fourth
                    [np.nan] * 5]) * 100.0
    SIGMA = 10.0

    def test_per_row_queries(self):
        x = np.array([[69], [58], [74], [60]]) * 100.0
        c, m = mpt.continuity(self.SEQ, x, self.SIGMA)
        assert c.shape == (4, 1) and m.shape == (4, 1)
        for i in range(3):
            row = self.SEQ[i][~np.isnan(self.SEQ[i])]
            cs, ms = mpt.continuity(row, x[i], self.SIGMA)
            assert c[i, 0] == pytest.approx(cs[0], abs=1e-12)
            assert m[i, 0] == pytest.approx(ms[0], abs=1e-12)
        assert np.isnan(c[3, 0]) and np.isnan(m[3, 0])

    def test_transposed_rows_agree(self):
        x = np.array([[69], [58], [74], [60]]) * 100.0
        c, m = mpt.continuity(self.SEQ, x, self.SIGMA)
        assert c[2, 0] == pytest.approx(c[0, 0], abs=1e-12)
        assert m[2, 0] == pytest.approx(m[0, 0], abs=1e-12)

    def test_shared_queries(self):
        x = np.array([5800.0, 6900.0, 7400.0])
        c, m = mpt.continuity(self.SEQ[:3], x, self.SIGMA, mode="lenient")
        assert c.shape == (3, 3)
        for i in range(3):
            row = self.SEQ[i][~np.isnan(self.SEQ[i])]
            cs, ms = mpt.continuity(row, x, self.SIGMA, mode="lenient")
            np.testing.assert_allclose(c[i], cs, atol=1e-12)
            np.testing.assert_allclose(m[i], ms, atol=1e-12)

    def test_nan_query_gives_nan(self):
        x = np.array([[6900.0, np.nan]] * 3)
        c, m = mpt.continuity(self.SEQ[:3], x, self.SIGMA)
        assert np.all(np.isnan(c[:, 1])) and np.all(np.isfinite(c[:, 0]))

    def test_weights(self):
        w = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        x = np.array([[6900.0]] * 3)
        c, m = mpt.continuity(self.SEQ[:3], x, self.SIGMA, w=w)
        cs, ms = mpt.continuity(self.SEQ[0], [6900.0], self.SIGMA, w=w)
        assert c[0, 0] == pytest.approx(cs[0], abs=1e-12)
        c2, _ = mpt.continuity(self.SEQ[:3], x, self.SIGMA,
                               w=np.tile(w, (3, 1)))
        np.testing.assert_allclose(c, c2, atol=1e-12)

    def test_wrong_query_rows_raises(self):
        with pytest.raises(ValueError):
            mpt.continuity(self.SEQ, np.zeros((2, 1)), self.SIGMA)

    def test_one_d_form_unchanged(self):
        c, m = mpt.continuity([3, 5, 7, 7, 9], [11], 0, mode="lenient")
        assert c.shape == (1,)
        assert c[0] == pytest.approx(3.0) and m[0] == pytest.approx(6.0)


# ---------------------------------------------------------------------
#  markov_s
# ---------------------------------------------------------------------

class TestMarkovSBatched:

    CLAVE = [0, 3, 6, 10, 12]
    P = np.array([[0, 3, 6, 10, 12],
                  [2, 5, 8, 12, 14],          # clave rotated by 2
                  [0, 4, 8, 12, np.nan],
                  [np.nan] * 5])

    def test_matches_single_form(self):
        Y = mpt.markov_s(self.P, None, 16)
        assert Y.shape == (4, 16)
        for i in range(3):
            row = self.P[i][~np.isnan(self.P[i])].astype(int)
            np.testing.assert_allclose(Y[i], mpt.markov_s(row, None, 16),
                                       atol=1e-12)
        assert np.all(np.isnan(Y[3]))

    def test_rotation_is_equivariant(self):
        Y = mpt.markov_s(self.P, None, 16)
        np.testing.assert_allclose(Y[1], np.roll(Y[0], 2), atol=1e-12)

    def test_weights_and_S(self):
        w = np.array([1.0, 0.5, 0.8, 0.7, 1.0])
        Y = mpt.markov_s(self.P[:2], w, 16, 5)
        np.testing.assert_allclose(
            Y[0], mpt.markov_s(self.CLAVE, w, 16, 5), atol=1e-12)
        np.testing.assert_allclose(
            Y[1], mpt.markov_s(self.P[1].astype(int), w, 16, 5),
            atol=1e-12)

    def test_symmetric_pattern(self):
        """A pattern equal to some of its own rotations (ties in the
        canonical search) still rotates back into place."""
        P = np.array([[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14]])
        Y = mpt.markov_s(P, None, 16)
        for i in range(3):
            np.testing.assert_allclose(
                Y[i], mpt.markov_s(P[i], None, 16), atol=1e-12)

    def test_bad_positions_raise(self):
        with pytest.raises(ValueError):
            mpt.markov_s(np.array([[0, 3], [1.5, 4]]), None, 16)
        with pytest.raises(ValueError):
            mpt.markov_s(np.array([[0, 3], [1, 16]]), None, 16)
