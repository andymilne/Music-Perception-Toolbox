"""Tests for the unified :func:`tensor_harmonicity`,
:func:`template_harmonicity`, and :func:`virtual_pitches`.

Coverage:

- Single-chord 1-D input (the v2.0 case) — scalar / tuple-of-scalars /
  tuple-of-arrays output.
- Batched 2-D input — ``(M,)`` for tensor_harmonicity, ``((M,), (M,))``
  for template_harmonicity, ``([M arrays], [M arrays])`` for
  virtual_pitches.
- NaN-padded rows; rows with too few valid pitches return ``NaN`` /
  empty arrays.
- Length-1 batched input (Option II — strict shape preservation).
- Cross-form numerical consistency (1-D matches 2-D single-row).
"""

import warnings

import numpy as np
import pytest

from mpt import (
    tensor_harmonicity,
    template_harmonicity,
    virtual_pitches,
)


# ---------------------------------------------------------------------
# tensor_harmonicity
# ---------------------------------------------------------------------


class TestTensorHarmonicityScalar:
    def test_returns_python_float(self):
        h = tensor_harmonicity([0.0, 400.0, 700.0])
        assert isinstance(h, float)
        assert h > 0

    def test_v20_signature_unchanged(self):
        # Major triad
        h_maj = tensor_harmonicity([0.0, 400.0, 700.0], None, 12.0)
        # Diminished triad
        h_dim = tensor_harmonicity([0.0, 300.0, 600.0], None, 12.0)
        # Major triad is more harmonic than diminished (well-known relation).
        assert h_maj > h_dim


class TestTensorHarmonicityBatched:
    def test_basic_shape(self):
        P = np.array([
            [0.0, 400.0, 700.0],     # major
            [0.0, 300.0, 700.0],     # minor
            [0.0, 300.0, 600.0],     # diminished
        ])
        h = tensor_harmonicity(P)
        assert h.shape == (3,)
        assert np.all(~np.isnan(h))

    def test_matches_per_row_calls(self):
        P = np.array([
            [0.0, 400.0, 700.0],
            [0.0, 300.0, 700.0],
        ])
        h = tensor_harmonicity(P)
        for i in range(P.shape[0]):
            ref = tensor_harmonicity(P[i].tolist())
            assert h[i] == pytest.approx(ref, abs=1e-12)

    def test_nan_padded_variable_cardinality(self):
        # Row 0: 3 pitches; row 1: 2 pitches with NaN padding.
        P = np.array([
            [0.0, 400.0, 700.0],
            [0.0, 700.0, np.nan],
        ])
        h = tensor_harmonicity(P)
        assert h.shape == (2,)
        assert np.all(~np.isnan(h))

    def test_invalid_row_returns_nan(self):
        # Row with only 1 valid pitch (< 2 required) → NaN.
        P = np.array([
            [0.0, 400.0, 700.0],
            [0.0, np.nan, np.nan],
        ])
        h = tensor_harmonicity(P)
        assert not np.isnan(h[0])
        assert np.isnan(h[1])

    def test_dedup_with_duplicate_chords(self):
        # Two identical chords + one different.
        P = np.array([
            [0.0, 400.0, 700.0],
            [0.0, 400.0, 700.0],
            [0.0, 300.0, 700.0],
        ])
        h = tensor_harmonicity(P)
        assert h[0] == h[1]
        assert h[0] != pytest.approx(h[2])

    def test_single_row_returns_length_1(self):
        """Option II: 1-row matrix returns shape (1,), not scalar."""
        P = np.array([[0.0, 400.0, 700.0]])
        h = tensor_harmonicity(P)
        assert isinstance(h, np.ndarray)
        assert h.shape == (1,)

    def test_with_weights(self):
        P = np.array([[0.0, 400.0, 700.0], [0.0, 300.0, 700.0]])
        W = np.array([[3.0, 1.0, 2.0], [1.0, 2.0, 3.0]])
        h = tensor_harmonicity(P, W)
        assert h.shape == (2,)


# ---------------------------------------------------------------------
# template_harmonicity
# ---------------------------------------------------------------------


class TestTemplateHarmonicityScalar:
    def test_returns_pair_of_floats(self):
        h_max, h_ent = template_harmonicity([0.0, 400.0, 700.0])
        assert isinstance(h_max, float)
        assert isinstance(h_ent, float)
        assert 0.0 <= h_max <= 1.0


class TestTemplateHarmonicityBatched:
    def test_basic_shape(self):
        P = np.array([
            [0.0, 400.0, 700.0],
            [0.0, 300.0, 700.0],
            [0.0, 300.0, 600.0],
        ])
        h_max, h_ent = template_harmonicity(P)
        assert h_max.shape == (3,)
        assert h_ent.shape == (3,)
        assert np.all(~np.isnan(h_max))
        assert np.all(~np.isnan(h_ent))

    def test_matches_per_row_calls(self):
        P = np.array([
            [0.0, 400.0, 700.0],
            [0.0, 300.0, 700.0],
        ])
        h_max, h_ent = template_harmonicity(P)
        for i in range(P.shape[0]):
            ref_max, ref_ent = template_harmonicity(P[i].tolist())
            assert h_max[i] == pytest.approx(ref_max, abs=1e-12)
            assert h_ent[i] == pytest.approx(ref_ent, abs=1e-12)

    def test_nan_padded_rows(self):
        P = np.array([
            [0.0, 400.0, 700.0],
            [0.0, 700.0, np.nan],
        ])
        h_max, h_ent = template_harmonicity(P)
        assert h_max.shape == (2,)
        assert h_ent.shape == (2,)

    def test_dedup_with_duplicate_chords(self):
        # Two identical major triads (dedup target) and a chromatic
        # cluster. Note: h_max from template_harmonicity is structurally
        # inversion-invariant for pure-tone chords (it depends only on
        # the chord's pairwise-interval multiset), so e.g. major and
        # minor cannot be distinguished here. We use a chord with a
        # different cardinality / interval-multiset for the inequality
        # check.
        P = np.array([
            [0.0, 400.0, 700.0],
            [0.0, 400.0, 700.0],
            [0.0, 1200.0, 700.0],   # different intervals → different multiset
        ])
        h_max, h_ent = template_harmonicity(P)
        # Dedup verified by exact equality across identical rows.
        assert h_max[0] == h_max[1]
        assert h_ent[0] == h_ent[1]
        # Distinct chord gives a different result.
        assert h_max[0] != h_max[2] or h_ent[0] != h_ent[2]

    def test_single_row_returns_length_1(self):
        P = np.array([[0.0, 400.0, 700.0]])
        h_max, h_ent = template_harmonicity(P)
        assert isinstance(h_max, np.ndarray)
        assert h_max.shape == (1,)
        assert h_ent.shape == (1,)


# ---------------------------------------------------------------------
# virtual_pitches
# ---------------------------------------------------------------------


class TestVirtualPitchesScalar:
    def test_returns_two_arrays(self):
        vp_p, vp_w = virtual_pitches([0.0, 400.0, 700.0])
        assert isinstance(vp_p, np.ndarray)
        assert isinstance(vp_w, np.ndarray)
        assert vp_p.shape == vp_w.shape


class TestVirtualPitchesBatched:
    def test_returns_two_lists(self):
        P = np.array([
            [0.0, 400.0, 700.0],
            [0.0, 300.0, 700.0],
        ])
        vp_p_list, vp_w_list = virtual_pitches(P)
        assert isinstance(vp_p_list, list)
        assert isinstance(vp_w_list, list)
        assert len(vp_p_list) == 2
        assert len(vp_w_list) == 2
        for vp_p, vp_w in zip(vp_p_list, vp_w_list):
            assert isinstance(vp_p, np.ndarray)
            assert isinstance(vp_w, np.ndarray)
            assert vp_p.shape == vp_w.shape

    def test_matches_per_row_calls(self):
        P = np.array([
            [0.0, 400.0, 700.0, np.nan],
            [0.0, 300.0, 700.0, 1000.0],
        ])
        vp_p_list, vp_w_list = virtual_pitches(P)
        for i in range(P.shape[0]):
            mask = ~np.isnan(P[i])
            ref_p, ref_w = virtual_pitches(P[i][mask].tolist())
            np.testing.assert_allclose(vp_p_list[i], ref_p, atol=1e-12)
            np.testing.assert_allclose(vp_w_list[i], ref_w, atol=1e-12)

    def test_variable_length_profiles(self):
        """Different chord ranges → different profile lengths."""
        P = np.array([
            [0.0, 400.0, 700.0, np.nan],          # max = 700
            [0.0, 300.0, 700.0, 1200.0],           # max = 1200 → longer profile
        ])
        vp_p_list, vp_w_list = virtual_pitches(P)
        assert len(vp_p_list[0]) < len(vp_p_list[1])

    def test_invalid_row_returns_empty(self):
        # Row with no valid pitches.
        P = np.array([
            [0.0, 400.0, 700.0],
            [np.nan, np.nan, np.nan],
        ])
        vp_p_list, vp_w_list = virtual_pitches(P)
        assert vp_p_list[0].size > 0
        assert vp_p_list[1].size == 0
        assert vp_w_list[1].size == 0

    def test_single_row_returns_length_1_list(self):
        P = np.array([[0.0, 400.0, 700.0]])
        vp_p_list, vp_w_list = virtual_pitches(P)
        assert len(vp_p_list) == 1
        assert len(vp_w_list) == 1


# ---------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------


class TestErrors:
    def test_3d_input_raises(self):
        P = np.zeros((2, 2, 2))
        with pytest.raises(ValueError, match="1-D"):
            tensor_harmonicity(P)
        with pytest.raises(ValueError, match="1-D"):
            template_harmonicity(P)
        with pytest.raises(ValueError, match="1-D"):
            virtual_pitches(P)


class TestTemplateHarmonicityVerboseEstimate:
    """v2.1.1+: template_harmonicity prints a time estimate via
    estimate_comp_time when ``verbose=True`` (default), and is silent
    when ``verbose=False``.
    """

    def test_scalar_verbose_true_prints(self, capsys):
        h_max, h_ent = template_harmonicity(
            [0, 400, 700], None, 12.0, verbose=True,
        )
        captured = capsys.readouterr()
        assert "template_harmonicity" in captured.out
        assert "estimated time" in captured.out
        assert isinstance(h_max, float)

    def test_scalar_verbose_false_silent(self, capsys):
        h_max, h_ent = template_harmonicity(
            [0, 400, 700], None, 12.0, verbose=False,
        )
        captured = capsys.readouterr()
        assert captured.out == ""
        assert isinstance(h_max, float)

    def test_batched_verbose_true_prints_rows_count(self, capsys):
        chords = np.array([
            [0, 400, 700],
            [0, 300, 700],
            [0, 300, 600],
        ])
        h_max_arr, h_ent_arr = template_harmonicity(
            chords, None, 12.0, verbose=True,
        )
        captured = capsys.readouterr()
        assert "batched" in captured.out
        assert "3 rows" in captured.out
        assert "estimated time" in captured.out
        assert h_max_arr.shape == (3,)

    def test_batched_verbose_false_silent(self, capsys):
        chords = np.array([
            [0, 400, 700],
            [0, 300, 700],
        ])
        h_max_arr, h_ent_arr = template_harmonicity(
            chords, None, 12.0, verbose=False,
        )
        captured = capsys.readouterr()
        assert captured.out == ""
        assert h_max_arr.shape == (2,)

    def test_verbose_default_is_true(self, capsys):
        # Behaviour parity with eval_exp_tens / cos_sim_exp_tens.
        template_harmonicity([0, 400, 700], None, 12.0)
        captured = capsys.readouterr()
        assert "template_harmonicity" in captured.out

    def test_numerical_results_unchanged_by_verbose(self):
        # Verbose flag should not affect the returned values.
        chord = [0, 400, 700]
        a_max, a_ent = template_harmonicity(chord, None, 12.0, verbose=True)
        b_max, b_ent = template_harmonicity(chord, None, 12.0, verbose=False)
        assert a_max == b_max
        assert a_ent == b_ent

    def test_batched_all_invalid_rows_no_estimate_attempt(self, capsys):
        # A batch where every row has no valid pitches must not crash
        # (no first-valid row to calibrate against). Output should
        # still be NaN-filled arrays.
        P_all_nan = np.full((3, 4), np.nan)
        h_max_arr, h_ent_arr = template_harmonicity(
            P_all_nan, None, 12.0, verbose=True,
        )
        # No exception, NaN output, no estimate line printed.
        assert np.all(np.isnan(h_max_arr))
        assert np.all(np.isnan(h_ent_arr))
