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
    spectral_entropy,
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

    def test_transposition_dedup(self):
        """v2.2+: structurally-identical canonical chords share a
        cached vp_w. Two rows that are transpositions of each other
        must produce identical vp_w arrays and vp_p arrays that
        differ by exactly the transposition amount."""
        P = np.array([
            [0.0, 400.0, 700.0],
            [100.0, 500.0, 800.0],     # row 1 + 100
            [1200.0, 1600.0, 1900.0],  # row 1 + 1200
        ])
        vp_p_list, vp_w_list = virtual_pitches(P)

        # vp_w identical across all three (chord shape is invariant).
        np.testing.assert_array_equal(vp_w_list[0], vp_w_list[1])
        np.testing.assert_array_equal(vp_w_list[0], vp_w_list[2])

        # vp_p shifts by exactly the transposition.
        np.testing.assert_allclose(vp_p_list[1], vp_p_list[0] + 100.0, atol=1e-12)
        np.testing.assert_allclose(vp_p_list[2], vp_p_list[0] + 1200.0, atol=1e-12)

    def test_permutation_dedup(self):
        """Rows that are permutations of each other must produce
        identical (vp_p, vp_w) pairs."""
        P = np.array([
            [0.0, 400.0, 700.0],
            [700.0, 0.0, 400.0],
            [400.0, 700.0, 0.0],
        ])
        vp_p_list, vp_w_list = virtual_pitches(P)
        np.testing.assert_array_equal(vp_w_list[0], vp_w_list[1])
        np.testing.assert_array_equal(vp_w_list[0], vp_w_list[2])
        np.testing.assert_allclose(vp_p_list[0], vp_p_list[1], atol=1e-12)
        np.testing.assert_allclose(vp_p_list[0], vp_p_list[2], atol=1e-12)


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

    def test_scalar_verbose_true_silent_for_fast_calls(self, capsys):
        # New contract (commit 14+): scalar paths use ``min_print_sec=0.5``,
        # so verbose=True for a typical fast chord stays silent. The
        # estimate-print code path is exercised separately in
        # ``test_estimate_print_path_via_helper`` below and via the
        # batched-mode tests.
        h_max, h_ent = template_harmonicity(
            [0, 400, 700], None, 12.0, verbose=True,
        )
        captured = capsys.readouterr()
        assert captured.out == ""
        assert isinstance(h_max, float)

    def test_scalar_verbose_false_silent(self, capsys):
        h_max, h_ent = template_harmonicity(
            [0, 400, 700], None, 12.0, verbose=False,
        )
        captured = capsys.readouterr()
        assert captured.out == ""
        assert isinstance(h_max, float)

    def test_batched_verbose_true_silent_for_fast(self, capsys):
        # New contract (commit 14+): batched-mode estimates are gated
        # at the same 10-s threshold as scalar mode. A 3-row fast call
        # is silent. Print-path coverage lives in
        # ``TestMaybePrintBatchedEstimate``.
        chords = np.array([
            [0, 400, 700],
            [0, 300, 700],
            [0, 300, 600],
        ])
        h_max_arr, h_ent_arr = template_harmonicity(
            chords, None, 12.0, verbose=True,
        )
        captured = capsys.readouterr()
        assert captured.out == ""
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

    def test_verbose_default_is_true_but_silent_for_fast(self, capsys):
        # Default verbose=True still applies, but the 0.5 s threshold
        # means typical fast scalar calls produce no output. Behaviour
        # parity with eval_exp_tens / cos_sim_exp_tens (which print
        # unconditionally) is intentionally NOT preserved here — the
        # demo loops in templateHarmonicity-driven workflows generate
        # too much noise when each scalar call announces ~1 ms.
        template_harmonicity([0, 400, 700], None, 12.0)
        captured = capsys.readouterr()
        assert captured.out == ""

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


class TestTensorHarmonicityVerboseEstimate:
    """Bundle 1: tensor_harmonicity prints a time estimate via
    estimate_comp_time when ``verbose=True`` (default), and is silent
    when ``verbose=False``.

    Scalar mode forwards verbose to build_exp_tens, which prints its
    own estimate. Batched mode runs an empirical calibration with a
    warm-up sample.
    """

    def test_scalar_verbose_true_prints(self, capsys):
        h = tensor_harmonicity([0, 400, 700], None, 12.0, verbose=True)
        captured = capsys.readouterr()
        # Scalar prints a brief eval-time diagnostic.
        assert ("eval" in captured.out.lower() or
                "build" in captured.out.lower() or
                "estimated" in captured.out)
        assert isinstance(h, float)

    def test_scalar_verbose_false_silent(self, capsys):
        h = tensor_harmonicity([0, 400, 700], None, 12.0, verbose=False)
        captured = capsys.readouterr()
        assert captured.out == ""
        assert isinstance(h, float)

    def test_batched_verbose_true_silent_for_fast(self, capsys):
        chords = np.array([
            [0, 400, 700],
            [0, 300, 700],
            [0, 300, 600],
        ])
        out = tensor_harmonicity(chords, None, 12.0, verbose=True)
        captured = capsys.readouterr()
        assert captured.out == ""
        assert out.shape == (3,)

    def test_batched_verbose_false_silent(self, capsys):
        chords = np.array([[0, 400, 700], [0, 300, 700]])
        out = tensor_harmonicity(chords, None, 12.0, verbose=False)
        captured = capsys.readouterr()
        assert captured.out == ""
        assert out.shape == (2,)

    def test_verbose_default_is_true_but_silent_for_fast(self, capsys):
        chords = np.array([[0, 400, 700], [0, 300, 700]])
        tensor_harmonicity(chords, None, 12.0)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_numerical_results_unchanged_by_verbose(self):
        chord = [0, 400, 700]
        a = tensor_harmonicity(chord, None, 12.0, verbose=True)
        b = tensor_harmonicity(chord, None, 12.0, verbose=False)
        assert a == b

    def test_batched_all_invalid_rows_no_crash(self, capsys):
        P_all_nan = np.full((3, 4), np.nan)
        out = tensor_harmonicity(P_all_nan, None, 12.0, verbose=True)
        assert out.shape == (3,)
        assert np.all(np.isnan(out))


class TestVirtualPitchesVerboseEstimate:
    """Bundle 1: virtual_pitches prints a time estimate via
    estimate_comp_time when ``verbose=True`` (default), and is silent
    when ``verbose=False``.
    """

    def test_scalar_verbose_true_silent_for_fast_calls(self, capsys):
        # New contract (commit 14+): scalar paths use ``min_print_sec=0.5``;
        # verbose=True alone is not enough to print for a fast call.
        vp_p, vp_w = virtual_pitches([0, 400, 700], None, 12.0, verbose=True)
        captured = capsys.readouterr()
        assert captured.out == ""
        assert len(vp_p) > 0

    def test_scalar_verbose_false_silent(self, capsys):
        vp_p, vp_w = virtual_pitches([0, 400, 700], None, 12.0, verbose=False)
        captured = capsys.readouterr()
        assert captured.out == ""
        assert len(vp_p) > 0

    def test_batched_verbose_true_silent_for_fast(self, capsys):
        chords = np.array([
            [0, 400, 700],
            [0, 300, 700],
        ])
        vp_p_list, vp_w_list = virtual_pitches(chords, None, 12.0, verbose=True)
        captured = capsys.readouterr()
        assert captured.out == ""
        assert len(vp_p_list) == 2

    def test_batched_verbose_false_silent(self, capsys):
        chords = np.array([[0, 400, 700], [0, 300, 700]])
        vp_p_list, _ = virtual_pitches(chords, None, 12.0, verbose=False)
        captured = capsys.readouterr()
        assert captured.out == ""
        assert len(vp_p_list) == 2

    def test_verbose_default_is_true_but_silent_for_fast(self, capsys):
        virtual_pitches([0, 400, 700], None, 12.0)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_numerical_results_unchanged_by_verbose(self):
        chord = [0, 400, 700]
        a_p, a_w = virtual_pitches(chord, None, 12.0, verbose=True)
        b_p, b_w = virtual_pitches(chord, None, 12.0, verbose=False)
        assert np.allclose(a_p, b_p)
        assert np.allclose(a_w, b_w)

    def test_batched_all_invalid_rows_no_crash(self, capsys):
        P_all_nan = np.full((3, 4), np.nan)
        vp_p_list, vp_w_list = virtual_pitches(P_all_nan, None, 12.0, verbose=True)
        assert len(vp_p_list) == 3
        for vp_p in vp_p_list:
            assert vp_p.size == 0


class TestSpectralEntropyBatchedAndVerbose:
    """Bundle 2: spectral_entropy gains 2-D batched dispatch
    (rows = chords) plus a ``verbose`` argument with empirical-
    calibration upfront estimate in batched mode and kernel-only
    estimate in scalar mode.
    """

    def test_scalar_1d_unchanged(self):
        # v2.0 contract: 1-D in, scalar float out.
        H = spectral_entropy([0, 400, 700], None, 12.0, verbose=False)
        assert isinstance(H, float)

    def test_scalar_verbose_true_silent_for_fast_calls(self, capsys):
        # New contract (commit 14+): scalar paths use ``min_print_sec=0.5``;
        # verbose=True alone is not enough to print for a fast call.
        spectral_entropy([0, 400, 700], None, 12.0, verbose=True)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_scalar_verbose_false_silent(self, capsys):
        spectral_entropy([0, 400, 700], None, 12.0, verbose=False)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_batched_2d_returns_M_vector(self):
        chords = np.array([
            [0, 400, 700],
            [0, 300, 700],
            [0, 100, 200],
        ])
        H = spectral_entropy(chords, None, 12.0, verbose=False)
        assert H.shape == (3,)

    def test_batched_matches_scalar_per_row(self):
        chords = np.array([
            [0, 400, 700],
            [0, 1200, 2400],
            [0, 100, 200],
        ])
        H_batched = spectral_entropy(chords, None, 12.0, verbose=False)
        for i, row in enumerate(chords):
            H_scalar = spectral_entropy(row, None, 12.0, verbose=False)
            assert abs(H_batched[i] - H_scalar) < 1e-12

    def test_batched_dedup(self):
        # Identical canonical-form chords share a cached entropy.
        # Note: [0, 400, 700] and [0, 300, 700] are gap-permutation
        # symmetric (gaps 400+300 vs 300+400), so spectral entropy
        # is the same to machine precision. Use a chord whose
        # spectrum differs structurally for the inequality check.
        chords = np.array([
            [0, 400, 700],
            [100, 500, 800],   # transposition of row 0
            [0, 100, 200],     # tightly-spaced: different spectrum
        ])
        H = spectral_entropy(chords, None, 12.0, verbose=False)
        assert H[0] == H[1]   # transpositions give identical entropy
        assert H[0] != H[2]   # different spacing → different entropy

    def test_batched_verbose_true_silent_for_fast(self, capsys):
        chords = np.array([[0, 400, 700], [0, 300, 700]])
        spectral_entropy(chords, None, 12.0, verbose=True)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_batched_verbose_false_silent(self, capsys):
        chords = np.array([[0, 400, 700], [0, 300, 700]])
        spectral_entropy(chords, None, 12.0, verbose=False)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_batched_nan_padded_variable_cardinality(self):
        chords = np.array([
            [0, 400, 700, np.nan],
            [0, 1200, np.nan, np.nan],
            [0, 300, 600, 900],
        ])
        H = spectral_entropy(chords, None, 12.0, verbose=False)
        assert H.shape == (3,)
        assert not np.any(np.isnan(H))   # all rows have ≥1 valid pitch

    def test_batched_all_nan_row(self):
        chords = np.array([
            [0, 400, 700],
            [np.nan, np.nan, np.nan],
        ])
        H = spectral_entropy(chords, None, 12.0, verbose=False)
        assert not np.isnan(H[0])
        assert np.isnan(H[1])

    def test_batched_verbose_default_silent_for_fast(self, capsys):
        chords = np.array([[0, 400, 700], [0, 300, 700]])
        spectral_entropy(chords, None, 12.0)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_batched_3d_input_raises(self):
        P = np.zeros((2, 2, 2))
        with pytest.raises(ValueError, match="1-D|2-D"):
            spectral_entropy(P)


class TestEstimateCompTimePrintGating:
    """Bundle 14: ``estimate_comp_time`` gains a ``min_print_sec``
    parameter so callers in tight loops can suppress trivial estimates
    while still exposing them when work is genuinely expensive.
    """

    def test_default_min_print_sec_suppresses_tiny(self, capsys):
        # Default min_print_sec=0.5: 100-pair workload (sub-millisecond)
        # is silently dropped. (Bundle 14: gating applied at the
        # estimate_comp_time default rather than per-caller.)
        from mpt._utils import estimate_comp_time
        estimate_comp_time(100, 1, "tinywork", verbose=True)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_explicit_zero_disables_gating(self, capsys):
        # Pass min_print_sec=0 to recover the always-print behaviour.
        from mpt._utils import estimate_comp_time
        estimate_comp_time(100, 1, "tinywork", verbose=True, min_print_sec=0)
        captured = capsys.readouterr()
        assert "tinywork" in captured.out

    def test_min_print_sec_suppresses_tiny_estimates(self, capsys):
        # min_print_sec=0.5: 100-pair workload (sub-millisecond) is
        # silently dropped.
        from mpt._utils import estimate_comp_time
        est = estimate_comp_time(
            100, 1, "tinywork", verbose=True, min_print_sec=0.5,
        )
        captured = capsys.readouterr()
        assert captured.out == ""
        assert est > 0   # estimate is still returned

    def test_min_print_sec_does_not_suppress_large_estimates(self, capsys):
        # A workload large enough to plausibly exceed 0.5 s prints
        # even with min_print_sec=0.5. We construct n_pairs from the
        # cached calibration rate so the estimate is well above
        # threshold; the rate cache is populated by the first call
        # above.
        from mpt._utils import estimate_comp_time, _rate_cache
        rate = _rate_cache.get(1)
        if rate is None:
            estimate_comp_time(100, 1, "", verbose=False)
            rate = _rate_cache[1]
        n_for_2_sec = int(rate * 2.0)
        estimate_comp_time(
            n_for_2_sec, 1, "bigwork", verbose=True, min_print_sec=0.5,
        )
        captured = capsys.readouterr()
        assert "bigwork" in captured.out
        assert "estimated time" in captured.out


class TestMaybePrintBatchedEstimate:
    """Bundle 14: ``maybe_print_batched_estimate`` (and its MATLAB
    counterpart ``printBatchedEstimate``) gates the batched-mode
    upfront print on the same 10-s threshold as ``estimate_comp_time``.
    """

    def test_default_threshold_suppresses_short(self, capsys):
        from mpt._utils import maybe_print_batched_estimate
        maybe_print_batched_estimate("foo", 100, 5.0, verbose=True)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_default_threshold_prints_long(self, capsys):
        from mpt._utils import maybe_print_batched_estimate
        maybe_print_batched_estimate("foo", 100, 30.0, verbose=True)
        captured = capsys.readouterr()
        assert "foo" in captured.out
        assert "batched, 100 rows" in captured.out
        assert "estimated time" in captured.out
        assert "(Ctrl+C to cancel)" in captured.out

    def test_verbose_false_silences(self, capsys):
        from mpt._utils import maybe_print_batched_estimate
        maybe_print_batched_estimate("foo", 100, 30.0, verbose=False)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_explicit_min_print_sec_zero_prints_short(self, capsys):
        from mpt._utils import maybe_print_batched_estimate
        maybe_print_batched_estimate(
            "foo", 100, 0.05, verbose=True, min_print_sec=0,
        )
        captured = capsys.readouterr()
        assert "foo" in captured.out
        # Time formatting: 50 ms case
        assert "50 ms" in captured.out

    def test_time_format_units(self, capsys):
        from mpt._utils import maybe_print_batched_estimate
        # Force print regardless of size via min_print_sec=0
        for est, expected in [
            (0.5, "500 ms"),
            (5.0, "5.0 s"),
            (300.0, "5.0 min"),
            (10800.0, "3.0 hr"),
        ]:
            maybe_print_batched_estimate(
                "foo", 1, est, verbose=True, min_print_sec=0,
            )
            captured = capsys.readouterr()
            assert expected in captured.out, (
                f"expected '{expected}' for est={est}, got: {captured.out}"
            )
