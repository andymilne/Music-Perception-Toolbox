"""Regression tests for the thinned :func:`batch_cos_sim_exp_tens`.

After commit 3, ``batch_cos_sim_exp_tens`` is thinned to wrap the
polymorphic ``cos_sim_exp_tens``: chord-level dedup of density
construction stays in the function, but pair-level dedup is delegated
to the polymorphic consumer. These tests verify that the thinned
function produces results numerically identical to a manual loop over
:func:`cos_sim_exp_tens_raw` (the v2.0 per-row reference path), across
all four ``(is_rel, is_per)`` mode combinations, with and without
spectral augmentation, and on edge cases.

The existing :func:`test_mpt.test_batch_cos_sim` tests the shape and a
simple correctness check; these tests are tighter — every numerical
result must agree with the per-row reference to floating-point
tolerance.
"""

import numpy as np
import pytest

import mpt
from mpt import batch_cos_sim_exp_tens, cos_sim_exp_tens_raw


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def chord_corpus():
    """A small corpus with known structure for testing dedup behaviour.

    Includes:
    - duplicate chords (test chord-level dedup),
    - jointly-transposed pairs (test pair-level dedup in absolute mode),
    - independently-transposed pairs (test pair-level dedup in relative mode),
    - one row whose A is too short for r=2 (test invalid-row handling).
    """
    A = np.array([
        [0.0, 400.0, 700.0],     # major
        [0.0, 400.0, 700.0],     # major (duplicate row of 1)
        [200.0, 600.0, 900.0],   # major +200 (jointly transposed of 1)
        [0.0, 300.0, 700.0],     # minor
        [0.0, 300.0, 600.0],     # diminished
        [np.nan, np.nan, np.nan],  # invalid row
    ])
    B = np.array([
        [0.0, 200.0, 400.0, 500.0, 700.0, 900.0, 1100.0],   # diatonic
        [0.0, 200.0, 400.0, 500.0, 700.0, 900.0, 1100.0],   # diatonic
        [200.0, 400.0, 600.0, 700.0, 900.0, 1100.0, 1300.0],  # diatonic +200
        [0.0, 200.0, 400.0, 500.0, 700.0, 900.0, 1100.0],   # diatonic
        [0.0, 200.0, 400.0, 500.0, 700.0, 900.0, 1100.0],   # diatonic
        [0.0, 200.0, 400.0, 500.0, 700.0, 900.0, 1100.0],   # diatonic (paired with NaN A)
    ])
    return A, B


# ---------------------------------------------------------------------
# Reference implementation: manual loop calling cos_sim_exp_tens_raw
# ---------------------------------------------------------------------


def _manual_batch(A, B, sigma, r, is_rel, is_per, period,
                  weights_a=None, weights_b=None, spectrum=None):
    """Compute batch cosine similarity by looping over rows manually."""
    n_rows = A.shape[0]
    s = np.full(n_rows, np.nan)
    for i in range(n_rows):
        a_row = A[i]
        b_row = B[i]
        a_valid = a_row[~np.isnan(a_row)]
        b_valid = b_row[~np.isnan(b_row)]
        if len(a_valid) < r or len(b_valid) < r:
            continue
        wa = weights_a[i, ~np.isnan(a_row)] if weights_a is not None else None
        wb = weights_b[i, ~np.isnan(b_row)] if weights_b is not None else None
        if spectrum is not None:
            from mpt.spectra import add_spectra
            a_aug, wa_aug = add_spectra(
                a_valid, wa if wa is not None else np.ones_like(a_valid),
                *spectrum,
            )
            b_aug, wb_aug = add_spectra(
                b_valid, wb if wb is not None else np.ones_like(b_valid),
                *spectrum,
            )
            s[i] = cos_sim_exp_tens_raw(
                a_aug, wa_aug, b_aug, wb_aug,
                sigma, r, is_rel, is_per, period, verbose=False,
            )
        else:
            s[i] = cos_sim_exp_tens_raw(
                a_valid, wa, b_valid, wb,
                sigma, r, is_rel, is_per, period, verbose=False,
            )
    return s


# ---------------------------------------------------------------------
# Mode coverage: four (is_rel, is_per) combinations
# ---------------------------------------------------------------------


MODES = [
    pytest.param(False, False, id="abs-nonper"),
    pytest.param(False, True, id="abs-per"),
    pytest.param(True, False, id="rel-nonper"),
    pytest.param(True, True, id="rel-per"),
]


class TestNumericalIdentity:
    """Thinned batch_cos_sim_exp_tens must produce results numerically
    identical to a manual loop over cos_sim_exp_tens_raw."""

    @pytest.mark.parametrize("is_rel,is_per", MODES)
    def test_identity_uniform_weights(self, chord_corpus, is_rel, is_per):
        A, B = chord_corpus
        sigma, r, period = 15.0, 2, 1200.0

        batch_result = batch_cos_sim_exp_tens(
            A, B, sigma, r, is_rel, is_per, period, verbose=False,
        )
        manual_result = _manual_batch(A, B, sigma, r, is_rel, is_per, period)

        # Both arrays should have NaN in the same place (invalid row 5).
        np.testing.assert_array_equal(np.isnan(batch_result), np.isnan(manual_result))
        # Where valid, values should agree to floating-point tolerance.
        valid = ~np.isnan(batch_result)
        np.testing.assert_allclose(
            batch_result[valid], manual_result[valid],
            atol=1e-12, rtol=1e-12,
        )

    @pytest.mark.parametrize("is_rel,is_per", MODES)
    def test_identity_with_weights(self, chord_corpus, is_rel, is_per):
        A, B = chord_corpus
        # Reasonable per-row weights; NaN where pitches are NaN.
        wA = np.ones_like(A)
        wA[np.isnan(A)] = np.nan
        wA[0, :] = [3.0, 1.0, 2.0]   # major triad weighted asymmetrically
        wA[3, :] = [2.0, 1.5, 1.0]   # minor triad weighted differently

        wB = np.ones_like(B)
        wB[np.isnan(B)] = np.nan

        batch_result = batch_cos_sim_exp_tens(
            A, B, 15.0, 2, is_rel, is_per, 1200.0,
            weights_a=wA, weights_b=wB, verbose=False,
        )
        manual_result = _manual_batch(
            A, B, 15.0, 2, is_rel, is_per, 1200.0,
            weights_a=wA, weights_b=wB,
        )
        valid = ~np.isnan(batch_result)
        np.testing.assert_allclose(
            batch_result[valid], manual_result[valid],
            atol=1e-12, rtol=1e-12,
        )

    @pytest.mark.parametrize("is_rel,is_per", MODES)
    def test_identity_with_spectrum(self, chord_corpus, is_rel, is_per):
        """Spectral augmentation path must also be byte-identical."""
        A, B = chord_corpus
        # Standard harmonic spectrum: 6 partials, geometric weight type, alpha=0.7
        spec = ["harmonic", 6, "geometric", 0.7]

        batch_result = batch_cos_sim_exp_tens(
            A, B, 15.0, 2, is_rel, is_per, 1200.0,
            spectrum=spec, verbose=False,
        )
        manual_result = _manual_batch(
            A, B, 15.0, 2, is_rel, is_per, 1200.0, spectrum=spec,
        )
        valid = ~np.isnan(batch_result)
        np.testing.assert_allclose(
            batch_result[valid], manual_result[valid],
            atol=1e-12, rtol=1e-12,
        )


# ---------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------


class TestEdgeCases:

    def test_single_row(self):
        """Single-row batch (which would trigger the length-1 collapse
        of cos_sim_exp_tens internally) returns shape (1,)."""
        A = np.array([[0.0, 400.0, 700.0]])
        B = np.array([[0.0, 200.0, 400.0, 500.0, 700.0, 900.0, 1100.0]])
        result = batch_cos_sim_exp_tens(
            A, B, 15.0, 2, False, True, 1200.0, verbose=False,
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)
        assert not np.isnan(result[0])

    def test_all_invalid_rows(self):
        """Every row's A or B is too short for r — all results NaN."""
        A = np.array([[np.nan, np.nan]])
        B = np.array([[0.0, 400.0, 700.0]])
        result = batch_cos_sim_exp_tens(
            A, B, 15.0, 2, False, True, 1200.0, verbose=False,
        )
        assert result.shape == (1,)
        assert np.isnan(result[0])

    def test_chord_dedup_preserved(self, chord_corpus):
        """Verify chord-level dedup still happens: rows 0 and 1 (identical
        major triads) should produce the same result, as should rows 2
        in absolute periodic mode (jointly transposed pair)."""
        A, B = chord_corpus
        result = batch_cos_sim_exp_tens(
            A, B, 15.0, 2, False, True, 1200.0, verbose=False,
        )
        # Rows 0 and 1: identical inputs.
        assert result[0] == pytest.approx(result[1])
        # Rows 0 and 2: A2 = A0 + 200, B2 = B0 + 200 → joint co-transposition,
        # absolute periodic invariant.
        assert result[0] == pytest.approx(result[2])

    def test_relative_mode_chord_dedup(self, chord_corpus):
        """In relative mode, the major-vs-diatonic and (major+200)-vs-(diatonic+200)
        pairs should produce the same result (independent transposition
        invariance)."""
        A, B = chord_corpus
        result = batch_cos_sim_exp_tens(
            A, B, 15.0, 2, True, True, 1200.0, verbose=False,
        )
        # Row 0 (major-vs-diatonic) and row 2 (major+200-vs-diatonic+200):
        # both A and B independently transpose-invariant in relative mode.
        assert result[0] == pytest.approx(result[2])

    def test_verbose_output_does_not_crash(self, chord_corpus, capsys):
        """verbose=True must produce sensible output without crashing.

        Note: as of the v3 refactor the batched-raw implementation
        lives inside cos_sim_exp_tens (batch_cos_sim_exp_tens is a
        thin shim that forwards), so console output is labelled
        'cos_sim_exp_tens:' rather than 'batch_cos_sim_exp_tens:'.
        """
        A, B = chord_corpus
        result = batch_cos_sim_exp_tens(
            A, B, 15.0, 2, False, True, 1200.0, verbose=True,
        )
        captured = capsys.readouterr()
        # Some informative output should appear.
        assert "cos_sim_exp_tens" in captured.out
        assert "unique" in captured.out
        # Result is still computed correctly.
        assert result.shape == (6,)


class TestPrecisionRounding:
    """The precision kwarg should still work after thinning."""

    def test_precision_collapses_fp_noise(self):
        """Two rows that differ only by floating-point noise should
        deduplicate when precision is set."""
        A = np.array([
            [0.0, 400.0, 700.0],
            [1e-13, 400.0 - 1e-13, 700.0 + 1e-13],  # essentially the same chord
        ])
        B = np.array([
            [0.0, 300.0, 700.0],
            [0.0, 300.0, 700.0],
        ])
        # With precision rounding, row 0 and row 1 canonicalise to the same key.
        result = batch_cos_sim_exp_tens(
            A, B, 15.0, 2, False, True, 1200.0, precision=4, verbose=False,
        )
        assert result[0] == pytest.approx(result[1], abs=1e-12)
