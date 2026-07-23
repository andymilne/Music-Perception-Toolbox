"""Tests for the unified :func:`eval_exp_tens`.

Coverage:

- Pre-built density input (the v2.0 case, scalar) — output shape ``(nQ,)``.
- Density list input — output shape ``(M, nQ)``, with optional dedup.
- Raw single-multiset scalar input via unified entry — matches v2.0 raw form.
- Raw single-attribute batched input — output shape ``(M, nQ)``, with chord-level dedup.
- Raw MA scalar input via unified entry.
- ``normalize`` positional and keyword forms.
- ``spectrum`` / ``precision`` / ``dedup`` kwargs.
- ``eval_exp_tens_raw`` deprecation shim still works correctly.
"""

import warnings

import numpy as np
import pytest

import mpt
from mpt import build_exp_tens, eval_exp_tens


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def density_set():
    """Three single-multiset densities for testing.

    Uses a wider sigma (60) and a coarse 1-D-equivalent grid (interval-axis
    style at sparse query points) so that density values are not numerically
    underflowed to zero at all query points — important for testing whether
    different densities or normalisations actually produce different output.
    """
    sigma, r, period = 60.0, 2, 1200.0
    return {
        "major": build_exp_tens(
            [0.0, 400.0, 700.0], None, sigma, r, False, True, period,
            verbose=False,
        ),
        "minor": build_exp_tens(
            [0.0, 300.0, 700.0], None, sigma, r, False, True, period,
            verbose=False,
        ),
        "dim": build_exp_tens(
            [0.0, 300.0, 600.0], None, sigma, r, False, True, period,
            verbose=False,
        ),
    }


@pytest.fixture
def query_grid():
    """A 2-D query (matching dim = r - is_rel = 2) at points near the
    chord pitches so density values are non-trivial.
    """
    pa = np.array([0.0, 100.0, 300.0, 400.0, 500.0, 700.0, 800.0, 1000.0])
    pb = np.array([400.0, 500.0, 600.0, 700.0, 700.0, 0.0, 300.0, 400.0])
    return np.array([pa, pb])  # shape (2, 8)


# =====================================================================
# Density scalar (v2.0)
# =====================================================================


class TestDensityScalar:
    def test_returns_1d_array(self, density_set, query_grid):
        vals = eval_exp_tens(density_set["major"], query_grid, verbose=False)
        assert isinstance(vals, np.ndarray)
        assert vals.shape == (8,)

    def test_positional_normalize(self, density_set, query_grid):
        vals_default = eval_exp_tens(density_set["major"], query_grid, verbose=False)
        vals_pdf = eval_exp_tens(density_set["major"], query_grid, "pdf", verbose=False)
        # PDF normalisation should produce different magnitudes.
        assert not np.allclose(vals_default, vals_pdf)

    def test_kwarg_normalize(self, density_set, query_grid):
        vals_pos = eval_exp_tens(density_set["major"], query_grid, "pdf", verbose=False)
        vals_kw = eval_exp_tens(
            density_set["major"], query_grid, normalize="pdf", verbose=False,
        )
        np.testing.assert_array_equal(vals_pos, vals_kw)


# =====================================================================
# Density list
# =====================================================================


class TestDensityList:
    def test_basic_shape(self, density_set, query_grid):
        ds = [density_set["major"], density_set["minor"], density_set["dim"]]
        vals = eval_exp_tens(ds, query_grid, verbose=False)
        assert isinstance(vals, np.ndarray)
        assert vals.shape == (3, 8)

    def test_values_match_per_density_calls(self, density_set, query_grid):
        ds = [density_set["major"], density_set["minor"], density_set["dim"]]
        vals = eval_exp_tens(ds, query_grid, verbose=False)
        for i, d in enumerate(ds):
            ref = eval_exp_tens(d, query_grid, verbose=False)
            np.testing.assert_allclose(vals[i], ref, atol=1e-14)

    def test_length_one_list_returns_2d(self, density_set, query_grid):
        """Option II: length-1 list returns shape (1, nQ), not (nQ,)."""
        vals = eval_exp_tens([density_set["major"]], query_grid, verbose=False)
        assert vals.shape == (1, 8)

    def test_empty_list_returns_empty_array(self, density_set, query_grid):
        vals = eval_exp_tens([], query_grid, verbose=False)
        assert vals.shape == (0, 8)

    def test_dedup_with_duplicate_densities(self, density_set, query_grid):
        # Two of the three are the same; dedup should evaluate only twice.
        ds = [density_set["major"], density_set["major"], density_set["minor"]]
        vals = eval_exp_tens(ds, query_grid, verbose=False)
        # The first two rows should be exactly equal (same density).
        np.testing.assert_array_equal(vals[0], vals[1])
        # Third row differs.
        assert not np.allclose(vals[0], vals[2])

    def test_dedup_default_matches_no_dedup(self, density_set, query_grid):
        ds = [density_set["major"], density_set["minor"], density_set["major"]]
        vals_dedup = eval_exp_tens(ds, query_grid, verbose=False)
        vals_no_dedup = eval_exp_tens(ds, query_grid, dedup=False, verbose=False)
        np.testing.assert_allclose(vals_dedup, vals_no_dedup, atol=1e-14)

    def test_normalize_kwarg(self, density_set, query_grid):
        ds = [density_set["major"], density_set["minor"]]
        vals = eval_exp_tens(ds, query_grid, normalize="pdf", verbose=False)
        assert vals.shape == (2, 8)


# =====================================================================
# Raw single-multiset scalar
# =====================================================================


class TestRawSAScalar:
    def test_matches_density_scalar(self, query_grid):
        p = [0.0, 400.0, 700.0]
        sigma, r, period = 15.0, 2, 1200.0

        d = build_exp_tens(p, None, sigma, r, False, True, period, verbose=False)
        ref = eval_exp_tens(d, query_grid, verbose=False)
        # Raw single-multiset scalar via unified entry.
        vals = eval_exp_tens(
            p, None, sigma, r, False, True, period, query_grid, verbose=False,
        )
        np.testing.assert_allclose(vals, ref, atol=1e-14)

    def test_with_normalize_positional(self, query_grid):
        p = [0.0, 400.0, 700.0]
        d = build_exp_tens(p, None, 15.0, 2, False, True, 1200.0, verbose=False)
        ref = eval_exp_tens(d, query_grid, "pdf", verbose=False)
        vals = eval_exp_tens(
            p, None, 15.0, 2, False, True, 1200.0, True, query_grid, "pdf",
            verbose=False,
        )
        np.testing.assert_allclose(vals, ref, atol=1e-14)

    def test_with_spectrum(self, query_grid):
        """Spectrum kwarg applies addSpectra before density construction."""
        p = [0.0, 400.0, 700.0]
        vals = eval_exp_tens(
            p, None, 15.0, 2, False, True, 1200.0, query_grid,
            spectrum=("harmonic", 6, "geometric", 0.7),
            verbose=False,
        )
        assert vals.shape == (8,)


# =====================================================================
# Raw single-attribute batched
# =====================================================================


class TestRawSABatched:
    def test_basic_shape(self, query_grid):
        P = np.array([
            [0.0, 400.0, 700.0],
            [0.0, 300.0, 700.0],
            [0.0, 300.0, 600.0],
        ])
        vals = eval_exp_tens(
            P, None, 15.0, 2, False, True, 1200.0, query_grid, verbose=False,
        )
        assert vals.shape == (3, 8)
        assert np.all(~np.isnan(vals))

    def test_matches_per_row_density_calls(self, query_grid):
        P = np.array([
            [0.0, 400.0, 700.0],
            [0.0, 300.0, 700.0],
        ])
        vals = eval_exp_tens(
            P, None, 15.0, 2, False, True, 1200.0, query_grid, verbose=False,
        )
        ref = np.stack([
            eval_exp_tens(
                P[i].tolist(), None, 15.0, 2, False, True, 1200.0,
                query_grid, verbose=False,
            )
            for i in range(P.shape[0])
        ], axis=0)
        np.testing.assert_allclose(vals, ref, atol=1e-12)

    def test_nan_padded_rows(self, query_grid):
        P = np.array([
            [0.0, 400.0, 700.0],
            [0.0, 300.0, np.nan],   # 2 valid pitches
        ])
        vals = eval_exp_tens(
            P, None, 15.0, 2, False, True, 1200.0, query_grid, verbose=False,
        )
        assert vals.shape == (2, 8)
        assert np.all(~np.isnan(vals))

    def test_invalid_row_returns_nan(self, query_grid):
        P = np.array([
            [0.0, 400.0, 700.0],
            [np.nan, np.nan, np.nan],  # invalid: K < r
        ])
        vals = eval_exp_tens(
            P, None, 15.0, 2, False, True, 1200.0, query_grid, verbose=False,
        )
        assert vals.shape == (2, 8)
        assert np.all(~np.isnan(vals[0]))
        assert np.all(np.isnan(vals[1]))

    def test_chord_dedup_preserved(self, query_grid):
        """Two identical canonical chords should produce identical evaluations."""
        P = np.array([
            [0.0, 400.0, 700.0],
            [0.0, 400.0, 700.0],   # duplicate
            [200.0, 600.0, 900.0],  # joint-shift of row 0 → same canonical (abs/per)
        ])
        vals = eval_exp_tens(
            P, None, 15.0, 2, False, True, 1200.0, query_grid, verbose=False,
        )
        np.testing.assert_array_equal(vals[0], vals[1])
        # Note: row 2 is a transposed version of row 0; the *density itself*
        # depends on absolute pitches even though the canonical-form key
        # in absolute periodic mode collapses across joint shifts of (A,B).
        # For single-chord eval, transposition shifts the density in pitch-
        # space, so vals[0] and vals[2] are NOT equal even though their
        # canonical-form chord keys differ in shift only. The dedup applies
        # per absolute density, so vals[0] != vals[2] is expected.

    def test_with_precision(self, query_grid):
        """Precision rounding collapses FP-noise rows."""
        P = np.array([
            [0.0, 400.0, 700.0],
            [0.0 + 1e-13, 400.0 - 1e-13, 700.0 + 1e-13],
        ])
        vals = eval_exp_tens(
            P, None, 15.0, 2, False, True, 1200.0, query_grid,
            precision=4, verbose=False,
        )
        # Both rows canonicalise to same key with precision rounding;
        # results should be (essentially) equal.
        np.testing.assert_allclose(vals[0], vals[1], atol=1e-12)

    def test_with_spectrum(self, query_grid):
        P = np.array([[0.0, 400.0, 700.0], [0.0, 300.0, 700.0]])
        vals = eval_exp_tens(
            P, None, 15.0, 2, False, True, 1200.0, query_grid,
            spectrum=("harmonic", 6, "geometric", 0.7),
            verbose=False,
        )
        assert vals.shape == (2, 8)


# =====================================================================
# Raw MA scalar
# =====================================================================


class TestRawMAScalar:
    def test_basic_call(self):
        """Raw MA via unified entry matches building-then-evaluating."""
        p_attr = [np.array([[0.0, 400.0, 700.0]]).T]   # (3, 1) — 3 events, 1 attr-dim
        # Density has dim_per_attr = [r - is_rel] = [2]; query is (2, nQ).
        x = np.array([
            [0.0, 100.0, 400.0, 700.0, 800.0],
            [400.0, 700.0, 700.0, 0.0, 300.0],
        ])

        # Density-then-eval.
        d = build_exp_tens(
            p_attr, None, [60.0], [2], [False], [True], [1200.0],
            verbose=False,
        )
        ref = eval_exp_tens(d, x, verbose=False)

        # Raw-via-unified.
        vals = eval_exp_tens(
            p_attr, None, [60.0], [2], [False], [True], [1200.0], x,
            verbose=False,
        )
        np.testing.assert_allclose(vals, ref, atol=1e-14)


# =====================================================================
# Errors
# =====================================================================


class TestErrors:
    def test_too_few_args_raises(self):
        with pytest.raises(TypeError, match="at least 2 positional"):
            eval_exp_tens(verbose=False)

    def test_density_mode_extra_args_raises(self, density_set, query_grid):
        with pytest.raises(TypeError, match="2 or 3 positional"):
            eval_exp_tens(
                density_set["major"], query_grid, "none", "extra",
                verbose=False,
            )

    def test_spectrum_with_density_raises(self, density_set, query_grid):
        with pytest.raises(TypeError, match="spectrum"):
            eval_exp_tens(
                density_set["major"], query_grid,
                spectrum=("harmonic", 6, "geometric", 0.7),
                verbose=False,
            )

    def test_precision_with_density_raises(self, density_set, query_grid):
        with pytest.raises(TypeError, match="precision"):
            eval_exp_tens(
                density_set["major"], query_grid, precision=4, verbose=False,
            )


# =====================================================================
# eval_exp_tens_raw deprecation
# =====================================================================


class TestDeprecatedShim:
    def test_emits_deprecation_warning(self, query_grid):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mpt.eval_exp_tens_raw(
                [0.0, 400.0, 700.0], None, 15.0, 2, False, True, 1200.0,
                query_grid, verbose=False,
            )
            assert any(issubclass(wi.category, DeprecationWarning) for wi in w)

    def test_shim_produces_correct_results(self, query_grid):
        """The deprecated wrapper produces the same result as the unified entry."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            v_shim = mpt.eval_exp_tens_raw(
                [0.0, 400.0, 700.0], None, 15.0, 2, False, True, 1200.0,
                query_grid, verbose=False,
            )
        v_unified = eval_exp_tens(
            [0.0, 400.0, 700.0], None, 15.0, 2, False, True, 1200.0,
            query_grid, verbose=False,
        )
        np.testing.assert_allclose(v_shim, v_unified, atol=1e-14)
