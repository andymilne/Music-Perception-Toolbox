"""Tests for the unified :func:`entropy_exp_tens`.

Coverage of the new v3 capabilities:

- Density list input — output shape ``(M,)``, with optional dedup.
- Raw single-attribute batched input (2-D ``P``) — output shape ``(M,)``, with
  chord-level dedup; ``np.nan`` for invalid rows (K < r).
- ``spectrum``, ``precision``, ``dedup`` kwargs.

The pre-existing scalar-density and raw single-multiset/MA scalar paths are
exercised by ``test_mpt.py`` and continue to work.
"""

import warnings

import numpy as np
import pytest

import mpt
from mpt import build_exp_tens, entropy_exp_tens


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def density_set():
    """Three single-multiset densities for testing."""
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


# =====================================================================
# Density scalar (v2.0 — regression check via unified entry)
# =====================================================================


class TestDensityScalar:
    def test_returns_python_float(self, density_set):
        # method='normalized' to preserve the [0, 1] range assertion;
        # the default 'shannon' returns raw H which can exceed 1.
        h = entropy_exp_tens(
            density_set["major"], method='normalized', n_points_per_dim=200,
        )
        assert isinstance(h, float)
        assert 0.0 <= h <= 1.0

    def test_method_shannon_is_raw(self, density_set):
        # method='shannon' now returns raw H = -sum q log_b q
        # (the v2.0 default normalize=True behaviour has been moved to
        # method='normalized'). Raw H is unbounded above; we only check
        # the return type here.
        h = entropy_exp_tens(
            density_set["major"], method='shannon', n_points_per_dim=200,
        )
        assert isinstance(h, float)


# =====================================================================
# Density list
# =====================================================================


class TestDensityList:
    def test_basic_shape(self, density_set):
        ds = [density_set["major"], density_set["minor"], density_set["dim"]]
        out = entropy_exp_tens(ds, n_points_per_dim=200)
        assert isinstance(out, np.ndarray)
        assert out.shape == (3,)

    def test_values_match_per_density_calls(self, density_set):
        ds = [density_set["major"], density_set["minor"], density_set["dim"]]
        out = entropy_exp_tens(ds, n_points_per_dim=200)
        ref = np.array([
            entropy_exp_tens(d, n_points_per_dim=200) for d in ds
        ])
        np.testing.assert_allclose(out, ref, atol=1e-12)

    def test_length_one_list_returns_array(self, density_set):
        """Option II: length-1 list returns shape (1,), not scalar."""
        out = entropy_exp_tens([density_set["major"]], n_points_per_dim=200)
        assert isinstance(out, np.ndarray)
        assert out.shape == (1,)

    def test_empty_list_returns_empty_array(self):
        out = entropy_exp_tens([], n_points_per_dim=200)
        assert isinstance(out, np.ndarray)
        assert out.shape == (0,)

    def test_dedup_with_duplicate_densities(self, density_set):
        ds = [density_set["major"], density_set["major"], density_set["minor"]]
        out = entropy_exp_tens(ds, n_points_per_dim=200)
        # First two are the same density — entropy should be exactly equal.
        assert out[0] == out[1]

    def test_dedup_default_matches_no_dedup(self, density_set):
        ds = [density_set["major"], density_set["minor"], density_set["major"]]
        out_dedup = entropy_exp_tens(ds, n_points_per_dim=200)
        out_no_dedup = entropy_exp_tens(ds, dedup=False, n_points_per_dim=200)
        np.testing.assert_allclose(out_dedup, out_no_dedup, atol=1e-12)


# =====================================================================
# Raw single-attribute batched
# =====================================================================


class TestRawSABatched:
    def test_basic_shape(self):
        P = np.array([
            [0.0, 400.0, 700.0],
            [0.0, 300.0, 700.0],
            [0.0, 300.0, 600.0],
        ])
        out = entropy_exp_tens(
            P, None, 60.0, 2, False, True, 1200.0, n_points_per_dim=200,
        )
        assert out.shape == (3,)
        assert np.all(~np.isnan(out))

    def test_matches_per_row_scalar_calls(self):
        P = np.array([
            [0.0, 400.0, 700.0],
            [0.0, 300.0, 700.0],
        ])
        out = entropy_exp_tens(
            P, None, 60.0, 2, False, True, 1200.0, n_points_per_dim=200,
        )
        ref = np.array([
            entropy_exp_tens(
                P[i].tolist(), None, 60.0, 2, False, True, 1200.0,
                n_points_per_dim=200,
            )
            for i in range(P.shape[0])
        ])
        np.testing.assert_allclose(out, ref, atol=1e-12)

    def test_nan_padded_rows(self):
        P = np.array([
            [0.0, 400.0, 700.0],
            [0.0, 300.0, np.nan],
        ])
        out = entropy_exp_tens(
            P, None, 60.0, 2, False, True, 1200.0, n_points_per_dim=200,
        )
        assert out.shape == (2,)
        assert np.all(~np.isnan(out))

    def test_invalid_row_returns_nan(self):
        P = np.array([
            [0.0, 400.0, 700.0],
            [np.nan, np.nan, np.nan],   # K < r=2
        ])
        out = entropy_exp_tens(
            P, None, 60.0, 2, False, True, 1200.0, n_points_per_dim=200,
        )
        assert not np.isnan(out[0])
        assert np.isnan(out[1])

    def test_with_weights(self):
        P = np.array([[0.0, 400.0, 700.0], [0.0, 300.0, 700.0]])
        W = np.array([[3.0, 1.0, 2.0], [1.0, 2.0, 3.0]])
        out = entropy_exp_tens(
            P, W, 60.0, 2, False, True, 1200.0, n_points_per_dim=200,
        )
        assert out.shape == (2,)

    def test_single_row_returns_length_1(self):
        """Option II: 1-row matrix returns (1,), not scalar."""
        P = np.array([[0.0, 400.0, 700.0]])
        out = entropy_exp_tens(
            P, None, 60.0, 2, False, True, 1200.0, n_points_per_dim=200,
        )
        assert isinstance(out, np.ndarray)
        assert out.shape == (1,)

    def test_with_spectrum(self):
        P = np.array([[0.0, 400.0, 700.0], [0.0, 300.0, 700.0]])
        out = entropy_exp_tens(
            P, None, 60.0, 2, False, True, 1200.0,
            spectrum=("harmonic", 6, "geometric", 0.7),
            n_points_per_dim=200,
        )
        assert out.shape == (2,)

    def test_with_precision(self):
        """Precision rounding collapses FP-noise rows."""
        P = np.array([
            [0.0, 400.0, 700.0],
            [0.0 + 1e-13, 400.0 - 1e-13, 700.0 + 1e-13],
        ])
        out = entropy_exp_tens(
            P, None, 60.0, 2, False, True, 1200.0,
            precision=4, n_points_per_dim=200,
        )
        # Both rows canonicalise to the same key under precision rounding.
        assert out[0] == pytest.approx(out[1], abs=1e-12)

    def test_dedup_default_matches_no_dedup(self):
        P = np.array([[0.0, 400.0, 700.0], [0.0, 300.0, 700.0]])
        r_dedup = entropy_exp_tens(
            P, None, 60.0, 2, False, True, 1200.0, n_points_per_dim=200,
        )
        r_no_dedup = entropy_exp_tens(
            P, None, 60.0, 2, False, True, 1200.0,
            dedup=False, n_points_per_dim=200,
        )
        np.testing.assert_allclose(r_dedup, r_no_dedup, atol=1e-12)


# =====================================================================
# Errors
# =====================================================================


class TestErrors:
    def test_density_with_extra_args_raises(self, density_set):
        with pytest.raises(TypeError, match="no further positional"):
            entropy_exp_tens(density_set["major"], "extra")

    def test_density_list_with_extra_args_raises(self, density_set):
        with pytest.raises(TypeError, match="no further positional"):
            entropy_exp_tens(
                [density_set["major"], density_set["minor"]], "extra",
            )

    def test_spectrum_with_density_raises(self, density_set):
        with pytest.raises(TypeError, match="spectrum"):
            entropy_exp_tens(
                density_set["major"],
                spectrum=("harmonic", 6, "geometric", 0.7),
            )

    def test_precision_with_density_raises(self, density_set):
        with pytest.raises(TypeError, match="precision"):
            entropy_exp_tens(density_set["major"], precision=4)

    def test_precision_with_raw_scalar_raises(self):
        with pytest.raises(TypeError, match="batched"):
            entropy_exp_tens(
                [0.0, 400.0, 700.0], None,
                60.0, 2, False, True, 1200.0,
                precision=4,
            )


# =====================================================================
# Cross-form consistency
# =====================================================================


class TestCrossFormConsistency:
    def test_scalar_density_matches_raw_scalar(self):
        d = build_exp_tens(
            [0.0, 400.0, 700.0], None, 60.0, 2, False, True, 1200.0,
            verbose=False,
        )
        h_dens = entropy_exp_tens(d, n_points_per_dim=200)
        h_raw = entropy_exp_tens(
            [0.0, 400.0, 700.0], None, 60.0, 2, False, True, 1200.0,
            n_points_per_dim=200,
        )
        assert h_dens == pytest.approx(h_raw, abs=1e-12)

    def test_raw_batch_single_row_matches_scalar(self):
        """A 1-row raw batch produces a (1,) array whose value matches the
        equivalent scalar call."""
        h_scalar = entropy_exp_tens(
            [0.0, 400.0, 700.0], None, 60.0, 2, False, True, 1200.0,
            n_points_per_dim=200,
        )
        h_batch = entropy_exp_tens(
            np.array([[0.0, 400.0, 700.0]]), None,
            60.0, 2, False, True, 1200.0,
            n_points_per_dim=200,
        )
        assert h_scalar == pytest.approx(float(h_batch[0]), abs=1e-12)


class TestEntropyExpTensVerboseEstimate:
    """Bundle 2: entropy_exp_tens prints a time estimate via empirical
    calibration when ``verbose=True`` (default) in the single-attribute batched
    dispatch, and is silent when ``verbose=False``.
    """

    def test_batched_verbose_true_silent_for_fast(self, capsys):
        # New contract (commit 14+): batched-mode estimates are gated
        # at the same 10-s threshold as scalar mode. A 3-row fast call
        # is silent.
        import mpt
        P = np.array([
            [0, 100, 200, 300],
            [0, 200, 400, 600],
            [0, 100, 200, 300],
        ])
        H = mpt.entropy_exp_tens(
            P, None, 12.0, 1, False, False, 1200,
            x_min=0, x_max=600, n_points_per_dim=1200, verbose=True,
        )
        captured = capsys.readouterr()
        assert captured.out == ""
        assert H.shape == (3,)

    def test_batched_verbose_false_silent(self, capsys):
        import mpt
        P = np.array([[0, 100, 200], [0, 200, 400]])
        H = mpt.entropy_exp_tens(
            P, None, 12.0, 1, False, False, 1200,
            x_min=0, x_max=600, n_points_per_dim=1200, verbose=False,
        )
        captured = capsys.readouterr()
        assert captured.out == ""
        assert H.shape == (2,)

    def test_verbose_default_is_true_but_silent_for_fast(self, capsys):
        import mpt
        P = np.array([[0, 100, 200], [0, 200, 400]])
        mpt.entropy_exp_tens(
            P, None, 12.0, 1, False, False, 1200,
            x_min=0, x_max=600, n_points_per_dim=1200,
        )
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_numerical_results_unchanged_by_verbose(self):
        import mpt
        P = np.array([[0, 100, 200], [0, 200, 400]])
        H_a = mpt.entropy_exp_tens(
            P, None, 12.0, 1, False, False, 1200,
            x_min=0, x_max=600, n_points_per_dim=1200, verbose=True,
        )
        H_b = mpt.entropy_exp_tens(
            P, None, 12.0, 1, False, False, 1200,
            x_min=0, x_max=600, n_points_per_dim=1200, verbose=False,
        )
        assert np.allclose(H_a, H_b, equal_nan=True)
