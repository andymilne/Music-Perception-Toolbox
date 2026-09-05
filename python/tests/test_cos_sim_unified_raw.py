"""Tests for raw input dispatch in the unified :func:`cos_sim_exp_tens`.

The unified entry accepts:

- pre-built density (single or list) — exercised by ``test_cos_sim_polymorphic.py``;
- raw 1-D ndarrays for single-multiset single chord pair (replaces ``cos_sim_exp_tens_raw``);
- raw 2-D ndarrays for single-attribute batched chord pairs (replaces ``batch_cos_sim_exp_tens``);
- raw list-of-arrays for MA single chord pair (replaces ``cos_sim_exp_tens_raw`` MA path).

These tests verify the raw paths via the unified entry directly (without
going through the deprecation shims), matching results to the v2.0 raw
forms numerically.
"""

import warnings

import numpy as np
import pytest

import mpt
from mpt import build_exp_tens, cos_sim_exp_tens


# ---------------------------------------------------------------------
# Raw single-multiset scalar
# ---------------------------------------------------------------------


class TestRawSAScalar:
    """Raw single-multiset scalar input (1-D arrays) via cos_sim_exp_tens."""

    def test_returns_python_float(self):
        s = cos_sim_exp_tens(
            [0.0, 400.0, 700.0], None, [0.0, 300.0, 700.0], None,
            15.0, 2, False, True, 1200.0,
            verbose=False,
        )
        assert isinstance(s, float)

    def test_self_cosine_is_one(self):
        s = cos_sim_exp_tens(
            [0.0, 400.0, 700.0], None, [0.0, 400.0, 700.0], None,
            15.0, 2, False, True, 1200.0,
            verbose=False,
        )
        assert s == pytest.approx(1.0)

    def test_matches_density_scalar(self):
        """Raw 1-D input produces the same result as building densities then comparing."""
        p1, p2 = [0.0, 400.0, 700.0], [0.0, 300.0, 700.0]
        s_raw = cos_sim_exp_tens(
            p1, None, p2, None, 15.0, 2, False, True, 1200.0,
            verbose=False,
        )
        d1 = build_exp_tens(p1, None, 15.0, 2, False, True, 1200.0, verbose=False)
        d2 = build_exp_tens(p2, None, 15.0, 2, False, True, 1200.0, verbose=False)
        s_dens = cos_sim_exp_tens(d1, d2, verbose=False)
        assert s_raw == pytest.approx(s_dens, abs=1e-14)

    def test_with_weights(self):
        p1, w1 = [0.0, 400.0, 700.0], [3.0, 1.0, 2.0]
        p2, w2 = [0.0, 300.0, 700.0], [1.0, 2.0, 3.0]
        s = cos_sim_exp_tens(
            p1, w1, p2, w2, 15.0, 2, False, True, 1200.0,
            verbose=False,
        )
        assert 0.0 < s < 1.0

    def test_with_spectrum(self):
        """Spectrum kwarg applies in raw single-multiset scalar mode."""
        p1, p2 = [0.0, 400.0, 700.0], [0.0, 400.0, 700.0]
        s_no_spec = cos_sim_exp_tens(
            p1, None, p2, None, 15.0, 2, False, True, 1200.0,
            verbose=False,
        )
        s_spec = cos_sim_exp_tens(
            p1, None, p2, None, 15.0, 2, False, True, 1200.0,
            spectrum=("harmonic", 6, "geometric", 0.7),
            verbose=False,
        )
        # Self-similarity is still 1.0 with or without spectrum.
        assert s_no_spec == pytest.approx(1.0)
        assert s_spec == pytest.approx(1.0)

    def test_numpy_array_input(self):
        """ndarray input works the same as list input."""
        p1 = np.array([0.0, 400.0, 700.0])
        p2 = np.array([0.0, 300.0, 700.0])
        s = cos_sim_exp_tens(
            p1, None, p2, None, 15.0, 2, False, True, 1200.0,
            verbose=False,
        )
        assert isinstance(s, float)
        assert 0.0 < s < 1.0


# ---------------------------------------------------------------------
# Raw single-attribute batched
# ---------------------------------------------------------------------


class TestRawSABatched:
    """Raw single-attribute batched input (2-D ndarrays) via cos_sim_exp_tens."""

    def test_basic_shape(self):
        P1 = np.array([
            [0.0, 400.0, 700.0],
            [0.0, 300.0, 700.0],
            [0.0, 300.0, 600.0],
        ])
        P2 = np.array([
            [0.0, 200.0, 400.0, 500.0, 700.0, 900.0, 1100.0],
            [0.0, 200.0, 400.0, 500.0, 700.0, 900.0, 1100.0],
            [0.0, 200.0, 400.0, 500.0, 700.0, 900.0, 1100.0],
        ])
        result = cos_sim_exp_tens(
            P1, None, P2, None, 15.0, 2, False, True, 1200.0,
            verbose=False,
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        assert np.all(~np.isnan(result))

    def test_matches_per_row_density_calls(self):
        """Batched raw input produces the same results as per-row density calls."""
        P1 = np.array([
            [0.0, 400.0, 700.0],
            [0.0, 300.0, 700.0],
        ])
        P2 = np.array([
            [0.0, 200.0, 400.0, 500.0, 700.0, 900.0, 1100.0],
            [0.0, 200.0, 400.0, 500.0, 700.0, 900.0, 1100.0],
        ])
        result = cos_sim_exp_tens(
            P1, None, P2, None, 15.0, 2, False, True, 1200.0,
            verbose=False,
        )
        per_row = np.array([
            cos_sim_exp_tens(
                P1[i].tolist(), None, P2[i].tolist(), None,
                15.0, 2, False, True, 1200.0, verbose=False,
            )
            for i in range(P1.shape[0])
        ])
        np.testing.assert_allclose(result, per_row, atol=1e-12)

    def test_nan_padded_rows(self):
        """Variable-cardinality rows handled via NaN padding."""
        P1 = np.array([
            [0.0, 400.0, 700.0],          # 3 notes
            [0.0, 300.0, np.nan],          # 2 notes (NaN-padded)
        ])
        P2 = np.array([
            [0.0, 200.0, 400.0, 700.0],
            [0.0, 200.0, 400.0, 700.0],
        ])
        result = cos_sim_exp_tens(
            P1, None, P2, None, 15.0, 2, False, True, 1200.0,
            verbose=False,
        )
        assert result.shape == (2,)
        assert np.all(~np.isnan(result))

    def test_invalid_row_returns_nan(self):
        """Row with K < r returns NaN."""
        P1 = np.array([
            [0.0, 400.0, 700.0],
            [np.nan, np.nan, np.nan],   # invalid: K < r=2
        ])
        P2 = np.array([
            [0.0, 200.0, 400.0, 700.0],
            [0.0, 200.0, 400.0, 700.0],
        ])
        result = cos_sim_exp_tens(
            P1, None, P2, None, 15.0, 2, False, True, 1200.0,
            verbose=False,
        )
        assert not np.isnan(result[0])
        assert np.isnan(result[1])

    def test_with_weights(self):
        P1 = np.array([[0.0, 400.0, 700.0], [0.0, 300.0, 700.0]])
        W1 = np.array([[3.0, 1.0, 2.0], [1.0, 2.0, 3.0]])
        P2 = np.array([
            [0.0, 200.0, 400.0, 500.0, 700.0, 900.0, 1100.0],
            [0.0, 200.0, 400.0, 500.0, 700.0, 900.0, 1100.0],
        ])
        result = cos_sim_exp_tens(
            P1, W1, P2, None, 15.0, 2, False, True, 1200.0,
            verbose=False,
        )
        assert result.shape == (2,)

    def test_single_row_returns_length_1_array(self):
        """A single-row matrix returns shape (1,) — Option II strict shape preservation."""
        P1 = np.array([[0.0, 400.0, 700.0]])
        P2 = np.array([[0.0, 300.0, 700.0]])
        result = cos_sim_exp_tens(
            P1, None, P2, None, 15.0, 2, False, True, 1200.0,
            verbose=False,
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)

    def test_with_spectrum(self):
        P1 = np.array([[0.0, 400.0, 700.0], [0.0, 300.0, 700.0]])
        P2 = np.array([[0.0, 400.0, 700.0], [0.0, 300.0, 700.0]])
        result = cos_sim_exp_tens(
            P1, None, P2, None, 15.0, 2, False, True, 1200.0,
            spectrum=("harmonic", 6, "geometric", 0.7),
            verbose=False,
        )
        # Each row pair is self-comparison: cosine = 1.0
        np.testing.assert_allclose(result, 1.0, atol=1e-12)

    def test_with_precision(self):
        """Precision rounding collapses FP-noise rows."""
        P1 = np.array([
            [0.0, 400.0, 700.0],
            [0.0 + 1e-13, 400.0 - 1e-13, 700.0 + 1e-13],   # essentially same
        ])
        P2 = np.array([
            [0.0, 300.0, 700.0],
            [0.0, 300.0, 700.0],
        ])
        result = cos_sim_exp_tens(
            P1, None, P2, None, 15.0, 2, False, True, 1200.0,
            precision=4, verbose=False,
        )
        # With precision rounding, both rows canonicalise to the same key
        # and produce the same numerical result.
        assert result[0] == pytest.approx(result[1], abs=1e-12)

    def test_dedup_default_matches_no_dedup(self):
        """dedup=True (default) produces identical results to dedup=False."""
        P1 = np.array([[0.0, 400.0, 700.0], [0.0, 300.0, 700.0]])
        P2 = np.array([
            [0.0, 200.0, 400.0, 500.0, 700.0, 900.0, 1100.0],
            [0.0, 200.0, 400.0, 500.0, 700.0, 900.0, 1100.0],
        ])
        r_dedup = cos_sim_exp_tens(
            P1, None, P2, None, 15.0, 2, False, True, 1200.0, verbose=False,
        )
        r_no_dedup = cos_sim_exp_tens(
            P1, None, P2, None, 15.0, 2, False, True, 1200.0,
            dedup=False, verbose=False,
        )
        np.testing.assert_allclose(r_dedup, r_no_dedup, atol=1e-12)


# ---------------------------------------------------------------------
# Raw MA scalar
# ---------------------------------------------------------------------


class TestRawMAScalar:
    """Raw multi-attribute scalar input via cos_sim_exp_tens."""

    def test_self_cosine_is_one(self):
        # Single attribute, single event with 3 pitches.
        p_attr = [np.array([[0.0, 400.0, 700.0]]).T]   # shape (3, 1)
        s = cos_sim_exp_tens(
            p_attr, None, p_attr, None,
            [15.0], [2], [False], [True], [1200.0],
            verbose=False,
        )
        assert s == pytest.approx(1.0)

    def test_matches_density_call(self):
        """Raw MA input matches building MA densities then comparing."""
        p_attr1 = [np.array([[0.0, 400.0, 700.0]]).T]
        p_attr2 = [np.array([[0.0, 300.0, 700.0]]).T]
        s_raw = cos_sim_exp_tens(
            p_attr1, None, p_attr2, None,
            [15.0], [2], [False], [True], [1200.0],
            verbose=False,
        )
        d1 = build_exp_tens(
            p_attr1, None, [15.0], [2], [False], [True], [1200.0],
            verbose=False,
        )
        d2 = build_exp_tens(
            p_attr2, None, [15.0], [2], [False], [True], [1200.0],
            verbose=False,
        )
        s_dens = cos_sim_exp_tens(d1, d2, verbose=False)
        assert s_raw == pytest.approx(s_dens, abs=1e-14)


# ---------------------------------------------------------------------
# Cross-form consistency
# ---------------------------------------------------------------------


class TestCrossFormConsistency:
    """Numerical results agree across input forms."""

    def test_raw_scalar_matches_density_matches_batch(self):
        """The same chord pair gives the same cosine via three input forms."""
        p1 = [0.0, 400.0, 700.0]
        p2 = [0.0, 300.0, 700.0]
        sigma, r, period = 15.0, 2, 1200.0

        # Form 1: density mode
        d1 = build_exp_tens(p1, None, sigma, r, False, True, period, verbose=False)
        d2 = build_exp_tens(p2, None, sigma, r, False, True, period, verbose=False)
        s_dens = cos_sim_exp_tens(d1, d2, verbose=False)

        # Form 2: raw single-multiset scalar
        s_raw_scalar = cos_sim_exp_tens(
            p1, None, p2, None, sigma, r, False, True, period, verbose=False,
        )

        # Form 3: raw single-attribute batched (single row)
        P1 = np.array([p1])
        P2 = np.array([p2])
        s_raw_batch = cos_sim_exp_tens(
            P1, None, P2, None, sigma, r, False, True, period, verbose=False,
        )

        assert s_dens == pytest.approx(s_raw_scalar, abs=1e-14)
        assert s_dens == pytest.approx(float(s_raw_batch[0]), abs=1e-14)


# ---------------------------------------------------------------------
# Deprecated shims still work
# ---------------------------------------------------------------------


class TestDeprecatedShims:
    """The deprecated wrappers still produce correct results, with warnings."""

    def test_cos_sim_exp_tens_raw_emits_deprecation_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mpt.cos_sim_exp_tens_raw(
                [0.0, 400.0, 700.0], None, [0.0, 300.0, 700.0], None,
                15.0, 2, False, True, 1200.0, verbose=False,
            )
            assert any(issubclass(wi.category, DeprecationWarning) for wi in w)

    def test_batch_cos_sim_exp_tens_emits_deprecation_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            P1 = np.array([[0.0, 400.0, 700.0]])
            P2 = np.array([[0.0, 300.0, 700.0]])
            mpt.batch_cos_sim_exp_tens(
                P1, P2, 15.0, 2, False, True, 1200.0, verbose=False,
            )
            assert any(issubclass(wi.category, DeprecationWarning) for wi in w)

    def test_cos_sim_exp_tens_raw_still_correct(self):
        """The shim produces the same result as the unified entry."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            s_shim = mpt.cos_sim_exp_tens_raw(
                [0.0, 400.0, 700.0], None, [0.0, 300.0, 700.0], None,
                15.0, 2, False, True, 1200.0, verbose=False,
            )
        s_unified = cos_sim_exp_tens(
            [0.0, 400.0, 700.0], None, [0.0, 300.0, 700.0], None,
            15.0, 2, False, True, 1200.0, verbose=False,
        )
        assert s_shim == pytest.approx(s_unified, abs=1e-14)


class TestRawSABroadcast:
    """v3+ broadcasting in batched-raw mode.

    When one of P1, P2 is a vector of length K (1-D, ``(1, K)``,
    or ``(K, 1)``) and the other is an ``(M, K)`` matrix with
    M > 1, the vector is broadcast across the matrix's M rows in
    NumPy implicit-expansion style. Weights, if not None, are
    broadcast in lockstep.
    """

    @staticmethod
    def _ref_explicit():
        ref = np.array([0.0, 386.31, 701.96])
        candidates = np.array([
            [0.0, 400.0, 700.0],
            [0.0, 300.0, 700.0],
            [0.0, 300.0, 600.0],
            [0.0, 400.0, 800.0],
        ])
        # Equivalent paired form via np.tile
        sims_explicit = cos_sim_exp_tens(
            np.tile(ref, (4, 1)), None, candidates, None,
            10.0, 1, False, True, 1200.0, verbose=False,
        )
        return ref, candidates, sims_explicit

    def test_p1_1d_broadcasts(self):
        ref, candidates, sims_explicit = self._ref_explicit()
        sims = cos_sim_exp_tens(
            ref, None, candidates, None,
            10.0, 1, False, True, 1200.0, verbose=False,
        )
        assert sims.shape == (4,)
        np.testing.assert_allclose(sims, sims_explicit, atol=1e-12)

    def test_p1_row_matrix_broadcasts(self):
        ref, candidates, sims_explicit = self._ref_explicit()
        sims = cos_sim_exp_tens(
            ref.reshape(1, -1), None, candidates, None,
            10.0, 1, False, True, 1200.0, verbose=False,
        )
        assert sims.shape == (4,)
        np.testing.assert_allclose(sims, sims_explicit, atol=1e-12)

    def test_p2_vector_broadcasts(self):
        """Symmetric case: P2 is the vector reference."""
        ref, candidates, sims_explicit = self._ref_explicit()
        sims = cos_sim_exp_tens(
            candidates, None, ref, None,
            10.0, 1, False, True, 1200.0, verbose=False,
        )
        np.testing.assert_allclose(sims, sims_explicit, atol=1e-12)

    def test_weights_broadcast_in_lockstep(self):
        ref = np.array([0.0, 386.31, 701.96])
        ref_w = np.array([1.0, 0.8, 0.6])
        candidates = np.array([
            [0.0, 400.0, 700.0],
            [0.0, 300.0, 700.0],
            [0.0, 300.0, 600.0],
        ])
        sims_bcast = cos_sim_exp_tens(
            ref, ref_w, candidates, None,
            10.0, 1, False, True, 1200.0, verbose=False,
        )
        sims_explicit = cos_sim_exp_tens(
            np.tile(ref, (3, 1)), np.tile(ref_w, (3, 1)),
            candidates, None,
            10.0, 1, False, True, 1200.0, verbose=False,
        )
        np.testing.assert_allclose(sims_bcast, sims_explicit, atol=1e-12)

    def test_broadcast_composes_with_spectrum(self):
        ref = np.array([0.0, 386.31, 701.96])
        candidates = np.array([
            [0.0, 400.0, 700.0],
            [0.0, 300.0, 700.0],
            [0.0, 300.0, 600.0],
        ])
        spec = ["harmonic", 12, "powerlaw", 1]
        sims_bcast = cos_sim_exp_tens(
            ref, None, candidates, None,
            10.0, 1, False, True, 1200.0,
            spectrum=spec, verbose=False,
        )
        sims_explicit = cos_sim_exp_tens(
            np.tile(ref, (3, 1)), None, candidates, None,
            10.0, 1, False, True, 1200.0,
            spectrum=spec, verbose=False,
        )
        np.testing.assert_allclose(sims_bcast, sims_explicit, atol=1e-12)

    def test_mismatched_rows_errors_clearly(self):
        with pytest.raises(ValueError, match="matching row counts"):
            cos_sim_exp_tens(
                np.zeros((4, 3)), None, np.zeros((5, 3)), None,
                10.0, 1, False, True, 1200.0, verbose=False,
            )

    def test_single_pair_via_1d_inputs_is_still_scalar(self):
        """Both operands 1-D → existing scalar single-multiset path; returns float."""
        s = cos_sim_exp_tens(
            np.array([0.0, 400.0, 700.0]), None,
            np.array([0.0, 300.0, 700.0]), None,
            10.0, 1, False, True, 1200.0, verbose=False,
        )
        assert isinstance(s, float)
