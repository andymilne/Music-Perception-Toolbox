"""Tests for v3 batched dispatch in circular measures.

Bundle 3 (Tier 1): the DFT-equivariant family — ``dft_circular``,
``mean_offset``, ``edges``, ``proj_centroid``, ``circ_apm``. All five
share the same canonical-form dedup pattern (sort modular pitches +
matching weights), with permutation and period symmetries collapsed
onto one cached result per row. Transposition dedup is *not* applied
at this stage.
"""

import numpy as np
import pytest

from mpt import circ_apm, dft_circular, edges, mean_offset, proj_centroid


class TestDftCircularBatched:
    """``dft_circular`` accepts 2-D ``(M, K)`` input and returns
    length-``M`` lists of ``(F, mag)`` per row, with permutation +
    period dedup but not transposition dedup.
    """

    def test_scalar_1d_unchanged(self):
        # v2.0 contract: 1-D in, two ndarrays out.
        F, mag = dft_circular([0, 200, 400], None, 1200)
        assert F.shape == (3,)
        assert mag.shape == (3,)

    def test_batched_returns_lists(self):
        P = np.array([
            [0, 200, 400],
            [0, 100, 200],
        ])
        F_list, mag_list = dft_circular(P, None, 1200)
        assert isinstance(F_list, list)
        assert isinstance(mag_list, list)
        assert len(F_list) == 2
        assert len(mag_list) == 2

    def test_batched_matches_scalar_per_row(self):
        # Pad to common K with NaN for batching
        P_padded = np.full((3, 7), np.nan)
        P_padded[0, :7] = [0, 200, 400, 500, 700, 900, 1100]   # major
        P_padded[1, :7] = [0, 100, 300, 500, 700, 900, 1000]   # something else
        P_padded[2, :5] = [0, 100, 200, 300, 400]              # different K
        F_list, mag_list = dft_circular(P_padded, None, 1200)
        for i in range(3):
            mask = ~np.isnan(P_padded[i])
            row = P_padded[i, mask]
            F_scalar, mag_scalar = dft_circular(row, None, 1200)
            assert np.allclose(F_list[i], F_scalar)
            assert np.allclose(mag_list[i], mag_scalar)

    def test_batched_dedup_permutation(self):
        # Two rows with same multiset content (different order):
        # canonical sort should make them dedup.
        P = np.array([
            [0, 200, 400],
            [400, 0, 200],   # permutation of row 0
        ])
        F_list, mag_list = dft_circular(P, None, 1200)
        assert np.array_equal(F_list[0], F_list[1])
        assert np.array_equal(mag_list[0], mag_list[1])

    def test_batched_dedup_period_equivalent(self):
        # Pitches outside [0, period) get reduced mod period.
        P = np.array([
            [0, 200, 400],
            [0, 200, 1600],   # 1600 mod 1200 = 400 -> same multiset
        ])
        F_list, mag_list = dft_circular(P, None, 1200)
        assert np.array_equal(F_list[0], F_list[1])
        assert np.array_equal(mag_list[0], mag_list[1])

    def test_batched_no_transposition_dedup(self):
        # Transposed copies are NOT dedup'd at this v3 stage —
        # outputs differ in F's phases, identical only in mag.
        P = np.array([
            [0, 200, 400],
            [100, 300, 500],   # transposition by 100
        ])
        F_list, mag_list = dft_circular(P, None, 1200)
        # Magnitudes are transposition-invariant, so they DO match.
        assert np.allclose(mag_list[0], mag_list[1])
        # But F itself differs in phase.
        assert not np.allclose(F_list[0], F_list[1])

    def test_batched_nan_padded_variable_K(self):
        P = np.array([
            [0, 200, 400, np.nan],
            [0, 100, 200, 300],
            [np.nan, np.nan, np.nan, np.nan],
        ])
        F_list, mag_list = dft_circular(P, None, 1200)
        assert len(F_list[0]) == 3
        assert len(F_list[1]) == 4
        assert F_list[2].size == 0
        assert mag_list[2].size == 0

    def test_batched_with_weight_matrix(self):
        P = np.array([[0, 200, 400], [0, 100, 200]])
        W = np.array([[1, 1, 1], [1, 2, 1]])
        F_list, mag_list = dft_circular(P, W, 1200)
        # Per-row matches scalar
        for i in range(2):
            F_s, mag_s = dft_circular(P[i], W[i], 1200)
            assert np.allclose(F_list[i], F_s)
            assert np.allclose(mag_list[i], mag_s)

    def test_batched_with_broadcast_weight_vector(self):
        P = np.array([[0, 200, 400], [0, 100, 200]])
        w_vec = np.array([1, 2, 1])
        F_list, mag_list = dft_circular(P, w_vec, 1200)
        for i in range(2):
            F_s, mag_s = dft_circular(P[i], w_vec, 1200)
            assert np.allclose(F_list[i], F_s)

    def test_batched_invalid_weight_shape_raises(self):
        P = np.array([[0, 200, 400], [0, 100, 200]])
        W_wrong = np.array([1, 2])   # length 2, but K=3
        with pytest.raises(ValueError):
            dft_circular(P, W_wrong, 1200)


# ---------------------------------------------------------------------
# Bundle 3 remainder: mean_offset, edges, proj_centroid, circ_apm
# ---------------------------------------------------------------------


class TestMeanOffsetBatched:
    """``mean_offset`` 2-D dispatch with permutation + period dedup."""

    def test_scalar_1d_unchanged(self):
        h = mean_offset([0, 4, 7], None, 12)
        assert h.shape == (12,)

    def test_batched_returns_list(self):
        from mpt import mean_offset
        P = np.array([[0, 4, 7], [0, 3, 7]])
        h_list = mean_offset(P, None, 12)
        assert isinstance(h_list, list)
        assert len(h_list) == 2
        assert h_list[0].shape == (12,)

    def test_batched_matches_scalar(self):
        from mpt import mean_offset
        P = np.array([[0, 4, 7], [0, 3, 7]])
        h_list = mean_offset(P, None, 12)
        for i in range(2):
            h_scalar = mean_offset(P[i], None, 12)
            assert np.allclose(h_list[i], h_scalar)

    def test_batched_dedup_permutation(self):
        from mpt import mean_offset
        P = np.array([[0, 4, 7], [4, 0, 7]])
        h_list = mean_offset(P, None, 12)
        assert np.array_equal(h_list[0], h_list[1])

    def test_batched_dedup_period_equivalent(self):
        from mpt import mean_offset
        P = np.array([[0, 4, 7], [0, 4, 19]])  # 19 mod 12 = 7
        h_list = mean_offset(P, None, 12)
        assert np.array_equal(h_list[0], h_list[1])

    def test_batched_nan_padded(self):
        from mpt import mean_offset
        P = np.array([
            [0, 4, 7, np.nan],
            [0, 3, 6, 9],
            [np.nan, np.nan, np.nan, np.nan],
        ])
        h_list = mean_offset(P, None, 12)
        assert h_list[0].size == 12
        assert h_list[1].size == 12
        assert h_list[2].size == 0


class TestEdgesBatched:
    """``edges`` 2-D dispatch."""

    def test_batched_returns_two_lists(self):
        from mpt import edges
        P = np.array([[0, 4, 7], [0, 3, 7]])
        e_list, es_list = edges(P, None, 12)
        assert isinstance(e_list, list)
        assert isinstance(es_list, list)
        assert len(e_list) == 2

    def test_batched_matches_scalar(self):
        from mpt import edges
        P = np.array([[0, 4, 7], [0, 3, 7]])
        e_list, es_list = edges(P, None, 12)
        for i in range(2):
            e_scalar, es_scalar = edges(P[i], None, 12)
            assert np.allclose(e_list[i], e_scalar)
            assert np.allclose(es_list[i], es_scalar)

    def test_batched_dedup(self):
        from mpt import edges
        P = np.array([[0, 4, 7], [4, 7, 0]])
        e_list, es_list = edges(P, None, 12)
        assert np.array_equal(e_list[0], e_list[1])
        assert np.array_equal(es_list[0], es_list[1])


class TestProjCentroidBatched:
    """``proj_centroid`` 2-D dispatch (3 outputs)."""

    def test_batched_returns_three_lists(self):
        from mpt import proj_centroid
        P = np.array([[0, 4, 7], [0, 3, 7]])
        y_list, cm_list, cp_list = proj_centroid(P, None, 12)
        assert len(y_list) == 2
        assert len(cm_list) == 2
        assert len(cp_list) == 2

    def test_batched_matches_scalar(self):
        from mpt import proj_centroid
        P = np.array([[0, 4, 7], [0, 3, 7]])
        y_list, cm_list, cp_list = proj_centroid(P, None, 12)
        for i in range(2):
            y_s, cm_s, cp_s = proj_centroid(P[i], None, 12)
            assert np.allclose(y_list[i], y_s)
            assert abs(cm_list[i] - cm_s) < 1e-12
            assert abs(cp_list[i] - cp_s) < 1e-12

    def test_batched_with_sigma(self):
        from mpt import proj_centroid
        P = np.array([[0, 4, 7], [0, 3, 7]])
        y_list, cm_list, cp_list = proj_centroid(P, None, 12, None, 0.5)
        assert all(cm > 0 for cm in cm_list)


class TestCircApmBatched:
    """``circ_apm`` 2-D dispatch (3 outputs, R is period×period dense)."""

    def test_batched_returns_three_lists(self):
        from mpt import circ_apm
        P = np.array([
            [0, 3, 6, 10, 12],
            [0, 2, 4, 6, 8],
        ])
        R_list, rp_list, rl_list = circ_apm(P, None, 16)
        assert len(R_list) == 2
        assert R_list[0].shape == (16, 16)

    def test_batched_matches_scalar(self):
        from mpt import circ_apm
        P = np.array([
            [0, 3, 6, 10, 12],
            [0, 2, 4, 6, 8],
        ])
        R_list, rp_list, rl_list = circ_apm(P, None, 16)
        for i in range(2):
            # Scalar: sort to canonical form, since batched dedup uses
            # canonical-form keys.
            row_sorted = np.sort(P[i] % 16)
            R_s, rp_s, rl_s = circ_apm(row_sorted, None, 16)
            assert np.array_equal(R_list[i], R_s)

    def test_batched_dedup(self):
        from mpt import circ_apm
        P = np.array([
            [0, 3, 6, 10, 12],
            [3, 12, 0, 10, 6],   # permutation of row 0
        ])
        R_list, _, _ = circ_apm(P, None, 16)
        assert np.array_equal(R_list[0], R_list[1])

    def test_batched_period_reduction(self):
        from mpt import circ_apm
        # 19 mod 16 = 3; 12 mod 16 = 12 (unchanged); etc.
        P = np.array([
            [0, 3, 6, 10, 12],
            [0, 19, 6, 10, 12],
        ])
        R_list, _, _ = circ_apm(P, None, 16)
        assert np.array_equal(R_list[0], R_list[1])

    def test_batched_non_integer_raises(self):
        from mpt import circ_apm
        P = np.array([[0.5, 1.0, 2.0]])
        with pytest.raises(ValueError, match="integer"):
            circ_apm(P, None, 16)


# ---------------------------------------------------------------------
# Regression: canonical key FP-noise bug (commit 20)
# ---------------------------------------------------------------------


class TestCanonicalKeyFloatingPointBug:
    """``_cyclic_canonical`` previously picked the wrong rotation for
    transposition-equivalent multisets when FP noise from mod-reduction
    affected the lex comparison. The fix rounds shifted values to 9
    decimal places before comparison."""

    def test_gen_chain_transposition_equivalents_share_key(self):
        from mpt.tensor import _chord_canonical_key

        n = 19
        period = 1200
        sigma = 10
        # gen=g and gen=period-g are transpositions on the circle
        # (chain shifted by -(n-1)g = chain generated by -g).
        # With the previous implementation, FP noise on non-integer
        # gens caused them to map to different canonical keys.
        for g in [100.1, 137.5, 350.7]:
            chain_g  = np.mod(np.arange(n) * g,            period)
            chain_pg = np.mod(np.arange(n) * (period - g), period)
            k_g, _, _  = _chord_canonical_key(
                chain_g,  None, sigma=sigma, r=2,
                is_rel=True, is_per=True, period=period,
            )
            k_pg, _, _ = _chord_canonical_key(
                chain_pg, None, sigma=sigma, r=2,
                is_rel=True, is_per=True, period=period,
            )
            assert k_g == k_pg, (
                f"gen={g} and gen={period - g} should share canonical key "
                f"(transposition-equivalent on the circle)"
            )

    def test_full_sweep_dedup_collapses_pairs(self):
        from mpt.tensor import _chord_canonical_key

        n = 19
        period = 1200
        sigma = 10
        gen_step = 0.1
        gens = np.arange(0, period, gen_step)

        keys = set()
        for g in gens:
            chain = np.mod(np.arange(n) * g, period)
            k, _, _ = _chord_canonical_key(
                chain, None, sigma=sigma, r=2,
                is_rel=True, is_per=True, period=period,
            )
            keys.add(k)
        # Each gen and period-gen pair collapse to one key, except
        # self-reflective gens (~3-5 in this sweep).
        assert len(keys) < 0.55 * len(gens), (
            f"Expected ~half of {len(gens)} gens to collapse; "
            f"got {len(keys)} unique keys"
        )
