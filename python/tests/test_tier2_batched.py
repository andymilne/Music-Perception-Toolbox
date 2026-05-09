"""Tests for v2.1 batched dispatch in Tier-2 structural functions:
``coherence``, ``sameness``, ``n_tuple_entropy``.

All three accept a 2-D ``(M, K)`` matrix in addition to 1-D ``p`` and
return per-row arrays for scalar outputs (and lists for variable-shape
outputs like ``n_tuple_entropy``'s ``tuples``).

Per-row dedup keys differ across the three:

- ``coherence`` and ``sameness`` use the **necklace canonical form**
  of the cyclic adjacent intervals — collapses permutation, period,
  and transposition symmetries onto a single cached result. Both
  functions are fully transposition-invariant on the circle.
- ``n_tuple_entropy`` uses sorted-modular only (permutation + period).
  Its H is transposition-invariant but its ``tuples`` output reflects
  cyclic order from sort-min and so differs across transpositions.
"""

import numpy as np
import pytest

from mpt import coherence, n_tuple_entropy, sameness


class TestCoherenceBatched:
    def test_scalar_unchanged(self):
        c, nc = coherence([0, 4, 7], 12)
        assert isinstance(c, float) and isinstance(nc, float)

    def test_batched_returns_two_arrays(self):
        P = np.array([[0, 4, 7], [0, 3, 7]])
        c, nc = coherence(P, 12)
        assert c.shape == (2,) and nc.shape == (2,)

    def test_batched_matches_scalar(self):
        # Pad to common K
        P_pad = np.full((3, 4), np.nan)
        P_pad[0, :3] = [0, 4, 7]
        P_pad[1, :3] = [0, 3, 7]
        P_pad[2, :4] = [0, 1, 5, 6]
        c, nc = coherence(P_pad, 12)
        for i, row in enumerate([[0, 4, 7], [0, 3, 7], [0, 1, 5, 6]]):
            c_s, nc_s = coherence(row, 12)
            assert abs(c[i] - c_s) < 1e-12
            assert abs(nc[i] - nc_s) < 1e-12

    def test_batched_dedup_permutation(self):
        P = np.array([[0, 4, 7], [4, 0, 7]])
        c, _ = coherence(P, 12)
        assert c[0] == c[1]

    def test_batched_dedup_period_equivalent(self):
        P = np.array([[0, 4, 7], [0, 4, 19]])  # 19 mod 12 = 7
        c, _ = coherence(P, 12)
        assert c[0] == c[1]

    def test_batched_nan_padded(self):
        P = np.array([
            [0, 4, 7, np.nan],
            [0, 1, 5, 6],
            [np.nan, np.nan, np.nan, np.nan],
        ])
        c, nc = coherence(P, 12)
        assert not np.isnan(c[0])
        assert not np.isnan(c[1])
        assert np.isnan(c[2])

    def test_batched_with_sigma(self):
        P = np.array([[0, 4, 7], [0, 3, 7]])
        c, _ = coherence(P, 12, 0.25, sigma_space="position")
        assert all(0 <= v <= 1 for v in c)


class TestSamenessBatched:
    def test_batched_returns_two_arrays(self):
        P = np.array([[0, 4, 7], [0, 3, 7]])
        sq, nd = sameness(P, 12)
        assert sq.shape == (2,) and nd.shape == (2,)

    def test_batched_matches_scalar(self):
        P = np.array([[0, 4, 7], [0, 3, 7]])
        sq, nd = sameness(P, 12)
        for i in range(2):
            sq_s, nd_s = sameness(P[i], 12)
            assert abs(sq[i] - sq_s) < 1e-12
            assert abs(nd[i] - nd_s) < 1e-12

    def test_batched_dedup_permutation(self):
        P = np.array([[0, 4, 7], [7, 4, 0]])
        sq, _ = sameness(P, 12)
        assert sq[0] == sq[1]


class TestNTupleEntropyBatched:
    def test_batched_returns_array_and_list(self):
        P = np.array([[0, 4, 7], [0, 3, 7]])
        H, tuples = n_tuple_entropy(P, 12, 1)
        assert H.shape == (2,)
        assert isinstance(tuples, list)
        assert len(tuples) == 2

    def test_batched_matches_scalar(self):
        P = np.array([[0, 4, 7], [0, 3, 7]])
        H, tuples = n_tuple_entropy(P, 12, 1)
        for i in range(2):
            H_s, t_s = n_tuple_entropy(P[i], 12, 1)
            assert abs(H[i] - H_s) < 1e-12
            assert np.array_equal(tuples[i], t_s)

    def test_batched_dedup(self):
        P = np.array([[0, 4, 7], [4, 7, 0]])  # permutation
        H, _ = n_tuple_entropy(P, 12, 1)
        assert H[0] == H[1]

    def test_batched_nan_padded(self):
        P = np.array([
            [0, 4, 7, np.nan],
            [0, 3, 6, 9],
            [np.nan, np.nan, np.nan, np.nan],
        ])
        H, tuples = n_tuple_entropy(P, 12, 1)
        assert not np.isnan(H[0])
        assert not np.isnan(H[1])
        assert np.isnan(H[2])
        assert tuples[2].size == 0

    def test_batched_n_too_large_propagates_error(self):
        # Per-row K=2; n must be <= K-1 = 1. With n=2, scalar errors.
        # Batched should propagate that error from the first violating row.
        P = np.array([[0, 4], [0, 3]])
        with pytest.raises(ValueError, match="n must not exceed"):
            n_tuple_entropy(P, 12, 2)


class TestTranspositionDedup:
    """coherence and sameness dedup over transposition via necklace key.

    n_tuple_entropy does not (and these tests pin that down).
    """

    def test_coherence_all_major_triad_transpositions_dedup(self):
        # 12 transpositions of {0, 4, 7} all collapse to one cache entry,
        # all give the same coherence value.
        rows = [[(x + s) % 12 for x in [0, 4, 7]] for s in range(12)]
        P = np.array(rows)
        c, nc = coherence(P, 12)
        assert np.all(c == c[0])
        assert np.all(nc == nc[0])

    def test_coherence_major_vs_minor_distinct(self):
        # Major and minor triads have opposite chiralities — different
        # necklace forms, so different cache keys, but they happen to
        # be equally coherent (both perfectly proper).
        major = [[(x + s) % 12 for x in [0, 4, 7]] for s in range(12)]
        minor = [[(x + s) % 12 for x in [0, 3, 7]] for s in range(12)]
        P = np.array(major + minor)
        c, _ = coherence(P, 12)
        # All same value (both triads are coherent), but the cache
        # would have stored two distinct keys.
        assert np.all(c == c[0])

    def test_coherence_transposition_with_failures(self):
        # Use a scale that has coherence failures, so transposition dedup
        # is checked on a non-trivial value.
        rows = [[(x + s) % 12 for x in [0, 1, 5, 6]] for s in range(12)]
        P = np.array(rows)
        c, nc = coherence(P, 12)
        assert np.all(c == c[0])
        assert nc[0] == 5.0  # known value for {0, 1, 5, 6}

    def test_sameness_all_major_triad_transpositions_dedup(self):
        rows = [[(x + s) % 12 for x in [0, 4, 7]] for s in range(12)]
        P = np.array(rows)
        sq, nd = sameness(P, 12)
        assert np.all(sq == sq[0])
        assert np.all(nd == nd[0])

    def test_n_tuple_entropy_transposition_does_not_dedup_tuples(self):
        # H is transposition-invariant; tuples is NOT (in general).
        # Pick transpositions whose sort-min lands at different cyclic
        # positions: {0,4,7} sorts as [0,4,7]; transposing by 7 gives
        # {7,11,2} sorted [2,7,11], a cyclic rotation that yields
        # tuples [5,4,3] instead of [4,3,5].
        P = np.array([[0, 4, 7], [2, 7, 11]])
        H, tuples = n_tuple_entropy(P, 12, 1)
        assert H[0] == H[1]                       # H matches
        assert not np.array_equal(tuples[0], tuples[1])  # tuples differ in cyclic order
