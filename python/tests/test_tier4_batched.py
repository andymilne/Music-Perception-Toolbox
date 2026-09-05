"""Tests for v3 Tier-4 (Monte-Carlo) batched dispatch:
``balance`` and ``evenness``.

The new wrinkle vs Tiers 1/2 is RNG handling. The ``rng_scope``
argument controls how each row's seed is derived from the base
``rng_seed``:

- ``'canonical'`` (default): per-row seed = ``base + fnv1a32(canonical_key)``.
  Canonical-form-equivalent rows get identical seeds and identical
  Monte-Carlo realisations, so dedup works for ``sigma > 0``.
- ``'row'``: per-row seed = ``base + row_index``. Each row gets an
  independent reproducible realisation; dedup is disabled.

When ``rng_seed`` is None in batched mode, a session-random base is
generated once per call so within-call dedup is reproducible while
across-call results differ.
"""

import numpy as np
import pytest

from mpt import balance, evenness


# ---------------------------------------------------------------------
# sigma = 0 (deterministic)
# ---------------------------------------------------------------------


class TestBalanceSigmaZero:
    def test_scalar_unchanged(self):
        b = balance([0, 4, 7], None, 12)
        assert isinstance(b, float)

    def test_batched_returns_array(self):
        P = np.array([[0, 4, 7], [0, 3, 7]])
        b = balance(P, None, 12)
        assert b.shape == (2,)

    def test_batched_matches_scalar(self):
        P = np.array([[0, 4, 7], [0, 3, 7]])
        b = balance(P, None, 12)
        assert abs(b[0] - balance([0, 4, 7], None, 12)) < 1e-12

    def test_batched_dedup_permutation(self):
        P = np.array([[0, 4, 7], [4, 0, 7], [7, 4, 0]])
        b = balance(P, None, 12)
        assert b[0] == b[1] == b[2]

    def test_batched_with_return_std(self):
        P = np.array([[0, 4, 7], [0, 3, 7]])
        b, bs = balance(P, None, 12, return_std=True)
        assert b.shape == bs.shape == (2,)
        assert np.all(bs == 0.0)  # sigma=0 -> std is 0

    def test_batched_nan_padded(self):
        P = np.array([[0, 4, 7, np.nan], [0, 1, 5, 6], [np.nan]*4])
        b = balance(P, None, 12)
        assert not np.isnan(b[0]) and not np.isnan(b[1]) and np.isnan(b[2])


class TestEvennessSigmaZero:
    def test_batched_returns_array(self):
        P = np.array([[0, 4, 7], [0, 3, 7]])
        e = evenness(P, 12)
        assert e.shape == (2,)

    def test_batched_dedup(self):
        P = np.array([[0, 4, 7], [4, 0, 7]])
        e = evenness(P, 12)
        assert e[0] == e[1]


# ---------------------------------------------------------------------
# sigma > 0 (Monte Carlo) — rng_scope contract
# ---------------------------------------------------------------------


class TestBalanceMonteCarloCanonicalScope:
    """Default rng_scope='canonical': identical canonical keys give
    identical MC realisations within a single call."""

    def test_identical_inputs_identical_results(self):
        P = np.array([[0, 4, 7], [4, 0, 7], [0, 4, 7]])
        b = balance(P, None, 12, sigma=0.3, rng_seed=42)
        assert b[0] == b[1] == b[2]

    def test_different_canonicals_different_results(self):
        # Major and minor triads are distinct multisets — different
        # canonical keys -> different derived seeds -> different MC.
        P = np.array([[0, 4, 7], [0, 3, 7]])
        b = balance(P, None, 12, sigma=0.3, rng_seed=42)
        assert b[0] != b[1]

    def test_reproducible_across_calls(self):
        P = np.array([[0, 4, 7], [0, 3, 7]])
        b1 = balance(P, None, 12, sigma=0.3, rng_seed=42)
        b2 = balance(P, None, 12, sigma=0.3, rng_seed=42)
        assert np.array_equal(b1, b2)

    def test_different_seeds_different_results(self):
        P = np.array([[0, 4, 7], [0, 3, 7]])
        b1 = balance(P, None, 12, sigma=0.3, rng_seed=42)
        b2 = balance(P, None, 12, sigma=0.3, rng_seed=123)
        assert not np.array_equal(b1, b2)

    def test_rng_seed_none_within_call_dedup(self):
        # Within one call, dedup must work even without explicit seed.
        P = np.array([[0, 4, 7], [4, 7, 0], [0, 4, 7]])
        b = balance(P, None, 12, sigma=0.3)  # rng_seed=None
        assert b[0] == b[1] == b[2]

    def test_rng_seed_none_across_calls_differs(self):
        # Across calls without explicit seed, results should differ
        # (session-random base).
        P = np.array([[0, 4, 7]])
        b1 = balance(P, None, 12, sigma=0.3)
        b2 = balance(P, None, 12, sigma=0.3)
        # Vanishingly small chance of equality by coincidence.
        assert b1[0] != b2[0]


class TestBalanceMonteCarloRowScope:
    """rng_scope='row': identical inputs get DIFFERENT realisations."""

    def test_identical_inputs_different_results(self):
        P = np.array([[0, 4, 7], [0, 4, 7], [0, 4, 7]])
        b = balance(P, None, 12, sigma=0.3, rng_seed=42, rng_scope="row")
        # All three rows are the same scale but get distinct seeds.
        assert b[0] != b[1] and b[1] != b[2] and b[0] != b[2]

    def test_reproducible_across_calls(self):
        P = np.array([[0, 4, 7], [0, 4, 7]])
        b1 = balance(P, None, 12, sigma=0.3, rng_seed=42, rng_scope="row")
        b2 = balance(P, None, 12, sigma=0.3, rng_seed=42, rng_scope="row")
        assert np.array_equal(b1, b2)


class TestEvennessMonteCarloCanonical:
    """Spot-check evenness uses the same RNG infrastructure."""

    def test_dedup_canonical(self):
        P = np.array([[0, 4, 7], [4, 0, 7]])
        e = evenness(P, 12, sigma=0.3, rng_seed=42)
        assert e[0] == e[1]

    def test_row_independent(self):
        P = np.array([[0, 4, 7], [0, 4, 7]])
        e = evenness(P, 12, sigma=0.3, rng_seed=42, rng_scope="row")
        assert e[0] != e[1]


class TestRngScopeValidation:
    def test_invalid_rng_scope_raises(self):
        P = np.array([[0, 4, 7], [0, 3, 7]])
        with pytest.raises(ValueError, match="rng_scope"):
            balance(P, None, 12, rng_scope="invalid")
        with pytest.raises(ValueError, match="rng_scope"):
            evenness(P, 12, rng_scope="invalid")
