"""Tests for simplex_vertices.

Mirror of MATLAB tests/test_simplex_vertices.m.
"""
import numpy as np
import pytest
import warnings

import mpt
from mpt._utils import position_variance


class TestSimplexVertices:
    def test_shapes(self):
        for N in [2, 3, 4, 5, 10]:
            V = mpt.simplex_vertices(N)
            assert V.shape == (N, N - 1)

    def test_centroid_at_origin(self):
        for N in [2, 3, 4, 5, 7]:
            V = mpt.simplex_vertices(N)
            np.testing.assert_allclose(
                V.mean(axis=0), np.zeros(N - 1), atol=1e-12
            )

    def test_edge_length_default_unit(self):
        for N in [2, 3, 4, 5, 7]:
            V = mpt.simplex_vertices(N)
            # All pairwise distances should equal 1.
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.linalg.norm(V[i] - V[j])
                    assert d == pytest.approx(1.0, abs=1e-12)

    def test_edge_length_custom(self):
        for L in [0.5, 2.0, 100.0]:
            V = mpt.simplex_vertices(4, edge_length=L)
            d = np.linalg.norm(V[0] - V[1])
            assert d == pytest.approx(L, abs=1e-12)

    def test_n_2_collapses_to_1d(self):
        V = mpt.simplex_vertices(2)
        # Two points at +/- 0.5 (so distance 1).
        assert V.shape == (2, 1)
        assert abs(V[0, 0] - V[1, 0]) == pytest.approx(1.0, abs=1e-12)
        assert V.mean() == pytest.approx(0.0, abs=1e-12)

    def test_n_too_small(self):
        with pytest.raises(ValueError, match="N must be"):
            mpt.simplex_vertices(1)
        with pytest.raises(ValueError, match="N must be"):
            mpt.simplex_vertices(0)

    def test_negative_edge_length(self):
        with pytest.raises(ValueError, match="edge_length"):
            mpt.simplex_vertices(3, edge_length=-1)

    def test_zero_edge_length(self):
        with pytest.raises(ValueError, match="edge_length"):
            mpt.simplex_vertices(3, edge_length=0)
