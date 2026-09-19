"""Tests for the shape rule that selects batched-raw mode.

Python dispatches on ``ndim``: a 2-D array is a batch whatever its
second dimension, so ``(M, 1)`` is M rows of one element each. MATLAB
dispatches on a matrix with both dimensions greater than one, so an
M-by-1 column is a vector there and is broadcast as one shared
multiset. The two rules are each idiomatic in their own language and
are documented as a deliberate divergence in the entry-point
docstrings; these tests pin the Python side of it, and pin the
NaN-padded spelling that reads the same way in both languages. The
MATLAB twin is ``tests/test_batched_shape_rule.m``.
"""

import numpy as np
import pytest

import mpt


P = 1200.0
SIGMA = 10.0
SCALE = np.array([0.0, 200.0, 400.0, 500.0, 700.0, 900.0, 1100.0])
PROBES = np.array([0.0, 100.0, 200.0, 300.0])


def _context(n):
    return np.tile(SCALE, (n, 1))


class TestColumnVectorIsABatch:

    def test_column_of_probes_gives_one_result_per_row(self):
        n = PROBES.size
        s = mpt.sim_maet(_context(n), None, PROBES.reshape(n, 1), None,
                         SIGMA, 1, False, True, P, verbose=False)
        assert np.asarray(s).shape == (n,)

    def test_the_results_are_not_all_equal(self):
        """The diagnostic that caught the divergence: broadcasting the
        column as one shared multiset would make the profile flat."""
        n = PROBES.size
        s = np.asarray(mpt.sim_maet(_context(n), None,
                                    PROBES.reshape(n, 1), None,
                                    SIGMA, 1, False, True, P,
                                    verbose=False))
        assert np.ptp(s) > 1e-3

    def test_row_vector_broadcasts_instead(self):
        """A 1-D operand, or a ``(1, K)`` one, is the broadcast form."""
        n = 3
        one = np.array([0.0, 400.0, 700.0])
        flat = mpt.sim_maet(_context(n), None, one, None,
                            SIGMA, 1, False, True, P, verbose=False)
        two_d = mpt.sim_maet(_context(n), None, one.reshape(1, 3), None,
                             SIGMA, 1, False, True, P, verbose=False)
        assert np.allclose(np.asarray(flat), np.asarray(two_d))
        assert np.asarray(flat).shape == (n,)


class TestNanPaddedSpelling:

    def test_padding_agrees_with_the_bare_column(self):
        """The spelling that carries across to MATLAB: pad to two
        columns with NaN. The padding is stripped per row, so the
        values are those of the one-element rows."""
        n = PROBES.size
        bare = mpt.sim_maet(_context(n), None, PROBES.reshape(n, 1), None,
                            SIGMA, 1, False, True, P, verbose=False)
        padded = mpt.sim_maet(
            _context(n), None,
            np.column_stack([PROBES, np.full(n, np.nan)]), None,
            SIGMA, 1, False, True, P, verbose=False)
        assert np.allclose(np.asarray(bare), np.asarray(padded))

    def test_padded_rows_match_probe_by_probe_calls(self):
        n = PROBES.size
        padded = np.asarray(mpt.sim_maet(
            _context(n), None,
            np.column_stack([PROBES, np.full(n, np.nan)]), None,
            SIGMA, 1, False, True, P, verbose=False))
        one_by_one = np.array([
            mpt.sim_maet(SCALE, None, np.array([x]), None,
                         SIGMA, 1, False, True, P, verbose=False)
            for x in PROBES])
        assert np.allclose(padded, one_by_one)

    def test_padding_survives_the_spectrum_keyword(self):
        n = PROBES.size
        spectrum = ['harmonic', 8, 'powerlaw', 1]
        bare = mpt.sim_maet(_context(n), None, PROBES.reshape(n, 1), None,
                            SIGMA, 1, False, True, P,
                            spectrum=spectrum, verbose=False)
        padded = mpt.sim_maet(
            _context(n), None,
            np.column_stack([PROBES, np.full(n, np.nan)]), None,
            SIGMA, 1, False, True, P,
            spectrum=spectrum, verbose=False)
        assert np.allclose(np.asarray(bare), np.asarray(padded))


class TestEvalAndEntropy:

    def test_eval_reads_a_column_as_rows(self):
        vals = mpt.eval_maet(PROBES.reshape(-1, 1), None, SIGMA, 1,
                             False, True, P, np.array([[0.0]]),
                             verbose=False)
        assert np.asarray(vals).shape == (PROBES.size, 1)

    def test_entropy_reads_a_column_as_rows(self):
        H = mpt.entropy_maet(PROBES.reshape(-1, 1), None, SIGMA, 1,
                             False, True, P, n_points_per_dim=120,
                             verbose=False)
        assert np.asarray(H).shape == (PROBES.size,)
