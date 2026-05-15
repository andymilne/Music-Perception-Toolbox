"""Tests for serial-position feature: seq_weights.

Mirror of MATLAB tests/test_seq_weights.m.
"""
import numpy as np
import pytest
import warnings

import mpt
from mpt._utils import position_variance


class TestSeqWeights:
    def test_primacy(self):
        v = mpt.seq_weights(None, "primacy", n=5)
        assert v.shape == (5,) and v[0] == 1.0 and np.all(v[1:] == 0.0)

    def test_recency(self):
        v = mpt.seq_weights(None, "recency", n=5)
        assert v.shape == (5,) and v[-1] == 1.0 and np.all(v[:-1] == 0.0)

    def test_exp_zero_decay_uniform(self):
        v = mpt.seq_weights(None, "exponentialFromEnd", n=5, decay_rate=0.0)
        np.testing.assert_allclose(v, np.ones(5))

    def test_exp_from_start_shape(self):
        v = mpt.seq_weights(None, "exponentialFromStart", n=5, decay_rate=1.0)
        np.testing.assert_allclose(v, np.exp(-np.arange(5, dtype=float)))

    def test_exp_from_end_shape(self):
        v = mpt.seq_weights(None, "exponentialFromEnd", n=5, decay_rate=1.0)
        np.testing.assert_allclose(v, np.exp(-np.arange(4, -1, -1, dtype=float)))

    def test_ushape_symmetric(self):
        v = mpt.seq_weights(None, "uShape", n=7, decay_rate=0.5, alpha=0.5)
        np.testing.assert_allclose(v, v[::-1])

    def test_ushape_alpha_limits(self):
        v_s = mpt.seq_weights(None, "exponentialFromStart", n=5, decay_rate=0.5)
        v_e = mpt.seq_weights(None, "exponentialFromEnd", n=5, decay_rate=0.5)
        np.testing.assert_allclose(
            mpt.seq_weights(None, "uShape", n=5, decay_rate=0.5, alpha=1.0), v_s
        )
        np.testing.assert_allclose(
            mpt.seq_weights(None, "uShape", n=5, decay_rate=0.5, alpha=0.0), v_e
        )

    def test_explicit_passthrough(self):
        profile = [0.1, 0.2, 0.4, 0.2, 0.1]
        np.testing.assert_allclose(mpt.seq_weights(None, profile, n=5), profile)

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="length"):
            mpt.seq_weights(None, [0.1, 0.2, 0.3], n=5)

    def test_unit_t_matches_unit_spacing(self):
        v1 = mpt.seq_weights(None, "exponentialFromEnd", n=5, decay_rate=1.0)
        v2 = mpt.seq_weights(None, "exponentialFromEnd", n=5, decay_rate=1.0,
                              t=[1, 2, 3, 4, 5])
        np.testing.assert_allclose(v1, v2)

    def test_non_increasing_t_errors(self):
        with pytest.raises(ValueError, match="increasing"):
            mpt.seq_weights(None, "exponentialFromEnd", n=5, t=[1, 2, 2, 3, 4])

    def test_unknown_spec(self):
        with pytest.raises(ValueError, match="Unknown"):
            mpt.seq_weights(None, "wibble", n=5)

    def test_negative_decay(self):
        with pytest.raises(ValueError, match="decay_rate"):
            mpt.seq_weights(None, "exponentialFromEnd", n=5, decay_rate=-1.0)

    def test_alpha_out_of_range(self):
        with pytest.raises(ValueError, match="alpha"):
            mpt.seq_weights(None, "uShape", n=5, decay_rate=1.0, alpha=1.5)

    def test_w_uniform_via_none(self):
        # None for w is equivalent to all ones
        v_none = mpt.seq_weights(None, "exponentialFromEnd", n=5,
                                  decay_rate=0.5)
        v_ones = mpt.seq_weights(np.ones(5), "exponentialFromEnd",
                                  decay_rate=0.5)
        np.testing.assert_allclose(v_none, v_ones)

    def test_w_multiplies_profile(self):
        w = np.array([0.8, 0.5, 1.0, 0.3, 0.9])
        v = mpt.seq_weights(w, "recency")
        # Recency profile is [0, 0, 0, 0, 1]; product picks w[-1]
        expected = np.array([0.0, 0.0, 0.0, 0.0, 0.9])
        np.testing.assert_allclose(v, expected)

    def test_w_multiplies_explicit_profile(self):
        w = np.array([2.0, 2.0, 2.0])
        profile = np.array([0.1, 0.5, 0.4])
        v = mpt.seq_weights(w, profile)
        np.testing.assert_allclose(v, 2.0 * profile)

    def test_w_length_mismatch_errors(self):
        # Explicit n conflicts with length of w
        with pytest.raises(ValueError, match="does not match"):
            mpt.seq_weights(np.array([1.0, 2.0, 3.0]), "flat", n=5)

    def test_missing_n_with_none_w(self):
        with pytest.raises(ValueError, match="n must be supplied"):
            mpt.seq_weights(None, "flat")

    def test_missing_n_with_scalar_w(self):
        with pytest.raises(ValueError, match="n must be supplied"):
            mpt.seq_weights(0.5, "flat")

    def test_scalar_w_broadcasts(self):
        v = mpt.seq_weights(0.5, "flat", n=4)
        np.testing.assert_allclose(v, 0.5 * np.ones(4))

    def test_n_inferred_from_w_matches_explicit_n(self):
        w = np.array([0.2, 0.8, 0.5])
        v_inferred = mpt.seq_weights(w, "recency")
        v_explicit = mpt.seq_weights(w, "recency", n=3)
        np.testing.assert_allclose(v_inferred, v_explicit)


# ===================================================================
#  multi-attribute expectation tensor (MAET)
# ===================================================================
