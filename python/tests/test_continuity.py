"""Tests for serial-position feature: continuity.

Mirror of MATLAB tests/test_continuity.m.
"""
import numpy as np
import pytest
import warnings

import mpt
from mpt._utils import position_variance


class TestContinuity:
    def test_example_1_strict(self):
        c, m = mpt.continuity([3, 5, 7, 7, 9], [11], 0, mode="strict")
        assert c[0] == pytest.approx(1.0)
        assert m[0] == pytest.approx(2.0)

    def test_example_1_lenient(self):
        c, m = mpt.continuity([3, 5, 7, 7, 9], [11], 0, mode="lenient")
        assert c[0] == pytest.approx(3.0)
        assert m[0] == pytest.approx(6.0)

    def test_query_equals_last(self):
        c, m = mpt.continuity([3, 5, 7, 7, 9], [9], 0)
        assert c[0] == pytest.approx(0.0)
        assert m[0] == pytest.approx(0.0)

    def test_multi_query_shape(self):
        c, m = mpt.continuity([3, 5, 7, 7, 9], [10, 11, 12], 0,
                              mode="lenient")
        assert c.shape == (3,)
        assert m.shape == (3,)

    def test_seq_too_short(self):
        c, m = mpt.continuity([5], [11, 12, 13], 0)
        np.testing.assert_allclose(c, 0.0)
        np.testing.assert_allclose(m, 0.0)

    def test_descending_query_on_ascending_seq(self):
        c, m = mpt.continuity([1, 2, 3, 4, 5], [4], 0, mode="lenient")
        assert c[0] == pytest.approx(0.0)

    def test_magnitude_slope_on_arithmetic(self):
        c, m = mpt.continuity([1, 4, 7, 10, 13], [16], 0, mode="lenient")
        assert c[0] == pytest.approx(4.0)
        assert m[0] == pytest.approx(12.0)
        assert (m[0] / c[0]) == pytest.approx(3.0)

    def test_magnitude_signed_descending(self):
        c, m = mpt.continuity([10, 8, 6, 4], [2], 0, mode="strict")
        assert c[0] == pytest.approx(3.0)
        assert m[0] == pytest.approx(-6.0)

    def test_smoothing_converges(self):
        vals = [
            mpt.continuity([3, 5, 7, 7, 9], [11], s, mode="lenient")[0][0]
            for s in (5.0, 1.0, 0.1, 0.0)
        ]
        assert all(vals[i] <= vals[i + 1] + 1e-10
                   for i in range(len(vals) - 1))
        assert vals[-1] == pytest.approx(3.0)

    def test_explicit_theta_overrides_mode(self):
        c, _ = mpt.continuity([3, 5, 7, 7, 9], [11], 0, theta=-1.0)
        assert c[0] == pytest.approx(3.0)
        c, _ = mpt.continuity([3, 5, 7, 7, 9], [11], 0, theta=0.0)
        assert c[0] == pytest.approx(1.0)

    def test_theta_out_of_range(self):
        with pytest.raises(ValueError, match="theta"):
            mpt.continuity([3, 5, 7], [8], 0, theta=2.0)

    def test_bad_mode(self):
        with pytest.raises(ValueError, match="mode"):
            mpt.continuity([3, 5, 7], [8], 0, mode="wibble")

    # --- Weight-argument tests (v3) ---

    def test_w_none_equals_default(self):
        c1, m1 = mpt.continuity([3, 5, 7, 7, 9], [11], 0,
                                mode="lenient")
        c2, m2 = mpt.continuity([3, 5, 7, 7, 9], [11], 0,
                                w=None, mode="lenient")
        c3, m3 = mpt.continuity([3, 5, 7, 7, 9], [11], 0,
                                w=[], mode="lenient")
        np.testing.assert_allclose(c1, c2)
        np.testing.assert_allclose(m1, m2)
        np.testing.assert_allclose(c1, c3)
        np.testing.assert_allclose(m1, m3)

    def test_w_scalar_one_equals_unweighted(self):
        c_un, m_un = mpt.continuity([3, 5, 7, 7, 9], [11], 0,
                                    mode="lenient")
        c_w, m_w = mpt.continuity([3, 5, 7, 7, 9], [11], 0,
                                  w=1.0, mode="lenient")
        np.testing.assert_allclose(c_un, c_w)
        np.testing.assert_allclose(m_un, m_w)

    def test_w_scalar_scales_by_square(self):
        # Scalar c -> every difference event has salience c**2,
        # so count and magnitude both scale by c**2.
        c_un, m_un = mpt.continuity([3, 5, 7, 7, 9], [11], 0,
                                    mode="lenient")
        c_w, m_w = mpt.continuity([3, 5, 7, 7, 9], [11], 0,
                                  w=0.5, mode="lenient")
        np.testing.assert_allclose(c_w, 0.25 * c_un)
        np.testing.assert_allclose(m_w, 0.25 * m_un)

    def test_w_vector_recency_truncation(self):
        # Zero weights on the three oldest events zero out every
        # difference event except the most recent pair.
        c, m = mpt.continuity([3, 5, 7, 7, 9], [11], 0,
                              w=[0, 0, 0, 1, 1], mode="lenient")
        assert c[0] == pytest.approx(1.0)
        assert m[0] == pytest.approx(2.0)

    def test_w_vector_matches_rolling_product(self):
        # Explicit per-event weights — verify against hand calculation
        # using the rolling-product rule.
        seq = [3, 5, 7, 7, 9]
        w = [1, 1, 0.5, 1, 1]
        # Difference events:  (3->5)=2, (5->7)=2, (7->7)=0, (7->9)=2
        # Diff-event weights: 1*1=1,   1*0.5=0.5, 0.5*1=0.5, 1*1=1
        # Backward lenient walk from query 11:
        #   (7->9): a=1, contrib 1 * 1 = 1; c+=1, m+=2
        #   (7->7): a=0, contrib 0 * 0.5 = 0
        #   (5->7): a=1, contrib 1 * 0.5 = 0.5; c+=0.5, m+=1
        #   (3->5): a=1, contrib 1 * 1 = 1; c+=1, m+=2
        c, m = mpt.continuity(seq, [11], 0, w=w, mode="lenient")
        assert c[0] == pytest.approx(2.5)
        assert m[0] == pytest.approx(5.0)

    def test_w_matches_difference_events_directly(self):
        # Parity check: compute the rolling-product weights via
        # difference_events and verify continuity's internal scaling
        # agrees with the same weights applied outside.
        seq = [3, 5, 7, 7, 9]
        w = [0.8, 1.0, 0.7, 1.0, 0.9]
        p_d, w_d, _ = mpt.difference_events(
            [np.asarray(seq).reshape(1, -1)],
            [np.asarray(w).reshape(1, -1)],
            [1],
        )
        diff_weights = np.asarray(w_d[0]).reshape(-1)   # length N-1
        # Unweighted continuity for the same query, then apply the
        # rolling-product weights term by term — this should match
        # the weighted call because the break condition and sign
        # products are identical.
        ctx = np.asarray(p_d[0]).reshape(-1)
        # For σ = 0, sign-products give a_k ∈ {-1, 0, +1}. Step
        # through the backward walk manually.
        N = len(seq)
        a = np.sign(ctx) * np.sign(11 - seq[-1])
        c_expected = 0.0
        m_expected = 0.0
        for k in range(N - 2, -1, -1):
            if a[k] <= -1.0:
                break
            contrib = max(a[k], 0.0) * diff_weights[k]
            c_expected += contrib
            m_expected += contrib * ctx[k]
        c, m = mpt.continuity(seq, [11], 0, w=w, mode="lenient")
        assert c[0] == pytest.approx(c_expected)
        assert m[0] == pytest.approx(m_expected)

    def test_w_wrong_length_raises(self):
        with pytest.raises(ValueError, match="length N"):
            mpt.continuity([3, 5, 7, 7, 9], [11], 0, w=[1, 1, 1])

    def test_w_negative_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            mpt.continuity([3, 5, 7, 7, 9], [11], 0,
                           w=[1, 1, -0.1, 1, 1])

    def test_w_negative_scalar_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            mpt.continuity([3, 5, 7, 7, 9], [11], 0, w=-0.5)

    def test_w_with_smoothing(self):
        # Weights should compose with Gaussian smoothing: uniform
        # scalar w just scales both outputs by w**2 regardless of σ.
        seq = [3, 5, 7, 7, 9]
        c_un, m_un = mpt.continuity(seq, [11], 0.3, mode="lenient")
        c_w, m_w = mpt.continuity(seq, [11], 0.3, w=0.7, mode="lenient")
        np.testing.assert_allclose(c_w, 0.49 * c_un)
        np.testing.assert_allclose(m_w, 0.49 * m_un)


# ===================================================================
#  seq_weights
# ===================================================================
