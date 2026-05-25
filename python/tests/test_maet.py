"""Tests for Multi-Attribute Expectation Tensor (MAET, v2.1.0).

Mirror of MATLAB tests/test_maet.m.
"""
import numpy as np
import pytest
import warnings

import mpt
from mpt._utils import position_variance
from mpt.tensor import _windowed_inner_product


class TestMAET:
    """Multi-attribute expectation tensor tests.

    The v2.1.0 extension to ``build_exp_tens``. The single-attribute
    legacy path is covered by ``TestTensor`` above; these tests focus on
    the MAET-specific behaviours: SA-equivalence under degenerate mapping,
    per-attribute perm/comb enumeration, weight broadcasting, group
    canonicalisation, NaN handling for variable-size events, and the
    error paths introduced by the per-attribute / per-group parameter
    structure.
    """

    # --- SA-equivalence: (N=1, A=1, K_a x 1 column weight) == SA -------

    def test_ma_matches_sa_abs(self):
        """MA with one event and one attribute reproduces SA bit-for-bit."""
        p = [0.0, 400.0, 700.0]
        w = [1.0, 0.7, 0.5]
        sigma, r, is_rel, is_per, period = 10.0, 2, False, True, 1200.0

        dens_sa = mpt.build_exp_tens(
            p, w, sigma, r, is_rel, is_per, period, verbose=False
        )

        p_attr = [np.array(p, dtype=float).reshape(3, 1)]
        w_ma = [np.array(w, dtype=float).reshape(3, 1)]
        dens_ma = mpt.build_exp_tens(
            p_attr, w_ma, [sigma], [r], None,
            [is_rel], [is_per], [period], verbose=False,
        )

        assert dens_ma.tag == "MaetDensity"
        assert dens_ma.n_j == dens_sa.n_j
        assert dens_ma.n_k == dens_sa.n_k
        np.testing.assert_array_equal(dens_ma.u_perm[0], dens_sa.u_perm)
        np.testing.assert_array_equal(dens_ma.v_comb[0], dens_sa.v_comb)
        np.testing.assert_array_equal(dens_ma.centres[0], dens_sa.centres)
        np.testing.assert_array_almost_equal(dens_ma.w_j, dens_sa.w_j)
        np.testing.assert_array_almost_equal(dens_ma.wv_comb, dens_sa.wv_comb)

    def test_ma_matches_sa_rel(self):
        """Centres reduction: is_rel=True collapses r_a dims to r_a-1."""
        p = [0.0, 400.0, 700.0]
        w = [1.0, 0.7, 0.5]
        sigma, r, is_rel, is_per, period = 10.0, 2, True, True, 1200.0

        dens_sa = mpt.build_exp_tens(
            p, w, sigma, r, is_rel, is_per, period, verbose=False
        )
        p_attr = [np.array(p, dtype=float).reshape(3, 1)]
        w_ma = [np.array(w, dtype=float).reshape(3, 1)]
        dens_ma = mpt.build_exp_tens(
            p_attr, w_ma, [sigma], [r], None,
            [is_rel], [is_per], [period], verbose=False,
        )
        assert dens_ma.centres[0].shape == (r - 1, dens_ma.n_j)
        np.testing.assert_array_equal(dens_ma.centres[0], dens_sa.centres)

    # --- Struct basics ------------------------------------------------

    def test_ma_struct_fields(self):
        """Essential fields for a pitch + time two-attribute build."""
        pitch = np.array([[0, 12], [4, 15], [7, 19]], dtype=float)  # 3 x 2
        time = np.array([[0.0, 1.0]])                               # 1 x 2
        dens = mpt.build_exp_tens(
            [pitch, time], None,
            [10.0, 0.1], [3, 1], None,
            [True, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )
        assert dens.tag == "MaetDensity"
        assert dens.n_attrs == 2
        assert dens.n_groups == 2
        assert dens.n == 2
        np.testing.assert_array_equal(dens.group_of_attr, [0, 1])
        np.testing.assert_array_equal(dens.r, [3, 1])
        np.testing.assert_array_equal(dens.k, [3, 1])
        # pitch contributes r_p - 1 = 2, time contributes r_t = 1
        assert dens.dim == 3
        np.testing.assert_array_equal(dens.dim_per_attr, [2, 1])

    def test_ma_cartesian_product_count(self):
        """n_j = sum_n (product of per-attr per-event perm counts)."""
        pitch = np.array([[0, 12, 5], [4, 15, 9], [7, 19, 12]], dtype=float)
        time = np.array([[0.0, 1.0, 2.0]])
        dens = mpt.build_exp_tens(
            [pitch, time], None,
            [10.0, 0.1], [3, 1], None,
            [True, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )
        # Per event: pitch r=3, K=3 -> P(3,3)=6 perms, C(3,3)=1 comb.
        # Time r=1, K=1 -> 1 each. Cartesian per event: 6 perms, 1 comb.
        # Three events: 18 perms, 3 combs.
        assert dens.n_j == 18
        assert dens.n_k == 3

    def test_ma_event_bookkeeping(self):
        pitch = np.array([[0, 12, 5], [4, 15, 9], [7, 19, 12]], dtype=float)
        time = np.array([[0.0, 1.0, 2.0]])
        dens = mpt.build_exp_tens(
            [pitch, time], None,
            [10.0, 0.1], [3, 1], None,
            [True, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )
        np.testing.assert_array_equal(dens.event_of_j, np.repeat([0, 1, 2], 6))
        np.testing.assert_array_equal(dens.event_of_k, [0, 1, 2])

    # --- Weight broadcasting -----------------------------------------

    def test_ma_weight_none(self):
        pitch = np.array([[0, 4], [4, 8]], dtype=float)
        dens = mpt.build_exp_tens(
            [pitch], None, [10.0], [2], None,
            [False], [True], [1200.0], verbose=False,
        )
        np.testing.assert_array_equal(dens.w[0], np.ones((2, 2)))

    def test_ma_weight_scalar_top_level(self):
        pitch = np.array([[0, 4], [4, 8]], dtype=float)
        dens = mpt.build_exp_tens(
            [pitch], 0.5, [10.0], [2], None,
            [False], [True], [1200.0], verbose=False,
        )
        np.testing.assert_array_equal(dens.w[0], np.full((2, 2), 0.5))

    def test_ma_weight_2d_per_event_row(self):
        pitch = np.array([[0, 4, 5], [4, 8, 6]], dtype=float)  # K=2, N=3
        w_row = np.array([[0.5, 1.0, 2.0]])                    # (1, 3)
        dens = mpt.build_exp_tens(
            [pitch], [w_row], [10.0], [2], None,
            [False], [True], [1200.0], verbose=False,
        )
        np.testing.assert_array_equal(
            dens.w[0], np.array([[0.5, 1.0, 2.0], [0.5, 1.0, 2.0]])
        )

    def test_ma_weight_2d_per_slot_column(self):
        pitch = np.array(
            [[0, 4, 5], [4, 8, 6], [7, 10, 9]], dtype=float
        )  # K=3, N=3
        w_col = np.array([[0.5], [1.0], [2.0]])                # (3, 1)
        dens = mpt.build_exp_tens(
            [pitch], [w_col], [10.0], [2], None,
            [False], [True], [1200.0], verbose=False,
        )
        np.testing.assert_array_equal(
            dens.w[0], np.tile([[0.5], [1.0], [2.0]], (1, 3))
        )

    def test_ma_weight_2d_full_matrix(self):
        pitch = np.array([[0, 4], [4, 8]], dtype=float)
        W = np.array([[0.1, 0.2], [0.3, 0.4]])
        dens = mpt.build_exp_tens(
            [pitch], [W], [10.0], [2], None,
            [False], [True], [1200.0], verbose=False,
        )
        np.testing.assert_array_equal(dens.w[0], W)

    def test_ma_weight_1d_per_event_disambiguated(self):
        pitch = np.array([[0, 4], [4, 8], [7, 9]], dtype=float)  # K=3, N=2
        w_1d = np.array([0.5, 1.0])  # length N=2 (not K=3) -> per-event
        dens = mpt.build_exp_tens(
            [pitch], [w_1d], [10.0], [2], None,
            [False], [True], [1200.0], verbose=False,
        )
        expected = np.array([[0.5, 1.0], [0.5, 1.0], [0.5, 1.0]])
        np.testing.assert_array_equal(dens.w[0], expected)

    def test_ma_weight_1d_per_slot_disambiguated(self):
        pitch = np.array([[0, 4], [4, 8], [7, 9]], dtype=float)  # K=3, N=2
        w_1d = np.array([0.5, 1.0, 2.0])  # length K=3 (not N=2) -> per-slot
        dens = mpt.build_exp_tens(
            [pitch], [w_1d], [10.0], [2], None,
            [False], [True], [1200.0], verbose=False,
        )
        expected = np.tile([[0.5], [1.0], [2.0]], (1, 2))
        np.testing.assert_array_equal(dens.w[0], expected)

    def test_ma_weight_1d_ambiguous_when_K_equals_N(self):
        pitch = np.array(
            [[0, 4, 5], [4, 8, 6], [7, 10, 9]], dtype=float
        )  # K=3, N=3
        w_1d = np.array([0.5, 1.0, 2.0])
        with pytest.raises(ValueError, match="ambiguous"):
            mpt.build_exp_tens(
                [pitch], [w_1d], [10.0], [2], None,
                [False], [True], [1200.0], verbose=False,
            )

    # --- Groups -------------------------------------------------------

    def test_ma_groups_default_singleton(self):
        pitch = np.array([[0, 4]], dtype=float)
        time = np.array([[0.0, 1.0]])
        dens = mpt.build_exp_tens(
            [pitch, time], None, [10.0, 0.1], [1, 1], None,
            [False, False], [True, False], [1200.0, 0.0], verbose=False,
        )
        assert dens.n_groups == 2
        np.testing.assert_array_equal(dens.group_of_attr, [0, 1])

    def test_ma_groups_vector_and_cell_agree(self):
        pitch = np.array([[0, 4]], dtype=float)
        time = np.array([[0.0, 1.0]])
        x = np.array([[0.0, 0.5]])
        y = np.array([[0.0, 0.5]])
        z = np.array([[0.0, 0.5]])

        args_rest = (
            [10.0, 0.1, 0.2], [1, 1, 1, 1, 1],   # sigma_vec, r_vec
        )
        flags = ([False, False, False], [True, False, False],
                 [1200.0, 0.0, 0.0])

        dens_v = mpt.build_exp_tens(
            [pitch, time, x, y, z], None,
            *args_rest, [0, 1, 2, 2, 2], *flags, verbose=False,
        )
        dens_c = mpt.build_exp_tens(
            [pitch, time, x, y, z], None,
            *args_rest, [[0], [1], [2, 3, 4]], *flags, verbose=False,
        )
        np.testing.assert_array_equal(dens_v.group_of_attr, dens_c.group_of_attr)
        assert dens_v.n_groups == dens_c.n_groups
        for g in range(dens_v.n_groups):
            np.testing.assert_array_equal(
                dens_v.attrs_of_group[g], dens_c.attrs_of_group[g]
            )

    def test_ma_groups_noncontiguous_errors(self):
        pitch = np.array([[0, 4]], dtype=float)
        with pytest.raises(ValueError, match="contiguous"):
            mpt.build_exp_tens(
                [pitch, pitch], None, [10.0, 10.0], [1, 1], [0, 2],
                [False, False], [True, True], [1200.0, 1200.0], verbose=False,
            )

    def test_ma_groups_cell_duplicate_attr_errors(self):
        pitch = np.array([[0, 4]], dtype=float)
        with pytest.raises(ValueError, match="more than one group"):
            mpt.build_exp_tens(
                [pitch, pitch], None, [10.0], [1, 1], [[0, 1], [1]],
                [False], [True], [1200.0], verbose=False,
            )

    # --- NaN-padded variable-size events -----------------------------

    def test_ma_nan_padding(self):
        # Event 0: 3 pitches. Event 1: 2 pitches (third slot NaN).
        pitch = np.array([[0, 0], [4, 4], [7, np.nan]], dtype=float)
        time = np.array([[0.0, 1.0]])
        dens = mpt.build_exp_tens(
            [pitch, time], None, [10.0, 0.1], [2, 1], None,
            [False, False], [True, False], [1200.0, 0.0], verbose=False,
        )
        # Event 0: P(3,2)=6 perms, C(3,2)=3 combs. Event 1: P(2,2)=2, C(2,2)=1.
        # Cartesian x 1 time = same. Totals: n_j = 8, n_k = 4.
        assert dens.n_j == 8
        assert dens.n_k == 4

    # --- Per-tuple weight factorisation ------------------------------

    def test_ma_per_tuple_weight_product(self):
        # One event, K_p=2 with slot weights 2 and 3; r_p=2.
        # Time K=1, slot weight 5; r_t=1.
        # Each perm tuple weight = (w_i * w_j for pitch) * 5 for time.
        pitch = np.array([[0.0], [4.0]])
        time = np.array([[1.5]])
        w_pitch = np.array([[2.0], [3.0]])
        w_time = np.array([[5.0]])
        dens = mpt.build_exp_tens(
            [pitch, time], [w_pitch, w_time],
            [10.0, 0.1], [2, 1], None,
            [False, False], [True, False], [1200.0, 0.0], verbose=False,
        )
        # 2 pitch perms, each with weight 2*3*5 = 30.
        np.testing.assert_array_almost_equal(dens.w_j, [30.0, 30.0])
        # 1 pitch comb, weight 2*3*5 = 30.
        np.testing.assert_array_almost_equal(dens.wv_comb, [30.0])

    # --- Error paths -------------------------------------------------

    def test_ma_insufficient_slots_errors(self):
        # Event 1 has 1 valid slot, r=2 -> error
        pitch = np.array(
            [[0, 0], [4, np.nan], [np.nan, np.nan]], dtype=float
        )
        with pytest.raises(ValueError, match="non-NaN slot"):
            mpt.build_exp_tens(
                [pitch], None, [10.0], [2], None,
                [False], [True], [1200.0], verbose=False,
            )

    def test_ma_wrong_r_vec_length(self):
        pitch = np.array([[0, 4]], dtype=float)
        with pytest.raises(ValueError, match="r_vec"):
            mpt.build_exp_tens(
                [pitch, pitch], None, [10.0, 10.0], [1], None,
                [False, False], [True, True], [1200.0, 1200.0], verbose=False,
            )

    def test_ma_wrong_sigma_length(self):
        pitch = np.array([[0, 4]], dtype=float)
        with pytest.raises(ValueError, match="sigma_vec"):
            mpt.build_exp_tens(
                [pitch, pitch], None, [10.0], [1, 1], None,
                [False, False], [True, True], [1200.0, 1200.0], verbose=False,
            )

    def test_ma_mismatched_event_counts(self):
        pitch = np.array([[0, 4]], dtype=float)       # N=2
        time = np.array([[0.0, 1.0, 2.0]])             # N=3
        with pytest.raises(ValueError, match="share N"):
            mpt.build_exp_tens(
                [pitch, time], None, [10.0, 0.1], [1, 1], None,
                [False, False], [True, False], [1200.0, 0.0], verbose=False,
            )

    def test_ma_isrel_r1_warns(self):
        pitch = np.array([[0, 4]], dtype=float)
        with pytest.warns(UserWarning, match="degenerate"):
            mpt.build_exp_tens(
                [pitch], None, [10.0], [1], None,
                [True], [True], [1200.0], verbose=False,
            )

    def test_ma_wrong_positional_count(self):
        pitch = np.array([[0, 4]], dtype=float)
        # 7 positional args for MA is wrong (should be 8)
        with pytest.raises(ValueError, match="8 positional"):
            mpt.build_exp_tens(
                [pitch], None, [10.0], [1],
                [False], [True], [1200.0], verbose=False,
            )

    # --- evalExpTens MA path ------------------------------------------

    def test_ma_eval_matches_sa_abs(self):
        """MA eval matches SA at the same query points (is_rel=False)."""
        p = [0.0, 400.0, 700.0]
        w = [1.0, 0.7, 0.5]
        sigma, r, is_rel, is_per, period = 10.0, 2, False, True, 1200.0
        x_sa = np.array([[100, 500], [300, 600]], dtype=float)  # 2 x 2 (SA)

        dens_sa = mpt.build_exp_tens(
            p, w, sigma, r, is_rel, is_per, period, verbose=False
        )
        vals_sa = mpt.eval_exp_tens(dens_sa, x_sa, verbose=False)

        p_attr = [np.array(p, dtype=float).reshape(3, 1)]
        w_ma = [np.array(w, dtype=float).reshape(3, 1)]
        dens_ma = mpt.build_exp_tens(
            p_attr, w_ma, [sigma], [r], None,
            [is_rel], [is_per], [period], verbose=False,
        )
        vals_ma_cell = mpt.eval_exp_tens(dens_ma, [x_sa], verbose=False)
        vals_ma_mat = mpt.eval_exp_tens(dens_ma, x_sa, verbose=False)

        np.testing.assert_allclose(vals_ma_cell, vals_sa, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(vals_ma_mat, vals_sa, rtol=1e-12, atol=1e-12)

    def test_ma_eval_matches_sa_rel(self):
        """Same with is_rel=True: reduced-dim query points."""
        p = [0.0, 400.0, 700.0]
        w = [1.0, 0.7, 0.5]
        sigma, r, is_rel, is_per, period = 10.0, 3, True, True, 1200.0
        x_sa = np.array([[400, 200], [700, 500]], dtype=float)  # (r-1) x nQ

        dens_sa = mpt.build_exp_tens(
            p, w, sigma, r, is_rel, is_per, period, verbose=False
        )
        p_attr = [np.array(p, dtype=float).reshape(3, 1)]
        w_ma = [np.array(w, dtype=float).reshape(3, 1)]
        dens_ma = mpt.build_exp_tens(
            p_attr, w_ma, [sigma], [r], None,
            [is_rel], [is_per], [period], verbose=False,
        )

        vals_sa = mpt.eval_exp_tens(dens_sa, x_sa, verbose=False)
        vals_ma = mpt.eval_exp_tens(dens_ma, [x_sa], verbose=False)
        np.testing.assert_allclose(vals_ma, vals_sa, rtol=1e-12, atol=1e-12)

    def test_ma_eval_normalisation_matches_sa(self):
        """'gaussian' and 'pdf' normalisation modes match SA."""
        p = [0.0, 400.0, 700.0]
        w = [1.0, 0.7, 0.5]
        sigma, r, is_rel, is_per, period = 10.0, 3, True, True, 1200.0
        x_sa = np.array([[400, 200], [700, 500]], dtype=float)

        dens_sa = mpt.build_exp_tens(
            p, w, sigma, r, is_rel, is_per, period, verbose=False
        )
        p_attr = [np.array(p, dtype=float).reshape(3, 1)]
        w_ma = [np.array(w, dtype=float).reshape(3, 1)]
        dens_ma = mpt.build_exp_tens(
            p_attr, w_ma, [sigma], [r], None,
            [is_rel], [is_per], [period], verbose=False,
        )
        for mode in ("gaussian", "pdf"):
            vals_sa = mpt.eval_exp_tens(dens_sa, x_sa, mode, verbose=False)
            vals_ma = mpt.eval_exp_tens(dens_ma, [x_sa], mode, verbose=False)
            np.testing.assert_allclose(vals_ma, vals_sa, rtol=1e-12, atol=1e-12)

    def test_ma_eval_cell_vs_matrix_forms_agree(self):
        """Cell form and single-matrix form give identical results."""
        pitch = np.array([[0.0, 4.0, 7.0]]).T        # K=3, N=1
        time  = np.array([[1.0]])                     # K=1, N=1
        dens = mpt.build_exp_tens(
            [pitch, time], None,
            [10.0, 0.1], [2, 1], None,
            [False, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )
        # dim_per_attr = [2, 1], total dim = 3
        x_pitch = np.array([[0.0, 4.0], [4.0, 7.0]])  # 2 x 2
        x_time  = np.array([[1.0, 2.0]])               # 1 x 2
        vals_cell = mpt.eval_exp_tens(dens, [x_pitch, x_time], verbose=False)
        x_mat = np.vstack([x_pitch, x_time])           # 3 x 2
        vals_mat = mpt.eval_exp_tens(dens, x_mat, verbose=False)
        np.testing.assert_array_equal(vals_cell, vals_mat)

    def test_ma_eval_per_group_isper(self):
        """Periodic pitch wraps; nonperiodic time does not."""
        pitch = np.array([[0.0]])
        time  = np.array([[0.0]])
        dens = mpt.build_exp_tens(
            [pitch, time], None,
            [20.0, 20.0], [1, 1], None,
            [False, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )
        # Pitch at 0 vs 1200 under periodic pitch => equal density
        v_pitch_0    = mpt.eval_exp_tens(
            dens, [np.array([[0.0]]),    np.array([[0.0]])], verbose=False
        )[0]
        v_pitch_1200 = mpt.eval_exp_tens(
            dens, [np.array([[1200.0]]), np.array([[0.0]])], verbose=False
        )[0]
        np.testing.assert_allclose(v_pitch_0, v_pitch_1200, rtol=1e-12)

        # Time at 0 vs 1200 under nonperiodic time => strictly lower at 1200
        v_time_0    = mpt.eval_exp_tens(
            dens, [np.array([[0.0]]), np.array([[0.0]])], verbose=False
        )[0]
        v_time_1200 = mpt.eval_exp_tens(
            dens, [np.array([[0.0]]), np.array([[1200.0]])], verbose=False
        )[0]
        assert v_time_1200 < v_time_0

    def test_ma_eval_positive_at_tuple_centre(self):
        """Density at a tuple centre is positive and at least as high as
        at a point far from every tuple."""
        pitch = np.array([[0.0, 4.0, 7.0]]).T   # K=3, N=1
        time  = np.array([[1.0]])
        dens = mpt.build_exp_tens(
            [pitch, time], None,
            [10.0, 0.1], [2, 1], None,
            [False, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )
        # At tuple centre (pitch=(0,4), time=1.0): one of the perm tuples
        x_centre = [np.array([[0.0], [4.0]]), np.array([[1.0]])]
        x_far    = [np.array([[600.0], [800.0]]), np.array([[50.0]])]
        v_centre = mpt.eval_exp_tens(dens, x_centre, verbose=False)[0]
        v_far    = mpt.eval_exp_tens(dens, x_far,    verbose=False)[0]
        assert v_centre > 0
        assert v_centre > v_far

    def test_ma_eval_wrong_cell_length_errors(self):
        pitch = np.array([[0.0, 4.0]]).T
        time  = np.array([[1.0]])
        dens = mpt.build_exp_tens(
            [pitch, time], None,
            [10.0, 0.1], [2, 1], None,
            [False, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )
        # Only one cell provided, expected 2
        with pytest.raises(ValueError, match="length 2"):
            mpt.eval_exp_tens(
                dens, [np.array([[0.0], [4.0]])], verbose=False
            )

    def test_ma_eval_wrong_per_attr_rows_errors(self):
        pitch = np.array([[0.0, 4.0]]).T
        time  = np.array([[1.0]])
        dens = mpt.build_exp_tens(
            [pitch, time], None,
            [10.0, 0.1], [2, 1], None,
            [False, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )
        # pitch query has 3 rows instead of 2
        with pytest.raises(ValueError, match="attribute 0"):
            mpt.eval_exp_tens(
                dens, [np.zeros((3, 1)), np.zeros((1, 1))], verbose=False
            )

    def test_ma_eval_wrong_total_rows_errors(self):
        pitch = np.array([[0.0, 4.0]]).T
        time  = np.array([[1.0]])
        dens = mpt.build_exp_tens(
            [pitch, time], None,
            [10.0, 0.1], [2, 1], None,
            [False, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )
        # Single-matrix form: dim=3 expected, we pass 5 rows
        with pytest.raises(ValueError, match="total dim"):
            mpt.eval_exp_tens(dens, np.zeros((5, 1)), verbose=False)

    def test_ma_eval_empty_query(self):
        pitch = np.array([[0.0, 4.0]]).T
        dens = mpt.build_exp_tens(
            [pitch], None, [10.0], [2], None,
            [False], [True], [1200.0], verbose=False,
        )
        vals = mpt.eval_exp_tens(dens, np.zeros((2, 0)), verbose=False)
        assert vals.shape == (0,)

    def test_ma_eval_dispatch_on_type(self):
        """Public eval_exp_tens dispatches on dens type."""
        # ExpTensDensity -> SA path
        dens_sa = mpt.build_exp_tens(
            [0.0, 4.0], None, 10.0, 2, False, True, 1200.0, verbose=False
        )
        assert isinstance(dens_sa, mpt.ExpTensDensity)
        # MaetDensity -> MA path
        dens_ma = mpt.build_exp_tens(
            [np.array([[0.0, 4.0]]).T], None, [10.0], [2], None,
            [False], [True], [1200.0], verbose=False,
        )
        assert isinstance(dens_ma, mpt.MaetDensity)
        # Both evaluate successfully
        x = np.array([[0.0], [4.0]])
        mpt.eval_exp_tens(dens_sa, x, verbose=False)
        mpt.eval_exp_tens(dens_ma, x, verbose=False)

    # --- cosSimExpTens MA path ---------------------------------------

    def test_ma_cossim_matches_sa_abs(self):
        """MA cos-sim matches SA at the SA-equivalence mapping (is_rel=False)."""
        p_a = [0.0, 400.0, 700.0]
        p_b = [0.0, 300.0, 700.0]
        w_a = [1.0, 0.7, 0.5]
        w_b = [1.0, 0.6, 0.8]
        sigma, r, is_rel, is_per, period = 10.0, 2, False, True, 1200.0

        s_sa = mpt.cos_sim_exp_tens_raw(
            p_a, w_a, p_b, w_b, sigma, r, is_rel, is_per, period,
            verbose=False,
        )

        # MA form: one attribute, one event, column weight
        da = mpt.build_exp_tens(
            [np.array(p_a).reshape(3, 1)], [np.array(w_a).reshape(3, 1)],
            [sigma], [r], None, [is_rel], [is_per], [period], verbose=False,
        )
        db = mpt.build_exp_tens(
            [np.array(p_b).reshape(3, 1)], [np.array(w_b).reshape(3, 1)],
            [sigma], [r], None, [is_rel], [is_per], [period], verbose=False,
        )
        s_ma = mpt.cos_sim_exp_tens(da, db, verbose=False)

        np.testing.assert_allclose(s_ma, s_sa, rtol=1e-12, atol=1e-12)

    def test_ma_cossim_matches_sa_rel(self):
        """SA-equivalence with is_rel=True (uses pairwise-diff formula periodically)."""
        p_a = [0.0, 400.0, 700.0]
        p_b = [0.0, 300.0, 700.0]
        w_a = [1.0, 0.7, 0.5]
        w_b = [1.0, 0.6, 0.8]
        for r, is_per, period in [(2, True, 1200.0),
                                   (3, True, 1200.0),
                                   (3, False, 0.0)]:
            s_sa = mpt.cos_sim_exp_tens_raw(
                p_a, w_a, p_b, w_b, 10.0, r, True, is_per, period, verbose=False
            )
            da = mpt.build_exp_tens(
                [np.array(p_a).reshape(3, 1)], [np.array(w_a).reshape(3, 1)],
                [10.0], [r], None, [True], [is_per], [period], verbose=False,
            )
            db = mpt.build_exp_tens(
                [np.array(p_b).reshape(3, 1)], [np.array(w_b).reshape(3, 1)],
                [10.0], [r], None, [True], [is_per], [period], verbose=False,
            )
            s_ma = mpt.cos_sim_exp_tens(da, db, verbose=False)
            np.testing.assert_allclose(
                s_ma, s_sa, rtol=1e-12, atol=1e-12,
                err_msg=f"r={r}, is_per={is_per}, period={period}",
            )

    def test_ma_cossim_self_is_one(self):
        """cos_sim(d, d) == 1 for a non-degenerate MA density."""
        pitch = np.array([[0.0, 12.0], [4.0, 15.0], [7.0, 19.0]])
        time  = np.array([[0.0, 1.0]])
        d = mpt.build_exp_tens(
            [pitch, time], None,
            [10.0, 0.1], [3, 1], None,
            [True, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )
        s = mpt.cos_sim_exp_tens(d, d, verbose=False)
        np.testing.assert_allclose(s, 1.0, rtol=1e-12, atol=1e-12)

    def test_ma_cossim_symmetry(self):
        """cos_sim(a, b) == cos_sim(b, a)."""
        pitchA = np.array([[0.0, 12.0], [4.0, 15.0], [7.0, 19.0]])
        timeA  = np.array([[0.0, 1.0]])
        pitchB = np.array([[0.0, 10.0], [4.0, 13.0], [7.0, 17.0]])
        timeB  = np.array([[0.0, 1.2]])

        da = mpt.build_exp_tens(
            [pitchA, timeA], None,
            [10.0, 0.1], [3, 1], None,
            [True, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )
        db = mpt.build_exp_tens(
            [pitchB, timeB], None,
            [10.0, 0.1], [3, 1], None,
            [True, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )
        s_ab = mpt.cos_sim_exp_tens(da, db, verbose=False)
        s_ba = mpt.cos_sim_exp_tens(db, da, verbose=False)
        np.testing.assert_allclose(s_ab, s_ba, rtol=1e-12, atol=1e-12)

    def test_ma_cossim_isrel_transposition_invariance(self):
        """Shifting all pitches by a constant preserves cos_sim when
        the pitch group has is_rel=True (one event, so the shift affects
        every pitch slot equally)."""
        pitch = np.array([[0.0], [400.0], [700.0]])  # K=3, N=1
        time  = np.array([[1.0]])
        pitch_shifted = pitch + 137.0

        d1 = mpt.build_exp_tens(
            [pitch, time], None,
            [10.0, 0.1], [3, 1], None,
            [True, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )
        d2 = mpt.build_exp_tens(
            [pitch_shifted, time], None,
            [10.0, 0.1], [3, 1], None,
            [True, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )
        s = mpt.cos_sim_exp_tens(d1, d2, verbose=False)
        np.testing.assert_allclose(s, 1.0, rtol=1e-10, atol=1e-10)

    def test_ma_cossim_raw_matches_struct(self):
        """cos_sim_exp_tens_raw (MA) == build + cos_sim_exp_tens (MA)."""
        pitchA = np.array([[0.0, 12.0], [4.0, 15.0], [7.0, 19.0]])
        timeA  = np.array([[0.0, 1.0]])
        pitchB = np.array([[0.0, 10.0], [4.0, 13.0], [7.0, 17.0]])
        timeB  = np.array([[0.0, 1.2]])

        # Raw MA form: 10 positional args
        s_raw = mpt.cos_sim_exp_tens_raw(
            [pitchA, timeA], None, [pitchB, timeB], None,
            [10.0, 0.1], [3, 1], None,
            [True, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )
        da = mpt.build_exp_tens(
            [pitchA, timeA], None,
            [10.0, 0.1], [3, 1], None,
            [True, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )
        db = mpt.build_exp_tens(
            [pitchB, timeB], None,
            [10.0, 0.1], [3, 1], None,
            [True, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )
        s_struct = mpt.cos_sim_exp_tens(da, db, verbose=False)
        np.testing.assert_allclose(s_raw, s_struct, rtol=1e-12, atol=1e-12)

    def test_ma_cossim_raw_sa_still_works(self):
        """SA raw-args call unchanged from v2.0.0 behaviour."""
        s = mpt.cos_sim_exp_tens_raw(
            [0.0, 4.0, 7.0], None, [0.0, 4.0, 7.0], None,
            10.0, 2, True, True, 1200.0, verbose=False,
        )
        np.testing.assert_allclose(s, 1.0, rtol=1e-12, atol=1e-12)

    def test_ma_cossim_raw_mismatched_types_errors(self):
        """p1 is MA but p2 is SA. Under the unified cos_sim_exp_tens
        dispatcher, the first arg's type (here, list of arrays = MA)
        sets the dispatch arm, and the resulting positional-count
        mismatch (MA needs 10 positional args; SA-style call gives 9)
        produces a clear TypeError."""
        pitch_ma = [np.array([[0.0, 4.0]]).T]
        pitch_sa = [0.0, 4.0]
        with pytest.raises(TypeError, match="positional"):
            mpt.cos_sim_exp_tens_raw(
                pitch_ma, None, pitch_sa, None,
                10.0, 2, False, True, 1200.0, verbose=False,
            )

    def test_ma_cossim_mixed_struct_types_errors(self):
        """MaetDensity paired with ExpTensDensity -> TypeError."""
        d_sa = mpt.build_exp_tens(
            [0.0, 4.0, 7.0], None, 10.0, 2, False, True, 1200.0, verbose=False
        )
        d_ma = mpt.build_exp_tens(
            [np.array([[0.0, 4.0, 7.0]]).T], None,
            [10.0], [2], None, [False], [True], [1200.0], verbose=False,
        )
        with pytest.raises(TypeError, match="same type"):
            mpt.cos_sim_exp_tens(d_sa, d_ma, verbose=False)
        with pytest.raises(TypeError, match="same type"):
            mpt.cos_sim_exp_tens(d_ma, d_sa, verbose=False)

    def test_ma_cossim_parameter_mismatch_errors(self):
        """Mismatched MA densities raise specific ValueErrors."""
        pitch = np.array([[0.0, 4.0, 7.0]]).T
        base_kwargs = dict(
            p_attr=[pitch], w=None,
            sigma_vec=[10.0], r_vec=[2], groups=None,
            is_rel_vec=[False], is_per_vec=[True], period_vec=[1200.0],
        )
        d_ref = mpt.build_exp_tens(
            base_kwargs["p_attr"], base_kwargs["w"],
            base_kwargs["sigma_vec"], base_kwargs["r_vec"], base_kwargs["groups"],
            base_kwargs["is_rel_vec"], base_kwargs["is_per_vec"],
            base_kwargs["period_vec"], verbose=False,
        )
        # Different r_vec
        d_r = mpt.build_exp_tens(
            [pitch], None, [10.0], [3], None,
            [False], [True], [1200.0], verbose=False,
        )
        with pytest.raises(ValueError, match="r"):
            mpt.cos_sim_exp_tens(d_ref, d_r, verbose=False)
        # Different sigma
        d_s = mpt.build_exp_tens(
            [pitch], None, [20.0], [2], None,
            [False], [True], [1200.0], verbose=False,
        )
        with pytest.raises(ValueError, match="sigma"):
            mpt.cos_sim_exp_tens(d_ref, d_s, verbose=False)
        # Different is_rel
        d_rel = mpt.build_exp_tens(
            [pitch], None, [10.0], [2], None,
            [True], [True], [1200.0], verbose=False,
        )
        with pytest.raises(ValueError, match="is_rel"):
            mpt.cos_sim_exp_tens(d_ref, d_rel, verbose=False)
        # Different period on periodic group
        d_p = mpt.build_exp_tens(
            [pitch], None, [10.0], [2], None,
            [False], [True], [2400.0], verbose=False,
        )
        with pytest.raises(ValueError, match="period"):
            mpt.cos_sim_exp_tens(d_ref, d_p, verbose=False)

    # --- entropyExpTens MA path -------------------------------------

    def test_ma_entropy_sa_equivalence_periodic(self):
        """MA entropy matches SA entropy at the SA-equivalence mapping
        (single periodic group, is_rel=False)."""
        p = np.array([0.0, 4.0, 7.0])
        w = np.array([1.0, 1.0, 1.0])
        H_sa = mpt.entropy_exp_tens(
            p, w, 10.0, 1, False, True, 12.0,
            n_points_per_dim=400,
        )
        H_ma = mpt.entropy_exp_tens(
            [p.reshape(3, 1)], [w.reshape(3, 1)],
            [10.0], [1], None, [False], [True], [12.0],
            n_points_per_dim=400,
        )
        np.testing.assert_allclose(H_ma, H_sa, rtol=1e-10, atol=1e-10)

    def test_ma_entropy_sa_equivalence_nonperiodic(self):
        """MA entropy matches SA entropy for a non-periodic group with
        explicit bounds."""
        p = np.array([0.0, 4.0, 7.0])
        w = np.array([1.0, 1.0, 1.0])
        H_sa = mpt.entropy_exp_tens(
            p, w, 10.0, 1, False, False, 0.0,
            x_min=-3.0, x_max=10.0, n_points_per_dim=400,
        )
        H_ma = mpt.entropy_exp_tens(
            [p.reshape(3, 1)], [w.reshape(3, 1)],
            [10.0], [1], None, [False], [False], [0.0],
            x_min=-3.0, x_max=10.0, n_points_per_dim=400,
        )
        np.testing.assert_allclose(H_ma, H_sa, rtol=1e-10, atol=1e-10)

    def test_ma_entropy_uniform_pitch_high(self):
        """Chromatic scale with wide sigma gives near-uniform pmf,
        so normalised entropy is close to 1."""
        p = np.arange(12, dtype=np.float64)
        H = mpt.entropy_exp_tens(
            [p.reshape(12, 1)], None,
            [100.0], [1], None, [False], [True], [12.0],
            n_points_per_dim=400,
        )
        assert H > 0.95

    def test_ma_entropy_concentrated_below_uniform(self):
        """A single pitch is more concentrated than the chromatic
        scale, so gives lower normalised entropy."""
        p_one = np.array([5.0])
        H_one = mpt.entropy_exp_tens(
            [p_one.reshape(1, 1)], None,
            [20.0], [1], None, [False], [True], [12.0],
            n_points_per_dim=400,
        )
        p_all = np.arange(12, dtype=np.float64)
        H_all = mpt.entropy_exp_tens(
            [p_all.reshape(12, 1)], None,
            [20.0], [1], None, [False], [True], [12.0],
            n_points_per_dim=400,
        )
        assert H_one < H_all

    def test_ma_entropy_pitch_plus_time_runs(self):
        """A 3-attribute density with pitch (periodic) + time
        (non-periodic) produces a finite entropy and the grid
        dimensionality is handled correctly."""
        pitch = np.array([[0.0, 12.0], [4.0, 15.0], [7.0, 19.0]])  # 3 x 2
        time = np.array([[0.0, 1.0]])                               # 1 x 2
        dens = mpt.build_exp_tens(
            [pitch, time], None,
            [20.0, 0.1], [2, 1], None,
            [True, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )
        # dim = (2-1) + 1 = 2
        assert dens.dim == 2
        H = mpt.entropy_exp_tens(
            dens,
            x_min=-0.5, x_max=1.5,
            n_points_per_dim=80,
        )
        assert 0.0 < H < 1.0

    def test_ma_entropy_grid_limit_errors(self):
        """Excessively large grid requests error with a suggestion."""
        pitch = np.array([[0.0, 12.0], [4.0, 15.0], [7.0, 19.0]])
        time = np.array([[0.0, 1.0]])
        dens = mpt.build_exp_tens(
            [pitch, time], None,
            [20.0, 0.1], [2, 1], None,
            [True, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )
        with pytest.raises(ValueError, match="grid_limit"):
            mpt.entropy_exp_tens(
                dens,
                x_min=0.0, x_max=2.0,
                n_points_per_dim=20000,
                grid_limit=int(1e6),
            )

    def test_ma_entropy_missing_bounds_errors(self):
        """Non-periodic group without bounds raises ValueError."""
        p = np.array([0.0, 4.0, 7.0])
        with pytest.raises(ValueError, match="non-periodic"):
            mpt.entropy_exp_tens(
                [p.reshape(3, 1)], None,
                [10.0], [1], None, [False], [False], [0.0],
                n_points_per_dim=100,
            )

    def test_ma_entropy_per_group_bounds(self):
        """Length-G x_min/x_max vectors are accepted, with periodic
        group entries ignored."""
        pitch = np.array([[0.0, 12.0], [4.0, 15.0], [7.0, 19.0]])
        time = np.array([[0.0, 1.0]])
        H_scalar = mpt.entropy_exp_tens(
            [pitch, time], None,
            [20.0, 0.1], [2, 1], None,
            [True, False], [True, False], [1200.0, 0.0],
            x_min=-0.5, x_max=1.5,
            n_points_per_dim=60,
        )
        # Length-G vector with NaN for the periodic group — same result.
        H_vec = mpt.entropy_exp_tens(
            [pitch, time], None,
            [20.0, 0.1], [2, 1], None,
            [True, False], [True, False], [1200.0, 0.0],
            x_min=[float("nan"), -0.5],
            x_max=[float("nan"),  1.5],
            n_points_per_dim=60,
        )
        np.testing.assert_allclose(H_scalar, H_vec, rtol=1e-12, atol=1e-12)

    # --- differenceEvents -------------------------------------------

    def test_diff_order_0_identity(self):
        """Order 0 returns the input unchanged."""
        p = [np.array([[0.0, 2.0, 5.0, 7.0]])]
        pd, wd = mpt.difference_events(p, None, None, [0], [12.0])
        np.testing.assert_array_equal(pd[0], p[0])
        assert wd is None

    def test_diff_order_1_nonperiodic(self):
        """Order-1 differencing produces pairwise inter-event differences."""
        p = [np.array([[0.0, 2.0, 5.0, 7.0]])]
        pd, _ = mpt.difference_events(p, None, None, [1], [0.0])
        np.testing.assert_allclose(pd[0], [[2.0, 3.0, 2.0]])

    def test_diff_order_1_periodic_wrap(self):
        """Periodic wrapping maps large positive differences to negative
        shortest-arc values."""
        p = [np.array([[0.0, 11.0]])]  # diff = 11, wraps to -1 under P=12
        pd, _ = mpt.difference_events(p, None, None, [1], [12.0])
        np.testing.assert_allclose(pd[0], [[-1.0]])

    def test_diff_order_2(self):
        """Order 2 applies first-order differencing twice."""
        p = [np.array([[0.0, 2.0, 5.0, 7.0]])]
        pd, _ = mpt.difference_events(p, None, None, [2], [0.0])
        # 1st: [2, 3, 2]; 2nd: [1, -1]
        np.testing.assert_allclose(pd[0], [[1.0, -1.0]])

    def test_diff_weight_rolling_product_order_1(self):
        """Order-1 weights are a rolling product of width 2."""
        p = [np.array([[0.0, 2.0, 5.0, 7.0]])]
        w = [np.array([[0.5, 0.8, 1.0, 0.2]])]
        _, wd = mpt.difference_events(p, w, None, [1], [0.0])
        np.testing.assert_allclose(wd[0], [[0.5*0.8, 0.8*1.0, 1.0*0.2]])

    def test_diff_weight_rolling_product_order_2(self):
        """Order-2 weights are a rolling product of width 3 (each
        constituent weight appears exactly once per output column)."""
        p = [np.array([[0.0, 1.0, 3.0, 6.0]])]
        w = [np.array([[0.5, 0.8, 1.0, 0.2]])]
        _, wd = mpt.difference_events(p, w, None, [2], [0.0])
        np.testing.assert_allclose(
            wd[0], [[0.5*0.8*1.0, 0.8*1.0*0.2]]
        )

    def test_diff_weight_scalar_raised_to_power(self):
        """A top-level scalar c with uniform order k returns
        scalar c**(k+1), so scalar and vector-of-c inputs produce
        equivalent downstream densities."""
        p = [np.array([[0.0, 2.0, 5.0]])]
        _, wd = mpt.difference_events(p, 0.7, None, [1], [0.0])
        assert wd == pytest.approx(0.7 ** 2)

    def test_diff_mixed_orders_align(self):
        """Lower-order groups have leading events dropped to align with
        the highest-order group."""
        p = [np.array([[0.0, 2.0, 5.0, 7.0]]),
             np.array([[0.0, 1.0, 2.0, 3.5]])]
        pd, _ = mpt.difference_events(p, None, None, [0, 1], [0.0, 0.0])
        # Group 0 (k=0): max_order - k = 1 leading event dropped.
        np.testing.assert_allclose(pd[0], [[2.0, 5.0, 7.0]])
        # Group 1 (k=1): differenced, no further drop.
        np.testing.assert_allclose(pd[1], [[1.0, 1.0, 1.5]])
        # Both outputs have N' = 3.
        assert pd[0].shape[1] == 3 and pd[1].shape[1] == 3

    def test_diff_grouped_attrs_share_order(self):
        """Two attributes in one group receive the same differencing."""
        p = [np.array([[0.0, 1.0, 3.0]]), np.array([[10.0, 12.0, 16.0]])]
        groups = [0, 0]  # both in group 0 (0-indexed, Python)
        pd, _ = mpt.difference_events(p, None, groups, [1], [0.0])
        np.testing.assert_allclose(pd[0], [[1.0, 2.0]])
        np.testing.assert_allclose(pd[1], [[2.0, 4.0]])

    def test_diff_feeds_build_exp_tens(self):
        """Output of differenceEvents feeds directly into build_exp_tens."""
        p = [np.array([[0.0, 2.0, 5.0, 7.0]]),
             np.array([[0.0, 0.5, 1.2, 1.7]])]
        pd, wd = mpt.difference_events(
            p, None, None, [0, 1], [1200.0, 0.0],
        )
        dens = mpt.build_exp_tens(
            pd, wd, [10.0, 0.05], [1, 1], None,
            [False, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )
        assert dens.tag == "MaetDensity"
        assert dens.n == 3  # N' after max_order=1 drop

    def test_diff_too_high_order_errors(self):
        """Differencing order > N - 1 raises."""
        p = [np.array([[0.0, 2.0, 5.0]])]
        with pytest.raises(ValueError, match="too high"):
            mpt.difference_events(p, None, None, [3], [0.0])

    def test_diff_negative_order_errors(self):
        """Negative differencing order raises."""
        p = [np.array([[0.0, 2.0, 5.0]])]
        with pytest.raises(ValueError, match="non-negative"):
            mpt.difference_events(p, None, None, [-1], [0.0])

    def test_diff_wrong_length_diff_orders(self):
        """diff_orders length mismatch raises."""
        p = [np.array([[0.0, 2.0, 5.0]])]
        with pytest.raises(ValueError, match="diff_orders"):
            mpt.difference_events(p, None, None, [1, 1], [0.0, 0.0])

    def test_diff_mismatched_event_counts(self):
        """Attributes with different N raise."""
        p = [np.array([[0.0, 2.0, 5.0]]), np.array([[0.0, 1.0]])]
        with pytest.raises(ValueError, match="event count"):
            mpt.difference_events(p, None, None, [0, 0], [0.0, 0.0])

    def test_diff_multi_slot_attribute_errors(self):
        """A K_a = 2 attribute must raise. Column-wise differencing would
        impose a cross-event slot correspondence that within-event slot
        exchangeability does not license."""
        p = [np.array([[60.0, 62.0, 64.0], [67.0, 69.0, 71.0]])]  # K_a = 2
        with pytest.raises(ValueError, match=r"K_a\s*=\s*2"):
            mpt.difference_events(p, None, None, [1], [0.0])

    def test_diff_empty_attribute_errors(self):
        """K_a = 0 (empty attribute) is likewise rejected by the
        K_a = 1 check; there is nothing to difference."""
        p = [np.zeros((0, 3))]
        with pytest.raises(ValueError, match=r"K_a\s*=\s*0"):
            mpt.difference_events(p, None, None, [1], [0.0])

    def test_diff_multi_slot_error_names_offending_attribute(self):
        """With a mix of K_a = 1 and K_a > 1 attributes, the error must
        fire and its message must name the offending attribute index."""
        p = [
            np.array([[60.0, 62.0, 64.0]]),                         # K_a = 1
            np.array([[60.0, 62.0, 64.0], [67.0, 69.0, 71.0]]),     # K_a = 2
        ]
        with pytest.raises(ValueError, match=r"Attribute 1"):
            mpt.difference_events(p, None, None, [1, 1], [0.0, 0.0])

    def test_diff_voices_as_attrs_pipeline(self):
        """Round-trip test for the voices-as-attributes pipeline: encode
        each voice as its own K_a = 1 attribute, difference each, then
        stack the differenced attributes into a single multi-slot
        attribute before build_exp_tens. Verify the resulting density
        has the expected shape and that eval_exp_tens returns finite
        non-negative values at a few query points."""
        pS = np.array([[72.0, 74.0, 76.0, 77.0]])   # soprano
        pA = np.array([[67.0, 69.0, 71.0, 72.0]])   # alto
        pT = np.array([[60.0, 62.0, 64.0, 65.0]])   # tenor
        pB = np.array([[48.0, 50.0, 52.0, 53.0]])   # bass
        p_attr = [pS, pA, pT, pB]
        groups = [0, 0, 0, 0]   # shared group (0-indexed in Python)
        p_diff, _ = mpt.difference_events(p_attr, None, groups, [1], [0.0])
        # Each differenced attribute should be 1 x 3.
        assert all(M.shape == (1, 3) for M in p_diff)
        # Stack into a single K_a = 4, N' = 3 multi-slot attribute.
        p_bundled = [np.vstack(p_diff)]
        assert p_bundled[0].shape == (4, 3)
        dens = mpt.build_exp_tens(
            p_bundled, None, [10.0], [1], None,
            [False], [False], [0.0],
            verbose=False,
        )
        assert dens.tag == "MaetDensity"
        # Finite non-negative at a few query points.
        x_query = np.array([-3.0, 0.0, 2.0, 4.0, 7.0])
        vals = mpt.eval_exp_tens(dens, x_query)
        assert np.all(np.isfinite(vals))
        assert np.all(vals >= 0.0)

    # --- translateEvents -----------------------------------------------

    def test_translate_zero_is_identity(self):
        """Translating by zero returns an unchanged copy of every
        attribute matrix."""
        p = [np.array([[60.0, 62.0, 64.0]]),
             np.array([[0.0, 1.0, 2.0]])]
        groups = [0, 1]
        out = mpt.translate_events(
            p, groups, {0: 0.0, 1: 0.0},
            [False, False], [False, False], [0.0, 0.0],
        )
        for a in range(2):
            np.testing.assert_array_equal(out[a], p[a])

    def test_translate_empty_offsets_is_identity(self):
        """Empty offsets dict returns a copy of every attribute matrix."""
        p = [np.array([[60.0, 62.0, 64.0]])]
        out = mpt.translate_events(p, [0], {}, [False], [False], [0.0])
        np.testing.assert_array_equal(out[0], p[0])
        # Result must be a copy, not the same object.
        assert out[0] is not p[0]

    def test_translate_does_not_mutate_input(self):
        """The input attribute matrices are not modified in place."""
        p_orig = np.array([[60.0, 62.0, 64.0]])
        p = [p_orig.copy()]
        _ = mpt.translate_events(
            p, [0], {0: 5.0}, [False], [False], [0.0],
        )
        np.testing.assert_array_equal(p[0], p_orig)

    def test_translate_nonperiodic_absolute(self):
        """Non-periodic absolute shift adds mu to every value."""
        p = [np.array([[60.0, 62.0, 64.0]])]
        out = mpt.translate_events(
            p, [0], {0: 5.0}, [False], [False], [0.0],
        )
        expected = np.array([[65.0, 67.0, 69.0]])
        np.testing.assert_allclose(out[0], expected)

    def test_translate_periodic_does_not_wrap(self):
        """Translation outputs unwrapped values, even on periodic
        groups; the periodic kernel in build_exp_tens handles wrapping
        downstream."""
        p = [np.array([[10.0, 11.0, 0.0]])]
        out = mpt.translate_events(
            p, [0], {0: 3.0}, [False], [True], [12.0],
        )
        expected = np.array([[13.0, 14.0, 3.0]])
        np.testing.assert_allclose(out[0], expected)

    def test_translate_periodic_negative_mu_does_not_wrap(self):
        """Negative offsets on periodic groups stay unwrapped."""
        p = [np.array([[1.0, 2.0]])]
        out = mpt.translate_events(
            p, [0], {0: -3.0}, [False], [True], [12.0],
        )
        expected = np.array([[-2.0, -1.0]])
        np.testing.assert_allclose(out[0], expected)

    def test_translate_period_ignored_when_is_per_false(self):
        """A finite periods entry is ignored when is_per is False —
        matches build_exp_tens convention, where periods may be
        declared as the natural period of a group's domain even when
        the group is being treated non-periodically for a particular
        analysis."""
        p = [np.array([[10.0, 11.0]])]
        out = mpt.translate_events(
            p, [0], {0: 5.0}, [False], [False], [12.0],
        )
        # Non-periodic: no wrap, values become 15, 16.
        expected = np.array([[15.0, 16.0]])
        np.testing.assert_allclose(out[0], expected)

    def test_translate_k_a_greater_than_one(self):
        """Multi-slot attribute (K_a > 1) shifts every slot's every
        value by the same mu (unlike windowing, where per-event scalar
        weights are not available when K_a > 1)."""
        p = [np.array([[60.0, 64.0, 67.0],
                       [63.0, 67.0, 70.0]])]
        out = mpt.translate_events(
            p, [0], {0: 5.0}, [False], [False], [0.0],
        )
        expected = np.array([[65.0, 69.0, 72.0],
                             [68.0, 72.0, 75.0]])
        np.testing.assert_allclose(out[0], expected)

    def test_translate_relative_emits_warning_and_no_op(self):
        """Translating a relative group emits TranslateEventsNoOpWarning
        and leaves the group unchanged."""
        p = [np.array([[60.0, 64.0, 67.0]])]
        with pytest.warns(mpt.TranslateEventsNoOpWarning):
            out = mpt.translate_events(
                p, [0], {0: 5.0}, [True], [False], [0.0],
            )
        np.testing.assert_array_equal(out[0], p[0])

    def test_translate_skips_groups_not_in_offsets(self):
        """Groups absent from the offsets dict are left unchanged."""
        p = [np.array([[60.0, 64.0]]),       # group 0 (pitch)
             np.array([[0.0, 1.0]])]         # group 1 (time)
        out = mpt.translate_events(
            p, [0, 1], {0: 5.0},
            [False, False], [False, False], [0.0, 0.0],
        )
        np.testing.assert_allclose(out[0], [[65.0, 69.0]])
        np.testing.assert_array_equal(out[1], p[1])

    def test_translate_multi_group_simultaneous(self):
        """Translating multiple groups at once shifts each
        independently by its own mu."""
        p = [np.array([[60.0, 64.0]]),
             np.array([[0.0, 1.0]])]
        out = mpt.translate_events(
            p, [0, 1], {0: 5.0, 1: 0.5},
            [False, False], [False, False], [0.0, 0.0],
        )
        np.testing.assert_allclose(out[0], [[65.0, 69.0]])
        np.testing.assert_allclose(out[1], [[0.5, 1.5]])

    def test_translate_multi_attribute_shared_group(self):
        """Multiple attributes sharing one group all shift by the
        same mu in one call."""
        p = [np.array([[60.0, 64.0]]),       # voice 1
             np.array([[67.0, 71.0]])]       # voice 2 -- same group
        groups = [0, 0]
        out = mpt.translate_events(
            p, groups, {0: 5.0}, [False], [False], [0.0],
        )
        np.testing.assert_allclose(out[0], [[65.0, 69.0]])
        np.testing.assert_allclose(out[1], [[72.0, 76.0]])

    def test_translate_composition(self):
        """translate(translate(p, mu), nu) == translate(p, mu + nu)
        on non-periodic groups."""
        p = [np.array([[60.0, 62.0, 64.0]])]
        once = mpt.translate_events(
            p, [0], {0: 5.0}, [False], [False], [0.0],
        )
        twice = mpt.translate_events(
            once, [0], {0: 3.0}, [False], [False], [0.0],
        )
        direct = mpt.translate_events(
            p, [0], {0: 8.0}, [False], [False], [0.0],
        )
        np.testing.assert_allclose(twice[0], direct[0])

    def test_translate_composition_periodic(self):
        """Composition is plain addition on periodic groups; values
        stay unwrapped (the periodic kernel handles wrap downstream)."""
        p = [np.array([[10.0, 11.0]])]
        once = mpt.translate_events(
            p, [0], {0: 7.0}, [False], [True], [12.0],
        )
        twice = mpt.translate_events(
            once, [0], {0: 9.0}, [False], [True], [12.0],
        )
        # 10 + 7 + 9 = 26;  11 + 7 + 9 = 27.  No wrap.
        np.testing.assert_allclose(twice[0], [[26.0, 27.0]])

    def test_translate_self_ip_invariance(self):
        """<f^mu, f^mu> = <f, f> for non-periodic absolute groups
        (translation preserves the un-normalised inner product)."""
        p = [np.array([[60.0, 64.0, 67.0]])]
        groups = [0]
        sigma, r = [0.15], [1]
        is_rel, is_per, periods = [False], [False], [0.0]
        M = mpt.build_exp_tens(
            p, None, sigma, r, groups, is_rel, is_per, periods,
            verbose=False,
        )
        for mu in (-3.0, 1.5, 7.0):
            p_mu = mpt.translate_events(
                p, groups, {0: mu}, is_rel, is_per, periods,
            )
            M_mu = mpt.build_exp_tens(
                p_mu, None, sigma, r, groups, is_rel, is_per, periods,
                verbose=False,
            )
            ip_self = mpt.cos_sim_exp_tens(M, M, verbose=False)
            ip_mu_self = mpt.cos_sim_exp_tens(M_mu, M_mu, verbose=False)
            np.testing.assert_allclose(ip_mu_self, ip_self, rtol=1e-12)

    def test_translate_self_ip_invariance_periodic(self):
        """Self-IP is also invariant under translation on periodic
        groups (periodic kernel commutes with rigid shifts)."""
        p = [np.array([[0.0, 4.0, 7.0]])]
        groups = [0]
        sigma, r = [0.15], [1]
        is_rel, is_per, periods = [False], [True], [12.0]
        M = mpt.build_exp_tens(
            p, None, sigma, r, groups, is_rel, is_per, periods,
            verbose=False,
        )
        for mu in (-7.0, 1.5, 6.0, 15.0):
            p_mu = mpt.translate_events(
                p, groups, {0: mu}, is_rel, is_per, periods,
            )
            M_mu = mpt.build_exp_tens(
                p_mu, None, sigma, r, groups, is_rel, is_per, periods,
                verbose=False,
            )
            np.testing.assert_allclose(
                mpt.cos_sim_exp_tens(M_mu, M_mu, verbose=False),
                mpt.cos_sim_exp_tens(M, M, verbose=False),
                rtol=1e-12,
            )

    def test_translate_recovers_transposition_peak(self):
        """The cos-sim sweep over mu peaks at the true transposition
        offset, with peak value 1 (within numerical tolerance)."""
        # C major triad as query; D major triad (= +2 st) as context.
        p_q = [np.array([[60.0, 64.0, 67.0]])]
        p_c = [np.array([[62.0, 66.0, 69.0]])]
        groups = [0]
        sigma, r = [0.15], [1]
        is_rel, is_per, periods = [False], [False], [0.0]
        M_q = mpt.build_exp_tens(
            p_q, None, sigma, r, groups, is_rel, is_per, periods,
            verbose=False,
        )
        best_mu, best_s = None, -np.inf
        for mu in np.arange(-12.0, 12.01, 0.25):
            p_c_mu = mpt.translate_events(
                p_c, groups, {0: mu}, is_rel, is_per, periods,
            )
            M_c_mu = mpt.build_exp_tens(
                p_c_mu, None, sigma, r, groups, is_rel, is_per, periods,
                verbose=False,
            )
            s = mpt.cos_sim_exp_tens(M_q, M_c_mu, verbose=False)
            if s > best_s:
                best_s, best_mu = s, mu
        assert abs(best_mu - (-2.0)) < 0.01
        assert best_s > 1.0 - 1e-9

    def test_translate_bad_group_index_raises(self):
        """Group indices outside [0, G) raise ValueError."""
        p = [np.array([[1.0, 2.0]])]
        with pytest.raises(ValueError, match="group index"):
            mpt.translate_events(
                p, [0], {5: 1.0}, [False], [False], [0.0],
            )

    def test_translate_non_finite_offset_raises(self):
        """A non-finite offset (NaN or inf) raises ValueError."""
        p = [np.array([[1.0, 2.0]])]
        with pytest.raises(ValueError, match="must be finite"):
            mpt.translate_events(
                p, [0], {0: float("nan")},
                [False], [False], [0.0],
            )
        with pytest.raises(ValueError, match="must be finite"):
            mpt.translate_events(
                p, [0], {0: float("inf")},
                [False], [False], [0.0],
            )

    def test_translate_wrong_length_is_rel_raises(self):
        """is_rel of wrong length raises ValueError."""
        p = [np.array([[1.0, 2.0]])]
        with pytest.raises(ValueError, match="is_rel"):
            mpt.translate_events(
                p, [0], {0: 1.0},
                [False, False], [False], [0.0],
            )

    def test_translate_wrong_length_is_per_raises(self):
        """is_per of wrong length raises ValueError."""
        p = [np.array([[1.0, 2.0]])]
        with pytest.raises(ValueError, match="is_per"):
            mpt.translate_events(
                p, [0], {0: 1.0},
                [False], [False, False], [0.0],
            )

    def test_translate_wrong_length_periods_raises(self):
        """periods of wrong length raises ValueError."""
        p = [np.array([[1.0, 2.0]])]
        with pytest.raises(ValueError, match="periods"):
            mpt.translate_events(
                p, [0], {0: 1.0},
                [False], [False], [0.0, 12.0],
            )

    def test_translate_non_dict_offsets_raises(self):
        """Invalid numeric-form offsets shapes raise ValueError. Under
        the orientation grammar (rows = attributes, columns = sweep),
        only ndim > 2 and 2-D shapes whose row count is neither 1 nor
        A are rejected; scalar and 1-D inputs are valid (broadcast)."""
        p = [np.array([[1.0, 2.0]]), np.array([[3.0, 4.0]])]
        # 3-D array is never valid.
        with pytest.raises(ValueError, match="ndim"):
            mpt.translate_events(
                p, [0, 1], np.zeros((2, 3, 1)),
                [False, False], [False, False], [0.0, 0.0],
            )
        # 2-D array with row count neither 1 nor A.
        # A = 2 here, row count 3 is invalid.
        with pytest.raises(ValueError, match="row count"):
            mpt.translate_events(
                p, [0, 1], np.zeros((3, 5)),
                [False, False], [False, False], [0.0, 0.0],
            )

    # --- translate_events polymorphic dict (per-attribute) --------------

    def test_translate_dict_scalar_broadcasts_within_group(self):
        """Dict scalar value broadcasts across every attribute in
        the group (no sweep)."""
        # Two attributes in one group; one scalar offset applies to both.
        p = [np.array([[60.0, 64.0]]), np.array([[67.0, 71.0]])]
        groups = [[0, 1]]   # both attributes in group 0
        is_rel, is_per, periods = [False], [False], [0.0]
        out = mpt.translate_events(
            p, groups, {0: 5.0}, is_rel, is_per, periods,
        )
        np.testing.assert_allclose(out[0], [[65.0, 69.0]])
        np.testing.assert_allclose(out[1], [[72.0, 76.0]])

    def test_translate_dict_2d_n_gx1_per_attribute(self):
        """Dict 2-D (n_g, 1) column gives each attribute its own
        offset within the group (single translation, per-attribute).
        Returns a matrix-mode output (length-1 outer wrapper) because
        the 2-D shape was explicit."""
        p = [np.array([[60.0, 64.0]]), np.array([[67.0, 71.0]])]
        groups = [[0, 1]]   # both attributes in group 0
        is_rel, is_per, periods = [False], [False], [0.0]
        # attr 0 shifts by 5, attr 1 by -3, single translation.
        out = mpt.translate_events(
            p, groups, {0: np.array([[5.0], [-3.0]])},
            is_rel, is_per, periods,
        )
        assert len(out) == 1
        np.testing.assert_allclose(out[0][0], [[65.0, 69.0]])
        np.testing.assert_allclose(out[0][1], [[64.0, 68.0]])

    def test_translate_dict_2d_per_attr_equivalent_to_scalar_when_uniform(self):
        """A 2-D (n_g, 1) dict value of all-equal entries gives the
        same translated values as the matching scalar broadcast."""
        p = [np.array([[60.0, 64.0]]), np.array([[67.0, 71.0]])]
        groups = [[0, 1]]
        is_rel, is_per, periods = [False], [False], [0.0]
        out_scalar = mpt.translate_events(
            p, groups, {0: 5.0}, is_rel, is_per, periods,
        )
        out_col = mpt.translate_events(
            p, groups, {0: np.array([[5.0], [5.0]])},
            is_rel, is_per, periods,
        )
        # scalar form is vector-mode; column form is matrix-mode
        # (length-1 outer wrapper). Compare the underlying arrays.
        np.testing.assert_allclose(out_scalar[0], out_col[0][0])
        np.testing.assert_allclose(out_scalar[1], out_col[0][1])

    def test_translate_dict_1d_length_ng_is_broadcast_sweep(self):
        """A 1-D length-n_g value is a broadcast sweep with M = n_g
        positions (not a per-attribute single translation). Under
        the orientation convention, 1-D is always a row vector."""
        p = [np.array([[60.0, 64.0]]), np.array([[67.0, 71.0]])]
        groups = [[0, 1]]
        is_rel, is_per, periods = [False], [False], [0.0]
        # 1-D length-2 is broadcast across both attributes, M=2 sweep.
        out = mpt.translate_events(
            p, groups, {0: np.array([5.0, -3.0])},
            is_rel, is_per, periods,
        )
        assert len(out) == 2
        # col 0: both attributes shift by 5.
        np.testing.assert_allclose(out[0][0], [[65.0, 69.0]])
        np.testing.assert_allclose(out[0][1], [[72.0, 76.0]])
        # col 1: both attributes shift by -3.
        np.testing.assert_allclose(out[1][0], [[57.0, 61.0]])
        np.testing.assert_allclose(out[1][1], [[64.0, 68.0]])

    def test_translate_dict_2d_1xM_broadcast_sweep(self):
        """Dict 2-D (1, M) value: broadcast within group, sweep with
        M positions."""
        p = [np.array([[60.0, 64.0]]), np.array([[67.0, 71.0]])]
        groups = [[0, 1]]
        is_rel, is_per, periods = [False], [False], [0.0]
        # M = 3 sweep, broadcast within group: every position shifts both
        # attributes by the same offset.
        out = mpt.translate_events(
            p, groups, {0: np.array([[0.0, 5.0, 10.0]])},
            is_rel, is_per, periods,
        )
        assert len(out) == 3
        np.testing.assert_allclose(out[0][0], [[60.0, 64.0]])
        np.testing.assert_allclose(out[0][1], [[67.0, 71.0]])
        np.testing.assert_allclose(out[2][0], [[70.0, 74.0]])
        np.testing.assert_allclose(out[2][1], [[77.0, 81.0]])

    def test_translate_dict_2d_per_attribute_sweep(self):
        """Dict 2-D (n_g, M) value: per-attribute sweep with M
        positions; each attribute gets its own sweep grid."""
        p = [np.array([[60.0, 64.0]]), np.array([[67.0, 71.0]])]
        groups = [[0, 1]]
        is_rel, is_per, periods = [False], [False], [0.0]
        # attr 0 sweeps [0, 5, 10]; attr 1 sweeps [1, 2, 3].
        offs = np.array([[0.0, 5.0, 10.0],
                         [1.0, 2.0, 3.0]])
        out = mpt.translate_events(
            p, groups, {0: offs}, is_rel, is_per, periods,
        )
        assert len(out) == 3
        # col 0: attr 0 by 0, attr 1 by 1.
        np.testing.assert_allclose(out[0][0], [[60.0, 64.0]])
        np.testing.assert_allclose(out[0][1], [[68.0, 72.0]])
        # col 2: attr 0 by 10, attr 1 by 3.
        np.testing.assert_allclose(out[2][0], [[70.0, 74.0]])
        np.testing.assert_allclose(out[2][1], [[70.0, 74.0]])

    def test_translate_dict_2d_wrong_rows_raises(self):
        """2-D dict value with row count != 1 and != n_g raises."""
        p = [np.array([[60.0, 64.0]]), np.array([[67.0, 71.0]])]
        groups = [[0, 1]]
        is_rel, is_per, periods = [False], [False], [0.0]
        with pytest.raises(ValueError, match="row count"):
            mpt.translate_events(
                p, groups, {0: np.zeros((3, 4))},
                is_rel, is_per, periods,
            )

    def test_translate_dict_M_disagreement_raises(self):
        """Two 2-D dict entries with different M dimensions raise."""
        p_g0 = np.array([[60.0, 64.0]])
        p_g1 = np.array([[0.0, 1.0]])
        groups = [0, 1]
        is_rel, is_per, periods = [False, False], [False, False], [0.0, 0.0]
        with pytest.raises(ValueError, match="sweep dimension"):
            mpt.translate_events(
                [p_g0, p_g1], groups,
                {0: np.array([[1.0, 2.0, 3.0]]),
                 1: np.array([[10.0, 20.0]])},
                is_rel, is_per, periods,
            )

    def test_translate_dict_mixed_scalar_and_swept(self):
        """A scalar dict value broadcasts across the sweep when
        another entry establishes M; the scalar group gets the same
        offset at every sweep position."""
        p_g0 = np.array([[60.0, 64.0]])
        p_g1 = np.array([[0.0, 1.0]])
        groups = [0, 1]
        is_rel, is_per, periods = [False, False], [False, False], [0.0, 0.0]
        out = mpt.translate_events(
            [p_g0, p_g1], groups,
            {0: 100.0, 1: np.array([[0.0, 0.5, 1.0]])},
            is_rel, is_per, periods,
        )
        assert len(out) == 3   # matrix mode triggered by group 1's 2-D entry
        for m in range(3):
            np.testing.assert_allclose(out[m][0], [[160.0, 164.0]])
        np.testing.assert_allclose(out[0][1], [[0.0, 1.0]])
        np.testing.assert_allclose(out[1][1], [[0.5, 1.5]])
        np.testing.assert_allclose(out[2][1], [[1.0, 2.0]])

    def test_translate_dict_per_attribute_nan_skips_attribute(self):
        """A NaN entry in a 2-D (n_g, 1) per-attribute column skips
        that attribute."""
        p = [np.array([[60.0, 64.0]]), np.array([[67.0, 71.0]])]
        groups = [[0, 1]]
        is_rel, is_per, periods = [False], [False], [0.0]
        # attr 0 translated by 5, attr 1 untouched.
        out = mpt.translate_events(
            p, groups, {0: np.array([[5.0], [float("nan")]])},
            is_rel, is_per, periods,
        )
        assert len(out) == 1
        np.testing.assert_allclose(out[0][0], [[65.0, 69.0]])
        np.testing.assert_allclose(out[0][1], [[67.0, 71.0]])

    def test_translate_dict_relative_group_per_attribute_warns(self):
        """Per-attribute offsets on a relative group still trigger
        the one-per-call no-op warning and pass through unchanged."""
        p = [np.array([[60.0, 64.0]]), np.array([[67.0, 71.0]])]
        groups = [[0, 1]]
        is_rel, is_per, periods = [True], [False], [0.0]
        with pytest.warns(mpt.TranslateEventsNoOpWarning):
            out = mpt.translate_events(
                p, groups, {0: np.array([[5.0], [-3.0]])},
                is_rel, is_per, periods,
            )
        assert len(out) == 1
        np.testing.assert_allclose(out[0][0], [[60.0, 64.0]])
        np.testing.assert_allclose(out[0][1], [[67.0, 71.0]])

    def test_translate_top_level_AxM_per_attribute_sweep(self):
        """Top-level (A, M) numeric matrix sweeps each attribute on
        its own row, independent of group structure. A multi-attribute
        group's attributes can therefore receive distinct sweep grids
        without using the dict form."""
        # Single group with two attributes (n_g = 2 = A).
        p = [np.array([[60.0, 64.0]]), np.array([[67.0, 71.0]])]
        groups = [[0, 1]]
        is_rel, is_per, periods = [False], [False], [0.0]
        # A = 2, M = 3. Row 0 sweeps attr 0; row 1 sweeps attr 1.
        offs = np.array([[0.0, 5.0, 10.0],
                         [1.0, 2.0, 3.0]])
        out = mpt.translate_events(
            p, groups, offs, is_rel, is_per, periods,
        )
        assert len(out) == 3
        np.testing.assert_allclose(out[0][0], [[60.0, 64.0]])
        np.testing.assert_allclose(out[0][1], [[68.0, 72.0]])
        np.testing.assert_allclose(out[2][0], [[70.0, 74.0]])
        np.testing.assert_allclose(out[2][1], [[70.0, 74.0]])

    def test_translate_top_level_1xM_broadcast_sweep(self):
        """Top-level (1, M) numeric row broadcasts the sweep across
        every attribute, regardless of group structure."""
        # Two groups, three attributes total.
        p = [np.array([[60.0, 64.0]]),
             np.array([[67.0, 71.0]]),
             np.array([[0.0, 1.0]])]
        groups = [[0, 1], [2]]
        is_rel, is_per, periods = [False, False], [False, False], [0.0, 0.0]
        offs = np.array([[0.0, 5.0, 10.0]])   # (1, 3) row vector
        out = mpt.translate_events(
            p, groups, offs, is_rel, is_per, periods,
        )
        assert len(out) == 3
        # col 2: every attribute shifted by 10.
        np.testing.assert_allclose(out[2][0], [[70.0, 74.0]])
        np.testing.assert_allclose(out[2][1], [[77.0, 81.0]])
        np.testing.assert_allclose(out[2][2], [[10.0, 11.0]])

    def test_translate_top_level_1D_is_broadcast_sweep(self):
        """A 1-D length-M array at the top level is interpreted as a
        (1, M) row vector under the orientation grammar: broadcast
        across all attributes, M-position sweep. (Per-attribute
        single translation requires 2-D (A, 1) shape.)"""
        p = [np.array([[60.0, 64.0]]), np.array([[67.0, 71.0]])]
        groups = [[0, 1]]
        is_rel, is_per, periods = [False], [False], [0.0]
        # 1-D length-3: broadcast sweep with M=3.
        out = mpt.translate_events(
            p, groups, np.array([0.0, 5.0, 10.0]),
            is_rel, is_per, periods,
        )
        assert len(out) == 3
        np.testing.assert_allclose(out[0][0], [[60.0, 64.0]])
        np.testing.assert_allclose(out[0][1], [[67.0, 71.0]])
        np.testing.assert_allclose(out[2][0], [[70.0, 74.0]])
        np.testing.assert_allclose(out[2][1], [[77.0, 81.0]])

    # --- translate_events matrix form (sweep) ----------------------------
        """Matrix offsets of shape (G, M) return a length-M list of
        length-A lists. M = 1 still keeps the outer wrapper."""
        p = [np.array([[60.0, 64.0, 67.0]])]
        groups, is_rel, is_per, periods = [0], [False], [False], [0.0]
        offs = np.array([[0.0, 100.0, 200.0]])  # (G=1, M=3)
        out = mpt.translate_events(p, groups, offs, is_rel, is_per, periods)
        assert isinstance(out, list) and len(out) == 3
        for col in out:
            assert isinstance(col, list) and len(col) == 1
            assert col[0].shape == (1, 3)
        # M = 1 case: outer wrapper retained.
        offs_one = np.array([[50.0]])
        out_one = mpt.translate_events(p, groups, offs_one, is_rel, is_per, periods)
        assert isinstance(out_one, list) and len(out_one) == 1
        assert isinstance(out_one[0], list) and len(out_one[0]) == 1

    def test_translate_matrix_per_column_equivalence(self):
        """Each column of an (A, M) per-attribute matrix gives the
        same translated values as a single per-attribute call with
        the corresponding (A, 1) column."""
        p = [np.array([[60.0, 64., 67.]]), np.array([[0., 1., 2.]])]
        # A = 2 here (two singleton groups), so (A, M) is the
        # per-attribute matrix form under the orientation grammar.
        groups, is_rel, is_per = [0, 1], [False, False], [True, False]
        periods = [1200.0, 0.0]
        offs_mat = np.array([
            [0.0,   100.0, 200.0,   -50.0],
            [0.0,   0.5,   1.0,     -0.25],
        ])
        sweep = mpt.translate_events(
            p, groups, offs_mat, is_rel, is_per, periods,
        )
        for m in range(offs_mat.shape[1]):
            # (A, 1) column → per-attribute single translation
            # (matrix-mode output, length-1 outer wrapper).
            one = mpt.translate_events(
                p, groups, offs_mat[:, m:m+1], is_rel, is_per, periods,
            )
            for a in range(2):
                np.testing.assert_allclose(sweep[m][a], one[0][a])

    def test_translate_matrix_nan_per_column(self):
        """NaN entries in matrix offsets skip translation on that
        column's group, even when other columns translate the same
        group."""
        p = [np.array([[60.0, 64.0]]), np.array([[0.0, 1.0]])]
        groups, is_rel, is_per = [0, 1], [False, False], [False, False]
        periods = [0.0, 0.0]
        offs_mat = np.array([
            [10.0, np.nan, 30.0],
            [np.nan, 5.0,  np.nan],
        ])
        out = mpt.translate_events(p, groups, offs_mat,
                                   is_rel, is_per, periods)
        # col 0: group 0 by +10, group 1 untouched
        np.testing.assert_allclose(out[0][0], np.array([[70.0, 74.0]]))
        np.testing.assert_allclose(out[0][1], np.array([[0.0, 1.0]]))
        # col 1: group 0 untouched, group 1 by +5
        np.testing.assert_allclose(out[1][0], np.array([[60.0, 64.0]]))
        np.testing.assert_allclose(out[1][1], np.array([[5.0, 6.0]]))
        # col 2: group 0 by +30, group 1 untouched
        np.testing.assert_allclose(out[2][0], np.array([[90.0, 94.0]]))
        np.testing.assert_allclose(out[2][1], np.array([[0.0, 1.0]]))

    def test_translate_matrix_periodic_does_not_wrap(self):
        """Matrix form on a periodic group leaves values unwrapped
        per column; the periodic kernel handles wrap downstream."""
        p = [np.array([[10.0, 1190.0]])]
        groups, is_rel, is_per = [0], [False], [True]
        periods = [1200.0]
        offs_mat = np.array([[100.0, 1100.0]])
        out = mpt.translate_events(p, groups, offs_mat,
                                   is_rel, is_per, periods)
        # col 0: 10 + 100 = 110; 1190 + 100 = 1290 (unwrapped)
        np.testing.assert_allclose(out[0][0], np.array([[110.0, 1290.0]]))
        # col 1: 10 + 1100 = 1110; 1190 + 1100 = 2290 (unwrapped)
        np.testing.assert_allclose(out[1][0], np.array([[1110.0, 2290.0]]))

    def test_translate_matrix_relative_warns_once(self):
        """A relative group with any finite offset across columns
        triggers exactly one warning, not one per column."""
        p = [np.array([[60.0, 64.0, 67.0]]),
             np.array([[0.0, 1.0, 2.0]])]
        groups = [0, 1]
        is_rel = [True, False]
        is_per = [False, False]
        periods = [0.0, 0.0]
        offs_mat = np.array([
            [10.0, 20.0, 30.0],   # relative group: every column finite
            [0.0,  0.5,  1.0],
        ])
        with warnings.catch_warnings(record=True) as w_list:
            warnings.simplefilter("always")
            out = mpt.translate_events(
                p, groups, offs_mat, is_rel, is_per, periods,
            )
        rel_warnings = [w for w in w_list
                        if "is_rel=True" in str(w.message)]
        assert len(rel_warnings) == 1
        # Relative group passes through unchanged on every column.
        for col in out:
            np.testing.assert_allclose(col[0], p[0])
        # Absolute group is translated normally per column.
        for m, mu in enumerate([0.0, 0.5, 1.0]):
            np.testing.assert_allclose(out[m][1], p[1] + mu)

    def test_translate_matrix_dict_and_array_vector_parity(self):
        """The three vector-form inputs (dict, 1-D ndarray, single
        column of a 2-D ndarray) give identical results."""
        p = [np.array([[60.0, 64.0, 67.0]])]
        groups, is_rel, is_per, periods = [0], [False], [False], [0.0]
        r_dict = mpt.translate_events(p, groups, {0: 100.0},
                                      is_rel, is_per, periods)
        r_vec  = mpt.translate_events(p, groups, np.array([100.0]),
                                      is_rel, is_per, periods)
        r_mat  = mpt.translate_events(p, groups, np.array([[100.0]]),
                                      is_rel, is_per, periods)
        np.testing.assert_allclose(r_dict[0], r_vec[0])
        np.testing.assert_allclose(r_mat[0][0], r_vec[0])

    # --- cos_sim_exp_tens raw-MA list mode --------------------------------

    def _ma_inputs(self):
        """Common 2-attribute (pitch, time) inputs used by raw-MA tests."""
        ref_pAttr = [
            np.array([[60., 62., 64., 65., 67., 69., 71.]]) * 100.0,
            np.array([[0., 1., 2., 3., 4., 5., 6.]]),
        ]
        qry_pAttr = [
            np.array([[60., 64., 67.]]) * 100.0,
            np.array([[0., 1., 2.]]),
        ]
        params = dict(
            sigma=[50., 0.3], r=[1, 1], groups=[0, 1],
            is_rel=[False, False], is_per=[True, False],
            periods=[1200., 0.],
        )
        return ref_pAttr, qry_pAttr, params

    def test_raw_ma_list_scalar_dispatch_unchanged(self):
        """A single MA p_attr on each side still scalar-dispatches."""
        ref_pAttr, qry_pAttr, p = self._ma_inputs()
        s = mpt.cos_sim_exp_tens(
            ref_pAttr, None, qry_pAttr, None,
            p['sigma'], p['r'], p['groups'],
            p['is_rel'], p['is_per'], p['periods'],
            verbose=False,
        )
        assert np.isscalar(s) or (isinstance(s, np.ndarray) and s.ndim == 0)

    def test_raw_ma_list_broadcast_returns_ndarray(self):
        """Scalar-vs-list raw-MA returns a length-M ndarray."""
        ref_pAttr, qry_pAttr, p = self._ma_inputs()
        offs = np.array([[0., 100., 200., -50.],
                         [0., 1., 2., 3.]])
        qry_swept = mpt.translate_events(
            qry_pAttr, p['groups'], offs,
            p['is_rel'], p['is_per'], p['periods'],
        )
        S = mpt.cos_sim_exp_tens(
            ref_pAttr, None, qry_swept, None,
            p['sigma'], p['r'], p['groups'],
            p['is_rel'], p['is_per'], p['periods'],
            verbose=False,
        )
        assert isinstance(S, np.ndarray)
        assert S.shape == (4,)
        assert np.all(np.isfinite(S))

    def test_raw_ma_list_parity_with_manual_build_loop(self):
        """Internalised build matches the explicit per-entry loop."""
        ref_pAttr, qry_pAttr, p = self._ma_inputs()
        offs = np.array([[-100., 0., 100., 200., 700.],
                         [0., 1., 2., 1., 3.]])
        qry_swept = mpt.translate_events(
            qry_pAttr, p['groups'], offs,
            p['is_rel'], p['is_per'], p['periods'],
        )
        S = mpt.cos_sim_exp_tens(
            ref_pAttr, None, qry_swept, None,
            p['sigma'], p['r'], p['groups'],
            p['is_rel'], p['is_per'], p['periods'],
            verbose=False,
        )
        dens_ref = mpt.build_exp_tens(
            ref_pAttr, None, p['sigma'], p['r'], p['groups'],
            p['is_rel'], p['is_per'], p['periods'], verbose=False,
        )
        S_manual = np.array([
            mpt.cos_sim_exp_tens(
                dens_ref,
                mpt.build_exp_tens(
                    pa, None, p['sigma'], p['r'], p['groups'],
                    p['is_rel'], p['is_per'], p['periods'],
                    verbose=False,
                ),
                verbose=False,
            )
            for pa in qry_swept
        ])
        np.testing.assert_allclose(S, S_manual, atol=1e-12)

    def test_raw_ma_list_symmetric_in_operand_order(self):
        """Cosine is symmetric; passing the list as first or second
        operand gives the same profile."""
        ref_pAttr, qry_pAttr, p = self._ma_inputs()
        offs = np.array([[0., 100., 200.],
                         [0., 0., 0.]])
        qry_swept = mpt.translate_events(
            qry_pAttr, p['groups'], offs,
            p['is_rel'], p['is_per'], p['periods'],
        )
        S_ref_first = mpt.cos_sim_exp_tens(
            ref_pAttr, None, qry_swept, None,
            p['sigma'], p['r'], p['groups'],
            p['is_rel'], p['is_per'], p['periods'],
            verbose=False,
        )
        S_list_first = mpt.cos_sim_exp_tens(
            qry_swept, None, ref_pAttr, None,
            p['sigma'], p['r'], p['groups'],
            p['is_rel'], p['is_per'], p['periods'],
            verbose=False,
        )
        np.testing.assert_allclose(S_ref_first, S_list_first, atol=1e-12)

    def test_raw_ma_list_vs_list_rejected(self):
        """List-vs-list raw-MA is rejected with a clear message."""
        ref_pAttr, qry_pAttr, p = self._ma_inputs()
        offs = np.array([[0., 100.],
                         [0., 0.]])
        list1 = mpt.translate_events(
            ref_pAttr, p['groups'], offs,
            p['is_rel'], p['is_per'], p['periods'],
        )
        list2 = mpt.translate_events(
            qry_pAttr, p['groups'], offs,
            p['is_rel'], p['is_per'], p['periods'],
        )
        with pytest.raises(TypeError, match="list-vs-list"):
            mpt.cos_sim_exp_tens(
                list1, None, list2, None,
                p['sigma'], p['r'], p['groups'],
                p['is_rel'], p['is_per'], p['periods'],
                verbose=False,
            )

    def test_raw_ma_list_consumes_translate_events_output(self):
        """End-to-end: translate_events → cos_sim_exp_tens raw-MA list
        finds the self-match peak at offset 0."""
        ref_pAttr, _, p = self._ma_inputs()
        # Sweep the reference against itself: peak should be at mu = 0
        # for both pitch and time.
        pitch_offs = np.array([-200., -100., 0., 100., 200.])
        time_offs  = np.zeros_like(pitch_offs)
        offs = np.vstack([pitch_offs, time_offs])
        ref_swept = mpt.translate_events(
            ref_pAttr, p['groups'], offs,
            p['is_rel'], p['is_per'], p['periods'],
        )
        S = mpt.cos_sim_exp_tens(
            ref_pAttr, None, ref_swept, None,
            p['sigma'], p['r'], p['groups'],
            p['is_rel'], p['is_per'], p['periods'],
            verbose=False,
        )
        assert int(np.argmax(S)) == 2  # offset 0 is the third entry
        assert S[2] == pytest.approx(1.0, abs=1e-9)

    # --- windowTensor / windowedSimilarity ------------------------------

    def _make_time_pitch_dens(self, events):
        """Build a pitch+time density from a list of (pitch, time) tuples."""
        pitches = np.array([[p for (p, _) in events]])
        times   = np.array([[t for (_, t) in events]])
        return mpt.build_exp_tens(
            [pitches, times], None,
            [10.0, 0.1], [1, 1], None,
            [False, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )

    def test_window_tensor_returns_tagged_object(self):
        """window_tensor returns a WindowedMaetDensity with the right tag."""
        dens = self._make_time_pitch_dens([(60, 0), (62, 1)])
        spec = {"size": [np.inf, 1.0], "mix": [0.0, 0.0],
                "centre": [np.zeros(1), np.array([0.5])]}
        wmd = mpt.window_tensor(dens, spec)
        assert isinstance(wmd, mpt.WindowedMaetDensity)
        assert wmd.tag == "WindowedMaetDensity"

    def test_window_infinite_size_equals_unwindowed(self):
        """A sufficiently wide window, centred at the context's mean
        time, gives cos_sim ~= 1.0 vs the unwindowed self.

        Under cross-correlation semantics the query is translated so
        that its effective-space mean moves onto the window centre, so
        a centred window at the context's mean position is the correct
        analogue of the no-window case.
        """
        events = [(60, 0), (62, 1), (64, 2), (65, 3)]
        dens = self._make_time_pitch_dens(events)
        s_self = mpt.cos_sim_exp_tens(dens, dens, verbose=False)
        t_mean = float(np.mean([t for (_, t) in events]))
        spec = {"size": [np.inf, 1e6], "mix": [0.0, 0.0],
                "centre": [np.zeros(1), np.array([t_mean])]}
        wmd = mpt.window_tensor(dens, spec)
        s_wide = _windowed_inner_product(dens, wmd, verbose=False)
        np.testing.assert_allclose(s_wide, s_self, rtol=1e-3, atol=1e-3)

    def test_window_no_groups_spec_means_identity(self):
        """All groups with size=inf means no window is effectively applied."""
        dens = self._make_time_pitch_dens([(60, 0), (62, 1)])
        spec = {"size": [np.inf, np.inf], "mix": [0.0, 0.0]}
        wmd = mpt.window_tensor(dens, spec)
        s = _windowed_inner_product(dens, wmd, verbose=False)
        # Should equal self-similarity (no windowing means identity).
        np.testing.assert_allclose(s, 1.0, atol=1e-6)

    def test_window_narrow_reduces_cos_sim(self):
        """A very narrow window gives cos_sim much less than 1.0."""
        dens = self._make_time_pitch_dens(
            [(60, 0), (62, 1), (64, 2), (65, 3)]
        )
        spec = {"size": [np.inf, 0.2], "mix": [0.0, 0.0],
                "centre": [np.zeros(1), np.array([0.0])]}
        wmd = mpt.window_tensor(dens, spec)
        s = _windowed_inner_product(dens, wmd, verbose=False)
        # Should be small — only ~1 event in window out of 4.
        assert s < 0.5

    def test_window_rectangular_1d_time(self):
        """Rectangular window (mix=1) on 1-D time works."""
        dens = self._make_time_pitch_dens([(60, 0), (62, 1), (64, 2)])
        spec = {"size": [np.inf, 0.5], "mix": [0.0, 1.0],
                "centre": [np.zeros(1), np.array([1.0])]}
        wmd = mpt.window_tensor(dens, spec)
        # Should run without error and give a finite value.
        s = _windowed_inner_product(dens, wmd, verbose=False)
        assert np.isfinite(s)
        assert 0 < s < 1

    def test_window_raised_rect_1d_time(self):
        """Raised-rectangular window (0 < mix < 1) on 1-D time works."""
        dens = self._make_time_pitch_dens([(60, 0), (62, 1), (64, 2)])
        spec = {"size": [np.inf, 0.5], "mix": [0.0, 0.5],
                "centre": [np.zeros(1), np.array([1.0])]}
        wmd = mpt.window_tensor(dens, spec)
        s = _windowed_inner_product(dens, wmd, verbose=False)
        assert np.isfinite(s)
        assert 0 < s < 1

    def test_window_multi_d_rel_gaussian_works(self):
        """Multi-D relative group with Gaussian window (mix=0) works."""
        pitch = np.array([[60.0, 62.0], [64.0, 65.0], [67.0, 69.0]])
        dens = mpt.build_exp_tens(
            [pitch], None,
            [10.0], [3], None,
            [True], [True], [1200.0],
            verbose=False,
        )
        spec = {"size": [1.0], "mix": [0.0],
                "centre": [np.array([50.0, 100.0])]}
        wmd = mpt.window_tensor(dens, spec)
        s = _windowed_inner_product(dens, wmd, verbose=False)
        assert np.isfinite(s)
        assert 0 <= s <= 1

    def test_window_multi_d_rel_rect_raises(self):
        """Multi-D relative group with rect (mix=1) raises
        NotImplementedError."""
        pitch = np.array([[60.0, 62.0], [64.0, 65.0], [67.0, 69.0]])
        dens = mpt.build_exp_tens(
            [pitch], None,
            [10.0], [3], None,
            [True], [True], [1200.0],
            verbose=False,
        )
        spec = {"size": [1.0], "mix": [1.0],
                "centre": [np.array([50.0, 100.0])]}
        wmd = mpt.window_tensor(dens, spec)
        with pytest.raises(NotImplementedError, match="Multi-D relative"):
            _windowed_inner_product(dens, wmd, verbose=False)

    def test_window_multi_d_rel_raised_rect_raises(self):
        """Multi-D relative with raised-rect also raises."""
        pitch = np.array([[60.0, 62.0], [64.0, 65.0], [67.0, 69.0]])
        dens = mpt.build_exp_tens(
            [pitch], None,
            [10.0], [3], None,
            [True], [True], [1200.0],
            verbose=False,
        )
        spec = {"size": [1.0], "mix": [0.5],
                "centre": [np.array([50.0, 100.0])]}
        wmd = mpt.window_tensor(dens, spec)
        with pytest.raises(NotImplementedError, match="Multi-D relative"):
            _windowed_inner_product(dens, wmd, verbose=False)

    def test_window_entropy_works_on_rect(self):
        """entropy_exp_tens works on a rectangular-windowed density,
        even on a multi-D relative group (evaluation is pointwise)."""
        pitch = np.array([[60.0, 62.0], [64.0, 65.0], [67.0, 69.0]])
        dens = mpt.build_exp_tens(
            [pitch], None,
            [10.0], [3], None,
            [True], [True], [1200.0],
            verbose=False,
        )
        spec = {"size": [1.0], "mix": [1.0],
                "centre": [np.array([50.0, 100.0])]}
        wmd = mpt.window_tensor(dens, spec)
        # dim = 2, grid 20x20 = 400 points
        H = mpt.entropy_exp_tens(wmd, n_points_per_dim=20)
        assert np.isfinite(H)

    def test_window_entropy_narrower_lower(self):
        """Narrower window gives lower entropy on a time-windowed density."""
        dens = self._make_time_pitch_dens(
            [(60, 0), (62, 1), (64, 2), (65, 3)]
        )
        H_base = mpt.entropy_exp_tens(
            dens, x_min=[0.0, -1.0], x_max=[1200.0, 4.0],
            n_points_per_dim=60,
        )
        spec_narrow = {"size": [np.inf, 0.3], "mix": [0.0, 0.0],
                       "centre": [np.zeros(1), np.array([1.0])]}
        wmd = mpt.window_tensor(dens, spec_narrow)
        H_narrow = mpt.entropy_exp_tens(
            wmd, x_min=[0.0, -1.0], x_max=[1200.0, 4.0],
            n_points_per_dim=60,
        )
        assert H_narrow < H_base

    def test_windowed_similarity_profile(self):
        """windowed_similarity returns a length-M profile that peaks at the
        offset where the context has a pitch matching the query."""
        ctx = mpt.build_exp_tens(
            [np.array([[60.0, 62.0, 64.0, 65.0]]),
             np.array([[0.0, 1.0, 2.0, 3.0]])],
            None, [0.5, 0.1], [1, 1], None,
            [False, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )
        # Query: single event at pitch 62, time 0 (fixed, not swept).
        q = mpt.build_exp_tens(
            [np.array([[62.0]]), np.array([[0.0]])], None,
            [0.5, 0.1], [1, 1], None,
            [False, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )
        # Sweep offsets in time. Query-centroid is at t=0, so an offset
        # of t means the window sits at absolute context time t. The
        # pitch-62 context event lives at t=1.
        M = 21
        offs = np.linspace(-0.5, 3.5, M)
        offsets = np.zeros((2, M))
        offsets[1, :] = offs
        spec = {"size": [np.inf, 0.3], "mix": [0.0, 0.0]}
        profile = mpt.windowed_similarity(ctx, q, spec, offsets, verbose=False)
        peak_idx = np.argmax(profile)
        assert abs(offs[peak_idx] - 1.0) < 0.3

    def test_windowed_similarity_vector_length(self):
        """windowed_similarity output has length M."""
        dens = self._make_time_pitch_dens([(60, 0), (62, 1)])
        offsets = np.zeros((2, 7))
        offsets[1, :] = np.linspace(0, 1, 7)
        spec = {"size": [np.inf, 0.5], "mix": [0.0, 0.0]}
        profile = mpt.windowed_similarity(dens, dens, spec, offsets,
                                         verbose=False)
        assert profile.shape == (7,)

    def test_windowed_similarity_reference_none_equals_default(self):
        """reference=None reproduces the default unweighted-centroid path.

        Explicit None must give identical numerical output to the call
        without the keyword. This guards backward compatibility when
        the reference= keyword is added to the API.
        """
        ctx = self._make_time_pitch_dens(
            [(60, 0), (62, 1), (64, 2), (65, 3)])
        q = self._make_time_pitch_dens([(62, 0)])
        M = 11
        offsets = np.zeros((2, M))
        offsets[1, :] = np.linspace(-0.5, 3.5, M)
        spec = {"size": [np.inf, 0.3], "mix": [0.0, 0.0]}
        prof_default  = mpt.windowed_similarity(ctx, q, spec, offsets,
                                             verbose=False)
        prof_explicit = mpt.windowed_similarity(ctx, q, spec, offsets,
                                             reference=None,
                                             verbose=False)
        assert np.allclose(prof_default, prof_explicit)

    def test_windowed_similarity_reference_shifts_profile(self):
        """A user-supplied reference shifts the profile by exactly the
        offset between the new reference and the default (unweighted
        centroid), over the span of the sweep.

        Concretely: calling windowed_similarity with reference = mu_default
        + shift should produce the same profile values at every sweep
        column as the default call shifted by -shift in offset space.
        """
        ctx = self._make_time_pitch_dens(
            [(60, 0), (62, 1), (64, 2), (65, 3)])
        q = self._make_time_pitch_dens([(62, 0), (64, 0.5)])
        # Default reference (unweighted centroid per attribute).
        mu_default = [q.centres[a].mean(axis=1) for a in range(q.n_attrs)]
        shift = np.array([0.0])    # time-attribute shift only
        # Shift the time reference by +0.2 s. (Pitch reference unchanged.)
        ref_shifted = [mu_default[0].copy(),
                       mu_default[1] + 0.2]
        # Sweep only in time.
        M = 21
        offs_time = np.linspace(-1.0, 3.0, M)
        offsets = np.zeros((2, M))
        offsets[1, :] = offs_time
        spec = {"size": [np.inf, 0.3], "mix": [0.0, 0.0]}
        prof_default = mpt.windowed_similarity(ctx, q, spec, offsets,
                                            verbose=False)
        prof_shifted = mpt.windowed_similarity(ctx, q, spec, offsets,
                                            reference=ref_shifted,
                                            verbose=False)
        # At sweep column m (offset o), the default places the window
        # at mu_default + o; the shifted call places it at mu_default +
        # 0.2 + o. So prof_shifted at offset o equals prof_default at
        # offset o + 0.2. Check this for interior indices.
        for m in range(M):
            target_off = offs_time[m] + 0.2
            # Find the nearest default column to target_off
            j = int(np.argmin(np.abs(offs_time - target_off)))
            if abs(offs_time[j] - target_off) < 1e-9:
                assert abs(prof_shifted[m] - prof_default[j]) < 1e-10

    def test_windowed_similarity_reference_wrong_length_raises(self):
        """reference with wrong per-attribute length raises ValueError."""
        q = self._make_time_pitch_dens([(62, 0)])
        ctx = self._make_time_pitch_dens([(60, 0), (62, 1)])
        offsets = np.zeros((2, 3))
        offsets[1, :] = np.linspace(0, 1, 3)
        spec = {"size": [np.inf, 0.3], "mix": [0.0, 0.0]}
        # Too few entries (1 instead of 2)
        with pytest.raises(ValueError, match="reference"):
            mpt.windowed_similarity(ctx, q, spec, offsets,
                                  reference=[np.array([0.0])],
                                  verbose=False)
        # Correct number of entries but wrong inner length
        with pytest.raises(ValueError, match="reference"):
            mpt.windowed_similarity(ctx, q, spec, offsets,
                                  reference=[np.array([0.0, 0.0]),
                                             np.array([0.0])],
                                  verbose=False)

    # ---- Periodic windowing: wrapped-Gaussian image summation -------
    #
    # For periodic groups, the window is the wrapped Gaussian (or
    # wrapped rect-conv-Gaussian for mix > 0): the sum of line-case
    # window functions at all periodic images of the centre. The
    # toolbox sums these adaptively until the latest image-pair's
    # contribution falls below 1e-12 of the running maximum. The
    # WindowedSimilarityPeriodicApproxWarning of pre-v2.2 has been
    # removed because there is no longer an approximation to warn
    # about. See User Guide §3.1 "Post-tensor windowing".

    def test_windowed_similarity_periodic_emits_no_warning(self):
        """A periodic windowed group must not emit any UserWarning
        relating to the line-case approximation. The pre-v2.2
        WindowedSimilarityPeriodicApproxWarning is gone."""
        q   = self._make_time_pitch_dens([(60, 0)])
        ctx = self._make_time_pitch_dens([(60, 0), (62, 1)])
        offsets = np.zeros((2, 3))
        offsets[1, :] = np.linspace(0, 1, 3)
        spec = {"size": [40.0, 0.3], "mix": [0.0, 0.0]}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _ = mpt.windowed_similarity(ctx, q, spec, offsets,
                                        verbose=False)
        # No "approximation" or "line-case" themed warnings.
        bad = [w for w in caught
               if "approximat" in str(w.message).lower()
               or "line-case" in str(w.message).lower()
               or "periodic" in str(w.category.__name__).lower()]
        assert len(bad) == 0, (
            f"Expected no approximation-themed warnings; got: "
            f"{[str(w.message) for w in bad]}"
        )

    def test_windowed_similarity_periodic_warning_class_removed(self):
        """The WindowedSimilarityPeriodicApproxWarning class is no
        longer importable from mpt (removed in v2.2)."""
        assert not hasattr(mpt, "WindowedSimilarityPeriodicApproxWarning")

    def test_eval_exp_tens_periodic_representative_equivalence(self):
        """eval_exp_tens on a windowed periodic density must return
        identical values at periodic-equivalent query points (X, X+P,
        X-P, ...). Under the pre-v2.2 line-case window this was
        broken: the window did not wrap, so different representatives
        of the same point gave different answers."""
        P = 12.0
        sigma = 1.0
        pitches = np.array([[3.0, 7.0]])
        dens = mpt.build_exp_tens(
            [pitches], None,
            [sigma], [1], [0],
            [False], [True], [P],
            verbose=False,
        )
        mu = 2.0
        sigma_w = 3.0
        win_size = sigma_w / sigma
        wmd = mpt.window_tensor(
            dens, {"size": win_size, "mix": 0.0,
                   "centre": [np.array([mu])]},
        )
        # Three representatives of the same periodic point X = 0.5.
        v0 = float(np.asarray(mpt.eval_exp_tens(
            wmd, np.array([[0.5]]), verbose=False
        )).flatten()[0])
        v_plus = float(np.asarray(mpt.eval_exp_tens(
            wmd, np.array([[0.5 + P]]), verbose=False
        )).flatten()[0])
        v_minus = float(np.asarray(mpt.eval_exp_tens(
            wmd, np.array([[0.5 - P]]), verbose=False
        )).flatten()[0])
        assert np.isclose(v0, v_plus, atol=0, rtol=1e-12), (
            f"X=0.5 gave {v0:.12g}, X=0.5+P gave {v_plus:.12g}"
        )
        assert np.isclose(v0, v_minus, atol=0, rtol=1e-12), (
            f"X=0.5 gave {v0:.12g}, X=0.5-P gave {v_minus:.12g}"
        )


    def test_window_tensor_validates_shapes(self):
        """window_tensor rejects invalid size / mix / centre shapes."""
        dens = self._make_time_pitch_dens([(60, 0), (62, 1)])
        # Wrong size length
        with pytest.raises(ValueError, match="size"):
            mpt.window_tensor(dens, {"size": [1.0, 1.0, 1.0],
                                     "mix": [0.0, 0.0]})
        # Mix out of range
        with pytest.raises(ValueError, match="mix"):
            mpt.window_tensor(dens, {"size": [1.0, 1.0],
                                     "mix": [0.0, 1.5]})
        # Wrong centre length in list form
        with pytest.raises(ValueError, match="centre"):
            mpt.window_tensor(dens, {
                "size": [1.0, 1.0], "mix": [0.0, 0.0],
                "centre": [np.array([0.0, 0.0])],  # length 1 list, need A=2
            })
