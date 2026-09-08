"""Tests for Multi-Attribute Expectation Tensor (MAET, v3).

Mirror of MATLAB tests/test_maet.m.
"""
import numpy as np
import pytest

import mpt
from mpt._utils import position_variance


class TestMAET:
    """Multi-attribute expectation tensor tests.

    The v3 extension to ``build_exp_tens``. The single-multiset
    legacy path is covered by ``TestTensor`` above; these tests focus on
    the MAET-specific behaviours: Single-multiset equivalence under degenerate mapping,
    per-attribute perm/comb enumeration, weight broadcasting, group
    canonicalisation, NaN handling for variable-size events, and the
    error paths introduced by the per-attribute / per-group parameter
    structure.
    """

    # --- Single-multiset equivalence: MA at (A=1, N=1) == the flat form ---

    def test_ma_matches_single_multiset_abs(self):
        """MA with one event and one attribute reproduces single-multiset bit-for-bit."""
        p = [0.0, 400.0, 700.0]
        w = [1.0, 0.7, 0.5]
        sigma, r, is_rel, is_per, period = 10.0, 2, False, True, 1200.0

        dens_sm = mpt.build_exp_tens(
            p, w, sigma, r, is_rel, is_per, period, verbose=False
        )

        p_attr = [np.array(p, dtype=float).reshape(3, 1)]
        w_ma = [np.array(w, dtype=float).reshape(3, 1)]
        dens_ma = mpt.build_exp_tens(
            p_attr, w_ma, [sigma], [r], 
            [is_rel], [is_per], [period], verbose=False,
        )

        assert dens_ma.tag == "MaetDensity"
        assert dens_ma.n_j == dens_sm.n_j
        assert dens_ma.n_k == dens_sm.n_k
        np.testing.assert_array_equal(dens_ma.u_perm[0], dens_sm.u_perm[0])
        np.testing.assert_array_equal(dens_ma.v_comb[0], dens_sm.v_comb[0])
        np.testing.assert_array_equal(dens_ma.centres[0], dens_sm.centres[0])
        np.testing.assert_array_almost_equal(dens_ma.w_j, dens_sm.w_j)
        np.testing.assert_array_almost_equal(dens_ma.wv_comb, dens_sm.wv_comb)

    def test_ma_matches_single_multiset_rel(self):
        """Centres reduction: is_rel=True collapses r_a dims to r_a-1."""
        p = [0.0, 400.0, 700.0]
        w = [1.0, 0.7, 0.5]
        sigma, r, is_rel, is_per, period = 10.0, 2, True, True, 1200.0

        dens_sm = mpt.build_exp_tens(
            p, w, sigma, r, is_rel, is_per, period, verbose=False
        )
        p_attr = [np.array(p, dtype=float).reshape(3, 1)]
        w_ma = [np.array(w, dtype=float).reshape(3, 1)]
        dens_ma = mpt.build_exp_tens(
            p_attr, w_ma, [sigma], [r], 
            [is_rel], [is_per], [period], verbose=False,
        )
        assert dens_ma.centres[0].shape == (r - 1, dens_ma.n_j)
        np.testing.assert_array_equal(dens_ma.centres[0], dens_sm.centres[0])

    # --- Struct basics ------------------------------------------------

    def test_ma_struct_fields(self):
        """Essential fields for a pitch + time two-attribute build."""
        pitch = np.array([[0, 12], [4, 15], [7, 19]], dtype=float)  # 3 x 2
        time = np.array([[0.0, 1.0]])                               # 1 x 2
        dens = mpt.build_exp_tens(
            [pitch, time], None,
            [10.0, 0.1], [3, 1], 
            [True, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )
        assert dens.tag == "MaetDensity"
        assert dens.n_attrs == 2
        assert dens.n == 2
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
            [10.0, 0.1], [3, 1], 
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
            [10.0, 0.1], [3, 1], 
            [True, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )
        np.testing.assert_array_equal(dens.event_of_j, np.repeat([0, 1, 2], 6))
        np.testing.assert_array_equal(dens.event_of_k, [0, 1, 2])

    # --- Weight broadcasting -----------------------------------------

    def test_ma_weight_none(self):
        pitch = np.array([[0, 4], [4, 8]], dtype=float)
        dens = mpt.build_exp_tens(
            [pitch], None, [10.0], [2], 
            [False], [True], [1200.0], verbose=False,
        )
        np.testing.assert_array_equal(dens.w[0], np.ones((2, 2)))

    def test_ma_weight_scalar_top_level(self):
        pitch = np.array([[0, 4], [4, 8]], dtype=float)
        dens = mpt.build_exp_tens(
            [pitch], 0.5, [10.0], [2], 
            [False], [True], [1200.0], verbose=False,
        )
        np.testing.assert_array_equal(dens.w[0], np.full((2, 2), 0.5))

    def test_ma_weight_2d_per_event_row(self):
        pitch = np.array([[0, 4, 5], [4, 8, 6]], dtype=float)  # K=2, N=3
        w_row = np.array([[0.5, 1.0, 2.0]])                    # (1, 3)
        dens = mpt.build_exp_tens(
            [pitch], [w_row], [10.0], [2], 
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
            [pitch], [w_col], [10.0], [2], 
            [False], [True], [1200.0], verbose=False,
        )
        np.testing.assert_array_equal(
            dens.w[0], np.tile([[0.5], [1.0], [2.0]], (1, 3))
        )

    def test_ma_weight_2d_full_matrix(self):
        pitch = np.array([[0, 4], [4, 8]], dtype=float)
        W = np.array([[0.1, 0.2], [0.3, 0.4]])
        dens = mpt.build_exp_tens(
            [pitch], [W], [10.0], [2], 
            [False], [True], [1200.0], verbose=False,
        )
        np.testing.assert_array_equal(dens.w[0], W)

    def test_ma_weight_1d_per_event_disambiguated(self):
        pitch = np.array([[0, 4], [4, 8], [7, 9]], dtype=float)  # K=3, N=2
        w_1d = np.array([0.5, 1.0])  # length N=2 (not K=3) -> per-event
        dens = mpt.build_exp_tens(
            [pitch], [w_1d], [10.0], [2], 
            [False], [True], [1200.0], verbose=False,
        )
        expected = np.array([[0.5, 1.0], [0.5, 1.0], [0.5, 1.0]])
        np.testing.assert_array_equal(dens.w[0], expected)

    def test_ma_weight_1d_per_position_disambiguated(self):
        pitch = np.array([[0, 4], [4, 8], [7, 9]], dtype=float)  # K=3, N=2
        w_1d = np.array([0.5, 1.0, 2.0])  # length K=3 (not N=2) -> per-position
        dens = mpt.build_exp_tens(
            [pitch], [w_1d], [10.0], [2], 
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
                [pitch], [w_1d], [10.0], [2], 
                [False], [True], [1200.0], verbose=False,
            )

    # --- Groups -------------------------------------------------------

    # --- NaN-padded variable-size events -----------------------------

    def test_ma_nan_padding(self):
        # Event 0: 3 pitches. Event 1: 2 pitches (third position NaN).
        pitch = np.array([[0, 0], [4, 4], [7, np.nan]], dtype=float)
        time = np.array([[0.0, 1.0]])
        dens = mpt.build_exp_tens(
            [pitch, time], None, [10.0, 0.1], [2, 1], 
            [False, False], [True, False], [1200.0, 0.0], verbose=False,
        )
        # Event 0: P(3,2)=6 perms, C(3,2)=3 combs. Event 1: P(2,2)=2, C(2,2)=1.
        # Cartesian x 1 time = same. Totals: n_j = 8, n_k = 4.
        assert dens.n_j == 8
        assert dens.n_k == 4

    # --- Per-tuple weight factorisation ------------------------------

    def test_ma_per_tuple_weight_product(self):
        # One event, K_p=2 with position weights 2 and 3; r_p=2.
        # Time K=1, position weight 5; r_t=1.
        # Each perm tuple weight = (w_i * w_j for pitch) * 5 for time.
        pitch = np.array([[0.0], [4.0]])
        time = np.array([[1.5]])
        w_pitch = np.array([[2.0], [3.0]])
        w_time = np.array([[5.0]])
        dens = mpt.build_exp_tens(
            [pitch, time], [w_pitch, w_time],
            [10.0, 0.1], [2, 1], 
            [False, False], [True, False], [1200.0, 0.0], verbose=False,
        )
        # 2 pitch perms, each with weight 2*3*5 = 30.
        np.testing.assert_array_almost_equal(dens.w_j, [30.0, 30.0])
        # 1 pitch comb, weight 2*3*5 = 30.
        np.testing.assert_array_almost_equal(dens.wv_comb, [30.0])

    # --- Error paths -------------------------------------------------

    def test_ma_insufficient_values_errors(self):
        # Event 1 has 1 valid value, r=2 -> error
        pitch = np.array(
            [[0, 0], [4, np.nan], [np.nan, np.nan]], dtype=float
        )
        with pytest.raises(ValueError, match="non-NaN value"):
            mpt.build_exp_tens(
                [pitch], None, [10.0], [2], 
                [False], [True], [1200.0], verbose=False,
            )

    def test_ma_wrong_r_vec_length(self):
        pitch = np.array([[0, 4]], dtype=float)
        with pytest.raises(ValueError, match="r_vec"):
            mpt.build_exp_tens(
                [pitch, pitch], None, [10.0, 10.0], [1], 
                [False, False], [True, True], [1200.0, 1200.0], verbose=False,
            )

    def test_ma_wrong_sigma_length(self):
        pitch = np.array([[0, 4]], dtype=float)
        with pytest.raises(ValueError, match="sigma_vec"):
            mpt.build_exp_tens(
                [pitch, pitch], None, [10.0], [1, 1], 
                [False, False], [True, True], [1200.0, 1200.0], verbose=False,
            )

    def test_ma_mismatched_event_counts(self):
        pitch = np.array([[0, 4]], dtype=float)       # N=2
        time = np.array([[0.0, 1.0, 2.0]])             # N=3
        with pytest.raises(ValueError, match="share N"):
            mpt.build_exp_tens(
                [pitch, time], None, [10.0, 0.1], [1, 1], 
                [False, False], [True, False], [1200.0, 0.0], verbose=False,
            )

    def test_ma_isrel_r1_warns(self):
        pitch = np.array([[0, 4]], dtype=float)
        with pytest.warns(UserWarning, match="degenerate"):
            mpt.build_exp_tens(
                [pitch], None, [10.0], [1], 
                [True], [True], [1200.0], verbose=False,
            )

    def test_ma_wrong_positional_count(self):
        pitch = np.array([[0, 4]], dtype=float)
        # 6 positional args for MA is wrong (should be 7)
        with pytest.raises(ValueError, match="7 or 8 positional"):
            mpt.build_exp_tens(
                [pitch], None, [10.0], [1],
                [False], [True], verbose=False,
            )

    # --- evalExpTens MA path ------------------------------------------

    def test_ma_eval_matches_sa_abs(self):
        """MA eval matches single-multiset at the same query points (is_rel=False)."""
        p = [0.0, 400.0, 700.0]
        w = [1.0, 0.7, 0.5]
        sigma, r, is_rel, is_per, period = 10.0, 2, False, True, 1200.0
        x_sm = np.array([[100, 500], [300, 600]], dtype=float)  # 2 x 2 (single-multiset)

        dens_sm = mpt.build_exp_tens(
            p, w, sigma, r, is_rel, is_per, period, verbose=False
        )
        vals_sm = mpt.eval_exp_tens(dens_sm, x_sm, verbose=False)

        p_attr = [np.array(p, dtype=float).reshape(3, 1)]
        w_ma = [np.array(w, dtype=float).reshape(3, 1)]
        dens_ma = mpt.build_exp_tens(
            p_attr, w_ma, [sigma], [r], 
            [is_rel], [is_per], [period], verbose=False,
        )
        vals_ma_cell = mpt.eval_exp_tens(dens_ma, [x_sm], verbose=False)
        vals_ma_mat = mpt.eval_exp_tens(dens_ma, x_sm, verbose=False)

        np.testing.assert_allclose(vals_ma_cell, vals_sm, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(vals_ma_mat, vals_sm, rtol=1e-12, atol=1e-12)

    def test_ma_eval_matches_sa_rel(self):
        """Same with is_rel=True: reduced-dim query points."""
        p = [0.0, 400.0, 700.0]
        w = [1.0, 0.7, 0.5]
        sigma, r, is_rel, is_per, period = 10.0, 3, True, True, 1200.0
        x_sm = np.array([[400, 200], [700, 500]], dtype=float)  # (r-1) x nQ

        dens_sm = mpt.build_exp_tens(
            p, w, sigma, r, is_rel, is_per, period, verbose=False
        )
        p_attr = [np.array(p, dtype=float).reshape(3, 1)]
        w_ma = [np.array(w, dtype=float).reshape(3, 1)]
        dens_ma = mpt.build_exp_tens(
            p_attr, w_ma, [sigma], [r], 
            [is_rel], [is_per], [period], verbose=False,
        )

        vals_sm = mpt.eval_exp_tens(dens_sm, x_sm, verbose=False)
        vals_ma = mpt.eval_exp_tens(dens_ma, [x_sm], verbose=False)
        np.testing.assert_allclose(vals_ma, vals_sm, rtol=1e-12, atol=1e-12)

    def test_ma_eval_normalisation_matches_single_multiset(self):
        """'gaussian' and 'pdf' normalisation modes match single-multiset."""
        p = [0.0, 400.0, 700.0]
        w = [1.0, 0.7, 0.5]
        sigma, r, is_rel, is_per, period = 10.0, 3, True, True, 1200.0
        x_sm = np.array([[400, 200], [700, 500]], dtype=float)

        dens_sm = mpt.build_exp_tens(
            p, w, sigma, r, is_rel, is_per, period, verbose=False
        )
        p_attr = [np.array(p, dtype=float).reshape(3, 1)]
        w_ma = [np.array(w, dtype=float).reshape(3, 1)]
        dens_ma = mpt.build_exp_tens(
            p_attr, w_ma, [sigma], [r], 
            [is_rel], [is_per], [period], verbose=False,
        )
        for mode in ("gaussian", "pdf"):
            vals_sm = mpt.eval_exp_tens(dens_sm, x_sm, mode, verbose=False)
            vals_ma = mpt.eval_exp_tens(dens_ma, [x_sm], mode, verbose=False)
            np.testing.assert_allclose(vals_ma, vals_sm, rtol=1e-12, atol=1e-12)

    def test_ma_eval_cell_vs_matrix_forms_agree(self):
        """Cell form and single-matrix form give identical results."""
        pitch = np.array([[0.0, 4.0, 7.0]]).T        # K=3, N=1
        time  = np.array([[1.0]])                     # K=1, N=1
        dens = mpt.build_exp_tens(
            [pitch, time], None,
            [10.0, 0.1], [2, 1], 
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
            [20.0, 20.0], [1, 1], 
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
            [10.0, 0.1], [2, 1], 
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
            [10.0, 0.1], [2, 1], 
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
            [10.0, 0.1], [2, 1], 
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
            [10.0, 0.1], [2, 1], 
            [False, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )
        # Single-matrix form: dim=3 expected, we pass 5 rows
        with pytest.raises(ValueError, match="total dim"):
            mpt.eval_exp_tens(dens, np.zeros((5, 1)), verbose=False)

    def test_ma_eval_empty_query(self):
        pitch = np.array([[0.0, 4.0]]).T
        dens = mpt.build_exp_tens(
            [pitch], None, [10.0], [2], 
            [False], [True], [1200.0], verbose=False,
        )
        vals = mpt.eval_exp_tens(dens, np.zeros((2, 0)), verbose=False)
        assert vals.shape == (0,)

    def test_ma_eval_dispatch_on_type(self):
        """Public eval_exp_tens dispatches on dens type."""
        # Flat (single-multiset) form -> MaetDensity
        dens_sm = mpt.build_exp_tens(
            [0.0, 4.0], None, 10.0, 2, False, True, 1200.0, verbose=False
        )
        assert isinstance(dens_sm, mpt.MaetDensity)
        # MaetDensity -> MA path
        dens_ma = mpt.build_exp_tens(
            [np.array([[0.0, 4.0]]).T], None, [10.0], [2], 
            [False], [True], [1200.0], verbose=False,
        )
        assert isinstance(dens_ma, mpt.MaetDensity)
        # Both evaluate successfully
        x = np.array([[0.0], [4.0]])
        mpt.eval_exp_tens(dens_sm, x, verbose=False)
        mpt.eval_exp_tens(dens_ma, x, verbose=False)

    # --- cosSimExpTens MA path ---------------------------------------

    def test_ma_cossim_matches_sa_abs(self):
        """MA cos-sim matches single-multiset at the Single-multiset equivalence mapping (is_rel=False)."""
        p_a = [0.0, 400.0, 700.0]
        p_b = [0.0, 300.0, 700.0]
        w_a = [1.0, 0.7, 0.5]
        w_b = [1.0, 0.6, 0.8]
        sigma, r, is_rel, is_per, period = 10.0, 2, False, True, 1200.0

        s_sm = mpt.cos_sim_exp_tens_raw(
            p_a, w_a, p_b, w_b, sigma, r, is_rel, is_per, period,
            verbose=False,
        )

        # MA form: one attribute, one event, column weight
        da = mpt.build_exp_tens(
            [np.array(p_a).reshape(3, 1)], [np.array(w_a).reshape(3, 1)],
            [sigma], [r], [is_rel], [is_per], [period], verbose=False,
        )
        db = mpt.build_exp_tens(
            [np.array(p_b).reshape(3, 1)], [np.array(w_b).reshape(3, 1)],
            [sigma], [r], [is_rel], [is_per], [period], verbose=False,
        )
        s_ma = mpt.cos_sim_exp_tens(da, db, verbose=False)

        np.testing.assert_allclose(s_ma, s_sm, rtol=1e-12, atol=1e-12)

    def test_ma_cossim_matches_sa_rel(self):
        """Single-multiset equivalence with is_rel=True (uses pairwise-diff formula periodically)."""
        p_a = [0.0, 400.0, 700.0]
        p_b = [0.0, 300.0, 700.0]
        w_a = [1.0, 0.7, 0.5]
        w_b = [1.0, 0.6, 0.8]
        for r, is_per, period in [(2, True, 1200.0),
                                   (3, True, 1200.0),
                                   (3, False, 0.0)]:
            s_sm = mpt.cos_sim_exp_tens_raw(
                p_a, w_a, p_b, w_b, 10.0, r, True, is_per, period, verbose=False
            )
            da = mpt.build_exp_tens(
                [np.array(p_a).reshape(3, 1)], [np.array(w_a).reshape(3, 1)],
                [10.0], [r], [True], [is_per], [period], verbose=False,
            )
            db = mpt.build_exp_tens(
                [np.array(p_b).reshape(3, 1)], [np.array(w_b).reshape(3, 1)],
                [10.0], [r], [True], [is_per], [period], verbose=False,
            )
            s_ma = mpt.cos_sim_exp_tens(da, db, verbose=False)
            np.testing.assert_allclose(
                s_ma, s_sm, rtol=1e-12, atol=1e-12,
                err_msg=f"r={r}, is_per={is_per}, period={period}",
            )

    def test_ma_cossim_self_is_one(self):
        """cos_sim(d, d) == 1 for a non-degenerate MA density."""
        pitch = np.array([[0.0, 12.0], [4.0, 15.0], [7.0, 19.0]])
        time  = np.array([[0.0, 1.0]])
        d = mpt.build_exp_tens(
            [pitch, time], None,
            [10.0, 0.1], [3, 1], 
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
            [10.0, 0.1], [3, 1], 
            [True, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )
        db = mpt.build_exp_tens(
            [pitchB, timeB], None,
            [10.0, 0.1], [3, 1], 
            [True, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )
        s_ab = mpt.cos_sim_exp_tens(da, db, verbose=False)
        s_ba = mpt.cos_sim_exp_tens(db, da, verbose=False)
        np.testing.assert_allclose(s_ab, s_ba, rtol=1e-12, atol=1e-12)

    def test_ma_cossim_isrel_transposition_invariance(self):
        """Shifting all pitches by a constant preserves cos_sim when
        the pitch group has is_rel=True (one event, so the shift affects
        every pitch position equally)."""
        pitch = np.array([[0.0], [400.0], [700.0]])  # K=3, N=1
        time  = np.array([[1.0]])
        pitch_shifted = pitch + 137.0

        d1 = mpt.build_exp_tens(
            [pitch, time], None,
            [10.0, 0.1], [3, 1], 
            [True, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )
        d2 = mpt.build_exp_tens(
            [pitch_shifted, time], None,
            [10.0, 0.1], [3, 1], 
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
            [10.0, 0.1], [3, 1],
            [True, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )
        da = mpt.build_exp_tens(
            [pitchA, timeA], None,
            [10.0, 0.1], [3, 1], 
            [True, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )
        db = mpt.build_exp_tens(
            [pitchB, timeB], None,
            [10.0, 0.1], [3, 1], 
            [True, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )
        s_struct = mpt.cos_sim_exp_tens(da, db, verbose=False)
        np.testing.assert_allclose(s_raw, s_struct, rtol=1e-12, atol=1e-12)

    def test_ma_cossim_raw_sa_still_works(self):
        """single-multiset raw-args call unchanged from v2.0.0 behaviour."""
        s = mpt.cos_sim_exp_tens_raw(
            [0.0, 4.0, 7.0], None, [0.0, 4.0, 7.0], None,
            10.0, 2, True, True, 1200.0, verbose=False,
        )
        np.testing.assert_allclose(s, 1.0, rtol=1e-12, atol=1e-12)

    def test_ma_cossim_raw_mismatched_types_errors(self):
        """p1 is MA but p2 is single-multiset. Under the unified cos_sim_exp_tens
        dispatcher, the first arg's type (here, list of arrays = MA)
        sets the dispatch arm; the single-multiset-shaped second operand then builds
        an single-multiset density, and comparing densities of different types
        produces a clear TypeError."""
        pitch_ma = [np.array([[0.0, 4.0]]).T]
        pitch_sm = [0.0, 4.0]
        with pytest.raises(TypeError, match="same input form"):
            mpt.cos_sim_exp_tens_raw(
                pitch_ma, None, pitch_sm, None,
                10.0, 2, False, True, 1200.0, verbose=False,
            )

    def test_ma_cossim_parameter_mismatch_errors(self):
        """Mismatched MA densities raise specific ValueErrors."""
        pitch = np.array([[0.0, 4.0, 7.0]]).T
        base_kwargs = dict(
            p_attr=[pitch], w=None,
            sigma_vec=[10.0], r_vec=[2],
            is_rel_vec=[False], is_per_vec=[True], period_vec=[1200.0],
        )
        d_ref = mpt.build_exp_tens(
            base_kwargs["p_attr"], base_kwargs["w"],
            base_kwargs["sigma_vec"], base_kwargs["r_vec"], 
            base_kwargs["is_rel_vec"], base_kwargs["is_per_vec"],
            base_kwargs["period_vec"], verbose=False,
        )
        # Different r_vec
        d_r = mpt.build_exp_tens(
            [pitch], None, [10.0], [3], 
            [False], [True], [1200.0], verbose=False,
        )
        with pytest.raises(ValueError, match="r"):
            mpt.cos_sim_exp_tens(d_ref, d_r, verbose=False)
        # Different sigma
        d_s = mpt.build_exp_tens(
            [pitch], None, [20.0], [2], 
            [False], [True], [1200.0], verbose=False,
        )
        with pytest.raises(ValueError, match="sigma"):
            mpt.cos_sim_exp_tens(d_ref, d_s, verbose=False)
        # Different is_rel
        d_rel = mpt.build_exp_tens(
            [pitch], None, [10.0], [2], 
            [True], [True], [1200.0], verbose=False,
        )
        with pytest.raises(ValueError, match="is_rel"):
            mpt.cos_sim_exp_tens(d_ref, d_rel, verbose=False)
        # Different period on periodic group
        d_p = mpt.build_exp_tens(
            [pitch], None, [10.0], [2], 
            [False], [True], [2400.0], verbose=False,
        )
        with pytest.raises(ValueError, match="period"):
            mpt.cos_sim_exp_tens(d_ref, d_p, verbose=False)

    # --- entropyExpTens MA path -------------------------------------

    def test_ma_entropy_sa_equivalence_periodic(self):
        """MA entropy matches single-multiset entropy at the Single-multiset equivalence mapping
        (single periodic group, is_rel=False)."""
        p = np.array([0.0, 4.0, 7.0])
        w = np.array([1.0, 1.0, 1.0])
        H_sm = mpt.entropy_exp_tens(
            p, w, 10.0, 1, False, True, 12.0,
            n_points_per_dim=400,
        )
        H_ma = mpt.entropy_exp_tens(
            [p.reshape(3, 1)], [w.reshape(3, 1)],
            [10.0], [1], [False], [True], [12.0],
            n_points_per_dim=400,
        )
        np.testing.assert_allclose(H_ma, H_sm, rtol=1e-10, atol=1e-10)

    def test_ma_entropy_sa_equivalence_nonperiodic(self):
        """MA entropy matches single-multiset entropy for a non-periodic group with
        explicit bounds."""
        p = np.array([0.0, 4.0, 7.0])
        w = np.array([1.0, 1.0, 1.0])
        H_sm = mpt.entropy_exp_tens(
            p, w, 10.0, 1, False, False, 0.0,
            x_min=-3.0, x_max=10.0, n_points_per_dim=400,
        )
        H_ma = mpt.entropy_exp_tens(
            [p.reshape(3, 1)], [w.reshape(3, 1)],
            [10.0], [1], [False], [False], [0.0],
            x_min=-3.0, x_max=10.0, n_points_per_dim=400,
        )
        np.testing.assert_allclose(H_ma, H_sm, rtol=1e-10, atol=1e-10)

    def test_ma_entropy_uniform_pitch_high(self):
        """Chromatic scale with wide sigma gives near-uniform pmf,
        so normalised entropy is close to 1."""
        p = np.arange(12, dtype=np.float64)
        H = mpt.entropy_exp_tens(
            [p.reshape(12, 1)], None,
            [100.0], [1], [False], [True], [12.0],
            n_points_per_dim=400,
        )
        assert H > 0.95

    def test_ma_entropy_concentrated_below_uniform(self):
        """A single pitch is more concentrated than the chromatic
        scale, so gives lower normalised entropy."""
        # sigma must be substantially smaller than the period (12) for
        # the distributions to be distinguishable: at sigma >> period
        # the periodic kernel wraps many times and both densities are
        # analytically uniform (and bin-integration gives H_norm = 1
        # for both, exposing the degenerate setup; point-evaluation
        # was passing by sampling noise).
        p_one = np.array([5.0])
        H_one = mpt.entropy_exp_tens(
            [p_one.reshape(1, 1)], None,
            [1.0], [1], [False], [True], [12.0],
            n_points_per_dim=400,
        )
        p_all = np.arange(12, dtype=np.float64)
        H_all = mpt.entropy_exp_tens(
            [p_all.reshape(12, 1)], None,
            [1.0], [1], [False], [True], [12.0],
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
            [20.0, 0.1], [2, 1], 
            [True, False], [True, False], [1200.0, 0.0],
            verbose=False,
        )
        # dim = (2-1) + 1 = 2
        assert dens.dim == 2
        H = mpt.entropy_exp_tens(
            dens, method='normalized',
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
            [20.0, 0.1], [2, 1], 
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
                [10.0], [1], [False], [False], [0.0],
                n_points_per_dim=100,
            )

    def test_ma_entropy_per_group_bounds(self):
        """Length-G x_min/x_max vectors are accepted, with periodic
        group entries ignored."""
        pitch = np.array([[0.0, 12.0], [4.0, 15.0], [7.0, 19.0]])
        time = np.array([[0.0, 1.0]])
        H_scalar = mpt.entropy_exp_tens(
            [pitch, time], None,
            [20.0, 0.1], [2, 1],
            [True, False], [True, False], [1200.0, 0.0],
            x_min=-0.5, x_max=1.5,
            n_points_per_dim=60,
        )
        # Length-G vector with NaN for the periodic group — same result.
        H_vec = mpt.entropy_exp_tens(
            [pitch, time], None,
            [20.0, 0.1], [2, 1],
            [True, False], [True, False], [1200.0, 0.0],
            x_min=[float("nan"), -0.5],
            x_max=[float("nan"),  1.5],
            n_points_per_dim=60,
        )
        np.testing.assert_allclose(H_scalar, H_vec, rtol=1e-12, atol=1e-12)

    # --- differenceEvents -------------------------------------------
    # difference_events moved onto the (p_attr, w, specs) triple (3c-iv);
    # its tests live in tests/test_difference.py. The old groups/dict
    # contract and the K=1-only restriction were removed with Commit 3c-iv.

    # --- bind_events ----------------------------------------------------
    # bind_events now emits nested specs (3c); its tests live in
    # tests/test_bind.py. The old separate-attribute / groups contract
    # was removed with Commit 3c.

    # --- weight_events --------------------------------------------------

    def test_weight_pure_gaussian_gamma_zero(self):
        """gamma = 0 limit: pure Gaussian with std = width."""
        p = [np.array([[60.0, 62.0, 64.0, 67.0, 72.0]])]
        _, w_out, _ = mpt.unpack_pre_maet(mpt.weight_events(
            p, None,
            input_attr=0, target_attr=0,
            centre=64.0, sd=3.0, shape=0.0,
            is_per=False, period=0.0,
            drop_input_attr=False,
        ))
        expected = np.exp(
            -((np.array([60.0, 62.0, 64.0, 67.0, 72.0]) - 64.0) ** 2)
            / (2 * 3.0 ** 2)
        ).reshape(1, -1)
        np.testing.assert_allclose(np.asarray(w_out[0]), expected)

    def test_weight_pure_rectangle_gamma_one(self):
        """gamma = 1 limit: pure rectangle with half-width = width * sqrt(3)."""
        p = [np.array([[60.0, 62.0, 64.0, 67.0, 72.0]])]
        _, w_out, _ = mpt.unpack_pre_maet(mpt.weight_events(
            p, None,
            input_attr=0, target_attr=0,
            centre=64.0, sd=3.0, shape=1.0,
            is_per=False, period=0.0,
            drop_input_attr=False,
        ))
        half = 3.0 * np.sqrt(3.0)
        expected = (
            np.abs(np.array([60.0, 62.0, 64.0, 67.0, 72.0]) - 64.0) <= half
        ).astype(float).reshape(1, -1)
        np.testing.assert_array_equal(np.asarray(w_out[0]), expected)

    def test_weight_intermediate_gamma_peak_one(self):
        """Peak h(0) = 1 throughout the family, for every gamma in (0, 1)."""
        p = [np.array([[5.0]])]   # single event at the centre
        for g in [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]:
            _, w_out, _ = mpt.unpack_pre_maet(mpt.weight_events(
                p, None,
                input_attr=0, target_attr=0,
                centre=5.0, sd=2.0, shape=g,
                is_per=False, period=0.0,
                drop_input_attr=False,
            ))
            assert abs(np.asarray(w_out[0])[0, 0] - 1.0) < 1e-12, (
                f"peak at gamma={g} was {np.asarray(w_out[0])[0, 0]}, "
                f"expected 1.0"
            )

    def test_weight_fixed_variance_property(self):
        """Total variance of the window is width^2 for every gamma in [0, 1]
        (the manuscript's fixed-variance parametrisation)."""
        y = np.linspace(-30.0, 30.0, 60001).reshape(1, -1)
        p = [y]
        width = 4.0
        for g in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            _, w_out, _ = mpt.unpack_pre_maet(mpt.weight_events(
                p, None,
                input_attr=0, target_attr=0,
                centre=0.0, sd=width, shape=g,
                is_per=False, period=0.0,
                drop_input_attr=False,
            ))
            h = np.asarray(w_out[0]).ravel()
            dy = y[0, 1] - y[0, 0]
            area = h.sum() * dy
            variance = (y.ravel() ** 2 * h).sum() * dy / area
            assert abs(variance - width ** 2) < 5e-3, (
                f"variance at gamma={g} was {variance}, expected {width**2}"
            )

    def test_weight_returns_pre_maet(self):
        """Output is a pre-MAET: p_attr, w_attr, and specs."""
        p = [np.array([[1.0, 2.0]]), np.array([[3.0, 4.0]])]
        out = mpt.weight_events(
            p, None,
            input_attr=0, target_attr=0,
            centre=1.5, sd=1.0, shape=0.0,
            is_per=False, period=0.0,
            drop_input_attr=False,
        )
        assert sorted(out) == ["p_attr", "specs", "w_attr"]
        out = mpt.unpack_pre_maet(out)
        p_out, w_out, s_out = out
        assert isinstance(p_out, list) and len(p_out) == 2
        assert isinstance(w_out, list) and len(w_out) == 2
        # Third element is the triple's specs: one dict per output attribute.
        assert isinstance(s_out, list) and len(s_out) == 2
        assert all(isinstance(sp, dict) for sp in s_out)

    def test_weight_non_input_attributes_pass_through(self):
        """Attributes that are neither input nor target keep their incoming
        weight."""
        p = [np.array([[1.0, 2.0, 3.0]]),
             np.array([[10.0, 20.0, 30.0]]),
             np.array([[100.0, 200.0, 300.0]])]
        w_in = [None, None, 0.5]
        _, w_out, _ = mpt.unpack_pre_maet(mpt.weight_events(
            p, w_in,
            input_attr=0, target_attr=1,
            centre=2.0, sd=1.0, shape=0.0,
            is_per=False, period=0.0,
            drop_input_attr=False,
        ))
        # Attr 2 (not input, not target) keeps its weight unchanged.
        assert w_out[2] == 0.5

    def test_weight_input_ne_target_factor_to_target_only(self):
        """When input != target, the factor lands on the target attribute's
        weights and the input attribute's own are unchanged."""
        p = [np.array([[60.0, 64.0, 67.0]]),    # pitch (target)
             np.array([[0.0, 1.0, 2.0]])]        # time (input)
        _, w_out, _ = mpt.unpack_pre_maet(mpt.weight_events(
            p, None,
            input_attr=1, target_attr=0,
            centre=1.0, sd=1.0, shape=0.0,    # Gaussian on time at t=1
            is_per=False, period=0.0,
            drop_input_attr=False,
        ))
        expected_factor = np.exp(
            -((np.array([0.0, 1.0, 2.0]) - 1.0) ** 2) / 2.0
        )
        np.testing.assert_allclose(np.asarray(w_out[0]).ravel(), expected_factor)
        # Input attribute (time) unchanged: still None passes through as None.
        assert w_out[1] is None

    def test_weight_target_K_gt_1_broadcasts(self):
        """A (1, N) factor broadcasts across the target's K_target positions."""
        # pitch attr with K=3 (target), time attr with K=1 (input).
        p = [np.array([[60.0, 64.0],
                       [62.0, 65.0],
                       [64.0, 67.0]]),
             np.array([[0.0, 1.0]])]
        # Pre-existing weight on pitch with full (3, 2) shape, all 1s.
        w_in = [np.ones((3, 2)), None]
        _, w_out, _ = mpt.unpack_pre_maet(mpt.weight_events(
            p, w_in,
            input_attr=1, target_attr=0,
            centre=0.0, sd=1.0, shape=0.0,
            is_per=False, period=0.0,
            drop_input_attr=False,
        ))
        # The (1, 2) factor broadcasts across the 3 pitch positions, giving (3, 2).
        out0 = np.asarray(w_out[0])
        assert out0.shape == (3, 2)
        factor = np.exp(-((np.array([0.0, 1.0])) ** 2) / 2.0)
        for k in range(3):
            np.testing.assert_allclose(out0[k], factor)

    def test_weight_periodic_wrap(self):
        """Periodic input attribute wraps delta = v - c to [-P/2, P/2] before
        applying the shape function. Values themselves stay raw."""
        p = [np.array([[10.0, 11.0, 0.0, 1.0, 2.0]])]   # raw
        _, w_out, _ = mpt.unpack_pre_maet(mpt.weight_events(
            p, None,
            input_attr=0, target_attr=0,
            centre=0.0, sd=2.0, shape=0.0,
            is_per=True, period=12.0,
            drop_input_attr=False,
        ))
        deltas = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        expected = np.exp(-(deltas ** 2) / 8.0).reshape(1, -1)
        np.testing.assert_allclose(np.asarray(w_out[0]), expected)

    def test_weight_multiplies_into_existing_weight(self):
        """Window factor multiplies into the target's incoming weight."""
        p = [np.array([[1.0, 2.0, 3.0]])]
        w_in = 0.5
        _, w_out, _ = mpt.unpack_pre_maet(mpt.weight_events(
            p, w_in,
            input_attr=0, target_attr=0,
            centre=2.0, sd=1.0, shape=0.0,
            is_per=False, period=0.0,
            drop_input_attr=False,
        ))
        h = np.exp(-(np.array([1.0, 2.0, 3.0]) - 2.0) ** 2 / 2.0)
        np.testing.assert_allclose(np.asarray(w_out[0]).ravel(), 0.5 * h)

    def test_weight_sequential_composition_for_two_windows(self):
        """Multi-axis windowing via two sequential calls to the same target
        (the canonical replacement for old multi-input behaviour)."""
        # pitch (target), two scaffolding attrs (input1 = time, input2 = beat).
        p = [np.array([[60.0, 64.0, 67.0]]),
             np.array([[0.0, 1.0, 2.0]]),
             np.array([[0.0, 0.5, 1.0]])]
        # First call: time-window into pitch; keep input for the next call
        # (which expects all three attributes still present).
        p1, w1, g1 = mpt.unpack_pre_maet(mpt.weight_events(
            p, None,
            input_attr=1, target_attr=0,
            centre=1.0, sd=1.0, shape=0.0,
            is_per=False, period=0.0,
            drop_input_attr=False,
        ))
        # Second call: beat-window into pitch.
        _, w2, _ = mpt.unpack_pre_maet(mpt.weight_events(
            p1, w1,
            input_attr=2, target_attr=0,
            centre=0.5, sd=0.5, shape=0.0,
            is_per=False, period=0.0,
            drop_input_attr=False,
        ))
        # Result: the pitch attribute's weights carry the product of both factors.
        h_time = np.exp(-((np.array([0.0, 1.0, 2.0]) - 1.0) ** 2) / 2.0)
        h_beat = np.exp(-((np.array([0.0, 0.5, 1.0]) - 0.5) ** 2) / 0.5)
        np.testing.assert_allclose(
            np.asarray(w2[0]).ravel(), h_time * h_beat,
        )

    def test_weight_drop_input_attr_drops_attribute(self):
        """drop_input_attr=True (with input != target) drops the input attribute
        from the output structures."""
        p = [np.array([[60.0, 64.0, 67.0]]),
             np.array([[0.0, 1.0, 2.0]])]
        p_out, w_out, g_out = mpt.unpack_pre_maet(mpt.weight_events(
            p, None,
            input_attr=1, target_attr=0,
            centre=1.0, sd=1.0, shape=0.0,
            is_per=False, period=0.0,
            drop_input_attr=True,
        ))
        assert len(p_out) == 1
        assert len(w_out) == 1
        # The remaining attribute is the original pitch (attr 0).
        np.testing.assert_array_equal(p_out[0], p[0])

    def test_weight_drop_input_attr_index_when_input_after_target(self):
        """drop_input_attr with input_attr > target_attr: the input attribute's
        value, weight, and spec are dropped, and the target keeps its output
        index (nothing before it shifts)."""
        # 3 attrs; input = attr 1, target = attr 0 (input after target).
        p = [np.array([[1.0, 2.0]]),
             np.array([[3.0, 4.0]]),
             np.array([[5.0, 6.0]])]
        p_out, w_out, s_out = mpt.unpack_pre_maet(mpt.weight_events(
            p, None,
            input_attr=1, target_attr=0,
            centre=3.5, sd=1.0, shape=0.0,
            is_per=False, period=0.0,
            drop_input_attr=True,
        ))
        # Input attr 1 removed: originals 0 and 2 remain at output indices 0, 1.
        assert len(p_out) == 2 and len(w_out) == 2 and len(s_out) == 2
        np.testing.assert_array_equal(p_out[0], p[0])
        np.testing.assert_array_equal(p_out[1], p[2])
        # Factor (from input attr 1's values) lands on the target at index 0.
        factor = np.exp(-((np.array([3.0, 4.0]) - 3.5) ** 2) / 2.0)
        np.testing.assert_allclose(np.asarray(w_out[0]).ravel(), factor)

    def test_weight_drop_input_attr_index_when_input_before_target(self):
        """drop_input_attr with input_attr < target_attr: the input attribute is
        dropped and the target shifts down one output index, carrying the
        windowed factor with it."""
        # 3 attrs; input = attr 0, target = attr 2 (input before target).
        p = [np.array([[1.0, 2.0]]),
             np.array([[3.0, 4.0]]),
             np.array([[5.0, 6.0]])]
        p_out, w_out, s_out = mpt.unpack_pre_maet(mpt.weight_events(
            p, None,
            input_attr=0, target_attr=2,
            centre=1.5, sd=1.0, shape=0.0,
            is_per=False, period=0.0,
            drop_input_attr=True,
        ))
        # Input attr 0 removed: originals 1 and 2 remain at output indices 0, 1.
        assert len(p_out) == 2 and len(w_out) == 2 and len(s_out) == 2
        np.testing.assert_array_equal(p_out[0], p[1])
        np.testing.assert_array_equal(p_out[1], p[2])
        # Target (original attr 2) is now at output index 1 and carries the
        # factor computed from input attr 0's values.
        factor = np.exp(-((np.array([1.0, 2.0]) - 1.5) ** 2) / 2.0)
        np.testing.assert_allclose(np.asarray(w_out[1]).ravel(), factor)

    def test_weight_drop_input_attr_with_input_eq_target_errors(self):
        """drop_input_attr=True with input_attr == target_attr is incoherent
        and raises ValueError."""
        p = [np.array([[1.0, 2.0]])]
        with pytest.raises(ValueError, match="drop_input_attr"):
            mpt.weight_events(
                p, None,
                input_attr=0, target_attr=0,
                centre=1.0, sd=1.0, shape=0.0,
                is_per=False, period=0.0,
                drop_input_attr=True,
            )

    def test_weight_drop_input_attr_required_no_default(self):
        """drop_input_attr must be specified by the caller; no default."""
        p = [np.array([[1.0, 2.0]])]
        with pytest.raises(TypeError, match="drop_input_attr"):
            mpt.weight_events(
                p, None,
                input_attr=0, target_attr=0,
                centre=1.0, sd=1.0, shape=0.0,
                is_per=False, period=0.0,
            )

    def test_weight_k_input_greater_than_one_errors(self):
        """An input attribute with K > 1 is rejected (the factor must be a
        single value per event)."""
        p = [np.array([[60.0, 62.0],
                       [64.0, 65.0]])]   # K = 2
        with pytest.raises(ValueError, match="K = 1"):
            mpt.weight_events(
                p, None,
                input_attr=0, target_attr=0,
                centre=62.0, sd=2.0, shape=0.0,
                is_per=False, period=0.0,
                drop_input_attr=False,
            )

    def test_weight_zero_width_errors(self):
        """width = 0 is rejected (degenerate)."""
        p = [np.array([[1.0, 2.0]])]
        with pytest.raises(ValueError, match="sd"):
            mpt.weight_events(
                p, None,
                input_attr=0, target_attr=0,
                centre=1.0, sd=0.0, shape=0.5,
                is_per=False, period=0.0,
                drop_input_attr=False,
            )

    def test_weight_negative_width_errors(self):
        p = [np.array([[1.0, 2.0]])]
        with pytest.raises(ValueError, match="sd"):
            mpt.weight_events(
                p, None,
                input_attr=0, target_attr=0,
                centre=1.0, sd=-1.0, shape=0.5,
                is_per=False, period=0.0,
                drop_input_attr=False,
            )

    def test_weight_shape_out_of_range_errors(self):
        """shape (gamma) outside [0, 1] is rejected."""
        p = [np.array([[1.0, 2.0]])]
        with pytest.raises(ValueError, match="shape"):
            mpt.weight_events(
                p, None,
                input_attr=0, target_attr=0,
                centre=1.0, sd=1.0, shape=1.5,
                is_per=False, period=0.0,
                drop_input_attr=False,
            )
        with pytest.raises(ValueError, match="shape"):
            mpt.weight_events(
                p, None,
                input_attr=0, target_attr=0,
                centre=1.0, sd=1.0, shape=-0.1,
                is_per=False, period=0.0,
                drop_input_attr=False,
            )

    def test_weight_input_attr_out_of_range_errors(self):
        p = [np.array([[1.0, 2.0]])]
        with pytest.raises(ValueError, match="input_attr"):
            mpt.weight_events(
                p, None,
                input_attr=2, target_attr=0,
                centre=1.0, sd=1.0, shape=0.0,
                is_per=False, period=0.0,
                drop_input_attr=False,
            )

    def test_weight_target_attr_out_of_range_errors(self):
        p = [np.array([[1.0, 2.0]])]
        with pytest.raises(ValueError, match="target_attr"):
            mpt.weight_events(
                p, None,
                input_attr=0, target_attr=3,
                centre=1.0, sd=1.0, shape=0.0,
                is_per=False, period=0.0,
                drop_input_attr=False,
            )

    def test_weight_periodic_requires_positive_period(self):
        p = [np.array([[1.0, 2.0]])]
        with pytest.raises(ValueError, match="period"):
            mpt.weight_events(
                p, None,
                input_attr=0, target_attr=0,
                centre=1.0, sd=1.0, shape=0.0,
                is_per=True, period=0.0,
                drop_input_attr=False,
            )

    def test_weight_t_w_centre_shift_commutation(self):
        """T then W with centre c equals W with centre c-mu then T, on the
        same input attribute (centre-shift composition rule)."""
        p = [np.array([[60.0, 62.0, 64.0]])]
        mu = 5.0
        c = 64.0
        width = 3.0
        gamma = 0.3   # intermediate (non-trivial convolution)
        p_t, _, _ = mpt.unpack_pre_maet(mpt.translate_attributes(p, None, [mu]))
        _, w_after_t, _ = mpt.unpack_pre_maet(mpt.weight_events(
            p_t, None,
            input_attr=0, target_attr=0,
            centre=c, sd=width, shape=gamma,
            is_per=False, period=0.0,
            drop_input_attr=False,
        ))
        _, w_first, _ = mpt.unpack_pre_maet(mpt.weight_events(
            p, None,
            input_attr=0, target_attr=0,
            centre=c - mu, sd=width, shape=gamma,
            is_per=False, period=0.0,
            drop_input_attr=False,
        ))
        np.testing.assert_allclose(
            np.asarray(w_first[0]), np.asarray(w_after_t[0]),
        )

    # --- translate_attributes -------------------------------------------
    # translate_attributes moved onto the (p_attr, w, specs) triple
    # (3c-iv-d): per-attribute per-position offsets, is_rel read from specs,
    # groups/dict offset forms removed. Its tests live in
    # tests/test_translate.py.

    # --- Memory chunking on the per-attribute IP matrix ----------------

    def test_chunking_parity_r1_abs(self):
        """`_ma_per_attr_inner_matrix` r = 1 abs branch: chunked path
        (forced via a tiny kernel_chunk_bytes budget) matches the
        unchunked fast path to floating-point precision."""
        # Sized so that the fast path comfortably fits, and the chunked
        # path takes multiple chunks: N = 12, K = 6.
        rng = np.random.default_rng(1234)
        N, K = 12, 6
        p = rng.uniform(0, 1200, size=(K, N))
        w = rng.uniform(0.5, 1.5, size=(K, N))
        # Run with default (auto) budget — fast path.
        mpt.reset_defaults()
        from mpt._tensor._mobius_inner import (_ma_per_attr_inner_matrix)
        ip_default = _ma_per_attr_inner_matrix(
            p, w, p, w, sigma=80.0, r=1, is_rel=False,
            is_per=False, period=0.0,
        )
        # Force chunking with a small budget.
        mpt.set_default(kernel_chunk_bytes=4096)
        try:
            ip_chunked = _ma_per_attr_inner_matrix(
                p, w, p, w, sigma=80.0, r=1, is_rel=False,
                is_per=False, period=0.0,
            )
        finally:
            mpt.reset_defaults()
        np.testing.assert_allclose(ip_default, ip_chunked, rtol=1e-12, atol=1e-12)

    def test_chunking_parity_r2_abs_safe(self):
        """`_ma_per_attr_inner_matrix` r >= 2 abs safe-x-safe submatrix:
        chunked path matches the unchunked path."""
        rng = np.random.default_rng(5678)
        N, K = 8, 5
        # K = 5, r = 2 -> K - r = 3 >= 2, so all events are safe.
        p = rng.uniform(0, 1200, size=(K, N))
        w = rng.uniform(0.5, 1.5, size=(K, N))
        mpt.reset_defaults()
        from mpt._tensor._mobius_inner import (_ma_per_attr_inner_matrix)
        ip_default = _ma_per_attr_inner_matrix(
            p, w, p, w, sigma=80.0, r=2, is_rel=False,
            is_per=False, period=0.0,
        )
        mpt.set_default(kernel_chunk_bytes=4096)
        try:
            ip_chunked = _ma_per_attr_inner_matrix(
                p, w, p, w, sigma=80.0, r=2, is_rel=False,
                is_per=False, period=0.0,
            )
        finally:
            mpt.reset_defaults()
        np.testing.assert_allclose(ip_default, ip_chunked, rtol=1e-10, atol=1e-12)

    def test_chunking_parity_rel_per(self):
        """`_rel_inner_batched_per`: outer N_x chunking matches
        the unchunked single-shot pair-tensor allocation."""
        rng = np.random.default_rng(9012)
        N, K = 6, 4
        p = rng.uniform(0, 12, size=(K, N))
        w = rng.uniform(0.5, 1.5, size=(K, N))
        mpt.reset_defaults()
        from mpt._tensor._mobius_inner import (_ma_per_attr_inner_matrix)
        ip_default = _ma_per_attr_inner_matrix(
            p, w, p, w, sigma=0.3, r=2, is_rel=True,
            is_per=True, period=12.0,
        )
        mpt.set_default(kernel_chunk_bytes=8192)
        try:
            ip_chunked = _ma_per_attr_inner_matrix(
                p, w, p, w, sigma=0.3, r=2, is_rel=True,
                is_per=True, period=12.0,
            )
        finally:
            mpt.reset_defaults()
        np.testing.assert_allclose(ip_default, ip_chunked, rtol=1e-8, atol=1e-10)

    def test_chunking_large_NK_does_not_oom(self):
        """N = 272, K = 48 (BWV 347 + 12 partials × 4 voices): the
        configuration that motivated chunking. Runs to completion at
        default budget; verifies the OOM hazard is fixed."""
        rng = np.random.default_rng(2026)
        N, K = 272, 48
        p = rng.uniform(0, 12000, size=(K, N))
        w = rng.uniform(0.5, 1.5, size=(K, N))
        from mpt._tensor._mobius_inner import (_ma_per_attr_inner_matrix)
        ip = _ma_per_attr_inner_matrix(
            p, w, p, w, sigma=10.0, r=1, is_rel=False,
            is_per=False, period=0.0,
        )
        # Sanity: result is (N, N), finite, with positive diagonal.
        assert ip.shape == (N, N)
        assert np.all(np.isfinite(ip))
        assert np.all(np.diag(ip) > 0)
