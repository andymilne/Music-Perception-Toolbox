"""Tests for difference_events on the (p_attr, w, specs) carrier (3c-iv-b).

difference_events applies the k-th finite difference along the event axis,
slot-wise. It is well-defined exactly when slots have stable identity --- an
ordered attribute ([sym]=0) or a singleton (K=1) --- so a symmetric multiset
(K>1) raises, and the rule extends per level for a nested attribute. The spec
passes through unchanged (values change, structure does not); NaN propagates
(absent slot). With order = L the differenced-then-bound and bound-then-
differenced routes coincide (B o D == D o B).
"""

import numpy as np
import pytest

import mpt
from mpt import (build_exp_tens, eval_exp_tens, difference_events,
                 bind_events, flat_specs)


# --- Carrier basics --------------------------------------------------

def test_returns_three_tuple_with_specs():
    pd, wd, sd = difference_events([np.array([[0.0, 2.0, 5.0, 9.0]])], None, 1)
    assert isinstance(sd, list) and len(sd) == 1
    assert "tags" not in sd[0]                      # flat
    np.testing.assert_allclose(pd[0], [[2.0, 3.0, 4.0]])


def test_specs_synthesised_when_none():
    _, _, sd = difference_events([np.array([[0.0, 2.0, 5.0]])], None, 1)
    assert sd == [{"r": 1, "rel": False, "sym": True}]


def test_spec_passes_through_unchanged():
    s_in = flat_specs([np.zeros((1, 4))], r=2, rel=True, sym=False)
    _, _, s_out = difference_events([np.array([[0.0, 2.0, 5.0, 9.0]])], None,
                                    1, specs=s_in)
    assert s_out == s_in


def test_order_0_identity():
    M = np.array([[60.0, 62.0, 64.0]])
    pd, _, _ = difference_events([M], None, 0)
    np.testing.assert_allclose(pd[0], M)


def test_order_2_and_alignment():
    pd, _, _ = difference_events(
        [np.array([[0.0, 1.0, 4.0, 9.0, 16.0]]),
         np.array([[10.0, 20.0, 30.0, 40.0, 50.0]])],
        None, [2, 0])
    # max order 2 -> N' = 3; attr1 (order 0) drops 2 leading events
    assert pd[0].shape == (1, 3) and pd[1].shape == (1, 3)
    np.testing.assert_allclose(pd[0], [[2.0, 2.0, 2.0]])
    np.testing.assert_allclose(pd[1], [[30.0, 40.0, 50.0]])


def test_scalar_order_broadcasts():
    pd, _, _ = difference_events(
        [np.array([[0.0, 2.0, 5.0]]), np.array([[1.0, 4.0, 9.0]])], None, 1)
    assert pd[0].shape == (1, 2) and pd[1].shape == (1, 2)


def test_weight_rolling_product():
    _, wd, _ = difference_events(
        [np.array([[0.0, 1.0, 2.0, 3.0]])],
        [np.array([[1.0, 2.0, 3.0, 4.0]])], 1)
    # rolling product width 2: [1*2, 2*3, 3*4]
    np.testing.assert_allclose(np.asarray(wd[0]), [[2.0, 6.0, 12.0]])


# --- The K generalisation: ordered any-K, symmetric rejected ---------

def test_ordered_multislot_differences_slotwise():
    """K>1 ordered attribute differences slot-wise (the lifted K=1 rule)."""
    M = np.array([[0.0, 2.0, 5.0], [10.0, 13.0, 17.0]])   # K=2, N=3
    pd, _, _ = difference_events([M], None, 1, specs=flat_specs([M], sym=False))
    np.testing.assert_allclose(pd[0], [[2.0, 3.0], [3.0, 4.0]])


def test_symmetric_multislot_rejected():
    M = np.array([[0.0, 2.0, 5.0], [10.0, 13.0, 17.0]])
    with pytest.raises(ValueError):
        difference_events([M], None, 1, specs=flat_specs([M], sym=True))


def test_symmetric_multislot_order_zero_ok():
    """Order 0 never triggers the guard (no differencing happens)."""
    M = np.array([[0.0, 2.0, 5.0], [10.0, 13.0, 17.0]])
    pd, _, _ = difference_events([M], None, 0, specs=flat_specs([M], sym=True))
    np.testing.assert_allclose(pd[0], M)


def test_singleton_differences_regardless_of_sym():
    """K=1 is always differenceable (singleton has trivial slot identity)."""
    M = np.array([[0.0, 2.0, 5.0]])
    pd, _, _ = difference_events([M], None, 1, specs=flat_specs([M], sym=True))
    np.testing.assert_allclose(pd[0], [[2.0, 3.0]])


def test_nan_propagates_as_absent_slot():
    M = np.array([[0.0, 2.0, 5.0], [10.0, np.nan, 17.0]])
    pd, _, _ = difference_events([M], None, 1, specs=flat_specs([M], sym=False))
    out = pd[0]
    np.testing.assert_allclose(out[0], [2.0, 3.0])
    assert np.all(np.isnan(out[1]))     # both differences touching NaN are NaN


# --- Nested-D: difference a bound attribute --------------------------

def test_nested_difference_slotwise_spec_passthrough():
    raw = np.array([[0.0, 2.0, 5.0, 9.0, 14.0]])   # K=1, N=5
    pb, wb, specs = bind_events([raw], None, 2)  # N'=4
    pnd, _, snd = difference_events(pb, wb, 1, specs=specs)
    assert pnd[0].shape == (2, 3)                  # (L*K, N'-1)
    assert list(snd[0]["tags"]) == [0, 1]          # spec unchanged
    assert snd[0]["r"] == specs[0]["r"]


def test_nested_symmetric_outer_rejected():
    """A bag outer level (sym_outer=1) cannot be differenced."""
    raw = np.array([[0.0, 2.0, 5.0, 9.0, 14.0]])
    pb, wb, specs = bind_events([raw], None, 2, sym_outer=True)
    with pytest.raises(ValueError):
        difference_events(pb, wb, 1, specs=specs)


# --- Commutation B o D == D o B --------------------------------------

def test_bind_difference_commute_values_and_specs():
    P = np.array([[0.0, 3.0, 7.0, 12.0, 18.0]])    # K=1, N=5
    L = 2
    pD, wD, sD = difference_events([P], None, 1)
    pDB, wDB, sDB = bind_events(pD, wD, L, specs=sD)
    pB, wB, sB = bind_events([P], None, L)
    pBD, wBD, sBD = difference_events(pB, wB, 1, specs=sB)
    np.testing.assert_allclose(pDB[0], pBD[0], equal_nan=True)
    for key in ("tags", "r", "sym", "rel"):
        assert list(np.ravel(sDB[0][key])) == list(np.ravel(sBD[0][key]))


def test_bind_difference_commute_density_identical():
    """The two routes build eval-identical densities."""
    P = np.array([[0.0, 3.0, 7.0, 12.0, 18.0]])
    L = 2
    pD, wD, sD = difference_events([P], None, 1)
    pDB, wDB, sDB = bind_events(pD, wD, L, specs=sD)
    pB, wB, sB = bind_events([P], None, L)
    pBD, wBD, sBD = difference_events(pB, wB, 1, specs=sB)
    d1 = build_exp_tens(pDB, wDB, specs=sDB, sigma=[30.0], is_per=[False],
                        period=[0.0], verbose=False)
    d2 = build_exp_tens(pBD, wBD, specs=sBD, sigma=[30.0], is_per=[False],
                        period=[0.0], verbose=False)
    rng = np.random.default_rng(0)
    xs = rng.uniform(-5, 5, size=(2, 6))
    v1 = np.ravel(eval_exp_tens(d1, xs, verbose=False))
    v2 = np.ravel(eval_exp_tens(d2, xs, verbose=False))
    np.testing.assert_allclose(v1, v2, atol=1e-12)


# --- Circular + errors + entropy parity ------------------------------

def test_circular_keeps_n():
    pd, _, _ = difference_events([np.array([[0.0, 2.0, 5.0, 9.0]])], None, 1,
                                 circular=True)
    assert pd[0].shape == (1, 4)
    np.testing.assert_allclose(pd[0], [[0.0 - 9.0, 2.0, 3.0, 4.0]])


def test_too_high_order_errors():
    with pytest.raises(ValueError):
        difference_events([np.array([[1.0, 2.0, 3.0]])], None, 5)


def test_circular_order_geq_n_errors():
    with pytest.raises(ValueError):
        difference_events([np.array([[1.0, 2.0, 3.0]])], None, 3, circular=True)


def test_wrong_length_orders_errors():
    with pytest.raises(ValueError):
        difference_events([np.array([[1.0, 2.0, 3.0]]),
                           np.array([[4.0, 5.0, 6.0]])], None, [1, 1, 1])


def test_empty_attribute_errors():
    with pytest.raises(ValueError):
        difference_events([np.zeros((0, 3))], None, 1)


def test_n_tuple_entropy_circular_parity():
    """Whole-tone scale: every cyclic 2-tuple of steps is (2,2), H = 0."""
    H, _ = mpt.n_tuple_entropy([0, 2, 4, 6, 8, 10], 12, 2, method="shannon")
    assert abs(H) < 1e-12
