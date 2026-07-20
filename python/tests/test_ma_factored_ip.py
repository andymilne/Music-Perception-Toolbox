"""Factored-cull MA inner product (``method='factored'``).

The factored path computes the multi-attribute cosine through the
per-attribute / per-event-pair inner-product factorisation, without
materialising the joint tuple set. It equals the joint (``'bulger'``)
result exactly at the accuracy floor; at a finite truncation the two
differ only in the cull region.
"""
import numpy as np
import pytest

import mpt
from mpt import build_exp_tens, cos_sim_exp_tens


def _b(p, s, r, rel, per, pd, **k):
    return build_exp_tens(p, None, s, r, rel, per, pd, verbose=False, **k)


@pytest.fixture(autouse=True)
def _floor():
    old = mpt.get_default("truncation_sigmas")
    mpt.set_default(truncation_sigmas=float("inf"))
    yield
    mpt.set_default(truncation_sigmas=old)


def _match(dx, dy, tol=1e-9):
    b = cos_sim_exp_tens(dx, dy, method="bulger", verbose=False)
    f = cos_sim_exp_tens(dx, dy, method="factored", verbose=False)
    assert abs(b - f) < tol, (b, f)


def test_factored_abs_two_attr():
    dx = _b([np.array([[0.], [4.], [7.]]), np.array([[0.], [12.], [24.]])],
            [30., 30.], [2, 2], [False, False], [False, False], [0, 0])
    dy = _b([np.array([[0.], [3.], [7.]]), np.array([[0.], [12.], [19.]])],
            [30., 30.], [2, 2], [False, False], [False, False], [0, 0])
    _match(dx, dy)


def test_factored_rel_nonper_two_attr():
    dx = _b([np.array([[0.], [4.], [7.]]), np.array([[0.], [5.], [9.]])],
            [40., 40.], [2, 2], [True, True], [False, False], [0, 0])
    dy = _b([np.array([[0.], [3.], [7.]]), np.array([[0.], [4.], [9.]])],
            [40., 40.], [2, 2], [True, True], [False, False], [0, 0])
    _match(dx, dy)


def test_factored_three_attr_multi_event():
    px = [np.array([[0., 2.], [4., 5.], [7., 9.]]),
          np.array([[0., 1.], [12., 13.], [24., 25.]]),
          np.array([[0., 0.], [3., 4.], [7., 8.]])]
    py = [np.array([[0., 1.], [3., 6.], [7., 10.]]),
          np.array([[0., 2.], [11., 12.], [23., 26.]]),
          np.array([[0., 1.], [2., 5.], [7., 9.]])]
    dx = _b(px, [30., 30., 30.], [2, 2, 2],
            [False, True, True], [False, False, False], [0, 0, 0])
    dy = _b(py, [30., 30., 30.], [2, 2, 2],
            [False, True, True], [False, False, False], [0, 0, 0])
    _match(dx, dy)


def test_factored_ragged_cardinality():
    px = [np.array([[0.], [4.], [7.], [np.nan]]),
          np.array([[0.], [12.], [np.nan], [np.nan]])]
    py = [np.array([[0.], [3.], [7.], [11.]]),
          np.array([[0.], [12.], [19.], [np.nan]])]
    dx = _b(px, [30., 30.], [2, 2], [False, False], [False, False], [0, 0])
    dy = _b(py, [30., 30.], [2, 2], [False, False], [False, False], [0, 0])
    _match(dx, dy)


def test_factored_nested_dense_factor():
    spec = {"tags": np.array([0, 0, 0, 1, 1, 1]), "r": [2, 2],
            "sym": [True, False], "rel": "innermost"}
    dx = _b([np.array([[0.], [4.], [7.], [12.], [16.], [19.]])],
            [40.], [6], [False], [False], [0], nested=[spec])
    dy = _b([np.array([[0.], [3.], [7.], [12.], [15.], [19.]])],
            [40.], [6], [False], [False], [0], nested=[spec])
    _match(dx, dy)


def test_factored_self_ip_is_one():
    dx = _b([np.array([[0.], [4.], [7.]]), np.array([[0.], [12.], [24.]])],
            [30., 30.], [2, 2], [False, False], [False, False], [0, 0])
    assert abs(cos_sim_exp_tens(dx, dx, method="factored",
                                verbose=False) - 1.0) < 1e-12


def test_factored_rejects_rel_periodic():
    dx = _b([np.array([[0.], [4.], [7.]])], [40.], [2], [True], [True], [1200.])
    dy = _b([np.array([[0.], [3.], [7.]])], [40.], [2], [True], [True], [1200.])
    with pytest.raises(ValueError, match="relative-and-periodic"):
        cos_sim_exp_tens(dx, dy, method="factored", verbose=False)
