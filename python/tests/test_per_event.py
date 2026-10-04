"""The per-event form of an attribute's values and weights.

An attribute may be given as a list with one entry per event, each entry
the event's values (a scalar, a sequence, or empty). It is read exactly as
the NaN-padded K x N matrix it stands for. Per-event weights have one
entry per event, a scalar weighting all the event's values or a sequence
with one weight per value. A K x N matrix is given as a NumPy array.

Mirror of MATLAB tests/test_per_event.m.
"""

import numpy as np
import pytest

import mpt
from mpt import (build_maet, eval_maet, entropy_maet, flat_specs,
                 pack_pre_maet, sim_maet, transform_attributes)

NAN = np.nan
PE = [[[60, 64, 67], 62, 64, 65], [0, 1, 1.5, 2]]
MX = [np.array([[60, 62, 64, 65], [64, NAN, NAN, NAN],
                [67, NAN, NAN, NAN]]),
      np.array([[0, 1, 1.5, 2]])]
GEOM = dict(sigma=[0.5, 0.25], r=[1, 1], rel=[False, False],
            per=[True, False], period=[12, 0])


def _same(a, b):
    np.testing.assert_array_equal(np.asarray(a, float), np.asarray(b, float))


def _build(p, w=None):
    g = GEOM
    return build_maet(p, w, g["sigma"], g["r"], g["rel"], g["per"],
                      g["period"], verbose=False)


def test_pack_values_match_matrix():
    pm_e = pack_pre_maet(PE)
    pm_m = pack_pre_maet(MX)
    for a in range(2):
        _same(pm_e["p_attr"][a], pm_m["p_attr"][a])


def test_empty_and_none_entries_hold_no_value():
    pm = pack_pre_maet([[[60, 64], [], None, 62]])
    _same(pm["p_attr"][0], [[60, NAN, NAN, 62], [64, NAN, NAN, NAN]])


def test_list_of_lists_entry_is_refused():
    with pytest.raises(ValueError, match="NumPy array"):
        pack_pre_maet([[[[60, 64], [62, 65]], 1]])


def test_per_event_weights_scalar_and_vector():
    w = [[[1, 0.5, 0.25], 1, 1, 1], None]
    pm = pack_pre_maet(PE, w)
    _same(pm["w_attr"][0], [[1, 1, 1, 1], [0.5, 0, 0, 0], [0.25, 0, 0, 0]])
    pm = pack_pre_maet(PE, [[2, 1, 1, 1], None])
    _same(pm["w_attr"][0], [[2, 1, 1, 1], [2, 0, 0, 0], [2, 0, 0, 0]])


def test_per_value_flat_weights_keep_their_reading():
    # Length K (3) differs from N (4): one weight per value position.
    pm = pack_pre_maet(PE, [[1, 0.5, 0.25], None])
    _same(np.asarray(pm["w_attr"][0]).ravel(), [1, 0.5, 0.25])


def test_weight_count_mismatch_is_refused():
    with pytest.raises(ValueError):
        pack_pre_maet(PE, [[[1, 0.5], 1, 1, 1], None])


def test_build_eval_and_entropy_match_matrix():
    d_e, d_m = _build(PE), _build(MX)
    X = [np.array([[60.0, 61.0, 64.5]]), np.array([[0.0, 1.0, 1.5]])]
    _same(eval_maet(d_e, X), eval_maet(d_m, X))
    assert entropy_maet(d_e, method="renyi2") == pytest.approx(
        entropy_maet(d_m, method="renyi2"), abs=1e-12)


def test_raw_eval_maet_matches_matrix():
    g = GEOM
    tail = (g["sigma"], g["r"], g["rel"], g["per"], g["period"])
    X = [np.array([[61.0]]), np.array([[1.0]])]
    assert eval_maet(PE, None, *tail, X, verbose=False) == pytest.approx(
        eval_maet(MX, None, *tail, X, verbose=False), abs=1e-12)


def test_raw_sim_maet_matches_matrix():
    q_e = [[[60, 63], 62, [64, 67], 65], [0, 1, 1.5, 2]]
    q_m = [np.array([[60, 62, 64, 65], [63, NAN, 67, NAN]]),
           np.array([[0, 1, 1.5, 2]])]
    g = GEOM
    tail = (g["sigma"], g["r"], g["rel"], g["per"], g["period"])
    s_e = sim_maet(PE, None, q_e, None, *tail)
    s_m = sim_maet(MX, None, q_m, None, *tail)
    assert s_e == pytest.approx(s_m, abs=1e-12)


def test_transform_passes_absent_values_through():
    pm = transform_attributes([[[440, 880], 220], [1, 2]], None,
                              [("hz", "midi"), "log"])
    _same(pm["p_attr"][0], [[69, 57], [81, NAN]])
    np.testing.assert_allclose(pm["p_attr"][1], [[0, np.log(2)]])


def test_flat_list_is_one_value_per_event():
    pm = pack_pre_maet([[60, 62, 64]])
    _same(pm["p_attr"][0], [[60, 62, 64]])
