"""Keeping a selection of a pre-MAET's attributes and events.

Mirror of MATLAB tests/test_select_pre_maet.m.
"""

import numpy as np
import pytest

from mpt import (build_maet, flat_specs, pre_maet, select_pre_maet,
                 sim_maet, unpack_pre_maet)


def _pm():
    p = [np.array([[60.0, 64.0, 67.0], [72.0, 76.0, 79.0]]),
         np.array([[0.0, 1.0, 2.0]])]
    w = [np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]]),
         np.ones((1, 3))]
    return pre_maet(p, w, flat_specs(p, r=[2, 1], exch=[False, True],
                                     name=["pitch", "onset"]))


def test_attributes_by_name():
    p, w, specs = unpack_pre_maet(select_pre_maet(_pm(), attributes=["onset"]))
    assert [s["name"] for s in specs] == ["onset"]
    assert p[0].shape == (1, 3)


def test_attributes_by_index_and_order():
    _, _, specs = unpack_pre_maet(select_pre_maet(_pm(), attributes=[1, 0]))
    assert [s["name"] for s in specs] == ["onset", "pitch"]


def test_attributes_by_mask():
    _, _, specs = unpack_pre_maet(
        select_pre_maet(_pm(), attributes=[True, False]))
    assert [s["name"] for s in specs] == ["pitch"]


def test_events_by_index():
    p, w, _ = unpack_pre_maet(select_pre_maet(_pm(), events=[0, 2]))
    assert p[0].shape == (2, 2)
    np.testing.assert_array_equal(p[1], [[0.0, 2.0]])
    np.testing.assert_array_equal(w[0][0], [1.0, 3.0])


def test_events_by_mask():
    p, _, _ = unpack_pre_maet(
        select_pre_maet(_pm(), events=[False, True, False]))
    np.testing.assert_array_equal(p[1], [[1.0]])


def test_both_at_once():
    p, _, specs = unpack_pre_maet(
        select_pre_maet(_pm(), attributes=["pitch"], events=[2]))
    assert [s["name"] for s in specs] == ["pitch"]
    np.testing.assert_array_equal(p[0], [[67.0], [79.0]])


def test_an_attribute_keeps_its_tuple_size_and_flags():
    """A selection cannot change what an attribute means."""
    _, _, specs = unpack_pre_maet(select_pre_maet(_pm(), events=[0, 1]))
    assert specs[0]["r"] == 2
    assert specs[0]["exch"] is False


def test_a_multi_coordinate_attribute_moves_whole():
    """A level's simplex coordinates are one attribute, not several, so
    no selection over attributes can take part of one."""
    coords = np.array([[0.5, -0.5], [0.2887, 0.2887], [0.2041, 0.2041]])
    p = [np.array([[60.0, 64.0]]), coords]
    pm = pre_maet(p, None, flat_specs(p, r=[1, 3], exch=[True, False],
                                      name=["pitch", "voice"]))
    out, _, specs = unpack_pre_maet(select_pre_maet(pm, attributes=["voice"]))
    assert out[0].shape == (3, 2)
    assert specs[0]["r"] == 3


def test_keeping_no_attribute_is_refused():
    with pytest.raises(ValueError, match="keeps no attribute"):
        select_pre_maet(_pm(), attributes=[])


def test_an_unknown_name_is_refused():
    with pytest.raises(KeyError, match="velocity"):
        select_pre_maet(_pm(), attributes=["velocity"])


def test_an_out_of_range_index_is_refused():
    with pytest.raises(IndexError, match="out of range"):
        select_pre_maet(_pm(), events=[5])


def test_a_wrong_length_mask_is_refused():
    with pytest.raises(ValueError, match="length 3"):
        select_pre_maet(_pm(), events=[True, False])


def test_selecting_may_leave_an_event_with_no_value():
    """Allowed, and it means what it says: the event contributes nothing
    on that attribute while keeping its place."""
    p = [np.array([[60.0, np.nan]]), np.array([[0.0, 1.0]])]
    w = [np.array([[1.0, 0.0]]), np.ones((1, 2))]
    pm = pre_maet(p, w, flat_specs(p, name=["pitch", "onset"]))
    out, _, _ = unpack_pre_maet(select_pre_maet(pm, events=[1]))
    assert np.isnan(out[0][0, 0])


def test_the_result_is_a_pre_maet_that_builds():
    pm = select_pre_maet(_pm(), attributes=["pitch"], events=[0, 1])
    p, w, specs = unpack_pre_maet(pm)
    dens = build_maet(p, w, [1.0], [2], [False], [False], [0.0],
                      verbose=False)
    assert sim_maet(dens, dens, verbose=False) == pytest.approx(1.0)
