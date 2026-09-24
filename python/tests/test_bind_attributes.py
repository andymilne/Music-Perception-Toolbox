"""Binding along the attribute axis, and its inverse.

``bind_attributes`` gathers several attributes into one whose value at
an event is the tuple of all of them; ``separate_attributes`` splits one
back into a slot apiece. The two are inverse, and the second is also the
operation by which the conversion's two structural roles differ.

Mirror of MATLAB tests/test_bind_attributes.m.
"""

import os
import sys

import numpy as np
import pytest

import mpt
from mpt import (bind_attributes, separate_attributes, grid_attr_table,
                 pack_pre_maet, pre_maet_from_attr_table, unpack_pre_maet)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demos"))


@pytest.fixture
def axes():
    """Three spatial axes of one position, as three attributes."""
    return pack_pre_maet(
        [np.array([[1.0, 2.0, 3.0]]), np.array([[4.0, 5.0, 6.0]]),
         np.array([[7.0, 8.0, 9.0]])], None,
        [dict(name=n, r=1, exch=True, sigma=0.5, rel=False,
              is_per=False, period=0.0) for n in ("x", "y", "z")])


@pytest.fixture
def chorale():
    from jmm import jmm_data
    return grid_attr_table(jmm_data.bwv347_notes(), 0.25)


def test_binding_stacks_the_values_in_the_order_given(axes):
    p, _, specs = unpack_pre_maet(bind_attributes(
        axes, ["x", "y", "z"], name="position", r=3, exch=False))
    assert len(p) == 1
    assert p[0].shape == (3, 3)
    np.testing.assert_array_equal(p[0][:, 0], [1.0, 4.0, 7.0])
    assert specs[0]["name"] == "position"
    assert specs[0]["r"] == 3 and specs[0]["exch"] is False


def test_the_order_listed_is_the_order_bound(axes):
    p, _, _ = unpack_pre_maet(bind_attributes(
        axes, ["z", "x", "y"], name="position", r=3, exch=False))
    np.testing.assert_array_equal(p[0][:, 0], [7.0, 1.0, 4.0])


def test_the_bound_attribute_takes_the_place_of_the_first(axes):
    p, _, specs = unpack_pre_maet(bind_attributes(
        axes, ["y", "z"], name="yz", r=2, exch=False))
    assert [s["name"] for s in specs] == ["x", "yz"]
    assert [m.shape for m in p] == [(1, 3), (2, 3)]


def test_agreed_parameters_are_inherited(axes):
    _, _, specs = unpack_pre_maet(bind_attributes(
        axes, ["x", "y"], name="xy", r=2, exch=False))
    assert specs[0]["sigma"] == 0.5
    assert specs[0]["is_per"] is False


def test_disagreeing_parameters_must_be_given(axes):
    p_attr, w_attr, specs = unpack_pre_maet(axes)
    specs[1]["sigma"] = 2.0
    mixed = pack_pre_maet(p_attr, w_attr, specs)
    with pytest.raises(ValueError, match="disagree on sigma"):
        bind_attributes(mixed, ["x", "y"], name="xy", r=2, exch=False)
    _, _, out = unpack_pre_maet(bind_attributes(
        mixed, ["x", "y"], name="xy", r=2, exch=False, sigma=1.0))
    assert out[0]["sigma"] == 1.0


def test_the_bound_attribute_needs_r_and_exch(axes):
    """Neither follows from the inputs, and neither has an identity."""
    with pytest.raises(ValueError, match="needs r and exch"):
        bind_attributes(axes, ["x", "y"], name="xy")


def test_the_bound_attribute_needs_a_name(axes):
    with pytest.raises(ValueError, match="needs a name"):
        bind_attributes(axes, ["x", "y"], r=2, exch=False)


def test_binding_one_attribute_is_refused(axes):
    with pytest.raises(ValueError, match="at least two"):
        bind_attributes(axes, ["x"], name="xx", r=1, exch=False)


def test_an_attribute_cannot_be_bound_to_itself(axes):
    with pytest.raises(ValueError, match="cannot be bound to itself"):
        bind_attributes(axes, ["x", "x"], name="xx", r=2, exch=False)


def test_weights_follow_their_values(axes):
    p_attr, _, specs = unpack_pre_maet(axes)
    w = [np.full((1, 3), k + 1.0) for k in range(3)]
    p, wout, _ = unpack_pre_maet(bind_attributes(
        pack_pre_maet(p_attr, w, specs), ["x", "y", "z"],
        name="position", r=3, exch=False))
    np.testing.assert_array_equal(wout[0][:, 0], [1.0, 2.0, 3.0])


def test_separating_undoes_binding(axes):
    bound = bind_attributes(axes, ["x", "y", "z"], name="position",
                            r=3, exch=False)
    p, _, specs = unpack_pre_maet(separate_attributes(bound, "position"))
    assert [s["name"] for s in specs] == \
        ["position_1", "position_2", "position_3"]
    for part, original in zip(p, unpack_pre_maet(axes)[0]):
        np.testing.assert_array_equal(part, original)


def test_the_parts_carry_the_source_s_kernel(axes):
    bound = bind_attributes(axes, ["x", "y"], name="xy", r=2, exch=False,
                            sigma=3.0, is_per=True, period=12.0)
    _, _, specs = unpack_pre_maet(separate_attributes(bound, "xy"))
    for spec in specs[:2]:
        assert spec["sigma"] == 3.0
        assert spec["is_per"] is True and spec["period"] == 12.0
        assert spec["r"] == 1      # one value per event determines both
        assert spec["exch"] is True


def test_the_parts_may_be_named(axes):
    bound = bind_attributes(axes, ["x", "y"], name="xy", r=2, exch=False)
    _, _, specs = unpack_pre_maet(separate_attributes(
        bound, "xy", names=["left", "right"]))
    assert [s["name"] for s in specs[:2]] == ["left", "right"]


def test_separating_a_single_valued_attribute_is_refused(axes):
    with pytest.raises(ValueError, match="nothing to separate"):
        separate_attributes(axes, "x")


def test_the_two_structural_roles_differ_by_this_operation(chorale):
    """Under 'ordered_multiset' slot k is level k, so splitting that
    attribute slot by slot is what 'separate_attributes' builds from the
    table -- up to the names, which only the table carries."""
    attributes = (dict(column="pitch", sigma=0.5),)
    by_role = pre_maet_from_attr_table(
        chorale, attributes=attributes, time="beats",
        roles={"part": "separate_attributes"})
    by_operation = separate_attributes(pre_maet_from_attr_table(
        chorale, attributes=attributes, time="beats",
        roles={"part": "ordered_multiset"}), "pitch")

    p_role, _, s_role = unpack_pre_maet(by_role)
    p_op, _, s_op = unpack_pre_maet(by_operation)
    assert len(p_op) == len(p_role) == 4
    for a, b in zip(p_role, p_op):
        np.testing.assert_array_equal(a, b)
    assert [(s["r"], s["exch"], s["sigma"]) for s in s_role] == \
           [(s["r"], s["exch"], s["sigma"]) for s in s_op]


def test_a_bound_attribute_builds(axes):
    """The point of binding: the density reads the tuple, not the
    product of three separate attributes."""
    bound = bind_attributes(axes, ["x", "y", "z"], name="position",
                            r=3, exch=False)
    d = mpt.build_maet(bound, verbose=False)
    assert mpt.sim_maet(d, d, verbose=False) == pytest.approx(1.0)
