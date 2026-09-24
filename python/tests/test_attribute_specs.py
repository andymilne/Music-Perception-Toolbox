"""Attributes that carry their own parameters, and what a role fixes.

``pre_maet_from_attr_table`` takes one entry per attribute, either a column
name or a mapping of that name and the attribute's parameters. Where a
role also fixes ``r`` or ``exch``, ``'ordered_multiset'`` yields to the
analyst with a warning and ``'simplex'`` refuses.

Mirror of MATLAB tests/test_attribute_specs.m.
"""

import os
import sys

import numpy as np
import pytest

from mpt import pre_maet_from_attr_table, grid_attr_table, unpack_pre_maet

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demos"))


@pytest.fixture
def chorale():
    from jmm import jmm_data
    return grid_attr_table(jmm_data.bwv347_notes(), 0.25)


def _specs(pm):
    return unpack_pre_maet(pm)[2]


def test_a_bare_name_is_not_an_attribute(chorale):
    """A name carries no sigma, so it cannot describe an attribute."""
    with pytest.raises(TypeError, match="not as a bare name"):
        pre_maet_from_attr_table(chorale, attributes=("pitch",), time="beats")


def test_the_attributes_are_required(chorale):
    with pytest.raises(TypeError, match="attributes"):
        pre_maet_from_attr_table(chorale, time="beats")


def test_every_attribute_needs_a_width(chorale):
    with pytest.raises(ValueError, match="no sigma"):
        pre_maet_from_attr_table(
            chorale, attributes=(dict(column="onset"),), time="beats")


def test_several_values_at_an_event_need_r_and_exch(chorale):
    """Neither follows from the score, and neither has an identity: r
    says how many of the chord's values a tuple takes, exch whether
    their order signifies."""
    with pytest.raises(ValueError, match="needs r and exch"):
        pre_maet_from_attr_table(
            chorale, attributes=(dict(column="pitch", sigma=0.5),),
            time="beats")
    # One value per event determines both, so neither is asked for.
    specs = _specs(pre_maet_from_attr_table(
        chorale, attributes=(dict(column="pitch", sigma=0.5),),
        time="beats", chords="separate"))
    assert specs[0]["r"] == 1


def test_an_entry_carries_the_attribute_s_parameters(chorale):
    specs = _specs(pre_maet_from_attr_table(chorale, time="beats", attributes=(
        dict(column="pitch", name="pitchClass", sigma=0.5, r=4, exch=False,
             is_per=True, period=12.0),
        dict(column="pitch", name="pitchHeight", sigma=8.0, r=4, exch=False,
             rel=True),
        dict(column="onset", sigma=0.5))))
    assert [s["name"] for s in specs] == \
        ["pitchClass", "pitchHeight", "onset"]
    assert [s["sigma"] for s in specs] == [0.5, 8.0, 0.5]
    assert [s["is_per"] for s in specs] == [True, False, False]
    assert [s["period"] for s in specs] == [12.0, 0.0, 0.0]
    assert [s["rel"] for s in specs] == [False, True, False]


def test_the_parameters_do_not_touch_the_values(chorale):
    """Two attributes of one column under different parameters hold the
    same values, which is what makes pitch class and pitch height
    readable from one pitch column."""
    p, _, specs = unpack_pre_maet(pre_maet_from_attr_table(
        chorale, time="beats", chords="separate", attributes=(
            dict(column="pitch", name="pitchClass", sigma=0.5,
                 is_per=True, period=12.0),
            dict(column="pitch", name="pitchHeight", sigma=8.0))))
    np.testing.assert_array_equal(p[0], p[1])
    assert specs[0]["is_per"] is True and specs[1]["is_per"] is False


def test_a_mapping_needs_its_column(chorale):
    with pytest.raises(ValueError, match="needs a 'column' key"):
        pre_maet_from_attr_table(chorale, attributes=(dict(sigma=0.5),))


def test_an_unknown_parameter_is_refused(chorale):
    with pytest.raises(ValueError, match="unknown parameter"):
        pre_maet_from_attr_table(chorale,
                            attributes=(dict(column="pitch", width=0.5),))


def test_periodicity_needs_a_period(chorale):
    with pytest.raises(ValueError, match="needs a period"):
        pre_maet_from_attr_table(
            chorale, attributes=(dict(column="pitch", is_per=True),))


def test_a_role_may_carry_its_attribute_s_parameters(chorale):
    """The simplex role creates an attribute of its own, so its width
    has nowhere else to come from."""
    specs = _specs(pre_maet_from_attr_table(
        chorale, attributes=(dict(column="pitch", sigma=0.5),),
        time="beats", chords="separate",
        roles={"part": dict(role="simplex", sigma=0.2, name="voice")}))
    assert [s["name"] for s in specs] == ["pitch", "voice"]
    assert specs[1]["sigma"] == 0.2
    assert specs[1]["r"] == 3 and specs[1]["exch"] is False


def test_ordered_multiset_yields_the_tuple_size_with_a_warning(chorale):
    """The role fills the slots; how many are drawn from them is the
    analyst's question, and the article reads this shape at r = 2."""
    with pytest.warns(UserWarning, match="implies r = 4"):
        pm = pre_maet_from_attr_table(
            chorale, attributes=(dict(column="pitch", sigma=0.5, r=2),),
            time="beats", roles={"part": "ordered_multiset"})
    assert _specs(pm)[0]["r"] == 2


def test_ordered_multiset_yields_the_order_with_a_warning(chorale):
    with pytest.warns(UserWarning, match="implies exch = False"):
        pm = pre_maet_from_attr_table(
            chorale, attributes=(dict(column="pitch", sigma=0.5, exch=True),),
            time="beats", roles={"part": "ordered_multiset"})
    assert _specs(pm)[0]["exch"] is True


def test_ordered_multiset_is_silent_where_the_values_agree(chorale):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pre_maet_from_attr_table(
            chorale,
            attributes=(dict(column="pitch", sigma=0.5, r=4, exch=False),),
            time="beats", roles={"part": "ordered_multiset"})


def test_simplex_refuses_another_tuple_size(chorale):
    """A tuple of some of a point's coordinates is not a point; the
    message names the nesting route to the supplement's outer r."""
    with pytest.raises(ValueError, match="bind_attributes then bind_events"):
        pre_maet_from_attr_table(
            chorale, attributes=(dict(column="pitch", sigma=0.5),),
            time="beats", chords="separate",
            roles={"part": dict(role="simplex", sigma=0.2, r=2)})


def test_simplex_refuses_an_unordered_reading(chorale):
    with pytest.raises(ValueError, match="read in order"):
        pre_maet_from_attr_table(
            chorale, attributes=(dict(column="pitch", sigma=0.5),),
            time="beats", chords="separate",
            roles={"part": dict(role="simplex", sigma=0.2, exch=True)})


def test_separate_attributes_carries_the_parameters_to_every_level(chorale):
    specs = _specs(pre_maet_from_attr_table(
        chorale, time="beats",
        attributes=(dict(column="pitch", name="p", sigma=0.5,
                         is_per=True, period=12.0),),
        roles={"part": "separate_attributes"}))
    assert [s["name"] for s in specs] == \
        ["p_Soprano", "p_Alto", "p_Tenor", "p_Bass"]
    assert all(s["sigma"] == 0.5 and s["is_per"] for s in specs)


# --- columns the score reader never wrote --------------------------------


@pytest.fixture
def spatial():
    """A table that never saw a score: three spatial axes per event."""
    import pandas as pd
    return pd.DataFrame(dict(
        onset_beats=[0.0, 1.0, 2.0], onset_seconds=[0.0, 0.5, 1.0],
        duration_beats=[1.0] * 3, duration_seconds=[0.5] * 3,
        pitch=[60.0, 62.0, 64.0], velocity=[90.0] * 3,
        part=pd.Categorical(["V"] * 3), measure=[1, 1, 2],
        x=[0.0, 1.0, 2.0], y=[3.0, 4.0, 5.0], z=[6.0, 7.0, 8.0]))


def test_any_column_may_be_an_attribute(spatial):
    p, _, specs = unpack_pre_maet(pre_maet_from_attr_table(
        spatial, time="beats", chords="separate", attributes=(
            dict(column="x", sigma=0.5), dict(column="y", sigma=0.5),
            dict(column="z", sigma=0.5))))
    assert [s["name"] for s in specs] == ["x", "y", "z"]
    np.testing.assert_array_equal(p[0][0], [0.0, 1.0, 2.0])


def test_arbitrary_columns_bind_into_one_position(spatial):
    """The whole path: an arbitrary table converts, and the axes that
    describe one position become one attribute read whole."""
    from mpt import bind_attributes
    pm = pre_maet_from_attr_table(
        spatial, time="beats", chords="separate", attributes=(
            dict(column="x", sigma=0.5), dict(column="y", sigma=0.5),
            dict(column="z", sigma=0.5)))
    p, _, specs = unpack_pre_maet(bind_attributes(
        pm, ["x", "y", "z"], name="position", r=3, exch=False))
    assert len(p) == 1 and p[0].shape == (3, 3)
    np.testing.assert_array_equal(p[0][:, 0], [0.0, 3.0, 6.0])
    assert specs[0]["sigma"] == 0.5


def test_an_absent_column_is_not_an_attribute(spatial):
    with pytest.raises(ValueError, match="nor a column of the table"):
        pre_maet_from_attr_table(
            spatial, time="beats", attributes=(dict(column="w", sigma=0.5),))


def test_a_categorical_column_is_sent_to_the_roles(spatial):
    import pandas as pd
    t = spatial.copy()
    t["rule"] = pd.Categorical(["V_I", "Repeat", "V_I"])
    with pytest.raises(TypeError, match="Give it to 'roles' instead"):
        pre_maet_from_attr_table(
            t, time="beats", attributes=(dict(column="rule", sigma=0.5),))
    # Which is where it belongs: the supplement's unrolled grammar
    # encoding is a label attribute on its own simplex.
    _, _, specs = unpack_pre_maet(pre_maet_from_attr_table(
        t, time="beats", chords="separate",
        attributes=(dict(column="x", sigma=0.5),),
        roles={"rule": dict(role="simplex", sigma=0.1)}))
    assert [s["name"] for s in specs] == ["x", "rule"]
    assert specs[1]["sigma"] == 0.1
