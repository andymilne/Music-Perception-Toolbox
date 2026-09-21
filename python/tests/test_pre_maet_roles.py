"""How a categorical column reaches the pre-MAET.

The three encodings the JMM article contrasts on BWV 347's voicing
(Analysis 1.2): voice-aware, where the level is a position in an ordered
multiset; voice-agnostic, where it is not encoded at all; and
simplex-voice, where it is a vertex value on its own attribute and each
note is its own event.

Mirror of MATLAB tests/test_pre_maet_roles.m.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

import mpt
from mpt import (grid_events, pre_maet_from_score, simplex_vertices,
                 unpack_pre_maet)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demos"))


@pytest.fixture
def chorale():
    """BWV 347 on the sixteenth grid, so every event holds all four
    voices and the structural roles have a full slot per level."""
    from jmm import jmm_data
    return grid_events(jmm_data.bwv347_notes(), 0.25)


# --- the three encodings -------------------------------------------------


def test_ordered_multiset_puts_each_voice_in_its_own_position(chorale):
    p, _, specs = unpack_pre_maet(pre_maet_from_score(
        chorale, attributes=("pitch",), time="beats",
        roles={"part": "ordered_multiset"}))
    assert len(p) == 1
    assert p[0].shape == (4, 272)
    assert specs[0]["r"] == 4
    assert specs[0]["exch"] is False        # ordered: position is identity
    assert p[0][:, 0].tolist() == [69.0, 64.0, 61.0, 57.0]


def test_separate_attributes_gives_one_attribute_per_voice(chorale):
    p, _, specs = unpack_pre_maet(pre_maet_from_score(
        chorale, attributes=("pitch",), time="beats",
        roles={"part": "separate_attributes"}))
    assert [s["name"] for s in specs] == [
        "pitch_Soprano", "pitch_Alto", "pitch_Tenor", "pitch_Bass"]
    assert [m.shape for m in p] == [(1, 272)] * 4
    assert [float(m[0, 0]) for m in p] == [69.0, 64.0, 61.0, 57.0]


def test_simplex_tags_each_note_with_its_voice_vertex(chorale):
    p, _, specs = unpack_pre_maet(pre_maet_from_score(
        chorale, attributes=("pitch",), time="beats", chords="separate",
        roles={"part": "simplex"}))
    assert [s["name"] for s in specs] == ["pitch", "part"]
    assert p[1].shape[0] == 3                # V - 1 coordinates
    assert specs[1]["r"] == 3
    assert specs[1]["exch"] is False         # read whole: a level is one value
    np.testing.assert_allclose(p[1][:, :4].T, simplex_vertices(4), atol=1e-12)


def test_no_role_leaves_the_chord_an_unordered_multiset(chorale):
    p, _, specs = unpack_pre_maet(pre_maet_from_score(
        chorale, attributes=("pitch",), time="beats"))
    assert p[0].shape == (4, 272)
    assert specs[0]["r"] == 1
    assert specs[0]["exch"] is True


def test_drop_is_the_same_as_no_role(chorale):
    kw = dict(attributes=("pitch",), time="beats")
    a = unpack_pre_maet(pre_maet_from_score(chorale, **kw))[0][0]
    b = unpack_pre_maet(pre_maet_from_score(
        chorale, roles={"part": "drop"}, **kw))[0][0]
    np.testing.assert_array_equal(a, b)


def test_the_three_encodings_of_analysis_1_2(chorale):
    """Table 2 of the JMM article, on the same two pitch attributes:
    voice-aware is one event per chord with both ordered at r = 4;
    simplex-voice and voice-agnostic are one event per note at r = 1,
    and the agnostic one is the simplex one without its voice attribute.

    Listing 'pitch' twice gives the two attributes the analysis routes
    every pitch through -- a periodic pitch-class one and a non-periodic
    pitch-height one -- which the analyst then specs.
    """
    def encoding(**kw):
        return unpack_pre_maet(pre_maet_from_score(
            chorale, attributes=("pitch", "pitch"), time="beats", **kw))

    aware_p, _, aware_s = encoding(roles={"part": "ordered_multiset"})
    assert [m.shape for m in aware_p] == [(4, 272)] * 2
    assert [s["r"] for s in aware_s] == [4, 4]
    assert [s["exch"] for s in aware_s] == [False, False]

    simplex_p, _, simplex_s = encoding(chords="separate",
                                       roles={"part": "simplex"})
    agnostic_p, _, agnostic_s = encoding(chords="separate")
    assert [m.shape for m in agnostic_p] == [(1, 1088)] * 2
    assert [s["r"] for s in agnostic_s] == [1, 1]

    # "(c) is (b) without its voice attribute."
    assert [s["name"] for s in simplex_s] == ["pitch", "pitch", "part"]
    assert [s["name"] for s in agnostic_s] == ["pitch", "pitch"]
    for a, b in zip(agnostic_p, simplex_p[:2]):
        np.testing.assert_array_equal(a, b)


# --- what a structural category does and does not split ------------------


def test_a_structural_category_splits_every_per_note_attribute(chorale):
    _, _, specs = unpack_pre_maet(pre_maet_from_score(
        chorale, attributes=("pitch", "velocity"), time="beats",
        roles={"part": "separate_attributes"}))
    assert [s["name"] for s in specs] == [
        "pitch_Soprano", "pitch_Alto", "pitch_Tenor", "pitch_Bass",
        "velocity_Soprano", "velocity_Alto", "velocity_Tenor",
        "velocity_Bass"]


@pytest.mark.parametrize("event_level", ["onset", "measure"])
def test_event_level_attributes_are_not_split(chorale, event_level):
    """The notes gathered into one event share an onset and a bar, so
    splitting either would give one attribute per voice carrying the same
    number, raised to the fourth by the product across attributes."""
    p, _, specs = unpack_pre_maet(pre_maet_from_score(
        chorale, attributes=("pitch", event_level), time="beats",
        roles={"part": "ordered_multiset"}))
    assert [s["name"] for s in specs] == ["pitch", event_level]
    assert p[1].shape == (1, 272)
    assert specs[1]["r"] == 1


def test_a_simplex_category_is_tagged_within_each_structural_slot(chorale):
    t = chorale.copy()
    t["register"] = pd.Categorical(
        np.where(t["pitch"] > 64, "high", "low"), categories=["low", "high"])
    _, _, specs = unpack_pre_maet(pre_maet_from_score(
        t, attributes=("pitch",), time="beats",
        roles={"part": "ordered_multiset", "register": "simplex"}))
    assert [s["name"] for s in specs] == [
        "pitch", "register_Soprano", "register_Alto", "register_Tenor",
        "register_Bass"]
    assert all(s["r"] == 1 for s in specs[1:])   # two levels: one coordinate


# --- the event key -------------------------------------------------------


def test_a_gridded_table_takes_its_onset_from_the_grid(chorale):
    p, _, _ = unpack_pre_maet(pre_maet_from_score(
        chorale, attributes=("pitch", "onset"), time="beats"))
    np.testing.assert_allclose(p[1][0, :4], [0.0, 0.25, 0.5, 0.75])


def test_gridding_over_the_other_unit_is_refused(chorale):
    with pytest.raises(ValueError, match="gridded over beats"):
        pre_maet_from_score(chorale, attributes=("pitch", "onset"),
                            time="seconds")


def test_group_by_names_the_event_key(chorale):
    """An event is a contiguous run of equal keys, not every row sharing
    a value: the chorale's bars 1-4 repeat, and the two passes through
    bar 1 are two events rather than one."""
    measures = chorale["measure"].to_numpy()
    runs = 1 + int((measures[1:] != measures[:-1]).sum())
    p, _, _ = unpack_pre_maet(pre_maet_from_score(
        chorale, attributes=("pitch",), time="beats", group_by="measure"))
    assert p[0].shape[1] == runs
    assert runs > int(chorale["measure"].nunique())


# --- refusals and the warning -------------------------------------------


def test_two_structural_categories_are_refused(chorale):
    t = chorale.copy()
    t["register"] = pd.Categorical(
        np.where(t["pitch"] > 64, "high", "low"), categories=["low", "high"])
    with pytest.raises(ValueError, match="only one category may be structural"):
        pre_maet_from_score(t, attributes=("pitch",), time="beats",
                            roles={"part": "ordered_multiset",
                                   "register": "separate_attributes"})


def test_simplex_alone_needs_one_event_per_note(chorale):
    with pytest.raises(ValueError, match="chords='separate'"):
        pre_maet_from_score(chorale, attributes=("pitch",), time="beats",
                            roles={"part": "simplex"})


def test_a_structural_role_needs_bound_events(chorale):
    with pytest.raises(ValueError, match="chords='bind'"):
        pre_maet_from_score(chorale, attributes=("pitch",), time="beats",
                            chords="separate",
                            roles={"part": "ordered_multiset"})


def test_a_role_needs_a_categorical_column(chorale):
    with pytest.raises(TypeError, match="categorical"):
        pre_maet_from_score(chorale, attributes=("pitch",), time="beats",
                            roles={"pitch": "simplex"})


def test_an_unknown_role_is_refused(chorale):
    with pytest.raises(ValueError, match="unknown role"):
        pre_maet_from_score(chorale, attributes=("pitch",), time="beats",
                            roles={"part": "split"})


def test_an_event_missing_a_level_is_dropped_with_a_warning():
    """Binding the ungridded chorale by onset leaves many events without
    all four voices, since the parts do not all move together."""
    from jmm import jmm_data
    with pytest.warns(UserWarning, match="do not hold exactly one part"):
        p, _, _ = unpack_pre_maet(pre_maet_from_score(
            jmm_data.bwv347_notes(), attributes=("pitch",), time="beats",
            roles={"part": "ordered_multiset"}))
    assert p[0].shape[0] == 4
    assert p[0].shape[1] < 102          # some events lost, none ragged
    assert not np.isnan(p[0]).any()
