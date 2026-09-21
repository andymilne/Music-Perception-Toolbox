"""Sampling an event table on a grid.

The acceptance case is Analysis 1.1 of the JMM article, which samples
BWV 347 on a sixteenth-note grid: N = 272 events, held notes replicating
across the points they occupy, and no rests.

Mirror of MATLAB tests/test_grid_events.m.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

import mpt
from mpt import build_maet, eval_maet, grid_events, pre_maet_from_score, \
    unpack_pre_maet

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demos"))


def _gapped():
    """Two notes an octave of time apart, so the grid has empty points."""
    return pd.DataFrame({
        "onset_beats": [0.0, 3.0],
        "onset_seconds": [0.0, 1.5],
        "duration_beats": [1.0, 1.0],
        "duration_seconds": [0.5, 0.5],
        "pitch": [60.0, 64.0],
        "velocity": [90.0, 90.0],
        "part": pd.Categorical(["A", "A"], categories=["A", "B"]),
        "voice": np.array([1, 1]),
        "measure": np.array([1, 1]),
        "fermata": np.array([False, True]),
    })


def _held():
    """One note of four beats and one of one, so replication is visible."""
    return pd.DataFrame({
        "onset_beats": [0.0, 0.0],
        "onset_seconds": [0.0, 0.0],
        "duration_beats": [4.0, 1.0],
        "duration_seconds": [2.0, 0.5],
        "pitch": [60.0, 67.0],
        "velocity": [90.0, 90.0],
        "part": pd.Categorical(["A", "B"], categories=["A", "B"]),
        "voice": np.array([1, 1]),
        "measure": np.array([1, 1]),
        "fermata": np.array([False, False]),
    })


# --- the acceptance case -------------------------------------------------


def test_bwv347_gives_the_manuscript_grid():
    """Analysis 1.1: 272 grid points at 0.25 QN, four voices, no rests."""
    from jmm import jmm_data
    t = jmm_data.bwv347_notes()
    g = grid_events(t, 0.25)
    assert g["grid_index"].nunique() == 272
    assert len(g) == 272 * 4
    assert int(g["note_id"].isna().sum()) == 0


# --- replication and occupancy ------------------------------------------


def test_a_held_note_replicates_across_the_points_it_covers():
    g = grid_events(_held(), 1.0)
    held = g[g["note_id"] == 0]
    short = g[g["note_id"] == 1]
    assert len(held) == 4
    assert len(short) == 1
    assert held["grid_index"].tolist() == [0, 1, 2, 3]


def test_a_note_ending_on_a_grid_point_does_not_occupy_it():
    """Occupancy is half-open, so the note's end is the next note's point."""
    g = grid_events(_held(), 1.0)
    assert 1 not in g[g["note_id"] == 1]["grid_index"].tolist()


def test_duration_still_means_the_notes_own_duration():
    g = grid_events(_held(), 1.0)
    assert g[g["note_id"] == 0]["duration_beats"].unique().tolist() == [4.0]


# --- empty grid points ---------------------------------------------------


def test_an_empty_grid_point_is_a_row_with_its_note_columns_missing():
    g = grid_events(_gapped(), 1.0)
    empty = g[g["note_id"].isna()]
    assert len(empty) == 2
    assert empty["grid_index"].tolist() == [1, 2]
    assert empty["grid_onset_beats"].tolist() == [1.0, 2.0]
    assert empty[["pitch", "velocity", "voice", "measure",
                  "fermata", "part"]].isna().all().all()


def test_empty_points_survive_into_a_pre_maet_as_events_with_no_value():
    pm = pre_maet_from_score(grid_events(_gapped(), 1.0),
                             attributes=("pitch",), weights="weight",
                             time="beats")
    p, w, _ = unpack_pre_maet(pm)
    assert np.isnan(p[0][0, [1, 2]]).all()
    assert (w[0][0, [1, 2]] == 0).all()


# --- weight policies -----------------------------------------------------


def test_item_weighting_makes_each_note_count_once():
    from jmm import jmm_data
    t = jmm_data.bwv347_notes()
    g = grid_events(t, 0.25, weights="item")
    assert float(g["weight"].sum()) == pytest.approx(len(t))


def _straddling():
    """One note of a beat and a half, so it fills one slice and half of
    the next, and the three policies disagree."""
    return pd.DataFrame({
        "onset_beats": [0.0], "onset_seconds": [0.0],
        "duration_beats": [1.5], "duration_seconds": [0.75],
        "pitch": [60.0], "velocity": [90.0],
        "part": pd.Categorical(["A"], categories=["A"]),
        "voice": np.array([1]), "measure": np.array([1]),
        "fermata": np.array([False]),
    })


@pytest.mark.parametrize("policy, expected", [
    ("presence", [1.0, 1.0]),      # sounding in both slices
    ("coverage", [1.0, 0.5]),      # the fraction of each slice it fills
    ("item", [2 / 3, 1 / 3]),      # the fraction of the note in each
])
def test_the_three_policies_on_a_straddling_note(policy, expected):
    g = grid_events(_straddling(), 1.0, weights=policy)
    assert g["weight"].tolist() == pytest.approx(expected)


def test_coverage_is_the_default():
    g = grid_events(_straddling(), 1.0)
    assert g["weight"].tolist() == pytest.approx([1.0, 0.5])


def test_presence_counts_a_note_once_per_slice_it_sounds_in():
    g = grid_events(_held(), 1.0, weights="presence")
    assert float(g["weight"].sum()) == pytest.approx(5.0)


def test_a_note_shorter_than_the_step_is_not_lost():
    """A note falling between two grid points still overlaps a slice, so
    nothing disappears for being short."""
    t = _straddling()
    t["onset_beats"] = [0.25]
    t["duration_beats"] = [0.25]
    for policy, expected in (("presence", 1.0), ("coverage", 0.25),
                             ("item", 1.0)):
        g = grid_events(t, 1.0, weights=policy, limits=(0.0, 1.0))
        assert g["weight"].tolist() == pytest.approx([expected])


def test_item_weighting_reproduces_the_ungridded_density_at_r_one():
    """The kernel is linear in weight, so splitting a note's weight over
    the points it covers leaves the density unchanged."""
    from jmm import jmm_data
    t = jmm_data.bwv347_notes()
    g = grid_events(t, 0.25, weights="item")

    def density(pm):
        p, w, _ = unpack_pre_maet(pm)
        return build_maet(p, w, [50.0], [1], [False], [False], [0],
                          verbose=False)

    ungridded = density(pre_maet_from_score(
        t, attributes=("pitch",), chords="separate", weights="ones",
        time="beats"))
    gridded = density(pre_maet_from_score(
        g, attributes=("pitch",), weights="weight", time="beats"))
    query = [[64.0]]
    assert eval_maet(gridded, query, verbose=False) == pytest.approx(
        eval_maet(ungridded, query, verbose=False), rel=1e-12)


# --- the grid specification ---------------------------------------------


def test_the_grid_runs_from_zero_to_the_last_note_end_by_default():
    g = grid_events(_gapped(), 1.0)
    assert g["grid_onset_beats"].max() == pytest.approx(3.0)


def test_limits_override_the_extent():
    g = grid_events(_gapped(), 1.0, limits=(0.0, 2.0))
    assert g["grid_index"].nunique() == 2


def test_a_seconds_grid_uses_the_seconds_columns():
    g = grid_events(_gapped(), 0.5, time="seconds")
    assert "grid_onset_seconds" in g.columns
    assert g["grid_index"].nunique() == 4


def test_gridding_over_a_missing_time_base_says_so():
    t = _gapped().drop(columns=["onset_seconds", "duration_seconds"])
    with pytest.raises(KeyError, match="beat map"):
        grid_events(t, 1.0, time="seconds")


def test_the_sounding_duration_may_define_occupancy():
    t = _gapped()
    t["sounding_duration_beats"] = [3.0, 1.0]
    g = grid_events(t, 1.0, duration="sounding_duration")
    assert int(g["note_id"].isna().sum()) == 0


@pytest.mark.parametrize("bad", [0.0, -1.0, np.inf])
def test_the_step_must_be_positive_and_finite(bad):
    with pytest.raises(ValueError, match="positive"):
        grid_events(_gapped(), bad)


def test_the_result_records_that_it_is_gridded():
    g = grid_events(_gapped(), 1.0)
    assert g.attrs["granularity"] == "grid"
    assert g.attrs["grid_step"] == 1.0
    assert g.attrs["grid_time"] == "beats"
