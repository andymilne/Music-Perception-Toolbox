"""Sampling an attribute table on a grid.

The acceptance case is Analysis 1.1 of the JMM article, which samples
BWV 347 on a sixteenth-note grid: N = 272 events, held notes replicating
across the points they occupy, and no rests.

Mirror of MATLAB tests/test_grid_attr_table.m.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

import mpt
from mpt import build_maet, eval_maet, grid_attr_table, pre_maet_from_attr_table, \
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
    g = grid_attr_table(t, 0.25)
    assert g["grid_index"].nunique() == 272
    assert len(g) == 272 * 4
    assert int(g["note_id"].isna().sum()) == 0


# --- replication and occupancy ------------------------------------------


def test_a_held_note_replicates_across_the_points_it_covers():
    g = grid_attr_table(_held(), 1.0)
    held = g[g["note_id"] == 0]
    short = g[g["note_id"] == 1]
    assert len(held) == 4
    assert len(short) == 1
    assert held["grid_index"].tolist() == [0, 1, 2, 3]


def test_a_note_ending_on_a_grid_point_does_not_occupy_it():
    """Occupancy is half-open, so the note's end is the next note's point."""
    g = grid_attr_table(_held(), 1.0)
    assert 1 not in g[g["note_id"] == 1]["grid_index"].tolist()


def test_duration_still_means_the_notes_own_duration():
    g = grid_attr_table(_held(), 1.0)
    assert g[g["note_id"] == 0]["duration_beats"].unique().tolist() == [4.0]


# --- empty grid points ---------------------------------------------------


def test_an_empty_grid_point_is_a_row_with_its_note_columns_missing():
    g = grid_attr_table(_gapped(), 1.0)
    empty = g[g["note_id"].isna()]
    assert len(empty) == 2
    assert empty["grid_index"].tolist() == [1, 2]
    assert empty["grid_onset_beats"].tolist() == [1.0, 2.0]
    assert empty[["pitch", "velocity", "voice", "measure",
                  "fermata", "part"]].isna().all().all()


def test_empty_points_survive_into_a_pre_maet_as_events_with_no_value():
    pm = pre_maet_from_attr_table(grid_attr_table(_gapped(), 1.0),
                             attributes=(dict(column="pitch", sigma=1.0, r=1, exch=True),), weights="weight",
                             time="beats")
    p, w, _ = unpack_pre_maet(pm)
    assert np.isnan(p[0][0, [1, 2]]).all()
    assert (w[0][0, [1, 2]] == 0).all()


# --- weight policies -----------------------------------------------------


def test_item_weighting_makes_each_note_count_once():
    from jmm import jmm_data
    t = jmm_data.bwv347_notes()
    g = grid_attr_table(t, 0.25, weights="item")
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
    g = grid_attr_table(_straddling(), 1.0, weights=policy)
    assert g["weight"].tolist() == pytest.approx(expected)


def test_coverage_is_the_default():
    g = grid_attr_table(_straddling(), 1.0)
    assert g["weight"].tolist() == pytest.approx([1.0, 0.5])


def test_presence_counts_a_note_once_per_slice_it_sounds_in():
    g = grid_attr_table(_held(), 1.0, weights="presence")
    assert float(g["weight"].sum()) == pytest.approx(5.0)


def test_a_note_shorter_than_the_step_is_not_lost():
    """A note falling between two grid points still overlaps a slice, so
    nothing disappears for being short."""
    t = _straddling()
    t["onset_beats"] = [0.25]
    t["duration_beats"] = [0.25]
    for policy, expected in (("presence", 1.0), ("coverage", 0.25),
                             ("item", 1.0)):
        g = grid_attr_table(t, 1.0, weights=policy, limits=(0.0, 1.0))
        assert g["weight"].tolist() == pytest.approx([expected])


def test_item_weighting_reproduces_the_ungridded_density_at_r_one():
    """The kernel is linear in weight, so splitting a note's weight over
    the points it covers leaves the density unchanged."""
    from jmm import jmm_data
    t = jmm_data.bwv347_notes()
    g = grid_attr_table(t, 0.25, weights="item")

    def density(pm):
        p, w, _ = unpack_pre_maet(pm)
        return build_maet(p, w, [50.0], [1], [False], [False], [0],
                          verbose=False)

    ungridded = density(pre_maet_from_attr_table(
        t, attributes=(dict(column="pitch", sigma=1.0, r=1, exch=True),), chords="separate", weights="ones",
        time="beats"))
    gridded = density(pre_maet_from_attr_table(
        g, attributes=(dict(column="pitch", sigma=1.0, r=1, exch=True),), weights="weight", time="beats"))
    query = [[64.0]]
    assert eval_maet(gridded, query, verbose=False) == pytest.approx(
        eval_maet(ungridded, query, verbose=False), rel=1e-12)


# --- the grid specification ---------------------------------------------


def test_the_grid_runs_from_zero_to_the_last_note_end_by_default():
    g = grid_attr_table(_gapped(), 1.0)
    assert g["grid_onset_beats"].max() == pytest.approx(3.0)


def test_limits_override_the_extent():
    g = grid_attr_table(_gapped(), 1.0, limits=(0.0, 2.0))
    assert g["grid_index"].nunique() == 2


def test_a_seconds_grid_uses_the_seconds_columns():
    g = grid_attr_table(_gapped(), 0.5, time="seconds")
    assert "grid_onset_seconds" in g.columns
    assert g["grid_index"].nunique() == 4


def test_gridding_over_a_missing_time_base_says_so():
    t = _gapped().drop(columns=["onset_seconds", "duration_seconds"])
    with pytest.raises(KeyError, match="beat map"):
        grid_attr_table(t, 1.0, time="seconds")


def test_the_sounding_duration_may_define_occupancy():
    t = _gapped()
    t["sounding_duration_beats"] = [3.0, 1.0]
    g = grid_attr_table(t, 1.0, duration="sounding_duration")
    assert int(g["note_id"].isna().sum()) == 0


@pytest.mark.parametrize("bad", [0.0, -1.0, np.inf])
def test_the_step_must_be_positive_and_finite(bad):
    with pytest.raises(ValueError, match="positive"):
        grid_attr_table(_gapped(), bad)


def test_the_result_records_that_it_is_gridded():
    g = grid_attr_table(_gapped(), 1.0)
    assert g.attrs["granularity"] == "grid"
    assert g.attrs["grid_step"] == 1.0
    assert g.attrs["grid_time"] == "beats"


# --- the grid carries both time bases ------------------------------------


@pytest.fixture
def chorale():
    from jmm import jmm_data
    return jmm_data.bwv347_notes()


def test_the_grid_carries_both_time_bases(chorale):
    """The grid steps in one unit; its points have a time in both, so a
    metrical grid can be read on a clock."""
    g = grid_attr_table(chorale, 0.25)
    assert "grid_onset_beats" in g.columns
    assert "grid_onset_seconds" in g.columns
    points = g.drop_duplicates("grid_index")
    # BWV 347 is read at a constant tempo, so the map is exact.
    ratio = (points["grid_onset_seconds"].to_numpy()
             / np.where(points["grid_onset_beats"].to_numpy() == 0, np.nan,
                        points["grid_onset_beats"].to_numpy()))
    assert np.allclose(ratio[1:], ratio[1], equal_nan=False)


def test_a_beat_grid_may_be_read_in_seconds(chorale):
    """Which is the point: metrical slices, a sigma on a clock."""
    g = grid_attr_table(chorale, 0.25)
    p, _, specs = mpt.unpack_pre_maet(mpt.pre_maet_from_attr_table(
        g, time="seconds", attributes=(
            dict(column="pitch", sigma=0.5, r=4, exch=False),
            dict(column="onset", sigma=0.05))))
    np.testing.assert_allclose(p[1][0][:4], [0.0, 0.125, 0.25, 0.375])


def test_a_seconds_grid_carries_beats_too(chorale):
    g = grid_attr_table(chorale, 0.125, time="seconds")
    assert "grid_onset_beats" in g.columns
    points = g.drop_duplicates("grid_index")
    np.testing.assert_allclose(points["grid_onset_beats"].to_numpy()[:3],
                               [0.0, 0.25, 0.5])


def test_one_time_base_gives_one_grid_onset(chorale):
    """A bare performance has no beat map, so there is nothing to map
    the grid points onto."""
    bare = chorale.drop(columns=["onset_beats", "duration_beats"])
    g = grid_attr_table(bare, 0.125, time="seconds")
    assert "grid_onset_seconds" in g.columns
    assert "grid_onset_beats" not in g.columns
    with pytest.raises(ValueError, match="carries no grid onset in beats"):
        mpt.pre_maet_from_attr_table(
            g, time="beats", attributes=(dict(column="onset", sigma=1.0),))


def test_the_grid_collapses_back_to_the_table_it_came_from(chorale):
    """ungrid_attr_table is the inverse: every column comes back, with
    the dtypes the source had."""
    back = mpt.ungrid_attr_table(grid_attr_table(chorale, 0.25))
    assert list(back.columns) == list(chorale.columns)
    assert len(back) == len(chorale)
    for column in chorale.columns:
        assert back[column].dtype == chorale[column].dtype
        np.testing.assert_array_equal(back[column].to_numpy(),
                                      chorale[column].to_numpy())


def test_ungridding_drops_a_weight_the_grid_overwrote():
    """The grid adds weight to a table that had none and overwrites the
    weight of one that did, so the column it leaves is the slice's and
    not the note's. Comparing the two tables' columns would keep it;
    ungrid_attr_table removes it and says so."""
    import os
    midi = mpt.read_score(os.path.join(os.path.dirname(__file__), "data",
                                       "score_small.mid"))
    assert "weight" in midi.columns
    g = grid_attr_table(midi, 0.5)
    assert "weight" not in [c for c in g.columns if c not in midi.columns]
    back = mpt.ungrid_attr_table(g)
    assert "weight" not in back.columns
    for column in back.columns:
        np.testing.assert_array_equal(back[column].to_numpy(),
                                      midi[column].to_numpy())


def test_ungridding_needs_a_gridded_table(chorale):
    with pytest.raises(KeyError, match="no 'note_id' column"):
        mpt.ungrid_attr_table(chorale)


def test_limits_are_the_one_lossy_case(chorale):
    """Notes outside the grid's span are cut, which is a truncation the
    caller asked for rather than a failure of the collapse."""
    g = grid_attr_table(chorale, 0.25, limits=(0.0, 8.0))
    assert g["note_id"].nunique() < len(chorale)


# --- ungridding a table the caller has edited ----------------------------


def test_ungridding_survives_edits_that_keep_it_a_grid(chorale):
    """Rows and columns may go: the empty slices, a span of slices, or a
    column. A note whose first slice was dropped comes back from its
    next, the source's columns repeating unchanged across its slices."""
    g = grid_attr_table(chorale, 1.0)
    assert len(mpt.ungrid_attr_table(g.dropna(subset=["note_id"]))) == \
        len(chorale)
    fewer = mpt.ungrid_attr_table(g.drop(columns=["velocity"]))
    assert "velocity" not in fewer.columns and len(fewer) == len(chorale)

    cut = g[g["grid_index"] < 8]
    assert len(mpt.ungrid_attr_table(cut)) == cut["note_id"].nunique()

    # note 25 spans two slices; drop the first and it returns from the
    # second, with its own columns intact.
    without_first = g[~((g["note_id"] == 25) & (g["grid_index"] == 5))]
    back = mpt.ungrid_attr_table(without_first)
    k = sorted(without_first["note_id"].dropna().unique()).index(25)
    for column in ("onset_beats", "duration_beats", "pitch", "part"):
        assert back.iloc[k][column] == chorale.iloc[25][column]


def test_a_column_that_varies_within_a_note_warns(chorale):
    """A per-slice quantity the caller added has no single value to
    collapse to, so the first slice's is taken and said so."""
    g = grid_attr_table(chorale, 1.0)
    g = g.copy()
    g["per_slice"] = np.arange(len(g), dtype=float)
    with pytest.warns(UserWarning, match="per_slice vary within a note"):
        back = mpt.ungrid_attr_table(g)
    assert "per_slice" in back.columns


def test_a_column_constant_within_a_note_is_silent(chorale):
    import warnings as _w
    g = grid_attr_table(chorale, 1.0).copy()
    g["tag"] = g["pitch"] * 2
    with _w.catch_warnings():
        _w.simplefilter("error")
        back = mpt.ungrid_attr_table(g)
    np.testing.assert_allclose(back["tag"].to_numpy(),
                               chorale["pitch"].to_numpy() * 2)


# --- regridding -------------------------------------------------------------

@pytest.mark.parametrize("policy", ["coverage", "presence", "item"])
@pytest.mark.parametrize("fine,coarse", [(0.25, 1.0), (0.25, 0.5),
                                        (0.5, 1.0), (0.25, 2.0),
                                        (0.25, 0.25)])
def test_gridding_composes(chorale, policy, fine, coarse):
    """Gridding the grid equals gridding the notes. Each policy combines
    in the way that makes this hold: coverage sums and rescales by
    fine / coarse, item sums, presence takes the maximum."""
    limits = (0.0, 68.0)
    direct = grid_attr_table(chorale, coarse, weights=policy, limits=limits)
    regrid = grid_attr_table(
        grid_attr_table(chorale, fine, weights=policy, limits=limits),
        coarse, weights=policy)
    assert list(regrid.columns) == list(direct.columns)
    assert len(regrid) == len(direct)
    for column in ("grid_index", "note_id", "weight", "pitch",
                   "grid_onset_beats", "grid_onset_seconds",
                   "onset_beats", "duration_beats"):
        got = pd.to_numeric(regrid[column], errors="coerce").to_numpy(float)
        ref = pd.to_numeric(direct[column], errors="coerce").to_numpy(float)
        np.testing.assert_array_equal(np.isnan(got), np.isnan(ref))
        np.testing.assert_allclose(got, ref, atol=1e-12)


def test_regridding_keeps_empty_slices(chorale):
    """A coarse slice holding no note keeps its place, as it does when the
    notes are gridded directly."""
    fermatas = chorale[chorale["fermata"] == True]      # noqa: E712
    direct = grid_attr_table(fermatas, 1.0, limits=(0.0, 68.0))
    regrid = grid_attr_table(
        grid_attr_table(fermatas, 0.25, limits=(0.0, 68.0)), 1.0)
    assert len(regrid) == len(direct)
    assert int(regrid["note_id"].isna().sum()) == \
        int(direct["note_id"].isna().sum()) > 0
    assert list(regrid.dtypes) == list(direct.dtypes)


def test_regridding_carries_a_weighting_of_the_fine_slices(chorale):
    """The point of regridding: weight the fine slices as the analysis
    requires and the weights carry through the coarsening. Weighting the
    two eighths of a beat 1 and 1/2, normalized to mean one, is the
    metrical weighting of Analysis 1.4 of the JMM article; a note
    sounding through the beat then weighs 1 and one sounding only its
    second half weighs 1/3."""
    eighths = grid_attr_table(chorale, 0.5, limits=(0.0, 68.0)).copy()
    on_beat = np.isclose(eighths["grid_onset_beats"].to_numpy(float) % 1.0, 0.0)
    eighths["weight"] = eighths["weight"] * np.where(on_beat, 1.0, 0.5) / 0.75
    beats = grid_attr_table(eighths, 1.0)

    first = beats[beats["grid_index"] == 0]
    weights = sorted(np.round(first["weight"].to_numpy(float), 6))
    assert weights == [pytest.approx(1 / 3), pytest.approx(2 / 3),
                       1.0, 1.0, 1.0]


def test_regridding_refuses_what_it_cannot_align(chorale):
    g = grid_attr_table(chorale, 0.5, limits=(0.0, 68.0))
    with pytest.raises(ValueError, match="whole multiple"):
        grid_attr_table(g, 0.75)
    with pytest.raises(ValueError, match="is finer than"):
        grid_attr_table(g, 0.25)
    with pytest.raises(ValueError, match="how they combine"):
        grid_attr_table(g, 1.0, weights="item")
    with pytest.raises(ValueError, match="limits cannot be given"):
        grid_attr_table(g, 1.0, limits=(0.0, 4.0))
    with pytest.raises(ValueError, match="gridded over beats"):
        grid_attr_table(g, 1.0, time="seconds")


def test_ungridding_forgets_the_weighting_policy(chorale):
    """The policy is recorded so that a regrid knows how the weights
    combine; ungridding removes it with the rest of the grid."""
    g = grid_attr_table(chorale, 0.5)
    assert g.attrs["grid_weights"] == "coverage"
    assert "grid_weights" not in mpt.ungrid_attr_table(g).attrs
