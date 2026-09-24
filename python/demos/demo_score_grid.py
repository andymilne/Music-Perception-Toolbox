"""Sampling an attribute table on a grid: the step, the weighting, and empty slices.

grid_attr_table samples an attribute table on a time grid, and this demo
works through the three choices it takes: the step, what a slice takes
from a note that overlaps it, and what to do with a slice where nothing
sounds. demo_score_workflow.py makes one of each in passing -- a
sixteenth step, the default weighting -- and points here.
"""

import os

import mpt

SCORE = os.path.join(os.path.dirname(__file__), "jmm", "data",
                     "bwv347.musicxml")

#: A held note: the tenor's B, entering on beat 5 and lasting a beat and
#: a half, so that on a beat grid it fills one slice and half of the next.
HELD = 25

#: The columns printed below, a readable subset of the gridded table's.
COLUMNS = ["grid_index", "grid_onset_beats", "note_id", "weight", "pitch",
           "part", "duration_beats"]


def _weights(table, note_id):
    """The weights a note takes across the slices it occupies."""
    rows = table[table["note_id"] == note_id]
    return "  ".join(f"slice {int(i)}: {w:.3f}"
                     for i, w in zip(rows["grid_index"], rows["weight"]))


#: One pitch-class attribute, for the identities below: every note's
#: pitch read as a class, singly.
PITCH_CLASS = dict(column="pitch", name="pitchClass", sigma=0.5,
                   is_per=True, period=12.0)


def main():
    table = mpt.read_score(SCORE)
    end = (table["onset_beats"] + table["duration_beats"]).max()
    print(f"{len(table)} notes over {end:g} beats\n")

    # --- 1. The step ---------------------------------------------------
    # A slice is a step long, and a note belongs to every slice it
    # overlaps, so nothing falls between samples however fine or coarse
    # the grid. The step therefore sets how much of the score each event
    # gathers, not whether an event is seen.
    print("step   points   rows")
    for step in (0.25, 0.5, 1.0, 4.0):
        grid = mpt.grid_attr_table(table, step)
        print(f"{step:5g}   {grid['grid_index'].nunique():6d}   {len(grid):4d}")
    print("\nthe first six rows at a step of one beat:")
    print(mpt.grid_attr_table(table, 1.0)[COLUMNS].head(6).to_string())
    print("""
A step at or below the shortest note value gives one row per note, and
the grid is then a re-indexing of the score by time. A coarser step
gathers several notes of a voice into one event, and the weighting below
then applies.
""")

    # --- 2. The weighting ----------------------------------------------
    # What a slice takes from a note that overlaps it. On a beat grid the
    # three policies differ, because notes and slices no longer
    # coincide.
    print("the tenor's dotted-quarter B across the two slices it occupies:")
    for policy in ("coverage", "presence", "item"):
        grid = mpt.grid_attr_table(table, 1.0, weights=policy)
        print(f"  {policy:9s} {_weights(grid, HELD)}")
    print("\nits two rows under 'coverage':")
    grid = mpt.grid_attr_table(table, 1.0)
    print(grid[grid["note_id"] == HELD][COLUMNS].to_string())
    print("""
  coverage  the fraction of the slice the note fills: how the span is
            filled. It is full in the slice it covers and half in the
            slice it half covers.
  presence  full weight wherever the note appears at all: which notes
            are here, not how much of the slice each holds.
  item      the fraction of the note in the slice, so that a note
            counts once however many slices it spans.
""")

    # Two of the three are re-weightings of the ungridded attribute table: on an
    # attribute constant over the note and read at r = 1, each gives back
    # a density the ungridded table already had.
    def gridded(policy):
        return mpt.build_maet(mpt.pre_maet_from_attr_table(
            mpt.grid_attr_table(table, 1.0, weights=policy),
            attributes=(PITCH_CLASS,), time="beats", chords="separate",
            weights="weight"), verbose=False)

    def ungridded(weights):
        return mpt.build_maet(mpt.pre_maet_from_attr_table(
            table, attributes=(PITCH_CLASS,), time="beats",
            chords="separate", weights=weights), verbose=False)

    print("against the ungridded score, at r = 1:")
    for policy, weights in (("coverage", "duration"), ("item", "ones")):
        similarity = mpt.sim_maet(gridded(policy), ungridded(weights),
                                  verbose=False)
        print(f"  {policy:9s} grid  ~  {weights:8s} notes   {similarity:.3f}")
    print("""
so at r = 1 a grid is a re-weighting. It applies beyond that where the
events have to line up: a structural role needs every event to hold the
same voices (-> demo_score_categoricals.py), and differencing needs the
event index to be a uniform index of time.
""")

    # --- 3. Empty slices -----------------------------------------------
    # A slice with nothing sounding is kept as one row whose note columns
    # are all missing. It holds the place that makes the event index
    # uniform, and downstream it is an event contributing no tuple while
    # keeping its position. Selecting the fermata notes and gridding the
    # selection makes plenty of them.
    fermatas = table[table["fermata"] == True]      # noqa: E712
    grid = mpt.grid_attr_table(fermatas, 1.0)
    empty = int(grid["note_id"].isna().sum())
    print(f"{len(fermatas)} fermata notes over {grid['grid_index'].nunique()}"
          f" slices: {len(grid)} rows, of which {empty} are empty")
    print(grid[COLUMNS].iloc[4:10].to_string())
    print("dropping them is one selection away:",
          len(grid.dropna(subset=["note_id"])), "rows\n")

    # --- 4. Further points ---------------------------------------------
    # 'limits' fixes the span the grid covers, which is how two pieces
    # are put on the same grid. The default runs from 0 to the last
    # note's end.
    eight = mpt.grid_attr_table(table, 1.0, limits=(0.0, 8.0))
    print("first 8 beats only:", eight["grid_index"].nunique(), "slices")
    print(eight[COLUMNS].tail(4).to_string())

    # 'duration' chooses which duration defines occupancy. A score has
    # only the notated one; a MIDI file also has sounding_duration, the
    # notated one with the sustain and sostenuto pedals resolved into it,
    # and gridding over that is what makes a pedalled performance read as
    # held rather than detached.
    print("this table has a sounding duration:",
          any(c.startswith("sounding_duration") for c in table.columns),
          "- it is a score, and a score carries no pedal")

    # The grid steps in one unit but its points have a time in both, so a
    # metrical grid can be read on a clock: slices of a sixteenth, and a
    # sigma in milliseconds. The unit the grid did not step in is
    # interpolated from the attribute table's note samples, so it is exact wherever
    # the tempo is constant and approximate only across a tempo change.
    beat_grid = mpt.grid_attr_table(table, 0.25)
    print("\na beat grid's first four points in each unit:")
    print(beat_grid[["grid_onset_beats", "grid_onset_seconds"]]
          .drop_duplicates().head(4).to_string(index=False))

    # ungrid_attr_table is the inverse: note_id says which row of the
    # source each grid row came from, so keeping the first of each and
    # removing what the grid wrote returns the attribute table it came from. Which
    # columns those are is not a fixed list -- the grid adds weight to a
    # table that had none and overwrites the weight of one that did --
    # which is why this is a toolbox function and not four lines of
    # pandas in the caller.
    back = mpt.ungrid_attr_table(beat_grid)
    print(f"\nungridded: {len(back)} rows, against the {len(table)} read;"
          " identical:", back["pitch"].equals(table["pitch"]))


if __name__ == "__main__":
    main()
