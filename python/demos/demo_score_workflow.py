"""From a score (MusicXML, MIDI) to a MAET analysis.

The spine of the demo_score* family. It reads a score into an attribute
table, looks at that table, samples it on a grid, encodes a categorical
column, builds the density, and runs an analysis on it. Where a step has
more to it than the one choice made here, a comment names the demo that
goes further:

  demo_score_grid.py          choosing the grid step and the weighting
  demo_score_categoricals.py  the three ways to encode a category

The functions that get a score to a pre-MAET:

  score file (MIDI or MusicXML)
    read_score                -> attribute table, sampled per note
  attribute table (attributes across the columns)
    grid_attr_table           -> attribute table, sampled on a time grid
    ungrid_attr_table         -> attribute table, the time grid undone
    pre_maet_from_attr_table  -> pre-MAET
  pre-MAET (rows are attributes, columns are events; the values
  (p_attr), the weights (w_attr), and the per-attribute parameters
  (specs) are its three parts, and show_pre_maet renders the specs as a
  leading column)
    unpack_pre_maet           -> its three parts (p_attr, w_attr, specs)
    pack_pre_maet             -> a pre-MAET from those parts
    show_pre_maet                display it
    write_pre_maet               write it as markdown, LaTeX, or CSV
    read_pre_maet             -> a pre-MAET read back from that, which
                                 stands in for every step above

Rows of attribute tables are selected with pandas' own indexing rather
than a toolbox function. What happens to a pre-MAET next --
select_pre_maet and the other preprocessing operations, build_maet, and
the measures -- is a separate family, and demo_preprocessing.py covers
it.

The conversion from attribute table to pre-MAET transposes, and
regroups. An attribute table carries its attributes across the columns,
as any data table does; a pre-MAET carries attributes down the rows and
events across the columns, which is the layout of the article's pre-MAET
table and of show_pre_maet's output, so a table's column becomes a
pre-MAET's row and each attribute's values are a (K_a, N) array. This
horizontal/wide format is preferred because it corresponds to that used
in musical scores and DAWs.

The events are not the attribute table's rows. The conversion to
pre-MAET gathers rows into events -- a chord bound into one event, or a
grid point holding its voices -- so N counts events and K_a counts the
values one event carries on that attribute. Here 1088 attribute table
rows become 272 pre-MAET events of four pitches each.

Only three stages are needed to get from a score (MIDI or MusicXML) to a
MAET density: read_score, pre_maet_from_attr_table, build_maet.
Everything between them is optional. This demo takes read_score,
grid_attr_table, pre_maet_from_attr_table, select_pre_maet, build_maet,
sim_maet: it grids because the structural role of step 4 needs every
event to hold the same voices, and it selects because the question is
about two chords out of the 272.
"""

import os

import numpy as np

import mpt

SCORE = os.path.join(os.path.dirname(__file__), "jmm", "data",
                     "bwv347.musicxml")


def main():
    # --- 1. Read -------------------------------------------------------
    # One row per sounding note. A column is present only where the
    # source carries it: this is MusicXML, so it has voice, staff,
    # fermata, and the articulations. A MIDI file would instead have
    # channel, program, note_number, weight, and sounding_duration_*
    # (the last two being the loudness controllers and the pedals
    # resolved into the note's own columns). See read_score's docstring.
    table = mpt.read_score(SCORE)
    print(f"{len(table)} notes from a {table.attrs['source']} score;"
          " the first five rows:")
    print(table.head().to_string(), "\n")

    # --- 2. Look and select --------------------------------------------
    # The return is a pandas DataFrame, inspected and filtered with
    # pandas.
    print("parts:", list(table["part"].cat.categories))
    print("pitch range:", table["pitch"].min(), "to", table["pitch"].max())
    print("notes under a fermata:", int(table["fermata"].sum()))

    # A selection carries through everything below, and is written as a
    # pandas row selection. Here we keep the whole chorale.
    print('the first five notes of table[table["part"] != "Bass"]:')
    print(table[table["part"] != "Bass"].head().to_string(), "\n")

    # --- 3. Sample on a grid -------------------------------------------
    # A grid makes the event index a uniform index of time, and gives
    # every event the same voices, which the structural encoding of
    # step 4 needs. A sixteenth is the shortest note value here.
    # -> demo_score_grid.py for the step and the weight policies.
    grid = mpt.grid_attr_table(table, 0.25)
    print(f"gridded: {grid['grid_index'].nunique()} points, {len(grid)}"
          " rows; the first five:")
    print(grid[["grid_index", "grid_onset_beats", "note_id", "weight",
                "pitch", "part", "duration_beats"]].head().to_string(), "\n")

    # --- 4. Convert to a pre-MAET --------------------------------------
    # An attribute is a column of the table read under a set of
    # parameters, so each entry names both. Three things worth knowing
    # happen here.
    #
    # 'pitch' is listed twice, so the same values become two attributes
    # under different kernels: one reads pitch class, wrapping at the
    # octave, and the other pitch height, which does not. Their product
    # is Shepard's helix, so how much two chords agree can be asked
    # separately of their pitch classes and of their registers.
    #
    # A score fixes what the values are and not how tolerant a match is,
    # nor how many of an event's values a tuple takes, so sigma, r, and
    # exch are the analyst's and the conversion asks for them. It fills
    # in only what follows from the data or from another argument here:
    # r and exch under a structural role, and 'read as written' for rel
    # and is_per.
    #
    # The part is given a role, which says how a categorical column
    # reaches the pre-MAET. There are three:
    #   'ordered_multiset'    structural: the level becomes a position
    #                         within one attribute, so the four voices
    #                         occupy four slots and matching is voice by
    #                         voice. Used here, and it fixes r = 4 and
    #                         exch = False, which is why neither is given.
    #   'separate_attributes' structural: the level becomes an attribute
    #                         of its own, one per voice.
    #   'simplex'             a value: the level becomes the coordinates
    #                         of a simplex vertex on an attribute of its
    #                         own, so two chords can match on some voices
    #                         and not others.
    # -> demo_score_categoricals.py for what each asks of the same music.
    #
    # The arguments after the attributes carry the rest of the reading.
    # time="beats" puts the onset attribute's values in quarter notes
    # rather than in the default seconds, matching the unit the grid of
    # step 3 was built over.
    pm = mpt.pre_maet_from_attr_table(grid, attributes=(
        dict(column="pitch", name="pitchClass",
             sigma=0.5, is_per=True, period=12.0),
        dict(column="pitch", name="pitchHeight", sigma=8.0),
        dict(column="onset", sigma=0.5)),
        time="beats", roles={"part": "ordered_multiset"})
    p_attr, w_attr, specs = mpt.unpack_pre_maet(pm)
    mpt.show_pre_maet(pm, max_events=4, title="the pre-MAET, first events")
    print()

    # --- 5. Select what the question is about --------------------------
    # select_pre_maet keeps a selection of a pre-MAET's attributes and
    # of its events, in the order given, and returns a pre-MAET like any
    # other. The question below is about two chords, compared on their
    # pitches, so each chord becomes a one-event pre-MAET on the two
    # pitch attributes; onset located them and is not compared on. The
    # four values are in S, A, T, B order, the slots the ordered_multiset
    # role gave them.
    #
    # The two chords are the final chords of the first two cadences,
    # which fall on beats 7 and 15. The onset attribute is searched for
    # those two beats to get their event indices; on this grid of
    # sixteenths they are not events 7 and 15.
    full = mpt.pack_pre_maet(p_attr, w_attr, specs)
    onsets = p_attr[2][0]
    events = [int(np.nonzero(np.isclose(onsets, b))[0][0])
              for b in (7.0, 15.0)]
    cadence = [mpt.select_pre_maet(
        full, attributes=["pitchClass", "pitchHeight"], events=[n])
        for n in events]
    for k, pm_k in enumerate(cadence, start=1):
        mpt.show_pre_maet(pm_k, title=f"cadence {k} tonic")
        print()

    # --- 6. Ask something ----------------------------------------------
    # The two tonic chords are the same four pitch classes, differing
    # only in the octave of the bass. Whether that counts as the same
    # chord is what the pitch-height width decides, the two attributes
    # being separate. build_maet's sigma overrides the specs', so the
    # sweep needs no rebuild.
    print("similarity of the two, against the pitch-height width:")
    for sigma_height in (1.0, 4.0, 16.0, 64.0):
        densities = [mpt.build_maet(pm_k, sigma=[0.5, sigma_height],
                                    verbose=False) for pm_k in cadence]
        similarity = mpt.sim_maet(*densities, verbose=False)
        print(f"  sigma_pitchHeight = {sigma_height:5.1f} semitones"
              f"   ->  {similarity:.3f}")
    print("\nnarrow: two different chords, the bass octave counting."
          "\nwide:   one chord, the octave forgiven and the pitch classes"
          " agreeing.")


if __name__ == "__main__":
    main()
