"""The event table returned by ``read_score``.

``read_score`` returns a pandas ``DataFrame``, one row per sounding note,
so a score is inspected and filtered with pandas itself and no toolbox
function is needed to read it. ``pre_maet_from_score`` accepts either the
path or the table, so this reader is called directly only when the table
is wanted.

The MATLAB sibling is demos/demo_scoreTable.m.
"""

import os

import mpt

SCORE = os.path.join(os.path.dirname(__file__), "jmm", "data",
                     "bwv347.musicxml")


def main():
    t = mpt.read_score(SCORE)

    # --- What comes back ----------------------------------------------
    print(f"{len(t)} notes from a {t.attrs['source']} score\n")
    print(t.dtypes.to_string(), "\n")
    print(t.head(4).to_string(index=False), "\n")

    # part is categorical and its categories are the part names, so the
    # names are in the column rather than in a field beside it.
    print("parts:", list(t["part"].cat.categories))

    # A MusicXML score carries voice and fermata; a MIDI file would carry
    # channel instead, and neither stands in for the other.
    print("columns this source carries:", list(t.columns), "\n")

    # --- Selection is pandas ------------------------------------------
    soprano = t[t["part"] == "Soprano"]
    print(f"soprano notes: {len(soprano)}, "
          f"range {soprano['pitch'].min():.0f}-{soprano['pitch'].max():.0f}")

    held = t[t["fermata"]]
    print(f"notes under a fermata: {len(held)}, at beats "
          f"{sorted(held['onset_beats'].unique().tolist())}\n")

    # Anything pandas does, the table does: here, the mean duration of
    # each part, which no toolbox function has to provide.
    print("mean duration in quarter notes, by part:")
    print(t.groupby("part", observed=True)["duration_beats"]
           .mean().round(3).to_string(), "\n")

    # --- Sampling on a grid -------------------------------------------
    # Analysis 1.1 of the JMM article reads this chorale on a
    # sixteenth-note grid. grid_events is table to table, so the result
    # is selected from the same way; a held note replicates across the
    # points it covers, and a point with nothing sounding would be a row
    # whose note columns are all missing (this chorale has no rests).
    g = mpt.grid_events(t, 0.25)
    print("--- on a sixteenth-note grid ---")
    print("grid points:", g["grid_index"].nunique(),
          " rows:", len(g),
          " empty points:", int(g["note_id"].isna().sum()))
    print("the first two points:")
    print(g.head(8)[["grid_index", "grid_onset_beats", "pitch", "part",
                     "weight"]].to_string(index=False), "\n")

    # --- On to a pre-MAET ---------------------------------------------
    # The table goes to pre_maet_from_score in place of the path, so any
    # selection made above carries through.
    pm = mpt.pre_maet_from_score(soprano, attributes=("pitch", "onset"),
                                 time="beats", chords="separate")
    p, w, specs = mpt.unpack_pre_maet(pm)
    print("pre-MAET from the soprano alone:",
          f"{p[0].shape[1]} events,",
          f"attributes {[s['name'] for s in specs]}")


if __name__ == "__main__":
    main()
