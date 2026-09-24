"""Encoding a categorical column: three ways, and what each asks.

A categorical column reaches a pre-MAET by one of three roles, and this
demo contrasts them on one chorale: what question each asks, and how
they are related. Voice is the example, but the same three carry any
category an attribute table holds -- instrument, articulation, an
experimental condition, a cluster label. demo_score_workflow.py makes
one of these choices in passing and points here.
"""

import os

import numpy as np

import mpt

SCORE = os.path.join(os.path.dirname(__file__), "jmm", "data",
                     "bwv347.musicxml")

#: The attributes every encoding below converts to. Pitch class wraps at
#: the octave and is read strictly; pitch height is read loosely, so that
#: a displacement of register is forgiven and what is left to disagree
#: about is the voicing. demo_score_workflow.py sweeps that width.
ATTRIBUTES = (dict(column="pitch", name="pitchClass", sigma=0.5,
                   is_per=True, period=12.0),
              dict(column="pitch", name="pitchHeight", sigma=24.0),
              dict(column="onset", sigma=0.5))

#: The simplex role builds an attribute of its own, so it carries its
#: own width rather than taking one from the list above.
VOICE = dict(role="simplex", sigma=0.2)


def _chord_events(pm, beat, per_chord):
    """The events of the chord sounding at ``beat``."""
    p_attr, _, specs = mpt.unpack_pre_maet(pm)
    onsets = p_attr[[s["name"] for s in specs].index("onset")][0]
    first = int(np.nonzero(np.isclose(onsets, beat))[0][0])
    return list(range(first, first + per_chord))


def main():
    grid = mpt.grid_attr_table(mpt.read_score(SCORE), 0.25)

    def convert(**kw):
        return mpt.pre_maet_from_attr_table(
            grid, attributes=ATTRIBUTES, time="beats", **kw)

    # --- the three encodings -------------------------------------------
    # The two structural roles realize the level as *where the value
    # sits*: "separate_attributes" gives each level an attribute of its
    # own, and "ordered_multiset" gives each level a fixed position
    # within one attribute. Either way, the binding of a value to its
    # level is carried by the layout, and an event holds the whole
    # chord. What is adjustable afterwards differs: separate attributes
    # carry their own kernel parameters and can be selected one at a
    # time, but each holds a single value, so r = 1 and every level is
    # always read together; one ordered multiset shares a kernel across
    # the levels and takes r > 1, which is how some of the voices are
    # read at a time rather than all of them. "ordered_multiset" is the
    # one used here.
    aware = convert(roles={"part": "ordered_multiset"})

    # The value role, "simplex", realizes the level as *a value of its
    # own*, on its own attribute, so each note becomes its own event and
    # the binding is the product of the two attributes at that event.
    simplex = convert(chords="separate", roles={"part": VOICE})

    # No role at all, at the same one-event-per-note grain: the voice
    # is not part of the encoding.
    agnostic = convert(chords="separate")

    for name, pm in (("voice-aware", aware), ("simplex-voice", simplex),
                     ("voice-agnostic", agnostic)):
        p_attr, _, specs = mpt.unpack_pre_maet(pm)
        print(f"{name:15s} {len(p_attr)} attributes, "
              f"{p_attr[0].shape[1]:4d} events, "
              f"r = {[s['r'] for s in specs]}")
    print()
    for name, pm in (("voice-aware", aware), ("simplex-voice", simplex),
                     ("voice-agnostic", agnostic)):
        mpt.show_pre_maet(pm, max_events=4, title=name)
        print()

    # --- what each asks ------------------------------------------------
    # Two E major chords a beat apart, in quite different voicings:
    # (64, 59, 56, 40) and (71, 68, 64, 52). The same four pitch classes
    # in both, but only the bass keeps its own, and every voice moves in
    # register.
    print("similarity of two voicings of one chord:")
    for name, pm, per_chord in (("voice-aware", aware, 1),
                                ("simplex-voice", simplex, 4),
                                ("voice-agnostic", agnostic, 4)):
        # Compare on the pitch content and the voice encoding, not on
        # when the chord happens: the onset attribute is what located it.
        keep = [s["name"] for s in mpt.unpack_pre_maet(pm)[2]
                if s["name"] != "onset"]
        densities = [
            mpt.build_maet(mpt.select_pre_maet(
                pm, attributes=keep,
                events=_chord_events(pm, beat, per_chord)),
                verbose=False)
            for beat in (7.0, 8.0)]
        print(f"  {name:15s} {mpt.sim_maet(*densities, verbose=False):.3f}")

    print("""
  voice-aware    asks whether *all* voices match, so the re-voicing
                 zeroes the product: a multiplicative AND across voices,
                 and a pitch-class mismatch never relaxes with width.
  simplex-voice  asks what *fraction* of voices match: the bass alone
                 keeps its pitch class, and contributes its quarter.
  voice-agnostic asks whether the *pitch contents* match, and they do --
                 the same multiset of pitch classes, voiced differently.
""")

    # --- the family relation -------------------------------------------
    # The agnostic encoding is the simplex one with its voice attribute
    # removed, which is one selection rather than another conversion.
    # The attributes are selected by index rather than by name, because
    # the two pitch attributes share a name and a name selects the first
    # of them.
    without_voice = mpt.select_pre_maet(simplex, attributes=[0, 1, 2])
    same = all(np.array_equal(a, b) for a, b in
               zip(mpt.unpack_pre_maet(without_voice)[0],
                   mpt.unpack_pre_maet(agnostic)[0]))
    print("dropping the voice attribute recovers the agnostic reading:", same)

    # --- one trap ------------------------------------------------------
    # With bound chords and no role, nothing has fixed exch, so the
    # conversion asks; exch=True reads the chord as an unordered multiset
    # on *every* attribute, and two attributes describing the same notes
    # then pair every value of one with every value of the other -- the
    # soprano's pitch class with the bass's height. Binding a note's
    # attributes to each other needs one event per note, which is what
    # the agnostic encoding above does.
    bound = mpt.pre_maet_from_attr_table(grid, attributes=tuple(
        dict(a, r=1, exch=True) if a["column"] == "pitch" else a
        for a in ATTRIBUTES), time="beats")
    print("bound and role-free:",
          mpt.unpack_pre_maet(bound)[0][0].shape,
          "- a chord per event, but each attribute unordered and so"
          " unpaired with the other")


if __name__ == "__main__":
    main()
