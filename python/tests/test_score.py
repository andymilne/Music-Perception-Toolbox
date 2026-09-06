"""``read_score`` and ``events_from_score`` on the shared fixtures in
tests/data (a format-1 MIDI file and a MusicXML score, plain and
compressed). Mirror of MATLAB tests/test_score.m; both suites assert
the same note tables.
"""
import os

import numpy as np
import pytest

import mpt
from mpt import build_exp_tens, cos_sim_exp_tens, events_from_score, read_score

DATA = os.path.join(os.path.dirname(__file__), "data")


# The fixtures (see tools/gen_scores.py in the handover): a 3/4 MIDI file
# at 120 bpm switching to 60 bpm at beat 4, with a melody track and a
# chord track that uses running status and leaves two notes open; a
# MusicXML score with a soprano (tie, rest, grace note, dynamics) and a
# piano part (chord, backup into a second voice, tempo change).
MIDI_TABLE = {
    "onset_beats":      [0, 0, 0, 0, 1, 2, 2, 2, 4],
    "onset_seconds":    [0, 0, 0, 0, 0.5, 1, 1, 1, 2],
    "duration_beats":   [1, 2, 2, 2, 0.5, 2, 2, 2, 1],
    "duration_seconds": [0.5, 1, 1, 1, 0.25, 1, 1, 1, 1],
    "pitch":            [60, 48, 52, 55, 64, 67, 53, 57, 72],
    "velocity":         [96, 100, 100, 100, 80, 112, 100, 100, 64],
    "part":             [1, 2, 2, 2, 1, 1, 2, 2, 1],
    "channel":          [1, 2, 2, 2, 1, 1, 2, 2, 1],
    "measure":          [1, 1, 1, 1, 1, 1, 1, 1, 2],
}
XML_TABLE = {
    "onset_beats":      [0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 4],
    "onset_seconds":    [0, 0, 0, 0, 0, 0.5, 0.5, 1, 1, 1.5, 2.5],
    "duration_beats":   [1, 1, 2, 2, 2, 0.5, 1, 2, 1, 3, 2],
    "duration_seconds": [0.5, 0.5, 1, 1, 1, 0.25, 0.5, 1.5, 0.5, 3, 2],
    "pitch":            [60, 36, 48, 52, 55, 64, 43, 67, 41, 53, 70],
    "velocity":         [90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 54],
    "part":             [1, 2, 2, 2, 2, 1, 2, 1, 2, 2, 1],
    "channel":          [1, 2, 1, 1, 1, 1, 2, 1, 2, 1, 1],
    "measure":          [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2],
}


def _check_table(table, expect):
    for key, vals in expect.items():
        np.testing.assert_allclose(table[key], vals, atol=1e-12,
                                   err_msg=key)


def test_midi_table():
    t = read_score(os.path.join(DATA, "score_small.mid"))
    assert t["source"] == "midi"
    assert t["part_names"] == ["Melody", "Chords"]
    _check_table(t, MIDI_TABLE)


@pytest.mark.parametrize("name", ["score_small.musicxml", "score_small.mxl"])
def test_musicxml_table(name):
    t = read_score(os.path.join(DATA, name))
    assert t["source"] == "musicxml"
    assert t["part_names"] == ["Soprano", "Piano"]
    _check_table(t, XML_TABLE)


def test_unknown_extension():
    with pytest.raises(ValueError, match="extension"):
        read_score("score.abc")


def test_events_bind_chords_by_default():
    p, w, specs = events_from_score(os.path.join(DATA, "score_small.mid"))
    assert [s["name"] for s in specs] == ["pitch", "onset"]
    assert p[0].shape == (4, 4) and p[1].shape == (1, 4)
    np.testing.assert_array_equal(p[0][:, 0], [60, 48, 52, 55])
    assert np.isnan(p[0][1:, 1]).all()
    np.testing.assert_allclose(p[1][0], [0, 0.5, 1, 2])
    np.testing.assert_allclose(w[0][:, 0], np.array([96, 100, 100, 100]) / 127)
    assert (w[0][1:, 1] == 0).all()
    np.testing.assert_array_equal(w[1], np.ones((1, 4)))


def test_events_options():
    p, w, specs = events_from_score(
        os.path.join(DATA, "score_small.musicxml"),
        attributes=("pitch", "onset", "duration"), pitch="cents",
        time="beats", weights="ones", chords="separate", parts=2)
    assert w is None
    assert [s["name"] for s in specs] == ["pitch", "onset", "duration"]
    assert all(m.shape == (1, 7) for m in p)
    np.testing.assert_allclose(p[0][0], [3600, 4800, 5200, 5500, 4300, 4100, 5300])
    np.testing.assert_allclose(p[1][0], [0, 0, 0, 0, 1, 2, 3])
    np.testing.assert_allclose(p[2][0], [1, 2, 2, 2, 1, 1, 3])
    p2, w2, _ = events_from_score(os.path.join(DATA, "score_small.musicxml"),
                                  weights="duration", chords="separate",
                                  time="beats")
    np.testing.assert_allclose(w2[0][0], XML_TABLE["duration_beats"])


def test_events_from_a_table_and_chord_tolerance():
    t = read_score(os.path.join(DATA, "score_small.mid"))
    p, _, _ = events_from_score(t, chords="bind", chord_tolerance=0.6,
                                time="seconds", weights="ones")
    # 0.5 s (E4) binds to the notes at 0 s; 2 s stays alone
    assert p[0].shape[1] == 3 and p[0].shape[0] == 5


def test_bad_arguments():
    path = os.path.join(DATA, "score_small.mid")
    with pytest.raises(ValueError, match="Unknown attribute"):
        events_from_score(path, attributes=("pitch", "colour"))
    with pytest.raises(ValueError, match="time must"):
        events_from_score(path, time="ticks")
    with pytest.raises(ValueError, match="chords must"):
        events_from_score(path, chords="merge")


def test_carrier_feeds_the_pipeline():
    """A pitch-class dyad density of the chord track, and its similarity
    with itself, without any hand-built carrier."""
    p, w, specs = events_from_score(os.path.join(DATA, "score_small.mid"),
                                    parts=2)
    specs[0]["r"] = 2
    d = build_exp_tens(p, w, specs=specs, sigma=[1.0, 0.2],
                       is_per=[True, False], period=[12.0, 0.0],
                       verbose=False)
    assert cos_sim_exp_tens(d, d, verbose=False) == pytest.approx(1.0)
