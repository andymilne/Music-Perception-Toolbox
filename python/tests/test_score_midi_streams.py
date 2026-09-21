"""Sustain, sostenuto, pitch bend, and the loudness controllers.

``read_score`` resolves the MIDI controller streams that change a note's
own columns: sustain and sostenuto into ``sounding_duration_*``, pitch
bend into ``pitch``, and channel volume and expression into ``weight``.
The fixtures here are written by hand so that each rule is isolated.

Mirror of MATLAB tests/test_score_midi_streams.m.
"""

import numpy as np
import pytest

from mpt import read_score

TPQ = 480


# --- a minimal Standard MIDI File writer ---------------------------------


def _vlq(n):
    out = bytearray([n & 0x7F])
    n >>= 7
    while n:
        out.insert(0, (n & 0x7F) | 0x80)
        n >>= 7
    return bytes(out)


def _track(events, end_tick=None):
    """events: (tick, bytes) at absolute ticks, in the order given."""
    body, prev = bytearray(), 0
    for tick, payload in events:
        body += _vlq(tick - prev) + payload
        prev = tick
    tail = 0 if end_tick is None else max(0, end_tick - prev)
    body += _vlq(tail) + bytes([0xFF, 0x2F, 0x00])
    return b"MTrk" + len(body).to_bytes(4, "big") + bytes(body)


def _smf(tracks, tpq=TPQ):
    head = b"MThd" + (6).to_bytes(4, "big") + b"\x00\x01" \
        + len(tracks).to_bytes(2, "big") + tpq.to_bytes(2, "big")
    return head + b"".join(tracks)


def _note_on(ch, note, vel=100):
    return bytes([0x90 | ch, note, vel])


def _note_off(ch, note):
    return bytes([0x80 | ch, note, 0])


def _cc(ch, number, value):
    return bytes([0xB0 | ch, number, value])


def _bend(ch, value):
    raw = value + 8192
    return bytes([0xE0 | ch, raw & 0x7F, (raw >> 7) & 0x7F])


def _rpn(ch, msb, lsb, data):
    return [_cc(ch, 101, msb), _cc(ch, 100, lsb), _cc(ch, 6, data)]


def _read(tmp_path, tracks, name="x.mid"):
    path = tmp_path / name
    path.write_bytes(_smf(tracks))
    return read_score(str(path))


# --- sustain -------------------------------------------------------------


def test_sustain_extends_the_sounding_duration_only(tmp_path):
    """The recorded duration stays as it was; the resolved one grows."""
    t = _read(tmp_path, [_track([
        (0, _cc(0, 64, 127)),
        (0, _note_on(0, 60)),
        (TPQ, _note_off(0, 60)),
        (3 * TPQ, _cc(0, 64, 0)),
    ], end_tick=4 * TPQ)])
    assert t["duration_beats"].iloc[0] == pytest.approx(1.0)
    assert t["sounding_duration_beats"].iloc[0] == pytest.approx(3.0)


def test_a_pedal_pressed_after_the_note_off_does_not_hold_it(tmp_path):
    t = _read(tmp_path, [_track([
        (0, _note_on(0, 60)),
        (TPQ, _note_off(0, 60)),
        (2 * TPQ, _cc(0, 64, 127)),
        (3 * TPQ, _cc(0, 64, 0)),
    ], end_tick=4 * TPQ)])
    assert t["sounding_duration_beats"].iloc[0] == pytest.approx(1.0)


def test_a_pedal_never_released_holds_to_the_end(tmp_path):
    t = _read(tmp_path, [_track([
        (0, _cc(0, 64, 127)),
        (0, _note_on(0, 60)),
        (TPQ, _note_off(0, 60)),
    ], end_tick=8 * TPQ)])
    assert t["sounding_duration_beats"].iloc[0] == pytest.approx(8.0)


def test_half_pedalling_counts_as_up(tmp_path):
    """Below 64 is up; the collapse is a documented convention."""
    t = _read(tmp_path, [_track([
        (0, _cc(0, 64, 63)),
        (0, _note_on(0, 60)),
        (TPQ, _note_off(0, 60)),
    ], end_tick=8 * TPQ)])
    assert t["sounding_duration_beats"].iloc[0] == pytest.approx(1.0)


# --- the re-strike rule --------------------------------------------------


def test_a_restrike_on_the_same_channel_damps_the_tail(tmp_path):
    t = _read(tmp_path, [_track([
        (0, _cc(0, 64, 127)),
        (0, _note_on(0, 60)),
        (TPQ, _note_off(0, 60)),
        (2 * TPQ, _note_on(0, 60)),
        (3 * TPQ, _note_off(0, 60)),
        (4 * TPQ, _cc(0, 64, 0)),
    ], end_tick=4 * TPQ)])
    first = t.iloc[0]
    assert first["sounding_duration_beats"] == pytest.approx(2.0)


def test_a_restrike_on_another_channel_does_not(tmp_path):
    """Two channels may be two instruments, so the pitches overlap."""
    t = _read(tmp_path, [_track([
        (0, _cc(0, 64, 127)),
        (0, _cc(1, 64, 127)),
        (0, _note_on(0, 60)),
        (TPQ, _note_off(0, 60)),
        (2 * TPQ, _note_on(1, 60)),
        (3 * TPQ, _note_off(1, 60)),
        (4 * TPQ, _cc(0, 64, 0)),
        (4 * TPQ, _cc(1, 64, 0)),
    ], end_tick=4 * TPQ)])
    on_ch0 = t[t["channel"] == 1].iloc[0]
    assert on_ch0["sounding_duration_beats"] == pytest.approx(4.0)


# --- sostenuto -----------------------------------------------------------


def test_sostenuto_holds_only_what_was_down_when_it_was_pressed(tmp_path):
    t = _read(tmp_path, [_track([
        (0, _note_on(0, 60)),
        (TPQ // 2, _cc(0, 66, 127)),          # pressed while 60 is down
        (TPQ, _note_off(0, 60)),
        (TPQ, _note_on(0, 64)),               # struck after; not held
        (2 * TPQ, _note_off(0, 64)),
        (4 * TPQ, _cc(0, 66, 0)),
    ], end_tick=4 * TPQ)])
    held = t[t["note_number"] == 60].iloc[0]
    later = t[t["note_number"] == 64].iloc[0]
    assert held["sounding_duration_beats"] == pytest.approx(4.0)
    assert later["sounding_duration_beats"] == pytest.approx(1.0)


# --- pitch bend ----------------------------------------------------------


def test_bend_lands_in_pitch_and_leaves_note_number_alone(tmp_path):
    """A quarter-tone up at the 2-semitone default."""
    t = _read(tmp_path, [_track(
        _rpn_events(0, 0, 0, 2) + [
            (0, _bend(0, 2048)),
            (0, _note_on(0, 60)),
            (TPQ, _note_off(0, 60)),
        ], end_tick=TPQ)])
    assert t["pitch"].iloc[0] == pytest.approx(60.5)
    assert t["note_number"].iloc[0] == 60


def test_rpn_zero_sets_the_range(tmp_path):
    t = _read(tmp_path, [_track(
        _rpn_events(0, 0, 0, 12) + [
            (0, _bend(0, 4096)),
            (0, _note_on(0, 60)),
            (TPQ, _note_off(0, 60)),
        ], end_tick=TPQ)])
    assert t["pitch"].iloc[0] == pytest.approx(66.0)


def test_an_mpe_zone_gives_member_channels_the_48_semitone_default(tmp_path):
    """An MPE Configuration Message on channel 1 opens a lower zone."""
    t = _read(tmp_path, [_track(
        _rpn_events(0, 0, 6, 3) + [
            (0, _bend(1, round(8192 / 48))),  # one semitone up
            (0, _note_on(1, 60)),
            (TPQ, _note_off(1, 60)),
        ], end_tick=TPQ)])
    assert t["pitch"].iloc[0] == pytest.approx(61.0, abs=0.01)


def test_bend_without_a_declared_range_warns(tmp_path):
    with pytest.warns(UserWarning, match="pitch-bend range"):
        _read(tmp_path, [_track([
            (0, _bend(0, 2048)),
            (0, _note_on(0, 60)),
            (TPQ, _note_off(0, 60)),
        ], end_tick=TPQ)])


def test_a_bend_before_the_first_note_on_that_channel_still_applies(tmp_path):
    """The first note of a channel may be preceded by nothing, so a bend
    sent just after it is read, provided no other note intervenes."""
    with pytest.warns(UserWarning):
        t = _read(tmp_path, [_track([
            (0, _note_on(0, 60)),
            (1, _bend(0, 4096)),
            (TPQ, _note_off(0, 60)),
        ], end_tick=TPQ)])
    assert t["pitch"].iloc[0] == pytest.approx(61.0)


# --- loudness controllers ------------------------------------------------


def test_volume_and_expression_fold_into_weight(tmp_path):
    t = _read(tmp_path, [_track([
        (0, _cc(0, 7, 64)),
        (0, _cc(0, 11, 127)),
        (0, _note_on(0, 60, 100)),
        (TPQ, _note_off(0, 60)),
    ], end_tick=TPQ)])
    expected = (100 / 127) * (64 / 127) ** 2
    assert t["weight"].iloc[0] == pytest.approx(expected)
    assert t["velocity"].iloc[0] == 100


def test_weight_is_velocity_alone_where_no_controller_is_sent(tmp_path):
    t = _read(tmp_path, [_track([
        (0, _note_on(0, 60, 64)),
        (TPQ, _note_off(0, 60)),
    ], end_tick=TPQ)])
    assert t["weight"].iloc[0] == pytest.approx(64 / 127)


# --- helper --------------------------------------------------------------


def _rpn_events(ch, msb, lsb, data, tick=0):
    return [(tick, payload) for payload in _rpn(ch, msb, lsb, data)]


# --- program change ------------------------------------------------------


def test_the_program_in_force_at_the_onset(tmp_path):
    """Program change selects the instrument sound on a channel; the
    column carries whichever is in force when the note starts."""
    t = _read(tmp_path, [_track([
        (0, bytes([0xC0, 40])),              # violin, channel 1
        (0, _note_on(0, 60)),
        (TPQ, _note_off(0, 60)),
        (TPQ, bytes([0xC0, 73])),            # flute from here
        (2 * TPQ, _note_on(0, 62)),
        (3 * TPQ, _note_off(0, 62)),
    ], end_tick=3 * TPQ)])
    assert t["program"].tolist() == [40, 73]


def test_no_program_change_reads_as_zero(tmp_path):
    t = _read(tmp_path, [_track([
        (0, _note_on(0, 60)), (TPQ, _note_off(0, 60)),
    ], end_tick=TPQ)])
    assert t["program"].tolist() == [0]


def test_program_is_per_channel(tmp_path):
    t = _read(tmp_path, [_track([
        (0, bytes([0xC0, 40])),
        (0, bytes([0xC1, 73])),
        (0, _note_on(0, 60)),
        (0, _note_on(1, 72)),
        (TPQ, _note_off(0, 60)),
        (TPQ, _note_off(1, 72)),
    ], end_tick=TPQ)])
    assert dict(zip(t["channel"], t["program"])) == {1: 40, 2: 73}
