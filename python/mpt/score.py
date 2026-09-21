"""Pre-MAET events from symbolic scores (MIDI and MusicXML).

Two functions. :func:`read_score` parses a Standard MIDI File (format 0
or 1) or a MusicXML file (``.musicxml``, ``.xml``, or compressed
``.mxl``) into an *event table* -- a :class:`pandas.DataFrame` with one
row per sounding note, carrying its onset and duration in beats and in
seconds, its MIDI pitch, its velocity, and its part, plus whatever else
its source records. :func:`pre_maet_from_score` turns an event table (or
a path) into
the ``(p_attr, w_attr, specs)`` that :func:`build_maet` and the
pre-MAET preprocessors consume, choosing the attributes, their units,
the weights, and whether simultaneous notes are bound into one
multi-value event.

Both parsers are self-contained (no third-party dependency) and mirror
``readScore`` / ``preMaetFromScore`` in MATLAB, which read the same files
to the same table.

Conventions
-----------
* A *beat* is a quarter note (MIDI ticks per quarter note; MusicXML
  ``divisions`` per quarter note), whatever the time signature.
* Seconds follow the tempo map: every MIDI ``set_tempo`` event, every
  MusicXML ``<sound tempo>`` or metronome direction (120 quarter notes
  per minute where a file gives none).
* MIDI pitch is the note number (A4 = 69). MusicXML pitches are converted
  from step, alter, and octave; unpitched notes and rests are not notes.
* A MIDI note-on with velocity 0 is a note-off. Notes left open at the
  end of a track are closed there.
* MusicXML tied notes are merged into one note (the start carries the
  summed duration); grace notes carry no duration and are skipped;
  ``<chord/>`` notes share the preceding note's onset.
* Velocity is the MIDI velocity (0–127); for MusicXML it is taken from
  the note's ``dynamics`` attribute (a percentage of forte, forte being
  90) and is 90 where absent.
"""
from __future__ import annotations

import io
import os
import struct
import warnings
from bisect import bisect_right
from collections import defaultdict
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from ._tensor.premaet import pre_maet
from ._tensor.preprocessing import flat_specs, simplex_vertices
from ._tensor.transform import _convert_scale

__all__ = ["read_score", "pre_maet_from_score"]

# The columns each source produces, in order. A column is present only
# where its source carries the information, so channel and voice are
# never the same column and never stand in for one another.
_MIDI_COLUMNS = ("onset_beats", "onset_seconds", "duration_beats",
                 "duration_seconds", "sounding_duration_beats",
                 "sounding_duration_seconds", "pitch", "note_number",
                 "velocity", "weight", "part", "channel", "program",
                 "measure")
_XML_COLUMNS = ("onset_beats", "onset_seconds", "duration_beats",
                "duration_seconds", "pitch", "velocity", "part", "voice",
                "staff", "measure", "fermata", "staccato", "accent",
                "tenuto")
_INT_COLUMNS = frozenset(("note_number", "channel", "voice", "staff",
                          "measure", "program"))
_BOOL_COLUMNS = frozenset(("fermata", "staccato", "accent", "tenuto"))

#: Articulations read from a MusicXML note's <notations><articulations>.
#: They are not mutually exclusive -- a note may be both staccato and
#: accented -- so each is its own two-level column rather than one column
#: with a level per marking.
_XML_ARTICULATIONS = ("staccato", "accent", "tenuto")
_STEP_TO_SEMITONE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


# ===================================================================
#  read_score
# ===================================================================

def read_score(path):
    """Parse a MIDI or MusicXML file into an event table.

    Parameters
    ----------
    path : str
        A ``.mid`` / ``.midi`` file (format 0 or 1), a ``.musicxml`` /
        ``.xml`` file (partwise or timewise), or a compressed ``.mxl``.

    Returns
    -------
    DataFrame
        One row per sounding note, sorted by onset, then part, then
        pitch. Columns common to both sources: ``onset_beats``,
        ``onset_seconds``, ``duration_beats``, ``duration_seconds``,
        ``pitch`` (in MIDI note numbers, and not an integer where the
        file bends or writes a microtone), ``velocity`` (0-127), ``part``
        (categorical, the part names as its categories), and ``measure``
        (1-based).

        A MIDI file adds ``channel`` (1-16), ``note_number`` (the note
        number as recorded, which is what note identity rests on),
        ``program`` (the program change in force on that channel at the
        note's onset, 0 where none was sent, which selects the
        instrument sound), ``weight``, and ``sounding_duration_beats`` /
        ``sounding_duration_seconds``.

        A MusicXML score adds ``voice`` (1-based within its part),
        ``staff`` (1-based; a part written on more than one staff, as a
        keyboard part is, says which each note belongs to), ``fermata``,
        and the articulations ``staccato``, ``accent``, and ``tenuto``.
        The boolean marks are not mutually exclusive -- a note may be
        both staccato and accented -- so each is its own column, and a
        merged tied note carries a mark any of its segments carries.

        A column is present only where the source carries it, so
        ``channel`` and ``voice`` are never the same column and never
        stand in for one another.

        ``df.attrs['source']`` is ``'midi'`` or ``'musicxml'``.

    Warns
    -----
    UserWarning
        Where a MIDI file bends pitch but declares no bend range.

    Notes
    -----
    A beat is a quarter note, whatever the time signature. Seconds
    follow the file's tempo map, at 120 quarter notes per minute where a
    file gives none. A MusicXML tie is merged into one note, a grace note
    is skipped, and a rest or unpitched note is not a note.

    **MIDI controller streams.** The three that change a note's own
    columns are resolved at read; every other controller is out of scope,
    since a value sampled at the onset would misrepresent a ramp inside a
    held note.

    *Sustain and sostenuto* give ``sounding_duration_*`` beside the
    recorded ``duration_*``, so either can feed an analysis. A note whose
    note-off falls while sustain (CC64, at or above 64) is down sounds
    until the pedal comes up, or to the end of the file where it never
    does; sostenuto (CC66) holds only what was already down when it was
    pressed; and the same note number struck again on the same channel
    damps what is left of the first, while the same pitch on another
    channel does not, since two channels may be two instruments. Pedal
    state and the damping rule are both per channel, and a channel
    belongs to the file rather than to a track.

    *Pitch bend* is resolved into ``pitch``, which is therefore not an
    integer: bend is how microtonal music is carried in MIDI, in the
    one-channel-per-note idiom and under MPE alike. The range is 2
    semitones unless RPN 0 sets it, or an MPE Configuration Message
    (RPN 6 on channel 1 or 16) opens a zone, whose member channels take
    the 48-semitone MPE default. A note takes the last bend at or before
    its onset tick, so a bend sent immediately before a note-on, or at
    the same tick in either file order, tunes it; where a channel has no
    earlier bend at all, the first bend after that note is used provided
    no further note-on intervenes. Under MPE the bend continues through
    the note as a slide, and the resolved value is the pitch at onset.

    *Channel volume (CC7) and expression (CC11)* fold into ``weight``::

        weight = (velocity / 127) * (cc7 / 127) ** 2 * (cc11 / 127) ** 2

    so that a passage played down by expression is not weighted as though
    it were at full strength. ``velocity`` keeps the value as recorded.

    The two factors rest on different grounds, and the formula is a
    hybrid. The squares are MIDI's specified default response for both
    controllers, an attenuation of ``40 * log10(cc / 127)`` dB; the two
    are cascaded gain stages, so in dB they add and in amplitude they
    multiply. The velocity factor is linear because MIDI specifies no
    velocity-to-amplitude curve -- it is instrument-dependent -- and
    because taking it linearly makes ``weight`` equal to the toolbox's
    ``weights='velocity'`` weighting on any file that sends no
    controller, which is nearly all of them. So ``weight`` is that
    weighting corrected by the channel's specified gain, and not an
    estimate of sounding amplitude. A different velocity curve is one
    transformation of the column away.
    """
    ext = os.path.splitext(str(path))[1].lower()
    if ext in (".mid", ".midi", ".smf", ".kar"):
        with open(path, "rb") as fh:
            notes = _parse_midi(fh.read())
    elif ext == ".mxl":
        notes = _parse_musicxml(_unzip_mxl(path))
    elif ext in (".musicxml", ".xml"):
        with open(path, "rb") as fh:
            notes = _parse_musicxml(fh.read())
    else:
        raise ValueError(
            f"Unrecognised score file extension {ext!r}: expected .mid, "
            f".midi, .musicxml, .xml, or .mxl.")
    return _finish_table(notes)


def _finish_table(notes):
    columns = notes["columns"]
    rows = notes["rows"]
    index = {name: i for i, name in enumerate(columns)}
    if rows:
        key = (index["onset_beats"], index["part"], index["pitch"])
        ordered = sorted(rows, key=lambda r: (r[key[0]], r[key[1]], r[key[2]]))
        cols = list(zip(*ordered))
    else:
        cols = [() for _ in columns]
    raw = {name: np.asarray(col, dtype=np.float64)
           for name, col in zip(columns, cols)}
    part_names = _part_categories(notes["part_names"],
                                  raw["part"].astype(np.intp))

    data = {}
    for name in columns:
        if name == "part":
            data[name] = pd.Categorical.from_codes(
                raw[name].astype(np.intp) - 1, categories=part_names)
        elif name in _INT_COLUMNS:
            data[name] = raw[name].astype(np.intp)
        elif name in _BOOL_COLUMNS:
            data[name] = raw[name].astype(bool)
        else:
            data[name] = raw[name]

    table = pd.DataFrame(data, columns=list(columns))
    table.attrs["source"] = notes["source"]
    return table


def _part_categories(names, part_index):
    """Part names as a unique, non-empty category list, one per part.

    The parser supplies one name per part, but a score may leave a part
    unnamed or repeat a name, and categories have to be distinct.
    """
    n_parts = max(int(part_index.max()) if part_index.size else 0,
                  len(names))
    out, seen = [], {}
    for i in range(n_parts):
        name = str(names[i]).strip() if i < len(names) else ""
        if not name:
            name = f"Part {i + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name} ({seen[name]})"
        else:
            seen[name] = 1
        out.append(name)
    return out


# -------------------------------------------------------------------
#  MIDI
# -------------------------------------------------------------------

def _read_varlen(data, pos):
    value = 0
    while True:
        b = data[pos]
        pos += 1
        value = (value << 7) | (b & 0x7F)
        if not b & 0x80:
            return value, pos


def _parse_midi(data):
    if data[:4] != b"MThd":
        raise ValueError("Not a Standard MIDI File (missing MThd header).")
    hlen, fmt, ntrk, division = struct.unpack(">IHHH", data[4:14])
    if division & 0x8000:
        raise ValueError(
            "SMPTE time division is not supported; use a file with "
            "ticks-per-quarter-note timing.")
    tpq = float(division)
    if fmt not in (0, 1):
        raise ValueError(f"MIDI format {fmt} is not supported (0 or 1).")
    pos = 8 + hlen
    tracks = []
    for _ in range(ntrk):
        if data[pos:pos + 4] != b"MTrk":
            raise ValueError("Malformed MIDI file (missing MTrk chunk).")
        length = struct.unpack(">I", data[pos + 4:pos + 8])[0]
        tracks.append(data[pos + 8:pos + 8 + length])
        pos += 8 + length

    # Pass 1: tempo and time-signature maps from every track (format 1
    # keeps them in track 0, but a lenient reader takes them anywhere).
    tempo_map = []          # (tick, microseconds per quarter)
    timesig_map = []        # (tick, quarter notes per bar)
    track_events = []       # per track: list of (tick, status, d1, d2)
    track_names = []
    ctrl_events = []        # (tick, channel, kind, d1, d2), all tracks
    file_end = 0
    for tdata in tracks:
        events, ctrl, tempos, sigs, name, end_tick = _midi_track_events(tdata)
        track_events.append((events, end_tick))
        tempo_map.extend(tempos)
        timesig_map.extend(sigs)
        track_names.append(name)
        ctrl_events.extend(ctrl)
        file_end = max(file_end, end_tick)
    # A channel is a property of the file, not of a track, so controller
    # state is gathered across tracks and read per channel.
    ctrl_events.sort(key=lambda e: e[0])
    streams = _controller_streams(ctrl_events)
    tempo_map.sort()
    timesig_map.sort()
    if not tempo_map or tempo_map[0][0] > 0:
        tempo_map.insert(0, (0, 500000))
    if not timesig_map or timesig_map[0][0] > 0:
        timesig_map.insert(0, (0, 4.0))

    def seconds_at(tick):
        t = 0.0
        prev_tick, prev_us = tempo_map[0]
        for tk, us in tempo_map[1:]:
            if tk >= tick:
                break
            t += (tk - prev_tick) / tpq * prev_us * 1e-6
            prev_tick, prev_us = tk, us
        return t + (tick - prev_tick) / tpq * prev_us * 1e-6

    def measure_at(tick):
        m = 1
        prev_tick, prev_q = timesig_map[0]
        for tk, q in timesig_map[1:]:
            if tk >= tick:
                break
            m += int((tk - prev_tick) / tpq // prev_q)
            prev_tick, prev_q = tk, q
        return m + int((tick - prev_tick) / tpq // prev_q)

    # Every note-on tick per (channel, note number), for the re-strike rule
    # that truncates a pedal-sustained tail.
    strikes = defaultdict(list)
    note_ons = defaultdict(list)
    for events, _ in track_events:
        for tick, status, d1, d2 in events:
            if status & 0xF0 == 0x90 and d2 > 0:
                strikes[(status & 0x0F, d1)].append(tick)
                note_ons[status & 0x0F].append(tick)
    for key in strikes:
        strikes[key].sort()
    for ch in note_ons:
        note_ons[ch].sort()

    rows = []
    part_names = []
    part_index = 0
    for ti, (events, end_tick) in enumerate(track_events):
        open_notes = {}
        track_rows = []
        for tick, status, d1, d2 in events:
            kind = status & 0xF0
            ch = status & 0x0F
            if kind == 0x90 and d2 > 0:
                key = (ch, d1)
                if key in open_notes:          # retrigger: close the old
                    t0, v0 = open_notes.pop(key)
                    track_rows.append((t0, tick, d1, v0, ch))
                open_notes[key] = (tick, d2)
            elif kind == 0x80 or (kind == 0x90 and d2 == 0):
                key = (ch, d1)
                if key in open_notes:
                    t0, v0 = open_notes.pop(key)
                    track_rows.append((t0, tick, d1, v0, ch))
        for (ch, pitch), (t0, v0) in open_notes.items():
            track_rows.append((t0, max(end_tick, t0), pitch, v0, ch))
        if not track_rows:
            continue
        part_index += 1
        name = track_names[ti] or f"track {ti + 1}"
        part_names.append(name)
        for t0, t1, pitch, vel, ch in track_rows:
            t_end = _sounding_end(t0, t1, ch, pitch, streams, strikes,
                                  file_end)
            bend = _bend_semitones(ch, t0, streams, note_ons[ch])
            volume = _state_at(streams["volume"][ch], t0, 127.0)
            express = _state_at(streams["expression"][ch], t0, 127.0)
            program = _state_at(streams["program"][ch], t0, 0.0)
            rows.append((t0 / tpq, seconds_at(t0),
                         (t1 - t0) / tpq,
                         seconds_at(t1) - seconds_at(t0),
                         (t_end - t0) / tpq,
                         seconds_at(t_end) - seconds_at(t0),
                         float(pitch) + bend, float(pitch), float(vel),
                         (vel / 127.0) * (volume / 127.0) ** 2
                         * (express / 127.0) ** 2,
                         part_index, ch + 1, program, measure_at(t0)))
    return {"rows": rows, "columns": _MIDI_COLUMNS,
            "part_names": part_names, "source": "midi"}


# --- controller streams ---------------------------------------------

_CC_VOLUME, _CC_EXPRESSION = 7, 11
_CC_SUSTAIN, _CC_SOSTENUTO = 64, 66
_CC_DATA_MSB, _CC_DATA_LSB, _CC_RPN_LSB, _CC_RPN_MSB = 6, 38, 100, 101
_RPN_BEND_RANGE, _RPN_MPE_CONFIG = (0, 0), (0, 6)
_DEFAULT_BEND_RANGE = 2.0
_MPE_BEND_RANGE = 48.0


def _controller_streams(ctrl_events):
    """Per-channel controller state, as step functions of tick.

    A value holds until the next message on that channel, so each stream
    is a sorted list of (tick, value) changes read back by :func:`_state_at`.
    """
    streams = {name: defaultdict(list) for name in
               ("volume", "expression", "sustain", "sostenuto", "bend",
                "program")}
    bend_range = {}
    rpn = defaultdict(lambda: (127, 127))     # channel -> selected RPN
    mpe_members = set()
    saw_bend = False
    saw_range = False

    for tick, ch, kind, d1, d2 in ctrl_events:
        if kind == 0xE0:
            streams["bend"][ch].append((tick, ((d2 << 7) | d1) - 8192))
            saw_bend = True
            continue
        if kind == 0xC0:
            streams["program"][ch].append((tick, float(d1)))
            continue
        if d1 == _CC_VOLUME:
            streams["volume"][ch].append((tick, float(d2)))
        elif d1 == _CC_EXPRESSION:
            streams["expression"][ch].append((tick, float(d2)))
        elif d1 == _CC_SUSTAIN:
            streams["sustain"][ch].append((tick, d2 >= 64))
        elif d1 == _CC_SOSTENUTO:
            streams["sostenuto"][ch].append((tick, d2 >= 64))
        elif d1 == _CC_RPN_MSB:
            rpn[ch] = (d2, rpn[ch][1])
        elif d1 == _CC_RPN_LSB:
            rpn[ch] = (rpn[ch][0], d2)
        elif d1 == _CC_DATA_MSB:
            if rpn[ch] == _RPN_BEND_RANGE:
                bend_range[ch] = float(d2)
                saw_range = True
            elif rpn[ch] == _RPN_MPE_CONFIG and ch in (0, 15) and d2 > 0:
                # An MPE Configuration Message: channel 1 opens a lower
                # zone, channel 16 an upper zone, over d2 member channels.
                members = (range(1, d2 + 1) if ch == 0
                           else range(15 - d2, 15))
                mpe_members.update(members)
                saw_range = True
        elif d1 == _CC_DATA_LSB and rpn[ch] == _RPN_BEND_RANGE:
            bend_range[ch] = bend_range.get(ch, 0.0) + d2 / 100.0

    if saw_bend and not saw_range:
        warnings.warn(
            "This file bends pitch but never sets a pitch-bend range "
            "(RPN 0) or an MPE zone, so the 2-semitone default is "
            "assumed; a file tuned for the 48-semitone MPE range will "
            "read 24 times too flat or sharp.", UserWarning, stacklevel=3)

    ranges = {}
    for ch in range(16):
        ranges[ch] = bend_range.get(
            ch, _MPE_BEND_RANGE if ch in mpe_members else _DEFAULT_BEND_RANGE)
    streams["bend_range"] = ranges
    return streams


def _state_at(changes, tick, default):
    """The value in force at ``tick``: the last change at or before it."""
    if not changes:
        return default
    i = bisect_right([c[0] for c in changes], tick)
    return changes[i - 1][1] if i else default


def _next_release(changes, tick, end_tick):
    """The first tick at or after ``tick`` where the pedal comes up."""
    for t, down in changes:
        if t >= tick and not down:
            return t
    return end_tick


def _sounding_end(t0, t1, ch, note, streams, strikes, end_tick):
    """When the note stops sounding, given the pedals and the re-strikes."""
    end = t1
    sustain = streams["sustain"][ch]
    if _state_at(sustain, t1, False):
        end = max(end, _next_release(sustain, t1, end_tick))
    # Sostenuto holds only what was already down when it was pressed.
    sostenuto = streams["sostenuto"][ch]
    for tp, down in sostenuto:
        if down and t0 <= tp < t1 and _state_at(sostenuto, t1, False):
            end = max(end, _next_release(sostenuto, t1, end_tick))
            break
    # The same pitch struck again on the same channel damps what is left
    # of this one; a different channel may be a different instrument, so
    # it does not.
    for t in strikes.get((ch, note), ()):
        if t >= t1:
            if t < end:
                end = t
            break
    return end


def _bend_semitones(ch, t0, streams, channel_note_ons):
    """The bend in force at a note's onset, in semitones.

    An exporter may send a note's bend just before its note-on, or at the
    same tick, and within a tick the ordering carries no meaning; both are
    covered by reading the last bend at or before the onset tick. Where a
    channel has no bend before its first note, the first bend after that
    note is used, provided no further note-on intervenes.
    """
    changes = streams["bend"][ch]
    if not changes:
        return 0.0
    ticks = [c[0] for c in changes]
    i = bisect_right(ticks, t0)
    if i:
        raw = changes[i - 1][1]
    else:
        nxt = next((t for t in channel_note_ons if t > t0), None)
        if nxt is not None and ticks[0] >= nxt:
            return 0.0
        raw = changes[0][1]
    return raw / 8192.0 * streams["bend_range"][ch]


def _midi_track_events(tdata):
    pos = 0
    tick = 0
    status = None
    events, ctrl, tempos, sigs = [], [], [], []
    name = ""
    n = len(tdata)
    while pos < n:
        delta, pos = _read_varlen(tdata, pos)
        tick += delta
        b = tdata[pos]
        if b == 0xFF:                                   # meta
            mtype = tdata[pos + 1]
            length, pos2 = _read_varlen(tdata, pos + 2)
            payload = tdata[pos2:pos2 + length]
            pos = pos2 + length
            if mtype == 0x51 and length == 3:
                tempos.append((tick, int.from_bytes(payload, "big")))
            elif mtype == 0x58 and length >= 2:
                num, den_pow = payload[0], payload[1]
                sigs.append((tick, num * 4.0 / (2 ** den_pow)))
            elif mtype == 0x03 and not name:
                name = payload.decode("latin-1", errors="replace").strip("\x00 ")
            elif mtype == 0x2F:
                return events, ctrl, tempos, sigs, name, tick
            continue
        if b in (0xF0, 0xF7):                           # sysex
            length, pos2 = _read_varlen(tdata, pos + 1)
            pos = pos2 + length
            continue
        if b & 0x80:
            status = b
            pos += 1
        if status is None:
            raise ValueError("Malformed MIDI track (data byte before status).")
        kind = status & 0xF0
        if kind in (0xC0, 0xD0):
            d1, d2 = tdata[pos], 0
            pos += 1
        else:
            d1, d2 = tdata[pos], tdata[pos + 1]
            pos += 2
        if kind in (0x80, 0x90):
            events.append((tick, status, d1, d2))
        elif kind in (0xB0, 0xC0, 0xE0):
            ctrl.append((tick, status & 0x0F, kind, d1, d2))
    return events, ctrl, tempos, sigs, name, tick


# -------------------------------------------------------------------
#  MusicXML
# -------------------------------------------------------------------

def _unzip_mxl(path):
    with zipfile.ZipFile(path) as zf:
        root_name = None
        if "META-INF/container.xml" in zf.namelist():
            cont = ET.fromstring(zf.read("META-INF/container.xml"))
            for rf in cont.iter():
                if rf.tag.endswith("rootfile") and rf.get("full-path"):
                    root_name = rf.get("full-path")
                    break
        if root_name is None:
            cands = [n for n in zf.namelist()
                     if n.lower().endswith((".xml", ".musicxml"))
                     and not n.startswith("META-INF")]
            if not cands:
                raise ValueError("No MusicXML document inside the .mxl.")
            root_name = cands[0]
        return zf.read(root_name)


def _strip_ns(tag):
    return tag.split("}", 1)[1] if "}" in tag else tag


def _text(el, name, default=None):
    child = el.find(name)
    if child is None or child.text is None:
        return default
    return child.text.strip()


def _parse_musicxml(data):
    parser = ET.XMLParser()
    try:
        root = ET.fromstring(data, parser=parser)
    except ET.ParseError as err:
        raise ValueError(f"Malformed MusicXML: {err}") from None
    for el in root.iter():
        el.tag = _strip_ns(el.tag)
    if root.tag == "score-timewise":
        root = _timewise_to_partwise(root)
    if root.tag != "score-partwise":
        raise ValueError(
            f"Not a MusicXML score (root element {root.tag!r}).")

    part_names = {}
    part_list = root.find("part-list")
    if part_list is not None:
        for sp in part_list.iter("score-part"):
            part_names[sp.get("id")] = _text(sp, "part-name", "") or ""

    # Pass 1: tempo changes at absolute quarter-note positions, from any
    # part (a change applies to the whole score).
    tempo_changes = []
    parts = root.findall("part")
    for part in parts:
        for pos_q, tempo in _walk_part(part, collect_tempo=True)[1]:
            tempo_changes.append((pos_q, tempo))
    tempo_changes.sort()
    if not tempo_changes or tempo_changes[0][0] > 0:
        tempo_changes.insert(0, (0.0, 120.0))

    def seconds_at(q):
        t = 0.0
        prev_q, prev_tempo = tempo_changes[0]
        for qq, tempo in tempo_changes[1:]:
            if qq >= q:
                break
            t += (qq - prev_q) * 60.0 / prev_tempo
            prev_q, prev_tempo = qq, tempo
        return t + (q - prev_q) * 60.0 / prev_tempo

    rows = []
    names = []
    for pi, part in enumerate(parts):
        notes, _ = _walk_part(part, collect_tempo=False)
        names.append(part_names.get(part.get("id"), "") or f"part {pi + 1}")
        for (onset_q, dur_q, pitch, vel, voice, staff, measure,
             marks) in notes:
            rows.append((onset_q, seconds_at(onset_q), dur_q,
                         seconds_at(onset_q + dur_q) - seconds_at(onset_q),
                         float(pitch), float(vel), pi + 1, voice, staff,
                         measure) + marks)
    return {"rows": rows, "columns": _XML_COLUMNS, "part_names": names,
            "source": "musicxml"}


def _xml_marks(notations):
    """A note's fermata and articulations, as 0/1 in ``_XML_COLUMNS``
    order. A merged tied note keeps a mark any of its segments carries,
    so these are combined by ``max`` when a tie is closed."""
    if notations is None:
        return (0,) * (1 + len(_XML_ARTICULATIONS))
    articulations = notations.find("articulations")
    return (int(notations.find("fermata") is not None),) + tuple(
        int(articulations is not None and articulations.find(a) is not None)
        for a in _XML_ARTICULATIONS)


def _walk_part(part, *, collect_tempo):
    """Notes ``(onset_q, dur_q, midi, velocity, voice, measure)`` and
    tempo changes ``(pos_q, bpm)`` of one part, in quarter notes."""
    divisions = 1.0
    pos = 0.0                      # current position in quarter notes
    notes = []
    tempos = []
    open_ties = {}                 # (voice, midi) -> index into notes
    for mi, measure in enumerate(part.findall("measure")):
        measure_start = pos
        try:
            measure_no = int(measure.get("number", mi + 1))
        except ValueError:
            measure_no = mi + 1
        last_onset = pos
        for el in measure:
            tag = el.tag
            if tag == "attributes":
                d = _text(el, "divisions")
                if d:
                    divisions = float(d)
            elif tag == "direction":
                for snd in el.iter("sound"):
                    if snd.get("tempo"):
                        tempos.append((pos, float(snd.get("tempo"))))
                for met in el.iter("metronome"):
                    unit = _text(met, "beat-unit", "quarter")
                    pm = _text(met, "per-minute")
                    if pm:
                        try:
                            bpm = float(pm) * _quarters_per_unit(unit)
                            if met.find("beat-unit-dot") is not None:
                                bpm *= 1.5
                            tempos.append((pos, bpm))
                        except ValueError:
                            pass
            elif tag == "sound" and el.get("tempo"):
                tempos.append((pos, float(el.get("tempo"))))
            elif tag == "backup":
                pos -= float(_text(el, "duration", "0")) / divisions
            elif tag == "forward":
                pos += float(_text(el, "duration", "0")) / divisions
            elif tag == "note":
                is_chord = el.find("chord") is not None
                is_grace = el.find("grace") is not None
                dur_div = _text(el, "duration")
                dur_q = float(dur_div) / divisions if dur_div else 0.0
                onset = last_onset if is_chord else pos
                voice = int(_text(el, "voice", "1") or 1)
                staff = int(_text(el, "staff", "1") or 1)
                pitch_el = el.find("pitch")
                is_rest = el.find("rest") is not None
                if not is_grace and pitch_el is not None and not is_rest:
                    midi = _midi_from_pitch(pitch_el)
                    dyn = el.get("dynamics")
                    vel = 90.0 if dyn is None else float(dyn) * 0.9
                    vel = float(min(127.0, max(0.0, vel)))
                    ties = {t.get("type") for t in el.findall("tie")}
                    notations = el.find("notations")
                    marks = _xml_marks(notations)
                    key = (voice, midi)
                    if "stop" in ties and key in open_ties:
                        idx = open_ties.pop(key)
                        o, dq, p_, v_, vo, st, me, old = notes[idx]
                        notes[idx] = (o, dq + dur_q, p_, v_, vo, st, me,
                                      tuple(max(a, b)
                                            for a, b in zip(old, marks)))
                        if "start" in ties:
                            open_ties[key] = idx
                    else:
                        notes.append((onset, dur_q, midi, vel, voice, staff,
                                      measure_no, marks))
                        if "start" in ties:
                            open_ties[key] = len(notes) - 1
                if not is_chord and not is_grace:
                    last_onset = pos
                    pos += dur_q
        # A measure's notes may leave pos short of its end (partial
        # voices); the next measure starts after the longest voice.
        pos = max(pos, measure_start) if pos < measure_start else pos
    return notes, tempos


def _quarters_per_unit(unit):
    return {"whole": 4.0, "half": 2.0, "quarter": 1.0, "eighth": 0.5,
            "16th": 0.25, "32nd": 0.125, "64th": 0.0625}.get(unit, 1.0)


def _midi_from_pitch(pitch_el):
    step = (_text(pitch_el, "step", "C") or "C").upper()
    alter = float(_text(pitch_el, "alter", "0") or 0.0)
    octave = int(_text(pitch_el, "octave", "4") or 4)
    return 12.0 * (octave + 1) + _STEP_TO_SEMITONE.get(step, 0) + alter


def _timewise_to_partwise(root):
    """Regroup a timewise score (measures containing parts) as partwise."""
    new = ET.Element("score-partwise")
    for child in root:
        if child.tag != "measure":
            new.append(child)
    parts = {}
    for measure in root.findall("measure"):
        for part in measure.findall("part"):
            pid = part.get("id")
            if pid not in parts:
                parts[pid] = ET.SubElement(new, "part", {"id": pid})
            m = ET.SubElement(parts[pid], "measure", dict(measure.attrib))
            for el in part:
                m.append(el)
    return new


# ===================================================================
#  pre_maet_from_score
# ===================================================================

#: How a categorical column reaches the pre-MAET. The first two are
#: structural -- the level is realized as which attribute you are in, or
#: as which position -- and aggregate a group's rows into one event. The
#: third is a value, carried alongside the note's own attributes.
_ROLES = ("separate_attributes", "ordered_multiset", "simplex", "drop")
_STRUCTURAL_ROLES = ("separate_attributes", "ordered_multiset")

#: Attributes belonging to the event rather than to the note, which a
#: structural category therefore does not split: the notes gathered into
#: one event share an onset and a bar, so splitting either by voice would
#: give one attribute per voice carrying the same number, and since the
#: attributes multiply, that factor would be raised to the fourth.
_EVENT_LEVEL = ("onset", "measure")


def pre_maet_from_score(source, *, attributes=("pitch", "onset"),
                        pitch="midi", time="seconds", weights="velocity",
                        parts=None, chords="bind", chord_tolerance=0.0,
                        roles=None, group_by=None, names=True):
    """Build the pre-MAET's parts from a score.

    Parameters
    ----------
    source : str or DataFrame
        A file path (parsed with :func:`read_score`) or an event table.
    attributes : sequence of {'pitch', 'onset', 'duration',
        'sounding_duration', 'velocity', 'weight', 'note_number', 'part',
        'measure', 'fermata'}
        The attributes, in order (default pitch and onset). The last four
        need a column the source carries, and raise where it does not.
        On a gridded table ``'onset'`` reads the grid's onset, the event
        there being the grid point rather than any one note.
    pitch : {'midi', 'cents', 'hz', 'octave', ...}
        Pitch scale (any pitch scale of :func:`transform_attributes`).
    time : {'seconds', 'beats'}
        Unit of onsets and durations.
    weights : {'velocity', 'ones', 'duration', 'weight'}
        Per-note weight: velocity / 127, one, the duration in the chosen
        time unit, or the table's ``weight`` column, which folds channel
        volume and expression into the velocity.
    parts : None, int, sequence of int, or sequence of str
        Parts to keep, either 1-based positions in the table's part
        categories or the part names themselves; ``None`` keeps all.
    chords : {'bind', 'separate'}
        ``'bind'`` gathers notes that start together (within
        ``chord_tolerance``, in the chosen time unit) into one event whose
        pitch, duration, velocity, and part attributes carry one value per
        note (``K`` = largest chord size, NaN-padded), so that a pitch
        attribute at ``r = 2`` reads the chords' dyads; ``'separate'``
        makes every note its own event (``K = 1`` throughout).
    roles : mapping, optional
        How each categorical column reaches the pre-MAET: column name to
        one of ``'separate_attributes'``, ``'ordered_multiset'``,
        ``'simplex'``, or ``'drop'``. A column with no entry is not
        encoded.

        The first two are **structural**: the level is realized as which
        attribute you are in, or as which position, so the binding of
        value to level is carried by the layout. They gather an event's
        rows into one event holding one slot per level, which needs
        ``chords='bind'`` and an event that holds exactly one row per
        level; events that do not are dropped, with a warning naming the
        count. Only one category may be structural, since a structural
        category individuates the values sounding together and two of
        them give an attribute set that can never be fully populated.

        A structural category splits every listed attribute except the
        event-level ones (``onset``), whose value belongs to the event
        rather than to the note.

        ``'simplex'`` is a **value**: the level becomes the coordinates
        of a vertex of a unit-edge regular simplex, carried as its own
        attribute read whole, and the binding of value to level is the
        tensor product of the two attributes at the event. Each
        concurrently-sounding note is then its own event, so it needs
        ``chords='separate'`` -- unless a structural category is also
        given, in which case the simplex is tagged within each of its
        slots.

        A caution about the no-role reading. With ``chords='bind'`` and
        no role, an event holds the chord as an unordered multiset on
        every attribute. Where two attributes describe the *same* notes
        -- a pitch-class attribute and a pitch-height one, say -- their
        product then pairs every value of one with every value of the
        other, including the soprano's pitch class with the bass's
        height, and matching rewards combinations the chord does not
        contain. Binding a note's attributes to each other needs one
        event per note (``chords='separate'``), which is what the JMM
        article's voice-agnostic encoding does.
    group_by : str, optional
        The column whose equal values in consecutive rows make one event.
        An event is a contiguous run, not every row sharing a value, so a
        bar number that comes round again after a repeat gives two events
        rather than one.
        The default is the grid position where the table has been
        gridded, and otherwise the onset within ``chord_tolerance``.
    chord_tolerance : float
        Onset tolerance for binding, in the chosen time unit.
    names : bool
        Name the specs after the attributes (default ``True``).

    Returns
    -------
    dict
        The pre-MAET. Its ``p_attr`` holds one ``K_a x N`` value matrix
        per attribute; its ``w_attr`` one ``K_a x N`` weight matrix per
        attribute (NaN-padded slots carry weight 0), or ``None`` under
        ``'ones'``; its ``specs`` flat specs, named after the attributes
        when ``names`` is set.
    """
    table = read_score(source) if isinstance(source, (str, os.PathLike)) \
        else source
    if not isinstance(table, pd.DataFrame):
        raise TypeError(
            "source must be a file path or an event table from "
            f"read_score; got {type(table).__name__}.")
    attributes = [str(a).lower() for a in attributes]
    allowed = ("pitch", "onset", "duration", "sounding_duration", "velocity",
               "weight", "note_number", "part", "measure", "fermata")
    for a in attributes:
        if a not in allowed:
            raise ValueError(
                f"Unknown attribute {a!r}; choose from {allowed}.")
    if time not in ("seconds", "beats"):
        raise ValueError("time must be 'seconds' or 'beats'.")
    if weights not in ("velocity", "ones", "duration", "weight"):
        raise ValueError(
            "weights must be 'velocity', 'ones', 'duration', or 'weight'.")
    if chords not in ("bind", "separate"):
        raise ValueError("chords must be 'bind' or 'separate'.")

    structural = None
    simplex_columns = []
    for column, role in dict(roles or {}).items():
        role = str(role).lower()
        if role not in _ROLES:
            raise ValueError(
                f"roles[{column!r}]: unknown role {role!r}; choose from "
                f"{_ROLES}.")
        if column not in table.columns:
            raise KeyError(
                f"roles names column {column!r}, which the table does not "
                "have.")
        if role == "drop":
            continue
        if not isinstance(table[column].dtype, pd.CategoricalDtype):
            raise TypeError(
                f"roles[{column!r}]: a role needs a categorical column, and "
                f"{column} is {table[column].dtype}.")
        if role == "simplex":
            simplex_columns.append(column)
            continue
        if structural is not None:
            raise ValueError(
                f"roles[{column!r}]: only one category may be structural "
                f"against a given value set, and {structural[0]!r} already "
                "is. A structural category individuates the values sounding "
                "together; two of them give an attribute set that can never "
                "be fully populated. Either keep "
                f"{structural[0]!r} structural and give {column!r} the "
                "'simplex' role, which tags it within each slot, or give "
                f"{structural[0]!r} the 'simplex' role too, which yields one "
                "event per note and additive partial credit across levels.")
        structural = (column, role)

    if structural is not None and chords != "bind":
        raise ValueError(
            f"roles[{structural[0]!r}] is structural, which gathers the rows "
            "of an event into one; that needs chords='bind'.")
    if simplex_columns and structural is None and chords == "bind":
        raise ValueError(
            "The 'simplex' role carries the level as a value at each event, "
            "so each concurrently-sounding note is its own event; that needs "
            "chords='separate', or a structural category to tag within.")

    _NEEDS_COLUMN = {"sounding_duration": "sounding_duration_beats",
                     "weight": "weight", "note_number": "note_number",
                     "fermata": "fermata"}
    for name in attributes:
        column = _NEEDS_COLUMN.get(name)
        if column is not None and column not in table.columns:
            raise ValueError(
                f"The table has no {column!r} column, so {name!r} cannot be "
                "an attribute; this source does not carry it.")
    if weights == "weight" and "weight" not in table.columns:
        raise ValueError(
            "weights='weight' needs a 'weight' column, which this source "
            "does not carry.")

    part_codes = table["part"].cat.codes.to_numpy() + 1
    keep = np.ones(len(table), dtype=bool)
    if parts is not None:
        wanted = np.atleast_1d(np.asarray(parts, dtype=object))
        numeric = all(isinstance(v, (int, np.integer))
                      or (isinstance(v, float) and float(v).is_integer())
                      for v in wanted)
        if numeric:
            keep &= np.isin(part_codes, wanted.astype(np.intp))
        else:
            keep &= np.isin(table["part"].astype(object).to_numpy(),
                            wanted.astype(str))

    def _col(name):
        # astype first, so that a nullable column a gridded table may
        # carry (an empty grid point has no measure, no voice) reads back
        # as NaN rather than raising.
        return table[name].astype(float).to_numpy()[keep]

    unit = "seconds" if time == "seconds" else "beats"
    # On a gridded table the event is the grid point, so its onset is the
    # grid's, not the onset of whichever note happens to be in the first
    # slot. The note's own onset stays in the table for selection.
    onset_column = f"onset_{unit}"
    if f"grid_onset_{unit}" in table.columns:
        onset_column = f"grid_onset_{unit}"
    elif any(c.startswith("grid_onset_") for c in table.columns):
        other = next(c for c in table.columns if c.startswith("grid_onset_"))
        raise ValueError(
            f"The table was gridded over {other.rsplit('_', 1)[1]}, so "
            f"time={time!r} has no grid onset to read; grid over {time} or "
            "convert with that unit.")
    onset = _col(onset_column)
    dur = _col(f"duration_{unit}")
    midi = _col("pitch")
    vel = _col("velocity")
    part = part_codes.astype(np.float64)[keep]
    measure = _col("measure")
    n_kept = int(keep.sum())

    def _optional(name):
        return _col(name) if name in table.columns else np.zeros(n_kept)

    sounding = _optional(f"sounding_duration_{unit}")
    note_number = _optional("note_number")
    weight_col = _optional("weight")
    fermata = _optional("fermata")
    n_notes = int(midi.size)

    pitch_vals = (midi if pitch.lower() == "midi"
                  else _convert_scale(midi, "midi", pitch))
    per_note = {"pitch": pitch_vals, "onset": onset, "duration": dur,
                "sounding_duration": sounding, "velocity": vel,
                "weight": weight_col, "note_number": note_number,
                "part": part, "measure": measure, "fermata": fermata}
    if weights == "velocity":
        w_note = vel / 127.0
    elif weights == "duration":
        w_note = dur.copy()
    elif weights == "weight":
        w_note = weight_col.copy()
    else:
        w_note = None

    # Group notes into events. An explicit key, or a gridded table's own
    # grid position, already says which rows share an event, and the
    # onset tolerance then does not apply.
    if group_by is not None and group_by not in table.columns:
        raise KeyError(
            f"group_by names column {group_by!r}, which the table does not "
            "have.")
    key_column = group_by or ("grid_index" if "grid_index" in table.columns
                              else None)
    if chords == "separate" or n_notes == 0:
        groups = [[i] for i in range(n_notes)]
    elif key_column is not None:
        key = table[key_column].astype(object).to_numpy()[keep]
        groups = []
        for i in range(n_notes):
            if groups and key[i] == key[groups[-1][0]]:
                groups[-1].append(i)
            else:
                groups.append([i])
    else:
        order = np.argsort(onset, kind="stable")
        groups = []
        for i in order:
            if groups and onset[i] - onset[groups[-1][0]] <= chord_tolerance:
                groups[-1].append(int(i))
            else:
                groups.append([int(i)])

    def _codes(column):
        return table[column].cat.codes.to_numpy()[keep]

    # A structural category puts one of its levels in each slot of every
    # event, so an event must hold exactly one row per level. One that
    # does not is dropped, with a count: an analyst may well accept
    # losing a few events to use the encoding.
    slots = None
    if structural is not None:
        column, role = structural
        levels = list(table[column].cat.categories)
        codes = _codes(column)
        slots, kept_groups, lost = [], [], 0
        for g in groups:
            row = [-1] * len(levels)
            for i in g:
                c = int(codes[i])
                if c < 0 or row[c] >= 0:
                    row = None
                    break
                row[c] = i
            if row is None or any(r < 0 for r in row):
                lost += 1
                continue
            slots.append(row)
            kept_groups.append(g)
        if lost:
            gridded = "; on a gridded table that breaks the uniform time " \
                      "index" if key_column == "grid_index" else ""
            warnings.warn(
                f"{lost} of {len(groups)} events do not hold exactly one "
                f"{column} per level, so they are dropped{gridded}. A "
                "structural category fills every slot of every event.",
                UserWarning, stacklevel=2)
        groups = kept_groups

    N = len(groups)
    K = max((len(g) for g in groups), default=1)

    def _weight_at(i):
        return 1.0 if w_note is None else float(w_note[i])

    p_attr, w_list = [], []
    spec_r, spec_exch, spec_names = [], [], []

    def _add(M, W, *, r=1, exch=True, name=None):
        W = np.asarray(W, dtype=float)
        M = np.asarray(M, dtype=float)
        W[np.isnan(M)] = 0.0
        p_attr.append(M)
        w_list.append(W)
        spec_r.append(r)
        spec_exch.append(exch)
        spec_names.append(name)

    for a in attributes:
        vals = per_note[a]
        if structural is None:
            if a in _EVENT_LEVEL or K == 1:
                M = np.full((1, N), np.nan)
                W = np.zeros((1, N))
                for n, g in enumerate(groups):
                    M[0, n] = vals[g[0]]
                    W[0, n] = 1.0 if a in _EVENT_LEVEL else _weight_at(g[0])
            else:
                M = np.full((K, N), np.nan)
                W = np.zeros((K, N))
                for n, g in enumerate(groups):
                    M[:len(g), n] = vals[g]
                    W[:len(g), n] = [_weight_at(i) for i in g]
            _add(M, W, name=a)
            continue

        column, role = structural
        if a in _EVENT_LEVEL:
            M = np.array([[vals[row[0]] for row in slots]], dtype=float)
            _add(M, np.ones((1, N)), name=a)
        elif role == "separate_attributes":
            for v, level in enumerate(levels):
                M = np.array([[vals[row[v]] for row in slots]], dtype=float)
                W = np.array([[_weight_at(row[v]) for row in slots]])
                _add(M, W, name=f"{a}_{level}")
        else:
            V = len(levels)
            M = np.empty((V, N))
            W = np.empty((V, N))
            for n, row in enumerate(slots):
                for v in range(V):
                    M[v, n] = vals[row[v]]
                    W[v, n] = _weight_at(row[v])
            _add(M, W, r=V, exch=False, name=a)

    # Simplex-coded categories. The coordinates of one level form one
    # ordered value read whole, so the attribute's tuple size is their
    # number and its weights are one: the note's own weight is carried by
    # its other attributes, and the attributes multiply.
    for column in simplex_columns:
        vertices = simplex_vertices(len(table[column].cat.categories))
        d = vertices.shape[1]
        codes = _codes(column)

        def _coords(i):
            c = int(codes[i])
            return vertices[c] if c >= 0 else np.full(d, np.nan)

        if structural is None:
            M = np.column_stack([_coords(g[0]) for g in groups]) if N \
                else np.zeros((d, 0))
            _add(M, np.ones((d, N)), r=d, exch=False, name=column)
        else:
            for v, level in enumerate(levels):
                M = np.column_stack([_coords(row[v]) for row in slots]) \
                    if N else np.zeros((d, 0))
                _add(M, np.ones((d, N)), r=d, exch=False,
                     name=f"{column}_{level}")

    w = None if w_note is None else w_list
    specs = flat_specs(p_attr, r=spec_r, exch=spec_exch,
                       name=spec_names if names else None)
    # A score determines the periodicity of its attributes and not their
    # kernel widths. Pitches, onsets, durations, velocities, parts, bars
    # and fermatas are all read as they are written --- absolute, on an
    # unbounded axis --- so [per] = 0 and the period is inert; octave
    # equivalence is an equivalence the analyst imposes, not one the score
    # states. Sigma is left unset rather than defaulted, because there is
    # no width a score implies: build_maet will then name the
    # attribute that still needs one.
    for spec in specs:
        spec["is_per"] = False
        spec["period"] = 0.0
    return pre_maet(p_attr, w, specs)
