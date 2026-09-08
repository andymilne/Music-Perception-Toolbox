"""Pre-MAET events from symbolic scores (MIDI and MusicXML).

Two functions. :func:`read_score` parses a Standard MIDI File (format 0
or 1) or a MusicXML file (``.musicxml``, ``.xml``, or compressed
``.mxl``) into a *note table*: one row per sounding note with its onset
and duration in beats and in seconds, its MIDI pitch, its velocity, and
its part. :func:`events_from_score` turns a note table (or a path) into
the ``(p_attr, w_attr, specs)`` that :func:`build_exp_tens` and the
pre-MAET preprocessors consume, choosing the attributes, their units,
the weights, and whether simultaneous notes are bound into one
multi-value event.

Both parsers are self-contained (no third-party dependency) and mirror
``readScore`` / ``eventsFromScore`` in MATLAB, which read the same files
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
import zipfile
import xml.etree.ElementTree as ET

import numpy as np

from ._tensor.premaet import pre_maet
from ._tensor.preprocessing import flat_specs
from ._tensor.transform import _convert_scale

__all__ = ["read_score", "events_from_score"]

_NOTE_FIELDS = ("onset_beats", "onset_seconds", "duration_beats",
                "duration_seconds", "pitch", "velocity", "part", "channel",
                "measure", "fermata")
_STEP_TO_SEMITONE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


# ===================================================================
#  read_score
# ===================================================================

def read_score(path):
    """Parse a MIDI or MusicXML file into a note table.

    Parameters
    ----------
    path : str
        A ``.mid`` / ``.midi`` file (format 0 or 1), a ``.musicxml`` /
        ``.xml`` file (partwise or timewise), or a compressed ``.mxl``.

    Returns
    -------
    dict
        ``{'onset_beats', 'onset_seconds', 'duration_beats',
        'duration_seconds', 'pitch', 'velocity', 'part', 'channel',
        'measure', 'fermata', 'part_names', 'source'}``: one float
        array per note field (``part``, ``channel``, and ``measure``
        are 1-based ints; ``channel`` is 0 for MusicXML, where it
        carries the voice number instead; ``fermata`` is 1 for a
        MusicXML note carrying a fermata, a merged tied note counting
        if any of its segments does, and 0 otherwise, MIDI having no
        fermatas), ``part_names`` a list of the parts' names,
        ``source`` ``'midi'`` or ``'musicxml'``. Rows are sorted by
        onset, then part, then pitch.
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
    rows = notes["rows"]
    table = {}
    if rows:
        order = sorted(range(len(rows)),
                       key=lambda i: (rows[i][0], rows[i][6], rows[i][4]))
        cols = list(zip(*[rows[i] for i in order]))
    else:
        cols = [[] for _ in _NOTE_FIELDS]
    for name, col in zip(_NOTE_FIELDS, cols):
        if name in ("part", "channel", "measure", "fermata"):
            table[name] = np.asarray(col, dtype=np.intp)
        else:
            table[name] = np.asarray(col, dtype=np.float64)
    table["part_names"] = list(notes["part_names"])
    table["source"] = notes["source"]
    return table


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
    for tdata in tracks:
        events, tempos, sigs, name, end_tick = _midi_track_events(tdata)
        track_events.append((events, end_tick))
        tempo_map.extend(tempos)
        timesig_map.extend(sigs)
        track_names.append(name)
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
            rows.append((t0 / tpq, seconds_at(t0), (t1 - t0) / tpq,
                         seconds_at(t1) - seconds_at(t0), float(pitch),
                         float(vel), part_index, ch + 1, measure_at(t0), 0))
    return {"rows": rows, "part_names": part_names, "source": "midi"}


def _midi_track_events(tdata):
    pos = 0
    tick = 0
    status = None
    events, tempos, sigs = [], [], []
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
                return events, tempos, sigs, name, tick
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
    return events, tempos, sigs, name, tick


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
        for onset_q, dur_q, pitch, vel, voice, measure, fermata in notes:
            rows.append((onset_q, seconds_at(onset_q), dur_q,
                         seconds_at(onset_q + dur_q) - seconds_at(onset_q),
                         float(pitch), float(vel), pi + 1, voice, measure,
                         fermata))
    return {"rows": rows, "part_names": names, "source": "musicxml"}


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
                pitch_el = el.find("pitch")
                is_rest = el.find("rest") is not None
                if not is_grace and pitch_el is not None and not is_rest:
                    midi = _midi_from_pitch(pitch_el)
                    dyn = el.get("dynamics")
                    vel = 90.0 if dyn is None else float(dyn) * 0.9
                    vel = float(min(127.0, max(0.0, vel)))
                    ties = {t.get("type") for t in el.findall("tie")}
                    notations = el.find("notations")
                    fermata = int(notations is not None
                                  and notations.find("fermata") is not None)
                    key = (voice, midi)
                    if "stop" in ties and key in open_ties:
                        idx = open_ties.pop(key)
                        o, dq, p_, v_, vo, me, fe = notes[idx]
                        notes[idx] = (o, dq + dur_q, p_, v_, vo, me,
                                      max(fe, fermata))
                        if "start" in ties:
                            open_ties[key] = idx
                    else:
                        notes.append((onset, dur_q, midi, vel, voice,
                                      measure_no, fermata))
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
#  events_from_score
# ===================================================================

def events_from_score(source, *, attributes=("pitch", "onset"),
                      pitch="midi", time="seconds", weights="velocity",
                      parts=None, chords="bind", chord_tolerance=0.0,
                      names=True):
    """Build the pre-MAET's parts from a score.

    Parameters
    ----------
    source : str or dict
        A file path (parsed with :func:`read_score`) or a note table.
    attributes : sequence of {'pitch', 'onset', 'duration', 'velocity',
        'part', 'measure', 'fermata'}
        The attributes, in order (default pitch and onset).
    pitch : {'midi', 'cents', 'hz', 'octave', ...}
        Pitch scale (any pitch scale of :func:`transform_attributes`).
    time : {'seconds', 'beats'}
        Unit of onsets and durations.
    weights : {'velocity', 'ones', 'duration'}
        Per-note weight: velocity / 127, one, or the duration in the
        chosen time unit.
    parts : None, int, or sequence of int
        Parts to keep (1-based, as in the table); ``None`` keeps all.
    chords : {'bind', 'separate'}
        ``'bind'`` gathers notes that start together (within
        ``chord_tolerance``, in the chosen time unit) into one event whose
        pitch, duration, velocity, and part attributes carry one value per
        note (``K`` = largest chord size, NaN-padded), so that a pitch
        attribute at ``r = 2`` reads the chords' dyads; ``'separate'``
        makes every note its own event (``K = 1`` throughout).
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
    attributes = [str(a).lower() for a in attributes]
    allowed = ("pitch", "onset", "duration", "velocity", "part", "measure",
               "fermata")
    for a in attributes:
        if a not in allowed:
            raise ValueError(
                f"Unknown attribute {a!r}; choose from {allowed}.")
    if time not in ("seconds", "beats"):
        raise ValueError("time must be 'seconds' or 'beats'.")
    if weights not in ("velocity", "ones", "duration"):
        raise ValueError("weights must be 'velocity', 'ones', or 'duration'.")
    if chords not in ("bind", "separate"):
        raise ValueError("chords must be 'bind' or 'separate'.")

    keep = np.ones(table["pitch"].size, dtype=bool)
    if parts is not None:
        parts_v = np.atleast_1d(np.asarray(parts, dtype=np.intp))
        keep &= np.isin(table["part"], parts_v)
    onset = (table["onset_seconds"] if time == "seconds"
             else table["onset_beats"])[keep]
    dur = (table["duration_seconds"] if time == "seconds"
           else table["duration_beats"])[keep]
    midi = table["pitch"][keep]
    vel = table["velocity"][keep]
    part = table["part"][keep].astype(np.float64)
    measure = table["measure"][keep].astype(np.float64)
    fermata = table.get("fermata", np.zeros(len(table["pitch"]), dtype=np.intp))[keep].astype(np.float64)
    n_notes = int(midi.size)

    pitch_vals = (midi if pitch.lower() == "midi"
                  else _convert_scale(midi, "midi", pitch))
    per_note = {"pitch": pitch_vals, "onset": onset, "duration": dur,
                "velocity": vel, "part": part, "measure": measure,
                "fermata": fermata}
    if weights == "velocity":
        w_note = vel / 127.0
    elif weights == "duration":
        w_note = dur.copy()
    else:
        w_note = None

    # Group notes into events.
    if chords == "separate" or n_notes == 0:
        groups = [[i] for i in range(n_notes)]
    else:
        order = np.argsort(onset, kind="stable")
        groups = []
        for i in order:
            if groups and onset[i] - onset[groups[-1][0]] <= chord_tolerance:
                groups[-1].append(int(i))
            else:
                groups.append([int(i)])
    N = len(groups)
    K = max((len(g) for g in groups), default=1)

    p_attr, w_list = [], []
    for a in attributes:
        vals = per_note[a]
        if a == "onset" or K == 1:
            M = np.full((1, N), np.nan)
            W = np.zeros((1, N))
            for n, g in enumerate(groups):
                M[0, n] = vals[g[0]] if a == "onset" else vals[g[0]]
                if a == "onset":
                    W[0, n] = 1.0
                else:
                    W[0, n] = 1.0 if w_note is None else w_note[g[0]]
        else:
            M = np.full((K, N), np.nan)
            W = np.zeros((K, N))
            for n, g in enumerate(groups):
                M[:len(g), n] = vals[g]
                W[:len(g), n] = 1.0 if w_note is None else w_note[g]
        p_attr.append(M)
        w_list.append(W)
    if w_note is None:
        w = None
    else:
        w = w_list
    specs = flat_specs(p_attr, name=attributes if names else None)
    # A score determines the periodicity of its attributes and not their
    # kernel widths. Pitches, onsets, durations, velocities, parts, bars
    # and fermatas are all read as they are written --- absolute, on an
    # unbounded axis --- so [per] = 0 and the period is inert; octave
    # equivalence is an equivalence the analyst imposes, not one the score
    # states. Sigma is left unset rather than defaulted, because there is
    # no width a score implies: build_exp_tens will then name the
    # attribute that still needs one.
    for spec in specs:
        spec["is_per"] = False
        spec["period"] = 0.0
    return pre_maet(p_attr, w, specs)
