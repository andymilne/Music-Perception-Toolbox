"""bwv_window.py — beat aggregates, nested context and query builders, and
pitch-derived flags for Analysis 1.3 (cadence localization in BWV 347).

The encoding is the article's (Section "Cadence localization using
nested multisets"):

* The chorale is read as eighth-note events on the half-QN grid: each
  event's chord is the SATB sonority sounding at that eighth, weighted by
  metrical position (1 on the beat, 0.5 off it) and raised by half again
  under a fermata (``jmm_data.bwv347_fermata_spans``).
* A window's events are merged to one beat aggregate: within each voice,
  the weights of equal pitches are summed across the window's events
  (multiplicities across voices are preserved — a doubled pitch stays
  doubled), and the result is normalized by the 1.5 a beat carries. A
  pitch sustained through the beat thus has weight 1 (1.5 under a
  fermata); an off-beat passing chord enters at half weight.
* A context window pair (or triple, for the three-chord prototypes) is
  one bound pitch attribute: the inner level is each aggregate's pitch
  multiset ([exch] = 1, r = r_inner), the outer level the ordered
  aggregates ([exch] = 0, r = L), taken relative at the outer level alone
  ([rel] = (0, 1)) and periodic (P = 12, sigma = 0.15). The optional
  inversion flag is a second, simplex-coded attribute (+/-0.5,
  sigma_flag = 0.1) carried by query and context alike.

All densities are built with the toolbox's ``bind_events`` ->
``build_maet`` pipeline; similarities use ``sim_maet``
(re-exported for the caller). Data come from ``jmm_data`` (the bundled
MusicXML read with ``mpt.read_score``).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import mpt
from mpt import unpack_pre_maet
from mpt import (bind_events, flat_specs, build_maet,  # noqa: F401
                 grid_attr_table, sim_maet, show_pre_maet)

from jmm_data import (bwv347_notes, bwv347_fermata_spans, GRID_STEP_QN)

# --- kernel parameters (the article's) ---------------------------------------
SIGMA_PITCH = 0.15      # semitones
PERIOD      = 12.0      # octave (MIDI semitones)
SIGMA_FLAG  = 0.1       # simplex-coded two-level flag
ROOT_YES    = +0.5      # flag level: predicate holds
ROOT_NO     = -0.5      # flag level: predicate fails

EIGHTH = 0.5            # the eighth-note event grain, QN
BEAT_WEIGHT_NORM = 1.5  # the weight a beat carries (1 on-beat + 0.5 off-beat)

# --- chorale (played-through, repeats expanded) ------------------------------
_NOTES = bwv347_notes()
_SIXTEENTHS = grid_attr_table(_NOTES, GRID_STEP_QN)
T0 = float(_SIXTEENTHS["grid_onset_beats"].min())
T1 = float(_SIXTEENTHS["grid_onset_beats"].max()) + GRID_STEP_QN

_FERMATA_SPANS = bwv347_fermata_spans()


def _under_fermata(t: float) -> bool:
    return any(s - 1e-9 <= t < e - 1e-9 for s, e in _FERMATA_SPANS)


def son_at(t: float) -> np.ndarray:
    """The sonority (MIDI pitches) sounding at QN time t."""
    rows = _SIXTEENTHS[np.isclose(_SIXTEENTHS["grid_onset_beats"], t)]
    return rows["pitch"].to_numpy(dtype=float)


# --- the chorale as beat aggregates ------------------------------------------
def _beat_table():
    """One row per note per beat, weighted as the article specifies.

    The eighth-note grain comes first, where coverage is the article's
    "fraction of the eighth each note sounds": 1 for a note sounding
    through the eighth, 1/2 for one sounding a single sixteenth. Those
    slices are then weighted by metrical position --- the on-beat eighth
    at 1 and the off-beat at 1/2, normalized to mean one so that a note
    sounding through the beat still weighs one --- and raised by half
    under a fermata. Regridding to the beat combines them: a note
    sounding in both eighths becomes one row whose weights have been
    summed, while two simultaneous notes of the same pitch stay two rows,
    each carrying its own note id, so a doubling stays doubled.
    """
    eighths = grid_attr_table(_NOTES, EIGHTH, weights="coverage",
                              limits=(T0, T1)).copy()
    onset = eighths["grid_onset_beats"].to_numpy(dtype=float)
    metric = np.where(np.isclose(onset % 1.0, 0.0), 1.0, 0.5)
    fermata = np.array([1.5 if _under_fermata(t) else 1.0 for t in onset])
    eighths["weight"] = (eighths["weight"] * fermata
                         * metric / (BEAT_WEIGHT_NORM / 2.0))
    return grid_attr_table(eighths, 1.0)


#: The pitch attribute of each beat, NaN-padded where beats hold unequal
#: numbers of notes, with the beat's own time alongside for locating it.
_BEATS = mpt.pre_maet_from_attr_table(
    _beat_table(),
    attributes=(dict(column="pitch", sigma=SIGMA_PITCH, r=1, exch=True,
                     is_per=True, period=PERIOD),
                dict(column="onset", sigma=1.0)),
    time="beats", weights="weight")


def _onsets(pm):
    p_attr, _, specs = unpack_pre_maet(pm)
    return p_attr[[spec["name"] for spec in specs].index("onset")][0]


#: The pitch-derived inversion predicates, as one value per beat. They are
#: rows rather than columns because the flag is a property of the beat, not
#: of each note in it: a column on the attribute table would arrive with one
#: value per note. Each pairs with the window that reads it --- a
#: three-beat window starts at its own antepenult, so ``six_four`` is read
#: at the window's first beat, while a two-beat window resolves on its
#: second, so ``root_position_next`` is the predicate one beat on.
def _flag_rows():
    beats = _onsets(_BEATS)
    six_four = [ROOT_YES if is_six_four(son_at(t)) else ROOT_NO for t in beats]
    root_next = [ROOT_YES if is_root_position(son_at(t + 1.0)) else ROOT_NO
                 if np.any(np.isclose(beats, t + 1.0)) else ROOT_NO
                 for t in beats]
    return {"six_four": np.array([six_four], dtype=float),
            "root_position_next": np.array([root_next], dtype=float)}


def _flag_spec(values):
    return flat_specs([values], r=1, rel=False, exch=False, name="flag",
                      sigma=SIGMA_FLAG, is_per=False, period=0.0)


def bound_context(L: int, r_inner: int = 1, flag: str | None = None):
    """Every window of L consecutive beats, bound and nested, in one call.

    ``bind_events`` slides the window along the whole chorale at once, so
    the piece is nested once rather than once per position: the result
    holds one super-event per window, the inner level each beat's pitch
    multiset ([exch] = 1, r = r_inner) and the outer level the L ordered
    beats ([exch] = 0, r = L), relative at the outer level alone.

    Per-attribute bind orders keep everything but the pitch flat at one
    value per window: the window's own start time, which locates it for a
    sweep, and the named inversion flag where one is asked for.
    """
    p_attr, w_attr, specs = unpack_pre_maet(_BEATS)
    p_attr, w_attr, specs = list(p_attr), list(w_attr or []), list(specs)
    specs[0] = dict(specs[0], r=r_inner, rel=False, exch=True)
    orders = [L, 1]
    if flag is not None:
        values = _FLAGS[flag]
        p_attr.append(values)
        w_attr.append(np.ones_like(values))
        specs.append(_flag_spec(values)[0])
        orders.append(1)
    return bind_events(p_attr, w_attr or None, orders, rel_outer=True,
                       specs=specs)


def query(chords, flag=None, r_inner: int = 1):
    """A literal chord succession as a bound query, read exactly as a
    window of the chorale is: the same conversion, the same bind orders,
    and a constant flag where the context carries one."""
    onset = [float(j) for j, c in enumerate(chords) for _ in c]
    pitch = [float(v) for c in chords for v in c]
    table = pd.DataFrame({"onset_beats": onset, "pitch": pitch,
                          "weight": np.ones(len(pitch))})
    pm = mpt.pre_maet_from_attr_table(
        table,
        attributes=(dict(column="pitch", sigma=SIGMA_PITCH, r=1, exch=True,
                         is_per=True, period=PERIOD),
                    dict(column="onset", sigma=1.0)),
        time="beats", weights="weight")
    p_attr, w_attr, specs = unpack_pre_maet(pm)
    p_attr, w_attr, specs = list(p_attr), list(w_attr or []), list(specs)
    specs[0] = dict(specs[0], r=r_inner, rel=False, exch=True)
    orders = [len(chords), 1]
    if flag is not None:
        values = np.full((1, len(chords)), float(flag))
        p_attr.append(values)
        w_attr.append(np.ones_like(values))
        specs.append(_flag_spec(values)[0])
        orders.append(1)
    return bind_events(p_attr, w_attr or None, orders, rel_outer=True,
                       specs=specs)


def as_compared(pm):
    """A bound pre-MAET without its placement axis: what the cosine
    actually receives, once the sweep has used the time to place the
    window."""
    kept = [spec["name"] for spec in unpack_pre_maet(pm)[2]
            if spec["name"] != "onset"]
    return mpt.select_pre_maet(pm, attributes=kept)


# --- the minimal cadential prototype (dyad skeleton) -------------------------
_DYAD_CHORDS = ([59.0, 65.0], [60.0, 64.0])      # B-F -> C-E


def dyad_query(flag=None, r_inner: int = 1):
    """The dyad-skeleton query: B-F -> C-E, the minimal cadential
    prototype, read as a two-beat window is."""
    return query(_DYAD_CHORDS, flag=flag, r_inner=r_inner)


# --- pitch-derived chord predicates ------------------------------------------
def _pcs_above_bass(son) -> list[int]:
    p = np.asarray(son, dtype=float)
    p = p[np.isfinite(p)]
    return sorted({int(round(x - p.min())) % 12 for x in p})


def is_root_position(son) -> bool:
    """Pitch-derived root-position test, per the specified rules: a fifth
    above the bass; or a major or minor third above the bass with no
    fourth, no fifth, and no sixth."""
    pcs = _pcs_above_bass(son)
    if 7 in pcs:
        return True
    return bool({3, 4} & set(pcs)) and not ({5, 8, 9} & set(pcs))


def is_six_four(son) -> bool:
    """Pitch-derived second-inversion test: a perfect fourth above the
    bass."""
    return 5 in _pcs_above_bass(son)


def b2bar(t):
    """Played-through bar coordinate from QN time (pickup at 0–1)."""
    return (np.asarray(t, dtype=float) - 1.0) / 4.0 + 1.0


#: Built last: the predicates above are what define them.
_FLAGS = _flag_rows()
