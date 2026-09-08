"""bwv_window.py — beat aggregates, nested context and query builders, and
pitch-derived flags for Analysis 1.4 (cadence localization in BWV 347).

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
  multiset ([sym] = 1, r = r_inner), the outer level the ordered
  aggregates ([sym] = 0, r = L), taken relative at the outer level alone
  ([rel] = (0, 1)) and periodic (P = 12, sigma = 0.15). The optional
  inversion flag is a second, simplex-coded attribute (+/-0.5,
  sigma_flag = 0.1) carried by query and context alike.

All densities are built with the toolbox's ``bind_events`` ->
``build_exp_tens`` pipeline; similarities use ``cos_sim_exp_tens``
(re-exported for the caller). Data come from ``jmm_data`` (the bundled
MusicXML read with ``mpt.read_score``).
"""
from __future__ import annotations
import numpy as np

import mpt
from mpt import unpack_pre_maet
mpt.set_default(show_hints=False)
from mpt import (bind_events, flat_specs, build_exp_tens,  # noqa: F401
                 cos_sim_exp_tens, show_pre_maet)

from jmm_data import bwv347_grid, bwv347_fermata_spans, GRID_STEP_QN

# --- kernel parameters (the article's) ---------------------------------------
SIGMA_PITCH = 0.15      # semitones
PERIOD      = 12.0      # octave (MIDI semitones)
SIGMA_FLAG  = 0.1       # simplex-coded two-level flag
ROOT_YES    = +0.5      # flag level: predicate holds
ROOT_NO     = -0.5      # flag level: predicate fails

EIGHTH = 0.5            # the eighth-note event grain, QN
BEAT_WEIGHT_NORM = 1.5  # the weight a beat carries (1 on-beat + 0.5 off-beat)

# --- chorale (played-through, repeats expanded) ------------------------------
_TIMES, _SATB, _BARS = bwv347_grid()
T0 = float(_TIMES[0])
T1 = float(_TIMES[-1]) + GRID_STEP_QN          # end of the last grid step

_FERMATA_SPANS = bwv347_fermata_spans()


def _under_fermata(t: float) -> bool:
    return any(s - 1e-9 <= t < e - 1e-9 for s, e in _FERMATA_SPANS)


# Eighth-note events: time, per-voice sounding pitches, metric/fermata
# weight. An event's sounding pitches are all pitches sounding during the
# eighth — a voice moving at the sixteenth level contributes both of its
# pitches, each at the event's weight.
_E8_TIMES = np.arange(T0, T1 - EIGHTH + 1e-9, EIGHTH)
_E8_W = np.where(np.isclose(_E8_TIMES % 1.0, 0.0), 1.0, 0.5)
_E8_W = _E8_W * np.array([1.5 if _under_fermata(t) else 1.0 for t in _E8_TIMES])


def son_at(t: float) -> np.ndarray:
    """SATB sonority (4 MIDI pitches) sounding at QN time t."""
    return _SATB[int(round((t - T0) / GRID_STEP_QN))].copy()


# --- the score's notes, recovered from the sampled grid ----------------------
# One note per (pitch, contiguous sounding span): within each part's stream a
# run of equal pitches across consecutive grid points is one note (the grid
# cannot see a re-articulation, and the merging rule treats a pitch persisting
# across consecutive events as one entry in any case). Simultaneous notes of
# the same pitch are distinct notes — a doubling stays doubled.
def _extract_notes():
    notes = []
    n_grid = len(_SATB)
    for stream in _SATB.T:
        start = 0
        for i in range(1, n_grid + 1):
            if i == n_grid or stream[i] != stream[start]:
                notes.append((float(stream[start]),
                              T0 + start * GRID_STEP_QN,
                              T0 + i * GRID_STEP_QN))
                start = i
    return notes


_NOTES = _extract_notes()


def _event_note_fracs(t: float):
    """Notes sounding during the eighth-note event at t, as a list of
    (note id, pitch, sounding fraction): fraction 1 when the note sounds
    through the eighth, 0.5 when it sounds for a single sixteenth."""
    out = []
    for nid, (p, a, b) in enumerate(_NOTES):
        frac = 0.0
        for g in (t, t + GRID_STEP_QN):
            if a - 1e-9 <= g < b - 1e-9:
                frac += 0.5
        if frac > 0.0:
            out.append((nid, p, frac))
    return out


def win_events(a: float, b: float):
    """Eighth-note events in [a, b): (times, per-event note lists, weights)."""
    m = (_E8_TIMES >= a - 1e-9) & (_E8_TIMES < b - 1e-9)
    times = _E8_TIMES[m]
    chords = [_event_note_fracs(t) for t in times]
    return times, chords, _E8_W[m]


def aggregate(win):
    """Merge a window's eighth events to one beat aggregate (p, w).

    One entry per note sounding in the window: a note sounding in both of
    the beat's events is a single entry whose weights sum (the merging
    rule is that its pitch persists across the merged events); two
    simultaneous notes of the same pitch remain two entries (a doubling
    stays doubled). Each note's weight in an event is the event weight
    times its sounding fraction; all weights are normalized by the 1.5 a
    beat carries.
    """
    _, chords, weights = win
    merged: dict[int, list] = {}
    for ev, w in zip(chords, weights):
        for nid, p, frac in ev:
            if nid in merged:
                merged[nid][1] += float(w) * frac
            else:
                merged[nid] = [p, float(w) * frac]
    p_out = np.array([v[0] for v in merged.values()])
    w_out = np.array([v[1] for v in merged.values()]) / BEAT_WEIGHT_NORM
    return p_out, w_out


def bound_density(aggs, flag=None, r_inner: int = 1):
    """MAET density of one bound super-event from L beat aggregates.

    The pitch attribute nests the aggregates: inner level the chord
    multiset ([sym] = 1, r = r_inner), outer level the L ordered
    aggregates ([sym] = 0, r = L), relative at the outer level alone
    ([rel] = (0, 1)), periodic at the octave. An optional flag value adds
    the simplex-coded inversion attribute.
    """
    L = len(aggs)
    k_max = max(len(p) for p, _ in aggs)
    # Unequal-K aggregates are NaN-padded to a common K (the standard
    # pre-MAET convention for variable per-event cardinality).
    P = np.full((k_max, L), np.nan)
    W = np.full((k_max, L), np.nan)
    for j, (p, w) in enumerate(aggs):
        P[:len(p), j] = p
        W[:len(w), j] = w
    specs = flat_specs([P], r=r_inner, rel=False, sym=True, name='pitch')
    pb, wb, sb = unpack_pre_maet(bind_events([P], [W], L, rel_outer=True, specs=specs))
    attrs, ws, sp = [pb[0]], [wb[0]], [sb[0]]
    sigma, is_per, period = [SIGMA_PITCH], [True], [PERIOD]
    if flag is not None:
        attrs.append(np.array([[float(flag)]]))
        ws.append(np.array([[1.0]]))
        sp.extend(flat_specs([attrs[-1]], r=1, rel=False, sym=False, name='flag'))
        sigma.append(SIGMA_FLAG)
        is_per.append(False)
        period.append(0.0)
    return build_exp_tens(attrs, ws, specs=sp, sigma=sigma, is_per=is_per,
                          period=period, verbose=False)


def build_pair(c1, c2, flag=None, r_inner: int = 1):
    """Context density: two windowed beat aggregates, bound and nested."""
    return bound_density([aggregate(c1), aggregate(c2)], flag=flag,
                         r_inner=r_inner)


# --- the minimal cadential prototype (dyad skeleton) -------------------------
_DYAD_CHORDS = ([59.0, 65.0], [60.0, 64.0])      # B-F -> C-E


def query(flag=None, r_inner: int = 1):
    """Dyad-skeleton query density (with optional root-position flag)."""
    aggs = [(np.array(c), np.ones(len(c))) for c in _DYAD_CHORDS]
    return bound_density(aggs, flag=flag, r_inner=r_inner)


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
