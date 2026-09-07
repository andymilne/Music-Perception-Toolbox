"""jmm_data.py — the three works analysed in the JMM article, as MAET inputs.

The demos in this folder reproduce the worked examples of the article
(Milne, *Music Perception Toolbox*, Journal of Mathematics and Music) and
its Online Supplement. This module holds the data they share, so that
each demo starts from the same encoding the article used:

* :func:`bwv347_grid` — Bach, *Ich dank dir, lieber Herre* (BWV 347),
  played through with the bars 1–4 repeat expanded, sampled on the
  sixteenth-note grid: the pitch sounding in each of the four voices at
  every grid point. The score is read from ``data/bwv347.musicxml`` with
  ``mpt.read_score`` (the article used the same score from the music21
  corpus; the two encodings agree to the note).
* :func:`bwv347_bar` — the played-through bar number of a grid time.
* :mod:`piano_phase` — Reich, *Piano Phase*: the twelve-note cell, the
  two voices rendered from the article's constants (base inter-onset
  interval, peak tempo deviation, smoothstep accelerandi), and the phase
  as a function of time.
* :func:`acknowledgement` — Coltrane, *Acknowledgement* (Theme 2): the
  melody as ``(pitch, onset_beats)``, read from ``data/theme_2.mid`` with
  ``mpt.read_score``. The transcription is not part of the toolbox
  distribution; place your own MIDI transcription at that path.
"""
from __future__ import annotations

import os

import numpy as np

import mpt

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
GRID_STEP_QN = 0.25          # sixteenth note: the smallest value in BWV 347


# ---------------------------------------------------------------------------
#  Bach, BWV 347
# ---------------------------------------------------------------------------

def bwv347_notes():
    """The note table of the played-through chorale (``mpt.read_score``)."""
    return mpt.read_score(os.path.join(DATA_DIR, "bwv347.musicxml"))


def bwv347_grid(grid_step=GRID_STEP_QN):
    """BWV 347 on the sixteenth-note grid.

    Returns
    -------
    times : (N,) ndarray
        Grid times in quarter notes, 0 to ~68 QN (272 points at the
        default step).
    pitches_satb : (N, 4) ndarray
        MIDI pitch sounding in soprano, alto, tenor, bass at each grid
        point (NaN where a voice rests; BWV 347 has no rests).
    bars : (N,) int ndarray
        Played-through bar number: 0 for the one-quarter pickup, then
        1–17.
    """
    t = bwv347_notes()
    n_parts = len(t["part_names"])
    t_end = float(np.max(t["onset_beats"] + t["duration_beats"]))
    times = np.arange(0.0, t_end, grid_step)
    pitches = np.full((times.size, n_parts), np.nan)
    for part in range(1, n_parts + 1):
        m = t["part"] == part
        on = t["onset_beats"][m]
        off = on + t["duration_beats"][m]
        pit = t["pitch"][m]
        for i, g in enumerate(times):
            k = np.nonzero((on <= g + 1e-9) & (g < off - 1e-9))[0]
            if k.size:
                pitches[i, part - 1] = pit[k[0]]
    bars = np.array([bwv347_bar(g) for g in times], dtype=int)
    return times, pitches, bars


def bwv347_bar(t):
    """Played-through bar of a grid time in quarter notes: the chorale has
    a one-quarter pickup at 0–1 QN, then 4/4 bars from 1 QN."""
    if t < 1.0:
        return 0
    return int((t - 1.0) // 4.0) + 1


def bwv347_fermata_spans():
    """``(start, end)`` quarter-note spans of the fermata-bearing notes of
    the played-through chorale, from the ``fermata`` column of the note
    table. Analysis 1.4 raises the weight of every eighth-note event under
    a fermata by half."""
    t = bwv347_notes()
    f = t["fermata"] == 1
    return sorted({(float(a), float(a + d))
                   for a, d in zip(t["onset_beats"][f], t["duration_beats"][f])})


# ---------------------------------------------------------------------------
#  Coltrane, Acknowledgement (Theme 2)
# ---------------------------------------------------------------------------

def acknowledgement(path=None):
    """``(pitch, onset_beats)`` of the melody, one row per note-on, in onset
    order, from a monophonic MIDI transcription (default
    ``data/theme_2.mid``)."""
    path = path or os.path.join(DATA_DIR, "theme_2.mid")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. The transcription of Acknowledgement is "
            f"not distributed with the toolbox; place a monophonic MIDI "
            f"transcription of Theme 2 at that path (or pass its path).")
    t = mpt.read_score(path)
    order = np.argsort(t["onset_beats"], kind="stable")
    return t["pitch"][order], t["onset_beats"][order]
