"""phase_encoding.py

Symbolic rendering of Reich's *Piano Phase* (1967) for Analyses 3.1-3.3,
to the manuscript's specification (Section on the Piano Phase encoding):

* Both pianos play the twelve-note cell E5, F#5, B5, C#6, D6, F#5, E5,
  C#6, B5, F#5, D6, C#6 in even notes. Piano 1 holds a fixed inter-onset
  interval throughout; Piano 2's twelve shifts advance the inter-voice
  phase k (in note-durations) from one integer to the next across each
  shift and hold it between shifts.
* Each advance interpolates k with a smoothstep (the cubic
  s(x) = 3x^2 - 2x^3 for x in [0, 1]), an S-shaped ramp beginning and
  ending with zero slope. Piano 2's instantaneous inter-onset interval,
  set by the phase's rate of change, dips below the base interval while
  k is moving and returns to it on each hold.
* The base inter-onset interval is 137.85 ms, the piece ~617 s
  (~8,965 events over the two voices), and the peak tempo deviation
  1.75% -- which fixes the shift duration: the smoothstep's peak slope
  is 1.5/T, so T = 1.5 * BASE_IOI / 0.0175 ~= 11.8 s.

The manuscript's schedule of hold and accelerando lengths is transcribed
from a reference recording (Steve Reich Ensemble, *Early Works*,
Nonesuch 1987); that transcription is not reproduced here, so this
reconstruction approximates it with a uniform schedule (LEAD_S, GAP_S
below) matching the published figures' shift centres. All other
quantities follow the manuscript exactly.

Exports
-------
CELL : (12,) int ndarray -- the canonical cell, MIDI semitones.
NC : int -- pulses per cell (12).
BASE_IOI : float -- steady inter-onset interval, seconds (0.13785).
N_REPS_V1 : int -- Piano-1 cells in the rendered piece (373).
render_voice(v) -> (pitch, onset) for v in {1, 2}.
render_piece() -> (pitch, onset, voice) -- both voices pooled, sorted.
lag_at(x) -> phase k (in pulses) at time x measured in CELLS.
shift_centre_times() -> (12,) ndarray of accelerando centres, seconds.
"""
from __future__ import annotations
import numpy as np

# --- canonical cell (E5, F#5, B5, C#6, D6, F#5, E5, C#6, B5, F#5, D6, C#6) ---
CELL = np.array([76, 78, 83, 85, 86, 78, 76, 85, 83, 78, 86, 85], dtype=int)
NC = 12

# --- manuscript constants ----------------------------------------------------
BASE_IOI  = 0.13785         # seconds; base inter-onset interval (manuscript)
PEAK_DEV  = 0.0175          # peak tempo deviation (manuscript: 1.75%)
N_SHIFTS  = 12              # one whole cell of phase across the piece
N_REPS_V1 = 373             # Piano-1 cells: 373 * 12 * 137.85 ms ~= 617 s

# Smoothstep peak slope is 1.5/T, and the peak tempo deviation is
# BASE_IOI * (dk/dt)_max, so the shift duration follows from the manuscript:
SHIFT_DUR = 1.5 * BASE_IOI / PEAK_DEV        # ~= 11.8 s per accelerando

# Uniform stand-in for the recording-transcribed schedule (see docstring):
LEAD_S = 40.0               # seconds of unison before the first shift
GAP_S  = 35.8               # seconds of steady phase between shifts

CELL_DUR = NC * BASE_IOI
T_END    = N_REPS_V1 * CELL_DUR

_SHIFT_STARTS = LEAD_S + np.arange(N_SHIFTS) * (SHIFT_DUR + GAP_S)
_SHIFT_CENTRES = _SHIFT_STARTS + SHIFT_DUR / 2.0


def shift_centre_times() -> np.ndarray:
    """Centres of the twelve accelerandi, in seconds."""
    return _SHIFT_CENTRES.copy()


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return 3.0 * x ** 2 - 2.0 * x ** 3


def lag_at(x_cells):
    """Phase k of Piano 2 ahead of Piano 1 (pulses) at time x_cells,
    measured in Piano-1 cells: the sum of the twelve smoothstep ramps."""
    t = np.asarray(x_cells, dtype=float) * CELL_DUR
    k = np.zeros_like(t, dtype=float)
    for s in _SHIFT_STARTS:
        k = k + _smoothstep((t - s) / SHIFT_DUR)
    return k if k.shape else float(k)


# --- Piano 2's cumulative pulse count: Phi2(t) = t/BASE_IOI + k(t) -----------
_GRID_DT = 0.0002                                  # 0.2 ms inversion grid
_TG = np.arange(0.0, T_END + _GRID_DT, _GRID_DT)
_PHI2 = _TG / BASE_IOI + lag_at(_TG / CELL_DUR)


def render_voice(v: int):
    """Render voice v in {1, 2} to (pitch, onset) arrays (MIDI, seconds)."""
    if v == 1:
        m = np.arange(N_REPS_V1 * NC)
        onset = m * BASE_IOI
    elif v == 2:
        n_pulses = int(np.floor(_PHI2[-1]))
        m = np.arange(n_pulses)
        onset = np.interp(m, _PHI2, _TG)   # invert the (increasing) phase
    else:
        raise ValueError("voice must be 1 or 2")
    pitch = CELL[m % NC].astype(float)
    return pitch, onset


def render_piece():
    """Both voices pooled and time-sorted: (pitch, onset, voice)."""
    p1, t1 = render_voice(1)
    p2, t2 = render_voice(2)
    pitch = np.concatenate([p1, p2])
    onset = np.concatenate([t1, t2])
    voice = np.concatenate([np.ones_like(t1), 2.0 * np.ones_like(t2)])
    order = np.argsort(onset, kind='stable')
    return pitch[order], onset[order], voice[order]


if __name__ == '__main__':
    print(f'cell duration {CELL_DUR:.4f} s; piece {T_END:.1f} s; '
          f'shift duration {SHIFT_DUR:.2f} s')
    p1, t1 = render_voice(1)
    p2, t2 = render_voice(2)
    dt = np.diff(t2)
    print(f'events: Piano 1 {len(t1)}, Piano 2 {len(t2)}, '
          f'total {len(t1) + len(t2)}')
    print(f'Piano 2 IOI {dt.min()*1000:.2f}..{dt.max()*1000:.2f} ms '
          f'(peak deviation {(BASE_IOI/dt.min() - 1)*100:.2f}%)')
    print('shift centres (s):', np.round(shift_centre_times(), 1))
    print('final k:', round(float(lag_at(N_REPS_V1)), 4))
