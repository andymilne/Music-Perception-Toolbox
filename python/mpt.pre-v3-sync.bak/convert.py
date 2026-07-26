"""Convert between pitch and frequency scales.

All conversions route through Hz as an intermediate representation.
"""

from __future__ import annotations

import numpy as np

_VALID_SCALES = ("hz", "midi", "cents", "mel", "bark", "erb", "greenwood")


def convert_pitch(
    values: np.ndarray | float,
    from_scale: str,
    to_scale: str,
) -> np.ndarray:
    """Convert between pitch and frequency scales.

    Supported scales (case-insensitive):

    * ``'hz'``        – Frequency in hertz.
    * ``'midi'``      – MIDI note number (A4 = 69, C4 = 60).
    * ``'cents'``     – Absolute cents (100 × MIDI note number).
    * ``'mel'``       – Mel scale (O'Shaughnessy 1987).
    * ``'bark'``      – Bark scale (Traunmüller 1990 approximation).
    * ``'erb'``       – ERB-rate scale (Glasberg & Moore 1990).
    * ``'greenwood'`` – Cochlear position (Greenwood 1961/1990).

    Parameters
    ----------
    values : array-like
        Values in the source scale.
    from_scale, to_scale : str
        Source and target scale names.

    Returns
    -------
    np.ndarray
        Converted values (same shape as *values*).
    """
    values = np.asarray(values, dtype=np.float64)
    src = from_scale.lower()
    tgt = to_scale.lower()

    if src not in _VALID_SCALES:
        raise ValueError(
            f"Unknown source scale '{from_scale}'. "
            f"Valid: {', '.join(_VALID_SCALES)}"
        )
    if tgt not in _VALID_SCALES:
        raise ValueError(
            f"Unknown target scale '{to_scale}'. "
            f"Valid: {', '.join(_VALID_SCALES)}"
        )

    if src == tgt:
        return values.copy()

    f = _to_hz(values, src)
    return _from_hz(f, tgt)


# -------------------------------------------------------------------
#  Internal helpers
# -------------------------------------------------------------------


def _to_hz(values: np.ndarray, scale: str) -> np.ndarray:
    if scale == "hz":
        return values
    if scale == "midi":
        return 440.0 * 2.0 ** ((values - 69.0) / 12.0)
    if scale == "cents":
        return 440.0 * 2.0 ** ((values - 6900.0) / 1200.0)
    if scale == "mel":
        return 700.0 * (10.0 ** (values / 2595.0) - 1.0)
    if scale == "bark":
        return 1960.0 * (values + 0.53) / (26.28 - values)
    if scale == "erb":
        return (10.0 ** (values / 21.4) - 1.0) / 0.00437
    if scale == "greenwood":
        return 165.4 * (10.0 ** (2.1 * values) - 0.88)
    raise ValueError(scale)  # pragma: no cover


def _from_hz(f: np.ndarray, scale: str) -> np.ndarray:
    if scale == "hz":
        return f
    if scale == "midi":
        return 69.0 + 12.0 * np.log2(f / 440.0)
    if scale == "cents":
        return 6900.0 + 1200.0 * np.log2(f / 440.0)
    if scale == "mel":
        return 2595.0 * np.log10(1.0 + f / 700.0)
    if scale == "bark":
        return 26.81 / (1.0 + 1960.0 / f) - 0.53
    if scale == "erb":
        return 21.4 * np.log10(0.00437 * f + 1.0)
    if scale == "greenwood":
        return np.log10(f / 165.4 + 0.88) / 2.1
    raise ValueError(scale)  # pragma: no cover
