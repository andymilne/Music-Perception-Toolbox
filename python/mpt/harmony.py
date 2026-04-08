"""Harmony and consonance measures.

Spectral entropy, template harmonicity, tensor harmonicity,
sensory roughness, and virtual pitch extraction.
"""

from __future__ import annotations

import warnings

import numpy as np

from ._utils import validate_weights
from .spectra import add_spectra
from .tensor import build_exp_tens, eval_exp_tens


# ===================================================================
#  Spectral entropy
# ===================================================================


def spectral_entropy(
    p: np.ndarray,
    w: np.ndarray | None = None,
    sigma: float = 12.0,
    *,
    spectrum: list | None = None,
    normalize: bool = True,
    base: float = 2.0,
    resolution: float = 1.0,
) -> float:
    """Spectral entropy of a pitch set.

    Lower entropy → greater consonance (more spectral overlap).

    Parameters
    ----------
    p : array-like
        Pitch values in cents (absolute, not pitch classes).
    w : array-like or None
        Weights.
    sigma : float
        Gaussian smoothing width in cents.
    spectrum : list, optional
        Arguments for :func:`~mpt.spectra.add_spectra`, e.g.
        ``['harmonic', 24, 'powerlaw', 1]``. If ``None``, pitches
        are used as-is (suitable for empirical spectral peaks).
    normalize : bool
        Divide by log_base(N) for [0, 1] range.
    base : float
        Logarithm base.
    resolution : float
        Grid spacing in cents.

    Returns
    -------
    float

    Examples
    --------
    Spectral entropy with synthetic harmonics::

        H = mpt.spectral_entropy([0, 386.31, 701.96], None, 12,
                                  spectrum=['harmonic', 24, 'powerlaw', 1])

    From empirical audio peaks (no add_spectra needed)::

        f, w, _ = mpt.audio_peaks('audio/Piano_Cmin_open.wav')
        p = mpt.convert_pitch(f, 'hz', 'cents')
        H = mpt.spectral_entropy(p, w, 12)
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    w = validate_weights(w, len(p))
    p = p - np.min(p)

    if spectrum is not None:
        spec_p, spec_w = add_spectra(p, w, *spectrum)
    else:
        spec_p, spec_w = p.copy(), w.copy()

    T = build_exp_tens(spec_p, spec_w, sigma, 1, False, False, 1200, verbose=False)

    margin = 4 * sigma
    x = np.arange(0, np.max(spec_p) + margin + resolution, resolution)
    t = eval_exp_tens(T, x, verbose=False)

    total = np.sum(t)
    if total == 0:
        return 0.0

    q = t / total
    N = len(q)
    q = q[q > 0]

    H = float(-np.sum(q * np.log(q) / np.log(base)))
    if normalize:
        H /= np.log(N) / np.log(base)
    return H


# ===================================================================
#  Template harmonicity
# ===================================================================


def template_harmonicity(
    p: np.ndarray,
    w: np.ndarray | None = None,
    sigma: float = 12.0,
    *,
    spectrum: list | None = None,
    chord_spectrum: list | None = None,
    normalize: bool = True,
    base: float = 2.0,
    resolution: float = 1.0,
) -> tuple[float, float]:
    """Harmonicity via template cross-correlation.

    Cross-correlates the chord's spectrum with a harmonic template.

    Parameters
    ----------
    p : array-like
        Pitch values in cents.
    w : array-like or None
        Weights.
    sigma : float
        Gaussian smoothing width.
    spectrum : list, optional
        Template spectrum args. Default: ``['harmonic', 36, 'powerlaw', 1]``.
    chord_spectrum : list, optional
        Chord spectrum args. Default: ``None`` (no enrichment).
    normalize, base : float
        Entropy normalisation.
    resolution : float
        Grid spacing in cents.

    Returns
    -------
    h_max : float
        Maximum normalized cross-correlation (Milne 2013).
    h_entropy : float
        Entropy of the cross-correlation (Harrison 2020).

    Examples
    --------
    With synthetic spectrum on chord::

        spec = ['harmonic', 36, 'powerlaw', 1]
        h_max, h_ent = mpt.template_harmonicity([0, 386.31, 701.96],
                           None, 12, chord_spectrum=spec)

    From empirical audio peaks::

        f, w, _ = mpt.audio_peaks('audio/Piano_Cmin_open.wav')
        p = mpt.convert_pitch(f, 'hz', 'cents')
        h_max, h_ent = mpt.template_harmonicity(p, w, 12)
    """
    if spectrum is None:
        spectrum = ["harmonic", 36, "powerlaw", 1]

    p = np.asarray(p, dtype=np.float64).ravel()
    w = validate_weights(w, len(p))
    p = p - np.min(p)

    # Build template
    tmpl_p, tmpl_w = add_spectra(np.array([0.0]), np.array([1.0]), *spectrum)

    # Build chord spectrum
    if chord_spectrum is not None:
        chord_p, chord_w = add_spectra(p, w, *chord_spectrum)
    else:
        chord_p, chord_w = p.copy(), w.copy()

    # Build densities
    tmpl_dens = build_exp_tens(tmpl_p, tmpl_w, sigma, 1, False, False, 1200, verbose=False)
    chord_dens = build_exp_tens(chord_p, chord_w, sigma, 1, False, False, 1200, verbose=False)

    margin = 4 * sigma
    x_tmpl = np.arange(0, np.max(tmpl_p) + margin + resolution, resolution)
    x_chord = np.arange(0, np.max(chord_p) + margin + resolution, resolution)

    tmpl_vals = eval_exp_tens(tmpl_dens, x_tmpl, verbose=False)
    chord_vals = eval_exp_tens(chord_dens, x_chord, verbose=False)

    # Cross-correlation
    xcorr = np.convolve(chord_vals, tmpl_vals[::-1], mode="full")

    # Normalize (cosine similarity at each lag)
    norm_factor = np.sqrt(np.sum(chord_vals**2) * np.sum(tmpl_vals**2))
    xcorr_norm = xcorr / norm_factor if norm_factor > 0 else xcorr

    h_max = float(np.max(xcorr_norm))

    # Entropy
    q = xcorr_norm.copy()
    N = len(q)
    total = np.sum(q)
    if total > 0:
        q = q / total
    q = q[q > 0]
    h_entropy = float(-np.sum(q * np.log(q) / np.log(base)))
    if normalize:
        h_entropy /= np.log(N) / np.log(base)

    return h_max, h_entropy


# ===================================================================
#  Tensor harmonicity
# ===================================================================


def tensor_harmonicity(
    p: np.ndarray,
    w: np.ndarray | None = None,
    sigma: float = 12.0,
    *,
    spectrum: list | None = None,
    duplicate: int = 0,
    normalize: str = "none",
) -> float:
    """Harmonicity via expectation tensor lookup.

    Evaluates the relative r-ad tensor of a harmonic series at the
    chord's interval vector.

    Parameters
    ----------
    p : array-like
        Pitch values in cents (≥ 2 pitches).
    w : array-like or None
        Weights.
    sigma : float
        Gaussian smoothing width.
    spectrum : list, optional
        Template spectrum args.
        Default: ``['harmonic', 64, 'powerlaw', 1]``.
    duplicate : int
        Template duplication count (0 = auto = chord cardinality).
    normalize : str
        ``'none'``, ``'gaussian'``, or ``'pdf'``.

    Returns
    -------
    float
        Harmonicity (higher = more harmonic).
    """
    if spectrum is None:
        spectrum = ["harmonic", 64, "powerlaw", 1]

    p = np.asarray(p, dtype=np.float64).ravel()
    n_pitches = len(p)
    if n_pitches < 2:
        raise ValueError(f"At least 2 pitches required (got {n_pitches}).")

    dup = duplicate if duplicate > 0 else n_pitches
    if dup > 3:
        warnings.warn(
            f"duplicate = {dup}: computation time grows rapidly. "
            "Consider reducing to 3 or fewer."
        )

    tmpl_p, tmpl_w = add_spectra(
        np.zeros(dup), np.ones(dup), *spectrum
    )

    T = build_exp_tens(
        tmpl_p, tmpl_w, sigma, n_pitches, True, False, 1200, verbose=False
    )

    p_sorted = np.sort(p)
    intervals = p_sorted[1:] - p_sorted[0]  # (n_pitches - 1,)

    h = eval_exp_tens(T, intervals.reshape(-1, 1), normalize, verbose=False)
    return float(h[0])


# ===================================================================
#  Virtual pitches
# ===================================================================


def virtual_pitches(
    p: np.ndarray,
    w: np.ndarray | None = None,
    sigma: float = 12.0,
    *,
    spectrum: list | None = None,
    chord_spectrum: list | None = None,
    resolution: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Virtual pitch salience profile via template cross-correlation.

    Parameters
    ----------
    p : array-like
        Pitch values in cents.
    w : array-like or None
        Weights.
    sigma : float
        Gaussian smoothing width.
    spectrum : list, optional
        Template spectrum args. Default: ``['harmonic', 36, 'powerlaw', 1]``.
    chord_spectrum : list, optional
        Chord spectrum args. Default: ``None`` (no enrichment).
    resolution : float
        Grid spacing in cents.

    Returns
    -------
    vp_p : np.ndarray
        Candidate fundamental pitches (cents).
    vp_w : np.ndarray
        Virtual pitch weights (cosine similarity at each lag).

    Examples
    --------
    Virtual pitches of a JI major triad (synthetic spectrum)::

        spec = ['harmonic', 36, 'powerlaw', 1]
        vp_p, vp_w = mpt.virtual_pitches([0, 386.31, 701.96],
                          None, 12, chord_spectrum=spec)

    From empirical audio peaks::

        f, w, _ = mpt.audio_peaks('audio/Piano_Cmin_open.wav')
        p = mpt.convert_pitch(f, 'hz', 'cents')
        vp_p, vp_w = mpt.virtual_pitches(p, w, 12)

    MIDI input via convert_pitch::

        p = mpt.convert_pitch([60, 64, 67], 'midi', 'cents')
        spec = ['harmonic', 36, 'powerlaw', 1]
        vp_p, vp_w = mpt.virtual_pitches(p, None, 12, chord_spectrum=spec)
    """
    if spectrum is None:
        spectrum = ["harmonic", 36, "powerlaw", 1]

    p = np.asarray(p, dtype=np.float64).ravel()
    w = validate_weights(w, len(p))

    p_offset = float(np.min(p))
    p = p - p_offset

    tmpl_p, tmpl_w = add_spectra(np.array([0.0]), np.array([1.0]), *spectrum)

    if chord_spectrum is not None:
        chord_p, chord_w = add_spectra(p, w, *chord_spectrum)
    else:
        chord_p, chord_w = p.copy(), w.copy()

    tmpl_dens = build_exp_tens(tmpl_p, tmpl_w, sigma, 1, False, False, 1200, verbose=False)
    chord_dens = build_exp_tens(chord_p, chord_w, sigma, 1, False, False, 1200, verbose=False)

    margin = 4 * sigma
    step = resolution
    x_tmpl = np.arange(0, np.max(tmpl_p) + margin + step, step)
    x_chord = np.arange(0, np.max(chord_p) + margin + step, step)

    tmpl_vals = eval_exp_tens(tmpl_dens, x_tmpl, verbose=False)
    chord_vals = eval_exp_tens(chord_dens, x_chord, verbose=False)

    xcorr = np.convolve(chord_vals, tmpl_vals[::-1], mode="full")
    norm_factor = np.sqrt(np.sum(chord_vals**2) * np.sum(tmpl_vals**2))
    xcorr_norm = xcorr / norm_factor if norm_factor > 0 else xcorr

    n_tmpl = len(tmpl_vals)
    n_xcorr = len(xcorr_norm)
    lag_indices = np.arange(n_xcorr) - (n_tmpl - 1)
    vp_p = lag_indices * step + p_offset
    vp_w = xcorr_norm

    return vp_p, vp_w


# ===================================================================
#  Roughness
# ===================================================================


def roughness(
    f: np.ndarray,
    w: np.ndarray | None = None,
    *,
    p_norm: float = 1.0,
    average: bool = False,
) -> float:
    """Sensory roughness of a set of partials.

    Uses Sethares' (1993) parameterisation of Plomp & Levelt (1965).

    Parameters
    ----------
    f : array-like
        Frequencies in Hz (must be positive).
    w : array-like or None
        Amplitudes (linear, not dB). ``None`` for uniform.
    p_norm : float
        Norm exponent for combining pairwise roughnesses (default 1).
    average : bool
        If True, divide by C(K, 2).

    Returns
    -------
    float
        Total (or average) roughness.
    """
    f = np.asarray(f, dtype=np.float64).ravel()
    K = len(f)
    w = validate_weights(w, K)

    if np.any(f <= 0):
        raise ValueError("Frequencies must be positive.")

    # Sethares (1993) parameters
    Dstar = 0.24
    S1 = 0.0207
    S2 = 18.96
    C1 = 5.0
    C2 = -5.0
    A1 = -3.51
    A2 = -5.75

    f_diff = f[None, :] - f[:, None]  # K x K
    f_min = np.minimum(f[None, :], f[:, None])
    w_min = np.minimum(w[None, :], w[:, None])

    mask = f_diff.ravel() > 0
    f_diff = f_diff.ravel()[mask]
    f_min = f_min.ravel()[mask]
    w_min = w_min.ravel()[mask]

    s = Dstar / (S1 * f_min + S2)
    pair_rough = w_min * (C1 * np.exp(A1 * s * f_diff) + C2 * np.exp(A2 * s * f_diff))

    r = float(np.sum(pair_rough**p_norm) ** (1 / p_norm))

    if average and K >= 2:
        from math import comb
        r /= comb(K, 2)

    return r
