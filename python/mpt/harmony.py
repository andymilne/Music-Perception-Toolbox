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
    """Spectral entropy of a weighted pitch multiset.

    Computes the Shannon entropy of the smoothed composite spectrum
    of a weighted pitch multiset. The spectrum is constructed by
    adding harmonics to each pitch via :func:`~mpt.spectra.add_spectra`,
    evaluating the resulting 1-D absolute non-periodic expectation
    tensor on a fine grid, normalising to a probability distribution,
    and computing the entropy.

    Spectral entropy aggregates the spectral pitch similarities of
    all pairs of sounds in the multiset: the greater the overlap
    of partials (after Gaussian smoothing), the lower the entropy.
    Lower entropy therefore indicates greater consonance.

    Parameters
    ----------
    p : array-like
        Pitch values in cents (absolute, not pitch classes).
    w : array-like or None
        Weights (``None`` for all ones).
    sigma : float
        Gaussian smoothing width in cents (typical: 6–15).
    spectrum : list or None
        Arguments for :func:`~mpt.spectra.add_spectra`.
    normalize : bool
        If True (default), divide by log₂(N) to give [0, 1].
    base : float
        Logarithm base (default 2 = bits).
    resolution : float
        Grid spacing in cents (default 1).

    Returns
    -------
    float
        Spectral entropy (lower = more consonant).

    References
    ----------
    Milne, A. J., Bulger, D., & Herff, S. A. (2017). Exploring the
    space of perfectly balanced rhythms and scales. *Journal of
    Mathematics and Music*, 11(2–3), 101–133.
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

    Measures the harmonicity of a weighted pitch multiset by
    cross-correlating its spectral expectation tensor with a harmonic
    template (a single complex tone). Two complementary measures:

    - *h_max*: maximum normalised cross-correlation (Milne, 2013).
      Cosine similarity at the best-matching transposition.
    - *h_entropy*: Shannon entropy of the cross-correlation treated
      as a probability distribution (Harrison & Pearce, 2020).

    Parameters
    ----------
    p : array-like
        Pitch values in cents (absolute, not pitch classes).
    w : array-like or None
        Weights (``None`` for all ones).
    sigma : float
        Gaussian smoothing width in cents (typical: 9–15).
    spectrum : list or None
        Arguments for :func:`~mpt.spectra.add_spectra` for the
        harmonic template.
    chord_spectrum : list or None
        Arguments for :func:`~mpt.spectra.add_spectra` for the
        chord. ``None`` = use pitches as given.
    normalize : bool
        If True (default), normalise entropy to [0, 1].
    base : float
        Logarithm base (default 2).
    resolution : float
        Grid spacing in cents (default 1).

    Returns
    -------
    h_max : float
        Maximum cross-correlation in [0, 1].
    h_entropy : float
        Entropy of the cross-correlation.

    References
    ----------
    Milne, A. J. (2013). *A computational model of the cognition
    of tonality*. PhD thesis, The Open University.

    Harrison, P. M. C. & Pearce, M. T. (2020). Simultaneous
    consonance in music perception and composition. *Psychological
    Review*, 127(2), 216–244.
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

    Measures the harmonicity of a weighted pitch multiset by
    evaluating the relative r-ad expectation tensor of a harmonic
    series at the multiset's interval vector. A high density at the
    chord's intervals indicates those intervals are likely to
    co-occur in a harmonic series.

    Parameters
    ----------
    p : array-like
        Pitch values in cents (length K ≥ 2, absolute).
    w : array-like or None
        Weights (``None`` for all ones).
    sigma : float
        Gaussian smoothing width in cents (typical: 9–15).
    spectrum : list or None
        Arguments for :func:`~mpt.spectra.add_spectra` for the
        harmonic template.
    duplicate : int
        Number of template copies (0 = auto = chord cardinality).
    normalize : str
        ``'none'`` (default), ``'gaussian'``, or ``'pdf'``.

    Returns
    -------
    float
        Harmonicity (non-negative; higher = more harmonic).

    References
    ----------
    Smit, E. A., Milne, A. J., Dean, R. T., & Weidemann, G.
    (2019). Perception of affect in unfamiliar musical chords.
    *PLOS ONE*, 14(6), e0218570.
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

    Computes the virtual pitch (fundamental) salience profile for a
    weighted pitch multiset by cross-correlating its spectral
    expectation tensor with a harmonic template. Peaks indicate
    strong virtual pitches — candidate fundamentals well-supported
    by the input spectrum.

    Parameters
    ----------
    p : array-like
        Pitch values in cents (absolute).
    w : array-like or None
        Weights (``None`` for all ones).
    sigma : float
        Gaussian smoothing width in cents.
    spectrum : list or None
        Arguments for :func:`~mpt.spectra.add_spectra` for the
        harmonic template.
    chord_spectrum : list or None
        Arguments for :func:`~mpt.spectra.add_spectra` for the
        chord. ``None`` = use pitches as given.
    resolution : float
        Grid spacing in cents (default 1).

    Returns
    -------
    vp_p : np.ndarray
        Candidate pitch values in cents.
    vp_w : np.ndarray
        Virtual pitch weights (normalised cross-correlation).

    References
    ----------
    Milne, A. J. (2013). *A computational model of the cognition
    of tonality*. PhD thesis, The Open University.
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
    """Sensory roughness of a weighted multiset of partials.

    Computes the total sensory roughness by summing the pairwise
    roughness contributions of all partial pairs, using Sethares'
    (1993) parameterisation of Plomp and Levelt's (1965) empirical
    dissonance curve.

    Frequencies must be in Hz. Use :func:`~mpt.convert.convert_pitch`
    to convert from other scales.

    Parameters
    ----------
    f : array-like
        Frequencies in Hz (positive).
    w : array-like or None
        Amplitudes/weights (``None`` for all ones).
    p_norm : float
        Norm exponent for combining pairwise roughnesses (default 1).
    average : bool
        If True, divide by the number of pairs (default False).

    Returns
    -------
    float
        Total (or average) roughness (non-negative).

    References
    ----------
    Sethares, W. A. (1993). Local consonance and the relationship
    between timbre and scale. *JASA*, 94(3), 1218–1228.
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
