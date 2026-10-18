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
#  Shared grid and cross-correlation helpers
# ===================================================================

#: Grid margin, in multiples of sigma, beyond the lowest and the highest
#: partial when ``per`` is False. At 6 sigma the omitted tail of each
#: Gaussian is below 1e-8 of its peak.
_MARGIN_SIGMAS = 6.0


def _periodic_count(period: float, step: float) -> int:
    """Number of grid points in one period, checking that ``step``
    divides ``period``."""
    if not (period > 0):
        raise ValueError("period must be positive.")
    n = int(round(period / step))
    if n < 2 or abs(n * step - period) > 1e-9 * period:
        raise ValueError(
            f"resolution ({step}) must divide the period ({period}) "
            "when per is True."
        )
    return n


def _grid(spec_max: float, sigma: float, step: float,
          per: bool, period: float) -> np.ndarray:
    """Evaluation grid for a spectrum whose lowest partial is at 0.

    Periodic: one period, ``0, step, ..., period - step``. Non-periodic:
    from ``-m`` to ``spec_max + m`` (to within one step), ``m`` being
    ``_MARGIN_SIGMAS * sigma``, so that every partial's Gaussian lies
    whole on the grid.
    """
    if not (step > 0):
        raise ValueError("resolution must be positive.")
    if per:
        return np.arange(_periodic_count(period, step)) * step
    m = _MARGIN_SIGMAS * sigma
    n = int(np.floor((spec_max + 2 * m) / step + 1e-9)) + 1
    return -m + np.arange(n) * step


def _template_xcorr(p, w, sigma, spectrum, chord_spectrum, step,
                    per, period):
    """Normalized cross-correlation of a chord's spectrum with a
    harmonic template.

    Returns ``(xcorr_norm, vp_p)``: the cosine similarity of the two
    smoothed spectra at each transposition of the template, and the
    pitch (or, if ``per``, pitch class) at which each transposition
    places the template's fundamental. Non-periodic: a full linear
    cross-correlation over both grids. Periodic: a circular
    cross-correlation over one period, with ``vp_p`` sorted.
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    w = validate_weights(w, len(p))
    p_offset = float(np.min(p))
    p = p - p_offset

    tmpl_p, tmpl_w = add_spectra(np.array([0.0]), np.array([1.0]),
                                 *spectrum)
    if chord_spectrum is not None:
        chord_p, chord_w = add_spectra(p, w, *chord_spectrum)
    else:
        chord_p, chord_w = p.copy(), w.copy()

    per = bool(per)
    period = float(period) if per else 1200.0
    tmpl_dens = build_exp_tens(tmpl_p, tmpl_w, sigma, 1, False, per,
                               period, verbose=False)
    chord_dens = build_exp_tens(chord_p, chord_w, sigma, 1, False, per,
                                period, verbose=False)

    x_tmpl = _grid(float(np.max(tmpl_p)), sigma, step, per, period)
    x_chord = _grid(float(np.max(chord_p)), sigma, step, per, period)
    tmpl_vals = np.ravel(eval_exp_tens(tmpl_dens, x_tmpl, verbose=False))
    chord_vals = np.ravel(eval_exp_tens(chord_dens, x_chord, verbose=False))

    if per:
        # xcorr[l] = sum_i chord[i + l] * tmpl[i]: the template shifted
        # up by l steps, its fundamental at l * step.
        xcorr = np.real(np.fft.ifft(np.fft.fft(chord_vals)
                                    * np.conj(np.fft.fft(tmpl_vals))))
        xcorr = np.maximum(xcorr, 0.0)   # remove FFT round-off below 0
        vp_p = np.mod(np.arange(len(xcorr)) * step + p_offset, period)
        order = np.argsort(vp_p, kind="stable")
        vp_p = vp_p[order]
        xcorr = xcorr[order]
    else:
        # Both grids start at -m, so lag k of the full correlation
        # places the template's fundamental (k - (n_tmpl - 1)) * step
        # above the chord's lowest pitch.
        xcorr = np.convolve(chord_vals, tmpl_vals[::-1], mode="full")
        n_tmpl = len(tmpl_vals)
        vp_p = (np.arange(len(xcorr)) - (n_tmpl - 1)) * step + p_offset

    norm_factor = np.sqrt(np.sum(chord_vals ** 2) * np.sum(tmpl_vals ** 2))
    xcorr_norm = xcorr / norm_factor if norm_factor > 0 else xcorr
    return xcorr_norm, vp_p


def _entropy(values, normalize, base):
    """Shannon entropy of non-negative values treated as a distribution,
    optionally divided by log_base(N), N the number of values."""
    q = np.asarray(values, dtype=np.float64).ravel()
    N = len(q)
    total = np.sum(q)
    if total == 0:
        return 0.0
    q = q / total
    q = q[q > 0]
    H = float(-np.sum(q * np.log(q)) / np.log(base))
    if normalize:
        H /= np.log(N) / np.log(base)
    return H


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
    per: bool = False,
    period: float = 1200.0,
) -> float:
    """Spectral entropy of a weighted pitch multiset.

    Computes the Shannon entropy of the smoothed composite spectrum
    of a weighted pitch multiset. The spectrum is constructed by
    adding harmonics to each pitch via :func:`~mpt.spectra.add_spectra`,
    evaluating the resulting 1-D absolute expectation tensor on a fine
    grid, normalising to a probability distribution, and computing the
    entropy.

    Spectral entropy aggregates the spectral pitch similarities of
    all pairs of sounds in the multiset: the greater the overlap
    of partials (after Gaussian smoothing), the lower the entropy.
    Lower entropy therefore indicates greater consonance.

    With ``per=False`` (the default) the spectrum is over pitch, and
    the grid runs from 6 sigma below the lowest partial to 6 sigma
    above the highest. With ``per=True`` it is over pitch class: every
    partial is folded into one ``period``, the Gaussian smoothing wraps
    around it, and the grid covers that period. This is the form used
    by Milne, Bulger, and Herff (2017).

    With ``normalize=True`` the entropy is divided by log(N), N the
    number of grid points. With ``per=False`` the grid spans the
    chord's spectrum, so the normalized value is an entropy relative
    to that span. For values that do not depend on the span, set
    ``normalize=False`` (the entropy on the grid, which is independent
    of its extent) or ``per=True`` (where N is fixed by the period).

    Parameters
    ----------
    p : array-like
        Pitch values in cents (absolute pitches; with ``per=True``,
        pitches or pitch classes).
    w : array-like or None
        Weights (``None`` for uniform).
    sigma : float
        Gaussian smoothing width in cents (typical: 6–15).
    spectrum : list or None
        Arguments for :func:`~mpt.spectra.add_spectra`.
    normalize : bool
        If True (default), divide by log(N) to give [0, 1]
        (see above).
    base : float
        Logarithm base (default 2 = bits).
    resolution : float
        Grid spacing in cents (default 1). With ``per=True`` it must
        divide ``period``.
    per : bool
        If True, compute over pitch class (default False).
    period : float
        Period in cents when ``per=True`` (default 1200).

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

    per = bool(per)
    period = float(period) if per else 1200.0
    T = build_exp_tens(spec_p, spec_w, sigma, 1, False, per, period,
                       verbose=False)
    x = _grid(float(np.max(spec_p)), sigma, resolution, per, period)
    t = eval_exp_tens(T, x, verbose=False)
    return _entropy(t, normalize, base)


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
    per: bool = False,
    period: float = 1200.0,
) -> tuple[float, float]:
    """Harmonicity via template cross-correlation.

    Measures the harmonicity of a weighted pitch multiset by
    cross-correlating its spectral expectation tensor with a harmonic
    template (a single complex tone). Two complementary measures:

    - *h_max*: maximum normalised cross-correlation (Milne, 2013).
      Cosine similarity at the best-matching transposition.
    - *h_entropy*: Shannon entropy of the cross-correlation treated
      as a probability distribution (Harrison & Pearce, 2020).

    With ``per=False`` (the default) the cross-correlation runs over
    pitch, each spectrum evaluated from 6 sigma below its lowest
    partial to 6 sigma above its highest. With ``per=True`` chord and
    template are folded into one ``period`` and the cross-correlation
    is circular, over pitch class.

    With ``normalize=True`` *h_entropy* is divided by log(N), N the
    number of transpositions in the cross-correlation. With
    ``per=False`` N grows with the span of the chord's spectrum, so
    the normalized value is an entropy relative to that span. For
    values that do not depend on the span, set ``normalize=False`` or
    ``per=True`` (where N is fixed by the period).

    Parameters
    ----------
    p : array-like
        Pitch values in cents (absolute, not pitch classes).
    w : array-like or None
        Weights (``None`` for uniform).
    sigma : float
        Gaussian smoothing width in cents (typical: 9–15).
    spectrum : list or None
        Arguments for :func:`~mpt.spectra.add_spectra` for the
        harmonic template.
    chord_spectrum : list or None
        Arguments for :func:`~mpt.spectra.add_spectra` for the
        chord. ``None`` = use pitches as given.
    normalize : bool
        If True (default), divide the entropy by log(N) to give [0, 1]
        (see above).
    base : float
        Logarithm base (default 2).
    resolution : float
        Grid spacing in cents (default 1). With ``per=True`` it must
        divide ``period``.
    per : bool
        If True, compute over pitch class (default False).
    period : float
        Period in cents when ``per=True`` (default 1200).

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

    xcorr_norm, _ = _template_xcorr(p, w, sigma, spectrum,
                                    chord_spectrum, resolution, per,
                                    period)
    h_max = float(np.max(xcorr_norm))
    h_entropy = _entropy(xcorr_norm, normalize, base)
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
        Weights (``None`` for uniform).
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
    per: bool = False,
    period: float = 1200.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Virtual pitch salience profile via template cross-correlation.

    Computes the virtual pitch (fundamental) salience profile for a
    weighted pitch multiset by cross-correlating its spectral
    expectation tensor with a harmonic template. Peaks indicate
    strong virtual pitches — candidate fundamentals well-supported
    by the input spectrum.

    With ``per=False`` (the default) the profile runs over pitch, each
    spectrum evaluated from 6 sigma below its lowest partial to 6 sigma
    above its highest. With ``per=True`` chord and template are folded
    into one ``period``, the cross-correlation is circular, and the
    profile is over pitch class, one value per grid point of the
    period, sorted by pitch class.

    Parameters
    ----------
    p : array-like
        Pitch values in cents (absolute).
    w : array-like or None
        Weights (``None`` for uniform).
    sigma : float
        Gaussian smoothing width in cents.
    spectrum : list or None
        Arguments for :func:`~mpt.spectra.add_spectra` for the
        harmonic template.
    chord_spectrum : list or None
        Arguments for :func:`~mpt.spectra.add_spectra` for the
        chord. ``None`` = use pitches as given.
    resolution : float
        Grid spacing in cents (default 1). With ``per=True`` it must
        divide ``period``.
    per : bool
        If True, compute over pitch class (default False).
    period : float
        Period in cents when ``per=True`` (default 1200).

    Returns
    -------
    vp_p : np.ndarray
        Candidate pitch values in cents (pitch classes in
        ``[0, period)`` when ``per=True``).
    vp_w : np.ndarray
        Virtual pitch weights (normalised cross-correlation).

    References
    ----------
    Milne, A. J. (2013). *A computational model of the cognition
    of tonality*. PhD thesis, The Open University.
    """
    if spectrum is None:
        spectrum = ["harmonic", 36, "powerlaw", 1]

    vp_w, vp_p = _template_xcorr(p, w, sigma, spectrum, chord_spectrum,
                                 resolution, per, period)
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
        Amplitudes/weights (``None`` for uniform).
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
