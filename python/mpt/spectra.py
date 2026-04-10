"""Spectral enrichment — add partials to weighted pitch multisets."""

from __future__ import annotations

import numpy as np

from ._utils import validate_weights


def add_spectra(
    p: np.ndarray,
    w: np.ndarray | None,
    mode: str,
    *args,
    units: float = 1200.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Add spectral partials to a weighted pitch multiset.

    Five modes determine the partial positions; for the first four,
    a sub-option selects the weight decay law.

    Parameters
    ----------
    p : array-like
        Pitch values (in the units specified by *units*; default
        cents).
    w : array-like or None
        Weights (``None`` for uniform).
    mode : str
        ``'harmonic'``, ``'stretched'``, ``'freqlinear'``,
        ``'stiff'``, or ``'custom'``.
    *args
        Mode-specific arguments (N, decay type, decay parameter,
        etc.). See the MATLAB ``help addSpectra`` for full details.
    units : float
        Cents per unit (default 1200 = one octave per unit of
        log₂ frequency).

    Returns
    -------
    p_out : np.ndarray
        Pitch values including partials.
    w_out : np.ndarray
        Corresponding weights.
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    w = validate_weights(w, len(p))

    mode = mode.lower()

    if mode == "harmonic":
        if len(args) < 3:
            raise ValueError(
                "Usage: add_spectra(p, w, 'harmonic', N, weight_type, param)"
            )
        N = _validate_n(args[0])
        n = np.arange(1, N + 1, dtype=np.float64)
        offsets = units * np.log2(n)
        spec_w = _parse_weights(n, args[1], args[2])

    elif mode == "stretched":
        if len(args) < 4:
            raise ValueError(
                "Usage: add_spectra(p, w, 'stretched', N, beta, weight_type, param)"
            )
        N = _validate_n(args[0])
        beta = float(args[1])
        if beta <= 0:
            raise ValueError("beta must be a positive scalar.")
        n = np.arange(1, N + 1, dtype=np.float64)
        offsets = beta * units * np.log2(n)
        spec_w = _parse_weights(n, args[2], args[3])

    elif mode == "freqlinear":
        if len(args) < 4:
            raise ValueError(
                "Usage: add_spectra(p, w, 'freqlinear', N, alpha, weight_type, param)"
            )
        N = _validate_n(args[0])
        alpha = float(args[1])
        if alpha <= -1:
            raise ValueError("alpha must be greater than -1.")
        n = np.arange(1, N + 1, dtype=np.float64)
        ratios = (alpha + n) / (alpha + 1)
        if np.any(ratios <= 0):
            bad = int(np.argmax(ratios <= 0))
            raise ValueError(
                f"All frequency ratios must be positive. "
                f"With alpha = {alpha}, ratio for n = {bad + 1} is {ratios[bad]:.4g}."
            )
        offsets = units * np.log2(ratios)
        spec_w = _parse_weights(n, args[2], args[3])

    elif mode == "stiff":
        if len(args) < 4:
            raise ValueError(
                "Usage: add_spectra(p, w, 'stiff', N, B, weight_type, param)"
            )
        N = _validate_n(args[0])
        B = float(args[1])
        if B < 0:
            raise ValueError("B must be a non-negative scalar.")
        n = np.arange(1, N + 1, dtype=np.float64)
        ratios = (n * np.sqrt(1 + B * n**2)) / np.sqrt(1 + B)
        offsets = units * np.log2(ratios)
        spec_w = _parse_weights(n, args[2], args[3])

    elif mode == "custom":
        if len(args) < 2:
            raise ValueError(
                "Usage: add_spectra(p, w, 'custom', offsets, spec_w)"
            )
        offsets = np.asarray(args[0], dtype=np.float64).ravel()
        spec_w = np.asarray(args[1], dtype=np.float64).ravel()
        if offsets.size != spec_w.size:
            raise ValueError("offsets and spec_w must have the same length.")
    else:
        raise ValueError(
            f"Unknown mode '{mode}'. Use 'harmonic', 'stretched', "
            "'freqlinear', 'stiff', or 'custom'."
        )

    # Build output: each pitch gets every partial offset.
    # p is (M,), offsets is (K,) → broadcasting gives (M, K).
    p_matrix = p[:, None] + offsets[None, :]
    w_matrix = w[:, None] * spec_w[None, :]

    return p_matrix.ravel(), w_matrix.ravel()


# -------------------------------------------------------------------
#  Helpers
# -------------------------------------------------------------------


def _validate_n(N) -> int:
    N = int(N)
    if N < 1:
        raise ValueError("N must be a positive integer.")
    return N


def _parse_weights(
    n: np.ndarray, weight_type: str, param: float
) -> np.ndarray:
    """Compute spectral weights for partial numbers *n*."""
    wt = str(weight_type).lower()
    param = float(param)

    if wt == "powerlaw":
        if param < 0:
            raise ValueError("rho must be a non-negative scalar.")
        return 1.0 / n**param

    if wt == "geometric":
        if not 0 <= param <= 1:
            raise ValueError("tau must be a scalar in [0, 1].")
        return param ** (n - 1)

    raise ValueError(
        f"Unknown weight type '{weight_type}'. Use 'powerlaw' or 'geometric'."
    )
