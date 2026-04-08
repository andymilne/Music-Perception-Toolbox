"""Entropy measures for pitch and rhythm structures."""

from __future__ import annotations

import numpy as np
from scipy.fft import fft as _fft, ifft as _ifft

from .spectra import add_spectra
from .tensor import ExpTensDensity, build_exp_tens, eval_exp_tens


# ===================================================================
#  entropy_exp_tens
# ===================================================================


def entropy_exp_tens(
    p_or_dens,
    w=None,
    sigma=None,
    r=None,
    is_rel=None,
    is_per=None,
    period=None,
    *,
    spectrum: list | None = None,
    normalize: bool = True,
    base: float = 2.0,
    n_points_per_dim: int = 1200,
    x_min: float = float("nan"),
    x_max: float = float("nan"),
) -> float:
    """Shannon entropy (in bits) of an expectation tensor.

    Can accept either a precomputed :class:`ExpTensDensity` or raw
    parameters.

    Parameters
    ----------
    p_or_dens : array-like or ExpTensDensity
        Pitch values *or* a precomputed density struct.
    w, sigma, r, is_rel, is_per, period :
        Required when *p_or_dens* is not a struct.
    spectrum : list, optional
        Arguments for :func:`~mpt.spectra.add_spectra`.
    normalize : bool
        Divide by log_base(N) to get [0, 1] range.
    base : float
        Logarithm base (default 2).
    n_points_per_dim : int
        Grid resolution per dimension.
    x_min, x_max : float
        Domain bounds (required when ``is_per`` is False).

    Returns
    -------
    float
    """
    if isinstance(p_or_dens, ExpTensDensity):
        T = p_or_dens
        is_per = T.is_per
        period = T.period
    else:
        p = np.asarray(p_or_dens, dtype=np.float64).ravel()
        if w is None or sigma is None or r is None or is_rel is None or is_per is None or period is None:
            raise ValueError(
                "When p is not a precomputed density, all positional "
                "arguments (p, w, sigma, r, is_rel, is_per, period) are required."
            )
        if spectrum is not None:
            p, w = add_spectra(p, w, *spectrum)
        T = build_exp_tens(p, w, sigma, r, is_rel, is_per, period, verbose=False)

    # Construct query points
    if is_per:
        x = np.linspace(0, period, n_points_per_dim + 1)[:-1]
    else:
        if np.isnan(x_min) or np.isnan(x_max):
            raise ValueError("x_min and x_max must be specified when is_per is False.")
        if x_min >= x_max:
            raise ValueError("x_min must be less than x_max.")
        x = np.linspace(x_min, x_max, n_points_per_dim)

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
#  n_tuple_entropy
# ===================================================================


def n_tuple_entropy(
    p: np.ndarray,
    period: int,
    n: int = 1,
    *,
    sigma: float = 0.0,
    normalize: bool = True,
    base: float = 2.0,
) -> tuple[float, np.ndarray]:
    """Entropy of n-tuples of consecutive step sizes.

    Parameters
    ----------
    p : array-like of int
        Positions (non-negative integers < *period*).
    period : int
        Equal-division size.
    n : int
        Tuple size (default 1 = IOI entropy).
    sigma : float
        Circular Gaussian smoothing width (default 0).
    normalize : bool
        Divide by log_base(period^n) for [0, 1] range.
    base : float
        Logarithm base.

    Returns
    -------
    H : float
        Shannon entropy.
    tuples : (K, n) array
        The n-tuples of consecutive step sizes.
    """
    p = np.asarray(p, dtype=np.int64).ravel()
    period = int(period)
    n = int(n)

    p = np.sort(p % period)
    K = len(p)

    if len(np.unique(p)) != K:
        raise ValueError("p must not contain duplicate pitch classes (mod period).")
    if K < 2:
        raise ValueError(f"At least 2 events required (got {K}).")
    if n > K - 1:
        raise ValueError(f"n must not exceed K - 1 = {K - 1} (got n = {n}).")

    N = period
    total_bins = N**n
    if total_bins > 1e9:
        raise ValueError(
            f"period^n = {N}^{n} = {total_bins:.2e} exceeds 10^9."
        )

    # Build circulant rotations
    idx = (np.arange(K)[:, None] + np.arange(K)[None, :]) % K
    rotations = p[idx]  # K x K

    # Step sizes
    all_steps = np.diff(rotations, axis=0) % N  # (K-1) x K
    steps = all_steps[:n, :]  # n x K

    tuples_out = steps.T  # K x n

    # Linear index for histogram
    multipliers = N ** np.arange(n)  # (n,)
    lin_idx = (tuples_out @ multipliers).astype(np.intp)  # (K,)

    counts = np.bincount(lin_idx, minlength=total_bins).astype(np.float64)

    # Gaussian smoothing
    if sigma > 0:
        counts = _smooth_histogram(counts, N, n, sigma)

    total = np.sum(counts)
    if total == 0:
        return 0.0, tuples_out

    q = counts / total
    q = q[q > 0]

    H = float(-np.sum(q * np.log(q) / np.log(base)))

    if normalize:
        H /= np.log(total_bins) / np.log(base)

    return H, tuples_out


def _smooth_histogram(counts, N, n, sigma):
    """Separable circular Gaussian convolution on an n-D histogram."""
    # Build 1-D circular Gaussian kernel
    x = np.arange(N, dtype=np.float64)
    d = np.minimum(x, N - x)
    kernel = np.exp(-d**2 / (2 * sigma**2))
    kernel /= np.sum(kernel)
    kernel_fft = _fft(kernel)

    # Reshape to n-D
    if n == 1:
        hist_nd = counts.copy()
    else:
        hist_nd = counts.reshape([N] * n)

    # Separable convolution along each dimension
    for dim in range(n):
        shape = [1] * max(n, 1)
        shape[dim] = N
        k_fft = kernel_fft.reshape(shape)
        hist_nd = np.real(_ifft(_fft(hist_nd, axis=dim) * k_fft, axis=dim))

    return np.maximum(hist_nd.ravel(), 0)
