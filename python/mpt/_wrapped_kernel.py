"""Wrapped-Gaussian 1D kernel with automatic image-sum vs Fourier dispatch.

The abs-per full-image kernel is a product across coordinates of the 1D
wrapped Gaussian

    theta(d) = sum_{n in Z} exp(-(d + n P)^2 / (2 tau^2))

for a width parameter ``tau``. Two exact representations of the sum
converge at reciprocal rates as ``sigma/P`` (or ``tau/P``) varies:

- **Image-sum:** truncate the lattice sum at |n| <= L. Cheap when tau is
  small (few images have nontrivial mass); each term is one exp.
- **Fourier (Poisson-summed):** truncate the reciprocal series
  ``theta(d) = (tau sqrt(2 pi) / P) sum_m exp(-2 pi^2 tau^2 m^2 / P^2)
  cos(2 pi m d / P)`` at |m| <= M. Cheap when tau is large (envelope
  drops quickly).

The 1D crossover between the two, derived analytically from the
truncation formulas at a shared accuracy floor, sits at
``sigma/P ~ 0.2`` for the density-kernel convention (``tau = sigma``)
and ``sigma/P ~ 0.24`` for the overlap-kernel convention
(``tau = sigma sqrt(2)``, exponent has ``4 sigma^2`` in the denominator).
Below the crossover image-sum wins; above it, Fourier wins. Both compute
the same wrapped Gaussian, so the choice is transparent to callers.

Conventions used throughout the toolbox:

- **Overlap kernel** (inner products): exponent
  ``exp(-d^2 / (4 sigma^2))`` --- pass ``exponent_denominator=4``.
- **Density kernel** (density evaluation): exponent
  ``exp(-d^2 / (2 sigma^2))`` --- pass ``exponent_denominator=2``.

The two share the same wrapped structure and Fourier envelope; the
denominator only rescales the effective width by ``sqrt(2)``.
"""

from __future__ import annotations

import math

import numpy as np


def _image_count_L(sigma: float, period: float, truncation_sigmas,
                   exponent_denominator: int) -> int:
    """Number of periodic images per side. The kernel is
    ``exp(-x^2 / (e_d sigma^2))``. After nearest-image reduction the
    first-omitted image at ``|d + n P| >= (L + 1/2) P`` is bounded by
    ``exp(-((L + 1/2) P)^2 / (e_d sigma^2)) < tol``. Solve:

        L + 1/2 > (sigma / P) sqrt(-e_d log tol),
        L > (sigma / P) sqrt(-e_d log tol) - 1/2.

    ``exponent_denominator`` is 4 for overlap, 2 for density.
    """
    if sigma <= 0.0 or period <= 0.0:
        return 0
    from ._defaults import truncation_floor
    floor = truncation_floor(truncation_sigmas)
    if floor <= 0.0 or floor >= 1.0:
        floor = 1e-15
    rhs = (sigma / period) * math.sqrt(
        -math.log(floor) * float(exponent_denominator)
    )
    return max(0, int(math.ceil(rhs - 0.5)))


def _fourier_count_M(sigma: float, period: float, truncation_sigmas,
                     exponent_denominator: int) -> int:
    """Number of Fourier modes per side. The Poisson-summed envelope at
    mode ``m`` is ``exp(-alpha m^2)`` with
    ``alpha = pi^2 e_d sigma^2 / P^2``. First-omitted mode below the
    floor:

        M > sqrt(-log tol / alpha)
          = (P / (pi sigma)) sqrt(-log tol / e_d).
    """
    if sigma <= 0.0 or period <= 0.0:
        return 1
    from ._defaults import truncation_floor
    floor = truncation_floor(truncation_sigmas)
    if floor <= 0.0 or floor >= 1.0:
        floor = 1e-15
    alpha = math.pi ** 2 * float(exponent_denominator) * sigma ** 2 / period ** 2
    return max(1, int(math.ceil(math.sqrt(-math.log(floor) / alpha))))


def _prefer_fourier(sigma: float, period: float, truncation_sigmas,
                    exponent_denominator: int) -> bool:
    """Cost-based crossover between image-sum and Fourier at fixed
    accuracy floor. Each per-component 1D kernel evaluation costs
    ``2 L + 1`` terms for image-sum and ``M`` (real cosine sum, one
    trigonometric per mode plus the DC constant) for Fourier. Prefer
    Fourier when its grid is narrower.
    """
    L = _image_count_L(sigma, period, truncation_sigmas,
                       exponent_denominator)
    M = _fourier_count_M(sigma, period, truncation_sigmas,
                         exponent_denominator)
    return M < (2 * L + 1)


def wrapped_gaussian_1d(d: np.ndarray, sigma: float, period: float,
                        truncation_sigmas, *,
                        exponent_denominator: int) -> np.ndarray:
    """Evaluate the 1D wrapped Gaussian
    ``theta(d) = sum_n exp(-(d + n P)^2 / (e_d sigma^2))``
    with the cheaper of image-sum and Fourier at the requested accuracy
    floor. ``d`` may have any shape; the return has the same shape.

    Nearest-image reduction is applied so image-sum truncation is
    correct for arbitrary input; Fourier is periodic and needs no
    reduction.
    """
    d = np.asarray(d)
    if _prefer_fourier(sigma, period, truncation_sigmas,
                       exponent_denominator):
        # Poisson-summed form. Kernel exp(-d^2 / (e_d sigma^2)) has
        # Fourier transform sqrt(pi e_d) sigma exp(-pi^2 e_d sigma^2
        # k^2 / (4 pi^2)) = sqrt(pi e_d) sigma exp(-e_d sigma^2 k^2 / 4)
        # evaluated at k = 2 pi m / P; the Poisson sum divides by P.
        # Combining, the prefactor is (sqrt(pi e_d) sigma) / P and the
        # envelope is exp(-pi^2 e_d sigma^2 m^2 / P^2).
        M = _fourier_count_M(sigma, period, truncation_sigmas,
                             exponent_denominator)
        alpha = (math.pi ** 2 * float(exponent_denominator)
                 * float(sigma) ** 2 / float(period) ** 2)
        m_arr = np.arange(1, M + 1, dtype=np.float64)
        env = np.exp(-alpha * m_arr * m_arr)
        two_pi_over_P = 2.0 * math.pi / float(period)
        prefactor = (math.sqrt(math.pi * float(exponent_denominator))
                     * float(sigma) / float(period))
        # cos(2 pi m d / P) broadcast: d shape (...,) -> (..., M).
        phase = two_pi_over_P * m_arr * d[..., None]
        return prefactor * (1.0 + 2.0 * (env * np.cos(phase)).sum(axis=-1))
    # Image-sum: reduce to nearest image, then truncated lattice sum.
    L = _image_count_L(sigma, period, truncation_sigmas,
                       exponent_denominator)
    d_red = d - period * np.floor(d / period + 0.5)
    inv = 1.0 / (float(exponent_denominator) * float(sigma) ** 2)
    if L == 0:
        return np.exp(-d_red * d_red * inv)
    # Unrolled image loop. The previous vectorised form built a
    # (..., 2L+1) tensor and squared/exp'd it in one shot, which is
    # tidy but forces peak memory to grow as (2L+1) x the base shape
    # and blocks NumPy from reusing intermediates. Loop-per-image with
    # in-place accumulation keeps working memory at the base shape
    # (theta + one 1-image scratch) and hands NumPy a fresh, fused
    # exp per iteration, which is measurably faster (2-3x at K=50,
    # r=2, sigma/P=0.10 on the whole cossim call) while giving bit-
    # identical output.
    theta = np.exp(-d_red * d_red * inv)
    scratch = np.empty_like(theta)
    for n in range(1, L + 1):
        shift = float(n) * float(period)
        np.add(d_red, shift, out=scratch)
        scratch *= scratch
        scratch *= -inv
        np.exp(scratch, out=scratch)
        theta += scratch
        np.subtract(d_red, shift, out=scratch)
        scratch *= scratch
        scratch *= -inv
        np.exp(scratch, out=scratch)
        theta += scratch
    return theta
