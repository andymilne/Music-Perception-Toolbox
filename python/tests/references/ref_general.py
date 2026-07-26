"""Independent reference for the wrapped-Gaussian kernel sum.

Purpose. Provide slow, transparent implementations of the periodic
kernel sum

    K(delta) = sum_{n in Lambda} exp(-Q(delta + P n) / (4 sigma^2)),

for absolute-periodic (``Q = I``, r-cube lattice) and relative-periodic
(``Q = I - (1/r) 1 1^T``, coset lattice ``Z^r / Z*1``). This kernel is
the object called "full-image" or "(C)" in the handoff.

Weighted event-pair sums are provided:

    Overlap(X, Y) = sum_{j,k} W^X_j W^Y_k K(c^X_j - c^Y_k),

returned in the raw kernel-sum convention (no ``(sigma sqrt pi)^r`` or
``(2 sigma sqrt(pi/r) / P)`` prefactors). Physical overlaps differ from
this by a global constant depending on ``(r, sigma, P)`` but not on the
event data; cosine similarities are invariant under it, and per-mode
prefactors can be reintroduced by callers when a true overlap value is
needed.

Two routes per mode:

- ``overlap_all_image_*`` : direct truncated lattice sum, over the
  coset ``Z^{r-1}`` (rel) or the full ``Z^r`` (abs), truncation set from
  ``tol``. This is the reference route.

- ``overlap_spectral_*`` : Poisson-summed Fourier form of the same
  object, truncation set from ``tol``. Cross-check against lattice.

A brute-force quadrature reference is also provided for absolute-
periodic (where prefactors are unambiguous). Rel-per quadrature is not
provided at the reference-module level because the (``r-1``)-slice
normalisation depends on ``Q``'s determinant on that slice, which makes
the constant of proportionality between quadrature and the raw kernel
sum ``r``-dependent; the two routes above already cross-check each
other exactly, which suffices.

Coordinate conventions match ``mpt._tensor.build`` / ``_tensor.dispatch``:
tuples are rows of a ``(K, r)`` matrix of centres in the periodic
coordinate ``[0, P)``; weights are 1-D over tuples; ``sigma`` is the
Gaussian width; ``P`` is the period.

Nothing in this module imports from ``mpt``. Any accidental coupling
would defeat its purpose as an independent reference.
"""

from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _truncation_L(sigma: float, period: float, tol: float) -> int:
    """Number of periodic images per side to include so the first omitted
    term is below ``tol`` for a kernel of the form
    ``exp(-(delta + n P)^2 / (4 sigma^2))`` — the absolute-periodic
    decay rate along each component.
    """
    if sigma <= 0.0 or period <= 0.0:
        return 0
    if tol <= 0.0 or tol >= 1.0:
        tol = 1e-15
    rhs = 2.0 * sigma / period * math.sqrt(-math.log(tol))
    return max(0, int(math.ceil(rhs - 0.5)))


def _truncation_L_rel(sigma: float, period: float, tol: float) -> int:
    """Rel-per truncation.

    Two considerations combine. First, the effective decay rate along
    the free lattice direction is slower than abs-per because ``Q``'s
    smallest non-null eigendirection contributes ``(P n)^2 / 2`` at
    r=2 (and similar scaling at higher r). Second, even after per-
    component ``[-P/2, P/2]`` reduction of ``delta``, the "best image"
    in the projected lattice can be at ``|n_i| = 1`` (not zero),
    because ``Q`` depends on projected differences rather than
    individual components. So we floor at ``L >= 1`` regardless of
    the decay bound:

        exp(-(P L)^2 / (8 sigma^2)) < tol
        L > (2 sqrt(2) sigma / P) sqrt(-log tol),
        L >= 1.

    Empirically this dominates the required truncation across
    (sigma/P, r) grids tested.
    """
    if sigma <= 0.0 or period <= 0.0:
        return 1
    if tol <= 0.0 or tol >= 1.0:
        tol = 1e-15
    rhs = 2.0 * math.sqrt(2.0) * sigma / period * math.sqrt(-math.log(tol))
    return max(1, int(math.ceil(rhs - 0.5)))


def _spectral_M(sigma: float, period: float, tol: float) -> int:
    """Number of Fourier modes per side so the first omitted mode is
    below ``tol`` for the envelope ``exp(-alpha m^2)`` with
    ``alpha = 4 pi^2 sigma^2 / P^2``.
    """
    if sigma <= 0.0 or period <= 0.0:
        return 1
    alpha = 4.0 * math.pi ** 2 * sigma * sigma / (period * period)
    return max(1, int(math.ceil(math.sqrt(-math.log(max(tol, 1e-300)) / alpha))))


def _rel_Q(v: np.ndarray) -> np.ndarray:
    """Return ``v^T M v`` with ``M = I - (1/r) 1 1^T`` (rank r-1) for
    each row of a ``(..., r)`` array.
    """
    v = np.asarray(v, dtype=np.float64)
    mean = v.mean(axis=-1, keepdims=True)
    d = v - mean
    return (d * d).sum(axis=-1)


# ---------------------------------------------------------------------------
# Absolute-periodic: kernel sum K(delta) and event-pair overlap
# ---------------------------------------------------------------------------


def kernel_all_image_abs(
    delta: np.ndarray, sigma: float, period: float, tol: float = 1e-15,
) -> np.ndarray:
    """Absolute-periodic kernel sum.

    ``K(delta) = prod_m sum_{n=-L..L} exp(-(delta_m + n P)^2 / (4 sigma^2))``,
    for an array of ``delta`` with shape ``(..., r)``. Returns shape
    ``(...,)``.
    """
    delta = np.asarray(delta, dtype=np.float64)
    # Reduce to nearest image per component: delta -> delta mod P in
    # [-P/2, P/2]. This puts the dominant kernel term at n=0 for every
    # component, so the truncation bound derived for |delta| <= P/2
    # applies. Without this, a raw delta near +/-P falls entirely on
    # the n=+/-1 image and L=0 misses it.
    delta = (delta + 0.5 * period) % period - 0.5 * period
    L = _truncation_L(sigma, period, tol)
    coef = -1.0 / (4.0 * sigma * sigma)
    # Sum over n=-L..L, applied elementwise then multiplied across r.
    n_grid = np.arange(-L, L + 1, dtype=np.float64)  # (2L+1,)
    # Shape (..., r, 2L+1)
    shifted = delta[..., :, None] + period * n_grid[None, :]
    theta = np.exp(coef * shifted * shifted).sum(axis=-1)  # (..., r)
    return theta.prod(axis=-1)  # (...,)


def overlap_all_image_abs(
    centres_x: np.ndarray, weights_x: np.ndarray,
    centres_y: np.ndarray, weights_y: np.ndarray,
    sigma: float, period: float, tol: float = 1e-15,
) -> float:
    """Absolute-periodic weighted event-pair overlap, raw kernel-sum
    convention."""
    centres_x = np.asarray(centres_x, dtype=np.float64)
    centres_y = np.asarray(centres_y, dtype=np.float64)
    weights_x = np.asarray(weights_x, dtype=np.float64)
    weights_y = np.asarray(weights_y, dtype=np.float64)
    # (Kx, 1, r) - (1, Ky, r) -> (Kx, Ky, r)
    delta = centres_x[:, None, :] - centres_y[None, :, :]
    K = kernel_all_image_abs(delta, sigma, period, tol)  # (Kx, Ky)
    return float(weights_x @ K @ weights_y)


def kernel_all_image_rel(
    delta: np.ndarray, sigma: float, period: float, tol: float = 1e-15,
) -> np.ndarray:
    """Relative-periodic kernel sum.

    ``K(delta) = sum_{n in Z^{r-1}} exp(-Q(delta + P tilde n) / (4 sigma^2))``
    with ``tilde n = (n_1, ..., n_{r-1}, 0)`` (fundamental domain of
    ``Z^r / Z 1``) and ``Q(v) = sum (v_m - mean(v))^2``. ``delta`` may
    have shape ``(..., r)``; returns shape ``(...,)``.

    The coset choice does not affect the value (``Q`` is diagonal-shift
    invariant), so this is a canonical representation.
    """
    delta = np.asarray(delta, dtype=np.float64)
    # Reduce to nearest image per component; keeps kernel evaluation
    # inside the |delta_m| <= P/2 region where the truncation bound
    # applies. Diagonal-shift invariance of the rel kernel means the
    # answer is unchanged, but the truncation is now correct.
    delta = (delta + 0.5 * period) % period - 0.5 * period
    r = delta.shape[-1]
    L = _truncation_L_rel(sigma, period, tol)
    coef = -1.0 / (4.0 * sigma * sigma)

    if r == 1:
        # Q is identically zero at r=1; degenerate. Return one term.
        return np.ones(delta.shape[:-1], dtype=np.float64) * (2 * L + 1)

    # Enumerate n = (n_1, ..., n_{r-1}, 0) in {-L..L}^{r-1} x {0}
    axes = [np.arange(-L, L + 1, dtype=np.float64)] * (r - 1)
    grids = np.meshgrid(*axes, indexing='ij')
    cols = [g.ravel() for g in grids]  # each shape ((2L+1)^{r-1},)
    zeros_col = np.zeros_like(cols[0])
    cols.append(zeros_col)
    n_lat = np.stack(cols, axis=1)  # ((2L+1)^{r-1}, r)

    # (..., 1, r) + P * (G, r) -> (..., G, r)
    shifted = delta[..., None, :] + period * n_lat
    Qv = _rel_Q(shifted)  # (..., G)
    return np.exp(coef * Qv).sum(axis=-1)  # (...,)


def overlap_all_image_rel(
    centres_x: np.ndarray, weights_x: np.ndarray,
    centres_y: np.ndarray, weights_y: np.ndarray,
    sigma: float, period: float, tol: float = 1e-15,
) -> float:
    """Relative-periodic weighted event-pair overlap, raw kernel-sum
    convention."""
    centres_x = np.asarray(centres_x, dtype=np.float64)
    centres_y = np.asarray(centres_y, dtype=np.float64)
    weights_x = np.asarray(weights_x, dtype=np.float64)
    weights_y = np.asarray(weights_y, dtype=np.float64)
    delta = centres_x[:, None, :] - centres_y[None, :, :]  # (Kx, Ky, r)
    K = kernel_all_image_rel(delta, sigma, period, tol)  # (Kx, Ky)
    return float(weights_x @ K @ weights_y)


# ---------------------------------------------------------------------------
# Spectral (Fourier) route: same kernel sum, in the same convention
# ---------------------------------------------------------------------------


def kernel_spectral_abs(
    delta: np.ndarray, sigma: float, period: float, tol: float = 1e-15,
) -> np.ndarray:
    """Absolute-periodic kernel sum by Fourier expansion.

    Poisson summation gives, per component,

        sum_n exp(-(delta_m + n P)^2 / (4 sigma^2))
          = (2 sigma sqrt(pi) / P) * sum_m e^{-alpha m^2} e^{2 pi i m delta_m / P}

    (real part only, by pairing m with -m). Product across components.
    """
    delta = np.asarray(delta, dtype=np.float64)
    r = delta.shape[-1]
    M = _spectral_M(sigma, period, tol)
    alpha = 4.0 * math.pi ** 2 * sigma * sigma / (period * period)
    two_pi_over_P = 2.0 * math.pi / period
    m_arr = np.arange(1, M + 1, dtype=np.float64)  # (M,)
    env = np.exp(-alpha * m_arr * m_arr)  # (M,)
    prefactor = 2.0 * sigma * math.sqrt(math.pi) / period

    # For each component of delta, evaluate theta_m(delta_m):
    # (..., r, 1) * m_arr[None, :] -> (..., r, M)
    phase = two_pi_over_P * m_arr[None, :] * delta[..., :, None]  # (..., r, M)
    cos_terms = np.cos(phase)  # (..., r, M)
    theta = prefactor * (1.0 + 2.0 * (env * cos_terms).sum(axis=-1))  # (..., r)
    return theta.prod(axis=-1)  # (...,)


def overlap_spectral_abs(
    centres_x: np.ndarray, weights_x: np.ndarray,
    centres_y: np.ndarray, weights_y: np.ndarray,
    sigma: float, period: float, tol: float = 1e-15,
) -> float:
    centres_x = np.asarray(centres_x, dtype=np.float64)
    centres_y = np.asarray(centres_y, dtype=np.float64)
    weights_x = np.asarray(weights_x, dtype=np.float64)
    weights_y = np.asarray(weights_y, dtype=np.float64)
    delta = centres_x[:, None, :] - centres_y[None, :, :]
    K = kernel_spectral_abs(delta, sigma, period, tol)
    return float(weights_x @ K @ weights_y)


def kernel_spectral_rel(
    delta: np.ndarray, sigma: float, period: float, tol: float = 1e-15,
) -> np.ndarray:
    """Relative-periodic kernel sum by Fourier expansion.

    From the tau-averaging identity
    ``k_rel_wrap(delta) = (1/P) int_0^P prod_m theta(delta_m + tau) dtau``,
    Poisson summation of ``theta`` and the tau-integral give

        k_rel_wrap(delta) = (2 sigma sqrt(pi) / P)^r
                           * sum_{m in Z^r : sum m_j = 0} e^{-alpha |m|^2}
                           * exp(2 pi i sum m_j delta_j / P)

    where the sum-zero constraint is what makes it the *relative*
    (diagonally invariant) kernel. To match the raw kernel-sum
    convention returned by :func:`kernel_all_image_rel` — related to
    ``k_rel_wrap`` by the ``r``-dependent constant
    ``2 sigma sqrt(pi/r) / P`` derived by direct integration — we
    include the reciprocal ``P sqrt(r) / (2 sigma sqrt(pi))`` to
    convert.
    """
    delta = np.asarray(delta, dtype=np.float64)
    r = delta.shape[-1]

    if r == 1:
        # Rel-per r=1: kernel is trivial (Q = 0), reference is L=0 term
        # count from the lattice route.
        return kernel_all_image_rel(delta, sigma, period, tol)

    M = _spectral_M(sigma, period, tol)
    alpha = 4.0 * math.pi ** 2 * sigma * sigma / (period * period)

    # Enumerate constrained multi-index (m_1, ..., m_{r-1}) in
    # {-M..M}^{r-1}; m_r = -sum(m_1..m_{r-1}).
    axes = [np.arange(-M, M + 1, dtype=np.int64)] * (r - 1)
    grids = np.meshgrid(*axes, indexing='ij')
    cols = [g.ravel() for g in grids]
    last = -sum(cols)
    m_mat = np.stack(cols + [last], axis=1)  # (G, r)
    env = np.exp(-alpha * np.sum(m_mat.astype(np.float64) ** 2, axis=1))  # (G,)
    keep = env > tol
    m_mat = m_mat[keep]
    env = env[keep]

    # exp(2 pi i m . delta / P). We use only real part since env is
    # real, m -> -m symmetry pairs conjugates.
    two_pi_over_P = 2.0 * math.pi / period
    phase = two_pi_over_P * (delta[..., None, :] * m_mat[None, :, :]
                             ).sum(axis=-1)  # (..., G)
    cos_terms = np.cos(phase)  # real part

    # Kernel sum in the tau-averaging convention:
    prefactor_tau = (2.0 * sigma * math.sqrt(math.pi) / period) ** r
    k_tau = prefactor_tau * (env * cos_terms).sum(axis=-1)  # (...,)

    # Convert to the raw kernel-sum convention (matching lattice route):
    # k_rel_wrap = (2 sigma sqrt(pi/r) / P) * K_lattice
    conv = period * math.sqrt(r) / (2.0 * sigma * math.sqrt(math.pi))
    return k_tau * conv


def overlap_spectral_rel(
    centres_x: np.ndarray, weights_x: np.ndarray,
    centres_y: np.ndarray, weights_y: np.ndarray,
    sigma: float, period: float, tol: float = 1e-15,
) -> float:
    centres_x = np.asarray(centres_x, dtype=np.float64)
    centres_y = np.asarray(centres_y, dtype=np.float64)
    weights_x = np.asarray(weights_x, dtype=np.float64)
    weights_y = np.asarray(weights_y, dtype=np.float64)
    delta = centres_x[:, None, :] - centres_y[None, :, :]
    K = kernel_spectral_rel(delta, sigma, period, tol)
    return float(weights_x @ K @ weights_y)


# ---------------------------------------------------------------------------
# Quadrature reference (abs-per only; rel-per prefactors are r-dependent
# and easier to cross-check via lattice-vs-spectral above).
# ---------------------------------------------------------------------------


def overlap_quadrature_abs(
    centres_x: np.ndarray, weights_x: np.ndarray,
    centres_y: np.ndarray, weights_y: np.ndarray,
    sigma: float, period: float, tol: float = 1e-15,
    grid_size: int = 256,
) -> float:
    """Absolute-periodic overlap by direct trapezoid quadrature on the
    r-torus. Materialises both mixtures on an ``r``-cube grid and
    integrates the product. Exponential in ``r``; keep ``r <= 3``.

    Returns the raw kernel-sum overlap: divides out the
    ``(sigma sqrt(pi))^r`` prefactor of a true overlap so the value
    matches :func:`overlap_all_image_abs`.
    """
    centres_x = np.asarray(centres_x, dtype=np.float64)
    centres_y = np.asarray(centres_y, dtype=np.float64)
    weights_x = np.asarray(weights_x, dtype=np.float64)
    weights_y = np.asarray(weights_y, dtype=np.float64)
    _, r = centres_x.shape
    L = _truncation_L(sigma, period, tol)
    grid = np.linspace(0.0, period, grid_size, endpoint=False)
    dx = period / grid_size
    coef_dens = -1.0 / (2.0 * sigma * sigma)

    def _factors(centres):
        # (K, r, G): wrapped Gaussian factor per event, per component.
        K, r_ = centres.shape
        out = np.zeros((K, r_, grid.size), dtype=np.float64)
        for n in range(-L, L + 1):
            d = grid[None, None, :] - centres[:, :, None] + n * period
            out += np.exp(coef_dens * d * d)
        return out

    fx_fac = _factors(centres_x)
    fy_fac = _factors(centres_y)

    def _materialise(fac, w):
        K, r_, G = fac.shape
        shape = (G,) * r_
        f = np.zeros(shape, dtype=np.float64)
        for j in range(K):
            outer = fac[j, 0]
            for m in range(1, r_):
                outer = np.multiply.outer(outer, fac[j, m])
            f += w[j] * outer
        return f

    fx = _materialise(fx_fac, weights_x)
    fy = _materialise(fy_fac, weights_y)
    overlap_true = (fx * fy).sum() * dx ** r
    prefactor = (sigma * math.sqrt(math.pi)) ** r
    return float(overlap_true / prefactor)


# ---------------------------------------------------------------------------
# Public convenience
# ---------------------------------------------------------------------------


def overlap_reference(
    kind: str, route: str,
    centres_x: np.ndarray, weights_x: np.ndarray,
    centres_y: np.ndarray, weights_y: np.ndarray,
    sigma: float, period: float, tol: float = 1e-15,
    grid_size: int = 256,
) -> float:
    """Dispatch on ``(kind, route)`` for convenience in tests.

    - ``kind in {'abs', 'rel'}``
    - ``route in {'lattice', 'spectral'}``
      (plus ``'quadrature'`` for ``kind='abs'`` only)
    """
    if kind == 'abs':
        if route == 'lattice':
            return overlap_all_image_abs(
                centres_x, weights_x, centres_y, weights_y, sigma, period, tol)
        if route == 'spectral':
            return overlap_spectral_abs(
                centres_x, weights_x, centres_y, weights_y, sigma, period, tol)
        if route == 'quadrature':
            return overlap_quadrature_abs(
                centres_x, weights_x, centres_y, weights_y,
                sigma, period, tol, grid_size)
    if kind == 'rel':
        if route == 'lattice':
            return overlap_all_image_rel(
                centres_x, weights_x, centres_y, weights_y, sigma, period, tol)
        if route == 'spectral':
            return overlap_spectral_rel(
                centres_x, weights_x, centres_y, weights_y, sigma, period, tol)
    raise ValueError(f"unsupported kind/route: {kind!r}/{route!r}")
