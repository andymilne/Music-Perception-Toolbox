"""Single-multiset inner-product references for the Möbius routes.

Test oracle only: nothing under ``mpt`` imports this module.

``inner_product_direct_abs`` enumerates ordered r-tuples on each side and
contracts the Gaussian kernel between them --- no Möbius alternating sum,
so it is exact for any ``K >= r`` and serves as the comparison point for
the batched Möbius per-attribute matrix. ``orbit_inner_abs`` and
``orbit_inner_rel`` are the ``N = 1`` specialisations of the live
per-attribute Möbius leaves (``mpt._mobius.inner_product_orbit`` and
``mpt._tensor._mobius_inner._rel_inner_batched``), the MATLAB twins of
which (``mobius.orbitInnerAbsSingleMultiset`` /
``orbitInnerRelSingleMultiset``) are live on the Rényi-2 single-multiset
path; Python routes that case through the batched matrix instead.
"""
from __future__ import annotations

import math
from itertools import permutations

import numpy as np
from scipy.special import comb as _comb

from mpt._tensor._mobius_inner import _rel_inner_batched, _trunc_kernel_exp
from mpt._tensor.density import _nchoosek_indices


def orbit_inner_abs(p_a, w_a, p_b, w_b, sigma, r, is_per, period,
                     *, return_cancellation_ratio=False,
                     truncation_sigmas=None, wrap_a='full-image'):
    """<T_A, T_B> in absolute mode via the Möbius method.

    With ``return_cancellation_ratio=True``, returns ``(value, ratio)``
    where ratio is ``|sum| / max(|term|)`` from the Möbius alternating
    partition sum (1.0 means no cancellation; <<1 means digits lost). See
    :func:`mpt._mobius.inner_product_orbit` for full semantics.

    ``truncation_sigmas`` is honoured on the kernel; ``None`` resolves
    to the global default.

    ``wrap_a`` selects the abs-per measure: ``'full-image'`` (default)
    uses the torus (all-image) 1-D wrapped Gaussian per coordinate, delivered
    by :func:`_wrapped_kernel.wrapped_gaussian_1d` in overlap
    convention. The r-tuple full-image kernel factors as
    :math:`\\prod_a \\theta(d_a)`, delivered by the orbit reduction
    over the 1-D theta values. ``'single-image'`` opts into the
    nearest-image kernel unchanged. Ignored when ``is_per=False``.
    """
    from mpt._mobius import inner_product_orbit
    from mpt._defaults import get_default
    from mpt._wrapped_kernel import wrapped_gaussian_1d

    if truncation_sigmas is None:
        truncation_sigmas = get_default('truncation_sigmas')

    diffs = p_a[:, None] - p_b[None, :]
    if is_per and wrap_a == 'full-image':
        # Overlap-kernel convention (exponent_denominator = 4). The
        # (sigma sqrt(pi))^r prefactor stays: the 1-D wrapped Gaussian's
        # integral over the circle equals the single Gaussian's over the
        # line, so the r-tuple normalisation is identical.
        K = wrapped_gaussian_1d(diffs, sigma, period, truncation_sigmas,
                                exponent_denominator=4)
    else:
        if is_per:
            diffs = diffs - period * np.floor(diffs / period + 0.5)
        K = _trunc_kernel_exp(diffs ** 2, sigma, truncation_sigmas)
    return inner_product_orbit(
        K, w_a, w_b, r, prefactor=(sigma * np.sqrt(np.pi)) ** r,
        return_cancellation_ratio=return_cancellation_ratio,
    )



def inner_product_direct_abs(p_x, w_x, p_y, w_y, sigma, r,
                                   is_per, period):
    """<T_X, T_Y> in absolute mode via direct r-tuple enumeration.

    Computes the single-multiset inner product
        <T_X, T_Y> = (sigma * sqrt(pi))**r *
                     sum_{J, K} wJ_x[J] * wJ_y[K] *
                                exp(-||centres_x[:, J] - centres_y[:, K]||^2
                                    / (4 sigma^2))
    by enumerating ordered r-tuples on each side. No Möbius
    alternating sum is involved, so the result is exact for any
    K_x, K_y >= r. This is a reference/direct implementation, retained
    as the enumerated comparison point for the Möbius route.

    NaN tolerance: NaN entries in ``p_x`` / ``w_x`` / ``p_y`` / ``w_y``
    are dropped per side before enumeration. If the dropped count
    leaves either side with fewer than r valid values, returns 0 by
    convention (cannot form an r-tuple).

    Cost: O(K_x! / (K_x - r)! * K_y! / (K_y - r)! * r) per call. Cheap
    when K is close to r (the unsafe regime).
    """
    p_x = np.asarray(p_x, dtype=np.float64).ravel()
    w_x = np.asarray(w_x, dtype=np.float64).ravel()
    p_y = np.asarray(p_y, dtype=np.float64).ravel()
    w_y = np.asarray(w_y, dtype=np.float64).ravel()

    valid_x = ~(np.isnan(p_x) | np.isnan(w_x))
    valid_y = ~(np.isnan(p_y) | np.isnan(w_y))
    p_x = p_x[valid_x]; w_x = w_x[valid_x]
    p_y = p_y[valid_y]; w_y = w_y[valid_y]
    K_x = p_x.size
    K_y = p_y.size

    if K_x < r or K_y < r:
        return 0.0

    if r == 1:
        diffs = p_x[:, None] - p_y[None, :]
        if is_per:
            diffs = diffs - period * np.floor(diffs / period + 0.5)
        K_mat = np.exp(-(diffs ** 2) / (4 * sigma ** 2))
        return float(sigma * np.sqrt(np.pi) *
                     np.einsum('i,ij,j->', w_x, K_mat, w_y))

    # r >= 2: enumerate ordered r-tuples and contract.
    U_x, wJ_x = build_ordered_r_tuples(p_x, w_x, r)   # (r, nJ_x), (nJ_x,)
    U_y, wJ_y = build_ordered_r_tuples(p_y, w_y, r)

    diffs = U_x[:, :, None] - U_y[:, None, :]   # (r, nJ_x, nJ_y)
    if is_per:
        diffs = diffs - period * np.floor(diffs / period + 0.5)
    Q = np.sum(diffs ** 2, axis=0)              # (nJ_x, nJ_y)
    K_mat = np.exp(-Q / (4 * sigma ** 2))

    return float((sigma * np.sqrt(np.pi)) ** r *
                 np.einsum('i,ij,j->', wJ_x, K_mat, wJ_y))


def build_ordered_r_tuples(p, w, r):
    """Ordered r-tuple construction for a single multiset.

    Returns ``(U, wJ)`` where ``U`` is ``(r, nJ)`` of position values
    along ordered r-tuples and ``wJ`` is ``(nJ,)`` of weight products.
    Used by :func:`inner_product_direct_abs` and any other helper
    that needs single-event ordered tuples without going through the
    full :func:`build_exp_tens` API.
    """
    K = p.size
    n_perms = math.factorial(r)
    n_combs = int(_comb(K, r, exact=True))
    n_j = n_perms * n_combs

    nck = _nchoosek_indices(K, r)             # r x n_combs
    all_perms = np.array(
        list(permutations(range(r))), dtype=np.intp,
    ).T                                        # r x r!

    j_idx = np.empty((r, n_j), dtype=np.intp)
    offset = 0
    for i in range(n_perms):
        j_idx[:, offset:offset + n_combs] = nck[all_perms[:, i], :]
        offset += n_combs

    U = p[j_idx]                               # r x nJ
    wJ = np.prod(w[j_idx], axis=0)             # (nJ,)
    return U, wJ



def orbit_inner_rel(p_a, w_a, p_b, w_b, sigma, r, is_per, period,
                     samples_per_sigma=None, *,
                     return_cancellation_ratio=False,
                     truncation_sigmas=None):
    """<T_A, T_B> in relative mode: the single-multiset (N = 1)
    specialisation of :func:`_rel_inner_batched`.

    All conventions are the batched core's: the shared ``[0, P)``
    grid with :func:`auto_ntau_default` nodes in periodic mode; in
    non-periodic mode a window of width
    ``spread_a + spread_b + 2 * _rel_window_margin(t) * sigma``
    centred on the weighted-mean offset, with ``samples_per_sigma``
    nodes per sigma, evaluated by plain Riemann sum (the margin places
    every kernel entry strictly outside the truncation radius at the
    window edges, so the endpoint integrand is exactly zero and the
    Riemann sum equals the trapezoidal rule exactly);
    slab-bounded contraction; and, when requested, the mass-aware
    cancellation diagnostic
    ``|sum_u F_u| / sum_u max_orb(|term_orb_u|)``.

    With ``return_cancellation_ratio=True``, returns ``(value, ratio)``.
    """
    Pa = np.asarray(p_a, dtype=np.float64).reshape(-1, 1)
    Pb = np.asarray(p_b, dtype=np.float64).reshape(-1, 1)
    Wa = np.asarray(w_a, dtype=np.float64).reshape(-1, 1)
    Wb = np.asarray(w_b, dtype=np.float64).reshape(-1, 1)
    out = _rel_inner_batched(
        Pa, Wa, Pb, Wb, sigma, r, is_per, period,
        return_cancellation_ratio=return_cancellation_ratio,
        truncation_sigmas=truncation_sigmas,
        samples_per_sigma=samples_per_sigma,
    )
    if return_cancellation_ratio:
        I, ratio = out
        return float(I[0, 0]), float(ratio)
    return float(out[0, 0])
