"""Unrestricted enumeration of tuple centres for the inner product.

This is the plainest of the three inner-product routes: it forms every
ordered r-tuple of distinct atoms on each side and sums the kernel over
every pair of them, with no restriction to combinations (Bulger's
route) and no partition algebra (the Möbius route). Its cost is
O(K^(2r)) per attribute and event pair, so it is the slowest route by a
wide margin at any appreciable K; it exists because it is the one route
that reads the definition of the symmetric power directly, which makes
it the reference against which the other two are checked, and because
it involves no alternating sum and so is immune to the cancellation the
Möbius route can suffer.

All four modes are covered by a single quadratic form. Writing
``delta = c_x - c_y`` for the difference between a pair of tuple
centres,

    Q(delta) = (1 / 2r) * sum_{i,j} wrap(delta_i - delta_j)^2   [relative]
    Q(delta) = sum_i wrap(delta_i)^2                            [absolute]

where ``wrap`` is the identity in the non-periodic modes and reduction
to the nearest image otherwise. The relative form is the pairwise
identity for ``delta^T M delta`` with ``M = I - J/r``, the projector
that annihilates the common-shift direction; wrapping the pairwise
differences rather than the raw ones is what makes the relative,
periodic case agree with Bulger's route, which computes the same
wrapped-difference kernel (the alternative to the transposition
average; see the Online Supplement of the MAET article).
"""
from __future__ import annotations

import numpy as np

#: Target working-set size for the chunked pair loop, in bytes.
CHUNK_BYTES = 64 << 20

def _tuple_centres(p, w, r):
    """Ordered r-tuples of distinct atoms, and their weight products.

    NaN entries are dropped per side first, matching the convention of
    the other routes. Returns ``(centres, weights)`` with shapes
    ``(n_tuples, r)`` and ``(n_tuples,)``; ``n_tuples`` is zero when
    fewer than ``r`` valid atoms remain, so no r-tuple can be formed.
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    w = np.asarray(w, dtype=np.float64).ravel()
    valid = ~(np.isnan(p) | np.isnan(w))
    p, w = p[valid], w[valid]
    if p.size < r:
        return np.zeros((0, r)), np.zeros(0)
    # The existing single-multiset builder returns (r, nJ) positions and
    # (nJ,) weight products; transpose to (nJ, r) for the pair algebra
    # below.
    from .cosine import _build_ordered_r_tuples
    u, w_j = _build_ordered_r_tuples(p, w, r)
    return np.asarray(u).T, np.asarray(w_j)


def _kernel(c_a, c_b, r, sigma, is_rel, is_per, period):
    """Kernel value for every pair of tuple centres, shape (n_a, n_b).

    The two periodic modes are periodized differently, and must be, to
    match the routes they are checked against.

    In the absolute modes the quadratic form is separable across
    coordinates, so the periodic kernel is a product of one-dimensional
    wrapped Gaussians -- a sum over every image, not merely the nearest.
    Truncating to the nearest image alone is accurate only while
    sigma/P is small, and departs from the other routes by ~1e-3 at
    sigma/P = 0.1.

    In the relative modes the form couples the coordinates, so no such
    factorisation is available; the minimum-image convention is used,
    which is what the other routes compute there (see the note on the
    wrapped-difference kernel in :mod:`_centres_inner`).
    """
    delta = c_a[:, None, :] - c_b[None, :, :]
    if is_rel:
        d = delta[..., :, None] - delta[..., None, :]      # (n_a, n_b, r, r)
        if is_per:
            d = d - period * np.round(d / period)
        q = (d ** 2).sum(axis=(-1, -2)) / (2.0 * r)
        return np.exp(-q / (4.0 * sigma ** 2))
    if not is_per:
        return np.exp(-(delta ** 2).sum(axis=-1) / (4.0 * sigma ** 2))
    # Absolute periodic: product over coordinates of the wrapped Gaussian.
    # Images beyond the truncation width contribute below double
    # precision; the width follows the kernel's combined variance
    # 2 sigma^2, hence the sqrt(2).
    n_img = int(np.ceil(8.0 * np.sqrt(2.0) * sigma / period)) + 1
    k = np.arange(-n_img, n_img + 1).reshape(-1, *([1] * delta.ndim))
    shifted = delta[None, ...] - k * period
    per_coord = np.exp(-(shifted ** 2) / (4.0 * sigma ** 2)).sum(axis=0)
    return per_coord.prod(axis=-1)


def centres_inner_product(p_x, w_x, p_y, w_y, sigma, r,
                          is_rel, is_per, period):
    """Unnormalised inner product by unrestricted centres enumeration.

    Covers all four modes. Returns a float; zero when either side has
    fewer than ``r`` valid atoms.

    The prefactor common to both routes is omitted, exactly as in the
    pairwise route, since it cancels in the cosine and in the one-sided
    normalisation.
    """
    c_x, ww_x = _tuple_centres(p_x, w_x, r)
    c_y, ww_y = _tuple_centres(p_y, w_y, r)
    if c_x.shape[0] == 0 or c_y.shape[0] == 0:
        return 0.0

    # The pair matrix is n_x by n_y, and the relative form needs a
    # further r by r per pair, so the peak allocation grows as K^(2r) and
    # would reach gigabytes well inside the range this route is used for
    # (4.2 GiB at r = 4, K = 12). The route is meant to be slow, not
    # unusable, so the X side is chunked to hold the working set near
    # CHUNK_BYTES; the result is a plain sum and so is unaffected by the
    # chunking, up to floating-point accumulation order.
    per_pair = r * r if is_rel else r
    rows = max(1, int(CHUNK_BYTES / max(1, 8 * per_pair * c_y.shape[0])))
    total = 0.0
    for start in range(0, c_x.shape[0], rows):
        stop = min(start + rows, c_x.shape[0])
        kern = _kernel(c_x[start:stop], c_y, r, float(sigma), bool(is_rel),
                       bool(is_per), float(period))
        total += float((ww_x[start:stop, None] * ww_y[None, :] * kern).sum())
    return total
