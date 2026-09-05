"""Independent reference enumeration, for tests only.

The shipped ``method='centres'`` route reuses ``_ip_core_ma`` so that its
constants match Bulger's and the two are comparable; see
``_cos_sim_exp_tens_ma_centres``. This module is the *other* thing a
reference can be: an implementation sharing no code with the core, so
that agreement with it would catch a bug the three shipped routes could
otherwise share. It is deliberately not importable from ``mpt`` and is
not a route.
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
    # The reference tuple builder returns (r, nJ) positions and (nJ,)
    # weight products; transpose to (nJ, r) for the pair algebra below.
    from tests.references.mobius_ip_reference import build_ordered_r_tuples
    u, w_j = build_ordered_r_tuples(p, w, r)
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
    wrapped-difference kernel in :func:`_kernel`).
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
