"""Per-event-pair reference for the nested-attribute inner product.

Test oracle only: nothing under ``mpt`` imports this module. It is the
Python twin of MATLAB ``internal.nestedContract``'s per-pair fallback
(``nestedIp`` / ``makeQuadrature``), retained here so the batched
contraction :func:`mpt._tensor._nested_contraction.nested_attr_matrix`
can be checked one event pair at a time against a transparent loop over
the shared quadrature grid. The live Python path never loops event pairs:
it folds the event-pair grid into the batch axis of ``_contract``.

The building blocks that the live path also uses (``_wrap``, ``_trunc``,
``_contract``, ``_shared_leaf_template``, ``auto_ntau``) are imported from
the live module, so the reference differs from it only in the per-pair
organisation, never in the kernel or the reduction.
"""
from __future__ import annotations

import math

import numpy as np

from mpt._tensor._nested_contraction import (
    _contract, _shared_leaf_template, _trunc, _wrap, auto_ntau,
)


def _theta_truncation_L(sigma, period, truncation_sigmas):
    """Number of periodic-image shifts per side to include in the 1D
    wrapped Gaussian ``theta(d) = sum_n exp(-(d + n P)^2 / (4 sigma^2))``
    so the first omitted term is below the truncation floor. Callers
    are responsible for having reduced ``d`` to ``[-P/2, P/2]`` first
    (``_wrap``), so the worst-case first-omitted term is at
    ``|d + n P| >= (L + 1/2) P``.
    """
    from mpt._defaults import truncation_floor
    if sigma <= 0.0 or period <= 0.0:
        return 0
    floor = truncation_floor(truncation_sigmas)
    if floor <= 0.0 or floor >= 1.0:
        floor = 1e-15
    rhs = 2.0 * sigma / period * math.sqrt(-math.log(floor))
    return max(0, int(math.ceil(rhs - 0.5)))




def _ip_absolute(recipe_x, recipe_y, vX, vY, wX, wY, sigma, is_per, period,
                 truncation_sigmas, wrap_a='full-image'):
    """Absolute-mode inner product for one nested attribute.

    The absolute-mode r-tuple kernel factors across coordinates (unlike
    relative-mode, whose ``Q`` couples them via the projected form),
    so the per-position 1D kernel here is the object the outer contraction
    multiplies across the r_a coordinates. In periodic mode that per-coordinate 1D
    kernel is the wrapped Gaussian (theta):
    ``theta(d) = sum_n exp(-(d + n P)^2 / (4 sigma^2))``. Reducing
    ``d`` to ``[-P/2, P/2]`` first lets ``L = 0`` — i.e. reduce to the
    single-image Gaussian — cover the small-sigma regime, and the
    image sum switches on only when the accuracy floor requires it.
    When the user has opted this attribute into
    ``wrap_a='single-image'`` the L is forced to 0 regardless.

    The abs-per r-tuple kernel is ``prod_a theta(d_a)``. Its lattice
    representation is the sum over ``Z^r`` of Gaussians in the shifted
    r-tuple; the product-of-theta form is the cheaper one to compute
    (``(2L+1) * r`` vs ``(2L+1)^r`` per pair). The r-dim outer product
    is applied downstream in ``_contract``, so this function returns
    the 1D per-position kernel matrix.
    """
    d = vX[:, None] - vY[None, :]
    if is_per:
        d = _wrap(d, period)
        L = (_theta_truncation_L(sigma, period, truncation_sigmas)
             if wrap_a == 'full-image' else 0)
        if L == 0:
            K = np.exp(-d ** 2 / (4.0 * sigma ** 2))
        else:
            n_shift = np.arange(-L, L + 1, dtype=np.float64) * period
            d_shift = d[..., None] + n_shift               # (nX, nY, 2L+1)
            K = np.exp(-d_shift ** 2 / (4.0 * sigma ** 2)).sum(axis=-1)
    else:
        K = np.exp(-d ** 2 / (4.0 * sigma ** 2))
    K = K[None, :, :]                                       # (1, nX, nY)
    K = K * (wX[None, :, None] * wY[None, None, :])
    _trunc(K, sigma, truncation_sigmas)
    return float(_contract(recipe_x, recipe_y, K)[0])




def _ip_rel_nonper_factored(recipe_x, recipe_y, vX, vY, wX, wY, sigma,
                            truncation_sigmas, taus):
    """Closed-form inner-partial reduction of the relative-non-periodic inner
    product for spectrally-augmented ordered cells.

    When each side is an ordered cell (outer ``[sym] = 0`` with tuple size equal
    to the cell length) whose tones carry a shared partial template, the inner
    partial index sums analytically into the template cross-correlation
    ``g(delta) = sum_{p,q} wX_p wY_q exp(-(delta + offX_p - offY_q)^2 / 4 sigma^2)``,
    and the cell overlap reduces to the reference-value differences alone:
    ``sum_tau prod_a g(refX_a - refY_a - tau)``. This evaluates only the
    per-position note overlaps, never the full partial-by-partial kernel, and is
    exact to floating-point summation order. Returns ``None`` when the structure
    is not of this form (then the caller uses the generic contraction).
    """
    if recipe_x.sym or recipe_y.sym:
        return None                    # need ordered cells (outer [sym] = 0)
    if (int(recipe_x.r) != len(recipe_x.children)
            or int(recipe_y.r) != len(recipe_y.children)):
        return None                    # need the whole cell as one ordered tuple
    tx = _shared_leaf_template(recipe_x, vX, wX)
    ty = _shared_leaf_template(recipe_y, vY, wY)
    if tx is None or ty is None:
        return None
    cX, offX, wtX = tx
    cY, offY, wtY = ty
    if cX.size != cY.size:             # diagonal needs equal cell lengths
        return None
    dpq = offX[:, None] - offY[None, :]                     # (Kx, Ky)
    wpq = wtX[:, None] * wtY[None, :]
    delta = (cX - cY)[None, :] - taus[:, None]              # (T, r)
    K = np.exp(-(delta[..., None, None] + dpq) ** 2
               / (4.0 * sigma ** 2)) * wpq                  # (T, r, Kx, Ky)
    from mpt._defaults import truncation_floor
    floor = truncation_floor(truncation_sigmas)
    K[K < floor] = 0.0             # per-term floor, matching _trunc exactly
    m_diag = K.sum(axis=(-1, -2))                           # (T, r)
    return float(m_diag.prod(axis=1).sum())   # common dtau cancels in the cosine




def _ip_rel_nonper_generic(recipe_x, recipe_y, vX, vY, wX, wY, sigma,
                           truncation_sigmas, taus):
    d = vX[:, None, None] - (vY[None, :, None] + taus[None, None, :])
    K = np.exp(-d ** 2 / (4.0 * sigma ** 2)).transpose(2, 0, 1)   # (T, nX, nY)
    K = K * (wX[None, :, None] * wY[None, None, :])
    _trunc(K, sigma, truncation_sigmas)
    return float(_contract(recipe_x, recipe_y, K).sum())  # common dtau cancels


def _ip_rel_nonper(recipe_x, recipe_y, vX, vY, wX, wY, sigma,
                   truncation_sigmas, taus):
    fast = _ip_rel_nonper_factored(recipe_x, recipe_y, vX, vY, wX, wY, sigma,
                                   truncation_sigmas, taus)
    if fast is not None:
        return fast
    return _ip_rel_nonper_generic(recipe_x, recipe_y, vX, vY, wX, wY, sigma,
                                  truncation_sigmas, taus)




def make_quadrature(is_rel, is_per, sigma, period, vmin, vmax,
                    truncation_sigmas):
    """Shared quadrature grid for all event-pairs and the IP triple.

    A common grid means the constant dtau / 1-over-ntau factor is identical
    across XY, XX, YY and cancels in the cosine. Returns a dict the IP
    helper consumes.
    """
    if not is_rel:
        return {"mode": "abs", "is_per": bool(is_per)}
    tol = max(math.exp(-0.5 * (truncation_sigmas or math.inf) ** 2), 1e-12)
    if is_per:
        ntau = auto_ntau(period, sigma, tol)
        return {"mode": "relper",
                "taus": np.linspace(0.0, period, ntau, endpoint=False)}
    spread = float(vmax - vmin)
    pad = (6.0 + 0.5 * max(0.0, -math.log10(max(tol, 1e-16)))) * sigma
    hi = spread + pad
    n = int(max(64, math.ceil(2.0 * hi / (sigma / 4.0))))
    return {"mode": "relnonper", "taus": np.linspace(-hi, hi, n)}


def nested_ip(recipe_x, recipe_y, vX, vY, wX, wY, sigma, period,
              truncation_sigmas, quad, wrap_a='full-image'):
    """Bare inner product for one event-pair, on the shared quadrature.

    ``recipe_x`` indexes the X (``vX``) axis of the rectangular kernel,
    ``recipe_y`` the Y (``vY``) axis; pass the same recipe for both when the
    two densities share their nested structure.
    """
    mode = quad["mode"]
    if mode == "abs":
        # Periodicity comes from the density's [per] flag (threaded through the
        # quadrature dict), not from whether ``period`` happens to be finite:
        # an absolute non-periodic attribute may still carry a finite period.
        return _ip_absolute(recipe_x, recipe_y, vX, vY, wX, wY, sigma,
                            bool(quad["is_per"]), period, truncation_sigmas,
                            wrap_a)
    if mode == "relper":
        taus = quad["taus"]
        d = vX[:, None, None] - (vY[None, :, None] + taus[None, None, :])
        if wrap_a == 'single-image':
            d = _wrap(d, period)
            K = np.exp(-d ** 2 / (4.0 * sigma ** 2))
        else:
            # All-image per-coordinate kernel; see the taugrid branch of
            # _nested_attr_matrix_impl.
            from mpt._wrapped_kernel import wrapped_gaussian_1d
            K = wrapped_gaussian_1d(d, sigma, period, truncation_sigmas,
                                    exponent_denominator=4)
        K = K.transpose(2, 0, 1)
        K = K * (wX[None, :, None] * wY[None, None, :])
        _trunc(K, sigma, truncation_sigmas)
        return float(_contract(recipe_x, recipe_y, K).sum())
    # relnonper
    return _ip_rel_nonper(recipe_x, recipe_y, vX, vY, wX, wY, sigma,
                          truncation_sigmas, quad["taus"])
