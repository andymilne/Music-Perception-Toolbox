"""The mass of a MAET in a region: ``mass_maet``.

A MAET is a sum of kernels, one per tuple, each carrying its tuple's
weight product. Normalized to unit mass, each kernel's share of a region
is the probability that a Gaussian centred at the tuple falls in it, so
the mass of the density in the region is

    m(R) = sum_tuples w_j * P_j(R),

and the whole density's mass is the sum of its weight products. The
density is a Cartesian product across attributes within each event, so
the sum factors:

    m(R) = sum_events prod_attributes S_a^(event)(R_a),

with ``S_a^(event)(R_a)`` the attribute's own tuple sum over its part of
the region, and the sum of the attribute's weight products where the
region leaves the attribute free. The joint tuple set is never built.

A region is a box (``(lo, hi)`` on every coordinate of an attribute, or
a row per coordinate), whose mass has a closed form in the error
function wherever the kernel's coordinates are independent; or a
Gaussian ``('gaussian', centre, sd)``, a soft region of unit height at
its centre, whose mass has a closed form for any kernel covariance on a
non-periodic attribute. Periodic kernels sum their images (full-image)
or are taken on the nearest image, renormalized to unit mass on the
circle (single-image; also the relative-periodic pairwise wrap).

The swept counterpart, :func:`swept_mass`, lives in :mod:`.swept`.
"""
from __future__ import annotations

import numpy as np
from scipy.special import erfc

from .premaet import is_pre_maet

_SQRT2 = np.sqrt(2.0)
# Kernel images beyond this many standard deviations from a region carry
# less than 1e-18 of their mass into it.
_IMAGE_REACH = 9.0

_NORMALIZE_VALUES = ("none", "total")


def mass_maet(dens, region=None, *, normalize="none", verbose=False):
    r"""Mass of a multi-attribute expectation tensor (MAET) in a region.

    **Overview.** The density of a MAET is a weighted sum of Gaussian
    kernels, one per tuple. Each kernel is taken here with unit mass, so
    the mass of the whole density is the sum of its tuples' weight
    products, and the mass in a region counts each tuple by the share of
    its kernel that falls in the region:

    .. math:: m(R) = \sum_j w_j \, P_j(R).

    The mass is the integral over ``R`` of the density
    :func:`eval_maet` returns under ``normalize='gaussian'``; with
    ``normalize='total'`` it is divided by the whole density's mass,
    giving the share of the density that lies in the region. Because a
    kernel has width, a tuple just outside a box still contributes the
    part of its kernel that crosses the edge.

    Parameters
    ----------
    dens : MaetDensity, pre-MAET, or list of them
        The density, from :func:`build_maet`, or a whole pre-MAET, which
        is built here. A list gives one value per entry.
    region : dict, optional
        ``{a: spec}``, keyed by attribute index. An attribute not named
        is integrated over entirely, so ``region=None`` gives the whole
        density's mass. ``spec`` is one of:

        - ``(lo, hi)``: a box, the same bounds on every coordinate of
          the attribute;
        - an array with a row ``(lo, hi)`` per coordinate;
        - ``('gaussian', centre, sd)``: a soft region, weighting the
          density by :math:`\exp(-\lVert x - c\rVert^2 / 2 s^2)`;
          ``centre`` is a scalar (every coordinate) or one value per
          coordinate.

        Bounds may be infinite. The coordinates are those of the
        attribute's query in :func:`eval_maet`: the ``r`` values of an
        absolute tuple, and the ``r - 1`` values above the first for a
        relative one. An exchangeable density holds every ordering of a
        tuple, so a region on a relative attribute that should not care
        about order must be symmetric: at r = 2, the region and its
        negative.
    normalize : {'none', 'total'}, default 'none'
        ``'none'`` returns the mass; ``'total'`` the share of the whole
        density's mass (NaN where that is zero).
    verbose : bool, default False
        Passed to :func:`build_maet` for a pre-MAET.

    Returns
    -------
    float, or np.ndarray for a list

    Notes
    -----
    On a periodic attribute a box spans at most one period, and bounds
    may wrap (``(1100, 1300)`` with period 1200). A Gaussian region on a
    periodic attribute is taken on the nearest image of its centre.

    A box needs the attribute's kernel to be independent across its
    coordinates, which holds for an absolute attribute and for a
    relative one of two values (and a nested attribute whose relative
    blocks hold two); a relative attribute of three or more has
    correlated coordinates, and there a box is refused and a Gaussian
    region, which has a closed form for any covariance, can be used on
    a non-periodic attribute. An attribute with a matrix-valued kernel
    covariance (``kernel_cov``) cannot be restricted. The integrals are
    exact: ``truncation_sigmas`` does not apply.

    See Also
    --------
    swept_mass : the mass at each of a list of sweep values.
    eval_maet : the density at points.
    """
    if isinstance(normalize, str) and normalize.lower() in _NORMALIZE_VALUES:
        normalize = normalize.lower()
    else:
        raise ValueError(
            f"normalize must be one of {_NORMALIZE_VALUES!r}; "
            f"got {normalize!r}.")
    if isinstance(dens, (list, tuple)) and not is_pre_maet(dens):
        return np.array([mass_maet(d, region, normalize=normalize,
                                   verbose=verbose) for d in dens],
                        dtype=float)
    if is_pre_maet(dens):
        from .build import build_maet
        dens = build_maet(dens, verbose=verbose)
    return _density_mass(dens, region, normalize)


# -------------------------------------------------------------------
#  One density
# -------------------------------------------------------------------


def _density_mass(dens, region, normalize):
    from .build import _enum_flat_attr, _nested_enum_indices
    from .dispatch import _inner_r_vec

    A = int(dens.n_attrs)
    N = int(dens.n)
    parsed = _parse_region(region, dens, A)

    P = [np.asarray(p, dtype=np.float64) for p in dens.p_attr]
    W = [np.asarray(w, dtype=np.float64) for w in dens.w]
    r_vec = [int(v) for v in np.atleast_1d(dens.r)]
    is_exch = [bool(v) for v in np.atleast_1d(dens.is_exch)]
    nested = dens.nested

    # Per-attribute tuple-index structure over the ever-valid values, as
    # the factored evaluation builds it: the index pattern is the same in
    # every event, and a tuple touching a value absent in its event
    # carries weight zero.
    perm = []
    for a in range(A):
        ever_valid = np.nonzero(
            (~np.isnan(P[a])).any(axis=1))[0].astype(np.intp)
        spec = nested[a]
        if spec is not None:
            tags = np.asarray(spec["tags"])[ever_valid]
            pm, _ = _nested_enum_indices(
                ever_valid, tags, np.asarray(spec["r"]).ravel(),
                np.asarray(spec["exch"]).ravel())
        elif r_vec[a] == 1:
            pm = ever_valid[None, :]
        elif ever_valid.size < r_vec[a]:
            pm = np.empty((r_vec[a], 0), dtype=np.intp)
        else:
            pm, _, _, _ = _enum_flat_attr(
                np.zeros(P[a].shape[0]), ever_valid, r_vec[a], is_exch[a],
                np.ones(P[a].shape[0]))
        perm.append(np.asarray(pm, dtype=np.intp))

    inner_r = _inner_r_vec(dens)
    geom = {a: _attr_geometry(dens, a, int(inner_r[a]), perm[a].shape[0])
            for a in parsed}
    for a, spec in parsed.items():
        _check_region(spec, geom[a], a)

    mass = 0.0
    total = 0.0
    for n in range(N):
        prod_r = 1.0
        prod_t = 1.0
        for a in range(A):
            pm = perm[a]
            p_col = P[a][:, n]
            w_col = W[a][:, n]
            absent = np.isnan(p_col)
            p_fill = np.where(absent, 0.0, p_col)
            w_fill = np.where(absent | np.isnan(w_col), 0.0, w_col)
            w_tuple = np.prod(w_fill[pm], axis=0)
            s_all = float(w_tuple.sum())
            prod_t *= s_all
            if a in parsed:
                coords = _coordinates(p_fill[pm], geom[a])
                share = _region_share(coords, parsed[a], geom[a])
                prod_r *= float(w_tuple @ share)
            else:
                prod_r *= s_all
        mass += prod_r
        total += prod_t
    if normalize == "total":
        return mass / total if total != 0.0 else float("nan")
    return float(mass)


def _attr_geometry(dens, a, inner_r, n_rows):
    """The attribute's kernel in its query coordinates: the metric ``M``
    (the kernel is ``exp(-d' M d / 2 sigma^2)``), the periodic mode, and
    whether it can be restricted at all."""
    is_rel = bool(np.atleast_1d(dens.is_rel)[a])
    is_per = bool(np.atleast_1d(dens.is_per)[a])
    period = float(np.atleast_1d(dens.period)[a])
    sigma = np.atleast_1d(dens.sigma)[a]
    cov = getattr(dens, "kernel_cov", None)
    has_cov = cov is not None and cov[a] is not None
    if inner_r > 0:
        b = inner_r
        n_blocks = n_rows // b
        block = np.eye(b - 1) - np.ones((b - 1, b - 1)) / b
        M = np.kron(np.eye(n_blocks), block)
        kind = "rel"
    elif is_rel:
        d = n_rows - 1
        M = np.eye(d) - np.ones((d, d)) / n_rows
        kind = "rel"
    else:
        M = np.eye(n_rows)
        kind = "abs"
    if not is_per:
        mode = "none"
    elif kind == "abs":
        wrap = getattr(dens, "wrap", None)
        w = str(wrap[a]) if wrap is not None and a < len(wrap) else "full-image"
        mode = "single" if w == "single-image" else "full"
    else:
        mode = "single"   # the relative-periodic pairwise wrap
    return {"M": M, "dim": M.shape[0], "diag": bool(np.allclose(M, np.diag(np.diag(M)))),
            "mode": mode, "period": period, "kind": kind, "inner_r": inner_r,
            "sigma": None if has_cov else float(sigma), "has_cov": has_cov}


def _coordinates(u, g):
    """Tuple values ``u`` (rows x tuples) in the attribute's query
    coordinates, as the evaluation paths reduce them."""
    if g["inner_r"] > 0:
        b = g["inner_r"]
        return np.vstack([u[k * b + 1:(k + 1) * b, :] - u[k * b:k * b + 1, :]
                          for k in range(u.shape[0] // b)])
    if g["kind"] == "rel":
        return u[1:, :] - u[:1, :]
    return u


# -------------------------------------------------------------------
#  Regions
# -------------------------------------------------------------------


def _parse_region(region, dens, A):
    if region is None:
        return {}
    if not isinstance(region, dict):
        raise TypeError(
            "`region` must be a dict keyed by attribute index, {a: spec}; "
            f"got {type(region).__name__}.")
    out = {}
    for k, spec in region.items():
        if isinstance(k, (bool, np.bool_)) or not isinstance(
                k, (int, np.integer)):
            raise TypeError(
                f"`region`: keys are attribute indices (int); got {k!r}.")
        a = int(k) if k >= 0 else A + int(k)
        if not 0 <= a < A:
            raise ValueError(
                f"`region` names attribute {k}, out of range for {A} "
                f"attributes.")
        if a in out:
            raise ValueError(f"`region` names attribute {a} twice.")
        out[a] = _parse_spec(spec, a)
    return out


def _parse_spec(spec, a):
    if isinstance(spec, (tuple, list)) and len(spec) > 0 \
            and isinstance(spec[0], str):
        if spec[0].lower() != "gaussian" or len(spec) != 3:
            raise ValueError(
                f"`region` for attribute {a}: a named region is "
                f"('gaussian', centre, sd); got {spec!r}.")
        centre = np.atleast_1d(np.asarray(spec[1], dtype=float)).ravel()
        sd = float(spec[2])
        if not (np.isfinite(sd) and sd > 0):
            raise ValueError(
                f"`region` for attribute {a}: the Gaussian's sd must be "
                f"finite and positive; got {spec[2]!r}.")
        if not np.all(np.isfinite(centre)):
            raise ValueError(
                f"`region` for attribute {a}: the Gaussian's centre must "
                f"be finite.")
        return {"type": "gaussian", "centre": centre, "sd": sd}
    B = np.asarray(spec, dtype=float)
    if B.ndim == 1:
        B = B[None, :]
    if B.ndim != 2 or B.shape[1] != 2:
        raise ValueError(
            f"`region` for attribute {a}: a box is (lo, hi), or one row "
            f"(lo, hi) per coordinate; got shape {np.shape(spec)}.")
    if np.any(np.isnan(B)) or np.any(B[:, 0] > B[:, 1]):
        raise ValueError(
            f"`region` for attribute {a}: each box row must be (lo, hi) "
            f"with lo <= hi.")
    return {"type": "box", "bounds": B}


def _check_region(spec, g, a):
    if g["has_cov"]:
        raise ValueError(
            f"`region` names attribute {a}, which has a matrix-valued "
            f"kernel covariance; its mass in a region is not supported. "
            f"Leave it out of the region to integrate over it.")
    d = g["dim"]
    if d == 0:
        raise ValueError(
            f"`region` names attribute {a}, which has no coordinates "
            f"(a relative attribute of one value); leave it out.")
    if spec["type"] == "box":
        B = spec["bounds"]
        if B.shape[0] not in (1, d):
            raise ValueError(
                f"`region` for attribute {a}: the box has {B.shape[0]} "
                f"rows, but the attribute has {d} coordinates; give one "
                f"row, or one per coordinate.")
        if not g["diag"]:
            raise ValueError(
                f"`region` for attribute {a}: a box has no closed form "
                f"on a relative attribute of three or more values, whose "
                f"coordinates are correlated. Use a Gaussian region "
                f"('gaussian', centre, sd), or r = 2.")
        if g["mode"] != "none" and np.any(
                B[:, 1] - B[:, 0] > g["period"] * (1 + 1e-12)):
            raise ValueError(
                f"`region` for attribute {a}: on a periodic attribute a "
                f"box spans at most one period ({g['period']:g}).")
    else:
        if spec["centre"].size not in (1, d):
            raise ValueError(
                f"`region` for attribute {a}: the Gaussian's centre has "
                f"{spec['centre'].size} values, but the attribute has {d} "
                f"coordinates; give one, or one per coordinate.")
        if g["mode"] != "none" and not g["diag"]:
            raise ValueError(
                f"`region` for attribute {a}: a Gaussian region on a "
                f"periodic relative attribute of three or more values is "
                f"not supported.")


def _region_share(coords, spec, g):
    """Each tuple's kernel's share of the region (unit-mass kernels)."""
    d, n = coords.shape
    if n == 0:
        return np.zeros(0)
    sigma = g["sigma"]
    M = g["M"]
    if g["diag"]:
        sds = sigma / np.sqrt(np.diag(M))
        share = np.ones(n)
        for k in range(d):
            if spec["type"] == "box":
                B = spec["bounds"]
                lo, hi = B[k if B.shape[0] > 1 else 0]
                reg = ("box", lo, hi)
            else:
                c = spec["centre"]
                reg = ("gauss", float(c[k if c.size > 1 else 0]), spec["sd"])
            share = share * _coord_share(coords[k], reg, sds[k], g["mode"],
                                         g["period"])
        return share
    # A Gaussian region on correlated coordinates (non-periodic):
    # integral of N(x; c, C) exp(-|x - x0|^2 / 2 s^2), C = sigma^2 M^-1.
    s = spec["sd"]
    x0 = np.broadcast_to(spec["centre"], (d,))
    C = sigma ** 2 * np.linalg.inv(M)
    S = C + s ** 2 * np.eye(d)
    fac = 1.0 / np.sqrt(np.linalg.det(np.eye(d) + C / s ** 2))
    diff = coords - x0[:, None]
    q = np.einsum("im,ij,jm->m", diff, np.linalg.inv(S), diff)
    return fac * np.exp(-0.5 * q)


def _ncdf_diff(a, b):
    """Phi(b) - Phi(a), computed on the side of zero that keeps it exact."""
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    upper = a > 0
    out = np.where(upper,
                   0.5 * erfc(a / _SQRT2) - 0.5 * erfc(b / _SQRT2),
                   0.5 * erfc(-b / _SQRT2) - 0.5 * erfc(-a / _SQRT2))
    return out


def _seg(a, b, mu, sd, x0=None, s=None):
    """Integral over [a, b] of g(y) phi_sd(y - mu), with g = 1 (a box) or
    g(y) = exp(-(y - x0)^2 / 2 s^2) (a Gaussian region). Zero where the
    interval is empty."""
    a, b, mu = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float),
                                   np.asarray(mu, float))
    if s is None:
        out = _ncdf_diff((a - mu) / sd, (b - mu) / sd)
    else:
        v = sd * sd + s * s
        tau = sd * s / np.sqrt(v)
        nu = (mu * s * s + x0 * sd * sd) / v
        amp = (s / np.sqrt(v)) * np.exp(-(mu - x0) ** 2 / (2.0 * v))
        out = amp * _ncdf_diff((a - nu) / tau, (b - nu) / tau)
    return np.where(b > a, out, 0.0)


def _coord_share(c, reg, sd, mode, period):
    """One coordinate's share for kernels centred at ``c``."""
    kind, p1, p2 = reg
    if mode == "none":
        if kind == "box":
            return _seg(p1, p2, c, sd)
        return _seg(-np.inf, np.inf, c, sd, p1, p2)
    P = period
    if kind == "box":
        a, b, x0, s = p1, p2, None, None
    else:
        a, b, x0, s = p1 - P / 2.0, p1 + P / 2.0, p1, p2
    total = np.zeros_like(c, dtype=float)
    if mode == "full":
        reach = _IMAGE_REACH * sd
        m_lo = int(np.floor((a - reach - c.max()) / P))
        m_hi = int(np.ceil((b + reach - c.min()) / P))
        for m in range(m_lo, m_hi + 1):
            total += _seg(a, b, c + m * P, sd, x0, s)
        return total
    # single image: the kernel is phi(wrap(y - c)), a Gaussian cut to one
    # period around each image of c, renormalized to unit mass.
    m_lo = int(np.floor((a - c.max()) / P - 0.5)) - 1
    m_hi = int(np.ceil((b - c.min()) / P + 0.5)) + 1
    for m in range(m_lo, m_hi + 1):
        cm = c + m * P
        total += _seg(np.maximum(a, cm - P / 2.0), np.minimum(b, cm + P / 2.0),
                      cm, sd, x0, s)
    return total / _ncdf_diff(-P / (2.0 * sd), P / (2.0 * sd))
