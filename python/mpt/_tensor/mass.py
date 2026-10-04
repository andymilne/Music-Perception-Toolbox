"""The total mass of a MAET: ``mass_maet``.

A MAET is a sum of kernels, one per tuple, each carrying its tuple's
weight product. Taken with unit mass, each kernel contributes its weight
product, so the mass of the whole density is the sum of its tuples'
weight products. The density is a Cartesian product across attributes
within each event, so the sum factors:

    M = sum_events prod_attributes S_a^(event),

with ``S_a^(event)`` the sum of the attribute's weight products over its
tuples in that event. The joint tuple set is never built.

The total mass is the size of what a density holds: its number of
tuples, where every weight is 1. It turns an unnormalized quantity into a
share, a one-sided similarity into the share of a context's tuples that
match a query, for example.

The swept counterpart, :func:`swept_mass`, lives in :mod:`.swept`.
"""
from __future__ import annotations

import numpy as np

from .premaet import is_pre_maet


def mass_maet(dens, *, verbose=False):
    r"""Total mass of a multi-attribute expectation tensor (MAET).

    **Overview.** The density of a MAET is a weighted sum of Gaussian
    kernels, one per tuple. Each kernel is taken here with unit mass, so
    the mass of the whole density is the sum of its tuples' weight
    products,

    .. math:: M = \sum_j w_j,

    the integral of the density :func:`eval_maet` returns under
    ``normalize='gaussian'``. Where every weight is 1 it is the number of
    tuples: at r = 2, the number of ordered or unordered pairs of values,
    as the attribute is exchangeable or not.

    The mass is the natural normalizer for a count. The one-sided
    similarity (``normalize='oneSidedDenom'`` in :func:`sim_maet`) of a
    context with a query counts, in units of the query, the context's
    tuples that match it; multiplied by the query's mass over the
    context's, it is their share of the context's tuples.

    Parameters
    ----------
    dens : MaetDensity, pre-MAET, or list of them
        The density, from :func:`build_maet`, or a whole pre-MAET, which
        is built here. A list gives one value per entry.
    verbose : bool, default False
        Passed to :func:`build_maet` for a pre-MAET.

    Returns
    -------
    float, or np.ndarray for a list

    Notes
    -----
    The density is a Cartesian product across attributes within each
    event, so the sum factors into per-event, per-attribute sums, and the
    joint tuple set is never built.

    See Also
    --------
    swept_mass : the mass at each of a list of sweep values.
    sim_maet : the one-sided similarity a mass normalizes.
    """
    if isinstance(dens, (list, tuple)) and not is_pre_maet(dens):
        return np.array([mass_maet(d, verbose=verbose) for d in dens],
                        dtype=float)
    if is_pre_maet(dens):
        from .build import build_maet
        dens = build_maet(dens, verbose=verbose)
    return _density_mass(dens)


# -------------------------------------------------------------------
#  One density
# -------------------------------------------------------------------

def _density_mass(dens):
    from .build import _enum_flat_attr, _nested_enum_indices

    A = int(dens.n_attrs)
    N = int(dens.n)
    P = [np.asarray(p, dtype=np.float64) for p in dens.p_attr]
    W = [np.asarray(w, dtype=np.float64) for w in dens.w]
    r_vec = [int(v) for v in np.atleast_1d(dens.r)]
    exch = [bool(v) for v in np.atleast_1d(dens.exch)]
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
                np.zeros(P[a].shape[0]), ever_valid, r_vec[a], exch[a],
                np.ones(P[a].shape[0]))
        perm.append(np.asarray(pm, dtype=np.intp))

    total = 0.0
    for n in range(N):
        prod_t = 1.0
        for a in range(A):
            p_col = P[a][:, n]
            w_col = W[a][:, n]
            absent = np.isnan(p_col) | np.isnan(w_col)
            w_fill = np.where(absent, 0.0, w_col)
            prod_t *= float(np.prod(w_fill[perm[a]], axis=0).sum())
        total += prod_t
    return float(total)
