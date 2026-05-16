"""Expectation tensor construction, evaluation, and similarity.

The central data structure is :class:`ExpTensDensity`, a precomputed
Gaussian-mixture density representing the expected distribution of
r-ads (ordered r-tuples) from a weighted pitch multiset.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from itertools import permutations
from math import comb as _math_comb, factorial

import numpy as np
from scipy.special import comb as _comb

from ._utils import estimate_comp_time, maybe_print_batched_estimate, validate_weights
from ._kernel import gaussian_kernel_sum
from ._defaults import _with_dispatch_scope
from .spectra import add_spectra



# ---------------------------------------------------------------------
#  Re-exports from migrated sub-modules.
#
#  As of the Tranche-2 (phases 1+2) refactor, the density classes, the
#  cross-event preprocessing utilities, and the windowing layer all
#  live in the ``_tensor`` sub-package. They are re-exported here so
#  external code that imports from ``mpt.tensor`` (and the toolbox's
#  own ``__init__.py``) continues to work unchanged.
#
#  See :doc:`/ARCHITECTURE` §3 ("Code layering") for the target
#  structure. The build, eval, and cosine machinery will move in a
#  subsequent phase.
# ---------------------------------------------------------------------

from ._tensor.density import (
    ExpTensDensity,
    MaetDensity,
    WindowedMaetDensity,
    _nchoosek_indices,
    _coerce_attr_matrix,
    _canonicalise_groups,
    _canon_groups_cell_form,
    _canon_groups_vector_form,
    _normalise_weights_ma,
    _broadcast_attr_weight,
    _cartesian_indices,
)
from ._tensor.preprocessing import (
    difference_events,
    bind_events,
    simplex_vertices,
)
from ._tensor.windowing import (
    WindowedSimilarityPeriodicApproxWarning,
    window_tensor,
    windowed_similarity,
    _windowed_inner_product,
    _evaluate_window_on_query,
)



# -------------------------------------------------------------------
#  build_exp_tens  (public dispatcher)
# -------------------------------------------------------------------


def build_exp_tens(p, w, *args, verbose: bool = True) -> ExpTensDensity | MaetDensity:
    """Precompute an r-ad expectation tensor density object.

    Dispatches on the type of the first argument:

      - numeric 1-D array, or a flat list/tuple of numbers -> single-
        attribute path, returns :class:`ExpTensDensity`.
      - list/tuple of attribute matrices (each element itself an
        array-like with ``len(...)`` > 0 or a 2-D ndarray) -> multi-
        attribute path, returns :class:`MaetDensity`.

    Single-attribute signature (legacy, unchanged)::

        build_exp_tens(p, w, sigma, r, is_rel, is_per, period, *, verbose=True)

    Multi-attribute signature::

        build_exp_tens(p_attr, w, sigma_vec, r_vec, groups,
                       is_rel_vec, is_per_vec, period_vec, *, verbose=True)

    The two paths differ only in positional-argument count (7 vs 8) and
    in the types of the individual arguments. See the MAET specification
    (``multi_attribute_tensor_specification.md``) §2 and §6 for the
    multi-attribute semantics.

    Parameters (single-attribute path)
    ----------------------------------
    p : array-like
        Pitch or position values (1-D, length *N*).
    w : None, scalar, or array-like
        Weights. ``None`` or a scalar for all ones or a uniform
        broadcast; a length-*N* vector for per-event values. See the
        toolbox's standard broadcast convention in User Guide §4.
    sigma : float
        Standard deviation of the Gaussian kernel.
    r : int
        Tuple size (positive integer; ``r >= 2`` if ``is_rel`` is true).
    is_rel : bool
        If true, use transposition-invariant (relative) quadratic form
        (effective dim = ``r - 1``).
    is_per : bool
        If true, wrap differences to the periodic interval
        ``[-period/2, period/2)``.
    period : float
        Period for periodic wrapping (e.g., 1200 for one octave in
        cents, or the cycle length for rhythmic analyses).

    Parameters (multi-attribute path)
    ---------------------------------
    p_attr : list or tuple of array-like
        Length-*A* sequence of attribute value matrices, each of shape
        K_a x N. A 1-D input is taken as a 1 x N row (K_a = 1).
    w : None, scalar, or list/tuple of per-attribute inputs
        Top-level weight specification. A list/tuple has length *A*,
        with each per-attribute input being ``None``, a scalar, a 1-D
        array of length *N* (per-event) or *K_a* (per-slot), a 2-D
        array of shape (1, N), (K_a, 1), or (K_a, N). See Section 2.8
        of the MAET specification.
    sigma_vec : (G,) array-like of float
        Per-group Gaussian widths.
    r_vec : (A,) array-like of int
        Per-attribute tuple sizes.
    groups : None, (A,) array-like of int, or list of length G
        Group assignment. ``None`` (or empty) defaults to each
        attribute its own singleton group. A length-*A* vector gives
        the 0-indexed group index per attribute (contiguous 0..G-1).
        A length-*G* list of attribute-index lists gives an explicit
        partition.
    is_rel_vec, is_per_vec : (G,) array-like of bool
        Per-group isRel and isPer flags.
    period_vec : (G,) array-like of float
        Per-group periods (use 0 for groups that are not periodic).

    Returns
    -------
    ExpTensDensity or MaetDensity
        Depending on which path is taken.

    See Also
    --------
    ExpTensDensity, MaetDensity, eval_exp_tens, cos_sim_exp_tens
    """
    if _looks_like_multi_attr(p):
        if len(args) != 6:
            raise ValueError(
                f"Multi-attribute call expects 8 positional arguments "
                f"(p_attr, w, sigma_vec, r_vec, groups, is_rel_vec, "
                f"is_per_vec, period_vec); got {2 + len(args)}."
            )
        sigma_vec, r_vec, groups, is_rel_vec, is_per_vec, period_vec = args
        return _build_exp_tens_ma(
            p, w, sigma_vec, r_vec, groups,
            is_rel_vec, is_per_vec, period_vec,
            verbose=verbose,
        )
    else:
        if len(args) != 5:
            raise ValueError(
                f"Single-attribute call expects 7 positional arguments "
                f"(p, w, sigma, r, is_rel, is_per, period); got "
                f"{2 + len(args)}."
            )
        sigma, r, is_rel, is_per, period = args
        return _build_exp_tens_sa(
            p, w, sigma, r, is_rel, is_per, period,
            verbose=verbose,
        )


def _looks_like_multi_attr(p) -> bool:
    """Return True if *p* is a list/tuple of attribute matrices.

    MA triggers require a list/tuple whose first element is itself an
    array-like (a list, tuple, or ndarray of length >= 1, or a 2-D
    ndarray). A flat list of scalars like ``[0, 4, 7]`` or a 1-D ndarray
    is routed to the single-attribute path — matching the original
    semantics where such inputs denote a single pitch multiset.
    """
    if isinstance(p, np.ndarray):
        # 2-D and higher would be ambiguous; require the explicit list
        # form for multi-attribute calls.
        return False
    if not isinstance(p, (list, tuple)):
        return False
    if len(p) == 0:
        return False
    first = p[0]
    if isinstance(first, np.ndarray):
        return True
    if isinstance(first, (list, tuple)):
        return True
    return False


# -------------------------------------------------------------------
#  _build_exp_tens_ma  (multi-attribute path)
# -------------------------------------------------------------------


def _build_exp_tens_ma(
    p_attr,
    w,
    sigma_vec,
    r_vec,
    groups,
    is_rel_vec,
    is_per_vec,
    period_vec,
    *,
    verbose: bool = True,
) -> MaetDensity:
    """Multi-attribute expectation tensor builder.

    Private: users call :func:`build_exp_tens`, which dispatches here
    when given a list/tuple of attribute matrices as the first argument.

    The body performs only the cheap input validation and metadata
    work and returns a :class:`MaetDensity` whose per-tuple arrays
    (``u_perm``, ``v_comb``, ``centres``, ``w_j``, ``wv_comb``,
    ``event_of_j``, ``event_of_k``, ``n_j``, ``n_k``) are deferred to
    first access. See :class:`MaetDensity` for the lazy semantics.
    """
    # --- Input normalisation (eager) ------------------------------

    if not isinstance(p_attr, (list, tuple)) or len(p_attr) == 0:
        raise ValueError(
            "p_attr must be a non-empty list/tuple of attribute matrices."
        )

    p_attr = [_coerce_attr_matrix(M) for M in p_attr]
    A = len(p_attr)

    Ns = np.array([M.shape[1] for M in p_attr])
    if not np.all(Ns == Ns[0]):
        raise ValueError(
            f"All attribute matrices must share N (event count); got "
            f"{Ns.tolist()}."
        )
    N = int(Ns[0])

    K_a = np.array([M.shape[0] for M in p_attr], dtype=np.intp)

    r_vec = np.asarray(r_vec, dtype=np.intp).ravel()
    if r_vec.size != A:
        raise ValueError(
            f"r_vec must have length {A} (n attributes), got {r_vec.size}."
        )
    if np.any(r_vec < 1):
        raise ValueError("All r_a must be positive integers.")

    group_of_attr, attrs_of_group, G = _canonicalise_groups(groups, A)

    sigma_vec  = np.asarray(sigma_vec,  dtype=np.float64).ravel()
    is_rel_vec = np.asarray(is_rel_vec, dtype=bool).ravel()
    is_per_vec = np.asarray(is_per_vec, dtype=bool).ravel()
    period_vec = np.asarray(period_vec, dtype=np.float64).ravel()

    for name, vec in (("sigma_vec",  sigma_vec),
                      ("is_rel_vec", is_rel_vec),
                      ("is_per_vec", is_per_vec),
                      ("period_vec", period_vec)):
        if vec.size != G:
            raise ValueError(
                f"{name} must have length {G} (n groups), got {vec.size}."
            )

    for a in range(A):
        g = int(group_of_attr[a])
        if is_rel_vec[g] and r_vec[a] < 2:
            warnings.warn(
                f"is_rel = True on group {g} combined with r_a = 1 for "
                f"attribute {a} produces a degenerate (constant) density. "
                f"For cross-event translation invariance, use "
                f"`difference_events` as a preprocessing step."
            )

    w_list = _normalise_weights_ma(w, A, K_a, N)

    # Eager per-event / per-attribute non-NaN slot count check. The
    # heavy r-ad enumeration is deferred to first access of a lazy
    # field, but this validation is cheap (one NaN scan per (a, n))
    # and users reasonably expect malformed inputs to fail fast at the
    # build call rather than later on first downstream consumer call.
    for n in range(N):
        for a in range(A):
            r_a = int(r_vec[a])
            valid_count = int(np.sum(~np.isnan(p_attr[a][:, n])))
            if valid_count < r_a:
                raise ValueError(
                    f"Event {n}, attribute {a} has {valid_count} non-NaN "
                    f"slot(s) but r_a = {r_a}."
                )

    # --- Per-attribute dim (eager; needed by callers without
    # materialisation) -------------------------------------------------
    dim_per_attr = np.empty(A, dtype=np.intp)
    for a in range(A):
        g = int(group_of_attr[a])
        r_a = int(r_vec[a])
        if is_rel_vec[g]:
            dim_per_attr[a] = r_a - 1 if r_a >= 2 else 0
        else:
            dim_per_attr[a] = r_a
    dim = int(dim_per_attr.sum())

    if verbose:
        print(
            f"build_exp_tens (MAET): {A} attributes, {G} groups, "
            f"{N} events (per-tuple arrays deferred to first access)."
        )

    # --- Lazy closure: heavy per-event/per-attribute build ----------
    #
    # Capture the validated inputs by closure. The closure is invoked
    # at most once per MaetDensity, on first access of any lazy
    # field (n_j, n_k, centres, u_perm, v_comb, w_j, wv_comb,
    # event_of_j, event_of_k). Returns a dict consumed by
    # MaetDensity._materialise.

    def _build_lazy():
        return _ma_build_perm_arrays(
            p_attr=p_attr, w_list=w_list, r_vec=r_vec,
            group_of_attr=group_of_attr,
            is_rel_vec=is_rel_vec,
            N=N, A=A,
        )

    return MaetDensity(
        tag="MaetDensity",
        n_attrs=A,
        n_groups=G,
        n=N,
        group_of_attr=group_of_attr,
        attrs_of_group=attrs_of_group,
        r=r_vec,
        k=K_a,
        p_attr=p_attr,
        w=w_list,
        sigma=sigma_vec,
        is_rel=is_rel_vec,
        is_per=is_per_vec,
        period=period_vec,
        dim=dim,
        dim_per_attr=dim_per_attr,
        _build_lazy=_build_lazy,
    )


def _ma_build_perm_arrays(
    *,
    p_attr,
    w_list,
    r_vec,
    group_of_attr,
    is_rel_vec,
    N,
    A,
):
    """Heavy per-event / per-attribute r-ad enumeration and assembly.

    Extracted from the the eager-build ``_build_exp_tens_ma`` body so it can be
    invoked lazily on first access of a per-tuple field. Returns a
    dict of the nine lazy-target fields:
    ``n_j, n_k, centres, u_perm, v_comb, w_j, wv_comb,
    event_of_j, event_of_k``.

    Logic is unchanged from the eager build; only when it runs
    has changed.
    """
    from itertools import combinations as _combinations

    perm_idx = [[None] * A for _ in range(N)]
    comb_idx = [[None] * A for _ in range(N)]
    perm_w   = [[None] * A for _ in range(N)]
    comb_w   = [[None] * A for _ in range(N)]

    for n in range(N):
        for a in range(A):
            val_col = p_attr[a][:, n]
            valid = np.nonzero(~np.isnan(val_col))[0].astype(np.intp)
            K_na = int(valid.size)
            r_a = int(r_vec[a])
            if K_na < r_a:
                raise ValueError(
                    f"Event {n}, attribute {a} has {K_na} non-NaN "
                    f"slot(s) but r_a = {r_a}."
                )

            collapsed = False
            if r_a == 1 and K_na > 1:
                vals_valid = val_col[valid]
                _, first_idx, inverse = np.unique(
                    vals_valid, return_index=True, return_inverse=True
                )
                if first_idx.size < K_na:
                    w_col_orig = w_list[a][:, n]
                    w_col_local = w_col_orig.copy()
                    summed = np.zeros(first_idx.size, dtype=np.float64)
                    np.add.at(summed, inverse, w_col_orig[valid])
                    w_col_local[valid[first_idx]] = summed
                    valid = valid[first_idx]
                    K_na = int(valid.size)
                    collapsed = True

            comb_list = list(_combinations(valid.tolist(), r_a))
            comb_mat = np.array(comb_list, dtype=np.intp).T  # r_a x C

            if r_a == 1:
                perm_mat = comb_mat.copy()
            else:
                all_perms = np.array(
                    list(permutations(range(r_a))), dtype=np.intp
                ).T  # r_a x r_a!
                n_combs = comb_mat.shape[1]
                n_perms = all_perms.shape[1]
                perm_mat = np.empty(
                    (r_a, n_combs * n_perms), dtype=np.intp
                )
                for pp in range(n_perms):
                    perm_mat[:, pp * n_combs:(pp + 1) * n_combs] = \
                        comb_mat[all_perms[:, pp], :]

            perm_idx[n][a] = perm_mat
            comb_idx[n][a] = comb_mat

            w_col = w_col_local if collapsed else w_list[a][:, n]
            if r_a == 1:
                perm_w[n][a] = w_col[perm_mat].ravel()
                comb_w[n][a] = w_col[comb_mat].ravel()
            else:
                perm_w[n][a] = np.prod(w_col[perm_mat], axis=0)
                comb_w[n][a] = np.prod(w_col[comb_mat], axis=0)

    n_j_per = np.array(
        [int(np.prod([perm_idx[n][a].shape[1] for a in range(A)]))
         for n in range(N)],
        dtype=np.intp,
    )
    n_k_per = np.array(
        [int(np.prod([comb_idx[n][a].shape[1] for a in range(A)]))
         for n in range(N)],
        dtype=np.intp,
    )
    n_j = int(n_j_per.sum())
    n_k = int(n_k_per.sum())

    u_perm = [np.empty((int(r_vec[a]), n_j), dtype=np.float64)
              for a in range(A)]
    v_comb = [np.empty((int(r_vec[a]), n_k), dtype=np.float64)
              for a in range(A)]
    w_j = np.ones(n_j, dtype=np.float64)
    wv_comb = np.ones(n_k, dtype=np.float64)
    event_of_j = np.empty(n_j, dtype=np.intp)
    event_of_k = np.empty(n_k, dtype=np.intp)

    off_j = 0
    off_k = 0
    for n in range(N):
        nJh = int(n_j_per[n])
        nKh = int(n_k_per[n])

        sizes_perm = [perm_idx[n][a].shape[1] for a in range(A)]
        sizes_comb = [comb_idx[n][a].shape[1] for a in range(A)]
        idx_perm = _cartesian_indices(sizes_perm)
        idx_comb = _cartesian_indices(sizes_comb)

        wJh = np.ones(nJh, dtype=np.float64)
        wKh = np.ones(nKh, dtype=np.float64)

        for a in range(A):
            r_a = int(r_vec[a])
            val_col = p_attr[a][:, n]

            slot_perm = perm_idx[n][a][:, idx_perm[a]]   # r_a x nJh
            u_perm[a][:, off_j:off_j + nJh] = val_col[slot_perm]
            wJh *= perm_w[n][a][idx_perm[a]]

            slot_comb = comb_idx[n][a][:, idx_comb[a]]   # r_a x nKh
            v_comb[a][:, off_k:off_k + nKh] = val_col[slot_comb]
            wKh *= comb_w[n][a][idx_comb[a]]

        w_j[off_j:off_j + nJh]       = wJh
        wv_comb[off_k:off_k + nKh]   = wKh
        event_of_j[off_j:off_j + nJh] = n
        event_of_k[off_k:off_k + nKh] = n

        off_j += nJh
        off_k += nKh

    centres = []
    for a in range(A):
        g = int(group_of_attr[a])
        r_a = int(r_vec[a])
        if is_rel_vec[g]:
            if r_a >= 2:
                centres.append(u_perm[a][1:, :] - u_perm[a][:1, :])
            else:
                centres.append(np.empty((0, n_j), dtype=np.float64))
        else:
            centres.append(u_perm[a].copy())

    return dict(
        n_j=n_j,
        n_k=n_k,
        centres=centres,
        u_perm=u_perm,
        v_comb=v_comb,
        w_j=w_j,
        wv_comb=wv_comb,
        event_of_j=event_of_j,
        event_of_k=event_of_k,
    )


# -------------------------------------------------------------------
#  _build_exp_tens_sa  (single-attribute legacy path)
# -------------------------------------------------------------------


def _build_exp_tens_sa(
    p: np.ndarray,
    w: np.ndarray | None,
    sigma: float,
    r: int,
    is_rel: bool,
    is_per: bool,
    period: float,
    *,
    verbose: bool = True,
) -> ExpTensDensity:
    """Precompute an r-ad expectation tensor density object.

    Precomputes the tuple index sets, pitch or position matrices,
    weight vectors, and (for the relative case) reduced interval
    centres for the weighted multiset (*p*, *w*). The returned
    object can be passed to :func:`eval_exp_tens` and
    :func:`cos_sim_exp_tens` in place of the raw arguments, avoiding
    redundant recomputation across multiple calls.

    Parameters
    ----------
    p : array-like
        Pitch or position values.
    w : array-like or None
        Weights (``None`` or empty for all ones).
    sigma : float
        Gaussian kernel standard deviation.
    r : int
        Tuple size (≥ 2 when *is_rel* is True).
    is_rel : bool
        Transposition-invariant (relative) quadratic form.
    is_per : bool
        Periodic wrapping to ``[-period/2, period/2)``.
    period : float
        Period for wrapping.
    verbose : bool
        Print progress information.

    Returns
    -------
    ExpTensDensity
        Precomputed density object.

    References
    ----------
    Milne, A. J., Sethares, W. A., Laney, R., & Sharp, D. B.
    (2011). Modelling the similarity of pitch collections with
    expectation tensors. *Journal of Mathematics and Music*,
    5(1), 1–20.
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    w = validate_weights(w, len(p))

    r = int(r)
    if r < 1 or r != int(r):
        raise ValueError("r must be a positive integer.")
    if r > len(p):
        raise ValueError("r must not exceed the number of values.")
    if is_rel and r < 2:
        raise ValueError("For relative densities, r must be at least 2.")

    # For r = 1, the density depends on the source multiset only through its
    # measure on the pitch line: events with equal pitch contribute additively
    # to the same Gaussian kernel, so they can be collapsed to a single event
    # whose weight is the sum of the originals. This is mathematically exact
    # at r = 1 and reduces downstream work proportionally to the number of
    # repeated pitches in the input. (For r >= 2, multiplicity in the source
    # multiset matters for the within-tuple structure, so collapsing would
    # alter the density and is therefore not applied.)
    if r == 1 and len(p) > 0:
        p_unique, inverse = np.unique(p, return_inverse=True)
        if len(p_unique) < len(p):
            w_summed = np.zeros(len(p_unique), dtype=np.float64)
            np.add.at(w_summed, inverse, w)
            p, w = p_unique, w_summed

    dim = r - int(is_rel)
    n = len(p)

    if verbose:
        # Cheap, allocation-free scalar — no longer reports "building
        # n_j tuples" because the tuple build is deferred to first
        # access of a per-tuple field.
        n_j_eager = factorial(r) * int(_comb(n, r, exact=True))
        print(
            f"build_exp_tens: SA density with K={n}, r={r} "
            f"(per-tuple arrays deferred; n_j={n_j_eager} on first "
            f"access)."
        )

    return ExpTensDensity(
        p=p,
        w=w,
        sigma=sigma,
        r=r,
        is_rel=is_rel,
        is_per=is_per,
        period=period,
        dim=dim,
    )


# -------------------------------------------------------------------
#  eval_exp_tens
# -------------------------------------------------------------------


# -------------------------------------------------------------------
#  eval_exp_tens  (public dispatcher)
# -------------------------------------------------------------------


@_with_dispatch_scope
def eval_exp_tens(*args,
                  normalize: str = "none",
                  dedup: bool = True,
                  spectrum=None,
                  precision: int | None = None,
                  method: str = "auto",
                  truncation_sigmas: float | None = None,
                  kernel_precision: str | None = None,
                  verbose: bool = True) -> np.ndarray:
    """Evaluate an expectation tensor density at query points.

    Unified entry point. Accepts five input forms, dispatched on the
    type of the first argument:

    **Pre-built density input** (plus polymorphic lists):

    - ``eval_exp_tens(dens, X)`` — scalar density.
      Returns ``(nQ,)``.
    - ``eval_exp_tens(dens, X, normalize)`` — same with positional
      ``normalize``.
    - ``eval_exp_tens([d1, d2, …], X)`` — list of densities at shared
      query ``X``. Returns ``(M, nQ)`` — every density evaluated at
      every query point. ``normalize`` (positional or kwarg) applies
      to all rows.

    **Raw single-attribute scalar input**:

    - ``eval_exp_tens(p, w, sigma, r, is_rel, is_per, period, X)``.
      Returns ``(nQ,)``.
    - ``eval_exp_tens(p, w, sigma, r, is_rel, is_per, period, X, normalize)``.

    **Raw single-attribute batched input**:

    - ``eval_exp_tens(P, W, sigma, r, is_rel, is_per, period, X)``
      with ``P`` and ``W`` 2-D ``(M, K)`` matrices (rows are chords).
      Returns ``(M, nQ)``.

    **Raw multi-attribute scalar input**:

    - ``eval_exp_tens(p_attr, w, sigma_vec, r_vec, groups,
      is_rel_vec, is_per_vec, period_vec, X)``. Returns ``(nQ,)``.

    Parameters
    ----------
    *args
        Positional arguments depending on input form.
    normalize : {'none', 'gaussian', 'pdf'}, default 'none'
        Density normalisation. Accepted as the trailing positional arg
        in original call patterns, or as a keyword.
    dedup : bool, default True
        Deduplicate structurally-identical chords (canonical-form,
        SA-only). For list/batch input only.
    spectrum : list/tuple, optional
        Per-row :func:`add_spectra` parameters. Raw SA modes only.
    precision : int, optional
        FP-noise tolerance for canonical-form dedup. Raw SA batched
        only.
    method : {'auto', 'centres', 'mobius'}, default 'auto'
        SA-path evaluation strategy. ``'auto'`` lets the dispatcher
        choose between the centres-array path (fast
        at low r) and the Möbius point evaluator (much
        faster at r >= 3 since it bypasses the ``(dim, n_j)`` centres
        tensor whose memory and runtime scale as ``K!/(K-r)!``).
        ``'centres'`` forces the centres path; ``'mobius'`` forces the
        Möbius method. Currently a no-op on the MA path (MA always
        uses centres).
    verbose : bool, default True
        Print progress.

    Returns
    -------
    np.ndarray
        Shape ``(nQ,)`` for scalar density / raw SA scalar / raw MA
        scalar; shape ``(M, nQ)`` for density list / raw SA batched.

    Notes
    -----
    Numerical precision envelope for ``method='mobius'``.

    The Möbius point evaluator is exact to floating-point precision
    when ``K >= r + 2`` and σ is not catastrophically small relative
    to P. The dispatcher enforces these conditions structurally — it
    falls back to the centres path when ``K < r + 2``, when
    ``σ/P > 0.03`` in periodic-relative mode, or when the Möbius
    output contains non-finite values (post-hoc safety net).

    What is *not* currently caught: a finite, but slightly inaccurate
    output from sub-catastrophic Möbius cancellation. None has been
    observed in extensive testing, but a sum-level cancellation
    diagnostic that would close this residual gap is planned. See
    :func:`cos_sim_exp_tens` Notes for the parallel discussion on the
    inner-product path.

    See Also
    --------
    build_exp_tens, cos_sim_exp_tens
    eval_exp_tens_raw : deprecated; superseded by raw input mode here.
    """
    if len(args) < 2:
        raise TypeError(
            "eval_exp_tens requires at least 2 positional arguments."
        )

    a = args[0]

    # ------------------------------------------------------------------
    # Density input dispatch
    # ------------------------------------------------------------------
    is_density_scalar = isinstance(
        a, (ExpTensDensity, MaetDensity, WindowedMaetDensity)
    )
    intends_density_list = False
    if isinstance(a, (list, tuple)):
        if len(a) == 0:
            intends_density_list = True
        elif isinstance(
            a[0], (ExpTensDensity, MaetDensity, WindowedMaetDensity)
        ):
            intends_density_list = True
    elif isinstance(a, np.ndarray) and a.dtype == object:
        intends_density_list = True

    if is_density_scalar or intends_density_list:
        # Density mode: 2 or 3 positional args (dens, x[, normalize]).
        if len(args) == 2:
            dens, x = args
        elif len(args) == 3:
            dens, x, normalize = args
        else:
            raise TypeError(
                f"Density input mode expects 2 or 3 positional "
                f"arguments (dens, x[, normalize]); got {len(args)}."
            )
        if spectrum is not None:
            raise TypeError(
                "'spectrum' kwarg is only valid in raw SA input mode."
            )
        if precision is not None:
            raise TypeError(
                "'precision' kwarg is only valid in raw SA batched input mode."
            )
        if is_density_scalar:
            return _eval_exp_tens_scalar(
                dens, x, normalize, method=method,
                truncation_sigmas=truncation_sigmas,
                kernel_precision=kernel_precision,
                verbose=verbose,
            )
        return _eval_exp_tens_density_list(
            dens, x, normalize, dedup=dedup, method=method,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Raw multi-attribute dispatch
    # ------------------------------------------------------------------
    if _looks_like_multi_attr(a):
        # 9 or 10 positional args.
        if len(args) == 9:
            (p_attr, w_in, sigma_vec, r_vec, groups,
             is_rel_vec, is_per_vec, period_vec, x) = args
        elif len(args) == 10:
            (p_attr, w_in, sigma_vec, r_vec, groups,
             is_rel_vec, is_per_vec, period_vec, x, normalize) = args
        else:
            raise TypeError(
                f"Raw multi-attribute input expects 9 or 10 positional "
                f"arguments (p_attr, w, sigma_vec, r_vec, groups, "
                f"is_rel_vec, is_per_vec, period_vec, x[, normalize]); "
                f"got {len(args)}."
            )
        if spectrum is not None:
            raise TypeError(
                "'spectrum' kwarg is only supported in raw single-attribute "
                "input mode."
            )
        if precision is not None:
            raise TypeError(
                "'precision' kwarg is only valid in raw SA batched input mode."
            )
        return _eval_exp_tens_raw_ma_scalar(
            p_attr, w_in, sigma_vec, r_vec, groups,
            is_rel_vec, is_per_vec, period_vec, x, normalize,
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Raw single-attribute dispatch (1-D = scalar, 2-D = batch)
    # ------------------------------------------------------------------
    if len(args) == 8:
        p, w, sigma, r_, is_rel, is_per, period, x = args
    elif len(args) == 9:
        p, w, sigma, r_, is_rel, is_per, period, x, normalize = args
    else:
        raise TypeError(
            f"Raw single-attribute input expects 8 or 9 positional "
            f"arguments (p, w, sigma, r, is_rel, is_per, period, x"
            f"[, normalize]); got {len(args)}."
        )

    try:
        a_arr = np.asarray(a, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"First argument must be a density object, list of densities, "
            f"numeric array (1-D for a single chord, 2-D for a batch), or "
            f"list of per-attribute matrices for MA raw input; got "
            f"{type(a).__name__}."
        ) from exc

    if a_arr.ndim == 1:
        if precision is not None:
            raise TypeError(
                "'precision' kwarg is only valid for raw SA batched input."
            )
        return _eval_exp_tens_raw_sa_scalar(
            p, w, sigma, r_, is_rel, is_per, period, x, normalize,
            spectrum=spectrum, method=method, verbose=verbose,
        )
    if a_arr.ndim == 2:
        return _eval_exp_tens_raw_sa_batch(
            p, w, sigma, r_, is_rel, is_per, period, x, normalize,
            spectrum=spectrum, precision=precision,
            dedup=dedup, method=method, verbose=verbose,
        )
    raise TypeError(
        f"First argument has unsupported shape {a_arr.shape}; "
        f"raw SA input must be 1-D (single chord) or 2-D (batched)."
    )


def _eval_exp_tens_scalar(
    dens, x, normalize: str, *, method: str = "auto",
    truncation_sigmas: float | None = None,
    kernel_precision: str | None = None,
    verbose: bool,
) -> np.ndarray:
    """Density-scalar dispatch for :func:`eval_exp_tens`.

    Threads ``method`` through to :func:`_eval_exp_tens_sa` for SA densities;
    MA path ignores ``method`` (an MA Möbius method is on the v2.3 roadmap).
    Threads ``truncation_sigmas`` / ``kernel_precision`` through to the
    SA centres path; MA centres routing is deferred (Stage 3).
    """
    if isinstance(dens, WindowedMaetDensity):
        # Evaluate underlying density, multiply elementwise by window.
        underlying = _eval_exp_tens_ma(
            dens.dens, x, normalize,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
            verbose=verbose,
        )
        # Reconstruct per-attribute x_list so we can apply the window.
        x_list = _split_query_to_attr_list(dens.dens, x)
        W_vals = _evaluate_window_on_query(dens, x_list)
        return underlying * W_vals
    if isinstance(dens, MaetDensity):
        return _eval_exp_tens_ma(
            dens, x, normalize,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
            verbose=verbose,
        )
    if isinstance(dens, ExpTensDensity):
        return _eval_exp_tens_sa(
            dens, x, normalize, method=method,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
            verbose=verbose,
        )
    raise TypeError(
        f"dens must be an ExpTensDensity, MaetDensity, or "
        f"WindowedMaetDensity; got {type(dens).__name__}."
    )


def _eval_exp_tens_density_list(
    dens_list, x, normalize: str,
    *, dedup: bool, method: str = "auto",
    truncation_sigmas: float | None = None,
    kernel_precision: str | None = None,
    verbose: bool,
) -> np.ndarray:
    """Evaluate a list of densities at shared query ``x``.

    Returns ``(M, nQ)``. With ``dedup=True``, structurally-identical
    SA densities are evaluated once (canonical-form dedup via
    :func:`_chord_canonical_key`); MA densities bypass dedup.
    """
    is_scalar, dens_tuple = _normalize_density_input(dens_list, name="dens")
    # Note: in this code path we always have a list (intends_density_list
    # was True in the dispatcher), so is_scalar is always False; dens_tuple
    # is the (possibly empty) list of densities.
    m = len(dens_tuple)

    if m == 0:
        # Empty list. Output shape is (0, nQ) where nQ is undetermined
        # without inspecting x; return (0,) for a 1-D query, else (0, 0).
        try:
            x_arr = np.asarray(x, dtype=np.float64)
            if x_arr.ndim == 0:
                return np.empty((0,), dtype=np.float64)
            n_q = x_arr.shape[-1] if x_arr.ndim >= 1 else 0
            return np.empty((0, n_q), dtype=np.float64)
        except (TypeError, ValueError):
            return np.empty((0, 0), dtype=np.float64)

    # Optional dedup for SA densities only.
    use_dedup = dedup and all(isinstance(d, ExpTensDensity) for d in dens_tuple)

    if use_dedup:
        result_cache: dict = {}
        rows = []
        for d in dens_tuple:
            key, _, _ = _chord_canonical_key(
                d.p, d.w, sigma=d.sigma, r=d.r,
                is_rel=d.is_rel, is_per=d.is_per, period=d.period,
            )
            if key not in result_cache:
                result_cache[key] = _eval_exp_tens_scalar(
                    d, x, normalize, method=method,
                    truncation_sigmas=truncation_sigmas,
                    kernel_precision=kernel_precision,
                    verbose=False,
                )
            rows.append(result_cache[key])
        if verbose:
            print(
                f"eval_exp_tens: {m} densities, "
                f"{len(result_cache)} unique after canonical-form dedup."
            )
    else:
        if dedup and verbose:
            print(
                "eval_exp_tens: dedup=True requested but input includes "
                "non-SA densities; computing without dedup."
            )
        rows = [
            _eval_exp_tens_scalar(
                d, x, normalize, method=method,
                truncation_sigmas=truncation_sigmas,
                kernel_precision=kernel_precision,
                verbose=False,
            )
            for d in dens_tuple
        ]

    return np.stack(rows, axis=0)


def _eval_exp_tens_raw_sa_scalar(
    p, w, sigma, r, is_rel, is_per, period,
    x, normalize: str,
    *, spectrum=None, method: str = "auto", verbose: bool,
) -> np.ndarray:
    """Raw SA scalar dispatch: build density (with optional spectrum), evaluate."""
    if spectrum is not None:
        p_arr = np.asarray(p, dtype=np.float64)
        w_arr = (np.ones_like(p_arr) if w is None
                 else np.asarray(w, dtype=np.float64))
        p, w = add_spectra(p_arr, w_arr, *spectrum)
    dens = build_exp_tens(
        p, w, sigma, r, is_rel, is_per, period, verbose=verbose,
    )
    return _eval_exp_tens_scalar(
        dens, x, normalize, method=method, verbose=verbose,
    )


def _eval_exp_tens_raw_sa_batch(
    P, W, sigma, r, is_rel, is_per, period,
    x, normalize: str,
    *, spectrum=None, precision: int | None = None,
    dedup: bool = True, method: str = "auto",
    verbose: bool,
) -> np.ndarray:
    """Raw SA batched dispatch.

    Per-row chord-level dedup of density construction (via
    :func:`_chord_canonical_key`); evaluates each unique density once
    and maps results to all rows. Returns ``(M, nQ)``.
    """
    P = np.asarray(P, dtype=np.float64)
    if P.ndim != 2:
        raise ValueError(f"P must be 2-D for batched mode; got shape {P.shape}.")
    M, K = P.shape
    use_w = W is not None
    if use_w:
        W = np.asarray(W, dtype=np.float64)
        if W.shape != P.shape:
            raise ValueError("W must be the same shape as P.")

    # Optional input precision rounding (collapses FP-noise rows).
    if precision is not None:
        P = np.round(P, precision)
        if use_w:
            W = np.round(W, precision)

    # First pass: build canonical-form keys + density cache.
    dens_cache: dict = {}
    row_to_key: list = [None] * M
    for i in range(M):
        p_row = P[i]
        mask = ~np.isnan(p_row)
        p_valid = p_row[mask]
        if len(p_valid) < r:
            continue   # invalid row -> NaN in output
        w_valid = W[i, mask] if use_w else None
        key, p_canon, w_canon = _chord_canonical_key(
            p_valid, w_valid,
            sigma=sigma, r=r, is_rel=is_rel, is_per=is_per, period=period,
            precision=precision,
        )
        if key not in dens_cache:
            if spectrum is not None:
                p_canon, w_canon_aug = add_spectra(
                    p_canon,
                    np.ones_like(p_canon) if w_canon is None else w_canon,
                    *spectrum,
                )
                w_canon = w_canon_aug
            dens_cache[key] = build_exp_tens(
                p_canon, w_canon, sigma, r, is_rel, is_per, period,
                verbose=False,
            )
        row_to_key[i] = key

    n_unique = len(dens_cache)
    n_valid = sum(1 for k in row_to_key if k is not None)

    if verbose:
        print(
            f"eval_exp_tens: {M} rows, {n_valid} valid, "
            f"{n_unique} unique chords after canonical-form dedup."
        )

    if n_valid == 0:
        # All rows invalid. Need to infer nQ from x to give the right output shape.
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.ndim == 1:
            n_q = x_arr.size
        elif x_arr.ndim == 2:
            n_q = x_arr.shape[1]
        else:
            n_q = 0
        return np.full((M, n_q), np.nan)

    # Evaluate each unique density once at x.
    eval_cache: dict = {}
    for key, dens in dens_cache.items():
        eval_cache[key] = _eval_exp_tens_scalar(
            dens, x, normalize, method=method, verbose=False,
        )

    # Determine nQ from a representative evaluation.
    sample_vals = next(iter(eval_cache.values()))
    n_q = sample_vals.shape[0] if sample_vals.ndim >= 1 else 1

    out = np.full((M, n_q), np.nan)
    for i in range(M):
        key = row_to_key[i]
        if key is not None:
            out[i] = eval_cache[key]
    return out


def _eval_exp_tens_raw_ma_scalar(
    p_attr, w, sigma_vec, r_vec, groups,
    is_rel_vec, is_per_vec, period_vec,
    x, normalize: str,
    *, verbose: bool,
) -> np.ndarray:
    """Raw MA scalar dispatch: build MA density, evaluate."""
    dens = build_exp_tens(
        p_attr, w, sigma_vec, r_vec, groups,
        is_rel_vec, is_per_vec, period_vec, verbose=verbose,
    )
    return _eval_exp_tens_scalar(dens, x, normalize, verbose=verbose)


def _split_query_to_attr_list(dens: MaetDensity, x):
    """Normalise query input to a list of A per-attribute (dim_a, nQ) arrays.

    Mirrors the logic inside _eval_exp_tens_ma but returns the per-
    attribute list rather than running evaluation.
    """
    A = dens.n_attrs
    dim_per = dens.dim_per_attr
    dim_total = int(dens.dim)

    if isinstance(x, (list, tuple)):
        x_list = []
        n_q = None
        for a, xa in enumerate(x):
            xa = np.asarray(xa, dtype=np.float64)
            if xa.ndim == 1 and dim_per[a] == 1:
                xa = xa.reshape(1, -1)
            if n_q is None:
                n_q = xa.shape[1]
            x_list.append(xa)
        return x_list
    xs = np.asarray(x, dtype=np.float64)
    if xs.ndim == 1 and dim_total == 1:
        xs = xs.reshape(1, -1)
    x_list = []
    row = 0
    for a in range(A):
        da = int(dim_per[a])
        x_list.append(xs[row:row + da, :])
        row += da
    return x_list


# -------------------------------------------------------------------
#  _eval_exp_tens_sa  (single-attribute legacy path)
# -------------------------------------------------------------------


def _eval_exp_tens_sa(
    dens: ExpTensDensity,
    x: np.ndarray,
    normalize: str = "none",
    *,
    method: str = "auto",
    truncation_sigmas: float | None = None,
    kernel_precision: str | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """Single-attribute expectation tensor evaluation (dispatcher).

    Routes between the centres-array path and the Möbius
    point evaluator according to ``method`` and the cost model
    in :func:`_select_sa_eval_method`.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[0] != dens.dim:
        raise ValueError(
            f"x must have {dens.dim} rows (each column is a "
            f"{dens.dim}-D query point)."
        )
    n_q = x.shape[1]
    if n_q == 0:
        return np.zeros(0, dtype=np.float64)

    K = int(dens.p.shape[0])
    sigma_over_P = (
        float(dens.sigma) / float(dens.period) if dens.is_per else 0.0
    )

    # ---- Routing axis ----
    # Hard rules and explicit overrides decide inline; only the
    # discretionary 'auto' case with r >= 2 and K - r >= 2 invokes the
    # dispatcher. Skipping the dispatcher call shaves measurable
    # per-call overhead in MATLAB; Python is less sensitive but we
    # apply the principle uniformly for cross-language parity.
    routing_reason = "user override"
    if method == "centres" or method == "direct":
        chosen = "centres"
        probed = False
        est_sec = 0.0
    elif method == "mobius":
        chosen = "mobius"
        probed = False
        est_sec = 0.0
    elif method == "auto":
        r = int(dens.r)
        if r <= 1 or (K - r) < 2:
            # Hard rules force centres without a dispatcher call.
            chosen = "centres"
            probed = False
            est_sec = 0.0
            if r <= 1:
                routing_reason = f"r = {r}"
            else:
                routing_reason = f"K - r = {K - r} < 2"
        else:
            chosen, probed, est_sec, routing_reason = _select_and_estimate_sa(
                dens, x, n_q,
                method=method,
                truncation_sigmas=truncation_sigmas,
                kernel_precision=kernel_precision,
                verbose=verbose,
            )
    else:
        raise ValueError(
            f"method must be 'auto', 'centres', 'direct', or 'mobius'; "
            f"got '{method}'."
        )

    # Dispatch-decision message: bypasses per-call verbose, gated by
    # the toolbox-wide show_hints flag and throttled once per
    # (function, chosen, routing_reason) per Python process. The
    # throttle is cleared by mpt.reset_defaults(). To fully silence:
    # mpt.set_default(show_hints=False).
    from ._defaults import _maybe_show_dispatch_msg
    _maybe_show_dispatch_msg(
        "eval_exp_tens", chosen, routing_reason, est_sec, probed,
    )

    # ---- Execution axis: detect default-kwargs mode ----
    # Important: ``None`` means "use the global default", not "no
    # feature". So we must consult the defaults before deciding
    # whether the fast path applies — a globally-set finite truncation
    # or 'single' precision must still route through the helper.
    from ._defaults import get_default
    trunc_resolved = (
        truncation_sigmas if truncation_sigmas is not None
        else get_default("truncation_sigmas")
    )
    prec_resolved = (
        kernel_precision if kernel_precision is not None
        else get_default("kernel_precision")
    )
    use_default_kwargs = (
        (trunc_resolved is None or not np.isfinite(trunc_resolved))
        and (prec_resolved == "double")
    )

    # Fire the kernel-evaluation hint once per session when the centres
    # path is about to run with default kwargs. Catches the bypass
    # case too (which skips gaussian_kernel_sum and would otherwise
    # miss the hint).
    if chosen == "centres" and use_default_kwargs:
        from ._defaults import _maybe_show_kernel_eval_hint
        _maybe_show_kernel_eval_hint(
            effective_truncation_sigmas=float("inf"),
            effective_kernel_precision="double",
        )

    if chosen == "mobius":
        vals = _eval_exp_tens_sa_orbit(
            dens, x, n_q,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
            verbose=False,
        )
        # Post-hoc finiteness check. Mirrors the cosine-path safety
        # net: if the Möbius alternating partition sum produces non-finite output
        # (extreme σ → 0 regime), fall back to centres rather than
        # propagating NaN/Inf into the user's result.
        if not np.all(np.isfinite(vals)):
            if verbose:
                warnings.warn(
                    "eval_exp_tens Möbius method produced non-finite "
                    "values; falling back to centres path."
                )
            if use_default_kwargs:
                vals = _eval_exp_tens_sa_centres_fast(dens, x, n_q)
            else:
                vals = _eval_exp_tens_sa_centres(
                    dens, x, n_q,
                    truncation_sigmas=truncation_sigmas,
                    kernel_precision=kernel_precision,
                    verbose=False,
                )
    else:  # 'centres'
        if use_default_kwargs:
            # inline direct broadcast. FP-identical to
            # the helper at default settings, but skips the helper's
            # parameter validation and kwarg construction.
            vals = _eval_exp_tens_sa_centres_fast(dens, x, n_q)
        else:
            vals = _eval_exp_tens_sa_centres(
                dens, x, n_q,
                truncation_sigmas=truncation_sigmas,
                kernel_precision=kernel_precision,
                verbose=False,
            )

    return _eval_exp_tens_sa_normalize(vals, dens, normalize)


def _eval_exp_tens_sa_centres_fast(
    dens: ExpTensDensity, x: np.ndarray, n_q: int,
) -> np.ndarray:
    """inline direct broadcast for the centres path.

    Used by :func:`_eval_exp_tens_sa` when default kwargs apply
    (no truncation, no precision override). FP-identical to the
    helper-routed :func:`_eval_exp_tens_sa_centres` at default
    settings, but skips the helper's per-call validation overhead.
    Handles all (r, is_rel, is_per) combinations.

    Memory-aware ``n_q`` chunking matches the helper's
    ``_exact_kernel_sum``: peak per-chunk allocation is
    ``(dim+1) * n_j * n_qc * 8`` bytes for the difference tensor plus
    per-block intermediates. Without chunking, large workloads
    (e.g. K=72 r=3 nQ=29161 → ~155 GB peak) hit MATLAB's array-size
    cap and Python's memory limits.
    """
    centres = dens.centres
    w_j = dens.w_j
    n_j = int(dens.n_j)
    sigma = float(dens.sigma)
    r = int(dens.r)
    dim = int(dens.dim)
    is_rel = bool(dens.is_rel)
    is_per = bool(dens.is_per)
    period = float(dens.period)

    if n_j == 0 or n_q == 0:
        return np.zeros(n_q, dtype=np.float64)

    bytes_per_scalar = 8  # default-mode is always double
    bytes_needed = (dim + 1) * n_j * n_q * bytes_per_scalar
    mem_limit = 1 * 1024 ** 3  # 1 GB per-chunk cap, matches helper

    if bytes_needed <= mem_limit:
        return _eval_centres_fast_chunk(
            centres, w_j, x, n_q, dim, n_j, sigma, r, is_rel, is_per, period,
        )

    chunk_size = max(1, mem_limit // ((dim + 1) * n_j * bytes_per_scalar))
    vals = np.zeros(n_q, dtype=np.float64)
    for c0 in range(0, n_q, chunk_size):
        c1 = min(c0 + chunk_size, n_q)
        vals[c0:c1] = _eval_centres_fast_chunk(
            centres, w_j, x[:, c0:c1], c1 - c0, dim, n_j,
            sigma, r, is_rel, is_per, period,
        )
    return vals


def _eval_centres_fast_chunk(
    centres, w_j, x, n_qc, dim, n_j, sigma, r, is_rel, is_per, period,
):
    """Single-chunk direct broadcast for the SA centres fast path.

    Mirrors :func:`mpt._kernel._eval_chunk` exactly so the default-mode
    output is FP-bit-identical to v2.0/v2.1.
    """
    D = centres[:, :, None] - x[:, None, :]
    if is_per:
        D = np.mod(D + period / 2, period) - period / 2
    if is_rel:
        Q = np.sum(D ** 2, axis=0) - np.sum(D, axis=0) ** 2 / r
    else:
        Q = np.sum(D ** 2, axis=0)
    E = np.exp(-Q / (2 * sigma ** 2))
    return w_j @ E


def _eval_exp_tens_sa_centres(
    dens: ExpTensDensity,
    x: np.ndarray,
    n_q: int,
    *,
    truncation_sigmas: float | None = None,
    kernel_precision: str | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """Centres-array path for SA evaluation.

    Routes through :func:`mpt._kernel.gaussian_kernel_sum` so
    the ``truncation_sigmas`` and ``kernel_precision`` options apply
    uniformly across centres-path consumers. Default settings
    (``truncation_sigmas=inf``, ``kernel_precision='double'``) produce
    FP-bit-identical output to the v2.0/v2.1 implementation.
    """
    centres = dens.centres
    w_j = dens.w_j
    n_j = dens.n_j
    sigma = dens.sigma
    r = dens.r
    dim = dens.dim
    is_rel = dens.is_rel
    is_per = dens.is_per
    period = dens.period

    # Forward kwargs to the helper. ``None`` means "consult mpt defaults".
    kw = {}
    if is_rel:
        kw["is_rel"] = True
        kw["r"] = int(r)
    if is_per:
        kw["is_per"] = True
        kw["period"] = float(period)
    if truncation_sigmas is not None:
        kw["truncation_sigmas"] = float(truncation_sigmas)
    if kernel_precision is not None:
        kw["kernel_precision"] = kernel_precision

    return gaussian_kernel_sum(centres, w_j, x, float(sigma), **kw)


def _eval_exp_tens_sa_orbit(
    dens: ExpTensDensity,
    x: np.ndarray,
    n_q: int,
    *,
    truncation_sigmas: float | None = None,
    kernel_precision: str | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """Orbit-Möbius point evaluator for SA evaluation.

    Dispatches to :func:`mpt._mobius.eval_orbit_abs` (absolute modes)
    or :func:`mpt._mobius.eval_orbit_rel` (relative modes). Bypasses
    the ``(dim, n_j, n_q)`` intermediate tensor that would dominate
    memory in the centres path at high r.

    Forwards ``truncation_sigmas`` /
    ``kernel_precision`` to the Möbius evaluators. The non-periodic
    per-block kernel sum routes through ``gaussian_kernel_sum`` with
    ``sigma_eff = sigma/sqrt(m)``, gaining truncation natively.
    """
    from ._mobius import eval_orbit_abs, eval_orbit_rel

    p = dens.p
    w = dens.w
    sigma = float(dens.sigma)
    r = int(dens.r)
    is_rel = bool(dens.is_rel)
    is_per = bool(dens.is_per)
    period = float(dens.period)

    if verbose:
        pass

    if is_rel:
        return eval_orbit_rel(
            p, w, sigma, r, x,
            is_per=is_per, period=period,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
        )
    return eval_orbit_abs(
        p, w, sigma, r, x,
        is_per=is_per, period=period,
        truncation_sigmas=truncation_sigmas,
        kernel_precision=kernel_precision,
    )


def _eval_exp_tens_sa_normalize(
    vals: np.ndarray,
    dens: ExpTensDensity,
    normalize: str,
) -> np.ndarray:
    """Apply Gaussian / pdf normalisation to raw SA tensor values."""
    if normalize == "none":
        return vals
    sigma = dens.sigma
    r = dens.r
    dim = dens.dim
    is_rel = dens.is_rel
    w_j = dens.w_j

    det_m = (1.0 / r) if is_rel else 1.0
    gauss_const = (2 * np.pi * sigma**2) ** (-dim / 2) * np.sqrt(det_m)
    vals = vals * gauss_const

    if normalize == "pdf":
        sum_w = np.sum(w_j)
        if sum_w > 0:
            vals = vals / sum_w
        else:
            warnings.warn(
                "Sum of weight products is zero; cannot normalize to pdf."
            )

    return vals


# -------------------------------------------------------------------
#  _eval_exp_tens_ma  (multi-attribute path)
# -------------------------------------------------------------------


def _eval_exp_tens_ma(
    dens: MaetDensity,
    x,
    normalize: str = "none",
    *,
    truncation_sigmas: float | None = None,
    kernel_precision: str | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """Multi-attribute expectation tensor evaluation."""
    A           = dens.n_attrs
    n_j         = dens.n_j
    dim         = dens.dim
    dim_per     = dens.dim_per_attr
    group_of    = dens.group_of_attr
    r_vec       = dens.r
    sigma_g     = dens.sigma
    is_rel_g    = dens.is_rel
    is_per_g    = dens.is_per
    period_g    = dens.period
    centres     = dens.centres
    w_j         = dens.w_j

    # --- Normalise query input to a list of A per-attribute matrices ---

    if isinstance(x, (list, tuple)):
        if len(x) != A:
            raise ValueError(
                f"Query list must have length {A} (n attributes); got {len(x)}."
            )
        x_list = []
        n_q = None
        for a, xa in enumerate(x):
            xa = np.asarray(xa, dtype=np.float64)
            if xa.ndim == 1 and dim_per[a] == 1:
                xa = xa.reshape(1, -1)
            if xa.ndim != 2 or xa.shape[0] != int(dim_per[a]):
                raise ValueError(
                    f"Query for attribute {a} must have {int(dim_per[a])} "
                    f"rows; got shape {xa.shape}."
                )
            if n_q is None:
                n_q = xa.shape[1]
            elif xa.shape[1] != n_q:
                raise ValueError(
                    f"All per-attribute query matrices must share the same "
                    f"number of columns (nQ). Got {n_q} and {xa.shape[1]}."
                )
            x_list.append(xa)
    else:
        xs = np.asarray(x, dtype=np.float64)
        if xs.ndim == 1 and dim == 1:
            xs = xs.reshape(1, -1)
        if xs.ndim != 2 or xs.shape[0] != dim:
            raise ValueError(
                f"Single-matrix query must have {dim} rows (total dim); "
                f"got shape {xs.shape}. For list-form input, wrap the "
                f"per-attribute query matrices in a length-{A} list/tuple."
            )
        n_q = xs.shape[1]
        x_list = []
        row = 0
        for a in range(A):
            da = int(dim_per[a])
            x_list.append(xs[row:row + da, :])
            row += da

    if n_q == 0:
        return np.zeros(0, dtype=np.float64)

    n_pairs = int(n_j) * int(n_q)
    estimate_comp_time(n_pairs, dim, "eval_exp_tens (MAET)", verbose)

    # --- Core evaluation with memory-aware chunking ---
    # Peak per-chunk memory is dominated by the largest per-attribute
    # (dim_a, nJ, nQc) difference tensor plus the (nJ, nQc) accumulator.
    max_dim_a = int(max(dim_per)) if A > 0 else 1
    bytes_per_col = (max_dim_a + 1) * int(n_j) * 8
    mem_limit = 4_000_000_000  # 4 GB default

    bytes_needed = bytes_per_col * int(n_q)
    if bytes_needed <= mem_limit:
        vals = _ma_eval_full(
            centres, w_j, n_j, x_list, n_q,
            A, dim_per, group_of, r_vec, sigma_g,
            is_rel_g, is_per_g, period_g,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
        )
    else:
        chunk_size = max(1, int(mem_limit // max(bytes_per_col, 1)))
        vals = np.zeros(n_q, dtype=np.float64)
        for c_start in range(0, n_q, chunk_size):
            c_end = min(c_start + chunk_size, n_q)
            n_qc = c_end - c_start
            x_chunk = [xa[:, c_start:c_end] for xa in x_list]
            vals[c_start:c_end] = _ma_eval_full(
                centres, w_j, n_j, x_chunk, n_qc,
                A, dim_per, group_of, r_vec, sigma_g,
                is_rel_g, is_per_g, period_g,
                truncation_sigmas=truncation_sigmas,
                kernel_precision=kernel_precision,
            )

    # --- Normalisation ---
    if normalize != "none":
        gauss_const = 1.0
        for a in range(A):
            g = int(group_of[a])
            da = int(dim_per[a])
            if is_rel_g[g] and r_vec[a] >= 2:
                det_m_a = 1.0 / float(r_vec[a])
            else:
                det_m_a = 1.0
            gauss_const *= (2 * np.pi * sigma_g[g]**2) ** (-da / 2) \
                           * np.sqrt(det_m_a)
        vals = vals * gauss_const

        if normalize == "pdf":
            sum_w = float(np.sum(w_j))
            if sum_w > 0:
                vals = vals / sum_w
            else:
                warnings.warn(
                    "Sum of weight products is zero; cannot normalize to pdf."
                )

    return vals


def _ma_eval_full(
    centres, w_j, n_j, x_list, n_qc,
    A, dim_per, group_of, r_vec, sigma_g,
    is_rel_g, is_per_g, period_g,
    *,
    truncation_sigmas=None,
    kernel_precision=None,
):
    """Single-chunk MAET evaluation.

    Accumulates the summed-quadratic exponent across attributes, then
    exponentiates once and does the weighted sum against ``w_j``.

    Default-mode bypass: when ``truncation_sigmas`` is None/Inf and
    ``kernel_precision`` is None/'double', runs the inline accumulation
    inline with no cast machinery and no post-filter branching. This
    keeps default-mode calls at inline cost; the feature kwargs
    only impose their cost when explicitly requested.
    """
    # ---- Resolve precision / truncation from defaults ----
    # ``None`` means "use the global default", not "no feature". A
    # globally-set finite truncation or 'single' precision must still
    # take the feature path, not the default-mode bypass below.
    if kernel_precision is None:
        from ._defaults import get_default
        kernel_precision = get_default("kernel_precision")
    if truncation_sigmas is None:
        from ._defaults import get_default
        truncation_sigmas = get_default("truncation_sigmas")

    # ---- Default-mode bypass ----
    use_default = (
        kernel_precision == "double"
        and (truncation_sigmas is None
             or not np.isfinite(truncation_sigmas))
    )

    if use_default:
        # Direct double accumulation, no casts, no
        # post-filter branching.
        q_total = np.zeros((int(n_j), int(n_qc)), dtype=np.float64)
        for a in range(A):
            g = int(group_of[a])
            da = int(dim_per[a])
            if da == 0:
                continue
            c_a = centres[a]
            x_a = x_list[a]
            d_a = c_a[:, :, None] - x_a[:, None, :]
            if is_per_g[g]:
                pg = float(period_g[g])
                d_a = np.mod(d_a + pg / 2, pg) - pg / 2
            if is_rel_g[g]:
                q_a = (np.sum(d_a ** 2, axis=0)
                       - np.sum(d_a, axis=0) ** 2 / float(r_vec[a]))
            else:
                q_a = np.sum(d_a ** 2, axis=0)
            q_total = q_total + q_a / (2 * sigma_g[g] ** 2)
        e = np.exp(-q_total)
        return w_j @ e

    # ---- Feature-kwargs path: precision casting and / or
    # post-filter truncation. ----
    dtype = np.float32 if kernel_precision == "single" else np.float64

    q_total = np.zeros((int(n_j), int(n_qc)), dtype=dtype)

    for a in range(A):
        g = int(group_of[a])
        da = int(dim_per[a])
        if da == 0:
            continue

        c_a = centres[a].astype(dtype, copy=False)
        x_a = x_list[a].astype(dtype, copy=False)
        d_a = c_a[:, :, None] - x_a[:, None, :]

        if is_per_g[g]:
            pg = dtype(period_g[g])
            d_a = np.mod(d_a + pg / 2, pg) - pg / 2

        if is_rel_g[g]:
            q_a = (np.sum(d_a ** 2, axis=0)
                   - np.sum(d_a, axis=0) ** 2 / dtype(r_vec[a]))
        else:
            q_a = np.sum(d_a ** 2, axis=0)

        q_total = q_total + q_a / (2 * dtype(sigma_g[g]) ** 2)

    use_truncation = (
        truncation_sigmas is not None
        and np.isfinite(truncation_sigmas)
        and truncation_sigmas > 0
    )
    if use_truncation:
        # q_total represents Q/(2sigma^2). The kernel is exp(-q_total).
        # exp(-q_total) is negligible when q_total > k^2/2.
        q_threshold = float(truncation_sigmas) ** 2 / 2.0
        mask = q_total <= q_threshold
        e = np.where(mask, np.exp(-q_total), dtype(0.0))
    else:
        e = np.exp(-q_total)

    result = w_j.astype(dtype, copy=False) @ e
    return result.astype(np.float64, copy=False)


def _eval_core(
    centres, w_j, n_j, x, n_q, dim, sigma, r, is_rel, is_per, period
):
    """Evaluate with automatic memory-aware chunking (SA path)."""
    bytes_needed = (dim + 1) * int(n_j) * int(n_q) * 8
    mem_limit = 4_000_000_000  # 4 GB default

    if bytes_needed <= mem_limit:
        return _eval_full(centres, w_j, n_j, x, n_q, dim, sigma, r, is_rel, is_per, period)

    chunk_size = max(1, int(mem_limit / ((dim + 1) * int(n_j) * 8)))
    vals = np.zeros(n_q)
    for c_start in range(0, n_q, chunk_size):
        c_end = min(c_start + chunk_size, n_q)
        idx = slice(c_start, c_end)
        n_qc = c_end - c_start
        vals[idx] = _eval_full(
            centres, w_j, n_j, x[:, idx], n_qc, dim, sigma, r, is_rel, is_per, period
        )
    return vals


def _eval_full(centres, w_j, n_j, x_q, n_qc, dim, sigma, r, is_rel, is_per, period):
    """Fully vectorized SA density evaluation."""
    # D shape: (dim, nJ, nQc)
    D = centres[:, :, None] - x_q[:, None, :]

    if is_per:
        D = np.mod(D + period / 2, period) - period / 2

    if is_rel:
        Q = np.sum(D**2, axis=0) - np.sum(D, axis=0) ** 2 / r
    else:
        Q = np.sum(D**2, axis=0)

    # E shape: (nJ, nQc)
    E = np.exp(-Q / (2 * sigma**2))

    # Weighted sum: (nJ,) @ (nJ, nQc) → (nQc,)
    return w_j @ E


# -------------------------------------------------------------------
#  eval_exp_tens from raw args (convenience)
# -------------------------------------------------------------------


def eval_exp_tens_raw(
    p: np.ndarray,
    w: np.ndarray | None,
    sigma: float,
    r: int,
    is_rel: bool,
    is_per: bool,
    period: float,
    x: np.ndarray,
    normalize: str = "none",
    *,
    verbose: bool = True,
) -> np.ndarray:
    """Deprecated. Use :func:`eval_exp_tens` directly with raw input.

    .. deprecated:: 2.1
       The raw-input dispatch has been folded into the unified
       :func:`eval_exp_tens` entry point. Pass raw arrays directly:

       .. code-block:: python

          # Old:
          vals = eval_exp_tens_raw(p, w, sigma, r, is_rel, is_per, period, x)
          # New (identical signature):
          vals = eval_exp_tens(p, w, sigma, r, is_rel, is_per, period, x)

    This shim will be removed in a future release.
    """
    warnings.warn(
        "eval_exp_tens_raw is deprecated. The same call signature is "
        "now supported directly by eval_exp_tens (pass raw arrays as the "
        "first arguments instead of pre-built density objects). This shim "
        "will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    return eval_exp_tens(
        p, w, sigma, r, is_rel, is_per, period, x, normalize,
        verbose=verbose,
    )


# -------------------------------------------------------------------
#  cos_sim_exp_tens
# -------------------------------------------------------------------


@_with_dispatch_scope
def cos_sim_exp_tens(*args,
                     mode: str = "auto",
                     dedup: bool = True,
                     spectrum=None,
                     precision: int | None = None,
                     method: str = "auto",
                     cancellation_threshold: float = 1e-12,
                     truncation_sigmas: float | None = None,
                     kernel_precision: str | None = None,
                     verbose: bool = True) -> float | np.ndarray:
    """Cosine similarity of two expectation tensor densities.

    Unified entry point. Accepts four input forms, dispatched on the
    type of the first argument:

    **Pre-built density input** (plus polymorphic lists):

    - ``cos_sim_exp_tens(dens_x, dens_y)`` — scalar.
    - ``cos_sim_exp_tens(dens_x, [d1, d2, …])`` — broadcast, returns
      ``(N,)``.
    - ``cos_sim_exp_tens([a1, a2, …], [b1, b2, …])`` — list-vs-list
      with ``mode='pairwise'`` (default ``'auto'``, resolves to
      pairwise for equal lengths) returning ``(M,)``, or
      ``mode='cartesian'`` returning ``(M, N)``.

    **Raw single-attribute scalar input**:

    - ``cos_sim_exp_tens(p1, w1, p2, w2, sigma, r, is_rel, is_per, period)``
      where ``p1`` and ``p2`` are 1-D arrays of pitches, ``w1``,
      ``w2`` are matching 1-D weight arrays (or ``None`` for uniform).
      Returns scalar.

    **Raw single-attribute batched input** (replaces ``batch_cos_sim_exp_tens``):

    - ``cos_sim_exp_tens(P1, W1, P2, W2, sigma, r, is_rel, is_per, period)``
      where at least one of ``P1``, ``P2`` is a 2-D ``(M, K)`` matrix
      (rows are chords; NaN-padded for variable cardinality), ``W1``,
      ``W2`` likewise (or ``None`` for uniform). Returns ``(M,)``. If
      only one operand is a matrix and the other is a 1-D vector of
      length ``K``, the vector is broadcast across the matrix's ``M``
      rows.

    **Raw multi-attribute scalar input**:

    - ``cos_sim_exp_tens(p_attr1, w1, p_attr2, w2, sigma_vec, r_vec, groups,
      is_rel_vec, is_per_vec, period_vec)`` where ``p_attr*`` are
      lists of per-attribute matrices. Returns scalar.

    Parameters
    ----------
    *args
        Positional arguments. Length depends on the input form:
        2 for density modes; 9 for raw SA modes; 10 for raw MA mode.
    mode : {'auto', 'pairwise', 'cartesian'}, default 'auto'
        For density list-vs-list. Ignored in scalar and broadcast cases.
    dedup : bool, default True
        Apply canonical-form deduplication. Currently supported for
        single-attribute pairs only; pairs involving ``MaetDensity``
        bypass dedup transparently. (``WindowedMaetDensity`` operands
        are rejected at the top of the function — use
        :func:`windowed_similarity` instead.)
    spectrum : list/tuple, optional
        Per-row spectral augmentation parameters passed to
        :func:`mpt.spectra.add_spectra`. Only valid in raw SA modes
        (scalar or batched).
    precision : int, optional
        Round canonical pitch and weight values to this many decimal
        places, to absorb FP noise when deduplicating. Only valid in
        raw SA batched mode.
    method : {'auto', 'bulger', 'mobius', 'direct'}, default 'auto'
        Inner-product method; threaded through to the per-pair SA/MA
        core. ``'auto'`` lets the dispatcher pick between Bulger's
        method (the partition-pair decomposition; small r and small K)
        and the Möbius method (large r or large K).
        ``'bulger'`` forces Bulger's method; ``'mobius'`` forces the
        Möbius method; ``'direct'`` forces direct ordered-tuple
        enumeration.
    cancellation_threshold : float, default 1e-12
        When the Möbius method is selected and ``|<A,B>|`` falls below
        this fraction of ``sqrt(<A,A><B,B>)``, fall back to Bulger's
        method to avoid catastrophic Möbius alternating-sum
        cancellation.
    verbose : bool, default True
        Print progress.

    Returns
    -------
    float or np.ndarray
        Scalar in scalar-vs-scalar density mode, raw SA scalar mode, and
        raw MA scalar mode. ``ndarray`` in all batched/list modes.

    Notes
    -----
    The Möbius method  is exact to floating-point
    precision when every per-attribute ``K_a`` satisfies
    ``K_a >= r_a + 2`` and σ is not catastrophically small relative to
    the period P. The dispatcher enforces these conditions structurally
    — it refuses the Möbius method and routes to Bulger's method when
    ``K_a < r_a + 2``, when ``σ/P > 0.03`` in periodic-relative mode,
    or when the σ → 0 fallback triggers. Pass ``method='bulger'`` to
    bypass the Möbius method entirely.

    See Also
    --------
    build_exp_tens : explicit density construction.
    eval_exp_tens : evaluate a density at query points.
    cos_sim_exp_tens_raw : deprecated; superseded by raw input mode here.
    batch_cos_sim_exp_tens : deprecated; superseded by raw SA batched input here.

    References
    ----------
    Originally by David Bulger, Macquarie University (2016).
    Adapted for the Music Perception Toolbox v2 by Andrew J. Milne.

    """
    if len(args) < 2:
        raise TypeError(
            "cos_sim_exp_tens requires at least 2 positional arguments."
        )

    a = args[0]

    # ------------------------------------------------------------------
    # Detect density-input intent based on the first argument.
    # ------------------------------------------------------------------
    is_density_scalar = isinstance(
        a, (ExpTensDensity, MaetDensity, WindowedMaetDensity)
    )
    intends_density_list = False
    if isinstance(a, (list, tuple)):
        if len(a) == 0:
            intends_density_list = True
        elif isinstance(
            a[0], (ExpTensDensity, MaetDensity, WindowedMaetDensity)
        ):
            intends_density_list = True
    elif isinstance(a, np.ndarray) and a.dtype == object:
        intends_density_list = True

    if is_density_scalar or intends_density_list:
        if len(args) != 2:
            raise TypeError(
                f"Density input mode expects 2 positional arguments "
                f"(dens_x, dens_y); got {len(args)}."
            )
        if spectrum is not None:
            raise TypeError(
                "'spectrum' kwarg is only valid in raw SA input mode."
            )
        if precision is not None:
            raise TypeError(
                "'precision' kwarg is only valid in raw SA batched input mode."
            )
        return _cos_sim_density_path(
            args[0], args[1],
            mode=mode, dedup=dedup,
            method=method,
            cancellation_threshold=cancellation_threshold,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Raw multi-attribute dispatch (list of per-attribute arrays)
    # ------------------------------------------------------------------
    if _looks_like_multi_attr(a):
        if len(args) != 10:
            raise TypeError(
                f"Raw multi-attribute input expects 10 positional arguments "
                f"(p_attr1, w1, p_attr2, w2, sigma_vec, r_vec, groups, "
                f"is_rel_vec, is_per_vec, period_vec); got {len(args)}."
            )
        if spectrum is not None:
            raise TypeError(
                "'spectrum' kwarg is only supported in raw single-attribute "
                "input mode."
            )
        if precision is not None:
            raise TypeError(
                "'precision' kwarg is only valid in raw SA batched input mode."
            )
        if mode != "auto":
            raise TypeError(
                "'mode' kwarg only applies to density list inputs."
            )
        return _cos_sim_raw_ma_scalar(
            *args,
            method=method,
            cancellation_threshold=cancellation_threshold,
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Raw single-attribute dispatch.
    # ------------------------------------------------------------------
    if len(args) != 9:
        raise TypeError(
            f"Raw single-attribute input expects 9 positional arguments "
            f"(p1, w1, p2, w2, sigma, r, is_rel, is_per, period); "
            f"got {len(args)}."
        )

    try:
        a_arr = np.asarray(a, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"First argument must be a density object, list of densities, "
            f"numeric array (1-D for a single chord, 2-D for a batch), or "
            f"list of per-attribute matrices for MA raw input; got "
            f"{type(a).__name__}."
        ) from exc

    try:
        b_arr = np.asarray(args[2], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Third positional argument (P2) must be a numeric array; got "
            f"{type(args[2]).__name__}."
        ) from exc

    if a_arr.ndim > 2 or b_arr.ndim > 2:
        raise TypeError(
            f"Raw SA inputs must be 1-D (single chord) or 2-D (batched); "
            f"got P1.ndim = {a_arr.ndim}, P2.ndim = {b_arr.ndim}."
        )

    # Batched dispatch fires whenever either operand is 2-D.
    if a_arr.ndim == 2 or b_arr.ndim == 2:
        sigma, r_, is_rel, is_per, period = args[4:9]
        W1_arg, W2_arg = args[1], args[3]

        # Reshape any 1-D operand to (1, K) so both are 2-D from here on.
        P1 = a_arr if a_arr.ndim == 2 else a_arr.reshape(1, -1)
        P2 = b_arr if b_arr.ndim == 2 else b_arr.reshape(1, -1)

        def _to_row_w(w):
            """Match a weights argument's shape to its (now 2-D) p."""
            if w is None:
                return None
            w_arr = np.asarray(w, dtype=np.float64)
            if w_arr.ndim == 1:
                return w_arr.reshape(1, -1)
            return w_arr

        W1 = _to_row_w(W1_arg)
        W2 = _to_row_w(W2_arg)

        M1, M2 = P1.shape[0], P2.shape[0]
        if M1 == 1 and M2 > 1:
            P1 = np.broadcast_to(P1, (M2, P1.shape[1])).copy()
            if W1 is not None:
                W1 = np.broadcast_to(W1, (M2, W1.shape[1])).copy()
        elif M2 == 1 and M1 > 1:
            P2 = np.broadcast_to(P2, (M1, P2.shape[1])).copy()
            if W2 is not None:
                W2 = np.broadcast_to(W2, (M1, W2.shape[1])).copy()
        elif M1 != M2:
            raise ValueError(
                f"Batched-raw P1 and P2 must either have matching row counts, "
                f"or one of them must be a single-row reference (1-D vector "
                f"or shape ``(1, K)``) to broadcast against the other. Got "
                f"{M1} and {M2} rows."
            )

        return _cos_sim_raw_sa_batch(
            P1, P2, sigma, r_, is_rel, is_per, period,
            weights_a=W1, weights_b=W2,
            spectrum=spectrum, precision=precision,
            dedup=dedup,
            method=method,
            cancellation_threshold=cancellation_threshold,
            verbose=verbose,
        )

    # Both operands are 1-D → existing scalar SA path.
    if precision is not None:
        raise TypeError(
            "'precision' kwarg is only valid for raw SA batched input "
            "(at least one of P1, P2 must be 2-D)."
        )
    if mode != "auto":
        raise TypeError(
            "'mode' kwarg only applies to density list inputs."
        )
    return _cos_sim_raw_sa_scalar(
        *args, spectrum=spectrum,
        method=method,
        cancellation_threshold=cancellation_threshold,
        verbose=verbose,
    )


# -------------------------------------------------------------------
#  Polymorphic-dispatch helpers for cos_sim_exp_tens
# -------------------------------------------------------------------


def _normalize_density_input(arg, *, name: str):
    """Detect whether ``arg`` is a single density or a list of densities.

    Returns
    -------
    is_scalar : bool
        True if ``arg`` is a single density object (not a list/tuple/array).
    densities : tuple
        Tuple of density objects.
    """
    if isinstance(arg, np.ndarray) and arg.dtype == object:
        arg = list(arg)

    if isinstance(arg, (ExpTensDensity, MaetDensity, WindowedMaetDensity)):
        return True, (arg,)

    if isinstance(arg, (list, tuple)):
        if len(arg) == 0:
            return False, ()
        for i, elem in enumerate(arg):
            if not isinstance(
                elem, (ExpTensDensity, MaetDensity, WindowedMaetDensity)
            ):
                raise TypeError(
                    f"{name}[{i}] must be an ExpTensDensity, MaetDensity, "
                    f"or WindowedMaetDensity; got {type(elem).__name__}."
                )
        return False, tuple(arg)

    raise TypeError(
        f"{name} must be a density object or a list/tuple of densities; "
        f"got {type(arg).__name__}."
    )


def _resolve_list_list_mode(mode: str, m: int, n: int) -> str:
    """Resolve ``mode`` for the list-vs-list case. Returns 'pairwise' or 'cartesian'."""
    if mode == "pairwise":
        if m != n:
            raise ValueError(
                f"mode='pairwise' requires equal-length lists; got M={m}, N={n}. "
                f"Pass mode='cartesian' for the M×N case."
            )
        return "pairwise"
    if mode == "cartesian":
        return "cartesian"
    if mode == "auto":
        if m == n:
            return "pairwise"
        raise ValueError(
            f"mode='auto' requires equal-length lists for pairwise resolution; "
            f"got M={m}, N={n}. Pass mode='cartesian' for the M×N case "
            f"or mode='pairwise' to assert equal lengths."
        )
    raise ValueError(
        f"mode must be one of 'auto', 'pairwise', 'cartesian'; got {mode!r}."
    )


def _all_sa_pairs(pairs):
    """Return True iff every (a, b) pair in ``pairs`` is two ExpTensDensity objects."""
    for a, b in pairs:
        if not (isinstance(a, ExpTensDensity) and isinstance(b, ExpTensDensity)):
            return False
    return True


def _compute_pair_results_with_dedup_sa(
    pairs, *, method: str, cancellation_threshold: float,
    truncation_sigmas=None, kernel_precision=None, verbose: bool,
):
    """Compute cos_sim for SA-density pairs with canonical-form dedup."""
    pair_key_to_idx: dict = {}
    pair_canon_idx: list[int] = []
    unique_pair_list: list = []

    for a, b in pairs:
        key_a, key_b, _, _, _, _ = _pair_canonical_key(
            a.p, a.w, b.p, b.w,
            sigma=a.sigma, r=a.r, is_rel=a.is_rel,
            is_per=a.is_per, period=a.period,
        )
        pk = (
            key_a,
            key_b,
            (b.sigma, b.r, b.is_rel, b.is_per, b.period),
        )
        if pk not in pair_key_to_idx:
            pair_key_to_idx[pk] = len(unique_pair_list)
            unique_pair_list.append((a, b))
        pair_canon_idx.append(pair_key_to_idx[pk])

    n_unique = len(unique_pair_list)
    if verbose:
        print(
            f"cos_sim_exp_tens: {len(pairs)} pairs, {n_unique} unique "
            f"after canonical-form dedup."
        )

    # Empirical-calibration time estimate. Warm-up plus a timed sample
    # of K ≤ 5 representative unique pairs, extrapolated over n_unique.
    # Gated on verbose; 10 s silence threshold via
    # maybe_print_batched_estimate. Parallels MATLAB localCosSimBatchedRaw
    # Phase 3.5 (cosSimExpTens.m).
    # Adaptive progress-print state. Defaults: silent. Overridden in
    # the calibration block when est_total is known.
    prog_stride = 1
    show_progress = False
    if verbose and n_unique >= 2:
        import time as _time
        from ._utils import progress_stride
        n_cal = min(5, n_unique)
        sample_idx = sorted(set(
            int(round(v)) for v in np.linspace(0, n_unique - 1, n_cal)
        ))
        # Warm-up call (absorbs one-time setup).
        a_w, b_w = unique_pair_list[sample_idx[0]]
        _cos_sim_exp_tens_sa(
            a_w, b_w,
            method=method,
            cancellation_threshold=cancellation_threshold,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
            verbose=False,
        )
        t_cal_start = _time.perf_counter()
        for ci in sample_idx:
            a_s, b_s = unique_pair_list[ci]
            _cos_sim_exp_tens_sa(
                a_s, b_s,
                method=method,
                cancellation_threshold=cancellation_threshold,
                truncation_sigmas=truncation_sigmas,
                kernel_precision=kernel_precision,
                verbose=False,
            )
        t_cal_total = _time.perf_counter() - t_cal_start
        t_per_pair = t_cal_total / len(sample_idx)
        est_total = t_cal_total + t_per_pair * n_unique
        maybe_print_batched_estimate(
            "cos_sim_exp_tens", n_unique, est_total,
        )
        prog_stride = progress_stride(t_per_pair)
        show_progress = est_total >= 5

    # Main loop over unique pairs (converted from list comprehension
    # so we can emit progress, matching MATLAB Phase 4).
    unique_results = []
    for up, (a, b) in enumerate(unique_pair_list):
        unique_results.append(_cos_sim_exp_tens_sa(
            a, b,
            method=method,
            cancellation_threshold=cancellation_threshold,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
            verbose=False,
        ))
        if verbose and show_progress \
                and ((up + 1) % prog_stride == 0 or up == n_unique - 1):
            print(f"  {up + 1} / {n_unique} unique pairs computed.")

    return [unique_results[idx] for idx in pair_canon_idx]


def _compute_pair_results_no_dedup(
    pairs, *, method: str, cancellation_threshold: float,
    truncation_sigmas=None, kernel_precision=None, verbose: bool,
):
    """Compute cos_sim for a list of pairs without dedup.

    No empirical calibration is run on this path because the input
    may include heterogeneous MA densities whose per-pair cost varies
    too widely for a stable extrapolation. Consequently no progress
    prints are emitted; users wanting feedback on long runs should
    enable canonical-form dedup (the default).
    """
    results = []
    for a, b in pairs:
        results.append(_cos_sim_pair_core(
            a, b,
            method=method,
            cancellation_threshold=cancellation_threshold,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
            verbose=False,
        ))
    return results


def _cos_sim_pair_core(
    dens_x, dens_y, *,
    method: str = "auto",
    cancellation_threshold: float = 1e-12,
    truncation_sigmas: float | None = None,
    kernel_precision: str | None = None,
    verbose: bool,
):
    """Internal: dispatch a single pair to the correct core IP routine.

    Routes to :func:`_cos_sim_exp_tens_sa` or :func:`_cos_sim_exp_tens_ma`,
    threading ``method``, ``cancellation_threshold``, ``truncation_sigmas``
    and ``kernel_precision`` through to the SA path (the MA path awaits
    its own helper-routing stage). ``WindowedMaetDensity`` operands are
    rejected here; user code reaches the windowed inner product via
    :func:`windowed_similarity`.
    """
    if isinstance(dens_x, WindowedMaetDensity) or \
            isinstance(dens_y, WindowedMaetDensity):
        raise TypeError(
            "cos_sim_exp_tens does not accept WindowedMaetDensity "
            "operands. Use windowed_similarity(dens_query, dens_context, "
            "window_spec, offsets) — pass a single-column offsets array "
            "for the scalar single-offset case, or a (dim, M) array for "
            "the M-offset sweep."
        )
    if isinstance(dens_x, MaetDensity):
        if not isinstance(dens_y, MaetDensity):
            raise TypeError(
                "dens_x is a MaetDensity but dens_y is not; both must be "
                "the same type."
            )
        return _cos_sim_exp_tens_ma(
            dens_x, dens_y,
            method=method,
            cancellation_threshold=cancellation_threshold,
            verbose=verbose,
        )
    if isinstance(dens_x, ExpTensDensity):
        if not isinstance(dens_y, ExpTensDensity):
            raise TypeError(
                "dens_x is an ExpTensDensity but dens_y is not; both must "
                "be the same type."
            )
        return _cos_sim_exp_tens_sa(
            dens_x, dens_y,
            method=method,
            cancellation_threshold=cancellation_threshold,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
            verbose=verbose,
        )
    raise TypeError(
        f"Both arguments must be ExpTensDensity, MaetDensity, or "
        f"WindowedMaetDensity; got {type(dens_x).__name__} and "
        f"{type(dens_y).__name__}."
    )


def _cos_sim_density_path(
    dens_x, dens_y, *,
    mode: str = "auto",
    dedup: bool = True,
    method: str = "auto",
    cancellation_threshold: float = 1e-12,
    truncation_sigmas: float | None = None,
    kernel_precision: str | None = None,
    verbose: bool = True,
):
    """Density-input dispatch for :func:`cos_sim_exp_tens`."""
    is_x_scalar, list_x = _normalize_density_input(dens_x, name="dens_x")
    is_y_scalar, list_y = _normalize_density_input(dens_y, name="dens_y")

    # Scalar-vs-scalar.
    if is_x_scalar and is_y_scalar:
        return _cos_sim_pair_core(
            list_x[0], list_y[0],
            method=method,
            cancellation_threshold=cancellation_threshold,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
            verbose=verbose,
        )

    m = len(list_x)
    n = len(list_y)

    if is_x_scalar:
        if n == 0:
            return np.empty((0,), dtype=np.float64)
        a = list_x[0]
        pairs = [(a, b) for b in list_y]
        out_shape = (n,)
    elif is_y_scalar:
        if m == 0:
            return np.empty((0,), dtype=np.float64)
        b = list_y[0]
        pairs = [(a, b) for a in list_x]
        out_shape = (m,)
    else:
        if m == 0 or n == 0:
            try:
                resolved = _resolve_list_list_mode(mode, m, n)
            except ValueError:
                resolved = "cartesian"
            if resolved == "pairwise":
                return np.empty((0,), dtype=np.float64)
            return np.empty((m, n), dtype=np.float64)

        resolved = _resolve_list_list_mode(mode, m, n)
        if resolved == "pairwise":
            pairs = list(zip(list_x, list_y))
            out_shape = (m,)
        else:
            pairs = [(a, b) for a in list_x for b in list_y]
            out_shape = (m, n)

    if dedup and _all_sa_pairs(pairs):
        results = _compute_pair_results_with_dedup_sa(
            pairs,
            method=method,
            cancellation_threshold=cancellation_threshold,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
            verbose=verbose,
        )
    else:
        if dedup and verbose:
            print(
                "cos_sim_exp_tens: dedup=True requested but input includes "
                "non-SA densities; computing without dedup."
            )
        results = _compute_pair_results_no_dedup(
            pairs,
            method=method,
            cancellation_threshold=cancellation_threshold,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
            verbose=verbose,
        )

    return np.array(results, dtype=np.float64).reshape(out_shape)


def _cos_sim_raw_sa_scalar(
    p1, w1, p2, w2,
    sigma, r, is_rel, is_per, period,
    *,
    spectrum=None,
    method: str = "auto",
    cancellation_threshold: float = 1e-12,
    verbose: bool = True,
) -> float:
    """Raw single-attribute scalar dispatch for :func:`cos_sim_exp_tens`."""
    if spectrum is not None:
        p1_aug, w1_aug = add_spectra(
            np.asarray(p1, dtype=np.float64),
            np.ones_like(np.asarray(p1, dtype=np.float64)) if w1 is None
            else np.asarray(w1, dtype=np.float64),
            *spectrum,
        )
        p2_aug, w2_aug = add_spectra(
            np.asarray(p2, dtype=np.float64),
            np.ones_like(np.asarray(p2, dtype=np.float64)) if w2 is None
            else np.asarray(w2, dtype=np.float64),
            *spectrum,
        )
    else:
        p1_aug, w1_aug = p1, w1
        p2_aug, w2_aug = p2, w2

    dx = build_exp_tens(
        p1_aug, w1_aug, sigma, r, is_rel, is_per, period, verbose=verbose,
    )
    dy = build_exp_tens(
        p2_aug, w2_aug, sigma, r, is_rel, is_per, period, verbose=verbose,
    )
    return _cos_sim_pair_core(
        dx, dy,
        method=method,
        cancellation_threshold=cancellation_threshold,
        verbose=verbose,
    )


def _cos_sim_raw_ma_scalar(
    p_attr1, w1, p_attr2, w2,
    sigma_vec, r_vec, groups, is_rel_vec, is_per_vec, period_vec,
    *,
    method: str = "auto",
    cancellation_threshold: float = 1e-12,
    verbose: bool = True,
) -> float:
    """Raw multi-attribute scalar dispatch for :func:`cos_sim_exp_tens`."""
    dx = build_exp_tens(
        p_attr1, w1, sigma_vec, r_vec, groups,
        is_rel_vec, is_per_vec, period_vec, verbose=verbose,
    )
    dy = build_exp_tens(
        p_attr2, w2, sigma_vec, r_vec, groups,
        is_rel_vec, is_per_vec, period_vec, verbose=verbose,
    )
    return _cos_sim_pair_core(
        dx, dy,
        method=method,
        cancellation_threshold=cancellation_threshold,
        verbose=verbose,
    )


# -------------------------------------------------------------------
#  _cos_sim_exp_tens_sa  (single-attribute legacy path)
# -------------------------------------------------------------------


_ORBIT_CANCELLATION_RATIO_MIN = 1e-10
"""Minimum acceptable cancellation ratio in the Möbius method's alternating partition sum.

When ``|sum| / max(|term|)`` drops below this threshold the result has
lost roughly 10 of its 16 significant decimal digits, leaving ~6
surviving — borderline acceptable for cosine accuracy at downstream
1e-6 user tolerance, but past this point the dispatcher falls back to
Bulger's method. See V22_DEV_LOG.md for the empirical regime where this
fires (sharp Gaussians + low K-r margin in absolute modes)."""


def _orbit_ips_look_corrupted(ip_xy, ip_xx, ip_yy):
    """Cheap post-hoc sanity check on Möbius-method-computed inner products.

    The Möbius method's alternating partition sum can break down
    catastrophically in two regimes documented during the May 2026
    audit:

    * σ → 0 with low K and r ≥ 3 (music-theoretical exact-match regime):
      auto-IP terms cancel to a value with magnitude near
      machine epsilon, then floating-point overflow can produce huge
      garbage values when the cosine ratio is taken.
    * Issue 4 sharp-Gaussian regime (σ small relative to data range):
      auto-IPs lose 4–8 decimal digits of precision while looking
      finite; this check does NOT catch that — only the catastrophic
      overflow / sign-corruption regime.

    Triggers on any of:
    * non-finite IP (NaN or Inf in any of the three),
    * negative auto-IP (a Gram-matrix diagonal must be ≥ 0; sign flip
      is unambiguous corruption),
    * cosine magnitude > 1 + 1e-6 (impossible for a genuine cosine).

    Parameters
    ----------
    ip_xy, ip_xx, ip_yy : float
        Cross and auto inner products from the Möbius method.

    Returns
    -------
    bool
        True if the IPs are unsuitable for use and the caller should
        fall back to Bulger's method.
    """
    if not (np.isfinite(ip_xy) and np.isfinite(ip_xx) and np.isfinite(ip_yy)):
        return True
    if ip_xx < 0 or ip_yy < 0:
        return True
    denom = np.sqrt(ip_xx * ip_yy)
    if denom > 0 and abs(ip_xy) > 1.000001 * denom:
        return True
    return False


def _cos_sim_exp_tens_sa(
    dens_x: ExpTensDensity,
    dens_y: ExpTensDensity,
    *,
    method: str = "auto",
    cancellation_threshold: float = 1e-12,
    truncation_sigmas: float | None = None,
    kernel_precision: str | None = None,
    verbose: bool = True,
) -> float:
    """Single-attribute cosine similarity.

    A ``method`` keyword routes between Bulger's method
    — the v1 / v2.1 decomposition with periodic pairwise-wrap form
    (``_ip_core``) — and the Möbius method. With
    the default ``method='auto'`` and perceptually typical parameters,
    the Möbius method is selected and the result agrees with v2.1 to
    floating-point precision.
    """
    if dens_x.r != dens_y.r:
        raise ValueError("Both densities must have the same r.")
    if dens_x.is_rel != dens_y.is_rel:
        raise ValueError("Both densities must have the same is_rel.")
    if dens_x.is_per != dens_y.is_per:
        raise ValueError("Both densities must have the same is_per.")
    if dens_x.is_per and dens_x.period != dens_y.period:
        raise ValueError("Both densities must have the same period.")
    if dens_x.sigma != dens_y.sigma:
        raise ValueError("Both densities must have the same sigma.")

    if method not in ("auto", "bulger", "direct", "mobius"):
        raise ValueError(
            f"method must be one of 'auto', 'bulger', 'direct'; "
            f"got {method!r}."
        )

    r = dens_x.r
    is_rel = dens_x.is_rel
    is_per = dens_x.is_per
    period = dens_x.period
    sigma = dens_x.sigma

    # Early return for degenerate case.
    if r > min(len(dens_x.p), len(dens_y.p)):
        return float("nan")

    n_max = max(len(dens_x.p), len(dens_y.p))
    n_min = min(len(dens_x.p), len(dens_y.p))
    sigma_over_P = sigma / period if (is_per and period > 0) else 0.0
    chosen, probed, est_sec, routing_reason = _select_and_estimate_sa_ip(
        dens_x, dens_y,
        method=method,
        truncation_sigmas=truncation_sigmas,
        kernel_precision=kernel_precision,
        verbose=verbose,
    )

    # Dispatch-decision message: bypasses per-call verbose, gated by
    # the toolbox-wide show_hints flag and throttled once per
    # (function, chosen, routing_reason) per Python process. The
    # throttle is cleared by mpt.reset_defaults(). To fully silence:
    # mpt.set_default(show_hints=False).
    from ._defaults import _maybe_show_dispatch_msg
    _maybe_show_dispatch_msg(
        "cos_sim_exp_tens", chosen, routing_reason, est_sec, probed,
    )

    if chosen == "mobius":
        ip_xy, ip_xx, ip_yy, worst_ratio = _cos_sim_exp_tens_sa_orbit(
            dens_x, dens_y,
        )
        # Three layers of Möbius-method result validation, fall back on any:
        # 1. Cross-cancellation guard: <A,B> small relative to
        #    sqrt(<A,A><B,B>) — the Möbius estimate may be dominated by
        #    cancellation between partition terms.
        denom_geo = np.sqrt(max(ip_xx * ip_yy, 0.0))
        cross_cancellation = (
            denom_geo > 0
            and abs(ip_xy) < cancellation_threshold * denom_geo
        )
        # 2. Post-hoc sanity on the IPs themselves (catches the σ→0
        #    catastrophic-overflow regime: non-finite, sign-corrupt, or
        #    cosine outside [-1, 1]).
        ips_corrupted = _orbit_ips_look_corrupted(ip_xy, ip_xx, ip_yy)
        # 3. Runtime cancellation diagnostic: the alternating Möbius
        #    sum has lost too many significant digits, even if the
        #    final values look superficially fine. Catches the quieter
        #    sharp-Gaussian regime where IPs are finite-looking but
        #    ~1e-4 to 1e-2 wrong.
        cancellation_too_severe = (
            worst_ratio < _ORBIT_CANCELLATION_RATIO_MIN
        )
        if cross_cancellation or ips_corrupted or cancellation_too_severe:
            ip_xy, ip_xx, ip_yy = _cos_sim_exp_tens_sa_pairwise(
                dens_x, dens_y, verbose=verbose,
                truncation_sigmas=truncation_sigmas,
                kernel_precision=kernel_precision,
            )
    else:  # 'bulger' or 'direct' — coincide in SA mode
        ip_xy, ip_xx, ip_yy = _cos_sim_exp_tens_sa_pairwise(
            dens_x, dens_y, verbose=verbose,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
        )

    denom = np.sqrt(ip_xx * ip_yy)
    if denom == 0:
        return float("nan")
    return float(ip_xy / denom)


# -------------------------------------------------------------------
#  _cos_sim_exp_tens_ma  (multi-attribute path)
# -------------------------------------------------------------------


def _cos_sim_exp_tens_ma(
    dens_x: MaetDensity,
    dens_y: MaetDensity,
    *,
    method: str = "auto",
    cancellation_threshold: float = 1e-12,
    verbose: bool = True,
) -> float:
    """Multi-attribute cosine similarity.

    A ``method`` keyword routes between Bulger's method
    — the v1 / v2.1 decomposition with periodic pairwise-wrap form
    (``_ip_core_ma``) — and the Möbius method.
    With the default ``method='auto'`` and perceptually typical
    parameters (no NaN-padded ``p_attr``, r_a ≤
    ``_ORBIT_R_MAX_SHIPPED``, σ/P ≤ ``_ORBIT_SIGMA_OVER_P_THRESHOLD``
    for periodic-relative groups), the Möbius method is selected and
    the result agrees with v2.1 to floating-point precision.

    Both densities must share the full parameter structure: number of
    attributes, group assignment, per-attribute ``r``, and per-group
    ``sigma``/``is_rel``/``is_per``/``period``. Weights and event/slot
    counts may differ freely — that's the whole point of the similarity
    measure.
    """
    # --- Structural compatibility ---
    if dens_x.n_attrs != dens_y.n_attrs:
        raise ValueError("Both MaetDensities must have the same n_attrs.")
    if not np.array_equal(dens_x.group_of_attr, dens_y.group_of_attr):
        raise ValueError(
            "Both MaetDensities must have the same group_of_attr."
        )
    if not np.array_equal(dens_x.r, dens_y.r):
        raise ValueError("Both MaetDensities must have the same r (per attribute).")
    if not np.array_equal(dens_x.sigma, dens_y.sigma):
        raise ValueError("Both MaetDensities must have the same sigma (per group).")
    if not np.array_equal(dens_x.is_rel, dens_y.is_rel):
        raise ValueError("Both MaetDensities must have the same is_rel (per group).")
    if not np.array_equal(dens_x.is_per, dens_y.is_per):
        raise ValueError("Both MaetDensities must have the same is_per (per group).")
    # Periods must match for groups where is_per is True (non-periodic
    # groups can carry any period value without affecting the kernel).
    per_mask = dens_x.is_per.astype(bool)
    if np.any(dens_x.period[per_mask] != dens_y.period[per_mask]):
        raise ValueError(
            "Both MaetDensities must have the same period for periodic groups."
        )

    if method not in ("auto", "bulger", "direct", "mobius"):
        raise ValueError(
            f"method must be one of 'auto', 'bulger', 'direct'; "
            f"got {method!r}."
        )

    # --- Dispatcher ---
    A = dens_x.n_attrs
    r_vec = dens_x.r
    is_rel_g = dens_x.is_rel
    is_per_g = dens_x.is_per
    sigma_g = dens_x.sigma
    period_g = dens_x.period

    r_max = int(np.max(r_vec)) if A > 0 else 1
    # Maximum σ/P across groups that are both relative AND periodic.
    sop_max = 0.0
    any_per = False
    any_rel_nonper = False
    any_rel_per = False
    for g in range(int(dens_x.n_groups)):
        if bool(is_per_g[g]):
            any_per = True
        if bool(is_rel_g[g]):
            if bool(is_per_g[g]):
                any_rel_per = True
                if float(period_g[g]) > 0:
                    sop_max = max(
                        sop_max,
                        float(sigma_g[g]) / float(period_g[g]),
                    )
            else:
                any_rel_nonper = True

    # Per-attribute slab dimension K_a (the kernel slab size; events
    # within an attribute may have lower K_eff via NaN padding, which
    # the Möbius-method wrapper handles via per-event safe/unsafe partition).
    k_vec = np.array(
        [int(M.shape[0]) for M in dens_x.p_attr], dtype=np.intp,
    ) if A > 0 else np.zeros(0, dtype=np.intp)

    chosen = _select_ma_inner_product_method(
        r_vec=r_vec, k_vec=k_vec, A=A,
        N_x=int(dens_x.n), N_y=int(dens_y.n),
        any_per=any_per,
        any_rel_nonper=any_rel_nonper,
        any_rel_per=any_rel_per,
        sigma_over_P_max=sop_max,
        user_method=method,
    )

    if chosen == "mobius":
        ip_xy, ip_xx, ip_yy = _cos_sim_exp_tens_ma_orbit(
            dens_x, dens_y,
        )
        # Two layers of Möbius-method result validation, fall back on either.
        # The per-entry worst_ratio diagnostic that previously gated
        # this fallback (analogous to the SA case) was found to fire
        # spuriously for self-IP matrices: it reports per-(n,m) entry
        # cancellation in the per-attribute Möbius alternating partition sums, but
        # the cosine consumes only Σ_{n,m} P[n,m], where individual
        # entries with bad ratios contribute negligibly. Empirically,
        # at typical musical sigmas the diagnostic flagged ~100% of
        # MA self-IPs while the values themselves matched Bulger's method
        # to FP precision. The cross-cancellation guard plus the
        # post-hoc IP corruption check below catch the residual real
        # failure modes (small/sign-flipped cosines and non-finite
        # IPs respectively).
        denom_geo = np.sqrt(max(ip_xx * ip_yy, 0.0))
        cross_cancellation = (
            denom_geo > 0
            and abs(ip_xy) < cancellation_threshold * denom_geo
        )
        ips_corrupted = _orbit_ips_look_corrupted(ip_xy, ip_xx, ip_yy)
        if cross_cancellation or ips_corrupted:
            ip_xy, ip_xx, ip_yy = _cos_sim_exp_tens_ma_pairwise(
                dens_x, dens_y, verbose=verbose,
            )
    else:  # 'bulger' or 'direct' (coincide in MA mode)
        ip_xy, ip_xx, ip_yy = _cos_sim_exp_tens_ma_pairwise(
            dens_x, dens_y, verbose=verbose,
        )

    denom = np.sqrt(ip_xx * ip_yy)
    if denom == 0:
        return float("nan")
    return float(ip_xy / denom)


def _ip_core_ma(
    u_cell, w_u, n_j, v_cell, w_v, n_k,
    A, group_of, r_vec, sigma_g, is_rel_g, is_per_g, period_g,
):
    """MA inner product with memory-aware chunking along the comb side.

    Peak per-chunk memory is dominated by the largest per-attribute
    (r_a, nJ, nKc) difference tensor. Use ``(max(r_a) + 2) * nJ * 8``
    bytes per K-column as the sizing heuristic.
    """
    max_r = int(np.max(r_vec)) if A > 0 else 1
    bytes_per_col = (max_r + 2) * int(n_j) * 8
    mem_limit = 4_000_000_000  # 4 GB default
    bytes_needed = bytes_per_col * int(n_k)

    if bytes_needed <= mem_limit:
        return _ip_full_ma(
            u_cell, w_u, n_j, v_cell, w_v, n_k,
            A, group_of, r_vec, sigma_g, is_rel_g, is_per_g, period_g,
        )

    chunk_size = max(1, int(mem_limit // max(bytes_per_col, 1)))
    acc = np.zeros(int(n_j), dtype=np.float64)
    for c_start in range(0, int(n_k), chunk_size):
        c_end = min(c_start + chunk_size, int(n_k))
        n_kc = c_end - c_start
        v_chunk = [V[:, c_start:c_end] for V in v_cell]
        log_kernel = _ma_log_kernel(
            u_cell, v_chunk, int(n_j), n_kc,
            A, group_of, r_vec, sigma_g, is_rel_g, is_per_g, period_g,
        )
        E = np.exp(log_kernel)
        acc = acc + E @ w_v[c_start:c_end]
    return float(w_u @ acc)


def _ip_full_ma(
    u_cell, w_u, n_j, v_cell, w_v, n_k,
    A, group_of, r_vec, sigma_g, is_rel_g, is_per_g, period_g,
):
    """Fully vectorized MA inner product (single chunk)."""
    log_kernel = _ma_log_kernel(
        u_cell, v_cell, int(n_j), int(n_k),
        A, group_of, r_vec, sigma_g, is_rel_g, is_per_g, period_g,
    )
    E = np.exp(log_kernel)
    return float(w_u @ (E @ w_v))


def _ma_log_kernel(
    u_cell, v_cell, n_j, n_k,
    A, group_of, r_vec, sigma_g, is_rel_g, is_per_g, period_g,
):
    """Accumulate the summed-Q / (4 sigma^2) log-kernel across attributes.

    For each attribute *a*:
      1. Compute (r_a, nJ, nK) differences between perm-side and comb-side.
      2. Apply periodic wrapping for the attribute's group.
      3. Compute the per-attribute quadratic form Q_a.
      4. Accumulate ``-Q_a / (4 sigma_g^2)`` into log_kernel.
    """
    log_kernel = np.zeros((int(n_j), int(n_k)), dtype=np.float64)
    for a in range(A):
        g = int(group_of[a])
        r_a = int(r_vec[a])
        D = u_cell[a][:, :, None] - v_cell[a][:, None, :]  # (r_a, nJ, nK)

        if is_per_g[g]:
            p_g = float(period_g[g])
            D = np.mod(D + p_g / 2, p_g) - p_g / 2

        Q_a = _compute_Q(D, r_a, bool(is_rel_g[g]), bool(is_per_g[g]),
                         float(period_g[g]))
        log_kernel = log_kernel - Q_a / (4 * float(sigma_g[g]) ** 2)

    return log_kernel


# -------------------------------------------------------------------
#  Möbius method dispatcher (multi-attribute path)
# -------------------------------------------------------------------
#
#  Per the MAET inner-product factorisation (JMM Eq. 3.4 with the
#  per-attribute integral separation noted in JMM §3.2.1, and the
#  Möbius–Bulger remark JMM Rem. 3.1), the cross-event inner product
#  decomposes as
#
#      <T_X, T_Y>_MA = Σ_{n_X, n_Y} Π_a I_a(n_X, n_Y)
#
#  where I_a(n_X, n_Y) is an SA-shaped Möbius inner product over
#  the K_a slot values of event n_X (X-side) against those of n_Y
#  (Y-side), with the group's mode parameters. Bulger's method
#  collapses this into a flat (n_J × n_K) bilinear form that scales
#  as N² · Π_a [r_a! · C(K_a, r_a)]²; the Möbius form scales as
#  N² · A · |Ω_{r_a}| · K_a², a substantial saving when K_a is
#  non-trivial.
#
#  Limitations of the Möbius method:
#  - NaN-padded ``p_attr`` (variable K_a per event) is not yet
#    supported by the per-event Möbius loop; dispatcher detects and
#    falls back to Bulger's method.
#  - Per-attribute r_a > _ORBIT_R_MAX_SHIPPED falls back (no orbit
#    table shipped at that order).
#
#  Per-attribute Möbius calls apply the SA convention's
#  (σ_a √π)^{r_a} prefactor, so the Möbius-method MA bare triple
#  (ip_xy, ip_xx, ip_yy) differs from Bulger's MA triple by
#  Π_a (σ_a √π)^{r_a} · r_a! — which cancels in the cosine.


def _ma_has_nan(dens):
    """True if any p_attr matrix has NaN entries (variable K_a per event)."""
    return any(np.isnan(M).any() for M in dens.p_attr)


# Per-r K thresholds for the Möbius-vs-Bulger crossover, established
# empirically on representative MAET workloads (N = 8-12, σ = 12,
# P = 1200, samples_per_sigma = 5). Retained for reference but
# superseded by the cost-model dispatcher below, which also accounts
# for N (which the K thresholds alone do not — at N = 2 the abs
# crossover is K ≥ 14 for r = 2, but at N = 16 it is K ≥ 6, a span a
# single threshold can't capture).
_K_THRESHOLD_ABS = {2: 7, 3: 6, 4: 5, 5: 4, 6: 4}
_K_THRESHOLD_REL_PER = {2: float("inf"), 3: 10, 4: 8, 5: 7, 6: 6}


# Cost-model constants for the dispatcher. Refit on a 328-cell wall-time
# benchmark covering all four modes (abs/rel × per/nonper) at A ∈ {1, 2},
# r ∈ {2, 3, 4}, N ∈ {2, 4, 8, 16}, K spanning each mode's feasible
# range. r ∈ {5, 6} extrapolated from |Ω_r| growth (4, 10, 33, 92, 306,
# 948). Predicts Bulger and Möbius wall times in milliseconds and picks
# the smaller. Validated against 277 measured cells: 94 % within 5 % of
# optimal, 0 mis-routes to Bulger's method (no OOM-zone violations), 7
# close-call mis-routes to the Möbius method (max 3.9 × slowdown, all at
# < 100 ms absolute).

# Bulger: per-entry cost of the (n_J × n_K) kernel matrix in ms. The
# periodic branches build a wrapped-difference tensor, which empirically
# costs ~2.0–2.3 × the non-periodic branch (modular arithmetic plus
# index-array growth). Verified across both abs and rel modes.
_PW_PER_ENTRY_MS_NONPER = 1.0e-4
_PW_PER_ENTRY_MS_PER = 7.0e-4   # p75 of measured per-entry cost (per bucket)


def _pw_per_entry_ms(any_per):
    """Pick the per-entry cost for Bulger's method based on whether any group wraps."""
    return _PW_PER_ENTRY_MS_PER if any_per else _PW_PER_ENTRY_MS_NONPER


# Möbius method (absolute modes, both per and nonper — empirically
# within ±5 % of each other). Vectorised across event pairs, so cost
# is roughly constant in N_x · N_y; linear in A at r = 2, 3 and slightly
# sub-linear at r = 4. Per-r baseline at A = 1. Entries for r >= 7
# extrapolated from the empirical 3× orbit-class count growth per r (anchored
# to measured r=2..6 values); these are conservative and may be refined later.
_ORBIT_ABS_PER_ATTR_MS = {
    2: 3.0, 3: 11.2, 4: 45.0, 5: 150.0, 6: 500.0,
    7: 1500.0, 8: 4500.0,
}

# Möbius method (relative-periodic): vectorised across event pairs but each
# pair carries a u-grid integration of N_u ≈ period/σ × samples_per_σ
# samples, plus a fixed per-call setup cost (~5 ms). Cost grows with
# A · N_x · N_y · K_max² · |Ω_r|.
_ORBIT_RELPER_BASE_MS = 5.0
_ORBIT_RELPER_PER_PAIR_K2_MS = {
    2: 0.06, 3: 0.40, 4: 1.0, 5: 5.0, 6: 20.0,
    7: 60.0, 8: 200.0,
}

# Möbius method (relative-aperiodic): the implementation here is a
# per-(n_X, n_Y) Python loop (not batched across event pairs), so the
# per-pair-K² constant is roughly 4 × the rel-periodic constant. At
# A = 1 this Möbius branch is almost always slower than Bulger's method;
# at A ≥ 2 it wins comfortably once K is moderate, because Bulger's
# method grows as ∏_a C(K_a, r_a)² which compounds across attributes
# whereas the Möbius method adds linearly.
_ORBIT_RELNONPER_BASE_MS = 5.0
_ORBIT_RELNONPER_PER_PAIR_K2_MS = {
    2: 0.25, 3: 1.05, 4: 3.30, 5: 12.0, 6: 50.0,
    7: 150.0, 8: 500.0,
}


def _orbit_beats_pairwise_per_attr(r, K, is_rel, is_per):
    """Per-attribute K-threshold heuristic for Möbius-vs-Bulger crossover (legacy; superseded).

    Retained for callers that haven't migrated; the cost-model
    dispatcher in ``_select_ma_inner_product_method`` is preferred.
    """
    if r == 1:
        return False
    if r > _ORBIT_R_MAX_SHIPPED:
        return False
    if is_rel and is_per:
        threshold = _K_THRESHOLD_REL_PER.get(r, 999)
    else:
        threshold = _K_THRESHOLD_ABS.get(r, 999)
    return K >= threshold


def _predict_pairwise_kernel_size(r_vec, k_vec, A, N_x, N_y):
    """Predicted n_J · n_K for the MA path under Bulger's method.

    n_J^X = N_x · ∏_a r_a! · C(K_a, r_a)
    n_K^Y = N_y · ∏_a C(K_a, r_a)
    so n_J · n_K = N_x · N_y · ∏_a r_a! · C(K_a, r_a)².
    """
    if A == 0:
        return float(N_x * N_y)
    size = float(N_x * N_y)
    for a in range(A):
        r_a = int(r_vec[a])
        K_a = int(k_vec[a])
        if K_a < r_a:
            return float('inf')
        c = float(_math_comb(K_a, r_a))
        size *= float(factorial(r_a)) * c * c
    return size


def _predict_orbit_cost_ms(
    r_max, A, N_x, N_y, k_vec, any_rel_nonper, any_rel_per,
):
    """Predicted Möbius-method MA wall time in milliseconds.

    Routes to the appropriate per-r constant based on the group mode:
    rel-aperiodic uses the per-pair Python-loop constants (largest);
    rel-periodic uses the u-grid-integration constants; absolute uses
    the vectorised batch constants. A scaling is linear (verified at
    r = 2, 3 to within ~5 %; slightly sub-linear at r = 4 but linear-A
    over-predicts conservatively, biasing the dispatcher toward
    Bulger's method in close calls at r = 4 — and at r = 4 Bulger's
    side explodes so quickly that this never matters in the OOM zone).
    """
    K_max = int(np.max(k_vec)) if A > 0 else 1
    if any_rel_nonper:
        c = _ORBIT_RELNONPER_PER_PAIR_K2_MS[r_max]
        return float(A * (_ORBIT_RELNONPER_BASE_MS
                          + N_x * N_y * K_max * K_max * c))
    if any_rel_per:
        c = _ORBIT_RELPER_PER_PAIR_K2_MS[r_max]
        return float(A * (_ORBIT_RELPER_BASE_MS
                          + N_x * N_y * K_max * K_max * c))
    return float(A * _ORBIT_ABS_PER_ATTR_MS[r_max])


def _select_ma_inner_product_method(
    *,
    r_vec, k_vec, A,
    N_x, N_y,
    any_per, any_rel_nonper, any_rel_per,
    sigma_over_P_max, user_method,
):
    """Pick the inner-product method for the MA case using a cost model.

    Routing rules, in order:

    1. ``user_method`` keyword override (anything other than 'auto').
    2. Hard fallbacks where the Möbius method cannot or should not run:
       - r_max ≤ 1: no within-tuple structure to exploit.
       - r_max > _ORBIT_R_MAX_SHIPPED: no orbit table available.
    3. Soft fallback: rel + per with σ/P beyond the integration-exact
       regime warns and routes to Bulger's method.
    4. Otherwise predict both wall times (in ms) and pick the smaller;
       ties favour Bulger's method (no orbit-table fetch, no Möbius
       cancellation risk).

    Ragged K_{a,n} (NaN-padded events) is handled inside
    :func:`_ma_per_attr_inner_matrix` via a per-event safe/unsafe
    partition: events with K_eff - r >= 2 (the Möbius-method precision margin)
    flow through the vectorised batched Möbius evaluator; pairs
    involving any K_eff - r < 2 event flow through direct r-tuple
    enumeration (no Möbius alternating sum, hence no cancellation).
    The dispatcher therefore does not route on the presence of NaN entries.

    The four modes (abs+nonper, abs+per, rel+nonper, rel+per) are
    routed as follows:

    - abs + nonper: cost model with `_PW_PER_ENTRY_MS_NONPER` and
      `_ORBIT_ABS_PER_ATTR_MS`.
    - abs + per: cost model with `_PW_PER_ENTRY_MS_PER` (wrap on δ
      tensor adds ~2 × Bulger overhead) and same Möbius constants
      (Möbius cost is mode-independent in benchmark, ±5 %).
    - rel + per: cost model with `_PW_PER_ENTRY_MS_PER` and
      `_ORBIT_RELPER_PER_PAIR_K2_MS` (Möbius u-grid integration
      scales with N_x · N_y · K_max² · |Ω_r|).
    - rel + nonper: cost model with `_PW_PER_ENTRY_MS_NONPER` and
      `_ORBIT_RELNONPER_PER_PAIR_K2_MS` (Möbius per-pair Python loop;
      ~4 × the rel-per per-K² constant). At A = 1 the cost model
      reliably routes to Bulger's method; at A ≥ 2 it routes to the
      Möbius method once Bulger's ∏_a C(K_a, r_a)² compounding
      overtakes the Möbius method's additive A · K_max² growth.

    Parameters
    ----------
    r_vec : (A,) intp
        Per-attribute r_a.
    k_vec : (A,) intp
        Per-attribute slab dimension K_a (the kernel slab size; events
        within an attribute may have lower K_eff via NaN padding,
        which the Möbius-method wrapper handles via per-event
        safe/unsafe partition).
    A : int
        Number of attributes.
    N_x, N_y : int
        Event counts of the two densities.
    any_per : bool
        True if any group has is_per=True (drives Bulger wrap cost).
    any_rel_nonper : bool
    any_rel_per : bool
    sigma_over_P_max : float
        Maximum σ/P across periodic-relative groups.
    user_method : {'auto', 'bulger', 'mobius', 'direct'}
    """
    if user_method != 'auto':
        return user_method
    r_max = int(np.max(r_vec)) if A > 0 else 1
    if r_max <= 1:
        return 'bulger'
    if r_max > _ORBIT_R_MAX_SHIPPED:
        return 'bulger'
    # K-vs-r precision guard. The Möbius method's auto-inner-products can
    # suffer catastrophic Möbius cancellation when any K_a is too close
    # to its r_a (see _ORBIT_K_MINUS_R_MIN block). The cross
    # cancellation guard at the call site does NOT catch this, since it
    # inspects only |<T_X,T_Y>|; corrupted <T_X,T_X> propagates silently
    # into the cosine denominator.
    if A > 0 and not _orbit_safe_for_precision(r_vec, k_vec):
        return 'bulger'
    # Periodic-relative beyond σ/P threshold: in this regime the Möbius
    # method computes the JMM Eq. 3.4 integral form, while Bulger's
    # method computes the single-nearest-image-wrap form.
    # The two diverge by O((σ/P)^∞) starting around σ/P ≈ 0.03. For
    # backward compatibility the toolbox treats Bulger's
    # pairwise-wrap form as canonical; the Möbius method is therefore
    # disabled above the threshold. Users who want the JMM-exact integral
    # explicitly may pass method='mobius'.
    if any_rel_per and sigma_over_P_max > _ORBIT_SIGMA_OVER_P_THRESHOLD:
        warnings.warn(
            f"Maximum σ/P = {sigma_over_P_max:.3f} across periodic-relative "
            f"groups exceeds the Möbius-method threshold "
            f"({_ORBIT_SIGMA_OVER_P_THRESHOLD}); falling back to Bulger's "
            f"method (the pairwise-wrap form). Pass method='bulger' "
            f"explicitly to silence this warning."
        )
        return 'bulger'

    pw_size = _predict_pairwise_kernel_size(r_vec, k_vec, A, N_x, N_y)
    pw_cost_ms = pw_size * _pw_per_entry_ms(any_per)
    orbit_cost_ms = _predict_orbit_cost_ms(
        r_max, A, N_x, N_y, k_vec, any_rel_nonper, any_rel_per,
    )
    if pw_cost_ms <= orbit_cost_ms:
        return 'bulger'
    return 'mobius'


def _ma_per_attr_inner_matrix(
    Px, Wx, Py, Wy, sigma, r, is_rel, is_per, period,
    *, return_cancellation_ratio=False,
):
    """Per-attribute (event_X, event_Y) inner product matrix for the
    MA path under the Möbius method.

    ``Px`` is (K, N_x), ``Wx`` is (K, N_x); same shape for Y. Returns
    an (N_x, N_y) matrix where entry (n_X, n_Y) is the per-attribute
    inner product over the K slot values of event n_X (X-side) against
    those of n_Y (Y-side).

    Strategy (in parity with MATLAB ``mobius.maPerAttrInnerMatrix``):

    - r = 1: direct kernel sum with NaN -> zero-weight padding (no
      Möbius decomposition, cancellation impossible).

    - r >= 2 abs: hybrid safe/unsafe partition. An event is "safe" on
      this attribute iff its non-NaN slot count K_eff satisfies
      ``K_eff - r >= _ORBIT_K_MINUS_R_MIN`` (= 2; the precision margin
      used elsewhere in the Möbius machinery). Safe-vs-safe pairs flow
      through the vectorised batched Möbius method with within-safe-group
      zero-padding. Pairs involving any unsafe event flow through
      :func:`_inner_product_direct_abs_sa`, which is exact for any
      K >= r (no Möbius alternating sum, so no cancellation).

    - r >= 2 rel: per-event-pair loop with zero-pad. Auto dispatch
      routes any rel group globally to Bulger's method; this path runs
      only on explicit ``method='mobius'`` opt-in. Events with K_eff - r
      below the precision margin in this niche regime may lose
      precision in the Möbius relative-mode u-grid integration; users
      wanting exact rel + ragged Möbius-method behaviour should either
      filter events to K_eff >= r + 2 or use ``method='auto'`` (which
      routes to Bulger's method).

    With ``return_cancellation_ratio=True``, additionally returns the
    worst-case (minimum) cancellation ratio across the (N_x, N_y)
    entries — a scalar in (0, 1]. Direct-enum entries always have
    ratio 1.0; the worst ratio comes from the safe-Möbius submatrix.
    If no safe pairs exist, the worst ratio is 1.0.
    """
    from ._mobius import inner_product_orbit_pw_batched

    K_x_max, N_x = Px.shape
    K_y_max, N_y = Py.shape

    # --- r = 1: direct kernel sum, zero-pad fine (no cancellation) ---
    if r == 1:
        Px_, Wx_, Py_, Wy_ = _zero_pad_nan(Px, Wx, Py, Wy)
        diffs = Px_[:, :, None, None] - Py_[None, None, :, :]
        if is_per:
            diffs = diffs - period * np.floor(diffs / period + 0.5)
        K_tens = np.exp(-(diffs ** 2) / (4 * sigma ** 2))
        out = np.einsum(
            'xn,xnym,ym->nm', Wx_, K_tens, Wy_, optimize=True,
        )
        result = out * (sigma * np.sqrt(np.pi)) ** r
        if return_cancellation_ratio:
            return result, 1.0
        return result

    # --- r >= 2 rel: per-pair loop, zero-pad (rare regime) ---
    if is_rel:
        Px_, Wx_, Py_, Wy_ = _zero_pad_nan(Px, Wx, Py, Wy)
        if is_per:
            return _ma_per_attr_inner_matrix_rel_per(
                Px_, Wx_, Py_, Wy_, sigma, r, period,
                return_cancellation_ratio=return_cancellation_ratio,
            )
        out = np.empty((N_x, N_y), dtype=np.float64)
        worst_ratio = 1.0
        for n_X in range(N_x):
            for n_Y in range(N_y):
                if return_cancellation_ratio:
                    v, ratio = _orbit_inner_rel(
                        Px_[:, n_X], Wx_[:, n_X], Py_[:, n_Y], Wy_[:, n_Y],
                        sigma, r, False, period,
                        return_cancellation_ratio=True,
                    )
                    out[n_X, n_Y] = v
                    if ratio < worst_ratio:
                        worst_ratio = ratio
                else:
                    out[n_X, n_Y] = _orbit_inner_rel(
                        Px_[:, n_X], Wx_[:, n_X], Py_[:, n_Y], Wy_[:, n_Y],
                        sigma, r, False, period,
                    )
        if return_cancellation_ratio:
            return out, worst_ratio
        return out

    # --- r >= 2 abs: hybrid safe/unsafe partition ---

    K_MARGIN_MIN = _ORBIT_K_MINUS_R_MIN

    # Per-event K_eff (count of non-NaN slots), per side.
    K_eff_x = np.sum(~(np.isnan(Px) | np.isnan(Wx)), axis=0)   # (N_x,)
    K_eff_y = np.sum(~(np.isnan(Py) | np.isnan(Wy)), axis=0)   # (N_y,)

    safe_x_mask = (K_eff_x - r) >= K_MARGIN_MIN
    safe_y_mask = (K_eff_y - r) >= K_MARGIN_MIN
    safe_x_idx = np.where(safe_x_mask)[0]
    unsafe_x_idx = np.where(~safe_x_mask)[0]
    safe_y_idx = np.where(safe_y_mask)[0]
    unsafe_y_idx = np.where(~safe_y_mask)[0]

    out = np.zeros((N_x, N_y), dtype=np.float64)
    worst_ratio = 1.0

    # --- Safe x Safe submatrix: vectorised batched Möbius method ---
    if safe_x_idx.size > 0 and safe_y_idx.size > 0:
        Px_s = Px[:, safe_x_idx]
        Wx_s = Wx[:, safe_x_idx]
        Py_s = Py[:, safe_y_idx]
        Wy_s = Wy[:, safe_y_idx]
        # Within-safe zero-pad (K still varies per event in safe group).
        Px_s, Wx_s, Py_s, Wy_s = _zero_pad_nan(Px_s, Wx_s, Py_s, Wy_s)

        N_xs = safe_x_idx.size
        N_ys = safe_y_idx.size
        diffs = Px_s[:, :, None, None] - Py_s[None, None, :, :]
        if is_per:
            diffs = diffs - period * np.floor(diffs / period + 0.5)
        K_tens = np.exp(-(diffs ** 2) / (4 * sigma ** 2))
        K_pairs = np.transpose(K_tens, (1, 3, 0, 2)).reshape(
            N_xs * N_ys, K_x_max, K_y_max,
        )
        w_A_pairs = np.broadcast_to(
            Wx_s.T[:, None, :], (N_xs, N_ys, K_x_max),
        ).reshape(N_xs * N_ys, K_x_max)
        w_B_pairs = np.broadcast_to(
            Wy_s.T[None, :, :], (N_xs, N_ys, K_y_max),
        ).reshape(N_xs * N_ys, K_y_max)

        if return_cancellation_ratio:
            flat, ratios = inner_product_orbit_pw_batched(
                K_pairs, w_A_pairs, w_B_pairs, r,
                prefactor=(sigma * np.sqrt(np.pi)) ** r,
                return_cancellation_ratio=True,
            )
            worst_ratio = min(worst_ratio, float(np.min(ratios)))
        else:
            flat = inner_product_orbit_pw_batched(
                K_pairs, w_A_pairs, w_B_pairs, r,
                prefactor=(sigma * np.sqrt(np.pi)) ** r,
            )
        # Use ix_ for fancy 2-D indexing into the output.
        out[np.ix_(safe_x_idx, safe_y_idx)] = flat.reshape(N_xs, N_ys)

    # --- Pairs involving any unsafe event: K-grouped batched direct ---
    # All pairs not in (safe_x, safe_y) flow through ordered-r-tuple
    # direct enumeration. Was previously a Python double-loop
    # (one ``_inner_product_direct_abs_sa`` call per pair); for
    # variable-K_a workloads with many unsafe events this dominated
    # the runtime by 10–100× over the actual computation.
    #
    # K_a grouping: partition the unsafe-involved event-index union
    # by K_eff value per side, then batch direct enumeration per
    # (K_eff_x, K_eff_y) sub-block. Within a sub-block, every event
    # shares an ordered-r-tuple shape (nJ = K_eff! / (K_eff - r)!),
    # so the IP matrix can be computed as a single contracted
    # tensor op. This removes the Python per-pair overhead entirely.
    #
    # Coverage: (unsafe_x, all_y) ∪ (safe_x, unsafe_y) covers every
    # pair where at least one side is unsafe, without double counting.
    needed_x_idx = np.concatenate([unsafe_x_idx, safe_x_idx]) \
        if unsafe_x_idx.size > 0 else np.array([], dtype=np.intp)
    needed_y_idx_full = np.arange(N_y)
    # Split needed pairs into two coverage zones to mirror the
    # inline-method structure exactly, preserving fill ordering.
    _ma_fill_direct_enum_groups(
        out, Px, Wx, Py, Wy,
        unsafe_x_idx, np.arange(N_y),
        K_eff_x, K_eff_y, sigma, r, is_per, period,
    )
    if unsafe_y_idx.size > 0 and safe_x_idx.size > 0:
        _ma_fill_direct_enum_groups(
            out, Px, Wx, Py, Wy,
            safe_x_idx, unsafe_y_idx,
            K_eff_x, K_eff_y, sigma, r, is_per, period,
        )

    if return_cancellation_ratio:
        return out, worst_ratio
    return out


def _ma_fill_direct_enum_groups(
    out, Px, Wx, Py, Wy, x_idx, y_idx,
    K_eff_x, K_eff_y, sigma, r, is_per, period,
):
    """K-grouped batched direct-enum fill into ``out`` for a rectangle
    of (x_idx, y_idx) pairs.

    Partitions ``x_idx`` by K_eff_x value and ``y_idx`` by K_eff_y
    value, then computes each (K_x_val, K_y_val) sub-block as a single
    vectorised tensor contraction. Output entries at (x_idx[i],
    y_idx[j]) are filled in place.

    No-op if either side is empty.
    """
    if x_idx.size == 0 or y_idx.size == 0:
        return

    # Unique K_eff values present on each side (within the index sets).
    unique_K_x = np.unique(K_eff_x[x_idx])
    unique_K_y = np.unique(K_eff_y[y_idx])

    for K_x_val in unique_K_x:
        x_grp = x_idx[K_eff_x[x_idx] == K_x_val]
        if x_grp.size == 0 or int(K_x_val) < r:
            # K < r: ordered r-tuple set is empty; IP = 0.
            continue
        # Pack non-NaN slots to the top of each group column. The
        # build_exp_tens convention has NaN already at the bottom, so
        # in the common case this is a memory-cheap slice; in the
        # general case _pack_nan_top handles arbitrary NaN positions.
        Px_grp, Wx_grp = _pack_nan_top(Px[:, x_grp], Wx[:, x_grp])
        Px_grp = Px_grp[:int(K_x_val), :]
        Wx_grp = Wx_grp[:int(K_x_val), :]
        for K_y_val in unique_K_y:
            y_grp = y_idx[K_eff_y[y_idx] == K_y_val]
            if y_grp.size == 0 or int(K_y_val) < r:
                continue
            Py_grp, Wy_grp = _pack_nan_top(Py[:, y_grp], Wy[:, y_grp])
            Py_grp = Py_grp[:int(K_y_val), :]
            Wy_grp = Wy_grp[:int(K_y_val), :]
            sub_ip = _batched_direct_enum_abs_sa(
                Px_grp, Wx_grp, Py_grp, Wy_grp,
                sigma, r, is_per, period,
            )
            out[np.ix_(x_grp, y_grp)] = sub_ip


def _pack_nan_top(P, W):
    """Pack non-NaN slots to the top of each column.

    Returns ``(P_packed, W_packed)`` of the same shape, where for each
    column ``n`` the first ``K_eff[n]`` rows are the valid slots
    (preserving their original order) and the rest are NaN. The
    ``build_exp_tens`` convention already places NaN at the bottom, in
    which case this is mathematically a no-op (still copies for
    cleanliness). Per-event packing handles user-constructed densities
    with arbitrary NaN positions.
    """
    K, N = P.shape
    P_packed = np.full_like(P, np.nan)
    W_packed = np.full_like(W, np.nan)
    for n in range(N):
        valid = ~(np.isnan(P[:, n]) | np.isnan(W[:, n]))
        k = int(valid.sum())
        if k == 0:
            continue
        P_packed[:k, n] = P[valid, n]
        W_packed[:k, n] = W[valid, n]
    return P_packed, W_packed


def _batched_direct_enum_abs_sa(
    Px_group, Wx_group, Py_group, Wy_group,
    sigma, r, is_per, period,
):
    """Batched direct r-tuple enumeration IP for groups at fixed K_x, K_y.

    Vectorised replacement for repeated calls to
    :func:`_inner_product_direct_abs_sa` when every event in
    ``Px_group`` has the same ``K_x = K_eff_x`` and every event in
    ``Py_group`` has the same ``K_y = K_eff_y`` (no NaN within the
    first K rows of either side).

    Inputs
    ------
    Px_group : (K_x, N_x) ndarray
        Slot positions, no NaN.
    Wx_group : (K_x, N_x) ndarray
        Slot weights, no NaN.
    Py_group, Wy_group : (K_y, N_y) ndarrays
        Same for Y side.
    sigma, r, is_per, period
        Group parameters.

    Returns
    -------
    ip : (N_x, N_y) ndarray
        Inner-product matrix (no Möbius alternating sum; exact for any
        K_x, K_y >= r).
    """
    K_x, N_x = Px_group.shape
    K_y, N_y = Py_group.shape

    if K_x < r or K_y < r:
        return np.zeros((N_x, N_y), dtype=np.float64)

    if r == 1:
        diffs = Px_group[:, :, None, None] - Py_group[None, None, :, :]
        if is_per:
            diffs = diffs - period * np.floor(diffs / period + 0.5)
        K_mat = np.exp(-(diffs ** 2) / (4 * sigma ** 2))
        ip = np.einsum(
            'xn,xnym,ym->nm', Wx_group, K_mat, Wy_group, optimize=True,
        )
        return ip * (sigma * np.sqrt(np.pi))

    # --- r >= 2: enumerate ordered r-tuple indices ---
    from itertools import permutations
    idx_x = np.array(list(permutations(range(K_x), r)),
                     dtype=np.intp)        # (nJ_x, r)
    idx_y = np.array(list(permutations(range(K_y), r)),
                     dtype=np.intp)        # (nJ_y, r)
    nJ_x = idx_x.shape[0]                  # K_x! / (K_x - r)!
    nJ_y = idx_y.shape[0]

    # Gather tuple slot positions and weights per event. The fancy
    # index Px_group[idx_x.T, :] has shape (r, nJ_x, N_x); we want
    # U_x of shape (r, N_x, nJ_x) and Wj_x of shape (N_x, nJ_x).
    U_x = Px_group[idx_x.T, :].transpose(0, 2, 1)
    U_y = Py_group[idx_y.T, :].transpose(0, 2, 1)
    Wj_x = np.prod(Wx_group[idx_x.T, :], axis=0).T  # (N_x, nJ_x)
    Wj_y = np.prod(Wy_group[idx_y.T, :], axis=0).T  # (N_y, nJ_y)

    # Memory estimate: the difference tensor is (r, N_x, nJ_x, N_y, nJ_y).
    # For unsafe events (K_eff in {r, r+1}), nJ_x = r! or (r+1)!/(1!),
    # which is small. For r=3, K=4 -> nJ=24; r=4, K=5 -> nJ=120. Even
    # with N_x = N_y = 100 this is <100 MB at worst. No chunking needed
    # in the unsafe regime. Document so future use on safe-K paths
    # adds a chunking guard.
    diffs = U_x[:, :, :, None, None] - U_y[:, None, None, :, :]
    if is_per:
        diffs = diffs - period * np.floor(diffs / period + 0.5)
    Q = np.sum(diffs ** 2, axis=0)                  # (N_x, nJ_x, N_y, nJ_y)
    K_mat = np.exp(-Q / (4 * sigma ** 2))

    ip = np.einsum(
        'xj,xjyk,yk->xy', Wj_x, K_mat, Wj_y, optimize=True,
    )
    return ip * (sigma * np.sqrt(np.pi)) ** r


def _zero_pad_nan(Px, Wx, Py, Wy):
    """Replace NaN entries in P / W with 0 (zero-weight padding).

    Returns new arrays (does not mutate inputs). The Möbius method's weighted
    contractions read ``w_i^m``, so a zero-weight slot kills any Möbius
    term involving that slot regardless of the corresponding p value
    — mathematically equivalent to per-event truncation.
    """
    nan_x = np.isnan(Px) | np.isnan(Wx)
    if nan_x.any():
        Px = np.where(nan_x, 0.0, Px)
        Wx = np.where(nan_x, 0.0, Wx)
    nan_y = np.isnan(Py) | np.isnan(Wy)
    if nan_y.any():
        Py = np.where(nan_y, 0.0, Py)
        Wy = np.where(nan_y, 0.0, Wy)
    return Px, Wx, Py, Wy


def _ma_per_attr_inner_matrix_rel_per(
    Px, Wx, Py, Wy, sigma, r, period, samples_per_sigma=5,
    *, return_cancellation_ratio=False,
):
    """Vectorised relative-periodic case of ``_ma_per_attr_inner_matrix``.

    Builds an (N_pairs · N_u, K, K) kernel tensor and runs a single
    batched Möbius-method call across both axes; the trapezoidal weights are
    applied after reshaping back to (N_pairs, N_u). Memory peak is
    ``N_pairs · N_u · K^2 · 8`` bytes plus a similar-sized intermediate
    diffs tensor; chunked along u to stay under a 1 GB ceiling.

    ``samples_per_sigma=5`` is a deliberate trade-off: the Gaussian
    integrand is smooth at the σ scale, so trapezoidal convergence is
    geometric and 5 samples/σ delivers cosine agreement well below
    1e-9 relative on representative MAET parameter ranges. The SA
    relative path uses 10 because per-call cost is small there; for
    MA the integration is run N_pairs = N_x · N_y times in parallel,
    so halving the grid roughly halves wall-clock cost.

    With ``return_cancellation_ratio=True``, additionally returns the
    worst-case ratio across the (N_pairs · N_u) batched Möbius-method cells.
    """
    from ._mobius import inner_product_orbit_pw_batched

    K, N_x = Px.shape
    _, N_y = Py.shape
    N_pairs = N_x * N_y

    N_u = max(64, int(np.ceil(period / sigma * samples_per_sigma)))
    u_grid = np.linspace(0.0, period, N_u, endpoint=False)
    du = period / N_u

    # Per-pair weights (independent of u) — shape (N_pairs, K).
    w_A_pairs = np.broadcast_to(
        Wx.T[:, None, :], (N_x, N_y, K),
    ).reshape(N_pairs, K)
    w_B_pairs = np.broadcast_to(
        Wy.T[None, :, :], (N_x, N_y, K),
    ).reshape(N_pairs, K)

    # Pair-wise raw differences, independent of u: shape (K, N_x, K, N_y).
    diffs_pair = Px[:, :, None, None] - Py[None, None, :, :]

    # Chunk along u to bound memory.
    bytes_per_u = N_pairs * K * K * 8 * 2  # kernel + diffs
    mem_limit = 1_000_000_000
    chunk_u = max(1, min(N_u, mem_limit // max(bytes_per_u, 1)))

    F = np.zeros((N_pairs, N_u), dtype=np.float64)
    worst_ratio = 1.0
    for u_start in range(0, N_u, chunk_u):
        u_end = min(u_start + chunk_u, N_u)
        n_uc = u_end - u_start
        u_slice = u_grid[u_start:u_end]
        # diffs[u, K_i, N_x, K_j, N_y] = diffs_pair + u
        diffs = diffs_pair[None, :, :, :, :] + u_slice[:, None, None, None, None]
        diffs = diffs - period * np.floor(diffs / period + 0.5)
        K_uc = np.exp(-(diffs ** 2) / (4 * sigma ** 2))  # (n_uc, K, N_x, K, N_y)
        # Reorder to (n_uc, N_x, N_y, K, K) and flatten leading axes.
        K_uc = np.transpose(K_uc, (0, 2, 4, 1, 3)).reshape(
            n_uc * N_pairs, K, K,
        )
        # Replicate weights across u-axis for each pair.
        w_A_uc = np.broadcast_to(
            w_A_pairs[None, :, :], (n_uc, N_pairs, K),
        ).reshape(n_uc * N_pairs, K)
        w_B_uc = np.broadcast_to(
            w_B_pairs[None, :, :], (n_uc, N_pairs, K),
        ).reshape(n_uc * N_pairs, K)
        if return_cancellation_ratio:
            flat, ratios = inner_product_orbit_pw_batched(
                K_uc, w_A_uc, w_B_uc, r, prefactor=1.0,
                return_cancellation_ratio=True,
            )
            chunk_min = float(np.min(ratios))
            if chunk_min < worst_ratio:
                worst_ratio = chunk_min
        else:
            flat = inner_product_orbit_pw_batched(
                K_uc, w_A_uc, w_B_uc, r, prefactor=1.0,
            )
        F[:, u_start:u_end] = flat.reshape(n_uc, N_pairs).T

    integral = F.sum(axis=1) * du  # periodic Riemann sum
    c = sigma * np.sqrt(2 * np.pi / r)
    out = (sigma * np.sqrt(np.pi)) ** r * integral / c ** 2
    if return_cancellation_ratio:
        return out.reshape(N_x, N_y), worst_ratio
    return out.reshape(N_x, N_y)


def _cos_sim_exp_tens_ma_orbit(dens_x, dens_y):
    """Compute (ip_xy, ip_xx, ip_yy) for the MA case via per-attribute
    Möbius method (JMM Eq. 3.4 plus Rem. 3.1).

    Caller is responsible for ensuring no NaN in ``p_attr`` and for
    structural compatibility of the two densities.

    Note: previous versions also returned a ``worst_ratio`` aggregating
    per-entry cancellation ratios across the (N_x × N_y) inner-product
    matrices. That diagnostic was found to over-conservatively flag
    correct results — the per-entry ratio reflects cancellation in
    individual Möbius cells, but the cosine consumes only the
    sums Σ_{n,m} P[n,m], where individual entries with bad ratios
    contribute negligibly when their absolute value is small. Removed
    in favour of relying on the cross-cancellation guard
    and the post-hoc IP corruption check (see
    ``_orbit_ips_look_corrupted``) at the dispatcher level.
    """
    A = dens_x.n_attrs
    N_x = dens_x.n
    N_y = dens_y.n

    P_xy = np.ones((N_x, N_y), dtype=np.float64)
    P_xx = np.ones((N_x, N_x), dtype=np.float64)
    P_yy = np.ones((N_y, N_y), dtype=np.float64)

    for a in range(A):
        g = int(dens_x.group_of_attr[a])
        r_a = int(dens_x.r[a])
        sigma = float(dens_x.sigma[g])
        is_rel = bool(dens_x.is_rel[g])
        is_per = bool(dens_x.is_per[g])
        period = float(dens_x.period[g])

        Px, Py = dens_x.p_attr[a], dens_y.p_attr[a]
        Wx, Wy = dens_x.w[a], dens_y.w[a]

        I_xy = _ma_per_attr_inner_matrix(
            Px, Wx, Py, Wy, sigma, r_a, is_rel, is_per, period,
        )
        I_xx = _ma_per_attr_inner_matrix(
            Px, Wx, Px, Wx, sigma, r_a, is_rel, is_per, period,
        )
        I_yy = _ma_per_attr_inner_matrix(
            Py, Wy, Py, Wy, sigma, r_a, is_rel, is_per, period,
        )
        P_xy *= I_xy
        P_xx *= I_xx
        P_yy *= I_yy

    return float(P_xy.sum()), float(P_xx.sum()), float(P_yy.sum())


def _cos_sim_exp_tens_ma_pairwise(dens_x, dens_y, *, verbose: bool = True):
    """Compute (ip_xy, ip_xx, ip_yy) for the MA case via the
    Bulger's method (``_ip_core_ma``).

    This is the body of the original ``_cos_sim_exp_tens_ma``
    factored out so the new dispatcher can route to it cleanly.
    """
    A = dens_x.n_attrs
    group_of = dens_x.group_of_attr
    r_vec = dens_x.r
    sigma_g = dens_x.sigma
    is_rel_g = dens_x.is_rel
    is_per_g = dens_x.is_per
    period_g = dens_x.period

    n_jx, n_kx = dens_x.n_j, dens_x.n_k
    n_jy, n_ky = dens_y.n_j, dens_y.n_k

    total_pairs = n_jx * n_ky + n_jx * n_kx + n_jy * n_ky
    max_r = int(np.max(r_vec)) if A > 0 else 1
    estimate_comp_time(total_pairs, max_r, "cos_sim_exp_tens (MAET)", verbose)

    ip_xy = _ip_core_ma(
        dens_x.u_perm, dens_x.w_j, n_jx,
        dens_y.v_comb, dens_y.wv_comb, n_ky,
        A, group_of, r_vec, sigma_g, is_rel_g, is_per_g, period_g,
    )
    ip_xx = _ip_core_ma(
        dens_x.u_perm, dens_x.w_j, n_jx,
        dens_x.v_comb, dens_x.wv_comb, n_kx,
        A, group_of, r_vec, sigma_g, is_rel_g, is_per_g, period_g,
    )
    ip_yy = _ip_core_ma(
        dens_y.u_perm, dens_y.w_j, n_jy,
        dens_y.v_comb, dens_y.wv_comb, n_ky,
        A, group_of, r_vec, sigma_g, is_rel_g, is_per_g, period_g,
    )
    return ip_xy, ip_xx, ip_yy


# -------------------------------------------------------------------
#  cos_sim_exp_tens_raw  (dispatches SA or MA based on input shape)
# -------------------------------------------------------------------


def cos_sim_exp_tens_raw(
    p1, w1, p2, w2, *args,
    method: str = "auto",
    cancellation_threshold: float = 1e-12,
    verbose: bool = True,
) -> float:
    """Deprecated. Use :func:`cos_sim_exp_tens` directly with raw input.

    .. deprecated:: 2.1
       The raw-input dispatch has been folded into the unified
       :func:`cos_sim_exp_tens` entry point. Pass raw arrays directly:

       - SA: ``cos_sim_exp_tens(p1, w1, p2, w2, sigma, r, is_rel, is_per, period)``
       - MA: ``cos_sim_exp_tens(p_attr1, w1, p_attr2, w2, sigma_vec, r_vec, groups, is_rel_vec, is_per_vec, period_vec)``

       This shim will be removed in a future release.
    """
    warnings.warn(
        "cos_sim_exp_tens_raw is deprecated. The same call signature is "
        "now supported directly by cos_sim_exp_tens (pass raw arrays as the "
        "first arguments instead of pre-built density objects). This shim "
        "will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    return cos_sim_exp_tens(
        p1, w1, p2, w2, *args,
        method=method,
        cancellation_threshold=cancellation_threshold,
        verbose=verbose,
    )


def _compute_Q(D, r, is_rel, is_per, period):
    """Compute the quadratic form from (already-wrapped) differences D.

    When *is_rel* and *is_per* are both True, pairwise differences
    between components of D are wrapped to ``[-period/2, period/2)``
    before squaring. This restores exact transposition invariance on
    the circle, which is otherwise broken by component-wise wrapping.

    The two formulas are algebraically identical in the non-periodic
    case: ``sum_{i<j} (d_i - d_j)^2 == r * (sum(d^2) - sum(d)^2/r)``.
    """
    if is_rel:
        if is_per:
            Q = np.zeros(D.shape[1:])
            for i in range(r):
                for j in range(i + 1, r):
                    delta = D[i] - D[j]
                    delta = np.mod(delta + period / 2, period) - period / 2
                    Q += delta**2
            Q = Q / r
        else:
            Q = np.sum(D**2, axis=0) - np.sum(D, axis=0) ** 2 / r
    else:
        Q = np.sum(D**2, axis=0)
    return Q


def _ip_core(U, wU, nJ, V, wV, nK, r, sigma, is_rel, is_per, period,
             truncation_sigmas=None, kernel_precision=None):
    """Core inner product (perm-side × comb-side).

    Two-axis routing (mirrors :func:`_eval_exp_tens_sa`):

    - **Routing axis** — abs and rel-non-periodic forms have a helper
      reduction (``sigma_eff = sigma * sqrt(2)``); rel+periodic does
      not yet and stays on the inline / chunked path.
    - **Execution axis** — even when the helper is available, route
      through it only when feature kwargs are explicitly requested
      (after resolving ``None`` against the global defaults). Default
      mode runs the ``_ip_full`` / chunked path inline, avoiding
      the helper's per-call argument validation overhead.

    This preserves the inline-direct cost profile for default-mode callers
    (e.g. ``cos_sim_exp_tens`` in per-pair tight loops) while
    enabling the helper's truncation / precision features whenever
    the user opts in.
    """
    # ---- Execution-axis decision: resolve defaults first ----
    # ``None`` means "consult global default", not "no feature".
    from ._defaults import get_default
    trunc_resolved = (
        truncation_sigmas if truncation_sigmas is not None
        else get_default("truncation_sigmas")
    )
    prec_resolved = (
        kernel_precision if kernel_precision is not None
        else get_default("kernel_precision")
    )
    use_default_kwargs = (
        (trunc_resolved is None or not np.isfinite(trunc_resolved))
        and (prec_resolved == "double")
    )

    can_use_helper = not (is_rel and is_per)

    # Fire the kernel-evaluation hint once per session when Bulger's IP
    # path is about to run with default kwargs. Bulger's path
    # forms a kernel-matrix-of-r-tuple-pairs that benefits from the
    # same truncation / single-precision controls as the centres path.
    if use_default_kwargs:
        from ._defaults import _maybe_show_kernel_eval_hint
        _maybe_show_kernel_eval_hint(
            effective_truncation_sigmas=float("inf"),
            effective_kernel_precision="double",
        )

    # Route through helper only when features are actually requested
    # AND the helper supports this quadratic form.
    if can_use_helper and not use_default_kwargs:
        return _ip_via_helper(
            U, wU, V, wV, r, sigma, is_rel, is_per, period,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
        )

    # Default-mode (or rel+per) path: inline / chunked.
    bytes_needed = (r + 2) * int(nJ) * int(nK) * 8
    mem_limit = 4_000_000_000

    if bytes_needed <= mem_limit:
        return _ip_full(U, wU, nJ, V, wV, nK, r, sigma, is_rel, is_per, period)

    chunk_size = max(1, int(mem_limit / ((r + 2) * int(nJ) * 8)))
    acc = np.zeros(nJ)
    for c in range(0, nK, chunk_size):
        c_end = min(c + chunk_size, nK)
        idx = slice(c, c_end)
        n_kc = c_end - c

        Dc = U[:, :, None] - V[:, idx][:, None, :]
        if is_per:
            Dc = np.mod(Dc + period / 2, period) - period / 2
        Qc = _compute_Q(Dc, r, is_rel, is_per, period)
        Ec = np.exp(-Qc / (4 * sigma**2))
        acc += Ec @ wV[idx]

    return float(wU @ acc)


def _ip_via_helper(U, wU, V, wV, r, sigma, is_rel, is_per, period,
                   truncation_sigmas=None, kernel_precision=None):
    """Route the centres-IP through :func:`gaussian_kernel_sum`.

    The helper computes ``g(q) = sum_j wJ(j) * exp(-Q(c_j - x_q) /
    (2 * sigma_eff^2))`` with ``sigma_eff = sigma * sqrt(2)``, so the
    kernel exponent matches the centres-IP's ``Q / (4 * sigma^2)``.
    The IP is then ``wU @ g``.

    Supports abs (per and non-per) and rel-non-periodic. The rel+per
    pairwise-wrap form is not yet supported by the helper.
    """
    kw = dict(is_rel=bool(is_rel), r=int(r),
              is_per=bool(is_per), period=float(period))
    if truncation_sigmas is not None:
        kw["truncation_sigmas"] = float(truncation_sigmas)
    if kernel_precision is not None:
        kw["kernel_precision"] = kernel_precision
    sigma_eff = float(sigma) * np.sqrt(2.0)
    g = gaussian_kernel_sum(V, wV.ravel(), U, sigma_eff, **kw)
    return float(np.asarray(g).ravel() @ wU.ravel())


def _ip_full(U, wU, nJ, V, wV, nK, r, sigma, is_rel, is_per, period):
    """Fully vectorized inner product."""
    D = U[:, :, None] - V[:, None, :]  # (r, nJ, nK)

    if is_per:
        D = np.mod(D + period / 2, period) - period / 2

    Q = _compute_Q(D, r, is_rel, is_per, period)

    E = np.exp(-Q / (4 * sigma**2))  # (nJ, nK)
    return float(wU @ (E @ wV))


# -------------------------------------------------------------------
#  Möbius method dispatcher (single-attribute path)
# -------------------------------------------------------------------
#
#  The Möbius method is layered — a partition-decomposition with
#  orbit collapse — on top of Bulger's existing ``_ip_core`` path
#  (the v1 / v2.1 decomposition). See ``v22_specification.md`` and
#  ``mpt/_mobius.py`` for the combinatorial details.
#
#  The user-facing ``cos_sim_exp_tens`` gains two keywords:
#
#    method='auto'   : dispatcher chooses the Möbius method or Bulger's
#                     method based on (r, n, mode, sigma/period).
#    method='bulger' : forces Bulger's method (the v1 / v2.1
#                     decomposition with periodic pairwise-wrap form;
#                     ``_ip_core``); this is the closed form of JMM
#                     Eq. 3.4 — the toolbox's defined value of
#                     the rel_per inner product, by definition. At
#                     sigma/P > 0.03 it differs from the alternative
#                     integration form computed by 'mobius' by an
#                     amount that grows as the periodic Theta-tail
#                     terms become non-negligible (see V22_DEV_LOG.md
#                     Issue 3 for details). Slower than the Möbius
#                     method at high r and large K, but correct
#                     across the full sigma/P range.
#    method='direct' : forces direct enumeration (no Möbius cancellation;
#                     useful for diagnosing near-zero cosines).
#                     In the single-attribute path, 'direct' coincides
#                     with 'bulger' (both route through ``_ip_core``);
#                     the distinction surfaces in later windowed paths.
#
#  cancellation_threshold = 1e-12 : when the Möbius method's cross
#    inner product falls below this fraction of sqrt(<A,A><B,B>), the
#    Möbius result may suffer from catastrophic alternating-sum
#    cancellation; in that case fall back to ``_ip_core`` (Bulger's
#    method). In typical use the guard never triggers; the cost is at
#    most one extra Bulger pass. Note: this guard inspects the cross
#    product only — corruption in the auto inner products (<A,A>,
#    <B,B>) propagates through the cosine denominator silently. The
#    K_a >= r_a + 2 margin in `_orbit_safe_for_precision` is the
#    primary protection against auto-IP cancellation; a runtime
#    cancellation diagnostic on auto IPs is on the roadmap
#    (see V22_DEV_LOG.md Issue 4).

_ORBIT_R_MAX_SHIPPED = 8  # orbit tables r=2..8 ship pre-built
_ORBIT_SIGMA_OVER_P_THRESHOLD = 0.03  # σ/P beyond which the periodic-relative Möbius method deviates
_ORBIT_K_MINUS_R_MIN = 2  # K_a >= r_a + this margin required for the Möbius method (precision guard)
# Rationale (May 2026 audit): the Möbius method expresses the
# distinct-r-tuple sum as a signed sum over set-partition orbits.
# When K_a is close to r_a, the expansion has very few orbit classes
# and the Möbius alternation can produce catastrophic cancellation in
# the auto-inner-products <T_X, T_X> and <T_Y, T_Y> (which are not
# protected by the cross-cancellation guard, since that guard only
# inspects |<T_X, T_Y>| / sqrt(<T_X,T_X><T_Y,T_Y>)). Empirical sweep
# (5 seeds × all four modes × r in {2..5}) shows: K = r usually
# catastrophic; K = r+1 typically OK but with marginal r=4,5 cases
# losing ~1e-6 precision; K >= r+2 reaches FP precision uniformly.
# This guard is conservative but cheap: realistic music applications
# have K >> r, so it almost never triggers.


def _orbit_safe_for_precision(r_vec, k_vec):
    """Return True if every attribute satisfies K_a >= r_a + margin.

    Used by both the SA and MA dispatchers to refuse the Möbius method
    when its Möbius cancellation could swamp the answer. See the
    `_ORBIT_K_MINUS_R_MIN` rationale block above.
    """
    r_arr = np.atleast_1d(np.asarray(r_vec, dtype=np.intp))
    k_arr = np.atleast_1d(np.asarray(k_vec, dtype=np.intp))
    return bool(np.all(k_arr - r_arr >= _ORBIT_K_MINUS_R_MIN))


def _select_sa_inner_product_method(r, n_max, is_rel, is_per,
                                    sigma_over_P, user_method,
                                    n_min=None):
    """Pick the inner-product path for the SA case.

    Parameters
    ----------
    r : int
        Tensor order.
    n_max : int
        max(n_x, n_y); the larger of the two source sizes (used for
        the small-problem cutoff).
    is_rel, is_per : bool
        Mode flags.
    sigma_over_P : float
        σ / period; ignored if not periodic.
    user_method : str
        One of 'auto', 'bulger', 'direct'. (Internal callers may also
        pass 'mobius' to force the Möbius method.)
    n_min : int, optional
        min(n_x, n_y); the smaller of the two source sizes. Used for
        the K-vs-r precision guard. Defaults to ``n_max`` (i.e., the
        guard is bypassed if the caller provides only n_max).

    Returns
    -------
    str
        One of 'mobius', 'bulger', 'direct'.
    """
    if user_method != 'auto':
        return user_method
    # r=1: the Möbius machinery is undefined for r<2 (single block, no
    # distinct-index structure); Bulger's method is trivially fast anyway.
    if r <= 1:
        return 'bulger'
    # r=2 with small n: Bulger's method dominates because the Möbius
    # method's overhead (4 orbit classes, numpy.einsum dispatch) exceeds the
    # kernel-matvec cost.
    if r == 2 and n_max <= 8:
        return 'bulger'
    # r > _ORBIT_R_MAX_SHIPPED: shipped orbit tables stop here. At
    # higher r the Möbius method still works correctly, but on first use
    # the table must be built from scratch (cost grows with B_r^2);
    # default to Bulger's method to avoid surprising users with a slow
    # first call. Users who explicitly want the Möbius method at higher r
    # can pass method='mobius'; the cost-preview helper in mobius will
    # print an estimate before the build begins.
    if r > _ORBIT_R_MAX_SHIPPED:
        return 'bulger'
    # K-vs-r precision guard. The Möbius method's auto-inner-products can
    # suffer catastrophic Möbius cancellation when the multiset size is
    # too close to r (see _ORBIT_K_MINUS_R_MIN block). The cross
    # cancellation guard at the call site does NOT catch this, since it
    # inspects only |<T_X,T_Y>|; corrupted <T_X,T_X> propagates silently
    # into the cosine denominator.
    n_for_guard = n_min if n_min is not None else n_max
    if not _orbit_safe_for_precision([r], [n_for_guard]):
        return 'bulger'
    # Periodic-relative beyond σ/P threshold: in this regime the Möbius
    # method computes the JMM Eq. 3.4 integral form, while Bulger's
    # method computes the single-nearest-image-wrap form.
    # The two diverge by O((σ/P)^∞) starting around σ/P ≈ 0.03. For
    # backward compatibility the toolbox treats Bulger's
    # pairwise-wrap form as canonical; the Möbius method is therefore
    # disabled above the threshold. Users who want the JMM-exact integral
    # explicitly may pass method='mobius'.
    if is_rel and is_per and sigma_over_P > _ORBIT_SIGMA_OVER_P_THRESHOLD:
        warnings.warn(
            f"σ/P = {sigma_over_P:.3f} exceeds the Möbius-method threshold "
            f"({_ORBIT_SIGMA_OVER_P_THRESHOLD}) for relative-periodic mode; "
            f"falling back to Bulger's method (the pairwise-wrap form). "
            f"Pass method='bulger' explicitly to silence this warning."
        )
        return 'bulger'
    return 'mobius'


def _select_sa_eval_method(r, K, n_q, is_rel, is_per, sigma_over_P,
                           user_method):
    """Pick the evaluation method for ``eval_exp_tens`` (SA case).

    The choice is between the centres-array path (build
    a ``(dim, n_j)`` centres tensor at ``build_exp_tens`` time, then
    evaluate as a vectorised Gaussian product against the queries) and
    the Möbius point evaluator (Möbius-decomposed sum over set
    partitions; ``O(B_r · r · K · n_q)`` per query independent of
    ``n_j``).

    Cost rule of thumb. The centres path scales as
    ``O(r · n_j · n_q)`` with ``n_j = K!/(K-r)!``, so it explodes at
    high r. The Möbius method replaces ``n_j`` with ``B_r · r · K``,
    where ``B_r`` is the Bell number of ``r`` (5 at r=3, 15 at r=4,
    52 at r=5, 203 at r=6). Crossover analysis (5 partitions × N work
    per partition vs N!/(N-r)!) shows the Möbius method is ~22× faster
    at r=3 N=20, ~100× at r=4. At r=2 the costs are comparable; the
    centres path is simpler and avoids partition-table dispatch
    overhead, so default to centres there.

    Precision guard. The Möbius method suffers catastrophic Möbius
    cancellation when ``K - r < 2`` (same regime as the IP path); fall
    back to centres.

    Convention guard. In periodic-relative mode at ``σ/P > 0.03``,
    ``eval_orbit_rel`` integrates the JMM Eq. 3.4 form while the
    centres path computes the single-nearest-image-wrap form.
    The two diverge at this regime; centres remains the canonical
    output for backward compatibility.

    Parameters
    ----------
    r : int
        Tensor order.
    K : int
        Number of source events (``len(p)``).
    n_q : int
        Number of query points.
    is_rel, is_per : bool
        Mode flags.
    sigma_over_P : float
        ``σ / period``; ignored if not periodic.
    user_method : str
        One of ``'auto'``, ``'centres'``, ``'mobius'``. Internal callers
        may also pass ``'direct'`` as a synonym for ``'centres'``.

    Returns
    -------
    str
        ``'mobius'`` or ``'centres'``.
    """
    if user_method in ('centres', 'direct'):
        return 'centres'
    if user_method == 'mobius':
        return 'mobius'
    if user_method != 'auto':
        raise ValueError(
            f"method must be 'auto', 'centres', or 'mobius'; got "
            f"{user_method!r}."
        )
    # r=1: the Möbius machinery reduces to the direct Σ_i w_i K_i sum
    # (one partition with μ=1). Centres path coincides; pick centres
    # for code simplicity.
    if r <= 1:
        return 'centres'
    # Relative mode: eval_orbit_rel performs u-grid quadrature with
    # N_u ~ max(64, P/σ * 10) per query. The per-query cost is
    # O(B_r · r · K · N_u), much larger than the centres path's
    # O(n_j) per query at typical σ/P (~0.025 → N_u ≈ 360, vs n_j
    # of 100s to 1000s for r in {3, 4}). The Möbius relative-mode
    # evaluator is only ever cheaper at very high r combined with very
    # large K and large σ — a corner case that's safer to route via
    # explicit method='mobius'. Default to centres for rel mode.
    if is_rel:
        return 'centres'
    # r=2 with small K: centres is competitive and avoids the
    # partition-table dispatch overhead.
    if r == 2 and K <= 8:
        return 'centres'
    # Beyond shipped orbit tables: the eval_orbit_* helpers use
    # set-partition machinery rather than orbit tables, so they work
    # at any r in principle, but we defer to centres for consistency
    # with the IP-path policy. At r > _ORBIT_R_MAX_SHIPPED the orbit
    # table would build on demand, which the cost-preview helper warns
    # about; the eval dispatcher prefers the always-fast centres path.
    if r > _ORBIT_R_MAX_SHIPPED:
        return 'centres'
    # K-vs-r precision guard. Without K - r >= 2 the Möbius method's
    # alternating sum can lose all significant digits.
    if not _orbit_safe_for_precision([r], [K]):
        return 'centres'
    return 'mobius'


# -----------------------------------------------------------------------
# Unified method-selection + time-estimate probe
#
# The probe-based dispatcher replaces the heuristic rule for the
# discretionary cases. Genuinely hard rules (correctness / feasibility)
# stay as rules; everything else is decided by timing both methods on a
# small probe and picking the faster. The probe time also produces the
# user-facing time estimate, so dispatcher and estimator share a single
# load-bearing measurement that auto-adapts to any future optimisation.
# -----------------------------------------------------------------------


def _format_time(t_sec: float) -> str:
    """Human-readable short form of a duration in seconds."""
    if t_sec < 1:
        return f"{t_sec * 1000:.0f} ms"
    if t_sec < 60:
        return f"{t_sec:.1f} s"
    if t_sec < 3600:
        return f"{t_sec / 60:.1f} min"
    return f"{t_sec / 3600:.1f} hr"


# Probing parameters.
_PROBE_MIN_N_Q = 200    # below this many queries, skip probing entirely
_PROBE_N = 50           # probe sample size
# Centres-path memory budget (bytes). The probe refuses to materialise
# the centres array if it would exceed this; the Möbius method is chosen instead.
_CENTRES_PROBE_MEM_BUDGET = 4 * 1024**3

# Above this r, the Möbius method becomes infeasible: B_r (Bell numbers)
# explodes from 115,975 at r=10 to 5x10^13 at r=20, and set-partition
# enumeration itself blows the Python recursion stack. r > this falls back
# to centres-only routing.
_ORBIT_R_MAX_FEASIBLE = 10

# Bell numbers up to r=10 (set partition counts). Used by the rel-mode
# pre-screen to estimate the Möbius relative-mode cost without enumerating
# partitions.
_BELL_NUMBERS = {
    1: 1, 2: 2, 3: 5, 4: 15, 5: 52, 6: 203, 7: 877,
    8: 4140, 9: 21147, 10: 115975,
}

# Pre-screen: if one method is favoured by more than this factor, skip
# probing entirely. Two pre-screens, one per mode:
#
#  - Rel-mode pre-screen: routes TO centres when centres clearly wins.
#    The Möbius relative-mode evaluator does u-grid quadrature with N_u
#    sub-evals per query, so its PROBE is expensive (a 50-query probe at
#    N_u=1000 is ~3 s); a generous margin here avoids unnecessary probe
#    overhead.
#  - Abs-mode pre-screen: routes TO the Möbius method when it clearly wins.
#    For abs mode, centres cost per query is K^r vs Möbius cost
#    B_r * r * K. The Möbius method wins by a factor K^(r-1) / (B_r * r);
#    for K=72 r=3 that's ~1000x. The tiny-workload shortcut would
#    otherwise force centres for n_q<200 even at these large K, so the
#    pre-screen must run BEFORE the tiny shortcut. Pattern-finding and
#    other common music-cog tasks legitimately use abs mode at large K.
#
# The dominance margins are conservative — probe still has the final
# word when the cost ratio is in the uncertain region.
_PRESCREEN_CENTRES_DOMINANCE = 10.0
_PRESCREEN_ORBIT_DOMINANCE = 10.0


def _estimate_centres_array_bytes(K: int, r: int, is_rel: bool) -> int:
    """Estimate the dominant centres-array allocation in bytes.

    Returns ``K!/(K-r)! * dim * 8`` where ``dim`` is the effective
    centres dimensionality (``r`` for abs, ``r-1`` for rel).
    """
    if K < r:
        return 0
    n_j = 1
    for k in range(K - r + 1, K + 1):
        n_j *= k
    dim = r - 1 if is_rel else r
    return n_j * max(dim, 1) * 8


def _probe_eval_path(
    dens: "ExpTensDensity",
    x_probe: np.ndarray,
    path: str,
    *,
    truncation_sigmas: float | None,
    kernel_precision: str | None,
) -> float:
    """Time a small slice of the real eval path. Returns seconds."""
    import time as _time
    t0 = _time.perf_counter()
    if path == "centres":
        _eval_exp_tens_sa_centres(
            dens, x_probe, x_probe.shape[1],
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
            verbose=False,
        )
    else:
        _eval_exp_tens_sa_orbit(
            dens, x_probe, x_probe.shape[1],
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
            verbose=False,
        )
    return _time.perf_counter() - t0


def _select_and_estimate_sa(
    dens: "ExpTensDensity",
    x: np.ndarray,
    n_q: int,
    *,
    method: str,
    truncation_sigmas: float | None,
    kernel_precision: str | None,
    verbose: bool,
) -> tuple[str, bool, float, str]:
    """Unified path selection + time estimate for SA eval_exp_tens.

    Hard rules decide first:
      1. user override → honour it.
      2. r <= 1 → centres (Möbius method mathematically degenerate).
      3. K - r < _ORBIT_K_MINUS_R_MIN → centres (orbit cancellation).
      4. centres-array memory > budget → Möbius method (centres infeasible).

    Everything else is decided by probing both paths on a small slice
    of queries and picking the faster. The probe time, extrapolated to
    the full workload, is the user-facing time estimate.

    Returns (chosen, probed, est_sec, routing_reason). routing_reason
    is a short string describing why the path was chosen (e.g.
    'r = 1', 'rel-mode pre-screen', 'probe'); the caller uses it to
    emit a dispatch message via :func:`_maybe_show_dispatch_msg`.
    """
    r = int(dens.r)
    K = int(dens.p.shape[0])
    is_rel = bool(dens.is_rel)

    # ---- Rule 1: user override ----
    if method in ("centres", "direct"):
        return "centres", False, 0.0, "user override"
    if method == "mobius":
        return "mobius", False, 0.0, "user override"
    if method != "auto":
        raise ValueError(
            f"method must be 'auto', 'centres', or 'mobius'; got {method!r}."
        )

    # ---- Rule 2: Möbius method degenerate at r <= 1 ----
    if r <= 1:
        return "centres", False, 0.0, f"r = {r}"

    # ---- Rule 3: Möbius cancellation guard ----
    if not _orbit_safe_for_precision([r], [K]):
        return "centres", False, 0.0, f"K - r = {K - r} < 2"

    # ---- Rule 4: centres memory budget ----
    centres_bytes = _estimate_centres_array_bytes(K, r, is_rel)
    if centres_bytes > _CENTRES_PROBE_MEM_BUDGET:
        # Centres infeasible. Orbit is the only candidate, but it has
        # its own r-limit (B_r explodes; r > ~10 is impractical).
        if r > _ORBIT_R_MAX_FEASIBLE:
            raise ValueError(
                f"eval_exp_tens: r={r} requires more than "
                f"{_CENTRES_PROBE_MEM_BUDGET // 1024**3} GB for the "
                f"centres array (K={K}), and the Möbius method is infeasible at "
                f"r > {_ORBIT_R_MAX_FEASIBLE} (B_r explodes). Reduce "
                f"r or check inputs."
            )
        return "mobius", False, 0.0, "centres memory budget exceeded"

    # ---- Abs-mode pre-screen: route TO the Möbius method when it clearly wins ----
    # For abs mode, centres cost per query is K^r (materialised density
    # has n_j = K^r tuples), and Möbius absolute-mode per-query cost is B_r * r * K
    # (sum over B_r partitions of K*m per block, summing to K*r per
    # partition). The ratio is K^(r-1) / (B_r * r); for K=72 r=3 it's
    # ~1000x, meaning the tiny-workload shortcut below would otherwise
    # force centres for n_q<200 even when the Möbius method is 1000x faster.
    #
    # This pre-screen must run BEFORE the tiny-workload shortcut so
    # large-K abs-mode workloads (common in pattern-finding and other
    # music-cog tasks at typical 24-72-partial harmonic templates) get
    # the cheap routing decision they deserve at any n_q.
    #
    # Probe still has the final word in the uncertain region; this only
    # fires when the Möbius method wins by a comfortable margin.
    if (not is_rel) and r >= 2 and r <= _ORBIT_R_MAX_FEASIBLE:
        B_r = _BELL_NUMBERS[r]
        centres_cost = float(K) ** r
        orbit_cost = float(B_r) * r * float(K)
        if orbit_cost * _PRESCREEN_ORBIT_DOMINANCE < centres_cost:
            return "mobius", False, 0.0, "abs-mode pre-screen"

    # ---- Shortcut: tiny workload, skip probing ----
    if n_q < _PROBE_MIN_N_Q:
        return "centres", False, 0.0, f"n_q = {n_q} < {_PROBE_MIN_N_Q}"

    # ---- Rel-mode pre-screen: route TO centres when centres clearly wins ----
    # The probe is robust but not free. For rel mode in particular,
    # the Möbius relative-mode evaluator does u-grid quadrature with N_u ≈ max(64, 10·P/σ)
    # sub-evals per query — its PROBE cost scales as
    # B_r · r · K · N_u · n_probe, which is prohibitive when N_u is
    # large. We pre-screen the cost ratio analytically and skip the
    # probe if centres clearly wins. The probe still has the final
    # word in the uncertain region.
    if is_rel and r >= 2:
        # Estimate N_u (the Möbius relative-mode u-grid size) using the same
        # formula eval_orbit_rel uses internally.
        sigma = float(dens.sigma)
        if dens.is_per:
            N_u_est = max(64, int(np.ceil(
                10.0 * float(dens.period) / sigma
            )))
        else:
            # Non-periodic u-grid: covers [p.min() - x.max() - 8σ,
            # p.max() - x.min() + 8σ]. Use the actual data extents.
            p_min = float(np.min(dens.p))
            p_max = float(np.max(dens.p))
            x_min_abs = float(np.min(x, initial=0.0))
            x_max_abs = float(np.max(x, initial=0.0))
            u_min = p_min - max(0.0, x_max_abs) - 8.0 * sigma
            u_max = p_max - min(0.0, x_min_abs) + 8.0 * sigma
            N_u_est = max(
                64,
                int(np.ceil(max(u_max - u_min, 1.0) / sigma * 10.0)),
            )
        B_r = _BELL_NUMBERS.get(r, 10 ** 9)
        centres_cost = float(K) ** (r - 1)
        orbit_cost = float(B_r) * r * N_u_est
        if centres_cost * _PRESCREEN_CENTRES_DOMINANCE < orbit_cost:
            return "centres", False, 0.0, "rel-mode pre-screen"

    # ---- Probe both paths ----
    # Warm the set-partition cache so the Möbius probe doesn't pay
    # one-time table-build cost. Skip for high r where the Möbius method is not a
    # realistic candidate — set-partition enumeration itself becomes
    # infeasible, and the recursion depth grows linearly in r.
    if 2 <= r <= _ORBIT_R_MAX_FEASIBLE:
        from ._mobius import get_set_partitions_with_mobius
        get_set_partitions_with_mobius(r)
    if r > _ORBIT_R_MAX_FEASIBLE:
        # No Möbius option at this r; skip the probe and use centres.
        return ("centres", False, 0.0,
                f"r = {r} > {_ORBIT_R_MAX_FEASIBLE} (Möbius infeasible)")

    n_probe = min(_PROBE_N, n_q)
    sample_idx = np.linspace(0, n_q - 1, n_probe).astype(int)
    x_probe = x[:, sample_idx]

    t_centres = _probe_eval_path(
        dens, x_probe, "centres",
        truncation_sigmas=truncation_sigmas,
        kernel_precision=kernel_precision,
    )
    t_orbit = _probe_eval_path(
        dens, x_probe, "mobius",
        truncation_sigmas=truncation_sigmas,
        kernel_precision=kernel_precision,
    )

    if t_centres <= t_orbit:
        chosen, t_probe = "centres", t_centres
    else:
        chosen, t_probe = "mobius", t_orbit

    est_sec = t_probe * (n_q / n_probe)
    return chosen, True, est_sec, "probe"


# -----------------------------------------------------------------------
# Probe-based dispatcher for the SA cos_sim_exp_tens IP path.
#
# Parallels :func:`_select_and_estimate_sa` for the inner-product side:
# hard rules first (correctness / feasibility), then an analytical
# pre-screen (catches clear-winner cases without paying probe overhead),
# then actually time both paths on a small subset of each density and
# pick the faster. The probe time, extrapolated to the full workload,
# becomes the user-facing time estimate (printed in verbose mode) and
# auto-adapts to future optimisations of either path.
#
# Probe extrapolation. Pairwise IP cost scales as
# ``falling_factorial(K_x, r) * falling_factorial(K_y, r)`` (ordered
# r-tuple enumeration on each side). Orbit IP cost scales as
# ``B_r * K_x * K_y`` (kernel matrix construction + per-partition
# einsum). The probe uses ``K_probe = min(K_x, K_y, _PROBE_K_IP_TARGET)``
# events from each side and extrapolates by the appropriate factor.
# -----------------------------------------------------------------------

# Target subset size for the IP probe. Small enough that probe cost is
# negligible, large enough that the K_probe-choose-r tuple count is
# meaningful (e.g., 12-choose-3 = 220) and the Möbius method's precision
# guard (n_min - r >= 2) is not contended. ``K_probe`` is capped to
# ``min(K_x, K_y)`` at call time; the hard precision rule
# (``n_min - r < 2``) fires upstream so K_probe never drops below r+2.
_PROBE_K_IP_TARGET = 12

# Pre-screen: skip the probe if one path's analytical cost dominates
# the other by this margin. Mirrors the eval-side pre-screen
# constants.
_PRESCREEN_IP_DOMINANCE = 10.0


def _falling_factorial(n: int, k: int) -> float:
    """``n * (n-1) * ... * (n-k+1)``; 0 if any factor is non-positive."""
    if n < k:
        return 0.0
    prod = 1.0
    for i in range(k):
        prod *= (n - i)
    return prod


def _probe_ip_path(
    dens_x: "ExpTensDensity",
    dens_y: "ExpTensDensity",
    K_probe: int,
    path: str,
    *,
    truncation_sigmas: float | None,
    kernel_precision: str | None,
) -> float:
    """Time one cos_sim_exp_tens IP path on the first ``K_probe`` events
    of each density. Returns seconds.

    Builds fresh subset densities outside the timed window so the
    measurement covers only the IP work itself (kernel-matrix
    construction + einsums for the Möbius method, or ordered-tuple
    enumeration + dot product for Bulger's method).
    """
    import time as _time

    sub_x = build_exp_tens(
        dens_x.p[:K_probe], dens_x.w[:K_probe],
        dens_x.sigma, int(dens_x.r),
        bool(dens_x.is_rel), bool(dens_x.is_per), float(dens_x.period),
        verbose=False,
    )
    sub_y = build_exp_tens(
        dens_y.p[:K_probe], dens_y.w[:K_probe],
        dens_y.sigma, int(dens_y.r),
        bool(dens_y.is_rel), bool(dens_y.is_per), float(dens_y.period),
        verbose=False,
    )

    t0 = _time.perf_counter()
    if path == "mobius":
        _cos_sim_exp_tens_sa_orbit(sub_x, sub_y)
    else:
        _cos_sim_exp_tens_sa_pairwise(
            sub_x, sub_y, verbose=False,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
        )
    return _time.perf_counter() - t0


def _select_and_estimate_sa_ip(
    dens_x: "ExpTensDensity",
    dens_y: "ExpTensDensity",
    *,
    method: str,
    truncation_sigmas: float | None,
    kernel_precision: str | None,
    verbose: bool,
) -> tuple[str, bool, float, str]:
    """Probe-based dispatcher for SA cos_sim_exp_tens IP path.

    Hard rules decide first:
      1. user override → honour it.
      2. r <= 1 → Bulger (Möbius method degenerate at r=1).
      3. r > _ORBIT_R_MAX_SHIPPED → pairwise (build cost).
      4. n_min - r < _ORBIT_K_MINUS_R_MIN → pairwise (orbit cancellation).
      5. periodic-relative beyond σ/P threshold → Bulger (convention).

    Then analytical pre-screen catches clear-winner cases without
    paying probe overhead. Otherwise, both paths are timed on a small
    subset (``min(K_x, K_y, _PROBE_K_IP_TARGET)``) and extrapolated to
    the full workload; the faster is picked.

    Returns ``(chosen, probed, est_sec, routing_reason)``. routing_reason
    is a short string describing why the path was chosen; the caller
    uses it to emit a dispatch message via
    :func:`_maybe_show_dispatch_msg`.
    """
    r = int(dens_x.r)
    K_x = int(dens_x.p.shape[0])
    K_y = int(dens_y.p.shape[0])
    n_min = min(K_x, K_y)
    is_rel = bool(dens_x.is_rel)
    is_per = bool(dens_x.is_per)
    sigma = float(dens_x.sigma)
    period = float(dens_x.period)
    sigma_over_P = sigma / period if (is_per and period > 0) else 0.0

    # ---- Hard rules ----
    if method in ("bulger", "direct"):
        return "bulger", False, 0.0, "user override"
    if method == "mobius":
        return "mobius", False, 0.0, "user override"
    if method != "auto":
        raise ValueError(
            f"method must be 'auto', 'bulger', 'direct', or 'mobius'; "
            f"got {method!r}."
        )
    if r <= 1:
        return "bulger", False, 0.0, f"r = {r}"
    if r > _ORBIT_R_MAX_SHIPPED:
        return "bulger", False, 0.0, f"r = {r} > {_ORBIT_R_MAX_SHIPPED} (Möbius infeasible)"
    if not _orbit_safe_for_precision([r], [n_min]):
        return "bulger", False, 0.0, f"min(K_x, K_y) - r = {n_min - r} < 2"
    if is_rel and is_per and sigma_over_P > _ORBIT_SIGMA_OVER_P_THRESHOLD:
        warnings.warn(
            f"σ/P = {sigma_over_P:.3f} exceeds the Möbius-method threshold "
            f"({_ORBIT_SIGMA_OVER_P_THRESHOLD}) for relative-periodic mode; "
            f"falling back to Bulger's method (the pairwise-wrap form). Pass method='bulger' "
            f"explicitly to silence this warning."
        )
        return "bulger", False, 0.0, "sigma/period > 0.03 (rel-per Möbius fallback)"
    if r > _ORBIT_R_MAX_FEASIBLE:
        return "bulger", False, 0.0, f"r = {r} > {_ORBIT_R_MAX_FEASIBLE} (Möbius infeasible)"

    # ---- Analytical cost models ----
    pairwise_full = _falling_factorial(K_x, r) * _falling_factorial(K_y, r)
    B_r = float(_BELL_NUMBERS[r])
    orbit_full = B_r * float(K_x) * float(K_y)

    # ---- Analytical pre-screen ----
    if orbit_full * _PRESCREEN_IP_DOMINANCE < pairwise_full:
        return "mobius", False, 0.0, "cost pre-screen"
    if pairwise_full * _PRESCREEN_IP_DOMINANCE < orbit_full:
        return "bulger", False, 0.0, "cost pre-screen"

    # ---- Probe both paths on a subset ----
    # Warm the orbit partition table so the Möbius probe doesn't pay a
    # one-time table-build cost.
    from ._mobius import get_set_partitions_with_mobius
    get_set_partitions_with_mobius(r)

    K_probe = min(K_x, K_y, _PROBE_K_IP_TARGET)
    # K_probe - r >= 2 is guaranteed by the precision hard rule above
    # (n_min - r >= _ORBIT_K_MINUS_R_MIN), so the orbit probe is safe.

    t_pairwise = _probe_ip_path(
        dens_x, dens_y, K_probe, "bulger",
        truncation_sigmas=truncation_sigmas,
        kernel_precision=kernel_precision,
    )
    t_orbit = _probe_ip_path(
        dens_x, dens_y, K_probe, "mobius",
        truncation_sigmas=truncation_sigmas,
        kernel_precision=kernel_precision,
    )

    # ---- Extrapolate to full workload ----
    pairwise_probe = _falling_factorial(K_probe, r) ** 2
    pairwise_factor = (
        pairwise_full / pairwise_probe if pairwise_probe > 0 else 1.0
    )
    orbit_probe = float(K_probe) ** 2
    orbit_factor = (
        float(K_x) * float(K_y) / orbit_probe if orbit_probe > 0 else 1.0
    )

    t_pairwise_est = t_pairwise * pairwise_factor
    t_orbit_est = t_orbit * orbit_factor

    if t_pairwise_est <= t_orbit_est:
        return "bulger", True, t_pairwise_est, "probe"
    return "mobius", True, t_orbit_est, "probe"


def _orbit_inner_abs(p_a, w_a, p_b, w_b, sigma, r, is_per, period,
                     *, return_cancellation_ratio=False):
    """<T_A, T_B> in absolute mode via the Möbius method.

    With ``return_cancellation_ratio=True``, returns ``(value, ratio)``
    where ratio is ``|sum| / max(|term|)`` from the Möbius alternating
    partition sum (1.0 means no cancellation; <<1 means digits lost). See
    :func:`mpt._mobius.inner_product_orbit` for full semantics.
    """
    from ._mobius import inner_product_orbit

    diffs = p_a[:, None] - p_b[None, :]
    if is_per:
        diffs = diffs - period * np.floor(diffs / period + 0.5)
    K = np.exp(-(diffs ** 2) / (4 * sigma ** 2))
    return inner_product_orbit(
        K, w_a, w_b, r, prefactor=(sigma * np.sqrt(np.pi)) ** r,
        return_cancellation_ratio=return_cancellation_ratio,
    )


def _inner_product_direct_abs_sa(p_x, w_x, p_y, w_y, sigma, r,
                                   is_per, period):
    """<T_X, T_Y> in absolute mode via direct r-tuple enumeration.

    Computes the SA inner product
        <T_X, T_Y> = (sigma * sqrt(pi))**r *
                     sum_{J, K} wJ_x[J] * wJ_y[K] *
                                exp(-||centres_x[:, J] - centres_y[:, K]||^2
                                    / (4 sigma^2))
    by enumerating ordered r-tuples on each side. No Möbius
    alternating sum is involved, so the result is exact (no
    catastrophic cancellation) for any K_x, K_y >= r. This is the
    "unsafe" path of the MA per-attribute IP matrix, used for event
    pairs where at least one event has K_eff - r below the
    Möbius-method precision margin (`_ORBIT_K_MINUS_R_MIN` = 2).

    NaN tolerance: NaN entries in ``p_x`` / ``w_x`` / ``p_y`` / ``w_y``
    are dropped per side before enumeration. If the dropped count
    leaves either side with fewer than r valid slots, returns 0 by
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
    U_x, wJ_x = _build_ordered_r_tuples(p_x, w_x, r)   # (r, nJ_x), (nJ_x,)
    U_y, wJ_y = _build_ordered_r_tuples(p_y, w_y, r)
    nJ_x = U_x.shape[1]
    nJ_y = U_y.shape[1]

    diffs = U_x[:, :, None] - U_y[:, None, :]   # (r, nJ_x, nJ_y)
    if is_per:
        diffs = diffs - period * np.floor(diffs / period + 0.5)
    Q = np.sum(diffs ** 2, axis=0)              # (nJ_x, nJ_y)
    K_mat = np.exp(-Q / (4 * sigma ** 2))

    return float((sigma * np.sqrt(np.pi)) ** r *
                 np.einsum('i,ij,j->', wJ_x, K_mat, wJ_y))


def _build_ordered_r_tuples(p, w, r):
    """Mirror of :class:`SAExpTensDensity` ordered-tuple construction.

    Returns ``(U, wJ)`` where ``U`` is ``(r, nJ)`` of position values
    along ordered r-tuples and ``wJ`` is ``(nJ,)`` of weight products.
    Used by :func:`_inner_product_direct_abs_sa` and any other helper
    that needs single-event ordered tuples without going through the
    full :func:`build_exp_tens` API.
    """
    import math
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


def _orbit_inner_rel(p_a, w_a, p_b, w_b, sigma, r, is_per, period,
                     samples_per_sigma=10, *,
                     return_cancellation_ratio=False):
    """<T_A, T_B> in relative mode via Möbius machinery + translation grid.

    Marginalises a translation u over either ``[0, P)`` (periodic) or a
    Gaussian-supported window around the alignment of A and B
    (non-periodic), and integrates the Möbius-evaluated kernel against u.
    The grid density is ``samples_per_sigma`` points per σ; the
    truncation in the non-periodic case extends 8σ beyond the natural
    overlap window. (The earlier 4σ default truncated tails of the
    Möbius integrand at ~5e-10 — small per kernel value, but
    enough to corrupt the auto-inner products at ~1e-6 relative
    precision once Möbius cancellation amplified them. 8σ pushes the
    truncation tail to FP noise; 12σ is empirically no improvement.)

    With ``return_cancellation_ratio=True``, returns ``(value, ratio)``
    where ratio is the worst-case (minimum) cancellation ratio across
    the u-grid points. A point with severe cancellation drives the
    integration result toward catastrophic loss of significance.
    """
    from ._mobius import inner_product_orbit_grid

    if is_per:
        N_u = max(64, int(np.ceil(period / sigma * samples_per_sigma)))
        u_grid = np.linspace(0.0, period, N_u, endpoint=False)
        du = period / N_u
        diffs = (p_a[None, :, None] - p_b[None, None, :]
                 + u_grid[:, None, None])
        diffs = diffs - period * np.floor(diffs / period + 0.5)
    else:
        u_min = p_b.min() - p_a.max() - 8.0 * sigma
        u_max = p_b.max() - p_a.min() + 8.0 * sigma
        N_u = max(
            64,
            int(np.ceil(max(u_max - u_min, 1.0) / sigma * samples_per_sigma)),
        )
        u_grid = np.linspace(u_min, u_max, N_u)
        diffs = (p_a[None, :, None] - p_b[None, None, :]
                 + u_grid[:, None, None])
    K_u = np.exp(-(diffs ** 2) / (4 * sigma ** 2))
    if return_cancellation_ratio:
        F, ratios = inner_product_orbit_grid(
            K_u, w_a, w_b, r, return_cancellation_ratio=True,
        )
    else:
        F = inner_product_orbit_grid(K_u, w_a, w_b, r)
    if is_per:
        integral = float(F.sum() * du)
    else:
        integral = float(np.trapezoid(F, u_grid))
    c = sigma * np.sqrt(2 * np.pi / r)
    value = (sigma * np.sqrt(np.pi)) ** r * integral / c ** 2
    if return_cancellation_ratio:
        # Worst case across u-grid is the relevant signal — even one
        # bad point could dominate the integration if it sits near a
        # peak of the integrand.
        ratio = float(np.min(ratios))
        return value, ratio
    return value


def _cos_sim_exp_tens_sa_orbit(dens_x, dens_y):
    """Compute (ip_xy, ip_xx, ip_yy, worst_ratio) for the SA case via
    Möbius method.

    ``worst_ratio`` is the minimum cancellation ratio across the three
    inner-product computations. Values below ~1e-10 indicate the
    Möbius alternating sum has lost most of its significant digits and
    the dispatcher should fall back to Bulger's method.
    """
    sigma = dens_x.sigma
    r = dens_x.r
    is_rel = dens_x.is_rel
    is_per = dens_x.is_per
    period = dens_x.period
    p_x, w_x = dens_x.p, dens_x.w
    p_y, w_y = dens_y.p, dens_y.w

    if is_rel:
        ip_xy, r_xy = _orbit_inner_rel(
            p_x, w_x, p_y, w_y, sigma, r, is_per, period,
            return_cancellation_ratio=True,
        )
        ip_xx, r_xx = _orbit_inner_rel(
            p_x, w_x, p_x, w_x, sigma, r, is_per, period,
            return_cancellation_ratio=True,
        )
        ip_yy, r_yy = _orbit_inner_rel(
            p_y, w_y, p_y, w_y, sigma, r, is_per, period,
            return_cancellation_ratio=True,
        )
    else:
        ip_xy, r_xy = _orbit_inner_abs(
            p_x, w_x, p_y, w_y, sigma, r, is_per, period,
            return_cancellation_ratio=True,
        )
        ip_xx, r_xx = _orbit_inner_abs(
            p_x, w_x, p_x, w_x, sigma, r, is_per, period,
            return_cancellation_ratio=True,
        )
        ip_yy, r_yy = _orbit_inner_abs(
            p_y, w_y, p_y, w_y, sigma, r, is_per, period,
            return_cancellation_ratio=True,
        )
    return ip_xy, ip_xx, ip_yy, min(r_xy, r_xx, r_yy)


def _cos_sim_exp_tens_sa_pairwise(dens_x, dens_y, *, verbose: bool = True,
                                  truncation_sigmas=None,
                                  kernel_precision=None):
    """Compute (ip_xy, ip_xx, ip_yy) for the SA case via the
    Bulger's method (``_ip_core``).

    This is the body of the original ``_cos_sim_exp_tens_sa``
    factored out so the new dispatcher can route to it cleanly.

    Forwards ``truncation_sigmas`` / ``kernel_precision`` to
    ``_ip_core`` so the helper-accelerated path is reached for the
    abs and rel-non-periodic modes.
    """
    r = dens_x.r
    sigma = dens_x.sigma
    is_rel = dens_x.is_rel
    is_per = dens_x.is_per
    period = dens_x.period

    n_jx, n_kx = dens_x.n_j_perm, dens_x.n_k
    n_jy, n_ky = dens_y.n_j_perm, dens_y.n_k

    total_pairs = n_jx * n_ky + n_jx * n_kx + n_jy * n_ky
    estimate_comp_time(total_pairs, r, "cos_sim_exp_tens", verbose)

    ip_xy = _ip_core(
        dens_x.u_perm, dens_x.w_perm, n_jx,
        dens_y.v_comb, dens_y.wv_comb, n_ky,
        r, sigma, is_rel, is_per, period,
        truncation_sigmas=truncation_sigmas,
        kernel_precision=kernel_precision,
    )
    ip_xx = _ip_core(
        dens_x.u_perm, dens_x.w_perm, n_jx,
        dens_x.v_comb, dens_x.wv_comb, n_kx,
        r, sigma, is_rel, is_per, period,
        truncation_sigmas=truncation_sigmas,
        kernel_precision=kernel_precision,
    )
    ip_yy = _ip_core(
        dens_y.u_perm, dens_y.w_perm, n_jy,
        dens_y.v_comb, dens_y.wv_comb, n_ky,
        r, sigma, is_rel, is_per, period,
        truncation_sigmas=truncation_sigmas,
        kernel_precision=kernel_precision,
    )
    return ip_xy, ip_xx, ip_yy


# -------------------------------------------------------------------
#  Canonicalization helpers (for batch_cos_sim_exp_tens)
# -------------------------------------------------------------------


def _lex_compare(a: tuple, b: tuple) -> int:
    """Lexicographic comparison. Returns -1, 0, or +1."""
    for ai, bi in zip(a, b):
        if ai < bi:
            return -1
        if ai > bi:
            return 1
    return 0


def _cyclic_canonical(
    p_sorted: np.ndarray,
    w_sorted: np.ndarray | None,
    period: float,
) -> tuple[tuple, tuple | None, float]:
    """Lexicographically smallest rotation of a periodic pitch set.

    Tries all n rotations (subtract each sorted pitch, mod period,
    re-sort with weights) and returns the lex-smallest form plus
    the shift that produced it. This captures all
    transposition-modulo-period equivalences.

    Pitch values are rounded to 9 decimal places before lex
    comparison and in the returned tuple, to absorb floating-point
    noise from mod-reduction. 9 decimals is below any musically-
    meaningful precision (1 attocent / 1 nanosecond) but well above
    typical FP roundoff. Without this rounding, two
    transposition-equivalent multisets with different FP error
    patterns can produce different canonical forms — a real bug
    that breaks consumer-level dedup for non-integer pitch data.

    Returns
    -------
    best_p : tuple
        Canonical pitch tuple (rounded to 9 decimals).
    best_w : tuple or None
        Canonical weight tuple (if weights provided).
    best_shift : float
        The pitch value subtracted to produce the canonical form
        (returned at full precision; only the canonical *form* is
        rounded, not the shift itself, so callers using the shift
        to apply to a paired set get exact arithmetic).
    """
    n = len(p_sorted)
    has_w = w_sorted is not None
    ROUND_DIGITS = 9

    best_p = tuple(np.round(p_sorted - p_sorted[0], ROUND_DIGITS))
    best_w = tuple(w_sorted) if has_w else None
    best_shift = p_sorted[0]

    for rot in range(1, n):
        shifted = np.mod(p_sorted - p_sorted[rot], period)
        si = np.argsort(shifted)
        shifted = np.round(shifted[si], ROUND_DIGITS)
        t_p = tuple(shifted)

        cmp = _lex_compare(t_p, best_p)
        if cmp < 0:
            best_p = t_p
            best_w = tuple(w_sorted[si]) if has_w else None
            best_shift = p_sorted[rot]
        elif cmp == 0 and has_w:
            t_w = tuple(w_sorted[si])
            if _lex_compare(t_w, best_w) < 0:
                best_w = t_w
                best_shift = p_sorted[rot]

    return best_p, best_w, best_shift


def _canonicalize_set(
    p: np.ndarray,
    w: np.ndarray | None,
    is_rel: bool,
    is_per: bool,
    period: float,
) -> tuple[tuple, tuple | None]:
    """Canonical form of a pitch/weight set under isPer/isRel.

    Returns hashable tuples suitable for use as dict keys.
    """
    has_w = w is not None

    # Sort pitches, align weights
    si = np.argsort(p)
    p = p[si]
    if has_w:
        w = w[si]

    # Reduce modulo period
    if is_per:
        p = np.mod(p, period)
        si = np.argsort(p)
        p = p[si]
        if has_w:
            w = w[si]

    # Remove transposition
    if is_rel:
        if is_per:
            # Cyclic canonical form: the lex-smallest rotation captures
            # all transposition-modulo-period equivalences.
            ca_p, ca_w, _ = _cyclic_canonical(p, w, period)
            return ca_p, ca_w
        else:
            p = p - p[0]

    return tuple(p), (tuple(w) if has_w else None)


# -------------------------------------------------------------------
#  Canonical-key primitives for batched / deduplicated workflows
# -------------------------------------------------------------------


def _chord_canonical_key(
    p,
    w,
    *,
    sigma: float,
    r: int,
    is_rel: bool,
    is_per: bool,
    period: float,
    precision: int | None = None,
):
    """Canonical hashable form of a single weighted multiset.

    Two chords ``(p1, w1)`` and ``(p2, w2)`` produce the same key iff
    their resulting density object is structurally identical (same
    ``(p, w, sigma, r, is_rel, is_per, period)``-determined density),
    regardless of input-side permutation or, in relative modes,
    in-batch translation. Used by the consumer-level deduplication in
    :func:`cos_sim_exp_tens`, :func:`windowed_similarity`, and the
    harmony wrappers when given batched chord input.

    Parameters
    ----------
    p : array-like
        Pitch values (will be flattened; NaN handling is the caller's
        responsibility — pass NaN-stripped arrays).
    w : array-like or None
        Weights, same length as ``p``. None means uniform weights.
    sigma, r, is_rel, is_per, period :
        Density-determining parameters. Baked into the returned key
        so different parameter settings produce different keys.
    precision : int, optional
        Round the canonical pitch and weight values to this many
        decimal places (after the canonicalisation, to absorb FP noise
        from mod-reduction and subtraction). Default: no rounding.

    Returns
    -------
    key : tuple
        Hashable canonical form, suitable as a dict key.
    p_canon : np.ndarray, dtype=float64
        Canonical pitch array (for downstream density caching).
    w_canon : np.ndarray | None
        Canonical weight array, or None if ``w`` is None.
    """
    p_arr = np.asarray(p, dtype=np.float64)
    w_arr = np.asarray(w, dtype=np.float64) if w is not None else None

    ca_p, ca_w = _canonicalize_set(p_arr, w_arr, is_rel, is_per, period)

    if precision is not None:
        ca_p = tuple(round(x, precision) for x in ca_p)
        if ca_w is not None:
            ca_w = tuple(round(x, precision) for x in ca_w)

    key = (ca_p, ca_w, sigma, r, is_rel, is_per, period)

    p_canon = np.array(ca_p, dtype=np.float64)
    w_canon = np.array(ca_w, dtype=np.float64) if ca_w is not None else None

    return key, p_canon, w_canon


def _pair_canonical_key(
    p_a,
    w_a,
    p_b,
    w_b,
    *,
    sigma: float,
    r: int,
    is_rel: bool,
    is_per: bool,
    period: float,
    precision: int | None = None,
):
    """Canonical hashable forms of a paired weighted multiset (A, B).

    The cosine similarity of two expectation tensor densities is
    invariant under certain joint transformations of the pair. The
    canonical pair form encodes these symmetries so that
    structurally-equivalent pairs produce the same key, enabling
    deduplication.

    The exploited symmetries depend on the mode:

    - **Relative** (``is_rel=True``): independent transposition of
      each set. Each side is canonicalised separately via
      :func:`_canonicalize_set`.
    - **Absolute** (``is_rel=False``): joint co-transposition.
      ``cos_sim_exp_tens(A + c, B + c) == cos_sim_exp_tens(A, B)``,
      so A's canonical form determines a shift, and the same shift
      is applied to B. For ``is_per=True``, A is reduced to its
      cyclic canonical form (the lex-smallest rotation), and B is
      shifted by the corresponding amount mod period; for
      ``is_per=False``, A is translated so its minimum is at 0, and
      B is shifted by the same amount.

    Parameters
    ----------
    p_a, p_b : array-like
        Pitch values for A and B (NaN-stripped).
    w_a, w_b : array-like or None
        Weights for A and B, or None for uniform.
    sigma, r, is_rel, is_per, period :
        Density-determining parameters. Baked into both returned keys.
    precision : int, optional
        Post-canonicalisation rounding. Default: no rounding.

    Returns
    -------
    key_a, key_b : tuple
        Hashable canonical keys for A and B. The pair key is
        ``(key_a, key_b)``; ``key_a`` and ``key_b`` are also valid as
        single-side dedup keys for caching A and B densities
        respectively.
    p_a_canon, p_b_canon : np.ndarray, dtype=float64
        Canonical pitch arrays.
    w_a_canon, w_b_canon : np.ndarray | None
        Canonical weight arrays, or None if the corresponding input
        weight was None.
    """
    pa_arr = np.asarray(p_a, dtype=np.float64)
    pb_arr = np.asarray(p_b, dtype=np.float64)
    wa_arr = np.asarray(w_a, dtype=np.float64) if w_a is not None else None
    wb_arr = np.asarray(w_b, dtype=np.float64) if w_b is not None else None

    if is_rel:
        # Independent canonicalisation per side.
        ca_p, ca_w = _canonicalize_set(pa_arr, wa_arr, is_rel, is_per, period)
        cb_p, cb_w = _canonicalize_set(pb_arr, wb_arr, is_rel, is_per, period)
    else:
        # Joint co-transposition: A determines the shift, B inherits it.
        si_a = np.argsort(pa_arr)
        pa_s = pa_arr[si_a]
        wa_s = wa_arr[si_a] if wa_arr is not None else None

        if is_per:
            pa_s = np.mod(pa_s, period)
            si = np.argsort(pa_s)
            pa_s = pa_s[si]
            if wa_s is not None:
                wa_s = wa_s[si]
            # Cyclic canonical form — collapses all rotations.
            ca_p, ca_w, shift = _cyclic_canonical(pa_s, wa_s, period)
        else:
            shift = pa_s[0]
            ca_p = tuple(pa_s - shift)
            ca_w = tuple(wa_s) if wa_s is not None else None

        # Apply the same shift to B.
        si_b = np.argsort(pb_arr)
        pb_s = pb_arr[si_b]
        wb_s = wb_arr[si_b] if wb_arr is not None else None

        if is_per:
            pb_shifted = np.mod(pb_s - shift, period)
            si = np.argsort(pb_shifted)
            cb_p = tuple(pb_shifted[si])
            cb_w = tuple(wb_s[si]) if wb_s is not None else None
        else:
            cb_p = tuple(pb_s - shift)
            cb_w = tuple(wb_s) if wb_s is not None else None

    if precision is not None:
        ca_p = tuple(round(x, precision) for x in ca_p)
        cb_p = tuple(round(x, precision) for x in cb_p)
        if ca_w is not None:
            ca_w = tuple(round(x, precision) for x in ca_w)
        if cb_w is not None:
            cb_w = tuple(round(x, precision) for x in cb_w)

    key_a = (ca_p, ca_w, sigma, r, is_rel, is_per, period)
    key_b = (cb_p, cb_w, sigma, r, is_rel, is_per, period)

    p_a_canon = np.array(ca_p, dtype=np.float64)
    p_b_canon = np.array(cb_p, dtype=np.float64)
    w_a_canon = np.array(ca_w, dtype=np.float64) if ca_w is not None else None
    w_b_canon = np.array(cb_w, dtype=np.float64) if cb_w is not None else None

    return key_a, key_b, p_a_canon, w_a_canon, p_b_canon, w_b_canon


# -------------------------------------------------------------------
#  Raw SA batched dispatch (used by the polymorphic cos_sim_exp_tens
#  for 2-D pitch-matrix input)
# -------------------------------------------------------------------


def _cos_sim_raw_sa_batch(
    p_mat_a: np.ndarray,
    p_mat_b: np.ndarray,
    sigma: float,
    r: int,
    is_rel: bool,
    is_per: bool,
    period: float,
    *,
    weights_a: np.ndarray | None = None,
    weights_b: np.ndarray | None = None,
    spectrum: list | None = None,
    precision: int | None = None,
    dedup: bool = True,
    method: str = "auto",
    cancellation_threshold: float = 1e-12,
    verbose: bool = True,
) -> np.ndarray:
    """Raw single-attribute batched dispatch for :func:`cos_sim_exp_tens`.

    Computes cosine similarity for many paired weighted multisets
    (*p* represents pitches or positions). Each row of *p_mat_a*
    and *p_mat_b* defines one pair.

    Two-stage dedup: Phase 1 builds canonical-form keys per side and
    constructs each unique density once (chord-level dedup); Phase 3
    delegates pair-level dedup to the polymorphic
    :func:`cos_sim_exp_tens` (in pairwise list-vs-list mode), which
    in turn threads ``method`` and ``cancellation_threshold`` through
    to the per-pair Möbius-vs-Bulger dispatcher.

    Parameters
    ----------
    p_mat_a, p_mat_b : 2-D arrays
        Multiset values per row. NaN entries are ignored.
    sigma, r, is_rel, is_per, period :
        Tensor parameters.
    weights_a, weights_b : 2-D arrays or None
        Weights matching the corresponding ``p_mat_*``.
    spectrum : list, optional
        Arguments for :func:`~mpt.spectra.add_spectra`.
    precision : int, optional
        Round canonical pitch and weight values to this many decimal
        places, to absorb FP noise when deduplicating.
    dedup : bool, default True
        Apply pair-level canonical-form dedup at Phase 3.
    method : {'auto', 'bulger', 'direct'}, default 'auto'
        Inner-product evaluation path; threaded through to the per-pair
        SA core via the inner ``cos_sim_exp_tens`` call.
    cancellation_threshold : float, default 1e-12
        Orbit-path cancellation guard threshold.
    verbose : bool
        Print progress.

    Returns
    -------
    np.ndarray
        ``(nRows,)`` cosine similarities. NaN for invalid rows.
    """
    p_mat_a = np.asarray(p_mat_a, dtype=np.float64)
    p_mat_b = np.asarray(p_mat_b, dtype=np.float64)
    if p_mat_a.ndim == 1:
        p_mat_a = p_mat_a.reshape(1, -1)
    if p_mat_b.ndim == 1:
        p_mat_b = p_mat_b.reshape(1, -1)

    n_rows = p_mat_a.shape[0]
    if p_mat_b.shape[0] != n_rows:
        raise ValueError("p_mat_a and p_mat_b must have the same number of rows.")

    use_wa = weights_a is not None
    use_wb = weights_b is not None
    use_spec = spectrum is not None

    if use_wa:
        weights_a = np.asarray(weights_a, dtype=np.float64)
        if weights_a.shape != p_mat_a.shape:
            raise ValueError("weights_a must be the same shape as p_mat_a.")
    if use_wb:
        weights_b = np.asarray(weights_b, dtype=np.float64)
        if weights_b.shape != p_mat_b.shape:
            raise ValueError("weights_b must be the same shape as p_mat_b.")

    if precision is not None:
        p_mat_a = np.round(p_mat_a, precision)
        p_mat_b = np.round(p_mat_b, precision)
        if use_wa:
            weights_a = np.round(weights_a, precision)
        if use_wb:
            weights_b = np.round(weights_b, precision)

    s = np.full(n_rows, np.nan)

    # ── Phase 1: Canonicalise and build individual-set keys ─────────
    key_a: list[tuple | None] = [None] * n_rows
    key_b: list[tuple | None] = [None] * n_rows
    valid = [False] * n_rows

    canon_data_a: dict[tuple, tuple[np.ndarray, np.ndarray | None]] = {}
    canon_data_b: dict[tuple, tuple[np.ndarray, np.ndarray | None]] = {}

    for i in range(n_rows):
        pa_row = p_mat_a[i]
        pb_row = p_mat_b[i]
        mask_a = ~np.isnan(pa_row)
        mask_b = ~np.isnan(pb_row)
        pa_valid = pa_row[mask_a]
        pb_valid = pb_row[mask_b]

        if len(pa_valid) < r or len(pb_valid) < r:
            continue

        wa_valid = weights_a[i, mask_a] if use_wa else None
        wb_valid = weights_b[i, mask_b] if use_wb else None

        ka, kb, ca_p_arr, ca_w_arr, cb_p_arr, cb_w_arr = _pair_canonical_key(
            pa_valid, wa_valid, pb_valid, wb_valid,
            sigma=sigma, r=r, is_rel=is_rel, is_per=is_per, period=period,
            precision=precision,
        )

        key_a[i] = ka
        key_b[i] = kb
        valid[i] = True

        if ka not in canon_data_a:
            canon_data_a[ka] = (ca_p_arr, ca_w_arr)
        if kb not in canon_data_b:
            canon_data_b[kb] = (cb_p_arr, cb_w_arr)

    # ── Phase 2: Build density structs for unique individual sets ───
    dens_cache_a: dict[tuple, object] = {}
    for ka, (p_arr, w_arr) in canon_data_a.items():
        if use_spec:
            p_arr, w_arr = add_spectra(p_arr, w_arr, *spectrum)
        dens_cache_a[ka] = build_exp_tens(
            p_arr, w_arr, sigma, r, is_rel, is_per, period, verbose=False
        )

    dens_cache_b: dict[tuple, object] = {}
    for kb, (p_arr, w_arr) in canon_data_b.items():
        if use_spec:
            p_arr, w_arr = add_spectra(p_arr, w_arr, *spectrum)
        dens_cache_b[kb] = build_exp_tens(
            p_arr, w_arr, sigma, r, is_rel, is_per, period, verbose=False
        )

    n_unique_a = len(dens_cache_a)
    n_unique_b = len(dens_cache_b)
    n_valid = sum(valid)

    if verbose:
        print(
            f"cos_sim_exp_tens: {n_rows} rows, {n_valid} valid, "
            f"{n_unique_a} unique A-sets, {n_unique_b} unique B-sets."
        )
        if is_rel:
            print(
                "  Canonicalization: A-sets and B-sets independently "
                "normalized for transposition"
                + (" and octave equivalence." if is_per else ".")
            )
        else:
            print(
                "  Canonicalization: joint co-transposition"
                + (" with octave equivalence" if is_per else "")
                + "; B-set counts reflect position relative to A."
            )
        print(
            f"cos_sim_exp_tens: built {n_unique_a + n_unique_b} "
            f"density structs ({n_unique_a} A + {n_unique_b} B)."
        )

    # ── Phase 3: Compute via polymorphic cos_sim_exp_tens ────────────
    if n_valid == 0:
        if verbose:
            print("cos_sim_exp_tens: done.")
        return s

    valid_indices = [i for i in range(n_rows) if valid[i]]
    list_a_dens = [dens_cache_a[key_a[i]] for i in valid_indices]
    list_b_dens = [dens_cache_b[key_b[i]] for i in valid_indices]

    cos_results = cos_sim_exp_tens(
        list_a_dens, list_b_dens,
        mode="pairwise", dedup=dedup,
        method=method,
        cancellation_threshold=cancellation_threshold,
        verbose=verbose,
    )

    # ── Phase 4: Map results back ────────────────────────────────────
    for k, idx in enumerate(valid_indices):
        s[idx] = cos_results[k]

    if verbose:
        print("cos_sim_exp_tens: done.")

    return s


# -------------------------------------------------------------------
#  batch_cos_sim_exp_tens (deprecated convenience wrapper)
# -------------------------------------------------------------------


@_with_dispatch_scope
def batch_cos_sim_exp_tens(
    p_mat_a: np.ndarray,
    p_mat_b: np.ndarray,
    sigma: float,
    r: int,
    is_rel: bool,
    is_per: bool,
    period: float,
    *,
    weights_a: np.ndarray | None = None,
    weights_b: np.ndarray | None = None,
    spectrum: list | None = None,
    precision: int | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """Deprecated. Use :func:`cos_sim_exp_tens` directly with 2-D matrices.

    .. deprecated:: 2.1
       The batched-raw-input dispatch has been folded into the unified
       :func:`cos_sim_exp_tens` entry point. Pass 2-D pitch matrices
       directly:

       .. code-block:: python

          # Old:
          s = batch_cos_sim_exp_tens(P1, P2, sigma, r, is_rel, is_per, period,
                                     weights_a=W1, weights_b=W2)
          # New:
          s = cos_sim_exp_tens(P1, W1, P2, W2, sigma, r, is_rel, is_per, period)

       Note the argument order: weights now follow each pitch matrix
       positionally (matching the SA scalar raw form), instead of being
       keyword-only. This shim preserves the old keyword-only weight API
       for backward compatibility but issues a ``DeprecationWarning``.
       This shim will be removed in a future release.
    """
    warnings.warn(
        "batch_cos_sim_exp_tens is deprecated. Pass 2-D pitch matrices "
        "directly to cos_sim_exp_tens (with weights as positional arguments "
        "after each pitch matrix, matching the SA scalar raw form). This "
        "shim will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _cos_sim_raw_sa_batch(
        p_mat_a, p_mat_b, sigma, r, is_rel, is_per, period,
        weights_a=weights_a, weights_b=weights_b,
        spectrum=spectrum, precision=precision,
        verbose=verbose,
    )

