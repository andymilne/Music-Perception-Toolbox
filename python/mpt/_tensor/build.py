"""Density construction: the build path.

Public entry point :func:`build_exp_tens` precomputes an r-ad
expectation tensor density object, dispatching on input shape:

* Numeric 1-D array / flat list of numbers -> single-attribute path,
  returns :class:`ExpTensDensity`.
* Cell-array-like input (list of per-attribute matrices) -> multi-
  attribute path, returns :class:`MaetDensity`.

Both paths share the small set of multi-attribute input-coercion and
weight-normalisation helpers from :mod:`._tensor.density`. The
single-attribute path produces an :class:`ExpTensDensity` whose
expensive per-tuple arrays are lazy by default (see the class
docstring); the multi-attribute path always materialises its perm/comb
arrays via :func:`_ma_build_perm_arrays`.

See USER_GUIDE §3 ("Building densities") for the user-facing
description and :doc:`/ARCHITECTURE` §2 for the layering.
"""
from __future__ import annotations

import warnings
from itertools import permutations
from math import factorial

import numpy as np
from scipy.special import comb as _comb

from .._utils import validate_weights
from .density import (
    ExpTensDensity,
    MaetDensity,
    _broadcast_attr_weight,
    _cartesian_indices,
    _coerce_attr_matrix,
    _nchoosek_indices,
    _normalise_weights_ma,
)


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

        build_exp_tens(p_attr, w, sigma_vec, r_vec,
                       is_rel_vec, is_per_vec, period_vec, *, verbose=True)

    Both paths take seven positional arguments; they are distinguished
    purely by the type of the first argument (a list/tuple of attribute
    matrices selects the multi-attribute path). Every attribute is
    self-contained, carrying its own geometry, so all geometry vectors
    are per-attribute (length *A*); shared geometry is expressed by
    repeating a value across the attributes that should share it.

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
    sigma_vec : (A,) array-like of float
        Per-attribute Gaussian widths.
    r_vec : (A,) array-like of int
        Per-attribute tuple sizes.
    is_rel_vec, is_per_vec : (A,) array-like of bool
        Per-attribute isRel and isPer flags.
    period_vec : (A,) array-like of float
        Per-attribute periods (use 0 for attributes that are not
        periodic).

    Returns
    -------
    ExpTensDensity or MaetDensity
        Depending on which path is taken.

    See Also
    --------
    ExpTensDensity, MaetDensity, eval_exp_tens, cos_sim_exp_tens
    """
    if _looks_like_multi_attr(p):
        if len(args) not in (5, 6):
            raise ValueError(
                f"Multi-attribute call expects 7 or 8 positional arguments "
                f"(p_attr, w, sigma_vec, r_vec, is_rel_vec, is_per_vec, "
                f"period_vec[, is_sym_vec]); got {2 + len(args)}."
            )
        sigma_vec, r_vec, is_rel_vec, is_per_vec, period_vec = args[:5]
        is_sym_vec = args[5] if len(args) == 6 else None
        return _build_exp_tens_ma(
            p, w, sigma_vec, r_vec,
            is_rel_vec, is_per_vec, period_vec, is_sym_vec,
            verbose=verbose,
        )
    else:
        if len(args) not in (5, 6):
            raise ValueError(
                f"Single-attribute call expects 7 or 8 positional arguments "
                f"(p, w, sigma, r, is_rel, is_per, period[, is_sym]); got "
                f"{2 + len(args)}."
            )
        sigma, r, is_rel, is_per, period = args[:5]
        is_sym = args[5] if len(args) == 6 else True
        return _build_exp_tens_sa(
            p, w, sigma, r, is_rel, is_per, period, is_sym,
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
    is_rel_vec,
    is_per_vec,
    period_vec,
    is_sym_vec=None,
    *,
    verbose: bool = True,
) -> MaetDensity:
    """Multi-attribute expectation tensor builder.

    Private: users call :func:`build_exp_tens`, which dispatches here
    when given a list/tuple of attribute matrices as the first argument.

    Every attribute is self-contained: the geometry vectors
    (``sigma_vec``, ``is_rel_vec``, ``is_per_vec``, ``period_vec``)
    are per-attribute (length *A*). Shared geometry is expressed by
    repeating a value across attributes.

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

    sigma_vec  = np.asarray(sigma_vec,  dtype=np.float64).ravel()
    is_rel_vec = np.asarray(is_rel_vec, dtype=bool).ravel()
    is_per_vec = np.asarray(is_per_vec, dtype=bool).ravel()
    period_vec = np.asarray(period_vec, dtype=np.float64).ravel()

    # [sym] is per-attribute; default all-True preserves the legacy
    # symmetrised (v2.0.0) semantics. [sym] = 1 symmetrises each
    # r-sub-tuple over the slot-permutation (S_r) orbit; [sym] = 0
    # keeps it ordered. (r_a = 1 makes the flag vacuous.)
    if is_sym_vec is None:
        is_sym_vec = np.ones(A, dtype=bool)
    else:
        is_sym_vec = np.asarray(is_sym_vec, dtype=bool).ravel()

    for name, vec in (("sigma_vec",  sigma_vec),
                      ("is_rel_vec", is_rel_vec),
                      ("is_per_vec", is_per_vec),
                      ("period_vec", period_vec),
                      ("is_sym_vec", is_sym_vec)):
        if vec.size != A:
            raise ValueError(
                f"{name} must have length {A} (n attributes), got {vec.size}."
            )

    for a in range(A):
        if is_rel_vec[a] and r_vec[a] < 2:
            warnings.warn(
                f"is_rel = True combined with r_a = 1 for attribute {a} "
                f"produces a degenerate (constant) density. For cross-event "
                f"translation invariance, use `difference_events` as a "
                f"preprocessing step."
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
        r_a = int(r_vec[a])
        if is_rel_vec[a]:
            dim_per_attr[a] = r_a - 1 if r_a >= 2 else 0
        else:
            dim_per_attr[a] = r_a
    dim = int(dim_per_attr.sum())

    if verbose:
        print(
            f"build_exp_tens (MAET): {A} attributes, "
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
            is_rel_vec=is_rel_vec, is_sym_vec=is_sym_vec,
            N=N, A=A,
        )

    return MaetDensity(
        tag="MaetDensity",
        n_attrs=A,
        n=N,
        r=r_vec,
        k=K_a,
        p_attr=p_attr,
        w=w_list,
        sigma=sigma_vec,
        is_rel=is_rel_vec,
        is_per=is_per_vec,
        period=period_vec,
        is_sym=is_sym_vec,
        dim=dim,
        dim_per_attr=dim_per_attr,
        _build_lazy=_build_lazy,
    )



def _ma_build_perm_arrays(
    *,
    p_attr,
    w_list,
    r_vec,
    is_rel_vec,
    is_sym_vec,
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

            # [sym] = 1 (default): symmetrise each combination into its
            # full S_r orbit (r! permuted copies) -- the perm side is the
            # symmetrised density. [sym] = 0: keep each combination in
            # listed order (one ordered kernel per combination), so the
            # perm side equals the comb side -- the de-reflected density
            # (upper triangle at r=2, the single ordered tuple at r=K).
            # r_a == 1 has no order to symmetrise, so both coincide there.
            if r_a == 1 or not is_sym_vec[a]:
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
        r_a = int(r_vec[a])
        if is_rel_vec[a]:
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
    is_sym: bool = True,
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
        is_sym=bool(is_sym),
    )