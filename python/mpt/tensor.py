"""Expectation tensor construction, evaluation, and similarity.

The central data structure is :class:`ExpTensDensity`, a precomputed
Gaussian-mixture density representing the expected distribution of
r-ads (ordered r-tuples) from a weighted pitch multiset.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from itertools import permutations
from math import factorial

import numpy as np
from scipy.special import comb as _comb

from ._utils import estimate_comp_time, validate_weights
from .spectra import add_spectra


def _nchoosek_indices(n: int, r: int) -> np.ndarray:
    """Return all r-combinations of range(n) as an (r, C(n,r)) array."""
    from itertools import combinations

    combos = list(combinations(range(n), r))
    return np.array(combos, dtype=np.intp).T  # r x C(n,r)


# -------------------------------------------------------------------
#  Data class
# -------------------------------------------------------------------


@dataclass
class ExpTensDensity:
    """Precomputed expectation tensor density.

    Attributes match the MATLAB struct fields; see :func:`build_exp_tens`.
    """

    p: np.ndarray
    w: np.ndarray
    sigma: float
    r: int
    is_rel: bool
    is_per: bool
    period: float
    dim: int

    # For eval_exp_tens
    centres: np.ndarray  # dim x nJ
    w_j: np.ndarray  # (nJ,)
    n_j: int

    # For cos_sim_exp_tens
    u_perm: np.ndarray  # r x nJ_perm
    w_perm: np.ndarray  # (nJ_perm,)
    n_j_perm: int
    v_comb: np.ndarray  # r x nK
    wv_comb: np.ndarray  # (nK,)
    n_k: int


# -------------------------------------------------------------------
#  MaetDensity  (multi-attribute expectation tensor density)
# -------------------------------------------------------------------


@dataclass
class MaetDensity:
    """Precomputed multi-attribute expectation tensor density (MAET).

    Returned by :func:`build_exp_tens` when called in multi-attribute
    form (first argument a list/tuple of attribute matrices). The
    single-attribute return type is :class:`ExpTensDensity`.

    See the MAET specification (``multi_attribute_tensor_specification.md``)
    §2 and §6, and :func:`build_exp_tens` for argument semantics.

    Conventions
    -----------
    Group and attribute indices in the fields below are 0-indexed
    (Python convention). Input accepts 0-indexed group-index vectors or
    cell-of-lists forms; the MATLAB struct exposes 1-indexed indices.

    Column *j* of ``centres[a]``, ``u_perm[a]``, ``w_j``, and
    ``event_of_j`` refer to the same global perm-side tuple. Similarly
    for column *k* across ``v_comb[a]``, ``wv_comb``, and
    ``event_of_k``.
    """

    tag: str
    n_attrs: int
    n_groups: int
    n: int

    # Grouping
    group_of_attr: np.ndarray            # (A,) intp, 0-indexed
    attrs_of_group: list                 # list of length G; each an (n_a,) intp array

    # Per-attribute parameters and data
    r: np.ndarray                        # (A,) intp, per-attribute tuple size
    k: np.ndarray                        # (A,) intp, per-attribute K_a
    p_attr: list                         # list of length A; each K_a x N float64
    w: list                              # list of length A; each K_a x N float64

    # Per-group parameters
    sigma: np.ndarray                    # (G,) float64
    is_rel: np.ndarray                   # (G,) bool
    is_per: np.ndarray                   # (G,) bool
    period: np.ndarray                   # (G,) float64

    # Dimensionality
    dim: int
    dim_per_attr: np.ndarray             # (A,) intp

    # Totals
    n_j: int
    n_k: int

    # For eval_exp_tens
    centres: list                        # list of length A; each (r_a - isRel_{g(a)}) x nJ

    # For cos_sim_exp_tens
    u_perm: list                         # list of length A; each r_a x nJ
    v_comb: list                         # list of length A; each r_a x nK
    w_j: np.ndarray                      # (nJ,)  perm-side per-tuple weight products
    wv_comb: np.ndarray                  # (nK,)  comb-side per-tuple weight products

    # Event bookkeeping
    event_of_j: np.ndarray               # (nJ,) intp
    event_of_k: np.ndarray               # (nK,) intp


# -------------------------------------------------------------------
#  WindowedMaetDensity  (MAET with a post-tensor window applied)
# -------------------------------------------------------------------


@dataclass
class WindowedMaetDensity:
    """A MaetDensity together with a post-tensor window specification.

    Returned by :func:`window_tensor`. Bundles an underlying
    :class:`MaetDensity` with per-group window parameters. No math is
    performed at construction time — the window is applied lazily by
    :func:`eval_exp_tens` (pointwise multiplication by the window
    function) and by :func:`cos_sim_exp_tens` (closed-form windowed
    inner product).

    See the MAET specification §4.3 for the full semantics.

    Fields
    ------
    tag : str
        Always ``"WindowedMaetDensity"``.
    dens : MaetDensity
        Underlying unwindowed density.
    size : (G,) float64
        Per-group window effective standard deviation in multiples of
        that group's ``sigma``. NaN or Inf means the group is not
        windowed.
    mix : (G,) float64
        Per-group shape parameter in [0, 1]: 0 = pure Gaussian,
        1 = pure rectangular, in between = rectangular-convolved-with-
        Gaussian. Ignored for groups with ``size`` NaN/Inf.
    centre : list of ndarray
        Length-A list; entry *a* is a 1-D array of length
        ``dim_per_attr[a]`` giving the per-attribute centre coordinates
        in effective space. Concatenating the entries for the attributes
        in a group gives the centre point in that group's effective
        subspace. Ignored for groups with ``size`` NaN/Inf.
    """

    tag: str
    dens: "MaetDensity"
    size: np.ndarray                     # (G,) float64
    mix: np.ndarray                      # (G,) float64
    centre: list                         # list of length A; each (dim_per_attr[a],)


# -------------------------------------------------------------------
#  MA-path helpers
# -------------------------------------------------------------------


def _coerce_attr_matrix(M) -> np.ndarray:
    """Coerce an attribute input to a 2-D K_a x N float64 matrix.

    A 1-D input is taken as K_a = 1 (a 1 x N row). 2-D inputs pass
    through. Higher dimensions are rejected.
    """
    M = np.asarray(M, dtype=np.float64)
    if M.ndim == 1:
        return M.reshape(1, -1)
    if M.ndim != 2:
        raise ValueError(
            f"Attribute matrix must be 1-D or 2-D, got ndim={M.ndim}."
        )
    return M


def _canonicalise_groups(groups, A: int):
    """Canonicalise the groups input to (group_of_attr, attrs_of_group, G).

    Accepts:
      - ``None`` or empty          -> each attribute its own singleton group
      - length-*A* vector of ints  -> group index per attribute (0-indexed,
                                      contiguous 0..G-1, no gaps)
      - list of length *G*, each entry a list/tuple/array of attribute
        indices -> explicit partition

    Returns
    -------
    group_of_attr : (A,) intp
    attrs_of_group : list of length G; each an intp ndarray of attr indices
    G : int
    """
    # None or empty -> singletons
    if groups is None:
        return (
            np.arange(A, dtype=np.intp),
            [np.array([a], dtype=np.intp) for a in range(A)],
            A,
        )

    if isinstance(groups, (list, tuple)):
        if len(groups) == 0:
            return (
                np.arange(A, dtype=np.intp),
                [np.array([a], dtype=np.intp) for a in range(A)],
                A,
            )
        # Cell-of-lists form: each element itself a list/tuple/ndarray
        if all(isinstance(g, (list, tuple, np.ndarray)) for g in groups):
            return _canon_groups_cell_form(groups, A)
        # Else fall through to vector form below

    arr = np.asarray(groups)
    if arr.size == 0:
        return (
            np.arange(A, dtype=np.intp),
            [np.array([a], dtype=np.intp) for a in range(A)],
            A,
        )
    return _canon_groups_vector_form(arr.astype(np.intp).ravel(), A)


def _canon_groups_cell_form(groups, A: int):
    G = len(groups)
    group_of_attr = np.full(A, -1, dtype=np.intp)
    attrs_of_group = []
    for g in range(G):
        idx = np.asarray(groups[g], dtype=np.intp).ravel()
        attrs_of_group.append(idx.copy())
        for a in idx.tolist():
            if a < 0 or a >= A:
                raise ValueError(
                    f"Group {g} references attribute {a}, out of range [0, {A-1}]."
                )
            if group_of_attr[a] != -1:
                raise ValueError(
                    f"Attribute {a} is listed in more than one group."
                )
            group_of_attr[a] = g
    if np.any(group_of_attr == -1):
        missing = np.nonzero(group_of_attr == -1)[0].tolist()
        raise ValueError(
            f"Attributes {missing} are not assigned to any group."
        )
    return group_of_attr, attrs_of_group, G


def _canon_groups_vector_form(arr: np.ndarray, A: int):
    if arr.size != A:
        raise ValueError(
            f"Group-index vector must have length {A} (n attributes)."
        )
    if np.any(arr < 0):
        raise ValueError("Group indices must be non-negative integers.")
    unique = np.unique(arr)
    expected = np.arange(unique.size)
    if not np.array_equal(unique, expected):
        raise ValueError(
            f"Group indices must be contiguous integers 0..G-1 with no "
            f"gaps. Got unique values: {unique.tolist()}."
        )
    G = int(unique.size)
    attrs_of_group = [np.nonzero(arr == g)[0].astype(np.intp) for g in range(G)]
    return arr.astype(np.intp), attrs_of_group, G


def _normalise_weights_ma(w, A: int, K_a: np.ndarray, N: int) -> list:
    """Normalise the top-level MA weight input to a length-*A* list of
    K_a x N matrices.

    Accepts ``None``, a scalar, or a list/tuple of per-attribute inputs.
    """
    if w is None:
        return [np.ones((int(K_a[a]), N), dtype=np.float64) for a in range(A)]

    if np.isscalar(w):
        if w == 0:
            warnings.warn("All weights are zero.")
        return [
            np.full((int(K_a[a]), N), float(w), dtype=np.float64)
            for a in range(A)
        ]

    # A 0-D ndarray is still a scalar in intent
    if isinstance(w, np.ndarray) and w.ndim == 0:
        val = float(w.item())
        if val == 0:
            warnings.warn("All weights are zero.")
        return [
            np.full((int(K_a[a]), N), val, dtype=np.float64)
            for a in range(A)
        ]

    if isinstance(w, (list, tuple)):
        if len(w) != A:
            raise ValueError(
                f"Weight list must have length {A} (n attributes), "
                f"got {len(w)}."
            )
        return [
            _broadcast_attr_weight(w[a], int(K_a[a]), N, a)
            for a in range(A)
        ]

    raise ValueError(
        "Top-level weight argument must be None, a scalar, or a list/tuple "
        "of per-attribute inputs. Got type "
        f"{type(w).__name__}."
    )


def _broadcast_attr_weight(wa, K: int, N: int, attr_idx: int) -> np.ndarray:
    """Broadcast a per-attribute weight input to a K x N matrix.

    Accepts:
      - ``None`` or empty           -> ones
      - scalar                      -> constant
      - 1-D of length N (K != N)    -> per-event row, broadcast across slots
      - 1-D of length K (K != N)    -> per-slot column, broadcast across events
      - 2-D (1, N)                  -> per-event row, broadcast across slots
      - 2-D (K, 1)                  -> per-slot column, broadcast across events
      - 2-D (K, N)                  -> full matrix

    When K == N, a 1-D input is ambiguous and rejected.
    """
    if wa is None:
        return np.ones((K, N), dtype=np.float64)

    wa = np.asarray(wa, dtype=np.float64)

    # Scalar (0-D or size 1)
    if wa.size == 1:
        return np.full((K, N), float(wa.item()))

    if wa.ndim == 1:
        if wa.size == N and wa.size != K:
            return np.broadcast_to(wa[np.newaxis, :], (K, N)).copy()
        if wa.size == K and wa.size != N:
            return np.broadcast_to(wa[:, np.newaxis], (K, N)).copy()
        if wa.size == N and wa.size == K:
            raise ValueError(
                f"Attribute {attr_idx} weight is a 1-D array of length "
                f"{wa.size}, but K_a == N == {K} makes the per-event vs "
                f"per-slot interpretation ambiguous. Supply as a 2-D array "
                f"(shape ({K}, 1) for per-slot or (1, {N}) for per-event)."
            )
        raise ValueError(
            f"Attribute {attr_idx} weight is a 1-D array of length "
            f"{wa.size}; expected {K} (per-slot), {N} (per-event), or a "
            f"scalar."
        )

    if wa.ndim == 2:
        sz = wa.shape
        if sz == (1, N):
            return np.broadcast_to(wa, (K, N)).copy()
        if sz == (K, 1):
            return np.broadcast_to(wa, (K, N)).copy()
        if sz == (K, N):
            return wa.copy()
        raise ValueError(
            f"Attribute {attr_idx} weight has shape {sz}; expected "
            f"(1, {N}), ({K}, 1), or ({K}, {N})."
        )

    raise ValueError(
        f"Attribute {attr_idx} weight must be at most 2-D; got "
        f"ndim={wa.ndim}."
    )


def _cartesian_indices(sizes) -> list:
    """Column-major Cartesian-product indices.

    Given axis sizes (s_0, s_1, ..., s_{A-1}), returns a list of *A*
    arrays, each of length prod(sizes). Array *a* gives the index along
    axis *a* for each linear position. First axis varies fastest
    (matches MATLAB ndgrid / column-major flatten).
    """
    A = len(sizes)
    out = []
    for a in range(A):
        rep_inner = int(np.prod(sizes[:a])) if a > 0 else 1
        rep_outer = int(np.prod(sizes[a + 1:])) if a < A - 1 else 1
        base = np.arange(sizes[a], dtype=np.intp)
        expanded = np.repeat(base, rep_inner) if rep_inner > 1 else base
        tiled = np.tile(expanded, rep_outer) if rep_outer > 1 else expanded
        out.append(tiled)
    return out


# -------------------------------------------------------------------
#  build_exp_tens  (public dispatcher)
# -------------------------------------------------------------------


def build_exp_tens(p, w, *args, verbose: bool = True):
    """Precompute an r-ad expectation tensor density object.

    Dispatches on the type of the first argument:

      - numeric 1-D array, or a flat list/tuple of numbers -> single-
        attribute path, returns :class:`ExpTensDensity` (v2.0.0
        behaviour, unchanged).
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
    is routed to the single-attribute path — matching the v2.0.0
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
    """
    # --- Input normalisation ---------------------------------------

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

    # --- Per-event, per-attribute r-ad enumeration ------------------

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

            # Combinations: r_a x C(K_na, r_a)
            comb_list = list(_combinations(valid.tolist(), r_a))
            comb_mat = np.array(comb_list, dtype=np.intp).T  # r_a x C

            # Permutations: r_a x (r_a! * C(K_na, r_a))
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

            # Slot-weight products (per-tuple)
            w_col = w_list[a][:, n]
            if r_a == 1:
                perm_w[n][a] = w_col[perm_mat].ravel()
                comb_w[n][a] = w_col[comb_mat].ravel()
            else:
                perm_w[n][a] = np.prod(w_col[perm_mat], axis=0)
                comb_w[n][a] = np.prod(w_col[comb_mat], axis=0)

    # --- Cartesian product within events; concatenate across events --

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

    if verbose:
        print(
            f"build_exp_tens (MAET): {A} attributes, {G} groups, "
            f"{N} events. Total tuples: n_j = {n_j} (perm), "
            f"n_k = {n_k} (comb)."
        )

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

            # Perm side
            slot_perm = perm_idx[n][a][:, idx_perm[a]]   # r_a x nJh
            u_perm[a][:, off_j:off_j + nJh] = val_col[slot_perm]
            wJh *= perm_w[n][a][idx_perm[a]]

            # Comb side
            slot_comb = comb_idx[n][a][:, idx_comb[a]]   # r_a x nKh
            v_comb[a][:, off_k:off_k + nKh] = val_col[slot_comb]
            wKh *= comb_w[n][a][idx_comb[a]]

        w_j[off_j:off_j + nJh]       = wJh
        wv_comb[off_k:off_k + nKh]   = wKh
        event_of_j[off_j:off_j + nJh] = n
        event_of_k[off_k:off_k + nKh] = n

        off_j += nJh
        off_k += nKh

    # --- Centres (per-attribute isRel reduction) --------------------

    centres = []
    dim_per_attr = np.empty(A, dtype=np.intp)
    for a in range(A):
        g = int(group_of_attr[a])
        r_a = int(r_vec[a])
        if is_rel_vec[g]:
            if r_a >= 2:
                centres.append(u_perm[a][1:, :] - u_perm[a][:1, :])
                dim_per_attr[a] = r_a - 1
            else:
                # Degenerate: 0-dim; a warning has already been emitted.
                centres.append(np.empty((0, n_j), dtype=np.float64))
                dim_per_attr[a] = 0
        else:
            centres.append(u_perm[a].copy())
            dim_per_attr[a] = r_a
    dim = int(dim_per_attr.sum())

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
#  _build_exp_tens_sa  (single-attribute legacy path; v2.0.0 verbatim)
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

    n_perms = factorial(r)
    n_combs = int(_comb(n, r, exact=True))
    n_j = n_perms * n_combs
    n_k = n_combs

    if verbose:
        print(
            f"build_exp_tens: building {n_j} ordered {r}-tuples "
            f"from {n} pitches."
        )

    # All r-combinations (r x C(n,r))
    nck = _nchoosek_indices(n, r)

    # All permutations of range(r) — each column is one permutation
    all_perms = np.array(list(permutations(range(r))), dtype=np.intp).T  # r x r!

    # Build ordered r-tuples (perm side)
    j_idx = np.empty((r, n_j), dtype=np.intp)
    offset = 0
    for i in range(n_perms):
        j_idx[:, offset : offset + n_combs] = nck[all_perms[:, i], :]
        offset += n_combs

    u_perm = p[j_idx]  # r x nJ
    w_perm = np.prod(w[j_idx], axis=0)  # (nJ,)

    # r-combinations (comb side, for cos_sim)
    v_comb = p[nck]  # r x nK
    wv_comb = np.prod(w[nck], axis=0)  # (nK,)

    # Reduce to interval centres if relative
    if is_rel:
        centres = u_perm[1:, :] - u_perm[0, :]  # (r-1) x nJ
    else:
        centres = u_perm.copy()  # r x nJ

    return ExpTensDensity(
        p=p,
        w=w,
        sigma=sigma,
        r=r,
        is_rel=is_rel,
        is_per=is_per,
        period=period,
        dim=dim,
        centres=centres,
        w_j=w_perm,
        n_j=n_j,
        u_perm=u_perm,
        w_perm=w_perm,
        n_j_perm=n_j,
        v_comb=v_comb,
        wv_comb=wv_comb,
        n_k=n_k,
    )


# -------------------------------------------------------------------
#  eval_exp_tens
# -------------------------------------------------------------------


# -------------------------------------------------------------------
#  eval_exp_tens  (public dispatcher)
# -------------------------------------------------------------------


def eval_exp_tens(
    dens,
    x,
    normalize: str = "none",
    *,
    verbose: bool = True,
) -> np.ndarray:
    """Evaluate an expectation tensor density at query points.

    Dispatches on the type of *dens*:

      - :class:`ExpTensDensity` -> single-attribute path (v2.0.0
        behaviour, unchanged).
      - :class:`MaetDensity`    -> multi-attribute path.

    Parameters
    ----------
    dens : ExpTensDensity or MaetDensity
        Precomputed density from :func:`build_exp_tens`.
    x : array-like
        Query points.
          - SA: a (dim, nQ) array with dim = r - is_rel. A 1-D input is
            coerced to a (1, nQ) row.
          - MA: either a list/tuple of *A* per-attribute query matrices
            (each ``(dim_a, nQ)`` where ``dim_a = r_a - is_rel[g(a)]``),
            or a single ``(dim, nQ)`` matrix with attribute rows
            stacked in attribute order. A 1-D input is coerced to
            ``(1, nQ)`` and is valid only when the total dim equals 1.
    normalize : str
        ``'none'`` (default), ``'gaussian'``, or ``'pdf'``. For the MA
        path, the Gaussian normalisation constant is the product of
        per-attribute constants:
        ``prod_a (2*pi*sigma[g(a)]^2)^(-dim_a/2) * sqrt(det_M_a)``
        where ``det_M_a = 1/r_a`` when ``is_rel[g(a)]`` and ``r_a >= 2``,
        else ``1``.
    verbose : bool
        Print progress.

    Returns
    -------
    np.ndarray
        (nQ,) density values.

    See Also
    --------
    build_exp_tens, cos_sim_exp_tens
    """
    if isinstance(dens, WindowedMaetDensity):
        # Evaluate underlying density, multiply elementwise by window.
        underlying = _eval_exp_tens_ma(dens.dens, x, normalize, verbose=verbose)
        # Reconstruct per-attribute x_list so we can apply the window.
        x_list = _split_query_to_attr_list(dens.dens, x)
        W_vals = _evaluate_window_on_query(dens, x_list)
        return underlying * W_vals
    if isinstance(dens, MaetDensity):
        return _eval_exp_tens_ma(dens, x, normalize, verbose=verbose)
    if isinstance(dens, ExpTensDensity):
        return _eval_exp_tens_sa(dens, x, normalize, verbose=verbose)
    raise TypeError(
        f"dens must be an ExpTensDensity, MaetDensity, or "
        f"WindowedMaetDensity; got {type(dens).__name__}."
    )


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
    verbose: bool = True,
) -> np.ndarray:
    """Single-attribute expectation tensor evaluation (v2.0.0 body)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)

    centres = dens.centres
    w_j = dens.w_j
    n_j = dens.n_j
    sigma = dens.sigma
    r = dens.r
    dim = dens.dim
    is_rel = dens.is_rel
    is_per = dens.is_per
    period = dens.period

    if x.shape[0] != dim:
        raise ValueError(
            f"x must have {dim} rows (each column is a {dim}-D query point)."
        )
    n_q = x.shape[1]

    n_pairs = int(n_j) * int(n_q)
    estimate_comp_time(n_pairs, dim, "eval_exp_tens", verbose)

    vals = _eval_core(centres, w_j, n_j, x, n_q, dim, sigma, r, is_rel, is_per, period)

    # Normalization
    if normalize != "none":
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
):
    """Single-chunk MAET evaluation.

    Accumulates the summed-quadratic exponent across attributes, then
    exponentiates once and does the weighted sum against ``w_j``.
    """
    q_total = np.zeros((int(n_j), int(n_qc)), dtype=np.float64)

    for a in range(A):
        g = int(group_of[a])
        da = int(dim_per[a])
        if da == 0:
            # Degenerate attribute (r_a=1, is_rel=true). Constant along
            # this axis: zero contribution to q_total. Skip.
            continue

        # D_a shape: (da, nJ, nQc)
        c_a = centres[a]
        x_a = x_list[a]
        d_a = c_a[:, :, None] - x_a[:, None, :]

        if is_per_g[g]:
            pg = float(period_g[g])
            d_a = np.mod(d_a + pg / 2, pg) - pg / 2

        if is_rel_g[g]:
            q_a = np.sum(d_a**2, axis=0) - np.sum(d_a, axis=0) ** 2 / float(r_vec[a])
        else:
            q_a = np.sum(d_a**2, axis=0)

        q_total = q_total + q_a / (2 * sigma_g[g] ** 2)

    e = np.exp(-q_total)           # (nJ, nQc)
    return w_j @ e                  # (nQc,)


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
    """Build and evaluate in one call. See :func:`eval_exp_tens`."""
    dens = build_exp_tens(p, w, sigma, r, is_rel, is_per, period, verbose=verbose)
    return eval_exp_tens(dens, x, normalize, verbose=verbose)


# -------------------------------------------------------------------
#  cos_sim_exp_tens
# -------------------------------------------------------------------


def cos_sim_exp_tens(
    dens_x,
    dens_y,
    *,
    verbose: bool = True,
) -> float:
    """Cosine similarity of two expectation tensor densities.

    Dispatches on the type of the density objects:

      - two :class:`ExpTensDensity` -> single-attribute path (v2.0.0
        behaviour, unchanged).
      - two :class:`MaetDensity`    -> multi-attribute path.

    The analytical inner product in the multi-attribute case factors as
    the elementwise product of per-attribute kernels (Section 2.7 of the
    MAET specification): no numerical integration is required.

    Parameters
    ----------
    dens_x, dens_y : ExpTensDensity or MaetDensity
        Precomputed density objects from :func:`build_exp_tens`. Both
        must be of the same type, and must share all structural
        parameters (``r``, ``sigma``, ``is_rel``, ``is_per``, ``period``
        in the SA case; ``n_attrs``, ``group_of_attr``, ``r``, ``sigma``,
        ``is_rel``, ``is_per``, and ``period`` per group in the MA
        case).
    verbose : bool
        Print progress.

    Returns
    -------
    float
        Cosine similarity in [0, 1] for non-negative weights.

    References
    ----------
    Originally by David Bulger, Macquarie University (2016).
    Adapted for the Music Perception Toolbox v2 by Andrew J. Milne.

    See Also
    --------
    build_exp_tens, eval_exp_tens, batch_cos_sim_exp_tens
    """
    if isinstance(dens_x, WindowedMaetDensity) or \
            isinstance(dens_y, WindowedMaetDensity):
        # At least one operand is windowed. Reject two MaetDensity-shape
        # incompatibilities up front.
        if isinstance(dens_x, WindowedMaetDensity):
            other = dens_y
        else:
            other = dens_x
        if not isinstance(other, (MaetDensity, WindowedMaetDensity)):
            raise TypeError(
                "Windowed density can only be compared with another "
                "MaetDensity or WindowedMaetDensity."
            )
        return _cos_sim_exp_tens_windowed(dens_x, dens_y, verbose=verbose)
    if isinstance(dens_x, MaetDensity):
        if not isinstance(dens_y, MaetDensity):
            raise TypeError(
                "dens_x is a MaetDensity but dens_y is not; both must be "
                "the same type."
            )
        return _cos_sim_exp_tens_ma(dens_x, dens_y, verbose=verbose)
    if isinstance(dens_x, ExpTensDensity):
        if not isinstance(dens_y, ExpTensDensity):
            raise TypeError(
                "dens_x is an ExpTensDensity but dens_y is not; both must "
                "be the same type."
            )
        return _cos_sim_exp_tens_sa(dens_x, dens_y, verbose=verbose)
    raise TypeError(
        f"Both arguments must be ExpTensDensity or MaetDensity; got "
        f"{type(dens_x).__name__} and {type(dens_y).__name__}."
    )


# -------------------------------------------------------------------
#  _cos_sim_exp_tens_sa  (single-attribute legacy path)
# -------------------------------------------------------------------


def _cos_sim_exp_tens_sa(
    dens_x: ExpTensDensity,
    dens_y: ExpTensDensity,
    *,
    verbose: bool = True,
) -> float:
    """Single-attribute cosine similarity (v2.0.0 body)."""
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

    r = dens_x.r
    sigma = dens_x.sigma
    is_rel = dens_x.is_rel
    is_per = dens_x.is_per
    period = dens_x.period

    # Early return for degenerate case
    if r > min(len(dens_x.p), len(dens_y.p)):
        return float("nan")

    n_jx, n_kx = dens_x.n_j_perm, dens_x.n_k
    n_jy, n_ky = dens_y.n_j_perm, dens_y.n_k

    total_pairs = n_jx * n_ky + n_jx * n_kx + n_jy * n_ky
    estimate_comp_time(total_pairs, r, "cos_sim_exp_tens", verbose)

    ip_xy = _ip_core(
        dens_x.u_perm, dens_x.w_perm, n_jx,
        dens_y.v_comb, dens_y.wv_comb, n_ky,
        r, sigma, is_rel, is_per, period,
    )
    ip_xx = _ip_core(
        dens_x.u_perm, dens_x.w_perm, n_jx,
        dens_x.v_comb, dens_x.wv_comb, n_kx,
        r, sigma, is_rel, is_per, period,
    )
    ip_yy = _ip_core(
        dens_y.u_perm, dens_y.w_perm, n_jy,
        dens_y.v_comb, dens_y.wv_comb, n_ky,
        r, sigma, is_rel, is_per, period,
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
    verbose: bool = True,
) -> float:
    """Multi-attribute cosine similarity.

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

    A          = dens_x.n_attrs
    group_of   = dens_x.group_of_attr
    r_vec      = dens_x.r
    sigma_g    = dens_x.sigma
    is_rel_g   = dens_x.is_rel
    is_per_g   = dens_x.is_per
    period_g   = dens_x.period

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
#  cos_sim_exp_tens_raw  (dispatches SA or MA based on input shape)
# -------------------------------------------------------------------


def cos_sim_exp_tens_raw(p1, w1, p2, w2, *args, verbose: bool = True) -> float:
    """Build two densities and compute cosine similarity in one call.

    Convenience wrapper around :func:`build_exp_tens` and
    :func:`cos_sim_exp_tens`. Dispatches on the form of *p1* and *p2*
    (which must agree):

      - two numeric 1-D arrays (or flat lists of numbers) -> SA path,
        with the v2.0.0 signature::

            cos_sim_exp_tens_raw(p1, w1, p2, w2,
                                 sigma, r, is_rel, is_per, period,
                                 *, verbose=True)

      - two list/tuple-of-attribute-matrices -> MA path::

            cos_sim_exp_tens_raw(p_attr1, w1, p_attr2, w2,
                                 sigma_vec, r_vec, groups,
                                 is_rel_vec, is_per_vec, period_vec,
                                 *, verbose=True)

    Parameters are the same as for :func:`build_exp_tens` in the
    corresponding path (both densities share the structural parameters
    — ``sigma``, ``r``, etc. — so they are passed once and applied to
    both).

    Returns
    -------
    float
        Cosine similarity.
    """
    p1_ma = _looks_like_multi_attr(p1)
    p2_ma = _looks_like_multi_attr(p2)
    if p1_ma != p2_ma:
        raise TypeError(
            "p1 and p2 must be the same kind: either both numeric "
            "1-D arrays (SA) or both lists/tuples of attribute matrices (MA)."
        )

    if p1_ma:
        if len(args) != 6:
            raise ValueError(
                f"Multi-attribute raw call expects 10 positional arguments "
                f"(p_attr1, w1, p_attr2, w2, sigma_vec, r_vec, groups, "
                f"is_rel_vec, is_per_vec, period_vec); got {4 + len(args)}."
            )
        sigma_vec, r_vec, groups, is_rel_vec, is_per_vec, period_vec = args
        dx = build_exp_tens(
            p1, w1, sigma_vec, r_vec, groups,
            is_rel_vec, is_per_vec, period_vec, verbose=verbose,
        )
        dy = build_exp_tens(
            p2, w2, sigma_vec, r_vec, groups,
            is_rel_vec, is_per_vec, period_vec, verbose=verbose,
        )
    else:
        if len(args) != 5:
            raise ValueError(
                f"Single-attribute raw call expects 9 positional arguments "
                f"(p1, w1, p2, w2, sigma, r, is_rel, is_per, period); got "
                f"{4 + len(args)}."
            )
        sigma, r, is_rel, is_per, period = args
        dx = build_exp_tens(
            p1, w1, sigma, r, is_rel, is_per, period, verbose=verbose,
        )
        dy = build_exp_tens(
            p2, w2, sigma, r, is_rel, is_per, period, verbose=verbose,
        )
    return cos_sim_exp_tens(dx, dy, verbose=verbose)


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


def _ip_core(U, wU, nJ, V, wV, nK, r, sigma, is_rel, is_per, period):
    """Core inner product (perm-side × comb-side)."""
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


def _ip_full(U, wU, nJ, V, wV, nK, r, sigma, is_rel, is_per, period):
    """Fully vectorized inner product."""
    D = U[:, :, None] - V[:, None, :]  # (r, nJ, nK)

    if is_per:
        D = np.mod(D + period / 2, period) - period / 2

    Q = _compute_Q(D, r, is_rel, is_per, period)

    E = np.exp(-Q / (4 * sigma**2))  # (nJ, nK)
    return float(wU @ (E @ wV))


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

    Returns
    -------
    best_p : tuple
        Canonical pitch tuple.
    best_w : tuple or None
        Canonical weight tuple (if weights provided).
    best_shift : float
        The pitch value subtracted to produce the canonical form.
    """
    n = len(p_sorted)
    has_w = w_sorted is not None

    best_p = tuple(p_sorted - p_sorted[0])
    best_w = tuple(w_sorted) if has_w else None
    best_shift = p_sorted[0]

    for rot in range(1, n):
        shifted = np.mod(p_sorted - p_sorted[rot], period)
        si = np.argsort(shifted)
        shifted = shifted[si]
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
#  batch_cos_sim_exp_tens
# -------------------------------------------------------------------


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
    """Batch cosine similarity of expectation tensors.

    Computes cosine similarity for many paired weighted multisets
    (*p* represents pitches or positions). Each row of *p_mat_a*
    and *p_mat_b* defines one pair.

    The function deduplicates equivalent rows before computation.
    Density structs (via ``build_exp_tens``) are cached per unique
    individual set and ``cos_sim_exp_tens`` is called once per
    unique (A, B) pair, with results mapped back to all matching
    rows. The equivalences exploited depend on the mode:

    - **Absolute non-periodic** (*is_rel* = False, *is_per* = False):
      co-transposition — (A+c, B+c) is equivalent to (A, B).
      Mechanism: subtract A's minimum from both A and B.
    - **Absolute periodic** (*is_rel* = False, *is_per* = True):
      co-transposition and octave displacement.
      Mechanism: mod-reduce both sets, then apply the cyclic
      canonical form to A and the same shift to B.
    - **Relative non-periodic** (*is_rel* = True, *is_per* = False):
      independent transposition of each set (co-transposition is a
      special case).
      Mechanism: subtract the minimum from each set independently.
    - **Relative periodic** (*is_rel* = True, *is_per* = True):
      independent transposition and octave displacement of each set
      (co-transposition is a special case).
      Mechanism: independent cyclic canonical form for each set.

    Parameters
    ----------
    p_mat_a : (nRows, nA) array
        Multiset A values. NaN entries are ignored.
    p_mat_b : (nRows, nB) array
        Multiset B values.
    sigma, r, is_rel, is_per, period :
        Tensor parameters.
    weights_a, weights_b : array, optional
        Weight matrices matching *p_mat_a* / *p_mat_b*.
    spectrum : list, optional
        Arguments for :func:`~mpt.spectra.add_spectra`.
    precision : int, optional
        Round pitch and weight values to this many decimal places
        before processing (and again after canonicalization, to
        absorb arithmetic noise from mod-reduction and subtraction).
        Ensures that nominally identical multisets differing only by
        floating-point noise are correctly deduplicated. For pitch
        data in cents on a 12-TET grid, 4 is more than sufficient;
        for fractional-cent values (e.g., from JI ratios), 6
        preserves all meaningful precision. Default: no rounding
        (full floating-point precision).

        **Limitation:** ``precision`` rounds to decimal places, so it
        cannot resolve discrepancies when pitches lie on an irrational
        grid — e.g., N-EDO tunings where the step size 1200/N is a
        repeating decimal (such as 22-EDO: 1200/22 ≈ 54.5454...).
        Different transpositions of the same set will produce
        different last-digit truncations that no decimal precision
        can collapse. In such cases, convert to integer EDO steps
        before calling this function (scaling *sigma* and *period*
        accordingly) to make deduplication exact.
    verbose : bool
        Print progress.

    Returns
    -------
    np.ndarray
        (nRows,) cosine similarities. NaN for invalid rows.
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

    # Apply precision rounding
    if precision is not None:
        p_mat_a = np.round(p_mat_a, precision)
        p_mat_b = np.round(p_mat_b, precision)
        if use_wa:
            weights_a = np.round(weights_a, precision)
        if use_wb:
            weights_b = np.round(weights_b, precision)

    s = np.full(n_rows, np.nan)

    # ── Phase 1: Canonicalize and build individual-set keys ──────────

    # key_a[i] and key_b[i] are hashable canonical forms for valid rows
    key_a: list[tuple | None] = [None] * n_rows
    key_b: list[tuple | None] = [None] * n_rows
    valid = [False] * n_rows

    # Also store the canonical pitches/weights for later extraction
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

        # Canonicalize each set and apply joint co-transposition
        # normalization for the absolute case.
        if is_rel:
            # Relative: independent canonicalization (each set normalized
            # for transposition independently)
            ca_p, ca_w = _canonicalize_set(pa_valid, wa_valid, is_rel, is_per, period)
            cb_p, cb_w = _canonicalize_set(pb_valid, wb_valid, is_rel, is_per, period)
        else:
            # Absolute: joint co-transposition normalization.
            # cosSimExpTens(A-c, B-c) = cosSimExpTens(A, B) because the
            # raw tuple differences cancel. We find A's canonical form
            # and apply the same shift to B.

            # Canonicalize A
            si_a = np.argsort(pa_valid)
            pa_s = pa_valid[si_a]
            wa_s = wa_valid[si_a] if wa_valid is not None else None

            if is_per:
                pa_s = np.mod(pa_s, period)
                si = np.argsort(pa_s)
                pa_s = pa_s[si]
                if wa_s is not None:
                    wa_s = wa_s[si]
                # Cyclic canonical form — collapses all rotations
                ca_p, ca_w, shift = _cyclic_canonical(pa_s, wa_s, period)
            else:
                shift = pa_s[0]
                ca_p = tuple(pa_s - shift)
                ca_w = tuple(wa_s) if wa_s is not None else None

            # Apply the same shift to B
            si_b = np.argsort(pb_valid)
            pb_s = pb_valid[si_b]
            wb_s = wb_valid[si_b] if wb_valid is not None else None

            if is_per:
                pb_shifted = np.mod(pb_s - shift, period)
                si = np.argsort(pb_shifted)
                cb_p = tuple(pb_shifted[si])
                cb_w = tuple(wb_s[si]) if wb_s is not None else None
            else:
                cb_p = tuple(pb_s - shift)
                cb_w = tuple(wb_s) if wb_s is not None else None

        # Re-round after canonicalization to collapse floating-point
        # noise introduced by mod-reduction and subtraction.
        if precision is not None:
            ca_p = tuple(round(x, precision) for x in ca_p)
            cb_p = tuple(round(x, precision) for x in cb_p)
            if ca_w is not None:
                ca_w = tuple(round(x, precision) for x in ca_w)
            if cb_w is not None:
                cb_w = tuple(round(x, precision) for x in cb_w)

        # Hashable key = (canonical_pitches, canonical_weights_or_None)
        ka = (ca_p, ca_w)
        kb = (cb_p, cb_w)

        key_a[i] = ka
        key_b[i] = kb
        valid[i] = True

        # Cache canonical arrays (first occurrence wins)
        if ka not in canon_data_a:
            canon_data_a[ka] = (
                np.array(ca_p, dtype=np.float64),
                np.array(ca_w, dtype=np.float64) if ca_w is not None else None,
            )
        if kb not in canon_data_b:
            canon_data_b[kb] = (
                np.array(cb_p, dtype=np.float64),
                np.array(cb_w, dtype=np.float64) if cb_w is not None else None,
            )

    # ── Phase 2: Deduplicate individual sets, then pairs ─────────────

    unique_a_keys = list(canon_data_a.keys())
    unique_b_keys = list(canon_data_b.keys())
    n_unique_a = len(unique_a_keys)
    n_unique_b = len(unique_b_keys)

    # Build pair keys and deduplicate
    pair_key_to_idx: dict[tuple, int] = {}
    row_to_pair: list[int | None] = [None] * n_rows
    unique_pair_list: list[tuple] = []

    for i in range(n_rows):
        if not valid[i]:
            continue
        pk = (key_a[i], key_b[i])
        if pk not in pair_key_to_idx:
            pair_key_to_idx[pk] = len(unique_pair_list)
            unique_pair_list.append(pk)
        row_to_pair[i] = pair_key_to_idx[pk]

    n_unique_pairs = len(unique_pair_list)

    if verbose:
        n_valid = sum(valid)
        print(
            f"batch_cos_sim_exp_tens: {n_rows} rows, {n_valid} valid, "
            f"{n_unique_a} unique A-sets, {n_unique_b} unique B-sets, "
            f"{n_unique_pairs} unique pairs."
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

    # ── Phase 3: Build density structs for unique individual sets ─────

    dens_a: dict[tuple, object] = {}
    for ka in unique_a_keys:
        p_arr, w_arr = canon_data_a[ka]
        if use_spec:
            p_arr, w_arr = add_spectra(p_arr, w_arr, *spectrum)
        dens_a[ka] = build_exp_tens(
            p_arr, w_arr, sigma, r, is_rel, is_per, period, verbose=False
        )

    dens_b: dict[tuple, object] = {}
    for kb in unique_b_keys:
        p_arr, w_arr = canon_data_b[kb]
        if use_spec:
            p_arr, w_arr = add_spectra(p_arr, w_arr, *spectrum)
        dens_b[kb] = build_exp_tens(
            p_arr, w_arr, sigma, r, is_rel, is_per, period, verbose=False
        )

    if verbose:
        print(
            f"batch_cos_sim_exp_tens: built {n_unique_a + n_unique_b} "
            f"density structs ({n_unique_a} A + {n_unique_b} B)."
        )

    # ── Phase 4: Compute similarity for each unique pair ─────────────

    unique_s = [None] * n_unique_pairs
    for up, (ka, kb) in enumerate(unique_pair_list):
        unique_s[up] = cos_sim_exp_tens(dens_a[ka], dens_b[kb], verbose=False)

        if verbose and ((up + 1) % 100 == 0 or up + 1 == n_unique_pairs):
            print(f"  {up + 1} / {n_unique_pairs} unique pairs computed.")

    # ── Phase 5: Map results back ────────────────────────────────────

    for i in range(n_rows):
        if valid[i]:
            s[i] = unique_s[row_to_pair[i]]

    if verbose:
        print("batch_cos_sim_exp_tens: done.")

    return s


# ===================================================================
#  difference_events (MAET preprocessing helper)
# ===================================================================


def difference_events(p_attr, w, groups, diff_orders, periods):
    """Replace selected groups' event sequences with inter-event differences.

    Cross-event preprocessing for multi-attribute tensor input. Takes
    the ``(p_attr, w)`` pair that one would otherwise feed to
    :func:`build_exp_tens` and returns a transformed
    ``(p_attr_diff, w_diff)`` pair with the same shape conventions, in
    which the event columns of each selected group have been replaced
    by *k*-fold inter-event differences. The output feeds directly into
    :func:`build_exp_tens` without any further massaging.

    See the MAET specification §7 for the full semantics. In brief:

    - **Values.** For a group with ``diff_orders[g] = k``, the value
      matrix of each attribute in that group is replaced by its *k*-th
      finite difference along the event axis, reducing the event count
      by *k*. Order 0 leaves a group unchanged. If ``periods[g] > 0``,
      each raw difference is wrapped to ``[-P/2, P/2)`` (shortest-arc
      convention, matching :func:`cos_sim_exp_tens`).

    - **Weights.** Weight inputs follow the toolbox's standard
      broadcast convention: ``None`` broadcasts 1, a scalar
      broadcasts uniformly, event-dependent inputs (``(1, N)`` row,
      ``(K_a, N)`` matrix) supply per-event values, and a
      ``(K_a, 1)`` column (or 1-D length ``K_a``) broadcasts per
      slot. The weight of a difference event is the *product* of
      the weights of its ``k + 1`` constituent input events — a
      rolling product of width ``k + 1`` along the event axis —
      interpretable as the probability that all constituents are
      perceived under the standard weights-as-salience reading.

    - **Event-count alignment.** The output event count is
      ``N' = N - max_g k_g``. Groups with ``k_g < max_g k_g`` have
      their leading ``max_g k_g - k_g`` events dropped to keep columns
      aligned across groups. Weights are dropped to match.

    Parameters
    ----------
    p_attr : list/tuple of array-like
        Length-A list of ``K_a x N`` per-attribute value matrices, with
        ``K_a = 1`` for every attribute. Same convention as
        :func:`build_exp_tens` but restricted to the single-slot case:
        within-event slot exchangeability does not license the cross-
        event slot correspondence that column-wise differencing
        imposes, so attributes with ``K_a != 1`` raise
        :class:`ValueError`. For voice-leading or step-size analyses,
        encode each voice as its own ``K_a = 1`` attribute in a shared
        group, difference that, then (optionally) stack the
        differenced attributes into a single multi-slot attribute
        before :func:`build_exp_tens`.
    w : None, scalar, or list/tuple
        Weights. ``None``, a scalar, or a length-A list of per-attribute
        weight inputs (each ``None``, scalar, 1-D, or 2-D). Same
        convention as :func:`build_exp_tens`.
    groups : array-like or list-of-lists or None
        Group assignment. ``None`` treats each attribute as its own
        singleton group; a length-A index vector, or a cell/list of
        attribute-index lists, gives explicit groupings. Matches
        :func:`build_exp_tens`.
    diff_orders : array-like of int
        Length-G vector of per-group differencing orders (non-negative
        integers). Order 0 leaves the group unchanged.
    periods : array-like of float
        Length-G vector of periods for shortest-arc wrapping of
        differences. An entry of 0 (or negative) means the group is
        treated as non-periodic and differences are left unwrapped.

    Returns
    -------
    p_attr_diff : list of ndarray
        Length-A list of transformed per-attribute matrices, each
        ``K_a x N'``.
    w_diff : same general form as *w*
        Transformed weights under the rule described above. Shape
        mirrors *w*: ``None`` stays ``None``; a scalar stays a scalar
        when all groups share the same order, expanding to a length-A
        list of per-attribute scalars when orders vary; a length-A
        list stays a length-A list, with per-attribute event-dependent
        entries becoming ``(K_a, N')`` matrices and non-event-
        dependent entries keeping their input shape.

    See Also
    --------
    build_exp_tens
    """
    # --- Normalize p_attr to list of 2-D float arrays ---
    if not isinstance(p_attr, (list, tuple)):
        raise TypeError("p_attr must be a list/tuple of per-attribute matrices.")
    p_attr = [np.asarray(M, dtype=np.float64) for M in p_attr]
    for a, M in enumerate(p_attr):
        if M.ndim != 2:
            raise ValueError(
                f"Attribute {a} value matrix must be 2-D; got ndim={M.ndim}."
            )
    A = len(p_attr)
    if A == 0:
        raise ValueError("p_attr must contain at least one attribute.")

    # --- Enforce K_a = 1 per attribute ---
    # Event differencing requires every attribute to have exactly one
    # slot per event. Column-wise subtraction across adjacent events
    # imposes a cross-event slot correspondence (slot i at event n-1
    # paired with slot i at event n) that within-event slot
    # exchangeability does not license; for multi-slot attributes the
    # output would silently depend on an arbitrary slot-listing
    # choice. K_a = 0 (empty attribute) is also rejected. The
    # principled route for voice-leading or step-size analyses is to
    # encode each voice as its own K_a = 1 attribute in a shared
    # group, difference that, then (optionally) stack the differenced
    # attributes into a single multi-slot attribute before
    # build_exp_tens. See USER_GUIDE Section 3 (Event differencing).
    for a, M in enumerate(p_attr):
        K_a = M.shape[0]
        if K_a != 1:
            raise ValueError(
                f"Attribute {a} has K_a = {K_a}; event differencing "
                f"requires every attribute to have K_a = 1. Column-wise "
                f"differencing imposes a cross-event slot alignment that "
                f"within-event slot exchangeability does not license, so "
                f"multi-slot attributes are rejected; empty attributes "
                f"(K_a = 0) are rejected likewise. For voice-leading or "
                f"step-size analyses, encode each voice as a separate "
                f"K_a = 1 attribute in a shared group, call "
                f"difference_events on that, then (optionally) stack "
                f"the differenced attributes into a single multi-slot "
                f"attribute before build_exp_tens. See USER_GUIDE "
                f"Section 3 (Event differencing)."
            )

    n_events = p_attr[0].shape[1]
    for a, M in enumerate(p_attr):
        if M.shape[1] != n_events:
            raise ValueError(
                f"All attributes must share the same event count N. "
                f"Attribute 0 has N={n_events}; attribute {a} has N={M.shape[1]}."
            )

    # --- Canonicalize groups ---
    group_of_attr, _attrs_of_group, G = _canonicalise_groups(groups, A)

    # --- Validate diff_orders and periods ---
    diff_orders = np.asarray(diff_orders, dtype=np.intp).ravel()
    if diff_orders.size != G:
        raise ValueError(
            f"diff_orders must have length G = {G} (number of groups); "
            f"got length {diff_orders.size}."
        )
    if np.any(diff_orders < 0):
        raise ValueError("All entries of diff_orders must be non-negative.")

    periods = np.asarray(periods, dtype=np.float64).ravel()
    if periods.size != G:
        raise ValueError(
            f"periods must have length G = {G}; got length {periods.size}."
        )

    max_order = int(diff_orders.max()) if diff_orders.size > 0 else 0
    n_prime = n_events - max_order
    if n_prime < 1:
        raise ValueError(
            f"Differencing orders are too high for the input event count: "
            f"max(diff_orders) = {max_order} but N = {n_events}."
        )

    # --- Difference each attribute's value matrix ---
    p_attr_diff = []
    for a, M in enumerate(p_attr):
        g = int(group_of_attr[a])
        k = int(diff_orders[g])
        P = float(periods[g])
        # Apply k-fold differencing along axis 1, with optional wrapping
        # after each first-order difference.
        M_diff = M
        for _ in range(k):
            M_diff = M_diff[:, 1:] - M_diff[:, :-1]
            if P > 0:
                M_diff = np.mod(M_diff + P / 2, P) - P / 2
        # Drop leading events to align with N'.
        extra_drop = max_order - k
        if extra_drop > 0:
            M_diff = M_diff[:, extra_drop:]
        assert M_diff.shape[1] == n_prime
        p_attr_diff.append(M_diff)

    # --- Transform weights ---
    w_diff = _difference_weights(w, A, group_of_attr, diff_orders,
                                  n_events, n_prime)

    return p_attr_diff, w_diff


def _difference_weights(w, A, group_of_attr, diff_orders,
                        n_events, n_prime):
    """Transform weights under the difference-events convention.

    The weight of each difference event is the product of the weights
    of the ``k + 1`` input events on which the difference depends —
    a rolling product of width ``k + 1`` along the event axis,
    applied semantically under the toolbox's broadcast convention.
    Under the ``K_a = 1`` restriction on :func:`difference_events`
    inputs, valid per-attribute weight inputs are ``None``, scalar,
    or ``(1, N)`` / 1-D of length ``N``.
    """
    if w is None:
        return None

    # --- Top-level scalar ---
    if np.isscalar(w):
        c = float(w)
        orders_per_attr = np.array(
            [int(diff_orders[int(group_of_attr[a])]) for a in range(A)],
            dtype=np.int64,
        )
        if np.all(orders_per_attr == orders_per_attr[0]):
            # Uniform orders — shape preserved as a scalar.
            return c ** int(orders_per_attr[0] + 1)
        # Varying orders — emit a per-attribute list.
        return [c ** int(k + 1) for k in orders_per_attr]

    if not isinstance(w, (list, tuple)):
        raise TypeError(
            "w must be None, a scalar, or a list/tuple of per-attribute "
            "weight inputs."
        )
    if len(w) != A:
        raise ValueError(
            f"Weight list must have length A = {A}; got length {len(w)}."
        )

    max_order = int(np.max(diff_orders)) if A > 0 else 0
    w_diff = []
    for a, wa in enumerate(w):
        g = int(group_of_attr[a])
        k = int(diff_orders[g])
        if not _weight_has_event_dependence(wa, n_events, a):
            # No event dependence — rolling product of a constant
            # reduces to raising each entry to power k + 1. None
            # stays None; a scalar stays a scalar.
            w_diff.append(_raise_no_event_dep(wa, k + 1))
            continue
        # Event-dependent: coerce to (1, N) then take a rolling
        # product of width k + 1 along the event axis.
        W = np.asarray(wa, dtype=np.float64).reshape(1, n_events)
        if k > 0:
            W = _rolling_product(W, k + 1)
        extra_drop = max_order - k
        if extra_drop > 0:
            W = W[:, extra_drop:]
        assert W.shape[1] == n_prime
        w_diff.append(W)
    return w_diff


def _raise_no_event_dep(wa, p: int):
    """Raise a non-event-dependent weight input to power *p*.

    Under the ``K_a = 1`` restriction on :func:`difference_events`
    inputs, the non-event-dependent inputs that reach this helper
    are limited to ``None``, Python scalars, 0-D arrays, and size-1
    1-D arrays.
    """
    if wa is None:
        return None
    if p == 1:
        return wa  # fast path: order 0 groups
    if np.isscalar(wa):
        return float(wa) ** int(p)
    # 0-D or size-1 1-D array: coerce to Python scalar for consistency
    # with the scalar branch.
    return float(np.asarray(wa).item()) ** int(p)


def _rolling_product(W, width):
    """Rolling product of width *width* along the last axis.

    Output shape: (K, N - width + 1). Output column *i* is
    ``prod(W[:, i : i + width], axis=1)``.
    """
    K, N = W.shape
    n_out = N - width + 1
    if n_out < 1:
        raise ValueError(
            f"Rolling-product width {width} exceeds event count {N}."
        )
    out = np.empty((K, n_out), dtype=np.float64)
    for i in range(n_out):
        out[:, i] = np.prod(W[:, i:i + width], axis=1)
    return out


def _weight_has_event_dependence(wa, N, attr_idx=None):
    """True iff *wa*'s shape carries the N axis.

    Under the ``K_a = 1`` restriction on :func:`difference_events`
    inputs, valid per-attribute weight shapes are ``None``, scalar,
    or ``(1, N)`` / 1-D of length ``N``.
    """
    if wa is None:
        return False
    arr = np.asarray(wa)
    if arr.size == 1:
        return False
    if arr.ndim == 1 and arr.size == N:
        return True
    if arr.ndim == 2 and arr.shape == (1, N):
        return True
    where = (
        f"Attribute {attr_idx} weight" if attr_idx is not None else "Weight"
    )
    raise ValueError(
        f"{where} has shape {arr.shape}; under the K_a = 1 restriction, "
        f"expected None, scalar, or (1, {N})."
    )


# ===================================================================
#  bind_events: sliding-window binding into n-attribute super-events
# ===================================================================


def bind_events(p, w=None, n=2, *, circular=False):
    """Bind n consecutive events into n-attribute super-events.

    Cross-event preprocessing helper for multi-attribute tensor input.
    Slides a window of width *n* across an event sequence and emits
    each window as an *n*-attribute super-event whose *j*-th
    attribute holds the value(s) at lag ``j-1`` (``j = 1, ..., n``).
    The output is a length-*n* list of ``(K_a, N')`` arrays, suitable
    for direct use as the ``p_attr`` argument of
    :func:`build_exp_tens` with all *n* attributes assigned to a
    single group.

    This complements :func:`difference_events`: differencing
    aggregates *across* event boundaries (collapsing ``k+1``
    consecutive events into a single value), while binding
    aggregates *within* a window (gathering *n* consecutive values
    into a single super-event with *n* separate slot attributes). The
    two operations compose naturally: differencing then binding gives
    *n*-tuples of consecutive step sizes, recovering the *n*-tuple
    entropy of Milne & Dean (2016) as a special case (``sigma -> 0``,
    integer-step grid, uniform weights, periodic domain) while
    extending it to non-zero ``sigma``, continuous-valued steps,
    non-periodic domains, and per-event weights propagated through
    both stages. The bound MAET is itself a density that can be fed
    into the rest of the toolbox: pairs of bound MAETs can be
    compared via :func:`cos_sim_exp_tens`, queried via
    :func:`windowed_similarity`, and so on.

    Lag slots are emitted as separate (``K_a``-valued) attributes,
    one per lag, rather than packed into a single attribute, because
    lag identity is not exchangeable: the value at lag *j* and the
    value at lag *j+1* carry distinct positional meanings within the
    bound super-event. By contrast, ``K_a > 1`` inputs (multi-value
    attributes whose slots are deliberately exchangeable) are
    permitted: each output attribute then carries the ``K_a`` slot
    values of one underlying event, and the within-attribute
    exchangeability is preserved per output attribute. Cross-event
    slot alignment is never imposed, because the cross-event
    structure is between output attributes, not within.

    ``N' = N - n + 1`` (default) or ``N`` (when *circular* is True).

    Parameters
    ----------
    p : array-like
        Event values: a ``(K_a, N)`` 2-D array, a ``(1, N)`` row, a
        length-*N* 1-D array (interpreted as ``K_a = 1``), or a
        length-1 list/tuple wrapping one of those (the wrapped form
        is accepted for symmetry with the output of
        :func:`difference_events`). ``K_a >= 1``.
    w : None, scalar, or array-like
        Weights. ``None``, scalar, length-*N* 1-D, ``(1, N)`` row,
        ``(K_a, 1)`` column, or ``(K_a, N)`` matrix. May also be a
        length-1 list/tuple wrapping any of those (matching the *p*
        form).
    n : int
        Window size (positive integer; default 2).
    circular : bool, keyword-only
        When True, the window wraps around the end of the sequence;
        ``N' = N``. When False (default), ``N' = N - n + 1``.

        The *circular* flag describes the *event sequence* (whether
        the last event connects back to the first), and is
        independent of the *positional periodicity* set in
        :func:`build_exp_tens` via its ``is_per`` / ``period``
        arguments. Both combinations are meaningful: a non-circular
        sequence on a periodic domain (a non-cyclic motif living in
        pitch-class space), and a circular sequence on a linear
        domain (a cyclic rhythm represented in linear time, e.g.,
        for windowed analysis). The two flags are orthogonal.

    Returns
    -------
    p_bound : list of ndarray
        Length-*n* list of ``(K_a, N')`` arrays. ``p_bound[j]``
        contains the value(s) at lag *j* for each window. For
        ``K_a = 1`` input each entry is ``(1, N')``.
    w_bound : None, scalar, or list of ndarray
        Per-attribute weight propagation, in the form that
        :func:`build_exp_tens` accepts directly:

        - ``None`` stays ``None``.
        - A scalar ``c`` stays ``c`` (broadcast in
          :func:`build_exp_tens`).
        - A ``(1, N)`` / length-*N* row becomes a length-*n* list of
          ``(1, N')`` rows.
        - A ``(K_a, 1)`` column becomes a length-*n* list of
          ``(K_a, 1)`` columns.
        - A ``(K_a, N)`` matrix becomes a length-*n* list of
          ``(K_a, N')`` matrices.

        The end-to-end numerics are equivalent to a rolling product
        of slot weights: each output attribute inherits the slot
        weights of the underlying event at its lag, and
        :func:`build_exp_tens` multiplies across attributes during
        tuple enumeration.

    Examples
    --------
    >>> # 2-tuple entropy of step sizes (diatonic scale, sigma -> 0)
    >>> import numpy as np
    >>> from mpt import (difference_events, bind_events, build_exp_tens,
    ...                   entropy_exp_tens)
    >>> p = np.array([[0, 2, 4, 5, 7, 9, 11]], dtype=float)
    >>> d = difference_events([p], None, None, [1], [12])
    >>> p_bound, w_bound = bind_events(d[0], None, 2, circular=True)
    >>> T = build_exp_tens(p_bound, w_bound, [1e-6], [1, 1], [1, 1],
    ...                     [False], [True], [12], verbose=False)
    >>> H = entropy_exp_tens(T, normalize=False)

    See Also
    --------
    difference_events
    build_exp_tens
    entropy_exp_tens
    cos_sim_exp_tens
    n_tuple_entropy
    """
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError(f"n must be a positive integer; got {n!r}.")
    n = int(n)

    # --- Unwrap length-1 list/tuple (symmetry with difference_events) ---
    input_was_list_p = isinstance(p, (list, tuple))
    if input_was_list_p:
        if len(p) != 1:
            raise ValueError(
                f"p as a list/tuple must contain exactly one attribute; "
                f"got {len(p)}. To bind multiple attributes, call "
                f"bind_events on each separately."
            )
        p = p[0]

    input_was_list_w = isinstance(w, (list, tuple))
    if input_was_list_w:
        if len(w) != 1:
            raise ValueError(
                f"w as a list/tuple must contain exactly one entry; "
                f"got {len(w)}."
            )
        w = w[0]

    # --- Validate p shape (allow any K_a >= 1) ---
    arr = np.asarray(p, dtype=np.float64)
    if arr.ndim == 1:
        p_mat = arr.reshape(1, -1)            # length-N -> (1, N)
    elif arr.ndim == 2:
        p_mat = arr
    else:
        raise ValueError(
            f"p has shape {arr.shape}; bind_events requires p to be a "
            f"length-N 1-D array or a (K_a, N) 2-D array."
        )
    Ka = p_mat.shape[0]
    N = p_mat.shape[1]

    # --- Determine window indices ---
    if circular:
        if n > N:
            raise ValueError(
                f"Circular window size n = {n} exceeds event count "
                f"N = {N}."
            )
        n_prime = N
        idx = (np.arange(n_prime)[:, None] + np.arange(n)[None, :]) % N
    else:
        n_prime = N - n + 1
        if n_prime < 1:
            raise ValueError(
                f"Window size n = {n} exceeds event count N = {N} "
                f"(non-circular mode)."
            )
        idx = np.arange(n_prime)[:, None] + np.arange(n)[None, :]

    # --- Build the n lag matrices, preserving K_a ---
    p_bound = [p_mat[:, idx[:, j]].copy() for j in range(n)]

    # --- Weights: per-attribute propagation ---
    #
    # Each output attribute inherits the slot weights of the
    # underlying event at its lag. build_exp_tens multiplies across
    # attributes during tuple enumeration, so the end-to-end weight
    # of a bound super-event equals the product of the n constituent
    # events' weights (the same numerical contribution as the prior
    # eager rolling product, for K_a = 1; the natural generalisation
    # for K_a > 1).

    if w is None:
        w_bound = None
    elif np.isscalar(w):
        # Scalar weight broadcasts in build_exp_tens; pass through.
        w_bound = float(w)
    else:
        w_arr = np.asarray(w, dtype=np.float64)

        # Scalar packed in a 0-D or single-element array
        if w_arr.size == 1:
            w_bound = float(w_arr.item())
        else:
            # Coerce to a 2-D matrix consistent with the p-shape.
            if w_arr.ndim == 1:
                if w_arr.size == N:
                    w_mat = w_arr.reshape(1, -1)         # (1, N)
                elif w_arr.size == Ka and Ka != N:
                    w_mat = w_arr.reshape(-1, 1)         # (K_a, 1)
                elif Ka == N:
                    # Ambiguous: prefer the row interpretation, matching
                    # the v2.1-and-earlier convention for length-N input.
                    w_mat = w_arr.reshape(1, -1)
                else:
                    raise ValueError(
                        f"w as a 1-D array must have length N = {N} or "
                        f"K_a = {Ka} (got length {w_arr.size})."
                    )
            elif w_arr.ndim == 2:
                w_mat = w_arr
            else:
                raise ValueError(
                    f"w has shape {w_arr.shape}; expected None, scalar, "
                    f"1-D, or 2-D."
                )

            r, c = w_mat.shape
            if r == 1 and c == N:
                w_bound = [w_mat[:, idx[:, j]].copy() for j in range(n)]
            elif r == Ka and c == 1:
                w_bound = [w_mat.copy() for _ in range(n)]
            elif r == Ka and c == N:
                w_bound = [w_mat[:, idx[:, j]].copy() for j in range(n)]
            else:
                raise ValueError(
                    f"w has shape {w_mat.shape}; expected None, scalar, "
                    f"length-{N} 1-D, ({1}, {N}) row, "
                    f"({Ka}, 1) column, or ({Ka}, {N}) matrix."
                )

    # --- Re-wrap weight in a length-1 list if input was list ---
    if input_was_list_w:
        w_bound = [w_bound]

    return p_bound, w_bound


# ===================================================================
#  Windowing: window_tensor, windowed_similarity, and supporting math
# ===================================================================


def window_tensor(dens, window_spec):
    """Wrap a MaetDensity with a post-tensor window specification.

    Returns a :class:`WindowedMaetDensity` that bundles the underlying
    density with a window spec. No math is performed at construction
    time; the window is applied lazily by :func:`eval_exp_tens` and
    :func:`cos_sim_exp_tens`. See the MAET specification §4.3.

    Parameters
    ----------
    dens : MaetDensity
        The density to be windowed.
    window_spec : dict
        Window specification with keys:

        ``size`` : scalar or length-G array
            Per-group window effective standard deviation in multiples
            of that group's ``sigma``. NaN or Inf means the group is
            not windowed. A scalar is broadcast across all groups.
        ``mix`` : scalar or length-G array
            Per-group shape parameter in [0, 1]: 0 = pure Gaussian,
            1 = pure rectangular, in between = rectangular-convolved-
            with-Gaussian. A scalar is broadcast.
        ``centre`` : length-A list of array-like, or a single array-like
            Per-attribute centre coordinates. Each entry has length
            ``dim_per_attr[a]``. A single 1-D array of total length
            ``dim`` is split across attributes in order. Entries whose
            attribute's group has ``size`` NaN/Inf are ignored.

    Returns
    -------
    WindowedMaetDensity
    """
    if not isinstance(dens, MaetDensity):
        raise TypeError(
            f"window_tensor expected a MaetDensity; got {type(dens).__name__}."
        )
    if not isinstance(window_spec, dict):
        raise TypeError("window_spec must be a dict.")

    A = dens.n_attrs
    G = dens.n_groups
    dim_per = dens.dim_per_attr
    dim_total = int(dens.dim)

    # --- size ---
    size_arr = np.asarray(window_spec.get("size"), dtype=np.float64).ravel()
    if size_arr.size == 1:
        size_arr = np.full(G, float(size_arr.item()))
    if size_arr.size != G:
        raise ValueError(
            f"window_spec['size'] must be a scalar or length-{G} array; "
            f"got length {size_arr.size}."
        )

    # --- mix ---
    mix_arr = np.asarray(window_spec.get("mix"), dtype=np.float64).ravel()
    if mix_arr.size == 1:
        mix_arr = np.full(G, float(mix_arr.item()))
    if mix_arr.size != G:
        raise ValueError(
            f"window_spec['mix'] must be a scalar or length-{G} array; "
            f"got length {mix_arr.size}."
        )
    if np.any((mix_arr < 0) | (mix_arr > 1)):
        raise ValueError("window_spec['mix'] entries must be in [0, 1].")

    # --- centre ---
    centre_in = window_spec.get("centre")
    if centre_in is None:
        # Default: centre at origin for all attributes.
        centre_list = [np.zeros(int(dim_per[a]), dtype=np.float64)
                       for a in range(A)]
    elif isinstance(centre_in, (list, tuple)) and not (
            len(centre_in) > 0 and np.isscalar(centre_in[0])):
        if len(centre_in) != A:
            raise ValueError(
                f"window_spec['centre'] (list form) must have length A = {A}; "
                f"got length {len(centre_in)}."
            )
        centre_list = []
        for a, ca in enumerate(centre_in):
            arr = np.asarray(ca, dtype=np.float64).ravel()
            if arr.size != int(dim_per[a]):
                raise ValueError(
                    f"window_spec['centre'][{a}] must have length "
                    f"{int(dim_per[a])}; got length {arr.size}."
                )
            centre_list.append(arr)
    else:
        # Flat array; split by dim_per_attr.
        flat = np.asarray(centre_in, dtype=np.float64).ravel()
        if flat.size != dim_total:
            raise ValueError(
                f"window_spec['centre'] (flat form) must have length "
                f"dim = {dim_total}; got length {flat.size}."
            )
        centre_list = []
        offset = 0
        for a in range(A):
            da = int(dim_per[a])
            centre_list.append(flat[offset:offset + da].astype(np.float64))
            offset += da

    return WindowedMaetDensity(
        tag="WindowedMaetDensity",
        dens=dens,
        size=size_arr,
        mix=mix_arr,
        centre=centre_list,
    )


# -------------------------------------------------------------------
#  Window pointwise evaluation (for eval_exp_tens dispatch)
# -------------------------------------------------------------------


def _is_windowed_group(size_g, mix_g):
    """True iff the group has an effective window (size finite and > 0)."""
    return np.isfinite(size_g) and size_g > 0


def _window_width_params(size_g, mix_g, sigma_g):
    """Derive (a, b) for a group: rect half-width a and Gaussian width b.

    Window is rectangular-of-half-width-a convolved with Gaussian-of-
    width-b:
        a = size_g * sigma_g * sqrt(3 * mix_g)
        b = size_g * sigma_g * sqrt(1 - mix_g)
    """
    s = float(size_g) * float(sigma_g)
    a = s * np.sqrt(3.0 * float(mix_g))
    b = s * np.sqrt(1.0 - float(mix_g))
    return a, b


def _window_factor_1d(u, a, b):
    """Evaluate the 1-D window function at *u*.

    Window is rectangular-of-half-width-a convolved with Gaussian-of-
    width-b. Returns the convolution value at u.

    Cases:
      - b == 0 (pure rectangular): indicator(|u| <= a), values in {0, 1}.
      - a == 0 (pure Gaussian): exp(-u^2 / (2 b^2)).
      - otherwise: 0.5 * [erf((a + u)/(sqrt(2)*b)) + erf((a - u)/(sqrt(2)*b))].
    """
    u = np.asarray(u, dtype=np.float64)
    if b == 0.0:
        return (np.abs(u) <= a).astype(np.float64)
    if a == 0.0:
        return np.exp(-u**2 / (2.0 * b**2))
    from scipy.special import erf
    arg_plus  = (a + u) / (np.sqrt(2.0) * b)
    arg_minus = (a - u) / (np.sqrt(2.0) * b)
    return 0.5 * (erf(arg_plus) + erf(arg_minus))


def _evaluate_window_on_query(wmd: "WindowedMaetDensity", x_list):
    """Evaluate the window W(x) at query points.

    Parameters
    ----------
    wmd : WindowedMaetDensity
    x_list : list of (dim_per_attr[a], nQ) arrays
        Per-attribute query matrices, one per attribute, with n_Q
        query points.

    Returns
    -------
    ndarray of shape (nQ,)
        Window values at each query point.
    """
    dens = wmd.dens
    A = dens.n_attrs
    group_of = dens.group_of_attr
    dim_per = dens.dim_per_attr
    sigma_g = dens.sigma

    n_q = x_list[0].shape[1] if A > 0 else 0
    result = np.ones(n_q, dtype=np.float64)
    for a in range(A):
        g = int(group_of[a])
        if not _is_windowed_group(wmd.size[g], wmd.mix[g]):
            continue
        a_, b_ = _window_width_params(wmd.size[g], wmd.mix[g], sigma_g[g])
        da = int(dim_per[a])
        # Per-axis factor, multiplied across axes within the attribute.
        centre_a = wmd.centre[a]  # shape (da,)
        x_a = x_list[a]           # shape (da, nQ)
        u = x_a - centre_a[:, None]  # (da, nQ)
        for i in range(da):
            result = result * _window_factor_1d(u[i], a_, b_)
    return result


# -------------------------------------------------------------------
#  windowed_similarity and closed-form pair-factor evaluator
# -------------------------------------------------------------------


class WindowedSimilarityPeriodicApproxWarning(UserWarning):
    """Issued when :func:`windowed_similarity` applies the line-case
    closed-form windowed inner product to a periodic group whose
    window standard deviation lambda*sigma is at least one quarter of
    the period P. The closed form in use is the small-window
    approximation; it degrades as window support approaches one
    period. For windows larger than a period the windowed inner
    product collapses to the unwindowed form, which can be obtained
    directly from :func:`cos_sim_exp_tens`. See User Guide §3.1
    "Post-tensor windowing" for the three-regime analysis.

    A module-level filter registers this warning with the ``"always"``
    action so that it fires on every offending call (rather than once
    per location, which is the default for :class:`UserWarning`),
    matching the MATLAB ``warning(id, ...)`` behaviour. Suppress it,
    if you have determined the approximation is acceptable for your
    use case, with::

        import warnings
        from mpt import WindowedSimilarityPeriodicApproxWarning
        warnings.filterwarnings(
            "ignore",
            category=WindowedSimilarityPeriodicApproxWarning,
        )
    """


# Ensure this warning is always shown, not only on the first
# occurrence at a given (module, lineno). Matches the per-call
# warning behaviour of the MATLAB implementation.
warnings.filterwarnings(
    "always", category=WindowedSimilarityPeriodicApproxWarning
)


def windowed_similarity(dens_query, dens_context, window_spec, offsets, *,
                        reference=None, verbose: bool = True):
    """Sliding-window similarity profile (cross-correlation).

    For each offset column, *dens_context* is windowed with
    *window_spec* at the corresponding centre, and its similarity
    against *dens_query* (unwindowed) is computed. The normaliser
    uses the unwindowed L2 norms of both operands (Option Z in the
    spec).

    Note on naming
    --------------
    This function was named ``windowed_cos_sim`` in earlier drafts.
    The output is a magnitude-aware *windowed similarity*: because
    the denominator uses unwindowed L2 norms (rather than the
    windowed norm of the context), the profile is not bounded in
    [-1, 1] across sweep positions and does not correspond to an
    inner product on a single Hilbert space. This is the intended
    behaviour for sliding-motif analysis -- a dense local match
    should outscore a sparse one -- but it means "cosine similarity"
    is not the right name for the object. The strict shape-only
    cosine similarity (with windowed denominator) is reserved as a
    separate notion in the manuscript and is not currently
    implemented in the toolbox. See manuscript §5.4.

    Reference-point semantics
    -------------------------
    Offsets are measured from a reference point to the window centre
    on each windowed attribute. Two options are provided:

      * Default (``reference=None``): the reference on each attribute
        is the unweighted column mean of the query's tuple centres.
        A purely geometric property of the tuple centres, independent
        of the tuple weights.

      * User-supplied (``reference`` given): one vector per query
        attribute, of length equal to that attribute's dimension. The
        reference does not depend on the query.

    The peak offset under either option equals ``P* - ref``, where
    ``P*`` is the window centre (in context coordinates) at which the
    profile peaks. Peak offsets under the default therefore track the
    quantity ``P* - mu_q`` across between-query variation; peak
    offsets under a fixed reference track ``P*`` directly.

    The choice matters most when a pitch attribute has more than one
    slot per event (chords with exchangeable voices, or partials added
    by ``add_spectra``), because queries can then vary in slot count,
    slot values, and slot weights. For slot-weight sweeps the two
    options coincide. For slot-value sweeps (e.g., stretching
    partials), the default's peak offset drifts while a fixed
    reference's stays put. For slot-count sweeps, the default's peak
    offset is stable only for harmonic queries -- those whose slots
    lie at (or close to) integer-harmonic values ``f_e + 1200*log2(n)``
    cents. See User Guide §3.1 "Post-tensor windowing" and the
    ``demo_windowing_reference`` demo for analysis and worked examples.

    In both cases, a peak at offset ``delta`` means the context has
    similarity-relevant structure at ``reference + delta``.

    Periodic groups
    ---------------
    The closed-form windowed inner product implemented here is the
    line-case formula -- exact for non-periodic groups, but only an
    approximation when applied to a periodic group whose window
    support is comparable to one period. For windows larger than a
    period the windowed inner product collapses to the unwindowed
    form, which can be obtained directly from
    :func:`cos_sim_exp_tens`. A unified closed form for the
    intermediate regime (a finite sum over periodic images) is left
    to future work. When this function is called on a windowed
    periodic group, a
    :class:`WindowedSimilarityPeriodicApproxWarning` is emitted on
    every call, with a brief informational message within the
    recommended bound ``lambda*sigma <= P/(2*sqrt(3))`` and a stronger
    message past the bound. See User Guide §3.1 "Post-tensor
    windowing" for the three-regime analysis and the warning
    specification.

    Parameters
    ----------
    dens_query : MaetDensity
        The query density (not windowed).
    dens_context : MaetDensity
        The context density to be windowed.
    window_spec : dict
        Window specification (see :func:`window_tensor`). Only the
        ``size`` and ``mix`` fields are read; any ``centre`` field is
        ignored (offsets replace it).
    offsets : (dim, M) array-like
        Per-sweep offsets in effective space, using the
        attribute-concatenated flat convention of
        :func:`window_tensor`. A 1-D array is accepted when dim == 1.
    reference : list of array-like, optional
        One entry per query attribute, each a 1-D array of length
        equal to that attribute's dimension. If given, overrides the
        default unweighted-centroid reference. If ``None`` (default),
        the unweighted mean of per-attribute tuple centres is used.
    verbose : bool

    Returns
    -------
    ndarray of shape (M,)
        Windowed similarity profile.
    """
    if not isinstance(dens_query, MaetDensity):
        raise TypeError("dens_query must be a MaetDensity.")
    if not isinstance(dens_context, MaetDensity):
        raise TypeError("dens_context must be a MaetDensity.")

    offsets = np.asarray(offsets, dtype=np.float64)
    if offsets.ndim == 1:
        offsets = offsets.reshape(-1, 1)
    if offsets.shape[0] != int(dens_context.dim):
        raise ValueError(
            f"offsets must have {int(dens_context.dim)} rows (dim of "
            f"dens_context); got shape {offsets.shape}."
        )
    M = offsets.shape[1]

    # ---- Periodic-window approximation check ------------------------
    # A WindowedSimilarityPeriodicApproxWarning is emitted per periodic
    # windowed group on every call. The message has two forms:
    #   - Within the recommended bound (lambda*sigma <= P/(2*sqrt(3))):
    #     a brief informational notice that the line-case approximation
    #     is in use, with the current SD/P against the bound. The
    #     approximation is sub-percent across the window shape family
    #     within this bound.
    #   - Past the bound (lambda*sigma > P/(2*sqrt(3))): a stronger
    #     notice reporting SD/P and phi (rect half-width) against
    #     their bounds, and describing the qualitative behaviour by
    #     mix (at mix=1 the rect window is no longer localized on the
    #     circle; at mix=0 the approximation degrades smoothly).
    # See User Guide §3.1 "Post-tensor windowing".
    SD_OVER_P_BOUND = 1.0 / (2.0 * np.sqrt(3.0))   # ~= 0.2887
    G = int(dens_context.n_groups)
    size_arr = np.atleast_1d(
        np.asarray(window_spec["size"], dtype=np.float64)
    ).ravel()
    if size_arr.size == 1:
        size_arr = np.repeat(size_arr, G)
    mix_arr = np.atleast_1d(
        np.asarray(window_spec["mix"], dtype=np.float64)
    ).ravel()
    if mix_arr.size == 1:
        mix_arr = np.repeat(mix_arr, G)
    for g in range(G):
        if not bool(dens_context.is_per[g]):
            continue
        lam = float(size_arr[g])
        if not np.isfinite(lam) or lam <= 0:
            continue
        period_g = float(dens_context.period[g])
        if period_g <= 0:
            continue
        eff_sigma = lam * float(dens_context.sigma[g])
        sd_over_p = eff_sigma / period_g
        gamma_g = float(mix_arr[g])

        if sd_over_p <= SD_OVER_P_BOUND:
            # Within-bound: brief informational form.
            msg = (
                f"Periodic windowed inner product on group {g} "
                f"applies the line-case formula at wrapped "
                f"differences -- an approximation that retains "
                f"only the leading periodic image of the window. "
                f"Within the recommended bound, the approximation "
                f"is sub-percent across the window shape family.\n"
                f"  Window SD (lambda*sigma) = {eff_sigma:g}\n"
                f"  Period P                 = {period_g:g}\n"
                f"  SD/P                     = {sd_over_p:.4f}\n"
                f"  Recommended bound (SD/P) = {SD_OVER_P_BOUND:.4f} "
                f"(= 1/(2*sqrt(3)))\n"
                f"See User Guide \u00a73.1 \"Post-tensor windowing\". "
                f"Suppress with warnings.filterwarnings('ignore', "
                f"category=mpt."
                f"WindowedSimilarityPeriodicApproxWarning)."
            )
        else:
            # Past-bound: stronger form, with phi and per-mix
            # behaviour.
            phi_g = eff_sigma * np.sqrt(3.0 * max(gamma_g, 0.0))
            msg = (
                f"Window SD exceeds the recommended bound for "
                f"periodic group {g}; the line-case approximation "
                f"is no longer reliable.\n"
                f"  Window SD (lambda*sigma) = {eff_sigma:g}\n"
                f"  Period P                 = {period_g:g}\n"
                f"  SD/P                     = {sd_over_p:.4f}  "
                f"(bound: {SD_OVER_P_BOUND:.4f})\n"
                f"  phi (rect half-width)    = {phi_g:g}  "
                f"(bound: {period_g / 2:g} = P/2)\n"
                f"  mix (gamma)              = {gamma_g:g}\n"
                f"Beyond the bound, behaviour depends on mix:\n"
                f"  mix = 1 (pure rect):     window is no longer "
                f"localized on the circle (pointless as a window).\n"
                f"  mix = 0 (pure Gaussian): line-case approximation "
                f"degrades smoothly; error grows with SD/P.\n"
                f"  intermediate mix:        between these two cases.\n"
                f"Reduce size or sigma so that lambda*sigma <= "
                f"P/(2*sqrt(3)). See User Guide \u00a73.1 "
                f"\"Post-tensor windowing\"."
            )

        warnings.warn(
            msg,
            WindowedSimilarityPeriodicApproxWarning,
            stacklevel=2,
        )

    # Reference point, per attribute.
    A = int(dens_query.n_attrs)
    dim_per_a = [int(d) for d in dens_query.dim_per_attr]
    if reference is None:
        # Default: unweighted mean of per-attribute tuple centres.
        ref_per_a = [dens_query.centres[a].mean(axis=1) for a in range(A)]
    else:
        if len(reference) != A:
            raise ValueError(
                f"reference must have {A} entries (one per query "
                f"attribute); got {len(reference)}."
            )
        ref_per_a = []
        for a in range(A):
            ref_a = np.asarray(reference[a], dtype=np.float64).reshape(-1)
            if ref_a.size != dim_per_a[a]:
                raise ValueError(
                    f"reference[{a}] must have length "
                    f"{dim_per_a[a]} (dim of attribute {a}); got "
                    f"{ref_a.size}."
                )
            ref_per_a.append(ref_a)

    # Strip any user-supplied 'centre' field; offsets replace it.
    base_spec = {k: v for k, v in window_spec.items() if k != "centre"}

    profile = np.empty(M, dtype=np.float64)
    for m in range(M):
        off_col = offsets[:, m]
        centre_list = []
        off_ptr = 0
        for a in range(A):
            da = dim_per_a[a]
            centre_list.append(ref_per_a[a] + off_col[off_ptr:off_ptr + da])
            off_ptr += da
        spec_m = dict(base_spec)
        spec_m["centre"] = centre_list
        wmd = window_tensor(dens_context, spec_m)
        profile[m] = cos_sim_exp_tens(dens_query, wmd, verbose=verbose)
    return profile


# -------------------------------------------------------------------
#  Cos-sim dispatch extension: unwindowed × windowed
# -------------------------------------------------------------------


def _cos_sim_exp_tens_windowed(dens_a, dens_b, *, verbose: bool):
    """Cosine similarity with one or both operands windowed.

    Normaliser uses unwindowed L2 norms for both operands (Option Z).
    Numerator uses the windowed per-group pair factor V_g (replacing
    U_g) for windowed groups.

    Currently supports one-sided windowing (exactly one of dens_a,
    dens_b is a WindowedMaetDensity). Two-sided is not needed for the
    windowed_similarity use case.
    """
    a_win = isinstance(dens_a, WindowedMaetDensity)
    b_win = isinstance(dens_b, WindowedMaetDensity)
    if a_win and b_win:
        raise NotImplementedError(
            "Two-sided windowing (both operands windowed) is not supported "
            "in v2.1.0. Use windowed_similarity for profile sweeps."
        )

    if a_win:
        # Swap: put windowed operand on the 'b' side canonically.
        dens_q, wmd = dens_b, dens_a
    else:
        dens_q, wmd = dens_a, dens_b

    dens_c = wmd.dens

    # --- Structural compatibility checks (delegate to underlying _ma) ---
    _check_ma_compatibility(dens_q, dens_c)

    # --- Unwindowed norms (denominator) ---
    ip_qq = _cos_sim_numerator_ma(dens_q, dens_q, windowed_c=None)
    ip_cc = _cos_sim_numerator_ma(dens_c, dens_c, windowed_c=None)

    # --- Windowed numerator ---
    ip_qc = _cos_sim_numerator_ma(dens_q, dens_c, windowed_c=wmd)

    denom = np.sqrt(ip_qq * ip_cc)
    if denom == 0:
        return float("nan")
    return float(ip_qc / denom)


def _check_ma_compatibility(dens_x: MaetDensity, dens_y: MaetDensity):
    """Structural compatibility check, mirroring _cos_sim_exp_tens_ma."""
    if dens_x.n_attrs != dens_y.n_attrs:
        raise ValueError("Both densities must have the same n_attrs.")
    if not np.array_equal(dens_x.group_of_attr, dens_y.group_of_attr):
        raise ValueError("Both densities must have the same group_of_attr.")
    if not np.array_equal(dens_x.r, dens_y.r):
        raise ValueError("Both densities must have the same r.")
    if not np.array_equal(dens_x.sigma, dens_y.sigma):
        raise ValueError("Both densities must have the same sigma.")
    if not np.array_equal(dens_x.is_rel, dens_y.is_rel):
        raise ValueError("Both densities must have the same is_rel.")
    if not np.array_equal(dens_x.is_per, dens_y.is_per):
        raise ValueError("Both densities must have the same is_per.")
    per_mask = dens_x.is_per.astype(bool)
    if np.any(dens_x.period[per_mask] != dens_y.period[per_mask]):
        raise ValueError("Both densities must agree on periods of periodic groups.")


def _cos_sim_numerator_ma(dens_x: MaetDensity, dens_y: MaetDensity, *,
                          windowed_c):
    """Compute a single inner product <f_x, W f_y> as a sum over (j, k)
    pairs. If *windowed_c* is None, it's the plain unwindowed inner
    product (used for norms and for unwindowed comparisons).

    Implements the full (size, mix) family closed-form for 1-D groups
    and multi-D absolute groups, and the Gaussian-only case for multi-D
    relative groups. Raises if (rho > 0) is requested on a multi-D
    relative group.
    """
    A          = dens_x.n_attrs
    group_of   = dens_x.group_of_attr
    r_vec      = dens_x.r
    sigma_g    = dens_x.sigma
    is_rel_g   = dens_x.is_rel
    is_per_g   = dens_x.is_per
    period_g   = dens_x.period

    n_jx, n_kx = dens_x.n_j, dens_x.n_k
    n_jy, n_ky = dens_y.n_j, dens_y.n_k

    # Use perm-side for x, comb-side for y (matching existing convention).
    # Evaluate pair factors per group and accumulate logs.
    n_jy_side, n_ky_side = int(n_jx), int(n_ky)
    # log_kernel[j, k] = sum_g log V_g(mu_j^(x), mu_k^(y), c_g if windowed)
    log_kernel = np.zeros((n_jy_side, n_ky_side), dtype=np.float64)

    # Track the window-dependent prefactor log(prod_g D_g). If not windowed
    # in that group, contribution is 0.
    log_prefactor = 0.0

    # =====================================================================
    # Cross-correlation translation (windowed case only).
    #
    # When windowed_c is not None, windowed_similarity asks for the cosine
    # similarity between the unwindowed query (dens_x) and the windowed
    # context (dens_y, with window spec wmd) interpreted as a CROSS-
    # CORRELATION: at sweep centre c in a windowed group g, the query is
    # translated so that its effective-space "position" mu_q_g moves onto
    # the window centre c_g. A peak at c_g thus means the query pattern
    # is present in the context near c_g.
    #
    # Mathematically this is done by the coordinate substitution
    #
    #     cx_g  ->  cx_g  - mu_q_g
    #     cy_g  ->  cy_g  - c_g
    #     c_g   ->  0
    #
    # inside the windowed-factor integrand (so the existing closed-form
    # helper is reused verbatim), and by adding a per-attribute shift
    #
    #     delta_a = { (c_g - mu_q_g)|_a            if group g is absolute
    #               { [0, (c_g - mu_q_g)|_a_eff]   if group g is relative
    #                                               (slot-0-anchored lift)
    #
    # to the r_a-slot tuple differences D = U - V before computing Q_a.
    # Groups that are not windowed receive no shift.
    #
    # For the unwindowed path (windowed_c is None) this whole block is
    # skipped and cos_sim_exp_tens semantics are preserved exactly.
    # =====================================================================
    shift_per_attr = {}     # attr index -> (r_a,) float64 shift to add to D
    eff_shift_per_g = {}    # group index -> (d_g,) effective-space shift
    mu_q_per_g = {}         # group index -> (d_g,) effective-space query mean

    if windowed_c is not None:
        wmd = windowed_c
        attrs_of_g = dens_x.attrs_of_group
        dim_per = dens_x.dim_per_attr

        for g in range(dens_x.n_groups):
            if not _is_windowed_group(wmd.size[g], wmd.mix[g]):
                continue
            attrs_g = attrs_of_g[g]

            # mu_q_g: unweighted mean over perm-side tuple centres of
            # the query's effective-space Gaussian centres, concatenated
            # across attributes in g. The unweighted mean gives the
            # offset coordinate a weight-independent meaning (see User
            # Guide §3 on windowing). For relative groups this is
            # (approximately) zero by perm-symmetry, which is the
            # correct convention: translation-invariant groups have no
            # canonical position. Where the window centre in a relative
            # group is non-zero in effective space, the shift is still
            # applied and lifted via slot-0 anchoring.
            mu_q_parts = [dens_x.centres[a].mean(axis=1) for a in attrs_g]
            mu_q_g = np.concatenate(mu_q_parts)
            centre_g = np.concatenate([wmd.centre[a] for a in attrs_g])
            delta_g = centre_g - mu_q_g                          # (d_g,)

            mu_q_per_g[g] = mu_q_g
            eff_shift_per_g[g] = delta_g

            # Lift delta_g into per-attribute r_a-slot shifts.
            offset = 0
            g_is_rel = bool(is_rel_g[g])
            for a in attrs_g:
                a = int(a)
                r_a = int(r_vec[a])
                d_a = int(dim_per[a])
                delta_a_eff = delta_g[offset:offset + d_a]        # (d_a,)
                offset += d_a
                if g_is_rel:
                    # Slot-0 anchored lift: shift[0]=0, shift[1:]=delta_a_eff.
                    # For r_a == 1 and isRel=True the group is degenerate
                    # (d_a == 0) and no shift is needed.
                    if r_a == 1:
                        shift_a = np.zeros(1, dtype=np.float64)
                    else:
                        shift_a = np.concatenate(
                            ([0.0], np.asarray(delta_a_eff, dtype=np.float64))
                        )
                else:
                    # Absolute: effective dim equals r_a, direct mapping.
                    shift_a = np.asarray(delta_a_eff, dtype=np.float64).copy()
                if shift_a.shape[0] != r_a:
                    raise RuntimeError(
                        f"Internal: shift for attribute {a} has shape "
                        f"{shift_a.shape}, expected ({r_a},)."
                    )
                shift_per_attr[a] = shift_a

    # ---- Main D / Q loop (with cross-correlation shift applied). ----
    for a in range(A):
        g = int(group_of[a])
        r_a = int(r_vec[a])
        U = dens_x.u_perm[a]   # (r_a, nJ_x)
        V = dens_y.v_comb[a]   # (r_a, nK_y)
        D = U[:, :, None] - V[:, None, :]  # (r_a, nJ_x, nK_y)

        # Apply per-attribute cross-correlation shift before wrap / Q_a.
        if a in shift_per_attr:
            D = D + shift_per_attr[a][:, None, None]

        if is_per_g[g]:
            p_g = float(period_g[g])
            D = np.mod(D + p_g / 2, p_g) - p_g / 2

        Q_a = _compute_Q(D, r_a, bool(is_rel_g[g]), bool(is_per_g[g]),
                         float(period_g[g]))
        log_kernel = log_kernel - Q_a / (4.0 * float(sigma_g[g]) ** 2)

    # ---- Windowed-factor contributions per group (cross-correlation
    # substitution applied). ----
    if windowed_c is not None:
        wmd = windowed_c
        attrs_of_g = dens_x.attrs_of_group
        eff_x_perm = _effective_centres_from_U(dens_x, side="perm")
        eff_y_comb = _effective_centres_from_V(dens_y, side="comb")

        for g in range(dens_x.n_groups):
            if not _is_windowed_group(wmd.size[g], wmd.mix[g]):
                continue
            attrs_g = attrs_of_g[g]

            cx_list = [eff_x_perm[a] for a in attrs_g]   # each (d_a, nJ_x)
            cy_list = [eff_y_comb[a] for a in attrs_g]   # each (d_a, nK_y)
            cx_g = np.concatenate(cx_list, axis=0)        # (d_g, nJ_x)
            cy_g = np.concatenate(cy_list, axis=0)        # (d_g, nK_y)
            centre_g = np.concatenate([wmd.centre[a] for a in attrs_g])
            mu_q_g = mu_q_per_g[g]                        # (d_g,)

            # Cross-correlation coordinate substitution: translate query
            # centres to origin via mu_q_g, translate context centres to
            # origin via centre_g, then apply the window at 0.
            cx_sub = cx_g - mu_q_g[:, None]
            cy_sub = cy_g - centre_g[:, None]
            centre_sub = np.zeros_like(centre_g)

            s_g = wmd.size[g]
            mix_g = wmd.mix[g]
            sigma_gv = float(sigma_g[g])
            is_rel = bool(is_rel_g[g])
            d_g = cx_sub.shape[0]

            contrib, log_D = _windowed_group_contribution(
                cx_sub, cy_sub, centre_sub,
                s_g, mix_g, sigma_gv, is_rel,
                int(r_vec[int(attrs_g[0])]), d_g,
            )
            log_kernel = log_kernel + contrib
            log_prefactor = log_prefactor + log_D

    # Assemble numerator from log_kernel + prefactor.
    E = np.exp(log_kernel + log_prefactor)
    w_u = dens_x.w_j       # (nJ_x,)
    w_v = dens_y.wv_comb   # (nK_y,)
    return float(w_u @ (E @ w_v))


def _effective_centres_from_U(dens: "MaetDensity", side: str):
    """Return per-attribute effective-space centres of dens on the given side.

    ``side='perm'`` uses u_perm (r_a x nJ); ``side='comb'`` uses v_comb
    (r_a x nK). Effective-space centre is obtained by projecting the r
    slot values onto the group's effective (dim_per_attr) space:
      - absolute group: centre is the slot vector itself (dim = r).
      - relative group: centre is (r - 1)-dim reduced coords; compute
        via the first (r - 1) pairwise differences from the mean, or
        equivalently the identity-projector-on-zero-mean. The specific
        effective-space coordinates are already stored in
        dens.centres[a] (which uses the perm-side ordering).

    For now, only ``side='perm'`` is supported natively (returns
    dens.centres). For comb-side, we reconstruct from v_comb using the
    same reduction as the perm-side.
    """
    if side == "perm":
        return dens.centres
    raise NotImplementedError(
        "Only side='perm' is cached; comb-side centres are reconstructed "
        "elsewhere."
    )


def _effective_centres_from_V(dens: "MaetDensity", side: str):
    """Reconstruct effective-space centres from v_comb (comb-side).

    Uses the same reduction as build_exp_tens stores in ``dens.centres``
    for the perm side:
      - Absolute groups (isRel=False): effective-space centre is just
        v_comb[a] (r_a rows, used directly).
      - Relative groups (isRel=True) with r_a >= 2: effective-space
        centre is v_comb[a][1:, :] - v_comb[a][0:1, :], i.e., the (r_a - 1)
        differences of every slot from slot 0.
      - Relative groups with r_a = 1: empty (0, nK) array (degenerate).
    """
    A = dens.n_attrs
    group_of = dens.group_of_attr
    is_rel_g = dens.is_rel
    r_vec = dens.r
    n_k = dens.n_k

    out = []
    for a in range(A):
        g = int(group_of[a])
        r_a = int(r_vec[a])
        V = dens.v_comb[a]                     # (r_a, nK)
        if not is_rel_g[g]:
            out.append(V)
        elif r_a >= 2:
            out.append(V[1:, :] - V[0:1, :])    # (r_a - 1, nK)
        else:
            out.append(np.empty((0, n_k), dtype=np.float64))
    return out


def _windowed_group_contribution(cx_g, cy_g, centre_g,
                                  s_g, mix_g, sigma_g, is_rel, r_a, d_g):
    """Compute log(V_g / U_g) as a (nJ_x, nK_y) array, and log(D_g) scalar.

    cx_g : (d_g, nJ_x) effective-space perm-side centres for x in group g.
    cy_g : (d_g, nK_y) effective-space comb-side centres for y in group g.
    centre_g : (d_g,) window centre in group g's effective subspace.
    s_g : window size (in sigma multiples).
    mix_g : window mix in [0, 1].
    sigma_g : group sigma.
    is_rel : whether group is relative.
    r_a : tuple size of any attribute in the group (they all share it).
    d_g : group effective dim.

    Returns
    -------
    log_ratio : (nJ_x, nK_y) float64
    log_D : float64
        log of the window-dependent constant prefactor D_g.
    """
    a_, b_ = _window_width_params(s_g, mix_g, sigma_g)

    # Determine case. "1-D" here means d_g == 1; "multi-D absolute" means
    # d_g >= 2 and not is_rel; "multi-D relative" means d_g >= 2 and is_rel.
    is_1d = (d_g == 1)
    is_multi_abs = (d_g >= 2) and (not is_rel)
    is_multi_rel = (d_g >= 2) and is_rel

    rho = float(mix_g)

    if is_multi_rel and rho > 0:
        raise NotImplementedError(
            f"Multi-D relative groups (d_g = {d_g}, r_a = {r_a}) do not "
            f"support rectangular or raised-rectangular windows (mix = "
            f"{rho}) in v2.1.0. Use mix = 0 (pure Gaussian window), or "
            f"wait for a future release with Gaussian-mixture-window "
            f"approximation."
        )

    if is_1d or is_multi_abs:
        # --- Full (size, mix) family: per-axis closed form in erf. ---
        return _windowed_contribution_factorisable(
            cx_g, cy_g, centre_g, a_, b_, sigma_g, is_rel, r_a, d_g,
        )

    # --- Multi-D relative, Gaussian window (rho = 0). ---
    # At this point we know: is_multi_rel and rho == 0, so b_ = s * sigma
    # and a_ = 0.
    return _windowed_contribution_gaussian_multi_rel(
        cx_g, cy_g, centre_g, b_, sigma_g, r_a, d_g,
    )


def _windowed_contribution_factorisable(cx_g, cy_g, centre_g,
                                         a_rect, b_conv, sigma_g,
                                         is_rel, r_a, d_g):
    """Closed-form windowed-vs-unwindowed log-ratio per (j, k) pair.

    Covers: 1-D groups (any type) and multi-D absolute groups, where
    the integrand factorises per axis.

    Mathematical form. For one pair (j, k) in one axis of the group:
    the unwindowed integrand is two toolbox-convention Gaussian kernels
    (exponent -(u - mu)^2/(2 sigma^2)) multiplied together. Their
    product is a wider Gaussian centred at m = (mu_j + mu_k)/2 with
    variance sigma^2/2, times a scalar factor. The unwindowed pair
    integral already sits inside the existing unwindowed machinery (the
    scalar factor is exp(-(mu_j - mu_k)^2 / (4 sigma^2)) — exactly what
    _ma_log_kernel computes — and the Gaussian integrates to a constant
    that cancels under Option Z normalisation).

    The windowed pair integral equals the unwindowed one times the
    "excess factor" F_g, which is the integral of the product Gaussian
    against the window, normalised so F_g = 1 when the window is
    constant (size -> infinity).

    For the rectangular-convolved-with-Gaussian window family,

        F_g = [erf((mu + a)/(sigma_t sqrt 2)) - erf((mu - a)/(sigma_t sqrt 2))]
              / (2 erf(a / (b sqrt 2)))

    where mu = m - c, and sigma_t^2 = sigma_pair^2 + b^2 with
    sigma_pair^2 = sigma_g^2/2 (absolute) or r_a * sigma_g^2/2 (1-D
    relative with r_a = 2). The denominator normalises the window to
    have peak value 1; the numerator is the unnormalised Gaussian
    integrated against a boxcar of half-width a, widened by the
    window's Gaussian-convolution component.

    The formula covers the whole (size, mix) family; limits are:
        rho = 0  (a = 0, b = s*sigma): F_g -> pure-Gaussian-window
                  formula, reducing via l'Hopital to
                  (b/sigma_t) exp(-mu^2 / (2 sigma_t^2)).
        rho = 1  (a = s*sigma*sqrt(3), b = 0): F_g -> boxcar-integral
                  formula, (1/2)[erf((mu+a)/sigma) - erf((mu-a)/sigma)].
        size -> infinity: F_g -> 1.

    For multi-D absolute groups, F_g factorises across axes as the
    product of per-axis F values (the integrand is isotropic in
    effective-space coordinates, so the multi-D integral reduces to a
    product of per-axis integrals).

    Returns
    -------
    log_F : (nJ_x, nK_y) float64
        log(F_g) per pair, which is what gets added to the existing
        unwindowed log-kernel to produce the windowed log-kernel.
    log_D : float
        Window-dependent prefactor outside the per-pair computation.
        Zero under Option Z (window-free integration constants cancel
        between numerator and denominator because the numerator and
        both unwindowed norms share the same per-group integration
        factors).
    """
    from scipy.special import erf

    # Effective variance of the (j, k) product Gaussian, per axis.
    # Absolute group:         sigma_pair^2 = sigma_g^2 / 2.
    # 1-D relative (r_a = 2): sigma_pair^2 = r_a * sigma_g^2 / 2.
    if is_rel:
        sigma_pair_sq = r_a * sigma_g**2 / 2.0
    else:
        sigma_pair_sq = sigma_g**2 / 2.0

    sigma_t_sq = sigma_pair_sq + b_conv**2
    sigma_t = np.sqrt(sigma_t_sq)

    # Per-pair midpoint m minus window centre c.
    m = 0.5 * (cx_g[:, :, None] + cy_g[:, None, :])    # (d_g, nJ_x, nK_y)
    mu_shift = m - centre_g[:, None, None]              # (d_g, nJ_x, nK_y)

    # Per-axis factor F(mu_shift; a_rect, b_conv, sigma_t).
    # Three boundary cases handled: a=0 (pure Gaussian), b=0 (pure
    # rectangular), and general (both > 0).
    if a_rect == 0.0 and b_conv > 0.0:
        # Pure Gaussian window. L'Hopital on the general formula gives:
        # F = (b / sigma_t) * exp(-mu_shift^2 / (2 sigma_t^2))
        # per axis.
        per_axis = (b_conv / sigma_t) * np.exp(-mu_shift**2 / (2 * sigma_t_sq))
    elif b_conv == 0.0 and a_rect > 0.0:
        # Pure rectangular window. sigma_t = sqrt(sigma_pair^2) here.
        # F = 0.5 * [erf((mu + a)/(sigma_t sqrt(2))) - erf((mu - a)/(sigma_t sqrt(2)))]
        denom = sigma_t * np.sqrt(2.0)
        arg_plus = (mu_shift + a_rect) / denom
        arg_minus = (mu_shift - a_rect) / denom
        per_axis = 0.5 * (erf(arg_plus) - erf(arg_minus))
    else:
        # General case (0 < rho < 1). Normalised rect-conv-Gaussian window.
        denom = sigma_t * np.sqrt(2.0)
        arg_plus = (mu_shift + a_rect) / denom
        arg_minus = (mu_shift - a_rect) / denom
        numer = erf(arg_plus) - erf(arg_minus)
        norm_denom = 2.0 * erf(a_rect / (b_conv * np.sqrt(2.0)))
        per_axis = numer / norm_denom

    # Guard against tiny-or-negative values before taking log (numerical
    # noise can give very small negatives at large distances).
    per_axis = np.clip(per_axis, 1e-300, None)
    log_F = np.sum(np.log(per_axis), axis=0)             # (nJ_x, nK_y)
    return log_F, 0.0


def _windowed_contribution_gaussian_multi_rel(cx_g, cy_g, centre_g,
                                                b_conv, sigma_g,
                                                r_a, d_g):
    """Pure-Gaussian-window contribution on a multi-D relative group.

    Only supported when rho = 0 (i.e., mix = 0, so a_rect = 0 and
    b_conv = size * sigma). For rho > 0 on a multi-D relative group,
    the caller raises NotImplementedError.

    Derivation. The per-pair product Gaussian lives in the group's
    (r_a - 1)-dimensional reduced space (effective space). In the
    "subtract the mean and drop the last coordinate" reduction used by
    build_exp_tens, the quadratic form on the reduced coordinates v is
    v^T A_g v, where

        A_g = I_{r_a - 1} + 1 1^T    (ones-matrix addition, size d_g x d_g),

    with eigenvalues 1 (multiplicity r_a - 2) and r_a (multiplicity 1).
    This gives single-density precision A_g / (2 sigma_g^2), and the
    (j, k) product Gaussian has precision A_g / sigma_g^2 and covariance
    Sigma_pair = sigma_g^2 * A_g^{-1}.

    Adding an isotropic Gaussian window of variance b^2 * I gives
    combined covariance Sigma_K = Sigma_pair + b^2 I. The
    excess factor F_g (normalised so F_g = 1 at size -> infinity) is

        F_g = sqrt(det(Sigma_pair) / det(Sigma_K))
              * exp(-0.5 * (m - c)^T Sigma_K^{-1} (m - c))

    where m is the pair midpoint and c is the window centre, both in
    reduced coordinates.

    Returns
    -------
    log_F : (nJ_x, nK_y) float64
    log_D : float
        Zero; the normalisation-related prefactor cancels between the
        windowed numerator and the unwindowed norms under Option Z
        (the unwindowed integration constants are the same per group
        regardless of window).
    """
    # In the "drop first slot, v[i] = u[i+1] - u[1]" reduction used by
    # build_exp_tens, the quadratic form in reduced coords is
    #
    #     Q(v) = v^T M_rel v    where M_rel = I - (1/r) * 1 1^T  (size d_g).
    #
    # The single-density kernel is exp(-Q/(4 sigma_g^2)), i.e. a Gaussian
    # with precision M_rel / (2 sigma_g^2) and covariance
    # 2 sigma_g^2 * M_rel^{-1}. The inverse is
    #
    #     M_rel^{-1} = I + 1 1^T    (size d_g x d_g, using Sherman-Morrison).
    #
    # Product-of-two-densities precision = M_rel / sigma_g^2,
    # covariance Sigma_pair = sigma_g^2 * M_rel^{-1} = sigma_g^2 (I + 1 1^T).
    A_g_inv = np.eye(d_g) + np.ones((d_g, d_g))   # = M_rel^{-1}
    Sigma_pair = sigma_g**2 * A_g_inv

    # Gaussian window covariance (isotropic).
    T = b_conv**2 * np.eye(d_g)

    # Combined covariance for the K factor.
    Sigma_K = Sigma_pair + T
    K_precision = np.linalg.inv(Sigma_K)

    # Normalisation factor log prefactor (per group, constant across pairs).
    det_pair = np.linalg.det(Sigma_pair)
    det_K = np.linalg.det(Sigma_K)
    log_prefactor = 0.5 * (np.log(det_pair) - np.log(det_K))

    # Per-pair midpoint m - c.
    m = 0.5 * (cx_g[:, :, None] + cy_g[:, None, :])    # (d_g, nJ_x, nK_y)
    mu_shift = m - centre_g[:, None, None]              # (d_g, nJ_x, nK_y)

    # Quadratic form (mu_shift)^T Sigma_K^{-1} (mu_shift), per pair.
    # Result shape: (nJ_x, nK_y).
    temp = np.einsum("ij,jab->iab", K_precision, mu_shift)
    K_quad = np.einsum("iab,iab->ab", mu_shift, temp)

    log_F = log_prefactor - 0.5 * K_quad
    return log_F, 0.0


# -------------------------------------------------------------------
#  MAET input helper: simplex coordinates for categorical encoding
# -------------------------------------------------------------------


def simplex_vertices(N: int, edge_length: float = 1.0) -> np.ndarray:
    """Vertices of a regular (N-1)-simplex centred at the origin.

    Returns an N-by-(N-1) array whose rows are the vertices of a regular
    (N-1)-simplex in R^{N-1}, centred at the origin, with all pairwise
    vertex distances equal to ``edge_length`` (default 1).

    This is the natural numerical encoding of an N-level categorical
    attribute (voice identity, instrument, articulation, etc.) for the
    multi-attribute expectation tensor (MAET) framework. Each level is
    represented by an (N-1)-dimensional coordinate vector --- a row of
    the returned array --- and the categorical attribute group then
    carries N-1 coordinate sub-attributes sharing a single sigma.
    Because all vertices are pairwise equidistant, no level is
    privileged over any other, in contrast to dummy or treatment coding.

    Construction: take the N standard basis vectors of R^N (which lie
    on the hyperplane sum(x) = 1 and are pairwise equidistant), centre
    them at the origin, and project onto an orthonormal basis of the
    (N-1)-dimensional subspace orthogonal to the all-ones vector. The
    result is independent of the choice of basis up to a rotation,
    which is irrelevant for downstream MAET computations.

    Parameters
    ----------
    N : int
        Number of categorical levels. Must be at least 2.
    edge_length : float, optional
        Pairwise distance between vertices. Default 1.0. Must be
        positive.

    Returns
    -------
    np.ndarray
        N-by-(N-1) array; row k is the coordinate vector for level k.

    Examples
    --------
    >>> simplex_vertices(2).shape
    (2, 1)
    >>> simplex_vertices(3).shape
    (3, 2)
    >>> simplex_vertices(4).shape
    (4, 3)

    SATB voice encoding with edge length matched to a chosen sigma:

    >>> V = simplex_vertices(4, edge_length=4)
    >>> V.shape
    (4, 3)

    See Also
    --------
    build_exp_tens
    """
    if not isinstance(N, (int, np.integer)) or N < 2:
        raise ValueError(f"N must be an integer >= 2, got {N!r}.")
    if edge_length <= 0:
        raise ValueError(
            f"edge_length must be positive, got {edge_length!r}."
        )

    # Centred standard basis: rows are unit vectors minus the centroid.
    # Pairwise distance between rows is sqrt(2).
    Vc = np.eye(N) - 1.0 / N

    # Orthonormal basis of the column space of Vc, which is 1^perp
    # (the (N-1)-dimensional subspace orthogonal to the all-ones vector).
    # Use SVD: the first N-1 left singular vectors span the column space.
    U, _, _ = np.linalg.svd(Vc, full_matrices=False)
    Q = U[:, : N - 1]

    # Express each row of Vc in this basis. The result has N rows and
    # N-1 columns, with pairwise row distance sqrt(2). Rescale to the
    # requested edge length.
    return (Vc @ Q) * (edge_length / np.sqrt(2))
