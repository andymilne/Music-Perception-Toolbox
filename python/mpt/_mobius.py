"""Möbius–Bulger inner product machinery (internal).

This module implements the orbit-collapsed Möbius reformulation of the
distinct-index inner product underlying ``cos_sim_exp_tens``,
``entropy_exp_tens`` (Rényi-2 mode), and the windowed inner product.

The reformulation has two layers:

1. **Möbius inversion on the partition lattice.** A sum over ordered
   distinct r-tuples can be written as an alternating sum over set
   partitions of the slot indices, with each partition's term factorising
   across blocks as a product of single-source-sum quantities.

2. **Orbit collapse under slot symmetry.** The kernel is invariant under
   common permutation of slots, so the |Π_r|² partition pairs collapse to
   |Ω_r| orbits under the joint S_r action. Each orbit is identified by a
   triple (m_A, m_B, M) where m_A, m_B are integer partitions of r and M
   is a non-negative integer matrix with row sums m_A and column sums
   m_B, considered up to within-size-group row and column permutations.

Combined cost: |Ω_r| tensor contractions per inner product, with
|Ω_r| = 4, 10, 33, 92, 306, 948, 3210 for r = 2, ..., 8 (compared to
B_r² = 4, 25, 225, 2704, 41209, 769129, 17139600 unsymmetrised).

The orbit table for each r is built once and cached. Pre-built tables
for r = 2, ..., 8 ship with the package; tables for higher r are built
on demand and cached to disk.
"""
from __future__ import annotations

import os
import pickle
from collections import Counter, defaultdict
from math import factorial
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------
# Combinatorial primitives
# ---------------------------------------------------------------------


def integer_partitions(r: int, max_part: int | None = None):
    """Yield all integer partitions of r as decreasing tuples.

    Examples
    --------
    >>> list(integer_partitions(4))
    [(4,), (3, 1), (2, 2), (2, 1, 1), (1, 1, 1, 1)]
    """
    if max_part is None:
        max_part = r
    if r == 0:
        yield ()
        return
    for p in range(min(r, max_part), 0, -1):
        for rest in integer_partitions(r - p, p):
            yield (p,) + rest


def aut_size(m_tuple: tuple[int, ...]) -> int:
    """Size of the automorphism group of an integer partition.

    For partition (3, 3, 2, 1, 1) this is 2! · 1! · 2! = 4 — permutations
    of the two 3-blocks and the two 1-blocks.
    """
    cnt = Counter(m_tuple)
    out = 1
    for c in cnt.values():
        out *= factorial(c)
    return out


def labelled_pairs_realising_M(M: tuple[tuple[int, ...], ...] | list[list[int]]) -> int:
    """Number of labelled (pi_A, pi_B) pairs with contingency table exactly M.

    For a fixed row/column ordering of M, this equals r! / prod(M_ij!)
    where r is the common total.
    """
    denom = 1
    total = 0
    for row in M:
        for v in row:
            denom *= factorial(v)
            total += v
    return factorial(total) // denom


def mobius_for_blocksizes(m_tuple: tuple[int, ...]) -> int:
    """Möbius coefficient μ(0̂, π) for a partition with given block sizes.

    Standard formula on the partition lattice (Rota 1964):

        μ(0̂, π) = ∏_l (-1)^(m_l - 1) (m_l - 1)!

    """
    out = 1
    for m in m_tuple:
        out *= (-1) ** (m - 1) * factorial(m - 1)
    return out


def enumerate_contingency_tables(row_sums, col_sums):
    """Yield all non-negative integer matrices with given margins.

    Parameters
    ----------
    row_sums, col_sums : sequence of int
        The required row and column sums. Must satisfy
        ``sum(row_sums) == sum(col_sums)``.

    Yields
    ------
    list[list[int]]
        Each yielded matrix is a fresh nested list.
    """
    qA, qB = len(row_sums), len(col_sums)
    if sum(row_sums) != sum(col_sums):
        return

    def helper(M, i, j, remaining_rows, remaining_cols):
        if i == qA:
            yield [row[:] for row in M]
            return
        if j == qB - 1:
            v = remaining_rows[i]
            if v <= remaining_cols[j]:
                M[i][j] = v
                new_cols = remaining_cols[:]
                new_cols[j] -= v
                yield from helper(M, i + 1, 0, remaining_rows, new_cols)
                M[i][j] = 0
        else:
            max_v = min(remaining_rows[i], remaining_cols[j])
            for v in range(max_v + 1):
                M[i][j] = v
                new_rows = remaining_rows[:]
                new_rows[i] -= v
                new_cols = remaining_cols[:]
                new_cols[j] -= v
                yield from helper(M, i, j + 1, new_rows, new_cols)
                M[i][j] = 0

    M = [[0] * qB for _ in range(qA)]
    yield from helper(M, 0, 0, list(row_sums), list(col_sums))


def canonical_form(M, row_sums, col_sums):
    """Canonicalise a contingency table under row/column permutations
    within size groups.

    Iteratively sorts rows by (size, content tuple) and columns likewise
    until stable.

    Returns
    -------
    tuple
        ``(canonical_row_sums, canonical_col_sums, canonical_M)`` where the
        last is a tuple of tuples.
    """
    M_arr = np.asarray(M, dtype=int)
    rs = list(row_sums)
    cs = list(col_sums)
    # Iterate to fixed point (max 20 iterations is far more than needed
    # for any realistic r).
    for _ in range(20):
        row_keys = [(rs[i], tuple(M_arr[i, :])) for i in range(len(rs))]
        row_perm = sorted(range(len(rs)), key=lambda i: row_keys[i])
        M_arr = M_arr[row_perm, :]
        rs = [rs[i] for i in row_perm]
        col_keys = [(cs[j], tuple(M_arr[:, j])) for j in range(len(cs))]
        col_perm = sorted(range(len(cs)), key=lambda j: col_keys[j])
        M_arr = M_arr[:, col_perm]
        cs = [cs[j] for j in col_perm]
    return tuple(rs), tuple(cs), tuple(map(tuple, M_arr.tolist()))


# ---------------------------------------------------------------------
# Orbit table
# ---------------------------------------------------------------------


class OrbitEntry:
    """A single orbit in the partition-pair quotient under S_r.

    Attributes
    ----------
    weight : int
        Number of (pi_A, pi_B) partition pairs in this orbit. The orbit
        weights sum to B_r² across all orbits for a given r.
    mu : int
        Möbius coefficient μ(0̂, π_A) · μ(0̂, π_B), shared across the orbit
        because both factors depend only on block sizes.
    m_A, m_B : tuple[int, ...]
        Block sizes of the canonical representatives, in canonical order.
    edges : tuple of (int, int, int)
        Non-zero entries of the multiplicity matrix as
        ``(alpha, beta, M[alpha, beta])`` triples.
    qA, qB : int
        Block counts (equivalently, lengths of m_A and m_B).
    einsum_str : str
        Pre-built einsum subscript string for ``inner_product_orbit``.
    einsum_str_grid : str
        Pre-built einsum subscript string for ``inner_product_orbit_grid``
        (gridded-K, weights without batch axis).
    einsum_str_pw_batched : str
        Pre-built einsum subscript string for
        ``inner_product_orbit_pw_batched`` (gridded-K, weights with batch
        axis). Computed at table-build time so the runtime path doesn't
        re-derive it per orbit per call.
    einsum_path, einsum_path_grid, einsum_path_pw_batched : list
        Precomputed contraction paths (in the form returned by
        ``np.einsum_path(..., optimize='greedy')[0]``), each
        corresponding to the matching ``einsum_str_*`` field. Computed
        at table-build time using reference shapes and passed to
        ``np.einsum(..., optimize=path)`` at runtime so NumPy can skip
        its internal path optimisation step.
    """

    __slots__ = (
        "weight", "mu", "m_A", "m_B", "edges", "qA", "qB",
        "einsum_str", "einsum_str_grid", "einsum_str_pw_batched",
        "einsum_path", "einsum_path_grid", "einsum_path_pw_batched",
    )

    def __init__(self, weight, mu, m_A, m_B, edges, qA, qB,
                 einsum_str, einsum_str_grid,
                 einsum_str_pw_batched=None,
                 einsum_path=None, einsum_path_grid=None,
                 einsum_path_pw_batched=None):
        self.weight = weight
        self.mu = mu
        self.m_A = m_A
        self.m_B = m_B
        self.edges = edges
        self.qA = qA
        self.qB = qB
        self.einsum_str = einsum_str
        self.einsum_str_grid = einsum_str_grid
        self.einsum_str_pw_batched = einsum_str_pw_batched
        self.einsum_path = einsum_path
        self.einsum_path_grid = einsum_path_grid
        self.einsum_path_pw_batched = einsum_path_pw_batched


def _build_orbit_table(r: int) -> list[OrbitEntry]:
    """Build the orbit table for tensor order r by direct enumeration.

    Algorithm:
    1. Enumerate integer partitions of r for m_A and m_B.
    2. For each (m_A, m_B), enumerate contingency tables with those margins.
    3. Canonicalise each table; group by canonical form.
    4. Compute orbit weight from the canonical representative.

    The resulting table has one entry per orbit; orbit weights sum to B_r²
    (cross-checked in tests).
    """
    A_LETTERS = "abcdefghij"
    B_LETTERS = "klmnopqrst"

    table = []
    for m_A in integer_partitions(r):
        mu_A = mobius_for_blocksizes(m_A)
        autA = aut_size(m_A)
        for m_B in integer_partitions(r):
            mu_B = mobius_for_blocksizes(m_B)
            autB = aut_size(m_B)

            # Enumerate contingency tables, group by canonical form
            seen = defaultdict(int)
            for M in enumerate_contingency_tables(m_A, m_B):
                seen[canonical_form(M, m_A, m_B)] += 1

            for (rs_canon, cs_canon, M_canon), tables_in_orbit in seen.items():
                lp_M = labelled_pairs_realising_M(M_canon)
                weight = tables_in_orbit * lp_M // (autA * autB)

                qA, qB = len(rs_canon), len(cs_canon)
                if qA > len(A_LETTERS) or qB > len(B_LETTERS):
                    raise RuntimeError(
                        f"Orbit table builder ran out of einsum labels at r={r}; "
                        f"qA={qA}, qB={qB}. Increase A_LETTERS / B_LETTERS."
                    )
                A_l = A_LETTERS[:qA]
                B_l = B_LETTERS[:qB]
                edges = tuple(
                    (alpha, beta, M_canon[alpha][beta])
                    for alpha in range(qA)
                    for beta in range(qB)
                    if M_canon[alpha][beta] > 0
                )
                # Static-K einsum subscripts (e.g. 'a,b,k,l,ak,bl->')
                subs_static = (
                    [A_l[a] for a in range(qA)]
                    + [B_l[b] for b in range(qB)]
                    + [A_l[a] + B_l[b] for a, b, _ in edges]
                )
                einsum_str = ",".join(subs_static) + "->"
                # Gridded-K subscripts for u-axis integration in relative mode
                subs_grid = (
                    [A_l[a] for a in range(qA)]
                    + [B_l[b] for b in range(qB)]
                    + ["u" + A_l[a] + B_l[b] for a, b, _ in edges]
                )
                einsum_str_grid = ",".join(subs_grid) + "->u"

                # Pw-batched: same as grid but with 'u' prepended to
                # each weight subscript too (per-batch weight vectors).
                # Precomputed here so inner_product_orbit_pw_batched
                # doesn't have to patch the string per orbit per call.
                in_part, _, out_part = einsum_str_grid.partition("->")
                specs = in_part.split(",")
                new_specs = [
                    ("u" + s) if i < qA + qB else s
                    for i, s in enumerate(specs)
                ]
                einsum_str_pw_batched = ",".join(new_specs) + "->" + out_part

                # Precompute contraction paths via np.einsum_path with
                # representative shapes. Path is shape-agnostic enough
                # that runtime variation in n_A, n_B is handled fine;
                # the goal is to skip NumPy's per-call path optimisation
                # step, which dominates for small operands.
                path_ip = _compute_einsum_path_ip(einsum_str, qA, qB, edges)
                path_grid = _compute_einsum_path_grid(
                    einsum_str_grid, qA, qB, edges)
                path_pw_batched = _compute_einsum_path_pw_batched(
                    einsum_str_pw_batched, qA, qB, edges)

                table.append(
                    OrbitEntry(
                        weight=weight,
                        mu=mu_A * mu_B,
                        m_A=list(rs_canon),
                        m_B=list(cs_canon),
                        edges=edges,
                        qA=qA,
                        qB=qB,
                        einsum_str=einsum_str,
                        einsum_str_grid=einsum_str_grid,
                        einsum_str_pw_batched=einsum_str_pw_batched,
                        einsum_path=path_ip,
                        einsum_path_grid=path_grid,
                        einsum_path_pw_batched=path_pw_batched,
                    )
                )
    return table


# ---------------------------------------------------------------------
# Path precomputation helpers
# ---------------------------------------------------------------------

# Reference shapes for path optimisation. The path is largely
# shape-insensitive within the typical orbit-IP range (n_A, n_B in
# 4..32; N_grid 1..200); these values give NumPy enough info to pick
# a sane path, and runtime sizes can vary freely afterwards.
_REF_N_A = 8
_REF_N_B = 8
_REF_N_GRID = 8


def _compute_einsum_path_ip(einsum_str, qA, qB, edges):
    """Path for inner_product_orbit: weights 1-D, kernels 2-D, no grid axis."""
    operands = []
    for _ in range(qA):
        operands.append(np.empty(_REF_N_A))
    for _ in range(qB):
        operands.append(np.empty(_REF_N_B))
    for _ in edges:
        operands.append(np.empty((_REF_N_A, _REF_N_B)))
    path_info = np.einsum_path(einsum_str, *operands, optimize='greedy')
    return path_info[0]


def _compute_einsum_path_grid(einsum_str_grid, qA, qB, edges):
    """Path for inner_product_orbit_grid: weights 1-D, kernels 3-D (u, n_A, n_B)."""
    operands = []
    for _ in range(qA):
        operands.append(np.empty(_REF_N_A))
    for _ in range(qB):
        operands.append(np.empty(_REF_N_B))
    for _ in edges:
        operands.append(np.empty((_REF_N_GRID, _REF_N_A, _REF_N_B)))
    path_info = np.einsum_path(einsum_str_grid, *operands, optimize='greedy')
    return path_info[0]


def _compute_einsum_path_pw_batched(einsum_str_pw_batched, qA, qB, edges):
    """Path for inner_product_orbit_pw_batched: every operand carries u axis."""
    operands = []
    for _ in range(qA):
        operands.append(np.empty((_REF_N_GRID, _REF_N_A)))
    for _ in range(qB):
        operands.append(np.empty((_REF_N_GRID, _REF_N_B)))
    for _ in edges:
        operands.append(np.empty((_REF_N_GRID, _REF_N_A, _REF_N_B)))
    path_info = np.einsum_path(
        einsum_str_pw_batched, *operands, optimize='greedy')
    return path_info[0]


# ---------------------------------------------------------------------
# Orbit table cache (in-memory + on-disk)
# ---------------------------------------------------------------------


# Pre-built tables shipped with the package live alongside this module.
_PREBUILT_DIR = Path(__file__).parent / "_orbit_tables"

# Per-user cache for orbit tables built on demand for r > 8 (or wherever
# the pre-built tables stop). Lives at ~/.mpt/orbit_tables/ by default.
_USER_CACHE_DIR = Path(os.environ.get("MPT_CACHE_DIR", "~/.mpt/orbit_tables")).expanduser()

# In-memory cache.
_orbit_cache: dict[int, list[OrbitEntry]] = {}

# Sanity ceiling on r. Building the table at r = 9 takes hours; r = 10
# is essentially infeasible. Beyond r = 8, users should know what they're
# asking for.
_R_HARD_CAP = 12


def get_orbit_table(r: int) -> list[OrbitEntry]:
    """Return the orbit table for tensor order r.

    Lookup order: in-memory cache → pre-built shipped tables → user disk
    cache → build from scratch (and cache to disk).

    Parameters
    ----------
    r : int
        Tensor order. Must be ≥ 2.

    Returns
    -------
    list[OrbitEntry]
        Orbit table.
    """
    if r < 2:
        raise ValueError(f"Orbit table requires r >= 2; got r={r}.")
    if r > _R_HARD_CAP:
        raise ValueError(
            f"Orbit table requested at r={r}, beyond hard cap _R_HARD_CAP={_R_HARD_CAP}. "
            "Building tables at this order takes prohibitive time and memory; "
            "if you really need this, raise the cap and accept the cost."
        )

    if r in _orbit_cache:
        return _orbit_cache[r]

    # Try shipped pre-built table
    prebuilt = _PREBUILT_DIR / f"orbit_r{r}.pkl"
    if prebuilt.is_file():
        with open(prebuilt, "rb") as f:
            table = pickle.load(f)
        table = _ensure_paths(table)
        _orbit_cache[r] = table
        return table

    # Try user cache
    user_cached = _USER_CACHE_DIR / f"orbit_r{r}.pkl"
    if user_cached.is_file():
        with open(user_cached, "rb") as f:
            table = pickle.load(f)
        table = _ensure_paths(table)
        _orbit_cache[r] = table
        return table

    # Build from scratch (paths embedded by _build_orbit_table)
    table = _build_orbit_table(r)
    _orbit_cache[r] = table

    # Cache to user disk if the table is non-trivial to rebuild
    if r >= 5:
        try:
            _USER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(user_cached, "wb") as f:
                pickle.dump(table, f, protocol=pickle.HIGHEST_PROTOCOL)
        except OSError:
            # Disk caching is best-effort; in-memory cache is still hot.
            pass

    return table


# ---------------------------------------------------------------------
# Backward-compat: old pre-built tables (shipped before path
# precomputation existed) get their paths computed and attached on
# first load. Path computation is fast (np.einsum_path on placeholder
# arrays); the cost is bounded and paid once per session. Tables
# freshly built by _build_orbit_table already have paths embedded and
# pass through unchanged.
# ---------------------------------------------------------------------


def _ensure_paths(table):
    """Augment loaded orbit-table entries with precomputed paths if missing."""
    if not table:
        return table
    first = table[0]
    needs_augment = (
        not hasattr(first, "einsum_path")
        or first.einsum_path is None
        or not hasattr(first, "einsum_str_pw_batched")
        or first.einsum_str_pw_batched is None
    )
    if not needs_augment:
        return table

    for orb in table:
        # Pre-build pw_batched string if missing.
        if not hasattr(orb, "einsum_str_pw_batched") \
                or orb.einsum_str_pw_batched is None:
            in_part, _, out_part = orb.einsum_str_grid.partition("->")
            specs = in_part.split(",")
            new_specs = [
                ("u" + s) if i < orb.qA + orb.qB else s
                for i, s in enumerate(specs)
            ]
            orb.einsum_str_pw_batched = (
                ",".join(new_specs) + "->" + out_part)

        # Compute the three paths.
        if not hasattr(orb, "einsum_path") or orb.einsum_path is None:
            orb.einsum_path = _compute_einsum_path_ip(
                orb.einsum_str, orb.qA, orb.qB, orb.edges)
        if not hasattr(orb, "einsum_path_grid") \
                or orb.einsum_path_grid is None:
            orb.einsum_path_grid = _compute_einsum_path_grid(
                orb.einsum_str_grid, orb.qA, orb.qB, orb.edges)
        if not hasattr(orb, "einsum_path_pw_batched") \
                or orb.einsum_path_pw_batched is None:
            orb.einsum_path_pw_batched = _compute_einsum_path_pw_batched(
                orb.einsum_str_pw_batched, orb.qA, orb.qB, orb.edges)
    return table


def _build_and_save_prebuilt_tables(max_r: int = 8) -> None:
    """Build orbit tables for r = 2, ..., max_r and save to the shipped
    location. Run during package release preparation; not called at runtime.

    Output: ``_orbit_tables/orbit_r{r}.pkl`` for each r.
    """
    _PREBUILT_DIR.mkdir(parents=True, exist_ok=True)
    for r in range(2, max_r + 1):
        out = _PREBUILT_DIR / f"orbit_r{r}.pkl"
        if out.is_file():
            continue
        table = _build_orbit_table(r)
        with open(out, "wb") as f:
            pickle.dump(table, f, protocol=pickle.HIGHEST_PROTOCOL)


# ---------------------------------------------------------------------
# Inner product evaluators
# ---------------------------------------------------------------------


def inner_product_orbit(
    K: np.ndarray,
    w_A: np.ndarray,
    w_B: np.ndarray,
    r: int,
    *,
    prefactor: float = 1.0,
    return_cancellation_ratio: bool = False,
) -> float:
    """Evaluate the unwindowed distinct-index inner product via Möbius
    + orbit collapse.

    Parameters
    ----------
    K : ndarray, shape (n_A, n_B)
        Pairwise kernel matrix. Conventionally
        ``K[a, b] = exp(-Q(p_A[a] - p_B[b]) / (4*sigma**2))`` for the
        single-attribute case, with appropriate Q-form for the active mode.
        For periodic mode, the differences should already be wrapped.
    w_A, w_B : ndarray
        Source weights for A and B.
    r : int
        Tensor order.
    prefactor : float
        Tuple-independent prefactor (e.g. ``(sigma * sqrt(pi))**r`` for the
        single-attribute case). Multiplied through at the end. Defaults to 1
        if the caller wants the raw orbit-sum quantity.
    return_cancellation_ratio : bool, default False
        If True, additionally return the alternating-sum cancellation
        ratio ``|sum| / max(|term|)``, where the max is taken across
        orbit classes. A value near 1 indicates no cancellation; a value
        much smaller than 1 indicates digits of precision lost to
        catastrophic cancellation. Threshold of ~1e-10 corresponds to
        roughly 6 surviving significant decimal digits in the result;
        callers should fall back to a non-cancelling method below that.

    Returns
    -------
    float
        The inner product ``<T_A, T_B>`` evaluated via the Möbius–Bulger
        reformulation. If ``return_cancellation_ratio`` is True, returns
        ``(value, ratio)`` instead.

    Notes
    -----
    The orbit table is cached on first call per r. Per-call cost is
    O(|Ω_r| · einsum_cost), with einsum cost dependent on the bipartite
    contraction graph but typically O(r·n²) per orbit.
    """
    table = get_orbit_table(r)
    total = 0.0
    max_abs_term = 0.0
    for orb in table:
        operands = []
        # Per-A-block weight vectors
        for alpha in range(orb.qA):
            operands.append(w_A ** orb.m_A[alpha])
        # Per-B-block weight vectors
        for beta in range(orb.qB):
            operands.append(w_B ** orb.m_B[beta])
        # Per-edge kernel powers
        for alpha, beta, m in orb.edges:
            if m == 1:
                operands.append(K)
            else:
                operands.append(K ** m)
        contribution = np.einsum(
            orb.einsum_str, *operands, optimize=orb.einsum_path)
        term = orb.weight * orb.mu * contribution
        total += term
        abs_term = abs(term)
        if abs_term > max_abs_term:
            max_abs_term = abs_term
    value = prefactor * float(total)
    if return_cancellation_ratio:
        # Ratio is a property of the unprefactored alternating sum;
        # multiplying through by the prefactor scales numerator and the
        # max term by the same constant, leaving the ratio invariant.
        # We compute it on the un-prefactored quantities for clarity.
        ratio = (abs(float(total)) / max_abs_term
                 if max_abs_term > 0 else 1.0)
        return value, ratio
    return value


def inner_product_orbit_grid(
    K_u: np.ndarray,
    w_A: np.ndarray,
    w_B: np.ndarray,
    r: int,
    *,
    prefactor: float = 1.0,
    return_cancellation_ratio: bool = False,
) -> np.ndarray:
    """Evaluate distinct-index inner product on a grid of u-shifts.

    Parameters
    ----------
    K_u : ndarray, shape (N_u, n_A, n_B)
        Stack of kernel matrices, one per u-grid point. Used by the
        relative-mode integration approach.
    w_A, w_B : ndarray
    r : int
    prefactor : float
    return_cancellation_ratio : bool, default False
        If True, additionally return per-grid-point cancellation ratios
        ``|sum_u| / max_orb(|term_orb_u|)``, shape (N_u,). See
        :func:`inner_product_orbit` for interpretation.

    Returns
    -------
    ndarray, shape (N_u,)
        The "absolute-mode inner product at each shift u", to be
        trapezoidally integrated and divided by appropriate prefactors
        to recover the relative-mode inner product. If
        ``return_cancellation_ratio`` is True, returns a 2-tuple
        ``(values, ratios)``.
    """
    table = get_orbit_table(r)
    N_u = K_u.shape[0]
    total = np.zeros(N_u, dtype=K_u.dtype)
    max_abs_term = np.zeros(N_u, dtype=K_u.dtype)
    for orb in table:
        operands = []
        for alpha in range(orb.qA):
            operands.append(w_A ** orb.m_A[alpha])
        for beta in range(orb.qB):
            operands.append(w_B ** orb.m_B[beta])
        for alpha, beta, m in orb.edges:
            if m == 1:
                operands.append(K_u)
            else:
                operands.append(K_u ** m)
        contribution = np.einsum(
            orb.einsum_str_grid, *operands, optimize=orb.einsum_path_grid)
        term = orb.weight * orb.mu * contribution
        total += term
        np.maximum(max_abs_term, np.abs(term), out=max_abs_term)
    values = prefactor * total
    if return_cancellation_ratio:
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = np.where(
                max_abs_term > 0,
                np.abs(total) / max_abs_term,
                1.0,
            )
        return values, ratios
    return values


def inner_product_orbit_pw_batched(
    K_g: np.ndarray,
    w_A_g: np.ndarray,
    w_B_g: np.ndarray,
    r: int,
    *,
    prefactor: float = 1.0,
    return_cancellation_ratio: bool = False,
) -> np.ndarray:
    """Per-grid-point weights variant of :func:`inner_product_orbit_grid`.

    Used by the v2.2 multi-attribute path where each grid point
    represents one (event_X, event_Y) pair and the source weights for
    a given attribute differ per event. The standard
    ``inner_product_orbit_grid`` requires shared ``w_A`` and ``w_B``
    across the grid axis; this variant lifts that restriction by
    prepending the batch index to the einsum specifications of the
    weight factors.

    Parameters
    ----------
    K_g : ndarray, shape (N, n_A, n_B)
        Stack of kernel matrices, one per batch index.
    w_A_g : ndarray, shape (N, n_A)
        Per-batch A-side weight vectors.
    w_B_g : ndarray, shape (N, n_B)
        Per-batch B-side weight vectors.
    r : int
        Tensor order.
    prefactor : float
        Tuple-independent prefactor multiplied through at the end.
    return_cancellation_ratio : bool, default False
        If True, additionally return per-batch cancellation ratios,
        shape (N,). See :func:`inner_product_orbit` for interpretation.

    Returns
    -------
    ndarray, shape (N,)
        The orbit-Möbius inner product evaluated at each batch index.
        If ``return_cancellation_ratio`` is True, returns a 2-tuple
        ``(values, ratios)``.

    Notes
    -----
    Built on top of the same orbit table as
    ``inner_product_orbit_grid``; the einsum string is patched on the
    fly by prepending ``u`` to the first ``qA + qB`` substrings (the
    weight factors), reusing ``u`` as the batch index since the orbit
    builder reserves it. K factors already carry ``u`` in
    ``einsum_str_grid``, so no further changes are needed.
    """
    table = get_orbit_table(r)
    N = K_g.shape[0]
    total = np.zeros(N, dtype=K_g.dtype)
    max_abs_term = np.zeros(N, dtype=K_g.dtype)
    for orb in table:
        operands = []
        for alpha in range(orb.qA):
            operands.append(w_A_g ** orb.m_A[alpha])
        for beta in range(orb.qB):
            operands.append(w_B_g ** orb.m_B[beta])
        for _, _, m in orb.edges:
            if m == 1:
                operands.append(K_g)
            else:
                operands.append(K_g ** m)
        contribution = np.einsum(
            orb.einsum_str_pw_batched, *operands,
            optimize=orb.einsum_path_pw_batched,
        )
        term = orb.weight * orb.mu * contribution
        total += term
        np.maximum(max_abs_term, np.abs(term), out=max_abs_term)
    values = prefactor * total
    if return_cancellation_ratio:
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = np.where(
                max_abs_term > 0,
                np.abs(total) / max_abs_term,
                1.0,
            )
        return values, ratios
    return values


# ---------------------------------------------------------------------
# Total-mass formula (used for Rényi-2 entropy and elsewhere)
# ---------------------------------------------------------------------


def total_mass_abs(p_unused, w: np.ndarray, sigma: float, r: int) -> float:
    """Total mass ∫T_abs(x)dx via Möbius for the absolute non-periodic case.

    The total mass is

        ∫T_abs(x)dx = (σ√(2π))^r · Σ_{distinct r-tuples} ∏_s w_{i_s},

    since each tuple's r-dimensional Gaussian integrates to (σ√(2π))^r
    independent of position. The distinct-tuple sum is then expanded via
    Möbius inversion: for set partition π with blocks of sizes (m_1, ..., m_q),
    the respecting-π sum equals ∏_l (Σ_i w_i^{m_l}), so

        ∫T_abs(x)dx = (σ√(2π))^r · Σ_m N(m) · μ(m) · ∏_l (Σ_i w_i^{m_l})

    where N(m) is the number of set partitions of [r] with block-size
    profile m, and μ(m) is the Möbius coefficient (block-size-dependent only).

    Parameters
    ----------
    p_unused : ignored
        (Position vector is not needed since the integral is position-invariant
        for each tuple. Included in the signature for symmetry with other
        orbit-based routines.)
    w : ndarray
    sigma : float
    r : int
    """
    sigma_factor = (sigma * np.sqrt(2 * np.pi)) ** r
    moebius_sum = 0.0
    for m in integer_partitions(r):
        # Number of set partitions of [r] with block-size profile m
        n_with_profile = factorial(r)
        for size in m:
            n_with_profile //= factorial(size)
        n_with_profile //= aut_size(m)

        mu_m = mobius_for_blocksizes(m)
        per_partition = 1.0
        for m_l in m:
            per_partition *= float((w ** m_l).sum())
        moebius_sum += n_with_profile * mu_m * per_partition
    return sigma_factor * moebius_sum


def total_mass_rel(p, w, sigma, r):
    """Total mass ∫T_rel(Δ)dΔ via the relation Z_rel = Z_abs / (σ√(2π/r))."""
    Z_abs = total_mass_abs(p, w, sigma, r)
    return Z_abs / (sigma * np.sqrt(2 * np.pi / r))


# ---------------------------------------------------------------------
# Set-partition machinery for the point evaluator
# ---------------------------------------------------------------------


def _set_partitions_with_mobius(r: int):
    """Enumerate set partitions of {0, ..., r-1} as (blocks, mu) pairs.

    Returns a list of ``(blocks, mu)`` tuples where ``blocks`` is a
    tuple of tuples (each inner tuple is the sorted slot-indices of one
    block) and ``mu`` is the Möbius coefficient for the partition's
    block-size profile.

    The number of partitions is the Bell number ``B_r``: B_2=2, B_3=5,
    B_4=15, B_5=52, B_6=203, B_7=877. The list is built from scratch
    on each call; for the orders of interest in MPT (r ≤ 6) this is
    sub-millisecond.
    """
    if r == 0:
        return [((), 1)]
    if r == 1:
        return [(((0,),), 1)]

    out = []

    def recurse(prev_blocks, next_idx):
        if next_idx == r:
            mu = 1
            for B in prev_blocks:
                m = len(B)
                mu *= (-1) ** (m - 1) * factorial(m - 1)
            blocks = tuple(tuple(sorted(B)) for B in prev_blocks)
            out.append((blocks, mu))
            return
        # Add next_idx to each existing block
        for i in range(len(prev_blocks)):
            new_block = prev_blocks[i] + [next_idx]
            new_blocks = (
                prev_blocks[:i] + [new_block] + prev_blocks[i + 1:]
            )
            recurse(new_blocks, next_idx + 1)
        # Or as a new singleton
        recurse(prev_blocks + [[next_idx]], next_idx + 1)

    recurse([], 0)
    return out


# Cache the result per r — set-partition enumeration is cheap but we
# call eval_orbit_abs in tight loops (rel-mode u-grid).
_SET_PARTITION_CACHE: dict[int, list] = {}


def get_set_partitions_with_mobius(r: int):
    """Cached accessor for ``_set_partitions_with_mobius``."""
    if r not in _SET_PARTITION_CACHE:
        _SET_PARTITION_CACHE[r] = _set_partitions_with_mobius(r)
    return _SET_PARTITION_CACHE[r]


# ---------------------------------------------------------------------
# Point-evaluator (absolute mode)
# ---------------------------------------------------------------------


def eval_orbit_abs(
    p: np.ndarray,
    w: np.ndarray,
    sigma: float,
    r: int,
    x: np.ndarray,
    *,
    is_per: bool = False,
    period: float = 0.0,
    return_cancellation_ratio: bool = False,
) -> np.ndarray:
    """Möbius point evaluator for the SA absolute-mode tensor.

    Computes ``T_abs(x_q)`` for each column of *x* without
    materialising the ``(r, n_distinct_tuples)`` centres array. Uses
    the set-partition Möbius decomposition

        T_abs(x) = Σ_π μ(π) · ∏_l [ Σ_i w_i^{m_l} · exp(-Σ_{k∈B_l}
                                              d²(x_k, p_i)/(2σ²)) ]

    where π ranges over set partitions of {0, ..., r-1} with blocks
    B_1, ..., B_q of sizes m_1, ..., m_q, and ``d`` is plain
    difference (non-periodic) or wrapped difference (periodic). The
    inner sum factorises across blocks because each block's factor
    depends only on its own block-index ``i``.

    *x* may be a 2-D array of shape ``(r, n_q)`` or a higher-
    dimensional array of shape ``(r, ...)`` where ``...`` is any
    product of trailing dimensions. The output has the same trailing
    shape (a 1-D array of length ``n_q`` when ``...`` is ``(n_q,)``).
    The higher-dimensional form is the entry point used by
    :func:`eval_orbit_rel` when it batches its u-grid loop into a
    single vectorised call.

    Memory: ``O(B_r · m_max · N · n_q_total)`` per partition
    (transient, freed before next partition), where ``n_q_total`` is
    the product of trailing dimensions. Compare to the centre-array
    path's ``O(N!/(N-r)! · n_q_total)`` peak. Callers responsible for
    sizing the trailing dims to fit in available memory;
    :func:`eval_orbit_rel` chunks the query axis when it batches its
    u-grid loop.

    Parameters
    ----------
    p : (N,) ndarray
        Source positions.
    w : (N,) ndarray
        Source weights.
    sigma : float
        Gaussian width.
    r : int
        Tensor order. Must be ≥ 1.
    x : (r, ...) ndarray
        Query points; the first dimension must equal ``r``, the
        remaining dimensions are query indices.
    is_per : bool
        Periodic mode flag. When True, all differences are wrapped to
        ``[-period/2, period/2)`` before squaring.
    period : float
        Period (only consulted when ``is_per`` is True).
    return_cancellation_ratio : bool, default False
        If True, additionally return the per-query-point cancellation
        ratios from the set-partition alternating sum, with the same
        trailing shape as the values. Same interpretation as in the
        IP orbit machinery.

    Returns
    -------
    ndarray
        Tensor values with shape matching the trailing dims of *x*.
        With ``return_cancellation_ratio=True``, returns a 2-tuple
        of ``(values, ratios)``.

    Notes
    -----
    For r=1 there is one set partition (a singleton block) with μ=1,
    and the formula reduces to the direct sum
    ``T(x) = Σ_i w_i · exp(-(x - p_i)²/(2σ²))``.
    """
    if r < 1:
        raise ValueError(f"r must be >= 1; got r={r}.")
    if x.ndim < 1 or x.shape[0] != r:
        raise ValueError(
            f"x must have shape (r, ...) with r={r}; got {x.shape}."
        )

    # Collapse trailing dimensions to a single query axis. Restore
    # shape on output. This lets the inner loop stay 2-D while the
    # API accepts any (r, ...) shape.
    out_shape = x.shape[1:]
    n_q_total = int(np.prod(out_shape)) if out_shape else 1
    if n_q_total == 0:
        empty = np.zeros(out_shape, dtype=np.float64)
        if return_cancellation_ratio:
            return empty, np.ones(out_shape, dtype=np.float64)
        return empty
    x_flat = x.reshape(r, n_q_total)

    N = p.shape[0]
    inv_2s2 = 1.0 / (2.0 * sigma * sigma)

    partitions = get_set_partitions_with_mobius(r)
    total = np.zeros(n_q_total, dtype=np.float64)
    max_abs_term = np.zeros(n_q_total, dtype=np.float64)

    for blocks, mu in partitions:
        # Per-partition factor: ∏_l (per-block scalar at each query)
        block_factor = np.ones(n_q_total, dtype=np.float64)
        for B in blocks:
            m = len(B)
            # x_B has shape (m, n_q_total); p has shape (N,)
            # We need Σ_{k∈B}(x_k(q) - p_i)² for each (i, q): shape (N, n_q_total)
            x_B = x_flat[list(B), :]  # (m, n_q_total)
            # (m, N, n_q_total) — broadcasted differences
            diffs = x_B[:, None, :] - p[None, :, None]
            if is_per:
                diffs = diffs - period * np.floor(diffs / period + 0.5)
            sq_sum = np.sum(diffs * diffs, axis=0)  # (N, n_q_total)
            kernel = np.exp(-sq_sum * inv_2s2)  # (N, n_q_total)
            # Multiply by w_i^m and sum over i
            wm = w ** m if m > 1 else w
            block_factor *= np.einsum('i,iq->q', wm, kernel, optimize=True)
        term = mu * block_factor
        total += term
        np.maximum(max_abs_term, np.abs(term), out=max_abs_term)

    # Restore output shape.
    if out_shape:
        values = total.reshape(out_shape)
    else:
        values = total.reshape(())  # 0-D scalar array

    if return_cancellation_ratio:
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios_flat = np.where(
                max_abs_term > 0,
                np.abs(total) / max_abs_term,
                1.0,
            )
        if out_shape:
            ratios = ratios_flat.reshape(out_shape)
        else:
            ratios = ratios_flat.reshape(())
        return values, ratios
    return values


# ---------------------------------------------------------------------
# Point-evaluator (relative mode)
# ---------------------------------------------------------------------


def eval_orbit_rel(
    p: np.ndarray,
    w: np.ndarray,
    sigma: float,
    r: int,
    x_rel: np.ndarray,
    *,
    is_per: bool = False,
    period: float = 0.0,
    samples_per_sigma: int = 10,
    return_cancellation_ratio: bool = False,
) -> np.ndarray:
    """Möbius point evaluator for the SA relative-mode tensor.

    Computes ``T_rel(x_rel_q)`` for each column of *x_rel* via
    u-grid quadrature wrapping :func:`eval_orbit_abs`:

        T_rel(Δ) = (1/Z_t) · ∫ T_abs(u, u + Δ_1, ..., u + Δ_{r-1}) du,

    where ``Z_t = σ√(2π/r)`` is the translation-mode normaliser
    (verified against grid integration for both periodic and
    non-periodic cases). The integration grid mirrors
    :func:`mpt.tensor._orbit_inner_rel`: periodic uses ``[0, P)``
    sampled at ``samples_per_sigma`` points per σ; non-periodic uses
    a Gaussian-supported window extending 8σ beyond the alignment of
    the source positions and the query trajectory.

    Memory and compute: each u-grid evaluation costs the same as one
    :func:`eval_orbit_abs` call. For periodic mode at σ/P = 0.025
    (typical for chord analysis) the grid has ~400 points; runtime
    scales linearly in this count.

    Parameters
    ----------
    p, w, sigma, r : as in :func:`eval_orbit_abs`.
    x_rel : (r-1, n_q) ndarray
        Relative-mode query: each column is a (r-1)-D point.
        Convention: ``x_rel`` represents differences from the
        implicit reference slot; the full r-vector at translation u
        is ``(u, u + x_rel_1, ..., u + x_rel_{r-1})``.
    is_per, period, return_cancellation_ratio : as in
        :func:`eval_orbit_abs`.
    samples_per_sigma : int, default 10
        u-grid density in points per σ. The default matches the IP
        rel-mode path; reduce to 5 for speed at the cost of ~1e-9
        relative precision.

    Returns
    -------
    ndarray, shape (n_q,)
        Tensor values at the relative query points. With
        ``return_cancellation_ratio=True``, returns
        ``(values, worst_ratios)`` where ``worst_ratios`` is the
        minimum cancellation ratio across u-grid points for each
        query (a corruption signal).
    """
    if r < 2:
        # r=1 is degenerate: rel space is 0-dim, T_rel is constant.
        # Total mass / 1 (since 0-dim "volume" is conventionally 1).
        # By convention return Σ_i w_i. Tests should not hit this.
        n_q = x_rel.shape[1] if x_rel.ndim == 2 else x_rel.shape[0]
        val = np.full(n_q, float(w.sum()), dtype=np.float64)
        if return_cancellation_ratio:
            return val, np.ones(n_q)
        return val

    if x_rel.ndim != 2 or x_rel.shape[0] != r - 1:
        raise ValueError(
            f"x_rel must have shape (r-1, n_q) with r-1={r-1}; "
            f"got {x_rel.shape}."
        )

    n_q = x_rel.shape[1]

    # Build u-grid (mirrors _orbit_inner_rel).
    if is_per:
        N_u = max(64, int(np.ceil(period / sigma * samples_per_sigma)))
        u_grid = np.linspace(0.0, period, N_u, endpoint=False)
        du = period / N_u
    else:
        # Non-periodic alignment window: u must let some source p_i
        # land near each query coordinate. Use a window that covers
        # all candidate shifts.
        x_min = float(x_rel.min(initial=0.0))
        x_max = float(x_rel.max(initial=0.0))
        u_min = p.min() - max(0.0, x_max) - 8.0 * sigma
        u_max = p.max() - min(0.0, x_min) + 8.0 * sigma
        N_u = max(
            64,
            int(np.ceil(max(u_max - u_min, 1.0) / sigma * samples_per_sigma)),
        )
        u_grid = np.linspace(u_min, u_max, N_u)

    # Evaluate T_abs at each u-grid point and accumulate, batching the
    # u-grid loop into a single vectorised call to eval_orbit_abs via
    # its (r, ...) trailing-dim API. The intermediate
    # (m, N, N_u, n_q) array can be very large for fine grids; we chunk
    # along the query axis to bound peak memory.
    #
    # Memory budget: heuristic ~1 GB. Per-block intermediate is
    # O(m · N · N_u · n_q_chunk · 8) bytes, dominated by the largest
    # block size m_max ≤ r. The chunk size is solved for given r, N,
    # N_u with a fudge factor for transient allocations during the
    # per-partition arithmetic.
    BUDGET_BYTES = 1024 ** 3
    per_chunk_bytes_per_query = 8 * r * N_u * p.shape[0] * 4  # m_max ≤ r, fudge ×4
    chunk_size = max(1, BUDGET_BYTES // max(per_chunk_bytes_per_query, 1))
    chunk_size = min(chunk_size, n_q)

    F = np.empty((N_u, n_q), dtype=np.float64)
    R = np.ones((N_u, n_q), dtype=np.float64) if return_cancellation_ratio else None

    for c0 in range(0, n_q, chunk_size):
        c1 = min(c0 + chunk_size, n_q)
        n_q_chunk = c1 - c0

        # Build (r, N_u, n_q_chunk) query stack: row 0 is u
        # (broadcast across queries), rows 1..r-1 are u + x_rel.
        x_full = np.empty((r, N_u, n_q_chunk), dtype=np.float64)
        # Row 0: u_grid broadcast across query axis.
        x_full[0, :, :] = u_grid[:, None]
        if r >= 2:
            # x_rel[:, c0:c1] has shape (r-1, n_q_chunk); broadcast u_grid.
            x_full[1:, :, :] = u_grid[None, :, None] + x_rel[:, None, c0:c1]

        if return_cancellation_ratio:
            vals_chunk, ratios_chunk = eval_orbit_abs(
                p, w, sigma, r, x_full,
                is_per=is_per, period=period,
                return_cancellation_ratio=True,
            )
            F[:, c0:c1] = vals_chunk
            R[:, c0:c1] = ratios_chunk
        else:
            vals_chunk = eval_orbit_abs(
                p, w, sigma, r, x_full,
                is_per=is_per, period=period,
            )
            F[:, c0:c1] = vals_chunk

    if is_per:
        integral = F.sum(axis=0) * du
    else:
        integral = np.trapezoid(F, u_grid, axis=0)

    Z_t = sigma * np.sqrt(2 * np.pi / r)
    values = integral / Z_t

    if return_cancellation_ratio:
        # Worst case across u-grid for each query.
        worst = R.min(axis=0)
        return values, worst
    return values
