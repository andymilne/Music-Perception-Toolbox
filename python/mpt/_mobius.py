"""Möbius–Bulger inner product machinery (internal).

This module implements the orbit-collapsed Möbius reformulation of the
distinct-index inner product underlying ``cos_sim_exp_tens``,
``entropy_exp_tens`` (Rényi-2 mode), and the windowed inner product.
It also implements the (layer-1-only) Möbius point evaluator and
total-mass calculation used by ``eval_exp_tens`` and the Rényi-2
denominator.

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

Layer 1 alone is enough for point evaluation and total mass (which
have only one side). Layer 2 is the inner-product-specific further
reduction. The two layers compose multiplicatively in the inner-product
case to give |Ω_r| tensor contractions per inner product, with
|Ω_r| = 4, 10, 33, 92, 306, 948, 3210 for r = 2, ..., 8 (compared to
B_r² = 4, 25, 225, 2704, 41209, 769129, 17139600 unsymmetrised).

The orbit table for each r is built once and cached. Pre-built tables
for r = 2, ..., 8 ship with the package; tables for higher r are built
on demand and cached to disk.

Public-in-the-module entry points (called from ``mpt.tensor`` and
consumer wrappers, never directly by user code):

  Layer 1 + 2 (inner product):
    inner_product_orbit         single-pair distinct-index IP
    inner_product_orbit_grid    grid evaluation for windowed contributions
    inner_product_orbit_pw_batched  batched IP over MA per-attr variations
  Layer 1 only (point eval and total mass):
    eval_orbit_abs              point evaluation in absolute mode
    eval_orbit_rel              point evaluation in relative mode
                                (with u-grid vectorisation)
    total_mass_abs              total-mass scalar in absolute mode
    total_mass_rel              total-mass scalar in relative mode
  Orbit-table access:
    get_orbit_table             load (or build and cache) the orbit
                                table for tensor order r

See :doc:`/ARCHITECTURE` (specifically the "orbit-table system" section)
for the design rationale, the cost-model story underlying the dispatcher
that routes between this module and Bulger's method, and the
twin-language ``matlab/+mobius`` package map.
"""
from __future__ import annotations

import os
import pickle
from collections import Counter, defaultdict
from collections.abc import Iterator
from math import factorial
from pathlib import Path

import numpy as np

from ._utils import kernel_chunk_bytes_resolved

# ---------------------------------------------------------------------
# Combinatorial primitives
# ---------------------------------------------------------------------


def integer_partitions(r: int, max_part: int | None = None) -> Iterator[tuple[int, ...]]:
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


def enumerate_contingency_tables(row_sums, col_sums) -> Iterator[list[list[int]]]:
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


def canonical_form(M, row_sums, col_sums) -> tuple[tuple[int, ...], tuple[int, ...], tuple[tuple[int, ...], ...]]:
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
    _maybe_warn_build_cost(r)
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


# Bell numbers B_r for r = 0..12; B_r is the number of set partitions
# of an r-element set, and the orbit table at order r has roughly
# B_r²/symmetry orbits. Tabulated up to the hard cap.
_BELL_NUMBERS = (
    1, 1, 2, 5, 15, 52, 203, 877, 4140, 21147, 115975, 678570, 4213597,
)


# Rough build-time estimates in seconds, calibrated against existing
# in-codebase measurements (test_dispatcher.py annotates r=7 at ~16 s
# and r=8 at ~3 min for Python). Numbers reflect typical desktop
# hardware; absolute times vary 2-5x across machines. r >= 9 are
# extrapolated from B_r^2 scaling and meant only to convey order of
# magnitude.
_BUILD_TIME_ESTIMATE_S = {
    2: 0.01, 3: 0.05, 4: 0.3, 5: 1.5, 6: 6.0,
    7: 16.0, 8: 180.0, 9: 4700.0, 10: 140000.0,
    11: 5e6, 12: 1.5e8,
}


def _format_duration(seconds: float) -> str:
    """Render a build-time estimate in a human-friendly unit."""
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.0f} s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    if seconds < 86400:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} days"


def _maybe_warn_build_cost(r: int) -> None:
    """Print a size + time estimate before building an orbit table.

    Fires when ``r`` is beyond the shipped range (the user is about to
    pay a non-trivial build cost that the package would normally have
    delivered pre-built). Silent for r ≤ 8 (the shipped range);
    bump ``SHIPPED_MAX`` below if more pickles are added later.
    Suppressed entirely by setting ``MPT_NO_BUILD_WARN=1`` for
    automation contexts.

    Output goes to stderr so it doesn't contaminate stdout-based
    pipelines.
    """
    import os
    import sys
    if os.environ.get("MPT_NO_BUILD_WARN"):
        return
    # Match _ORBIT_R_MAX_SHIPPED in tensor.py. Hardcoded here to avoid
    # the _mobius -> tensor import direction (tensor imports _mobius).
    SHIPPED_MAX = 8
    if r <= SHIPPED_MAX:
        return
    bell_r = _BELL_NUMBERS[r] if r < len(_BELL_NUMBERS) else None
    time_s = _BUILD_TIME_ESTIMATE_S.get(r)
    msg_lines = [
        f"mpt: building orbit table for r={r} (not shipped, not cached).",
    ]
    if bell_r is not None:
        msg_lines.append(
            f"     B_r = {bell_r:,}; build cost scales with B_r squared."
        )
    if time_s is not None:
        msg_lines.append(
            f"     Estimated build time: ~{_format_duration(time_s)}"
            f" (rough; depends on system)."
        )
    msg_lines.append(
        "     Result will be cached on disk; subsequent calls return"
        " instantly."
    )
    msg_lines.append(
        "     Suppress this message by setting MPT_NO_BUILD_WARN=1."
    )
    print("\n".join(msg_lines), file=sys.stderr)


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
        single-multiset case, with appropriate Q-form for the active mode.
        For periodic mode, the differences should already be wrapped.
    w_A, w_B : ndarray
        Source weights for A and B.
    r : int
        Tensor order.
    prefactor : float
        Tuple-independent prefactor (e.g. ``(sigma * sqrt(pi))**r`` for the
        single-multiset case). Multiplied through at the end. Defaults to 1
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
    return_term_mass: bool = False,
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
    return_term_mass : bool, default False
        If True (requires ``return_cancellation_ratio=True``),
        additionally return the per-grid-point worst-term magnitudes
        ``prefactor * max_orb(|term_orb_u|)``, shape (N_u,). Callers
        that integrate the values over the grid use this to form a
        mass-aware global cancellation diagnostic
        ``|sum_u values_u| / sum_u term_mass_u``: a grid point where
        the alternating sum cancels exactly to a true zero contributes
        nothing to the numerator or the integral, so — unlike the
        pointwise minimum of ``ratios`` — the global diagnostic is not
        driven to zero by zero-mass points.

    Returns
    -------
    ndarray, shape (N_u,)
        The "absolute-mode inner product at each shift u", to be
        trapezoidally integrated and divided by appropriate prefactors
        to recover the relative-mode inner product. If
        ``return_cancellation_ratio`` is True, returns a 2-tuple
        ``(values, ratios)``; with ``return_term_mass`` also True, a
        3-tuple ``(values, ratios, term_mass)``.
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
        if return_term_mass:
            return values, ratios, prefactor * max_abs_term
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
    return_term_mass: bool = False,
) -> np.ndarray:
    """Per-grid-point weights variant of :func:`inner_product_orbit_grid`.

    Used by the multi-attribute path where each grid point
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
        ``(values, ratios)``; with ``return_term_mass`` also True, a
        3-tuple ``(values, ratios, term_mass)`` (mirror of
        :func:`inner_product_orbit_grid`).

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
        if return_term_mass:
            return values, ratios, prefactor * max_abs_term
        return values, ratios
    return values


# ---------------------------------------------------------------------
# Sparse-kernel orbit inner product (spatial cull for large, sparse
# slot kernels). Each orbit's bipartite graph is contracted by
# min-degree elimination -- degree-1 nodes fold in as mat-vecs,
# degree-2 nodes become Gram products (sparse matmul) -- which for
# every shipped arity (r <= 8; graphs no worse than K_{2,m}) never
# needs more than a 2-D intermediate. A degree->=3 node (would need a
# higher-order intermediate) reverts that orbit to the dense einsum;
# a per-matrix density guard densifies any Gram that fills in. The
# value equals the dense orbit inner product to floating point.
# ---------------------------------------------------------------------


def _orbit_elementwise(A, B):
    import scipy.sparse as sp
    if sp.issparse(A) and sp.issparse(B):
        return A.multiply(B).tocsr()
    Ad = A.toarray() if sp.issparse(A) else A
    Bd = B.toarray() if sp.issparse(B) else B
    return Ad * Bd


def _contract_orbit_sparse(orb, K, wA, wB, density_thresh):
    """Contract one orbit's bipartite graph with a sparse kernel by
    min-degree elimination. Returns the scalar contraction, or ``None``
    to signal that a degree->=3 node was reached (dense fallback)."""
    import scipy.sparse as sp
    nA, nB = wA.size, wB.size
    Kpow = {1: K}

    def kpow(m):
        if m not in Kpow:
            Kpow[m] = K.power(m)
        return Kpow[m]

    node_w, node_n = {}, {}
    for al in range(orb.qA):
        node_w[al] = wA ** orb.m_A[al]
        node_n[al] = nA
    for be in range(orb.qB):
        nid = orb.qA + be
        node_w[nid] = wB ** orb.m_B[be]
        node_n[nid] = nB

    edges = {}

    def add_edge(p, q, M):
        lo, hi = (p, q) if p < q else (q, p)
        if (p, q) != (lo, hi):
            M = M.T
        edges[(lo, hi)] = _orbit_elementwise(edges[(lo, hi)], M) \
            if (lo, hi) in edges else M

    for (al, be, m) in orb.edges:
        add_edge(al, orb.qA + be, kpow(m))

    scalar = 1.0
    nodes = set(node_w)

    def incident(v):
        return [(lo, hi) for (lo, hi) in edges if lo == v or hi == v]

    while nodes:
        degree = {v: len(incident(v)) for v in nodes}
        v = min(nodes, key=lambda x: degree[x])
        d = degree[v]
        if d == 0:
            scalar *= float(node_w[v].sum())
            nodes.discard(v)
            del node_w[v]
        elif d == 1:
            (lo, hi), = incident(v)
            M = edges.pop((lo, hi))
            u = hi if lo == v else lo
            vec = (M.T @ node_w[v]) if lo == v else (M @ node_w[v])
            node_w[u] = node_w[u] * np.asarray(vec).ravel()
            nodes.discard(v)
            del node_w[v]
        elif d == 2:
            e1, e2 = incident(v)
            M1 = edges.pop(e1)
            M2 = edges.pop(e2)
            u1 = e1[1] if e1[0] == v else e1[0]
            u2 = e2[1] if e2[0] == v else e2[0]
            if e1[0] != v:
                M1 = M1.T
            if e2[0] != v:
                M2 = M2.T
            wv = node_w[v]
            M2s = sp.diags(wv) @ M2 if sp.issparse(M2) else wv[:, None] * M2
            G = M1.T @ M2s
            if sp.issparse(G):
                G = G.tocsr()
                if G.nnz > density_thresh * G.shape[0] * G.shape[1]:
                    G = G.toarray()
            nodes.discard(v)
            del node_w[v]
            add_edge(u1, u2, G)
        else:
            return None
    return scalar


def _contract_orbit_dense_from(orb, Kd, wA, wB):
    operands = []
    for al in range(orb.qA):
        operands.append(wA ** orb.m_A[al])
    for be in range(orb.qB):
        operands.append(wB ** orb.m_B[be])
    for (_, _, m) in orb.edges:
        operands.append(Kd if m == 1 else Kd ** m)
    return float(np.einsum(orb.einsum_str, *operands,
                           optimize=orb.einsum_path))


def inner_product_orbit_sparse(K_sp, wA, wB, r, *, prefactor=1.0,
                               return_cancellation_ratio=False,
                               return_term_mass=False,
                               density_thresh=0.34):
    """Sparse-kernel twin of :func:`inner_product_orbit`.

    ``K_sp`` is a scipy.sparse (nA x nB) truncated kernel. The value
    equals :func:`inner_product_orbit` on the same (densified) kernel to
    floating point. With ``return_cancellation_ratio=True`` returns
    ``(value, ratio)`` with ``ratio = |total| / max|term|`` over orbit
    classes, matching the dense contract. With ``return_term_mass=True``
    the prefactored ``max|term|`` is appended as the last element,
    mirroring :func:`inner_product_orbit_grid`; callers integrating over
    a u-grid use it to form a mass-aware cancellation diagnostic.
    """
    table = get_orbit_table(r)
    Kd = None
    total = 0.0
    max_abs_term = 0.0
    for orb in table:
        val = _contract_orbit_sparse(orb, K_sp, wA, wB, density_thresh)
        if val is None:
            if Kd is None:
                Kd = K_sp.toarray()
            val = _contract_orbit_dense_from(orb, Kd, wA, wB)
        term = orb.weight * orb.mu * val
        total += term
        if abs(term) > max_abs_term:
            max_abs_term = abs(term)
    value = prefactor * total
    out = [value]
    if return_cancellation_ratio:
        ratio = abs(total) / max_abs_term if max_abs_term > 0 else 1.0
        out.append(ratio)
    if return_term_mass:
        out.append(prefactor * max_abs_term)
    if len(out) == 1:
        return value
    return tuple(out)



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


def total_mass_rel(p, w, sigma, r) -> float:
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


def get_set_partitions_with_mobius(r: int) -> list[tuple[tuple[tuple[int, ...], ...], int]]:
    """Cached accessor for ``_set_partitions_with_mobius``."""
    if r not in _SET_PARTITION_CACHE:
        _SET_PARTITION_CACHE[r] = _set_partitions_with_mobius(r)
    return _SET_PARTITION_CACHE[r]


_PARTITION_BLOCK_STRUCTURE_CACHE: dict[int, tuple] = {}


def get_partition_block_structure(r: int):
    """Distinct blocks across the set partitions of ``{0, ..., r-1}``, with
    the map from each partition to its blocks' positions in that distinct
    list. Cached per ``r``.

    Both Möbius point evaluators --- :func:`eval_orbit_abs` and the factored
    strategy of :func:`eval_orbit_rel` --- form the set-partition sum
    ``Σ_π μ(π) ∏_{B∈π} f(B)`` in which a block ``B`` recurs across every
    partition that contains it, so evaluating each distinct block's factor
    once and reusing it is a shared acceleration. This accessor supplies the
    parts that depend only on ``r`` (the distinct-block list and the reuse
    map); the per-block factor ``f`` is mode-specific and the combine is
    shared (:func:`mobius_partition_combine`).

    Returns ``(unique_blocks, part_block_idx, mus)``: the distinct blocks in
    first-appearance order; ``part_block_idx[p]`` the ``unique_blocks``
    indices of partition ``p``'s blocks, in the partition's block order; and
    ``mus[p]`` the partition's Möbius weight.
    """
    cached = _PARTITION_BLOCK_STRUCTURE_CACHE.get(r)
    if cached is not None:
        return cached
    partitions = get_set_partitions_with_mobius(r)
    unique_blocks: list = []
    index: dict = {}
    part_block_idx: list = []
    mus: list = []
    for blocks, mu in partitions:
        idxs = []
        for B in blocks:
            k = index.get(B)
            if k is None:
                k = len(unique_blocks)
                index[B] = k
                unique_blocks.append(B)
            idxs.append(k)
        part_block_idx.append(idxs)
        mus.append(mu)
    result = (unique_blocks, part_block_idx, mus)
    _PARTITION_BLOCK_STRUCTURE_CACHE[r] = result
    return result


def mobius_partition_combine(block_contribs, part_block_idx, mus,
                             *, track_max=True):
    """Form the Möbius set-partition sum from precomputed block factors.

    Computes ``Σ_π μ(π) ∏_{B∈π} block_contribs[B]``, where ``block_contribs``
    is indexed by the distinct-block ordering of
    :func:`get_partition_block_structure` and ``part_block_idx`` / ``mus`` are
    its reuse map and Möbius weights. All contributions share a shape (the
    query axis, of any dimensionality), so this is mode-agnostic: ``(n_q,)``
    in absolute mode, ``(N_u, n_q)`` per chunk in the relative factored
    strategy. Returns ``(total, max_abs_term)``; ``max_abs_term`` is the
    per-query maximum ``|μ(π) ∏ ...|`` over partitions for the
    alternating-sum cancellation diagnostic, or ``None`` when
    ``track_max`` is False.
    """
    total = np.zeros_like(block_contribs[0])
    max_abs_term = np.zeros_like(block_contribs[0]) if track_max else None
    for idxs, mu in zip(part_block_idx, mus):
        prod = np.ones_like(block_contribs[0])
        for k in idxs:
            prod = prod * block_contribs[k]
        term = mu * prod
        total += term
        if track_max:
            np.maximum(max_abs_term, np.abs(term), out=max_abs_term)
    return total, max_abs_term


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
    truncation_sigmas: float | None = None,
    kernel_precision: str | None = None,
) -> np.ndarray:
    """Möbius point evaluator for the single-multiset absolute-mode tensor.

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

    # Resolve truncation_sigmas against the global default. ``None``
    # means "use the global default", so a user who set
    # ``set_default(truncation_sigmas=6)`` should see the orbit path
    # apply truncation even when the kwarg is omitted at the call.
    # (Same principle as the resolver call at :func:`eval_exp_tens`'s
    # entry.)
    from ._defaults import resolve_truncation_sigmas
    trunc_resolved = resolve_truncation_sigmas(truncation_sigmas)

    # Per-block factoring (non-periodic):
    #   Q_B(x_B, p) = var(x_B) + m * (mean(x_B) - p)^2
    # reduces the m-D block sum to a 1-D Gaussian kernel sum at
    # effective sigma_eff = sigma/sqrt(m), times a per-query prefactor
    # exp(-var/2sigma^2). Routing through gaussian_kernel_sum then
    # applies truncation via the vectorised 1-D path. The periodic
    # block does not factor cleanly and stays on the direct broadcast
    # path below. ``trunc_resolved`` is always finite now (inf resolves
    # to the accuracy-floor width), so the helper applies whenever the
    # mode is non-periodic.
    #
    # Non-periodic always uses the 1-D variance/mean reduction, whose
    # source sum is culled by gaussian_kernel_sum. Periodic uses the same
    # reduction where it survives wrapping --- the block span and the
    # truncation window both within half the circle --- which is exactly
    # the regime where the culled (circular) kernel sum applies; elsewhere
    # it falls back to the exact direct broadcast.
    from ._kernel import gaussian_kernel_sum
    if is_per:
        r_win = np.sqrt(2.0) * float(trunc_resolved) * sigma
        per_helper_global = period > 2.0 * r_win
    else:
        per_helper_global = False

    unique_blocks, part_block_idx, mus = get_partition_block_structure(r)

    # Each block (a subset of the r positions) recurs across the set
    # partitions, and its factor --- a 1-D Gaussian kernel sum over the N
    # sources --- is the dominant cost. Evaluate each distinct block's
    # factor once here; the reuse across partitions and the alternating
    # sum are shared with the relative evaluator through
    # :func:`mobius_partition_combine`.
    block_contribs = []
    for B in unique_blocks:
        m = len(B)
        x_B = x_flat[list(B), :]                      # (m, n_q_total)
        wm = w ** m if m > 1 else w
        sigma_eff = sigma / np.sqrt(m)

        if not is_per:
            if m == 1:
                mean_x = x_B[0, :]
                var_x = np.zeros(n_q_total, dtype=np.float64)
            else:
                mean_x = x_B.mean(axis=0)
                var_x = np.sum((x_B - mean_x) ** 2, axis=0)
            use_reduction = True
        else:
            # Circular mean/variance relative to the block's reference
            # slot (translation-invariant offsets), so a block sitting on
            # the period seam is handled correctly.
            if m == 1:
                mean_x = x_B[0, :]
                var_x = np.zeros(n_q_total, dtype=np.float64)
                span_ok = True
            else:
                off = x_B - x_B[0:1, :]
                off = off - period * np.floor(off / period + 0.5)
                mean_off = off.mean(axis=0)
                var_x = np.sum((off - mean_off) ** 2, axis=0)
                mean_x = x_B[0, :] + mean_off
                span = off.max(axis=0) - off.min(axis=0)
                span_ok = bool(np.all(span < 0.5 * period))
            use_reduction = per_helper_global and span_ok

        if use_reduction:
            kw: dict = dict(truncation_sigmas=float(trunc_resolved))
            if kernel_precision is not None:
                kw["kernel_precision"] = kernel_precision
            if is_per:
                kw["is_per"] = True
                kw["period"] = period
            kernel_sum = gaussian_kernel_sum(
                p.reshape(1, -1), wm,
                mean_x.reshape(1, -1), float(sigma_eff),
                **kw,
            )
            block_contribs.append(
                np.exp(-var_x * inv_2s2) * np.asarray(kernel_sum).ravel()
            )
        else:
            # Direct (m, N, n_q) broadcast: exact fallback for the
            # periodic small-circle / wide-block case.
            diffs = x_B[:, None, :] - p[None, :, None]
            diffs = diffs - period * np.floor(diffs / period + 0.5)
            sq_sum = np.sum(diffs * diffs, axis=0)
            kernel = np.exp(-sq_sum * inv_2s2)
            block_contribs.append(
                np.einsum('i,iq->q', wm, kernel, optimize=True)
            )

    total, max_abs_term = mobius_partition_combine(
        block_contribs, part_block_idx, mus,
        track_max=return_cancellation_ratio,
    )

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


# --- Factored (K-free) relative-mode evaluation ------------------------
#
# The relative tensor is the translation marginal of the absolute tensor,
#
#     T_rel(Δ) = (1/Z_t) · ∫ T_abs(u, u + Δ_1, ..., u + Δ_{r-1}) du,
#
# and the Möbius partition sum only factorises across slots at fixed u,
# so the integral is intrinsic to the Möbius realisation (integrating it
# analytically re-expands the per-partition product of block sums into
# the O(K^r) tuple enumeration the decomposition exists to avoid). The
# integral is evaluated on a u-grid; what is NOT intrinsic is the K
# factor in the integrand. In the non-periodic case each partition
# block B (with m = |B| slots and query offsets δ_B) factorises exactly:
#
#     f_B(u) = exp(-var(δ_B)/(2σ²)) · S_m(u + mean(δ_B)),
#     S_m(v) = Σ_i w_i^m exp(-m (v - p_i)² / (2σ²)),
#
# where S_m is a query-independent smoothed event distribution at width
# σ/√m. There are only r distinct block sizes, so tabulating S_1..S_r
# once and reading them back by local quintic (6-point Lagrange)
# interpolation collapses the per-(partition, query, u-node) cost from
# O(K) to O(1): per-query cost drops from O(B_r · r · K · N_u) to
# O(B_r · r · N_u) after a one-time O(Σ_m N_fine_m · K) tabulation.
#
# The read-back accuracy is tied to the toolbox-wide kernel-truncation
# floor: with ``truncation_sigmas = k`` the toolbox already treats
# relative contributions below exp(-k²/2) as negligible, so the
# tabulation step is chosen to keep the interpolation error at or below
# that same floor (clamped to [accuracy_floor_eps(), _FACTORED_EPS_CEIL];
# the default k = inf targets the floor, which sits at the noise level
# of the u-grid quadrature itself).
#
# The identity above is exact only in NON-PERIODIC mode: with per-
# component wrapping a block whose offsets straddle an image boundary
# does not separate into var + mean parts, so periodic relative mode
# always uses the direct per-node evaluation.

#: Ceiling for the factored read-back target accuracy (relative). The
#: floor is the toolbox-wide accuracy floor read dynamically via
#: :func:`mpt._defaults.accuracy_floor_eps` (honouring any override), so
#: the single accuracy floor governs every path.
_FACTORED_EPS_CEIL = 1e-3

#: Calibrated constant for the quintic read-back error model
#: ``err ≈ _FACTORED_CALIB_A6 · spp**-6`` (spp = samples per σ/√m; the
#: constant includes the empirically measured Möbius cancellation
#: amplification and a ×10 safety margin; see tests).
_FACTORED_CALIB_A6 = 1600.0

#: Samples-per-σ_m bounds for the S_m tabulation.
_FACTORED_SPP_MIN = 8
_FACTORED_SPP_MAX = 512

#: Cost of one quintic read-back at one (block, u-node, query), in units
#: of one kernel evaluation (exp + multiply-accumulate). Measured on the
#: demo_triadConsonance workload shape (K=72, r=3, n_q=7381): ~77 ns per
#: stencil evaluation (6 gathers + degree-5 weights) versus ~7.7 ns per
#: vectorised kernel operation.
_FACTORED_READBACK_COST = 10.0

#: 6-point Lagrange denominators ∏_{k≠j}(j - k) for j = 0..5.
_L6_DENOM = np.array([-120.0, 24.0, -12.0, 12.0, -24.0, 120.0])


def _factored_target_eps(truncation_sigmas, kernel_precision) -> float:
    """Target relative accuracy of the factored S_m read-back.

    Derived from the kernel-truncation floor ``exp(-k²/2)`` so that a
    single toolbox-wide knob (``truncation_sigmas``, per call or via
    ``mpt.set_default``) governs both the kernel floor and the read-back
    accuracy. ``None`` arguments resolve against the global defaults.
    """
    if kernel_precision is None:
        from ._defaults import get_default
        kernel_precision = get_default("kernel_precision")
    from ._defaults import truncation_floor, accuracy_floor_eps
    # The read-back target is the kernel-truncation floor exp(-k^2/2) at
    # the resolved width (None -> default, inf -> the accuracy floor), so
    # one knob governs both the kernel floor and this accuracy.
    eps = truncation_floor(truncation_sigmas)
    eps = min(max(eps, accuracy_floor_eps()), _FACTORED_EPS_CEIL)
    if kernel_precision == "single":
        # The S_m tabulation itself is only good to ~7 significant
        # figures under single-precision kernels; a tighter read-back
        # target would be spurious.
        eps = max(eps, 1e-7)
    return eps


def _factored_spp(eps: float) -> int:
    """Samples per σ/√m for the S_m tabulation, from the target eps."""
    spp = int(np.ceil((_FACTORED_CALIB_A6 / eps) ** (1.0 / 6.0)))
    return int(np.clip(spp, _FACTORED_SPP_MIN, _FACTORED_SPP_MAX))


def _lagrange6_uniform(y: np.ndarray, x0: float, h: float,
                       pts: np.ndarray) -> np.ndarray:
    """Quintic (6-point Lagrange) read-back on a uniform grid.

    Evaluates the local degree-5 interpolant of the samples ``y`` on the
    grid ``x0 + i·h`` at the points ``pts`` (any shape). Fully
    vectorised; the same closed-form stencil weights are used in the
    MATLAB twin (``mobius.evalOrbitRel``) for exact cross-language
    parity. Stencils are clamped at the grid ends; callers pad the grid
    so that clamping only occurs where ``y`` has decayed below the
    truncation floor.
    """
    n = y.shape[0]
    if n < 6:
        raise ValueError(f"read-back grid must have >= 6 nodes; got {n}.")
    s = (np.asarray(pts, dtype=np.float64) - x0) / h
    base = np.floor(s).astype(np.intp) - 2
    np.clip(base, 0, n - 6, out=base)
    t = s - base                                  # stencil coordinate
    d = t[..., None] - np.arange(6.0)             # (..., 6)
    # ∏_{k≠j}(t - k) via prefix/suffix products (no division by d,
    # which may be exactly zero at grid nodes).
    pref = np.ones_like(d)
    pref[..., 1:] = np.cumprod(d[..., :-1], axis=-1)
    suff = np.ones_like(d)
    suff[..., :-1] = np.cumprod(d[..., :0:-1], axis=-1)[..., ::-1]
    wgt = pref * suff / _L6_DENOM
    idx = base[..., None] + np.arange(6)
    return np.einsum('...j,...j->...', wgt, y[idx])


def _lagrange6_circular(y: np.ndarray, x0: float, h: float,
                        pts: np.ndarray) -> np.ndarray:
    """Quintic (6-point Lagrange) read-back on a periodic uniform grid.

    Circular twin of :func:`_lagrange6_uniform`. The samples ``y`` cover
    exactly one period on the grid ``x0 + i·h``, ``i = 0 .. n-1`` (so
    ``n·h`` is the period); the interpolant is periodic and the six-node
    stencil wraps around the ends modulo ``n`` rather than clamping. The
    same closed-form stencil weights as the non-circular twin are used,
    so the MATLAB port can share them.
    """
    n = y.shape[0]
    if n < 6:
        raise ValueError(f"read-back grid must have >= 6 nodes; got {n}.")
    s = (np.asarray(pts, dtype=np.float64) - x0) / h
    i0 = np.floor(s).astype(np.intp)
    base = i0 - 2
    t = s - i0 + 2.0                              # stencil coordinate in [2, 3)
    d = t[..., None] - np.arange(6.0)             # (..., 6)
    pref = np.ones_like(d)
    pref[..., 1:] = np.cumprod(d[..., :-1], axis=-1)
    suff = np.ones_like(d)
    suff[..., :-1] = np.cumprod(d[..., :0:-1], axis=-1)[..., ::-1]
    wgt = pref * suff / _L6_DENOM
    idx = np.mod(base[..., None] + np.arange(6), n)   # wrap
    return np.einsum('...j,...j->...', wgt, y[idx])


def _factored_worthwhile(K: int, r: int, n_q: int, N_u: int,
                         n_fine_total: int) -> bool:
    """Cost gate: is the factored path cheaper than direct evaluation?

    Direct cost ≈ B_r · r · K · N_u · n_q kernel evaluations; factored
    cost ≈ K · n_fine_total (tabulation) plus
    ``_FACTORED_READBACK_COST`` kernel-equivalents per (partition-block,
    u-node, query). For a single query with modest K the tabulation
    dominates and direct wins; the factored path wins as n_q or K grow.
    """
    B_r = float(len(get_set_partitions_with_mobius(r)))
    direct = B_r * r * float(K) * float(N_u) * float(n_q)
    fact = (float(K) * float(n_fine_total)
            + _FACTORED_READBACK_COST * B_r * r * float(N_u) * float(n_q))
    return fact < direct


def eval_orbit_rel(
    p: np.ndarray,
    w: np.ndarray,
    sigma: float,
    r: int,
    x_rel: np.ndarray,
    *,
    is_per: bool = False,
    period: float = 0.0,
    samples_per_sigma: int | None = None,
    return_cancellation_ratio: bool = False,
    truncation_sigmas: float | None = None,
    kernel_precision: str | None = None,
    factored: bool | None = None,
) -> np.ndarray:
    """Möbius point evaluator for the single-multiset relative-mode tensor.

    Computes ``T_rel(x_rel_q)`` for each column of *x_rel* via
    u-grid quadrature of the translation marginal:

        T_rel(Δ) = (1/Z_t) · ∫ T_abs(u, u + Δ_1, ..., u + Δ_{r-1}) du,

    where ``Z_t = σ√(2π/r)`` is the translation-mode normaliser
    (verified against grid integration for both periodic and
    non-periodic cases). The quadrature itself is intrinsic to the
    Möbius realisation of relative mode: the alternating partition sum
    only factorises across slots at fixed ``u``, and integrating it
    analytically re-expands into the ``O(K^r)`` tuple enumeration the
    decomposition exists to avoid. The grid mirrors
    :func:`mpt.tensor._orbit_inner_rel`: periodic uses ``[0, P)``
    sampled at ``samples_per_sigma`` points per σ; non-periodic uses
    a Gaussian-supported window extending 8σ beyond the alignment of
    the source positions and the query trajectory.

    Two integrand-evaluation strategies are available:

    - **Direct** — each u-node costs one :func:`eval_orbit_abs`
      evaluation; per-query cost ``O(B_r · r · K · N_u)``.
    - **Factored** (non-periodic only) — each partition block's factor
      separates exactly as ``exp(-var(δ_B)/2σ²) · S_m(u + mean(δ_B))``
      with ``S_m(v) = Σ_i w_i^m exp(-m (v - p_i)²/2σ²)`` a
      query-independent smoothed event distribution at width ``σ/√m``.
      Tabulating ``S_1..S_r`` once and reading them back by local
      quintic interpolation removes the ``K`` factor from the per-node
      cost: per-query cost ``O(B_r · r · N_u)`` after a one-time
      ``O(Σ_m N_fine_m · K)`` tabulation. The read-back accuracy is
      tied to ``truncation_sigmas``: the tabulation step targets a
      relative error at the kernel-truncation floor ``exp(-k²/2)``
      (clamped to ``[1e-12, 1e-3]``; ``inf`` targets ``1e-12``, at the
      noise level of the u-grid quadrature; under
      ``kernel_precision='single'`` the target is floored at ``1e-7``).

    By default (``factored=None``) a cost gate picks the cheaper
    strategy per call (direct for a single query at modest ``K``,
    factored for batches or large ``K``). Periodic relative mode always
    uses the direct strategy: with per-component wrapping a block whose
    offsets straddle an image boundary does not separate into
    variance and mean parts, so the factorisation identity does not
    hold on the circle.

    Memory: the direct strategy's intermediate is chunked along the
    query axis against the ``kernel_chunk_bytes`` budget; the factored
    strategy's ``(N_u, n_q_chunk)`` read-back blocks are chunked
    against the same budget.

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
    samples_per_sigma : int or None, default None
        u-grid density in points per σ. ``None`` derives the count from
        ``truncation_sigmas`` and ``r`` via
        :func:`mpt._defaults.resolve_samples_per_sigma`, so the
        quadrature error sits at or below the kernel truncation floor.
        An explicit integer is honoured unchanged.
    truncation_sigmas, kernel_precision : optional
        Kernel-evaluation controls, resolved against the global
        defaults when ``None``. Besides their usual kernel-floor
        semantics, they set the factored read-back accuracy target as
        described above.
    factored : bool or None, default None
        ``None`` — cost gate chooses per call. ``True`` — force the
        factored strategy (raises ``ValueError`` in periodic mode).
        ``False`` — force the direct strategy. Intended for testing
        and benchmarking; the gate is the supported default.

    Returns
    -------
    ndarray, shape (n_q,)
        Tensor values at the relative query points. With
        ``return_cancellation_ratio=True``, returns
        ``(values, worst_ratios)`` where ``worst_ratios`` is the
        minimum cancellation ratio across u-grid points for each
        query (a corruption signal). Ratio semantics are identical in
        both strategies (per-node alternating-sum ratio, worst case
        over the grid).
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

    from ._defaults import resolve_samples_per_sigma
    samples_per_sigma = resolve_samples_per_sigma(
        samples_per_sigma, r, truncation_sigmas
    )

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

    # ---- Strategy selection ----
    K = int(p.shape[0])
    eps = _factored_target_eps(truncation_sigmas, kernel_precision)
    spp = _factored_spp(eps)
    if is_per:
        # Factored-periodic is valid only where the circular variance/mean
        # block reduction survives wrapping: the truncation window must fit
        # within half the circle and each query's position span must too,
        # so no source that matters lies on the wrapped-far side. This is
        # exactly the regime where culling helps (window < half-circle);
        # below it, fall back to the direct strategy.
        from ._defaults import resolve_truncation_sigmas
        trunc_eff = resolve_truncation_sigmas(truncation_sigmas)
        r_win = np.sqrt(2.0) * float(trunc_eff) * sigma
        if r >= 2:
            pos_lo = np.minimum(0.0, x_rel.min(axis=0))
            pos_hi = np.maximum(0.0, x_rel.max(axis=0))
            max_spread = float(np.max(pos_hi - pos_lo))
        else:
            max_spread = 0.0
        factored_per_valid = (period > 2.0 * r_win) and (max_spread < 0.5 * period)
        if factored is True and not factored_per_valid:
            raise ValueError(
                "factored=True is not available for this periodic relative "
                "case: the truncation window or query span exceeds half the "
                "period, so the circular variance/mean block factorisation "
                "wraps. Use factored=None or factored=False."
            )
        if factored is None:
            n_fine_total = 0
            for m in range(1, r + 1):
                h_m = (sigma / np.sqrt(m)) / spp
                n_fine_total += max(6, int(round(period / h_m)))
            use_factored = (
                factored_per_valid
                and _factored_worthwhile(K, r, n_q, N_u, n_fine_total)
            )
        else:
            use_factored = bool(factored) and factored_per_valid
    else:
        # Read-back points are u + mean(δ_B) with the block means lying
        # inside the hull of the full offset rows (slot 0 carries δ=0).
        dmin = min(0.0, float(x_rel.min(initial=0.0)))
        dmax = max(0.0, float(x_rel.max(initial=0.0)))
        n_fine_total = 0
        for m in range(1, r + 1):
            h_m = (sigma / np.sqrt(m)) / spp
            n_fine_total += int(np.ceil((u_max + dmax - u_min - dmin) / h_m)) + 12
        if factored is None:
            use_factored = _factored_worthwhile(K, r, n_q, N_u, n_fine_total)
        else:
            use_factored = bool(factored)

    BUDGET_BYTES = kernel_chunk_bytes_resolved()

    if use_factored:
        # ---- Factored strategy: tabulate S_1..S_r, read back ----
        from ._kernel import gaussian_kernel_sum
        kw: dict = {}
        if truncation_sigmas is not None:
            kw["truncation_sigmas"] = float(truncation_sigmas)
        if kernel_precision is not None:
            kw["kernel_precision"] = kernel_precision
        tables: dict[int, tuple[float, float, np.ndarray]] = {}
        for m in range(1, r + 1):
            sig_m = sigma / np.sqrt(m)
            wm = w ** m if m > 1 else w
            if is_per:
                # Uniform grid over exactly one period; the read-back
                # wraps its stencil, so no padding is needed. Tabulate the
                # circular S_m directly (gaussian_kernel_sum's periodic
                # path is exact over the circle).
                n_m = max(6, int(round(period / (sig_m / spp))))
                h_m = period / n_m
                lo = 0.0
                grid_m = h_m * np.arange(n_m)
                vals = gaussian_kernel_sum(
                    p.reshape(1, -1), wm, grid_m.reshape(1, -1),
                    float(sig_m), is_per=True, period=period, **kw,
                )
            else:
                h_m = sig_m / spp
                lo = u_min + dmin - 3.0 * h_m
                hi = u_max + dmax + 3.0 * h_m
                n_m = int(np.ceil((hi - lo) / h_m)) + 7
                grid_m = lo + h_m * np.arange(n_m)
                vals = gaussian_kernel_sum(
                    p.reshape(1, -1), wm, grid_m.reshape(1, -1),
                    float(sig_m), **kw,
                )
            tables[m] = (lo, h_m, np.asarray(vals, dtype=np.float64).ravel())

        partitions = get_set_partitions_with_mobius(r)
        inv_2s2 = 1.0 / (2.0 * sigma * sigma)
        deltas_all = np.vstack([np.zeros((1, n_q)), x_rel])

        # Each block (a subset of the r positions) recurs across the set
        # partitions --- a singleton, for instance, reappears in every
        # partition that isolates it --- and the block read-back is the
        # dominant cost. Evaluate each distinct block's contribution once
        # per query chunk (here) and reuse it across partitions through the
        # shared :func:`mobius_partition_combine`.
        unique_blocks, part_block_idx, mus = get_partition_block_structure(r)
        n_unique_blocks = len(unique_blocks)

        # Chunk queries: dominant transient is the (N_u, n_q_chunk, 6)
        # stencil workspace, the per-block contribution cache
        # (n_unique_blocks × (N_u, n_q_chunk)), and a few accumulators.
        per_query_bytes = 8 * N_u * (40 + n_unique_blocks)
        chunk_size = max(1, BUDGET_BYTES // max(per_query_bytes, 1))
        chunk_size = min(chunk_size, n_q)

        F = np.empty((N_u, n_q), dtype=np.float64)
        R = (np.ones((N_u, n_q), dtype=np.float64)
             if return_cancellation_ratio else None)

        for c0 in range(0, n_q, chunk_size):
            c1 = min(c0 + chunk_size, n_q)
            dl = deltas_all[:, c0:c1]
            block_contribs = []
            for B in unique_blocks:
                Bl = list(B)
                m = len(Bl)
                dB = dl[Bl, :]
                mean_d = dB.mean(axis=0)
                var_d = np.sum((dB - mean_d) ** 2, axis=0)
                lo, h_m, ym = tables[m]
                pts = u_grid[:, None] + mean_d[None, :]
                if is_per:
                    Sm = _lagrange6_circular(ym, lo, h_m, pts)
                else:
                    Sm = _lagrange6_uniform(ym, lo, h_m, pts)
                block_contribs.append(np.exp(-var_d * inv_2s2)[None, :] * Sm)
            total, max_abs = mobius_partition_combine(
                block_contribs, part_block_idx, mus,
                track_max=return_cancellation_ratio,
            )
            F[:, c0:c1] = total
            if return_cancellation_ratio:
                with np.errstate(divide='ignore', invalid='ignore'):
                    R[:, c0:c1] = np.where(
                        max_abs > 0, np.abs(total) / max_abs, 1.0,
                    )
    else:
        # ---- Direct strategy: eval_orbit_abs at each u-node ----
        # The intermediate (m, N, N_u, n_q) array can be very large for
        # fine grids; chunk along the query axis to bound peak memory.
        per_chunk_bytes_per_query = 8 * r * N_u * p.shape[0] * 4  # m_max ≤ r, fudge ×4
        chunk_size = max(1, BUDGET_BYTES // max(per_chunk_bytes_per_query, 1))
        chunk_size = min(chunk_size, n_q)

        F = np.empty((N_u, n_q), dtype=np.float64)
        R = (np.ones((N_u, n_q), dtype=np.float64)
             if return_cancellation_ratio else None)

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
                    truncation_sigmas=truncation_sigmas,
                    kernel_precision=kernel_precision,
                )
                F[:, c0:c1] = vals_chunk
                R[:, c0:c1] = ratios_chunk
            else:
                vals_chunk = eval_orbit_abs(
                    p, w, sigma, r, x_full,
                    is_per=is_per, period=period,
                    truncation_sigmas=truncation_sigmas,
                    kernel_precision=kernel_precision,
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
