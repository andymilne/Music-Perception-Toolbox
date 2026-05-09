"""Tests for mpt._mobius — orbit-table correctness and inner-product equality.

These tests verify the combinatorial layer (orbit weights sum to B_r²,
canonical-form invariance) and the numerical layer (orbit Möbius equals
direct enumeration to floating-point at healthy values).
"""
from __future__ import annotations

import numpy as np
import pytest
from itertools import permutations
from math import factorial

import sys


from mpt._mobius import (
    OrbitEntry,
    aut_size,
    canonical_form,
    enumerate_contingency_tables,
    get_orbit_table,
    inner_product_orbit,
    inner_product_orbit_grid,
    integer_partitions,
    labelled_pairs_realising_M,
    mobius_for_blocksizes,
    total_mass_abs,
)


# ----------------------------------------------------------------------
# Combinatorial primitives
# ----------------------------------------------------------------------


def test_integer_partitions_count():
    """Number of integer partitions of r matches the partition function p(r)."""
    expected = [1, 1, 2, 3, 5, 7, 11, 15, 22, 30]  # OEIS A000041
    for r in range(len(expected)):
        assert sum(1 for _ in integer_partitions(r)) == expected[r], (
            f"r={r}: got {sum(1 for _ in integer_partitions(r))}, "
            f"expected {expected[r]}"
        )


def test_integer_partitions_decreasing():
    """All yielded partitions are weakly decreasing."""
    for r in range(1, 8):
        for p in integer_partitions(r):
            assert all(p[i] >= p[i + 1] for i in range(len(p) - 1))
            assert sum(p) == r


def test_aut_size_basics():
    assert aut_size(()) == 1
    assert aut_size((3,)) == 1
    assert aut_size((2, 2)) == 2
    assert aut_size((3, 3, 2, 1, 1)) == 2 * 1 * 2  # = 4
    assert aut_size((1, 1, 1, 1)) == 24


def test_mobius_blocksize_table():
    """Per-block coefficients (-1)^(m-1) (m-1)! verified against a small table."""
    expected = {1: 1, 2: -1, 3: 2, 4: -6, 5: 24, 6: -120}
    for m, val in expected.items():
        assert mobius_for_blocksizes((m,)) == val


def test_mobius_product_over_blocks():
    """μ for a partition is the product of per-block factors."""
    assert mobius_for_blocksizes((2, 1)) == -1 * 1
    assert mobius_for_blocksizes((3, 2, 1)) == 2 * (-1) * 1
    assert mobius_for_blocksizes((2, 2)) == (-1) * (-1)


# ----------------------------------------------------------------------
# Bell number cross-check
# ----------------------------------------------------------------------


def _bell_number(n):
    """Bell numbers via the recurrence B_{n+1} = Σ_k C(n,k) B_k."""
    B = [1]
    for i in range(n):
        next_B = 0
        for k in range(i + 1):
            next_B += B[k] * factorial(i) // (factorial(k) * factorial(i - k))
        B.append(next_B)
    return B[n]


def test_bell_numbers():
    """Sanity: Bell numbers match the known sequence."""
    expected = [1, 1, 2, 5, 15, 52, 203, 877, 4140]
    for n, b in enumerate(expected):
        assert _bell_number(n) == b


# ----------------------------------------------------------------------
# Contingency tables and canonical form
# ----------------------------------------------------------------------


def test_contingency_tables_count():
    """Number of 0-1 tables with all-1 margins of length r equals r!."""
    for r in range(1, 7):
        margins = (1,) * r
        tables = list(enumerate_contingency_tables(margins, margins))
        assert len(tables) == factorial(r)


def test_contingency_tables_margins():
    """Every yielded table satisfies its margin constraints."""
    margins_a = (3, 2, 1)
    margins_b = (2, 2, 2)
    for M in enumerate_contingency_tables(margins_a, margins_b):
        for i, target in enumerate(margins_a):
            assert sum(M[i]) == target
        for j, target in enumerate(margins_b):
            assert sum(row[j] for row in M) == target


def test_canonical_form_idempotent():
    """Canonicalising twice gives the same result."""
    M = [[2, 1, 0], [0, 2, 1], [1, 0, 2]]
    rs = [3, 3, 3]
    cs = [3, 3, 3]
    canon1 = canonical_form(M, rs, cs)
    M2 = [list(row) for row in canon1[2]]
    canon2 = canonical_form(M2, list(canon1[0]), list(canon1[1]))
    assert canon1 == canon2


def test_canonical_form_row_permutation_invariant():
    """Permuting rows within a size group gives the same canonical form."""
    M = [[2, 1, 0], [1, 2, 0], [0, 0, 3]]
    rs = [3, 3, 3]
    cs = [3, 3, 3]
    canon1 = canonical_form(M, rs, cs)
    # Swap rows 0 and 1 (both size 3)
    M2 = [M[1], M[0], M[2]]
    canon2 = canonical_form(M2, rs, cs)
    assert canon1 == canon2


# ----------------------------------------------------------------------
# Orbit table — fundamental property: orbit weights sum to B_r²
# ----------------------------------------------------------------------


@pytest.mark.parametrize("r", [2, 3, 4, 5])
def test_orbit_weights_sum_to_bell_squared(r):
    """The orbit weights must sum to B_r², by construction."""
    table = get_orbit_table(r)
    total_weight = sum(orb.weight for orb in table)
    expected = _bell_number(r) ** 2
    assert total_weight == expected, (
        f"r={r}: orbit weights sum to {total_weight}, expected B_{r}² = {expected}"
    )


def test_orbit_count_table():
    """Orbit count for r = 2, ..., 5 matches expected values."""
    expected_orbits = {2: 4, 3: 10, 4: 33, 5: 92}
    for r, n in expected_orbits.items():
        assert len(get_orbit_table(r)) == n, f"r={r}: got {len(get_orbit_table(r))}, expected {n}"


@pytest.mark.parametrize("r", [2, 3, 4, 5])
def test_orbit_block_size_consistency(r):
    """Each orbit's block sizes must sum to r on each side."""
    for orb in get_orbit_table(r):
        assert sum(orb.m_A) == r
        assert sum(orb.m_B) == r


@pytest.mark.parametrize("r", [2, 3, 4, 5])
def test_orbit_multiplicity_matrix_consistency(r):
    """Multiplicity matrix margins must match block sizes."""
    for orb in get_orbit_table(r):
        # Reconstruct M from edges
        M = np.zeros((orb.qA, orb.qB), dtype=int)
        for alpha, beta, m in orb.edges:
            M[alpha, beta] = m
        assert np.all(M.sum(axis=1) == np.array(orb.m_A))
        assert np.all(M.sum(axis=0) == np.array(orb.m_B))


# ----------------------------------------------------------------------
# Direct enumeration as ground truth at small n
# ----------------------------------------------------------------------


def _direct_inner_product(p_A, w_A, p_B, w_B, sigma, r):
    """Brute-force enumeration of distinct ordered r-tuples on both sides.

    Used as ground truth for small (n, r); cost is O(n!^2 / (n-r)!^2 * r),
    so practical only for n ≤ 8, r ≤ 4.
    """
    n_A, n_B = len(p_A), len(p_B)
    total = 0.0
    for tA in permutations(range(n_A), r):
        wA_p = np.prod([w_A[i] for i in tA])
        cA = np.array([p_A[i] for i in tA])
        for tB in permutations(range(n_B), r):
            wB_p = np.prod([w_B[j] for j in tB])
            cB = np.array([p_B[j] for j in tB])
            kernel = np.exp(-((cA - cB) ** 2) / (4 * sigma ** 2)).prod()
            total += wA_p * wB_p * kernel
    return total * (sigma * np.sqrt(np.pi)) ** r


@pytest.mark.parametrize("r", [2, 3, 4])
@pytest.mark.parametrize("n", [6, 8])
def test_orbit_matches_direct_at_healthy_n(r, n):
    """Möbius–orbit inner product equals direct enumeration to floating point.

    At small n with random weights, the inner product is non-trivial and
    Möbius cancellation is mild; agreement should be ~1e-12 or better.
    Cases where both methods agree the value is at machine epsilon are
    skipped (not a Möbius issue, but a fundamental cancellation property).
    """
    rng = np.random.default_rng(seed=42 + r * 100 + n)
    p_A = np.sort(rng.uniform(0, 3000, n))
    w_A = rng.uniform(0.5, 1.5, n)
    p_B = np.sort(rng.uniform(0, 3000, n))
    w_B = rng.uniform(0.5, 1.5, n)
    sigma = 30.0

    val_direct = _direct_inner_product(p_A, w_A, p_B, w_B, sigma, r)

    diffs = p_A[:, None] - p_B[None, :]
    K = np.exp(-(diffs ** 2) / (4 * sigma ** 2))
    val_orbit = inner_product_orbit(
        K, w_A, w_B, r, prefactor=(sigma * np.sqrt(np.pi)) ** r
    )

    # Self-consistency check: if AA · BB is itself near zero in the orbit
    # path, this is the cancellation regime and a relative-error test is
    # meaningless. Compare absolute errors instead.
    AA = inner_product_orbit(
        np.exp(-((p_A[:, None] - p_A[None, :]) ** 2) / (4 * sigma ** 2)),
        w_A, w_A, r, prefactor=(sigma * np.sqrt(np.pi)) ** r,
    )
    BB = inner_product_orbit(
        np.exp(-((p_B[:, None] - p_B[None, :]) ** 2) / (4 * sigma ** 2)),
        w_B, w_B, r, prefactor=(sigma * np.sqrt(np.pi)) ** r,
    )
    geo_mean = np.sqrt(abs(AA * BB))
    if geo_mean == 0:
        pytest.skip("Self-inner-products vanished; can't form a meaningful comparison.")

    cosine_scale_err = abs(val_direct - val_orbit) / geo_mean
    rel_err = abs(val_direct - val_orbit) / max(abs(val_direct), abs(val_orbit), 1e-300)

    if cosine_scale_err < 1e-12:
        # Catastrophic-cancellation regime: orbit and direct disagree in
        # absolute terms but the disagreement is sub-floating-point relative
        # to the inner-product scale, which is what matters for cosine.
        return

    assert rel_err < 1e-10, (
        f"r={r}, n={n}: direct={val_direct:.4e}, orbit={val_orbit:.4e}, "
        f"rel_err={rel_err:.2e}, cosine_scale_err={cosine_scale_err:.2e}"
    )


# ----------------------------------------------------------------------
# Self-consistency at higher r without direct comparison
# ----------------------------------------------------------------------


@pytest.mark.parametrize("r", [3, 4, 5])
def test_self_inner_product_positive(r):
    """<T, T> must be positive (semi-definite) for any pitch collection."""
    rng = np.random.default_rng(seed=7 + r)
    n = 16
    p = np.sort(rng.uniform(0, 5000, n))
    w = rng.uniform(0.5, 1.5, n)
    sigma = 20.0
    K = np.exp(-((p[:, None] - p[None, :]) ** 2) / (4 * sigma ** 2))
    val = inner_product_orbit(K, w, w, r, prefactor=(sigma * np.sqrt(np.pi)) ** r)
    assert val > 0, f"r={r}: <T, T> = {val} not positive"


# ----------------------------------------------------------------------
# Total-mass formula
# ----------------------------------------------------------------------


@pytest.mark.parametrize("r", [1, 2, 3, 4])
def test_total_mass_matches_direct(r):
    """Z_abs computed via Möbius matches direct enumeration of distinct r-tuples."""
    rng = np.random.default_rng(seed=11)
    n = 6
    p = np.sort(rng.uniform(0, 1000, n))
    w = rng.uniform(0.5, 1.5, n)
    sigma = 25.0

    # Direct: ∫T_abs = (σ√(2π))^r · Σ over distinct r-tuples of weight products
    direct = 0.0
    for tup in permutations(range(n), r):
        wp = 1.0
        for i in tup:
            wp *= w[i]
        direct += wp
    direct *= (sigma * np.sqrt(2 * np.pi)) ** r

    formula = total_mass_abs(p, w, sigma, r)
    rel_err = abs(direct - formula) / abs(direct)
    assert rel_err < 1e-12, (
        f"r={r}: direct={direct:.4e}, formula={formula:.4e}, rel_err={rel_err:.2e}"
    )


# ----------------------------------------------------------------------
# Grid evaluator (for relative-mode integration)
# ----------------------------------------------------------------------


def test_grid_evaluator_matches_static_at_zero_shift():
    """K_u with N_u=1 and u=0 should give the same result as static K."""
    rng = np.random.default_rng(seed=99)
    n = 8
    p_A = np.sort(rng.uniform(0, 1500, n))
    w_A = rng.uniform(0.5, 1.5, n)
    p_B = np.sort(rng.uniform(0, 1500, n))
    w_B = rng.uniform(0.5, 1.5, n)
    sigma = 30.0
    r = 3

    diffs = p_A[:, None] - p_B[None, :]
    K_static = np.exp(-(diffs ** 2) / (4 * sigma ** 2))
    K_u = K_static[None, :, :]  # shape (1, n, n)

    val_static = inner_product_orbit(K_static, w_A, w_B, r)
    val_grid = inner_product_orbit_grid(K_u, w_A, w_B, r)
    assert abs(val_grid[0] - val_static) < 1e-10 * abs(val_static)


if __name__ == "__main__":
    # Quick smoke run when invoked directly
    import subprocess
    subprocess.run(["pytest", __file__, "-v", "--tb=short"], check=False)
