"""Equivalence tests for precomputed-path orbit machinery.

The v3 Python orbit IP path stores a precomputed contraction path
on each ``OrbitEntry`` (``einsum_path``, ``einsum_path_grid``,
``einsum_path_pw_batched``) and passes it as ``optimize=path`` to
``np.einsum``. This file verifies that path precomputation produces
floating-point-identical results to the prior ``optimize=True``
behaviour across the orbit-IP consumer call patterns.

These are regression tests guarding against any latent issue with
the path-embedding logic — e.g., a path computed at reference shapes
mis-applying to runtime shapes, or a cached path going stale across
table rebuilds.
"""

import numpy as np
import pytest

import mpt
from mpt._mobius import (
    inner_product_orbit,
    inner_product_orbit_grid,
    inner_product_orbit_pw_batched,
    get_orbit_table,
)


RTOL = 1e-12
ATOL = 1e-14


def _make_kernel(K_x, K_y, seed=0):
    rng = np.random.default_rng(seed)
    p_x = np.sort(rng.uniform(0, 2000, K_x))
    p_y = np.sort(rng.uniform(0, 2000, K_y))
    sigma = 30.0
    K = np.exp(-(p_x[:, None] - p_y[None, :]) ** 2 / (4 * sigma ** 2))
    w_a = rng.uniform(0.5, 1.5, K_x)
    w_b = rng.uniform(0.5, 1.5, K_y)
    return K, w_a, w_b


def _einsum_with_optimize_true(table, K, w_A, w_B):
    """Reference computation: matches the un-precomputed path (optimize=True)."""
    total = 0.0
    for orb in table:
        operands = []
        for alpha in range(orb.qA):
            operands.append(w_A ** orb.m_A[alpha])
        for beta in range(orb.qB):
            operands.append(w_B ** orb.m_B[beta])
        for _, _, m in orb.edges:
            operands.append(K if m == 1 else K ** m)
        contribution = np.einsum(orb.einsum_str, *operands, optimize=True)
        total += orb.weight * orb.mu * contribution
    return float(total)


@pytest.mark.parametrize("r,K_size", [(2, 8), (3, 8), (3, 12), (4, 8)])
def test_inner_product_orbit_path_matches_optimize_true(r, K_size):
    """Precomputed path produces same result as optimize=True for IP."""
    K, w_A, w_B = _make_kernel(K_size, K_size, seed=r * 7 + K_size)
    out_path = inner_product_orbit(K, w_A, w_B, r, prefactor=1.0)
    table = get_orbit_table(r)
    out_ref = _einsum_with_optimize_true(table, K, w_A, w_B)
    assert abs(out_path - out_ref) < ATOL + RTOL * abs(out_ref)


@pytest.mark.parametrize("r,K_size,N_grid", [(2, 6, 5), (3, 8, 8), (4, 6, 4)])
def test_inner_product_orbit_grid_path_matches_optimize_true(r, K_size, N_grid):
    """Precomputed path matches reference for grid variant."""
    rng = np.random.default_rng(r * 11 + K_size)
    K_u = np.exp(-rng.uniform(0, 1, (N_grid, K_size, K_size)))
    w_A = rng.uniform(0.5, 1.5, K_size)
    w_B = rng.uniform(0.5, 1.5, K_size)

    out_path = inner_product_orbit_grid(K_u, w_A, w_B, r, prefactor=1.0)

    # Reference: rebuild the einsum_str_grid with optimize=True
    table = get_orbit_table(r)
    out_ref = np.zeros(N_grid, dtype=K_u.dtype)
    for orb in table:
        operands = []
        for alpha in range(orb.qA):
            operands.append(w_A ** orb.m_A[alpha])
        for beta in range(orb.qB):
            operands.append(w_B ** orb.m_B[beta])
        for _, _, m in orb.edges:
            operands.append(K_u if m == 1 else K_u ** m)
        contribution = np.einsum(
            orb.einsum_str_grid, *operands, optimize=True)
        out_ref += orb.weight * orb.mu * contribution
    assert np.allclose(out_path, out_ref, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("r,K_size,N_pairs", [(2, 6, 8), (3, 8, 5), (4, 6, 3)])
def test_inner_product_orbit_pw_batched_path_matches_optimize_true(
        r, K_size, N_pairs):
    """Precomputed path + precomputed string match runtime-patched + optimize=True."""
    rng = np.random.default_rng(r * 13 + K_size)
    K_g = np.exp(-rng.uniform(0, 1, (N_pairs, K_size, K_size)))
    w_A_g = rng.uniform(0.5, 1.5, (N_pairs, K_size))
    w_B_g = rng.uniform(0.5, 1.5, (N_pairs, K_size))

    out_path = inner_product_orbit_pw_batched(
        K_g, w_A_g, w_B_g, r, prefactor=1.0)

    # Reference: rebuild via runtime string-patching, optimize=True
    table = get_orbit_table(r)
    out_ref = np.zeros(N_pairs, dtype=K_g.dtype)
    for orb in table:
        in_part, _, out_part = orb.einsum_str_grid.partition("->")
        specs = in_part.split(",")
        new_specs = [
            ("u" + s) if i < orb.qA + orb.qB else s
            for i, s in enumerate(specs)
        ]
        new_es = ",".join(new_specs) + "->" + out_part
        operands = []
        for alpha in range(orb.qA):
            operands.append(w_A_g ** orb.m_A[alpha])
        for beta in range(orb.qB):
            operands.append(w_B_g ** orb.m_B[beta])
        for _, _, m in orb.edges:
            operands.append(K_g if m == 1 else K_g ** m)
        contribution = np.einsum(new_es, *operands, optimize=True)
        out_ref += orb.weight * orb.mu * contribution
    assert np.allclose(out_path, out_ref, rtol=RTOL, atol=ATOL)


def test_orbit_entries_have_paths_after_load():
    """Loaded orbit tables (via _ensure_paths) have all path fields populated."""
    for r in [2, 3, 4, 5]:
        table = get_orbit_table(r)
        assert len(table) > 0
        for orb in table:
            assert orb.einsum_path is not None, \
                f"einsum_path missing on r={r} orbit"
            assert orb.einsum_path_grid is not None, \
                f"einsum_path_grid missing on r={r} orbit"
            assert orb.einsum_path_pw_batched is not None, \
                f"einsum_path_pw_batched missing on r={r} orbit"
            assert orb.einsum_str_pw_batched is not None, \
                f"einsum_str_pw_batched missing on r={r} orbit"
            # Path is a list starting with 'einsum_path' followed by tuples.
            assert isinstance(orb.einsum_path, list)
            assert orb.einsum_path[0] == 'einsum_path'
