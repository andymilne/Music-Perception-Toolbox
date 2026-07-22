"""Sparse-orbit fast path in the Möbius multi-attribute inner product.

When a non-periodic attribute's slot kernel is large and well-separated,
the orbit path contracts a spatially-culled sparse kernel instead of the
dense batched kernel. The result must equal the dense-orbit result to
floating point, and the gate must stay dormant for small or dense kernels.
"""
import numpy as np
import pytest

import mpt
from mpt import build_exp_tens, cos_sim_exp_tens
import mpt._tensor.cosine as C


def _clustered(n, rng, ncl=8, span=70.0, gap=600.0):
    per = n // ncl
    return np.concatenate([np.sort(rng.uniform(k * gap, k * gap + span, per))
                           for k in range(ncl)]).reshape(-1, 1)


@pytest.fixture
def _restore_threshold():
    lo = C._ORBIT_SPARSE_MIN_KERNEL
    orig = C._orbit_safe_submatrix_sparse
    yield
    C._ORBIT_SPARSE_MIN_KERNEL = lo
    C._orbit_safe_submatrix_sparse = orig


def _patch_counter():
    calls = {"n": 0}
    orig = C._orbit_safe_submatrix_sparse

    def wrapped(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    C._orbit_safe_submatrix_sparse = wrapped
    return calls


def test_sparse_orbit_matches_dense_orbit(_restore_threshold):
    mpt.set_default(truncation_sigmas=6.0)   # cull active (conftest pins inf)
    rng = np.random.default_rng(4)
    K = 480
    px = [_clustered(K, rng), _clustered(K, rng)]
    py = [_clustered(K, rng), _clustered(K, rng)]
    dx = build_exp_tens(px, None, [50., 50.], [2, 2], [False, False],
                        [False, False], [0, 0], verbose=False)
    dy = build_exp_tens(py, None, [50., 50.], [2, 2], [False, False],
                        [False, False], [0, 0], verbose=False)

    calls = _patch_counter()
    C._ORBIT_SPARSE_MIN_KERNEL = 200_000
    v_sparse = cos_sim_exp_tens(dx, dy, method="mobius", verbose=False)
    assert calls["n"] > 0                      # gate fired

    C._ORBIT_SPARSE_MIN_KERNEL = 10 ** 12      # disable -> dense orbit
    v_dense = cos_sim_exp_tens(dx, dy, method="mobius", verbose=False)
    assert abs(v_sparse - v_dense) / abs(v_dense) < 1e-12


def test_sparse_orbit_dormant_for_small_kernels(_restore_threshold):
    mpt.set_default(truncation_sigmas=6.0)
    rng = np.random.default_rng(5)
    px = [np.sort(rng.uniform(0, 100, 8)).reshape(-1, 1)]
    py = [np.sort(rng.uniform(0, 100, 8)).reshape(-1, 1)]
    dx = build_exp_tens(px, None, [40.], [3], [False], [False], [0],
                        verbose=False)
    dy = build_exp_tens(py, None, [40.], [3], [False], [False], [0],
                        verbose=False)
    calls = _patch_counter()
    C._ORBIT_SPARSE_MIN_KERNEL = 200_000
    cos_sim_exp_tens(dx, dy, method="mobius", verbose=False)
    assert calls["n"] == 0                     # small kernel stays dense


def test_sparse_orbit_engine_matches_dense_all_arities():
    import scipy.sparse as sp
    from mpt._mobius import inner_product_orbit, inner_product_orbit_sparse
    rng = np.random.default_rng(3)
    for r in range(2, 9):
        n = 16
        pX = np.sort(rng.uniform(0, 200, n))
        pY = pX + rng.uniform(-3, 3, n)
        s = 40.0
        D = pX[:, None] - pY[None, :]
        Kd = np.exp(-(D * D) / (4 * s * s))
        Kd[(D * D) > 2 * 36 * s * s] = 0.0
        Ks = sp.csr_matrix(Kd)
        wA = rng.uniform(0.5, 1.5, n)
        wB = rng.uniform(0.5, 1.5, n)
        vd, rd = inner_product_orbit(Kd, wA, wB, r, prefactor=1.3,
                                     return_cancellation_ratio=True)
        vs, rs = inner_product_orbit_sparse(Ks, wA, wB, r, prefactor=1.3,
                                            return_cancellation_ratio=True)
        assert abs(vd - vs) / abs(vd) < 1e-12
        assert abs(rd - rs) < 1e-12


# ---------------------------------------------------------------------
# Periodic relative-mode sparse route
# ---------------------------------------------------------------------

def test_rel_per_sparse_kernel_bitwise():
    """The circular sparse builder densifies to the truncated dense
    kernel bit-for-bit (same retention decision, same arithmetic)."""
    from mpt._tensor.cosine import (
        _rel_per_sparse_prep, _build_sparse_kernel_rel_per,
        _trunc_kernel_exp)
    from mpt._defaults import truncation_ip_sqdist
    rng = np.random.default_rng(11)
    P = 1200.0
    for trial in range(6):
        Kx = int(rng.integers(5, 50))
        Ky = int(rng.integers(5, 50))
        sigma = float(rng.uniform(3, 15))
        ts = [6.0, np.inf][trial % 2]
        pX = rng.uniform(-500, 2000, Kx)     # unfolded on purpose
        pY = rng.uniform(-500, 2000, Ky)
        cutoff = float(truncation_ip_sqdist(ts, sigma))
        c3, j3 = _rel_per_sparse_prep(pY, P)
        for u in rng.uniform(0, P, 3):
            Ks = _build_sparse_kernel_rel_per(
                pX, c3, j3, pY, sigma, cutoff, P, float(u)).toarray()
            d = pX[:, None] + u - pY[None, :]
            d = d - P * np.floor(d / P + 0.5)
            Kd = _trunc_kernel_exp(d ** 2, sigma, ts)
            np.testing.assert_array_equal(Ks, Kd)


def test_rel_per_sparse_route_matches_dense():
    """Gate-forced sparse rel-per inner products match the dense slab
    route to floating point, value and mass-aware ratio alike."""
    import mpt._tensor.cosine as C
    from mpt._tensor.cosine import _ma_per_attr_inner_matrix_rel
    rng = np.random.default_rng(5)
    lo = C._ORBIT_SPARSE_MIN_KERNEL
    try:
        for trial in range(4):
            K = int(rng.integers(12, 35))
            P = 1200.0
            sigma = float(rng.uniform(3, 8))
            r = int(rng.integers(2, 4))
            ts = [6.0, np.inf][trial % 2]
            Px = rng.uniform(0, P, (K, 1))
            Py = rng.uniform(0, P, (K, 1))
            Wx = np.ones((K, 1))
            Wy = np.ones((K, 1))
            C._ORBIT_SPARSE_MIN_KERNEL = 1
            vs, rs = _ma_per_attr_inner_matrix_rel(
                Px, Wx, Py, Wy, sigma, r, True, P,
                return_cancellation_ratio=True, truncation_sigmas=ts)
            C._ORBIT_SPARSE_MIN_KERNEL = 10 ** 12
            vd, rd = _ma_per_attr_inner_matrix_rel(
                Px, Wx, Py, Wy, sigma, r, True, P,
                return_cancellation_ratio=True, truncation_sigmas=ts)
            assert abs(float(vs[0, 0]) - float(vd[0, 0])) \
                <= 1e-13 * max(abs(float(vd[0, 0])), 1e-300)
            assert abs(rs - rd) <= 1e-12
    finally:
        C._ORBIT_SPARSE_MIN_KERNEL = lo


def test_rel_per_sparse_matches_bulger():
    """Gate-forced sparse rel-per cosine agrees with the closed-form
    pairwise (bulger) reference to well within the accuracy floor."""
    import mpt
    import mpt._tensor.cosine as C
    rng = np.random.default_rng(9)
    lo = C._ORBIT_SPARSE_MIN_KERNEL
    try:
        C._ORBIT_SPARSE_MIN_KERNEL = 1
        for r in (2, 3):
            K = 20
            P = 1200.0
            sigma = 8.0
            p1 = np.sort(rng.uniform(0, P, K))
            p2 = np.sort(rng.uniform(0, P, K))
            w = np.ones(K)
            cm = mpt.cos_sim_exp_tens(p1, w, p2, w, sigma, r, True, True, P,
                                      method='mobius', verbose=False)
            cb = mpt.cos_sim_exp_tens(p1, w, p2, w, sigma, r, True, True, P,
                                      method='bulger', verbose=False)
            assert abs(cm - cb) < 1e-9
    finally:
        C._ORBIT_SPARSE_MIN_KERNEL = lo


def test_rel_per_gate_requires_window_inside_circle():
    """With the truncation window at least half the circle the sparse
    route must not engage (the replication trick would double count);
    the dense route runs and the value is unaffected by the size
    threshold."""
    import mpt._tensor.cosine as C
    from mpt._tensor.cosine import _ma_per_attr_inner_matrix_rel
    rng = np.random.default_rng(2)
    K = 15
    P = 1200.0
    sigma = 200.0          # 2*sqrt(2)*6*sigma >> P: window fails
    Px = rng.uniform(0, P, (K, 1))
    Py = rng.uniform(0, P, (K, 1))
    Wx = np.ones((K, 1))
    Wy = np.ones((K, 1))
    lo = C._ORBIT_SPARSE_MIN_KERNEL
    try:
        C._ORBIT_SPARSE_MIN_KERNEL = 1
        v1 = _ma_per_attr_inner_matrix_rel(
            Px, Wx, Py, Wy, sigma, 2, True, P, truncation_sigmas=6.0)
        C._ORBIT_SPARSE_MIN_KERNEL = 10 ** 12
        v2 = _ma_per_attr_inner_matrix_rel(
            Px, Wx, Py, Wy, sigma, 2, True, P, truncation_sigmas=6.0)
        np.testing.assert_array_equal(v1, v2)
    finally:
        C._ORBIT_SPARSE_MIN_KERNEL = lo
