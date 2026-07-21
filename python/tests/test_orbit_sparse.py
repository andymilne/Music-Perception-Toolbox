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
