"""The grid-bucket spatial index is used only where it pays for itself.

``gaussian_kernel_sum`` skips centres outside the truncation ball through
a bucket grid whose neighbour expansion costs ``3**dim`` lookups per
query. Where that exceeds the centre count there is nothing to save, and
at high dimension the ``(dim, nQ, 3**dim)`` expansion alone can exceed
memory: an r = 8 tuple-centre density with K = 9 sent 362 880 queries of
dimension 8 against 9 centres and asked for 142 GiB. The gate
``_bucket_index_worthwhile`` sends such calls to the exact chunked path,
which agrees with the bucketed sum inside the truncation floor. Twin of
the MATLAB ``test_dispatch_oom_guard.m`` r = 8, K = 9 case.
"""
import numpy as np
import pytest

import mpt
from mpt import build_exp_tens, cos_sim_exp_tens
from mpt._kernel import _bucket_index_worthwhile, gaussian_kernel_sum


@pytest.fixture(autouse=True)
def _quiet():
    prev = mpt.get_default('show_hints')
    mpt.set_default(show_hints=False)
    try:
        yield
    finally:
        mpt.set_default(show_hints=prev)


def test_gate_rejects_more_offsets_than_centres():
    assert not _bucket_index_worthwhile(8, 9, 362880)
    assert not _bucket_index_worthwhile(3, 27, 100)
    assert _bucket_index_worthwhile(3, 28, 100)


def test_r8_k9_single_multiset_cosine_completes():
    K, r = 9, 8
    g = np.arange(1, K + 1) * 37.0
    d = build_exp_tens(g, np.ones(K), 6.0, r, False, False, 0.0,
                       verbose=False)
    s = cos_sim_exp_tens(d, d, method='auto', verbose=False)
    assert s == pytest.approx(1.0, abs=1e-9)


def test_exact_and_bucketed_sums_agree_inside_the_floor():
    rng = np.random.default_rng(5)
    dim = 3
    C = rng.uniform(0.0, 100.0, size=(dim, 400))
    X = rng.uniform(0.0, 100.0, size=(dim, 50))
    w = rng.uniform(0.2, 1.0, size=400)
    v_bucket = gaussian_kernel_sum(C, w, X, 4.0, truncation_sigmas=6.0)
    v_exact = gaussian_kernel_sum(C, w, X, 4.0, truncation_sigmas=np.inf)
    assert np.max(np.abs(v_bucket - v_exact)) > 0.0   # truncation is real
    # The bucketed sum drops pairs beyond 6 sigma; the residual is that
    # discarded mass, exp(-18) ~ 1.5e-8 per discarded pair.
    np.testing.assert_allclose(v_bucket, v_exact, rtol=0.0, atol=1e-5)
