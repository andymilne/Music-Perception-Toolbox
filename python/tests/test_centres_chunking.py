"""v2.2.x — single-multiset centres-path n_q chunking regression test.

Catches the OOM that surfaced when demo_triadConsonance ran at the
10-cent grid: K=72, r=3, rel, non-periodic, n_q=29161 — the
difference tensor would have been 2 × 357840 × 29161 = 155 GB
unchunked.

This regression slipped through because every existing routing /
parity case used n_q in the 20-50 range, well below the chunking
threshold. Any new fast-path or routing change that touches
centres-path evaluation MUST keep this test passing — verifying
small-n_q parity alone is not sufficient.
"""
from __future__ import annotations

import numpy as np
import pytest

import mpt
from mpt import build_exp_tens, eval_exp_tens


class TestCentresChunking:
    """Regression tests for memory-aware n_q chunking in the single-multiset
    centres fast-path.
    """

    def test_completes_without_oom_at_demo_scale(self):
        """K=72, r=3, rel, non-per at n_q=300 triggers chunking
        under the 1 GB per-chunk budget (per-value ~8.6 MB →
        2.6 GB unchunked). Should complete and return finite output.
        """
        K = 72
        p = np.linspace(0, 1200, K, endpoint=False)
        dens = build_exp_tens(p, np.ones(K), 12.0, 3,
                              True, False, 0.0, verbose=False)
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 1200, (2, 300))

        v = eval_exp_tens(dens, x, method='centres', verbose=False)
        assert v.shape == (300,)
        assert np.all(np.isfinite(v))

    def test_chunked_matches_manual_split(self):
        """Chunking is purely a memory-management detail: per-query
        output depends only on the q-th query, so the full chunked
        call must equal the concatenation of per-half results to
        numerical precision. Matmul block-size differences between
        chunks of different sizes can introduce ULP-level variation,
        so we use a tight relative tolerance rather than bit-identity.
        """
        K = 72
        p = np.linspace(0, 1200, K, endpoint=False)
        dens = build_exp_tens(p, np.ones(K), 12.0, 3,
                              True, False, 0.0, verbose=False)
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 1200, (2, 300))

        v_full = eval_exp_tens(dens, x, method='centres', verbose=False)

        mid = x.shape[1] // 2
        v_half1 = eval_exp_tens(dens, x[:, :mid],
                                method='centres', verbose=False)
        v_half2 = eval_exp_tens(dens, x[:, mid:],
                                method='centres', verbose=False)
        v_manual = np.concatenate([v_half1, v_half2])

        assert v_full.shape == v_manual.shape == (300,)
        np.testing.assert_allclose(v_full, v_manual, rtol=1e-13, atol=0)

    def test_auto_routing_matches_forced_centres(self):
        """For K=48 r=3 rel PERIODIC at sigma/P = 1/120, auto routing
        must produce the same values as forced 'centres' to within the
        accuracy floor.

        Auto is free to route to whichever path the cost model prices
        cheaper (for this shape the Möbius path is measured several
        times faster than centres, so auto legitimately returns Möbius
        values); the invariant is routing-independent correctness, not
        path identity. Möbius values agree with centres only to the
        u-grid quadrature accuracy tied to truncation_sigmas (the suite
        baseline resolves to the 1e-12 floor), so the comparison is a
        floor-scaled allclose rather than bitwise.
        """
        K = 48
        p = np.linspace(0, 1200, K, endpoint=False)
        dens = build_exp_tens(p, np.ones(K), 10.0, 3,
                              True, True, 1200.0, verbose=False)
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 1200, (2, 300))

        # Pin kernel_chunk_bytes to a fixed value so both paths compute
        # their chunk sizes against the same byte budget. The function-
        # scoped defaults-reset fixture in conftest.py restores the
        # factory state on teardown.
        mpt.set_default(kernel_chunk_bytes=200_000_000)

        v_auto = eval_exp_tens(dens, x, verbose=False)
        v_centres = eval_exp_tens(dens, x, method='centres', verbose=False)

        peak = float(np.max(np.abs(v_centres)))
        np.testing.assert_allclose(v_auto, v_centres,
                                   rtol=0.0, atol=1e-10 * peak)
