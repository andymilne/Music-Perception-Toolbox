"""v2.2.x — SA centres-path n_q chunking regression test.

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
    """Regression tests for memory-aware n_q chunking in the SA
    centres fast-path.
    """

    def test_completes_without_oom_at_demo_scale(self):
        """K=72, r=3, rel, non-per at n_q=300 triggers chunking
        under the 1 GB per-chunk budget (per-slot ~8.6 MB →
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
        """For K=72 r=3 rel non-per at this sigma/period, the rel-mode
        pre-screen fires and routes to centres without probing —
        the same path that demo_triadConsonance uses. Verify that
        auto routing produces the same chunked output as forced
        'centres'.
        """
        K = 72
        p = np.linspace(0, 1200, K, endpoint=False)
        dens = build_exp_tens(p, np.ones(K), 12.0, 3,
                              True, False, 0.0, verbose=False)
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 1200, (2, 300))

        v_auto = eval_exp_tens(dens, x, verbose=False)
        v_centres = eval_exp_tens(dens, x, method='centres', verbose=False)

        # Auto routing and forced centres take chunker paths whose
        # chunk sizes are independently computed against the resolved
        # kernel_chunk_bytes budget. With 'auto' (the factory default),
        # the budget tracks available physical memory and so can vary
        # by a few bytes between calls, which produces 1-ulp-class
        # reduction-order differences for large summations. Bit
        # identity is not part of the contract here; numerical
        # agreement to ~1e-13 relative is.
        np.testing.assert_allclose(v_auto, v_centres, rtol=1e-13, atol=0)
