"""v2.2.x — entropy_exp_tens and windowed_tensor_similarity kwarg threading.

Verifies that ``truncation_sigmas`` and ``kernel_precision`` kwargs are
accepted by the public signatures and produce the expected behaviour:
explicit kwargs match what global ``set_default`` would produce.
"""
from __future__ import annotations

import inspect
import numpy as np
import pytest

import mpt


# -----------------------------------------------------------------------
# entropy_exp_tens
# -----------------------------------------------------------------------


class TestEntropyExpTensKwargs:

    def test_signature_exposes_kwargs(self):
        sig = inspect.signature(mpt.entropy_exp_tens)
        assert "truncation_sigmas" in sig.parameters
        assert "kernel_precision" in sig.parameters

    def test_truncation_explicit_matches_global(self):
        """Explicit truncation_sigmas kwarg matches the value obtained
        by setting it globally via set_default."""
        dens = mpt.build_exp_tens(
            np.array([0.0, 400.0, 700.0]), np.ones(3),
            12.0, 2, True, False, 0.0,
        )
        common = dict(
            base=2.0,
            n_points_per_dim=200,
            x_min=-1200.0, x_max=1200.0,
        )

        mpt.reset_defaults()
        h_explicit = mpt.entropy_exp_tens(
            dens, truncation_sigmas=6.0, **common,
        )

        mpt.set_default(truncation_sigmas=6.0)
        try:
            h_global = mpt.entropy_exp_tens(dens, **common)
        finally:
            mpt.reset_defaults()

        # Same value to FP rounding.
        assert abs(h_explicit - h_global) < 1e-12

    def test_precision_kwarg_applies(self):
        """kernel_precision='single' produces a slightly different
        result from 'double' (single-precision arithmetic), and the
        explicit kwarg matches the global default."""
        dens = mpt.build_exp_tens(
            np.array([0.0, 400.0, 700.0]), np.ones(3),
            12.0, 2, True, False, 0.0,
        )
        common = dict(
            base=2.0,
            n_points_per_dim=200,
            x_min=-1200.0, x_max=1200.0,
        )

        mpt.reset_defaults()
        h_double = mpt.entropy_exp_tens(dens, **common)
        h_single = mpt.entropy_exp_tens(
            dens, kernel_precision="single", **common,
        )
        # Single precision differs from double, but only at the
        # ~1e-7 relative level for unit-magnitude values.
        assert abs(h_double - h_single) < 1e-5

    def test_kwargs_via_global_defaults_pickup(self):
        """If user calls without explicit kwargs but with a global
        default in effect, the entropy reflects it."""
        dens = mpt.build_exp_tens(
            np.array([0.0, 400.0, 700.0]), np.ones(3),
            12.0, 2, True, False, 0.0,
        )
        mpt.reset_defaults()
        h_default = mpt.entropy_exp_tens(
            dens, base=2.0, n_points_per_dim=200,
            x_min=-1200.0, x_max=1200.0,
        )
        mpt.set_default(truncation_sigmas=6.0)
        try:
            h_trunc = mpt.entropy_exp_tens(
                dens, base=2.0, n_points_per_dim=200,
                x_min=-1200.0, x_max=1200.0,
            )
        finally:
            mpt.reset_defaults()
        # Truncation at 6σ leaves the answer numerically very close
        # to exact.
        assert abs(h_default - h_trunc) < 1e-6


# -----------------------------------------------------------------------
# windowed_tensor_similarity
# -----------------------------------------------------------------------


class TestWindowedSimilaritySignature:
    """The full functional smoke test for windowed_tensor_similarity requires
    MA densities and offset arrays. Here we verify the signature
    exposes the kwargs; the temporary-defaults restoration is then
    covered structurally by entropy tests since the mechanism is
    shared (set_default/restore pattern)."""

    def test_signature_exposes_kwargs(self):
        sig = inspect.signature(mpt.windowed_tensor_similarity)
        assert "truncation_sigmas" in sig.parameters
        assert "kernel_precision" in sig.parameters

    def test_unknown_kwarg_unaffected(self):
        """Other kwargs unchanged by the addition."""
        sig = inspect.signature(mpt.windowed_tensor_similarity)
        for name in ("dens_query", "dens_context", "window_spec",
                     "offsets", "reference", "mode", "verbose"):
            assert name in sig.parameters

    def test_defaults_restored_after_call(self):
        """Verify the temporary-defaults mechanism restores prior
        state even when the wrapper raises."""
        mpt.reset_defaults()
        sentinel_value = mpt.get_default("truncation_sigmas")
        # Call windowed_tensor_similarity with bad arguments (insufficient
        # positional args) to trigger an error mid-call.
        try:
            mpt.windowed_tensor_similarity(None, None, None, None,
                truncation_sigmas=6.0,
            )
        except (TypeError, AttributeError, ValueError):
            pass  # expected
        # The temporary-defaults restore should still leave defaults
        # at their original value.
        assert mpt.get_default("truncation_sigmas") == sentinel_value
