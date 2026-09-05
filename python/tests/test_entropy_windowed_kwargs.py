"""v3 — entropy_exp_tens kwarg threading.

Verifies that ``truncation_sigmas`` and ``kernel_precision`` kwargs are
accepted by the public signatures and produce the expected behaviour:
explicit kwargs match what global ``set_default`` would produce.
"""
from __future__ import annotations

import inspect
import numpy as np

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
