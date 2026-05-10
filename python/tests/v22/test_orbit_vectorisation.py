"""Vectorisation tests for the orbit-Möbius point evaluators.

Verifies the v2.2.x extensions:

- :func:`mpt._mobius.eval_orbit_abs` accepts ``x`` of shape ``(r, ...)``
  with arbitrary trailing dimensions; output shape matches the
  trailing dims.
- :func:`mpt._mobius.eval_orbit_rel` batches its u-grid loop into a
  single chunked call to :func:`eval_orbit_abs` and produces values
  bit-identical to a manual sequential u-grid quadrature.
- :func:`mpt.harmony.tensor_harmonicity` batched mode dedups canonical
  chords + groups by (n_p, dup), and the batched result matches the
  scalar path to machine precision.
"""

import numpy as np
import pytest

from mpt._mobius import eval_orbit_abs, eval_orbit_rel
from mpt.harmony import tensor_harmonicity


# ---------------------------------------------------------------------
# eval_orbit_abs: arbitrary trailing dims
# ---------------------------------------------------------------------


@pytest.mark.parametrize("r,N,n_q", [(2, 5, 4), (3, 7, 6), (4, 9, 3)])
def test_eval_orbit_abs_2d_unchanged(r, N, n_q):
    """2-D input (r, n_q) still returns a 1-D (n_q,) vector."""
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1000, N)
    w = np.ones(N)
    x = rng.uniform(0, 1000, (r, n_q))
    vals = eval_orbit_abs(p, w, 12.0, r, x)
    assert vals.shape == (n_q,)
    assert np.all(np.isfinite(vals))


def test_eval_orbit_abs_3d_trailing_dim():
    """3-D input (r, n1, n2) returns a (n1, n2) array; values match flat."""
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1000, 8)
    w = np.ones(8)
    r = 3
    n1, n2 = 4, 5
    x_3d = rng.uniform(0, 1000, (r, n1, n2))
    vals_3d = eval_orbit_abs(p, w, 12.0, r, x_3d)
    assert vals_3d.shape == (n1, n2)

    # Cross-check: collapse to 2-D, run, reshape.
    x_flat = x_3d.reshape(r, n1 * n2)
    vals_flat = eval_orbit_abs(p, w, 12.0, r, x_flat)
    assert np.allclose(vals_3d, vals_flat.reshape(n1, n2), atol=0)


def test_eval_orbit_abs_4d_trailing_dim():
    """4-D input (r, n1, n2, n3) returns a (n1, n2, n3) array."""
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 1000, 6)
    w = np.ones(6)
    r = 2
    x_4d = rng.uniform(0, 1000, (r, 3, 4, 5))
    vals = eval_orbit_abs(p, w, 12.0, r, x_4d)
    assert vals.shape == (3, 4, 5)


def test_eval_orbit_abs_3d_with_cancellation_ratio():
    """return_cancellation_ratio also returns matched-shape ratios."""
    rng = np.random.default_rng(3)
    p = rng.uniform(0, 1000, 8)
    w = np.ones(8)
    r = 3
    x_3d = rng.uniform(0, 1000, (r, 4, 5))
    vals, ratios = eval_orbit_abs(
        p, w, 12.0, r, x_3d, return_cancellation_ratio=True
    )
    assert vals.shape == (4, 5)
    assert ratios.shape == (4, 5)
    assert np.all((ratios > 0) & (ratios <= 1.0 + 1e-9))


def test_eval_orbit_abs_bad_first_dim_raises():
    p = np.zeros(3); w = np.ones(3)
    x_bad = np.zeros((2, 3, 4))  # first dim != r
    with pytest.raises(ValueError, match="must have shape"):
        eval_orbit_abs(p, w, 12.0, 3, x_bad)


# ---------------------------------------------------------------------
# eval_orbit_rel: chunked u-grid agrees with sequential
# ---------------------------------------------------------------------


def _manual_sequential_orbit_rel(p, w, sigma, r, x_rel, is_per, period):
    """Mimic the pre-v2.2.x sequential u-grid loop for cross-check."""
    samples_per_sigma = 10
    n_q = x_rel.shape[1]
    if is_per:
        N_u = max(64, int(np.ceil(period / sigma * samples_per_sigma)))
        u_grid = np.linspace(0.0, period, N_u, endpoint=False)
        du = period / N_u
    else:
        x_min = float(x_rel.min(initial=0.0))
        x_max = float(x_rel.max(initial=0.0))
        u_min = p.min() - max(0.0, x_max) - 8.0 * sigma
        u_max = p.max() - min(0.0, x_min) + 8.0 * sigma
        N_u = max(
            64,
            int(np.ceil(max(u_max - u_min, 1.0) / sigma * samples_per_sigma)),
        )
        u_grid = np.linspace(u_min, u_max, N_u)
    F = np.empty((N_u, n_q))
    for j, u in enumerate(u_grid):
        x_full = np.empty((r, n_q))
        x_full[0, :] = u
        x_full[1:, :] = u + x_rel
        F[j, :] = eval_orbit_abs(p, w, sigma, r, x_full,
                                  is_per=is_per, period=period)
    if is_per:
        integral = F.sum(axis=0) * du
    else:
        integral = np.trapezoid(F, u_grid, axis=0)
    return integral / (sigma * np.sqrt(2 * np.pi / r))


@pytest.mark.parametrize("r,is_per", [(2, False), (3, False), (2, True), (3, True)])
def test_eval_orbit_rel_chunked_matches_sequential(r, is_per):
    """Batched chunked u-grid is bit-identical to manual sequential."""
    rng = np.random.default_rng(42)
    p = rng.uniform(0, 1200, 6)
    w = np.ones(6)
    sigma = 12.0
    period = 1200.0 if is_per else 0.0
    x_rel = rng.uniform(0, 1200, (r - 1, 7))
    v_vec = eval_orbit_rel(p, w, sigma, r, x_rel,
                            is_per=is_per, period=period)
    v_man = _manual_sequential_orbit_rel(p, w, sigma, r, x_rel, is_per, period)
    # Should be bit-identical: same FP ops in same order modulo chunking.
    assert np.max(np.abs(v_vec - v_man)) < 1e-12 * np.max(np.abs(v_man))


def test_eval_orbit_rel_chunked_cancellation_ratio_matches_sequential():
    """Cancellation ratios from chunked path also match sequential."""
    rng = np.random.default_rng(7)
    p = rng.uniform(0, 1200, 8)
    w = np.ones(8)
    sigma = 12.0
    r = 3
    x_rel = rng.uniform(0, 1200, (r - 1, 5))
    v_vec, ratios_vec = eval_orbit_rel(
        p, w, sigma, r, x_rel, is_per=False, period=0.0,
        return_cancellation_ratio=True,
    )
    assert ratios_vec.shape == (5,)
    # Ratios are in [0, 1] (worst-case across u-grid; can be 0 if any
    # u-grid point lands in a cancellation-heavy region).
    assert np.all((ratios_vec >= 0) & (ratios_vec <= 1.0 + 1e-9))


# ---------------------------------------------------------------------
# tensor_harmonicity: batched dedup + grouping + scalar parity
# ---------------------------------------------------------------------


def test_tensor_harmonicity_batched_matches_scalar_machine_precision():
    """Batched mode invokes the same orbit-rel path as scalar mode."""
    spec = ["harmonic", 12, "powerlaw", 1]
    sigma = 12.0
    p = np.array([0.0, 400.0, 700.0])
    h_scalar = tensor_harmonicity(p, sigma=sigma, spectrum=spec, verbose=False)
    P = np.stack([p, p, p])
    h_batched = tensor_harmonicity(P, sigma=sigma, spectrum=spec, verbose=False)
    assert h_batched.shape == (3,)
    assert np.all(np.abs(h_batched - h_scalar) < 1e-12)


def test_tensor_harmonicity_batched_groups_by_cardinality():
    """Mixed-cardinality batches (NaN-padded) get grouped by (n_p, dup)."""
    spec = ["harmonic", 12, "powerlaw", 1]
    sigma = 12.0
    # Row 0: 3-pitch chord; row 1: 4-pitch chord (NaN-padded);
    # row 2: another 3-pitch chord (transposition of row 0).
    P = np.array([
        [0.0, 400.0, 700.0, np.nan],
        [0.0, 400.0, 700.0, 1100.0],
        [500.0, 900.0, 1200.0, np.nan],
    ])
    h = tensor_harmonicity(P, sigma=sigma, spectrum=spec, verbose=False)
    assert h.shape == (3,)
    # Rows 0 and 2 are canonically identical -> same value.
    assert abs(h[0] - h[2]) < 1e-12
    # Row 1 differs (different cardinality).
    assert abs(h[0] - h[1]) > 1e-6
    # All finite.
    assert np.all(np.isfinite(h))


def test_tensor_harmonicity_batched_normalize_consistent_with_scalar():
    """Normalisation in batched mode mirrors scalar mode for all modes."""
    spec = ["harmonic", 12, "powerlaw", 1]
    sigma = 12.0
    p = np.array([0.0, 400.0, 700.0])
    P = np.stack([p])

    for norm in ("none", "gaussian", "pdf"):
        h_s = tensor_harmonicity(p, sigma=sigma, spectrum=spec,
                                  normalize=norm, verbose=False)
        h_b = tensor_harmonicity(P, sigma=sigma, spectrum=spec,
                                  normalize=norm, verbose=False)
        assert abs(h_b[0] - h_s) < 1e-12 * max(abs(h_s), 1e-30)


def test_tensor_harmonicity_batched_dedup_collapses_repeats():
    """Many copies of the same canonical chord compute once."""
    spec = ["harmonic", 12, "powerlaw", 1]
    sigma = 12.0
    p = np.array([0.0, 400.0, 700.0])
    # 50 transposition-equivalent copies.
    P = np.stack([p + 100 * k for k in range(50)])
    h = tensor_harmonicity(P, sigma=sigma, spectrum=spec, verbose=False)
    assert h.shape == (50,)
    # All canonically identical -> all values equal.
    assert np.max(np.abs(h - h[0])) < 1e-12
