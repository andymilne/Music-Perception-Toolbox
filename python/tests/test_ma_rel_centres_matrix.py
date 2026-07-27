"""Centres route against the grid route, at inner-matrix level.

Twin of the MATLAB tests/test_ma_rel_centres.m unit check. The rest of
the Python centres coverage compares scalar cosines, which average the
near-zero cells away; this compares the inner matrices entrywise, which
is the granularity at which a divergence between the two routes first
shows.

Metric note. The two routes agree only up to a constant factor, so the
natural check is that the entrywise ratio is constant. But the inner
matrix spans some twenty orders of magnitude --- the Moebius
alternating sum cancels to the noise floor in a few cells --- and the
ratio at a cancellation-noise cell is arbitrary. A raw
``max(ratio)/min(ratio)`` is therefore set by whichever cell happened to
cancel hardest, not by any property of the routes. Two assertions are
used instead: the ratio is constant over entries carrying signal, and
the least-squares residual is small measured against the matrix scale,
so the discarded cells cannot hide a real divergence.
"""
import contextlib
import io
import sys

import numpy as np
import pytest

import mpt
from mpt._tensor import cosine as _c
import mpt._tensor._mobius_inner as _mobius_inner

N_EVENTS = 8
K = 4
R = 2
PITCH_ATTR = 1          # attribute 0 is the scalar onset
SIGMA = (15.0, 6.0)
PERIOD = (4000.0, 1200.0)


@contextlib.contextmanager
def _quiet():
    orig = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = orig


def _make(seed):
    rng = np.random.default_rng(seed)
    onsets = np.sort(rng.uniform(0, PERIOD[0], (K, N_EVENTS)), axis=0)
    pitches = rng.uniform(0, PERIOD[1], (K, N_EVENTS))
    ones = np.ones((K, N_EVENTS))
    with _quiet():
        return mpt.build_exp_tens(
            [onsets, pitches], [ones, ones],
            list(SIGMA), [1, R], [False, True], [False, True],
            list(PERIOD), verbose=False,
        )


def _grid_matrix(dx, dy, spectral):
    a = PITCH_ATTR
    prev = _mobius_inner._SPECTRAL_IP_ENABLED
    _mobius_inner._SPECTRAL_IP_ENABLED = spectral
    try:
        with _quiet():
            return np.asarray(_mobius_inner._rel_inner_batched(
                dx.p_attr[a], dx.w[a], dy.p_attr[a], dy.w[a],
                float(dx.sigma[a]), int(dx.r[a]), True,
                float(dx.period[a]), truncation_sigmas=float('inf'),
            ))
    finally:
        _mobius_inner._SPECTRAL_IP_ENABLED = prev


@pytest.fixture(scope="module")
def matrices():
    dx = _make(17)
    dy = _make(28)
    a = PITCH_ATTR
    cx = _mobius_inner._closed_form_attr_centres(dx, a)
    cy = _mobius_inner._closed_form_attr_centres(dy, a)
    I_centres = np.asarray(_mobius_inner._closed_form_attr_matrix_from(cx, cy))
    return dx, dy, I_centres


# ---------------------------------------------------------------------
# Both routes are proportional to the centres matrix, with the same
# constant, whichever computes the grid side
# ---------------------------------------------------------------------

@pytest.mark.parametrize("spectral", [False, True],
                         ids=["taugrid", "spectral"])
def test_ratio_constant_over_entries_carrying_signal(matrices, spectral):
    dx, dy, I_centres = matrices
    I_grid = _grid_matrix(dx, dy, spectral)
    scale = np.abs(I_grid).max()
    live = np.abs(I_grid) > 1e-8 * scale
    assert live.sum() >= I_grid.size - 2, (
        "too many cells discarded; the fixture is not exercising the check"
    )
    ratio = I_centres[live] / I_grid[live]
    spread = ratio.max() / ratio.min() - 1.0
    assert spread < 1e-6, f"ratio spread {spread:.3e}"


@pytest.mark.parametrize("spectral", [False, True],
                         ids=["taugrid", "spectral"])
def test_scale_relative_residual(matrices, spectral):
    dx, dy, I_centres = matrices
    I_grid = _grid_matrix(dx, dy, spectral)
    scale = np.abs(I_grid).max()
    c = float((I_centres.ravel() @ I_grid.ravel())
              / (I_grid.ravel() @ I_grid.ravel()))
    resid = np.abs(I_centres - c * I_grid).max() / (abs(c) * scale)
    assert resid < 1e-10, f"residual {resid:.3e}"


def test_both_routes_fit_the_same_constant(matrices):
    # The constant absorbs only sigma, r, and the mode flags, so it must
    # not depend on which route computed the grid side.
    dx, dy, I_centres = matrices
    consts = []
    for spectral in (False, True):
        I_grid = _grid_matrix(dx, dy, spectral)
        consts.append(float((I_centres.ravel() @ I_grid.ravel())
                            / (I_grid.ravel() @ I_grid.ravel())))
    rel = abs(consts[0] - consts[1]) / abs(consts[0])
    assert rel < 1e-10, f"constants differ by {rel:.3e}: {consts}"


# ---------------------------------------------------------------------
# The metric itself: a raw entrywise ratio is not a valid constancy
# test here, and this records why the masked form is used
# ---------------------------------------------------------------------

def test_inner_matrix_spans_many_orders_of_magnitude(matrices):
    # If this ever stops holding, the masking above becomes unnecessary
    # and the simpler raw-ratio assertion could be restored.
    dx, dy, _ = matrices
    I_grid = _grid_matrix(dx, dy, spectral=True)
    lo = np.abs(I_grid).min()
    hi = np.abs(I_grid).max()
    assert hi / max(lo, 1e-300) > 1e6, (
        "inner matrix no longer spans a wide dynamic range"
    )
