"""The nested tau-grid contraction averages the wrapped Gaussian.

The relative-periodic ``taugrid`` route is documented (module docstring of
``_nested_contraction``) as the all-image transposition average -- the
lattice-sum measure the flat Möbius integrator computes, which sums
periodic images before averaging over tau. The batched contraction and
the per-pair reference ``nested_ip`` used to average the *nearest-image*
Gaussian instead, a different measure once the accuracy floor asks for
more than one image (1.4e-5 in the cosine at sigma/P = 0.1, 1.8e-3 at
0.2), and one whose kink at |d| = P/2 also cost the trapezoidal rule its
spectral convergence.

The reference here is independent of the contraction: the nested tuple
set is enumerated explicitly and the transposition average taken on a
6000-node grid, once with the wrapped Gaussian per coordinate and once
with the nearest-image Gaussian, so the test can say which measure the
route computes rather than only that it changed.
"""
import itertools

import numpy as np
import pytest

import mpt
from mpt import build_exp_tens
from mpt._tensor.cosine import _nested_attr_matrix
from mpt._tensor._nested_contraction import (
    auto_ntau_default, build_recipe)
from tests.references.nested_ip_reference import make_quadrature, nested_ip
from mpt._wrapped_kernel import wrapped_gaussian_1d


@pytest.fixture(autouse=True)
def _at_the_accuracy_floor():
    prev = mpt.get_default('truncation_sigmas')
    mpt.set_default(truncation_sigmas=float('inf'))
    try:
        yield
    finally:
        mpt.set_default(truncation_sigmas=prev)


P = 12.0
CHORD, N_CHORDS, N_EVENTS = 2, 2, 2
TAGS = np.repeat(np.arange(N_CHORDS), CHORD).reshape(-1, 1)
SPEC = {"tags": TAGS, "r": [1, N_CHORDS], "sym": [True, True],
        "rel": [0, 1]}


def _pts(seed):
    rng = np.random.default_rng(seed)
    return np.sort(rng.uniform(0.0, P, size=(CHORD * N_CHORDS, N_EVENTS)),
                   axis=0)


def _dens(p, sigma):
    return build_exp_tens([p], None, specs=[dict(SPEC)], sigma=[sigma],
                          is_per=[True], period=[P], verbose=False)


def _tuples(col):
    """One value per chord, every ordering (the outer level is symmetric)."""
    chords = [col[TAGS[:, 0] == k] for k in range(N_CHORDS)]
    return np.array([perm for pick in itertools.product(*chords)
                     for perm in itertools.permutations(pick)])


def _ref_ip(px, py, sigma, full, ntau=6000):
    taus = np.linspace(0.0, P, ntau, endpoint=False)
    tot = 0.0
    for i in range(N_EVENTS):
        for j in range(N_EVENTS):
            X, Y = _tuples(px[:, i]), _tuples(py[:, j])
            d = X[:, None, :, None] - Y[None, :, :, None] - taus
            if full:
                K = wrapped_gaussian_1d(d, sigma, P, np.inf,
                                        exponent_denominator=4)
            else:
                dw = d - P * np.floor(d / P + 0.5)
                K = np.exp(-dw * dw / (4.0 * sigma ** 2))
            tot += K.prod(axis=2).sum(axis=(0, 1)).mean()
    return tot


def _ref_cos(px, py, sigma, full):
    return _ref_ip(px, py, sigma, full) / np.sqrt(
        _ref_ip(px, px, sigma, full) * _ref_ip(py, py, sigma, full))


def _taugrid_cos(px, py, sigma):
    x, y = _dens(px, sigma), _dens(py, sigma)
    taus = np.linspace(0.0, P, auto_ntau_default(P, sigma), endpoint=False)

    def m(a, b):
        return float(_nested_attr_matrix(a, b, 0, 'taugrid', taus).sum())
    return m(x, y) / np.sqrt(m(x, x) * m(y, y))


@pytest.mark.parametrize("sigma_over_P", [0.02, 0.05, 0.1, 0.2, 0.3])
def test_taugrid_is_the_all_image_transposition_average(sigma_over_P):
    sigma = sigma_over_P * P
    px, py = _pts(1), _pts(2)
    t = _taugrid_cos(px, py, sigma)
    assert t == pytest.approx(_ref_cos(px, py, sigma, True),
                              rel=1e-11, abs=1e-12)


def test_the_two_measures_differ_where_the_floor_asks_for_images():
    """At sigma/P = 0.2 the nearest-image average is a different number,
    so the agreement above is not both sides taking the same shortcut."""
    sigma = 0.2 * P
    px, py = _pts(1), _pts(2)
    assert abs(_ref_cos(px, py, sigma, True)
               - _ref_cos(px, py, sigma, False)) > 1e-4


def test_reference_nested_ip_matches_the_batched_matrix():
    """The per-pair reference and the batched contraction agree on the
    same grid at sigma/P = 0.2, where the image treatment matters."""
    sigma = 0.2 * P
    x, y = _dens(_pts(1), sigma), _dens(_pts(2), sigma)
    rec = build_recipe(np.asarray(SPEC["r"]), np.asarray(SPEC["sym"]),
                       TAGS, is_rel=True, is_per=True)
    PX = np.asarray(x.p_attr[0], float)
    PY = np.asarray(y.p_attr[0], float)
    taus = np.linspace(0.0, P, auto_ntau_default(P, sigma), endpoint=False)
    M = _nested_attr_matrix(x, y, 0, 'taugrid', taus)
    quad = make_quadrature(True, True, sigma, P, 0.0, P,
                           truncation_sigmas=np.inf)
    assert quad["mode"] == "relper" and np.allclose(quad["taus"], taus)
    wx = np.ones(PX.shape) if x.w[0] is None else np.asarray(x.w[0], float)
    wy = np.ones(PY.shape) if y.w[0] is None else np.asarray(y.w[0], float)
    for i in range(PX.shape[1]):
        for j in range(PY.shape[1]):
            ref = nested_ip(rec, rec, PX[:, i], PY[:, j], wx[:, i], wy[:, j],
                            sigma, P, np.inf, quad)
            # nested_ip sums over the grid; the batched matrix takes the
            # mean (the common 1/ntau cancels in the cosine).
            assert M[i, j] * len(taus) == pytest.approx(ref, rel=1e-12)
