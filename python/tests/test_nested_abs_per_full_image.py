"""The nested contraction computes the wrap the density declares.

An absolute-periodic attribute defaults to ``wrap='full-image'``: the
per-coordinate kernel is the wrapped Gaussian
``theta(d) = sum_n exp(-(d + n P)^2 / (4 sigma^2))``. The batched
contraction used to reduce ``d`` to the nearest image and exponentiate,
which is the ``'single-image'`` kernel -- a different measure once the
accuracy floor asks for more than one image. It agreed with the
materialised-centres path only below sigma/P ~ 0.05 and departed above
(4e-2 in the cosine at sigma/P = 0.2).

These pin the two routes together across sigma/P for both wraps, and pin
the batched contraction to the reference ``nested_ip`` / ``_ip_absolute``
that the parity suite treats as the definition.
"""
import numpy as np
import pytest

import mpt
from mpt import build_exp_tens, cos_sim_exp_tens


@pytest.fixture(autouse=True)
def _at_the_accuracy_floor():
    """Compare the routes at the accuracy floor, so a disagreement is a
    measure difference and not the 6-sigma truncation error."""
    prev = mpt.get_default('truncation_sigmas')
    mpt.set_default(truncation_sigmas=float('inf'))
    try:
        yield
    finally:
        mpt.set_default(truncation_sigmas=prev)

P = 12.0
CHORD, N_CHORDS, N_EVENTS = 3, 2, 2
TAGS = np.repeat(np.arange(N_CHORDS), CHORD).reshape(-1, 1)
SPEC = {"tags": TAGS, "r": [2, N_CHORDS], "sym": [True, False],
        "rel": [0, 0]}


def _dens(seed, sigma, wrap='full-image'):
    rng = np.random.default_rng(seed)
    p = np.sort(rng.uniform(0.0, P, size=(CHORD * N_CHORDS, N_EVENTS)),
                axis=0)
    return build_exp_tens([p], None, specs=[dict(SPEC)], sigma=[sigma],
                          is_per=[True], period=[P], wrap=[wrap],
                          verbose=False)


@pytest.mark.parametrize("sigma_over_P", [0.01, 0.05, 0.1, 0.2, 0.3])
@pytest.mark.parametrize("wrap", ['full-image', 'single-image'])
def test_contract_matches_centres_on_nested_abs_per(sigma_over_P, wrap):
    """The contraction and the materialised-centres path agree, at every
    sigma/P and under either wrap. Measured at the accuracy floor, so the
    tolerance is the floor and not the 6-sigma truncation error."""
    sigma = sigma_over_P * P
    x, y = _dens(1, sigma, wrap), _dens(2, sigma, wrap)
    c = cos_sim_exp_tens(x, y, method='contract', verbose=False)
    b = cos_sim_exp_tens(_dens(1, sigma, wrap), _dens(2, sigma, wrap),
                         method='bulger', verbose=False)
    assert c == pytest.approx(b, rel=1e-11, abs=1e-13)


def test_full_and_single_image_differ_where_the_floor_asks_for_images():
    """The two wraps are genuinely different measures at sigma/P = 0.2 --
    so the agreement above is not both routes taking the same shortcut."""
    sigma = 0.2 * P
    full = cos_sim_exp_tens(_dens(1, sigma, 'full-image'),
                            _dens(2, sigma, 'full-image'),
                            method='contract', verbose=False)
    single = cos_sim_exp_tens(_dens(1, sigma, 'single-image'),
                              _dens(2, sigma, 'single-image'),
                              method='contract', verbose=False)
    assert abs(full - single) > 1e-3


def test_batched_contraction_matches_the_reference_ip():
    """The (N_x, N_y) batched matrix equals the per-event-pair reference
    inner product, which sums the images explicitly."""
    from mpt._tensor._nested_contraction import (
        build_recipe, nested_attr_matrix, make_quadrature, nested_ip)
    sigma = 0.2 * P
    x, y = _dens(1, sigma), _dens(2, sigma)
    r_levels = np.asarray(SPEC["r"])
    sym_levels = np.asarray(SPEC["sym"])
    rec = build_recipe(r_levels, sym_levels, TAGS, is_rel=False, is_per=True)
    PX = np.asarray(x.p_attr[0], float)
    PY = np.asarray(y.p_attr[0], float)
    M = nested_attr_matrix(rec, rec, PX, PY, x.w[0], y.w[0], sigma,
                           True, P, np.inf, taus=None)
    quad = make_quadrature(False, True, sigma, P, 0.0, P,
                           truncation_sigmas=np.inf)
    wx = np.ones(PX.shape[0]) if x.w[0] is None else np.asarray(x.w[0], float)
    wy = np.ones(PY.shape[0]) if y.w[0] is None else np.asarray(y.w[0], float)
    if wx.ndim == 1:
        wx = np.tile(wx[:, None], (1, PX.shape[1]))
    if wy.ndim == 1:
        wy = np.tile(wy[:, None], (1, PY.shape[1]))
    for i in range(PX.shape[1]):
        for j in range(PY.shape[1]):
            ref = nested_ip(rec, rec, PX[:, i], PY[:, j], wx[:, i], wy[:, j],
                            sigma, P, np.inf, quad)
            assert M[i, j] == pytest.approx(ref, rel=1e-12)
