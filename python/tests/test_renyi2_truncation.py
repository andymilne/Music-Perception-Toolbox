"""The per-call ``truncation_sigmas`` and ``kernel_precision`` reach the
Rényi-2 self inner product.

``entropy_maet(method='renyi2')`` forms ``H_2 = -log_b(<T,T> / Z^2)``
with ``<T,T> = sim_maet(T, T, normalize='none')``. The total mass ``Z``
is closed-form and independent of the kernel cutoff, so a change of
``truncation_sigmas`` must move ``H_2`` by exactly
``-log_b(<T,T>_k / <T,T>_default)``, and the kernel-covariance
change-of-variables term ``log det(Sigma) / 2`` must be added once at
every width.
"""

import math

import numpy as np
import pytest

import mpt
from mpt import build_maet, entropy_maet, sim_maet


def _h(dens, **kw):
    return entropy_maet(dens, method="renyi2", verbose=False, **kw)


def _ip(dens, **kw):
    return float(sim_maet(dens, dens, normalize="none", verbose=False, **kw))


@pytest.fixture
def dens_periodic():
    p = np.array([0.0, 400.0, 700.0, 1000.0])
    return build_maet(p, np.ones(4), 60.0, 2, True, True, 1200.0,
                      verbose=False)


@pytest.fixture
def dens_kernel_cov():
    # Two absolute non-periodic dimensions; event spacings of a few
    # whitened sigmas so that a 3-sigma cutoff drops real overlap.
    Sigma = np.array([[1.0, 0.3], [0.3, 0.8]])
    P = np.array([[0.0, 2.2, 4.1, 7.0], [0.0, 1.9, 4.4, 6.3]])
    W = np.ones_like(P)
    return (build_maet([P], [W], [Sigma], [2], [False], [False], [0.0],
                       [False], verbose=False),
            Sigma, P, W)


class TestRenyi2Truncation:

    def test_small_width_moves_value(self, dens_periodic):
        h_def = _h(dens_periodic)
        h3 = _h(dens_periodic, truncation_sigmas=3.0)
        assert h3 != h_def
        assert abs(h3 - h_def) < 1e-3

    def test_equals_direct_formula(self, dens_periodic):
        # Z^2 from the accuracy-floor value; Z does not depend on the
        # cutoff, so H_2 at 3 sigma must equal -log_2(<T,T>_3 / Z^2).
        base = 2.0
        h_inf = _h(dens_periodic, truncation_sigmas=math.inf, base=base)
        ip_inf = _ip(dens_periodic, truncation_sigmas=math.inf)
        z2 = ip_inf * base ** h_inf
        ip3 = _ip(dens_periodic, truncation_sigmas=3.0)
        want = -math.log(ip3 / z2, base)
        np.testing.assert_allclose(
            _h(dens_periodic, truncation_sigmas=3.0, base=base), want,
            rtol=1e-12)

    def test_none_is_default(self, dens_periodic):
        assert _h(dens_periodic, truncation_sigmas=None) == _h(dens_periodic)

    def test_inf_is_accuracy_floor(self, dens_periodic):
        prev = mpt.get_default("truncation_sigmas")
        try:
            mpt.set_default(truncation_sigmas=math.inf)
            h_global = _h(dens_periodic)
        finally:
            mpt.set_default(truncation_sigmas=prev)
        h_inf = _h(dens_periodic, truncation_sigmas=math.inf)
        np.testing.assert_allclose(h_inf, h_global, rtol=1e-14)
        np.testing.assert_allclose(h_inf, _h(dens_periodic), rtol=1e-7)

    def test_kernel_cov_logdet_added_once(self, dens_kernel_cov):
        dens, Sigma, P, W = dens_kernel_cov
        h_def = _h(dens, base=math.e)
        h3 = _h(dens, truncation_sigmas=3.0, base=math.e)
        assert h3 != h_def
        ip_def = _ip(dens)
        ip3 = _ip(dens, truncation_sigmas=3.0)
        np.testing.assert_allclose(h3 - h_def, -math.log(ip3 / ip_def),
                                   rtol=1e-10, atol=1e-15)

    def test_kernel_cov_scalar_equivalent(self):
        # A matrix sigma^2 I and the scalar sigma agree at a small width
        # too: the log det term is not double counted when the cutoff
        # reaches the self inner product.
        P = np.array([[0.0, 1.5, 3.1, 4.0], [0.0, 1.2, 2.9, 4.6]])
        W = np.ones_like(P)
        s = 0.8
        geom = ([2], [False], [False], [0.0], [False])
        dm = build_maet([P], [W], [s * s * np.eye(2)], *geom, verbose=False)
        ds = build_maet([P], [W], [s], *geom, verbose=False)
        np.testing.assert_allclose(_h(dm, truncation_sigmas=3.0),
                                   _h(ds, truncation_sigmas=3.0),
                                   rtol=1e-12)

    @pytest.mark.filterwarnings("ignore:rel = True combined with r_a = 1")
    def test_sub_density_branch(self):
        # A relative r = 1 attribute routes through the rebuilt
        # sub-density; the width must reach that call too.
        P1 = np.array([[0.0, 1.5, 3.1, 4.0], [0.0, 1.2, 2.9, 4.6]])
        T = np.array([[0.0, 0.3, 0.9, 1.4]])
        w = [np.ones_like(P1), np.ones_like(T)]
        dens = build_maet([P1, T], w, [0.8, 0.2], [2, 1], [False, True],
                          [False, False], [0.0, 0.0], [False, True],
                          verbose=False)
        assert _h(dens, truncation_sigmas=3.0) != _h(dens)

    def test_kernel_precision_single(self, dens_periodic):
        h_s = _h(dens_periodic, kernel_precision="single")
        np.testing.assert_allclose(h_s, _h(dens_periodic), rtol=1e-5)
