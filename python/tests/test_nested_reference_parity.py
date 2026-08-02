"""The nested reference implementations agree with the shipped path.

``_nested_contraction`` keeps a small reference triple --- ``build_recipe``,
``make_quadrature`` and ``nested_ip`` --- that computes one event-pair inner
product directly from the recipe tree, without any of the optimisations the
production path applies. It is retained because MATLAB's ``nestedIp`` is
live (``internal/nestedContract.m``), so the Python trio documents what the
optimised route computes.

A reference implementation nothing exercises documents what someone once
believed the optimised path computed. These tests make the claim checkable:
on small cases the reference must reproduce ``cos_sim_exp_tens`` to
floating-point precision, in each of the three quadrature modes the
reference distinguishes (absolute, relative-periodic, relative
non-periodic).
"""
import numpy as np
import pytest

from mpt import build_exp_tens, cos_sim_exp_tens
from mpt._tensor._nested_contraction import (
    build_recipe, make_quadrature, nested_ip,
)

P = 1200.0


def _reference_cosine(p_x, p_y, w_x, w_y, sigma, r, is_rel, is_per, period):
    """Cosine assembled from the reference inner product alone."""
    tags = np.zeros(len(p_x), dtype=np.intp)
    recipe = build_recipe([r], [True], tags, is_rel=is_rel, is_per=is_per)
    lo = float(min(p_x.min(), p_y.min()))
    hi = float(max(p_x.max(), p_y.max()))
    quad = make_quadrature(is_rel, is_per, sigma, period, lo, hi,
                           truncation_sigmas=np.inf)

    def ip(a, b, wa, wb):
        return nested_ip(recipe, recipe, a, b, wa, wb, sigma, period,
                         np.inf, quad)

    xy = ip(p_x, p_y, w_x, w_y)
    xx = ip(p_x, p_x, w_x, w_x)
    yy = ip(p_y, p_y, w_y, w_y)
    return xy / np.sqrt(xx * yy)


@pytest.mark.parametrize("label,is_rel,is_per,period,sigma", [
    ("absolute non-periodic", False, False, 0.0, 40.0),
    ("absolute periodic", False, True, P, 40.0),
    ("relative non-periodic", True, False, 0.0, 40.0),
    ("relative periodic", True, True, P, 20.0),
])
def test_reference_reproduces_the_shipped_cosine(label, is_rel, is_per,
                                                 period, sigma):
    r = 2
    rng = np.random.default_rng(11)
    p_x = np.sort(rng.uniform(0.0, P, 5))
    p_y = np.sort(rng.uniform(0.0, P, 5))
    w_x = rng.uniform(0.5, 1.5, 5)
    w_y = rng.uniform(0.5, 1.5, 5)

    d_x = build_exp_tens(p_x, w_x, sigma, r, is_rel, is_per, period,
                         verbose=False)
    d_y = build_exp_tens(p_y, w_y, sigma, r, is_rel, is_per, period,
                         verbose=False)
    shipped = cos_sim_exp_tens(d_x, d_y, truncation_sigmas=np.inf,
                               verbose=False)
    reference = _reference_cosine(p_x, p_y, w_x, w_y, sigma, r,
                                  is_rel, is_per, period)

    # Measured agreement is 7e-14 to 2e-13 in every mode, including the
    # relative ones, whose quadrature grid is fine enough that it does not
    # set the error. The tolerance is ~50x that, tight enough that drift in
    # either implementation shows up rather than being absorbed.
    tol = 1e-11
    assert reference == pytest.approx(shipped, rel=tol, abs=tol), (
        f"{label}: reference {reference!r} vs shipped {shipped!r}"
    )


def test_reference_cosine_is_one_against_itself():
    """A sanity floor: the reference must be self-consistent."""
    rng = np.random.default_rng(3)
    p = np.sort(rng.uniform(0.0, P, 5))
    w = rng.uniform(0.5, 1.5, 5)
    got = _reference_cosine(p, p, w, w, 40.0, 2, False, False, 0.0)
    assert got == pytest.approx(1.0, abs=1e-12)
