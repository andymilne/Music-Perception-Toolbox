"""Translation sweeps with an anisotropic kernel covariance.

A kernel covariance is carried on whitened values at ``sigma = 1``, so
an attribute that is *not* swept enters the mixture's fixed weight
exactly, while a swept one is refused (a uniform translation of the
original values is not uniform in the whitened coordinates). Every test
compares the mixture route against the per-offset path it replaces:
translate the query explicitly, call :func:`~mpt.sim_maet`, and require
agreement. Mirrors ``matlab/tests/test_sweep_kernel_cov.m``.
"""
import numpy as np
import pytest

from mpt import build_maet, sim_maet, sweep_sim_maet
from mpt._tensor.sweep import sweep_eligibility

PARITY = 1e-12
BASE = np.array([-3.1, -1.4, -0.35, 0.0, 0.8, 2.2, 5.0])

SIG2 = np.array([[1.0, 0.4], [0.4, 0.7]])
SIG3 = np.array([[1.0, 0.3, 0.1], [0.3, 0.9, 0.2], [0.1, 0.2, 1.2]])


def _case(K0, r0, exch0, Sig, seed):
    """Attribute 0 isotropic (swept); attribute 1 with covariance ``Sig``."""
    rng = np.random.default_rng(seed)
    d1 = Sig.shape[0]
    p_x = [rng.normal(0.0, 3.0, (K0, 6)), rng.normal(0.0, 1.5, (d1, 6))]
    p_y = [rng.normal(0.0, 3.0, (K0, 4)), rng.normal(0.0, 1.5, (d1, 4))]
    geom = ([r0, d1], [False, False], [False, False], [0.0, 0.0],
            [exch0, False])
    return p_x, p_y, geom


def _reference(p_x, p_y, sigma, geom, off, *, normalize, truncation_sigmas):
    dx = build_maet(p_x, None, sigma, *geom, verbose=False)
    out = []
    for m in range(off.shape[1]):
        p_ym = [p_y[a] + off[a, m] for a in range(len(p_y))]
        dy = build_maet(p_ym, None, sigma, *geom, verbose=False)
        out.append(sim_maet(dx, dy, method="bulger", normalize=normalize,
                            truncation_sigmas=truncation_sigmas,
                            verbose=False))
    return np.array(out)


def _rel_dev(got, ref):
    return float(np.max(np.abs(got - ref)) / np.max(np.abs(ref)))


def _offsets():
    off = np.zeros((2, BASE.size))
    off[0] = BASE
    return off


CASES = [(3, 2, True, SIG2), (2, 2, False, SIG2), (2, 2, False, SIG3)]


@pytest.mark.parametrize("K0,r0,exch0,Sig", CASES)
@pytest.mark.parametrize("normalize", ["cosine", "oneSidedDenom"])
@pytest.mark.parametrize("ts", [np.inf, None])
def test_unswept_kernel_cov_matches_per_offset(K0, r0, exch0, Sig,
                                               normalize, ts):
    p_x, p_y, geom = _case(K0, r0, exch0, Sig, seed=K0 + 10 * Sig.shape[0])
    sigma = [0.9, Sig]
    dx = build_maet(p_x, None, sigma, *geom, verbose=False)
    dy = build_maet(p_y, None, sigma, *geom, verbose=False)
    off = _offsets()
    ref = _reference(p_x, p_y, sigma, geom, off, normalize=normalize,
                     truncation_sigmas=ts)
    for method in ("mixture", "auto"):
        got = sweep_sim_maet(dx, dy, off, method=method, normalize=normalize,
                             truncation_sigmas=ts, verbose=False)
        assert _rel_dev(got, ref) <= PARITY


def test_scalar_equivalent_covariance_matches_scalar_sigma():
    p_x, p_y, geom = _case(3, 3, True, SIG2, seed=4)
    s = 0.9
    off = _offsets()
    vals = []
    for sig1 in (s * s * np.eye(2), s):
        dx = build_maet(p_x, None, [0.9, sig1], *geom, verbose=False)
        dy = build_maet(p_y, None, [0.9, sig1], *geom, verbose=False)
        vals.append(sweep_sim_maet(dx, dy, off, method="mixture",
                                   truncation_sigmas=np.inf, verbose=False))
    assert _rel_dev(vals[0], vals[1]) <= PARITY


def test_swept_kernel_cov_is_refused_by_the_mixture():
    p_x, p_y, geom = _case(3, 2, True, SIG2, seed=5)
    dx = build_maet(p_x, None, [0.9, SIG2], *geom, verbose=False)
    dy = build_maet(p_y, None, [0.9, SIG2], *geom, verbose=False)
    off = np.zeros((2, BASE.size))
    off[1] = BASE
    ok, why = sweep_eligibility(dx, dy, off)
    assert not ok and "swept attribute" in why
    with pytest.raises(ValueError, match="kernel"):
        sweep_sim_maet(dx, dy, off, method="mixture", verbose=False)


def test_mismatched_kernel_covs_are_refused():
    p_x, p_y, geom = _case(3, 2, True, SIG2, seed=6)
    dx = build_maet(p_x, None, [0.9, SIG2], *geom, verbose=False)
    dy = build_maet(p_y, None, [0.9, 2.0 * SIG2], *geom, verbose=False)
    for method in ("auto", "mixture"):
        with pytest.raises(ValueError, match="different kernel"):
            sweep_sim_maet(dx, dy, _offsets(), method=method, verbose=False)
