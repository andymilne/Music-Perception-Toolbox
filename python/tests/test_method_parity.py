"""Cross-language parity of the inner-product and evaluation method names.

There are three algorithms in this area: enumerate the tuple centres
(``'centres'``), enumerate them with one side restricted to combinations
and multiply by r! (``'bulger'``), and sum over the partition lattice
(``'mobius'``). Evaluation has two of them, there being no two-sided
pairing for Bulger's identity to exploit.

These tests pin the vocabulary, because it drifted before: ``'direct'``
named Bulger's method on the inner product while promising an
unrestricted enumeration, and named the centres route on evaluation, so
one word meant two things and neither matched its docstring. The MATLAB
twin of this file is ``matlab/tests/test_method_parity.m``; the two must
accept and reject the same strings.
"""
import pathlib
import sys

import numpy as np
import pytest

from mpt import build_exp_tens, cos_sim_exp_tens, eval_exp_tens

# The independent reference lives beside this file rather than in the
# package: it is not a route, and must share no code with the core.
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _independent_ip import centres_inner_product  # noqa: E402


def _pair(K=7, r=3, sigma=30.0, is_rel=False, is_per=False, period=1200.0):
    rng = np.random.default_rng(20260823)
    p = np.sort(rng.uniform(0.0, period, K))
    q = np.sort(rng.uniform(0.0, period, K))
    w = rng.uniform(0.4, 1.0, K)
    wq = rng.uniform(0.4, 1.0, K)
    a = build_exp_tens(p, w, sigma, r, is_rel, is_per, period, verbose=False)
    b = build_exp_tens(q, wq, sigma, r, is_rel, is_per, period, verbose=False)
    return a, b


IP_ACCEPTED = ("auto", "bulger", "centres", "mobius")
EVAL_ACCEPTED = ("auto", "centres", "mobius")
RETIRED = "direct"


@pytest.mark.parametrize("method", IP_ACCEPTED)
def test_inner_product_accepts(method):
    a, b = _pair()
    assert np.isfinite(cos_sim_exp_tens(a, b, method=method, verbose=False))


@pytest.mark.parametrize("method", EVAL_ACCEPTED)
def test_eval_accepts(method):
    a, _ = _pair()
    pts = np.array([[0.0], [100.0], [250.0]])
    assert np.all(np.isfinite(eval_exp_tens(a, pts, method=method,
                                            verbose=False)))


def test_direct_retired_on_both_entry_points():
    a, b = _pair()
    with pytest.raises(ValueError):
        cos_sim_exp_tens(a, b, method=RETIRED, verbose=False)
    with pytest.raises(ValueError):
        eval_exp_tens(a, np.array([[0.0]]), method=RETIRED, verbose=False)


@pytest.mark.parametrize("is_rel,is_per", [(False, False), (False, True),
                                           (True, False), (True, True)])
def test_three_routes_agree(is_rel, is_per):
    """All three inner-product routes compute the same value.

    The centres route shares no reduction with the other two, so this is
    a stronger check than Bulger against Möbius alone.

    Relative-periodic is the documented exception: there Bulger and
    centres compute the wrapped-difference kernel while Möbius computes
    the transposition average, and the two kernels agree only as
    sigma/P tends to zero. The sigma used here keeps sigma/P at 0.025,
    well inside the regime where they coincide.
    """
    a, b = _pair(is_rel=is_rel, is_per=is_per)
    vals = {m: float(cos_sim_exp_tens(a, b, method=m,
                                      truncation_sigmas=float("inf"),
                                      verbose=False))
            for m in ("bulger", "centres", "mobius")}
    ref = vals["bulger"]
    for m, v in vals.items():
        abs_err = abs(v - ref)
        rel_err = abs_err / max(abs(v), abs(ref), 1e-300)
        assert abs_err < 1e-12 or rel_err < 1e-9, f"{m}: {vals}"


@pytest.mark.parametrize("is_rel,is_per", [(False, False), (False, True),
                                           (True, False), (True, True)])
def test_routes_agree_with_an_independent_implementation(is_rel, is_per):
    """The shipped routes agree with code that shares nothing with them.

    All three shipped routes go through ``_ip_core_ma`` -- that is what
    makes their timings comparable -- so agreement among them cannot
    catch a bug in the core. This check uses a separate enumeration
    written from the definition, so a shared-core error would show up.
    """
    rng = np.random.default_rng(20260823)
    K, r, sigma, period = 7, 3, 30.0, 1200.0
    p = np.sort(rng.uniform(0.0, period, K))
    q = np.sort(rng.uniform(0.0, period, K))
    w = rng.uniform(0.4, 1.0, K)
    wq = rng.uniform(0.4, 1.0, K)

    def ind(pa, wa, pb, wb):
        return centres_inner_product(pa, wa, pb, wb, sigma, r,
                                     is_rel, is_per, period)

    expected = ind(p, w, q, wq) / np.sqrt(ind(p, w, p, w) * ind(q, wq, q, wq))

    a = build_exp_tens(p, w, sigma, r, is_rel, is_per, period, verbose=False)
    b = build_exp_tens(q, wq, sigma, r, is_rel, is_per, period, verbose=False)
    for method in ("bulger", "centres", "mobius"):
        got = float(cos_sim_exp_tens(a, b, method=method,
                                     truncation_sigmas=float("inf"),
                                     verbose=False))
        # Relative-periodic mobius computes the transposition average,
        # a different measure from the wrapped-difference kernel the
        # other routes and the independent implementation use; at this
        # sigma/P (0.025) they coincide well inside the tolerance.
        assert abs(got - expected) <= 1e-9 * max(abs(expected), 1.0), (
            f"{method}: {got} vs independent {expected}")


# --- an explicit method must be the method that runs -------------------

REL_ROUTES = ("auto", "centres", "mobius")


@pytest.mark.parametrize("route", REL_ROUTES)
@pytest.mark.parametrize("is_per", [False, True])
def test_forced_mobius_runs_mobius_on_relative_attributes(route, is_per):
    """method='mobius' must not be served by centres enumeration.

    Inside the Moebius route a relative attribute's inner matrices may be
    computed either by the Moebius decomposition under a shift quadrature
    or by unrestricted enumeration over materialised tuple centres. The
    second is a different algorithm, and the cost gate used to substitute
    it under 'auto' even when the caller had asked for Moebius by name.
    An explicit method now pins the sub-route, mirroring the top-level
    rule where a forced method returns ahead of the cost model.
    """
    import mpt
    from mpt._tensor._mobius_inner import _ma_rel_attr_prefers_centres

    rng = np.random.default_rng(7)
    K, r, sigma, period = 8, 2, 10.0, 1200.0
    p = np.sort(rng.uniform(0.0, period, K))
    q = np.sort(rng.uniform(0.0, period, K))

    prev = mpt.get_default("rel_attr_route")
    try:
        mpt.set_default(rel_attr_route=route)
        forced = _ma_rel_attr_prefers_centres(
            p.reshape(-1, 1), q.reshape(-1, 1), sigma, r, True, is_per,
            period, truncation_sigmas=float("inf"),
            user_forced_mobius=True)
        if route == "centres":
            assert forced is True, "explicit 'centres' must still win"
        else:
            assert forced is False, (
                f"rel_attr_route={route!r} with method='mobius' selected "
                "centres enumeration")
    finally:
        mpt.set_default(rel_attr_route=prev)


def test_rel_attr_route_accepts_mobius_and_aliases_grid():
    """One word per algorithm at both levels; 'grid' kept as an alias."""
    import mpt

    prev = mpt.get_default("rel_attr_route")
    try:
        mpt.set_default(rel_attr_route="mobius")
        assert mpt.get_default("rel_attr_route") == "mobius"
        mpt.set_default(rel_attr_route="grid")
        assert mpt.get_default("rel_attr_route") == "mobius", (
            "'grid' should normalise to 'mobius'")
        with pytest.raises(ValueError):
            mpt.set_default(rel_attr_route="bulger")
    finally:
        mpt.set_default(rel_attr_route=prev)


@pytest.mark.parametrize("route", REL_ROUTES)
def test_value_is_unchanged_by_the_sub_route(route):
    """The sub-route is a cost choice: the value must not depend on it."""
    import mpt

    rng = np.random.default_rng(8)
    K, r, sigma, period = 9, 3, 10.0, 1200.0
    p = np.sort(rng.uniform(0.0, period, K))
    q = np.sort(rng.uniform(0.0, period, K))
    w = rng.uniform(0.4, 1.0, K)
    wq = rng.uniform(0.4, 1.0, K)

    prev = mpt.get_default("rel_attr_route")
    got = {}
    try:
        for is_per in (False, True):
            a = build_exp_tens(p, w, sigma, r, True, is_per, period,
                               verbose=False)
            b = build_exp_tens(q, wq, sigma, r, True, is_per, period,
                               verbose=False)
            mpt.set_default(rel_attr_route=route)
            got[is_per] = float(cos_sim_exp_tens(
                a, b, method="mobius", truncation_sigmas=float("inf"),
                verbose=False))
            mpt.set_default(rel_attr_route="auto")
            ref = float(cos_sim_exp_tens(
                a, b, method="bulger", truncation_sigmas=float("inf"),
                verbose=False))
            assert abs(got[is_per] - ref) <= 1e-9 * max(abs(ref), 1.0), (
                f"route={route} is_per={is_per}: {got[is_per]} vs {ref}")
    finally:
        mpt.set_default(rel_attr_route=prev)


def test_spectral_force_changes_cost_not_value():
    """The spectral-branch lever must not move the answer.

    Both the spectral branch and the translation grid compute the
    full-image measure, so bypassing the branch's cost gate is a cost
    decision only. The gate is known to misroute outside the shapes it
    was calibrated on -- it models the grid route as costing K^2 per
    event pair, omitting the node count, which scales with span/sigma --
    so benchmarks pin the branch rather than measure the routing.
    """
    import mpt._tensor._mobius_inner as mi

    rng = np.random.default_rng(3)
    K, r, sigma, period = 40, 3, 10.0, 1200.0
    p = np.sort(rng.uniform(0.0, 3 * period, K))
    q = np.sort(rng.uniform(0.0, 3 * period, K))
    w = rng.uniform(0.4, 1.0, K)
    wq = rng.uniform(0.4, 1.0, K)

    prev = mi._SPECTRAL_IP_FORCE
    got = {}
    try:
        for forced in (False, True):
            mi._SPECTRAL_IP_FORCE = forced
            a = build_exp_tens(p, w, sigma, r, True, False, period,
                               verbose=False)
            b = build_exp_tens(q, wq, sigma, r, True, False, period,
                               verbose=False)
            got[forced] = float(cos_sim_exp_tens(
                a, b, method="mobius", truncation_sigmas=float("inf"),
                verbose=False))
    finally:
        mi._SPECTRAL_IP_FORCE = prev

    assert got[False] == pytest.approx(got[True], rel=1e-12, abs=1e-15), got
