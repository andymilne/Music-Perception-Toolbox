"""Regressions from the routing-map parity audit (September 2026).

Two Python routes dropped an attribute's declared ``wrap`` and so computed
the full-image measure for an absolute-periodic attribute declared
``wrap='single-image'``: the flat attribute of a nested-MA density
(``_try_nested_contract_ma`` pass 2) and the flat symmetric attribute of
the Rényi-2 entropy loop. The MATLAB twins had always passed it. These pin
the fix by checking that the two wraps give different numbers where the
truncation budget admits images beyond the nearest one (sigma/P = 0.2),
and that the contraction agrees with the enumeration, which honours
``wrap``, under both.
"""
import numpy as np
import pytest

import mpt
from mpt import build_exp_tens, cos_sim_exp_tens, entropy_exp_tens


@pytest.fixture(autouse=True)
def _at_the_accuracy_floor():
    """Routes are compared at the accuracy floor, as the other nested
    tests do; at the 6-sigma default the two truncate differently."""
    prev = mpt.get_default('truncation_sigmas')
    mpt.set_default(truncation_sigmas=float('inf'))
    try:
        yield
    finally:
        mpt.set_default(truncation_sigmas=prev)

P = 12.0
SIG = 0.2 * P          # sigma/P = 0.2: L >= 1 at the 6-sigma default


def _nested_plus_flat(seed, wrap):
    """A nested harmonic attribute tensored with a flat abs-per r = 2
    attribute carrying the given wrap."""
    rng = np.random.default_rng(seed)
    tags = np.repeat(np.arange(2), 3)
    p0 = np.sort(rng.uniform(0.0, P, size=(6, 2)), axis=0)
    p1 = np.sort(rng.uniform(0.0, P, size=(4, 2)), axis=0)
    specs = [{"tags": tags, "r": [1, 2], "sym": [True, True], "rel": [0, 0]},
             {"r": 2, "sym": True, "rel": False}]
    return build_exp_tens([p0, p1], None, specs=specs, sigma=[0.05 * P, SIG],
                          is_per=[True, True], period=[P, P],
                          wrap=['full-image', wrap], verbose=False)


@pytest.mark.parametrize("wrap", ["full-image", "single-image"])
def test_nested_ma_flat_attribute_honours_wrap(wrap):
    x, y = _nested_plus_flat(1, wrap), _nested_plus_flat(2, wrap)
    c = cos_sim_exp_tens(x, y, method="contract", verbose=False)
    b = cos_sim_exp_tens(x, y, method="bulger", verbose=False)
    assert c == pytest.approx(b, rel=1e-9, abs=1e-12)


def test_nested_ma_flat_attribute_wraps_differ():
    cf = cos_sim_exp_tens(_nested_plus_flat(1, "full-image"),
                          _nested_plus_flat(2, "full-image"),
                          method="contract", verbose=False)
    cs = cos_sim_exp_tens(_nested_plus_flat(1, "single-image"),
                          _nested_plus_flat(2, "single-image"),
                          method="contract", verbose=False)
    assert abs(cf - cs) > 1e-4


def _flat(wrap):
    rng = np.random.default_rng(3)
    p = np.sort(rng.uniform(0.0, P, size=(5, 2)), axis=0)
    return build_exp_tens([p], None, [SIG], [2], [False], [True], [P],
                          wrap=[wrap], verbose=False)


def test_renyi2_flat_attribute_honours_wrap():
    hf = entropy_exp_tens(_flat("full-image"), method="renyi2",
                          verbose=False)
    hs = entropy_exp_tens(_flat("single-image"), method="renyi2",
                          verbose=False)
    assert np.isfinite(hf) and np.isfinite(hs)
    assert abs(hf - hs) > 1e-4


def test_eval_mobius_is_refused_on_a_nested_density():
    """A forced ``method='mobius'`` on a nested density used to evaluate
    the flattened multiset silently (8.67 against 3.78 for the centres
    route on this density); it is refused, as on an ordered attribute."""
    from mpt import eval_exp_tens
    rng = np.random.default_rng(1)
    tags = np.repeat(np.arange(2), 3)
    p = np.sort(rng.uniform(0.0, P, size=(6, 2)), axis=0)
    spec = {"tags": tags, "r": [1, 2], "sym": [True, True], "rel": [0, 1]}
    d = build_exp_tens([p], None, specs=[spec], sigma=[0.5], is_per=[True],
                       period=[P], verbose=False)
    X = np.zeros((d.dim, 3))
    with pytest.raises(ValueError, match="nested"):
        eval_exp_tens(d, X, method="mobius", verbose=False)
    v_auto = eval_exp_tens(d, X, method="auto", verbose=False)
    v_cent = eval_exp_tens(d, X, method="centres", verbose=False)
    np.testing.assert_allclose(v_auto, v_cent, rtol=1e-12)
