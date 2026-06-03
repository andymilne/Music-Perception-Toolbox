"""Tests for the canonical ``specs=`` form of build_exp_tens (3c-i).

``specs`` is the single home for level-structured geometry (toolbox spec
§6.4): a per-attribute list of dicts. A flat attribute is a one-level spec
``{r, rel?, sym?, name?}`` (scalar r, bool rel/sym); a nested attribute
carries ``tags`` plus per-level vectors ``{tags, r, sym, rel, name?,
names?}``. Scalar geometry not structured by nesting (sigma, is_per,
period) stays outside the spec, supplied as keywords. The old positional
form is retained unchanged as a shim.
"""

import numpy as np
import pytest

from mpt import build_exp_tens, eval_exp_tens, cos_sim_exp_tens


def _ev(d, x):
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    return float(np.ravel(eval_exp_tens(d, x, verbose=False))[0])


P2 = [np.array([[0.0, 4.0], [7.0, 11.0]]), np.array([[100.0, 140.0]])]


def test_flat_specs_match_old_positional():
    """A flat-spec build equals the equivalent old positional build."""
    d_old = build_exp_tens(P2, None, [50.0, 30.0], [2, 1], [True, False],
                           [False, False], [0.0, 0.0], verbose=False)
    specs = [dict(r=2, rel=True, sym=True), dict(r=1, rel=False)]
    d_spec = build_exp_tens(P2, None, specs=specs, sigma=[50.0, 30.0],
                            is_per=[False, False], period=[0.0, 0.0],
                            verbose=False)
    assert d_old.dim == d_spec.dim
    for x in ([5.0, 100.0], [7.0, 140.0], [3.0, 120.0]):
        assert _ev(d_old, x) == pytest.approx(_ev(d_spec, x), abs=1e-12)


def test_flat_spec_defaults():
    """Flat-spec defaults: rel=False, sym=True."""
    d_min = build_exp_tens(P2, None, specs=[dict(r=2), dict(r=1)],
                           sigma=[50.0, 30.0], is_per=[False, False],
                           period=[0.0, 0.0], verbose=False)
    d_exp = build_exp_tens(P2, None,
                           specs=[dict(r=2, rel=False, sym=True),
                                  dict(r=1, rel=False, sym=True)],
                           sigma=[50.0, 30.0], is_per=[False, False],
                           period=[0.0, 0.0], verbose=False)
    assert d_min.dim == d_exp.dim
    assert bool(d_min.is_rel[0]) is False and bool(d_min.is_sym[0]) is True


def test_nested_spec_matches_nested_kwarg():
    """A nested-spec build equals the equivalent nested= build."""
    pv = [np.array([[0.0], [4.0], [7.0], [11.0]])]
    nsp = dict(tags=[0, 0, 1, 1], r=[2, 2], sym=[True, False], rel="outermost")
    d_kw = build_exp_tens(pv, None, [50.0], [1], [False], [False], [0.0],
                          nested=[dict(nsp)], verbose=False)
    d_spec = build_exp_tens(pv, None, specs=[dict(nsp)], sigma=[50.0],
                            is_per=[False], period=[0.0], verbose=False)
    assert d_kw.dim == d_spec.dim == 3
    assert float(cos_sim_exp_tens(d_kw, d_spec, verbose=False)) == pytest.approx(1.0, abs=1e-9)


def test_attribute_names_stored_and_prune_invariant():
    """Attribute names are stored and survive pruning."""
    specs = [dict(r=2, rel=True, name="pitch"), dict(r=1, name="time")]
    d = build_exp_tens(P2, None, specs=specs, sigma=[50.0, 30.0],
                       is_per=[False, False], period=[0.0, 0.0], verbose=False)
    assert d.names == ["pitch", "time"]
    assert d.pruned().names == ["pitch", "time"]


def test_level_names_carried_in_nested_spec():
    """Per-level names ride inside the nested spec entry."""
    nsp = dict(tags=[0, 0, 1, 1], r=[2, 2], sym=[True, False], rel="innermost",
               name="chordprog", names=["note", "chord"])
    d = build_exp_tens([np.array([[0.0], [4.0], [7.0], [11.0]])], None,
                       specs=[nsp], sigma=[50.0], is_per=[False], period=[0.0],
                       verbose=False)
    assert d.names == ["chordprog"]
    assert d.nested[0].get("names") == ["note", "chord"]


def test_unnamed_attributes_default_to_none():
    d = build_exp_tens(P2, None, specs=[dict(r=2), dict(r=1)],
                       sigma=[50.0, 30.0], is_per=[False, False],
                       period=[0.0, 0.0], verbose=False)
    assert d.names == [None, None]


def test_specs_guards():
    specs = [dict(r=2, rel=True), dict(r=1)]
    # specs with positional geometry
    with pytest.raises(ValueError):
        build_exp_tens(P2, None, [50.0, 30.0], specs=specs, sigma=[1.0, 1.0],
                       is_per=[False, False], period=[0.0, 0.0], verbose=False)
    # specs missing scalar geometry
    with pytest.raises(ValueError):
        build_exp_tens(P2, None, specs=specs, is_per=[False, False],
                       period=[0.0, 0.0], verbose=False)
    # scalar-geometry keywords without specs
    with pytest.raises(ValueError):
        build_exp_tens(P2, None, [50.0, 30.0], [2, 1], [True, False],
                       [False, False], [0.0, 0.0], sigma=[1.0, 1.0],
                       verbose=False)
    # wrong specs length
    with pytest.raises(ValueError):
        build_exp_tens(P2, None, specs=[dict(r=2)], sigma=[1.0, 1.0],
                       is_per=[False, False], period=[0.0, 0.0], verbose=False)
    # flat spec missing r
    with pytest.raises(ValueError):
        build_exp_tens(P2, None, specs=[dict(rel=True), dict(r=1)],
                       sigma=[1.0, 1.0], is_per=[False, False],
                       period=[0.0, 0.0], verbose=False)


def test_specs_rejected_for_single_attribute():
    """specs= is multi-attribute only."""
    with pytest.raises(ValueError):
        build_exp_tens([0.0, 4.0, 7.0], None, specs=[dict(r=2)],
                       sigma=[50.0], is_per=[False], period=[0.0],
                       verbose=False)
