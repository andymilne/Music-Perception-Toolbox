"""Deep-nesting (L >= 3) build tests: matrix tags + L-general enumeration.

A nested attribute is represented as a flat ``K_total x N`` value array
plus a ``(K_total, L-1)`` integer ``tags`` matrix (one grouping column per
level, innermost-grouping first) and per-level ``r``/``sym``/``rel``
vectors. These exercise the L-general enumeration and the ``absolute`` and
``outer`` (global-transposition) projections at three levels; the per-group
``inner``/``intermediate`` reductions at L >= 3 are a later step and must
defer here.
"""

import numpy as np
import pytest

import mpt
from mpt import build_exp_tens, cos_sim_exp_tens, entropy_exp_tens
from mpt._tensor.build import _nested_enum_indices


# A three-level attribute: 2 bars x 2 chords/bar x 2 notes/chord = 8 slots.
# tags columns: 0 = chord (finest grouping above leaves), 1 = bar (outermost).
_TAGS3 = np.array([[0, 0], [0, 0], [1, 0], [1, 0],
                   [2, 1], [2, 1], [3, 1], [3, 1]])
_P3 = [np.array([[0., 4., 7., 11., 12., 16., 19., 23.]]).T]   # K_total=8, N=1
_KW = dict(sigma=[30.0], is_per=[False], period=[0.0], verbose=False)


def _spec(rel):
    return [dict(r=[2, 2, 2], sym=[True, True, False], tags=_TAGS3, rel=rel)]


# --- Enumeration ------------------------------------------------------

def test_enum_L2_parity_single_column():
    """A single-column tag matrix reproduces the two-level enumeration."""
    valid = np.array([0, 1, 2, 3], dtype=np.intp)
    tags = np.array([0, 0, 1, 1])                 # 2 events x 2 inner slots
    perm, comb = _nested_enum_indices(valid, tags, [2, 2], [True, False])
    assert comb.shape == (4, 1)
    np.testing.assert_array_equal(comb[:, 0], [0, 1, 2, 3])
    assert perm.shape == (4, 4)                   # 2! inner x 2 events ordered


def test_enum_L3_read_all_one_combination():
    valid = np.arange(8, dtype=np.intp)
    perm, comb = _nested_enum_indices(valid, _TAGS3, [2, 2, 2],
                                      [True, True, False])
    assert perm.shape[0] == 8 and comb.shape == (8, 1)
    np.testing.assert_array_equal(comb[:, 0], np.arange(8))


def test_enum_L3_middle_level_combinations():
    """Reading 2 of 3 chords in each of 2 bars gives C(3,2)^2 = 9 tuples."""
    valid = np.arange(12, dtype=np.intp)
    chord = np.repeat([0, 1, 2, 3, 4, 5], 2)
    bar = np.repeat([0, 1], 6)
    tags = np.column_stack([chord, bar])
    _, comb = _nested_enum_indices(valid, tags, [2, 2, 2],
                                   [True, True, False])
    assert comb.shape == (8, 9)


# --- Build: dimensions ------------------------------------------------

def test_L3_absolute_dim_is_full_D():
    d = build_exp_tens(_P3, None, specs=_spec(None), **_KW)
    assert d.dim == 8                               # D = 2*2*2


def test_L3_outer_dim_is_D_minus_one():
    d = build_exp_tens(_P3, None, specs=_spec([0, 0, 1]), **_KW)
    assert d.dim == 7                               # whole-tuple all-ones removed


def test_L3_outermost_string_matches_vector():
    dv = build_exp_tens(_P3, None, specs=_spec([0, 0, 1]), **_KW)
    ds = build_exp_tens(_P3, None, specs=_spec('outermost'), **_KW)
    assert ds.dim == dv.dim == 7


# --- Build: similarity / entropy --------------------------------------

def test_L3_absolute_cosine_self_match():
    d = build_exp_tens(_P3, None, specs=_spec(None), **_KW)
    assert float(cos_sim_exp_tens(d, d, verbose=False)) == pytest.approx(1.0, abs=1e-9)


def test_L3_outer_cosine_self_match():
    d = build_exp_tens(_P3, None, specs=_spec([0, 0, 1]), **_KW)
    assert float(cos_sim_exp_tens(d, d, verbose=False)) == pytest.approx(1.0, abs=1e-9)


def test_L3_outer_is_global_transposition_invariant():
    """The outermost unit quotients out a shift of the whole tuple."""
    d = build_exp_tens(_P3, None, specs=_spec([0, 0, 1]), **_KW)
    dT = build_exp_tens([_P3[0] + 5.0], None, specs=_spec([0, 0, 1]), **_KW)
    assert float(cos_sim_exp_tens(d, dT, verbose=False)) == pytest.approx(1.0, abs=1e-9)


def test_L3_absolute_not_transposition_invariant():
    d = build_exp_tens(_P3, None, specs=_spec(None), **_KW)
    dT = build_exp_tens([_P3[0] + 5.0], None, specs=_spec(None), **_KW)
    assert float(cos_sim_exp_tens(d, dT, verbose=False)) < 0.999


@pytest.mark.xfail(reason="renyi2/Mobius entropy re-derives the full S_dim "
                          "orbit instead of the tag-restricted nested orbit "
                          "at L>=3; entropy-orbit generalisation is a later "
                          "step. Cosine (pairwise IP) already works.",
                   raises=MemoryError, strict=False)
def test_L3_entropy_runs():
    d = build_exp_tens(_P3, None, specs=_spec(None), **_KW)
    h = float(entropy_exp_tens(d, method='renyi2', verbose=False))
    assert np.isfinite(h)


# --- Deferred projections (per-group reduction at L >= 3) -------------

@pytest.mark.parametrize("rel", [[1, 0, 0], [0, 1, 0]])
def test_L3_inner_and_intermediate_defer(rel):
    with pytest.raises(NotImplementedError):
        build_exp_tens(_P3, None, specs=_spec(rel), **_KW)


# --- Representation validation ----------------------------------------

def test_L3_one_d_tags_rejected_for_deep_nesting():
    bad = [dict(r=[2, 2, 2], sym=[True, True, False],
                tags=np.zeros(8, dtype=int), rel=None)]
    with pytest.raises(ValueError):
        build_exp_tens(_P3, None, specs=bad, **_KW)


def test_L3_tag_matrix_wrong_shape_rejected():
    bad = [dict(r=[2, 2, 2], sym=[True, True, False],
                tags=_TAGS3[:, :1], rel=None)]      # (8,1) but L-1 = 2
    with pytest.raises(ValueError):
        build_exp_tens(_P3, None, specs=bad, **_KW)


def test_L3_scalar_rel_rejected_when_nested():
    with pytest.raises(ValueError):
        build_exp_tens(_P3, None, specs=_spec(True), **_KW)


def test_L3_infeasible_read_errors():
    """r asks for more groups than the slots provide."""
    bad = [dict(r=[2, 2, 3], sym=[True, True, False], tags=_TAGS3, rel=None)]
    with pytest.raises(ValueError):
        build_exp_tens(_P3, None, specs=bad, **_KW)
