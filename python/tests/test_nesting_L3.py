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


# --- inner / intermediate per-group reduction at L = 3 ---------------
#
# Co-transposition at unit u removes each level-u sub-tuple's all-ones:
# inner (u=0) is per-chord, intermediate (u=1) per-bar, outer (u=2)
# global. dim = D - G_u with G_u = prod(r[u+1:]). Each unit is invariant
# to transposition at its own level and coarser, and not finer.

# Per-slot offsets: per-chord (tags col 0), per-bar (tags col 1), global.
_PER_CHORD = [_P3[0] + np.array([[0., 0, 60, 60, 0, 0, 60, 60]]).T]
_PER_BAR = [_P3[0] + np.array([[10., 10, 10, 10, 20, 20, 20, 20]]).T]
_GLOBAL = [_P3[0] + 5.0]


def _build(rel, p=_P3):
    return build_exp_tens(p, None, specs=_spec(rel), **_KW)


def test_L3_inner_dim_and_self_match():
    d = _build([1, 0, 0])
    assert d.dim == 4                                # D - G_0 = 8 - 4
    assert float(cos_sim_exp_tens(d, d, verbose=False)) == pytest.approx(1.0, abs=1e-9)


def test_L3_inner_is_per_chord_transposition_invariant():
    d = _build([1, 0, 0])
    dC = _build([1, 0, 0], _PER_CHORD)
    assert float(cos_sim_exp_tens(d, dC, verbose=False)) == pytest.approx(1.0, abs=1e-6)


def test_L3_intermediate_dim_and_self_match():
    d = _build([0, 1, 0])
    assert d.dim == 6                                # D - G_1 = 8 - 2
    assert float(cos_sim_exp_tens(d, d, verbose=False)) == pytest.approx(1.0, abs=1e-9)


def test_L3_intermediate_is_per_bar_invariant_not_per_chord():
    d = _build([0, 1, 0])
    dB = _build([0, 1, 0], _PER_BAR)
    dC = _build([0, 1, 0], _PER_CHORD)
    assert float(cos_sim_exp_tens(d, dB, verbose=False)) == pytest.approx(1.0, abs=1e-6)
    assert float(cos_sim_exp_tens(d, dC, verbose=False)) < 0.5    # finer: not invariant


def test_L3_outer_invariant_global_not_per_bar():
    d = _build([0, 0, 1])
    dG = _build([0, 0, 1], _GLOBAL)
    dB = _build([0, 0, 1], _PER_BAR)
    assert float(cos_sim_exp_tens(d, dG, verbose=False)) == pytest.approx(1.0, abs=1e-6)
    assert float(cos_sim_exp_tens(d, dB, verbose=False)) < 0.999  # coarser unit only


def test_L3_unit_dims_strictly_increase_inner_to_outer():
    """Finer co-transposition quotients more: dim inner < intermediate < outer < absolute."""
    dims = [_build(r).dim for r in ([1, 0, 0], [0, 1, 0], [0, 0, 1], None)]
    assert dims == [4, 6, 7, 8]


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


# --- bind_events deepening: flat -> L=2 -> L=3 -----------------------
#
# bind_events deepens an already-nested attribute by tiling the existing
# tag columns and appending a new outermost grouping level; r/sym/rel
# extend by the bound outer level.

def test_bind_deepens_nested_to_L3():
    p = [np.array([[0., 2, 4, 5, 7, 9]])]            # flat, K=1, N=6
    p1, w1, s1 = mpt.bind_events(p, None, 2)         # -> L=2
    p2, w2, s2 = mpt.bind_events(p1, w1, 2, specs=s1)  # -> L=3
    sp = s2[0]
    assert np.asarray(sp["tags"]).shape == (4, 2)    # K_total=4, L-1=2 columns
    assert list(sp["r"]) == [1, 2, 2]
    assert len(sp["sym"]) == 3 and len(sp["rel"]) == 3
    # New outermost column distinguishes the two bound super-events;
    # inner column is the tiled level-1 grouping.
    np.testing.assert_array_equal(np.asarray(sp["tags"])[:, 1], [0, 0, 1, 1])
    np.testing.assert_array_equal(np.asarray(sp["tags"])[:, 0], [0, 1, 0, 1])


def test_bind_deepened_L3_builds_and_self_matches():
    p = [np.array([[0., 2, 4, 5, 7, 9]])]
    p1, w1, s1 = mpt.bind_events(p, None, 2)
    p2, w2, s2 = mpt.bind_events(p1, w1, 2, specs=s1)
    d = build_exp_tens(p2, w2, specs=s2, **_KW)
    assert d.dim == 4                                 # prod(r) = 1*2*2
    assert float(cos_sim_exp_tens(d, d, verbose=False)) == pytest.approx(1.0, abs=1e-9)


def test_bind_deepen_rel_outer_is_global_transposition_invariant():
    p = [np.array([[0., 2, 4, 5, 7, 9]])]
    p1, w1, s1 = mpt.bind_events(p, None, 2)
    p2, w2, s2 = mpt.bind_events(p1, w1, 2, specs=s1, rel_outer=True)
    assert list(s2[0]["rel"]) == [0, 0, 1]            # new outermost relative
    d = build_exp_tens(p2, w2, specs=s2, **_KW)
    dT = build_exp_tens([p2[0] + 5.0], w2, specs=s2, **_KW)
    assert d.dim == 3                                 # D - 1
    assert float(cos_sim_exp_tens(d, dT, verbose=False)) == pytest.approx(1.0, abs=1e-6)


def test_bind_deepen_rejects_level_names():
    p = [np.array([[0., 2, 4, 5, 7, 9]])]
    p1, w1, s1 = mpt.bind_events(p, None, 2)
    with pytest.raises(ValueError):
        mpt.bind_events(p1, w1, 2, specs=s1, level_names=["x", "y", "z"])
