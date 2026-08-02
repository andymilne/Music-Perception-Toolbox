"""Tests for the triple-form bind_events (3c-ii / 3c-iv-c).

bind_events nests sliding windows of consecutive events into a single nested
attribute per input attribute (toolbox spec §6.1/§6.5): the bound events form
an ordered outer level (sym_outer = 0 by default), each event's own multiset
is the inner level. The inner level's geometry (r/rel/sym) is read from the
incoming triple's specs (flat_specs defaults when specs=None). It returns
(p_attr_bound, w_bound, specs) ready for build_exp_tens(..., specs=...). L = 1
is a flat passthrough. Outer r = L with rel = [rel_in, 0] reproduces the old
separate-attribute tensor join (§6.5).
"""

import numpy as np
import pytest

import mpt
from mpt import build_exp_tens, eval_exp_tens, flat_specs
from mpt._tensor.preprocessing import bind_events


def _ev(d, x):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return np.ravel(eval_exp_tens(d, x, verbose=False))


def test_returns_three_tuple_with_specs():
    p = [np.array([[0.0, 4.0, 7.0, 11.0, 2.0]])]
    pb, wb, specs = bind_events(p, None, 2)
    assert isinstance(specs, list) and len(specs) == 1
    assert "tags" in specs[0]  # nested


def test_order_1_is_flat_passthrough():
    """L = 1 passes the incoming flat spec through (no tags)."""
    p = [np.array([[0.0, 4.0, 7.0, 11.0, 2.0]])]
    pb, _, specs = bind_events(p, None, 1,
                               specs=flat_specs(p, r=3, rel=True, sym=True))
    assert "tags" not in specs[0]
    assert specs[0] == {"r": 3, "rel": True, "sym": True}
    np.testing.assert_allclose(pb[0], p[0])  # circular=False, max_order=1 -> N'=N


def test_inner_geometry_read_from_specs():
    """Inner level inherits r/rel/sym from the incoming spec; outer r=L,
    sym=0, rel=0."""
    p = [np.array([[0.0, 4.0, 7.0], [10.0, 12.0, 14.0]])]    # K=2, N=3
    pb, _, specs = bind_events(p, None, 2,
                               specs=flat_specs(p, r=2, rel=True, sym=True))
    s = specs[0]
    assert s["r"] == [2, 2]            # inner = incoming r=2, outer = L=2
    assert s["sym"] == [True, False]   # inner inherits, outer ordered
    assert s["rel"] == [1, 0]          # inner inherits rel=True, outer off
    assert list(s["tags"]) == [0, 0, 1, 1]
    assert pb[0].shape == (4, 2)       # (L*K, N') = (2*2, 3-2+1)


def test_synthesised_specs_default_inner_geometry():
    """specs=None synthesises flat defaults: inner r=1, rel=0, sym=1."""
    p = [np.array([[0.0, 4.0, 7.0, 11.0]])]
    _, _, specs = bind_events(p, None, 2)
    s = specs[0]
    assert s["r"] == [1, 2] and s["sym"] == [True, False] and s["rel"] == [0, 0]


def test_rel_default_inherits_inner_off_outer():
    """rel defaults to [rel_in, 0]; absolute attr -> [0, 0]."""
    p = [np.array([[0.0, 4.0, 7.0, 11.0]])]
    _, _, s_abs = bind_events(p, None, 2)
    _, _, s_rel = bind_events(p, None, 2, specs=flat_specs(p, rel=True))
    assert s_abs[0]["rel"] == [0, 0]
    assert s_rel[0]["rel"] == [1, 0]


def test_rel_outer_gives_global_transposition_quotient():
    """rel_outer=True on an absolute attr -> rel = [0, 1] (the new lever)."""
    p = [np.array([[0.0, 4.0, 7.0, 11.0]])]
    _, _, specs = bind_events(p, None, 2, rel_outer=True)
    assert specs[0]["rel"] == [0, 1]


def test_sym_outer_adjustable():
    p = [np.array([[0.0, 4.0, 7.0, 11.0]])]
    _, _, s0 = bind_events(p, None, 2)
    _, _, s1 = bind_events(p, None, 2, sym_outer=True)
    assert s0[0]["sym"] == [True, False]
    assert s1[0]["sym"] == [True, True]


def test_reproduces_old_tensor_join():
    """Outer r=L, rel=[rel_in,0] equals the old separate-attribute join.

    Single-value events (K=1): the nested density equals the tensor join
    of L single-value attributes (eval-identical), the §6.5 parity that
    n-gram entropy relies on.
    """
    diffs = np.array([[2.0, -1.0, 3.0, 0.0, -2.0, 1.0, 4.0]])
    n = 3
    pb, wb, specs = bind_events([diffs], None, n, circular=True)
    d_new = build_exp_tens(pb, wb, specs=specs, sigma=[10.0], is_per=[True],
                           period=[12.0], verbose=False)
    idx = lambda ell: (np.arange(7) + ell) % 7
    p_old = [diffs[:, idx(ell)] for ell in range(n)]
    d_old = build_exp_tens(p_old, None, [10.0] * n, [1] * n, [False] * n,
                           [True] * n, [12.0] * n, verbose=False)
    assert d_new.dim == d_old.dim == n
    rng = np.random.default_rng(0)
    xs = rng.uniform(-6, 6, size=(n, 6))
    np.testing.assert_allclose(_ev(d_new, xs), _ev(d_old, xs), atol=1e-12)


def test_circular_vs_noncircular_sizes():
    p = [np.array([[0.0, 4.0, 7.0, 11.0, 2.0]])]   # N=5
    pb_nc, _, _ = bind_events(p, None, 2, circular=False)
    pb_c, _, _ = bind_events(p, None, 2, circular=True)
    assert pb_nc[0].shape == (2, 4)   # N' = 5 - 2 + 1
    assert pb_c[0].shape == (2, 5)    # N' = N


def test_per_attribute_orders_and_alignment():
    """Per-attribute L: smaller-L attribute keeps leading N' windows."""
    p = [np.array([[0.0, 1.0, 2.0, 3.0, 4.0]]),
         np.array([[10.0, 11.0, 12.0, 13.0, 14.0]])]
    pb, _, specs = bind_events(p, None, [1, 3])
    n_prime = 5 - 3 + 1
    assert "tags" not in specs[0] and "tags" in specs[1]
    assert pb[0].shape == (1, n_prime)        # L=1 flat, trailing-aligned
    assert pb[1].shape == (3, n_prime)        # L=3 nested
    np.testing.assert_allclose(pb[0], p[0][:, :n_prime])


def test_k_a_greater_than_one_tags():
    """Inner multiset (K_a>1): tags repeat per event block."""
    p = [np.array([[0.0, 7.0], [4.0, 11.0]])]   # K=2, N=2
    pb, _, specs = bind_events(p, None, 2, specs=flat_specs(p, r=2))
    assert list(specs[0]["tags"]) == [0, 0, 1, 1]
    assert pb[0].shape == (4, 1)                # (L*K, N') = (4, 2-2+1)


def test_event_dependent_weight_stacks():
    """Per-event weights are windowed and stacked to match the value layout."""
    p = [np.array([[0.0, 4.0, 7.0, 11.0]])]
    w = [np.array([[1.0, 2.0, 3.0, 4.0]])]      # (1, N) per-event
    _, wb, _ = bind_events(p, w, 2)
    # L=2, N'=3: block ell=0 -> w[0:3], ell=1 -> w[1:4]
    np.testing.assert_allclose(np.asarray(wb[0]),
                               np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]))


def test_scalar_weight_passes_through():
    p = [np.array([[0.0, 4.0, 7.0, 11.0]])]
    _, wb, _ = bind_events(p, 0.7, 2)
    assert wb == 0.7


def test_deep_nesting_supported():
    """bind_events deepens an already-nested attribute: feeding back the
    specs returned by a prior bind appends a new outermost level, so r/sym/
    rel each extend by one entry (verified to three levels)."""
    p = [np.array([[0.0, 4.0, 7.0, 11.0, 2.0, 9.0]])]
    pb, wb, s1 = bind_events(p, None, 2, sym_outer=True)           # two-level
    pb2, wb2, s2 = bind_events(pb, wb, 3, sym_outer=False, specs=s1)  # three-level
    assert s2[0]["r"] == [1, 2, 3]
    assert s2[0]["sym"] == [True, True, False]
    assert s2[0]["rel"] == [0, 0, 0]
    assert np.asarray(s2[0]["tags"]).ndim == 2


def test_deep_nesting_without_specs_silently_flattens():
    """The documented footgun: omitting specs on an already-bound triple
    re-synthesises flat specs and discards the existing nesting (shallower
    result), rather than deepening."""
    p = [np.array([[0.0, 4.0, 7.0, 11.0, 2.0, 9.0]])]
    pb, wb, _ = bind_events(p, None, 2, sym_outer=True)
    _, _, s_flat = bind_events(pb, wb, 3, sym_outer=False)   # no specs threaded
    assert s_flat[0]["r"] == [1, 3]                          # beat level lost


def test_name_from_kwarg_and_inherited():
    p = [np.array([[0.0, 4.0, 7.0, 11.0]])]
    # kwarg name stamps and overrides
    _, _, specs = bind_events(p, None, 2, name="steps",
                              level_names=["step", "ngram"])
    assert specs[0]["name"] == "steps"
    assert specs[0]["names"] == ["step", "ngram"]
    # name carried on the incoming spec is inherited when no kwarg
    _, _, sp2 = bind_events(p, None, 2, specs=flat_specs(p, name="pitch"))
    assert sp2[0]["name"] == "pitch"


def test_invalid_orders_error():
    p = [np.array([[0.0, 4.0, 7.0, 11.0]])]
    with pytest.raises(ValueError):
        bind_events(p, None, 0)
    with pytest.raises(ValueError):
        bind_events(p, None, 1.5)
    with pytest.raises(ValueError):
        bind_events(p, None, 5)  # L > N noncircular


def test_specs_wrong_length_errors():
    p = [np.array([[0.0, 4.0, 7.0, 11.0]]), np.array([[1.0, 2.0, 3.0, 4.0]])]
    with pytest.raises(ValueError):
        bind_events(p, None, 2, specs=[{"r": 1, "rel": False, "sym": True}])


def test_n_tuple_entropy_still_works():
    """The migrated internal caller produces finite, n-scaling entropy."""
    p = np.array([0.0, 2.0, 5.0, 7.0, 9.0, 11.0, 4.0, 6.0])
    h1, _ = mpt.n_tuple_entropy(p, 12.0, 1, sigma=20.0, method="shannon")
    h2, _ = mpt.n_tuple_entropy(p, 12.0, 2, sigma=20.0, method="shannon")
    assert np.isfinite(h1) and np.isfinite(h2)
    assert h2 > h1


# ---------------------------------------------------------------------------
#  step: hop between consecutive bound windows along the event axis
# ---------------------------------------------------------------------------

def test_step_default_is_overlapping_slide():
    """step=1 (default) is the fully overlapping slide: N' = N - L + 1."""
    p = [np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]])]
    pb, _, _ = bind_events(p, None, 2)
    pb1, _, _ = bind_events(p, None, 2, step=1)
    assert np.array_equal(np.asarray(pb[0]), np.asarray(pb1[0]))
    assert np.asarray(pb[0]).shape == (2, 5)


def test_step_nonoverlapping_blocks():
    """step=L gives non-overlapping blocks, matching the ::step subsample
    of the overlapping slide."""
    p = [np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]])]
    pb2, _, _ = bind_events(p, None, 2, step=2)
    assert np.array_equal(np.asarray(pb2[0]), np.array([[0, 2, 4], [1, 3, 5]]))
    pb1, _, _ = bind_events(p, None, 2)
    assert np.array_equal(np.asarray(pb2[0]), np.asarray(pb1[0])[:, ::2])


def test_step_weights_hop_with_windows():
    p = [np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]])]
    w = [np.array([[1.0, 0.5, 1.0, 0.5, 1.0, 0.5]])]
    _, wb, _ = bind_events(p, w, 2, step=2)
    assert np.array_equal(np.asarray(wb[0]),
                          np.array([[1.0, 1.0, 1.0], [0.5, 0.5, 0.5]]))


def test_step_multiattr_common_hop():
    pA = np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]])
    pB = np.array([[10.0, 11.0, 12.0, 13.0, 14.0, 15.0]])
    pb, _, _ = bind_events([pA, pB], None, [2, 2], step=2)
    assert np.asarray(pb[0]).shape == (2, 3)
    assert np.asarray(pb[1]).shape == (2, 3)


def test_step_circular_requires_divisibility():
    p6 = [np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]])]
    pb, _, _ = bind_events(p6, None, 2, circular=True, step=2)
    assert np.asarray(pb[0]).shape[1] == 3            # N' = N / step
    p5 = [np.array([[0.0, 1.0, 2.0, 3.0, 4.0]])]
    with pytest.raises(ValueError):
        bind_events(p5, None, 2, circular=True, step=2)


def test_step_validation():
    p = [np.array([[0.0, 1.0, 2.0, 3.0]])]
    with pytest.raises(ValueError):
        bind_events(p, None, 2, step=0)
    with pytest.raises(TypeError):
        bind_events(p, None, 2, step=2.5)
    with pytest.raises(ValueError):
        bind_events(p, None, 2, step=np.array([2, 2]))


def test_step_two_stage_metrical_grouping():
    """eighths -> beats (L=2, step=2) -> cadence (L=3, step=1) builds the
    three-level metrical structure with no manual subsample."""
    e = [np.array([[0.0, 2.0, 7.0, 7.0, 0.0, 0.0]])]
    w = [np.array([[1.0, 0.5, 1.0, 0.5, 1.0, 0.5]])]
    pb, wb, s = bind_events(e, w, 2, step=2, sym_outer=True)
    assert np.asarray(pb[0]).shape == (2, 3)          # 3 clean beats
    pb, wb, s = bind_events(pb, wb, 3, sym_outer=False, specs=s)
    assert s[0]["r"] == [1, 2, 3]
    assert s[0]["sym"] == [True, True, False]
