"""Tests for the specs-emitting bind_events (3c-ii).

bind_events nests sliding windows of consecutive events into a single
nested attribute per input attribute (toolbox spec §6.1/§6.5): the bound
events form an ordered outer level (sym_outer = 0 by default), each event's
own multiset is the inner level (inheriting the original r/is_rel/is_sym).
It returns (p_attr_bound, w_bound, specs) ready for build_exp_tens(...,
specs=...). L = 1 is a flat passthrough. Outer r = L with rel = [is_rel, 0]
reproduces the old separate-attribute tensor join (§6.5).
"""

import numpy as np
import pytest

import mpt
from mpt import build_exp_tens, eval_exp_tens
from mpt._tensor.preprocessing import bind_events


def _ev(d, x):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return np.ravel(eval_exp_tens(d, x, verbose=False))


def test_returns_three_tuple_with_specs():
    p = [np.array([[0.0, 4.0, 7.0, 11.0, 2.0]])]
    pb, wb, specs = bind_events(p, None, 2, [1], [False], [True])
    assert isinstance(specs, list) and len(specs) == 1
    assert "tags" in specs[0]  # nested


def test_order_1_is_flat_passthrough():
    """L = 1 emits a flat spec (no tags) and the trailing-aligned values."""
    p = [np.array([[0.0, 4.0, 7.0, 11.0, 2.0]])]
    pb, _, specs = bind_events(p, None, 1, [3], [True], [True])
    assert "tags" not in specs[0]
    assert specs[0] == {"r": 3, "rel": True, "sym": True}
    np.testing.assert_allclose(pb[0], p[0])  # circular=False, max_order=1 -> N'=N


def test_nested_spec_structure_and_inheritance():
    """Inner level inherits original r/is_rel/is_sym; outer r=L, sym=0, rel=0."""
    p = [np.array([[0.0, 4.0], [7.0, 11.0], [1.0, 2.0]]).T]  # K=2, N=3 -> shape (2,3)
    p = [np.array([[0.0, 4.0, 7.0], [10.0, 12.0, 14.0]])]    # K=2, N=3
    pb, _, specs = bind_events(p, None, 2, [2], [True], [True])
    s = specs[0]
    assert s["r"] == [2, 2]          # inner = original r=2, outer = L=2
    assert s["sym"] == [True, False]  # inner inherits, outer ordered
    assert s["rel"] == [1, 0]         # inner inherits is_rel=True, outer off
    assert list(s["tags"]) == [0, 0, 1, 1]
    assert pb[0].shape == (4, 2)      # (L*K, N') = (2*2, 3-2+1)


def test_rel_default_inherits_inner_off_outer():
    """rel defaults to [is_rel, 0]; absolute attr -> [0, 0]."""
    p = [np.array([[0.0, 4.0, 7.0, 11.0]])]
    _, _, s_abs = bind_events(p, None, 2, [1], [False], [True])
    _, _, s_rel = bind_events(p, None, 2, [1], [True], [True])
    assert s_abs[0]["rel"] == [0, 0]
    assert s_rel[0]["rel"] == [1, 0]


def test_rel_outer_gives_global_transposition_quotient():
    """rel_outer=True on an absolute attr -> rel = [0, 1] (the new lever)."""
    p = [np.array([[0.0, 4.0, 7.0, 11.0]])]
    _, _, specs = bind_events(p, None, 2, [1], [False], [True], rel_outer=True)
    assert specs[0]["rel"] == [0, 1]


def test_sym_outer_adjustable():
    p = [np.array([[0.0, 4.0, 7.0, 11.0]])]
    _, _, s0 = bind_events(p, None, 2, [1], [False], [True])
    _, _, s1 = bind_events(p, None, 2, [1], [False], [True], sym_outer=True)
    assert s0[0]["sym"] == [True, False]
    assert s1[0]["sym"] == [True, True]


def test_reproduces_old_tensor_join():
    """Outer r=L, rel=[is_rel,0] equals the old separate-attribute join.

    Single-value events (K=1): the nested density equals the tensor join
    of L single-value attributes (eval-identical), the §6.5 parity that
    n-gram entropy relies on.
    """
    diffs = np.array([[2.0, -1.0, 3.0, 0.0, -2.0, 1.0, 4.0]])
    n = 3
    pb, wb, specs = bind_events([diffs], None, n, [1], [False], [True],
                                circular=True)
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
    pb_nc, _, _ = bind_events(p, None, 2, [1], [False], [True], circular=False)
    pb_c, _, _ = bind_events(p, None, 2, [1], [False], [True], circular=True)
    assert pb_nc[0].shape == (2, 4)   # N' = 5 - 2 + 1
    assert pb_c[0].shape == (2, 5)    # N' = N


def test_per_attribute_orders_and_alignment():
    """Per-attribute L: smaller-L attribute keeps leading N' windows."""
    p = [np.array([[0.0, 1.0, 2.0, 3.0, 4.0]]),
         np.array([[10.0, 11.0, 12.0, 13.0, 14.0]])]
    pb, _, specs = bind_events(p, None, [1, 3], [1, 1], [False, False],
                               [True, True])
    n_prime = 5 - 3 + 1
    assert "tags" not in specs[0] and "tags" in specs[1]
    assert pb[0].shape == (1, n_prime)        # L=1 flat, trailing-aligned
    assert pb[1].shape == (3, n_prime)        # L=3 nested
    np.testing.assert_allclose(pb[0], p[0][:, :n_prime])


def test_k_a_greater_than_one_tags():
    """Inner multiset (K_a>1): tags repeat per event block."""
    p = [np.array([[0.0, 7.0], [4.0, 11.0]])]   # K=2, N=2
    pb, _, specs = bind_events(p, None, 2, [2], [False], [True])
    assert list(specs[0]["tags"]) == [0, 0, 1, 1]
    assert pb[0].shape == (4, 1)                # (L*K, N') = (4, 2-2+1)


def test_event_dependent_weight_stacks():
    """Per-event weights are windowed and stacked to match the value layout."""
    p = [np.array([[0.0, 4.0, 7.0, 11.0]])]
    w = [np.array([[1.0, 2.0, 3.0, 4.0]])]      # (1, N) per-event
    _, wb, _ = bind_events(p, w, 2, [1], [False], [True])
    # L=2, N'=3: block ell=0 -> w[0:3], ell=1 -> w[1:4]
    np.testing.assert_allclose(np.asarray(wb[0]),
                               np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]))


def test_scalar_weight_passes_through():
    p = [np.array([[0.0, 4.0, 7.0, 11.0]])]
    _, wb, _ = bind_events(p, 0.7, 2, [1], [False], [True])
    assert wb == 0.7


def test_names_stamped():
    p = [np.array([[0.0, 4.0, 7.0, 11.0]])]
    _, _, specs = bind_events(p, None, 2, [1], [False], [True],
                              name="steps", level_names=["step", "ngram"])
    assert specs[0]["name"] == "steps"
    assert specs[0]["names"] == ["step", "ngram"]


def test_invalid_orders_error():
    p = [np.array([[0.0, 4.0, 7.0, 11.0]])]
    with pytest.raises(ValueError):
        bind_events(p, None, 0, [1], [False], [True])
    with pytest.raises(ValueError):
        bind_events(p, None, 1.5, [1], [False], [True])
    with pytest.raises(ValueError):
        bind_events(p, None, 5, [1], [False], [True])  # L > N noncircular


def test_n_tuple_entropy_still_works():
    """The migrated internal caller produces finite, n-scaling entropy."""
    p = np.array([0.0, 2.0, 5.0, 7.0, 9.0, 11.0, 4.0, 6.0])
    h1, _ = mpt.n_tuple_entropy(p, 12.0, 1, sigma=20.0, method="shannon")
    h2, _ = mpt.n_tuple_entropy(p, 12.0, 2, sigma=20.0, method="shannon")
    assert np.isfinite(h1) and np.isfinite(h2)
    assert h2 > h1
