"""Tests for MaetDensity lazy materialisation (v2.2).

Parallel to ``test_lazy_density.py`` for the single-multiset path. Verifies:

1. ``build_exp_tens`` returns a lazy MA density.
2. Reading any of the eager fields (``p_attr``, ``w``, ``sigma``,
   ``r``, ``k``, ``is_rel``, ``is_per``, ``period``,
   ``n_attrs``,
   ``n``, ``dim``, ``dim_per_attr``, ``tag``) does not materialise.
3. Reading any of the lazy fields (``n_j``, ``n_k``, ``centres``,
   ``u_perm``, ``v_comb``, ``w_j``, ``wv_comb``, ``event_of_j``,
   ``event_of_k``) materialises and caches.
4. MA orbit-only consumers (cosine ``method='mobius'``, Rényi-2)
   do not materialise.
5. MA centres-path / pairwise consumers do materialise.
6. Lazy build produces numerically identical output to the eager
   v2.1 path (regression check via cosine self-similarity = 1 and
   orbit-vs-pairwise agreement).
7. Eager input-validation errors (insufficient non-NaN slots,
   wrong-shaped vectors) still fire at the ``build_exp_tens`` call
   rather than being deferred.
"""
import numpy as np
import pytest

import mpt
from mpt.tensor import build_exp_tens, cos_sim_exp_tens, eval_exp_tens
from mpt.entropy import entropy_exp_tens


P = 1200.0


def _make_ma(seed=0):
    """Build a small 2-attribute MA density (pitch + phase)."""
    rng = np.random.default_rng(seed)
    A, N = 2, 5
    p_attr = [
        rng.uniform(0, P, (3, N)),
        rng.uniform(0, 1.0, (1, N)),
    ]
    w = [
        rng.uniform(0.5, 1.5, (3, N)),
        rng.uniform(0.5, 1.5, (1, N)),
    ]
    return build_exp_tens(
        p_attr, w, [33.0, 0.05], [2, 1], 
        [False, False], [True, True], [P, 1.0], verbose=False,
    )


# -------------------------------------------------------------------
#  1–3. Materialisation triggers
# -------------------------------------------------------------------


def test_ma_build_returns_lazy_density():
    T = _make_ma()
    assert T.materialised is False


@pytest.mark.parametrize(
    "field",
    ["p_attr", "w", "sigma", "r", "k", "is_rel", "is_per", "period",
     "n_attrs", "n",
     "dim", "dim_per_attr", "tag"],
)
def test_ma_eager_field_reads_do_not_materialise(field):
    T = _make_ma()
    _ = getattr(T, field)
    assert T.materialised is False


@pytest.mark.parametrize(
    "field",
    ["n_j", "n_k", "centres", "u_perm", "v_comb", "w_j", "wv_comb",
     "event_of_j", "event_of_k"],
)
def test_ma_lazy_field_reads_trigger_materialisation(field):
    T = _make_ma()
    assert T.materialised is False
    _ = getattr(T, field)
    assert T.materialised is True


def test_ma_lazy_arrays_cache_across_reads():
    T = _make_ma()
    c1 = T.centres
    c2 = T.centres
    # Same list AND same per-attribute arrays — no rebuild.
    assert c1 is c2
    for a in range(len(c1)):
        assert c1[a] is c2[a]


# -------------------------------------------------------------------
#  4. Orbit-only consumers do not materialise
# -------------------------------------------------------------------


def test_ma_cos_sim_orbit_does_not_materialise():
    T_x = _make_ma(seed=0)
    T_y = _make_ma(seed=1)
    cos_sim_exp_tens(T_x, T_y, method="mobius", verbose=False)
    assert T_x.materialised is False
    assert T_y.materialised is False


def test_ma_renyi2_does_not_materialise():
    T = _make_ma()
    entropy_exp_tens(T, method="renyi2")
    assert T.materialised is False


# -------------------------------------------------------------------
#  5. Centres-path / pairwise consumers materialise
# -------------------------------------------------------------------


def test_ma_eval_materialises():
    rng = np.random.default_rng(42)
    T = _make_ma()
    # Query dim = sum of dim_per_attr = (r_a or r_a-1 by isRel per a)
    dim = int(T.dim)
    x = np.empty((dim, 5))
    x[0, :] = rng.uniform(0, P, 5)
    x[1, :] = rng.uniform(0, P, 5)
    x[2, :] = rng.uniform(0, 1.0, 5)
    eval_exp_tens(T, x, verbose=False)
    assert T.materialised is True


def test_ma_cos_sim_pairwise_materialises():
    T_x = _make_ma(seed=0)
    T_y = _make_ma(seed=1)
    cos_sim_exp_tens(T_x, T_y, method="bulger", verbose=False)
    assert T_x.materialised is True
    assert T_y.materialised is True


# -------------------------------------------------------------------
#  6. Numerical contract preserved
# -------------------------------------------------------------------


def test_ma_self_similarity_equals_one():
    T = _make_ma()
    c = cos_sim_exp_tens(T, T, method="bulger", verbose=False)
    assert np.isclose(c, 1.0, atol=1e-12, rtol=1e-12)


def test_ma_orbit_vs_pairwise_agree():
    """At a clean cell, MA orbit cosine and MA pairwise cosine agree
    to FP precision. Orbit reads only eager fields; pairwise
    materialises. Both paths must see the same density mathematically."""
    T_x = _make_ma(seed=0)
    T_y = _make_ma(seed=1)
    c_orbit = cos_sim_exp_tens(T_x, T_y, method="mobius", verbose=False)
    # Use distinct density objects so the orbit call's "no
    # materialisation" assertion stays meaningful.
    T_x2 = _make_ma(seed=0)
    T_y2 = _make_ma(seed=1)
    c_pw = cos_sim_exp_tens(T_x2, T_y2, method="bulger", verbose=False)
    assert np.isclose(c_orbit, c_pw, atol=1e-12, rtol=1e-10)


# -------------------------------------------------------------------
#  7. Eager validation errors fire at build time
# -------------------------------------------------------------------


def test_ma_insufficient_slots_eager_error():
    """The K_na < r_a check is eager: it fires at the build call,
    not deferred to first lazy-field access. Users reasonably expect
    malformed inputs to fail fast."""
    pitch = np.array([[0.0, 0.0], [4.0, np.nan], [np.nan, np.nan]])
    with pytest.raises(ValueError, match="non-NaN value"):
        build_exp_tens(
            [pitch], None, [10.0], [2], 
            [False], [True], [1200.0], verbose=False,
        )


def test_ma_wrong_r_vec_length_eager_error():
    """r_vec length mismatch fires eagerly at the build call."""
    pitch = np.array([[0.0, 4.0]])
    with pytest.raises(ValueError, match="r_vec"):
        build_exp_tens(
            [pitch, pitch], None, [10.0, 10.0], [1], 
            [False, False], [True, True], [1200.0, 1200.0],
            verbose=False,
        )
