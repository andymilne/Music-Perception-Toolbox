"""Tests for MaetDensity lazy materialisation (v2.2+).

``build_exp_tens`` builds the per-tuple permutation arrays
(``centres``, ``u_perm``, ``w_perm``, ``v_comb``, ``wv_comb``) lazily.
At high K and high r, ``n_j = K!/(K-r)!`` makes those arrays
prohibitive: K=256, r=4 gives ~9·10⁸ four-tuples. Consumers that only
need the Möbius path (``eval_exp_tens method='mobius'``,
``cos_sim_exp_tens method='mobius'``, ``entropy_exp_tens
method='renyi2'``) read just ``p``, ``w``, ``sigma``, ``r``, and the
like, so the per-tuple arrays stay unbuilt.

The build lives in lazy properties on ``MaetDensity``: each per-tuple
field triggers a single shared build pass on first access, and
subsequent reads return the cached array. Orbit-only consumer chains
never trigger materialisation.

These tests verify:
1. ``build_exp_tens`` returns a non-materialised density.
2. Reading scalar inputs (``p``, ``w``, ``sigma``, ``r``, etc.)
   does not trigger materialisation.
3. Reading any per-tuple field triggers materialisation; subsequent
   reads return the cached object.
4. Orbit-path consumers (cosine, Rényi-2, eval-orbit) leave the
   density lazy.
5. Centres-path and pairwise-path consumers materialise the density.
6. The lazily-built fields produce numerically identical output to
   the v2.1 eager build (regression check via the orbit-vs-centres
   eval comparison and self-similarity).
7. The headline use case: building a K=256, r=4 density and
   evaluating via orbit succeeds without materialising the
   ``9·10⁸``-tuple intermediates.
"""
import time

import numpy as np
from mpt._tensor.density import single_multiset_view
import pytest

from mpt.tensor import build_exp_tens, cos_sim_exp_tens, eval_exp_tens
from mpt.entropy import entropy_exp_tens


P = 1200.0


# -------------------------------------------------------------------
#  1–3. Materialisation triggers
# -------------------------------------------------------------------


def _make_dens(K=8, r=3, sigma=33.0, is_rel=False, is_per=True, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0, P, K)
    w = rng.uniform(0.5, 1.5, K)
    return build_exp_tens(p, w, sigma, r, is_rel, is_per, P, verbose=False)


def test_build_returns_lazy_density():
    """Right after construction, the per-tuple arrays must not exist."""
    T = _make_dens()
    assert T.materialised is False


@pytest.mark.parametrize(
    "field",
    ["p", "w", "sigma", "r", "is_rel", "is_per", "period", "dim"],
)
def test_scalar_input_reads_do_not_materialise(field):
    """Reading any of the eagerly-stored input fields must not
    trigger the per-tuple build."""
    T = _make_dens()
    T = single_multiset_view(T)
    _ = getattr(T, field)
    assert T.materialised is False


@pytest.mark.parametrize(
    "field",
    ["centres", "u_perm", "w_perm", "w_j", "v_comb", "wv_comb",
     "n_j", "n_j_perm", "n_k"],
)
def test_per_tuple_field_reads_trigger_materialisation(field):
    """Reading any per-tuple field triggers the build and the
    density becomes materialised."""
    T = _make_dens()
    T = single_multiset_view(T)
    assert T.materialised is False
    _ = getattr(T, field)
    assert T.materialised is True


def test_per_tuple_array_caches_across_reads():
    """Per-tuple arrays are cached; repeated reads return the same
    object (no rebuild)."""
    T = _make_dens()
    c1 = T.centres
    c2 = T.centres
    assert c1 is c2


def test_w_j_aliases_w_perm():
    """w_j and w_perm exposed the same array in v2.1; the lazy class
    preserves this alias."""
    T = _make_dens()
    T = single_multiset_view(T)
    assert T.w_j is T.w_perm


def test_n_j_aliases_n_j_perm():
    """n_j and n_j_perm exposed the same value in v2.1; preserved."""
    T = _make_dens()
    T = single_multiset_view(T)
    assert T.n_j == T.n_j_perm


# -------------------------------------------------------------------
#  4. Orbit-only consumers do not materialise
# -------------------------------------------------------------------


def test_cos_sim_orbit_does_not_materialise():
    T_x = _make_dens(K=10, r=3, seed=0)
    T_y = _make_dens(K=10, r=3, seed=1)
    cos_sim_exp_tens(T_x, T_y, method="mobius", verbose=False)
    assert T_x.materialised is False
    assert T_y.materialised is False


def test_entropy_renyi2_does_not_materialise():
    T = _make_dens(K=10, r=3)
    entropy_exp_tens(T, method="renyi2")
    assert T.materialised is False


def test_eval_orbit_does_not_materialise():
    rng = np.random.default_rng(0)
    T = _make_dens(K=10, r=3)
    x = rng.uniform(0, P, (3, 5))
    eval_exp_tens(T, x, method="mobius", verbose=False)
    assert T.materialised is False


# -------------------------------------------------------------------
#  5. Centres-path / pairwise consumers materialise as expected
# -------------------------------------------------------------------


def test_eval_centres_materialises():
    rng = np.random.default_rng(0)
    T = _make_dens(K=8, r=3)
    x = rng.uniform(0, P, (3, 5))
    eval_exp_tens(T, x, method="centres", verbose=False)
    assert T.materialised is True


def test_cos_sim_pairwise_materialises():
    T_x = _make_dens(K=8, r=3, seed=0)
    T_y = _make_dens(K=8, r=3, seed=1)
    cos_sim_exp_tens(T_x, T_y, method="bulger", verbose=False)
    assert T_x.materialised is True
    assert T_y.materialised is True


# -------------------------------------------------------------------
#  6. Numerical identity with v2.1 eager build
# -------------------------------------------------------------------


def test_lazy_centres_match_eager_build():
    """The centres-path eval result must be unchanged compared to
    the eager build. Self-similarity = 1 is the simplest invariant
    that exercises the centres array, the perm-side weights, and
    the comb-side weights together."""
    T = _make_dens(K=10, r=3)
    c = cos_sim_exp_tens(T, T, method="bulger", verbose=False)
    assert np.isclose(c, 1.0, atol=1e-12, rtol=1e-12)


def test_orbit_vs_centres_after_lazy_build():
    """Evaluating the same density via both paths must agree to
    FP precision. Touching the centres path materialises the per-
    tuple arrays; the second (orbit) call is unaffected by that
    materialisation."""
    rng = np.random.default_rng(0)
    T = _make_dens(K=10, r=3)
    x = rng.uniform(0, P, (3, 20))
    v_centres = eval_exp_tens(T, x, method="centres", verbose=False)
    v_orbit = eval_exp_tens(T, x, method="mobius", verbose=False)
    # inf resolves to the accuracy-floor width; centres and Möbius truncate
    # that boundary via different code and disagree by up to a few x1e-12.
    assert np.allclose(v_centres, v_orbit, atol=1e-11, rtol=1e-9)


# -------------------------------------------------------------------
#  7. Headline case: K=256, r=4, orbit eval succeeds
# -------------------------------------------------------------------


def test_high_K_high_r_orbit_eval_does_not_oom():
    """K=256, r=4 yields n_j ≈ 9·10⁸ four-tuples — pre-v2.2 the
    eager build allocated arrays totalling >25 GB and OOM'd. Lazy
    materialisation skips the allocation entirely when only the
    Möbius method is exercised. Build time should be sub-millisecond
    and the orbit eval at a small query batch should complete in
    a sane time budget."""
    rng = np.random.default_rng(0)
    K, r = 256, 4
    p = rng.uniform(0, P, K)
    w = rng.uniform(0.5, 1.5, K)

    t0 = time.perf_counter()
    T = build_exp_tens(p, w, 33.0, r, False, True, P, verbose=False)
    build_elapsed = time.perf_counter() - t0
    # Build should be near-instant — no permutation allocation.
    assert build_elapsed < 0.1, (
        f"build took {build_elapsed:.3f} s; lazy build should be < 0.1 s"
    )
    assert T.materialised is False

    # Orbit eval at 4 queries.
    x = rng.uniform(0, P, (r, 4))
    t0 = time.perf_counter()
    v = eval_exp_tens(T, x, method="mobius", verbose=False)
    eval_elapsed = time.perf_counter() - t0

    assert np.all(np.isfinite(v))
    assert np.all(v > 0)
    assert eval_elapsed < 60.0, (
        f"orbit eval took {eval_elapsed:.1f} s; expected < 60 s"
    )
    # Crucially, orbit eval did NOT trigger a build of the per-tuple
    # arrays — that would have OOM'd at K=256 r=4.
    assert T.materialised is False
