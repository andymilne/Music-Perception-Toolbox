"""Tests for the per-attribute ``[sym]`` (symmetrisation) flag.

Covers the predictions of the sym-flag specification §10:

* ``r = 1``: the flag is vacuous; ``[sym] = 0`` and ``[sym] = 1``
  coincide (single-attribute and multi-attribute paths).
* ``r = K``: ``[sym] = 0`` deposits the single ordered tuple (one
  kernel); ``[sym] = 1`` deposits the full ``S_K`` orbit (``K!``
  kernels).
* ``1 < r < K``: ``[sym] = 0`` is the ``[sym] = 1`` density with the
  reflections removed (the de-reflected density). Verified through the
  exact orbit-sum relation between the two readings.
* OPT-completeness: ``[sym] = 0`` with ``[rel] = 1`` reaches the
  ordered transposition-invariant spaces, distinguishing an ordered
  interval from its inversion (which ``[sym] = 1`` cannot).
* Cross-cardinality comparability at fixed ``r``: a triad and a seventh
  chord overlap is well posed, and a doubling reweights without
  equalising (anti-C).

The default value of ``[sym]`` is ``1`` (legacy symmetric reading), so
omitting the flag must reproduce the pre-flag behaviour; this is checked
implicitly throughout the rest of the suite and explicitly here.
"""

import numpy as np
import pytest

from mpt import build_exp_tens, eval_exp_tens, cos_sim_exp_tens


# ---------------------------------------------------------------------
#  r = 1: the flag is vacuous
# ---------------------------------------------------------------------

class TestVacuousAtR1:
    def test_sa_eval_coincides(self):
        p = [0.0, 4.0, 7.0, 11.0]
        x = np.linspace(-3, 14, 60)
        d_sym = build_exp_tens(p, None, 1.0, 1, False, False, 0.0, True,
                               verbose=False)
        d_ord = build_exp_tens(p, None, 1.0, 1, False, False, 0.0, False,
                               verbose=False)
        np.testing.assert_allclose(
            eval_exp_tens(d_sym, x, verbose=False),
            eval_exp_tens(d_ord, x, verbose=False),
            atol=1e-13,
        )

    def test_ma_eval_coincides(self):
        P = [np.array([[0.0, 4.0, 7.0]])]   # one attribute, K=1 slot/event
        x = np.linspace(-3, 12, 50).reshape(1, -1)
        d_sym = build_exp_tens(P, None, [1.0], [1], [False], [False], [0.0],
                               [True], verbose=False)
        d_ord = build_exp_tens(P, None, [1.0], [1], [False], [False], [0.0],
                               [False], verbose=False)
        np.testing.assert_allclose(
            eval_exp_tens(d_sym, x, verbose=False),
            eval_exp_tens(d_ord, x, verbose=False),
            atol=1e-13,
        )


# ---------------------------------------------------------------------
#  Centre counts: ordered vs symmetric
# ---------------------------------------------------------------------

class TestCentreCounts:
    @pytest.mark.parametrize("r,expect_ord,expect_sym", [
        (1, 4, 4),     # C(4,1)=4; r! = 1
        (2, 6, 12),    # C(4,2)=6; x2!
        (3, 4, 24),    # C(4,3)=4; x3!
        (4, 1, 24),    # C(4,4)=1; x4!  (r=K: single tuple vs S_K orbit)
    ])
    def test_sa_u_perm_columns(self, r, expect_ord, expect_sym):
        p = [0.0, 4.0, 7.0, 11.0]   # K = 4
        d_ord = build_exp_tens(p, None, 1.0, r, False, False, 0.0, False,
                               verbose=False)
        d_sym = build_exp_tens(p, None, 1.0, r, False, False, 0.0, True,
                               verbose=False)
        assert d_ord.u_perm.shape[1] == expect_ord
        assert d_sym.u_perm.shape[1] == expect_sym

    def test_r_eq_k_single_ordered_tuple(self):
        p = [3.0, 1.0, 8.0]   # K = 3, deliberately unsorted
        d_ord = build_exp_tens(p, None, 1.0, 3, False, False, 0.0, False,
                               verbose=False)
        # Exactly one centre, the tuple in listed order.
        assert d_ord.u_perm.shape[1] == 1
        np.testing.assert_array_equal(d_ord.u_perm.ravel(), [3.0, 1.0, 8.0])

    def test_ma_u_perm_columns(self):
        P = [np.array([[0.0], [4.0], [7.0]])]   # one event, K=3 slots
        d_ord = build_exp_tens(P, None, [1.0], [2], [False], [False], [0.0],
                               [False], verbose=False)
        d_sym = build_exp_tens(P, None, [1.0], [2], [False], [False], [0.0],
                               [True], verbose=False)
        assert d_ord.u_perm[0].shape[1] == 3    # C(3,2)
        assert d_sym.u_perm[0].shape[1] == 6    # x2!


# ---------------------------------------------------------------------
#  1 < r < K: the de-reflected density relation
# ---------------------------------------------------------------------

class TestDeReflection:
    def test_r2_symmetric_is_ordered_plus_reflection(self):
        """At r=2, eval of the symmetric density at (qx, qy) equals the
        ordered eval at (qx, qy) plus the ordered eval at the swapped
        point (qy, qx): the orbit is exactly {identity, transposition}.
        """
        p = [0.0, 4.0, 7.0]
        d_ord = build_exp_tens(p, None, 1.3, 2, False, False, 0.0, False,
                               verbose=False)
        d_sym = build_exp_tens(p, None, 1.3, 2, False, False, 0.0, True,
                               verbose=False)
        # A scatter of 2-D query points (r=2, absolute -> dim 2).
        rng = np.random.default_rng(0)
        q = rng.uniform(-2, 9, (2, 25))
        q_swap = q[::-1, :]
        ev_sym = eval_exp_tens(d_sym, q, verbose=False)
        ev_ord = eval_exp_tens(d_ord, q, verbose=False)
        ev_ord_swap = eval_exp_tens(d_ord, q_swap, verbose=False)
        np.testing.assert_allclose(ev_sym, ev_ord + ev_ord_swap, atol=1e-12)

    def test_ordered_distinguishes_order_symmetric_does_not(self):
        """Ascending vs descending: identical under [sym]=1 (order
        ignored), distinct under [sym]=0."""
        asc = [0.0, 4.0, 7.0]
        desc = [7.0, 3.0, 0.0]
        s_sym = cos_sim_exp_tens(
            build_exp_tens(asc, None, 50.0, 2, False, False, 0.0, True,
                           verbose=False),
            build_exp_tens(desc, None, 50.0, 2, False, False, 0.0, True,
                           verbose=False),
            verbose=False,
        )
        s_ord = cos_sim_exp_tens(
            build_exp_tens(asc, None, 50.0, 2, False, False, 0.0, False,
                           verbose=False),
            build_exp_tens(desc, None, 50.0, 2, False, False, 0.0, False,
                           verbose=False),
            verbose=False,
        )
        assert s_ord < s_sym - 1e-4

    def test_ordered_cosine_large_k_not_symmetrised(self):
        """At a cardinality large enough that the orbit (Möbius) path
        would otherwise be selected, an ordered cosine must still be
        forced onto the centres path and stay distinct from the
        symmetric reading (regression for the orbit-symmetrises bug)."""
        asc = [float(x) for x in range(8)]
        desc = [float(x) for x in range(7, -1, -1)]
        s_ord = cos_sim_exp_tens(
            build_exp_tens(asc, None, 50.0, 2, False, False, 0.0, False,
                           verbose=False),
            build_exp_tens(desc, None, 50.0, 2, False, False, 0.0, False,
                           verbose=False),
            verbose=False,
        )
        s_sym = cos_sim_exp_tens(
            build_exp_tens(asc, None, 50.0, 2, False, False, 0.0, True,
                           verbose=False),
            build_exp_tens(desc, None, 50.0, 2, False, False, 0.0, True,
                           verbose=False),
            verbose=False,
        )
        assert s_sym == pytest.approx(1.0, abs=1e-9)
        assert s_ord < 1.0 - 1e-4

    def test_ordered_cosine_large_k_ma(self):
        """MA analogue: an ordered attribute at K large enough for the
        orbit path must still route to centres and stay distinct."""
        M = [np.array([[float(x)] for x in range(6)])]      # 6 slots, 1 event
        Mr = [np.array([[float(x)] for x in range(5, -1, -1)])]
        s_ord = cos_sim_exp_tens(M, None, Mr, None, [50.0], [2], [False],
                                 [False], [0.0], [False], verbose=False)
        s_sym = cos_sim_exp_tens(M, None, Mr, None, [50.0], [2], [False],
                                 [False], [0.0], [True], verbose=False)
        assert s_sym == pytest.approx(1.0, abs=1e-9)
        assert s_ord < 1.0 - 1e-4


# ---------------------------------------------------------------------
#  OPT-completeness: ordered transposition-invariant reach
# ---------------------------------------------------------------------

class TestOrderedTranspositionInvariant:
    def test_rel_dim_drops_by_one(self):
        p = [0.0, 4.0, 7.0]
        d = build_exp_tens(p, None, 1.0, 3, True, False, 0.0, False,
                           verbose=False)
        # [rel] removes one dimension regardless of [sym].
        assert d.dim == 2

    def test_ordered_interval_vs_inversion(self):
        """[sym]=0 + [rel]=1 distinguishes an ascending interval from a
        descending one (ordered, transposition-invariant); [sym]=1
        cannot (it symmetrises the pair, so +d and -d coincide)."""
        up = [0.0, 4.0]      # ordered interval +4
        down = [0.0, -4.0]   # ordered interval -4
        kw = dict(verbose=False)
        s_sym = cos_sim_exp_tens(
            build_exp_tens(up, None, 1.0, 2, True, False, 0.0, True, **kw),
            build_exp_tens(down, None, 1.0, 2, True, False, 0.0, True, **kw),
            **kw,
        )
        s_ord = cos_sim_exp_tens(
            build_exp_tens(up, None, 1.0, 2, True, False, 0.0, False, **kw),
            build_exp_tens(down, None, 1.0, 2, True, False, 0.0, False, **kw),
            **kw,
        )
        # Symmetric: the two read identically (interval magnitude only).
        assert s_sym == pytest.approx(1.0, abs=1e-9)
        # Ordered: +4 and -4 are far apart at this sigma.
        assert s_ord < 0.5


# ---------------------------------------------------------------------
#  Cross-cardinality comparability and anti-C
# ---------------------------------------------------------------------

class TestCrossCardinality:
    def test_triad_vs_seventh_well_posed(self):
        triad = [0.0, 400.0, 700.0]
        seventh = [0.0, 400.0, 700.0, 1000.0]
        s = cos_sim_exp_tens(
            build_exp_tens(triad, None, 30.0, 2, False, False, 0.0, False,
                           verbose=False),
            build_exp_tens(seventh, None, 30.0, 2, False, False, 0.0, False,
                           verbose=False),
            verbose=False,
        )
        assert np.isfinite(s)
        assert 0.0 < s < 1.0

    def test_doubling_reweights_not_equalises(self):
        """Anti-C: a doubled chord reads as similar to but distinct from
        its undoubled form under the ordered reading."""
        triad = [0.0, 400.0, 700.0]
        doubled = [0.0, 0.0, 400.0, 700.0]   # doubled root
        s = cos_sim_exp_tens(
            build_exp_tens(triad, None, 30.0, 2, False, False, 0.0, False,
                           verbose=False),
            build_exp_tens(doubled, None, 30.0, 2, False, False, 0.0, False,
                           verbose=False),
            verbose=False,
        )
        assert np.isfinite(s)
        assert s < 1.0 - 1e-6      # distinct (not equalised)
        assert s > 0.5             # but still similar


# ---------------------------------------------------------------------
#  Default value and self-similarity
# ---------------------------------------------------------------------

class TestDefaultAndSelf:
    def test_default_is_symmetric(self):
        p = [0.0, 4.0, 7.0]
        d_default = build_exp_tens(p, None, 1.0, 2, False, False, 0.0,
                                   verbose=False)
        d_sym = build_exp_tens(p, None, 1.0, 2, False, False, 0.0, True,
                               verbose=False)
        assert d_default.u_perm.shape[1] == d_sym.u_perm.shape[1]
        assert bool(np.all(d_default.is_sym))

    def test_ordered_self_similarity_is_one(self):
        p = [0.0, 4.0, 7.0]
        d = build_exp_tens(p, None, 30.0, 2, False, False, 0.0, False,
                           verbose=False)
        assert cos_sim_exp_tens(d, d, verbose=False) == pytest.approx(1.0,
                                                                      abs=1e-9)


# ---------------------------------------------------------------------
#  Ordered ([sym]=0) rejection on the batched / analytic paths
# ---------------------------------------------------------------------

class TestOrderedRejections:
    """The batched dedup keys rows by a sorted multiset and would
    over-merge order-distinct rows, so the batched cosine/eval paths
    reject [sym]=0 at r>1 rather than return a wrong answer. (Single-
    density ordered renyi2 is supported — see TestOrderedRenyi2.)"""

    def test_batched_cosine_rejects_ordered(self):
        P = np.array([[0.0, 4.0, 7.0], [7.0, 4.0, 0.0]])
        with pytest.raises(NotImplementedError):
            cos_sim_exp_tens(P, None, P, None, 30.0, 2, False, False, 0.0,
                             False, verbose=False)

    def test_batched_eval_rejects_ordered(self):
        from mpt import eval_exp_tens
        P = np.array([[0.0, 4.0, 7.0], [7.0, 4.0, 0.0]])
        x = np.array([[0.0], [4.0]])
        with pytest.raises(NotImplementedError):
            eval_exp_tens(P, None, 30.0, 2, False, False, 0.0, False, x,
                          verbose=False)

    def test_batched_entropy_rejects_ordered(self):
        from mpt import entropy_exp_tens
        P = np.array([[0.0, 4.0, 7.0], [7.0, 4.0, 0.0]])
        with pytest.raises(NotImplementedError):
            entropy_exp_tens(P, None, 1.0, 2, False, False, 0.0, False,
                             method="shannon", n_points_per_dim=50,
                             x_min=-3, x_max=12)


def _grid_renyi2(d, sig):
    """Brute-force grid estimate of -log integral p~^2 (natural log)."""
    from mpt import eval_exp_tens
    c = d.centres[0]
    dim = d.dim
    ng = 70 if dim == 1 else 45
    axes = [np.linspace(c[k].min() - 6 * sig, c[k].max() + 6 * sig, ng)
            for k in range(dim)]
    mesh = np.meshgrid(*axes, indexing="ij")
    X = np.vstack([m.ravel() for m in mesh])
    vals = eval_exp_tens(d, X, verbose=False)
    dv = float(np.prod([axes[k][1] - axes[k][0] for k in range(dim)]))
    z = vals.sum() * dv
    return -np.log(((vals / z) ** 2).sum() * dv)


class TestOrderedRenyi2:
    """Ordered ([sym]=0) renyi2 at r>1 is computed via the direct double
    sum of Gaussian overlaps (no orbit), matching a brute-force grid."""

    def test_renyi2_ordered_r1_allowed(self):
        from mpt import entropy_exp_tens
        # r=1: [sym] vacuous, so ordered must NOT raise.
        val = entropy_exp_tens([0.0, 4.0, 7.0], None, 1.0, 1, False, False,
                               0.0, False, method="renyi2")
        assert np.isfinite(val)

    @pytest.mark.parametrize("r", [2, 3])
    @pytest.mark.parametrize("rel", [False, True])
    def test_renyi2_ordered_sa_matches_grid(self, r, rel):
        from mpt import entropy_exp_tens, build_exp_tens
        P = np.array([0.0, 4.0, 7.0, 11.0])
        sig = 2.0
        h = float(entropy_exp_tens(list(P), None, sig, r, rel, False, 0.0,
                                   False, method="renyi2", base=np.e,
                                   verbose=False))
        d = build_exp_tens([P[:, None]], None, [sig], [r], [rel], [False],
                           [0.0], [False], verbose=False)
        assert h == pytest.approx(float(_grid_renyi2(d, sig)), abs=2e-2)

    def test_renyi2_ordered_ma_matches_grid(self):
        from mpt import entropy_exp_tens, build_exp_tens
        P = np.array([0.0, 4.0, 7.0, 11.0])
        sig = 2.0
        d = build_exp_tens([P[:, None]], None, [sig], [2], [False], [False],
                           [0.0], [False], verbose=False)
        h = float(entropy_exp_tens(d, method="renyi2", base=np.e,
                                   verbose=False))
        assert h == pytest.approx(float(_grid_renyi2(d, sig)), abs=2e-2)

    def test_renyi2_ordered_differs_from_symmetric(self):
        from mpt import entropy_exp_tens, build_exp_tens
        # An order-bearing tuple set: ordered and symmetric readings give
        # genuinely different collision entropies.
        P = np.array([0.0, 3.0, 8.0])
        sig = 1.5
        dO = build_exp_tens([P[:, None]], None, [sig], [2], [False], [False],
                            [0.0], [False], verbose=False)
        dS = build_exp_tens([P[:, None]], None, [sig], [2], [False], [False],
                            [0.0], [True], verbose=False)
        hO = float(entropy_exp_tens(dO, method="renyi2", verbose=False))
        hS = float(entropy_exp_tens(dS, method="renyi2", verbose=False))
        assert np.isfinite(hO) and np.isfinite(hS)
        assert abs(hO - hS) > 1e-6
