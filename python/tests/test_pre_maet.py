"""Tests for ``pack_pre_maet`` and ``unpack_pre_maet``.

``pack_pre_maet`` holds the three parts of a pre-MAET -- ``p_attr``,
``w_attr``, and ``specs`` -- in one object. What is pinned here is that
it is genuinely the same pre-MAET however it is passed: every operator
takes the whole pre-MAET or the loose triple and returns the whole, the
parts survive the round trip, and the two call forms give identical
results. The MATLAB twin is ``tests/test_pre_maet.m``.
"""

import numpy as np
import pytest

import mpt


def _pm():
    p = [np.array([[60.0, 62.0, 64.0, 65.0]]),
         np.array([[0.0, 1.0, 2.0, 3.0]])]
    w = [np.array([[1.0, 0.5, 0.75, 0.5]]), np.array([[1.0, 0.5, 0.75, 0.5]])]
    sp = mpt.flat_specs(p, r=1, names=["pitch", "time"])
    return p, w, sp


class TestConstruction:

    def test_keys_and_round_trip(self):
        p, w, sp = _pm()
        pm = mpt.pack_pre_maet(p, w, sp)
        assert sorted(pm) == ["p_attr", "specs", "w_attr"]
        p2, w2, sp2 = mpt.unpack_pre_maet(pm)
        assert all(np.array_equal(a, b) for a, b in zip(p2, p))
        assert all(np.array_equal(a, b) for a, b in zip(w2, w))
        assert sp2 == sp

    def test_unset_parts_are_none(self):
        p, _, _ = _pm()
        pm = mpt.pack_pre_maet(p)
        assert pm["w_attr"] is None and pm["specs"] is None

    def test_pre_maet_in_pre_maet_out(self):
        p, w, sp = _pm()
        pm = mpt.pack_pre_maet(p, w, sp)
        assert mpt.unpack_pre_maet(mpt.pack_pre_maet(pm)) == \
            mpt.unpack_pre_maet(pm)

    def test_one_part_replaced_leaves_the_rest(self):
        p, w, sp = _pm()
        pm = mpt.pack_pre_maet(p, w, sp)
        sp2 = [dict(s) for s in sp]
        sp2[0]["rel"] = True
        pm2 = mpt.pack_pre_maet(pm, specs=sp2)
        assert pm2["specs"][0]["rel"] is True
        assert np.array_equal(pm2["w_attr"][0], pm["w_attr"][0])
        assert pm["specs"][0]["rel"] is False      # the original is untouched

    def test_scalar_weights_broadcast_untouched(self):
        p, _, _ = _pm()
        assert mpt.pack_pre_maet(p, 2.0)["w_attr"] == 2.0

    @pytest.mark.parametrize("bad", [([1.0], "w"), ("specs", "s")])
    def test_length_mismatch_errors(self, bad):
        p, _, sp = _pm()
        with pytest.raises(ValueError, match="length A"):
            if bad[1] == "w":
                mpt.pack_pre_maet(p, [np.ones((1, 4))])
            else:
                mpt.pack_pre_maet(p, None, sp[:1])

    def test_single_attribute_must_be_wrapped(self):
        with pytest.raises(TypeError, match="Wrap a single attribute"):
            mpt.pack_pre_maet(np.array([[60.0, 62.0]]))
        p, _, sp = _pm()
        with pytest.raises(TypeError, match="Wrap a single attribute"):
            mpt.pack_pre_maet(p, None, sp[0])

    def test_unpack_rejects_a_non_pre_maet(self):
        p, _, _ = _pm()
        with pytest.raises(TypeError, match="expected a pre-MAET"):
            mpt.unpack_pre_maet(p)


class TestOperators:
    """Every operator returns a whole pre-MAET and accepts one."""

    def test_difference_events(self):
        p, w, sp = _pm()
        a = mpt.difference_events(mpt.pack_pre_maet(p, w, sp), [1, 0])
        b = mpt.difference_events(p, w, [1, 0], specs=sp)
        assert _same(a, b)

    def test_bind_events(self):
        p, w, sp = _pm()
        a = mpt.bind_events(mpt.pack_pre_maet(p, w, sp), [2, 2])
        b = mpt.bind_events(p, w, [2, 2], specs=sp)
        assert _same(a, b)

    def test_translate_attributes(self):
        p, w, sp = _pm()
        a = mpt.translate_attributes(mpt.pack_pre_maet(p, w, sp), [5.0, 0.0])
        b = mpt.translate_attributes(p, w, [5.0, 0.0], specs=sp)
        assert _same(a, b)

    def test_transform_attributes(self):
        p, w, sp = _pm()
        a = mpt.transform_attributes(mpt.pack_pre_maet(p, w, sp),
                                     [("affine", {"scale": 2.0}), None])
        b = mpt.transform_attributes(p, w, [("affine", {"scale": 2.0}), None],
                                     specs=sp)
        assert _same(a, b)

    def test_weight_events(self):
        p, w, sp = _pm()
        kw = dict(input_attr=1, target_attr=1, centre=1.5, shape=0.0, sd=1.0,
                  drop_input_attr=False)
        a = mpt.weight_events(mpt.pack_pre_maet(p, w, sp), specs=sp, **kw)
        b = mpt.weight_events(p, w, specs=sp, **kw)
        assert _same(a, b)

    def test_positional_arguments_shift_behind_a_pre_maet(self):
        p, w, sp = _pm()
        a = mpt.weight_events(mpt.pack_pre_maet(p, w, sp), 1, 1, 1.5, 0.0, sd=1.0,
                              drop_input_attr=False)
        b = mpt.weight_events(p, w, 1, 1, 1.5, 0.0, sd=1.0,
                              drop_input_attr=False, specs=sp)
        assert _same(a, b)

    def test_operators_compose_without_threading_specs(self):
        p, w, sp = _pm()
        pm = mpt.pack_pre_maet(p, w, sp)
        chained = mpt.difference_events(mpt.bind_events(pm, [2, 2]), [1, 1])
        pb, wb, sb = mpt.unpack_pre_maet(mpt.bind_events(p, w, [2, 2],
                                                         specs=sp))
        threaded = mpt.difference_events(pb, wb, [1, 1], specs=sb)
        assert _same(chained, threaded)

    def test_weights_may_not_be_passed_twice(self):
        p, w, sp = _pm()
        with pytest.raises(TypeError, match="must not be passed again"):
            mpt.difference_events(mpt.pack_pre_maet(p, w, sp), w, [1, 0])

    def test_read_pre_maet_returns_a_pre_maet(self, tmp_path):
        p, w, sp = _pm()
        pm = mpt.pack_pre_maet(p, w, sp)
        f = tmp_path / "pm.csv"
        mpt.write_pre_maet(str(f), pm)
        back = mpt.read_pre_maet(str(f))
        assert sorted(back) == ["p_attr", "specs", "w_attr"]
        assert np.allclose(back["p_attr"][0], p[0])


class TestBoundary:
    """The pre-MAET at the boundary: build, eval, and cosine."""

    KW = dict(sigma=[0.5, 0.25], per=[False, False], period=[0.0, 0.0],
              verbose=False)

    def test_build_maet_takes_a_pre_maet(self):
        p, w, sp = _pm()
        d1 = mpt.build_maet(mpt.pack_pre_maet(p, w, sp), **self.KW)
        d2 = mpt.build_maet(p, w, specs=sp, **self.KW)
        X = np.array([[60.0], [0.0]])
        assert np.allclose(mpt.eval_maet(d1, X, verbose=False),
                           mpt.eval_maet(d2, X, verbose=False))

    def test_build_rejects_weights_passed_twice(self):
        p, w, sp = _pm()
        with pytest.raises(TypeError, match="must not be passed again"):
            mpt.build_maet(mpt.pack_pre_maet(p, w, sp), w, **self.KW)

    def test_entropy_takes_a_pre_maet(self):
        p, w, sp = _pm()
        pm = mpt.pack_pre_maet(p, w, mpt.flat_specs(
            p, r=1, names=["pitch", "time"]))
        for a in range(2):
            pm["specs"][a].update(sigma=self.KW["sigma"][a], per=False,
                                  period=0.0)
        d = mpt.build_maet(pm, verbose=False)
        assert mpt.entropy_maet(pm, method="renyi2", verbose=False) \
            == pytest.approx(
                mpt.entropy_maet(d, method="renyi2", verbose=False))

    def test_eval_and_cosine_take_a_pre_maet(self):
        p, w, sp = _pm()
        pm = mpt.pack_pre_maet(p, w, mpt.flat_specs(
            p, r=1, names=["pitch", "time"]))
        for a in range(2):
            pm["specs"][a].update(sigma=self.KW["sigma"][a], per=False,
                                  period=0.0)
        d = mpt.build_maet(pm, verbose=False)
        X = np.array([[62.0], [1.0]])
        assert np.allclose(mpt.eval_maet(pm, X, verbose=False),
                           mpt.eval_maet(d, X, verbose=False))
        assert np.isclose(float(mpt.sim_maet(pm, pm, verbose=False)),
                          1.0)


class TestWindowed:
    """swept_similarity and swept_entropy take whole pre-MAETs."""

    def test_a_nested_query_may_hold_fewer_values_than_the_context(self):
        """The two sides of one comparison share the nesting they are read
        under, not their inner cardinality: a context whose events hold
        four values compares against a query whose events hold two."""
        ctx_vals = np.array([[60.0, 62.0, 64.0, 65.0],
                             [67.0, 69.0, 71.0, 72.0],
                             [60.0, 62.0, 64.0, 65.0]]).T
        q_vals = np.array([[60.0, 64.0], [67.0, 71.0]]).T
        axis = np.arange(ctx_vals.shape[1], dtype=float).reshape(1, -1)

        def nest(vals, r_outer):
            p = [vals]
            specs = mpt.flat_specs(p, r=1, rel=False, exch=True, names='p',
                                   sigma=0.5, per=False, period=0.0)
            return mpt.bind_events(p, None, r_outer, rel_outer=True,
                                   specs=specs)

        ctx = mpt.bind_events([ctx_vals, axis], None, [2, 1], rel_outer=True,
                              specs=mpt.flat_specs(
                                  [ctx_vals, axis], r=1, rel=False, exch=True,
                                  sigma=0.5, per=False, period=0.0))
        q_axis = np.arange(q_vals.shape[1], dtype=float).reshape(1, -1)
        qry = mpt.bind_events([q_vals, q_axis], None, [2, 1], rel_outer=True,
                              specs=mpt.flat_specs(
                                  [q_vals, q_axis], r=1, rel=False, exch=True,
                                  sigma=0.5, per=False, period=0.0))
        at = mpt.unpack_pre_maet(ctx)[0][1][0]
        out = np.asarray(mpt.swept_similarity(
            ctx, qry, sweep={1: at}, align={1: 'window'},
            window={1: {"shape": 1.0, "width": 1.0}}, drop=[1],
            normalize='oneSidedDenom', verbose=False)).ravel()
        assert out.shape == at.shape and np.all(np.isfinite(out))

        # position by position, against the same comparison made directly
        for k, _ in enumerate(at):
            one = mpt.select_pre_maet(ctx, attributes=[0], events=[k])
            direct = mpt.sim_maet(
                mpt.build_maet(one, verbose=False),
                mpt.build_maet(mpt.select_pre_maet(qry, attributes=[0]),
                               verbose=False),
                normalize='oneSidedDenom', verbose=False)
            assert out[k] == pytest.approx(float(direct), abs=1e-12)

    def test_the_two_sides_must_share_the_nesting(self):
        """Flat against nested is not one comparison."""
        vals = np.array([[60.0, 64.0], [67.0, 71.0]]).T
        flat = mpt.pack_pre_maet([vals], None, mpt.flat_specs(
            [vals], r=1, rel=False, exch=True, sigma=0.5))
        nested = mpt.bind_events([vals], None, 2, rel_outer=True,
                                 specs=mpt.flat_specs([vals], r=1, rel=False,
                                                      exch=True, sigma=0.5))
        with pytest.raises(ValueError, match="nests attribute|disagree on"):
            mpt.swept_similarity(nested, flat, sweep={0: [0.0]},
                                 align={0: 'both'},
                                 window={0: {"shape": 1.0, "width": 1.0}},
                                 verbose=False)

    @staticmethod
    def _pair():
        pC = [np.array([[60.0, 64, 67, 60, 64, 67]]),
              np.array([[0.0, 1, 2, 5, 6, 7]])]
        pQ = [np.array([[60.0, 64, 67]]), np.array([[0.0, 1, 2]])]
        sp = [{"name": "pitch", "r": 1, "rel": False, "exch": True,
               "sigma": 0.5, "per": True, "period": 12.0},
              {"name": "time", "r": 1, "rel": False, "exch": True,
               "sigma": 0.25, "per": False, "period": 0.0}]
        return pC, pQ, sp

    KW = dict(align={1: "both"},
              window={1: {"shape": "rect", "width": 2.5}}, verbose=False)

    def test_similarity_matches_the_positional_form(self):
        pC, pQ, sp = self._pair()
        ctr = np.arange(0.0, 7.01, 0.5)
        got = mpt.swept_similarity(mpt.pack_pre_maet(pC, specs=sp),
                                   mpt.pack_pre_maet(pQ, specs=sp),
                                   sweep={1: ctr}, **self.KW)
        ref = mpt.swept_similarity(
            pC, None, pQ, None, [0.5, 0.25], [1, 1], [False, False],
            [True, False], [12.0, 0.0], sweep={1: ctr}, **self.KW)
        np.testing.assert_allclose(got, ref, rtol=0, atol=0)

    def test_entropy_matches_the_positional_form(self):
        pC, _, sp = self._pair()
        ctr = np.arange(0.0, 7.01, 0.5)
        kw = dict(sweep={1: ctr}, window={1: {"shape": "gauss", "width": 2.0}},
                  method="renyi2", verbose=False)
        got = mpt.swept_entropy(mpt.pack_pre_maet(pC, specs=sp), **kw)
        ref = mpt.swept_entropy(
            pC, None, [0.5, 0.25], [1, 1], [False, False], [True, False],
            [12.0, 0.0], **kw)
        np.testing.assert_allclose(got, ref, rtol=0, atol=0)

    def test_selective_override_sweeps_one_attribute(self):
        pC, pQ, sp = self._pair()
        ctr = np.arange(0.0, 7.01, 0.5)
        base = mpt.swept_similarity(mpt.pack_pre_maet(pC, specs=sp),
                                    mpt.pack_pre_maet(pQ, specs=sp),
                                    sweep={1: ctr}, **self.KW)
        got = mpt.swept_similarity(mpt.pack_pre_maet(pC, specs=sp),
                                   mpt.pack_pre_maet(pQ, specs=sp),
                                   sweep={1: ctr}, sigma=[2.0, None],
                                   **self.KW)
        ref = mpt.swept_similarity(
            pC, None, pQ, None, [2.0, 0.25], [1, 1], [False, False],
            [True, False], [12.0, 0.0], sweep={1: ctr}, **self.KW)
        np.testing.assert_allclose(got, ref, rtol=0, atol=0)
        assert not np.allclose(got, base)      # the override took effect

    def test_specs_are_required(self):
        pC, pQ, _ = self._pair()
        with pytest.raises(ValueError, match="must carry its specs"):
            mpt.swept_similarity(mpt.pack_pre_maet(pC), mpt.pack_pre_maet(pQ),
                                 sweep={1: [0.0]}, **self.KW)

    def test_structural_geometry_must_agree(self):
        pC, pQ, sp = self._pair()
        sp2 = [dict(s) for s in sp]
        sp2[0]["rel"] = True
        with pytest.raises(ValueError, match="disagree on 'rel'"):
            mpt.swept_similarity(mpt.pack_pre_maet(pC, specs=sp),
                                 mpt.pack_pre_maet(pQ, specs=sp2),
                                 sweep={1: [0.0]}, **self.KW)


class TestLists:
    """A list of pre-MAETs stands wherever a list of densities does."""

    @staticmethod
    def _pm(vals):
        p = [np.array([vals], dtype=float), np.array([[0.0, 1.0, 2.0]])]
        return mpt.pack_pre_maet(p, specs=mpt.flat_specs(
            p, sigma=[0.5, 0.25], per=[True, False], period=[12.0, 0.0]))

    def test_list_versus_list(self):
        a, b = self._pm([60, 64, 67]), self._pm([62, 65, 69])
        got = mpt.sim_maet([a, a], [b, b], verbose=False)
        d_a = mpt.build_maet(a, verbose=False)
        d_b = mpt.build_maet(b, verbose=False)
        ref = mpt.sim_maet([d_a, d_a], [d_b, d_b], verbose=False)
        np.testing.assert_allclose(got, ref, rtol=0, atol=0)

    def test_scalar_versus_list(self):
        a, b = self._pm([60, 64, 67]), self._pm([62, 65, 69])
        got = mpt.sim_maet(a, [b, a], verbose=False)
        assert got[1] == pytest.approx(1.0)

    def test_eval_takes_a_list(self):
        a, b = self._pm([60, 64, 67]), self._pm([62, 65, 69])
        X = np.array([[60.0], [0.0]])
        got = mpt.eval_maet([a, b], X, verbose=False)
        ref = mpt.eval_maet(
            [mpt.build_maet(a, verbose=False),
             mpt.build_maet(b, verbose=False)], X, verbose=False)
        np.testing.assert_allclose(got, ref, rtol=0, atol=0)

    def test_entropy_takes_a_list(self):
        a, b = self._pm([60, 64, 67]), self._pm([62, 65, 69])
        kw = dict(method="shannon", n_points_per_dim=24,
                  x_min=[0.0, -1.0], x_max=[12.0, 3.0], verbose=False)
        got = mpt.entropy_maet([a, b], **kw)
        ref = mpt.entropy_maet(
            [mpt.build_maet(a, verbose=False),
             mpt.build_maet(b, verbose=False)], **kw)
        np.testing.assert_allclose(got, ref, rtol=0, atol=0)

    def test_translated_copies_stand_as_a_list(self):
        """Translated copies, one translate_attributes call each, stand as
        a list of pre-MAETs, built on the shared geometry."""
        a = self._pm([60, 64, 67])
        copies = [mpt.translate_attributes(a, [mu, None])
                  for mu in (0.0, 3.0, 7.0)]
        got = mpt.sim_maet(a, copies, verbose=False)
        ref = mpt.sim_maet(mpt.build_maet(a, verbose=False),
                           [mpt.build_maet(c, verbose=False) for c in copies],
                           verbose=False)
        np.testing.assert_allclose(got, ref, rtol=0, atol=0)
        assert got[0] == pytest.approx(1.0)     # the zero offset

    def test_translate_attributes_refuses_a_sweep(self):
        a = self._pm([60, 64, 67])
        with pytest.raises(ValueError, match="swept_similarity"):
            mpt.translate_attributes(a, [np.array([[0.0, 3.0, 7.0]]), None])


class TestGeometryOverrides:
    """r, rel, and exch override the specs, as sigma and its kin do."""

    KW = dict(sigma=[0.5, 0.25], per=[False, False], period=[0.0, 0.0],
              verbose=False)

    @staticmethod
    def _chords():
        # r > 1 needs at least r values per event, so the pitch attribute
        # carries a dyad at every event while the time attribute stays at
        # r = 1; the override is given per attribute to match.
        p = [np.array([[60.0, 62.0, 64.0], [64.0, 65.0, 67.0]]),
             np.array([[0.0, 1.0, 2.0]])]
        return p, None, mpt.flat_specs(p, r=1)

    def test_scalar_and_per_attribute_r(self):
        p, w, sp = self._chords()
        pm = mpt.pack_pre_maet(p, w, sp)
        assert list(mpt.build_maet(pm, r=[2, 1], **self.KW).r) == [2, 1]
        assert list(mpt.build_maet(pm, r=1, **self.KW).r) == [1, 1]

    def test_rel_and_exch(self):
        p, w, sp = self._chords()
        pm = mpt.pack_pre_maet(p, w, sp)
        d = mpt.build_maet(pm, r=[2, 1], rel=[True, False],
                               exch=[False, True], **self.KW)
        assert list(d.rel) == [True, False]
        assert list(d.exch) == [False, True]

    def test_a_sweep_is_one_call_per_value(self):
        p, w, sp = _pm()
        pm = mpt.pack_pre_maet(p, w, sp)
        vals = [mpt.build_maet(pm, sigma=[s, 0.25],
                                   per=[False, False],
                                   period=[0.0, 0.0], verbose=False)
                for s in (0.25, 0.5, 1.0)]
        assert [v.sigma[0] for v in vals] == [0.25, 0.5, 1.0]
        assert pm["specs"][0].get("sigma") is None   # the pre-MAET is intact

    def test_nested_geometry_is_not_overridable(self):
        p, w, sp = _pm()
        pmb = mpt.bind_events(mpt.pack_pre_maet(p, w, sp), [2, 2])
        with pytest.raises(ValueError, match="nested attribute"):
            mpt.build_maet(pmb, r=2, **self.KW)

    def test_overrides_need_specs(self):
        p, w, _ = _pm()
        with pytest.raises(ValueError, match="only valid alongside specs"):
            mpt.build_maet(p, w, [0.5, 0.25], [1, 1], [False, False],
                               [False, False], [0.0, 0.0], r=2,
                               verbose=False)

    def test_wrong_length_override_errors(self):
        p, w, sp = _pm()
        with pytest.raises(ValueError, match="length A"):
            mpt.build_maet(mpt.pack_pre_maet(p, w, sp), r=[1, 1, 1], **self.KW)


def _same(a, b):
    pa, wa, sa = mpt.unpack_pre_maet(a)
    pb, wb, sb = mpt.unpack_pre_maet(b)
    if len(pa) != len(pb):
        return False
    for x, y in zip(pa, pb):
        if not np.allclose(np.asarray(x), np.asarray(y), equal_nan=True):
            return False
    for x, y in zip(wa or [], wb or []):
        if not np.allclose(np.asarray(x, dtype=float),
                           np.asarray(y, dtype=float), equal_nan=True):
            return False
    for x, y in zip(sa or [], sb or []):
        if set(x) != set(y):
            return False
        for k in x:
            if list(np.ravel(x[k])) != list(np.ravel(y[k])):
                return False
    return True
