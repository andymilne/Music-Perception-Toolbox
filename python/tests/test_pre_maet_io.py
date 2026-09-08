"""test_pre_maet_io.py -- kernel geometry in specs, and CSV I/O.

Two related things are tested here. First, that a spec may carry the
attribute's sigma, is_per and period --- the parameters Milne (2026,
Def. 2.6) counts as part of the pre-MAET --- that an explicit keyword
overrides them, and that a value missing from both places is refused by
name. Second, that a pre-MAET survives a round trip through CSV
unchanged, the writer and the reader sharing one cell grammar.
"""
import numpy as np
import pytest

import mpt


def _p():
    return [np.array([[60.0, 62.0, 64.0]]), np.array([[0.0, 1.0, 2.0]])]


def _specs():
    return [{"name": "pitch", "r": 1, "rel": False, "sym": True,
             "sigma": 0.5, "is_per": True, "period": 12.0},
            {"name": "time", "r": 1, "rel": False, "sym": True,
             "sigma": 0.25, "is_per": False, "period": 0.0}]


class TestResolution:

    def test_specs_alone_suffice(self):
        d = mpt.build_exp_tens(_p(), None, specs=_specs(), verbose=False)
        np.testing.assert_allclose(np.atleast_1d(d.sigma), [0.5, 0.25])
        assert list(np.atleast_1d(d.is_per)) == [True, False]

    def test_keyword_overrides_silently(self):
        """A sweep supplies sigma per call while the specs hold a
        baseline, so the two disagreeing is the ordinary idiom."""
        d = mpt.build_exp_tens(_p(), None, specs=_specs(), sigma=[9.0, 9.0],
                               verbose=False)
        np.testing.assert_allclose(np.atleast_1d(d.sigma), [9.0, 9.0])

    def test_missing_is_refused_by_name(self):
        sp = _specs()
        del sp[1]["sigma"]
        with pytest.raises(ValueError, match=r"No sigma for attribute 'time'"):
            mpt.build_exp_tens(_p(), None, specs=sp, verbose=False)

    def test_na_is_refused_and_says_why(self):
        sp = _specs()
        sp[0]["sigma"] = np.nan
        with pytest.raises(ValueError,
                           match=r"sigma for attribute 'pitch' is NA"):
            mpt.build_exp_tens(_p(), None, specs=sp, verbose=False)

    def test_na_is_recoverable_at_the_call(self):
        sp = _specs()
        sp[0]["sigma"] = np.nan
        d = mpt.build_exp_tens(_p(), None, specs=sp, sigma=[0.5, 0.25],
                               verbose=False)
        np.testing.assert_allclose(np.atleast_1d(d.sigma), [0.5, 0.25])

    def test_period_defaults_to_zero(self):
        """Only sigma and is_per are compulsory: a period is inert on a
        non-periodic attribute."""
        sp = _specs()
        del sp[1]["period"]
        d = mpt.build_exp_tens(_p(), None, specs=sp, verbose=False)
        assert float(np.atleast_1d(d.period)[1]) == 0.0

    def test_camel_case_alias_is_read(self):
        """A spec written for either language reads in both."""
        sp = _specs()
        sp[0]["isPer"] = sp[0].pop("is_per")
        d = mpt.build_exp_tens(_p(), None, specs=sp, verbose=False)
        assert bool(np.atleast_1d(d.is_per)[0])

    def test_covariance_in_a_spec_is_honoured(self):
        """The matrix path is chosen after resolution, not from the
        keyword: a covariance may live in the spec."""
        C = mpt.interval_kernel_cov(3, sd_position=0.2, sd_interval=0.3)
        pb, wb, sb = mpt.unpack_pre_maet(mpt.bind_events([np.array([[1.0, 2, 3, 4]])], None, [3]))
        by_kw = mpt.build_exp_tens(pb, wb, specs=sb, sigma=[C],
                                   is_per=[False], period=[0.0],
                                   verbose=False)
        sb[0].update(sigma=C, is_per=False, period=0.0)
        in_spec = mpt.build_exp_tens(pb, wb, specs=sb, verbose=False)
        assert in_spec.kernel_cov[0] is not None
        np.testing.assert_allclose(in_spec.kernel_cov[0], by_kw.kernel_cov[0])


class TestOperatorRules:

    def test_difference_scales_sigma_by_root_binomial(self):
        sp = [{"name": "p", "r": 1, "rel": False, "sym": True,
               "sigma": 10.0, "is_per": False, "period": 0.0}]
        p = [np.array([[1.0, 2, 3, 4]])]
        _, _, s1 = mpt.unpack_pre_maet(mpt.difference_events(p, None, 1, specs=sp))
        _, _, s2 = mpt.unpack_pre_maet(mpt.difference_events(p, None, 2, specs=sp))
        assert s1[0]["sigma"] == pytest.approx(10.0 * np.sqrt(2))
        assert s2[0]["sigma"] == pytest.approx(10.0 * np.sqrt(6))

    def test_difference_scales_a_covariance_by_the_variance_factor(self):
        """A covariance is in squared units, so it takes C(2k, k) where a
        width takes its root."""
        C = np.diag([0.04, 0.09, 0.16])
        # Differencing needs K = 1 per event, so the covariance rides on
        # a single-valued attribute across four events.
        sp = [{"name": "t", "r": 1, "rel": False, "sym": True,
               "sigma": C, "is_per": False, "period": 0.0}]
        _, _, s = mpt.unpack_pre_maet(mpt.difference_events([np.array([[1.0, 2, 3, 4]])],
                                        None, 1, specs=sp))
        np.testing.assert_allclose(np.asarray(s[0]["sigma"]), C * 2.0)

    def test_bind_translate_weight_carry_geometry_through(self):
        sp = _specs()[:1]
        p = [np.array([[60.0, 62, 64, 65]])]
        for out in (mpt.bind_events(p, None, [2], specs=sp)["specs"],
                    mpt.translate_attributes(p, None, [5.0],
                                             specs=sp)["specs"],
                    mpt.weight_events(p, None, input_attr=0, target_attr=0,
                                      centre=62.0, sd=2.0, shape=0.0,
                                      drop_input_attr=False,
                                      specs=sp)["specs"]):
            assert out[0]["sigma"] == 0.5
            assert out[0]["period"] == 12.0

    @pytest.mark.parametrize("transform,gain", [
        (("midi", "cents"), 100.0),
        (("affine", {"scale": 3.0}), 3.0),
    ])
    def test_affine_maps_scale_sigma_and_period(self, transform, gain):
        sp = _specs()[:1]
        _, _, s = mpt.unpack_pre_maet(mpt.transform_attributes([np.array([[60.0, 62, 64]])],
                                           None, [transform], specs=sp))
        assert s[0]["sigma"] == pytest.approx(0.5 * gain)
        assert s[0]["period"] == pytest.approx(12.0 * gain)

    @pytest.mark.parametrize("transform", [
        ("midi", "hz"), "log", (lambda x: np.sqrt(x) + 1.0)])
    def test_non_linear_maps_leave_na(self, transform):
        """A non-linear map carries no width across.

        A width is still meaningful in the new coordinate --- after a log
        it expresses a ratio rather than a difference --- but the local
        scaling varies across the range, so no single value is the image
        of the old one, and NA marks the absence of a canonical choice.
        """
        sp = _specs()[:1]
        _, _, s = mpt.unpack_pre_maet(mpt.transform_attributes([np.array([[60.0, 62, 64]])],
                                           None, [transform], specs=sp))
        assert np.isnan(s[0]["sigma"])
        assert np.isnan(s[0]["period"])


class TestFromScore:

    def test_score_gives_periodicity_but_not_width(self):
        """A score states its attributes' periodicity and implies no
        kernel width, so sigma is left for the analyst."""
        import os
        src = os.path.join(os.path.dirname(__file__), "data",
                           "score_small.musicxml")
        _, _, specs = mpt.unpack_pre_maet(mpt.pre_maet_from_score(
            src, attributes=("pitch",), chords="separate"))
        assert specs[0]["is_per"] is False
        assert specs[0]["period"] == 0.0
        assert "sigma" not in specs[0]


FLAT = '''name,sigma,r,rel,per,P,sym,n = 1,n = 2,n = 3
pitch,0.5,2,0,1,12,1,"{60, 64, 67}","{62, 65, 69}","{60, 64, 67}"
onset,0.25,1,0,0,,1,0,1,2
'''

NESTED = '''name,sigma,r,rel,per,P,sym,n = 1,n = 2
pitch,0.15,"(1, 3)","(0, 1)",1,12,"(1, 0)","({60, 64, 67}, {62, 67, 71}, {60, 64, 67})","({62, 65, 69}, {55, 59, 62}, {60, 64, 67})"
metre,0.1,1,0,0,,1,1^(1),0.5^(0.5)
'''

COV = '''name,sigma,r,rel,per,P,sym,n = 1,n = 2
trigram,"cov(sd_position=0.2, sd_interval=0.3, sd_shift=0.5)",3,0,0,,0,"(60, 62, 64)","(62, 64, 65)"
'''


class TestCsvRoundTrip:

    @pytest.mark.parametrize("src", [FLAT, NESTED, COV])
    def test_round_trip_is_byte_exact(self, src):
        p, w, specs = mpt.unpack_pre_maet(mpt.read_pre_maet(src))
        assert mpt.write_pre_maet(None, p, w, specs) == src

    def test_ragged_events_pad_and_unpad(self):
        src = ('name,sigma,r,rel,per,P,sym,a,b,c\n'
               'pitch,0.5,1,0,0,,1,"{60, 64, 67}","{62, 65}",'
               '"{60, 64, 67, 71}"\n')
        p, w, specs = mpt.unpack_pre_maet(mpt.read_pre_maet(src))
        assert p[0].shape == (4, 3)
        assert np.isnan(p[0][3, 0]) and np.isnan(p[0][2, 1])
        assert mpt.write_pre_maet(None, p, w, specs,
                                  headings=["a", "b", "c"]) == src

    def test_tags_are_rebuilt_from_the_brackets(self):
        """A file never writes tags down: the bracket structure is the
        level structure."""
        p, w, specs = mpt.unpack_pre_maet(mpt.read_pre_maet(NESTED))
        assert "tags" in specs[0]
        assert np.asarray(specs[0]["tags"]).ravel().tolist() == \
            [0, 0, 0, 1, 1, 1, 2, 2, 2]

    def test_unit_weights_are_written_bare(self):
        """A bare value already means unit weight, so only a weight that
        carries information is written."""
        p, w, specs = mpt.unpack_pre_maet(mpt.read_pre_maet(NESTED))
        out = mpt.write_pre_maet(None, p, w, specs)
        assert "{60, 64, 67}" in out            # pitch, all weights 1
        assert "0.5^(0.5)" in out               # metre, weights vary

    def test_builds_from_the_file_with_nothing_supplied(self):
        p, w, specs = mpt.unpack_pre_maet(mpt.read_pre_maet(FLAT))
        d = mpt.build_exp_tens(p, w, specs=specs, verbose=False)
        assert d.dim == 3
        assert float(mpt.cos_sim_exp_tens(d, d, verbose=False)) == \
            pytest.approx(1.0)

    def test_either_covariance_spelling_is_read(self):
        """A file may be written by hand in a spreadsheet or by the
        MATLAB twin; the parameter is the same either way."""
        camel = COV.replace("sd_position", "sdPosition") \
                   .replace("sd_interval", "sdInterval") \
                   .replace("sd_shift", "sdShift")
        a = mpt.read_pre_maet(COV)["specs"][0]["sigma"]
        b = mpt.read_pre_maet(camel)["specs"][0]["sigma"]
        np.testing.assert_allclose(np.asarray(a), np.asarray(b))

    def test_covariance_outside_the_family_is_refused(self):
        C = np.array([[1.0, 0.9, 0.1], [0.9, 1.0, 0.2], [0.1, 0.2, 1.0]])
        p, w, specs = mpt.unpack_pre_maet(mpt.read_pre_maet(COV))
        specs[0]["sigma"] = C
        with pytest.raises(ValueError, match="not of that family"):
            mpt.write_pre_maet(None, p, w, specs)

    def test_na_survives_the_round_trip(self):
        src = ('name,sigma,r,rel,per,P,sym,n = 1\n'
               'pitch,NA,1,0,0,,1,60\n')
        p, w, specs = mpt.unpack_pre_maet(mpt.read_pre_maet(src))
        assert np.isnan(specs[0]["sigma"])
        assert mpt.write_pre_maet(None, p, w, specs) == src

    def test_bad_header_is_named(self):
        with pytest.raises(ValueError, match="The header must begin"):
            mpt.read_pre_maet("a,b,c\n1,2,3\n")

    def test_csv_elides_nothing(self):
        """A file records the pre-MAET; it does not display it."""
        p = [np.arange(20.0).reshape(1, 20)]
        sp = [{"name": "x", "r": 1, "rel": False, "sym": True,
               "sigma": 1.0, "is_per": False, "period": 0.0}]
        out = mpt.write_pre_maet(None, p, None, sp, max_events=3)
        assert "n = 20" in out and "..." not in out
