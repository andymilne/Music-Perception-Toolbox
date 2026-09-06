"""Tests for ``transform_attributes`` (named transforms, scale
conversions, user functions, the sign attribute, and the domain
refusals). Mirror of MATLAB tests/test_transform_attributes.m; the
reference values in ``TestCrossLanguageReference`` are shared with it.
"""
import math

import numpy as np
import pytest

import mpt
from mpt import transform_attributes


# ===================================================================
#  Scale conversions (formerly convert_pitch)
# ===================================================================

class TestScaleConversions:
    def test_hz_to_midi(self):
        assert transform_attributes(440, None, ("hz", "midi")) == pytest.approx(69)

    def test_midi_to_hz(self):
        assert transform_attributes(60, None, ("midi", "hz")) == pytest.approx(261.6256, rel=1e-4)

    def test_hz_to_cents(self):
        assert transform_attributes(440, None, ("hz", "cents")) == pytest.approx(6900)

    def test_octave_is_midi_over_12(self):
        assert transform_attributes(440, None, ("hz", "octave")) == pytest.approx(69 / 12)
        assert transform_attributes(5.75, None, ("octave", "midi")) == pytest.approx(69)

    def test_identity(self):
        arr = np.array([100.0, 200.0, 300.0])
        np.testing.assert_array_equal(transform_attributes(arr, None, ("hz", "hz")), arr)

    @pytest.mark.parametrize("scale", ["midi", "cents", "octave", "mel", "bark", "erb", "greenwood"])
    def test_pitch_roundtrip(self, scale):
        val = 440.0
        rt = transform_attributes(transform_attributes(val, None, ("hz", scale)), None, (scale, "hz"))
        assert rt == pytest.approx(val, rel=1e-8)

    def test_vectorised_keeps_shape(self):
        out = transform_attributes([261.63, 440, 880], None, ("hz", "midi"))
        np.testing.assert_allclose(out, [60, 69, 81], atol=0.01)
        out2 = transform_attributes(np.array([[261.63, 440], [880, 220]]), None, ("hz", "midi"))
        assert out2.shape == (2, 2)

    def test_unknown_scale(self):
        with pytest.raises(ValueError, match="Unknown scale"):
            transform_attributes(440, None, ("hz", "bogus"))

    def test_scale_pair_takes_no_parameters(self):
        with pytest.raises(ValueError, match="no parameters"):
            transform_attributes(440, None, {"from": "hz", "to": "midi", "ref": 2})

    def test_list_form_of_a_pair_gets_a_hint(self):
        with pytest.raises(ValueError, match="scale name"):
            transform_attributes([np.array([440.0]), np.array([1.0])], None, ["hz", "cents"])


# ===================================================================
#  Named transforms
# ===================================================================

class TestNamedTransforms:
    def test_log_default_base_and_offset(self):
        assert transform_attributes(math.e, None, "log") == pytest.approx(1.0)
        assert transform_attributes(8.0, None, ("log", {"base": 2})) == pytest.approx(3.0)
        assert transform_attributes(1000.0, None, {"name": "log", "base": 10}) == pytest.approx(3.0)
        # log(x + offset): the offset admits zeros explicitly
        assert transform_attributes(0.0, None, ("log", {"offset": 1})) == 0.0
        assert transform_attributes(7.0, None, ("log", {"base": 2, "offset": 1})) == pytest.approx(3.0)

    def test_power_and_affine(self):
        assert transform_attributes(9.0, None, ("power", {"exponent": 0.5})) == pytest.approx(3.0)
        assert transform_attributes(2.0, None, ("power", {"exponent": 3})) == pytest.approx(8.0)
        assert transform_attributes(100.0, None, ("affine", {"scale": 0.01, "offset": -1})) == pytest.approx(0.0)
        assert transform_attributes(5.0, None, "affine") == 5.0

    def test_required_and_unknown_parameters(self):
        with pytest.raises(ValueError, match="requires"):
            transform_attributes(2.0, None, "power")
        with pytest.raises(ValueError, match="does not take"):
            transform_attributes(2.0, None, ("log", {"period": 3}))
        with pytest.raises(ValueError, match="unknown transform"):
            transform_attributes(2.0, None, "cube")
        with pytest.raises(ValueError, match="unknown transform"):
            transform_attributes(2.0, None, "log1p")

    def test_names_are_case_insensitive(self):
        assert transform_attributes(8.0, None, ("LOG", {"Base": 2})) == pytest.approx(3.0)
        assert transform_attributes(440, None, ("Hz", "MIDI")) == pytest.approx(69)


# ===================================================================
#  Domain refusals
# ===================================================================

class TestDomain:
    def test_zero_under_log_is_refused_with_remedies(self):
        with pytest.raises(ValueError) as ei:
            transform_attributes([np.array([0.5, 0.0, 0.25])], None, ["log"])
        msg = str(ei.value)
        assert "undefined at zero" in msg and "event 1" in msg
        assert "bind_events" in msg and "offset" in msg

    def test_log_offset_domain_is_x_plus_offset(self):
        # -0.5 + 1 > 0 is admitted only with the sign attribute (the
        # negative itself is refused first); 0 + 1 > 0 is fine
        out, _, _ = transform_attributes([np.array([0.0, 3.0])], None,
                                         [("log", {"offset": 1, "base": 2})])
        np.testing.assert_allclose(out[0], [[0.0, 2.0]])
        with pytest.raises(ValueError, match=r"x \+ offset <= 0"):
            transform_attributes([np.array([0.5, 1.0])], None,
                                 [("log", {"offset": -0.5})])

    def test_negative_under_log_recommends_sign(self):
        with pytest.raises(ValueError, match="sign=True"):
            transform_attributes([np.array([1.0, -2.0])], None, ["log"])

    def test_negative_under_a_non_magnitude_transform_has_no_sign_hint(self):
        with pytest.raises(ValueError) as ei:
            transform_attributes([np.array([-1.0])], None, [("hz", "midi")])
        assert "sign=True" not in str(ei.value)

    def test_zero_hz_is_refused(self):
        with pytest.raises(ValueError, match="undefined at zero"):
            transform_attributes([np.array([0.0])], None, [("hz", "cents")])

    def test_non_finite_input_is_refused(self):
        with pytest.raises(ValueError, match="finite"):
            transform_attributes([np.array([1.0, np.inf])], None, ["affine"])

    def test_callable_must_return_finite_same_shape(self):
        with pytest.raises(ValueError, match="non-finite"):
            transform_attributes([np.array([1.0, 2.0])], None, [lambda x: np.log(x - 1.0)])
        with pytest.raises(ValueError, match="shape"):
            transform_attributes([np.array([1.0, 2.0])], None, [lambda x: x.ravel()[:1]])

    def test_message_names_the_attribute(self):
        specs = mpt.flat_specs([np.zeros(2), np.zeros(2)], name=["pitch", "ioi"])
        with pytest.raises(ValueError, match=r"attribute 1 \('ioi'\)"):
            transform_attributes([np.array([1.0, 2.0]), np.array([0.5, 0.0])], None,
                                 [None, "log"], specs=specs)


# ===================================================================
#  List form: threading, callables, sign attribute
# ===================================================================

class TestListForm:
    def test_none_leaves_attribute_and_broadcast_applies_to_all(self):
        p = [np.array([1.0, 2.0]), np.array([4.0, 9.0])]
        out, w, specs = transform_attributes(p, None, [None, ("power", {"exponent": 0.5})])
        np.testing.assert_array_equal(out[0], [[1.0, 2.0]])
        np.testing.assert_allclose(out[1], [[2.0, 3.0]])
        assert w is None and len(specs) == 2
        out2, _, _ = transform_attributes(p, None, ("power", {"exponent": 0.5}))
        np.testing.assert_allclose(out2[0], [[1.0, math.sqrt(2)]])

    def test_weights_and_specs_pass_through(self):
        p = [np.array([1.0, 2.0])]
        specs = mpt.flat_specs(p, r=2, name="x")
        out, w, s = transform_attributes(p, [np.array([0.5, 0.5])], ["log"], specs=specs)
        assert s[0] is specs[0]
        np.testing.assert_array_equal(w[0], [0.5, 0.5])

    def test_callable_and_nested_values(self):
        p = [np.array([[1.0, 4.0], [9.0, 16.0]])]          # K_total = 2
        out, _, _ = transform_attributes(p, None, [np.sqrt])
        np.testing.assert_allclose(out[0], [[1.0, 2.0], [3.0, 4.0]])

    def test_sign_attribute_is_inserted_after_its_source(self):
        p = [np.array([2.0, -3.0, 0.0]), np.array([1.0, 1.0, 1.0])]
        specs = mpt.flat_specs(p, name=["ivl", "t"])
        out, w, s = transform_attributes(p, [1.0, 2.0], [("log", {"offset": 1}), None],
                                         specs=specs, sign=[True, False])
        assert len(out) == 3 and len(w) == 3 and len(s) == 3
        np.testing.assert_allclose(out[0], [[math.log(3), math.log(4), 0.0]])
        np.testing.assert_array_equal(out[1], [[1.0, -1.0, 0.0]])
        np.testing.assert_array_equal(out[2], [[1.0, 1.0, 1.0]])
        assert s[1] == {"r": 1, "rel": False, "sym": True, "name": "ivl_sign"}
        assert w == [1.0, 1.0, 2.0]

    def test_sign_on_a_nested_spec_clears_rel(self):
        p = [np.array([[1.0, -2.0], [-3.0, 4.0]])]
        spec = {"tags": [0, 1], "r": [1, 2], "sym": [True, True], "rel": [0, 1]}
        out, _, s = transform_attributes(p, None, [("power", {"exponent": 0.5})],
                                         specs=[spec], sign=True)
        np.testing.assert_array_equal(out[1], [[1.0, -1.0], [-1.0, 1.0]])
        assert s[1]["rel"] == [False, False] and s[1]["tags"] == [0, 1]
        assert s[1]["name"] == "sign"

    def test_sign_with_power(self):
        out, _, _ = transform_attributes([np.array([-4.0, 9.0])], None,
                                         [("power", {"exponent": 0.5})], sign=True)
        np.testing.assert_allclose(out[0], [[2.0, 3.0]])
        np.testing.assert_array_equal(out[1], [[-1.0, 1.0]])

    def test_sign_requires_a_magnitude_transform(self):
        with pytest.raises(ValueError, match="magnitude"):
            transform_attributes([np.array([1.0])], None, ["affine"], sign=True)
        with pytest.raises(ValueError, match="magnitude"):
            transform_attributes([np.array([1.0])], None, [None], sign=True)
        with pytest.raises(ValueError, match="magnitude"):
            transform_attributes([np.array([1.0])], None, [("hz", "midi")], sign=True)

    def test_zero_under_log_still_refused_with_sign(self):
        with pytest.raises(ValueError, match="undefined at zero"):
            transform_attributes([np.array([1.0, 0.0])], None, ["log"], sign=True)

    def test_bare_form_rejects_list_arguments(self):
        with pytest.raises(ValueError, match="bare-array"):
            transform_attributes(np.array([1.0]), [1.0], "log")
        with pytest.raises(ValueError, match="bare-array"):
            transform_attributes(np.array([1.0]), None, ["log"])

    def test_length_mismatches(self):
        p = [np.array([1.0]), np.array([1.0])]
        with pytest.raises(ValueError, match="length-A"):
            transform_attributes(p, None, ["log"])
        with pytest.raises(ValueError, match="sign must be"):
            transform_attributes(p, None, "log", sign=[True])
        with pytest.raises(ValueError, match="event count"):
            transform_attributes([np.array([1.0]), np.array([1.0, 2.0])], None, "log")


# ===================================================================
#  Composition with differencing and build
# ===================================================================

class TestComposition:
    def test_log_then_difference_gives_log_ratios(self):
        ioi = np.array([0.25, 0.5, 0.5, 1.0])
        p, w, specs = transform_attributes([ioi], None, [("log", {"base": 2})])
        d, _, _ = mpt.difference_events(p, w, 1, specs=specs)
        np.testing.assert_allclose(d[0], [[1.0, 0.0, 1.0]])

    def test_difference_then_log_offset_with_sign_feeds_build(self):
        pitch = np.array([60.0, 64.0, 62.0, 62.0, 67.0])
        d, w, specs = mpt.difference_events([pitch.reshape(1, -1)], None, 1)
        p, w, specs = transform_attributes(d, w, [("log", {"offset": 1})],
                                           specs=specs, sign=True)
        dens = mpt.build_exp_tens(p, w, specs=specs, sigma=[0.2, 0.3],
                                  is_per=[False, False], period=[0, 0],
                                  verbose=False)
        assert len(dens.sigma) == 2
        s = mpt.cos_sim_exp_tens(dens, dens, verbose=False)
        assert s == pytest.approx(1.0)


# ===================================================================
#  Cross-language reference values (shared with the MATLAB test)
# ===================================================================

class TestCrossLanguageReference:
    """The same fixed inputs in both languages; both suites assert these
    numbers to 1e-9."""
    X = np.array([0.25, 0.5, 1.0, 2.0, 4.0])

    def test_reference_table(self):
        ref = [
            (("log", {"base": 2}),         [-2.0, -1.0, 0.0, 1.0, 2.0]),
            (("log", {"offset": 1}),       [0.22314355131421, 0.405465108108164,
                                            0.693147180559945, 1.09861228866811,
                                            1.6094379124341]),
            (("power", {"exponent": 0.5}), [0.5, 0.707106781186548, 1.0,
                                            1.4142135623731, 2.0]),
            (("power", {"exponent": 1.5}), [0.125, 0.353553390593274, 1.0,
                                            2.82842712474619, 8.0]),
            (("affine", {"scale": 2, "offset": -1}), [-0.5, 0.0, 1.0, 3.0, 7.0]),
            (("hz", "erb"),                [0.0101480454786181, 0.020285022354027,
                                            0.040525866692578, 0.0808757881114107,
                                            0.16105386008722]),
            (("hz", "octave"),             [-5.03135971352466, -4.03135971352466,
                                            -3.03135971352466, -2.03135971352466,
                                            -1.03135971352466]),
        ]
        for tr, expect in ref:
            got = transform_attributes(self.X, None, tr)
            np.testing.assert_allclose(got, expect, rtol=1e-9, atol=1e-9,
                                       err_msg=str(tr))
