"""Tests for convert_pitch unit/round-trip/error tests.

Mirror of MATLAB tests/test_convert_pitch.m.
"""
import numpy as np
import pytest
import warnings

import mpt
from mpt._utils import position_variance


class TestConvertPitch:
    def test_hz_to_midi(self):
        assert mpt.convert_pitch(440, "hz", "midi") == pytest.approx(69)

    def test_midi_to_hz(self):
        assert mpt.convert_pitch(60, "midi", "hz") == pytest.approx(261.6256, rel=1e-4)

    def test_hz_to_cents(self):
        assert mpt.convert_pitch(440, "hz", "cents") == pytest.approx(6900)

    def test_identity(self):
        arr = np.array([100, 200, 300])
        np.testing.assert_array_equal(mpt.convert_pitch(arr, "hz", "hz"), arr)

    @pytest.mark.parametrize("scale", ["midi", "cents", "mel", "bark", "erb", "greenwood"])
    def test_roundtrip(self, scale):
        val = 440.0
        rt = mpt.convert_pitch(mpt.convert_pitch(val, "hz", scale), scale, "hz")
        assert rt == pytest.approx(val, rel=1e-8)

    def test_vectorised(self):
        out = mpt.convert_pitch([261.63, 440, 880], "hz", "midi")
        np.testing.assert_allclose(out, [60, 69, 81], atol=0.01)

    def test_unknown_scale(self):
        with pytest.raises(ValueError, match="Unknown"):
            mpt.convert_pitch(440, "hz", "bogus")


# ===================================================================
#  add_spectra
# ===================================================================
