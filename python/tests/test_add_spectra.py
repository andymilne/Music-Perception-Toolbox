"""Tests for add_spectra spectrum-generation tests.

Mirror of MATLAB tests/test_add_spectra.m.
"""
import numpy as np
import pytest
import warnings

import mpt
from mpt._utils import position_variance


class TestAddSpectra:
    def test_harmonic_count(self):
        p, w = mpt.add_spectra(np.array([0, 400, 700]), None, "harmonic", 8, "powerlaw", 1)
        assert len(p) == 24  # 3 pitches x 8 harmonics

    def test_harmonic_fundamental(self):
        p, w = mpt.add_spectra(np.array([0.0]), np.array([1.0]), "harmonic", 4, "powerlaw", 0)
        # With rho=0 (flat), all weights = 1
        np.testing.assert_allclose(w, 1.0)
        # Offsets: 1200*log2(1), 1200*log2(2), 1200*log2(3), 1200*log2(4)
        expected = 1200 * np.log2([1, 2, 3, 4])
        np.testing.assert_allclose(p, expected, atol=1e-10)

    def test_stretched(self):
        p, w = mpt.add_spectra(np.array([0.0]), np.array([1.0]), "stretched", 3, 1.02, "powerlaw", 1)
        assert len(p) == 3
        # beta=1.02 should give slightly wider spacing than harmonic
        p_harm, _ = mpt.add_spectra(np.array([0.0]), np.array([1.0]), "harmonic", 3, "powerlaw", 1)
        assert p[2] > p_harm[2]

    def test_stiff(self):
        p, _ = mpt.add_spectra(np.array([0.0]), np.array([1.0]), "stiff", 4, 0.0003, "powerlaw", 1)
        # With B > 0, higher partials should be sharper than harmonic
        p_harm, _ = mpt.add_spectra(np.array([0.0]), np.array([1.0]), "harmonic", 4, "powerlaw", 1)
        assert p[3] > p_harm[3]

    def test_custom(self):
        p, w = mpt.add_spectra(np.array([0, 700]), None, "custom", [0, 1200], [1, 0.5])
        np.testing.assert_allclose(p, [0, 1200, 700, 1900])
        np.testing.assert_allclose(w, [1, 0.5, 1, 0.5])

    def test_geometric_weights(self):
        _, w = mpt.add_spectra(np.array([0.0]), np.array([1.0]), "harmonic", 4, "geometric", 0.5)
        np.testing.assert_allclose(w, [1, 0.5, 0.25, 0.125])

    def test_freqlinear_alpha_zero_equals_harmonic(self):
        """At alpha=0, ratio(n) = n/(0+1) = n, identical to the harmonic series."""
        p_lin, w_lin = mpt.add_spectra(
            np.array([0.0]), np.array([1.0]),
            "freqlinear", 4, 0.0, "powerlaw", 0,
        )
        p_har, w_har = mpt.add_spectra(
            np.array([0.0]), np.array([1.0]),
            "harmonic", 4, "powerlaw", 0,
        )
        np.testing.assert_allclose(p_lin, p_har, atol=1e-10)
        np.testing.assert_allclose(w_lin, w_har, atol=1e-10)

    def test_freqlinear_alpha_one_partial_ratios(self):
        """At alpha=1, ratio(n) = (1+n)/2, giving partials at 1, 1.5, 2, 2.5, ...
        which is 1200 * log2 of those ratios in cents."""
        p, _ = mpt.add_spectra(
            np.array([0.0]), np.array([1.0]),
            "freqlinear", 4, 1.0, "powerlaw", 0,
        )
        expected = 1200 * np.log2(np.array([1.0, 1.5, 2.0, 2.5]))
        np.testing.assert_allclose(p, expected, atol=1e-10)

    def test_freqlinear_alpha_le_minus_one_errors(self):
        """alpha <= -1 makes ratio(n) non-positive for some n; must raise."""
        with pytest.raises(ValueError, match="alpha"):
            mpt.add_spectra(
                np.array([0.0]), np.array([1.0]),
                "freqlinear", 4, -1.0, "powerlaw", 0,
            )


class TestAddSpectraPreMaet:
    """The pre-MAET form: one attribute of every event at once."""

    SPEC = ("harmonic", 3, "powerlaw", 1.0)

    def _pm(self, values, weights=None, **spec):
        base = dict(name="pitch", sigma=20.0, r=1, exch=True, rel=False,
                    is_per=False, period=0.0)
        base.update(spec)
        return mpt.pack_pre_maet([np.asarray(values, dtype=float)],
                                 None if weights is None
                                 else [np.asarray(weights, dtype=float)],
                                 [base])

    def test_it_expands_every_event_as_the_single_multiset_form_does(self):
        """Event by event, the pre-MAET form gives what the primitive
        gives: K becomes K * P, N and the spec stand."""
        values = np.array([[0.0, 100.0], [1200.0, 1300.0]])
        out = mpt.add_spectra(self._pm(values), *self.SPEC,
                              attribute="pitch")
        p_attr, w_attr, specs = mpt.unpack_pre_maet(out)
        assert p_attr[0].shape == (6, 2)
        for n in range(2):
            p, w = mpt.add_spectra(values[:, n], None, *self.SPEC)
            np.testing.assert_allclose(p_attr[0][:, n], p)
            np.testing.assert_allclose(w_attr[0][:, n], w)
        assert specs[0]["r"] == 1 and specs[0]["sigma"] == 20.0

    def test_padded_slots_stay_padded(self):
        """A missing value has no spectrum, so its partials are missing
        too and weigh nothing."""
        out = mpt.add_spectra(
            self._pm([[0.0, 100.0], [1200.0, np.nan]],
                     [[1.0, 1.0], [1.0, 0.0]]),
            "harmonic", 2, "powerlaw", 1.0, attribute=0)
        p_attr, w_attr, _ = mpt.unpack_pre_maet(out)
        assert np.isnan(p_attr[0][2:, 1]).all()
        np.testing.assert_allclose(w_attr[0][2:, 1], [0.0, 0.0])
        assert np.isfinite(p_attr[0][:, 0]).all()

    def test_a_weightless_pre_maet_comes_back_weighted(self):
        """Partials of one value differ in weight, so weights become
        necessary and every attribute gets them."""
        pm = mpt.pack_pre_maet([np.array([[0.0]]), np.array([[7.0]])], None,
                               [dict(name="a", sigma=1.0, r=1, exch=True,
                                     rel=False, is_per=False, period=0.0),
                                dict(name="b", sigma=1.0, r=1, exch=True,
                                     rel=False, is_per=False, period=0.0)])
        assert mpt.unpack_pre_maet(pm)[1] is None
        out = mpt.add_spectra(pm, *self.SPEC, attribute="a")
        w_attr = mpt.unpack_pre_maet(out)[1]
        assert w_attr is not None and len(w_attr) == 2
        np.testing.assert_allclose(w_attr[1], [[1.0]])

    def test_an_ordered_attribute_is_refused(self):
        """Where the positions carry meaning, expanding them would make a
        tuple take one value's partials rather than one value per
        position."""
        with pytest.raises(ValueError, match="read in order"):
            mpt.add_spectra(self._pm([[0.0], [700.0]], r=2, exch=False),
                            *self.SPEC, attribute=0)

    def test_it_asks_for_the_mode_and_the_attribute(self):
        pm = self._pm([[0.0]])
        with pytest.raises(ValueError, match="needs a mode"):
            mpt.add_spectra(pm)
        with pytest.raises(ValueError, match="needs the attribute"):
            mpt.add_spectra(pm, *self.SPEC)
        with pytest.raises(TypeError, match="single multiset"):
            mpt.add_spectra(np.array([0.0]), None, *self.SPEC, attribute=0)

    def test_the_positional_arguments_sit_one_slot_early(self):
        """Given a pre-MAET the mode takes the slot the weights take
        otherwise, and the rest follow it."""
        pm = self._pm([[0.0, 100.0]])
        grown = mpt.unpack_pre_maet(mpt.add_spectra(pm, *self.SPEC,
                                                   attribute=0))[0][0]
        np.testing.assert_allclose(grown[:3, 0],
                                   [0.0, 1200.0, 1200 * np.log2(3)])
        # A named mode reaches the mode parser rather than being read as
        # weights, so its own arguments are still asked for.
        with pytest.raises(ValueError, match="'harmonic', N"):
            mpt.add_spectra(pm, mode="harmonic", attribute=0)


# ===================================================================
#  Circular measures
# ===================================================================
