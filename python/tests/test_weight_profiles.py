"""Weight profiles beyond the rectangle-Gaussian window family.

``weight_events`` accepts three kinds of profile: a numeric ``gamma``
(the fixed-variance convolution family, tested in ``test_maet.py``),
one of seven named profiles, and any callable of the centred
difference. The named profiles absorb the serial-position curves that
``seq_weights`` supplied before v3, so the values pinned here are the
ones that function produced: a rate of 1 over event numbers
1, ..., N decays by a factor of e per event.

The MATLAB twin is ``matlab/tests/test_weight_profiles.m``; the two
pin the same numbers.
"""
import numpy as np
import pytest

import mpt


IDX = np.array([[1.0, 2.0, 3.0, 4.0]])
PITCH = np.array([[60.0, 62.0, 64.0, 65.0]])


def _weights(shape, centre=None, **kw):
    """Profile applied to pitch through an event-number attribute."""
    pm = mpt.pack_pre_maet([PITCH, IDX])
    out = mpt.weight_events(pm, 1, 0, centre, shape,
                            drop_input_attr=True, **kw)
    return np.asarray(out["w_attr"][0]).ravel()


# --- the anchored serial-position profiles ---------------------------

def test_exponential_from_end_is_the_old_recency_curve():
    got = _weights("exponentialFromEnd")
    np.testing.assert_allclose(got, np.exp(-np.array([3.0, 2.0, 1.0, 0.0])))


def test_exponential_from_start_mirrors_it():
    np.testing.assert_allclose(_weights("exponentialFromStart"),
                               _weights("exponentialFromEnd")[::-1])


def test_decay_rate_and_sd_are_two_spellings_of_one_scale():
    np.testing.assert_allclose(_weights("exponentialFromEnd", decay_rate=0.5),
                               _weights("exponentialFromEnd", sd=2.0))


def test_u_shape_is_symmetric_and_alpha_selects_its_components():
    u = _weights("uShape")
    np.testing.assert_allclose(u, u[::-1])
    np.testing.assert_allclose(_weights("uShape", alpha=1.0),
                               _weights("exponentialFromStart"))
    np.testing.assert_allclose(_weights("uShape", alpha=0.0),
                               _weights("exponentialFromEnd"))


def test_u_asym_gives_its_two_components_their_own_rates():
    got = _weights("uAsym", decay_rate_start=0.2, decay_rate_end=0.8,
                   alpha=0.4)
    idx = IDX.ravel()
    expected = (0.4 * np.exp(-0.2 * (idx - idx[0]))
                + 0.6 * np.exp(-0.8 * (idx[-1] - idx)))
    np.testing.assert_allclose(got, expected)


def test_u_asym_rates_fall_back_to_decay_rate():
    np.testing.assert_allclose(_weights("uAsym", decay_rate=0.5),
                               _weights("uAsym", decay_rate_start=0.5,
                                        decay_rate_end=0.5))


# --- the centre-anchored profiles ------------------------------------

def test_exponential_before_matches_the_anchored_form_at_the_last_event():
    np.testing.assert_allclose(
        _weights("exponentialBefore", centre=4.0),
        _weights("exponentialFromEnd"))


def test_exponential_after_is_zero_below_its_centre():
    got = _weights("exponentialAfter", centre=2.0)
    assert got[0] == 0.0
    np.testing.assert_allclose(got[1:], np.exp(-np.array([0.0, 1.0, 2.0])))


def test_two_sided_exponential_holds_its_standard_deviation():
    # The Laplace standard deviation is tau * sqrt(2), so sd fixes the
    # variance as it does across the convolution family.
    got = _weights("exponential", centre=2.0, sd=1.0)
    d = IDX.ravel() - 2.0
    np.testing.assert_allclose(got, np.exp(-np.abs(d) * np.sqrt(2.0)))


# --- callables --------------------------------------------------------

def test_a_callable_supplies_any_profile():
    got = _weights(lambda d: np.exp(0.5 * d) * (d <= 0), centre=4.0)
    np.testing.assert_allclose(got, np.exp(0.5 * np.array([-3.0, -2.0, -1.0, 0.0])))


def test_a_callable_is_exempt_from_the_kernel_truncation():
    # A profile that grows with distance would be zeroed by the window
    # truncation; it survives, because an arbitrary profile need not decay.
    got = _weights(lambda d: 1.0 + np.abs(d), centre=1.0)
    np.testing.assert_allclose(got, 1.0 + np.abs(IDX.ravel() - 1.0))


# --- rejected combinations -------------------------------------------

def test_a_centre_is_refused_by_the_anchored_profiles():
    with pytest.raises(ValueError, match="anchors itself"):
        _weights("exponentialFromEnd", centre=4.0)


def test_width_is_refused_by_the_named_profiles():
    with pytest.raises(TypeError, match="support of the rectangle"):
        _weights("exponentialFromEnd", width=2.0)


def test_sd_and_decay_rate_together_are_refused():
    with pytest.raises(TypeError, match="two spellings"):
        _weights("exponentialFromEnd", sd=1.0, decay_rate=1.0)


def test_asymmetric_rates_are_refused_by_the_one_rate_profiles():
    with pytest.raises(TypeError, match="uAsym"):
        _weights("uShape", decay_rate_start=0.5)


def test_a_rate_is_refused_by_the_numeric_family():
    with pytest.raises(TypeError, match="rectangle-Gaussian"):
        _weights(0.0, centre=2.0, decay_rate=1.0)


def test_a_scale_is_refused_with_a_callable():
    with pytest.raises(TypeError, match="carries its own scale"):
        _weights(lambda d: np.ones_like(d), centre=2.0, sd=1.0)


def test_an_unknown_profile_name_is_refused():
    with pytest.raises(ValueError, match="Unknown profile"):
        _weights("exponentialFromMiddle")


def test_a_callable_must_return_one_non_negative_factor_per_event():
    with pytest.raises(ValueError, match="one factor per event"):
        _weights(lambda d: np.ones(2), centre=2.0)
    with pytest.raises(ValueError, match="non-negative"):
        _weights(lambda d: -np.ones_like(d), centre=2.0)


# --- composition ------------------------------------------------------

def test_the_driving_attribute_is_dropped_and_the_pitches_survive():
    pm = mpt.pack_pre_maet([PITCH, IDX])
    out = mpt.weight_events(pm, 1, 0, None, "exponentialFromEnd",
                            drop_input_attr=True)
    p_out, w_out, specs = mpt.unpack_pre_maet(out)
    assert len(p_out) == 1 and len(specs) == 1
    np.testing.assert_allclose(np.asarray(p_out[0]), PITCH)
    np.testing.assert_allclose(np.asarray(w_out[0]).ravel(),
                               np.exp(-np.array([3.0, 2.0, 1.0, 0.0])))


def test_an_existing_weight_is_multiplied_into_not_replaced():
    salience = np.array([[1.0, 0.6, 0.6, 1.3]])
    pm = mpt.pack_pre_maet([PITCH, IDX], [salience, None])
    out = mpt.weight_events(pm, 1, 0, None, "exponentialFromEnd",
                            drop_input_attr=True)
    np.testing.assert_allclose(
        np.asarray(out["w_attr"][0]).ravel(),
        salience.ravel() * np.exp(-np.array([3.0, 2.0, 1.0, 0.0])))
