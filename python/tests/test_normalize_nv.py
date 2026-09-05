"""Tests for the ``normalize`` NV exposed on ``cos_sim_exp_tens``.

Covers:
  * Default-mode behaviour matches the legacy formula on each function.
  * ``'oneSidedDenom'`` recovers magnitude when the first operand is
    a positive scalar multiple of the second.
  * British spelling alias and case-insensitive value matching.
  * Mutual-exclusion of ``normalize`` and ``normalise``.
  * Bad-value error.
  * Cross-language parity is left to the MATLAB suite; this file
    fixes the Python expected values to floating-point precision.
"""
import numpy as np
import pytest
import mpt


@pytest.fixture(autouse=True)
def _silence_hints():
    mpt.set_default(show_hints=False)
    yield


# ---------------------------------------------------------------------
# cos_sim_exp_tens
# ---------------------------------------------------------------------


def _make_sa_density(p, w=None, sigma=100.0, r=1, period=1200.0):
    return mpt.build_exp_tens(
        np.asarray(p, dtype=float),
        None if w is None else np.asarray(w, dtype=float),
        sigma, r, False, True, period,
        verbose=False,
    )


class TestCosSimNormalize:

    def test_default_is_cosine_at_r1(self):
        d = _make_sa_density([0., 200., 400., 700.])
        s_default = mpt.cos_sim_exp_tens(d, d, verbose=False)
        s_explicit = mpt.cos_sim_exp_tens(d, d, normalize='cosine',
                                          verbose=False)
        assert s_default == pytest.approx(1.0)
        assert s_explicit == pytest.approx(s_default)

    def test_self_match_one_in_both_modes(self):
        d = _make_sa_density([0., 200., 400., 700.])
        s_cosine = mpt.cos_sim_exp_tens(d, d, normalize='cosine',
                                        verbose=False)
        s_one = mpt.cos_sim_exp_tens(d, d, normalize='oneSidedDenom',
                                     verbose=False)
        assert s_cosine == pytest.approx(1.0)
        assert s_one == pytest.approx(1.0)

    def test_scalar_invariance_cosine_vs_magnitude_one_sided(self):
        # At r=1 the tensor scales linearly in the weights, so X = 3·Y
        # gives <X, Y> = 3·<Y, Y>; cosine returns 1, oneSidedDenom returns 3.
        d_unit = _make_sa_density([0., 200., 400., 700.])
        d_big = _make_sa_density([0., 200., 400., 700.],
                                  w=[3., 3., 3., 3.])
        s_cos = mpt.cos_sim_exp_tens(d_big, d_unit, normalize='cosine',
                                     verbose=False)
        s_one = mpt.cos_sim_exp_tens(d_big, d_unit, normalize='oneSidedDenom',
                                     verbose=False)
        assert s_cos == pytest.approx(1.0)
        assert s_one == pytest.approx(3.0)

    def test_british_alias(self):
        d = _make_sa_density([0., 200., 400., 700.])
        s_us = mpt.cos_sim_exp_tens(d, d, normalize='oneSidedDenom',
                                    verbose=False)
        s_gb = mpt.cos_sim_exp_tens(d, d, normalise='oneSidedDenom',
                                    verbose=False)
        assert s_us == pytest.approx(s_gb)

    def test_case_insensitive_value(self):
        d = _make_sa_density([0., 200., 400., 700.])
        s_low = mpt.cos_sim_exp_tens(d, d, normalize='onesideddenom',
                                     verbose=False)
        s_mix = mpt.cos_sim_exp_tens(d, d, normalize='OneSidedDenom',
                                     verbose=False)
        s_can = mpt.cos_sim_exp_tens(d, d, normalize='oneSidedDenom',
                                     verbose=False)
        assert s_low == pytest.approx(s_can)
        assert s_mix == pytest.approx(s_can)

    def test_mutual_exclusion_normalize_normalise(self):
        d = _make_sa_density([0., 200., 400., 700.])
        with pytest.raises(TypeError, match="not both"):
            mpt.cos_sim_exp_tens(d, d, normalize='cosine',
                                 normalise='cosine', verbose=False)

    def test_bad_value_raises(self):
        d = _make_sa_density([0., 200., 400., 700.])
        with pytest.raises(ValueError, match="normalize"):
            mpt.cos_sim_exp_tens(d, d, normalize='bogus', verbose=False)
