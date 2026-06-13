"""Tests for the ``normalize`` NV exposed on both ``cos_sim_exp_tens``
and ``windowed_tensor_similarity``.

Covers:
  * Default-mode behaviour matches the legacy formula on each function.
  * ``'oneSidedDenom'`` recovers magnitude when the first operand is
    a positive scalar multiple of the second.
  * British spelling alias and case-insensitive value matching.
  * Mutual-exclusion of ``normalize`` and ``normalise``.
  * Bad-value error.
  * On ``windowed_tensor_similarity``, ``'cosine'`` is bounded above by 1 at
    full self-match and accepts mix=0 and mix=1 but raises on
    intermediate mix.
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


# ---------------------------------------------------------------------
# windowed_tensor_similarity
# ---------------------------------------------------------------------


def _make_ma_density(p, sigma=100., r=1, period=1200.):
    return mpt.build_exp_tens(
        [np.asarray(p, dtype=float).reshape(1, -1)], None,
        [sigma], [r], 
        [False], [True], [period], verbose=False,
    )


class TestWindowedNormalize:

    def _setup(self):
        dC = _make_ma_density([0., 200., 400., 700.])
        dQ = _make_ma_density([0., 200., 400., 700.])
        spec = {'size': np.array([3.0]), 'mix': np.array([0.0])}
        offsets = np.array([[-200., -100., 0., 100., 200.]])
        return dC, dQ, spec, offsets

    def test_default_is_one_sided_denom(self):
        dC, dQ, spec, offsets = self._setup()
        p_default = mpt.windowed_tensor_similarity(dC, dQ, spec, offsets,
                                            verbose=False)
        p_explicit = mpt.windowed_tensor_similarity(dC, dQ, spec, offsets,
                                             normalize='oneSidedDenom',
                                             verbose=False)
        np.testing.assert_allclose(p_default, p_explicit, atol=1e-12)

    def test_cosine_bounded_by_one_at_self_match(self):
        dC, dQ, spec, offsets = self._setup()
        p_cos = mpt.windowed_tensor_similarity(dC, dQ, spec, offsets,
                                        normalize='cosine', verbose=False)
        # |s_cosine| <= 1 across all offsets.
        assert np.all(np.abs(p_cos) <= 1 + 1e-10)

    def test_cosine_at_mix_one_boxcar(self):
        dC, dQ, _, offsets = self._setup()
        spec_box = {'size': np.array([3.0]), 'mix': np.array([1.0])}
        p_box = mpt.windowed_tensor_similarity(dC, dQ, spec_box, offsets,
                                        normalize='cosine', verbose=False)
        assert np.all(np.abs(p_box) <= 1 + 1e-10)

    def test_cosine_intermediate_mix_raises(self):
        dC, dQ, _, offsets = self._setup()
        spec_mid = {'size': np.array([3.0]), 'mix': np.array([0.5])}
        with pytest.raises(ValueError, match=r"(?i)strict shape-only cosine"):
            mpt.windowed_tensor_similarity(dC, dQ, spec_mid, offsets,
                                    normalize='cosine', verbose=False)

    def test_one_sided_accepts_intermediate_mix(self):
        # The mix-in-{0, 1} constraint applies to the cosine option
        # only; oneSidedDenom is closed-form for the full (size, mix)
        # family.
        dC, dQ, _, offsets = self._setup()
        spec_mid = {'size': np.array([3.0]), 'mix': np.array([0.5])}
        # Should not raise.
        p_mid = mpt.windowed_tensor_similarity(dC, dQ, spec_mid, offsets,
                                        normalize='oneSidedDenom',
                                        verbose=False)
        assert p_mid.shape == (5,)
        assert np.all(np.isfinite(p_mid))

    def test_british_alias(self):
        dC, dQ, spec, offsets = self._setup()
        p_us = mpt.windowed_tensor_similarity(dC, dQ, spec, offsets,
                                       normalize='cosine', verbose=False)
        p_gb = mpt.windowed_tensor_similarity(dC, dQ, spec, offsets,
                                       normalise='cosine', verbose=False)
        np.testing.assert_allclose(p_us, p_gb, atol=1e-12)

    def test_mutual_exclusion(self):
        dC, dQ, spec, offsets = self._setup()
        with pytest.raises(TypeError, match="not both"):
            mpt.windowed_tensor_similarity(dC, dQ, spec, offsets,
                                    normalize='cosine', normalise='cosine',
                                    verbose=False)

    def test_bad_value_raises(self):
        dC, dQ, spec, offsets = self._setup()
        with pytest.raises(ValueError, match="normalize"):
            mpt.windowed_tensor_similarity(dC, dQ, spec, offsets,
                                    normalize='bogus', verbose=False)
