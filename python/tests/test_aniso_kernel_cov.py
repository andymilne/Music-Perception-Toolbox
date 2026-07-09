"""Tests for matrix-valued (anisotropic) kernel covariances.

Covers ``interval_kernel_cov`` (the constructor), the whitening
implementation in ``build_exp_tens`` / ``eval_exp_tens`` /
``cos_sim_exp_tens`` / ``entropy_exp_tens``, the raw sliding-comparison
path (``windowed_similarity``), the mode-constraint error paths, and
the analytical cross-checks agreed for the release:

- reduction to the scalar-``sigma`` behaviour at ``Sigma = sigma**2 I``;
- whitened-machinery equality with a direct numpy evaluation of the
  anisotropic density and inner product;
- convergence of the ``sd_shift`` ridge to the exact ``is_rel=True``
  cosine as ``sd_shift`` grows;
- the ``D D^T`` construction against Monte Carlo propagation of
  position noise through differencing;
- single-Gaussian closed forms for the Renyi-2 and differential
  entropies with the ``log det(Sigma) / 2`` change-of-variables term.
"""

import numpy as np
import pytest

import mpt
from mpt import interval_kernel_cov


RNG = np.random.default_rng(20260709)


def _random_spd(dim, rng=RNG, scale=1.0):
    """A well-conditioned random SPD matrix."""
    A = rng.standard_normal((dim, dim))
    return scale * (A @ A.T + dim * np.eye(dim))


def _direct_density(p_tuples, w_tuples, Sigma, X):
    """Direct anisotropic mixture: sum_j W_j exp(-d^T Sigma^{-1} d / 2)."""
    Sinv = np.linalg.inv(Sigma)
    out = np.zeros(X.shape[1])
    for c, W in zip(p_tuples, w_tuples):
        d = X - np.asarray(c, dtype=float).reshape(-1, 1)
        out += W * np.exp(-0.5 * np.einsum("iq,ij,jq->q", d, Sinv, d))
    return out


def _direct_cosine(cx, wx, cy, wy, Sigma):
    """Direct anisotropic cosine: kernels exp(-d^T Sigma^{-1} d / 4)."""
    Sinv = np.linalg.inv(Sigma)

    def ip(ca, wa, cb, wb):
        s = 0.0
        for a, Wa in zip(ca, wa):
            for b, Wb in zip(cb, wb):
                d = np.asarray(a, float) - np.asarray(b, float)
                s += Wa * Wb * np.exp(-0.25 * d @ Sinv @ d)
        return s

    num = ip(cx, wx, cy, wy)
    return num / np.sqrt(ip(cx, wx, cx, wx) * ip(cy, wy, cy, wy))


class TestIntervalKernelCov:
    """The constructor's algebraic structure and error paths."""

    def test_structure(self):
        r = 4
        sp, si, ss = 0.3, 0.7, 1.9
        Sigma = interval_kernel_cov(r, sd_position=sp, sd_interval=si,
                                    sd_shift=ss)
        ddt = 2.0 * np.eye(r) - np.eye(r, k=1) - np.eye(r, k=-1)
        expected = sp**2 * ddt + si**2 * np.eye(r) + ss**2 * np.ones((r, r))
        np.testing.assert_allclose(Sigma, expected, rtol=0, atol=0)

    def test_ddt_is_differencing_map_gram(self):
        """The tridiagonal term is literally D D^T for the r x (r+1)
        first-differencing map."""
        r = 5
        D = np.zeros((r, r + 1))
        for i in range(r):
            D[i, i], D[i, i + 1] = -1.0, 1.0
        Sigma = interval_kernel_cov(r, sd_position=1.0)
        np.testing.assert_allclose(Sigma, D @ D.T, atol=1e-15)

    def test_monte_carlo_position_noise(self):
        """Differenced iid position noise has covariance sd^2 D D^T."""
        r, sd, n = 3, 0.8, 400_000
        onsets = RNG.standard_normal((n, r + 1)) * sd
        intervals = np.diff(onsets, axis=1)
        emp = np.cov(intervals.T)
        np.testing.assert_allclose(
            emp, interval_kernel_cov(r, sd_position=sd), atol=0.02)

    def test_errors(self):
        with pytest.raises(ValueError, match="positive integer"):
            interval_kernel_cov(0, sd_interval=1.0)
        with pytest.raises(ValueError, match="non-negative"):
            interval_kernel_cov(3, sd_interval=-1.0)
        with pytest.raises(ValueError, match="is_rel=True"):
            interval_kernel_cov(3, sd_shift=np.inf)
        with pytest.raises(ValueError, match="singular"):
            interval_kernel_cov(3, sd_shift=1.0)  # rank one alone

    def test_r1_rejected(self):
        """r = 1 is rejected for cross-language parity: a 1x1
        covariance is indistinguishable from a scalar sigma in
        MATLAB."""
        with pytest.raises(ValueError, match="indistinguishable"):
            interval_kernel_cov(1, sd_shift=2.0)
        with pytest.raises(ValueError, match="indistinguishable"):
            mpt.build_exp_tens(np.array([1.0]), np.ones(1),
                               np.array([[4.0]]), 1, False, False, 0.0,
                               False, verbose=False)


class TestScalarReduction:
    """Sigma = sigma**2 I reproduces the scalar-sigma machinery."""

    # One event: an ordered 3-tuple (r == K == 3).
    P = np.array([0.0, 3.0, -3.0])
    W = np.array([1.0, 0.8, 0.6])
    SIG = 1.3

    def _dens_pair(self):
        d_mat = mpt.build_exp_tens(
            self.P, self.W, self.SIG**2 * np.eye(3), 3,
            False, False, 0.0, False, verbose=False)
        d_sca = mpt.build_exp_tens(
            self.P, self.W, self.SIG, 3,
            False, False, 0.0, False, verbose=False)
        return d_mat, d_sca

    def test_eval(self):
        d_mat, d_sca = self._dens_pair()
        X = RNG.standard_normal((3, 40)) * 3.0
        for nrm in ("none", "gaussian", "pdf"):
            np.testing.assert_allclose(
                mpt.eval_exp_tens(d_mat, X, nrm, verbose=False),
                mpt.eval_exp_tens(d_sca, X, nrm, verbose=False),
                rtol=1e-12)

    def test_cosine(self):
        Q = np.array([0.2, 3.4, -2.9])
        v_mat = mpt.cos_sim_exp_tens(
            self.P, self.W, Q, self.W, self.SIG**2 * np.eye(3), 3,
            False, False, 0.0, False, verbose=False)
        v_sca = mpt.cos_sim_exp_tens(
            self.P, self.W, Q, self.W, self.SIG, 3,
            False, False, 0.0, False, verbose=False)
        np.testing.assert_allclose(v_mat, v_sca, rtol=1e-12)

    def test_entropies(self):
        d_mat, d_sca = self._dens_pair()
        for method in ("renyi2", "differential"):
            np.testing.assert_allclose(
                mpt.entropy_exp_tens(d_mat, method=method, verbose=False),
                mpt.entropy_exp_tens(d_sca, method=method, verbose=False),
                rtol=1e-9)


class TestWhitenedVsDirect:
    """The whitened machinery equals a direct anisotropic computation."""

    def test_eval_sa_single_tuple(self):
        r = 3
        Sigma = _random_spd(r, scale=0.5)
        p = np.array([1.0, -0.5, 2.0])
        w = np.array([1.0, 0.7, 0.9])
        dens = mpt.build_exp_tens(p, w, Sigma, r, False, False, 0.0,
                                  False, verbose=False)
        X = RNG.standard_normal((r, 60))
        got = mpt.eval_exp_tens(dens, X, "none", verbose=False)
        # Ordered [sym]=0 at r == K: one tuple, weight the product.
        want = _direct_density([p], [np.prod(w)], Sigma, X)
        np.testing.assert_allclose(got, want, rtol=1e-12)

    def test_eval_normalized(self):
        r = 2
        Sigma = _random_spd(r, scale=0.3)
        p = np.array([0.5, 1.5])
        w = np.array([1.0, 1.0])
        dens = mpt.build_exp_tens(p, w, Sigma, r, False, False, 0.0,
                                  False, verbose=False)
        X = RNG.standard_normal((r, 50))
        got = mpt.eval_exp_tens(dens, X, "gaussian", verbose=False)
        const = (2 * np.pi) ** (-r / 2) * np.linalg.det(Sigma) ** (-0.5)
        want = const * _direct_density([p], [1.0], Sigma, X)
        np.testing.assert_allclose(got, want, rtol=1e-12)

    def test_pdf_integrates_to_one(self):
        """'pdf' really is a pdf in the original coordinates."""
        r = 2
        Sigma = np.array([[0.09, 0.05], [0.05, 0.16]])
        p = np.array([0.3, -0.2])
        dens = mpt.build_exp_tens(p, np.ones(2), Sigma, r, False, False,
                                  0.0, False, verbose=False)
        g = np.linspace(-3.0, 3.0, 301)
        GX, GY = np.meshgrid(g, g, indexing="ij")
        X = np.vstack([GX.ravel(), GY.ravel()])
        vals = mpt.eval_exp_tens(dens, X, "pdf", verbose=False)
        integral = np.sum(vals) * (g[1] - g[0]) ** 2
        assert abs(integral - 1.0) < 1e-6

    def test_cosine_multi_event(self):
        """Several events (each one ordered tuple), raw-input path."""
        r = 3
        Sigma = interval_kernel_cov(r, sd_position=0.4, sd_interval=0.2,
                                    sd_shift=0.6)
        # MA form: one attribute, r x N value matrices (N events).
        cx = [np.array([0.0, 1.0, 0.5]), np.array([0.2, 1.1, 0.4]),
              np.array([-1.0, 0.0, 2.0])]
        cy = [np.array([0.1, 0.9, 0.55]), np.array([2.0, -1.0, 0.3])]
        PX = np.column_stack(cx)
        PY = np.column_stack(cy)
        wx = np.ones((r, len(cx)))
        wy = np.ones((r, len(cy)))
        got = mpt.cos_sim_exp_tens(
            [PX], [wx], [PY], [wy], [Sigma], [r],
            [False], [False], [0.0], [False], verbose=False)
        want = _direct_cosine(cx, [1.0] * len(cx), cy, [1.0] * len(cy),
                              Sigma)
        np.testing.assert_allclose(got, want, rtol=1e-12)

    def test_ma_mixed_attributes(self):
        """One anisotropic attribute tensored with one scalar attribute."""
        r = 2
        Sigma = _random_spd(r, scale=0.2)
        sig_t = 0.5
        # Two events; attribute 1: ordered pairs; attribute 2: a time.
        P1 = np.array([[0.0, 1.0], [2.0, 2.5]])          # r x N
        P2 = np.array([[0.0, 4.0]])                       # 1 x N
        W = np.ones((1, 2))
        dens = mpt.build_exp_tens(
            [P1, P2], [np.ones((r, 2)), W], [Sigma, sig_t], [r, 1],
            [False, False], [False, False], [0.0, 0.0], [False, True],
            verbose=False)
        X = RNG.standard_normal((r + 1, 40))
        got = mpt.eval_exp_tens(dens, X, "none", verbose=False)
        want = np.zeros(X.shape[1])
        for n in range(2):
            f1 = _direct_density([P1[:, n]], [1.0], Sigma, X[:r])
            d2 = X[r] - P2[0, n]
            f2 = np.exp(-d2**2 / (2 * sig_t**2))
            want += f1 * f2
        np.testing.assert_allclose(got, want, rtol=1e-12)


class TestShiftRidgeLimits:
    """The sd_shift ridge interpolates towards the exact relative mode."""

    P1 = np.array([0.0, 0.35, 0.15])   # log-IOI triples (r = K = 3)
    P2 = np.array([0.9, 1.25, 1.05])   # the same shape, shifted by 0.9
    P3 = np.array([0.0, 0.30, 0.35])   # a different shape
    W = np.ones(3)

    def _cos_aniso(self, a, b, sd_shift):
        Sigma = interval_kernel_cov(3, sd_interval=0.1, sd_shift=sd_shift)
        return float(mpt.cos_sim_exp_tens(
            a, self.W, b, self.W, Sigma, 3, False, False, 0.0, False,
            verbose=False))

    def _cos_rel(self, a, b):
        return float(mpt.cos_sim_exp_tens(
            a, self.W, b, self.W, 0.1, 3, True, False, 0.0, False,
            verbose=False))

    def test_convergence_to_rel(self):
        for a, b in ((self.P1, self.P2), (self.P1, self.P3)):
            target = self._cos_rel(a, b)
            prev_err = np.inf
            for ss in (1.0, 10.0, 100.0):
                err = abs(self._cos_aniso(a, b, ss) - target)
                assert err < prev_err
                prev_err = err
            assert prev_err < 1e-3

    def test_lambda_zero_penalizes_shift(self):
        """Without the ridge, a common shift decays the match; the
        ridge restores it gradedly."""
        no_ridge = self._cos_aniso(self.P1, self.P2, 0.0)
        with_ridge = self._cos_aniso(self.P1, self.P2, 5.0)
        assert no_ridge < 1e-6
        assert with_ridge > 0.9

    def test_shape_tolerance_independent_of_ridge(self):
        """The ridge leaves the shape comparison essentially unchanged:
        for shifted-identical tuples the similarity is ~1 at large
        sd_shift while a shape mismatch is still resolved."""
        same_shape = self._cos_aniso(self.P1, self.P2, 100.0)
        diff_shape = self._cos_aniso(self.P1, self.P3, 100.0)
        assert same_shape > 0.999
        assert diff_shape < same_shape - 0.05


class TestEntropyClosedForms:
    """Single-Gaussian closed forms including the log det term."""

    def test_renyi2_single_gaussian(self):
        r = 3
        Sigma = _random_spd(r, scale=0.2)
        p = np.array([0.0, 1.0, -1.0])
        dens = mpt.build_exp_tens(p, np.ones(r), Sigma, r, False, False,
                                  0.0, False, verbose=False)
        got = mpt.entropy_exp_tens(dens, method="renyi2", base=np.e,
                                   verbose=False)
        # H2 of N(mu, Sigma): (d/2) log(4 pi) + (1/2) log det Sigma.
        want = 0.5 * r * np.log(4 * np.pi) + 0.5 * np.linalg.slogdet(Sigma)[1]
        np.testing.assert_allclose(got, want, rtol=1e-10)

    def test_differential_single_gaussian(self):
        r = 2
        Sigma = np.array([[0.04, -0.01], [-0.01, 0.09]])
        p = np.array([0.0, 0.5])
        dens = mpt.build_exp_tens(p, np.ones(r), Sigma, r, False, False,
                                  0.0, False, verbose=False)
        got = mpt.entropy_exp_tens(dens, method="differential", base=np.e,
                                   verbose=False)
        want = 0.5 * r * np.log(2 * np.pi * np.e) \
            + 0.5 * np.linalg.slogdet(Sigma)[1]
        np.testing.assert_allclose(got, want, rtol=1e-4)


class TestWindowedSimilarity:
    """End-to-end sliding comparison with an anisotropic interval
    attribute, swept on a scalar time attribute."""

    def test_sweep_peaks_at_match(self):
        r = 2
        Sigma = interval_kernel_cov(r, sd_interval=0.05, sd_shift=5.0)
        # Context: five events, each an ordered log-IOI pair plus an
        # onset. Event 3 matches the query's shape at a different
        # "tempo" (common shift of the log-IOI pair).
        shapes = np.array([
            [0.00, 0.40],   # n=0
            [0.30, 0.10],   # n=1
            [0.20, 0.20],   # n=2  <- query shape + 0.9
            [0.50, 0.00],   # n=3
            [0.05, 0.45],   # n=4
        ]).T + np.array([0.0, 0.0, 0.9, 0.0, 0.0])
        onsets = np.array([[0.0, 1.0, 2.0, 3.0, 4.0]])
        p_context = [shapes, onsets]
        w_context = [np.ones((r, 5)), np.ones((1, 5))]
        p_query = [np.array([[-0.70], [-0.70]]), np.array([[0.0]])]
        w_query = [np.ones((r, 1)), np.ones((1, 1))]
        prof = mpt.windowed_similarity(
            p_context, w_context, p_query, w_query,
            [Sigma, 0.25], [r, 1], [False, False], [False, False],
            [0.0, 0.0], is_sym=[False, True],
            centres=onsets.ravel(), window_attr=1, drop_window_attr=True,
            context_window=("rect", 0.5),
            normalize="oneSidedDenom", verbose=False)
        assert prof.shape == (5,)
        assert int(np.argmax(prof)) == 2
        # The match is up to a common shift, absorbed by the ridge.
        assert prof[2] > 0.9
        # Manual check of one off-peak step: window drops the time
        # attribute, so the step-n comparison is the plain SA cosine...
        # (oneSidedDenom) of the anisotropic pairs.
        got_0 = prof[0]
        Sinv = np.linalg.inv(Sigma)
        d = shapes[:, 0] - p_query[0].ravel()
        num = np.exp(-0.25 * d @ Sinv @ d)
        assert abs(got_0 - num) < 1e-10

    def test_guards_on_posttensor_windowing(self):
        r = 2
        Sigma = interval_kernel_cov(r, sd_interval=0.1)
        P1 = np.array([[0.0], [1.0]])
        P2 = np.array([[0.0]])
        dens = mpt.build_exp_tens(
            [P1, P2], [np.ones((r, 1)), np.ones((1, 1))],
            [Sigma, 0.5], [r, 1], [False, False], [False, False],
            [0.0, 0.0], [False, True], verbose=False)
        spec = {"size": [np.nan, 2.0], "mix": 0.0,
                "centre": [None, np.array([0.0])]}
        with pytest.raises(NotImplementedError, match="whitened"):
            mpt.window_tensor(dens, spec)


class TestConstraints:
    """Mode-constraint and validation error paths."""

    P = np.array([0.0, 1.0, 2.0])
    W = np.ones(3)

    def _build(self, sigma, r=3, is_rel=False, is_per=False, period=0.0,
               is_sym=False):
        return mpt.build_exp_tens(self.P, self.W, sigma, r, is_rel,
                                  is_per, period, is_sym, verbose=False)

    def test_rejects_sym(self):
        with pytest.raises(ValueError, match="ordered multiset"):
            self._build(np.eye(3), is_sym=True)

    def test_rejects_rel(self):
        with pytest.raises(ValueError, match="is_rel=False"):
            self._build(np.eye(3), is_rel=True)

    def test_rejects_per(self):
        with pytest.raises(ValueError, match="is_per=False"):
            self._build(np.eye(3), is_per=True, period=12.0)

    def test_rejects_r_lt_K(self):
        with pytest.raises(ValueError, match="r == K"):
            self._build(np.eye(2), r=2)

    def test_rejects_wrong_size(self):
        with pytest.raises(ValueError, match="tuple dimension"):
            self._build(np.eye(4))

    def test_rejects_asymmetric(self):
        S = np.eye(3)
        S[0, 1] = 0.5
        with pytest.raises(ValueError, match="symmetric"):
            self._build(S)

    def test_rejects_indefinite(self):
        S = np.eye(3)
        S[0, 0] = -1.0
        with pytest.raises(ValueError, match="positive definite"):
            self._build(S)

    def test_rejects_nonfinite(self):
        S = np.eye(3)
        S[1, 1] = np.inf
        with pytest.raises(ValueError, match="finite"):
            self._build(S)

    def test_rejects_nan_values(self):
        p = np.array([0.0, np.nan, 2.0])
        with pytest.raises(ValueError, match="NaN"):
            mpt.build_exp_tens(p, self.W, np.eye(3), 3, False, False,
                               0.0, False, verbose=False)

    def test_rejects_spectrum(self):
        with pytest.raises(TypeError, match="spectrum"):
            mpt.eval_exp_tens(
                self.P, self.W, np.eye(3), 3, False, False, 0.0, False,
                np.zeros((3, 1)), spectrum=[12, 0.67], verbose=False)

    def test_rejects_mismatched_covs_in_cosine(self):
        d1 = self._build(np.eye(3))
        d2 = self._build(2.0 * np.eye(3))
        with pytest.raises(ValueError, match="kernel covariance"):
            mpt.cos_sim_exp_tens(d1, d2, verbose=False)

    def test_rejects_cov_vs_scalar_in_cosine(self):
        d1 = self._build(np.eye(3))
        d2 = self._build(1.0)
        with pytest.raises((ValueError, TypeError)):
            mpt.cos_sim_exp_tens(d1, d2, verbose=False)


class TestOrderedTupleEqualsBoundSingletons:
    """Manuscript Sec. 3: an ordered K-tuple on one attribute with a
    diagonal covariance coincides with K singleton attributes bound
    together, coordinate order matching attribute order."""

    def test_diag_cov_equals_singleton_attributes(self):
        s1, s2 = 0.4, 0.9
        # Ordered pairs as one attribute with diag covariance.
        PX = np.array([[0.0, 1.0], [2.0, 3.0]])     # r x N (2 events)
        PY = np.array([[0.1, 0.8], [2.2, 2.9]])
        Wr = np.ones((2, 2))
        v_aniso = mpt.cos_sim_exp_tens(
            [PX], [Wr], [PY], [Wr], [np.diag([s1**2, s2**2])], [2],
            [False], [False], [0.0], [False], verbose=False)
        # The same data as two singleton attributes.
        v_two = mpt.cos_sim_exp_tens(
            [PX[:1], PX[1:]], [np.ones((1, 2)), np.ones((1, 2))],
            [PY[:1], PY[1:]], [np.ones((1, 2)), np.ones((1, 2))],
            [s1, s2], [1, 1], [False, False], [False, False],
            [0.0, 0.0], [True, True], verbose=False)
        np.testing.assert_allclose(v_aniso, v_two, rtol=1e-12)
