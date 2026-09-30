"""Tests for matrix-valued (anisotropic) kernel covariances.

Covers ``kernel_cov`` (the constructor), the whitening
implementation in ``build_maet`` / ``eval_maet`` /
``sim_maet`` / ``entropy_maet``, the raw sliding-comparison
path (``swept_similarity``), the mode-constraint error paths, and
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
from mpt import kernel_cov


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


class TestKernelCov:
    """The constructor's algebraic structure and error paths."""

    def test_structure(self):
        r = 4
        sp, si, ss = 0.3, 0.7, 1.9
        Sigma = kernel_cov(r, sd_value=sp, sd_interval=si,
                           sd_shift=ss, differenced=True)
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
        Sigma = kernel_cov(r, sd_value=1.0, differenced=True)
        np.testing.assert_allclose(Sigma, D @ D.T, atol=1e-15)

    def test_monte_carlo_position_noise(self):
        """Differenced iid position noise has covariance sd^2 D D^T."""
        r, sd, n = 3, 0.8, 400_000
        onsets = RNG.standard_normal((n, r + 1)) * sd
        intervals = np.diff(onsets, axis=1)
        emp = np.cov(intervals.T)
        np.testing.assert_allclose(
            emp, kernel_cov(r, sd_value=sd, differenced=True), atol=0.02)

    def test_errors(self):
        with pytest.raises(ValueError, match="positive integer"):
            kernel_cov(0, sd_interval=1.0, differenced=True)
        with pytest.raises(ValueError, match="non-negative"):
            kernel_cov(3, sd_interval=-1.0, differenced=True)
        with pytest.raises(ValueError, match="is_rel=True"):
            kernel_cov(3, sd_shift=np.inf, differenced=True)
        with pytest.raises(ValueError, match="singular"):
            kernel_cov(3, sd_shift=1.0, differenced=True)  # rank one alone

    def test_differenced_flag_is_mandatory(self):
        with pytest.raises(TypeError):
            kernel_cov(3, sd_interval=1.0)
        with pytest.raises(TypeError, match="differenced"):
            kernel_cov(3, sd_interval=1.0, differenced="yes")

    def test_undifferenced_structure(self):
        """Positions: sd_value^2 I + sd_interval^2 P S S^T P
        + sd_shift^2 J, the walk centred on the tuple's mean."""
        r = 4
        sp, si, ss = 0.3, 0.7, 1.9
        Sigma = kernel_cov(r, sd_value=sp, sd_interval=si, sd_shift=ss,
                           differenced=False)
        S = np.tril(np.ones((r, r - 1)), k=-1)
        P = np.eye(r) - np.ones((r, r)) / r
        expected = (sp**2 * np.eye(r) + si**2 * P @ S @ S.T @ P
                    + ss**2 * np.ones((r, r)))
        np.testing.assert_allclose(Sigma, expected, atol=1e-15)
        # The walk term is centred: it annihilates the all-ones vector.
        walk = kernel_cov(r, sd_interval=1.0, sd_shift=1.0,
                          differenced=False) - np.ones((r, r))
        np.testing.assert_allclose(walk @ np.ones(r), 0.0, atol=1e-14)

    def test_differenced_is_undifferenced_pushed_through_D(self):
        """D Sigma_pos D^T = sd_value^2 D D^T + sd_interval^2 I:
        the two cases are one model, the ridge falling away under D."""
        r = 5
        D = np.zeros((r - 1, r))
        for i in range(r - 1):
            D[i, i], D[i, i + 1] = -1.0, 1.0
        Sp = kernel_cov(r, sd_value=0.4, sd_interval=0.9, sd_shift=3.0,
                        differenced=False)
        Sd = kernel_cov(r - 1, sd_value=0.4, sd_interval=0.9,
                        differenced=True)
        np.testing.assert_allclose(D @ Sp @ D.T, Sd, atol=1e-12)

    def test_monte_carlo_undifferenced_interval_noise(self):
        """Centred cumulative sums of iid interval noise have covariance
        sd^2 P S S^T P."""
        r, sd, n = 4, 0.8, 400_000
        eps = RNG.standard_normal((n, r - 1)) * sd
        pos = np.concatenate([np.zeros((n, 1)), np.cumsum(eps, axis=1)], axis=1)
        pos -= pos.mean(axis=1, keepdims=True)
        emp = np.cov(pos.T)
        np.testing.assert_allclose(
            emp, kernel_cov(r, sd_interval=sd, sd_shift=1.0, differenced=False)
            - np.ones((r, r)), atol=0.02)

    def test_undifferenced_singular_cases(self):
        with pytest.raises(ValueError, match="singular"):
            kernel_cov(3, sd_interval=1.0, differenced=False)  # no tolerance along 1
        # sd_value alone, and sd_shift with sd_interval, are fine.
        kernel_cov(3, sd_value=1.0, differenced=False)
        kernel_cov(3, sd_interval=1.0, sd_shift=0.1, differenced=False)

    def test_r1_rejected(self):
        """r = 1 is rejected for cross-language parity: a 1x1
        covariance is indistinguishable from a scalar sigma in
        MATLAB."""
        with pytest.raises(ValueError, match="indistinguishable"):
            kernel_cov(1, sd_shift=2.0, differenced=True)
        with pytest.raises(ValueError, match="indistinguishable"):
            mpt.build_maet(np.array([1.0]), np.ones(1),
                               np.array([[4.0]]), 1, False, False, 0.0,
                               False, verbose=False)


class TestScalarReduction:
    """Sigma = sigma**2 I reproduces the scalar-sigma machinery."""

    # One event: an ordered 3-tuple (r == K == 3).
    P = np.array([0.0, 3.0, -3.0])
    W = np.array([1.0, 0.8, 0.6])
    SIG = 1.3

    def _dens_pair(self):
        d_mat = mpt.build_maet(
            self.P, self.W, self.SIG**2 * np.eye(3), 3,
            False, False, 0.0, False, verbose=False)
        d_sca = mpt.build_maet(
            self.P, self.W, self.SIG, 3,
            False, False, 0.0, False, verbose=False)
        return d_mat, d_sca

    def test_eval(self):
        d_mat, d_sca = self._dens_pair()
        X = RNG.standard_normal((3, 40)) * 3.0
        for nrm in ("none", "gaussian", "pdf"):
            np.testing.assert_allclose(
                mpt.eval_maet(d_mat, X, nrm, verbose=False),
                mpt.eval_maet(d_sca, X, nrm, verbose=False),
                rtol=1e-12)

    def test_cosine(self):
        Q = np.array([0.2, 3.4, -2.9])
        v_mat = mpt.sim_maet(
            self.P, self.W, Q, self.W, self.SIG**2 * np.eye(3), 3,
            False, False, 0.0, False, verbose=False)
        v_sca = mpt.sim_maet(
            self.P, self.W, Q, self.W, self.SIG, 3,
            False, False, 0.0, False, verbose=False)
        np.testing.assert_allclose(v_mat, v_sca, rtol=1e-12)

    def test_entropies(self):
        d_mat, d_sca = self._dens_pair()
        # Certifying tight accuracy needs an infeasibly fine grid in 3-D;
        # pin a feasible accuracy on both sides (the materialised-vs-scalar
        # comparison uses the same grid, so it is accuracy-independent).
        for method in ("renyi2", "differential"):
            np.testing.assert_allclose(
                mpt.entropy_maet(
                    d_mat, method=method,
                    truncation_sigmas=4.0, verbose=False),
                mpt.entropy_maet(
                    d_sca, method=method,
                    truncation_sigmas=4.0, verbose=False),
                rtol=1e-9)


class TestWhitenedVsDirect:
    """The whitened machinery equals a direct anisotropic computation."""

    def test_eval_single_multiset_single_tuple(self):
        r = 3
        Sigma = _random_spd(r, scale=0.5)
        p = np.array([1.0, -0.5, 2.0])
        w = np.array([1.0, 0.7, 0.9])
        dens = mpt.build_maet(p, w, Sigma, r, False, False, 0.0,
                                  False, verbose=False)
        X = RNG.standard_normal((r, 60))
        got = mpt.eval_maet(dens, X, "none", verbose=False)
        # Ordered [exch]=0 at r == K: one tuple, weight the product.
        want = _direct_density([p], [np.prod(w)], Sigma, X)
        np.testing.assert_allclose(got, want, rtol=1e-12)

    def test_eval_normalized(self):
        r = 2
        Sigma = _random_spd(r, scale=0.3)
        p = np.array([0.5, 1.5])
        w = np.array([1.0, 1.0])
        dens = mpt.build_maet(p, w, Sigma, r, False, False, 0.0,
                                  False, verbose=False)
        X = RNG.standard_normal((r, 50))
        got = mpt.eval_maet(dens, X, "gaussian", verbose=False)
        const = (2 * np.pi) ** (-r / 2) * np.linalg.det(Sigma) ** (-0.5)
        want = const * _direct_density([p], [1.0], Sigma, X)
        np.testing.assert_allclose(got, want, rtol=1e-12)

    def test_pdf_integrates_to_one(self):
        """'pdf' really is a pdf in the original coordinates."""
        r = 2
        Sigma = np.array([[0.09, 0.05], [0.05, 0.16]])
        p = np.array([0.3, -0.2])
        dens = mpt.build_maet(p, np.ones(2), Sigma, r, False, False,
                                  0.0, False, verbose=False)
        g = np.linspace(-3.0, 3.0, 301)
        GX, GY = np.meshgrid(g, g, indexing="ij")
        X = np.vstack([GX.ravel(), GY.ravel()])
        vals = mpt.eval_maet(dens, X, "pdf", verbose=False)
        integral = np.sum(vals) * (g[1] - g[0]) ** 2
        assert abs(integral - 1.0) < 1e-6

    def test_cosine_multi_event(self):
        """Several events (each one ordered tuple), raw-input path."""
        r = 3
        Sigma = kernel_cov(r, sd_value=0.4, sd_interval=0.2,
                           sd_shift=0.6, differenced=True)
        # MA form: one attribute, r x N value matrices (N events).
        cx = [np.array([0.0, 1.0, 0.5]), np.array([0.2, 1.1, 0.4]),
              np.array([-1.0, 0.0, 2.0])]
        cy = [np.array([0.1, 0.9, 0.55]), np.array([2.0, -1.0, 0.3])]
        PX = np.column_stack(cx)
        PY = np.column_stack(cy)
        wx = np.ones((r, len(cx)))
        wy = np.ones((r, len(cy)))
        got = mpt.sim_maet(
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
        dens = mpt.build_maet(
            [P1, P2], [np.ones((r, 2)), W], [Sigma, sig_t], [r, 1],
            [False, False], [False, False], [0.0, 0.0], [False, True],
            verbose=False)
        X = RNG.standard_normal((r + 1, 40))
        got = mpt.eval_maet(dens, X, "none", verbose=False)
        want = np.zeros(X.shape[1])
        for n in range(2):
            f1 = _direct_density([P1[:, n]], [1.0], Sigma, X[:r])
            d2 = X[r] - P2[0, n]
            f2 = np.exp(-d2**2 / (2 * sig_t**2))
            want += f1 * f2
        # inf resolves to the accuracy-floor width, so near-zero query
        # points differ from the exhaustive reference by a sub-1e-12
        # absolute tail; add an absolute floor scaled to the peak.
        np.testing.assert_allclose(got, want, rtol=1e-12,
                                   atol=1e-11 * float(np.max(np.abs(want))))


class TestShiftRidgeLimits:
    """The sd_shift ridge interpolates towards the exact relative mode."""

    P1 = np.array([0.0, 0.35, 0.15])   # log-IOI triples (r = K = 3)
    P2 = np.array([0.9, 1.25, 1.05])   # the same shape, shifted by 0.9
    P3 = np.array([0.0, 0.30, 0.35])   # a different shape
    W = np.ones(3)

    def _cos_aniso(self, a, b, sd_shift):
        Sigma = kernel_cov(3, sd_interval=0.1, sd_shift=sd_shift, differenced=True)
        return float(mpt.sim_maet(
            a, self.W, b, self.W, Sigma, 3, False, False, 0.0, False,
            verbose=False))

    def _cos_rel(self, a, b):
        return float(mpt.sim_maet(
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
        dens = mpt.build_maet(p, np.ones(r), Sigma, r, False, False,
                                  0.0, False, verbose=False)
        got = mpt.entropy_maet(dens, method="renyi2", base=np.e,
                                   verbose=False)
        # H2 of N(mu, Sigma): (d/2) log(4 pi) + (1/2) log det Sigma.
        want = 0.5 * r * np.log(4 * np.pi) + 0.5 * np.linalg.slogdet(Sigma)[1]
        np.testing.assert_allclose(got, want, rtol=1e-10)

    def test_differential_single_gaussian(self):
        r = 2
        Sigma = np.array([[0.04, -0.01], [-0.01, 0.09]])
        p = np.array([0.0, 0.5])
        dens = mpt.build_maet(p, np.ones(r), Sigma, r, False, False,
                                  0.0, False, verbose=False)
        # This entropy is near zero, so the rtol=1e-4 closed-form check is
        # unusually sensitive and needs the tighter accuracy (still a
        # feasible ~13M-point grid in 2-D).
        got = mpt.entropy_maet(dens, method="differential", base=np.e,
                                   truncation_sigmas=6.0, verbose=False)
        want = 0.5 * r * np.log(2 * np.pi * np.e) \
            + 0.5 * np.linalg.slogdet(Sigma)[1]
        np.testing.assert_allclose(got, want, rtol=1e-4)


class TestSweptSimilarity:
    """End-to-end sliding comparison with an anisotropic interval
    attribute, swept on a scalar time attribute."""

    def test_sweep_peaks_at_match(self):
        r = 2
        Sigma = kernel_cov(r, sd_interval=0.05, sd_shift=5.0, differenced=True)
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
        prof = mpt.swept_similarity(
            p_context, w_context, p_query, w_query,
            [Sigma, 0.25], [r, 1], [False, False], [False, False],
            [0.0, 0.0], is_exch=[False, True],
            sweep={1: onsets.ravel()}, align={1: "window"}, drop=[1],
            window={1: {"shape": "rect", "width": 0.5}},
            normalize="oneSidedDenom", verbose=False)
        assert prof.shape == (5,)
        assert int(np.argmax(prof)) == 2
        # The match is up to a common shift, absorbed by the ridge.
        assert prof[2] > 0.9
        # Manual check of one off-peak step: window drops the time
        # attribute, so the step-n comparison is the plain single-multiset cosine...
        # (oneSidedDenom) of the anisotropic pairs.
        got_0 = prof[0]
        Sinv = np.linalg.inv(Sigma)
        d = shapes[:, 0] - p_query[0].ravel()
        num = np.exp(-0.25 * d @ Sinv @ d)
        assert abs(got_0 - num) < 1e-10


class TestConstraints:
    """Mode-constraint and validation error paths."""

    P = np.array([0.0, 1.0, 2.0])
    W = np.ones(3)

    def _build(self, sigma, r=3, is_rel=False, is_per=False, period=0.0,
               is_exch=False):
        return mpt.build_maet(self.P, self.W, sigma, r, is_rel,
                                  is_per, period, is_exch, verbose=False)

    def test_rejects_exch(self):
        with pytest.raises(ValueError, match="ordered multiset"):
            self._build(np.eye(3), is_exch=True)

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
            mpt.build_maet(p, self.W, np.eye(3), 3, False, False,
                               0.0, False, verbose=False)

    def test_rejects_spectrum(self):
        with pytest.raises(TypeError, match="spectrum"):
            mpt.eval_maet(
                self.P, self.W, np.eye(3), 3, False, False, 0.0, False,
                np.zeros((3, 1)), spectrum=[12, 0.67], verbose=False)

    def test_rejects_mismatched_covs_in_cosine(self):
        d1 = self._build(np.eye(3))
        d2 = self._build(2.0 * np.eye(3))
        with pytest.raises(ValueError, match="kernel covariance"):
            mpt.sim_maet(d1, d2, verbose=False)

    def test_rejects_cov_vs_scalar_in_cosine(self):
        d1 = self._build(np.eye(3))
        d2 = self._build(1.0)
        with pytest.raises((ValueError, TypeError)):
            mpt.sim_maet(d1, d2, verbose=False)


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
        v_aniso = mpt.sim_maet(
            [PX], [Wr], [PY], [Wr], [np.diag([s1**2, s2**2])], [2],
            [False], [False], [0.0], [False], verbose=False)
        # The same data as two singleton attributes.
        v_two = mpt.sim_maet(
            [PX[:1], PX[1:]], [np.ones((1, 2)), np.ones((1, 2))],
            [PY[:1], PY[1:]], [np.ones((1, 2)), np.ones((1, 2))],
            [s1, s2], [1, 1], [False, False], [False, False],
            [0.0, 0.0], [True, True], verbose=False)
        np.testing.assert_allclose(v_aniso, v_two, rtol=1e-12)


class TestDegenerateNestedFlattening:
    """A matrix-valued kernel covariance on a degenerate nested
    attribute -- bind_events over flat single-value events -- is
    flattened to the equivalent flat ordered tuple (v3+). The
    bound and manually stacked flat triples must agree exactly;
    non-degenerate nesting is rejected, and outer-level exch/rel on a
    degenerate spec are rejected by the canonical constraint messages.
    """

    SIG = kernel_cov(3, sd_value=0.07, sd_shift=0.2, differenced=True)

    def _triples(self, seed=7, n=12):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(1, n))
        pb, _, specs = mpt.unpack_pre_maet(mpt.bind_events([x], None, 3))
        P = pb[0]
        flat = {"r": 3, "exch": False, "rel": False}
        return P, specs[0], flat, rng

    def _build(self, P, spec, sigma):
        N = P.shape[1]
        return mpt.build_maet(
            [P], [np.ones((3, N))], specs=[spec], sigma=[sigma],
            is_per=[False], period=[0.0], verbose=False)

    def test_bound_equals_flat_eval_matrix_sigma(self):
        P, nested, flat, rng = self._triples()
        d_n = self._build(P, nested, self.SIG)
        d_f = self._build(P, flat, self.SIG)
        pts = rng.normal(size=(3, 6))
        np.testing.assert_allclose(
            np.asarray(mpt.eval_maet(d_n, pts), dtype=float),
            np.asarray(mpt.eval_maet(d_f, pts), dtype=float),
            rtol=1e-12)

    def test_bound_equals_flat_cosine_matrix_sigma(self):
        P, nested, flat, _ = self._triples()
        d_n = self._build(P, nested, self.SIG)
        d_f = self._build(P, flat, self.SIG)
        v = mpt.sim_maet(d_n, d_f, verbose=False)
        np.testing.assert_allclose(float(v), 1.0, rtol=1e-12)

    def test_bound_equals_flat_scalar_sigma_baseline(self):
        P, nested, flat, rng = self._triples(seed=11)
        d_n = self._build(P, nested, 0.3)
        d_f = self._build(P, flat, 0.3)
        pts = rng.normal(size=(3, 6))
        np.testing.assert_allclose(
            np.asarray(mpt.eval_maet(d_n, pts), dtype=float),
            np.asarray(mpt.eval_maet(d_f, pts), dtype=float),
            rtol=1e-12)

    def test_windowed_bound_specs_equals_flat_is_exch(self):
        # The demo pipeline: difference -> log -> bind, swept with
        # swept_similarity via specs=, against the manually stacked
        # flat surface via is_exch=.
        onsets = np.array([0.0, 0.5, 0.75, 1.0, 2.0, 2.5, 2.75, 3.0,
                           4.0, 4.4, 4.6, 4.8])
        p_d, w_d, sp_d = mpt.unpack_pre_maet(mpt.difference_events([onsets[None, :]],
                                               None, [1]))
        p_d[0] = np.log(p_d[0])
        p_b, w_b, sp_b = mpt.unpack_pre_maet(mpt.bind_events(p_d, w_d, [3], specs=sp_d))
        n_tri = p_b[0].shape[1]
        tri_times = onsets[:n_tri]
        li = np.log(np.diff(onsets))
        tri_manual = np.stack([li[i:i + n_tri] for i in range(3)])
        np.testing.assert_array_equal(p_b[0], tri_manual)
        q = np.log(np.array([0.5, 0.25, 0.25]))
        w_ctx = [np.ones((3, n_tri)), np.ones((1, n_tri))]
        p_q = [q[:, None], np.array([[0.0]])]
        w_q = [np.ones((3, 1)), np.ones((1, 1))]
        tsp = {"r": 1, "exch": True, "rel": False}
        kw = dict(sweep={1: tri_times}, align={1: "window"}, drop=[1],
                  window={1: {"shape": "rect", "width": 0.1}},
                  normalize="oneSidedDenom", verbose=False)
        a = mpt.swept_similarity(
            [p_b[0], tri_times[None, :]], w_ctx, p_q, w_q,
            [self.SIG, 0.25], [3, 1], [False, False], [False, False],
            [0.0, 0.0], specs=[sp_b[0], tsp], **kw)
        b = mpt.swept_similarity(
            [tri_manual, tri_times[None, :]], w_ctx, p_q, w_q,
            [self.SIG, 0.25], [3, 1], [False, False], [False, False],
            [0.0, 0.0], is_exch=[False, True], **kw)
        np.testing.assert_allclose(np.asarray(a), np.asarray(b),
                                   rtol=1e-12, atol=1e-15)

    def test_flat_spec_with_matrix_sigma_in_specs_form(self):
        # specs= with a matrix sigma on a *flat* spec was previously
        # blanket-rejected; it must now match the positional form.
        P, _, flat, _ = self._triples(seed=5)
        N = P.shape[1]
        d_s = self._build(P, flat, self.SIG)
        d_p = mpt.build_maet(
            [P], [np.ones((3, N))], [self.SIG], [3], [False], [False],
            [0.0], [False], verbose=False)
        v = mpt.sim_maet(d_s, d_p, verbose=False)
        np.testing.assert_allclose(float(v), 1.0, rtol=1e-12)

    def test_non_degenerate_nested_rejected(self):
        rng = np.random.default_rng(2)
        x2 = rng.normal(size=(2, 12))            # K = 2 constituents
        pb2, _, sp2 = mpt.unpack_pre_maet(mpt.bind_events([x2], None, 3))
        with pytest.raises(ValueError, match="not[ ]?degenerate"):
            mpt.build_maet(
                [pb2[0]], None, specs=[sp2[0]],
                sigma=[np.eye(6) * 0.01], is_per=[False],
                period=[0.0], verbose=False)

    def test_outer_exch_rejected_canonically(self):
        rng = np.random.default_rng(3)
        pb, _, sp = mpt.unpack_pre_maet(mpt.bind_events([rng.normal(size=(1, 12))], None, 3,
                                    exch_outer=True))
        with pytest.raises(ValueError, match="ordered multiset"):
            self._build(pb[0], sp[0], np.eye(3) * 0.01)

    def test_outer_rel_rejected_canonically(self):
        rng = np.random.default_rng(4)
        pb, _, sp = mpt.unpack_pre_maet(mpt.bind_events([rng.normal(size=(1, 12))], None, 3,
                                    rel_outer=True))
        with pytest.raises(ValueError, match="is_rel=False"):
            self._build(pb[0], sp[0], np.eye(3) * 0.01)


class TestTruncationParity:
    """The kernel-covariance route truncates exactly as the isotropic one.

    A matrix-valued covariance is evaluated by whitening onto the
    isotropic unit-sigma kernel, and the truncation rule (drop a kernel
    contribution whose value falls below ``exp(-k**2/2)`` of its peak)
    then holds in the Mahalanobis metric. Regression: the whitened
    tuple (dim = r >= 2, a handful of centres) always reached the
    exhaustive branch of the kernel-sum helper, which skipped the
    cutoff, so ``kernel_cov`` similarities were untruncated while the
    equivalent paired (relative + absolute) attributes were truncated.
    """

    Q = np.array([6000.0, 6200.0, 6400.0, 6700.0])
    SD_REL = 30.0

    @staticmethod
    def _pm(v, copies):
        p = [np.reshape(v, (-1, 1))] * len(copies)
        sig, rel = (list(c) for c in zip(*copies))
        n = len(copies)
        return mpt.pack_pre_maet(p, None, mpt.flat_specs(
            p, r=len(v), rel=rel, exch=[False] * n, sigma=sig,
            is_per=[False] * n, period=[0.0] * n))

    def _pair(self, s, ts, v_x):
        """(paired isotropic, kernel_cov) similarity of Q and v_x."""
        vs = self.SD_REL ** 2 + s * s
        c = kernel_cov(4, sd_value=self.SD_REL * s / np.sqrt(vs),
                       sd_shift=s * s / np.sqrt(4 * vs), differenced=False)
        two = [(self.SD_REL, True), (s, False)]
        a = mpt.sim_maet(self._pm(self.Q, two), self._pm(v_x, two),
                         truncation_sigmas=ts, verbose=False)
        b = mpt.sim_maet(self._pm(self.Q, [(c, False)]),
                         self._pm(v_x, [(c, False)]),
                         truncation_sigmas=ts, verbose=False)
        return float(a), float(b)

    @pytest.mark.parametrize("s, ts, expect_zero", [
        (158.0, 6, True),        # 3e-9 < floor exp(-18) = 1.5e-8
        (158.0, np.inf, False),  # 3e-9 > accuracy floor 1e-12
        (100.0, np.inf, True),   # 5e-22 < 1e-12
        (63.0, np.inf, True),    # 2e-54 < 1e-12
        (300.0, 6, False),       # 4e-3, well above either floor
    ])
    def test_pairing_equals_kernel_cov_under_truncation(self, s, ts,
                                                       expect_zero):
        a, b = self._pair(s, ts, self.Q + 700.0)
        if expect_zero:
            assert a == 0.0 and b == 0.0
        else:
            assert a > 0.0
            np.testing.assert_allclose(b, a, rtol=1e-12, atol=0.0)

    @pytest.mark.parametrize("normalize", ["cosine", "oneSidedDenom",
                                           "none"])
    @pytest.mark.parametrize("s, expect_zero", [(158.0, True),
                                                (300.0, False)])
    def test_scaled_identity_matches_scalar_sigma_sim(self, normalize, s,
                                                      expect_zero):
        x = self.Q + 700.0
        a = mpt.sim_maet(self._pm(self.Q, [(s, False)]),
                         self._pm(x, [(s, False)]),
                         normalize=normalize, truncation_sigmas=6,
                         verbose=False)
        C = np.eye(4) * s * s
        b = mpt.sim_maet(self._pm(self.Q, [(C, False)]),
                         self._pm(x, [(C, False)]),
                         normalize=normalize, truncation_sigmas=6,
                         verbose=False)
        if expect_zero:
            assert float(a) == 0.0 and float(b) == 0.0
        else:
            np.testing.assert_allclose(float(b), float(a), rtol=1e-12)

    def test_scaled_identity_matches_scalar_sigma_eval(self):
        s = 158.0
        dI = mpt.build_maet(self._pm(self.Q, [(s, False)]), verbose=False)
        dC = mpt.build_maet(self._pm(self.Q, [(np.eye(4) * s * s, False)]),
                            verbose=False)
        # Offsets 0, 300 inside the 6-sigma ball; 600, 900 outside it.
        X = np.stack([self.Q + d for d in (0.0, 300.0, 600.0, 900.0)],
                     axis=1)
        for norm in ("none", "gaussian"):
            vi = mpt.eval_maet(dI, X, normalize=norm, truncation_sigmas=6,
                               verbose=False)
            vc = mpt.eval_maet(dC, X, normalize=norm, truncation_sigmas=6,
                               verbose=False)
            np.testing.assert_allclose(vc, vi, rtol=1e-12, atol=0.0)
            assert vi[2] == 0.0 and vi[3] == 0.0 and vi[1] > 0.0

    def test_exhaustive_kernel_sum_branch_truncates(self):
        # dim = 4 with one centre: the bucket index is not worthwhile,
        # so the exhaustive branch runs; it must apply the same Q-ball
        # cutoff as the bucketed one.
        from mpt._kernel import gaussian_kernel_sum
        C = np.zeros((4, 1))
        k = 6.0
        u = np.ones((4, 1)) / 2.0                 # unit vector
        X = np.hstack([u * (k - 1e-6), u * (k + 1e-6)])
        v = gaussian_kernel_sum(C, np.ones(1), X, 1.0,
                                truncation_sigmas=k)
        np.testing.assert_allclose(v[0], np.exp(-0.5 * (k - 1e-6) ** 2),
                                   rtol=1e-12)
        assert v[1] == 0.0


class TestBareInnerProductScale:
    """``normalize='none'`` is on the canonical scale in the original
    coordinates.

    Whitening ``x = R y`` (``Sigma = R R^T``) carries the anisotropic
    kernel to the isotropic unit-sigma one, but the inner product is an
    integral, so the change of variables contributes the Jacobian
    ``det(Sigma)^(1/2)`` per attribute. Regression: the bare value
    omitted it, so ``Sigma = sigma**2 I`` of dimension ``d`` returned
    ``sigma**-d`` times the scalar-sigma value (at ``sigma = 158``,
    ``d = 4``: 2.95e-8 against 18.39). The factor cancels under
    ``'cosine'`` and ``'oneSidedDenom'``, which are pinned elsewhere.
    """

    SIG = 1.7

    @staticmethod
    def _direct_bare(cx, wx, cy, wy, Sigma):
        """Direct canonical-scale inner product of two sums of
        unnormalized anisotropic kernels (one ordered tuple each)."""
        d = len(cx[0])
        Sinv = np.linalg.inv(Sigma)
        pref = np.pi ** (d / 2) * np.sqrt(np.linalg.det(Sigma))
        s = 0.0
        for a, Wa in zip(cx, wx):
            for b, Wb in zip(cy, wy):
                dd = np.asarray(a, float) - np.asarray(b, float)
                s += Wa * Wb * np.exp(-0.25 * dd @ Sinv @ dd)
        return pref * s

    @pytest.mark.parametrize("r", [2, 3])
    def test_raw_single_multiset(self, r):
        P = np.array([0.0, 3.0, -3.0])[:r]
        Q = np.array([0.2, 3.4, -2.9])[:r]
        W = np.array([1.0, 0.8, 0.6])[:r]
        args = (False, False, 0.0, False)
        v_mat = mpt.sim_maet(P, W, Q, W, self.SIG**2 * np.eye(r), r,
                             *args, normalize="none", verbose=False)
        v_sca = mpt.sim_maet(P, W, Q, W, self.SIG, r,
                             *args, normalize="none", verbose=False)
        np.testing.assert_allclose(v_mat, v_sca, rtol=1e-12)

    def test_density_and_density_list(self):
        r = 3
        P = np.array([0.0, 3.0, -3.0])
        Q = np.array([0.2, 3.4, -2.9])
        W = np.array([1.0, 0.8, 0.6])

        def dens(p, s):
            return mpt.build_maet(p, W, s, r, False, False, 0.0, False,
                                  verbose=False)

        C = self.SIG**2 * np.eye(r)
        np.testing.assert_allclose(
            mpt.sim_maet(dens(P, C), dens(Q, C), normalize="none",
                         verbose=False),
            mpt.sim_maet(dens(P, self.SIG), dens(Q, self.SIG),
                         normalize="none", verbose=False),
            rtol=1e-12)
        np.testing.assert_allclose(
            mpt.sim_maet(dens(P, C), [dens(Q, C), dens(P, C)],
                         normalize="none", verbose=False),
            mpt.sim_maet(dens(P, self.SIG),
                         [dens(Q, self.SIG), dens(P, self.SIG)],
                         normalize="none", verbose=False),
            rtol=1e-12)

    def test_dedup_does_not_merge_different_covariances(self):
        """Two covariances whose whitened values coincide give different
        bare values; the canonical-form dedup must keep them apart."""
        r = 2
        W = np.ones(r)
        P = np.array([0.0, 1.0])
        out = []
        for s in (1.0, 2.0):
            dx = mpt.build_maet(s * P, W, s * s * np.eye(r), r, False,
                                False, 0.0, False, verbose=False)
            out.append(dx)
        vals = mpt.sim_maet(out, out, mode="pairwise", normalize="none",
                            verbose=False)
        np.testing.assert_allclose(vals[1] / vals[0], 2.0 ** r, rtol=1e-12)

    def test_multi_attribute(self):
        """Two matrix-sigma attributes (r = 3 and r = 2) tensored with a
        scalar one; raw and density forms."""
        rng = np.random.default_rng(7)
        N = 3
        P1x, P1y = rng.normal(size=(3, N)), rng.normal(size=(3, 1))
        P2x, P2y = rng.normal(size=(2, N)), rng.normal(size=(2, 1))
        Tx, Ty = np.array([[0.0, 0.5, 1.0]]), np.array([[0.3]])
        s1, s2, st = 0.9, 1.4, 0.4

        def ones(P):
            return np.ones_like(P)

        geom = ([3, 2, 1], [False] * 3, [False] * 3, [0.0] * 3,
                [False, False, True])
        px, wx = [P1x, P2x, Tx], [ones(P1x), ones(P2x), ones(Tx)]
        py, wy = [P1y, P2y, Ty], [ones(P1y), ones(P2y), ones(Ty)]
        sig_mat = [s1**2 * np.eye(3), s2**2 * np.eye(2), st]
        sig_sca = [s1, s2, st]
        v_mat = mpt.sim_maet(px, wx, py, wy, sig_mat, *geom,
                             normalize="none", verbose=False)
        v_sca = mpt.sim_maet(px, wx, py, wy, sig_sca, *geom,
                             normalize="none", verbose=False)
        np.testing.assert_allclose(v_mat, v_sca, rtol=1e-12)
        d_mat = [mpt.build_maet(p, w, sig_mat, *geom, verbose=False)
                 for p, w in ((px, wx), (py, wy))]
        d_sca = [mpt.build_maet(p, w, sig_sca, *geom, verbose=False)
                 for p, w in ((px, wx), (py, wy))]
        np.testing.assert_allclose(
            mpt.sim_maet(*d_mat, normalize="none", verbose=False),
            mpt.sim_maet(*d_sca, normalize="none", verbose=False),
            rtol=1e-12)
        # Entropy is unchanged (its log det term is added once).
        for dm, ds in zip(d_mat, d_sca):
            np.testing.assert_allclose(
                mpt.entropy_maet(dm, method="renyi2", verbose=False),
                mpt.entropy_maet(ds, method="renyi2", verbose=False),
                rtol=1e-12)

    def test_entropy_with_relative_point_mass_attribute(self):
        """A relative r = 1 attribute takes the Renyi-2 sub-density
        branch; the log det term must still be added exactly once."""
        rng = np.random.default_rng(11)
        P1 = rng.normal(size=(2, 4))
        T = np.array([[0.0, 0.3, 0.9, 1.4]])
        geom = ([2, 1], [False, True], [False, False], [0.0, 0.0],
                [False, True])
        w = [np.ones_like(P1), np.ones_like(T)]
        s = 0.6
        dm = mpt.build_maet([P1, T], w, [s * s * np.eye(2), 0.2], *geom,
                            verbose=False)
        ds = mpt.build_maet([P1, T], w, [s, 0.2], *geom, verbose=False)
        np.testing.assert_allclose(
            mpt.entropy_maet(dm, method="renyi2", verbose=False),
            mpt.entropy_maet(ds, method="renyi2", verbose=False),
            rtol=1e-12)

    def test_non_isotropic_against_direct(self):
        r = 3
        Sigma = _random_spd(r, rng=np.random.default_rng(3), scale=0.4)
        cx = [np.array([0.0, 1.0, 0.5]), np.array([-1.0, 0.0, 2.0])]
        cy = [np.array([0.1, 0.9, 0.55]), np.array([2.0, -1.0, 0.3])]
        got = mpt.sim_maet(
            [np.column_stack(cx)], [np.ones((r, 2))],
            [np.column_stack(cy)], [np.ones((r, 2))],
            [Sigma], [r], [False], [False], [0.0], [False],
            normalize="none", verbose=False)
        want = self._direct_bare(cx, [1.0, 1.0], cy, [1.0, 1.0], Sigma)
        np.testing.assert_allclose(got, want, rtol=1e-12)
        # Diagonal covariance, single-multiset form.
        D = np.diag([0.3, 1.1, 2.5])
        P, Q = cx[0], cy[0]
        got = mpt.sim_maet(P, np.ones(r), Q, np.ones(r), D, r,
                           False, False, 0.0, False,
                           normalize="none", verbose=False)
        want = self._direct_bare([P], [1.0], [Q], [1.0], D)
        np.testing.assert_allclose(got, want, rtol=1e-12)

    def test_swept_similarity(self):
        r = 2
        shapes = np.array([[0.0, 0.3, 0.2, 0.5],
                           [0.4, 0.1, 0.2, 0.0]])
        onsets = np.array([[0.0, 1.0, 2.0, 3.0]])
        pc = [shapes, onsets]
        wc = [np.ones((r, 4)), np.ones((1, 4))]
        pq = [np.array([[0.2], [0.25]]), np.array([[0.0]])]
        wq = [np.ones((r, 1)), np.ones((1, 1))]
        s = 0.3

        def run(sig, drop):
            return mpt.swept_similarity(
                pc, wc, pq, wq, [sig, 0.25], [r, 1], [False, False],
                [False, False], [0.0, 0.0], is_exch=[False, True],
                sweep={1: onsets.ravel()},
                align={1: "window" if drop else "both"},
                drop=[1] if drop else None, window={1: {"shape": "rect", "width": 0.5}},
                normalize="none", verbose=False)

        for drop in (True, False):
            np.testing.assert_allclose(run(s * s * np.eye(r), drop),
                                       run(s, drop), rtol=1e-12)
