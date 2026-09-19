"""Tests for the spatial cull on the relative periodic centres path.

Relative periodic was the one evaluation path with no cull: every tuple
was evaluated against every query. It admits one, because the quadratic
form bounds each wrapped coordinate. ``Q`` is the pairwise-wrap sum over
``r`` positions divided by ``r``, and the inner terms are non-negative,
so a pair inside the truncation threshold has every
``|wrap(D_k)| <= sqrt(r) k sigma``; on that region no inner pair can
wrap, so ``Q`` there is exactly ``w' (I - J/r) w``, whose inverse has
diagonal 2, tightening the box to ``sqrt(2) k sigma`` whatever ``r`` is.
The lattice is then pitched on the circle and its neighbour indices
taken modulo the bucket count.

The claim the tests have to catch failing is that the box is wide
enough: too narrow a box silently drops contributions, and does so by
amounts far below any tolerance chosen for other reasons. So the
comparisons here are against the untruncated evaluation at the
accuracy floor, where a dropped term shows up.

Mirror of MATLAB tests/test_periodic_cull.m.
"""

import numpy as np
import pytest

import mpt
from mpt._tensor.eval import (_rel_per_cull_half_width,
                              _rel_per_cull_worthwhile,
                              _truncated_kernel_sum_culled_rel_per)


PERIOD = 1200.0
SCALE = np.array([0.0, 200.0, 400.0, 500.0, 700.0, 900.0, 1100.0])
#: The width ``truncation_sigmas=inf`` resolves to: the kernel is below
#: the toolbox's 1e-12 parity floor beyond it.
K_FLOOR = float(np.sqrt(-2.0 * np.log(1e-12)))


@pytest.fixture
def fixed_chunk():
    prev = mpt.set_default(kernel_chunk_bytes=1 << 30)
    yield
    mpt.set_default(**prev)


def density(r, sigma, is_exch=True, is_per=True):
    p_attr = [SCALE[:, None]]
    specs = mpt.flat_specs(p_attr, r=r, rel=True, exch=is_exch)
    return mpt.build_maet(p_attr, None, specs=specs, sigma=[sigma],
                          is_per=[is_per], period=[PERIOD], verbose=False)


def grid(dim, n, rng):
    return rng.uniform(0.0, PERIOD, (dim, n))


class TestAgainstTheUntruncatedEvaluation:
    """At the accuracy floor the cull must lose nothing measurable."""

    @pytest.mark.parametrize('r', [2, 3, 4, 5])
    @pytest.mark.parametrize('is_exch', [True, False])
    def test_the_culled_sum_matches(self, fixed_chunk, r, is_exch):
        sigma = 15.0
        dens = density(r, sigma, is_exch)
        centres = np.asarray(mpt.maet_centres(dens)[0], dtype=float)
        dim = centres.shape[0]
        w_j = np.linspace(0.5, 1.5, centres.shape[1])
        x_q = grid(dim, 4000, np.random.default_rng(r))
        got = _truncated_kernel_sum_culled_rel_per(
            centres, w_j, x_q, sigma, r, PERIOD, K_FLOOR)
        # Untruncated reference, built directly from the definition.
        d_q = centres[:, :, None] - x_q[:, None, :]
        pos0 = d_q - PERIOD * np.floor(d_q / PERIOD + 0.5)
        q_form = np.sum(pos0 ** 2, axis=0)
        for i in range(dim):
            for j in range(i + 1, dim):
                delta = d_q[i] - d_q[j]
                delta = delta - PERIOD * np.floor(delta / PERIOD + 0.5)
                q_form = q_form + delta ** 2
        q_form = q_form / r
        want = w_j @ np.exp(-q_form / (2.0 * sigma * sigma))
        assert np.abs(got - want).max() < 1e-10

    @pytest.mark.parametrize('sigma', [2.0, 15.0, 60.0])
    def test_across_kernel_widths(self, fixed_chunk, sigma):
        """The box scales with sigma, and at the widest the cull should
        stand down rather than wrap onto itself."""
        r = 4
        dens = density(r, sigma)
        centres = np.asarray(mpt.maet_centres(dens)[0], dtype=float)
        w_j = np.ones(centres.shape[1])
        x_q = grid(centres.shape[0], 3000, np.random.default_rng(1))
        applicable = _rel_per_cull_worthwhile(
            centres.shape[0], centres.shape[1], x_q.shape[1], r, sigma,
            K_FLOOR, PERIOD)
        if not applicable:
            return
        got = _truncated_kernel_sum_culled_rel_per(
            centres, w_j, x_q, sigma, r, PERIOD, K_FLOOR)
        dense = np.asarray(mpt.eval_maet(dens, x_q, 'none',
                                         truncation_sigmas=np.inf,
                                         verbose=False)).ravel()
        assert np.abs(got - dense).max() < 1e-9


class TestThroughTheEntryPoint:

    @pytest.mark.parametrize('r', [3, 4])
    def test_the_default_path_agrees_with_the_untruncated_one(
            self, fixed_chunk, r):
        """What a user sees: the toolbox's own default truncation
        against the exact result, on a density that now takes the cull."""
        sigma = 15.0
        dens = density(r, sigma)
        x_q = grid(r - 1, 20000, np.random.default_rng(7))
        got = np.asarray(mpt.eval_maet(dens, x_q, 'none',
                                       verbose=False)).ravel()
        exact = np.asarray(mpt.eval_maet(dens, x_q, 'none',
                                         truncation_sigmas=np.inf,
                                         verbose=False)).ravel()
        assert np.abs(got - exact).max() < 1e-7

    def test_the_density_at_its_own_centres_is_at_least_one(
            self, fixed_chunk):
        """Each centre carries a kernel, so the density there cannot be
        below that kernel's own weight. Catches a cull that misses the
        query's own bucket."""
        dens = density(4, 15.0)
        centres = np.asarray(mpt.maet_centres(dens)[0], dtype=float)
        vals = np.asarray(mpt.eval_maet(dens, centres, 'none',
                                        verbose=False)).ravel()
        assert (vals >= 1.0 - 1e-9).all()

    def test_queries_outside_the_principal_period(self, fixed_chunk):
        """A query is reduced onto the circle, so an unreduced one gives
        the same value as its reduced image."""
        dens = density(4, 15.0)
        rng = np.random.default_rng(3)
        x_q = grid(3, 2000, rng)
        shifted = x_q + PERIOD * rng.integers(-3, 4, x_q.shape)
        a = np.asarray(mpt.eval_maet(dens, x_q, 'none', verbose=False))
        b = np.asarray(mpt.eval_maet(dens, shifted, 'none', verbose=False))
        assert np.allclose(a, b, rtol=0, atol=1e-9)


class TestTheHalfWidth:

    @pytest.mark.parametrize('r', [2, 3, 4, 5, 8])
    def test_the_tight_bound_is_independent_of_r(self, r):
        """sqrt(2) k sigma, from the inverse of I - J/r having diagonal
        2 whatever r is."""
        got = _rel_per_cull_half_width(r, 10.0, 6.0, 100000.0)
        assert np.isclose(got, np.sqrt(2.0) * 60.0)

    @pytest.mark.parametrize('r', [2, 3, 4, 5])
    def test_the_loose_bound_is_used_when_the_period_is_tight(self, r):
        """Where an inner pair could wrap, only the a-priori bound
        holds."""
        period = 4.0 * np.sqrt(r) * 60.0            # exactly at the edge
        got = _rel_per_cull_half_width(r, 10.0, 6.0, period)
        assert np.isclose(got, np.sqrt(r) * 60.0)

    def test_the_bound_actually_contains_the_region(self):
        """The property both bounds exist to guarantee, checked by
        sampling: nothing inside the threshold lies outside the box."""
        rng = np.random.default_rng(5)
        for r in (2, 3, 4, 5):
            dim = r - 1
            sigma, k = 10.0, 6.0
            half = _rel_per_cull_half_width(r, sigma, k, PERIOD)
            w = rng.uniform(-PERIOD / 2, PERIOD / 2, (dim, 400000))
            q = np.sum(w ** 2, axis=0)
            for i in range(dim):
                for j in range(i + 1, dim):
                    delta = w[i] - w[j]
                    delta = delta - PERIOD * np.floor(delta / PERIOD + 0.5)
                    q = q + delta ** 2
            q = q / r
            inside = q <= (k * sigma) ** 2
            assert np.abs(w[:, inside]).max() <= half + 1e-9


class TestTheGate:

    def test_it_declines_when_the_window_fills_the_circle(self):
        """A kernel wide relative to the period cannot be culled: the
        three-bucket span would wrap onto itself."""
        assert not _rel_per_cull_worthwhile(3, 840, 10000, 4, 400.0,
                                            6.0, PERIOD)

    def test_it_declines_when_there_are_too_few_tuples(self):
        """Below 3**dim tuples the neighbourhood lookup costs more than
        the tuples it saves."""
        assert not _rel_per_cull_worthwhile(3, 20, 10000, 4, 15.0,
                                            6.0, PERIOD)

    def test_it_accepts_the_case_it_exists_for(self):
        assert _rel_per_cull_worthwhile(3, 840, 10000, 4, 15.0, 6.0,
                                        PERIOD)

    def test_a_declined_gate_still_gives_the_right_answer(self,
                                                          fixed_chunk):
        """The fallback is the dense path, so a wide kernel must still
        evaluate correctly."""
        sigma = 400.0
        dens = density(4, sigma)
        x_q = grid(3, 500, np.random.default_rng(9))
        got = np.asarray(mpt.eval_maet(dens, x_q, 'none', verbose=False))
        exact = np.asarray(mpt.eval_maet(dens, x_q, 'none',
                                         truncation_sigmas=np.inf,
                                         verbose=False))
        assert np.abs(got - exact).max() < 1e-7
