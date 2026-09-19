"""Tests for the padded bucket lattice in the truncated kernel sums.

The spatial cull pitches a bucket lattice on the tuple centres and, for
each query, visits the centres in its own bucket's 3**dim neighbourhood.
The lattice carries one empty bucket of margin on every face, so a
neighbour of an occupied-region bucket can never fall off it. That is
what lets the neighbourhood be computed as arithmetic on lattice
strides rather than as a bounds-tested coordinate array, and it puts
the whole weight of the arrangement on one claim: a bucket outside the
occupied region contributes nothing, whether it is skipped for lying
out of bounds or visited and found empty.

The queries that exercise the margin are the ones whose bucket is
clamped to the lattice edge -- everything outside the centres' bounding
box, which for a density on a scale is most of a plotting grid. What is
pinned here is that the culled sum equals a direct sum over the centres
inside the truncation ball, for queries far outside the box, on its
faces, and within it, in one, two and three dimensions and in both
absolute and relative mode.

Mirror of MATLAB tests/test_bucket_padding.m.
"""

import numpy as np
import pytest

import mpt
from mpt._kernel import _truncated_kernel_sum
from mpt._tensor.eval import _truncated_kernel_sum_culled


SIGMA = 15.0
K_SIGMA = 6.0
SCALE = np.array([0.0, 200.0, 400.0, 500.0, 700.0, 900.0, 1100.0])


def centres_of(r, is_rel, is_exch=True):
    p_attr = [SCALE[:, None]]
    specs = mpt.flat_specs(p_attr, r=r, rel=is_rel, exch=is_exch)
    dens = mpt.build_maet(p_attr, None, specs=specs, sigma=[SIGMA],
                          is_per=[False], period=[1200.0], verbose=False)
    return np.asarray(mpt.maet_centres(dens)[0], dtype=float)


def direct(centres, w_j, x_q, is_rel, r):
    """The quantity the cull is an optimisation of: every centre within
    the truncation ball of the query, and no other."""
    threshold2 = (K_SIGMA * SIGMA) ** 2
    inv_2s2 = 1.0 / (2.0 * SIGMA * SIGMA)
    out = np.empty(x_q.shape[1])
    for q in range(x_q.shape[1]):
        dq = centres - x_q[:, q:q + 1]
        if is_rel:
            q_form = (dq * dq).sum(axis=0) - dq.sum(axis=0) ** 2 / r
        else:
            q_form = (dq * dq).sum(axis=0)
        near = q_form <= threshold2
        out[q] = np.sum(w_j[near] * np.exp(-q_form[near] * inv_2s2))
    return out


def probe_points(centres, rng):
    """Queries inside the centres' box, on its faces, and well outside
    it in every direction. The outside ones are the point of the test:
    their buckets clamp to the lattice edge, so their neighbourhoods
    reach into the margin."""
    dim = centres.shape[0]
    lo = centres.min(axis=1)[:, None]
    hi = centres.max(axis=1)[:, None]
    span = np.maximum(hi - lo, 1.0)
    inside = lo + rng.uniform(0.0, 1.0, (dim, 60)) * span
    on_face = np.concatenate([np.tile(lo, (1, 10)), np.tile(hi, (1, 10))],
                             axis=1)
    outside = np.concatenate([lo - rng.uniform(0.01, 3.0, (dim, 60)) * span,
                              hi + rng.uniform(0.01, 3.0, (dim, 60)) * span],
                             axis=1)
    # A few just beyond the truncation radius of the box, the distance at
    # which the answer turns from nonzero to exactly zero.
    just_out = hi + K_SIGMA * SIGMA * rng.uniform(0.9, 1.1, (dim, 20))
    return np.concatenate([inside, on_face, outside, just_out], axis=1)


CASES = [(1, False, 'absolute, dim 1'),
         (2, False, 'absolute, dim 2'),
         (3, False, 'absolute, dim 3'),
         (2, True, 'relative, dim 1'),
         (3, True, 'relative, dim 2'),
         (4, True, 'relative, dim 3')]


@pytest.mark.parametrize('r,is_rel,label', CASES)
class TestCulledSumMatchesADirectSum:

    def test_culled_centres_path(self, r, is_rel, label):
        centres = centres_of(r, is_rel)
        w_j = np.linspace(0.5, 1.5, centres.shape[1])
        x_q = probe_points(centres, np.random.default_rng(0))
        got = _truncated_kernel_sum_culled(centres, w_j, x_q, SIGMA,
                                           is_rel, r, K_SIGMA)
        assert np.allclose(got, direct(centres, w_j, x_q, is_rel, r),
                           rtol=0, atol=1e-12)

    def test_bucketed_kernel_sum(self, r, is_rel, label):
        centres = centres_of(r, is_rel)
        w_j = np.linspace(0.5, 1.5, centres.shape[1])
        x_q = probe_points(centres, np.random.default_rng(1))
        inv_2s2 = 1.0 / (2.0 * SIGMA * SIGMA)
        got = _truncated_kernel_sum(centres, w_j, x_q, SIGMA, is_rel, r,
                                    K_SIGMA, inv_2s2)
        assert np.allclose(got, direct(centres, w_j, x_q, is_rel, r),
                           rtol=0, atol=1e-12)


class TestDegenerateLattices:
    """A lattice can be one bucket wide, or one bucket in total, and the
    margin has to hold there too."""

    def test_one_occupied_bucket(self):
        """Every centre in a single bucket: the lattice is the margin
        and one cell."""
        rng = np.random.default_rng(4)
        centres = rng.uniform(-5.0, 5.0, (2, 40))
        w_j = np.ones(40)
        x_q = np.array([[-1e4, 0.0, 20.0, 1e4],
                        [-1e4, 0.0, -20.0, 1e4]])
        got = _truncated_kernel_sum_culled(centres, w_j, x_q, SIGMA,
                                           False, 2, K_SIGMA)
        assert np.allclose(got, direct(centres, w_j, x_q, False, 2),
                           rtol=0, atol=1e-12)

    def test_centres_collinear_in_one_axis(self):
        """Zero extent in the second coordinate: that axis holds exactly
        one occupied bucket."""
        centres = np.vstack([np.linspace(0.0, 1200.0, 13),
                             np.zeros(13)])
        w_j = np.ones(13)
        rng = np.random.default_rng(2)
        x_q = np.vstack([rng.uniform(-3000.0, 3000.0, 400),
                         rng.uniform(-300.0, 300.0, 400)])
        got = _truncated_kernel_sum_culled(centres, w_j, x_q, SIGMA,
                                           False, 2, K_SIGMA)
        assert np.allclose(got, direct(centres, w_j, x_q, False, 2),
                           rtol=0, atol=1e-12)

    def test_every_query_outside_the_box(self):
        centres = np.vstack([np.linspace(0.0, 100.0, 60)] * 3)
        w_j = np.ones(60)
        rng = np.random.default_rng(3)
        x_q = rng.uniform(5000.0, 6000.0, (3, 200))
        got = _truncated_kernel_sum_culled(centres, w_j, x_q, SIGMA,
                                           False, 3, K_SIGMA)
        assert np.array_equal(got, np.zeros(200))


class TestThroughThePublicEntryPoint:

    def test_a_plotting_grid_agrees_with_the_untruncated_result(self):
        """The demo case: a grid far wider than the scale it plots, so
        most of its points sit outside the centres' box."""
        p_attr = [SCALE[:, None]]
        specs = mpt.flat_specs(p_attr, r=2, rel=False, exch=True)
        dens = mpt.build_maet(p_attr, None, specs=specs, sigma=[SIGMA],
                              is_per=[False], period=[1200.0],
                              verbose=False)
        g = np.linspace(-2400.0, 3600.0, 400)
        ga, gb = np.meshgrid(g, g)
        pts = np.vstack([ga.ravel(), gb.ravel()])
        truncated = np.asarray(mpt.eval_maet(dens, pts, 'none',
                                             verbose=False)).ravel()
        exact = np.asarray(mpt.eval_maet(dens, pts, 'none',
                                         truncation_sigmas=np.inf,
                                         verbose=False)).ravel()
        assert np.abs(truncated - exact).max() < 1e-7
