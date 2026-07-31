"""Tests for the full-image relative-periodic inner product.

The relative-periodic inner product marginalises a rigid common shift.
Taking that average over a kernel carrying every periodic image yields
the lattice-sum (full-image) measure exactly; taking it over a
nearest-image kernel yields a third measure, which is neither the
full-image one nor the single-image closed form and departs from both as
sigma/period grows.

The reference below is built independently of the toolbox: a direct sum
over the rank-(r-1) image lattice of the unwrapped relative quadratic
form, evaluated over ordered tuples of distinct indices.
"""
import itertools

import numpy as np
import pytest

from mpt._tensor import cosine as _c
import mpt._tensor._mobius_inner as _mobius_inner

PERIOD = 1200.0


def _lattice_cos(p, w, q, v, sigma, r, n_max=6):
    """Cosine under the full-image lattice measure.

    Q0(x) = sum_i x_i^2 - (sum_i x_i)^2 / r, summed over translates
    x = delta + P n for n in Z^r / Z.1, represented by fixing the last
    component at zero.
    """
    tX = list(itertools.permutations(range(len(p)), r))
    tY = list(itertools.permutations(range(len(q)), r))
    cX = np.array([[p[i] for i in t] for t in tX], dtype=float)
    cY = np.array([[q[i] for i in t] for t in tY], dtype=float)
    WX = np.array([np.prod([w[i] for i in t]) for t in tX], dtype=float)
    WY = np.array([np.prod([v[i] for i in t]) for t in tY], dtype=float)

    def ip(cA, WA, cB, WB):
        d = cA[:, None, :] - cB[None, :, :]
        acc = np.zeros(d.shape[:2], dtype=float)
        for n in itertools.product(range(-n_max, n_max + 1), repeat=r - 1):
            x = d + np.array(list(n) + [0], dtype=float) * PERIOD
            Q = (x ** 2).sum(-1) - x.sum(-1) ** 2 / r
            acc += np.exp(-Q / (4.0 * sigma ** 2))
        return float(WA @ acc @ WB)

    return ip(cX, WX, cY, WY) / np.sqrt(
        ip(cX, WX, cX, WX) * ip(cY, WY, cY, WY)
    )


def _taugrid_cos(p, w, q, v, sigma, r):
    """Cosine from the tau-grid route, with the spectral branch off."""
    def ip(a, wa, b, wb):
        Px = np.asarray(a, float).reshape(-1, 1)
        Wx = np.asarray(wa, float).reshape(-1, 1)
        Py = np.asarray(b, float).reshape(-1, 1)
        Wy = np.asarray(wb, float).reshape(-1, 1)
        prev = _mobius_inner._SPECTRAL_IP_ENABLED
        _mobius_inner._SPECTRAL_IP_ENABLED = False
        try:
            out = _mobius_inner._rel_inner_batched(
                Px, Wx, Py, Wy, float(sigma), int(r), True, PERIOD,
                truncation_sigmas=float('inf'),
            )
            return float(np.asarray(out)[0, 0])
        finally:
            _mobius_inner._SPECTRAL_IP_ENABLED = prev

    return ip(p, w, q, v) / np.sqrt(ip(p, w, p, w) * ip(q, v, q, v))


def _spectral_cos(p, w, q, v, sigma, r):
    def ip(a, wa, b, wb):
        Px = np.asarray(a, float).reshape(-1, 1)
        Wx = np.asarray(wa, float).reshape(-1, 1)
        Py = np.asarray(b, float).reshape(-1, 1)
        Wy = np.asarray(wb, float).reshape(-1, 1)
        out = _mobius_inner._spectral_rel_inner_matrix(
            Px, Wx, Py, Wy, float(sigma), int(r), True, PERIOD)
        return None if out is None else float(out[0, 0])

    xy = ip(p, w, q, v)
    if xy is None:
        return None
    xx = ip(p, w, p, w)
    yy = ip(q, v, q, v)
    return None if (xx is None or yy is None) else xy / np.sqrt(xx * yy)


@pytest.fixture(scope="module")
def multisets():
    rng = np.random.default_rng(7)
    K = 5
    p = np.sort(rng.uniform(0, PERIOD, K))
    q = np.sort(rng.uniform(0, PERIOD, K))
    return p, np.ones(K), q, np.ones(K)


# ---------------------------------------------------------------------
# The tau-grid computes the full-image measure at every sigma/P
# ---------------------------------------------------------------------

@pytest.mark.parametrize("r", [2, 3, 4])
@pytest.mark.parametrize("sigma_over_P", [0.02, 0.05, 0.08, 0.10, 0.20, 0.30])
def test_taugrid_matches_lattice(multisets, r, sigma_over_P):
    p, w, q, v = multisets
    sigma = sigma_over_P * PERIOD
    got = _taugrid_cos(p, w, q, v, sigma, r)
    ref = _lattice_cos(p, w, q, v, sigma, r)
    assert abs(got - ref) < 1e-11, f"{got} vs {ref}"


# ---------------------------------------------------------------------
# The two routes now agree, so gate disagreement is no longer a
# correctness matter
# ---------------------------------------------------------------------

@pytest.mark.parametrize("r", [2, 3, 4])
@pytest.mark.parametrize("sigma_over_P", [0.05, 0.10, 0.20, 0.30])
def test_spectral_and_taugrid_agree(multisets, r, sigma_over_P):
    p, w, q, v = multisets
    sigma = sigma_over_P * PERIOD
    spec = _spectral_cos(p, w, q, v, sigma, r)
    if spec is None:
        pytest.skip("spectral branch declined this cell")
    grid = _taugrid_cos(p, w, q, v, sigma, r)
    assert abs(spec - grid) < 1e-11, f"{spec} vs {grid}"


# ---------------------------------------------------------------------
# The image count is driven by the caller's accuracy floor, and is zero
# in the range musical work normally occupies
# ---------------------------------------------------------------------

@pytest.mark.parametrize("sigma_over_P", [0.001, 0.0125, 0.02, 0.03])
def test_image_count_is_zero_at_musical_sigma_over_P(sigma_over_P):
    # Zero images means the full-image and nearest-image kernels are the
    # same object, so the measure upgrade costs nothing here.
    n = _mobius_inner._rel_per_image_count(sigma_over_P * PERIOD, PERIOD, 6.0)
    assert n == 0


@pytest.mark.parametrize("sigma_over_P", [0.10, 0.20, 0.30])
def test_image_count_positive_where_measures_diverge(sigma_over_P):
    assert _mobius_inner._rel_per_image_count(sigma_over_P * PERIOD, PERIOD, 6.0) >= 1


def test_image_count_rises_with_accuracy_demanded():
    # A tighter kernel floor needs at least as many images.
    sigma = 0.20 * PERIOD
    loose = _mobius_inner._rel_per_image_count(sigma, PERIOD, 4.0)
    tight = _mobius_inner._rel_per_image_count(sigma, PERIOD, float('inf'))
    assert tight >= loose


def test_image_count_monotone_in_sigma_over_P():
    counts = [_mobius_inner._rel_per_image_count(s * PERIOD, PERIOD, 6.0)
              for s in (0.02, 0.05, 0.10, 0.20, 0.30, 0.50)]
    assert counts == sorted(counts)


@pytest.mark.parametrize("sigma, period", [
    (float('nan'), PERIOD),
    (100.0, 0.0),
    (100.0, -1.0),
    (float('inf'), PERIOD),
])
def test_image_count_degenerate_inputs_return_zero(sigma, period):
    assert _mobius_inner._rel_per_image_count(sigma, period, 6.0) == 0


# ---------------------------------------------------------------------
# Evaluation carries the same measure as the inner product
# ---------------------------------------------------------------------
#
# The tests above compare inner products. Evaluation is the other half:
# eval_exp_tens returns the density at a set of query points, and it must
# return the full-image density, not the nearest-image approximation to
# it. The two coincide as sigma/period tends to zero and part company as
# it grows, so the cells below run from the musical range up to
# sigma/period = 0.30, well past the point where the nearest-image
# kernel ceases to be positive-definite.
#
# The reference is the same independent lattice sum used above, applied
# to a single density rather than a pair.
#
# Agreement is measured against the peak of the reference, not entry by
# entry. These densities span many orders of magnitude across a set of
# query points, and at a query far from every tuple both sides return
# numbers that are zero for any purpose; an entrywise relative test
# there compares noise with noise.


def _lattice_eval(p, w, x, sigma, r, n_max=8):
    """Full-image density at the query points, by direct lattice sum.

    Q0(x) = sum_i x_i^2 - (sum_i x_i)^2 / r over the r slots, summed over
    translates x + P n for n in Z^r / Z.1, the quotient represented by
    fixing the zeroth component at zero. The query carries r - 1 reduced
    coordinates, so a zeroth slot at the origin is prepended.
    """
    tuples = list(itertools.permutations(range(len(p)), r))
    C = np.array([[p[i] for i in t] for t in tuples], dtype=float)
    W = np.array([np.prod([w[i] for i in t]) for t in tuples], dtype=float)
    x = np.asarray(x, dtype=float)
    x_full = np.vstack([np.zeros((1, x.shape[1])), x])
    out = np.zeros(x.shape[1])
    for shift in itertools.product(range(-n_max, n_max + 1), repeat=r - 1):
        n = np.array((0,) + shift, dtype=float)
        d = C[:, :, None] - x_full[None, :, :] + PERIOD * n[None, :, None]
        Q = np.sum(d * d, axis=1) - np.sum(d, axis=1) ** 2 / r
        out = out + W @ np.exp(-Q / (2 * sigma ** 2))
    return out


def _eval_case(r, sigma_over_P, n_q=10, K=5):
    import mpt
    rng = np.random.default_rng(3 + r)
    p = np.sort(rng.uniform(0, PERIOD, K))
    w = np.ones(K)
    x = rng.uniform(0, PERIOD, (r - 1, n_q))
    sigma = sigma_over_P * PERIOD
    dens = mpt.build_exp_tens(p, w, sigma, r, True, True, PERIOD, verbose=False)
    ref = _lattice_eval(p, w, x, sigma, r)
    return mpt, dens, x, ref


@pytest.mark.parametrize("r", [2, 3, 4])
@pytest.mark.parametrize("sigma_over_P", [0.02, 0.05, 0.10, 0.20, 0.30])
def test_eval_matches_lattice_untruncated(r, sigma_over_P):
    """With truncation off, evaluation reproduces the lattice measure to
    reduction-order noise at every sigma/period."""
    import math
    from mpt._defaults import accuracy_floor_context
    mpt, dens, x, ref = _eval_case(r, sigma_over_P)
    with accuracy_floor_context(1e-300):
        got = mpt.eval_exp_tens(dens, x, truncation_sigmas=math.inf,
                                verbose=False)
    err = np.max(np.abs(got - ref)) / ref.max()
    assert err < 1e-12, f"r={r}, sigma/P={sigma_over_P}: {err:.3e}"


@pytest.mark.parametrize("r", [2, 3, 4])
@pytest.mark.parametrize("sigma_over_P", [0.02, 0.05, 0.10, 0.20, 0.30])
def test_eval_matches_lattice_at_default_truncation(r, sigma_over_P):
    """At the default 6-sigma cutoff the same comparison must hold to the
    truncation floor rather than to machine precision. The bound is the
    floor with an order of magnitude of headroom, since the discarded
    mass accumulates over tuples and images."""
    mpt, dens, x, ref = _eval_case(r, sigma_over_P)
    got = mpt.eval_exp_tens(dens, x, verbose=False)
    err = np.max(np.abs(got - ref)) / ref.max()
    assert err < 1.5e-7, f"r={r}, sigma/P={sigma_over_P}: {err:.3e}"


def _nearest_image_eval(p, w, x, sigma, r):
    """Density under the nearest-image kernel: the within-tuple
    differences are wrapped to [-P/2, P/2) and the quadratic formed from
    them. This is the cheap approximation the toolbox is entitled to use
    only while sigma/period is small."""
    tuples = list(itertools.permutations(range(len(p)), r))
    C = np.array([[p[i] for i in t] for t in tuples], dtype=float)
    W = np.array([np.prod([w[i] for i in t]) for t in tuples], dtype=float)
    x = np.asarray(x, dtype=float)
    x_full = np.vstack([np.zeros((1, x.shape[1])), x])
    D = C[:, :, None] - x_full[None, :, :]
    Q = np.zeros(D.shape[::2])
    for i in range(r):
        for j in range(i + 1, r):
            d = D[:, i, :] - D[:, j, :]
            d = np.mod(d + PERIOD / 2, PERIOD) - PERIOD / 2
            Q = Q + d * d
    return W @ np.exp(-(Q / r) / (2 * sigma ** 2))


def test_eval_departs_from_the_nearest_image_kernel_as_sigma_grows():
    """The measure under test is not the nearest-image one: pin the
    difference so a silent reversion to the cheap kernel fails.

    Below the threshold the two are indistinguishable, which is what
    entitles the toolbox to the cheap kernel there; well above it they
    differ by an appreciable fraction of the peak.
    """
    import math
    from mpt._defaults import accuracy_floor_context
    r = 3
    gaps = {}
    for sigma_over_P in (0.02, 0.30):
        mpt, dens, x, ref = _eval_case(r, sigma_over_P)
        rng = np.random.default_rng(3 + r)
        p = np.sort(rng.uniform(0, PERIOD, 5))
        near = _nearest_image_eval(p, np.ones(5), x, sigma_over_P * PERIOD, r)
        with accuracy_floor_context(1e-300):
            got = mpt.eval_exp_tens(dens, x, truncation_sigmas=math.inf,
                                    verbose=False)
        gaps[sigma_over_P] = float(np.max(np.abs(got - near)) / ref.max())
    assert gaps[0.02] < 1e-6, gaps
    assert gaps[0.30] > 1e-3, gaps


# ---------------------------------------------------------------------
# Non-periodic relative: the translation window must cover its support
# ---------------------------------------------------------------------
#
# In non-periodic relative mode every pair marginalises a translation
# over a window of shared width, positioned per pair. The support of the
# cross integrand runs from (min_y - max_x) to (max_y - min_x), so its
# midpoint is the midrange offset. Centring the window on the weighted
# mean offset instead displaces it: on ordinary random data with
# non-uniform weights the two differ by over 100 cents, against a margin
# of 8 sigma, and the far end of the support is clipped.
#
# The lost mass is the extreme pairs' contribution, so the error is
# data-dependent, survives grid refinement, and reached 4.8e-4 -- four
# orders above the truncation floor -- on 8 of 25 random draws at
# r = 2, K = 40. It was invisible in ordinary use because the spectral
# branch runs first for 2 <= r <= 4 and is accurate; the dense route
# carries it whenever that branch stands down, which includes every
# cancellation-ratio request.
#
# Direct enumeration is the arbiter here: it evaluates the tuple sum
# without any translation window at all.


def _nonper_case(K=40, decay_x=(0.0, 6.0), decay_y=(6.0, 0.0),
                 extent=1200.0):
    """Values evenly spaced over EXTENT with weights decaying towards a
    chosen end of each side.

    Constructed rather than drawn, so the property under test holds in
    both languages: MATLAB's RandStream and numpy's generator do not
    produce the same numbers from the same seed, so a seeded draw that
    displaces the window in one language need not do so in the other.
    Opposite decays put the weighted-mean offset 835 cents from the
    midrange offset, against a margin of 8 sigma.
    """
    p = np.linspace(0.0, extent, K)
    q = np.linspace(0.0, extent, K)
    wp = np.exp(-np.linspace(decay_x[0], decay_x[1], K))
    wq = np.exp(-np.linspace(decay_y[0], decay_y[1], K))
    return p, wp, q, wq


@pytest.mark.parametrize("K, r", [(40, 2), (20, 2), (12, 3), (10, 4)])
def test_nonper_dense_matches_direct_enumeration(K, r):
    """The dense translation-grid route must agree with direct
    enumeration to the truncation floor. Under a window centred on the
    weighted mean this construction errs by 2.1e-6 at r = 2, K = 40."""
    import mpt
    import mpt._tensor._mobius_inner as _mi
    p, wp, q, wq = _nonper_case(K=K)
    ref = mpt.cos_sim_exp_tens(p, wp, q, wq, 6.0, r, 1, 0, 0.0,
                               method="bulger", verbose=False)
    spectral_was = _mi._SPECTRAL_IP_ENABLED
    _mi._SPECTRAL_IP_ENABLED = False
    try:
        got = mpt.cos_sim_exp_tens(p, wp, q, wq, 6.0, r, 1, 0, 0.0,
                                   method="mobius", verbose=False)
    finally:
        _mi._SPECTRAL_IP_ENABLED = spectral_was
    assert abs(got - ref) < 1.5e-8, f"K={K} r={r}: {abs(got - ref):.3e}"


def test_nonper_window_offsets_differ_by_more_than_the_margin():
    """Pin the mechanism, not just the symptom: on this construction the
    weighted-mean offset sits far outside the window's own margin, so a
    window placed there cannot cover the support."""
    import mpt._tensor._mobius_inner as _mi
    p, wp, q, wq = _nonper_case()
    mid_offset = 0.5 * (q.max() + q.min()) - 0.5 * (p.max() + p.min())
    mean_offset = ((q * wq).sum() / wq.sum()) - ((p * wp).sum() / wp.sum())
    margin = _mi._rel_window_margin(6.0) * 6.0
    assert abs(mean_offset - mid_offset) > 10.0 * margin
