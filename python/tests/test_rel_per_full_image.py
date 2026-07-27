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
