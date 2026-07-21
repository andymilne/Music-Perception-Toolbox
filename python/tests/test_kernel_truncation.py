"""Tests for the Gaussian-kernel-sum helper and global-defaults machinery.

Covers:

- Exact path (``truncation_sigmas=inf``) is FP-bit-identical to a
  direct broadcast-subtract-exp-sum reference.
- Truncated path agrees with exact to better than ``exp(-k^2/2)``
  relative error.
- Rel-mode quadratic form correctness (with and without truncation).
- Periodic mode falls through to exact regardless of
  ``truncation_sigmas``.
- Single precision degrades to ~1e-7 relative.
- Defaults are honoured by the helper when called without explicit
  keywords.
- Defaults round-trip (set → save prev → set new → restore from prev).
"""

import math

import numpy as np
import pytest

import mpt
from mpt._kernel import gaussian_kernel_sum


# ---------------------------------------------------------------------
# Reference: a plain broadcast-subtract-exp-sum
# ---------------------------------------------------------------------

def _ref_kernel_sum(C, wJ, X, sigma, *, is_rel=False, r=0,
                    is_per=False, period=0.0):
    """Direct reference: no truncation, no chunking, double precision."""
    dim, nJ = C.shape
    nQ = X.shape[1]
    D = C[:, :, None] - X[:, None, :]
    if is_per:
        D = D - period * np.floor(D / period + 0.5)
    if is_rel:
        Q = np.sum(D * D, axis=0) - np.sum(D, axis=0) ** 2 / r
    else:
        Q = np.sum(D * D, axis=0)
    E = np.exp(-Q / (2 * sigma ** 2))
    return wJ @ E


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_defaults_each_test():
    """Ensure no cross-test default leakage."""
    yield
    mpt.reset_defaults()


@pytest.fixture
def small_problem():
    rng = np.random.default_rng(0)
    dim, nJ, nQ = 2, 60, 8
    C = rng.uniform(0, 100, (dim, nJ))
    wJ = rng.uniform(0.5, 1.5, nJ)
    X = rng.uniform(0, 100, (dim, nQ))
    sigma = 5.0
    return C, wJ, X, sigma


# ---------------------------------------------------------------------
# Exact path
# ---------------------------------------------------------------------

def test_exact_matches_reference(small_problem):
    C, wJ, X, sigma = small_problem
    v = gaussian_kernel_sum(C, wJ, X, sigma)
    ref = _ref_kernel_sum(C, wJ, X, sigma)
    assert np.max(np.abs(v - ref)) < 1e-12 * np.max(np.abs(ref))


def test_exact_rel_matches_reference(small_problem):
    C, wJ, X, sigma = small_problem
    # Rel mode requires (dim) centres = (r - 1); for r=3 we have dim=2.
    v = gaussian_kernel_sum(C, wJ, X, sigma, is_rel=True, r=3)
    ref = _ref_kernel_sum(C, wJ, X, sigma, is_rel=True, r=3)
    assert np.max(np.abs(v - ref)) < 1e-12 * np.max(np.abs(ref))


# ---------------------------------------------------------------------
# Truncated path
# ---------------------------------------------------------------------

@pytest.mark.parametrize("k", [4, 5, 6, 8])
def test_truncated_abs_within_bound(small_problem, k):
    C, wJ, X, sigma = small_problem
    v_trunc = gaussian_kernel_sum(C, wJ, X, sigma, truncation_sigmas=k)
    ref = _ref_kernel_sum(C, wJ, X, sigma)
    # Discarded centres' per-centre kernel value is at most exp(-k^2/2).
    # Cumulative error scales with the discarded mass: bound at
    # sum(|wJ|) * exp(-k^2/2). For this problem sum(|wJ|) ≈ 60.
    # Use 200x for safety (factor ~3 above the cumulative upper bound).
    bound = 200 * math.exp(-k ** 2 / 2)
    rel = np.max(np.abs(v_trunc - ref) / np.maximum(np.abs(ref), 1e-30))
    assert rel < bound, f"k={k}: rel={rel:.3e}, bound={bound:.3e}"


@pytest.mark.parametrize("k", [4, 5, 6, 8])
def test_truncated_rel_within_bound(small_problem, k):
    C, wJ, X, sigma = small_problem
    v_trunc = gaussian_kernel_sum(C, wJ, X, sigma, is_rel=True, r=3,
                                   truncation_sigmas=k)
    ref = _ref_kernel_sum(C, wJ, X, sigma, is_rel=True, r=3)
    bound = 200 * math.exp(-k ** 2 / 2)
    rel = np.max(np.abs(v_trunc - ref) / np.maximum(np.abs(ref), 1e-30))
    assert rel < bound


def test_truncation_inf_floors_to_accuracy_eps(small_problem):
    """truncation_sigmas=inf resolves to the accuracy-floor width, so it
    agrees with the exact reference to the accuracy floor (1e-12), and
    becomes bit-exact under a maximal-accuracy override."""
    from mpt._defaults import accuracy_floor_context
    C, wJ, X, sigma = small_problem
    ref = _ref_kernel_sum(C, wJ, X, sigma)
    # Default: inf means the 1e-12 accuracy floor. Agreement to that
    # floor where the reference is non-negligible.
    v_inf = gaussian_kernel_sum(C, wJ, X, sigma, truncation_sigmas=math.inf)
    scale = np.max(np.abs(ref))
    assert np.max(np.abs(v_inf - ref)) < 1e-11 * scale
    # Maximal-accuracy override: inf becomes effectively exhaustive, so
    # the result matches the untruncated reference to FP round-off.
    with accuracy_floor_context(1e-300):
        v_exact = gaussian_kernel_sum(
            C, wJ, X, sigma, truncation_sigmas=math.inf)
    assert np.max(np.abs(v_exact - ref)) < 1e-12 * scale


# ---------------------------------------------------------------------
# Periodic mode falls through to exact
# ---------------------------------------------------------------------

def test_periodic_truncates_on_circle():
    """Periodic 1-D kernel sums truncate on the circle: the result
    approximates the exact (un-truncated) sum to the accuracy floor set
    by ``truncation_sigmas``, and tightens as that floor is lowered."""
    rng = np.random.default_rng(7)
    dim, nJ, nQ = 1, 40, 5
    C = rng.uniform(0, 1200, (dim, nJ))
    wJ = rng.uniform(0.5, 1.5, nJ)
    X = rng.uniform(0, 1200, (dim, nQ))
    sigma = 30.0
    ref = _ref_kernel_sum(C, wJ, X, sigma, is_per=True, period=1200.0)
    peak = np.max(np.abs(ref))
    v6 = gaussian_kernel_sum(
        C, wJ, X, sigma, truncation_sigmas=6,
        is_per=True, period=1200.0,
    )
    # 6 sigma: circular truncation is active (drops terms beyond the
    # window, so the result departs from exact) but bounded by the
    # ~1e-8 floor.
    err6 = np.max(np.abs(v6 - ref))
    assert 1e-11 * peak < err6 < 1e-6 * peak
    v_inf = gaussian_kernel_sum(
        C, wJ, X, sigma, truncation_sigmas=np.inf,
        is_per=True, period=1200.0,
    )
    # Accuracy floor: matches exact to ~1e-12.
    assert np.max(np.abs(v_inf - ref)) < 1e-11 * peak


# ---------------------------------------------------------------------
# Precision
# ---------------------------------------------------------------------

def test_single_precision_within_bound(small_problem):
    C, wJ, X, sigma = small_problem
    v_single = gaussian_kernel_sum(C, wJ, X, sigma, kernel_precision='single')
    ref = _ref_kernel_sum(C, wJ, X, sigma)
    rel = np.max(np.abs(v_single - ref) / np.maximum(np.abs(ref), 1e-30))
    assert rel < 1e-5, f"single rel error {rel:.3e} exceeds 1e-5"


def test_single_returns_float64(small_problem):
    C, wJ, X, sigma = small_problem
    v = gaussian_kernel_sum(C, wJ, X, sigma, kernel_precision='single')
    assert v.dtype == np.float64


# ---------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------

def test_empty_queries():
    C = np.array([[0.0, 1.0]])
    wJ = np.array([1.0, 1.0])
    X = np.zeros((1, 0))
    v = gaussian_kernel_sum(C, wJ, X, 1.0, truncation_sigmas=6)
    assert v.shape == (0,)


def test_single_centre():
    C = np.array([[5.0]])
    wJ = np.array([2.0])
    X = np.array([[5.0, 5.5, 10.0]])
    sigma = 1.0
    v = gaussian_kernel_sum(C, wJ, X, sigma, truncation_sigmas=6)
    ref = _ref_kernel_sum(C, wJ, X, sigma)
    assert np.max(np.abs(v - ref)) < 1e-12 * max(np.max(np.abs(ref)), 1e-30)


def test_query_outside_centres_bbox():
    """Truncation should still work when query is far from any centre."""
    C = np.array([[0.0, 1.0, 2.0]])
    wJ = np.array([1.0, 1.0, 1.0])
    X = np.array([[100.0]])  # very far
    sigma = 1.0
    v = gaussian_kernel_sum(C, wJ, X, sigma, truncation_sigmas=6)
    # All centres are >> 6σ away → output should be ~0.
    assert v[0] < 1e-30


# ---------------------------------------------------------------------
# Regression: maxWin == 1 edge case in 1-D truncated path.
#
# A bug existed in the equivalent MATLAB function where, at
# maxWin == 1, the (nQ, 1) index matrix triggered MATLAB's "vector
# source + vector index → match source orientation" rule and
# returned the wrong shape, causing a (nQ, nQ) outer-broadcast OOM.
# NumPy's indexing rule is uniform (always matches the index shape),
# so Python was never affected. These tests pin that down so a
# future refactor can't introduce the same bug.
# ---------------------------------------------------------------------

def test_truncated_1d_maxwin_one():
    """sigma chosen so each window contains at most one centre."""
    C = np.array([[0.0, 200.0, 400.0, 500.0, 700.0, 900.0, 1100.0]])
    wJ = np.ones(7)
    sigma = 10.0  # threshold = 6*sigma = 60 < min source gap (100)
    nQ = 120000
    X = np.linspace(0, 1200, nQ).reshape(1, -1)
    v = gaussian_kernel_sum(C, wJ, X, sigma, truncation_sigmas=6)
    ref = _ref_kernel_sum(C, wJ, X, sigma)
    assert v.shape == (nQ,)
    assert np.max(np.abs(v - ref)) < 200 * np.exp(-18) * max(np.max(np.abs(ref)), 1e-30)


def test_truncated_1d_demo_expTensorPlots_config_3():
    """Exact inputs from the MATLAB demo's failing config 3 call.

    sigma_eff = 10/sqrt(2) (m=2 partition block), query points are
    means of pairs from the upper triangle of a 481×481 grid. This
    is the precise scenario that hit the MATLAB OOM.
    """
    C = np.array([[0.0, 200.0, 400.0, 500.0, 700.0, 900.0, 1100.0]])
    wJ = np.ones(7)
    sigma_eff = 10.0 / np.sqrt(2)
    res = 481
    x_1d = np.linspace(0, 1200, res)
    Ga, Gb = np.meshgrid(x_1d, x_1d)
    mask = np.triu(np.ones((res, res), dtype=bool))
    Xu = np.stack([Ga[mask], Gb[mask]], axis=0)
    mean_x = Xu.sum(axis=0).reshape(1, -1) / 2   # (1, 115921)
    v = gaussian_kernel_sum(C, wJ, mean_x, sigma_eff, truncation_sigmas=6)
    ref = _ref_kernel_sum(C, wJ, mean_x, sigma_eff)
    assert v.shape == (mean_x.shape[1],)
    assert np.max(np.abs(v - ref)) < 200 * np.exp(-18) * max(np.max(np.abs(ref)), 1e-30)


def test_bad_inputs_raise():
    C = np.array([[0.0, 1.0]])
    wJ = np.array([1.0, 1.0])
    X = np.array([[0.5]])
    with pytest.raises(ValueError, match="positive"):
        gaussian_kernel_sum(C, wJ, X, 1.0, truncation_sigmas=-1)
    with pytest.raises(ValueError, match="kernel_precision"):
        gaussian_kernel_sum(C, wJ, X, 1.0, kernel_precision='quad')
    with pytest.raises(ValueError, match="rel mode requires r"):
        gaussian_kernel_sum(C, wJ, X, 1.0, is_rel=True)


# ---------------------------------------------------------------------
# Defaults machinery
# ---------------------------------------------------------------------

def test_factory_defaults():
    # The conftest baseline pins the exact path; reset to observe the
    # true factory values.
    mpt.reset_defaults()
    d = mpt.get_defaults()
    assert d["truncation_sigmas"] == 6.0
    assert d["kernel_precision"] == "double"


def test_set_default_and_get_default():
    mpt.set_default(truncation_sigmas=6)
    assert mpt.get_default("truncation_sigmas") == 6.0
    assert mpt.get_default("kernel_precision") == "double"


def test_set_default_returns_previous_values():
    mpt.set_default(truncation_sigmas=4)
    prev = mpt.set_default(truncation_sigmas=8, kernel_precision='single')
    assert prev == {"truncation_sigmas": 4.0, "kernel_precision": "double"}
    # Restore via the prev dict round-trip.
    mpt.set_default(**prev)
    assert mpt.get_default("truncation_sigmas") == 4.0
    assert mpt.get_default("kernel_precision") == "double"


def test_reset_defaults():
    mpt.set_default(truncation_sigmas=6, kernel_precision='single')
    mpt.reset_defaults()
    assert mpt.get_defaults() == {
        "truncation_sigmas": 6.0,
        "kernel_precision": "double",
        "show_hints": True,
        "kernel_chunk_bytes": "auto",
    }


def test_get_default_unknown_raises():
    with pytest.raises(KeyError, match="Unknown"):
        mpt.get_default("nonsense")


def test_set_default_unknown_raises():
    with pytest.raises(ValueError, match="Unknown"):
        mpt.set_default(nonsense=42)


def test_set_default_bad_value_raises():
    with pytest.raises(ValueError, match="positive"):
        mpt.set_default(truncation_sigmas=0)
    with pytest.raises(ValueError, match="positive"):
        mpt.set_default(truncation_sigmas=-1)
    with pytest.raises(ValueError, match="'single'"):
        mpt.set_default(kernel_precision='quad')


def test_kernel_consults_defaults(small_problem):
    """When called without explicit keywords, the helper uses defaults."""
    C, wJ, X, sigma = small_problem
    # With default truncation_sigmas=inf, must match reference exactly.
    v1 = gaussian_kernel_sum(C, wJ, X, sigma)
    ref = _ref_kernel_sum(C, wJ, X, sigma)
    assert np.max(np.abs(v1 - ref)) < 1e-12 * np.max(np.abs(ref))

    # After setting default to 6, the same call uses truncation.
    mpt.set_default(truncation_sigmas=6)
    v2 = gaussian_kernel_sum(C, wJ, X, sigma)
    rel = np.max(np.abs(v2 - ref) / np.maximum(np.abs(ref), 1e-30))
    assert rel < 200 * math.exp(-18)   # k=6 cumulative bound


def test_per_call_overrides_default(small_problem):
    """Per-call keyword wins over the global default."""
    C, wJ, X, sigma = small_problem
    mpt.set_default(truncation_sigmas=4)   # default lax
    # But this call explicitly asks for exact:
    v = gaussian_kernel_sum(C, wJ, X, sigma, truncation_sigmas=math.inf)
    ref = _ref_kernel_sum(C, wJ, X, sigma)
    assert np.max(np.abs(v - ref)) < 1e-12 * np.max(np.abs(ref))


# ---------------------------------------------------------------------
# show_hints flag
# ---------------------------------------------------------------------

def test_show_hints_factory_default_is_true():
    mpt.reset_defaults()
    assert mpt.get_default("show_hints") is True


def test_show_hints_validation():
    with pytest.raises(ValueError, match="show_hints"):
        mpt.set_default(show_hints="yes")


# ---------------------------------------------------------------------
# Centralized truncation-scalar identity
# ---------------------------------------------------------------------
#
# truncation_floor / truncation_radius / truncation_ip_sqdist are the
# single home for the exp(-k^2/2) value floor and its two distance
# manifestations. These pin the identity between them and the resolved
# inf -> accuracy-floor policy, so a drift in any one call site would
# surface here.


def test_truncation_floor_inf_is_accuracy_floor_eps():
    """The exact sentinel resolves to the finite accuracy-floor width,
    whose value floor is exactly the accuracy-floor eps (1e-12 by
    default) --- not "nothing discarded"."""
    from mpt._defaults import truncation_floor, accuracy_floor_eps
    assert truncation_floor(math.inf) == accuracy_floor_eps()


def test_truncation_floor_none_takes_default():
    """None resolves to the global default width, not to inf."""
    from mpt._defaults import truncation_floor
    mpt.reset_defaults()
    k = mpt.get_default("truncation_sigmas")
    assert truncation_floor(None) == math.exp(-0.5 * k * k)


def test_truncation_radius_is_k_sigma():
    """Density-kernel radius: distance k*sigma where G(.;sigma) hits the floor."""
    from mpt._defaults import truncation_radius
    assert truncation_radius(6.0, 2.0) == 6.0 * 2.0


def test_truncation_ip_sqdist_has_factor_two():
    """Inner-product kernel (sigma*sqrt2 wide) hits the same value floor at
    twice the squared distance of the density kernel."""
    from mpt._defaults import truncation_radius, truncation_ip_sqdist
    r = truncation_radius(6.0, 2.0)
    assert truncation_ip_sqdist(6.0, 2.0) == 2.0 * r ** 2


def test_ip_sqdist_and_floor_describe_the_same_cutoff():
    """A pair at exactly the IP-kernel cutoff distance has kernel value
    equal to the shared floor: this is the identity the two helpers
    must jointly satisfy."""
    from mpt._defaults import truncation_floor, truncation_ip_sqdist
    sigma, k = 3.0, 6.0
    sqdist = truncation_ip_sqdist(k, sigma)          # |d|^2 at the cutoff
    ip_kernel_value = math.exp(-sqdist / (4.0 * sigma ** 2))
    assert abs(ip_kernel_value - truncation_floor(k)) < 1e-15


def test_nested_contraction_honours_inf_truncation_floor():
    """Regression: the nested-contraction inner product must truncate at
    the 1e-12 floor for the exact sentinel, like every other path ---
    previously it skipped truncation entirely for inf/None, leaving
    sub-floor kernel values in the sum."""
    import numpy as np
    from mpt._tensor._nested_contraction import _trunc
    from mpt._defaults import accuracy_floor_eps

    # A kernel tensor with one value straddling the 1e-12 floor.
    K = np.array([[[1.0, 1e-20]]])
    _trunc(K, 1.0, math.inf)
    assert K[0, 0, 1] == 0.0, "inf must floor at the accuracy floor, not skip"
    # And a value comfortably above the floor is kept.
    K2 = np.array([[[1.0, 1e-6]]])
    _trunc(K2, 1.0, math.inf)
    assert K2[0, 0, 1] == 1e-6
    # The floor used is exactly the accuracy-floor eps.
    K3 = np.array([[[accuracy_floor_eps() * 0.9]]])
    _trunc(K3, 1.0, math.inf)
    assert K3[0, 0, 0] == 0.0
