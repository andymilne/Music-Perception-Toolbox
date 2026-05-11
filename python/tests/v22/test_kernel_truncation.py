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


def test_truncation_inf_is_exact(small_problem):
    """truncation_sigmas=inf must produce bit-identical output to exact."""
    C, wJ, X, sigma = small_problem
    v_inf = gaussian_kernel_sum(C, wJ, X, sigma, truncation_sigmas=math.inf)
    ref = _ref_kernel_sum(C, wJ, X, sigma)
    # Bit-identical at FP level (same compute path).
    assert np.max(np.abs(v_inf - ref)) < 1e-12 * np.max(np.abs(ref))


# ---------------------------------------------------------------------
# Periodic mode falls through to exact
# ---------------------------------------------------------------------

def test_periodic_ignores_truncation_for_now():
    rng = np.random.default_rng(7)
    dim, nJ, nQ = 1, 40, 5
    C = rng.uniform(0, 1200, (dim, nJ))
    wJ = rng.uniform(0.5, 1.5, nJ)
    X = rng.uniform(0, 1200, (dim, nQ))
    sigma = 30.0
    v_trunc = gaussian_kernel_sum(
        C, wJ, X, sigma, truncation_sigmas=6,
        is_per=True, period=1200.0,
    )
    ref = _ref_kernel_sum(C, wJ, X, sigma, is_per=True, period=1200.0)
    # Periodic mode currently falls through to exact: bit-identical.
    assert np.max(np.abs(v_trunc - ref)) < 1e-12 * np.max(np.abs(ref))


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
    d = mpt.get_defaults()
    assert d["truncation_sigmas"] == math.inf
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
        "truncation_sigmas": math.inf,
        "kernel_precision": "double",
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
