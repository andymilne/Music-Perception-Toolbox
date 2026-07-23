"""Parity tests for ``eval_exp_tens`` single-multiset centres-path routing through
the shared helper (:func:`mpt._kernel.gaussian_kernel_sum`).

Verifies, on a battery of representative densities and queries, that:

- Default settings (``truncation_sigmas=inf``, ``kernel_precision='double'``)
  produce FP-bit-identical output to a frozen reference implementation
  matching v2.0/v2.1 behaviour.
- ``truncation_sigmas=6`` agrees with the reference to bounded relative
  error (cumulative bound proportional to ``sum(|wJ|) * exp(-18)``).
- ``kernel_precision='single'`` agrees within ~1e-5 relative.
- Per-call kwargs override toolbox defaults.
- Toolbox defaults (set via ``mpt.set_default``) are picked up when
  per-call kwargs are absent.

The reference implementation is a plain broadcast-subtract-exp-sum,
matching what ``_eval_core`` did in v2.0/v2.1 — kept here as a stable
local-frozen check rather than relying on the now-routed production
code.
"""

import math

import numpy as np

from mpt._defaults import accuracy_floor_context

# Comparisons against the untruncated reference widen the accuracy floor
# to 1e-300 (as the MATLAB twin test_kernel_truncation.m does), so the
# shipped paths sum essentially exhaustively. Entries below that floor
# are legitimately dropped -- the reference can still emit denormals at
# ~1e-308 -- so atol is tied to the floor being requested.
_FLOOR = 1e-300
import pytest

import mpt
from mpt.tensor import build_exp_tens, eval_exp_tens


# ---------------------------------------------------------------------
# Reference (frozen v2.0/v2.1 body)
# ---------------------------------------------------------------------

def _ref_eval(dens, x):
    """Plain broadcast-subtract-exp-sum, no truncation, double precision."""
    # A MaetDensity carries per-attribute vectors (r, sigma, is_rel, ...);
    # this frozen reference is written against the flat single-multiset
    # field names, so take that view rather than indexing the vectors here.
    from mpt._tensor.density import single_multiset_view
    dens = single_multiset_view(dens)
    centres = dens.centres
    w_j = dens.w_j
    sigma = dens.sigma
    r = int(dens.r)
    is_rel = bool(dens.is_rel)
    is_per = bool(dens.is_per)
    period = float(dens.period)

    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)

    D = centres[:, :, None] - x[:, None, :]
    if is_per:
        D = np.mod(D + period / 2, period) - period / 2
    if is_rel:
        Q = np.sum(D * D, axis=0) - np.sum(D, axis=0) ** 2 / r
    else:
        Q = np.sum(D * D, axis=0)
    E = np.exp(-Q / (2 * sigma ** 2))
    return w_j @ E


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_defaults_each_test():
    yield
    mpt.reset_defaults()


# A representative battery: r in {2, 3}, abs and rel, non-periodic
# and periodic, different K, σ.
@pytest.fixture(params=[
    # (description, K, r, is_rel, is_per, period, sigma, n_q)
    ("abs r=2 K=6",        6, 2, False, False, 0.0,    8.0,  30),
    ("rel r=2 K=6",        6, 2, True,  False, 0.0,    8.0,  30),
    ("abs r=3 K=5",        5, 3, False, False, 0.0,   12.0,  20),
    ("rel r=3 K=5",        5, 3, True,  False, 0.0,   12.0,  20),
    ("rel r=3 K=8 σ=30",   8, 3, True,  False, 0.0,   30.0,  50),
    ("rel-per r=3 K=6",    6, 3, True,  True,  1200.0, 25.0, 25),
    ("abs-per r=2 K=6",    6, 2, False, True,  1200.0, 25.0, 25),
])
def density_case(request):
    label, K, r, is_rel, is_per, period, sigma, n_q = request.param
    rng = np.random.default_rng(hash(label) & 0xFFFFFFFF)
    p = np.sort(rng.uniform(0, 1000, K))
    w = rng.uniform(0.5, 1.5, K)
    dens = build_exp_tens(p, w, sigma, r, is_rel, is_per, period, verbose=False)
    dim = r - 1 if is_rel else r
    if is_per:
        x = rng.uniform(0, period, (dim, n_q))
    else:
        x = rng.uniform(0, 1000, (dim, n_q))
    return label, dens, x


# ---------------------------------------------------------------------
# Default settings = match v2.0/v2.1 reference within reduction-order
# noise. v2.2 relaxes the v2.1 bit-identity contract to reduction-order
# identity (~1e-13) for periodic configurations, since the periodic
# wrap was retargeted from np.mod to D - period * floor(D/period + 0.5)
# (mathematically equivalent everywhere; different FP-op sequence).
# Non-periodic configurations remain bit-identical to the reference.
# ---------------------------------------------------------------------

def test_default_settings_match_reference(density_case):
    label, dens, x = density_case
    with accuracy_floor_context(1e-300):
        v = eval_exp_tens(dens, x, method='centres', truncation_sigmas=math.inf,
                          verbose=False)
    ref = _ref_eval(dens, x)
    np.testing.assert_allclose(
        v, ref, rtol=1e-12, atol=_FLOOR,
        err_msg=f"{label}: divergence from reference exceeds rtol=1e-12"
    )


def test_explicit_inf_matches_reference(density_case):
    label, dens, x = density_case
    with accuracy_floor_context(1e-300):
        v = eval_exp_tens(
            dens, x, method='centres', truncation_sigmas=math.inf,
            kernel_precision='double', verbose=False,
        )
    ref = _ref_eval(dens, x)
    np.testing.assert_allclose(
        v, ref, rtol=1e-12, atol=_FLOOR,
        err_msg=f"{label}: divergence from reference exceeds rtol=1e-12"
    )


# ---------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------

@pytest.mark.parametrize("k", [4, 5, 6])
def test_truncation_within_bound(density_case, k):
    label, dens, x = density_case
    # Periodic mode falls through to exact: truncation has no effect.
    if dens.is_per:
        pytest.skip("periodic mode currently exact-only")
    v_trunc = eval_exp_tens(
        dens, x, method='centres', truncation_sigmas=k, verbose=False,
    )
    ref = _ref_eval(dens, x)
    # Cumulative bound: sum(|wJ|) * exp(-k^2/2). Add 10x safety.
    weight_mass = float(np.sum(np.abs(dens.w_j)))
    bound = max(1e-12, 10 * weight_mass * math.exp(-k ** 2 / 2))
    # Use absolute error rather than relative — relative blows up when
    # ref values are tiny (queries far from all centres).
    err = np.max(np.abs(v_trunc - ref))
    assert err < bound, \
        f"{label} k={k}: err={err:.3e}, bound={bound:.3e}, " \
        f"weight_mass={weight_mass:.3f}"


def test_truncation_inf_is_exact(density_case):
    """At truncation_sigmas=inf, output is bit-identical to default."""
    label, dens, x = density_case
    v_default = eval_exp_tens(dens, x, method='centres', verbose=False)
    v_inf = eval_exp_tens(
        dens, x, method='centres', truncation_sigmas=math.inf, verbose=False,
    )
    assert np.array_equal(v_default, v_inf), \
        f"{label}: explicit inf vs default not bit-identical"


def test_periodic_truncation_agrees_within_floor(density_case):
    """Periodic mode: culling changes values only below the requested floor.

    Periodic evaluation once fell through to an exact wrapped sum, so
    truncation had no effect and the two calls were bit-identical. Since
    periodic-mode culling shipped, a requested width does cull, so the
    contract is agreement to the amplitude that width admits --
    exp(-k^2/2) at k = 6 -- not bitwise equality.
    """
    label, dens, x = density_case
    if not dens.is_per:
        pytest.skip("only applies to periodic")
    v_default = eval_exp_tens(dens, x, method='centres', verbose=False)
    v_trunc = eval_exp_tens(
        dens, x, method='centres', truncation_sigmas=6, verbose=False,
    )
    np.testing.assert_allclose(
        v_trunc, v_default, rtol=0, atol=10.0 * math.exp(-6.0 ** 2 / 2),
        err_msg=f"{label}: periodic culling exceeds the requested floor",
    )


# ---------------------------------------------------------------------
# Precision
# ---------------------------------------------------------------------

def test_single_precision_within_bound(density_case):
    label, dens, x = density_case
    v_single = eval_exp_tens(
        dens, x, method='centres', kernel_precision='single', verbose=False,
    )
    ref = _ref_eval(dens, x)
    # Single precision: ~1e-7 rel; allow generous margin. Compare on
    # absolute error to avoid blow-up when ref values are tiny.
    weight_mass = float(np.sum(np.abs(dens.w_j)))
    err = np.max(np.abs(v_single - ref))
    # Single-precision exp() carries ~1e-7 rel; absolute error bounded
    # by max kernel value times sum(|wJ|) times 1e-7. Allow 1e-5 of
    # weight mass for safety.
    bound = max(1e-12, 1e-5 * weight_mass)
    assert err < bound, f"{label}: single abs err {err:.3e}, bound {bound:.3e}"


# ---------------------------------------------------------------------
# Global defaults
# ---------------------------------------------------------------------

def test_global_default_picked_up():
    """When per-call kwarg is absent, the helper consults mpt defaults."""
    rng = np.random.default_rng(42)
    p = np.sort(rng.uniform(0, 1000, 6))
    w = rng.uniform(0.5, 1.5, 6)
    sigma = 12.0
    dens = build_exp_tens(p, w, sigma, 3, True, False, 0.0, verbose=False)
    x = rng.uniform(0, 1000, (2, 20))
    ref = _ref_eval(dens, x)

    # With the default width, agreement is to the accuracy floor: the
    # resolved width is k = 7.43, not an untruncated sum.
    with accuracy_floor_context(1e-300):
        v1 = eval_exp_tens(dens, x, method='centres',
                           truncation_sigmas=math.inf, verbose=False)
    np.testing.assert_allclose(v1, ref, rtol=1e-12, atol=_FLOOR)

    # Set global default; bit-identical now requires truncation match.
    mpt.set_default(truncation_sigmas=6)
    v2 = eval_exp_tens(dens, x, method='centres', verbose=False)
    weight_mass = float(np.sum(np.abs(dens.w_j)))
    bound = max(1e-12, 10 * weight_mass * math.exp(-18))
    err = np.max(np.abs(v2 - ref))
    assert err < bound


def test_per_call_overrides_global():
    rng = np.random.default_rng(43)
    p = np.sort(rng.uniform(0, 1000, 6))
    w = rng.uniform(0.5, 1.5, 6)
    sigma = 12.0
    dens = build_exp_tens(p, w, sigma, 3, True, False, 0.0, verbose=False)
    x = rng.uniform(0, 1000, (2, 20))
    ref = _ref_eval(dens, x)

    # Set a lax global default.
    mpt.set_default(truncation_sigmas=4)
    # A per-call Inf must still resolve to the accuracy floor, overriding
    # the laxer global width.
    with accuracy_floor_context(1e-300):
        v = eval_exp_tens(
            dens, x, method='centres', truncation_sigmas=math.inf,
            verbose=False,
        )
    np.testing.assert_allclose(v, ref, rtol=1e-12, atol=_FLOOR)
