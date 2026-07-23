"""Tests for ``entropy_exp_tens(..., method='renyi2')``.

Verifies the analytical Rényi-2 entropy

    H_2 = -log_b(<T,T> / Z²)

against direct numerical integration of (T/Z)² across the four single-multiset
modes and several MA configurations. Also checks the API contract:
``method`` argument validation, migration error on the removed
``normalize`` kwarg, and the
single-multiset/MA dispatch.
"""
import numpy as np
import pytest

from mpt import entropy_exp_tens
from mpt.tensor import build_exp_tens, eval_exp_tens


# Periodic-mode tests can match the orbit value to FP precision because
# the integration domain [0, P)^d is exact and the grid matches the
# wrapped support exactly. Non-periodic tests use a wider grid and
# tolerate quadrature error — the orbit answer is the analytical
# ground truth so the test really verifies grid → analytical
# convergence.
TOL_FP = 1e-10
TOL_GRID = 5e-2  # generous: discretisation, not orbit precision


def _grid_renyi2_single_multiset(T, n_per_dim, ax_range):
    """Direct grid evaluation of -log2(∫(T/Z)² dx) for an single-multiset tensor."""
    # A MaetDensity carries per-attribute vectors; this helper is written
    # against a single multiset, so read the flat view for r/is_rel/is_per.
    from mpt._tensor.density import single_multiset_view
    Tq = T                      # keep the original for eval_exp_tens
    T = single_multiset_view(T)
    if T.is_per:
        ax = np.linspace(0, T.period, n_per_dim, endpoint=False)
        dx = T.period / n_per_dim
    else:
        a, b = ax_range
        ax = np.linspace(a, b, n_per_dim)
        dx = (b - a) / (n_per_dim - 1)
    if T.is_rel:
        # Relative single-multiset tensor lives on dim r-1 (one translation removed).
        # Build a (r-1)-D mesh.
        d = T.r - 1
    else:
        d = T.r
    if d == 0:
        # Degenerate (r=1 rel): density is a constant, H2=0.
        return 0.0
    if d == 1:
        X = ax[None, :]
        vol = dx
    else:
        mesh = np.meshgrid(*([ax] * d), indexing='ij')
        X = np.stack([m.ravel() for m in mesh], axis=0)
        vol = dx ** d
    t = eval_exp_tens(Tq, X, verbose=False)
    Z = float(t.sum() * vol)
    if Z <= 0:
        raise RuntimeError("grid Z is non-positive")
    return -float(np.log2(((t / Z) ** 2).sum() * vol))


# ---- single-multiset cases: all four modes at moderate r and K -----------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_renyi2_single_multiset_abs_per_matches_grid(seed):
    rng = np.random.default_rng(seed)
    P = 1200.0
    sigma = 50.0
    r, K = 2, 5
    p = rng.uniform(0, P, K)
    w = rng.uniform(0.5, 1.5, K)
    T = build_exp_tens(p, w, sigma, r, False, True, P, verbose=False)
    H2 = entropy_exp_tens(T, method='renyi2')
    H2_grid = _grid_renyi2_single_multiset(T, 600, None)
    assert abs(H2 - H2_grid) < TOL_FP


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_renyi2_sa_rel_per_matches_grid(seed):
    rng = np.random.default_rng(seed)
    P = 1200.0
    sigma = 50.0
    r, K = 2, 5
    p = rng.uniform(0, P, K)
    w = rng.uniform(0.5, 1.5, K)
    T = build_exp_tens(p, w, sigma, r, True, True, P, verbose=False)
    H2 = entropy_exp_tens(T, method='renyi2')
    H2_grid = _grid_renyi2_single_multiset(T, 600, None)
    assert abs(H2 - H2_grid) < TOL_FP


@pytest.mark.parametrize("seed", [0, 1])
def test_renyi2_sa_abs_nonper_matches_grid(seed):
    rng = np.random.default_rng(seed)
    sigma = 50.0
    r, K = 2, 5
    p = rng.uniform(-200, 200, K)
    w = rng.uniform(0.5, 1.5, K)
    T = build_exp_tens(p, w, sigma, r, False, False, 0.0, verbose=False)
    H2 = entropy_exp_tens(T, method='renyi2')
    H2_grid = _grid_renyi2_single_multiset(T, 1000, (-1000, 1000))
    assert abs(H2 - H2_grid) < TOL_GRID


@pytest.mark.parametrize("seed", [0, 1])
def test_renyi2_sa_rel_nonper_matches_grid(seed):
    rng = np.random.default_rng(seed)
    sigma = 50.0
    r, K = 2, 5
    p = rng.uniform(-200, 200, K)
    w = rng.uniform(0.5, 1.5, K)
    T = build_exp_tens(p, w, sigma, r, True, False, 0.0, verbose=False)
    H2 = entropy_exp_tens(T, method='renyi2')
    H2_grid = _grid_renyi2_single_multiset(T, 4000, (-1000, 1000))
    assert abs(H2 - H2_grid) < TOL_GRID


# ---- single-multiset at r >= 3 — the regime where grid-Shannon strains -------


def test_renyi2_sa_high_r_works_where_grid_would_fail():
    """At r=4 abs_per, the per-attribute grid path needs n_per_dim^4
    evaluations. Renyi-2 is closed-form so doesn't care."""
    rng = np.random.default_rng(0)
    P = 1200.0
    sigma = 50.0
    r, K = 4, 7
    p = rng.uniform(0, P, K)
    w = rng.uniform(0.5, 1.5, K)
    T = build_exp_tens(p, w, sigma, r, False, True, P, verbose=False)
    H2 = entropy_exp_tens(T, method='renyi2')
    # Sanity bounds: bounded above by log_2(P^r) (uniform on the
    # r-torus) and below by ~0 for typical configs at this sigma.
    assert np.isfinite(H2)
    assert H2 <= np.log2(P ** r) + 1e-9
    assert H2 > 0


# ---- MA cases ----------------------------------------------------


def test_renyi2_ma_two_attr_periodic_matches_grid():
    rng = np.random.default_rng(0)
    P = 1200.0
    N = 5
    p_attr = [rng.uniform(0, P, (1, N)), rng.uniform(0, 1.0, (1, N))]
    w = [rng.uniform(0.5, 1.5, (1, N)), rng.uniform(0.5, 1.5, (1, N))]
    dens = build_exp_tens(
        p_attr, w, [50.0, 0.05], [1, 1], 
        [False, False], [True, True], [P, 1.0], verbose=False,
    )
    H2 = entropy_exp_tens(dens, method='renyi2')

    # Direct grid: each attribute contributes 1 dimension (r=1 abs)
    ax0 = np.linspace(0, P, 200, endpoint=False)
    ax1 = np.linspace(0, 1.0, 200, endpoint=False)
    mesh = np.meshgrid(ax0, ax1, indexing='ij')
    X = np.stack([m.ravel() for m in mesh], axis=0)
    t = eval_exp_tens(dens, X, verbose=False)
    vol = (P / 200) * (1.0 / 200)
    Z = float(t.sum() * vol)
    H2_grid = -np.log2(float(((t / Z) ** 2).sum() * vol))
    assert abs(H2 - H2_grid) < TOL_FP


def test_renyi2_ma_mixed_r_matches_grid():
    """Two attributes: pitch with r=2 K=3 abs_per; phase with r=1 K=1
    abs_per. Density dim = 2 + 1 = 3."""
    rng = np.random.default_rng(0)
    P = 1200.0
    N = 4
    p_attr = [rng.uniform(0, P, (3, N)), rng.uniform(0, 1.0, (1, N))]
    w = [rng.uniform(0.5, 1.5, (3, N)), rng.uniform(0.5, 1.5, (1, N))]
    dens = build_exp_tens(
        p_attr, w, [50.0, 0.05], [2, 1], 
        [False, False], [True, True], [P, 1.0], verbose=False,
    )
    H2 = entropy_exp_tens(dens, method='renyi2')

    ax0 = np.linspace(0, P, 100, endpoint=False)
    ax1 = np.linspace(0, 1.0, 100, endpoint=False)
    mesh = np.meshgrid(ax0, ax0, ax1, indexing='ij')
    X = np.stack([m.ravel() for m in mesh], axis=0)
    t = eval_exp_tens(dens, X, verbose=False)
    vol = (P / 100) ** 2 * (1.0 / 100)
    Z = float(t.sum() * vol)
    H2_grid = -np.log2(float(((t / Z) ** 2).sum() * vol))
    assert abs(H2 - H2_grid) < TOL_FP


def test_renyi2_ma_relative_attribute():
    """One periodic relative pitch attribute, one periodic absolute
    phase attribute."""
    rng = np.random.default_rng(0)
    P = 1200.0
    N = 5
    p_attr = [rng.uniform(0, P, (3, N)), rng.uniform(0, 1.0, (1, N))]
    w = [rng.uniform(0.5, 1.5, (3, N)), rng.uniform(0.5, 1.5, (1, N))]
    dens = build_exp_tens(
        p_attr, w, [50.0, 0.05], [2, 1], 
        [True, False], [True, True], [P, 1.0], verbose=False,
    )
    H2 = entropy_exp_tens(dens, method='renyi2')
    assert np.isfinite(H2)


# ---- API contract tests ------------------------------------------


def test_renyi2_normalize_kwarg_raises_migration():
    """Legacy ``normalize`` kwarg was removed in v2.2; passing it
    (with any value) raises a migration-error TypeError pointing to
    the four-method API."""
    rng = np.random.default_rng(0)
    P = 1200.0
    T = build_exp_tens(
        rng.uniform(0, P, 5), rng.uniform(0.5, 1.5, 5),
        50.0, 2, False, True, P, verbose=False,
    )
    with pytest.raises(TypeError, match="'normalize'.*removed in v2.2"):
        entropy_exp_tens(T, method='renyi2', normalize=False)
    with pytest.raises(TypeError, match="'normalize'.*removed in v2.2"):
        entropy_exp_tens(T, method='renyi2', normalize=True)


def test_invalid_method_rejected():
    rng = np.random.default_rng(0)
    P = 1200.0
    T = build_exp_tens(
        rng.uniform(0, P, 5), rng.uniform(0.5, 1.5, 5),
        50.0, 2, False, True, P, verbose=False,
    )
    with pytest.raises(ValueError, match="method must be"):
        entropy_exp_tens(T, method='bogus')


def test_renyi2_default_is_shannon():
    """Default method is Shannon — adding the parameter must not
    change behaviour for callers who don't pass it. Tests single-multiset r=1
    case which Shannon supports."""
    rng = np.random.default_rng(0)
    P = 1200.0
    T = build_exp_tens(
        rng.uniform(0, P, 5), rng.uniform(0.5, 1.5, 5),
        50.0, 1, False, True, P, verbose=False,
    )
    H_default = entropy_exp_tens(T, n_points_per_dim=1200)
    H_explicit = entropy_exp_tens(T, method='shannon', n_points_per_dim=1200)
    assert H_default == H_explicit


def test_renyi2_inequality_with_shannon():
    """For a discrete pmf the Rényi-α decreases in α, so
    H_2 <= H_1 = Shannon. For continuous densities the same
    inequality holds when both are computed on the same support
    with consistent normalisation. We test the unnormalised case
    on r=1 single-multiset where Shannon's grid path works.
    """
    rng = np.random.default_rng(0)
    P = 1200.0
    T = build_exp_tens(
        rng.uniform(0, P, 8), rng.uniform(0.5, 1.5, 8),
        50.0, 1, False, True, P, verbose=False,
    )
    # On a periodic grid with N points, the discrete Shannon entropy
    # is bounded by log2(N). The continuous Rényi-2 is bounded by
    # log2(P). Convert Shannon to its continuous equivalent by adding
    # log2(P/N) (the "differential entropy" correction).
    N = 1200
    H_shannon_disc = entropy_exp_tens(
        T, method='shannon',
        n_points_per_dim=N,
    )
    H_shannon_cont = H_shannon_disc + np.log2(P / N)
    H2 = entropy_exp_tens(T, method='renyi2')
    # Continuous Rényi-2 ≤ continuous Shannon, with rough equality
    # when the density is near-uniform.
    assert H2 <= H_shannon_cont + 1e-3


# ---- Fallback to pairwise self-IP --------------------------------


def test_renyi2_sa_nan_on_orbit_negative(monkeypatch):
    """If the orbit self-IP returns a non-positive value (sign-flip from
    FP cancellation), the single-multiset Rényi-2 path returns NaN.

    Rationale: empirical sweeps across all 7 standard regimes
    (sweep_self_ip.py) found no real-world case where orbit produces a
    corrupt self-IP, so a non-positive self-IP indicates either a
    degenerate (zero-mass) density or a parameter regime far outside what
    the orbit machinery can handle. Either way the collision entropy is
    undefined, so the finite-and-positive guard returns NaN rather than a
    value the caller cannot interpret. (A second-method numerical fallback
    is deliberately out of scope; orbit and pairwise use different
    normalisation conventions in rel mode.)
    """
    import mpt.entropy as ent_mod
    rng = np.random.default_rng(0)
    P = 1200.0
    T = build_exp_tens(
        rng.uniform(0, P, 5), rng.uniform(0.5, 1.5, 5),
        50.0, 2, False, True, P, verbose=False,
    )
    # The Rényi-2 path composes per-attribute inner matrices; patch the
    # function it actually calls so the finite-and-positive guard sees a
    # corrupt self-IP.
    monkeypatch.setattr(
        ent_mod, '_ma_per_attr_inner_matrix',
        lambda *a, **k: np.full((1, 1), -1.0),
    )
    assert np.isnan(entropy_exp_tens(T, method='renyi2'))


def test_renyi2_sa_nan_on_orbit_nonfinite(monkeypatch):
    """If the orbit self-IP returns NaN or inf, the single-multiset path returns NaN."""
    import mpt.entropy as ent_mod
    rng = np.random.default_rng(0)
    P = 1200.0
    T = build_exp_tens(
        rng.uniform(0, P, 5), rng.uniform(0.5, 1.5, 5),
        50.0, 2, False, True, P, verbose=False,
    )
    monkeypatch.setattr(
        ent_mod, '_ma_per_attr_inner_matrix',
        lambda *a, **k: np.full((1, 1), float('nan')),
    )
    assert np.isnan(entropy_exp_tens(T, method='renyi2'))


def test_renyi2_ma_nan_on_orbit_negative(monkeypatch):
    """MA path: if a corrupt per-attribute matrix yields negative summed
    ip_xx, the path returns NaN (finite-and-positive guard); no silent
    fallback to a differently-normalised method."""
    import mpt.entropy as ent_mod
    rng = np.random.default_rng(0)
    P = 1200.0
    N = 5
    p_attr = [rng.uniform(0, P, (1, N)), rng.uniform(0, 1.0, (1, N))]
    w = [rng.uniform(0.5, 1.5, (1, N)), rng.uniform(0.5, 1.5, (1, N))]
    dens = build_exp_tens(
        p_attr, w, [50.0, 0.05], [1, 1], 
        [False, False], [True, True], [P, 1.0], verbose=False,
    )

    real_matrix = ent_mod._ma_per_attr_inner_matrix
    call_count = [0]

    def fake_matrix(*args, **kwargs):
        call_count[0] += 1
        I = real_matrix(*args, **kwargs)
        if call_count[0] == 1:
            return -I
        return I
    monkeypatch.setattr(ent_mod, '_ma_per_attr_inner_matrix', fake_matrix)

    assert np.isnan(entropy_exp_tens(dens, method='renyi2'))
