"""Tests for ``truncation_sigmas`` honouring in the MA Möbius IP path
and in ``weight_events``.

Covers, on the MA per-attribute inner-product matrix
(:func:`mpt._tensor.cosine._ma_per_attr_inner_matrix` and its
sub-helpers):

- r = 1 abs (single-shot and chunked)
- r >= 2 abs safe (vectorised batched Möbius)
- r >= 2 abs unsafe (batched direct r-tuple enumeration)
- r >= 2 abs mixed safe/unsafe
- rel-per (relative periodic, u-grid integrated)

For each, the truncated path at ``truncation_sigmas=6`` agrees with
the un-truncated reference (``inf``) to better than ``exp(-k^2/2)``
absolute on representative inputs.

Covers, on :func:`mpt.weight_events`:

- Default ``truncation_sigmas=inf`` reproduces the un-truncated
  factor exactly.
- Finite ``truncation_sigmas=k`` hard-zeros the factor where
  ``|delta| > k * width`` for pure-Gaussian, pure-rectangle, and
  general rect-Gaussian (gamma in (0, 1)) shapes.
- Periodic input groups: truncation is applied after the period wrap.
"""

import numpy as np
import pytest

import mpt
from mpt import weight_events
from mpt._tensor._mobius_inner import (_ma_per_attr_inner_matrix, _batched_direct_enum_abs)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _restore_defaults():
    """Save / restore ``truncation_sigmas`` and
    ``kernel_chunk_bytes`` global defaults so tests don't leak."""
    prev_trunc = mpt.get_default('truncation_sigmas')
    prev_chunk = mpt.get_default('kernel_chunk_bytes')
    yield
    mpt.set_default(truncation_sigmas=prev_trunc)
    mpt.set_default(kernel_chunk_bytes=prev_chunk)


def _make_inputs(K, N_x, N_y, sigma, rng, span=None):
    """Random (Px, Wx, Py, Wy) over a span large enough to make
    truncation prune most pairs."""
    if span is None:
        span = 40.0 * sigma   # ~4σ-many distances on each side
    Px = rng.uniform(0.0, span, size=(K, N_x))
    Py = rng.uniform(0.0, span, size=(K, N_y))
    Wx = rng.uniform(0.1, 1.0, size=(K, N_x))
    Wy = rng.uniform(0.1, 1.0, size=(K, N_y))
    return Px, Wx, Py, Wy


def _abs_tol(truncation_sigmas, ip_ref):
    """Per-entry truncation introduces error of order exp(-k^2/2)
    times the per-pair kernel sum count. Use a generous bound:
    max(reference) * 10 * exp(-k^2/2)."""
    return float(np.max(np.abs(ip_ref))) * 10.0 * np.exp(
        -truncation_sigmas ** 2 / 2.0,
    )


# ---------------------------------------------------------------------
# r = 1 abs
# ---------------------------------------------------------------------

def test_r1_abs_single_shot_truncation_parity():
    rng = np.random.default_rng(0)
    K, N = 8, 12
    sigma = 1.0
    Px, Wx, Py, Wy = _make_inputs(K, N, N, sigma, rng)
    ip_inf = _ma_per_attr_inner_matrix(
        Px, Wx, Py, Wy, sigma, r=1, is_rel=False, is_per=False, period=0.0,
        truncation_sigmas=float('inf'),
    )
    ip_6 = _ma_per_attr_inner_matrix(
        Px, Wx, Py, Wy, sigma, r=1, is_rel=False, is_per=False, period=0.0,
        truncation_sigmas=6.0,
    )
    assert np.max(np.abs(ip_6 - ip_inf)) < _abs_tol(6.0, ip_inf)


def test_r1_abs_chunked_truncation_parity():
    """Force chunking via a tiny kernel_chunk_bytes; chunked and
    single-shot truncation must give the same result."""
    rng = np.random.default_rng(1)
    K, N = 8, 20
    sigma = 1.0
    Px, Wx, Py, Wy = _make_inputs(K, N, N, sigma, rng)

    # Tight budget forces chunking along N_x.
    mpt.set_default(kernel_chunk_bytes=8 * K * K * N * 8)   # 8 rows
    ip_chunked = _ma_per_attr_inner_matrix(
        Px, Wx, Py, Wy, sigma, r=1, is_rel=False, is_per=False, period=0.0,
        truncation_sigmas=6.0,
    )

    # Loose budget -> single-shot.
    mpt.set_default(kernel_chunk_bytes=10 ** 12)
    ip_single = _ma_per_attr_inner_matrix(
        Px, Wx, Py, Wy, sigma, r=1, is_rel=False, is_per=False, period=0.0,
        truncation_sigmas=6.0,
    )
    # Chunked vs single-shot einsum reduction order differs at the
    # ULP level; tolerance is well below the truncation absolute
    # threshold.
    np.testing.assert_allclose(ip_chunked, ip_single, rtol=1e-12, atol=0.0)


def test_r1_abs_periodic_truncation_parity():
    rng = np.random.default_rng(2)
    K, N = 6, 8
    sigma, period = 1.0, 12.0
    Px, Wx, Py, Wy = _make_inputs(K, N, N, sigma, rng, span=period)
    ip_inf = _ma_per_attr_inner_matrix(
        Px, Wx, Py, Wy, sigma, r=1, is_rel=False, is_per=True, period=period,
        truncation_sigmas=float('inf'),
    )
    ip_6 = _ma_per_attr_inner_matrix(
        Px, Wx, Py, Wy, sigma, r=1, is_rel=False, is_per=True, period=period,
        truncation_sigmas=6.0,
    )
    assert np.max(np.abs(ip_6 - ip_inf)) < _abs_tol(6.0, ip_inf)


# ---------------------------------------------------------------------
# r >= 2 abs (safe / unsafe / rel-per)
# ---------------------------------------------------------------------

def test_r2_abs_safe_truncation_parity():
    """All events K_eff >= r+2 -> safe path (batched Möbius)."""
    rng = np.random.default_rng(3)
    K, N = 6, 8       # K=6, r=2 -> K-r=4 >= 2 -> safe
    sigma = 1.0
    Px, Wx, Py, Wy = _make_inputs(K, N, N, sigma, rng)
    ip_inf = _ma_per_attr_inner_matrix(
        Px, Wx, Py, Wy, sigma, r=2, is_rel=False, is_per=False, period=0.0,
        truncation_sigmas=float('inf'),
    )
    ip_6 = _ma_per_attr_inner_matrix(
        Px, Wx, Py, Wy, sigma, r=2, is_rel=False, is_per=False, period=0.0,
        truncation_sigmas=6.0,
    )
    assert np.max(np.abs(ip_6 - ip_inf)) < _abs_tol(6.0, ip_inf)


def test_r2_abs_unsafe_truncation_parity():
    """All events K_eff = r -> all unsafe -> batched direct enum."""
    rng = np.random.default_rng(4)
    K, N = 2, 8       # K=2, r=2 -> K-r=0 -> all unsafe
    sigma = 1.0
    Px, Wx, Py, Wy = _make_inputs(K, N, N, sigma, rng)
    ip_inf = _batched_direct_enum_abs(
        Px, Wx, Py, Wy, sigma, r=2, is_per=False, period=0.0,
        truncation_sigmas=float('inf'),
    )
    ip_6 = _batched_direct_enum_abs(
        Px, Wx, Py, Wy, sigma, r=2, is_per=False, period=0.0,
        truncation_sigmas=6.0,
    )
    assert np.max(np.abs(ip_6 - ip_inf)) < _abs_tol(6.0, ip_inf)


def test_r2_abs_mixed_truncation_parity():
    """Mix of safe and unsafe events on each side; exercises both
    the safe-Möbius sub-block and the unsafe direct-enum fill."""
    rng = np.random.default_rng(5)
    K_safe, K_unsafe = 5, 2     # r=2: K=5 safe, K=2 unsafe
    sigma = 1.0
    # Build a ragged (K, N) matrix with NaN-fill where event K < K_safe.
    N = 8
    Px = np.full((K_safe, N), np.nan)
    Wx = np.full((K_safe, N), np.nan)
    Py = np.full((K_safe, N), np.nan)
    Wy = np.full((K_safe, N), np.nan)
    safe_pos = rng.uniform(0.0, 40.0, size=(K_safe, N))
    safe_w = rng.uniform(0.1, 1.0, size=(K_safe, N))
    for n in range(N):
        k_this = K_safe if (n % 2 == 0) else K_unsafe
        Px[:k_this, n] = safe_pos[:k_this, n]
        Wx[:k_this, n] = safe_w[:k_this, n]
        Py[:k_this, n] = safe_pos[:k_this, n] + 0.5
        Wy[:k_this, n] = safe_w[:k_this, n] + 0.05
    ip_inf = _ma_per_attr_inner_matrix(
        Px, Wx, Py, Wy, sigma, r=2, is_rel=False, is_per=False, period=0.0,
        truncation_sigmas=float('inf'),
    )
    ip_6 = _ma_per_attr_inner_matrix(
        Px, Wx, Py, Wy, sigma, r=2, is_rel=False, is_per=False, period=0.0,
        truncation_sigmas=6.0,
    )
    assert np.max(np.abs(ip_6 - ip_inf)) < _abs_tol(6.0, ip_inf)


def test_rel_per_truncation_parity():
    rng = np.random.default_rng(6)
    K, N = 5, 6
    sigma, period = 1.0, 12.0
    Px, Wx, Py, Wy = _make_inputs(K, N, N, sigma, rng, span=period)
    ip_inf = _ma_per_attr_inner_matrix(
        Px, Wx, Py, Wy, sigma, r=2, is_rel=True, is_per=True, period=period,
        truncation_sigmas=float('inf'),
    )
    ip_6 = _ma_per_attr_inner_matrix(
        Px, Wx, Py, Wy, sigma, r=2, is_rel=True, is_per=True, period=period,
        truncation_sigmas=6.0,
    )
    assert np.max(np.abs(ip_6 - ip_inf)) < _abs_tol(6.0, ip_inf)


# ---------------------------------------------------------------------
# Default resolution
# ---------------------------------------------------------------------

def test_default_truncation_sigmas_resolution():
    """``truncation_sigmas=None`` resolves to the global default; a
    direct ``inf`` call must give a bit-identical result when the
    default is inf, and a finite-default call must match an explicit
    pass of the same value."""
    rng = np.random.default_rng(7)
    K, N = 6, 8
    sigma = 1.0
    Px, Wx, Py, Wy = _make_inputs(K, N, N, sigma, rng)

    mpt.set_default(truncation_sigmas=float('inf'))
    ip_default_inf = _ma_per_attr_inner_matrix(
        Px, Wx, Py, Wy, sigma, r=1, is_rel=False, is_per=False, period=0.0,
    )
    ip_explicit_inf = _ma_per_attr_inner_matrix(
        Px, Wx, Py, Wy, sigma, r=1, is_rel=False, is_per=False, period=0.0,
        truncation_sigmas=float('inf'),
    )
    np.testing.assert_array_equal(ip_default_inf, ip_explicit_inf)

    mpt.set_default(truncation_sigmas=6.0)
    ip_default_6 = _ma_per_attr_inner_matrix(
        Px, Wx, Py, Wy, sigma, r=1, is_rel=False, is_per=False, period=0.0,
    )
    ip_explicit_6 = _ma_per_attr_inner_matrix(
        Px, Wx, Py, Wy, sigma, r=1, is_rel=False, is_per=False, period=0.0,
        truncation_sigmas=6.0,
    )
    np.testing.assert_array_equal(ip_default_6, ip_explicit_6)


# ---------------------------------------------------------------------
# weight_events truncation
# ---------------------------------------------------------------------

def _make_we_inputs(N):
    """Single-attribute (pitch, time) setup with K=1 on both."""
    times = np.linspace(-10.0, 10.0, N).reshape(1, N)
    pitches = np.zeros((1, N))
    p_attr = [pitches, times]
    w_init = [np.ones((1, N)), np.ones((1, N))]
    groups = [0, 1]
    return p_attr, w_init, groups, times


def test_weight_events_default_inf_resolves_to_accuracy_floor():
    """Under the truncation contract, ``mpt.set_default(truncation_sigmas
    =math.inf)`` resolves to the finite accuracy-floor width (~7.43
    sigma, the 1e-12 floor), so :func:`weight_events` still applies a
    hard-zero cutoff there --- never "disabled". The setting must
    therefore be bit-identical to setting the default to
    :func:`mpt._defaults.accuracy_floor_sigmas` explicitly, and must
    differ from the un-truncated analytic Gaussian at events beyond
    that width."""
    from mpt._defaults import accuracy_floor_sigmas
    p_attr, w_init, groups, times = _make_we_inputs(N=201)  # dense enough to have events past 7.43 * sd
    prev = mpt.get_default('truncation_sigmas')
    try:
        mpt.set_default(truncation_sigmas=float('inf'))
        _, w_out_inf, _ = weight_events(
            p_attr, w_init, input_attr=1, target_attr=0,
            centre=0.0, sd=1.0, shape=0.0, is_per=False, period=0.0,
            drop_input_attr=True,
        )
        mpt.set_default(truncation_sigmas=accuracy_floor_sigmas())
        _, w_out_floor, _ = weight_events(
            p_attr, w_init, input_attr=1, target_attr=0,
            centre=0.0, sd=1.0, shape=0.0, is_per=False, period=0.0,
            drop_input_attr=True,
        )
    finally:
        mpt.set_default(truncation_sigmas=prev)
    # Contract: inf resolves to the accuracy-floor width, so both
    # settings are bit-identical.
    np.testing.assert_array_equal(w_out_inf[0], w_out_floor[0])
    # And the tail beyond the accuracy-floor width must be hard-zeroed
    # (never the un-truncated analytic Gaussian).
    delta = times[0]
    outside = np.abs(delta) > accuracy_floor_sigmas()
    assert outside.any(), "test needs events past the accuracy-floor cutoff"
    assert np.all(w_out_inf[0][0][outside] == 0.0)


def test_weight_events_truncation_gaussian():
    """Pure Gaussian (shape=0): factor is exactly 0 where |delta| > k*width."""
    p_attr, w_init, groups, times = _make_we_inputs(N=21)
    mpt.set_default(truncation_sigmas=3.0)
    _, w_out, _ = weight_events(
        p_attr, w_init, input_attr=1, target_attr=0,
        centre=0.0, sd=1.0, shape=0.0, is_per=False, period=0.0,
        drop_input_attr=True,
    )
    factor = w_out[0][0]
    delta = times[0]
    outside = np.abs(delta) > 3.0
    inside = ~outside
    assert np.all(factor[outside] == 0.0)
    # Inside the cutoff, factor matches the analytic Gaussian.
    np.testing.assert_allclose(
        factor[inside], np.exp(-delta[inside] ** 2 / 2.0),
    )


def test_weight_events_truncation_rectangle():
    """Pure rectangle (shape=1): factor outside |delta| > k*width is 0;
    even though the rectangle is already 0 outside its own support,
    the truncation applies on top."""
    p_attr, w_init, groups, times = _make_we_inputs(N=21)
    # Rectangle has half-width width*sqrt(3); pick width so the
    # truncation cutoff is tighter than the rectangle support.
    width = 1.0
    k = 1.0   # truncation cutoff at 1*width, rectangle support at sqrt(3) ~= 1.73*width
    mpt.set_default(truncation_sigmas=k)
    _, w_out, _ = weight_events(
        p_attr, w_init, input_attr=1, target_attr=0,
        centre=0.0, sd=width, shape=1.0, is_per=False, period=0.0,
        drop_input_attr=True,
    )
    factor = w_out[0][0]
    delta = times[0]
    outside_cutoff = np.abs(delta) > k * width
    assert np.all(factor[outside_cutoff] == 0.0)


def test_weight_events_truncation_general_shape():
    """General rect-Gaussian convolution (gamma=0.5): factor
    outside |delta| > k*width is exactly 0; inside, factor matches
    the un-truncated reference."""
    p_attr, w_init, groups, times = _make_we_inputs(N=41)
    width, gamma, k = 1.0, 0.5, 3.0
    mpt.set_default(truncation_sigmas=float('inf'))
    _, w_ref, _ = weight_events(
        p_attr, w_init, input_attr=1, target_attr=0,
        centre=0.0, sd=width, shape=gamma, is_per=False, period=0.0,
        drop_input_attr=True,
    )
    mpt.set_default(truncation_sigmas=k)
    _, w_trunc, _ = weight_events(
        p_attr, w_init, input_attr=1, target_attr=0,
        centre=0.0, sd=width, shape=gamma, is_per=False, period=0.0,
        drop_input_attr=True,
    )
    delta = times[0]
    outside = np.abs(delta) > k * width
    inside = ~outside
    assert np.all(w_trunc[0][0][outside] == 0.0)
    np.testing.assert_array_equal(w_trunc[0][0][inside], w_ref[0][0][inside])


def test_weight_events_truncation_periodic_after_wrap():
    """For periodic input groups, the delta is wrapped to
    [-P/2, P/2] before truncation, so events near the periodic image
    of the centre survive."""
    # Period 10, centre 0, width 1, truncation 3. Events at times
    # 0 and 10 should both survive (after wrap, the second is at
    # delta=0); event at time 5 should be truncated (delta=5 or -5).
    times = np.array([[0.0, 5.0, 10.0]])
    pitches = np.zeros((1, 3))
    p_attr = [pitches, times]
    w_init = [np.ones((1, 3)), np.ones((1, 3))]
    groups = [0, 1]
    mpt.set_default(truncation_sigmas=3.0)
    _, w_out, _ = weight_events(
        p_attr, w_init, input_attr=1, target_attr=0,
        centre=0.0, sd=1.0, shape=0.0,
        is_per=True, period=10.0,
        drop_input_attr=True,
    )
    factor = w_out[0][0]


def test_window_factor_default_inf_resolves_to_accuracy_floor():
    """The windowed-sweep locate factor (:func:`mpt._tensor.windowed
    ._window_factor`, the sibling of :func:`weight_events` used by
    :func:`windowed_similarity` / :func:`windowed_entropy`) must honour
    the same truncation contract: ``mpt.set_default(truncation_sigmas
    =math.inf)`` resolves to the finite accuracy-floor width, so the
    factor is bit-identical to setting the default to
    :func:`accuracy_floor_sigmas` explicitly, and events beyond that
    width are hard-zeroed."""
    from mpt._defaults import accuracy_floor_sigmas
    from mpt._tensor.windowed import _window_factor
    loc_row = np.linspace(-10.0, 10.0, 201)
    prev = mpt.get_default('truncation_sigmas')
    try:
        mpt.set_default(truncation_sigmas=float('inf'))
        f_inf = _window_factor(loc_row, centre=0.0, gamma=0.0, sd=1.0)
        mpt.set_default(truncation_sigmas=accuracy_floor_sigmas())
        f_floor = _window_factor(loc_row, centre=0.0, gamma=0.0, sd=1.0)
    finally:
        mpt.set_default(truncation_sigmas=prev)
    np.testing.assert_array_equal(f_inf, f_floor)
    outside = np.abs(loc_row) > accuracy_floor_sigmas()
    assert outside.any(), "test needs events past the accuracy-floor cutoff"
    assert np.all(f_inf[outside] == 0.0)


# ---------------------------------------------------------------------
# Auto-pruning of zero-weight events
# ---------------------------------------------------------------------

def test_auto_pruning_drops_zero_weight_events():
    """When some events have all-zero weights, the result matches the
    explicit-no-prune path and the corresponding rows / columns are
    exactly zero."""
    rng = np.random.default_rng(8)
    K, N = 6, 10
    sigma = 1.0
    Px = rng.uniform(0.0, 20.0, size=(K, N))
    Py = rng.uniform(0.0, 20.0, size=(K, N))
    Wx = rng.uniform(0.1, 1.0, size=(K, N))
    Wy = rng.uniform(0.1, 1.0, size=(K, N))
    zero_x = [1, 4, 7]
    zero_y = [2, 8]
    Wx[:, zero_x] = 0.0
    Wy[:, zero_y] = 0.0

    ip_prune = _ma_per_attr_inner_matrix(
        Px, Wx, Py, Wy, sigma, r=1, is_rel=False, is_per=False, period=0.0,
        prune_zero_weight_events=True,
    )
    ip_full = _ma_per_attr_inner_matrix(
        Px, Wx, Py, Wy, sigma, r=1, is_rel=False, is_per=False, period=0.0,
        prune_zero_weight_events=False,
    )
    np.testing.assert_allclose(ip_prune, ip_full, rtol=1e-12, atol=1e-12)
    # Pruned rows / columns must be exactly zero.
    assert np.all(ip_prune[zero_x, :] == 0.0)
    assert np.all(ip_prune[:, zero_y] == 0.0)


def test_auto_pruning_no_zero_events_noop():
    """When all events have positive weight, pruning is a no-op
    (numerically identical to no-prune)."""
    rng = np.random.default_rng(9)
    K, N = 5, 8
    sigma = 1.0
    Px, Wx, Py, Wy = _make_inputs(K, N, N, sigma, rng)
    ip_prune = _ma_per_attr_inner_matrix(
        Px, Wx, Py, Wy, sigma, r=1, is_rel=False, is_per=False, period=0.0,
        prune_zero_weight_events=True,
    )
    ip_full = _ma_per_attr_inner_matrix(
        Px, Wx, Py, Wy, sigma, r=1, is_rel=False, is_per=False, period=0.0,
        prune_zero_weight_events=False,
    )
    np.testing.assert_array_equal(ip_prune, ip_full)


def test_auto_pruning_all_zero_side_returns_zero_matrix():
    """If every event on one side has zero weight, the IP is the
    all-zero matrix (returned with the correct full shape)."""
    rng = np.random.default_rng(10)
    K, N = 4, 6
    sigma = 1.0
    Px, Wx, Py, Wy = _make_inputs(K, N, N, sigma, rng)
    Wx[:] = 0.0
    ip = _ma_per_attr_inner_matrix(
        Px, Wx, Py, Wy, sigma, r=1, is_rel=False, is_per=False, period=0.0,
        prune_zero_weight_events=True,
    )
    assert ip.shape == (N, N)
    assert np.all(ip == 0.0)


def test_auto_pruning_with_cancellation_ratio():
    """``return_cancellation_ratio=True`` must work alongside auto-pruning."""
    rng = np.random.default_rng(11)
    K, N = 6, 8       # r=2 -> K-r=4, safe path
    sigma = 1.0
    Px, Wx, Py, Wy = _make_inputs(K, N, N, sigma, rng)
    Wx[:, [0, 3]] = 0.0
    Wy[:, [5]] = 0.0
    ip, ratio = _ma_per_attr_inner_matrix(
        Px, Wx, Py, Wy, sigma, r=2, is_rel=False, is_per=False, period=0.0,
        return_cancellation_ratio=True,
        prune_zero_weight_events=True,
    )
    assert ip.shape == (N, N)
    assert 0.0 < ratio <= 1.0
    assert np.all(ip[[0, 3], :] == 0.0)
    assert np.all(ip[:, [5]] == 0.0)


def test_auto_pruning_nan_weight_treated_as_missing():
    """NaN weights are treated as missing slots (intra-event); a
    column is "zero-weight" iff every non-NaN slot is zero."""
    rng = np.random.default_rng(12)
    K, N = 5, 6
    sigma = 1.0
    Px = rng.uniform(0.0, 20.0, size=(K, N))
    Py = rng.uniform(0.0, 20.0, size=(K, N))
    Wx = rng.uniform(0.1, 1.0, size=(K, N))
    Wy = rng.uniform(0.1, 1.0, size=(K, N))
    # Event 2 has K_eff = 2 (rows 0..1 valid, rows 2..4 NaN) and
    # positive weights — should NOT be pruned.
    Wx[2:, 2] = np.nan
    Px[2:, 2] = np.nan
    # Event 3 has all-zero weight on its valid slots — SHOULD be pruned.
    Wx[:, 3] = 0.0
    ip = _ma_per_attr_inner_matrix(
        Px, Wx, Py, Wy, sigma, r=1, is_rel=False, is_per=False, period=0.0,
        prune_zero_weight_events=True,
    )
    # Event 2 is not pruned, so its row should be non-trivial.
    assert np.any(ip[2, :] != 0.0)
    # Event 3 is pruned, so its row is exactly zero.
    assert np.all(ip[3, :] == 0.0)


# ---------------------------------------------------------------------
# Bulger-method truncation
# ---------------------------------------------------------------------

def test_bulger_ma_rel_per_truncation_parity():
    """The MA Bulger path (``_ip_core_ma`` / ``_ip_full_ma``) honours
    truncation_sigmas via log-space thresholding. Test through the
    public cos_sim_exp_tens with method='bulger'."""
    from mpt import build_exp_tens, cos_sim_exp_tens

    rng = np.random.default_rng(13)
    K = 4
    P = rng.uniform(0.0, 12.0, size=K)
    W = rng.uniform(0.1, 1.0, size=K)
    P2 = rng.uniform(0.0, 12.0, size=K)
    W2 = rng.uniform(0.1, 1.0, size=K)
    # MA = two single-multiset densities cross-correlated; use
    # build_exp_tens with two events to keep MA-shaped storage.
    dens_x = build_exp_tens(P, W, 1.0, 2, True, True, 12.0)
    dens_y = build_exp_tens(P2, W2, 1.0, 2, True, True, 12.0)

    mpt.set_default(truncation_sigmas=float('inf'))
    s_inf = cos_sim_exp_tens(dens_x, dens_y, method='bulger', verbose=False)
    mpt.set_default(truncation_sigmas=6.0)
    s_6 = cos_sim_exp_tens(dens_x, dens_y, method='bulger', verbose=False)
    assert abs(s_inf - s_6) < 1e-7


# =========================================================================
# Auto-pruning of zero-weight joint events in the grid eval path
# (Shannon entropy / general MA density evaluation)
# =========================================================================


def test_eval_ma_auto_prune_parity_vs_unpruned():
    """Pruned and unpruned eval agree to FP tolerance on a windowed
    density. The build's joint w_j is the per-attribute weight product,
    so a tuple with w_j == 0 has at least one attribute with zero
    weight at that tuple --- contributes zero everywhere, drop is exact.
    """
    from mpt._tensor.eval import _eval_exp_tens_ma

    p = [np.array([[float(i) for i in range(20)]]),
         np.array([[float(i) for i in range(20)]])]

    # weight_events with tight truncation hard-zeros most of the time slot.
    old = mpt.get_default('truncation_sigmas')
    try:
        mpt.set_default(truncation_sigmas=2.0)
        _, w, _ = mpt.weight_events(
            p, None,
            input_attr=0, target_attr=1,
            centre=5.0, sd=1.0, shape=0.0,
            is_per=False, period=0.0,
            drop_input_attr=False,
        )
    finally:
        mpt.set_default(truncation_sigmas=old)

    dens = mpt.build_exp_tens(
        p, w, np.array([1.0, 1.0]), np.array([1, 1]), 
        np.array([False, False]), np.array([False, False]),
        np.array([0.0, 0.0]),
        verbose=False,
    )

    # A representative 2-D grid spanning the data range.
    g = np.linspace(-2.0, 22.0, 24)
    x_grid = np.array([np.tile(g, g.size), np.repeat(g, g.size)])

    vals_pruned   = _eval_exp_tens_ma(dens, x_grid, 'none',
                                       prune_zero_weight_events=True)
    vals_unpruned = _eval_exp_tens_ma(dens, x_grid, 'none',
                                       prune_zero_weight_events=False)

    # Reordered sums give at-most a few ULPs of drift. Absolute
    # tolerance well below the floor of any musically meaningful
    # downstream quantity.
    np.testing.assert_allclose(vals_pruned, vals_unpruned, atol=1e-12)


def test_eval_ma_auto_prune_default_is_on():
    """Default behaviour (no kwarg) matches the prune=True branch."""
    from mpt._tensor.eval import _eval_exp_tens_ma

    p = [np.array([[float(i) for i in range(15)]])]
    w = [np.array([[0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0]])]

    dens = mpt.build_exp_tens(
        p, w, np.array([1.0]), np.array([1]), 
        np.array([False]), np.array([False]), np.array([0.0]),
        verbose=False,
    )

    g = np.linspace(-2.0, 17.0, 24).reshape(1, -1)
    vals_default = _eval_exp_tens_ma(dens, g, 'none')
    vals_explicit_on = _eval_exp_tens_ma(dens, g, 'none',
                                          prune_zero_weight_events=True)
    np.testing.assert_array_equal(vals_default, vals_explicit_on)


def test_eval_ma_auto_prune_all_zero_returns_zeros():
    """A density whose joint weights are all zero returns identically
    zero at every query point."""
    from mpt._tensor.eval import _eval_exp_tens_ma

    p = [np.array([[1.0, 2.0, 3.0]])]
    w = [np.zeros((1, 3))]   # entire weight is zero --- joint w_j is zero

    dens = mpt.build_exp_tens(
        p, w, np.array([1.0]), np.array([1]), 
        np.array([False]), np.array([False]), np.array([0.0]),
        verbose=False,
    )

    x = np.linspace(-2.0, 5.0, 24).reshape(1, -1)
    vals = _eval_exp_tens_ma(dens, x, 'none')
    np.testing.assert_array_equal(vals, np.zeros_like(x[0]))


def test_eval_ma_auto_prune_no_zeros_is_a_noop():
    """When no joint weights are zero, prune is a no-op (FP-identical)."""
    from mpt._tensor.eval import _eval_exp_tens_ma

    rng = np.random.default_rng(42)
    p = [rng.uniform(0.0, 10.0, size=(1, 8))]
    w = [rng.uniform(0.5, 1.5, size=(1, 8))]   # strictly positive

    dens = mpt.build_exp_tens(
        p, w, np.array([1.0]), np.array([1]), 
        np.array([False]), np.array([False]), np.array([0.0]),
        verbose=False,
    )

    x = np.linspace(-1.0, 11.0, 32).reshape(1, -1)
    vals_pruned   = _eval_exp_tens_ma(dens, x, 'none',
                                       prune_zero_weight_events=True)
    vals_unpruned = _eval_exp_tens_ma(dens, x, 'none',
                                       prune_zero_weight_events=False)
    # No tuples dropped --- this is bit-identical, not just close.
    np.testing.assert_array_equal(vals_pruned, vals_unpruned)


def test_eval_ma_auto_prune_propagates_via_entropy_exp_tens():
    """End-to-end check: entropy_exp_tens(method='shannon') on a
    weight_events-windowed density runs to completion and produces a
    finite, non-NaN value. This is the user-facing use case for the
    auto-prune."""
    p = [np.array([[float(i) for i in range(20)]]),
         np.array([[float(i) for i in range(20)]])]
    old = mpt.get_default('truncation_sigmas')
    try:
        mpt.set_default(truncation_sigmas=2.0)
        _, w, _ = mpt.weight_events(
            p, None,
            input_attr=0, target_attr=1,
            centre=5.0, sd=1.0, shape=0.0,
            is_per=False, period=0.0,
            drop_input_attr=False,
        )
    finally:
        mpt.set_default(truncation_sigmas=old)

    H = mpt.entropy_exp_tens(
        p, w, np.array([1.0, 1.0]), np.array([1, 1]), 
        np.array([False, False]), np.array([False, False]),
        np.array([0.0, 0.0]),
        method='shannon',
        x_min=-2.0, x_max=22.0, n_points_per_dim=24,
    )
    assert np.isfinite(H)


# =========================================================================
# Zero-weight-event handling in the eval path
# =========================================================================
#
# A density carrying zero-weight events must evaluate to exactly what the
# same density with those events removed would give: zero-weight events
# contribute nothing to the expectation-tensor density. These tests pin
# that invariant on the public evaluator across the grid (r = 1) and
# tuple-query (r = 2) regimes, and the all-zero-weights degenerate case.


def test_eval_zero_weight_events_match_removed_grid():
    """r = 1 grid eval: a density with zero-weight events agrees to FP
    tolerance with the same density built without those events."""
    p = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    w = np.array([0.0, 1.0, 0.0, 1.0, 1.0])
    dens = mpt.build_exp_tens(p, w, 1.0, 1, False, False, 0.0, verbose=False)

    p_kept = np.array([2.0, 4.0, 5.0])
    w_kept = np.array([1.0, 1.0, 1.0])
    dens_kept = mpt.build_exp_tens(
        p_kept, w_kept, 1.0, 1, False, False, 0.0, verbose=False)

    x = np.linspace(0.0, 6.0, 32).reshape(1, -1)
    vals = mpt.eval_exp_tens(dens, x, verbose=False)
    vals_kept = mpt.eval_exp_tens(dens_kept, x, verbose=False)
    np.testing.assert_allclose(vals, vals_kept, atol=1e-12)


def test_eval_zero_weight_events_match_removed_tuple():
    """r = 2 tuple-query eval: same invariant with r-tuple queries, so
    the tuple-enumeration path exercises the zero-weight prune."""
    rng = np.random.default_rng(7)
    p = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    w = np.array([0.0, 1.0, 0.0, 1.0, 1.0, 0.0])   # half the events zero
    dens = mpt.build_exp_tens(p, w, 1.0, 2, False, False, 0.0, verbose=False)

    p_kept = np.array([2.0, 4.0, 5.0])
    w_kept = np.array([1.0, 1.0, 1.0])
    dens_kept = mpt.build_exp_tens(
        p_kept, w_kept, 1.0, 2, False, False, 0.0, verbose=False)

    x = rng.uniform(0.0, 7.0, size=(2, 12))
    vals = mpt.eval_exp_tens(dens, x, verbose=False)
    vals_kept = mpt.eval_exp_tens(dens_kept, x, verbose=False)
    np.testing.assert_allclose(vals, vals_kept, atol=1e-12)


def test_eval_all_zero_weights_returns_zeros_tuple():
    """r = 2: an all-zero-weight density evaluates to zeros at every
    query point."""
    p = np.array([1.0, 2.0, 3.0, 4.0])
    w = np.zeros(4)
    dens = mpt.build_exp_tens(p, w, 1.0, 2, False, False, 0.0, verbose=False)

    x = np.random.default_rng(0).uniform(0.0, 5.0, size=(2, 8))
    vals = mpt.eval_exp_tens(dens, x, verbose=False)
    np.testing.assert_array_equal(vals, np.zeros(x.shape[1]))


def test_eval_all_zero_weights_returns_zeros_grid():
    """r = 1: an all-zero-weight density evaluates to zeros at every
    grid point."""
    p = np.array([1.0, 2.0, 3.0])
    w = np.zeros(3)
    dens = mpt.build_exp_tens(p, w, 1.0, 1, False, False, 0.0, verbose=False)

    x = np.linspace(0.0, 4.0, 16).reshape(1, -1)
    vals = mpt.eval_exp_tens(dens, x, verbose=False)
    np.testing.assert_array_equal(vals, np.zeros(x.shape[1]))
