"""Entropy measures for pitch and rhythm structures."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.special import erf as _erf

from ._utils import maybe_print_batched_estimate
from ._defaults import _with_dispatch_scope
from .spectra import add_spectra
from .tensor import (
    ExpTensDensity,
    MaetDensity,
    WindowedMaetDensity,
    bind_events,
    build_exp_tens,
    difference_events,
    eval_exp_tens,
    _orbit_inner_abs,
    _orbit_inner_rel,
    _ma_per_attr_inner_matrix,
)


# Default grid-size ceiling for the Cartesian-product grid. If
# n_points_per_dim**dim exceeds this, entropy_exp_tens raises a
# clear error suggesting a lower n_points_per_dim.
_DEFAULT_GRID_LIMIT = int(1e8)


# ===================================================================
#  v2.2 migration: 'normalize' kwarg removed
# ===================================================================
#  The 'normalize' boolean kwarg on entropy_exp_tens and
#  spectral_entropy was the v2.1 mechanism for selecting between raw
#  discrete Shannon (normalize=False) and the Pielou-style ratio
#  H/log_b(N) in [0, 1] (normalize=True, the legacy default). In v2.2
#  the four-method API supersedes it: method='shannon' for raw H and
#  method='normalized' for the [0, 1] ratio. The continuous methods
#  ('differential', 'renyi2') had no [0, 1] reference and triggered an
#  error under the v2.1 default. Detection of the removed kwarg in
#  callers raises this message with the calling function name spliced
#  in via .format(fn=...).
_NORMALIZE_REMOVED_MSG = (
    "{fn}: the 'normalize' kwarg has been removed in v2.2. "
    "Use method='normalized' for H/log_b(N) in [0, 1] (the v2.1 "
    "default behaviour), or method='shannon' for raw H = -sum q log_b q. "
    "method='differential' and method='renyi2' are continuous-form "
    "entropies and have no [0, 1] reference."
)


# ===================================================================
#  Method validation and aliases
# ===================================================================
_VALID_METHODS = ("differential", "shannon", "normalized", "renyi2")


def _canonicalize_method(method: str) -> str:
    """Validate and canonicalize the ``method`` kwarg.

    Accepts the British spelling ``'normalised'`` as an alias for
    ``'normalized'``. Raises ``ValueError`` for unrecognised names.
    """
    if not isinstance(method, str):
        raise TypeError(f"method must be a string; got {type(method).__name__}.")
    m = method.strip().lower()
    if m == "normalised":
        m = "normalized"
    if m not in _VALID_METHODS:
        raise ValueError(
            f"method must be one of {_VALID_METHODS} (or 'normalised'); "
            f"got {method!r}."
        )
    return m


def _require_explicit_grid(n_points_per_dim) -> None:
    """Require an explicit grid resolution for the discrete methods.

    The previous toolbox-wide default of 1200 silently masked grid-
    fragility bugs in callers; making the grid explicit forces the
    choice up front. Called from each compute-path branch of the
    Shannon dispatch (after input-form validation, so the input-form
    errors surface first when both apply).
    """
    if n_points_per_dim is None:
        raise TypeError(
            "discrete entropy (method='shannon' or 'normalized') "
            "requires an explicit `n_points_per_dim` (the previous "
            "toolbox-wide default of 1200 has been dropped, since the "
            "right grid resolution is density- and sigma-dependent). "
            "For a grid-free continuous quantity, use "
            "method='differential' (adaptive) or method='renyi2' "
            "(analytical)."
        )


def _raise_if_any_sigma_zero(dens, *, method_name: str) -> None:
    """Reject sigma=0 for the continuous methods ('differential', 'renyi2').

    Both are continuous-form entropies that diverge at sigma=0:
    differential h_hat -> -inf, analytical Rényi-2 -log<f,f> -> -inf.
    The discrete forms ('shannon', 'normalized') handle sigma=0
    correctly (delta masses sit exactly in their cells) and are not
    guarded here. Reads sigma from any density-object form
    (ExpTensDensity, MaetDensity, WindowedMaetDensity).
    """
    if isinstance(dens, WindowedMaetDensity):
        sigma = dens.dens.sigma
    else:
        sigma = dens.sigma
    sigma_arr = np.asarray(sigma, dtype=np.float64)
    if sigma_arr.size > 0 and np.any(sigma_arr <= 0.0):
        raise ValueError(
            f"method={method_name!r} requires sigma > 0 for every group "
            f"(the continuous form diverges at sigma=0). Got sigma="
            f"{sigma_arr.tolist() if sigma_arr.ndim else float(sigma_arr)}. "
            f"For categorical (sigma=0) entropy, use method='shannon' "
            f"or method='normalized' on a category grid."
        )


# ===================================================================
#  Bin-integration core (cell masses via per-axis Phi-differences)
# ===================================================================
#
# The categorical-path discretization for `shannon` and `normalized`:
# the discrete pmf entry at grid cell j is the actual probability mass
# inside that cell, ∫_{cell_j} f dx, not the density sample f(x_j)·Δ.
# For a Gaussian-mixture density with diagonal kernel covariance in
# the effective grid coordinates --- which holds for is_rel=False
# (every group absolute) --- the cell mass factorizes into a product
# of per-axis erf differences, summed over tuples.
#
# n_tuple_entropy reaches this path differently per sigma_space.
# 'interval' differences events externally (difference_events +
# bind_events) and builds an absolute (is_rel=False) MAET on the steps,
# so it uses exact bin-integration. 'position' binds n+1 pitches and
# builds a relative (is_rel=True) window MAET, whose grid cell masses
# fall back to point-evaluation, within ~1e-4 of bin-integration on the
# fine grids these uses require anyway; the full multivariate-normal box
# treatment for relative mode is a v2.3 item.

_SQRT2 = float(np.sqrt(2.0))


def _phi_diff_axis(centres: np.ndarray, edges_lo: np.ndarray,
                   edges_hi: np.ndarray, sigma: float) -> np.ndarray:
    """Non-periodic per-axis erf-difference cell mass.

    Returns ``(n_j, n_cells)`` array with entry ``[t, j]`` equal to
    ``Phi((edges_hi[j] - centres[t]) / sigma) - Phi((edges_lo[j] -
    centres[t]) / sigma)``, the 1-D Gaussian probability mass in cell
    ``j`` for the tuple-slot at ``centres[t]``.
    """
    inv = 1.0 / (sigma * _SQRT2)
    z_hi = (edges_hi[None, :] - centres[:, None]) * inv
    z_lo = (edges_lo[None, :] - centres[:, None]) * inv
    return 0.5 * (_erf(z_hi) - _erf(z_lo))


def _phi_diff_axis_periodic(centres: np.ndarray, edges_lo: np.ndarray,
                            edges_hi: np.ndarray, sigma: float,
                            period: float,
                            truncation_sigmas: float = 6.0) -> np.ndarray:
    """Periodic per-axis erf-difference cell mass.

    Sums wraps of the Gaussian across the period grid for wraps within
    ``truncation_sigmas`` of every centre. ``period`` is the group
    period; ``edges_lo``/``edges_hi`` partition one full period.
    """
    inv = 1.0 / (sigma * _SQRT2)
    n_wraps = int(np.ceil(truncation_sigmas * sigma / period)) + 1
    n_j = int(centres.size)
    n_cells = int(edges_lo.size)
    out = np.zeros((n_j, n_cells), dtype=np.float64)
    for w in range(-n_wraps, n_wraps + 1):
        shift = float(w) * period
        z_hi = (edges_hi[None, :] - centres[:, None] - shift) * inv
        z_lo = (edges_lo[None, :] - centres[:, None] - shift) * inv
        out += 0.5 * (_erf(z_hi) - _erf(z_lo))
    return out


def _axis_edges(ax: np.ndarray, is_per: bool, period: float):
    """Cell edges for a 1-D axis.

    For a periodic group, ``ax`` is ``linspace(0, P, n+1)[:-1]`` and
    each cell is symmetric of width ``P/n`` around its grid point.
    For non-periodic, the interior cells are bounded by mid-points
    between adjacent grid points; the boundary cells extend by
    half-step on each side. Returns ``(lo, hi)`` arrays both shaped
    like ``ax``.
    """
    ax = np.asarray(ax, dtype=float)
    n = int(ax.size)
    if n < 2:
        raise ValueError("Each axis needs >= 2 points.")
    if is_per:
        step = float(period) / float(n)
        return ax - step / 2.0, ax + step / 2.0
    mids = 0.5 * (ax[:-1] + ax[1:])
    step0 = float(ax[1] - ax[0])
    stepN = float(ax[-1] - ax[-2])
    lo = np.concatenate([[ax[0] - step0 / 2.0], mids])
    hi = np.concatenate([mids, [ax[-1] + stepN / 2.0]])
    return lo, hi


def _cell_masses_ma_absolute(dens, axes: list,
                             truncation_sigmas: float = 6.0) -> np.ndarray:
    """Cell masses on the Cartesian-product grid built from ``axes``.

    Parallels ``_entropy_exp_tens_ma``'s grid evaluation but returns
    ``int_{cell} f dx`` (via per-axis erf differences) instead of
    point-evaluated density values. Restricted to absolute-mode
    densities (every group ``is_rel=False``); the caller is responsible
    for routing relative-mode densities elsewhere.

    The output is flat ``(prod(n_axis_d),)`` in C order, matching the
    layout of ``numpy.meshgrid(*axes, indexing='ij').ravel()`` so it
    drops in where ``eval_exp_tens`` would return point values.
    """
    if np.any(np.asarray(dens.is_rel)):
        raise ValueError(
            "_cell_masses_ma_absolute: relative-mode densities are not "
            "supported by this path. Route is_rel=True via point-eval."
        )

    A = int(dens.n_attrs)
    dim_per = np.asarray(dens.dim_per_attr).astype(int)
    sigma_per_attr = np.asarray(dens.sigma).astype(float)  # (A,)
    is_per_g = np.asarray(dens.is_per).astype(bool)
    period_g = np.asarray(dens.period).astype(float)

    centres = dens.centres  # length-A list; each (dim_per[a], n_j)
    w_j = np.asarray(dens.w_j).astype(float)

    # Auto-prune zero-weight tuples. Mirrors the eval-path prune in
    # _tensor/eval.py: zero-weight tuples contribute exactly zero to
    # the einsum, so dropping them is mathematically exact and avoids
    # building (n_j, n_cells) erf-difference matrices over tuples that
    # weight_events has truncated to zero.
    mask = w_j > 0
    if not bool(np.all(mask)):
        w_j = w_j[mask]
        centres = [np.asarray(c, dtype=float)[:, mask] for c in centres]

    Mats = []  # one (n_j, n_cells_d) matrix per effective axis
    axis_d = 0
    for a in range(A):
        da = int(dim_per[a])
        sig = float(sigma_per_attr[a])
        is_per_a = bool(is_per_g[a])
        per_a = float(period_g[a]) if is_per_a else 0.0
        Ca = np.asarray(centres[a], dtype=float)  # (da, n_j)
        for sub in range(da):
            ax = axes[axis_d]
            lo, hi = _axis_edges(ax, is_per_a, per_a)
            cents = Ca[sub, :]
            if is_per_a:
                Mat = _phi_diff_axis_periodic(
                    cents, lo, hi, sig, per_a, truncation_sigmas)
            else:
                Mat = _phi_diff_axis(cents, lo, hi, sig)
            Mats.append(Mat)
            axis_d += 1

    D = axis_d
    if D == 0:
        return np.array([float(np.sum(w_j))])

    letters = "abcdefghijklmnopqrstuvwxyz"[:D]
    operands = ",".join("t" + L for L in letters)
    cells = np.einsum(f"t,{operands}->{letters}", w_j, *Mats)
    return cells.ravel()


def _cell_masses_sa_absolute(T, ax: np.ndarray,
                             truncation_sigmas: float = 6.0) -> np.ndarray:
    """Cell masses for an ``ExpTensDensity`` (single-attribute path).

    Parallels ``_cell_masses_ma_absolute`` for the SA case: builds the
    Cartesian-product grid implicitly from a single 1-D axis ``ax``
    repeated across the ``T.dim`` effective dimensions, returns flat
    cell masses. Restricted to ``is_rel=False`` densities.
    """
    if bool(T.is_rel):
        raise ValueError(
            "_cell_masses_sa_absolute: is_rel=True not supported by this path."
        )
    dim = int(T.dim)
    sig = float(T.sigma)
    is_per = bool(T.is_per)
    per = float(T.period) if is_per else 0.0
    C = np.asarray(T.centres, dtype=float)  # (r, n_j) == (dim, n_j) when is_rel=False
    w_j = np.asarray(T.w_j, dtype=float)

    # Auto-prune zero-weight tuples (see _cell_masses_ma_absolute).
    mask = w_j > 0
    if not bool(np.all(mask)):
        w_j = w_j[mask]
        C = C[:, mask]

    lo, hi = _axis_edges(ax, is_per, per)
    if is_per:
        Mat_template = _phi_diff_axis_periodic(
            C[0, :], lo, hi, sig, per, truncation_sigmas)
    else:
        Mat_template = _phi_diff_axis(C[0, :], lo, hi, sig)
    # We have a per-tuple per-axis matrix for axis 0. For dim>1 each
    # axis uses the same edges but a different centres row.
    Mats = [Mat_template]
    for d in range(1, dim):
        if is_per:
            Mats.append(_phi_diff_axis_periodic(
                C[d, :], lo, hi, sig, per, truncation_sigmas))
        else:
            Mats.append(_phi_diff_axis(C[d, :], lo, hi, sig))

    if dim == 0:
        return np.array([float(np.sum(w_j))])
    letters = "abcdefghijklmnopqrstuvwxyz"[:dim]
    operands = ",".join("t" + L for L in letters)
    cells = np.einsum(f"t,{operands}->{letters}", w_j, *Mats)
    return cells.ravel()


# ===================================================================
#  entropy_exp_tens
# ===================================================================


@_with_dispatch_scope
def entropy_exp_tens(
    p_or_dens,
    *args,
    spectrum: list | None = None,
    method: str = "shannon",
    precision: int | None = None,
    dedup: bool = True,
    base: float = 2.0,
    n_points_per_dim: int | None = None,
    x_min=float("nan"),
    x_max=float("nan"),
    grid_limit: int = _DEFAULT_GRID_LIMIT,
    truncation_sigmas: float | None = None,
    kernel_precision: str | None = None,
    verbose: bool = True,
    **legacy_kwargs,
) -> float | np.ndarray:
    """Entropy of an expectation tensor density.

    Four methods are supported. The discrete methods take an explicit
    grid; the continuous methods do not.

    - ``method='differential'``: adaptive nested-grid evaluation of the
      differential entropy ĥ = H_disc + log_b(Δ-volume). The span
      auto-derives from event centres ± ``truncation_sigmas * sigma``
      per group (non-periodic) or ``[0, period]`` (periodic); the grid
      doubles from a sample-per-sigma initial resolution until
      successive Richardson-extrapolated estimates fall below tolerance
      (anchored at ``max(exp(-truncation_sigmas^2 / 2), 1e-12)``).
      Grid-independent (no caller choice of grid), and the principled
      scale-free quantity for comparisons across densities of different
      cardinality or spread. Errors at ``sigma=0`` (the continuous form
      diverges).

    - ``method='shannon'``: raw discrete Shannon entropy
      ``H = -Σ q log_b q`` on an explicit Cartesian-product grid of
      resolution ``n_points_per_dim`` per effective dimension. Bin-mass
      integration via per-axis Φ-difference contractions
      (``is_rel=False``) or point-evaluation (``is_rel=True``). Supports
      the full polymorphic-input dispatch (single density, list of
      densities, raw scalar/batched SA, raw MA).

    - ``method='normalized'`` (alias ``'normalised'``): the Pielou-style
      ratio ``H / log_b(N)`` in ``[0, 1]``. Same grid as ``'shannon'``.
      Reproduces the values reported in Milne et al. (2017) and Smit
      et al. (2019).

    - ``method='renyi2'``: analytical Rényi-2 (collision) entropy via
      the Möbius inner product and the closed-form total mass. Returns
      ``H_2 = -log_b(<T,T> / Z²)``, the continuous Rényi-2 entropy of
      the normalised density ``q = T/Z``. Computed in closed form with
      no grid; works at arbitrary tensor order ``r`` where the grid
      path would exhaust memory. Currently restricted to single-density
      input. Errors at ``sigma=0`` (the continuous form diverges).

    The ``normalize`` kwarg of v2.1 has been removed; pick the
    appropriate ``method`` instead (a migration error is raised if
    ``normalize`` is passed).

    The continuous methods ('differential', 'renyi2') do not accept
    ``n_points_per_dim``, ``x_min``, or ``x_max``; the discrete methods
    ('shannon', 'normalized') require an explicit ``n_points_per_dim``
    (no toolbox-wide default).

    Input forms (Shannon and normalized support all; differential and
    Rényi-2 support only scalar forms — pre-built density, raw SA
    scalar, raw MA scalar — and raise ``NotImplementedError`` on list
    or batched forms):

    **Pre-built density input**:

    - ``entropy_exp_tens(dens)`` — scalar density.
      Returns a Python float.
    - ``entropy_exp_tens([d1, d2, …])`` — list of densities
      (Shannon/normalized only). Returns ``(M,)`` ndarray.

    **Raw single-attribute scalar input**:

    - ``entropy_exp_tens(p, w, sigma, r, is_rel, is_per, period)``.
      Returns a Python float. Optional ``spectrum``.

    **Raw single-attribute batched input** (Shannon/normalized only):

    - ``entropy_exp_tens(P, W, sigma, r, is_rel, is_per, period)``
      with ``P`` and ``W`` 2-D ``(M, K)`` matrices (rows are chords).
      Returns ``(M,)``. Optional ``spectrum``, ``precision``,
      ``dedup``.

    **Raw multi-attribute scalar input**:

    - ``entropy_exp_tens(p_attr, w, sigma_vec, r_vec,
      is_rel_vec, is_per_vec, period_vec)``. Returns a Python float.

    Parameters
    ----------
    p_or_dens : various
        See input forms above.
    *args : tuple
        Raw-args tail. Empty for density input; 6 trailing for raw SA;
        7 trailing for raw MA.
    spectrum : list or None
        Arguments for :func:`~mpt.spectra.add_spectra`. Raw SA only.
    method : {'differential', 'shannon', 'normalized', 'renyi2'}, default 'shannon'
        Entropy variant. See the introduction above. ``'normalised'``
        is accepted as an alias for ``'normalized'``.
    precision : int, optional
        Round canonical values to this many decimal places, to absorb
        FP noise when deduplicating. Raw SA batched only
        (Shannon / normalized).
    dedup : bool, default True
        Deduplicate structurally-identical chords. List/batch only
        (Shannon / normalized).
    base : float, default 2.0
        Logarithm base. For ``method='normalized'`` the base cancels.
    n_points_per_dim : int, optional
        Required for ``method='shannon'`` and ``method='normalized'``;
        ignored by the continuous methods. The previous toolbox-wide
        default of 1200 has been dropped, since the right grid
        resolution is density- and sigma-dependent.
    x_min, x_max
        Shannon / normalized, non-periodic only: grid bounds. SA scalar;
        MA scalar (broadcast) or length-G vector.
    grid_limit : int
        Ceiling on total grid size before allocation.
    truncation_sigmas : float, optional
        Truncate the kernel beyond this many sigmas. For
        ``method='differential'`` this also anchors the convergence
        tolerance ``max(exp(-truncation_sigmas^2 / 2), 1e-12)``.

    Returns
    -------
    float or np.ndarray
        Scalar in scalar input modes; ``(M,)`` ndarray in list/batch
        modes. The continuous methods always return a scalar (they are
        currently restricted to scalar input).

    Notes
    -----
    Numerical precision envelope for ``method='renyi2'``.

    The Möbius method is exact to floating-point precision when
    every per-attribute ``K_a`` satisfies ``K_a >= r_a + 2`` and σ is
    not catastrophically small relative to P. The dispatcher enforces
    these conditions structurally — it routes to Bulger's method
    when ``K_a < r_a + 2``, when ``σ/P > 0.03`` in periodic-relative
    mode, or when the σ → 0 fallback heuristic triggers. A post-hoc
    check on the Möbius-method self-IP raises ``FloatingPointError`` if the
    result is non-finite, non-positive, or sign-flipped.

    What is *not* currently caught: a finite, positive, but slightly
    inaccurate self-IP from sub-catastrophic Möbius cancellation in
    the Möbius alternating partition sum. No instance of this has been
    observed in extensive testing (1475 cells covering
    ``r ∈ {2..6}``, K up to 100, σ down to ``10⁻⁵`` cents, all four
    mode combinations, multi-attribute self-IPs, adversarial pitch
    configurations, and harmonic spectra up to K=64). Within typical
    music-cognition usage the returned Rényi-2 entropy is therefore
    treated as bit-exact. A sum-level cancellation diagnostic that
    would close this residual gap is on the v2.3 roadmap.

    For ``method='shannon'``, accuracy is set by the grid resolution
    ``n_points_per_dim`` and is independent of the Möbius method.
    """
    # Detect the legacy normalize kwarg (removed in v2.2) and emit a
    # migration error pointing to the four-method API. Other unknown
    # kwargs surface as a standard TypeError.
    if "normalize" in legacy_kwargs:
        raise TypeError(_NORMALIZE_REMOVED_MSG.format(fn="entropy_exp_tens"))
    if legacy_kwargs:
        unknown = ", ".join(repr(k) for k in legacy_kwargs)
        raise TypeError(
            f"entropy_exp_tens: unexpected keyword argument(s): {unknown}"
        )

    # Canonicalize the method kwarg (accepts British 'normalised') and
    # validate against the four supported methods.
    method = _canonicalize_method(method)

    # 'differential' routes to the adaptive nested-grid evaluator.
    if method == "differential":
        return _entropy_exp_tens_differential_dispatch(
            p_or_dens, args,
            spectrum=spectrum, precision=precision, dedup=dedup, base=base,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
            grid_limit=grid_limit, verbose=verbose,
        )

    # Discrete methods: 'normalized' forces H/log_b(N) in [0, 1];
    # 'shannon' is the raw discrete H = -sum q log_b q. Both share the
    # bin-integration core via _entropy_exp_tens_shannon_dispatch.
    if method == "normalized":
        return _entropy_exp_tens_shannon_dispatch(
            p_or_dens, args,
            spectrum=spectrum, precision=precision, dedup=dedup,
            normalize=True, base=base,
            n_points_per_dim=n_points_per_dim,
            x_min=x_min, x_max=x_max, grid_limit=grid_limit,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
            verbose=verbose,
        )

    if method == "shannon":
        return _entropy_exp_tens_shannon_dispatch(
            p_or_dens, args,
            spectrum=spectrum, precision=precision, dedup=dedup,
            normalize=False, base=base,
            n_points_per_dim=n_points_per_dim,
            x_min=x_min, x_max=x_max, grid_limit=grid_limit,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
            verbose=verbose,
        )

    # 'renyi2': analytical, continuous-form, no [0, 1] reference.
    return _entropy_exp_tens_renyi2_dispatch(
        p_or_dens, args,
        spectrum=spectrum, precision=precision, dedup=dedup, base=base,
    )


# =========================================================================
#  _entropy_exp_tens_shannon_dispatch — input-form resolution for Shannon
# =========================================================================


def _entropy_exp_tens_shannon_dispatch(
    p_or_dens, args, *,
    spectrum, precision, dedup,
    normalize, base, n_points_per_dim, x_min, x_max, grid_limit,
    truncation_sigmas, kernel_precision, verbose,
):
    """Resolve input form and route to SA / MA helper, Shannon path.

    Shannon entropy of the density evaluated on a Cartesian-product
    grid; supports the full input surface (precomputed density object,
    list of densities, MA raw args, SA raw args, SA batched 2-D matrix).
    """
    # --- Density inputs first (scalar or list) ---
    if isinstance(p_or_dens, (ExpTensDensity, MaetDensity, WindowedMaetDensity)):
        if len(args) > 0:
            raise TypeError(
                f"Precomputed density takes no further positional args; "
                f"got {len(args)}."
            )
        if spectrum is not None or precision is not None:
            raise TypeError(
                "'spectrum' and 'precision' kwargs are only valid in raw input mode."
            )
        _require_explicit_grid(n_points_per_dim)
        return _entropy_exp_tens_scalar(
            p_or_dens,
            normalize=normalize, base=base,
            n_points_per_dim=n_points_per_dim,
            x_min=x_min, x_max=x_max, grid_limit=grid_limit,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
        )

    # Density list dispatch (list/tuple/object-array of densities)
    intends_density_list = False
    if isinstance(p_or_dens, (list, tuple)):
        if len(p_or_dens) == 0:
            intends_density_list = True
        elif isinstance(
            p_or_dens[0], (ExpTensDensity, MaetDensity, WindowedMaetDensity)
        ):
            intends_density_list = True
    elif isinstance(p_or_dens, np.ndarray) and p_or_dens.dtype == object:
        intends_density_list = True

    if intends_density_list:
        if len(args) > 0:
            raise TypeError(
                f"Density list input takes no further positional args; "
                f"got {len(args)}."
            )
        if spectrum is not None or precision is not None:
            raise TypeError(
                "'spectrum' and 'precision' kwargs are only valid in raw input mode."
            )
        _require_explicit_grid(n_points_per_dim)
        return _entropy_exp_tens_density_list(
            p_or_dens,
            dedup=dedup,
            normalize=normalize, base=base,
            n_points_per_dim=n_points_per_dim,
            x_min=x_min, x_max=x_max, grid_limit=grid_limit,
        )

    # --- Raw args: dispatch on type of p ---
    # Positional order: ..., period[, is_sym]. 6 trailing args omit
    # is_sym (defaults symmetric); 7 supply it.
    if _looks_like_ma_p(p_or_dens):
        if len(args) not in (6, 7):
            raise ValueError(
                f"Multi-attribute raw call expects 7 or 8 positional "
                f"arguments (p_attr, w, sigma_vec, r_vec, is_rel_vec, "
                f"is_per_vec, period_vec[, is_sym_vec]); got {1 + len(args)}."
            )
        if spectrum is not None:
            raise TypeError(
                "'spectrum' kwarg is only valid in raw single-attribute "
                "input mode."
            )
        if precision is not None:
            raise TypeError(
                "'precision' kwarg is only valid in raw SA batched input mode."
            )
        w, sigma_vec, r_vec, is_rel_vec, is_per_vec, period_vec = args[:6]
        is_sym_vec = args[6] if len(args) == 7 else None
        dens = build_exp_tens(
            p_or_dens, w, sigma_vec, r_vec,
            is_rel_vec, is_per_vec, period_vec, is_sym_vec,
            verbose=False,
        )
        _require_explicit_grid(n_points_per_dim)
        return _entropy_exp_tens_ma(
            dens,
            normalize=normalize, base=base,
            n_points_per_dim=n_points_per_dim,
            x_min=x_min, x_max=x_max, grid_limit=grid_limit,
        )

    # SA raw args. Distinguish scalar (1-D) from batched (2-D) by shape.
    if len(args) not in (6, 7):
        raise ValueError(
            f"Single-attribute raw call expects 7 or 8 positional arguments "
            f"(p, w, sigma, r, is_rel, is_per, period[, is_sym]); "
            f"got {1 + len(args)}."
        )
    w, sigma, r, is_rel, is_per, period = args[:6]
    is_sym = args[6] if len(args) == 7 else None

    try:
        p_arr = np.asarray(p_or_dens, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"First argument must be a density object, list of densities, "
            f"numeric array (1-D for a single chord, 2-D for a batch), or "
            f"list of per-attribute matrices for MA raw input; got "
            f"{type(p_or_dens).__name__}."
        ) from exc

    if p_arr.ndim == 1:
        if precision is not None:
            raise TypeError(
                "'precision' kwarg is only valid for raw SA batched input."
            )
        _require_explicit_grid(n_points_per_dim)
        return _entropy_exp_tens_sa(
            p_or_dens, w, sigma, r, is_rel, is_per, period, is_sym,
            spectrum=spectrum, normalize=normalize, base=base,
            n_points_per_dim=n_points_per_dim,
            x_min=x_min, x_max=x_max,
        )
    if p_arr.ndim == 2:
        _require_explicit_grid(n_points_per_dim)
        return _entropy_exp_tens_raw_sa_batch(
            p_arr, w, sigma, r, is_rel, is_per, period, is_sym,
            spectrum=spectrum, precision=precision,
            dedup=dedup,
            normalize=normalize, base=base,
            n_points_per_dim=n_points_per_dim,
            x_min=x_min, x_max=x_max, grid_limit=grid_limit,
            verbose=verbose,
        )
    raise TypeError(
        f"First argument has unsupported shape {p_arr.shape}; "
        f"raw SA input must be 1-D (single chord) or 2-D (batched)."
    )


# =========================================================================
#  _entropy_exp_tens_renyi2_dispatch — input-form resolution for Rényi-2
# =========================================================================


def _entropy_exp_tens_renyi2_dispatch(
    p_or_dens, args, *,
    spectrum, precision, dedup, base,
):
    """Resolve input form and route to SA / MA helper, Rényi-2 path.

    Analytical Rényi-2 (collision) entropy via the orbit-Möbius
    inner-product machinery. Restricted to single-density input
    (scalar density object, raw scalar SA, or raw scalar MA). List
    and batched input forms raise ``NotImplementedError``. Windowed
    MA is also not yet supported.
    """
    # Reject list inputs explicitly with a helpful message.
    if isinstance(p_or_dens, (list, tuple)):
        if len(p_or_dens) > 0 and isinstance(
            p_or_dens[0],
            (ExpTensDensity, MaetDensity, WindowedMaetDensity),
        ):
            raise NotImplementedError(
                "method='renyi2' does not yet support list input. "
                "Apply it to each density individually."
            )
    elif isinstance(p_or_dens, np.ndarray) and p_or_dens.dtype == object:
        raise NotImplementedError(
            "method='renyi2' does not yet support list input. "
            "Apply it to each density individually."
        )
    # Reject 2-D raw SA input (batched) explicitly.
    if (not isinstance(
            p_or_dens,
            (ExpTensDensity, MaetDensity, WindowedMaetDensity),
        )
        and not _looks_like_ma_p(p_or_dens)):
        try:
            p_arr_check = np.asarray(p_or_dens, dtype=np.float64)
            if p_arr_check.ndim == 2:
                raise NotImplementedError(
                    "method='renyi2' does not yet support raw SA "
                    "batched (2-D) input. Pass each chord row "
                    "individually, or pre-build densities."
                )
        except (TypeError, ValueError):
            pass  # let _resolve_density produce a clearer error
    if precision is not None or dedup is not True:
        raise TypeError(
            "'precision' and 'dedup' kwargs are only valid for "
            "method='shannon'."
        )
    dens, is_sa = _resolve_density(p_or_dens, args, spectrum)

    # Policy: the analytical Rényi-2 is the differential (continuous)
    # Rényi-2, -log <f,f>, which diverges at sigma=0 (the self-IP of a
    # sum of deltas blows up). Reject sigma=0 explicitly so users see a
    # clear message rather than a divergent number. The discrete Rényi-2
    # (-log sum q^2) on a category grid is the right object at sigma=0
    # but is a separate computation not currently exposed.
    _raise_if_any_sigma_zero(dens, method_name="renyi2")

    # Announce dispatch for the Rényi-2 path. Analytical Möbius is the
    # only method for Rényi-2 (no probe, no method choice), so the
    # message is the unprobed form: ``entropy_exp_tens: chose 'mobius'
    # path.`` Parity with MATLAB ``localEntropyRenyi2Dispatch``.
    from ._defaults import _maybe_show_dispatch_msg
    _maybe_show_dispatch_msg(
        "entropy_exp_tens", "mobius", "renyi2",
        est_sec=0.0, is_probed=False,
    )
    if is_sa:
        return _renyi2_exp_tens_sa(dens, base=base)
    return _renyi2_exp_tens_ma(dens, base=base)


# ===================================================================
#  Adaptive differential entropy (method='differential')
# ===================================================================
#
# h_hat = H_disc + log_b(cell_volume), converged on a per-axis nested-grid
# refinement to a truncation-sigma-anchored tolerance. The span auto-
# derives per group from `centres ± truncation_sigmas * sigma` (non-
# periodic) or `[0, period]` (periodic). The initial resolution is set
# at ~2 samples per sigma per axis; N then doubles each iteration until
# successive h_hat differences fall below tolerance, the differences
# stop decreasing (numerical-floor guard), or `grid_limit` is hit.


def _diff_spans_sa(T, ts: float):
    """Auto-spans for an ExpTensDensity (single attribute).

    Returns
    -------
    x_min, x_max : float
        Span bounds (only meaningful for non-periodic).
    n0 : int
        Initial N per axis (~2 samples per sigma).
    dim : int
        Effective dimension (= T.dim).
    per_axis_W : list of float
        Width per axis (period if periodic, x_max - x_min otherwise).
    per_axis_per : list of bool
        Periodicity flag per axis.
    """
    sig = float(T.sigma)
    is_per = bool(T.is_per)
    per = float(T.period)
    dim = int(T.dim)

    if is_per:
        x_min = float("nan")  # not consumed by the periodic grid construction
        x_max = float("nan")
        W = per
    else:
        # ExpTensDensity centres: (dim, n_j) when is_rel=False
        centres = np.asarray(T.centres, dtype=float)
        c_flat = centres.ravel()
        c_min = float(c_flat.min())
        c_max = float(c_flat.max())
        x_min = c_min - ts * sig
        x_max = c_max + ts * sig
        W = x_max - x_min

    per_axis_W = [W] * dim
    per_axis_per = [is_per] * dim
    n0 = max(4, int(np.ceil(2.0 * W / sig)))
    return x_min, x_max, n0, dim, per_axis_W, per_axis_per


def _diff_spans_ma(dens, ts: float):
    """Auto-spans for a MaetDensity.

    Returns x_min_a, x_max_a (length-A arrays; NaN for periodic
    attributes), initial N (max across attributes), total dim, per-axis
    widths, and per-axis periodicity flags (in attribute-then-sub-axis
    order, matching the grid layout used by _entropy_exp_tens_ma).
    """
    A = int(dens.n_attrs)
    dim_per = np.asarray(dens.dim_per_attr).astype(int)
    sigma_a = np.asarray(dens.sigma).astype(float)   # (A,)
    is_per_g = np.asarray(dens.is_per).astype(bool)  # (A,)
    period_g = np.asarray(dens.period).astype(float)
    centres = dens.centres  # length-A list

    x_min_g = np.full(A, float("nan"))
    x_max_g = np.full(A, float("nan"))
    n0_per_attr = np.zeros(A, dtype=int)

    for a in range(A):
        sig = float(sigma_a[a])
        if bool(is_per_g[a]):
            W_g = float(period_g[a])
        else:
            c_flat = np.asarray(centres[a], dtype=float).ravel()
            if c_flat.size:
                c_min = float(c_flat.min())
                c_max = float(c_flat.max())
            else:
                c_min = c_max = 0.0
            x_min_g[a] = c_min - ts * sig
            x_max_g[a] = c_max + ts * sig
            W_g = x_max_g[a] - x_min_g[a]
        n0_per_attr[a] = max(4, int(np.ceil(2.0 * W_g / sig)))

    per_axis_W = []
    per_axis_per = []
    for a in range(A):
        if bool(is_per_g[a]):
            W_a = float(period_g[a])
        else:
            W_a = float(x_max_g[a] - x_min_g[a])
        for _ in range(int(dim_per[a])):
            per_axis_W.append(W_a)
            per_axis_per.append(bool(is_per_g[a]))

    dim = int(np.sum(dim_per))
    n0 = int(n0_per_attr.max()) if n0_per_attr.size > 0 else 4
    return x_min_g, x_max_g, n0, dim, per_axis_W, per_axis_per


def _differential_adaptive(
    dens, *, is_sa: bool, base: float,
    truncation_sigmas: float, kernel_precision,
    grid_limit: int, verbose: bool,
) -> float:
    """Adaptive nested-grid evaluation of differential entropy h_hat.

    Reuses the existing Shannon dispatch for the per-grid H_disc and
    adds log_b(cell_volume) to produce h_hat. Doubles N from the
    sample-per-sigma initial size until convergence to the truncation-
    sigma-anchored tolerance, the floor is reached, or grid_limit hits.
    """
    import math
    ts = float(truncation_sigmas)
    # truncation_sigmas controls kernel truncation, where +inf is valid
    # ("no truncation"). The differential span and tolerance anchoring
    # need a finite extent, so cap any non-finite ts at the sensible
    # default 6.0 -- this matches the _cell_masses_ma_absolute internal
    # default and keeps span/tolerance well-defined when the user (or
    # mpt.get_default('truncation_sigmas')) is set to inf.
    ts_span = ts if math.isfinite(ts) else 6.0
    tol = max(math.exp(-0.5 * ts_span * ts_span), 1e-12)
    max_iter = 10

    if is_sa:
        x_min, x_max, n0, dim, per_axis_W, per_axis_per = _diff_spans_sa(dens, ts_span)
    else:
        x_min_g, x_max_g, n0, dim, per_axis_W, per_axis_per = _diff_spans_ma(dens, ts_span)

    N = max(int(n0), 4)
    h_history = []   # list of h_hat values, in iteration order
    R_history = []   # list of Richardson-extrapolated estimates
    log_b = math.log(base)

    for _iter in range(max_iter):
        if dim > 0:
            total = N ** dim
            if total > grid_limit:
                if not h_history:
                    raise ValueError(
                        f"method='differential' needs grid_limit >= "
                        f"{total} for an initial N={N} at dim={dim}; got "
                        f"grid_limit={grid_limit}. Increase grid_limit, "
                        f"or use method='renyi2' (closed-form, no grid)."
                    )
                if verbose:
                    warnings.warn(
                        f"method='differential' hit grid_limit={grid_limit} "
                        f"at N={N} (dim={dim}); returning h_hat from the "
                        f"last feasible grid -- may not be fully converged.",
                        UserWarning, stacklevel=2,
                    )
                break

        if is_sa:
            H_disc = _entropy_exp_tens_sa(
                dens, None, None, None, None, None, None,
                spectrum=None, normalize=False, base=base,
                n_points_per_dim=N, x_min=x_min, x_max=x_max,
                truncation_sigmas=ts, kernel_precision=kernel_precision,
            )
        else:
            H_disc = _entropy_exp_tens_ma(
                dens, normalize=False, base=base,
                n_points_per_dim=N,
                x_min=x_min_g, x_max=x_max_g,
                grid_limit=grid_limit,
                truncation_sigmas=ts, kernel_precision=kernel_precision,
            )

        # Sum log_base(delta_d) over axes.
        log_cell_vol = 0.0
        for is_per_d, W_d in zip(per_axis_per, per_axis_W):
            delta = W_d / N if is_per_d else W_d / (N - 1)
            log_cell_vol += math.log(delta) / log_b
        h_hat = float(H_disc + log_cell_vol)
        h_history.append(h_hat)

        if len(h_history) >= 2:
            h_prev, h_curr = h_history[-2], h_history[-1]
            # Direct h_hat convergence (1-D regime: fast).
            if abs(h_curr - h_prev) < tol:
                return h_curr
            # Richardson extrapolation: bin-integration h_hat converges
            # at second order in Δ, so h ≈ h_curr + (h_curr - h_prev)/3
            # has fourth-order error -- O(Δ⁴), i.e. 16× per doubling.
            # Crucial in dim >= 2 where O(Δ²) alone is too slow.
            R = h_curr + (h_curr - h_prev) / 3.0
            R_history.append(R)
            if len(R_history) >= 2:
                dR = abs(R_history[-1] - R_history[-2])
                if dR < tol:
                    return R_history[-1]
                # Floor guard on the Richardson sequence: stop when the
                # successive Richardson differences stop shrinking.
                if len(R_history) >= 3:
                    dR_prev = abs(R_history[-2] - R_history[-3])
                    if dR_prev > 0 and dR >= 0.95 * dR_prev:
                        return R_history[-1]
        N *= 2

    # Fell off the loop without an explicit-tolerance return: prefer
    # Richardson if available (4th-order), else last raw h_hat.
    return R_history[-1] if R_history else h_history[-1]


def _entropy_exp_tens_differential_dispatch(
    p_or_dens, args, *,
    spectrum, precision, dedup, base,
    truncation_sigmas, kernel_precision,
    grid_limit, verbose,
):
    """Adaptive differential entropy dispatch (parallel to renyi2 path).

    Single-density input only; list and batched input forms raise
    NotImplementedError. Windowed MA is also not yet supported.
    """
    # Reject list inputs.
    if isinstance(p_or_dens, (list, tuple)):
        if len(p_or_dens) > 0 and isinstance(
            p_or_dens[0],
            (ExpTensDensity, MaetDensity, WindowedMaetDensity),
        ):
            raise NotImplementedError(
                "method='differential' does not yet support list input. "
                "Apply it to each density individually."
            )
    elif isinstance(p_or_dens, np.ndarray) and p_or_dens.dtype == object:
        raise NotImplementedError(
            "method='differential' does not yet support list input. "
            "Apply it to each density individually."
        )
    # Reject 2-D raw SA input (batched).
    if (not isinstance(
            p_or_dens,
            (ExpTensDensity, MaetDensity, WindowedMaetDensity),
        )
        and not _looks_like_ma_p(p_or_dens)):
        try:
            p_arr_check = np.asarray(p_or_dens, dtype=np.float64)
            if p_arr_check.ndim == 2:
                raise NotImplementedError(
                    "method='differential' does not yet support raw SA "
                    "batched (2-D) input. Pass each chord row "
                    "individually, or pre-build densities."
                )
        except (TypeError, ValueError):
            pass
    if precision is not None or dedup is not True:
        raise TypeError(
            "'precision' and 'dedup' kwargs are only valid for "
            "method='shannon'."
        )

    dens, is_sa = _resolve_density(p_or_dens, args, spectrum)
    _raise_if_any_sigma_zero(dens, method_name="differential")

    if isinstance(dens, WindowedMaetDensity):
        raise NotImplementedError(
            "method='differential' with WindowedMaetDensity is not yet "
            "implemented. Apply differential entropy to the unwindowed "
            "density, or use a Shannon/normalized path with an explicit "
            "grid for windowed densities."
        )

    if truncation_sigmas is None:
        ts = 6.0  # match _cell_masses_ma_absolute internal default
    else:
        ts = float(truncation_sigmas)

    return _differential_adaptive(
        dens, is_sa=is_sa, base=base,
        truncation_sigmas=ts, kernel_precision=kernel_precision,
        grid_limit=grid_limit, verbose=verbose,
    )


def _entropy_exp_tens_scalar(
    dens, *, normalize, base, n_points_per_dim, x_min, x_max, grid_limit,
    truncation_sigmas=None, kernel_precision=None,
):
    """Single-density entropy dispatch."""
    eval_kw = dict(
        truncation_sigmas=truncation_sigmas,
        kernel_precision=kernel_precision,
    )
    if isinstance(dens, WindowedMaetDensity):
        return _entropy_exp_tens_ma(
            dens,
            normalize=normalize, base=base,
            n_points_per_dim=n_points_per_dim,
            x_min=x_min, x_max=x_max, grid_limit=grid_limit,
            **eval_kw,
        )
    if isinstance(dens, MaetDensity):
        return _entropy_exp_tens_ma(
            dens,
            normalize=normalize, base=base,
            n_points_per_dim=n_points_per_dim,
            x_min=x_min, x_max=x_max, grid_limit=grid_limit,
            **eval_kw,
        )
    if isinstance(dens, ExpTensDensity):
        return _entropy_exp_tens_sa(
            dens, None, None, None, None, None, None,
            spectrum=None, normalize=normalize, base=base,
            n_points_per_dim=n_points_per_dim,
            x_min=x_min, x_max=x_max,
            **eval_kw,
        )
    raise TypeError(
        f"dens must be an ExpTensDensity, MaetDensity, or "
        f"WindowedMaetDensity; got {type(dens).__name__}."
    )


def _entropy_exp_tens_density_list(
    dens_list, *, dedup, normalize, base, n_points_per_dim,
    x_min, x_max, grid_limit,
):
    """List-of-densities entropy dispatch.

    With ``dedup=True``, structurally-identical SA densities are
    computed once (canonical-form dedup); MA densities bypass dedup.
    Returns ``(M,)``.
    """
    # Import lazily to avoid circular import at module load time.
    from .tensor import _chord_canonical_key

    dens_list = list(dens_list)
    m = len(dens_list)
    if m == 0:
        return np.empty((0,), dtype=np.float64)

    use_dedup = dedup and all(isinstance(d, ExpTensDensity) for d in dens_list)
    out = np.empty(m, dtype=np.float64)

    if use_dedup:
        result_cache: dict = {}
        for i, d in enumerate(dens_list):
            key, _, _ = _chord_canonical_key(
                d.p, d.w, sigma=d.sigma, r=d.r,
                is_rel=d.is_rel, is_per=d.is_per, period=d.period,
            )
            if key not in result_cache:
                result_cache[key] = _entropy_exp_tens_scalar(
                    d, normalize=normalize, base=base,
                    n_points_per_dim=n_points_per_dim,
                    x_min=x_min, x_max=x_max, grid_limit=grid_limit,
                )
            out[i] = result_cache[key]
    else:
        for i, d in enumerate(dens_list):
            if not isinstance(
                d, (ExpTensDensity, MaetDensity, WindowedMaetDensity)
            ):
                raise TypeError(
                    f"Density list element {i} must be a density object; "
                    f"got {type(d).__name__}."
                )
            out[i] = _entropy_exp_tens_scalar(
                d, normalize=normalize, base=base,
                n_points_per_dim=n_points_per_dim,
                x_min=x_min, x_max=x_max, grid_limit=grid_limit,
            )
    return out


def _entropy_exp_tens_raw_sa_batch(
    P, W, sigma, r, is_rel, is_per, period, is_sym=None,
    *, spectrum, precision, dedup,
    normalize, base, n_points_per_dim, x_min, x_max, grid_limit,
    verbose=True,
):
    """Raw SA batched entropy dispatch.

    Per-row chord-level dedup of density construction (via canonical
    keys); each unique density's entropy is computed once. Returns
    ``(M,)`` with ``np.nan`` for invalid rows (K < r).
    """
    import time
    from .tensor import _chord_canonical_key

    P = np.asarray(P, dtype=np.float64)
    M, K = P.shape

    # The per-row dedup keys rows by a multiset canonical form, which
    # collapses rows that share a multiset but differ in order. That is
    # correct only for the symmetric reading: under [sym]=0 the order is
    # significant, so the dedup would silently merge distinct ordered
    # densities (and hence entropies). Reject rather than return a wrong
    # answer. Order-aware batched dedup is a tracked follow-up; use
    # scalar input for ordered densities.
    if (is_sym is not None) and (not bool(np.all(is_sym))) and r > 1:
        raise NotImplementedError(
            "entropy_exp_tens batched (2-D) input does not yet support "
            "[sym]=0 (ordered) densities at r > 1: the batched dedup "
            "canonicalises each row's multiset and would merge "
            "order-distinct rows. Compute ordered densities one row at a "
            "time (scalar input)."
        )

    use_w = W is not None
    if use_w:
        W = np.asarray(W, dtype=np.float64)
        if W.shape != P.shape:
            raise ValueError("W must be the same shape as P.")

    # Input precision rounding (collapses FP-noise rows).
    if precision is not None:
        P = np.round(P, precision)
        if use_w:
            W = np.round(W, precision)

    # Up-front time estimate (printed once for the whole batch).
    # Empirical calibration with warm-up; see
    # _template_harmonicity_batched for rationale.
    # Adaptive progress-print state. Defaults: silent.
    prog_stride = 1
    show_progress = False
    if verbose and M > 1:
        n_cal = min(10, M)
        sample_idx = np.unique(np.linspace(0, M - 1, n_cal).astype(int))

        def _run_one(s_idx):
            p_row_s = P[s_idx]
            mask_s = ~np.isnan(p_row_s)
            p_valid_s = p_row_s[mask_s]
            if len(p_valid_s) < r:
                return False
            w_valid_s = W[s_idx, mask_s] if use_w else None
            if spectrum is not None:
                p_aug, w_aug = add_spectra(
                    p_valid_s,
                    np.ones_like(p_valid_s) if w_valid_s is None else w_valid_s,
                    *spectrum,
                )
            else:
                p_aug = p_valid_s
                w_aug = w_valid_s
            T = build_exp_tens(
                p_aug, w_aug, sigma, r, is_rel, is_per, period,
                True if is_sym is None else is_sym, verbose=False,
            )
            _entropy_exp_tens_scalar(
                T, normalize=normalize, base=base,
                n_points_per_dim=n_points_per_dim,
                x_min=x_min, x_max=x_max, grid_limit=grid_limit,
            )
            return True

        # Warm-up
        warmup_done = False
        for s_idx in sample_idx:
            if _run_one(s_idx):
                warmup_done = True
                break

        if warmup_done:
            t_cal_start = time.perf_counter()
            n_valid_cal = 0
            for s_idx in sample_idx:
                if _run_one(s_idx):
                    n_valid_cal += 1
            if n_valid_cal > 0:
                from ._utils import progress_stride
                t_cal_total = time.perf_counter() - t_cal_start
                t_per_row = t_cal_total / n_valid_cal
                est_total = t_cal_total + t_per_row * M
                maybe_print_batched_estimate(

                    "entropy_exp_tens", M, est_total,

                )
                prog_stride = progress_stride(t_per_row)
                show_progress = est_total >= 5

    out = np.full(M, np.nan)

    dens_cache: dict = {}
    entropy_cache: dict = {}
    row_to_key: list = [None] * M
    for i in range(M):
        p_row = P[i]
        mask = ~np.isnan(p_row)
        p_valid = p_row[mask]
        if len(p_valid) < r:
            continue
        w_valid = W[i, mask] if use_w else None
        key, p_canon, w_canon = _chord_canonical_key(
            p_valid, w_valid,
            sigma=sigma, r=r, is_rel=is_rel, is_per=is_per, period=period,
            precision=precision,
        )
        if key not in dens_cache:
            if spectrum is not None:
                p_canon, w_canon_aug = add_spectra(
                    p_canon,
                    np.ones_like(p_canon) if w_canon is None else w_canon,
                    *spectrum,
                )
                w_canon = w_canon_aug
            dens_cache[key] = build_exp_tens(
                p_canon, w_canon, sigma, r, is_rel, is_per, period,
                True if is_sym is None else is_sym,
                verbose=False,
            )
        if not dedup or key not in entropy_cache:
            entropy_cache[key] = _entropy_exp_tens_scalar(
                dens_cache[key], normalize=normalize, base=base,
                n_points_per_dim=n_points_per_dim,
                x_min=x_min, x_max=x_max, grid_limit=grid_limit,
            )
        row_to_key[i] = key

        if verbose and show_progress \
                and ((i + 1) % prog_stride == 0 or i == M - 1):
            print(f"  {i + 1} / {M} rows computed.")

    for i, key in enumerate(row_to_key):
        if key is not None:
            out[i] = entropy_cache[key]
    return out


# -------------------------------------------------------------------
#  _resolve_density  (shared by Shannon and renyi2 single-density paths)
# -------------------------------------------------------------------


def _resolve_density(p_or_dens, args, spectrum):
    """Coerce the ``entropy_exp_tens`` first argument plus tail args
    into either an :class:`ExpTensDensity` (SA) or a
    :class:`MaetDensity` / :class:`WindowedMaetDensity` (MA),
    independent of the entropy estimator. Returns ``(dens, is_sa)``.

    Centralises the build_exp_tens / spectrum / passthrough logic so
    that both the Shannon and renyi2 branches see a uniform input.
    Used only on the single-density input path; the polymorphic
    Shannon dispatch in ``entropy_exp_tens`` handles list / batched
    inputs separately.
    """
    # --- Dispatch on precomputed densities first ---
    if isinstance(p_or_dens, WindowedMaetDensity):
        if len(args) > 0:
            raise TypeError(
                "Precomputed WindowedMaetDensity takes no further positional args."
            )
        return p_or_dens, False
    if isinstance(p_or_dens, MaetDensity):
        if len(args) > 0:
            raise TypeError(
                "Precomputed MaetDensity takes no further positional args."
            )
        return p_or_dens, False
    if isinstance(p_or_dens, ExpTensDensity):
        if len(args) > 0:
            raise TypeError(
                "Precomputed ExpTensDensity takes no further positional args."
            )
        return p_or_dens, True

    # --- Raw args: dispatch on type of p ---
    # Positional order: ..., period[, is_sym]. 6 trailing args omit
    # is_sym (defaults symmetric); 7 supply it.
    if _looks_like_ma_p(p_or_dens):
        if len(args) not in (6, 7):
            raise ValueError(
                f"Multi-attribute raw call expects 7 or 8 positional "
                f"arguments (p_attr, w, sigma_vec, r_vec, is_rel_vec, "
                f"is_per_vec, period_vec[, is_sym_vec]); got {1 + len(args)}."
            )
        w, sigma_vec, r_vec, is_rel_vec, is_per_vec, period_vec = args[:6]
        is_sym_vec = args[6] if len(args) == 7 else None
        dens = build_exp_tens(
            p_or_dens, w, sigma_vec, r_vec,
            is_rel_vec, is_per_vec, period_vec, is_sym_vec,
            verbose=False,
        )
        return dens, False

    # SA raw args.
    if len(args) not in (6, 7):
        raise ValueError(
            f"Single-attribute raw call expects 7 or 8 positional arguments "
            f"(p, w, sigma, r, is_rel, is_per, period[, is_sym]); "
            f"got {1 + len(args)}."
        )
    w, sigma, r, is_rel, is_per, period = args[:6]
    is_sym = args[6] if len(args) == 7 else None
    p = np.asarray(p_or_dens, dtype=np.float64).ravel()
    if spectrum is not None:
        p, w = add_spectra(p, w, *spectrum)
    dens = build_exp_tens(
        p, w, sigma, r, is_rel, is_per, period,
        True if is_sym is None else is_sym, verbose=False,
    )
    return dens, True


# -------------------------------------------------------------------
#  Rényi-2 entropy helpers (analytical, Möbius)
# -------------------------------------------------------------------


def _renyi2_finalise(ip_xx, Z, base):
    """Return ``H_2 = -log_b(ip_xx / Z**2)``, or NaN for a degenerate
    (zero-mass / non-finite) density.

    A zero-mass density (every weight zero, or a window with no event in
    support) has ``<T,T> = 0`` and ``Z = 0`` exactly, so its collision
    entropy is undefined. Returning NaN rather than raising is friendlier
    for sweep-style callers: a windowed sweep already wants NaN at
    out-of-support centres, and the caller need not wrap each evaluation
    in ``try``/``except``. (Finite-precision catastrophic cancellation in
    a self-inner-product is not produced analytically and has not been
    observed; if a genuine cancellation regime ever surfaces it would
    also land here as NaN rather than a wrong number.)
    """
    if (not np.isfinite(ip_xx)) or ip_xx <= 0.0 \
            or (not np.isfinite(Z)) or Z <= 0.0:
        return float("nan")
    return -float(np.log(ip_xx / (Z * Z)) / np.log(base))


def _renyi2_exp_tens_sa(dens, *, base: float) -> float:
    """Analytical Rényi-2 entropy of a SA expectation tensor.

    Computes ``H_2 = -log_b(<T,T> / Z²)`` where ``<T,T>`` is evaluated
    via the Möbius inner-product machinery (or a direct
    pairwise formula at ``r = 1`` where the orbit table is undefined)
    and ``Z = ∫T(x)dx`` via the closed-form total-mass formulae in
    :mod:`mpt._mobius`.
    """
    # An ordered ([sym] = 0) density at r > 1 has no orbit, so the Möbius
    # collision inner product (which presumes symmetrisation) does not
    # apply. Compute it numerically via the direct double sum of Gaussian
    # overlaps over the C(K, r) ordered tuples, reusing the shared
    # per-attribute machinery on a single-attribute, single-event density.
    # r = 1 is exempt ([sym] vacuous; ordered and symmetric coincide) and
    # falls through to the closed-form path below.
    if (not bool(getattr(dens, "is_sym", True))) and int(dens.r) > 1:
        from ._tensor.build import build_exp_tens

        dens_p = dens.pruned()
        p2 = np.asarray(dens_p.p, dtype=np.float64).reshape(-1, 1)
        w2 = np.asarray(dens_p.w, dtype=np.float64).reshape(-1, 1)
        da_dens = build_exp_tens(
            [p2], [w2], [float(dens_p.sigma)], [int(dens_p.r)],
            [bool(dens_p.is_rel)], [bool(dens_p.is_per)],
            [float(dens_p.period)], [False], verbose=False,
        )
        I_a, Z_a = _renyi2_per_attr_numerical(da_dens, 0)
        ip_xx = float(I_a[0, 0])
        Z = float(Z_a[0])
        return _renyi2_finalise(ip_xx, Z, base)
    from ._mobius import total_mass_abs, total_mass_rel

    dens = dens.pruned()
    p, w = dens.p, dens.w
    sigma, r = dens.sigma, dens.r
    is_rel, is_per, period = dens.is_rel, dens.is_per, dens.period

    # r=1 rel is degenerate: the relative density lives on a 0-D space
    # (one position has no internal relative structure); H_2 is
    # undefined as a continuous quantity. Return 0 by convention,
    # matching the MA path's dim==0 short circuit.
    if r == 1 and is_rel:
        return 0.0

    if r == 1:
        # Direct r=1 abs path: T = Σ_i w_i G_σ(x - p_i), so
        #   <T,T> = σ√π · Σ_{i,j} w_i w_j exp(-(p_i-p_j)²/(4σ²))
        # (with wrapped differences in periodic mode).
        diffs = p[:, None] - p[None, :]
        if is_per:
            diffs = diffs - period * np.floor(diffs / period + 0.5)
        K = np.exp(-(diffs ** 2) / (4 * sigma ** 2))
        ip_xx = float(sigma * np.sqrt(np.pi) * (w[:, None] * w[None, :] * K).sum())
        Z = total_mass_abs(p, w, sigma, r)
        return _renyi2_finalise(ip_xx, Z, base)

    # r >= 2: Möbius machinery. Empirical sweeps across all 7 regimes
    # (precision_audit/ + sweep_self_ip.py) show the Möbius-method self-IP is
    # robust at every tested musical sigma; the per-orbit-class
    # cancellation ratio in abs mode dips to ~0.13 in the worst tested
    # case, well above the 1e-10 corruption threshold. We therefore
    # rely on a post-hoc finite/positive check rather than a ratio-
    # based fallback. The Bulger fallback explored earlier was
    # abandoned: the Möbius method and Bulger's method use different
    # normalisation conventions in rel mode, so the fallback gave a
    # different (also wrong) answer rather than recovering the correct
    # value.
    if is_rel:
        ip_xx = _orbit_inner_rel(
            p, w, p, w, sigma, r, is_per, period,
        )
    else:
        ip_xx = _orbit_inner_abs(
            p, w, p, w, sigma, r, is_per, period,
        )

    # Z via closed-form Möbius total mass.
    if is_rel:
        Z = total_mass_rel(p, w, sigma, r)
    else:
        Z = total_mass_abs(p, w, sigma, r)

    return _renyi2_finalise(ip_xx, Z, base)


def _renyi2_per_attr_numerical(dens, a):
    """Per-attribute (event, event) inner matrix and per-event total mass
    computed numerically via explicit tuple enumeration and the (block-
    diagonal) co-transposition metric. Handles both *nested* attributes
    and *ordered* (``[sym] = 0``) flat attributes at ``r > 1``.

    Returns ``(I_a, Z_a)`` where ``I_a[n, m] = integral k_a^n(x) k_a^m(x)``
    over event *n*'s and event *m*'s attribute-*a* kernels, and
    ``Z_a[n] = integral k_a^n`` is event *n*'s total mass. These compose
    with the flat-symmetric Möbius matrices in the MA Rényi-2
    factorisation ``integral p^2 = sum_{n,m} prod_a I_a[n,m]`` and
    ``Z = sum_n prod_a Z_a[n]``.

    The flat Möbius per-attribute matrix presumes a single *symmetric*
    ``r_a``-tuple over the slots and re-derives the full S_{r_a} orbit;
    that orbit is wrong for an ordered attribute (no symmetrisation) and,
    for a nested attribute (``r_a = prod(r_levels)``), both wrong and
    infeasible. The numerical reading here builds the attribute's density,
    whose tuples and metric are correct in either case, and forms the
    overlap integrals in closed form: for two kernels of common metric
    ``M`` and width ``sigma`` the Gaussian overlap is
    ``(pi sigma^2)^{d/2} / sqrt(det M) * exp(-Q_M(c_t - c_s) / (4 sigma^2))``
    and the single-kernel mass is ``(2 pi sigma^2)^{d/2} / sqrt(det M)``.
    """
    from ._tensor.build import build_exp_tens
    from ._tensor.dispatch import (
        _compute_Q_inner_blocks, _compute_Q, _inner_r_vec,
    )

    nested = getattr(dens, "nested", None)
    spec = nested[a] if nested is not None else None
    sigma = float(dens.sigma[a])
    is_per = bool(dens.is_per[a])
    period = float(dens.period[a])
    if spec is not None:
        # Nested attribute: rebuild from its resolved spec.
        da = build_exp_tens(
            [dens.p_attr[a]], [dens.w[a]], specs=[spec],
            sigma=[sigma], is_per=[is_per], period=[period], verbose=False,
        )
    else:
        # Flat ordered attribute: rebuild from its flat parameters with
        # is_sym=False, so the materialised tuples are the C(K, r_a)
        # ordered sub-tuples (one kernel each, no orbit).
        r_a0 = int(dens.r[a])
        is_rel0 = bool(dens.is_rel[a])
        da = build_exp_tens(
            [dens.p_attr[a]], [dens.w[a]],
            [sigma], [r_a0], [is_rel0], [is_per], [period], [False],
            verbose=False,
        )
    centres = da.centres[0]            # (d_a, n_j) reduced centres
    w_j = da.w_j                       # (n_j,)
    event_of_j = da.event_of_j         # (n_j,) -> event index 0..N-1
    d_a = centres.shape[0]
    n_j = w_j.shape[0]
    N = int(dens.n)

    block_size = int(_inner_r_vec(da)[0])   # s_u (inner/intermediate) or 0
    is_rel = bool(da.is_rel[0])
    r_a = int(da.r[0])
    if block_size >= 2:
        det_m = (1.0 / block_size) ** (r_a // block_size)
    elif is_rel and r_a >= 2:
        det_m = 1.0 / r_a
    else:
        det_m = 1.0
    vol = (2 * np.pi * sigma ** 2) ** (d_a / 2) / np.sqrt(det_m)   # mass
    pref = (np.pi * sigma ** 2) ** (d_a / 2) / np.sqrt(det_m)      # overlap

    I_a = np.zeros((N, N), dtype=np.float64)
    Z_a = np.zeros(N, dtype=np.float64)
    if n_j > 0:
        # Pairwise (block-)metric quadratic form on the reduced centres.
        D = centres[:, :, None] - centres[:, None, :]   # (d_a, n_j, n_j)
        if block_size >= 2:
            Q = _compute_Q_inner_blocks(
                D, block_size, is_per, period, reduced=True)
        else:
            if is_per and not is_rel:
                D = D - period * np.floor(D / period + 0.5)
            Q = _compute_Q(D, r_a, is_rel, is_per, period, reduced=is_rel)
        overlap = pref * np.exp(-Q / (4 * sigma ** 2))   # (n_j, n_j)
        wo = (w_j[:, None] * w_j[None, :]) * overlap
        # Aggregate tuples into their events (G is the N x n_j incidence).
        G = np.zeros((N, n_j), dtype=np.float64)
        G[event_of_j, np.arange(n_j)] = 1.0
        I_a = G @ wo @ G.T
        Z_a = vol * (G @ w_j)
    return I_a, Z_a


def _renyi2_exp_tens_ma(dens_or_windowed, *, base: float) -> float:
    """Analytical Rényi-2 entropy of an MA expectation tensor.

    Uses the per-attribute Möbius IP factorisation
    ``<T,T> = Σ_{n,m} Π_a I_a[n,m]``, with the per-attribute matrix
    coming from the same machinery the cosine path uses, and
    ``Z = Σ_n Π_a Z_a^(n)`` where each ``Z_a^(n)`` is the closed-form
    SA total mass evaluated on event ``n``'s attribute-``a`` slot
    pitches and weights.

    Windowed densities are not yet supported on this path; raises
    NotImplementedError.
    """
    from ._mobius import total_mass_abs, total_mass_rel

    if isinstance(dens_or_windowed, WindowedMaetDensity):
        raise NotImplementedError(
            "method='renyi2' is not yet implemented for "
            "WindowedMaetDensity. Use method='shannon' for windowed "
            "MA densities, or compute on the underlying MaetDensity."
        )
    dens = dens_or_windowed.pruned()
    A = dens.n_attrs
    N = dens.n
    if A == 0:
        return 0.0
    if N == 0:
        # Every event pruned away: a zero-mass density (e.g. a windowed
        # sweep centre with no event in support). Collision entropy is
        # undefined; return NaN rather than 0, matching the SA path and
        # the value a windowed sweep wants at out-of-support centres.
        return float("nan")

    # Per-attribute inner matrices compose as
    # <T,T> = sum_{n,m} prod_a I_a[n,m] and Z = sum_n prod_a Z_a^(n).
    # Symmetric flat attributes take the Möbius per-attribute matrix and
    # closed-form total mass (the fast path; orbit-collapse assumes
    # symmetrisation). Nested attributes, and ordered ([sym] = 0) flat
    # attributes at r > 1, take the numerical inner matrix
    # (:func:`_renyi2_per_attr_numerical`): an ordered attribute has no
    # orbit, so its tuples are summed directly. r = 1 flat attributes are
    # symmetric-equivalent ([sym] vacuous) and stay on the Möbius path.
    #
    # The per-(n, m) cancellation ratio aggregated across attributes was
    # empirically shown to fire spuriously in 100% of typical musical
    # regimes for self-IPs (sweep_self_ip.py): off-diagonal entries can
    # have low ratios while the diagonal entries (which dominate the sum)
    # are clean, so the sum Σ P_xx[n,m] is correct even when some entries
    # are noisy. We therefore rely solely on a post-hoc finite/positive
    # check. The Bulger fallback was abandoned for the same convention-
    # mismatch reason as in the SA path.
    is_sym = np.asarray(getattr(dens, "is_sym", np.ones(A, dtype=bool)))
    r_vec = np.asarray(dens.r)
    nested = getattr(dens, "nested", [None] * A)
    P_xx = np.ones((N, N), dtype=np.float64)
    Z_per_event_attr = np.empty((N, A), dtype=np.float64)
    for a in range(A):
        ordered_flat = (nested[a] is None) and (not bool(is_sym[a])) \
            and (int(r_vec[a]) > 1)
        if nested[a] is not None or ordered_flat:
            I_xx, Z_a = _renyi2_per_attr_numerical(dens, a)
        else:
            r_a = int(dens.r[a])
            sigma = float(dens.sigma[a])
            is_rel = bool(dens.is_rel[a])
            is_per = bool(dens.is_per[a])
            period = float(dens.period[a])
            Pa = dens.p_attr[a]
            Wa = dens.w[a]
            I_xx = _ma_per_attr_inner_matrix(
                Pa, Wa, Pa, Wa, sigma, r_a, is_rel, is_per, period,
            )
            Z_a = np.empty(N, dtype=np.float64)
            for n in range(N):
                if is_rel:
                    Z_a[n] = total_mass_rel(Pa[:, n], Wa[:, n], sigma, r_a)
                else:
                    Z_a[n] = total_mass_abs(Pa[:, n], Wa[:, n], sigma, r_a)
        P_xx *= I_xx
        Z_per_event_attr[:, a] = Z_a
    ip_xx = float(P_xx.sum())

    # ---- Z = Σ_n Π_a Z_a^(n) ----
    Z = float(np.prod(Z_per_event_attr, axis=1).sum())

    return _renyi2_finalise(ip_xx, Z, base)


def _looks_like_ma_p(p) -> bool:
    """Return True if p is a list/tuple of attribute matrices, i.e. the
    MA raw-args input form (as opposed to a 1-D SA pitch vector)."""
    if isinstance(p, np.ndarray):
        return False  # an ndarray is always SA input
    if not isinstance(p, (list, tuple)):
        return False
    if len(p) == 0:
        return False
    first = p[0]
    # SA: p is a list/tuple of numbers (e.g., [0, 4, 7]).
    if np.isscalar(first):
        return False
    # MA: first is an array-like (matrix) with rows (slots) and cols (events).
    return True


# -------------------------------------------------------------------
#  _entropy_exp_tens_sa  (single-attribute legacy path)
# -------------------------------------------------------------------


def _entropy_exp_tens_sa(
    p_or_dens, w, sigma, r, is_rel, is_per, period, is_sym=None,
    *,
    spectrum, normalize, base,
    n_points_per_dim, x_min, x_max,
    truncation_sigmas=None, kernel_precision=None,
) -> float:
    """Single-attribute Shannon entropy."""
    if isinstance(p_or_dens, ExpTensDensity):
        T = p_or_dens
        is_per = T.is_per
        period = T.period
    else:
        p = np.asarray(p_or_dens, dtype=np.float64).ravel()
        if (sigma is None or r is None or is_rel is None
                or is_per is None or period is None):
            raise ValueError(
                "When p is not a precomputed density, the structural "
                "arguments (sigma, r, is_rel, is_per, period) are required. "
                "(w may be None for uniform weights.)"
            )
        if w is None:
            w = np.ones_like(p)
        if spectrum is not None:
            p, w = add_spectra(p, w, *spectrum)
        T = build_exp_tens(
            p, w, sigma, r, is_rel, is_per, period,
            True if is_sym is None else is_sym, verbose=False,
        )

    # Construct query points. For dim == 1 the grid is a single 1-D
    # linspace; for dim > 1 it is a Cartesian product, mirroring the
    # MA path.
    dim = int(T.dim)
    if is_per:
        ax = np.linspace(0, period, n_points_per_dim + 1)[:-1]
    else:
        x_min_s = float(np.asarray(x_min).item()) if np.ndim(x_min) == 0 else float("nan")
        x_max_s = float(np.asarray(x_max).item()) if np.ndim(x_max) == 0 else float("nan")
        if np.isnan(x_min_s) or np.isnan(x_max_s):
            raise ValueError("x_min and x_max must be specified when is_per is False.")
        if x_min_s >= x_max_s:
            raise ValueError("x_min must be less than x_max.")
        ax = np.linspace(x_min_s, x_max_s, n_points_per_dim)

    if dim == 1:
        x = ax
    else:
        mesh = np.meshgrid(*([ax] * dim), indexing="ij")
        x = np.stack([m.ravel() for m in mesh], axis=0)  # (dim, total_points)

    # Absolute-mode densities use bin-integration (analytic cell mass
    # via per-axis erf differences); relative-mode falls back to
    # point-evaluation pending v2.3 covariance work.
    if not bool(T.is_rel):
        ts = (6.0 if truncation_sigmas is None
              else float(truncation_sigmas))
        t = _cell_masses_sa_absolute(T, ax, truncation_sigmas=ts)
    else:
        t = eval_exp_tens(
            T, x, verbose=False,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
        )

    total = np.sum(t)
    if total == 0:
        return 0.0

    q = t / total
    N = q.size
    q = q[q > 0]

    H = float(-np.sum(q * np.log(q) / np.log(base)))

    if normalize:
        H /= np.log(N) / np.log(base)

    return H


# -------------------------------------------------------------------
#  _entropy_exp_tens_ma  (multi-attribute path)
# -------------------------------------------------------------------


def _entropy_exp_tens_ma(
    dens,
    *,
    normalize: bool,
    base: float,
    n_points_per_dim: int,
    x_min, x_max,
    grid_limit: int,
    truncation_sigmas=None, kernel_precision=None,
) -> float:
    """Multi-attribute Shannon entropy.

    Builds a Cartesian-product grid with one 1-D linspace per effective
    dimension of the density's domain (one per non-``isRel`` tuple slot
    for each attribute), evaluates the density at every grid point,
    normalises to a pmf, and returns Shannon entropy.

    Accepts either a :class:`MaetDensity` or a
    :class:`WindowedMaetDensity`. Structural fields (dim, dim_per_attr,
    etc.) are read from the underlying density; evaluation
    itself calls :func:`eval_exp_tens` on the input object, so window
    application (if present) is handled automatically.
    """
    # Structural fields — same on windowed or unwindowed objects.
    if isinstance(dens, WindowedMaetDensity):
        base_dens = dens.dens
    else:
        base_dens = dens
    dim      = int(base_dens.dim)
    dim_per  = base_dens.dim_per_attr
    A        = base_dens.n_attrs
    is_per_g = base_dens.is_per
    period_g = base_dens.period

    if dim == 0:
        # Degenerate: no effective axes (e.g. every attribute is isRel
        # with r=1). Density is a constant; entropy is 0.
        return 0.0

    # --- Resolve x_min/x_max to per-attribute arrays ---
    x_min_g = _broadcast_bounds(x_min, A, "x_min")
    x_max_g = _broadcast_bounds(x_max, A, "x_max")

    # --- Check non-periodic attributes have valid bounds ---
    needs_bounds = np.flatnonzero(~is_per_g)
    for g in needs_bounds:
        if np.isnan(x_min_g[g]) or np.isnan(x_max_g[g]):
            raise ValueError(
                f"x_min and x_max must be specified for non-periodic "
                f"attribute {int(g)}."
            )
        if x_min_g[g] >= x_max_g[g]:
            raise ValueError(
                f"x_min must be less than x_max (attribute {int(g)})."
            )

    # --- Grid-size guard ---
    total_points = int(n_points_per_dim) ** dim
    if total_points > grid_limit:
        # Suggest the largest n_points_per_dim that would fit.
        suggested = int(np.floor(grid_limit ** (1.0 / dim)))
        raise ValueError(
            f"Grid size {n_points_per_dim}**{dim} = {total_points} "
            f"exceeds grid_limit = {grid_limit}. Reduce n_points_per_dim "
            f"to {suggested} or lower, or raise grid_limit."
        )

    # --- Build one 1-D linspace per effective dimension ---
    # Each effective dimension belongs to an attribute, which carries
    # its own domain.
    axes = []
    for a in range(A):
        da = int(dim_per[a])
        if is_per_g[a]:
            P = float(period_g[a])
            ax = np.linspace(0.0, P, int(n_points_per_dim) + 1)[:-1]
        else:
            ax = np.linspace(
                float(x_min_g[a]), float(x_max_g[a]), int(n_points_per_dim)
            )
        for _ in range(da):
            axes.append(ax)

    # --- Cartesian product as (dim, total_points) query matrix ---
    # Use np.meshgrid with 'ij' indexing so the flatten order is
    # consistent (first axis varies slowest).
    mesh = np.meshgrid(*axes, indexing="ij")
    X = np.stack([m.ravel() for m in mesh], axis=0)  # (dim, total_points)

    # --- Evaluate density on the grid ---
    # For absolute-mode unwindowed densities (is_rel=False everywhere)
    # the categorical pmf is the genuine bin masses (int_{cell} f dx),
    # obtained analytically via per-axis erf differences. For
    # relative-mode densities the bin integral is a multivariate-
    # normal box probability (off-diagonal covariance in the effective
    # coordinates); pending the v2.3 covariance machinery we fall back
    # to point-evaluation, which agrees with bin-integration to ~1e-4
    # on the fine grids relative-mode use-cases require. Windowed
    # densities also use point-evaluation here -- windowed
    # cell-integration is a separate problem not yet addressed.
    is_windowed = isinstance(dens, WindowedMaetDensity)
    if not is_windowed and not bool(np.any(np.asarray(base_dens.is_rel))):
        ts = (6.0 if truncation_sigmas is None
              else float(truncation_sigmas))
        t = _cell_masses_ma_absolute(base_dens, axes, truncation_sigmas=ts)
    else:
        t = eval_exp_tens(
            dens, X, verbose=False,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
        )

    # --- Shannon entropy ---
    total = float(np.sum(t))
    if total == 0.0:
        return 0.0

    q = t / total
    N = int(q.size)
    q = q[q > 0]
    H = float(-np.sum(q * np.log(q) / np.log(base)))

    if normalize:
        H /= np.log(N) / np.log(base)

    return H


def _broadcast_bounds(v, A, name):
    """Coerce x_min or x_max input to a length-A float array.

    Accepts NaN, a scalar (broadcast), or a length-A array. Entries for
    periodic attributes are not validated here (they're never used).
    """
    arr = np.asarray(v, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(A, float(arr), dtype=np.float64)
    if arr.ndim == 1 and arr.size == A:
        return arr.astype(np.float64, copy=False)
    raise ValueError(
        f"{name} must be a scalar or a length-{A} vector (one entry per "
        f"attribute); got shape {arr.shape}."
    )


# ===================================================================
#  n_tuple_entropy
# ===================================================================


def n_tuple_entropy(
    p,
    period: float,
    n: int = 1,
    *,
    sigma: float = 0.0,
    sigma_space: str = "position",
    method: str = "normalized",
    base: float = 2.0,
    n_points_per_dim: int | None = None,
) -> tuple[float, np.ndarray]:
    """Entropy of n-tuples of consecutive step sizes.

    Accepts two input forms, dispatched on ``p``'s shape:

    - 1-D ``p``: single multiset, returns ``(H, tuples)``.
    - 2-D ``P`` (shape ``(M, K)``): batched, returns
      ``(H_array, tuples_list)`` where ``H_array`` is length-``M``
      and ``tuples_list`` is a length-``M`` list of per-row tuple
      matrices. Per-row dedup over permutation + period symmetries
      (not transposition — would require per-row tuple post-transform
      that loses input fidelity).

    Convenience wrapper around the bind-and-compute pipeline of
    :func:`bind_events`, :func:`build_exp_tens`, and
    :func:`entropy_exp_tens`. With default arguments — ``sigma = 0``,
    ``method = 'normalized'``, and ``n_points_per_dim = None`` (which
    selects the integer-step grid ``period``) — this exactly replicates
    the discrete *n*-tuple entropy of Milne & Dean (2016).

    Parameters
    ----------
    p : array-like
        Pitch or position values. Non-negative; values less than
        *period*. Must be integer when ``sigma == 0``; may be float
        when ``sigma > 0``. Duplicates not allowed.
    period : float
        Size of the equal division. Must be integer when
        ``sigma == 0``.
    n : int
        Tuple size (default 1). Must satisfy ``1 <= n <= K - 1``.
    sigma : float
        Smoothing bandwidth (non-negative; default 0). In the same
        units as *p* and *period*.
    sigma_space : {'position', 'interval'}
        How sigma is interpreted (default 'position'). 'position'
        treats sigma as positional uncertainty on each ``p_k``;
        'interval' treats sigma as independent uncertainty per
        derived step. See "Sigma semantics" below.
    method : {'normalized', 'shannon', 'differential', 'renyi2'}
        Entropy variant (default ``'normalized'``). See
        :func:`entropy_exp_tens` for the four-method API. ``'normalised'``
        is accepted as an alias. The continuous methods
        (``'differential'``, ``'renyi2'``) require ``sigma > 0``.
    base : float
        Logarithm base (default 2). Cancels for ``method='normalized'``.
    n_points_per_dim : int or None
        Grid resolution per dimension (used by ``'normalized'`` and
        ``'shannon'``; ignored by ``'differential'`` and ``'renyi2'``).
        ``None`` (default) selects ``period``, which (with integer
        centres and a periodic kernel) gives the Milne & Dean (2016)
        mass-conserving Gaussian-confusion grid.

    Sigma semantics
    ---------------
    Under the toolbox convention, sigma applies to the input
    quantity. For *n_tuple_entropy* the input is positions *p*, so
    ``sigma_space = 'position'`` is the default and matches behavior
    elsewhere in the toolbox (sameness, coherence, etc.).

    For ``sigma_space = 'position'``:

      - Each ``p_k`` is treated as ``N(p_k, sigma**2)``.
      - Derived steps ``d_k = p_{k+1} - p_k`` then have variance
        ``2 * sigma**2`` per step, with anti-correlation ``-sigma**2``
        between adjacent steps (they share an endpoint with opposite
        signs), i.e. covariance ``sigma**2 * tridiag(2, -1)`` over the
        ``n`` steps of a tuple.
      - This full covariance is captured exactly, at every ``n``,
        without an off-diagonal kernel. Rather than placing a kernel
        on the steps, the implementation binds ``n + 1`` consecutive
        pitches and takes the window relative (``[rel] = 1`` at the
        outer level). Projecting the isotropic positional jitter
        ``sigma**2 I`` onto the within-window difference space
        reproduces ``sigma**2 * tridiag(2, -1)`` in step coordinates
        from isotropic kernels alone.
      - At ``n == 1`` there is no neighbour to correlate with, so this
        reduces to a single step of variance ``2 * sigma**2``.
      - The relative density lives on the within-window difference
        space (an orthonormal basis of the quotient), so for
        ``sigma > 0`` the continuous (``differential``, ``renyi2``)
        and grid (``shannon``, ``normalized``) entropies are reported
        in those coordinates, not in step coordinates; they differ
        from ``sigma_space = 'interval'`` by both the sigma semantics
        and this coordinate convention. At ``sigma == 0`` the
        coordinate convention is immaterial (see below).

    For ``sigma_space = 'interval'``:

      - Each step ``d_k`` is treated as ``N(d_k, sigma**2)``
        independently.
      - This is the legacy "step-size" interpretation: each step is
        the primitive, with its own independent uncertainty.
      - Use this if your psychological model treats per-step
        uncertainty as the primitive (rather than positional
        uncertainty).

    At ``sigma == 0`` the two flags coincide (no smoothing).

    Returns
    -------
    H : float
        Shannon entropy of the n-tuple distribution.
    tuples : np.ndarray
        ``(K, n)`` matrix of n-tuples.

    See Also
    --------
    bind_events
    entropy_exp_tens
    build_exp_tens
    difference_events
    sameness
    coherence

    References
    ----------
    Milne, A. J. & Dean, R. T. (2016). Computational creation and
    morphing of multilevel rhythms by control of evenness. *Computer
    Music Journal*, 40(1), 35–53.

    Milne, A. J. (2024). Commentary on Buechele, Cooke, &
    Berezovsky (2024): Entropic models of scales and some
    extensions. *Empirical Musicology Review*, 19(2), 143–152.
    """
    if sigma_space not in ("position", "interval"):
        raise ValueError(
            f"sigma_space must be 'position' or 'interval' "
            f"(got {sigma_space!r})."
        )

    # Canonicalize method (accepts British 'normalised').
    method = _canonicalize_method(method)

    # Continuous-form methods diverge at sigma=0. Reject explicitly
    # rather than letting the entropy_exp_tens guard surface a less-
    # specific error after the kludge that nudges sigma off zero.
    if sigma == 0.0 and method in ("differential", "renyi2"):
        raise ValueError(
            f"n_tuple_entropy: method={method!r} requires sigma > 0 "
            f"(the continuous form diverges at sigma=0). For categorical "
            f"sigma=0 n-tuple entropy use method='shannon' or "
            f"method='normalized' (the default)."
        )

    p_arr = np.asarray(p, dtype=np.float64)
    if p_arr.ndim == 2:
        return _n_tuple_entropy_batched(
            p_arr, period, n,
            sigma=sigma, sigma_space=sigma_space,
            method=method, base=base,
            n_points_per_dim=n_points_per_dim,
        )

    p = p_arr.ravel()
    period = float(period)
    n = int(n)
    sigma = float(sigma)

    p = np.sort(p % period)
    K = len(p)

    if len(np.unique(p)) != K:
        raise ValueError("p must not contain duplicate values (mod period).")
    if K < 2:
        raise ValueError(f"At least 2 events required (got {K}).")
    if n > K - 1:
        raise ValueError(f"n must not exceed K - 1 = {K - 1} (got n = {n}).")

    if sigma == 0.0:
        if not np.all(np.abs(p - np.round(p)) == 0):
            raise ValueError(
                "For sigma == 0, p must contain integers. "
                "Use sigma > 0 for non-integer positions."
            )
        if abs(period - round(period)) != 0:
            raise ValueError(
                f"For sigma == 0, period must be integer (got {period})."
            )

    if n_points_per_dim is None:
        n_grid = int(round(period))
    else:
        n_grid = int(n_points_per_dim)
        if n_grid < 1:
            raise ValueError(
                f"n_points_per_dim must be a positive integer "
                f"(got {n_grid})."
            )

    # --- Step-tuples: cyclic first differences, then bind n consecutive
    #     steps. These are the returned n-tuples for both modes, and the
    #     density for sigma_space='interval'. ---
    # difference_events with circular=True wraps at the sequence boundary
    # (output position 0 holds p(0) - p(N-1)); the downstream periodic
    # kernel handles mod-period wrapping at evaluation time, so no explicit
    # mod is needed here.
    p_row = p.astype(np.float64).reshape(1, -1)
    p_diff_list, _, _ = difference_events(
        [p_row], None, 1, circular=True,
    )
    diffs_row = p_diff_list[0]
    p_step, w_step, step_specs = bind_events(
        [diffs_row], None, n, circular=True,
    )
    tuples_out = p_step[0].T

    sigma_use = sigma if sigma > 0 else 1e-12

    # --- Build the MAET per the sigma_space flag ---
    if sigma_space == "interval":
        # sigma is per-step uncertainty: each bound step is an independent
        # N(d_k, sigma**2). The n bound steps form one absolute ordered
        # attribute; sigma/is_per/period pass straight through.
        T = build_exp_tens(
            p_step, w_step,
            specs=step_specs, sigma=[sigma_use], is_per=[True],
            period=[period], verbose=False,
        )
    else:  # sigma_space == "position"
        # sigma is positional uncertainty on each p_k. Bind n+1 consecutive
        # pitches and take the window relative ([rel]=1 at the outer level):
        # projecting the isotropic positional jitter sigma**2 I onto the
        # within-window difference space gives each step variance 2*sigma**2
        # with -sigma**2 anti-correlation between adjacent steps -- the exact
        # position model. The relative projection supplies this correlated
        # covariance from isotropic kernels, so no off-diagonal kernel
        # covariance is needed. Exact at every n; at sigma = 0 it reduces to
        # the integer step histogram, matching 'interval' and Milne & Dean
        # (2016).
        p_win, w_win, win_specs = bind_events(
            [p_row], None, n + 1, circular=True,
        )
        # Two nesting levels: inner singleton pitch, outer window of n+1
        # pitches. Take the outer window relative, inner absolute. The inner
        # singleton's flags are inert (Section "Sigma semantics"), so the
        # level collapses to a flat ordered relative (n+1)-tuple whose
        # within-tuple differences are the n consecutive steps.
        win_specs[0]["rel"] = [0, 1]
        T = build_exp_tens(
            p_win, w_win,
            specs=win_specs, sigma=[sigma_use], is_per=[True],
            period=[period], verbose=False,
        )

    # --- Entropy on the chosen grid / via the chosen method ---
    # Grid-based methods ('shannon', 'normalized') use the pinned period
    # grid n_grid. 'differential' and 'renyi2' bypass the grid (adaptive
    # and analytical respectively).
    if method == "shannon":
        H = entropy_exp_tens(
            T, method="shannon",
            base=base, n_points_per_dim=n_grid,
        )
    elif method == "normalized":
        H = entropy_exp_tens(
            T, method="normalized",
            base=base, n_points_per_dim=n_grid,
        )
    elif method == "differential":
        H = entropy_exp_tens(
            T, method="differential", base=base,
        )
    else:  # method == "renyi2"
        H = entropy_exp_tens(
            T, method="renyi2", base=base,
        )

    return H, tuples_out


def _n_tuple_entropy_batched(
    P, period, n,
    *,
    sigma, sigma_space, method, base, n_points_per_dim,
):
    """Batched dispatch for ``n_tuple_entropy``.

    Returns ``(H_array, tuples_list)``. NaN-padded rows are dropped
    per row; rows with no valid pitches give NaN in ``H_array`` and
    an empty array in ``tuples_list``. Per-row dedup via sorted-
    modular canonical key.
    """
    M, K = P.shape
    H_out = np.full(M, np.nan)
    tuples_list: list = [np.array([]) for _ in range(M)]
    cache: dict = {}

    for i in range(M):
        p_row = P[i]
        mask = ~np.isnan(p_row)
        p_valid = p_row[mask]
        if len(p_valid) == 0:
            continue
        p_canon = np.sort(np.mod(p_valid, float(period)))
        key = tuple(np.round(p_canon, 12).tolist())

        if key in cache:
            H_out[i], tuples_list[i] = cache[key]
            continue

        H_i, t_i = n_tuple_entropy(
            p_valid, period, n,
            sigma=sigma, sigma_space=sigma_space,
            method=method, base=base,
            n_points_per_dim=n_points_per_dim,
        )
        H_out[i] = H_i
        tuples_list[i] = t_i
        cache[key] = (H_i, t_i)

    return H_out, tuples_list
