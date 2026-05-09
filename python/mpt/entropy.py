"""Entropy measures for pitch and rhythm structures."""

from __future__ import annotations

import warnings

import numpy as np

from .spectra import add_spectra
from .tensor import (
    ExpTensDensity,
    MaetDensity,
    WindowedMaetDensity,
    bind_events,
    build_exp_tens,
    eval_exp_tens,
)


# Default grid-size ceiling for the Cartesian-product grid. If
# n_points_per_dim**dim exceeds this, entropy_exp_tens raises a
# clear error suggesting a lower n_points_per_dim.
_DEFAULT_GRID_LIMIT = int(1e8)


# ===================================================================
#  entropy_exp_tens
# ===================================================================


def entropy_exp_tens(
    p_or_dens,
    *args,
    spectrum: list | None = None,
    normalize: bool = True,
    base: float = 2.0,
    n_points_per_dim: int = 1200,
    x_min=float("nan"),
    x_max=float("nan"),
    grid_limit: int = _DEFAULT_GRID_LIMIT,
) -> float:
    """Shannon entropy of an expectation tensor density.

    Dispatches on the type of the first argument:

      - :class:`ExpTensDensity`, or SA raw args (*p* a 1-D array) ->
        single-attribute path. Signature::

            entropy_exp_tens(p, w, sigma, r, is_rel, is_per, period, ...)
            entropy_exp_tens(dens_sa, ...)

      - :class:`MaetDensity`, or MA raw args (*p* a list/tuple of
        attribute matrices) -> multi-attribute path. Signature::

            entropy_exp_tens(p_attr, w, sigma_vec, r_vec, groups,
                             is_rel_vec, is_per_vec, period_vec, ...)
            entropy_exp_tens(dens_ma, ...)

    Parameters
    ----------
    p_or_dens : array-like, list of matrices, ExpTensDensity, or MaetDensity
        Either a pitch/position input or a precomputed density object.
    *args : tuple
        Raw-args tail (SA: 6 further args; MA: 7 further args). Ignored
        for precomputed densities.
    spectrum : list or None
        Arguments for :func:`~mpt.spectra.add_spectra`. SA only; for MA,
        apply spectral enrichment to the pitch attribute upstream.
    normalize : bool
        If True (default), divide by log(N) for a [0, 1] value.
    base : float
        Logarithm base (default 2).
    n_points_per_dim : int
        Grid resolution per effective dimension (default 1200).
    x_min, x_max : float or length-G array
        Domain bounds for non-periodic axes. SA: scalars. MA: scalar
        (broadcast to all non-periodic groups) or length-G vector
        (entries for periodic groups ignored). Required when any axis
        is non-periodic.
    grid_limit : int
        Hard cap on the total grid size (``n_points_per_dim ** dim``)
        before allocation. Default 1e8. Raises ValueError if exceeded,
        suggesting a lower *n_points_per_dim*.

    Returns
    -------
    float
        Shannon entropy. In [0, 1] when *normalize* is True.
    """
    # --- Dispatch on precomputed densities first ---
    if isinstance(p_or_dens, WindowedMaetDensity):
        if len(args) > 0:
            raise TypeError(
                "Precomputed WindowedMaetDensity takes no further positional args."
            )
        return _entropy_exp_tens_ma(
            p_or_dens,
            normalize=normalize, base=base,
            n_points_per_dim=n_points_per_dim,
            x_min=x_min, x_max=x_max, grid_limit=grid_limit,
        )
    if isinstance(p_or_dens, MaetDensity):
        if len(args) > 0:
            raise TypeError(
                "Precomputed MaetDensity takes no further positional args."
            )
        return _entropy_exp_tens_ma(
            p_or_dens,
            normalize=normalize, base=base,
            n_points_per_dim=n_points_per_dim,
            x_min=x_min, x_max=x_max, grid_limit=grid_limit,
        )
    if isinstance(p_or_dens, ExpTensDensity):
        if len(args) > 0:
            raise TypeError(
                "Precomputed ExpTensDensity takes no further positional args."
            )
        return _entropy_exp_tens_sa(
            p_or_dens, None, None, None, None, None, None,
            spectrum=None, normalize=normalize, base=base,
            n_points_per_dim=n_points_per_dim,
            x_min=x_min, x_max=x_max,
        )

    # --- Raw args: dispatch on type of p ---
    if _looks_like_ma_p(p_or_dens):
        if len(args) != 7:
            raise ValueError(
                f"Multi-attribute raw call expects 8 positional arguments "
                f"(p_attr, w, sigma_vec, r_vec, groups, is_rel_vec, "
                f"is_per_vec, period_vec); got {1 + len(args)}."
            )
        w, sigma_vec, r_vec, groups, is_rel_vec, is_per_vec, period_vec = args
        dens = build_exp_tens(
            p_or_dens, w, sigma_vec, r_vec, groups,
            is_rel_vec, is_per_vec, period_vec,
            verbose=False,
        )
        return _entropy_exp_tens_ma(
            dens,
            normalize=normalize, base=base,
            n_points_per_dim=n_points_per_dim,
            x_min=x_min, x_max=x_max, grid_limit=grid_limit,
        )
    # SA raw args.
    if len(args) != 6:
        raise ValueError(
            f"Single-attribute raw call expects 7 positional arguments "
            f"(p, w, sigma, r, is_rel, is_per, period); got {1 + len(args)}."
        )
    w, sigma, r, is_rel, is_per, period = args
    return _entropy_exp_tens_sa(
        p_or_dens, w, sigma, r, is_rel, is_per, period,
        spectrum=spectrum, normalize=normalize, base=base,
        n_points_per_dim=n_points_per_dim,
        x_min=x_min, x_max=x_max,
    )


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
    p_or_dens, w, sigma, r, is_rel, is_per, period,
    *,
    spectrum, normalize, base,
    n_points_per_dim, x_min, x_max,
) -> float:
    """Single-attribute Shannon entropy (v2.0.0 body)."""
    if isinstance(p_or_dens, ExpTensDensity):
        T = p_or_dens
        is_per = T.is_per
        period = T.period
    else:
        p = np.asarray(p_or_dens, dtype=np.float64).ravel()
        if w is None or sigma is None or r is None or is_rel is None or is_per is None or period is None:
            raise ValueError(
                "When p is not a precomputed density, all positional "
                "arguments (p, w, sigma, r, is_rel, is_per, period) are required."
            )
        if spectrum is not None:
            p, w = add_spectra(p, w, *spectrum)
        T = build_exp_tens(p, w, sigma, r, is_rel, is_per, period, verbose=False)

    # Construct query points
    if is_per:
        x = np.linspace(0, period, n_points_per_dim + 1)[:-1]
    else:
        x_min_s = float(np.asarray(x_min).item()) if np.ndim(x_min) == 0 else float("nan")
        x_max_s = float(np.asarray(x_max).item()) if np.ndim(x_max) == 0 else float("nan")
        if np.isnan(x_min_s) or np.isnan(x_max_s):
            raise ValueError("x_min and x_max must be specified when is_per is False.")
        if x_min_s >= x_max_s:
            raise ValueError("x_min must be less than x_max.")
        x = np.linspace(x_min_s, x_max_s, n_points_per_dim)

    t = eval_exp_tens(T, x, verbose=False)

    total = np.sum(t)
    if total == 0:
        return 0.0

    q = t / total
    N = len(q)
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
) -> float:
    """Multi-attribute Shannon entropy.

    Builds a Cartesian-product grid with one 1-D linspace per effective
    dimension of the density's domain (one per non-``isRel`` tuple slot
    for each attribute), evaluates the density at every grid point,
    normalises to a pmf, and returns Shannon entropy.

    Accepts either a :class:`MaetDensity` or a
    :class:`WindowedMaetDensity`. Structural fields (dim, dim_per_attr,
    groups, etc.) are read from the underlying density; evaluation
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
    G        = base_dens.n_groups
    group_of = base_dens.group_of_attr
    is_per_g = base_dens.is_per
    period_g = base_dens.period

    if dim == 0:
        # Degenerate: no effective axes (e.g. every attribute is isRel
        # with r=1). Density is a constant; entropy is 0.
        return 0.0

    # --- Resolve x_min/x_max to per-group arrays ---
    x_min_g = _broadcast_bounds(x_min, G, "x_min")
    x_max_g = _broadcast_bounds(x_max, G, "x_max")

    # --- Check non-periodic groups have valid bounds ---
    needs_bounds = np.flatnonzero(~is_per_g)
    for g in needs_bounds:
        if np.isnan(x_min_g[g]) or np.isnan(x_max_g[g]):
            raise ValueError(
                f"x_min and x_max must be specified for non-periodic "
                f"group {int(g)}."
            )
        if x_min_g[g] >= x_max_g[g]:
            raise ValueError(
                f"x_min must be less than x_max (group {int(g)})."
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
    # Each effective dimension belongs to an attribute, which belongs
    # to a group. Each 1-D linspace uses that group's domain.
    axes = []
    for a in range(A):
        da = int(dim_per[a])
        g = int(group_of[a])
        if is_per_g[g]:
            P = float(period_g[g])
            ax = np.linspace(0.0, P, int(n_points_per_dim) + 1)[:-1]
        else:
            ax = np.linspace(
                float(x_min_g[g]), float(x_max_g[g]), int(n_points_per_dim)
            )
        for _ in range(da):
            axes.append(ax)

    # --- Cartesian product as (dim, total_points) query matrix ---
    # Use np.meshgrid with 'ij' indexing so the flatten order is
    # consistent (first axis varies slowest).
    mesh = np.meshgrid(*axes, indexing="ij")
    X = np.stack([m.ravel() for m in mesh], axis=0)  # (dim, total_points)

    # --- Evaluate density ---
    t = eval_exp_tens(dens, X, verbose=False)

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


def _broadcast_bounds(v, G, name):
    """Coerce x_min or x_max input to a length-G float array.

    Accepts NaN, a scalar (broadcast), or a length-G array. Entries for
    periodic groups are not validated here (they're never used).
    """
    arr = np.asarray(v, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(G, float(arr), dtype=np.float64)
    if arr.ndim == 1 and arr.size == G:
        return arr.astype(np.float64, copy=False)
    raise ValueError(
        f"{name} must be a scalar or a length-{G} vector (one entry per "
        f"group); got shape {arr.shape}."
    )


# ===================================================================
#  n_tuple_entropy
# ===================================================================


def n_tuple_entropy(
    p: np.ndarray,
    period: float,
    n: int = 1,
    *,
    sigma: float = 0.0,
    sigma_space: str = "position",
    normalize: bool = True,
    base: float = 2.0,
    n_points_per_dim: int | None = None,
) -> tuple[float, np.ndarray]:
    """Entropy of n-tuples of consecutive step sizes.

    Convenience wrapper around the bind-and-compute pipeline of
    :func:`bind_events`, :func:`build_exp_tens`, and
    :func:`entropy_exp_tens`. With default arguments — ``sigma = 0``
    and ``n_points_per_dim = None`` (which selects the integer-step
    grid ``period``) — this exactly replicates the discrete *n*-tuple
    entropy of Milne & Dean (2016).

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
    normalize : bool
        If True (default), divide by ``log_base(n_points_per_dim ** n)``.
    base : float
        Logarithm base (default 2). Cancels when *normalize* is True.
    n_points_per_dim : int or None
        Grid resolution per dimension. ``None`` (default) selects
        ``period``.

    Sigma semantics
    ---------------
    Under the toolbox convention, sigma applies to the input
    quantity. For *n_tuple_entropy* the input is positions *p*, so
    ``sigma_space = 'position'`` is the default and matches behavior
    elsewhere in the toolbox (sameness, coherence, etc.).

    For ``sigma_space = 'position'``:

      - Each ``p_k`` is treated as ``N(p_k, sigma**2)``.
      - Derived steps ``d_k = p_{k+1} - p_k`` have variance
        ``2 * sigma**2`` per step, with anti-correlation
        ``-sigma**2`` between adjacent steps (they share an endpoint
        with opposite signs).
      - At ``n == 1``, only the marginal step variance matters, and
        the entropy is identical to ``sigma_space = 'interval'`` with
        ``sigma_eff = sigma * sqrt(2)``. This case is handled
        exactly.
      - At ``n >= 2``, the cross-step anti-correlation in principle
        shifts the entropy. The current implementation uses the
        marginal-matched approximation (``sigma_eff = sigma * sqrt(2)``
        per slot, slots independent). Full cross-slot covariance
        handling at ``n >= 2`` is planned for a future release; a
        warning is issued when this approximation is in effect.

    For ``sigma_space = 'interval'``:

      - Each step ``d_k`` is treated as ``N(d_k, sigma**2)``
        independently.
      - This is exactly the v2.0 behavior of this function.
      - Use this if you want the v2 numerical results, or if your
        psychological model treats per-step uncertainty as the
        primitive (rather than positional uncertainty).

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

    p = np.asarray(p, dtype=np.float64).ravel()
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

    # --- Cyclic step sizes: K events -> K cyclic differences ---
    diffs = np.mod(
        np.diff(np.concatenate([p, [p[0] + period]])),
        period,
    )
    diffs_row = diffs.astype(np.float64).reshape(1, -1)

    # --- Bind n consecutive cyclic step sizes ---
    p_bound, w_bound = bind_events(diffs_row, None, n, circular=True)

    # --- Resolve sigma per the sigma_space flag ---
    #
    # 'interval': sigma is per-step uncertainty (v2.0 semantics);
    #             slots are independent with variance sigma**2 each.
    #
    # 'position': sigma is positional uncertainty; each step inherits
    #             variance 2*sigma**2 (since step = p_{k+1} - p_k).
    #             The full position model also includes -sigma**2
    #             anti-correlation between adjacent slots, but this is
    #             not yet implemented; the marginal-matched
    #             approximation (sigma_eff = sigma*sqrt(2), slots
    #             independent) is used at n >= 2. Exact at n = 1.

    if sigma_space == "position":
        sigma_use = sigma * np.sqrt(2.0)
        if n >= 2 and sigma > 0:
            warnings.warn(
                "sigma_space='position' at n >= 2 currently uses a "
                "marginal-matched approximation; cross-slot anti-"
                "correlations are not yet captured. Full position-"
                "aware n-tuple support is planned for a future "
                "release.",
                category=UserWarning,
                stacklevel=2,
            )
    else:
        sigma_use = sigma

    if sigma_use <= 0:
        sigma_use = 1e-12

    # --- Build MAET ---
    T = build_exp_tens(
        p_bound, w_bound,
        [sigma_use], [1] * n, [0] * n,
        [False], [True], [period],
        verbose=False,
    )

    # --- Shannon entropy on the chosen grid ---
    H = entropy_exp_tens(
        T,
        normalize=normalize,
        base=base,
        n_points_per_dim=n_grid,
    )

    # --- Tuples matrix (K, n) for compatibility with the prior API ---
    tuples_out = np.column_stack(
        [row.ravel() for row in p_bound]
    )

    return H, tuples_out
