"""Entropy measures for pitch and rhythm structures."""

from __future__ import annotations

import warnings

import numpy as np

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
    normalize: bool = True,
    base: float = 2.0,
    n_points_per_dim: int = 1200,
    x_min=float("nan"),
    x_max=float("nan"),
    grid_limit: int = _DEFAULT_GRID_LIMIT,
    truncation_sigmas: float | None = None,
    kernel_precision: str | None = None,
    verbose: bool = True,
) -> float | np.ndarray:
    """Entropy of an expectation tensor density.

    Two methods are supported:

    - ``method='shannon'`` (default): grid-based Shannon entropy.
      Discretises the density on a Cartesian-product grid of resolution
      ``n_points_per_dim`` per effective dimension, normalises to a
      pmf, and returns ``H = -Σ q log q``. With ``normalize=True``,
      divides by ``log N`` (grid size) for a value in ``[0, 1]``.
      Supports the full polymorphic-input dispatch (single density,
      list of densities, raw scalar/batched SA, raw MA).

    - ``method='renyi2'``: analytical Rényi-2 (collision) entropy via
      the Möbius inner product and the closed-form total mass.
      Returns ``H_2 = -log_b(<T,T> / Z²)``, the continuous Rényi-2
      entropy of the normalised density ``q = T/Z``. Computed in
      closed form with no grid; works at arbitrary tensor order ``r``
      where the grid path would exhaust memory. Currently restricted
      to single-density input (scalar density or raw scalar SA/MA);
      list and batched input forms are not yet supported with this
      method.

      ``normalize=True`` is not currently supported with
      ``method='renyi2'`` — the natural normaliser ``log_b(V)`` (where
      V is the support volume) yields a [-∞, 1] range rather than
      Shannon's [0, 1], and resolving the discrepancy is a separate
      question. Pass ``normalize=False`` to use this method.

    Input forms (Shannon supports all; Rényi-2 supports only the
    scalar forms — scalar pre-built density, raw SA scalar, raw MA
    scalar — and raises ``NotImplementedError`` on list / batched
    forms):

    **Pre-built density input**:

    - ``entropy_exp_tens(dens)`` — scalar density.
      Returns a Python float.
    - ``entropy_exp_tens([d1, d2, …])`` — list of densities
      (Shannon only). Returns ``(M,)`` ndarray.

    **Raw single-attribute scalar input**:

    - ``entropy_exp_tens(p, w, sigma, r, is_rel, is_per, period)``.
      Returns a Python float. Optional ``spectrum``.

    **Raw single-attribute batched input** (Shannon only):

    - ``entropy_exp_tens(P, W, sigma, r, is_rel, is_per, period)``
      with ``P`` and ``W`` 2-D ``(M, K)`` matrices (rows are chords).
      Returns ``(M,)``. Optional ``spectrum``, ``precision``,
      ``dedup``.

    **Raw multi-attribute scalar input**:

    - ``entropy_exp_tens(p_attr, w, sigma_vec, r_vec, groups,
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
    method : {'shannon', 'renyi2'}, default 'shannon'
        Entropy variant. See the introduction above.
    precision : int, optional
        Round canonical values to this many decimal places, to absorb
        FP noise when deduplicating. Raw SA batched only (Shannon).
    dedup : bool, default True
        Deduplicate structurally-identical chords. List/batch only
        (Shannon).
    normalize : bool, default True
        Shannon only: divide by ``log_b(N)`` to give ``[0, 1]``.
        ``method='renyi2'`` with ``normalize=True`` raises.
    base : float, default 2.0
        Logarithm base. For Shannon with ``normalize=True`` the base
        cancels.
    n_points_per_dim : int, default 1200
        Shannon only: grid resolution per effective dimension.
    x_min, x_max
        Shannon, non-periodic only: grid bounds. SA scalar; MA scalar
        (broadcast) or length-G vector.
    grid_limit : int
        Shannon only: ceiling on total grid size before allocation.

    Returns
    -------
    float or np.ndarray
        Scalar in scalar input modes; ``(M,)`` ndarray in list/batch
        modes. ``method='renyi2'`` always returns a scalar (it is
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
    # Validate the method kwarg and reject the unimplementable combination
    # renyi2 + normalize=True (the analytical Rényi-2 form has no natural
    # [0, 1] reference).
    if method not in ("shannon", "renyi2"):
        raise ValueError(
            f"method must be 'shannon' or 'renyi2'; got {method!r}."
        )
    if method == "renyi2" and normalize:
        raise NotImplementedError(
            "method='renyi2' with normalize=True is not implemented. "
            "The continuous Rényi-2 entropy ranges over (-∞, log_b V] "
            "rather than Shannon's [0, log_b N], so a uniform "
            "normaliser does not yield a [0, 1] value. Pass "
            "normalize=False to use this method."
        )

    # Dispatch on method. Both methods do parallel per-input-form
    # resolution; see ``_entropy_exp_tens_shannon_dispatch`` and
    # ``_entropy_exp_tens_renyi2_dispatch`` for the per-form branching.
    # Shannon supports the full input surface (single density, list of
    # densities, raw scalar SA/MA, raw batched SA, windowed MA). Renyi-2
    # is restricted to single-density input — list, batched, and
    # windowed forms are not yet implemented and produce informative
    # errors.
    if method == "shannon":
        return _entropy_exp_tens_shannon_dispatch(
            p_or_dens, args,
            spectrum=spectrum, precision=precision, dedup=dedup,
            normalize=normalize, base=base,
            n_points_per_dim=n_points_per_dim,
            x_min=x_min, x_max=x_max, grid_limit=grid_limit,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
            verbose=verbose,
        )
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
        return _entropy_exp_tens_density_list(
            p_or_dens,
            dedup=dedup,
            normalize=normalize, base=base,
            n_points_per_dim=n_points_per_dim,
            x_min=x_min, x_max=x_max, grid_limit=grid_limit,
        )

    # --- Raw args: dispatch on type of p ---
    if _looks_like_ma_p(p_or_dens):
        if len(args) != 7:
            raise ValueError(
                f"Multi-attribute raw call expects 8 positional arguments "
                f"(p_attr, w, sigma_vec, r_vec, groups, is_rel_vec, "
                f"is_per_vec, period_vec); got {1 + len(args)}."
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

    # SA raw args. Distinguish scalar (1-D) from batched (2-D) by shape.
    if len(args) != 6:
        raise ValueError(
            f"Single-attribute raw call expects 7 positional arguments "
            f"(p, w, sigma, r, is_rel, is_per, period); got {1 + len(args)}."
        )
    w, sigma, r, is_rel, is_per, period = args

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
        return _entropy_exp_tens_sa(
            p_or_dens, w, sigma, r, is_rel, is_per, period,
            spectrum=spectrum, normalize=normalize, base=base,
            n_points_per_dim=n_points_per_dim,
            x_min=x_min, x_max=x_max,
        )
    if p_arr.ndim == 2:
        return _entropy_exp_tens_raw_sa_batch(
            p_arr, w, sigma, r, is_rel, is_per, period,
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
    P, W, sigma, r, is_rel, is_per, period,
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
                p_aug, w_aug, sigma, r, is_rel, is_per, period, verbose=False,
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
        return dens, False

    # SA raw args.
    if len(args) != 6:
        raise ValueError(
            f"Single-attribute raw call expects 7 positional arguments "
            f"(p, w, sigma, r, is_rel, is_per, period); got {1 + len(args)}."
        )
    w, sigma, r, is_rel, is_per, period = args
    p = np.asarray(p_or_dens, dtype=np.float64).ravel()
    if spectrum is not None:
        p, w = add_spectra(p, w, *spectrum)
    dens = build_exp_tens(
        p, w, sigma, r, is_rel, is_per, period, verbose=False,
    )
    return dens, True


# -------------------------------------------------------------------
#  Rényi-2 entropy helpers (analytical, Möbius)
# -------------------------------------------------------------------


def _renyi2_exp_tens_sa(dens, *, base: float) -> float:
    """Analytical Rényi-2 entropy of a SA expectation tensor.

    Computes ``H_2 = -log_b(<T,T> / Z²)`` where ``<T,T>`` is evaluated
    via the Möbius inner-product machinery (or a direct
    pairwise formula at ``r = 1`` where the orbit table is undefined)
    and ``Z = ∫T(x)dx`` via the closed-form total-mass formulae in
    :mod:`mpt._mobius`.
    """
    from ._mobius import total_mass_abs, total_mass_rel

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
        if not np.isfinite(ip_xx) or ip_xx <= 0:
            raise FloatingPointError(
                f"Computed <T,T>={ip_xx} is non-positive or non-finite."
            )
        if not np.isfinite(Z) or Z <= 0:
            raise FloatingPointError(
                f"Computed Z={Z} is non-positive or non-finite."
            )
        return -float(np.log(ip_xx / (Z * Z)) / np.log(base))

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

    if not np.isfinite(ip_xx) or ip_xx <= 0:
        raise FloatingPointError(
            f"Computed <T,T>={ip_xx} via the Möbius method is "
            "non-positive or non-finite. The input density may be "
            "degenerate (all weights zero), or the parameters may lie "
            "in a regime where the alternating Möbius sum has lost all "
            "significant digits. Try a less extreme σ/P ratio, smaller "
            "r, or larger K-r margin."
        )

    # Z via closed-form Möbius total mass.
    if is_rel:
        Z = total_mass_rel(p, w, sigma, r)
    else:
        Z = total_mass_abs(p, w, sigma, r)
    if not np.isfinite(Z) or Z <= 0:
        raise FloatingPointError(
            f"Computed Z={Z} is non-positive or non-finite."
        )

    return -float(np.log(ip_xx / (Z * Z)) / np.log(base))


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
    dens = dens_or_windowed
    A = dens.n_attrs
    N = dens.n
    if A == 0 or N == 0:
        return 0.0

    # ---- <T, T> via per-attribute Möbius IP ----
    # The per-(n, m) cancellation ratio aggregated across attributes
    # was empirically shown to fire spuriously in 100% of typical
    # musical regimes for self-IPs (sweep_self_ip.py): off-diagonal
    # entries can have low ratios while the diagonal entries (which
    # dominate the sum) are clean, so the sum Σ P_xx[n,m] is correct
    # even when some entries are noisy. We therefore rely solely on
    # a post-hoc finite/positive check. The Bulger fallback was
    # abandoned for the same convention-mismatch reason as in the
    # SA path.
    P_xx = np.ones((N, N), dtype=np.float64)
    for a in range(A):
        g = int(dens.group_of_attr[a])
        r_a = int(dens.r[a])
        sigma = float(dens.sigma[g])
        is_rel = bool(dens.is_rel[g])
        is_per = bool(dens.is_per[g])
        period = float(dens.period[g])
        Pa = dens.p_attr[a]
        Wa = dens.w[a]
        I_xx = _ma_per_attr_inner_matrix(
            Pa, Wa, Pa, Wa, sigma, r_a, is_rel, is_per, period,
        )
        P_xx *= I_xx
    ip_xx = float(P_xx.sum())

    if not np.isfinite(ip_xx) or ip_xx <= 0:
        raise FloatingPointError(
            f"Computed <T,T>={ip_xx} via the Möbius method is "
            "non-positive or non-finite. The input density may be "
            "degenerate, or the parameters may lie in a regime where "
            "the per-attribute alternating sum has lost all significant "
            "digits. Try a less extreme σ/P ratio, smaller r, or larger "
            "K-r margin."
        )

    # ---- Z = Σ_n Π_a Z_a^(n) ----
    # Each per-event-per-attribute factor is the SA total mass
    # computed on that event's slot vector. The (Möbius) per-slot
    # σ_a factors carry over without modification.
    Z_per_event_attr = np.empty((N, A), dtype=np.float64)
    for a in range(A):
        g = int(dens.group_of_attr[a])
        r_a = int(dens.r[a])
        sigma = float(dens.sigma[g])
        is_rel = bool(dens.is_rel[g])
        Pa = dens.p_attr[a]   # (K_a, N)
        Wa = dens.w[a]        # (K_a, N)
        for n in range(N):
            if is_rel:
                Z_an = total_mass_rel(Pa[:, n], Wa[:, n], sigma, r_a)
            else:
                Z_an = total_mass_abs(Pa[:, n], Wa[:, n], sigma, r_a)
            Z_per_event_attr[n, a] = Z_an
    Z = float(np.prod(Z_per_event_attr, axis=1).sum())
    if not np.isfinite(Z) or Z <= 0:
        raise FloatingPointError(
            f"Computed Z={Z} is non-positive or non-finite."
        )

    return -float(np.log(ip_xx / (Z * Z)) / np.log(base))


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
        T = build_exp_tens(p, w, sigma, r, is_rel, is_per, period, verbose=False)

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
    p,
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

    p_arr = np.asarray(p, dtype=np.float64)
    if p_arr.ndim == 2:
        return _n_tuple_entropy_batched(
            p_arr, period, n,
            sigma=sigma, sigma_space=sigma_space,
            normalize=normalize, base=base,
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

    # --- Cyclic first differences via the framework's circular mode ---
    # difference_events with circular=True wraps at the sequence
    # boundary (output position 0 holds p(0) - p(N-1)); the
    # downstream periodic kernel handles mod-period wrapping at
    # evaluation time, so no explicit mod is needed here. The
    # resulting multiset of consecutive-difference n-grams is
    # invariant under the cyclic rotation that distinguishes this
    # ordering from the equivalent "diff first, wrap difference at
    # position N" convention.
    p_row = p.astype(np.float64).reshape(1, -1)
    p_diff_list, _, _ = difference_events(
        [p_row], None, None, 1, circular=True,
    )
    diffs_row = p_diff_list[0]

    # --- Bind n consecutive cyclic step sizes ---
    p_bound, w_bound, _ = bind_events(
        [diffs_row], None, None, n, circular=True,
    )

    # --- Resolve sigma per the sigma_space flag ---
    #
    # 'interval': sigma is per-step uncertainty (legacy step-size mode);
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


def _n_tuple_entropy_batched(
    P, period, n,
    *,
    sigma, sigma_space, normalize, base, n_points_per_dim,
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
            normalize=normalize, base=base,
            n_points_per_dim=n_points_per_dim,
        )
        H_out[i] = H_i
        tuples_list[i] = t_i
        cache[key] = (H_i, t_i)

    return H_out, tuples_list
