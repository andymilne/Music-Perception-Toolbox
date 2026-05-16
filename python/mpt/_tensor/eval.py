"""Density evaluation: the eval path.

Public entry points:

* :func:`eval_exp_tens` --- evaluate a density at query points, with
  dispatch over centres / Möbius / direct paths and over normalise
  modes (``none``, ``gaussian``, ``pdf``).
* :func:`eval_exp_tens_raw` --- deprecated shim; raw-array signature
  is now accepted by :func:`eval_exp_tens` directly.

The bulk of the file is the per-method implementations (SA centres
fast, SA centres chunked, SA orbit, MA centres) and the small set of
shape/normalise helpers (``_split_query_to_attr_list``,
``_eval_exp_tens_sa_normalize``, etc.).

The eval path reaches into :mod:`._tensor.windowing` for the per-query
window evaluation (:func:`_evaluate_window_on_query`) when the input
density is a :class:`WindowedMaetDensity`, and into
:mod:`._tensor.dispatch` for path selection.

See USER_GUIDE §4 ("Method selection") and :doc:`/ARCHITECTURE` §4
("Dispatcher pattern") for the conceptual description.
"""
from __future__ import annotations

import warnings

import numpy as np

from .._defaults import _maybe_show_dispatch_msg, _with_dispatch_scope
from .._kernel import gaussian_kernel_sum
from .._utils import estimate_comp_time
from ..spectra import add_spectra

from .build import _looks_like_multi_attr, build_exp_tens
from .canonical import _chord_canonical_key
from .density import ExpTensDensity, MaetDensity, WindowedMaetDensity
from .dispatch import _normalize_density_input, _select_and_estimate_sa
from .windowing import _evaluate_window_on_query




# -------------------------------------------------------------------
#  eval_exp_tens
# -------------------------------------------------------------------


# -------------------------------------------------------------------
#  eval_exp_tens  (public dispatcher)
# -------------------------------------------------------------------


@_with_dispatch_scope
def eval_exp_tens(*args,
                  normalize: str = "none",
                  dedup: bool = True,
                  spectrum=None,
                  precision: int | None = None,
                  method: str = "auto",
                  truncation_sigmas: float | None = None,
                  kernel_precision: str | None = None,
                  verbose: bool = True) -> np.ndarray:
    """Evaluate an expectation tensor density at query points.

    Unified entry point. Accepts five input forms, dispatched on the
    type of the first argument:

    **Pre-built density input** (plus polymorphic lists):

    - ``eval_exp_tens(dens, X)`` — scalar density.
      Returns ``(nQ,)``.
    - ``eval_exp_tens(dens, X, normalize)`` — same with positional
      ``normalize``.
    - ``eval_exp_tens([d1, d2, …], X)`` — list of densities at shared
      query ``X``. Returns ``(M, nQ)`` — every density evaluated at
      every query point. ``normalize`` (positional or kwarg) applies
      to all rows.

    **Raw single-attribute scalar input**:

    - ``eval_exp_tens(p, w, sigma, r, is_rel, is_per, period, X)``.
      Returns ``(nQ,)``.
    - ``eval_exp_tens(p, w, sigma, r, is_rel, is_per, period, X, normalize)``.

    **Raw single-attribute batched input**:

    - ``eval_exp_tens(P, W, sigma, r, is_rel, is_per, period, X)``
      with ``P`` and ``W`` 2-D ``(M, K)`` matrices (rows are chords).
      Returns ``(M, nQ)``.

    **Raw multi-attribute scalar input**:

    - ``eval_exp_tens(p_attr, w, sigma_vec, r_vec, groups,
      is_rel_vec, is_per_vec, period_vec, X)``. Returns ``(nQ,)``.

    Parameters
    ----------
    *args
        Positional arguments depending on input form.
    normalize : {'none', 'gaussian', 'pdf'}, default 'none'
        Density normalisation. Accepted as the trailing positional arg
        in original call patterns, or as a keyword.
    dedup : bool, default True
        Deduplicate structurally-identical chords (canonical-form,
        SA-only). For list/batch input only.
    spectrum : list/tuple, optional
        Per-row :func:`add_spectra` parameters. Raw SA modes only.
    precision : int, optional
        FP-noise tolerance for canonical-form dedup. Raw SA batched
        only.
    method : {'auto', 'centres', 'mobius'}, default 'auto'
        SA-path evaluation strategy. ``'auto'`` lets the dispatcher
        choose between the centres-array path (fast
        at low r) and the Möbius point evaluator (much
        faster at r >= 3 since it bypasses the ``(dim, n_j)`` centres
        tensor whose memory and runtime scale as ``K!/(K-r)!``).
        ``'centres'`` forces the centres path; ``'mobius'`` forces the
        Möbius method. Currently a no-op on the MA path (MA always
        uses centres).
    verbose : bool, default True
        Print progress.

    Returns
    -------
    np.ndarray
        Shape ``(nQ,)`` for scalar density / raw SA scalar / raw MA
        scalar; shape ``(M, nQ)`` for density list / raw SA batched.

    Notes
    -----
    Numerical precision envelope for ``method='mobius'``.

    The Möbius point evaluator is exact to floating-point precision
    when ``K >= r + 2`` and σ is not catastrophically small relative
    to P. The dispatcher enforces these conditions structurally — it
    falls back to the centres path when ``K < r + 2``, when
    ``σ/P > 0.03`` in periodic-relative mode, or when the Möbius
    output contains non-finite values (post-hoc safety net).

    What is *not* currently caught: a finite, but slightly inaccurate
    output from sub-catastrophic Möbius cancellation. None has been
    observed in extensive testing, but a sum-level cancellation
    diagnostic that would close this residual gap is planned. See
    :func:`cos_sim_exp_tens` Notes for the parallel discussion on the
    inner-product path.

    See Also
    --------
    build_exp_tens, cos_sim_exp_tens
    eval_exp_tens_raw : deprecated; superseded by raw input mode here.
    """
    if len(args) < 2:
        raise TypeError(
            "eval_exp_tens requires at least 2 positional arguments."
        )

    a = args[0]

    # ------------------------------------------------------------------
    # Density input dispatch
    # ------------------------------------------------------------------
    is_density_scalar = isinstance(
        a, (ExpTensDensity, MaetDensity, WindowedMaetDensity)
    )
    intends_density_list = False
    if isinstance(a, (list, tuple)):
        if len(a) == 0:
            intends_density_list = True
        elif isinstance(
            a[0], (ExpTensDensity, MaetDensity, WindowedMaetDensity)
        ):
            intends_density_list = True
    elif isinstance(a, np.ndarray) and a.dtype == object:
        intends_density_list = True

    if is_density_scalar or intends_density_list:
        # Density mode: 2 or 3 positional args (dens, x[, normalize]).
        if len(args) == 2:
            dens, x = args
        elif len(args) == 3:
            dens, x, normalize = args
        else:
            raise TypeError(
                f"Density input mode expects 2 or 3 positional "
                f"arguments (dens, x[, normalize]); got {len(args)}."
            )
        if spectrum is not None:
            raise TypeError(
                "'spectrum' kwarg is only valid in raw SA input mode."
            )
        if precision is not None:
            raise TypeError(
                "'precision' kwarg is only valid in raw SA batched input mode."
            )
        if is_density_scalar:
            return _eval_exp_tens_scalar(
                dens, x, normalize, method=method,
                truncation_sigmas=truncation_sigmas,
                kernel_precision=kernel_precision,
                verbose=verbose,
            )
        return _eval_exp_tens_density_list(
            dens, x, normalize, dedup=dedup, method=method,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Raw multi-attribute dispatch
    # ------------------------------------------------------------------
    if _looks_like_multi_attr(a):
        # 9 or 10 positional args.
        if len(args) == 9:
            (p_attr, w_in, sigma_vec, r_vec, groups,
             is_rel_vec, is_per_vec, period_vec, x) = args
        elif len(args) == 10:
            (p_attr, w_in, sigma_vec, r_vec, groups,
             is_rel_vec, is_per_vec, period_vec, x, normalize) = args
        else:
            raise TypeError(
                f"Raw multi-attribute input expects 9 or 10 positional "
                f"arguments (p_attr, w, sigma_vec, r_vec, groups, "
                f"is_rel_vec, is_per_vec, period_vec, x[, normalize]); "
                f"got {len(args)}."
            )
        if spectrum is not None:
            raise TypeError(
                "'spectrum' kwarg is only supported in raw single-attribute "
                "input mode."
            )
        if precision is not None:
            raise TypeError(
                "'precision' kwarg is only valid in raw SA batched input mode."
            )
        return _eval_exp_tens_raw_ma_scalar(
            p_attr, w_in, sigma_vec, r_vec, groups,
            is_rel_vec, is_per_vec, period_vec, x, normalize,
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Raw single-attribute dispatch (1-D = scalar, 2-D = batch)
    # ------------------------------------------------------------------
    if len(args) == 8:
        p, w, sigma, r_, is_rel, is_per, period, x = args
    elif len(args) == 9:
        p, w, sigma, r_, is_rel, is_per, period, x, normalize = args
    else:
        raise TypeError(
            f"Raw single-attribute input expects 8 or 9 positional "
            f"arguments (p, w, sigma, r, is_rel, is_per, period, x"
            f"[, normalize]); got {len(args)}."
        )

    try:
        a_arr = np.asarray(a, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"First argument must be a density object, list of densities, "
            f"numeric array (1-D for a single chord, 2-D for a batch), or "
            f"list of per-attribute matrices for MA raw input; got "
            f"{type(a).__name__}."
        ) from exc

    if a_arr.ndim == 1:
        if precision is not None:
            raise TypeError(
                "'precision' kwarg is only valid for raw SA batched input."
            )
        return _eval_exp_tens_raw_sa_scalar(
            p, w, sigma, r_, is_rel, is_per, period, x, normalize,
            spectrum=spectrum, method=method, verbose=verbose,
        )
    if a_arr.ndim == 2:
        return _eval_exp_tens_raw_sa_batch(
            p, w, sigma, r_, is_rel, is_per, period, x, normalize,
            spectrum=spectrum, precision=precision,
            dedup=dedup, method=method, verbose=verbose,
        )
    raise TypeError(
        f"First argument has unsupported shape {a_arr.shape}; "
        f"raw SA input must be 1-D (single chord) or 2-D (batched)."
    )



def _eval_exp_tens_scalar(
    dens, x, normalize: str, *, method: str = "auto",
    truncation_sigmas: float | None = None,
    kernel_precision: str | None = None,
    verbose: bool,
) -> np.ndarray:
    """Density-scalar dispatch for :func:`eval_exp_tens`.

    Threads ``method`` through to :func:`_eval_exp_tens_sa` for SA densities;
    MA path ignores ``method`` (an MA Möbius method is on the v2.3 roadmap).
    Threads ``truncation_sigmas`` / ``kernel_precision`` through to the
    SA centres path; MA centres routing is deferred (Stage 3).
    """
    if isinstance(dens, WindowedMaetDensity):
        # Evaluate underlying density, multiply elementwise by window.
        underlying = _eval_exp_tens_ma(
            dens.dens, x, normalize,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
            verbose=verbose,
        )
        # Reconstruct per-attribute x_list so we can apply the window.
        x_list = _split_query_to_attr_list(dens.dens, x)
        W_vals = _evaluate_window_on_query(dens, x_list)
        return underlying * W_vals
    if isinstance(dens, MaetDensity):
        return _eval_exp_tens_ma(
            dens, x, normalize,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
            verbose=verbose,
        )
    if isinstance(dens, ExpTensDensity):
        return _eval_exp_tens_sa(
            dens, x, normalize, method=method,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
            verbose=verbose,
        )
    raise TypeError(
        f"dens must be an ExpTensDensity, MaetDensity, or "
        f"WindowedMaetDensity; got {type(dens).__name__}."
    )



def _eval_exp_tens_density_list(
    dens_list, x, normalize: str,
    *, dedup: bool, method: str = "auto",
    truncation_sigmas: float | None = None,
    kernel_precision: str | None = None,
    verbose: bool,
) -> np.ndarray:
    """Evaluate a list of densities at shared query ``x``.

    Returns ``(M, nQ)``. With ``dedup=True``, structurally-identical
    SA densities are evaluated once (canonical-form dedup via
    :func:`_chord_canonical_key`); MA densities bypass dedup.
    """
    is_scalar, dens_tuple = _normalize_density_input(dens_list, name="dens")
    # Note: in this code path we always have a list (intends_density_list
    # was True in the dispatcher), so is_scalar is always False; dens_tuple
    # is the (possibly empty) list of densities.
    m = len(dens_tuple)

    if m == 0:
        # Empty list. Output shape is (0, nQ) where nQ is undetermined
        # without inspecting x; return (0,) for a 1-D query, else (0, 0).
        try:
            x_arr = np.asarray(x, dtype=np.float64)
            if x_arr.ndim == 0:
                return np.empty((0,), dtype=np.float64)
            n_q = x_arr.shape[-1] if x_arr.ndim >= 1 else 0
            return np.empty((0, n_q), dtype=np.float64)
        except (TypeError, ValueError):
            return np.empty((0, 0), dtype=np.float64)

    # Optional dedup for SA densities only.
    use_dedup = dedup and all(isinstance(d, ExpTensDensity) for d in dens_tuple)

    if use_dedup:
        result_cache: dict = {}
        rows = []
        for d in dens_tuple:
            key, _, _ = _chord_canonical_key(
                d.p, d.w, sigma=d.sigma, r=d.r,
                is_rel=d.is_rel, is_per=d.is_per, period=d.period,
            )
            if key not in result_cache:
                result_cache[key] = _eval_exp_tens_scalar(
                    d, x, normalize, method=method,
                    truncation_sigmas=truncation_sigmas,
                    kernel_precision=kernel_precision,
                    verbose=False,
                )
            rows.append(result_cache[key])
        if verbose:
            print(
                f"eval_exp_tens: {m} densities, "
                f"{len(result_cache)} unique after canonical-form dedup."
            )
    else:
        if dedup and verbose:
            print(
                "eval_exp_tens: dedup=True requested but input includes "
                "non-SA densities; computing without dedup."
            )
        rows = [
            _eval_exp_tens_scalar(
                d, x, normalize, method=method,
                truncation_sigmas=truncation_sigmas,
                kernel_precision=kernel_precision,
                verbose=False,
            )
            for d in dens_tuple
        ]

    return np.stack(rows, axis=0)



def _eval_exp_tens_raw_sa_scalar(
    p, w, sigma, r, is_rel, is_per, period,
    x, normalize: str,
    *, spectrum=None, method: str = "auto", verbose: bool,
) -> np.ndarray:
    """Raw SA scalar dispatch: build density (with optional spectrum), evaluate."""
    if spectrum is not None:
        p_arr = np.asarray(p, dtype=np.float64)
        w_arr = (np.ones_like(p_arr) if w is None
                 else np.asarray(w, dtype=np.float64))
        p, w = add_spectra(p_arr, w_arr, *spectrum)
    dens = build_exp_tens(
        p, w, sigma, r, is_rel, is_per, period, verbose=verbose,
    )
    return _eval_exp_tens_scalar(
        dens, x, normalize, method=method, verbose=verbose,
    )



def _eval_exp_tens_raw_sa_batch(
    P, W, sigma, r, is_rel, is_per, period,
    x, normalize: str,
    *, spectrum=None, precision: int | None = None,
    dedup: bool = True, method: str = "auto",
    verbose: bool,
) -> np.ndarray:
    """Raw SA batched dispatch.

    Per-row chord-level dedup of density construction (via
    :func:`_chord_canonical_key`); evaluates each unique density once
    and maps results to all rows. Returns ``(M, nQ)``.
    """
    P = np.asarray(P, dtype=np.float64)
    if P.ndim != 2:
        raise ValueError(f"P must be 2-D for batched mode; got shape {P.shape}.")
    M, K = P.shape
    use_w = W is not None
    if use_w:
        W = np.asarray(W, dtype=np.float64)
        if W.shape != P.shape:
            raise ValueError("W must be the same shape as P.")

    # Optional input precision rounding (collapses FP-noise rows).
    if precision is not None:
        P = np.round(P, precision)
        if use_w:
            W = np.round(W, precision)

    # First pass: build canonical-form keys + density cache.
    dens_cache: dict = {}
    row_to_key: list = [None] * M
    for i in range(M):
        p_row = P[i]
        mask = ~np.isnan(p_row)
        p_valid = p_row[mask]
        if len(p_valid) < r:
            continue   # invalid row -> NaN in output
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
        row_to_key[i] = key

    n_unique = len(dens_cache)
    n_valid = sum(1 for k in row_to_key if k is not None)

    if verbose:
        print(
            f"eval_exp_tens: {M} rows, {n_valid} valid, "
            f"{n_unique} unique chords after canonical-form dedup."
        )

    if n_valid == 0:
        # All rows invalid. Need to infer nQ from x to give the right output shape.
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.ndim == 1:
            n_q = x_arr.size
        elif x_arr.ndim == 2:
            n_q = x_arr.shape[1]
        else:
            n_q = 0
        return np.full((M, n_q), np.nan)

    # Evaluate each unique density once at x.
    eval_cache: dict = {}
    for key, dens in dens_cache.items():
        eval_cache[key] = _eval_exp_tens_scalar(
            dens, x, normalize, method=method, verbose=False,
        )

    # Determine nQ from a representative evaluation.
    sample_vals = next(iter(eval_cache.values()))
    n_q = sample_vals.shape[0] if sample_vals.ndim >= 1 else 1

    out = np.full((M, n_q), np.nan)
    for i in range(M):
        key = row_to_key[i]
        if key is not None:
            out[i] = eval_cache[key]
    return out



def _eval_exp_tens_raw_ma_scalar(
    p_attr, w, sigma_vec, r_vec, groups,
    is_rel_vec, is_per_vec, period_vec,
    x, normalize: str,
    *, verbose: bool,
) -> np.ndarray:
    """Raw MA scalar dispatch: build MA density, evaluate."""
    dens = build_exp_tens(
        p_attr, w, sigma_vec, r_vec, groups,
        is_rel_vec, is_per_vec, period_vec, verbose=verbose,
    )
    return _eval_exp_tens_scalar(dens, x, normalize, verbose=verbose)



def _split_query_to_attr_list(dens: MaetDensity, x):
    """Normalise query input to a list of A per-attribute (dim_a, nQ) arrays.

    Mirrors the logic inside _eval_exp_tens_ma but returns the per-
    attribute list rather than running evaluation.
    """
    A = dens.n_attrs
    dim_per = dens.dim_per_attr
    dim_total = int(dens.dim)

    if isinstance(x, (list, tuple)):
        x_list = []
        n_q = None
        for a, xa in enumerate(x):
            xa = np.asarray(xa, dtype=np.float64)
            if xa.ndim == 1 and dim_per[a] == 1:
                xa = xa.reshape(1, -1)
            if n_q is None:
                n_q = xa.shape[1]
            x_list.append(xa)
        return x_list
    xs = np.asarray(x, dtype=np.float64)
    if xs.ndim == 1 and dim_total == 1:
        xs = xs.reshape(1, -1)
    x_list = []
    row = 0
    for a in range(A):
        da = int(dim_per[a])
        x_list.append(xs[row:row + da, :])
        row += da
    return x_list



# -------------------------------------------------------------------
#  _eval_exp_tens_sa  (single-attribute legacy path)
# -------------------------------------------------------------------


def _eval_exp_tens_sa(
    dens: ExpTensDensity,
    x: np.ndarray,
    normalize: str = "none",
    *,
    method: str = "auto",
    truncation_sigmas: float | None = None,
    kernel_precision: str | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """Single-attribute expectation tensor evaluation (dispatcher).

    Routes between the centres-array path and the Möbius
    point evaluator according to ``method`` and the cost model
    in :func:`_select_sa_eval_method`.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[0] != dens.dim:
        raise ValueError(
            f"x must have {dens.dim} rows (each column is a "
            f"{dens.dim}-D query point)."
        )
    n_q = x.shape[1]
    if n_q == 0:
        return np.zeros(0, dtype=np.float64)

    K = int(dens.p.shape[0])
    sigma_over_P = (
        float(dens.sigma) / float(dens.period) if dens.is_per else 0.0
    )

    # ---- Routing axis ----
    # Hard rules and explicit overrides decide inline; only the
    # discretionary 'auto' case with r >= 2 and K - r >= 2 invokes the
    # dispatcher. Skipping the dispatcher call shaves measurable
    # per-call overhead in MATLAB; Python is less sensitive but we
    # apply the principle uniformly for cross-language parity.
    routing_reason = "user override"
    if method == "centres" or method == "direct":
        chosen = "centres"
        probed = False
        est_sec = 0.0
    elif method == "mobius":
        chosen = "mobius"
        probed = False
        est_sec = 0.0
    elif method == "auto":
        r = int(dens.r)
        if r <= 1 or (K - r) < 2:
            # Hard rules force centres without a dispatcher call.
            chosen = "centres"
            probed = False
            est_sec = 0.0
            if r <= 1:
                routing_reason = f"r = {r}"
            else:
                routing_reason = f"K - r = {K - r} < 2"
        else:
            chosen, probed, est_sec, routing_reason = _select_and_estimate_sa(
                dens, x, n_q,
                method=method,
                truncation_sigmas=truncation_sigmas,
                kernel_precision=kernel_precision,
                verbose=verbose,
            )
    else:
        raise ValueError(
            f"method must be 'auto', 'centres', 'direct', or 'mobius'; "
            f"got '{method}'."
        )

    # Dispatch-decision message: bypasses per-call verbose, gated by
    # the toolbox-wide show_hints flag and throttled once per
    # (function, chosen, routing_reason) per Python process. The
    # throttle is cleared by mpt.reset_defaults(). To fully silence:
    # mpt.set_default(show_hints=False).
    from .._defaults import _maybe_show_dispatch_msg
    _maybe_show_dispatch_msg(
        "eval_exp_tens", chosen, routing_reason, est_sec, probed,
    )

    # ---- Execution axis: detect default-kwargs mode ----
    # Important: ``None`` means "use the global default", not "no
    # feature". So we must consult the defaults before deciding
    # whether the fast path applies — a globally-set finite truncation
    # or 'single' precision must still route through the helper.
    from .._defaults import get_default
    trunc_resolved = (
        truncation_sigmas if truncation_sigmas is not None
        else get_default("truncation_sigmas")
    )
    prec_resolved = (
        kernel_precision if kernel_precision is not None
        else get_default("kernel_precision")
    )
    use_default_kwargs = (
        (trunc_resolved is None or not np.isfinite(trunc_resolved))
        and (prec_resolved == "double")
    )

    # Fire the kernel-evaluation hint once per session when the centres
    # path is about to run with default kwargs. Catches the bypass
    # case too (which skips gaussian_kernel_sum and would otherwise
    # miss the hint).
    if chosen == "centres" and use_default_kwargs:
        from .._defaults import _maybe_show_kernel_eval_hint
        _maybe_show_kernel_eval_hint(
            effective_truncation_sigmas=float("inf"),
            effective_kernel_precision="double",
        )

    if chosen == "mobius":
        vals = _eval_exp_tens_sa_orbit(
            dens, x, n_q,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
            verbose=False,
        )
        # Post-hoc finiteness check. Mirrors the cosine-path safety
        # net: if the Möbius alternating partition sum produces non-finite output
        # (extreme σ → 0 regime), fall back to centres rather than
        # propagating NaN/Inf into the user's result.
        if not np.all(np.isfinite(vals)):
            if verbose:
                warnings.warn(
                    "eval_exp_tens Möbius method produced non-finite "
                    "values; falling back to centres path."
                )
            if use_default_kwargs:
                vals = _eval_exp_tens_sa_centres_fast(dens, x, n_q)
            else:
                vals = _eval_exp_tens_sa_centres(
                    dens, x, n_q,
                    truncation_sigmas=truncation_sigmas,
                    kernel_precision=kernel_precision,
                    verbose=False,
                )
    else:  # 'centres'
        if use_default_kwargs:
            # inline direct broadcast. FP-identical to
            # the helper at default settings, but skips the helper's
            # parameter validation and kwarg construction.
            vals = _eval_exp_tens_sa_centres_fast(dens, x, n_q)
        else:
            vals = _eval_exp_tens_sa_centres(
                dens, x, n_q,
                truncation_sigmas=truncation_sigmas,
                kernel_precision=kernel_precision,
                verbose=False,
            )

    return _eval_exp_tens_sa_normalize(vals, dens, normalize)



def _eval_exp_tens_sa_centres_fast(
    dens: ExpTensDensity, x: np.ndarray, n_q: int,
) -> np.ndarray:
    """inline direct broadcast for the centres path.

    Used by :func:`_eval_exp_tens_sa` when default kwargs apply
    (no truncation, no precision override). FP-identical to the
    helper-routed :func:`_eval_exp_tens_sa_centres` at default
    settings, but skips the helper's per-call validation overhead.
    Handles all (r, is_rel, is_per) combinations.

    Memory-aware ``n_q`` chunking matches the helper's
    ``_exact_kernel_sum``: peak per-chunk allocation is
    ``(dim+1) * n_j * n_qc * 8`` bytes for the difference tensor plus
    per-block intermediates. Without chunking, large workloads
    (e.g. K=72 r=3 nQ=29161 → ~155 GB peak) hit MATLAB's array-size
    cap and Python's memory limits.
    """
    centres = dens.centres
    w_j = dens.w_j
    n_j = int(dens.n_j)
    sigma = float(dens.sigma)
    r = int(dens.r)
    dim = int(dens.dim)
    is_rel = bool(dens.is_rel)
    is_per = bool(dens.is_per)
    period = float(dens.period)

    if n_j == 0 or n_q == 0:
        return np.zeros(n_q, dtype=np.float64)

    bytes_per_scalar = 8  # default-mode is always double
    bytes_needed = (dim + 1) * n_j * n_q * bytes_per_scalar
    mem_limit = 1 * 1024 ** 3  # 1 GB per-chunk cap, matches helper

    if bytes_needed <= mem_limit:
        return _eval_centres_fast_chunk(
            centres, w_j, x, n_q, dim, n_j, sigma, r, is_rel, is_per, period,
        )

    chunk_size = max(1, mem_limit // ((dim + 1) * n_j * bytes_per_scalar))
    vals = np.zeros(n_q, dtype=np.float64)
    for c0 in range(0, n_q, chunk_size):
        c1 = min(c0 + chunk_size, n_q)
        vals[c0:c1] = _eval_centres_fast_chunk(
            centres, w_j, x[:, c0:c1], c1 - c0, dim, n_j,
            sigma, r, is_rel, is_per, period,
        )
    return vals



def _eval_centres_fast_chunk(
    centres, w_j, x, n_qc, dim, n_j, sigma, r, is_rel, is_per, period,
):
    """Single-chunk direct broadcast for the SA centres fast path.

    Mirrors :func:`mpt._kernel._eval_chunk` exactly so the default-mode
    output is FP-bit-identical to v2.0/v2.1.
    """
    D = centres[:, :, None] - x[:, None, :]
    if is_per:
        D = np.mod(D + period / 2, period) - period / 2
    if is_rel:
        Q = np.sum(D ** 2, axis=0) - np.sum(D, axis=0) ** 2 / r
    else:
        Q = np.sum(D ** 2, axis=0)
    E = np.exp(-Q / (2 * sigma ** 2))
    return w_j @ E



def _eval_exp_tens_sa_centres(
    dens: ExpTensDensity,
    x: np.ndarray,
    n_q: int,
    *,
    truncation_sigmas: float | None = None,
    kernel_precision: str | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """Centres-array path for SA evaluation.

    Routes through :func:`mpt._kernel.gaussian_kernel_sum` so
    the ``truncation_sigmas`` and ``kernel_precision`` options apply
    uniformly across centres-path consumers. Default settings
    (``truncation_sigmas=inf``, ``kernel_precision='double'``) produce
    FP-bit-identical output to the v2.0/v2.1 implementation.
    """
    centres = dens.centres
    w_j = dens.w_j
    n_j = dens.n_j
    sigma = dens.sigma
    r = dens.r
    dim = dens.dim
    is_rel = dens.is_rel
    is_per = dens.is_per
    period = dens.period

    # Forward kwargs to the helper. ``None`` means "consult mpt defaults".
    kw = {}
    if is_rel:
        kw["is_rel"] = True
        kw["r"] = int(r)
    if is_per:
        kw["is_per"] = True
        kw["period"] = float(period)
    if truncation_sigmas is not None:
        kw["truncation_sigmas"] = float(truncation_sigmas)
    if kernel_precision is not None:
        kw["kernel_precision"] = kernel_precision

    return gaussian_kernel_sum(centres, w_j, x, float(sigma), **kw)



def _eval_exp_tens_sa_orbit(
    dens: ExpTensDensity,
    x: np.ndarray,
    n_q: int,
    *,
    truncation_sigmas: float | None = None,
    kernel_precision: str | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """Orbit-Möbius point evaluator for SA evaluation.

    Dispatches to :func:`mpt._mobius.eval_orbit_abs` (absolute modes)
    or :func:`mpt._mobius.eval_orbit_rel` (relative modes). Bypasses
    the ``(dim, n_j, n_q)`` intermediate tensor that would dominate
    memory in the centres path at high r.

    Forwards ``truncation_sigmas`` /
    ``kernel_precision`` to the Möbius evaluators. The non-periodic
    per-block kernel sum routes through ``gaussian_kernel_sum`` with
    ``sigma_eff = sigma/sqrt(m)``, gaining truncation natively.
    """
    from .._mobius import eval_orbit_abs, eval_orbit_rel

    p = dens.p
    w = dens.w
    sigma = float(dens.sigma)
    r = int(dens.r)
    is_rel = bool(dens.is_rel)
    is_per = bool(dens.is_per)
    period = float(dens.period)

    if verbose:
        pass

    if is_rel:
        return eval_orbit_rel(
            p, w, sigma, r, x,
            is_per=is_per, period=period,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
        )
    return eval_orbit_abs(
        p, w, sigma, r, x,
        is_per=is_per, period=period,
        truncation_sigmas=truncation_sigmas,
        kernel_precision=kernel_precision,
    )



def _eval_exp_tens_sa_normalize(
    vals: np.ndarray,
    dens: ExpTensDensity,
    normalize: str,
) -> np.ndarray:
    """Apply Gaussian / pdf normalisation to raw SA tensor values."""
    if normalize == "none":
        return vals
    sigma = dens.sigma
    r = dens.r
    dim = dens.dim
    is_rel = dens.is_rel
    w_j = dens.w_j

    det_m = (1.0 / r) if is_rel else 1.0
    gauss_const = (2 * np.pi * sigma**2) ** (-dim / 2) * np.sqrt(det_m)
    vals = vals * gauss_const

    if normalize == "pdf":
        sum_w = np.sum(w_j)
        if sum_w > 0:
            vals = vals / sum_w
        else:
            warnings.warn(
                "Sum of weight products is zero; cannot normalize to pdf."
            )

    return vals



# -------------------------------------------------------------------
#  _eval_exp_tens_ma  (multi-attribute path)
# -------------------------------------------------------------------


def _eval_exp_tens_ma(
    dens: MaetDensity,
    x,
    normalize: str = "none",
    *,
    truncation_sigmas: float | None = None,
    kernel_precision: str | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """Multi-attribute expectation tensor evaluation."""
    A           = dens.n_attrs
    n_j         = dens.n_j
    dim         = dens.dim
    dim_per     = dens.dim_per_attr
    group_of    = dens.group_of_attr
    r_vec       = dens.r
    sigma_g     = dens.sigma
    is_rel_g    = dens.is_rel
    is_per_g    = dens.is_per
    period_g    = dens.period
    centres     = dens.centres
    w_j         = dens.w_j

    # --- Normalise query input to a list of A per-attribute matrices ---

    if isinstance(x, (list, tuple)):
        if len(x) != A:
            raise ValueError(
                f"Query list must have length {A} (n attributes); got {len(x)}."
            )
        x_list = []
        n_q = None
        for a, xa in enumerate(x):
            xa = np.asarray(xa, dtype=np.float64)
            if xa.ndim == 1 and dim_per[a] == 1:
                xa = xa.reshape(1, -1)
            if xa.ndim != 2 or xa.shape[0] != int(dim_per[a]):
                raise ValueError(
                    f"Query for attribute {a} must have {int(dim_per[a])} "
                    f"rows; got shape {xa.shape}."
                )
            if n_q is None:
                n_q = xa.shape[1]
            elif xa.shape[1] != n_q:
                raise ValueError(
                    f"All per-attribute query matrices must share the same "
                    f"number of columns (nQ). Got {n_q} and {xa.shape[1]}."
                )
            x_list.append(xa)
    else:
        xs = np.asarray(x, dtype=np.float64)
        if xs.ndim == 1 and dim == 1:
            xs = xs.reshape(1, -1)
        if xs.ndim != 2 or xs.shape[0] != dim:
            raise ValueError(
                f"Single-matrix query must have {dim} rows (total dim); "
                f"got shape {xs.shape}. For list-form input, wrap the "
                f"per-attribute query matrices in a length-{A} list/tuple."
            )
        n_q = xs.shape[1]
        x_list = []
        row = 0
        for a in range(A):
            da = int(dim_per[a])
            x_list.append(xs[row:row + da, :])
            row += da

    if n_q == 0:
        return np.zeros(0, dtype=np.float64)

    n_pairs = int(n_j) * int(n_q)
    estimate_comp_time(n_pairs, dim, "eval_exp_tens (MAET)", verbose)

    # --- Core evaluation with memory-aware chunking ---
    # Peak per-chunk memory is dominated by the largest per-attribute
    # (dim_a, nJ, nQc) difference tensor plus the (nJ, nQc) accumulator.
    max_dim_a = int(max(dim_per)) if A > 0 else 1
    bytes_per_col = (max_dim_a + 1) * int(n_j) * 8
    mem_limit = 4_000_000_000  # 4 GB default

    bytes_needed = bytes_per_col * int(n_q)
    if bytes_needed <= mem_limit:
        vals = _ma_eval_full(
            centres, w_j, n_j, x_list, n_q,
            A, dim_per, group_of, r_vec, sigma_g,
            is_rel_g, is_per_g, period_g,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
        )
    else:
        chunk_size = max(1, int(mem_limit // max(bytes_per_col, 1)))
        vals = np.zeros(n_q, dtype=np.float64)
        for c_start in range(0, n_q, chunk_size):
            c_end = min(c_start + chunk_size, n_q)
            n_qc = c_end - c_start
            x_chunk = [xa[:, c_start:c_end] for xa in x_list]
            vals[c_start:c_end] = _ma_eval_full(
                centres, w_j, n_j, x_chunk, n_qc,
                A, dim_per, group_of, r_vec, sigma_g,
                is_rel_g, is_per_g, period_g,
                truncation_sigmas=truncation_sigmas,
                kernel_precision=kernel_precision,
            )

    # --- Normalisation ---
    if normalize != "none":
        gauss_const = 1.0
        for a in range(A):
            g = int(group_of[a])
            da = int(dim_per[a])
            if is_rel_g[g] and r_vec[a] >= 2:
                det_m_a = 1.0 / float(r_vec[a])
            else:
                det_m_a = 1.0
            gauss_const *= (2 * np.pi * sigma_g[g]**2) ** (-da / 2) \
                           * np.sqrt(det_m_a)
        vals = vals * gauss_const

        if normalize == "pdf":
            sum_w = float(np.sum(w_j))
            if sum_w > 0:
                vals = vals / sum_w
            else:
                warnings.warn(
                    "Sum of weight products is zero; cannot normalize to pdf."
                )

    return vals



def _ma_eval_full(
    centres, w_j, n_j, x_list, n_qc,
    A, dim_per, group_of, r_vec, sigma_g,
    is_rel_g, is_per_g, period_g,
    *,
    truncation_sigmas=None,
    kernel_precision=None,
):
    """Single-chunk MAET evaluation.

    Accumulates the summed-quadratic exponent across attributes, then
    exponentiates once and does the weighted sum against ``w_j``.

    Default-mode bypass: when ``truncation_sigmas`` is None/Inf and
    ``kernel_precision`` is None/'double', runs the inline accumulation
    inline with no cast machinery and no post-filter branching. This
    keeps default-mode calls at inline cost; the feature kwargs
    only impose their cost when explicitly requested.
    """
    # ---- Resolve precision / truncation from defaults ----
    # ``None`` means "use the global default", not "no feature". A
    # globally-set finite truncation or 'single' precision must still
    # take the feature path, not the default-mode bypass below.
    if kernel_precision is None:
        from .._defaults import get_default
        kernel_precision = get_default("kernel_precision")
    if truncation_sigmas is None:
        from .._defaults import get_default
        truncation_sigmas = get_default("truncation_sigmas")

    # ---- Default-mode bypass ----
    use_default = (
        kernel_precision == "double"
        and (truncation_sigmas is None
             or not np.isfinite(truncation_sigmas))
    )

    if use_default:
        # Direct double accumulation, no casts, no
        # post-filter branching.
        q_total = np.zeros((int(n_j), int(n_qc)), dtype=np.float64)
        for a in range(A):
            g = int(group_of[a])
            da = int(dim_per[a])
            if da == 0:
                continue
            c_a = centres[a]
            x_a = x_list[a]
            d_a = c_a[:, :, None] - x_a[:, None, :]
            if is_per_g[g]:
                pg = float(period_g[g])
                d_a = np.mod(d_a + pg / 2, pg) - pg / 2
            if is_rel_g[g]:
                q_a = (np.sum(d_a ** 2, axis=0)
                       - np.sum(d_a, axis=0) ** 2 / float(r_vec[a]))
            else:
                q_a = np.sum(d_a ** 2, axis=0)
            q_total = q_total + q_a / (2 * sigma_g[g] ** 2)
        e = np.exp(-q_total)
        return w_j @ e

    # ---- Feature-kwargs path: precision casting and / or
    # post-filter truncation. ----
    dtype = np.float32 if kernel_precision == "single" else np.float64

    q_total = np.zeros((int(n_j), int(n_qc)), dtype=dtype)

    for a in range(A):
        g = int(group_of[a])
        da = int(dim_per[a])
        if da == 0:
            continue

        c_a = centres[a].astype(dtype, copy=False)
        x_a = x_list[a].astype(dtype, copy=False)
        d_a = c_a[:, :, None] - x_a[:, None, :]

        if is_per_g[g]:
            pg = dtype(period_g[g])
            d_a = np.mod(d_a + pg / 2, pg) - pg / 2

        if is_rel_g[g]:
            q_a = (np.sum(d_a ** 2, axis=0)
                   - np.sum(d_a, axis=0) ** 2 / dtype(r_vec[a]))
        else:
            q_a = np.sum(d_a ** 2, axis=0)

        q_total = q_total + q_a / (2 * dtype(sigma_g[g]) ** 2)

    use_truncation = (
        truncation_sigmas is not None
        and np.isfinite(truncation_sigmas)
        and truncation_sigmas > 0
    )
    if use_truncation:
        # q_total represents Q/(2sigma^2). The kernel is exp(-q_total).
        # exp(-q_total) is negligible when q_total > k^2/2.
        q_threshold = float(truncation_sigmas) ** 2 / 2.0
        mask = q_total <= q_threshold
        e = np.where(mask, np.exp(-q_total), dtype(0.0))
    else:
        e = np.exp(-q_total)

    result = w_j.astype(dtype, copy=False) @ e
    return result.astype(np.float64, copy=False)



def _eval_core(
    centres, w_j, n_j, x, n_q, dim, sigma, r, is_rel, is_per, period
):
    """Evaluate with automatic memory-aware chunking (SA path)."""
    bytes_needed = (dim + 1) * int(n_j) * int(n_q) * 8
    mem_limit = 4_000_000_000  # 4 GB default

    if bytes_needed <= mem_limit:
        return _eval_full(centres, w_j, n_j, x, n_q, dim, sigma, r, is_rel, is_per, period)

    chunk_size = max(1, int(mem_limit / ((dim + 1) * int(n_j) * 8)))
    vals = np.zeros(n_q)
    for c_start in range(0, n_q, chunk_size):
        c_end = min(c_start + chunk_size, n_q)
        idx = slice(c_start, c_end)
        n_qc = c_end - c_start
        vals[idx] = _eval_full(
            centres, w_j, n_j, x[:, idx], n_qc, dim, sigma, r, is_rel, is_per, period
        )
    return vals



def _eval_full(centres, w_j, n_j, x_q, n_qc, dim, sigma, r, is_rel, is_per, period):
    """Fully vectorized SA density evaluation."""
    # D shape: (dim, nJ, nQc)
    D = centres[:, :, None] - x_q[:, None, :]

    if is_per:
        D = np.mod(D + period / 2, period) - period / 2

    if is_rel:
        Q = np.sum(D**2, axis=0) - np.sum(D, axis=0) ** 2 / r
    else:
        Q = np.sum(D**2, axis=0)

    # E shape: (nJ, nQc)
    E = np.exp(-Q / (2 * sigma**2))

    # Weighted sum: (nJ,) @ (nJ, nQc) → (nQc,)
    return w_j @ E



# -------------------------------------------------------------------
#  eval_exp_tens from raw args (convenience)
# -------------------------------------------------------------------


def eval_exp_tens_raw(
    p: np.ndarray,
    w: np.ndarray | None,
    sigma: float,
    r: int,
    is_rel: bool,
    is_per: bool,
    period: float,
    x: np.ndarray,
    normalize: str = "none",
    *,
    verbose: bool = True,
) -> np.ndarray:
    """Deprecated. Use :func:`eval_exp_tens` directly with raw input.

    .. deprecated:: 2.1
       The raw-input dispatch has been folded into the unified
       :func:`eval_exp_tens` entry point. Pass raw arrays directly:

       .. code-block:: python

          # Old:
          vals = eval_exp_tens_raw(p, w, sigma, r, is_rel, is_per, period, x)
          # New (identical signature):
          vals = eval_exp_tens(p, w, sigma, r, is_rel, is_per, period, x)

    This shim will be removed in a future release.
    """
    warnings.warn(
        "eval_exp_tens_raw is deprecated. The same call signature is "
        "now supported directly by eval_exp_tens (pass raw arrays as the "
        "first arguments instead of pre-built density objects). This shim "
        "will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    return eval_exp_tens(
        p, w, sigma, r, is_rel, is_per, period, x, normalize,
        verbose=verbose,
    )