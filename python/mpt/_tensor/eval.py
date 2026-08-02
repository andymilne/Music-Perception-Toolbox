"""Density evaluation: the eval path.

Public entry points:

* :func:`eval_exp_tens` --- evaluate a density at query points, with
  dispatch over centres / Möbius / direct paths and over normalise
  modes (``none``, ``gaussian``, ``pdf``).
* :func:`eval_exp_tens_raw` --- deprecated shim; raw-array signature
  is now accepted by :func:`eval_exp_tens` directly.

The bulk of the file is the per-method implementations (single-multiset centres
fast, single-multiset centres chunked, single-multiset orbit, multi-attribute centres) and the small set of
shape/normalise helpers (``_split_query_to_attr_list``,
the normalisation helpers, etc.).

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
from .._utils import (
    kernel_chunk_bytes_resolved,
    with_kernel_chunk_bytes_pin,
)
from .._kernel import gaussian_kernel_sum
from ..spectra import add_spectra

from .build import _looks_like_multi_attr, build_exp_tens
from .canonical import _chord_canonical_key
from .density import is_single_multiset
from .density import MaetDensity, WindowedMaetDensity
from .dispatch import (_compute_Q, _compute_Q_inner_blocks, _inner_r_vec,
                       _normalize_density_input,
                       _quadratic_form_det, _gaussian_mass_const)
from .windowing import _evaluate_window_on_query




# -------------------------------------------------------------------
#  eval_exp_tens
# -------------------------------------------------------------------


# -------------------------------------------------------------------
#  eval_exp_tens  (public dispatcher)
# -------------------------------------------------------------------


@_with_dispatch_scope
@with_kernel_chunk_bytes_pin
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

    **Raw single-multiset scalar input**:

    - ``eval_exp_tens(p, w, sigma, r, is_rel, is_per, period, X)``.
      Returns ``(nQ,)``.
    - ``eval_exp_tens(p, w, sigma, r, is_rel, is_per, period, X, normalize)``.

    **Raw single-multiset batched input**:

    - ``eval_exp_tens(P, W, sigma, r, is_rel, is_per, period, X)``
      with ``P`` and ``W`` 2-D ``(M, K)`` matrices (rows are chords).
      Returns ``(M, nQ)``.

    **Raw multi-attribute scalar input**:

    - ``eval_exp_tens(p_attr, w, sigma_vec, r_vec,
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
        single-multiset-only). For list/batch input only.
    spectrum : list/tuple, optional
        Per-row :func:`add_spectra` parameters. Raw single-multiset modes only.
    precision : int, optional
        FP-noise tolerance for canonical-form dedup. Raw single-multiset batched
        only.
    method : {'auto', 'centres', 'mobius'}, default 'auto'
        Single-multiset-path evaluation strategy. ``'auto'`` lets the dispatcher
        choose between the centres-array path and the Möbius point
        evaluator, weighing both wall time and the centres working-set
        memory (see :func:`mpt._tensor.dispatch._select_ma_eval`). The
        Möbius evaluator bypasses the ``(dim, n_j)`` centres tensor
        (``n_j = K!/(K-r)!``), so it wins decisively at large ``K``
        where that array explodes. In **absolute** mode it is also
        faster than centres at ``r >= 3`` outright, being grid-free at
        ``O(B_r · r · K)`` per query. In **relative** mode it is *not*
        grid-free — it integrates the translation marginal on a u-grid
        of ``N_u`` points, at ``O(B_r · r · K · N_u)`` per query on its
        direct strategy, or ``O(B_r · r · N_u)`` per query (plus an
        amortised tabulation) on its factored strategy, which removes
        the ``K`` factor from the integrand by reading back ``r``
        precomputed smoothed event distributions (non-periodic only;
        picked by an internal cost gate for batched workloads;
        read-back accuracy tied to ``truncation_sigmas``). The ``N_u``
        overhead still makes the centres path (no ``N_u`` factor, cost
        ``O(K!/(K-r)!)``) cheaper for scalar queries at small ``K``;
        the dispatcher crosses over to Möbius as ``K`` or the batch
        size grows.
        ``'centres'`` forces the centres path; ``'mobius'`` forces the
        Möbius method. Currently a no-op on the MA path (there is no MA
        Möbius point evaluator yet; MA eval always uses centres).
    verbose : bool, default True
        Print progress.

    Returns
    -------
    np.ndarray
        Shape ``(nQ,)`` for scalar density / raw single-multiset scalar / raw MA
        scalar; shape ``(M, nQ)`` for density list / raw single-multiset batched.

    Notes
    -----
    Numerical precision envelope for ``method='mobius'``.

    Accuracy is governed by ``truncationSigmas``: the Möbius point
    evaluator's agreement with the centres path tracks the truncation
    budget, and how close ``K`` is to ``r`` does not bear on it. The
    dispatcher falls back to the centres path when ``σ/P > 0.03`` in
    periodic-relative mode, and when the Möbius output contains
    non-finite values; otherwise it chooses on cost.

    What is *not* currently caught: a finite, but slightly inaccurate
    output from accumulated Möbius per-term error. None has been
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

    # Resolve the truncation width once, at the single entry point, so
    # every downstream path (centres fast, chunked, Möbius, MA) receives
    # a finite width and truncates identically. Per the toolbox contract
    # a user-supplied ``inf`` resolves to the finite accuracy-floor width
    # (the 1e-12 floor), NOT to unbounded/exact summation; genuinely
    # exhaustive summation is reachable only internally, by widening the
    # accuracy-floor epsilon via ``accuracy_floor_context`` (as the
    # golden-value regeneration does). ``None`` takes the global default.
    from .._defaults import resolve_truncation_sigmas
    truncation_sigmas = resolve_truncation_sigmas(truncation_sigmas)

    a = args[0]

    # ------------------------------------------------------------------
    # Density input dispatch
    # ------------------------------------------------------------------
    is_density_scalar = isinstance(
        a, (MaetDensity, WindowedMaetDensity)
    )
    intends_density_list = False
    if isinstance(a, (list, tuple)):
        if len(a) == 0:
            intends_density_list = True
        elif isinstance(
            a[0], (MaetDensity, WindowedMaetDensity)
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
                "'spectrum' kwarg is only valid in raw single-multiset input mode."
            )
        if precision is not None:
            raise TypeError(
                "'precision' kwarg is only valid in raw single-multiset batched input mode."
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
        # Positional geometry order: p_attr, w, sigma_vec, r_vec,
        # is_rel_vec, is_per_vec, period_vec, [is_sym_vec], x.
        # 8 args omit is_sym_vec (defaults to symmetric); 9 supply it.
        # normalize is keyword-only in raw mode.
        is_sym_vec = None
        if len(args) == 8:
            (p_attr, w_in, sigma_vec, r_vec,
             is_rel_vec, is_per_vec, period_vec, x) = args
        elif len(args) == 9:
            (p_attr, w_in, sigma_vec, r_vec,
             is_rel_vec, is_per_vec, period_vec, is_sym_vec, x) = args
        elif len(args) == 10:
            (p_attr, w_in, sigma_vec, r_vec,
             is_rel_vec, is_per_vec, period_vec, is_sym_vec, x, normalize) = args
        else:
            raise TypeError(
                f"Raw multi-attribute input expects 8, 9, or 10 positional "
                f"arguments (p_attr, w, sigma_vec, r_vec, "
                f"is_rel_vec, is_per_vec, period_vec[, is_sym_vec], x"
                f"[, normalize]); got {len(args)}."
            )
        from .aniso import sigma_vec_has_kernel_cov
        if sigma_vec_has_kernel_cov(sigma_vec):
            if spectrum is not None:
                raise TypeError(
                    "'spectrum' is not supported with a matrix-valued "
                    "kernel covariance."
                )
            from .build import build_exp_tens as _bet
            dens_a = _bet(
                p_attr, w_in, sigma_vec, r_vec,
                is_rel_vec, is_per_vec, period_vec, is_sym_vec,
                verbose=False,
            )
            return _eval_exp_tens_scalar(
                dens_a, x, normalize, method=method,
                truncation_sigmas=truncation_sigmas,
                kernel_precision=kernel_precision, verbose=verbose,
            )
        if spectrum is not None:
            raise TypeError(
                "'spectrum' kwarg is only supported in raw single-multiset "
                "input mode."
            )
        if precision is not None:
            raise TypeError(
                "'precision' kwarg is only valid in raw single-multiset batched input mode."
            )
        return _eval_exp_tens_raw_ma_scalar(
            p_attr, w_in, sigma_vec, r_vec,
            is_rel_vec, is_per_vec, period_vec, is_sym_vec, x, normalize,
            method=method, truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision, verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Raw single-multiset dispatch (1-D = scalar, 2-D = batch)
    # ------------------------------------------------------------------
    # Positional geometry order: p, w, sigma, r, is_rel, is_per, period,
    # [is_sym], x [, normalize]. 8 args omit is_sym (defaults symmetric)
    # and normalize (kwarg/default); 9 supply is_sym; 10 supply both.
    is_sym = None
    if len(args) == 8:
        p, w, sigma, r_, is_rel, is_per, period, x = args
    elif len(args) == 9:
        p, w, sigma, r_, is_rel, is_per, period, is_sym, x = args
    elif len(args) == 10:
        p, w, sigma, r_, is_rel, is_per, period, is_sym, x, normalize = args
    else:
        raise TypeError(
            f"Raw single-multiset input expects 8, 9, or 10 positional "
            f"arguments (p, w, sigma, r, is_rel, is_per, period"
            f"[, is_sym], x[, normalize]); got {len(args)}."
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

    from .aniso import is_kernel_cov
    if is_kernel_cov(sigma):
        if spectrum is not None:
            raise TypeError(
                "'spectrum' is not supported with a matrix-valued "
                "kernel covariance (spectral augmentation changes the "
                "multiset size, breaking r == K)."
            )
        if a_arr.ndim != 1:
            raise NotImplementedError(
                "Raw single-multiset batched (2-D) input is not supported with a "
                "matrix-valued kernel covariance; carry the tuples as "
                "events of an ordered multi-attribute form, or build "
                "per-row density objects."
            )
        from .build import build_exp_tens as _bet
        dens_a = _bet(
            p, w, sigma, r_, is_rel, is_per, period,
            (True if is_sym is None else is_sym), verbose=False,
        )
        return _eval_exp_tens_scalar(
            dens_a, x, normalize, method=method,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision, verbose=verbose,
        )

    if a_arr.ndim == 1:
        if precision is not None:
            raise TypeError(
                "'precision' kwarg is only valid for raw single-multiset batched input."
            )
        return _eval_exp_tens_raw_single_multiset_scalar(
            p, w, sigma, r_, is_rel, is_per, period, is_sym, x, normalize,
            spectrum=spectrum, method=method,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision, verbose=verbose,
        )
    if a_arr.ndim == 2:
        return _eval_exp_tens_raw_single_multiset_batch(
            p, w, sigma, r_, is_rel, is_per, period, is_sym, x, normalize,
            spectrum=spectrum, precision=precision,
            dedup=dedup, method=method,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision, verbose=verbose,
        )
    raise TypeError(
        f"First argument has unsupported shape {a_arr.shape}; "
        f"raw single-multiset input must be 1-D (single chord) or 2-D (batched)."
    )



def _eval_exp_tens_scalar(
    dens, x, normalize: str, *, method: str = "auto",
    truncation_sigmas: float | None = None,
    kernel_precision: str | None = None,
    verbose: bool,
) -> np.ndarray:
    """Density-scalar dispatch for :func:`eval_exp_tens`.

    Threads ``truncation_sigmas`` / ``kernel_precision`` through to the
    single-multiset centres path; MA centres routing is deferred (Stage 3).
    """
    from .aniso import density_has_kernel_cov, whiten_query, \
        density_logdet_sum
    if isinstance(dens, WindowedMaetDensity):
        if density_has_kernel_cov(dens.dens):
            raise NotImplementedError(
                "Matrix-valued kernel covariances are not supported on "
                "windowed densities; use windowed_similarity, whose "
                "internal builds accept them."
            )
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
    _aniso = density_has_kernel_cov(dens)
    if _aniso:
        # Whitened coordinates: transform the query once; the internal
        # sigma is 1. The Gaussian normalization constant acquires
        # det(Sigma)^{-1/2}, applied after the isotropic machinery.
        x = whiten_query(dens, x)
    # Single-multiset corner (A = N = 1, flat): when the query uses
    # the single-multiset dialect (a plain array whose row count is
    # A single-multiset (A=1, N=1) density is just the multi-attribute
    # density at that corner, and the multi-attribute path now handles it
    # correctly and quickly: its factored Möbius evaluator reduces to the
    # single-multiset Möbius evaluator at A=1, and its cost-model
    # dispatch selects centres or Möbius exactly as the single-multiset
    # selector would. (An earlier interception routed the corner through
    # the single-multiset evaluator to avoid a large regression when the
    # multi-attribute path lacked a Möbius route; that route now exists,
    # so the interception is no longer needed --- the corner is the
    # general case at A=1.)
    if isinstance(dens, MaetDensity):
        vals = _eval_exp_tens_ma(
            dens, x, normalize,
            method=method,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
            verbose=verbose,
        )
        if _aniso and normalize != "none":
            vals = vals * np.exp(-0.5 * density_logdet_sum(dens))
        return vals
    raise TypeError(
        f"dens must be a MaetDensity or "
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
    single-multiset densities are evaluated once (canonical-form dedup via
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

    # Optional dedup for single-multiset densities only. Deduplicates
    # by canonical chord key so repeated collections are evaluated once.
    use_dedup = dedup and all(is_single_multiset(d) for d in dens_tuple)

    if use_dedup:
        result_cache: dict = {}
        rows = []
        for d in dens_tuple:
            _pd, _wd = d.p_attr[0][:, 0], d.w[0][:, 0]
            key, _, _ = _chord_canonical_key(
                _pd, _wd,
                sigma=float(d.sigma[0]), r=int(d.r[0]),
                is_rel=bool(d.is_rel[0]), is_per=bool(d.is_per[0]),
                period=float(d.period[0]),
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
                "multi-attribute densities; computing without dedup."
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



def _eval_exp_tens_raw_single_multiset_scalar(
    p, w, sigma, r, is_rel, is_per, period, is_sym,
    x, normalize: str,
    *, spectrum=None, method: str = "auto",
    truncation_sigmas: float | None = None,
    kernel_precision: str | None = None,
    verbose: bool,
) -> np.ndarray:
    """Raw single-multiset scalar dispatch: build density (with optional spectrum), evaluate."""
    if spectrum is not None:
        p_arr = np.asarray(p, dtype=np.float64)
        w_arr = (np.ones_like(p_arr) if w is None
                 else np.asarray(w, dtype=np.float64))
        p, w = add_spectra(p_arr, w_arr, *spectrum)
    dens = build_exp_tens(
        p, w, sigma, r, is_rel, is_per, period,
        True if is_sym is None else is_sym, verbose=verbose,
    )
    return _eval_exp_tens_scalar(
        dens, x, normalize, method=method,
        truncation_sigmas=truncation_sigmas,
        kernel_precision=kernel_precision, verbose=verbose,
    )



def _eval_exp_tens_raw_single_multiset_batch(
    P, W, sigma, r, is_rel, is_per, period, is_sym,
    x, normalize: str,
    *, spectrum=None, precision: int | None = None,
    dedup: bool = True, method: str = "auto",
    truncation_sigmas: float | None = None,
    kernel_precision: str | None = None,
    verbose: bool,
) -> np.ndarray:
    """Raw single-multiset batched dispatch.

    Per-row chord-level dedup of density construction (via
    :func:`_chord_canonical_key`); evaluates each unique density once
    and maps results to all rows. Returns ``(M, nQ)``.
    """
    P = np.asarray(P, dtype=np.float64)
    if P.ndim != 2:
        raise ValueError(f"P must be 2-D for batched mode; got shape {P.shape}.")

    # The per-row dedup keys rows by a multiset canonical form, which
    # collapses rows that share a multiset but differ in order. That is
    # correct only for the symmetric reading: under [sym]=0 the order is
    # significant, so the dedup would silently merge distinct ordered
    # densities. Reject rather than return a wrong answer. Order-aware
    # batched dedup is a tracked follow-up; use scalar input for ordered
    # densities.
    if (is_sym is not None) and (not bool(np.all(is_sym))) and r > 1:
        raise NotImplementedError(
            "eval_exp_tens batched (2-D) input does not yet support "
            "[sym]=0 (ordered) densities at r > 1: the batched dedup "
            "canonicalises each row's multiset and would merge "
            "order-distinct rows. Evaluate ordered densities one row at "
            "a time (scalar input)."
        )

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
                True if is_sym is None else is_sym,
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
            dens, x, normalize, method=method,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision, verbose=False,
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
    p_attr, w, sigma_vec, r_vec,
    is_rel_vec, is_per_vec, period_vec, is_sym_vec,
    x, normalize: str,
    *, method: str = "auto",
    truncation_sigmas: float | None = None,
    kernel_precision: str | None = None,
    verbose: bool,
) -> np.ndarray:
    """Raw MA scalar dispatch: build MA density, evaluate."""
    dens = build_exp_tens(
        p_attr, w, sigma_vec, r_vec,
        is_rel_vec, is_per_vec, period_vec, is_sym_vec, verbose=verbose,
    )
    return _eval_exp_tens_scalar(
        dens, x, normalize, method=method,
        truncation_sigmas=truncation_sigmas,
        kernel_precision=kernel_precision, verbose=verbose,
    )



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


# -------------------------------------------------------------------
#  _eval_exp_tens_ma  (multi-attribute path)
# -------------------------------------------------------------------


def _ma_join_query(dens: MaetDensity, x) -> np.ndarray:
    """Return the query as a single ``(D, n_q)`` matrix.

    Accepts either the list form (one ``(dim_per_attr[a], n_q)`` matrix
    per attribute) or an already-joined ``(D, n_q)`` matrix, validating
    shapes identically to the joint-centres path, and returns the
    row-stacked joint matrix the factored evaluator consumes.
    """
    A = int(dens.n_attrs)
    dim = int(dens.dim)
    dim_per = [int(v) for v in np.atleast_1d(dens.dim_per_attr)]
    if isinstance(x, (list, tuple)):
        if len(x) != A:
            raise ValueError(
                f"Query list must have length {A} (n attributes); "
                f"got {len(x)}."
            )
        blocks = []
        n_q = None
        for a, xa in enumerate(x):
            xa = np.asarray(xa, dtype=np.float64)
            if xa.ndim == 1 and dim_per[a] == 1:
                xa = xa.reshape(1, -1)
            if xa.ndim != 2 or xa.shape[0] != dim_per[a]:
                raise ValueError(
                    f"Query for attribute {a} must have {dim_per[a]} "
                    f"rows; got shape {xa.shape}."
                )
            if n_q is None:
                n_q = xa.shape[1]
            elif xa.shape[1] != n_q:
                raise ValueError(
                    "All per-attribute query matrices must share the "
                    f"same number of columns (nQ). Got {n_q} and "
                    f"{xa.shape[1]}."
                )
            blocks.append(xa)
        return np.vstack(blocks) if blocks else np.zeros((0, 0))
    xs = np.asarray(x, dtype=np.float64)
    if xs.ndim == 1 and dim == 1:
        xs = xs.reshape(1, -1)
    if xs.ndim != 2 or xs.shape[0] != dim:
        raise ValueError(
            f"Single-matrix query must have {dim} rows (total dim); "
            f"got shape {xs.shape}. For list-form input, wrap the "
            f"per-attribute query matrices in a length-{A} list/tuple."
        )
    return xs


def _ma_eval_normalize(dens: MaetDensity, vals: np.ndarray,
                       normalize: str) -> np.ndarray:
    """Apply the MA normalisation to raw density values.

    Shared by the joint-centres and factored-Möbius paths so both
    produce identical normalised output. Multiplies by the per-attribute
    Gaussian constant (carrying the co-transposition metric determinant)
    and, for ``'pdf'``, divides by the total weight-product mass.
    """
    if normalize == "none":
        return vals
    A = int(dens.n_attrs)
    dim_per = [int(v) for v in np.atleast_1d(dens.dim_per_attr)]
    r_vec = [int(v) for v in np.atleast_1d(dens.r)]
    is_rel = [bool(v) for v in np.atleast_1d(dens.is_rel)]
    sigma = np.atleast_1d(dens.sigma)
    inner_r = _inner_r_vec(dens)
    gauss_const = 1.0
    for a in range(A):
        da = dim_per[a]
        det_m_a = _quadratic_form_det(r_vec[a], inner_r[a], is_rel[a])
        gauss_const *= 1.0 / _gaussian_mass_const(sigma[a], da, det_m_a)
    vals = vals * gauss_const
    if normalize == "pdf":
        sum_w = float(np.sum(dens.w_j))
        if sum_w > 0:
            vals = vals / sum_w
        else:
            warnings.warn(
                "Sum of weight products is zero; cannot normalize to pdf."
            )
    return vals


def _eval_exp_tens_ma(
    dens: MaetDensity,
    x,
    normalize: str = "none",
    *,
    method: str = "auto",
    truncation_sigmas: float | None = None,
    kernel_precision: str | None = None,
    verbose: bool = True,
    prune_zero_weight_events: bool = True,
) -> np.ndarray:
    """Multi-attribute expectation tensor evaluation.

    Routes between the joint-centres path (materialises the joint tuple
    set and sums a Gaussian per joint centre) and the factored Möbius
    evaluator (:func:`mpt._tensor._ma_eval_orbit.eval_ma_orbit`, which
    never materialises joint centres) according to ``method`` and the
    cost model in :func:`mpt._tensor.dispatch._select_ma_eval`. Because
    the MAET density factorises across attributes, the cost model is a
    pure closed-form comparison (no probe): the joint tuple count grows
    as a product across attributes while the factored cost grows as a
    sum, so the factored path wins as soon as more than one attribute
    carries a non-trivial tuple set.

    When ``prune_zero_weight_events=True`` (default), joint perm-side
    tuples whose product weight ``w_j[j] == 0`` are dropped before
    evaluation on the joint-centres path. The joint weight already
    incorporates the per-attribute weight product, so a zero entry
    indicates that at least one attribute has zero weight at that tuple
    --- the tuple contributes zero to the density at every query point,
    so dropping it is exact. Set to ``False`` to bypass (only useful for
    testing the pre-prune work). The cost saving scales with the
    fraction of zero-weight tuples, which can be very large after
    :func:`weight_events` has hard-zeroed factors outside the truncation
    radius.
    """
    # ---- Path selection (cost model, no probe) ----
    # Defensive resolve: every eval entry resolves the truncation width to
    # a finite value, but internal callers can reach here directly, so
    # re-resolve idempotently (finite -> unchanged, inf -> accuracy-floor
    # width, None -> global default). This guarantees the centres kernels
    # below always truncate --- None must never fall through as "no
    # truncation" (i.e. dense summation over every joint tuple).
    from .._defaults import resolve_truncation_sigmas
    truncation_sigmas = resolve_truncation_sigmas(truncation_sigmas)
    from .dispatch import _select_ma_eval
    n_q_hint = 0
    if isinstance(x, (list, tuple)):
        if len(x) and hasattr(x[0], "shape"):
            xa0 = np.asarray(x[0])
            n_q_hint = xa0.shape[-1] if xa0.ndim >= 1 else 1
    else:
        xa = np.asarray(x)
        n_q_hint = xa.shape[-1] if xa.ndim >= 1 else 1
    chosen, routing_reason = _select_ma_eval(dens, n_q_hint, method=method)
    # Dispatch messages are gated by show_hints, not per-call verbose, so
    # users see the routing decision even from internal callers that pass
    # verbose=False (matching the single-multiset path). The MA cost
    # model is probe-free, and the dispatch message reports the
    # decision only, so no estimate accompanies it.
    _maybe_show_dispatch_msg(
        "eval_exp_tens (MAET)", chosen, routing_reason,
    )
    from ._timeest import _maybe_warn_eval_time
    _maybe_warn_eval_time("eval_exp_tens (MAET)", dens, n_q_hint, chosen)
    if chosen == "mobius":
        from ._ma_eval_orbit import eval_ma_orbit
        # The factored evaluator takes the joint query as a single
        # (D, n_q) matrix and splits it internally by dim_per_attr.
        xq = _ma_join_query(dens, x)
        vals = eval_ma_orbit(
            dens, xq, truncation_sigmas=truncation_sigmas,
        )
        return _ma_eval_normalize(dens, vals, normalize)

    # --- Single-multiset centres fast path -------------------------------
    # For the single-multiset corner (A = 1, flat) the general per-
    # attribute _ma_eval_full loop reduces to one attribute, so route
    # through the tight _eval_core / _eval_full kernel instead (one 3-D
    # broadcast, one _compute_Q, one exp, one weighted sum), avoiding the
    # A-loop, attribute-splitting, and per-attribute dispatch overhead.
    # The kernel applies the *resolved* truncation (None -> the global
    # default) with the same floor as _ma_eval_full, so this is value-
    # identical to the general path --- purely a speed choice.
    if is_single_multiset(dens):
        # truncation_sigmas is already resolved to a finite width at the
        # eval entry (inf -> accuracy-floor width per the contract), so
        # pass it straight through to the fast kernel.
        c0 = dens.centres[0]
        wj0 = dens.w_j
        if prune_zero_weight_events and wj0.size:
            keep = wj0 != 0
            if not bool(keep.all()):
                c0 = c0[:, keep]
                wj0 = wj0[keep]
        xs = _split_query_to_attr_list(dens, x)[0]
        if xs.shape[1] == 0:
            return np.zeros(0, dtype=np.float64)
        vals = _eval_core(
            c0, wj0, int(wj0.size), xs, int(xs.shape[1]), c0.shape[0],
            float(dens.sigma[0]), int(dens.r[0]),
            bool(dens.is_rel[0]), bool(dens.is_per[0]), float(dens.period[0]),
            truncation_sigmas=truncation_sigmas,
            wrap=(str(dens.wrap[0]) if hasattr(dens, 'wrap')
                  and dens.wrap is not None else 'full-image'),
        )
        return _ma_eval_normalize(dens, vals, normalize)

    # --- Multi-attribute factored centres path ---------------------------
    # The joint density factors within each event as a product across
    # attributes, so eval = sum_events prod_attributes S_a^(event). Each
    # per-attribute factor is evaluated through the culled kernel, and the
    # joint tuple set (product of per-attribute counts) is never built.
    # Falls back to the joint materialisation below when unsupported.
    factored = _ma_eval_factored(
        dens, x, truncation_sigmas=truncation_sigmas,
        prune_zero_weight_events=prune_zero_weight_events,
    )
    if factored is not None:
        return _ma_eval_normalize(dens, factored, normalize)

    A           = dens.n_attrs
    n_j         = dens.n_j
    dim         = dens.dim
    dim_per     = dens.dim_per_attr
    r_vec       = dens.r
    sigma       = dens.sigma
    is_rel      = dens.is_rel
    is_per      = dens.is_per
    period      = dens.period
    centres     = dens.centres
    w_j         = dens.w_j

    # --- Auto-prune zero-weight joint perm-side tuples ---
    # The MaetDensity build expands per-attribute value combinations into
    # joint perm-side tuples and computes w_j as the product of
    # per-attribute weights. A tuple with w_j == 0 contributes zero at
    # every query point, so dropping it is exact (matches the strict
    # zero convention of the IP-path prune in mobius.maPerAttrInnerMatrix).
    # The rebinding here is local; dens is untouched on disk.
    if prune_zero_weight_events and n_j > 0:
        keep = w_j != 0
        if not bool(keep.all()):
            n_j = int(keep.sum())
            w_j = w_j[keep]
            centres = [c[:, keep] for c in centres]

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

    # --- Core evaluation with memory-aware chunking ---
    # Peak per-chunk memory is dominated by the largest per-attribute
    # (dim_a, nJ, nQc) difference tensor, its square, and the
    # summed/exponentiated intermediate co-resident during chunk eval.
    max_dim_a = int(max(dim_per)) if A > 0 else 1
    bytes_per_col = (2 * max_dim_a + 2) * int(n_j) * 8
    mem_limit = kernel_chunk_bytes_resolved()

    bytes_needed = bytes_per_col * int(n_q)
    inner_r = _inner_r_vec(dens)
    _wrap_dens = getattr(dens, 'wrap', None)
    # Distinct-value tables for the abs-per full-image branch, resolved
    # once: the centres are the same in every chunk.
    if kernel_precision is None:
        from .._defaults import get_default
        _kp = get_default("kernel_precision")
    else:
        _kp = kernel_precision
    _value_tables = _ma_value_tables(
        centres, A, dim_per, is_rel, is_per, inner_r, _wrap_dens,
        np.float32 if _kp == "single" else np.float64, n_q)
    if bytes_needed <= mem_limit:
        vals = _ma_eval_full(
            centres, w_j, n_j, x_list, n_q,
            A, dim_per, r_vec, sigma,
            is_rel, is_per, period,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
            inner_r=inner_r,
            wrap=_wrap_dens,
            value_tables=_value_tables,
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
                A, dim_per, r_vec, sigma,
                is_rel, is_per, period,
                truncation_sigmas=truncation_sigmas,
                kernel_precision=kernel_precision,
                inner_r=inner_r,
                wrap=_wrap_dens,
                value_tables=_value_tables,
            )

    # --- Normalisation (shared with the factored path) ---
    return _ma_eval_normalize(dens, vals, normalize)



def _ma_eval_factored(
    dens, x, *, truncation_sigmas=None, prune_zero_weight_events=True,
):
    """Factored multi-attribute centres evaluation.

    The joint density is a Cartesian product across attributes within
    each event, so its value factors::

        eval(q) = sum_events prod_attributes S_a^(event)(q_a)

    where ``S_a^(event)`` is attribute ``a``'s r-ad Gaussian mixture for
    that event, evaluated at the split query. Each per-attribute factor
    is computed through the culled single-multiset kernel (:func:`_eval_core`)
    for flat attributes, or a dense block-diagonal form for nested ones,
    so the joint tuple set --- whose size is the *product* of the
    per-attribute tuple counts --- is never materialised; the cost is the
    *sum* of the per-attribute counts instead.

    Absent values (NaN in a given event) are handled as zero-weight values
    on a shared enumeration over the ever-valid indices, so events with
    differing valid-value patterns need no special case: a tuple touching a
    value absent in its event carries weight zero and contributes nothing.

    Returns the raw (un-normalised) values ``(n_q,)`` --- the caller
    applies :func:`_ma_eval_normalize`, identically to the joint path ---
    or ``None`` when the shape is outside this path's support (any
    ``r_a == 1`` attribute, whose event-dependent equal-value collapse
    breaks the shared enumeration), signalling a fall-back to the joint
    :func:`_ma_eval_full` route.
    """
    from .._defaults import resolve_truncation_sigmas
    from .build import _enum_flat_attr, _nested_enum_indices

    A = int(dens.n_attrs)
    N = int(dens.n)
    r_vec = [int(v) for v in np.atleast_1d(dens.r)]
    if any(r_a < 2 for r_a in r_vec):
        return None  # r = 1 collapse is event-dependent; use the joint path
    if getattr(dens, "kernel_cov", None) is not None:
        return None  # matrix-sigma covariance: scalar per-attribute form
        #              does not apply; use the joint path

    x_list = _split_query_to_attr_list(dens, x)
    n_q = x_list[0].shape[1] if x_list else 0
    if n_q == 0:
        return np.zeros(0, dtype=np.float64)

    P = [np.asarray(p, dtype=np.float64) for p in dens.p_attr]
    W = [np.asarray(w, dtype=np.float64) for w in dens.w]
    is_rel = [bool(v) for v in np.atleast_1d(dens.is_rel)]
    is_per = [bool(v) for v in np.atleast_1d(dens.is_per)]
    period = [float(v) for v in np.atleast_1d(dens.period)]
    sigma = [float(v) for v in np.atleast_1d(dens.sigma)]
    is_sym = [bool(v) for v in np.atleast_1d(dens.is_sym)]
    inner_r = _inner_r_vec(dens)
    nested = dens.nested
    ts = resolve_truncation_sigmas(truncation_sigmas)

    # Per-attribute tuple-index structure, enumerated once over the
    # ever-valid indices (non-NaN in at least one event). The index
    # pattern is event-invariant; only the per-event values and weights
    # change, so this is built a single time per attribute.
    perm = []
    for a in range(A):
        ever_valid = np.nonzero((~np.isnan(P[a])).any(axis=1))[0].astype(np.intp)
        if ever_valid.size < r_vec[a]:
            return None  # too few values for a full tuple; joint path errors cleanly
        spec = nested[a]
        if spec is not None:
            tags = np.asarray(spec["tags"])[ever_valid]
            pm, _ = _nested_enum_indices(
                ever_valid, tags,
                np.asarray(spec["r"]).ravel(), np.asarray(spec["sym"]).ravel(),
            )
        else:
            pm, _, _, _ = _enum_flat_attr(
                np.zeros(P[a].shape[0]), ever_valid, r_vec[a], is_sym[a],
                np.ones(P[a].shape[0]),
            )
        perm.append(pm)

    total = np.zeros(n_q, dtype=np.float64)
    for n in range(N):
        prod = np.ones(n_q, dtype=np.float64)
        for a in range(A):
            pm = perm[a]
            p_col = P[a][:, n]
            w_col = W[a][:, n]
            absent = np.isnan(p_col)
            # Finite placeholder keeps the kernel finite; zero weight
            # nulls any tuple touching an absent value.
            p_fill = np.where(absent, 0.0, p_col)
            w_fill = np.where(absent | np.isnan(w_col), 0.0, w_col)
            u = p_fill[pm]                      # (r_a, M)
            w_tuple = np.prod(w_fill[pm], axis=0)   # (M,)
            r_in = int(inner_r[a])
            if r_in > 0:
                # Nested: reduce each inner block by its own first value,
                # then a dense block-diagonal quadratic form (nested M is
                # small, so the cull is not needed here).
                r_out = u.shape[0] // r_in
                c = np.vstack([
                    u[b * r_in + 1:(b + 1) * r_in, :] - u[b * r_in:b * r_in + 1, :]
                    for b in range(r_out)
                ]) if r_in > 1 else np.empty((0, u.shape[1]))
                d = c[:, :, None] - x_list[a][:, None, :]
                q = _compute_Q_inner_blocks(
                    d, r_in, is_per[a], period[a], reduced=True,
                ) / (2.0 * sigma[a] ** 2)
                s_a = (w_tuple[:, None] * np.exp(-q)).sum(axis=0)
            else:
                c = (u[1:, :] - u[:1, :]) if is_rel[a] else u
                s_a = _eval_core(
                    c, w_tuple, int(w_tuple.size), x_list[a], n_q,
                    int(c.shape[0]), sigma[a], r_vec[a],
                    is_rel[a], is_per[a], period[a],
                    truncation_sigmas=ts,
                )
            prod *= s_a
        total += prod
    return total


def _tuple_values_repeat(c_a, n_q, min_queries=100, min_entries=256,
                         min_saving=4):
    """Does tabulating the kernel on the distinct values pay?

    Every coordinate of an r-tuple is a value of the same multiset, so
    the distinct arguments number the multiset size rather than the
    tuple count, and the saving per query grows with the tuple count.
    Against that stands the sort that finds the distinct values, which
    is paid once per call however many queries follow. The trade is
    settled by ``n_q``: measured on the calibration grid, the sort
    costs 0.17 ms at 4512 centre entries rising to 59.8 ms at 1020096,
    against a per-query saving of 0.005 ms to 0.77 ms over the same
    range, putting the break-even between 35 and 78 queries. The
    default ``min_queries`` of 100 sits above that across the range.

    The query-count and size tests precede the sort: the predicate must
    not cost what it is deciding whether to spend. An arbitrary centre
    array with no repeats fails ``min_saving`` and takes the direct
    route.
    """
    n_entries = int(c_a.shape[0]) * int(c_a.shape[1])
    if int(n_q) < min_queries or n_entries < min_entries:
        return False
    n_distinct = int(np.unique(c_a).size)
    return n_distinct * min_saving <= n_entries


def _distinct_value_table(c_a, n_q):
    """``(values, inverse)`` for the tabulated kernel, or None.

    The centres do not vary across query chunks, so the sort that finds
    their distinct values is resolved once by the caller and passed into
    each chunk rather than repeated per chunk.
    """
    if not _tuple_values_repeat(c_a, n_q):
        return None
    vals, inv = np.unique(c_a, return_inverse=True)
    return vals, inv.reshape(c_a.shape)


def _ma_value_tables(centres, A, dim_per, is_rel, is_per, inner_r, wrap,
                     dtype, n_q):
    """Per-attribute distinct-value tables for the abs-per branch.

    Entry ``a`` is ``None`` unless that attribute takes the abs-per
    full-image branch and its centres repeat values. Built on the cast
    array, so the tabulated arguments are the ones the branch would
    have formed itself.
    """
    tables = [None] * int(A)
    for a in range(int(A)):
        if int(dim_per[a]) == 0:
            continue
        r_in = 0 if inner_r is None else int(inner_r[a])
        wrap_a = 'full-image'
        if wrap is not None and a < len(wrap):
            wrap_a = str(wrap[a])
        if (r_in == 0 and is_per[a] and not is_rel[a]
                and wrap_a == 'full-image'):
            tables[a] = _distinct_value_table(
                np.asarray(centres[a]).astype(dtype, copy=False), n_q)
    return tables


def _ma_eval_full(
    centres, w_j, n_j, x_list, n_qc,
    A, dim_per, r_vec, sigma,
    is_rel, is_per, period,
    *,
    truncation_sigmas=None,
    kernel_precision=None,
    inner_r=None,
    wrap=None,
    value_tables=None,
):
    """Single-chunk MAET evaluation.

    Accumulates the summed-quadratic exponent across attributes, then
    exponentiates once and does the weighted sum against ``w_j``.
    Geometry (``sigma``, ``is_rel``, ``is_per``, ``period``) is
    per-attribute, indexed directly by ``a``.

    Untruncated-double bypass: when the resolved ``truncation_sigmas``
    is Inf and the resolved ``kernel_precision`` is 'double', runs the
    inline accumulation inline with no cast machinery and no
    post-filter branching. This
    keeps default-mode calls at inline cost; the feature kwargs
    only impose their cost when explicitly requested.
    """
    # ---- Resolve precision from defaults ----
    # ``truncation_sigmas`` is already resolved to a finite width at
    # the :func:`eval_exp_tens` entry (None -> global default; inf ->
    # the accuracy-floor width per the truncation contract), so it is
    # never None or non-finite on any reachable call and truncation
    # always applies. The former "default-mode bypass" that ran an
    # untruncated inline double accumulation for the ``inf`` case is
    # therefore removed --- it would silently produce untruncated
    # values in violation of the contract.
    if kernel_precision is None:
        from .._defaults import get_default
        kernel_precision = get_default("kernel_precision")

    # ---- Precision casting and post-filter truncation. ----
    dtype = np.float32 if kernel_precision == "single" else np.float64

    q_total = np.zeros((int(n_j), int(n_qc)), dtype=dtype)
    # Abs-per full-image factor accumulator: product over abs-per
    # attributes of prod_slots theta(d). Stays 1 when every abs-per
    # attribute has L=0 (the small-sigma/P regime), preserving the
    # exact single-Gaussian arithmetic in that case.
    abs_per_factor = None  # allocate lazily to keep the small-L path alloc-free

    for a in range(A):
        da = int(dim_per[a])
        if da == 0:
            continue

        c_a = centres[a].astype(dtype, copy=False)
        x_a = x_list[a].astype(dtype, copy=False)

        r_in = 0 if inner_r is None else int(inner_r[a])
        wrap_a = 'full-image'
        if wrap is not None and a < len(wrap):
            wrap_a = str(wrap[a])

        # Abs-per full-image factorises over tuple positions, and each
        # position draws from the same small set of values: an r-tuple's
        # coordinates are values of the attribute's multiset, so the
        # distinct arguments number K per event rather than one per
        # tuple. Evaluating the wrapped Gaussian once per distinct value
        # and reading the tuple layout off that table gives the same
        # floating-point arguments in the same order, so the result is
        # identical, not merely equal to tolerance. It also avoids the
        # (dim, n_j, n_q) difference tensor entirely.
        table_a = None if value_tables is None else value_tables[a]
        if (r_in == 0 and is_per[a] and not is_rel[a]
                and wrap_a == 'full-image' and table_a is not None):
            from .._wrapped_kernel import wrapped_gaussian_1d
            vals, inv = table_a
            factor_a = None
            for k in range(da):
                table_k = wrapped_gaussian_1d(
                    vals[:, None] - x_a[k][None, :],
                    float(sigma[a]), float(period[a]),
                    truncation_sigmas, exponent_denominator=2,
                )
                theta_k = table_k[inv[k], :]
                factor_a = (theta_k if factor_a is None
                            else factor_a * theta_k)
            factor_a = factor_a.astype(dtype, copy=False)
            abs_per_factor = (factor_a if abs_per_factor is None
                              else abs_per_factor * factor_a)
            continue

        d_a = c_a[:, :, None] - x_a[:, None, :]

        if r_in > 0:
            # Inner [rel] unit: block-diagonal metric over event blocks
            # (pairwise wrap applied inside _compute_Q).
            q_a = _compute_Q_inner_blocks(
                d_a, r_in, bool(is_per[a]), float(period[a]), reduced=True)
            q_total = q_total + q_a / (2 * dtype(sigma[a]) ** 2)
            continue

        # Outer wrap only needed for abs+per. For rel+per, _compute_Q
        # applies the pairwise wrap inside (Eq 6).
        if is_per[a] and not is_rel[a]:
            pg = dtype(period[a])
            if wrap_a == 'full-image':
                # Abs-per full-image via the shared wrapped-Gaussian
                # helper (image-sum or Fourier, whichever is cheaper).
                # Density-kernel convention: exponent_denominator=2.
                from .._wrapped_kernel import wrapped_gaussian_1d
                theta_per_position = wrapped_gaussian_1d(
                    d_a, float(sigma[a]), float(period[a]),
                    truncation_sigmas, exponent_denominator=2,
                )
                factor_a = theta_per_position.prod(axis=0).astype(dtype,
                                                              copy=False)
                abs_per_factor = (factor_a if abs_per_factor is None
                                  else abs_per_factor * factor_a)
                # Contribution now in abs_per_factor; skip q_total.
                continue
            # Single-image opt-in: pre-reduce and fall through to
            # the q_total accumulator.
            d_a = d_a - pg * np.floor(d_a / pg + 0.5)

        # _compute_Q matches d_a.dtype, preserving the single-precision
        # accumulator when kernel_precision='single'.
        q_a = _compute_Q(d_a, int(r_vec[a]), bool(is_rel[a]),
                         bool(is_per[a]), float(period[a]),
                         reduced=bool(is_rel[a]))

        q_total = q_total + q_a / (2 * dtype(sigma[a]) ** 2)

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

    if abs_per_factor is not None:
        e = e * abs_per_factor.astype(dtype, copy=False)

    result = w_j.astype(dtype, copy=False) @ e
    return result.astype(np.float64, copy=False)



def _neighbour_offsets(dim):
    """3**dim integer offset columns in {-1, 0, 1}**dim, shape (dim, 3**dim)."""
    return np.indices((3,) * dim).reshape(dim, -1) - 1


def _truncated_kernel_sum_culled(centres, w_j, x_q, sigma, is_rel, r, k_sigma):
    """Bucket-grid spatial cull for the non-periodic single-multiset centres path.

    Twin of MATLAB ``internal.gaussianKernelSum``'s ``localTruncatedKernelSum``.
    The kernel is negligible beyond a Q-ball of squared radius
    ``(k_sigma * sigma)**2``, so rather than evaluate every (tuple, query) pair,
    whiten the coordinates so that ball is a Euclidean sphere (rel:
    ``Q(D) = |T D|**2`` with ``M = I - 11'/r``; abs: identity), hash the tuple
    centres into ``k_sigma * sigma`` buckets, and for each query evaluate only the
    centres in its ``3**dim`` neighbouring buckets. Q is computed on the original
    coordinates, so the result is identical to the dense path up to the truncation
    floor --- the culled pairs contribute below the floor by construction.
    """
    dim, n_j = centres.shape
    n_q = x_q.shape[1]
    v = np.zeros(n_q)
    if n_j == 0 or n_q == 0:
        return v
    inv_2s2 = 1.0 / (2.0 * sigma * sigma)
    threshold2 = (k_sigma * sigma) ** 2

    # Whitening transform (used only to make the Q-ball a Euclidean sphere for
    # bucketing; the returned kernel is computed on the original coordinates).
    if is_rel:
        e = np.ones(dim)
        metric = np.eye(dim) - (1.0 / r) * np.outer(e, e)
        lams, u = np.linalg.eigh(metric)
        t_white = (u * np.sqrt(np.maximum(lams, 0.0))).T
    else:
        t_white = np.eye(dim)
    ct = t_white @ centres
    xt = t_white @ x_q

    # Bucket grid over the transformed centres' bounding box.
    bucket_size = k_sigma * sigma
    c_min = ct.min(axis=1, keepdims=True)
    c_max = ct.max(axis=1, keepdims=True)
    n_buckets = np.maximum(
        1, np.ceil((c_max - c_min).ravel() / bucket_size).astype(np.int64) + 1
    )
    buck_c = np.clip(
        np.floor((ct - c_min) / bucket_size).astype(np.int64), 0, n_buckets[:, None] - 1
    )
    lin_c = np.ravel_multi_index(tuple(buck_c), tuple(n_buckets))

    # Group centres by bucket: sort, find runs, and look buckets up by
    # membership in the sorted list of occupied ones.
    #
    # The occupied list is at most n_j long, where a dense table over the
    # whole lattice costs prod(n_buckets) however few centres there are,
    # and that grows as (spread / (k_sigma*sigma))**dim: at r = 4 over a
    # 9600-cent span with sigma = 5 it is 3.3e9 cells for 11880 centres.
    # A dense table therefore had to be guarded, and the guard declined
    # to cull in exactly the regime where culling saves most --- fine
    # kernels over a wide range --- falling back to the dense kernel at
    # around thirty times the cost. Indexing by occupancy needs no such
    # guard. Twin of the run table in _kernel._truncated_kernel_sum and
    # of internal.gaussianKernelSum on the MATLAB side.
    perm = np.argsort(lin_c, kind="stable")
    sorted_lin = lin_c[perm]
    bounds = np.concatenate(([0], np.nonzero(np.diff(sorted_lin))[0] + 1))
    run_start = bounds
    run_end = np.concatenate((bounds[1:], [n_j]))          # exclusive
    run_lin = sorted_lin[bounds]                           # ascending

    buck_x = np.clip(
        np.floor((xt - c_min) / bucket_size).astype(np.int64), 0, n_buckets[:, None] - 1
    )
    offsets = _neighbour_offsets(dim)
    n_off = offsets.shape[1]

    # Query-chunk loop bounds the transient (query, centre) pair workspace;
    # per-query pair count ~ n_off * (n_j / total buckets).
    per_query = max(1.0, n_off * n_j / max(float(np.prod(n_buckets)), 1.0))
    chunk = max(1, int(kernel_chunk_bytes_resolved() / ((2 * dim + 4) * 8 * per_query)))
    for c0 in range(0, n_q, chunk):
        c1 = min(c0 + chunk, n_q)
        n_qc = c1 - c0

        # Expand each query to its 3**dim neighbour buckets.
        nb = (buck_x[:, c0:c1][:, :, None] + offsets[:, None, :]).reshape(dim, n_qc * n_off)
        q_of = np.repeat(np.arange(n_qc), n_off)
        in_bounds = np.all((nb >= 0) & (nb < n_buckets[:, None]), axis=0)
        if not in_bounds.any():
            continue
        nb = nb[:, in_bounds]
        q_of = q_of[in_bounds]

        nb_lin = np.ravel_multi_index(tuple(nb), tuple(n_buckets))
        pos = np.searchsorted(run_lin, nb_lin)
        pos = np.minimum(pos, run_lin.size - 1)
        has_run = run_lin[pos] == nb_lin
        if not has_run.any():
            continue
        run_idx = pos[has_run]
        q_of = q_of[has_run]

        # Ragged-expand each (query, run) into (query, centre) pairs via cumsum.
        lens = run_end[run_idx] - run_start[run_idx]
        total = int(lens.sum())
        if total == 0:
            continue
        member = np.repeat(np.arange(lens.size), lens)
        start_pos = np.cumsum(lens) - lens
        centre = perm[run_start[run_idx[member]] + (np.arange(total) - start_pos[member])]
        query = q_of[member]

        dq = centres[:, centre] - x_q[:, c0 + query]
        if is_rel:
            q_form = (dq * dq).sum(axis=0) - dq.sum(axis=0) ** 2 / r
        else:
            q_form = (dq * dq).sum(axis=0)
        keep = q_form <= threshold2
        if not keep.any():
            continue
        np.add.at(
            v, c0 + query[keep], w_j[centre[keep]] * np.exp(-q_form[keep] * inv_2s2)
        )
    return v


def _eval_core(
    centres, w_j, n_j, x, n_q, dim, sigma, r, is_rel, is_per, period,
    *, truncation_sigmas=None, wrap='full-image',
):
    """Evaluate with automatic memory-aware chunking (single-multiset path)."""
    # Non-periodic + finite truncation: bucket-grid spatial cull (mirrors MATLAB
    # internal.gaussianKernelSum). Only tuples within the Q-ball of each query
    # are evaluated, avoiding the dense n_j x n_q kernel; value-identical to the
    # dense path up to the truncation floor. Periodic mode stays dense (the
    # pairwise wrap is not a tail-truncatable ball).
    if (
        truncation_sigmas is not None
        and np.isfinite(truncation_sigmas)
        and truncation_sigmas > 0
        and not is_per
    ):
        return _truncated_kernel_sum_culled(
            centres, w_j, x, sigma, is_rel, r, float(truncation_sigmas)
        )

    # Peak per-chunk transient ~ (2*dim + 2) × n_j × n_q × 8 (broadcast
    # difference, its square, and the summed/exponentiated intermediate
    # are briefly co-resident).
    bytes_needed = (2 * dim + 2) * int(n_j) * int(n_q) * 8
    mem_limit = kernel_chunk_bytes_resolved()

    # Distinct-value table for the abs-per full-image branch, resolved
    # once: the centres are the same in every chunk.
    value_table = None
    if is_per and not is_rel and wrap == 'full-image':
        value_table = _distinct_value_table(centres, n_q)

    if bytes_needed <= mem_limit:
        return _eval_full(centres, w_j, n_j, x, n_q, dim, sigma, r, is_rel, is_per, period,
                          truncation_sigmas=truncation_sigmas, wrap=wrap,
                          value_table=value_table)

    chunk_size = max(1, int(mem_limit / ((2 * dim + 2) * int(n_j) * 8)))
    vals = np.zeros(n_q)
    for c_start in range(0, n_q, chunk_size):
        c_end = min(c_start + chunk_size, n_q)
        idx = slice(c_start, c_end)
        n_qc = c_end - c_start
        vals[idx] = _eval_full(
            centres, w_j, n_j, x[:, idx], n_qc, dim, sigma, r, is_rel, is_per, period,
            truncation_sigmas=truncation_sigmas, wrap=wrap,
            value_table=value_table,
        )
    return vals



def _eval_full(centres, w_j, n_j, x_q, n_qc, dim, sigma, r, is_rel, is_per, period,
               *, truncation_sigmas=None, wrap='full-image',
               value_table=None):
    """Fully vectorized single-multiset density evaluation.

    Uses the pairwise-wrap form (Eq 6 of the preprint) for
    periodic+relative, matching cosSimExpTens. The algebraic form
    used by v2.0 / v2.1 in this mode silently differed from the
    inner-product convention; v2.X corrects it.

    ``truncation_sigmas`` applies the same kernel-value floor as the
    general :func:`_ma_eval_full` path (the kernel is negligible where
    ``Q/(2 sigma^2) > k^2/2``), so routing a single-multiset density
    through this fast kernel is value-identical to the general path,
    not merely close --- the two differ only in per-attribute loop
    overhead, never in the truncation policy.

    ``wrap`` selects the abs-per measure (v3+): ``'full-image'``
    (default) sums the kernel over periodic images; ``'single-image'``
    evaluates the nearest image only. Ignored for rel-per and
    non-periodic modes.
    """
    # Outer wrap only needed for abs+per (see _compute_Q docstring).
    if is_per and not is_rel:
        if wrap == 'full-image':
            # Full-image via the shared wrapped-Gaussian helper
            # (image-sum or Fourier, whichever is cheaper).
            from .._wrapped_kernel import wrapped_gaussian_1d
            if value_table is not None:
                # Every coordinate of an r-tuple is a value of the same
                # multiset, so the distinct arguments number K rather
                # than one per tuple. Evaluating the wrapped Gaussian
                # once per distinct value and reading the tuple layout
                # off that table presents the same floating-point
                # arguments in the same order, so the result is
                # identical rather than equal to a tolerance.
                vals, inv = value_table
                E = None
                for k in range(centres.shape[0]):
                    table_k = wrapped_gaussian_1d(
                        vals[:, None] - x_q[k][None, :],
                        float(sigma), float(period), truncation_sigmas,
                        exponent_denominator=2,
                    )
                    theta_k = table_k[inv[k], :]
                    E = theta_k if E is None else E * theta_k
            else:
                D = centres[:, :, None] - x_q[:, None, :]
                theta_per_position = wrapped_gaussian_1d(
                    D, float(sigma), float(period), truncation_sigmas,
                    exponent_denominator=2,
                )
                E = theta_per_position.prod(axis=0)  # (nJ, nQc)
            use_truncation = (
                truncation_sigmas is not None
                and np.isfinite(truncation_sigmas)
                and truncation_sigmas > 0
            )
            if use_truncation:
                from .._defaults import truncation_floor
                floor = truncation_floor(truncation_sigmas)
                E = np.where(E > floor, E, 0.0)
            return w_j @ E
        # Single-image: reduce to nearest image and fall through to
        # the q_total accumulator.
        D = centres[:, :, None] - x_q[:, None, :]
        D = D - period * np.floor(D / period + 0.5)
    else:
        D = centres[:, :, None] - x_q[:, None, :]

    Q = _compute_Q(D, r, is_rel, is_per, period, reduced=is_rel)

    q_total = Q / (2 * sigma**2)
    use_truncation = (
        truncation_sigmas is not None
        and np.isfinite(truncation_sigmas)
        and truncation_sigmas > 0
    )
    if use_truncation:
        # Same floor as _ma_eval_full: exp(-q_total) is negligible when
        # q_total > k^2 / 2.
        q_threshold = float(truncation_sigmas) ** 2 / 2.0
        E = np.where(q_total <= q_threshold, np.exp(-q_total), 0.0)
    else:
        E = np.exp(-q_total)

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
        p, w, sigma, r, is_rel, is_per, period, x, normalize=normalize,
        verbose=verbose,
    )