"""Cosine similarity: the cos_sim path.

Public entry points:

* :func:`cos_sim_exp_tens` --- compute cosine similarity between two
  expectation tensor densities (scalar, list-list, or batched-raw),
  with dispatch over Bulger / Möbius / direct methods.
* :func:`batch_cos_sim_exp_tens` --- the fully batched path with
  canonical-form dedup.
* :func:`cos_sim_exp_tens_raw` --- deprecated shim; raw-array
  signature is now accepted by :func:`cos_sim_exp_tens` directly.

The bulk of the file is the per-method implementations (Bulger's
method on centres, the Möbius alternating-sum, direct enumeration,
multi-attribute variants thereof) and the inner-product cores
(:func:`_ip_core`, :func:`_ip_full`, :func:`_ip_core_ma`, etc.).

The cosine path reaches into :mod:`._tensor.dispatch` for path
selection and into :mod:`._tensor.canonical` for the batched-mode
dedup keys.

See USER_GUIDE §4 ("Method selection") and :doc:`/ARCHITECTURE` §4
("Dispatcher pattern") for the conceptual description.
"""
from __future__ import annotations

import warnings
from itertools import permutations
from math import factorial

import numpy as np
from scipy.special import comb as _comb

from .._defaults import _maybe_show_dispatch_msg, _with_dispatch_scope
from .._kernel import gaussian_kernel_sum
from .._utils import (
    estimate_comp_time,
    kernel_chunk_bytes_resolved,
    maybe_print_batched_estimate,
    progress_stride,
    with_kernel_chunk_bytes_pin,
)
from ..spectra import add_spectra

from .build import _looks_like_multi_attr, build_exp_tens
from .canonical import _pair_canonical_key
from .density import (
    ExpTensDensity,
    MaetDensity,
    WindowedMaetDensity,
    _nchoosek_indices,
)
from .dispatch import (
    _compute_Q,
    _compute_Q_inner_blocks,
    _inner_r_vec,
    _normalize_density_input,
    _orbit_ips_look_corrupted,
    _resolve_list_list_mode,
    _select_and_estimate_sa_ip,
    _select_ma_inner_product_method,
    # Orbit-table policy constants used by the Möbius-method router.
    _ORBIT_K_MINUS_R_MIN,
    _ORBIT_R_MAX_SHIPPED,
    _ORBIT_SIGMA_OVER_P_THRESHOLD,
)


# -------------------------------------------------------------------
#  Normalisation helpers (shared by cos_sim_exp_tens and
#  windowed_similarity, which expose the same ``normalize`` keyword
#  with the same set of values).
# -------------------------------------------------------------------

#: The canonical value set for the ``normalize`` keyword. ``'cosine'`` is
#: the strict shape-only cosine similarity, with denominator equal to
#: the geometric mean of the two operands' self inner products.
#: ``'oneSidedDenom'`` divides only by the *second* operand's self inner
#: product, yielding a magnitude-aware reading that takes the value 1
#: on a perfect self-match at full coverage and may exceed 1 when the
#: first operand carries more matching mass than the second.
_NORMALIZE_VALUES = ("cosine", "oneSidedDenom")


def _canonical_normalize(normalize: str) -> str:
    """Return the canonical form of a ``normalize`` keyword argument.

    Accepts British (``'normalise'`` flavour) and American
    (``'normalize'`` flavour) inputs alike, and accepts the value
    ``'oneSidedDenom'`` in any case. Raises :class:`ValueError` for
    anything outside the canonical set.
    """
    if not isinstance(normalize, str):
        raise ValueError(
            f"normalize must be a string, got {type(normalize).__name__}."
        )
    s = normalize.strip()
    if s.lower() == "cosine":
        return "cosine"
    if s.lower() == "onesideddenom":
        return "oneSidedDenom"
    raise ValueError(
        f"normalize must be one of {_NORMALIZE_VALUES!r}; got {normalize!r}."
    )


def _finalise_normalisation(
    ip_xy: float, ip_xx: float, ip_yy: float, normalize: str,
) -> float:
    """Combine numerator and self inner products into the final value.

    ``ip_xy`` is :math:`\\langle X, Y \\rangle`; ``ip_xx`` and ``ip_yy``
    are the two operands' self inner products. With ``normalize`` set
    to ``'cosine'`` the denominator is :math:`\\sqrt{ip_{xx} \\cdot ip_{yy}}`;
    with ``'oneSidedDenom'`` the denominator is :math:`ip_{yy}` alone.
    Either denominator equal to zero returns ``NaN``.
    """
    if normalize == "cosine":
        denom = float(np.sqrt(max(ip_xx * ip_yy, 0.0)))
    elif normalize == "oneSidedDenom":
        denom = float(ip_yy)
    else:
        # Already canonicalised by callers, but defensive.
        raise ValueError(
            f"normalize must be one of {_NORMALIZE_VALUES!r}; got {normalize!r}."
        )
    if denom == 0:
        return float("nan")
    return float(ip_xy / denom)




# -------------------------------------------------------------------
#  cos_sim_exp_tens
# -------------------------------------------------------------------


@_with_dispatch_scope
@with_kernel_chunk_bytes_pin
def cos_sim_exp_tens(*args,
                     mode: str = "auto",
                     dedup: bool = True,
                     spectrum=None,
                     precision: int | None = None,
                     method: str = "auto",
                     normalize: str | None = None,
                     normalise: str | None = None,
                     cancellation_threshold: float = 1e-12,
                     truncation_sigmas: float | None = None,
                     kernel_precision: str | None = None,
                     verbose: bool = True) -> float | np.ndarray:
    """Cosine similarity of two expectation tensor densities.

    Unified entry point. Accepts four input forms, dispatched on the
    type of the first argument:

    **Pre-built density input** (plus polymorphic lists):

    - ``cos_sim_exp_tens(dens_x, dens_y)`` — scalar.
    - ``cos_sim_exp_tens(dens_x, [d1, d2, …])`` — broadcast, returns
      ``(N,)``.
    - ``cos_sim_exp_tens([a1, a2, …], [b1, b2, …])`` — list-vs-list
      with ``mode='pairwise'`` (default ``'auto'``, resolves to
      pairwise for equal lengths) returning ``(M,)``, or
      ``mode='cartesian'`` returning ``(M, N)``.

    **Raw single-attribute scalar input**:

    - ``cos_sim_exp_tens(p1, w1, p2, w2, sigma, r, is_rel, is_per, period)``
      where ``p1`` and ``p2`` are 1-D arrays of pitches, ``w1``,
      ``w2`` are matching 1-D weight arrays (or ``None`` for uniform).
      Returns scalar.

    **Raw single-attribute batched input** (replaces ``batch_cos_sim_exp_tens``):

    - ``cos_sim_exp_tens(P1, W1, P2, W2, sigma, r, is_rel, is_per, period)``
      where at least one of ``P1``, ``P2`` is a 2-D ``(M, K)`` matrix
      (rows are chords; NaN-padded for variable cardinality), ``W1``,
      ``W2`` likewise (or ``None`` for uniform). Returns ``(M,)``. If
      only one operand is a matrix and the other is a 1-D vector of
      length ``K``, the vector is broadcast across the matrix's ``M``
      rows.

    **Raw multi-attribute scalar input**:

    - ``cos_sim_exp_tens(p_attr1, w1, p_attr2, w2, sigma_vec, r_vec,
      is_rel_vec, is_per_vec, period_vec)`` where ``p_attr*`` are
      lists of per-attribute matrices. Returns scalar.

    **Raw multi-attribute scalar-vs-list (sweep)**:

    - ``cos_sim_exp_tens(p_attr_ref, w_ref, p_attr_list, w_shared,
      sigma_vec, r_vec, is_rel_vec, is_per_vec, period_vec)``
      where exactly one of the two ``p_attr`` arguments is a list of
      ``p_attr`` blocks (a list of lists; e.g. the matrix-form output of
      :func:`translate_attributes`) and the other is a single ``p_attr``.
      Build is internalised: the scalar operand is built once, the
      list operand once per entry. Weights for the list side are
      shared across every entry — a single ``w`` value, not a list of
      weights. Returns an ``ndarray`` of length M. The output index
      matches the order of entries in the list operand.

    Parameters
    ----------
    *args
        Positional arguments. Length depends on the input form:
        2 for density modes; 9 for raw SA modes; 10 for raw MA mode.
    mode : {'auto', 'pairwise', 'cartesian'}, default 'auto'
        For density list-vs-list. Ignored in scalar and broadcast cases.
    dedup : bool, default True
        Apply canonical-form deduplication. Currently supported for
        single-attribute pairs only; pairs involving ``MaetDensity``
        bypass dedup transparently. (``WindowedMaetDensity`` operands
        are rejected at the top of the function — use
        :func:`windowed_similarity` instead.)
    spectrum : list/tuple, optional
        Per-row spectral augmentation parameters passed to
        :func:`mpt.spectra.add_spectra`. Only valid in raw SA modes
        (scalar or batched).
    precision : int, optional
        Round canonical pitch and weight values to this many decimal
        places, to absorb FP noise when deduplicating. Only valid in
        raw SA batched mode.
    method : {'auto', 'bulger', 'mobius', 'direct'}, default 'auto'
        Inner-product method; threaded through to the per-pair SA/MA
        core. ``'auto'`` lets the dispatcher pick between Bulger's
        method (the partition-pair decomposition; small r and small K)
        and the Möbius method (large r or large K).
        ``'bulger'`` forces Bulger's method; ``'mobius'`` forces the
        Möbius method; ``'direct'`` forces direct ordered-tuple
        enumeration.
    normalize : {'cosine', 'oneSidedDenom'}, default 'cosine'
        Selects the denominator applied to the inner product
        :math:`\\langle X, Y \\rangle`. ``'cosine'`` (default) gives the
        strict shape-only cosine similarity, dividing by the geometric
        mean :math:`\\sqrt{\\langle X, X \\rangle \\, \\langle Y, Y \\rangle}`;
        the result is bounded in :math:`[-1, 1]` and is invariant to a
        positive scalar on either operand. ``'oneSidedDenom'`` divides
        by the second operand's self inner product
        :math:`\\langle Y, Y \\rangle` alone, yielding a magnitude-aware
        reading that takes the value 1 on a self-match (``X == Y``)
        and is sensitive to scalar reweightings of ``X``. The British
        spelling ``'normalise'`` is also accepted as an alias for the
        keyword name, and matching is case-insensitive on the value.
    cancellation_threshold : float, default 1e-12
        When the Möbius method is selected and ``|<A,B>|`` falls below
        this fraction of ``sqrt(<A,A><B,B>)``, fall back to Bulger's
        method to avoid catastrophic Möbius alternating-sum
        cancellation.
    verbose : bool, default True
        Print progress.

    Returns
    -------
    float or np.ndarray
        Scalar in scalar-vs-scalar density mode, raw SA scalar mode, and
        raw MA scalar mode. ``ndarray`` in all batched/list modes.

    Notes
    -----
    The Möbius method  is exact to floating-point
    precision when every per-attribute ``K_a`` satisfies
    ``K_a >= r_a + 2`` and σ is not catastrophically small relative to
    the period P. The dispatcher enforces these conditions structurally
    — it refuses the Möbius method and routes to Bulger's method when
    ``K_a < r_a + 2``, when ``σ/P > 0.03`` in periodic-relative mode,
    or when the σ → 0 fallback triggers. Pass ``method='bulger'`` to
    bypass the Möbius method entirely.

    See Also
    --------
    build_exp_tens : explicit density construction.
    eval_exp_tens : evaluate a density at query points.
    cos_sim_exp_tens_raw : deprecated; superseded by raw input mode here.
    batch_cos_sim_exp_tens : deprecated; superseded by raw SA batched input here.

    References
    ----------
    Originally by David Bulger, Macquarie University (2016).
    Adapted for the Music Perception Toolbox v2 by Andrew J. Milne.

    """
    if len(args) < 2:
        raise TypeError(
            "cos_sim_exp_tens requires at least 2 positional arguments."
        )

    # Accept ``normalize`` (canonical) or ``normalise`` (British alias).
    if normalize is not None and normalise is not None:
        raise TypeError(
            "Pass either 'normalize' or 'normalise', not both."
        )
    normalize = _canonical_normalize(
        normalize if normalize is not None
        else (normalise if normalise is not None else "cosine")
    )

    a = args[0]

    # ------------------------------------------------------------------
    # Detect density-input intent based on the first argument.
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
        if len(args) != 2:
            raise TypeError(
                f"Density input mode expects 2 positional arguments "
                f"(dens_x, dens_y); got {len(args)}."
            )
        if spectrum is not None:
            raise TypeError(
                "'spectrum' kwarg is only valid in raw SA input mode."
            )
        if precision is not None:
            raise TypeError(
                "'precision' kwarg is only valid in raw SA batched input mode."
            )
        return _cos_sim_density_path(
            args[0], args[1],
            mode=mode, dedup=dedup,
            method=method,
            normalize=normalize,
            cancellation_threshold=cancellation_threshold,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Raw multi-attribute dispatch (list of per-attribute arrays, or a
    # list of such lists for the sweep / broadcast use case).
    # ------------------------------------------------------------------
    if _looks_like_multi_attr(a):
        if spectrum is not None:
            raise TypeError(
                "'spectrum' kwarg is only supported in raw single-attribute "
                "input mode."
            )
        if precision is not None:
            raise TypeError(
                "'precision' kwarg is only valid in raw SA batched input mode."
            )
        if mode != "auto":
            raise TypeError(
                "'mode' kwarg only applies to density list inputs."
            )

        # Distinguish single MA p_attr (list of ndarrays) from a list
        # of MA p_attr blocks (list of lists). The detection only
        # examines the first element: ndarray → single MA;
        # list/tuple → list of MA. This matches the convention
        # used elsewhere in the toolbox and is what the matrix-form
        # output of translate_attributes produces.
        a_is_list = isinstance(a[0], (list, tuple))
        b = args[2] if len(args) >= 3 else None
        b_is_list = (
            _looks_like_multi_attr(b)
            and isinstance(b[0], (list, tuple))
        )

        if a_is_list and b_is_list:
            raise TypeError(
                "Raw multi-attribute list-vs-list is not supported; pass "
                "explicit density structs via the density list mode "
                "instead (build each entry with build_exp_tens first)."
            )

        if not a_is_list and not b_is_list:
            # Single MA scalar-vs-scalar — existing path.
            if len(args) not in (9, 10):
                raise TypeError(
                    f"Raw multi-attribute input expects 9 or 10 positional "
                    f"arguments (p_attr1, w1, p_attr2, w2, sigma_vec, "
                    f"r_vec, is_rel_vec, is_per_vec, "
                    f"period_vec[, is_sym_vec]); got {len(args)}."
                )
            return _cos_sim_raw_ma_scalar(
                *args,
                method=method,
                normalize=normalize,
                cancellation_threshold=cancellation_threshold,
                verbose=verbose,
            )

        # Scalar-vs-list broadcast. Build the scalar side once, then
        # iterate over the list side. Weights on the list side are
        # shared across every list entry.
        if len(args) not in (9, 10):
            raise TypeError(
                f"Raw multi-attribute scalar-vs-list input expects 9 or 10 "
                f"positional arguments (p_attr1, w1, p_attr2, w2, "
                f"sigma_vec, r_vec, is_rel_vec, is_per_vec, "
                f"period_vec[, is_sym_vec]); got {len(args)}."
            )
        return _cos_sim_raw_ma_broadcast(
            *args,
            a_is_list=a_is_list,
            b_is_list=b_is_list,
            method=method,
            normalize=normalize,
            cancellation_threshold=cancellation_threshold,
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Raw single-attribute dispatch.
    # ------------------------------------------------------------------
    if len(args) not in (9, 10):
        raise TypeError(
            f"Raw single-attribute input expects 9 or 10 positional "
            f"arguments (p1, w1, p2, w2, sigma, r, is_rel, is_per, "
            f"period[, is_sym]); got {len(args)}."
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

    try:
        b_arr = np.asarray(args[2], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Third positional argument (P2) must be a numeric array; got "
            f"{type(args[2]).__name__}."
        ) from exc

    if a_arr.ndim > 2 or b_arr.ndim > 2:
        raise TypeError(
            f"Raw SA inputs must be 1-D (single chord) or 2-D (batched); "
            f"got P1.ndim = {a_arr.ndim}, P2.ndim = {b_arr.ndim}."
        )

    # Batched dispatch fires whenever either operand is 2-D.
    if a_arr.ndim == 2 or b_arr.ndim == 2:
        sigma, r_, is_rel, is_per, period = args[4:9]
        is_sym = args[9] if len(args) == 10 else None
        W1_arg, W2_arg = args[1], args[3]

        # Reshape any 1-D operand to (1, K) so both are 2-D from here on.
        P1 = a_arr if a_arr.ndim == 2 else a_arr.reshape(1, -1)
        P2 = b_arr if b_arr.ndim == 2 else b_arr.reshape(1, -1)

        def _to_row_w(w):
            """Match a weights argument's shape to its (now 2-D) p."""
            if w is None:
                return None
            w_arr = np.asarray(w, dtype=np.float64)
            if w_arr.ndim == 1:
                return w_arr.reshape(1, -1)
            return w_arr

        W1 = _to_row_w(W1_arg)
        W2 = _to_row_w(W2_arg)

        M1, M2 = P1.shape[0], P2.shape[0]
        if M1 == 1 and M2 > 1:
            P1 = np.broadcast_to(P1, (M2, P1.shape[1])).copy()
            if W1 is not None:
                W1 = np.broadcast_to(W1, (M2, W1.shape[1])).copy()
        elif M2 == 1 and M1 > 1:
            P2 = np.broadcast_to(P2, (M1, P2.shape[1])).copy()
            if W2 is not None:
                W2 = np.broadcast_to(W2, (M1, W2.shape[1])).copy()
        elif M1 != M2:
            raise ValueError(
                f"Batched-raw P1 and P2 must either have matching row counts, "
                f"or one of them must be a single-row reference (1-D vector "
                f"or shape ``(1, K)``) to broadcast against the other. Got "
                f"{M1} and {M2} rows."
            )

        return _cos_sim_raw_sa_batch(
            P1, P2, sigma, r_, is_rel, is_per, period, is_sym,
            weights_a=W1, weights_b=W2,
            spectrum=spectrum, precision=precision,
            dedup=dedup,
            method=method,
            normalize=normalize,
            cancellation_threshold=cancellation_threshold,
            verbose=verbose,
        )

    # Both operands are 1-D → existing scalar SA path.
    if precision is not None:
        raise TypeError(
            "'precision' kwarg is only valid for raw SA batched input "
            "(at least one of P1, P2 must be 2-D)."
        )
    if mode != "auto":
        raise TypeError(
            "'mode' kwarg only applies to density list inputs."
        )
    return _cos_sim_raw_sa_scalar(
        *args, spectrum=spectrum,
        method=method,
        normalize=normalize,
        cancellation_threshold=cancellation_threshold,
        verbose=verbose,
    )



def _all_sa_pairs(pairs):
    """Return True iff every (a, b) pair in ``pairs`` is two ExpTensDensity objects."""
    for a, b in pairs:
        if not (isinstance(a, ExpTensDensity) and isinstance(b, ExpTensDensity)):
            return False
    return True



def _compute_pair_results_with_dedup_sa(
    pairs, *, method: str, normalize: str = "cosine",
    cancellation_threshold: float,
    truncation_sigmas=None, kernel_precision=None, verbose: bool,
):
    """Compute cos_sim for SA-density pairs with canonical-form dedup."""
    pair_key_to_idx: dict = {}
    pair_canon_idx: list[int] = []
    unique_pair_list: list = []

    for a, b in pairs:
        key_a, key_b, _, _, _, _ = _pair_canonical_key(
            a.p, a.w, b.p, b.w,
            sigma=a.sigma, r=a.r, is_rel=a.is_rel,
            is_per=a.is_per, period=a.period,
        )
        pk = (
            key_a,
            key_b,
            (b.sigma, b.r, b.is_rel, b.is_per, b.period),
        )
        if pk not in pair_key_to_idx:
            pair_key_to_idx[pk] = len(unique_pair_list)
            unique_pair_list.append((a, b))
        pair_canon_idx.append(pair_key_to_idx[pk])

    n_unique = len(unique_pair_list)
    if verbose:
        print(
            f"cos_sim_exp_tens: {len(pairs)} pairs, {n_unique} unique "
            f"after canonical-form dedup."
        )

    # Empirical-calibration time estimate. Warm-up plus a timed sample
    # of K ≤ 5 representative unique pairs, extrapolated over n_unique.
    # Gated on verbose; 10 s silence threshold via
    # maybe_print_batched_estimate. Parallels MATLAB localCosSimBatchedRaw
    # Phase 3.5 (cosSimExpTens.m).
    # Adaptive progress-print state. Defaults: silent. Overridden in
    # the calibration block when est_total is known.
    prog_stride = 1
    show_progress = False
    if verbose and n_unique >= 2:
        import time as _time
        from .._utils import progress_stride
        n_cal = min(5, n_unique)
        sample_idx = sorted(set(
            int(round(v)) for v in np.linspace(0, n_unique - 1, n_cal)
        ))
        # Warm-up call (absorbs one-time setup).
        a_w, b_w = unique_pair_list[sample_idx[0]]
        _cos_sim_exp_tens_sa(
            a_w, b_w,
            method=method,
            normalize=normalize,
            cancellation_threshold=cancellation_threshold,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
            verbose=False,
        )
        t_cal_start = _time.perf_counter()
        for ci in sample_idx:
            a_s, b_s = unique_pair_list[ci]
            _cos_sim_exp_tens_sa(
                a_s, b_s,
                method=method,
                normalize=normalize,
                cancellation_threshold=cancellation_threshold,
                truncation_sigmas=truncation_sigmas,
                kernel_precision=kernel_precision,
                verbose=False,
            )
        t_cal_total = _time.perf_counter() - t_cal_start
        t_per_pair = t_cal_total / len(sample_idx)
        est_total = t_cal_total + t_per_pair * n_unique
        maybe_print_batched_estimate(
            "cos_sim_exp_tens", n_unique, est_total,
        )
        prog_stride = progress_stride(t_per_pair)
        show_progress = est_total >= 5

    # Main loop over unique pairs (converted from list comprehension
    # so we can emit progress, matching MATLAB Phase 4).
    unique_results = []
    for up, (a, b) in enumerate(unique_pair_list):
        unique_results.append(_cos_sim_exp_tens_sa(
            a, b,
            method=method,
            normalize=normalize,
            cancellation_threshold=cancellation_threshold,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
            verbose=False,
        ))
        if verbose and show_progress \
                and ((up + 1) % prog_stride == 0 or up == n_unique - 1):
            print(f"  {up + 1} / {n_unique} unique pairs computed.")

    return [unique_results[idx] for idx in pair_canon_idx]



def _compute_pair_results_no_dedup(
    pairs, *, method: str, normalize: str = "cosine",
    cancellation_threshold: float,
    truncation_sigmas=None, kernel_precision=None, verbose: bool,
):
    """Compute cos_sim for a list of pairs without dedup.

    No empirical calibration is run on this path because the input
    may include heterogeneous MA densities whose per-pair cost varies
    too widely for a stable extrapolation. Consequently no progress
    prints are emitted; users wanting feedback on long runs should
    enable canonical-form dedup (the default).
    """
    results = []
    for a, b in pairs:
        results.append(_cos_sim_pair_core(
            a, b,
            method=method,
            normalize=normalize,
            cancellation_threshold=cancellation_threshold,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
            verbose=False,
        ))
    return results



def _cos_sim_pair_core(
    dens_x, dens_y, *,
    method: str = "auto",
    normalize: str = "cosine",
    cancellation_threshold: float = 1e-12,
    truncation_sigmas: float | None = None,
    kernel_precision: str | None = None,
    verbose: bool,
):
    """Internal: dispatch a single pair to the correct core IP routine.

    Routes to :func:`_cos_sim_exp_tens_sa` or :func:`_cos_sim_exp_tens_ma`,
    threading ``method``, ``normalize``, ``cancellation_threshold``,
    ``truncation_sigmas`` and ``kernel_precision`` through to the SA
    path (the MA path awaits its own helper-routing stage).
    ``WindowedMaetDensity`` operands are rejected here; user code
    reaches the windowed inner product via :func:`windowed_similarity`.
    """
    if isinstance(dens_x, WindowedMaetDensity) or \
            isinstance(dens_y, WindowedMaetDensity):
        raise TypeError(
            "cos_sim_exp_tens does not accept WindowedMaetDensity "
            "operands. Use windowed_similarity(dens_context, "
            "dens_query, window_spec, offsets) — pass a single-column "
            "offsets array for the scalar single-offset case, or a "
            "(dim, M) array for the M-offset sweep."
        )
    if isinstance(dens_x, MaetDensity):
        if not isinstance(dens_y, MaetDensity):
            raise TypeError(
                "dens_x is a MaetDensity but dens_y is not; both must be "
                "the same type."
            )
        return _cos_sim_exp_tens_ma(
            dens_x, dens_y,
            method=method,
            normalize=normalize,
            cancellation_threshold=cancellation_threshold,
            verbose=verbose,
        )
    if isinstance(dens_x, ExpTensDensity):
        if not isinstance(dens_y, ExpTensDensity):
            raise TypeError(
                "dens_x is an ExpTensDensity but dens_y is not; both must "
                "be the same type."
            )
        return _cos_sim_exp_tens_sa(
            dens_x, dens_y,
            method=method,
            normalize=normalize,
            cancellation_threshold=cancellation_threshold,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
            verbose=verbose,
        )
    raise TypeError(
        f"Both arguments must be ExpTensDensity, MaetDensity, or "
        f"WindowedMaetDensity; got {type(dens_x).__name__} and "
        f"{type(dens_y).__name__}."
    )



def _cos_sim_density_path(
    dens_x, dens_y, *,
    mode: str = "auto",
    dedup: bool = True,
    method: str = "auto",
    normalize: str = "cosine",
    cancellation_threshold: float = 1e-12,
    truncation_sigmas: float | None = None,
    kernel_precision: str | None = None,
    verbose: bool = True,
):
    """Density-input dispatch for :func:`cos_sim_exp_tens`."""
    is_x_scalar, list_x = _normalize_density_input(dens_x, name="dens_x")
    is_y_scalar, list_y = _normalize_density_input(dens_y, name="dens_y")

    # Scalar-vs-scalar.
    if is_x_scalar and is_y_scalar:
        return _cos_sim_pair_core(
            list_x[0], list_y[0],
            method=method,
            normalize=normalize,
            cancellation_threshold=cancellation_threshold,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
            verbose=verbose,
        )

    m = len(list_x)
    n = len(list_y)

    if is_x_scalar:
        if n == 0:
            return np.empty((0,), dtype=np.float64)
        a = list_x[0]
        pairs = [(a, b) for b in list_y]
        out_shape = (n,)
    elif is_y_scalar:
        if m == 0:
            return np.empty((0,), dtype=np.float64)
        b = list_y[0]
        pairs = [(a, b) for a in list_x]
        out_shape = (m,)
    else:
        if m == 0 or n == 0:
            try:
                resolved = _resolve_list_list_mode(mode, m, n)
            except ValueError:
                resolved = "cartesian"
            if resolved == "pairwise":
                return np.empty((0,), dtype=np.float64)
            return np.empty((m, n), dtype=np.float64)

        resolved = _resolve_list_list_mode(mode, m, n)
        if resolved == "pairwise":
            pairs = list(zip(list_x, list_y))
            out_shape = (m,)
        else:
            pairs = [(a, b) for a in list_x for b in list_y]
            out_shape = (m, n)

    if dedup and _all_sa_pairs(pairs):
        results = _compute_pair_results_with_dedup_sa(
            pairs,
            method=method,
            normalize=normalize,
            cancellation_threshold=cancellation_threshold,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
            verbose=verbose,
        )
    else:
        if dedup and verbose:
            print(
                "cos_sim_exp_tens: dedup=True requested but input includes "
                "non-SA densities; computing without dedup."
            )
        results = _compute_pair_results_no_dedup(
            pairs,
            method=method,
            normalize=normalize,
            cancellation_threshold=cancellation_threshold,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
            verbose=verbose,
        )

    return np.array(results, dtype=np.float64).reshape(out_shape)



def _cos_sim_raw_sa_scalar(
    p1, w1, p2, w2,
    sigma, r, is_rel, is_per, period, is_sym=None,
    *,
    spectrum=None,
    method: str = "auto",
    normalize: str = "cosine",
    cancellation_threshold: float = 1e-12,
    verbose: bool = True,
) -> float:
    """Raw single-attribute scalar dispatch for :func:`cos_sim_exp_tens`."""
    if is_sym is None:
        is_sym = True
    if spectrum is not None:
        p1_aug, w1_aug = add_spectra(
            np.asarray(p1, dtype=np.float64),
            np.ones_like(np.asarray(p1, dtype=np.float64)) if w1 is None
            else np.asarray(w1, dtype=np.float64),
            *spectrum,
        )
        p2_aug, w2_aug = add_spectra(
            np.asarray(p2, dtype=np.float64),
            np.ones_like(np.asarray(p2, dtype=np.float64)) if w2 is None
            else np.asarray(w2, dtype=np.float64),
            *spectrum,
        )
    else:
        p1_aug, w1_aug = p1, w1
        p2_aug, w2_aug = p2, w2

    dx = build_exp_tens(
        p1_aug, w1_aug, sigma, r, is_rel, is_per, period, is_sym,
        verbose=verbose,
    )
    dy = build_exp_tens(
        p2_aug, w2_aug, sigma, r, is_rel, is_per, period, is_sym,
        verbose=verbose,
    )
    return _cos_sim_pair_core(
        dx, dy,
        method=method,
        normalize=normalize,
        cancellation_threshold=cancellation_threshold,
        verbose=verbose,
    )



def _cos_sim_raw_ma_scalar(
    p_attr1, w1, p_attr2, w2,
    sigma_vec, r_vec, is_rel_vec, is_per_vec, period_vec, is_sym_vec=None,
    *,
    method: str = "auto",
    normalize: str = "cosine",
    cancellation_threshold: float = 1e-12,
    verbose: bool = True,
) -> float:
    """Raw multi-attribute scalar dispatch for :func:`cos_sim_exp_tens`."""
    dx = build_exp_tens(
        p_attr1, w1, sigma_vec, r_vec,
        is_rel_vec, is_per_vec, period_vec, is_sym_vec, verbose=verbose,
    )
    dy = build_exp_tens(
        p_attr2, w2, sigma_vec, r_vec,
        is_rel_vec, is_per_vec, period_vec, is_sym_vec, verbose=verbose,
    )
    return _cos_sim_pair_core(
        dx, dy,
        method=method,
        normalize=normalize,
        cancellation_threshold=cancellation_threshold,
        verbose=verbose,
    )


def _cos_sim_raw_ma_broadcast(
    p_attr1, w1, p_attr2, w2,
    sigma_vec, r_vec, is_rel_vec, is_per_vec, period_vec, is_sym_vec=None,
    *,
    a_is_list: bool,
    b_is_list: bool,
    method: str = "auto",
    normalize: str = "cosine",
    cancellation_threshold: float = 1e-12,
    verbose: bool = True,
) -> np.ndarray:
    """Raw multi-attribute scalar-vs-list broadcast.

    Exactly one of the two operands is a list of per-attribute ``p_attr``
    blocks (cell-of-cells). The scalar operand is built once and reused
    against every list entry. Weights for the list operand are shared
    across all entries (one ``w`` value, not a list of weights).

    Returns a 1-D ``ndarray`` of length M, the list length.
    """
    if a_is_list == b_is_list:
        # Caller (cos_sim_exp_tens) is responsible for ensuring exactly
        # one operand is a list; this is a sanity guard.
        raise RuntimeError(
            "_cos_sim_raw_ma_broadcast called without a clear "
            "scalar-vs-list configuration."
        )

    # Identify the list side and build the scalar side once.
    if b_is_list:
        scalar_pAttr, scalar_w = p_attr1, w1
        list_pAttr,  list_w   = p_attr2, w2
        scalar_first = True   # densX is scalar, densY is per-entry
    else:
        scalar_pAttr, scalar_w = p_attr2, w2
        list_pAttr,  list_w   = p_attr1, w1
        scalar_first = False  # densX is per-entry, densY is scalar

    dens_scalar = build_exp_tens(
        scalar_pAttr, scalar_w, sigma_vec, r_vec,
        is_rel_vec, is_per_vec, period_vec, is_sym_vec, verbose=verbose,
    )

    M = len(list_pAttr)
    out = np.empty(M, dtype=np.float64)
    for m in range(M):
        dens_m = build_exp_tens(
            list_pAttr[m], list_w, sigma_vec, r_vec,
            is_rel_vec, is_per_vec, period_vec, is_sym_vec, verbose=False,
        )
        if scalar_first:
            out[m] = _cos_sim_pair_core(
                dens_scalar, dens_m,
                method=method,
                normalize=normalize,
                cancellation_threshold=cancellation_threshold,
                verbose=False,
            )
        else:
            out[m] = _cos_sim_pair_core(
                dens_m, dens_scalar,
                method=method,
                normalize=normalize,
                cancellation_threshold=cancellation_threshold,
                verbose=False,
            )
    return out



# -------------------------------------------------------------------
#  _cos_sim_exp_tens_sa  (single-attribute legacy path)
# -------------------------------------------------------------------


_ORBIT_CANCELLATION_RATIO_MIN = 1e-10
"""Minimum acceptable cancellation ratio in the Möbius method's alternating partition sum.

When ``|sum| / max(|term|)`` drops below this threshold the result has
lost roughly 10 of its 16 significant decimal digits, leaving ~6
surviving — borderline acceptable for cosine accuracy at downstream
1e-6 user tolerance, but past this point the dispatcher falls back to
Bulger's method. See V22_DEV_LOG.md for the empirical regime where this
fires (sharp Gaussians + low K-r margin in absolute modes)."""



def _cos_sim_exp_tens_sa(
    dens_x: ExpTensDensity,
    dens_y: ExpTensDensity,
    *,
    method: str = "auto",
    normalize: str = "cosine",
    cancellation_threshold: float = 1e-12,
    truncation_sigmas: float | None = None,
    kernel_precision: str | None = None,
    verbose: bool = True,
) -> float:
    """Single-attribute cosine similarity.

    A ``method`` keyword routes between Bulger's method
    — the v1 / v2.1 decomposition with periodic pairwise-wrap form
    (``_ip_core``) — and the Möbius method. With
    the default ``method='auto'`` and perceptually typical parameters,
    the Möbius method is selected and the result agrees with v2.1 to
    floating-point precision.
    """
    dens_x = dens_x.pruned()
    dens_y = dens_y.pruned()
    if dens_x.r != dens_y.r:
        raise ValueError("Both densities must have the same r.")
    if dens_x.is_rel != dens_y.is_rel:
        raise ValueError("Both densities must have the same is_rel.")
    if dens_x.is_per != dens_y.is_per:
        raise ValueError("Both densities must have the same is_per.")
    if dens_x.is_per and dens_x.period != dens_y.period:
        raise ValueError("Both densities must have the same period.")
    if dens_x.sigma != dens_y.sigma:
        raise ValueError("Both densities must have the same sigma.")

    if method not in ("auto", "bulger", "direct", "mobius"):
        raise ValueError(
            f"method must be one of 'auto', 'bulger', 'direct', 'mobius'; "
            f"got {method!r}."
        )

    r = dens_x.r
    is_rel = dens_x.is_rel
    is_per = dens_x.is_per
    period = dens_x.period
    sigma = dens_x.sigma

    # Early return for degenerate case.
    if r > min(len(dens_x.p), len(dens_y.p)):
        return float("nan")

    n_max = max(len(dens_x.p), len(dens_y.p))
    n_min = min(len(dens_x.p), len(dens_y.p))
    sigma_over_P = sigma / period if (is_per and period > 0) else 0.0
    chosen, probed, est_sec, routing_reason = _select_and_estimate_sa_ip(
        dens_x, dens_y,
        method=method,
        truncation_sigmas=truncation_sigmas,
        kernel_precision=kernel_precision,
        verbose=verbose,
    )

    # Ordered ([sym]=0) densities are not symmetrised, so the orbit
    # (Möbius) inner product — which reconstructs the full S_r orbit
    # from p/w/r — does not represent them. The pairwise/centres path
    # reads the actual stored centres and is correct for either reading,
    # so force it whenever either operand is ordered at r > 1 (r = 1 is
    # vacuous: ordered and symmetric coincide).
    if (
        ((not bool(np.all(dens_x.is_sym)))
         or (not bool(np.all(dens_y.is_sym))))
        and r > 1
    ):
        chosen = "bulger"
        routing_reason = "ordered density (sym=0) requires centres path"

    # Dispatch-decision message: bypasses per-call verbose, gated by
    # the toolbox-wide show_hints flag and throttled once per
    # (function, chosen, routing_reason) per Python process. The
    # throttle is cleared by mpt.reset_defaults(). To fully silence:
    # mpt.set_default(show_hints=False).
    from .._defaults import _maybe_show_dispatch_msg
    _maybe_show_dispatch_msg(
        "cos_sim_exp_tens", chosen, routing_reason, est_sec, probed,
    )

    if chosen == "mobius":
        ip_xy, ip_xx, ip_yy, worst_ratio = _cos_sim_exp_tens_sa_orbit(
            dens_x, dens_y,
        )
        # Three layers of Möbius-method result validation, fall back on any:
        # 1. Cross-cancellation guard: <A,B> small relative to
        #    sqrt(<A,A><B,B>) — the Möbius estimate may be dominated by
        #    cancellation between partition terms.
        denom_geo = np.sqrt(max(ip_xx * ip_yy, 0.0))
        cross_cancellation = (
            denom_geo > 0
            and abs(ip_xy) < cancellation_threshold * denom_geo
        )
        # 2. Post-hoc sanity on the IPs themselves (catches the σ→0
        #    catastrophic-overflow regime: non-finite, sign-corrupt, or
        #    cosine outside [-1, 1]).
        ips_corrupted = _orbit_ips_look_corrupted(ip_xy, ip_xx, ip_yy)
        # 3. Runtime cancellation diagnostic: the alternating Möbius
        #    sum has lost too many significant digits, even if the
        #    final values look superficially fine. Catches the quieter
        #    sharp-Gaussian regime where IPs are finite-looking but
        #    ~1e-4 to 1e-2 wrong.
        cancellation_too_severe = (
            worst_ratio < _ORBIT_CANCELLATION_RATIO_MIN
        )
        if cross_cancellation or ips_corrupted or cancellation_too_severe:
            ip_xy, ip_xx, ip_yy = _cos_sim_exp_tens_sa_pairwise(
                dens_x, dens_y, verbose=verbose,
                truncation_sigmas=truncation_sigmas,
                kernel_precision=kernel_precision,
            )
    else:  # 'bulger' or 'direct' — coincide in SA mode
        ip_xy, ip_xx, ip_yy = _cos_sim_exp_tens_sa_pairwise(
            dens_x, dens_y, verbose=verbose,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
        )

    return _finalise_normalisation(ip_xy, ip_xx, ip_yy, normalize)



# -------------------------------------------------------------------
#  _cos_sim_exp_tens_ma  (multi-attribute path)
# -------------------------------------------------------------------


def _cos_sim_exp_tens_ma(
    dens_x: MaetDensity,
    dens_y: MaetDensity,
    *,
    method: str = "auto",
    normalize: str = "cosine",
    cancellation_threshold: float = 1e-12,
    verbose: bool = True,
) -> float:
    """Multi-attribute cosine similarity.

    A ``method`` keyword routes between Bulger's method
    — the v1 / v2.1 decomposition with periodic pairwise-wrap form
    (``_ip_core_ma``) — and the Möbius method.
    With the default ``method='auto'`` and perceptually typical
    parameters (no NaN-padded ``p_attr``, r_a ≤
    ``_ORBIT_R_MAX_SHIPPED``, σ/P ≤ ``_ORBIT_SIGMA_OVER_P_THRESHOLD``
    for periodic-relative groups), the Möbius method is selected and
    the result agrees with v2.1 to floating-point precision.

    Both densities must share the full parameter structure: number of
    attributes, group assignment, per-attribute ``r``, and per-group
    ``sigma``/``is_rel``/``is_per``/``period``. Weights and event/slot
    counts may differ freely — that's the whole point of the similarity
    measure.
    """
    dens_x = dens_x.pruned()
    dens_y = dens_y.pruned()

    # --- Structural compatibility ---
    if dens_x.n_attrs != dens_y.n_attrs:
        raise ValueError("Both MaetDensities must have the same n_attrs.")
    if not np.array_equal(dens_x.r, dens_y.r):
        raise ValueError("Both MaetDensities must have the same r (per attribute).")
    if not np.array_equal(dens_x.sigma, dens_y.sigma):
        raise ValueError("Both MaetDensities must have the same sigma (per attribute).")
    if not np.array_equal(dens_x.is_rel, dens_y.is_rel):
        raise ValueError("Both MaetDensities must have the same is_rel (per attribute).")
    if not np.array_equal(dens_x.is_per, dens_y.is_per):
        raise ValueError("Both MaetDensities must have the same is_per (per attribute).")
    # Periods must match for attributes where is_per is True (non-periodic
    # attributes can carry any period value without affecting the kernel).
    per_mask = dens_x.is_per.astype(bool)
    if np.any(dens_x.period[per_mask] != dens_y.period[per_mask]):
        raise ValueError(
            "Both MaetDensities must have the same period for periodic attributes."
        )

    if method not in ("auto", "bulger", "direct", "mobius"):
        raise ValueError(
            f"method must be one of 'auto', 'bulger', 'direct', 'mobius'; "
            f"got {method!r}."
        )

    # --- Dispatcher ---
    A = dens_x.n_attrs
    r_vec = dens_x.r
    is_rel = dens_x.is_rel
    is_per = dens_x.is_per
    sigma = dens_x.sigma
    period = dens_x.period

    r_max = int(np.max(r_vec)) if A > 0 else 1
    # Maximum σ/P across attributes that are both relative AND periodic.
    sop_max = 0.0
    any_per = False
    any_rel_nonper = False
    any_rel_per = False
    for a in range(A):
        if bool(is_per[a]):
            any_per = True
        if bool(is_rel[a]):
            if bool(is_per[a]):
                any_rel_per = True
                if float(period[a]) > 0:
                    sop_max = max(
                        sop_max,
                        float(sigma[a]) / float(period[a]),
                    )
            else:
                any_rel_nonper = True

    # Per-attribute slab dimension K_a (the kernel slab size; events
    # within an attribute may have lower K_eff via NaN padding, which
    # the Möbius-method wrapper handles via per-event safe/unsafe partition).
    k_vec = np.array(
        [int(M.shape[0]) for M in dens_x.p_attr], dtype=np.intp,
    ) if A > 0 else np.zeros(0, dtype=np.intp)

    chosen = _select_ma_inner_product_method(
        r_vec=r_vec, k_vec=k_vec, A=A,
        N_x=int(dens_x.n), N_y=int(dens_y.n),
        any_per=any_per,
        any_rel_nonper=any_rel_nonper,
        any_rel_per=any_rel_per,
        sigma_over_P_max=sop_max,
        user_method=method,
    )

    # Ordered ([sym]=0) attributes are not symmetrised, so the orbit
    # (Möbius) per-attribute inner product does not represent them. Force
    # the pairwise/centres path whenever any attribute is ordered at
    # r_a > 1 (r_a = 1 is vacuous). The centres path reads the actual
    # stored per-attribute centres and is correct for either reading.
    is_sym_x = np.asarray(getattr(dens_x, "is_sym", np.ones(A, dtype=bool)))
    is_sym_y = np.asarray(getattr(dens_y, "is_sym", np.ones(A, dtype=bool)))
    ordered_any = (
        np.any((~is_sym_x) & (r_vec > 1))
        or np.any((~is_sym_y) & (r_vec > 1))
    )
    if ordered_any:
        chosen = "bulger"

    # Nested attributes use a custom enumeration (and, for the inner unit,
    # a block-diagonal metric) that the orbit / Möbius re-enumeration does
    # not represent. Force the centres-based pairwise path, which reads the
    # stored per-attribute centres directly.
    nested_x = getattr(dens_x, "nested", None)
    nested_y = getattr(dens_y, "nested", None)
    nested_any = (
        (nested_x is not None and any(s is not None for s in nested_x))
        or (nested_y is not None and any(s is not None for s in nested_y))
    )
    if nested_any:
        if method == "auto":
            triple = _try_nested_contract(
                dens_x, dens_y, normalize=normalize, verbose=verbose)
            if triple is not None:
                return _finalise_normalisation(*triple, normalize)
        chosen = "bulger"

    if chosen == "mobius":
        ip_xy, ip_xx, ip_yy = _cos_sim_exp_tens_ma_orbit(
            dens_x, dens_y,
        )
        # Two layers of Möbius-method result validation, fall back on either.
        # The per-entry worst_ratio diagnostic that previously gated
        # this fallback (analogous to the SA case) was found to fire
        # spuriously for self-IP matrices: it reports per-(n,m) entry
        # cancellation in the per-attribute Möbius alternating partition sums, but
        # the cosine consumes only Σ_{n,m} P[n,m], where individual
        # entries with bad ratios contribute negligibly. Empirically,
        # at typical musical sigmas the diagnostic flagged ~100% of
        # MA self-IPs while the values themselves matched Bulger's method
        # to FP precision. The cross-cancellation guard plus the
        # post-hoc IP corruption check below catch the residual real
        # failure modes (small/sign-flipped cosines and non-finite
        # IPs respectively).
        denom_geo = np.sqrt(max(ip_xx * ip_yy, 0.0))
        cross_cancellation = (
            denom_geo > 0
            and abs(ip_xy) < cancellation_threshold * denom_geo
        )
        ips_corrupted = _orbit_ips_look_corrupted(ip_xy, ip_xx, ip_yy)
        if cross_cancellation or ips_corrupted:
            ip_xy, ip_xx, ip_yy = _cos_sim_exp_tens_ma_pairwise(
                dens_x, dens_y, verbose=verbose,
            )
    else:  # 'bulger' or 'direct' (coincide in MA mode)
        ip_xy, ip_xx, ip_yy = _cos_sim_exp_tens_ma_pairwise(
            dens_x, dens_y, verbose=verbose,
        )

    return _finalise_normalisation(ip_xy, ip_xx, ip_yy, normalize)



def _ip_core_ma(
    u_cell, w_u, n_j, v_cell, w_v, n_k,
    A, r_vec, sigma, is_rel, is_per, period,
    *, truncation_sigmas=None, inner_r=None,
):
    """MA inner product with memory-aware chunking along the comb side.

    Peak per-chunk memory is dominated by the largest per-attribute
    (r_a, nJ, nKc) difference tensor. Use ``(max(r_a) + 2) * nJ * 8``
    bytes per K-column as the sizing heuristic.

    ``truncation_sigmas`` is honoured via log-space thresholding on
    the accumulated MA log-kernel; ``None`` resolves to the global
    default ``mpt.get_default('truncation_sigmas')``.
    """
    from .._defaults import get_default

    if truncation_sigmas is None:
        truncation_sigmas = get_default('truncation_sigmas')

    max_r = int(np.max(r_vec)) if A > 0 else 1
    bytes_per_col = (max_r + 2) * int(n_j) * 8
    mem_limit = kernel_chunk_bytes_resolved()
    bytes_needed = bytes_per_col * int(n_k)

    if bytes_needed <= mem_limit:
        return _ip_full_ma(
            u_cell, w_u, n_j, v_cell, w_v, n_k,
            A, r_vec, sigma, is_rel, is_per, period,
            truncation_sigmas=truncation_sigmas, inner_r=inner_r,
        )

    chunk_size = max(1, int(mem_limit // max(bytes_per_col, 1)))
    acc = np.zeros(int(n_j), dtype=np.float64)
    for c_start in range(0, int(n_k), chunk_size):
        c_end = min(c_start + chunk_size, int(n_k))
        n_kc = c_end - c_start
        v_chunk = [V[:, c_start:c_end] for V in v_cell]
        log_kernel = _ma_log_kernel(
            u_cell, v_chunk, int(n_j), n_kc,
            A, r_vec, sigma, is_rel, is_per, period,
            inner_r=inner_r,
        )
        E = _trunc_log_kernel_exp(log_kernel, truncation_sigmas)
        acc = acc + E @ w_v[c_start:c_end]
    return float(w_u @ acc)



def _ip_full_ma(
    u_cell, w_u, n_j, v_cell, w_v, n_k,
    A, r_vec, sigma, is_rel, is_per, period,
    *, truncation_sigmas=None, inner_r=None,
):
    """Fully vectorized MA inner product (single chunk).

    ``truncation_sigmas`` is honoured via log-space thresholding;
    ``None`` resolves to the global default.
    """
    from .._defaults import get_default

    if truncation_sigmas is None:
        truncation_sigmas = get_default('truncation_sigmas')

    log_kernel = _ma_log_kernel(
        u_cell, v_cell, int(n_j), int(n_k),
        A, r_vec, sigma, is_rel, is_per, period,
        inner_r=inner_r,
    )
    E = _trunc_log_kernel_exp(log_kernel, truncation_sigmas)
    return float(w_u @ (E @ w_v))



def _ma_log_kernel(
    u_cell, v_cell, n_j, n_k,
    A, r_vec, sigma, is_rel, is_per, period,
    *, inner_r=None,
):
    """Accumulate the summed-Q / (4 sigma^2) log-kernel across attributes.

    For each attribute *a*:
      1. Compute (r_a, nJ, nK) differences between perm-side and comb-side.
      2. Apply periodic wrapping for the attribute's group.
      3. Compute the per-attribute quadratic form Q_a.
      4. Accumulate ``-Q_a / (4 sigma^2)`` into log_kernel.

    ``inner_r[a] > 0`` selects the inner ``[rel]`` co-transposition unit
    for attribute *a*: the block-diagonal metric over its event blocks
    (full-tuple convention, ``reduced=False``).
    """
    log_kernel = np.zeros((int(n_j), int(n_k)), dtype=np.float64)
    for a in range(A):
        r_a = int(r_vec[a])
        D = u_cell[a][:, :, None] - v_cell[a][:, None, :]  # (r_a, nJ, nK)

        r_in = 0 if inner_r is None else int(inner_r[a])
        if r_in > 0:
            # Inner unit: block-diagonal sum of per-event quotient forms.
            # _compute_Q applies the pairwise wrap inside, so the full
            # tuples enter without an outer wrap.
            Q_a = _compute_Q_inner_blocks(
                D, r_in, bool(is_per[a]), float(period[a]), reduced=False)
            log_kernel = log_kernel - Q_a / (4 * float(sigma[a]) ** 2)
            continue

        # The outer wrap is only needed when _compute_Q does not re-wrap
        # the pairwise component differences (i.e., for is_per and not
        # is_rel: Q = sum(D**2), which requires wrapped D components).
        # For rel+per, _compute_Q wraps each pairwise (D[i]-D[j]) inside
        # (Eq 6 form); that inner wrap is invariant under integer-period
        # shifts, so wrapping D first is redundant.
        if is_per[a] and not is_rel[a]:
            p_a = float(period[a])
            D = D - p_a * np.floor(D / p_a + 0.5)

        Q_a = _compute_Q(D, r_a, bool(is_rel[a]), bool(is_per[a]),
                         float(period[a]))
        log_kernel = log_kernel - Q_a / (4 * float(sigma[a]) ** 2)

    return log_kernel



# -------------------------------------------------------------------
#  Möbius method dispatcher (multi-attribute path)
# -------------------------------------------------------------------
#
#  Per the MAET inner-product factorisation (JMM Eq. 3.4 with the
#  per-attribute integral separation noted in JMM §3.2.1, and the
#  Möbius–Bulger remark JMM Rem. 3.1), the cross-event inner product
#  decomposes as
#
#      <T_X, T_Y>_MA = Σ_{n_X, n_Y} Π_a I_a(n_X, n_Y)
#
#  where I_a(n_X, n_Y) is an SA-shaped Möbius inner product over
#  the K_a slot values of event n_X (X-side) against those of n_Y
#  (Y-side), with the group's mode parameters. Bulger's method
#  collapses this into a flat (n_J × n_K) bilinear form that scales
#  as N² · Π_a [r_a! · C(K_a, r_a)]²; the Möbius form scales as
#  N² · A · |Ω_{r_a}| · K_a², a substantial saving when K_a is
#  non-trivial.
#
#  Limitations of the Möbius method:
#  - NaN-padded ``p_attr`` (variable K_a per event) is not yet
#    supported by the per-event Möbius loop; dispatcher detects and
#    falls back to Bulger's method.
#  - Per-attribute r_a > _ORBIT_R_MAX_SHIPPED falls back (no orbit
#    table shipped at that order).
#
#  Per-attribute Möbius calls apply the SA convention's
#  (σ_a √π)^{r_a} prefactor, so the Möbius-method MA bare triple
#  (ip_xy, ip_xx, ip_yy) differs from Bulger's MA triple by
#  Π_a (σ_a √π)^{r_a} · r_a! — which cancels in the cosine.


def _ma_has_nan(dens):
    """True if any p_attr matrix has NaN entries (variable K_a per event)."""
    return any(np.isnan(M).any() for M in dens.p_attr)



def _trunc_kernel_exp(exp_arg, sigma, truncation_sigmas):
    """Evaluate ``exp(-exp_arg / (4 σ²))`` with optional truncation.

    ``exp_arg`` is the non-negative quantity entering the kernel
    exponent. For a 1-D pairwise Gaussian inner-product kernel,
    ``exp_arg = (p_x - p_y)²``; for an r-D r-tuple kernel,
    ``exp_arg = ||d||² = Σ_a d_a²``.

    When ``truncation_sigmas`` is finite, entries with
    ``exp_arg > 2 · (truncation_sigmas · σ)²`` are zeroed without
    evaluating ``np.exp``, saving work proportional to the pruned
    fraction. The threshold uniformly matches "the inner-product
    kernel value falls below ``exp(-truncation_sigmas² / 2)``": that
    kernel is ``G(d; σ√2)``, so its value at distance ``|d|`` is
    ``exp(-|d|² / (4 σ²))``, and the cutoff condition is
    ``|d|² > 2 (truncation_sigmas · σ)²``.

    For r-tuple kernels the same threshold on ``Σ_a d_a²`` is correct
    because the r-D kernel is ``Π_a G(d_a; σ√2) = exp(-Σ_a d_a² /
    (4 σ²))``.

    When ``truncation_sigmas`` is ``None`` or non-finite, the full
    exponential is computed and no masking work is done.
    """
    if truncation_sigmas is None or not np.isfinite(truncation_sigmas):
        return np.exp(-exp_arg / (4 * sigma ** 2))
    cutoff = 2.0 * (truncation_sigmas * sigma) ** 2
    mask = exp_arg <= cutoff
    out = np.zeros_like(exp_arg)
    out[mask] = np.exp(-exp_arg[mask] / (4 * sigma ** 2))
    return out



def _trunc_log_kernel_exp(log_kernel, truncation_sigmas):
    """Evaluate ``exp(log_kernel)`` with optional truncation in log space.

    ``log_kernel`` is the (non-positive) log of the kernel — typically
    ``-Σ_a d_a² / (4 σ_a²)`` accumulated across attributes (the
    Bulger-method MA log-kernel pattern), where each attribute may
    have its own σ.

    When ``truncation_sigmas`` is finite, entries with
    ``log_kernel < -truncation_sigmas² / 2`` are zeroed without
    evaluating ``np.exp``. The threshold uniformly matches: kernel
    value falls below ``exp(-truncation_sigmas² / 2)``. This is the
    log-space counterpart of :func:`_trunc_kernel_exp` and applies
    cleanly to the MA log-kernel case where per-attribute σ values
    differ (so a single quadratic-form cutoff doesn't apply).

    When ``truncation_sigmas`` is ``None`` or non-finite, the full
    exponential is computed and no masking work is done.
    """
    if truncation_sigmas is None or not np.isfinite(truncation_sigmas):
        return np.exp(log_kernel)
    threshold = -0.5 * truncation_sigmas ** 2
    mask = log_kernel >= threshold
    out = np.zeros_like(log_kernel)
    out[mask] = np.exp(log_kernel[mask])
    return out



def _ma_per_attr_inner_matrix(
    Px, Wx, Py, Wy, sigma, r, is_rel, is_per, period,
    *, return_cancellation_ratio=False, truncation_sigmas=None,
    prune_zero_weight_events=True,
):
    """Per-attribute (event_X, event_Y) inner product matrix for the
    MA path under the Möbius method.

    ``Px`` is (K, N_x), ``Wx`` is (K, N_x); same shape for Y. Returns
    an (N_x, N_y) matrix where entry (n_X, n_Y) is the per-attribute
    inner product over the K slot values of event n_X (X-side) against
    those of n_Y (Y-side).

    Strategy (in parity with MATLAB ``mobius.maPerAttrInnerMatrix``):

    - r = 1: direct kernel sum with NaN -> zero-weight padding (no
      Möbius decomposition, cancellation impossible).

    - r >= 2 abs: hybrid safe/unsafe partition. An event is "safe" on
      this attribute iff its non-NaN slot count K_eff satisfies
      ``K_eff - r >= _ORBIT_K_MINUS_R_MIN`` (= 2; the precision margin
      used elsewhere in the Möbius machinery). Safe-vs-safe pairs flow
      through the vectorised batched Möbius method with within-safe-group
      zero-padding. Pairs involving any unsafe event flow through
      :func:`_inner_product_direct_abs_sa`, which is exact for any
      K >= r (no Möbius alternating sum, so no cancellation).

    - r >= 2 rel: per-event-pair loop with zero-pad. Auto dispatch
      routes any rel group globally to Bulger's method; this path runs
      only on explicit ``method='mobius'`` opt-in. Events with K_eff - r
      below the precision margin in this niche regime may lose
      precision in the Möbius relative-mode u-grid integration; users
      wanting exact rel + ragged Möbius-method behaviour should either
      filter events to K_eff >= r + 2 or use ``method='auto'`` (which
      routes to Bulger's method).

    With ``return_cancellation_ratio=True``, additionally returns the
    worst-case (minimum) cancellation ratio across the (N_x, N_y)
    entries — a scalar in (0, 1]. Direct-enum entries always have
    ratio 1.0; the worst ratio comes from the safe-Möbius submatrix.
    If no safe pairs exist, the worst ratio is 1.0.

    ``truncation_sigmas`` is honoured in every kernel-evaluation
    branch (r=1 abs, r>=2 abs safe, r>=2 abs unsafe via
    :func:`_batched_direct_enum_abs_sa`, and rel-per via
    :func:`_ma_per_attr_inner_matrix_rel_per`): kernel entries whose
    underlying squared distance exceeds the truncation cutoff are
    zeroed without evaluating ``np.exp``. ``None`` resolves to the
    global default ``mpt.get_default('truncation_sigmas')``.

    ``prune_zero_weight_events`` (default ``True``) drops events with
    column-wise weight identically zero (treating NaN as missing)
    before dispatching to any sub-helper. Such events contribute zero
    to every kernel entry, so the result is mathematically
    unchanged; the saving is wall-clock — the einsum / batched
    Möbius contraction operates on smaller matrices. Combines
    naturally with the global ``truncation_sigmas`` since
    :func:`mpt.weight_events` hard-zeros the factor outside the
    cutoff. Result is scattered back into the full (N_x, N_y) output
    shape with zeros in the dropped rows / columns. Set
    ``prune_zero_weight_events=False`` to bypass.
    """
    from .._mobius import inner_product_orbit_pw_batched
    from .._defaults import get_default

    if truncation_sigmas is None:
        truncation_sigmas = get_default('truncation_sigmas')

    K_x_max, N_x = Px.shape
    K_y_max, N_y = Py.shape

    # --- Zero-weight-event pruning (auto, before dispatch) ---
    # An event contributes zero to every output entry iff its weight
    # column is identically zero in this attribute (NaN entries are
    # missing slots — equivalent to zero in the IP). Drop such events,
    # recurse on the smaller matrices, scatter the result back.
    if prune_zero_weight_events and (N_x > 0) and (N_y > 0):
        # nanmax over a column returns NaN only when ALL entries are
        # NaN (which is an invalid event); for any partial-NaN column
        # it returns the max over the non-NaN slots. Compare > 0 to
        # find columns with at least one positive-magnitude slot.
        with np.errstate(invalid='ignore'):
            col_max_x = np.nanmax(np.abs(Wx), axis=0)
            col_max_y = np.nanmax(np.abs(Wy), axis=0)
        keep_x = np.isfinite(col_max_x) & (col_max_x > 0.0)
        keep_y = np.isfinite(col_max_y) & (col_max_y > 0.0)
        if not (keep_x.all() and keep_y.all()):
            if not (keep_x.any() and keep_y.any()):
                # Every event on at least one side has zero weight;
                # the IP is the all-zero matrix.
                result = np.zeros((N_x, N_y), dtype=np.float64)
                if return_cancellation_ratio:
                    return result, 1.0
                return result
            sub = _ma_per_attr_inner_matrix(
                Px[:, keep_x], Wx[:, keep_x],
                Py[:, keep_y], Wy[:, keep_y],
                sigma, r, is_rel, is_per, period,
                return_cancellation_ratio=return_cancellation_ratio,
                truncation_sigmas=truncation_sigmas,
                prune_zero_weight_events=False,   # avoid infinite recursion
            )
            if return_cancellation_ratio:
                sub_ip, sub_ratio = sub
            else:
                sub_ip = sub
            result = np.zeros((N_x, N_y), dtype=np.float64)
            result[np.ix_(np.where(keep_x)[0], np.where(keep_y)[0])] = sub_ip
            if return_cancellation_ratio:
                return result, sub_ratio
            return result

    # --- r = 1: direct kernel sum, zero-pad fine (no cancellation) ---
    if r == 1:
        Px_, Wx_, Py_, Wy_ = _zero_pad_nan(Px, Wx, Py, Wy)
        prefactor = sigma * np.sqrt(np.pi)
        # Memory: each of diffs, diffs**2, K_tens is
        # (K_x_max, chunk_N_x, K_y_max, N_y) * 8 bytes. Up to ~3 live
        # arrays during evaluation; budget accordingly.
        per_row_bytes = 3 * K_x_max * K_y_max * N_y * 8
        mem_limit = kernel_chunk_bytes_resolved()
        chunk_N_x = max(1, min(N_x, mem_limit // max(per_row_bytes, 1)))

        if chunk_N_x >= N_x:
            # Fast path: single shot.
            diffs = Px_[:, :, None, None] - Py_[None, None, :, :]
            if is_per:
                diffs = diffs - period * np.floor(diffs / period + 0.5)
            K_tens = _trunc_kernel_exp(diffs ** 2, sigma, truncation_sigmas)
            out = np.einsum(
                'xn,xnym,ym->nm', Wx_, K_tens, Wy_, optimize=True,
            )
            result = out * prefactor
        else:
            # Chunked path: process N_x in chunks.
            result = np.empty((N_x, N_y), dtype=np.float64)
            for n_start in range(0, N_x, chunk_N_x):
                n_end = min(n_start + chunk_N_x, N_x)
                Px_chunk = Px_[:, n_start:n_end]
                Wx_chunk = Wx_[:, n_start:n_end]
                diffs = Px_chunk[:, :, None, None] - Py_[None, None, :, :]
                if is_per:
                    diffs = diffs - period * np.floor(diffs / period + 0.5)
                K_tens = _trunc_kernel_exp(diffs ** 2, sigma, truncation_sigmas)
                out_chunk = np.einsum(
                    'xn,xnym,ym->nm', Wx_chunk, K_tens, Wy_, optimize=True,
                )
                result[n_start:n_end, :] = out_chunk * prefactor

        if return_cancellation_ratio:
            return result, 1.0
        return result

    # --- r >= 2 rel: per-pair loop, zero-pad (rare regime) ---
    if is_rel:
        Px_, Wx_, Py_, Wy_ = _zero_pad_nan(Px, Wx, Py, Wy)
        if is_per:
            return _ma_per_attr_inner_matrix_rel_per(
                Px_, Wx_, Py_, Wy_, sigma, r, period,
                return_cancellation_ratio=return_cancellation_ratio,
                truncation_sigmas=truncation_sigmas,
            )
        out = np.empty((N_x, N_y), dtype=np.float64)
        worst_ratio = 1.0
        for n_X in range(N_x):
            for n_Y in range(N_y):
                if return_cancellation_ratio:
                    v, ratio = _orbit_inner_rel(
                        Px_[:, n_X], Wx_[:, n_X], Py_[:, n_Y], Wy_[:, n_Y],
                        sigma, r, False, period,
                        return_cancellation_ratio=True,
                    )
                    out[n_X, n_Y] = v
                    if ratio < worst_ratio:
                        worst_ratio = ratio
                else:
                    out[n_X, n_Y] = _orbit_inner_rel(
                        Px_[:, n_X], Wx_[:, n_X], Py_[:, n_Y], Wy_[:, n_Y],
                        sigma, r, False, period,
                    )
        if return_cancellation_ratio:
            return out, worst_ratio
        return out

    # --- r >= 2 abs: hybrid safe/unsafe partition ---

    K_MARGIN_MIN = _ORBIT_K_MINUS_R_MIN

    # Per-event K_eff (count of non-NaN slots), per side.
    K_eff_x = np.sum(~(np.isnan(Px) | np.isnan(Wx)), axis=0)   # (N_x,)
    K_eff_y = np.sum(~(np.isnan(Py) | np.isnan(Wy)), axis=0)   # (N_y,)

    safe_x_mask = (K_eff_x - r) >= K_MARGIN_MIN
    safe_y_mask = (K_eff_y - r) >= K_MARGIN_MIN
    safe_x_idx = np.where(safe_x_mask)[0]
    unsafe_x_idx = np.where(~safe_x_mask)[0]
    safe_y_idx = np.where(safe_y_mask)[0]
    unsafe_y_idx = np.where(~safe_y_mask)[0]

    out = np.zeros((N_x, N_y), dtype=np.float64)
    worst_ratio = 1.0

    # --- Safe x Safe submatrix: vectorised batched Möbius method ---
    if safe_x_idx.size > 0 and safe_y_idx.size > 0:
        Px_s = Px[:, safe_x_idx]
        Wx_s = Wx[:, safe_x_idx]
        Py_s = Py[:, safe_y_idx]
        Wy_s = Wy[:, safe_y_idx]
        # Within-safe zero-pad (K still varies per event in safe group).
        Px_s, Wx_s, Py_s, Wy_s = _zero_pad_nan(Px_s, Wx_s, Py_s, Wy_s)

        N_xs = safe_x_idx.size
        N_ys = safe_y_idx.size
        prefactor = (sigma * np.sqrt(np.pi)) ** r
        # Memory: each of diffs, diffs**2, K_tens is
        # (K_x_max, chunk_N_xs, K_y_max, N_ys) * 8 bytes; ~3 live arrays.
        # K_pairs reshape adds another N_pairs * K_x_max * K_y_max * 8.
        per_row_bytes = 4 * K_x_max * K_y_max * N_ys * 8
        mem_limit = kernel_chunk_bytes_resolved()
        chunk_N_xs = max(1, min(N_xs, mem_limit // max(per_row_bytes, 1)))

        if chunk_N_xs >= N_xs:
            # Fast path: single shot.
            diffs = Px_s[:, :, None, None] - Py_s[None, None, :, :]
            if is_per:
                diffs = diffs - period * np.floor(diffs / period + 0.5)
            K_tens = _trunc_kernel_exp(diffs ** 2, sigma, truncation_sigmas)
            K_pairs = np.transpose(K_tens, (1, 3, 0, 2)).reshape(
                N_xs * N_ys, K_x_max, K_y_max,
            )
            w_A_pairs = np.broadcast_to(
                Wx_s.T[:, None, :], (N_xs, N_ys, K_x_max),
            ).reshape(N_xs * N_ys, K_x_max)
            w_B_pairs = np.broadcast_to(
                Wy_s.T[None, :, :], (N_xs, N_ys, K_y_max),
            ).reshape(N_xs * N_ys, K_y_max)

            if return_cancellation_ratio:
                flat, ratios = inner_product_orbit_pw_batched(
                    K_pairs, w_A_pairs, w_B_pairs, r,
                    prefactor=prefactor,
                    return_cancellation_ratio=True,
                )
                worst_ratio = min(worst_ratio, float(np.min(ratios)))
            else:
                flat = inner_product_orbit_pw_batched(
                    K_pairs, w_A_pairs, w_B_pairs, r,
                    prefactor=prefactor,
                )
            out[np.ix_(safe_x_idx, safe_y_idx)] = flat.reshape(N_xs, N_ys)
        else:
            # Chunked path: process safe_x_idx in chunks of chunk_N_xs.
            safe_flat = np.empty((N_xs, N_ys), dtype=np.float64)
            for n_start in range(0, N_xs, chunk_N_xs):
                n_end = min(n_start + chunk_N_xs, N_xs)
                n_chunk = n_end - n_start
                Px_chunk = Px_s[:, n_start:n_end]
                Wx_chunk = Wx_s[:, n_start:n_end]
                diffs = Px_chunk[:, :, None, None] - Py_s[None, None, :, :]
                if is_per:
                    diffs = diffs - period * np.floor(diffs / period + 0.5)
                K_tens = _trunc_kernel_exp(diffs ** 2, sigma, truncation_sigmas)
                K_pairs = np.transpose(K_tens, (1, 3, 0, 2)).reshape(
                    n_chunk * N_ys, K_x_max, K_y_max,
                )
                w_A_pairs = np.broadcast_to(
                    Wx_chunk.T[:, None, :], (n_chunk, N_ys, K_x_max),
                ).reshape(n_chunk * N_ys, K_x_max)
                w_B_pairs = np.broadcast_to(
                    Wy_s.T[None, :, :], (n_chunk, N_ys, K_y_max),
                ).reshape(n_chunk * N_ys, K_y_max)

                if return_cancellation_ratio:
                    flat, ratios = inner_product_orbit_pw_batched(
                        K_pairs, w_A_pairs, w_B_pairs, r,
                        prefactor=prefactor,
                        return_cancellation_ratio=True,
                    )
                    worst_ratio = min(worst_ratio, float(np.min(ratios)))
                else:
                    flat = inner_product_orbit_pw_batched(
                        K_pairs, w_A_pairs, w_B_pairs, r,
                        prefactor=prefactor,
                    )
                safe_flat[n_start:n_end, :] = flat.reshape(n_chunk, N_ys)
            out[np.ix_(safe_x_idx, safe_y_idx)] = safe_flat

    # --- Pairs involving any unsafe event: K-grouped batched direct ---
    # All pairs not in (safe_x, safe_y) flow through ordered-r-tuple
    # direct enumeration. Was previously a Python double-loop
    # (one ``_inner_product_direct_abs_sa`` call per pair); for
    # variable-K_a workloads with many unsafe events this dominated
    # the runtime by 10–100× over the actual computation.
    #
    # K_a grouping: partition the unsafe-involved event-index union
    # by K_eff value per side, then batch direct enumeration per
    # (K_eff_x, K_eff_y) sub-block. Within a sub-block, every event
    # shares an ordered-r-tuple shape (nJ = K_eff! / (K_eff - r)!),
    # so the IP matrix can be computed as a single contracted
    # tensor op. This removes the Python per-pair overhead entirely.
    #
    # Coverage: (unsafe_x, all_y) ∪ (safe_x, unsafe_y) covers every
    # pair where at least one side is unsafe, without double counting.
    needed_x_idx = np.concatenate([unsafe_x_idx, safe_x_idx]) \
        if unsafe_x_idx.size > 0 else np.array([], dtype=np.intp)
    needed_y_idx_full = np.arange(N_y)
    # Split needed pairs into two coverage zones to mirror the
    # inline-method structure exactly, preserving fill ordering.
    _ma_fill_direct_enum_groups(
        out, Px, Wx, Py, Wy,
        unsafe_x_idx, np.arange(N_y),
        K_eff_x, K_eff_y, sigma, r, is_per, period,
        truncation_sigmas=truncation_sigmas,
    )
    if unsafe_y_idx.size > 0 and safe_x_idx.size > 0:
        _ma_fill_direct_enum_groups(
            out, Px, Wx, Py, Wy,
            safe_x_idx, unsafe_y_idx,
            K_eff_x, K_eff_y, sigma, r, is_per, period,
            truncation_sigmas=truncation_sigmas,
        )

    if return_cancellation_ratio:
        return out, worst_ratio
    return out



def _ma_fill_direct_enum_groups(
    out, Px, Wx, Py, Wy, x_idx, y_idx,
    K_eff_x, K_eff_y, sigma, r, is_per, period,
    *, truncation_sigmas=None,
):
    """K-grouped batched direct-enum fill into ``out`` for a rectangle
    of (x_idx, y_idx) pairs.

    Partitions ``x_idx`` by K_eff_x value and ``y_idx`` by K_eff_y
    value, then computes each (K_x_val, K_y_val) sub-block as a single
    vectorised tensor contraction. Output entries at (x_idx[i],
    y_idx[j]) are filled in place.

    ``truncation_sigmas`` is forwarded to
    :func:`_batched_direct_enum_abs_sa`; ``None`` resolves to the
    global default.

    No-op if either side is empty.
    """
    if x_idx.size == 0 or y_idx.size == 0:
        return

    # Unique K_eff values present on each side (within the index sets).
    unique_K_x = np.unique(K_eff_x[x_idx])
    unique_K_y = np.unique(K_eff_y[y_idx])

    for K_x_val in unique_K_x:
        x_grp = x_idx[K_eff_x[x_idx] == K_x_val]
        if x_grp.size == 0 or int(K_x_val) < r:
            # K < r: ordered r-tuple set is empty; IP = 0.
            continue
        # Pack non-NaN slots to the top of each group column. The
        # build_exp_tens convention has NaN already at the bottom, so
        # in the common case this is a memory-cheap slice; in the
        # general case _pack_nan_top handles arbitrary NaN positions.
        Px_grp, Wx_grp = _pack_nan_top(Px[:, x_grp], Wx[:, x_grp])
        Px_grp = Px_grp[:int(K_x_val), :]
        Wx_grp = Wx_grp[:int(K_x_val), :]
        for K_y_val in unique_K_y:
            y_grp = y_idx[K_eff_y[y_idx] == K_y_val]
            if y_grp.size == 0 or int(K_y_val) < r:
                continue
            Py_grp, Wy_grp = _pack_nan_top(Py[:, y_grp], Wy[:, y_grp])
            Py_grp = Py_grp[:int(K_y_val), :]
            Wy_grp = Wy_grp[:int(K_y_val), :]
            sub_ip = _batched_direct_enum_abs_sa(
                Px_grp, Wx_grp, Py_grp, Wy_grp,
                sigma, r, is_per, period,
                truncation_sigmas=truncation_sigmas,
            )
            out[np.ix_(x_grp, y_grp)] = sub_ip



def _pack_nan_top(P, W):
    """Pack non-NaN slots to the top of each column.

    Returns ``(P_packed, W_packed)`` of the same shape, where for each
    column ``n`` the first ``K_eff[n]`` rows are the valid slots
    (preserving their original order) and the rest are NaN. The
    ``build_exp_tens`` convention already places NaN at the bottom, in
    which case this is mathematically a no-op (still copies for
    cleanliness). Per-event packing handles user-constructed densities
    with arbitrary NaN positions.
    """
    K, N = P.shape
    P_packed = np.full_like(P, np.nan)
    W_packed = np.full_like(W, np.nan)
    for n in range(N):
        valid = ~(np.isnan(P[:, n]) | np.isnan(W[:, n]))
        k = int(valid.sum())
        if k == 0:
            continue
        P_packed[:k, n] = P[valid, n]
        W_packed[:k, n] = W[valid, n]
    return P_packed, W_packed



def _batched_direct_enum_abs_sa(
    Px_group, Wx_group, Py_group, Wy_group,
    sigma, r, is_per, period,
    *, truncation_sigmas=None,
):
    """Batched direct r-tuple enumeration IP for groups at fixed K_x, K_y.

    Vectorised replacement for repeated calls to
    :func:`_inner_product_direct_abs_sa` when every event in
    ``Px_group`` has the same ``K_x = K_eff_x`` and every event in
    ``Py_group`` has the same ``K_y = K_eff_y`` (no NaN within the
    first K rows of either side).

    Inputs
    ------
    Px_group : (K_x, N_x) ndarray
        Slot positions, no NaN.
    Wx_group : (K_x, N_x) ndarray
        Slot weights, no NaN.
    Py_group, Wy_group : (K_y, N_y) ndarrays
        Same for Y side.
    sigma, r, is_per, period
        Group parameters.
    truncation_sigmas : float, optional
        Kernel-truncation cutoff in σ units. Kernel entries whose
        squared distance exceeds the cutoff are zeroed without
        evaluating ``np.exp``. ``None`` resolves to the global default
        ``mpt.get_default('truncation_sigmas')``.

    Returns
    -------
    ip : (N_x, N_y) ndarray
        Inner-product matrix (no Möbius alternating sum; exact for any
        K_x, K_y >= r).
    """
    from .._defaults import get_default

    if truncation_sigmas is None:
        truncation_sigmas = get_default('truncation_sigmas')

    K_x, N_x = Px_group.shape
    K_y, N_y = Py_group.shape

    if K_x < r or K_y < r:
        return np.zeros((N_x, N_y), dtype=np.float64)

    if r == 1:
        diffs = Px_group[:, :, None, None] - Py_group[None, None, :, :]
        if is_per:
            diffs = diffs - period * np.floor(diffs / period + 0.5)
        K_mat = _trunc_kernel_exp(diffs ** 2, sigma, truncation_sigmas)
        ip = np.einsum(
            'xn,xnym,ym->nm', Wx_group, K_mat, Wy_group, optimize=True,
        )
        return ip * (sigma * np.sqrt(np.pi))

    # --- r >= 2: enumerate ordered r-tuple indices ---
    from itertools import permutations
    idx_x = np.array(list(permutations(range(K_x), r)),
                     dtype=np.intp)        # (nJ_x, r)
    idx_y = np.array(list(permutations(range(K_y), r)),
                     dtype=np.intp)        # (nJ_y, r)
    nJ_x = idx_x.shape[0]                  # K_x! / (K_x - r)!
    nJ_y = idx_y.shape[0]

    # Gather tuple slot positions and weights per event. The fancy
    # index Px_group[idx_x.T, :] has shape (r, nJ_x, N_x); we want
    # U_x of shape (r, N_x, nJ_x) and Wj_x of shape (N_x, nJ_x).
    U_x = Px_group[idx_x.T, :].transpose(0, 2, 1)
    U_y = Py_group[idx_y.T, :].transpose(0, 2, 1)
    Wj_x = np.prod(Wx_group[idx_x.T, :], axis=0).T  # (N_x, nJ_x)
    Wj_y = np.prod(Wy_group[idx_y.T, :], axis=0).T  # (N_y, nJ_y)

    # Memory estimate: the difference tensor is (r, N_x, nJ_x, N_y, nJ_y).
    # For unsafe events (K_eff in {r, r+1}), nJ_x = r! or (r+1)!/(1!),
    # which is small. For r=3, K=4 -> nJ=24; r=4, K=5 -> nJ=120. Even
    # with N_x = N_y = 100 this is <100 MB at worst. No chunking needed
    # in the unsafe regime. Document so future use on safe-K paths
    # adds a chunking guard.
    diffs = U_x[:, :, :, None, None] - U_y[:, None, None, :, :]
    if is_per:
        diffs = diffs - period * np.floor(diffs / period + 0.5)
    Q = np.sum(diffs ** 2, axis=0)                  # (N_x, nJ_x, N_y, nJ_y)
    K_mat = _trunc_kernel_exp(Q, sigma, truncation_sigmas)

    ip = np.einsum(
        'xj,xjyk,yk->xy', Wj_x, K_mat, Wj_y, optimize=True,
    )
    return ip * (sigma * np.sqrt(np.pi)) ** r



def _zero_pad_nan(Px, Wx, Py, Wy):
    """Replace NaN entries in P / W with 0 (zero-weight padding).

    Returns new arrays (does not mutate inputs). The Möbius method's weighted
    contractions read ``w_i^m``, so a zero-weight slot kills any Möbius
    term involving that slot regardless of the corresponding p value
    — mathematically equivalent to per-event truncation.
    """
    nan_x = np.isnan(Px) | np.isnan(Wx)
    if nan_x.any():
        Px = np.where(nan_x, 0.0, Px)
        Wx = np.where(nan_x, 0.0, Wx)
    nan_y = np.isnan(Py) | np.isnan(Wy)
    if nan_y.any():
        Py = np.where(nan_y, 0.0, Py)
        Wy = np.where(nan_y, 0.0, Wy)
    return Px, Wx, Py, Wy



def _ma_per_attr_inner_matrix_rel_per(
    Px, Wx, Py, Wy, sigma, r, period, samples_per_sigma=5,
    *, return_cancellation_ratio=False, truncation_sigmas=None,
):
    """Vectorised relative-periodic case of ``_ma_per_attr_inner_matrix``.

    Builds an (N_pairs · N_u, K, K) kernel tensor and runs a single
    batched Möbius-method call across both axes; the trapezoidal weights are
    applied after reshaping back to (N_pairs, N_u). Memory peak is
    ``N_pairs · N_u · K^2 · 8`` bytes plus a similar-sized intermediate
    diffs tensor; chunked along u to stay under a 1 GB ceiling.

    ``samples_per_sigma=5`` is a deliberate trade-off: the Gaussian
    integrand is smooth at the σ scale, so trapezoidal convergence is
    geometric and 5 samples/σ delivers cosine agreement well below
    1e-9 relative on representative MAET parameter ranges. The SA
    relative path uses 10 because per-call cost is small there; for
    MA the integration is run N_pairs = N_x · N_y times in parallel,
    so halving the grid roughly halves wall-clock cost.

    With ``return_cancellation_ratio=True``, additionally returns the
    worst-case ratio across the (N_pairs · N_u) batched Möbius-method cells.

    ``truncation_sigmas`` is honoured on the per-u kernel tensor:
    entries whose squared (period-wrapped) distance exceeds the
    cutoff are zeroed without evaluating ``np.exp``. ``None``
    resolves to the global default ``mpt.get_default('truncation_sigmas')``.
    """
    from .._mobius import inner_product_orbit_pw_batched
    from .._defaults import get_default

    if truncation_sigmas is None:
        truncation_sigmas = get_default('truncation_sigmas')

    K, N_x = Px.shape
    _, N_y = Py.shape

    N_u = max(64, int(np.ceil(period / sigma * samples_per_sigma)))
    u_grid = np.linspace(0.0, period, N_u, endpoint=False)
    du = period / N_u

    # Memory: pair tensor diffs_pair has shape (K, N_x, K, N_y); inner
    # u-loop's kernel tensor has shape (n_uc, K, N_x_chunk, K, N_y).
    # Chunk along N_x to keep diffs_pair under budget; the existing
    # u-chunking nests inside, sized for the per-chunk N_pairs.
    bytes_per_row = K * K * N_y * 8
    mem_limit = kernel_chunk_bytes_resolved()
    # Reserve half the budget for the pair tensor; the u-chunk loop
    # uses the rest for the kernel tensor.
    chunk_N_x = max(1, min(N_x, mem_limit // max(2 * bytes_per_row, 1)))

    F = np.zeros((N_x * N_y, N_u), dtype=np.float64)
    worst_ratio = 1.0

    for n_start in range(0, N_x, chunk_N_x):
        n_end = min(n_start + chunk_N_x, N_x)
        nc_x = n_end - n_start
        Px_chunk = Px[:, n_start:n_end]
        Wx_chunk = Wx[:, n_start:n_end]
        N_pairs_chunk = nc_x * N_y

        # Per-pair weights for this chunk (independent of u).
        w_A_pairs = np.broadcast_to(
            Wx_chunk.T[:, None, :], (nc_x, N_y, K),
        ).reshape(N_pairs_chunk, K)
        w_B_pairs = np.broadcast_to(
            Wy.T[None, :, :], (nc_x, N_y, K),
        ).reshape(N_pairs_chunk, K)

        # Pair-wise raw differences for this chunk: (K, nc_x, K, N_y).
        diffs_pair = Px_chunk[:, :, None, None] - Py[None, None, :, :]

        # Inner: chunk along u.
        bytes_per_u = N_pairs_chunk * K * K * 8 * 2  # kernel + diffs
        chunk_u = max(1, min(N_u, mem_limit // max(bytes_per_u, 1)))

        F_chunk = np.zeros((N_pairs_chunk, N_u), dtype=np.float64)
        for u_start in range(0, N_u, chunk_u):
            u_end_u = min(u_start + chunk_u, N_u)
            n_uc = u_end_u - u_start
            u_slice = u_grid[u_start:u_end_u]
            # diffs[u, K_i, nc_x, K_j, N_y] = diffs_pair + u
            diffs = diffs_pair[None, :, :, :, :] + u_slice[:, None, None, None, None]
            diffs = diffs - period * np.floor(diffs / period + 0.5)
            K_uc = _trunc_kernel_exp(diffs ** 2, sigma, truncation_sigmas)
            # Reorder to (n_uc, nc_x, N_y, K, K) and flatten leading axes.
            K_uc = np.transpose(K_uc, (0, 2, 4, 1, 3)).reshape(
                n_uc * N_pairs_chunk, K, K,
            )
            # Replicate weights across u-axis for each pair.
            w_A_uc = np.broadcast_to(
                w_A_pairs[None, :, :], (n_uc, N_pairs_chunk, K),
            ).reshape(n_uc * N_pairs_chunk, K)
            w_B_uc = np.broadcast_to(
                w_B_pairs[None, :, :], (n_uc, N_pairs_chunk, K),
            ).reshape(n_uc * N_pairs_chunk, K)
            if return_cancellation_ratio:
                flat, ratios = inner_product_orbit_pw_batched(
                    K_uc, w_A_uc, w_B_uc, r, prefactor=1.0,
                    return_cancellation_ratio=True,
                )
                chunk_min = float(np.min(ratios))
                if chunk_min < worst_ratio:
                    worst_ratio = chunk_min
            else:
                flat = inner_product_orbit_pw_batched(
                    K_uc, w_A_uc, w_B_uc, r, prefactor=1.0,
                )
            F_chunk[:, u_start:u_end_u] = flat.reshape(n_uc, N_pairs_chunk).T

        # Insert into global F: pair indices for rows [n_start, n_end)
        # are the contiguous block [n_start * N_y : n_end * N_y).
        F[n_start * N_y : n_end * N_y, :] = F_chunk

    integral = F.sum(axis=1) * du  # periodic Riemann sum
    c = sigma * np.sqrt(2 * np.pi / r)
    out = (sigma * np.sqrt(np.pi)) ** r * integral / c ** 2
    if return_cancellation_ratio:
        return out.reshape(N_x, N_y), worst_ratio
    return out.reshape(N_x, N_y)



def _cos_sim_exp_tens_ma_orbit(dens_x, dens_y):
    """Compute (ip_xy, ip_xx, ip_yy) for the MA case via per-attribute
    Möbius method (JMM Eq. 3.4 plus Rem. 3.1).

    Caller is responsible for ensuring no NaN in ``p_attr`` and for
    structural compatibility of the two densities.

    Note: previous versions also returned a ``worst_ratio`` aggregating
    per-entry cancellation ratios across the (N_x × N_y) inner-product
    matrices. That diagnostic was found to over-conservatively flag
    correct results — the per-entry ratio reflects cancellation in
    individual Möbius cells, but the cosine consumes only the
    sums Σ_{n,m} P[n,m], where individual entries with bad ratios
    contribute negligibly when their absolute value is small. Removed
    in favour of relying on the cross-cancellation guard
    and the post-hoc IP corruption check (see
    ``_orbit_ips_look_corrupted``) at the dispatcher level.
    """
    A = dens_x.n_attrs
    N_x = dens_x.n
    N_y = dens_y.n

    P_xy = np.ones((N_x, N_y), dtype=np.float64)
    P_xx = np.ones((N_x, N_x), dtype=np.float64)
    P_yy = np.ones((N_y, N_y), dtype=np.float64)

    for a in range(A):
        r_a = int(dens_x.r[a])
        sigma = float(dens_x.sigma[a])
        is_rel = bool(dens_x.is_rel[a])
        is_per = bool(dens_x.is_per[a])
        period = float(dens_x.period[a])

        Px, Py = dens_x.p_attr[a], dens_y.p_attr[a]
        Wx, Wy = dens_x.w[a], dens_y.w[a]

        I_xy = _ma_per_attr_inner_matrix(
            Px, Wx, Py, Wy, sigma, r_a, is_rel, is_per, period,
        )
        I_xx = _ma_per_attr_inner_matrix(
            Px, Wx, Px, Wx, sigma, r_a, is_rel, is_per, period,
        )
        I_yy = _ma_per_attr_inner_matrix(
            Py, Wy, Py, Wy, sigma, r_a, is_rel, is_per, period,
        )
        P_xy *= I_xy
        P_xx *= I_xx
        P_yy *= I_yy

    return float(P_xy.sum()), float(P_xx.sum()), float(P_yy.sum())



def _try_nested_contract(dens_x, dens_y, *, normalize, verbose):
    """Fast tree-contraction of a single nested attribute's inner product.

    Returns (ip_xy, ip_xx, ip_yy) when the case is covered -- one nested
    attribute, outer/no ``[rel]``, no NaN-padding, cosine normalisation --
    and the contraction is estimated cheaper than the enumeration; otherwise
    ``None``, and the caller routes to the exact enumeration. Absolute and
    relative-non-periodic are exact; relative-periodic uses the
    transposition-average surrogate and warns when ``sigma/period`` exceeds
    the ``truncation_sigmas``-implied tolerance.
    """
    import math as _m
    from .._defaults import get_default

    if normalize != "cosine":
        return None
    if dens_x.n_attrs != 1 or dens_y.n_attrs != 1:
        return None
    spec = dens_x.nested[0]
    if spec is None or dens_y.nested[0] is None:
        return None
    if int(_inner_r_vec(dens_x)[0]) != 0 or int(_inner_r_vec(dens_y)[0]) != 0:
        return None  # inner [rel] unit not covered by the contraction yet

    PX = np.asarray(dens_x.p_attr[0], dtype=np.float64)
    PY = np.asarray(dens_y.p_attr[0], dtype=np.float64)
    if np.isnan(PX).any() or np.isnan(PY).any():
        return None  # variable-K per event: exact enumeration only
    WX = np.asarray(dens_x.w[0], dtype=np.float64)
    WY = np.asarray(dens_y.w[0], dtype=np.float64)

    from ._nested_contraction import (
        build_recipe, tuple_counts, recipe_work, quad_nodes,
        make_quadrature, nested_ip,
    )

    r_levels = np.asarray(spec["r"]).ravel()
    sym_levels = np.asarray(spec["sym"]).ravel()
    tags = np.asarray(spec["tags"])
    is_rel = bool(dens_x.is_rel[0])
    is_per = bool(dens_x.is_per[0])
    period = float(dens_x.period[0])
    sigma = float(dens_x.sigma[0])
    ts = get_default("truncation_sigmas")

    n_x = PX.shape[1]
    n_y = PY.shape[1]
    vmin = float(min(PX.min(), PY.min()))
    vmax = float(max(PX.max(), PY.max()))
    recipe = build_recipe(r_levels, sym_levels, tags)

    # Speed dispatch (deterministic integer/float counts -> identical in
    # both languages). Enumeration ~ event-pairs * M_perm * M_comb;
    # contraction ~ event-pairs * quadrature-nodes * tree combine-work.
    m_perm, m_comb = tuple_counts(r_levels, sym_levels, tags)
    Q = quad_nodes(is_rel, is_per, sigma, period, vmin, vmax, ts)
    pair_terms = n_x * n_y + n_x * n_x + n_y * n_y
    cost_enum = pair_terms * m_perm * m_comb
    cost_contract = pair_terms * Q * recipe_work(recipe)
    if cost_contract >= cost_enum:
        return None  # enumeration is the faster route

    if is_rel and is_per:
        tol = (max(_m.exp(-0.5 * ts ** 2), 1e-12)
               if _m.isfinite(ts) else 1e-12)
        sop_max = (0.85 / (4.0 * _m.sqrt(_m.log(1.0 / tol)))
                   if tol < 1.0 else _m.inf)
        if sigma / period > sop_max:
            warnings.warn(
                f"Nested relative-periodic similarity at sigma/period = "
                f"{sigma / period:.3f} exceeds the surrogate accuracy "
                f"threshold {sop_max:.3f} implied by truncation_sigmas "
                f"(tolerance {tol:.1e}); the transposition-average value "
                f"may depart from the exact inner product. Pass "
                f"method='bulger' for the exact enumeration.",
                stacklevel=2,
            )

    quad = make_quadrature(is_rel, is_per, sigma, period, vmin, vmax, ts)

    def trip(pa, wa, na, pb, wb, nb):
        s = 0.0
        for i in range(na):
            ai = pa[:, i]
            wi = wa[:, i]
            for j in range(nb):
                s += nested_ip(recipe, ai, pb[:, j], wi, wb[:, j],
                               sigma, period, ts, quad)
        return s

    ip_xy = trip(PX, WX, n_x, PY, WY, n_y)
    ip_xx = trip(PX, WX, n_x, PX, WX, n_x)
    ip_yy = trip(PY, WY, n_y, PY, WY, n_y)
    return ip_xy, ip_xx, ip_yy


def _cos_sim_exp_tens_ma_pairwise(dens_x, dens_y, *, verbose: bool = True):
    """Compute (ip_xy, ip_xx, ip_yy) for the MA case via the
    Bulger's method (``_ip_core_ma``).

    This is the body of the original ``_cos_sim_exp_tens_ma``
    factored out so the new dispatcher can route to it cleanly.
    """
    A = dens_x.n_attrs
    r_vec = dens_x.r
    sigma = dens_x.sigma
    is_rel = dens_x.is_rel
    is_per = dens_x.is_per
    period = dens_x.period

    n_jx, n_kx = dens_x.n_j, dens_x.n_k
    n_jy, n_ky = dens_y.n_j, dens_y.n_k

    inner_r = _inner_r_vec(dens_x)

    total_pairs = n_jx * n_ky + n_jx * n_kx + n_jy * n_ky
    max_r = int(np.max(r_vec)) if A > 0 else 1
    estimate_comp_time(total_pairs, max_r, "cos_sim_exp_tens (MAET)", verbose)

    ip_xy = _ip_core_ma(
        dens_x.u_perm, dens_x.w_j, n_jx,
        dens_y.v_comb, dens_y.wv_comb, n_ky,
        A, r_vec, sigma, is_rel, is_per, period,
        inner_r=inner_r,
    )
    ip_xx = _ip_core_ma(
        dens_x.u_perm, dens_x.w_j, n_jx,
        dens_x.v_comb, dens_x.wv_comb, n_kx,
        A, r_vec, sigma, is_rel, is_per, period,
        inner_r=inner_r,
    )
    ip_yy = _ip_core_ma(
        dens_y.u_perm, dens_y.w_j, n_jy,
        dens_y.v_comb, dens_y.wv_comb, n_ky,
        A, r_vec, sigma, is_rel, is_per, period,
        inner_r=inner_r,
    )
    return ip_xy, ip_xx, ip_yy



# -------------------------------------------------------------------
#  cos_sim_exp_tens_raw  (dispatches SA or MA based on input shape)
# -------------------------------------------------------------------


def cos_sim_exp_tens_raw(
    p1, w1, p2, w2, *args,
    method: str = "auto",
    cancellation_threshold: float = 1e-12,
    verbose: bool = True,
) -> float:
    """Deprecated. Use :func:`cos_sim_exp_tens` directly with raw input.

    .. deprecated:: 2.1
       The raw-input dispatch has been folded into the unified
       :func:`cos_sim_exp_tens` entry point. Pass raw arrays directly:

       - SA: ``cos_sim_exp_tens(p1, w1, p2, w2, sigma, r, is_rel, is_per, period)``
       - MA: ``cos_sim_exp_tens(p_attr1, w1, p_attr2, w2, sigma_vec, r_vec, is_rel_vec, is_per_vec, period_vec)``

       This shim will be removed in a future release.
    """
    warnings.warn(
        "cos_sim_exp_tens_raw is deprecated. The same call signature is "
        "now supported directly by cos_sim_exp_tens (pass raw arrays as the "
        "first arguments instead of pre-built density objects). This shim "
        "will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    return cos_sim_exp_tens(
        p1, w1, p2, w2, *args,
        method=method,
        cancellation_threshold=cancellation_threshold,
        verbose=verbose,
    )



def _ip_core(U, wU, nJ, V, wV, nK, r, sigma, is_rel, is_per, period,
             truncation_sigmas=None, kernel_precision=None):
    """Core inner product (perm-side × comb-side).

    Two-axis routing (mirrors :func:`_eval_exp_tens_sa`):

    - **Routing axis** — abs and rel-non-periodic forms have a helper
      reduction (``sigma_eff = sigma * sqrt(2)``); rel+periodic does
      not yet and stays on the inline / chunked path.
    - **Execution axis** — even when the helper is available, route
      through it only when feature kwargs are explicitly requested
      (after resolving ``None`` against the global defaults). Default
      mode runs the ``_ip_full`` / chunked path inline, avoiding
      the helper's per-call argument validation overhead.

    This preserves the inline-direct cost profile for default-mode callers
    (e.g. ``cos_sim_exp_tens`` in per-pair tight loops) while
    enabling the helper's truncation / precision features whenever
    the user opts in.
    """
    # ---- Execution-axis decision: resolve defaults first ----
    # ``None`` means "consult global default", not "no feature".
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

    can_use_helper = not (is_rel and is_per)

    # Fire the kernel-evaluation hint once per session when Bulger's IP
    # path is about to run with default kwargs. Bulger's path
    # forms a kernel-matrix-of-r-tuple-pairs that benefits from the
    # same truncation / single-precision controls as the centres path.
    if use_default_kwargs:
        from .._defaults import _maybe_show_kernel_eval_hint
        _maybe_show_kernel_eval_hint(
            effective_truncation_sigmas=float("inf"),
            effective_kernel_precision="double",
        )

    # Route through helper only when features are actually requested
    # AND the helper supports this quadratic form.
    if can_use_helper and not use_default_kwargs:
        return _ip_via_helper(
            U, wU, V, wV, r, sigma, is_rel, is_per, period,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
        )

    # Default-mode (or rel+per) path: inline / chunked.
    bytes_needed = (r + 2) * int(nJ) * int(nK) * 8
    mem_limit = kernel_chunk_bytes_resolved()

    if bytes_needed <= mem_limit:
        return _ip_full(
            U, wU, nJ, V, wV, nK, r, sigma, is_rel, is_per, period,
            truncation_sigmas=trunc_resolved,
        )

    chunk_size = max(1, int(mem_limit / ((r + 2) * int(nJ) * 8)))
    acc = np.zeros(nJ)
    for c in range(0, nK, chunk_size):
        c_end = min(c + chunk_size, nK)
        idx = slice(c, c_end)
        n_kc = c_end - c

        Dc = U[:, :, None] - V[:, idx][:, None, :]
        # See note in _ip_full: outer wrap is only needed when
        # _compute_Q does not re-wrap pairwise component differences.
        if is_per and not is_rel:
            Dc = Dc - period * np.floor(Dc / period + 0.5)
        Qc = _compute_Q(Dc, r, is_rel, is_per, period)
        Ec = _trunc_kernel_exp(Qc, sigma, trunc_resolved)
        acc += Ec @ wV[idx]

    return float(wU @ acc)



def _ip_via_helper(U, wU, V, wV, r, sigma, is_rel, is_per, period,
                   truncation_sigmas=None, kernel_precision=None):
    """Route the centres-IP through :func:`gaussian_kernel_sum`.

    The helper computes ``g(q) = sum_j wJ(j) * exp(-Q(c_j - x_q) /
    (2 * sigma_eff^2))`` with ``sigma_eff = sigma * sqrt(2)``, so the
    kernel exponent matches the centres-IP's ``Q / (4 * sigma^2)``.
    The IP is then ``wU @ g``.

    Supports abs (per and non-per) and rel-non-periodic. The rel+per
    pairwise-wrap form is not yet supported by the helper.
    """
    kw = dict(is_rel=bool(is_rel), r=int(r),
              is_per=bool(is_per), period=float(period))
    if truncation_sigmas is not None:
        kw["truncation_sigmas"] = float(truncation_sigmas)
    if kernel_precision is not None:
        kw["kernel_precision"] = kernel_precision
    sigma_eff = float(sigma) * np.sqrt(2.0)
    g = gaussian_kernel_sum(V, wV.ravel(), U, sigma_eff, **kw)
    return float(np.asarray(g).ravel() @ wU.ravel())



def _ip_full(U, wU, nJ, V, wV, nK, r, sigma, is_rel, is_per, period,
             *, truncation_sigmas=None):
    """Fully vectorized inner product.

    ``truncation_sigmas`` is honoured on the kernel; ``None``
    resolves to the global default.
    """
    from .._defaults import get_default

    if truncation_sigmas is None:
        truncation_sigmas = get_default('truncation_sigmas')

    D = U[:, :, None] - V[:, None, :]  # (r, nJ, nK)

    # The outer wrap is needed only when _compute_Q does not re-wrap
    # the pairwise component differences (i.e., for is_per and not
    # is_rel: Q = sum(D**2), which requires wrapped D components).
    # For is_rel+is_per, _compute_Q wraps each (D[i]-D[j]) inside
    # (the pairwise-wrap form of Eq 6); that inner wrap is invariant
    # under integer-period shifts of the operands, so wrapping D first
    # is redundant. Skipping it saves ~30-45% of _ip_full time across K.
    if is_per and not is_rel:
        D = D - period * np.floor(D / period + 0.5)

    Q = _compute_Q(D, r, is_rel, is_per, period)

    E = _trunc_kernel_exp(Q, sigma, truncation_sigmas)   # (nJ, nK)
    return float(wU @ (E @ wV))



def _orbit_inner_abs(p_a, w_a, p_b, w_b, sigma, r, is_per, period,
                     *, return_cancellation_ratio=False,
                     truncation_sigmas=None):
    """<T_A, T_B> in absolute mode via the Möbius method.

    With ``return_cancellation_ratio=True``, returns ``(value, ratio)``
    where ratio is ``|sum| / max(|term|)`` from the Möbius alternating
    partition sum (1.0 means no cancellation; <<1 means digits lost). See
    :func:`mpt._mobius.inner_product_orbit` for full semantics.

    ``truncation_sigmas`` is honoured on the kernel; ``None`` resolves
    to the global default.
    """
    from .._mobius import inner_product_orbit
    from .._defaults import get_default

    if truncation_sigmas is None:
        truncation_sigmas = get_default('truncation_sigmas')

    diffs = p_a[:, None] - p_b[None, :]
    if is_per:
        diffs = diffs - period * np.floor(diffs / period + 0.5)
    K = _trunc_kernel_exp(diffs ** 2, sigma, truncation_sigmas)
    return inner_product_orbit(
        K, w_a, w_b, r, prefactor=(sigma * np.sqrt(np.pi)) ** r,
        return_cancellation_ratio=return_cancellation_ratio,
    )



def _inner_product_direct_abs_sa(p_x, w_x, p_y, w_y, sigma, r,
                                   is_per, period):
    """<T_X, T_Y> in absolute mode via direct r-tuple enumeration.

    Computes the SA inner product
        <T_X, T_Y> = (sigma * sqrt(pi))**r *
                     sum_{J, K} wJ_x[J] * wJ_y[K] *
                                exp(-||centres_x[:, J] - centres_y[:, K]||^2
                                    / (4 sigma^2))
    by enumerating ordered r-tuples on each side. No Möbius
    alternating sum is involved, so the result is exact (no
    catastrophic cancellation) for any K_x, K_y >= r. This is the
    "unsafe" path of the MA per-attribute IP matrix, used for event
    pairs where at least one event has K_eff - r below the
    Möbius-method precision margin (`_ORBIT_K_MINUS_R_MIN` = 2).

    NaN tolerance: NaN entries in ``p_x`` / ``w_x`` / ``p_y`` / ``w_y``
    are dropped per side before enumeration. If the dropped count
    leaves either side with fewer than r valid slots, returns 0 by
    convention (cannot form an r-tuple).

    Cost: O(K_x! / (K_x - r)! * K_y! / (K_y - r)! * r) per call. Cheap
    when K is close to r (the unsafe regime).
    """
    p_x = np.asarray(p_x, dtype=np.float64).ravel()
    w_x = np.asarray(w_x, dtype=np.float64).ravel()
    p_y = np.asarray(p_y, dtype=np.float64).ravel()
    w_y = np.asarray(w_y, dtype=np.float64).ravel()

    valid_x = ~(np.isnan(p_x) | np.isnan(w_x))
    valid_y = ~(np.isnan(p_y) | np.isnan(w_y))
    p_x = p_x[valid_x]; w_x = w_x[valid_x]
    p_y = p_y[valid_y]; w_y = w_y[valid_y]
    K_x = p_x.size
    K_y = p_y.size

    if K_x < r or K_y < r:
        return 0.0

    if r == 1:
        diffs = p_x[:, None] - p_y[None, :]
        if is_per:
            diffs = diffs - period * np.floor(diffs / period + 0.5)
        K_mat = np.exp(-(diffs ** 2) / (4 * sigma ** 2))
        return float(sigma * np.sqrt(np.pi) *
                     np.einsum('i,ij,j->', w_x, K_mat, w_y))

    # r >= 2: enumerate ordered r-tuples and contract.
    U_x, wJ_x = _build_ordered_r_tuples(p_x, w_x, r)   # (r, nJ_x), (nJ_x,)
    U_y, wJ_y = _build_ordered_r_tuples(p_y, w_y, r)
    nJ_x = U_x.shape[1]
    nJ_y = U_y.shape[1]

    diffs = U_x[:, :, None] - U_y[:, None, :]   # (r, nJ_x, nJ_y)
    if is_per:
        diffs = diffs - period * np.floor(diffs / period + 0.5)
    Q = np.sum(diffs ** 2, axis=0)              # (nJ_x, nJ_y)
    K_mat = np.exp(-Q / (4 * sigma ** 2))

    return float((sigma * np.sqrt(np.pi)) ** r *
                 np.einsum('i,ij,j->', wJ_x, K_mat, wJ_y))



def _build_ordered_r_tuples(p, w, r):
    """Mirror of :class:`SAExpTensDensity` ordered-tuple construction.

    Returns ``(U, wJ)`` where ``U`` is ``(r, nJ)`` of position values
    along ordered r-tuples and ``wJ`` is ``(nJ,)`` of weight products.
    Used by :func:`_inner_product_direct_abs_sa` and any other helper
    that needs single-event ordered tuples without going through the
    full :func:`build_exp_tens` API.
    """
    import math
    K = p.size
    n_perms = math.factorial(r)
    n_combs = int(_comb(K, r, exact=True))
    n_j = n_perms * n_combs

    nck = _nchoosek_indices(K, r)             # r x n_combs
    all_perms = np.array(
        list(permutations(range(r))), dtype=np.intp,
    ).T                                        # r x r!

    j_idx = np.empty((r, n_j), dtype=np.intp)
    offset = 0
    for i in range(n_perms):
        j_idx[:, offset:offset + n_combs] = nck[all_perms[:, i], :]
        offset += n_combs

    U = p[j_idx]                               # r x nJ
    wJ = np.prod(w[j_idx], axis=0)             # (nJ,)
    return U, wJ



def _orbit_inner_rel(p_a, w_a, p_b, w_b, sigma, r, is_per, period,
                     samples_per_sigma=10, *,
                     return_cancellation_ratio=False,
                     truncation_sigmas=None):
    """<T_A, T_B> in relative mode via Möbius machinery + translation grid.

    Marginalises a translation u over either ``[0, P)`` (periodic) or a
    Gaussian-supported window around the alignment of A and B
    (non-periodic), and integrates the Möbius-evaluated kernel against u.
    The grid density is ``samples_per_sigma`` points per σ; the
    truncation in the non-periodic case extends 8σ beyond the natural
    overlap window. (The earlier 4σ default truncated tails of the
    Möbius integrand at ~5e-10 — small per kernel value, but
    enough to corrupt the auto-inner products at ~1e-6 relative
    precision once Möbius cancellation amplified them. 8σ pushes the
    truncation tail to FP noise; 12σ is empirically no improvement.)

    With ``return_cancellation_ratio=True``, returns ``(value, ratio)``
    where ratio is the worst-case (minimum) cancellation ratio across
    the u-grid points. A point with severe cancellation drives the
    integration result toward catastrophic loss of significance.

    ``truncation_sigmas`` is honoured on the kernel; ``None`` resolves
    to the global default.
    """
    from .._mobius import inner_product_orbit_grid
    from .._defaults import get_default

    if truncation_sigmas is None:
        truncation_sigmas = get_default('truncation_sigmas')

    if is_per:
        N_u = max(64, int(np.ceil(period / sigma * samples_per_sigma)))
        u_grid = np.linspace(0.0, period, N_u, endpoint=False)
        du = period / N_u
        diffs = (p_a[None, :, None] - p_b[None, None, :]
                 + u_grid[:, None, None])
        diffs = diffs - period * np.floor(diffs / period + 0.5)
    else:
        u_min = p_b.min() - p_a.max() - 8.0 * sigma
        u_max = p_b.max() - p_a.min() + 8.0 * sigma
        N_u = max(
            64,
            int(np.ceil(max(u_max - u_min, 1.0) / sigma * samples_per_sigma)),
        )
        u_grid = np.linspace(u_min, u_max, N_u)
        diffs = (p_a[None, :, None] - p_b[None, None, :]
                 + u_grid[:, None, None])
    K_u = _trunc_kernel_exp(diffs ** 2, sigma, truncation_sigmas)
    if return_cancellation_ratio:
        F, ratios = inner_product_orbit_grid(
            K_u, w_a, w_b, r, return_cancellation_ratio=True,
        )
    else:
        F = inner_product_orbit_grid(K_u, w_a, w_b, r)
    if is_per:
        integral = float(F.sum() * du)
    else:
        integral = float(np.trapezoid(F, u_grid))
    c = sigma * np.sqrt(2 * np.pi / r)
    value = (sigma * np.sqrt(np.pi)) ** r * integral / c ** 2
    if return_cancellation_ratio:
        # Worst case across u-grid is the relevant signal — even one
        # bad point could dominate the integration if it sits near a
        # peak of the integrand.
        ratio = float(np.min(ratios))
        return value, ratio
    return value



def _cos_sim_exp_tens_sa_orbit(dens_x, dens_y):
    """Compute (ip_xy, ip_xx, ip_yy, worst_ratio) for the SA case via
    Möbius method.

    ``worst_ratio`` is the minimum cancellation ratio across the three
    inner-product computations. Values below ~1e-10 indicate the
    Möbius alternating sum has lost most of its significant digits and
    the dispatcher should fall back to Bulger's method.
    """
    sigma = dens_x.sigma
    r = dens_x.r
    is_rel = dens_x.is_rel
    is_per = dens_x.is_per
    period = dens_x.period
    p_x, w_x = dens_x.p, dens_x.w
    p_y, w_y = dens_y.p, dens_y.w

    if is_rel:
        ip_xy, r_xy = _orbit_inner_rel(
            p_x, w_x, p_y, w_y, sigma, r, is_per, period,
            return_cancellation_ratio=True,
        )
        ip_xx, r_xx = _orbit_inner_rel(
            p_x, w_x, p_x, w_x, sigma, r, is_per, period,
            return_cancellation_ratio=True,
        )
        ip_yy, r_yy = _orbit_inner_rel(
            p_y, w_y, p_y, w_y, sigma, r, is_per, period,
            return_cancellation_ratio=True,
        )
    else:
        ip_xy, r_xy = _orbit_inner_abs(
            p_x, w_x, p_y, w_y, sigma, r, is_per, period,
            return_cancellation_ratio=True,
        )
        ip_xx, r_xx = _orbit_inner_abs(
            p_x, w_x, p_x, w_x, sigma, r, is_per, period,
            return_cancellation_ratio=True,
        )
        ip_yy, r_yy = _orbit_inner_abs(
            p_y, w_y, p_y, w_y, sigma, r, is_per, period,
            return_cancellation_ratio=True,
        )
    return ip_xy, ip_xx, ip_yy, min(r_xy, r_xx, r_yy)



def _cos_sim_exp_tens_sa_pairwise(dens_x, dens_y, *, verbose: bool = True,
                                  truncation_sigmas=None,
                                  kernel_precision=None):
    """Compute (ip_xy, ip_xx, ip_yy) for the SA case via the
    Bulger's method (``_ip_core``).

    This is the body of the original ``_cos_sim_exp_tens_sa``
    factored out so the new dispatcher can route to it cleanly.

    Forwards ``truncation_sigmas`` / ``kernel_precision`` to
    ``_ip_core`` so the helper-accelerated path is reached for the
    abs and rel-non-periodic modes.
    """
    r = dens_x.r
    sigma = dens_x.sigma
    is_rel = dens_x.is_rel
    is_per = dens_x.is_per
    period = dens_x.period

    n_jx, n_kx = dens_x.n_j_perm, dens_x.n_k
    n_jy, n_ky = dens_y.n_j_perm, dens_y.n_k

    total_pairs = n_jx * n_ky + n_jx * n_kx + n_jy * n_ky
    estimate_comp_time(total_pairs, r, "cos_sim_exp_tens", verbose)

    ip_xy = _ip_core(
        dens_x.u_perm, dens_x.w_perm, n_jx,
        dens_y.v_comb, dens_y.wv_comb, n_ky,
        r, sigma, is_rel, is_per, period,
        truncation_sigmas=truncation_sigmas,
        kernel_precision=kernel_precision,
    )
    ip_xx = _ip_core(
        dens_x.u_perm, dens_x.w_perm, n_jx,
        dens_x.v_comb, dens_x.wv_comb, n_kx,
        r, sigma, is_rel, is_per, period,
        truncation_sigmas=truncation_sigmas,
        kernel_precision=kernel_precision,
    )
    ip_yy = _ip_core(
        dens_y.u_perm, dens_y.w_perm, n_jy,
        dens_y.v_comb, dens_y.wv_comb, n_ky,
        r, sigma, is_rel, is_per, period,
        truncation_sigmas=truncation_sigmas,
        kernel_precision=kernel_precision,
    )
    return ip_xy, ip_xx, ip_yy



# -------------------------------------------------------------------
#  Raw SA batched dispatch (used by the polymorphic cos_sim_exp_tens
#  for 2-D pitch-matrix input)
# -------------------------------------------------------------------


def _cos_sim_raw_sa_batch(
    p_mat_a: np.ndarray,
    p_mat_b: np.ndarray,
    sigma: float,
    r: int,
    is_rel: bool,
    is_per: bool,
    period: float,
    is_sym=None,
    *,
    weights_a: np.ndarray | None = None,
    weights_b: np.ndarray | None = None,
    spectrum: list | None = None,
    precision: int | None = None,
    dedup: bool = True,
    method: str = "auto",
    normalize: str = "cosine",
    cancellation_threshold: float = 1e-12,
    verbose: bool = True,
) -> np.ndarray:
    """Raw single-attribute batched dispatch for :func:`cos_sim_exp_tens`.

    Computes cosine similarity for many paired weighted multisets
    (*p* represents pitches or positions). Each row of *p_mat_a*
    and *p_mat_b* defines one pair.

    Two-stage dedup: Phase 1 builds canonical-form keys per side and
    constructs each unique density once (chord-level dedup); Phase 3
    delegates pair-level dedup to the polymorphic
    :func:`cos_sim_exp_tens` (in pairwise list-vs-list mode), which
    in turn threads ``method`` and ``cancellation_threshold`` through
    to the per-pair Möbius-vs-Bulger dispatcher.

    Parameters
    ----------
    p_mat_a, p_mat_b : 2-D arrays
        Multiset values per row. NaN entries are ignored.
    sigma, r, is_rel, is_per, period :
        Tensor parameters.
    weights_a, weights_b : 2-D arrays or None
        Weights matching the corresponding ``p_mat_*``.
    spectrum : list, optional
        Arguments for :func:`~mpt.spectra.add_spectra`.
    precision : int, optional
        Round canonical pitch and weight values to this many decimal
        places, to absorb FP noise when deduplicating.
    dedup : bool, default True
        Apply pair-level canonical-form dedup at Phase 3.
    method : {'auto', 'bulger', 'direct'}, default 'auto'
        Inner-product evaluation path; threaded through to the per-pair
        SA core via the inner ``cos_sim_exp_tens`` call.
    cancellation_threshold : float, default 1e-12
        Orbit-path cancellation guard threshold.
    verbose : bool
        Print progress.

    Returns
    -------
    np.ndarray
        ``(nRows,)`` cosine similarities. NaN for invalid rows.
    """
    p_mat_a = np.asarray(p_mat_a, dtype=np.float64)
    p_mat_b = np.asarray(p_mat_b, dtype=np.float64)

    # The batched path deduplicates rows by a multiset canonical key,
    # which collapses rows that share a multiset but differ in order.
    # That is correct only for the symmetric reading: under [sym]=0 the
    # order is significant, so the dedup would silently merge distinct
    # ordered densities. Reject it rather than return a wrong answer.
    # Order-aware batched dedup is a tracked follow-up; for now use the
    # scalar or density-list forms for ordered densities.
    if (is_sym is not None) and (not bool(np.all(is_sym))) and r > 1:
        raise NotImplementedError(
            "cos_sim_exp_tens batched (2-D) input does not yet support "
            "[sym]=0 (ordered) densities at r > 1: the batched dedup "
            "canonicalises each row's multiset and would merge "
            "order-distinct rows. Build densities individually (scalar "
            "or density-list input) for ordered comparisons."
        )

    if p_mat_a.ndim == 1:
        p_mat_a = p_mat_a.reshape(1, -1)
    if p_mat_b.ndim == 1:
        p_mat_b = p_mat_b.reshape(1, -1)

    n_rows = p_mat_a.shape[0]
    if p_mat_b.shape[0] != n_rows:
        raise ValueError("p_mat_a and p_mat_b must have the same number of rows.")

    use_wa = weights_a is not None
    use_wb = weights_b is not None
    use_spec = spectrum is not None

    if use_wa:
        weights_a = np.asarray(weights_a, dtype=np.float64)
        if weights_a.shape != p_mat_a.shape:
            raise ValueError("weights_a must be the same shape as p_mat_a.")
    if use_wb:
        weights_b = np.asarray(weights_b, dtype=np.float64)
        if weights_b.shape != p_mat_b.shape:
            raise ValueError("weights_b must be the same shape as p_mat_b.")

    if precision is not None:
        p_mat_a = np.round(p_mat_a, precision)
        p_mat_b = np.round(p_mat_b, precision)
        if use_wa:
            weights_a = np.round(weights_a, precision)
        if use_wb:
            weights_b = np.round(weights_b, precision)

    s = np.full(n_rows, np.nan)

    # ── Phase 1: Canonicalise and build individual-set keys ─────────
    key_a: list[tuple | None] = [None] * n_rows
    key_b: list[tuple | None] = [None] * n_rows
    valid = [False] * n_rows

    canon_data_a: dict[tuple, tuple[np.ndarray, np.ndarray | None]] = {}
    canon_data_b: dict[tuple, tuple[np.ndarray, np.ndarray | None]] = {}

    for i in range(n_rows):
        pa_row = p_mat_a[i]
        pb_row = p_mat_b[i]
        mask_a = ~np.isnan(pa_row)
        mask_b = ~np.isnan(pb_row)
        pa_valid = pa_row[mask_a]
        pb_valid = pb_row[mask_b]

        if len(pa_valid) < r or len(pb_valid) < r:
            continue

        wa_valid = weights_a[i, mask_a] if use_wa else None
        wb_valid = weights_b[i, mask_b] if use_wb else None

        ka, kb, ca_p_arr, ca_w_arr, cb_p_arr, cb_w_arr = _pair_canonical_key(
            pa_valid, wa_valid, pb_valid, wb_valid,
            sigma=sigma, r=r, is_rel=is_rel, is_per=is_per, period=period,
            precision=precision,
        )

        key_a[i] = ka
        key_b[i] = kb
        valid[i] = True

        if ka not in canon_data_a:
            canon_data_a[ka] = (ca_p_arr, ca_w_arr)
        if kb not in canon_data_b:
            canon_data_b[kb] = (cb_p_arr, cb_w_arr)

    # ── Phase 2: Build density structs for unique individual sets ───
    dens_cache_a: dict[tuple, object] = {}
    for ka, (p_arr, w_arr) in canon_data_a.items():
        if use_spec:
            p_arr, w_arr = add_spectra(p_arr, w_arr, *spectrum)
        dens_cache_a[ka] = build_exp_tens(
            p_arr, w_arr, sigma, r, is_rel, is_per, period,
            True if is_sym is None else is_sym, verbose=False
        )

    dens_cache_b: dict[tuple, object] = {}
    for kb, (p_arr, w_arr) in canon_data_b.items():
        if use_spec:
            p_arr, w_arr = add_spectra(p_arr, w_arr, *spectrum)
        dens_cache_b[kb] = build_exp_tens(
            p_arr, w_arr, sigma, r, is_rel, is_per, period,
            True if is_sym is None else is_sym, verbose=False
        )

    n_unique_a = len(dens_cache_a)
    n_unique_b = len(dens_cache_b)
    n_valid = sum(valid)

    if verbose:
        print(
            f"cos_sim_exp_tens: {n_rows} rows, {n_valid} valid, "
            f"{n_unique_a} unique A-sets, {n_unique_b} unique B-sets."
        )
        if is_rel:
            print(
                "  Canonicalization: A-sets and B-sets independently "
                "normalized for transposition"
                + (" and octave equivalence." if is_per else ".")
            )
        else:
            print(
                "  Canonicalization: joint co-transposition"
                + (" with octave equivalence" if is_per else "")
                + "; B-set counts reflect position relative to A."
            )
        print(
            f"cos_sim_exp_tens: built {n_unique_a + n_unique_b} "
            f"density structs ({n_unique_a} A + {n_unique_b} B)."
        )

    # ── Phase 3: Compute via polymorphic cos_sim_exp_tens ────────────
    if n_valid == 0:
        if verbose:
            print("cos_sim_exp_tens: done.")
        return s

    valid_indices = [i for i in range(n_rows) if valid[i]]
    list_a_dens = [dens_cache_a[key_a[i]] for i in valid_indices]
    list_b_dens = [dens_cache_b[key_b[i]] for i in valid_indices]

    cos_results = cos_sim_exp_tens(
        list_a_dens, list_b_dens,
        mode="pairwise", dedup=dedup,
        method=method,
        normalize=normalize,
        cancellation_threshold=cancellation_threshold,
        verbose=verbose,
    )

    # ── Phase 4: Map results back ────────────────────────────────────
    for k, idx in enumerate(valid_indices):
        s[idx] = cos_results[k]

    if verbose:
        print("cos_sim_exp_tens: done.")

    return s



# -------------------------------------------------------------------
#  batch_cos_sim_exp_tens (deprecated convenience wrapper)
# -------------------------------------------------------------------


@_with_dispatch_scope
def batch_cos_sim_exp_tens(
    p_mat_a: np.ndarray,
    p_mat_b: np.ndarray,
    sigma: float,
    r: int,
    is_rel: bool,
    is_per: bool,
    period: float,
    *,
    weights_a: np.ndarray | None = None,
    weights_b: np.ndarray | None = None,
    spectrum: list | None = None,
    precision: int | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """Deprecated. Use :func:`cos_sim_exp_tens` directly with 2-D matrices.

    .. deprecated:: 2.1
       The batched-raw-input dispatch has been folded into the unified
       :func:`cos_sim_exp_tens` entry point. Pass 2-D pitch matrices
       directly:

       .. code-block:: python

          # Old:
          s = batch_cos_sim_exp_tens(P1, P2, sigma, r, is_rel, is_per, period,
                                     weights_a=W1, weights_b=W2)
          # New:
          s = cos_sim_exp_tens(P1, W1, P2, W2, sigma, r, is_rel, is_per, period)

       Note the argument order: weights now follow each pitch matrix
       positionally (matching the SA scalar raw form), instead of being
       keyword-only. This shim preserves the old keyword-only weight API
       for backward compatibility but issues a ``DeprecationWarning``.
       This shim will be removed in a future release.
    """
    warnings.warn(
        "batch_cos_sim_exp_tens is deprecated. Pass 2-D pitch matrices "
        "directly to cos_sim_exp_tens (with weights as positional arguments "
        "after each pitch matrix, matching the SA scalar raw form). This "
        "shim will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _cos_sim_raw_sa_batch(
        p_mat_a, p_mat_b, sigma, r, is_rel, is_per, period,
        weights_a=weights_a, weights_b=weights_b,
        spectrum=spectrum, precision=precision,
        verbose=verbose,
    )