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

import math
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

from .build import (
    _enum_flat_attr,
    _looks_like_multi_attr,
    _nested_enum_indices,
    build_exp_tens,
)
from .canonical import _pair_canonical_key
from .density import (
    MaetDensity,
    WindowedMaetDensity,
    _nchoosek_indices,
    is_single_multiset,
)
from .dispatch import (
    _compute_Q,
    _compute_Q_inner_blocks,
    _inner_r_vec,
    _normalize_density_input,
    _orbit_ips_look_corrupted,
    _resolve_list_list_mode,
    _select_ma_inner_product_method,
    # Orbit-table policy constants used by the Möbius-method router.
    _ORBIT_K_MINUS_R_MIN,
    _ORBIT_R_MAX_SHIPPED,
    _ORBIT_SIGMA_OVER_P_THRESHOLD,
    _warn_rel_per_all_image,
)


# -------------------------------------------------------------------
#  Normalisation helpers (shared by cos_sim_exp_tens and
#  windowed_tensor_similarity, which expose the same ``normalize`` keyword
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

    **Raw single-multiset scalar input**:

    - ``cos_sim_exp_tens(p1, w1, p2, w2, sigma, r, is_rel, is_per, period)``
      where ``p1`` and ``p2`` are 1-D arrays of pitches, ``w1``,
      ``w2`` are matching 1-D weight arrays (or ``None`` for uniform).
      Returns scalar.

    **Raw single-multiset batched input** (replaces ``batch_cos_sim_exp_tens``):

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
        2 for density modes; 9 for raw single-multiset modes; 10 for raw MA mode.
    mode : {'auto', 'pairwise', 'cartesian'}, default 'auto'
        For density list-vs-list. Ignored in scalar and broadcast cases.
    dedup : bool, default True
        Apply canonical-form deduplication. Currently supported for
        single-multiset pairs only; pairs involving ``MaetDensity``
        bypass dedup transparently. (``WindowedMaetDensity`` operands
        are rejected at the top of the function — use
        :func:`windowed_tensor_similarity` instead.)
    spectrum : list/tuple, optional
        Per-row spectral augmentation parameters passed to
        :func:`mpt.spectra.add_spectra`. Only valid in raw single-multiset modes
        (scalar or batched).
    precision : int, optional
        Round canonical pitch and weight values to this many decimal
        places, to absorb FP noise when deduplicating. Only valid in
        raw single-multiset batched mode.
    method : {'auto', 'bulger', 'mobius', 'direct'}, default 'auto'
        Inner-product method; threaded through to the per-pair single-multiset/multi-attribute
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
        Scalar in scalar-vs-scalar density mode, raw single-multiset scalar mode, and
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
    batch_cos_sim_exp_tens : deprecated; superseded by raw single-multiset batched input here.

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
        if len(args) != 2:
            raise TypeError(
                f"Density input mode expects 2 positional arguments "
                f"(dens_x, dens_y); got {len(args)}."
            )
        if spectrum is not None:
            raise TypeError(
                "'spectrum' kwarg is only valid in raw single-multiset input mode."
            )
        if precision is not None:
            raise TypeError(
                "'precision' kwarg is only valid in raw single-multiset batched input mode."
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
                "'spectrum' kwarg is only supported in raw single-multiset "
                "input mode."
            )
        if precision is not None:
            raise TypeError(
                "'precision' kwarg is only valid in raw single-multiset batched input mode."
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
        if b is not None and not _looks_like_multi_attr(b):
            raise TypeError(
                "Raw operands must use the same input form: the first "
                "operand is multi-attribute (a list of per-attribute "
                "matrices) but the second is a flat vector. Use the same "
                "form for both, or build each density explicitly with "
                "build_exp_tens."
            )
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

        # Matrix-valued kernel covariance: whiten both operands once
        # (shared geometry, so a single Cholesky factor per attribute)
        # and fall through to the isotropic machinery with sigma = 1.
        # No normalization correction is needed: the tuple-independent
        # prefactor is identical on both sides of every normalization
        # and cancels.
        from .aniso import sigma_vec_has_kernel_cov
        if sigma_vec_has_kernel_cov(args[4]):
            from .build import _resolve_aniso_ma
            from .aniso import whiten_p_attr
            (p1_in, w1_in, p2_in, w2_in) = args[0], args[1], args[2], args[3]
            sigma_vec_in, r_vec_in = args[4], args[5]
            is_rel_in, is_per_in = args[6], args[7]
            is_sym_in = args[9] if len(args) == 10 else None
            probe = p1_in[0] if a_is_list else p1_in
            _, sigma_res, _, chol_list = _resolve_aniso_ma(
                probe, sigma_vec_in, r_vec_in, is_rel_in, is_per_in,
                is_sym_in, None,
            )
            # The resolver validated the constraints and produced the
            # per-attribute Cholesky factors; whiten every structure
            # with them (whiten_values rejects row-count mismatches,
            # which enforces r == K on the remaining operands).
            if a_is_list:
                p1_w = [whiten_p_attr(blk, chol_list) for blk in p1_in]
            else:
                p1_w = whiten_p_attr(p1_in, chol_list)
            if b_is_list:
                p2_w = [whiten_p_attr(blk, chol_list) for blk in p2_in]
            else:
                p2_w = whiten_p_attr(p2_in, chol_list)
            args = (p1_w, w1_in, p2_w, w2_in, sigma_res) + tuple(args[5:])
            a = args[0]

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
                truncation_sigmas=truncation_sigmas,
                kernel_precision=kernel_precision,
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
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Raw single-multiset dispatch.
    # ------------------------------------------------------------------
    if len(args) not in (9, 10):
        raise TypeError(
            f"Raw single-multiset input expects 9 or 10 positional "
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
            f"Raw single-multiset inputs must be 1-D (single chord) or 2-D (batched); "
            f"got P1.ndim = {a_arr.ndim}, P2.ndim = {b_arr.ndim}."
        )

    # Matrix-valued kernel covariance: whiten both operands and fall
    # through to the isotropic machinery with sigma = 1 (prefactors
    # cancel under either normalization).
    from .aniso import is_kernel_cov as _is_kc
    if _is_kc(args[4]):
        from .aniso import validate_kernel_cov, check_aniso_constraints, \
            whiten_values
        if spectrum is not None:
            raise TypeError(
                "'spectrum' is not supported with a matrix-valued kernel "
                "covariance (spectral augmentation changes the multiset "
                "size, breaking r == K)."
            )
        r_in, is_rel_in, is_per_in = args[5], args[6], args[7]
        is_sym_in = args[9] if len(args) == 10 else True
        for nm, arr in (("P1", a_arr), ("P2", b_arr)):
            K_side = arr.shape[-1]
            check_aniso_constraints(
                r=r_in, K=K_side, is_rel=is_rel_in, is_per=is_per_in,
                is_sym=is_sym_in, name=f"sigma ({nm})",
            )
        Sigma_in, R_in = validate_kernel_cov(
            args[4], dim=int(r_in), name="sigma")
        a_arr = (whiten_values(R_in, a_arr) if a_arr.ndim == 1
                 else whiten_values(R_in, a_arr.T).T)
        b_arr = (whiten_values(R_in, b_arr) if b_arr.ndim == 1
                 else whiten_values(R_in, b_arr.T).T)
        args = (a_arr, args[1], b_arr, args[3], 1.0) + tuple(args[5:])

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

        return _cos_sim_raw_single_multiset_batch(
            P1, P2, sigma, r_, is_rel, is_per, period, is_sym,
            weights_a=W1, weights_b=W2,
            spectrum=spectrum, precision=precision,
            dedup=dedup,
            method=method,
            normalize=normalize,
            cancellation_threshold=cancellation_threshold,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
            verbose=verbose,
        )

    # Both operands are 1-D → existing scalar single-multiset path.
    if precision is not None:
        raise TypeError(
            "'precision' kwarg is only valid for raw single-multiset batched input "
            "(at least one of P1, P2 must be 2-D)."
        )
    if mode != "auto":
        raise TypeError(
            "'mode' kwarg only applies to density list inputs."
        )
    return _cos_sim_raw_single_multiset_scalar(
        *args, spectrum=spectrum,
        method=method,
        normalize=normalize,
        cancellation_threshold=cancellation_threshold,
        truncation_sigmas=truncation_sigmas,
        kernel_precision=kernel_precision,
        verbose=verbose,
    )



def _all_single_multiset_pairs(pairs):
    """Return True iff every (a, b) pair consists of two single-multiset
    densities (A = 1, flat), for which canonical-form dedup applies."""
    for a, b in pairs:
        if not (is_single_multiset(a) and is_single_multiset(b)):
            return False
    return True



def _compute_pair_results_with_dedup(
    pairs, *, method: str, normalize: str = "cosine",
    cancellation_threshold: float,
    truncation_sigmas=None, kernel_precision=None, verbose: bool,
):
    """Compute cos_sim for single-multiset density pairs with
    canonical-form dedup. Repeated collections (same canonical chord
    and shared structural parameters) are evaluated once and reused."""
    pair_key_to_idx: dict = {}
    pair_canon_idx: list[int] = []
    unique_pair_list: list = []

    def _fields(d):
        pd, wd = d.p_attr[0][:, 0], d.w[0][:, 0]
        return (
            pd, wd,
            float(d.sigma[0]), int(d.r[0]),
            bool(d.is_rel[0]), bool(d.is_per[0]), float(d.period[0]),
        )

    for a, b in pairs:
        pa, wa, sig_a, r_a, rel_a, per_a, period_a = _fields(a)
        pb, wb, sig_b, r_b, rel_b, per_b, period_b = _fields(b)
        key_a, key_b, _, _, _, _ = _pair_canonical_key(
            pa, wa, pb, wb,
            sigma=sig_a, r=r_a, is_rel=rel_a,
            is_per=per_a, period=period_a,
        )
        pk = (
            key_a,
            key_b,
            (sig_b, r_b, rel_b, per_b, period_b),
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
    # of K <= 5 representative unique pairs, extrapolated over n_unique.
    # Gated on verbose; 10 s silence threshold via
    # maybe_print_batched_estimate.
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
        _cos_sim_pair_core(
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
            _cos_sim_pair_core(
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

    # Main loop over unique pairs.
    unique_results = []
    for up, (a, b) in enumerate(unique_pair_list):
        unique_results.append(_cos_sim_pair_core(
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

    Routes to :func:`_cos_sim_exp_tens_ma`, threading ``method``,
    ``normalize``, ``cancellation_threshold``,
    ``truncation_sigmas`` and ``kernel_precision`` through; that routine
    handles both the single-multiset and multi-attribute cases.
    ``WindowedMaetDensity`` operands are rejected here; user code
    reaches the windowed inner product via :func:`windowed_tensor_similarity`.
    """
    if isinstance(dens_x, WindowedMaetDensity) or \
            isinstance(dens_y, WindowedMaetDensity):
        raise TypeError(
            "cos_sim_exp_tens does not accept WindowedMaetDensity "
            "operands. Use windowed_tensor_similarity(dens_context, "
            "dens_query, window_spec, offsets) — pass a single-column "
            "offsets array for the scalar single-offset case, or a "
            "(dim, M) array for the M-offset sweep."
        )
    from .aniso import density_has_kernel_cov, density_kernel_covs_compatible
    if density_has_kernel_cov(dens_x) or density_has_kernel_cov(dens_y):
        if not density_kernel_covs_compatible(dens_x, dens_y):
            raise ValueError(
                "The two densities were built with different kernel "
                "covariances (or one with a matrix-valued sigma and one "
                "without); inner products require a shared kernel per "
                "attribute."
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
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
            verbose=verbose,
        )
    raise TypeError(
        f"Both arguments must be MaetDensity or "
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

    if dedup and _all_single_multiset_pairs(pairs):
        results = _compute_pair_results_with_dedup(
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
                "multi-attribute densities; computing without dedup."
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



def _cos_sim_raw_single_multiset_scalar(
    p1, w1, p2, w2,
    sigma, r, is_rel, is_per, period, is_sym=None,
    *,
    spectrum=None,
    method: str = "auto",
    normalize: str = "cosine",
    cancellation_threshold: float = 1e-12,
    truncation_sigmas: float | None = None,
    kernel_precision: str | None = None,
    verbose: bool = True,
) -> float:
    """Raw single-multiset scalar dispatch for :func:`cos_sim_exp_tens`."""
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
        truncation_sigmas=truncation_sigmas,
        kernel_precision=kernel_precision,
        verbose=verbose,
    )



def _cos_sim_raw_ma_scalar(
    p_attr1, w1, p_attr2, w2,
    sigma_vec, r_vec, is_rel_vec, is_per_vec, period_vec, is_sym_vec=None,
    *,
    method: str = "auto",
    normalize: str = "cosine",
    cancellation_threshold: float = 1e-12,
    truncation_sigmas: float | None = None,
    kernel_precision: str | None = None,
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
        truncation_sigmas=truncation_sigmas,
        kernel_precision=kernel_precision,
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
    truncation_sigmas: float | None = None,
    kernel_precision: str | None = None,
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
                truncation_sigmas=truncation_sigmas,
                kernel_precision=kernel_precision,
                verbose=False,
            )
        else:
            out[m] = _cos_sim_pair_core(
                dens_m, dens_scalar,
                method=method,
                normalize=normalize,
                cancellation_threshold=cancellation_threshold,
                truncation_sigmas=truncation_sigmas,
                kernel_precision=kernel_precision,
                verbose=False,
            )
    return out



# -------------------------------------------------------------------
#  _cos_sim_exp_tens_ma  (multi-attribute inner product; orbit constants)
# -------------------------------------------------------------------


_ORBIT_GRID_SLAB_ELEMS = 1 << 19
"""Maximum kernel entries per translation-grid slab in the rel-mode
orbit inner product (~4 MB of doubles). The contraction makes several
permute and power copies of each slab, so the live working set is a
small multiple of this; the value keeps it memory-resident on typical
hardware, where the contraction's per-op cost is flat in K. Slab count
grows only the loop overhead, which is negligible against the per-slab
contraction work."""


_ORBIT_CANCELLATION_RATIO_MIN = 1e-10
"""Minimum acceptable cancellation ratio in the Möbius method's alternating partition sum.

When ``|sum| / max(|term|)`` drops below this threshold the result has
lost roughly 10 of its 16 significant decimal digits, leaving ~6
surviving — borderline acceptable for cosine accuracy at downstream
1e-6 user tolerance, but past this point the dispatcher falls back to
Bulger's method. See V22_DEV_LOG.md for the empirical regime where this
fires (sharp Gaussians + low K-r margin in absolute modes)."""



def _cos_sim_exp_tens_ma(
    dens_x: MaetDensity,
    dens_y: MaetDensity,
    *,
    method: str = "auto",
    normalize: str = "cosine",
    cancellation_threshold: float = 1e-12,
    truncation_sigmas=None,
    kernel_precision=None,
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

    # An empty operand has no events to overlap, so the inner product -- and
    # hence the similarity -- is zero. A windowed carrier whose window caught
    # nothing prunes to zero events here; without this guard it reaches the
    # nested contraction's value-range scan, which has no identity over an
    # empty attribute column. (The raw single-multiset path is unaffected: it
    # is reached only without specs, and an empty windowed carrier always
    # carries specs.)
    if dens_x.n == 0 or dens_y.n == 0:
        return 0.0

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

    if method not in ("auto", "bulger", "direct", "mobius", "contract",
                      "factored"):
        raise ValueError(
            f"method must be one of 'auto', 'bulger', 'direct', 'mobius', "
            f"'contract', 'factored'; got {method!r}."
        )

    # Explicit factored-cull route: computes the triple through the
    # per-attribute / per-event-pair factorisation without materialising
    # the joint tuple set. Covers every mode except relative-periodic
    # (minimum-image), whose per-slot factor the culled helper cannot take.
    if method == "factored":
        if not _ma_factored_ip_supported(dens_x, dens_y):
            raise ValueError(
                "method='factored' does not support relative-and-periodic "
                "attributes under the minimum-image convention; use 'auto', "
                "'bulger', or 'mobius'."
            )
        from .._defaults import _maybe_show_dispatch_msg
        _maybe_show_dispatch_msg("cos_sim_exp_tens", "factored", "user")
        ip_xy, ip_xx, ip_yy = _cos_sim_exp_tens_ma_factored(
            dens_x, dens_y, verbose=verbose,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
        )
        return _finalise_normalisation(ip_xy, ip_xx, ip_yy, normalize)

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

    # Per-attribute vectors for the Möbius-side cost model: which
    # attributes are relative, and each one's translation-grid node
    # estimate (matching the grid rules of the batched rel helper; 1
    # for absolute attributes, where no grid exists).
    from ._nested_contraction import auto_ntau_default
    rel_vec = np.array([bool(is_rel[a]) for a in range(A)], dtype=bool)
    nu_vec = np.ones(max(A, 1))[:A]
    for a in range(A):
        if not rel_vec[a] or int(r_vec[a]) < 2:
            continue
        if bool(is_per[a]):
            nu_vec[a] = auto_ntau_default(
                float(period[a]), float(sigma[a]))
        else:
            Pxa = dens_x.p_attr[a]
            Pya = dens_y.p_attr[a]
            from .._defaults import get_default as _gd
            _margin = _rel_window_margin(_gd('truncation_sigmas'))
            span = (float(np.nanmax(Pxa) - np.nanmin(Pxa))
                    + float(np.nanmax(Pya) - np.nanmin(Pya))
                    + 2.0 * _margin * float(sigma[a]))
            nu_vec[a] = max(
                64,
                int(np.ceil(max(span, 1.0) / float(sigma[a]) * 10.0)),
            )

    # Nested densities route through the hierarchical contraction
    # (_try_nested_contract), not the flat Bulger pairwise path, so the
    # flat forced-Bulger feasibility guard must not fire for them. Detect
    # nesting before the selector runs.
    nested_x = getattr(dens_x, "nested", None)
    nested_y = getattr(dens_y, "nested", None)
    nested_any = (
        (nested_x is not None and any(s is not None for s in nested_x))
        or (nested_y is not None and any(s is not None for s in nested_y))
    )

    chosen = _select_ma_inner_product_method(
        r_vec=r_vec, k_vec=k_vec, A=A,
        N_x=int(dens_x.n), N_y=int(dens_y.n),
        any_per=any_per,
        any_rel_nonper=any_rel_nonper,
        any_rel_per=any_rel_per,
        sigma_over_P_max=sop_max,
        user_method=method,
        rel_vec=rel_vec, nu_vec=nu_vec,
        guard_forced_bulger=not nested_any,
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

    # Nested attributes are not handled by the flat orbit/Möbius entry
    # point: that path would have to flatten the levels into one slot set,
    # but the inner unit's metric is block-diagonal (slots couple only
    # within an aligned inner unit), which the flat re-enumeration cannot
    # represent. Route instead to the hierarchical contraction, which
    # contracts the tag tree level by level and itself selects the orbit
    # (Möbius) reduction or permutation/combination enumeration per level
    # (see _nested_contraction.build_recipe); it is not enumeration-only.
    if method == "contract" and not nested_any:
        raise ValueError(
            "method='contract' applies to a nested attribute only; use "
            "'auto' or 'bulger' for non-nested densities."
        )
    if nested_any:
        if method in ("auto", "contract"):
            triple = _try_nested_contract(
                dens_x, dens_y, normalize=normalize, verbose=verbose,
                force=(method == "contract"))
            if triple is not None:
                return _finalise_normalisation(*triple, normalize)
            # When forced, _try_nested_contract raises on any uncovered case,
            # so a None here means method == "auto" chose the centres path.
        chosen = "bulger"

    # Dispatch-decision message: announce which inner-product path ran,
    # matching the single-multiset path's behaviour. Gated by the
    # toolbox-wide show_hints flag and throttled once per
    # (function, chosen) per top-level call; not gated by per-call
    # verbose. The multi-attribute selector does not run the empirical
    # probe, so no time estimate accompanies the message.
    from .._defaults import _maybe_show_dispatch_msg
    _maybe_show_dispatch_msg(
        "cos_sim_exp_tens", chosen, "ma_select",
    )

    if chosen == "mobius":
        ip_xy, ip_xx, ip_yy = _cos_sim_exp_tens_ma_orbit(
            dens_x, dens_y,
        )
        # Two layers of Möbius-method result validation, fall back on either.
        # The per-entry worst_ratio diagnostic that previously gated
        # this fallback (analogous to the single-multiset case) was found to fire
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
                truncation_sigmas=truncation_sigmas,
                kernel_precision=kernel_precision,
            )
    else:  # 'bulger' or 'direct' (coincide in MA mode)
        ip_xy, ip_xx, ip_yy = _cos_sim_exp_tens_ma_pairwise(
            dens_x, dens_y, verbose=verbose,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
        )

    return _finalise_normalisation(ip_xy, ip_xx, ip_yy, normalize)



def _ip_core_ma(
    u_cell, w_u, n_j, v_cell, w_v, n_k,
    A, r_vec, sigma, is_rel, is_per, period,
    *, truncation_sigmas=None, kernel_precision=None, inner_r=None,
):
    """MA inner product with memory-aware chunking along the comb side.

    Peak per-chunk memory is dominated by the largest per-attribute
    (r_a, nJ, nKc) difference tensor. Use ``(max(r_a) + 2) * nJ * 8``
    bytes per K-column as the sizing heuristic.

    ``truncation_sigmas`` is honoured via log-space thresholding on
    the accumulated MA log-kernel; ``None`` resolves to the global
    default ``mpt.get_default('truncation_sigmas')``.

    For a single attribute whose quadratic form the Gaussian
    kernel-sum helper supports (any absolute mode, or relative
    non-periodic), and when truncation or single precision is
    actually requested, the inner product is routed through
    :func:`_ip_via_helper`. That helper carries a spatial-index
    truncation that both honours the requested accuracy floor exactly
    and gives a substantial speedup on harmonic-template-sized
    collections; the log-kernel accumulation below serves the
    remaining forms (relative-periodic, multi-attribute, or the
    exact untruncated double-precision default).
    """
    from .._defaults import resolve_truncation_sigmas

    # Resolve None -> global default; inf -> accuracy-floor width. After
    # resolution, ``truncation_sigmas`` is always a finite positive
    # float --- truncation always applies, so the single-attribute
    # spatial-index helper (which honours truncation exactly and gives
    # a substantial speed-up on harmonic-template-sized collections) is
    # always the preferred route where it applies.
    truncation_sigmas = resolve_truncation_sigmas(truncation_sigmas)

    single_attr_helper_ok = (
        A == 1
        and (inner_r is None or int(inner_r[0]) == 0)
        and not (bool(is_rel[0]) and bool(is_per[0]))
    )
    if single_attr_helper_ok:
        return _ip_via_helper(
            u_cell[0], w_u, v_cell[0], w_v,
            int(r_vec[0]), float(sigma[0]),
            bool(is_rel[0]), bool(is_per[0]), float(period[0]),
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
        )

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
#  where I_a(n_X, n_Y) is an single-multiset-shaped Möbius inner product over
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
#  Per-attribute Möbius calls apply the single-multiset convention's
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

    When ``truncation_sigmas`` is ``None`` it resolves against the
    global default. The "exact" sentinel ``math.inf`` resolves to the
    finite accuracy-floor width (:func:`mpt._defaults.accuracy_floor_sigmas`,
    default 1e-12), uniformly with every other truncation path, so
    truncation always applies.
    """
    from .._defaults import truncation_ip_sqdist
    cutoff = truncation_ip_sqdist(truncation_sigmas, sigma)
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

    When ``truncation_sigmas`` is ``None`` it resolves against the
    global default. The "exact" sentinel ``math.inf`` resolves to the
    finite accuracy-floor width, uniformly with every other truncation
    path, so truncation always applies.
    """
    from .._defaults import resolve_truncation_sigmas
    truncation_sigmas = resolve_truncation_sigmas(truncation_sigmas)
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
      :func:`_inner_product_direct_abs`, which is exact for any
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
    :func:`_batched_direct_enum_abs`, and rel-per via
    :func:`_ma_per_attr_inner_matrix_rel`): kernel entries whose
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

    # --- r >= 2 rel: batched translation-grid integration, zero-pad ---
    if is_rel:
        Px_, Wx_, Py_, Wy_ = _zero_pad_nan(Px, Wx, Py, Wy)
        return _ma_per_attr_inner_matrix_rel(
            Px_, Wx_, Py_, Wy_, sigma, r, is_per, period,
            return_cancellation_ratio=return_cancellation_ratio,
            truncation_sigmas=truncation_sigmas,
        )

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
    # (one ``_inner_product_direct_abs`` call per pair); for
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
    :func:`_batched_direct_enum_abs`; ``None`` resolves to the
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
            sub_ip = _batched_direct_enum_abs(
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



def _batched_direct_enum_abs(
    Px_group, Wx_group, Py_group, Wy_group,
    sigma, r, is_per, period,
    *, truncation_sigmas=None,
):
    """Batched direct r-tuple enumeration IP for groups at fixed K_x, K_y.

    Vectorised replacement for repeated calls to
    :func:`_inner_product_direct_abs` when every event in
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



def _ma_per_attr_inner_matrix_rel(
    Px, Wx, Py, Wy, sigma, r, is_per, period,
    *, return_cancellation_ratio=False, truncation_sigmas=None,
    samples_per_sigma=10,
):
    """Batched relative-mode case of ``_ma_per_attr_inner_matrix``
    (periodic and non-periodic), all (event_X, event_Y) pairs at once.

    Every pair's inner product marginalises a translation u over a
    grid. In periodic mode the grid is the shared uniform grid over
    ``[0, P)`` with ``auto_ntau_default(period, sigma)`` nodes — the
    single node-count source shared with the flat single-multiset and
    nested relative-periodic paths, so the same level returns the same
    value whether reached flat or nested. In non-periodic mode each
    pair uses a grid of the same shape *centred on its own mean
    offset*: by translation invariance the integrand for pair
    (n_X, n_Y) depends on u only through u + (mean_Y - mean_X), so
    shifting each pair's window by that offset lets all pairs share
    one grid whose extent is the maximum within-pair spread plus the
    margin of :func:`_rel_window_margin` per side, rather than the
    global span of the data. That margin places every kernel entry
    strictly outside the truncation radius at the window edges, so the
    endpoint integrand is exactly zero for any finite truncation and
    the plain Riemann sum equals the trapezoidal rule exactly.

    The pair-and-grid batch is processed in slabs of at most
    ``_ORBIT_GRID_SLAB_ELEMS`` kernel entries, built directly in the
    contraction's (batch, K, K) layout — no transposition copies — so
    the working set stays memory-resident and the per-op cost of the
    batched Möbius contraction is flat in N and K (mirroring the
    single-multiset slabbing in ``_orbit_inner_rel``).

    Ragged (NaN-padded) events arrive zero-padded: a zero-weight slot
    contributes a zero factor to every Möbius term in which its axis
    value appears, so the result is exact for the K_eff events.
    Events with K_eff - r below the precision margin in this regime
    may lose precision in the alternating sum; ``method='auto'``
    routing accounts for this via the dispatcher's precision guard.

    With ``return_cancellation_ratio=True``, additionally returns the
    minimum across event pairs of the per-pair mass-aware cancellation
    diagnostic ``|sum_u F_u| / sum_u max_orb(|term_orb_u|)`` — the
    magnitude of each pair's integrated alternating sum relative to
    the integral of its worst-magnitude partition term, which bounds
    the relative error of that pair's integral. A pointwise worst-case
    over individual (pair, u) cells is the wrong aggregate: at sharp
    sigma, translation bands where the true integrand is exactly zero
    arise from exact cancellation of nonzero orbit terms, and such a
    band's pointwise ratio is ~0 while contributing nothing to any
    integral. Pairs with zero accumulated term mass report 1.0.

    ``truncation_sigmas`` is honoured on the per-u kernel tensor;
    ``None`` resolves to ``mpt.get_default('truncation_sigmas')``.
    """
    from .._mobius import (
        inner_product_orbit_grid,
        inner_product_orbit_pw_batched,
    )
    from .._defaults import get_default
    from ._nested_contraction import auto_ntau_default

    if truncation_sigmas is None:
        truncation_sigmas = get_default('truncation_sigmas')

    K_x, N_x = Px.shape
    K_y, N_y = Py.shape

    # Shared-weights fast path: when every event carries the same
    # weight vector on this attribute (the common case — uniform
    # weights, no ragged padding), all batch cells share w_A and w_B,
    # so the cheaper shared-weights grid contraction applies (the
    # per-batch-weights variant prepends the batch index to every
    # weight operand of the einsum, which costs ~2-3x per kernel op).
    shared_w = (np.all(Wx == Wx[:, :1]) and np.all(Wy == Wy[:, :1]))

    if is_per:
        N_u = auto_ntau_default(period, sigma)
        u_grid = np.linspace(0.0, period, N_u, endpoint=False)
        du = period / N_u
        centres = np.zeros((N_x, N_y), dtype=np.float64)
    else:
        # Per-pair centred common grid. Weighted-slot means keep the
        # centre finite for zero-padded events (all-zero-weight events
        # contribute nothing regardless of centre).
        def _col_means(P, W):
            wsum = W.sum(axis=0)
            safe = np.where(wsum > 0, wsum, 1.0)
            return (P * W).sum(axis=0) / safe

        mx = _col_means(Px, Wx)
        my = _col_means(Py, Wy)
        centres = my[None, :] - mx[:, None]          # (N_x, N_y)

        def _spread(P, W):
            masked = np.where(W > 0, P, np.nan)
            lo = np.nanmin(masked, axis=0)
            hi = np.nanmax(masked, axis=0)
            s = hi - lo
            return np.where(np.isfinite(s), s, 0.0)

        margin = _rel_window_margin(truncation_sigmas)
        span = (float(np.max(_spread(Px, Wx)) + np.max(_spread(Py, Wy)))
                + 2.0 * margin * sigma)
        N_u = max(
            64,
            int(np.ceil(max(span, 1.0) / sigma * samples_per_sigma)),
        )
        u_grid = np.linspace(-0.5 * span, 0.5 * span, N_u)
        du = span / (N_u - 1)

    PxT = np.ascontiguousarray(Px.T)                 # (N_x, K_x)
    PyT = np.ascontiguousarray(Py.T)                 # (N_y, K_y)
    WxT = np.ascontiguousarray(Wx.T)
    WyT = np.ascontiguousarray(Wy.T)

    integral = np.zeros((N_x, N_y), dtype=np.float64)
    worst_ratio = 1.0   # min over pairs of the per-pair mass-aware ratio

    # Slab sizing: one u-node across a row-chunk of pairs, widened in u
    # while the kernel slab stays under the element budget.
    per_pair = K_x * K_y
    nc_x = max(1, min(N_x, _ORBIT_GRID_SLAB_ELEMS // max(N_y * per_pair, 1)))
    for n_start in range(0, N_x, nc_x):
        n_end = min(n_start + nc_x, N_x)
        nc = n_end - n_start
        n_pairs = nc * N_y
        n_uc = max(1, _ORBIT_GRID_SLAB_ELEMS // max(n_pairs * per_pair, 1))

        # Base differences and per-pair centres for this row chunk,
        # flattened to (pairs, K_x, K_y) so the u broadcast below is 4-D
        # with no singleton axes (cheaper numpy staging than the
        # equivalent 5-D form). The two collections may differ in size,
        # so the A-side carries K_x pitches and the B-side K_y.
        base = (PxT[n_start:n_end, None, :, None]
                - PyT[None, :, None, :]
                + centres[n_start:n_end, :, None, None]
                ).reshape(n_pairs, K_x, K_y)
        if not shared_w:
            w_A = np.broadcast_to(
                WxT[n_start:n_end, None, :], (nc, N_y, K_x),
            ).reshape(n_pairs, K_x)
            w_B = np.broadcast_to(
                WyT[None, :, :], (nc, N_y, K_y),
            ).reshape(n_pairs, K_y)

        F_sum = np.zeros(n_pairs, dtype=np.float64)
        mass_sum = np.zeros(n_pairs, dtype=np.float64)
        for u_start in range(0, N_u, n_uc):
            u_end = min(u_start + n_uc, N_u)
            u_s = u_grid[u_start:u_end]
            nu = u_end - u_start
            diffs = base[None, :, :, :] + u_s[:, None, None, None]
            if is_per:
                diffs = diffs - period * np.floor(diffs / period + 0.5)
            K_uc = _trunc_kernel_exp(diffs ** 2, sigma, truncation_sigmas)
            K_uc = K_uc.reshape(nu * n_pairs, K_x, K_y)
            if shared_w:
                if return_cancellation_ratio:
                    flat, _, mass = inner_product_orbit_grid(
                        K_uc, Wx[:, 0], Wy[:, 0], r,
                        return_cancellation_ratio=True,
                        return_term_mass=True,
                    )
                else:
                    flat = inner_product_orbit_grid(K_uc, Wx[:, 0], Wy[:, 0], r)
                    mass = None
            else:
                w_A_uc = np.broadcast_to(
                    w_A[None, :, :], (nu, n_pairs, K_x),
                ).reshape(nu * n_pairs, K_x)
                w_B_uc = np.broadcast_to(
                    w_B[None, :, :], (nu, n_pairs, K_y),
                ).reshape(nu * n_pairs, K_y)
                if return_cancellation_ratio:
                    flat, _, mass = inner_product_orbit_pw_batched(
                        K_uc, w_A_uc, w_B_uc, r, prefactor=1.0,
                        return_cancellation_ratio=True,
                        return_term_mass=True,
                    )
                else:
                    flat = inner_product_orbit_pw_batched(
                        K_uc, w_A_uc, w_B_uc, r, prefactor=1.0,
                    )
                    mass = None
            if mass is not None:
                mass_sum += mass.reshape(nu, n_pairs).sum(axis=0)
            F_sum += flat.reshape(nu, n_pairs).sum(axis=0)

        if return_cancellation_ratio:
            with np.errstate(divide='ignore', invalid='ignore'):
                pair_ratios = np.where(
                    mass_sum > 0, np.abs(F_sum) / mass_sum, 1.0,
                )
            cmin = float(np.min(pair_ratios)) if pair_ratios.size else 1.0
            if cmin < worst_ratio:
                worst_ratio = cmin
        integral[n_start:n_end, :] = F_sum.reshape(nc, N_y) * du

    c = sigma * np.sqrt(2 * np.pi / r)
    out = (sigma * np.sqrt(np.pi)) ** r * integral / c ** 2
    if return_cancellation_ratio:
        return out, worst_ratio
    return out


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

        if _ma_rel_attr_prefers_centres(
            Px, Py, sigma, r_a, is_rel, is_per, period,
        ):
            # Relative attribute at small K: the pairwise closed form
            # over materialised tuple-centres ((r!·C(K, r))² kernel ops
            # per event pair) undercuts the translation-grid
            # contraction (N_u·K² ops per pair) by orders of
            # magnitude, and below the σ/P measure threshold the
            # minimum-image and all-image readings coincide. The
            # per-attribute constant prefactor dropped by the closed
            # form multiplies all three matrices of this attribute
            # identically, so it cancels in every supported
            # normalisation. Centres bundles are built once per
            # density and shared by the cross and self matrices.
            cx = _closed_form_attr_centres(dens_x, a)
            cy = _closed_form_attr_centres(dens_y, a)
            I_xy = _closed_form_attr_matrix_from(cx, cy)
            I_xx = _closed_form_attr_matrix_from(cx, cx)
            I_yy = _closed_form_attr_matrix_from(cy, cy)
        else:
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


def _rel_window_margin(truncation_sigmas):
    """Margin, in sigmas per side, of the non-periodic relative
    translation window.

    The inner-product kernel is G(d; sigma*sqrt(2)) (a convolution of
    two sigma-kernels), and the toolbox truncation zeroes entries whose
    kernel value falls below exp(-t^2 / 2), i.e. at distances beyond
    sqrt(2)*t*sigma.

    The margin's contract is subordination to the kernel-truncation
    contract: the toolbox's accuracy guarantee is stated in
    truncation-sigmas terms (t = 6 accepts ~2e-8 worst-case error;
    t = 8 sits below the 1e-12 cross-language parity floor), so the
    window need only keep its own error comfortably below the kernel
    error the caller has already accepted. A margin of 8 achieves that
    for every t: each kernel factor is at least e^-16 down at the
    window edge, an r-tuple term needs r such factors, so the window
    tail is bounded near 1e-14 relative independent of t (and measures
    at ~1e-27 in practice). The rule is therefore
    ``min(sqrt(2)*t + 0.1, 8)``: for t below ~5.59 the integrand's
    compact support (which ends exactly sqrt(2)*t*sigma beyond the
    extreme pair difference) fits inside the cap, so the cheaper exact
    window is taken — the endpoint integrand vanishes identically and
    the plain Riemann sum equals the trapezoidal rule exactly (the
    0.1-sigma clearance is needed because the extreme pair sits
    exactly at margin*sigma at the endpoint and the truncation mask is
    inclusive; it exceeds the floating-point wobble of the span
    arithmetic by ~12 orders of magnitude). For larger or disabled t
    the margin caps at the empirically validated 8, where the window
    error is subdominant to the kernel contract at every t.
    """
    if truncation_sigmas is None or not np.isfinite(truncation_sigmas):
        return 8.0
    return min(np.sqrt(2.0) * float(truncation_sigmas) + 0.1, 8.0)


def _ma_rel_attr_prefers_centres(Px, Py, sigma, r_a, is_rel, is_per, period):
    """True when a relative attribute's inner matrices should use the
    pairwise closed form over tuple-centres rather than the
    translation-grid contraction.

    The centres route costs (r_a!·C(K, r_a))² kernel ops per event
    pair; the grid route costs N_u·K² (N_u from the shared node-count
    source in periodic mode, or the centred-window rule in
    non-periodic mode). The centres route is chosen when it is
    cheaper AND measure-safe: it computes the minimum-image
    relative-periodic reading (the toolbox's defined measure), which
    coincides with the grid route's all-image reading only below the
    σ/P threshold — above it the grid route is kept so an explicit
    ``method='mobius'`` opt-in preserves the all-image measure. The
    non-periodic closed form is exact, so it is always measure-safe.
    """
    import math
    from .dispatch import _ORBIT_SIGMA_OVER_P_THRESHOLD
    from ._nested_contraction import auto_ntau_default

    if not is_rel or r_a < 2:
        return False
    if is_per and (sigma / period) > _ORBIT_SIGMA_OVER_P_THRESHOLD:
        return False
    K = int(Px.shape[0])
    if K < r_a:
        return False
    centres_pair_ops = (math.factorial(r_a) * math.comb(K, r_a)) ** 2
    if is_per:
        n_u = auto_ntau_default(period, sigma)
    else:
        from .._defaults import get_default
        margin = _rel_window_margin(get_default('truncation_sigmas'))
        span = (float(np.nanmax(Px) - np.nanmin(Px))
                + float(np.nanmax(Py) - np.nanmin(Py))
                + 2.0 * margin * sigma)
        n_u = max(64, int(np.ceil(max(span, 1.0) / sigma * 10.0)))
    grid_pair_ops = n_u * K * K
    return centres_pair_ops < grid_pair_ops



def _closed_form_attr_centres(dens, a):
    """Materialised tuple-centres and metric parameters for attribute ``a``,
    rebuilt in isolation as a single-multiset density.

    The attribute's expectation tensor is a finite Gaussian mixture over its
    materialised perm-side tuple-centres (the same reduction the entropy
    reading uses), so its inner product with another attribute is the pairwise
    Gaussian overlap of those centres. This is exact for the absolute and
    absolute-periodic readings, the analytic relative quadratic (also exact)
    for relative-non-periodic, and the minimum-image pairwise-wrap for
    relative-periodic -- a deliberate measure choice, not the all-image torus
    inner product the tau-grid contraction computes (consistent with the flat
    per-attribute matrix; see the note in
    :func:`_closed_form_attr_matrix_from`).

    Returns ``(centres, w_j, event_of_j, inner_block_size, is_per, period,
    r_a, is_rel, sigma, n_events)``. Variable-K (NaN-padded) slots are carried
    by the rebuild, which drops every tuple touching a padded slot to zero
    weight.
    """
    from .build import build_exp_tens as _bld
    nested = getattr(dens, "nested", None)
    spec = nested[a] if nested is not None else None
    sigma = float(dens.sigma[a])
    is_per = bool(dens.is_per[a])
    period = float(dens.period[a])
    if spec is not None:
        da = _bld([dens.p_attr[a]], [dens.w[a]], specs=[spec],
                  sigma=[sigma], is_per=[is_per], period=[period],
                  verbose=False)
    else:
        is_sym_vec = np.asarray(
            getattr(dens, "is_sym", np.ones(int(dens.n_attrs), dtype=bool))
        ).ravel()
        da = _bld([dens.p_attr[a]], [dens.w[a]], [sigma], [int(dens.r[a])],
                  [bool(dens.is_rel[a])], [is_per], [period],
                  [bool(is_sym_vec[a])], verbose=False)
    return (da.centres[0], da.w_j, da.event_of_j, int(_inner_r_vec(da)[0]),
            is_per, period, int(da.r[0]), bool(da.is_rel[0]), sigma,
            int(da.n))


def _closed_form_attr_matrix_from(cx, cy):
    """(N_x, N_y) per-attribute inner matrix from precomputed tuple-centres.

    The full pairwise centre-overlap is formed as one ``(n_jx, n_jy)`` array
    and aggregated to events by two incidence matmuls, vectorising the
    event-pair grid that the per-event-pair contraction loops scalar-wise. The
    constant per-attribute Gaussian prefactor is dropped: it is identical
    across this matrix and the self matrices, so it cancels in the cosine and
    one-sided ratios.

    Relative-periodic uses the minimum-image pairwise-wrap metric of
    :func:`_compute_Q` (exactly period-shift invariant, and matching the flat
    per-attribute path). This is the toolbox's defined relative-periodic
    measure. It does not equal the all-image transposition average (the torus
    inner product), which the tau-grid contraction computes; the two coincide
    for sigma << period and diverge as sigma approaches the period. Because the
    minimum-image kernel couples all slots within a tuple (a pairwise quadratic
    form), it does not factor per level, so the per-level orbit (Möbius)
    reduction is unavailable here and a symmetric level is enumerated over its
    full orbit. When that enumeration becomes the dominant cost, the dispatch
    in :func:`_nested_attr_matrix_dispatch` routes the attribute to the
    all-image tau-grid contraction instead -- a deliberate measure change,
    documented there.
    """
    (Cx, Wx, Ex, bs, is_per, period, r_a, is_rel, sigma, Nx) = cx
    (Cy, Wy, Ey, _bs, _ip, _pe, _ra, _ir, _sg, Ny) = cy
    Wx = np.ones(Cx.shape[1]) if Wx is None else np.asarray(Wx, float).ravel()
    Wy = np.ones(Cy.shape[1]) if Wy is None else np.asarray(Wy, float).ravel()
    d = Cx.shape[0]
    njx, njy = Cx.shape[1], Cy.shape[1]
    Ex = np.asarray(Ex)
    Ey = np.asarray(Ey)
    # Y-side incidence (njy, Ny): njy is a contraction axis, summed by event.
    GY = np.zeros((njy, Ny))
    GY[np.arange(njy), Ey] = 1.0
    inv4s2 = 1.0 / (4.0 * sigma ** 2)
    # Chunk the X-tuple axis so the pairwise difference array never exceeds a
    # fixed budget: the full (njx, njy) overlap is materialised only one
    # |chunk| x njy block at a time, which keeps the materialised-centre path
    # within memory for large-K factors while the per-event aggregation stays
    # exact. For the common small-tuple case (one chunk) this is identical to
    # the unchunked form.
    chunk = max(1, min(njx, int(16_000_000 // max(njy * max(d, 1), 1))))
    M = np.zeros((Nx, Ny))
    for s in range(0, njx, chunk):
        e = min(s + chunk, njx)
        D = Cx[:, s:e, None] - Cy[:, None, :]
        if bs >= 2:
            Q = _compute_Q_inner_blocks(D, bs, is_per, period, reduced=True)
        else:
            if is_per and not is_rel:
                D = D - period * np.floor(D / period + 0.5)
            Q = _compute_Q(D, r_a, is_rel, is_per, period, reduced=is_rel)
        ov = (Wx[s:e, None] * Wy[None, :]) * np.exp(-Q * inv4s2)
        # X-side incidence matmul (mirror of the MATLAB implementation):
        # scatter-adds row-by-row are far slower than aggregating the
        # chunk with a second incidence product.
        nc = e - s
        GX = np.zeros((Nx, nc))
        GX[Ex[s:e], np.arange(nc)] = 1.0
        M += GX @ (ov @ GY)
    return M


def _attr_value_range(dens_x, dens_y, a):
    """(vmin, vmax) over both densities' slot values for attribute ``a``,
    ignoring NaN padding -- the span the relative-non-periodic translation grid
    must cover."""
    px = np.asarray(dens_x.p_attr[a], dtype=np.float64)
    py = np.asarray(dens_y.p_attr[a], dtype=np.float64)
    return float(min(np.nanmin(px), np.nanmin(py))), \
        float(max(np.nanmax(px), np.nanmax(py)))


def _rel_contract_cheaper(spec_x, spec_y, sigma, period, is_per, vmin, vmax):
    """True when a relative attribute's per-level contraction is cheaper than
    its materialised-centres path.

    The relative kernel couples all slots within a tuple, so the centres path
    must materialise every tuple -- including each symmetric level's full orbit,
    and every selection at an ``r = 1`` level compounded across the ordered
    positions above it (the spectral-cell case: one partial per note across the
    cell). The contraction sidesteps this: each transposition/translation node
    is a one-body product, so a symmetric level reduces by the orbit (Möbius)
    and an ``r = 1`` level by an einsum, never materialising the tuple set. This
    compares the costs directly -- materialised pairwise tuples (centres)
    against quadrature nodes times per-level contraction work -- using the
    analytic :func:`tuple_counts` and :func:`recipe_work`, deterministic in both
    languages.

    For relative-non-periodic the two routes compute the same measure (the
    translation grid converges to the analytic relative quadratic), so this is a
    pure speed choice. For relative-periodic the contraction is the all-image
    tau-grid, a different measure from the minimum-image centres path (they
    coincide for sigma << period); the caller documents that switch.
    """
    from ._nested_contraction import (
        tuple_counts, build_recipe, recipe_work, quad_nodes)
    from .._defaults import get_default
    r_levels = np.asarray(spec_x["r"]).ravel()
    sym_levels = np.asarray(spec_x["sym"]).ravel()
    tags_x = np.asarray(spec_x["tags"])
    tags_y = np.asarray(spec_y["tags"])
    m_perm_x = float(tuple_counts(r_levels, sym_levels, tags_x)[0])
    m_perm_y = float(tuple_counts(r_levels, sym_levels, tags_y)[0])
    centres_cost = m_perm_x * m_perm_y
    ts = get_default("truncation_sigmas")
    n_tau = quad_nodes(True, is_per, sigma, period, vmin, vmax, ts)
    rx = build_recipe(r_levels, sym_levels, tags_x, True, is_per)
    same = (tags_x.shape == tags_y.shape
            and bool(np.array_equal(tags_x, tags_y)))
    ry = rx if same else build_recipe(r_levels, sym_levels, tags_y, True,
                                      is_per)
    contract_cost = float(n_tau) * float(max(recipe_work(rx), recipe_work(ry)))
    return contract_cost < centres_cost


def _nested_attr_plan(dens_x, dens_y, a):
    """Route plus the *shared* quadrature grid for a nested attribute, decided
    once from the (x, y) pair.

    Returning the grid here -- rather than recomputing it inside each of xy, xx
    and yy -- is what makes the cosine normalise exactly: the relative grids are
    value-dependent, so a per-call grid would discretise the three inner
    products differently and the ratio would drift off 1 (breaking, e.g.,
    transposition invariance). One grid spanning both densities is used for all
    three. See :func:`_nested_attr_route` for the route meanings.
    """
    route = _nested_attr_route(dens_x, dens_y, a)
    if route in ("centres", "contract"):
        return route, None
    from ._nested_contraction import (auto_ntau, auto_ntau_default,
                                       auto_taus_line)
    from .._defaults import get_default, truncation_floor
    sigma = float(dens_x.sigma[a])
    period = float(dens_x.period[a])
    # Same kernel-value floor as every other truncation path
    # (:func:`truncation_floor` resolves None -> default; inf ->
    # accuracy-floor width; ``accuracy_floor_context`` honoured).
    tol = truncation_floor(get_default("truncation_sigmas"))
    if route == "taugrid":
        # The taugrid route is the faster all-image form; warn when it
        # materially differs from the canonical single-wrap measure (the same
        # warning the flat path raises), pointing to method='bulger'.
        if period > 0 and sigma / period > _ORBIT_SIGMA_OVER_P_THRESHOLD:
            _warn_rel_per_all_image(sigma / period)
        # Period-only grid; node count from the shared helper so the flat and
        # nested all-image grids coincide exactly.
        return route, np.linspace(0.0, period, auto_ntau_default(period, sigma),
                                  endpoint=False)
    # contract_relnonper: one line grid spanning both densities' values.
    px = np.asarray(dens_x.p_attr[a], dtype=np.float64)
    py = np.asarray(dens_y.p_attr[a], dtype=np.float64)
    allv = np.concatenate([px[np.isfinite(px)].ravel(),
                           py[np.isfinite(py)].ravel()])
    return route, auto_taus_line(allv, allv, sigma, tol)


def _nested_attr_route(dens_x, dens_y, a):
    """Per-attribute route for a nested attribute, decided once so xy, xx and
    yy share a single measure.

    - ``'contract'`` -- absolute and absolute-periodic: the kernel is a
      one-body product across slots, so the event-pair-vectorised per-level
      contraction applies the orbit (Möbius) reduction at symmetric levels and
      enumeration at ordered ones, mirroring the flat per-attribute matrix and
      never materialising the tuple set.
    - ``'centres'`` -- relative modes when the materialised-centres path is the
      cheaper route; for relative-non-periodic this is its exact analytic
      quadratic, for relative-periodic the minimum-image measure.
    - ``'contract_relnonper'`` -- relative-non-periodic when the translation-grid
      contraction is cheaper (large ``r = 1`` or symmetric levels, e.g. spectral
      cells). Same measure as the centres quadratic, to grid accuracy; a pure
      speed choice.
    - ``'taugrid'`` -- relative-periodic when the all-image tau-grid contraction
      is cheaper than the minimum-image centres path (see
      :func:`_rel_contract_cheaper`). This is the one place the toolbox's
      relative-periodic measure depends on the dispatch: it computes the
      all-image transposition average rather than minimum-image (the two
      coincide for sigma << period), accepted because no structurally cheap
      minimum-image route exists once the centres materialisation dominates.
    """
    is_rel = bool(dens_x.is_rel[a])
    is_per = bool(dens_x.is_per[a])
    if not is_rel:
        return "contract"
    spec_x = dens_x.nested[a]
    spec_y = dens_y.nested[a]
    sigma = float(dens_x.sigma[a])
    period = float(dens_x.period[a])
    vmin, vmax = ((0.0, period) if is_per
                  else _attr_value_range(dens_x, dens_y, a))
    if _rel_contract_cheaper(spec_x, spec_y, sigma, period, is_per, vmin, vmax):
        return "taugrid" if is_per else "contract_relnonper"
    return "centres"


def _nested_attr_matrix(dens_x, dens_y, a, route, taus):
    """(N_x, N_y) per-attribute inner matrix for a nested attribute, on the
    given ``route`` and shared ``taus`` from :func:`_nested_attr_plan` (passed
    in so xy, xx and yy share one measure and one grid)."""
    if route == "centres":
        cx = _closed_form_attr_centres(dens_x, a)
        cy = _closed_form_attr_centres(dens_y, a)
        return _closed_form_attr_matrix_from(cx, cy)
    from ._nested_contraction import build_recipe, nested_attr_matrix
    from .._defaults import get_default
    is_rel = bool(dens_x.is_rel[a])
    is_per = bool(dens_x.is_per[a])
    sigma = float(dens_x.sigma[a])
    period = float(dens_x.period[a])
    spec_x = dens_x.nested[a]
    spec_y = dens_y.nested[a]
    r_levels = np.asarray(spec_x["r"]).ravel()
    sym_levels = np.asarray(spec_x["sym"]).ravel()
    ts = get_default("truncation_sigmas")
    tags_x = np.asarray(spec_x["tags"])
    tags_y = np.asarray(spec_y["tags"])
    rx = build_recipe(r_levels, sym_levels, tags_x, is_rel, is_per)
    same = (tags_x.shape == tags_y.shape
            and bool(np.array_equal(tags_x, tags_y)))
    ry = rx if same else build_recipe(r_levels, sym_levels, tags_y,
                                      is_rel, is_per)
    PX = np.asarray(dens_x.p_attr[a], dtype=np.float64)
    PY = np.asarray(dens_y.p_attr[a], dtype=np.float64)
    if route == "taugrid":
        return nested_attr_matrix(rx, ry, PX, PY, dens_x.w[a], dens_y.w[a],
                                  sigma, is_per, period, ts, taus=taus,
                                  periodic_taus=True, taus_reduce="mean")
    if route == "contract_relnonper":
        return nested_attr_matrix(rx, ry, PX, PY, dens_x.w[a], dens_y.w[a],
                                  sigma, False, period, ts, taus=taus,
                                  periodic_taus=False, taus_reduce="sum")
    # 'contract': absolute / absolute-periodic
    return nested_attr_matrix(rx, ry, PX, PY, dens_x.w[a], dens_y.w[a],
                              sigma, is_per, period, ts, taus=None)


def _try_nested_contract(dens_x, dens_y, *, normalize, verbose, force=False):
    """Closed-form inner product of a single nested attribute.

    Returns (ip_xy, ip_xx, ip_yy) when the case is covered -- one nested
    attribute, outer/no ``[rel]``, cosine or one-sided normalisation,
    NaN-padded (variable-K) slots included -- via the per-level dispatch of
    :func:`_nested_attr_plan` / :func:`_nested_attr_matrix`; otherwise ``None``,
    and the caller routes to the exact enumeration. The route, decided once so
    xy, xx and yy share one measure, is the cheaper of the event-pair
    contraction and the materialised centres: absolute and absolute-periodic go
    to the contraction (exact, per-level Möbius/Bulger); relative-non-periodic
    to the centres analytic quadratic or, when cheaper, the translation-grid
    contraction (same measure, to grid accuracy); relative-periodic to the
    minimum-image centres or, when the centres materialisation dominates, the
    all-image tau-grid contraction -- the one route that changes the measure
    (all-image rather than minimum-image; identical for sigma << period). Every
    route costs no more than the equivalent flat attribute.
    """
    def _decline(reason):
        if force:
            raise ValueError(
                f"method='contract' is not available here: {reason}. "
                f"Use method='auto' (which falls back automatically) or "
                f"method='bulger'."
            )
        return None

    if normalize not in ("cosine", "oneSidedDenom"):
        return _decline(f"unsupported normalisation {normalize!r}")
    if dens_x.n_attrs != dens_y.n_attrs:
        return _decline("the two densities have different attribute counts")
    if dens_x.n_attrs != 1:
        # Nested attribute(s) tensored with further attributes: the cosine
        # factorises per event-pair across attributes (JMM Eq 3.4), so the
        # nested factor goes through the contraction and the rest through the
        # per-attribute MA matrices, instead of enumerating the joint tuple.
        return _try_nested_contract_ma(
            dens_x, dens_y, normalize=normalize, verbose=verbose, force=force)
    spec = dens_x.nested[0]
    spec_y = dens_y.nested[0]
    if spec is None or spec_y is None:
        return _decline("both densities must carry the same nested attribute")
    if int(_inner_r_vec(dens_x)[0]) != 0 or int(_inner_r_vec(dens_y)[0]) != 0:
        return _decline("an inner/intermediate [rel] unit is not yet covered")

    r_levels = np.asarray(spec["r"]).ravel()
    sym_levels = np.asarray(spec["sym"]).ravel()
    # The two densities must agree on the per-level read-arities and [sym]
    # flags (same nested attribute); only the leaf cardinalities may differ
    # -- a 4-pitch prototype against an 8-pitch window, say.
    if (not np.array_equal(r_levels, np.asarray(spec_y["r"]).ravel())
            or not np.array_equal(sym_levels,
                                  np.asarray(spec_y["sym"]).ravel())):
        return _decline("the two nested attributes differ in [r]/[sym]")

    # Per-level dispatch (see _nested_attr_route / _nested_attr_matrix):
    # absolute and absolute-periodic reduce through the event-pair-vectorised
    # per-level Möbius/Bulger contraction; relative-non-periodic and
    # minimum-image relative-periodic through the materialised centres; a large
    # compounded symmetric relative-periodic level falls back to the all-image
    # tau-grid. The route is decided once so xy, xx and yy share one measure.
    route, taus = _nested_attr_plan(dens_x, dens_y, 0)
    ip_xy = float(_nested_attr_matrix(dens_x, dens_y, 0, route, taus).sum())
    ip_xx = float(_nested_attr_matrix(dens_x, dens_x, 0, route, taus).sum())
    ip_yy = float(_nested_attr_matrix(dens_y, dens_y, 0, route, taus).sum())
    return ip_xy, ip_xx, ip_yy


def _try_nested_contract_ma(dens_x, dens_y, *, normalize, verbose, force=False):
    """MA cosine when one or more attributes are nested or ordered.

    The MAET cross-event inner product factorises per event-pair across
    attributes (JMM Eq 3.4): ``<X,Y> = Σ_{i,j} Π_a I_a(i,j)``. Each attribute
    contributes an (N_x, N_y) per-event-pair inner matrix. A nested attribute
    goes through the per-level dispatch (:func:`_nested_attr_plan` /
    :func:`_nested_attr_matrix`): the event-pair contraction for absolute and
    absolute-periodic and for the cost-selected relative grids, the
    materialised centres otherwise -- with the route decided once so its xy, xx
    and yy share one measure (see the relative-periodic minimum-image vs
    all-image note there). An ordered-flat attribute (``[sym]=0``, r>1, not
    nested) goes through the centres path
    (:func:`_closed_form_attr_matrix_from`): a single ordered level has no
    symmetric orbit to reduce, and routing it through the orbit/Möbius matrix
    would wrongly symmetrise it (summing its full ``S_r`` orbit). Flat-symmetric
    and r=1 attributes go through the orbit/Möbius per-attribute matrix
    (:func:`_ma_per_attr_inner_matrix`). The matrices multiply element-wise then
    sum, mirroring :func:`_cos_sim_exp_tens_ma_orbit`. Per-attribute prefactors
    are constant and cancel, so mixing the matrix conventions is exact.

    Returns the (ip_xy, ip_xx, ip_yy) triple, or ``None`` (route to the exact
    enumeration) for an uncovered nested case.
    """
    def _decline(reason):
        if force:
            raise ValueError(
                f"method='contract' is not available here: {reason}. "
                f"Use method='auto' (which falls back automatically) or "
                f"method='bulger'."
            )
        return None

    if normalize not in ("cosine", "oneSidedDenom"):
        return _decline(f"unsupported normalisation {normalize!r}")
    A = int(dens_x.n_attrs)
    nested_x = getattr(dens_x, "nested", None) or [None] * A
    nested_y = getattr(dens_y, "nested", None) or [None] * A
    inner_rx = _inner_r_vec(dens_x)
    inner_ry = _inner_r_vec(dens_y)
    is_sym_x = np.asarray(
        getattr(dens_x, "is_sym", np.ones(A, dtype=bool))).ravel()

    N_x = int(dens_x.n)
    N_y = int(dens_y.n)
    P_xy = np.ones((N_x, N_y), dtype=np.float64)
    P_xx = np.ones((N_x, N_x), dtype=np.float64)
    P_yy = np.ones((N_y, N_y), dtype=np.float64)

    for a in range(A):
        is_nested = (nested_x[a] is not None) or (nested_y[a] is not None)
        r_a = int(dens_x.r[a])
        ordered_flat = ((not is_nested) and (not bool(is_sym_x[a]))
                        and (r_a > 1))

        if not (is_nested or ordered_flat):
            # Flat-symmetric or r=1: the orbit/Möbius per-attribute matrix,
            # which correctly symmetrises these readings.
            sigma = float(dens_x.sigma[a])
            is_rel = bool(dens_x.is_rel[a])
            is_per = bool(dens_x.is_per[a])
            period = float(dens_x.period[a])
            Pxa, Pya = dens_x.p_attr[a], dens_y.p_attr[a]
            Wxa, Wya = dens_x.w[a], dens_y.w[a]
            P_xy *= _ma_per_attr_inner_matrix(
                Pxa, Wxa, Pya, Wya, sigma, r_a, is_rel, is_per, period)
            P_xx *= _ma_per_attr_inner_matrix(
                Pxa, Wxa, Pxa, Wxa, sigma, r_a, is_rel, is_per, period)
            P_yy *= _ma_per_attr_inner_matrix(
                Pya, Wya, Pya, Wya, sigma, r_a, is_rel, is_per, period)
            continue

        if is_nested:
            if nested_x[a] is None or nested_y[a] is None:
                return _decline("an attribute is nested on only one side")
            if int(inner_rx[a]) != 0 or int(inner_ry[a]) != 0:
                return _decline(
                    "an inner/intermediate [rel] unit is not yet covered")
            r_levels = np.asarray(nested_x[a]["r"]).ravel()
            sym_levels = np.asarray(nested_x[a]["sym"]).ravel()
            if (not np.array_equal(
                    r_levels, np.asarray(nested_y[a]["r"]).ravel())
                    or not np.array_equal(
                        sym_levels, np.asarray(nested_y[a]["sym"]).ravel())):
                return _decline("the two nested attributes differ in [r]/[sym]")
            # Nested: the mode-aware per-level dispatch (contraction for
            # absolute/abs-periodic and the large-symmetric rel-periodic
            # tau-grid; centres for relative-non-periodic and minimum-image
            # rel-periodic). Route decided once so xy, xx and yy share one
            # measure.
            route, taus = _nested_attr_plan(dens_x, dens_y, a)
            P_xy *= _nested_attr_matrix(dens_x, dens_y, a, route, taus)
            P_xx *= _nested_attr_matrix(dens_x, dens_x, a, route, taus)
            P_yy *= _nested_attr_matrix(dens_y, dens_y, a, route, taus)
            continue

        # Ordered flat ([sym]=0, r>1, not nested): the materialised centres,
        # which honour the ordered reading (no symmetrisation -- the orbit
        # matrix would wrongly symmetrise) and use the minimum-image
        # pairwise-wrap for relative-periodic. A single ordered level has no
        # symmetric orbit to reduce, so there is no per-level Möbius to gain.
        cx = _closed_form_attr_centres(dens_x, a)
        cy = _closed_form_attr_centres(dens_y, a)
        P_xy *= _closed_form_attr_matrix_from(cx, cy)
        P_xx *= _closed_form_attr_matrix_from(cx, cx)
        P_yy *= _closed_form_attr_matrix_from(cy, cy)

    # The joint-tuple enumeration (bulger) mis-shapes a nested attribute's
    # per-event tuples in the MA tensor build, so a nested multi-attribute
    # density must not fall back to it. Every per-attribute route here costs no
    # more than the equivalent flat attribute, so always return the triple.
    return float(P_xy.sum()), float(P_xx.sum()), float(P_yy.sum())


def _cos_sim_exp_tens_ma_pairwise(dens_x, dens_y, *, verbose: bool = True,
                                  truncation_sigmas=None,
                                  kernel_precision=None):
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
        truncation_sigmas=truncation_sigmas,
        kernel_precision=kernel_precision,
        inner_r=inner_r,
    )
    ip_xx = _ip_core_ma(
        dens_x.u_perm, dens_x.w_j, n_jx,
        dens_x.v_comb, dens_x.wv_comb, n_kx,
        A, r_vec, sigma, is_rel, is_per, period,
        truncation_sigmas=truncation_sigmas,
        kernel_precision=kernel_precision,
        inner_r=inner_r,
    )
    ip_yy = _ip_core_ma(
        dens_y.u_perm, dens_y.w_j, n_jy,
        dens_y.v_comb, dens_y.wv_comb, n_ky,
        A, r_vec, sigma, is_rel, is_per, period,
        truncation_sigmas=truncation_sigmas,
        kernel_precision=kernel_precision,
        inner_r=inner_r,
    )
    return ip_xy, ip_xx, ip_yy



# -------------------------------------------------------------------
#  cos_sim_exp_tens_raw  (dispatches single-multiset or multi-attribute based on input shape)
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

       - Single-multiset: ``cos_sim_exp_tens(p1, w1, p2, w2, sigma, r, is_rel, is_per, period)``
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



def _ma_ip_per_event_factors(dens, side):
    """Per-attribute, per-event centres and weights for the factored IP.

    Returns ``factors[a]`` = list over the density's events of
    ``(centres, w)``, where ``centres`` is the ``(r_a, M)`` array of the
    attribute's r-ad centres for that event and ``w`` the matching
    ``(M,)`` weight-product vector. ``side='perm'`` enumerates the
    permutation side (the X convention), ``side='comb'`` the combination
    side (the Y convention); the r_a! ratio between them cancels in the
    cosine, exactly as in the joint build.

    Each event enumerates its own non-NaN slots, so variable cardinality
    (NaN-padded ``p_attr``) is handled per event without a common-slab
    zero-pad.
    """
    A = dens.n_attrs
    N = dens.n
    r_vec = dens.r
    is_sym = np.asarray(getattr(dens, "is_sym", np.ones(A, dtype=bool)))
    nested = dens.nested if getattr(dens, "nested", None) is not None \
        else [None] * A
    want_perm = (side == "perm")
    factors = [[None] * N for _ in range(A)]
    for a in range(A):
        r_a = int(r_vec[a])
        P = dens.p_attr[a]
        W = dens.w[a]
        spec = nested[a]
        for n in range(N):
            val_col = P[:, n]
            w_col = W[:, n]
            valid = np.nonzero(~np.isnan(val_col))[0].astype(np.intp)
            if spec is not None:
                tags_valid = np.asarray(spec["tags"])[valid]
                perm_mat, comb_mat = _nested_enum_indices(
                    valid, tags_valid,
                    np.asarray(spec["r"]).ravel(),
                    np.asarray(spec["sym"]).ravel(),
                )
                idx = perm_mat if want_perm else comb_mat
            else:
                perm_mat, comb_mat, _, _ = _enum_flat_attr(
                    val_col, valid, r_a, bool(is_sym[a]), w_col)
                idx = perm_mat if want_perm else comb_mat
            factors[a][n] = (val_col[idx], np.prod(w_col[idx], axis=0))
    return factors


def _ma_ip_factor_dense(u, wU, v, wV, r, sigma, is_rel, is_per, period, r_in):
    """One attribute's IP factor by dense evaluation.

    Serves the forms the culled helper does not: the relative-periodic
    pairwise-wrap quadratic and the nested block-diagonal metric
    (``r_in > 0``).
    """
    D = u[:, :, None] - v[:, None, :]                 # (r, M_u, M_v)
    if r_in > 0:
        Q = _compute_Q_inner_blocks(D, r_in, bool(is_per), float(period),
                                    reduced=False)
    else:
        if is_per and not is_rel:
            p = float(period)
            D = D - p * np.floor(D / p + 0.5)
        Q = _compute_Q(D, r, bool(is_rel), bool(is_per), float(period))
    K = np.exp(-Q / (4.0 * float(sigma) ** 2))
    return float(wU @ K @ wV)


def _ma_ip_factored(dens_perm, dens_comb, *, truncation_sigmas=None,
                    kernel_precision=None):
    """One MA inner product ``<perm density, comb density>`` factored
    over attributes and event pairs, without materialising the joint
    tuple set.

    Uses the per-attribute inner-product factorisation
    ``<T_X, T_Y> = sum_{n, m} prod_a I_a(n, m)``: each event pair's joint
    tuples are the Cartesian product of the per-attribute tuples, so the
    joint bilinear form distributes into a product of per-attribute
    factors, and the whole is summed over event pairs.

    A flat, non-relative-periodic attribute's factor routes through the
    spatially-culled ``gaussian_kernel_sum`` helper (the same cull the
    single-attribute centres path uses); a relative-periodic or nested
    attribute's factor is evaluated densely.
    """
    A = dens_perm.n_attrs
    r_vec = dens_perm.r
    sigma = dens_perm.sigma
    is_rel = dens_perm.is_rel
    is_per = dens_perm.is_per
    period = dens_perm.period
    inner_r = _inner_r_vec(dens_perm)

    cullable = [int(inner_r[a]) == 0
                and not (bool(is_rel[a]) and bool(is_per[a]))
                for a in range(A)]

    pf = _ma_ip_per_event_factors(dens_perm, "perm")
    cf = _ma_ip_per_event_factors(dens_comb, "comb")
    Nx = dens_perm.n
    Ny = dens_comb.n

    ip = 0.0
    for n in range(Nx):
        for m in range(Ny):
            prod = 1.0
            for a in range(A):
                u, wU = pf[a][n]
                v, wV = cf[a][m]
                if cullable[a]:
                    factor = _ip_via_helper(
                        u, wU, v, wV, int(r_vec[a]), float(sigma[a]),
                        bool(is_rel[a]), bool(is_per[a]), float(period[a]),
                        truncation_sigmas=truncation_sigmas,
                        kernel_precision=kernel_precision)
                else:
                    factor = _ma_ip_factor_dense(
                        u, wU, v, wV, int(r_vec[a]), float(sigma[a]),
                        bool(is_rel[a]), bool(is_per[a]), float(period[a]),
                        int(inner_r[a]))
                prod *= factor
                if prod == 0.0:
                    break
            ip += prod
    return ip


def _ma_factored_ip_supported(dens_x, dens_y):
    """True when the factored culled IP covers this density pair.

    The factored path serves every attribute mode except relative-and-
    periodic under the minimum-image convention, whose per-slot factor
    does not admit the culled helper. Ordered ([sym]=0) and nested
    attributes are supported (the nested factor is evaluated densely
    within its block-diagonal metric).
    """
    from .aniso import density_has_kernel_cov
    if density_has_kernel_cov(dens_x) or density_has_kernel_cov(dens_y):
        return False
    A = dens_x.n_attrs
    is_rel = dens_x.is_rel
    is_per = dens_x.is_per
    return not any(bool(is_rel[a]) and bool(is_per[a]) for a in range(A))


def _cos_sim_exp_tens_ma_factored(dens_x, dens_y, *, verbose=True,
                                  truncation_sigmas=None,
                                  kernel_precision=None):
    """Factored-cull twin of :func:`_cos_sim_exp_tens_ma_pairwise`.

    Computes the ``(ip_xy, ip_xx, ip_yy)`` triple through
    :func:`_ma_ip_factored`, so the joint tuple set is never built. The
    value equals the pairwise (joint) triple exactly at the accuracy
    floor; at a finite truncation the two differ only in the cull region
    (the factored form truncates each attribute independently, enclosing
    a superset of the joint form's culled pairs).
    """
    return (
        _ma_ip_factored(dens_x, dens_y, truncation_sigmas=truncation_sigmas,
                        kernel_precision=kernel_precision),
        _ma_ip_factored(dens_x, dens_x, truncation_sigmas=truncation_sigmas,
                        kernel_precision=kernel_precision),
        _ma_ip_factored(dens_y, dens_y, truncation_sigmas=truncation_sigmas,
                        kernel_precision=kernel_precision),
    )


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



def _inner_product_direct_abs(p_x, w_x, p_y, w_y, sigma, r,
                                   is_per, period):
    """<T_X, T_Y> in absolute mode via direct r-tuple enumeration.

    Computes the single-multiset inner product
        <T_X, T_Y> = (sigma * sqrt(pi))**r *
                     sum_{J, K} wJ_x[J] * wJ_y[K] *
                                exp(-||centres_x[:, J] - centres_y[:, K]||^2
                                    / (4 sigma^2))
    by enumerating ordered r-tuples on each side. No Möbius
    alternating sum is involved, so the result is exact (no
    catastrophic cancellation) for any K_x, K_y >= r. This is a reference/direct implementation used for event
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
    """Ordered r-tuple construction for a single multiset.

    Returns ``(U, wJ)`` where ``U`` is ``(r, nJ)`` of position values
    along ordered r-tuples and ``wJ`` is ``(nJ,)`` of weight products.
    Used by :func:`_inner_product_direct_abs` and any other helper
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
    """<T_A, T_B> in relative mode: the single-multiset (N = 1)
    specialisation of :func:`_ma_per_attr_inner_matrix_rel`.

    All conventions are the batched core's: the shared ``[0, P)``
    grid with :func:`auto_ntau_default` nodes in periodic mode; in
    non-periodic mode a window of width
    ``spread_a + spread_b + 2 * _rel_window_margin(t) * sigma``
    centred on the weighted-mean offset, with ``samples_per_sigma``
    nodes per sigma, evaluated by plain Riemann sum (the margin places
    every kernel entry strictly outside the truncation radius at the
    window edges, so the endpoint integrand is exactly zero and the
    Riemann sum equals the trapezoidal rule exactly);
    slab-bounded contraction; and, when requested, the mass-aware
    cancellation diagnostic
    ``|sum_u F_u| / sum_u max_orb(|term_orb_u|)``.

    With ``return_cancellation_ratio=True``, returns ``(value, ratio)``.
    """
    Pa = np.asarray(p_a, dtype=np.float64).reshape(-1, 1)
    Pb = np.asarray(p_b, dtype=np.float64).reshape(-1, 1)
    Wa = np.asarray(w_a, dtype=np.float64).reshape(-1, 1)
    Wb = np.asarray(w_b, dtype=np.float64).reshape(-1, 1)
    out = _ma_per_attr_inner_matrix_rel(
        Pa, Wa, Pb, Wb, sigma, r, is_per, period,
        return_cancellation_ratio=return_cancellation_ratio,
        truncation_sigmas=truncation_sigmas,
        samples_per_sigma=samples_per_sigma,
    )
    if return_cancellation_ratio:
        I, ratio = out
        return float(I[0, 0]), float(ratio)
    return float(out[0, 0])


def _cos_sim_raw_single_multiset_batch(
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
    truncation_sigmas: float | None = None,
    kernel_precision: str | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """Raw single-multiset batched dispatch for :func:`cos_sim_exp_tens`.

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
        single-multiset core via the inner ``cos_sim_exp_tens`` call.
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
        truncation_sigmas=truncation_sigmas,
        kernel_precision=kernel_precision,
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
       positionally (matching the single-multiset scalar raw form), instead of being
       keyword-only. This shim preserves the old keyword-only weight API
       for backward compatibility but issues a ``DeprecationWarning``.
       This shim will be removed in a future release.
    """
    warnings.warn(
        "batch_cos_sim_exp_tens is deprecated. Pass 2-D pitch matrices "
        "directly to cos_sim_exp_tens (with weights as positional arguments "
        "after each pitch matrix, matching the single-multiset scalar raw form). This "
        "shim will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _cos_sim_raw_single_multiset_batch(
        p_mat_a, p_mat_b, sigma, r, is_rel, is_per, period,
        weights_a=weights_a, weights_b=weights_b,
        spectrum=spectrum, precision=precision,
        verbose=verbose,
    )