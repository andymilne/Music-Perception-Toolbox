"""Post-tensor windowing and sliding-window similarity profiles.

This module hosts the windowing layer of the MAET pipeline:

* :func:`window_tensor` --- wrap a :class:`MaetDensity` with a per-attribute
  window specification (size, mix, optional centre), returning a
  :class:`WindowedMaetDensity`. No math is done at construction time;
  windowing is applied lazily.
* :func:`windowed_tensor_similarity` --- sweep a query density's centroid
  across a context density via offset positions, returning a 1xM
  windowed-similarity profile (or list of profiles for list-mode
  input).

The closed-form windowed inner product (:func:`_windowed_inner_product`)
is also exposed at module level for internal use by the cosine-similarity
machinery in :mod:`mpt.tensor` (which currently still hosts the
non-windowed cosine path).

Periodic attributes are handled exactly (to floating-point precision) by
summing line-case window contributions over periodic images of the
window centre. The image sum is truncated adaptively when the latest
image-pair's contribution falls below
:data:`_IMAGE_SUM_TOL_DOUBLE` of the running maximum. See User Guide
§3.1 ("Post-tensor windowing").

See USER_GUIDE §3.1 ("Post-tensor windowing") for the user-facing
description and :doc:`/ARCHITECTURE` §2 for the layering.

Cross-module dependencies: this module reaches into
:mod:`._tensor.dispatch` for a small number of dispatch helpers
(``_normalize_density_input``, ``_resolve_list_list_mode``,
``_compute_Q``). Those imports are deferred to call time to avoid
import-cycle issues during package load (dispatch is imported by
cosine and eval, which load before this module reaches the dispatch
call sites).
"""
from __future__ import annotations

import warnings

import numpy as np

from .._utils import maybe_print_batched_estimate, validate_weights
from .._defaults import _with_dispatch_scope
from .density import (
    MaetDensity,
    WindowedMaetDensity,
)


# Convergence tolerance for the periodic-image sum used inside the
# windowed numerator and windowed evaluator. Matches the existing
# Möbius cancellation-guard convention (1e-12) for double precision
# and the kernel_precision='single' floor (1e-7) for single. Both are
# documented in USER_GUIDE §3.1.
_IMAGE_SUM_TOL_DOUBLE = 1e-12
_IMAGE_SUM_TOL_SINGLE = 1e-7


def window_tensor(dens, window_spec) -> WindowedMaetDensity:
    """Wrap a MaetDensity with a post-tensor window specification.

    Returns a :class:`WindowedMaetDensity` that bundles the underlying
    density with a window spec. No math is performed at construction
    time; the window is applied lazily by :func:`eval_exp_tens` and
    :func:`cos_sim_exp_tens`. See the MAET specification §4.3.

    Densities built with a matrix-valued kernel covariance are not
    accepted: their stored coordinates are whitened, so a post-tensor
    window (specified in the attribute's original coordinates) would
    be applied in the wrong frame. Use :func:`mpt.windowed_similarity`
    or :func:`mpt.windowed_entropy`, whose windows reweight the raw
    events before the density is built.

    Parameters
    ----------
    dens : MaetDensity
        The density to be windowed.
    window_spec : dict
        Window specification with keys:

        ``size`` : scalar or length-A array
            Per-attribute window effective standard deviation in
            multiples of that attribute's ``sigma``. NaN or Inf means the
            attribute is not windowed. A scalar is broadcast across all
            attributes.
        ``mix`` : scalar or length-A array
            Per-attribute shape parameter in [0, 1]: 0 = pure Gaussian,
            1 = pure rectangular, in between = rectangular-convolved-
            with-Gaussian. A scalar is broadcast.
        ``centre`` : length-A list of array-like, or a single array-like
            Per-attribute centre coordinates. Each entry has length
            ``dim_per_attr[a]``. A single 1-D array of total length
            ``dim`` is split across attributes in order. Entries whose
            attribute has ``size`` NaN/Inf are ignored.

    Returns
    -------
    WindowedMaetDensity
    """
    if not isinstance(dens, MaetDensity):
        raise TypeError(
            f"window_tensor expected a MaetDensity; got {type(dens).__name__}."
        )
    if not isinstance(window_spec, dict):
        raise TypeError("window_spec must be a dict.")
    from .aniso import density_has_kernel_cov
    if density_has_kernel_cov(dens):
        raise NotImplementedError(
            "window_tensor does not support densities built with a "
            "matrix-valued kernel covariance: their stored coordinates "
            "are whitened, so a window specified in the attribute's "
            "original coordinates would be applied in the wrong frame. "
            "Use windowed_similarity or windowed_entropy, whose windows "
            "reweight the raw events before the density is built."
        )

    A = dens.n_attrs
    dim_per = dens.dim_per_attr
    dim_total = int(dens.dim)

    # --- size ---
    size_arr = np.asarray(window_spec.get("size"), dtype=np.float64).ravel()
    if size_arr.size == 1:
        size_arr = np.full(A, float(size_arr.item()))
    if size_arr.size != A:
        raise ValueError(
            f"window_spec['size'] must be a scalar or length-{A} array; "
            f"got length {size_arr.size}."
        )

    # --- mix ---
    mix_arr = np.asarray(window_spec.get("mix"), dtype=np.float64).ravel()
    if mix_arr.size == 1:
        mix_arr = np.full(A, float(mix_arr.item()))
    if mix_arr.size != A:
        raise ValueError(
            f"window_spec['mix'] must be a scalar or length-{A} array; "
            f"got length {mix_arr.size}."
        )
    if np.any((mix_arr < 0) | (mix_arr > 1)):
        raise ValueError("window_spec['mix'] entries must be in [0, 1].")

    # --- centre ---
    centre_in = window_spec.get("centre")
    if centre_in is None:
        # Default: centre at origin for all attributes.
        centre_list = [np.zeros(int(dim_per[a]), dtype=np.float64)
                       for a in range(A)]
    elif isinstance(centre_in, (list, tuple)) and not (
            len(centre_in) > 0 and np.isscalar(centre_in[0])):
        if len(centre_in) != A:
            raise ValueError(
                f"window_spec['centre'] (list form) must have length A = {A}; "
                f"got length {len(centre_in)}."
            )
        centre_list = []
        for a, ca in enumerate(centre_in):
            arr = np.asarray(ca, dtype=np.float64).ravel()
            if arr.size != int(dim_per[a]):
                raise ValueError(
                    f"window_spec['centre'][{a}] must have length "
                    f"{int(dim_per[a])}; got length {arr.size}."
                )
            centre_list.append(arr)
    else:
        # Numeric input (scalar, 0-D ndarray, or ndarray of any shape).
        # Two interpretations:
        #   * size 1: scalar broadcast — fill every per-attribute slot
        #     uniformly with the scalar value.
        #   * size dim_total: flat form, split by ``dim_per_attr``.
        #   * anything else: error.
        # List/tuple inputs do NOT take this path even when size 1;
        # they are interpreted structurally above. Pre-existing test
        # contract: a cell with the wrong number of entries raises
        # rather than silently broadcasting.
        try:
            arr = np.asarray(centre_in, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"window_spec['centre'] must be numeric or a list of "
                f"per-attribute centre vectors; could not convert "
                f"{type(centre_in).__name__} to a numeric array."
            ) from exc

        if arr.size == 1:
            val = float(arr.reshape(()))
            centre_list = [np.full(int(dim_per[a]), val, dtype=np.float64)
                           for a in range(A)]
        else:
            flat = arr.ravel()
            if flat.size != dim_total:
                raise ValueError(
                    f"window_spec['centre'] (flat form) must have length "
                    f"dim = {dim_total}; got length {flat.size}."
                )
            centre_list = []
            offset = 0
            for a in range(A):
                da = int(dim_per[a])
                centre_list.append(flat[offset:offset + da].astype(np.float64))
                offset += da

    return WindowedMaetDensity(
        tag="WindowedMaetDensity",
        dens=dens,
        size=size_arr,
        mix=mix_arr,
        centre=centre_list,
    )


# -------------------------------------------------------------------
#  Window pointwise evaluation (for eval_exp_tens dispatch)
# -------------------------------------------------------------------


def _is_windowed_attr(size_a, mix_a):
    """True iff the attribute has an effective window (size finite and > 0)."""
    return np.isfinite(size_a) and size_a > 0


def _window_width_params(size_g, mix_g, sigma_g):
    """Derive (a, b) for a group: rect half-width a and Gaussian width b.

    Window is rectangular-of-half-width-a convolved with Gaussian-of-
    width-b:
        a = size_g * sigma_g * sqrt(3 * mix_g)
        b = size_g * sigma_g * sqrt(1 - mix_g)
    """
    s = float(size_g) * float(sigma_g)
    a = s * np.sqrt(3.0 * float(mix_g))
    b = s * np.sqrt(1.0 - float(mix_g))
    return a, b


def _window_factor_1d(u, a, b):
    """Evaluate the 1-D window function at *u*.

    Window is rectangular-of-half-width-a convolved with Gaussian-of-
    width-b. Returns the convolution value at u.

    Cases:
      - b == 0 (pure rectangular): indicator(|u| <= a), values in {0, 1}.
      - a == 0 (pure Gaussian): exp(-u^2 / (2 b^2)).
      - otherwise: 0.5 * [erf((a + u)/(sqrt(2)*b)) + erf((a - u)/(sqrt(2)*b))].
    """
    u = np.asarray(u, dtype=np.float64)
    if b == 0.0:
        return (np.abs(u) <= a).astype(np.float64)
    if a == 0.0:
        return np.exp(-u**2 / (2.0 * b**2))
    from scipy.special import erf
    arg_plus  = (a + u) / (np.sqrt(2.0) * b)
    arg_minus = (a - u) / (np.sqrt(2.0) * b)
    return 0.5 * (erf(arg_plus) + erf(arg_minus))


def _evaluate_window_on_query(wmd: "WindowedMaetDensity", x_list):
    """Evaluate the window W(x) at query points.

    For periodic groups, the window is the wrapped Gaussian (or the
    wrapped rect-conv-Gaussian for ``mix > 0``): the sum of line-case
    window values at all periodic images of the window centre. The sum
    is truncated adaptively when successive image-pair contributions
    fall below :data:`_IMAGE_SUM_TOL_DOUBLE` of the running maximum.

    For non-periodic groups, the window is evaluated directly as a
    single line-case factor (no summation).

    Parameters
    ----------
    wmd : WindowedMaetDensity
    x_list : list of (dim_per_attr[a], nQ) arrays
        Per-attribute query matrices, one per attribute, with n_Q
        query points.

    Returns
    -------
    ndarray of shape (nQ,)
        Window values at each query point.
    """
    dens = wmd.dens
    A = dens.n_attrs
    dim_per = dens.dim_per_attr
    sigma = dens.sigma
    is_per = dens.is_per
    period = dens.period

    n_q = x_list[0].shape[1] if A > 0 else 0
    result = np.ones(n_q, dtype=np.float64)
    for a in range(A):
        if not _is_windowed_attr(wmd.size[a], wmd.mix[a]):
            continue
        a_, b_ = _window_width_params(wmd.size[a], wmd.mix[a], sigma[a])
        da = int(dim_per[a])
        centre_a = wmd.centre[a]  # shape (da,)
        x_a = x_list[a]           # shape (da, nQ)
        u = x_a - centre_a[:, None]  # (da, nQ)
        per = bool(is_per[a])
        P_g = float(period[a]) if per else 0.0
        for i in range(da):
            if per:
                # Wrapped window: sum line-case window at u + n*P for
                # n = 0, +/-1, +/-2, ..., truncated when the latest
                # image-pair's largest contribution falls below
                # _IMAGE_SUM_TOL_DOUBLE * running max.
                w_axis = _wrapped_window_factor_1d(
                    u[i], a_, b_, P_g, _IMAGE_SUM_TOL_DOUBLE,
                )
            else:
                w_axis = _window_factor_1d(u[i], a_, b_)
            result = result * w_axis
    return result


def _wrapped_window_factor_1d(u, a, b, period, image_tol,
                              n_max_cap: int = 100):
    """Sum of line-case window factors at all periodic images of u.

    Equivalent to evaluating a wrapped Gaussian (or wrapped rect-conv-
    Gaussian for mix > 0) at u. Truncates adaptively when the latest
    image-pair's largest contribution falls below ``image_tol`` times
    the running max.
    """
    u = np.asarray(u, dtype=np.float64)
    # Reduce to the minimal image in [-period/2, period/2) so the n=0 term
    # is the dominant image. Without this, an offset many periods from the
    # centre underflows the near images to 0, the running max stays 0, and
    # the sum terminates before reaching the dominant (distant) image.
    if period > 0.0:
        u = u - period * np.round(u / period)
    acc = _window_factor_1d(u, a, b)
    running_max = float(np.max(np.abs(acc)))
    for n in range(1, n_max_cap + 1):
        shift = n * period
        f_pos = _window_factor_1d(u + shift, a, b)
        f_neg = _window_factor_1d(u - shift, a, b)
        acc = acc + f_pos + f_neg
        new_max = float(max(np.max(np.abs(f_pos)),
                            np.max(np.abs(f_neg))))
        running_max = max(running_max, float(np.max(np.abs(acc))))
        if running_max == 0.0:
            break
        if new_max / running_max < image_tol:
            break
    else:
        import warnings as _warnings
        _warnings.warn(
            f"Wrapped window evaluation hit the safety cap of "
            f"{n_max_cap} image pairs without converging to relative "
            f"tolerance {image_tol:g}. This usually indicates "
            f"sigma_w >> P; consider evaluating without a window.",
            RuntimeWarning, stacklevel=4,
        )
    return acc


# -------------------------------------------------------------------
#  windowed_tensor_similarity and closed-form pair-factor evaluator
# -------------------------------------------------------------------


def _resolve_windowed_similarity_reference(reference, q_list):
    """Resolve the polymorphic ``reference`` argument of
    :func:`windowed_tensor_similarity` to a list (length ``n_q``) of per-query
    reference lists (each of length ``n_attrs``, with each entry a 1-D
    array of length ``dim_per_attr[a]``).

    ``None`` is returned as a list of ``None``s, signalling that each
    query's auto-centroid should be computed in the per-pair core.
    """
    n_q = len(q_list)
    if reference is None:
        return [None] * n_q

    if not isinstance(reference, (list, tuple)):
        raise TypeError(
            f"reference must be None, a list of per-attribute 1-D arrays "
            f"(shared form), or a list of length n_q of such lists "
            f"(per-query form); got {type(reference).__name__}."
        )

    template = q_list[0]
    n_attrs = int(template.n_attrs)
    dim_per_a = [int(d) for d in template.dim_per_attr]

    outer_len = len(reference)

    if outer_len == 0:
        raise ValueError("reference must be non-empty.")

    # The "shared" form has elements that are 1-D arrays of numbers; the
    # "per-query" form has elements that are themselves lists of arrays.
    def _is_per_query_outer(ref):
        first_el = ref[0]
        if isinstance(first_el, (list, tuple)):
            return True
        if isinstance(first_el, np.ndarray) and first_el.dtype == object:
            return True
        return False

    is_per_query = _is_per_query_outer(reference)

    if is_per_query:
        if outer_len != n_q:
            raise ValueError(
                f"Per-query reference must have length n_q = {n_q}; "
                f"got {outer_len}."
            )
        out = []
        for i, ref_q in enumerate(reference):
            if not isinstance(ref_q, (list, tuple)) and not (
                isinstance(ref_q, np.ndarray) and ref_q.dtype == object
            ):
                raise TypeError(
                    f"reference[{i}] must be a list/tuple of "
                    f"{n_attrs} per-attribute 1-D arrays; got "
                    f"{type(ref_q).__name__}."
                )
            if len(ref_q) != n_attrs:
                raise ValueError(
                    f"reference[{i}] must have {n_attrs} entries (one per "
                    f"query attribute); got {len(ref_q)}."
                )
            ref_q_validated = []
            for a in range(n_attrs):
                ref_a = np.asarray(ref_q[a], dtype=np.float64).reshape(-1)
                if ref_a.size != dim_per_a[a]:
                    raise ValueError(
                        f"reference[{i}][{a}] must have length "
                        f"{dim_per_a[a]} (dim of attribute {a}); got "
                        f"{ref_a.size}."
                    )
                ref_q_validated.append(ref_a)
            out.append(ref_q_validated)
        return out

    # Shared form: outer_len must equal n_attrs.
    if outer_len != n_attrs:
        raise ValueError(
            f"Shared reference must have {n_attrs} entries (one per query "
            f"attribute); got {outer_len}. For per-query references, pass a "
            f"list of length n_q = {n_q} of such per-attribute lists."
        )
    shared = []
    for a in range(n_attrs):
        ref_a = np.asarray(reference[a], dtype=np.float64).reshape(-1)
        if ref_a.size != dim_per_a[a]:
            raise ValueError(
                f"reference[{a}] must have length {dim_per_a[a]} (dim of "
                f"attribute {a}); got {ref_a.size}."
            )
        shared.append(ref_a)
    return [shared] * n_q


def _windowed_similarity_pair(dens_context, dens_query, window_spec, offsets,
                               *, ref_per_a,
                               normalize: str = "oneSidedDenom",
                               truncation_sigmas: float | None = None,
                               kernel_precision: str | None = None,
                               verbose: bool) -> np.ndarray:
    """Per-pair offset sweep for a single (context, query) pair.

    ``ref_per_a`` is either ``None`` (auto-centroid) or a list of
    pre-validated per-attribute 1-D arrays (length ``n_attrs``).
    Returns the ``(M,)`` similarity profile.

    ``normalize``, ``truncation_sigmas`` and ``kernel_precision`` are
    forwarded to the per-offset :func:`_windowed_inner_product`
    calls. ``None`` on the optional flags defers to the global
    default (resolved inside the helper).
    """
    offsets = np.asarray(offsets, dtype=np.float64)
    if offsets.ndim == 1:
        offsets = offsets.reshape(-1, 1)
    if offsets.shape[0] != int(dens_context.dim):
        raise ValueError(
            f"offsets must have {int(dens_context.dim)} rows (dim of "
            f"dens_context); got shape {offsets.shape}."
        )
    M = offsets.shape[1]

    A = int(dens_query.n_attrs)
    dim_per_a = [int(d) for d in dens_query.dim_per_attr]

    if ref_per_a is None:
        # Default: unweighted mean of per-attribute tuple centres.
        ref_per_a = [dens_query.centres[a].mean(axis=1) for a in range(A)]

    # Strip any user-supplied 'centre' field; offsets replace it.
    base_spec = {k: v for k, v in window_spec.items() if k != "centre"}

    # --- Dispatch announce ---
    # windowed_tensor_similarity uses a single algorithmic path: the closed-form
    # windowed inner product (no Bulger / Möbius / centres choice to
    # make). The announce reads 'chose direct path' to surface the
    # method to the user; throttled to once per top-level call.
    from .._defaults import _maybe_show_dispatch_msg
    _maybe_show_dispatch_msg(
        "windowed_tensor_similarity", "direct",
        "closed-form windowed inner product (single algorithmic path)",
    )

    # --- Up-front time estimate + adaptive progress stride ---
    # Calibrate empirically (warm-up + timed sample) and extrapolate to
    # the full M-point sweep, matching the pattern used by the other
    # batched helpers (cos_sim_exp_tens batched-raw, entropy_exp_tens,
    # etc.). Threshold 10 s via maybe_print_batched_estimate; gated on
    # verbose. For M = 1 the calibration is skipped entirely.
    import time
    from .._utils import progress_stride

    prog_stride = 1
    show_progress = False
    # --- Pre-compute the unwindowed L2 norm of the query (denominator) ---
    # The unwindowed query self inner product <dens_q, dens_q> appears
    # in the denominator under both 'oneSidedDenom' (where it IS the
    # denominator) and 'cosine' (where it is one factor of the
    # geometric mean). It depends only on dens_query, not on the
    # window offset, so we compute it once and pass it as a cache to
    # every per-offset _windowed_inner_product call.
    norms_cache = _cos_sim_numerator_ma(dens_query, dens_query, windowed_c=None)

    if verbose and M >= 2:
        n_cal = min(5, M)
        sample_idx = np.unique(np.round(
            np.linspace(0, M - 1, n_cal)
        ).astype(int))

        def _build_wmd_for_idx(idx):
            off_col = offsets[:, idx]
            centre_list = []
            off_ptr = 0
            for a in range(A):
                da = dim_per_a[a]
                centre_list.append(ref_per_a[a] + off_col[off_ptr:off_ptr + da])
                off_ptr += da
            spec = dict(base_spec)
            spec["centre"] = centre_list
            return window_tensor(dens_context, spec)

        # Warm-up: one iteration of the loop body to absorb one-time
        # setup (cache populate, etc.) before the timed sample.
        wmd_w = _build_wmd_for_idx(sample_idx[0])
        _windowed_inner_product(dens_query, wmd_w, verbose=False,
                                _cached_norms=norms_cache,
                                normalize=normalize)

        # Timed calibration sample.
        t_cal_start = time.perf_counter()
        for cs in sample_idx:
            wmd_s = _build_wmd_for_idx(cs)
            _windowed_inner_product(dens_query, wmd_s, verbose=False,
                                    _cached_norms=norms_cache,
                                    normalize=normalize)
        t_cal_total = time.perf_counter() - t_cal_start
        t_per_point = t_cal_total / len(sample_idx)
        est_total = t_cal_total + t_per_point * M
        maybe_print_batched_estimate(
            "windowed_tensor_similarity", M, est_total,
        )
        prog_stride = progress_stride(t_per_point)
        show_progress = est_total >= 5

    profile = np.empty(M, dtype=np.float64)
    for m in range(M):
        off_col = offsets[:, m]
        centre_list = []
        off_ptr = 0
        for a in range(A):
            da = dim_per_a[a]
            centre_list.append(ref_per_a[a] + off_col[off_ptr:off_ptr + da])
            off_ptr += da
        spec_m = dict(base_spec)
        spec_m["centre"] = centre_list
        wmd = window_tensor(dens_context, spec_m)
        profile[m] = _windowed_inner_product(dens_query, wmd, verbose=False,
                                             _cached_norms=norms_cache,
                                             normalize=normalize)

        if verbose and show_progress and (
            (m + 1) % prog_stride == 0 or (m + 1) == M
        ):
            print(f"  {m + 1} / {M} points computed.")

    if verbose:
        print("windowed_tensor_similarity: done.")

    return profile


@_with_dispatch_scope
def windowed_tensor_similarity(dens_context, dens_query, window_spec, offsets, *,
                        reference=None, mode: str = "auto",
                        normalize: str | None = None,
                        normalise: str | None = None,
                        truncation_sigmas: float | None = None,
                        kernel_precision: str | None = None,
                        verbose: bool = True) -> np.ndarray:
    """Sliding-window similarity profile (cross-correlation).

    For each offset column, *dens_context* is windowed with
    *window_spec* at the corresponding centre, and the resulting
    windowed inner product against the unwindowed *dens_query* is
    finalised via the ``normalize`` keyword (default
    ``'oneSidedDenom'``).

    The two operands play asymmetric roles:

      * ``dens_context`` is the operand the window multiplies. As the
        sweep proceeds, the window shifts to each centre defined by
        the ``offsets`` matrix, selecting different regions of
        ``dens_context`` at each step.
      * ``dens_query`` is the unwindowed operand whose self inner
        product appears in the denominator. The query supplies the
        comparison template against which each windowed context
        region is scored.

    Normalisation: 'oneSidedDenom' versus 'cosine'
    ---------------------------------------------
    The numerator at each sweep position is the windowed inner
    product :math:`\\langle h \\, f_C, f_Q \\rangle`. The denominator
    depends on the ``normalize`` keyword:

      * ``'oneSidedDenom'`` (default) — divide by the unwindowed
        query self inner product
        :math:`\\langle f_Q, f_Q \\rangle`. The result is magnitude-
        aware: self-similarity at full window coverage equals 1,
        silent regions of the context score near zero, and a region
        where the windowed context has more matching mass than the
        query holds in total may score above 1. This is the intended
        reading for sliding-motif analysis -- a dense local match
        should outscore a sparse one.

      * ``'cosine'`` — divide by
        :math:`\\sqrt{\\langle h \\, f_C, h \\, f_C \\rangle \\,
        \\langle f_Q, f_Q \\rangle}`. The result is the strict
        shape-only cosine, bounded in :math:`[-1, 1]` and invariant
        to a positive scalar on either operand. Closed-form across
        the ``(size, mix)`` family only for pure-Gaussian
        (``mix = 0``) and pure-boxcar (``mix = 1``) windows;
        intermediate ``mix`` raises and directs the user to
        ``'oneSidedDenom'``.

    Periodic attributes
    -------------------
    For periodic attributes, the window is the wrapped Gaussian (or
    wrapped rect-conv-Gaussian for ``mix > 0``): the sum of line-case
    window functions at all periodic images of the centre. The
    toolbox sums these contributions adaptively, truncating when the
    latest image-pair's contribution falls below the floating-point
    threshold (1e-12 for double, 1e-7 for ``kernel_precision='single'``).
    For multi-D absolute periodic attributes, the image sum factorises per
    axis (linear in dimension, not exponential). For multi-D relative
    periodic attributes, image summation is deferred to a future release and
    the existing line-case formula is used.

    See USER_GUIDE §3.1 "Post-tensor windowing".

    Parameters
    ----------
    dens_context : MaetDensity, or list/tuple of MaetDensity
        The context density to be windowed (positional arg 1). A
        single density gives the scalar behaviour; a list/tuple is
        broadcast or paired against the query (see Returns).
    dens_query : MaetDensity, or list/tuple of MaetDensity
        The query density, unwindowed (positional arg 2). As above,
        scalar or list.
    window_spec : dict
        Window specification (see :func:`window_tensor`). Only the
        ``size`` and ``mix`` fields are read; any ``centre`` field is
        ignored (offsets replace it).
    offsets : (dim, M) array-like
        Per-sweep offsets in effective space, using the
        attribute-concatenated flat convention of
        :func:`window_tensor`. A 1-D array is accepted when dim == 1.
    reference : optional
        Reference point(s) for the offset frame. Three forms:

          * ``None`` (default): per-query auto-centroid (the unweighted
            mean of the query's tuple centres on each attribute). Peak
            offsets track ``P* − μ_q`` and so vary with the query.
          * length-``n_attrs`` list of 1-D arrays: shared reference,
            broadcast to every query. Each entry has length
            ``dim_per_attr[a]``.
          * length-``n_q`` list of (length-``n_attrs`` list of 1-D
            arrays): per-query reference, one full reference list per
            query in the batch.

        Disambiguation when both forms are syntactically possible is
        on element type: outer-list elements that are 1-D
        arrays/lists-of-numbers indicate the shared form; outer-list
        elements that are themselves lists/tuples indicate per-query.
    mode : {'auto', 'pairwise', 'cartesian'}, default 'auto'
        For list-vs-list. Ignored otherwise.
    normalize : {'oneSidedDenom', 'cosine'}, default 'oneSidedDenom'
        Selects the denominator applied to the windowed inner
        product. See "Normalisation" section above. The British
        spelling ``normalise`` is accepted as an alias keyword name;
        matching on the value is case-insensitive.
    verbose : bool

    Returns
    -------
    np.ndarray
        - scalar context, scalar query → ``(M,)``.
        - scalar context, list of n_q queries → ``(n_q, M)``.
        - list of n_c contexts, scalar query → ``(n_c, M)``.
        - list-vs-list, ``mode='pairwise'`` (requires n_c == n_q) →
          ``(n_c, M)``.
        - list-vs-list, ``mode='cartesian'`` → ``(n_c, n_q, M)``.

        Length-1 lists do NOT collapse to scalars (strict shape
        preservation).
    """
    # Accept ``normalize`` (canonical) or ``normalise`` (British alias).
    if normalize is not None and normalise is not None:
        raise TypeError(
            "Pass either 'normalize' or 'normalise', not both."
        )
    if normalize is None and normalise is None:
        normalize = "oneSidedDenom"
    elif normalize is None:
        normalize = normalise
    # Canonicalise (raises on bad value).
    from .cosine import _canonical_normalize
    normalize = _canonical_normalize(normalize)

    return _windowed_similarity_core(
        dens_context, dens_query, window_spec, offsets,
        reference=reference, mode=mode,
        normalize=normalize,
        truncation_sigmas=truncation_sigmas,
        kernel_precision=kernel_precision,
        verbose=verbose,
    )


def _windowed_similarity_core(dens_context, dens_query, window_spec, offsets, *,
                              reference=None, mode: str = "auto",
                              normalize: str = "oneSidedDenom",
                              truncation_sigmas: float | None = None,
                              kernel_precision: str | None = None,
                              verbose: bool = True):
    """Body of :func:`windowed_tensor_similarity`.

    ``normalize``, ``truncation_sigmas`` and ``kernel_precision`` are
    forwarded through to each per-offset :func:`_windowed_inner_product`
    call. ``None`` on the optional flags defers to the global default;
    explicit values flow through directly.
    """
    # Lazy import to avoid import cycles: dispatch is imported by
    # cosine and eval, which in turn import from windowing's
    # parent package at load time.
    from .dispatch import _normalize_density_input, _resolve_list_list_mode
    from .aniso import density_has_kernel_cov

    def _any_kernel_cov(op):
        seq = op if isinstance(op, (list, tuple)) else [op]
        return any(density_has_kernel_cov(d) for d in seq)

    if _any_kernel_cov(dens_context) or _any_kernel_cov(dens_query):
        raise NotImplementedError(
            "windowed_tensor_similarity does not support densities "
            "built with a matrix-valued kernel covariance: their stored "
            "coordinates are whitened, so windows and offsets (specified "
            "in original coordinates) would be applied in the wrong "
            "frame. Use windowed_similarity, whose sweep translates and "
            "windows the raw events before each density is built."
        )

    # ------------------------------------------------------------------
    # Normalise context and query inputs.
    # ------------------------------------------------------------------
    c_scalar, c_list = _normalize_density_input(dens_context, name="dens_context")
    q_scalar, q_list = _normalize_density_input(dens_query, name="dens_query")

    # Validate every density is a plain MaetDensity (not Windowed, not single-multiset).
    for label, scalar_flag, densities in (
        ("dens_context", c_scalar, c_list),
        ("dens_query", q_scalar, q_list),
    ):
        for i, d in enumerate(densities):
            if not isinstance(d, MaetDensity):
                idx = "" if scalar_flag else f"[{i}]"
                raise TypeError(
                    f"{label}{idx} must be a MaetDensity (not "
                    f"WindowedMaetDensity); got "
                    f"{type(d).__name__}."
                )

    n_c = len(c_list)
    n_q = len(q_list)

    # ------------------------------------------------------------------
    # Validate offsets shape and handle empty-list cases up front.
    # ------------------------------------------------------------------
    if n_c == 0 or n_q == 0:
        offsets_arr = np.asarray(offsets, dtype=np.float64)
        if offsets_arr.ndim == 1:
            offsets_arr = offsets_arr.reshape(-1, 1)
        M = offsets_arr.shape[1] if offsets_arr.ndim >= 2 else 0
        if c_scalar:
            return np.empty((0, M), dtype=np.float64)
        if q_scalar:
            return np.empty((0, M), dtype=np.float64)
        # both lists, at least one empty
        if mode == "auto":
            mode_resolved = "pairwise" if n_c == n_q else "cartesian"
        else:
            mode_resolved = mode
        if mode_resolved == "pairwise":
            return np.empty((0, M), dtype=np.float64)
        return np.empty((n_c, n_q, M), dtype=np.float64)

    # Resolve reference into a per-query list.
    references_per_query = _resolve_windowed_similarity_reference(
        reference, q_list,
    )

    # Common kwargs forwarded to every _windowed_similarity_pair call.
    _pair_kw = {
        "normalize": normalize,
        "truncation_sigmas": truncation_sigmas,
        "kernel_precision": kernel_precision,
        "verbose": verbose,
    }

    # ------------------------------------------------------------------
    # Scalar-vs-scalar.
    # ------------------------------------------------------------------
    if c_scalar and q_scalar:
        return _windowed_similarity_pair(
            c_list[0], q_list[0], window_spec, offsets,
            ref_per_a=references_per_query[0], **_pair_kw,
        )

    if c_scalar:
        # 1 context × n_q queries → (n_q, M).
        rows = [
            _windowed_similarity_pair(
                c_list[0], q, window_spec, offsets,
                ref_per_a=ref, **_pair_kw,
            )
            for q, ref in zip(q_list, references_per_query)
        ]
        return np.stack(rows, axis=0)

    if q_scalar:
        # n_c contexts × 1 query → (n_c, M).
        rows = [
            _windowed_similarity_pair(
                c, q_list[0], window_spec, offsets,
                ref_per_a=references_per_query[0], **_pair_kw,
            )
            for c in c_list
        ]
        return np.stack(rows, axis=0)

    # Both lists.
    resolved = _resolve_list_list_mode(mode, n_c, n_q)
    if resolved == "pairwise":
        rows = [
            _windowed_similarity_pair(
                c, q, window_spec, offsets,
                ref_per_a=ref, **_pair_kw,
            )
            for c, q, ref in zip(c_list, q_list, references_per_query)
        ]
        return np.stack(rows, axis=0)

    # Cartesian: row i = context i, column j = query j → (n_c, n_q, M).
    probe = _windowed_similarity_pair(
        c_list[0], q_list[0], window_spec, offsets,
        ref_per_a=references_per_query[0], **_pair_kw,
    )
    M = probe.shape[0]
    out = np.empty((n_c, n_q, M), dtype=np.float64)
    out[0, 0, :] = probe
    for j in range(1, n_q):
        out[0, j, :] = _windowed_similarity_pair(
            c_list[0], q_list[j], window_spec, offsets,
            ref_per_a=references_per_query[j], **_pair_kw,
        )
    for i in range(1, n_c):
        for j in range(n_q):
            out[i, j, :] = _windowed_similarity_pair(
                c_list[i], q_list[j], window_spec, offsets,
                ref_per_a=references_per_query[j], **_pair_kw,
            )
    return out


# -------------------------------------------------------------------
#  Cos-sim dispatch extension: unwindowed × windowed
# -------------------------------------------------------------------


def _windowed_inner_product(dens_a, dens_b, *, verbose: bool,
                            _cached_norms=None,
                            normalize: str = "oneSidedDenom"):
    """Closed-form windowed similarity with one operand windowed.

    The numerator is the windowed inner product
    :math:`\\langle h \\, f_C, f_Q \\rangle` where ``f_Q`` is the
    unwindowed query and ``h \\cdot f_C`` is the windowed context. The
    denominator depends on ``normalize``:

      * ``'oneSidedDenom'`` (default): divide by the unwindowed
        query self inner product, :math:`\\langle f_Q, f_Q \\rangle`.
        The result is magnitude-aware: self-similarity at full window
        coverage equals 1, silent regions of the context score near
        zero, and a region where the windowed context has more
        matching mass than the query holds in total may score above 1.

      * ``'cosine'``: divide by
        :math:`\\sqrt{\\langle h \\, f_C, h \\, f_C \\rangle \\,
        \\langle f_Q, f_Q \\rangle}`. The result is the strict
        shape-only cosine, bounded in :math:`[-1, 1]`. Closed-form
        across the ``(size, mix)`` family only for pure-Gaussian
        (``mix = 0``) and pure-boxcar (``mix = 1``) windows;
        intermediate ``mix`` raises and directs the user to
        ``'oneSidedDenom'``.

    Currently supports one-sided windowing (exactly one of dens_a,
    dens_b is a WindowedMaetDensity). Two-sided is not needed for the
    windowed_tensor_similarity use case.

    Internal optimisation: when called repeatedly with the same
    (dens_q, dens_c) pair (as :func:`windowed_tensor_similarity` does for
    each offset in its sweep), the unwindowed query self inner
    product depends only on ``dens_q`` and can be computed once
    outside the loop. Callers may pass it as
    ``_cached_norms = ip_qq`` to skip the redundant per-call work.
    """
    a_win = isinstance(dens_a, WindowedMaetDensity)
    b_win = isinstance(dens_b, WindowedMaetDensity)
    if a_win and b_win:
        raise NotImplementedError(
            "Two-sided windowing (both operands windowed) is not "
            "supported. Use windowed_tensor_similarity for profile sweeps."
        )

    if a_win:
        # Swap: put windowed operand on the 'b' side canonically.
        dens_q, wmd = dens_b, dens_a
    else:
        dens_q, wmd = dens_a, dens_b

    dens_c = wmd.dens

    # --- Structural compatibility checks (delegate to underlying _ma) ---
    _check_ma_compatibility(dens_q, dens_c)

    # --- Query's unwindowed self inner product ---
    if _cached_norms is not None:
        if isinstance(_cached_norms, tuple):
            ip_qq = _cached_norms[0]
        else:
            ip_qq = _cached_norms
    else:
        ip_qq = _cos_sim_numerator_ma(dens_q, dens_q, windowed_c=None)

    # --- Windowed cross inner product: <h * dens_c, dens_q> ---
    ip_qc = _cos_sim_numerator_ma(dens_q, dens_c, windowed_c=wmd)

    if normalize == "oneSidedDenom":
        if ip_qq == 0:
            return float("nan")
        return float(ip_qc / ip_qq)
    if normalize == "cosine":
        # Strict shape-only cosine: divide by
        # sqrt(<h*dens_c, h*dens_c> * <dens_q, dens_q>). Closed-form
        # for windows whose pointwise square h^2 is itself in the
        # (size, mix) family: pure Gaussian (mix = 0; h^2 is Gaussian
        # with size scaled by 1/sqrt(2)) and pure boxcar (mix = 1;
        # h^2 = h). Intermediate mix raises.
        wmd_squared = _window_squared(wmd)
        ip_cc_h = _cos_sim_numerator_ma(dens_c, dens_c, windowed_c=wmd_squared)
        denom = float(np.sqrt(max(ip_cc_h * ip_qq, 0.0)))
        if denom == 0:
            return float("nan")
        return float(ip_qc / denom)
    raise ValueError(
        f"normalize must be 'cosine' or 'oneSidedDenom'; got {normalize!r}."
    )


def _window_squared(wmd: "WindowedMaetDensity") -> "WindowedMaetDensity":
    """Construct a WindowedMaetDensity carrying the pointwise square
    :math:`h^2` of the original window :math:`h`.

    For each group ``g`` with ``mix_g == 0`` (pure Gaussian window of
    width ``size_g * sigma_g``), :math:`h^2` is itself a Gaussian of
    width :math:`(size_g \\cdot \\sigma_g) / \\sqrt 2`; equivalently
    ``size_g' = size_g / sqrt(2)`` with ``mix_g' = 0``.

    For each group ``g`` with ``mix_g == 1`` (pure boxcar window),
    :math:`h(z) \\in \\{0, 1\\}` pointwise so :math:`h^2 = h`;
    ``size_g`` and ``mix_g`` are unchanged.

    For ``mix_g`` in ``(0, 1)``, :math:`h^2` is the square of a
    rectangular-convolved-with-Gaussian and is not in the
    ``(size, mix)`` family; this case raises and directs the user to
    the ``'oneSidedDenom'`` option.
    """
    size_arr = np.array(wmd.size, dtype=np.float64, copy=True)
    mix_arr = np.array(wmd.mix, dtype=np.float64, copy=True)

    G = len(size_arr)
    for g in range(G):
        sz_g = size_arr[g]
        mix_g = mix_arr[g]

        if not np.isfinite(sz_g) or sz_g == 0:
            # Group is not windowed (size = inf or NaN); leave alone.
            continue

        if mix_g == 0:
            size_arr[g] = sz_g / np.sqrt(2)
        elif mix_g == 1:
            # h^2 = h; no change.
            pass
        else:
            raise ValueError(
                "Strict shape-only cosine (normalize = 'cosine') requires "
                "every attribute's window_spec['mix'] to be 0 (pure Gaussian) "
                f"or 1 (pure boxcar). Attribute {g} has mix = {mix_g}. Use "
                "normalize = 'oneSidedDenom' for intermediate mix values, "
                "or set the mix to 0 or 1."
            )

    spec = {
        "size":   size_arr,
        "mix":    mix_arr,
        "centre": wmd.centre,
    }
    return window_tensor(wmd.dens, spec)


def _check_ma_compatibility(dens_x: MaetDensity, dens_y: MaetDensity):
    """Structural compatibility check, mirroring _cos_sim_exp_tens_ma."""
    if dens_x.n_attrs != dens_y.n_attrs:
        raise ValueError("Both densities must have the same n_attrs.")
    if dens_x.n_attrs != dens_y.n_attrs:
        raise ValueError("Both densities must have the same n_attrs.")
    if not np.array_equal(dens_x.r, dens_y.r):
        raise ValueError("Both densities must have the same r.")
    if not np.array_equal(dens_x.sigma, dens_y.sigma):
        raise ValueError("Both densities must have the same sigma.")
    if not np.array_equal(dens_x.is_rel, dens_y.is_rel):
        raise ValueError("Both densities must have the same is_rel.")
    if not np.array_equal(dens_x.is_per, dens_y.is_per):
        raise ValueError("Both densities must have the same is_per.")
    per_mask = dens_x.is_per.astype(bool)
    if np.any(dens_x.period[per_mask] != dens_y.period[per_mask]):
        raise ValueError("Both densities must agree on periods of periodic groups.")


def _cos_sim_numerator_ma(dens_x: MaetDensity, dens_y: MaetDensity, *,
                          windowed_c, _skip_symmetrisation=False):
    """Compute a single inner product <f_x, W f_y> as a sum over (j, k)
    pairs. If *windowed_c* is None, it's the plain unwindowed inner
    product (used for norms and for unwindowed comparisons).

    Implements the full (size, mix) family closed-form for 1-D groups
    and multi-D absolute groups, and the Gaussian-only case for multi-D
    relative groups. Raises if (rho > 0) is requested on a multi-D
    relative group.

    Within-attribute symmetrisation. The MAET density is symmetric
    under permutations of components within each attribute's
    effective coordinates (JMM windowing-theorem remark on
    within-attribute symmetry of the windowed integral). The
    integral therefore depends on the within-attribute centre
    components only through their multiset. The toolbox uses a
    perm-comb summation for efficiency, which only matches the
    framework-correct integral when within-attribute centre
    components are uniform. For non-uniform within-attribute
    centres, this function detects the situation and averages the
    inner product over within-attribute permutations of the centre,
    one Cartesian product across all non-uniform attributes. Result
    matches the perm-perm (framework-correct) form. Uniform-centre
    inputs (the common case) bypass this and take the existing fast
    path.

    The ``_skip_symmetrisation`` parameter is a private recursion
    guard — set to ``True`` in the inner self-calls that consume one
    specific permuted centre, to prevent infinite re-entry.
    """
    # Lazy import to avoid import cycles with .dispatch (dispatch
    # is imported by cosine, which is imported via the parent
    # package init).
    from .dispatch import _compute_Q

    # ----- IP-level within-attribute centre symmetrisation -----
    # Detect non-uniform within-attribute centre vectors. If any are
    # present, average IPs over within-attribute centre permutations
    # to recover the framework-correct (perm-perm) value.
    if windowed_c is not None and not _skip_symmetrisation:
        non_uniform_attr_perms = []   # list of (a, list_of_perms)
        for a in range(dens_x.n_attrs):
            d_a = int(dens_x.dim_per_attr[a])
            if d_a < 2:
                continue
            # Only attributes that are actually windowed need
            # symmetrising.
            if not _is_windowed_attr(windowed_c.size[a], windowed_c.mix[a]):
                continue
            c_a = np.asarray(windowed_c.centre[a], dtype=np.float64)
            if c_a.shape[0] != d_a:
                continue
            if not np.allclose(c_a, c_a[0]):
                from itertools import permutations
                non_uniform_attr_perms.append(
                    (a, list(permutations(range(d_a))))
                )

        if non_uniform_attr_perms:
            # Enumerate Cartesian product of within-attribute permutations.
            # For each combination, construct a permuted wmd, recurse with
            # the symmetrisation guard set, sum.
            from itertools import product
            attr_indices = [a for (a, _) in non_uniform_attr_perms]
            perm_lists = [perms for (_, perms) in non_uniform_attr_perms]

            total_ip = 0.0
            n_combos = 0
            for combo in product(*perm_lists):
                centre_perm = list(windowed_c.centre)
                for a, sigma_a in zip(attr_indices, combo):
                    c_a = np.asarray(centre_perm[a], dtype=np.float64)
                    centre_perm[a] = c_a[list(sigma_a)]
                wmd_perm = WindowedMaetDensity(
                    tag=windowed_c.tag,
                    dens=windowed_c.dens,
                    size=windowed_c.size,
                    mix=windowed_c.mix,
                    centre=centre_perm,
                )
                ip_combo = _cos_sim_numerator_ma(
                    dens_x, dens_y, windowed_c=wmd_perm,
                    _skip_symmetrisation=True,
                )
                total_ip += ip_combo
                n_combos += 1

            return total_ip / n_combos
    # ----- end symmetrisation wrapper -----

    A          = dens_x.n_attrs
    r_vec      = dens_x.r
    sigma_g    = dens_x.sigma
    is_rel_g   = dens_x.is_rel
    is_per_g   = dens_x.is_per
    period_g   = dens_x.period

    n_jx, n_kx = dens_x.n_j, dens_x.n_k
    n_jy, n_ky = dens_y.n_j, dens_y.n_k

    # Use perm-side for x, comb-side for y (matching existing convention).
    # Evaluate pair factors per group and accumulate logs.
    n_jy_side, n_ky_side = int(n_jx), int(n_ky)
    # log_kernel[j, k] = sum_g log V_g(mu_j^(x), mu_k^(y), c_g if windowed)
    log_kernel = np.zeros((n_jy_side, n_ky_side), dtype=np.float64)

    # Track the window-dependent prefactor log(prod_g D_g). If not windowed
    # in that group, contribution is 0.
    log_prefactor = 0.0

    # =====================================================================
    # Cross-correlation translation (windowed case only).
    #
    # When windowed_c is not None, windowed_tensor_similarity asks for the cosine
    # similarity between the unwindowed query (dens_x) and the windowed
    # context (dens_y, with window spec wmd) interpreted as a CROSS-
    # CORRELATION: at sweep centre c in a windowed group g, the query is
    # translated so that its effective-space "position" mu_q_g moves onto
    # the window centre c_g. A peak at c_g thus means the query pattern
    # is present in the context near c_g.
    #
    # Mathematically this is done by the coordinate substitution
    #
    #     cx_g  ->  cx_g  - mu_q_g
    #     cy_g  ->  cy_g  - c_g
    #     c_g   ->  0
    #
    # inside the windowed-factor integrand (so the existing closed-form
    # helper is reused verbatim), and by adding a per-attribute shift
    #
    #     delta_a = { (c_g - mu_q_g)|_a            if group g is absolute
    #               { [0, (c_g - mu_q_g)|_a_eff]   if group g is relative
    #                                               (slot-0-anchored lift)
    #
    # to the r_a-slot tuple differences D = U - V before computing Q_a.
    # Groups that are not windowed receive no shift.
    #
    # For the unwindowed path (windowed_c is None) this whole block is
    # skipped and cos_sim_exp_tens semantics are preserved exactly.
    # =====================================================================
    shift_per_attr = {}     # attr index -> (r_a,) float64 shift to add to D
    eff_shift_per_attr = {} # attr index -> (d_a,) effective-space shift
    mu_q_per_attr = {}      # attr index -> (d_a,) effective-space query mean

    if windowed_c is not None:
        wmd = windowed_c
        dim_per = dens_x.dim_per_attr

        for a in range(A):
            if not _is_windowed_attr(wmd.size[a], wmd.mix[a]):
                continue

            # mu_q_a: unweighted mean over perm-side tuple centres of
            # the query's effective-space Gaussian centres for this
            # attribute. The unweighted mean gives the offset coordinate
            # a weight-independent meaning (see User Guide §3 on
            # windowing). For relative attributes this is (approximately)
            # zero by perm-symmetry, which is the correct convention:
            # translation-invariant attributes have no canonical
            # position. Where the window centre in a relative attribute
            # is non-zero in effective space, the shift is still applied
            # and lifted via slot-0 anchoring.
            r_a = int(r_vec[a])
            d_a = int(dim_per[a])
            mu_q_a = dens_x.centres[a].mean(axis=1)               # (d_a,)
            centre_a = np.asarray(wmd.centre[a], dtype=np.float64) # (d_a,)
            delta_a_eff = centre_a - mu_q_a                       # (d_a,)

            mu_q_per_attr[a] = mu_q_a
            eff_shift_per_attr[a] = delta_a_eff

            # Lift delta_a_eff into the attribute's r_a-slot shift.
            if bool(is_rel_g[a]):
                # Slot-0 anchored lift: shift[0]=0, shift[1:]=delta_a_eff.
                # For r_a == 1 and isRel=True the attribute is degenerate
                # (d_a == 0) and no shift is needed.
                if r_a == 1:
                    shift_a = np.zeros(1, dtype=np.float64)
                else:
                    shift_a = np.concatenate(
                        ([0.0], np.asarray(delta_a_eff, dtype=np.float64))
                    )
            else:
                # Absolute: effective dim equals r_a, direct mapping.
                shift_a = np.asarray(delta_a_eff, dtype=np.float64).copy()
            if shift_a.shape[0] != r_a:
                raise RuntimeError(
                    f"Internal: shift for attribute {a} has shape "
                    f"{shift_a.shape}, expected ({r_a},)."
                )
            shift_per_attr[a] = shift_a

    # ---- Main D / Q loop (with cross-correlation shift applied). ----
    for a in range(A):
        r_a = int(r_vec[a])
        U = dens_x.u_perm[a]   # (r_a, nJ_x)
        V = dens_y.v_comb[a]   # (r_a, nK_y)
        D = U[:, :, None] - V[:, None, :]  # (r_a, nJ_x, nK_y)

        # Apply per-attribute cross-correlation shift before wrap / Q_a.
        if a in shift_per_attr:
            D = D + shift_per_attr[a][:, None, None]

        # See note in cosine._ma_log_kernel: outer wrap is only needed
        # when _compute_Q does not re-wrap the pairwise component
        # differences (i.e., for is_per and not is_rel).
        if is_per_g[a] and not is_rel_g[a]:
            p_g = float(period_g[a])
            D = D - p_g * np.floor(D / p_g + 0.5)

        Q_a = _compute_Q(D, r_a, bool(is_rel_g[a]), bool(is_per_g[a]),
                         float(period_g[a]))
        log_kernel = log_kernel - Q_a / (4.0 * float(sigma_g[a]) ** 2)

    # ---- Windowed-factor contributions per attribute (cross-correlation
    # substitution applied). ----
    if windowed_c is not None:
        wmd = windowed_c
        eff_x_perm = _effective_centres_from_U(dens_x, side="perm")
        eff_y_comb = _effective_centres_from_V(dens_y, side="comb")

        for a in range(A):
            if not _is_windowed_attr(wmd.size[a], wmd.mix[a]):
                continue

            cx_g = eff_x_perm[a]                          # (d_a, nJ_x)
            cy_g = eff_y_comb[a]                          # (d_a, nK_y)
            centre_g = np.asarray(wmd.centre[a], dtype=np.float64)
            mu_q_g = mu_q_per_attr[a]                     # (d_a,)

            # Cross-correlation coordinate substitution: translate query
            # centres to origin via mu_q_g, translate context centres to
            # origin via centre_g, then apply the window at 0.
            cx_sub = cx_g - mu_q_g[:, None]
            cy_sub = cy_g - centre_g[:, None]
            centre_sub = np.zeros_like(centre_g)

            s_g = wmd.size[a]
            mix_g = wmd.mix[a]
            sigma_gv = float(sigma_g[a])
            is_rel = bool(is_rel_g[a])
            d_g = cx_sub.shape[0]

            if bool(is_per_g[a]):
                # Periodic attribute: sum line-case contributions over
                # periodic images of the window centre. The wrapped
                # Gaussian window equals the sum of line-case Gaussians
                # at all integer-multiples of the period; the windowed
                # inner product inherits this linearity. See User Guide
                # §3.1.
                contrib, log_D = _periodic_image_sum_contribution(
                    cx_sub, cy_sub, centre_sub,
                    s_g, mix_g, sigma_gv, is_rel,
                    int(r_vec[a]), d_g,
                    float(period_g[a]),
                    _IMAGE_SUM_TOL_DOUBLE,    # FP-precision relative tolerance
                )
            else:
                contrib, log_D = _windowed_group_contribution(
                    cx_sub, cy_sub, centre_sub,
                    s_g, mix_g, sigma_gv, is_rel,
                    int(r_vec[a]), d_g,
                )
            log_kernel = log_kernel + contrib
            log_prefactor = log_prefactor + log_D

    # Assemble numerator from log_kernel + prefactor.
    E = np.exp(log_kernel + log_prefactor)
    w_u = dens_x.w_j       # (nJ_x,)
    w_v = dens_y.wv_comb   # (nK_y,)
    return float(w_u @ (E @ w_v))


def _effective_centres_from_U(dens: "MaetDensity", side: str):
    """Return per-attribute effective-space centres of dens on the given side.

    ``side='perm'`` uses u_perm (r_a x nJ); ``side='comb'`` uses v_comb
    (r_a x nK). Effective-space centre is obtained by projecting the r
    slot values onto the group's effective (dim_per_attr) space:
      - absolute group: centre is the slot vector itself (dim = r).
      - relative group: centre is (r - 1)-dim reduced coords; compute
        via the first (r - 1) pairwise differences from the mean, or
        equivalently the identity-projector-on-zero-mean. The specific
        effective-space coordinates are already stored in
        dens.centres[a] (which uses the perm-side ordering).

    For now, only ``side='perm'`` is supported natively (returns
    dens.centres). For comb-side, we reconstruct from v_comb using the
    same reduction as the perm-side.
    """
    if side == "perm":
        return dens.centres
    raise NotImplementedError(
        "Only side='perm' is cached; comb-side centres are reconstructed "
        "elsewhere."
    )


def _effective_centres_from_V(dens: "MaetDensity", side: str):
    """Reconstruct effective-space centres from v_comb (comb-side).

    Uses the same reduction as build_exp_tens stores in ``dens.centres``
    for the perm side:
      - Absolute groups (isRel=False): effective-space centre is just
        v_comb[a] (r_a rows, used directly).
      - Relative groups (isRel=True) with r_a >= 2: effective-space
        centre is v_comb[a][1:, :] - v_comb[a][0:1, :], i.e., the (r_a - 1)
        differences of every slot from slot 0.
      - Relative groups with r_a = 1: empty (0, nK) array (degenerate).
    """
    A = dens.n_attrs
    is_rel_g = dens.is_rel
    r_vec = dens.r
    n_k = dens.n_k

    out = []
    for a in range(A):
        r_a = int(r_vec[a])
        V = dens.v_comb[a]                     # (r_a, nK)
        if not is_rel_g[a]:
            out.append(V)
        elif r_a >= 2:
            out.append(V[1:, :] - V[0:1, :])    # (r_a - 1, nK)
        else:
            out.append(np.empty((0, n_k), dtype=np.float64))
    return out


def _periodic_image_sum_contribution(
    cx_sub, cy_sub, centre_sub,
    s_g, mix_g, sigma_g, is_rel, r_a, d_g,
    period_g, image_tol,
):
    """Wrapped-Gaussian per-pair contribution for a periodic windowed group.

    The wrapped Gaussian window is the sum of line-case Gaussians at all
    periodic images of the window centre; the windowed inner product
    inherits this linearity (see User Guide §3.1). For groups where the
    per-pair F factor factorises across axes (1-D and multi-D absolute),
    the total image sum factorises into per-axis image sums:

        F_total = prod_i (sum_n_i F_axis_i(n_i * P))

    so we can sum each axis independently in O(n_max) calls per axis
    instead of O(n_max^d_g) over the Cartesian product. For multi-D
    relative groups the F factor does not factorise across axes and a
    Cartesian-product sum would be needed; that path is deferred to a
    future release. For now, multi-D relative periodic groups fall
    through to the existing line-case formula (the pre-v2.2 behaviour).

    Returns the same ``(log_F, log_D)`` signature as
    :func:`_windowed_group_contribution`.

    Parameters
    ----------
    cx_sub, cy_sub, centre_sub :
        As in :func:`_windowed_group_contribution`, in the post-
        cross-correlation-substituted frame.
    period_g : float
        Period of group g, > 0.
    image_tol : float
        Per-axis relative convergence tolerance for the image sum.

    Returns
    -------
    log_F : (nJ_x, nK_y) float64
    log_D : float
    """
    # Multi-D relative groups: defer to the existing line-case formula.
    # Image summation for the non-factorisable multi-D relative geometry
    # is future scope.
    if is_rel and d_g > 1:
        return _windowed_group_contribution(
            cx_sub, cy_sub, centre_sub,
            s_g, mix_g, sigma_g, is_rel, r_a, d_g,
        )

    # 1-D groups (any kind) and multi-D absolute groups: F factorises
    # per axis, so the image sum factorises into per-axis sums.
    a_rect, b_conv = _window_width_params(s_g, mix_g, sigma_g)

    # Effective variance of the (j, k) product Gaussian, per axis.
    # Absolute group:         sigma_pair^2 = sigma_g^2 / 2.
    # 1-D relative (r_a = 2): sigma_pair^2 = r_a * sigma_g^2 / 2.
    if is_rel:
        sigma_pair_sq = r_a * sigma_g**2 / 2.0
    else:
        sigma_pair_sq = sigma_g**2 / 2.0
    sigma_t_sq = sigma_pair_sq + b_conv**2
    sigma_t = np.sqrt(sigma_t_sq)

    # Per-pair midpoint m minus window centre c, per axis. Shape:
    # (d_g, nJ_x, nK_y).
    m = 0.5 * (cx_sub[:, :, None] + cy_sub[:, None, :])
    mu_shift = m - centre_sub[:, None, None]

    # Build a closure to evaluate the per-axis F at a given mu value
    # (matching the inner formula of _windowed_contribution_factorisable).
    if a_rect == 0.0 and b_conv > 0.0:
        # Pure Gaussian window.
        prefactor = b_conv / sigma_t
        def axis_F(mu_arr):
            return prefactor * np.exp(-mu_arr**2 / (2 * sigma_t_sq))
    elif b_conv == 0.0 and a_rect > 0.0:
        # Pure rectangular window.
        from scipy.special import erf
        denom = sigma_t * np.sqrt(2.0)
        def axis_F(mu_arr):
            arg_plus = (mu_arr + a_rect) / denom
            arg_minus = (mu_arr - a_rect) / denom
            return 0.5 * (erf(arg_plus) - erf(arg_minus))
    else:
        # General rectangular-convolved-with-Gaussian window.
        from scipy.special import erf
        denom = sigma_t * np.sqrt(2.0)
        norm_denom = 2.0 * erf(a_rect / (b_conv * np.sqrt(2.0)))
        def axis_F(mu_arr):
            arg_plus = (mu_arr + a_rect) / denom
            arg_minus = (mu_arr - a_rect) / denom
            return (erf(arg_plus) - erf(arg_minus)) / norm_denom

    # Per-axis image sum: for each axis i, F_axis_i_wrapped(j, k) =
    # sum_n F_axis_i(mu_shift_i + n*P). Stop when the latest |n|-pair's
    # max contribution falls below image_tol * running max.
    n_max_cap = 100
    per_axis_wrapped = np.empty_like(mu_shift)
    for i in range(d_g):
        mu_i = mu_shift[i]
        # Reduce to the minimal image so the n=0 term is dominant (see
        # _wrapped_window_factor_1d): a midpoint many periods from the
        # centre would otherwise underflow the near images to 0 and the
        # sum would terminate before reaching the dominant image.
        if period_g > 0.0:
            mu_i = mu_i - period_g * np.round(mu_i / period_g)
        acc = axis_F(mu_i)
        running_max = float(np.max(np.abs(acc)))
        for n in range(1, n_max_cap + 1):
            shift_arg_pos = mu_i + n * period_g
            shift_arg_neg = mu_i - n * period_g
            F_pos = axis_F(shift_arg_pos)
            F_neg = axis_F(shift_arg_neg)
            acc = acc + F_pos + F_neg
            new_max = float(max(np.max(np.abs(F_pos)),
                                np.max(np.abs(F_neg))))
            running_max = max(running_max, float(np.max(np.abs(acc))))
            if running_max == 0.0:
                break
            if new_max / running_max < image_tol:
                break
        else:
            import warnings as _warnings
            _warnings.warn(
                f"Periodic image sum on axis {i} hit the safety cap "
                f"of {n_max_cap} image pairs without converging to "
                f"relative tolerance {image_tol:g}. This usually "
                f"indicates sigma_w >> P, in which case the windowed "
                f"inner product approaches the unwindowed one; call "
                f"cos_sim_exp_tens directly instead.",
                RuntimeWarning, stacklevel=4,
            )
        per_axis_wrapped[i] = acc

    # Guard against tiny-or-negative values before taking log.
    per_axis_wrapped = np.clip(per_axis_wrapped, 1e-300, None)
    log_F = np.sum(np.log(per_axis_wrapped), axis=0)
    return log_F, 0.0


def _windowed_group_contribution(cx_g, cy_g, centre_g,
                                  s_g, mix_g, sigma_g, is_rel, r_a, d_g):
    """Compute log(V_g / U_g) as a (nJ_x, nK_y) array, and log(D_g) scalar.

    cx_g : (d_g, nJ_x) effective-space perm-side centres for x in group g.
    cy_g : (d_g, nK_y) effective-space comb-side centres for y in group g.
    centre_g : (d_g,) window centre in group g's effective subspace.
    s_g : window size (in sigma multiples).
    mix_g : window mix in [0, 1].
    sigma_g : group sigma.
    is_rel : whether group is relative.
    r_a : tuple size of any attribute in the group (they all share it).
    d_g : group effective dim.

    Returns
    -------
    log_ratio : (nJ_x, nK_y) float64
    log_D : float64
        log of the window-dependent constant prefactor D_g.
    """
    a_, b_ = _window_width_params(s_g, mix_g, sigma_g)

    # Determine case. "1-D" here means d_g == 1; "multi-D absolute" means
    # d_g >= 2 and not is_rel; "multi-D relative" means d_g >= 2 and is_rel.
    is_1d = (d_g == 1)
    is_multi_abs = (d_g >= 2) and (not is_rel)
    is_multi_rel = (d_g >= 2) and is_rel

    rho = float(mix_g)

    if is_multi_rel and rho > 0:
        raise NotImplementedError(
            f"Multi-D relative groups (d_g = {d_g}, r_a = {r_a}) do not "
            f"support rectangular or raised-rectangular windows (mix = "
            f"{rho}). Use mix = 0 (pure Gaussian window), or "
            f"wait for a future release with Gaussian-mixture-window "
            f"approximation."
        )

    if is_1d or is_multi_abs:
        # --- Full (size, mix) family: per-axis closed form in erf. ---
        return _windowed_contribution_factorisable(
            cx_g, cy_g, centre_g, a_, b_, sigma_g, is_rel, r_a, d_g,
        )

    # --- Multi-D relative, Gaussian window (rho = 0). ---
    # At this point we know: is_multi_rel and rho == 0, so b_ = s * sigma
    # and a_ = 0.
    return _windowed_contribution_gaussian_multi_rel(
        cx_g, cy_g, centre_g, b_, sigma_g, r_a, d_g,
    )


def _windowed_contribution_factorisable(cx_g, cy_g, centre_g,
                                         a_rect, b_conv, sigma_g,
                                         is_rel, r_a, d_g):
    """Closed-form windowed-vs-unwindowed log-ratio per (j, k) pair.

    Covers: 1-D groups (any type) and multi-D absolute groups, where
    the integrand factorises per axis.

    Mathematical form. For one pair (j, k) in one axis of the group:
    the unwindowed integrand is two toolbox-convention Gaussian kernels
    (exponent -(u - mu)^2/(2 sigma^2)) multiplied together. Their
    product is a wider Gaussian centred at m = (mu_j + mu_k)/2 with
    variance sigma^2/2, times a scalar factor. The unwindowed pair
    integral already sits inside the existing unwindowed machinery (the
    scalar factor is exp(-(mu_j - mu_k)^2 / (4 sigma^2)) — exactly what
    _ma_log_kernel computes — and the Gaussian integrates to a constant
    that cancels under Option Z normalisation).

    The windowed pair integral equals the unwindowed one times the
    "excess factor" F_g, which is the integral of the product Gaussian
    against the window, normalised so F_g = 1 when the window is
    constant (size -> infinity).

    For the rectangular-convolved-with-Gaussian window family,

        F_g = [erf((mu + a)/(sigma_t sqrt 2)) - erf((mu - a)/(sigma_t sqrt 2))]
              / (2 erf(a / (b sqrt 2)))

    where mu = m - c, and sigma_t^2 = sigma_pair^2 + b^2 with
    sigma_pair^2 = sigma_g^2/2 (absolute) or r_a * sigma_g^2/2 (1-D
    relative with r_a = 2). The denominator normalises the window to
    have peak value 1; the numerator is the unnormalised Gaussian
    integrated against a boxcar of half-width a, widened by the
    window's Gaussian-convolution component.

    The formula covers the whole (size, mix) family; limits are:
        rho = 0  (a = 0, b = s*sigma): F_g -> pure-Gaussian-window
                  formula, reducing via l'Hopital to
                  (b/sigma_t) exp(-mu^2 / (2 sigma_t^2)).
        rho = 1  (a = s*sigma*sqrt(3), b = 0): F_g -> boxcar-integral
                  formula, (1/2)[erf((mu+a)/sigma) - erf((mu-a)/sigma)].
        size -> infinity: F_g -> 1.

    For multi-D absolute groups, F_g factorises across axes as the
    product of per-axis F values (the integrand is isotropic in
    effective-space coordinates, so the multi-D integral reduces to a
    product of per-axis integrals).

    Returns
    -------
    log_F : (nJ_x, nK_y) float64
        log(F_g) per pair, which is what gets added to the existing
        unwindowed log-kernel to produce the windowed log-kernel.
    log_D : float
        Window-dependent prefactor outside the per-pair computation.
        Zero under Option Z (window-free integration constants cancel
        between numerator and denominator because the numerator and
        both unwindowed norms share the same per-group integration
        factors).
    """
    from scipy.special import erf

    # Effective variance of the (j, k) product Gaussian, per axis.
    # Absolute group:         sigma_pair^2 = sigma_g^2 / 2.
    # 1-D relative (r_a = 2): sigma_pair^2 = r_a * sigma_g^2 / 2.
    if is_rel:
        sigma_pair_sq = r_a * sigma_g**2 / 2.0
    else:
        sigma_pair_sq = sigma_g**2 / 2.0

    sigma_t_sq = sigma_pair_sq + b_conv**2
    sigma_t = np.sqrt(sigma_t_sq)

    # Per-pair midpoint m minus window centre c.
    m = 0.5 * (cx_g[:, :, None] + cy_g[:, None, :])    # (d_g, nJ_x, nK_y)
    mu_shift = m - centre_g[:, None, None]              # (d_g, nJ_x, nK_y)

    # Per-axis factor F(mu_shift; a_rect, b_conv, sigma_t).
    # Three boundary cases handled: a=0 (pure Gaussian), b=0 (pure
    # rectangular), and general (both > 0).
    if a_rect == 0.0 and b_conv > 0.0:
        # Pure Gaussian window. L'Hopital on the general formula gives:
        # F = (b / sigma_t) * exp(-mu_shift^2 / (2 sigma_t^2))
        # per axis.
        per_axis = (b_conv / sigma_t) * np.exp(-mu_shift**2 / (2 * sigma_t_sq))
    elif b_conv == 0.0 and a_rect > 0.0:
        # Pure rectangular window. sigma_t = sqrt(sigma_pair^2) here.
        # F = 0.5 * [erf((mu + a)/(sigma_t sqrt(2))) - erf((mu - a)/(sigma_t sqrt(2)))]
        denom = sigma_t * np.sqrt(2.0)
        arg_plus = (mu_shift + a_rect) / denom
        arg_minus = (mu_shift - a_rect) / denom
        per_axis = 0.5 * (erf(arg_plus) - erf(arg_minus))
    else:
        # General case (0 < rho < 1). Normalised rect-conv-Gaussian window.
        denom = sigma_t * np.sqrt(2.0)
        arg_plus = (mu_shift + a_rect) / denom
        arg_minus = (mu_shift - a_rect) / denom
        numer = erf(arg_plus) - erf(arg_minus)
        norm_denom = 2.0 * erf(a_rect / (b_conv * np.sqrt(2.0)))
        per_axis = numer / norm_denom

    # Guard against tiny-or-negative values before taking log (numerical
    # noise can give very small negatives at large distances).
    per_axis = np.clip(per_axis, 1e-300, None)
    log_F = np.sum(np.log(per_axis), axis=0)             # (nJ_x, nK_y)
    return log_F, 0.0


def _windowed_contribution_gaussian_multi_rel(cx_g, cy_g, centre_g,
                                                b_conv, sigma_g,
                                                r_a, d_g):
    """Pure-Gaussian-window contribution on a multi-D relative group.

    Only supported when rho = 0 (i.e., mix = 0, so a_rect = 0 and
    b_conv = size * sigma). For rho > 0 on a multi-D relative group,
    the caller raises NotImplementedError.

    Derivation. The per-pair product Gaussian lives in the group's
    (r_a - 1)-dimensional reduced space (effective space). In the
    "subtract the mean and drop the last coordinate" reduction used by
    build_exp_tens, the quadratic form on the reduced coordinates v is
    v^T A_g v, where

        A_g = I_{r_a - 1} + 1 1^T    (ones-matrix addition, size d_g x d_g),

    with eigenvalues 1 (multiplicity r_a - 2) and r_a (multiplicity 1).
    This gives single-density precision A_g / (2 sigma_g^2), and the
    (j, k) product Gaussian has precision A_g / sigma_g^2 and covariance
    Sigma_pair = sigma_g^2 * A_g^{-1}.

    Adding an isotropic Gaussian window of variance b^2 * I gives
    combined covariance Sigma_K = Sigma_pair + b^2 I. The
    excess factor F_g (normalised so F_g = 1 at size -> infinity) is

        F_g = sqrt(det(Sigma_pair) / det(Sigma_K))
              * exp(-0.5 * (m - c)^T Sigma_K^{-1} (m - c))

    where m is the pair midpoint and c is the window centre, both in
    reduced coordinates.

    Returns
    -------
    log_F : (nJ_x, nK_y) float64
    log_D : float
        Zero; the normalisation-related prefactor cancels between the
        windowed numerator and the unwindowed norms under Option Z
        (the unwindowed integration constants are the same per group
        regardless of window).
    """
    # In the "drop first slot, v[i] = u[i+1] - u[1]" reduction used by
    # build_exp_tens, the quadratic form in reduced coords is
    #
    #     Q(v) = v^T M_rel v    where M_rel = I - (1/r) * 1 1^T  (size d_g).
    #
    # The single-density kernel is exp(-Q/(4 sigma_g^2)), i.e. a Gaussian
    # with precision M_rel / (2 sigma_g^2) and covariance
    # 2 sigma_g^2 * M_rel^{-1}. The inverse is
    #
    #     M_rel^{-1} = I + 1 1^T    (size d_g x d_g, using Sherman-Morrison).
    #
    # Product-of-two-densities precision = M_rel / sigma_g^2,
    # covariance Sigma_pair = sigma_g^2 * M_rel^{-1} = sigma_g^2 (I + 1 1^T).
    A_g_inv = np.eye(d_g) + np.ones((d_g, d_g))   # = M_rel^{-1}
    Sigma_pair = sigma_g**2 * A_g_inv

    # Gaussian window covariance (isotropic).
    T = b_conv**2 * np.eye(d_g)

    # Combined covariance for the K factor.
    Sigma_K = Sigma_pair + T
    K_precision = np.linalg.inv(Sigma_K)

    # Normalisation factor log prefactor (per group, constant across pairs).
    det_pair = np.linalg.det(Sigma_pair)
    det_K = np.linalg.det(Sigma_K)
    log_prefactor = 0.5 * (np.log(det_pair) - np.log(det_K))

    # Per-pair midpoint m - c.
    m = 0.5 * (cx_g[:, :, None] + cy_g[:, None, :])    # (d_g, nJ_x, nK_y)
    mu_shift = m - centre_g[:, None, None]              # (d_g, nJ_x, nK_y)

    # Quadratic form (mu_shift)^T Sigma_K^{-1} (mu_shift), per pair.
    # Result shape: (nJ_x, nK_y).
    temp = np.einsum("ij,jab->iab", K_precision, mu_shift)
    K_quad = np.einsum("iab,iab->ab", mu_shift, temp)

    log_F = log_prefactor - 0.5 * K_quad
    return log_F, 0.0


# -------------------------------------------------------------------
#  MAET input helper: simplex coordinates for categorical encoding
# -------------------------------------------------------------------


