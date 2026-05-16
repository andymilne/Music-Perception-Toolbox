"""Post-tensor windowing and sliding-window similarity profiles.

This module hosts the windowing layer of the MAET pipeline:

* :func:`window_tensor` --- wrap a :class:`MaetDensity` with a per-group
  window specification (size, mix, optional centre), returning a
  :class:`WindowedMaetDensity`. No math is done at construction time;
  windowing is applied lazily.
* :func:`windowed_similarity` --- sweep a query density's centroid
  across a context density via offset positions, returning a 1xM
  windowed-similarity profile (or list of profiles for list-mode
  input).
* :class:`WindowedSimilarityPeriodicApproxWarning` --- the warning
  raised when a windowed-similarity computation applies the line-case
  closed-form expression to a periodic group; see USER_GUIDE §3.1
  ("Periodic groups: line-case approximation").

The closed-form windowed inner product (:func:`_windowed_inner_product`)
is also exposed at module level for internal use by the cosine-similarity
machinery in :mod:`mpt.tensor` (which currently still hosts the
non-windowed cosine path).

See USER_GUIDE §3.1 ("Post-tensor windowing") for the user-facing
description and :doc:`/ARCHITECTURE` §2 for the layering.

Cross-module dependencies: this module reaches into the not-yet-migrated
parts of :mod:`mpt.tensor` for a small number of dispatch helpers
(``_normalize_density_input``, ``_resolve_list_list_mode``,
``_compute_Q``). Those imports are deferred to call time to avoid
import-cycle issues during package load. After Tranche-2 phase 3
of the refactor, the helpers will live in a sibling
``_tensor.dispatch`` module and these lazy imports can become
top-level.
"""
from __future__ import annotations

import warnings

import numpy as np

from .._utils import maybe_print_batched_estimate, validate_weights
from .._defaults import _with_dispatch_scope
from .density import (
    ExpTensDensity,
    MaetDensity,
    WindowedMaetDensity,
)


def window_tensor(dens, window_spec) -> WindowedMaetDensity:
    """Wrap a MaetDensity with a post-tensor window specification.

    Returns a :class:`WindowedMaetDensity` that bundles the underlying
    density with a window spec. No math is performed at construction
    time; the window is applied lazily by :func:`eval_exp_tens` and
    :func:`cos_sim_exp_tens`. See the MAET specification §4.3.

    Parameters
    ----------
    dens : MaetDensity
        The density to be windowed.
    window_spec : dict
        Window specification with keys:

        ``size`` : scalar or length-G array
            Per-group window effective standard deviation in multiples
            of that group's ``sigma``. NaN or Inf means the group is
            not windowed. A scalar is broadcast across all groups.
        ``mix`` : scalar or length-G array
            Per-group shape parameter in [0, 1]: 0 = pure Gaussian,
            1 = pure rectangular, in between = rectangular-convolved-
            with-Gaussian. A scalar is broadcast.
        ``centre`` : length-A list of array-like, or a single array-like
            Per-attribute centre coordinates. Each entry has length
            ``dim_per_attr[a]``. A single 1-D array of total length
            ``dim`` is split across attributes in order. Entries whose
            attribute's group has ``size`` NaN/Inf are ignored.

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

    A = dens.n_attrs
    G = dens.n_groups
    dim_per = dens.dim_per_attr
    dim_total = int(dens.dim)

    # --- size ---
    size_arr = np.asarray(window_spec.get("size"), dtype=np.float64).ravel()
    if size_arr.size == 1:
        size_arr = np.full(G, float(size_arr.item()))
    if size_arr.size != G:
        raise ValueError(
            f"window_spec['size'] must be a scalar or length-{G} array; "
            f"got length {size_arr.size}."
        )

    # --- mix ---
    mix_arr = np.asarray(window_spec.get("mix"), dtype=np.float64).ravel()
    if mix_arr.size == 1:
        mix_arr = np.full(G, float(mix_arr.item()))
    if mix_arr.size != G:
        raise ValueError(
            f"window_spec['mix'] must be a scalar or length-{G} array; "
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


def _is_windowed_group(size_g, mix_g):
    """True iff the group has an effective window (size finite and > 0)."""
    return np.isfinite(size_g) and size_g > 0


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
    group_of = dens.group_of_attr
    dim_per = dens.dim_per_attr
    sigma_g = dens.sigma

    n_q = x_list[0].shape[1] if A > 0 else 0
    result = np.ones(n_q, dtype=np.float64)
    for a in range(A):
        g = int(group_of[a])
        if not _is_windowed_group(wmd.size[g], wmd.mix[g]):
            continue
        a_, b_ = _window_width_params(wmd.size[g], wmd.mix[g], sigma_g[g])
        da = int(dim_per[a])
        # Per-axis factor, multiplied across axes within the attribute.
        centre_a = wmd.centre[a]  # shape (da,)
        x_a = x_list[a]           # shape (da, nQ)
        u = x_a - centre_a[:, None]  # (da, nQ)
        for i in range(da):
            result = result * _window_factor_1d(u[i], a_, b_)
    return result


# -------------------------------------------------------------------
#  windowed_similarity and closed-form pair-factor evaluator
# -------------------------------------------------------------------


class WindowedSimilarityPeriodicApproxWarning(UserWarning):
    """Issued when :func:`windowed_similarity` applies the line-case
    closed-form windowed inner product to a periodic group whose
    window standard deviation lambda*sigma is at least one quarter of
    the period P. The closed form in use is the small-window
    approximation; it degrades as window support approaches one
    period. For windows larger than a period the windowed inner
    product collapses to the unwindowed form, which can be obtained
    directly from :func:`cos_sim_exp_tens`. See User Guide §3.1
    "Post-tensor windowing" for the three-regime analysis.

    A module-level filter registers this warning with the ``"always"``
    action so that it fires on every offending call (rather than once
    per location, which is the default for :class:`UserWarning`),
    matching the MATLAB ``warning(id, ...)`` behaviour. Suppress it,
    if you have determined the approximation is acceptable for your
    use case, with::

        import warnings
        from mpt import WindowedSimilarityPeriodicApproxWarning
        warnings.filterwarnings(
            "ignore",
            category=WindowedSimilarityPeriodicApproxWarning,
        )
    """


# Ensure this warning is always shown, not only on the first
# occurrence at a given (module, lineno). Matches the per-call
# warning behaviour of the MATLAB implementation.
warnings.filterwarnings(
    "always", category=WindowedSimilarityPeriodicApproxWarning
)


def _resolve_windowed_similarity_reference(reference, q_list):
    """Resolve the polymorphic ``reference`` argument of
    :func:`windowed_similarity` to a list (length ``n_q``) of per-query
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


def _emit_periodic_approx_warnings(dens_context, window_spec):
    """Emit a ``WindowedSimilarityPeriodicApproxWarning`` for each
    periodic group whose window crosses the recommended SD/P bound.

    Two regimes are distinguished:
      * Within bound: brief informational notice that the line-case
        formula is in use; the approximation is sub-percent across
        the window shape family.
      * Past bound: stronger notice describing per-mix behaviour.

    See User Guide §3.1 "Post-tensor windowing".
    """
    SD_OVER_P_BOUND = 1.0 / (2.0 * np.sqrt(3.0))   # ~= 0.2887
    G = int(dens_context.n_groups)
    size_arr = np.atleast_1d(
        np.asarray(window_spec["size"], dtype=np.float64)
    ).ravel()
    if size_arr.size == 1:
        size_arr = np.repeat(size_arr, G)
    mix_arr = np.atleast_1d(
        np.asarray(window_spec["mix"], dtype=np.float64)
    ).ravel()
    if mix_arr.size == 1:
        mix_arr = np.repeat(mix_arr, G)
    for g in range(G):
        if not bool(dens_context.is_per[g]):
            continue
        lam = float(size_arr[g])
        if not np.isfinite(lam) or lam <= 0:
            continue
        period_g = float(dens_context.period[g])
        if period_g <= 0:
            continue
        eff_sigma = lam * float(dens_context.sigma[g])
        sd_over_p = eff_sigma / period_g
        gamma_g = float(mix_arr[g])

        if sd_over_p <= SD_OVER_P_BOUND:
            msg = (
                f"Periodic windowed inner product on group {g} "
                f"applies the line-case formula at wrapped "
                f"differences -- an approximation that retains "
                f"only the leading periodic image of the window. "
                f"Within the recommended bound, the approximation "
                f"is sub-percent across the window shape family.\n"
                f"  Window SD (lambda*sigma) = {eff_sigma:g}\n"
                f"  Period P                 = {period_g:g}\n"
                f"  SD/P                     = {sd_over_p:.4f}\n"
                f"  Recommended bound (SD/P) = {SD_OVER_P_BOUND:.4f} "
                f"(= 1/(2*sqrt(3)))\n"
                f"See User Guide \u00a73.1 \"Post-tensor windowing\". "
                f"Suppress with warnings.filterwarnings('ignore', "
                f"category=mpt."
                f"WindowedSimilarityPeriodicApproxWarning)."
            )
        else:
            phi_g = eff_sigma * np.sqrt(3.0 * max(gamma_g, 0.0))
            msg = (
                f"Window SD exceeds the recommended bound for "
                f"periodic group {g}; the line-case approximation "
                f"is no longer reliable.\n"
                f"  Window SD (lambda*sigma) = {eff_sigma:g}\n"
                f"  Period P                 = {period_g:g}\n"
                f"  SD/P                     = {sd_over_p:.4f}  "
                f"(bound: {SD_OVER_P_BOUND:.4f})\n"
                f"  phi (rect half-width)    = {phi_g:g}  "
                f"(bound: {period_g / 2:g} = P/2)\n"
                f"  mix (gamma)              = {gamma_g:g}\n"
                f"Beyond the bound, behaviour depends on mix:\n"
                f"  mix = 1 (pure rect):     window is no longer "
                f"localized on the circle (pointless as a window).\n"
                f"  mix = 0 (pure Gaussian): line-case approximation "
                f"degrades smoothly; error grows with SD/P.\n"
                f"  intermediate mix:        between these two cases.\n"
                f"Reduce size or sigma so that lambda*sigma <= "
                f"P/(2*sqrt(3)). See User Guide \u00a73.1 "
                f"\"Post-tensor windowing\"."
            )

        warnings.warn(
            msg,
            WindowedSimilarityPeriodicApproxWarning,
            stacklevel=3,
        )


def _windowed_similarity_pair(dens_query, dens_context, window_spec, offsets,
                               *, ref_per_a,
                               truncation_sigmas: float | None = None,
                               kernel_precision: str | None = None,
                               verbose: bool) -> np.ndarray:
    """Per-pair offset sweep for a single (query, context) pair.

    ``ref_per_a`` is either ``None`` (auto-centroid) or a list of
    pre-validated per-attribute 1-D arrays (length ``n_attrs``).
    Returns the ``(M,)`` similarity profile.

    ``truncation_sigmas`` and ``kernel_precision`` are forwarded to the
    per-offset :func:`_windowed_inner_product` calls. ``None`` defers to
    the global default (resolved inside the helper).

    Emits :class:`WindowedSimilarityPeriodicApproxWarning` once per
    (query, context) pair for any periodic group whose window crosses
    the recommended SD/P bound.
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

    # Periodic-window approximation warnings . Emitted once per
    # (query, context) pair, before the offset loop.
    _emit_periodic_approx_warnings(dens_context, window_spec)

    A = int(dens_query.n_attrs)
    dim_per_a = [int(d) for d in dens_query.dim_per_attr]

    if ref_per_a is None:
        # Default: unweighted mean of per-attribute tuple centres.
        ref_per_a = [dens_query.centres[a].mean(axis=1) for a in range(A)]

    # Strip any user-supplied 'centre' field; offsets replace it.
    base_spec = {k: v for k, v in window_spec.items() if k != "centre"}

    # --- Dispatch announce ---
    # windowed_similarity uses a single algorithmic path: the closed-form
    # windowed inner product (no Bulger / Möbius / centres choice to
    # make). The announce reads 'chose direct path' to surface the
    # method to the user; throttled to once per top-level call.
    from .._defaults import _maybe_show_dispatch_msg
    _maybe_show_dispatch_msg(
        "windowed_similarity", "direct",
        "closed-form windowed inner product (single algorithmic path)",
        0.0, False,
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
        _windowed_inner_product(dens_query, wmd_w, verbose=False)

        # Timed calibration sample.
        t_cal_start = time.perf_counter()
        for cs in sample_idx:
            wmd_s = _build_wmd_for_idx(cs)
            _windowed_inner_product(dens_query, wmd_s, verbose=False)
        t_cal_total = time.perf_counter() - t_cal_start
        t_per_point = t_cal_total / len(sample_idx)
        est_total = t_cal_total + t_per_point * M
        maybe_print_batched_estimate(
            "windowed_similarity", M, est_total,
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
        profile[m] = _windowed_inner_product(dens_query, wmd, verbose=False)

        if verbose and show_progress and (
            (m + 1) % prog_stride == 0 or (m + 1) == M
        ):
            print(f"  {m + 1} / {M} points computed.")

    if verbose:
        print("windowed_similarity: done.")

    return profile


@_with_dispatch_scope
def windowed_similarity(dens_query, dens_context, window_spec, offsets, *,
                        reference=None, mode: str = "auto",
                        truncation_sigmas: float | None = None,
                        kernel_precision: str | None = None,
                        verbose: bool = True) -> np.ndarray:
    """Sliding-window similarity profile (cross-correlation).

    For each offset column, *dens_context* is windowed with
    *window_spec* at the corresponding centre, and its similarity
    against *dens_query* (unwindowed) is computed. The normaliser
    uses the unwindowed L2 norms of both operands (Option Z in the
    spec).

    Note on naming
    --------------
    This function was named ``windowed_cos_sim`` in earlier drafts.
    The output is a magnitude-aware *windowed similarity*: because
    the denominator uses unwindowed L2 norms (rather than the
    windowed norm of the context), the profile is not bounded in
    [-1, 1] across sweep positions and does not correspond to an
    inner product on a single Hilbert space. This is the intended
    behaviour for sliding-motif analysis -- a dense local match
    should outscore a sparse one -- but it means "cosine similarity"
    is not the right name for the object. The strict shape-only
    cosine similarity (with windowed denominator) is reserved as a
    separate notion in the manuscript and is not currently
    implemented in the toolbox. See manuscript §5.4.

    Periodic groups
    ---------------
    The closed-form windowed inner product implemented here is the
    line-case formula -- exact for non-periodic groups, but only an
    approximation when applied to a periodic group whose window
    support is comparable to one period. When this function is
    called on a windowed periodic group, a
    :class:`WindowedSimilarityPeriodicApproxWarning` is emitted on
    every call. See User Guide §3.1 "Post-tensor windowing".

    Parameters
    ----------
    dens_query : MaetDensity, or list/tuple of MaetDensity
        The query density (not windowed). A single density gives the
        scalar behaviour; a list/tuple is broadcast or paired
        against the context (see Returns).
    dens_context : MaetDensity, or list/tuple of MaetDensity
        The context density to be windowed. As above, scalar or list.
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
    verbose : bool

    Returns
    -------
    np.ndarray
        - scalar query, scalar context → ``(M,)``.
        - scalar query, list of n_c contexts → ``(n_c, M)``.
        - list of n_q queries, scalar context → ``(n_q, M)``.
        - list-vs-list, ``mode='pairwise'`` (requires n_q == n_c) →
          ``(n_q, M)``.
        - list-vs-list, ``mode='cartesian'`` → ``(n_q, n_c, M)``.

        Length-1 lists do NOT collapse to scalars (Option II — strict
        shape preservation).
    """
    # ------------------------------------------------------------------
    # Direct kwarg forwarding (replaces the temp-defaults stop-gap that
    # was used historically). ``truncation_sigmas`` and ``kernel_precision``
    # flow through ``_windowed_similarity_core`` →
    # ``_windowed_similarity_pair`` → ``cos_sim_exp_tens``, where they
    # are consumed. ``None`` defers to the global default (resolved by
    # ``cos_sim_exp_tens`` itself).
    # ------------------------------------------------------------------
    return _windowed_similarity_core(
        dens_query, dens_context, window_spec, offsets,
        reference=reference, mode=mode,
        truncation_sigmas=truncation_sigmas,
        kernel_precision=kernel_precision,
        verbose=verbose,
    )


def _windowed_similarity_core(dens_query, dens_context, window_spec, offsets, *,
                              reference=None, mode: str = "auto",
                              truncation_sigmas: float | None = None,
                              kernel_precision: str | None = None,
                              verbose: bool = True):
    """Body of :func:`windowed_similarity`.

    ``truncation_sigmas`` and ``kernel_precision`` are forwarded through
    to each per-offset :func:`_windowed_inner_product` call. ``None``
    defers to the global default; explicit values flow directly without
    the temporary-defaults indirection used historically.
    """
    # Lazy import to avoid import cycles with the not-yet-migrated
    # parts of mpt.tensor (build / eval / cosine / dispatch).
    from ..tensor import _normalize_density_input, _resolve_list_list_mode

    # ------------------------------------------------------------------
    # Normalise query and context inputs.
    # ------------------------------------------------------------------
    q_scalar, q_list = _normalize_density_input(dens_query, name="dens_query")
    c_scalar, c_list = _normalize_density_input(dens_context, name="dens_context")

    # Validate every density is a plain MaetDensity (not Windowed, not SA).
    for label, scalar_flag, densities in (
        ("dens_query", q_scalar, q_list),
        ("dens_context", c_scalar, c_list),
    ):
        for i, d in enumerate(densities):
            if not isinstance(d, MaetDensity):
                idx = "" if scalar_flag else f"[{i}]"
                raise TypeError(
                    f"{label}{idx} must be a MaetDensity (not "
                    f"WindowedMaetDensity, not ExpTensDensity); got "
                    f"{type(d).__name__}."
                )

    n_q = len(q_list)
    n_c = len(c_list)

    # ------------------------------------------------------------------
    # Validate offsets shape and handle empty-list cases up front.
    # ------------------------------------------------------------------
    if n_c == 0 or n_q == 0:
        offsets_arr = np.asarray(offsets, dtype=np.float64)
        if offsets_arr.ndim == 1:
            offsets_arr = offsets_arr.reshape(-1, 1)
        M = offsets_arr.shape[1] if offsets_arr.ndim >= 2 else 0
        if q_scalar:
            return np.empty((0, M), dtype=np.float64)
        if c_scalar:
            return np.empty((0, M), dtype=np.float64)
        # both lists, at least one empty
        if mode == "auto":
            mode_resolved = "pairwise" if n_q == n_c else "cartesian"
        else:
            mode_resolved = mode
        if mode_resolved == "pairwise":
            return np.empty((0, M), dtype=np.float64)
        return np.empty((n_q, n_c, M), dtype=np.float64)

    # Resolve reference into a per-query list.
    references_per_query = _resolve_windowed_similarity_reference(
        reference, q_list,
    )

    # Common kwargs forwarded to every _windowed_similarity_pair call.
    _pair_kw = {
        "truncation_sigmas": truncation_sigmas,
        "kernel_precision": kernel_precision,
        "verbose": verbose,
    }

    # ------------------------------------------------------------------
    # Scalar-vs-scalar.
    # ------------------------------------------------------------------
    if q_scalar and c_scalar:
        return _windowed_similarity_pair(
            q_list[0], c_list[0], window_spec, offsets,
            ref_per_a=references_per_query[0], **_pair_kw,
        )

    if q_scalar:
        # 1 query × n_c contexts → (n_c, M).
        rows = [
            _windowed_similarity_pair(
                q_list[0], c, window_spec, offsets,
                ref_per_a=references_per_query[0], **_pair_kw,
            )
            for c in c_list
        ]
        return np.stack(rows, axis=0)

    if c_scalar:
        # n_q queries × 1 context → (n_q, M).
        rows = [
            _windowed_similarity_pair(
                q, c_list[0], window_spec, offsets,
                ref_per_a=ref, **_pair_kw,
            )
            for q, ref in zip(q_list, references_per_query)
        ]
        return np.stack(rows, axis=0)

    # Both lists.
    resolved = _resolve_list_list_mode(mode, n_q, n_c)
    if resolved == "pairwise":
        rows = [
            _windowed_similarity_pair(
                q, c, window_spec, offsets,
                ref_per_a=ref, **_pair_kw,
            )
            for q, c, ref in zip(q_list, c_list, references_per_query)
        ]
        return np.stack(rows, axis=0)

    # cartesian
    probe = _windowed_similarity_pair(
        q_list[0], c_list[0], window_spec, offsets,
        ref_per_a=references_per_query[0], **_pair_kw,
    )
    M = probe.shape[0]
    out = np.empty((n_q, n_c, M), dtype=np.float64)
    out[0, 0, :] = probe
    for j in range(1, n_c):
        out[0, j, :] = _windowed_similarity_pair(
            q_list[0], c_list[j], window_spec, offsets,
            ref_per_a=references_per_query[0], **_pair_kw,
        )
    for i in range(1, n_q):
        ref = references_per_query[i]
        for j in range(n_c):
            out[i, j, :] = _windowed_similarity_pair(
                q_list[i], c_list[j], window_spec, offsets,
                ref_per_a=ref, **_pair_kw,
            )
    return out


# -------------------------------------------------------------------
#  Cos-sim dispatch extension: unwindowed × windowed
# -------------------------------------------------------------------


def _windowed_inner_product(dens_a, dens_b, *, verbose: bool):
    """Cosine similarity with one or both operands windowed.

    Normaliser uses unwindowed L2 norms for both operands (Option Z).
    Numerator uses the windowed per-group pair factor V_g (replacing
    U_g) for windowed groups.

    Currently supports one-sided windowing (exactly one of dens_a,
    dens_b is a WindowedMaetDensity). Two-sided is not needed for the
    windowed_similarity use case.
    """
    a_win = isinstance(dens_a, WindowedMaetDensity)
    b_win = isinstance(dens_b, WindowedMaetDensity)
    if a_win and b_win:
        raise NotImplementedError(
            "Two-sided windowing (both operands windowed) is not supported "
            "in v2.1.0. Use windowed_similarity for profile sweeps."
        )

    if a_win:
        # Swap: put windowed operand on the 'b' side canonically.
        dens_q, wmd = dens_b, dens_a
    else:
        dens_q, wmd = dens_a, dens_b

    dens_c = wmd.dens

    # --- Structural compatibility checks (delegate to underlying _ma) ---
    _check_ma_compatibility(dens_q, dens_c)

    # --- Unwindowed norms (denominator) ---
    ip_qq = _cos_sim_numerator_ma(dens_q, dens_q, windowed_c=None)
    ip_cc = _cos_sim_numerator_ma(dens_c, dens_c, windowed_c=None)

    # --- Windowed numerator ---
    ip_qc = _cos_sim_numerator_ma(dens_q, dens_c, windowed_c=wmd)

    denom = np.sqrt(ip_qq * ip_cc)
    if denom == 0:
        return float("nan")
    return float(ip_qc / denom)


def _check_ma_compatibility(dens_x: MaetDensity, dens_y: MaetDensity):
    """Structural compatibility check, mirroring _cos_sim_exp_tens_ma."""
    if dens_x.n_attrs != dens_y.n_attrs:
        raise ValueError("Both densities must have the same n_attrs.")
    if not np.array_equal(dens_x.group_of_attr, dens_y.group_of_attr):
        raise ValueError("Both densities must have the same group_of_attr.")
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
    # Lazy import to avoid import cycles with the not-yet-migrated
    # parts of mpt.tensor (build / eval / cosine / dispatch).
    from ..tensor import _compute_Q

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
            g = int(dens_x.group_of_attr[a])
            # Only attributes whose group is actually windowed need
            # symmetrising.
            if not _is_windowed_group(windowed_c.size[g], windowed_c.mix[g]):
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
    group_of   = dens_x.group_of_attr
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
    # When windowed_c is not None, windowed_similarity asks for the cosine
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
    eff_shift_per_g = {}    # group index -> (d_g,) effective-space shift
    mu_q_per_g = {}         # group index -> (d_g,) effective-space query mean

    if windowed_c is not None:
        wmd = windowed_c
        attrs_of_g = dens_x.attrs_of_group
        dim_per = dens_x.dim_per_attr

        for g in range(dens_x.n_groups):
            if not _is_windowed_group(wmd.size[g], wmd.mix[g]):
                continue
            attrs_g = attrs_of_g[g]

            # mu_q_g: unweighted mean over perm-side tuple centres of
            # the query's effective-space Gaussian centres, concatenated
            # across attributes in g. The unweighted mean gives the
            # offset coordinate a weight-independent meaning (see User
            # Guide §3 on windowing). For relative groups this is
            # (approximately) zero by perm-symmetry, which is the
            # correct convention: translation-invariant groups have no
            # canonical position. Where the window centre in a relative
            # group is non-zero in effective space, the shift is still
            # applied and lifted via slot-0 anchoring.
            mu_q_parts = [dens_x.centres[a].mean(axis=1) for a in attrs_g]
            mu_q_g = np.concatenate(mu_q_parts)
            centre_g = np.concatenate([wmd.centre[a] for a in attrs_g])
            delta_g = centre_g - mu_q_g                          # (d_g,)

            mu_q_per_g[g] = mu_q_g
            eff_shift_per_g[g] = delta_g

            # Lift delta_g into per-attribute r_a-slot shifts.
            offset = 0
            g_is_rel = bool(is_rel_g[g])
            for a in attrs_g:
                a = int(a)
                r_a = int(r_vec[a])
                d_a = int(dim_per[a])
                delta_a_eff = delta_g[offset:offset + d_a]        # (d_a,)
                offset += d_a
                if g_is_rel:
                    # Slot-0 anchored lift: shift[0]=0, shift[1:]=delta_a_eff.
                    # For r_a == 1 and isRel=True the group is degenerate
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
        g = int(group_of[a])
        r_a = int(r_vec[a])
        U = dens_x.u_perm[a]   # (r_a, nJ_x)
        V = dens_y.v_comb[a]   # (r_a, nK_y)
        D = U[:, :, None] - V[:, None, :]  # (r_a, nJ_x, nK_y)

        # Apply per-attribute cross-correlation shift before wrap / Q_a.
        if a in shift_per_attr:
            D = D + shift_per_attr[a][:, None, None]

        if is_per_g[g]:
            p_g = float(period_g[g])
            D = np.mod(D + p_g / 2, p_g) - p_g / 2

        Q_a = _compute_Q(D, r_a, bool(is_rel_g[g]), bool(is_per_g[g]),
                         float(period_g[g]))
        log_kernel = log_kernel - Q_a / (4.0 * float(sigma_g[g]) ** 2)

    # ---- Windowed-factor contributions per group (cross-correlation
    # substitution applied). ----
    if windowed_c is not None:
        wmd = windowed_c
        attrs_of_g = dens_x.attrs_of_group
        eff_x_perm = _effective_centres_from_U(dens_x, side="perm")
        eff_y_comb = _effective_centres_from_V(dens_y, side="comb")

        for g in range(dens_x.n_groups):
            if not _is_windowed_group(wmd.size[g], wmd.mix[g]):
                continue
            attrs_g = attrs_of_g[g]

            cx_list = [eff_x_perm[a] for a in attrs_g]   # each (d_a, nJ_x)
            cy_list = [eff_y_comb[a] for a in attrs_g]   # each (d_a, nK_y)
            cx_g = np.concatenate(cx_list, axis=0)        # (d_g, nJ_x)
            cy_g = np.concatenate(cy_list, axis=0)        # (d_g, nK_y)
            centre_g = np.concatenate([wmd.centre[a] for a in attrs_g])
            mu_q_g = mu_q_per_g[g]                        # (d_g,)

            # Cross-correlation coordinate substitution: translate query
            # centres to origin via mu_q_g, translate context centres to
            # origin via centre_g, then apply the window at 0.
            cx_sub = cx_g - mu_q_g[:, None]
            cy_sub = cy_g - centre_g[:, None]
            centre_sub = np.zeros_like(centre_g)

            s_g = wmd.size[g]
            mix_g = wmd.mix[g]
            sigma_gv = float(sigma_g[g])
            is_rel = bool(is_rel_g[g])
            d_g = cx_sub.shape[0]

            contrib, log_D = _windowed_group_contribution(
                cx_sub, cy_sub, centre_sub,
                s_g, mix_g, sigma_gv, is_rel,
                int(r_vec[int(attrs_g[0])]), d_g,
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
    group_of = dens.group_of_attr
    is_rel_g = dens.is_rel
    r_vec = dens.r
    n_k = dens.n_k

    out = []
    for a in range(A):
        g = int(group_of[a])
        r_a = int(r_vec[a])
        V = dens.v_comb[a]                     # (r_a, nK)
        if not is_rel_g[g]:
            out.append(V)
        elif r_a >= 2:
            out.append(V[1:, :] - V[0:1, :])    # (r_a - 1, nK)
        else:
            out.append(np.empty((0, n_k), dtype=np.float64))
    return out


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
            f"{rho}) in v2.1.0. Use mix = 0 (pure Gaussian window), or "
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


