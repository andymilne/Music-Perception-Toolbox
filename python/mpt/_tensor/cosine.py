"""Cosine similarity: the cos_sim path.

Public entry points:

* :func:`cos_sim_exp_tens` --- compute cosine similarity between two
  expectation tensor densities (scalar, list-list, or batched-raw),
  with dispatch over Bulger / Möbius / centres methods.
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

import numpy as np

from .._defaults import _with_dispatch_scope
from .._kernel import gaussian_kernel_sum
from .._utils import (
    estimate_comp_time,
    kernel_chunk_bytes_resolved,
    maybe_print_batched_estimate,
    with_kernel_chunk_bytes_pin,
)
from ..spectra import add_spectra

from .build import (
    _canonicalise_nested_rel,
    _enum_flat_attr,
    _looks_like_multi_attr,
    _nested_enum_indices,
    build_exp_tens,
)
from .canonical import _pair_canonical_key
from ._mobius_inner import (
    _closed_form_attr_centres,
    _closed_form_attr_matrix_from,
    _ma_per_attr_inner_matrix,
    _ma_rel_attr_prefers_centres,
    _rel_inner_batched,
    _rel_window_margin,
    _trunc_kernel_exp,
)
from .density import (
    MaetDensity,
    is_single_multiset,
)
from .dispatch import (
    _compute_Q,
    _compute_Q_inner_blocks,
    _inner_r_vec,
    _normalize_density_input,
    _resolve_list_list_mode,
    _select_ma_inner_product_method,
)


# -------------------------------------------------------------------
#  Normalisation helpers for the ``normalize`` keyword of
#  cos_sim_exp_tens (and, through it, windowed_similarity).
# -------------------------------------------------------------------

#: The canonical value set for the ``normalize`` keyword. ``'cosine'`` is
#: the strict shape-only cosine similarity, with denominator equal to
#: the geometric mean of the two operands' self inner products.
#: ``'oneSidedDenom'`` divides only by the *second* operand's self inner
#: product, yielding a magnitude-aware reading that takes the value 1
#: on a perfect self-match at full coverage and may exceed 1 when the
#: first operand carries more matching mass than the second.
#: ``'none'`` returns the bare inner product :math:`\langle X, Y \rangle`
#: on the canonical scale (see :func:`_ip_canonical_scale`), whichever
#: route ran; no self inner product is formed. It is what the Rényi-2
#: entropy consumes, and what a caller who wants magnitudes rather than
#: a ratio asks for.
_NORMALIZE_VALUES = ("cosine", "oneSidedDenom", "none")


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
    if s.lower() == "none":
        return "none"
    raise ValueError(
        f"normalize must be one of {_NORMALIZE_VALUES!r}; got {normalize!r}."
    )


def _ip_canonical_scale(dens, chosen, nested_routes=None):
    """Factor that puts a route's bare inner product on the canonical scale.

    The canonical scale is the physical one: :math:`\langle X, Y \rangle =
    \int T_X T_Y` with each event's density the sum, over the attribute's
    full ordered tuple set (every arrangement a symmetric level admits),
    of unnormalised Gaussian kernels :math:`\exp(-Q(x - c)/2\sigma^2)`. It
    is the scale on which the Möbius per-attribute matrix and the
    closed-form total masses of :mod:`mpt._mobius` already agree, and so
    the one the Rényi-2 entropy has always been computed on.

    Every route drops constant per-attribute prefactors because they
    cancel in a ratio. Per attribute, with :math:`g_a = (\sigma_a
    \sqrt\pi)^{d_a}` for an absolute attribute of tuple dimension
    :math:`d_a` and :math:`g_a = \big[(\sigma_a \sqrt\pi)^{s_u - 1}
    \sqrt{s_u}\big]^{d_a / (s_u - 1)}` for a relative one whose
    co-transposition blocks have :math:`s_u` slots (the block metric's
    determinant), the factors are:

    * flat Möbius matrix: 1;
    * flat centres closed form, and the ordered-flat centres inside the
      nested plan: :math:`g_a`;
    * Bulger's enumeration: :math:`r_a!\, g_a` on a symmetric flat
      attribute, :math:`g_a` on an ordered one, :math:`|G_a|\, g_a` on a
      nested one (the wreath-product orbit order of
      :func:`~mpt._tensor._mobius_inner._nested_orbit_mult`);
    * nested contraction of an absolute attribute: :math:`|G_a|\, g_a`;
      the nested centres route: :math:`g_a`; the relative non-periodic
      contraction (a trapezoid over the alignment line of the product of
      the :math:`s` leaf kernels): :math:`|G_a|\, s\, (\sigma_a
      \sqrt\pi)^{s - 2} / 2`; the relative-periodic τ-grid (the mean over
      the period of the same integrand): :math:`P_a |G_a|\, s\,
      (\sigma_a \sqrt\pi)^{s - 2} / 2`.

    These are the constants :func:`_self_ip_cache_key` lists; they are
    pinned by ``tests/test_inner_product_scale.py``, which checks every
    route against an enumeration reference on every shape.
    """
    import math as _m
    from ._mobius_inner import _nested_orbit_mult
    A = int(dens.n_attrs)
    nested = getattr(dens, "nested", None) or [None] * A
    is_sym = np.asarray(getattr(dens, "is_sym", np.ones(A, dtype=bool))).ravel()
    scale = 1.0
    for a in range(A):
        sigma = float(dens.sigma[a])
        is_rel = bool(dens.is_rel[a])
        sp = sigma * _m.sqrt(_m.pi)
        spec = nested[a]
        if spec is None:
            r_a = int(dens.r[a])
            if is_rel and r_a < 2:
                continue       # 0-D point mass: no kernel, no prefactor
            g = (sp ** (r_a - 1) * _m.sqrt(r_a)) if is_rel else sp ** r_a
            ordered = (not bool(is_sym[a])) and r_a > 1
            if chosen == "mobius":
                f = 1.0
            elif chosen == "centres":
                f = g
            elif chosen == "bulger":
                f = g if ordered else _m.factorial(r_a) * g
            else:  # nested plan: flat attributes take the Möbius matrix,
                f = g if ordered else 1.0       # ordered ones the centres
            scale *= f
            continue
        r_levels = [int(v) for v in np.atleast_1d(spec["r"])]
        s_tot = int(np.prod(r_levels))
        rel_unit = spec.get("rel_unit")
        if rel_unit is None:
            g = sp ** s_tot
        else:
            s_u = int(np.prod(r_levels[:int(rel_unit) + 1]))
            g = (sp ** (s_u - 1) * _m.sqrt(s_u)) ** (s_tot // s_u)
        G = float(_nested_orbit_mult(r_levels, spec["sym"]))
        if chosen == "bulger":
            f = G * g
        elif chosen == "centres":
            f = g
        else:
            route = (nested_routes[a] if nested_routes is not None
                     and a < len(nested_routes) else "contract")
            if route == "centres":
                f = g
            elif route == "contract_relnonper":
                # Trapezoid over the alignment line: an integral over the
                # translation of the s leaf kernels, in the ones direction.
                f = G * s_tot * sp ** (s_tot - 2) / 2.0
            elif route == "taugrid":
                # Mean over the period of the same integrand.
                f = (float(dens.period[a]) * G * s_tot
                     * sp ** (s_tot - 2) / 2.0)
            else:
                f = G * g
        scale *= f
    return scale


def _finalise_normalisation(
    ip_xy: float, ip_xx: float, ip_yy: float, normalize: str,
) -> float:
    """Combine numerator and self inner products into the final value.

    ``ip_xy`` is :math:`\\langle X, Y \\rangle`; ``ip_xx`` and ``ip_yy``
    are the two operands' self inner products. With ``normalize`` set
    to ``'cosine'`` the denominator is :math:`\\sqrt{ip_{xx} \\cdot ip_{yy}}`;
    with ``'oneSidedDenom'`` the denominator is :math:`ip_{yy}` alone.
    Either denominator equal to zero returns ``NaN``.

    ``ip_xx`` may be ``None`` under ``'oneSidedDenom'``, whose
    denominator does not consume it (the triple routines skip its
    computation in that case); passing ``None`` under ``'cosine'`` is a
    caller defect and raises. Under ``'none'`` the bare ``ip_xy`` is
    returned; the caller has already put it on the canonical scale.
    """
    if normalize == "none":
        return float(ip_xy)
    if normalize == "cosine":
        if ip_xx is None:
            raise ValueError(
                "normalize='cosine' requires <X,X>, but it was not "
                "computed. This is an internal routing defect."
            )
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
                     truncation_sigmas: float | None = None,
                     kernel_precision: str | None = None,
                     verbose: bool = True) -> float | np.ndarray:
    """Cosine similarity of two expectation tensor densities.

    Input forms, in the order to reach for them: a single multiset; a
    pre-MAET, the canonical entry for everything else; densities built
    by :func:`build_exp_tens`; then the raw positional multi-attribute,
    sweep, and batched forms.

    Unified entry point. Accepts four input forms, dispatched on the
    type of the first argument:

    **Raw single-multiset scalar input**:

    - ``cos_sim_exp_tens(p1, w1, p2, w2, sigma, r, is_rel, is_per, period)``
      where ``p1`` and ``p2`` are 1-D arrays of pitches, ``w1``,
      ``w2`` are matching 1-D weight arrays (or ``None`` for uniform).
      Returns scalar.

    **Pre-MAET input**:

    - ``cos_sim_exp_tens(pm1, pm2)``. A pre-MAET
      (:func:`~mpt.pre_maet`) holds everything :func:`build_exp_tens`
      needs, so it stands wherever a density does: each side is built
      internally and a scalar returned. Either side may equally be a
      density, so the two forms mix freely.

    **Pre-built density input** (plus polymorphic lists):

    - ``cos_sim_exp_tens(dens_x, dens_y)`` — scalar.
    - ``cos_sim_exp_tens(dens_x, [d1, d2, …])`` — broadcast, returns
      ``(N,)``.
    - ``cos_sim_exp_tens([a1, a2, …], [b1, b2, …])`` — list-vs-list
      with ``mode='pairwise'`` (default ``'auto'``, resolves to
      pairwise for equal lengths) returning ``(M,)``, or
      ``mode='cartesian'`` returning ``(M, N)``.

    **Raw multi-attribute scalar input**:

    - ``cos_sim_exp_tens(p_attr1, w_attr1, p_attr2, w_attr2, sigma_vec,
      r_vec, is_rel_vec, is_per_vec, period_vec)`` where ``p_attr*`` are
      lists of per-attribute matrices and ``w_attr*`` the matching
      per-attribute weights. Returns scalar.

    **Raw multi-attribute scalar-vs-list (sweep)**:

    - ``cos_sim_exp_tens(p_attr_ref, w_attr_ref, p_attr_list, w_attr_shared,
      sigma_vec, r_vec, is_rel_vec, is_per_vec, period_vec)``
      where exactly one of the two ``p_attr`` arguments is a list of
      ``p_attr`` blocks (a list of lists; e.g. the matrix-form output of
      :func:`translate_attributes`) and the other is a single ``p_attr``.
      Build is internalised: the scalar operand is built once, the
      list operand once per entry. Weights for the list side are
      shared across every entry — a single ``w_attr`` value, not a list
      of weights. Returns an ``ndarray`` of length M. The output index
      matches the order of entries in the list operand. When every
      ``r_a = 1`` and ``method`` is ``'auto'`` or ``'bulger'`` the whole
      list is evaluated in one batched kernel pass (the same fast path
      the density scalar-vs-list form and MATLAB's
      ``localR1BroadcastFast`` take). *Python only:* a list tagged by
      :func:`translate_attributes` with ``method='auto'`` is first
      reduced to one mixture in the offset through
      :func:`sweep_cos_sim_exp_tens`; MATLAB has no tagged-sweep type.

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
        bypass dedup transparently.
    spectrum : list/tuple, optional
        Per-row spectral augmentation parameters passed to
        :func:`mpt.spectra.add_spectra`. Only valid in raw single-multiset modes
        (scalar or batched).
    precision : int, optional
        Round canonical pitch and weight values to this many decimal
        places, to absorb FP noise when deduplicating. Only valid in
        raw single-multiset batched mode.
    method : {'auto', 'bulger', 'centres', 'mobius', 'contract'}, default 'auto'
        Inner-product method; threaded through to the per-pair single-multiset/multi-attribute
        core. ``'auto'`` lets the dispatcher pick between Bulger's
        method (the partition-pair decomposition; small r and small K)
        and the Möbius method (large r or large K).
        ``'bulger'`` forces Bulger's method; ``'mobius'`` forces the
        Möbius method; ``'centres'`` forces unrestricted enumeration of
        the tuple centres, the O(K^(2r)) route that reads the definition
        directly. ``'centres'`` is far slower than either at any
        appreciable K and is intended as a reference: it involves no
        alternating sum, so it is immune to the cancellation the Möbius
        route can suffer, and it shares no reduction with the other two.

        On a **nested** density the names select among that path's own
        routes, since the flat orbit entry point cannot represent a nested
        attribute's block-diagonal inner metric. ``'bulger'`` is still the
        joint-tuple enumeration. ``'contract'`` forces the hierarchical
        contraction plan, raising rather than falling back on any case it
        does not cover; it is rejected on a non-nested density.
        ``'mobius'`` names the same plan --- the per-level orbit (Möbius)
        reduction is exactly what the contraction applies at every symmetric
        level, so for a nested density ``'mobius'`` and ``'contract'``
        coincide. ``'centres'`` forces the materialised-centres route for
        every nested attribute; on a relative-periodic attribute whose
        declared measure is the default full-image one, that route is
        admissible only up to the sigma/period threshold, above which
        ``'centres'`` raises a ``ValueError`` naming the
        ``wrap='single-image'`` opt-in.
    normalize : {'cosine', 'oneSidedDenom', 'none'}, default 'cosine'
        Selects the denominator applied to the inner product
        :math:`\\langle X, Y \\rangle`. ``'cosine'`` (default) gives the
        strict shape-only cosine similarity, dividing by the geometric
        mean :math:`\\sqrt{\\langle X, X \\rangle \\, \\langle Y, Y \\rangle}`;
        the result is bounded in :math:`[-1, 1]` and is invariant to a
        positive scalar on either operand. ``'oneSidedDenom'`` divides
        by the second operand's self inner product
        :math:`\\langle Y, Y \\rangle` alone, yielding a magnitude-aware
        reading that takes the value 1 on a self-match (``X == Y``)
        and is sensitive to scalar reweightings of ``X``. ``'none'``
        returns the bare inner product :math:`\\langle X, Y \\rangle`
        itself, on one canonical scale whichever route computed it (the
        physical integral of the two densities' product, each event's
        density being the sum of unnormalised Gaussian kernels over its
        full ordered tuple set); no self inner product is formed, and
        the value is what ``entropy_exp_tens(method='renyi2')`` is
        computed from. Batched fast paths decline it and the per-pair
        route runs instead. The British spelling ``'normalise'`` is also
        accepted as an alias for the keyword name, and matching is
        case-insensitive on the value.
    truncation_sigmas : float, optional
        Per-call kernel truncation width in sigmas (``None`` takes the
        global default; ``inf`` resolves to the accuracy-floor width).
        Honoured on every route, flat and nested, and forwarded through
        every input form; it also sizes the quadrature grids the routes
        are priced on.
    kernel_precision : {'double', 'single'}, optional
        Forwarded through every input form, but consumed by one
        inner-product leaf only: the single-attribute kernel-sum helper
        (``A == 1``, not relative-periodic), which evaluates in float32
        under ``'single'``. The multi-attribute log-kernel core and the
        Möbius and nested routes have no float32 form and do not read
        it. The MATLAB twin takes the same helper leaf
        (``internal.gaussianKernelSum``) and honours it there too; both
        point evaluators honour it.
    verbose : bool, default True
        Print progress.

    Returns
    -------
    float or np.ndarray
        Scalar in scalar-vs-scalar density mode, raw single-multiset scalar mode, and
        raw MA scalar mode. ``ndarray`` in all batched/list modes.

    Notes
    -----
    Accuracy is governed by ``truncationSigmas``: the Möbius method's
    agreement with enumeration tracks the truncation budget, and how
    close each ``K_a`` is to its ``r_a`` does not bear on it. The
    dispatcher routes to Bulger's method when ``σ/P > 0.03`` in
    periodic-relative mode, or when the σ → 0 fallback triggers;
    otherwise it chooses on cost. Pass ``method='bulger'`` to bypass
    the Möbius method entirely.

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
    args = _build_pre_maet_args(args, verbose=verbose)
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
    is_density_scalar = isinstance(a, MaetDensity)
    intends_density_list = False
    if isinstance(a, (list, tuple)):
        if len(a) == 0:
            intends_density_list = True
        elif isinstance(a[0], MaetDensity):
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
        truncation_sigmas=truncation_sigmas,
        kernel_precision=kernel_precision,
        verbose=verbose,
    )



def _r1_broadcast_fast(pairs, *, shared_is_x, normalize,
                       truncation_sigmas=None, kernel_precision=None):
    """Batched broadcast for the all-r = 1 shape, or ``None``.

    Applies when one operand is shared across every pair and all
    densities are flat ``MaetDensity`` structures of identical
    geometry with every ``r_a = 1`` (no nesting, no ordered-at-r>1
    concern — r = 1 symmetrisation is vacuous — no kernel covariance
    mismatch). The cross inner products of the whole sweep are then
    one kernel evaluation over the concatenated query columns, and
    the self terms go through the same memo keys the per-pair route
    uses, so a subsequent per-pair call sees the same cache state.

    Per-pair equality: each attribute's log-kernel column block is
    computed by the same expressions in the same order as
    :func:`_ip_r1_direct`, and the truncation threshold is applied
    per column segment with that pair's own ``n_terms``, so the set
    of zeroed entries matches the per-pair path exactly. The only
    floating-point difference is the contraction association
    (``(w_x @ E) . w_y`` here versus ``w_x @ (E @ w_y)`` per pair),
    within the toolbox-wide ≤ 1e-12 parity discipline.

    Returns the per-pair similarity array, or ``None`` when any
    condition fails — the caller's ordinary loops then run and raise
    the errors a genuine mismatch deserves.
    """
    if normalize == "none":
        return None            # the per-pair route puts the bare value on scale
    from .._defaults import resolve_truncation_sigmas
    from .aniso import density_has_kernel_cov

    if len(pairs) == 0:
        return None
    shared = pairs[0][0] if shared_is_x else pairs[0][1]
    entries = [(p[1] if shared_is_x else p[0]) for p in pairs]

    dens_all = [shared] + entries
    for d in dens_all:
        if not isinstance(d, MaetDensity):
            return None
        if density_has_kernel_cov(d):
            return None
        nested = getattr(d, "nested", None)
        if nested is not None and any(s is not None for s in nested):
            return None

    shared_p = shared.pruned()
    A = shared_p.n_attrs
    if A == 0 or not all(int(r) == 1 for r in shared_p.r):
        return None
    if any(int(x) != 0 for x in _inner_r_vec(shared_p)):
        return None

    entries_p = [d.pruned() for d in entries]
    for d in entries_p:
        if (d.n_attrs != A
                or not np.array_equal(d.r, shared_p.r)
                or not np.array_equal(d.sigma, shared_p.sigma)
                or not np.array_equal(d.is_rel, shared_p.is_rel)
                or not np.array_equal(d.is_per, shared_p.is_per)):
            return None
        per_mask = shared_p.is_per.astype(bool)
        if np.any(d.period[per_mask] != shared_p.period[per_mask]):
            return None
        wrap_d = list(getattr(d, "wrap", ["full-image"] * A))
        wrap_s = list(getattr(shared_p, "wrap", ["full-image"] * A))
        if [str(v) for v in wrap_d] != [str(v) for v in wrap_s]:
            return None

    ts = resolve_truncation_sigmas(truncation_sigmas)
    sigma = shared_p.sigma
    is_rel = shared_p.is_rel
    is_per = shared_p.is_per
    period = shared_p.period
    wrap = list(getattr(shared_p, "wrap", ["full-image"] * A))
    need_xx_shared = (normalize == "cosine") if shared_is_x else True
    key = _self_ip_cache_key("bulger", truncation_sigmas, kernel_precision)

    if shared_p.n == 0:
        return np.zeros(len(entries), dtype=np.float64)

    n_j = shared_p.n_j
    u_cell = shared_p.u_perm
    w_u = shared_p.w_j

    # Concatenate the entries' comb-side columns; record segment
    # boundaries and per-segment n_terms for the truncation threshold.
    seg_len = np.array([d.n_k if d.n != 0 else 0 for d in entries_p],
                       dtype=np.intp)
    T = int(seg_len.sum())
    starts = np.concatenate([[0], np.cumsum(seg_len)])
    live = [d for d in entries_p if d.n != 0]
    if T > 0:
        v_cell = [np.concatenate([d.v_comb[a] for d in live], axis=1)
                  for a in range(A)]
        # Per-column log-space threshold: -k^2/2 - log(n_j * m_i) for
        # the segment the column belongs to (matching each pair's own
        # _trunc_log_kernel_exp threshold; n_terms = 1 keeps the bare
        # floor, as there).
        thr_col = np.empty(T, dtype=np.float64)
        for i, d in enumerate(entries_p):
            if seg_len[i] == 0:
                continue
            n_terms = int(n_j) * int(d.n_k)
            t = -0.5 * ts ** 2
            if n_terms > 1:
                t -= math.log(float(n_terms))
            thr_col[starts[i]:starts[i + 1]] = t

        # One kernel pass over the concatenated columns. Chunk width is
        # the smaller of the memory-limit heuristic and a cache-resident
        # cap: the per-pair path's skinny blocks stay in cache, and a
        # single (n_j, T) block at large n_j leaves it, dropping kernel
        # throughput by ~2-3x. A few MB per block keeps the batched
        # path's locality while amortising the per-chunk Python cost.
        bytes_per_col = 3 * int(n_j) * 8
        mem_limit = kernel_chunk_bytes_resolved()
        cache_cap = max(1, int(4_000_000 // max(int(n_j) * 8, 1)))
        chunk = max(1, min(int(mem_limit // max(bytes_per_col, 1)),
                           cache_cap))
        row = np.empty(T, dtype=np.float64)   # (w_u @ E) per column
        for c0 in range(0, T, chunk):
            c1 = min(c0 + chunk, T)
            L = np.zeros((int(n_j), c1 - c0), dtype=np.float64)
            for a in range(A):
                if bool(is_rel[a]):
                    continue
                d = (u_cell[a][0][:, None] - v_cell[a][0][None, c0:c1])
                wrap_a = str(wrap[a]) if a < len(wrap) else 'full-image'
                if bool(is_per[a]) and wrap_a == 'full-image':
                    from .._wrapped_kernel import wrapped_gaussian_1d
                    theta = wrapped_gaussian_1d(
                        d, float(sigma[a]), float(period[a]), ts,
                        exponent_denominator=4,
                    )
                    L += np.log(theta)
                    continue
                if bool(is_per[a]):
                    p_a = float(period[a])
                    d = d - p_a * np.floor(d / p_a + 0.5)
                L -= (d * d) / (4 * float(sigma[a]) ** 2)
            below = L < thr_col[None, c0:c1]
            np.exp(L, out=L)
            L[below] = 0.0
            row[c0:c1] = w_u @ L
        ip_xy = np.array([
            float(row[starts[i]:starts[i + 1]]
                  @ entries_p[i].wv_comb) if seg_len[i] else 0.0
            for i in range(len(entries_p))
        ])
    else:
        ip_xy = np.zeros(len(entries_p), dtype=np.float64)

    # Shared operand's self term, through the same memo key the
    # per-pair route uses. As X under 'oneSidedDenom' it is not
    # consumed and not computed.
    ip_shared = None
    if key in shared._self_ip_cache:
        ip_shared = shared._self_ip_cache[key]
    elif need_xx_shared:
        ip_shared = _ip_core_ma(
            shared_p.u_perm, shared_p.w_j, shared_p.n_j,
            shared_p.v_comb, shared_p.wv_comb, shared_p.n_k,
            A, shared_p.r, sigma, is_rel, is_per, period,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
            inner_r=_inner_r_vec(shared_p), wrap=wrap,
        )
        shared._self_ip_cache[key] = ip_shared
        shared_p._self_ip_cache[key] = ip_shared

    # Entries' self terms: as Y (shared_is_x) always consumed; as X,
    # consumed only under 'cosine'.
    need_self_entry = shared_is_x or (normalize == "cosine")
    out = np.empty(len(entries_p), dtype=np.float64)
    for i, (d_orig, d) in enumerate(zip(entries, entries_p)):
        if d.n == 0 or shared_p.n == 0:
            out[i] = 0.0
            continue
        if key in d._self_ip_cache:
            ip_self = d._self_ip_cache[key]
        elif need_self_entry:
            ip_self = _ip_core_ma(
                d.u_perm, d.w_j, d.n_j, d.v_comb, d.wv_comb, d.n_k,
                A, d.r, sigma, is_rel, is_per, period,
                truncation_sigmas=truncation_sigmas,
                kernel_precision=kernel_precision,
                inner_r=_inner_r_vec(d), wrap=wrap,
            )
            d._self_ip_cache[key] = ip_self
            d_orig._self_ip_cache[key] = ip_self
        else:
            ip_self = None
        if shared_is_x:
            ip_xx, ip_yy = ip_shared, ip_self
        else:
            ip_xx, ip_yy = ip_self, ip_shared
        out[i] = _finalise_normalisation(
            float(ip_xy[i]), ip_xx, ip_yy, normalize)
    return out


def _all_single_multiset_pairs(pairs):
    """Return True iff every (a, b) pair consists of two single-multiset
    densities (A = 1, flat), for which canonical-form dedup applies."""
    for a, b in pairs:
        if not (is_single_multiset(a) and is_single_multiset(b)):
            return False
    return True



def _compute_pair_results_with_dedup(
    pairs, *, method: str, normalize: str = "cosine",
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

    def _declared(d):
        # The per-density declarations the pair core reads beyond the
        # chord and its parameters: the wrap names the measure on a
        # periodic attribute and [sym] the tuple reading, so two pairs
        # may share a key only when both agree (the MATLAB twin
        # localDensityPairKey bakes in the same two).
        wrap = getattr(d, "wrap", None)
        sym = getattr(d, "is_sym", None)
        return (str(wrap[0]) if wrap is not None else "full-image",
                bool(sym[0]) if sym is not None else True)

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
            _declared(a), _declared(b),
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
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
            verbose=False,
        ))
    return results



def _cos_sim_pair_core(
    dens_x, dens_y, *,
    method: str = "auto",
    normalize: str = "cosine",
    truncation_sigmas: float | None = None,
    kernel_precision: str | None = None,
    verbose: bool,
):
    """Internal: dispatch a single pair to the correct core IP routine.

    Routes to :func:`_cos_sim_exp_tens_ma`, threading ``method``,
    ``normalize``, ``truncation_sigmas`` and ``kernel_precision``
    through; that routine
    handles both the single-multiset and multi-attribute cases.
    """
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
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
            verbose=verbose,
        )
    raise TypeError(
        f"Both arguments must be MaetDensity; got "
        f"{type(dens_x).__name__} and {type(dens_y).__name__}."
    )



def _cos_sim_density_path(
    dens_x, dens_y, *,
    mode: str = "auto",
    dedup: bool = True,
    method: str = "auto",
    normalize: str = "cosine",
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

    # Batched all-r = 1 broadcast: one shared operand against many
    # queries of identical geometry evaluates every cross term in a
    # single kernel pass (per-segment truncation thresholds keep each
    # pair's threshold equal to its per-pair value), amortising the
    # per-pair dispatch that otherwise dominates point-set-shaped
    # sweeps. Returns None whenever any structural condition fails, and
    # the ordinary per-pair loops below then run — raising exactly the
    # errors a genuine mismatch deserves.
    if (is_x_scalar != is_y_scalar) and method in ("auto", "bulger"):
        fast = _r1_broadcast_fast(
            pairs,
            shared_is_x=is_x_scalar,
            normalize=normalize,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
        )
        if fast is not None:
            return np.asarray(fast, dtype=np.float64).reshape(out_shape)

    if dedup and _all_single_multiset_pairs(pairs):
        results = _compute_pair_results_with_dedup(
            pairs,
            method=method,
            normalize=normalize,
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
    truncation_sigmas: float | None = None,
    kernel_precision: str | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """Raw multi-attribute scalar-vs-list broadcast.

    Exactly one of the two operands is a list of per-attribute ``p_attr``
    blocks (cell-of-cells). The scalar operand is built once and reused
    against every list entry. Weights for the list operand are shared
    across all entries (one ``w`` value, not a list of weights).

    Route order on this form:

    1. **Python only.** A list tagged by
       :func:`~mpt.translate_attributes` (a
       :class:`~mpt._tensor.preprocessing.TranslatedSweep`) is reduced to
       one mixture in the offset through :func:`sweep_cos_sim_exp_tens`
       when ``method='auto'`` and the sweep is eligible. The MATLAB
       toolbox has no tagged-sweep type, so this reduction has no twin
       there; an untagged list never reaches it.
    2. The all-``r = 1`` broadcast fast path (:func:`_r1_broadcast_fast`),
       when ``method in ('auto', 'bulger')`` and every density is a flat
       ``MaetDensity`` of identical geometry with every ``r_a = 1``. This
       is the twin of MATLAB's ``localR1BroadcastFast`` on the same form,
       gated identically, so an untagged raw-MA list takes the same route
       in both languages.
    3. Otherwise the per-entry loop through :func:`_cos_sim_pair_core`
       with the caller's ``method`` and widths.

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

    # A tagged sweep from translate_attributes carries the offsets that
    # produced it, so the whole list can be evaluated as one mixture in
    # the offset instead of one inner product per entry. The reduction
    # declines on any shape it does not cover, and the per-entry loop
    # below then runs unchanged.
    fast = _try_sweep_reduction(
        list_pAttr, list_w, dens_scalar,
        sigma_vec, r_vec, is_rel_vec, is_per_vec, period_vec, is_sym_vec,
        scalar_first=scalar_first,
        normalize=normalize,
        method=method,
        truncation_sigmas=truncation_sigmas,
        kernel_precision=kernel_precision,
        verbose=verbose,
    )
    if fast is not None:
        return fast

    M = len(list_pAttr)
    dens_list = [
        build_exp_tens(
            list_pAttr[m], list_w, sigma_vec, r_vec,
            is_rel_vec, is_per_vec, period_vec, is_sym_vec, verbose=False,
        )
        for m in range(M)
    ]

    # Batched all-r = 1 broadcast (twin of MATLAB's localR1BroadcastFast
    # on this form): one shared operand against many queries of
    # identical geometry evaluates every cross term in a single kernel
    # pass, with per-segment truncation thresholds so each pair's
    # threshold equals its per-pair value. Gated on the same methods as
    # the density-list form; a forced 'mobius'/'centres'/'contract'
    # names a route through the per-pair core and is honoured there.
    # Returns None whenever any structural condition fails, and the
    # per-entry loop below then runs unchanged.
    if M > 0 and method in ("auto", "bulger"):
        pairs = ([(dens_scalar, d) for d in dens_list] if scalar_first
                 else [(d, dens_scalar) for d in dens_list])
        fast = _r1_broadcast_fast(
            pairs,
            shared_is_x=scalar_first,
            normalize=normalize,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
        )
        if fast is not None:
            return np.asarray(fast, dtype=np.float64).reshape(M)

    out = np.empty(M, dtype=np.float64)
    for m in range(M):
        dens_m = dens_list[m]
        if scalar_first:
            out[m] = _cos_sim_pair_core(
                dens_scalar, dens_m,
                method=method,
                normalize=normalize,
                truncation_sigmas=truncation_sigmas,
                kernel_precision=kernel_precision,
                verbose=False,
            )
        else:
            out[m] = _cos_sim_pair_core(
                dens_m, dens_scalar,
                method=method,
                normalize=normalize,
                truncation_sigmas=truncation_sigmas,
                kernel_precision=kernel_precision,
                verbose=False,
            )
    return out



def _try_sweep_reduction(
    list_pAttr, list_w, dens_scalar,
    sigma_vec, r_vec, is_rel_vec, is_per_vec, period_vec, is_sym_vec,
    *, scalar_first, normalize, method,
    truncation_sigmas, kernel_precision, verbose,
):
    """Evaluate a tagged translation sweep as a mixture in the offset.

    Returns the length-M result array, or ``None`` when the reduction
    does not apply --- an untagged list, a non-uniform or unrecoverable
    offset, a mode the reduction refuses, or an explicitly forced
    ``method``. Returning ``None`` leaves the caller's per-entry loop to
    run, so every input still reaches a correct answer by some route.
    """
    from .preprocessing import TranslatedSweep
    from .sweep import sweep_cos_sim_exp_tens, sweep_eligibility

    if not isinstance(list_pAttr, TranslatedSweep):
        return None
    # A forced method names a route through the per-pair core; honour it
    # rather than substituting a different computation.
    if method not in ("auto",):
        return None
    off = np.asarray(list_pAttr.sweep_offsets, dtype=np.float64)
    if off.size == 0 or not np.all(np.isfinite(off)):
        return None

    dens_base = build_exp_tens(
        list_pAttr.sweep_base, list_w, sigma_vec, r_vec,
        is_rel_vec, is_per_vec, period_vec, is_sym_vec, verbose=False,
    )
    # The sweep translates the query; when the tagged list is the first
    # operand the roles reverse, and translating X by mu is translating
    # Y by -mu with the operands exchanged (the inner product is
    # symmetric, and 'oneSidedDenom' divides by the second operand,
    # which is the tagged one either way).
    if normalize == "none":
        return None
    if scalar_first:
        dens_x, dens_y, off_use = dens_scalar, dens_base, off
    else:
        if normalize != "cosine":
            return None
        dens_x, dens_y, off_use = dens_base, dens_scalar, -off

    ok, _ = sweep_eligibility(dens_x, dens_y, off_use)
    if not ok:
        return None
    return sweep_cos_sim_exp_tens(
        dens_x, dens_y, off_use,
        normalize=normalize,
        truncation_sigmas=truncation_sigmas,
        kernel_precision=kernel_precision,
        verbose=verbose,
    )



# -------------------------------------------------------------------
#  _cos_sim_exp_tens_ma  (multi-attribute inner product; orbit constants)
# -------------------------------------------------------------------


"""Maximum kernel entries per translation-grid slab in the rel-mode
orbit inner product (~4 MB of doubles). The contraction makes several
permute and power copies of each slab, so the live working set is a
small multiple of this; the value keeps it memory-resident on typical
hardware, where the contraction's per-op cost is flat in K. Slab count
grows only the loop overhead, which is negligible against the per-slab
contraction work."""





def _flat_selector_inputs(dens_x, dens_y, *, normalize, truncation_sigmas):
    """The flat selector's inputs for a cosine between two pruned densities.

    Returns ``(kwargs, ordered_any, nested_any)``: the keyword arguments
    :func:`~mpt._tensor.dispatch._select_ma_inner_product_method` is
    called with (everything but ``user_method``), whether any attribute
    on either side is ordered at ``r > 1`` --- which overrides the
    selector's choice with Bulger's method --- and whether either density
    is nested. One function builds these for the real call and for
    :func:`~mpt._tensor.explain.explain_dispatch`, so the report cannot
    drift from the route the call takes (the report used to omit the
    wrap vector, the grid node counts, and the memo flags, and so named
    the wrong route wherever those decided).
    """
    A = dens_x.n_attrs
    r_vec = dens_x.r
    is_rel = dens_x.is_rel
    is_per = dens_x.is_per
    sigma = dens_x.sigma
    period = dens_x.period

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
    # the Möbius per-attribute matrix carries as zero-weight padding).
    # One vector per density: the two need not carry the same number of
    # values in an attribute, and a chord against a scale, or a reference
    # tuning against an equal division, is the ordinary case.
    k_vec = np.array(
        [int(M.shape[0]) for M in dens_x.p_attr], dtype=np.intp,
    ) if A > 0 else np.zeros(0, dtype=np.intp)
    k_vec_y = np.array(
        [int(M.shape[0]) for M in dens_y.p_attr], dtype=np.intp,
    ) if A > 0 else np.zeros(0, dtype=np.intp)

    # Per-attribute vectors for the Möbius-side cost model: which
    # attributes are relative, and each one's translation-grid node
    # estimate (matching the grid rules of the batched rel helper; 1
    # for absolute attributes, where no grid exists).
    from ._nested_contraction import auto_ntau_default
    from .._defaults import resolve_truncation_sigmas as _resolve_ts
    # The per-call truncation width sizes the grids the routes are priced
    # on, as it sizes the kernels they run: pricing at the global default
    # while truncating at the per-call width would race the routes on a
    # grid neither of them uses (MATLAB: cosSimExpTens nuVecSel).
    _ts_sel = _resolve_ts(truncation_sigmas)
    rel_vec = np.array([bool(is_rel[a]) for a in range(A)], dtype=bool)
    nu_vec = np.ones(max(A, 1))[:A]
    for a in range(A):
        if not rel_vec[a] or int(r_vec[a]) < 2:
            continue
        if bool(is_per[a]):
            nu_vec[a] = auto_ntau_default(
                float(period[a]), float(sigma[a]), _ts_sel)
        else:
            Pxa = dens_x.p_attr[a]
            Pya = dens_y.p_attr[a]
            _margin = _rel_window_margin(_ts_sel)
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

    # Rel-per wrap axis (v3+): the two densities must agree, because the
    # wrap declares the measure and a cosine between two measures is not
    # a cosine. A disagreement is an error rather than a silent reading
    # of dens_x's declaration (MATLAB: mpt:wrapMismatch).
    wrap_vec_x = [_declared_wrap(dens_x, dens_y, a) for a in range(A)]

    # A self inner product costs nothing at call time when it is
    # memoised on its density, or (for <X,X>) when the requested
    # normalisation does not consume it; tell the selector so its
    # pricing reflects the work this call will actually perform. The
    # flags are *shared* by the two routes' prices --- see
    # :func:`_self_ip_memoised` for why a per-route flag makes the
    # comparison unfair, and :func:`_self_ip_cache_key` for why the
    # memoised values themselves stay per route.
    need_xx = (normalize == "cosine")
    need_yy = (normalize != "none")
    need_yy = (normalize != "none")
    skip_xx = (not need_xx) or _self_ip_memoised(dens_x)
    skip_yy = (not need_yy) or _self_ip_memoised(dens_y)

    # Ordered ([sym]=0) attributes at r_a > 1 on either side.
    is_sym_x = np.asarray(getattr(dens_x, "is_sym", np.ones(A, dtype=bool)))
    is_sym_y = np.asarray(getattr(dens_y, "is_sym", np.ones(A, dtype=bool)))
    ordered_any = bool(
        np.any((~is_sym_x) & (r_vec > 1))
        or np.any((~is_sym_y) & (r_vec > 1))
    )

    kwargs = dict(
        r_vec=r_vec, k_vec=k_vec, k_vec_y=k_vec_y, A=A,
        N_x=int(dens_x.n), N_y=int(dens_y.n),
        any_per=any_per,
        any_rel_nonper=any_rel_nonper,
        any_rel_per=any_rel_per,
        sigma_over_P_max=sop_max,
        rel_vec=rel_vec, nu_vec=nu_vec,
        guard_forced_bulger=not nested_any,
        wrap_vec=wrap_vec_x,
        per_vec=[bool(is_per[a]) for a in range(A)],
        sym_vec=getattr(dens_x, "is_sym", None),
        truncation_sigmas=truncation_sigmas,
        skip_xx=skip_xx, skip_yy=skip_yy,
    )
    return kwargs, ordered_any, nested_any


def _cos_sim_exp_tens_ma(
    dens_x: MaetDensity,
    dens_y: MaetDensity,
    *,
    method: str = "auto",
    normalize: str = "cosine",
    truncation_sigmas=None,
    kernel_precision=None,
    verbose: bool = True,
) -> float:
    """Multi-attribute cosine similarity.

    A ``method`` keyword routes between Bulger's method
    — the v1 / v2.0 decomposition with periodic pairwise-wrap form
    (``_ip_core_ma``) — and the Möbius method. With the default
    ``method='auto'`` the flat selector
    (:func:`~mpt._tensor.dispatch._select_ma_inner_product_method`)
    decides: structural rules first (tuple order, feasibility, the
    working-set guard, the rel-per wrap rule above
    :func:`_orbit_sigma_over_p_threshold`), then the cost race. Where
    both methods are admissible they agree to within the truncation
    floor.

    Both densities must share the full parameter structure: number of
    attributes, group assignment, per-attribute ``r``, and per-group
    ``sigma``/``is_rel``/``is_per``/``period``. Weights and event/value
    counts may differ freely — that's the whole point of the similarity
    measure.
    """
    dens_x = dens_x.pruned()
    dens_y = dens_y.pruned()

    # An empty operand has no events to overlap, so the inner product -- and
    # hence the similarity -- is zero. An event-weighted density whose window
    # caught nothing prunes to zero events here; without this guard it reaches
    # the nested contraction's value-range scan, which has no identity over an
    # empty attribute column. (The raw single-multiset path is unaffected: it
    # is reached only without specs, and an empty weighted density always
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

    if method not in ("auto", "bulger", "centres", "mobius", "contract"):
        raise ValueError(
            f"method must be one of 'auto', 'bulger', 'centres', 'mobius', "
            f"'contract'; got {method!r}."
        )

    # --- Dispatcher ---
    sel_kw, ordered_any, nested_any = _flat_selector_inputs(
        dens_x, dens_y, normalize=normalize,
        truncation_sigmas=truncation_sigmas)
    chosen = _select_ma_inner_product_method(user_method=method, **sel_kw)
    need_xx = (normalize == "cosine")
    need_yy = (normalize != "none")

    # Ordered ([sym]=0) attributes are not symmetrised, so the orbit
    # (Möbius) per-attribute inner product does not represent them. Force
    # the pairwise/centres path whenever any attribute is ordered at
    # r_a > 1 (r_a = 1 is vacuous). The centres path reads the actual
    # stored per-attribute centres and is correct for either reading.
    if ordered_any:
        chosen = "bulger"
    # Nested attributes are not handled by the flat orbit/Möbius entry
    # point: that path would have to flatten the levels into one value set,
    # but the inner unit's metric is block-diagonal (positions couple only
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
        # ``method`` semantics on a nested density:
        #   'bulger'   -- the joint-tuple enumeration, as for a flat density.
        #   'contract' -- the nested contraction plan, forced: an uncovered
        #                 case raises rather than falling back.
        #   'mobius'   -- the same plan. The per-level orbit (Möbius)
        #                 reduction *is* what the contraction applies at every
        #                 symmetric level, so on a nested density 'mobius' and
        #                 'contract' name one route; there is no separate flat
        #                 orbit entry point to ask for (the flat one would have
        #                 to re-enumerate the levels into a single value set,
        #                 which the block-diagonal inner metric forbids).
        #   'centres'  -- the plan with the materialised-centres route forced
        #                 for every nested attribute; raises where that route
        #                 cannot carry the attribute's declared measure.
        # 'centres' and 'mobius' formerly fell through to 'bulger', so the
        # method name described something other than what ran.
        if method in ("auto", "contract", "mobius", "centres"):
            triple = _try_nested_contract(
                dens_x, dens_y, normalize=normalize, verbose=verbose,
                force=(method != "auto"), method_name=method,
                force_route=("centres" if method == "centres" else None),
                truncation_sigmas=truncation_sigmas)
            if triple is not None:
                from .._defaults import _maybe_show_dispatch_msg as _msg
                _msg("cos_sim_exp_tens", "contract",
                     "nested: " + ",".join(_LAST_NESTED_ROUTES))
                ip_xy, ip_xx, ip_yy = triple
                if normalize == "none":
                    ip_xy = float(ip_xy) * _ip_canonical_scale(
                        dens_x, "contract", list(_LAST_NESTED_ROUTES))
                return _finalise_normalisation(ip_xy, ip_xx, ip_yy, normalize)
            # A None here means method == 'auto' and the case is not covered
            # by the contraction (the forced methods raise instead), so the
            # joint-tuple enumeration takes it.
        chosen = "bulger"

    # Dispatch-decision message: announce which inner-product path ran,
    # matching the single-multiset path's behaviour. Gated by the
    # toolbox-wide show_hints flag and throttled once per
    # (function, chosen) per top-level call; not gated by per-call
    # verbose. The multi-attribute selector does not run the empirical
    # probe, so no time estimate accompanies the message. On an ordered
    # attribute the tuple-pair path has no permutation expansion to
    # exploit (the perm side equals the comb side), so Bulger's
    # combinations-vs-permutations organisation never runs there and
    # the announce says so.
    from .._defaults import _maybe_show_dispatch_msg
    _maybe_show_dispatch_msg(
        "cos_sim_exp_tens",
        ("bulger (direct on ordered attributes)"
         if (ordered_any and chosen == "bulger") else chosen),
        "ma_select",
    )

    if chosen == "mobius":
        ip_xy, ip_xx, ip_yy = _cos_sim_exp_tens_ma_orbit(
            dens_x, dens_y, user_forced_mobius=(method == "mobius"), truncation_sigmas=truncation_sigmas,
            need_xx=need_xx, need_yy=need_yy,
        )
        # Post-hoc correctness check only. Accuracy is governed by
        # truncationSigmas: the Möbius route's agreement with
        # enumeration tracks the truncation budget, so a small cosine is
        # a legitimate value rather than a symptom, and no route is
        # diverted on the size of the result. What remains is the
        # impossible-value test (non-finite inner product, negative Gram
        # diagonal, or a cosine outside [-1, 1]), which signals a broken
        # value rather than an inaccurate one.
        #
        # ``post_hoc_guards`` off skips it: the check inspects a result
        # already computed and, when it diverts, pays for Bulger's method
        # on top of this one, so with it active the measured cost of the
        # Möbius route is not the cost of choosing it.
        from .._defaults import get_default as _gd_guard
        from .dispatch import _impossible_value_reason
        _bad = (_impossible_value_reason(ip_xy, ip_xx, ip_yy)
                if _gd_guard("post_hoc_guards") else None)
        if _bad is not None:
            warnings.warn(
                f"The Mobius route returned a value that cannot be "
                f"correct: {_bad}. This is a defect, not a loss of "
                f"accuracy, so it is not something truncationSigmas "
                f"governs. Enumeration was used instead; please report "
                f"the inputs.",
                RuntimeWarning, stacklevel=2)
            # A run the guard rejected must not seed the memoised
            # self inner products: purge this route's entries from
            # both densities before falling back.
            for _d in (dens_x, dens_y):
                for _k in [k for k in _d._self_ip_cache
                           if isinstance(k, tuple) and len(k) > 0
                           and k[0] == "mobius"]:
                    del _d._self_ip_cache[_k]
            ip_xy, ip_xx, ip_yy = _cos_sim_exp_tens_ma_pairwise(
                dens_x, dens_y, verbose=verbose,
                truncation_sigmas=truncation_sigmas,
                kernel_precision=kernel_precision,
                need_xx=need_xx, need_yy=need_yy,
            )
    elif chosen == "centres":
        ip_xy, ip_xx, ip_yy = _cos_sim_exp_tens_ma_centres(
            dens_x, dens_y, verbose=verbose,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
            need_xx=need_xx, need_yy=need_yy,
        )
    else:  # 'bulger'
        ip_xy, ip_xx, ip_yy = _cos_sim_exp_tens_ma_pairwise(
            dens_x, dens_y, verbose=verbose,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
            need_xx=need_xx, need_yy=need_yy,
        )

    if normalize == "none":
        ip_xy = float(ip_xy) * _ip_canonical_scale(dens_x, chosen)
    return _finalise_normalisation(ip_xy, ip_xx, ip_yy, normalize)



def _ip_core_ma(
    u_cell, w_u, n_j, v_cell, w_v, n_k,
    A, r_vec, sigma, is_rel, is_per, period,
    *, truncation_sigmas=None, kernel_precision=None, inner_r=None,
    wrap=None,
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
        wrap_a = 'full-image'
        if wrap is not None and len(wrap) > 0:
            wrap_a = str(wrap[0])
        return _ip_via_helper(
            u_cell[0], w_u, v_cell[0], w_v,
            int(r_vec[0]), float(sigma[0]),
            bool(is_rel[0]), bool(is_per[0]), float(period[0]),
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
            wrap_a=wrap_a,
        )

    # Dedicated route for the all-r = 1 shape (each event contributes a
    # single joint kernel; the inner product is a smoothed
    # cross-correlation --- the simplest shape the framework supports,
    # and the one point-set query sweeps exercise). The generic path
    # below allocates a per-attribute (r_a, nJ, nK) difference tensor,
    # a per-attribute quadratic form, a separate log-kernel
    # accumulator, and a masked fancy-indexed exponential; at r = 1
    # none of that structure is needed, and the direct evaluation here
    # computes the identical quantity --- same per-attribute terms, same
    # accumulation order, same truncation threshold --- with one
    # accumulator, in-place updates, and a plain exponential. Relative
    # attributes at r = 1 have a vanishing quadratic form (a 1-tuple
    # has no within-tuple differences) and contribute nothing, exactly
    # as ``_compute_Q`` evaluates them.
    all_r1 = (A >= 1 and all(int(r_vec[a]) == 1 for a in range(A))
              and (inner_r is None
                   or all(int(x) == 0 for x in inner_r)))
    if all_r1:
        return _ip_r1_direct(
            u_cell, w_u, int(n_j), v_cell, w_v, int(n_k),
            A, sigma, is_rel, is_per, period,
            truncation_sigmas=truncation_sigmas, wrap=wrap,
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
            wrap=wrap,
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
            inner_r=inner_r, wrap=wrap,
            truncation_sigmas=truncation_sigmas,
        )
        E = _trunc_log_kernel_exp(
            log_kernel, truncation_sigmas,
            n_terms=int(n_j) * int(n_k),
        )
        acc = acc + E @ w_v[c_start:c_end]
    return float(w_u @ acc)



def _ip_r1_direct(
    u_cell, w_u, n_j, v_cell, w_v, n_k,
    A, sigma, is_rel, is_per, period,
    *, truncation_sigmas, wrap=None,
):
    """Direct MA inner product for the all-r = 1 shape.

    Computes the same quantity as the generic ``_ip_full_ma`` /
    ``_ma_log_kernel`` path --- per-attribute terms accumulated in the
    same order, with the same truncation threshold (including the
    ``n_terms`` tightening) --- but with a single (nJ, nK) accumulator,
    in-place arithmetic, and a plain exponential in place of the
    generic path's per-attribute tensors and masked fancy-indexed
    ``exp``. Chunking along the comb side follows the generic path's
    memory heuristic, so ``kernel_chunk_bytes`` is honoured.

    ``truncation_sigmas`` must arrive resolved (finite positive), as
    ``_ip_core_ma`` guarantees.
    """
    threshold = -0.5 * float(truncation_sigmas) ** 2
    n_terms = int(n_j) * int(n_k)
    if n_terms > 1:
        threshold = threshold - math.log(float(n_terms))

    bytes_per_col = 3 * int(n_j) * 8
    mem_limit = kernel_chunk_bytes_resolved()
    chunk_size = max(1, int(mem_limit // max(bytes_per_col, 1)))

    acc = np.zeros(int(n_j), dtype=np.float64)
    for c_start in range(0, int(n_k), chunk_size):
        c_end = min(c_start + chunk_size, int(n_k))
        L = np.zeros((int(n_j), c_end - c_start), dtype=np.float64)
        for a in range(A):
            if bool(is_rel[a]):
                # A 1-tuple has no within-tuple differences: the
                # relative quadratic form vanishes identically, as
                # _compute_Q evaluates it, so the attribute
                # contributes nothing to the log-kernel.
                continue
            d = (u_cell[a][0][:, None]
                 - v_cell[a][0][None, c_start:c_end])
            wrap_a = 'full-image'
            if wrap is not None and a < len(wrap):
                wrap_a = str(wrap[a])
            if bool(is_per[a]) and wrap_a == 'full-image':
                from .._wrapped_kernel import wrapped_gaussian_1d
                theta = wrapped_gaussian_1d(
                    d, float(sigma[a]), float(period[a]),
                    float(truncation_sigmas),
                    exponent_denominator=4,
                )
                L += np.log(theta)
                continue
            if bool(is_per[a]):
                p_a = float(period[a])
                d = d - p_a * np.floor(d / p_a + 0.5)
            np.multiply(d, d, out=d)
            d /= (4 * float(sigma[a]) ** 2)
            L -= d
        below = L < threshold
        np.exp(L, out=L)
        L[below] = 0.0
        acc += L @ w_v[c_start:c_end]
    return float(w_u @ acc)


def _ip_full_ma(
    u_cell, w_u, n_j, v_cell, w_v, n_k,
    A, r_vec, sigma, is_rel, is_per, period,
    *, truncation_sigmas=None, inner_r=None, wrap=None,
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
        inner_r=inner_r, wrap=wrap,
        truncation_sigmas=truncation_sigmas,
    )
    E = _trunc_log_kernel_exp(
        log_kernel, truncation_sigmas, n_terms=int(n_j) * int(n_k),
    )
    return float(w_u @ (E @ w_v))



def _gram_is_accurate_enough(U, V, sigma, truncation_sigmas):
    """Whether the Gram form's rounding sits inside the accuracy floor.

    The Gram identity forms ``|u|^2 + |v|^2 - 2 u.v``, so its rounding is
    relative to the size of those squares rather than to the distance
    they encode. After the shared shift the coordinates are of the order
    of the attribute's own spread ``s``, giving an error in the exponent
    of about ``eps * s^2 / (4 sigma^2)``. That is negligible whenever
    the spread is comparable to sigma, and grows as sigma shrinks
    against it --- measured at 1.6e-12 for a spread of ~20 at
    ``sigma = 0.1``. Compared against the floor ``truncation_sigmas``
    implies, so a caller asking for accuracy-floor accuracy gets the
    difference form and one asking for the default gets the fast one.
    """
    from .._defaults import truncation_floor

    s2 = 0.0
    for M in (U, V):
        if M.size:
            origin = float(U.flat[0])
            s2 = max(s2, float(np.max(np.abs(M - origin))) ** 2)
    if s2 == 0.0:
        return True
    predicted = np.finfo(np.float64).eps * s2 / (4.0 * float(sigma) ** 2)
    return predicted <= 0.1 * truncation_floor(truncation_sigmas)


def _gram_quadratic_form(U, V, block):
    """``Q(u_j - v_k)`` for a non-periodic attribute, without the tensor.

    In every non-periodic mode the quadratic form is a squared Euclidean
    distance between (possibly quotiented) coordinates, so it is a Gram
    matrix: ``|u|^2 + |v|^2 - 2 u.v``, one ``gemm`` in place of an
    ``(r, nJ, nK)`` difference array. ``block`` selects the quotient ---
    ``0`` for absolute (raw coordinates), the tuple length for relative
    (the whole tuple's all-ones removed), or the co-transposition unit
    size for a nested attribute (each block's own all-ones removed, the
    form being the sum over blocks).

    Both operands are shifted by one of the attribute's own values
    first. The Gram identity cancels two large numbers when the
    coordinates sit far from the origin, which costs significant digits
    -- measured against the difference form, the log-kernel departed by
    4e-3 at magnitude 1e6 and 5e-1 at 1e7. Everything here depends on
    the operands only through their differences, so a shift shared by
    both is exact; taking it from the data rather than from a mean makes
    the subtraction itself exact as well (Sterbenz), where a mean would
    inject a rounding error at just those magnitudes.
    """
    r = int(U.shape[0])
    origin = float(U.flat[0]) if U.size else 0.0
    U = U - origin
    V = V - origin
    if block <= 0 or block >= r:
        parts = [(U, V)] if block <= 0 else [
            (U - U.mean(axis=0)[None, :], V - V.mean(axis=0)[None, :])]
    else:
        parts = []
        for b in range(r // block):
            sl = slice(b * block, (b + 1) * block)
            Ub, Vb = U[sl], V[sl]
            parts.append((Ub - Ub.mean(axis=0)[None, :],
                          Vb - Vb.mean(axis=0)[None, :]))
    Q = None
    for Ub, Vb in parts:
        blk = (np.einsum("ij,ij->j", Ub, Ub)[:, None]
               + np.einsum("ij,ij->j", Vb, Vb)[None, :]
               - 2.0 * (Ub.T @ Vb))
        Q = blk if Q is None else Q + blk
    np.maximum(Q, 0.0, out=Q)
    return Q


def _ma_log_kernel(
    u_cell, v_cell, n_j, n_k,
    A, r_vec, sigma, is_rel, is_per, period,
    *, inner_r=None, wrap=None, truncation_sigmas=None,
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

    ``wrap`` (optional) is a per-attribute sequence selecting the
    abs-per measure: 'full-image' (default) uses the torus (all-image)
    1-D wrapped Gaussian per coordinate; 'single-image' uses the nearest-
    image reduction (the pre-v3 behaviour). Non-periodic and rel
    attributes ignore this axis. ``None`` matches the pre-v3 default,
    i.e. 'full-image' everywhere.
    """
    log_kernel = np.zeros((int(n_j), int(n_k)), dtype=np.float64)
    for a in range(A):
        r_a = int(r_vec[a])
        r_in = 0 if inner_r is None else int(inner_r[a])

        # Non-periodic modes: the quadratic form is a squared Euclidean
        # distance, so it comes out of one gemm rather than an
        # (r_a, nJ, nK) difference array. Periodic attributes keep the
        # tensor path below, where the wrap makes the form non-Euclidean.
        if not bool(is_per[a]) and _gram_is_accurate_enough(
                u_cell[a], v_cell[a], float(sigma[a]), truncation_sigmas):
            block = r_in if r_in > 0 else (r_a if bool(is_rel[a]) else 0)
            Q_a = _gram_quadratic_form(u_cell[a], v_cell[a], block)
            log_kernel = log_kernel - Q_a / (4 * float(sigma[a]) ** 2)
            continue

        D = u_cell[a][:, :, None] - v_cell[a][:, None, :]  # (r_a, nJ, nK)

        if r_in > 0:
            # Inner unit: block-diagonal sum of per-event quotient forms.
            # _compute_Q applies the pairwise wrap inside, so the full
            # tuples enter without an outer wrap.
            Q_a = _compute_Q_inner_blocks(
                D, r_in, bool(is_per[a]), float(period[a]), reduced=False)
            log_kernel = log_kernel - Q_a / (4 * float(sigma[a]) ** 2)
            continue

        # Abs-per full-image via the shared 1-D wrapped Gaussian
        # (overlap convention, exponent_denominator=4). The r-tuple
        # full-image kernel factors as prod_k theta(d_k), so
        # log kernel = sum_k log theta(d_k). Single-image opt-in
        # falls through to the Q-form path below with a nearest-image
        # reduction, matching the pre-v3 behaviour.
        wrap_a = 'full-image'
        if wrap is not None and a < len(wrap):
            wrap_a = str(wrap[a])
        if (is_per[a] and not is_rel[a]
                and wrap_a == 'full-image'):
            from .._wrapped_kernel import wrapped_gaussian_1d, _image_count_L
            from .._defaults import get_default
            ts_a = (float(get_default('truncation_sigmas'))
                    if truncation_sigmas is None
                    else float(truncation_sigmas))
            # Single-image short-circuit. When the truncation budget
            # admits no images beyond the nearest one (L = 0, which at
            # the 6-sigma default holds for sigma/P <= 0.059 in this
            # convention), theta(d) *is* the nearest-image Gaussian
            # exp(-d^2 / (4 sigma^2)), so sum_k log theta(d_k) is
            # -Q / (4 sigma^2) on the nearest-image-reduced
            # differences --- exactly what the Q-form path below
            # computes, without the exp-then-log round trip on the
            # (r_a, nJ, nK) tensor that the theta accumulation pays.
            # Same measure, same number to ~3e-16; measured 1.6x
            # (r = 2) to 3.3x (r = 3) cheaper.
            if _image_count_L(float(sigma[a]), float(period[a]), ts_a,
                              4) > 0:
                theta = wrapped_gaussian_1d(
                    D, float(sigma[a]), float(period[a]), ts_a,
                    exponent_denominator=4,
                )
                log_kernel = log_kernel + np.sum(np.log(theta), axis=0)
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
#  the K_a values of event n_X (X-side) against those of n_Y
#  (Y-side), with the group's mode parameters. Bulger's method
#  collapses this into a flat (n_J × n_K) bilinear form that scales
#  as N² · Π_a [r_a! · C(K_a, r_a)]²; the Möbius form scales as
#  N² · A · |Ω_{r_a}| · K_a², a substantial saving when K_a is
#  non-trivial.
#
#  NaN-padded ``p_attr`` (variable K_a per event) is handled inside
#  ``_ma_per_attr_inner_matrix`` by zero-weight padding, which makes
#  every orbit term containing a padded value vanish; the selector does
#  not route on it. Per-attribute r_a > _ORBIT_R_MAX_SHIPPED falls back
#  to Bulger's method (no orbit table shipped at that order).
#
#  Per-attribute Möbius calls apply the single-multiset convention's
#  (σ_a √π)^{r_a} prefactor, so the Möbius-method MA bare triple
#  (ip_xy, ip_xx, ip_yy) differs from Bulger's MA triple by
#  Π_a (σ_a √π)^{r_a} · r_a! — which cancels in the cosine.


def _trunc_log_kernel_exp(log_kernel, truncation_sigmas, *, n_terms=None):
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

    ``n_terms`` states how many entries the caller will sum. The floor
    is stated per entry, but the inner product is a sum, so discarding
    ``n_terms`` entries each just under the floor admits an error of
    ``n_terms`` times the floor on the summed value. This holds in every
    mode: the periodic wrap makes the tail broadest, so the shortfall is
    largest there, but an untightened threshold overshoots the stated
    accuracy wherever the block is large enough. Passing the count
    lowers the per-entry threshold to ``floor / n_terms``, which bounds
    the total discarded mass by the floor itself --- the scale the
    accuracy is stated on. In log space this is a shift of
    ``-log(n_terms)``, so the equivalent width is
    ``sqrt(k^2 + 2 log(n_terms))`` and the cost is a modestly wider
    kernel window rather than a different algorithm.
    """
    from .._defaults import resolve_truncation_sigmas
    truncation_sigmas = resolve_truncation_sigmas(truncation_sigmas)
    threshold = -0.5 * truncation_sigmas ** 2
    if n_terms is not None and n_terms > 1:
        threshold = threshold - math.log(float(n_terms))
    mask = log_kernel >= threshold
    out = np.zeros_like(log_kernel)
    out[mask] = np.exp(log_kernel[mask])
    return out









































def _cos_sim_exp_tens_ma_orbit(dens_x, dens_y, *, truncation_sigmas=None,
                               need_xx: bool = True,
                               need_yy: bool = True,
                               user_forced_mobius: bool = False):
    """Compute (ip_xy, ip_xx, ip_yy) for the MA case via per-attribute
    Möbius method (JMM Eq. 3.4 plus Rem. 3.1).

    Caller is responsible for ensuring no NaN in ``p_attr`` and for
    structural compatibility of the two densities.

    ``need_xx=False`` skips <X,X> when it is neither memoised nor
    consumed by the caller's normalisation; the triple's first self
    slot is then ``None``. Both self inner products are memoised on
    their densities, keyed on this route's per-attribute
    closed-form-vs-grid choices; the dispatcher purges this route's
    entries if its post-hoc impossible-value guard trips, so a broken
    run never seeds the cache.

    Note: previous versions also returned a ``worst_ratio`` aggregating
    per-entry cancellation ratios across the (N_x × N_y) inner-product
    matrices. That diagnostic was found to over-conservatively flag
    correct results — the per-entry ratio reflects cancellation in
    individual Möbius cells, but the cosine consumes only the
    sums Σ_{n,m} P[n,m], where individual entries with bad ratios
    contribute negligibly when their absolute value is small. Removed
    in favour of the post-hoc impossible-value check (see
    ``_impossible_value_reason``) at the dispatcher level; accuracy
    short of impossibility is governed by ``truncation_sigmas``.
    """
    A = dens_x.n_attrs
    N_x = dens_x.n
    N_y = dens_y.n

    # Per-attribute route choice, hoisted because it is part of the
    # self-IP cache key: the closed form drops a per-attribute constant
    # prefactor that the grid contraction keeps, so a self inner product
    # is reusable only against triples that made the same per-attribute
    # choices (the prefactor cancels only within one such triple).
    choices = tuple(
        bool(_ma_rel_attr_prefers_centres(
            dens_x.p_attr[a], dens_y.p_attr[a],
            float(dens_x.sigma[a]), int(dens_x.r[a]),
            bool(dens_x.is_rel[a]), bool(dens_x.is_per[a]),
            float(dens_x.period[a]),
            truncation_sigmas=truncation_sigmas,
            user_forced_mobius=user_forced_mobius,
        ))
        for a in range(A)
    )
    key = _self_ip_cache_key("mobius", truncation_sigmas, None, choices)
    xx_cached = key in dens_x._self_ip_cache
    yy_cached = key in dens_y._self_ip_cache
    compute_xx = need_xx and not xx_cached
    compute_yy = need_yy and not yy_cached

    P_xy = np.ones((N_x, N_y), dtype=np.float64)
    P_xx = np.ones((N_x, N_x), dtype=np.float64) if compute_xx else None
    P_yy = np.ones((N_y, N_y), dtype=np.float64) if compute_yy else None

    for a in range(A):
        r_a = int(dens_x.r[a])
        sigma = float(dens_x.sigma[a])
        is_rel = bool(dens_x.is_rel[a])
        is_per = bool(dens_x.is_per[a])
        period = float(dens_x.period[a])

        Px, Py = dens_x.p_attr[a], dens_y.p_attr[a]
        Wx, Wy = dens_x.w[a], dens_y.w[a]

        wrap_a = (str(dens_x.wrap[a])
                  if hasattr(dens_x, 'wrap') and dens_x.wrap is not None
                  else 'full-image')
        if choices[a]:
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
            P_xy *= _closed_form_attr_matrix_from(cx, cy, truncation_sigmas,
                                                  wrap_a)
            if P_xx is not None:
                P_xx *= _closed_form_attr_matrix_from(
                    cx, cx, truncation_sigmas, wrap_a)
            if P_yy is not None:
                P_yy *= _closed_form_attr_matrix_from(
                    cy, cy, truncation_sigmas, wrap_a)
        else:
            P_xy *= _ma_per_attr_inner_matrix(
                Px, Wx, Py, Wy, sigma, r_a, is_rel, is_per, period,
                truncation_sigmas=truncation_sigmas,
                wrap=wrap_a,
            )
            if P_xx is not None:
                P_xx *= _ma_per_attr_inner_matrix(
                    Px, Wx, Px, Wx, sigma, r_a, is_rel, is_per, period,
                    truncation_sigmas=truncation_sigmas,
                    wrap=wrap_a,
                )
            if P_yy is not None:
                P_yy *= _ma_per_attr_inner_matrix(
                    Py, Wy, Py, Wy, sigma, r_a, is_rel, is_per, period,
                    truncation_sigmas=truncation_sigmas,
                    wrap=wrap_a,
                )

    ip_xy = float(P_xy.sum())
    if xx_cached:
        ip_xx = dens_x._self_ip_cache[key]
    elif compute_xx:
        ip_xx = float(P_xx.sum())
        dens_x._self_ip_cache[key] = ip_xx
    else:
        ip_xx = None
    if yy_cached:
        ip_yy = dens_y._self_ip_cache[key]
    elif compute_yy:
        ip_yy = float(P_yy.sum())
        dens_y._self_ip_cache[key] = ip_yy
    else:
        ip_yy = None
    return ip_xy, ip_xx, ip_yy


def _declared_wrap(dens_x, dens_y, a):
    """The wrap the two densities declare on attribute ``a``.

    The wrap declares a measure, and the measure of an inner product must
    be one thing, so the two densities must agree wherever the wrap is
    read --- on a periodic attribute. A disagreement there raises rather
    than being resolved in favour of ``dens_x``, which made the result
    depend on operand order. On a non-periodic attribute the wrap axis has
    no meaning and is not compared. Twin of the MATLAB ``wrapPair``
    helpers (``mpt:wrapMismatch``).
    """
    def _wrap_of(d):
        w = getattr(d, 'wrap', None)
        return (str(w[a]) if w is not None and a < len(w) else 'full-image')
    wx, wy = _wrap_of(dens_x), _wrap_of(dens_y)
    if wx != wy and bool(dens_x.is_per[a]):
        raise ValueError(
            f"wrap mismatch on attribute {a}: dens_x declares {wx!r} and "
            f"dens_y declares {wy!r}. The wrap declares the measure, so "
            f"the two densities must declare the same wrap on every "
            f"periodic attribute."
        )
    return wx


def _nested_admissible_routes(dens_x, dens_y, a, ts=None):
    """The routes for nested attribute ``a`` that carry its declared measure.

    This is the measure rule and nothing else: it says which routes are on
    offer, never which is taken. :func:`_nested_attr_route` applies
    ``force_route`` and, where more than one route survives, the cost model of
    :mod:`~mpt._tensor._nested_cost` picks among them.

    An absolute attribute reports ``['contract']`` alone. The materialised
    centres carry the absolute measure too (both routes read the attribute's
    declared ``wrap``), but ``auto`` has always kept absolute attributes on
    the contraction and still does; ``method='centres'`` reaches the centres
    route there through ``force_route``.

    A relative-periodic attribute is governed by ``wrap`` and by the
    sigma/period threshold together, and the rule is symmetric in the two
    declarations, exactly as on the flat path:

    * above the threshold the two readings differ by more than the truncation
      floor, so only the route that computes the declared one is admissible
      --- the tau-grid under ``wrap='full-image'``, the centres under
      ``wrap='single-image'``;
    * below it they agree inside the floor, so **both** routes are admissible
      under **either** declaration and the price decides. This is what
      :func:`~mpt._tensor.dispatch._select_ma_inner_product_method` does with
      Bulger's method and the Möbius method: its ``wrap`` override is reached
      only above the threshold, and below it the two are raced whatever
      ``wrap`` says.

    ``ts`` is the resolved per-call truncation width (``None`` resolves
    the default); the threshold is a function of it.
    """
    is_rel = bool(dens_x.is_rel[a])
    is_per = bool(dens_x.is_per[a])
    if not is_rel:
        return ["contract"]
    if not is_per:
        return ["centres", "contract_relnonper"]
    from .dispatch import _orbit_sigma_over_p_threshold
    from .._defaults import resolve_truncation_sigmas
    wrap_a = _declared_wrap(dens_x, dens_y, a)
    sigma = float(dens_x.sigma[a])
    period = float(dens_x.period[a])
    limit = _orbit_sigma_over_p_threshold(resolve_truncation_sigmas(ts))
    if period > 0.0 and sigma / period > limit:
        # Beyond the floor the declared measure has exactly one carrier.
        return ["centres"] if wrap_a == 'single-image' else ["taugrid"]
    return ["centres", "taugrid"]


def _nested_attr_plan(dens_x, dens_y, a, force_route=None,
                      skip_xx=False, skip_yy=False, ts=None):
    """Route plus the *shared* quadrature grid for a nested attribute, decided
    once from the (x, y) pair.

    Returning the grid here -- rather than recomputing it inside each of xy, xx
    and yy -- is what makes the cosine normalise exactly: the relative grids are
    value-dependent, so a per-call grid would discretise the three inner
    products differently and the ratio would drift off 1 (breaking, e.g.,
    transposition invariance). One grid spanning both densities is used for all
    three. See :func:`_nested_attr_route` for the route meanings and for what
    ``force_route`` may ask for.

    ``skip_xx`` / ``skip_yy`` are passed to the cost model so a memoised (or
    unconsumed) self inner product is not priced, mirroring the flat
    selector's per-route skip flags. ``ts`` is the resolved per-call
    truncation width (``None`` resolves the default); it sets the
    quadrature tolerance and the tau-grid node count.
    """
    from .._defaults import resolve_truncation_sigmas, truncation_floor
    ts = resolve_truncation_sigmas(ts)
    route = _nested_attr_route(dens_x, dens_y, a, force_route=force_route,
                               skip_xx=skip_xx, skip_yy=skip_yy, ts=ts)
    if route in ("centres", "contract"):
        return route, None
    from ._nested_contraction import (auto_ntau_default, auto_taus_line)
    sigma = float(dens_x.sigma[a])
    period = float(dens_x.period[a])
    # Same kernel-value floor as every other truncation path
    # (:func:`truncation_floor` resolves None -> default; inf ->
    # accuracy-floor width; ``accuracy_floor_context`` honoured).
    tol = truncation_floor(ts)
    if route == "taugrid":
        # The taugrid route computes (C) full-image via the tau-average of the
        # wrapped Gaussian (v3+). The ``wrap='single-image'`` opt-in, which
        # asks for (A) instead, is honoured in :func:`_nested_attr_route`, so
        # by here the declared measure is full-image.
        # Period-only grid; node count from the shared helper so the flat and
        # nested all-image grids coincide exactly.
        return route, np.linspace(0.0, period,
                                  auto_ntau_default(period, sigma, ts),
                                  endpoint=False)
    # contract_relnonper: one line grid spanning both densities' values.
    px = np.asarray(dens_x.p_attr[a], dtype=np.float64)
    py = np.asarray(dens_y.p_attr[a], dtype=np.float64)
    allv = np.concatenate([px[np.isfinite(px)].ravel(),
                           py[np.isfinite(py)].ravel()])
    return route, auto_taus_line(allv, allv, sigma, tol)


#: Per-attribute nested routes taken by the most recent nested-contraction
#: call, in attribute order (``'-'`` for an attribute that took neither a
#: nested nor an ordered-flat route). Diagnostic only: it feeds the
#: dispatch-decision message's routing reason and gives the tests a way to
#: assert which route ran without timing it. Nothing routes on it.
_LAST_NESTED_ROUTES: list = []

#: Prices behind the most recent nested plan-versus-enumeration decision:
#: ``chosen``, ``plan_ms``, ``enum_ms`` and the per-attribute ``detail`` of
#: :func:`~mpt._tensor._nested_cost.select_nested_method`. Diagnostic only,
#: like ``_LAST_NESTED_ROUTES``; :func:`~mpt._tensor.explain.explain_dispatch`
#: reports the same quantities by re-running the model rather than by reading
#: this.
_LAST_NESTED_COSTS: dict = {}


def _nested_self_ip_skip_flags(dens_x, dens_y, normalize):
    """``(skip_xx, skip_yy)`` for the nested cost model.

    A self inner product that is already memoised on its density, or that the
    requested normalisation does not consume, costs nothing at call time and
    must not be priced. As on the flat path the flags are shared by the two
    sides of the comparison --- the contraction plan and the joint-tuple
    enumeration --- rather than read off each side's own memo; see
    :func:`_self_ip_memoised` for why an asymmetric flag locks the first
    winner in, and what the shared flag trades for that.
    """
    need_xx = (normalize == "cosine")
    need_yy = (normalize != "none")
    return (((not need_xx) or _self_ip_memoised(dens_x)),
            ((not need_yy) or _self_ip_memoised(dens_y)))


def _nested_enumeration_admissible(dens_x, dens_y, ts=None):
    """True when the joint-tuple enumeration carries the declared measure.

    The enumeration evaluates the minimum-image wrapped-difference kernel on a
    relative-periodic attribute --- measure (A). It may therefore serve a
    ``wrap='full-image'`` attribute only below the sigma/period threshold,
    where the two readings agree inside the truncation floor, and serves a
    ``wrap='single-image'`` attribute at any sigma/period. This is the rule
    :func:`~mpt._tensor.dispatch._select_ma_inner_product_method` applies to
    Bulger's method on the flat path, read off both densities instead of
    a wrap vector. ``ts`` is the resolved per-call truncation width
    (``None`` resolves the default).
    """
    from .dispatch import _orbit_sigma_over_p_threshold
    from .._defaults import resolve_truncation_sigmas
    limit = _orbit_sigma_over_p_threshold(resolve_truncation_sigmas(ts))
    for a in range(int(dens_x.n_attrs)):
        if not (bool(dens_x.is_rel[a]) and bool(dens_x.is_per[a])):
            continue
        if _declared_wrap(dens_x, dens_y, a) == 'single-image':
            continue
        period = float(dens_x.period[a])
        if period > 0.0 and float(dens_x.sigma[a]) / period > limit:
            return False
    return True


def _nested_prefers_enumeration(dens_x, dens_y, routes_by_attr, *,
                                skip_xx, skip_yy, ts=None):
    """True when the enumeration is priced cheaper than the planned routes.

    ``routes_by_attr`` maps each nested attribute to the route already chosen
    for it by the measure rule and the per-attribute cost model, so the plan
    is priced as what would actually run rather than re-raced here. The
    comparison mirrors the flat selector's Bulger-versus-Möbius one and
    records its prices in ``_LAST_NESTED_COSTS``.
    """
    from ._nested_cost import select_nested_method
    chosen, plan_ms, enum_ms, detail = select_nested_method(
        dens_x, dens_y,
        admissible_by_attr={a: [rt] for a, rt in routes_by_attr.items()},
        enumeration_ok=_nested_enumeration_admissible(dens_x, dens_y, ts),
        skip_xx=skip_xx, skip_yy=skip_yy,
        return_costs=True, truncation_sigmas=ts)
    _LAST_NESTED_COSTS.clear()
    _LAST_NESTED_COSTS.update(chosen=chosen, plan_ms=plan_ms,
                              enum_ms=enum_ms, detail=detail)
    return chosen == "bulger"


def _nested_attr_route(dens_x, dens_y, a, force_route=None,
                       skip_xx=False, skip_yy=False, ts=None):
    """Per-attribute route for a nested attribute, decided once so xy, xx and
    yy share a single measure.

    - ``'contract'`` -- absolute and absolute-periodic: the kernel is a
      one-body product across coordinates, so the event-pair-vectorised per-level
      contraction applies the orbit (Möbius) reduction at symmetric levels and
      enumeration at ordered ones, mirroring the flat per-attribute matrix and
      never materialising the tuple set.
    - ``'centres'`` -- relative modes on the materialised-centres path; for
      relative-non-periodic its exact analytic quadratic, for relative-periodic
      the minimum-image measure.
    - ``'contract_relnonper'`` -- relative-non-periodic when the translation-grid
      contraction is cheaper (large ``r = 1`` or symmetric levels, e.g. spectral
      cells). Same measure as the centres quadratic, to grid accuracy; a pure
      speed choice.
    - ``'taugrid'`` -- relative-periodic on the all-image tau-grid contraction,
      the transposition average over the period.

    **Raw single-multiset batched input** (replaces ``batch_cos_sim_exp_tens``):

    - ``cos_sim_exp_tens(P1, W1, P2, W2, sigma, r, is_rel, is_per, period)``
      where at least one of ``P1``, ``P2`` is a 2-D ``(M, K)`` matrix
      (rows are chords; NaN-padded for variable cardinality), ``W1``,
      ``W2`` likewise (or ``None`` for uniform). Returns ``(M,)``. If
      only one operand is a matrix and the other is a 1-D vector of
      length ``K``, the vector is broadcast across the matrix's ``M``
      rows.

    **The measure is declared by ``wrap``, not by the dispatch.** This mirrors
    the rule the flat relative-periodic path already enforces in
    :func:`~mpt._tensor.dispatch._select_ma_inner_product_method`:

    - ``wrap='single-image'`` on a relative-periodic attribute declares the (A)
      minimum-image measure; ``wrap='full-image'`` (the default) declares the
      (C) all-image measure.
    - Above ``sigma/period = _orbit_sigma_over_p_threshold(truncation_sigmas)``
      the two readings differ by more than the truncation floor, so each
      declaration has exactly one carrier and that route is taken whatever it
      costs: the centres route under (A), the tau-grid under (C). A cheaper
      route to a different number is not a cheaper route.
    - At or below the threshold the two agree inside the floor, so both routes
      serve either declaration and the price decides --- as on the flat path,
      whose ``wrap`` override is likewise reached only above the threshold.

    Only where both routes carry the declared measure does the cost model of
    :mod:`~mpt._tensor._nested_cost` decide, by pricing each survivor in
    milliseconds from its fitted law and diverting the materialising centres
    route where its bundle exceeds
    ``dispatch._CENTRES_WORKING_SET_SOFT_BUDGET``. That model prices in wall
    time rather than by a raw operation count, because a kernel entry and a
    unit of quadrature work do not cost the same. Relative-non-periodic
    attributes offer only same-measure choices, so they stay cost-driven
    throughout; absolute attributes stay on the contraction.

    ``force_route`` names a route the caller has forced (``method='centres'``
    forces ``'centres'``); it overrides the cost race but not the measure rule,
    which raises instead of silently returning a different measure. ``ts``
    is the resolved per-call truncation width (``None`` resolves the
    default).
    """
    is_rel = bool(dens_x.is_rel[a])
    is_per = bool(dens_x.is_per[a])
    if not is_rel:
        if force_route == "centres":
            return "centres"
        return "contract"
    sigma = float(dens_x.sigma[a])
    period = float(dens_x.period[a])
    admissible = _nested_admissible_routes(dens_x, dens_y, a, ts)
    if is_per and admissible == ["taugrid"]:
        if force_route == "centres":
            from .dispatch import _orbit_sigma_over_p_threshold
            from .._defaults import resolve_truncation_sigmas
            limit = _orbit_sigma_over_p_threshold(
                resolve_truncation_sigmas(ts))
            raise ValueError(
                f"method='centres' cannot be honoured on relative-periodic "
                f"nested attribute {a} at sigma/period = "
                f"{sigma / period:.4g}: above {limit:g} the minimum-image "
                f"centres route no longer computes the declared "
                f"full-image measure. Pass wrap='single-image' on this "
                f"attribute to ask for the minimum-image measure, or use "
                f"method='auto'."
            )
        return "taugrid"
    if force_route == "centres":
        return "centres"
    if len(admissible) == 1:
        return admissible[0]
    from ._nested_cost import price_nested_attr
    return price_nested_attr(dens_x, dens_y, a, admissible,
                             skip_xx=skip_xx, skip_yy=skip_yy,
                             truncation_sigmas=ts)[0]


def _nested_attr_matrix(dens_x, dens_y, a, route, taus, truncation_sigmas=None):
    """(N_x, N_y) per-attribute inner matrix for a nested attribute, on the
    given ``route`` and shared ``taus`` from :func:`_nested_attr_plan` (passed
    in so xy, xx and yy share one measure and one grid).

    ``truncation_sigmas`` is the per-call width (``None`` resolves the
    default); every kernel on every route is truncated at it, as in the
    MATLAB ``nestedContract``, which resolves it once at entry."""
    from .._defaults import resolve_truncation_sigmas
    ts = resolve_truncation_sigmas(truncation_sigmas)
    if route == "centres":
        cx = _closed_form_attr_centres(dens_x, a)
        cy = _closed_form_attr_centres(dens_y, a)
        wrap_a = _declared_wrap(dens_x, dens_y, a)
        return _closed_form_attr_matrix_from(cx, cy, ts, wrap_a)
    from ._nested_contraction import build_recipe, nested_attr_matrix
    is_rel = bool(dens_x.is_rel[a])
    is_per = bool(dens_x.is_per[a])
    sigma = float(dens_x.sigma[a])
    period = float(dens_x.period[a])
    spec_x = dens_x.nested[a]
    spec_y = dens_y.nested[a]
    r_levels = np.asarray(spec_x["r"]).ravel()
    sym_levels = np.asarray(spec_x["sym"]).ravel()
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
    # 'contract': absolute / absolute-periodic. The abs-per kernel is the
    # attribute's declared wrap (full-image by default), the same object
    # the centres route above reads.
    wrap_a = _declared_wrap(dens_x, dens_y, a)
    return nested_attr_matrix(rx, ry, PX, PY, dens_x.w[a], dens_y.w[a],
                              sigma, is_per, period, ts, taus=None,
                              wrap_a=wrap_a)


def _try_nested_contract(dens_x, dens_y, *, normalize, verbose, force=False,
                         force_route=None, method_name="contract",
                         truncation_sigmas=None):
    """Closed-form inner product of a single nested attribute.

    Returns (ip_xy, ip_xx, ip_yy) when the case is covered -- one nested
    attribute, outer/no ``[rel]``, cosine or one-sided normalisation,
    NaN-padded (variable-K) values included -- via the per-level dispatch of
    :func:`_nested_attr_plan` / :func:`_nested_attr_matrix`; otherwise ``None``,
    and the caller routes to the exact enumeration. The route is decided once
    so xy, xx and yy share one measure: absolute and absolute-periodic go to the
    contraction (exact, per-level Möbius/Bulger); relative-non-periodic to the
    centres analytic quadratic or, when cheaper, the translation-grid
    contraction (same measure, to grid accuracy); relative-periodic to the route
    its declared ``wrap`` calls for -- the all-image tau-grid for the default
    full-image measure above the sigma/period threshold, the minimum-image
    centres under ``wrap='single-image'``, and whichever is cheaper below the
    threshold, where the two agree inside the truncation floor (see
    :func:`_nested_attr_route`). Every route costs no more than the equivalent
    flat attribute.

    ``force_route`` is passed through to :func:`_nested_attr_plan`; it forces a
    route for every nested attribute, and raises where that route cannot carry
    the declared measure.

    ``truncation_sigmas`` is the caller's per-call width. It is resolved
    once here (``None`` -> the default, ``inf`` -> the accuracy-floor
    width) and the resolved value passed down to the measure rule, the
    quadrature tolerance, every kernel, the centres route, the cost model
    and the memo key, exactly as the MATLAB ``nestedContract`` resolves
    ``truncationSigmas`` at entry.
    """
    def _decline(reason):
        if force:
            raise ValueError(
                f"method={method_name!r} is not available here: {reason}. "
                f"Use method='auto' (which falls back automatically) or "
                f"method='bulger'."
            )
        return None

    from .._defaults import resolve_truncation_sigmas
    ts = resolve_truncation_sigmas(truncation_sigmas)
    if normalize not in ("cosine", "oneSidedDenom", "none"):
        return _decline(f"unsupported normalisation {normalize!r}")
    if dens_x.n_attrs != dens_y.n_attrs:
        return _decline("the two densities have different attribute counts")
    if dens_x.n_attrs != 1:
        # Nested attribute(s) tensored with further attributes: the cosine
        # factorises per event-pair across attributes (JMM Eq 3.4), so the
        # nested factor goes through the contraction and the rest through the
        # per-attribute MA matrices, instead of enumerating the joint tuple.
        return _try_nested_contract_ma(
            dens_x, dens_y, normalize=normalize, verbose=verbose, force=force,
            force_route=force_route, method_name=method_name,
            truncation_sigmas=ts)
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
    # per-level Möbius/Bulger contraction; relative-non-periodic through the
    # materialised centres or, when cheaper, the translation grid;
    # relative-periodic through whichever route carries its declared ``wrap``
    # measure. The route is decided once so xy, xx and yy share one measure.
    skip_xx, skip_yy = _nested_self_ip_skip_flags(
        dens_x, dens_y, normalize)
    route, taus = _nested_attr_plan(dens_x, dens_y, 0,
                                    force_route=force_route,
                                    skip_xx=skip_xx, skip_yy=skip_yy, ts=ts)
    # The plan is a candidate, not a conclusion: under ``auto`` it is priced
    # against the joint-tuple enumeration and the cheaper is taken, mirroring
    # the flat selector's Bulger-versus-Möbius comparison. Declining here
    # returns the caller to the enumeration, which is what a ``None`` has
    # always meant. A forced method is never diverted.
    if not force and _nested_prefers_enumeration(
            dens_x, dens_y, {0: route},
            skip_xx=skip_xx, skip_yy=skip_yy, ts=ts):
        _LAST_NESTED_ROUTES[:] = []
        return None
    _LAST_NESTED_ROUTES[:] = [route]
    ip_xy = float(_nested_attr_matrix(dens_x, dens_y, 0, route, taus,
                                      truncation_sigmas=ts).sum())
    # The two self inner products are memoised on their densities, as the
    # flat Bulger, centres and Möbius routes already do -- a sweep against
    # one prototype, or any repeated call on the same pair, then pays for
    # the cross term alone. The key carries the route *and* the shared
    # quadrature grid, because the grid routes discretise the self inner
    # product too: a different partner can widen the grid (the
    # relative-non-periodic line spans both densities' values), and a value
    # taken under one grid must never be reused under another.
    _tau_sig = (None if taus is None
                else (int(np.size(taus)), float(taus[0]), float(taus[-1])))
    _key = _self_ip_cache_key("contract", ts, None, (route, _tau_sig))
    # <X,X> is consumed by the cosine only: under 'oneSidedDenom' it is
    # neither computed nor memoised, as on the flat routes, and the
    # finaliser receives None for it.
    need_xx = (normalize == "cosine")
    need_yy = (normalize != "none")
    if _key in dens_x._self_ip_cache:
        ip_xx = dens_x._self_ip_cache[_key]
    elif not need_xx:
        ip_xx = None
    else:
        ip_xx = float(_nested_attr_matrix(dens_x, dens_x, 0, route, taus,
                                          truncation_sigmas=ts).sum())
        dens_x._self_ip_cache[_key] = ip_xx
    if _key in dens_y._self_ip_cache:
        ip_yy = dens_y._self_ip_cache[_key]
    elif not need_yy:
        ip_yy = None
    else:
        ip_yy = float(_nested_attr_matrix(dens_y, dens_y, 0, route, taus,
                                          truncation_sigmas=ts).sum())
        dens_y._self_ip_cache[_key] = ip_yy
    return ip_xy, ip_xx, ip_yy


def _try_nested_contract_ma(dens_x, dens_y, *, normalize, verbose,
                            force=False, force_route=None,
                            method_name="contract", truncation_sigmas=None):
    """MA cosine when one or more attributes are nested or ordered.

    The MAET cross-event inner product factorises per event-pair across
    attributes (JMM Eq 3.4): ``<X,Y> = Σ_{i,j} Π_a I_a(i,j)``. Each attribute
    contributes an (N_x, N_y) per-event-pair inner matrix. A nested attribute
    goes through the per-level dispatch (:func:`_nested_attr_plan` /
    :func:`_nested_attr_matrix`): the event-pair contraction for absolute and
    absolute-periodic and for the cost-selected relative grids, the
    materialised centres otherwise -- with the route decided once so its xy, xx
    and yy share one measure, and, on a relative-periodic attribute, decided by
    the declared ``wrap`` rather than by cost wherever the two differ (see
    :func:`_nested_attr_route`). An ordered-flat attribute (``[sym]=0``, r>1, not
    nested) goes through the centres path
    (:func:`_closed_form_attr_matrix_from`): a single ordered level has no
    symmetric orbit to reduce, and routing it through the orbit/Möbius matrix
    would wrongly symmetrise it (summing its full ``S_r`` orbit). Flat-symmetric
    and r=1 attributes go through the orbit/Möbius per-attribute matrix
    (:func:`_ma_per_attr_inner_matrix`). The matrices multiply element-wise then
    sum, mirroring :func:`_cos_sim_exp_tens_ma_orbit`. Per-attribute prefactors
    are constant and cancel, so mixing the matrix conventions is exact.

    Returns the (ip_xy, ip_xx, ip_yy) triple, or ``None`` (route to the exact
    enumeration) for an uncovered nested case. ``truncation_sigmas`` is the
    per-call width, resolved here as in :func:`_try_nested_contract`.
    """
    def _decline(reason):
        if force:
            raise ValueError(
                f"method={method_name!r} is not available here: {reason}. "
                f"Use method='auto' (which falls back automatically) or "
                f"method='bulger'."
            )
        return None

    if normalize not in ("cosine", "oneSidedDenom", "none"):
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

    # ---- Pass 1: validate and plan. The per-attribute routes and shared
    # quadrature grids are settled before any matrix is formed, so the two
    # self inner products can be looked up in the densities' memo before the
    # work that would produce them is done. Splitting the pass is what makes
    # the memoisation possible at all: the MA self inner product is a product
    # across *all* attributes, so its cache key is not known until every
    # attribute has been planned.
    plans = []
    from .._defaults import resolve_truncation_sigmas
    _ts_ma = resolve_truncation_sigmas(truncation_sigmas)
    _skip_xx, _skip_yy = _nested_self_ip_skip_flags(
        dens_x, dens_y, normalize)
    for a in range(A):
        is_nested = (nested_x[a] is not None) or (nested_y[a] is not None)
        r_a = int(dens_x.r[a])
        ordered_flat = ((not is_nested) and (not bool(is_sym_x[a]))
                        and (r_a > 1))
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
            route, taus = _nested_attr_plan(dens_x, dens_y, a,
                                            force_route=force_route,
                                            skip_xx=_skip_xx,
                                            skip_yy=_skip_yy, ts=_ts_ma)
            plans.append(("nested", a, route, taus))
        elif ordered_flat:
            plans.append(("ordered", a, None, None))
        else:
            plans.append(("flat", a, None, None))
    # The plan is priced against the joint-tuple enumeration, as in the
    # one-attribute plan (_try_nested_contract): the per-attribute routes chosen above are what
    # the plan would run, and their prices plus the flat companions' are what
    # the enumeration has to beat. A forced method is never diverted.
    if not force and _nested_prefers_enumeration(
            dens_x, dens_y,
            {a: route for kind, a, route, _t in plans if kind == "nested"},
            skip_xx=_skip_xx, skip_yy=_skip_yy, ts=_ts_ma):
        _LAST_NESTED_ROUTES[:] = []
        return None
    _LAST_NESTED_ROUTES[:] = [
        (route if kind == "nested" else "-") for kind, _a, route, _t in plans]

    # The self-inner-product memo key carries every attribute's route and
    # shared grid, for the reason the one-attribute plan records: a grid
    # route discretises the self inner product too, and a different partner
    # can widen the grid, so a value taken under one grid must never be
    # reused under another. ``wrap`` is part of each density's own immutable
    # contents, so it needs no separate key entry.
    _ma_sig = tuple(
        (kind, int(a), route,
         (None if taus is None
          else (int(np.size(taus)), float(taus[0]), float(taus[-1]))))
        for kind, a, route, taus in plans)
    _ma_key = _self_ip_cache_key("contract_ma", _ts_ma, None, _ma_sig)
    _have_xx = _ma_key in dens_x._self_ip_cache
    _have_yy = _ma_key in dens_y._self_ip_cache
    # <X,X> is consumed by the cosine only: under 'oneSidedDenom' it is
    # neither formed nor memoised, as on the flat routes.
    _need_xx = (normalize == "cosine")
    _form_xx = _need_xx and not _have_xx
    _form_yy = (normalize != "none") and not _have_yy

    # ---- Pass 2: form the matrices.
    for kind, a, route, taus in plans:
        r_a = int(dens_x.r[a])

        if kind == "flat":
            # Flat-symmetric or r=1: the orbit/Möbius per-attribute matrix,
            # which correctly symmetrises these readings.
            sigma = float(dens_x.sigma[a])
            is_rel = bool(dens_x.is_rel[a])
            is_per = bool(dens_x.is_per[a])
            period = float(dens_x.period[a])
            Pxa, Pya = dens_x.p_attr[a], dens_y.p_attr[a]
            Wxa, Wya = dens_x.w[a], dens_y.w[a]
            # The attribute's declared wrap and the truncation width go
            # with it, as on every other route: without them an abs-per
            # attribute declared 'single-image' was computed full-image
            # whenever it shared a density with a nested attribute (the
            # MATLAB twin, nestedContract.m, has always passed both).
            wrap_a = _declared_wrap(dens_x, dens_y, a)
            P_xy *= _ma_per_attr_inner_matrix(
                Pxa, Wxa, Pya, Wya, sigma, r_a, is_rel, is_per, period,
                truncation_sigmas=_ts_ma, wrap=wrap_a)
            if _form_xx:
                P_xx *= _ma_per_attr_inner_matrix(
                    Pxa, Wxa, Pxa, Wxa, sigma, r_a, is_rel, is_per, period,
                    truncation_sigmas=_ts_ma, wrap=wrap_a)
            if _form_yy:
                P_yy *= _ma_per_attr_inner_matrix(
                    Pya, Wya, Pya, Wya, sigma, r_a, is_rel, is_per, period,
                    truncation_sigmas=_ts_ma, wrap=wrap_a)
            continue

        if kind == "nested":
            # Nested: the mode-aware per-level dispatch planned above
            # (contraction for absolute/abs-periodic and for the
            # cost-selected relative grids; centres otherwise), with the
            # route decided once so xy, xx and yy share one measure.
            P_xy *= _nested_attr_matrix(dens_x, dens_y, a, route, taus,
                                        truncation_sigmas=_ts_ma)
            if _form_xx:
                P_xx *= _nested_attr_matrix(dens_x, dens_x, a, route, taus,
                                            truncation_sigmas=_ts_ma)
            if _form_yy:
                P_yy *= _nested_attr_matrix(dens_y, dens_y, a, route, taus,
                                            truncation_sigmas=_ts_ma)
            continue

        # Ordered flat ([sym]=0, r>1, not nested): the materialised centres,
        # which honour the ordered reading (no symmetrisation -- the orbit
        # matrix would wrongly symmetrise) and use the minimum-image
        # pairwise-wrap for relative-periodic. A single ordered level has no
        # symmetric orbit to reduce, so there is no per-level Möbius to gain.
        cx = _closed_form_attr_centres(dens_x, a)
        cy = _closed_form_attr_centres(dens_y, a)
        wrap_a = _declared_wrap(dens_x, dens_y, a)
        P_xy *= _closed_form_attr_matrix_from(cx, cy, _ts_ma, wrap_a)
        if _form_xx:
            P_xx *= _closed_form_attr_matrix_from(cx, cx, _ts_ma, wrap_a)
        if _form_yy:
            P_yy *= _closed_form_attr_matrix_from(cy, cy, _ts_ma, wrap_a)

    # The two self inner products are memoised on their densities, as the
    # single-nested-attribute path and the flat Bulger, centres and Möbius
    # routes already do: a sweep against one prototype then pays for the
    # cross term alone.
    if _have_xx:
        ip_xx = dens_x._self_ip_cache[_ma_key]
    elif not _need_xx:
        ip_xx = None
    else:
        ip_xx = float(P_xx.sum())
        dens_x._self_ip_cache[_ma_key] = ip_xx
    if _have_yy:
        ip_yy = dens_y._self_ip_cache[_ma_key]
    elif _form_yy:
        ip_yy = float(P_yy.sum())
        dens_y._self_ip_cache[_ma_key] = ip_yy
    else:
        ip_yy = None

    # No enumeration fallback from *here*: the comparison with the
    # enumeration was made before any matrix was formed, above, where
    # declining still costs nothing. ``method='bulger'`` on a multi-attribute
    # nested density agrees with this route to floating point in every mode
    # (``tests/test_nested_measure_rule.py``), so the choice between them is
    # a matter of price and of per-attribute measure control, not of shape.
    return float(P_xy.sum()), ip_xx, ip_yy


#: Cache-key prefixes of the routes that memoise a self inner product
#: consumed by an inner-product triple. A memo under any of them means
#: "some route has already paid for this density's self inner product",
#: which is what :func:`_self_ip_memoised` reports and what both sides
#: of every route comparison are priced against. The sweep path's own
#: ``'sweep'`` memo is deliberately absent: it is produced by a different
#: evaluator and is consumed by neither route here, so it would not spare
#: either of them any work.
_SELF_IP_ROUTES = ("bulger", "centres", "mobius", "contract", "contract_ma")


def _self_ip_memoised(dens):
    """True when any inner-product route has memoised ``dens``'s self IP.

    The flag is shared by the routes a selector compares, rather than read
    off each route's own memo, and that is deliberate. The memoised
    *values* are per route (see :func:`_self_ip_cache_key`), so a route
    that finds only another route's memo will still recompute its own self
    matrices on this call. Pricing each route against its own memo
    nonetheless makes the comparison unfair in a way that compounds: the
    first call seeds only the winner's memo, so on the second call the
    winner is priced at one matrix and the loser at three, and the choice
    locks in even where the loser, once warm, is the cheaper route. Sharing
    the flag prices the comparison on the routes' per-matrix costs, which
    is what the selector is meant to decide on.

    The trade is per-call: on the one call where the comparison flips, the
    newly chosen route does pay for the self matrices the flag priced as
    free. It memoises them, so the flag is honest from the next call
    onwards; the mispricing is bounded by a single call per crossover, and
    it buys amortised correctness over the repeated calls a sweep makes.
    """
    return any(isinstance(k, tuple) and len(k) > 0 and k[0] in _SELF_IP_ROUTES
               for k in dens._self_ip_cache)


def _self_ip_cache_key(route, truncation_sigmas, kernel_precision=None,
                       extra=None):
    """Cache key for a memoised self inner product on a density.

    The key carries everything the value depends on beyond the
    density's own (immutable) contents: the route, the resolved
    truncation budget, the kernel precision, and any route-specific
    choices (``extra`` — the Möbius route's per-attribute
    closed-form-vs-grid selections, or a nested route's quadrature grid
    signature). Chunking granularity is deliberately not keyed: it
    perturbs only the floating-point accumulation order, within the
    toolbox-wide ≤ 1e-12 parity discipline.

    Why the route stays in the key, when the routes' scales are related
    in closed form. The bare triples differ by a constant that cancels
    within one route's triple, and the constant is known exactly:
    per attribute, Bulger's enumeration against the Möbius per-attribute
    matrix is ``r_a! (sigma_a sqrt(pi))^{r_a}`` in an absolute mode and
    ``r_a! (sigma_a sqrt(pi))^{r_a - 1} sqrt(r_a)`` in relative
    non-periodic; against the tuple-centres closed form (and the
    unrestricted centres route) it is ``r_a!``; against a nested
    attribute's per-level contraction it is 1, and against the nested
    centres route the wreath-product orbit order of
    :func:`~mpt._tensor._mobius_inner._nested_orbit_mult`. Converting a
    memo to a canonical scale is therefore arithmetically possible.

    It is not *numerically* possible. Each route applies the truncation
    budget to its own arrays --- a different threshold tightening for a
    different array size, a different set of kernel entries dropped, and
    for the Möbius route an alternating sum where the enumeration has a
    plain one --- so after the exact rescaling the routes hold different
    numbers, not the same number in different units. Measured on the
    self inner product with the shipped 6-sigma default: the Möbius
    route departs from Bulger's by up to 3e-9 relative (absolute
    non-periodic, r = 2..3, K = 6..9), the centres route by up to 1e-12,
    and at ``truncation_sigmas=4`` those become 9e-5 and 2e-8; at the
    accuracy floor (``inf``) they fall to 1e-13 and 1e-16. A shared memo
    would put that difference into the returned cosine whenever a route
    consumed a value another route produced, so two identical calls with
    the same forced ``method`` would return different numbers depending
    on what ran before them. Route-keyed values keep each route's answer
    reproducible; the *pricing* is shared instead, via
    :func:`_self_ip_memoised`.

    The one case where sharing is not even arithmetically available is
    worth naming separately, because it is a difference of measure
    rather than of accuracy: on a relative-periodic attribute the
    tau-grid computes the all-image transposition average (C) while the
    enumeration and the tuple-centres closed form compute the
    minimum-image reading (A). Below the sigma/period threshold they
    agree inside the truncation floor but are still not the same number
    (measured 1.9e-5 relative at sigma/P = 0.058, 4.8e-2 at 0.125), and
    above it they are different quantities.
    """
    from .._defaults import resolve_truncation_sigmas
    return (route, float(resolve_truncation_sigmas(truncation_sigmas)),
            kernel_precision, extra)


def _cos_sim_exp_tens_ma_centres(dens_x, dens_y, *, verbose: bool = True,
                                 truncation_sigmas=None,
                                 kernel_precision=None, need_xx: bool = True,
                                 need_yy: bool = True):
    """Inner-product triple by unrestricted enumeration of tuple centres.

    The O(K^(2r)) baseline: every ordered r-tuple of distinct atoms on
    each side against every such tuple on the other. It differs from
    Bulger's route in exactly one respect -- the permutation side is used
    on *both* sides, where Bulger uses the permutation side against the
    combination side and multiplies by r!. Everything else is shared:
    the same ``_ip_core_ma``, so the same truncation, kernel precision,
    wrap convention, chunking and log-space accumulation.

    Routing through the shared core is what makes the two comparable.
    An independent re-implementation would measure its own constants
    rather than the algorithms', and would silently ignore settings the
    core honours; the independent enumeration is kept in the test suite,
    where sharing no code with the core is the point.
    """
    from .._defaults import _maybe_show_dispatch_msg

    _maybe_show_dispatch_msg(
        "cos_sim_exp_tens", "centres",
        "unrestricted enumeration of tuple centres (reference route)",
    )

    A = int(dens_x.n_attrs)
    r_vec = np.atleast_1d(dens_x.r)
    sigma = np.atleast_1d(dens_x.sigma)
    is_rel = np.atleast_1d(dens_x.is_rel)
    is_per = np.atleast_1d(dens_x.is_per)
    period = np.atleast_1d(dens_x.period)
    inner_r = _inner_r_vec(dens_x)
    n_jx, n_jy = dens_x.n_j, dens_y.n_j

    def core(dx, nx, dy, ny):
        return _ip_core_ma(
            dx.u_perm, dx.w_j, nx,
            dy.u_perm, dy.w_j, ny,
            A, r_vec, sigma, is_rel, is_per, period,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
            inner_r=inner_r,
            wrap=getattr(dx, 'wrap', None),
        )

    # Memoise the self terms under a route-specific key, exactly as the
    # pairwise arm does. Without this the route recomputes <X,X> and
    # <Y,Y> on every call while Bulger's arm reuses them, so a repeated
    # comparison would time three products against one and the ratio
    # between the routes would not be r!.
    key = _self_ip_cache_key("centres", truncation_sigmas, kernel_precision)
    ip_xy = core(dens_x, n_jx, dens_y, n_jy)
    if need_xx:
        if key in dens_x._self_ip_cache:
            ip_xx = dens_x._self_ip_cache[key]
        else:
            ip_xx = core(dens_x, n_jx, dens_x, n_jx)
            dens_x._self_ip_cache[key] = ip_xx
    else:
        ip_xx = None
    if key in dens_y._self_ip_cache:
        ip_yy = dens_y._self_ip_cache[key]
    elif need_yy:
        ip_yy = core(dens_y, n_jy, dens_y, n_jy)
        dens_y._self_ip_cache[key] = ip_yy
    else:
        ip_yy = None
    return ip_xy, ip_xx, ip_yy


def _cos_sim_exp_tens_ma_pairwise(dens_x, dens_y, *, verbose: bool = True,
                                  truncation_sigmas=None,
                                  kernel_precision=None,
                                  need_xx: bool = True,
                                  need_yy: bool = True):
    """Compute (ip_xy, ip_xx, ip_yy) for the MA case via the
    Bulger's method (``_ip_core_ma``).

    This is the body of the original ``_cos_sim_exp_tens_ma``
    factored out so the new dispatcher can route to it cleanly.

    ``need_xx=False`` skips <X,X> when it is neither memoised nor
    consumed by the caller's normalisation (``'oneSidedDenom'``); the
    triple's first self slot is then ``None``. Both self inner products
    are memoised on their densities (``_self_ip_cache``), so a sweep of
    many queries against one context pays each self term once.
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

    key = _self_ip_cache_key("bulger", truncation_sigmas, kernel_precision)
    xx_cached = key in dens_x._self_ip_cache
    yy_cached = key in dens_y._self_ip_cache
    compute_xx = need_xx and not xx_cached
    compute_yy = need_yy and not yy_cached

    # The estimate covers only the kernel work this call performs:
    # memoised self terms cost nothing here, and a skipped <X,X>
    # (``need_xx=False`` under 'oneSidedDenom') is never evaluated.
    total_pairs = n_jx * n_ky
    if compute_xx:
        total_pairs += n_jx * n_kx
    if compute_yy:
        total_pairs += n_jy * n_ky
    max_r = int(np.max(r_vec)) if A > 0 else 1
    estimate_comp_time(total_pairs, max_r, "cos_sim_exp_tens (MAET)", verbose)

    ip_xy = _ip_core_ma(
        dens_x.u_perm, dens_x.w_j, n_jx,
        dens_y.v_comb, dens_y.wv_comb, n_ky,
        A, r_vec, sigma, is_rel, is_per, period,
        truncation_sigmas=truncation_sigmas,
        kernel_precision=kernel_precision,
        inner_r=inner_r,
        wrap=getattr(dens_x, 'wrap', None),
    )
    if xx_cached:
        ip_xx = dens_x._self_ip_cache[key]
    elif need_xx:
        ip_xx = _ip_core_ma(
            dens_x.u_perm, dens_x.w_j, n_jx,
            dens_x.v_comb, dens_x.wv_comb, n_kx,
            A, r_vec, sigma, is_rel, is_per, period,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
            inner_r=inner_r,
            wrap=getattr(dens_x, 'wrap', None),
        )
        dens_x._self_ip_cache[key] = ip_xx
    else:
        ip_xx = None
    if yy_cached:
        ip_yy = dens_y._self_ip_cache[key]
    elif compute_yy:
        ip_yy = _ip_core_ma(
            dens_y.u_perm, dens_y.w_j, n_jy,
            dens_y.v_comb, dens_y.wv_comb, n_ky,
            A, r_vec, sigma, is_rel, is_per, period,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
            inner_r=inner_r,
            wrap=getattr(dens_y, 'wrap', None),
        )
        dens_y._self_ip_cache[key] = ip_yy
    else:
        ip_yy = None
    return ip_xy, ip_xx, ip_yy



# -------------------------------------------------------------------
#  cos_sim_exp_tens_raw  (dispatches single-multiset or multi-attribute based on input shape)
# -------------------------------------------------------------------


def cos_sim_exp_tens_raw(
    p1, w1, p2, w2, *args,
    method: str = "auto",
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
        verbose=verbose,
    )



def _ip_via_helper(U, wU, V, wV, r, sigma, is_rel, is_per, period,
                   truncation_sigmas=None, kernel_precision=None,
                   wrap_a='full-image'):
    """Route the centres-IP through :func:`gaussian_kernel_sum`.

    The helper computes ``g(q) = sum_j wJ(j) * exp(-Q(c_j - x_q) /
    (2 * sigma_eff^2))`` with ``sigma_eff = sigma * sqrt(2)``, so the
    kernel exponent matches the centres-IP's ``Q / (4 * sigma^2)``.
    The IP is then ``wU @ g``.

    Supports abs (per and non-per) and rel-non-periodic. The rel+per
    pairwise-wrap form is not yet supported by the helper.
    """
    kw = dict(is_rel=bool(is_rel), r=int(r),
              is_per=bool(is_per), period=float(period),
              wrap=str(wrap_a))
    if truncation_sigmas is not None:
        kw["truncation_sigmas"] = float(truncation_sigmas)
    if kernel_precision is not None:
        kw["kernel_precision"] = kernel_precision
    sigma_eff = float(sigma) * np.sqrt(2.0)
    # The kernel sum is reduced to a single inner product below, so the
    # truncation floor has to bound the summed discarded mass over all
    # centre-query pairs rather than each pair individually.
    kw["n_terms"] = int(np.asarray(U).shape[-1]) * int(np.asarray(V).shape[-1])
    g = gaussian_kernel_sum(V, wV.ravel(), U, sigma_eff, **kw)
    return float(np.asarray(g).ravel() @ wU.ravel())


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
    in turn threads ``method`` through to the per-pair dispatcher.

    Parameters
    ----------
    p_mat_a, p_mat_b : 2-D arrays
        Multiset positions per row. NaN entries are ignored.
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
    method : {'auto', 'bulger', 'centres'}, default 'auto'
        Inner-product evaluation path; threaded through to the per-pair
        single-multiset core via the inner ``cos_sim_exp_tens`` call.
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

def _build_pre_maet_args(args, *, verbose=True):
    """Replace any whole pre-MAET among the positional arguments.

    A pre-MAET is a density in waiting: it holds everything build_exp_tens
    needs, so it stands wherever a density does and is built here. That
    goes for a *list* of them too: a list of pre-MAETs stands wherever a
    list of densities does, so the list and scalar-vs-list forms take
    them without the caller building each one first. A translation sweep
    (:class:`~mpt._tensor.preprocessing.TranslatedSweep`) carried inside
    a pre-MAET is one such list, sharing one geometry, and is expanded
    the same way, its offsets carried through so the mixture reduction
    still applies.

    The loose triple has no such form, since the three parts are not
    distinguishable from the surrounding positional geometry.

    ``verbose`` governs these builds too, so a quiet call stays quiet.
    """
    from .premaet import is_pre_maet
    from .preprocessing import TranslatedSweep

    def _is_sweep_pm(a):
        return is_pre_maet(a) and isinstance(a.get("p_attr"), TranslatedSweep)

    def _listish(a):
        return isinstance(a, (list, tuple)) and any(is_pre_maet(x) for x in a)

    if not any(is_pre_maet(a) or _listish(a) for a in args):
        return args
    from .build import build_exp_tens

    def _one(a):
        if _is_sweep_pm(a):
            # One geometry, one density per sweep entry; the offsets ride
            # along so cos_sim_exp_tens can still reduce the sweep to a
            # mixture in the offset.
            sweep = a["p_attr"]
            built = [build_exp_tens({"p_attr": list(block),
                                     "w_attr": a.get("w_attr"),
                                     "specs": a.get("specs")},
                                    verbose=verbose)
                     for block in sweep]
            return TranslatedSweep(built,
                                   sweep_offsets=sweep.sweep_offsets,
                                   sweep_base=sweep.sweep_base)
        if is_pre_maet(a):
            return build_exp_tens(a, verbose=verbose)
        if _listish(a):
            return [build_exp_tens(x, verbose=verbose) if is_pre_maet(x) else x
                    for x in a]
        return a

    return tuple(_one(a) for a in args)
