"""Cross-event preprocessing and categorical-encoding utilities.

This module hosts the small preprocessing layer that sits *before*
``build_exp_tens`` in the MAET pipeline:

* :func:`difference_events` --- replace event sequences with their
  k-th finite differences along the event axis.
* :func:`bind_events` --- slide a length-n window across an event
  sequence, emitting each window as an n-attribute super-event.
* :func:`translate_attributes` --- shift every value of every attribute
  in selected groups by a per-group offset (rigid translation).

It also exposes :func:`simplex_vertices`, the categorical-encoding
helper for level-symmetric MAET inputs.

See USER_GUIDE §3.1 ("Cross-event preprocessing") for the conceptual
introduction and :doc:`/ARCHITECTURE` §2 for the layering.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.special import erf as _erf

from .._utils import validate_weights


class TranslateAttributesNoOpWarning(UserWarning):
    """Emitted when ``translate_attributes`` is asked to translate a
    relative-mode group, which is a structural no-op (the relative
    MAET depends only on within-tuple differences, so a uniform shift
    of all values cancels in every pairwise difference). The group is
    left unchanged."""



# ===================================================================
#  difference_events
# ===================================================================


def difference_events(p_attr, w, diff_orders, *, circular=False, specs=None):
    """Replace selected attributes' event sequences with inter-event differences.

    Cross-event preprocessing on the canonical ``(p_attr, w, specs)``
    specifications. The ``k_a``-th finite difference is applied along the event
    axis to each attribute; the returned ``(p_attr_diff, w_diff, specs)``
    chains into another pre-MAET operation or into ``build_exp_tens(...,
    specs=...)``.

    Differencing pairs values **index by index**: event *i*'s value at index
    *k* differences against
    event *i+1*'s value at index *k*. This is well-defined exactly when the
    indices
    have stable identity --- an ordered attribute (``[sym] = 0``) or a
    singleton (``K = 1``). A symmetric multiset (``K > 1``, ``[sym] = 1``)
    is a bag with no index correspondence, so differencing it is undefined
    and raises. The rule extends per level for a nested attribute: every
    level must be ordered (or of size 1). Ragged ordered data (events of
    differing length) is represented by NaN-padding to a common ``K``; a
    difference touching a NaN value is NaN, so absence propagates rather
    than fabricating an interval.

    Differencing changes **values only**; the spec (``tags``, ``r``,
    ``sym``, ``rel``) passes through unchanged. Output values are raw;
    periodic wrapping, when desired, is the kernel's job in
    :func:`build_exp_tens`.

    Event-axis alignment. With ``circular = False`` (default), an attribute
    of order ``k_a`` yields ``N - k_a`` events and all attributes are
    brought onto ``N' = N - max_a k_a`` by dropping leading events; with
    ``circular = True`` the index wraps and every attribute keeps ``N``.
    Per-attribute weights propagate as a rolling product over the
    ``k_a + 1`` constituent events.

    Parameters
    ----------
    p_attr : list/tuple of array-like
        Length-A list of ``(K_a, N)`` per-attribute value matrices.
        ``K_a >= 1``; ``K_a = 0`` is rejected. ``K_a > 1`` is differenced
        index by index when the attribute is ordered (see above).
    w : None, scalar, or length-A list
        Weights (``build_exp_tens`` convention).
    diff_orders : scalar or length-A array-like
        Per-attribute differencing orders (non-negative integers; a scalar
        broadcasts to all attributes).
    circular : bool, keyword-only
        Wrap the difference at the event-sequence boundary (``N' = N``).
    specs : None or length-A list, keyword-only
        The attribute specifications. ``None`` synthesises flat specs
        (:func:`flat_specs` defaults). The ordered-or-singleton guard reads
        ``[sym]`` from here, and the specs pass through to the output
        unchanged.

    Returns
    -------
    p_attr_diff : list of ndarray
        Length-A list of differenced matrices, each ``(K_a, N')``.
    w_diff : same general form as *w*
    specs : list of dict
        The attribute specifications, unchanged from the input (or synthesised).

    See Also
    --------
    build_exp_tens, bind_events, flat_specs, translate_attributes
    """
    if not isinstance(p_attr, (list, tuple)):
        raise TypeError(
            "p_attr must be a list/tuple of per-attribute matrices."
        )
    p_attr = [np.asarray(M, dtype=np.float64) for M in p_attr]
    for a, M in enumerate(p_attr):
        if M.ndim != 2:
            raise ValueError(
                f"Attribute {a} value matrix must be 2-D; got ndim={M.ndim}."
            )
    A = len(p_attr)
    if A == 0:
        raise ValueError("p_attr must contain at least one attribute.")
    for a, M in enumerate(p_attr):
        if M.shape[0] == 0:
            raise ValueError(
                f"Attribute {a} has K_a = 0 (empty attribute); empty "
                f"attributes are not permitted."
            )
    n_events = p_attr[0].shape[1]
    for a, M in enumerate(p_attr):
        if M.shape[1] != n_events:
            raise ValueError(
                f"All attributes must share the same event count N. "
                f"Attribute 0 has N={n_events}; attribute {a} has "
                f"N={M.shape[1]}."
            )
    K = [M.shape[0] for M in p_attr]

    # --- Attribute specifications: synthesise flat if none supplied --------------
    if specs is None:
        specs_out = flat_specs(p_attr)
    else:
        if not isinstance(specs, (list, tuple)) or len(specs) != A:
            raise ValueError(
                f"specs must be a length-A ({A}) list, one per attribute."
            )
        specs_out = list(specs)

    orders = _canonicalise_diff_orders(diff_orders, A)

    # --- Ordered-or-singleton guard for each differenced attribute ----
    for a in range(A):
        if int(orders[a]) > 0:
            _check_differenceable(specs_out[a], K[a], a)

    max_order = int(orders.max()) if A > 0 else 0
    if circular:
        n_prime = n_events
        if max_order >= n_events:
            raise ValueError(
                f"Differencing order too high for circular mode: max order "
                f"= {max_order} but N = {n_events} (need max order < N)."
            )
    else:
        n_prime = n_events - max_order
        if n_prime < 1:
            raise ValueError(
                f"Differencing orders are too high for the input event "
                f"count: max order = {max_order} but N = {n_events}."
            )

    # --- Difference each position matrix (row by row; NaN propagates) -----
    p_attr_diff = []
    for a, M in enumerate(p_attr):
        k = int(orders[a])
        M_diff = M
        if circular:
            for _ in range(k):
                M_diff = M_diff - np.roll(M_diff, 1, axis=1)
        else:
            for _ in range(k):
                M_diff = M_diff[:, 1:] - M_diff[:, :-1]
            extra_drop = max_order - k
            if extra_drop > 0:
                M_diff = M_diff[:, extra_drop:]
        assert M_diff.shape[1] == n_prime
        p_attr_diff.append(M_diff)

    w_diff = _difference_weights(
        w, A, orders, n_events, n_prime, circular,
    )
    return p_attr_diff, w_diff, specs_out


def _check_differenceable(spec, K_a, a):
    """Guard: an attribute is differenceable only if its positions have stable
    identity across events --- ordered (``[sym] = 0``) or singleton at
    every level. A symmetric multiset of size > 1 is a bag with no positional
    correspondence, so differencing it is undefined.
    """
    if isinstance(spec, dict) and "tags" in spec:
        tags = np.asarray(spec["tags"]).ravel()
        sym = spec.get("sym")
        sym = (np.ones(2, dtype=bool) if sym is None
               else np.asarray(sym, dtype=bool).ravel())
        n_groups = int(np.unique(tags).size)            # outer level size
        inner_sz = int(tags.size // max(n_groups, 1))   # values per group
        inner_ok = (not bool(sym[0])) or inner_sz == 1
        outer_ok = (not bool(sym[-1])) or n_groups == 1
        if not (inner_ok and outer_ok):
            bad = "inner" if not inner_ok else "outer"
            raise ValueError(
                f"attribute {a}: differencing requires every level ordered "
                f"(or of size 1); the {bad} level is symmetric with size > "
                f"1. Set that level's [sym] = 0 to difference it."
            )
    else:
        sym = bool(spec.get("sym", True)) if isinstance(spec, dict) else True
        if sym and K_a > 1:
            raise ValueError(
                f"attribute {a}: differencing requires an ordered attribute "
                f"([sym] = 0) or K = 1; got a symmetric multiset with K = "
                f"{K_a}. A symmetric multiset is a bag with no positional "
                f"correspondence across events. Set [sym] = 0 (e.g. via "
                f"flat_specs(..., sym=False)) to difference it."
            )


def _canonicalise_diff_orders(diff_orders, A):
    """Coerce ``diff_orders`` to a length-A int array (scalar or per-attr)."""
    arr = np.asarray(diff_orders)
    if arr.dtype.kind not in "iuf":
        raise TypeError(
            f"diff_orders must be numeric; got dtype={arr.dtype}."
        )
    if arr.ndim == 0:
        result = np.full(A, float(arr), dtype=np.float64)
    elif arr.ndim == 1:
        if arr.size == 1:
            result = np.full(A, float(arr[0]), dtype=np.float64)
        elif arr.size == A:
            result = arr.astype(np.float64, copy=True)
        else:
            raise ValueError(
                f"diff_orders has {arr.size} entries; expected scalar (1) "
                f"or per-attribute (A = {A})."
            )
    else:
        raise ValueError(
            f"diff_orders must be a scalar or 1-D array; got ndim = "
            f"{arr.ndim}."
        )
    _validate_orders(result)
    return result.astype(np.int64, copy=False)
def _validate_orders(orders):
    if np.any(orders < 0):
        raise ValueError(
            "All entries of diff_orders must be non-negative."
        )
    if not np.all(orders == np.round(orders)):
        raise ValueError(
            "All entries of diff_orders must be integers."
        )


def _difference_weights(w, A, orders_per_attr, n_events, n_prime, circular):
    """Transform weights under per-attribute differencing orders.

    Each differenced attribute's weights are propagated via a rolling
    product of width ``k_a + 1``. In non-circular mode, pass-through
    attributes (``k_a = 0``) have their leading events dropped to
    match the common output grid. In circular mode the rolling
    product wraps at the event-sequence boundary and pass-through
    attributes are kept at length ``N``.
    """
    if w is None:
        return None

    # --- Top-level scalar ---
    if np.isscalar(w):
        c = float(w)
        if np.all(orders_per_attr == orders_per_attr[0]):
            # Uniform orders — shape preserved as a scalar.
            return c ** int(orders_per_attr[0] + 1)
        # Varying orders — emit a per-attribute list.
        return [c ** int(k + 1) for k in orders_per_attr]

    if not isinstance(w, (list, tuple)):
        raise TypeError(
            "w must be None, a scalar, or a list/tuple of per-attribute "
            "weight inputs."
        )
    if len(w) != A:
        raise ValueError(
            f"Weight list must have length A = {A}; got length {len(w)}."
        )

    max_order = int(orders_per_attr.max()) if A > 0 else 0
    w_diff = []
    for a, wa in enumerate(w):
        k = int(orders_per_attr[a])
        if not _weight_has_event_dependence(wa, n_events, a):
            # No event dependence — rolling product of a constant
            # reduces to raising each entry to power k + 1.
            w_diff.append(_raise_no_event_dep(wa, k + 1))
            continue
        # Event-dependent: (1, N) row, length-N 1-D, or (K_a, N) matrix.
        W = np.asarray(wa, dtype=np.float64)
        if W.ndim == 1:
            W = W.reshape(1, n_events)
        if k > 0:
            W = _rolling_product(W, k + 1, circular)
        if not circular:
            extra_drop = max_order - k
            if extra_drop > 0:
                W = W[:, extra_drop:]
        assert W.shape[1] == n_prime
        w_diff.append(W)
    return w_diff


def _raise_no_event_dep(wa, p: int):
    """Raise a non-event-dependent weight input to power *p*.

    Non-event-dependent inputs reaching this helper are ``None``,
    Python scalars, 0-D arrays, size-1 1-D arrays, or ``(K_a, 1)``
    column broadcasts.
    """
    if wa is None:
        return None
    if p == 1:
        return wa  # fast path: order 0 attribute
    if np.isscalar(wa):
        return float(wa) ** int(p)
    arr = np.asarray(wa)
    if arr.size == 1:
        return float(arr.item()) ** int(p)
    # K_a x 1 column broadcast.
    return arr.astype(np.float64) ** int(p)


def _rolling_product(W, width, circular=False):
    """Rolling product of length-N row windows of width *width*.

    With ``circular=False``: output shape ``(K, N - width + 1)``;
    output column ``i`` is ``prod(W[:, i : i + width], axis=1)``.

    With ``circular=True``: output shape ``(K, N)``; output column
    ``n`` is the product of ``W`` over the ``width`` indices
    ``(n, n-1, n-2, ..., n-width+1)`` taken modulo ``N``. This matches
    the cyclic differencing operator's ``prev(n) = (n - 1) mod N``
    convention, so the weight attached to the ``k``-th cyclic
    difference at output position ``n`` is the product of
    ``w(n), w(prev(n)), ..., w(prev^{width-1}(n))``.
    """
    K, N = W.shape
    if circular:
        if width > N:
            raise ValueError(
                f"Circular rolling-product width {width} exceeds event "
                f"count {N}."
            )
        out = np.empty((K, N), dtype=np.float64)
        for n in range(N):
            idx = (n - np.arange(width)) % N
            out[:, n] = np.prod(W[:, idx], axis=1)
        return out
    n_out = N - width + 1
    if n_out < 1:
        raise ValueError(
            f"Rolling-product width {width} exceeds event count {N}."
        )
    out = np.empty((K, n_out), dtype=np.float64)
    for i in range(n_out):
        out[:, i] = np.prod(W[:, i:i + width], axis=1)
    return out


def _weight_has_event_dependence(wa, N, attr_idx=None):
    """True iff *wa*'s shape carries the N axis.

    Accepts ``None``, scalar, ``(K_a, 1)`` column broadcast, ``(1, N)``
    row, length-N 1-D, or ``(K_a, N)`` matrix.
    """
    if wa is None:
        return False
    arr = np.asarray(wa)
    if arr.size == 1:
        return False
    if arr.ndim == 1 and arr.size == N:
        return True
    if arr.ndim == 2:
        if arr.shape[1] == N:
            # (1, N) row or (K_a, N) matrix.
            return True
        if arr.shape[1] == 1:
            # (K_a, 1) column broadcast — no event dependence.
            return False
    where = (
        f"Attribute {attr_idx} weight" if attr_idx is not None else "Weight"
    )
    raise ValueError(
        f"{where} has shape {arr.shape}; expected None, scalar, "
        f"(K_a, 1) column, (1, {N}) row, length-{N} 1-D, or "
        f"(K_a, {N}) matrix."
    )


# ===================================================================
#  bind_events
# ===================================================================


def _bcast_geom(x, A, name, *, cast):
    """Broadcast a scalar or length-A geometry argument to a length-A list."""
    if np.isscalar(x) or isinstance(x, (bool, np.bool_, int, np.integer, float)):
        return [cast(x)] * A
    arr = list(x)
    if len(arr) == 1:
        return [cast(arr[0])] * A
    if len(arr) != A:
        raise ValueError(
            f"{name} must be a scalar or length-A ({A}); got {len(arr)}."
        )
    return [cast(v) for v in arr]


def _bcast_names(name, A):
    if name is None:
        return [None] * A
    if isinstance(name, str):
        return [name] * A
    names = list(name)
    if len(names) != A:
        raise ValueError(f"name must be None, a string, or length-A ({A}).")
    return names


def flat_specs(p_attr, *, r=1, rel=False, sym=True, name=None):
    """Build a list of flat (one-level) specs for bare attributes.

    Convenience constructor for the canonical attribute specifications: wraps a list
    of per-attribute value matrices in flat spec dicts ``{r, rel, sym,
    name?}``, broadcasting scalar geometry across attributes. This is the
    trivial flat-specs synthesis at the entry of a pre-MAET chain (raw
    attributes carry no level structure yet) and an ergonomic alternative
    to hand-writing flat dicts for ``build_exp_tens(..., specs=...)``.

    Parameters
    ----------
    p_attr : list/tuple of array-like
        Length-A list of per-attribute value matrices (used only for its
        length A; values are not inspected).
    r : int or length-A, keyword-only
        Per-attribute tuple size (default 1).
    rel, sym : bool or length-A, keyword-only
        Per-attribute ``[rel]`` / ``[sym]`` (defaults ``False`` / ``True``).
    name : None, str, or length-A, keyword-only
        Optional per-attribute names.

    Returns
    -------
    list of dict
        Length-A list of flat specs, ready for ``build_exp_tens(specs=...)``
        or to thread through the pre-MAET operators.
    """
    if not isinstance(p_attr, (list, tuple)):
        raise TypeError(
            "p_attr must be a list/tuple of per-attribute matrices."
        )
    A = len(p_attr)
    r_v = _bcast_geom(r, A, "r", cast=int)
    rel_v = _bcast_geom(rel, A, "rel", cast=bool)
    sym_v = _bcast_geom(sym, A, "sym", cast=bool)
    name_v = _bcast_names(name, A)
    specs = []
    for a in range(A):
        s = {"r": r_v[a], "rel": rel_v[a], "sym": sym_v[a]}
        if name_v[a] is not None:
            s["name"] = name_v[a]
        specs.append(s)
    return specs


def bind_events(
    p_attr,
    w,
    bind_orders,
    *,
    circular: bool = False,
    step: int = 1,
    specs=None,
    r_outer=None,
    sym_outer=False,
    rel_outer=False,
    name=None,
    level_names=None,
    group_by=None,
    group_atol: float = 0.0,
) -> tuple[list[np.ndarray], object, list]:
    """Bind sliding windows of consecutive events into nested attributes.

    Cross-event preprocessing on the canonical ``(p_attr, w, specs)``
    specifications. For each input attribute *a*, a sliding window of width
    ``L_a`` (``bind_orders``) is laid across the event axis and the
    ``L_a`` consecutive events are nested into a single output attribute
    (toolbox spec §6.1): the bound events form an **ordered outer level**
    (event order; ``sym_outer = 0`` by default, lossless), and each
    event's own atom multiset is the **inner level**.

    The inner level's geometry (``r``/``rel``/``sym``) is read from the
    incoming ``specs`` --- the attribute's existing specification supplies
    the inner level(s). ``specs = None`` synthesises flat specs
    (:func:`flat_specs` defaults: ``r = 1``, ``rel = 0``, ``sym = 1``).
    The outer level defaults to ``r = L_a`` (read the whole bound
    window), ``sym = 0``, ``rel = 0``. ``L_a = 1`` is the no-op: the
    incoming (flat) spec passes through unchanged.

    With the defaults and ``rel = [rel_in, 0]``, the outer ``r = L_a``
    reading is the tensor product of the events' inner densities --- it
    reproduces the old separate-attribute binding (§6.5). The genuinely
    new lever is ``rel_outer = 1`` on an absolute attribute, giving the
    global-transposition quotient ``rel = [0, 1]``.

    Event-axis alignment. At the default ``step = 1`` the common
    output event count is ``N' = N - max_a L_a + 1`` (non-circular) or
    ``N`` (circular); attributes with ``L_a < max_a L_a`` keep their
    leading ``N'`` windows, so D-then-B equals B-then-D with
    :func:`difference_events`. A ``step > 1`` hops the windows (see the
    ``step`` parameter), shrinking ``N'``; the difference-composition
    identity then holds at ``step = 1`` only.

    Parameters
    ----------
    p_attr : list/tuple of array-like
        Length-A list of ``(K_a, N)`` per-attribute value matrices.
    w : None, scalar, or length-A list
        Weights (same convention as :func:`build_exp_tens`). Each bound
        attribute's value weights are the windowed-and-stacked input
        weights, so the kernel product over the nested tuple recovers
        the rolling product.
    bind_orders : scalar or length-A array-like
        Per-attribute window widths ``L_a >= 1`` (``L_a = 1`` no-op).
    circular : bool, keyword-only
        Wrap the window around the event axis (``N' = N``).
    step : int, keyword-only
        Hop between consecutive bound windows along the event axis
        (default ``1``, the fully overlapping slide). Super-event ``i``
        reads events ``[i*step, i*step + L_a)``, so ``step = L_a``
        gives non-overlapping blocks (e.g. eighths into beats). A single
        scalar applies to all attributes: the hop is a property of the
        shared event axis, not per-attribute. ``N' = (N - max_a L_a) //
        step + 1`` (non-circular); for ``circular = True`` the event
        count ``N`` must be divisible by ``step`` and ``N' = N //
        step``. The bind/difference composition identity holds at
        ``step = 1`` only.
    specs : None or length-A list, keyword-only
        The attribute specifications supplying the inner geometry. ``None``
        synthesises flat specs. An incoming spec may be flat or already
        nested: a flat spec becomes the inner level of a new two-level
        attribute, while an already-nested spec is deepened --- a new
        outermost level (``r_outer``/``sym_outer``/``rel_outer``) is
        appended above the existing nesting, and ``tags``, ``r``,
        ``sym``, and ``rel`` each extend by one entry. Repeated binds
        nest to arbitrary depth, but each call must be given the
        ``specs`` returned by the previous one: passing ``None`` (or
        omitting ``specs``) on already-bound attributes re-synthesises
        flat specs, silently discarding the existing nesting and
        producing a shallower result.
    r_outer : None, scalar, or length-A, keyword-only
        Outer-level ``r`` (how many bound events to read). ``None``
        defaults to ``L_a`` (the whole window).
    sym_outer, rel_outer : bool / scalar / length-A, keyword-only
        Outer-level ``[sym]`` and ``[rel]``. Default ``0``/``0``.
    name : None, str, or length-A, keyword-only
        Optional per-attribute name(s). Overrides any ``name`` carried
        on the incoming spec; otherwise the incoming name is preserved.
    level_names : None or length-2 list, keyword-only
        Optional ``[inner, outer]`` level names, stamped onto each
        nested spec's ``names`` field. Applies only when the incoming
        spec is flat (the flat-to-two-level case); supplying it while
        deepening an already-nested attribute is rejected, since the
        per-level names carry through from the incoming spec.

    Returns
    -------
    p_attr_bound : list of ndarray
        Length-A list. For ``L_a >= 2`` a stacked ``(L_a * K_a, N')``
        value matrix (the ``L_a`` lag windows vertically stacked); for
        ``L_a = 1`` the leading-aligned ``(K_a, N')`` original.
    w_bound : same general form as *w*
        Transformed weights aligned to the value layout.
    specs : list of dict
        Length-A. A nested spec ``{tags, r, sym, rel, name?, names?}``
        for ``L_a >= 2``; the incoming spec unchanged (flat or nested)
        for ``L_a = 1``.

    See Also
    --------
    build_exp_tens, difference_events, flat_specs, translate_attributes
    """
    if not isinstance(p_attr, (list, tuple)):
        raise TypeError(
            "p_attr must be a list/tuple of per-attribute matrices."
        )
    p_attr = [np.asarray(M, dtype=np.float64) for M in p_attr]
    A = len(p_attr)
    if A == 0:
        raise ValueError("p_attr must contain at least one attribute.")
    for a, M in enumerate(p_attr):
        if M.ndim != 2:
            raise ValueError(
                f"Attribute {a} value matrix must be 2-D; got ndim={M.ndim}."
            )
        if M.shape[0] == 0:
            raise ValueError(
                f"Attribute {a} has K_a = 0 (empty attribute); empty "
                f"attributes are not permitted."
            )
    n_events = p_attr[0].shape[1]
    for a, M in enumerate(p_attr):
        if M.shape[1] != n_events:
            raise ValueError(
                f"All attributes must share the same event count N. "
                f"Attribute 0 has N={n_events}; attribute {a} has "
                f"N={M.shape[1]}."
            )
    K = [M.shape[0] for M in p_attr]

    if group_by is not None:
        return _bind_events_run_length(
            p_attr, w, K, A, n_events, group_by, group_atol, specs,
            r_outer, sym_outer, rel_outer, name, level_names,
            bind_orders, circular, step,
        )

    orders = _canonicalise_bind_orders(bind_orders, A)

    step_arr = np.asarray(step)
    if step_arr.ndim != 0:
        raise ValueError(
            "step must be a scalar; a single hop applies to all "
            "attributes (the hop is a property of the shared event axis)."
        )
    if step_arr.dtype.kind not in "iuf" or float(step_arr) != int(step_arr):
        raise TypeError("step must be an integer.")
    step = int(step_arr)
    if step < 1:
        raise ValueError(
            "step must be >= 1 (1 is the fully overlapping slide)."
        )

    # --- Inner geometry from the attribute specifications ---------------
    if specs is None:
        specs_in = flat_specs(p_attr)
    else:
        if not isinstance(specs, (list, tuple)) or len(specs) != A:
            raise ValueError(
                f"specs must be a length-A ({A}) list, one per attribute."
            )
        specs_in = list(specs)
    for a, s in enumerate(specs_in):
        if (isinstance(s, dict) and "tags" in s and level_names is not None):
            raise ValueError(
                f"attribute {a}: level_names is not supported when deepening "
                f"an already-nested attribute; the per-level names carry "
                f"through from the incoming spec."
            )

    # --- Outer-level overrides ----------------------------------------
    if r_outer is None:
        r_out = [int(orders[a]) for a in range(A)]
    else:
        r_out = _bcast_geom(r_outer, A, "r_outer", cast=int)
    sym_out = _bcast_geom(sym_outer, A, "sym_outer", cast=bool)
    rel_out = _bcast_geom(rel_outer, A, "rel_outer", cast=bool)
    names_attr = _bcast_names(name, A)
    if level_names is not None and len(level_names) != 2:
        raise ValueError(
            "level_names must be a length-2 [inner, outer] list (two-level "
            "binding)."
        )

    max_order = int(orders.max()) if A > 0 else 0
    if circular:
        if step > 1 and n_events % step != 0:
            raise ValueError(
                f"Circular binding with step = {step} requires the "
                f"event count N = {n_events} to be divisible by step."
            )
        n_prime = n_events // step
        if max_order > n_events:
            raise ValueError(
                f"Circular window size max L = {max_order} exceeds event "
                f"count N = {n_events}."
            )
    else:
        n_prime = (n_events - max_order) // step + 1
        if n_prime < 1:
            raise ValueError(
                f"Bind orders too high for the input event count: max L = "
                f"{max_order} but N = {n_events} (non-circular)."
            )

    def _lag_index(ell):
        if circular:
            return (np.arange(n_prime) * step + ell) % n_events
        return np.arange(n_prime) * step + ell

    p_attr_bound = []
    specs_out = []
    for a in range(A):
        L_a = int(orders[a])
        K_a = K[a]                       # flat value count K_total
        M = p_attr[a]
        s_in = specs_in[a] if isinstance(specs_in[a], dict) else {}
        is_nested_in = "tags" in s_in
        name_in_a = s_in.get("name")
        nm = names_attr[a] if names_attr[a] is not None else name_in_a
        if L_a == 1:
            # No-op: the incoming spec passes through (name override).
            # Works for both flat and already-nested inputs.
            p_attr_bound.append(M[:, _lag_index(0)])
            spec = dict(s_in)
            if nm is not None:
                spec["name"] = nm
            specs_out.append(spec)
            continue

        # Lag and stack the (super-)event matrix over the new outer level.
        blocks = [M[:, _lag_index(ell)] for ell in range(L_a)]
        p_attr_bound.append(np.vstack(blocks))
        new_col = np.repeat(np.arange(L_a, dtype=np.intp), K_a)

        if is_nested_in:
            # Deepen: append a new outermost grouping level above the
            # existing nesting. The existing tag columns are tiled once per
            # bound super-event; the new column distinguishes the L_a bound
            # super-events. r/sym/rel extend by the new outer level.
            tags_in = np.asarray(s_in["tags"])
            if tags_in.ndim == 1:
                tags_in = tags_in.reshape(-1, 1)
            tags = np.column_stack([np.tile(tags_in, (L_a, 1)), new_col])
            spec = {
                "tags": tags,
                "r": [int(x) for x in np.asarray(s_in["r"]).ravel()]
                     + [int(r_out[a])],
                "sym": [bool(x) for x in np.asarray(s_in["sym"]).ravel()]
                       + [bool(sym_out[a])],
                "rel": [int(x) for x in np.asarray(s_in["rel"]).ravel()]
                       + [int(rel_out[a])],
            }
            if "names" in s_in and s_in["names"] is not None:
                spec["names"] = list(s_in["names"]) + [None]
        else:
            # Flat input -> two-level nested attribute (unchanged).
            r_in_a = int(s_in.get("r", 1))
            rel_in_a = bool(s_in.get("rel", False))
            sym_in_a = bool(s_in.get("sym", True))
            spec = {
                "tags": new_col,
                "r": [r_in_a, int(r_out[a])],
                "sym": [sym_in_a, bool(sym_out[a])],
                "rel": [int(rel_in_a), int(rel_out[a])],
            }
            if level_names is not None:
                spec["names"] = list(level_names)
        if nm is not None:
            spec["name"] = nm
        specs_out.append(spec)

    w_bound = _bind_weights_nested(
        w, A, orders, K, n_events, n_prime, circular, step,
    )
    return p_attr_bound, w_bound, specs_out



def _run_length_groups(vals, atol):
    """Consecutive-run boundaries on a 1-D value array.

    Returns a list of 1-D index arrays, one per maximal run of (near-)equal
    adjacent values. With ``atol == 0`` the comparison is exact; otherwise a
    new run starts where ``|v[i] - v[i-1]| > atol``.
    """
    n = vals.size
    if n == 0:
        return []
    if atol == 0.0:
        changes = vals[1:] != vals[:-1]
    else:
        changes = np.abs(vals[1:] - vals[:-1]) > atol
    starts = np.concatenate(([0], np.nonzero(changes)[0] + 1, [n]))
    return [np.arange(starts[i], starts[i + 1], dtype=np.intp)
            for i in range(starts.size - 1)]


def _bind_events_run_length(p_attr, w, K, A, n_events, group_by, group_atol,
                            specs, r_outer, sym_outer, rel_outer, name,
                            level_names, bind_orders, circular, step):
    """Run-length (bind-by-attribute) binding.

    Consecutive events sharing a constant value on attribute ``group_by``
    are gathered into one super-event; a new group begins wherever that
    value changes. Group sizes vary, so the outer level is ragged: each
    super-event is padded to the maximum group size with NaN values carrying
    zero weight (the padded-value-at-zero-weight convention the nested inner
    product already consumes). The inner level preserves each attribute's
    existing per-attribute parameters; the outer tuple size ``r_outer``
    defaults to the smallest group size --- the largest tuple size at which every
    group is feasible, so all groups contribute uniform ``r_outer``-tuples
    into one density.
    """
    if bind_orders is not None:
        raise ValueError(
            "bind_orders and group_by are mutually exclusive: run-length "
            "binding reads the group sizes from the data, so pass "
            "bind_orders=None when group_by is given."
        )
    if circular:
        raise NotImplementedError(
            "Circular run-length binding is not yet supported; pass "
            "circular=False."
        )
    if step != 1:
        raise ValueError(
            "step has no meaning for run-length binding (groups are read "
            "from the data, not hopped); leave step at its default."
        )
    if not isinstance(group_by, (int, np.integer)) or not (0 <= group_by < A):
        raise ValueError(
            f"group_by must be an attribute index in [0, {A}); got {group_by}."
        )
    if K[group_by] != 1:
        raise ValueError(
            f"group_by attribute {group_by} must have K = 1 (one value per "
            f"event); got K = {K[group_by]}. Constancy across multiple values "
            f"is ambiguous."
        )

    if specs is None:
        specs_in = flat_specs(p_attr)
    else:
        if not isinstance(specs, (list, tuple)) or len(specs) != A:
            raise ValueError(
                f"specs must be a length-A ({A}) list, one per attribute."
            )
        specs_in = list(specs)
    for a, s in enumerate(specs_in):
        if isinstance(s, dict) and "tags" in s:
            raise NotImplementedError(
                f"attribute {a}: run-length binding of an already-nested "
                f"attribute is not yet supported (flat inputs only)."
            )

    groups = _run_length_groups(p_attr[group_by][0], float(group_atol))
    n_prime = len(groups)
    if n_prime == 0:
        raise ValueError("group_by produced no groups (empty event axis).")
    sizes = np.array([g.size for g in groups], dtype=np.intp)
    L_max = int(sizes.max())
    L_min = int(sizes.min())

    if r_outer is None:
        r_out = [L_min for _ in range(A)]
    else:
        r_out = _bcast_geom(r_outer, A, "r_outer", cast=int)
    sym_out = _bcast_geom(sym_outer, A, "sym_outer", cast=bool)
    rel_out = _bcast_geom(rel_outer, A, "rel_outer", cast=bool)
    names_attr = _bcast_names(name, A)
    if level_names is not None and len(level_names) != 2:
        raise ValueError(
            "level_names must be a length-2 [inner, outer] list (two-level "
            "binding)."
        )

    # Per outer position, the source event index for each group and a validity
    # mask (False where the group is shorter than the position).
    src = np.zeros((L_max, n_prime), dtype=np.intp)
    valid = np.zeros((L_max, n_prime), dtype=bool)
    for j, g in enumerate(groups):
        src[:g.size, j] = g
        valid[:g.size, j] = True

    p_attr_bound = []
    w_bound = []
    specs_out = []
    for a in range(A):
        M = p_attr[a]
        Wa = (None if w is None
              else (w[a] if isinstance(w, (list, tuple)) else w))
        K_a = K[a]
        blocks = []
        wblocks = []
        for ell in range(L_max):
            cols = M[:, src[ell]]                       # (K_a, n_prime)
            blocks.append(np.where(valid[ell][None, :], cols, np.nan))
            if Wa is None:
                wblocks.append(np.where(valid[ell][None, :], 1.0, 0.0))
            else:
                wblocks.append(np.where(valid[ell][None, :],
                                        Wa[:, src[ell]], 0.0))
        p_attr_bound.append(np.vstack(blocks))          # (L_max*K_a, n_prime)
        w_bound.append(np.vstack(wblocks))

        new_col = np.repeat(np.arange(L_max, dtype=np.intp), K_a)
        s_in = specs_in[a] if isinstance(specs_in[a], dict) else {}
        spec = {
            "tags": new_col,
            "r": [int(s_in.get("r", 1)), int(r_out[a])],
            "sym": [bool(s_in.get("sym", True)), bool(sym_out[a])],
            "rel": [int(bool(s_in.get("rel", False))), int(rel_out[a])],
        }
        if level_names is not None:
            spec["names"] = list(level_names)
        nm = names_attr[a] if names_attr[a] is not None else s_in.get("name")
        if nm is not None:
            spec["name"] = nm
        specs_out.append(spec)

    return p_attr_bound, w_bound, specs_out


def _canonicalise_bind_orders(bind_orders, A):
    """Coerce ``bind_orders`` to a length-A int array (scalar or per-attr)."""
    arr = np.asarray(bind_orders)
    if arr.dtype.kind not in "iuf":
        raise TypeError(
            f"bind_orders must be numeric; got dtype={arr.dtype}."
        )
    if arr.ndim == 0:
        result = np.full(A, float(arr), dtype=np.float64)
    elif arr.ndim == 1:
        if arr.size == 1:
            result = np.full(A, float(arr[0]), dtype=np.float64)
        elif arr.size == A:
            result = arr.astype(np.float64, copy=True)
        else:
            raise ValueError(
                f"bind_orders has {arr.size} entries; expected scalar (1) "
                f"or per-attribute (A = {A})."
            )
    else:
        raise ValueError(
            f"bind_orders must be a scalar or 1-D array; got ndim = "
            f"{arr.ndim}."
        )
    _validate_bind_orders(result)
    return result.astype(np.int64, copy=False)


def _validate_bind_orders(orders):
    if np.any(orders < 1):
        raise ValueError(
            "All entries of bind_orders must be positive integers "
            "(>= 1; L = 1 is the no-op)."
        )
    if not np.all(orders == np.round(orders)):
        raise ValueError(
            "All entries of bind_orders must be integers."
        )


def _bind_weights_nested(w, A, orders, K, n_events, n_prime, circular, step=1):
    """Transform weights to match the nested value layout.

    For ``L_a >= 2`` the per-event weight slices are windowed and
    stacked into a ``(L_a * K_a, N')`` column aligned with the value
    stack (per-event weights are expanded across the ``K_a`` values of
    their event); for ``L_a = 1`` the weight is trailing-aligned.
    Non-event-dependent inputs (``None``, scalar, ``(K_a, 1)`` column)
    are inherited / tiled across the bound values.
    """
    if w is None:
        return None
    if np.isscalar(w):
        return float(w)
    if not isinstance(w, (list, tuple)):
        raise TypeError(
            "w must be None, a scalar, or a list/tuple of per-attribute "
            "weight inputs."
        )
    if len(w) != A:
        raise ValueError(
            f"Weight list must have length A = {A}; got length {len(w)}."
        )

    def _lag_index(ell):
        if circular:
            return (np.arange(n_prime) * step + ell) % n_events
        return np.arange(n_prime) * step + ell

    w_bound = []
    for a, wa in enumerate(w):
        L_a = int(orders[a])
        K_a = K[a]
        event_dep = _weight_has_event_dependence(wa, n_events, a)
        if not event_dep:
            if wa is None or np.isscalar(wa):
                w_bound.append(wa)
            else:
                W = np.asarray(wa, dtype=np.float64)   # (K_a, 1) per-value
                w_bound.append(W if L_a == 1 else np.tile(W, (L_a, 1)))
            continue
        # Event-dependent: materialise to (K_a, N), window per lag, stack.
        W = np.asarray(wa, dtype=np.float64)
        if W.ndim == 1:
            W = W.reshape(1, -1)
        if W.shape[0] == 1 and K_a > 1:
            W = np.tile(W, (K_a, 1))
        if L_a == 1:
            w_bound.append(W[:, _lag_index(0)])
        else:
            blocks = [W[:, _lag_index(ell)] for ell in range(L_a)]
            w_bound.append(np.vstack(blocks))
    return w_bound


# ===================================================================
#  weight_events
# ===================================================================


def _scalarize(x, name, *, dtype):
    """Accept Python scalar or 0-D / length-1 numpy array. Returns a Python
    scalar of the requested dtype. Used by weight_events for its scalar
    keyword arguments."""
    if isinstance(x, str):
        raise TypeError(f"{name} must be a scalar number; got string.")
    arr = np.asarray(x)
    if arr.ndim > 1 or arr.size != 1:
        raise ValueError(
            f"{name} must be a scalar; got shape {arr.shape}."
        )
    val = arr.item()
    if dtype is bool:
        return bool(val)
    return dtype(val)


def weight_events(
    p_attr,
    w,
    input_attr,
    target_attr,
    centre,
    shape,
    *,
    specs=None,
    sd=None,
    width=None,
    is_per=False,
    period=0.0,
    drop_input_attr,
) -> tuple:
    r"""Apply a per-event weight via an input-to-target window factor.

    Per-event preprocessing for multi-attribute tensor input. Reads the
    K=1 value at every event from ``input_attr``, evaluates a window
    function :math:`h` centred at ``centre`` with shape parameter
    ``shape`` (:math:`= \gamma`), and writes the resulting
    :math:`(1, N)` per-event factor into the weight entry of
    ``target_attr``, multiplied into any existing weight already there.
    ``target_attr`` may differ from ``input_attr`` (the typical case
    --- e.g., time-driven windowing of pitch events) or coincide with
    it (the input attribute weights itself).

    The window size is specified through exactly one of two
    keyword-only arguments, ``sd`` or ``width``. Both name the same
    underlying scale on different terms:

    - ``sd`` is the **standard deviation** of the window. ``sd = 1.0``
      gives a Gaussian of standard deviation 1 at ``shape = 0`` and a
      rectangle whose standard deviation is 1 (i.e., full support
      :math:`2\sqrt 3`) at ``shape = 1``.
    - ``width`` is the **full support of the rectangle** at
      ``shape = 1``. ``width = 1.0`` gives a rectangle on
      :math:`[-1/2, +1/2]` at ``shape = 1`` and a Gaussian of standard
      deviation :math:`1/(2\sqrt 3)` at ``shape = 0``. The conversion
      is ``sd = width / (2 sqrt(3))``.

    The two conventions exist because each is the natural way to
    specify the *kind* of kernel a particular analysis is built
    around: Gaussian users typically think in standard deviations,
    rectangle users typically think in full supports. Across the full
    ``shape`` family the SD is held constant regardless of which
    parameter the caller supplied (variance-normalised behaviour), so
    the only effect of the parameter choice is the numerical value
    the user types.

    When ``drop_input_attr=True`` and ``input_attr`` differs from
    ``target_attr``, the input attribute is removed from the returned
    ``p_attr_out`` / ``w_out`` / ``specs_out`` after the factor has
    been transferred to the target. This is the canonical windowed-
    entropy / windowed-mass workflow: the input attribute provides the
    scaffolding for the window and is no longer needed downstream.
    When ``drop_input_attr=False``, the input attribute is preserved
    unchanged in the output. ``drop_input_attr=True`` paired with
    ``input_attr == target_attr`` is rejected as incoherent (deleting
    the input would discard the factor just written to it).

    The window family is the peak-normalised convolution of a rectangle
    and a Gaussian. Internally, in terms of the standard deviation
    :math:`s` (= ``sd`` directly, or ``width / (2 sqrt(3))``):

    .. math::

        \phi = s\sqrt{3\gamma}, \qquad
        \xi  = s\sqrt{1-\gamma},

    parameterised so the total variance equals :math:`s^2` for every
    :math:`\gamma \in [0, 1]`. Limits:

    - :math:`\gamma = 0`: pure Gaussian
      :math:`h(\delta) = \exp(-\delta^2 / (2 s^2))`.
    - :math:`\gamma = 1`: pure rectangle
      :math:`h(\delta) = \mathbb{1}[|\delta| \le s\sqrt 3]`,
      i.e., total support :math:`2 s\sqrt 3` (equivalently
      ``= width`` when the caller supplied ``width``).

    The window is peak-normalised so :math:`h(0) = 1`. For a periodic
    input group (``is_per=True``), the difference
    :math:`\delta = v - \text{centre}` is wrapped to
    :math:`[-P/2, P/2]` before applying :math:`h`; the stored values
    in ``p_attr`` are not modified.

    The per-event factor is broadcast across the target attribute's
    ``K_target`` values, so every value of every event sees the same
    factor.

    Factor entries whose distance from the centre exceeds the global
    ``truncation_sigmas`` cutoff (i.e., :math:`|\delta| > \text{
    truncation\_sigmas} \cdot s`, where :math:`s` is the kernel's
    standard deviation, equal to ``sd`` or ``width / (2 sqrt(3))``)
    are hard-zeroed. The threshold is the same one the IP / evaluation
    kernels use: at that distance a Gaussian window's value is
    :math:`\exp(-\text{truncation\_sigmas}^2 / 2)`. The default global
    value is ``6``; set ``mpt.set_default(truncation_sigmas=math.inf)``
    for no truncation, or another ``k`` to change the hard-truncation
    distance at :math:`k\,s`.

    Parameters
    ----------
    p_attr : list/tuple of array-like
        Length-``A`` list of ``(K_a, N)`` per-attribute value matrices.
        ``K_a >= 1``.
    w : None, scalar, or list/tuple
        Existing weights. ``None``, scalar, or length-``A`` list of
        per-attribute weights (each ``None``, scalar, 1-D row, or
        ``(K_a, N)`` matrix). Same convention as :func:`build_exp_tens`.
    specs : None or length-A list of dict, keyword-only
        Attribute specifications (per-attribute level geometry). ``None``
        synthesises flat specs via :func:`flat_specs`. Threaded through
        unchanged, except that ``drop_input_attr=True`` drops the input
        attribute's entry. ``weight_events`` does not otherwise consult
        the specs; the window is computed from the input attribute's
        values, ``centre``, ``shape``, ``sd``/``width``, and (for a
        periodic input) ``is_per``/``period``.
    input_attr : int
        Index of the attribute supplying the window's values. Must
        satisfy ``0 <= input_attr < A`` and the attribute's
        ``K_input == 1`` (single value per event).
    target_attr : int
        Index of the attribute receiving the window factor in its
        weight entry. Must satisfy ``0 <= target_attr < A``. May equal
        ``input_attr``. ``K_target`` may be any positive integer; the
        factor broadcasts across values.
    centre : float
        Window centre, in the input attribute's units.
    shape : float
        Shape parameter :math:`\gamma \in [0, 1]`. ``0`` is pure
        Gaussian; ``1`` is pure rectangle; intermediate values
        interpolate via the fixed-variance convolution family.
    is_per : bool, keyword-only, default False
        Whether the input attribute is periodic. If ``True``,
        :math:`\delta = v - \text{centre}` is wrapped to
        :math:`[-P/2, P/2]` before evaluating :math:`h`.
    period : float, keyword-only, default 0.0
        Period of the input attribute. Used only when
        ``is_per=True`` (must then be ``> 0``); ignored otherwise.
    sd : float, keyword-only
        Window standard deviation, ``> 0``, in the input attribute's
        units. Exactly one of ``sd`` or ``width`` must be supplied.
    width : float, keyword-only
        Full support of the rectangle at ``shape = 1``, ``> 0``, in
        the input attribute's units. Internally translated to a
        standard deviation as ``sd = width / (2 sqrt(3))``. Exactly
        one of ``sd`` or ``width`` must be supplied.
    drop_input_attr : bool, keyword-only, REQUIRED
        Whether to remove the input attribute from the output. If
        ``True`` and ``input_attr != target_attr``, drops the input
        attribute's value matrix, weight, and spec from the returned
        triple. ``drop_input_attr=True`` paired with
        ``input_attr == target_attr`` raises ``ValueError``. There is
        no default; callers must specify explicitly.

    Returns
    -------
    p_attr_out : list of (K_a, N) ndarrays
        Per-attribute value matrices. Length ``A`` if
        ``drop_input_attr=False``, else ``A - 1``.
    w_out : list
        Per-attribute weights, length matching ``p_attr_out``. The
        entry at ``target_attr`` (in the output indexing) carries the
        windowed weights.
    specs_out : list of dict
        The attribute specifications for the output attribute list. Same as the
        input specs (synthesised flat if ``specs`` was ``None``), with
        the input attribute's entry removed when ``drop_input_attr=True``.

    See Also
    --------
    build_exp_tens, difference_events, bind_events, translate_attributes
    """
    # --- Normalise p_attr ---
    if not isinstance(p_attr, (list, tuple)):
        raise TypeError(
            "p_attr must be a list/tuple of per-attribute matrices."
        )
    p_attr = [np.asarray(M, dtype=np.float64) for M in p_attr]
    A = len(p_attr)
    if A == 0:
        raise ValueError("p_attr must contain at least one attribute.")
    for a, M in enumerate(p_attr):
        if M.ndim != 2:
            raise ValueError(
                f"Attribute {a} value matrix must be 2-D; got ndim={M.ndim}."
            )
        if M.shape[0] == 0:
            raise ValueError(
                f"Attribute {a} has K_a = 0 (empty attribute); empty "
                f"attributes are not permitted."
            )

    # --- Shared N ---
    n_events = p_attr[0].shape[1]
    for a, M in enumerate(p_attr):
        if M.shape[1] != n_events:
            raise ValueError(
                f"All attributes must share the same event count N. "
                f"Attribute 0 has N={n_events}; attribute {a} has "
                f"N={M.shape[1]}."
            )

    # --- Attribute specifications: synthesise flat if absent, else validate length ---
    if specs is None:
        specs_in = flat_specs(p_attr)
    else:
        if not isinstance(specs, (list, tuple)) or len(specs) != A:
            raise ValueError(
                f"specs must be a length-A ({A}) list, one per attribute."
            )
        specs_in = list(specs)

    # --- Validate input_attr ---
    if isinstance(input_attr, (bool, np.bool_)):
        raise TypeError("input_attr must be an integer index, not bool.")
    try:
        input_attr_int = int(input_attr)
    except (TypeError, ValueError):
        raise TypeError(
            f"input_attr must be an integer index; got "
            f"{type(input_attr).__name__}."
        )
    if input_attr_int != input_attr:
        raise ValueError(
            f"input_attr must be an integer index; got {input_attr}."
        )
    if not (0 <= input_attr_int < A):
        raise ValueError(
            f"input_attr must be in 0..{A - 1}; got {input_attr_int}."
        )
    if p_attr[input_attr_int].shape[0] != 1:
        raise ValueError(
            f"input_attr {input_attr_int} has K = "
            f"{p_attr[input_attr_int].shape[0]}; weight_events requires the "
            f"input attribute to have K = 1 (single value per event)."
        )

    # --- Validate target_attr ---
    if isinstance(target_attr, (bool, np.bool_)):
        raise TypeError("target_attr must be an integer index, not bool.")
    try:
        target_attr_int = int(target_attr)
    except (TypeError, ValueError):
        raise TypeError(
            f"target_attr must be an integer index; got "
            f"{type(target_attr).__name__}."
        )
    if target_attr_int != target_attr:
        raise ValueError(
            f"target_attr must be an integer index; got {target_attr}."
        )
    if not (0 <= target_attr_int < A):
        raise ValueError(
            f"target_attr must be in 0..{A - 1}; got {target_attr_int}."
        )

    # --- Validate drop_input_attr (required, no default) ---
    if not isinstance(drop_input_attr, (bool, np.bool_)):
        raise TypeError(
            "drop_input_attr must be a bool; the keyword is required and has "
            "no default."
        )
    drop_input_attr = bool(drop_input_attr)

    if drop_input_attr and input_attr_int == target_attr_int:
        raise ValueError(
            f"drop_input_attr=True is incoherent when input_attr == "
            f"target_attr (={input_attr_int}): deleting the input would "
            f"discard the weight factor just written to it. Set "
            f"drop_input_attr=False, or choose a different target_attr."
        )

    # --- Validate centre, sd/width (XOR), shape, is_per, period (scalars) ---
    centre_f = _scalarize(centre, "centre", dtype=float)
    # Exactly one of sd or width must be supplied. Convert width →
    # sd internally; the rest of the body operates on sd_f.
    if (sd is None) == (width is None):
        raise TypeError(
            "weight_events requires exactly one of `sd` or `width` "
            "(keyword-only). `sd` is the window standard deviation; "
            "`width` is the full support of the rectangle at "
            "shape=1, equivalent to sd * 2 * sqrt(3). Got "
            f"sd={sd!r}, width={width!r}."
        )
    if sd is not None:
        sd_f = _scalarize(sd, "sd", dtype=float)
        if not np.isfinite(sd_f) or sd_f <= 0:
            raise ValueError(f"sd must be finite and > 0; got {sd_f}.")
    else:
        width_f = _scalarize(width, "width", dtype=float)
        if not np.isfinite(width_f) or width_f <= 0:
            raise ValueError(f"width must be finite and > 0; got {width_f}.")
        sd_f = width_f / (2.0 * np.sqrt(3.0))
    shape_f = _scalarize(shape, "shape", dtype=float)
    is_per_b = _scalarize(is_per, "is_per", dtype=bool)
    period_f = _scalarize(period, "period", dtype=float)

    if not np.isfinite(centre_f):
        raise ValueError(f"centre must be finite; got {centre_f}.")
    if not (0.0 <= shape_f <= 1.0):
        raise ValueError(
            f"shape (gamma) must lie in [0, 1]: gamma = 0 is pure Gaussian, "
            f"gamma = 1 is pure rectangle, intermediate values are the "
            f"fixed-variance convolution family. Got {shape_f}."
        )
    if is_per_b and period_f <= 0:
        raise ValueError(
            f"period must be > 0 when is_per is True; got {period_f}."
        )

    # --- Compute factor h(delta) from input attribute values ---
    val_row = p_attr[input_attr_int].astype(np.float64, copy=False)  # (1, N)
    delta = val_row - centre_f
    if is_per_b:
        delta = delta - period_f * np.floor(delta / period_f + 0.5)
    factor = _evaluate_shape(delta, sd_f, shape_f)  # (1, N)

    # Truncate: zero factor entries whose distance exceeds
    # truncation_sigmas · sd. Uniform convention with the kernel
    # truncation in the IP/eval paths: at that distance a Gaussian
    # window's value is exp(-truncation_sigmas² / 2), the same
    # threshold the kernel truncation uses. Reads the global default
    # so changes via mpt.set_default(truncation_sigmas=...) propagate
    # without an extra kwarg. Per the truncation contract, math.inf
    # resolves to the finite accuracy-floor width (the 1e-12 floor),
    # so truncation always applies --- never a "disabled" state.
    from .._defaults import get_default, resolve_truncation_sigmas
    trunc_sig = resolve_truncation_sigmas(get_default('truncation_sigmas'))
    factor[np.abs(delta) > trunc_sig * sd_f] = 0.0

    # --- Normalise w to length-A list; multiply factor into target entry ---
    w_out = _normalise_weights_to_list(w, A)
    w_out[target_attr_int] = _multiply_weights(
        w_out[target_attr_int], factor, target_attr_int,
    )

    # --- Build output structures, applying drop_input_attr if requested ---
    if drop_input_attr:
        keep = [a for a in range(A) if a != input_attr_int]
        p_attr_out = [p_attr[a] for a in keep]
        w_out_kept = [w_out[a] for a in keep]
        specs_out = [specs_in[a] for a in keep]
    else:
        p_attr_out = list(p_attr)
        w_out_kept = list(w_out)
        specs_out = list(specs_in)

    return p_attr_out, w_out_kept, specs_out


def _evaluate_shape(delta, width, gamma):
    r"""Peak-normalised fixed-variance window family (Eqs.~5.2.1 of MAET).

    Three parameters: window centre ``c`` (already absorbed into
    ``delta = v - c``), the window standard deviation ``width``
    (= :math:`\lambda\sigma` in the manuscript's notation), and a
    shape parameter ``gamma`` :math:`\in [0, 1]` interpolating between
    pure Gaussian (``gamma=0``) and pure rectangle (``gamma=1``).

    The window is the convolution :math:`\mathrm{rect}_\phi *
    \mathcal{G}_\xi` with derived parameters

    .. math::

        \phi = \text{width}\cdot\sqrt{3\gamma}, \qquad
        \xi  = \text{width}\cdot\sqrt{1-\gamma}.

    These constrain the total variance to ``width**2`` across the
    entire family (rectangle on :math:`[-\phi, \phi]` has variance
    :math:`\phi^2/3 = \text{width}^2\,\gamma`; Gaussian has variance
    :math:`\xi^2 = \text{width}^2\,(1-\gamma)`; the two variances sum
    to :math:`\text{width}^2`). The window is peak-normalised so
    ``h(0) = 1`` throughout.

    Limits (computed directly):

    - ``gamma = 0``: pure Gaussian
      :math:`h(\delta) = \exp(-\delta^2 / (2\,\text{width}^2))`.
    - ``gamma = 1``: pure rectangle
      :math:`h(\delta) = \mathbb{1}[|\delta| \le \text{width}\sqrt{3}]`.
    """
    if not (0.0 <= gamma <= 1.0):
        raise ValueError(
            f"shape entries (gamma) must lie in [0, 1]; got {gamma}."
        )
    if width <= 0.0:
        raise ValueError(
            f"width entries must be > 0; got {width}."
        )
    if gamma == 0.0:
        return np.exp(-(delta ** 2) / (2.0 * width ** 2))
    if gamma == 1.0:
        phi = width * np.sqrt(3.0)
        # Half-open support [-phi, phi): lower edge included, upper edge
        # excluded. A closed interval over-counts even widths (each window
        # spans an odd number of pulses, so widths 1..5 collapse to pulse
        # counts 1, 3, 3, 5, 5) and, at a between-pulse centre, can drop
        # both flanking edge pulses, leaving a zero-mass density. The
        # half-open rule gives exactly N pulses for full support N*IOI at
        # every N. The tolerance keeps the edge test robust to floating-
        # point error, so a pulse landing exactly on an edge cannot flip
        # membership.
        delta = np.asarray(delta, dtype=np.float64)
        scale = max(abs(phi), 1.0)
        if delta.size:
            scale = max(scale, float(np.max(np.abs(delta))))
        tol = 1e-9 * scale
        return ((delta >= -phi - tol) & (delta < phi - tol)).astype(np.float64)
    phi = width * np.sqrt(3.0 * gamma)
    xi = width * np.sqrt(1.0 - gamma)
    scale = xi * np.sqrt(2.0)
    num = _erf((delta + phi) / scale) - _erf((delta - phi) / scale)
    peak = 2.0 * _erf(phi / scale)
    return num / peak



class TranslatedSweep(list):
    """A translation sweep, carrying the offsets that produced it.

    A plain ``list`` of length-*A* value-lists --- exactly what
    :func:`translate_attributes` has always returned in sweep mode ---
    with the generating offsets attached. Every existing consumer sees a
    list and is unaffected; :func:`~mpt.cos_sim_exp_tens` reads the
    attached offsets and, where they describe a uniform per-attribute
    translation, evaluates the sweep as a mixture in the offset rather
    than one inner product per entry.

    The offsets are carried rather than recovered. Recovering them from
    the translated values would mean comparing floating-point
    differences against a tolerance, and no tolerance both admits every
    honestly translated sweep and preserves the toolbox's parity floor;
    reading them from the call that produced them has neither problem.

    Attributes
    ----------
    sweep_offsets : ndarray
        ``(A, M)`` array of per-attribute uniform translations, with
        ``NaN`` in any (attribute, sweep index) cell whose offset was
        not uniform across the attribute's positions.
    sweep_base : list of ndarray
        The length-*A* untranslated value matrices.
    """

    __slots__ = ("sweep_offsets", "sweep_base")

    def __init__(self, entries, *, sweep_offsets, sweep_base):
        super().__init__(entries)
        self.sweep_offsets = sweep_offsets
        self.sweep_base = sweep_base


def _normalise_weights_to_list(w, A):
    """Coerce ``w`` to a length-A list, preserving entries."""
    if w is None:
        return [None] * A
    if np.isscalar(w):
        return [float(w)] * A
    if isinstance(w, (list, tuple)):
        if len(w) != A:
            raise ValueError(
                f"Weight list must have length A = {A}; got length {len(w)}."
            )
        return list(w)
    raise TypeError(
        "w must be None, a scalar, or a length-A list of per-attribute "
        "weight inputs."
    )


def _multiply_weights(w_existing, factor, attr_idx):
    """Multiply existing per-attribute weight by ``factor`` ((K_a, N)).

    Broadcasting follows the toolbox convention (scalar / (1, N) row /
    (K_a, 1) column / (K_a, N) matrix all multiply naturally into the
    (K_a, N) factor).
    """
    if w_existing is None:
        return factor
    if np.isscalar(w_existing):
        return float(w_existing) * factor
    arr = np.asarray(w_existing, dtype=np.float64)
    # numpy broadcasting handles the per-value / per-event cases.
    return arr * factor


# ===================================================================
#  translate_attributes
# ===================================================================


def translate_attributes(p_attr, w, offsets, *, specs=None):
    """Translate attributes' positions by per-row offsets.

    Per-attribute preprocessing on the ``(p_attr, w, specs)`` triple.
    Selected attributes' positions are shifted by a chosen offset and the
    transformed triple feeds straight into :func:`build_exp_tens` (or a
    further pre-MAET step). Weights and specs pass through unchanged;
    only the positions move.

    **Value-axis alignment (read this first).** Everything hangs off one
    axis: the **value axis** of an attribute, whose length is ``K_total``
    (the number of leaf values in one event/super-event). In the value
    matrix the value axis is the **rows** (``K_total x N``: values down,
    events across). The spec's ``tags`` label that same axis
    (one entry per row). An offset is likewise per-value: one offset per
    row, held **constant across the sequence (column) axis** --- that
    constancy is what makes ``D(T(p)) == D(p)``. A scalar broadcasts to
    every value (a global transposition).

    Offsets are supplied as a **length-A list**, one entry per attribute,
    each entry one of:

    - ``None`` --- do not translate this attribute.
    - scalar or 1-D length 1 --- broadcast to all ``K_total`` values.
    - 1-D length ``K_total`` --- per-value (typed as a plain vector; it is
      aligned to the rows internally, no transpose needed).
    - 2-D ``(1, M)`` --- a per-sweep global shift: one scalar per sweep
      index, broadcast across values.
    - 2-D ``(K_total, M)`` --- per-value by sweep index: values down, sweep
      index across (the only meaningful 2-D layout; the second axis is an
      enumeration of the ``M`` candidate offsets, unrelated to events).

    ``NaN`` entries skip the corresponding value (left untranslated);
    ``+/-inf`` is rejected. All 2-D entries must agree on ``M`` (scalar,
    1-D, and single-column entries broadcast across the call's ``M``).

    **Sweep.** When any entry implies ``M > 1`` the call is a batched
    sweep: it returns ``M`` translated copies --- a length-``M`` list of
    length-``A`` value-lists --- each a separate pre-MAET input to build
    and compare (the canonical sliding-transposition cosine use), sharing
    one ``w`` and one ``specs``. With ``M = 1`` it returns a single
    length-``A`` value-list.

    **Relative attributes.** ``is_rel`` is read per-attribute from
    ``specs`` (no separate argument). A *uniform* shift cancels in every
    within-tuple difference, so on an attribute whose **outermost level
    is relative** a uniform finite offset is a structural no-op: that
    column is left unchanged and a single
    :class:`TranslateAttributesNoOpWarning` is emitted per call. A
    *non-uniform* (per-value) offset is **not** a no-op even on a relative
    attribute --- it shifts the within-tuple differences --- so it
    applies. ``is_per``/``period`` are not consulted here (translation
    emits unwrapped values; the periodic kernel in
    :func:`build_exp_tens` wraps downstream), and stay separate scalar
    geometry passed to build.

    Parameters
    ----------
    p_attr : list/tuple of array-like
        Length-A list of ``K_total x N`` per-attribute value matrices
        (a 1-D entry is taken as a ``1 x N`` row).
    w : None, scalar, or length-A list
        Weights. Passed through unchanged (translation does not touch
        weights); returned as-is for clean chaining.
    offsets : length-A list
        Per-attribute offsets; see the layouts above.
    specs : None or length-A list, keyword-only
        Attribute specifications supplying per-attribute ``is_rel`` (outermost
        level) and value structure. ``None`` synthesises flat specs.

    Returns
    -------
    p_out : list of ndarray, or list of list of ndarray
        Single translation (``M = 1``): a length-A list of ``K_total x N``
        arrays. Sweep (``M > 1``): a length-M list of such lists.
    w : same as input
    specs : list of dict
        The attribute specifications, unchanged (or synthesised).

    Warns
    -----
    TranslateAttributesNoOpWarning
        When a uniform finite offset is applied to an attribute whose
        outermost level is relative (at most once per call).

    See Also
    --------
    difference_events, bind_events, flat_specs, build_exp_tens,
    windowed_similarity
    """
    if not isinstance(p_attr, (list, tuple)):
        raise ValueError(
            "p_attr must be a list/tuple of attribute value matrices."
        )
    A = len(p_attr)
    if A == 0:
        raise ValueError("p_attr must contain at least one attribute.")
    p_arr = []
    for a, M in enumerate(p_attr):
        Marr = np.asarray(M, dtype=np.float64)
        if Marr.ndim == 1:
            Marr = Marr.reshape(1, -1)
        elif Marr.ndim != 2:
            raise ValueError(
                f"Attribute {a} must be 1-D or 2-D; got ndim={Marr.ndim}."
            )
        p_arr.append(Marr)
    n_events = p_arr[0].shape[1]
    for a, M in enumerate(p_arr):
        if M.shape[1] != n_events:
            raise ValueError(
                f"All attributes must share the same event count N. "
                f"Attribute 0 has N={n_events}; attribute {a} has "
                f"N={M.shape[1]}."
            )
    K = [M.shape[0] for M in p_arr]            # K_total (rows) per attribute

    if specs is None:
        specs_out = flat_specs(p_arr)
    else:
        if not isinstance(specs, (list, tuple)) or len(specs) != A:
            raise ValueError(
                f"specs must be a length-A ({A}) list, one per attribute."
            )
        specs_out = list(specs)

    matrix_mode, M_sweep, blocks = _normalise_translate_offsets(offsets, K, A)

    # --- Relative no-op: a uniform finite shift on an outermost-relative
    # --- attribute is a structural no-op; skip that column, warn once. ---
    warned = False
    for a in range(A):
        if not _outermost_relative(specs_out[a]):
            continue
        col = blocks[a]
        for m in range(M_sweep):
            cm = col[:, m]
            finite = np.isfinite(cm)
            if finite.all() and finite.size and np.allclose(cm, cm.flat[0]):
                blocks[a][:, m] = np.nan
                warned = True
    if warned:
        warnings.warn(
            "A uniform finite offset was applied to an attribute whose "
            "outermost level is relative; a uniform shift cancels in "
            "every within-tuple difference, so it is a structural no-op "
            "and that column is left unchanged. (A non-uniform per-value "
            "offset would apply, as it shifts the relative structure.)",
            TranslateAttributesNoOpWarning,
            stacklevel=2,
        )

    # --- Apply: value + per-value offset, broadcast across events; NaN ---
    # --- values are left untranslated. ---
    cols_out: list[list[np.ndarray]] = []
    for m in range(M_sweep):
        col_list: list[np.ndarray] = []
        for a in range(A):
            Marr = p_arr[a]
            off = blocks[a][:, m]
            finite = np.isfinite(off)
            if not finite.any():
                col_list.append(Marr.copy())
            else:
                add = np.where(finite, off, 0.0).reshape(-1, 1)
                col_list.append(Marr + add)
        cols_out.append(col_list)

    if matrix_mode:
        # Carry the offsets with the sweep. A cell is uniform when every
        # value of that attribute moved by the same finite amount (a NaN
        # entry leaves its value in place, so it breaks uniformity unless
        # the whole column is NaN, which is no translation at all).
        uni = np.full((A, M_sweep), np.nan)
        for a in range(A):
            for m in range(M_sweep):
                cm = blocks[a][:, m]
                finite = np.isfinite(cm)
                if not finite.any():
                    uni[a, m] = 0.0
                elif finite.all() and np.all(cm == cm.flat[0]):
                    uni[a, m] = float(cm.flat[0])
        return (
            TranslatedSweep(
                cols_out, sweep_offsets=uni,
                sweep_base=[M.copy() for M in p_arr],
            ),
            w, specs_out,
        )
    return cols_out[0], w, specs_out         # single length-A list


def _normalise_translate_offsets(offsets, K, A):
    """Coerce a length-A offsets list to per-attribute ``(K_a, M)`` blocks.

    Each block is a float array with ``NaN`` marking values to skip; a
    scalar/1-D/single-column entry is broadcast across the common sweep
    width ``M``. Returns ``(matrix_mode, M, blocks)``.
    """
    if not isinstance(offsets, (list, tuple)) or len(offsets) != A:
        raise ValueError(
            f"offsets must be a length-A ({A}) list, one entry per "
            f"attribute (scalar, per-value vector, (1, M) or (K_total, M) "
            f"block, or None)."
        )
    raw = []
    M = 1
    for a, o in enumerate(offsets):
        if o is None:
            raw.append(None)
            continue
        arr = np.asarray(o, dtype=np.float64)
        if np.any(np.isinf(arr)):
            raise ValueError(
                f"offsets[{a}] contains +/-inf; entries must be finite or "
                f"NaN (NaN skips a row)."
            )
        if arr.ndim == 0:
            raw.append(arr.reshape(1, 1))
        elif arr.ndim == 1:
            if arr.size not in (1, K[a]):
                raise ValueError(
                    f"offsets[{a}] is a 1-D length-{arr.size} vector; "
                    f"expected length 1 or K_total = {K[a]}."
                )
            raw.append(arr.reshape(-1, 1))
        elif arr.ndim == 2:
            if arr.shape[0] not in (1, K[a]):
                raise ValueError(
                    f"offsets[{a}] has {arr.shape[0]} rows; expected 1 or "
                    f"K_total = {K[a]} (values down)."
                )
            raw.append(arr)
            M = max(M, arr.shape[1])
        else:
            raise ValueError(
                f"offsets[{a}] must be None, scalar, 1-D, or 2-D; got "
                f"ndim = {arr.ndim}."
            )
    matrix_mode = M > 1
    blocks = []
    for a, r in enumerate(raw):
        if r is None:
            blocks.append(np.full((K[a], M), np.nan))
            continue
        rows, cols = r.shape
        if rows == 1 and K[a] > 1:
            r = np.repeat(r, K[a], axis=0)
        if cols == 1 and M > 1:
            r = np.repeat(r, M, axis=1)
        elif cols not in (1, M):
            raise ValueError(
                f"offsets[{a}] has {cols} sweep columns; expected 1 or "
                f"M = {M} (all swept entries must agree on M)."
            )
        blocks.append(r)
    return matrix_mode, M, blocks


def _outermost_relative(spec):
    """Whether the attribute's outermost level is relative.

    A uniform global shift cancels exactly when the outermost (global)
    reading is relative: flat ``rel`` truthy, nested ``rel`` vector with a
    truthy last entry, or the ``rel = "outermost"`` selector.
    """
    if not isinstance(spec, dict):
        return False
    if "tags" in spec:
        rel = spec.get("rel")
        if isinstance(rel, str):
            return rel == "outermost"
        if isinstance(rel, (list, tuple, np.ndarray)):
            flat = np.ravel(rel)
            return bool(flat[-1]) if flat.size else False
        return bool(rel) if rel is not None else False
    return bool(spec.get("rel", False))
# ===================================================================
#  simplex_vertices
# ===================================================================


def simplex_vertices(N: int, edge_length: float = 1.0) -> np.ndarray:
    """Vertices of a regular (N-1)-simplex centred at the origin.

    Returns an N-by-(N-1) array whose rows are the vertices of a regular
    (N-1)-simplex in R^{N-1}, centred at the origin, with all pairwise
    vertex distances equal to ``edge_length`` (default 1).

    This is the natural numerical encoding of an N-level categorical
    attribute (voice identity, instrument, articulation, etc.) for the
    multi-attribute expectation tensor (MAET) framework. Each level is
    represented by an (N-1)-dimensional coordinate vector --- a row of
    the returned array --- and the categorical attribute group then
    carries N-1 coordinate sub-attributes sharing a single sigma.
    Because all vertices are pairwise equidistant, no level is
    privileged over any other, in contrast to dummy or treatment coding.

    Construction: take the N standard basis vectors of R^N (which lie
    on the hyperplane sum(x) = 1 and are pairwise equidistant), centre
    them at the origin, and project onto an orthonormal basis of the
    (N-1)-dimensional subspace orthogonal to the all-ones vector. The
    result is independent of the choice of basis up to a rotation,
    which is irrelevant for downstream MAET computations.

    Parameters
    ----------
    N : int
        Number of categorical levels. Must be at least 2.
    edge_length : float, optional
        Pairwise distance between vertices. Default 1.0. Must be
        positive.

    Returns
    -------
    np.ndarray
        N-by-(N-1) array; row k is the coordinate vector for level k.

    Examples
    --------
    >>> simplex_vertices(2).shape
    (2, 1)
    >>> simplex_vertices(3).shape
    (3, 2)
    >>> simplex_vertices(4).shape
    (4, 3)

    SATB voice encoding with edge length matched to a chosen sigma:

    >>> V = simplex_vertices(4, edge_length=4)
    >>> V.shape
    (4, 3)

    See Also
    --------
    build_exp_tens
    """
    if not isinstance(N, (int, np.integer)) or N < 2:
        raise ValueError(f"N must be an integer >= 2, got {N!r}.")
    if edge_length <= 0:
        raise ValueError(
            f"edge_length must be positive, got {edge_length!r}."
        )

    # Centred standard basis: rows are unit vectors minus the centroid.
    # Pairwise distance between rows is sqrt(2).
    Vc = np.eye(N) - 1.0 / N

    # Orthonormal basis of the column space of Vc, which is 1^perp
    # (the (N-1)-dimensional subspace orthogonal to the all-ones vector).
    # Use SVD: the first N-1 left singular vectors span the column space.
    U, _, _ = np.linalg.svd(Vc, full_matrices=False)
    Q = U[:, : N - 1]

    # Express each row of Vc in this basis. The result has N rows and
    # N-1 columns, with pairwise row distance sqrt(2). Rescale to the
    # requested edge length.
    return (Vc @ Q) * (edge_length / np.sqrt(2))
