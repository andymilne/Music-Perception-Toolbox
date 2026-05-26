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
from .density import _canonicalise_groups


class TranslateAttributesNoOpWarning(UserWarning):
    """Emitted when ``translate_attributes`` is asked to translate a
    relative-mode group, which is a structural no-op (the relative
    MAET depends only on within-tuple differences, so a uniform shift
    of all values cancels in every pairwise difference). The group is
    left unchanged."""



# ===================================================================
#  difference_events
# ===================================================================


def difference_events(
    p_attr,
    w,
    groups,
    diff_orders,
) -> tuple[list[np.ndarray], None | float | list[float] | list[np.ndarray], object]:
    """Replace selected attributes' event sequences with inter-event differences.

    Cross-event preprocessing for multi-attribute tensor input. Takes
    the ``(p_attr, w, groups)`` triple that one would otherwise feed
    to :func:`build_exp_tens` and returns a transformed
    ``(p_attr_diff, w_diff, groups_diff)`` triple ready to chain into
    another pre-MAET operation or into :func:`build_exp_tens`.

    Differencing orders are specified per attribute (via Option C
    syntax — see ``diff_orders`` below). The ``k_a``-th finite
    difference is applied along the event axis to each attribute,
    reducing its event count by ``k_a``. Order 0 leaves an attribute
    unchanged. Attributes are brought onto a common output event grid
    of length ``N' = N - max_a k_a`` by dropping leading
    ``max_a k_a - k_a`` events from each.

    Output values are emitted raw; periodic groups are NOT wrapped
    here, regardless of ``[per]`` settings. Wrapping (when desired)
    is the kernel's job in :func:`build_exp_tens`, consulting the
    group's ``[per]`` flag.

    Differencing requires ``K_a = 1`` for any attribute being
    differenced (``k_a > 0``). Multi-slot attributes (``K_a > 1``)
    are permitted in the input but only as pass-through (``k_a = 0``);
    if the analyst specifies a non-zero order for a ``K_a > 1``
    attribute, a :class:`UserWarning` is issued and that attribute is
    treated as ``k_a = 0`` (still subject to leading-event drop for
    alignment). The warning is emitted at most once per call.

    Per-attribute weights propagate as a rolling product over the
    ``k_a + 1`` constituent input events for each differenced
    attribute, under the standard weights-as-salience reading.
    Pass-through attributes have their leading events dropped to
    match the common grid.

    Parameters
    ----------
    p_attr : list/tuple of array-like
        Length-A list of ``(K_a, N)`` per-attribute value matrices.
        ``K_a >= 1``; ``K_a = 0`` (empty attribute) is rejected.
        ``K_a > 1`` is permitted as pass-through (``k_a = 0`` only).
    w : None, scalar, or list/tuple
        Weights. ``None``, a scalar, or a length-A list of per-
        attribute weight inputs (each ``None``, scalar, 1-D, or 2-D).
        Same convention as :func:`build_exp_tens`.
    groups : array-like or list-of-lists or None
        Group assignment. ``None`` treats each attribute as its own
        singleton group; a length-A index vector, or a list of
        attribute-index lists, gives explicit groupings.
    diff_orders : scalar, array-like, or dict
        Per-attribute or per-group differencing orders (non-negative
        integers). Option C syntax:

        - scalar: broadcast to all attributes.
        - length-``A`` array: per-attribute.
        - length-``G`` array (``G != A``): per-group, broadcast within
          group. The ``A == G`` case is read as per-attribute,
          producing identical output for either reading.
        - dict ``{g: value}`` (0-indexed groups): each value can be
          a scalar (broadcast within group) or a length-``n_g``
          vector (per-attribute within group). Omitted groups are
          treated as order 0.

    Returns
    -------
    p_attr_diff : list of ndarray
        Length-A list of transformed per-attribute matrices, each
        ``(K_a, N')``.
    w_diff : same general form as *w*
        Transformed weights. ``None`` stays ``None``; a scalar stays
        a scalar when all attributes share the same order, expanding
        to a length-A list of per-attribute scalars when orders vary;
        a length-A list stays a length-A list, with per-attribute
        event-dependent entries becoming ``(K_a, N')`` matrices and
        non-event-dependent entries keeping their input shape.
    groups_diff : same as *groups*
        Group structure passes through unchanged (differencing does
        not alter group membership). Returned for clean chaining of
        pre-MAET operations.

    Warns
    -----
    UserWarning
        If any attribute has ``K_a > 1`` and is assigned a non-zero
        order. That attribute is then treated as order 0
        (pass-through). The warning is emitted at most once per call.

    See Also
    --------
    build_exp_tens, bind_events, translate_attributes
    """
    # --- Normalize p_attr to list of 2-D float arrays ---
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

    # --- Reject empty (K_a = 0) attributes ---
    for a, M in enumerate(p_attr):
        if M.shape[0] == 0:
            raise ValueError(
                f"Attribute {a} has K_a = 0 (empty attribute); empty "
                f"attributes are not permitted."
            )

    # --- Verify shared event count N ---
    n_events = p_attr[0].shape[1]
    for a, M in enumerate(p_attr):
        if M.shape[1] != n_events:
            raise ValueError(
                f"All attributes must share the same event count N. "
                f"Attribute 0 has N={n_events}; attribute {a} has N={M.shape[1]}."
            )

    # --- Canonicalize groups ---
    group_of_attr, attrs_of_group, G = _canonicalise_groups(groups, A)

    # --- Parse diff_orders via Option C → orders_per_attr (length A) ---
    orders_per_attr = _canonicalise_diff_orders(
        diff_orders, A, G, attrs_of_group,
    )

    # --- Handle K_a > 1 with order > 0: warn once and pass through ---
    warned_multi_slot = False
    for a in range(A):
        K_a = p_attr[a].shape[0]
        if K_a > 1 and orders_per_attr[a] > 0:
            if not warned_multi_slot:
                warnings.warn(
                    f"Attribute {a} has K_a = {K_a} but was assigned "
                    f"order {orders_per_attr[a]}; event differencing "
                    f"requires K_a = 1 for differenced attributes "
                    f"(column-wise subtraction imposes a cross-event "
                    f"slot alignment that within-event slot "
                    f"exchangeability does not license). The attribute "
                    f"is treated as order 0 (passed through with "
                    f"leading-event drop). For voice-leading or "
                    f"step-size analyses, encode each voice as a "
                    f"K_a = 1 attribute and difference those. "
                    f"Subsequent multi-slot attributes in this call "
                    f"are silenced.",
                    UserWarning,
                    stacklevel=2,
                )
                warned_multi_slot = True
            orders_per_attr[a] = 0

    # --- Compute max_order, output event count ---
    max_order = int(orders_per_attr.max()) if A > 0 else 0
    n_prime = n_events - max_order
    if n_prime < 1:
        raise ValueError(
            f"Differencing orders are too high for the input event count: "
            f"max order = {max_order} but N = {n_events}."
        )

    # --- Difference each attribute's value matrix ---
    p_attr_diff = []
    for a, M in enumerate(p_attr):
        k = int(orders_per_attr[a])
        M_diff = M
        for _ in range(k):
            M_diff = M_diff[:, 1:] - M_diff[:, :-1]
        extra_drop = max_order - k
        if extra_drop > 0:
            M_diff = M_diff[:, extra_drop:]
        assert M_diff.shape[1] == n_prime
        p_attr_diff.append(M_diff)

    # --- Transform weights ---
    w_diff = _difference_weights(w, A, orders_per_attr, n_events, n_prime)

    # --- Groups unchanged ---
    return p_attr_diff, w_diff, groups


def _canonicalise_diff_orders(diff_orders, A, G, attrs_of_group):
    """Coerce ``diff_orders`` to a length-A int array via Option C."""
    if isinstance(diff_orders, dict):
        return _canonicalise_diff_orders_dict(
            diff_orders, A, G, attrs_of_group,
        )

    arr = np.asarray(diff_orders)
    if arr.dtype.kind not in "iuf":
        raise TypeError(
            f"diff_orders must be numeric or a dict; got dtype={arr.dtype}."
        )

    # 0-D scalar.
    if arr.ndim == 0:
        result = np.full(A, float(arr), dtype=np.float64)
    elif arr.ndim == 1:
        n = arr.size
        if n == 1:
            result = np.full(A, float(arr[0]), dtype=np.float64)
        elif n == A:
            # Per-attribute. (Also handles A == G case.)
            result = arr.astype(np.float64, copy=True)
        elif n == G:
            # Per-group, broadcast within group.
            result = np.zeros(A, dtype=np.float64)
            for g in range(G):
                result[attrs_of_group[g]] = arr[g]
        else:
            raise ValueError(
                f"diff_orders has {n} entries; expected scalar (1), "
                f"per-attribute (A = {A}), or per-group (G = {G})."
            )
    else:
        raise ValueError(
            f"diff_orders must be a scalar, 1-D array, or dict; "
            f"got ndim = {arr.ndim}."
        )

    _validate_orders(result)
    return result.astype(np.int64, copy=False)


def _canonicalise_diff_orders_dict(d, A, G, attrs_of_group):
    """Process the per-group dict form of diff_orders."""
    result = np.zeros(A, dtype=np.float64)
    for g_key, val in d.items():
        g = int(g_key)
        if g < 0 or g >= G:
            raise ValueError(
                f"diff_orders dict key {g} out of range; groups are "
                f"0-indexed, valid range [0, {G - 1}]."
            )
        attrs = attrs_of_group[g]
        n_g = len(attrs)
        val_arr = np.asarray(val).ravel()
        if val_arr.size == 1:
            result[attrs] = float(val_arr[0])
        elif val_arr.size == n_g:
            result[attrs] = val_arr.astype(np.float64)
        else:
            raise ValueError(
                f"diff_orders[{g}] has {val_arr.size} entries; expected "
                f"scalar or n_g = {n_g}."
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


def _difference_weights(w, A, orders_per_attr, n_events, n_prime):
    """Transform weights under per-attribute differencing orders.

    Each differenced attribute's weights are propagated via a rolling
    product of width ``k_a + 1``. Pass-through attributes (``k_a = 0``)
    have their leading events dropped to match the common output grid.
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
            W = _rolling_product(W, k + 1)
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


def _rolling_product(W, width):
    """Rolling product of width *width* along the last axis.

    Output shape: (K, N - width + 1). Output column *i* is
    ``prod(W[:, i : i + width], axis=1)``.
    """
    K, N = W.shape
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


def bind_events(
    p_attr,
    w,
    groups,
    bind_orders,
    *,
    circular: bool = False,
) -> tuple[list[np.ndarray], object, object]:
    """Bind sliding windows of consecutive events into super-attributes.

    Cross-event preprocessing for multi-attribute tensor input. Takes
    the ``(p_attr, w, groups)`` triple that one would otherwise feed
    to :func:`build_exp_tens` and returns a transformed
    ``(p_attr_bound, w_bound, groups_bound)`` triple ready to chain
    into another pre-MAET operation or into :func:`build_exp_tens`.

    Bind orders are specified per attribute (via Option C syntax —
    see ``bind_orders`` below). For an input attribute *a* with bind
    order ``L_a``, a sliding window of width ``L_a`` is laid across
    the event axis and each lag in the window is emitted as a
    separate output super-attribute. The total output attribute
    count is ``A' = sum_a L_a``; each input attribute contributes
    ``L_a`` super-attributes to the output, all in the same group as
    the source attribute. Lag identity is non-exchangeable, so the
    ``L_a`` copies are emitted as separate attributes rather than
    packed into a multi-slot one. The original ``K_a`` slot
    structure of each input attribute is preserved in every
    super-attribute.

    Event-axis alignment. The natural output event count of an
    attribute with bind order ``L_a`` is ``N - L_a + 1``
    (non-circular) or ``N`` (circular). With per-attribute orders,
    the common output event count is ``N' = N - max_a L_a + 1``
    (non-circular) or ``N`` (circular); attributes with
    ``L_a < max_a L_a`` have their trailing ``max_a L_a - L_a``
    super-events dropped to align all attributes on the same output
    grid. This is the natural composition partner of
    :func:`difference_events`' leading-drop alignment: D then B
    gives the same output (super-attribute by super-attribute) as B
    then D, for any choice of per-attribute orders.

    Weights. Each output super-attribute inherits the slot weights
    of the underlying input event at its lag, propagated under the
    toolbox's standard broadcast convention. :func:`build_exp_tens`
    then multiplies across attributes during tuple enumeration, so
    the end-to-end weight of a bound super-event equals the product
    of the ``L_a`` constituent events' weights — the natural
    pre-MAET factoring of the rolling product.

    Parameters
    ----------
    p_attr : list/tuple of array-like
        Length-A list of ``(K_a, N)`` per-attribute value matrices.
        ``K_a >= 1``; ``K_a = 0`` (empty attribute) is rejected.
    w : None, scalar, or list/tuple
        Weights. ``None``, a scalar, or a length-A list of per-
        attribute weight inputs (each ``None``, scalar, 1-D, or
        2-D). Same convention as :func:`build_exp_tens`.
    groups : array-like or list-of-lists or None
        Group assignment. ``None`` treats each attribute as its own
        singleton group; a length-A index vector, or a list of
        attribute-index lists, gives explicit groupings.
    bind_orders : scalar, array-like, or dict
        Per-attribute or per-group bind orders (positive integers,
        ``>= 1``). ``L = 1`` is the no-op (each input event becomes
        a one-event super-event = itself). Option C syntax:

        - scalar: broadcast to all attributes.
        - length-``A`` array: per-attribute.
        - length-``G`` array (``G != A``): per-group, broadcast
          within group. The ``A == G`` case is read as per-
          attribute, producing identical output for either reading.
        - dict ``{g: value}`` (0-indexed groups): each value can be
          a scalar (broadcast within group) or a length-``n_g``
          vector (per-attribute within group). Omitted groups are
          treated as ``L = 1`` (no-op).

    circular : bool, keyword-only
        When True, the sliding window wraps around the event axis
        and ``N' = N`` regardless of ``L_a``. When False (default),
        ``N' = N - max_a L_a + 1``.

    Returns
    -------
    p_attr_bound : list of ndarray
        Length-``A'`` list of super-attribute value matrices, each
        ``(K_a, N')``, where ``A' = sum_a L_a``.
    w_bound : same general form as *w*
        Transformed weights. ``None`` stays ``None``; a scalar stays
        a scalar; a length-A list becomes a length-``A'`` list with
        each super-attribute carrying the lag-indexed slice (event-
        dependent weights) or the inherited non-event-dependent
        input (scalar / ``None`` / ``(K_a, 1)`` column).
    groups_bound : ndarray of int
        Length-``A'`` array of group labels (1-indexed). Each input
        attribute's ``L_a`` super-attributes are placed in the same
        group as the source attribute (the group count is unchanged;
        group membership expands).

    See Also
    --------
    build_exp_tens, difference_events, translate_attributes
    """
    # --- Normalise p_attr to a list of 2-D float arrays ---
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

    # --- Verify shared event count N ---
    n_events = p_attr[0].shape[1]
    for a, M in enumerate(p_attr):
        if M.shape[1] != n_events:
            raise ValueError(
                f"All attributes must share the same event count N. "
                f"Attribute 0 has N={n_events}; attribute {a} has N={M.shape[1]}."
            )

    # --- Canonicalise groups ---
    group_of_attr, attrs_of_group, G = _canonicalise_groups(groups, A)

    # --- Parse bind_orders via Option C → orders_per_attr (length A) ---
    orders_per_attr = _canonicalise_bind_orders(
        bind_orders, A, G, attrs_of_group,
    )

    # --- Compute output sizes ---
    max_order = int(orders_per_attr.max()) if A > 0 else 0
    if circular:
        n_prime = n_events
        if max_order > n_events:
            raise ValueError(
                f"Circular window size max L = {max_order} exceeds "
                f"event count N = {n_events}."
            )
    else:
        n_prime = n_events - max_order + 1
        if n_prime < 1:
            raise ValueError(
                f"Bind orders too high for the input event count: "
                f"max L = {max_order} but N = {n_events} (non-circular)."
            )

    # --- Build output value matrices and group labels ---
    A_prime = int(orders_per_attr.sum())
    p_attr_bound = []
    groups_bound = np.zeros(A_prime, dtype=np.int64)
    out_idx = 0
    for a in range(A):
        L_a = int(orders_per_attr[a])
        g_a = int(group_of_attr[a])
        M = p_attr[a]
        for ell in range(L_a):
            if circular:
                indices = (np.arange(n_prime) + ell) % n_events
            else:
                indices = np.arange(ell, ell + n_prime)
            p_attr_bound.append(M[:, indices])
            groups_bound[out_idx] = g_a
            out_idx += 1

    # --- Transform weights ---
    w_bound = _bind_weights(
        w, A, orders_per_attr, n_events, n_prime, circular,
    )

    return p_attr_bound, w_bound, groups_bound


def _canonicalise_bind_orders(bind_orders, A, G, attrs_of_group):
    """Coerce ``bind_orders`` to a length-A int array via Option C."""
    if isinstance(bind_orders, dict):
        return _canonicalise_bind_orders_dict(
            bind_orders, A, G, attrs_of_group,
        )

    arr = np.asarray(bind_orders)
    if arr.dtype.kind not in "iuf":
        raise TypeError(
            f"bind_orders must be numeric or a dict; got dtype={arr.dtype}."
        )

    if arr.ndim == 0:
        result = np.full(A, float(arr), dtype=np.float64)
    elif arr.ndim == 1:
        n = arr.size
        if n == 1:
            result = np.full(A, float(arr[0]), dtype=np.float64)
        elif n == A:
            result = arr.astype(np.float64, copy=True)
        elif n == G:
            result = np.zeros(A, dtype=np.float64)
            for g in range(G):
                result[attrs_of_group[g]] = arr[g]
        else:
            raise ValueError(
                f"bind_orders has {n} entries; expected scalar (1), "
                f"per-attribute (A = {A}), or per-group (G = {G})."
            )
    else:
        raise ValueError(
            f"bind_orders must be a scalar, 1-D array, or dict; "
            f"got ndim = {arr.ndim}."
        )

    _validate_bind_orders(result)
    return result.astype(np.int64, copy=False)


def _canonicalise_bind_orders_dict(d, A, G, attrs_of_group):
    """Process the per-group dict form of bind_orders.

    Omitted groups default to ``L = 1`` (no-op).
    """
    result = np.ones(A, dtype=np.float64)  # default L = 1
    for g_key, val in d.items():
        g = int(g_key)
        if g < 0 or g >= G:
            raise ValueError(
                f"bind_orders dict key {g} out of range; groups are "
                f"0-indexed, valid range [0, {G - 1}]."
            )
        attrs = attrs_of_group[g]
        n_g = len(attrs)
        val_arr = np.asarray(val).ravel()
        if val_arr.size == 1:
            result[attrs] = float(val_arr[0])
        elif val_arr.size == n_g:
            result[attrs] = val_arr.astype(np.float64)
        else:
            raise ValueError(
                f"bind_orders[{g}] has {val_arr.size} entries; expected "
                f"scalar or n_g = {n_g}."
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


def _bind_weights(w, A, orders_per_attr, n_events, n_prime, circular):
    """Transform weights under per-attribute binding.

    Each output super-attribute carries the slot weights of the input
    event at its lag. Non-event-dependent inputs (``None``, scalar,
    ``(K_a, 1)`` column) are inherited as-is by every super-
    attribute; the kernel product over the ``L_a`` super-attributes
    in :func:`build_exp_tens` recovers the rolling product naturally.
    Event-dependent inputs (length-N 1-D, ``(1, N)`` row,
    ``(K_a, N)`` matrix) are sliced into the output's lag-indexed
    columns.
    """
    if w is None:
        return None

    # --- Top-level scalar ---
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

    w_bound = []
    for a, wa in enumerate(w):
        L_a = int(orders_per_attr[a])
        event_dep = _weight_has_event_dependence(wa, n_events, a)
        for ell in range(L_a):
            if not event_dep:
                # Inherit as-is.
                w_bound.append(wa)
                continue
            W = np.asarray(wa, dtype=np.float64)
            if W.ndim == 1:
                W = W.reshape(1, n_events)
            if circular:
                indices = (np.arange(n_prime) + ell) % n_events
            else:
                indices = np.arange(ell, ell + n_prime)
            w_bound.append(W[:, indices])
    return w_bound


# ===================================================================
#  weight_events
# ===================================================================


def weight_events(
    p_attr,
    w,
    groups,
    input_attrs,
    centre,
    width,
    shape,
    is_per,
    periods,
) -> list:
    r"""Apply a per-event weight via a multi-attribute window.

    Per-event preprocessing for multi-attribute tensor input. Computes
    a per-event weight contribution from a window in input-attribute
    space and multiplies it into the existing weight *w*, returning
    the new weight ``w_out``. Values and group structure pass through
    unchanged; only the per-attribute weight slots are updated.

    The window is specified per input attribute by three numeric
    parameters: ``centre_i``, ``width_i`` (window standard deviation
    in absolute units), and ``shape_i`` :math:`= \gamma_i \in [0, 1]`
    (dimensionless shape interpolating between Gaussian and
    rectangle). The factor written into the *i*-th input attribute's
    weight slot is the peak-normalised convolution
    :math:`\mathrm{rect}_{\phi_i} * \mathcal{G}_{\xi_i}` with
    derived sub-parameters

    .. math::

        \phi_i = \text{width}_i\sqrt{3\gamma_i},\qquad
        \xi_i  = \text{width}_i\sqrt{1-\gamma_i}.

    This fixed-variance parametrisation keeps the total variance equal
    to :math:`\text{width}_i^2` for every :math:`\gamma_i \in [0, 1]`
    (rectangle on :math:`[-\phi_i, \phi_i]` contributes variance
    :math:`\phi_i^2/3 = \text{width}_i^2\,\gamma_i`; Gaussian
    contributes :math:`\xi_i^2 = \text{width}_i^2\,(1-\gamma_i)`).
    The window's standard deviation is therefore :math:`\text{width}_i`
    throughout the family, regardless of :math:`\gamma_i`. The window
    is peak-normalised so :math:`h_i(0) = 1`.

    Limits (computed directly for numerical cleanliness):

    - :math:`\gamma_i = 0`: pure Gaussian
      :math:`h_i(\delta) = \exp(-\delta^2/(2\,\text{width}_i^2))`.
    - :math:`\gamma_i = 1`: pure rectangle
      :math:`h_i(\delta) = \mathbb{1}[|\delta| \le \text{width}_i\sqrt 3]`.

    Note that at :math:`\gamma_i = 1` the rectangle's half-width is
    :math:`\text{width}_i\sqrt 3` (so its variance equals
    :math:`\text{width}_i^2`); the parameter ``width`` always means
    the standard deviation, not the half-extent.

    Design P. Each input attribute contributes a *separate* factor
    :math:`h_i` to its own per-attribute weight slot. The kernel
    product over attributes in :func:`build_exp_tens` then recovers
    the joint window naturally as
    :math:`W_n = \prod_i h_i(p_{a_i, n})`. This factoring keeps each
    input attribute's weight independent and makes ``weight_events``
    commute cleanly with :func:`translate_attributes` (under a
    centre-shift rule) and pass through :func:`difference_events` /
    :func:`bind_events`.

    For multi-slot attributes (``K_a > 1``), the window factor is
    evaluated per slot value: ``h_i`` is applied independently to
    each of the ``K_a`` entries in every event column, yielding a
    ``(K_a, N)`` factor matrix that multiplies the input attribute's
    weight slot under :func:`build_exp_tens`' broadcast convention.

    The pre-MAET values themselves are not modified. For periodic
    groups (``is_per_i = True``), only the difference
    ``δ = v - centre_i`` used inside ``h_i`` is wrapped to
    ``[-P/2, P/2]``; the stored values ``v`` stay raw. (The kernel in
    :func:`build_exp_tens` handles the value-axis periodicity
    downstream via the group ``[per]`` flag.)

    Parameters
    ----------
    p_attr : list/tuple of array-like
        Length-A list of ``(K_a, N)`` per-attribute value matrices.
        ``K_a >= 1``; ``K_a = 0`` is rejected.
    w : None, scalar, or list/tuple
        Existing weights to multiply into. ``None``, a scalar, or a
        length-A list of per-attribute weight inputs (each ``None``,
        scalar, 1-D, or 2-D). Same convention as
        :func:`build_exp_tens`.
    groups : array-like or list-of-lists or None
        Group assignment (validated for shape only; not used by the
        window math).
    input_attrs : array-like of int
        Length-M array of attribute indices (0-based, in ``0..A-1``)
        selecting the attributes the window spans. Repeats are not
        allowed. If empty, ``w`` is returned unchanged
        (canonicalised to a length-A list).
    centre : array-like of float
        Length-M array of window centres.
    width : array-like of float
        Length-M array of window standard deviations (``> 0``), in
        absolute units of the attribute. The window's total variance
        is ``width**2`` for every value of ``shape``.
    shape : array-like of float
        Length-M array of shape parameters :math:`\gamma \in [0, 1]`.
        ``shape = 0`` is pure Gaussian; ``shape = 1`` is pure
        rectangle (with half-width :math:`\text{width}\sqrt 3`);
        intermediate values are the fixed-variance convolution family.
    is_per : array-like of bool
        Length-M boolean array. When ``True``, the difference
        ``δ = v - centre[i]`` is wrapped to
        ``[-periods[i]/2, periods[i]/2]`` before applying ``h_i``.
        Must mirror the ``[per]`` flag of ``input_attrs[i]``'s group
        in the downstream :func:`build_exp_tens` call.
    periods : array-like of float
        Length-M array of periods. Used only when ``is_per[i]`` is
        ``True``; required to be ``> 0`` in that case.

    Returns
    -------
    w_out : list
        Length-A list of per-attribute weights, ready to feed into
        :func:`build_exp_tens` (possibly after further pre-MAET
        chaining). Each input attribute's slot holds the existing
        input weight broadcast times ``h_i`` as a ``(K_a, N)`` matrix;
        non-input attributes pass through unchanged.

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
                f"Attribute 0 has N={n_events}; attribute {a} has N={M.shape[1]}."
            )

    # --- Validate groups (shape only) ---
    _canonicalise_groups(groups, A)

    # --- Validate input_attrs ---
    input_attrs = np.asarray(input_attrs).ravel()
    n_inputs = input_attrs.size
    if n_inputs == 0:
        return _normalise_weights_to_list(w, A)
    if input_attrs.dtype.kind not in "iu":
        if not np.all(input_attrs == np.round(input_attrs)):
            raise ValueError(
                "input_attrs entries must be integers."
            )
        input_attrs = input_attrs.astype(np.int64)
    if np.any(input_attrs < 0) or np.any(input_attrs >= A):
        raise ValueError(
            f"input_attrs entries must be in 0..A-1 = 0..{A - 1}."
        )
    if np.unique(input_attrs).size != n_inputs:
        raise ValueError(
            "input_attrs must not contain repeated indices."
        )

    # --- Validate centre, width, shape (all numeric, length M) ---
    centre = np.asarray(centre, dtype=np.float64).ravel()
    width = np.asarray(width, dtype=np.float64).ravel()
    shape = np.asarray(shape, dtype=np.float64).ravel()
    if centre.size != n_inputs:
        raise ValueError(
            f"centre must have length M = {n_inputs} (one per input "
            f"attribute); got {centre.size}."
        )
    if width.size != n_inputs:
        raise ValueError(
            f"width must have length M = {n_inputs}; got {width.size}."
        )
    if shape.size != n_inputs:
        raise ValueError(
            f"shape must have length M = {n_inputs} (gamma per input "
            f"attribute); got {shape.size}."
        )
    if not (np.all(np.isfinite(centre)) and np.all(np.isfinite(width))
            and np.all(np.isfinite(shape))):
        raise ValueError(
            "centre, width, and shape entries must be finite."
        )
    if np.any(width <= 0):
        raise ValueError("width entries must be > 0.")
    if np.any((shape < 0) | (shape > 1)):
        raise ValueError(
            "shape entries (gamma) must lie in [0, 1]: gamma = 0 is "
            "pure Gaussian, gamma = 1 is pure rectangle, intermediate "
            "values are the fixed-variance convolution family."
        )

    # --- Validate is_per, periods ---
    is_per_arr = np.asarray(is_per, dtype=bool).ravel()
    periods_arr = np.asarray(periods, dtype=np.float64).ravel()
    if is_per_arr.size != n_inputs:
        raise ValueError(
            f"is_per must have length M = {n_inputs}; got {is_per_arr.size}."
        )
    if periods_arr.size != n_inputs:
        raise ValueError(
            f"periods must have length M = {n_inputs}; got {periods_arr.size}."
        )
    if np.any(is_per_arr & (periods_arr <= 0)):
        raise ValueError(
            "periods entries must be > 0 wherever is_per is True "
            "(non-periodic entries are unused and may be 0)."
        )

    # --- Compute factor h_i for each input attribute ---
    factors = []
    for i in range(n_inputs):
        a = int(input_attrs[i])
        val_mat = p_attr[a].astype(np.float64, copy=False)   # (K_a, N)
        delta = val_mat - centre[i]
        if is_per_arr[i]:
            P = periods_arr[i]
            delta = delta - P * np.floor(delta / P + 0.5)
        factors.append(_evaluate_shape(delta, width[i], shape[i]))

    # --- Normalise w to length-A list and multiply factors in ---
    w_out = _normalise_weights_to_list(w, A)
    for i in range(n_inputs):
        a = int(input_attrs[i])
        w_out[a] = _multiply_weights(w_out[a], factors[i], a)
    return w_out


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
        return (np.abs(delta) <= phi).astype(np.float64)
    phi = width * np.sqrt(3.0 * gamma)
    xi = width * np.sqrt(1.0 - gamma)
    scale = xi * np.sqrt(2.0)
    num = _erf((delta + phi) / scale) - _erf((delta - phi) / scale)
    peak = 2.0 * _erf(phi / scale)
    return num / peak


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
    # numpy broadcasting handles the per-slot / per-event cases.
    return arr * factor


# ===================================================================
#  translate_attributes
# ===================================================================


def translate_attributes(
    p_attr,
    groups,
    offsets,
    is_rel,
    is_per,
    periods,
) -> list[np.ndarray] | list[list[np.ndarray]]:
    """Translate selected attributes' values by a chosen offset.

    Per-attribute preprocessing for multi-attribute tensor input. Takes
    the ``p_attr`` list one would otherwise feed to
    :func:`build_exp_tens` and returns a transformed ``p_attr_translated``
    list with the same shape conventions, in which selected attributes'
    values have been shifted by a chosen offset. The output feeds
    directly into :func:`build_exp_tens` without any further massaging.

    Sliding-comparison context. ``translate_attributes`` is the pre-tensor
    route to a sliding comparison along one or more attribute-group
    axes: for each candidate offset ``mu`` on a sweep grid, translate
    the events and compute a similarity against an un-shifted reference.
    The post-tensor counterpart is :func:`windowed_similarity`. Both
    answer related "slide along an axis" questions but with different
    operational properties; see the USER_GUIDE §3.1 discussion.

    Two offset-input forms are accepted: a single numeric block (for
    uniform broadcast or fully per-attribute layouts), or a dict
    keyed by group (for mixed-per-group layouts, e.g. broadcast on
    one group and per-attribute on another in the same call).

    **Numeric form.** ``offsets`` is a scalar or ndarray, with rows
    indexing attributes and columns indexing sweep positions. Row
    count must be 1 (broadcast across all attributes) or ``A``
    (per-attribute). Within-group broadcast is expressed by setting
    that group's rows equal.

    - 0-D scalar or 1-D length-1: broadcast across all attributes,
      single translation. Returns a length-``A`` list.
    - 1-D length-``M`` (``M > 1``) or 2-D ``(1, M)``: broadcast
      across all attributes, ``M``-position sweep. Returns a
      length-``M`` list of length-``A`` lists.
    - 2-D ``(A, M)``: per-attribute, ``M``-position sweep (``M = 1``
      acceptable for a single per-attribute translation). Returns a
      length-``M`` list of length-``A`` lists. Also handles ``A == G``
      (per-attribute and per-group equivalent).
    - 2-D ``(G, M)`` with ``G != A``: per-group, broadcast within
      group; ``M``-position sweep (``M = 1`` acceptable, returns a
      length-1 outer list — matrix mode, consistent with ``(A, M)``).
      Each group's row is replicated across its attributes.
    - 2-D with rows not in ``{1, A, G}``: ValueError.

    **Dict form.** ``offsets`` is ``{group_index: value}`` (0-indexed
    groups). Each value follows the same orientation convention, with
    ``n_g`` (the number of attributes in group ``g``) playing the role
    of ``A``:

    - 0-D scalar or 1-D length-1: broadcast within group, no sweep.
    - 1-D length-``M`` (``M > 1``) or 2-D ``(1, M)``: broadcast
      within group, ``M``-position sweep.
    - 2-D ``(n_g, M)``: per-attribute within group, ``M``-sweep.
    - 2-D with rows not in ``{1, n_g}``: ValueError.

    Groups omitted from the dict are not translated. All swept
    entries across the call (top-level columns or dict entries with
    ``M > 1``) must agree on ``M``; scalar and 1-column entries
    broadcast across the sweep. When any entry implies a sweep, the
    output is a length-``M`` list of length-``A`` lists; otherwise a
    single length-``A`` list.

    NaN entries in any numeric block skip the corresponding
    ``(attribute, column)`` cell. ``±inf`` is rejected.

    Semantics by group geometry (apply per offset column in matrix form):

    - **Absolute non-periodic** (``is_per[g] = False``): every value of
      every attribute in group ``g`` is replaced by ``value + mu``.
      ``periods[g]`` is ignored regardless of its sign.
    - **Absolute periodic** (``is_per[g] = True`` with
      ``is_rel[g] = False`` and ``periods[g] > 0``): every value is
      replaced by ``value + mu``, unwrapped. The wrapped periodic
      Gaussian kernel of :func:`build_exp_tens` is invariant under any
      additive shift by a multiple of ``P``, so no canonical wrap of
      the translated values is required --- the kernel handles
      periodicity downstream.
    - **Relative** (``is_rel[g] = True``): a uniform shift of every
      value cancels in every within-tuple difference, so translation
      on a relative group is a structural no-op. The group is left
      unchanged. A :class:`TranslateAttributesNoOpWarning` is emitted at
      most once per call, even when the matrix form has many columns
      with finite entries on the relative-group row.

    Weights are unaffected by translation and are not part of this
    function's signature; the caller passes the same ``w`` to
    :func:`build_exp_tens` after translation that they would have
    passed without it.

    Parameters
    ----------
    p_attr : list/tuple of array-like
        Length-A list of ``K_a x N`` per-attribute value matrices. Same
        convention as :func:`build_exp_tens`. A 1-D input is taken as
        a ``1 x N`` row.
    groups : array-like, list-of-lists, or None
        Group assignment, same convention as :func:`build_exp_tens` and
        :func:`difference_events`. ``None`` treats each attribute as its
        own singleton group.
    offsets : scalar, ndarray, or dict[int, scalar or array_like]
        Numeric form (single block, rows = attributes, columns =
        sweep) or dict form (keyed by group, polymorphic per-group
        values). See the "Two offset-input forms" block above.
    is_rel : array-like of bool
        Length-G vector of relative-mode flags, same convention as
        :func:`build_exp_tens`. Groups with ``is_rel[g] = True`` that
        have at least one finite offset entry emit a single
        :class:`TranslateAttributesNoOpWarning` and pass through unchanged
        on every column.
    is_per : array-like of bool
        Length-G vector of periodic-mode flags, same convention as
        :func:`build_exp_tens`. Accepted for signature parallelism with
        the rest of the MAET pipeline; not consulted by
        ``translate_attributes`` itself, since translation outputs
        unwrapped values and the periodic kernel in
        :func:`build_exp_tens` handles wrapping downstream.
    periods : array-like of float
        Length-G vector of periods, same convention as
        :func:`build_exp_tens`. Accepted for signature parallelism;
        not consulted by ``translate_attributes`` itself.

    Returns
    -------
    list of np.ndarray, or list of list of np.ndarray
        For vector-form offsets: a length-A list of ``K_a × N``
        ndarrays. For matrix-form offsets: a length-M list of such
        lists, one per offset column. The input ``p_attr`` is not
        mutated.

    Raises
    ------
    ValueError
        If ``offsets`` has the wrong shape (not a length-G vector or
        ``(G, M)`` matrix, and not a dict), if ``is_rel``, ``is_per``,
        or ``periods`` has the wrong length, or if any offset entry is
        ``±inf``.

    Warns
    -----
    TranslateAttributesNoOpWarning
        When the offsets specify a finite translation on a group with
        ``is_rel[g] = True``. At most one warning is emitted per call,
        regardless of how many columns or relative groups are involved.

    See Also
    --------
    difference_events : Replace event sequences with k-th differences.
    bind_events : Slide a length-n window over events to form n-attribute
        super-events.
    build_exp_tens : Consumes the output of this function.
    cos_sim_exp_tens : Raw-MA list mode consumes the matrix-form output
        directly.
    windowed_similarity : Post-tensor counterpart for sliding comparisons.

    Examples
    --------
    Transpose a chord progression by 6 semitones for a sliding-cosine
    sweep, with a single absolute non-periodic pitch group::

        >>> import numpy as np
        >>> from mpt import (
        ...     translate_attributes, build_exp_tens, cos_sim_exp_tens,
        ... )
        >>> p_q = [np.array([[60., 64., 67.], [63., 64., 67.]])]
        >>> p_c = [np.array([[62., 66., 69.], [65., 66., 69.]])]
        >>> groups = [0]
        >>> is_rel, is_per, periods = [False], [False], [0.0]
        >>> best = -np.inf
        >>> for mu in np.arange(-12., 12.01, 0.25):
        ...     p_c_mu = translate_attributes(
        ...         p_c, groups, {0: mu}, is_rel, is_per, periods,
        ...     )
        ...     M_q = build_exp_tens(
        ...         p_q, None, [0.15], [1], groups,
        ...         is_rel, is_per, periods, verbose=False,
        ...     )
        ...     M_c_mu = build_exp_tens(
        ...         p_c_mu, None, [0.15], [1], groups,
        ...         is_rel, is_per, periods, verbose=False,
        ...     )
        ...     best = max(best, cos_sim_exp_tens(M_q, M_c_mu, verbose=False))
        >>> bool(best > 0.99)
        True

    Sweep all transpositions in a single call using the matrix form::

        >>> mu_grid = np.arange(-12., 12.01, 0.25).reshape(1, -1)  # (G, M)
        >>> p_c_sweep = translate_attributes(
        ...     p_c, groups, mu_grid, is_rel, is_per, periods,
        ... )  # list of length M, each entry a length-A list
        >>> len(p_c_sweep) == mu_grid.shape[1]
        True
    """
    # ---- normalise / validate p_attr ----
    if not isinstance(p_attr, (list, tuple)):
        raise ValueError(
            "p_attr must be a list/tuple of attribute value matrices."
        )
    A = len(p_attr)
    if A == 0:
        raise ValueError("p_attr must contain at least one attribute.")
    p_attr_arrays = []
    for a, M in enumerate(p_attr):
        Marr = np.asarray(M, dtype=np.float64)
        if Marr.ndim == 1:
            Marr = Marr.reshape(1, -1)
        elif Marr.ndim != 2:
            raise ValueError(
                f"Attribute {a} must be 1-D or 2-D; got ndim={Marr.ndim}."
            )
        p_attr_arrays.append(Marr)

    n_events = p_attr_arrays[0].shape[1]
    for a, M in enumerate(p_attr_arrays):
        if M.shape[1] != n_events:
            raise ValueError(
                f"All attributes must share the same event count N. "
                f"Attribute 0 has N={n_events}; attribute {a} has "
                f"N={M.shape[1]}."
            )

    # ---- canonicalise groups ----
    group_of_attr, attrs_of_group, G = _canonicalise_groups(groups, A)

    # ---- validate is_rel, is_per, periods ----
    is_rel_arr = np.asarray(is_rel, dtype=bool).ravel()
    if is_rel_arr.size != G:
        raise ValueError(
            f"is_rel must have length G = {G} (number of groups); "
            f"got length {is_rel_arr.size}."
        )
    is_per_arr = np.asarray(is_per, dtype=bool).ravel()
    if is_per_arr.size != G:
        raise ValueError(
            f"is_per must have length G = {G}; got length "
            f"{is_per_arr.size}."
        )
    periods_arr = np.asarray(periods, dtype=np.float64).ravel()
    if periods_arr.size != G:
        raise ValueError(
            f"periods must have length G = {G}; got length "
            f"{periods_arr.size}."
        )

    # ---- normalise offsets to (A, M) per-attribute array, with NaN ----
    # ---- meaning "do not translate this attribute on this column". ----
    # ---- matrix_mode True means a sweep (return list of M lists); ----
    # ---- False means a single translation (return list of A). ----
    matrix_mode, offsets_per_attr = _normalise_offsets(
        offsets, A, G, attrs_of_group,
    )

    # ---- Identify relative groups carrying any finite per-attribute ----
    # ---- offset; emit at most one no-op warning per call, and zero ----
    # ---- out the rows so the hot loop's NaN check handles the skip. ----
    finite_mask = np.isfinite(offsets_per_attr)
    warned_relative = False
    for g in range(G):
        if not is_rel_arr[g]:
            continue
        attrs = attrs_of_group[g]
        if not any(np.any(finite_mask[a]) for a in attrs):
            continue
        if not warned_relative:
            warnings.warn(
                f"Group {g} has is_rel=True; translation is a "
                f"structural no-op on relative groups (a uniform shift "
                f"of all values cancels in every within-tuple "
                f"difference). The group is left unchanged on every "
                f"offset column.",
                TranslateAttributesNoOpWarning,
                stacklevel=2,
            )
            warned_relative = True
        for a in attrs:
            offsets_per_attr[a, :] = np.nan

    # ---- apply translation, column by column ----
    M_cols = offsets_per_attr.shape[1]
    cols_out: list[list[np.ndarray]] = []
    for m in range(M_cols):
        col_translated: list[np.ndarray] = []
        for a, Marr in enumerate(p_attr_arrays):
            mu = offsets_per_attr[a, m]
            if np.isnan(mu):
                col_translated.append(Marr.copy())
                continue
            col_translated.append(Marr + float(mu))
        cols_out.append(col_translated)

    if matrix_mode:
        return cols_out         # length-M list of length-A lists
    return cols_out[0]          # length-A list (single translation)


def _normalise_offsets(
    offsets,
    A: int,
    G: int,
    attrs_of_group: list[np.ndarray],
) -> tuple[bool, np.ndarray]:
    """Coerce the public offsets input to an ``(A, M)`` per-attribute
    ndarray with NaN-skip.

    Returns ``(matrix_mode, offsets_per_attr)``. ``matrix_mode`` is
    True when the user passed a 2-D numeric shape, a 1-D sweep, or a
    dict containing any 2-D value (output is wrapped as a length-``M``
    list of length-``A`` lists). False when the user passed a scalar
    or a dict of only scalar/1-D-length-1 entries (output is a
    length-``A`` list).

    Two top-level forms:

    1. **Numeric** (scalar or ndarray). Rows index attributes; columns
       index sweep positions. Row count must be exactly 1 (broadcast
       across all attributes) or ``A`` (per-attribute).

       - 0-D scalar or 1-D length-1: broadcast across all attributes,
         no sweep. Output ``(A, 1)``, ``matrix_mode = False``.
       - 1-D length-``M`` (``M > 1``): broadcast across all attributes,
         ``M``-position sweep. Equivalent to a 2-D ``(1, M)`` row.
       - 2-D ``(1, M)``: broadcast across all attributes, ``M``-position
         sweep (``matrix_mode = True`` regardless of ``M``).
       - 2-D ``(A, M)``: per-attribute, ``M``-position sweep
         (``matrix_mode = True`` regardless of ``M``).
       - 2-D with rows not in ``{1, A}``: ValueError.

    2. **Dict**, keyed by group index; per-group values follow an
       analogous orientation convention with ``n_g`` (number of
       attributes in group ``g``) playing the role of ``A``. Each
       per-group value is one of:

       - 0-D scalar or 1-D length-1: broadcast within group, no sweep.
       - 1-D length-``M`` (``M > 1``) or 2-D ``(1, M)``: broadcast
         within group, ``M``-position sweep.
       - 2-D ``(n_g, M)``: per-attribute within group, ``M``-sweep.
       - 2-D with rows not in ``{1, n_g}``: ValueError.

       Groups omitted from the dict are not translated. All entries
       (across both top-level and dict cases) carrying ``M > 1`` must
       agree on ``M``; scalar and 1-column entries broadcast across
       the sweep.

    NaN entries in any numeric block skip that ``(attribute, column)``
    cell; ``±inf`` is rejected.
    """
    if isinstance(offsets, dict):
        return _normalise_offsets_dict(offsets, A, G, attrs_of_group)

    arr = np.asarray(offsets, dtype=np.float64)

    # 0-D scalar (or 1-D length 1): broadcast no-sweep
    if arr.ndim == 0 or (arr.ndim == 1 and arr.size == 1):
        mu = float(arr.ravel()[0])
        if not np.isfinite(mu):
            raise ValueError(
                f"Scalar offset must be finite; got {mu!r}."
            )
        result = np.full((A, 1), mu, dtype=np.float64)
        return False, result

    # 1-D length > 1: broadcast sweep, treated as (1, M)
    if arr.ndim == 1:
        if np.any(np.isinf(arr)):
            raise ValueError(
                "offsets entries must be finite (or NaN to skip a cell)."
            )
        M = arr.size
        result = np.tile(arr.reshape(1, M), (A, 1))
        return True, result

    # 2-D: row count must be 1 (broadcast), A (per-attribute), or G (per-group)
    if arr.ndim == 2:
        if np.any(np.isinf(arr)):
            raise ValueError(
                "offsets entries must be finite (or NaN to skip a cell)."
            )
        n_rows, M = arr.shape
        if n_rows == 1:
            result = np.tile(arr, (A, 1))
            return True, result
        if n_rows == A:
            # Per-attribute (also handles A == G case, equivalent to per-group there).
            return True, arr.astype(np.float64, copy=True)
        if n_rows == G:
            # Per-group, broadcast within group. Expand to per-attribute
            # by replicating each group's row across its attributes.
            # Always returns matrix mode (M-position wrapper) for
            # consistency with the (A, M) and dict-form (n_g, M)
            # conventions: any 2-D input with n_rows > 1 is a per-axis
            # spec and gets a sweep wrapper. The A == G case is handled
            # above (per-attribute interpretation; identical numeric
            # output, same matrix-mode wrapping).
            result = np.zeros((A, M), dtype=np.float64)
            for g in range(G):
                attrs = attrs_of_group[g]
                result[attrs, :] = arr[g, :]
            return True, result
        raise ValueError(
            f"offsets is a 2-D array with shape {arr.shape}; row "
            f"count must be 1 (broadcast across all attributes), "
            f"A = {A} (per-attribute), or G = {G} (per-group, "
            f"broadcast within group). For mixed-per-group layouts, "
            f"use the dict form."
        )

    raise ValueError(
        f"offsets must be a scalar, a 1-D or 2-D ndarray, or a dict; "
        f"got an array with ndim = {arr.ndim}."
    )


def _normalise_offsets_dict(
    offsets_dict: dict,
    A: int,
    G: int,
    attrs_of_group: list[np.ndarray],
) -> tuple[bool, np.ndarray]:
    """Process the polymorphic dict form into a per-attribute (A, M) array."""
    # ---- First pass: validate keys, determine sweep dimension M ----
    M = 1
    matrix_mode = False
    parsed: list[tuple[int, np.ndarray]] = []
    for gkey, val in offsets_dict.items():
        try:
            gi = int(gkey)
        except (TypeError, ValueError):
            raise ValueError(
                f"offsets keys must be integer group indices; got {gkey!r}."
            ) from None
        if gi < 0 or gi >= G:
            raise ValueError(
                f"offsets contains group index {gi}, which is out of "
                f"range for G = {G} groups."
            )
        val_arr = np.asarray(val, dtype=np.float64)
        # Determine whether this entry establishes M and whether it
        # flips matrix_mode. Any 2-D entry forces matrix_mode = True;
        # a 1-D length > 1 entry also implies a sweep.
        if val_arr.ndim == 2:
            matrix_mode = True
            this_M = val_arr.shape[1]
            if this_M > 1:
                if M == 1:
                    M = this_M
                elif this_M != M:
                    raise ValueError(
                        f"offsets[{gi}] has shape {val_arr.shape}; "
                        f"sweep dimension {this_M} does not match "
                        f"the {M} sweep positions established by "
                        f"other entries."
                    )
        elif val_arr.ndim == 1 and val_arr.size > 1:
            matrix_mode = True
            this_M = val_arr.size
            if M == 1:
                M = this_M
            elif this_M != M:
                raise ValueError(
                    f"offsets[{gi}] is a 1-D array of length "
                    f"{this_M}; this does not match the {M} sweep "
                    f"positions established by other entries."
                )
        parsed.append((gi, val_arr))

    # ---- Second pass: distribute values to per-attribute (A, M) ----
    result = np.full((A, M), np.nan, dtype=np.float64)
    for gi, val_arr in parsed:
        attrs = attrs_of_group[gi]
        n_g = len(attrs)

        # 0-D scalar (or 1-D length 1): broadcast within group, no sweep
        if val_arr.ndim == 0 or (val_arr.ndim == 1 and val_arr.size == 1):
            mu = float(val_arr.ravel()[0])
            if not np.isfinite(mu):
                raise ValueError(
                    f"offsets[{gi}]: scalar offset must be finite "
                    f"(use omission from the dict to skip a group); "
                    f"got {mu!r}."
                )
            for a in attrs:
                result[a, :] = mu
            continue

        # 1-D length > 1: broadcast within group, sweep
        if val_arr.ndim == 1:
            if np.any(np.isinf(val_arr)):
                raise ValueError(
                    f"offsets[{gi}]: entries must be finite (or NaN "
                    f"to skip a column)."
                )
            for a in attrs:
                result[a, :] = val_arr
            continue

        # 2-D: row count must be 1 (broadcast within group) or n_g
        if val_arr.ndim == 2:
            if np.any(np.isinf(val_arr)):
                raise ValueError(
                    f"offsets[{gi}]: entries must be finite (or NaN "
                    f"to skip an (attribute, column) cell)."
                )
            n_rows = val_arr.shape[0]
            if n_rows == 1:
                for a in attrs:
                    result[a, :] = val_arr[0]
                continue
            if n_rows == n_g:
                for i, a in enumerate(attrs):
                    result[a, :] = val_arr[i]
                continue
            raise ValueError(
                f"offsets[{gi}] is 2-D with shape {val_arr.shape}; "
                f"row count must be 1 (broadcast within group) or "
                f"{n_g} (per-attribute, matching the number of "
                f"attributes in group {gi})."
            )

        raise ValueError(
            f"offsets[{gi}] must be a scalar, a 1-D array, or a 2-D "
            f"array; got an array with ndim = {val_arr.ndim}."
        )

    return matrix_mode, result# ===================================================================
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
