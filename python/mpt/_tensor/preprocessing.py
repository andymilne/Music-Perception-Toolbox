"""Cross-event preprocessing and categorical-encoding utilities.

This module hosts the small preprocessing layer that sits *before*
``build_exp_tens`` in the MAET pipeline:

* :func:`difference_events` --- replace event sequences with their
  k-th finite differences along the event axis.
* :func:`bind_events` --- slide a length-n window across an event
  sequence, emitting each window as an n-attribute super-event.
* :func:`translate_events` --- shift every value of every attribute
  in selected groups by a per-group offset (rigid translation).

It also exposes :func:`simplex_vertices`, the categorical-encoding
helper for level-symmetric MAET inputs.

See USER_GUIDE §3.1 ("Cross-event preprocessing") for the conceptual
introduction and :doc:`/ARCHITECTURE` §2 for the layering.
"""
from __future__ import annotations

import warnings

import numpy as np

from .._utils import validate_weights
from .density import _canonicalise_groups


class TranslateEventsNoOpWarning(UserWarning):
    """Emitted when ``translate_events`` is asked to translate a
    relative-mode group, which is a structural no-op (the relative
    MAET depends only on within-tuple differences, so a uniform shift
    of all values cancels in every pairwise difference). The group is
    left unchanged."""



# ===================================================================
#  difference_events
# ===================================================================


def difference_events(p_attr, w, groups, diff_orders, periods) -> tuple[list[np.ndarray], None | float | list[float] | list[np.ndarray]]:
    """Replace selected groups' event sequences with inter-event differences.

    Cross-event preprocessing for multi-attribute tensor input. Takes
    the ``(p_attr, w)`` pair that one would otherwise feed to
    :func:`build_exp_tens` and returns a transformed
    ``(p_attr_diff, w_diff)`` pair with the same shape conventions, in
    which the event columns of each selected group have been replaced
    by *k*-fold inter-event differences. The output feeds directly into
    :func:`build_exp_tens` without any further massaging.

    See the MAET specification §7 for the full semantics. In brief:

    - **Values.** For a group with ``diff_orders[g] = k``, the value
      matrix of each attribute in that group is replaced by its *k*-th
      finite difference along the event axis, reducing the event count
      by *k*. Order 0 leaves a group unchanged. If ``periods[g] > 0``,
      each raw difference is wrapped to ``[-P/2, P/2)`` (shortest-arc
      convention, matching :func:`cos_sim_exp_tens`).

    - **Weights.** Weight inputs follow the toolbox's standard
      broadcast convention: ``None`` broadcasts 1, a scalar
      broadcasts uniformly, event-dependent inputs (``(1, N)`` row,
      ``(K_a, N)`` matrix) supply per-event values, and a
      ``(K_a, 1)`` column (or 1-D length ``K_a``) broadcasts per
      slot. The weight of a difference event is the *product* of
      the weights of its ``k + 1`` constituent input events — a
      rolling product of width ``k + 1`` along the event axis —
      interpretable as the probability that all constituents are
      perceived under the standard weights-as-salience reading.

    - **Event-count alignment.** The output event count is
      ``N' = N - max_g k_g``. Groups with ``k_g < max_g k_g`` have
      their leading ``max_g k_g - k_g`` events dropped to keep columns
      aligned across groups. Weights are dropped to match.

    Parameters
    ----------
    p_attr : list/tuple of array-like
        Length-A list of ``K_a x N`` per-attribute value matrices, with
        ``K_a = 1`` for every attribute. Same convention as
        :func:`build_exp_tens` but restricted to the single-slot case:
        within-event slot exchangeability does not license the cross-
        event slot correspondence that column-wise differencing
        imposes, so attributes with ``K_a != 1`` raise
        :class:`ValueError`. For voice-leading or step-size analyses,
        encode each voice as its own ``K_a = 1`` attribute in a shared
        group, difference that, then (optionally) stack the
        differenced attributes into a single multi-slot attribute
        before :func:`build_exp_tens`.
    w : None, scalar, or list/tuple
        Weights. ``None``, a scalar, or a length-A list of per-attribute
        weight inputs (each ``None``, scalar, 1-D, or 2-D). Same
        convention as :func:`build_exp_tens`.
    groups : array-like or list-of-lists or None
        Group assignment. ``None`` treats each attribute as its own
        singleton group; a length-A index vector, or a cell/list of
        attribute-index lists, gives explicit groupings. Matches
        :func:`build_exp_tens`.
    diff_orders : array-like of int
        Length-G vector of per-group differencing orders (non-negative
        integers). Order 0 leaves the group unchanged.
    periods : array-like of float
        Length-G vector of periods for shortest-arc wrapping of
        differences. An entry of 0 (or negative) means the group is
        treated as non-periodic and differences are left unwrapped.

    Returns
    -------
    p_attr_diff : list of ndarray
        Length-A list of transformed per-attribute matrices, each
        ``K_a x N'``.
    w_diff : same general form as *w*
        Transformed weights under the rule described above. Shape
        mirrors *w*: ``None`` stays ``None``; a scalar stays a scalar
        when all groups share the same order, expanding to a length-A
        list of per-attribute scalars when orders vary; a length-A
        list stays a length-A list, with per-attribute event-dependent
        entries becoming ``(K_a, N')`` matrices and non-event-
        dependent entries keeping their input shape.

    See Also
    --------
    build_exp_tens
    """
    # --- Normalize p_attr to list of 2-D float arrays ---
    if not isinstance(p_attr, (list, tuple)):
        raise TypeError("p_attr must be a list/tuple of per-attribute matrices.")
    p_attr = [np.asarray(M, dtype=np.float64) for M in p_attr]
    for a, M in enumerate(p_attr):
        if M.ndim != 2:
            raise ValueError(
                f"Attribute {a} value matrix must be 2-D; got ndim={M.ndim}."
            )
    A = len(p_attr)
    if A == 0:
        raise ValueError("p_attr must contain at least one attribute.")

    # --- Enforce K_a = 1 per attribute ---
    # Event differencing requires every attribute to have exactly one
    # slot per event. Column-wise subtraction across adjacent events
    # imposes a cross-event slot correspondence (slot i at event n-1
    # paired with slot i at event n) that within-event slot
    # exchangeability does not license; for multi-slot attributes the
    # output would silently depend on an arbitrary slot-listing
    # choice. K_a = 0 (empty attribute) is also rejected. The
    # principled route for voice-leading or step-size analyses is to
    # encode each voice as its own K_a = 1 attribute in a shared
    # group, difference that, then (optionally) stack the differenced
    # attributes into a single multi-slot attribute before
    # build_exp_tens. See USER_GUIDE Section 3 (Event differencing).
    for a, M in enumerate(p_attr):
        K_a = M.shape[0]
        if K_a != 1:
            raise ValueError(
                f"Attribute {a} has K_a = {K_a}; event differencing "
                f"requires every attribute to have K_a = 1. Column-wise "
                f"differencing imposes a cross-event slot alignment that "
                f"within-event slot exchangeability does not license, so "
                f"multi-slot attributes are rejected; empty attributes "
                f"(K_a = 0) are rejected likewise. For voice-leading or "
                f"step-size analyses, encode each voice as a separate "
                f"K_a = 1 attribute in a shared group, call "
                f"difference_events on that, then (optionally) stack "
                f"the differenced attributes into a single multi-slot "
                f"attribute before build_exp_tens. See USER_GUIDE "
                f"Section 3 (Event differencing)."
            )

    n_events = p_attr[0].shape[1]
    for a, M in enumerate(p_attr):
        if M.shape[1] != n_events:
            raise ValueError(
                f"All attributes must share the same event count N. "
                f"Attribute 0 has N={n_events}; attribute {a} has N={M.shape[1]}."
            )

    # --- Canonicalize groups ---
    group_of_attr, _attrs_of_group, G = _canonicalise_groups(groups, A)

    # --- Validate diff_orders and periods ---
    diff_orders = np.asarray(diff_orders, dtype=np.intp).ravel()
    if diff_orders.size != G:
        raise ValueError(
            f"diff_orders must have length G = {G} (number of groups); "
            f"got length {diff_orders.size}."
        )
    if np.any(diff_orders < 0):
        raise ValueError("All entries of diff_orders must be non-negative.")

    periods = np.asarray(periods, dtype=np.float64).ravel()
    if periods.size != G:
        raise ValueError(
            f"periods must have length G = {G}; got length {periods.size}."
        )

    max_order = int(diff_orders.max()) if diff_orders.size > 0 else 0
    n_prime = n_events - max_order
    if n_prime < 1:
        raise ValueError(
            f"Differencing orders are too high for the input event count: "
            f"max(diff_orders) = {max_order} but N = {n_events}."
        )

    # --- Difference each attribute's value matrix ---
    p_attr_diff = []
    for a, M in enumerate(p_attr):
        g = int(group_of_attr[a])
        k = int(diff_orders[g])
        P = float(periods[g])
        # Apply k-fold differencing along axis 1, with optional wrapping
        # after each first-order difference.
        M_diff = M
        for _ in range(k):
            M_diff = M_diff[:, 1:] - M_diff[:, :-1]
            if P > 0:
                M_diff = M_diff - P * np.floor(M_diff / P + 0.5)
        # Drop leading events to align with N'.
        extra_drop = max_order - k
        if extra_drop > 0:
            M_diff = M_diff[:, extra_drop:]
        assert M_diff.shape[1] == n_prime
        p_attr_diff.append(M_diff)

    # --- Transform weights ---
    w_diff = _difference_weights(w, A, group_of_attr, diff_orders,
                                  n_events, n_prime)

    return p_attr_diff, w_diff


def _difference_weights(w, A, group_of_attr, diff_orders,
                        n_events, n_prime):
    """Transform weights under the difference-events convention.

    The weight of each difference event is the product of the weights
    of the ``k + 1`` input events on which the difference depends —
    a rolling product of width ``k + 1`` along the event axis,
    applied semantically under the toolbox's broadcast convention.
    Under the ``K_a = 1`` restriction on :func:`difference_events`
    inputs, valid per-attribute weight inputs are ``None``, scalar,
    or ``(1, N)`` / 1-D of length ``N``.
    """
    if w is None:
        return None

    # --- Top-level scalar ---
    if np.isscalar(w):
        c = float(w)
        orders_per_attr = np.array(
            [int(diff_orders[int(group_of_attr[a])]) for a in range(A)],
            dtype=np.int64,
        )
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

    max_order = int(np.max(diff_orders)) if A > 0 else 0
    w_diff = []
    for a, wa in enumerate(w):
        g = int(group_of_attr[a])
        k = int(diff_orders[g])
        if not _weight_has_event_dependence(wa, n_events, a):
            # No event dependence — rolling product of a constant
            # reduces to raising each entry to power k + 1. None
            # stays None; a scalar stays a scalar.
            w_diff.append(_raise_no_event_dep(wa, k + 1))
            continue
        # Event-dependent: coerce to (1, N) then take a rolling
        # product of width k + 1 along the event axis.
        W = np.asarray(wa, dtype=np.float64).reshape(1, n_events)
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

    Under the ``K_a = 1`` restriction on :func:`difference_events`
    inputs, the non-event-dependent inputs that reach this helper
    are limited to ``None``, Python scalars, 0-D arrays, and size-1
    1-D arrays.
    """
    if wa is None:
        return None
    if p == 1:
        return wa  # fast path: order 0 groups
    if np.isscalar(wa):
        return float(wa) ** int(p)
    # 0-D or size-1 1-D array: coerce to Python scalar for consistency
    # with the scalar branch.
    return float(np.asarray(wa).item()) ** int(p)


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

    Under the ``K_a = 1`` restriction on :func:`difference_events`
    inputs, valid per-attribute weight shapes are ``None``, scalar,
    or ``(1, N)`` / 1-D of length ``N``.
    """
    if wa is None:
        return False
    arr = np.asarray(wa)
    if arr.size == 1:
        return False
    if arr.ndim == 1 and arr.size == N:
        return True
    if arr.ndim == 2 and arr.shape == (1, N):
        return True
    where = (
        f"Attribute {attr_idx} weight" if attr_idx is not None else "Weight"
    )
    raise ValueError(
        f"{where} has shape {arr.shape}; under the K_a = 1 restriction, "
        f"expected None, scalar, or (1, {N})."
    )


# ===================================================================
#  bind_events: sliding-window binding into n-attribute super-events
# ===================================================================



# ===================================================================
#  bind_events
# ===================================================================


def bind_events(p, w=None, n=2, *, circular=False) -> tuple[list[np.ndarray], None | float | list[np.ndarray]]:
    """Bind n consecutive events into n-attribute super-events.

    Cross-event preprocessing helper for multi-attribute tensor input.
    Slides a window of width *n* across an event sequence and emits
    each window as an *n*-attribute super-event whose *j*-th
    attribute holds the value(s) at lag ``j-1`` (``j = 1, ..., n``).
    The output is a length-*n* list of ``(K_a, N')`` arrays, suitable
    for direct use as the ``p_attr`` argument of
    :func:`build_exp_tens` with all *n* attributes assigned to a
    single group.

    This complements :func:`difference_events`: differencing
    aggregates *across* event boundaries (collapsing ``k+1``
    consecutive events into a single value), while binding
    aggregates *within* a window (gathering *n* consecutive values
    into a single super-event with *n* separate slot attributes). The
    two operations compose naturally: differencing then binding gives
    *n*-tuples of consecutive step sizes, recovering the *n*-tuple
    entropy of Milne & Dean (2016) as a special case (``sigma -> 0``,
    integer-step grid, uniform weights, periodic domain) while
    extending it to non-zero ``sigma``, continuous-valued steps,
    non-periodic domains, and per-event weights propagated through
    both stages. The bound MAET is itself a density that can be fed
    into the rest of the toolbox: pairs of bound MAETs can be
    compared via :func:`cos_sim_exp_tens`, queried via
    :func:`windowed_similarity`, and so on.

    Lag slots are emitted as separate (``K_a``-valued) attributes,
    one per lag, rather than packed into a single attribute, because
    lag identity is not exchangeable: the value at lag *j* and the
    value at lag *j+1* carry distinct positional meanings within the
    bound super-event. By contrast, ``K_a > 1`` inputs (multi-value
    attributes whose slots are deliberately exchangeable) are
    permitted: each output attribute then carries the ``K_a`` slot
    values of one underlying event, and the within-attribute
    exchangeability is preserved per output attribute. Cross-event
    slot alignment is never imposed, because the cross-event
    structure is between output attributes, not within.

    ``N' = N - n + 1`` (default) or ``N`` (when *circular* is True).

    Parameters
    ----------
    p : array-like
        Event values: a ``(K_a, N)`` 2-D array, a ``(1, N)`` row, a
        length-*N* 1-D array (interpreted as ``K_a = 1``), or a
        length-1 list/tuple wrapping one of those (the wrapped form
        is accepted for symmetry with the output of
        :func:`difference_events`). ``K_a >= 1``.
    w : None, scalar, or array-like
        Weights. ``None``, scalar, length-*N* 1-D, ``(1, N)`` row,
        ``(K_a, 1)`` column, or ``(K_a, N)`` matrix. May also be a
        length-1 list/tuple wrapping any of those (matching the *p*
        form).
    n : int
        Window size (positive integer; default 2).
    circular : bool, keyword-only
        When True, the window wraps around the end of the sequence;
        ``N' = N``. When False (default), ``N' = N - n + 1``.

        The *circular* flag describes the *event sequence* (whether
        the last event connects back to the first), and is
        independent of the *positional periodicity* set in
        :func:`build_exp_tens` via its ``is_per`` / ``period``
        arguments. Both combinations are meaningful: a non-circular
        sequence on a periodic domain (a non-cyclic motif living in
        pitch-class space), and a circular sequence on a linear
        domain (a cyclic rhythm represented in linear time, e.g.,
        for windowed analysis). The two flags are orthogonal.

    Returns
    -------
    p_bound : list of ndarray
        Length-*n* list of ``(K_a, N')`` arrays. ``p_bound[j]``
        contains the value(s) at lag *j* for each window. For
        ``K_a = 1`` input each entry is ``(1, N')``.
    w_bound : None, scalar, or list of ndarray
        Per-attribute weight propagation, in the form that
        :func:`build_exp_tens` accepts directly:

        - ``None`` stays ``None``.
        - A scalar ``c`` stays ``c`` (broadcast in
          :func:`build_exp_tens`).
        - A ``(1, N)`` / length-*N* row becomes a length-*n* list of
          ``(1, N')`` rows.
        - A ``(K_a, 1)`` column becomes a length-*n* list of
          ``(K_a, 1)`` columns.
        - A ``(K_a, N)`` matrix becomes a length-*n* list of
          ``(K_a, N')`` matrices.

        The end-to-end numerics are equivalent to a rolling product
        of slot weights: each output attribute inherits the slot
        weights of the underlying event at its lag, and
        :func:`build_exp_tens` multiplies across attributes during
        tuple enumeration.

    Examples
    --------
    >>> # 2-tuple entropy of step sizes (diatonic scale, sigma -> 0)
    >>> import numpy as np
    >>> from mpt import (difference_events, bind_events, build_exp_tens,
    ...                   entropy_exp_tens)
    >>> p = np.array([[0, 2, 4, 5, 7, 9, 11]], dtype=float)
    >>> d = difference_events([p], None, None, [1], [12])
    >>> p_bound, w_bound = bind_events(d[0], None, 2, circular=True)
    >>> T = build_exp_tens(p_bound, w_bound, [1e-6], [1, 1], [1, 1],
    ...                     [False], [True], [12], verbose=False)
    >>> H = entropy_exp_tens(T, normalize=False)

    See Also
    --------
    difference_events
    build_exp_tens
    entropy_exp_tens
    cos_sim_exp_tens
    n_tuple_entropy
    """
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError(f"n must be a positive integer; got {n!r}.")
    n = int(n)

    # --- Unwrap length-1 list/tuple (symmetry with difference_events) ---
    input_was_list_p = isinstance(p, (list, tuple))
    if input_was_list_p:
        if len(p) != 1:
            raise ValueError(
                f"p as a list/tuple must contain exactly one attribute; "
                f"got {len(p)}. To bind multiple attributes, call "
                f"bind_events on each separately."
            )
        p = p[0]

    input_was_list_w = isinstance(w, (list, tuple))
    if input_was_list_w:
        if len(w) != 1:
            raise ValueError(
                f"w as a list/tuple must contain exactly one entry; "
                f"got {len(w)}."
            )
        w = w[0]

    # --- Validate p shape (allow any K_a >= 1) ---
    arr = np.asarray(p, dtype=np.float64)
    if arr.ndim == 1:
        p_mat = arr.reshape(1, -1)            # length-N -> (1, N)
    elif arr.ndim == 2:
        p_mat = arr
    else:
        raise ValueError(
            f"p has shape {arr.shape}; bind_events requires p to be a "
            f"length-N 1-D array or a (K_a, N) 2-D array."
        )
    Ka = p_mat.shape[0]
    N = p_mat.shape[1]

    # --- Determine window indices ---
    if circular:
        if n > N:
            raise ValueError(
                f"Circular window size n = {n} exceeds event count "
                f"N = {N}."
            )
        n_prime = N
        idx = (np.arange(n_prime)[:, None] + np.arange(n)[None, :]) % N
    else:
        n_prime = N - n + 1
        if n_prime < 1:
            raise ValueError(
                f"Window size n = {n} exceeds event count N = {N} "
                f"(non-circular mode)."
            )
        idx = np.arange(n_prime)[:, None] + np.arange(n)[None, :]

    # --- Build the n lag matrices, preserving K_a ---
    p_bound = [p_mat[:, idx[:, j]].copy() for j in range(n)]

    # --- Weights: per-attribute propagation ---
    #
    # Each output attribute inherits the slot weights of the
    # underlying event at its lag. build_exp_tens multiplies across
    # attributes during tuple enumeration, so the end-to-end weight
    # of a bound super-event equals the product of the n constituent
    # events' weights (the same numerical contribution as the prior
    # eager rolling product, for K_a = 1; the natural generalisation
    # for K_a > 1).

    if w is None:
        w_bound = None
    elif np.isscalar(w):
        # Scalar weight broadcasts in build_exp_tens; pass through.
        w_bound = float(w)
    else:
        w_arr = np.asarray(w, dtype=np.float64)

        # Scalar packed in a 0-D or single-element array
        if w_arr.size == 1:
            w_bound = float(w_arr.item())
        else:
            # Coerce to a 2-D matrix consistent with the p-shape.
            if w_arr.ndim == 1:
                if w_arr.size == N:
                    w_mat = w_arr.reshape(1, -1)         # (1, N)
                elif w_arr.size == Ka and Ka != N:
                    w_mat = w_arr.reshape(-1, 1)         # (K_a, 1)
                elif Ka == N:
                    # Ambiguous: prefer the row interpretation, matching
                    # the convention for length-N input.
                    w_mat = w_arr.reshape(1, -1)
                else:
                    raise ValueError(
                        f"w as a 1-D array must have length N = {N} or "
                        f"K_a = {Ka} (got length {w_arr.size})."
                    )
            elif w_arr.ndim == 2:
                w_mat = w_arr
            else:
                raise ValueError(
                    f"w has shape {w_arr.shape}; expected None, scalar, "
                    f"1-D, or 2-D."
                )

            r, c = w_mat.shape
            if r == 1 and c == N:
                w_bound = [w_mat[:, idx[:, j]].copy() for j in range(n)]
            elif r == Ka and c == 1:
                w_bound = [w_mat.copy() for _ in range(n)]
            elif r == Ka and c == N:
                w_bound = [w_mat[:, idx[:, j]].copy() for j in range(n)]
            else:
                raise ValueError(
                    f"w has shape {w_mat.shape}; expected None, scalar, "
                    f"length-{N} 1-D, ({1}, {N}) row, "
                    f"({Ka}, 1) column, or ({Ka}, {N}) matrix."
                )

    # --- Re-wrap weight in a length-1 list if input was list ---
    if input_was_list_w:
        w_bound = [w_bound]

    return p_bound, w_bound


# ===================================================================
#  Windowing: window_tensor, windowed_similarity, and supporting math
# ===================================================================



# ===================================================================
#  translate_events
# ===================================================================


def translate_events(
    p_attr,
    groups,
    offsets,
    is_rel,
    is_per,
    periods,
) -> list[np.ndarray] | list[list[np.ndarray]]:
    """Translate selected groups' event values by a per-group offset.

    Cross-event preprocessing for multi-attribute tensor input. Takes
    the ``p_attr`` list one would otherwise feed to
    :func:`build_exp_tens` and returns a transformed ``p_attr_translated``
    list with the same shape conventions, in which every value of every
    attribute belonging to a selected group has been shifted by the
    group's offset. The output feeds directly into
    :func:`build_exp_tens` without any further massaging.

    Sliding-comparison context. ``translate_events`` is the pre-tensor
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
      length-``M`` list of length-``A`` lists.
    - 2-D with rows not in ``{1, A}``: ValueError.

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
      unchanged. A :class:`TranslateEventsNoOpWarning` is emitted at
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
        :class:`TranslateEventsNoOpWarning` and pass through unchanged
        on every column.
    is_per : array-like of bool
        Length-G vector of periodic-mode flags, same convention as
        :func:`build_exp_tens`. Accepted for signature parallelism with
        the rest of the MAET pipeline; not consulted by
        ``translate_events`` itself, since translation outputs
        unwrapped values and the periodic kernel in
        :func:`build_exp_tens` handles wrapping downstream.
    periods : array-like of float
        Length-G vector of periods, same convention as
        :func:`build_exp_tens`. Accepted for signature parallelism;
        not consulted by ``translate_events`` itself.

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
    TranslateEventsNoOpWarning
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
        ...     translate_events, build_exp_tens, cos_sim_exp_tens,
        ... )
        >>> p_q = [np.array([[60., 64., 67.], [63., 64., 67.]])]
        >>> p_c = [np.array([[62., 66., 69.], [65., 66., 69.]])]
        >>> groups = [0]
        >>> is_rel, is_per, periods = [False], [False], [0.0]
        >>> best = -np.inf
        >>> for mu in np.arange(-12., 12.01, 0.25):
        ...     p_c_mu = translate_events(
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
        >>> p_c_sweep = translate_events(
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
                TranslateEventsNoOpWarning,
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

    # 2-D: row count must be 1 (broadcast) or A (per-attribute)
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
            return True, arr.astype(np.float64, copy=True)
        raise ValueError(
            f"offsets is a 2-D array with shape {arr.shape}; row "
            f"count must be 1 (broadcast across all attributes) or "
            f"A = {A} (per-attribute). For per-group offsets, use "
            f"the dict form."
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
