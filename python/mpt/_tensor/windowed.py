"""Pre-MAET windowed sweeps: ``windowed_similarity`` and ``windowed_entropy``.

Both slide a window along one attribute axis of a *pre-MAET* carrier and
read out a profile at a sequence of centres, sharing the placement and
sweep-geometry helpers below. ``windowed_similarity`` loops over context
centres and scores the context, placed once per centre, against the whole
trailing-axis batch of query placements in a single call (recovering the
cross-correlation batching); ``windowed_entropy`` is a plain per-centre
loop, there being one density and one entropy per centre.

Each operand undergoes, at each sweep step, exactly one placement
transform on the window axis:

* ``window is None``  -> the operand is *translated* whole, so its mean on
  the window axis lands on the centre (template placement, via
  :func:`translate_attributes`);
* ``window=(shape, width)`` -> the operand is *windowed* at the centre
  (selection, via :func:`weight_events`), ``shape`` in ``[0, 1]``
  (0 Gaussian, 1 rectangular) or the aliases ``'gaussian'`` / ``'rect'``,
  ``width`` the rectangular full support (variance-matched for other
  shapes: ``sd = width / (2 sqrt 3)``).

``windowed_similarity`` takes two operands (a context and a query) and
reads out their finalised inner product; ``windowed_entropy`` takes one
and reads out its entropy. A future ``windowed_harmonicity`` (or similar)
would be one more wrapper over the same core.
"""

from __future__ import annotations

import numpy as np

from .preprocessing import weight_events, translate_attributes
from .build import build_exp_tens
from .cosine import cos_sim_exp_tens
from .density import _weight_is_live

_SHAPE_ALIASES = {
    "rect": 1.0, "rectangular": 1.0, "box": 1.0,
    "gaussian": 0.0, "gauss": 0.0, "normal": 0.0,
}


def _resolve_shape(shape):
    if isinstance(shape, str):
        try:
            return _SHAPE_ALIASES[shape.lower()]
        except KeyError:
            raise ValueError(
                f"unknown window shape {shape!r}; pass a float in [0, 1] "
                f"(0 Gaussian, 1 rectangular) or one of "
                f"{sorted(_SHAPE_ALIASES)}"
            )
    s = float(shape)
    if not (0.0 <= s <= 1.0):
        raise ValueError(
            f"window shape must be in [0, 1] (0 Gaussian, 1 rectangular); "
            f"got {s}"
        )
    return s


def _abs_idx(idx, n_attr):
    a = idx if idx >= 0 else n_attr + idx
    if not (0 <= a < n_attr):
        raise ValueError(f"window_attr {idx} out of range for {n_attr} attributes")
    return a


def _axis_values(p_attr, axis):
    """Flat finite values on the window axis (a K=1 attribute carries one
    value per event)."""
    M = np.asarray(p_attr[axis], dtype=float)
    v = M.ravel()
    return v[np.isfinite(v)]


def _resolve_centres(p_attr, axis, centres, start, stop, step, default_step):
    if centres is not None:
        if start is not None or stop is not None or step is not None:
            raise ValueError(
                "pass either `centres` or `start`/`stop`/`step`, not both"
            )
        return np.asarray(centres, dtype=float).ravel()
    vals = _axis_values(p_attr, axis)
    if vals.size == 0:
        raise ValueError("cannot derive a sweep range: window axis has no finite values")
    lo = float(vals.min()) if start is None else float(start)
    hi = float(vals.max()) if stop is None else float(stop)
    st = float(default_step) if step is None else float(step)
    if not (st > 0):
        raise ValueError("step must be positive")
    n = int(np.floor((hi - lo) / st + 1e-9)) + 1
    return lo + st * np.arange(max(n, 1))


def _translate_to(p_attr, w, axis, centre, specs=None):
    """Translate the carrier on `axis` so the axis mean lands at `centre`.

    Returns ``(p_attr, w, specs)``; the third value is the carrier's specs
    threaded through :func:`translate_attributes` unchanged (translation does
    not alter nesting). It is ``None``-derived (flat) when ``specs is None``.
    """
    mu = float(np.nanmean(np.asarray(p_attr[axis], dtype=float)))
    offsets = [None] * len(p_attr)
    offsets[axis] = np.array([[centre - mu]], dtype=float)
    pt, wt, st = translate_attributes(p_attr, w, offsets, specs=specs)
    return pt, wt, st


def _prune_dead_carrier(p_attr, w, specs):
    """Drop events the window hard-zeroed, before the (heavy) build.

    A windowed carrier carries out-of-window / beyond-truncation events
    at weight zero (``weight_events`` writes its factor as such). Those
    events contribute nothing to any inner product or to the density an
    entropy integrates, so dropping them here -- at the single windowing
    seam, before ``build_exp_tens`` runs its eager feasibility scan and
    r-ad enumeration over every column -- is exact and saves the bulk of
    a sliding sweep's cost, without touching the core build contract or
    the density-level ``pruned()`` path. The liveness rule is the shared
    one (``_weight_is_live``): an event is live iff every weighted
    attribute has a finite, nonzero slot in its column. Skipped when
    nothing is dead (the un-windowed common case pays only a mask scan)
    or when everything is dead (an empty window keeps its existing path).
    """
    if not p_attr or not isinstance(w, (list, tuple)):
        return p_attr, w, specs
    N = int(np.asarray(p_attr[0]).shape[1])
    if N == 0:
        return p_attr, w, specs
    live = np.ones(N, dtype=bool)
    for W in w:
        if W is None:
            continue
        Wa = np.asarray(W)
        if Wa.ndim == 1:
            Wa = Wa.reshape(1, -1)
        if Wa.ndim != 2 or Wa.shape[1] != N:
            continue                      # per-slot / scalar: cannot kill an event alone
        live &= _weight_is_live(Wa).any(axis=0)
    n_live = int(live.sum())
    if n_live == N or n_live == 0:
        return p_attr, w, specs
    keep = np.nonzero(live)[0]
    p_out = [np.asarray(P)[:, keep] for P in p_attr]
    w_out = []
    for W in w:
        if W is None:
            w_out.append(None)
            continue
        Wa = np.asarray(W)
        if Wa.ndim == 2 and Wa.shape[1] == N:
            w_out.append(Wa[:, keep])
        elif Wa.ndim == 1 and Wa.shape[0] == N:
            w_out.append(Wa[keep])
        else:
            w_out.append(W)               # per-slot / scalar: unchanged
    return p_out, w_out, specs            # specs are per-slot -> unchanged


def _window_at(p_attr, w, axis, target, centre, shape, width, *,
               drop_window_attr, specs=None):
    """Window the carrier at `centre`; returns ``(p_attr, w, specs)`` with the
    threaded-through (and, under ``drop_window_attr``, axis-pruned) specs.

    Events the window hard-zeroes are dropped before return so the build
    only sees in-window events (see :func:`_prune_dead_carrier`); this is
    exact and is the single seam every windowing function shares."""
    pt, wt, st = weight_events(
        p_attr, w, axis, target, float(centre), shape,
        width=width, is_per=False, period=0.0,
        drop_input_attr=drop_window_attr, specs=specs,
    )
    return _prune_dead_carrier(pt, wt, st)


def _as_query_batch(pqs):
    """Normalise :func:`translate_attributes`' query output to a list of
    carriers. A batched translate (T > 1) returns a length-T list of
    length-A carriers (``pqs[0]`` is itself a list); a single translate
    returns one length-A carrier (``pqs[0]`` is an array)."""
    if pqs and isinstance(pqs[0], (list, tuple)):
        return list(pqs)
    return [pqs]


def _similarity_finalise(ctx_transformed, query_transformed, sigma, r,
                         is_rel, is_per, period, normalize, truncation_sigmas):
    """Finalised windowed similarity for one sweep step. This is the single
    seam at which the cosine/one-sided continuum (alpha) will later drop in:
    it currently delegates to the `normalize` keyword of
    ``cos_sim_exp_tens`` ('oneSidedDenom' = alpha 0, 'cosine' = alpha 1/2);
    the alpha form will instead form num / (ss_ctx**alpha * ss_q**(1-alpha))
    from three raw inner products here, leaving every caller unchanged."""
    pc, wc = ctx_transformed
    pq, wq = query_transformed
    return float(cos_sim_exp_tens(
        pc, wc, pq, wq, sigma, r, is_rel, is_per, period,
        normalize=normalize, truncation_sigmas=truncation_sigmas,
        verbose=False,
    ))


def windowed_similarity(
    p_context, w_context, p_query, w_query,
    sigma, r, is_rel, is_per, period,
    centres=None, *,
    start=None, stop=None, step=None,
    query_centres=None,
    context_window=("rect", None),
    query_window=None,
    target_attr=None,
    normalize="oneSidedDenom",
    window_attr=-1,
    drop_window_attr,
    truncation_sigmas=None,
    specs=None,
    verbose=True,
):
    """Sliding pre-MAET similarity profile (cross-correlation).

    At each sweep centre the context is placed by `context_window` and the
    query by `query_window`, and their finalised inner product is recorded.
    With the defaults the context is windowed by a rectangle of the query's
    extent and the query is translated to the same centre (the locked
    template sweep). Pass `query_centres` to decouple the query's placement
    from the context's (e.g. a lag sweep). A `window` of ``None`` translates
    that operand whole; ``(shape, width)`` windows it.

    Nested carriers. Pass `specs` (a length-A list, exactly as returned by
    :func:`bind_events` and consumed by :func:`build_exp_tens`) to score
    nested attributes -- bound super-events, spectral inner multisets, the
    ``rel = 1`` transposition quotient, and so on. The per-attribute
    geometry (`r`, `sym`, `rel`, including any nested levels) is then read
    from `specs`; the positional `r`/`is_rel` are unused and `sigma`,
    `is_per`, `period` continue to supply the kernel widths and periodicity
    that `specs` does not carry. With ``specs=None`` (the default) the carrier
    is flat and every result is bit-identical to before -- the nesting is
    purely additive.
    """
    p_context = list(p_context)
    p_query = list(p_query)
    n_attr = len(p_context)
    axis = _abs_idx(window_attr, n_attr)
    if target_attr is None:
        target = 0 if axis != 0 else (1 if n_attr > 1 else 0)
    else:
        target = _abs_idx(target_attr, n_attr)
    if target == axis:
        raise ValueError("target_attr must differ from window_attr")

    # Context window. A ``None`` shape translates the context whole; a
    # concrete shape windows it. The width (used as the sweep-step default
    # and, when windowing, as the window support) defaults to the query's
    # extent on the axis. The shape is resolved only when windowing, so the
    # translate path never passes ``None`` through ``_resolve_shape``.
    cw_shape_raw, cw_width = context_window
    translate_context = cw_shape_raw is None
    if cw_width is None:
        qv = _axis_values(p_query, axis)
        cw_width = float(qv.max() - qv.min()) if qv.size else 0.0
    if not translate_context:
        cw_shape = _resolve_shape(cw_shape_raw)

    ctx_centres = _resolve_centres(p_context, axis, centres, start, stop, step,
                                   default_step=cw_width)
    A = ctx_centres.shape[0]
    # Per-row query centres: row a holds the trailing-axis query placements
    # scored against the single context placed at ctx_centres[a]. Output
    # shape follows `query_centres` (context broadcast along its trailing
    # axis): None -> locked (query at each context centre), 1-D length A ->
    # element-wise paired, 2-D (A, T) -> grid R[a, t].
    if query_centres is None:
        out_shape = (A,)
        q_rows = ctx_centres.reshape(A, 1)
    else:
        qc = np.asarray(query_centres, dtype=float)
        if qc.ndim == 1:
            if qc.shape[0] != A:
                raise ValueError(
                    f"1-D query_centres must match the context length A={A}; "
                    f"got {qc.shape[0]}. For a grid pass a 2-D (A, T) array."
                )
            out_shape = (A,)
            q_rows = qc.reshape(A, 1)
        elif qc.ndim == 2:
            if qc.shape[0] != A:
                raise ValueError(
                    f"2-D query_centres must have first axis == context "
                    f"length A={A}; got {qc.shape}."
                )
            out_shape = qc.shape
            q_rows = qc
        else:
            raise ValueError("query_centres must be None, 1-D, or 2-D")

    nested = specs is not None

    if drop_window_attr and n_attr < 2:
        raise ValueError(
            "drop_window_attr=True drops the window axis from the comparison, so "
            "the carrier must have at least two attributes."
        )

    def _drop_window_axis(p, w, sp):
        keep = [i for i in range(n_attr) if i != axis]
        p2 = [p[i] for i in keep]
        if isinstance(w, (list, tuple)) and len(w) == n_attr:
            w2 = [w[i] for i in keep]
        else:
            w2 = w
        sp2 = [sp[i] for i in keep] if sp is not None else None
        return p2, w2, sp2

    def _drop_seq(seq):
        if (drop_window_attr and isinstance(seq, (list, tuple, np.ndarray))
                and len(seq) == n_attr):
            return [seq[i] for i in range(n_attr) if i != axis]
        return seq

    # When the window axis is dropped it is no longer a compared dimension, so
    # the per-attribute kernel geometry collapses to the retained attributes.
    sigma_b, is_per_b, period_b = (_drop_seq(sigma), _drop_seq(is_per),
                                   _drop_seq(period))
    r_b, is_rel_b = _drop_seq(r), _drop_seq(is_rel)

    def place_context(centre):
        if translate_context:
            pc, wc, sc = _translate_to(p_context, w_context, axis, centre,
                                       specs=specs)
            return _drop_window_axis(pc, wc, sc) if drop_window_attr else (pc, wc, sc)
        return _window_at(p_context, w_context, axis, target, centre,
                          cw_shape, cw_width, drop_window_attr=drop_window_attr,
                          specs=specs)

    translate_query = query_window is None
    if not translate_query:
        qw_shape = _resolve_shape(query_window[0])
        qw_width = query_window[1]
        mu_q = None
    else:
        mu_q = float(np.nanmean(np.asarray(p_query[axis], dtype=float)))

    # With the window axis dropped the query carries no compared placement
    # along it, so it reduces to a single fixed template built once.
    dq_fixed = None
    if drop_window_attr:
        pq_k, wq_k, sq_k = _drop_window_axis(p_query, w_query, specs)
        dq_fixed = (build_exp_tens(pq_k, wq_k, sigma=sigma_b, is_per=is_per_b,
                                   period=period_b, specs=sq_k, verbose=False)
                    if nested else (pq_k, wq_k))

    out = np.empty((A, q_rows.shape[1]), dtype=float)
    for a in range(A):
        pc, wc, sc = place_context(ctx_centres[a])
        # Nested: build the windowed-context density once per centre; the
        # geometry rides in `specs`, so the positional r/is_rel are unused.
        dc = (build_exp_tens(pc, wc, sigma=sigma_b, is_per=is_per_b,
                             period=period_b, specs=sc, verbose=False)
              if nested else None)
        if drop_window_attr:
            if nested:
                val = float(cos_sim_exp_tens(
                    dc, dq_fixed, normalize=normalize,
                    truncation_sigmas=truncation_sigmas, verbose=False))
            else:
                pq_k, wq_k = dq_fixed
                val = float(cos_sim_exp_tens(
                    pc, wc, pq_k, wq_k, sigma_b, r_b, is_rel_b, is_per_b,
                    period_b, normalize=normalize,
                    truncation_sigmas=truncation_sigmas, verbose=False))
            out[a, :] = val
            continue
        row = q_rows[a]
        if translate_query:
            # One translate produces all T shifted query copies; one
            # cos_sim scores the single context against the batch.
            offsets = [None] * n_attr
            offsets[axis] = (row - mu_q).reshape(1, row.size)
            pqs, wqs, sqs = translate_attributes(p_query, w_query, offsets,
                                                 specs=specs)
            if nested:
                dq = [build_exp_tens(qc, wqs, sigma=sigma, is_per=is_per,
                                     period=period, specs=sqs, verbose=False)
                      for qc in _as_query_batch(pqs)]
                vals = cos_sim_exp_tens(
                    dc, dq if len(dq) != 1 else dq[0],
                    normalize=normalize, truncation_sigmas=truncation_sigmas,
                    verbose=False)
            else:
                vals = cos_sim_exp_tens(
                    pc, wc, pqs, wqs, sigma, r, is_rel, is_per, period,
                    normalize=normalize, truncation_sigmas=truncation_sigmas,
                    verbose=False)
            out[a, :] = np.atleast_1d(np.asarray(vals, dtype=float)).ravel()
        else:
            for t, qcen in enumerate(row):
                pq, wq, sq = _window_at(p_query, w_query, axis, target, qcen,
                                        qw_shape, qw_width, drop_window_attr=False,
                                        specs=specs)
                if nested:
                    dq = build_exp_tens(pq, wq, sigma=sigma, is_per=is_per,
                                        period=period, specs=sq, verbose=False)
                    out[a, t] = float(cos_sim_exp_tens(
                        dc, dq, normalize=normalize,
                        truncation_sigmas=truncation_sigmas, verbose=False))
                else:
                    out[a, t] = _similarity_finalise(
                        (pc, wc), (pq, wq), sigma, r, is_rel, is_per, period,
                        normalize, truncation_sigmas)
    return out.reshape(out_shape)


def windowed_entropy(
    p, w, sigma, r, is_rel, is_per, period,
    centres=None, *,
    start=None, stop=None, step=None,
    window=("rect", None),
    method="differential",
    target_attr=None,
    window_attr=-1,
    drop_window_attr,
    marginalise=None,
    truncation_sigmas=None,
    specs=None,
    base=2.0,
    verbose=True,
):
    """Sliding pre-MAET entropy profile.

    At each sweep centre the carrier is windowed on `window_attr` and the
    entropy of the resulting density is recorded. `drop_window_attr` is
    required and fixes the structural role of the window axis: with ``True``
    it is a placement coordinate only and is removed from the density (it
    must then be an ``r = 1`` attribute, for which deletion equals
    marginalisation); with ``False`` it is retained as a dimension of the
    density whose entropy is taken. `marginalise` (default ``None``) is the
    separate, general operation of integrating a *retained* axis out of the
    density before the entropy is taken; it is not yet implemented, and
    naming the dropped axis in it is an error. The window `width` has no
    default (there is no query to borrow from) and must be supplied.

    Nested carriers. Pass `specs` (a length-A list, as returned by
    :func:`bind_events`) to take the entropy of a density over nested
    attributes; the per-attribute geometry (`r`, `sym`, `rel`, nested
    levels) is then read from `specs` and the positional `r`/`is_rel` supply
    only the window-axis order used by the deletion guard. With
    ``specs=None`` the carrier is flat and every result is unchanged.
    """
    p = list(p)
    n_attr = len(p)
    axis = _abs_idx(window_attr, n_attr)
    if target_attr is None:
        target = 0 if axis != 0 else (1 if n_attr > 1 else 0)
    else:
        target = _abs_idx(target_attr, n_attr)
    if target == axis:
        raise ValueError("target_attr must differ from window_attr")

    w_shape = _resolve_shape(window[0])
    w_width = window[1]
    if w_width is None:
        raise ValueError(
            "windowed_entropy requires an explicit window width "
            "(window=(shape, width)); there is no query to default it from"
        )

    # The window axis is either dropped (placement only, removed from the
    # density) or retained as a compared dimension; the caller must state
    # which. Dropping equals marginalisation only at r = 1.
    if drop_window_attr and int(np.atleast_1d(r)[axis]) != 1:
        raise ValueError(
            "drop_window_attr=True requires the window axis to be an r = 1 "
            "attribute; for r >= 2 deletion does not equal marginalisation"
        )

    # `marginalise` is the separate, general integrate-out operation over a
    # retained axis (not yet implemented). A dropped axis is already gone, so
    # it cannot also be marginalised.
    marg = set() if marginalise is None else {
        _abs_idx(a, n_attr) for a in np.atleast_1d(marginalise)
    }
    if drop_window_attr and axis in marg:
        raise ValueError(
            "the window axis is dropped (drop_window_attr=True), so it cannot "
            "also appear in marginalise"
        )
    if marg:
        raise NotImplementedError(
            "marginalise (integrating a retained axis out of the density) is "
            f"not yet implemented; got axes {sorted(marg)}"
        )

    # Specs for the density built after the (possibly axis-deleting) window.
    def _kept(seq):
        s = list(seq)
        return [s[k] for k in range(n_attr)
                if not (drop_window_attr and k == axis)]
    sig_k, r_k, rel_k, per_k, pd_k = (
        _kept(sigma), _kept(r), _kept(is_rel), _kept(is_per), _kept(period))

    ctr = _resolve_centres(p, axis, centres, start, stop, step,
                           default_step=w_width)

    # Lazy import to avoid a build-time cycle (entropy imports from _tensor).
    from ..entropy import entropy_exp_tens

    nested = specs is not None
    H = np.empty(ctr.shape[0], dtype=float)
    for i, c in enumerate(ctr):
        pw, ww, sw = _window_at(p, w, axis, target, c, w_shape, w_width,
                                drop_window_attr=drop_window_attr, specs=specs)
        if nested:
            # Geometry rides in the (axis-pruned) specs; r/is_rel unused.
            dens = build_exp_tens(pw, ww, sigma=sig_k, is_per=per_k,
                                  period=pd_k, specs=sw, verbose=False)
        else:
            dens = build_exp_tens(pw, ww, sig_k, r_k, rel_k, per_k, pd_k,
                                  verbose=False)
        H[i] = float(entropy_exp_tens(dens, method=method, base=base,
                                      verbose=False))
    return H
