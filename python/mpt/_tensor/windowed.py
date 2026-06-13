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


def _translate_to(p_attr, w, axis, centre):
    """Translate the carrier on `axis` so the axis mean lands at `centre`."""
    mu = float(np.nanmean(np.asarray(p_attr[axis], dtype=float)))
    offsets = [None] * len(p_attr)
    offsets[axis] = np.array([[centre - mu]], dtype=float)
    pt, wt, _ = translate_attributes(p_attr, w, offsets)
    return pt, wt


def _window_at(p_attr, w, axis, target, centre, shape, width, *,
               delete_input):
    pt, wt, _ = weight_events(
        p_attr, w, axis, target, float(centre), shape,
        width=width, is_per=False, period=0.0,
        delete_input=delete_input,
    )
    return pt, wt


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
    truncation_sigmas=None,
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

    def place_context(centre):
        if translate_context:
            return _translate_to(p_context, w_context, axis, centre)
        return _window_at(p_context, w_context, axis, target, centre,
                          cw_shape, cw_width, delete_input=False)

    translate_query = query_window is None
    if not translate_query:
        qw_shape = _resolve_shape(query_window[0])
        qw_width = query_window[1]
        mu_q = None
    else:
        mu_q = float(np.nanmean(np.asarray(p_query[axis], dtype=float)))

    out = np.empty((A, q_rows.shape[1]), dtype=float)
    for a in range(A):
        pc, wc = place_context(ctx_centres[a])
        row = q_rows[a]
        if translate_query:
            # One translate produces all T shifted query copies; one
            # cos_sim scores the single context against the batch.
            offsets = [None] * n_attr
            offsets[axis] = (row - mu_q).reshape(1, row.size)
            pqs, wqs, _ = translate_attributes(p_query, w_query, offsets)
            vals = cos_sim_exp_tens(
                pc, wc, pqs, wqs, sigma, r, is_rel, is_per, period,
                normalize=normalize, truncation_sigmas=truncation_sigmas,
                verbose=False)
            out[a, :] = np.atleast_1d(np.asarray(vals, dtype=float)).ravel()
        else:
            for t, qcen in enumerate(row):
                pq, wq = _window_at(p_query, w_query, axis, target, qcen,
                                    qw_shape, qw_width, delete_input=False)
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
    marginalise=None,
    truncation_sigmas=None,
    base=2.0,
    verbose=True,
):
    """Sliding pre-MAET entropy profile.

    At each sweep centre the carrier is windowed on `window_attr` and the
    entropy of the resulting density is recorded. Name axes in `marginalise`
    (default ``None``) to integrate them out before the entropy is taken;
    only the window axis may be marginalised at present, and it must be an
    ``r = 1`` attribute (absolute or periodic), for which deletion from the
    carrier equals marginalisation of the density. The window `width` has no
    default (there is no query to borrow from) and must be supplied.
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

    # Marginalisation: only the window axis, and only if it is r = 1.
    marg = set() if marginalise is None else {
        _abs_idx(a, n_attr) for a in np.atleast_1d(marginalise)
    }
    extra = marg - {axis}
    if extra:
        raise NotImplementedError(
            "windowed_entropy currently marginalises only the window axis; "
            f"marginalising other axes {sorted(extra)} is not yet supported"
        )
    delete_axis = axis in marg
    if delete_axis and int(np.atleast_1d(r)[axis]) != 1:
        raise ValueError(
            "marginalising the window axis requires it to be an r = 1 "
            "attribute; for r >= 2 deletion does not equal marginalisation"
        )

    # Specs for the density built after the (possibly axis-deleting) window.
    def _kept(seq):
        s = list(seq)
        return [s[k] for k in range(n_attr) if not (delete_axis and k == axis)]
    sig_k, r_k, rel_k, per_k, pd_k = (
        _kept(sigma), _kept(r), _kept(is_rel), _kept(is_per), _kept(period))

    ctr = _resolve_centres(p, axis, centres, start, stop, step,
                           default_step=w_width)

    # Lazy import to avoid a build-time cycle (entropy imports from _tensor).
    from ..entropy import entropy_exp_tens

    H = np.empty(ctr.shape[0], dtype=float)
    for i, c in enumerate(ctr):
        pw, ww = _window_at(p, w, axis, target, c, w_shape, w_width,
                            delete_input=delete_axis)
        dens = build_exp_tens(pw, ww, sig_k, r_k, rel_k, per_k, pd_k,
                              verbose=False)
        H[i] = float(entropy_exp_tens(dens, method=method, base=base,
                                      verbose=False))
    return H
