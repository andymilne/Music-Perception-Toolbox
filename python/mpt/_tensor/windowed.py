"""Pre-MAET windowed sweeps: ``windowed_similarity`` and ``windowed_entropy``.

Both slide a window across one or more attribute axes of a *pre-MAET*
carrier and read out a profile, sharing one placement-and-window seam.
``windowed_similarity`` translates a query to each swept position and
scores it against the locally windowed context; ``windowed_entropy`` has
no query and reads the entropy of the windowed (and, for any dropped axis,
axis-reduced) density.

Each function offers two equivalent argument surfaces:

* **Single axis** (the common case): ``window_attr`` names the swept axis,
  ``centres`` (or ``start`` / ``stop`` / ``step``) its positions, and
  ``drop_window_attr`` whether that axis is compared (``False``) or only
  places the comparison (``True``). ``query_centres`` decouples the query's
  placement from the window's, which a 2-D ``(A, T)`` array turns into the
  lagged correlogram surface.
* **Multiple axes**: ``sweep={axis: positions, ...}`` with a parallel
  ``drop={axis: bool, ...}`` gives one output dimension per swept axis.

A window is centred on the query's ``locate`` value (the multiset centroid
by default; also ``'start'`` / ``'end'`` / ``'mid'`` or a callable) and, by
default, is a rectangle of the query's extent; ``context_window`` overrides
its shape and width. The per-axis window factors multiply onto one
``target_attr`` and prune the context to the swept box before the build, so
a single bundled time (or pitch) attribute can be swept, windowed, and
compared at once, with no separate locating copy.
"""

from __future__ import annotations

import numpy as np

from .preprocessing import (
    weight_events, translate_attributes,
    _evaluate_shape, _multiply_weights, _normalise_weights_to_list,
)
from .build import build_exp_tens
from .cosine import cos_sim_exp_tens
from .density import _weight_is_live

_SQRT12 = 2.0 * np.sqrt(3.0)

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
    attribute has a finite, nonzero slot in its column. Skipped only when
    nothing is dead (the un-windowed common case pays only a mask scan).
    An all-dead window (one that caught nothing) prunes to zero events, so
    the build is trivial and the resulting empty density scores zero at the
    comparison rather than paying a full build over zeroed events.
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
    if n_live == N:
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


def _locate_row(M, locate):
    """Reduce a ``(K, N)`` attribute to the ``(1, N)`` value its window
    centres on: the multiset centroid by default, else ``'start'`` / ``'end'``
    / ``'mid'`` / a callable. Raw values for a relative axis (the window
    selects in absolute position; the comparison is translation-invariant)."""
    M = np.asarray(M, dtype=float)
    if callable(locate):
        return np.asarray(locate(M), dtype=float).reshape(1, -1)
    if locate == "centroid":
        return np.nanmean(M, axis=0, keepdims=True)
    if locate == "start":
        return M[0:1, :]
    if locate == "end":
        return M[-1:, :]
    if locate == "mid":
        return 0.5 * (M[0:1, :] + M[-1:, :])
    raise ValueError(
        f"locate must be 'centroid', 'start', 'end', 'mid', or a callable; "
        f"got {locate!r}")


def _resolve_locate(locate, axis):
    return locate.get(axis, "centroid") if isinstance(locate, dict) else locate


def _window_factor(loc_row, centre, gamma, sd):
    """Per-event window factor over a reduced locating row, matching the
    ``weight_events`` profile and (global-default) truncation exactly."""
    from .._defaults import get_default
    delta = loc_row - centre
    factor = _evaluate_shape(delta, sd, gamma)
    trunc = get_default("truncation_sigmas")
    if np.isfinite(trunc):
        factor = factor.copy()
        factor[np.abs(delta) > trunc * sd] = 0.0
    return factor


def _query_extent(p_query, axis):
    v = np.asarray(p_query[axis], dtype=float).ravel()
    v = v[np.isfinite(v)]
    return float(v.max() - v.min()) if v.size else 0.0


def _resolve_window(spec, query_extent, axis):
    """``(gamma, sd)`` for one swept axis; default is a rectangle of the
    query's extent on that axis."""
    if spec is None:
        if query_extent <= 0:
            raise ValueError(
                f"axis {axis}: the query has zero extent there, so the default "
                f"window width is undefined; give 'width' or 'sd' in "
                f"context_window[{axis}].")
        return 1.0, query_extent / _SQRT12
    gamma = _resolve_shape(spec.get("shape", "rect"))
    has_sd, has_w = "sd" in spec, "width" in spec
    if has_sd == has_w:
        raise ValueError(
            f"axis {axis}: give exactly one of 'width' or 'sd' in "
            f"context_window[{axis}].")
    sd = float(spec["sd"]) if has_sd else float(spec["width"]) / _SQRT12
    if not (sd > 0):
        raise ValueError(f"axis {axis}: window width/sd must be > 0.")
    return gamma, sd


def _axis_is_rel(specs, is_rel, a):
    if specs is not None:
        sp = specs[a]
        if isinstance(sp, dict):
            rel = sp.get("rel", False)
            if isinstance(rel, (list, tuple, np.ndarray)):
                return bool(rel[-1]) if len(rel) else False
            return bool(rel)
        return False
    return bool(is_rel[a]) if a < len(is_rel) else False


def _apply_windows(p, w, specs, centres, win, locate, target):
    w_out = _normalise_weights_to_list(w, len(p))
    for a, centre in centres.items():
        loc = _locate_row(p[a], _resolve_locate(locate, a))
        gamma, sd = win[a]
        factor = _window_factor(loc, centre, gamma, sd)
        w_out[target] = _multiply_weights(w_out[target], factor, target)
    return _prune_dead_carrier(p, w_out, specs)


def _drop_axes(p, w, specs, drop_axes):
    keep = [i for i in range(len(p)) if i not in drop_axes]
    p2 = [p[i] for i in keep]
    if isinstance(w, (list, tuple)) and len(w) == len(p):
        w2 = [w[i] for i in keep]
    else:
        w2 = w
    s2 = [specs[i] for i in keep] if specs is not None else None
    return p2, w2, s2, keep


def _sub(seq, keep):
    if isinstance(seq, (list, tuple, np.ndarray)) and len(seq) >= max(keep) + 1:
        return [seq[i] for i in keep]
    return seq


def _prep_sweep(p_context, p_query, sweep, drop, context_window, target_attr,
                require_window=False):
    n = len(p_context)
    keys = list(sweep.keys())
    if not keys:
        raise ValueError("`sweep` must name at least one attribute to slide.")
    if set(drop.keys()) != set(keys):
        raise ValueError("`drop` must have exactly one entry per `sweep` key.")
    drop_axes = {a for a in keys if drop[a]}
    kept = [i for i in range(n) if i not in drop_axes]
    if not kept:
        raise ValueError(
            "every attribute is dropped; nothing is left to compare or to take "
            "the entropy of.")
    if target_attr is None:
        target = kept[0]
    else:
        target = target_attr if target_attr >= 0 else n + target_attr
        if target in drop_axes:
            raise ValueError(
                f"target_attr={target} is a dropped axis; its weights are "
                f"removed before the build, so the window factors would be "
                f"lost. Choose a compared attribute.")
    cw = context_window if isinstance(context_window, dict) else {}
    if require_window and any(cw.get(a) is None for a in keys):
        raise ValueError(
            "windowed_entropy has no query to size the window; give an explicit "
            "context_window entry (width or sd) for every swept axis.")
    win = {a: _resolve_window(cw.get(a), _query_extent(p_query, a), a)
           for a in keys}
    grids = [np.asarray(sweep[a], dtype=float).ravel() for a in keys]
    return keys, drop_axes, target, win, grids


def _single_window(context_window, p_query, axis):
    """(gamma, sd) for the single-axis path from the ``(shape, width)`` tuple;
    width defaults to the query's extent on the axis."""
    shape_raw, width = context_window
    if width is None:
        width = _query_extent(p_query, axis)
    if not (width > 0):
        raise ValueError(
            f"window width on axis {axis} is zero or undefined; pass "
            f"context_window=(shape, width) with width > 0.")
    gamma = _resolve_shape("rect" if shape_raw is None else shape_raw)
    return gamma, width / _SQRT12


def windowed_similarity(p_context, w_context, p_query, w_query,
                        sigma, r, is_rel, is_per, period,
                        centres=None, *, start=None, stop=None, step=None,
                        query_centres=None, context_window=("rect", None),
                        query_window=None, window_attr=-1, drop_window_attr=None,
                        sweep=None, drop=None, locate="centroid",
                        target_attr=None, normalize="oneSidedDenom", specs=None,
                        verbose=False):
    r"""Slide a query across a context and measure their similarity at each
    position (a pre-MAET cross-correlation).

    Two equivalent argument surfaces:

    * **Single axis** (the common case): name the swept axis with
      ``window_attr`` and its positions with ``centres`` (or ``start`` /
      ``stop`` / ``step``); ``drop_window_attr`` says whether that axis is
      compared (``False``) or only places the comparison (``True``). The
      window is centred on the query's ``locate`` value (the multiset
      centroid by default) and sized to the query's extent unless
      ``context_window=(shape, width)`` overrides it. ``query_centres``
      decouples the query's placement from the window's: ``None`` locks them
      (ordinary cross-correlation); a 2-D ``(A, T)`` array fixes the window at
      each ``centres[a]`` while the query slides across ``query_centres[a]``,
      giving the lagged correlogram surface.
    * **Multiple axes**: give ``sweep={axis: positions, ...}`` and a parallel
      ``drop={axis: bool, ...}``; the output gains one dimension per swept
      axis (Cartesian product), each axis windowed and, by default, query-
      locked. ``context_window`` is then a per-axis ``dict``.

    ``target_attr`` is the attribute whose weights absorb the window factors
    (default: the first compared attribute; it may coincide with a swept
    axis). ``specs`` carries nested geometry from :func:`bind_events`.
    """
    if sweep is not None:
        if drop is None:
            raise ValueError("multi-axis `sweep` requires a parallel `drop`.")
        if query_centres is not None or query_window is not None:
            raise ValueError(
                "`query_centres`/`query_window` are single-axis arguments; "
                "the multi-axis `sweep` form locks the query to the sweep.")
        return _ws_multi(
            p_context, w_context, p_query, w_query, sigma, r, is_rel, is_per,
            period, sweep, drop, context_window if isinstance(context_window, dict)
            else None, locate, normalize, target_attr, specs)
    if drop_window_attr is None:
        raise ValueError(
            "`drop_window_attr` is required (True places only, False compares).")
    return _ws_single(
        p_context, w_context, p_query, w_query, sigma, r, is_rel, is_per, period,
        centres, start, stop, step, query_centres, context_window, query_window,
        window_attr, drop_window_attr, locate, target_attr, normalize, specs)


def _ws_multi(p_context, w_context, p_query, w_query, sigma, r, is_rel, is_per,
              period, sweep, drop, context_window, locate, normalize,
              target_attr, specs):
    p_context, p_query = list(p_context), list(p_query)
    keys, drop_axes, target, win, grids = _prep_sweep(
        p_context, p_query, sweep, drop, context_window, target_attr)
    nested = specs is not None
    out = np.empty(tuple(g.size for g in grids), dtype=float)
    for idx in np.ndindex(*out.shape):
        centres = {keys[j]: float(grids[j][idx[j]]) for j in range(len(keys))}
        offs = [None] * len(p_query)
        for a in keys:
            if a in drop_axes or _axis_is_rel(specs, is_rel, a):
                continue
            q_loc = float(np.nanmean(_locate_row(
                p_query[a], _resolve_locate(locate, a))))
            offs[a] = np.array([[centres[a] - q_loc]], dtype=float)
        if any(o is not None for o in offs):
            pq_t, wq_t, sq_t = translate_attributes(p_query, w_query, offs,
                                                     specs=specs)
        else:
            pq_t, wq_t, sq_t = p_query, w_query, specs
        pc_w, wc_w, sc_w = _apply_windows(p_context, w_context, specs, centres,
                                          win, locate, target)
        pc, wc, sc, keep = _drop_axes(pc_w, wc_w, sc_w, drop_axes)
        pq, wq, sq, _ = _drop_axes(pq_t, wq_t, sq_t, drop_axes)
        sg, rr, rl, pr, pd = (_sub(sigma, keep), _sub(r, keep), _sub(is_rel, keep),
                              _sub(is_per, keep), _sub(period, keep))
        if nested:
            dc = build_exp_tens(pc, wc, sigma=sg, is_per=pr, period=pd, specs=sc,
                                verbose=False)
            dq = build_exp_tens(pq, wq, sigma=sg, is_per=pr, period=pd, specs=sq,
                                verbose=False)
            out[idx] = float(cos_sim_exp_tens(
                dc, dq, normalize=normalize,
                verbose=False))
        else:
            out[idx] = float(cos_sim_exp_tens(
                pc, wc, pq, wq, sg, rr, rl, pr, pd, normalize=normalize,
                verbose=False))
    return out


def _ws_single(p_context, w_context, p_query, w_query, sigma, r, is_rel, is_per,
               period, centres, start, stop, step, query_centres, context_window,
               query_window, window_attr, drop_window_attr, locate, target_attr,
               normalize, specs):
    p_context, p_query = list(p_context), list(p_query)
    n = len(p_context)
    axis = _abs_idx(window_attr, n)
    nested = specs is not None
    gamma, sd = _single_window(context_window, p_query, axis)
    win = {axis: (gamma, sd)}
    drop_axes = {axis} if drop_window_attr else set()
    keep = [i for i in range(n) if i not in drop_axes]
    if not keep:
        raise ValueError("dropping the only attribute leaves nothing to compare.")
    target = (keep[0] if target_attr is None else _abs_idx(target_attr, n))
    if target in drop_axes:
        raise ValueError(
            f"target_attr={target} is the dropped axis; its weights are removed "
            f"before the build. Choose a compared attribute.")
    ctx_centres = _resolve_centres(p_context, axis, centres, start, stop, step,
                                   default_step=gamma and sd * _SQRT12 or sd)
    A = ctx_centres.size
    if query_centres is None:
        q_rows, out_shape = ctx_centres.reshape(A, 1), (A,)
    else:
        qc = np.asarray(query_centres, dtype=float)
        if qc.ndim == 1:
            if qc.shape[0] != A:
                raise ValueError(
                    f"1-D query_centres must have length A={A}; got {qc.shape[0]}.")
            q_rows, out_shape = qc.reshape(A, 1), (A,)
        elif qc.ndim == 2:
            if qc.shape[0] != A:
                raise ValueError(
                    f"2-D query_centres must have first axis A={A}; got {qc.shape}.")
            q_rows, out_shape = qc, qc.shape
        else:
            raise ValueError("query_centres must be None, 1-D, or 2-D.")
    rel_axis = _axis_is_rel(specs, is_rel, axis)
    sg, rr, rl, pr, pd = (_sub(sigma, keep), _sub(r, keep), _sub(is_rel, keep),
                          _sub(is_per, keep), _sub(period, keep))
    out = np.empty((A, q_rows.shape[1]), dtype=float)
    for a in range(A):
        pc_w, wc_w, sc_w = _apply_windows(
            p_context, w_context, specs, {axis: float(ctx_centres[a])}, win,
            locate, target)
        pc, wc, sc, _ = _drop_axes(pc_w, wc_w, sc_w, drop_axes)
        dc = (build_exp_tens(pc, wc, sigma=sg, is_per=pr, period=pd, specs=sc,
                             verbose=False) if nested else None)
        for t in range(q_rows.shape[1]):
            if drop_window_attr or rel_axis:
                pq_t, wq_t, sq_t = p_query, w_query, specs
            else:
                q_loc = float(np.nanmean(_locate_row(
                    p_query[axis], _resolve_locate(locate, axis))))
                offs = [None] * n
                offs[axis] = np.array([[float(q_rows[a, t]) - q_loc]], dtype=float)
                pq_t, wq_t, sq_t = translate_attributes(p_query, w_query, offs,
                                                         specs=specs)
            pq, wq, sq, _ = _drop_axes(pq_t, wq_t, sq_t, drop_axes)
            if nested:
                dq = build_exp_tens(pq, wq, sigma=sg, is_per=pr, period=pd,
                                    specs=sq, verbose=False)
                out[a, t] = float(cos_sim_exp_tens(
                    dc, dq, normalize=normalize,
                    verbose=False))
            else:
                out[a, t] = float(cos_sim_exp_tens(
                    pc, wc, pq, wq, sg, rr, rl, pr, pd, normalize=normalize,
                    verbose=False))
    return out.reshape(out_shape)


def windowed_entropy(p_context, w_context, sigma, r, is_rel, is_per, period,
                     centres=None, *, start=None, stop=None, step=None,
                     context_window=("rect", None), window_attr=-1,
                     drop_window_attr=None, sweep=None, drop=None,
                     locate="centroid", method="differential", base=2.0,
                     marginalise=None, target_attr=None, specs=None,
                     verbose=False):
    r"""Slide a window across a context and read its entropy at each position.

    Shares the placement, window, ``locate``, and ``drop`` machinery of
    :func:`windowed_similarity`, with the same single-axis
    (``window_attr`` + ``centres`` + ``drop_window_attr``) and multi-axis
    (``sweep`` + ``drop``) surfaces; there is no query, so at each position
    the windowed (and, for a dropped axis, axis-reduced) density is built and
    its entropy taken. ``marginalise`` is reserved for integrating a retained
    axis out of the density and is not yet implemented.
    """
    if marginalise is not None:
        raise NotImplementedError(
            "marginalise (integrating a retained axis out of the density) is "
            "not yet implemented.")
    if sweep is not None:
        if drop is None:
            raise ValueError("multi-axis `sweep` requires a parallel `drop`.")
        return _we_multi(
            p_context, w_context, sigma, r, is_rel, is_per, period, sweep, drop,
            context_window if isinstance(context_window, dict) else None, locate,
            method, base, target_attr, specs)
    if drop_window_attr is None:
        raise ValueError(
            "`drop_window_attr` is required (True drops the window axis, False "
            "retains it).")
    return _we_single(
        p_context, w_context, sigma, r, is_rel, is_per, period, centres, start,
        stop, step, context_window, window_attr, drop_window_attr, locate,
        method, base, target_attr, specs)


def _we_multi(p_context, w_context, sigma, r, is_rel, is_per, period, sweep,
              drop, context_window, locate, method, base, target_attr, specs):
    p_context = list(p_context)
    keys, drop_axes, target, win, grids = _prep_sweep(
        p_context, p_context, sweep, drop, context_window, target_attr,
        require_window=True)
    nested = specs is not None
    from ..entropy import entropy_exp_tens
    out = np.empty(tuple(g.size for g in grids), dtype=float)
    for idx in np.ndindex(*out.shape):
        centres = {keys[j]: float(grids[j][idx[j]]) for j in range(len(keys))}
        pc_w, wc_w, sc_w = _apply_windows(p_context, w_context, specs, centres,
                                          win, locate, target)
        pc, wc, sc, keep = _drop_axes(pc_w, wc_w, sc_w, drop_axes)
        sg, rr, rl, pr, pd = (_sub(sigma, keep), _sub(r, keep), _sub(is_rel, keep),
                              _sub(is_per, keep), _sub(period, keep))
        if nested:
            dens = build_exp_tens(pc, wc, sigma=sg, is_per=pr, period=pd,
                                  specs=sc, verbose=False)
        else:
            dens = build_exp_tens(pc, wc, sg, rr, rl, pr, pd, verbose=False)
        out[idx] = float(entropy_exp_tens(dens, method=method, base=base,
                                          verbose=False))
    return out


def _we_single(p_context, w_context, sigma, r, is_rel, is_per, period, centres,
               start, stop, step, context_window, window_attr, drop_window_attr,
               locate, method, base, target_attr, specs):
    p_context = list(p_context)
    n = len(p_context)
    axis = _abs_idx(window_attr, n)
    nested = specs is not None
    shape_raw, width = context_window
    if width is None:
        raise ValueError(
            "windowed_entropy has no query to size the window; pass an explicit "
            "context_window=(shape, width).")
    if not (width > 0):
        raise ValueError("context_window width must be > 0.")
    gamma = _resolve_shape("rect" if shape_raw is None else shape_raw)
    win = {axis: (gamma, width / _SQRT12)}
    drop_axes = {axis} if drop_window_attr else set()
    keep = [i for i in range(n) if i not in drop_axes]
    if not keep:
        raise ValueError("dropping the only attribute leaves no density.")
    target = (keep[0] if target_attr is None else _abs_idx(target_attr, n))
    if target in drop_axes:
        raise ValueError(f"target_attr={target} is the dropped axis.")
    ctx_centres = _resolve_centres(p_context, axis, centres, start, stop, step,
                                   default_step=width)
    from ..entropy import entropy_exp_tens
    sg, rr, rl, pr, pd = (_sub(sigma, keep), _sub(r, keep), _sub(is_rel, keep),
                          _sub(is_per, keep), _sub(period, keep))
    out = np.empty(ctx_centres.size, dtype=float)
    for i, c in enumerate(ctx_centres):
        pc_w, wc_w, sc_w = _apply_windows(
            p_context, w_context, specs, {axis: float(c)}, win, locate, target)
        pc, wc, sc, _ = _drop_axes(pc_w, wc_w, sc_w, drop_axes)
        if nested:
            dens = build_exp_tens(pc, wc, sigma=sg, is_per=pr, period=pd,
                                  specs=sc, verbose=False)
        else:
            dens = build_exp_tens(pc, wc, sg, rr, rl, pr, pd, verbose=False)
        out[i] = float(entropy_exp_tens(dens, method=method, base=base,
                                        verbose=False))
    return out
