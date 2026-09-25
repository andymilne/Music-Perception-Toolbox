"""Pre-MAET windowed comparison and entropy: ``windowed_similarity`` and
``windowed_entropy``.

Both restrict a *pre-MAET* context to a local region by event weighting
(§3 of the article): a *window*, a non-negative profile centred at a value
``c``, multiplies each event's weights on one attribute (the target
attribute) by its value at the event's value on the *window attribute*,
the attribute the window is defined over. The window is placed at a series
of *centres*, values of the window attribute, and a profile is read out,
one value per centre. ``windowed_similarity`` compares a query with the
windowed context at each centre; ``windowed_entropy`` has no query and
reads the entropy of the windowed density, from which a window attribute
may first be marginalized by removing it from the pre-MAET.

Each function takes one window attribute (``window_attr``, with
``centres`` or ``start`` / ``stop`` / ``step`` and ``drop_window_attr``,
which says whether the window attribute is compared or marginalized) or
several (``sweep={a: centres, ...}`` with a parallel ``drop={a: bool,
...}``, ``a`` an attribute index), the output then having one dimension
per window attribute.

``locate`` reduces each event's element multiset on a window attribute to
the one value the window reads (the mean by default; also ``'start'`` /
``'end'`` / ``'mid'`` or a callable). By default the window is a rectangle
as wide as the query's extent on the window attribute; ``context_window``
overrides its shape and width. The window factors multiply onto one
``target_attr`` and prune the context to the windowed region before the
build, so a single bundled time (or pitch) attribute can be windowed and
compared at once, with no separate locating copy.
"""

from __future__ import annotations

import numpy as np

from .preprocessing import (
    weight_events, translate_attributes,
    _evaluate_shape, _multiply_weights, _normalise_weights_to_list,
)
from .build import build_maet
from .premaet import is_pre_maet, unpack_pre_maet
from .cosine import (sim_maet)
from .sweep import sweep_sim_maet
from .._defaults import _with_dispatch_scope
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
    """Flat finite values of the window attribute (an attribute with one
    element per event carries one value per event)."""
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
        raise ValueError("cannot derive the centres: the window attribute has no finite values")
    lo = float(vals.min()) if start is None else float(start)
    hi = float(vals.max()) if stop is None else float(stop)
    st = float(default_step) if step is None else float(step)
    if not (st > 0):
        raise ValueError("step must be positive")
    n = int(np.floor((hi - lo) / st + 1e-9)) + 1
    return lo + st * np.arange(max(n, 1))


def _prune_dead_events(p_attr, w, specs):
    """Drop events the window hard-zeroed, before the (heavy) build.

    A windowed density carries out-of-window / beyond-truncation events
    at weight zero (``weight_events`` writes its factor as such). Those
    events contribute nothing to any inner product or to the density an
    entropy integrates, so dropping them here -- at the single windowing
    seam, before ``build_maet`` runs its eager feasibility scan and
    r-ad enumeration over every column -- is exact and saves the bulk of
    a sliding sweep's cost, without touching the core build contract or
    the density-level ``pruned()`` path. The liveness rule is the shared
    one (``_weight_is_live``): an event is live iff every weighted
    attribute has a finite, nonzero value in its column. Skipped only when
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
            continue                      # per-value / scalar: cannot kill an event alone
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
            w_out.append(W)               # per-value / scalar: unchanged
    return p_out, w_out, specs            # specs are per-value -> unchanged


def _locate_row(M, locate):
    """Reduce a ``(K, N)`` attribute to the ``(1, N)`` value its window
    centres on: the multiset centroid by default, else ``'start'`` / ``'end'``
    / ``'mid'`` / a callable. Raw values for a relative attribute (the
    window selects in absolute position; the comparison is
    translation-invariant)."""
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
    ``weight_events`` profile and truncation exactly. The global-default
    truncation width is resolved through the contract helper, so
    ``mpt.set_default(truncation_sigmas=math.inf)`` truncates at the
    finite accuracy-floor width (never "disabled")."""
    from .._defaults import get_default, resolve_truncation_sigmas
    delta = loc_row - centre
    factor = _evaluate_shape(delta, sd, gamma)
    trunc = resolve_truncation_sigmas(get_default("truncation_sigmas"))
    factor = factor.copy()
    factor[np.abs(delta) > trunc * sd] = 0.0
    return factor


def _query_extent(p_query, axis):
    v = np.asarray(p_query[axis], dtype=float).ravel()
    v = v[np.isfinite(v)]
    return float(v.max() - v.min()) if v.size else 0.0


def _resolve_window(spec, query_extent, axis):
    """``(gamma, sd)`` for one window attribute; the default is a rectangle
    as wide as the query's extent on that attribute."""
    if spec is None:
        if query_extent <= 0:
            raise ValueError(
                f"attribute {axis}: the query has zero extent there, so the default "
                f"window width is undefined; give 'width' or 'sd' in "
                f"context_window[{axis}].")
        return 1.0, query_extent / _SQRT12
    gamma = _resolve_shape(spec.get("shape", "rect"))
    has_sd, has_w = "sd" in spec, "width" in spec
    if has_sd == has_w:
        raise ValueError(
            f"attribute {axis}: give exactly one of 'width' or 'sd' in "
            f"context_window[{axis}].")
    sd = float(spec["sd"]) if has_sd else float(spec["width"]) / _SQRT12
    if not (sd > 0):
        raise ValueError(f"attribute {axis}: window width/sd must be > 0.")
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
    return _prune_dead_events(p, w_out, specs)


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


def _check_is_exch_vs_specs(is_exch, specs):
    if is_exch is not None and specs is not None:
        raise ValueError(
            "`is_exch` applies to the flat per-attribute surface; nested "
            "geometry carries its per-level exch inside `specs`. Pass one "
            "or the other.")


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
                f"target_attr={target} is a marginalized attribute; its weights are "
                f"removed before the build, so the window factors would be "
                f"lost. Choose a compared attribute.")
    cw = context_window if isinstance(context_window, dict) else {}
    if require_window and any(cw.get(a) is None for a in keys):
        raise ValueError(
            "windowed_entropy has no query to size the window; give an explicit "
            "context_window entry (width or sd) for every window attribute.")
    win = {a: _resolve_window(cw.get(a), _query_extent(p_query, a), a)
           for a in keys}
    grids = [np.asarray(sweep[a], dtype=float).ravel() for a in keys]
    return keys, drop_axes, target, win, grids


def _single_window(context_window, p_query, axis):
    """(gamma, sd) for one window attribute from the ``(shape, width)``
    tuple; the width defaults to the query's extent on that attribute."""
    shape_raw, width = context_window
    if width is None:
        width = _query_extent(p_query, axis)
    if not (width > 0):
        raise ValueError(
            f"window width on attribute {axis} is zero or undefined; pass "
            f"context_window=(shape, width) with width > 0.")
    gamma = _resolve_shape("rect" if shape_raw is None else shape_raw)
    return gamma, width / _SQRT12


def _query_position(p_query, attr, locate):
    """The query's position on attribute ``attr``: the mean, over its
    events, of each event's element multiset reduced by ``locate``."""
    return float(np.nanmean(_locate_row(p_query[attr],
                                        _resolve_locate(locate, attr))))


def _offsets_single(p_query, specs, is_rel, offsets, centres, start, stop,
                    step, window_attr, drop_window_attr, locate, n):
    """Translate ``offsets`` for one window attribute into the placement
    (centres and query positions) the comparison uses.

    Returns ``(centres, query_pos, drop_window_attr)``. With ``centres``
    absent the window travels with the query (its centre is the offset
    plus the query's position) and ``query_pos`` is None; with ``centres``
    given the window stays at each centre while the query is translated by
    each offset, the correlogram, and ``query_pos`` is the ``(A, T)``
    array of the query's positions (offset plus its position)."""
    if any(v is not None for v in (start, stop, step)):
        raise ValueError(
            "`start`/`stop`/`step` lay out window centres; with `offsets`, "
            "give the window centres as `centres` (or omit them to let the "
            "window travel with the query).")
    if drop_window_attr is None:
        drop_window_attr = False
    if drop_window_attr:
        raise ValueError(
            "`offsets` translate the query along the window attribute, so "
            "that attribute must be compared (drop_window_attr=False).")
    axis = _abs_idx(window_attr, n)
    if _axis_is_rel(specs, [False] * n if is_rel is None else is_rel, axis):
        raise ValueError(
            f"attribute {axis} is relative: translating it leaves every "
            f"within-tuple difference unchanged, so there is nothing to "
            f"sweep. Give window positions as `centres` instead.")
    off = np.asarray(offsets, dtype=float)
    q_pos = _query_position(p_query, axis, locate)
    if centres is None:
        if off.ndim != 1:
            raise ValueError("without `centres`, `offsets` must be 1-D.")
        return off + q_pos, None, drop_window_attr
    c = np.asarray(centres, dtype=float).ravel()
    if off.ndim == 1:
        qc = np.broadcast_to(off[None, :] + q_pos, (c.size, off.size)).copy()
    elif off.ndim == 2 and off.shape[0] == c.size:
        qc = off + q_pos
    else:
        raise ValueError(
            f"with {c.size} centres, `offsets` must be 1-D (shared by every "
            f"centre) or 2-D with {c.size} rows; got shape {off.shape}.")
    return c, qc, drop_window_attr


def _offsets_multi(p_query, specs, is_rel, offsets, sweep, drop, locate,
                   context_window):
    """Merge an ``offsets`` map ``{a: offsets}`` into ``sweep``/``drop``.

    Each named attribute is translated by its offsets and compared. It
    carries a window only if ``context_window`` names it (the window then
    travels with the query); otherwise it is translated with no window.
    Returns ``(sweep, drop, translate_only)``, the merged maps ordered by
    attribute index, and the set of attributes translated with no window."""
    n = len(p_query)
    sweep = dict(sweep or {})
    drop = dict(drop or {})
    cw = context_window if isinstance(context_window, dict) else {}
    translate_only = set()
    for a_in, vals in offsets.items():
        a = _abs_idx(a_in, n)
        if a in sweep:
            raise ValueError(
                f"attribute {a} is named in both `offsets` and `sweep`; an "
                f"attribute is either translated (offsets) or only windowed "
                f"(sweep).")
        if drop.get(a, False):
            raise ValueError(
                f"attribute {a} is translated, so it is compared: it cannot "
                f"be dropped.")
        if _axis_is_rel(specs, [False] * n if is_rel is None else is_rel, a):
            raise ValueError(
                f"attribute {a} is relative: translating it leaves every "
                f"within-tuple difference unchanged, so there is nothing to "
                f"sweep.")
        sweep[a] = (np.asarray(vals, dtype=float).ravel()
                    + _query_position(p_query, a, locate))
        drop[a] = False
        if cw.get(a) is None:
            translate_only.add(a)
    order = sorted(sweep)
    return ({a: sweep[a] for a in order}, {a: drop[a] for a in order},
            frozenset(translate_only))


def _sweep_row(pc, wc, sc, pq, wq, sq, sg, rr, rl, pr, pd, exch_args,
               nested, offs, normalize):
    """The query, translated by every offset, against a fixed (already
    windowed) context, in one pass through :func:`sweep_sim_maet`.
    ``offs`` is ``(A_kept, M)``: one row per attribute left after
    marginalization, one column per offset. Returns ``None`` where no such
    route applies, so the caller can fall back to comparing offset by
    offset. A density with a nested attribute
    takes sweep_sim_maet's contraction route, which contracts the nesting
    level by level with the offsets as a batch dimension."""
    try:
        if nested:
            dc = build_maet(pc, wc, sigma=sg, is_per=pr, period=pd, specs=sc,
                            verbose=False)
            dq = build_maet(pq, wq, sigma=sg, is_per=pr, period=pd, specs=sq,
                            verbose=False)
        else:
            dc = build_maet(pc, wc, sg, rr, rl, pr, pd, *exch_args,
                            verbose=False)
            dq = build_maet(pq, wq, sg, rr, rl, pr, pd, *exch_args,
                            verbose=False)
        out = np.asarray(sweep_sim_maet(dc, dq, offs, normalize=normalize,
                                        verbose=False), dtype=float).ravel()
    except Exception:
        return None
    if out.size != offs.shape[1] or not np.all(np.isfinite(out)):
        return None
    return out


@_with_dispatch_scope
def windowed_similarity(p_context, w_context=None, p_query=None,
                        w_query=None, sigma=None, r=None,
                        is_rel=None, is_per=None, period=None,
                        centres=None, *, is_exch=None, rel=None,
                        exch=None,
                        start=None, stop=None, step=None,
                        offsets=None,
                        context_window=("rect", None),
                        query_window=None, window_attr=-1, drop_window_attr=None,
                        sweep=None, drop=None, locate="centroid",
                        target_attr=None, normalize="oneSidedDenom", specs=None,
                        verbose=False):
    r"""Slide a query across a context and measure their similarity at each
    position (a pre-MAET cross-correlation).

    Input forms, in the order to reach for them: two whole pre-MAETs, the
    canonical entry; then the raw positional form, with the operands'
    parts and the five geometry vectors written out.

    **Pre-MAET input**:

    - ``windowed_similarity(pm_context, pm_query, centres, ...)``. The
      shared geometry is read from their specs, and any of the six
      per-attribute parameters --- ``sigma``, ``is_per``, ``period``,
      ``r``, ``rel``, ``exch`` --- may be given alongside to override it,
      as at :func:`build_maet`. An override may name every attribute
      or be selective, a length-A list whose ``None`` entries keep what
      the spec carries: ``sigma=[None, s, None]`` sweeps the second
      attribute's width and leaves the rest to the pre-MAET. The two
      pre-MAETs describe one comparison, so they must agree on ``r``,
      ``rel``, ``exch`` and the nesting; ``sigma``, ``is_per`` and
      ``period`` may differ and are taken from the context.

    **Raw positional input**:

    - ``windowed_similarity(p_context, w_context, p_query, w_query,
      sigma, r, is_rel, is_per, period, centres, ...)``.

    **Terms** (§3 of the article, event weighting and attribute
    translation). The context is restricted to a local region by a
    *window*, a non-negative profile :math:`h` centred at a value
    :math:`c`: each event's weights on one attribute (``target_attr``,
    below) are multiplied by :math:`h(p_S(n) - c)`. The *window attribute*
    is the attribute :math:`S` the window is defined over, and each
    *centre* is a value of it at which the window is placed. An *offset*
    translates every element of one of the query's attributes by that
    amount.

    One window attribute (the common case): name it with ``window_attr``
    (default: the last attribute) and give the centres as ``centres``, or
    as ``start`` / ``stop`` / ``step``. ``drop_window_attr`` says whether
    the window attribute is compared (``False``) or, once the window has
    weighted the events, marginalized by removing it from both pre-MAETs
    (``True``). ``locate`` reduces each event's element multiset on the
    window attribute to the one value the window reads: ``'centroid'``
    (the mean; default), ``'start'`` (the first element), ``'end'`` (the
    last), ``'mid'`` (the midpoint of the first and last), or a callable.
    The query's *position* on an attribute is the mean, over its events,
    of their located values. Unless ``offsets`` are given, at each centre
    the query is translated along the window attribute so that its
    position is the centre (ordinary cross-correlation), or left as it is
    where the window attribute is marginalized or relative. The window
    defaults to a rectangle as wide as the query's extent on the window
    attribute (the range of its values there); ``context_window=(shape,
    width)`` overrides it.

    Several window attributes: give ``sweep={a: centres, ...}``, where
    ``a`` is an attribute index, and a parallel ``drop={a: bool, ...}``.
    The output has one dimension per attribute named, in the order named,
    and holds every combination of their centres. ``context_window`` is
    then a ``dict`` ``{a: {'shape': ..., 'width' or 'sd': ...}, ...}``, and
    ``locate`` may be one too (``{a: rule, ...}``, an attribute it does
    not name taking ``'centroid'``; MATLAB: the ``{a, rule; ...}`` cell).

    ``offsets`` translate the query, and the output is indexed by the
    offsets, each measured from the query's values as given. Aligned
    before any preprocessing, so that the query's first time value equals
    the context's, an offset is the time from the start of the context to
    the start of the query; differencing and binding leave the surviving
    values unchanged, so the reading carries through them. With one window
    attribute, a 1-D ``offsets`` and no ``centres`` lets the window travel
    with the query: at each offset the query is translated along the
    window attribute and the window is centred on its translated position
    (the offset plus its position). With ``centres`` as well, the window
    stays at each centre while the query is translated by each offset,
    giving a *correlogram*: ``offsets`` is 1-D, shared by every centre, or
    ``(C, T)``, with ``C`` the number of centres and ``T`` the number of
    offsets, and the output is ``(C, T)``. The window attribute must then
    be compared and absolute. With several window attributes, ``offsets``
    is a map ``{a: offsets, ...}`` naming the attributes to translate:
    each is compared, and is windowed, the window travelling with the
    query, only if ``context_window`` names it; no attribute may be named
    in both ``offsets`` and ``sweep``. Where the window does not move with
    the query --- the correlogram, or a translated attribute with no
    window --- the offsets at each window position are computed in one
    pass by :func:`sweep_sim_maet` (a nested attribute on its contraction
    route), falling back to one comparison per offset where no such route
    applies.

    ``target_attr`` is the attribute whose per-event weights the window
    multiplies (default: the first compared attribute; it may be the
    window attribute). ``specs`` carries nested geometry from
    :func:`bind_events`. ``is_exch`` is the per-attribute exchangeability
    vector of the raw positional form (``None`` keeps the unordered
    default); required, in particular, for ordered attributes carrying a
    matrix-valued kernel covariance. It is mutually exclusive with
    ``specs``, whose nesting carries its own per-level exch.
    """
    query_specs = specs
    if is_pre_maet(p_context):
        (p_attrs, w_attrs, sigma, r, is_rel, is_per, period, is_exch,
         side_specs, centres) = _windowed_pre_maet_args(
            [p_context, w_context], p_query if p_query is not None
            else centres,
            {"sigma": sigma, "is_per": is_per, "period": period, "r": r,
             "rel": rel if rel is not None else is_rel, "exch": exch},
            "windowed_similarity")
        p_context, p_query = p_attrs
        w_context, w_query = w_attrs
        specs, query_specs = side_specs
    translate_only = frozenset()
    query_pos = None
    if offsets is not None:
        if isinstance(offsets, dict):
            if query_window is not None:
                raise ValueError(
                    "`query_window` applies with one window attribute; with "
                    "several, the query is placed at each combination of "
                    "centres.")
            sweep, drop, translate_only = _offsets_multi(
                p_query, specs, is_rel, offsets, sweep, drop, locate,
                context_window)
        else:
            if sweep is not None:
                raise ValueError(
                    "with several window attributes (`sweep`), give "
                    "`offsets` as a map {a: offsets}.")
            centres, query_pos, drop_window_attr = _offsets_single(
                p_query, specs, is_rel, offsets, centres, start, stop, step,
                window_attr, drop_window_attr, locate, len(p_context))
    if sweep is not None:
        if drop is None:
            raise ValueError("`sweep` requires a parallel `drop`.")
        if query_window is not None:
            raise ValueError(
                "`query_window` applies with one window attribute; with "
                "several (`sweep`), the query is placed at each combination "
                "of centres.")
        return _ws_multi(
            p_context, w_context, p_query, w_query, sigma, r, is_rel, is_per,
            period, is_exch, sweep, drop,
            context_window if isinstance(context_window, dict)
            else None, locate, normalize, target_attr, specs, query_specs,
            translate_only)
    if drop_window_attr is None:
        raise ValueError(
            "`drop_window_attr` is required (True places only, False compares).")
    return _ws_single(
        p_context, w_context, p_query, w_query, sigma, r, is_rel, is_per, period,
        is_exch, centres, start, stop, step, query_pos, context_window,
        query_window, window_attr, drop_window_attr, locate, target_attr,
        normalize, specs, query_specs)


def _ws_multi(p_context, w_context, p_query, w_query, sigma, r, is_rel, is_per,
              period, is_exch, sweep, drop, context_window, locate, normalize,
              target_attr, specs, query_specs=None,
              translate_only=frozenset()):
    if query_specs is None:
        query_specs = specs
    p_context, p_query = list(p_context), list(p_query)
    _check_is_exch_vs_specs(is_exch, specs)
    keys, drop_axes, target, win, grids = _prep_sweep(
        p_context, p_query, sweep, drop, context_window, target_attr)
    nested = specs is not None
    out = np.empty(tuple(g.size for g in grids), dtype=float)
    # Translated attributes with no window leave the windowed context
    # unchanged across their offsets, so at each position of the other
    # (windowed-only) attributes the context is fixed and the offsets can
    # be swept in one pass. That holds when every other swept attribute is
    # windowed but not translated (dropped, or relative).
    t_axes = [a for a in keys if a in translate_only]
    w_axes = [a for a in keys if a not in translate_only]
    routable = bool(t_axes) and all(
        a in drop_axes or _axis_is_rel(specs, is_rel, a) for a in w_axes)
    done = np.zeros(out.shape, dtype=bool)
    if routable:
        t_pos = [keys.index(a) for a in t_axes]
        w_pos = [keys.index(a) for a in w_axes]
        q_pos = {a: _query_position(p_query, a, locate) for a in t_axes}
        pq0, wq0, sq0, keep = _drop_axes(p_query, w_query, query_specs,
                                         drop_axes)
        sg, rr, rl, pr, pd = (_sub(sigma, keep), _sub(r, keep),
                              _sub(is_rel, keep), _sub(is_per, keep),
                              _sub(period, keep))
        sy = None if is_exch is None else _sub(is_exch, keep)
        exch_args = () if sy is None else (sy,)
        t_shape = tuple(grids[j].size for j in t_pos)
        t_idx = list(np.ndindex(*t_shape))
        offs = np.zeros((len(keep), len(t_idx)), dtype=float)
        for m, ti in enumerate(t_idx):
            for k, a in enumerate(t_axes):
                offs[keep.index(a), m] = (grids[t_pos[k]][ti[k]] - q_pos[a])
        for wi in np.ndindex(*tuple(grids[j].size for j in w_pos)):
            centres = {w_axes[k]: float(grids[w_pos[k]][wi[k]])
                       for k in range(len(w_axes))}
            pc_w, wc_w, sc_w = _apply_windows(p_context, w_context, specs,
                                              centres, win, locate, target)
            pc, wc, sc, _ = _drop_axes(pc_w, wc_w, sc_w, drop_axes)
            row = _sweep_row(pc, wc, sc, pq0, wq0, sq0, sg, rr, rl, pr, pd,
                             exch_args, nested, offs, normalize)
            if row is None:
                continue
            for m, ti in enumerate(t_idx):
                full = [0] * len(keys)
                for k, j in enumerate(w_pos):
                    full[j] = wi[k]
                for k, j in enumerate(t_pos):
                    full[j] = ti[k]
                out[tuple(full)] = row[m]
                done[tuple(full)] = True
    for idx in np.ndindex(*out.shape):
        if done[idx]:
            continue
        centres = {keys[j]: float(grids[j][idx[j]]) for j in range(len(keys))}
        offs = [None] * len(p_query)
        for a in keys:
            if a in drop_axes or _axis_is_rel(specs, is_rel, a):
                continue
            q_loc = float(np.nanmean(_locate_row(
                p_query[a], _resolve_locate(locate, a))))
            offs[a] = np.array([[centres[a] - q_loc]], dtype=float)
        if any(o is not None for o in offs):
            pq_t, wq_t, sq_t = unpack_pre_maet(translate_attributes(
                p_query, w_query, offs, specs=query_specs))
        else:
            pq_t, wq_t, sq_t = p_query, w_query, query_specs
        w_centres = {a: c for a, c in centres.items()
                     if a not in translate_only}
        pc_w, wc_w, sc_w = _apply_windows(p_context, w_context, specs,
                                          w_centres, win, locate, target)
        pc, wc, sc, keep = _drop_axes(pc_w, wc_w, sc_w, drop_axes)
        pq, wq, sq, _ = _drop_axes(pq_t, wq_t, sq_t, drop_axes)
        sg, rr, rl, pr, pd = (_sub(sigma, keep), _sub(r, keep), _sub(is_rel, keep),
                              _sub(is_per, keep), _sub(period, keep))
        sy = None if is_exch is None else _sub(is_exch, keep)
        if nested:
            dc = build_maet(pc, wc, sigma=sg, is_per=pr, period=pd, specs=sc,
                                verbose=False)
            dq = build_maet(pq, wq, sigma=sg, is_per=pr, period=pd, specs=sq,
                                verbose=False)
            out[idx] = float(sim_maet(
                dc, dq, normalize=normalize,
                verbose=False))
        else:
            exch_args = () if sy is None else (sy,)
            out[idx] = float(sim_maet(
                pc, wc, pq, wq, sg, rr, rl, pr, pd, *exch_args,
                normalize=normalize, verbose=False))
    return out


def _ws_single(p_context, w_context, p_query, w_query, sigma, r, is_rel, is_per,
               period, is_exch, centres, start, stop, step, query_pos,
               context_window, query_window, window_attr, drop_window_attr,
               locate, target_attr, normalize, specs, query_specs=None):
    """``query_pos`` is None (the query placed with the window at each
    centre) or the ``(A, T)`` query positions of the correlogram, from
    :func:`_offsets_single`."""
    if query_specs is None:
        query_specs = specs
    p_context, p_query = list(p_context), list(p_query)
    _check_is_exch_vs_specs(is_exch, specs)
    n = len(p_context)
    axis = _abs_idx(window_attr, n)
    nested = specs is not None
    gamma, sd = _single_window(context_window, p_query, axis)
    win = {axis: (gamma, sd)}
    if query_window is None:
        q_win = None
    else:
        q_win = {axis: _single_window(query_window, p_query, axis)}
    drop_axes = {axis} if drop_window_attr else set()
    keep = [i for i in range(n) if i not in drop_axes]
    if not keep:
        raise ValueError("dropping the only attribute leaves nothing to compare.")
    target = (keep[0] if target_attr is None else _abs_idx(target_attr, n))
    if target in drop_axes:
        raise ValueError(
            f"target_attr={target} is the marginalized window attribute; its weights are removed "
            f"before the build. Choose a compared attribute.")
    ctx_centres = _resolve_centres(p_context, axis, centres, start, stop, step,
                                   default_step=gamma and sd * _SQRT12 or sd)
    A = ctx_centres.size
    if query_pos is None:
        q_rows, out_shape = ctx_centres.reshape(A, 1), (A,)
    else:
        q_rows = np.asarray(query_pos, dtype=float)
        out_shape = q_rows.shape
    rel_axis = _axis_is_rel(specs, is_rel, axis)
    sg, rr, rl, pr, pd = (_sub(sigma, keep), _sub(r, keep), _sub(is_rel, keep),
                          _sub(is_per, keep), _sub(period, keep))
    sy = None if is_exch is None else _sub(is_exch, keep)
    exch_args = () if sy is None else (sy,)
    out = np.empty((A, q_rows.shape[1]), dtype=float)
    # The query window attaches to the query, not to the sweep: it is centred
    # on the query's own position on the window attribute and applied before the
    # per-offset translation, so a template's finite extent is a property of
    # the template and does not change as it slides. Resolved once, outside
    # both loops, because neither the query nor its window varies with the
    # sweep position.
    if q_win is not None:
        q_centre = float(np.nanmean(_locate_row(
            p_query[axis], _resolve_locate(locate, axis))))
        q_target = target
        p_query, w_query, specs_q = _apply_windows(
            p_query, w_query, query_specs, {axis: q_centre}, q_win, locate,
            q_target)
        if nested:
            query_specs = specs_q
    for a in range(A):
        pc_w, wc_w, sc_w = _apply_windows(
            p_context, w_context, specs, {axis: float(ctx_centres[a])}, win,
            locate, target)
        pc, wc, sc, _ = _drop_axes(pc_w, wc_w, sc_w, drop_axes)
        # Correlogram (offsets with centres): the window stays at this
        # centre while the query is translated, so the windowed context is
        # fixed across the offsets and they can be swept in one pass.
        # (_offsets_single has refused a dropped or relative window
        # attribute.)
        if query_pos is not None:
            q_loc = float(np.nanmean(_locate_row(
                p_query[axis], _resolve_locate(locate, axis))))
            pq0, wq0, sq0, _ = _drop_axes(p_query, w_query, query_specs,
                                          drop_axes)
            offs_row = np.zeros((len(keep), q_rows.shape[1]), dtype=float)
            offs_row[keep.index(axis)] = q_rows[a] - q_loc
            row = _sweep_row(pc, wc, sc, pq0, wq0, sq0, sg, rr, rl, pr, pd,
                             exch_args, nested, offs_row, normalize)
            if row is not None:
                out[a, :] = row
                continue
        dc = (build_maet(pc, wc, sigma=sg, is_per=pr, period=pd, specs=sc,
                             verbose=False) if nested else None)
        for t in range(q_rows.shape[1]):
            if drop_window_attr or rel_axis:
                pq_t, wq_t, sq_t = p_query, w_query, query_specs
            else:
                q_loc = float(np.nanmean(_locate_row(
                    p_query[axis], _resolve_locate(locate, axis))))
                offs = [None] * n
                offs[axis] = np.array([[float(q_rows[a, t]) - q_loc]], dtype=float)
                pq_t, wq_t, sq_t = unpack_pre_maet(translate_attributes(
                    p_query, w_query, offs, specs=query_specs))
            pq, wq, sq, _ = _drop_axes(pq_t, wq_t, sq_t, drop_axes)
            if nested:
                dq = build_maet(pq, wq, sigma=sg, is_per=pr, period=pd,
                                    specs=sq, verbose=False)
                out[a, t] = float(sim_maet(
                    dc, dq, normalize=normalize,
                    verbose=False))
            else:
                out[a, t] = float(sim_maet(
                    pc, wc, pq, wq, sg, rr, rl, pr, pd, *exch_args,
                    normalize=normalize, verbose=False))
    return out.reshape(out_shape)


@_with_dispatch_scope
def windowed_entropy(p_context, w_context=None, sigma=None, r=None,
                     is_rel=None, is_per=None, period=None,
                     centres=None, *, is_exch=None, rel=None,
                     exch=None,
                     start=None, stop=None, step=None,
                     context_window=("rect", None), window_attr=-1,
                     drop_window_attr=None, sweep=None, drop=None,
                     locate="centroid", method="differential", base=2.0,
                     marginalize=None, target_attr=None, specs=None,
                     marginalise=None,
                     verbose=False):
    r"""Slide a window across a context and read its entropy at each position.

    Input forms, in the order to reach for them: a whole pre-MAET, the
    canonical entry; then the raw positional form, with ``p_context``,
    ``w_context`` and the five geometry vectors written out.

    **Pre-MAET input**:

    - ``windowed_entropy(pm, centres, ...)``. The geometry is read from
      its specs, and any of the six per-attribute parameters ---
      ``sigma``, ``is_per``, ``period``, ``r``, ``rel``, ``exch`` --- may
      be given alongside to override it, as at :func:`build_maet`.
      An override may name every attribute or be selective, a length-A
      list whose ``None`` entries keep what the spec carries:
      ``sigma=[None, s, None]`` sweeps the second attribute's width and
      leaves the rest to the pre-MAET.

    **Raw positional input**:

    - ``windowed_entropy(p_context, w_context, sigma, r, is_rel,
      is_per, period, centres, ...)``.

    **Terms** (§3 of the article, event weighting). A *window*, a
    non-negative profile :math:`h` centred at a value :math:`c`, multiplies
    each event's weights on one attribute (``target_attr``) by
    :math:`h(p_S(n) - c)`. The *window attribute* is the attribute
    :math:`S` the window is defined over, and each *centre* is a value of
    it at which the window is placed.

    Takes the window attribute, centres, window and ``locate`` as
    :func:`windowed_similarity` does: one window attribute (``window_attr``
    with ``centres`` and ``drop_window_attr``) or several (``sweep`` with
    ``drop``). There is no query, so at each centre (or combination of
    centres) the windowed density is built, with any window attribute
    whose ``drop`` is true first marginalized by removing it from the
    pre-MAET, and its entropy taken. With no query to size a default
    window from, ``context_window`` must give a width (or sd) for every
    window attribute. ``marginalize`` (also accepted as ``marginalise``)
    is reserved for integrating a
    compared attribute out of the density and is not yet implemented.
    """
    if is_pre_maet(p_context):
        (p_attrs, w_attrs, sigma, r, is_rel, is_per, period, is_exch,
         side_specs, centres) = _windowed_pre_maet_args(
            [p_context], w_context if w_context is not None else centres,
            {"sigma": sigma, "is_per": is_per, "period": period, "r": r,
             "rel": rel if rel is not None else is_rel, "exch": exch},
            "windowed_entropy")
        p_context, = p_attrs
        w_context, = w_attrs
        specs, = side_specs
    if marginalise is not None:
        if marginalize is not None:
            raise ValueError(
                "give `marginalize` or its alternative spelling "
                "`marginalise`, not both.")
        marginalize = marginalise
    if marginalize is not None:
        raise NotImplementedError(
            "marginalize (integrating a compared attribute out of the density) is "
            "not yet implemented.")
    if sweep is not None:
        if drop is None:
            raise ValueError("`sweep` requires a parallel `drop`.")
        return _we_multi(
            p_context, w_context, sigma, r, is_rel, is_per, period, is_exch,
            sweep, drop,
            context_window if isinstance(context_window, dict) else None, locate,
            method, base, target_attr, specs)
    if drop_window_attr is None:
        raise ValueError(
            "`drop_window_attr` is required (True marginalizes the window attribute, False "
            "retains it).")
    return _we_single(
        p_context, w_context, sigma, r, is_rel, is_per, period, is_exch, centres,
        start, stop, step, context_window, window_attr, drop_window_attr, locate,
        method, base, target_attr, specs)


def _we_multi(p_context, w_context, sigma, r, is_rel, is_per, period, is_exch,
              sweep, drop, context_window, locate, method, base, target_attr,
              specs):
    p_context = list(p_context)
    _check_is_exch_vs_specs(is_exch, specs)
    keys, drop_axes, target, win, grids = _prep_sweep(
        p_context, p_context, sweep, drop, context_window, target_attr,
        require_window=True)
    nested = specs is not None
    from ..entropy import entropy_maet
    out = np.empty(tuple(g.size for g in grids), dtype=float)
    for idx in np.ndindex(*out.shape):
        centres = {keys[j]: float(grids[j][idx[j]]) for j in range(len(keys))}
        pc_w, wc_w, sc_w = _apply_windows(p_context, w_context, specs, centres,
                                          win, locate, target)
        pc, wc, sc, keep = _drop_axes(pc_w, wc_w, sc_w, drop_axes)
        sg, rr, rl, pr, pd = (_sub(sigma, keep), _sub(r, keep), _sub(is_rel, keep),
                              _sub(is_per, keep), _sub(period, keep))
        sy = None if is_exch is None else _sub(is_exch, keep)
        if nested:
            dens = build_maet(pc, wc, sigma=sg, is_per=pr, period=pd,
                                  specs=sc, verbose=False)
        else:
            exch_args = () if sy is None else (sy,)
            dens = build_maet(pc, wc, sg, rr, rl, pr, pd, *exch_args,
                                  verbose=False)
        out[idx] = float(entropy_maet(dens, method=method, base=base,
                                          verbose=False))
    return out


def _we_single(p_context, w_context, sigma, r, is_rel, is_per, period, is_exch,
               centres, start, stop, step, context_window, window_attr,
               drop_window_attr, locate, method, base, target_attr, specs):
    p_context = list(p_context)
    _check_is_exch_vs_specs(is_exch, specs)
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
        raise ValueError(f"target_attr={target} is the marginalized window attribute.")
    ctx_centres = _resolve_centres(p_context, axis, centres, start, stop, step,
                                   default_step=width)
    from ..entropy import entropy_maet
    sg, rr, rl, pr, pd = (_sub(sigma, keep), _sub(r, keep), _sub(is_rel, keep),
                          _sub(is_per, keep), _sub(period, keep))
    sy = None if is_exch is None else _sub(is_exch, keep)
    exch_args = () if sy is None else (sy,)
    out = np.empty(ctx_centres.size, dtype=float)
    for i, c in enumerate(ctx_centres):
        pc_w, wc_w, sc_w = _apply_windows(
            p_context, w_context, specs, {axis: float(c)}, win, locate, target)
        pc, wc, sc, _ = _drop_axes(pc_w, wc_w, sc_w, drop_axes)
        if nested:
            dens = build_maet(pc, wc, sigma=sg, is_per=pr, period=pd,
                                  specs=sc, verbose=False)
        else:
            dens = build_maet(pc, wc, sg, rr, rl, pr, pd, *exch_args,
                                  verbose=False)
        out[i] = float(entropy_maet(dens, method=method, base=base,
                                        verbose=False))
    return out


def _windowed_pre_maet_args(pms, centres, kw, func):
    """Resolve a pre-MAET call of the windowed functions.

    ``windowed_similarity`` and ``windowed_entropy`` take their geometry
    positionally, as vectors shared by both operands. Given whole
    pre-MAETs instead, this reads the shared geometry out of their specs,
    applies any of the six per-attribute overrides passed as keywords,
    and returns the parts the workers already speak.

    The two pre-MAETs of ``windowed_similarity`` describe one comparison,
    so they must agree on the structural geometry: same attribute count,
    and the same ``r``, ``rel``, ``exch``, and nesting on every attribute.
    The context supplies the specs; a disagreement is an error rather
    than a silent choice between them.

    Parameters
    ----------
    pms : list of Mapping
        The pre-MAET operands, context first.
    centres : object
        The argument sitting in the ``centres`` slot of the call.
    kw : dict
        The six overrides, keyed ``sigma``, ``is_per``, ``period``,
        ``r``, ``rel``, ``exch``; ``None`` where not given.
    func : str
        The caller's name, for error messages.

    Returns
    -------
    tuple
        ``(p_attrs, w_attrs, sigma, r, is_rel, is_per, period, is_exch,
        specs, centres)``, with ``specs`` ``None`` unless the geometry is
        nested.
    """
    from .build import (_normalise_specs, _override_specs,
                        _resolve_kernel_param)

    specs = pms[0].get("specs")
    if not specs:
        raise ValueError(
            f"{func}: a pre-MAET passed here must carry its specs — they "
            "are where the shared geometry is read from. Build it with "
            "pack_pre_maet(p_attr, w_attr, specs), or use the positional form.")
    A = len(pms[0]["p_attr"])
    for k, pm in enumerate(pms[1:], start=1):
        _check_specs_agree(specs, pm.get("specs"), A, func)

    specs = _override_specs(specs, kw.get("r"), kw.get("rel"),
                            kw.get("exch"), A)
    _, _, _, _, names, spec_kernel = _normalise_specs(specs, A)
    sigma = _resolve_kernel_param(kw.get("sigma"), spec_kernel["sigma"],
                                  "sigma", names, A)
    is_per = _resolve_kernel_param(kw.get("is_per"), spec_kernel["is_per"],
                                   "is_per", names, A)
    period = _resolve_kernel_param(kw.get("period"), spec_kernel["period"],
                                   "period", names, A, default=0.0)
    r_vec, is_rel_vec, is_exch_vec, nested_list, _, _ = \
        _normalise_specs(specs, A)

    nested = any(n is not None for n in nested_list)
    # Each side keeps its own 'tags', and takes every other field from the
    # first: the comparison's geometry is shared, the grouping is not.
    side_specs = [specs]
    for pm in pms[1:]:
        own = pm.get("specs")
        side_specs.append([dict(shared, tags=one.get("tags"))
                           if one.get("tags") is not None
                           else {k: v for k, v in shared.items()
                                 if k != "tags"}
                           for shared, one in zip(specs, own)])
    return ([pm["p_attr"] for pm in pms], [pm.get("w_attr") for pm in pms],
            sigma, r_vec, is_rel_vec, is_per, period,
            None if nested else is_exch_vec,
            [sp if nested else None for sp in side_specs], centres)


def _check_specs_agree(specs_a, specs_b, A, func):
    """The pre-MAETs of one comparison must share the structural geometry.

    ``tags`` is not among the fields compared. It says which slot of its
    own side belongs to which nesting group, so its length is that side's
    padded inner cardinality --- a chorale's beat may hold seven notes
    where the prototype it is compared against holds two. What must agree
    is the nesting the comparison is read under, which ``r``, ``rel``, and
    ``exch`` carry.
    """
    if not specs_b or len(specs_b) != A:
        raise ValueError(
            f"{func}: the two pre-MAETs must have the same attribute count "
            "and both carry specs — they describe one comparison.")
    for a in range(A):
        for f in ("r", "rel", "exch"):
            va = list(np.ravel(specs_a[a].get(f, [])))
            vb = list(np.ravel(specs_b[a].get(f, [])))
            if va != vb:
                raise ValueError(
                    f"{func}: the two pre-MAETs disagree on '{f}' for "
                    f"attribute {a}. They describe one comparison, so the "
                    "structural geometry must match; sigma, is_per and "
                    "period may differ and are taken from the first.")
        if (specs_a[a].get("tags") is None) != (specs_b[a].get("tags") is None):
            raise ValueError(
                f"{func}: one pre-MAET nests attribute {a} and the other "
                "does not. They describe one comparison, so both sides must "
                "be read under the same nesting.")
