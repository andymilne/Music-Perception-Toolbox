"""Pre-MAET swept comparison, entropy, and mass: ``swept_similarity``,
``swept_entropy``, and ``swept_mass``.

``swept_similarity`` compares a query with a context at each of a list of
*sweep values* on one or more attributes. By default each sweep value
translates the query (attribute translation, article Sec. 3) and it is
compared with the whole context; a window (event weighting, article Sec. 3)
may be added. One rule places everything: at each sweep value a translated
query has its reference value ``query_ref`` there, and a window has its
reference value (displacement 0) there; per attribute, ``align`` says which
of the two are placed. ``swept_entropy`` and ``swept_mass`` have no query: at each sweep value
they align a window on the context and take the entropy of the windowed
density, or its mass in a region.

The window factors prune the context to the windowed region before the
build. Where the windowed context is fixed across the query's translations,
those translations are computed together by :func:`sweep_sim_maet`.
"""

from __future__ import annotations

import warnings

import numpy as np

from .preprocessing import (
    weight_events, translate_attributes,
    _multiply_weights, _normalise_weights_to_list,
    _resolve_profile, _resolve_edges, _locate_row, _weight_factor,
    _SHAPE_ALIASES, _EDGES,
)
from .build import build_maet
from .premaet import _parts_per_event, is_pre_maet, unpack_pre_maet
from .cosine import (sim_maet)
from .sweep import sweep_sim_maet
from .._defaults import _with_dispatch_scope
from .density import _weight_is_live

_SQRT12 = 2.0 * np.sqrt(3.0)

def _abs_idx(idx, n_attr):
    a = idx if idx >= 0 else n_attr + idx
    if not (0 <= a < n_attr):
        raise ValueError(f"attribute index {idx} out of range for {n_attr} attributes")
    return a


def _axis_values(p_attr, axis):
    """Flat finite values of the window attribute (an attribute with one
    element per event carries one value per event)."""
    M = np.asarray(p_attr[axis], dtype=float)
    v = M.ravel()
    return v[np.isfinite(v)]


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


def _resolve_locate(locate, axis):
    return locate.get(axis, "centroid") if isinstance(locate, dict) else locate


class _Window(tuple):
    """A window on one attribute: ``(profile, closed, is_per, period)``,
    the resolved weighting profile of :mod:`preprocessing` and the
    geometry its displacement is measured in."""

    __slots__ = ()

    def __new__(cls, profile, closed, is_per=False, period=0.0):
        return tuple.__new__(cls, (profile, bool(closed), bool(is_per),
                                   float(period)))

    profile = property(lambda self: self[0])
    closed = property(lambda self: self[1])
    is_per = property(lambda self: self[2])
    period = property(lambda self: self[3])

    @property
    def default_step(self):
        """The default step of a window-only sweep: half the window's
        standard deviation. The profile changes on the scale of the window,
        as each event's weight follows it, so this leaves every feature
        within a quarter-sd of a grid point (as half the peaks' sd does for a
        translation sweep). NaN for a profile function, which has no scale
        of its own."""
        return self.profile.sd / 2.0

    def with_geometry(self, is_per, period):
        return _Window(self.profile, self.closed, is_per, period)


def _window_factor(loc_row, at, win):
    """Per-event factor of window ``win`` aligned at ``at``, over a reduced
    locating row: the event weighting of :func:`weight_events`, through
    the same implementation."""
    return _weight_factor(loc_row, at, win.profile, is_per=win.is_per,
                          period=win.period, closed=win.closed)


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


def _apply_windows(p, w, specs, at, win, locate, target):
    w_out = _normalise_weights_to_list(w, len(p))
    for a, s in at.items():
        loc = _locate_row(p[a], _resolve_locate(locate, a))
        factor = _window_factor(loc, s, win[a])
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


def _query_position(p_query, attr, locate):
    """The default queryRef on attribute ``attr``: the mean, over the
    query's events, of their located values."""
    return float(np.nanmean(_locate_row(p_query[attr],
                                        _resolve_locate(locate, attr))))


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


# ---------------------------------------------------------------------------
#  The sweep plan
# ---------------------------------------------------------------------------

_ALIGN = ("window", "query", "both", "independent")


def _attr_map(m, n, name):
    """A per-attribute map with its keys resolved to attribute indices."""
    if m is None:
        return {}
    if not isinstance(m, dict):
        raise TypeError(
            f"`{name}` must be a dict keyed by attribute index, "
            f"{{a: ...}}; got {type(m).__name__}.")
    out = {}
    for k, v in m.items():
        if isinstance(k, (bool, np.bool_)) or not isinstance(
                k, (int, np.integer)):
            raise TypeError(
                f"`{name}`: keys are attribute indices (int); got {k!r}.")
        a = _abs_idx(int(k), n)
        if a in out:
            raise ValueError(f"`{name}` names attribute {a} twice.")
        out[a] = v
    return out


def _bare_generator(v, name, sweep):
    """A bare number for ``start``, ``stop``, or ``step``, as ``{a: v}``.

    It applies to the attribute ``sweep`` names, and only where it names
    exactly one; a map passes through unchanged.
    """
    if v is None or isinstance(v, dict):
        return v
    if isinstance(v, (bool, np.bool_)) or not np.isscalar(v) or not \
            isinstance(v, (int, float, np.integer, np.floating)):
        return v
    if len(sweep) != 1:
        raise ValueError(
            f"a bare `{name}` applies to the one swept attribute, but "
            f"`sweep` names {len(sweep)}; give {name}={{a: value}}.")
    return {next(iter(sweep)): v}


def _attr_list(v, n, name):
    """``drop``: an attribute index or a list of them."""
    if v is None:
        return set()
    if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
        v = [v]
    out = set()
    for k in v:
        if isinstance(k, (bool, np.bool_)) or not isinstance(
                k, (int, np.integer)):
            raise TypeError(
                f"`{name}` lists attribute indices (int); got {k!r}.")
        out.add(_abs_idx(int(k), n))
    return out




def _parse_window(spec, a, name, default_width=None):
    """A :class:`_Window` from its specification: ``(shape, width)``,
    ``(shape, width, edges)``, a profile function, or a dict with
    ``'shape'`` and ``'width'``, ``'sd'``, or ``'decay_rate'``, and
    ``'edges'``.

    The profiles are those of :func:`weight_events`: the
    rectangle-Gaussian family (``'rect'``, ``'gaussian'``, or a number in
    [0, 1]), scaled by ``'width'`` or ``'sd'``; the exponentials aligned
    at the window's reference value (``'exponential'``, and
    ``'exponentialBefore'`` and ``'exponentialAfter'``, which extend to
    one side of it only), scaled by ``'sd'`` or ``'decay_rate'``; and a callable of the displacement. The
    serial-position profiles, anchored at the first and last events
    rather than at the sweep value, are refused. The width may be left
    out (``None``, or the whole spec ``None``) only where
    ``default_width``, a callable, supplies one; a rectangle whose width
    is defaulted is closed unless ``edges`` says otherwise, and one whose
    width is given is half-open unless ``edges`` says otherwise."""
    edges = None
    sd = width = rate = None
    if spec is None:
        spec = ("rect", None)
    if callable(spec):
        shape = spec
    elif isinstance(spec, dict):
        shape = spec.get("shape", "rect")
        edges = spec.get("edges")
        given = [k for k in ("sd", "width", "decay_rate") if k in spec]
        if len(given) > 1:
            raise ValueError(
                f"`{name}` for attribute {a}: give one of 'width', 'sd', "
                f"or 'decay_rate', not {' and '.join(given)}.")
        sd, width, rate = (spec.get("sd"), spec.get("width"),
                           spec.get("decay_rate"))
    elif isinstance(spec, (tuple, list)) and len(spec) in (2, 3):
        shape, width = spec[0], spec[1]
        if len(spec) == 3:
            edges = spec[2]
    else:
        raise ValueError(
            f"`{name}` for attribute {a}: give (shape, width), (shape, "
            f"width, edges), a profile function, or {{'shape': ..., "
            f"'width' | 'sd' | 'decay_rate': ..., 'edges': ...}}; got "
            f"{spec!r}.")
    if shape is None:
        shape = "rect"
    named = isinstance(shape, str) and shape.lower() not in _SHAPE_ALIASES
    defaulted = False
    if not named and not callable(shape) and sd is None and width is None:
        if default_width is None:
            raise ValueError(
                f"`{name}` for attribute {a}: a width is required, "
                f"(shape, width).")
        width = default_width()
        defaulted = True
    if named and width is not None and not isinstance(spec, dict):
        raise ValueError(
            f"`{name}` for attribute {a}: profile {shape!r} has no width; "
            f"give {{'shape': {shape!r}, 'sd': ...}} or "
            f"{{'shape': {shape!r}, 'decay_rate': ...}}.")
    try:
        profile = _resolve_profile(shape, sd=sd, width=width,
                                   decay_rate=rate)
    except (TypeError, ValueError) as err:
        raise ValueError(f"`{name}` for attribute {a}: {err}") from None
    if profile.kind == "anchored":
        raise ValueError(
            f"`{name}` for attribute {a}: profile {shape!r} is anchored at "
            f"the first and last events' values, not at the sweep value, "
            f"so it cannot be aligned; weight the events with "
            f"weight_events before the call instead.")
    if edges is None and profile.kind == "family" and profile.shape == 1.0:
        edges = "closed" if defaulted else "halfOpen"
    try:
        closed = _resolve_edges(edges, profile)
    except ValueError as err:
        raise ValueError(f"`{name}` for attribute {a}: {err}") from None
    return _Window(profile, closed)


def _is_rect(win):
    """Whether a window is a pure rectangle, whose profile is piecewise
    constant in the sweep value."""
    prof = win.profile
    return (getattr(prof, "kind", None) == "family"
            and float(getattr(prof, "shape", np.nan)) == 1.0
            and np.isfinite(prof.sd))


def _rect_pieces(p_context, a, locate, win, lo, hi, is_per=False,
                 period=0.0):
    """Default sweep values for a rectangular window placed alone.

    As the window moves, the windowed context changes only where an event
    enters or leaves it: at each event's located value plus or minus half
    the window's width (and their images a period apart, on a periodic
    attribute). Between these breakpoints the profile is constant, so each
    piece is sampled just inside both its ends, and a line through the
    values draws the steps exactly, every value being the profile's value
    at its sweep value. The range ``[lo, hi]`` is sampled at its ends.
    """
    hw = float(win.profile.sd) * np.sqrt(3.0)
    loc = np.asarray(_locate_row(p_context[a], _resolve_locate(locate, a)),
                     dtype=float).ravel()
    loc = loc[np.isfinite(loc)]
    b = np.concatenate([loc - hw, loc + hw])
    if is_per and period and period > 0:
        k_lo = int(np.floor((lo - b.max()) / period)) - 1
        k_hi = int(np.ceil((hi - b.min()) / period)) + 1
        b = np.concatenate([b + k * period for k in range(k_lo, k_hi + 1)])
    tol = 1e-9 * max(1.0, abs(lo), abs(hi), hw)
    b = np.unique(b[(b > lo + tol) & (b < hi - tol)])
    if b.size:
        b = b[np.concatenate([[True], np.diff(b) > tol])]
    edges = np.concatenate([[lo], b, [hi]])
    gaps = np.diff(edges)
    eps = min(1e-6 * hw, 0.25 * float(gaps[gaps > 0].min())) \
        if np.any(gaps > 0) else 0.0
    vals = [lo]
    for x in b:
        vals.extend([x - eps, x + eps])
    if hi > lo:
        vals.append(hi)
    return np.asarray(vals, dtype=float)


def _window_default(p_context, a, start, stop, step, default_step, win,
                    locate, is_per, period):
    """Generated sweep values where only a window is placed: a pure
    rectangle's pieces (see :func:`_rect_pieces`) where no step is given,
    and otherwise the uniform grid of :func:`_generate`."""
    if (step.get(a) is None and win is not None and _is_rect(win)):
        v = _axis_values(p_context, a)
        if v.size == 0:
            raise ValueError(
                f"attribute {a}: the context has no finite values there to "
                f"take a default `start` / `stop` from.")
        lo = float(v.min()) if start.get(a) is None else float(start[a])
        hi = float(v.max()) if stop.get(a) is None else float(stop[a])
        if hi < lo:
            raise ValueError(f"attribute {a}: `stop` is below `start`.")
        per = bool(is_per[a]) if is_per is not None else False
        pd_ = float(period[a]) if (period is not None
                                   and period[a] is not None) else 0.0
        return _rect_pieces(p_context, a, locate, win, lo, hi, per, pd_)
    return _generate(p_context, a, start.get(a), stop.get(a), step.get(a),
                     default_step)


def _values(v, a, name):
    arr = np.asarray(v, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"`{name}` for attribute {a} is empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            f"`{name}` for attribute {a} holds a non-finite value.")
    return arr


def _generate(p_context, a, start, stop, step, default_step,
              default_range=None, open_stop=False):
    """Sweep values from ``start`` to ``stop`` in steps of ``step``.

    ``start`` and ``stop`` default to ``default_range`` where given, and
    otherwise to the lowest and highest of the context's values on
    attribute ``a``; ``step`` defaults to ``default_step``. With
    ``open_stop`` (a periodic range) the stop value itself is left out."""
    if step is None and default_step is None:
        raise ValueError(
            f"attribute {a}: give `step`; there is no kernel width or window "
            f"sd on this attribute to take a default step from.")
    if default_range is None and (start is None or stop is None):
        vals = _axis_values(p_context, a)
        if vals.size == 0:
            raise ValueError(
                f"attribute {a}: the context has no finite values there to "
                f"take a default `start` / `stop` from.")
        default_range = (float(vals.min()), float(vals.max()))
    lo = float(default_range[0]) if start is None else float(start)
    hi = float(default_range[1]) if stop is None else float(stop)
    st = float(default_step) if step is None else float(step)
    if not (np.isfinite(st) and st > 0):
        raise ValueError(f"attribute {a}: `step` must be finite and > 0.")
    if hi < lo:
        raise ValueError(f"attribute {a}: `stop` is below `start`.")
    count = int(np.floor((hi - lo) / st + 1e-9)) + 1
    vals = lo + st * np.arange(count)
    if open_stop and stop is None:
        vals = vals[vals < hi - 1e-9 * max(1.0, abs(hi))]
    return vals


def _kernel_width(sigma, a):
    """One kernel standard deviation for attribute ``a``: the scalar sigma,
    the square root of the largest diagonal entry of a kernel covariance,
    or the smallest finite entry of a per-level vector; ``None`` where
    there is none."""
    if sigma is None:
        return None
    try:
        sg = sigma[a]
    except (IndexError, KeyError, TypeError):
        return None
    if sg is None:
        return None
    arr = np.asarray(sg, dtype=float)
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1] and arr.size > 1:
        d = np.diag(arr)
        d = d[np.isfinite(d) & (d > 0)]
        return float(np.sqrt(d.max())) if d.size else None
    v = arr.ravel()
    v = v[np.isfinite(v) & (v > 0)]
    return float(v.min()) if v.size else None


def _tuple_dim(r, specs, a):
    """The number of coordinates of attribute ``a``'s tuple: the product of
    a nested attribute's per-level tuple sizes, or its tuple size ``r``;
    1 where neither is given."""
    if specs is not None and a < len(specs) and specs[a] is not None:
        rs = specs[a].get("r") if isinstance(specs[a], dict) else None
        if rs is not None:
            return int(np.prod(np.atleast_1d(np.asarray(rs, dtype=float))))
    if r is None:
        return 1
    try:
        ra = r[a]
    except (IndexError, KeyError, TypeError):
        return 1
    if ra is None:
        return 1
    return int(np.prod(np.atleast_1d(np.asarray(ra, dtype=float))))


def _peak_width(sigma, r, specs, a):
    """The standard deviation of the peaks of a translation profile on
    attribute ``a``. Translating the query moves all D coordinates of the
    attribute's tuple alike, so each pair of tuples contributes a Gaussian
    in the offset of variance 2 / (1' Sigma^-1 1) (Milne 2026, Eq. 10 and
    Online Supplement Sec. 4): sigma * sqrt(2 / D) for an isotropic kernel
    of width sigma, narrower the larger the tuple. ``None`` where there is
    no kernel width."""
    try:
        sg = None if sigma is None else sigma[a]
    except (IndexError, KeyError, TypeError):
        sg = None
    if sg is not None:
        arr = np.asarray(sg, dtype=float)
        if arr.ndim == 2 and arr.shape[0] == arr.shape[1] and arr.size > 1:
            try:
                q = float(np.ones(arr.shape[0])
                          @ np.linalg.solve(arr, np.ones(arr.shape[0])))
            except np.linalg.LinAlgError:
                q = np.nan
            if np.isfinite(q) and q > 0:
                return float(np.sqrt(2.0 / q))
    kw = _kernel_width(sigma, a)
    if kw is None:
        return None
    return kw * float(np.sqrt(2.0 / max(_tuple_dim(r, specs, a), 1)))


def _lattice_step(p_context, p_query, a, half, period=0.0):
    """The default translation step on attribute ``a``: at most ``half``
    (half the standard deviation of the profile's peaks), and a whole
    fraction of the lattice the exact matches lie on, where they lie on
    one.

    Every exact match is at an offset that is a difference between a
    context value and a query value. Where the context's values are whole
    multiples of a spacing ``g`` apart, and so are the query's (and ``g``
    divides the period, on a periodic attribute), every such offset is the
    lowest one, min(context) - max(query), plus a whole multiple of ``g``.
    A step of ``g / k``, with ``k`` the smallest whole number that brings
    it to ``half`` or below, then puts every one of them on a grid through
    that lowest offset: onsets on a grid of 1 with a kernel width of 0.307
    step at 1/7, not at 0.1535. Where ``g`` is below ``half``, it is the
    step itself, provided it is at least ``half / 4``; otherwise (values on
    no lattice, or on one too fine to step at) the step is ``half``, which
    keeps every peak within a quarter of its standard deviation of a grid
    point.
    """
    diffs = []
    for p_attr in (p_context, p_query):
        v = _axis_values(p_attr, a)
        if v.size:
            diffs.append(v - v.min())
    if period and period > 0:
        diffs.append(np.array([float(period)]))
    if not diffs:
        return half
    tol = 1e-6 * half
    d = np.unique(np.concatenate(diffs))
    d = d[d > tol]
    if d.size == 0:
        return half
    floor = half / 4.0
    g = float(d[0])
    for x in d[1:]:
        big, small = max(g, float(x)), min(g, float(x))
        while small > tol:
            r = np.fmod(big, small)
            if small - r <= tol:
                r = 0.0
            big, small = small, r
        g = big
        if g < floor:
            return half
    # The tolerant Euclid can drift; accept g only if every difference is a
    # whole multiple of it.
    if np.max(np.abs(d / g - np.round(d / g))) * g > 1e-4 * half:
        return half
    return float(g / np.ceil(g / half - 1e-9))


def _translation_range(p_context, p_query, a, ref, periodic, period):
    """The sweep values at which the query, placed by its reference
    ``ref``, overlaps the context on attribute ``a``: one period from 0
    (plus ``ref``) on a periodic attribute; otherwise from the value that
    puts the query's highest value on the context's lowest to the value
    that puts its lowest on the context's highest."""
    if periodic and period and period > 0:
        return (ref, ref + float(period)), True
    c = _axis_values(p_context, a)
    q = _axis_values(p_query, a)
    if c.size == 0 or q.size == 0:
        raise ValueError(
            f"attribute {a}: the context or the query has no finite values "
            f"there to take a default sweep range from; give `sweep` values "
            f"or `start` / `stop`.")
    return (float(c.min() - q.max()) + ref, float(c.max() - q.min()) + ref), \
        False


def _is_rel(specs, is_rel, a, n):
    return _axis_is_rel(specs, [False] * n if is_rel is None else is_rel, a)


def _build_plan(p_context, p_query, specs, is_rel, sweep, start, stop, step,
                align, window, drop, query_ref, locate, func, sigma=None,
                is_per=None, period=None, r=None):
    """Validate the placement arguments and return the sweep plan.

    One rule places everything: at each sweep value ``s``, a window has its
    reference (displacement 0) at ``s``, and a translated query has its
    reference ``query_ref`` at ``s``.

    Returns ``(dims, drop, ctx_win, q_ref)``: ``dims`` is the list of
    output dimensions in order, each ``(kind, a, values, pair)`` with
    ``kind`` ``'ctx'`` (a window is aligned at each value; for ``'both'``
    the query's reference is placed there too) or ``'query'`` (only the
    query is translated), and ``pair`` the index of the window's dimension
    where an ``'independent'`` query list holds one row per window value;
    ``ctx_win`` maps each attribute where a window is aligned to ``(gamma,
    sd, closed)`` and to whether the query is translated with it; ``q_ref``
    maps each attribute the query is translated on to ``query_ref``.
    """
    n = len(p_context)
    has_query = p_query is not None
    # A bare attribute index, or a list of them, asks for default sweep
    # values on each.
    if isinstance(sweep, (int, np.integer)) and not isinstance(sweep, bool):
        sweep = {int(sweep): None}
    elif isinstance(sweep, (list, tuple)) and sweep and all(
            isinstance(k, (int, np.integer)) and not isinstance(k, bool)
            for k in sweep):
        sweep = {int(k): None for k in sweep}
    sweep = _attr_map(sweep, n, "sweep")
    # A bare number for start, stop, or step applies to the one swept
    # attribute: sweep=a, step=0.5.
    start = _attr_map(_bare_generator(start, "start", sweep), n, "start")
    stop = _attr_map(_bare_generator(stop, "stop", sweep), n, "stop")
    step = _attr_map(_bare_generator(step, "step", sweep), n, "step")
    window = _attr_map(window, n, "window")
    drop = _attr_list(drop, n, "drop")
    swept = set(sweep) | set(start) | set(stop) | set(step)
    if has_query:
        align = _attr_map(align, n, "align")
        query_ref = _attr_map(query_ref, n, "query_ref")
        # Attribute translation over the whole context is the default role.
        for a in swept - set(align):
            align[a] = "query"
    else:
        # swept_entropy: every swept attribute's values align the window.
        align = {a: "window" for a in swept | set(window)}
        query_ref = {}

    for a, m in align.items():
        if m not in _ALIGN:
            raise ValueError(
                f"`align` for attribute {a} must be one of {_ALIGN}; "
                f"got {m!r}.")
    for a in sorted(set(align) - swept):
        raise ValueError(
            f"attribute {a} has an `align` entry but no sweep values; give "
            f"them with sweep={{{a}: values}} (or `start` / `stop` / "
            f"`step`).")
    if not align:
        raise ValueError(
            f"{func}: name at least one attribute to sweep, with `sweep` (or "
            f"`start` / `stop` / `step`).")

    # The query's reference, wherever the query is translated: the point of
    # the query placed at each sweep value. By default 0 where there is no
    # window ('query': sweep values are then the offsets added to the query
    # as written), and the query's middle where there is one ('both',
    # 'independent': window and query sweep values then both refer to
    # middles, so under 'both' the window is aligned at the query's middle).
    q_ref = {}
    for a, m in align.items():
        if m == "window":
            continue
        if a in query_ref:
            q_ref[a] = float(query_ref[a])
        elif m in ("both", "independent"):
            q_ref[a] = _query_position(p_query, a, locate)
        else:
            q_ref[a] = 0.0

    dims, ctx_win = [], {}
    for a in sorted(align):
        m = align[a]
        gen = any(a in d for d in (start, stop, step))
        rel = _is_rel(specs, is_rel, a, n)
        # --- the window ---
        if m == "query":
            if a in window:
                raise ValueError(
                    f"attribute {a} has a window, but its `align` is 'query' "
                    f"(the default: translation over the whole context), "
                    f"which has none. Say where the window goes: "
                    f"align={{{a}: 'both'}} (window and query at each sweep "
                    f"value), 'window' (the window only), or 'independent'.")
            win = None
        elif m == "window":
            if a not in window:
                raise ValueError(
                    (f"attribute {a}: align='window' aligns a window, so "
                     f"give its " if has_query else
                     f"attribute {a} is swept, so give its window's ")
                    + f"shape and width: window={{{a}: (shape, width)}}.")
            win = _parse_window(window[a], a, "window")
        elif m == "independent":
            if a not in window:
                raise ValueError(
                    f"attribute {a}: align='independent' aligns a window "
                    f"apart from the query, so give its shape and width: "
                    f"window={{{a}: (shape, width)}}.")
            win = _parse_window(window[a], a, "window")
        else:
            # 'both': window and query reference share each sweep value. The
            # window defaults to the smallest closed rectangle that, so
            # placed, holds the query.
            win = _parse_window(
                window.get(a), a, "window",
                lambda a=a: _holding_width(p_query, a, locate, q_ref[a]))
            _warn_if_query_cut(p_query, a, locate, q_ref[a], win)
        if win is not None and is_per is not None and bool(is_per[a]):
            # On a periodic attribute the window's displacement wraps, as
            # weight_events wraps it.
            win = win.with_geometry(True, float(period[a]))
        # --- what may be translated, given the geometry ---
        if m != "window":
            if rel:
                raise ValueError(
                    f"attribute {a} is relative: translating the query "
                    f"changes none of its within-tuple differences, so "
                    f"align={m!r} does nothing there. Use "
                    f"align='window'.")
            if a in drop:
                raise ValueError(
                    f"attribute {a}: align={m!r} translates the query along "
                    f"it, so it is compared and cannot be dropped.")
        # --- the sweep values ---
        default_step = (None if win is None
                        or not np.isfinite(win.default_step)
                        else win.default_step)
        default_range, open_stop = None, False
        if m in ("query", "both"):
            # Where the sweep values translate the query, every placement at
            # which it overlaps the context, stepped at no more than half the
            # standard deviation of the profile's peaks, sigma * sqrt(2 / D)
            # for a tuple of D coordinates (window or no window), on the
            # values' lattice where they lie on one, so that every exact
            # match is on the grid.
            per = bool(is_per[a]) if is_per is not None else False
            pd_ = float(period[a]) if (period is not None
                                       and period[a] is not None) else 0.0
            kw_ = _peak_width(sigma, r, specs, a)
            if kw_ is not None:
                default_step = _lattice_step(p_context, p_query, a,
                                             kw_ / 2.0, pd_ if per else 0.0)
            elif m == "query":
                default_step = None
            if a not in sweep or sweep[a] is None:
                default_range, open_stop = _translation_range(
                    p_context, p_query, a, q_ref[a], per, pd_)
                if open_stop and default_step is not None \
                        and step.get(a) is None:
                    # One period, on the grid through the lowest offset, so
                    # the lattice step lands on every exact match.
                    o = float(_axis_values(p_context, a).min()
                              - _axis_values(p_query, a).max())
                    sh = float(np.mod(o, default_step))
                    if default_step - sh < 1e-9 * default_step:
                        sh = 0.0
                    default_range = (default_range[0] + sh,
                                     default_range[1] + sh)
        if m == "independent":
            pair = sweep.get(a)
            if pair is None:
                raise ValueError(
                    f"attribute {a}: align='independent' needs the query's "
                    f"sweep values given explicitly, sweep={{{a}: "
                    f"(window_values, query_values)}} (the window's may be "
                    f"None, generated by `start` / `stop` / `step`).")
            if not (isinstance(pair, (tuple, list)) and len(pair) == 2):
                raise ValueError(
                    f"attribute {a}: align='independent' takes two lists, "
                    f"sweep={{{a}: (window_values, query_values)}} (the "
                    f"window's may be None when `start` / `stop` / `step` "
                    f"generate it).")
            w_vals, q_vals = pair
            if q_vals is None:
                raise ValueError(
                    f"attribute {a}: the query's sweep values must be given "
                    f"explicitly in sweep={{{a}: (window_values, "
                    f"query_values)}}.")
            if w_vals is None:
                w_vals = _window_default(p_context, a, start, stop, step,
                                         default_step, win, locate, is_per,
                                         period)
            elif gen:
                raise ValueError(
                    f"attribute {a}: give the window's sweep values either "
                    f"in `sweep` or by `start` / `stop` / `step`, not both.")
            w_vals = _values(w_vals, a, "sweep")
            q_arr = np.asarray(q_vals, dtype=float)
            if q_arr.ndim == 2:
                # One row of query values per window value: the query's
                # placements may depend on where the window is.
                if q_arr.shape[0] != w_vals.size or q_arr.shape[1] == 0:
                    raise ValueError(
                        f"attribute {a}: a 2-D list of query values needs one "
                        f"row per window value ({w_vals.size}); got shape "
                        f"{q_arr.shape}.")
                if not np.all(np.isfinite(q_arr)):
                    raise ValueError(
                        f"`sweep` for attribute {a} holds a non-finite "
                        f"value.")
            else:
                q_arr = _values(q_arr, a, "sweep")
            dims.append(("ctx", a, w_vals, None))
            dims.append(("query", a, q_arr, len(dims) - 1))
        else:
            listed = a in sweep and sweep[a] is not None
            if listed and gen:
                raise ValueError(
                    f"attribute {a}: give its sweep values either in `sweep` "
                    f"or by `start` / `stop` / `step`, not both.")
            if listed:
                vals = _values(sweep[a], a, "sweep")
            elif m == "window":
                vals = _window_default(p_context, a, start, stop, step,
                                       default_step, win, locate, is_per,
                                       period)
            else:
                vals = _generate(p_context, a, start.get(a), stop.get(a),
                                 step.get(a), default_step, default_range,
                                 open_stop)
            dims.append(("query" if m == "query" else "ctx", a, vals, None))
        if win is not None:
            ctx_win[a] = (win, m == "both")

    bad = sorted(d for d in drop if align.get(d) != "window")
    if bad:
        raise ValueError(
            f"`drop` names attribute {bad[0]}, which is not a window "
            f"attribute swept alone (align='window'). Dropping "
            f"marginalizes "
            f"a window attribute after the window has weighted the events; "
            f"to leave an attribute out of the comparison altogether, leave "
            f"it out of the pre-MAETs.")
    if len(drop) >= n:
        raise ValueError(
            "every attribute is dropped; nothing is left to compare or "
            "measure.")

    for a in query_ref:
        if a not in q_ref:
            raise ValueError(
                f"`query_ref` for attribute {a}: the query is not translated "
                f"along attribute {a} (align='window'), so it has no "
                f"reference there to place.")
    return dims, drop, ctx_win, q_ref


def _query_located(p_query, a, locate):
    loc = _locate_row(p_query[a], _resolve_locate(locate, a)).ravel()
    return loc[np.isfinite(loc)]


def _holding_width(p_query, a, locate, ref):
    """The width of the smallest window that, aligned at the point where
    the query's reference ``ref`` lands, holds every one of the query's
    located values on attribute ``a``."""
    loc = _query_located(p_query, a, locate)
    w = 2.0 * float(np.max(np.abs(loc - ref))) if loc.size else 0.0
    if not w > 0:
        raise ValueError(
            f"attribute {a}: the query's events all lie at one value there, "
            f"so there is no width to take a default window from; give one, "
            f"window={{{a}: (shape, width)}}.")
    return w


def _warn_if_query_cut(p_query, a, locate, ref, win):
    """Warn where the window, aligned at the sweep value where the query's
    reference ``ref`` lands, leaves out some of the query's own events: the
    query can then never be matched in full."""
    loc = _query_located(p_query, a, locate)
    if not loc.size:
        return
    out = int(np.sum(_window_factor(loc, ref, win) == 0.0))
    if out:
        warnings.warn(
            f"attribute {a}: the window, aligned where the query's reference "
            f"lands, leaves out {out} of the query's {loc.size} events, so "
            f"the query can never be matched in full. Widen the window"
            + (", move query_ref towards the query's middle, or close the "
               "rectangle (edges 'closed')"
               if (win.profile.kind == "family" and win.profile.shape == 1.0
                   and not win.closed) else
               " or move query_ref towards the query's middle")
            + ".", UserWarning, stacklevel=5)


def _target(target_attr, drop, n):
    keep = [i for i in range(n) if i not in drop]
    target = keep[0] if target_attr is None else _abs_idx(target_attr, n)
    if target in drop:
        raise ValueError(
            f"target_attr={target} is a dropped attribute: its weights are "
            f"removed before the build, so the window factors would be "
            f"lost. Choose a compared attribute.")
    return target, keep


def _translate(p, w, specs, shifts, n):
    """The query with attribute ``a`` translated by ``shifts[a]``."""
    if not shifts:
        return p, w, specs
    offs = [None] * n
    for a, mu in shifts.items():
        offs[a] = np.array([[float(mu)]], dtype=float)
    return unpack_pre_maet(translate_attributes(p, w, offs, specs=specs))


def _run_similarity(p_context, w_context, p_query, w_query, sigma, r, is_rel,
                    is_per, period, is_exch, specs, query_specs, plan, locate,
                    target_attr, normalize):
    dims, drop, ctx_win, q_ref = plan
    n = len(p_context)
    p_context, p_query = list(p_context), list(p_query)
    target, keep = _target(target_attr, drop, n)
    nested = specs is not None
    sg, rr, rl, pr, pd = (_sub(sigma, keep), _sub(r, keep), _sub(is_rel, keep),
                          _sub(is_per, keep), _sub(period, keep))
    sy = None if is_exch is None else _sub(is_exch, keep)
    exch_args = () if sy is None else (sy,)
    win = {a: spec for a, (spec, _) in ctx_win.items()}
    shape = tuple(d[2].shape[-1] for d in dims)
    out = np.empty(shape, dtype=float)
    c_pos = [i for i, d in enumerate(dims) if d[0] == "ctx"]
    q_pos = [i for i, d in enumerate(dims) if d[0] == "query"]
    q_shape = tuple(shape[i] for i in q_pos)
    q_idx = list(np.ndindex(*q_shape))

    def compare(pc, wc, sc, pq, wq, sq):
        if nested:
            dc = build_maet(pc, wc, sigma=sg, is_per=pr, period=pd, specs=sc,
                            verbose=False)
            dq = build_maet(pq, wq, sigma=sg, is_per=pr, period=pd, specs=sq,
                            verbose=False)
            return float(sim_maet(dc, dq, normalize=normalize, verbose=False))
        return float(sim_maet(pc, wc, pq, wq, sg, rr, rl, pr, pd, *exch_args,
                              normalize=normalize, verbose=False))

    for ci in np.ndindex(*tuple(shape[i] for i in c_pos)):
        at = {dims[i][1]: float(dims[i][2][j]) for i, j in zip(c_pos, ci)}
        # Every window is aligned at its sweep value. For 'both', the query
        # is translated so that its reference lands at the same value.
        both = {a: s - q_ref[a] for a, s in at.items() if ctx_win[a][1]}
        pc_w, wc_w, sc_w = _apply_windows(p_context, w_context, specs,
                                          at, win, locate, target)
        pc, wc, sc, _ = _drop_axes(pc_w, wc_w, sc_w, drop)
        pq_b, wq_b, sq_b = _translate(p_query, w_query, query_specs, both, n)
        full = [0] * len(dims)
        for i, j in zip(c_pos, ci):
            full[i] = j
        if not q_pos:
            pq, wq, sq, _ = _drop_axes(pq_b, wq_b, sq_b, drop)
            out[tuple(full)] = compare(pc, wc, sc, pq, wq, sq)
            continue
        # The windowed context is fixed across the query's own translations, so
        # they are computed together where sweep_sim_maet applies.
        def q_value(k, j):
            vals, pair = dims[k][2], dims[k][3]
            return vals[full[pair], j] if vals.ndim == 2 else vals[j]

        offs = np.zeros((len(keep), len(q_idx)), dtype=float)
        for m, qi in enumerate(q_idx):
            for k, j in zip(q_pos, qi):
                a = dims[k][1]
                offs[keep.index(a), m] = q_value(k, j) - q_ref[a]
        pq0, wq0, sq0, _ = _drop_axes(pq_b, wq_b, sq_b, drop)
        row = _sweep_row(pc, wc, sc, pq0, wq0, sq0, sg, rr, rl, pr, pd,
                         exch_args, nested, offs, normalize)
        for m, qi in enumerate(q_idx):
            for k, j in zip(q_pos, qi):
                full[k] = j
            if row is not None:
                out[tuple(full)] = row[m]
                continue
            shifts = {dims[k][1]: q_value(k, j) - q_ref[dims[k][1]]
                      for k, j in zip(q_pos, qi)}
            pq_t, wq_t, sq_t = _translate(pq_b, wq_b, sq_b, shifts, n)
            pq, wq, sq, _ = _drop_axes(pq_t, wq_t, sq_t, drop)
            out[tuple(full)] = compare(pc, wc, sc, pq, wq, sq)
    return out


def _run_local(p_context, w_context, sigma, r, is_rel, is_per, period,
               is_exch, specs, plan, locate, target_attr, measure):
    """At each sweep value, window the context, build the windowed
    density, and apply ``measure`` to it (``swept_entropy``,
    ``swept_mass``)."""
    dims, drop, ctx_win, _ = plan
    n = len(p_context)
    p_context = list(p_context)
    target, keep = _target(target_attr, drop, n)
    nested = specs is not None
    sg, rr, rl, pr, pd = (_sub(sigma, keep), _sub(r, keep), _sub(is_rel, keep),
                          _sub(is_per, keep), _sub(period, keep))
    sy = None if is_exch is None else _sub(is_exch, keep)
    exch_args = () if sy is None else (sy,)
    win = {a: spec for a, (spec, _) in ctx_win.items()}
    out = np.empty(tuple(d[2].size for d in dims), dtype=float)
    for idx in np.ndindex(*out.shape):
        at = {dims[i][1]: float(dims[i][2][j]) for i, j in enumerate(idx)}
        pc_w, wc_w, sc_w = _apply_windows(p_context, w_context, specs, at,
                                          win, locate, target)
        pc, wc, sc, _ = _drop_axes(pc_w, wc_w, sc_w, drop)
        if nested:
            dens = build_maet(pc, wc, sigma=sg, is_per=pr, period=pd,
                              specs=sc, verbose=False)
        else:
            dens = build_maet(pc, wc, sg, rr, rl, pr, pd, *exch_args,
                              verbose=False)
        out[idx] = float(measure(dens))
    return out


def _run_entropy(p_context, w_context, sigma, r, is_rel, is_per, period,
                 is_exch, specs, plan, locate, target_attr, method, base,
                 grid=None):
    from ..entropy import entropy_maet
    grid = {k: v for k, v in (grid or {}).items() if v is not None}
    return _run_local(
        p_context, w_context, sigma, r, is_rel, is_per, period, is_exch,
        specs, plan, locate, target_attr,
        lambda dens: entropy_maet(dens, method=method, base=base,
                                  verbose=False, **grid))


def _run_mass(p_context, w_context, sigma, r, is_rel, is_per, period,
              is_exch, specs, plan, locate, target_attr, region, normalize):
    from .mass import mass_maet
    n = len(p_context)
    drop = plan[1]
    keep = [i for i in range(n) if i not in drop]
    reg = {}
    for a, spec in _attr_map(region, n, "region").items():
        if a in drop:
            raise ValueError(
                f"`region` names attribute {a}, which is dropped: it is "
                f"marginalized before the mass is taken. Keep it, or "
                f"leave it out of the region.")
        reg[keep.index(a)] = spec
    return _run_local(
        p_context, w_context, sigma, r, is_rel, is_per, period, is_exch,
        specs, plan, locate, target_attr,
        lambda dens: mass_maet(dens, reg or None, normalize=normalize))


@_with_dispatch_scope
def swept_similarity(p_context, w_context=None, p_query=None,
                     w_query=None, sigma=None, r=None,
                     is_rel=None, is_per=None, period=None, *,
                     sweep=None, start=None, stop=None, step=None,
                     align=None, window=None, drop=None,
                     query_ref=None, locate="centroid",
                     target_attr=None, normalize="oneSidedDenom",
                     is_exch=None, rel=None, exch=None, specs=None,
                     return_offsets=False, return_sweep_values=False,
                     verbose=False):
    r"""Compare a query with a context at each of a list of sweep values
    (a pre-MAET cross-correlation).

    ``S = swept_similarity(pm_context, pm_query, sweep={a: values})``
    or ``sweep=a`` for default sweep values.

    **Overview.** The query is compared with the context at each of a list
    of values on an attribute ``a``, the *sweep values*, giving a profile
    ``S`` whose peaks show where the query best matches the context. At
    each sweep value the query is translated there (attribute translation,
    as by :func:`translate_attributes`), a window on the context is aligned
    there (event weighting, as by :func:`weight_events`), or both. By
    default only the query is translated, and it is compared with the whole
    context: the cross-correlation of the query against the context. A
    window restricts each comparison to a local region of the context; on
    its own, it compares the query, as written, with each region in turn.

    The similarities are computed in one of two ways. Where the context is
    the same across the query's translations (always under ``'query'``, and
    across the query list of ``'independent'`` at each window position),
    :func:`sweep_sim_maet` computes them all in one pass. Where the window
    changes with each sweep value (``'both'``, ``'window'``),
    :func:`sim_maet` computes them one sweep value at a time. Both give the
    same values, to numerical precision: :func:`sweep_sim_maet` returns what
    :func:`sim_maet` would at each translation, but much faster.

    **Two attributes.** The *swept attribute* ``a`` is the one the sweep
    values lie on: the query is translated along it, and a window is a
    function of displacement along it. The *target attribute*
    (``target_attr``, by default the first attribute not dropped) is the one
    whose per-event weights a window multiplies. They are usually different:
    a window over time weights the pitch events.

    **The rule.** At each sweep value :math:`s` on the swept attribute:

    - the query, where it is translated, has its reference value
      ``query_ref`` at :math:`s`: it is translated by
      :math:`\mu = s - \mathrm{queryRef}`;
    - a window, where there is one, has its reference value,
      :math:`\delta = 0` (the midpoint of its symmetric shape), at :math:`s`.

    Nothing else places anything. For each swept attribute, ``align`` says
    which of the two are placed:

    - ``'query'``: the query only (the default): attribute translation over
      the whole context. ``query_ref`` defaults to 0, so the sweep values
      are the offsets added to the query as written (transpositions in
      cents, time shifts).
    - ``'both'``: the query and a window, at the same :math:`s`: a local
      comparison. ``query_ref`` defaults to the query's middle (the mean of
      its events' values on the swept attribute), so the window is centred
      on the query.
    - ``'window'``: a window only; the query is left as written.
    - ``'independent'``: a window at each value of one list and the query
      at each value of another, in every combination (a correlogram).
      ``query_ref`` defaults to the query's middle.

    A window :math:`h(\delta)` aligned at :math:`s` weights each context
    event :math:`n` on the target attribute:

    .. math:: w'(n) = w(n)\, h\bigl(p_a(n) - s\bigr),

    where :math:`p_a(n)` is event :math:`n`'s value on the swept attribute
    :math:`a` (where the event holds several values, ``locate`` reduces
    them to one), so events far from :math:`s` are attenuated.

    The window acts on the pre-MAET, before any density is built:
    :math:`p_a(n)` is the value the pre-MAET holds for event :math:`n` (its
    onset time, say), and only the weights change. Whether the swept
    attribute is then built in absolute or relative mode, or dropped,
    matters only afterwards, when the density is built from the weighted
    events. A relative time attribute of bound events, for example, is
    still windowed by onset time (each event's onsets reduced to one by
    ``locate``), and then compared through its onsets measured from the
    first within each event.

    With ``return_offsets=True`` the translation applied to the query at
    each sweep value, :math:`\mu = s - \mathrm{queryRef}`, is returned too:
    its shift from where it was written, whatever ``query_ref`` is. Plot
    against it to read a windowed sweep as offsets. With query and context
    written from a common origin (the query at the time it was taken from,
    say), :math:`\mu` keeps its meaning through preprocessing that keeps
    the values, such as differencing, so profiles with and without it share
    one axis.

    The query itself is not windowed here. To weight the query's own events
    by a window, apply :func:`weight_events` to the query before the call.

    **Choosing a role.**

    - ``'query'``: the canonical sweep: where in the context, or at which
      transposition, the query best matches the context as a whole. Only
      the kernel width ``sigma`` limits which context events count; all
      sweep values are computed in one pass.
    - ``'both'``: a local ``'query'``: the window fixes the region of the
      context that counts around the query. Under ``normalize='cosine'``
      unmatched material inside the window lowers the score and material
      outside it is ignored; as the window widens, ``'both'`` becomes
      ``'query'``.
    - ``'window'``: the query is not translated: the window steps through
      the context and the query, as written, is compared with each region
      in turn. What this measures depends on the treatment of the swept
      attribute (below).
    - ``'independent'``: each window position gives a whole profile: for a
      best placement that changes across the context, such as a lag
      between two parts that drifts over time.

    **Treatment of the swept attribute.** Whether the swept attribute is
    absolute or relative (its specs), and whether it is dropped (``drop``),
    decides what it contributes to each comparison, whatever the role:

    - *Absolute*: compared by position, so translating the query along it
      changes where the query matches, and all four roles apply. Under
      ``'window'`` the query is compared in place: the profile shows where
      in the context its match with the query as written comes from (two
      parts of a piece on a shared time axis, say, whose similarity the
      window resolves in time; under the default normalization, windows
      that tile the context give contributions that sum to the whole-piece
      similarity). To find where the query occurs, translate it
      (``'query'`` or ``'both'``).
    - *Relative*: compared only up to a common translation of each tuple,
      that is, through its values relative to the lowest (a chord's
      intervals above its bass, or a bound event's onsets measured from its
      first), so the query's internal spacing must match but its position
      does not matter. Translation leaves these relative values unchanged,
      so only ``'window'`` applies (it gives what ``'both'`` would). The
      events are still windowed by the values the pre-MAET holds (above).
    - *Dropped*: marginalized after the window has weighted the events
      (``drop``), so not compared at all: the query is compared with what
      the region contains, not where in it (a local key, say). Translation
      has nothing to act on, so only ``'window'`` applies.

    Event differencing (:func:`difference_events`) is not a further
    treatment but a change of values: the attribute then holds first
    differences between successive events (inter-onset intervals, pitch
    steps), and is absolute or relative like any other. As the swept
    attribute, its windows therefore select by interval size, and
    translation adds the same amount to every interval (on logarithmically
    rescaled inter-onset intervals, a tempo change). Usually the attribute
    differenced (pitch, say) is not the one swept (time), which differencing
    passes through unchanged at order 0, keeping each event's onset.

    **When a window on the context is needed.** Translation already
    localizes on a compared absolute attribute: the kernel lets the query
    match only material near where it is translated. A window on the
    context is indispensable where translation cannot localize: on a
    dropped or relative swept attribute, alone or alongside translation on
    another attribute (each bar windowed on time, time dropped, and the
    query translated in pitch: the bar and the transposition of each
    statement at once). On a translated attribute (``'both'``,
    ``'independent'``) it does nearly what weighting the query would
    (:func:`weight_events`, aligned at the matching point of the query,
    then translated), the two differing only in whether a near miss is
    weighted where the context's event lies or where the query's does; what
    it adds there is the ``'cosine'`` denominator, the norm of what the
    window keeps. To ask which part of the query matches, weight the query.

    **Reading the output.** ``S`` has one dimension per sweep list, in
    attribute order; ``'independent'`` contributes two, the window's first.
    A sweep value is where the query's reference lands (under ``'window'``,
    where the window is aligned). The offsets are a dict ``{a: mu}``, one
    array per attribute the query is translated along, the same shape as
    its (query) sweep list. With ``return_sweep_values=True`` the sweep
    values themselves are returned too, as a dict ``{a: values}``, the axes
    of ``S``: under ``'query'`` they equal the offsets (``query_ref`` is
    0), but under ``'both'`` they are where the query's middle and the
    window lie, and under ``'window'`` there are no offsets at all.

    **Input forms**, in the order to reach for them:

    - ``swept_similarity(pm_context, pm_query, ...)``, with two whole
      pre-MAETs (the canonical entry). The geometry is taken from their
      specs. Any of the six per-attribute parameters (``sigma``,
      ``is_per``, ``period``, ``r``, ``rel``, ``exch``) may be given
      alongside to override it, as at :func:`build_maet`, either in full or
      selectively as a length-A list whose ``None`` entries keep the spec's
      value. The two pre-MAETs describe one comparison, so they must agree
      on ``r``, ``rel``, ``exch``, and the nesting; ``sigma``, ``is_per``,
      and ``period`` may differ, and the context's are used.
    - ``swept_similarity(p_context, w_context, p_query, w_query, sigma,
      r, is_rel, is_per, period, ...)``, the raw positional form, with each
      operand's per-attribute values and weights and the shared geometry
      written out as for :func:`sim_maet`.

    Parameters
    ----------
    p_context, w_context, p_query, w_query
        Raw form: each operand's per-attribute value matrices and weights,
        as at :func:`sim_maet`. Pre-MAET form: the context and query
        pre-MAETs are the first two arguments.
    sigma, r, is_rel, is_per, period
        Raw form: the shared per-attribute geometry, as at
        :func:`build_maet`. Pre-MAET form: optional overrides of the specs,
        with ``rel`` and ``exch`` naming the other two.
    sweep : dict, int, or list of int
        ``{a: values}``: the sweep values of attribute ``a``. A bare
        attribute index ``a``, or a list of them, asks for default sweep
        values on each (see ``start``, ``stop``, ``step``). For
        ``'independent'``, ``{a: (window_values, query_values)}``: the
        window's list may be ``None`` when ``start`` / ``stop`` / ``step``
        generate it, and the query's may be 2-D, one row per window value,
        when the query's placements depend on where the window is (a lag
        measured from each window value, say).
    start, stop, step : dict or float, optional
        ``{a: value}``: generate attribute ``a``'s sweep values from
        ``start`` to ``stop`` in steps of ``step``, in place of listing
        them; each overrides one default. A bare number applies to the
        swept attribute where ``sweep`` names one (``sweep=1,
        step=0.5``). The defaults depend on the role. Where the sweep
        values translate the query (``'query'``, ``'both'``), ``start``
        and ``stop`` cover every placement at which the query overlaps
        the context (from its highest value on the context's lowest to
        its lowest on the context's highest), or one period on a
        periodic attribute. ``step`` is then at most ``h``, half the
        standard deviation of the profile's peaks: translating the query
        moves all D coordinates of the attribute's tuple alike, so the
        peaks have standard deviation ``sigma * sqrt(2 / D)``, narrower
        the larger the tuple (D = r, or the product of a nested
        attribute's per-level r; for a kernel covariance Sigma,
        ``sqrt(2 / (1' Sigma^-1 1))``), window or no window. The step is
        also chosen so that every exact match lies on the grid. Where
        the context's values are whole multiples of a spacing ``g``
        apart, and so are the query's (and ``g`` divides the period, on
        a periodic attribute), every exact match is at the lowest
        offset plus a whole multiple of ``g``, so the step is ``g /
        k``, with ``k`` the smallest whole number that brings it to
        ``h`` or below: onsets on whole beats with ``h = 0.15`` step at
        1/7, not at 0.15, which would miss the whole-beat offsets. Where
        ``g`` is below ``h`` it is the step itself, if at least
        ``h / 4``. Otherwise (values on no such lattice, or on one too
        fine) the step is ``h``, which leaves every peak within a
        quarter of its standard deviation of a grid point, at about 97%
        of its height or more. Where they place a
        window only (``'window'``, and the window's list of
        ``'independent'``), ``start`` and ``stop`` are the lowest and
        highest of the context's values on the attribute, and ``step``
        is half the window's standard deviation, since the profile
        changes on the scale of the window (a profile function has no
        width, so give ``step``). For largely separate windows, as when
        the profile's values are to be used as data, give ``step`` as half
        the window's width (``sqrt(3)`` sd), so that neighbouring windows
        overlap by half. A pure rectangle (``'rect'``, or shape 1) makes
        the profile piecewise constant: the windowed context changes only
        where an event enters or leaves it, at each event's value plus or
        minus half the width. Without a given ``step``, its default sweep
        values are these pieces, each sampled just inside both its ends,
        so that every value is the profile's value at its sweep value and
        a line plot draws the steps exactly. The query's list of
        ``'independent'`` is always given explicitly.
    align : dict, optional
        ``{a: 'query' | 'both' | 'window' | 'independent'}``: what is placed
        at attribute ``a``'s sweep values (*The rule*). Default ``'query'``
        for every swept attribute.
    window : dict
        ``{a: (shape, width)}``, ``{a: (shape, width, edges)}``,
        ``{a: {'shape': ..., 'width' | 'sd' | 'decay_rate': ...,
        'edges': ...}}``, or ``{a: f}``: the window :math:`h` on attribute
        ``a``, any profile of :func:`weight_events`, which evaluates it.
        ``shape`` is ``'rect'``, ``'gaussian'``, or a number in [0, 1]
        blending the two (0 Gaussian, 1 rectangle); ``width`` is the full
        width of the rectangle, and a Gaussian of the same width has
        standard deviation width / (2 sqrt 3), which ``sd`` may give
        instead. ``'exponential'`` decays on both sides of the window's
        reference value, and ``'exponentialBefore'`` /
        ``'exponentialAfter'`` on one side only (zero on the other), scaled
        by ``sd`` or ``decay_rate``; a callable ``f`` takes the
        displacement :math:`p_a(n) - s` and returns the factors. The
        serial-position profiles of :func:`weight_events`, anchored at the
        first and last events rather than at the sweep value, are refused.
        On a periodic attribute the displacement wraps. ``edges``, for
        rectangles: ``'halfOpen'`` (the default for a given width: the lower
        edge included, the upper not, so that windows a width apart share
        no event, for tiling a context) or ``'closed'`` (both edges, for
        holding a query). Required for ``'window'`` and ``'independent'``,
        where the width is the scale of the local region and nothing in the
        data can supply it; not allowed for ``'query'``. For ``'both'`` it
        may be left out, or given with width ``None``: the window is then
        the smallest one that, placed by *the rule*, holds the query, with a
        closed rectangle unless another shape or ``edges`` is given, so an
        exact match scores 1. A window given for ``'both'`` that leaves out
        some of the query's own events draws a warning, since the query can
        then never be matched in full.
    drop : int or list of int, optional
        Attributes marginalized after the window has weighted the events.
        Only attributes whose ``align`` is ``'window'``.
    query_ref : dict, optional
        ``{a: value}``: the query's reference value on attribute ``a``, the
        point of the query placed at each sweep value; only where the query
        is translated. Default 0 for ``'query'`` and the query's middle for
        ``'both'`` and ``'independent'``. Under ``'query'`` it only
        relabels the output (a sweep value :math:`s` under reference
        :math:`r` is the same comparison as :math:`s - r + r'` under
        :math:`r'`); under ``'both'`` it also decides which point of the
        query lies at the window's centre. Useful values:

        - 0: sweep values are the offsets :math:`\mu` added to the query as
          written.
        - The query's middle: sweep values are where the middle lands, and
          under ``'both'`` the window is aligned at the query's middle.
        - A particular point of the query, such as its first onset or its
          root: sweep values are where that point lands, the time at which
          a match starts or the key of a transposition (under ``'both'``,
          the window's centre then sits at that point).
    locate : str, callable, or dict, default ``'centroid'``
        Which single value :math:`p_a(n)` stands for an event that holds
        several values on the swept attribute (the onsets of a bound
        super-event, say), both where the window is evaluated and in the
        query's middle: ``'centroid'`` (their mean), ``'start'`` (the first),
        ``'end'`` (the last), ``'mid'`` (the midpoint of the first and last), a
        callable taking the ``(K, N)`` value matrix and returning ``N`` values,
        or a dict ``{a: rule}``, an attribute it does not name taking
        ``'centroid'``. It has no effect where each event holds one value.
    target_attr : int, optional
        The attribute whose weights the windows multiply (default: the
        first attribute not dropped).
    normalize : {'oneSidedDenom', 'cosine', 'none'}, default 'oneSidedDenom'
        As at :func:`sim_maet`, with the windowed context as the first
        operand and the query as the second. ``'oneSidedDenom'`` divides by
        the query's self inner product, so a windowed context identical to
        the query scores 1. ``'cosine'`` gives the shape-only cosine
        similarity, bounded in [-1, 1]; ``'none'`` the bare inner product.
    is_exch : array_like of bool, optional
        Per-attribute exchangeability for the raw form (``None`` keeps the
        unordered default). Needed, in particular, for ordered attributes
        carrying a matrix-valued kernel covariance. Not allowed together
        with ``specs``, whose nesting gives exchangeability level by level.
    specs : list, optional
        Nested geometry from :func:`bind_events` (raw form).
    return_offsets : bool, default False
        Also return the offsets :math:`\mu = s - \mathrm{queryRef}`, as a
        dict ``{a: mu}``.
    return_sweep_values : bool, default False
        Also return the sweep values, as a dict ``{a: values}``, one array
        per swept attribute: the values listed, or those generated from
        the defaults and ``start`` / ``stop`` / ``step``. Under
        ``'independent'`` the entry is the pair ``(window_values,
        query_values)``. Plot the profile against them.
    verbose : bool, default False
        Accepted for consistency with the other entry points; the inner
        comparisons pass ``verbose=False``. The dispatcher's one-line
        announcement of the route it chose follows the toolbox-wide
        ``show_hints`` setting instead.

    Returns
    -------
    S : np.ndarray
        One dimension per sweep list, in attribute order.
    offsets : dict
        Only with ``return_offsets=True``: ``{a: mu}``, the translation
        applied to the query along each attribute it is translated on.
    sweep_values : dict
        Only with ``return_sweep_values=True``: ``{a: values}``, the sweep
        values of each swept attribute. With both flags the order is
        ``(S, offsets, sweep_values)``.

    Examples
    --------
    Where does E-G occur in the melody C D E G C E G (one note per time
    unit)? Translating the query in time, with sweep values that are
    offsets from the query as written (it starts at time 0, so an offset is
    the time at which it starts):

    >>> import numpy as np, mpt
    >>> prev = mpt.set_default(show_hints=False)   # no route announcements
    >>> cents = lambda m: mpt.transform_attributes(
    ...     np.array(m), None, ('midi', 'cents'))[None, :]
    >>> ctx = [cents([60, 62, 64, 67, 60, 64, 67]), np.arange(7.)[None, :]]
    >>> qry = [cents([64, 67]), np.array([[0., 1.]])]
    >>> geom = ([10., 0.2], [1, 1], [False, False], [True, False],
    ...         [1200., 0.])
    >>> s = np.arange(6.)
    >>> S = mpt.swept_similarity(ctx, None, qry, None, *geom,
    ...                             sweep={1: s})
    >>> s[S > 0.99].tolist()
    [2.0, 5.0]

    The same search, local: query and window aligned together
    (``'both'``; the window by default the smallest closed rectangle that
    holds the query), stepped across the melody and read against the
    offsets:

    >>> S, mu = mpt.swept_similarity(
    ...     ctx, None, qry, None, *geom, step={1: 0.5}, align={1: 'both'},
    ...     return_offsets=True)
    >>> mu[1][S > 0.99].tolist()
    [2.0, 5.0]
    >>> _ = mpt.set_default(**prev)
    """
    query_specs = specs
    if is_pre_maet(p_context):
        if p_query is not None:
            raise TypeError(
                "swept_similarity(pm_context, pm_query, ...): the pre-MAET "
                "form takes no third positional argument; give the sweep "
                "values as sweep={a: values}.")
        (p_attrs, w_attrs, sigma, r, is_rel, is_per, period, is_exch,
         side_specs) = _swept_pre_maet_args(
            [p_context, w_context],
            {"sigma": sigma, "is_per": is_per, "period": period, "r": r,
             "rel": rel if rel is not None else is_rel, "exch": exch},
            "swept_similarity")
        p_context, p_query = p_attrs
        w_context, w_query = w_attrs
        specs, query_specs = side_specs
    elif p_query is None:
        raise TypeError(
            "swept_similarity: give a query, as a second pre-MAET or as "
            "p_query in the raw positional form.")
    else:
        p_context, w_context = _parts_per_event(p_context, w_context)
        p_query, w_query = _parts_per_event(p_query, w_query)
    _check_is_exch_vs_specs(is_exch, specs)
    plan = _build_plan(p_context, p_query, specs, is_rel, sweep, start, stop,
                       step, align, window, drop, query_ref,
                       locate, "swept_similarity", sigma, is_per, period, r)
    out = _run_similarity(p_context, w_context, p_query, w_query, sigma, r,
                          is_rel, is_per, period, is_exch, specs,
                          query_specs, plan, locate, target_attr, normalize)
    extra = []
    if return_offsets:
        extra.append(_offsets(plan))
    if return_sweep_values:
        extra.append(_sweep_values(plan))
    return (out, *extra) if extra else out


def _sweep_values(plan):
    """The sweep values of each swept attribute, as a dict ``{a: values}``;
    under ``'independent'`` the pair ``(window_values, query_values)``."""
    dims = plan[0]
    out = {}
    for _, a, vals, pair in dims:
        v = np.asarray(vals, dtype=float)
        out[a] = (out[a], v) if pair is not None else v
    return out


def _offsets(plan):
    """The translation applied to the query at each of its sweep values,
    mu = s - query_ref, per translated attribute."""
    dims, _, ctx_win, q_ref = plan
    offs = {}
    for kind, a, vals, _ in dims:
        if a in q_ref and (kind == "query" or ctx_win.get(a, (None, False))[1]):
            offs[a] = np.asarray(vals, dtype=float) - q_ref[a]
    return offs


@_with_dispatch_scope
def swept_entropy(p_context, w_context=None, sigma=None, r=None,
                  is_rel=None, is_per=None, period=None, *,
                  sweep=None, start=None, stop=None, step=None,
                  window=None, drop=None, locate="centroid",
                  target_attr=None, method="differential", base=2.0,
                  n_points_per_dim=None, x_min=float("nan"),
                  x_max=float("nan"), grid_limit=None,
                  is_exch=None, rel=None, exch=None, specs=None,
                  return_sweep_values=False, verbose=False):
    r"""Align a window on a context at each of a list of sweep values and
    take the entropy of the windowed density at each.

    **Overview.** At each of a list of values :math:`s` on an attribute,
    the sweep values, a window :math:`h(\delta)` on the context is aligned
    with its reference value, :math:`\delta = 0`, at :math:`s`. It weights
    each event :math:`n` on the target attribute (event weighting, as by
    :func:`weight_events`):

    .. math:: w'(n) = w(n)\, h\bigl(p_a(n) - s\bigr),

    where :math:`p_a(n)` is event :math:`n`'s value on the swept attribute
    :math:`a`. The
    windowed density is then built and its entropy taken, tracing how the
    entropy changes across the context. Windows, ``locate``, and generated
    sweep values are as at :func:`swept_similarity`; there is no query,
    so the sweep values always align the window.

    **Input forms**: ``swept_entropy(pm, ...)`` with a whole pre-MAET,
    whose specs give the geometry (any of ``sigma``, ``is_per``,
    ``period``, ``r``, ``rel``, ``exch`` may be given alongside to override
    it); or the raw positional form ``swept_entropy(p_context,
    w_context, sigma, r, is_rel, is_per, period, ...)``.

    Parameters
    ----------
    sweep : dict, int, or list of int
        ``{a: values}``: the sweep values of attribute ``a``; a bare
        attribute index, or a list of them, asks for the defaults below.
    start, stop, step : dict or float, optional
        ``{a: value}``: generate attribute ``a``'s sweep values in place of
        listing them; a bare number applies to the swept attribute where
        ``sweep`` names one. ``start`` and ``stop`` default to the lowest and
        highest of the context's values on the attribute; ``step`` defaults
        to half the window's standard deviation, and a pure rectangle
        without a given ``step`` takes its pieces (as at
        :func:`swept_similarity`).
    window : dict
        ``{a: (shape, width)}``, ``{a: (shape, width, edges)}``,
        ``{a: {'shape': ..., 'width' | 'sd' | 'decay_rate': ...,
        'edges': ...}}``, or ``{a: f}``: the window on each swept
        attribute, any profile aligned at the sweep value, as at
        :func:`swept_similarity`; ``edges`` is ``'halfOpen'`` (the
        default) or ``'closed'`` (rectangles only). Required for every
        swept attribute: its scale is that of the local region, which
        nothing in the data can supply.
    drop : int or list of int, optional
        Swept attributes marginalized after the window has weighted the
        events. An attribute kept stays in the density whose entropy is
        taken.
    locate : str, callable, or dict, default ``'centroid'``
        As at :func:`swept_similarity`.
    target_attr : int, optional
        The attribute whose weights the window multiplies (default: the
        first attribute not dropped).
    method, base
        As at :func:`entropy_maet`.
    n_points_per_dim, x_min, x_max, grid_limit
        The grid of the discrete methods (``'shannon'``, ``'normalized'``),
        passed to :func:`entropy_maet` at every sweep value, so every
        window's entropy is taken on the same grid; ``n_points_per_dim`` is
        required for those methods, and ``x_min`` / ``x_max`` for a
        non-periodic attribute that is kept. The continuous methods ignore
        them.
    is_exch, specs, verbose
        As at :func:`swept_similarity`.
    return_sweep_values : bool, default False
        Also return the sweep values, as a dict ``{a: values}``, one array
        per swept attribute: the values listed, or those generated from
        the defaults and ``start`` / ``stop`` / ``step``. Plot the profile
        against them.

    Returns
    -------
    np.ndarray
        One dimension per swept attribute, in attribute order.
    sweep_values : dict
        Only with ``return_sweep_values=True``: ``{a: values}``.
    """
    if is_pre_maet(p_context):
        if w_context is not None:
            raise TypeError(
                "swept_entropy(pm, ...): the pre-MAET form takes no second "
                "positional argument; give the sweep values as "
                "sweep={a: values}.")
        (p_attrs, w_attrs, sigma, r, is_rel, is_per, period, is_exch,
         side_specs) = _swept_pre_maet_args(
            [p_context],
            {"sigma": sigma, "is_per": is_per, "period": period, "r": r,
             "rel": rel if rel is not None else is_rel, "exch": exch},
            "swept_entropy")
        p_context, = p_attrs
        w_context, = w_attrs
        specs, = side_specs
    if not is_pre_maet(p_context):
        p_context, w_context = _parts_per_event(p_context, w_context)
    _check_is_exch_vs_specs(is_exch, specs)
    plan = _build_plan(p_context, None, specs, is_rel, sweep, start, stop,
                       step, None, window, drop, None, locate,
                       "swept_entropy", sigma, is_per, period)
    H = _run_entropy(p_context, w_context, sigma, r, is_rel, is_per,
                     period, is_exch, specs, plan, locate, target_attr,
                     method, base,
                     grid={"n_points_per_dim": n_points_per_dim,
                           "x_min": x_min, "x_max": x_max,
                           "grid_limit": grid_limit})
    return (H, _sweep_values(plan)) if return_sweep_values else H


def swept_mass(p_context, w_context=None, sigma=None, r=None,
               is_rel=None, is_per=None, period=None, *,
               sweep=None, start=None, stop=None, step=None,
               window=None, drop=None, locate="centroid",
               target_attr=None, region=None, normalize="none",
               is_exch=None, rel=None, exch=None, specs=None,
               return_sweep_values=False, verbose=False):
    r"""Align a window on a context at each of a list of sweep values and
    take the mass of the windowed density in a region at each.

    **Overview.** At each sweep value :math:`s`, a window on the context
    is aligned at :math:`s` and weights each event on the target
    attribute, :math:`w'(n) = w(n)\, h(p_a(n) - s)`, as at
    :func:`swept_entropy`. The windowed density is then built and its
    mass in ``region`` taken by :func:`mass_maet`: how much of the local
    material lies in the region, or, with ``normalize='total'``, what
    share of it does. The window weights events before the density is
    built; the region is read from the density, so a tuple just outside
    it still contributes the part of its kernel that crosses the edge,
    and a region can select tuples (the intervals of a relative
    attribute, say) where a window can only weight events.

    **Input forms**: ``swept_mass(pm, ...)`` with a whole pre-MAET, or
    the raw positional form ``swept_mass(p_context, w_context, sigma, r,
    is_rel, is_per, period, ...)``, as at :func:`swept_entropy`.

    Parameters
    ----------
    sweep, start, stop, step, window, drop, locate, target_attr
        As at :func:`swept_entropy`. A dropped attribute is marginalized
        before the mass is taken, so it cannot be restricted.
    region : dict, optional
        ``{a: spec}``, keyed by the context's attribute indices, as at
        :func:`mass_maet`. Without it, the mass of the whole windowed
        density: the window's weighted tuple count.
    normalize : {'none', 'total'}, default 'none'
        As at :func:`mass_maet`: the mass, or its share of the windowed
        density's mass.
    is_exch, specs, verbose
        As at :func:`swept_similarity`.
    return_sweep_values : bool, default False
        Also return the sweep values, as a dict ``{a: values}``, one array
        per swept attribute: the values listed, or those generated from
        the defaults and ``start`` / ``stop`` / ``step``. Plot the profile
        against them.

    Returns
    -------
    np.ndarray
        One dimension per swept attribute, in attribute order.
    sweep_values : dict
        Only with ``return_sweep_values=True``: ``{a: values}``.

    See Also
    --------
    mass_maet, swept_entropy, weight_events
    """
    if is_pre_maet(p_context):
        if w_context is not None:
            raise TypeError(
                "swept_mass(pm, ...): the pre-MAET form takes no second "
                "positional argument; give the sweep values as "
                "sweep={a: values}.")
        (p_attrs, w_attrs, sigma, r, is_rel, is_per, period, is_exch,
         side_specs) = _swept_pre_maet_args(
            [p_context],
            {"sigma": sigma, "is_per": is_per, "period": period, "r": r,
             "rel": rel if rel is not None else is_rel, "exch": exch},
            "swept_mass")
        p_context, = p_attrs
        w_context, = w_attrs
        specs, = side_specs
    if not is_pre_maet(p_context):
        p_context, w_context = _parts_per_event(p_context, w_context)
    _check_is_exch_vs_specs(is_exch, specs)
    plan = _build_plan(p_context, None, specs, is_rel, sweep, start, stop,
                       step, None, window, drop, None, locate,
                       "swept_mass", sigma, is_per, period)
    M = _run_mass(p_context, w_context, sigma, r, is_rel, is_per,
                  period, is_exch, specs, plan, locate, target_attr,
                  region, normalize)
    return (M, _sweep_values(plan)) if return_sweep_values else M


def _swept_pre_maet_args(pms, kw, func):
    """Resolve a pre-MAET call of the swept functions.

    ``swept_similarity`` and ``swept_entropy`` take their geometry
    positionally, as vectors shared by both operands. Given whole
    pre-MAETs instead, this reads the shared geometry out of their specs,
    applies any of the six per-attribute overrides passed as keywords,
    and returns the parts the workers already speak.

    The two pre-MAETs of ``swept_similarity`` describe one comparison,
    so they must agree on the structural geometry: same attribute count,
    and the same ``r``, ``rel``, ``exch``, and nesting on every attribute.
    The context supplies the specs; a disagreement is an error rather
    than a silent choice between them.

    Parameters
    ----------
    pms : list of Mapping
        The pre-MAET operands, context first.
    kw : dict
        The six overrides, keyed ``sigma``, ``is_per``, ``period``,
        ``r``, ``rel``, ``exch``; ``None`` where not given.
    func : str
        The caller's name, for error messages.

    Returns
    -------
    tuple
        ``(p_attrs, w_attrs, sigma, r, is_rel, is_per, period, is_exch,
        specs)``, with ``specs`` ``None`` unless the geometry is nested.
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
            [sp if nested else None for sp in side_specs])


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
