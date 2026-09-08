"""Elementwise transforms of attribute values (pre-MAET).

:func:`transform_attributes` maps every value of selected attributes
through a named transform, a scale conversion, or a user function. It
sits with :func:`difference_events`, :func:`bind_events`,
:func:`translate_attributes`, and :func:`weight_events` in the
preprocessing layer before :func:`build_exp_tens`, and it absorbs the
pitch and frequency scale conversions that ``convert_pitch`` used to
provide on bare arrays.

See USER_GUIDE §3.1 ("Cross-event preprocessing") for the conceptual
introduction, including the order in which a transform composes with
differencing.
"""
from __future__ import annotations

import math

import numpy as np

from .premaet import pre_maet, shift_lead
from .preprocessing import flat_specs

__all__ = ["transform_attributes", "TRANSFORM_NAMES", "PITCH_SCALES"]


# ===================================================================
#  Scale conversions (routed through Hz)
# ===================================================================

# Every pitch scale converts to and from Hz. The 'octave' scale
# is MIDI / 12 (octaves above MIDI 0 = 8.1758 Hz), so that it shares the
# origin of 'midi' and 'cents' rather than needing a reference.
_PITCH_TO_HZ = {
    "hz":        lambda v: v,
    "midi":      lambda v: 440.0 * 2.0 ** ((v - 69.0) / 12.0),
    "cents":     lambda v: 440.0 * 2.0 ** ((v - 6900.0) / 1200.0),
    "octave":    lambda v: 440.0 * 2.0 ** (v - 69.0 / 12.0),
    "mel":       lambda v: 700.0 * (10.0 ** (v / 2595.0) - 1.0),
    "bark":      lambda v: 1960.0 * (v + 0.53) / (26.28 - v),
    "erb":       lambda v: (10.0 ** (v / 21.4) - 1.0) / 0.00437,
    "greenwood": lambda v: 165.4 * (10.0 ** (2.1 * v) - 0.88),
}
_PITCH_FROM_HZ = {
    "hz":        lambda f: f,
    "midi":      lambda f: 69.0 + 12.0 * np.log2(f / 440.0),
    "cents":     lambda f: 6900.0 + 1200.0 * np.log2(f / 440.0),
    "octave":    lambda f: 69.0 / 12.0 + np.log2(f / 440.0),
    "mel":       lambda f: 2595.0 * np.log10(1.0 + f / 700.0),
    "bark":      lambda f: 26.81 / (1.0 + 1960.0 / f) - 0.53,
    "erb":       lambda f: 21.4 * np.log10(0.00437 * f + 1.0),
    "greenwood": lambda f: np.log10(f / 165.4 + 0.88) / 2.1,
}

PITCH_SCALES = tuple(_PITCH_TO_HZ)


def _convert_scale(values, from_scale, to_scale):
    """Convert a float array between two pitch scales (through Hz)."""
    src, tgt = from_scale.lower(), to_scale.lower()
    for s in (src, tgt):
        if s not in PITCH_SCALES:
            raise ValueError(
                f"Unknown scale '{s}'. Known scales: "
                f"{', '.join(PITCH_SCALES)}.")
    if src == tgt:
        return values.copy()
    return _PITCH_FROM_HZ[tgt](_PITCH_TO_HZ[src](values))


# ===================================================================
#  Named transforms
# ===================================================================

# name -> (function(x, **params), domain, defaults, required, magnitude)
#   domain     : 'any' | 'nonneg' | 'positive' | 'nonzero'
#   magnitude  : whether ``sign=True`` (transform |x|, append sign) applies
def _log(x, base, offset):
    return np.log(x + offset) / math.log(base)


def _power(x, exponent):
    return np.power(x, exponent)


def _affine(x, scale, offset):
    return scale * x + offset


# 'log' takes log(x + offset): the domain is x + offset > 0, so with the
# default offset = 0 a zero is refused, and admitting zeros means writing
# the constant down (log(x + c) is not unit-free).
_NAMED = {
    "log":    (_log, "positive", {"base": math.e, "offset": 0.0}, (), True),
    "power":  (_power, "nonneg", {}, ("exponent",), True),
    "affine": (_affine, "any", {"scale": 1.0, "offset": 0.0}, (), False),
}
TRANSFORM_NAMES = tuple(_NAMED)

# Scale pairs whose source domain is restricted (values that convert to a
# non-finite result are refused with the same messages as the named
# transforms).
_SCALE_DOMAIN = {
    "hz": "positive",
    "mel": "nonneg", "erb": "nonneg",
}


# ===================================================================
#  Parsing a transform entry
# ===================================================================

class _Transform:
    """A parsed transform: ``kind`` is 'named', 'scale', or 'callable'."""
    __slots__ = ("kind", "name", "func", "params", "domain", "magnitude",
                 "label")

    def __init__(self, kind, name, func, params, domain, magnitude, label):
        self.kind = kind
        self.name = name
        self.func = func
        self.params = params
        self.domain = domain
        self.magnitude = magnitude
        self.label = label

    def apply(self, x):
        if self.kind == "scale":
            return _convert_scale(x, *self.name, **self.params)
        if self.kind == "named":
            return self.func(x, **self.params)
        return self.func(x)


def _parse_transform(entry, a):
    """Coerce one transforms entry to a ``_Transform`` (or None)."""
    if entry is None:
        return None
    where = f"transforms[{a}]"
    # --- callable ---
    if callable(entry) and not isinstance(entry, (str, dict, tuple)):
        return _Transform("callable", getattr(entry, "__name__", "callable"),
                          entry, {}, "any", False, "user function")
    # --- dict form ---
    if isinstance(entry, dict):
        d = {k.lower() if isinstance(k, str) else k: v
             for k, v in entry.items()}
        if "from" in d or "to" in d:
            if not ("from" in d and "to" in d):
                raise ValueError(
                    f"{where}: a scale conversion needs both 'from' and "
                    f"'to'.")
            params = {k: v for k, v in d.items() if k not in ("from", "to")}
            return _scale_transform(d["from"], d["to"], params, where)
        if "name" not in d:
            raise ValueError(
                f"{where}: a dict transform needs 'name' (or 'from'/'to' "
                f"for a scale conversion).")
        params = {k: v for k, v in d.items() if k != "name"}
        return _named_transform(d["name"], params, where)
    # --- string ---
    if isinstance(entry, str):
        return _named_transform(entry, {}, where)
    # --- tuple: (from, to) scale pair, or (name, params dict) ---
    if isinstance(entry, tuple):
        if len(entry) == 2 and isinstance(entry[0], str):
            if isinstance(entry[1], str):
                return _scale_transform(entry[0], entry[1], {}, where)
            if isinstance(entry[1], dict):
                return _named_transform(entry[0], dict(entry[1]), where)
        raise ValueError(
            f"{where}: a tuple transform must be a scale pair "
            f"('from', 'to') or a named transform with parameters "
            f"('name', {{...}}).")
    raise ValueError(
        f"{where}: unrecognised transform {entry!r}. Use a name from "
        f"{TRANSFORM_NAMES}, a scale pair such as ('hz', 'cents'), a dict, "
        f"or a callable; None leaves the attribute unchanged.")


def _named_transform(name, params, where):
    key = name.lower()
    if key not in _NAMED:
        hint = ""
        if key in PITCH_SCALES:
            hint = (f" '{key}' is a scale name; a scale conversion is a "
                    f"pair such as ('{key}', 'cents'), and a per-attribute "
                    f"list of transforms must be a list, not a tuple.")
        raise ValueError(
            f"{where}: unknown transform '{name}'. Known transforms: "
            f"{', '.join(TRANSFORM_NAMES)}; scale conversions are given as "
            f"a pair ('from', 'to').{hint}")
    func, domain, defaults, required, magnitude = _NAMED[key]
    p = {k.lower(): v for k, v in params.items()}
    unknown = set(p) - set(defaults) - set(required)
    if unknown:
        raise ValueError(
            f"{where}: '{key}' does not take parameter(s) "
            f"{sorted(unknown)}; it takes {sorted(set(defaults) | set(required))}.")
    missing = [r for r in required if r not in p]
    if missing:
        raise ValueError(
            f"{where}: '{key}' requires parameter(s) {missing}.")
    full = dict(defaults)
    full.update({k: float(v) for k, v in p.items()})
    if key == "log" and not (full["base"] > 0.0 and full["base"] != 1.0):
        raise ValueError(f"{where}: base must be positive and not 1.")
    return _Transform("named", key, func, full, domain, magnitude,
                      f"'{key}'")


def _scale_transform(src, tgt, params, where):
    src, tgt = src.lower(), tgt.lower()
    if params:
        raise ValueError(
            f"{where}: a scale conversion takes no parameters; got "
            f"{sorted(params)}.")
    # Validate the pair now so that a bad name is reported before any
    # values are touched.
    _convert_scale(np.ones(1), src, tgt)
    domain = _SCALE_DOMAIN.get(src, "any")
    return _Transform("scale", (src, tgt), None, {}, domain, False,
                      f"('{src}', '{tgt}')")


# ===================================================================
#  Domain checks and messages
# ===================================================================

def _attr_label(a, spec):
    name = spec.get("name") if isinstance(spec, dict) else None
    return f"attribute {a}" + (f" ('{name}')" if name else "")


def _offending(mask, limit=6):
    """Format the first few (row, event) indices where ``mask`` holds."""
    idx = np.argwhere(mask)
    parts = [f"(value {r}, event {c})" for r, c in idx[:limit]]
    more = "" if len(idx) <= limit else f", ... ({len(idx)} in all)"
    return ", ".join(parts) + more


def _check_domain(x, src, tr, a, spec, sign_on):
    """Refuse values outside the transform's domain, with remedies.

    ``x`` is the attribute's values and ``src`` the array the transform
    will see (``|x|`` when the sign attribute is requested). For
    ``'log'`` the domain applies to ``src + offset``.
    """
    label = _attr_label(a, spec)
    if tr.domain == "any":
        return
    neg = x < 0.0
    if neg.any() and not sign_on:
        if tr.magnitude:
            remedy = (
                " Pass sign=True for this attribute to transform the "
                "magnitudes |x| and append a sign attribute (-1/2, 0, +1/2) "
                "immediately after it.")
        else:
            remedy = ""
        raise ValueError(
            f"{label}: {tr.label} is undefined for negative values; "
            f"negatives at {_offending(neg)}.{remedy}")
    if tr.domain != "positive":
        return
    shifted = src + tr.params.get("offset", 0.0)
    bad = shifted <= 0.0
    if bad.any():
        if tr.kind == "named" and tr.params.get("offset", 0.0) != 0.0:
            raise ValueError(
                f"{label}: {tr.label} with offset {tr.params['offset']:g} "
                f"is undefined where x + offset <= 0; values at "
                f"{_offending(bad)}.")
        raise ValueError(
            f"{label}: {tr.label} is undefined at zero; zero values at "
            f"{_offending(bad)}. Remedies, in the usual order of "
            f"preference: bind simultaneous events first (bind_events) "
            f"if the zeros are the inter-onset intervals of chords or "
            f"grace notes; drop those events deliberately; or admit them "
            f"with an explicit offset, ('log', {{'offset': c}}) = "
            f"log(x + c), bearing in mind that log(x + c) is not "
            f"unit-free, so the unit of x is then part of the model.")


def _check_finite_output(y, x, tr, a, spec):
    bad = ~np.isfinite(y)
    if bad.any():
        raise ValueError(
            f"{_attr_label(a, spec)}: {tr.label} produced non-finite values "
            f"at {_offending(bad)} (inputs "
            f"{', '.join(f'{v:g}' for v in x[bad][:6])}). The transform "
            f"must return a finite value for every input.")


# ===================================================================
#  transform_attributes
# ===================================================================


# ===================================================================
#  Kernel geometry under a transform
# ===================================================================

#: Scales that are affine images of one another --- all log-frequency,
#: differing only in unit --- so that a conversion among them multiplies a
#: kernel width by a constant. Every other scale (Hz, mel, bark, erb,
#: greenwood) is a non-linear map of these, under which a single width has
#: no image.
_LOG_FREQ_SCALES = {"midi": 1.0, "cents": 100.0, "octave": 1.0 / 12.0}


def _transform_gain(tr):
    """The constant a transform multiplies a kernel width by, or None.

    ``None`` means no width carries across. A width remains perfectly
    meaningful in the new coordinate --- after a log it is a width on the
    log axis, so it expresses a ratio rather than a difference --- but a
    non-linear map has a local scaling that varies with position, so no
    single value is the image of the old one. The caller records NA to
    say there is no canonical choice, not that a width is meaningless.
    """
    if tr is None:
        return 1.0
    if tr.kind == "scale":
        src, tgt = (str(v).lower() for v in tr.name)
        if src in _LOG_FREQ_SCALES and tgt in _LOG_FREQ_SCALES:
            return _LOG_FREQ_SCALES[tgt] / _LOG_FREQ_SCALES[src]
        return None
    if tr.kind == "named":
        if tr.name == "affine":
            return abs(float(tr.params.get("scale", 1.0)))
        if tr.name == "power" and float(tr.params.get("exponent", 0.0)) == 1.0:
            return 1.0
        return None
    return None                     # a user callable is opaque


def _transformed_spec(spec, tr, magnitude):
    """One attribute's spec after a transform, with its kernel geometry.

    An affine map (or a conversion within the log-frequency family)
    carries the width and the period across by the same constant. Anything
    non-linear leaves NA --- not because a width would be meaningless in
    the new coordinate, but because there is no canonical value to carry
    over, the local scaling varying across the attribute's range; the
    analyst supplies the width the new units call for. Taking magnitudes
    for a sign attribute folds the axis, which no width survives either.
    """
    out = dict(spec)
    gain = None if magnitude else _transform_gain(tr)
    for key in ("sigma", "period"):
        val = out.get(key)
        if val is None:
            continue
        if gain is None:
            out[key] = float("nan")
            continue
        arr = np.asarray(val, dtype=float)
        if arr.ndim >= 2:
            # A kernel covariance is in squared units, so an affine map
            # of gain g scales it by g^2 where a width scales by g.
            out[key] = arr * (gain ** 2)
        else:
            out[key] = float(arr) * gain
    return out


def transform_attributes(p_attr, w_attr=None, transforms=None, *,
                         specs=None, sign=False):
    """Map attribute values through named transforms, scale conversions,
    or user functions.

    Per-attribute preprocessing on the ``(p_attr, w, specs)`` triple: every
    value of each selected attribute is passed through the transform given
    for that attribute, and the triple feeds straight into
    :func:`build_exp_tens` or a further pre-MAET step. Weights pass through
    unchanged. The map is elementwise, so it composes with the other
    preprocessors in either order, and the order carries meaning: ``'log'``
    *then* :func:`difference_events` gives log ratios (the natural
    representation of inter-onset-interval ratios, and of intervals from
    frequencies in Hz), whereas :func:`difference_events` *then* a
    compressive transform with ``sign=True`` gives signed compressed
    magnitudes.

    **Bare-array form.** When ``p_attr`` is a numeric array rather than a
    list, it is treated as a single attribute and the transformed array is
    returned alone (``w`` and ``specs`` must be ``None``; ``sign`` must be
    ``False``). This is the one-line conversion that ``convert_pitch``
    used to provide::

        cents = transform_attributes(f_hz, None, ('hz', 'cents'))

    **Transforms.** ``transforms`` is a length-A list (one entry per
    attribute) or a single entry broadcast to every attribute. Each entry
    is one of:

    - ``None`` --- leave the attribute unchanged.
    - a name, optionally with parameters as ``('name', {...})`` or
      ``{'name': ..., param: ...}``:

      ============ ================================ ================ ====
      name         parameters                       domain           sign
      ============ ================================ ================ ====
      ``'log'``    ``base`` (e), ``offset`` (0)     x + offset > 0   yes
      ``'power'``  ``exponent`` (required)          x >= 0           yes
      ``'affine'`` ``scale`` (1), ``offset`` (0)    any              no
      ============ ================================ ================ ====

      ``'log'`` computes log(x + offset) / log(base); with the default
      offset a zero is refused, and admitting zeros means writing the
      constant down (log(x + c) is not unit-free: the unit of x is then
      part of the model). ``'affine'`` is the only transform compatible
      with a periodic attribute (``is_per=True`` at build); the others
      change the metric and so cannot be wrapped.
    - a scale pair ``('from', 'to')`` or ``{'from': ..., 'to': ...}``
      among the pitch scales ``'hz'``, ``'midi'``, ``'cents'`` (100 x
      MIDI), ``'octave'`` (MIDI / 12), ``'mel'``, ``'bark'``, ``'erb'``,
      ``'greenwood'``; every pair routes through Hz.
    - a callable ``f(x) -> y`` applied to the attribute's ``K_total x N``
      value matrix; ``y`` must have the same shape and be finite
      everywhere.

    **Domain.** Values outside a transform's domain are refused with a
    message naming the attribute and events and the remedies. In
    particular a zero under ``'log'`` (a zero inter-onset interval, say)
    is an error rather than ``-inf``: bind simultaneous events first, drop
    them deliberately, or admit them with an explicit ``offset``.

    **Sign.** ``sign`` (bool, or a length-A list of bools) applies a
    magnitude transform (``'log'`` or ``'power'``) to ``|x|`` and inserts a **sign attribute** with
    values in {-1/2, 0, +1/2} immediately after the source attribute. The
    attribute count grows by one for each such attribute, so downstream
    per-attribute arguments (``sigma``, ``r``, ``rel``, ``sym``, ``wrap``,
    ``diff_orders``, ...) must include the new column; this is why the
    insertion is explicit rather than automatic. The sign attribute copies
    the source's spec with ``rel=False`` and the name suffixed ``'_sign'``,
    and its weights (when ``w`` is a per-attribute list) copy the source's.
    A small kernel width on the sign attribute makes it effectively
    categorical.

    Parameters
    ----------
    pm : dict, optional
        The pre-MAET, whole, as :func:`~mpt.pre_maet` builds it,
        passed in place of ``p_attr``, in which case
        ``w_attr`` and ``specs`` come from it and the positional
        arguments below move one place earlier.
    p_attr : list/tuple of array-like, or array-like
        Length-A list of ``K_total x N`` per-attribute value matrices (a
        1-D entry is a ``1 x N`` row), or a bare array (see above).
    w_attr : None, scalar, or length-A list
        Weights; passed through unchanged (a per-attribute list gains a
        copy of the source's entry for each sign attribute).
    transforms : entry or length-A list of entries
        See above.
    specs : None or length-A list, keyword-only
        Attribute specifications; ``None`` synthesises flat specs.
    sign : bool or length-A list of bool, keyword-only
        Append a sign attribute after the given attributes (default
        ``False``).

    Returns
    -------
    dict, or ndarray in the bare-array form
        The pre-MAET. Its ``p_attr`` is the list of transformed
        value matrices, its ``w_attr`` is as input, and its ``specs`` are
        as input, both extended for sign attributes. In the bare-array
        form the transformed array is returned instead.

    See Also
    --------
    difference_events, bind_events, translate_attributes, weight_events,
    build_exp_tens
    """
    p_attr, w_attr, (transforms,), specs = shift_lead(
        p_attr, w_attr, [transforms], specs, func="transform_attributes")
    w = w_attr
    # --- bare-array form (a numeric array, or a list of plain numbers) ---
    if not isinstance(p_attr, (list, tuple)) or (
            len(p_attr) > 0 and all(np.isscalar(v) for v in p_attr)):
        if w is not None or specs is not None:
            raise ValueError(
                "In the bare-array form (p_attr a numeric array) w and "
                "specs must be None.")
        if sign is not False:
            raise ValueError(
                "In the bare-array form sign must be False; use the list "
                "form to append a sign attribute.")
        if isinstance(transforms, list):
            raise ValueError(
                "In the bare-array form transforms must be a single "
                "transform, not a list.")
        x = np.asarray(p_attr, dtype=np.float64)
        bare = transform_attributes([x.reshape(1, -1) if x.ndim <= 1
                                     else x], None, [transforms])
        out = bare["p_attr"]
        return float(out[0][0, 0]) if x.ndim == 0 else out[0].reshape(x.shape)

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
                f"Attribute {a} must be 1-D or 2-D; got ndim={Marr.ndim}.")
        p_arr.append(Marr)
    n_events = p_arr[0].shape[1]
    for a, M in enumerate(p_arr):
        if M.shape[1] != n_events:
            raise ValueError(
                f"All attributes must share the same event count N. "
                f"Attribute 0 has N={n_events}; attribute {a} has "
                f"N={M.shape[1]}.")

    if specs is None:
        specs_in = flat_specs(p_arr)
    else:
        if not isinstance(specs, (list, tuple)) or len(specs) != A:
            raise ValueError(
                f"specs must be a length-A ({A}) list, one per attribute.")
        specs_in = list(specs)

    # --- transforms: list per attribute, or one entry broadcast ---
    if isinstance(transforms, list):
        if len(transforms) != A:
            raise ValueError(
                f"transforms must be a length-A ({A}) list, one entry per "
                f"attribute, or a single transform applied to every "
                f"attribute.")
        entries = list(transforms)
    else:
        entries = [transforms] * A
    parsed = [_parse_transform(e, a) for a, e in enumerate(entries)]

    # --- sign flags ---
    if isinstance(sign, (list, tuple, np.ndarray)):
        if len(sign) != A:
            raise ValueError(
                f"sign must be a bool or a length-A ({A}) list of bools.")
        sign_v = [bool(s) for s in sign]
    else:
        sign_v = [bool(sign)] * A
    for a in range(A):
        if sign_v[a] and (parsed[a] is None or not parsed[a].magnitude):
            what = "no transform" if parsed[a] is None else parsed[a].label
            raise ValueError(
                f"{_attr_label(a, specs_in[a])}: sign=True applies only to "
                f"a magnitude transform ('log' or 'power'); got {what}.")

    # --- weights as a per-attribute list when they are one ---
    w_list = None
    if isinstance(w, (list, tuple)):
        if len(w) != A:
            raise ValueError(
                f"w must be None, a scalar, or a length-A ({A}) list.")
        w_list = list(w)

    p_out, w_out, specs_out = [], [], []
    for a in range(A):
        x = p_arr[a]
        tr = parsed[a]
        if tr is None:
            p_out.append(x.copy())
            specs_out.append(specs_in[a])
            if w_list is not None:
                w_out.append(w_list[a])
            continue
        if not np.isfinite(x).all():
            raise ValueError(
                f"{_attr_label(a, specs_in[a])}: values must be finite; "
                f"non-finite at {_offending(~np.isfinite(x))}.")
        src = np.abs(x) if sign_v[a] else x
        _check_domain(x, src, tr, a, specs_in[a], sign_v[a])
        try:
            y = np.asarray(tr.apply(src), dtype=np.float64)
        except ValueError as err:
            if tr.kind == "callable":
                raise
            raise ValueError(
                f"{_attr_label(a, specs_in[a])}: {tr.label}: {err}") from None
        if y.shape != x.shape:
            raise ValueError(
                f"{_attr_label(a, specs_in[a])}: {tr.label} returned shape "
                f"{y.shape}; expected {x.shape}.")
        _check_finite_output(y, src, tr, a, specs_in[a])
        p_out.append(y)
        specs_out.append(_transformed_spec(specs_in[a], tr, sign_v[a]))
        if w_list is not None:
            w_out.append(w_list[a])
        if sign_v[a]:
            # The two signs are the vertices of the 2-point simplex, at
            # the toolbox's default unit edge length (simplex_vertices(2)
            # is [+1/2, -1/2]), with a zero step at the centroid. Coding
            # them +/-1 would put them at edge length 2 and so on a
            # different scale from every other categorical attribute.
            p_out.append(0.5 * np.sign(x))
            # The sign axis is a fresh three-level categorical: its width
            # is a modelling choice, not an image of the source's.
            sgn = dict(_sign_spec(specs_in[a]))
            for key in ("sigma", "period"):
                if key in sgn:
                    sgn[key] = float("nan")
            specs_out.append(sgn)
            if w_list is not None:
                w_out.append(w_list[a])

    if w_list is None:
        w_out = w
    return pre_maet(p_out, w_out, specs_out)


def _sign_spec(spec):
    """The spec of a sign attribute: the source's structure, ``rel``
    cleared, name suffixed ``'_sign'``."""
    if not isinstance(spec, dict):
        return {"r": 1, "rel": False, "sym": True, "name": "sign"}
    s = dict(spec)
    rel = s.get("rel", False)
    if isinstance(rel, str):
        s["rel"] = False
    elif np.ndim(rel) > 0:
        s["rel"] = [False] * len(rel)
    else:
        s["rel"] = False
    s["name"] = f"{s['name']}_sign" if s.get("name") else "sign"
    return s
