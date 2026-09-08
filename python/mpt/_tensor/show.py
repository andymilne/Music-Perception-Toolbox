"""show.py -- render a pre-MAET as a table.

The layout follows the pre-MAET tables of the worked examples in Milne
(2026): a leading ``attribute`` column carrying each attribute's name and
parameters, one column per event, brace-delimited cells for unordered
attributes and parenthesis-delimited cells for ordered ones, nested
attributes shown as brackets within brackets, and weights, when they are
not uniform, as parenthesized superscripts on their values.

Two renderings are produced from the same model: a plain-ASCII markdown
table for a terminal, and a LaTeX ``booktabs`` tabular for the
manuscript. The public
entry point is :func:`~mpt.show_pre_maet`; the MATLAB twin is
``showPreMaet``.
"""
from __future__ import annotations

import numpy as np

from .premaet import shift_lead

# Text glyphs. The LaTeX rendering uses the corresponding macros.
# The markdown rendering is plain ASCII, so that its column widths are the
# same in Python and in MATLAB (Octave counts a multi-byte glyph as several
# characters) and it survives any terminal encoding. The LaTeX rendering
# carries the article's own symbols.
_ELLIPSIS_H = "..."
_ELLIPSIS_C = "..."
_SIGMA = "sigma"


# -------------------------------------------------------------------
#  Input normalisation
# -------------------------------------------------------------------

def _unpack(p_attr, w, specs, sigma, is_rel, is_per, period, names):
    """Return ``(P, W, specs, params)`` from either input form.

    Accepts a built density, from which every field is recovered, or the
    raw pre-MAET triple with the kernel parameters supplied alongside.
    """
    if hasattr(p_attr, "n_attrs") and hasattr(p_attr, "p_attr"):
        d = p_attr
        A = int(d.n_attrs)
        P = [np.atleast_2d(np.asarray(M, dtype=float)) for M in d.p_attr]
        W = ([np.atleast_2d(np.asarray(M, dtype=float)) for M in d.w]
             if d.w is not None else None)
        nested = getattr(d, "nested", [None] * A)
        sp = []
        for a in range(A):
            if nested[a] is not None:
                sp.append(dict(nested[a]))
            else:
                sp.append({"r": int(np.atleast_1d(d.r)[a]),
                           "rel": bool(np.atleast_1d(d.is_rel)[a]),
                           "sym": bool(np.atleast_1d(d.is_sym)[a])})
        params = {
            "sigma": [float(v) for v in np.atleast_1d(d.sigma)],
            "is_per": [bool(v) for v in np.atleast_1d(d.is_per)],
            "period": [float(v) for v in np.atleast_1d(d.period)],
        }
        return P, W, sp, params, _names(names, sp, A)

    if not isinstance(p_attr, (list, tuple)):
        p_attr = [p_attr]
    P = [np.atleast_2d(np.asarray(M, dtype=float)) for M in p_attr]
    A = len(P)
    if w is None:
        W = None
    else:
        if not isinstance(w, (list, tuple)):
            w = [w]
        W = [np.atleast_2d(np.asarray(M, dtype=float)) for M in w]
        if len(W) != A:
            raise ValueError(
                f"w has {len(W)} attributes but p_attr has {A}.")
    if specs is None:
        sp = [{"r": 1, "rel": False, "sym": True} for _ in range(A)]
    else:
        sp = [dict(s) for s in specs]
        if len(sp) != A:
            raise ValueError(
                f"specs has {len(sp)} attributes but p_attr has {A}.")
    # A spec may carry the attribute's kernel geometry (Milne 2026,
    # Def. 2.6); an explicit argument overrides it, as at build time.
    params = {}
    for key, given, cast in (("sigma", sigma, float),
                             ("is_per", is_per, bool),
                             ("period", period, float)):
        vals = _bcast(given, A, key, cast)
        for a in range(A):
            if vals[a] is None:
                alias = "isPer" if key == "is_per" else key
                vals[a] = sp[a].get(key, sp[a].get(alias))
        params[key] = vals
    if is_rel is not None:
        for a, v in enumerate(_bcast(is_rel, A, "is_rel", bool)):
            sp[a]["rel"] = v
    return P, W, sp, params, _names(names, sp, A)


def _bcast(v, A, what, cast):
    """Per-attribute values, tolerating a non-scalar entry.

    An attribute carrying a kernel covariance (Sec. maet-cov) has a
    matrix where the others have a width, so the sequence cannot be
    made into one numeric array; entries are taken as given and cast
    only where they are scalar.
    """
    if v is None:
        return [None] * A

    def one(x):
        arr = np.asarray(x, dtype=float)
        return cast(arr.reshape(-1)[0]) if arr.ndim == 0 or arr.size == 1 \
            else arr

    if isinstance(v, (list, tuple)):
        if len(v) == 1:
            return [one(v[0])] * A
        if len(v) != A:
            raise ValueError(f"{what} must be a scalar or length {A}.")
        return [one(x) for x in v]

    arr = np.asarray(v, dtype=float)
    if arr.ndim == 0 or arr.size == 1:
        return [cast(arr.reshape(-1)[0])] * A
    if arr.shape[0] != A:
        raise ValueError(f"{what} must be a scalar or length {A}.")
    return [one(x) for x in arr]


def _names(names, specs, A):
    if names is None:
        out = [s.get("name") for s in specs]
    elif isinstance(names, str):
        out = [names] * A
    else:
        out = list(names)
        if len(out) != A:
            raise ValueError(f"names must be a string or length {A}.")
    return [f"a_{a + 1}" if nm is None else str(nm)
            for a, nm in enumerate(out)]


def _levels(spec, key, default):
    """Per-level values of a spec field, innermost first."""
    v = spec.get(key, default)
    if np.isscalar(v) or isinstance(v, (bool, np.bool_)):
        return [v]
    return list(np.atleast_1d(v).ravel())


# -------------------------------------------------------------------
#  Number formatting
# -------------------------------------------------------------------

def _num(x, decimals):
    """Format a value, dropping a trailing integral zero."""
    if not np.isfinite(x):
        return "-" if np.isnan(x) else ("\\infty" if x > 0 else "-\\infty")
    s = f"{x:.{decimals}f}".rstrip("0").rstrip(".")
    if s in ("", "-", "-0", "0"):
        # A value that rounds to all zeros is shown in scientific notation
        # rather than as an exact zero, since a table is read for which
        # entries vanish and a rounded-away tail is not one of them. Below
        # the dirt floor, six decades finer than the requested precision,
        # the value is floating-point residue and is shown as the zero it
        # is meant to be.
        if abs(x) < 0.5 * 10.0 ** -(decimals + 6):
            return "0"
        return f"{x:.{max(decimals - 2, 1)}e}"
    return s


def _param_num(x):
    """A parameter as it appears in the attribute row.

    A kernel covariance is named by its shape rather than printed: the
    row states the density's settings, and a matrix belongs in the text
    that discusses it.
    """
    if x is None:
        return None
    arr = np.asarray(x, dtype=float)
    if arr.ndim >= 2 and arr.size > 1:
        return f"{arr.shape[0]}x{arr.shape[1]} covariance"
    v = float(arr.reshape(-1)[0])
    # NA is shown as such rather than as a number or an omission: a
    # preprocessing step could not carry this parameter forward, and the
    # table is where the user should see that before the build refuses it.
    return "NA" if np.isnan(v) else f"{v:.6g}"


# -------------------------------------------------------------------
#  Cell model: one attribute at one event, as a bracket tree
# -------------------------------------------------------------------

def _cell_tree(p_col, w_col, spec, max_elements):
    """Nested list of leaf strings-to-be, honouring the tag hierarchy.

    Returns ``(node, sym_levels)`` where ``node`` is either a list of
    ``(value, weight)`` leaves (flat attribute) or a nested list of such
    lists (one per group, outermost grouping first), and ``sym_levels``
    are the per-level symmetry flags, innermost first.
    """
    finite = np.isfinite(p_col)
    sym_levels = _levels(spec, "sym", True)
    tags = spec.get("tags")

    if tags is None:
        leaves = [(float(p_col[k]),
                   None if w_col is None else float(w_col[k]))
                  for k in np.flatnonzero(finite)]
        return _elide(leaves, max_elements), sym_levels

    T = np.asarray(tags)
    # A single grouping column arrives one-dimensional; it is a column of
    # the tag matrix, not a row of it.
    T = T.reshape(-1, 1) if T.ndim == 1 else T
    if T.shape[0] != p_col.shape[0]:
        raise ValueError(
            f"spec tags have {T.shape[0]} rows but the attribute carries "
            f"{p_col.shape[0]}.")

    def build(rows, col):
        if col < 0:
            leaves = [(float(p_col[k]),
                       None if w_col is None else float(w_col[k]))
                      for k in rows if finite[k]]
            return _elide(leaves, max_elements)
        groups = []
        for g in _stable_unique(T[rows, col]):
            sub = [k for k in rows if T[k, col] == g]
            node = build(sub, col - 1)
            if node:
                groups.append(node)
        return groups

    node = build(list(range(T.shape[0])), T.shape[1] - 1)
    return node, sym_levels


def _stable_unique(v):
    seen, out = set(), []
    for x in v:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _elide(leaves, max_elements):
    if max_elements is None or len(leaves) <= max_elements:
        return leaves
    keep = max_elements - 1
    return leaves[:keep] + ["..."]


def _render_node(node, sym_levels, decimals, show_w, latex, top=True):
    """Bracket a cell tree.

    ``sym_levels`` runs innermost first, so a node whose bracket depth is
    ``d`` takes ``sym_levels[d]``: the leaves take the innermost flag and
    the outermost bracket the outermost flag. A flat attribute holding a
    single element is written bare, as the article writes it; an inner
    level of a nest keeps its brackets, so that the level stays visible.
    """
    depth = _depth(node)
    sym = (sym_levels[depth] if depth < len(sym_levels)
           else (sym_levels[-1] if sym_levels else True))
    if depth == 0:
        return _render_leaves(node, decimals, show_w, latex, sym, top)
    sep = ", "
    parts = [_render_node(child, sym_levels, decimals, show_w, latex,
                          top=False)
             for child in node]
    return _bracket(sep.join(parts), sym, latex)


def _depth(node):
    if not node or not isinstance(node[0], list):
        return 0
    return 1 + _depth(node[0])


def _render_leaves(leaves, decimals, show_w, latex, sym, top):
    items = []
    for leaf in leaves:
        if leaf == "...":
            items.append("\\dots" if latex else _ELLIPSIS_C)
            continue
        v, wt = leaf
        s = _num(v, decimals)
        if show_w and wt is not None:
            ws = _num(wt, decimals)
            s = f"{s}^{{({ws})}}" if latex else f"{s}^({ws})"
        items.append(s)
    if top and len(items) == 1:
        return items[0]
    return _bracket(", ".join(items), sym, latex)


def _bracket(inner, sym, latex):
    if sym:
        return ("\\{" + inner + "\\}") if latex else "{" + inner + "}"
    return "(" + inner + ")"


# -------------------------------------------------------------------
#  Attribute stub: name and parameters
# -------------------------------------------------------------------

def _stub_lines(name, spec, params, a, latex):
    """``(name line, parameter line)`` for one attribute."""
    r = _levels(spec, "r", 1)
    rel = [bool(v) for v in _levels(spec, "rel", False)]
    sigma = params["sigma"][a]
    is_per = params["is_per"][a]
    period = params["period"][a]

    def tup(vals, cast=str):
        if len(vals) == 1:
            return cast(vals[0])
        return "(" + ", ".join(cast(v) for v in vals) + ")"

    sig = "\\sigma" if latex else _SIGMA
    rel_t = "{[\\mathrm{rel}]}" if latex else "[rel]"
    per_t = "{[\\mathrm{per}]}" if latex else "[per]"

    bits = []
    if sigma is not None:
        bits.append(f"{sig} = {_param_num(sigma)}")
    bits.append("r = " + tup(r, lambda v: str(int(v))))

    rel_s = tup([int(v) for v in rel], str)
    per_v = 0 if is_per is None else int(bool(is_per))
    # The manuscript collapses the two flags when both are the scalar 0.
    if len(rel) == 1 and int(rel[0]) == 0 and per_v == 0:
        bits.append(f"{rel_t}, {per_t} = 0")
    else:
        bits.append(f"{rel_t} = {rel_s}")
        bits.append(f"{per_t} = {per_v}")
        if per_v and period:
            bits.append(f"P = {_param_num(period)}")
    sep = ", "
    return name, sep.join(bits)


# -------------------------------------------------------------------
#  Column selection
# -------------------------------------------------------------------

def _event_columns(N, max_events):
    """Indices to display, with ``None`` marking an elision column."""
    if max_events is None or N <= max_events:
        return list(range(N))
    head = max_events - 2
    return list(range(head)) + [None, N - 1]


# -------------------------------------------------------------------
#  Renderers
# -------------------------------------------------------------------

def _render_markdown(names, stubs, cells, cols, N, title):
    """A GitHub-flavoured markdown table, padded to align in a terminal."""
    n_col = len(cols)
    headers = [(_ELLIPSIS_H if c is None else f"n = {c + 1}") for c in cols]
    stub_col = [f"{names[a]}: {stubs[a][1]}" for a in range(len(names))]

    left_w = max([len(s) for s in stub_col] + [len("attribute")])
    col_w = [max([len(headers[j])]
                 + [len(cells[a][j]) for a in range(len(names))])
             for j in range(n_col)]

    def centre(s, width):
        # Explicit, rather than str.center, so that the padding rule is
        # the one the MATLAB twin applies: any odd character goes right.
        pad = width - len(s)
        if pad <= 0:
            return s
        lhs = pad // 2
        return " " * lhs + s + " " * (pad - lhs)

    def row(left, fields):
        cs = " | ".join(centre(f, col_w[j]) for j, f in enumerate(fields))
        return f"| {left:<{left_w}} | {cs} |"

    out = []
    if title:
        out.extend([title, ""])
    out.append(row("attribute", headers))
    out.append("|:" + "-" * left_w + "-|"
               + "|".join(":" + "-" * w + ":" for w in col_w) + "|")
    for a in range(len(names)):
        out.append(row(stub_col[a], cells[a]))
    return "\n".join(out)


def _render_latex(names, stubs, cells, cols, N, caption, label):
    n_col = len(cols)
    headers = [("$\\cdots$" if c is None else
                (f"$n = {c + 1}$" if c != N - 1 or N <= n_col
                 else f"$n = {N}$"))
               for c in cols]

    out = ["\\begin{table}[]", "\\centering", "\\footnotesize"]
    if caption:
        out.append("\\caption{" + caption + "}")
    if label:
        out.append("\\label{" + label + "}")
    out.append("\\smallskip")
    out.append("\\begin{tabular}{@{}" + "c" * (n_col + 1) + "@{}}")
    out.append("\\toprule")
    out.append("attribute & " + " & ".join(headers) + " \\\\")
    out.append("\\midrule")
    for a in range(len(names)):
        stub = ("$\\begin{array}{@{}c@{}} \\text{" + names[a] + "} \\\\ "
                + stubs[a][1] + " \\end{array}$")
        row = [c if c.startswith("$") else f"${c}$" for c in cells[a]]
        out.append(stub + " & " + " & ".join(row) + " \\\\")
        if a != len(names) - 1:
            out.append("\\midrule")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table}")
    return "\n".join(out)


# -------------------------------------------------------------------
#  Public entry point
# -------------------------------------------------------------------

def show_pre_maet(p_attr, w_attr=None, specs=None, *, sigma=None,
                  is_rel=None,
                  is_per=None, period=None, names=None, format="markdown",
                  max_events=8, max_elements=8, decimals=4,
                  weights="auto", title=None, caption=None, label=None,
                  headings=None, delimiter=",", verbose=True):
    """Print a pre-MAET as a table, in the layout of Milne (2026).

    The table carries one row per attribute and one column per event.
    An attribute's row is headed by its name and the parameters that
    determine its density --- ``sigma``, the tuple size ``r``, the
    ``[rel]`` and ``[per]`` flags, and the period where it is periodic
    --- and its cells hold the elements from which the admitted tuples
    are formed. A cell is brace-delimited where the attribute is
    unordered (``[sym] = 1``) and parenthesis-delimited where it is
    ordered; a nested attribute is bracketed level by level, the
    outermost level outermost. A single element is written bare. Where
    the weights are not uniform they are written as parenthesized
    superscripts on their values, ``60^(0.6)``.

    Two inputs are accepted, as elsewhere in the toolbox: a density
    built by :func:`~mpt.build_exp_tens`, from which every field is
    recovered, or the raw pre-MAET triple with the kernel parameters
    supplied alongside.

    Parameters
    ----------
    p_attr : MaetDensity or list of array-like
        A built density, or the length-A list of per-attribute
        ``(K_a, N)`` value matrices. NaN entries are absent elements.
    w : list of array-like, optional
        Per-attribute weight matrices, matching ``p_attr``. None is
        uniform.
    specs : list of dict, optional
        Per-attribute specs, ``{r, rel, sym}`` flat or carrying ``tags``
        when nested, as produced by :func:`~mpt.flat_specs` and the
        pre-MAET operators. Defaults to flat, ``r = 1``, unordered.
    sigma, is_rel, is_per, period : scalar or length-A, optional
        Kernel parameters, shown in the attribute row. Ignored when a
        density is passed. ``is_rel`` overrides the specs' ``rel``.
    names : str or length-A, optional
        Attribute names. Defaults to the specs' ``name`` entries, then
        to ``a_1``, ``a_2``, and so on.
    format : {'markdown', 'latex', 'csv'}, default 'markdown'
        A markdown table padded to align in a terminal, a LaTeX
        ``booktabs`` tabular, or the CSV that :func:`~mpt.read_pre_maet`
        reads. CSV elides nothing, whatever ``max_events`` says.
    max_events : int or None, default 8
        Columns to display before eliding the middle ones. None shows
        every event.
    max_elements : int or None, default 8
        Elements to display within one cell before eliding the rest.
    decimals : int, default 4
        Decimal places for values and weights; trailing zeros are
        dropped.
    weights : {'auto', True, False}, default 'auto'
        Whether to show the weights. 'auto' shows them where they are
        supplied and not uniform.
    title : str, optional
        A line printed above the markdown table. Ignored for LaTeX.
    caption, label : str, optional
        LaTeX caption and label. Ignored for the other formats.
    headings, delimiter : optional
        CSV only: the event-column headings (default ``n = 1``, ...) and
        the field separator.
    verbose : bool, default True
        Print the table. The string is returned either way.

    Returns
    -------
    str
        The rendered table.

    Examples
    --------
    >>> import numpy as np, mpt
    >>> p = [np.array([[67., 66., 64.]]), np.array([[5., 6., 7.]])]
    >>> _ = mpt.show_pre_maet(p, names=['pitch', 'time'], sigma=[0.5, 0.25],
    ...                       is_per=[True, False], period=[12., 0.],
    ...                       verbose=False)

    See also
    --------
    build_exp_tens, flat_specs, difference_events, bind_events
    """
    p_attr, w_attr, _, specs = shift_lead(
        p_attr, w_attr, [], specs, func="show_pre_maet")
    w = w_attr
    if format not in ("markdown", "latex", "csv"):
        raise ValueError(
            f"format must be 'markdown', 'latex' or 'csv'; got {format!r}.")
    latex = (format == "latex")
    if format == "csv":
        # A file is not a display: it carries the whole pre-MAET, so
        # neither events nor elements are elided.
        max_events = max_elements = None

    P, W, sp, params, nm = _unpack(
        p_attr, w, specs, sigma, is_rel, is_per, period, names)
    A = len(P)
    if A == 0:
        raise ValueError("a pre-MAET must carry at least one attribute.")
    N = P[0].shape[1]
    for a in range(A):
        if P[a].shape[1] != N:
            raise ValueError(
                f"attribute {a} has {P[a].shape[1]} events but attribute 0 "
                f"has {N}; every attribute must span the same passage.")

    if weights == "auto":
        show_w = W is not None and not all(
            _uniform(W[a], P[a]) for a in range(A))
    else:
        show_w = bool(weights) and W is not None

    cols = _event_columns(N, max_events)
    gap = ("$\\cdots$" if latex else _ELLIPSIS_H)

    cells = []
    for a in range(A):
        row = []
        for c in cols:
            if c is None:
                row.append(gap)
                continue
            node, sym_levels = _cell_tree(
                P[a][:, c], None if W is None else W[a][:, c],
                sp[a], max_elements)
            row.append(_render_node(
                node, sym_levels, decimals, show_w, latex))
        cells.append(row)

    stubs = [_stub_lines(nm[a], sp[a], params, a, latex) for a in range(A)]

    if format == "csv":
        from .premaet_io import _render_csv
        out = _render_csv(nm, sp, params, P, W, cols, decimals,
                          headings, delimiter)
    elif latex:
        out = _render_latex(nm, stubs, cells, cols, N, caption, label)
    else:
        out = _render_markdown(nm, stubs, cells, cols, N, title)
    if verbose:
        print(out)
    return out


def _uniform(W_a, P_a):
    live = np.isfinite(P_a) & np.isfinite(W_a)
    if not live.any():
        return True
    v = W_a[live]
    return bool(np.allclose(v, v.flat[0]))
