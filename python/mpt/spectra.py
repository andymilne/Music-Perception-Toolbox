"""Spectral enrichment — add partials to weighted pitch multisets."""

from __future__ import annotations

import numpy as np

from ._utils import validate_weights


def add_spectra(
    p,
    w=None,
    mode=None,
    *args,
    attribute=None,
    specs=None,
    units: float = 1200.0,
):
    """Add spectral partials to a weighted pitch multiset.

    Five modes determine the partial positions; for the first four,
    a sub-option selects the weight decay law.

    Two forms. Given a single weighted multiset, ``add_spectra(p, w,
    mode, ...)`` returns the expanded ``(p, w)`` pair, and this is the
    primitive the harmony, entropy and consonance functions call.
    Given a pre-MAET, ``add_spectra(pm, mode, ..., attribute=a)``
    expands attribute ``a`` of every event at once and returns a
    pre-MAET, as the other pre-MAET preprocessors do. The expansion
    multiplies the attribute's ``K`` by the number of partials and
    leaves ``N`` and the spec alone; padded slots stay padded, a
    missing value having no spectrum. Partials of one value differ in
    weight, so the result always carries weights, even where the input
    carried none. The expanded rows come in the order the
    single-multiset form gives them, which each language flattens in its
    own way.

    The bare ``p_attr`` / ``w_attr`` form the other preprocessors
    offer is not available here: its positional layout cannot be told
    apart from the single-multiset form, which is this function's
    alone.

    Parameters
    ----------
    p : array-like
        Pitch values (in the units specified by *units*; default
        cents).
    w : array-like or None
        Weights (``None`` for all ones).
    mode : str
        ``'harmonic'``, ``'stretched'``, ``'freqlinear'``,
        ``'stiff'``, or ``'custom'``.
    *args
        Mode-specific arguments (N, decay type, decay parameter,
        etc.). See the MATLAB ``help addSpectra`` for full details.
    attribute : int or str, optional
        Which attribute of a pre-MAET takes the partials, by position
        or by name. Required in the pre-MAET form and refused in the
        other.
    specs : sequence of dict, optional
        The specs to read in place of the pre-MAET's own.
    units : float
        Cents per unit (default 1200 = one octave per unit of
        log₂ frequency).

    Returns
    -------
    p_out : np.ndarray
        Pitch values including partials.
    w_out : np.ndarray
        Corresponding weights.
    """
    if _is_pre_maet(p):
        # Given a pre-MAET, every positional argument after it sits one
        # slot early; the weights come from the pre-MAET itself.
        if w is None:
            lead, rest = mode, args
        else:
            lead, rest = w, ((mode,) if mode is not None else ()) + args
        return _add_to_pre_maet(p, lead, rest, attribute=attribute,
                                specs=specs, units=units)
    if attribute is not None:
        raise TypeError(
            "add_spectra: 'attribute' names which attribute of a pre-MAET "
            "takes partials, and this call gave a single multiset.")

    p = np.asarray(p, dtype=np.float64).ravel()
    w = validate_weights(w, len(p))
    offsets, spec_w = _partials(mode, args, units)

    # Build output: each pitch gets every partial offset.
    # p is (M,), offsets is (K,) → broadcasting gives (M, K).
    p_matrix = p[:, None] + offsets[None, :]
    w_matrix = w[:, None] * spec_w[None, :]

    return p_matrix.ravel(), w_matrix.ravel()


def _is_pre_maet(obj):
    from ._tensor.premaet import is_pre_maet
    return is_pre_maet(obj)


def _add_to_pre_maet(pm, mode, args, *, attribute, specs, units):
    """Give one attribute of a pre-MAET its partials, at every event."""
    from ._tensor.premaet import pack_pre_maet, unpack_pre_maet
    from ._tensor.preprocessing import _select_indices, flat_specs

    p_attr, w_attr, pm_specs = unpack_pre_maet(pm)
    p_attr = [np.asarray(M, dtype=np.float64) for M in p_attr]
    A = len(p_attr)
    specs_in = list(flat_specs(p_attr) if (specs is None and pm_specs is None)
                    else (pm_specs if specs is None else specs))
    if len(specs_in) != A:
        raise ValueError(
            f"specs must be a length-A ({A}) sequence, one per attribute.")
    if mode is None:
        raise ValueError("add_spectra needs a mode.")
    if attribute is None:
        raise ValueError(
            "add_spectra needs the attribute whose values take partials: a "
            "pre-MAET may carry several and a spectrum belongs to one.")
    names = [s.get("name") for s in specs_in]
    idx = _select_indices([attribute], A, "attribute", names)
    if len(idx) != 1:
        raise ValueError(
            "add_spectra takes one attribute; a spectrum belongs to one set "
            "of values.")
    at = int(idx[0])
    spec = dict(specs_in[at])
    if not spec.get("exch", True) and int(spec.get("r", 1)) > 1:
        raise ValueError(
            f"attribute {names[at]!r} is read in order at r = "
            f"{int(spec['r'])}, so its positions carry meaning that adding "
            "partials would scramble: a tuple would take the first value's "
            "partials rather than one value from each position. Add the "
            "partials before the attributes are bound, or read this one as "
            "a multiset.")

    offsets, spec_w = _partials(mode, args, units)
    values = p_attr[at]
    K, N = values.shape
    weights = (np.ones((K, N)) if w_attr is None
               else np.asarray(w_attr[at], dtype=np.float64))
    P = offsets.size
    grown = (values[:, None, :] + offsets[None, :, None]).reshape(K * P, N)
    grownW = (weights[:, None, :] * spec_w[None, :, None]).reshape(K * P, N)

    p_out, w_out = list(p_attr), []
    p_out[at] = grown
    for a in range(A):
        if a == at:
            w_out.append(grownW)
        elif w_attr is None:
            w_out.append(np.ones(p_attr[a].shape))
        else:
            w_out.append(np.asarray(w_attr[a], dtype=np.float64))
    return pack_pre_maet(p_out, w_out, specs_in)

def _partials(mode, args, units):
    """The partial offsets and their weights, for one mode."""
    mode = mode.lower()

    if mode == "harmonic":
        if len(args) < 3:
            raise ValueError(
                "Usage: add_spectra(p, w, 'harmonic', N, weight_type, param)"
            )
        N = _validate_n(args[0])
        n = np.arange(1, N + 1, dtype=np.float64)
        offsets = units * np.log2(n)
        spec_w = _parse_weights(n, args[1], args[2])

    elif mode == "stretched":
        if len(args) < 4:
            raise ValueError(
                "Usage: add_spectra(p, w, 'stretched', N, beta, weight_type, param)"
            )
        N = _validate_n(args[0])
        beta = float(args[1])
        if beta <= 0:
            raise ValueError("beta must be a positive scalar.")
        n = np.arange(1, N + 1, dtype=np.float64)
        offsets = beta * units * np.log2(n)
        spec_w = _parse_weights(n, args[2], args[3])

    elif mode == "freqlinear":
        if len(args) < 4:
            raise ValueError(
                "Usage: add_spectra(p, w, 'freqlinear', N, alpha, weight_type, param)"
            )
        N = _validate_n(args[0])
        alpha = float(args[1])
        if alpha <= -1:
            raise ValueError("alpha must be greater than -1.")
        n = np.arange(1, N + 1, dtype=np.float64)
        ratios = (alpha + n) / (alpha + 1)
        if np.any(ratios <= 0):
            bad = int(np.argmax(ratios <= 0))
            raise ValueError(
                f"All frequency ratios must be positive. "
                f"With alpha = {alpha}, ratio for n = {bad + 1} is {ratios[bad]:.4g}."
            )
        offsets = units * np.log2(ratios)
        spec_w = _parse_weights(n, args[2], args[3])

    elif mode == "stiff":
        if len(args) < 4:
            raise ValueError(
                "Usage: add_spectra(p, w, 'stiff', N, B, weight_type, param)"
            )
        N = _validate_n(args[0])
        B = float(args[1])
        if B < 0:
            raise ValueError("B must be a non-negative scalar.")
        n = np.arange(1, N + 1, dtype=np.float64)
        ratios = (n * np.sqrt(1 + B * n**2)) / np.sqrt(1 + B)
        offsets = units * np.log2(ratios)
        spec_w = _parse_weights(n, args[2], args[3])

    elif mode == "custom":
        if len(args) < 2:
            raise ValueError(
                "Usage: add_spectra(p, w, 'custom', offsets, spec_w)"
            )
        offsets = np.asarray(args[0], dtype=np.float64).ravel()
        spec_w = np.asarray(args[1], dtype=np.float64).ravel()
        if offsets.size != spec_w.size:
            raise ValueError("offsets and spec_w must have the same length.")
    else:
        raise ValueError(
            f"Unknown mode '{mode}'. Use 'harmonic', 'stretched', "
            "'freqlinear', 'stiff', or 'custom'."
        )

    return offsets, spec_w

# -------------------------------------------------------------------
#  Helpers
# -------------------------------------------------------------------


def _validate_n(N) -> int:
    N = int(N)
    if N < 1:
        raise ValueError("N must be a positive integer.")
    return N


def _parse_weights(
    n: np.ndarray, weight_type: str, param: float
) -> np.ndarray:
    """Compute spectral weights for partial numbers *n*."""
    wt = str(weight_type).lower()
    param = float(param)

    if wt == "powerlaw":
        if param < 0:
            raise ValueError("rho must be a non-negative scalar.")
        return 1.0 / n**param

    if wt == "geometric":
        if not 0 <= param <= 1:
            raise ValueError("tau must be a scalar in [0, 1].")
        return param ** (n - 1)

    raise ValueError(
        f"Unknown weight type '{weight_type}'. Use 'powerlaw' or 'geometric'."
    )
