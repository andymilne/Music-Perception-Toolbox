"""The pre-MAET as one object: the triple (p_attr, w_attr, specs).

A pre-MAET (Milne 2026, Def. 2.6) is an event sequence together with the
element multisets each event contributes to each attribute and the
parameters that turn those elements into a density. In the toolbox that is
carried by three objects -- ``p_attr``, ``w_attr``, and ``specs`` -- which
are always passed together and always describe the same pre-MAET. This
module gives them a single named-field object so that a pre-MAET can be
held in one variable, returned from one output, and passed on whole.

That object is a plain :class:`dict` with the keys ``p_attr``, ``w_attr``,
and ``specs``, built and validated by :func:`pre_maet`. It is not a class:
the parts stay ordinary Python objects, so a caller may read or replace any
of them directly, and every function that accepts a whole pre-MAET equally
accepts the three parts written out.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

__all__ = ["pre_maet", "unpack_pre_maet", "is_pre_maet"]

_KEYS = ("p_attr", "w_attr", "specs")


def is_pre_maet(obj) -> bool:
    """Return True if ``obj`` is a pre-MAET.

    The test is a mapping carrying all three of ``p_attr``, ``w_attr``,
    and ``specs`` --- what :func:`pre_maet` builds and what every
    pre-MAET operator returns. It decides only whether an argument is a
    whole pre-MAET or a bare ``p_attr``; :func:`pre_maet` performs the
    real validation.
    """
    return isinstance(obj, Mapping) and all(k in obj for k in _KEYS)


def pre_maet(p_attr, w_attr=None, specs=None):
    """Build a validated pre-MAET.

    Parameters
    ----------
    p_attr : sequence of array-like, or Mapping
        A length-A sequence of per-attribute value matrices, each of shape
        ``(K_a, N)``. An existing pre-MAET is also accepted, in which
        case its parts supply any of ``w_attr`` and ``specs`` not given
        here, and the result is a fresh one.
    w_attr : None, scalar, or sequence, optional
        Per-attribute weights: ``None`` for unweighted, a scalar applied to
        every attribute, or a length-A sequence whose entries are scalars,
        ``(N,)`` vectors, or ``(K_a, N)`` matrices.
    specs : None or sequence of dict, optional
        A length-A sequence of per-attribute specifications.

    Returns
    -------
    dict
        A dict with the keys ``p_attr``, ``w_attr``, and ``specs``.

    Raises
    ------
    TypeError, ValueError
        If the parts are not mutually consistent -- most often a
        ``w_attr`` or ``specs`` whose length does not match A.

    See Also
    --------
    unpack_pre_maet : Split a pre-MAET back into the three parts.
    """
    if is_pre_maet(p_attr):
        base = p_attr
        _reject_unknown_keys(base)
        p_attr = base["p_attr"]
        if w_attr is None:
            w_attr = base.get("w_attr")
        if specs is None:
            specs = base.get("specs")
    elif isinstance(p_attr, Mapping):
        raise TypeError(
            "pre_maet: a mapping first argument must be a pre-MAET, "
            f"carrying the keys {list(_KEYS)}; got keys "
            f"{sorted(p_attr)}."
        )

    p_list = _normalise_p_attr(p_attr)
    A = len(p_list)
    w_list = _normalise_w_attr(w_attr, A)
    specs_list = _normalise_specs(specs, A)
    return {"p_attr": p_list, "w_attr": w_list, "specs": specs_list}


def make_pre_maet(p_attr, w_attr=None, specs=None):
    """Assemble a pre-MAET without the cross-part checks.

    For the sweep form, where p_attr holds one length-A entry per sweep
    index and so does not share its length with w_attr and specs.
    """
    return {"p_attr": p_attr, "w_attr": w_attr, "specs": specs}


def unpack_pre_maet(pm):
    """Split a pre-MAET into ``(p_attr, w_attr, specs)``.

    Parameters
    ----------
    pm : Mapping
        A pre-MAET, as built by :func:`pre_maet` or returned by any
        pre-MAET operator.

    Returns
    -------
    tuple
        The three parts, in the order the loose-triple signatures take
        them. ``w_attr`` and ``specs`` are ``None`` where unset.
    """
    if not is_pre_maet(pm):
        raise TypeError(
            "unpack_pre_maet: expected a pre-MAET: a mapping "
            f"with the keys {list(_KEYS)}."
        )
    return pm["p_attr"], pm.get("w_attr"), pm.get("specs")


def shift_lead(p_attr, w_attr, following, specs, *, func):
    """Resolve the leading arguments of a pre-MAET operator.

    The operators accept either a whole pre-MAET followed by their own
    positional arguments, or ``p_attr`` and ``w_attr`` followed by those
    same positional arguments. Where a whole one was given, every
    positional argument sits one slot early and is shifted back here, and
    its ``specs`` stands unless the call named ``specs`` itself.

    Parameters
    ----------
    p_attr : sequence or Mapping
        The operator's first argument as received.
    w_attr : object
        The operator's second argument as received.
    following : list
        The operator's remaining positional arguments, in order.
    specs : object
        The operator's ``specs`` keyword as received.
    func : str
        The operator's name, for error messages.

    Returns
    -------
    tuple
        ``(p_attr, w_attr, following, specs)``, all resolved.
    """
    if not is_pre_maet(p_attr):
        return p_attr, w_attr, list(following), specs
    pm = p_attr
    following = list(following)
    if w_attr is None:
        # Everything after the pre-MAET was named, so nothing has moved.
        shifted = following
    elif not following:
        raise TypeError(
            f"{func}: too many positional arguments. Given a pre-MAET "
            "pre-MAET, the weights are taken from it and must not be "
            "passed again."
        )
    else:
        if following[-1] is not None:
            raise TypeError(
                f"{func}: too many positional arguments. Given a pre-MAET "
                "pre-MAET, the weights are taken from it and must not be "
                "passed again."
            )
        shifted = [w_attr] + following[:-1]
    if specs is None:
        specs = pm.get("specs")
    return pm["p_attr"], pm.get("w_attr"), shifted, specs


def _reject_unknown_keys(pm):
    extra = [k for k in pm if k not in _KEYS]
    if extra:
        raise ValueError(
            "pre_maet: a pre-MAET carries only the keys "
            f"{list(_KEYS)}; got the extra key(s) {sorted(extra)}."
        )


def _normalise_p_attr(p_attr):
    if isinstance(p_attr, np.ndarray) and p_attr.ndim <= 2:
        raise TypeError(
            "pre_maet: p_attr must be a sequence of per-attribute value "
            "matrices, not a single array. Wrap a single attribute as "
            "[values]."
        )
    if isinstance(p_attr, (str, bytes)) or not hasattr(p_attr, "__len__"):
        raise TypeError(
            "pre_maet: p_attr must be a sequence of per-attribute value "
            "matrices."
        )
    p_list = list(p_attr)
    if not p_list:
        raise ValueError("pre_maet: p_attr must hold at least one attribute.")
    return p_list


def _normalise_w_attr(w_attr, A):
    if w_attr is None:
        return None
    if np.isscalar(w_attr):
        return w_attr
    if isinstance(w_attr, np.ndarray) and w_attr.ndim == 0:
        return w_attr
    if not hasattr(w_attr, "__len__"):
        raise TypeError(
            "pre_maet: w_attr must be None, a scalar, or a length-A "
            "sequence of per-attribute weights."
        )
    w_list = list(w_attr)
    if len(w_list) != A:
        raise ValueError(
            f"pre_maet: w_attr must have length A = {A}; got "
            f"{len(w_list)}."
        )
    return w_list


def _normalise_specs(specs, A):
    if specs is None:
        return None
    if isinstance(specs, Mapping):
        raise TypeError(
            "pre_maet: specs must be a length-A sequence of per-attribute "
            "specifications, not a single specification. Wrap a single "
            "attribute as [spec]."
        )
    if not hasattr(specs, "__len__"):
        raise TypeError(
            "pre_maet: specs must be None or a length-A sequence of "
            "per-attribute specifications."
        )
    specs_list = list(specs)
    if len(specs_list) != A:
        raise ValueError(
            f"pre_maet: specs must have length A = {A}; got "
            f"{len(specs_list)}."
        )
    return specs_list
