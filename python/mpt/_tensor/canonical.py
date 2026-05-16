"""Canonical-form key helpers for batched cosine-similarity dedup.

Pure stateless utilities used by :func:`batch_cos_sim_exp_tens` (and a
single per-chord call from :func:`eval_exp_tens` cache plumbing) to
produce hashable keys identifying inputs up to the relevant symmetry
group:

* :func:`_chord_canonical_key` --- single-multiset key, invariant
  under permutation of pitches; under cyclic rotation in the periodic
  case; and under translation in the relative case.
* :func:`_pair_canonical_key` --- ordered-pair key with the same
  symmetries applied jointly to both multisets.

Lower-level building blocks (:func:`_lex_compare`,
:func:`_cyclic_canonical`, :func:`_canonicalize_set`) are exposed for
testing but not part of the public API.

See :doc:`/ARCHITECTURE` §5 ("Canonical-form dedup") for the conceptual
introduction; this module is the implementation.
"""
from __future__ import annotations

import numpy as np




# -------------------------------------------------------------------
#  Canonicalization helpers (for batch_cos_sim_exp_tens)
# -------------------------------------------------------------------


def _lex_compare(a: tuple, b: tuple) -> int:
    """Lexicographic comparison. Returns -1, 0, or +1."""
    for ai, bi in zip(a, b):
        if ai < bi:
            return -1
        if ai > bi:
            return 1
    return 0



def _cyclic_canonical(
    p_sorted: np.ndarray,
    w_sorted: np.ndarray | None,
    period: float,
) -> tuple[tuple, tuple | None, float]:
    """Lexicographically smallest rotation of a periodic pitch set.

    Tries all n rotations (subtract each sorted pitch, mod period,
    re-sort with weights) and returns the lex-smallest form plus
    the shift that produced it. This captures all
    transposition-modulo-period equivalences.

    Pitch values are rounded to 9 decimal places before lex
    comparison and in the returned tuple, to absorb floating-point
    noise from mod-reduction. 9 decimals is below any musically-
    meaningful precision (1 attocent / 1 nanosecond) but well above
    typical FP roundoff. Without this rounding, two
    transposition-equivalent multisets with different FP error
    patterns can produce different canonical forms — a real bug
    that breaks consumer-level dedup for non-integer pitch data.

    Returns
    -------
    best_p : tuple
        Canonical pitch tuple (rounded to 9 decimals).
    best_w : tuple or None
        Canonical weight tuple (if weights provided).
    best_shift : float
        The pitch value subtracted to produce the canonical form
        (returned at full precision; only the canonical *form* is
        rounded, not the shift itself, so callers using the shift
        to apply to a paired set get exact arithmetic).
    """
    n = len(p_sorted)
    has_w = w_sorted is not None
    ROUND_DIGITS = 9

    best_p = tuple(np.round(p_sorted - p_sorted[0], ROUND_DIGITS))
    best_w = tuple(w_sorted) if has_w else None
    best_shift = p_sorted[0]

    for rot in range(1, n):
        shifted = np.mod(p_sorted - p_sorted[rot], period)
        si = np.argsort(shifted)
        shifted = np.round(shifted[si], ROUND_DIGITS)
        t_p = tuple(shifted)

        cmp = _lex_compare(t_p, best_p)
        if cmp < 0:
            best_p = t_p
            best_w = tuple(w_sorted[si]) if has_w else None
            best_shift = p_sorted[rot]
        elif cmp == 0 and has_w:
            t_w = tuple(w_sorted[si])
            if _lex_compare(t_w, best_w) < 0:
                best_w = t_w
                best_shift = p_sorted[rot]

    return best_p, best_w, best_shift



def _canonicalize_set(
    p: np.ndarray,
    w: np.ndarray | None,
    is_rel: bool,
    is_per: bool,
    period: float,
) -> tuple[tuple, tuple | None]:
    """Canonical form of a pitch/weight set under isPer/isRel.

    Returns hashable tuples suitable for use as dict keys.
    """
    has_w = w is not None

    # Sort pitches, align weights
    si = np.argsort(p)
    p = p[si]
    if has_w:
        w = w[si]

    # Reduce modulo period
    if is_per:
        p = np.mod(p, period)
        si = np.argsort(p)
        p = p[si]
        if has_w:
            w = w[si]

    # Remove transposition
    if is_rel:
        if is_per:
            # Cyclic canonical form: the lex-smallest rotation captures
            # all transposition-modulo-period equivalences.
            ca_p, ca_w, _ = _cyclic_canonical(p, w, period)
            return ca_p, ca_w
        else:
            p = p - p[0]

    return tuple(p), (tuple(w) if has_w else None)



# -------------------------------------------------------------------
#  Canonical-key primitives for batched / deduplicated workflows
# -------------------------------------------------------------------


def _chord_canonical_key(
    p,
    w,
    *,
    sigma: float,
    r: int,
    is_rel: bool,
    is_per: bool,
    period: float,
    precision: int | None = None,
):
    """Canonical hashable form of a single weighted multiset.

    Two chords ``(p1, w1)`` and ``(p2, w2)`` produce the same key iff
    their resulting density object is structurally identical (same
    ``(p, w, sigma, r, is_rel, is_per, period)``-determined density),
    regardless of input-side permutation or, in relative modes,
    in-batch translation. Used by the consumer-level deduplication in
    :func:`cos_sim_exp_tens`, :func:`windowed_similarity`, and the
    harmony wrappers when given batched chord input.

    Parameters
    ----------
    p : array-like
        Pitch values (will be flattened; NaN handling is the caller's
        responsibility — pass NaN-stripped arrays).
    w : array-like or None
        Weights, same length as ``p``. None means uniform weights.
    sigma, r, is_rel, is_per, period :
        Density-determining parameters. Baked into the returned key
        so different parameter settings produce different keys.
    precision : int, optional
        Round the canonical pitch and weight values to this many
        decimal places (after the canonicalisation, to absorb FP noise
        from mod-reduction and subtraction). Default: no rounding.

    Returns
    -------
    key : tuple
        Hashable canonical form, suitable as a dict key.
    p_canon : np.ndarray, dtype=float64
        Canonical pitch array (for downstream density caching).
    w_canon : np.ndarray | None
        Canonical weight array, or None if ``w`` is None.
    """
    p_arr = np.asarray(p, dtype=np.float64)
    w_arr = np.asarray(w, dtype=np.float64) if w is not None else None

    ca_p, ca_w = _canonicalize_set(p_arr, w_arr, is_rel, is_per, period)

    if precision is not None:
        ca_p = tuple(round(x, precision) for x in ca_p)
        if ca_w is not None:
            ca_w = tuple(round(x, precision) for x in ca_w)

    key = (ca_p, ca_w, sigma, r, is_rel, is_per, period)

    p_canon = np.array(ca_p, dtype=np.float64)
    w_canon = np.array(ca_w, dtype=np.float64) if ca_w is not None else None

    return key, p_canon, w_canon



def _pair_canonical_key(
    p_a,
    w_a,
    p_b,
    w_b,
    *,
    sigma: float,
    r: int,
    is_rel: bool,
    is_per: bool,
    period: float,
    precision: int | None = None,
):
    """Canonical hashable forms of a paired weighted multiset (A, B).

    The cosine similarity of two expectation tensor densities is
    invariant under certain joint transformations of the pair. The
    canonical pair form encodes these symmetries so that
    structurally-equivalent pairs produce the same key, enabling
    deduplication.

    The exploited symmetries depend on the mode:

    - **Relative** (``is_rel=True``): independent transposition of
      each set. Each side is canonicalised separately via
      :func:`_canonicalize_set`.
    - **Absolute** (``is_rel=False``): joint co-transposition.
      ``cos_sim_exp_tens(A + c, B + c) == cos_sim_exp_tens(A, B)``,
      so A's canonical form determines a shift, and the same shift
      is applied to B. For ``is_per=True``, A is reduced to its
      cyclic canonical form (the lex-smallest rotation), and B is
      shifted by the corresponding amount mod period; for
      ``is_per=False``, A is translated so its minimum is at 0, and
      B is shifted by the same amount.

    Parameters
    ----------
    p_a, p_b : array-like
        Pitch values for A and B (NaN-stripped).
    w_a, w_b : array-like or None
        Weights for A and B, or None for uniform.
    sigma, r, is_rel, is_per, period :
        Density-determining parameters. Baked into both returned keys.
    precision : int, optional
        Post-canonicalisation rounding. Default: no rounding.

    Returns
    -------
    key_a, key_b : tuple
        Hashable canonical keys for A and B. The pair key is
        ``(key_a, key_b)``; ``key_a`` and ``key_b`` are also valid as
        single-side dedup keys for caching A and B densities
        respectively.
    p_a_canon, p_b_canon : np.ndarray, dtype=float64
        Canonical pitch arrays.
    w_a_canon, w_b_canon : np.ndarray | None
        Canonical weight arrays, or None if the corresponding input
        weight was None.
    """
    pa_arr = np.asarray(p_a, dtype=np.float64)
    pb_arr = np.asarray(p_b, dtype=np.float64)
    wa_arr = np.asarray(w_a, dtype=np.float64) if w_a is not None else None
    wb_arr = np.asarray(w_b, dtype=np.float64) if w_b is not None else None

    if is_rel:
        # Independent canonicalisation per side.
        ca_p, ca_w = _canonicalize_set(pa_arr, wa_arr, is_rel, is_per, period)
        cb_p, cb_w = _canonicalize_set(pb_arr, wb_arr, is_rel, is_per, period)
    else:
        # Joint co-transposition: A determines the shift, B inherits it.
        si_a = np.argsort(pa_arr)
        pa_s = pa_arr[si_a]
        wa_s = wa_arr[si_a] if wa_arr is not None else None

        if is_per:
            pa_s = np.mod(pa_s, period)
            si = np.argsort(pa_s)
            pa_s = pa_s[si]
            if wa_s is not None:
                wa_s = wa_s[si]
            # Cyclic canonical form — collapses all rotations.
            ca_p, ca_w, shift = _cyclic_canonical(pa_s, wa_s, period)
        else:
            shift = pa_s[0]
            ca_p = tuple(pa_s - shift)
            ca_w = tuple(wa_s) if wa_s is not None else None

        # Apply the same shift to B.
        si_b = np.argsort(pb_arr)
        pb_s = pb_arr[si_b]
        wb_s = wb_arr[si_b] if wb_arr is not None else None

        if is_per:
            pb_shifted = np.mod(pb_s - shift, period)
            si = np.argsort(pb_shifted)
            cb_p = tuple(pb_shifted[si])
            cb_w = tuple(wb_s[si]) if wb_s is not None else None
        else:
            cb_p = tuple(pb_s - shift)
            cb_w = tuple(wb_s) if wb_s is not None else None

    if precision is not None:
        ca_p = tuple(round(x, precision) for x in ca_p)
        cb_p = tuple(round(x, precision) for x in cb_p)
        if ca_w is not None:
            ca_w = tuple(round(x, precision) for x in ca_w)
        if cb_w is not None:
            cb_w = tuple(round(x, precision) for x in cb_w)

    key_a = (ca_p, ca_w, sigma, r, is_rel, is_per, period)
    key_b = (cb_p, cb_w, sigma, r, is_rel, is_per, period)

    p_a_canon = np.array(ca_p, dtype=np.float64)
    p_b_canon = np.array(cb_p, dtype=np.float64)
    w_a_canon = np.array(ca_w, dtype=np.float64) if ca_w is not None else None
    w_b_canon = np.array(cb_w, dtype=np.float64) if cb_w is not None else None

    return key_a, key_b, p_a_canon, w_a_canon, p_b_canon, w_b_canon