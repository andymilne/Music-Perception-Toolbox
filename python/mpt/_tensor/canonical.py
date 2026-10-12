"""Canonical-form key helpers for batched cosine-similarity dedup.

Pure stateless utilities used by the batched-raw mode of
:func:`~mpt.sim_maet` (and a
single per-chord call from :func:`eval_maet` cache plumbing) to
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

import math

import numpy as np

# _cyclic_canonical: sets of at most _CYCLIC_LOOP_MAX_N values are searched
# rotation by rotation; sets with more than _CYCLIC_NARROW_MIN_N values
# narrow their candidate rotations before exact comparison, until at
# most _CYCLIC_EXACT_MAX_ROWS remain; exact rows are evaluated in blocks
# of about _CYCLIC_BLOCK_ELEMS elements (8 MB of float64 per array).
_CYCLIC_LOOP_MAX_N = 6

# Decimal places of canonical forms where no kernel width is known (the
# structural, sigma-free callers). Where sigma is known the places are
# set by _key_round_digits instead, from the accuracy floor.
_DEFAULT_ROUND_DIGITS = 9

# Largest change in a pairwise inner-product kernel exp(-|d|^2/(4 sigma^2))
# per unit change of |d|, times sigma: sqrt(2) * exp(-1/2). The point
# kernel exp(-|d|^2/(2 sigma^2)) has the smaller exp(-1/2), so this bounds
# both.
_KERNEL_SLOPE = 2.0 ** 0.5 * 0.6065306597126334
_CYCLIC_NARROW_MIN_N = 16
_CYCLIC_EXACT_MAX_ROWS = 4
_CYCLIC_BLOCK_ELEMS = 1_000_000




# -------------------------------------------------------------------
#  Canonicalization helpers (for the batched-raw mode)
# -------------------------------------------------------------------


def _lex_compare(a: tuple, b: tuple) -> int:
    """Lexicographic comparison. Returns -1, 0, or +1."""
    for ai, bi in zip(a, b):
        if ai < bi:
            return -1
        if ai > bi:
            return 1
    return 0



def _key_round_digits(sigma, r, truncation_sigmas=None) -> int:
    """Decimal places for canonical forms, set by the accuracy floor.

    Rounding a canonical form moves each value by at most
    ``delta = 0.5 * 10**-digits``, so a tuple centre moves by at most
    ``sqrt(r) * delta`` and a difference of two centres by twice that.
    A Gaussian kernel of width ``sigma`` then changes by at most
    ``_KERNEL_SLOPE * 2 * sqrt(r) * delta / sigma`` of its peak. The
    places are the fewest that keep this within the truncation floor
    ``exp(-k^2/2)`` (k = the resolved truncation_sigmas;
    :func:`mpt._defaults.truncation_floor`), so rounding perturbs no
    kernel contribution by more than truncation may omit, and the
    accuracy bound truncation_sigmas states is kept. Rounding as coarsely
    as that allows absorbs as much floating-point noise as possible, so
    equivalent inputs computed by different arithmetic share a key.
    """
    from .._defaults import truncation_floor

    sigma = float(sigma)
    if not (math.isfinite(sigma) and sigma > 0):
        return _DEFAULT_ROUND_DIGITS
    tau = truncation_floor(truncation_sigmas)
    delta_max = tau * sigma / (2.0 * _KERNEL_SLOPE * math.sqrt(max(int(r), 1)))
    digits = math.ceil(math.log10(0.5 / delta_max))
    return int(min(max(digits, -12), 15))


def _lex_min_row(P: np.ndarray, W: np.ndarray | None) -> int:
    """Index of the lexicographically smallest row of P (ties broken by
    the rows of W, if given, then by the lowest index).

    Repeatedly finds the first column in which the remaining rows
    differ and keeps those with its smallest value, so columns in which
    they all agree are skipped together.
    """
    cand = np.arange(P.shape[0])
    for M in (P, W):
        if M is None:
            continue
        j0 = 0
        while cand.size > 1:
            sub = M[cand, j0:]
            differs = np.flatnonzero((sub != sub[0]).any(axis=0))
            if differs.size == 0:
                break
            j = differs[0]
            col = sub[:, j]
            cand = cand[col == col.min()]
            j0 += j + 1
    return int(cand[0])


def _cyclic_canonical(
    p_sorted: np.ndarray,
    w_sorted: np.ndarray | None,
    period: float,
    return_ties: bool = False,
    digits: int = _DEFAULT_ROUND_DIGITS,
):
    """Lexicographically smallest rotation of a periodic pitch set.

    Considers all n rotations (subtract each sorted pitch, mod period,
    re-sort with weights) and returns the lex-smallest form plus the
    shift that produced it, ties going to the earliest rotation. This
    captures all transposition-modulo-period equivalences. The
    rotations are narrowed in O(n) per column before the few that
    remain are compared in full (see the comments below); the result is
    that of a rotation-by-rotation search (:func:`_cyclic_canonical_loop`,
    used directly for very small sets).

    Pitch values are rounded to ``digits`` decimal places (default 9;
    callers that know the kernel width pass :func:`_key_round_digits`)
    before lex comparison and in the returned tuple, to absorb
    floating-point noise from mod-reduction. Without this rounding, two
    transposition-equivalent multisets with different FP error
    patterns can produce different canonical forms — a real bug
    that breaks consumer-level dedup for non-integer pitch data.

    Returns
    -------
    best_p : tuple
        Canonical pitch tuple (rounded to ``digits`` decimals).
    best_w : tuple or None
        Canonical weight tuple (if weights provided).
    best_shift : float
        The pitch value subtracted to produce the canonical form
        (returned at full precision; only the canonical *form* is
        rounded, not the shift itself, so callers using the shift
        to apply to a paired set get exact arithmetic).
    """
    p_sorted = np.asarray(p_sorted, dtype=np.float64)
    n = len(p_sorted)
    has_w = w_sorted is not None
    if has_w:
        w_sorted = np.asarray(w_sorted)
    if n == 0:
        raise ValueError("_cyclic_canonical needs at least one value.")
    if n <= _CYCLIC_LOOP_MAX_N:
        out = _cyclic_canonical_loop(p_sorted, w_sorted, period, True,
                                     digits)
        return out if return_ties else out[:3]

    # Candidate rotations. The result is the earliest rotation whose
    # rounded, re-sorted form is lex-smallest. Evaluating that form for
    # every rotation costs O(n^2 log n), which dominated
    # canonicalisation of spectrally-expanded sets (n = K * partials),
    # so the rotations are first narrowed one column at a time, each
    # column costing O(n), until only a few remain for the exact
    # comparison in _cyclic_rows_best.
    #
    # Only rotations starting at the first of a run of equal values
    # need narrowing (a later member gives the same pitches). For such
    # a rotation k the form, before re-sorting, is mod(p[i] - p[k]) for
    # i = k, ..., n-1 and then i = 0, ..., k-1. Both runs are
    # non-decreasing in floating point (subtraction, the mod's single
    # adjustment, and rounding are all monotone), so the re-sorted form
    # is this cyclic order provided the first run's last value does not
    # exceed the second run's first, which is checked. Its j-th value
    # is then the same number _cyclic_rows_best computes, so the
    # narrowing compares exact values and drops only rotations that are
    # lex-greater than another.
    cand = np.arange(n)
    if n > _CYCLIC_NARROW_MIN_N:
        def colvals(rots, j):
            d = p_sorted[(rots + j) % n] - p_sorted[rots]
            return np.round(np.mod(d, period), digits)

        first = np.ones(n, dtype=bool)
        first[1:] = p_sorted[1:] != p_sorted[:-1]
        heads = np.flatnonzero(first)
        k = heads[heads > 0]
        cyclic_ok = bool(np.all(
            np.round(np.mod(p_sorted[-1] - p_sorted[k], period), digits)
            <= np.round(np.mod(p_sorted[0] - p_sorted[k], period), digits)))
        if cyclic_ok:
            cand = heads
            j = 1
            while cand.size > _CYCLIC_EXACT_MAX_ROWS and j < n:
                col = colvals(cand, j)
                cand = cand[col == col.min()]
                j += 1
            # A rotation starting at a later member of a run of equal
            # values yields the same pitches, but its weights may be
            # ordered differently, so every member of each surviving
            # run is compared exactly.
            run_id = np.cumsum(first) - 1
            cand = np.flatnonzero(np.isin(run_id, run_id[cand]))

    out = _cyclic_rows_best(p_sorted, w_sorted, period, cand, digits)
    return out if return_ties else out[:3]


def _cyclic_canonical_loop(p_sorted, w_sorted, period, return_ties=False,
                           digits=_DEFAULT_ROUND_DIGITS):
    """Rotation-by-rotation form of :func:`_cyclic_canonical`, faster
    for very small sets (n <= _CYCLIC_LOOP_MAX_N), where the
    vectorised search's fixed overhead dominates."""
    n = len(p_sorted)
    has_w = w_sorted is not None
    ROUND_DIGITS = digits

    best_p = tuple(np.round(p_sorted - p_sorted[0], ROUND_DIGITS))
    best_w = tuple(w_sorted) if has_w else None
    best_shift = p_sorted[0]
    ties = [0]

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
            ties = [rot]
        elif cmp == 0 and has_w:
            t_w = tuple(w_sorted[si])
            cmp_w = _lex_compare(t_w, best_w)
            if cmp_w < 0:
                best_w = t_w
                best_shift = p_sorted[rot]
                ties = [rot]
            elif cmp_w == 0:
                ties.append(rot)
        elif cmp == 0:
            ties.append(rot)

    if return_ties:
        return best_p, best_w, best_shift, p_sorted[np.array(ties)]
    return best_p, best_w, best_shift


def _cyclic_rows_best(p_sorted, w_sorted, period, cand,
                      digits=_DEFAULT_ROUND_DIGITS):
    """Lex-smallest exact rotated form among the rotations ``cand``.

    The form for rotation k is mod(p - p[k], period), re-sorted (with
    the weights) and rounded to ``digits`` decimals; rotation 0 is p - p[0] in
    the given order. Rows are evaluated in blocks to bound memory, and
    ties go to the earliest rotation. Returns the same triple as
    :func:`_cyclic_canonical`, plus the shifts of every rotation in
    ``cand`` that attains the same form (the first being the shift
    returned).
    """
    n = len(p_sorted)
    has_w = w_sorted is not None
    ROUND_DIGITS = digits
    block = max(1, _CYCLIC_BLOCK_ELEMS // n)
    best_p = best_w = None
    best_rot = 0
    ties = []
    for b0 in range(0, len(cand), block):
        rots = cand[b0:b0 + block]
        shifted = np.mod(p_sorted[None, :] - p_sorted[rots, None], period)
        si = np.argsort(shifted, axis=1)
        is0 = rots == 0
        if is0.any():
            shifted[is0] = p_sorted - p_sorted[0]
            si[is0] = np.arange(n)
        P = np.round(np.take_along_axis(shifted, si, axis=1), ROUND_DIGITS)
        W = w_sorted[si] if has_w else None
        b = _lex_min_row(P, W)
        cand_p = tuple(P[b])
        cand_w = tuple(W[b]) if has_w else None
        same = (P == P[b]).all(axis=1)
        if has_w:
            same &= (W == W[b]).all(axis=1)
        block_ties = [int(x) for x in rots[same]]
        if best_p is None:
            better, equal = True, False
        else:
            cmp = _lex_compare(cand_p, best_p)
            cmp_w = (_lex_compare(cand_w, best_w)
                     if (cmp == 0 and has_w) else 0)
            better = cmp < 0 or (cmp == 0 and cmp_w < 0)
            equal = cmp == 0 and cmp_w == 0
        if better:
            best_p, best_w, best_rot = cand_p, cand_w, int(rots[b])
            ties = block_ties
        elif equal:
            ties.extend(block_ties)
    return best_p, best_w, p_sorted[best_rot], p_sorted[np.array(ties)]



def _rounded_sorted(p, w, period=None, digits=_DEFAULT_ROUND_DIGITS):
    """Values rounded to ``digits`` decimals (as in :func:`_cyclic_canonical`),
    folded back into [0, period) when a period is given, and re-sorted
    with their weights. Rounding absorbs the floating-point noise left
    by mod-reduction and subtraction, which would otherwise give
    equivalent sets different keys (e.g. 22-EDO pitches reached by
    different arithmetic)."""
    p = np.round(np.asarray(p, dtype=np.float64), digits)
    if period is not None:
        p = np.mod(p, period)
    si = np.argsort(p)
    return p[si], (None if w is None else np.asarray(w)[si])


def _canonicalize_set(
    p: np.ndarray,
    w: np.ndarray | None,
    rel: bool,
    per: bool,
    period: float,
    digits: int = _DEFAULT_ROUND_DIGITS,
) -> tuple[tuple, tuple | None]:
    """Canonical form of a pitch/weight set under per/rel.

    Returns hashable tuples suitable for use as dict keys.
    """
    has_w = w is not None

    # Sort pitches, align weights
    si = np.argsort(p)
    p = p[si]
    if has_w:
        w = w[si]

    # Reduce modulo period
    if per:
        p = np.mod(p, period)
        si = np.argsort(p)
        p = p[si]
        if has_w:
            w = w[si]

    # Remove transposition
    if rel:
        if per:
            # Cyclic canonical form: the lex-smallest rotation captures
            # all transposition-modulo-period equivalences.
            ca_p, ca_w, _ = _cyclic_canonical(p, w, period, digits=digits)
            return ca_p, ca_w
        else:
            p = p - p[0]

    if per or rel:
        # Rounded to absorb the noise of the mod-reduction or subtraction
        # above. An absolute non-periodic set has had no arithmetic done
        # to it, so it is kept exactly as given.
        p, w = _rounded_sorted(p, w, period if per else None, digits)
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
    rel: bool,
    per: bool,
    period: float,
    precision: int | None = None,
    truncation_sigmas=None,
):
    """Canonical hashable form of a single weighted multiset.

    Two chords ``(p1, w1)`` and ``(p2, w2)`` produce the same key iff
    their resulting density object is structurally identical (same
    ``(p, w, sigma, r, rel, per, period)``-determined density),
    regardless of input-side permutation or, in relative modes,
    in-batch translation. Used by the consumer-level deduplication in
    :func:`sim_maet` and the
    harmony wrappers when given batched chord input.

    Parameters
    ----------
    p : array-like
        Pitch values (will be flattened; NaN handling is the caller's
        responsibility — pass NaN-stripped arrays).
    w : array-like or None
        Weights, same length as ``p``. None means uniform weights.
    sigma, r, rel, per, period :
        Density-determining parameters. Baked into the returned key
        so different parameter settings produce different keys.
    precision : int, optional
        Round the canonical pitch and weight values to this many
        decimal places (after the canonicalisation, to absorb FP noise
        from mod-reduction and subtraction). Default: no rounding
        beyond that set by ``truncation_sigmas``.
    truncation_sigmas : float, optional
        The caller's truncation setting (None: the toolbox default).
        Canonical values are rounded to the decimal places
        :func:`_key_round_digits` derives from it and ``sigma``, which
        keeps the rounding within the accuracy truncation_sigmas sets.

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

    digits = _key_round_digits(sigma, r, truncation_sigmas)
    if precision is not None:
        # An explicit precision coarser than the accuracy floor's
        # places governs the whole canonicalisation, including which
        # rotation is canonical, so that it can merge what it rounds.
        digits = min(digits, int(precision))
    ca_p, ca_w = _canonicalize_set(p_arr, w_arr, rel, per, period, digits)

    if precision is not None:
        ca_p = tuple(round(x, precision) for x in ca_p)
        if ca_w is not None:
            ca_w = tuple(round(x, precision) for x in ca_w)

    key = (ca_p, ca_w, sigma, r, rel, per, period)

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
    rel: bool,
    per: bool,
    period: float,
    precision: int | None = None,
    truncation_sigmas=None,
):
    """Canonical hashable forms of a paired weighted multiset (A, B).

    The cosine similarity of two expectation tensor densities is
    invariant under certain joint transformations of the pair. The
    canonical pair form encodes these symmetries so that
    structurally-equivalent pairs produce the same key, enabling
    deduplication.

    The exploited symmetries depend on the mode:

    - **Relative** (``rel=True``): independent transposition of
      each set. Each side is canonicalised separately via
      :func:`_canonicalize_set`.
    - **Absolute** (``rel=False``): joint co-transposition.
      ``sim_maet(A + c, B + c) == sim_maet(A, B)``,
      so A's canonical form determines a shift, and the same shift
      is applied to B. For ``per=True``, A is reduced to its
      cyclic canonical form (the lex-smallest rotation), and B is
      shifted by the corresponding amount mod period (where several
      rotations of a symmetric A give that form, by the one giving
      B's lex-smallest form); for ``per=False``, A is translated so
      its minimum is at 0, and B is shifted by the same amount. Both
      forms are rounded to the places :func:`_key_round_digits`
      derives from ``sigma``, ``r``, and ``truncation_sigmas``.

    Parameters
    ----------
    p_a, p_b : array-like
        Pitch values for A and B (NaN-stripped).
    w_a, w_b : array-like or None
        Weights for A and B, or None for uniform.
    sigma, r, rel, per, period :
        Density-determining parameters. Baked into both returned keys.
    precision : int, optional
        Post-canonicalisation rounding. Default: no rounding beyond
        that set by ``truncation_sigmas``.
    truncation_sigmas : float, optional
        The caller's truncation setting (None: the toolbox default).
        Canonical values are rounded to the decimal places
        :func:`_key_round_digits` derives from it and ``sigma``, which
        keeps the rounding within the accuracy truncation_sigmas sets.

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

    digits = _key_round_digits(sigma, r, truncation_sigmas)
    if precision is not None:
        # An explicit precision coarser than the accuracy floor's
        # places governs the whole canonicalisation, including which
        # rotation is canonical, so that it can merge what it rounds.
        digits = min(digits, int(precision))
    if rel:
        # Independent canonicalisation per side.
        ca_p, ca_w = _canonicalize_set(pa_arr, wa_arr, rel, per, period,
                                       digits)
        cb_p, cb_w = _canonicalize_set(pb_arr, wb_arr, rel, per, period,
                                       digits)
    else:
        # Joint co-transposition: A determines the shift, B inherits it.
        si_a = np.argsort(pa_arr)
        pa_s = pa_arr[si_a]
        wa_s = wa_arr[si_a] if wa_arr is not None else None

        if per:
            pa_s = np.mod(pa_s, period)
            si = np.argsort(pa_s)
            pa_s = pa_s[si]
            if wa_s is not None:
                wa_s = wa_s[si]
            # Cyclic canonical form — collapses all rotations. When A
            # is symmetric (several rotations give its canonical form,
            # e.g. a diminished seventh), each of those shifts is an
            # equally valid co-transposition, so the one giving B's
            # lex-smallest form is taken; otherwise transposed copies
            # of a pair could keep different B forms.
            ca_p, ca_w, _, tie_shifts = _cyclic_canonical(
                pa_s, wa_s, period, return_ties=True, digits=digits)
        else:
            tie_shifts = pa_s[:1]
            a_p, a_w = _rounded_sorted(pa_s - pa_s[0], wa_s, None, digits)
            ca_p = tuple(a_p)
            ca_w = tuple(a_w) if a_w is not None else None

        # Apply the same shift to B.
        si_b = np.argsort(pb_arr)
        pb_s = pb_arr[si_b]
        wb_s = wb_arr[si_b] if wb_arr is not None else None

        cb_p = cb_w = None
        for sh in tie_shifts:
            if per:
                b_p, b_w = _rounded_sorted(np.mod(pb_s - sh, period), wb_s,
                                           period, digits)
            else:
                b_p, b_w = _rounded_sorted(pb_s - sh, wb_s, None, digits)
            t_p = tuple(b_p)
            t_w = tuple(b_w) if b_w is not None else None
            if cb_p is None:
                better = True
            else:
                cmp = _lex_compare(t_p, cb_p)
                better = cmp < 0 or (cmp == 0 and t_w is not None
                                     and _lex_compare(t_w, cb_w) < 0)
            if better:
                cb_p, cb_w = t_p, t_w

    if precision is not None:
        ca_p = tuple(round(x, precision) for x in ca_p)
        cb_p = tuple(round(x, precision) for x in cb_p)
        if ca_w is not None:
            ca_w = tuple(round(x, precision) for x in ca_w)
        if cb_w is not None:
            cb_w = tuple(round(x, precision) for x in cb_w)

    key_a = (ca_p, ca_w, sigma, r, rel, per, period)
    key_b = (cb_p, cb_w, sigma, r, rel, per, period)

    p_a_canon = np.array(ca_p, dtype=np.float64)
    p_b_canon = np.array(cb_p, dtype=np.float64)
    w_a_canon = np.array(ca_w, dtype=np.float64) if ca_w is not None else None
    w_b_canon = np.array(cb_w, dtype=np.float64) if cb_w is not None else None

    return key_a, key_b, p_a_canon, w_a_canon, p_b_canon, w_b_canon