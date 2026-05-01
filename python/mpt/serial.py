"""Serial-position features for ordered sequences.

This module provides two utilities for analyses of ordered event
sequences:

:func:`continuity`
    Expected length and signed magnitude of the backward
    same-direction run leading up to each query, under Gaussian
    pitch uncertainty. Consumes the same
    ``(p_attr_diff, w_diff)`` stream that
    :func:`~mpt.difference_events` produces at differencing order 1,
    but reads it as an ordered sequence with a directional gate and
    a break condition rather than aggregating it order-free into a
    tensor.
:func:`seq_weights`
    Constructor for length-*N* position-weight vectors from named
    specifications (``'flat'``, ``'primacy'``, ``'recency'``,
    ``'exponentialFromStart'``, ``'exponentialFromEnd'``,
    ``'uShape'``) or explicit vectors, with optional time-based
    decay. The output is a plain non-negative numeric vector,
    usable anywhere a weight argument is accepted — e.g. as the
    event weights of :func:`~mpt.build_exp_tens`, the ``w`` argument
    of :func:`~mpt.add_spectra`, or the ``w`` argument of
    :func:`continuity`.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erf as _erf

from .tensor import difference_events


# =====================================================================
#  continuity — backward same-direction run
# =====================================================================


def continuity(
    seq,
    x,
    sigma: float,
    *,
    w=None,
    mode: str = "strict",
    theta: float | None = None,
):
    """Expected length and signed magnitude of the backward
    same-direction run leading up to each query, under Gaussian pitch
    uncertainty, with optional per-event salience weights.

    Parameters
    ----------
    seq : array-like
        1-D ordered sequence of pitches or positions (length *N*).
    x : array-like
        Query points. Scalar or 1-D array of length *M*.
    sigma : float
        Gaussian pitch uncertainty. Use ``0`` for the discrete limit.
    w : None, scalar, or array-like, optional
        Per-event salience weights. ``None`` (default) or an empty
        array broadcasts weight 1 to every event. A scalar broadcasts
        to a length-*N* uniform vector. A length-*N* vector provides
        per-event salience. All weights must be non-negative. The
        salience of difference event *k* — the interval from
        ``seq[k]`` to ``seq[k+1]`` — is the product ``w[k] * w[k+1]``
        (rolling product of width 2, matching
        :func:`difference_events` at order 1), interpretable as the
        probability that both endpoints are perceived. The ``count``
        and ``magnitude`` contributions from interval *k* are scaled
        by this salience; the directional-break threshold *θ* acts on
        the unweighted sign-product, so weights modulate contribution
        size without affecting when the backward walk halts.
    mode : {'strict', 'lenient'}, optional
        ``'strict'`` sets θ = 0; ``'lenient'`` sets θ = −1.
        Default ``'strict'``.
    theta : float or None, optional
        Explicit break threshold in [−1, +1], overrides ``mode``.

    Returns
    -------
    count, magnitude : ndarray, ndarray
        Each of length *M*. ``count`` is non-negative; ``magnitude``
        is signed (positive ascending, negative descending). The ratio
        ``magnitude / count`` gives a trend-slope measure.

    Notes
    -----
    Defined only on linearly ordered domains — those where the
    ordering is inherited from the real line. For pitch, these
    include pitch heights, pitch intervals (differences), signed
    interval changes (second differences), and higher-order
    differences. For time, the analogous sequence starts one level
    higher — IOIs (differences between successive event times),
    signed IOI changes, and so on — because event times are by
    convention monotonically increasing, so direction on raw time
    stamps is trivially always positive and carries no information
    relevant to continuity; only from IOIs onward can the sequence
    change direction. Not applicable to periodic (pitch-class) data,
    for which direction is inherently ambiguous on a cycle.

    The difference-event representation and its rolling-product
    weight propagation are shared with the MAET-with-differencing
    pipeline: ``continuity`` consumes the same
    ``(p_attr_diff, w_diff)`` stream that :func:`difference_events`
    produces at order 1, but reads it as an *ordered* sequence with a
    directional gate and a break condition, rather than aggregating
    it order-free into a tensor.

    See Also
    --------
    difference_events, seq_weights
    """
    seq = np.asarray(seq, dtype=np.float64).ravel()
    x_arr = np.atleast_1d(np.asarray(x, dtype=np.float64)).ravel()

    if mode not in ("strict", "lenient"):
        raise ValueError(
            f"mode must be 'strict' or 'lenient' (got {mode!r})."
        )
    if theta is None:
        theta = 0.0 if mode == "strict" else -1.0
    else:
        theta = float(theta)
        if theta < -1.0 - 1e-12 or theta > 1.0 + 1e-12:
            raise ValueError(f"theta must be in [-1, +1] (got {theta}).")

    N = seq.size
    M = x_arr.size
    count = np.zeros(M, dtype=np.float64)
    magnitude = np.zeros(M, dtype=np.float64)
    if N < 2:
        return count, magnitude

    # --- Normalise weights to None or a length-N vector ---
    w_vec = _normalise_continuity_weights(w, N)

    # --- Compute difference events and their weights via
    #     difference_events, so the per-event-salience reading and
    #     rolling-product rule match the MAET preprocessing helper
    #     exactly. ---
    seq_row = seq.reshape(1, -1)
    if w_vec is None:
        p_diff, _ = difference_events([seq_row], None, None, [1], [0])
        diff_weights = None
    else:
        p_diff, w_diff = difference_events(
            [seq_row], [w_vec.reshape(1, -1)], None, [1], [0]
        )
        diff_weights = np.asarray(w_diff[0]).reshape(-1)

    ctx_intervals = np.asarray(p_diff[0]).reshape(-1)

    def _erf_sigma(arr):
        if sigma <= 0:
            return np.sign(arr)
        return _erf(arr / (2.0 * sigma))

    d_ctx = _erf_sigma(ctx_intervals)

    for i in range(M):
        i_N = x_arr[i] - seq[-1]
        d_N = _erf_sigma(np.array([i_N]))[0]
        a = d_ctx * d_N
        c = 0.0
        m = 0.0
        for k in range(N - 2, -1, -1):
            a_k = a[k]
            if a_k <= theta:
                break
            contrib = max(a_k, 0.0)
            if diff_weights is not None:
                contrib = contrib * diff_weights[k]
            c += contrib
            m += contrib * ctx_intervals[k]
        count[i] = c
        magnitude[i] = m

    return count, magnitude


def _normalise_continuity_weights(w, N: int):
    """Return a length-N float array, or None if *w* signals uniform.

    Accepts None, an empty array, a non-negative scalar, or a
    non-negative 1-D array of length *N*. A scalar is broadcast to a
    uniform length-*N* vector so that the downstream rolling-product
    rule applies consistently (uniform-scalar *c* yields difference-
    event salience *c²*).
    """
    if w is None:
        return None
    wa = np.asarray(w, dtype=np.float64)
    if wa.size == 0:
        return None
    if wa.ndim == 0 or wa.size == 1:
        val = float(wa.reshape(-1)[0])
        if val < 0:
            raise ValueError(
                "continuity weights must be non-negative "
                f"(got scalar {val})."
            )
        return np.full(N, val, dtype=np.float64)
    wa = wa.ravel()
    if wa.size != N:
        raise ValueError(
            "continuity weight vector must have length N = "
            f"{N} (got length {wa.size})."
        )
    if np.any(wa < 0):
        raise ValueError("continuity weights must be non-negative.")
    return wa


# =====================================================================
#  seq_weights — position-weight vector constructor
# =====================================================================


def seq_weights(
    w,
    spec,
    *,
    n=None,
    decay_rate: float = 1.0,
    alpha: float = 0.5,
    t=None,
) -> np.ndarray:
    """Apply a position-weighting profile to an existing weight vector.

    Constructs a length-N profile from the named or explicit
    specification and returns its pointwise product with ``w``.

    The length N of the output is inferred from ``w`` when ``w`` is a
    non-empty, non-scalar array-like. When ``w`` is ``None`` or scalar,
    ``n`` must be supplied explicitly as a keyword argument.

    Parameters
    ----------
    w : array-like, scalar, or None
        Length-N vector of per-position weights, ``None`` for all
        ones — requires ``n`` —, or a scalar broadcast to length
        N — requires ``n``.
    spec : str or array-like
        Named specification — ``'flat'``, ``'primacy'``, ``'recency'``,
        ``'exponentialFromStart'``, ``'exponentialFromEnd'``,
        ``'uShape'`` — or an explicit length-N numeric vector
        (passthrough with length validation).
    n : int or None, optional
        Output length. Required when ``w`` is ``None`` or scalar;
        otherwise inferred from ``len(w)`` and validated if also
        supplied.
    decay_rate : float, optional
        Non-negative decay rate for exponential and uShape specs.
        Zero decay gives a uniform profile. Default 1.0.
    alpha : float, optional
        Mixing in [0, 1] for ``'uShape'``. ``alpha = 1`` ≡
        ``'exponentialFromStart'``; ``alpha = 0`` ≡
        ``'exponentialFromEnd'``; ``alpha = 0.5`` gives symmetric U.
        Default 0.5.
    t : array-like or None, optional
        Strictly increasing time index of length N. When supplied,
        decay operates over elapsed time from the relevant endpoint
        rather than over position index. Default None (unit spacing).

    Returns
    -------
    ndarray
        Length-N non-negative weight vector equal to
        ``profile(spec) * w``.
    """
    # Determine N and normalise w to a length-N array
    if w is None:
        if n is None:
            raise ValueError(
                "n must be supplied as a keyword argument "
                "when w is None (all ones)."
            )
        n = int(n)
        if n < 1:
            raise ValueError(f"n must be >= 1 (got {n}).")
        w_arr = np.ones(n, dtype=np.float64)
    elif np.isscalar(w):
        if n is None:
            raise ValueError(
                "n must be supplied as a keyword argument "
                "when w is a scalar."
            )
        n = int(n)
        if n < 1:
            raise ValueError(f"n must be >= 1 (got {n}).")
        w_arr = np.full(n, float(w), dtype=np.float64)
    else:
        w_arr = np.asarray(w, dtype=np.float64).ravel()
        inferred_n = w_arr.size
        if inferred_n < 1:
            raise ValueError("w must be non-empty.")
        if n is None:
            n = inferred_n
        else:
            n = int(n)
            if n < 1:
                raise ValueError(f"n must be >= 1 (got {n}).")
            if n != inferred_n:
                raise ValueError(
                    f"n = {n} does not match length of w ({inferred_n}). "
                    f"Either omit n or supply a consistent value."
                )

    if not isinstance(spec, str):
        profile = np.asarray(spec, dtype=np.float64).ravel()
        if profile.size != n:
            raise ValueError(
                f"Profile vector length must be {n} "
                f"(got {profile.size})."
            )
        return profile * w_arr

    if decay_rate < 0:
        raise ValueError(
            f"decay_rate must be non-negative (got {decay_rate})."
        )
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1] (got {alpha}).")

    if t is None:
        t_arr = np.arange(n, dtype=np.float64)
    else:
        t_arr = np.asarray(t, dtype=np.float64).ravel()
        if t_arr.size != n:
            raise ValueError(
                f"t must have length {n} (got {t_arr.size})."
            )
        if np.any(np.diff(t_arr) <= 0):
            raise ValueError("t must be strictly increasing.")
        t_arr = t_arr - t_arr[0]

    if spec == "flat":
        profile = np.ones(n, dtype=np.float64)
    elif spec == "primacy":
        profile = np.zeros(n, dtype=np.float64)
        profile[0] = 1.0
    elif spec == "recency":
        profile = np.zeros(n, dtype=np.float64)
        profile[-1] = 1.0
    elif spec == "exponentialFromStart":
        profile = np.exp(-decay_rate * t_arr)
    elif spec == "exponentialFromEnd":
        profile = np.exp(-decay_rate * (t_arr[-1] - t_arr))
    elif spec == "uShape":
        v_s = np.exp(-decay_rate * t_arr)
        v_e = np.exp(-decay_rate * (t_arr[-1] - t_arr))
        profile = alpha * v_s + (1.0 - alpha) * v_e
    else:
        raise ValueError(
            f"Unknown weight specification {spec!r}. Expected "
            f"'flat', 'primacy', 'recency', 'exponentialFromStart', "
            f"'exponentialFromEnd', 'uShape', or an explicit vector."
        )

    return profile * w_arr
