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

from ._tensor.premaet import unpack_pre_maet
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
) -> tuple[np.ndarray, np.ndarray]:
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
        p_diff, _, _ = unpack_pre_maet(difference_events([seq_row], None, [1]))
        diff_weights = None
    else:
        p_diff, w_diff, _ = unpack_pre_maet(difference_events(
            [seq_row], [w_vec.reshape(1, -1)], [1]
        ))
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
    decay_rate_start=None,
    decay_rate_end=None,
    alpha: float = 0.5,
    t=None,
) -> np.ndarray:
    """Apply a position-weighting profile to an existing weight vector.

    Constructs a length-N profile from the named, callable, or explicit
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
    spec : str, callable, or array-like
        Named specification — ``'flat'``, ``'primacy'``, ``'recency'``,
        ``'exponentialFromStart'``, ``'exponentialFromEnd'``,
        ``'uShape'``, ``'uAsym'`` — a callable ``f(t) -> profile`` of
        length N applied to the (possibly user-supplied) time vector,
        or an explicit length-N numeric vector (passthrough with
        length validation).
    n : int or None, optional
        Output length. Required when ``w`` is ``None`` or scalar;
        otherwise inferred from ``len(w)`` and validated if also
        supplied.
    decay_rate : float, optional
        Non-negative decay rate for ``'exponentialFromStart'``,
        ``'exponentialFromEnd'``, and ``'uShape'``. Used as a default
        for ``'uAsym'`` when ``decay_rate_start`` or ``decay_rate_end``
        are not supplied. Zero gives a uniform component. Default 1.0.
    decay_rate_start : float or None, optional
        Decay rate for the primacy component of ``'uAsym'``. Falls
        back to ``decay_rate`` when None. Default None.
    decay_rate_end : float or None, optional
        Decay rate for the recency component of ``'uAsym'``. Falls
        back to ``decay_rate`` when None. Default None.
    alpha : float, optional
        Mixing in [0, 1] for ``'uShape'`` and ``'uAsym'``. ``alpha = 1``
        gives pure primacy; ``alpha = 0`` gives pure recency;
        ``alpha = 0.5`` gives a balanced mix. Default 0.5.
    t : array-like or None, optional
        Strictly increasing time index of length N. When supplied,
        decay operates over elapsed time from the relevant endpoint
        rather than over position index. Default None (unit spacing).

    Returns
    -------
    ndarray
        Length-N non-negative weight vector equal to
        ``profile(spec) * w``.

    Notes
    -----
    For ``'uAsym'``, ``decay_rate_start`` has no effect when
    ``alpha = 0`` (pure recency) and ``decay_rate_end`` has no effect
    when ``alpha = 1`` (pure primacy).
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

    if callable(spec):
        profile = np.asarray(spec(t_arr), dtype=np.float64).ravel()
        if profile.size != n:
            raise ValueError(
                f"Callable spec returned length {profile.size}; "
                f"expected {n}."
            )
        return profile * w_arr

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

    # Resolve uAsym decay rates with fallback to decay_rate
    if spec == "uAsym":
        d_start = decay_rate if decay_rate_start is None else float(decay_rate_start)
        d_end   = decay_rate if decay_rate_end   is None else float(decay_rate_end)
        if d_start < 0:
            raise ValueError(
                f"decay_rate_start must be non-negative (got {d_start})."
            )
        if d_end < 0:
            raise ValueError(
                f"decay_rate_end must be non-negative (got {d_end})."
            )

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
    elif spec == "uAsym":
        v_s = np.exp(-d_start * t_arr)
        v_e = np.exp(-d_end * (t_arr[-1] - t_arr))
        profile = alpha * v_s + (1.0 - alpha) * v_e
    else:
        raise ValueError(
            f"Unknown weight specification {spec!r}. Expected "
            f"'flat', 'primacy', 'recency', 'exponentialFromStart', "
            f"'exponentialFromEnd', 'uShape', 'uAsym', a callable, "
            f"or an explicit vector."
        )

    return profile * w_arr


# =====================================================================
#  interval_kernel_cov — anisotropic kernel covariance constructor
# =====================================================================


def interval_kernel_cov(r, sd_position=0.0, sd_interval=0.0, sd_shift=0.0):
    """Kernel covariance for an ordered tuple of consecutive differences.

    Builds the ``r x r`` covariance matrix

        ``Sigma = sd_position**2 * D D^T + sd_interval**2 * I
                  + sd_shift**2 * ones((r, r))``

    for an ordered attribute whose event tuples are *r* consecutive
    differences (intervals) of ``r + 1`` underlying positions, where
    ``D`` is the ``r x (r + 1)`` first-differencing map.

    This parametrization is meaningful **only for first-differenced
    multisets**. ``sd_position`` builds ``D D^T``, whose off-diagonal
    entries encode the endpoints that neighbouring differences share;
    an undifferenced multiset has no such shared endpoints, so on one
    the term imposes correlations the data do not contain. Nothing
    here inspects the multiset, so passing the result for an
    undifferenced attribute raises no error: the caller is responsible
    for applying it only to interval tuples. For an undifferenced
    attribute, independent per-value noise is what the ordinary scalar
    ``sigma`` already provides, and a matrix covariance is warranted
    only for a common-shift ridge.

    The three terms are three independently specified sources of
    perceptual uncertainty, added because their sources are
    independent:

    - ``sd_position``: uncertainty on the underlying *positions* from
      which the differences are formed. Shared endpoints propagate it
      to the tridiagonal ``sd_position**2 * D D^T`` (``2 sd**2`` on
      the diagonal, ``-sd**2`` on the first off-diagonals): perturbing
      one interior position lengthens one interval and shortens its
      neighbour. This is the exact counterpart of
      ``sigma_space='position'`` in :func:`~mpt.n_tuple_entropy`.
    - ``sd_interval``: uncertainty on each *interval* itself,
      independent across intervals (``sigma_space='interval'``).
    - ``sd_shift``: graded tolerance for a *common shift* of the whole
      tuple, the rank-one ridge ``sd_shift**2 * ones``. A common shift
      of an interval tuple is a transposition when the values are
      pitch intervals and a tempo change when they are log inter-onset
      intervals. As ``sd_shift`` grows the kernel's precision tends to
      the relative-mode projector, so ``is_rel=True`` is the exact
      (infinite-``sd_shift``) limit; a matrix covariance expresses the
      graded counterpart.

    In the time reading, the first two terms are the two levels of the
    Wing & Kristofferson (1973) timing model: motor implementation
    delays attach to onsets (``sd_position``), central timekeeper
    variance attaches to intervals (``sd_interval``).

    The covariance is expressed in whatever coordinates the attribute
    carries: log inter-onset intervals for multiplicative tempo
    tolerance, semitones (or cents) for pitch steps. All three
    arguments are standard deviations in those coordinates; they are
    squared internally.

    Parameters
    ----------
    r : int
        Tuple size (number of consecutive differences); ``r >= 1``.
    sd_position : float, default 0
        Standard deviation of independent noise on each underlying
        position.
    sd_interval : float, default 0
        Standard deviation of independent noise on each interval.
    sd_shift : float, default 0
        Standard deviation of a common shift of the whole tuple.

    Returns
    -------
    (r, r) ndarray
        The kernel covariance, ready to be passed as the ``sigma``
        argument of :func:`~mpt.build_exp_tens`,
        :func:`~mpt.eval_exp_tens`, :func:`~mpt.cos_sim_exp_tens`,
        :func:`~mpt.entropy_exp_tens`, or
        :func:`~mpt.windowed_similarity` for an ordered
        (``is_sym=False``), absolute (``is_rel=False``), non-periodic
        (``is_per=False``) attribute with ``r == K``.

    Raises
    ------
    ValueError
        If ``r < 1``, any argument is negative, or the resulting
        matrix is singular (``sd_shift`` alone is rank one, so at
        least one of ``sd_position`` and ``sd_interval`` must be
        positive).

    References
    ----------
    Wing, A. M., & Kristofferson, A. B. (1973). Response delays and
    the timing of discrete motor responses. *Perception &
    Psychophysics*, 14(1), 5-12.
    """
    r = int(r)
    if r < 1:
        raise ValueError("interval_kernel_cov: r must be a positive integer.")
    if r == 1:
        raise ValueError(
            "interval_kernel_cov: at r = 1 the covariance reduces to a "
            "scalar variance, which is indistinguishable from a scalar "
            "sigma in the MATLAB toolbox; pass the equivalent standard "
            "deviation sqrt(2*sd_position**2 + sd_interval**2 + "
            "sd_shift**2) as the ordinary sigma argument instead."
        )
    for nm, v in (("sd_position", sd_position),
                  ("sd_interval", sd_interval),
                  ("sd_shift", sd_shift)):
        if not np.isfinite(v) or v < 0:
            raise ValueError(
                f"interval_kernel_cov: {nm} must be a finite "
                f"non-negative standard deviation; got {v!r}. For "
                f"exact common-shift invariance use is_rel=True rather "
                f"than an infinite sd_shift."
            )
    # First-differencing map D: r x (r + 1); D D^T is tridiagonal with
    # 2 on the diagonal and -1 on the first off-diagonals.
    ddt = 2.0 * np.eye(r) - np.eye(r, k=1) - np.eye(r, k=-1)
    Sigma = (float(sd_position) ** 2 * ddt
             + float(sd_interval) ** 2 * np.eye(r)
             + float(sd_shift) ** 2 * np.ones((r, r)))
    # Definiteness check (PSD validation always errors): sd_shift alone
    # is rank one for r >= 2.
    try:
        np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "interval_kernel_cov: the resulting covariance is singular. "
            "sd_shift alone is rank one, so at least one of sd_position "
            "and sd_interval must be positive."
        ) from exc
    return Sigma
