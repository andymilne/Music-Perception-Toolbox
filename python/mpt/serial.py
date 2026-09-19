"""Serial-position features for ordered sequences.

:func:`continuity`
    Expected length and signed magnitude of the backward
    same-direction run leading up to each query, under Gaussian
    pitch uncertainty. Consumes the same
    ``(p_attr_diff, w_diff)`` stream that
    :func:`~mpt.difference_events` produces at differencing order 1,
    but reads it as an ordered sequence with a directional gate and
    a break condition rather than aggregating it order-free into a
    tensor.
:func:`kernel_cov`
    Constructor for the matrix-valued kernel covariance of an
    ordered tuple, of values or of consecutive differences, from
    three sources of variance; consumed as the ``sigma`` of the
    tensor functions.

Serial-position weight profiles are built with
:func:`~mpt.weight_events`, whose named and callable profiles apply
decay over any attribute — an event-number attribute, dropped
afterwards, gives the position-indexed case.
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
    difference_events, weight_events
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
#  kernel_cov -- kernel covariance from three sources of variance
# ---------------------------------------------------------------------

def kernel_cov(r, sd_value=0.0, sd_interval=0.0, sd_shift=0.0, *,
               differenced):
    """Kernel covariance for an ordered tuple, from three sources of variance.

    Builds the ``r x r`` covariance matrix of an ordered attribute's
    tuples from three independent sources of perceptual uncertainty,
    each included only when its width is set: independent noise on
    the *values* themselves (onsets, or pitches), ``sd_value``; independent
    noise on the *intervals* between consecutive values,
    ``sd_interval``; and a *common shift* of the whole tuple,
    ``sd_shift``. How each reaches the tuple depends on whether the
    tuple holds the values themselves or their first differences,
    which ``differenced`` (mandatory) declares.

    With ``nabla`` the first-differencing map,
    ``(nabla p)_i = p_{i+1} - p_i``, taken at the size its operand
    requires -- ``(r - 1) x r`` on a tuple of ``r`` values,
    ``r x (r + 1)`` on a tuple of the ``r`` differences of ``r + 1``
    values -- ``nabla+`` its pseudoinverse, and ``J = ones((r, r))``::

        differenced=False:
            Sigma = sd_value**2 * I
                    + sd_interval**2 * nabla+ (nabla+)^T
                    + sd_shift**2 * J
        differenced=True:
            Sigma = sd_value**2 * nabla nabla^T + sd_interval**2 * I
                    + sd_shift**2 * J

    ``nabla+ (nabla+)^T`` equals the centred cumulative sum
    ``P S S^T P``, with ``S`` the ``r x (r - 1)`` cumulative-sum map and
    ``P = I - J/r`` the centring projector, which is how it is built
    here. The two cases are one model: differencing a tuple of ``r``
    values carries the first to the second at tuple size ``r - 1``,
    since ``nabla 1 = 0`` annihilates the ridge and
    ``nabla nabla+ = I``.

    - **Undifferenced values** (``r`` values). Interval noise
      accumulates from one value to the next, a random walk that
      ``P`` centres on the tuple's mean so that no value is
      privileged (``P S S^T P`` is the covariance of the centred
      cumulative sums of ``r - 1`` independent interval errors; ``S`` is
      fixed only up to a base point, and the choices differ by a
      multiple of ``1``, which ``P`` removes). The
      ridge tolerates a common shift of every value: a transposition
      of pitches, a displacement of onsets. As ``sd_shift`` grows the
      kernel's precision tends to the relative-mode projector, so
      ``is_rel=True`` is the exact (infinite-``sd_shift``) limit and
      the ridge its graded counterpart.
    - **Differenced values** (``r`` consecutive differences of
      ``r + 1`` values). Value noise reaches each interval
      through its two endpoints: ``D D^T`` is tridiagonal, ``2`` on the
      diagonal and ``-1`` beside it, since adjacent intervals share an
      endpoint (perturbing one interior value lengthens one interval
      and shortens its neighbour). Interval noise is independent per
      interval. On times the two are the two levels of the Wing &
      Kristofferson (1973) model, motor delay variance on onsets and
      central timekeeper variance on intervals. The ridge adds a
      constant to every interval, seldom the equivalence wanted for
      uneven rhythms, so ``sd_shift`` is usually omitted here; on
      *log*-differenced values it becomes a common factor on the
      intervals (a tempo change, or intervallic augmentation), and is
      wanted again. ``sd_value`` corresponds to
      ``sigma_space='position'`` and ``sd_interval`` to
      ``sigma_space='interval'`` in :func:`~mpt.n_tuple_entropy`.

    The covariance is expressed in whatever coordinates the attribute
    carries: cents or semitones for pitch, seconds for onsets, log
    inter-onset intervals for multiplicative tempo tolerance. All three
    widths are standard deviations in those coordinates; they are
    squared internally.

    Parameters
    ----------
    r : int
        Tuple size; ``r >= 2``.
    sd_value : float, default 0
        Standard deviation of independent noise on each value.
    sd_interval : float, default 0
        Standard deviation of independent noise on each interval
        between consecutive values.
    sd_shift : float, default 0
        Standard deviation of a common shift of the whole tuple.
    differenced : bool, keyword-only, no default
        ``False`` when the tuple holds values, ``True`` when it holds
        their first differences (as :func:`~mpt.difference_events`
        produces). There is no safe default, so it must be given.

    Returns
    -------
    (r, r) ndarray
        The kernel covariance, ready to be passed as the ``sigma``
        argument of :func:`~mpt.build_maet`, :func:`~mpt.eval_maet`,
        :func:`~mpt.sim_maet`, :func:`~mpt.entropy_maet`, or
        :func:`~mpt.windowed_similarity` for an ordered
        (``is_exch=False``), absolute (``is_rel=False``), non-periodic
        (``is_per=False``) attribute with ``r == K``.

    Raises
    ------
    ValueError
        If ``r < 2``, any width is negative or infinite, or the result
        is not positive-definite. On undifferenced values that needs
        ``sd_value > 0``, or ``sd_interval`` and ``sd_shift`` both
        non-zero: the centred walk annihilates ``1`` and the ridge is
        rank one, so neither serves alone. On differenced values it
        needs ``sd_value > 0`` or ``sd_interval > 0``, only the ridge
        alone failing.

    References
    ----------
    Wing, A. M., & Kristofferson, A. B. (1973). Response delays and
    the timing of discrete motor responses. *Perception &
    Psychophysics*, 14(1), 5-12.
    """
    r = int(r)
    if r < 1:
        raise ValueError("kernel_cov: r must be a positive integer.")
    if r == 1:
        raise ValueError(
            "kernel_cov: at r = 1 the covariance reduces to a scalar "
            "variance, which is indistinguishable from a scalar sigma; "
            "pass the equivalent standard deviation as the ordinary "
            "sigma argument instead (sqrt(2*sd_value**2 + "
            "sd_interval**2 + sd_shift**2) if differenced, "
            "sqrt(sd_value**2 + sd_shift**2) if not)."
        )
    if not isinstance(differenced, (bool, np.bool_)):
        raise TypeError(
            "kernel_cov: differenced must be True (the tuple holds first "
            "differences) or False (it holds values)."
        )
    differenced = bool(differenced)
    for nm, v in (("sd_value", sd_value),
                  ("sd_interval", sd_interval),
                  ("sd_shift", sd_shift)):
        if not np.isfinite(v) or v < 0:
            raise ValueError(
                f"kernel_cov: {nm} must be a finite non-negative "
                f"standard deviation; got {v!r}. For exact common-shift "
                f"invariance use is_rel=True rather than an infinite "
                f"sd_shift."
            )
    sp2, si2, ss2 = (float(sd_value) ** 2, float(sd_interval) ** 2,
                     float(sd_shift) ** 2)
    ones = np.ones((r, r))
    if differenced:
        # D D^T: tridiagonal, 2 on the diagonal, -1 on the first
        # off-diagonals.
        ddt = 2.0 * np.eye(r) - np.eye(r, k=1) - np.eye(r, k=-1)
        Sigma = sp2 * ddt + si2 * np.eye(r) + ss2 * ones
        why = ("Positive-definiteness needs sd_value > 0 or "
               "sd_interval > 0; the ridge alone is rank one.")
    else:
        # S: r x (r - 1) cumulative sums (value i carries the first
        # i - 1 interval errors); P centres them on the tuple's mean.
        S = np.tril(np.ones((r, r - 1)), k=-1)
        P = np.eye(r) - ones / r
        pssp = P @ S @ S.T @ P
        Sigma = sp2 * np.eye(r) + si2 * pssp + ss2 * ones
        why = ("Positive-definiteness needs sd_value > 0, or "
               "sd_interval and sd_shift both non-zero: the centred "
               "walk annihilates the common-shift direction and the "
               "ridge is rank one, so neither serves alone.")
    # Definiteness check (PSD validation always errors). An eigenvalue
    # test rather than a Cholesky attempt: the singular cases are exactly
    # singular, and rounding can let a factorization of one through.
    ev = np.linalg.eigvalsh(Sigma)
    if ev[0] <= 1e-12 * max(ev[-1], np.finfo(float).tiny):
        raise ValueError(
            "kernel_cov: the resulting covariance is singular. " + why)
    return Sigma
