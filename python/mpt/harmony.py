"""Harmony and consonance measures.

Spectral entropy, template harmonicity, tensor harmonicity,
sensory roughness, and virtual pitch extraction.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
from scipy.signal import correlate as _sp_correlate

from ._utils import estimate_comp_time, maybe_print_batched_estimate, validate_weights
from ._defaults import _with_dispatch_scope
from .entropy import entropy_exp_tens
from .spectra import add_spectra
from .tensor import _chord_canonical_key, build_exp_tens, eval_exp_tens


# ===================================================================
#  Internal: shared template cross-correlation chord-side compute
# ===================================================================


def _template_xcorr_chord_side(
    chord_p, chord_w, sigma, tmpl_vals, tmpl_norm_sq, margin, step,
    truncation_sigmas, kernel_precision,
):
    """Build the chord density, evaluate it on the chord grid, and
    return the normalised cross-correlation against the pre-evaluated
    template.

    Shared by :func:`template_harmonicity` (which returns ``max`` of the
    profile plus optional Harrison-2020 entropy) and
    :func:`virtual_pitches` (which returns the profile re-packaged as
    ``(vp_p, vp_w)``). Hoisting this into a single helper means the
    build-eval-conv-normalise sequence lives in exactly one place;
    each caller adds only its function-specific postprocessing.

    Parameters
    ----------
    chord_p, chord_w : ndarray
        Chord pitches and weights (already shifted / spectrum-enriched
        by the caller).
    sigma : float
        Gaussian smoothing width (cents).
    tmpl_vals : ndarray
        Pre-evaluated template values on its own grid.
    tmpl_norm_sq : float
        ``sum(tmpl_vals ** 2)`` (caller pre-computes once per batch).
    margin : float
        Grid margin in cents (typically ``4 * sigma``).
    step : float
        Grid spacing in cents (typically 1).
    truncation_sigmas, kernel_precision : forwarded to ``eval_exp_tens``.

    Returns
    -------
    ndarray
        Normalised cross-correlation profile, length
        ``len(chord_vals) + len(tmpl_vals) - 1``.
    """
    chord_dens = build_exp_tens(
        chord_p, chord_w, sigma, 1, False, False, 1200, verbose=False,
    )
    x_chord = np.arange(0, np.max(chord_p) + margin + step, step)
    chord_vals = eval_exp_tens(
        chord_dens, x_chord, verbose=False,
        truncation_sigmas=truncation_sigmas,
        kernel_precision=kernel_precision,
    )

    xcorr = _sp_correlate(chord_vals, tmpl_vals, mode="full", method="auto")
    norm_factor = np.sqrt(float(np.sum(chord_vals ** 2)) * tmpl_norm_sq)
    if norm_factor > 0:
        return xcorr / norm_factor
    return xcorr


# ===================================================================
#  Spectral entropy
# ===================================================================


@_with_dispatch_scope
def spectral_entropy(
    p,
    w=None,
    sigma: float = 12.0,
    *,
    spectrum: list | None = None,
    method: str = "differential",
    base: float = 2.0,
    truncation_sigmas: float | None = None,
    kernel_precision: str | None = None,
    verbose: bool = True,
    **legacy_kwargs,
) -> float | np.ndarray:
    """Spectral entropy of a weighted pitch multiset.

    Returns the entropy of the smoothed composite spectrum of a
    weighted pitch multiset, used as a consonance measure: the greater
    the overlap of partials (after Gaussian smoothing for perceptual
    uncertainty), the lower the entropy. Lower entropy therefore
    indicates greater consonance.

    ``spectral_entropy`` is a thin wrapper around
    :func:`~mpt.entropy_exp_tens` with ``r=1``, ``is_rel=False``,
    ``is_per=False`` (1-D absolute non-periodic density). It applies
    :func:`~mpt.spectra.add_spectra` to enrich the pitches with
    partials (if a ``spectrum`` argument is supplied), shifts the
    lowest pitch to 0, computes appropriate grid bounds where needed,
    and delegates the entropy computation. Four methods are supported:

    - ``method='differential'`` (default): adaptive evaluation of the
      differential entropy ĥ; grid-independent and the principled
      scale-free choice. Lower ĥ → more consonant. Note: adaptive
      convergence (nested-grid doubling to a truncation-sigma-anchored
      tolerance) costs several discrete passes per call --- typically
      10-30× the cost of ``method='normalized'`` at the default
      ``truncation_sigmas`` (≈ 6). Passing ``truncation_sigmas=3``
      loosens the tolerance to ``exp(-9/2) ≈ 1.1e-2`` and brings
      differential to comparable cost to the discrete methods, at the
      price of fifth-decimal drift (consonance ordering is preserved).
      For consonance comparisons across many chords, prefer
      ``'normalized'`` (faster and the method established in the
      consonance literature).
    - ``method='normalized'`` (alias ``'normalised'``): the Pielou-style
      ratio ``H / log_b(N)`` in ``[0, 1]``. Reproduces the values
      reported in Milne et al. (2017) and Smit et al. (2019). Computed
      on an explicit grid of resolution ``n_points_per_dim=1200`` over
      ``[0, max(spec_p) + 4*sigma]``.
    - ``method='shannon'``: raw discrete Shannon entropy
      ``H = -Σ q log_b q`` on the same grid as ``'normalized'``.
    - ``method='renyi2'``: analytical (grid-independent) Rényi-2 /
      collision entropy via the inner-product / Möbius machinery.

    The ``normalize`` kwarg of v2.1 has been removed; pick the
    appropriate ``method`` instead (a migration error is raised if
    ``normalize`` is passed).

    Users needing finer control over the grid resolution should call
    :func:`entropy_exp_tens` directly with a pre-built density and
    their own ``n_points_per_dim`` or ``grid_limit``.

    Accepts two input forms, dispatched on ``p``'s shape:

    - 1-D ``p``: single chord, returns a Python float.
    - 2-D ``P`` (shape ``(M, K)``): batched chords, returns ``(M,)``
      ndarray. NaN-padded rows are accepted; rows with no valid
      pitches return ``np.nan``. Per-row dedup via canonical-form
      chord identity (transposition + permutation symmetry).

    Parameters
    ----------
    p : array-like
        Pitch values in cents (1-D for a single chord; 2-D for a batch,
        rows are chords).
    w : array-like or None
        Weights (``None`` for all ones). If ``p`` is 2-D, ``w`` may be
        ``None``, the same shape as ``P``, or a length-``K`` vector
        broadcast across rows.
    sigma : float
        Gaussian smoothing width in cents (typical: 6-15).
    spectrum : list or None
        Arguments for :func:`~mpt.spectra.add_spectra`.
    method : {'differential', 'normalized', 'shannon', 'renyi2'}
        Entropy variant (default ``'differential'``; ``'normalised'``
        accepted as an alias for ``'normalized'``). See above.
    base : float
        Logarithm base (default 2 = bits).
    verbose : bool
        If True (default), print an upfront time estimate. Scalar mode
        prints a kernel-only ``estimate_comp_time`` estimate; batched
        mode prints an empirical calibration (warm-up plus
        ``min(10, M)`` sampled rows). Suppressed by ``verbose=False``.

    Returns
    -------
    float or np.ndarray
        Spectral entropy (lower = more consonant). Scalar for 1-D
        input, ``(M,)`` ndarray for 2-D input.

    References
    ----------
    Milne, A. J., Bulger, D., & Herff, S. A. (2017). Exploring the
    space of perfectly balanced rhythms and scales. *Journal of
    Mathematics and Music*, 11(2-3), 101-133.
    """
    from .entropy import _canonicalize_method, _NORMALIZE_REMOVED_MSG
    if "normalize" in legacy_kwargs:
        raise TypeError(_NORMALIZE_REMOVED_MSG.format(fn="spectral_entropy"))
    if legacy_kwargs:
        unknown = ", ".join(repr(k) for k in legacy_kwargs)
        raise TypeError(
            f"spectral_entropy: unexpected keyword argument(s): {unknown}"
        )

    method = _canonicalize_method(method)

    p_arr = np.asarray(p, dtype=np.float64)
    if p_arr.ndim == 1:
        return _spectral_entropy_scalar(
            p_arr, w, sigma, spectrum, method, base,
            truncation_sigmas, kernel_precision, verbose,
        )
    if p_arr.ndim == 2:
        return _spectral_entropy_batched(
            p_arr, w, sigma, spectrum, method, base,
            truncation_sigmas, kernel_precision, verbose,
        )
    raise ValueError(
        f"p must be 1-D (single chord) or 2-D (batched, rows are "
        f"chords); got shape {p_arr.shape}."
    )


def _spectral_entropy_scalar(p, w, sigma, spectrum, method, base,
                             truncation_sigmas, kernel_precision, verbose):
    """Single-chord scalar dispatch.

    Prepares ``(spec_p, spec_w)`` (transposition shift + optional
    add_spectra) and delegates to :func:`entropy_exp_tens`. For the
    grid-based methods (``'shannon'`` and ``'normalized'``), the wrapper
    passes explicit non-periodic bounds ``x_min=0``,
    ``x_max=max(spec_p) + 4*sigma`` and ``n_points_per_dim=1200``. For
    ``'differential'`` the span and grid are derived adaptively. For
    ``'renyi2'`` the analytical form is used and no grid is needed.
    """
    p = p.ravel()
    w = validate_weights(w, len(p))
    p = p - np.min(p)

    if spectrum is not None:
        spec_p, spec_w = add_spectra(p, w, *spectrum)
    else:
        spec_p, spec_w = p.copy(), w.copy()

    # Up-front time estimate for grid-based methods only ('shannon',
    # 'normalized'). 'differential' uses an adaptive grid whose final
    # resolution is data-dependent; 'renyi2' is analytical (no grid).
    if method in ("shannon", "normalized"):
        n_grid = 1200  # explicit grid for the discrete methods
        n_pairs = int(len(spec_p)) * n_grid
        estimate_comp_time(n_pairs, 1, "spectral_entropy", verbose)

    return _spectral_entropy_delegate(
        spec_p, spec_w, sigma, method, base,
        truncation_sigmas, kernel_precision,
    )


def _spectral_entropy_delegate(spec_p, spec_w, sigma, method, base,
                               truncation_sigmas, kernel_precision):
    """Delegate the entropy computation to entropy_exp_tens.

    Used by both the scalar path and the batched per-row path.

    For grid-based methods ('shannon', 'normalized'), supplies explicit
    bounds (``x_min=0``, ``x_max=max(spec_p) + 4*sigma``) and an
    explicit ``n_points_per_dim=1200``. For 'differential', the span
    auto-derives from event centres +/- ``truncation_sigmas * sigma``
    and the grid is refined adaptively. For 'renyi2', no grid is
    constructed (analytical inner-product form).
    """
    if method == "renyi2":
        return entropy_exp_tens(
            spec_p, spec_w, sigma, 1, False, False, 1200,
            method="renyi2",
            base=base,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
            verbose=False,
        )

    if method == "differential":
        return entropy_exp_tens(
            spec_p, spec_w, sigma, 1, False, False, 1200,
            method="differential",
            base=base,
            truncation_sigmas=truncation_sigmas,
            kernel_precision=kernel_precision,
            verbose=False,
        )

    # Discrete methods: 'shannon' (raw H) or 'normalized' (H/log_b N).
    # Both share an explicit grid; the method kwarg selects the variant.
    margin = 4 * sigma
    x_max = float(np.max(spec_p)) + margin
    return entropy_exp_tens(
        spec_p, spec_w, sigma, 1, False, False, 1200,
        method=method,
        base=base,
        n_points_per_dim=1200,
        x_min=0.0, x_max=x_max,
        truncation_sigmas=truncation_sigmas,
        kernel_precision=kernel_precision,
        verbose=False,
    )


def _spectral_entropy_batched(P, W, sigma, spectrum, method, base,
                              truncation_sigmas, kernel_precision, verbose):
    """Batched dispatch over rows of a 2-D pitch matrix.

    Returns ``(M,)``. Per-row chord-level dedup of the full
    computation: rows with structurally-identical canonical chords
    (transposition + permutation symmetry) share one cached entropy.
    NaN-padded rows are accepted; rows with no valid pitches
    contribute ``np.nan``.
    """
    M, K = P.shape

    # Weight handling: None, full matrix, or row-broadcast vector.
    use_w = W is not None
    if use_w:
        W_arr = np.asarray(W, dtype=np.float64)
        if W_arr.ndim == 1 and W_arr.size == K:
            W_broadcast = W_arr
            W_full = None
        elif W_arr.shape == P.shape:
            W_broadcast = None
            W_full = W_arr
        else:
            raise ValueError(
                "W must be None, a matrix the same shape as P, or "
                "a length-K vector broadcast across rows."
            )
    else:
        W_broadcast = None
        W_full = None

    out = np.full(M, np.nan)
    result_cache: dict = {}

    # Adaptive progress-print state. Defaults: silent. Up-front time
    # estimate + adaptive countdown: method-agnostic empirical
    # calibration that benefits all four methods (differential most;
    # renyi2 is usually fast enough that show_progress's >= 5 s gate
    # suppresses the countdown automatically).
    prog_stride = 1
    show_progress = False
    if verbose and M > 1:
        n_cal = min(10, M)
        sample_idx = np.unique(np.linspace(0, M - 1, n_cal).astype(int))

        warmup_done = False
        for s_idx in sample_idx:
            p_row_s = P[s_idx]
            mask_s = ~np.isnan(p_row_s)
            p_valid_s = p_row_s[mask_s]
            if len(p_valid_s) < 1:
                continue
            if W_full is not None:
                w_valid_s = W_full[s_idx, mask_s]
            elif W_broadcast is not None:
                w_valid_s = W_broadcast[mask_s]
            else:
                w_valid_s = None
            _spectral_entropy_scalar(
                p_valid_s, w_valid_s, sigma, spectrum, method, base,
                truncation_sigmas, kernel_precision,
                verbose=False,
            )
            warmup_done = True
            break

        if warmup_done:
            t_cal_start = time.perf_counter()
            n_valid_cal = 0
            for s_idx in sample_idx:
                p_row_s = P[s_idx]
                mask_s = ~np.isnan(p_row_s)
                p_valid_s = p_row_s[mask_s]
                if len(p_valid_s) < 1:
                    continue
                if W_full is not None:
                    w_valid_s = W_full[s_idx, mask_s]
                elif W_broadcast is not None:
                    w_valid_s = W_broadcast[mask_s]
                else:
                    w_valid_s = None
                _spectral_entropy_scalar(
                    p_valid_s, w_valid_s, sigma, spectrum, method,
                    base, truncation_sigmas, kernel_precision,
                    verbose=False,
                )
                n_valid_cal += 1
            if n_valid_cal > 0:
                from ._utils import progress_stride
                t_cal_total = time.perf_counter() - t_cal_start
                t_per_row = t_cal_total / n_valid_cal
                est_total = t_cal_total + t_per_row * M
                maybe_print_batched_estimate(
                    "spectral_entropy", M, est_total,
                )
                prog_stride = progress_stride(t_per_row)
                show_progress = est_total >= 5

    for i in range(M):
        p_row = P[i]
        mask = ~np.isnan(p_row)
        p_valid = p_row[mask]
        if len(p_valid) < 1:
            continue
        if W_full is not None:
            w_valid = W_full[i, mask]
        elif W_broadcast is not None:
            w_valid = W_broadcast[mask]
        else:
            w_valid = None

        # Canonical key: spectral_entropy transposes internally
        # (p -= min), so transposition is part of the symmetry. r=1,
        # is_rel=True (transposition-invariant after the internal
        # shift), is_per=False.
        key, _, _ = _chord_canonical_key(
            p_valid, w_valid,
            sigma=sigma, r=1, is_rel=True, is_per=False, period=1200.0,
        )

        if key in result_cache:
            out[i] = result_cache[key]
        else:
            h = _spectral_entropy_scalar(
                p_valid, w_valid, sigma, spectrum, method, base,
                truncation_sigmas, kernel_precision, verbose=False,
            )
            result_cache[key] = h
            out[i] = h

        if verbose and show_progress \
                and ((i + 1) % prog_stride == 0 or i == M - 1):
            print(f"  {i + 1} / {M} rows computed.")

    return out


# ===================================================================
#  Template harmonicity
# ===================================================================


@_with_dispatch_scope
def template_harmonicity(
    p,
    w=None,
    sigma: float = 12.0,
    *,
    spectrum: list | None = None,
    chord_spectrum: list | None = None,
    normalize: bool = True,
    base: float = 2.0,
    resolution: float = 1.0,
    truncation_sigmas: float | None = None,
    kernel_precision: str | None = None,
    verbose: bool = True,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Harmonicity via template cross-correlation.

    Measures the harmonicity of a weighted pitch multiset by
    cross-correlating its spectral expectation tensor with a harmonic
    template (a single complex tone). Two complementary measures:

    - *h_max*: maximum normalised cross-correlation (Milne, 2013).
      Cosine similarity at the best-matching transposition.
    - *h_entropy*: Shannon entropy of the cross-correlation treated
      as a probability distribution (Harrison & Pearce, 2020).

    Accepts two input forms, dispatched on ``p``'s shape:

    - 1-D ``p``: single chord, returns ``(h_max, h_entropy)``.
    - 2-D ``P`` (shape ``(M, K)``): batched chords, returns
      ``(h_max_arr, h_entropy_arr)`` with each of shape ``(M,)``.
      Rows may use NaN-padding; rows with no valid pitches return
      ``NaN`` in both outputs.

    Parameters
    ----------
    p : array-like
        Pitch values in cents (1-D for a single chord; 2-D for a batch).
    w : array-like or None
        Weights, matching ``p``'s shape (``None`` for all ones).
    sigma : float
        Gaussian smoothing width in cents (typical: 9–15).
    spectrum : list or None
        Arguments for :func:`~mpt.spectra.add_spectra` for the
        harmonic template.
    chord_spectrum : list or None
        Arguments for :func:`~mpt.spectra.add_spectra` for the
        chord. ``None`` = use pitches as given.
    normalize : bool
        If True (default), normalise entropy to [0, 1].
    base : float
        Logarithm base (default 2).
    resolution : float
        Grid spacing in cents (default 1).

    Returns
    -------
    h_max : float or np.ndarray
        Maximum cross-correlation in [0, 1]; scalar in single-chord
        mode, ``(M,)`` in batched mode.
    h_entropy : float or np.ndarray
        Entropy of the cross-correlation; scalar or ``(M,)``.

    References
    ----------
    Milne, A. J. (2013). *A computational model of the cognition
    of tonality*. PhD thesis, The Open University.

    Harrison, P. M. C. & Pearce, M. T. (2020). Simultaneous
    consonance in music perception and composition. *Psychological
    Review*, 127(2), 216–244.
    """
    if spectrum is None:
        spectrum = ["harmonic", 36, "powerlaw", 1]

    p_arr = np.asarray(p, dtype=np.float64)
    if p_arr.ndim == 1:
        return _template_harmonicity_scalar(
            p_arr, w, sigma, spectrum, chord_spectrum,
            normalize, base, resolution,
            truncation_sigmas, kernel_precision, verbose,
        )
    if p_arr.ndim == 2:
        return _template_harmonicity_batched(
            p_arr, w, sigma, spectrum, chord_spectrum,
            normalize, base, resolution,
            truncation_sigmas, kernel_precision, verbose,
        )
    raise ValueError(
        f"p must be 1-D (single chord) or 2-D (batched, rows are "
        f"chords); got shape {p_arr.shape}."
    )


def _template_harmonicity_chord_only(
    chord_p, chord_w, sigma, tmpl_vals, tmpl_norm_sq, margin,
    resolution, normalize, base,
    truncation_sigmas, kernel_precision,
):
    """Compute (h_max, h_entropy) for one chord, given pre-built
    template evaluation and its norm-square.

    Hoisted from :func:`_template_harmonicity_scalar` so the batched
    dispatch can build the harmonic template once for the whole
    batch — the template depends only on (spectrum, sigma, resolution),
    not on the chord. Callers are responsible for applying the
    ``chord_spectrum`` (if any) to ``chord_p, chord_w`` before
    invocation; this function performs only the chord-side
    evaluation, cross-correlation, and entropy computation.

    The build-eval-conv-normalise core is shared with
    :func:`virtual_pitches` via :func:`_template_xcorr_chord_side`;
    this wrapper adds template-harmonicity-specific postprocessing
    (max plus optional Harrison-2020 entropy of the profile).
    """
    xcorr_norm = _template_xcorr_chord_side(
        chord_p, chord_w, sigma, tmpl_vals, tmpl_norm_sq, margin,
        resolution, truncation_sigmas, kernel_precision,
    )

    h_max = float(np.max(xcorr_norm))

    # Harrison-2020 entropy of the profile. (Treated as a probability
    # distribution; this is a discrete-Shannon computation on a vector,
    # not on an expectation-tensor density, so it does not delegate to
    # entropy_exp_tens.)
    q = xcorr_norm.copy()
    N = len(q)
    total = np.sum(q)
    if total > 0:
        q = q / total
    q = q[q > 0]
    h_entropy = float(-np.sum(q * np.log(q) / np.log(base)))
    if normalize:
        h_entropy /= np.log(N) / np.log(base)

    return h_max, h_entropy


def _template_harmonicity_scalar(p, w, sigma, spectrum, chord_spectrum,
                                  normalize, base, resolution,
                                  truncation_sigmas, kernel_precision, verbose):
    """Single-chord scalar dispatch."""
    p = p.ravel()
    w = validate_weights(w, len(p))
    p = p - np.min(p)

    # Build template
    tmpl_p, tmpl_w = add_spectra(np.array([0.0]), np.array([1.0]), *spectrum)

    # Build chord spectrum
    if chord_spectrum is not None:
        chord_p, chord_w = add_spectra(p, w, *chord_spectrum)
    else:
        chord_p, chord_w = p.copy(), w.copy()

    margin = 4 * sigma
    x_tmpl = np.arange(0, np.max(tmpl_p) + margin + resolution, resolution)
    x_chord_len = len(np.arange(
        0, np.max(chord_p) + margin + resolution, resolution,
    ))

    # Time estimate (kernel cost only; convolve and other overheads
    # not included, so this is a lower bound). Pair count is the sum
    # of the two eval_exp_tens workloads. dim = 1 since both densities
    # use r = 1, is_rel = False.
    n_pairs = (
        int(len(chord_p)) * x_chord_len
        + int(len(tmpl_p)) * int(len(x_tmpl))
    )
    estimate_comp_time(n_pairs, 1, "template_harmonicity", verbose)

    tmpl_dens = build_exp_tens(
        tmpl_p, tmpl_w, sigma, 1, False, False, 1200, verbose=False,
    )
    tmpl_vals = eval_exp_tens(
        tmpl_dens, x_tmpl, verbose=False,
        truncation_sigmas=truncation_sigmas,
        kernel_precision=kernel_precision,
    )
    tmpl_norm_sq = float(np.sum(tmpl_vals ** 2))

    return _template_harmonicity_chord_only(
        chord_p, chord_w, sigma, tmpl_vals, tmpl_norm_sq, margin,
        resolution, normalize, base,
        truncation_sigmas, kernel_precision,
    )


def _template_harmonicity_batched(P, W, sigma, spectrum, chord_spectrum,
                                   normalize, base, resolution,
                                   truncation_sigmas, kernel_precision, verbose):
    """Batched dispatch over rows of a 2-D pitch matrix.

    The harmonic template is built once for the whole batch (it's
    independent of the chord). Per-row chord-level dedup of the
    full computation: structurally-identical canonical chords share
    one cached ``(h_max, h_entropy)`` result.
    """
    M, K = P.shape
    use_w = W is not None
    if use_w:
        W = np.asarray(W, dtype=np.float64)
        if W.shape != P.shape:
            raise ValueError("W must be the same shape as P.")

    h_max_out = np.full(M, np.nan)
    h_ent_out = np.full(M, np.nan)

    # Template (independent of chord) — built once.
    tmpl_p, tmpl_w = add_spectra(np.array([0.0]), np.array([1.0]), *spectrum)
    tmpl_dens = build_exp_tens(
        tmpl_p, tmpl_w, sigma, 1, False, False, 1200, verbose=False,
    )
    margin = 4 * sigma
    x_tmpl = np.arange(0, np.max(tmpl_p) + margin + resolution, resolution)
    tmpl_vals = eval_exp_tens(tmpl_dens, x_tmpl, verbose=False,
                              truncation_sigmas=truncation_sigmas,
                              kernel_precision=kernel_precision)
    tmpl_norm_sq = float(np.sum(tmpl_vals ** 2))

    # Up-front time estimate (printed once for the whole batch). The
    # kernel-only nPairs-based estimate (as used by estimateCompTime in
    # scalar paths and in evalExpTens) underestimates the actual cost
    # of templateHarmonicity batched runs by 3-5x because it omits
    # convolve, addSpectra, and per-row Python-loop overheads. So we
    # instead run a small empirical calibration: pick K rows spaced
    # uniformly across the input, time them via the scalar code path
    # (results discarded), and extrapolate. K is bounded so the
    # calibration cost stays small relative to a non-trivial batch.
    #
    # A single warm-up call is run before timing starts so first-call
    # overheads (numpy dispatch caching, lazily-populated module-level
    # caches) don't bias the K-sample mean upward. The warm-up's wall
    # time is not part of the printed estimate, but the estimate does
    # add the K-sample calibration time itself, since the caller pays
    # for it.
    # Adaptive progress-print state. Defaults: silent.
    prog_stride = 1
    show_progress = False
    if verbose and M > 1:
        n_cal = min(10, M)
        sample_idx = np.unique(np.linspace(0, M - 1, n_cal).astype(int))

        # Warm-up: run the first valid sample once, untimed, to absorb
        # any first-call overhead. Result discarded. Calibration uses
        # the same _template_harmonicity_chord_only path the main loop
        # uses, so the timed work matches what's actually paid per row;
        # routing through _template_harmonicity_scalar would re-include
        # a template rebuild that the main loop does not pay.
        warmup_done = False
        for s_idx in sample_idx:
            p_row_s = P[s_idx]
            mask_s = ~np.isnan(p_row_s)
            p_valid_s = p_row_s[mask_s]
            if len(p_valid_s) < 1:
                continue
            w_valid_s = W[s_idx, mask_s] if use_w \
                else np.ones_like(p_valid_s)
            p_shifted_s = p_valid_s - np.min(p_valid_s)
            if chord_spectrum is not None:
                chord_p_s, chord_w_s = add_spectra(
                    p_shifted_s, w_valid_s, *chord_spectrum,
                )
            else:
                chord_p_s, chord_w_s = p_shifted_s.copy(), w_valid_s.copy()
            _template_harmonicity_chord_only(
                chord_p_s, chord_w_s, sigma, tmpl_vals, tmpl_norm_sq,
                margin, resolution, normalize, base,
                truncation_sigmas, kernel_precision,
            )
            warmup_done = True
            break

        if warmup_done:
            t_cal_start = time.perf_counter()
            n_valid_cal = 0
            for s_idx in sample_idx:
                p_row_s = P[s_idx]
                mask_s = ~np.isnan(p_row_s)
                p_valid_s = p_row_s[mask_s]
                if len(p_valid_s) < 1:
                    continue
                w_valid_s = W[s_idx, mask_s] if use_w \
                    else np.ones_like(p_valid_s)
                p_shifted_s = p_valid_s - np.min(p_valid_s)
                if chord_spectrum is not None:
                    chord_p_s, chord_w_s = add_spectra(
                        p_shifted_s, w_valid_s, *chord_spectrum,
                    )
                else:
                    chord_p_s, chord_w_s = \
                        p_shifted_s.copy(), w_valid_s.copy()
                _template_harmonicity_chord_only(
                    chord_p_s, chord_w_s, sigma, tmpl_vals, tmpl_norm_sq,
                    margin, resolution, normalize, base,
                    truncation_sigmas, kernel_precision,
                )
                n_valid_cal += 1
            if n_valid_cal > 0:
                from ._utils import progress_stride
                t_cal_total = time.perf_counter() - t_cal_start
                t_per_row = t_cal_total / n_valid_cal
                # Total estimate covers the calibration we just did (which the
                # caller is already paying for) plus the M-row main loop.
                est_total = t_cal_total + t_per_row * M
                maybe_print_batched_estimate(

                    "template_harmonicity", M, est_total,

                )
                prog_stride = progress_stride(t_per_row)
                show_progress = est_total >= 5

    result_cache: dict = {}
    for i in range(M):
        p_row = P[i]
        mask = ~np.isnan(p_row)
        p_valid = p_row[mask]
        if len(p_valid) < 1:
            continue
        w_valid = W[i, mask] if use_w else np.ones_like(p_valid)

        # Canonical key. Template-harmonicity is invariant under joint
        # transposition (the function shifts so min = 0 anyway), so we
        # use a relative, non-periodic canonical form.
        key, p_canon, w_canon = _chord_canonical_key(
            p_valid, w_valid,
            sigma=sigma, r=1, is_rel=True, is_per=False, period=1200.0,
        )
        if key in result_cache:
            h_max_out[i], h_ent_out[i] = result_cache[key]
        else:
            # Compute (h_max, h_entropy) for this canonical chord.
            p_shifted = p_canon - np.min(p_canon)
            if chord_spectrum is not None:
                chord_p, chord_w = add_spectra(p_shifted, w_canon, *chord_spectrum)
            else:
                chord_p, chord_w = p_shifted.copy(), w_canon.copy()

            h_max, h_ent = _template_harmonicity_chord_only(
                chord_p, chord_w, sigma, tmpl_vals, tmpl_norm_sq, margin,
                resolution, normalize, base,
                truncation_sigmas, kernel_precision,
            )

            result_cache[key] = (h_max, h_ent)
            h_max_out[i], h_ent_out[i] = h_max, h_ent

        if verbose and show_progress \
                and ((i + 1) % prog_stride == 0 or i == M - 1):
            print(f"  {i + 1} / {M} rows computed.")

    return h_max_out, h_ent_out


# ===================================================================
#  Tensor harmonicity
# ===================================================================


@_with_dispatch_scope
def tensor_harmonicity(
    p,
    w=None,
    sigma: float = 12.0,
    *,
    spectrum: list | None = None,
    duplicate: int = 0,
    normalize: str = "none",
    truncation_sigmas: float | None = None,
    kernel_precision: str | None = None,
    verbose: bool = True,
) -> float | np.ndarray:
    """Harmonicity via expectation tensor lookup.

    Measures the harmonicity of a weighted pitch multiset by
    evaluating the relative r-ad expectation tensor of a harmonic
    series at the multiset's interval vector. A high density at the
    chord's intervals indicates those intervals are likely to
    co-occur in a harmonic series.

    Accepts two input forms, dispatched on ``p``'s shape:

    - 1-D ``p`` (length K ≥ 2): single chord, returns scalar.
    - 2-D ``P`` (shape ``(M, K)``): batched chords, returns ``(M,)``.
      Rows may use NaN-padding for variable cardinality. Rows with
      fewer than 2 valid pitches return ``NaN``.

    Parameters
    ----------
    p : array-like
        Pitch values in cents (1-D for a single chord; 2-D for a batch).
    w : array-like or None
        Weights, matching ``p``'s shape (``None`` for all ones).
    sigma : float
        Gaussian smoothing width in cents (typical: 9–15).
    spectrum : list or None
        Arguments for :func:`~mpt.spectra.add_spectra` for the
        harmonic template.
    duplicate : int
        Number of template copies (0 = auto = chord cardinality;
        in batched mode with ``duplicate=0``, each row uses its own
        valid-pitch count).
    normalize : str
        ``'none'`` (default), ``'gaussian'``, or ``'pdf'``.

    Returns
    -------
    float or np.ndarray
        Scalar in single-chord mode; ``(M,)`` ndarray in batched mode.

    Notes
    -----
    Internal computation is delegated to :func:`~mpt.eval_exp_tens`,
    whose dispatcher chooses between the centres-array path and the
    Möbius point evaluator (see :func:`mpt._tensor.dispatch._select_ma_eval`).
    The harmonic template has ``K = duplicate × n_partials`` events, so
    with the default 64-partial template even a triad reaches
    ``K ≥ 192`` and the centres array (``(r-1, K!/(K-r)!)`` floats,
    order ``10⁹`` for a 4-pitch chord) is far too large to materialise;
    the dispatcher's working-set memory guard therefore routes such
    calls to the Möbius evaluator, which evaluates the relative tensor
    at the chord's interval vector without building the centres array.

    The Möbius route is *not* grid-free in relative mode: it integrates
    the absolute tensor over the translation coordinate on a u-grid of
    ``N_u`` points (the relative tensor is a translation marginal, and
    the alternating partition sum only factorises across slots at fixed
    ``u``). Its per-query cost is ``B_r · r · K · N_u`` on the direct
    strategy, dropping to ``B_r · r · N_u`` (plus an amortised
    ``O(Σ_m N_fine_m · K)`` tabulation) when the evaluator's factored
    strategy engages — an exact per-block factorisation of the
    non-periodic integrand into read-backs of ``r`` precomputed
    smoothed event distributions, chosen by an internal cost gate for
    batched workloads, with read-back accuracy tied to
    ``truncation_sigmas``. This unblocks ``K > 3`` chord cardinality
    where the centres path was infeasible. Small-template calls (small ``K``, e.g. a low
    ``duplicate`` with a short custom spectrum) keep the centres path,
    which is cheaper there — it carries no ``N_u`` factor and its
    ``n_j`` is small.

    References
    ----------
    Smit, E. A., Milne, A. J., Dean, R. T., & Weidemann, G.
    (2019). Perception of affect in unfamiliar musical chords.
    *PLOS ONE*, 14(6), e0218570.
    """
    if spectrum is None:
        spectrum = ["harmonic", 64, "powerlaw", 1]

    p_arr = np.asarray(p, dtype=np.float64)
    if p_arr.ndim == 1:
        return _tensor_harmonicity_scalar(
            p_arr, w, sigma, spectrum, duplicate, normalize,
            truncation_sigmas, kernel_precision, verbose,
        )
    if p_arr.ndim == 2:
        return _tensor_harmonicity_batched(
            p_arr, w, sigma, spectrum, duplicate, normalize,
            truncation_sigmas, kernel_precision, verbose,
        )
    raise ValueError(
        f"p must be 1-D (single chord) or 2-D (batched, rows are "
        f"chords); got shape {p_arr.shape}."
    )


def _tensor_harmonicity_scalar(p, w, sigma, spectrum, duplicate, normalize,
                               truncation_sigmas, kernel_precision, verbose):
    """Single-chord scalar dispatch.

    Routes through :func:`eval_exp_tens` so the centres-vs-Möbius
    choice is made by the cost-model dispatcher inside
    ``eval_exp_tens`` rather than hard-coded here. This lets the
    helper-accelerated centres path apply at typical regimes.
    """
    n_pitches = len(p)
    if n_pitches < 2:
        raise ValueError(f"At least 2 pitches required (got {n_pitches}).")

    dup = duplicate if duplicate > 0 else n_pitches

    # Build the harmonic template's (p, w) source.
    tmpl_p, tmpl_w = add_spectra(
        np.zeros(dup), np.ones(dup), *spectrum
    )

    p_sorted = np.sort(p)
    intervals = p_sorted[1:] - p_sorted[0]  # (n_pitches - 1,)
    x_query = intervals.reshape(-1, 1)

    if verbose:
        print(
            f"tensor_harmonicity: eval at K = {dup}, r = {n_pitches}, "
            f"sigma = {sigma:g}."
        )

    h = _tensor_harmonicity_via_eval(
        tmpl_p, tmpl_w, sigma, n_pitches, x_query, normalize,
        truncation_sigmas, kernel_precision,
    )
    return float(h[0])


def _tensor_harmonicity_via_eval(
    tmpl_p: np.ndarray,
    tmpl_w: np.ndarray,
    sigma: float,
    r: int,
    x_query: np.ndarray,
    normalize: str,
    truncation_sigmas: float | None,
    kernel_precision: str | None,
) -> np.ndarray:
    """Evaluate the rel-mode template tensor at *x_query* by building
    the template density and routing through :func:`eval_exp_tens`.

    The centres-vs-Möbius choice is made by the cost-model dispatcher
    inside ``eval_exp_tens``; this wrapper no longer hard-codes a
    routing choice.

    Normalisation note: ``tensor_harmonicity``'s 'pdf' divides the
    gaussian-normalised value by ``sum(tmpl_w)`` — sum of single-
    partial weights — to preserve v2.0/v2.1 numerical convention.
    This differs from :func:`eval_exp_tens`'s own 'pdf' (which
    divides by ``sum(wJ)``, the sum of r-tuple weight products);
    the difference is a factor of ``(K-1)·(K-2)·...·(K-r+1)`` for
    an all-ones template. We therefore evaluate at 'none' below
    and apply the normalisation ourselves.
    """
    from .tensor import build_exp_tens, eval_exp_tens

    dens = build_exp_tens(
        np.asarray(tmpl_p).ravel(),
        np.asarray(tmpl_w).ravel(),
        sigma, r, True, False, 0.0,
        verbose=False,
    )

    vals = eval_exp_tens(
        dens, x_query, 'none',
        truncation_sigmas=truncation_sigmas,
        kernel_precision=kernel_precision,
        verbose=False,
    )

    if normalize == "none":
        return vals

    from ._tensor.dispatch import _quadratic_form_det, _gaussian_mass_const
    dim = r - 1
    det_m = _quadratic_form_det(r, 0, True)   # flat, relative: 1 / r
    gauss_const = 1.0 / _gaussian_mass_const(sigma, dim, det_m)
    vals = vals * gauss_const

    if normalize == "pdf":
        sum_w = float(np.sum(tmpl_w))
        if sum_w > 0:
            vals = vals / sum_w
        else:
            warnings.warn(
                "Sum of template weights is zero; cannot normalize to pdf."
            )
    elif normalize != "gaussian":
        raise ValueError(
            f"normalize must be 'none', 'gaussian', or 'pdf'; "
            f"got {normalize!r}."
        )

    return vals


def _tensor_harmonicity_batched(P, W, sigma, spectrum, duplicate, normalize,
                                truncation_sigmas, kernel_precision, verbose):
    """Batched dispatch over rows of a 2-D pitch matrix.

    Groups rows by (effective n_p, dup), deduplicates canonical chord
    intervals within each group, and issues a single batched call to
    :func:`_tensor_harmonicity_via_eval` per group. The cost-model
    dispatcher inside ``eval_exp_tens`` then chooses centres vs the Möbius method
    for that batched query matrix.
    """
    M, K = P.shape
    use_w = W is not None
    if use_w:
        W = np.asarray(W, dtype=np.float64)
        if W.shape != P.shape:
            raise ValueError("W must be the same shape as P.")

    out = np.full(M, np.nan)

    # Pass 1: per-row metadata. Rows with fewer than 2 valid pitches
    # keep out[i] = NaN and are excluded from grouping below.
    row_n_p = np.zeros(M, dtype=np.int64)
    row_dup = np.zeros(M, dtype=np.int64)
    row_keys: list = [None] * M
    row_intervals: list = [None] * M
    large_dup_warned = False
    for i in range(M):
        p_row = P[i]
        mask = ~np.isnan(p_row)
        p_valid = p_row[mask]
        n_p = len(p_valid)
        if n_p < 2:
            continue
        dup = duplicate if duplicate > 0 else n_p
        if dup > 3 and not large_dup_warned:
            warnings.warn(
                f"duplicate = {dup}: computation time grows rapidly "
                f"with duplication. Consider reducing to 3 or fewer.",
                stacklevel=2,
            )
            large_dup_warned = True

        # Canonical key for this chord (relative, non-periodic).
        key, p_canon, _ = _chord_canonical_key(
            p_valid,
            W[i, mask] if use_w else None,
            sigma=sigma, r=n_p, is_rel=True, is_per=False, period=1200.0,
        )

        row_n_p[i] = n_p
        row_dup[i] = dup
        row_keys[i] = (key, dup, normalize)
        row_intervals[i] = p_canon[1:] - p_canon[0]

    # Pass 2: group by (n_p, dup); within each group dedup canonical
    # chords; one batched Möbius-method call per group; distribute back.
    valid_rows = np.flatnonzero(row_n_p > 0)
    if valid_rows.size == 0:
        return out

    group_tags = [(int(row_n_p[i]), int(row_dup[i])) for i in valid_rows]
    unique_groups = sorted(set(group_tags))

    if verbose:
        # Gate the groups print on a row-count threshold matching the
        # `maybe_print_batched_estimate` "silent for fast" semantics
        # used by the other batched functions. The threshold is
        # deliberately rough: at >~100 rows the batched Möbius-method call is
        # likely to exceed the 10s estimate-print threshold; tiny
        # batches stay silent.
        n_valid = int(valid_rows.size)
        if n_valid >= 100:
            n_groups = len(unique_groups)
            if n_groups == 1:
                print(
                    f"tensor_harmonicity: {n_valid} valid rows in 1 "
                    f"(n_p, dup) group."
                )
            else:
                print(
                    f"tensor_harmonicity: {n_valid} valid rows across "
                    f"{n_groups} (n_p, dup) groups."
                )

    for n_p, dup in unique_groups:
        rows_in_group = [int(i) for i in valid_rows
                         if row_n_p[i] == n_p and row_dup[i] == dup]
        r = n_p

        # Dedup canonical chords within the group.
        key_to_idx: dict = {}
        unique_intervals: list = []
        row_to_unique_idx = []
        for i in rows_in_group:
            ck = row_keys[i]
            if ck in key_to_idx:
                row_to_unique_idx.append(key_to_idx[ck])
            else:
                idx = len(unique_intervals)
                key_to_idx[ck] = idx
                unique_intervals.append(row_intervals[i])
                row_to_unique_idx.append(idx)

        n_unique = len(unique_intervals)
        # Stack into (r-1, n_unique) query matrix.
        query_mat = np.empty((r - 1, n_unique), dtype=np.float64)
        for u_idx, ivs in enumerate(unique_intervals):
            query_mat[:, u_idx] = ivs

        # Build harmonic template once per group, then ONE batched
        # call to _tensor_harmonicity_via_eval (which routes through
        # eval_exp_tens; its dispatcher chooses centres vs the Möbius method).
        # Forwards truncation_sigmas / kernel_precision so the batched
        # path picks up the same speed/accuracy controls as scalar.
        tmpl_p, tmpl_w = add_spectra(
            np.zeros(dup), np.ones(dup), *spectrum,
        )
        vals = _tensor_harmonicity_via_eval(
            tmpl_p, tmpl_w, sigma, r, query_mat, normalize,
            truncation_sigmas, kernel_precision,
        )

        # Distribute back to rows.
        for ii, i in enumerate(rows_in_group):
            out[i] = float(vals[row_to_unique_idx[ii]])

    return out


# ===================================================================
#  Virtual pitches
# ===================================================================


@_with_dispatch_scope
def virtual_pitches(
    p,
    w=None,
    sigma: float = 12.0,
    *,
    spectrum: list | None = None,
    chord_spectrum: list | None = None,
    resolution: float = 1.0,
    truncation_sigmas: float | None = None,
    kernel_precision: str | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray | list[np.ndarray], np.ndarray | list[np.ndarray]]:
    """Virtual pitch salience profile via template cross-correlation.

    Computes the virtual pitch (fundamental) salience profile for a
    weighted pitch multiset by cross-correlating its spectral
    expectation tensor with a harmonic template. Peaks indicate
    strong virtual pitches — candidate fundamentals well-supported
    by the input spectrum.

    Accepts two input forms, dispatched on ``p``'s shape:

    - 1-D ``p``: single chord, returns ``(vp_p, vp_w)``.
    - 2-D ``P`` (shape ``(M, K)``): batched chords, returns
      ``(vp_p_list, vp_w_list)`` where each is a length-``M`` list of
      1-D arrays. Profile lengths can differ between rows because the
      cross-correlation grid extends to ``max(chord_p) + margin``,
      which depends on each chord. NaN-padded rows are accepted; rows
      with no valid pitches return empty arrays in both lists.

    Parameters
    ----------
    p : array-like
        Pitch values in cents (1-D for a single chord; 2-D for a batch).
    w : array-like or None
        Weights, matching ``p``'s shape (``None`` for all ones).
    sigma : float
        Gaussian smoothing width in cents.
    spectrum : list or None
        Arguments for :func:`~mpt.spectra.add_spectra` for the
        harmonic template.
    chord_spectrum : list or None
        Arguments for :func:`~mpt.spectra.add_spectra` for the
        chord. ``None`` = use pitches as given.
    resolution : float
        Grid spacing in cents (default 1).

    Returns
    -------
    vp_p : np.ndarray or list of np.ndarray
        Candidate pitch values in cents. Single ``ndarray`` in
        single-chord mode; list of ``M`` ``ndarray``s in batched mode
        (one per row, possibly differing in length).
    vp_w : np.ndarray or list of np.ndarray
        Virtual pitch weights (normalised cross-correlation).

    References
    ----------
    Milne, A. J. (2013). *A computational model of the cognition
    of tonality*. PhD thesis, The Open University.
    """
    if spectrum is None:
        spectrum = ["harmonic", 36, "powerlaw", 1]

    p_arr = np.asarray(p, dtype=np.float64)
    if p_arr.ndim == 1:
        return _virtual_pitches_scalar(
            p_arr, w, sigma, spectrum, chord_spectrum, resolution,
            truncation_sigmas, kernel_precision, verbose,
        )
    if p_arr.ndim == 2:
        return _virtual_pitches_batched(
            p_arr, w, sigma, spectrum, chord_spectrum, resolution,
            truncation_sigmas, kernel_precision, verbose,
        )
    raise ValueError(
        f"p must be 1-D (single chord) or 2-D (batched, rows are "
        f"chords); got shape {p_arr.shape}."
    )


def _virtual_pitches_chord_only(
    chord_p, chord_w, sigma, tmpl_vals, tmpl_norm_sq, margin, step,
    truncation_sigmas, kernel_precision,
):
    """Chord-side normalised cross-correlation, returning the profile
    and its length.

    Returns ``(vp_w, n_xcorr)`` — the offset-independent profile and
    its length. The caller reconstructs ``vp_p = (np.arange(n_xcorr) -
    (n_tmpl - 1)) * step + p_offset`` in the input coordinate system;
    that arithmetic is row-dependent and so is not part of what gets
    cached when this helper is called from the batched path.

    Hoisted from :func:`_virtual_pitches_scalar` so the batched
    dispatch can build the template once for the whole batch and
    cache per-canonical-chord results in the offset-independent
    representation.

    The build-eval-conv-normalise core is shared with
    :func:`template_harmonicity` via :func:`_template_xcorr_chord_side`;
    this wrapper exists only to extract ``n_xcorr`` alongside the
    profile (the caller needs the length to reconstruct ``vp_p``).
    """
    vp_w = _template_xcorr_chord_side(
        chord_p, chord_w, sigma, tmpl_vals, tmpl_norm_sq, margin,
        step, truncation_sigmas, kernel_precision,
    )
    return vp_w, len(vp_w)


def _virtual_pitches_scalar(p, w, sigma, spectrum, chord_spectrum, resolution,
                            truncation_sigmas, kernel_precision, verbose):
    """Single-chord scalar dispatch."""
    p = p.ravel()
    w = validate_weights(w, len(p))

    p_offset = float(np.min(p))
    p = p - p_offset

    tmpl_p, tmpl_w = add_spectra(np.array([0.0]), np.array([1.0]), *spectrum)

    if chord_spectrum is not None:
        chord_p, chord_w = add_spectra(p, w, *chord_spectrum)
    else:
        chord_p, chord_w = p.copy(), w.copy()

    margin = 4 * sigma
    step = resolution
    x_tmpl = np.arange(0, np.max(tmpl_p) + margin + step, step)
    x_chord_len = len(np.arange(
        0, np.max(chord_p) + margin + step, step,
    ))

    # Time estimate (kernel cost only; convolve and other overheads
    # not included, so this is a lower bound). Pair count is the sum
    # of the two eval_exp_tens workloads. dim = 1 since both densities
    # use r = 1, is_rel = False.
    n_pairs = (
        int(len(chord_p)) * x_chord_len
        + int(len(tmpl_p)) * int(len(x_tmpl))
    )
    estimate_comp_time(n_pairs, 1, "virtual_pitches", verbose)

    tmpl_dens = build_exp_tens(
        tmpl_p, tmpl_w, sigma, 1, False, False, 1200, verbose=False,
    )
    tmpl_vals = eval_exp_tens(
        tmpl_dens, x_tmpl, verbose=False,
        truncation_sigmas=truncation_sigmas,
        kernel_precision=kernel_precision,
    )
    tmpl_norm_sq = float(np.sum(tmpl_vals ** 2))
    n_tmpl = len(tmpl_vals)

    vp_w, n_xcorr = _virtual_pitches_chord_only(
        chord_p, chord_w, sigma, tmpl_vals, tmpl_norm_sq, margin, step,
        truncation_sigmas, kernel_precision,
    )

    lag_indices = np.arange(n_xcorr) - (n_tmpl - 1)
    vp_p = lag_indices * step + p_offset

    return vp_p, vp_w


def _virtual_pitches_batched(P, W, sigma, spectrum, chord_spectrum, resolution,
                             truncation_sigmas, kernel_precision, verbose):
    """Batched dispatch over rows of a 2-D pitch matrix.

    Returns ``(vp_p_list, vp_w_list)`` — length-``M`` lists of 1-D
    arrays, one per row. Profile lengths can differ across rows
    because each chord's cross-correlation grid extent depends on
    its highest pitch. Empty arrays are returned for rows with no
    valid pitches.

    Builds the harmonic template once for the whole batch and
    deduplicates chord-side work via canonical-key caching: the
    offset-independent profile ``vp_w`` and length ``n_xcorr`` are
    cached per canonical chord; per-row ``vp_p`` is reconstructed
    from the cached length plus the row's ``p_offset``.
    """
    M, K = P.shape
    use_w = W is not None
    if use_w:
        W = np.asarray(W, dtype=np.float64)
        if W.shape != P.shape:
            raise ValueError("W must be the same shape as P.")

    vp_p_list: list = [np.array([], dtype=np.float64) for _ in range(M)]
    vp_w_list: list = [np.array([], dtype=np.float64) for _ in range(M)]

    # The harmonic template is shared across rows.
    tmpl_p, tmpl_w = add_spectra(np.array([0.0]), np.array([1.0]), *spectrum)
    tmpl_dens = build_exp_tens(
        tmpl_p, tmpl_w, sigma, 1, False, False, 1200, verbose=False,
    )
    margin = 4 * sigma
    step = resolution
    x_tmpl = np.arange(0, np.max(tmpl_p) + margin + step, step)
    tmpl_vals = eval_exp_tens(tmpl_dens, x_tmpl, verbose=False,
                              truncation_sigmas=truncation_sigmas,
                              kernel_precision=kernel_precision)
    n_tmpl = len(tmpl_vals)
    tmpl_norm_sq = float(np.sum(tmpl_vals ** 2))

    result_cache: dict = {}

    # Adaptive progress-print state. Defaults: silent.
    prog_stride = 1
    show_progress = False
    # Up-front time estimate (printed once for the whole batch).
    # Calibration uses the same _virtual_pitches_chord_only path the
    # main loop uses, so the timed work matches per-row main-loop
    # cost (template rebuilds are not counted, matching reality).
    if verbose and M > 1:
        n_cal = min(10, M)
        sample_idx = np.unique(np.linspace(0, M - 1, n_cal).astype(int))

        warmup_done = False
        for s_idx in sample_idx:
            p_row_s = P[s_idx]
            mask_s = ~np.isnan(p_row_s)
            p_valid_s = p_row_s[mask_s]
            if len(p_valid_s) < 1:
                continue
            w_valid_s = W[s_idx, mask_s] if use_w \
                else np.ones_like(p_valid_s)
            p_shifted_s = p_valid_s - np.min(p_valid_s)
            if chord_spectrum is not None:
                chord_p_s, chord_w_s = add_spectra(
                    p_shifted_s, w_valid_s, *chord_spectrum,
                )
            else:
                chord_p_s, chord_w_s = p_shifted_s.copy(), w_valid_s.copy()
            _virtual_pitches_chord_only(
                chord_p_s, chord_w_s, sigma,
                tmpl_vals, tmpl_norm_sq, margin, step,
                truncation_sigmas, kernel_precision,
            )
            warmup_done = True
            break

        if warmup_done:
            t_cal_start = time.perf_counter()
            n_valid_cal = 0
            for s_idx in sample_idx:
                p_row_s = P[s_idx]
                mask_s = ~np.isnan(p_row_s)
                p_valid_s = p_row_s[mask_s]
                if len(p_valid_s) < 1:
                    continue
                w_valid_s = W[s_idx, mask_s] if use_w \
                    else np.ones_like(p_valid_s)
                p_shifted_s = p_valid_s - np.min(p_valid_s)
                if chord_spectrum is not None:
                    chord_p_s, chord_w_s = add_spectra(
                        p_shifted_s, w_valid_s, *chord_spectrum,
                    )
                else:
                    chord_p_s, chord_w_s = \
                        p_shifted_s.copy(), w_valid_s.copy()
                _virtual_pitches_chord_only(
                    chord_p_s, chord_w_s, sigma,
                    tmpl_vals, tmpl_norm_sq, margin, step,
                    truncation_sigmas, kernel_precision,
                )
                n_valid_cal += 1
            if n_valid_cal > 0:
                from ._utils import progress_stride
                t_cal_total = time.perf_counter() - t_cal_start
                t_per_row = t_cal_total / n_valid_cal
                est_total = t_cal_total + t_per_row * M
                maybe_print_batched_estimate(

                    "virtual_pitches", M, est_total,

                )
                prog_stride = progress_stride(t_per_row)
                show_progress = est_total >= 5

    for i in range(M):
        p_row = P[i]
        mask = ~np.isnan(p_row)
        p_valid = p_row[mask]
        if len(p_valid) < 1:
            continue
        w_valid = W[i, mask] if use_w else np.ones_like(p_valid)

        p_offset = float(np.min(p_valid))

        # Canonical key for chord-side dedup. virtualPitches transposes
        # internally (p -= min), so the canonical form is taken in
        # rel mode. The vp_w profile depends only on the canonical
        # chord shape (and chord_spectrum, sigma, etc., which are
        # constant across the batch); vp_p reconstruction uses the
        # per-row p_offset.
        key, _, _ = _chord_canonical_key(
            p_valid, w_valid,
            sigma=sigma, r=1, is_rel=True, is_per=False, period=1200.0,
        )

        if key in result_cache:
            vp_w, n_xcorr = result_cache[key]
        else:
            p_shifted = p_valid - p_offset
            if chord_spectrum is not None:
                chord_p, chord_w = add_spectra(
                    p_shifted, w_valid, *chord_spectrum,
                )
            else:
                chord_p, chord_w = p_shifted.copy(), w_valid.copy()
            vp_w, n_xcorr = _virtual_pitches_chord_only(
                chord_p, chord_w, sigma,
                tmpl_vals, tmpl_norm_sq, margin, step,
                truncation_sigmas, kernel_precision,
            )
            result_cache[key] = (vp_w, n_xcorr)

        lag_indices = np.arange(n_xcorr) - (n_tmpl - 1)
        vp_p_list[i] = lag_indices * step + p_offset
        vp_w_list[i] = vp_w

        if verbose and show_progress \
                and ((i + 1) % prog_stride == 0 or i == M - 1):
            print(f"  {i + 1} / {M} rows computed.")

    return vp_p_list, vp_w_list


# ===================================================================
#  Roughness
# ===================================================================


def roughness(
    f: np.ndarray,
    w: np.ndarray | None = None,
    *,
    p_norm: float = 1.0,
    average: bool = False,
) -> float:
    """Sensory roughness of a weighted multiset of partials.

    Computes the total sensory roughness by summing the pairwise
    roughness contributions of all partial pairs, using Sethares'
    (1993) parameterisation of Plomp and Levelt's (1965) empirical
    dissonance curve.

    Frequencies must be in Hz. Use :func:`~mpt.convert.convert_pitch`
    to convert from other scales.

    Parameters
    ----------
    f : array-like
        Frequencies in Hz (positive).
    w : array-like or None
        Amplitudes/weights (``None`` for all ones).
    p_norm : float
        Norm exponent for combining pairwise roughnesses (default 1).
    average : bool
        If True, divide by the number of pairs (default False).

    Returns
    -------
    float
        Total (or average) roughness (non-negative).

    References
    ----------
    Sethares, W. A. (1993). Local consonance and the relationship
    between timbre and scale. *JASA*, 94(3), 1218–1228.
    """
    f = np.asarray(f, dtype=np.float64).ravel()
    K = len(f)
    w = validate_weights(w, K)

    if np.any(f <= 0):
        raise ValueError("Frequencies must be positive.")

    # Sethares (1993) parameters
    Dstar = 0.24
    S1 = 0.0207
    S2 = 18.96
    C1 = 5.0
    C2 = -5.0
    A1 = -3.51
    A2 = -5.75

    f_diff = f[None, :] - f[:, None]  # K x K
    f_min = np.minimum(f[None, :], f[:, None])
    w_min = np.minimum(w[None, :], w[:, None])

    mask = f_diff.ravel() > 0
    f_diff = f_diff.ravel()[mask]
    f_min = f_min.ravel()[mask]
    w_min = w_min.ravel()[mask]

    s = Dstar / (S1 * f_min + S2)
    pair_rough = w_min * (C1 * np.exp(A1 * s * f_diff) + C2 * np.exp(A2 * s * f_diff))

    r = float(np.sum(pair_rough**p_norm) ** (1 / p_norm))

    if average and K >= 2:
        from math import comb
        r /= comb(K, 2)

    return r
