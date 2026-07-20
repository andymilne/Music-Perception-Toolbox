"""Self-calibrated up-front time estimate for ``eval_exp_tens``.

The dispatch cost model (:func:`mpt._tensor.dispatch._ma_eval_costs_ms`)
prices each evaluation path in milliseconds on the machine whose timings
its constants were fitted to. Absolute wall-clock time on the user's
machine differs from that by a hardware scale factor. The factor cancels
in the dispatch *ratio* (centres versus Möbius), which is why the
dispatcher needs no timing, but it does not cancel in an *absolute*
estimate.

This module recovers the factor once per session by timing one small
fixed reference evaluation and dividing measured time by the cost
model's prediction for it, then scales every prediction by the result.
The reference is a real evaluation, so it tracks the same optimized
kernels the estimate is about, and the machine factor it yields applies
across evaluation shapes because the cost model already carries the
shape dependence.

The estimate is approximate --- good to roughly a factor of two, since
the shape model is not perfectly uniform across regimes and runtime
varies with load, thermal state, and BLAS threading. It is presented
only as a cancel prompt, never as a precise figure.
"""

import time

import numpy as np

# Cached machine scale (measured reference time / predicted reference
# time). ``None`` until the first session calibration.
_SESSION_SCALE = None

# Reentrancy guard: True while the reference evaluation runs, so its own
# dispatch does not recurse back into calibration.
_CALIBRATING = False

# An evaluation is flagged for an up-front warning above this wall-clock
# estimate, in seconds. The warning's only role is to let the user
# cancel a long call, so a coarse threshold suffices.
_EVAL_WARN_THRESHOLD_SEC = 10.0

# Reference evaluation shape: a two-attribute relative density that the
# cost model routes on its own (not a forced path), sized to run in a
# few tens of milliseconds --- long enough for a stable timing, short
# enough to be an imperceptible one-off at session start.
_REF_N_Q = 160
_REF_TIMING_REPEATS = 3


def _build_reference():
    """Build the fixed reference density (deterministic shape and data)."""
    from ..tensor import build_exp_tens

    p0 = np.array(
        [[0.0, 120.0, 290.0, 410.0, 560.0, 700.0, 830.0, 980.0,
          1100.0, 1240.0, 1370.0, 1490.0]]
    ).T
    p1 = np.array(
        [[0.0, 100.0, 300.0, 400.0, 550.0, 690.0, 820.0, 970.0,
          1090.0, 1230.0, 1360.0, 1480.0]]
    ).T
    return build_exp_tens(
        [p0, p1], None, [15.0, 15.0], [2, 2],
        [True, True], [False, False], [0.0, 0.0], verbose=False,
    )


def _session_time_scale():
    """Return the cached machine scale, calibrating once on first use.

    On failure (any exception while building or timing the reference)
    the scale falls back to ``1.0`` --- the estimate then reads in
    calibration-machine units, which is wrong in absolute terms but
    never crashes a real evaluation for the sake of a warning.
    """
    global _SESSION_SCALE, _CALIBRATING
    if _SESSION_SCALE is not None:
        return _SESSION_SCALE

    _CALIBRATING = True
    from .._defaults import _DEFAULTS
    saved_hints = _DEFAULTS.get("show_hints", True)
    _DEFAULTS["show_hints"] = False  # keep the reference call silent
    try:
        from .dispatch import _select_ma_eval, _predict_ma_eval_cost_ms
        from ..tensor import eval_exp_tens

        dens = _build_reference()
        rng = np.random.default_rng(0xC0FFEE)
        x = rng.uniform(0.0, 1200.0, size=(int(dens.dim), _REF_N_Q))
        chosen, _ = _select_ma_eval(dens, _REF_N_Q, method="auto")
        pred_ms = _predict_ma_eval_cost_ms(dens, _REF_N_Q, chosen)

        eval_exp_tens(dens, x, verbose=False)  # warm caches
        samples = []
        for _ in range(_REF_TIMING_REPEATS):
            t0 = time.perf_counter()
            eval_exp_tens(dens, x, verbose=False)
            samples.append((time.perf_counter() - t0) * 1000.0)
        actual_ms = sorted(samples)[len(samples) // 2]  # median
        _SESSION_SCALE = (actual_ms / pred_ms) if pred_ms > 0 else 1.0
    except Exception:
        _SESSION_SCALE = 1.0
    finally:
        _DEFAULTS["show_hints"] = saved_hints
        _CALIBRATING = False
    return _SESSION_SCALE


def _estimate_eval_seconds(dens, n_q, chosen):
    """Approximate wall-clock seconds for the chosen evaluation path."""
    from .dispatch import _predict_ma_eval_cost_ms

    pred_ms = _predict_ma_eval_cost_ms(dens, n_q, chosen)
    return pred_ms * _session_time_scale() / 1000.0


def _format_time_warning(label, seconds):
    if seconds >= 90.0:
        amount = f"~{seconds / 60.0:.0f} min"
    else:
        amount = f"~{seconds:.0f} s"
    return (
        f"{label}: estimated {amount} "
        f"(approximate; press Ctrl+C to cancel)."
    )


def _maybe_warn_eval_time(label, dens, n_q, chosen):
    """Emit one approximate time warning per top-level call, if warranted.

    Gated by ``show_hints`` (so disabling hints also skips the reference
    calibration entirely) and latched once per top-level call via
    :data:`mpt._defaults._TIME_WARN_EMITTED`, so a wrapper making several
    inner evaluations warns at most once --- from the first whose
    estimate crosses the threshold.
    """
    if _CALIBRATING:
        return
    from .._defaults import _DEFAULTS, _TIME_WARN_EMITTED

    if not _DEFAULTS.get("show_hints", True):
        return
    if _TIME_WARN_EMITTED[0]:
        return
    seconds = _estimate_eval_seconds(dens, n_q, chosen)
    if seconds < _EVAL_WARN_THRESHOLD_SEC:
        return
    _TIME_WARN_EMITTED[0] = True
    print(_format_time_warning(label, seconds))
