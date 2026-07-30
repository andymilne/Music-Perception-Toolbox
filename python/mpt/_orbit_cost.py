"""Power-law cost model for the orbit-versus-enumeration choice.

Mirror of MATLAB ``+internal/orbitCostModel.m``. See that file for the
derivation, the validation figures, and the known weak spots.
"""
from __future__ import annotations

import math

# Fitted 2026-07-30 against 444 timed cells, r = 2..8, K = r..22,
# B = 1..2048, absolute and relative-periodic modes (which timed
# identically). Each route's time was fitted as a power law in the
# quantities it works on and the two fits subtracted, so only their
# difference appears here. Leave-one-tuple-size-out validation reproduced
# the measured crossover within one in K in 19 of 21 (r, B) combinations,
# and held-out performance matched in-sample, indicating the fit captured
# the scaling rather than memorising individual tuple sizes.
#
# Accuracy, measured over 484 timed cells on the machine the model was fitted
# on: the predicted time ratio sits within a factor of about 2.5 of the
# measured one typically (standard deviation 0.93 in log), with occasional
# cells further out. That is adequate for the purpose --- the two routes
# differ by 10x to 200x away from the crossover, so a factor of 2.5 changes
# nothing there, and near the crossover choosing the wrong route costs little.
# Refitting the difference directly rather than each route separately, and
# adding a second machine-run's cells, left the coefficients and the crossover
# placement unchanged, so this scatter is the model's own and not a sign the
# form is wrong.
#
# Only the intercept is machine-specific; the rest are scaling exponents.
# The intercept ships as the ``orbit_cost_intercept`` default rather than as
# a constant here, so it can be recalibrated per machine.
_C_LOG_N_ORBITS = 1.0708
_C_LOG_K = 1.4033
_C_LOG_B = -0.3608
_C_LOG_TUPLE_PAIRS = -0.8188
_C_LOG_R = -0.2261

#: Tuple sizes at or below which the model gets one element of headroom.
#: Every hold-out miss sat here, where the fit has the fewest orbit
#: classes to work with and switched to the Möbius route one or two K
#: early. Testing one K lower delays the switch by one.
_MARGIN_R_MAX = 3


def _log_tuple_pairs(r: int, K: int) -> float:
    """``log(Tx * Ty)`` for the tuple counts enumeration materialises."""
    log_tx = math.lgamma(K + 1) - math.lgamma(K - r + 1)
    log_ty = (math.lgamma(K + 1) - math.lgamma(r + 1)
              - math.lgamma(K - r + 1))
    return log_tx + log_ty


def orbit_cost_log_ratio(r: int, K: int, B: int = 1) -> float:
    """Predicted ``log(t_orbit / t_enum)``. Negative favours Möbius.

    The intercept is read from the ``orbit_cost_intercept`` default, so a
    machine other than the one the model was fitted on can be calibrated
    without touching the exponents. Larger values favour enumeration.
    """
    from ._defaults import get_default
    from ._tensor._nested_contraction import _n_orbits
    return (float(get_default("orbit_cost_intercept"))
            + _C_LOG_N_ORBITS * math.log(_n_orbits(r))
            + _C_LOG_K * math.log(K)
            + _C_LOG_B * math.log(B)
            + _C_LOG_TUPLE_PAIRS * _log_tuple_pairs(r, K)
            + _C_LOG_R * math.log(r))


def orbit_cost_model(r: int, K: int, B: int = 1) -> tuple[bool, float]:
    """Is the Möbius route the faster combine at this shape?

    Returns ``(is_faster, log_ratio)``. The margin is returned alongside
    the verdict so a caller can report how close the decision was, and a
    harness can compare predicted against measured on a continuous scale
    rather than as a win/lose label.

    The batch extent is an argument because the crossover moves by up to
    11 in K across the batch range --- for r = 2, from K = 19 at B = 1
    down to K = 8 at B = 2048 --- as the two routes amortise differently
    over the batch. A predicate without it cannot express that.
    """
    if r > K or r < 2:
        # No r-tuples to form, or an r = 1 factor that is exact either way.
        return False, math.inf
    k_eff = max(r, K - 1) if r <= _MARGIN_R_MAX else K
    log_ratio = orbit_cost_log_ratio(r, k_eff, B)
    return log_ratio < 0.0, log_ratio
