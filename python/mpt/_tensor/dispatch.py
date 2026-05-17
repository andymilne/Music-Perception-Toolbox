"""Path-selection cost model and dispatch helpers.

This module hosts the toolbox's path-selection policy --- the cost-
model + timing-probe logic that decides between the centres path and
the Möbius path for a given evaluation or inner-product call, plus the
small set of pure helpers (``_normalize_density_input``,
``_resolve_list_list_mode``, ``_compute_Q``, ``_format_time``,
``_falling_factorial``) that are shared between :mod:`._tensor.cosine`,
:mod:`._tensor.eval`, and :mod:`._tensor.windowing`.

See :doc:`/ARCHITECTURE` §4 ("Dispatcher pattern") for the per-call
flow (pre-screen / cost-model / timing-probe / cancellation-guard) and
USER_GUIDE §4 ("Method selection") for the user-facing description.

Cross-module dependencies: the probe functions
(:func:`_probe_eval_path`, :func:`_probe_ip_path`) reach back into
:mod:`._tensor.cosine` / :mod:`._tensor.eval` / :mod:`._tensor.build`
to time the actual implementations. Those imports are deferred to
call time to avoid an import cycle at package load (this module is
imported by ``cosine`` and ``eval``).
"""
from __future__ import annotations

import warnings
from math import comb as _math_comb, factorial

import numpy as np

from .._defaults import _maybe_show_dispatch_msg
from .density import ExpTensDensity, MaetDensity, WindowedMaetDensity




# -------------------------------------------------------------------
#  Polymorphic-dispatch helpers for cos_sim_exp_tens
# -------------------------------------------------------------------


def _normalize_density_input(arg, *, name: str):
    """Detect whether ``arg`` is a single density or a list of densities.

    Returns
    -------
    is_scalar : bool
        True if ``arg`` is a single density object (not a list/tuple/array).
    densities : tuple
        Tuple of density objects.
    """
    if isinstance(arg, np.ndarray) and arg.dtype == object:
        arg = list(arg)

    if isinstance(arg, (ExpTensDensity, MaetDensity, WindowedMaetDensity)):
        return True, (arg,)

    if isinstance(arg, (list, tuple)):
        if len(arg) == 0:
            return False, ()
        for i, elem in enumerate(arg):
            if not isinstance(
                elem, (ExpTensDensity, MaetDensity, WindowedMaetDensity)
            ):
                raise TypeError(
                    f"{name}[{i}] must be an ExpTensDensity, MaetDensity, "
                    f"or WindowedMaetDensity; got {type(elem).__name__}."
                )
        return False, tuple(arg)

    raise TypeError(
        f"{name} must be a density object or a list/tuple of densities; "
        f"got {type(arg).__name__}."
    )



def _resolve_list_list_mode(mode: str, m: int, n: int) -> str:
    """Resolve ``mode`` for the list-vs-list case. Returns 'pairwise' or 'cartesian'."""
    if mode == "pairwise":
        if m != n:
            raise ValueError(
                f"mode='pairwise' requires equal-length lists; got M={m}, N={n}. "
                f"Pass mode='cartesian' for the M×N case."
            )
        return "pairwise"
    if mode == "cartesian":
        return "cartesian"
    if mode == "auto":
        if m == n:
            return "pairwise"
        raise ValueError(
            f"mode='auto' requires equal-length lists for pairwise resolution; "
            f"got M={m}, N={n}. Pass mode='cartesian' for the M×N case "
            f"or mode='pairwise' to assert equal lengths."
        )
    raise ValueError(
        f"mode must be one of 'auto', 'pairwise', 'cartesian'; got {mode!r}."
    )



def _orbit_ips_look_corrupted(ip_xy, ip_xx, ip_yy):
    """Cheap post-hoc sanity check on Möbius-method-computed inner products.

    The Möbius method's alternating partition sum can break down
    catastrophically in two regimes documented during the May 2026
    audit:

    * σ → 0 with low K and r ≥ 3 (music-theoretical exact-match regime):
      auto-IP terms cancel to a value with magnitude near
      machine epsilon, then floating-point overflow can produce huge
      garbage values when the cosine ratio is taken.
    * Issue 4 sharp-Gaussian regime (σ small relative to data range):
      auto-IPs lose 4–8 decimal digits of precision while looking
      finite; this check does NOT catch that — only the catastrophic
      overflow / sign-corruption regime.

    Triggers on any of:
    * non-finite IP (NaN or Inf in any of the three),
    * negative auto-IP (a Gram-matrix diagonal must be ≥ 0; sign flip
      is unambiguous corruption),
    * cosine magnitude > 1 + 1e-6 (impossible for a genuine cosine).

    Parameters
    ----------
    ip_xy, ip_xx, ip_yy : float
        Cross and auto inner products from the Möbius method.

    Returns
    -------
    bool
        True if the IPs are unsuitable for use and the caller should
        fall back to Bulger's method.
    """
    if not (np.isfinite(ip_xy) and np.isfinite(ip_xx) and np.isfinite(ip_yy)):
        return True
    if ip_xx < 0 or ip_yy < 0:
        return True
    denom = np.sqrt(ip_xx * ip_yy)
    if denom > 0 and abs(ip_xy) > 1.000001 * denom:
        return True
    return False



# Per-r K thresholds for the Möbius-vs-Bulger crossover, established
# empirically on representative MAET workloads (N = 8-12, σ = 12,
# P = 1200, samples_per_sigma = 5). Retained for reference but
# superseded by the cost-model dispatcher below, which also accounts
# for N (which the K thresholds alone do not — at N = 2 the abs
# crossover is K ≥ 14 for r = 2, but at N = 16 it is K ≥ 6, a span a
# single threshold can't capture).
_K_THRESHOLD_ABS = {2: 7, 3: 6, 4: 5, 5: 4, 6: 4}

_K_THRESHOLD_REL_PER = {2: float("inf"), 3: 10, 4: 8, 5: 7, 6: 6}



# Cost-model constants for the dispatcher. Refit on a 328-cell wall-time
# benchmark covering all four modes (abs/rel × per/nonper) at A ∈ {1, 2},
# r ∈ {2, 3, 4}, N ∈ {2, 4, 8, 16}, K spanning each mode's feasible
# range. r ∈ {5, 6} extrapolated from |Ω_r| growth (4, 10, 33, 92, 306,
# 948). Predicts Bulger and Möbius wall times in milliseconds and picks
# the smaller. Validated against 277 measured cells: 94 % within 5 % of
# optimal, 0 mis-routes to Bulger's method (no OOM-zone violations), 7
# close-call mis-routes to the Möbius method (max 3.9 × slowdown, all at
# < 100 ms absolute).

# Bulger: per-entry cost of the (n_J × n_K) kernel matrix in ms. The
# periodic branches build a wrapped-difference tensor, which empirically
# costs ~2.0–2.3 × the non-periodic branch (modular arithmetic plus
# index-array growth). Verified across both abs and rel modes.
_PW_PER_ENTRY_MS_NONPER = 1.0e-4

_PW_PER_ENTRY_MS_PER = 7.0e-4   # p75 of measured per-entry cost (per bucket)



def _pw_per_entry_ms(any_per):
    """Pick the per-entry cost for Bulger's method based on whether any group wraps."""
    return _PW_PER_ENTRY_MS_PER if any_per else _PW_PER_ENTRY_MS_NONPER



# Möbius method (absolute modes, both per and nonper — empirically
# within ±5 % of each other). Vectorised across event pairs, so cost
# is roughly constant in N_x · N_y; linear in A at r = 2, 3 and slightly
# sub-linear at r = 4. Per-r baseline at A = 1. Entries for r >= 7
# extrapolated from the empirical 3× orbit-class count growth per r (anchored
# to measured r=2..6 values); these are conservative and may be refined later.
_ORBIT_ABS_PER_ATTR_MS = {
    2: 3.0, 3: 11.2, 4: 45.0, 5: 150.0, 6: 500.0,
    7: 1500.0, 8: 4500.0,
}


# Möbius method (relative-periodic): vectorised across event pairs but each
# pair carries a u-grid integration of N_u ≈ period/σ × samples_per_σ
# samples, plus a fixed per-call setup cost (~5 ms). Cost grows with
# A · N_x · N_y · K_max² · |Ω_r|.
_ORBIT_RELPER_BASE_MS = 5.0

_ORBIT_RELPER_PER_PAIR_K2_MS = {
    2: 0.06, 3: 0.40, 4: 1.0, 5: 5.0, 6: 20.0,
    7: 60.0, 8: 200.0,
}


# Möbius method (relative-aperiodic): the implementation here is a
# per-(n_X, n_Y) Python loop (not batched across event pairs), so the
# per-pair-K² constant is roughly 4 × the rel-periodic constant. At
# A = 1 this Möbius branch is almost always slower than Bulger's method;
# at A ≥ 2 it wins comfortably once K is moderate, because Bulger's
# method grows as ∏_a C(K_a, r_a)² which compounds across attributes
# whereas the Möbius method adds linearly.
_ORBIT_RELNONPER_BASE_MS = 5.0

_ORBIT_RELNONPER_PER_PAIR_K2_MS = {
    2: 0.25, 3: 1.05, 4: 3.30, 5: 12.0, 6: 50.0,
    7: 150.0, 8: 500.0,
}



def _orbit_beats_pairwise_per_attr(r, K, is_rel, is_per):
    """Per-attribute K-threshold heuristic for Möbius-vs-Bulger crossover (legacy; superseded).

    Retained for callers that haven't migrated; the cost-model
    dispatcher in ``_select_ma_inner_product_method`` is preferred.
    """
    if r == 1:
        return False
    if r > _ORBIT_R_MAX_SHIPPED:
        return False
    if is_rel and is_per:
        threshold = _K_THRESHOLD_REL_PER.get(r, 999)
    else:
        threshold = _K_THRESHOLD_ABS.get(r, 999)
    return K >= threshold



def _predict_pairwise_kernel_size(r_vec, k_vec, A, N_x, N_y):
    """Predicted n_J · n_K for the MA path under Bulger's method.

    n_J^X = N_x · ∏_a r_a! · C(K_a, r_a)
    n_K^Y = N_y · ∏_a C(K_a, r_a)
    so n_J · n_K = N_x · N_y · ∏_a r_a! · C(K_a, r_a)².
    """
    if A == 0:
        return float(N_x * N_y)
    size = float(N_x * N_y)
    for a in range(A):
        r_a = int(r_vec[a])
        K_a = int(k_vec[a])
        if K_a < r_a:
            return float('inf')
        c = float(_math_comb(K_a, r_a))
        size *= float(factorial(r_a)) * c * c
    return size



def _predict_orbit_cost_ms(
    r_max, A, N_x, N_y, k_vec, any_rel_nonper, any_rel_per,
):
    """Predicted Möbius-method MA wall time in milliseconds.

    Routes to the appropriate per-r constant based on the group mode:
    rel-aperiodic uses the per-pair Python-loop constants (largest);
    rel-periodic uses the u-grid-integration constants; absolute uses
    the vectorised batch constants. A scaling is linear (verified at
    r = 2, 3 to within ~5 %; slightly sub-linear at r = 4 but linear-A
    over-predicts conservatively, biasing the dispatcher toward
    Bulger's method in close calls at r = 4 — and at r = 4 Bulger's
    side explodes so quickly that this never matters in the OOM zone).
    """
    K_max = int(np.max(k_vec)) if A > 0 else 1
    if any_rel_nonper:
        c = _ORBIT_RELNONPER_PER_PAIR_K2_MS[r_max]
        return float(A * (_ORBIT_RELNONPER_BASE_MS
                          + N_x * N_y * K_max * K_max * c))
    if any_rel_per:
        c = _ORBIT_RELPER_PER_PAIR_K2_MS[r_max]
        return float(A * (_ORBIT_RELPER_BASE_MS
                          + N_x * N_y * K_max * K_max * c))
    return float(A * _ORBIT_ABS_PER_ATTR_MS[r_max])



def _select_ma_inner_product_method(
    *,
    r_vec, k_vec, A,
    N_x, N_y,
    any_per, any_rel_nonper, any_rel_per,
    sigma_over_P_max, user_method,
):
    """Pick the inner-product method for the MA case using a cost model.

    Routing rules, in order:

    1. ``user_method`` keyword override (anything other than 'auto').
    2. Hard fallbacks where the Möbius method cannot or should not run:
       - r_max ≤ 1: no within-tuple structure to exploit.
       - r_max > _ORBIT_R_MAX_SHIPPED: no orbit table available.
    3. Soft fallback: rel + per with σ/P beyond the integration-exact
       regime warns and routes to Bulger's method.
    4. Otherwise predict both wall times (in ms) and pick the smaller;
       ties favour Bulger's method (no orbit-table fetch, no Möbius
       cancellation risk).

    Ragged K_{a,n} (NaN-padded events) is handled inside
    :func:`_ma_per_attr_inner_matrix` via a per-event safe/unsafe
    partition: events with K_eff - r >= 2 (the Möbius-method precision margin)
    flow through the vectorised batched Möbius evaluator; pairs
    involving any K_eff - r < 2 event flow through direct r-tuple
    enumeration (no Möbius alternating sum, hence no cancellation).
    The dispatcher therefore does not route on the presence of NaN entries.

    The four modes (abs+nonper, abs+per, rel+nonper, rel+per) are
    routed as follows:

    - abs + nonper: cost model with `_PW_PER_ENTRY_MS_NONPER` and
      `_ORBIT_ABS_PER_ATTR_MS`.
    - abs + per: cost model with `_PW_PER_ENTRY_MS_PER` (wrap on δ
      tensor adds ~2 × Bulger overhead) and same Möbius constants
      (Möbius cost is mode-independent in benchmark, ±5 %).
    - rel + per: cost model with `_PW_PER_ENTRY_MS_PER` and
      `_ORBIT_RELPER_PER_PAIR_K2_MS` (Möbius u-grid integration
      scales with N_x · N_y · K_max² · |Ω_r|).
    - rel + nonper: cost model with `_PW_PER_ENTRY_MS_NONPER` and
      `_ORBIT_RELNONPER_PER_PAIR_K2_MS` (Möbius per-pair Python loop;
      ~4 × the rel-per per-K² constant). At A = 1 the cost model
      reliably routes to Bulger's method; at A ≥ 2 it routes to the
      Möbius method once Bulger's ∏_a C(K_a, r_a)² compounding
      overtakes the Möbius method's additive A · K_max² growth.

    Parameters
    ----------
    r_vec : (A,) intp
        Per-attribute r_a.
    k_vec : (A,) intp
        Per-attribute slab dimension K_a (the kernel slab size; events
        within an attribute may have lower K_eff via NaN padding,
        which the Möbius-method wrapper handles via per-event
        safe/unsafe partition).
    A : int
        Number of attributes.
    N_x, N_y : int
        Event counts of the two densities.
    any_per : bool
        True if any group has is_per=True (drives Bulger wrap cost).
    any_rel_nonper : bool
    any_rel_per : bool
    sigma_over_P_max : float
        Maximum σ/P across periodic-relative groups.
    user_method : {'auto', 'bulger', 'mobius', 'direct'}
    """
    if user_method != 'auto':
        return user_method
    r_max = int(np.max(r_vec)) if A > 0 else 1
    if r_max <= 1:
        return 'bulger'
    if r_max > _ORBIT_R_MAX_SHIPPED:
        return 'bulger'
    # K-vs-r precision guard. The Möbius method's auto-inner-products can
    # suffer catastrophic Möbius cancellation when any K_a is too close
    # to its r_a (see _ORBIT_K_MINUS_R_MIN block). The cross
    # cancellation guard at the call site does NOT catch this, since it
    # inspects only |<T_X,T_Y>|; corrupted <T_X,T_X> propagates silently
    # into the cosine denominator.
    if A > 0 and not _orbit_safe_for_precision(r_vec, k_vec):
        return 'bulger'
    # Periodic-relative beyond σ/P threshold: in this regime the Möbius
    # method computes the JMM Eq. 3.4 integral form, while Bulger's
    # method computes the single-nearest-image-wrap form.
    # The two diverge by O((σ/P)^∞) starting around σ/P ≈ 0.03. For
    # backward compatibility the toolbox treats Bulger's
    # pairwise-wrap form as canonical; the Möbius method is therefore
    # disabled above the threshold. Users who want the JMM-exact integral
    # explicitly may pass method='mobius'.
    if any_rel_per and sigma_over_P_max > _ORBIT_SIGMA_OVER_P_THRESHOLD:
        warnings.warn(
            f"Maximum σ/P = {sigma_over_P_max:.3f} across periodic-relative "
            f"groups exceeds the Möbius-method threshold "
            f"({_ORBIT_SIGMA_OVER_P_THRESHOLD}); falling back to Bulger's "
            f"method (the pairwise-wrap form). Pass method='bulger' "
            f"explicitly to silence this warning."
        )
        return 'bulger'

    pw_size = _predict_pairwise_kernel_size(r_vec, k_vec, A, N_x, N_y)
    pw_cost_ms = pw_size * _pw_per_entry_ms(any_per)
    orbit_cost_ms = _predict_orbit_cost_ms(
        r_max, A, N_x, N_y, k_vec, any_rel_nonper, any_rel_per,
    )
    if pw_cost_ms <= orbit_cost_ms:
        return 'bulger'
    return 'mobius'



def _compute_Q(D, r, is_rel, is_per, period):
    """Compute the quadratic form from (already-wrapped) differences D.

    When *is_rel* and *is_per* are both True, pairwise differences
    between components of D are wrapped to ``[-period/2, period/2)``
    before squaring. This restores exact transposition invariance on
    the circle, which is otherwise broken by component-wise wrapping.

    The two formulas are algebraically identical in the non-periodic
    case: ``sum_{i<j} (d_i - d_j)^2 == r * (sum(d^2) - sum(d)^2/r)``.
    """
    if is_rel:
        if is_per:
            Q = np.zeros(D.shape[1:])
            for i in range(r):
                for j in range(i + 1, r):
                    delta = D[i] - D[j]
                    delta = delta - period * np.floor(delta / period + 0.5)
                    Q += delta**2
            Q = Q / r
        else:
            Q = np.sum(D**2, axis=0) - np.sum(D, axis=0) ** 2 / r
    else:
        Q = np.sum(D**2, axis=0)
    return Q



# -------------------------------------------------------------------
#  Möbius method dispatcher (single-attribute path)
# -------------------------------------------------------------------
#
#  The Möbius method is layered — a partition-decomposition with
#  orbit collapse — on top of Bulger's existing ``_ip_core`` path
#  (the v1 / v2.1 decomposition). See ``v22_specification.md`` and
#  ``mpt/_mobius.py`` for the combinatorial details.
#
#  The user-facing ``cos_sim_exp_tens`` gains two keywords:
#
#    method='auto'   : dispatcher chooses the Möbius method or Bulger's
#                     method based on (r, n, mode, sigma/period).
#    method='bulger' : forces Bulger's method (the v1 / v2.1
#                     decomposition with periodic pairwise-wrap form;
#                     ``_ip_core``); this is the closed form of JMM
#                     Eq. 3.4 — the toolbox's defined value of
#                     the rel_per inner product, by definition. At
#                     sigma/P > 0.03 it differs from the alternative
#                     integration form computed by 'mobius' by an
#                     amount that grows as the periodic Theta-tail
#                     terms become non-negligible (see V22_DEV_LOG.md
#                     Issue 3 for details). Slower than the Möbius
#                     method at high r and large K, but correct
#                     across the full sigma/P range.
#    method='direct' : forces direct enumeration (no Möbius cancellation;
#                     useful for diagnosing near-zero cosines).
#                     In the single-attribute path, 'direct' coincides
#                     with 'bulger' (both route through ``_ip_core``);
#                     the distinction surfaces in later windowed paths.
#
#  cancellation_threshold = 1e-12 : when the Möbius method's cross
#    inner product falls below this fraction of sqrt(<A,A><B,B>), the
#    Möbius result may suffer from catastrophic alternating-sum
#    cancellation; in that case fall back to ``_ip_core`` (Bulger's
#    method). In typical use the guard never triggers; the cost is at
#    most one extra Bulger pass. Note: this guard inspects the cross
#    product only — corruption in the auto inner products (<A,A>,
#    <B,B>) propagates through the cosine denominator silently. The
#    K_a >= r_a + 2 margin in `_orbit_safe_for_precision` is the
#    primary protection against auto-IP cancellation; a runtime
#    cancellation diagnostic on auto IPs is on the roadmap
#    (see V22_DEV_LOG.md Issue 4).

_ORBIT_R_MAX_SHIPPED = 8  # orbit tables r=2..8 ship pre-built

_ORBIT_SIGMA_OVER_P_THRESHOLD = 0.03  # σ/P beyond which the periodic-relative Möbius method deviates

_ORBIT_K_MINUS_R_MIN = 2  # K_a >= r_a + this margin required for the Möbius method (precision guard)

# Rationale (May 2026 audit): the Möbius method expresses the
# distinct-r-tuple sum as a signed sum over set-partition orbits.
# When K_a is close to r_a, the expansion has very few orbit classes
# and the Möbius alternation can produce catastrophic cancellation in
# the auto-inner-products <T_X, T_X> and <T_Y, T_Y> (which are not
# protected by the cross-cancellation guard, since that guard only
# inspects |<T_X, T_Y>| / sqrt(<T_X,T_X><T_Y,T_Y>)). Empirical sweep
# (5 seeds × all four modes × r in {2..5}) shows: K = r usually
# catastrophic; K = r+1 typically OK but with marginal r=4,5 cases
# losing ~1e-6 precision; K >= r+2 reaches FP precision uniformly.
# This guard is conservative but cheap: realistic music applications
# have K >> r, so it almost never triggers.


def _orbit_safe_for_precision(r_vec, k_vec):
    """Return True if every attribute satisfies K_a >= r_a + margin.

    Used by both the SA and MA dispatchers to refuse the Möbius method
    when its Möbius cancellation could swamp the answer. See the
    `_ORBIT_K_MINUS_R_MIN` rationale block above.
    """
    r_arr = np.atleast_1d(np.asarray(r_vec, dtype=np.intp))
    k_arr = np.atleast_1d(np.asarray(k_vec, dtype=np.intp))
    return bool(np.all(k_arr - r_arr >= _ORBIT_K_MINUS_R_MIN))



def _select_sa_inner_product_method(r, n_max, is_rel, is_per,
                                    sigma_over_P, user_method,
                                    n_min=None):
    """Pick the inner-product path for the SA case.

    Parameters
    ----------
    r : int
        Tensor order.
    n_max : int
        max(n_x, n_y); the larger of the two source sizes (used for
        the small-problem cutoff).
    is_rel, is_per : bool
        Mode flags.
    sigma_over_P : float
        σ / period; ignored if not periodic.
    user_method : str
        One of 'auto', 'bulger', 'direct'. (Internal callers may also
        pass 'mobius' to force the Möbius method.)
    n_min : int, optional
        min(n_x, n_y); the smaller of the two source sizes. Used for
        the K-vs-r precision guard. Defaults to ``n_max`` (i.e., the
        guard is bypassed if the caller provides only n_max).

    Returns
    -------
    str
        One of 'mobius', 'bulger', 'direct'.
    """
    if user_method != 'auto':
        return user_method
    # r=1: the Möbius machinery is undefined for r<2 (single block, no
    # distinct-index structure); Bulger's method is trivially fast anyway.
    if r <= 1:
        return 'bulger'
    # r=2 with small n: Bulger's method dominates because the Möbius
    # method's overhead (4 orbit classes, numpy.einsum dispatch) exceeds the
    # kernel-matvec cost.
    if r == 2 and n_max <= 8:
        return 'bulger'
    # r > _ORBIT_R_MAX_SHIPPED: shipped orbit tables stop here. At
    # higher r the Möbius method still works correctly, but on first use
    # the table must be built from scratch (cost grows with B_r^2);
    # default to Bulger's method to avoid surprising users with a slow
    # first call. Users who explicitly want the Möbius method at higher r
    # can pass method='mobius'; the cost-preview helper in mobius will
    # print an estimate before the build begins.
    if r > _ORBIT_R_MAX_SHIPPED:
        return 'bulger'
    # K-vs-r precision guard. The Möbius method's auto-inner-products can
    # suffer catastrophic Möbius cancellation when the multiset size is
    # too close to r (see _ORBIT_K_MINUS_R_MIN block). The cross
    # cancellation guard at the call site does NOT catch this, since it
    # inspects only |<T_X,T_Y>|; corrupted <T_X,T_X> propagates silently
    # into the cosine denominator.
    n_for_guard = n_min if n_min is not None else n_max
    if not _orbit_safe_for_precision([r], [n_for_guard]):
        return 'bulger'
    # Periodic-relative beyond σ/P threshold: in this regime the Möbius
    # method computes the JMM Eq. 3.4 integral form, while Bulger's
    # method computes the single-nearest-image-wrap form.
    # The two diverge by O((σ/P)^∞) starting around σ/P ≈ 0.03. For
    # backward compatibility the toolbox treats Bulger's
    # pairwise-wrap form as canonical; the Möbius method is therefore
    # disabled above the threshold. Users who want the JMM-exact integral
    # explicitly may pass method='mobius'.
    if is_rel and is_per and sigma_over_P > _ORBIT_SIGMA_OVER_P_THRESHOLD:
        warnings.warn(
            f"σ/P = {sigma_over_P:.3f} exceeds the Möbius-method threshold "
            f"({_ORBIT_SIGMA_OVER_P_THRESHOLD}) for relative-periodic mode; "
            f"falling back to Bulger's method (the pairwise-wrap form). "
            f"Pass method='bulger' explicitly to silence this warning."
        )
        return 'bulger'
    return 'mobius'



def _select_sa_eval_method(r, K, n_q, is_rel, is_per, sigma_over_P,
                           user_method):
    """Pick the evaluation method for ``eval_exp_tens`` (SA case).

    The choice is between the centres-array path (build
    a ``(dim, n_j)`` centres tensor at ``build_exp_tens`` time, then
    evaluate as a vectorised Gaussian product against the queries) and
    the Möbius point evaluator (Möbius-decomposed sum over set
    partitions; ``O(B_r · r · K · n_q)`` per query independent of
    ``n_j``).

    Cost rule of thumb. The centres path scales as
    ``O(r · n_j · n_q)`` with ``n_j = K!/(K-r)!``, so it explodes at
    high r. The Möbius method replaces ``n_j`` with ``B_r · r · K``,
    where ``B_r`` is the Bell number of ``r`` (5 at r=3, 15 at r=4,
    52 at r=5, 203 at r=6). Crossover analysis (5 partitions × N work
    per partition vs N!/(N-r)!) shows the Möbius method is ~22× faster
    at r=3 N=20, ~100× at r=4. At r=2 the costs are comparable; the
    centres path is simpler and avoids partition-table dispatch
    overhead, so default to centres there.

    Precision guard. The Möbius method suffers catastrophic Möbius
    cancellation when ``K - r < 2`` (same regime as the IP path); fall
    back to centres.

    Convention guard. In periodic-relative mode at ``σ/P > 0.03``,
    ``eval_orbit_rel`` integrates the JMM Eq. 3.4 form while the
    centres path computes the single-nearest-image-wrap form.
    The two diverge at this regime; centres remains the canonical
    output for backward compatibility.

    Parameters
    ----------
    r : int
        Tensor order.
    K : int
        Number of source events (``len(p)``).
    n_q : int
        Number of query points.
    is_rel, is_per : bool
        Mode flags.
    sigma_over_P : float
        ``σ / period``; ignored if not periodic.
    user_method : str
        One of ``'auto'``, ``'centres'``, ``'mobius'``. Internal callers
        may also pass ``'direct'`` as a synonym for ``'centres'``.

    Returns
    -------
    str
        ``'mobius'`` or ``'centres'``.
    """
    if user_method in ('centres', 'direct'):
        return 'centres'
    if user_method == 'mobius':
        return 'mobius'
    if user_method != 'auto':
        raise ValueError(
            f"method must be 'auto', 'centres', or 'mobius'; got "
            f"{user_method!r}."
        )
    # r=1: the Möbius machinery reduces to the direct Σ_i w_i K_i sum
    # (one partition with μ=1). Centres path coincides; pick centres
    # for code simplicity.
    if r <= 1:
        return 'centres'
    # Relative mode: eval_orbit_rel performs u-grid quadrature with
    # N_u ~ max(64, P/σ * 10) per query. The per-query cost is
    # O(B_r · r · K · N_u), much larger than the centres path's
    # O(n_j) per query at typical σ/P (~0.025 → N_u ≈ 360, vs n_j
    # of 100s to 1000s for r in {3, 4}). The Möbius relative-mode
    # evaluator is only ever cheaper at very high r combined with very
    # large K and large σ — a corner case that's safer to route via
    # explicit method='mobius'. Default to centres for rel mode.
    if is_rel:
        return 'centres'
    # r=2 with small K: centres is competitive and avoids the
    # partition-table dispatch overhead.
    if r == 2 and K <= 8:
        return 'centres'
    # Beyond shipped orbit tables: the eval_orbit_* helpers use
    # set-partition machinery rather than orbit tables, so they work
    # at any r in principle, but we defer to centres for consistency
    # with the IP-path policy. At r > _ORBIT_R_MAX_SHIPPED the orbit
    # table would build on demand, which the cost-preview helper warns
    # about; the eval dispatcher prefers the always-fast centres path.
    if r > _ORBIT_R_MAX_SHIPPED:
        return 'centres'
    # K-vs-r precision guard. Without K - r >= 2 the Möbius method's
    # alternating sum can lose all significant digits.
    if not _orbit_safe_for_precision([r], [K]):
        return 'centres'
    return 'mobius'



# -----------------------------------------------------------------------
# Unified method-selection + time-estimate probe
#
# The probe-based dispatcher replaces the heuristic rule for the
# discretionary cases. Genuinely hard rules (correctness / feasibility)
# stay as rules; everything else is decided by timing both methods on a
# small probe and picking the faster. The probe time also produces the
# user-facing time estimate, so dispatcher and estimator share a single
# load-bearing measurement that auto-adapts to any future optimisation.
# -----------------------------------------------------------------------


def _format_time(t_sec: float) -> str:
    """Human-readable short form of a duration in seconds."""
    if t_sec < 1:
        return f"{t_sec * 1000:.0f} ms"
    if t_sec < 60:
        return f"{t_sec:.1f} s"
    if t_sec < 3600:
        return f"{t_sec / 60:.1f} min"
    return f"{t_sec / 3600:.1f} hr"



# Probing parameters.
_PROBE_MIN_N_Q = 200    # below this many queries, skip probing entirely

_PROBE_N = 50           # probe sample size

# Centres-path memory budget (bytes). The probe refuses to materialise
# the centres array if it would exceed this; the Möbius method is chosen instead.
_CENTRES_PROBE_MEM_BUDGET = 4 * 1024**3


# Above this r, the Möbius method becomes infeasible: B_r (Bell numbers)
# explodes from 115,975 at r=10 to 5x10^13 at r=20, and set-partition
# enumeration itself blows the Python recursion stack. r > this falls back
# to centres-only routing.
_ORBIT_R_MAX_FEASIBLE = 10


# Bell numbers up to r=10 (set partition counts). Used by the rel-mode
# pre-screen to estimate the Möbius relative-mode cost without enumerating
# partitions.
_BELL_NUMBERS = {
    1: 1, 2: 2, 3: 5, 4: 15, 5: 52, 6: 203, 7: 877,
    8: 4140, 9: 21147, 10: 115975,
}


# Pre-screen: if one method is favoured by more than this factor, skip
# probing entirely. Two pre-screens, one per mode:
#
#  - Rel-mode pre-screen: routes TO centres when centres clearly wins.
#    The Möbius relative-mode evaluator does u-grid quadrature with N_u
#    sub-evals per query, so its PROBE is expensive (a 50-query probe at
#    N_u=1000 is ~3 s); a generous margin here avoids unnecessary probe
#    overhead.
#  - Abs-mode pre-screen: routes TO the Möbius method when it clearly wins.
#    For abs mode, centres cost per query is K^r vs Möbius cost
#    B_r * r * K. The Möbius method wins by a factor K^(r-1) / (B_r * r);
#    for K=72 r=3 that's ~1000x. The tiny-workload shortcut would
#    otherwise force centres for n_q<200 even at these large K, so the
#    pre-screen must run BEFORE the tiny shortcut. Pattern-finding and
#    other common music-cog tasks legitimately use abs mode at large K.
#
# The dominance margins are conservative — probe still has the final
# word when the cost ratio is in the uncertain region.
_PRESCREEN_CENTRES_DOMINANCE = 3.0

_PRESCREEN_ORBIT_DOMINANCE = 3.0



def _estimate_centres_array_bytes(K: int, r: int, is_rel: bool) -> int:
    """Estimate the dominant centres-array allocation in bytes.

    Returns ``K!/(K-r)! * dim * 8`` where ``dim`` is the effective
    centres dimensionality (``r`` for abs, ``r-1`` for rel).
    """
    if K < r:
        return 0
    n_j = 1
    for k in range(K - r + 1, K + 1):
        n_j *= k
    dim = r - 1 if is_rel else r
    return n_j * max(dim, 1) * 8



def _probe_eval_path(
    dens: "ExpTensDensity",
    x_probe: np.ndarray,
    path: str,
    *,
    truncation_sigmas: float | None,
    kernel_precision: str | None,
) -> float:
    """Time a small slice of the real eval path. Returns seconds.

    Runs the path twice on ``x_probe``: a warmup pass (discarded) to
    stabilise CPU caches, NumPy JIT state, and one-shot table loads,
    then a timed pass. Without the warmup, whichever path ran most
    recently on the full workload comes into the probe with hot caches
    and gets unfairly favoured; the dispatcher would then deterministically
    flip back to the other path on subsequent calls with identical inputs.
    """
    # Lazy import to break the dispatch <-> eval cycle: dispatch is
    # imported by eval at module-load time; eval cannot reciprocate
    # without circularity.
    from .eval import _eval_exp_tens_sa_centres, _eval_exp_tens_sa_orbit

    import time as _time
    if path == "centres":
        fn = _eval_exp_tens_sa_centres
    else:
        fn = _eval_exp_tens_sa_orbit

    # Warmup pass (discarded).
    fn(
        dens, x_probe, x_probe.shape[1],
        truncation_sigmas=truncation_sigmas,
        kernel_precision=kernel_precision,
        verbose=False,
    )

    # Timed pass.
    t0 = _time.perf_counter()
    fn(
        dens, x_probe, x_probe.shape[1],
        truncation_sigmas=truncation_sigmas,
        kernel_precision=kernel_precision,
        verbose=False,
    )
    return _time.perf_counter() - t0



def _select_and_estimate_sa(
    dens: "ExpTensDensity",
    x: np.ndarray,
    n_q: int,
    *,
    method: str,
    truncation_sigmas: float | None,
    kernel_precision: str | None,
    verbose: bool,
) -> tuple[str, bool, float, str]:
    """Unified path selection + time estimate for SA eval_exp_tens.

    Hard rules decide first:
      1. user override → honour it.
      2. r <= 1 → centres (Möbius method mathematically degenerate).
      3. K - r < _ORBIT_K_MINUS_R_MIN → centres (orbit cancellation).
      4. centres-array memory > budget → Möbius method (centres infeasible).

    Everything else is decided by probing both paths on a small slice
    of queries and picking the faster. The probe time, extrapolated to
    the full workload, is the user-facing time estimate.

    Returns (chosen, probed, est_sec, routing_reason). routing_reason
    is a short string describing why the path was chosen (e.g.
    'r = 1', 'rel-mode pre-screen', 'probe'); the caller uses it to
    emit a dispatch message via :func:`_maybe_show_dispatch_msg`.
    """
    r = int(dens.r)
    K = int(dens.p.shape[0])
    is_rel = bool(dens.is_rel)

    # ---- Rule 1: user override ----
    if method in ("centres", "direct"):
        return "centres", False, 0.0, "user override"
    if method == "mobius":
        return "mobius", False, 0.0, "user override"
    if method != "auto":
        raise ValueError(
            f"method must be 'auto', 'centres', or 'mobius'; got {method!r}."
        )

    # ---- Rule 2: Möbius method degenerate at r <= 1 ----
    if r <= 1:
        return "centres", False, 0.0, f"r = {r}"

    # ---- Rule 3: Möbius cancellation guard ----
    if not _orbit_safe_for_precision([r], [K]):
        return "centres", False, 0.0, f"K - r = {K - r} < 2"

    # ---- Rule 4: centres memory budget ----
    centres_bytes = _estimate_centres_array_bytes(K, r, is_rel)
    if centres_bytes > _CENTRES_PROBE_MEM_BUDGET:
        # Centres infeasible. Orbit is the only candidate, but it has
        # its own r-limit (B_r explodes; r > ~10 is impractical).
        if r > _ORBIT_R_MAX_FEASIBLE:
            raise ValueError(
                f"eval_exp_tens: r={r} requires more than "
                f"{_CENTRES_PROBE_MEM_BUDGET // 1024**3} GB for the "
                f"centres array (K={K}), and the Möbius method is infeasible at "
                f"r > {_ORBIT_R_MAX_FEASIBLE} (B_r explodes). Reduce "
                f"r or check inputs."
            )
        return "mobius", False, 0.0, "centres memory budget exceeded"

    # ---- Abs-mode pre-screen: route TO the Möbius method when it clearly wins ----
    # For abs mode, centres cost per query is K^r (materialised density
    # has n_j = K^r tuples), and Möbius absolute-mode per-query cost is B_r * r * K
    # (sum over B_r partitions of K*m per block, summing to K*r per
    # partition). The ratio is K^(r-1) / (B_r * r); for K=72 r=3 it's
    # ~1000x, meaning the tiny-workload shortcut below would otherwise
    # force centres for n_q<200 even when the Möbius method is 1000x faster.
    #
    # This pre-screen must run BEFORE the tiny-workload shortcut so
    # large-K abs-mode workloads (common in pattern-finding and other
    # music-cog tasks at typical 24-72-partial harmonic templates) get
    # the cheap routing decision they deserve at any n_q.
    #
    # Probe still has the final word in the uncertain region; this only
    # fires when the Möbius method wins by a comfortable margin.
    if (not is_rel) and r >= 2 and r <= _ORBIT_R_MAX_FEASIBLE:
        B_r = _BELL_NUMBERS[r]
        centres_cost = float(K) ** r
        orbit_cost = float(B_r) * r * float(K)
        if orbit_cost * _PRESCREEN_ORBIT_DOMINANCE < centres_cost:
            return "mobius", False, 0.0, "abs-mode pre-screen"

    # ---- Shortcut: tiny workload, skip probing ----
    if n_q < _PROBE_MIN_N_Q:
        return "centres", False, 0.0, f"n_q = {n_q} < {_PROBE_MIN_N_Q}"

    # ---- Rel-mode pre-screen: route TO centres when centres clearly wins ----
    # The probe is robust but not free. For rel mode in particular,
    # the Möbius relative-mode evaluator does u-grid quadrature with N_u ≈ max(64, 10·P/σ)
    # sub-evals per query — its PROBE cost scales as
    # B_r · r · K · N_u · n_probe, which is prohibitive when N_u is
    # large. We pre-screen the cost ratio analytically and skip the
    # probe if centres clearly wins. The probe still has the final
    # word in the uncertain region.
    if is_rel and r >= 2:
        # Estimate N_u (the Möbius relative-mode u-grid size) using the same
        # formula eval_orbit_rel uses internally.
        sigma = float(dens.sigma)
        if dens.is_per:
            N_u_est = max(64, int(np.ceil(
                10.0 * float(dens.period) / sigma
            )))
        else:
            # Non-periodic u-grid: covers [p.min() - x.max() - 8σ,
            # p.max() - x.min() + 8σ]. Use the actual data extents.
            p_min = float(np.min(dens.p))
            p_max = float(np.max(dens.p))
            x_min_abs = float(np.min(x, initial=0.0))
            x_max_abs = float(np.max(x, initial=0.0))
            u_min = p_min - max(0.0, x_max_abs) - 8.0 * sigma
            u_max = p_max - min(0.0, x_min_abs) + 8.0 * sigma
            N_u_est = max(
                64,
                int(np.ceil(max(u_max - u_min, 1.0) / sigma * 10.0)),
            )
        B_r = _BELL_NUMBERS.get(r, 10 ** 9)
        centres_cost = float(K) ** (r - 1)
        orbit_cost = float(B_r) * r * N_u_est
        if centres_cost * _PRESCREEN_CENTRES_DOMINANCE < orbit_cost:
            return "centres", False, 0.0, "rel-mode pre-screen"

    # ---- Probe both paths ----
    # Warm the set-partition cache so the Möbius probe doesn't pay
    # one-time table-build cost. Skip for high r where the Möbius method is not a
    # realistic candidate — set-partition enumeration itself becomes
    # infeasible, and the recursion depth grows linearly in r.
    if 2 <= r <= _ORBIT_R_MAX_FEASIBLE:
        from .._mobius import get_set_partitions_with_mobius
        get_set_partitions_with_mobius(r)
    if r > _ORBIT_R_MAX_FEASIBLE:
        # No Möbius option at this r; skip the probe and use centres.
        return ("centres", False, 0.0,
                f"r = {r} > {_ORBIT_R_MAX_FEASIBLE} (Möbius infeasible)")

    n_probe = min(_PROBE_N, n_q)
    sample_idx = np.linspace(0, n_q - 1, n_probe).astype(int)
    x_probe = x[:, sample_idx]

    t_centres = _probe_eval_path(
        dens, x_probe, "centres",
        truncation_sigmas=truncation_sigmas,
        kernel_precision=kernel_precision,
    )
    t_orbit = _probe_eval_path(
        dens, x_probe, "mobius",
        truncation_sigmas=truncation_sigmas,
        kernel_precision=kernel_precision,
    )

    if t_centres <= t_orbit:
        chosen, t_probe = "centres", t_centres
    else:
        chosen, t_probe = "mobius", t_orbit

    est_sec = t_probe * (n_q / n_probe)
    return chosen, True, est_sec, "probe"



# -----------------------------------------------------------------------
# Probe-based dispatcher for the SA cos_sim_exp_tens IP path.
#
# Parallels :func:`_select_and_estimate_sa` for the inner-product side:
# hard rules first (correctness / feasibility), then an analytical
# pre-screen (catches clear-winner cases without paying probe overhead),
# then actually time both paths on a small subset of each density and
# pick the faster. The probe time, extrapolated to the full workload,
# becomes the user-facing time estimate (printed in verbose mode) and
# auto-adapts to future optimisations of either path.
#
# Probe extrapolation. Pairwise IP cost scales as
# ``falling_factorial(K_x, r) * falling_factorial(K_y, r)`` (ordered
# r-tuple enumeration on each side). Orbit IP cost scales as
# ``B_r * K_x * K_y`` (kernel matrix construction + per-partition
# einsum). The probe uses ``K_probe = min(K_x, K_y, _PROBE_K_IP_TARGET)``
# events from each side and extrapolates by the appropriate factor.
# -----------------------------------------------------------------------

# Target subset size for the IP probe. Small enough that probe cost is
# negligible, large enough that the K_probe-choose-r tuple count is
# meaningful (e.g., 12-choose-3 = 220) and the Möbius method's precision
# guard (n_min - r >= 2) is not contended. ``K_probe`` is capped to
# ``min(K_x, K_y)`` at call time; the hard precision rule
# (``n_min - r < 2``) fires upstream so K_probe never drops below r+2.
_PROBE_K_IP_TARGET = 12


# Pre-screen: skip the probe if one path's analytical cost dominates
# the other by this margin. Mirrors the eval-side pre-screen
# constants.
_PRESCREEN_IP_DOMINANCE = 3.0



def _falling_factorial(n: int, k: int) -> float:
    """``n * (n-1) * ... * (n-k+1)``; 0 if any factor is non-positive."""
    if n < k:
        return 0.0
    prod = 1.0
    for i in range(k):
        prod *= (n - i)
    return prod



def _probe_ip_path(
    dens_x: "ExpTensDensity",
    dens_y: "ExpTensDensity",
    K_probe: int,
    path: str,
    *,
    truncation_sigmas: float | None,
    kernel_precision: str | None,
) -> float:
    """Time one cos_sim_exp_tens IP path on the first ``K_probe`` events
    of each density. Returns seconds.

    Builds fresh subset densities outside the timed window so the
    measurement covers only the IP work itself (kernel-matrix
    construction + einsums for the Möbius method, or ordered-tuple
    enumeration + dot product for Bulger's method).

    Runs the work twice: a warmup pass (discarded) to stabilise CPU
    caches and one-shot table loads, then a timed pass. Without the
    warmup, the path that ran most recently on the full workload comes
    into the probe with hot caches and gets unfairly favoured.
    """
    # Lazy imports to break the dispatch <-> cosine and
    # dispatch <-> build cycles (dispatch is imported by both).
    from .build import build_exp_tens
    from .cosine import _cos_sim_exp_tens_sa_orbit, _cos_sim_exp_tens_sa_pairwise

    import time as _time

    sub_x = build_exp_tens(
        dens_x.p[:K_probe], dens_x.w[:K_probe],
        dens_x.sigma, int(dens_x.r),
        bool(dens_x.is_rel), bool(dens_x.is_per), float(dens_x.period),
        verbose=False,
    )
    sub_y = build_exp_tens(
        dens_y.p[:K_probe], dens_y.w[:K_probe],
        dens_y.sigma, int(dens_y.r),
        bool(dens_y.is_rel), bool(dens_y.is_per), float(dens_y.period),
        verbose=False,
    )

    def _run() -> None:
        if path == "mobius":
            _cos_sim_exp_tens_sa_orbit(sub_x, sub_y)
        else:
            _cos_sim_exp_tens_sa_pairwise(
                sub_x, sub_y, verbose=False,
                truncation_sigmas=truncation_sigmas,
                kernel_precision=kernel_precision,
            )

    # Warmup pass (discarded).
    _run()

    # Timed pass.
    t0 = _time.perf_counter()
    _run()
    return _time.perf_counter() - t0



def _select_and_estimate_sa_ip(
    dens_x: "ExpTensDensity",
    dens_y: "ExpTensDensity",
    *,
    method: str,
    truncation_sigmas: float | None,
    kernel_precision: str | None,
    verbose: bool,
) -> tuple[str, bool, float, str]:
    """Probe-based dispatcher for SA cos_sim_exp_tens IP path.

    Hard rules decide first:
      1. user override → honour it.
      2. r <= 1 → Bulger (Möbius method degenerate at r=1).
      3. r > _ORBIT_R_MAX_SHIPPED → pairwise (build cost).
      4. n_min - r < _ORBIT_K_MINUS_R_MIN → pairwise (orbit cancellation).
      5. periodic-relative beyond σ/P threshold → Bulger (convention).

    Then analytical pre-screen catches clear-winner cases without
    paying probe overhead. Otherwise, both paths are timed on a small
    subset (``min(K_x, K_y, _PROBE_K_IP_TARGET)``) and extrapolated to
    the full workload; the faster is picked.

    Returns ``(chosen, probed, est_sec, routing_reason)``. routing_reason
    is a short string describing why the path was chosen; the caller
    uses it to emit a dispatch message via
    :func:`_maybe_show_dispatch_msg`.
    """
    r = int(dens_x.r)
    K_x = int(dens_x.p.shape[0])
    K_y = int(dens_y.p.shape[0])
    n_min = min(K_x, K_y)
    is_rel = bool(dens_x.is_rel)
    is_per = bool(dens_x.is_per)
    sigma = float(dens_x.sigma)
    period = float(dens_x.period)
    sigma_over_P = sigma / period if (is_per and period > 0) else 0.0

    # ---- Hard rules ----
    if method in ("bulger", "direct"):
        return "bulger", False, 0.0, "user override"
    if method == "mobius":
        return "mobius", False, 0.0, "user override"
    if method != "auto":
        raise ValueError(
            f"method must be 'auto', 'bulger', 'direct', or 'mobius'; "
            f"got {method!r}."
        )
    if r <= 1:
        return "bulger", False, 0.0, f"r = {r}"
    if r > _ORBIT_R_MAX_SHIPPED:
        return "bulger", False, 0.0, f"r = {r} > {_ORBIT_R_MAX_SHIPPED} (Möbius infeasible)"
    if not _orbit_safe_for_precision([r], [n_min]):
        return "bulger", False, 0.0, f"min(K_x, K_y) - r = {n_min - r} < 2"
    if is_rel and is_per and sigma_over_P > _ORBIT_SIGMA_OVER_P_THRESHOLD:
        warnings.warn(
            f"σ/P = {sigma_over_P:.3f} exceeds the Möbius-method threshold "
            f"({_ORBIT_SIGMA_OVER_P_THRESHOLD}) for relative-periodic mode; "
            f"falling back to Bulger's method (the pairwise-wrap form). Pass method='bulger' "
            f"explicitly to silence this warning."
        )
        return "bulger", False, 0.0, "sigma/period > 0.03 (rel-per Möbius fallback)"
    if r > _ORBIT_R_MAX_FEASIBLE:
        return "bulger", False, 0.0, f"r = {r} > {_ORBIT_R_MAX_FEASIBLE} (Möbius infeasible)"

    # ---- Analytical cost models ----
    pairwise_full = _falling_factorial(K_x, r) * _falling_factorial(K_y, r)
    B_r = float(_BELL_NUMBERS[r])
    orbit_full = B_r * float(K_x) * float(K_y)

    # ---- Analytical pre-screen ----
    if orbit_full * _PRESCREEN_IP_DOMINANCE < pairwise_full:
        return "mobius", False, 0.0, "cost pre-screen"
    if pairwise_full * _PRESCREEN_IP_DOMINANCE < orbit_full:
        return "bulger", False, 0.0, "cost pre-screen"

    # ---- Probe both paths on a subset ----
    # Warm the orbit partition table so the Möbius probe doesn't pay a
    # one-time table-build cost.
    from .._mobius import get_set_partitions_with_mobius
    get_set_partitions_with_mobius(r)

    K_probe = min(K_x, K_y, _PROBE_K_IP_TARGET)
    # K_probe - r >= 2 is guaranteed by the precision hard rule above
    # (n_min - r >= _ORBIT_K_MINUS_R_MIN), so the orbit probe is safe.

    t_pairwise = _probe_ip_path(
        dens_x, dens_y, K_probe, "bulger",
        truncation_sigmas=truncation_sigmas,
        kernel_precision=kernel_precision,
    )
    t_orbit = _probe_ip_path(
        dens_x, dens_y, K_probe, "mobius",
        truncation_sigmas=truncation_sigmas,
        kernel_precision=kernel_precision,
    )

    # ---- Extrapolate to full workload ----
    pairwise_probe = _falling_factorial(K_probe, r) ** 2
    pairwise_factor = (
        pairwise_full / pairwise_probe if pairwise_probe > 0 else 1.0
    )
    orbit_probe = float(K_probe) ** 2
    orbit_factor = (
        float(K_x) * float(K_y) / orbit_probe if orbit_probe > 0 else 1.0
    )

    t_pairwise_est = t_pairwise * pairwise_factor
    t_orbit_est = t_orbit * orbit_factor

    if t_pairwise_est <= t_orbit_est:
        return "bulger", True, t_pairwise_est, "probe"
    return "mobius", True, t_orbit_est, "probe"