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
from .density import (
    MaetDensity, WindowedMaetDensity,
)




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

    if isinstance(arg, (MaetDensity, WindowedMaetDensity)):
        return True, (arg,)

    if isinstance(arg, (list, tuple)):
        if len(arg) == 0:
            return False, ()
        for i, elem in enumerate(arg):
            if not isinstance(
                elem, (MaetDensity, WindowedMaetDensity)
            ):
                raise TypeError(
                    f"{name}[{i}] must be a MaetDensity "
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



# Per-r, per-mode K thresholds for the orbit-vs-enumeration crossover at a
# single symmetric attribute or nesting level, established empirically on
# representative MAET workloads (N = 8-12, σ = 12, P = 1200). Mode matters: the
# relative-periodic orbit path carries a transposition-average u-grid that
# enumeration avoids, so its crossover sits well above the absolute one, and at
# r = 2 enumeration always wins (hence the inf entry — a pure op-count
# comparison, which sees only the orbit-class reduction and not the u-grid
# cost, would wrongly route every r = 2 rel-per level to orbit). These drive
# the per-level decision in the nested contraction via
# _orbit_beats_pairwise_per_attr. The flat multi-attribute and single-multiset
# inner-product paths make a whole-call decision instead, through the cost
# model and probe below, which additionally account for N (e.g. the absolute
# r = 2 crossover ranges from K ≥ 14 at N = 2 to K ≥ 6 at N = 16, a span no
# single K threshold can capture); the per-level predicate stays
# threshold-based because it runs once per level with no room for a probe.
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
# Bulger's method: per-entry cost of the joint tuple-pair kernel in
# the cost model's n_J·n_K units (perm side × comb side; the r!
# asymmetry between the two sides is absorbed into the constant, so
# it rises mildly with r). Measured on the Python implementation at
# representative scale (K = 12, n_J·n_K = 1.4e7-4.2e7): r = 2
# ~110 ns, r = 3 ~145 ns non-periodic; the periodic wrap adds
# ~10-20 % (not the multiples an earlier small-workload calibration
# suggested). Values sit at the lower end of the measured band so
# near-crossover routing under-prices Bulger's method — the
# cheap-to-mispick side. Orders above 3 reuse the r = 3 value.
_PW_PER_ENTRY_MS_NONPER = {2: 1.0e-4, 3: 1.4e-4}

_PW_PER_ENTRY_MS_PER = {2: 1.1e-4, 3: 1.6e-4}



def _pw_per_entry_ms(any_per, r_max=3):
    """Per-entry cost for Bulger's method: wrap mode and tensor order."""
    table = _PW_PER_ENTRY_MS_PER if any_per else _PW_PER_ENTRY_MS_NONPER
    r_key = min(max(int(r_max), 2), max(table))
    return table[r_key]



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

# Möbius method (relative modes): each relative attribute's
# (event_X, event_Y) inner matrices are computed by whichever of two
# routes is cheaper per event pair (see
# cosine._ma_rel_attr_prefers_centres): the pairwise closed form over
# materialised tuple-centres at (r_a!·C(K_a, r_a))² kernel ops per
# pair, or the slab-batched translation-grid contraction at
# N_u·K_a² ops per pair. The cost model prices both with measured
# per-op constants and takes the same minimum the orchestrator takes;
# three matrices (cross plus both self-norms) per attribute. Constants
# measured on the Python implementation (rel-per, K = 4-5, r = 2-3,
# N = 20-120): centres ~87 ns per centre-pair op; grid contraction
# ~27 ns per kernel op (slab-resident).
_ORBIT_REL_BASE_MS = 5.0
_ORBIT_REL_CENTRES_OP_MS = {2: 4.0e-5, 3: 9.0e-5}
_ORBIT_REL_GRID_OP_MS = {2: 3.0e-5, 3: 5.0e-5}


def _orbit_rel_op_ms(table, r_a):
    """Per-op cost for a relative-attribute Möbius route at order r_a.

    Measured entries cover r = 2, 3 (centres ~35 and ~87 ns; grid ~27
    and ~48 ns, each set at the upper end of its measured band so the
    Möbius side is over-priced near crossovers — routing bias toward
    Bulger's method, the cheap-to-mispick side). Orders above the
    table extrapolate by doubling per order, conservative in the same
    direction."""
    r_key = min(max(int(r_a), 2), max(table))
    val = table[r_key]
    if r_a > max(table):
        val *= 2.0 ** (int(r_a) - max(table))
    return val



def _orbit_beats_pairwise_per_attr(r, K, is_rel, is_per):
    """Per-level orbit-vs-enumeration decision for one symmetric attribute or
    nesting level, from the mode-aware K thresholds above.

    This is the live predicate for the per-level decision in the nested
    contraction (``_nested_contraction._orbit_eligible``): it runs once per
    level, so it uses a cheap mode-aware threshold rather than a probe. The flat
    multi-attribute inner-product path instead makes a
    whole-call decision through the cost model (see
    ``_select_ma_inner_product_method``),
    which also weighs N. The two mechanisms are matched to their contexts, not
    redundant: the per-level predicate cannot afford a probe, and its thresholds
    encode the relative-periodic u-grid overhead that an op-count comparison
    would miss. ``True`` means orbit (Möbius) is the cheaper route here.
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
    r_vec, k_vec, A, N_x, N_y, rel_vec, nu_vec, centres_ok=True,
):
    """Predicted Möbius-method MA wall time in milliseconds.

    Per-attribute sum. Absolute attributes with r_a >= 2 cost the
    vectorised-batch constant for their order (r_a = 1 attributes are
    a single direct kernel sum, absorbed into the base). Relative
    attributes cost three (N_x, N_y)-shaped matrices at the cheaper of
    the two per-pair routes the orchestrator itself chooses between:
    the tuple-centres closed form ((r_a!·C(K_a, r_a))² ops per pair)
    or the slab-batched translation-grid contraction (nu_a·K_a² ops
    per pair, with nu_a the caller's per-attribute grid node
    estimate). A-linearity of the absolute constants is verified at
    r = 2, 3 to within ~5 % and slightly sub-linear at r = 4, where
    linear-A over-predicts conservatively, biasing the dispatcher
    toward Bulger's method in close calls.
    """
    total = _ORBIT_REL_BASE_MS if np.any(rel_vec) else 0.0
    pairs = float(N_x) * float(N_y)
    for a in range(A):
        r_a = int(r_vec[a])
        K_a = int(k_vec[a])
        if bool(rel_vec[a]) and r_a >= 2:
            # The centres route is measure-blocked above the sigma/P
            # threshold (the orchestrator keeps the all-image grid
            # there), so above it the grid route is priced alone.
            grid_ops = float(nu_vec[a]) * K_a * K_a
            per_pair = grid_ops * _orbit_rel_op_ms(
                _ORBIT_REL_GRID_OP_MS, r_a)
            if centres_ok:
                centres_ops = float(
                    factorial(r_a) * _math_comb(K_a, r_a)) ** 2
                per_pair = min(
                    per_pair,
                    centres_ops * _orbit_rel_op_ms(
                        _ORBIT_REL_CENTRES_OP_MS, r_a),
                )
            total += 3.0 * pairs * per_pair
        elif r_a >= 2:
            total += float(_ORBIT_ABS_PER_ATTR_MS[r_a])
    return float(total)



def _select_ma_inner_product_method(
    *,
    r_vec, k_vec, A,
    N_x, N_y,
    any_per, any_rel_nonper, any_rel_per,
    sigma_over_P_max, user_method,
    rel_vec=None, nu_vec=None,
    guard_forced_bulger=True,
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
    - rel (per and nonper): per-attribute cost model pricing each
      relative attribute at the cheaper of its two Möbius routes
      (tuple-centres closed form vs slab-batched translation-grid
      contraction; see `_predict_orbit_cost_ms`), against Bulger's
      `_PW_PER_ENTRY_MS_*`. The Möbius method's per-attribute
      factorisation grows additively across attributes where Bulger's
      ∏_a r_a!·C(K_a, r_a)² compounds, so multi-attribute relative
      densities route to the Möbius method once that compounding
      bites.

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
        if guard_forced_bulger:
            _guard_forced_bulger_feasible_ma(
                k_vec, r_vec, rel_vec, N_x, N_y,
                reason="r above the shipped orbit order",
            )
        return 'bulger'
    # K-vs-r precision guard. The Möbius method's auto-inner-products can
    # suffer catastrophic Möbius cancellation when any K_a is too close
    # to its r_a (see _ORBIT_K_MINUS_R_MIN block). The cross
    # cancellation guard at the call site does NOT catch this, since it
    # inspects only |<T_X,T_Y>|; corrupted <T_X,T_X> propagates silently
    # into the cosine denominator.
    if A > 0 and not _orbit_safe_for_precision(r_vec, k_vec):
        if guard_forced_bulger:
            _guard_forced_bulger_feasible_ma(
                k_vec, r_vec, rel_vec, N_x, N_y,
                reason="the K - r precision floor",
            )
        return 'bulger'
    # ---- Memory-safety guard (explicit invariant) ----
    # Bulger's MA IP materialises each side's joint perm-side working
    # set n_J = N · ∏_a r_a!·C(K_a, r_a) (lazy, built on first access);
    # the Möbius MA IP is n_J-free (per-event, per-attribute additive
    # work). As in the single-multiset IP path the cost model below already routes
    # large workloads to Möbius, because its Bulger cost keys on the
    # tuple-pair size n_J^X·n_J^Y — the square of the per-side working
    # set — so any density big enough to blow memory is diverted on cost
    # alone. This guard makes that invariant explicit: if either side's
    # perm-side working set exceeds the soft budget and Möbius is
    # convention-safe, take Möbius now. Precision and feasibility are
    # already ensured above; the per-side count reuses the exact
    # _predict_pairwise_kernel_size formula (n_J = N · ∏_a r_a!·C).
    if A > 0 and not (any_rel_per
                      and sigma_over_P_max > _ORBIT_SIGMA_OVER_P_THRESHOLD):
        per_side_tuples = 1.0
        dim_sum = 0
        for a in range(A):
            r_a = int(r_vec[a]); K_a = int(k_vec[a])
            per_side_tuples *= float(factorial(r_a)) * float(_math_comb(K_a, r_a))
            dim_sum += r_a
        n_J_max = max(int(N_x), int(N_y)) * per_side_tuples
        # working-set bytes ≈ n_J · (2·Σr_a) · 8 (perm + centres + index
        # arrays, mirroring the single-multiset row-factor), capped to avoid overflow.
        if n_J_max * (2 * max(dim_sum, 1)) * 8 > _CENTRES_WORKING_SET_SOFT_BUDGET:
            return 'mobius'
    # Relative-periodic measure note: the Möbius method computes the all-image
    # (JMM Eq. 3.4 transposition-integral) form; Bulger's method computes the
    # single-wrap (minimum-image) form. They diverge by O((σ/P)^∞) above
    # σ/P ≈ 0.03. The dispatch always takes the faster path (cost model below);
    # when that path is the all-image Möbius method and σ/P is above the
    # threshold (so the two measures differ), it warns and points to
    # method='bulger' for the canonical single-wrap measure.
    pw_size = _predict_pairwise_kernel_size(r_vec, k_vec, A, N_x, N_y)
    pw_cost_ms = pw_size * _pw_per_entry_ms(any_per, r_max)
    # Aggregate-only callers (the legacy selector API) supply no
    # per-attribute vectors; reconstruct conservative defaults. Marking
    # every r_a >= 2 attribute as relative whenever either rel flag is
    # set over-prices the Möbius side, biasing near-crossover routing
    # toward Bulger's method — the cheap-to-mispick side; the
    # representative grid size matters only where the grid route is
    # already the cheaper Möbius branch (large K), where routing is
    # decided by orders of magnitude, not the node count.
    if rel_vec is None:
        any_rel = any_rel_nonper or any_rel_per
        rel_vec = np.array(
            [any_rel and int(r_vec[a]) >= 2 for a in range(A)],
            dtype=bool,
        )
    if nu_vec is None:
        nu_vec = np.full(max(A, 1), 2000.0)[:A]
    orbit_cost_ms = _predict_orbit_cost_ms(
        r_vec, k_vec, A, N_x, N_y, rel_vec, nu_vec,
        centres_ok=(sigma_over_P_max <= _ORBIT_SIGMA_OVER_P_THRESHOLD),
    )
    if pw_cost_ms <= orbit_cost_ms:
        return 'bulger'
    if any_rel_per and sigma_over_P_max > _ORBIT_SIGMA_OVER_P_THRESHOLD:
        _warn_rel_per_all_image(sigma_over_P_max)
    return 'mobius'



def _compute_Q_inner_blocks(D, r_inner, is_per, period, *, reduced):
    """Block-diagonal quadratic form for the inner ``[rel]`` co-transposition unit.

    The inner unit removes each source event's own all-ones, leaving the
    within-event intervals tensor-joined across events (toolbox spec
    §6.3). Concretely the metric is block-diagonal: ``D``'s rows are
    ``r_outer`` event-blocks and the form is the sum over blocks of the
    per-block flat relative quotient :func:`_compute_Q` (``is_rel=True``).

    ``reduced=False``: each block is a full ``r_inner``-tuple, so ``D`` has
    ``r_outer * r_inner`` rows (the inner-product / cosine convention,
    which carries the full perm-side tuples). ``reduced=True``: each block
    is the ``(r_inner-1)``-row slot-0 reduction of an ``r_inner``-tuple, so
    ``D`` has ``r_outer * (r_inner-1)`` rows (the centres-array evaluation
    convention). The periodic pairwise wrap is applied inside
    :func:`_compute_Q`, so callers must *not* pre-wrap ``D`` for the inner
    unit.
    """
    block = r_inner if not reduced else (r_inner - 1)
    if block <= 0:
        # r_inner == 1: each event's interval space is trivial (dim 0).
        return np.zeros(D.shape[1:], dtype=D.dtype)
    n_blocks = D.shape[0] // block
    Q = np.zeros(D.shape[1:], dtype=D.dtype)
    for b in range(n_blocks):
        sl = slice(b * block, (b + 1) * block)
        Q = Q + _compute_Q(D[sl], r_inner, True, is_per, period,
                            reduced=reduced)
    return Q


def _inner_r_vec(dens):
    """Per-attribute co-transposition block size (0 where not block-metric).

    Returns an int array of length ``n_attrs`` whose entry *a* is the
    block size ``s_u = prod(r[:u+1])`` when attribute *a* is a nested
    attribute resolved to an ``inner`` or ``intermediate`` co-transposition
    unit *u* (innermost-outward, 0-based), and 0 otherwise (flat
    attributes, ``absolute``, and the whole-tuple ``outer`` unit, which
    rides the ordinary ``is_rel`` path). The block-diagonal metric removes
    each level-*u* sub-tuple's own all-ones; there are ``D_a / s_u`` such
    blocks. Used by the evaluation and inner-product paths to switch on
    :func:`_compute_Q_inner_blocks`.
    """
    A = int(dens.n_attrs)
    out = np.zeros(A, dtype=np.intp)
    nested = getattr(dens, "nested", None)
    if nested is None:
        return out
    for a in range(A):
        s = nested[a]
        if (s is not None and isinstance(s, dict)
                and s.get("proj") in ("inner", "intermediate")):
            r_levels = np.asarray(s["r"]).ravel()
            u = int(s["rel_unit"])
            out[a] = int(np.prod(r_levels[:u + 1]))   # block size s_u
    return out


def _quadratic_form_det(r, inner_r, is_rel) -> float:
    """Determinant of the relative-mode quadratic form ``M`` for one attribute.

    A single flat attribute of tuple order ``r`` in relative mode carries
    the metric ``M = I - e e^T / r`` (the all-ones removed once), whose
    determinant is ``1 / r``. A nested attribute whose active
    co-transposition unit has block size ``s_u`` (from
    :func:`_inner_r_vec`) removes an all-ones within each of its
    ``r / s_u`` block-diagonal blocks, giving ``(1 / s_u) ^ (r / s_u)``.
    Absolute mode (and the vacuous ``r = 1`` relative case) has ``M = I``,
    determinant ``1``.

    Parameters mirror the three-way branch every normalisation site used
    to inline: pass the attribute's tuple order ``r``, its block size
    ``inner_r`` (0 when flat), and its ``is_rel`` flag.
    """
    r = int(r)
    inner_r = int(inner_r)
    if inner_r >= 2:
        return (1.0 / float(inner_r)) ** (r // inner_r)
    if bool(is_rel) and r >= 2:
        return 1.0 / float(r)
    return 1.0


def _gaussian_mass_const(sigma, dim, det_m, *, half: bool = False) -> float:
    """Gaussian normalisation constant ``(c pi sigma^2)^(dim/2) / sqrt(det M)``.

    With ``half=False`` (default) ``c = 2``: the single-kernel *mass*,
    the integral of one un-normalised Gaussian ``exp(-Q_M(d) / (2
    sigma^2))`` over the ``dim``-dimensional attribute space. With
    ``half=True`` ``c = 1``: the *overlap* form ``(pi sigma^2)^(dim/2) /
    sqrt(det M)``, the mass of the product of two such kernels (used by
    the entropy read-outs). ``det_m`` comes from
    :func:`_quadratic_form_det`.

    Density values are divided by the mass to normalise to unit peak or
    to a pdf; the eval and harmony paths equivalently multiply by its
    reciprocal, which is ``(2 pi sigma^2)^(-dim/2) * sqrt(det M)``.
    """
    c = 1.0 if half else 2.0
    return (c * np.pi * float(sigma) ** 2) ** (float(dim) / 2.0) / np.sqrt(det_m)


def _compute_Q(D, r, is_rel, is_per, period, *, reduced=False):
    """Compute the quadratic form from differences D.

    Two input conventions, controlled by ``reduced``:

    - ``reduced=False`` (default): ``D`` has *r* components representing
      a full r-tuple ``d = (d_0, …, d_{r-1})``. This is what the
      inner-product (cosine-similarity) path uses, where centres are
      stored as full r-tuples.
    - ``reduced=True``: ``D`` has *r-1* components representing the
      "slot 0 anchored" reduction ``D[k] = d_{k+1} − d_0`` of an
      r-tuple. This is what the single-multiset centres-array evaluation path uses,
      where ``centres = u_perm[1:] − u_perm[0]`` are stored in
      effective coordinates. Only meaningful when ``is_rel=True``;
      ignored for absolute mode (which has dim = r and is unaffected).

    Four cases of (is_rel, is_per):

    - **abs (is_rel=False):** ``Q = sum(D**2, axis=0)``. The caller
      must pre-wrap D into the principal period interval when
      ``is_per``. ``reduced`` is ignored here.
    - **rel non-periodic:** ``Q = sum(D**2) - sum(D)**2 / r``. The
      same formula serves both conventions, because the algebraic
      identity that produces it doesn't depend on whether the
      first-slot zero is materialised.
    - **rel periodic (Eq 6 of the preprint):** the pairwise-wrap
      form, ``Q = sum_{i<j over r slots} wrap(d_i - d_j)**2 / r``.
      Pairwise wrapping (not component-wise outer wrap) is what
      preserves exact transposition invariance on the circle. In
      reduced form the implicit slot 0 contributes pairs
      ``(0, k+1) -> -D[k]`` together with the within-reduced-block
      pairs ``(i+1, j+1) -> D[i] - D[j]``; both sets are wrapped and
      accumulated.

    The result dtype matches ``D.dtype`` (relevant for callers that
    pass single-precision tensors via ``kernel_precision='single'``).
    """
    if is_rel:
        if is_per:
            dt = D.dtype
            p_g = dt.type(period)
            half = dt.type(0.5)
            if reduced:
                # Slot-0 pairs (0, k+1) for k = 0..r-2: each contributes
                # wrap(D[k])^2 (since wrap(-x)^2 = wrap(x)^2). Vectorise
                # the wrap-and-sum across all slot-0 pairs in one numpy
                # pass on D as a whole.
                slot0_wrapped = D - p_g * np.floor(D / p_g + half)
                Q = np.sum(slot0_wrapped ** 2, axis=0)
                inner_range = range(r - 1)
            else:
                # Full r-tuple representation: all (r choose 2) pairs.
                Q = np.zeros(D.shape[1:], dtype=dt)
                inner_range = range(r)
            # Inner pairs (i+1, j+1) for 0 <= i < j <= len(inner_range)-1.
            # Pre-allocate a workspace and use in-place ops to avoid
            # per-iteration array allocation. At high r this dominates
            # over the underlying arithmetic.
            inner_pair_count = (r - 1) * (r - 2) // 2 if reduced else r * (r - 1) // 2
            if inner_pair_count > 0:
                delta = np.empty(D.shape[1:], dtype=dt)
                tmp = np.empty(D.shape[1:], dtype=dt)
                for i in inner_range:
                    for j in range(i + 1, inner_range.stop):
                        np.subtract(D[i], D[j], out=delta)
                        np.divide(delta, p_g, out=tmp)
                        np.add(tmp, half, out=tmp)
                        np.floor(tmp, out=tmp)
                        np.multiply(tmp, p_g, out=tmp)
                        np.subtract(delta, tmp, out=delta)
                        np.multiply(delta, delta, out=delta)
                        np.add(Q, delta, out=Q)
            Q = Q / dt.type(r)
        else:
            Q = (np.sum(D**2, axis=0)
                 - np.sum(D, axis=0) ** 2 / D.dtype.type(r))
    else:
        Q = np.sum(D**2, axis=0)
    return Q



# -------------------------------------------------------------------
#  Möbius method dispatcher (single-multiset path)
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
#                     In the single-multiset path, 'direct' coincides
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


def _warn_rel_per_all_image(sigma_over_P, *, operation="inner product",
                            canonical_method="bulger", stacklevel=3):
    """Warn that the dispatch took the faster all-image form in
    relative-periodic mode, which above the σ/P threshold differs from
    the canonical single-image measure.

    The all-image (transposition-integral) form is simply what the
    Möbius method computes in relative-periodic mode --- at every σ/P,
    not only above the threshold. Below the threshold it closely
    approximates the canonical single-image (minimum-image) measure and
    is cheaper, so it is preferred with no warning. Above the threshold
    it is a genuinely different measure, so its selection is announced
    here, with the single-image measure available on demand via
    ``method=canonical_method`` (``'bulger'`` for the inner product,
    ``'centres'`` for evaluation).

    Emitted by every relative-periodic path --- flat single-multiset,
    flat multi-attribute, the nested contraction, and evaluation --- so
    the message is uniform wherever the substitution happens.
    """
    warnings.warn(
        f"σ/P = {sigma_over_P:.3f} exceeds {_ORBIT_SIGMA_OVER_P_THRESHOLD}: "
        f"the faster all-image (transposition-integral) form of the "
        f"relative-periodic {operation} has been used. Above this σ/P it "
        f"differs from the canonical single-image (minimum-image) measure "
        f"(the two agree below it). To compute the single-image measure "
        f"instead, pass method='{canonical_method}', which takes the exact "
        f"route this fast path avoids; that route can be substantially "
        f"slower, and infeasible for a large or high-order collection "
        f"(precisely the case that made the all-image form the faster path "
        f"here).",
        stacklevel=stacklevel,
    )


class SingleImageInfeasibleError(MemoryError):
    """Raised when the single-image (minimum-image) measure is the only
    available route but its materialisation would exhaust memory.

    Arises at high tuple order in relative-periodic (and, for the inner
    product, any) mode when the Möbius method is *unavailable* --- refused
    by the precision floor (``K - r`` below the guard) or the feasibility
    bound (``r`` above the shipped/feasible orbit order) --- so no cheaper
    all-image substitute exists, and the exact single-image route
    (``centres`` for evaluation, ``bulger`` for the inner product) would
    need an infeasibly large tuple(-pair) kernel. There is no correct
    cheaper answer to fall back to (unlike the σ/P convention case, where
    the all-image form is a legitimate cheaper measure), so the honest
    outcome is a clear error rather than an out-of-memory crash.
    """


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
    """Return True if every r_a >= 2 attribute satisfies K_a >= r_a + margin.

    Used by both the single-multiset and multi-attribute dispatchers to refuse the Möbius method
    when its Möbius cancellation could swamp the answer. See the
    `_ORBIT_K_MINUS_R_MIN` rationale block above.

    The margin applies only to attributes with r_a >= 2: the Möbius
    alternating sum over set partitions is trivial at r_a = 1 (a single
    partition, no signs), so an r_a = 1 attribute carries no
    cancellation risk regardless of its K_a. Scalar attributes
    (K_a = 1, r_a = 1) are the canonical multi-attribute pattern — an
    onset or duration alongside pitch content — and must not veto the
    Möbius method for the density.
    """
    r_arr = np.atleast_1d(np.asarray(r_vec, dtype=np.intp))
    k_arr = np.atleast_1d(np.asarray(k_vec, dtype=np.intp))
    mask = r_arr >= 2
    return bool(np.all(k_arr[mask] - r_arr[mask] >= _ORBIT_K_MINUS_R_MIN))



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


# Soft centres working-set budget (bytes). Distinct from the 4 GB hard
# ceiling above: that ceiling only fires when the centres array alone
# exceeds it, and it estimates the *final* centres array only. But
# _build_perm_arrays materialises the full ordered-tuple working set
# (j_idx, u_perm, and centres each scale as n_j = K!/(K-r)!), so the true
# peak is several times the centres-array estimate. When that working set
# is large but still under the hard ceiling, the centres path is feasible
# on op-count yet allocates hundreds of MB — which the rel-mode cost
# model, being wall-time oriented, does not see. On memory-constrained
# machines this thrashes. The Möbius point evaluator is n_j-free (its cost
# is at most B_r · r · K · N_u per query, and less when its factored
# strategy engages), so when the centres working set exceeds
# this soft budget and Möbius is feasible and convention-safe, we route to
# Möbius to keep peak memory bounded. Set generously so the validated
# benchmark cells (rel r <= 4 at moderate K, working set < ~10 MB) are
# never perturbed; it fires only for the large-template regime (e.g.
# tensor_harmonicity at duplicate >= 4, where a 12-partial template
# becomes K = 48 partials and n_j ~ 4.7e6).
_CENTRES_WORKING_SET_SOFT_BUDGET = 256 * 1024**2


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
#    nodes (per-node cost K per block on its direct strategy; K-free on
#    its factored strategy, which its internal gate prefers for batches)
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

#: Calibrated constants for the MA eval cost model, in milliseconds.
#: Fitted to the selection-quality grid (single-attribute, r = 2..4,
#: K = 6..48, all four mode combinations, n_q = 1 and 200, sigma = 15
#: and 100 cents over spans of 1200--3600 cents; July 2026 harness).
#: Absolute values are machine-specific; selection depends only on
#: their ratios, which are stable across the grid. MATLAB carries its
#: own constants in ``internal.selectMaEval`` (same functional form,
#: per-language calibration): its non-periodic centres kernel culls more
#: aggressively, so there the joint-count growth surfaces in the per-call
#: term rather than the per-query one, but the form is identical.
#:
#: Centres: a per-call materialisation term and a per-query kernel term,
#: both linear in the joint tuple count. The non-periodic kernel is
#: bucket-culled, the periodic kernel dense; at the calibration scales
#: their per-tuple constants are of the same order, so one pair of
#: constants serves both (the cull's growing advantage at very large
#: shapes only strengthens a centres pick already made).
_MA_COST_CENTRES_SETUP_MS = 0.15
_MA_COST_CENTRES_CALL_PER_JOINT_MS = 4e-5
_MA_COST_CENTRES_QUERY_PER_JOINT_MS = 4.5e-5

#: Per-attribute per-query overhead of the factored centres route
#: (bucket lookup and gather in the culled per-attribute kernel),
#: fitted to measured wall times of that route.
_MA_COST_CENTRES_FACTORED_QUERY_BASE_MS = 1.5e-3
# Culling onset for the centres per-query term. The joint-centres path
# truncates each Gaussian at truncation_sigmas, so per query only the
# centres within a few sigma of the query contribute. The near-centre
# fraction scales as sigma / (source spread): wider kernels (or tighter
# source spreads) reach more centres, saturating at 1. Calibrated
# against measured culled per-query cost across sigma, K, and spread
# (worst-case shape error about 2.5x, typically within 1.5x). Applies
# to the per-query term only --- all joint centres are still
# materialised, so the setup and call terms are unculled.
_MA_COST_CENTRES_CULL_C = 15.0

#: Möbius: a per-call setup that scales with the partition count B_r,
#: and per-query work linear in the distinct-block op count
#: ``(2^r - 1) r K`` (each distinct block's factor is computed once and
#: reused across partitions). Relative attributes multiply the
#: per-query work by the u-grid node count; each node costs the cheaper
#: of the direct strategy (op-count linear) and, non-periodically, the
#: factored strategy (K-free after tabulation).
_MA_COST_MOBIUS_SETUP_MS = 0.30
_MA_COST_MOBIUS_SETUP_PER_BELL_MS = 0.05
_MA_COST_MOBIUS_QUERY_PER_OP_MS = 5e-7
#: Relative-mode u-grid node costs, per distinct-block op per query.
#: Periodic direct nodes cost more per op than non-periodic ones
#: (per-component wrapping inside the kernel, and no factored
#: tabulation on the circle); both are calibrated separately.
#: The non-periodic direct node also carries a fixed per-node cost,
#: independent of the op count: each u-grid node pays a setup (the
#: alignment shift and read-back) that dominates at low op counts. The
#: pure per-op form underprices small-r relative attributes, whose node
#: cost is floor-bound rather than op-bound, so a base term is carried
#: alongside the per-op slope.
_MA_COST_MOBIUS_REL_NODE_DIRECT_BASE_MS = 2.0e-4
_MA_COST_MOBIUS_REL_NODE_DIRECT_PER_OP_MS = 4e-6
_MA_COST_MOBIUS_REL_NODE_DIRECT_PER_OP_PER_MS = 1e-5
_MA_COST_MOBIUS_REL_NODE_FACTORED_PER_BELL_MS = 3.5e-4

#: u-grid tabulation setup, paid once per call: building the interpolation
#: table costs K source evaluations over the N_u grid nodes. In Python the
#: per-query node cost is large and already prices small-n_q calls
#: near-realistically, so the measured tabulation setup is negligible and
#: this constant is ~0; MATLAB's lean per-query readback leaves the setup
#: as the dominant Möbius cost at small n_q, so its twin constant is
#: nonzero. Same term, per-language magnitude.
_MA_COST_MOBIUS_REL_TABULATION_PER_NODE_MS = 0.0

#: u-grid nodes per sigma for the relative-mode node-count estimate are
#: derived per attribute via
#: :func:`mpt._defaults.resolve_samples_per_sigma`, mirroring the
#: evaluators' accuracy-tied resolution.

#: Safety factor favouring Möbius at near-ties: Möbius is chosen
#: whenever its estimate is below the centres estimate times this
#: factor. The asymmetry is deliberate: Möbius is failure-safe (flat,
#: bounded cost) while the centres path materialises the joint tuple
#: set and can exhaust memory, so a near-tie should break toward
#: Möbius rather than risk the explosive path.
_MA_MOBIUS_SAFETY = 1.5



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


def _guard_forced_bulger_feasible_ma(k_vec, r_vec, rel_vec, N_x, N_y, *, reason):
    """Raise if a *forced* multi-attribute Bulger inner product would be
    infeasible.

    Bulger's MA inner product materialises each side's joint perm-side
    working set ``n_J = N * prod_a nj_a`` with ``nj_a = K_a! / (K_a -
    r_a)!``, and the tuple-pair kernel is ``n_J_x * n_J_y`` float64
    entries. When the Möbius method is *forced* off (r above the
    shipped orbit order, or the K - r precision floor) there is no
    cheaper all-image substitute; at high tuple order the pair kernel
    can exhaust memory. Rather than let it crash the process, raise a
    clear error naming the shape. Explicit ``method='bulger'`` overrides
    are honoured earlier and do not reach here, so this guards only
    auto-dispatch.
    """
    A = len(r_vec)

    def _nj_side(N):
        n_j = float(N)
        for a in range(A):
            K_a = int(k_vec[a])
            r_a = int(r_vec[a])
            if K_a < r_a:
                return 0.0
            fac = 1.0
            for k in range(K_a - r_a + 1, K_a + 1):
                fac *= k
            n_j *= fac
            if n_j > 1e18:      # already hopeless; stop growing
                return 1e18
        return n_j

    nj_x = _nj_side(N_x)
    nj_y = _nj_side(N_y)
    pair_bytes = nj_x * nj_y * 8
    if pair_bytes > _CENTRES_PROBE_MEM_BUDGET:
        raise SingleImageInfeasibleError(
            f"The inner product requires the single-image Bulger route "
            f"({reason}, so the Möbius method is not available), but its "
            f"tuple-pair kernel would need ~{pair_bytes / 1024**3:.1f} GB "
            f"(n_J_x = {nj_x:.2e}, n_J_y = {nj_y:.2e}). Reduce the tuple "
            f"order r or the collection sizes."
        )


def _estimate_ma_joint_working_set_bytes(r_vec, k_vec, is_rel) -> int:
    """Estimate the multi-attribute joint-centres working set in bytes.

    The multi-attribute centres path materialises the *joint* tuple
    set: the product across attributes of each attribute's ordered-tuple
    count ``r_a! * C(K_a, r_a)``. The stored joint centres array is
    ``(D, n_joint)`` with ``D = sum_a (r_a - [rel]_a)``, plus per-attribute
    index bookkeeping of the same ``n_joint`` length; a row factor of
    ``2 * D`` over-counts honestly for a memory guard. Used only to
    detect when a convention- or precision-forced centres pick would be
    infeasible, so an over-count is the right bias.
    """
    A = len(r_vec)
    n_joint = 1
    D = 0
    for a in range(A):
        r_a = int(r_vec[a])
        K_a = int(k_vec[a])
        if K_a < r_a:
            return 0
        # ordered-tuple count r_a! * C(K_a, r_a) = K_a! / (K_a - r_a)!
        cnt = 1
        for k in range(K_a - r_a + 1, K_a + 1):
            cnt *= k
        n_joint *= max(cnt, 1)
        D += r_a - (1 if bool(is_rel[a]) else 0)
        # Cap to avoid unbounded big-int growth in the estimate itself;
        # anything past the budget is already "infeasible".
        if n_joint * max(D, 1) * 8 > (1 << 60):
            return 1 << 60
    return n_joint * max(D, 1) * 2 * 8



def _ma_eval_costs_ms(dens, n_q):
    """Closed-form ``(centres_ms, mobius_ms)`` cost estimates for a flat
    multi-attribute density, in milliseconds on the calibration machine.

    Depends only on the density shape ``(r_a, K_a, N)``, the geometry,
    and the query count ``n_q`` --- no probe, no timing. Shared by the
    eval path selector :func:`_select_ma_eval` (which compares the two)
    and by the up-front time estimate (which scales the chosen one by a
    per-session machine factor). Keeping one implementation guarantees
    the estimate and the dispatch decision price identical work.
    """
    A = int(dens.n_attrs)
    r_vec = [int(v) for v in np.atleast_1d(dens.r)]
    k_vec = [int(v) for v in np.atleast_1d(dens.k)]
    is_rel = [bool(v) for v in np.atleast_1d(dens.is_rel)]
    is_per = [bool(v) for v in np.atleast_1d(dens.is_per)]
    sigma = [float(v) for v in np.atleast_1d(dens.sigma)]
    period = [float(v) for v in np.atleast_1d(dens.period)]

    n_q_eff = float(max(int(n_q), 1))
    joint_tuples = 1.0
    for a in range(A):
        r_a, K_a = r_vec[a], k_vec[a]
        if r_a < 1:
            continue
        joint_tuples *= float(factorial(r_a)) * float(_math_comb(K_a, r_a))

    # Per-query culling factor for the single-multiset non-periodic centres
    # route (_eval_core): tuple centres are hashed into a truncation-radius
    # bucket grid and each query evaluates only its neighbouring buckets,
    # so the near-centre fraction is min(1, c * sigma / spread). The
    # multi-attribute factored centres route evaluates each attribute
    # through the same culled kernel, so the discount applies per
    # attribute there; the joint-materialisation fallback and the periodic
    # single-multiset route run dense (the pairwise wrap is not a
    # tail-truncatable ball) and take no discount.
    from .density import is_single_multiset

    def _attr_spread(a):
        p_attr = getattr(dens, "p_attr", None)
        if p_attr is not None and a < len(p_attr) and p_attr[a] is not None:
            arr = np.asarray(p_attr[a], dtype=np.float64)
            if arr.size:
                return float(np.max(arr) - np.min(arr))
        return 0.0

    def _attr_cull(a):
        if is_per[a] or sigma[a] <= 0:
            return 1.0
        spread = _attr_spread(a)
        if spread <= 0:
            return 1.0
        return min(1.0, _MA_COST_CENTRES_CULL_C * sigma[a] / spread)

    cull = 1.0
    if is_single_multiset(dens) and sigma[0] > 0 and not is_per[0]:
        cull = _attr_cull(0)

    # The factored centres route (all r_a >= 2, scalar sigma) never
    # materialises the joint tuple set: cost is the SUM of per-attribute
    # tuple counts through the culled per-attribute kernels, plus a small
    # per-attribute per-query overhead (bucket lookup and gather). The
    # joint-materialisation pricing applies only where that route is
    # unsupported (any r_a < 2, or a matrix kernel covariance), mirroring
    # the support predicate of the evaluator that actually runs.
    factored_supported = (
        A > 1
        and all(r_a >= 2 for r_a in r_vec)
        and getattr(dens, "kernel_cov", None) is None
    )
    if factored_supported:
        centres_ms = _MA_COST_CENTRES_SETUP_MS
        for a in range(A):
            r_a, K_a = r_vec[a], k_vec[a]
            T_a = float(factorial(r_a)) * float(_math_comb(K_a, r_a))
            centres_ms += (
                _MA_COST_CENTRES_CALL_PER_JOINT_MS * T_a
                + n_q_eff * (
                    _MA_COST_CENTRES_FACTORED_QUERY_BASE_MS
                    + _MA_COST_CENTRES_QUERY_PER_JOINT_MS
                    * T_a * _attr_cull(a)
                )
            )
    else:
        centres_ms = (
            _MA_COST_CENTRES_SETUP_MS
            + _MA_COST_CENTRES_CALL_PER_JOINT_MS * joint_tuples
            + _MA_COST_CENTRES_QUERY_PER_JOINT_MS * joint_tuples * cull * n_q_eff
        )

    mobius_ms = _MA_COST_MOBIUS_SETUP_MS
    for a in range(A):
        r_a, K_a = r_vec[a], k_vec[a]
        if r_a < 2:
            continue  # r_a <= 1: a plain kernel sum either way
        B_r = float(_BELL_NUMBERS.get(r_a, float("inf")))
        ops = float(2 ** r_a - 1) * r_a * K_a
        mobius_ms += _MA_COST_MOBIUS_SETUP_PER_BELL_MS * B_r
        per_query_ms = _MA_COST_MOBIUS_QUERY_PER_OP_MS * ops
        if is_rel[a]:
            from .._defaults import resolve_samples_per_sigma
            sps = float(resolve_samples_per_sigma(None, r_a, None))
            if is_per[a] and period[a] > 0:
                n_u = max(64.0, np.ceil(sps * period[a] / sigma[a]))
                node_ms = (
                    _MA_COST_MOBIUS_REL_NODE_DIRECT_PER_OP_PER_MS * ops)
            else:
                spread = 0.0
                p_a = getattr(dens, "p_attr", None)
                if p_a is not None and a < len(p_a) and p_a[a] is not None:
                    arr = np.asarray(p_a[a], dtype=np.float64)
                    if arr.size:
                        spread = float(np.max(arr) - np.min(arr))
                window = 2.0 * spread + 16.0 * sigma[a]
                n_u = max(64.0, np.ceil(sps * window / sigma[a]))
                node_ms = min(
                    _MA_COST_MOBIUS_REL_NODE_DIRECT_BASE_MS
                    + _MA_COST_MOBIUS_REL_NODE_DIRECT_PER_OP_MS * ops,
                    _MA_COST_MOBIUS_REL_NODE_FACTORED_PER_BELL_MS * B_r,
                )
            # Tabulation setup is paid once per call, not per query.
            mobius_ms += _MA_COST_MOBIUS_REL_TABULATION_PER_NODE_MS * K_a * n_u
            per_query_ms = n_u * node_ms
        mobius_ms += per_query_ms * n_q_eff

    return centres_ms, mobius_ms


def _predict_ma_eval_cost_ms(dens, n_q, chosen):
    """Predicted cost (ms, calibration machine) of the chosen eval path."""
    centres_ms, mobius_ms = _ma_eval_costs_ms(dens, n_q)
    return mobius_ms if chosen == "mobius" else centres_ms


def _select_ma_eval(dens, n_q, *, method):
    """Cost-model path selection for multi-attribute ``eval_exp_tens``.

    Chooses between the joint-centres path (``_eval_exp_tens_ma``, which
    materialises the joint tuple set --- the product across attributes
    of each attribute's ordered-tuple set --- and sums a Gaussian per
    joint centre) and the factored Möbius evaluator
    (:func:`mpt._tensor._ma_eval_orbit.eval_ma_orbit`, which never
    materialises joint centres: it evaluates each attribute's
    single-multiset density by the Möbius decomposition and takes the
    product per event).

    Unlike the single-multiset eval selector, this is a *pure cost
    model with no probe*: the MAET density factorises across attributes
    (Milne 2026, Eq. maet-density), so both paths' costs are estimated
    in closed form from the shape ``(r_a, K_a, N)``, the geometry, and
    the query count. Each estimate is a per-call setup term plus
    per-query work: the joint-centres path's work is linear in the
    joint tuple count ``prod_a [r_a! · C(K_a, r_a)]``, which grows as a
    *product* across attributes; the factored path's work is the *sum*
    across attributes of the per-attribute distinct-block op count
    ``(2^{r_a} - 1) · r_a · K_a``, with relative attributes further
    multiplied by a u-grid node count estimated from the geometry. The
    constant factors are the module-level ``_MA_COST_*`` calibration.
    The product-vs-sum contrast means the factored path wins decisively
    as soon as more than one attribute has a non-trivial tuple set,
    while single-attribute relative shapes favour the centres path far
    beyond the absolute-mode crossover, the u-grid multiplying the
    factored path's per-query cost by hundreds to thousands of nodes.

    Hard rules first: a user override is honoured; any attribute whose
    ``K_a - r_a`` is too small for the Möbius cancellation floor, or
    whose ``r_a`` exceeds the feasible orbit bound, forces the
    joint-centres path (the factored evaluator would lose precision or
    be infeasible on that attribute); and periodic-relative attributes
    whose ``sigma/P`` exceeds the convention threshold keep the
    joint-centres path, whose wrapped-difference form is the exact
    convention there (the Möbius relative evaluator yields the
    transposition-average form, which departs above that threshold ---
    Milne 2026, Sec. maet-nesting).

    Parameters
    ----------
    dens : MaetDensity
        A flat multi-attribute density.
    n_q : int
        Number of query points (scales both costs equally; retained for
        the estimate and for parity with the single-multiset selector signature).
    method : {'auto', 'centres', 'direct', 'mobius'}
        User override or ``'auto'`` for the cost model.

    Returns
    -------
    (chosen, routing_reason) : tuple[str, str]
        ``chosen`` is ``'centres'`` or ``'mobius'``; ``routing_reason``
        is a short explanation for the dispatch message.
    """
    if method in ("centres", "direct"):
        return "centres", "user override"
    if method == "mobius":
        return "mobius", "user override"
    if method != "auto":
        raise ValueError(
            f"method must be 'auto', 'centres', 'direct', or 'mobius'; "
            f"got {method!r}."
        )

    A = int(dens.n_attrs)
    r_vec = [int(v) for v in np.atleast_1d(dens.r)]
    k_vec = [int(v) for v in np.atleast_1d(dens.k)]
    is_rel = [bool(v) for v in np.atleast_1d(dens.is_rel)]
    is_per = [bool(v) for v in np.atleast_1d(dens.is_per)]
    sigma = [float(v) for v in np.atleast_1d(dens.sigma)]
    period = [float(v) for v in np.atleast_1d(dens.period)]

    # ---- Hard rule: nested attributes are not handled by the flat
    # factored evaluator; keep the joint-centres path. ----
    nested = getattr(dens, "nested", [None] * A)
    if any(nested[a] is not None for a in range(A)):
        return "centres", "nested attribute (flat Möbius not applicable)"

    # ---- Hard rule: r <= 1 on every attribute => Möbius is degenerate
    # (one singleton partition); centres is trivially cheap. ----
    if all(r_vec[a] <= 1 for a in range(A)):
        return "centres", "all r <= 1"

    # ---- Hard rules per attribute: precision floor and feasibility
    # force the single-image centres route, because the Möbius method is
    # genuinely unavailable there (it would lose precision or is beyond
    # its shipped order) --- not merely slower. These forced-centres
    # picks are guarded against out-of-memory below: since no cheaper
    # all-image substitute exists here, an infeasible shape must raise a
    # clear error, not crash. (The σ/P convention is handled separately,
    # after the loop: there the all-image Möbius form is a legitimate
    # cheaper measure, so it becomes the preferred default rather than a
    # reason to force centres.) ----
    force_centres_reason = None
    for a in range(A):
        r_a, K_a = r_vec[a], k_vec[a]
        if r_a < 2:
            continue  # r_a = 1 factor is exact either way
        if not _orbit_safe_for_precision([r_a], [K_a]):
            force_centres_reason = (
                f"attr {a}: K - r = {K_a - r_a} below precision floor")
            break
        if r_a > _ORBIT_R_MAX_FEASIBLE:
            force_centres_reason = (
                f"attr {a}: r = {r_a} exceeds orbit feasibility bound")
            break

    if force_centres_reason is not None:
        # Möbius is unavailable for correctness/feasibility; centres is
        # the only route. Guard it: if the joint tuple set is too large
        # to materialise, there is no cheaper fallback (Möbius is
        # refused here), so raise rather than OOM.
        joint_ws = _estimate_ma_joint_working_set_bytes(r_vec, k_vec, is_rel)
        if joint_ws > _CENTRES_PROBE_MEM_BUDGET:
            raise SingleImageInfeasibleError(
                f"eval_exp_tens requires the single-image centres route "
                f"({force_centres_reason}, so the Möbius method is not "
                f"available), but its joint tuple set would need "
                f"~{joint_ws / 1024**3:.1f} GB. Reduce the tuple order r "
                f"or the collection size K."
            )
        return "centres", force_centres_reason

    # ---- Relative-periodic measure preference (takes precedence over
    # the cost model). Above the σ/P threshold the Möbius method computes
    # the all-image (transposition-average) form while centres computes
    # the single-image (minimum-image) form --- different measures, not
    # two routes to one answer. The toolbox prefers the all-image form
    # here whenever it is available (it is cheaper and memory-safe: the
    # factored evaluator is n_j-free), matching the inner-product path.
    # The single-image measure remains available on demand via
    # method='centres'. Warn that the substitution has occurred. ----
    for a in range(A):
        if (is_rel[a] and is_per[a] and period[a] > 0
                and sigma[a] / period[a] > _ORBIT_SIGMA_OVER_P_THRESHOLD):
            _warn_rel_per_all_image(
                sigma[a] / period[a],
                operation="evaluation",
                canonical_method="centres",
                stacklevel=3,
            )
            return "mobius", "rel-per all-image measure"

    # ---- Cost model: two closed-form per-call time estimates (ms),
    # each a per-call setup term plus per-query work scaled by n_q. The
    # functional forms and constants live in :func:`_ma_eval_costs_ms`,
    # shared with the up-front time estimate so both price identical
    # work. Centres cost grows with the joint tuple count (product
    # across attributes); Möbius cost is the summed per-attribute
    # distinct-block work, with relative attributes multiplied by a
    # u-grid node count from the geometry. ----
    centres_ms, mobius_ms = _ma_eval_costs_ms(dens, n_q)

    if mobius_ms < centres_ms * _MA_MOBIUS_SAFETY:
        return "mobius", "cost model (factored Möbius cheaper)"
    return "centres", "cost model (joint centres cheaper)"



# -----------------------------------------------------------------------
# Probe-based dispatcher for the single-multiset cos_sim_exp_tens IP path.
#
# Parallels :func:`_select_ma_eval` for the inner-product side:
# hard rules first (correctness / feasibility), then an analytical
# pre-screen (catches clear-winner cases without paying probe overhead),
# then actually time both paths on a small subset of each density and
# pick the faster. The probe time, extrapolated to the full workload,
# becomes the user-facing time estimate (printed in verbose mode) and
# auto-adapts to future optimisations of either path.
#
# Probe extrapolation. Both paths compute three inner products (cross
# term plus both self-norms), so the cost models count all three:
# pairwise cost scales as ``P_x*P_y + P_x^2 + P_y^2`` with
# ``P = falling_factorial(K, r)`` (ordered r-tuple enumeration on each
# side); orbit cost scales as ``B_r * (N_xy*K_x*K_y + N_xx*K_x^2 +
# N_yy*K_y^2)`` where the ``N`` factors are the relative-mode
# translation-grid sizes (1 in absolute mode). The probe uses
# ``(min(K_x, target), min(K_y, target))`` events — per side, so an
# asymmetric workload is probed with the same asymmetry — and
# extrapolates by the ratio of the corresponding op counts. The
# pairwise estimate removes its fixed per-call cost with a two-point
# fit before scaling, and the probe decision requires the orbit
# estimate to beat the pairwise estimate by
# ``_PROBE_IP_MOBIUS_DECISION_MARGIN``.
# -----------------------------------------------------------------------

# Target per-side subset size for the IP probe. Small enough that probe
# cost is negligible, large enough that the K_probe-choose-r tuple count
# is meaningful (e.g., 12-choose-3 = 220) and the Möbius method's
# precision guard (n_min - r >= 2) is not contended. Each side's probe
# size is capped to its own K at call time; the hard precision rule
# (``n_min - r < 2``) fires upstream so neither probe size drops below
# r+2.
_PROBE_K_IP_TARGET = 12

_PROBE_MIN_SAMPLE_SEC = 0.008
_PROBE_MAX_REPS = 64


_PROBE_TIME_CACHE: dict = {}
"""Per-session cache of probe timings, keyed by the probe's structural
parameters (path, r, probe sizes, mode flags, sigma, period,
truncation, precision). Within a batched sweep every pair probes at
identical structure, so re-measuring per pair adds cost without
information — and where the probed path is intrinsically slow (an
uncalibrated order routing via the probe), the repeated measurement
would dominate the sweep. Pitch values differ across cache hits; probe
timings depend on them only through kernel sparsity, which is
immaterial at routing precision."""

_PROBE_IP_MOBIUS_DECISION_MARGIN = 1.4
"""Margin the orbit probe estimate must beat the pairwise probe estimate
by before the probe routes to the Möbius method. The orbit probe's
working set is cache-resident while the full-size rel-mode path is
memory-bound, so the probe systematically under-measures the full-scale
per-op cost by ~1.3x; the pairwise estimate carries no matching bias
(its fixed per-call cost is removed by the two-point fit). Near-tie
estimates therefore route to the pairwise path, the cheap-to-mispick
side, mirroring the asymmetric pre-screen margins."""


# Pre-screen: skip the probe if one path's analytical cost dominates
# the other by this margin. Mirrors the eval-side pre-screen
# constants.
_PRESCREEN_IP_DOMINANCE = 3.0

# Fixed-overhead term for the Möbius inner-product cost estimate, in units of
# K^2 (i.e. added to the grid-scaled kernel-op count inside the orbit-cost
# expression). The Möbius method carries a per-call setup cost (orbit-table
# lookup, einsum planning, partition iteration) that scales with the partition
# count B_r but is independent of K; the bare operation count omits it and so
# under-estimates Möbius at small K, making the analytical pre-screen route to
# Möbius well before it is actually faster than Bulger's method. Adding
# B_r*_ORBIT_IP_FIXED_OVERHEAD to the orbit cost shifts the analytical
# equal-cost point to (just below) the empirically measured Bulger/Möbius
# crossover per r, so the pre-screen never claims 'mobius' prematurely; the
# probe still has the final word in the near-crossover region. The value is
# calibrated for the three-inner-product cost model (cross term plus both
# self-norms, so 3*K^2 kernel-op units at symmetric K in absolute mode): it is
# 3x the per-IP setup constant calibrated against measured absolute-mode
# crossovers K_cross = {r2:14, r3:9, r4:8, r5:8, r6:8}, which keeps the
# symmetric-K absolute-mode equal-cost point at those measured values.
_ORBIT_IP_FIXED_OVERHEAD = 24000.0

# Dominance margin for the *Möbius* side of the analytical inner-product
# pre-screen. Larger than _PRESCREEN_IP_DOMINANCE so that near-crossover cases
# (where the analytical model is least reliable) defer to the timing probe
# rather than committing to Möbius on an under-estimate. The Bulger side keeps
# the tighter _PRESCREEN_IP_DOMINANCE because over-predicting Bulger is cheap
# (Bulger's cost is genuinely low in that regime) whereas prematurely choosing
# Möbius pays its fixed overhead needlessly.
_PRESCREEN_IP_MOBIUS_DOMINANCE = 10.0



def _falling_factorial(n: int, k: int) -> float:
    """``n * (n-1) * ... * (n-k+1)``; 0 if any factor is non-positive."""
    if n < k:
        return 0.0
    prod = 1.0
    for i in range(k):
        prod *= (n - i)
    return prod



_ORBIT_GRID_OP_UNIT_COST = {2: 0.7, 3: 1.7, 4: 1.7, 5: 20.0}
"""Per-op cost of a translation-grid orbit kernel op relative to a
pairwise kernel op, per tensor order r, measured on the Python
implementation (slabbed grid einsum contraction against pairwise
kernel evaluation; the orbit side is ~9-18 ns/op and flat in both K
and r, while the pairwise side's per-op cost varies with r — ~17-21
ns at r = 2, ~8-9 at r = 3, ~4-17 at r = 4, ~1 at r = 5 — which is
what makes the ratio r-dependent; r = 4 and 5 measured with a K_x = 8
reference, tests/bench_ip_dispatch.m). The pairwise and orbit cost models
below count kernel ops in a shared unit; this factor prices the
rel-mode orbit ops so the modelled equal-cost point matches the
measured one. The per-orbit contraction work per kernel op varies with
r beyond what the Bell-number factor captures, so the calibration is
per-r. Each value sits at the small-K end of its measured ratio range
(the ratio falls slightly with K, and the pairwise reference is its
cache-resident per-op cost, which large working sets degrade well
beyond), so near-crossover routing biases toward the pairwise path,
the cheap-to-mispick side.

Orders without a calibrated entry use a unit factor of 1.0 (the raw
grid size) and — see :func:`_select_ma_inner_product_method` — withhold the
Möbius-side pre-screen, so routing at those orders defers to the
timing probe rather than trusting an uncalibrated model to commit to
the expensive-to-mispick path.

Applied to the relative-mode grid factors only: the absolute-mode
orbit cost calibration (_ORBIT_IP_FIXED_OVERHEAD against measured
absolute-mode crossovers) predates no such factor and is left
untouched. The values are per-implementation: the MATLAB sibling in
cosSimExpTens.m is calibrated the same way on the MATLAB paths via
tests/bench_ip_dispatch.m."""


def _orbit_grid_unit_cost(r: int) -> tuple[float, bool]:
    """Unit-cost factor for tensor order ``r`` and whether it is a
    calibrated value (uncalibrated orders return ``(1.0, False)``)."""
    cost = _ORBIT_GRID_OP_UNIT_COST.get(int(r))
    if cost is None:
        return 1.0, False
    return float(cost), True



