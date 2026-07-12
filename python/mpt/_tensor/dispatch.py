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



# Per-r, per-mode K thresholds for the orbit-vs-enumeration crossover at a
# single symmetric attribute or nesting level, established empirically on
# representative MAET workloads (N = 8-12, σ = 12, P = 1200). Mode matters: the
# relative-periodic orbit path carries a transposition-average u-grid that
# enumeration avoids, so its crossover sits well above the absolute one, and at
# r = 2 enumeration always wins (hence the inf entry — a pure op-count
# comparison, which sees only the orbit-class reduction and not the u-grid
# cost, would wrongly route every r = 2 rel-per level to orbit). These drive
# the per-level decision in the nested contraction via
# _orbit_beats_pairwise_per_attr. The flat multi-attribute and single-attribute
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
    """Per-level orbit-vs-enumeration decision for one symmetric attribute or
    nesting level, from the mode-aware K thresholds above.

    This is the live predicate for the per-level decision in the nested
    contraction (``_nested_contraction._orbit_eligible``): it runs once per
    level, so it uses a cheap mode-aware threshold rather than a probe. The flat
    multi-attribute and single-attribute inner-product paths instead make a
    whole-call decision through the cost model and probe (see
    ``_select_ma_inner_product_method`` and ``_select_and_estimate_sa_ip``),
    which also weigh N. The two mechanisms are matched to their contexts, not
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
    # ---- Memory-safety guard (explicit invariant) ----
    # Bulger's MA IP materialises each side's joint perm-side working
    # set n_J = N · ∏_a r_a!·C(K_a, r_a) (lazy, built on first access);
    # the Möbius MA IP is n_J-free (per-event, per-attribute additive
    # work). As in the SA IP path the cost model below already routes
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
        # arrays, mirroring the SA row-factor), capped to avoid overflow.
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
    pw_cost_ms = pw_size * _pw_per_entry_ms(any_per)
    orbit_cost_ms = _predict_orbit_cost_ms(
        r_max, A, N_x, N_y, k_vec, any_rel_nonper, any_rel_per,
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


def _compute_Q(D, r, is_rel, is_per, period, *, reduced=False):
    """Compute the quadratic form from differences D.

    Two input conventions, controlled by ``reduced``:

    - ``reduced=False`` (default): ``D`` has *r* components representing
      a full r-tuple ``d = (d_0, …, d_{r-1})``. This is what the
      inner-product (cosine-similarity) path uses, where centres are
      stored as full r-tuples.
    - ``reduced=True``: ``D`` has *r-1* components representing the
      "slot 0 anchored" reduction ``D[k] = d_{k+1} − d_0`` of an
      r-tuple. This is what the SA centres-array evaluation path uses,
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


def _warn_rel_per_all_image(sigma_over_P):
    """Warn that the dispatch took the faster all-image form of the
    relative-periodic inner product, which above the σ/P threshold differs from
    the canonical single-wrap measure.

    Emitted by every relative-periodic path -- flat single-attribute, flat
    multi-attribute, and the nested contraction -- so the message is identical
    wherever the substitution happens. It fires only when the all-image
    (transposition-integral) form has actually been selected as the faster path
    *and* σ/P exceeds the threshold where the two measures materially diverge;
    below the threshold the two agree and no warning is raised.
    """
    warnings.warn(
        f"σ/P = {sigma_over_P:.3f} exceeds {_ORBIT_SIGMA_OVER_P_THRESHOLD}: the "
        f"faster all-image (transposition-integral) form of the "
        f"relative-periodic inner product has been used. Above this σ/P it "
        f"differs from the canonical single-wrap (minimum-image) measure "
        f"(the two agree below it). To compute the single-wrap measure "
        f"instead, pass method='bulger', which enumerates the full symmetric "
        f"orbit this fast path avoids; that enumeration can be substantially "
        f"slower, and infeasible for a large or compounded symmetric level "
        f"(precisely the case that made the all-image form the faster path "
        f"here).",
        stacklevel=3,
    )

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

    This is the probe-free SA routing reference: it encodes the same routing
    rules as the live cosine-path selector ``_select_and_estimate_sa_ip`` (hard
    guards, then the fastest path with the relative-periodic warning) but
    without the wall-time probe, so the policy can be exercised directly from
    the structural inputs. The cosine path itself uses
    ``_select_and_estimate_sa_ip``; this lighter form is what the dispatcher
    unit tests assert against.

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
    # Relative-periodic measure note: the Möbius method computes the all-image
    # (JMM Eq. 3.4 transposition-integral) form; Bulger's method computes the
    # single-wrap (minimum-image) form. They diverge by O((σ/P)^∞) above
    # σ/P ≈ 0.03. The Möbius method is the faster SA path here, so the dispatch
    # takes it; when σ/P is above the threshold (so the two measures differ) it
    # warns and points to method='bulger' for the canonical single-wrap measure.
    if is_rel and is_per and sigma_over_P > _ORBIT_SIGMA_OVER_P_THRESHOLD:
        _warn_rel_per_all_image(sigma_over_P)
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


def _estimate_centres_working_set_bytes(K: int, r: int, is_rel: bool) -> int:
    """Estimate the *full* centres-path working set in bytes.

    Unlike :func:`_estimate_centres_array_bytes` (which counts only the
    final ``(dim, n_j)`` centres array, and is pinned by the dispatcher
    tests), this reflects everything :meth:`ExpTensDensity._build_perm_arrays`
    holds live at once. The three arrays that scale with
    ``n_j = K!/(K-r)!`` are ``j_idx`` (``r x n_j`` int), ``u_perm``
    (``r x n_j`` float) and ``centres`` (``dim x n_j`` float); the
    ``C(K, r)``-sized comb arrays are smaller and omitted. Total row
    factor is therefore ``2*r + dim`` (with ``dim = r - 1`` for rel,
    ``r`` for abs), each element 8 bytes. Used only by the soft memory
    guard, so a rough but honest over-count of the array estimate is the
    right bias.
    """
    if K < r:
        return 0
    n_j = 1
    for k in range(K - r + 1, K + 1):
        n_j *= k
    dim = r - 1 if is_rel else r
    row_factor = 2 * r + max(dim, 1)
    return n_j * row_factor * 8



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

    # ---- Rule 4b: soft working-set guard (memory-aware, n_q-independent) ----
    # The rel-mode cost model is wall-time oriented: it compares op-counts
    # (centres n_j vs Möbius direct-strategy B_r · r · K · N_u, or its
    # K-free factored strategy for batched workloads) and is blind to the fact
    # that the centres path must first *materialise* the full ordered-tuple
    # working set — j_idx, u_perm and centres, each O(n_j) with
    # n_j = K!/(K-r)!. In the large-template regime (e.g. tensor_harmonicity
    # at duplicate >= 4, K = 48) that working set is hundreds of MB even
    # though it stays under the 4 GB hard ceiling and even though its
    # op-count can look competitive. The Möbius point evaluator is n_j-free,
    # so its memory is bounded regardless. This guard mirrors the abs-mode
    # pre-screen below (it must likewise run BEFORE the tiny-workload
    # shortcut, so single-chord / small-batch calls are covered) but keys
    # off memory rather than time: when the centres working set exceeds the
    # soft budget and Möbius is feasible, route to Möbius to keep peak
    # memory bounded. Convention guard: in periodic-relative mode the two
    # methods diverge above sigma/P ~ 0.03, so only divert there when the
    # convention still agrees; otherwise leave the (memory-heavy but
    # convention-exact) centres path in place. Precision (K - r >= 2) and
    # the r <= feasible bound are already ensured by Rules 3 and the hard
    # ceiling's r-check; Möbius is a valid candidate here.
    working_set = _estimate_centres_working_set_bytes(K, r, is_rel)
    if working_set > _CENTRES_WORKING_SET_SOFT_BUDGET and r <= _ORBIT_R_MAX_FEASIBLE:
        sigma_over_P = (
            float(dens.sigma) / float(dens.period) if dens.is_per else 0.0
        )
        convention_safe = (
            (not dens.is_per)
            or sigma_over_P <= _ORBIT_SIGMA_OVER_P_THRESHOLD
        )
        if convention_safe:
            return "mobius", False, 0.0, "centres working-set soft budget"

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
    # sub-evals per query — its PROBE cost scales with
    # B_r · r · N_u · n_probe (times K on the direct strategy; K-free
    # plus an amortised tabulation on the factored strategy), which is
    # prohibitive when N_u is large. We pre-screen the cost ratio analytically and skip the
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
        # Möbius per-query cost on the same per-K basis. The direct
        # strategy costs B_r · r · N_u; the factored strategy
        # (non-periodic, selected by eval_orbit_rel's internal cost
        # gate for batched workloads) removes the K factor from the
        # per-node cost, leaving read-back kernel-equivalents per
        # (partition-block, node) plus a tabulation amortised over the
        # queries. Use the cheaper of the two so this pre-screen does
        # not wrongly route batched rel workloads to centres; the
        # probe below still has the final word (and, timing the real
        # Möbius path, adapts automatically to the strategy the gate
        # picks).
        orbit_cost = float(B_r) * r * N_u_est
        if not dens.is_per:
            from .._mobius import (
                _FACTORED_READBACK_COST,
                _factored_spp,
                _factored_target_eps,
            )
            spp = _factored_spp(
                _factored_target_eps(truncation_sigmas, kernel_precision)
            )
            sum_sqrt_m = float(np.sum(np.sqrt(np.arange(1, r + 1))))
            extent_fine = (u_max - u_min) + (
                max(0.0, x_max_abs) - min(0.0, x_min_abs)
            )
            n_fine_est = extent_fine / sigma * spp * sum_sqrt_m
            orbit_factored = (
                _FACTORED_READBACK_COST * float(B_r) * r * N_u_est / K
                + n_fine_est / max(n_q, 1)
            )
            orbit_cost = min(orbit_cost, orbit_factored)
        # Finite truncation prunes the centres path's kernel work
        # (often by 10-100x on sparse-support workloads) and does not
        # prune the factored read-back, so the cost model's error is
        # one-sided: whenever it says centres is cheaper under
        # truncation, reality agrees. Require no dominance margin in
        # that case; keep the 3x margin when truncation is off.
        trunc = truncation_sigmas
        if trunc is None:
            from .._defaults import get_default
            trunc = get_default("truncation_sigmas")
        dominance = (1.0 if (trunc is not None and np.isfinite(trunc))
                     else _PRESCREEN_CENTRES_DOMINANCE)
        if centres_cost * dominance < orbit_cost:
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



_ORBIT_GRID_OP_UNIT_COST = 0.7
"""Per-op cost of a translation-grid orbit kernel op relative to a pairwise
kernel op, measured on the Python implementation (slabbed grid einsum
contraction ~9-12 ns/op against pairwise kernel evaluation ~17-21 ns/op
in its cache-resident regime). The pairwise and orbit cost models below
count kernel ops in a shared unit; this factor prices the rel-mode
orbit ops so the modelled equal-cost point matches the measured one.
The value sits at the small-K end of the measured ratio range (the
ratio falls slightly with K, and the pairwise reference is its
cache-resident per-op cost, which large working sets degrade well
beyond), so near-crossover routing biases toward the pairwise path,
the cheap-to-mispick side. Applied to the relative-mode grid factors
only: the absolute-mode orbit cost calibration
(_ORBIT_IP_FIXED_OVERHEAD against measured absolute-mode crossovers)
predates no such factor and is left untouched. The constant is
per-implementation: the MATLAB sibling in cosSimExpTens.m is
calibrated the same way on the MATLAB paths via
tests/bench_ip_unit_cost.m."""


def _orbit_ip_grid_factors(
    p_x: np.ndarray,
    p_y: np.ndarray,
    sigma: float,
    is_rel: bool,
    is_per: bool,
    period: float,
) -> tuple[float, float, float]:
    """Cost-model grid weights ``(N_xy, N_xx, N_yy)`` of the three orbit
    inner products (cross term plus both self-norms).

    The relative-mode orbit inner product marginalises a translation u
    over a grid and runs the orbit contraction at every grid point, so
    its kernel-op count carries the grid size as a multiplicative
    factor. The three factors mirror the grid-sizing rules of
    ``_orbit_inner_rel``: in periodic mode the grid covers one period
    with ``auto_ntau_default(period, sigma)`` nodes (identical for all
    three inner products); in non-periodic mode the line grid spans the
    two operands' spreads plus the 16-sigma truncation margin at 10
    samples per sigma, so each inner product has its own size. The
    absolute-mode orbit inner product is grid-free, so all three
    factors are 1.

    In relative mode each grid size is scaled by
    ``_ORBIT_GRID_OP_UNIT_COST`` so that the orbit and pairwise cost
    models price their kernel ops in a shared unit. Both the full-size
    and probe-size cost expressions call this helper, so the scaling
    cancels in the probe's extrapolation ratio: it moves the analytical
    pre-screen boundaries only.
    """
    if not is_rel:
        return 1.0, 1.0, 1.0
    if is_per:
        # Lazy import to keep dispatch free of a hard dependency on
        # the nested-contraction module at import time.
        from ._nested_contraction import auto_ntau_default
        n = float(auto_ntau_default(period, sigma)) * _ORBIT_GRID_OP_UNIT_COST
        return n, n, n
    samples_per_sigma = 10.0

    def _n_u(p_a: np.ndarray, p_b: np.ndarray) -> float:
        span = (float(p_a.max()) - float(p_a.min())
                + float(p_b.max()) - float(p_b.min())
                + 16.0 * sigma)
        return float(max(
            64,
            int(np.ceil(max(span, 1.0) / sigma * samples_per_sigma)),
        )) * _ORBIT_GRID_OP_UNIT_COST

    return _n_u(p_x, p_y), _n_u(p_x, p_x), _n_u(p_y, p_y)



def _probe_ip_path(
    dens_x: "ExpTensDensity",
    dens_y: "ExpTensDensity",
    K_probe_x: int,
    K_probe_y: int,
    path: str,
    *,
    truncation_sigmas: float | None,
    kernel_precision: str | None,
) -> float:
    """Time one cos_sim_exp_tens IP path on the first ``K_probe_x`` /
    ``K_probe_y`` events of the respective densities. Returns seconds.

    The probe sizes are per side so an asymmetric workload (a small
    reference against a large candidate set) is probed with the same
    asymmetry: probing both sides at the smaller K misrepresents the
    per-op cost of the larger side's self-norm, which dominates the
    pairwise path at scale.

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
        dens_x.p[:K_probe_x], dens_x.w[:K_probe_x],
        dens_x.sigma, int(dens_x.r),
        bool(dens_x.is_rel), bool(dens_x.is_per), float(dens_x.period),
        verbose=False,
    )
    sub_y = build_exp_tens(
        dens_y.p[:K_probe_y], dens_y.w[:K_probe_y],
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
    subset (``min(K_x, target)`` by ``min(K_y, target)``, per side) and
    extrapolated to
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
    # Relative-periodic measure note: 'mobius' is the all-image
    # (transposition-integral) form, 'bulger' the single-wrap (minimum-image)
    # form; they diverge by O((σ/P)^∞) above σ/P ≈ 0.03. The dispatch takes the
    # faster path (cost pre-screen / probe below); when that path is the
    # all-image Möbius method and σ/P is above the threshold (so the measures
    # differ) it warns at the return point and points to method='bulger' for
    # the canonical single-wrap measure.
    _rel_per_above = (is_rel and is_per
                      and sigma_over_P > _ORBIT_SIGMA_OVER_P_THRESHOLD)
    if r > _ORBIT_R_MAX_FEASIBLE:
        return "bulger", False, 0.0, f"r = {r} > {_ORBIT_R_MAX_FEASIBLE} (Möbius infeasible)"

    # ---- Memory-safety guard (explicit invariant) ----
    # The Bulger IP materialises each side's O(n_j = K!/(K-r)!) tuple
    # working set (via build_perm_arrays on both densities) plus the
    # chunked (n_Jx x n_Jy) kernel matrix. The Möbius IP is n_j-free.
    # In absolute mode the op-count pre-screen below already diverts
    # every large-K workload to Möbius, because the pairwise cost's
    # self-norm terms grow as n_j^2 per side while the orbit cost grows
    # only as K^2. In relative modes the orbit cost carries the
    # translation-grid factor, so Bulger is legitimately the faster path
    # up to much larger n_j and the cost comparison alone no longer
    # bounds the pairwise working set. This guard makes the memory
    # invariant explicit rather than emergent from the cost constants:
    # if either side's centres working set exceeds the soft budget and
    # Möbius is convention-safe, take Möbius now. Precision
    # (n_min - r >= 2) and feasibility (r <= feasible) are already
    # ensured by the hard rules above.
    ws_x = _estimate_centres_working_set_bytes(K_x, r, is_rel)
    ws_y = _estimate_centres_working_set_bytes(K_y, r, is_rel)
    if max(ws_x, ws_y) > _CENTRES_WORKING_SET_SOFT_BUDGET and not _rel_per_above:
        return "mobius", False, 0.0, "centres working-set soft budget"

    # ---- Relative-periodic measure preference (takes precedence over cost) ----
    # Above the sigma/P threshold the Möbius method computes the all-image
    # transposition average while Bulger's method computes the single-wrap
    # (minimum-image) form -- these are *different measures*, not two routes to
    # the same answer. The toolbox's default measure there is the all-image
    # form, so we must choose Möbius on measure grounds regardless of the cost
    # comparison below (which assumes both methods compute the same object, as
    # they do in absolute mode and in relative mode below the threshold). Emit
    # the warning pointing to method='bulger' for the single-wrap measure.
    if _rel_per_above:
        _warn_rel_per_all_image(sigma_over_P)
        return "mobius", False, 0.0, "rel-per all-image measure"

    # ---- Analytical cost models ----
    # Both paths compute three inner products: the cross term <x, y> and
    # the two self-norms <x, x> and <y, y>. The self-norm terms must be
    # counted: the pairwise path's <y, y> costs FF(K_y, r)^2 tuple pairs,
    # which dominates the cross term's FF(K_x, r)*FF(K_y, r) whenever the
    # operand sizes are asymmetric (a small reference against a large
    # candidate set makes <y, y> the whole cost, not a correction).
    P_x = _falling_factorial(K_x, r)
    P_y = _falling_factorial(K_y, r)
    pairwise_full = P_x * P_y + P_x * P_x + P_y * P_y
    B_r = float(_BELL_NUMBERS[r])
    # Orbit cost = B_r * (grid-scaled kernel-op count + fixed overhead).
    # In relative mode each orbit inner product runs the contraction at
    # every node of a translation grid, so the kernel-op count carries
    # the per-inner-product grid size as a multiplicative factor (3332
    # nodes for sigma = 3, period = 1200 — three orders of magnitude, not
    # a correction). In absolute mode the factors are 1. The
    # fixed-overhead term (see _ORBIT_IP_FIXED_OVERHEAD) captures the
    # K-independent Möbius setup cost that the bare operation count
    # omits; without it the pre-screen routes to Möbius well before the
    # measured Bulger/Möbius crossover.
    N_xy, N_xx, N_yy = _orbit_ip_grid_factors(
        dens_x.p, dens_y.p, sigma, is_rel, is_per, period,
    )
    orbit_var = (N_xy * float(K_x) * float(K_y)
                 + N_xx * float(K_x) * float(K_x)
                 + N_yy * float(K_y) * float(K_y))
    orbit_full = B_r * (orbit_var + _ORBIT_IP_FIXED_OVERHEAD)

    # ---- Analytical pre-screen ----
    # The Möbius side uses a larger dominance margin than the Bulger side.
    # Even with the fixed-overhead correction (_ORBIT_IP_FIXED_OVERHEAD) the
    # analytical orbit cost slightly under-predicts the measured Bulger/Möbius
    # crossover at higher r, so firing 'mobius' on a bare 3x margin can still
    # route one K-step early. Requiring a larger margin keeps near-crossover
    # cases in the probe's hands (the probe times both paths and is portable
    # across machines), while still short-circuiting the clear-win region.
    if orbit_full * _PRESCREEN_IP_MOBIUS_DOMINANCE < pairwise_full:
        return "mobius", False, 0.0, "cost pre-screen"
    if pairwise_full * _PRESCREEN_IP_DOMINANCE < orbit_full:
        return "bulger", False, 0.0, "cost pre-screen"

    # ---- Probe both paths on a subset ----
    # Warm the orbit partition table so the Möbius probe doesn't pay a
    # one-time table-build cost.
    from .._mobius import get_set_partitions_with_mobius
    get_set_partitions_with_mobius(r)

    K_probe_x = min(K_x, _PROBE_K_IP_TARGET)
    K_probe_y = min(K_y, _PROBE_K_IP_TARGET)
    # K_probe - r >= 2 is guaranteed per side by the precision hard rule
    # above (n_min - r >= _ORBIT_K_MINUS_R_MIN), so the orbit probe is safe.

    t_pairwise = _probe_ip_path(
        dens_x, dens_y, K_probe_x, K_probe_y, "bulger",
        truncation_sigmas=truncation_sigmas,
        kernel_precision=kernel_precision,
    )
    t_orbit = _probe_ip_path(
        dens_x, dens_y, K_probe_x, K_probe_y, "mobius",
        truncation_sigmas=truncation_sigmas,
        kernel_precision=kernel_precision,
    )

    # ---- Extrapolate to full workload ----
    # The probe runs the full three-inner-product computation on
    # (K_probe_x, K_probe_y) subsets, so the probe-scale op counts are
    # the same three-term expressions evaluated at the probe sizes. For
    # the orbit path the grid factors at probe scale come from the
    # subset pitch arrays: in periodic-relative mode they equal the
    # full-scale factors (the grid depends only on sigma and period, so
    # they cancel in the ratio); in non-periodic relative mode the
    # subset spans set smaller grids and the ratio carries the
    # difference; in absolute mode all factors are 1.
    P_px = _falling_factorial(K_probe_x, r)
    P_py = _falling_factorial(K_probe_y, r)
    pairwise_probe = P_px * P_py + P_px * P_px + P_py * P_py
    pairwise_factor = (
        pairwise_full / pairwise_probe if pairwise_probe > 0 else 1.0
    )

    # Two-point fixed-cost removal for the pairwise estimate. A probe
    # at the target sizes measures mostly fixed per-call cost (its op
    # count is small relative to the per-call setup), so scaling the
    # raw timing by the op-count ratio inflates the estimate by that
    # fixed share times the ratio — a factor of 2–3 at large
    # extrapolation ratios, all of it biasing the decision toward the
    # Möbius method. A second probe at a smaller subset separates the
    # two components: fit t = a + b*ops through the two points, carry
    # the fixed part a unscaled, and scale only the variable part b.
    # Skipped when the extrapolation ratio is small (raw scaling is
    # then accurate) or when a meaningfully smaller second point is
    # unavailable.
    if pairwise_factor > 2.0:
        K2_x = max(r + 2, K_probe_x // 2)
        K2_y = max(r + 2, K_probe_y // 2)
        P2_x = _falling_factorial(K2_x, r)
        P2_y = _falling_factorial(K2_y, r)
        pairwise_probe_2 = P2_x * P2_y + P2_x * P2_x + P2_y * P2_y
        if pairwise_probe_2 < 0.7 * pairwise_probe:
            t_pairwise_2 = _probe_ip_path(
                dens_x, dens_y, K2_x, K2_y, "bulger",
                truncation_sigmas=truncation_sigmas,
                kernel_precision=kernel_precision,
            )
            b = max(
                (t_pairwise - t_pairwise_2)
                / (pairwise_probe - pairwise_probe_2),
                0.0,
            )
            a = max(t_pairwise - b * pairwise_probe, 0.0)
            t_pairwise_est = a + b * pairwise_full
        else:
            t_pairwise_est = t_pairwise * pairwise_factor
    else:
        t_pairwise_est = t_pairwise * pairwise_factor

    Np_xy, Np_xx, Np_yy = _orbit_ip_grid_factors(
        dens_x.p[:K_probe_x], dens_y.p[:K_probe_y],
        sigma, is_rel, is_per, period,
    )
    orbit_probe = (Np_xy * float(K_probe_x) * float(K_probe_y)
                   + Np_xx * float(K_probe_x) * float(K_probe_x)
                   + Np_yy * float(K_probe_y) * float(K_probe_y))
    orbit_factor = (
        orbit_var / orbit_probe if orbit_probe > 0 else 1.0
    )

    t_orbit_est = t_orbit * orbit_factor

    if t_pairwise_est <= t_orbit_est * _PROBE_IP_MOBIUS_DECISION_MARGIN:
        return "bulger", True, t_pairwise_est, "probe"
    # Note: rel-per-above-threshold is handled by the measure-preference guard
    # above (it returns before reaching the probe), so the probe only runs
    # where Bulger and Möbius compute the same object; no measure warning here.
    return "mobius", True, t_orbit_est, "probe"