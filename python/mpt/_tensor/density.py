"""Path-selection cost model and dispatch helpers.

This module hosts the toolbox's path-selection policy --- the cost-
model + timing-probe logic that decides between the centres path and
the Möbius path for a given evaluation or inner-product call, plus the
small set of pure helpers (``_normalize_density_input``,
``_resolve_list_list_mode``, ``_compute_Q``, ``_format_time``) that
are shared between :mod:`._tensor.cosine`,
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



def _orbit_ips_impossible(ip_xy, ip_xx, ip_yy):
    """Cheap post-hoc check on Möbius-method-computed inner products.

    True when the three inner products take a combination of values no
    inner product can take. See ``_impossible_value_reason`` for the
    three conditions and the phrase each produces.

    The Möbius method's alternating partition sum can break down in two
    regimes:

    * σ → 0 with low K and r ≥ 3 (music-theoretical exact-match regime):
      auto-inner-product terms cancel to a value with magnitude near
      machine epsilon, and floating-point overflow can then produce
      arbitrary values when the cosine ratio is taken. This check
      catches that.
    * σ small relative to the data range: auto-inner-products lose 4–8
      decimal digits of precision while remaining finite and of
      plausible magnitude. This check does NOT catch that, because
      such values are merely inaccurate, not impossible. Accuracy in
      that regime is governed by ``truncationSigmas``.

    Parameters
    ----------
    ip_xy, ip_xx, ip_yy : float
        Cross and auto inner products from the Möbius method.

    Returns
    -------
    bool
        True if the inner products are unusable and the caller should
        fall back to Bulger's method.
    """
    return _impossible_value_reason(ip_xy, ip_xx, ip_yy) is not None


def _impossible_value_reason(ip_xy, ip_xx, ip_yy):
    """Describe why these inner products cannot be correct, or None.

    Returns a phrase naming the specific impossibility, for the warning
    the caller raises before rerouting. The three conditions are
    mathematically impossible rather than merely inaccurate, so the
    message says a defect occurred, not that accuracy was lost.

    ``ip_xx`` may be ``None`` when the first operand's self inner
    product was not computed (``normalize='oneSidedDenom'`` does not
    consume it); the checks that need it are then skipped and the
    remaining values are still validated.
    """
    if ip_xx is None:
        if not (np.isfinite(ip_xy) and np.isfinite(ip_yy)):
            return (f"an inner product is not finite "
                    f"(<X,Y> = {ip_xy!r}, <Y,Y> = {ip_yy!r})")
        if ip_yy < 0:
            return (f"<Y,Y> = {ip_yy:.3e} is negative, and a self inner "
                    f"product cannot be")
        return None
    if not (np.isfinite(ip_xy) and np.isfinite(ip_xx) and np.isfinite(ip_yy)):
        return (f"an inner product is not finite "
                f"(<X,Y> = {ip_xy!r}, <X,X> = {ip_xx!r}, "
                f"<Y,Y> = {ip_yy!r})")
    if ip_xx < 0 or ip_yy < 0:
        neg = "<X,X>" if ip_xx < 0 else "<Y,Y>"
        val = ip_xx if ip_xx < 0 else ip_yy
        return (f"{neg} = {val:.3e} is negative, and a self inner "
                f"product cannot be")
    denom = np.sqrt(ip_xx * ip_yy)
    if denom > 0 and abs(ip_xy) > 1.000001 * denom:
        return (f"the cosine similarity is {ip_xy / denom:.6f}, outside "
                f"[-1, 1]")
    return None
# Routing for the nested contraction is settled by _orbit_cost, a power-law
# model in the quantities each route works on, which additionally takes the
# batch extent. The measured K threshold tables that used to drive it are
# gone: they covered r = 2..6 only, and could not express a crossover that
# moves by up to 11 in K across the batch range. The flat multi-attribute and
# single-multiset inner-product paths make a whole-call decision instead,
# through the cost model and probe below, which account for N as well.


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
#
# Those measurements gave the two densities the same value count and the
# same event count, where the three matrices the inner product needs are
# each the size of the cross matrix. The fit was performed against the
# cross matrix alone, so the fitted figures (110 and 145 ns non-periodic,
# and the periodic pair) each absorb that factor of three. They are
# written below as those figures divided by three, which leaves every
# equal-count workload predicting exactly what it predicted when the fit
# was made, while _predict_pairwise_kernel_size now returns the total
# across the three matrices and so responds correctly when the counts
# differ.
_PW_PER_ENTRY_MS_NONPER = {2: 1.0e-4 / 3.0, 3: 1.4e-4 / 3.0}

_PW_PER_ENTRY_MS_PER = {2: 1.1e-4 / 3.0, 3: 1.6e-4 / 3.0}



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
# A single representative time per tuple order, deliberately
# size-independent. It is enough because the routing decision is
# structurally one-sided: Bulger's method is competitive only in a thin
# band at the smallest value counts of each order, and never by more
# than a small factor there; beyond it the Möbius orbit reduction wins
# by up to two or three orders of magnitude. Those large margins are
# ratios of operation counts (Bulger enumerates r!·C(K, r) tuples per
# density; the reduction does not), so they hold on any machine
# regardless of its absolute speed or thread count. The only cells a
# machine change can re-route are the near-crossover ones — where both
# methods are cheap and a wrong pick costs a fraction of a millisecond
# — so a size-dependent law fitted to one machine's timings buys correct
# routing only where routing barely matters, at the price of tracking
# hardware that does not transfer. A constant that gets the far-field
# sign right on every machine is the robust choice; the near-crossover
# flips are bounded-regret behaviour, not a defect. (A size-scaled
# absolute law was fitted and cross-validated during development and
# discarded for exactly this reason; see the project history if a
# per-route time *estimate*, as opposed to the routing sign, is ever
# needed.)


# Möbius method (relative modes): each relative attribute's
# (event_X, event_Y) inner matrices are computed by whichever of two
# routes is cheaper per event pair (see
# cosine._ma_rel_attr_prefers_centres): the pairwise closed form over
# materialised tuple-centres, or the slab-batched translation-grid
# contraction. The cost model prices both and takes the same minimum
# the orchestrator takes; three matrices (cross plus both self-norms)
# per attribute.
#: Cost model for the method comparison: one power law per route and
#: tuple order,
#:
#:     t_ms = exp(a_r) * term ** b_r
#:
#: on the quantity each route works over --- Bulger's method and the
#: tuple-centres route on the tuple-pair entries they materialise, the
#: translation grid on the node count times the larger value count. Every
#: term carries the event-pair count, since both methods price per pair.
#: The Möbius side takes the smaller of its two routes, as the
#: orchestrator does. Each law is fitted against the quantity the caller
#: passes, not an idealisation of it, so the intercepts absorb the
#: constant factors between them; a refit must use the same terms. Orders
#: above 4 reuse the r = 4 row.
#:
#: Fitted on 666 cells: r in {2, 3, 4}, value counts 5 to 40, event counts
#: 1 to 64, three kernel widths, both periodicities, three weight
#: profiles, equal and unequal value counts, each route timed in
#: isolation with ``rel_attr_route`` pinning it. Arms that could not run
#: are censored rather than dropped, a method that cannot run being
#: decisively the slower one.
#:
#: Cross-validated on the routing decision, eight-fold: 0.84 here against
#: 0.68 for the count-based model it replaces, and 0.93 against 0.62 on
#: the MATLAB side.
#:
#: Three earlier fits scored well and shipped badly, each because an axis
#: was missing from the sweep: equal value counts only, then one event
#: per density, then a build signature that flattened the events away.
#: Each regressed a case earlier work had fixed. Before refitting, run
#: tools/calibrate_rel_ip_cost.py --check: it asserts that each axis
#: varies what it claims to.
#:
#: The exponents are empirical and below what operation counts alone
#: would give --- Bulger's method enumerates M tuples per side, so a flop
#: count would say 1.0 against the 0.82 fitted here. They absorb
#: amortisation: these calls span 0.3 ms to tens of seconds, and the cost
#: per entry falls as the arrays grow. Fixing the exponents at their
#: structural values and adding an explicit overhead term was tried and
#: scores worse, so the shortfall is not fixed overhead in disguise.
#:
#: Constants are per-language: the two implementations amortise
#: differently. Refit with tools/calibrate_rel_ip_cost.py.
_REL_COST_LAW = {
    "bulger":  {2: (-8.5858, 0.8245), 3: (-8.1774, 0.7890),
                4: (-6.7951, 0.7278)},
    "centres": {2: (-8.8001, 0.8195), 3: (-10.0569, 0.9269),
                4: (-9.7867, 0.9251)},
    "grid":    {2: (-5.0440, 0.4817), 3: (-3.6893, 0.5881),
                4: (-3.2639, 0.7894)},
}


def _rel_route_cost_ms(route, r_a, term):
    """Predicted wall time in ms for one route, from its fitted law."""
    a, b = _REL_COST_LAW[route][min(max(int(r_a), 2), 4)]
    return float(np.exp(a) * max(float(term), 1.0) ** b)


def _predict_pairwise_kernel_size(r_vec, k_vec, A, N_x, N_y, k_vec_y=None,
                                  skip_xx=False, skip_yy=False):
    """Predicted tuple-pair kernel entries for the MA path under Bulger's
    method, summed over the three matrices the inner product needs.

    Each density contributes a permutation-side and a combination-side
    tuple count,

        n_J = N · ∏_a r_a! · C(K_a, r_a),    n_K = N · ∏_a C(K_a, r_a),

    and the three matrices are the cross matrix at n_J^X·n_K^Y and one
    self matrix per density at n_J^X·n_K^X and n_J^Y·n_K^Y. Where the two
    densities carry the same number of values in every attribute and the
    same number of events the three coincide, and the total is three
    times the cross term; where they do not, the larger density's self
    matrix dominates. For a five-value density against an 80-value one at
    r = 2 -- the ordinary shape of a chord compared against a scale -- the
    second self matrix holds 19971200 of the 20034600 entries, so pricing
    the cross matrix alone understates the work by 317.

    ``k_vec`` and ``k_vec_y`` are the two densities' per-attribute value
    counts; omitting the second means they agree, which is the
    aggregate-only legacy calling convention.

    ``skip_xx`` / ``skip_yy`` exclude the corresponding self matrix from
    the count: a self inner product that is already memoised on the
    density, or that the requested normalisation does not consume, costs
    nothing at call time and must not be priced (mispricing it steers
    near-crossover routing away from Bulger's method on exactly the
    repeated-context sweeps where Bulger's marginal cost is lowest).
    """
    if A == 0:
        return float(N_x * N_y)
    if k_vec_y is None:
        k_vec_y = k_vec
    perm_x = float(N_x)
    comb_x = float(N_x)
    perm_y = float(N_y)
    comb_y = float(N_y)
    for a in range(A):
        r_a = int(r_vec[a])
        K_x_a = int(k_vec[a])
        K_y_a = int(k_vec_y[a])
        if K_x_a < r_a or K_y_a < r_a:
            return float('inf')
        f_a = float(factorial(r_a))
        c_x = float(_math_comb(K_x_a, r_a))
        c_y = float(_math_comb(K_y_a, r_a))
        perm_x *= f_a * c_x
        comb_x *= c_x
        perm_y *= f_a * c_y
        comb_y *= c_y
    size = perm_x * comb_y
    if not skip_xx:
        size += perm_x * comb_x
    if not skip_yy:
        size += perm_y * comb_y
    return size



def _predict_orbit_cost_ms(
    r_vec, k_vec, A, N_x, N_y, rel_vec, nu_vec, centres_ok=True,
    k_vec_y=None, skip_xx=False, skip_yy=False,
):
    """Predicted Möbius-method MA wall time in milliseconds.

    Per-attribute sum. Absolute attributes with r_a >= 2 cost the
    vectorised-batch constant for their order (r_a = 1 attributes are
    a single direct kernel sum, absorbed into the base). Relative
    attributes cost three (N_x, N_y)-shaped matrices — the cross matrix
    and one self matrix per density — at the cheaper of the two per-pair
    routes the orchestrator itself chooses between: the tuple-centres
    closed form, at M_x·M_y + M_x² + M_y² ops per pair with
    M = r_a!·C(K_a, r_a) read from each density's own value count, or the
    slab-batched translation-grid contraction (nu_a·K_a² ops per matrix,
    with nu_a the caller's per-attribute grid node estimate).
    A-linearity of the absolute constants is verified at r = 2, 3 to
    within ~5 % and slightly sub-linear at r = 4, where linear-A
    over-predicts conservatively, biasing the dispatcher toward Bulger's
    method in close calls.

    ``k_vec`` and ``k_vec_y`` are the two densities' per-attribute value
    counts; omitting the second means they agree, which is the
    aggregate-only legacy calling convention. The centres term reads both
    because the second density's self matrix dominates it whenever that
    density carries more values. The grid term reads the first count
    alone, matching :func:`_mobius_inner._ma_rel_attr_prefers_centres`,
    the gate this function is predicting the outcome of; that leaves the
    grid route under-priced for unequal counts, and so the Möbius side
    under-priced wherever the grid route is the cheaper of its two, which
    is a bias toward the Möbius method. Whether the grid estimate should
    depend on the value count at all is the subject of the pending
    bench_ip_unit_cost extension.
    """
    # No flat relative base: each route's law carries its own intercept,
    # so adding one would double-count the setup it already prices.
    #
    # ``skip_xx`` / ``skip_yy`` exclude the corresponding self matrix
    # from the pricing (memoised on the density, or not consumed by the
    # requested normalisation), mirroring
    # :func:`_predict_pairwise_kernel_size`. The relative-attribute
    # terms drop the skipped self work exactly; the per-order absolute
    # constants were fitted on the full three-matrix computation, so
    # they are scaled by the fraction of matrices still to be computed
    # --- an approximation, and one that under-discounts (setup is not
    # per-matrix), which biases near-crossover routing toward Bulger's
    # method, the cheap-to-mispick side.
    n_matrices = 1 + (0 if skip_xx else 1) + (0 if skip_yy else 1)
    total = 0.0
    pairs = float(N_x) * float(N_y)
    if k_vec_y is None:
        k_vec_y = k_vec
    for a in range(A):
        r_a = int(r_vec[a])
        K_a = int(k_vec[a])
        K_y_a = int(k_vec_y[a])
        if bool(rel_vec[a]) and r_a >= 2:
            # The centres route is measure-blocked above the sigma/P
            # threshold (the orchestrator keeps the all-image grid
            # there), so above it the grid route is priced alone.
            # The smaller of the two routes, as the orchestrator takes.
            # Both terms read each density's own value count and carry the
            # event-pair count, since both routes price per pair.
            per_pair = _rel_route_cost_ms(
                "grid", r_a,
                pairs * float(nu_vec[a]) * max(K_a, K_y_a)
                * (n_matrices / 3.0))
            if centres_ok and K_a >= r_a and K_y_a >= r_a:
                m_x = float(factorial(r_a) * _math_comb(K_a, r_a))
                m_y = float(factorial(r_a) * _math_comb(K_y_a, r_a))
                centres_size = pairs * m_x * m_y
                if not skip_xx:
                    centres_size += pairs * m_x * m_x
                if not skip_yy:
                    centres_size += pairs * m_y * m_y
                per_pair = min(
                    per_pair,
                    _rel_route_cost_ms("centres", r_a, centres_size),
                )
            total += per_pair
        elif r_a >= 2:
            total += float(_ORBIT_ABS_PER_ATTR_MS[r_a]) * (n_matrices / 3.0)
    return float(total)



def _select_ma_inner_product_method(
    *,
    r_vec, k_vec, A,
    N_x, N_y,
    any_per, any_rel_nonper, any_rel_per,
    sigma_over_P_max, user_method,
    rel_vec=None, nu_vec=None,
    guard_forced_bulger=True,
    wrap_vec=None,
    k_vec_y=None,
    truncation_sigmas=None,
    return_costs=False,
    pw_skip_xx=False, pw_skip_yy=False,
    orbit_skip_xx=False, orbit_skip_yy=False,
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
        First density's per-attribute value count K_a (the kernel slab
        size; events within an attribute may have lower K_eff via NaN
        padding, which the Möbius-method wrapper handles via
        per-event safe/unsafe partition).
    k_vec_y : (A,) intp, optional
        Second density's per-attribute value count. The two densities
        need not agree — a chord against a scale, or a reference tuning
        against an equal division, is the ordinary case — and both
        Bulger's tuple-pair size and the Möbius side's centres term are
        products over the two. Omitted means the counts agree, which is
        the aggregate-only legacy calling convention.
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
    return_costs : bool, default False
        Also return the two predicted wall times in milliseconds that the
        comparison rests on, as ``(chosen, pw_cost_ms, orbit_cost_ms)``.
        Both are NaN on the returns that decide without pricing, so a
        caller can tell a structural decision from a costed one. Exposed
        for calibration: fitting the cost model needs the prediction
        beside the measurement.
    """
    if user_method != 'auto':
        return (user_method, float('nan'), float('nan')) \
            if return_costs else user_method
    r_max = int(np.max(r_vec)) if A > 0 else 1
    if r_max <= 1:
        return ('bulger', float('nan'), float('nan')) \
            if return_costs else 'bulger'
    if r_max > _ORBIT_R_MAX_SHIPPED:
        if guard_forced_bulger:
            _guard_forced_bulger_feasible_ma(
                k_vec, r_vec, rel_vec, N_x, N_y,
                reason="r above the shipped orbit order",
                k_vec_y=k_vec_y,
            )
        return ('bulger', float('nan'), float('nan')) \
            if return_costs else 'bulger'
    # Accuracy is governed by ``truncationSigmas``, not by the collection
    # size: the Möbius method's agreement with enumeration tracks the
    # truncation budget and is closest at K_a = r_a. The route is
    # therefore chosen on cost alone from here on.
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
                      and sigma_over_P_max
                      > _orbit_sigma_over_p_threshold(truncation_sigmas)):
        k_y = k_vec if k_vec_y is None else k_vec_y
        tuples_x = 1.0
        tuples_y = 1.0
        dim_sum = 0
        for a in range(A):
            r_a = int(r_vec[a])
            f_a = float(factorial(r_a))
            tuples_x *= f_a * float(_math_comb(int(k_vec[a]), r_a))
            tuples_y *= f_a * float(_math_comb(int(k_y[a]), r_a))
            dim_sum += r_a
        # "Either side" is meant literally: each density's working set is
        # its own event count times its own tuple count, and the two
        # densities need not carry the same number of values.
        n_J_max = max(int(N_x) * tuples_x, int(N_y) * tuples_y)
        # working-set bytes ≈ n_J · (2·Σr_a) · 8 (perm + centres + index
        # arrays, mirroring the single-multiset row-factor), capped to avoid overflow.
        if n_J_max * (2 * max(dim_sum, 1)) * 8 > _CENTRES_WORKING_SET_SOFT_BUDGET:
            return ('mobius', float('nan'), float('nan')) \
                if return_costs else 'mobius'
    # Wrap-based override for rel-per attributes at high sigma/P (v3+).
    # ``wrap='single-image'`` means the user wants (A) minimum-image
    # pairwise-wrap, computed by Bulger's method. ``wrap='full-image'``
    # (the default) means the user wants (C) all-image, computed by
    # the Möbius method. At low sigma/P the two agree numerically so
    # either method is fine; above the threshold they diverge and the
    # wrap axis picks the intended measure. Legacy callers without a
    # wrap vector see no change: the pre-v3 dispatch order stands.
    if wrap_vec is not None and any_rel_per and rel_vec is not None:
        wants_single = any(
            (wrap_vec is not None and w == 'single-image' and rel_vec[a])
            for a, w in enumerate(wrap_vec) if a < len(rel_vec)
        )
        wants_full = any(
            (w == 'full-image' and rel_vec[a])
            for a, w in enumerate(wrap_vec) if a < len(rel_vec)
        )
        if wants_single and wants_full:
            raise ValueError(
                "Mixed rel-per wrap on a single density is not yet supported; "
                "all rel-per attributes must share a wrap value."
            )
        if sigma_over_P_max > _orbit_sigma_over_p_threshold(truncation_sigmas):
            if wants_single:
                return ('bulger', float('nan'), float('nan')) \
                    if return_costs else 'bulger'
            if wants_full:
                return ('mobius', float('nan'), float('nan')) \
                    if return_costs else 'mobius'
    # A self inner product that is memoised on its density, or that the
    # requested normalisation does not consume, costs nothing at call
    # time; the per-route skip flags exclude it from that route's price.
    # The flags are per-route because the two routes' memoised values
    # live under different cache keys (their self-IP scales differ by a
    # constant prefactor that cancels only within one route's triple).
    pw_size = _predict_pairwise_kernel_size(
        r_vec, k_vec, A, N_x, N_y, k_vec_y=k_vec_y,
        skip_xx=pw_skip_xx, skip_yy=pw_skip_yy)
    # Priced by the same fitted law. The per-entry form this replaces
    # assumed a fixed cost per kernel entry; measurement contradicts that,
    # the per-entry cost falling as the arrays grow, which is what the
    # fitted exponent below 1 carries.
    pw_cost_ms = _rel_route_cost_ms("bulger", r_max, pw_size)
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
        centres_ok=(sigma_over_P_max
                    <= _orbit_sigma_over_p_threshold(truncation_sigmas)),
        k_vec_y=k_vec_y,
        skip_xx=orbit_skip_xx, skip_yy=orbit_skip_yy,
    )
    chosen = 'bulger' if pw_cost_ms <= orbit_cost_ms else 'mobius'
    if return_costs:
        return chosen, float(pw_cost_ms), float(orbit_cost_ms)
    return chosen



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
    is the ``(r_inner-1)``-row position-0 reduction of an ``r_inner``-tuple, so
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
      "position 0 anchored" reduction ``D[k] = d_{k+1} − d_0`` of an
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
      first-position zero is materialised.
    - **rel periodic (Eq 6 of the preprint):** the pairwise-wrap
      form, ``Q = sum_{i<j in {1, ..., r}} wrap(d_i - d_j)**2 / r``.
      Pairwise wrapping (not component-wise outer wrap) is what
      preserves exact transposition invariance on the circle. In
      reduced form the implicit position 0 contributes pairs
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
                # Position-0 pairs (0, k+1) for k = 0..r-2: each contributes
                # wrap(D[k])^2 (since wrap(-x)^2 = wrap(x)^2). Vectorise
                # the wrap-and-sum across all position-0 pairs in one numpy
                # pass on D as a whole.
                position0_wrapped = D - p_g * np.floor(D / p_g + half)
                Q = np.sum(position0_wrapped ** 2, axis=0)
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
#                     decomposition; ``_ip_core``). In relative
#                     periodic mode this evaluates the kernel that
#                     wraps the pairwise component differences, which
#                     is an approximation to the density's definition:
#                     the relative periodic density is the
#                     transposition average of the absolute periodic
#                     density, and the two coincide only as
#                     sigma/P -> 0. The wrapped-difference form is
#                     cheaper, so the toolbox uses it in that regime,
#                     but as sigma/P grows it ceases to be
#                     positive-definite --- its cosine similarity
#                     exceeds 1 for some density pairs from around
#                     sigma/P = 0.06 --- and is then not an inner
#                     product at all. The transposition average, being
#                     an average of Gaussians, is positive-definite at
#                     every sigma/P.
#    method='direct' : forces direct enumeration (no Möbius cancellation;
#                     useful for diagnosing near-zero cosines).
#                     In the single-multiset path, 'direct' coincides
#                     with 'bulger' (both route through ``_ip_core``);
#                     the distinction surfaces in later windowed paths.
#
#  cancellation_threshold : accepted for backward compatibility; it
#    affects neither the result nor the route. Accuracy is governed by
#    ``truncationSigmas`` --- the Möbius method's agreement with
#    enumeration tracks the truncation budget --- and the route is
#    chosen on cost.

_ORBIT_R_MAX_SHIPPED = 8  # orbit tables r=2..8 ship pre-built

#: Measured departure of the wrapped-difference relative-periodic kernel
#: from the transposition average that defines the measure, on the value
#: scale. Worst over tuple orders 2 to 4, value counts 4 to 12, and six
#: weight profiles --- flat, linear, two exponential rolloffs, bimodal,
#: and a single dominant value at 1000:1 --- from
#: tools/calibrate_sigma_over_p.py.
#:
#: The two entries below 0.04 sit at floating-point noise rather than at
#: a measured departure, so they bound the threshold no more tightly
#: than the arithmetic allows. The departure peaks at r = 3 rather than
#: at the largest order, so a calibration taken at r = 2 alone would be
#: too loose.
#:
#: The table does not set the limit at every accuracy setting. At
#: truncation_sigmas = 4 the departures admit 0.055, and at 2 they admit
#: 0.100; both are cut to 0.05 by ``_REL_PER_PD_CEILING`` below. So the
#: table binds at truncation_sigmas = 5 and tighter, and the ceiling
#: binds at 4 and looser.
_REL_PER_DEPARTURE = (
    (0.020, 1.67e-16),
    (0.030, 4.14e-14),
    (0.040, 3.27e-08),
    (0.050, 1.82e-05),
    (0.055, 1.09e-04),
    (0.060, 3.99e-04),
    (0.065, 1.03e-03),
    (0.070, 3.08e-03),
    (0.080, 1.10e-02),
    (0.100, 4.07e-02),
)

#: Hard ceiling on σ/P for the wrapped-difference form, whatever
#: accuracy is asked for.
#:
#: Beyond it the kernel stops being positive-definite: its induced
#: cosine similarity exceeds 1, so Cauchy-Schwarz fails and the quantity
#: is not a similarity at all. That is a failure of admissibility rather
#: than of accuracy, and no ``truncation_sigmas`` setting has authority
#: to loosen it. Three searches over the same grid have first reached a
#: violation at σ/P = 0.06, 0.07 and 0.08 respectively. A search only
#: ever bounds the onset from above --- failing to find a violation
#: proves nothing, and the spread across runs shows how little a single
#: onset settles --- so the ceiling sits below the earliest onset
#: anyone has found.
#:
#: This is not merely a backstop: at truncation_sigmas = 4 and looser it
#: is the binding test, since the departures admit 0.055 and 0.100 there
#: while the ceiling admits 0.05.
_REL_PER_PD_CEILING = 0.05


def _orbit_sigma_over_p_threshold(truncation_sigmas=None,
                                  return_binding=False):
    """σ/P above which the wrapped-difference form is inadmissible.

    Two tests, the stricter winning. The accuracy test is the toolbox's
    usual one: the departure from the defining measure must sit inside
    the floor ``truncation_sigmas`` implies, judged as an absolute error
    on the value scale. The positive-definiteness test is a fixed
    ceiling, since a form that is not an inner product cannot be made
    into one by relaxing a tolerance.

    Accuracy is the binding test at the factory default and at every
    tighter setting, giving 0.03 at the default; the ceiling binds at
    truncation_sigmas = 4 and looser. So this returns a threshold that
    tightens as the caller asks for more accuracy, where a single
    constant could only be right at one setting. The shipped 0.03 was
    the value at the default.

    The table is the calibration; no functional form is fitted to it,
    and the largest entry inside the floor is taken rather than
    interpolated, so the answer is always one the measurements support.

    With ``return_binding``, also returns which of the two tests set the
    answer: ``'accuracy'`` or ``'positive-definiteness'``. The caller
    that reports a routing decision needs to say why the limit is what
    it is, and the two carry different weight --- an accuracy limit
    moves with ``truncation_sigmas`` and the ceiling does not.
    """
    from .._defaults import truncation_floor
    floor = truncation_floor(truncation_sigmas)
    admissible = [sop for sop, dev in _REL_PER_DEPARTURE if dev <= floor]
    limit = max(admissible) if admissible else _REL_PER_DEPARTURE[0][0]
    if not return_binding:
        return min(limit, _REL_PER_PD_CEILING)
    if _REL_PER_PD_CEILING < limit:
        return _REL_PER_PD_CEILING, "positive-definiteness"
    return limit, "accuracy"

#: σ/P beyond which the absolute-periodic single-image (minimum-image)
#: measure departs materially from the full-image measure --- the sum of
#: the Gaussian kernel over every periodic image of the difference.
#:
#: Set at 0.05 on two independent grounds, whichever binds first:
#:
#: - Accuracy. Below it the two measures agree to within the toolbox's
#:   own floor: the cosine differs by 0.0 up to σ/P = 0.03 and by 2.5e-13
#:   at 0.05, against an accuracy floor of ~1e-12 at
#:   ``truncation_sigmas=inf`` and ~1.5e-8 at the default of 6. It rises
#:   to 2.6e-6 at σ/P = 0.08 and 8.5e-5 at 0.10.
#: - Positive definiteness. The single-image kernel's Fourier
#:   coefficients on the circle are non-negative only up to about this
#:   point; above it they go negative (-4.9e-5 at σ/P = 0.10, -1.4e-2 at
#:   0.20), so by Bochner's theorem the kernel is not the autocorrelation
#:   of any density and the form it induces is not an inner product.
#:   Cauchy-Schwarz then fails: cosines of 1.07 at σ/P = 0.20 and 1.12 at
#:   0.30 are reachable with ordinary non-negative weights.
_ABS_PER_SIGMA_OVER_P_THRESHOLD = 0.04


def _warn_abs_per_single_image(sigma_over_P, *, stacklevel=3):
    """Warn that an absolute-periodic attribute has been opted into
    the single-image (minimum-image) measure at a σ/P where that
    departs from the full-image measure (the v3+ default).

    Raised at density construction rather than at any one operation,
    because the choice is a property of the density: the absolute
    periodic kernel wraps each difference to its nearest image, so
    everything computed downstream --- evaluation, inner product,
    entropy --- inherits it.

    Fires only when the user has explicitly passed
    ``wrap='single-image'`` on this attribute. The default full-image
    measure is positive definite by construction; the warning is silent
    in the ordinary case. The two measures agree below the threshold, so
    the warning is also silent in the range musical work normally
    occupies.

    The consequence worth acting on is not only the size of the
    departure but its character. Above roughly σ/P = 0.15 the
    single-image kernel is no longer positive definite, so a cosine
    computed from it is not constrained to [-1, 1] and can exceed 1.
    See ``_ABS_PER_SIGMA_OVER_P_THRESHOLD`` for the measurements.
    """
    warnings.warn(
        f"σ/P = {sigma_over_P:.3f} exceeds "
        f"{_ABS_PER_SIGMA_OVER_P_THRESHOLD}: this absolute-periodic "
        f"attribute has been opted into the single-image (minimum-image) "
        f"measure, which above this σ/P departs from the full-image "
        f"measure that sums the kernel over every periodic image (the "
        f"two agree below it). Everything computed from this density "
        f"inherits the choice. Above roughly σ/P = 0.15 the single-image "
        f"kernel also stops being positive definite, so a cosine "
        f"similarity computed from it is not bounded by 1. Drop "
        f"``wrap='single-image'`` (the v3+ default is the full-image "
        f"measure) or reduce sigma relative to the period if the measure "
        f"matters at this scale.",
        stacklevel=stacklevel,
    )


def _warn_rel_per_all_image(*args, **kwargs):
    """Retired in v3.

    Retained as a no-op for backward compatibility with any external
    call sites; the substitution it warned about is no longer a
    substitution: rel-per full-image (C) is the toolbox's default
    measure and the ``wrap='single-image'`` opt-in gives (A) explicitly.
    """
    return None


class SingleImageInfeasibleError(MemoryError):
    """Raised when the single-image (minimum-image) measure is the only
    available route but its materialisation would exhaust memory.

    Arises at high tuple order in relative-periodic (and, for the inner
    product, any) mode when the Möbius method is *unavailable* --- refused
    by the feasibility bound (``r`` above the shipped/feasible orbit
    order) --- so no cheaper
    all-image substitute exists, and the exact single-image route
    (``centres`` for evaluation, ``bulger`` for the inner product) would
    need an infeasibly large tuple(-pair) kernel. There is no correct
    cheaper answer to fall back to (unlike the σ/P convention case, where
    the all-image form is a legitimate cheaper measure), so the honest
    outcome is a clear error rather than an out-of-memory crash.
    """


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
_MA_COST_CENTRES_SETUP_MS = 0.0250
_MA_COST_CENTRES_CALL_PER_JOINT_MS = 4.877e-5
#: Per-query cost has a floor no culling removes --- the bucket lookup
#: and gather each query pays --- plus a term linear in the joint tuple
#: count, and the two kernels carry different constants: the
#: non-periodic kernel is bucket-culled, the periodic one runs dense.
#: One shared pair cannot express that.
#:
#: Fitted on 220 cells of tools/calibrate_ma_eval_cost.py spanning sigma
#: from 3 to 120 cents over spans of 600 to 9600 cents. MATLAB carries
#: its own values, fitted the same way on its own measurements.
_MA_COST_CENTRES_QUERY_BASE_MS = 1.158e-3
_MA_COST_CENTRES_QUERY_PER_JOINT_MS = 9.805e-6
_MA_COST_CENTRES_QUERY_BASE_PER_MS = 2.854e-4
_MA_COST_CENTRES_QUERY_PER_JOINT_PER_MS = 8.822e-6

#: Per-attribute per-query overhead of the factored centres route
#: (bucket lookup and gather in the culled per-attribute kernel),
#: fitted to measured wall times of that route.
_MA_COST_CENTRES_FACTORED_QUERY_BASE_MS = 1.5e-3
# Culling geometry factor for the non-periodic kernels. The truncated
# kernel visits only the centres inside a ball of radius k*sigma about
# the query, so the surviving share is a volume ratio in the attribute's
# own dimension, r_a - [rel]_a, capped at 1. Applies to the per-query
# term only --- all joint centres are still materialised, so the setup
# and call terms are unculled.
#
# Fitted independently in each language and agreeing to about 2 per cent
# (25.6 in MATLAB against 26.2 here), which is what a geometric factor
# should do: it describes the truncation ball, not the implementation.
# Held one geometry out at a time it lands between 24.3 and 44.5.
_MA_COST_CENTRES_CULL_C = 26.2

#: Möbius: a per-call setup that scales with the partition count B_r,
#: and per-query work linear in the distinct-block op count
#: ``(2^r - 1) r K`` (each distinct block's factor is computed once and
#: reused across partitions). Relative attributes multiply the
#: per-query work by the u-grid node count; each node costs the cheaper
#: of the direct strategy (op-count linear) and, non-periodically, the
#: factored strategy (K-free after tabulation).
_MA_COST_MOBIUS_SETUP_MS = 0.0541
_MA_COST_MOBIUS_SETUP_PER_BELL_MS = 0.0208
_MA_COST_MOBIUS_QUERY_PER_OP_MS = 1.732e-6
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
_MA_COST_MOBIUS_REL_NODE_DIRECT_PER_OP_MS = 1.528e-6
_MA_COST_MOBIUS_REL_NODE_DIRECT_PER_OP_PER_MS = 2.532e-6
_MA_COST_MOBIUS_REL_NODE_FACTORED_PER_BELL_MS = 5.069e-5

#: u-grid tabulation setup, paid once per call: building the interpolation
#: table costs K source evaluations over the N_u grid nodes. In Python the
#: per-query node cost is large and already prices small-n_q calls
#: near-realistically, so the measured tabulation setup is negligible and
#: this constant is ~0; MATLAB's lean per-query readback leaves the setup
#: as the dominant Möbius cost at small n_q, so its twin constant is
#: nonzero. Same term, per-language magnitude.
_MA_COST_MOBIUS_REL_TABULATION_PER_NODE_MS = 6.953e-6

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


def _guard_forced_bulger_feasible_ma(k_vec, r_vec, rel_vec, N_x, N_y, *,
                                     reason, k_vec_y=None):
    """Raise if a *forced* multi-attribute Bulger inner product would be
    infeasible.

    Bulger's MA inner product materialises each side's joint perm-side
    working set ``n_J = N * prod_a nj_a`` with ``nj_a = K_a! / (K_a -
    r_a)!``, and the tuple-pair kernel is ``n_J_x * n_J_y`` float64
    entries. When the Möbius method is *forced* off (r above the
    shipped orbit order) there is no
    cheaper all-image substitute; at high tuple order the pair kernel
    can exhaust memory. Rather than let it crash the process, raise a
    clear error naming the shape. Explicit ``method='bulger'`` overrides
    are honoured earlier and do not reach here, so this guards only
    auto-dispatch.

    Each side is sized from its own density's per-attribute value counts,
    since the two need not agree; ``k_vec_y`` omitted means they do.
    """
    A = len(r_vec)
    k_y = k_vec if k_vec_y is None else k_vec_y

    def _nj_side(N, k_side):
        n_j = float(N)
        for a in range(A):
            K_a = int(k_side[a])
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

    nj_x = _nj_side(N_x, k_vec)
    nj_y = _nj_side(N_y, k_y)
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
    def _attr_spread(a):
        p_attr = getattr(dens, "p_attr", None)
        if p_attr is not None and a < len(p_attr) and p_attr[a] is not None:
            arr = np.asarray(p_attr[a], dtype=np.float64)
            if arr.size:
                return float(np.max(arr) - np.min(arr))
        return 0.0

    def _attr_cull(a):
        """Share of attribute ``a``'s tuple set a query reaches.

        The truncated non-periodic kernel visits only the centres inside
        a ball of radius ``k * sigma`` about the query, so the surviving
        share is a volume ratio in the attribute's own dimension,
        ``r_a - [rel]_a``. The periodic kernel is not truncated --- it
        sums over images rather than discarding a tail --- so it takes
        no discount.
        """
        if is_per[a] or sigma[a] <= 0:
            return 1.0
        spread = _attr_spread(a)
        if spread <= 0:
            return 1.0
        dim = max(1, int(r_vec[a]) - (1 if is_rel[a] else 0))
        return min(1.0, (_MA_COST_CENTRES_CULL_C * sigma[a] / spread) ** dim)

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
                    + (_MA_COST_CENTRES_QUERY_BASE_PER_MS if is_per[a]
                       else _MA_COST_CENTRES_QUERY_BASE_MS)
                    + (_MA_COST_CENTRES_QUERY_PER_JOINT_PER_MS if is_per[a]
                       else _MA_COST_CENTRES_QUERY_PER_JOINT_MS)
                    * T_a * _attr_cull(a)
                )
            )
    else:
        # Joint materialisation. A query reaches a joint centre only if
        # it reaches that centre in every attribute, so the joint culled
        # fraction is the product of the per-attribute ones.
        cull_joint = 1.0
        for a in range(A):
            cull_joint *= _attr_cull(a)
        any_per = any(bool(is_per[a]) for a in range(A))
        q_base = (_MA_COST_CENTRES_QUERY_BASE_PER_MS if any_per
                  else _MA_COST_CENTRES_QUERY_BASE_MS)
        q_per_joint = (_MA_COST_CENTRES_QUERY_PER_JOINT_PER_MS if any_per
                       else _MA_COST_CENTRES_QUERY_PER_JOINT_MS)
        centres_ms = (
            _MA_COST_CENTRES_SETUP_MS
            + _MA_COST_CENTRES_CALL_PER_JOINT_MS * joint_tuples
            + n_q_eff * (q_base + q_per_joint * joint_tuples * cull_joint)
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
            # The spectral (Fourier) strategy engages inside the mobius
            # relative evaluator for r_a in 2..4 above its query
            # thresholds (see the gate in mpt._mobius.eval_orbit_rel);
            # where it would engage, price the per-query cost with its
            # measured, K-free slope. The slope scales with the mode
            # count, i.e. with window/sigma (session-calibrated: at
            # window/sigma ~ 270, measured ~0.06, ~0.6, and ~1.8
            # ms/query for r = 2, 3, 4 at the 6-sigma floor). The
            # r = 2 constant is the geometric mean of the two
            # calibration configs (window/sigma 270 and 131), whose
            # setup shares differ; the crude linear-in-modes model sits
            # within ~2x of both.
            _four_per_mode = {2: 2.613e-5, 3: 4.522e-4, 4: 1.138e-4}
            # Periodic-only K term (per r) added to the K-free slope: the
            # per-event spectrum build carries K, which the fixed-period
            # window does not absorb. In periodic mode the spectral branch
            # fires for r = 2 and 3 (the r = 4 mode grid exceeds the memory
            # guard and falls to the node path, so its K growth is priced
            # there, not here). Fitted from the periodic engaging cells of
            # bench_ma_eval_calibration: the per-query cost rises linearly
            # in K with slope ~0.28 ms/K at r = 2 and ~0.60 at r = 3 over
            # window/sigma = 80, nQ = 200.
            _FOUR_PER_PERIODIC_K_MS = {2: 4.897e-5, 3: 1.501e-4, 4: 0.0}
            _four_minq = {2: 16, 3: 32, 4: 64}
            spread = 0.0
            p_a = getattr(dens, "p_attr", None)
            if p_a is not None and a < len(p_a) and p_a[a] is not None:
                arr = np.asarray(p_a[a], dtype=np.float64)
                if arr.size:
                    spread = float(np.max(arr) - np.min(arr))
            if is_per[a] and period[a] > 0:
                window = float(period[a])
            else:
                window = 2.0 * spread + 16.0 * sigma[a]
            # The spectral branch stands down when its own mode grid would
            # exceed the memory guard (see _SPECTRAL_IP_MAX_POINTS in
            # cosine.py): the grid is (r_a - 1)-dimensional, so at small
            # sigma/P and r_a = 4 it can pass the query/K thresholds yet
            # still decline, falling through to the u-grid node path. Mirror
            # that decline here so the cost model prices the path that
            # actually runs, not the spectral one it would otherwise assume.
            from ._mobius_inner import (_SPECTRAL_IP_MODE_SIGMAS as _mode_sig, _SPECTRAL_IP_MAX_POINTS as _max_pts)
            if is_per[a] and period[a] > 0:
                _L = float(period[a])
            else:
                _L = 2.0 * spread + 2.0 * (_mode_sig + 2.0) * sigma[a]
            _M = int(np.ceil(_mode_sig / np.sqrt(2.0)
                             * _L / (2.0 * np.pi * sigma[a]))) + 2
            _spectral_fits = (2 * _M + 1) ** (r_a - 1) <= _max_pts
            if (r_a in _four_per_mode and n_q >= _four_minq[r_a]
                    and k_vec[a] >= (2, 8, 16)[r_a - 2]
                    and _spectral_fits):
                # The K-free slope prices the per-pair matmul, but the
                # per-event spectrum build A_m(eta) = sum_i w^m
                # exp(-i eta p_i) carries K. In periodic mode the window
                # is fixed at the period, so that K-dependence is not
                # already absorbed through the window and shows up as an
                # underprice growing with K (measured pred/actual ~0.1 by
                # K = 48). Add a periodic K term where the spectral branch
                # fires (r = 2 and 3); r = 4 declines to the node path in
                # periodic mode and is priced there.
                four_ms = (_four_per_mode[r_a]
                           * (window / max(sigma[a], 1e-12)) * n_q_eff)
                if is_per[a] and _FOUR_PER_PERIODIC_K_MS.get(r_a, 0.0):
                    four_ms += (_FOUR_PER_PERIODIC_K_MS[r_a] * k_vec[a]
                                * (window / max(sigma[a], 1e-12)) * n_q_eff)
                mobius_ms += _MA_COST_MOBIUS_SETUP_MS + four_ms
                continue
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


def _has_ordered_attr(dens) -> bool:
    """True if any flat attribute is ordered (``[sym] = 0``) at ``r > 1``.

    Such an attribute carries no position-permutation symmetry, so the
    Möbius orbit decomposition does not apply to it: the partition sum
    realises the symmetrised tuple set, which is a *different* density
    rather than the same one computed faster. ``r = 1`` is exempt
    ([sym] is vacuous at a single value), as are nested attributes, whose
    per-attribute density is built by contraction rather than by a
    single Möbius sum.
    """
    A = int(dens.n_attrs)
    is_sym = np.atleast_1d(
        getattr(dens, "is_sym", np.ones(A, dtype=bool))
    )
    r_vec = np.atleast_1d(dens.r)
    nested = getattr(dens, "nested", [None] * A)
    for a in range(A):
        if nested[a] is None and not bool(is_sym[a]) and int(r_vec[a]) > 1:
            return True
    return False


def _reject_ordered_for_mobius(dens) -> None:
    """Raise if an explicit ``method='mobius'`` names an ordered density.

    Silently substituting the centres path would hide the fact that the
    requested method does not apply; silently proceeding would return
    the symmetrised density's values. Both are worse than an error.
    """
    if _has_ordered_attr(dens):
        raise ValueError(
            "method='mobius' is not available for an ordered ([sym]=0) "
            "attribute at r > 1: the Möbius decomposition sums over set "
            "partitions of {1, ..., r}, which realises the "
            "symmetrised tuple set and so evaluates a different density. "
            "Use method='centres' (or method='auto', which selects it)."
        )


def _select_ma_eval(dens, n_q, *, method, truncation_sigmas=None):
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
        _reject_ordered_for_mobius(dens)
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

    # ---- Hard rule: ordered ([sym] = 0) attributes at r > 1 have no
    # orbit. The Möbius decomposition sums over set partitions of
    # {1, ..., r}, which counts every ordering of each block and so
    # realises the symmetrised tuple set; on an ordered attribute that
    # is a different density, not a faster route to the same one. Keep
    # the joint-centres path, which enumerates the ordered tuple set as
    # given. ----
    if _has_ordered_attr(dens):
        return "centres", "ordered ([sym]=0) attribute (no orbit to collapse)"

    # ---- Hard rule: nested attributes are not handled by the flat
    # factored evaluator; keep the joint-centres path. ----
    nested = getattr(dens, "nested", [None] * A)
    if any(nested[a] is not None for a in range(A)):
        return "centres", "nested attribute (flat Möbius not applicable)"

    # ---- Hard rule: r <= 1 on every attribute => Möbius is degenerate
    # (one singleton partition); centres is trivially cheap. ----
    if all(r_vec[a] <= 1 for a in range(A)):
        return "centres", "all r <= 1"

    # ---- Hard rule per attribute: feasibility forces the single-image
    # centres route, because the Möbius method is beyond its shipped
    # order there --- not merely slower. These forced-centres picks are
    # guarded against out-of-memory below: since no cheaper all-image
    # substitute exists here, an infeasible shape must raise a clear
    # error, not crash. (The σ/P convention is handled separately,
    # after the loop: there the all-image Möbius form is a legitimate
    # cheaper measure, so it becomes the preferred default rather than a
    # reason to force centres.) ----
    force_centres_reason = None
    for a in range(A):
        r_a = r_vec[a]
        if r_a < 2:
            continue  # r_a = 1 factor is exact either way
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
    # the cost model). The Möbius method computes the (C) full-image
    # (transposition-average) form; the centres path computes the (A)
    # single-image form. The v3+ default is full-image and the toolbox
    # prefers the Möbius route whenever it is available (it is cheaper
    # and memory-safe: the factored evaluator is n_j-free). If the user
    # has opted the rel-per attribute into ``wrap='single-image'`` the
    # centres route is selected instead, on the same threshold. ----
    wrap = getattr(dens, 'wrap', None)
    for a in range(A):
        if (is_rel[a] and is_per[a] and period[a] > 0
                and sigma[a] / period[a]
                > _orbit_sigma_over_p_threshold(truncation_sigmas)):
            wrap_a = (str(wrap[a]) if wrap is not None
                      and a < len(wrap) else 'full-image')
            if wrap_a == 'single-image':
                return "centres", "rel-per single-image measure (wrap opt-in)"
            return "mobius", "rel-per full-image measure"

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



