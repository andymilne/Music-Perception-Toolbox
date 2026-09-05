"""Cost model for the nested-attribute inner product.

The nested path of :mod:`mpt._tensor.cosine` chooses, per attribute, among
routes that all carry the attribute's *declared* measure, and then chooses,
for the density as a whole, between the contraction plan and the joint-tuple
enumeration. Until this module existed those choices were made on raw
analytic operation counts --- ``m_comb * m_perm`` against ``Q * work`` --- with
no fitted constants, no memory guard, and no enumeration candidate. This
module supplies the missing half, in the same shape as the flat model of
:mod:`mpt._tensor.dispatch`:

* one fitted power law ``t_ms = exp(a) * term ** b`` per route and coarse
  structure key, keyed the way ``_REL_COST_LAW`` is keyed (there by tuple
  order, here by the nested attribute's *total* tuple order);
* a per-route setup floor applied with ``max`` (never added), as
  ``_ORBIT_REL_FLOOR_MS`` is, because a multiplicative law carries no fixed
  cost and extrapolates below what the route can do once the term is small;
* a working-set guard that diverts away from the materialising route above
  ``_CENTRES_WORKING_SET_SOFT_BUDGET``, as the flat selector's guard does;
* the self-inner-product skip flags, so a memoised ``<X,X>`` or ``<Y,Y>`` is
  not priced.

**The measure rule is not part of this model.** The routes admissible for an
attribute are settled first, by
:func:`~mpt._tensor.cosine._nested_admissible_routes`; the cost model only
orders the survivors. A cheaper route to a different number is not a cheaper
route.

The nested rule is the flat one. ``wrap`` is consulted only above the
sigma/period threshold, where the two readings of "periodic" differ by more
than the truncation floor and each declaration has a single carrier; below it
they agree inside the floor, so both routes serve either declaration and the
price decides. The rule is pinned by ``tests/test_nested_measure_rule.py``.

Terms
-----
Every term is summed over the inner-product matrices this call will actually
compute --- the cross matrix always, each self matrix unless its skip flag is
set --- with each matrix carrying *its own* event-pair count (``N_x * N_y``,
``N_x ** 2``, ``N_y ** 2``). Writing ``m_perm`` and ``m_comb`` for the
per-event permutation- and combination-side tuple counts of the nested
attribute (:func:`~mpt._tensor._nested_contraction.tuple_counts`):

``centres``
    Kernel entries. Per event pair the route forms one
    ``(m_comb_X, m_perm_Y)`` array where Bulger's orbit restriction holds and
    one ``(m_perm_X, m_perm_Y)`` array where it does not. The restriction
    (:func:`~mpt._tensor._mobius_inner._comb_side_restriction`) is priced
    only where the route applies it: it is declined at ``|G| < 2`` (every
    level ordered), when the global ``_COMB_RESTRICTION_ENABLED`` switch is
    off, and when the build's perm side is not the free orbit tiling
    ``m_perm = |G| * m_comb`` of its comb side; in each of those cases the
    unrestricted ``m_perm_X * m_perm_Y`` is the price. Pricing the
    unrestricted form unconditionally over-priced the centres route by
    ``|G|`` and sent shapes to the tau-grid that the centres route would
    have computed more cheaply.

``taugrid``
    ``n_tau * work``: transposition-average nodes times the per-level
    contraction work of the recipe (the larger of the two sides'
    :func:`~mpt._tensor._nested_contraction.recipe_work`).

``contract_relnonper``
    The same product with the line grid's node count in place of ``n_tau``.

``contract``
    The same product with one node: the absolute kernel needs no quadrature.

``bulger``
    The joint tuple-pair kernel of the whole density, mirroring
    :func:`~mpt._tensor.dispatch._predict_pairwise_kernel_size` but with each
    nested attribute contributing its own ``m_perm`` / ``m_comb`` in place of
    ``r!*C(K, r)`` / ``C(K, r)``: this is what
    :func:`~mpt._tensor.cosine._cos_sim_exp_tens_ma_pairwise` enumerates, one
    perm-side and one comb-side tuple count per attribute, multiplied across
    attributes and by the event counts.

The laws are per-language and per-machine: they absorb interpreter overhead,
array layout and BLAS constants, exactly as the flat laws do. Refit with
``tools/calibrate_nested_cost.py`` and ``tools/fit_nested_cost.py``.
"""
from __future__ import annotations

import math

import numpy as np

#: Structure key for the fitted laws: the nested attribute's *total* tuple
#: order, ``prod(r_levels)`` --- the number of leaf positions a tuple carries,
#: which is what ``dens.r[a]`` holds. The per-entry and per-node costs both
#: grow with it (the relative quadratic form is ``O(R ** 2)`` per kernel
#: entry, the centres array ``O(R)`` per tuple), so it is the one coarse index
#: the routes share. Rows exist for the orders the calibration grid reaches;
#: a lookup takes the largest row at or below the requested order, and orders
#: below the smallest row use it. That is the nested analogue of the flat
#: laws' ``min(max(r, 2), 4)`` clamp, generalised because a nested attribute's
#: total order is a product of level arities and so skips values.
_NESTED_LAW_KEYS = (2, 3, 4, 6)


def _nested_cost_key(total_order):
    """Largest law row at or below ``total_order`` (never below the first)."""
    R = int(total_order)
    key = _NESTED_LAW_KEYS[0]
    for k in _NESTED_LAW_KEYS:
        if k <= R:
            key = k
    return key


#: Fitted per-route laws, ``t_ms = exp(a) * term ** b``, by structure key.
#:
#: Fitted September 2026 on three runs of ``tools/calibrate_nested_cost.py``
#: (3 x 258 cells) in the Cowork Linux VM on the maintainer's Mac, by
#: ``tools/fit_nested_cost.py --min-cells 90`` (a structure key with fewer
#: than 90 measured cells for a route takes that route's pooled law, which
#: is what all four taugrid and line-grid keys now carry). Predicted over
#: measured, geometric mean / spread / worst: centres 0.99 / 1.42 / 3.9,
#: taugrid 1.00 / 1.80 / 4.0, line grid 1.00 / 1.76 / 3.9, contract 0.97 /
#: 1.14 / 1.4, enumeration 0.98 / 2.37 / 9.1. Routing regret against the
#: measured oracle over the 702 cells: per-attribute route 1.032 (22 cells
#: beyond the oracle, worst 2.7x); plan against enumeration 1.045. The raw
#: operation-count rule this replaced scored 1.49 with a worst cell of 4.0x
#: on the sandbox grid. To re-measure and refit::
#:
#:     PYTHONPATH=. python3 tools/calibrate_nested_cost.py --out cal.csv
#:     PYTHONPATH=. python3 tools/fit_nested_cost.py cal.csv --min-cells 90
#:
#: What the dispatch consumes is the *ratio* between two routes, which is
#: far less sensitive to the machine than either number.
#:
#: **Cross-validation.** ``tools/fit_nested_cost.py --compare-pooling``
#: scores three pooling levels --- these per-key laws, one law per route
#: with the structure key dropped, and one exponent shared by all five
#: routes with per-route intercepts --- by held-out routing regret over 40
#: random halves, each at the ``--min-cells`` that is held-out optimal for a
#: half of that dataset. Per key wins on every dataset and both languages:
#: 1.043 +- 0.017 against 1.129 +- 0.052 (per route) and 1.100 +- 0.024
#: (shared exponent) on these VM cells; 1.044 +- 0.018 against 1.105 +-
#: 0.035 and 1.128 +- 0.035 on the MATLAB cells; 1.063 +- 0.033 against
#: 1.240 +- 0.205 and 1.137 +- 0.068 on a sandbox grid. So the per-key
#: dimension is kept. What the same sweep did move is ``--min-cells``: the
#: held-out optimum sits at 12--16% of the cells fitted, which is 90--120
#: for these 774 and not the 60 first shipped; at 90 four laws are dropped
#: and both regrets improve (1.060 to 1.032, worst cell 5.4x to 2.7x), the
#: earlier key-6 tau-grid row having been fitted on too few cells to hold
#: up.
#:
#: **Transfer.** Fitting on the sandbox grid and scoring on the VM cells
#: gives 1.033, and the reverse 1.029 --- at or below either machine's own
#: in-sample regret --- while the prediction log-ratio error across
#: machines is 1.4 to 1.5 in the log, a factor of four in absolute time.
#: The constants do not transfer; the ratios they are consumed as do.
_NESTED_COST_LAW = {
    "centres": {2: (-2.9601, 0.2919), 3: (-5.0127, 0.5157), 4: (-3.6949, 0.4074), 6: (-4.0540, 0.4886)},
    "taugrid": {2: (-7.3893, 0.6125), 3: (-7.3893, 0.6125), 4: (-7.3893, 0.6125), 6: (-7.3893, 0.6125)},
    "contract_relnonper": {2: (-7.2386, 0.5984), 3: (-7.2386, 0.5984), 4: (-7.2386, 0.5984), 6: (-7.2386, 0.5984)},
    "contract": {2: (-1.4719, 0.0472), 3: (-1.2901, 0.0001), 4: (-2.3945, 0.1239), 6: (-2.2457, 0.0969)},
    "bulger": {2: (-4.1758, 0.4428), 3: (-6.3699, 0.6668), 4: (-3.6142, 0.4489), 6: (-2.6820, 0.4636)},
}

#: Per-route setup floor in milliseconds, as ``(fixed, per_matrix)`` by
#: structure key: a call computing ``n_matrices`` of the three inner matrices
#: cannot go under ``fixed + per_matrix * n_matrices``. Applied with ``max``,
#: not added, for the reason ``_ORBIT_REL_FLOOR_MS`` gives: the laws above are
#: multiplicative in their term and so carry no fixed cost, while the routes
#: pay a recipe build, an orbit-table fetch and a per-attribute dispatch
#: whatever the term is. Taken from the smallest cells of the calibration
#: grid, where each route's measured time is flat in the term.
#:
#: Fitted with the laws above, from the smallest-term cells of each key.
_NESTED_FLOOR_MS = {
    "centres": {2: (0, 0.08577), 3: (0, 0.1201), 4: (0, 0.0861), 6: (0, 0.1289)},
    "taugrid": {2: (0, 0.1594), 3: (0, 0.1594), 4: (0, 0.1594), 6: (0, 0.1594)},
    "contract_relnonper": {2: (0, 0.1548), 3: (0, 0.1548), 4: (0, 0.1548), 6: (0, 0.1548)},
    "contract": {2: (0, 0.08933), 3: (0, 0.0895), 4: (0, 0.0604), 6: (0, 0.0691)},
    "bulger": {2: (0, 0.05703), 3: (0, 0.09857), 4: (0, 0.05533), 6: (0, 0.1006)},
}


#: Safety factor favouring the contraction plan at near-ties: the joint-tuple
#: enumeration is taken only when its estimate is below the plan's estimate
#: divided by this factor. The asymmetry mirrors ``_MA_MOBIUS_SAFETY``'s and
#: points the same way --- toward the route that does not materialise the
#: joint tuple set --- for two reasons.
#:
#: The enumeration builds ``n_J = N * prod_a m_perm_a`` tuples of
#: ``sum_a r_a`` rows on each side, which the contraction never does; a
#: near-tie in predicted time is not a near-tie in memory.
#:
#: And where the plan's relative-periodic attribute runs on the tau-grid, the
#: plan computes the declared all-image measure *exactly* while the
#: enumeration computes the minimum-image reading, admissible below the
#: sigma/period threshold only because the two agree inside the truncation
#: floor. Trading an exact measure for an approximate one is worth a real
#: saving, not a marginal one.
#:
#: The value is set at the scale of the near-crossover cells the calibration
#: found rather than fitted: on the small multi-attribute shapes where the
#: three routes are measured within about 1.5x of one another, the model's
#: ranking is not to be trusted to the percent.
_NESTED_ENUM_SAFETY = 2.0


def nested_route_cost_ms(route, total_order, term, n_matrices=3):
    """Predicted wall time in ms for one nested route, from its fitted law.

    ``term`` is the route's own term (see the module docstring); it already
    carries the event-pair counts and the matrix count, so ``n_matrices`` is
    used only by the floor.
    """
    key = _nested_cost_key(total_order)
    a, b = _NESTED_COST_LAW[route][key]
    t = float(np.exp(a) * max(float(term), 1.0) ** b)
    f, pm = _NESTED_FLOOR_MS[route][key]
    return max(t, f + pm * float(n_matrices))


# ----------------------------------------------------------------- counts


def _fact(m):
    return float(math.factorial(int(m)))


def _comb(n, k):
    return float(math.comb(int(n), int(k))) if 0 <= k <= n else 0.0


def _attr_tuple_counts(dens, a):
    """``(m_perm, m_comb)`` per event for attribute ``a`` of ``dens``.

    Nested attributes read the analytic
    :func:`~mpt._tensor._nested_contraction.tuple_counts`; flat ones the
    ``r!*C`` / ``C`` pair the build enumerates, with the ``r!`` dropped on an
    ordered attribute, whose perm side the build sets equal to its comb side.
    """
    from ._nested_contraction import tuple_counts
    nested = getattr(dens, "nested", None)
    spec = None if nested is None else nested[a]
    if spec is not None:
        mp, mc = tuple_counts(np.asarray(spec["r"]).ravel(),
                              np.asarray(spec["sym"]).ravel(),
                              np.asarray(spec["tags"]))
        return float(mp), float(mc)
    r_a = int(dens.r[a])
    K = int(np.asarray(dens.p_attr[a]).shape[0])
    is_sym = bool(np.asarray(
        getattr(dens, "is_sym", np.ones(int(dens.n_attrs), dtype=bool))
    ).ravel()[a])
    c = _comb(K, r_a)
    return (c * (_fact(r_a) if (is_sym and r_a > 1) else 1.0), c)


def _centres_restricted(dens, a, m_perm, m_comb):
    """True where the centres route may restrict its X side to combinations.

    The conditions are those
    :func:`~mpt._tensor._mobius_inner._comb_side_restriction` applies, read
    from the counts rather than from the materialised arrays: a non-trivial
    tuple-symmetry group, the global switch on, and the perm side the free
    orbit tiling ``m_perm == |G| * m_comb`` of the comb side.
    """
    from ._mobius_inner import _COMB_RESTRICTION_ENABLED, _nested_orbit_mult
    if not _COMB_RESTRICTION_ENABLED:
        return False, 1.0
    nested = getattr(dens, "nested", None)
    spec = None if nested is None else nested[a]
    if spec is not None:
        mult = float(_nested_orbit_mult(np.asarray(spec["r"]).ravel(),
                                        np.asarray(spec["sym"]).ravel()))
    else:
        r_a = int(dens.r[a])
        is_sym = bool(np.asarray(
            getattr(dens, "is_sym", np.ones(int(dens.n_attrs), dtype=bool))
        ).ravel()[a])
        mult = _fact(r_a) if (is_sym and r_a > 1) else 1.0
    ok = (mult >= 2.0 and m_comb > 0.0
          and abs(m_perm - mult * m_comb) < 0.5)
    return ok, mult


def _matrix_pairs(N_x, N_y, skip_xx, skip_yy):
    """The matrices this call computes, as ``(side_x, side_y, pair_count)``.

    ``side`` is 0 for the X density and 1 for the Y density. Each matrix
    carries its own event-pair count, which is where this differs from the
    flat model's shorthand of pricing every matrix at ``N_x * N_y``.
    """
    out = [(0, 1, float(N_x) * float(N_y))]
    if not skip_xx:
        out.append((0, 0, float(N_x) * float(N_x)))
    if not skip_yy:
        out.append((1, 1, float(N_y) * float(N_y)))
    return out


def nested_attr_terms(dens_x, dens_y, a, *, skip_xx=False, skip_yy=False,
                      truncation_sigmas=None):
    """Per-route terms for nested attribute ``a``, plus the shape it read.

    Returns ``(terms, info)``. ``terms`` maps each of ``'centres'``,
    ``'taugrid'``, ``'contract_relnonper'`` and ``'contract'`` to its term;
    ``info`` carries the counts the term was built from, for reporting and
    for the calibration harness. ``truncation_sigmas`` is the per-call
    width (``None`` takes the default); it enters through the quadrature
    node counts, as in the MATLAB ``internal.nestedCostTerms``.
    """
    from ._nested_contraction import (
        build_recipe, quad_nodes, recipe_work, auto_ntau_default,
        auto_taus_line)
    from .._defaults import truncation_floor

    is_rel = bool(dens_x.is_rel[a])
    is_per = bool(dens_x.is_per[a])
    sigma = float(dens_x.sigma[a])
    period = float(dens_x.period[a])
    N_x = int(dens_x.n)
    N_y = int(dens_y.n)
    R = int(dens_x.r[a])

    mp = [0.0, 0.0]
    mc = [0.0, 0.0]
    restricted = [False, False]
    works = [0.0, 0.0]
    for side, dens in ((0, dens_x), (1, dens_y)):
        mp[side], mc[side] = _attr_tuple_counts(dens, a)
        restricted[side] = _centres_restricted(dens, a, mp[side], mc[side])[0]
        nested = getattr(dens, "nested", None)
        spec = None if nested is None else nested[a]
        if spec is not None:
            works[side] = float(recipe_work(build_recipe(
                np.asarray(spec["r"]).ravel(),
                np.asarray(spec["sym"]).ravel(),
                np.asarray(spec["tags"]), is_rel, is_per)))
        else:
            works[side] = mp[side] * mc[side]

    ts = truncation_sigmas
    if is_rel and is_per:
        n_tau = float(auto_ntau_default(period, sigma, ts))
    else:
        n_tau = 1.0
    if is_rel and not is_per:
        tol = truncation_floor(ts)
        px = np.asarray(dens_x.p_attr[a], dtype=np.float64)
        py = np.asarray(dens_y.p_attr[a], dtype=np.float64)
        allv = np.concatenate([px[np.isfinite(px)].ravel(),
                               py[np.isfinite(py)].ravel()])
        n_line = float(np.size(auto_taus_line(allv, allv, sigma, tol)))
    else:
        n_line = 1.0

    terms = {"centres": 0.0, "taugrid": 0.0,
             "contract_relnonper": 0.0, "contract": 0.0}
    for sx, sy, pairs in _matrix_pairs(N_x, N_y, skip_xx, skip_yy):
        x_side = (mc[sx] if restricted[sx] else mp[sx])
        terms["centres"] += pairs * x_side * mp[sy]
        work = max(works[sx], works[sy])
        terms["taugrid"] += pairs * n_tau * work
        terms["contract_relnonper"] += pairs * n_line * work
        terms["contract"] += pairs * work

    info = dict(m_perm_x=mp[0], m_comb_x=mc[0], m_perm_y=mp[1],
                m_comb_y=mc[1], restricted_x=restricted[0],
                restricted_y=restricted[1], work_x=works[0], work_y=works[1],
                n_tau=n_tau, n_line=n_line, total_order=R,
                N_x=N_x, N_y=N_y)
    return terms, info


def nested_centres_working_set_bytes(dens_x, dens_y, a):
    """Peak bundle size of the nested centres route for attribute ``a``.

    The route rebuilds each side as a single-attribute density and holds its
    materialised perm-side centres ``(dim, N * m_perm)`` together with the
    matching weight and event-index arrays; ``dim`` is ``R`` for an absolute
    attribute and ``R - 1`` for a relative one, whose centres are anchored at
    position 0. The estimate is the larger side's centres array with the row
    factor of two the flat guard uses for its index and weight companions
    (:func:`~mpt._tensor.dispatch._select_ma_inner_product_method`), so the
    two guards read the same quantity in the same units.
    """
    is_rel = bool(dens_x.is_rel[a])
    R = int(dens_x.r[a])
    dim = max(1, R - (1 if is_rel else 0))
    n_j_x = float(dens_x.n) * _attr_tuple_counts(dens_x, a)[0]
    n_j_y = float(dens_y.n) * _attr_tuple_counts(dens_y, a)[0]
    return max(n_j_x, n_j_y) * (2.0 * dim) * 8.0


def price_nested_attr(dens_x, dens_y, a, admissible, *,
                      skip_xx=False, skip_yy=False, truncation_sigmas=None):
    """Price the admissible routes for nested attribute ``a``.

    ``admissible`` is the list of routes that carry the attribute's declared
    measure, in the order the caller wants ties broken; the measure rule has
    already been applied, so nothing here can change the number computed.
    Returns ``(route, cost_ms, prices, info)``, where ``prices`` maps every
    admissible route to its predicted milliseconds --- ``inf`` for a
    materialising route diverted by the working-set guard, which is how the
    guard shows up in a report.
    """
    from .dispatch import _CENTRES_WORKING_SET_SOFT_BUDGET

    terms, info = nested_attr_terms(dens_x, dens_y, a,
                                    skip_xx=skip_xx, skip_yy=skip_yy,
                                    truncation_sigmas=truncation_sigmas)
    n_matrices = 1 + (0 if skip_xx else 1) + (0 if skip_yy else 1)
    ws = nested_centres_working_set_bytes(dens_x, dens_y, a)
    info["centres_working_set_bytes"] = ws
    prices = {}
    for route in admissible:
        cost = nested_route_cost_ms(route, info["total_order"], terms[route],
                                    n_matrices)
        # Memory guard: the centres route is the only materialising route
        # here, so it is the only one the soft budget can divert. It is
        # diverted only when another admissible route is on offer --- above
        # the sigma/period threshold under ``wrap='single-image'`` centres is
        # the sole carrier of the declared measure and no budget may move it.
        if (route == "centres" and len(admissible) > 1
                and ws > _CENTRES_WORKING_SET_SOFT_BUDGET):
            cost = float("inf")
        prices[route] = cost
    best = min(admissible, key=lambda rt: prices[rt])
    return best, prices[best], prices, info


# ------------------------------------------------- whole-density pricing


def predict_nested_pairwise_kernel_size(dens_x, dens_y, *,
                                        skip_xx=False, skip_yy=False):
    """Joint tuple-pair kernel entries for the enumeration on this pair.

    The nested-aware twin of
    :func:`~mpt._tensor.dispatch._predict_pairwise_kernel_size`: each density
    contributes ``n_J = N * prod_a m_perm_a`` and ``n_K = N * prod_a
    m_comb_a``, with a nested attribute's counts coming from
    :func:`~mpt._tensor._nested_contraction.tuple_counts` instead of
    ``r!*C(K, r)``, and the three matrices are ``n_J^X * n_K^Y``,
    ``n_J^X * n_K^X`` and ``n_J^Y * n_K^Y``.
    """
    A = int(dens_x.n_attrs)
    perm_x = comb_x = float(dens_x.n)
    perm_y = comb_y = float(dens_y.n)
    for a in range(A):
        px, cx = _attr_tuple_counts(dens_x, a)
        py, cy = _attr_tuple_counts(dens_y, a)
        if min(px, cx, py, cy) <= 0.0:
            return float("inf")
        perm_x *= px
        comb_x *= cx
        perm_y *= py
        comb_y *= cy
    size = perm_x * comb_y
    if not skip_xx:
        size += perm_x * comb_x
    if not skip_yy:
        size += perm_y * comb_y
    return size


def _flat_companion_cost_ms(dens_x, dens_y, flat_a, ordered_a, *,
                            skip_xx=False, skip_yy=False,
                            truncation_sigmas=None):
    """Cost of the plan's non-nested attributes, priced by the flat model.

    Flat-symmetric and ``r = 1`` attributes go through the flat
    per-attribute orbit matrix, so they are priced by
    :func:`~mpt._tensor.dispatch._predict_orbit_cost_ms` on exactly those
    attributes. An ordered flat attribute goes through the materialised
    centres, so it is priced by this module's ``centres`` law on the same
    kernel-entry term the nested centres route uses.
    """
    from .dispatch import _predict_orbit_cost_ms, _orbit_sigma_over_p_threshold
    from ._nested_contraction import auto_ntau_default

    ts = truncation_sigmas
    total = 0.0
    if flat_a:
        r_vec = np.array([int(dens_x.r[a]) for a in flat_a], dtype=np.intp)
        k_vec = np.array([int(np.asarray(dens_x.p_attr[a]).shape[0])
                          for a in flat_a], dtype=np.intp)
        k_vec_y = np.array([int(np.asarray(dens_y.p_attr[a]).shape[0])
                            for a in flat_a], dtype=np.intp)
        rel_vec = np.array([bool(dens_x.is_rel[a]) for a in flat_a])
        nu_vec = np.ones(len(flat_a))
        sop = 0.0
        for i, a in enumerate(flat_a):
            if rel_vec[i] and bool(dens_x.is_per[a]):
                nu_vec[i] = auto_ntau_default(float(dens_x.period[a]),
                                              float(dens_x.sigma[a]), ts)
                sop = max(sop, float(dens_x.sigma[a])
                          / max(float(dens_x.period[a]), 1e-300))
            elif rel_vec[i]:
                nu_vec[i] = 2000.0
        total += _predict_orbit_cost_ms(
            r_vec, k_vec, len(flat_a), int(dens_x.n), int(dens_y.n),
            rel_vec, nu_vec,
            centres_ok=(sop <= _orbit_sigma_over_p_threshold(ts)),
            k_vec_y=k_vec_y, skip_xx=skip_xx, skip_yy=skip_yy)
    for a in ordered_a:
        terms, info = nested_attr_terms(dens_x, dens_y, a,
                                        skip_xx=skip_xx, skip_yy=skip_yy,
                                        truncation_sigmas=ts)
        n_matrices = 1 + (0 if skip_xx else 1) + (0 if skip_yy else 1)
        total += nested_route_cost_ms("centres", info["total_order"],
                                      terms["centres"], n_matrices)
    return total


def select_nested_method(dens_x, dens_y, *, admissible_by_attr,
                         enumeration_ok, skip_xx=False, skip_yy=False,
                         return_costs=False, truncation_sigmas=None):
    """Choose between the contraction plan and the joint-tuple enumeration.

    ``admissible_by_attr`` maps each nested attribute index to the routes its
    declared measure admits, in tie-break order; it is produced by the measure
    rule and never by this model. ``enumeration_ok`` says whether the
    enumeration carries the declared measure at all --- it computes the
    minimum-image reading of a relative-periodic attribute, so a
    ``wrap='full-image'`` attribute above the sigma/period threshold rules it
    out, exactly as the flat selector rules Bulger's method out there.

    Returns ``chosen`` (``'contract'`` or ``'bulger'``) or, with
    ``return_costs``, ``(chosen, plan_ms, enum_ms, detail)``. ``detail`` maps
    each nested attribute to ``(route, cost_ms, prices, info)`` and carries
    the flat companions' cost under the key ``'flat'``.

    The two sides share one pair of self-inner-product skip flags, as the
    flat selector's two routes do. The contraction and the enumeration
    memoise their self inner products under different cache keys, so a warm
    contraction memo does not literally spare the enumeration its self
    matrices; pricing each side against its own memo nonetheless decides the
    comparison on which side happened to run first rather than on what the
    two sides cost, and locks that first choice in. See
    :func:`~mpt._tensor.cosine._self_ip_memoised` for the full argument and
    for what the shared flag trades away (one call's under-pricing at each
    crossover).

    ``truncation_sigmas`` is the per-call width (``None`` takes the
    default); it sizes the quadrature grids the routes are priced on.
    """
    A = int(dens_x.n_attrs)
    nested_a = set(admissible_by_attr)
    is_sym_x = np.asarray(
        getattr(dens_x, "is_sym", np.ones(A, dtype=bool))).ravel()
    flat_a, ordered_a = [], []
    for a in range(A):
        if a in nested_a:
            continue
        if (not bool(is_sym_x[a])) and int(dens_x.r[a]) > 1:
            ordered_a.append(a)
        else:
            flat_a.append(a)

    detail = {}
    plan_ms = 0.0
    for a, admissible in admissible_by_attr.items():
        route, cost, prices, info = price_nested_attr(
            dens_x, dens_y, a, admissible, skip_xx=skip_xx, skip_yy=skip_yy,
            truncation_sigmas=truncation_sigmas)
        detail[a] = (route, cost, prices, info)
        plan_ms += cost
    companions = _flat_companion_cost_ms(
        dens_x, dens_y, flat_a, ordered_a, skip_xx=skip_xx, skip_yy=skip_yy,
        truncation_sigmas=truncation_sigmas)
    detail["flat"] = companions
    plan_ms += companions

    if enumeration_ok:
        size = predict_nested_pairwise_kernel_size(
            dens_x, dens_y, skip_xx=skip_xx, skip_yy=skip_yy)
        r_max = max((int(dens_x.r[a]) for a in range(A)), default=2)
        n_matrices = (1 + (0 if skip_xx else 1)
                      + (0 if skip_yy else 1))
        enum_ms = nested_route_cost_ms("bulger", r_max, size, n_matrices)
    else:
        enum_ms = float("inf")

    # Ties, and near-ties, favour the contraction: see _NESTED_ENUM_SAFETY.
    chosen = ("bulger" if enum_ms * _NESTED_ENUM_SAFETY < plan_ms
              else "contract")
    if return_costs:
        return chosen, float(plan_ms), float(enum_ms), detail
    return chosen
