"""Report how a call would be routed, and why.

The toolbox chooses between two ways of computing the same quantity ---
materialising the joint tuple set, or the Möbius decomposition that
never does --- and the choice is governed by three tests applied in
order: is the route *feasible*, is it *accurate enough* for the
accuracy the caller asked for, and if more than one survives, which is
*fastest*.

That rule is easy to state and, until now, impossible to inspect. The
selectors return a short reason string, but not the quantities behind
it: the predicted time for each route, the accuracy floor in force, the
σ/P limit that follows from it, or where the call sits relative to any
of them. This module reports all of it for a given call, without
running the call.

It is a diagnostic, not a decision: :func:`explain` calls the same
selectors the toolbox uses, so what it reports is what would happen,
not a reconstruction of it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .._defaults import truncation_floor, resolve_truncation_sigmas


@dataclass
class Route:
    """One candidate route and how it fared.

    Feasibility is not reported per route: the selector excludes an
    infeasible route before pricing, and says so in the chosen route's
    reason rather than marking the excluded one.
    """

    name: str
    reason: str = ""
    predicted_ms: float | None = None
    chosen: bool = False


@dataclass
class Explanation:
    """How a call would be routed, and the quantities behind it."""

    call: str
    shape: str
    truncation_sigmas: float
    floor: float
    routes: list[Route] = field(default_factory=list)
    sigma_over_p: float | None = None
    sigma_over_p_limit: float | None = None
    limit_set_by: str | None = None
    measure: str | None = None
    decided_by: str = ""

    @property
    def chosen(self):
        for r in self.routes:
            if r.chosen:
                return r.name
        return None

    def __str__(self):
        w = 62
        out = [f"{self.call}: {self.shape}", "-" * w]
        out.append(f"accuracy      truncation_sigmas = {self.truncation_sigmas:g}"
                   f"  ->  floor {self.floor:.2e}")
        if self.sigma_over_p is not None:
            within = "within" if (self.sigma_over_p_limit is None
                                  or self.sigma_over_p
                                  <= self.sigma_over_p_limit) else "beyond"
            out.append(f"periodicity   sigma/P = {self.sigma_over_p:.4f}, "
                       f"{within} the limit of "
                       f"{self.sigma_over_p_limit:g}")
            out.append(f"              the limit is set by "
                       f"{self.limit_set_by}")
        if self.measure:
            out.append(f"measure       {self.measure}")
        out.append("")
        out.append(f"{'route':<12}{'predicted':>12}   why")
        for r in self.routes:
            pred = ("--" if r.predicted_ms is None
                    else f"{r.predicted_ms:.3f} ms")
            mark = "*" if r.chosen else " "
            out.append(f"{mark}{r.name:<11}{pred:>12}   {r.reason}")
        out.append("")
        out.append(f"chosen        {self.chosen}  ({self.decided_by})")
        return "\n".join(out)


def _shape_of(dens):
    r = np.asarray(dens.r).ravel()
    k = np.asarray(dens.k).ravel()
    n = getattr(dens, "n", None)
    n = 1 if n is None else int(n)
    mode = []
    for a in range(len(r)):
        rel = bool(np.asarray(dens.is_rel).ravel()[a])
        per = bool(np.asarray(dens.is_per).ravel()[a])
        mode.append(f"{'rel' if rel else 'abs'}{'per' if per else 'np'}")
    parts = [f"r={int(r[a])} K={int(k[a])} {mode[a]}" for a in range(len(r))]
    return f"{len(r)} attribute(s), {n} event(s); " + "; ".join(parts)


def _sigma_over_p(dens):
    """Largest sigma/period over the relative-periodic attributes."""
    sig = np.asarray(dens.sigma, dtype=float).ravel()
    per = np.asarray(dens.period, dtype=float).ravel()
    is_rel = np.asarray(dens.is_rel).ravel()
    is_per = np.asarray(dens.is_per).ravel()
    vals = [sig[a] / per[a] for a in range(len(sig))
            if a < len(per) and per[a] > 0 and is_per[a] and is_rel[a]]
    return max(vals) if vals else None


def explain_dispatch(dens, other=None, *, n_q=None, method="auto",
                     truncation_sigmas=None):
    """Report how a call on this density would be routed.

    With one density and ``n_q``, explains an evaluation; with two,
    explains a cosine similarity. Returns an :class:`Explanation`,
    which prints as a short report.
    """
    from .dispatch import (
        _eval_costs_ms,
        _orbit_sigma_over_p_threshold,
        _select_ma_eval,
    )

    ts = resolve_truncation_sigmas(truncation_sigmas)
    sop = _sigma_over_p(dens)
    limit, limit_set_by = _orbit_sigma_over_p_threshold(
        truncation_sigmas, return_binding=True)

    if other is not None:
        return _explain_cosine(dens, other, ts, sop, limit,
                               limit_set_by, method,
                               truncation_sigmas=truncation_sigmas)

    n_q = 200 if n_q is None else int(n_q)
    chosen, reason = _select_ma_eval(
        dens, n_q, method=method, truncation_sigmas=truncation_sigmas)
    try:
        centres_ms, mobius_ms = _eval_costs_ms(dens, n_q)
    except Exception:
        centres_ms = mobius_ms = None

    priced = "cost model" in reason
    routes = [
        Route("centres",
              reason if chosen == "centres" and not priced else "",
              centres_ms, chosen == "centres"),
        Route("mobius",
              reason if chosen == "mobius" and not priced else "",
              mobius_ms, chosen == "mobius"),
    ]
    for r in routes:
        if not r.reason:
            r.reason = "priced" if priced else "not selected"

    measure = None
    if sop is not None:
        measure = ("transposition average (the definition)"
                   if chosen == "mobius"
                   else "wrapped-difference approximation")
    return Explanation(
        call="eval_exp_tens", shape=f"{_shape_of(dens)}; n_q={n_q}",
        truncation_sigmas=ts, floor=truncation_floor(truncation_sigmas),
        routes=routes, sigma_over_p=sop, sigma_over_p_limit=limit,
        limit_set_by=limit_set_by, measure=measure, decided_by=reason,
    )


def _nested_attrs(dens):
    """Indices of the density's nested attributes."""
    nested = getattr(dens, "nested", None)
    if nested is None:
        return []
    return [a for a, spec in enumerate(nested) if spec is not None]


def _explain_cosine_nested(dens_x, dens_y, ts, sop, limit, limit_set_by,
                           method, nested_a):
    """Route report for a cosine on a nested density.

    A nested attribute never reaches the flat orbit (Möbius) entry point, so
    the two routes the flat report prices are not the two on offer here. The
    candidates are the hierarchical contraction plan --- which itself picks a
    route per nested attribute --- and the joint-tuple enumeration.

    Both are priced, as the flat report prices Bulger's method against the
    Möbius method: each nested attribute's admissible routes are costed in
    milliseconds by :mod:`~mpt._tensor._nested_cost`, the chosen ones summed
    with the flat companions' cost to give the plan's price, and that compared
    with the enumeration's. The per-attribute prices are reported as the
    contraction route's reason, an ``inf`` there marking a centres route
    diverted by the working-set guard. Where the measure rule leaves a nested
    attribute one admissible route the price is still shown, but the decision
    was not a cost decision and the report says so.
    """
    from .cosine import (_nested_admissible_routes, _nested_attr_route,
                         _nested_enumeration_admissible)
    from ._nested_cost import _NESTED_ENUM_SAFETY, select_nested_method

    force_route = "centres" if method == "centres" else None
    per_attr = []
    chosen_routes = {}
    blocked = None
    for a in nested_a:
        try:
            rt = _nested_attr_route(dens_x, dens_y, a,
                                    force_route=force_route)
            chosen_routes[a] = rt
            per_attr.append(f"attr {a}: {rt}")
        except ValueError as exc:
            blocked = str(exc)
            per_attr.append(f"attr {a}: unavailable")

    plan_ms = enum_ms = None
    priced = False
    if blocked is None:
        enum_ok = _nested_enumeration_admissible(dens_x, dens_y)
        _pick, plan_ms, enum_ms, detail = select_nested_method(
            dens_x, dens_y,
            admissible_by_attr={a: [rt] for a, rt in chosen_routes.items()},
            enumeration_ok=enum_ok, return_costs=True)
        priced = True
        per_attr = []
        for a in nested_a:
            adm = _nested_admissible_routes(dens_x, dens_y, a)
            _r, _c, prices, _i = detail[a]
            if len(adm) > 1:
                from ._nested_cost import price_nested_attr
                _r2, _c2, prices, _i2 = price_nested_attr(
                    dens_x, dens_y, a, adm)
            quoted = ", ".join(f"{k} {v:.3f} ms" for k, v in prices.items())
            per_attr.append(f"attr {a}: {chosen_routes[a]} ({quoted})")
        if detail.get("flat"):
            per_attr.append(f"flat companions: {detail['flat']:.3f} ms")

    if method == "bulger":
        chosen_name = "bulger"
    elif method != "auto" or not priced:
        chosen_name = "contract"
    else:
        chosen_name = ("bulger"
                       if enum_ms * _NESTED_ENUM_SAFETY < plan_ms
                       else "contract")
    reason = "; ".join(per_attr) if blocked is None else blocked
    if method == "bulger":
        enum_why = "forced"
    elif priced and enum_ms == float("inf"):
        enum_why = ("inadmissible: it computes the minimum-image measure, "
                    "which is not the declared one above the sigma/P limit")
        enum_ms = None
    elif priced:
        enum_why = "priced"
    else:
        enum_why = "not selected"
    routes = [
        Route("contract", reason, plan_ms, chosen_name == "contract"),
        Route("bulger", enum_why, enum_ms, chosen_name == "bulger"),
    ]
    measure = None
    if sop is not None:
        measure = ("transposition average (the definition)"
                   if any("taugrid" in r for r in per_attr)
                   else "wrapped-difference approximation")
    if method != "auto":
        decided = "user method"
    elif blocked is not None:
        decided = "measure rule"
    else:
        decided = "measure rule, then the nested cost model"
    return Explanation(
        call="cos_sim_exp_tens",
        shape=_shape_of(dens_x) + " (nested)",
        truncation_sigmas=ts, floor=truncation_floor(ts),
        routes=routes, sigma_over_p=sop, sigma_over_p_limit=limit,
        limit_set_by=limit_set_by, measure=measure,
        decided_by=decided,
    )


def _explain_cosine(dens_x, dens_y, ts, sop, limit, limit_set_by,
                    method, truncation_sigmas=None):
    """Route report for a cosine, built from the call's own inputs.

    The densities are pruned and the empty-operand rule applied first,
    as :func:`~mpt._tensor.cosine._cos_sim_exp_tens_ma` does; the flat
    selector then receives exactly the inputs the call gives it ---
    the wrap vector, the per-attribute grid node counts, and the memo
    flags read from the densities' caches (:func:`_flat_selector_inputs`)
    --- and the ordered-attribute override is applied after it. The
    report therefore names the route the call takes, including the
    rel-per wrap rule above the sigma/P threshold, which decides
    without pricing.
    """
    from .cosine import _flat_selector_inputs
    from .dispatch import _select_ma_inner_product_method

    nested_a = sorted(set(_nested_attrs(dens_x)) | set(_nested_attrs(dens_y)))
    if nested_a:
        return _explain_cosine_nested(dens_x, dens_y, ts, sop, limit,
                                      limit_set_by, method, nested_a)

    dens_x = dens_x.pruned()
    dens_y = dens_y.pruned()
    k_y = np.asarray(dens_y.k).ravel()
    shape = _shape_of(dens_x) + f" against K={[int(v) for v in k_y]}"
    if dens_x.n == 0 or dens_y.n == 0:
        # No events to overlap: the call returns 0.0 before any selector
        # runs, so there is no route to report.
        return Explanation(
            call="cos_sim_exp_tens", shape=shape,
            truncation_sigmas=ts, floor=truncation_floor(ts),
            routes=[Route("bulger", "not selected", None, False),
                    Route("mobius", "not selected", None, False)],
            sigma_over_p=sop, sigma_over_p_limit=limit,
            limit_set_by=limit_set_by, measure=None,
            decided_by="empty operand (the similarity is 0 without a route)",
        )

    # The raw per-call width goes in, as on the real call; the selector
    # resolves it where it needs a number.
    sel_kw, ordered_any, _nested = _flat_selector_inputs(
        dens_x, dens_y, normalize="cosine",
        truncation_sigmas=truncation_sigmas)
    chosen, pw_ms, orbit_ms = _select_ma_inner_product_method(
        user_method=method, return_costs=True, **sel_kw)
    priced = not (math.isnan(pw_ms) or math.isnan(orbit_ms))
    if ordered_any:
        # An ordered ([sym]=0) attribute has no orbit, so the call takes
        # Bulger's method whatever the selector said (and whatever the
        # user asked for).
        chosen = "bulger"
        decided = "ordered ([sym]=0) attribute (no orbit to collapse)"
    elif method != "auto":
        decided = "user method"
    elif priced:
        decided = "cost model"
    else:
        decided = "structural rule"
    why = "priced" if priced else "decided structurally"
    routes = [
        Route("bulger", why,
              None if math.isnan(pw_ms) else pw_ms, chosen == "bulger"),
        Route("mobius", why,
              None if math.isnan(orbit_ms) else orbit_ms, chosen == "mobius"),
    ]
    measure = None
    if sop is not None:
        measure = ("transposition average (the definition)"
                   if chosen == "mobius"
                   else "wrapped-difference approximation")
    return Explanation(
        call="cos_sim_exp_tens", shape=shape,
        truncation_sigmas=ts, floor=truncation_floor(ts),
        routes=routes, sigma_over_p=sop, sigma_over_p_limit=limit,
        limit_set_by=limit_set_by, measure=measure,
        decided_by=decided,
    )
