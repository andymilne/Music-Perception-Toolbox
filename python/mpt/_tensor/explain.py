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
    """One candidate route and how it fared against the three tests."""

    name: str
    feasible: bool = True
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
        out.append(f"{'route':<12}{'feasible':<10}{'predicted':>12}   why")
        for r in self.routes:
            pred = ("--" if r.predicted_ms is None
                    else f"{r.predicted_ms:.3f} ms")
            mark = "*" if r.chosen else " "
            out.append(f"{mark}{r.name:<11}{str(r.feasible):<10}{pred:>12}   "
                       f"{r.reason}")
        out.append("")
        out.append(f"chosen        {self.chosen}  ({self.decided_by})")
        return "\n".join(out)


def _shape_of(dens):
    r = np.asarray(dens.r).ravel()
    k = np.asarray(dens.k).ravel()
    n = int(getattr(dens, "n", 1) or 1)
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
        _ma_eval_costs_ms,
        _orbit_sigma_over_p_threshold,
        _select_ma_eval,
    )

    ts = resolve_truncation_sigmas(truncation_sigmas)
    sop = _sigma_over_p(dens)
    limit, limit_set_by = _orbit_sigma_over_p_threshold(
        truncation_sigmas, return_binding=True)

    if other is not None:
        return _explain_cosine(dens, other, ts, sop, limit,
                               limit_set_by, method)

    n_q = 200 if n_q is None else int(n_q)
    chosen, reason = _select_ma_eval(dens, n_q, method=method)
    try:
        centres_ms, mobius_ms = _ma_eval_costs_ms(dens, n_q)
    except Exception:
        centres_ms = mobius_ms = None

    priced = "cost model" in reason
    routes = [
        Route("centres", True,
              reason if chosen == "centres" and not priced else "",
              centres_ms, chosen == "centres"),
        Route("mobius", True,
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


def _explain_cosine(dens_x, dens_y, ts, sop, limit, limit_set_by,
                    method):
    from .dispatch import _select_ma_inner_product_method

    r = np.asarray(dens_x.r).ravel()
    k = np.asarray(dens_x.k).ravel()
    k_y = np.asarray(dens_y.k).ravel()
    is_rel = np.asarray(dens_x.is_rel).ravel().astype(bool)
    is_per = np.asarray(dens_x.is_per).ravel().astype(bool)
    n_x = int(getattr(dens_x, "n", 1) or 1)
    n_y = int(getattr(dens_y, "n", 1) or 1)

    chosen, pw_ms, orbit_ms = _select_ma_inner_product_method(
        r_vec=r, k_vec=k, A=len(r), N_x=n_x, N_y=n_y,
        any_per=bool(is_per.any()),
        any_rel_nonper=bool((is_rel & ~is_per).any()),
        any_rel_per=bool((is_rel & is_per).any()),
        sigma_over_P_max=(sop or 0.0), user_method=method,
        rel_vec=is_rel, k_vec_y=k_y, return_costs=True,
    )
    priced = not (math.isnan(pw_ms) or math.isnan(orbit_ms))
    why = "priced" if priced else "decided structurally"
    routes = [
        Route("bulger", True, why,
              None if math.isnan(pw_ms) else pw_ms, chosen == "bulger"),
        Route("mobius", True, why,
              None if math.isnan(orbit_ms) else orbit_ms, chosen == "mobius"),
    ]
    measure = None
    if sop is not None:
        measure = ("transposition average (the definition)"
                   if chosen == "mobius"
                   else "wrapped-difference approximation")
    return Explanation(
        call="cos_sim_exp_tens",
        shape=_shape_of(dens_x) + f" against K={[int(v) for v in k_y]}",
        truncation_sigmas=ts, floor=truncation_floor(ts),
        routes=routes, sigma_over_p=sop, sigma_over_p_limit=limit,
        limit_set_by=limit_set_by, measure=measure,
        decided_by=("cost model" if priced else "structural rule"),
    )
