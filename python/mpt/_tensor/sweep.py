"""Translation sweeps as a weighted Gaussian mixture in the offset.

When one density is compared against uniformly translated copies of a
second at many offsets, every pair of tuples contributes, *as a function
of the offset*, a single Gaussian whose centre and weight can be
computed once, before the sweep begins.

Starting from the multi-attribute inner product
(:func:`~mpt._tensor.cosine._ip_core_ma`),

.. math::

    \\langle T_X, T_Y \\rangle
      = \\sum_{j,k} w_j w_k \\prod_a
        \\exp\\!\\big(-Q_a(c_{a,j} - c_{a,k}) / (4\\sigma_a^2)\\big),

translate every value of attribute *a* on the *Y* side by a common
:math:`\\mu_a`. On an absolute attribute :math:`Q_a` is a sum of squared
components, and writing :math:`d = c_{a,j} - c_{a,k}` with mean
:math:`\\bar{d}`,

.. math::

    Q_a(d - \\mu_a \\mathbf{1})
      = r_a (\\mu_a - \\bar{d})^2 + \\sum_i (d_i - \\bar{d})^2 .

The second term does not depend on :math:`\\mu_a`. Each (tuple pair,
attribute) therefore contributes a fixed *shape* weight and a *placement*
Gaussian in the offset, centred at the mean difference :math:`\\bar{d}`
with effective width :math:`\\sigma_a / \\sqrt{r_a}`. Placement and shape
separate exactly.

A sweep over *M* offsets then costs one pass to build the mixture,
followed by *M* evaluations of that mixture, rather than *M* full inner
products over :math:`r_a`-dimensional tuples.

Two properties make the reduction exact rather than approximate:

* The shape term is the *relative*-mode quadratic form, so it is
  computed as a Gram matrix of within-tuple-centred coordinates --- no
  :math:`(r_a, n_J, n_K)` difference tensor is ever formed.
* The truncation rule is the one the generic path applies (a floor on
  the accumulated log-kernel). Because the placement term is
  non-positive, a component whose shape weight alone falls below the
  floor can never contribute at any offset, so it is dropped up front
  without changing the result.

Attributes that are *not* swept (offset identically zero) contribute an
offset-independent factor, computed by the generic log-kernel routine
itself. Those attributes may be of any mode --- relative, periodic, or
nested --- because their factor is never split.

See ARCHITECTURE.md §3 ("Code layering"); the reduction is the sweep
form of the closed-form inner product.
"""
from __future__ import annotations

import math

import numpy as np

from .density import MaetDensity

__all__ = ["sweep_cos_sim_exp_tens", "sweep_eligibility"]


# -------------------------------------------------------------------
#  Eligibility
# -------------------------------------------------------------------


class SweepNotEligible(Exception):
    """A sweep cannot be reduced to a mixture in the offset.

    Carries a human-readable reason. The explicit entry point turns it
    into a raised error; the automatic route catches it and falls back
    to the per-offset path.
    """


def sweep_eligibility(dens_x, dens_y, offsets, truncation_sigmas=None):
    """Whether ``offsets`` can be reduced to a mixture on these densities.

    Returns ``(True, None)`` when the reduction applies, and
    ``(False, reason)`` otherwise. ``offsets`` is an ``(A, M)`` array of
    per-attribute uniform translations of the *Y* operand.

    An attribute is *swept* when its offset is non-zero at any sweep
    index. A swept attribute must be absolute, non-periodic, flat (not
    nested), and isotropic (no kernel covariance): those are exactly the
    conditions under which its quadratic form is a plain sum of squared
    components and the placement/shape split holds. Attributes that are
    not swept are unconstrained.
    """
    if not isinstance(dens_x, MaetDensity) or not isinstance(dens_y, MaetDensity):
        return False, "both operands must be MaetDensity instances"
    A = int(dens_x.n_attrs)
    off = np.asarray(offsets, dtype=np.float64)
    if off.ndim != 2 or off.shape[0] != A:
        return False, (
            f"offsets must be an (A, M) array with A = {A}; got shape "
            f"{off.shape}"
        )
    if not np.all(np.isfinite(off)):
        return False, "offsets must be finite"

    from .dispatch import _inner_r_vec

    inner_r = _inner_r_vec(dens_x)
    swept = np.any(off != 0.0, axis=1)
    for a in range(A):
        if not swept[a]:
            # A relative-and-periodic attribute contributes an
            # offset-independent factor, and the reduction computes it
            # through the pairwise wrapped-difference form. That form is
            # one of two measures which coincide only below a sigma/P
            # limit, so the same limit the inner-product dispatcher uses
            # governs acceptance here: below it the two agree inside the
            # floor ``truncation_sigmas`` implies, and there is no choice
            # left to make silently. Above it, an explicit
            # ``wrap='single-image'`` names the measure this form
            # computes and is honoured; the default full-image reading
            # is refused, and the per-offset path decides it by
            # ``method``.
            if bool(dens_x.is_rel[a]) and bool(dens_x.is_per[a]):
                ok, why = _rel_per_measure_admissible(
                    dens_x, a, truncation_sigmas)
                if not ok:
                    return False, why
            continue
        if bool(dens_x.is_rel[a]):
            return False, (
                f"attribute {a} is relative and swept; a uniform "
                f"translation cancels in every within-tuple difference, "
                f"so there is nothing to sweep"
            )
        if bool(dens_x.is_per[a]):
            return False, (
                f"attribute {a} is periodic and swept; the wrapped kernel "
                f"does not admit the placement/shape split, and the "
                f"reduction is untested on the torus"
            )
        if inner_r is not None and int(inner_r[a]) > 0:
            return False, (
                f"attribute {a} is nested at an inner or intermediate "
                f"co-transposition unit and swept; each block removes its "
                f"own all-ones, so a uniform translation cancels within "
                f"every block and there is nothing to sweep (the attribute "
                f"is supported when it is not translated)"
            )
    for d, nm in ((dens_x, "context"), (dens_y, "query")):
        if getattr(d, "kernel_cov", None) is not None:
            kc = d.kernel_cov
            if any(kc[a] is not None and swept[a] for a in range(A)):
                return False, (
                    f"the {nm} density carries an anisotropic kernel "
                    f"covariance on a swept attribute"
                )
    if int(dens_x.n_j) == 0 or int(dens_y.n_k) == 0:
        return False, "one operand has no contributing tuples"
    return True, None


# -------------------------------------------------------------------
#  Mixture construction
# -------------------------------------------------------------------


def _rel_per_measure_admissible(dens, a, truncation_sigmas):
    """Whether an untranslated relative-periodic attribute is unambiguous.

    The wrapped-difference (single-wrap) form and the transposition-
    average (all-image) form are different measures that agree only at
    small ``sigma / P``. The limit is not a constant of this module: it
    is the one the inner-product dispatcher already calibrates, which
    tightens as ``truncation_sigmas`` asks for more accuracy and is
    capped by the positive-definiteness ceiling. Reusing it keeps the
    sweep held to the same standard as every other route.
    """
    from .dispatch import _orbit_sigma_over_p_threshold

    sigma = float(dens.sigma[a])
    period = float(dens.period[a]) if dens.period[a] is not None else 0.0
    if period <= 0.0:
        return True, None
    sop = sigma / period
    limit, binding = _orbit_sigma_over_p_threshold(
        truncation_sigmas, return_binding=True)
    if sop <= limit:
        return True, None
    wrap_a = "full-image"
    wrap_all = getattr(dens, "wrap", None)
    if wrap_all is not None and a < len(wrap_all):
        wrap_a = str(wrap_all[a])
    if wrap_a == "single-image":
        # The caller has named the measure this form computes.
        return True, None
    return False, (
        f"attribute {a} is relative and periodic at sigma/P = {sop:.3f}, "
        f"above the {binding} limit of {limit:.3f} for this "
        f"truncation_sigmas; the single-wrap and transposition-average "
        f"kernels differ there, so pass wrap='single-image' to name the "
        f"former or compare offset by offset, where method selects it"
    )



def _shape_blocks(u, block):
    """Within-block centred residuals of an ``(r, n)`` tuple array.

    ``block`` is the row count of one co-transposition unit: 0 (or the
    whole tuple) for a flat attribute, or the nested attribute's inner
    unit size. The quadratic form removes each block's own all-ones, so
    the shape term is the sum over blocks of the squared distance
    between block-centred residuals --- one Gram matrix per block.
    """
    r = int(u.shape[0])
    if block <= 0 or block >= r:
        return [u - u.mean(axis=0)[None, :]]
    n_blocks = r // block
    return [u[b * block:(b + 1) * block]
            - u[b * block:(b + 1) * block].mean(axis=0)[None, :]
            for b in range(n_blocks)]


def _splittable(dens, inner_r, a):
    """Whether attribute *a* admits the placement/shape split.

    The split needs the attribute's quadratic form to decompose into a
    within-tuple shape term and a placement term in the tuple means,
    which holds for a flat, non-periodic, isotropic attribute in either
    mode. Absolute contributes both terms; relative contributes the
    shape term alone, since the relative quadratic form *is* the shape
    term --- a uniform translation cancels in every within-tuple
    difference, which is the same statement as having no placement term. A nested
    attribute resolved to an inner or intermediate co-transposition unit
    is that same case read per block: its form is the sum over blocks of
    each block's relative form, so it contributes one shape term per
    block and no placement term, and cannot be swept either.
    Handling both here keeps the generic log-kernel out of everything
    but the periodic and nested cases.
    """
    if bool(dens.is_per[a]):
        return False
    kc = getattr(dens, "kernel_cov", None)
    return kc is None or kc[a] is None


def _build_mixture(dens_x, dens_y, swept, *, truncation_sigmas):
    """Build the offset mixture for one (context, query) density pair.

    Returns ``(centres, log_w, amp, threshold, swept_idx)`` where
    ``centres`` is ``(P, S)`` --- one row per surviving tuple pair, one
    column per swept attribute, ``swept_idx`` naming them --- ``log_w`` is the ``(P,)`` fixed log-kernel (shape terms
    plus every non-swept attribute's factor), ``amp`` is the ``(P,)``
    product of the two tuple weights, and ``threshold`` is the log-kernel
    floor the generic path applies.
    """
    from .cosine import _ma_log_kernel
    from .dispatch import _inner_r_vec

    A = int(dens_x.n_attrs)
    n_j = int(dens_x.n_j)
    n_k = int(dens_y.n_k)
    sigma = dens_x.sigma
    r_vec = dens_x.r
    inner_r = _inner_r_vec(dens_x)
    split_idx = [a for a in range(A) if _splittable(dens_x, inner_r, a)]
    # A splittable attribute that is never translated has a constant
    # placement term, so its whole contribution folds into the fixed
    # weight and it costs the mixture no axis at all --- which keeps the
    # cull as sharp as it would be if the attribute were handled by the
    # generic log-kernel, without calling it.
    blocks = {a: (int(inner_r[a]) if inner_r is not None else 0)
              for a in split_idx}
    # An attribute carries a placement term only where its quadratic
    # form keeps the tuple's own mean: absolute, and not block-quotiented.
    has_placement = {a: (not bool(dens_x.is_rel[a])) and blocks[a] == 0
                     for a in split_idx}
    swept_idx = [a for a in split_idx if swept[a] and has_placement[a]]
    still_idx = [a for a in split_idx if not swept[a]]
    fixed_idx = [a for a in range(A) if a not in split_idx]

    threshold = -0.5 * float(truncation_sigmas) ** 2
    n_terms = n_j * n_k
    if n_terms > 1:
        threshold -= math.log(float(n_terms))

    u_perm = dens_x.u_perm
    v_comb = dens_y.v_comb
    w_j = np.asarray(dens_x.w_j, dtype=np.float64)
    w_k = np.asarray(dens_y.wv_comb, dtype=np.float64)

    # Per swept attribute: tuple means (which set the placement centres)
    # and within-tuple-centred residuals (which set the shape weights).
    means_u, means_v, cu, cv = {}, {}, {}, {}
    for a in split_idx:
        # Everything below depends on the two operands only through
        # their differences, so a shift shared by both is exact --- and
        # it keeps the shape term well conditioned. That term is a Gram
        # form, ||u||^2 + ||v||^2 - 2u.v, whose cancellation costs
        # significant digits when the coordinates are far from the
        # origin: measured against the difference form, the log-kernel
        # departed by 4e-3 at magnitude 1e6 and 5e-1 at 1e7, and sat at
        # the floor once shifted. The shift is one of the attribute's
        # own values rather than their mean, because a data value is
        # exactly representable and subtracting it from a nearby value
        # is itself exact (Sterbenz), where a mean would inject a
        # rounding error at just those magnitudes.
        origin = float(u_perm[a].flat[0]) if u_perm[a].size else 0.0
        u_a = u_perm[a] - origin
        v_a = v_comb[a] - origin
        cu[a] = _shape_blocks(u_a, blocks[a])
        cv[a] = _shape_blocks(v_a, blocks[a])
        if has_placement[a]:
            means_u[a] = u_a.mean(axis=0)
            means_v[a] = v_a.mean(axis=0)

    if fixed_idx:
        inner_r_fixed = (None if inner_r is None
                         else np.asarray(inner_r)[fixed_idx])
        wrap_all = getattr(dens_x, "wrap", None)
        wrap_fixed = (None if wrap_all is None
                      else [wrap_all[a] for a in fixed_idx])

    # Chunk along the comb side so the transient (n_j, n_kc) blocks stay
    # within the kernel memory budget; the same heuristic the generic
    # inner product uses, sized for the per-attribute blocks this path
    # forms (one shape block plus one centre block per swept attribute).
    from .._utils import kernel_chunk_bytes_resolved

    per_col = max(1, (2 * max(len(swept_idx), 1) + 2)) * n_j * 8
    chunk = max(1, int(kernel_chunk_bytes_resolved() // max(per_col, 1)))

    swept_set = set(swept_idx)
    swept_pos = {a: i for i, a in enumerate(swept_idx)}
    cen_parts, logw_parts, amp_parts = [], [], []
    for c0 in range(0, n_k, chunk):
        c1 = min(c0 + chunk, n_k)
        log_fixed = np.zeros((n_j, c1 - c0), dtype=np.float64)
        cen_blk = np.empty((n_j, c1 - c0, len(swept_idx)), dtype=np.float64)

        for a in split_idx:
            inv = 1.0 / (4.0 * float(sigma[a]) ** 2)
            # Shape term: ||cu_j - cv_k||^2 as a Gram matrix, avoiding
            # the (r_a, n_j, n_k) difference tensor entirely.
            spread = None
            for U, V_full in zip(cu[a], cv[a]):
                V = V_full[:, c0:c1]
                blk = (np.einsum("ij,ij->j", U, U)[:, None]
                       + np.einsum("ij,ij->j", V, V)[None, :]
                       - 2.0 * (U.T @ V))
                spread = blk if spread is None else spread + blk
            np.maximum(spread, 0.0, out=spread)
            log_fixed -= spread * inv
            if not has_placement[a]:
                # No placement term: for a relative attribute the form
                # *is* the shape term, and for a block-quotiented nested
                # one each block removes its own all-ones.
                continue
            dbar = means_u[a][:, None] - means_v[a][None, c0:c1]
            if a in swept_set:
                cen_blk[:, :, swept_pos[a]] = dbar
            else:
                log_fixed -= (float(u_perm[a].shape[0]) * dbar * dbar) * inv

        if fixed_idx:
            log_fixed += _ma_log_kernel(
                [u_perm[a] for a in fixed_idx],
                [v_comb[a][:, c0:c1] for a in fixed_idx],
                n_j, c1 - c0, len(fixed_idx),
                [r_vec[a] for a in fixed_idx],
                [sigma[a] for a in fixed_idx],
                [dens_x.is_rel[a] for a in fixed_idx],
                [dens_x.is_per[a] for a in fixed_idx],
                [dens_x.period[a] for a in fixed_idx],
                inner_r=inner_r_fixed, wrap=wrap_fixed,
                truncation_sigmas=truncation_sigmas,
            )

        # A component whose fixed log-kernel already falls below the
        # floor can never rise above it: the placement term is
        # non-positive at every offset. Dropping it here is exact.
        keep = log_fixed >= threshold
        if not keep.any():
            continue
        amp = w_j[:, None] * w_k[None, c0:c1]
        cen_parts.append(cen_blk[keep])
        logw_parts.append(log_fixed[keep])
        amp_parts.append(amp[keep])

    if not cen_parts:
        return (np.zeros((0, len(swept_idx))), np.zeros(0), np.zeros(0),
                threshold, swept_idx)
    return (np.concatenate(cen_parts, axis=0),
            np.concatenate(logw_parts),
            np.concatenate(amp_parts),
            threshold, swept_idx)


# -------------------------------------------------------------------
#  Mixture evaluation
# -------------------------------------------------------------------


def _evaluate_mixture(centres, log_w, amp, scales, threshold, offsets):
    """Evaluate the mixture at every offset.

    ``centres`` is ``(P, S)``, ``offsets`` is ``(S, M)``, and ``scales``
    is the ``(S,)`` vector of :math:`r_a / (4\\sigma_a^2)` coefficients.
    Components outside the truncation radius are culled by a sorted
    index on the axis whose radius is tightest, then tested exactly, so
    the result matches a dense evaluation term for term.
    """
    P, S = centres.shape
    M = offsets.shape[1]
    out = np.zeros(M, dtype=np.float64)
    if P == 0 or M == 0:
        return out
    if S == 0:
        # Nothing is translated: the mixture has no placement term and
        # takes the same value at every sweep index.
        return np.full(M, float(np.dot(np.exp(log_w), amp)))

    # Every surviving component has log_w >= threshold, so its placement
    # budget is at most -threshold. That bounds the cull radius on each
    # axis uniformly.
    budget = float(np.max(log_w) - threshold)
    if not np.isfinite(budget) or budget < 0.0:
        return out
    radii = np.sqrt(budget / np.maximum(scales, np.finfo(float).tiny))

    # Index on the axis with the tightest radius relative to the spread
    # of centres: that is the axis whose window excludes most.
    spans = centres.max(axis=0) - centres.min(axis=0)
    ratio = np.where(radii > 0, spans / radii, np.inf)
    axis = int(np.argmax(ratio))
    order = np.argsort(centres[:, axis], kind="stable")
    cen_s = centres[order]
    logw_s = log_w[order]
    amp_s = amp[order]
    key = cen_s[:, axis]

    lo = np.searchsorted(key, offsets[axis] - radii[axis], side="left")
    hi = np.searchsorted(key, offsets[axis] + radii[axis], side="right")

    # Two regimes. The culled loop visits one offset at a time and pays a
    # fixed per-offset cost, which is the right trade when each offset
    # sees a small slice of a large mixture. When the mixture is small or
    # the cull excludes little, that per-offset cost dominates the
    # arithmetic it saves, and evaluating every component against a block
    # of offsets at once is faster. The threshold compares the two
    # directly rather than guessing: dense work is P per offset, culled
    # work is the mean slice plus the per-offset overhead expressed in
    # component-equivalents. The constant is calibrated against both
    # regimes: sparse mixtures (point-set-seeded sweeps, where each offset
    # sees a slice of order ten components out of thousands) and dense ones
    # (a single wide-kernel attribute, where nearly every component is
    # live). Culling's advantage where it wins reaches an order of
    # magnitude, dense's is under a factor of two, so the constant is set
    # to favour culling when the two are close.
    _OFFSET_OVERHEAD_IN_COMPONENTS = 512
    mean_slice = float(np.mean(np.maximum(hi - lo, 0))) if M else 0.0
    if P <= mean_slice + _OFFSET_OVERHEAD_IN_COMPONENTS:
        return _evaluate_dense(cen_s, logw_s, amp_s, scales, threshold,
                               offsets)

    for m in range(M):
        i0, i1 = int(lo[m]), int(hi[m])
        if i1 <= i0:
            continue
        d = cen_s[i0:i1] - offsets[:, m][None, :]
        L = logw_s[i0:i1] - (d * d) @ scales
        live = L >= threshold
        if not live.any():
            continue
        out[m] = float(np.dot(np.exp(L[live]), amp_s[i0:i1][live]))
    return out


def _evaluate_dense(centres, log_w, amp, scales, threshold, offsets):
    """Evaluate every component against a block of offsets at once.

    Computes exactly what the culled loop computes --- same threshold,
    same terms --- without the per-offset dispatch. Blocked over the
    offsets so the transient (block, P) array honours the kernel memory
    budget.
    """
    from .._utils import kernel_chunk_bytes_resolved

    P = centres.shape[0]
    M = offsets.shape[1]
    out = np.zeros(M, dtype=np.float64)
    block = max(1, int(kernel_chunk_bytes_resolved() // max(P * 8 * 4, 1)))
    for m0 in range(0, M, block):
        m1 = min(m0 + block, M)
        # (block, P) accumulated over the swept axes.
        L = np.repeat(log_w[None, :], m1 - m0, axis=0)
        for i, sc in enumerate(scales):
            d = centres[None, :, i] - offsets[i, m0:m1][:, None]
            L -= sc * d * d
        below = L < threshold
        np.exp(L, out=L)
        L[below] = 0.0
        out[m0:m1] = L @ amp
    return out



#: Work-unit ratio at which the orbit route overtakes the mixture.
#:
#: The two estimates count different things --- tuple pairs for the
#: mixture, orbit contractions over the value kernel for the orbit route
#: --- so the crossover is not at a ratio of one, and the constant
#: converts between them. Measured on a single-attribute sweep, minimum
#: of three runs, at (K, r) of (6, 3), (6, 4), (8, 3), (8, 4), (9, 4),
#: (10, 4), and (10, 5): the routes came out level at a ratio near 40,
#: with the mixture ahead by 4.6x at 270 and 22x at 443, and the orbit
#: route ahead by 4.5x at 9.4, 16x at 4.2, and 33x at 2.4. Only the
#: constant is machine-specific; the scaling either side of it is not,
#: and near the crossover the two routes are within a small factor, so
#: a misplaced choice there costs little.
_ORBIT_WORK_RATIO = 64.0

#: Tuple-pair count below which the mixture always wins.
#:
#: The ratio above models per-unit work, not the orbit route's fixed
#: per-call cost, so on small problems it can favour the orbit route
#: where the mixture is in fact several times faster. Measured crossover
#: sat between (6, 4) at 8.6e5 pairs, where the mixture led by 22x, and
#: (8, 3) at 3.0e6 pairs, where the orbit route led.
_ORBIT_MIN_PAIRS = 1.0e6



# -------------------------------------------------------------------
#  Orbit route
# -------------------------------------------------------------------


def orbit_sweep_supported(dens_x, dens_y, offsets=None,
                          truncation_sigmas=None):
    """Whether the orbit route can carry this sweep.

    The route evaluates the Möbius/orbit inner product at the shifted
    values, so it never forms the placement/shape split and is
    indifferent to periodicity, which the wrapped kernel absorbs. It
    computes both self inner products through the same per-attribute
    routine as the numerator, so the two carry one convention and a
    relative attribute is admissible here --- untranslated, since a
    uniform shift cancels in every within-tuple difference either way.

    On a relative-*and*-periodic attribute the route computes the
    transposition-average (all-image) kernel. Below the dispatcher's
    sigma/P limit that agrees with the single-wrap form inside the
    accuracy floor. Above it the two differ, and the attribute's
    ``wrap`` decides: ``'full-image'`` (the default) is the measure this
    route computes and is accepted, while ``'single-image'`` names the
    other one and is declined, exactly as the per-offset dispatcher
    resolves the same question.
    """
    from .dispatch import _inner_r_vec, _orbit_sigma_over_p_threshold

    A = int(dens_x.n_attrs)
    inner_r = _inner_r_vec(dens_x)
    swept = (np.zeros(A, dtype=bool) if offsets is None
             else np.any(np.asarray(offsets) != 0.0, axis=1))
    # The orbit decomposition sums over unordered value subsets with
    # multiplicity, and the per-attribute routine it calls takes no
    # symmetry flag: it computes the symmetrised inner product and
    # nothing else. On an ordered attribute that is a different
    # quantity, not an approximation of the right one --- measured
    # departures up to 0.22 --- so the route declines rather than
    # silently symmetrising.
    is_sym = getattr(dens_x, "is_sym", None)
    if is_sym is not None and not all(bool(v) for v in np.atleast_1d(is_sym)):
        return False
    for a in range(A):
        if inner_r is not None and int(inner_r[a]) > 0:
            return False
        if bool(dens_x.is_rel[a]) and swept[a]:
            # A uniform translation cancels in every within-tuple
            # difference; there is nothing to sweep.
            return False
        if bool(dens_x.is_rel[a]) and bool(dens_x.is_per[a]):
            period = (float(dens_x.period[a])
                      if dens_x.period[a] is not None else 0.0)
            if period > 0.0:
                sop = float(dens_x.sigma[a]) / period
                if sop > _orbit_sigma_over_p_threshold(truncation_sigmas):
                    wrap_all = getattr(dens_x, "wrap", None)
                    wrap_a = ("full-image" if wrap_all is None
                              or a >= len(wrap_all) else str(wrap_all[a]))
                    if wrap_a != "full-image":
                        return False
    for d in (dens_x, dens_y):
        if getattr(d, "kernel_cov", None) is not None:
            if any(c is not None for c in d.kernel_cov):
                return False
    return True


def _orbit_attr_matrix_sweep(Px, Wx, Py, Wy, sigma, r, mus, is_per, period,
                             wrap, truncation_sigmas):
    """Per-attribute inner-product matrices at every offset.

    Returns ``(N_x, N_y, M)``: entry ``(n_x, n_y, m)`` is the
    per-attribute inner product between event ``n_x`` of *X* and event
    ``n_y`` of *Y* translated by ``mus[m]``. This is the sweep form of
    :func:`~mpt._tensor._mobius_inner._ma_per_attr_inner_matrix`'s
    absolute branch: the (event pair, offset) index rides the orbit
    routine's batch axis, so the tuple enumeration never appears and
    the cost scales with the orbit count rather than with
    ``[C(K, r) r!]^2``.
    """
    from .._defaults import get_default
    from .._mobius import inner_product_orbit_pw_batched
    from .._utils import kernel_chunk_bytes_resolved
    from .._wrapped_kernel import wrapped_gaussian_1d
    from ._mobius_inner import _trunc_kernel_exp, _zero_pad_nan

    Px, Wx, Py, Wy = _zero_pad_nan(Px, Wx, Py, Wy)
    K_x, N_x = Px.shape
    K_y, N_y = Py.shape
    M = int(mus.size)
    prefactor = (sigma * np.sqrt(np.pi)) ** r
    use_orbit = int(r) >= 2      # r = 1 has no orbit decomposition
    out = np.empty((N_x, N_y, M), dtype=np.float64)

    full_image = bool(is_per) and str(wrap) == "full-image"
    if full_image:
        trunc_eff = float(get_default("truncation_sigmas")
                          if truncation_sigmas is None
                          else truncation_sigmas)

    # Chunk the offsets: the transient kernel block is
    # (K_x, N_x, K_y, N_y, chunk), with a few live copies.
    per_offset = 4 * K_x * N_x * K_y * N_y * 8
    chunk = max(1, int(kernel_chunk_bytes_resolved() // max(per_offset, 1)))

    w_a = np.broadcast_to(Wx.T[:, None, :], (N_x, N_y, K_x))
    w_b = np.broadcast_to(Wy.T[None, :, :], (N_x, N_y, K_y))

    for m0 in range(0, M, chunk):
        m1 = min(m0 + chunk, M)
        mc = m1 - m0
        # (K_x, N_x, K_y, N_y, mc)
        diffs = (Px[:, :, None, None, None]
                 - Py[None, None, :, :, None]
                 - mus[m0:m1][None, None, None, None, :])
        if full_image:
            K_tens = wrapped_gaussian_1d(
                diffs, sigma, period, trunc_eff, exponent_denominator=4,
            )
        else:
            if is_per:
                diffs = diffs - period * np.floor(diffs / period + 0.5)
            K_tens = _trunc_kernel_exp(diffs ** 2, sigma, truncation_sigmas)
        # -> (N_x, N_y, mc, K_x, K_y), then flatten the batch axis.
        K_pairs = np.transpose(K_tens, (1, 3, 4, 0, 2)).reshape(
            N_x * N_y * mc, K_x, K_y,
        )
        wa = np.broadcast_to(
            w_a[:, :, None, :], (N_x, N_y, mc, K_x),
        ).reshape(-1, K_x)
        wb = np.broadcast_to(
            w_b[:, :, None, :], (N_x, N_y, mc, K_y),
        ).reshape(-1, K_y)
        if use_orbit:
            flat = inner_product_orbit_pw_batched(
                K_pairs, wa, wb, r, prefactor=prefactor,
            )
        else:
            # r = 1: each event contributes a single kernel, so the
            # per-attribute matrix is the weighted kernel sum directly.
            flat = prefactor * np.einsum(
                "gx,gxy,gy->g", wa, K_pairs, wb, optimize=True)
        out[:, :, m0:m1] = flat.reshape(N_x, N_y, mc)
    return out


def _orbit_sweep(dens_x, dens_y, off, truncation_sigmas):
    """Numerator of the sweep via the orbit decomposition.

    The multi-attribute inner product is a sum over event pairs of a
    product over attributes, so each attribute contributes an
    ``(N_x, N_y, M)`` block and the blocks multiply. An attribute that
    is never translated contributes the same block at every offset, so
    it is computed once and broadcast.
    """
    from ._mobius_inner import _ma_per_attr_inner_matrix

    A = int(dens_x.n_attrs)
    M = off.shape[1]
    P = np.ones((int(dens_x.n), int(dens_y.n), M), dtype=np.float64)
    for a in range(A):
        sigma = float(dens_x.sigma[a])
        r_a = int(dens_x.r[a])
        is_per = bool(dens_x.is_per[a])
        period = float(dens_x.period[a]) if dens_x.period[a] is not None else 0.0
        wrap_a = (str(dens_x.wrap[a])
                  if getattr(dens_x, "wrap", None) is not None
                  else "full-image")
        Px, Wx = dens_x.p_attr[a], dens_x.w[a]
        Py, Wy = dens_y.p_attr[a], dens_y.w[a]
        if not np.any(off[a] != 0.0):
            block = _ma_per_attr_inner_matrix(
                Px, Wx, Py, Wy, sigma, r_a, bool(dens_x.is_rel[a]),
                is_per, period,
                truncation_sigmas=truncation_sigmas, wrap=wrap_a,
            )
            P *= block[:, :, None]
        else:
            P *= _orbit_attr_matrix_sweep(
                Px, Wx, Py, Wy, sigma, r_a, off[a], is_per, period,
                wrap_a, truncation_sigmas,
            )
    return P.sum(axis=(0, 1))



def _orbit_self_ip(dens, truncation_sigmas):
    """<T, T> through the same per-attribute routine as the numerator.

    The orbit path's own similarity triple chooses per attribute between
    the closed-form centres route and the grid contraction, and the two
    differ by a constant per-attribute prefactor that cancels only
    within one route's own triple. Taking the self terms from the
    routine the numerator uses keeps numerator and denominator on one
    convention, and is what lets a relative attribute ride this route at
    all.
    """
    from ._mobius_inner import _ma_per_attr_inner_matrix

    A = int(dens.n_attrs)
    P = np.ones((int(dens.n), int(dens.n)), dtype=np.float64)
    for a in range(A):
        period = (float(dens.period[a])
                  if dens.period[a] is not None else 0.0)
        wrap_a = (str(dens.wrap[a])
                  if getattr(dens, "wrap", None) is not None
                  else "full-image")
        P *= _ma_per_attr_inner_matrix(
            dens.p_attr[a], dens.w[a], dens.p_attr[a], dens.w[a],
            float(dens.sigma[a]), int(dens.r[a]),
            bool(dens.is_rel[a]), bool(dens.is_per[a]), period,
            truncation_sigmas=truncation_sigmas, wrap=wrap_a,
        )
    return float(P.sum())


def _finalise_orbit_sweep(dx, dy, off, ts, *, normalize, truncation_sigmas,
                          verbose):
    """Sweep numerator and denominators, all in one convention."""
    from .cosine import _finalise_normalisation

    ip_xy = _orbit_sweep(dx, dy, off, truncation_sigmas)
    ip_yy = _orbit_self_ip(dy, truncation_sigmas)
    ip_xx = (_orbit_self_ip(dx, truncation_sigmas)
             if normalize == "cosine" else None)
    return np.array(
        [_finalise_normalisation(float(v), ip_xx, ip_yy, normalize)
         for v in ip_xy],
        dtype=np.float64,
    )


def _choose_sweep_route(dx, dy, off, mixture_ok, orbit_ok):
    """Pick between the mixture and the orbit route.

    The two scale differently in the same problem. The mixture pays one
    pass over the tuple pairs --- ``n_J * n_K``, which grows as
    ``[C(K, r) r!]^2`` --- and must also *store* the survivors, so it is
    the memory-bound route at high tuple order. The orbit route pays
    per offset instead, but its unit of work is an orbit contraction
    over the ``K_x * K_y`` value kernel, with no tuple enumeration
    anywhere. The comparison below is between those two products.
    """
    if not orbit_ok:
        return "mixture"
    if not mixture_ok:
        return "orbit"
    from .._mobius import get_orbit_table

    n_pairs = float(dx.n_j) * float(dy.n_k)
    M = float(off.shape[1])
    A = int(dx.n_attrs)
    orbit_work = 0.0
    for a in range(A):
        if not np.any(off[a] != 0.0):
            continue
        r_a = int(dx.r[a])
        # r = 1 has no orbit decomposition to reduce: the per-attribute
        # matrix is a plain kernel sum, and the mixture handles that
        # shape at least as cheaply.
        if r_a < 2:
            return "mixture"
        n_orb = float(len(get_orbit_table(r_a)))
        k_x = float(dx.p_attr[a].shape[0])
        k_y = float(dy.p_attr[a].shape[0])
        orbit_work += n_orb * k_x * k_y * r_a
    if orbit_work <= 0.0:
        return "mixture"
    orbit_total = M * float(dx.n) * float(dy.n) * orbit_work

    # Memory decides before speed does. The mixture must hold its
    # surviving components --- one centre per swept attribute, plus a
    # weight and an amplitude, per tuple pair --- and at high tuple
    # order that array is what fails first, whatever the timings say.
    from .._utils import kernel_chunk_bytes_resolved

    n_swept = sum(1 for a in range(A) if np.any(off[a] != 0.0))
    mixture_bytes = n_pairs * (n_swept + 2) * 8.0
    if mixture_bytes > float(kernel_chunk_bytes_resolved()):
        return "orbit"

    # Below this the mixture's whole pass is cheap in absolute terms and
    # the orbit route's fixed per-call overhead --- orbit-table lookup
    # and |omega_r| contractions, which the ratio above does not model
    # --- dominates whatever the ratio says.
    if n_pairs < _ORBIT_MIN_PAIRS:
        return "mixture"
    return ("orbit" if orbit_total < _ORBIT_WORK_RATIO * n_pairs
            else "mixture")



# -------------------------------------------------------------------
#  Public entry point
# -------------------------------------------------------------------


def sweep_cos_sim_exp_tens(
    dens_x,
    dens_y,
    offsets,
    *,
    method: str = "auto",
    normalize: str = "cosine",
    truncation_sigmas: float | None = None,
    kernel_precision: str | None = None,
    verbose: bool = True,
):
    """Similarity of one density against uniform translates of another.

    Computes, for each sweep index *m*, the similarity of ``dens_x``
    against ``dens_y`` with every value of attribute *a* shifted by
    ``offsets[a, m]``. The whole sweep costs one pass over the tuple
    pairs plus *M* evaluations of a mixture in the offset, rather than
    *M* inner products; see the module docstring for the identity.

    The offsets are supplied directly, so nothing is inferred: the
    result is the sweep the caller asked for, and agrees with the
    per-offset path to the truncation floor. Building the translated
    value matrices with :func:`~mpt.translate_attributes` and passing
    them to :func:`~mpt.cos_sim_exp_tens` reaches the same computation,
    since that output carries its offsets with it.

    Parameters
    ----------
    dens_x : MaetDensity
        The fixed (context) density.
    dens_y : MaetDensity
        The query density, translated by each offset in turn.
    offsets : array-like
        ``(A, M)`` array of per-attribute translations, or a 1-D
        length-``M`` vector when ``A == 1``. One column per sweep index.
        A row of zeros leaves that attribute untranslated.
    method : {'auto', 'mixture', 'orbit'}, default 'auto'
        Which decomposition carries the sweep. ``'mixture'`` is the
        placement/shape split described above: one pass over the tuple
        pairs, then a mixture evaluation per offset. ``'orbit'``
        evaluates the Möbius/orbit inner product at the shifted values,
        with the (event pair, offset) index riding the orbit routine's
        batch axis; its cost scales with the orbit count rather than
        with the tuple-pair count :math:`[C(K, r) r!]^2`, which the
        mixture must both enumerate and store. ``'auto'`` compares the
        two costs and picks. The orbit route also covers a swept
        *periodic* attribute, which the mixture refuses: it never forms
        the split, so the wrapped kernel absorbs the periodicity.
    normalize : {'cosine', 'oneSidedDenom'}, default 'cosine'
        As in :func:`~mpt.cos_sim_exp_tens`. Both self inner products
        are translation-invariant here, so each is computed once for the
        whole sweep.
    truncation_sigmas, kernel_precision, verbose
        As in :func:`~mpt.cos_sim_exp_tens`.

    Returns
    -------
    ndarray
        Length-``M`` array of similarities, indexed as the columns of
        ``offsets``.

    Raises
    ------
    ValueError
        When the sweep cannot be reduced --- a swept attribute that is
        relative, periodic, nested, or anisotropic. The message names
        the attribute and the reason. Sweeping a *relative* attribute is
        a no-op by construction; sweeping a *periodic* one is untested
        on the torus and is refused rather than approximated.

    See Also
    --------
    cos_sim_exp_tens, translate_attributes, windowed_tensor_similarity
    """
    from .._defaults import resolve_truncation_sigmas
    from .cosine import _finalise_normalisation

    if normalize not in ("cosine", "oneSidedDenom"):
        raise ValueError(
            f"normalize must be 'cosine' or 'oneSidedDenom'; got "
            f"{normalize!r}."
        )
    off = np.asarray(offsets, dtype=np.float64)
    if off.ndim == 1:
        off = off.reshape(1, -1)
    if off.ndim != 2:
        raise ValueError(
            f"offsets must be 1-D or 2-D; got ndim = {off.ndim}."
        )

    if method not in ("auto", "mixture", "orbit"):
        raise ValueError(
            f"method must be 'auto', 'mixture', or 'orbit'; got {method!r}."
        )
    # Shape and finiteness are contract violations, not routing
    # questions, so they are checked before any route is considered.
    A_in = int(dens_x.n_attrs)
    if off.shape[0] != A_in:
        raise ValueError(
            f"offsets must be an (A, M) array with A = {A_in}; got shape "
            f"{off.shape}."
        )
    if not np.all(np.isfinite(off)):
        raise ValueError("offsets must be finite.")

    dx = dens_x.pruned()
    dy = dens_y.pruned()
    mixture_ok, mixture_reason = sweep_eligibility(
        dx, dy, off, truncation_sigmas)
    orbit_ok = orbit_sweep_supported(
        dx, dy, off, truncation_sigmas)

    if method == "auto":
        chosen = _choose_sweep_route(dx, dy, off, mixture_ok, orbit_ok)
    else:
        chosen = method
    if chosen == "mixture" and not mixture_ok:
        raise ValueError(
            f"This sweep cannot be reduced to a mixture in the offset: "
            f"{mixture_reason}. Translate the query with "
            f"translate_attributes and compare offset by offset instead."
        )
    if chosen == "orbit" and not orbit_ok:
        raise ValueError(
            "The orbit route does not support a relative, nested, or "
            "anisotropic attribute in a sweep."
        )

    ts = resolve_truncation_sigmas(truncation_sigmas)

    if chosen == "orbit":
        return _finalise_orbit_sweep(
            dx, dy, off, ts, normalize=normalize,
            truncation_sigmas=truncation_sigmas, verbose=verbose,
        )
    A = int(dx.n_attrs)
    swept = np.any(off != 0.0, axis=1)
    centres, log_w, amp, threshold, swept_idx = _build_mixture(
        dx, dy, swept, truncation_sigmas=ts,
    )
    scales = np.array(
        [float(dx.u_perm[a].shape[0]) / (4.0 * float(dx.sigma[a]) ** 2)
         for a in swept_idx],
        dtype=np.float64,
    )
    if not swept_idx:
        # Nothing is translated: the mixture is a single constant.
        centres = centres.reshape(centres.shape[0], 0)
        scales = np.zeros(0)
    ip_xy = _evaluate_mixture(
        centres, log_w, amp, scales, threshold,
        off[swept_idx, :] if swept_idx
        else np.zeros((0, off.shape[1])),
    )

    # Both self inner products are invariant under a uniform translation
    # of their own values, so the whole sweep shares one denominator.
    need_xx = normalize == "cosine"
    ip_xx = None
    if need_xx:
        ip_xx = _self_ip(dx, truncation_sigmas=ts,
                         kernel_precision=kernel_precision, verbose=verbose)
    ip_yy = _self_ip(dy, truncation_sigmas=ts,
                     kernel_precision=kernel_precision, verbose=verbose)

    return np.array(
        [_finalise_normalisation(float(v), ip_xx, ip_yy, normalize)
         for v in ip_xy],
        dtype=np.float64,
    )


def _self_ip(dens, *, truncation_sigmas, kernel_precision, verbose):
    """<T, T> for one density, through this module's own mixture.

    The self inner product is invariant under a uniform translation of
    the density's own values, so the whole sweep shares one denominator.
    It is computed here by the routine that produces the numerator,
    evaluated at zero offset, so numerator and denominator carry the
    same convention and the same truncation treatment --- and so the two
    languages compute it identically. Memoised on the density under its
    own route key, which never crosses the pairwise or orbit memos.
    """
    key = ("sweep", float(truncation_sigmas), str(kernel_precision))
    if key in dens._self_ip_cache:
        return dens._self_ip_cache[key]
    A = int(dens.n_attrs)
    zero = np.zeros((A, 1))
    centres, log_w, amp, threshold, swept_idx = _build_mixture(
        dens, dens, np.zeros(A, dtype=bool), truncation_sigmas=truncation_sigmas,
    )
    val = float(_evaluate_mixture(
        centres, log_w, amp, np.zeros(0), threshold, zero[swept_idx, :],
    )[0])
    dens._self_ip_cache[key] = val
    return val
