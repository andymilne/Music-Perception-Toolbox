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


def sweep_eligibility(dens_x, dens_y, offsets):
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
        # Relative-and-periodic attributes are refused whether or not
        # they are swept. Above sigma/P ~ 0.03 the pairwise and orbit
        # routes compute genuinely different measures there --- the
        # single-wrap and the transposition-average kernel --- and the
        # reduction's fixed factor commits to the single-wrap form. A
        # sweep must not decide that question silently on the caller's
        # behalf, so the choice is left with the per-offset path, where
        # ``method`` selects it explicitly.
        if bool(dens_x.is_rel[a]) and bool(dens_x.is_per[a]):
            return False, (
                f"attribute {a} is both relative and periodic; the "
                f"single-wrap and transposition-average kernels differ "
                f"there, and the reduction would fix that choice silently"
            )
        if not swept[a]:
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
                f"attribute {a} is nested and swept; its quadratic form is "
                f"a block-diagonal quotient, not a sum of squared components"
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


def _tuple_means_and_centred(u):
    """Split an ``(r, n)`` tuple array into means and centred residuals."""
    mean = u.mean(axis=0)
    return mean, u - mean[None, :]


def _splittable(dens, inner_r, a):
    """Whether attribute *a* admits the placement/shape split.

    The split needs the attribute's quadratic form to decompose into a
    within-tuple shape term and a placement term in the tuple means,
    which holds for a flat, non-periodic, isotropic attribute in either
    mode. Absolute contributes both terms; relative contributes the
    shape term alone, since the relative quadratic form *is* the shape
    term --- a uniform translation cancels in every within-tuple
    difference, which is the same statement as having no placement term.
    Handling both here keeps the generic log-kernel out of everything
    but the periodic and nested cases.
    """
    if bool(dens.is_per[a]):
        return False
    if inner_r is not None and int(inner_r[a]) > 0:
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
    swept_idx = [a for a in split_idx
                 if swept[a] and not bool(dens_x.is_rel[a])]
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
        means_u[a], cu[a] = _tuple_means_and_centred(u_perm[a])
        means_v[a], cv[a] = _tuple_means_and_centred(v_comb[a])

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
            U, V = cu[a], cv[a][:, c0:c1]
            spread = (np.einsum("ij,ij->j", U, U)[:, None]
                      + np.einsum("ij,ij->j", V, V)[None, :]
                      - 2.0 * (U.T @ V))
            np.maximum(spread, 0.0, out=spread)
            log_fixed -= spread * inv
            if bool(dens_x.is_rel[a]):
                # No placement term: the relative form is the shape term.
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
    # component-equivalents.
    _OFFSET_OVERHEAD_IN_COMPONENTS = 4096
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



# -------------------------------------------------------------------
#  Public entry point
# -------------------------------------------------------------------


def sweep_cos_sim_exp_tens(
    dens_x,
    dens_y,
    offsets,
    *,
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

    ok, reason = sweep_eligibility(dens_x, dens_y, off)
    if not ok:
        raise ValueError(
            f"This sweep cannot be reduced to a mixture in the offset: "
            f"{reason}. Translate the query with translate_attributes "
            f"and compare offset by offset instead."
        )

    dx = dens_x.pruned()
    dy = dens_y.pruned()
    ok, reason = sweep_eligibility(dx, dy, off)
    if not ok:
        raise ValueError(
            f"This sweep cannot be reduced to a mixture in the offset: "
            f"{reason}."
        )

    ts = resolve_truncation_sigmas(truncation_sigmas)
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
