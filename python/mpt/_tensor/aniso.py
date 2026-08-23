"""Anisotropic (matrix-valued) kernel covariance support.

A per-attribute kernel covariance ``Sigma`` (a ``dim x dim`` symmetric
positive-definite matrix, in squared units of the attribute's positions)
generalizes the isotropic ``sigma**2 * I`` kernel on ordered,
absolute-mode, non-periodic attributes whose tuple is the whole
multiset (``r == K``). The kernel is

    exp(-(x - c)^T Sigma^{-1} (x - c) / 2)

for the density, and the inner product of two densities sharing
``Sigma`` depends on centre differences through
``exp(-d^T Sigma^{-1} d / 4)``.

Implementation strategy: *whitening*. With the Cholesky factorization
``Sigma = R R^T`` (``R`` lower triangular), the coordinate change
``y = R^{-1} x`` carries the anisotropic kernel to the isotropic
unit-``sigma`` kernel, so every downstream computation (density
evaluation, inner products, entropies, truncation, precision options)
runs unchanged on whitened values with ``sigma = 1``. The only
corrections are the Gaussian normalization constant, which acquires a
factor ``det(Sigma)^{-1/2}``, and the continuous entropies, which
acquire the additive constant ``log det(Sigma) / 2`` (the change-of-
variables term of the linear map).

Constraints (validated at build time):

- ``is_sym`` must be False (the symmetric power requires a
  permutation-invariant kernel; a general ``Sigma`` is not);
- ``is_rel`` must be False (the exact common-shift quotient remains
  the province of the ``is_rel`` flag; graded shift tolerance is
  expressed *within* ``Sigma`` via an ``sd_shift**2 * ones`` ridge);
- ``is_per`` must be False (componentwise wrapping does not commute
  with the whitening change of coordinates);
- ``r == K`` (``Sigma`` is the covariance of the whole ordered tuple,
  so each event's tuple must be its full atom multiset);
- the attribute must not be nested, except degenerately: a two-level
  spec whose nesting encodes nothing beyond an ordered tuple of
  scalars (every group a singleton read whole, both levels absolute,
  ordered outer read of all groups) is order-isomorphic to the flat
  ordered tuple and is flattened automatically (see
  :func:`flatten_degenerate_nested_spec`). :func:`bind_events` applied
  to flat single-value events produces exactly this degenerate form.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular


# -------------------------------------------------------------------
#  Detection
# -------------------------------------------------------------------


def is_kernel_cov(sigma_entry) -> bool:
    """True if *sigma_entry* is a matrix-valued kernel covariance.

    A kernel covariance is any 2-D array-like; scalars, 0-D and 1-D
    inputs are the ordinary isotropic ``sigma``. Lists/tuples of
    lists/tuples are accepted as matrices.
    """
    if isinstance(sigma_entry, np.ndarray):
        return sigma_entry.ndim == 2
    if isinstance(sigma_entry, (list, tuple)):
        return (len(sigma_entry) > 0
                and isinstance(sigma_entry[0], (list, tuple, np.ndarray)))
    return False


def sigma_vec_has_kernel_cov(sigma_vec) -> bool:
    """True if any entry of a per-attribute sigma vector is a matrix."""
    if is_kernel_cov(sigma_vec):
        # A bare 2-D array as the whole sigma_vec is ambiguous; the MA
        # path requires a list/tuple with one entry per attribute, so a
        # 2-D ndarray here is a single-attribute matrix only when it is
        # the single-attribute path's scalar value -- callers handle that separately.
        return True
    if isinstance(sigma_vec, (list, tuple)):
        return any(is_kernel_cov(s) for s in sigma_vec)
    return False


# -------------------------------------------------------------------
#  Degenerate nesting
# -------------------------------------------------------------------


def flatten_degenerate_nested_spec(spec):
    """Return the flat spec equivalent to a degenerate nested *spec*,
    or ``None`` when the spec is not degenerate.

    A two-level nested spec is *degenerate* when its nesting encodes
    nothing beyond a tuple of scalars: every inner group is a singleton
    read whole (inner ``r`` = 1; the inner ``sym`` flag is vacuous on a
    singleton), the inner level is absolute (inner ``rel`` falsy), and
    the outer level reads all groups (outer ``r`` = the number of
    groups). The flat equivalent is ``{r: K, sym: sym[outer],
    rel: rel[outer]}`` over the same ``(K, N)`` value matrix (their
    densities are identical). :func:`bind_events` applied to flat
    single-value events produces exactly this form; the isotropic build
    recognises the same structure downstream (the singleton-group fast
    path in ``_build_exp_tens_ma``), but the matrix-covariance path
    must flatten *before* spec normalisation, since whitening and the
    ``r == K`` constraint need the true tuple size. Outer ``sym``/
    ``rel`` are carried through so :func:`check_aniso_constraints` can
    reject them with its canonical messages.
    """
    if not isinstance(spec, dict) or "tags" not in spec:
        return None
    tags = np.asarray(spec["tags"])
    tags = tags.ravel() if tags.ndim == 1 else tags[:, 0]
    K = tags.size
    r_lv = list(np.asarray(spec.get("r", []), dtype=object).ravel())
    sym_lv = list(np.asarray(spec.get("sym", []), dtype=object).ravel())
    rel_lv = list(np.asarray(spec.get("rel", []), dtype=object).ravel())
    if len(r_lv) != 2 or len(sym_lv) != 2 or len(rel_lv) != 2:
        return None
    if np.unique(tags).size != K:           # every group a singleton
        return None
    if int(r_lv[0]) != 1 or int(r_lv[1]) != K:
        return None                         # each read whole, all read
    if bool(rel_lv[0]):
        return None                         # inner level absolute
    flat = {"r": int(K), "sym": bool(sym_lv[1]), "rel": bool(rel_lv[1])}
    if spec.get("name") is not None:
        flat["name"] = spec["name"]
    return flat


def resolve_specs_for_kernel_cov(specs, sigma_vec, name: str = "sigma"):
    """Flatten degenerate nested specs on matrix-sigma attributes.

    Returns a new specs list in which every attribute carrying a
    matrix-valued kernel covariance has a flat spec: degenerate nested
    specs (see :func:`flatten_degenerate_nested_spec`) are replaced by
    their flat equivalents; a non-degenerate nested spec on a
    matrix-sigma attribute raises. Attributes with isotropic sigma are
    left untouched, nested or not. Outer-level ``sym``/``rel`` flags on
    a flattened spec are rejected downstream by
    :func:`check_aniso_constraints` exactly as on a flat attribute.
    """
    if not isinstance(specs, (list, tuple)):
        return specs
    if not isinstance(sigma_vec, (list, tuple)):
        return list(specs)
    out = list(specs)
    for a, s in enumerate(out):
        if a >= len(sigma_vec) or not is_kernel_cov(sigma_vec[a]):
            continue
        if not (isinstance(s, dict) and "tags" in s):
            continue
        flat = flatten_degenerate_nested_spec(s)
        if flat is None:
            raise ValueError(
                f"{name}[{a}]: a matrix-valued kernel covariance is "
                f"supported on a nested attribute only in the degenerate "
                f"case (two levels, every group a singleton read whole, "
                f"inner level absolute, outer level reading all groups), "
                f"which is order-isomorphic to a flat tuple and is "
                f"flattened automatically. This spec's nesting is not "
                f"degenerate."
            )
        out[a] = flat
    return out


# -------------------------------------------------------------------
#  Validation
# -------------------------------------------------------------------


def validate_kernel_cov(Sigma, dim: int | None = None, *,
                        name: str = "sigma"):
    """Validate a kernel covariance and return ``(Sigma, R)``.

    ``Sigma`` is returned as a float64 array; ``R`` is its lower
    Cholesky factor (``Sigma = R R^T``). Raises ``ValueError`` on a
    non-square, wrongly sized, asymmetric, or non-positive-definite
    input (a failed Cholesky factorization is the definiteness test,
    consistent with the toolbox rule that PSD validation always
    errors rather than repairing).
    """
    Sigma = np.asarray(Sigma, dtype=np.float64)
    if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError(
            f"{name}: a matrix-valued kernel covariance must be square; "
            f"got shape {Sigma.shape}."
        )
    if Sigma.shape[0] < 2:
        raise ValueError(
            f"{name}: a 1x1 kernel covariance is indistinguishable from "
            f"a scalar sigma in the MATLAB toolbox; pass the equivalent "
            f"standard deviation as the ordinary scalar sigma instead."
        )
    if dim is not None and Sigma.shape[0] != dim:
        raise ValueError(
            f"{name}: kernel covariance is {Sigma.shape[0]}x"
            f"{Sigma.shape[1]} but the attribute's tuple dimension is "
            f"{dim}; the covariance must be r x r with r the tuple size."
        )
    if not np.allclose(Sigma, Sigma.T, rtol=1e-12, atol=0.0):
        raise ValueError(
            f"{name}: kernel covariance must be symmetric."
        )
    if not np.all(np.isfinite(Sigma)):
        raise ValueError(
            f"{name}: kernel covariance must be finite. For exact "
            f"common-shift (transposition/tempo) invariance use "
            f"is_rel=True rather than an infinite shift variance."
        )
    try:
        R = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            f"{name}: kernel covariance must be positive definite "
            f"(Cholesky factorization failed). A singular covariance "
            f"means infinite tolerance along a null direction, which "
            f"a finite Gaussian kernel cannot represent; for the exact "
            f"common-shift quotient use is_rel=True."
        ) from exc
    return Sigma, R


def check_aniso_constraints(*, r: int, K: int, is_rel, is_per, is_sym,
                            nested_attr: bool = False,
                            name: str = "sigma"):
    """Enforce the mode constraints for a matrix-sigma attribute."""
    if nested_attr:
        raise ValueError(
            f"{name}: a matrix-valued kernel covariance is not "
            f"supported on nested attributes."
        )
    if bool(is_sym):
        raise ValueError(
            f"{name}: a matrix-valued kernel covariance requires an "
            f"ordered multiset (is_sym=False); the symmetric power "
            f"requires a permutation-invariant kernel."
        )
    if bool(is_rel):
        raise ValueError(
            f"{name}: a matrix-valued kernel covariance requires "
            f"is_rel=False; exact common-shift invariance remains the "
            f"province of is_rel=True, and graded shift tolerance is "
            f"expressed within the covariance (an sd_shift**2 * ones "
            f"ridge; see interval_kernel_cov)."
        )
    if bool(is_per):
        raise ValueError(
            f"{name}: a matrix-valued kernel covariance requires "
            f"is_per=False; periodic wrapping does not commute with "
            f"the anisotropic change of coordinates."
        )
    if int(r) != int(K):
        raise ValueError(
            f"{name}: a matrix-valued kernel covariance requires the "
            f"attribute's tuple to be its whole multiset (r == K); got "
            f"r={int(r)}, K={int(K)}. The covariance is that of the "
            f"complete ordered tuple."
        )


# -------------------------------------------------------------------
#  Whitening
# -------------------------------------------------------------------


def whiten_values(R: np.ndarray, V):
    """Whiten value columns: solve ``R y = v`` for each column of *V*.

    *V* is ``(dim, n)`` (or 1-D of length ``dim``, returned 1-D).
    NaNs are rejected: whitening mixes coordinates, so a NaN pad would
    contaminate the whole column.
    """
    V = np.asarray(V, dtype=np.float64)
    one_d = (V.ndim == 1)
    if one_d:
        V = V.reshape(-1, 1)
    if V.shape[0] != R.shape[0]:
        raise ValueError(
            f"Values have {V.shape[0]} rows but the kernel covariance "
            f"is {R.shape[0]}x{R.shape[0]}."
        )
    if not np.all(np.isfinite(V)):
        raise ValueError(
            "NaN or infinite values are not supported on an attribute "
            "with a matrix-valued kernel covariance (whitening mixes "
            "coordinates within each tuple)."
        )
    Y = solve_triangular(R, V, lower=True)
    return Y.ravel() if one_d else Y


def logdet_cov(R: np.ndarray) -> float:
    """``log det(Sigma)`` from the lower Cholesky factor ``R``."""
    return 2.0 * float(np.sum(np.log(np.diag(R))))


def kernel_cov_equal(a, b) -> bool:
    """Compatibility test for two per-density kernel covariances."""
    if a is None and b is None:
        return True
    if (a is None) != (b is None):
        return False
    return (np.asarray(a).shape == np.asarray(b).shape
            and np.allclose(a, b, rtol=1e-12, atol=0.0))


def density_has_kernel_cov(dens) -> bool:
    """True if *dens* carries any matrix-valued kernel covariance."""
    kc = getattr(dens, "kernel_cov", None)
    if kc is None:
        return False
    if isinstance(kc, (list, tuple)):
        return any(c is not None for c in kc)
    return True


def density_logdet_sum(dens) -> float:
    """Sum of ``log det(Sigma_a)`` over the density's covariances."""
    chol = getattr(dens, "kernel_chol", None)
    if chol is None:
        return 0.0
    if isinstance(chol, (list, tuple)):
        return float(sum(logdet_cov(R) for R in chol if R is not None))
    return logdet_cov(chol)


def density_kernel_covs_compatible(dens_x, dens_y) -> bool:
    """True if two densities carry equal kernel covariances throughout."""
    a = getattr(dens_x, "kernel_cov", None)
    b = getattr(dens_y, "kernel_cov", None)
    a_list = a if isinstance(a, (list, tuple)) else [a]
    b_list = b if isinstance(b, (list, tuple)) else [b]
    if (a is None) and (b is None):
        return True
    if isinstance(a, (list, tuple)) != isinstance(b, (list, tuple)):
        # SA vs MA pairing is rejected elsewhere on type grounds; a
        # bare-vs-list mismatch here means one side is scalar-form.
        if a is None or b is None:
            return not (density_has_kernel_cov(dens_x)
                        or density_has_kernel_cov(dens_y))
        return False
    if len(a_list) != len(b_list):
        return False
    return all(kernel_cov_equal(x, y) for x, y in zip(a_list, b_list))


def whiten_query(dens, x):
    """Whiten a query array for a density built with kernel covariance.

    SA: ``x`` is ``(dim, nQ)`` (1-D accepted when ``dim == 1``).
    MA: ``x`` is ``(D, nQ)`` with per-attribute row blocks in attribute
    order; only blocks whose attribute carries a covariance are
    transformed.
    """
    chol = getattr(dens, "kernel_chol", None)
    if chol is None:
        return x
    x = np.asarray(x, dtype=np.float64)
    if not isinstance(chol, (list, tuple)):
        # Single-attribute density.
        dim = int(dens.dim)
        one_d = (x.ndim == 1)
        xx = x.reshape(1, -1) if one_d else x.copy()
        if xx.shape[0] != dim:
            raise ValueError(
                f"x must have {dim} rows (each column is a {dim}-D "
                f"query point)."
            )
        xx = whiten_values(chol, xx)
        return xx.ravel() if one_d else xx
    # Multi-attribute density: slice rows by per-attribute dims.
    dims = np.asarray(dens.dim_per_attr, dtype=int)
    one_d = (x.ndim == 1)
    xx = x.reshape(-1, 1) if one_d else x.copy()
    if xx.shape[0] != int(np.sum(dims)):
        raise ValueError(
            f"x must have {int(np.sum(dims))} rows (each column is a "
            f"joint query point across the attributes)."
        )
    row = 0
    for a, da in enumerate(dims):
        if chol[a] is not None:
            xx[row:row + da, :] = whiten_values(chol[a], xx[row:row + da, :])
        row += int(da)
    return xx.ravel() if one_d else xx


def whiten_p_attr(p_attr, chol_list):
    """Whiten the matrix-sigma attributes of one MA position structure.

    ``chol_list`` is length-A with ``None`` for isotropic attributes;
    matrix-sigma attributes' value matrices (``r x N``) are transformed
    column-wise. Returns a new list.
    """
    out = list(p_attr)
    for a, R in enumerate(chol_list):
        if R is None:
            continue
        P = np.asarray(out[a], dtype=np.float64)
        if P.ndim == 1:
            P = P.reshape(1, -1)
        out[a] = whiten_values(R, P)
    return out
