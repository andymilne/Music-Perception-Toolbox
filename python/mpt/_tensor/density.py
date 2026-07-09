"""Density classes and multi-attribute input preprocessing helpers.

This module defines the three density data structures used throughout
the toolbox:

* :class:`ExpTensDensity` --- single-attribute expectation tensor density
* :class:`MaetDensity` --- multi-attribute expectation tensor density
* :class:`WindowedMaetDensity` --- a :class:`MaetDensity` paired with a
  post-tensor windowing spec

It also exposes the small set of multi-attribute input-coercion and
weight-normalisation helpers consumed by ``build_exp_tens`` and by the
preprocessing utilities (``difference_events``). The helpers live here
rather than in ``build.py`` because they are tied to the shape and
field conventions of :class:`MaetDensity` itself, not to any specific
consumer.

See :doc:`/ARCHITECTURE` §2 (Mathematical layering) and §3 (Code
layering) for the layered design.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from itertools import permutations
from math import factorial

import numpy as np
from scipy.special import comb as _comb

from .._utils import validate_weights


def _nchoosek_indices(n: int, r: int) -> np.ndarray:
    """Return all r-combinations of range(n) as an (r, C(n,r)) array."""
    from itertools import combinations

    combos = list(combinations(range(n), r))
    return np.array(combos, dtype=np.intp).T  # r x C(n,r)


# -------------------------------------------------------------------
#  Data class
# -------------------------------------------------------------------


def _weight_is_live(w: np.ndarray) -> np.ndarray:
    """Boolean mask of weights that contribute to a density.

    A weight contributes iff it is finite and of nonzero magnitude.
    NaN (a structurally absent slot) and ``0`` (present but
    zero-weighted, e.g. hard-zeroed outside a window's truncation
    support) both fail the test. This is the single definition of a
    "live" weight used by the event-level prune below.
    """
    w = np.asarray(w)
    return np.isfinite(w) & (np.abs(w) > 0.0)


class ExpTensDensity:
    """Precomputed single-attribute expectation tensor density.

    Stores the source multiset (``p``, ``w``) plus tensor parameters
    (``sigma``, ``r``, ``is_rel``, ``is_per``, ``period``, ``dim``)
    eagerly, and constructs the per-tuple permutation/combination
    arrays (``centres``, ``u_perm``, ``w_perm``, ``w_j``, ``v_comb``,
    ``wv_comb``, ``n_j``, ``n_j_perm``, ``n_k``) lazily on first
    access.

    Lazy materialisation matters at high-r/high-K where
    ``n_j = K!/(K-r)!`` makes the per-tuple arrays prohibitively
    expensive (e.g., K=256, r=4 → ~9·10⁸ tuples). Consumers that need
    only ``p``, ``w``, and the scalar parameters — for example,
    ``eval_exp_tens(method='mobius')``, ``cos_sim_exp_tens(method='mobius')``
    via the Möbius method's IP path, or ``entropy_exp_tens(method='renyi2')``
    — read just the eagerly-stored inputs and never trigger the build.
    Consumers that do need them (the centres path, Bulger's
    method, or any direct field access) trigger the build on
    first read; subsequent reads return the cached result. The
    materialisation is one-shot — once built, the arrays persist on
    the object and are not rebuilt.

    Use :attr:`materialised` to check whether the per-tuple arrays
    have been built without triggering a build.
    """

    # Anisotropic kernel covariance metadata (set post-construction by
    # build_exp_tens when a matrix-valued sigma is supplied; the
    # stored p values are then in whitened coordinates and sigma == 1).
    kernel_cov = None
    kernel_chol = None

    # Slots are not used because numpy arrays are stored as attributes
    # and the lazy cache adds attributes after construction; keeping
    # the class slot-less avoids surprising failures in extension code
    # that introspects via ``__dict__``.

    def __init__(
        self,
        *,
        p: np.ndarray,
        w: np.ndarray,
        sigma: float,
        r: int,
        is_rel: bool,
        is_per: bool,
        period: float,
        dim: int,
        is_sym: bool = True,
    ) -> None:
        self.p = p
        self.w = w
        self.sigma = sigma
        self.r = r
        self.is_rel = is_rel
        self.is_per = is_per
        self.period = period
        self.dim = dim
        # Per-attribute symmetrisation flag. Default True preserves the
        # legacy symmetric reading (the source multiset's r-subsets read
        # as unordered). is_sym = False reads each r-subset in listed
        # order (the de-reflected density).
        self.is_sym = bool(is_sym)
        # Lazy cache: built on first access to any per-tuple field
        # (``centres``, ``u_perm``, ``w_perm``, ``w_j``, ``v_comb``,
        # ``wv_comb``, ``n_j``, ``n_j_perm``, ``n_k``). All five
        # fields populate atomically — one build pass, no partial
        # state.
        self._centres = None
        self._u_perm = None
        self._w_perm = None
        self._v_comb = None
        self._wv_comb = None
        self._n_j = None
        self._n_k = None

    @property
    def materialised(self) -> bool:
        """``True`` if the per-tuple arrays have been built."""
        return self._centres is not None

    @property
    def live_events(self) -> np.ndarray:
        """Boolean ``(n,)`` mask of source elements that can contribute.

        An element is live iff its weight is finite and nonzero; a
        zero- or NaN-weight element adds nothing to any inner product
        or total mass. Cached on first access.
        """
        live = getattr(self, "_live_events", None)
        if live is None:
            live = _weight_is_live(self.w)
            self._live_events = live
        return live

    def pruned(self) -> "ExpTensDensity":
        """Equivalent density restricted to live elements.

        Dropping zero-/NaN-weight elements leaves every inner product
        and total mass unchanged (they contribute nothing) while
        shrinking the work. Returns ``self`` when nothing is dead, so
        the common (un-windowed) path pays only one mask scan.
        """
        live = self.live_events
        if live.all():
            return self
        if self.kernel_cov is not None:
            # r == K on matrix-sigma densities: dropping a value would
            # change the tuple dimension. The dead value already zeroes
            # the (single) tuple's weight, so pruning is a no-op.
            return self
        out = ExpTensDensity(
            p=self.p[live], w=self.w[live], sigma=self.sigma, r=self.r,
            is_rel=self.is_rel, is_per=self.is_per, period=self.period,
            dim=self.dim, is_sym=self.is_sym,
        )
        out.kernel_cov = self.kernel_cov
        out.kernel_chol = self.kernel_chol
        return out

    def _build_perm_arrays(self) -> None:
        """Build the per-tuple permutation / combination arrays.

        Populates the seven cached fields in one pass. No-op if
        already materialised. This is the only place that allocates
        the ``O(K!/(K-r)!)`` intermediate tensors.
        """
        if self._centres is not None:
            return

        p = self.p
        w = self.w
        r = self.r
        n = len(p)

        n_combs = int(_comb(n, r, exact=True))

        # All r-combinations (r x C(n,r))
        nck = _nchoosek_indices(n, r)

        # is_sym = True (default): symmetrise each combination over its
        # full S_r orbit (the perm side has r! copies). is_sym = False:
        # keep each combination in listed order, so the perm side equals
        # the comb side (the de-reflected, ordered density). r = 1 has
        # no order to symmetrise, so permutations(range(1)) gives the
        # single identity either way.
        if self.is_sym:
            all_perms = np.array(
                list(permutations(range(r))), dtype=np.intp,
            ).T  # r x r!
        else:
            all_perms = np.arange(r, dtype=np.intp).reshape(r, 1)  # identity
        n_perms = all_perms.shape[1]
        n_j = n_perms * n_combs
        n_k = n_combs

        # Build ordered r-tuples (perm side)
        j_idx = np.empty((r, n_j), dtype=np.intp)
        offset = 0
        for i in range(n_perms):
            j_idx[:, offset:offset + n_combs] = nck[all_perms[:, i], :]
            offset += n_combs

        u_perm = p[j_idx]                      # r x nJ
        w_perm = np.prod(w[j_idx], axis=0)     # (nJ,)

        # r-combinations (comb side, for cos_sim)
        v_comb = p[nck]                        # r x nK
        wv_comb = np.prod(w[nck], axis=0)      # (nK,)

        # Reduce to interval centres if relative
        if self.is_rel:
            centres = u_perm[1:, :] - u_perm[0, :]   # (r-1) x nJ
        else:
            centres = u_perm.copy()                  # r x nJ

        self._u_perm = u_perm
        self._w_perm = w_perm
        self._v_comb = v_comb
        self._wv_comb = wv_comb
        self._centres = centres
        self._n_j = n_j
        self._n_k = n_k

    # The seven lazy fields. Each property triggers the build on
    # first access; subsequent reads return the cached array.

    @property
    def centres(self) -> np.ndarray:
        """``(dim, n_j)`` array of per-tuple centres (lazy)."""
        if self._centres is None:
            self._build_perm_arrays()
        return self._centres

    @property
    def u_perm(self) -> np.ndarray:
        """``(r, n_j)`` array of per-tuple ordered pitch tuples (lazy)."""
        if self._u_perm is None:
            self._build_perm_arrays()
        return self._u_perm

    @property
    def w_perm(self) -> np.ndarray:
        """``(n_j,)`` per-tuple weight products, perm side (lazy)."""
        if self._w_perm is None:
            self._build_perm_arrays()
        return self._w_perm

    # Alias for w_perm — the v2.0 dataclass exposed both ``w_j`` and
    # ``w_perm`` pointing at the same array. Preserved for back-compat.
    @property
    def w_j(self) -> np.ndarray:
        """Alias for :attr:`w_perm` (lazy)."""
        return self.w_perm

    @property
    def v_comb(self) -> np.ndarray:
        """``(r, n_k)`` array of unordered r-combinations (lazy)."""
        if self._v_comb is None:
            self._build_perm_arrays()
        return self._v_comb

    @property
    def wv_comb(self) -> np.ndarray:
        """``(n_k,)`` per-combination weight products (lazy)."""
        if self._wv_comb is None:
            self._build_perm_arrays()
        return self._wv_comb

    @property
    def n_j(self) -> int:
        """Number of perm-side tuples ``K!/(K-r)!`` (lazy)."""
        if self._n_j is None:
            self._build_perm_arrays()
        return self._n_j

    # Alias for n_j — v2.0 dataclass exposed both names.
    @property
    def n_j_perm(self) -> int:
        """Alias for :attr:`n_j` (lazy)."""
        return self.n_j

    @property
    def n_k(self) -> int:
        """Number of comb-side tuples ``C(K, r)`` (lazy)."""
        if self._n_k is None:
            self._build_perm_arrays()
        return self._n_k

    def __repr__(self) -> str:
        built = "materialised" if self.materialised else "lazy"
        return (
            f"ExpTensDensity(K={len(self.p)}, r={self.r}, "
            f"sigma={self.sigma}, is_rel={self.is_rel}, "
            f"is_per={self.is_per}, dim={self.dim}, {built})"
        )


# -------------------------------------------------------------------
#  MaetDensity  (multi-attribute expectation tensor density)
# -------------------------------------------------------------------


class MaetDensity:
    """Precomputed multi-attribute expectation tensor density (MAET).

    Returned by :func:`build_exp_tens` when called in multi-attribute
    form (first argument a list/tuple of attribute matrices). The
    single-attribute return type is :class:`ExpTensDensity`.

    See the MAET specification (``multi_attribute_tensor_specification.md``)
    §2 and §6, and :func:`build_exp_tens` for argument semantics.

    Lazy materialisation
    --------------------
    The eager-stored fields (``p_attr``, ``w``, ``sigma``, ``r``,
    ``k``, ``is_rel``, ``is_per``, ``period``, ``n_attrs``, ``n``,
    ``dim``, ``dim_per_attr``, ``tag``) are populated by
    ``build_exp_tens``. Every attribute is self-contained, so the
    geometry fields ``sigma``, ``is_rel``, ``is_per``, and ``period``
    are per-attribute (length *A*). The per-tuple fields (``n_j``,
    ``n_k``, ``centres``, ``u_perm``, ``v_comb``, ``w_j``, ``wv_comb``,
    ``event_of_j``, ``event_of_k``) are constructed lazily on first
    access and cached. This keeps ``build_exp_tens`` cheap and avoids
    OOM at high cardinality when only the Möbius method is exercised
    (MA cosine ``method='mobius'``, MA Rényi-2 entropy). Use
    :attr:`materialised` to check the cache state without triggering a
    build.

    Conventions
    -----------
    Attribute indices in the fields below are 0-indexed (Python
    convention).

    Column *j* of ``centres[a]``, ``u_perm[a]``, ``w_j``, and
    ``event_of_j`` refer to the same global perm-side tuple. Similarly
    for column *k* across ``v_comb[a]``, ``wv_comb``, and
    ``event_of_k``.
    """

    # Anisotropic kernel covariance metadata (set post-construction by
    # build_exp_tens when any attribute's sigma is matrix-valued): a
    # length-A list with None for isotropic attributes and the
    # covariance / lower Cholesky factor for matrix-sigma attributes.
    # Matrix-sigma attributes store their p_attr values in whitened
    # coordinates with sigma == 1.
    kernel_cov = None
    kernel_chol = None

    def __init__(
        self,
        *,
        tag: str,
        n_attrs: int,
        n: int,
        r: np.ndarray,
        k: np.ndarray,
        p_attr: list,
        w: list,
        sigma: np.ndarray,
        is_rel: np.ndarray,
        is_per: np.ndarray,
        period: np.ndarray,
        dim: int,
        dim_per_attr: np.ndarray,
        is_sym: np.ndarray | None = None,
        nested: list | None = None,
        names: list | None = None,
        # The build closure: a no-arg callable that returns a dict
        # populating the lazy fields. Stored on the instance and
        # called on first access of any lazy field.
        _build_lazy,
    ) -> None:
        # Eager fields
        self.tag = tag
        self.n_attrs = n_attrs
        self.n = n
        self.r = r
        self.k = k
        self.p_attr = p_attr
        self.w = w
        self.sigma = sigma
        self.is_rel = is_rel
        self.is_per = is_per
        self.period = period
        self.dim = dim
        self.dim_per_attr = dim_per_attr
        # Per-attribute symmetrisation flag. Default all-True (legacy
        # symmetric reading) when a caller constructs the struct without
        # specifying it.
        self.is_sym = (np.ones(n_attrs, dtype=bool) if is_sym is None
                       else np.asarray(is_sym, dtype=bool).ravel())
        # Per-attribute nesting spec (representation B): None per attribute
        # for flat attributes, or a dict {tags, r, sym, rel, ...} (per-level
        # r/sym vectors and the resolved [rel] projection) for a nested one.
        # Per-slot tags are row-indexed, so event (column) pruning leaves
        # them untouched.
        self.nested = ([None] * n_attrs if nested is None else list(nested))
        # Optional per-attribute user-defined names (None where unnamed).
        # Attribute-indexed and prune-invariant (pruning drops events /
        # tuples, never attributes), so it is carried through unchanged.
        # Per-level names for a nested attribute live inside nested[a].
        self.names = ([None] * n_attrs if names is None else list(names))

        # Lazy slots
        self._build_lazy_fn = _build_lazy
        self._n_j = None
        self._n_k = None
        self._centres = None
        self._u_perm = None
        self._v_comb = None
        self._w_j = None
        self._wv_comb = None
        self._event_of_j = None
        self._event_of_k = None

    @property
    def materialised(self) -> bool:
        """``True`` if the per-tuple arrays have been built."""
        return self._n_j is not None

    @property
    def live_events(self) -> np.ndarray:
        """Boolean ``(n,)`` mask of events that can contribute.

        An event is live iff every attribute has at least one finite,
        nonzero weight slot in that event's column. An attribute whose
        column is all-zero or all-NaN kills the event (the
        per-attribute factors multiply); a partly-zero column does not.
        Cached on first access.
        """
        live = getattr(self, "_live_events", None)
        if live is None:
            live = np.ones(self.n, dtype=bool)
            for W in self.w:
                live &= _weight_is_live(W).any(axis=0)
            self._live_events = live
        return live

    def pruned(self) -> "MaetDensity":
        """Equivalent density restricted to live events.

        Dead events contribute nothing to any per-attribute inner
        product or total mass, so dropping them leaves results
        unchanged while shrinking the O(n) / O(n^2) work and the
        per-tuple expansion. Returns ``self`` when nothing is dead.
        The subset is rebuilt through the same lazy machinery
        ``build_exp_tens`` uses, so the per-tuple fields stay correct
        for any consumer that later materialises them.
        """
        live = self.live_events
        if live.all():
            return self
        from .build import _ma_build_perm_arrays

        p_attr = [P[:, live] for P in self.p_attr]
        w = [W[:, live] for W in self.w]
        n_k = int(live.sum())

        def _build_lazy():
            return _ma_build_perm_arrays(
                p_attr=p_attr, w_list=w, r_vec=self.r,
                is_rel_vec=self.is_rel, is_sym_vec=self.is_sym,
                N=n_k, A=self.n_attrs, nested=self.nested,
            )

        out = MaetDensity(
            tag=self.tag, n_attrs=self.n_attrs,
            n=n_k, r=self.r, k=self.k,
            p_attr=p_attr, w=w, sigma=self.sigma, is_rel=self.is_rel,
            is_per=self.is_per, period=self.period, dim=self.dim,
            dim_per_attr=self.dim_per_attr, is_sym=self.is_sym,
            nested=self.nested,
            names=self.names,
            _build_lazy=_build_lazy,
        )
        out.kernel_cov = self.kernel_cov
        out.kernel_chol = self.kernel_chol
        return out

    def _materialise(self) -> None:
        """Trigger the lazy build. No-op if already materialised."""
        if self._n_j is not None:
            return
        out = self._build_lazy_fn()
        self._n_j = out['n_j']
        self._n_k = out['n_k']
        self._centres = out['centres']
        self._u_perm = out['u_perm']
        self._v_comb = out['v_comb']
        self._w_j = out['w_j']
        self._wv_comb = out['wv_comb']
        self._event_of_j = out['event_of_j']
        self._event_of_k = out['event_of_k']
        # Drop the closure once consumed so Python can release the
        # input references it captured.
        self._build_lazy_fn = None

    @property
    def n_j(self) -> int:
        """Total number of perm-side tuples summed across events (lazy)."""
        if self._n_j is None:
            self._materialise()
        return self._n_j

    @property
    def n_k(self) -> int:
        """Total number of comb-side tuples summed across events (lazy)."""
        if self._n_k is None:
            self._materialise()
        return self._n_k

    @property
    def centres(self) -> list:
        """List of length A; each entry is ``(r_a - is_rel[g(a)]) x n_j`` (lazy)."""
        if self._centres is None:
            self._materialise()
        return self._centres

    @property
    def u_perm(self) -> list:
        """List of length A; each entry is ``r_a x n_j`` (lazy)."""
        if self._u_perm is None:
            self._materialise()
        return self._u_perm

    @property
    def v_comb(self) -> list:
        """List of length A; each entry is ``r_a x n_k`` (lazy)."""
        if self._v_comb is None:
            self._materialise()
        return self._v_comb

    @property
    def w_j(self) -> np.ndarray:
        """``(n_j,)`` perm-side per-tuple weight products (lazy)."""
        if self._w_j is None:
            self._materialise()
        return self._w_j

    @property
    def wv_comb(self) -> np.ndarray:
        """``(n_k,)`` comb-side per-tuple weight products (lazy)."""
        if self._wv_comb is None:
            self._materialise()
        return self._wv_comb

    @property
    def event_of_j(self) -> np.ndarray:
        """``(n_j,)`` event index per perm-side tuple (lazy)."""
        if self._event_of_j is None:
            self._materialise()
        return self._event_of_j

    @property
    def event_of_k(self) -> np.ndarray:
        """``(n_k,)`` event index per comb-side tuple (lazy)."""
        if self._event_of_k is None:
            self._materialise()
        return self._event_of_k

    def __repr__(self) -> str:
        built = "materialised" if self.materialised else "lazy"
        return (
            f"MaetDensity(A={self.n_attrs}, "
            f"N={self.n}, dim={self.dim}, {built})"
        )


# -------------------------------------------------------------------
#  WindowedMaetDensity  (MAET with a post-tensor window applied)
# -------------------------------------------------------------------


@dataclass
class WindowedMaetDensity:
    """A MaetDensity together with a post-tensor window specification.

    Returned by :func:`window_tensor`. Bundles an underlying
    :class:`MaetDensity` with per-group window parameters. No math is
    performed at construction time — the window is applied lazily by
    :func:`eval_exp_tens` (pointwise multiplication by the window
    function) and by :func:`cos_sim_exp_tens` (closed-form windowed
    inner product).

    See the MAET specification §4.3 for the full semantics.

    Fields
    ------
    tag : str
        Always ``"WindowedMaetDensity"``.
    dens : MaetDensity
        Underlying unwindowed density.
    size : (A,) float64
        Per-attribute window effective standard deviation in multiples
        of that attribute's ``sigma``. NaN or Inf means the attribute is
        not windowed.
    mix : (A,) float64
        Per-attribute shape parameter in [0, 1]: 0 = pure Gaussian,
        1 = pure rectangular, in between = rectangular-convolved-with-
        Gaussian. Ignored for attributes with ``size`` NaN/Inf.
    centre : list of ndarray
        Length-A list; entry *a* is a 1-D array of length
        ``dim_per_attr[a]`` giving the per-attribute centre point in
        that attribute's effective subspace. Ignored for attributes
        with ``size`` NaN/Inf.
    """

    tag: str
    dens: "MaetDensity"
    size: np.ndarray                     # (A,) float64
    mix: np.ndarray                      # (A,) float64
    centre: list                         # list of length A; each (dim_per_attr[a],)


# -------------------------------------------------------------------
#  MA-path helpers
# -------------------------------------------------------------------


def _coerce_attr_matrix(M) -> np.ndarray:
    """Coerce an attribute input to a 2-D K_a x N float64 matrix.

    A 1-D input is taken as K_a = 1 (a 1 x N row). 2-D inputs pass
    through. Higher dimensions are rejected.
    """
    M = np.asarray(M, dtype=np.float64)
    if M.ndim == 1:
        return M.reshape(1, -1)
    if M.ndim != 2:
        raise ValueError(
            f"Attribute matrix must be 1-D or 2-D, got ndim={M.ndim}."
        )
    return M


def _normalise_weights_ma(w, A: int, K_a: np.ndarray, N: int) -> list:
    """Normalise the top-level MA weight input to a length-*A* list of
    K_a x N matrices.

    Accepts ``None``, a scalar, or a list/tuple of per-attribute inputs.
    """
    if w is None:
        return [np.ones((int(K_a[a]), N), dtype=np.float64) for a in range(A)]

    if np.isscalar(w):
        if w == 0:
            warnings.warn("All weights are zero.")
        return [
            np.full((int(K_a[a]), N), float(w), dtype=np.float64)
            for a in range(A)
        ]

    # A 0-D ndarray is still a scalar in intent
    if isinstance(w, np.ndarray) and w.ndim == 0:
        val = float(w.item())
        if val == 0:
            warnings.warn("All weights are zero.")
        return [
            np.full((int(K_a[a]), N), val, dtype=np.float64)
            for a in range(A)
        ]

    if isinstance(w, (list, tuple)):
        if len(w) != A:
            raise ValueError(
                f"Weight list must have length {A} (n attributes), "
                f"got {len(w)}."
            )
        return [
            _broadcast_attr_weight(w[a], int(K_a[a]), N, a)
            for a in range(A)
        ]

    raise ValueError(
        "Top-level weight argument must be None, a scalar, or a list/tuple "
        "of per-attribute inputs. Got type "
        f"{type(w).__name__}."
    )


def _broadcast_attr_weight(wa, K: int, N: int, attr_idx: int) -> np.ndarray:
    """Broadcast a per-attribute weight input to a K x N matrix.

    Accepts:
      - ``None`` or empty           -> ones
      - scalar                      -> constant
      - 1-D of length N (K != N)    -> per-event row, broadcast across slots
      - 1-D of length K (K != N)    -> per-slot column, broadcast across events
      - 2-D (1, N)                  -> per-event row, broadcast across slots
      - 2-D (K, 1)                  -> per-slot column, broadcast across events
      - 2-D (K, N)                  -> full matrix

    When K == N, a 1-D input is ambiguous and rejected.
    """
    if wa is None:
        return np.ones((K, N), dtype=np.float64)

    wa = np.asarray(wa, dtype=np.float64)

    # Scalar (0-D or size 1)
    if wa.size == 1:
        return np.full((K, N), float(wa.item()))

    if wa.ndim == 1:
        if wa.size == N and wa.size != K:
            return np.broadcast_to(wa[np.newaxis, :], (K, N)).copy()
        if wa.size == K and wa.size != N:
            return np.broadcast_to(wa[:, np.newaxis], (K, N)).copy()
        if wa.size == N and wa.size == K:
            raise ValueError(
                f"Attribute {attr_idx} weight is a 1-D array of length "
                f"{wa.size}, but K_a == N == {K} makes the per-event vs "
                f"per-slot interpretation ambiguous. Supply as a 2-D array "
                f"(shape ({K}, 1) for per-slot or (1, {N}) for per-event)."
            )
        raise ValueError(
            f"Attribute {attr_idx} weight is a 1-D array of length "
            f"{wa.size}; expected {K} (per-slot), {N} (per-event), or a "
            f"scalar."
        )

    if wa.ndim == 2:
        sz = wa.shape
        if sz == (1, N):
            return np.broadcast_to(wa, (K, N)).copy()
        if sz == (K, 1):
            return np.broadcast_to(wa, (K, N)).copy()
        if sz == (K, N):
            return wa.copy()
        raise ValueError(
            f"Attribute {attr_idx} weight has shape {sz}; expected "
            f"(1, {N}), ({K}, 1), or ({K}, {N})."
        )

    raise ValueError(
        f"Attribute {attr_idx} weight must be at most 2-D; got "
        f"ndim={wa.ndim}."
    )


def _cartesian_indices(sizes) -> list:
    """Column-major Cartesian-product indices.

    Given axis sizes (s_0, s_1, ..., s_{A-1}), returns a list of *A*
    arrays, each of length prod(sizes). Array *a* gives the index along
    axis *a* for each linear position. First axis varies fastest
    (matches MATLAB ndgrid / column-major flatten).
    """
    A = len(sizes)
    out = []
    for a in range(A):
        rep_inner = int(np.prod(sizes[:a])) if a > 0 else 1
        rep_outer = int(np.prod(sizes[a + 1:])) if a < A - 1 else 1
        base = np.arange(sizes[a], dtype=np.intp)
        expanded = np.repeat(base, rep_inner) if rep_inner > 1 else base
        tiled = np.tile(expanded, rep_outer) if rep_outer > 1 else expanded
        out.append(tiled)
    return out

