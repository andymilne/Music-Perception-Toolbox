"""Density classes and multi-attribute input preprocessing helpers.

This module defines the three density data structures used throughout
the toolbox:

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


class MaetDensity:
    """Precomputed multi-attribute expectation tensor density (MAET).

    Returned by :func:`build_exp_tens` when called in multi-attribute
    form (first argument a list/tuple of attribute matrices). The
    single-multiset densities are MaetDensity at A = N = 1.

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
        """Equivalent density with dead contributors dropped.

        The single-multiset corner (A = N = 1) prunes at the value
        level --- drop zero-/NaN-weight values --- because its events
        have already pooled into one and an event-level pass cannot
        reach a dead value inside the single live event. Every other
        shape prunes at the event level: dead events contribute nothing
        to any per-attribute inner product or total mass, so dropping
        them leaves results unchanged while shrinking the O(n) / O(n^2)
        work and the per-tuple expansion. This value-/event-level split
        mirrors the MATLAB ``prunedExpTens`` branches. Returns ``self``
        when nothing is dead. The subset is rebuilt through the same
        lazy machinery ``build_exp_tens`` uses, so the per-tuple fields
        stay correct for any consumer that later materialises them.
        """
        if is_single_multiset(self):
            v = single_multiset_view(self)
            if bool(v.live_events.any()):
                return v.pruned()._d
            # Every value dead: a zero-mass density. Fall through to the
            # event-level path, which drops the one dead event to N = 0;
            # renyi2's N == 0 guard then returns NaN (collision entropy of
            # zero mass is undefined), rather than attempting to build an
            # empty single multiset.
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

def is_single_multiset(dens):
    """True when a density is the single-multiset corner (A = N = 1).

    A single flat (non-nested) attribute whose values are one weighted
    multiset (the ET of Milne 2011). The ``A == 1, r == 1`` case with
    ``N > 1`` never reaches here as such: :func:`_build_exp_tens_ma`
    collapses it into one pooled event at build (a tuple is a lone value
    at r = 1), so every downstream consumer only ever meets the
    canonical ``N == 1`` form. Evaluation strategy for this corner is
    shape-gated, not type-gated.
    """
    if isinstance(dens, MaetDensity):
        return (dens.n_attrs == 1 and dens.n == 1
                and dens.nested[0] is None)
    return False


class _SingleMultisetView:
    """Single-multiset view of a :class:`MaetDensity`.

    Exposes the flat field names (``p``, ``w``, scalar
    parameters, and the lazily materialised per-tuple arrays) over a
    ``MaetDensity`` at the ``A = N = 1`` corner, so single-multiset
    evaluation code reads one layout. Construct via
    :func:`single_multiset_view`.
    """

    __slots__ = ("_d", "__weakref__")

    def __init__(self, d):
        self._d = d

    # --- flat parameters ---
    @property
    def p(self):
        return self._d.p_attr[0][:, 0]

    @property
    def w(self):
        return self._d.w[0][:, 0]

    @property
    def sigma(self):
        return float(self._d.sigma[0])

    @property
    def r(self):
        return int(self._d.r[0])

    @property
    def is_rel(self):
        return bool(self._d.is_rel[0])

    @property
    def is_per(self):
        return bool(self._d.is_per[0])

    @property
    def period(self):
        return float(self._d.period[0])

    @property
    def is_sym(self):
        return bool(self._d.is_sym[0])

    @property
    def dim(self):
        return int(self._d.dim_per_attr[0])

    # --- lazily materialised per-tuple arrays (joint == single-
    #     attribute at this corner; per-attribute lists indexed at 0) ---
    @property
    def materialised(self):
        return self._d.materialised

    @property
    def centres(self):
        return self._d.centres[0]

    @property
    def u_perm(self):
        return self._d.u_perm[0]

    @property
    def w_perm(self):
        return self._d.w_j

    @property
    def w_j(self):
        return self._d.w_j

    @property
    def v_comb(self):
        return self._d.v_comb[0]

    @property
    def wv_comb(self):
        return self._d.wv_comb

    @property
    def n_j(self):
        return self._d.n_j

    @property
    def n_j_perm(self):
        return self._d.n_j

    @property
    def n_k(self):
        return self._d.n_k

    @property
    def live_events(self):
        """Boolean ``(K,)`` mask of live values (single-multiset
        semantics: an element is live iff its weight is finite and
        nonzero)."""
        return _weight_is_live(self.w)

    @property
    def kernel_cov(self):
        return getattr(self._d, "kernel_cov", None)

    @property
    def kernel_chol(self):
        return getattr(self._d, "kernel_chol", None)

    def pruned(self):
        """Equivalent single-multiset density restricted to live
        values (mirrors the historical value-level pruning rule: drop
        zero-/NaN-weight values; no-op under a matrix-valued kernel).
        """
        live = self.live_events
        if live.all():
            return self
        if self.kernel_cov is not None:
            return self
        from .build import _build_exp_tens_single_multiset
        out = _build_exp_tens_single_multiset(
            self.p[live], self.w[live], self.sigma, self.r,
            self.is_rel, self.is_per, self.period, self.is_sym,
            verbose=False,
        )
        return single_multiset_view(out)


def single_multiset_view(dens):
    """Single-multiset view of a density (idempotent).

    Returns ``dens`` unchanged for an
    existing view; wraps a :class:`MaetDensity` at the single-
    collection corner (see :func:`is_single_multiset`) in a
    :class:`_SingleMultisetView`. Raises for any other shape.
    """
    if isinstance(dens, _SingleMultisetView):
        return dens
    if is_single_multiset(dens):
        # Cache the view on the density so repeated wrapping is
        # identity-stable (batch deduplication pairs operands by
        # object identity). The cache slot holds only a weak
        # reference: the view keeps the density alive (callers may
        # hold just the view), but the density must not keep the view
        # alive, or every viewed density becomes a reference cycle
        # whose numpy arrays wait for the cyclic collector instead of
        # dying by refcount.
        import weakref
        ref = getattr(dens, "_single_multiset_view_cache", None)
        v = ref() if ref is not None else None
        if v is None:
            v = _SingleMultisetView(dens)
            dens._single_multiset_view_cache = weakref.ref(v)
        return v
    raise ValueError(
        "single_multiset_view requires a single-multiset density (one flat "
        "attribute, one event); got "
        f"n_attrs={getattr(dens, 'n_attrs', '?')}, "
        f"n={getattr(dens, 'n', '?')}."
    )
