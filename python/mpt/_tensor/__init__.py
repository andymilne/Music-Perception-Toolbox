"""Internal sub-package implementing expectation tensor primitives.

Public toolbox names continue to be exposed at ``mpt.X`` (re-exported
by ``mpt/__init__.py`` from ``mpt.tensor``, which is now itself a
re-export shim over this sub-package).

Module layout:
  density.py        Density classes + MA-input helpers.
  build.py          build_maet (multi-attribute; the single-multiset
                    density is its A = N = 1 corner).
  preprocessing.py  difference_events, bind_events, translate_attributes,
                    simplex_vertices.
  transform.py      transform_attributes (scale conversions and elementwise
                    transforms).
  windowed.py       windowed_similarity, windowed_entropy (event weighting).
  canonical.py      Canonical-form key helpers for batched dedup.
  dispatch.py       Path-selection cost model + shared helpers.
  eval.py           eval_maet (joint centres / factored / Möbius).
  cosine.py         sim_maet and sweep_sim_maet.
  sweep.py          Translation sweeps as a mixture in the offset.

See ARCHITECTURE.md §3 ("Code layering") for the layered design.
"""
from .density import (
    MaetDensity,
    _broadcast_attr_weight,
    _cartesian_indices,
    _coerce_attr_matrix,
    _nchoosek_indices,
    _normalise_weights_ma,
)

from .build import (
    build_maet,
    _looks_like_multi_attr,
)

from .sweep import (
    sweep_sim_maet,
    sweep_eligibility,
)

from .preprocessing import (
    TranslateAttributesNoOpWarning,
    TranslatedSweep,
    bind_attributes,
    bind_events,
    difference_events,
    separate_attributes,
    simplex_vertices,
    translate_attributes,
    weight_events,
)
from .premaet import is_pre_maet, pack_pre_maet, unpack_pre_maet
from .transform import transform_attributes

from .windowed import (
    windowed_similarity,
    windowed_entropy,
)

from .canonical import (
    _chord_canonical_key,
    _pair_canonical_key,
)

from .dispatch import (
    _compute_Q,
    _normalize_density_input,
    _resolve_list_list_mode,
)

from .eval import (
    eval_maet,
)

from .cosine import sim_maet


__all__ = [
    # Density (public)
    "MaetDensity",
    # Build / eval / cosine (public)
    "build_maet",
    "eval_maet",
    "sim_maet",
    # Preprocessing (public)
    "bind_events",
    "difference_events",
    "simplex_vertices",
    "translate_attributes",
    "transform_attributes",
    "TranslateAttributesNoOpWarning",
    "weight_events",
    # Windowing (public)
    "windowed_similarity",
    "windowed_entropy",
]
