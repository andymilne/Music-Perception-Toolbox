"""Internal sub-package implementing expectation tensor primitives.

Public toolbox names continue to be exposed at ``mpt.X`` (re-exported
by ``mpt/__init__.py`` from ``mpt.tensor``, which is now itself a
re-export shim over this sub-package).

Module layout:
  density.py        Density classes + MA-input helpers.
  build.py          build_exp_tens (multi-attribute; the single-multiset
                    density is its A = N = 1 corner).
  preprocessing.py  difference_events, bind_events, translate_attributes,
                    simplex_vertices.
  windowing.py      window_tensor, windowed_tensor_similarity, windowed IP.
  canonical.py      Canonical-form key helpers for batched dedup.
  dispatch.py       Path-selection cost model + shared helpers.
  eval.py           eval_exp_tens (joint centres / factored / Möbius).
  cosine.py         cos_sim_exp_tens + batch_cos_sim_exp_tens.
  sweep.py          Translation sweeps as a mixture in the offset.

See ARCHITECTURE.md §3 ("Code layering") for the layered design.
"""
from .density import (
    MaetDensity,
    WindowedMaetDensity,
    _broadcast_attr_weight,
    _cartesian_indices,
    _coerce_attr_matrix,
    _nchoosek_indices,
    _normalise_weights_ma,
)

from .build import (
    build_exp_tens,
    _looks_like_multi_attr,
)

from .sweep import (
    sweep_cos_sim_exp_tens,
    sweep_eligibility,
)

from .preprocessing import (
    TranslateAttributesNoOpWarning,
    TranslatedSweep,
    bind_events,
    difference_events,
    simplex_vertices,
    translate_attributes,
    weight_events,
)

from .windowing import (
    window_tensor,
    windowed_tensor_similarity,
    _evaluate_window_on_query,
    _windowed_inner_product,
)
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
    eval_exp_tens,
    eval_exp_tens_raw,
)

from .cosine import (batch_cos_sim_exp_tens, cos_sim_exp_tens, cos_sim_exp_tens_raw)


__all__ = [
    # Density (public)
    "MaetDensity",
    "WindowedMaetDensity",
    # Build / eval / cosine (public)
    "build_exp_tens",
    "eval_exp_tens",
    "eval_exp_tens_raw",
    "cos_sim_exp_tens",
    "cos_sim_exp_tens_raw",
    "batch_cos_sim_exp_tens",
    # Preprocessing (public)
    "bind_events",
    "difference_events",
    "simplex_vertices",
    "translate_attributes",
    "TranslateAttributesNoOpWarning",
    "weight_events",
    # Windowing (public)
    "window_tensor",
    "windowed_tensor_similarity",
    "windowed_similarity",
    "windowed_entropy",
]
