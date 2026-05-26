"""Internal sub-package implementing expectation tensor primitives.

Public toolbox names continue to be exposed at ``mpt.X`` (re-exported
by ``mpt/__init__.py`` from ``mpt.tensor``, which is now itself a
re-export shim over this sub-package).

Module layout:
  density.py        Density classes + MA-input helpers.
  build.py          build_exp_tens (SA + MA paths).
  preprocessing.py  difference_events, bind_events, translate_attributes,
                    simplex_vertices.
  windowing.py      window_tensor, windowed_similarity, windowed IP.
  canonical.py      Canonical-form key helpers for batched dedup.
  dispatch.py       Path-selection cost model + shared helpers.
  eval.py           eval_exp_tens (SA centres / orbit / fast, MA).
  cosine.py         cos_sim_exp_tens + batch_cos_sim_exp_tens.

See ARCHITECTURE.md §3 ("Code layering") for the layered design.
"""
from .density import (
    ExpTensDensity,
    MaetDensity,
    WindowedMaetDensity,
    _broadcast_attr_weight,
    _canon_groups_cell_form,
    _canon_groups_vector_form,
    _canonicalise_groups,
    _cartesian_indices,
    _coerce_attr_matrix,
    _nchoosek_indices,
    _normalise_weights_ma,
)

from .build import (
    build_exp_tens,
    _looks_like_multi_attr,
)

from .preprocessing import (
    TranslateAttributesNoOpWarning,
    bind_events,
    difference_events,
    simplex_vertices,
    translate_attributes,
    weight_events,
)

from .windowing import (
    window_tensor,
    windowed_similarity,
    _evaluate_window_on_query,
    _windowed_inner_product,
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

from .cosine import (
    batch_cos_sim_exp_tens,
    cos_sim_exp_tens,
    cos_sim_exp_tens_raw,
    _cos_sim_exp_tens_sa_orbit,
    _cos_sim_exp_tens_sa_pairwise,
)


__all__ = [
    # Density (public)
    "ExpTensDensity",
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
    "windowed_similarity",
]
