"""Expectation tensor construction, evaluation, and similarity.

Re-export shim over the :mod:`mpt._tensor` sub-package. Every name
that was previously importable from ``mpt.tensor`` (whether public or
private) remains importable from here. New code should prefer
importing from ``mpt`` (public names) or directly from the
``_tensor.X`` sub-modules (developer-facing internals).

See ARCHITECTURE.md §3 ("Code layering") for the sub-package layout
and §1 ("Overview") for the public API surface.
"""
from __future__ import annotations

# --- Public API (re-exported by mpt/__init__.py) ---
from ._tensor.density import MaetDensity
from ._tensor.build import build_exp_tens
from ._tensor.eval import eval_exp_tens, eval_exp_tens_raw
from ._tensor.cosine import (batch_cos_sim_exp_tens, cos_sim_exp_tens, cos_sim_exp_tens_raw)
from ._tensor.sweep import sweep_cos_sim_exp_tens, sweep_eligibility
from ._tensor.preprocessing import (
    TranslateAttributesNoOpWarning,
    TranslatedSweep,
    bind_events,
    difference_events,
    flat_specs,
    simplex_vertices,
    translate_attributes,
    weight_events,
)
from ._tensor.windowed import (
    windowed_similarity,
    windowed_entropy,
)

# --- Developer-facing names re-exported for back-compat ---
# These are imported directly from mpt.tensor by tests, demos, and
# downstream code. They retain that import path here. Their canonical
# home is the corresponding _tensor.* sub-module; prefer importing
# from there in new code.

# Density-layer helpers (mostly used by MA input plumbing).
from ._tensor.density import (
    _broadcast_attr_weight,
    _cartesian_indices,
    _coerce_attr_matrix,
    _nchoosek_indices,
    _normalise_weights_ma,
)

# Build-layer.
from ._tensor.build import _looks_like_multi_attr

# Canonical-key helpers (used by some tests via mpt.tensor).
from ._tensor.canonical import (
    _chord_canonical_key,
    _pair_canonical_key,
)

# Dispatch-layer helpers (used by tests).
from ._tensor.dispatch import (
    _compute_Q,
    _normalize_density_input,
    _resolve_list_list_mode,
    _select_ma_inner_product_method,
    # Policy constants (imported by dispatcher tests).
    _CENTRES_WORKING_SET_SOFT_BUDGET,
    _DISPATCH_MEM_BUDGET,
    _ORBIT_R_MAX_SHIPPED,
    _orbit_sigma_over_p_threshold,
)

# Cosine-layer MA-method dispatchers and the Möbius per-attribute matrix
# (imported by tests).
from ._tensor.cosine import (
    _cos_sim_exp_tens_ma_orbit,
    _cos_sim_exp_tens_ma_pairwise,
)
from ._tensor._mobius_inner import _ma_per_attr_inner_matrix
