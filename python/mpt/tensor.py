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
from ._tensor.density import (
    ExpTensDensity,
    MaetDensity,
    WindowedMaetDensity,
)
from ._tensor.build import build_exp_tens
from ._tensor.eval import eval_exp_tens, eval_exp_tens_raw
from ._tensor.cosine import (
    batch_cos_sim_exp_tens,
    cos_sim_exp_tens,
    cos_sim_exp_tens_raw,
)
from ._tensor.preprocessing import (
    TranslateAttributesNoOpWarning,
    bind_events,
    difference_events,
    flat_specs,
    simplex_vertices,
    translate_attributes,
    weight_events,
)
from ._tensor.windowing import (
    window_tensor,
    windowed_similarity,
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

# Windowing-layer.
from ._tensor.windowing import (
    _evaluate_window_on_query,
    _window_width_params,
    _windowed_inner_product,
)

# Canonical-key helpers (used by some tests via mpt.tensor).
from ._tensor.canonical import (
    _chord_canonical_key,
    _pair_canonical_key,
)

# Dispatch-layer helpers (used by windowing's lazy imports and by tests).
from ._tensor.dispatch import (
    _compute_Q,
    _estimate_centres_array_bytes,
    _normalize_density_input,
    _orbit_ips_look_corrupted,
    _resolve_list_list_mode,
    _select_and_estimate_sa,
    _select_and_estimate_sa_ip,
    _select_ma_inner_product_method,
    _select_sa_eval_method,
    _select_sa_inner_product_method,
    # Policy constants (imported by dispatcher tests).
    _CENTRES_PROBE_MEM_BUDGET,
    _ORBIT_R_MAX_SHIPPED,
    _ORBIT_SIGMA_OVER_P_THRESHOLD,
    _PRESCREEN_IP_DOMINANCE,
    _PROBE_K_IP_TARGET,
    _PROBE_MIN_N_Q,
)

# Cosine-layer SA/MA-method dispatchers and IP helpers (imported by tests).
from ._tensor.cosine import (
    _batched_direct_enum_abs_sa,
    _build_ordered_r_tuples,
    _cos_sim_exp_tens_ma_orbit,
    _cos_sim_exp_tens_ma_pairwise,
    _cos_sim_exp_tens_sa_orbit,
    _cos_sim_exp_tens_sa_pairwise,
    _inner_product_direct_abs_sa,
    _ma_has_nan,
    _ma_per_attr_inner_matrix,
    _orbit_inner_abs,
    _orbit_inner_rel,
    _pack_nan_top,
)
