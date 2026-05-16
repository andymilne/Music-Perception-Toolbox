"""Internal sub-package of mpt.tensor.

This sub-package is the in-progress refactor target for the
expectation-tensor primitives that currently live in
``mpt.tensor``. Public toolbox names continue to be exposed at
``mpt.X`` (re-exported by ``mpt/__init__.py`` from ``mpt.tensor``).
``mpt.tensor`` itself imports from this sub-package for the
already-migrated subsystems.

Migration status (post-Tranche-2-phase-1+2):
  density.py        Migrated.
  preprocessing.py  Migrated.
  windowing.py      Migrated.

Still in mpt.tensor (will move in Tranche 2 phase 3):
  build / eval / cosine / dispatch / canonical machinery.

See ARCHITECTURE.md §3 ("Code layering") for the target structure.
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

from .preprocessing import (
    bind_events,
    difference_events,
    simplex_vertices,
)

from .windowing import (
    WindowedSimilarityPeriodicApproxWarning,
    window_tensor,
    windowed_similarity,
)

__all__ = [
    # Density classes (public)
    "ExpTensDensity",
    "MaetDensity",
    "WindowedMaetDensity",
    # Preprocessing (public)
    "bind_events",
    "difference_events",
    "simplex_vertices",
    # Windowing (public)
    "WindowedSimilarityPeriodicApproxWarning",
    "window_tensor",
    "windowed_similarity",
]
