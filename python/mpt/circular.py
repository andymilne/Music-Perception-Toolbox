"""Measures on circular multisets of pitches or positions.

Re-export shim over the :mod:`mpt._circular` sub-package. Every name
that was previously importable from ``mpt.circular`` remains
importable from here. New code should prefer importing from ``mpt``
(public names) or directly from the ``_circular.X`` sub-modules
(developer-facing internals).

See ARCHITECTURE.md §3 ("Code layering") for the sub-package layout.
"""
from __future__ import annotations

from ._circular.dft import (
    balance,
    dft_circular,
    dft_circular_simulate,
    evenness,
    proj_centroid,
)

from ._circular.scale import (
    coherence,
    sameness,
)

from ._circular.pulse import (
    circ_apm,
    edges,
    markov_s,
    mean_offset,
)
