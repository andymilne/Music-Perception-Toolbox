"""Internal sub-package implementing circular-multiset measures.

Public toolbox names are exposed at ``mpt.X`` (re-exported by
``mpt/__init__.py`` from ``mpt.circular``, which is now itself a
re-export shim over this sub-package).

Module layout:
  dft.py    DFT engine and DFT-based measures: dft_circular,
            dft_circular_simulate, balance, evenness, proj_centroid.
  scale.py  Integer-position scale-theoretic measures (non-Fourier):
            coherence, sameness.
  pulse.py  Per-position pulse-level measures (non-Fourier):
            edges, mean_offset, circ_apm, markov_s.

See ARCHITECTURE.md §3 ("Code layering") for the layered design.
"""
from .dft import (
    balance,
    dft_circular,
    dft_circular_simulate,
    evenness,
    proj_centroid,
)

from .scale import (
    coherence,
    sameness,
)

from .pulse import (
    circ_apm,
    edges,
    markov_s,
    mean_offset,
)


__all__ = [
    # DFT-based
    "balance",
    "dft_circular",
    "dft_circular_simulate",
    "evenness",
    "proj_centroid",
    # Scale-theoretic
    "coherence",
    "sameness",
    # Pulse-level
    "circ_apm",
    "edges",
    "markov_s",
    "mean_offset",
]
