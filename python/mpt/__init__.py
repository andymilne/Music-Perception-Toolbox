"""Music Perception Toolbox (mpt).

A Python package for computational music perception research,
ported from the MATLAB Music Perception Toolbox v2.0.0.

Andrew J. Milne, MARCS Institute, Western Sydney University.
David Bulger credited as co-author of the original ``cos_sim_exp_tens``
and ``markov_s`` functions.
"""

from __future__ import annotations

__version__ = "2.0.0"

# --- Pitch/frequency conversion ---
from .convert import convert_pitch

# --- Spectral enrichment ---
from .spectra import add_spectra

# --- Expectation tensors ---
from .tensor import (
    ExpTensDensity,
    MaetDensity,
    WindowedMaetDensity,
    WindowedSimilarityPeriodicApproxWarning,
    batch_cos_sim_exp_tens,
    build_exp_tens,
    cos_sim_exp_tens,
    cos_sim_exp_tens_raw,
    eval_exp_tens,
    eval_exp_tens_raw,
    difference_events,
    bind_events,
    simplex_vertices,
    window_tensor,
    windowed_similarity,
)

# --- Circular measures ---
from .circular import (
    balance,
    circ_apm,
    coherence,
    dft_circular,
    dft_circular_simulate,
    edges,
    evenness,
    markov_s,
    mean_offset,
    proj_centroid,
    sameness,
)

# --- Entropy ---
from .entropy import entropy_exp_tens, n_tuple_entropy

# --- Harmony / consonance ---
from .harmony import (
    roughness,
    spectral_entropy,
    template_harmonicity,
    tensor_harmonicity,
    virtual_pitches,
)

# --- Utility ---
from ._utils import estimate_comp_time

# --- Audio ---
from .audio import AudioPeaksDetail, audio_peaks

# --- Serial / sequential analysis ---
from .serial import continuity, seq_weights

__all__ = [
    # convert
    "convert_pitch",
    # spectra
    "add_spectra",
    # tensor
    "ExpTensDensity",
    "MaetDensity",
    "build_exp_tens",
    "eval_exp_tens",
    "eval_exp_tens_raw",
    "cos_sim_exp_tens",
    "cos_sim_exp_tens_raw",
    "batch_cos_sim_exp_tens",
    "difference_events",
    "bind_events",
    "simplex_vertices",
    "window_tensor",
    "windowed_similarity",
    "WindowedMaetDensity",
    "WindowedSimilarityPeriodicApproxWarning",
    # circular
    "dft_circular",
    "dft_circular_simulate",
    "balance",
    "evenness",
    "coherence",
    "sameness",
    "edges",
    "proj_centroid",
    "mean_offset",
    "circ_apm",
    "markov_s",
    # entropy
    "entropy_exp_tens",
    "n_tuple_entropy",
    # harmony
    "spectral_entropy",
    "template_harmonicity",
    "tensor_harmonicity",
    "virtual_pitches",
    "roughness",
    # utility
    "estimate_comp_time",
    # audio
    "audio_peaks",
    "AudioPeaksDetail",
    # serial
    "continuity",
    "seq_weights",
]
