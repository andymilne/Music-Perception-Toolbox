"""Music Perception Toolbox (mpt).

A Python package for computational music perception research, with a
sibling MATLAB implementation maintained in parallel.

Andrew J. Milne, MARCS Institute, Western Sydney University.
David Bulger credited as co-author of the original ``cos_sim_exp_tens``
and ``markov_s`` functions.
"""

from __future__ import annotations

__version__ = "3.0.0"

# --- Pitch/frequency conversion ---
from .convert import convert_pitch

# --- Spectral enrichment ---
from .spectra import add_spectra

# --- Expectation tensors ---
from .tensor import (
    MaetDensity,
    batch_cos_sim_exp_tens,
    build_exp_tens,
    cos_sim_exp_tens,
    cos_sim_exp_tens_raw,
    sweep_cos_sim_exp_tens,
    eval_exp_tens,
    eval_exp_tens_raw,
    difference_events,
    bind_events,
    flat_specs,
    simplex_vertices,
    translate_attributes,
    weight_events,
    TranslateAttributesNoOpWarning,
    TranslatedSweep,
    windowed_similarity,
    windowed_entropy,
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

# Diagnostic: report how a call would be routed, and why.
from ._tensor.explain import explain_dispatch

# --- Audio ---
from .audio import AudioPeaksDetail, audio_peaks

# --- Serial / sequential analysis ---
from .serial import continuity, interval_kernel_cov, seq_weights

# --- Global defaults ---
from ._defaults import (
    TruncationDefaultWarning,
    get_default,
    get_defaults,
    reset_defaults,
    set_default,
    show_defaults,
)

__all__ = [
    "explain_dispatch",
    # convert
    "convert_pitch",
    # spectra
    "add_spectra",
    # tensor
    "MaetDensity",
    "build_exp_tens",
    "eval_exp_tens",
    "eval_exp_tens_raw",
    "cos_sim_exp_tens",
    "cos_sim_exp_tens_raw",
    "batch_cos_sim_exp_tens",
    "sweep_cos_sim_exp_tens",
    "difference_events",
    "bind_events",
    "flat_specs",
    "simplex_vertices",
    "translate_attributes",
    "TranslatedSweep",
    "weight_events",
    "TranslateAttributesNoOpWarning",
    "windowed_similarity",
    "windowed_entropy",
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
    "interval_kernel_cov",
    "seq_weights",
    # defaults
    "get_default",
    "get_defaults",
    "reset_defaults",
    "set_default",
    "show_defaults",
    "TruncationDefaultWarning",
]
