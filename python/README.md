# Music Perception Toolbox (Python)

Perceptually and cognitively motivated measures of pitch similarity,
consonance, and the structure of scales and rhythms.

The toolbox is available in parallel MATLAB and Python implementations with
identical semantics. This package is the Python one; the full repository,
including the MATLAB implementation, the User Guide, and the demos, is at
<https://github.com/andymilne/Music-Perception-Toolbox>.

## Install

```bash
pip install music-perception-toolbox
```

## What it provides

**Expectation tensors and their cosine similarity.** A weighted multiset of
pitches or time points is smoothed into a continuous density over its
*r*-tuples, with a single parameter σ modelling perceptual uncertainty. The
density and the cosine similarity of two densities are computed analytically,
without discretization, in all four modes (absolute or relative, non-periodic
or periodic). The (S)P(C)S family of similarity measures are instances of
this.

**Consonance and harmonicity.** Spectral entropy, template harmonicity,
tensor harmonicity, virtual pitch profiles, and Sethares roughness.

**Scale and rhythm structure.** Circular DFT, balance, evenness, coherence,
sameness, *n*-tuple entropy, edge detection, projected centroid, mean offset,
the circular autocorrelation phase matrix, and Markov prediction.

**Audio input.** Spectral peaks extracted from a recording can be used
wherever symbolic pitches can.

## Quick example

```python
import numpy as np
import mpt

major = np.array([0.0, 400.0, 700.0])          # cents
minor = np.array([0.0, 300.0, 700.0])

# Spectral pitch class similarity: harmonic partials added, octave-periodic.
spectrum = ["harmonic", 16, "powerlaw", 1.0]
p_a, w_a = mpt.add_spectra(major, None, *spectrum)
p_b, w_b = mpt.add_spectra(minor, None, *spectrum)
d_a = mpt.build_exp_tens(p_a, w_a, 10.0, 1, False, True, 1200)
d_b = mpt.build_exp_tens(p_b, w_b, 10.0, 1, False, True, 1200)
print(mpt.cos_sim_exp_tens(d_a, d_b))
```

## Documentation

The [User Guide](https://github.com/andymilne/Music-Perception-Toolbox/blob/master/USER_GUIDE.md)
covers the conceptual foundations and gives a complete function reference for
both languages. Nine demo scripts, also provided as Jupyter notebooks, are in
`python/demos/`.

## Citation

See `CITATION.cff` in the repository.

## Licence

MIT.
