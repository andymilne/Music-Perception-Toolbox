"""Sensitivity of the consonance measures to four properties of a chord.

The four are made orthogonal by construction:
  type      the pitch-class set (augmented excluded: its inversions are transpositions)
  inversion which chord tone is lowest
  spacing   the chord's pitch range, widened by dropping the bass an octave or two
  register  the chord's mean pitch, set independently of the above
"""
import os
import sys, warnings; warnings.filterwarnings('ignore')
# The toolbox is imported as `mpt`. If it is not installed, point MPT_PYTHON
# at the `python` directory of a version 3.0 checkout.
_MPT = os.environ.get("MPT_PYTHON")
if _MPT:
    sys.path.insert(0, _MPT)
import numpy as np, mpt
from scipy.stats import rankdata

# The published values were computed with an untruncated kernel;
# version 3 truncates at six sigma by default, so the untruncated
# setting is restored here.
mpt.set_default(truncation_sigmas=float('inf'), show_hints=False)

SP, S, ST = ['harmonic', 16, 'powerlaw', 1.0], 12.0, 12.0*np.sqrt(2)
TYPES = {'maj': (0,4,7), 'min': (0,3,7), 'dim': (0,3,6), 'sus4': (0,5,7),
         'tritone': (0,2,6), 'cluster': (0,1,2)}   # augmented excluded
DROPS = [0, 12, 24]          # bass displacement -> spacing
MEANS = [54, 60, 66]         # target mean pitch -> register

rows = []
for t, pcs in TYPES.items():
    for inv in range(3):
        base = [(pcs[(inv+i) % 3] - pcs[inv]) % 12 for i in range(3)]
        for d in DROPS:
            shape = np.array([-d] + base[1:], float)
            shape = shape - shape.min()
            for m in MEANS:
                v = shape + (m - shape.mean())          # set mean pitch exactly
                c = v * 100.0                            # semitones -> cents
                hmax, hent = mpt.template_harmonicity(c, None, S, spectrum=SP,
                                                     chord_spectrum=SP)
                p, w = mpt.add_spectra(c, None, *SP)
                rows.append(dict(
                    type=t, inv=inv, spacing=d, register=m,
                    range=float(v.max()-v.min()), mean=float(v.mean()),
                    tensor=mpt.tensor_harmonicity(c, None, ST, spectrum=SP,
                                                  verbose=False),
                    hMax=hmax, hEnt=-hent,
                    specEnt=-mpt.spectral_entropy(c, None, S, spectrum=SP,
                                                  method='normalized'),
                    rough=-mpt.roughness(
                        mpt.transform_attributes(p, None, ('cents', 'hz')), w)))

def eta2(y, fac):
    y = np.asarray(y, float); g = np.array(fac)
    tot = ((y - y.mean())**2).sum()
    return sum(len(y[g == l])*(y[g == l].mean()-y.mean())**2 for l in set(g))/tot

facs = {k: [r[k] for r in rows] for k in ('type', 'inv', 'spacing', 'register')}
print(f"n = {len(rows)}  ({len(TYPES)} types x 3 inversions x {len(DROPS)} spacings x {len(MEANS)} registers)")
print(f"{'measure':12s}{'type':>8s}{'inversion':>11s}{'spacing':>9s}{'register':>10s}{'resid':>8s}")
for k in ['tensor', 'hMax', 'hEnt', 'specEnt', 'rough']:
    y = rankdata([r[k] for r in rows])
    e = [eta2(y, facs[f]) for f in ('type', 'inv', 'spacing', 'register')]
    print(f"{k:12s}{e[0]:8.2f}{e[1]:11.2f}{e[2]:9.2f}{e[3]:10.2f}{max(0,1-sum(e)):8.2f}")
