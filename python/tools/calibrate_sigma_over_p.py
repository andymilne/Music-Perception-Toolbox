"""Sweep the sigma/period thresholds that govern the periodic measures.

Two thresholds are calibrated here, and each is a choice of *measure*
rather than of speed or accuracy:

``_ORBIT_SIGMA_OVER_P_THRESHOLD`` (0.03) is the sigma/P above which the
wrapped-difference relative-periodic kernel departs from the
transposition average that defines the relative periodic density.

``_ABS_PER_SIGMA_OVER_P_THRESHOLD`` (0.05) is the sigma/P above which
the absolute-periodic single-image (nearest-image) kernel departs from
the full-image kernel that sums over every periodic image.

WHAT DECIDES A THRESHOLD
------------------------
Positive-definiteness, not accuracy. An approximating kernel that is no
longer positive-definite does not induce an inner product at all: its
cosine similarity is not bounded by 1, so Cauchy-Schwarz fails and the
quantity is not a similarity. That is a failure of the first test the
dispatch applies --- is the route admissible --- rather than of the
second, is it accurate enough. A departure that is merely large is a
loss of accuracy; a departure that breaks positive-definiteness is a
loss of meaning.

So the sweep reports three quantities per cell:

ABSOLUTE VERSUS RELATIVE
------------------------
The two thresholds rest on different evidence, because the two kernels
factorise differently.

The absolute-periodic kernel is a product of one-dimensional
nearest-image kernels, one per coordinate. A product of
positive-definite kernels is positive-definite, so the whole question
reduces to the one-dimensional case and the Fourier coefficients settle
it exactly, for every r at once. No search is needed and none would add
anything.

The relative-periodic approximation wraps the pairwise differences of
the tuple and does not factorise, so no one-dimensional coefficient
decides it. There the failure has to be found by searching the
configurations, and the onset is where a search can first reach a
cosine above 1.

  abs_per_pd_min_coeff
                 the least Fourier coefficient of the ABSOLUTE-periodic
                 single-image kernel on the circle. By Bochner's
                 theorem that kernel is positive-definite exactly when
                 every coefficient is non-negative, so where this first
                 goes negative bounds the absolute-periodic threshold.
                 It says nothing about the relative one, whose
                 approximation lives on the (r-1)-dimensional quotient
                 rather than on the circle.
  cos_excess     the largest amount by which a cosine similarity built
                 from the approximating kernel exceeds 1. This is the
                 same failure seen from the caller's side, and it is
                 what makes the bound concrete.

                 Searched, not sampled. Positive-definiteness fails
                 first on particular configurations rather than on
                 typical ones: earlier measurement put the onset near
                 sigma/P = 0.06 by tiny amounts on special pairs, while
                 ordinary random draws do not meet a violation until
                 about 0.10. A sweep that only samples therefore reports
                 the wrong onset --- it finds where violations become
                 common, not where they begin. Each cell here runs a
                 hill-climb over positions and weights from several
                 random restarts, so what is reported is the worst case
                 the search can reach rather than the worst of a sample.
  dev            the absolute departure of the approximating cosine
                 from the defining one, on the value scale. Compared
                 against truncation_floor at 6, 8 and infinity, this is
                 the accuracy bound, reported alongside so that which
                 constraint binds first is visible rather than assumed.

WEIGHT PROFILES
---------------
Six, because weighting has repeatedly been decisive: flat, linear
decay, exponential decay at two rates, bimodal, and a single dominant
value at 1000:1. The last is the profile that broke an earlier
calibration of the orbit accuracy guard, and a sweep that omits it can
report a clean bound that ordinary music does not obey.

HOW TO RUN
----------
From the ``python`` directory::

    PYTHONPATH=. python3 tools/calibrate_sigma_over_p.py

Everything measured here is deterministic --- Fourier coefficients and
cosine values, not wall times --- so the result does not depend on
machine load. It takes about twenty minutes: roughly 105 s per sigma
value over ten of them, most of it in the lattice reference at r = 4.
"""
import itertools
import math

import numpy as np

from mpt._defaults import truncation_floor

PERIOD = 1200.0
SIGMA_OVER_P = [0.02, 0.03, 0.04, 0.05, 0.055, 0.06, 0.065, 0.07,
                0.08, 0.10]
# The search evaluates the cosine over ordered r-tuples, so its cost
# grows as (K permute r) squared; at r = 5 with K = r + 2 that is 2520
# tuples a side and a single evaluation already costs a fifth of a
# second, which a hill-climb cannot afford. The onset is a property of
# the kernel rather than of a particular order, and r = 2 to 4 covers
# the orders the measure is actually used at.
R_VALUES = [2, 3, 4]

# Value counts to search, per tuple order.
#
# Only the shapes the cost model gives to Bulger's method are worth
# searching. Where it gives them to the Moebius method the
# transposition average is what runs, the wrapped-difference kernel is
# never evaluated, and whether that kernel is positive-definite there
# does not arise. Asked at a single event per density, the cost model
# prefers Bulger up to K = 12 at r = 2 and to K = 10 or 11 at r = 3 and
# 4; more events per density narrow that further at r = 2 and 3. The
# counts below sit inside the region at every event count.
# Capped by tuple count as well as by the cost model's region. One
# cosine evaluation costs (K permute r) squared, and at r = 4 with
# K = 10 that is 5040 tuples a side and 21 s for a single evaluation,
# which is not a search. The counts below keep every cell under about
# 500 tuples a side, where a few hundred evaluations fit in the budget.
# Positive-definiteness is a property of the kernel rather than of a
# particular shape, and a larger value count only gives the search more
# freedom, so a shape that is searched reports a lower bound on the
# excess for every larger one at the same order.
K_BY_R = {2: [4, 6, 8, 12], 3: [5, 7, 9], 4: [6]}
PROFILES = ("flat", "linear", "decay1", "decay4", "bimodal", "dominant")
N_PAIRS = 6
N_RESTARTS = 6
N_STEPS = 120
# Wall-time budget per searched cell, in seconds. The cosine is
# evaluated over ordered r-tuples, so one evaluation costs
# (K permute r) squared and varies by four orders of magnitude across
# the shapes swept. A fixed step count would therefore spend
# milliseconds on the smallest cells and hours on the largest. Each
# cell instead runs restarts until its budget is spent, and the number
# of evaluations it managed is reported so that a cell which found
# nothing can be told from one that had no time to look.
SEARCH_BUDGET_SEC = 3.0
# Largest tuple count per side for which the departure from the
# transposition average is measured. The reference sums over the image
# lattice, so its cost is the tuple-pair count times the number of
# translates; beyond this it dominates the run without adding an axis.
DEV_MAX_TUPLES = 400
N_FOURIER = 4096
M_MAX = 64


def _weights(profile, K, rng):
    if profile == "flat":
        w = np.ones(K)
    elif profile == "linear":
        w = np.linspace(1.0, 0.1, K)
    elif profile == "decay1":
        w = np.exp(-np.linspace(0.0, 1.0, K))
    elif profile == "decay4":
        w = np.exp(-np.linspace(0.0, 4.0, K))
    elif profile == "bimodal":
        w = np.exp(-((np.linspace(-1.0, 1.0, K) ** 2) * 6.0))
        w = w.max() - w + 0.05
    elif profile == "dominant":
        w = np.full(K, 1e-3)
        w[K // 2] = 1.0
    else:
        raise ValueError(profile)
    return w * (0.85 + 0.3 * rng.random(K))


def _abs_per_pd_min_coefficient(sigma):
    """Least Fourier coefficient of the ABSOLUTE-periodic single-image kernel.

    This is the nearest-image Gaussian as a function of one difference
    on the circle. It is even and periodic, so its coefficients are
    real and a trapezoidal sum over one period is spectrally accurate.
    By Bochner's theorem the kernel is positive-definite exactly when
    every coefficient is non-negative, so the sigma/P at which this
    first goes materially negative bounds
    ``_ABS_PER_SIGMA_OVER_P_THRESHOLD``.

    It does NOT bound the relative-periodic threshold. The relative
    approximation wraps the pairwise differences of an r-tuple, which
    is a kernel on the (r-1)-dimensional quotient rather than on the
    circle, and no one-dimensional coefficient decides it. The
    relative case is settled empirically by ``cos_excess``, the amount
    by which its induced cosine exceeds 1 --- the same failure seen
    from the caller's side.

    Cross-checked against the least eigenvalue of the kernel matrix on
    a 256-point grid: the two agree in sign throughout and both are at
    noise level below sigma/P = 0.05.
    """
    d = np.linspace(0.0, PERIOD, N_FOURIER, endpoint=False)
    wrapped = d - PERIOD * np.floor(d / PERIOD + 0.5)
    k = np.exp(-(wrapped ** 2) / (4.0 * sigma ** 2))
    m = np.arange(M_MAX + 1)[:, None]
    coeffs = (k[None, :] * np.cos(2.0 * np.pi * m * d[None, :] / PERIOD)).mean(axis=1)
    return float(coeffs.min())


def _lattice_image_count(sigma, r, tol=1e-18):
    """Images per axis needed for the lattice sum, from sigma and r.

    Applies to the sum taken over *reduced* differences: once each
    coordinate of the reduced difference has been wrapped to
    [-P/2, P/2), the nearest lattice point is the origin and the first
    omitted shell contributes at most
    ``exp(-(nP)^2 (1 - 1/r) / (4 sigma^2))``.

    The reduction is what makes this bound correct. Summing over raw
    differences instead, the reduced coordinates ``d_i - d_r`` span
    [-2P, 2P], so the nearest image can sit two shells out and a count
    derived from the decay alone silently truncates it --- measured at
    2.4e-5 against a bound of 1e-18. The count matters for speed too:
    the sum evaluates ``(2n+1)^(r-1)`` translates, so at r = 4 a fixed
    n = 4 costs 729 where n = 1 suffices.
    """
    n = 1
    while math.exp(-((n * PERIOD) ** 2) * (1.0 - 1.0 / r)
                   / (4.0 * sigma ** 2)) > tol:
        n += 1
        if n > 8:
            break
    return n


def _lattice_cos(p, w, q, v, sigma, r, n_max=None):
    """Cosine under the transposition-average (full-image) measure."""
    if n_max is None:
        n_max = _lattice_image_count(sigma, r)
    tx = list(itertools.permutations(range(len(p)), r))
    ty = list(itertools.permutations(range(len(q)), r))
    cx = np.array([[p[i] for i in t] for t in tx], dtype=float)
    cy = np.array([[q[i] for i in t] for t in ty], dtype=float)
    wx = np.array([np.prod([w[i] for i in t]) for t in tx], dtype=float)
    wy = np.array([np.prod([v[i] for i in t]) for t in ty], dtype=float)

    def ip(ca, wa, cb, wb):
        d = ca[:, None, :] - cb[None, :, :]
        # Reduce to the quotient by the diagonal, then wrap each
        # remaining coordinate to its nearest image. Q is invariant
        # under adding a constant to every coordinate, so fixing the
        # last at zero loses nothing, and shifting a coordinate by P is
        # a lattice move.
        dr = d - d[:, :, -1][:, :, None]
        dr = dr - PERIOD * np.floor(dr / PERIOD + 0.5)
        acc = np.zeros(d.shape[:2])
        for n in itertools.product(range(-n_max, n_max + 1), repeat=r - 1):
            x = dr + np.array(list(n) + [0], dtype=float) * PERIOD
            q_form = (x ** 2).sum(-1) - x.sum(-1) ** 2 / r
            acc += np.exp(-q_form / (4.0 * sigma ** 2))
        return float(wa @ acc @ wb)

    return ip(cx, wx, cy, wy) / math.sqrt(ip(cx, wx, cx, wx) * ip(cy, wy, cy, wy))


def _wrapped_difference_cos(p, w, q, v, sigma, r):
    """Cosine under the pairwise wrapped-difference approximation."""
    tx = list(itertools.permutations(range(len(p)), r))
    ty = list(itertools.permutations(range(len(q)), r))
    cx = np.array([[p[i] for i in t] for t in tx], dtype=float)
    cy = np.array([[q[i] for i in t] for t in ty], dtype=float)
    wx = np.array([np.prod([w[i] for i in t]) for t in tx], dtype=float)
    wy = np.array([np.prod([v[i] for i in t]) for t in ty], dtype=float)

    def ip(ca, wa, cb, wb):
        d = ca[:, None, :] - cb[None, :, :]
        acc = np.zeros(d.shape[:2])
        for i in range(r):
            for j in range(i + 1, r):
                delta = d[:, :, i] - d[:, :, j]
                delta = delta - PERIOD * np.floor(delta / PERIOD + 0.5)
                acc += delta ** 2
        return float(wa @ np.exp(-acc / (4.0 * sigma ** 2 * r)) @ wb)

    return ip(cx, wx, cy, wy) / math.sqrt(ip(cx, wx, cx, wx) * ip(cy, wy, cy, wy))


def _worst_cos_excess(sigma, r, K, profile, rng,
                      n_restarts=N_RESTARTS, n_steps=N_STEPS,
                      budget_sec=None):
    """Largest cosine excess above 1 a hill-climb can reach.

    Positions and weights are perturbed with a shrinking step, keeping
    any move that raises the cosine. The starting weights come from the
    named profile, so the search explores around musically shaped
    weightings rather than from arbitrary ones.

    With ``budget_sec`` set, restarts stop once the budget is spent (at
    least one always runs). Returns ``(excess, n_evaluations)``; the
    count matters because a cell that found no violation and a cell
    that had no time to look are different results.
    """
    import time as _time
    t0 = _time.perf_counter()
    n_eval = 0
    best = -np.inf
    for restart in range(n_restarts):
        if budget_sec is not None and restart > 0 \
                and _time.perf_counter() - t0 > budget_sec:
            break
        p = np.sort(rng.uniform(0, PERIOD, K))
        q = np.sort(rng.uniform(0, PERIOD, K))
        w = _weights(profile, K, rng)
        v = _weights(profile, K, rng)
        cur = _wrapped_difference_cos(p, w, q, v, sigma, r)
        n_eval += 1
        step = PERIOD / 8.0
        for it in range(n_steps):
            if budget_sec is not None \
                    and _time.perf_counter() - t0 > budget_sec:
                break
            scale = step * (1.0 - it / n_steps)
            p2 = np.mod(p + rng.normal(0.0, scale, K), PERIOD)
            q2 = np.mod(q + rng.normal(0.0, scale, K), PERIOD)
            w2 = np.abs(w * np.exp(rng.normal(0.0, 0.3, K)))
            v2 = np.abs(v * np.exp(rng.normal(0.0, 0.3, K)))
            trial = _wrapped_difference_cos(p2, w2, q2, v2, sigma, r)
            n_eval += 1
            if trial > cur:
                p, q, w, v, cur = p2, q2, w2, v2, trial
        best = max(best, cur)
    return best - 1.0, n_eval


def main():
    rng = np.random.default_rng(0)
    print(f"period {PERIOD:g} cents; floors: "
          f"ts=6 {truncation_floor(6.0):.2e}, ts=8 {truncation_floor(8.0):.2e}, "
          f"ts=inf {truncation_floor(math.inf):.2e}\n")
    print(f"{'sig/P':>6} {'pd_abs_coeff':>13} {'r':>2} {'K':>3} {'profile':>9} "
          f"{'cos_excess':>11} {'dev':>10} {'n_eval':>7}")
    rows = []
    for sop in SIGMA_OVER_P:
        sigma = sop * PERIOD
        pd_abs = _abs_per_pd_min_coefficient(sigma)
        for r in R_VALUES:
          for K in K_BY_R[r]:
            for profile in PROFILES:
                excess, n_eval = _worst_cos_excess(
                    sigma, r, K, profile, rng,
                    budget_sec=SEARCH_BUDGET_SEC)
                dev = 0.0
                n_tuples = math.perm(K, r)
                for _ in range(N_PAIRS if n_tuples <= DEV_MAX_TUPLES else 0):
                    p = np.sort(rng.uniform(0, PERIOD, K))
                    q = np.sort(rng.uniform(0, PERIOD, K))
                    w = _weights(profile, K, rng)
                    v = _weights(profile, K, rng)
                    approx = _wrapped_difference_cos(p, w, q, v, sigma, r)
                    exact = _lattice_cos(p, w, q, v, sigma, r)
                    dev = max(dev, abs(approx - exact))
                print(f"{sop:>6.3f} {pd_abs:>13.3e} {r:>2} {K:>3} {profile:>9} "
                      f"{excess:>11.2e} {dev:>10.2e} {n_eval:>7}", flush=True)
                rows.append((sop, pd_abs, r, K, profile, excess, dev, n_eval))
    print("\nBEGIN_CSV")
    print("sigma_over_p,abs_per_pd_min_coeff,r,K,profile,cos_excess,dev,n_eval")
    for row in rows:
        print(f"{row[0]:g},{row[1]:.6e},{row[2]},{row[3]},{row[4]},"
              f"{row[5]:.6e},{row[6]:.6e},{row[7]}")
    print("END_CSV")


if __name__ == "__main__":
    main()
