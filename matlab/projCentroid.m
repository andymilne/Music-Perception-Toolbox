function [y, centMag, centPhase] = projCentroid(p, w, period, x, sigma)
%PROJCENTROID Projected centroid of a weighted circular multiset.
%
%   y = projCentroid(p, w, period) computes the projection of the
%   circular centroid (centre of gravity) onto each angular position in
%   the sequence 0, 1, ..., period-1. The centroid is the k = 0 Fourier
%   coefficient of the multiset (see dftCircular); its projection onto
%   an angle theta is:
%
%     y(theta) = |F(0)| * cos( angle(F(0)) - 2*pi*theta/period )
%
%   Positive values indicate positions on the "heavy" side of the
%   centroid; negative values indicate the opposite side.
%
%   y = projCentroid(p, w, period, x) evaluates the projection at the
%   query points specified in the vector x (in the same units as p and
%   period).
%
%   y = projCentroid(p, w, period, x, sigma) returns the *expected*
%   projection under independent Gaussian positional jitter on each
%   event. Because y(x) is linear in F(0) and F(0) is permutation-
%   invariant, the result has a clean closed form:
%
%     E[y(x)] = alpha_1 * y_deterministic(x)
%
%   where alpha_1 = exp(-2 * pi^2 * sigma^2 / period^2). No Monte
%   Carlo is needed; the deterministic projection is simply damped
%   by the kernel-smoothing factor alpha_1. Phase is preserved in
%   expectation, so centPhase is unchanged from the deterministic
%   case. centMag returns alpha_1 * |F(0)| — the magnitude of the
%   *expected* centroid, which together with centPhase reproduces
%   y(x) consistently. (For E[|F(0)|] — the average centroid
%   magnitude under jitter, a different scalar that picks up
%   positive bias from the Rayleigh-style geometry — call
%   balanceCircular(..., sigma) and read off 1 - b.)
%
%   At sigma = 0 the v2.0 deterministic value is recovered exactly.
%
%   [y, centMag, centPhase] = projCentroid(...) also returns the
%   centroid magnitude and its phase angle (in the units of p, not
%   radians).
%
%   Inputs:
%     p      — Pitch or position values (vector of length K).
%     w      — Weights (vector of length K, or empty for all ones).
%     period — Period of the circular domain.
%     x      — (Optional) Query points (vector). Default: 0:period-1.
%     sigma  — (Optional) Positional jitter standard deviation
%              (non-negative scalar; default 0).
%
%   Outputs:
%     y         — Mean projected centroid values (row vector).
%     centMag   — Centroid magnitude scaled by alpha_1 (scalar in
%                 [0, 1]).
%     centPhase — Centroid phase in user units, unchanged from
%                 the deterministic case (scalar in [0, period)).
%
%   Examples:
%     % Projected centroid of a major triad (deterministic)
%     y = projCentroid([0, 4, 7], [], 12);
%
%     % Same triad, expected projection under sigma = 0.25 semitone
%     y = projCentroid([0, 4, 7], [], 12, 0:11, 0.25);
%
%   References:
%     Milne, A. J., Dean, R. T., & Bulger, D. (2023). The effects of
%       rhythmic structure on tapping accuracy. Attention, Perception,
%       & Psychophysics, 85, 2673-2699.
%
%   See also meanOffset, edges, dftCircular, balanceCircular.

% === Input validation ===

if nargin < 4 || isempty(x)
    x = 0:period-1;
end
x = x(:).';

if nargin < 5 || isempty(sigma)
    sigma = 0;
end
if sigma < 0
    error('projCentroid:negativeSigma', 'sigma must be non-negative.');
end

% === Compute deterministic DFT and damp by alpha_1 if sigma > 0 ===

F = dftCircular(p, w, period);
F0 = F(1);                              % MATLAB 1-based: F(1) is k = 0

if sigma > 0
    alpha1 = exp(-2 * pi^2 * sigma^2 / period^2);
    F0 = alpha1 * F0;                   % E[F̃(0)] = alpha_1 * F(0)
end

centMag = abs(F0);
centPhaseRad = mod(angle(F0), 2 * pi);

% === Project onto query points ===

queryAngles = 2 * pi * x / period;
y = centMag * cos(centPhaseRad - queryAngles);

% === Convert phase to user units ===

centPhase = centPhaseRad * period / (2 * pi);

end
