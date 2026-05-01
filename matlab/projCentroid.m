function [y, centMag, centPhase] = projCentroid(p, w, period, x)
%PROJCENTROID Projected centroid of a weighted circular multiset.
%
%   y = projCentroid(p, w, period) computes the projection of the
%   circular centroid (centre of gravity) onto each angular position in
%   the sequence 0, 1, ..., period-1. The centroid is the k = 0 Fourier
%   coefficient of the multiset (see dftCircular); its projection onto an
%   angle theta is:
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
%   [y, centMag, centPhase] = projCentroid(...) also returns the
%   centroid magnitude (|F(0)|, the degree of imbalance) and its phase
%   angle (in the same units as p, not radians).
%
%   The centroid magnitude is related to balance: balance = 1 - centMag
%   (see balanceCircular). A centMag of 0 means the multiset is perfectly
%   balanced and all projections are zero; a centMag near 1 means the
%   multiset is maximally unbalanced.
%
%   Inputs:
%     p      — Pitch or position values (vector of length K).
%     w      — Weights (vector of length K, or empty for all ones).
%     period — Period of the circular domain.
%     x      — (Optional) Query points at which to evaluate the
%              projection (vector). Default: 0:period-1.
%
%   Outputs:
%     y         — Projected centroid values (row vector, same length
%                 as x or as 0:period-1).
%     centMag   — Centroid magnitude |F(0)| (scalar, range [0, 1]).
%     centPhase — Centroid phase in the units of p/period (scalar,
%                 range [0, period)).
%
%   Examples:
%     % Projected centroid of a major triad (12 chromatic positions)
%     y = projCentroid([0, 4, 7], [], 12);
%     bar(0:11, y);
%     xlabel('Pitch class'); ylabel('Projection');
%
%     % Fine grid in cents (0.1-cent resolution)
%     x = 0:0.1:1199.9;
%     [y, cm, cp] = projCentroid([0, 400, 700], [], 1200, x);
%     plot(x, y);
%     fprintf('Centroid magnitude = %.3f, phase = %.1f cents\n', cm, cp);
%
%     % Son clave rhythm (16-step cycle)
%     y = projCentroid([0, 3, 6, 10, 12], [], 16);
%     bar(0:15, y);
%     xlabel('Pulse'); ylabel('Projected centroid');
%
%   References:
%     Milne, A. J., Dean, R. T., & Bulger, D. (2023). The effects of
%       rhythmic structure on tapping accuracy. Attention, Perception,
%       & Psychophysics, 85, 2673-2699.
%       (Introduced this predictor as proj_cent — a pulse-level
%       generalization of the rhythm-level balance predictor.)
%
%   See also meanOffset, edges, dftCircular, balanceCircular.

% === Input validation ===

if nargin < 4 || isempty(x)
    x = 0:period-1;
end
x = x(:).';

% === Compute DFT via dftCircular ===

F = dftCircular(p, w, period);

% === Extract centroid (k = 0 coefficient) ===

F0 = F(1);  % MATLAB 1-based: F(1) is k = 0
centMag = abs(F0);
centPhaseRad = mod(angle(F0), 2 * pi);  % in [0, 2*pi)

% === Project onto query points ===

queryAngles = 2 * pi * x / period;
y = centMag * cos(centPhaseRad - queryAngles);

% === Convert phase to user units ===

centPhase = centPhaseRad * period / (2 * pi);

end