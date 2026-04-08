function [F, mag] = dftCircular(p, w, period)
%DFTCIRCULAR Discrete Fourier transform of a set of points on a circle.
%
%   [F, mag] = dftCircular(p, w, period):
%
%   Computes the DFT of a set of K points distributed around a circle of
%   circumference 'period'. This applies to pitch-class sets (where the
%   circle represents one octave or other period of pitch-class
%   equivalence), time-class sets (where the circle represents one
%   rhythmic cycle), or any other periodic domain.
%
%   The procedure is:
%     1. Sort p ascending (reordering w to match).
%     2. Map each element to the unit circle: z(j) = exp(2*pi*1i*p(j)/period).
%     3. Compute the DFT of the complex vector z, normalized by K:
%          F(k) = (1/K) * sum_j z(j) * exp(-2*pi*1i * (j-1) * k / K)
%        for k = 0, 1, ..., K-1.
%
%   The magnitudes of the coefficients have well-established music-
%   theoretical interpretations (see balanceCircular and evennessCircular):
%     |F(0)|: imbalance — distance of the centre of gravity from the
%             origin. When |F(0)| = 0, the set is perfectly balanced.
%     |F(1)|: evenness — closeness to a maximally even (equal-step)
%             distribution. When |F(1)| = 1, the set is maximally even.
%   Higher coefficients capture finer distributional properties (e.g.,
%   |F(2)| relates to dyadic structure, |F(3)| to triadic, etc.).
%
%   For further information, see:
%     Milne, A. J., Bulger, D., & Herff, S. A. (2017). Exploring the
%       space of perfectly balanced rhythms and scales. Journal of
%       Mathematics and Music, 11(2-3), 101-133.
%     Milne, A. J. & Herff, S. A. (2020). The perceptual relevance of
%       balance, evenness, and entropy in musical rhythms. Cognition,
%       203, 104233.
%
%   Inputs:
%     p      — Pitch-class (or time-class) values (vector of length K).
%              Values are interpreted modulo 'period'. The function sorts
%              p internally; the caller does not need to pre-sort.
%     w      — Weights (vector of length K, or empty for uniform). If
%              provided, each z(j) is scaled by w(j) before the DFT.
%     period — Period of the circular domain (e.g., 1200 for one octave
%              in cents, or the cycle length for rhythmic patterns).
%
%   Outputs:
%     F      — Complex Fourier coefficients (1 x K row vector).
%              F(1) is the k = 0 coefficient, F(2) is k = 1, etc.
%              (MATLAB's 1-based indexing: F(k+1) corresponds to the
%              k-th coefficient.)
%     mag    — Magnitudes |F(k)| for each k (1 x K row vector).
%
%   Examples:
%     % DFT of a 12-EDO diatonic scale (in cents)
%     [F, mag] = dftCircular([0, 200, 400, 500, 700, 900, 1100], [], 1200);
%     fprintf('Balance = %.3f, Evenness = %.3f\n', 1 - mag(1), mag(2));
%
%     % DFT of a rhythmic pattern (onsets in a 16-step cycle)
%     [F, mag] = dftCircular([0, 3, 6, 8, 10, 12, 14], [], 16);
%
%   See also balanceCircular, evennessCircular.

% === Input validation ===

p = p(:);
K = numel(p);

if isempty(w)
    w = ones(K, 1);
end
if isscalar(w)
    w = w * ones(K, 1);
end
w = w(:);

if numel(w) ~= K
    error('w must have the same number of entries as p (or be empty).');
end

% === Sort by pitch class ===

[p, sortIdx] = sort(p);
w = w(sortIdx);

% === Map to unit circle and compute DFT ===

z = w .* exp(2 * pi * 1i * p / period);  % K x 1, weighted
F = fft(z).' / sum(w);                    % 1 x K row vector
                                           % (sum(w) = K for uniform weights)

% === Magnitudes ===

mag = abs(F);

end
