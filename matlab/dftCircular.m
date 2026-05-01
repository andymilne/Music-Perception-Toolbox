function [F, mag] = dftCircular(p, w, period)
%DFTCIRCULAR Discrete Fourier transform of a weighted circular multiset.
%
%   [F, mag] = dftCircular(p, w, period):
%
%   Computes the DFT of a weighted multiset of K points distributed
%   around a circle of circumference 'period' (p represents pitches or
%   positions).
%
%   The procedure is:
%     1. Sort p ascending (reordering w to match).
%     2. Map each element to the unit circle and scale by its weight:
%          z(j) = w(j) * exp(2*pi*1i*p(j)/period).
%     3. Compute the DFT of z, normalized by the sum of weights:
%          F(k) = (1/sum(w)) * sum_j z(j) * exp(-2*pi*1i * (j-1) * k / K)
%        for k = 0, 1, ..., K-1.
%        (For uniform weights, sum(w) = K.)
%
%   The magnitudes of the coefficients have well-established music-
%   theoretical interpretations (see balanceCircular and evennessCircular):
%     |F(0)|: imbalance — distance of the centre of gravity from the
%             origin. When |F(0)| = 0, the multiset is perfectly balanced.
%     |F(1)|: evenness — closeness to a maximally even (equal-step)
%             distribution. When |F(1)| = 1, the multiset is maximally
%   Higher coefficients capture additional distributional properties.
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
%     p      — Pitch or position values (vector of length K).
%              Values are interpreted modulo 'period'. The function sorts
%              p internally; the caller does not need to pre-sort.
%     w      — Weights (vector of length K, or empty for all ones). If
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
                                           % (sum(w) = K for all-ones weights)

% === Magnitudes ===

mag = abs(F);

end
