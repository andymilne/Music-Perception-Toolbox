function [sq, nDiff] = sameness(p, period)
%SAMENESS Sameness quotient of a circular set.
%
%   sq = sameness(p, period) returns the sameness quotient of the set
%   of pitches or positions p within an equal division of
%   size period. The sameness quotient (Carey 2002, 2007) is 1 minus
%   the ratio of the number of ambiguities to the maximum possible
%   number for a set of K elements:
%
%     sq = 1 - nDiff / maxDiff
%
%   An ambiguity occurs when two different generic spans (numbers of
%   scale steps) share the same specific size (number of chromatic
%   steps). In a scale with no ambiguities (sq = 1), each specific
%   size belongs to exactly one generic span — hearing an interval's
%   chromatic size uniquely identifies its scale-step distance.
%
%   [sq, nDiff] = sameness(...) also returns the number of
%   ambiguities.
%
%   Inputs:
%     p      — Pitch or position values (vector of
%              length K). Must be non-negative integers less than
%              period. Duplicates (modulo period) are not allowed.
%     period — Size of the equal division (positive integer, e.g.,
%              12 for the chromatic scale, 16 for a 16-step rhythmic
%              cycle).
%
%   Outputs:
%     sq     — Sameness quotient (scalar, range [0, 1]). A value of
%              1 indicates perfect sameness (no ambiguities).
%     nDiff  — Number of ambiguities (non-negative integer).
%
%   Examples:
%     % Diatonic scale in 12-EDO
%     sq = sameness([0, 2, 4, 5, 7, 9, 11], 12)
%
%     % Whole-tone scale in 12-EDO (maximally even, high sameness)
%     sq = sameness([0, 2, 4, 6, 8, 10], 12)
%
%     % Also return the ambiguity count
%     [sq, nDiff] = sameness([0, 2, 4, 5, 7, 9, 11], 12)
%
%     % Son clave rhythm (16-step cycle)
%     sq = sameness([0, 3, 6, 10, 12], 16)
%
%   References:
%     Carey, N. (2002). On coherence and sameness, and the evaluation
%       of scale candidacy claims. Journal of Music Theory, 46(1/2),
%       1-56.
%     Carey, N. (2007). Coherence and sameness in well-formed and
%       pairwise well-formed scales. Journal of Mathematics and Music,
%       1(2), 79-98.
%     Milne, A. J., Dean, R. T., & Bulger, D. (2023). The effects of
%       rhythmic structure on tapping accuracy. Attention, Perception,
%       & Psychophysics, 85, 2673-2699.
%       (Applied the sameness quotient to rhythmic patterns.)
%
%   See also coherence.

    arguments
        p (:,1) {mustBeNonnegative, mustBeInteger}
        period (1,1) {mustBePositive, mustBeInteger}
    end

    % === Input validation ===

    p = sort(mod(p, period));
    K = numel(p);

    if numel(unique(p)) ~= K
        error('sameness:duplicates', ...
              'p must not contain duplicate values (modulo period).');
    end

    if K < 2
        error('sameness:tooFewEvents', ...
              'At least 2 events are required (got %d).', K);
    end

    % === Build the interval-size table ===
    % sizeSpan(g, k) is the specific size (in chromatic steps) of the
    % interval of generic span g starting from the k-th scale degree.

    sizeSpan = zeros(K - 1, K);
    for g = 1:K-1
        sizeSpan(g, :) = mod(p(mod((1:K) + g - 1, K) + 1)' - p', period);
    end

    % === Count ambiguities ===
    % An ambiguity is an ordered pair of generic spans (g1, g2) with
    % g1 ~= g2 that share the same specific size. We count by building
    % a histogram of specific sizes for each generic span, then
    % counting cross-span collisions.
    %
    % For each generic span g, sizeCounts(g, s+1) is the number of
    % intervals of generic span g with specific size s.

    sizeCounts = zeros(K - 1, period);
    for g = 1:K-1
        for k = 1:K
            s = sizeSpan(g, k);
            sizeCounts(g, s + 1) = sizeCounts(g, s + 1) + 1;
        end
    end

    % The total number of ordered pairs sharing the same specific size
    % across all generic spans is:
    %   sum over s of: (sum_g sizeCounts(g,s))^2 - sum_g sizeCounts(g,s)^2
    % divided by 2 (to get unordered pairs of generic spans).

    colTotals = sum(sizeCounts, 1);           % 1 x period
    nDiff = (sum(colTotals.^2) - sum(sizeCounts(:).^2)) / 2;

    % === Sameness quotient ===

    maxDiff = K * (K - 1)^2 / 2;
    sq = 1 - nDiff / maxDiff;

end