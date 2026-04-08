function [c, nc] = coherence(p, period, nvArgs)
%COHERENCE Coherence quotient of a pitch-class or time-class set.
%
%   c = coherence(p, period) returns the coherence quotient of the set
%   of pitch classes (or time classes) p within an equal division of
%   size period. The coherence quotient (Carey 2002, 2007) is 1 minus
%   the ratio of coherence failures to the maximum possible number for
%   a set of K events:
%
%     c = 1 - nc / maxNC
%
%   A coherence failure occurs when a pair of events with a larger
%   generic span (number of scale steps) than another pair does not
%   have a strictly greater specific size (number of chromatic steps).
%   This corresponds to Rothenberg's (1978) strict propriety.
%
%   c = coherence(p, period, 'strict', false) uses the non-strict
%   criterion: a failure occurs only when the larger generic span has
%   a strictly smaller specific size (not merely equal). This
%   corresponds to Rothenberg's propriety.
%
%   [c, nc] = coherence(...) also returns the number of coherence
%   failures.
%
%   Inputs:
%     p      — Pitch-class (or time-class) positions (vector of
%              length K). Non-negative values less than period.
%              May be integers (within an equal division) or floats
%              (e.g. cents values of a just-intonation scale).
%              Duplicates (modulo period) are not allowed.
%     period — Period of the circular domain (positive number, e.g.,
%              12 for 12-EDO, 1200 for cents, 16 for a 16-step
%              rhythmic cycle).
%
%   Name-Value Arguments:
%     'strict' — Logical (default: true). If true, a coherence
%                failure occurs when a larger generic span does not
%                have a strictly greater specific size (Rothenberg's
%                strict propriety). If false, a failure occurs only
%                when it has a strictly smaller specific size
%                (Rothenberg's propriety).
%
%   Outputs:
%     c      — Coherence quotient (scalar, range [0, 1]). A value of
%              1 indicates perfect coherence (no failures).
%     nc     — Number of coherence failures (non-negative integer).
%
%   Examples:
%     % Diatonic scale in 12-EDO (strictly proper)
%     c = coherence([0, 2, 4, 5, 7, 9, 11], 12)
%
%     % Whole-tone scale in 12-EDO
%     c = coherence([0, 2, 4, 6, 8, 10], 12)
%
%     % Non-strict coherence (propriety)
%     c = coherence([0, 2, 4, 5, 7, 9, 11], 12, 'strict', false)
%
%     % Also return the failure count
%     [c, nc] = coherence([0, 2, 4, 5, 7, 9, 11], 12)
%
%     % Son clave rhythm (16-step cycle)
%     c = coherence([0, 3, 6, 10, 12], 16)
%
%     % Just-intonation diatonic scale (cents, period = 1200)
%     c = coherence([0, 203.91, 386.31, 498.04, 701.96, 884.36, 1088.27], 1200)
%
%     % Porcupine[7] in 22-EDO (cents)
%     c = coherence([0, 4, 7, 10, 13, 16, 19] * 1200/22, 1200)
%
%   References:
%     Balzano, G. J. (1982). The pitch set as a level of description
%       for studying musical pitch perception. In M. Clynes (Ed.),
%       Music, Mind, and Brain (pp. 321-351). Plenum.
%     Carey, N. (2002). On coherence and sameness, and the evaluation
%       of scale candidacy claims. Journal of Music Theory, 46(1/2),
%       1-56.
%     Carey, N. (2007). Coherence and sameness in well-formed and
%       pairwise well-formed scales. Journal of Mathematics and Music,
%       1(2), 79-98.
%     Milne, A. J., Dean, R. T., & Bulger, D. (2023). The effects of
%       rhythmic structure on tapping accuracy. Attention, Perception,
%       & Psychophysics, 85, 2673-2699.
%       (Applied the coherence quotient to rhythmic patterns.)
%     Rothenberg, D. (1978). A model for pattern perception with
%       musical applications. Part I. Mathematical Systems Theory,
%       11, 199-234.
%
%   See also sameness.

    arguments
        p (:,1) {mustBeNumeric, mustBeNonnegative}
        period (1,1) {mustBePositive}
        nvArgs.strict (1,1) logical = true
    end

    % === Input validation ===

    p = sort(mod(p, period));
    K = numel(p);

    if numel(unique(p)) ~= K
        error('coherence:duplicates', ...
              'p must not contain duplicate pitch classes (modulo period).');
    end

    if K < 2
        error('coherence:tooFewEvents', ...
              'At least 2 events are required (got %d).', K);
    end

    % === Build the interval-size table ===
    % sizeSpan(g, k) is the specific size (in chromatic steps) of the
    % interval of generic span g starting from the k-th scale degree.
    % Generic span g = 1 is seconds, g = 2 is thirds, etc.

    sizeSpan = zeros(K - 1, K);
    for g = 1:K-1
        sizeSpan(g, :) = mod(p(mod((1:K) + g - 1, K) + 1)' - p', period);
    end

    % === Count coherence failures ===
    % A coherence failure occurs when a pair of events with generic
    % span g2 > g1 does not have a greater (strict) or has a smaller
    % (non-strict) specific size than a pair with generic span g1.

    nc = 0;
    for g2 = 2:K-1
        sizes2 = sizeSpan(g2, :);
        for g1 = 1:g2-1
            sizes1 = sizeSpan(g1, :);
            % All pairwise differences: each size2 minus each size1
            diffs = sizes2(:) - sizes1(:)';
            if nvArgs.strict
                nc = nc + sum(diffs(:) <= 0);
            else
                nc = nc + sum(diffs(:) < 0);
            end
        end
    end

    % === Coherence quotient ===

    maxNC = K * (K - 1) * (K - 2) * (3*K - 5) / 24;
    c = 1 - nc / maxNC;

end