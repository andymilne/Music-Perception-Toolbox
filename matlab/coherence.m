function [c, nc] = coherence(p, period, sigma, nvArgs)
%COHERENCE Coherence quotient of a circular set, optionally smoothed.
%
%   c = coherence(p, period) returns the coherence quotient of the set
%   of pitches or positions p within an equal division of size period.
%   The coherence quotient (Carey 2002, 2007) is 1 minus the ratio of
%   coherence failures to the maximum possible number for a set of K
%   elements:
%
%     c = 1 - nc / maxNC
%
%   A coherence failure occurs when a pair of events with a larger
%   generic span (number of scale steps) than another pair does not
%   have a strictly greater specific size (number of chromatic steps).
%   This corresponds to Rothenberg's (1978) strict propriety.
%
%   c = coherence(p, period, sigma) returns a soft (smoothed)
%   coherence quotient under positional uncertainty sigma. At sigma = 0
%   the discrete v2 count is recovered exactly (honouring the strict
%   flag). For sigma > 0, each indicator [d2 <= d1] is replaced by
%   the Gaussian-CDF probability P(D2 <= D1) under jitter, which
%   smoothly interpolates between 0 and 1 as the means cross.
%
%   p and period may be non-integer when sigma > 0.
%
%   Name-Value Arguments:
%     'strict' — Logical (default: true). Controls tie handling at
%                sigma = 0 only. If true, ties (a larger generic span
%                with equal specific size) count as failures
%                (Rothenberg's strict propriety). If false, only
%                strictly smaller specific sizes count (Rothenberg's
%                propriety). At sigma > 0 ties have measure zero and
%                this flag has no effect; the soft path uses
%                P(D2 <= D1), which assigns 0.5 to ties as a natural
%                limiting case.
%
%     'sigmaSpace' — How sigma is interpreted. 'position' (default)
%                    treats sigma as positional uncertainty on each
%                    p_k, propagated through index sharing among
%                    interval pairs. 'interval' treats sigma as
%                    independent uncertainty per derived interval
%                    (V = 2 sigma^2 uniformly).
%
%                    The position model captures shared-endpoint
%                    correlations between intervals: per-pair variance
%                    V ranges from 2 sigma^2 (one shared endpoint with
%                    cancelling signs) through 4 sigma^2 (disjoint
%                    endpoints) up to 8 sigma^2 (the tritone-style
%                    configuration where both endpoints are shared
%                    with reinforcing signs). The famous diatonic
%                    tritone — a fourth and a fifth that share both F
%                    and B as endpoints — falls in the 8 sigma^2 case
%                    and produces P = 0.5 exactly under noise of any
%                    magnitude.
%
%                    At sigma = 0 the two models coincide; the flag
%                    has no effect.
%
%   [c, nc] = coherence(...) also returns the (soft or hard) failure
%   count.
%
%   Inputs:
%     p      — Pitch or position values (vector of length K).
%              Non-negative; values less than period.
%              May be float when sigma > 0.
%              Duplicates (modulo period) are not allowed.
%     period — Period of the circular domain (positive number, e.g.,
%              12 for 12-EDO, 1200 for cents, 16 for a 16-step
%              rhythmic cycle).
%     sigma  — Positional or interval uncertainty (non-negative
%              scalar; default 0). In the same units as p and period.
%
%   Outputs:
%     c      — Coherence quotient (scalar). [0, 1] when sigma = 0;
%              may go below 0 at large sigma when soft failures exceed
%              maxNC.
%     nc     — Number of coherence failures (non-negative scalar;
%              integer when sigma = 0).
%
%   Examples:
%     % Diatonic in 12-EDO (one strict failure: the tritone)
%     c = coherence([0, 2, 4, 5, 7, 9, 11], 12)
%
%     % Soft coherence, position-jitter, sigma = 0.25 semitone
%     c = coherence([0, 2, 4, 5, 7, 9, 11], 12, 0.25)
%
%     % Same query with the interval-jitter model
%     c = coherence([0, 2, 4, 5, 7, 9, 11], 12, 0.25, ...
%                   'sigmaSpace', 'interval')
%
%     % Just-intonation diatonic in cents
%     pJI = [0, 203.91, 386.31, 498.04, 701.96, 884.36, 1088.27];
%     c = coherence(pJI, 1200, 25)
%
%     % Son clave rhythm (16-step cycle), sigma = 1/2 pulse
%     c = coherence([0, 3, 6, 10, 12], 16, 0.5)
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
%     Rothenberg, D. (1978). A model for pattern perception with
%       musical applications. Part I. Mathematical Systems Theory,
%       11, 199-234.
%
%   See also sameness.

    arguments
        p (:,1) {mustBeNumeric, mustBeNonnegative}
        period (1,1) {mustBePositive}
        sigma (1,1) {mustBeNumeric, mustBeNonnegative} = 0
        nvArgs.strict (1,1) logical = true
        nvArgs.sigmaSpace (1,:) char ...
            {mustBeMember(nvArgs.sigmaSpace, {'position', 'interval'})} ...
            = 'position'
    end

    p = sort(mod(p, period));
    K = numel(p);

    if numel(unique(p)) ~= K
        error('coherence:duplicates', ...
              'p must not contain duplicate values (modulo period).');
    end
    if K < 2
        error('coherence:tooFewEvents', ...
              'At least 2 events are required (got %d).', K);
    end

    % === Build interval-size table with index provenance ===

    sizeSpan = zeros(K - 1, K);
    srcFrom = zeros(K - 1, K);
    srcTo = zeros(K - 1, K);
    for g = 1:K-1
        toIdx = mod((1:K) + g - 1, K) + 1;
        sizeSpan(g, :) = mod(p(toIdx)' - p', period);
        srcFrom(g, :) = 1:K;
        srcTo(g, :) = toIdx;
    end

    if sigma == 0
        % --- Discrete v2 count (preserves strict flag exactly) ---
        nc = 0;
        for g2 = 2:K-1
            sizes2 = sizeSpan(g2, :);
            for g1 = 1:g2-1
                sizes1 = sizeSpan(g1, :);
                diffs = sizes2(:) - sizes1(:)';
                if nvArgs.strict
                    nc = nc + sum(diffs(:) <= 0);
                else
                    nc = nc + sum(diffs(:) < 0);
                end
            end
        end
    else
        % --- Soft count under sigma jitter ---
        % nc_soft = sum over (g2, i, g1, j) with g2 > g1 of
        %   Phi( - (sizeSpan(g2, i) - sizeSpan(g1, j)) / sqrt(V) )
        % where V is the variance of D2 - D1 under jitter, set per
        % pair according to sigmaSpace.

        usePosition = strcmp(nvArgs.sigmaSpace, 'position');
        nc = 0;
        for g2 = 2:K-1
            for g1 = 1:g2-1
                for i = 1:K
                    for j = 1:K
                        Delta = sizeSpan(g2, i) - sizeSpan(g1, j);

                        if usePosition
                            V = positionVariance( ...
                                    [srcTo(g2, i), srcFrom(g2, i), ...
                                     srcTo(g1, j), srcFrom(g1, j)], ...
                                    [+1, -1, -1, +1], sigma);
                        else
                            V = 2 * sigma^2;
                        end

                        if V == 0
                            % Deterministic limit; tie -> 0.5
                            if Delta < 0
                                nc = nc + 1;
                            elseif Delta == 0
                                nc = nc + 0.5;
                            end
                        else
                            % Standard normal CDF via erfc, avoiding
                            % the Stats Toolbox dependency that
                            % normcdf would impose:
                            %   Phi(z) = 0.5 * erfc(-z / sqrt(2))
                            nc = nc + 0.5 * erfc(Delta / sqrt(2 * V));
                        end
                    end
                end
            end
        end
    end

    maxNC = K * (K - 1) * (K - 2) * (3*K - 5) / 24;
    c = 1 - nc / maxNC;
end
