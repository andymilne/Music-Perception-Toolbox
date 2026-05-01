function [sq, nDiff] = sameness(p, period, sigma, nvArgs)
%SAMENESS Sameness quotient of a circular set, optionally smoothed.
%
%   sq = sameness(p, period) returns the sameness quotient of the set
%   of pitches or positions p within an equal division of size period.
%   The sameness quotient (Carey 2002, 2007) is 1 minus the ratio of
%   the number of ambiguities to the maximum possible number for a set
%   of K elements:
%
%     sq = 1 - nDiff / maxDiff
%
%   An ambiguity occurs when two different generic spans (numbers of
%   scale steps) share the same specific size (number of chromatic
%   steps). In a scale with no ambiguities (sq = 1), each specific
%   size belongs to exactly one generic span — hearing an interval's
%   chromatic size uniquely identifies its scale-step distance.
%
%   sq = sameness(p, period, sigma) returns a soft (smoothed)
%   sameness quotient under positional uncertainty sigma. At sigma = 0
%   the discrete v2 count is recovered exactly. For sigma > 0, the
%   strict equality test [d1 == d2] is replaced by a Gaussian match
%   kernel exp(-(d1 - d2)^2 / (2 V)), where V is the variance of the
%   difference-of-differences under the chosen jitter model. This
%   kernel equals 1 when intervals coincide, decays smoothly as the
%   gap grows, and recovers the discrete indicator at sigma = 0.
%
%   p and period may be non-integer when sigma > 0. The integer
%   restriction is required only for the sigma = 0 histogram-based
%   discrete count.
%
%   Name-Value Arguments:
%     'sigmaSpace' — How sigma is interpreted. 'position' (default)
%                    treats sigma as positional uncertainty on each
%                    p_k, propagated through index sharing among
%                    interval pairs. 'interval' treats sigma as
%                    independent uncertainty per derived interval,
%                    ignoring shared-position correlations.
%
%                    For sameness, the position model is more
%                    defensible because intervals are derived from a
%                    common position set, and adjacent or wrapping
%                    intervals share endpoints. Per-pair variance V
%                    therefore ranges from 2 sigma^2 (one shared
%                    endpoint, opposite signs) through 4 sigma^2
%                    (disjoint endpoints) up to 8 sigma^2 (the
%                    tritone-style configuration where both endpoints
%                    are shared with same-sign contributions).
%
%                    The interval model uses V = 2 sigma^2 uniformly
%                    and corresponds to treating each interval as an
%                    independent draw from a per-interval density.
%                    This is the natural reading of sigma when the
%                    soft sameness is computed as the self inner
%                    product of a MAET density built from
%                    differenceEvents output.
%
%                    At sigma = 0 the two models coincide; the flag
%                    has no effect.
%
%   [sq, nDiff] = sameness(...) also returns the (soft or hard)
%   ambiguity count.
%
%   Inputs:
%     p      — Pitch or position values (vector of length K).
%              Non-negative; values less than period.
%              Must be integer if sigma = 0; may be float if sigma > 0.
%              Duplicates (modulo period) are not allowed.
%     period — Period of the circular domain (positive number, e.g.,
%              12 for 12-EDO, 1200 for cents, 16 for a 16-step
%              rhythmic cycle). Must be integer if sigma = 0.
%     sigma  — Positional or interval uncertainty (non-negative
%              scalar; default 0). In the same units as p and period.
%
%   Outputs:
%     sq     — Sameness quotient (scalar). [0, 1] when sigma = 0; may
%              go below 0 at large sigma when the soft count exceeds
%              maxDiff.
%     nDiff  — Number of ambiguities (non-negative scalar; integer
%              when sigma = 0).
%
%   Examples:
%     % Diatonic scale in 12-EDO (v2 hard count)
%     sq = sameness([0, 2, 4, 5, 7, 9, 11], 12)
%
%     % Soft sameness, position-jitter, sigma = 0.25 semitone
%     sq = sameness([0, 2, 4, 5, 7, 9, 11], 12, 0.25)
%
%     % Same query with the interval-jitter model
%     sq = sameness([0, 2, 4, 5, 7, 9, 11], 12, 0.25, ...
%                   'sigmaSpace', 'interval')
%
%     % Just-intonation diatonic in cents (sigma > 0 required, since
%     % positions are non-integer)
%     pJI = [0, 203.91, 386.31, 498.04, 701.96, 884.36, 1088.27];
%     sq = sameness(pJI, 1200, 25)
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
%
%   See also coherence, nTupleEntropy.

    arguments
        p (:,1) {mustBeNumeric, mustBeNonnegative}
        period (1,1) {mustBePositive}
        sigma (1,1) {mustBeNumeric, mustBeNonnegative} = 0
        nvArgs.sigmaSpace (1,:) char ...
            {mustBeMember(nvArgs.sigmaSpace, {'position', 'interval'})} ...
            = 'position'
    end

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

    % === Build interval-size table with index provenance ===
    % sizeSpan(g, k) is the specific size of the interval covering g
    % scale steps starting at scale degree k. srcFrom(g, k) and
    % srcTo(g, k) are the position indices of its endpoints (1-based).

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
        % --- Discrete v2 count ---
        if abs(period - round(period)) > 0
            error('sameness:nonIntegerPeriod', ...
                  ['For sigma = 0, period must be integer ' ...
                   '(got %g). Use sigma > 0 for non-integer periods.'], ...
                  period);
        end
        if any(abs(p - round(p)) > 0)
            error('sameness:nonIntegerPositions', ...
                  ['For sigma = 0, p must contain integers. ' ...
                   'Use sigma > 0 for non-integer positions.']);
        end
        period = round(period);
        sizeCounts = zeros(K - 1, period);
        for g = 1:K-1
            for k = 1:K
                s = round(sizeSpan(g, k));
                sizeCounts(g, s + 1) = sizeCounts(g, s + 1) + 1;
            end
        end
        colTotals = sum(sizeCounts, 1);
        nDiff = (sum(colTotals.^2) - sum(sizeCounts(:).^2)) / 2;
    else
        % --- Soft count under sigma jitter ---
        % nDiff_soft = (1/2) * sum over (g1, k), (g2, l) with g1 ~= g2 of
        %   exp( - (d1 - d2)^2 / (2 V) )
        % where d1 = sizeSpan(g1, k), d2 = sizeSpan(g2, l), and V is
        % set per pair according to sigmaSpace.

        usePosition = strcmp(nvArgs.sigmaSpace, 'position');
        nDiff = 0;
        for g1 = 1:K-1
            for g2 = 1:K-1
                if g1 == g2, continue; end
                for k = 1:K
                    for l = 1:K
                        % Straight difference between specific sizes.
                        % We do NOT wrap to the circle: sizes are
                        % treated as absolute magnitudes in [0, period),
                        % preserving the v2 semantic that size 0 and
                        % size near period are distinct intervals
                        % (e.g., a unison and a near-octave are not
                        % perceptually confusable).
                        dx = sizeSpan(g1, k) - sizeSpan(g2, l);

                        if usePosition
                            V = positionVariance( ...
                                    [srcTo(g1, k), srcFrom(g1, k), ...
                                     srcTo(g2, l), srcFrom(g2, l)], ...
                                    [+1, -1, -1, +1], sigma);
                        else
                            V = 2 * sigma^2;
                        end

                        if V == 0
                            % Deterministic comparison: equal => 1
                            if dx == 0, nDiff = nDiff + 1; end
                        else
                            nDiff = nDiff + exp(-dx^2 / (2 * V));
                        end
                    end
                end
            end
        end
        nDiff = nDiff / 2;   % unordered pairs of generic spans
    end

    maxDiff = K * (K - 1)^2 / 2;
    sq = 1 - nDiff / maxDiff;
end
