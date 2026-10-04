function [pCan, wCan] = canonicalizeSet(p, w, rel, per, period)
%CANONICALIZESET Canonical form of a pitch/weight set under per/rel.
%   [pCan, wCan] = internal.canonicalizeSet(p, w, rel, per, period)
%
%   Sorts pitches (aligning weights), reduces modulo period if per,
%   removes transposition if rel, and applies the cyclic canonical
%   form when both flags are true. Two weighted multisets that differ
%   only by permutation, transposition (if rel), or rotation around
%   the period (if per && rel) produce identical (pCan, wCan)
%   outputs.
%
%   The cyclic canonical form is valid because simMaet (and the
%   other downstream consumers) wrap pairwise differences in the rel
%   quadratic form when per is true, restoring exact
%   transposition-modulo-period invariance on the circle.
%
%   Used by:
%     - simMaet's batched-raw mode (paired-set dedup with shift-locking)
%     - The harmony-batched dispatchers (templateHarmonicity,
%       virtualPitches, spectralEntropy) for chord-side dedup.
%
%   Inputs
%     p        : numeric vector, pitch values (any ordering).
%     w        : numeric vector, weights (same length as p), or [].
%     rel      : logical, whether the consumer is in relative mode.
%     per      : logical, whether the consumer is in periodic mode.
%     period   : numeric scalar, period in cents (used only if per).
%
%   Outputs
%     pCan     : 1-by-n row vector, canonical pitches.
%     wCan     : 1-by-n row vector of canonical weights, or [] if w
%                was empty.

    hasWeights = ~isempty(w);

    % Sort p-values and align weights.
    [p, si] = sort(p);
    if hasWeights
        w = w(si);
    end

    % Reduce modulo period (only meaningful when per).
    if per
        p = mod(p, period);
        [p, si] = sort(p);
        if hasWeights
            w = w(si);
        end
    end

    % Remove transposition.
    if rel
        if per
            % Cyclic canonical form: the lexicographically smallest
            % rotation captures all transposition-modulo-period
            % equivalences.
            [pCan, wCan, ~] = internal.cyclicCanonical(p, w, hasWeights, period);
        else
            % Non-periodic relative: subtract minimum (sort order
            % preserved).
            p = p - p(1);
            pCan = p(:)';
            if hasWeights
                wCan = w(:)';
            else
                wCan = [];
            end
        end
    else
        pCan = p(:)';
        if hasWeights
            wCan = w(:)';
        else
            wCan = [];
        end
    end
end
