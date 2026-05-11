function [pCan, wCan] = canonicalizeSet(p, w, isRel, isPer, period)
%CANONICALIZESET Canonical form of a pitch/weight set under isPer/isRel.
%   [pCan, wCan] = internal.canonicalizeSet(p, w, isRel, isPer, period)
%
%   Sorts pitches (aligning weights), reduces modulo period if isPer,
%   removes transposition if isRel, and applies the cyclic canonical
%   form when both flags are true. Two weighted multisets that differ
%   only by permutation, transposition (if isRel), or rotation around
%   the period (if isPer && isRel) produce identical (pCan, wCan)
%   outputs.
%
%   The cyclic canonical form is valid because cosSimExpTens (and the
%   other downstream consumers) wrap pairwise differences in the isRel
%   quadratic form when isPer is true, restoring exact
%   transposition-modulo-period invariance on the circle.
%
%   Used by:
%     - batchCosSimExpTens (paired-set dedup with shift-locking)
%     - The harmony-batched dispatchers (templateHarmonicity,
%       virtualPitches, spectralEntropy) for chord-side dedup.
%
%   Inputs
%     p        : numeric vector, pitch values (any ordering).
%     w        : numeric vector, weights (same length as p), or [].
%     isRel    : logical, whether the consumer is in relative mode.
%     isPer    : logical, whether the consumer is in periodic mode.
%     period   : numeric scalar, period in cents (used only if isPer).
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

    % Reduce modulo period (only meaningful when isPer).
    if isPer
        p = mod(p, period);
        [p, si] = sort(p);
        if hasWeights
            w = w(si);
        end
    end

    % Remove transposition.
    if isRel
        if isPer
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
