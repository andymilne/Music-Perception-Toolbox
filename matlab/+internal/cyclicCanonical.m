function [pBest, wBest, bestShift] = cyclicCanonical(pSorted, wSorted, hasWeights, period)
%CYCLICCANONICAL Lexicographically smallest rotation of a periodic set.
%   [pBest, wBest, bestShift] = internal.cyclicCanonical(pSorted, ...
%                                  wSorted, hasWeights, period)
%
%   For n pitches, tries all n rotations (subtract p(i), mod period,
%   re-sort with weights) and returns the lexicographically smallest
%   (pitch, weight) vector plus the shift that produced it.
%
%   Pitch values are rounded to 9 decimal places before lex
%   comparison, to absorb floating-point noise from mod-reduction.
%   9 decimals is below any musically-meaningful precision (1
%   attocent / 1 nanosecond) but well above typical FP roundoff.
%   Without this rounding, two transposition-equivalent multisets
%   with different FP error patterns can produce different canonical
%   forms, which breaks consumer-level dedup for non-integer pitch
%   data. ``bestShift`` is returned at full precision so callers
%   using it to apply to a paired set get exact arithmetic.
%
%   Inputs
%     pSorted    : 1-by-n sorted pitch row (reduced mod period if
%                  applicable).
%     wSorted    : 1-by-n weights aligned with pSorted, or [].
%     hasWeights : logical, whether wSorted is non-empty.
%     period     : period in cents.

    n = numel(pSorted);
    ROUND_DIGITS = 9;

    % Rotation 0: subtract the first element.
    pBest = round(pSorted(:)' - pSorted(1), ROUND_DIGITS);
    if hasWeights
        wBest = wSorted(:)';
    else
        wBest = [];
    end
    bestShift = pSorted(1);

    for rot = 2:n
        shifted = mod(pSorted - pSorted(rot), period);
        [shifted, si] = sort(shifted);
        shifted = round(shifted(:)', ROUND_DIGITS);

        cmp = internal.lexCompare(shifted, pBest);
        if cmp < 0
            pBest = shifted;
            bestShift = pSorted(rot);
            if hasWeights
                wBest = wSorted(si)';
            end
        elseif cmp == 0 && hasWeights
            wRot = wSorted(si)';
            if internal.lexCompare(wRot, wBest) < 0
                wBest = wRot;
                bestShift = pSorted(rot);
            end
        end
    end
end
