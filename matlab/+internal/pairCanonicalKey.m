function [pAc, wAc, pBc, wBc] = pairCanonicalKey(pAv, wAv, pBv, wBv, ...
                                                 isRel, isPer, period, nDec)
%PAIRCANONICALKEY Canonical form of a paired weighted multiset (A, B).
%
%   [pAc, wAc, pBc, wBc] = internal.pairCanonicalKey(pAv, wAv, pBv, wBv, ...
%                                                    isRel, isPer, period, nDec)
%
%   The cosine similarity of two densities is invariant under certain joint
%   transformations of the pair, so structurally equivalent pairs share a
%   canonical form and need be evaluated only once. This returns that form;
%   the caller turns it into whatever key its deduplication needs.
%
%   The symmetry exploited depends on the mode:
%
%     Relative ([rel] = 1). Each side may be transposed independently, so
%     each is canonicalized on its own via internal.canonicalizeSet.
%
%     Absolute ([rel] = 0). Only a joint co-transposition preserves the
%     similarity, since cosSimExpTens(A + c, B + c) = cosSimExpTens(A, B):
%     the raw tuple differences cancel. A therefore determines a shift and
%     B inherits it. Under octave equivalence ([per] = 1) A is reduced to
%     its cyclic canonical form, the lexicographically smallest rotation,
%     and B is shifted by the corresponding amount modulo the period;
%     otherwise A is translated so its smallest value sits at 0 and B is
%     translated by the same amount.
%
%   Values are re-rounded after canonicalization when nDec is non-empty:
%   the mod-reduction and subtraction above introduce floating-point noise
%   that would otherwise separate keys which ought to coincide.
%
%   Inputs:
%     pAv, pBv  - Row vectors of values for A and B, already NaN-stripped.
%     wAv, wBv  - Matching weights, or [] for uniform weights.
%     isRel     - Logical. Relative ([rel]) reading.
%     isPer     - Logical. Periodic ([per]) reading.
%     period    - Numeric scalar. The period, used when isPer is true.
%     nDec      - Decimal places for post-canonicalization rounding, or []
%                 for none.
%
%   Outputs:
%     pAc, pBc  - Canonical value row vectors.
%     wAc, wBc  - Canonical weight row vectors, [] where the input was [].
%
%   This is the MATLAB twin of the Python _pair_canonical_key. The Python
%   side additionally returns hashable tuple keys; here the caller packs
%   fixed-width numeric rows instead, because MATLAB deduplicates with
%   unique(..., 'rows') where Python uses dictionary keys.
%
%   See also internal.canonicalizeSet, internal.cyclicCanonical.

    if isRel
        % Relative: independent canonicalization
        [pAc, wAc] = internal.canonicalizeSet(pAv, wAv, isRel, isPer, period);
        [pBc, wBc] = internal.canonicalizeSet(pBv, wBv, isRel, isPer, period);
    else
        % Absolute: joint co-transposition normalization.
        % cosSimExpTens(A-c, B-c) = cosSimExpTens(A, B) because the
        % raw tuple differences cancel. Find A's canonical form and
        % apply the same shift to B.

        hasWA = ~isempty(wAv);
        hasWB = ~isempty(wBv);

        % Canonicalize A
        [pAs, siA] = sort(pAv);
        if hasWA, wAs = wAv(siA); else, wAs = []; end

        if isPer
            pAs = mod(pAs, period);
            [pAs, siA2] = sort(pAs);
            if hasWA, wAs = wAs(siA2); end
            % Cyclic canonical form — collapses all rotations
            [pAc, wAc, shift] = internal.cyclicCanonical(pAs, wAs, hasWA, period);
        else
            shift = pAs(1);
            pAc = pAs(:)' - shift;
            if hasWA, wAc = wAs(:)'; else, wAc = []; end
        end

        % Apply the same shift to B
        [pBs, siB] = sort(pBv);
        if hasWB, wBs = wBv(siB); else, wBs = []; end

        if isPer
            pBshifted = mod(pBs - shift, period);
            [pBshifted, siB2] = sort(pBshifted);
            pBc = pBshifted(:)';
            if hasWB, wBc = wBs(siB2)'; else, wBc = []; end
        else
            pBc = pBs(:)' - shift;
            if hasWB, wBc = wBs(:)'; else, wBc = []; end
        end
    end

    % Re-round after canonicalization to collapse floating-point
    % noise introduced by mod-reduction and subtraction.
    if ~isempty(nDec)
        pAc = round(pAc, nDec);
        pBc = round(pBc, nDec);
        if ~isempty(wAc), wAc = round(wAc, nDec); end
        if ~isempty(wBc), wBc = round(wBc, nDec); end
    end
end
