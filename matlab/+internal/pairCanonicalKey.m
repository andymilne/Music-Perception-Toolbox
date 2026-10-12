function [pAc, wAc, pBc, wBc] = pairCanonicalKey(pAv, wAv, pBv, wBv, ...
                                                 rel, per, period, nDec, roundDigits)
%PAIRCANONICALKEY Canonical form of a paired weighted multiset (A, B).
%
%   [pAc, wAc, pBc, wBc] = internal.pairCanonicalKey(pAv, wAv, pBv, wBv, ...
%                                                    rel, per, period, nDec)
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
%     similarity, since simMaet(A + c, B + c) = simMaet(A, B):
%     the raw tuple differences cancel. A therefore determines a shift and
%     B inherits it. Under octave equivalence ([per] = 1) A is reduced to
%     its cyclic canonical form, the lexicographically smallest rotation,
%     and B is shifted by the corresponding amount modulo the period
%     (where several rotations of a symmetric A give that form, by the
%     one giving B's lex-smallest form); otherwise A is translated so its
%     smallest value sits at 0 and B is translated by the same amount.
%     Both forms are rounded to roundDigits decimal places (from
%     internal.keyRoundDigits, which links them to the accuracy floor
%     set by truncationSigmas), as the cyclic form is.
%
%   Values are re-rounded after canonicalization when nDec is non-empty:
%   the mod-reduction and subtraction above introduce floating-point noise
%   that would otherwise separate keys which ought to coincide.
%
%   Inputs:
%     pAv, pBv  - Row vectors of values for A and B, already NaN-stripped.
%     wAv, wBv  - Matching weights, or [] for uniform weights.
%     rel       - Logical. Relative ([rel]) reading.
%     per       - Logical. Periodic ([per]) reading.
%     period    - Numeric scalar. The period, used when per is true.
%     nDec      - Decimal places for post-canonicalization rounding, or []
%                 for none beyond roundDigits.
%     roundDigits - Decimal places for every canonical form, normally
%                 internal.keyRoundDigits(sigma, r, truncationSigmas);
%                 optional, default 9.
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

    if nargin < 9 || isempty(roundDigits)
        roundDigits = 9;
    end
    if ~isempty(nDec)
        % An explicit precision coarser than the accuracy floor's places
        % governs the whole canonicalization, including which rotation
        % is canonical, so that it can merge what it rounds.
        roundDigits = min(roundDigits, nDec);
    end
    if rel
        % Relative: independent canonicalization
        [pAc, wAc] = internal.canonicalizeSet(pAv, wAv, rel, per, ...
                                              period, roundDigits);
        [pBc, wBc] = internal.canonicalizeSet(pBv, wBv, rel, per, ...
                                              period, roundDigits);
    else
        % Absolute: joint co-transposition normalization.
        % simMaet(A-c, B-c) = simMaet(A, B) because the
        % raw tuple differences cancel. Find A's canonical form and
        % apply the same shift to B.

        hasWA = ~isempty(wAv);
        hasWB = ~isempty(wBv);

        % Canonicalize A
        [pAs, siA] = sort(pAv);
        if hasWA, wAs = wAv(siA); else, wAs = []; end

        if per
            pAs = mod(pAs, period);
            [pAs, siA2] = sort(pAs);
            if hasWA, wAs = wAs(siA2); end
            % Cyclic canonical form — collapses all rotations. When A is
            % symmetric (several rotations give its canonical form, e.g.
            % a diminished seventh), each of those shifts is an equally
            % valid co-transposition, so the one giving B's lex-smallest
            % form is taken; otherwise transposed copies of a pair could
            % keep different B forms.
            [pAc, wAc, ~, tieShifts] = internal.cyclicCanonical(pAs, ...
                wAs, hasWA, period, roundDigits);
        else
            tieShifts = pAs(1);
            [pAc, wAc] = internal.roundedSorted(pAs - pAs(1), wAs, [], ...
                                               roundDigits);
        end

        % Apply the same shift to B (rounded as A's form is, so that
        % floating-point noise does not separate equivalent pairs)
        [pBs, siB] = sort(pBv);
        if hasWB, wBs = wBv(siB); else, wBs = []; end

        pBc = [];
        wBc = [];
        for t = 1:numel(tieShifts)
            if per
                [pT, wT] = internal.roundedSorted( ...
                    mod(pBs - tieShifts(t), period), wBs, period, ...
                    roundDigits);
            else
                [pT, wT] = internal.roundedSorted(pBs - tieShifts(t), ...
                                                  wBs, [], roundDigits);
            end
            if t == 1
                better = true;
            else
                cmp = internal.lexCompare(pT, pBc);
                better = cmp < 0 || (cmp == 0 && hasWB && ...
                    internal.lexCompare(wT, wBc) < 0);
            end
            if better
                pBc = pT;
                wBc = wT;
            end
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
