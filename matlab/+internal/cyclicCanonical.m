function [pBest, wBest, bestShift, tieShifts] = cyclicCanonical(pSorted, wSorted, hasWeights, period, roundDigits)
%CYCLICCANONICAL Lexicographically smallest rotation of a periodic set.
%   [pBest, wBest, bestShift] = internal.cyclicCanonical(pSorted, ...
%                                  wSorted, hasWeights, period)
%   [pBest, wBest, bestShift, tieShifts] = internal.cyclicCanonical(...)
%   [...] = internal.cyclicCanonical(..., roundDigits)
%
%   For n pitches, considers all n rotations (subtract p(i), mod period,
%   re-sort with weights) and returns the lexicographically smallest
%   (pitch, weight) vector plus the shift that produced it; ties go to
%   the earliest rotation. tieShifts lists the shifts of every rotation
%   attaining that form (a symmetric set has several), in rotation
%   order, the first being bestShift.
%
%   Pitch values are rounded to roundDigits decimal places (default 9;
%   callers that know the kernel width pass internal.keyRoundDigits)
%   before lex comparison, to absorb floating-point noise from
%   mod-reduction. Without this rounding, two transposition-equivalent multisets
%   with different FP error patterns can produce different canonical
%   forms, which breaks consumer-level dedup for non-integer pitch
%   data. ``bestShift`` is returned at full precision so callers
%   using it to apply to a paired set get exact arithmetic.
%
%   Evaluating every rotation costs O(n^2 log n), which dominated the
%   canonicalisation of spectrally-expanded sets (n = K * partials).
%   For n > NARROW_MIN_N the rotations are therefore first narrowed
%   one column at a time, each column costing O(n), and only the few
%   that remain are compared in full (see the comments below). The
%   result is identical to the rotation-by-rotation search, which is
%   kept (localLoop) for very small sets, where it is faster. Twin of
%   _cyclic_canonical in python/mpt/_tensor/canonical.py.
%
%   Weight ties (rotations giving the same pitches) are broken by the
%   lex-smallest weights, as in Python. Before this revision the
%   comparison mixed a row with a column, so those ties were not
%   broken by lex order; canonical keys for weighted sets with such
%   symmetries may therefore differ from earlier versions (results do
%   not: a key only decides which rows share one computation).
%
%   Inputs
%     pSorted    : 1-by-n sorted pitch row (reduced mod period if
%                  applicable).
%     wSorted    : 1-by-n weights aligned with pSorted, or [].
%     hasWeights : logical, whether wSorted is non-empty.
%     period     : period in cents.

    LOOP_MAX_N = 6;        % at most this many values: localLoop
    NARROW_MIN_N = 16;     % more than this many values: narrow first
    EXACT_MAX_ROWS = 4;    % narrow until at most this many remain
    if nargin < 5 || isempty(roundDigits)
        roundDigits = 9;
    end
    ROUND_DIGITS = roundDigits;

    n = numel(pSorted);
    if n <= LOOP_MAX_N
        [pBest, wBest, bestShift, tieShifts] = localLoop(pSorted, ...
            wSorted, hasWeights, period, ROUND_DIGITS);
        return
    end

    % Candidate rotations. Only rotations starting at the first of a
    % run of equal values need narrowing (a later member gives the same
    % pitches). For such a rotation k the form, before re-sorting, is
    % mod(p(i) - p(k)) for i = k, ..., n and then i = 1, ..., k-1. Both
    % runs are non-decreasing in floating point (subtraction, the mod's
    % single adjustment, and rounding are all monotone), so the
    % re-sorted form is this cyclic order provided the first run's last
    % value does not exceed the second run's first, which is checked.
    % Its j-th value is then the same number localRowsBest computes, so
    % the narrowing compares exact values and drops only rotations
    % that are lex-greater than another.
    cand = 1:n;
    if n > NARROW_MIN_N
        p = pSorted(:)';
        first = [true, p(2:end) ~= p(1:end-1)];
        heads = find(first);
        k = heads(heads > 1);
        cyclicOk = all( ...
            round(mod(p(end) - p(k), period), ROUND_DIGITS) <= ...
            round(mod(p(1) - p(k), period), ROUND_DIGITS));
        if cyclicOk
            cand = heads;
            j = 1;
            while numel(cand) > EXACT_MAX_ROWS && j < n
                idx = mod(cand - 1 + j, n) + 1;
                col = round(mod(p(idx) - p(cand), period), ROUND_DIGITS);
                cand = cand(col == min(col));
                j = j + 1;
            end
            % A rotation starting at a later member of a run of equal
            % values yields the same pitches, but its weights may be
            % ordered differently, so every member of each surviving
            % run is compared exactly.
            runId = cumsum(first);
            cand = find(ismember(runId, runId(cand)));
        end
    end

    [pBest, wBest, bestShift, tieShifts] = localRowsBest(pSorted, ...
        wSorted, hasWeights, period, cand, ROUND_DIGITS);
end


function [pBest, wBest, bestShift, tieShifts] = localRowsBest(pSorted, ...
        wSorted, hasWeights, period, cand, ROUND_DIGITS)
%LOCALROWSBEST Lex-smallest exact rotated form among rotations cand.
%   The form for rotation k is mod(p - p(k), period), re-sorted (stable,
%   with the weights) and rounded; rotation 1 is p - p(1) in the given
%   order. Rows are evaluated in blocks to bound memory; ties go to the
%   earliest rotation. Outputs are oriented as in localLoop.
    BLOCK_ELEMS = 1e6;
    n = numel(pSorted);
    p = pSorted(:)';
    block = max(1, floor(BLOCK_ELEMS / n));
    pBest = [];
    wBest = [];
    wBestRow = [];
    bestRot = 1;
    bestSi = 1:n;
    ties = [];
    for b0 = 1:block:numel(cand)
        rots = cand(b0 : min(numel(cand), b0 + block - 1));
        rots = rots(:);
        shifted = mod(p - p(rots).', period);
        [shifted, si] = sort(shifted, 2);
        is1 = rots == 1;
        if any(is1)
            shifted(is1, :) = repmat(p - p(1), nnz(is1), 1);
            si(is1, :) = repmat(1:n, nnz(is1), 1);
        end
        P = round(shifted, ROUND_DIGITS);
        if hasWeights
            w = wSorted(:).';
            W = reshape(w(si), size(si));
        else
            W = [];
        end
        b = localLexMinRow(P, W);
        candP = P(b, :);
        same = all(P == candP, 2);
        if hasWeights
            same = same & all(W == W(b, :), 2);
        end
        blockTies = rots(same);
        if isempty(pBest)
            better = true;
            equal = false;
        else
            cmp = internal.lexCompare(candP, pBest);
            cmpW = 0;
            if cmp == 0 && hasWeights
                cmpW = internal.lexCompare(W(b, :), wBestRow);
            end
            better = cmp < 0 || (cmp == 0 && cmpW < 0);
            equal = cmp == 0 && cmpW == 0;
        end
        if better
            pBest = candP;
            bestRot = rots(b);
            bestSi = si(b, :);
            if hasWeights
                wBestRow = W(b, :);
            end
            ties = blockTies(:)';
        elseif equal
            ties = [ties, blockTies(:)'];
        end
    end
    tieShifts = pSorted(ties);
    bestShift = pSorted(bestRot);
    if hasWeights
        if bestRot == 1
            wBest = wSorted(:)';
        else
            % As localLoop: wSorted indexed by a sort index shaped like
            % pSorted, then transposed.
            wBest = wSorted(reshape(bestSi, size(pSorted)))';
        end
    end
end


function b = localLexMinRow(P, W)
%LOCALLEXMINROW Index of the lexicographically smallest row of P (ties
%   broken by the rows of W, if non-empty, then by the lowest index).
%   Repeatedly finds the first column in which the remaining rows
%   differ and keeps those with its smallest value.
    cand = (1:size(P, 1)).';
    mats = {P, W};
    for m = 1:2
        M = mats{m};
        if isempty(M)
            continue
        end
        j0 = 1;
        while numel(cand) > 1
            sub = M(cand, j0:end);
            differs = find(any(sub ~= sub(1, :), 1), 1);
            if isempty(differs)
                break
            end
            col = sub(:, differs);
            cand = cand(col == min(col));
            j0 = j0 + differs;
        end
    end
    b = cand(1);
end


function [pBest, wBest, bestShift, tieShifts] = localLoop(pSorted, ...
        wSorted, hasWeights, period, ROUND_DIGITS)
%LOCALLOOP Rotation-by-rotation search (faster for very small sets).
    n = numel(pSorted);

    % Rotation 0: subtract the first element.
    pBest = round(pSorted(:)' - pSorted(1), ROUND_DIGITS);
    if hasWeights
        wBest = wSorted(:)';
    else
        wBest = [];
    end
    bestShift = pSorted(1);
    ties = 1;

    for rot = 2:n
        shifted = mod(pSorted - pSorted(rot), period);
        [shifted, si] = sort(shifted);
        shifted = round(shifted(:)', ROUND_DIGITS);

        cmp = internal.lexCompare(shifted, pBest);
        if cmp < 0
            pBest = shifted;
            bestShift = pSorted(rot);
            ties = rot;
            if hasWeights
                wBest = wSorted(si)';
            end
        elseif cmp == 0 && ~hasWeights
            ties(end+1) = rot; %#ok<AGROW>
        elseif cmp == 0 && hasWeights
            wRot = wSorted(si)';
            % Compare as rows: wRot and wBest can differ in orientation
            % (wBest starts as a row; wSorted(si)' of a row is a column),
            % and lexCompare of a row with a column compared the wrong
            % elements, so weight ties were not broken by lex order.
            cmpW = internal.lexCompare(wRot(:)', wBest(:)');
            if cmpW < 0
                wBest = wRot;
                bestShift = pSorted(rot);
                ties = rot;
            elseif cmpW == 0
                ties(end+1) = rot; %#ok<AGROW>
            end
        end
    end
    tieShifts = pSorted(ties);
end
