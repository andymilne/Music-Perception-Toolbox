function s = batchCosSimExpTens(pMatA, pMatB, sigma, r, isRel, isPer, period, varargin)
%BATCHCOSSIMEXPTENS Batch cosine similarity of expectation tensors.
%
%   s = batchCosSimExpTens(pMatA, pMatB, sigma, r, isRel, isPer, period):
%   Computes the cosine similarity between the r-ad expectation tensors of
%   paired weighted multisets (p represents pitches or positions). Each row
%   of pMatA and pMatB defines one weighted multiset; the function returns
%   one similarity value per row.
%
%   The function deduplicates equivalent rows before computation.
%   Density structs (via buildExpTens) are cached per unique individual
%   set and cosSimExpTens is called once per unique (A, B) pair, with
%   results mapped back to all matching rows. The equivalences exploited
%   depend on the mode:
%
%     Absolute non-periodic (isRel = false, isPer = false):
%       Co-transposition — (A+c, B+c) is equivalent to (A, B).
%       Mechanism: subtract A's minimum from both A and B.
%
%     Absolute periodic (isRel = false, isPer = true):
%       Co-transposition and octave displacement.
%       Mechanism: mod-reduce both sets, then apply the cyclic
%       canonical form to A and the same shift to B.
%
%     Relative non-periodic (isRel = true, isPer = false):
%       Independent transposition of each set (co-transposition is
%       a special case).
%       Mechanism: subtract the minimum from each set independently.
%
%     Relative periodic (isRel = true, isPer = true):
%       Independent transposition and octave displacement of each
%       set (co-transposition is a special case).
%       Mechanism: independent cyclic canonical form for each set.
%
%   Inputs:
%     pMatA — nRows x nA matrix of pitch or position values for
%                multiset A. Each row is one observation; columns are
%                individual values. NaN values are ignored (rows may have
%                varying numbers of valid values).
%     pMatB — nRows x nB matrix of pitch or position values for
%                multiset B (same number of rows as pMatA).
%     sigma    — Standard deviation of the Gaussian kernel
%     r        — Tuple size (positive integer; r >= 2 if isRel == true)
%     isRel    — If true, use transposition-invariant quadratic form
%     isPer    — If true, wrap differences to periodic interval
%     period   — Period for periodic wrapping
%
%   Optional name-value pairs:
%     'weightsA', wA — nRows x nA matrix of weights for multiset A. If
%                      omitted or empty, uniform weights are used.
%     'weightsB', wB — nRows x nB matrix of weights for multiset B. If
%                      omitted or empty, uniform weights are used.
%     'spectrum', specArgs — Cell array of arguments to pass to addSpectra
%                      (everything after p and w). If provided, partials
%                      are added to each multiset before computing
%                      similarity. Example:
%                        'spectrum', {'harmonic', 12, 'powerlaw', 1}
%                        'spectrum', {'stretched', 8, 1.02, 'powerlaw', 1}
%                      If omitted, pitches are used as-is.
%
%                      When both weights and spectrum are provided, the
%                      final weight of each partial is the product of the
%                      pitch weight and the spectral weight. For example,
%                      if a pitch has weight 0.5 and the 5th harmonic has
%                      spectral weight 0.2 (from a power-law rolloff),
%                      the final weight of that partial is 0.5 * 0.2 = 0.1.
%                      This multiplication is performed inside addSpectra.
%     'verbose', tf  — Logical (default: true). If false, suppresses all
%                      console output.
%     'precision', nDec — Round pitch and weight values to nDec decimal
%                      places before processing (and again after
%                      canonicalization, to absorb arithmetic noise from
%                      mod-reduction and subtraction). This ensures that
%                      nominally identical multisets differing only by
%                      floating-point noise are correctly deduplicated.
%                      For pitch data in cents on a 12-TET grid, 4 is
%                      more than sufficient; for fractional-cent values
%                      (e.g., from JI ratios), 6 preserves all meaningful
%                      precision. Default: no rounding (full
%                      floating-point precision).
%
%                      Limitation: 'precision' rounds to decimal places,
%                      so it cannot resolve discrepancies when pitches lie
%                      on an irrational grid — e.g., N-EDO tunings where
%                      the step size 1200/N is a repeating decimal (such
%                      as 22-EDO: 1200/22 ≈ 54.5454...). Different
%                      transpositions of the same set will produce
%                      different last-digit truncations that no decimal
%                      precision can collapse. In such cases, convert to
%                      integer EDO steps before calling this function
%                      (scaling sigma and period accordingly) to make
%                      deduplication exact.
%
%   Output:
%     s — nRows x 1 column vector of cosine similarities. Rows where
%         either multiset is empty or has fewer valid elements than r
%         are set to NaN.
%
%   Examples:
%     % Simple: two sets of pitches in cents, no spectra
%     A = [0 200 400 500 700 900 1100;   % diatonic scale
%          0 200 400 500 700 900 1100];   % same scale (repeated row)
%     B = [0 400 700;                     % major triad
%          0 300 700];                     % minor triad
%     s = batchCosSimExpTens(A, B, 10, 1, false, true, 1200);
%
%     % With harmonic spectra (power-law rolloff)
%     s = batchCosSimExpTens(A, B, 10, 1, false, true, 1200, ...
%                            'spectrum', {'harmonic', 12, 'powerlaw', 1});
%
%     % With weights and verbose off
%     wA = ones(size(A));
%     wB = [1 0.8 0.6; 1 0.8 0.6];
%     s = batchCosSimExpTens(A, B, 10, 1, false, true, 1200, ...
%                            'weightsA', wA, 'weightsB', wB, ...
%                            'verbose', false);
%
%   When to use batchCosSimExpTens vs the lower-level functions:
%
%     Use batchCosSimExpTens when you have many pairs of weighted multisets
%     and want the similarity of each pair. It handles spectral enrichment
%     (via addSpectra), deduplication of repeated multiset pairs, progress
%     reporting, and NaN handling in a single call. Typical use cases:
%       - Processing experimental data with one similarity per trial
%       - Computing similarity matrices (all pairs from two lists)
%       - Any situation where the same multisets appear in multiple rows
%         (deduplication avoids redundant computation)
%
%     Use addSpectra + cosSimExpTens directly when you need more control
%     over the intermediate steps — for example:
%       - Inspecting or modifying the spectralized pitch/weight vectors
%         before computing similarity
%       - Using buildExpTens to precompute a density object that is reused
%         across many different comparisons (e.g., one fixed reference vs
%         a changing comparison)
%       - Computing a single similarity value in a context where the
%         overhead of matrix construction is unnecessary
%
%     Use addSpectra + evalExpTens when you want to evaluate the density
%     at specific query points rather than computing a similarity between
%     two weighted multisets.
%
%   See also cosSimExpTens, addSpectra, buildExpTens, evalExpTens.

% === Parse optional arguments ===

weightsA = [];
weightsB = [];
specArgs = {};
verbose  = true;
nDec     = [];    % no rounding by default

i = 1;
while i <= numel(varargin)
    if ischar(varargin{i}) || isstring(varargin{i})
        switch lower(varargin{i})
            case 'weightsa'
                weightsA = varargin{i + 1};
                i = i + 2;
            case 'weightsb'
                weightsB = varargin{i + 1};
                i = i + 2;
            case 'spectrum'
                specArgs = varargin{i + 1};
                if ~iscell(specArgs)
                    error('''spectrum'' value must be a cell array of addSpectra arguments.');
                end
                i = i + 2;
            case 'verbose'
                verbose = logical(varargin{i + 1});
                i = i + 2;
            case 'precision'
                nDec = varargin{i + 1};
                i = i + 2;
            otherwise
                error('Unknown option ''%s''.', varargin{i});
        end
    else
        error('Expected a name-value pair; got a %s.', class(varargin{i}));
    end
end

% === Input validation ===

nRows = size(pMatA, 1);
if size(pMatB, 1) ~= nRows
    error('pMatA and pMatB must have the same number of rows.');
end

if ~isempty(weightsA) && ~isequal(size(weightsA), size(pMatA))
    error('weightsA must be the same size as pMatA.');
end
if ~isempty(weightsB) && ~isequal(size(weightsB), size(pMatB))
    error('weightsB must be the same size as pMatB.');
end

useSpectra  = ~isempty(specArgs);
useWeightsA = ~isempty(weightsA);
useWeightsB = ~isempty(weightsB);

% === Apply precision rounding ===

if ~isempty(nDec)
    pMatA = round(pMatA, nDec);
    pMatB = round(pMatB, nDec);
    if useWeightsA
        weightsA = round(weightsA, nDec);
    end
    if useWeightsB
        weightsB = round(weightsB, nDec);
    end
end

% === Phase 1: Canonicalize and build individual-set keys ===
% Each set is independently canonicalized under isPer/isRel so that
% equivalent pitch sets (differing only by octave displacement or
% transposition) map to the same key. Keys for A-sets and B-sets are
% built separately to enable individual-set density struct caching.

nA = size(pMatA, 2);
nB = size(pMatB, 2);

keyWidthA = nA * (1 + useWeightsA);
keyWidthB = nB * (1 + useWeightsB);

keysA = NaN(nRows, keyWidthA);
keysB = NaN(nRows, keyWidthB);
valid = false(nRows, 1);
s     = NaN(nRows, 1);

for i = 1:nRows
    pA = pMatA(i, :);
    pB = pMatB(i, :);

    maskA = ~isnan(pA);
    maskB = ~isnan(pB);
    pAv   = pA(maskA);
    pBv   = pB(maskB);

    % Check minimum p-value count
    if numel(pAv) < r || numel(pBv) < r
        continue;
    end

    % Get weights (or empty for all ones)
    if useWeightsA
        wAv = weightsA(i, maskA);
    else
        wAv = [];
    end
    if useWeightsB
        wBv = weightsB(i, maskB);
    else
        wBv = [];
    end

    % Canonicalize each set and apply joint co-transposition
    % normalization for the absolute case.
    if isRel
        % Relative: independent canonicalization
        [pAc, wAc] = canonicalizeSet(pAv, wAv, isRel, isPer, period);
        [pBc, wBc] = canonicalizeSet(pBv, wBv, isRel, isPer, period);
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
            [pAc, wAc, shift] = cyclicCanonical(pAs, wAs, hasWA, period);
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

    % Build NaN-padded keys
    keyA = NaN(1, keyWidthA);
    keyA(1:numel(pAc)) = pAc;
    if useWeightsA
        keyA(nA + 1 : nA + numel(wAc)) = wAc;
    end

    keyB = NaN(1, keyWidthB);
    keyB(1:numel(pBc)) = pBc;
    if useWeightsB
        keyB(nB + 1 : nB + numel(wBc)) = wBc;
    end

    keysA(i, :) = keyA;
    keysB(i, :) = keyB;
    valid(i)    = true;
end

% === Phase 2: Deduplicate individual sets, then pairs ===

validIdx = find(valid);
[uniqueKeysA, ~, mapA] = unique(keysA(validIdx, :), 'rows');
[uniqueKeysB, ~, mapB] = unique(keysB(validIdx, :), 'rows');
nUniqueA = size(uniqueKeysA, 1);
nUniqueB = size(uniqueKeysB, 1);

% Deduplicate (A, B) pairs by their individual-set indices
pairKeys = [mapA, mapB];
[uniquePairs, ~, pairMap] = unique(pairKeys, 'rows');
nUniquePairs = size(uniquePairs, 1);

if verbose
    fprintf(['batchCosSimExpTens: %d rows, %d valid, ' ...
             '%d unique A-sets, %d unique B-sets, ' ...
             '%d unique pairs.\n'], ...
        nRows, numel(validIdx), nUniqueA, nUniqueB, nUniquePairs);
    if isRel
        if isPer
            fprintf(['  Canonicalization: A-sets and B-sets independently ' ...
                     'normalized for transposition and octave equivalence.\n']);
        else
            fprintf(['  Canonicalization: A-sets and B-sets independently ' ...
                     'normalized for transposition.\n']);
        end
    else
        if isPer
            fprintf(['  Canonicalization: joint co-transposition with ' ...
                     'octave equivalence; B-set counts reflect position ' ...
                     'relative to A.\n']);
        else
            fprintf(['  Canonicalization: joint co-transposition; ' ...
                     'B-set counts reflect position relative to A.\n']);
        end
    end
end

% === Phase 3: Build density structs for unique individual sets ===
% Each unique set is processed once (addSpectra + buildExpTens), then
% reused across all pairs that reference it.

densA = cell(nUniqueA, 1);
for ua = 1:nUniqueA
    [pA, wA] = extractFromKey(uniqueKeysA(ua, :), nA, useWeightsA);
    if useSpectra
        [pA, wA] = addSpectra(pA, wA, specArgs{:});
    end
    densA{ua} = buildExpTens(pA, wA, sigma, r, isRel, isPer, period, ...
                             'verbose', false);
end

densB = cell(nUniqueB, 1);
for ub = 1:nUniqueB
    [pB, wB] = extractFromKey(uniqueKeysB(ub, :), nB, useWeightsB);
    if useSpectra
        [pB, wB] = addSpectra(pB, wB, specArgs{:});
    end
    densB{ub} = buildExpTens(pB, wB, sigma, r, isRel, isPer, period, ...
                             'verbose', false);
end

if verbose
    fprintf('batchCosSimExpTens: built %d density structs (%d A + %d B).\n', ...
        nUniqueA + nUniqueB, nUniqueA, nUniqueB);
end

% === Phase 4: Compute similarity for each unique pair ===

uniqueS = NaN(nUniquePairs, 1);
for up = 1:nUniquePairs
    dA = densA{uniquePairs(up, 1)};
    dB = densB{uniquePairs(up, 2)};
    uniqueS(up) = cosSimExpTens(dA, dB, 'verbose', false);

    % Progress
    if verbose && (mod(up, 100) == 0 || up == nUniquePairs)
        fprintf('  %d / %d unique pairs computed.\n', up, nUniquePairs);
    end
end

% === Phase 5: Map results back to all valid rows ===

s(validIdx) = uniqueS(pairMap);

if verbose
    fprintf('batchCosSimExpTens: done.\n');
end

end


% =====================================================================
%  LOCAL HELPER FUNCTIONS
% =====================================================================

function [pCan, wCan] = canonicalizeSet(p, w, isRel, isPer, period)
%CANONICALIZESET Canonical form of a pitch/weight set under isPer/isRel.
%   Sorts pitches (aligning weights), reduces modulo period if isPer,
%   removes transposition if isRel, and applies the cyclic canonical
%   form when both flags are true.
%
%   The cyclic canonical form is valid because cosSimExpTens wraps
%   pairwise differences in the isRel quadratic form (when isPer is
%   true), restoring exact transposition invariance on the circle.

    hasWeights = ~isempty(w);

    % Sort p-values and align weights
    [p, si] = sort(p);
    if hasWeights
        w = w(si);
    end

    % Reduce modulo period
    if isPer
        p = mod(p, period);
        [p, si] = sort(p);
        if hasWeights
            w = w(si);
        end
    end

    % Remove transposition
    if isRel
        if isPer
            % Cyclic canonical form: the lexicographically smallest
            % rotation (subtract each p-value in turn, mod period, re-sort
            % with weights) captures all transposition-modulo-period
            % equivalences.
            [pCan, wCan, ~] = cyclicCanonical(p, w, hasWeights, period);
        else
            % Subtract minimum (sort order is preserved)
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


function [pBest, wBest, bestShift] = cyclicCanonical(pSorted, wSorted, hasWeights, period)
%CYCLICCANONICAL Lexicographically smallest rotation of a periodic set.
%   For n pitches, tries all n rotations (subtract p(i), mod period,
%   re-sort with weights) and returns the lexicographically smallest
%   (pitch, weight) vector plus the shift that produced it.

    n = numel(pSorted);

    % Rotation 0: subtract the first element
    pBest = pSorted(:)' - pSorted(1);
    if hasWeights
        wBest = wSorted(:)';
    else
        wBest = [];
    end
    bestShift = pSorted(1);

    for rot = 2:n
        shifted = mod(pSorted - pSorted(rot), period);
        [shifted, si] = sort(shifted);

        % Compare p-values first; break ties with weights
        cmp = lexCompare(shifted(:)', pBest);
        if cmp < 0
            pBest = shifted(:)';
            bestShift = pSorted(rot);
            if hasWeights
                wBest = wSorted(si)';
            end
        elseif cmp == 0 && hasWeights
            wRot = wSorted(si)';
            if lexCompare(wRot, wBest) < 0
                wBest = wRot;
                bestShift = pSorted(rot);
            end
        end
    end
end


function cmp = lexCompare(a, b)
%LEXCOMPARE Lexicographic comparison of two row vectors.
%   Returns -1 if a < b, 0 if a == b, +1 if a > b.
    idx = find(a ~= b, 1);
    if isempty(idx)
        cmp = 0;
    elseif a(idx) < b(idx)
        cmp = -1;
    else
        cmp = 1;
    end
end


function [p, w] = extractFromKey(key, nMax, hasWeights)
%EXTRACTFROMKEY Extract pitch and weight vectors from a NaN-padded key row.
    pPart = key(1:nMax);
    p = pPart(~isnan(pPart));
    p = p(:);
    if hasWeights
        wPart = key(nMax + 1 : end);
        w = wPart(~isnan(wPart));
        w = w(:);
    else
        w = [];
    end
end