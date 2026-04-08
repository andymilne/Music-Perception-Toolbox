function s = batchCosSimExpTens(pitchesA, pitchesB, sigma, r, isRel, isPer, period, varargin)
%BATCHCOSSIMEXPTENS Batch cosine similarity of expectation tensors.
%
%   s = batchCosSimExpTens(pitchesA, pitchesB, sigma, r, isRel, isPer, period):
%   Computes the cosine similarity between the r-ad expectation tensors of
%   paired pitch sets. Each row of pitchesA and pitchesB defines one pitch
%   set; the function returns one similarity value per row.
%
%   Rows with identical (sorted) pitch content share the same similarity
%   value. The function automatically deduplicates, computing cosSimExpTens
%   only once per unique (A, B) pair, then maps results back to all rows.
%   This can dramatically reduce computation time when many rows share the
%   same pitch sets (e.g., repeated experimental conditions).
%
%   Inputs:
%     pitchesA — nRows x nA matrix of pitches for set A. Each row is one
%                observation; columns are individual pitches. NaN values
%                are ignored (rows may have varying numbers of valid pitches).
%     pitchesB — nRows x nB matrix of pitches for set B (same number of
%                rows as pitchesA).
%     sigma    — Standard deviation of the Gaussian kernel
%     r        — Tuple size (positive integer; r >= 2 if isRel == true)
%     isRel    — If true, use transposition-invariant quadratic form
%     isPer    — If true, wrap differences to periodic interval
%     period   — Period for periodic wrapping
%
%   Optional name-value pairs:
%     'weightsA', wA — nRows x nA matrix of weights for set A. If omitted
%                      or empty, uniform weights are used.
%     'weightsB', wB — nRows x nB matrix of weights for set B. If omitted
%                      or empty, uniform weights are used.
%     'spectrum', specArgs — Cell array of arguments to pass to addSpectra
%                      (everything after p and w). If provided, partials
%                      are added to each pitch set before computing
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
%
%   Output:
%     s — nRows x 1 column vector of cosine similarities. Rows where
%         either pitch set is empty or has fewer valid pitches than r
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
%     Use batchCosSimExpTens when you have many pairs of pitch sets and
%     want the similarity of each pair. It handles spectral enrichment
%     (via addSpectra), deduplication of repeated pitch-set pairs, progress
%     reporting, and NaN handling in a single call. Typical use cases:
%       - Processing experimental data with one similarity per trial
%       - Computing similarity matrices (all pairs from two lists)
%       - Any situation where the same pitch sets appear in multiple rows
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
%     two pitch sets.
%
%   See also cosSimExpTens, addSpectra, buildExpTens, evalExpTens.

% === Parse optional arguments ===

weightsA = [];
weightsB = [];
specArgs = {};
verbose  = true;

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
            otherwise
                error('Unknown option ''%s''.', varargin{i});
        end
    else
        error('Expected a name-value pair; got a %s.', class(varargin{i}));
    end
end

% === Input validation ===

nRows = size(pitchesA, 1);
if size(pitchesB, 1) ~= nRows
    error('pitchesA and pitchesB must have the same number of rows.');
end

if ~isempty(weightsA) && ~isequal(size(weightsA), size(pitchesA))
    error('weightsA must be the same size as pitchesA.');
end
if ~isempty(weightsB) && ~isequal(size(weightsB), size(pitchesB))
    error('weightsB must be the same size as pitchesB.');
end

useSpectra  = ~isempty(specArgs);
useWeightsA = ~isempty(weightsA);
useWeightsB = ~isempty(weightsB);

% === Identify valid rows and build deduplication keys ===
% A valid row has at least r non-NaN pitches in both A and B.
% The key for deduplication is the sorted non-NaN pitches (and weights,
% if provided) concatenated into a fixed-width vector.

nA = size(pitchesA, 2);
nB = size(pitchesB, 2);

% Key width: pitches + optional weights for both sides
keyWidth = nA + nB;
if useWeightsA, keyWidth = keyWidth + nA; end
if useWeightsB, keyWidth = keyWidth + nB; end

keys    = NaN(nRows, keyWidth);
valid   = false(nRows, 1);
s       = NaN(nRows, 1);

for i = 1:nRows
    pA = pitchesA(i, :);
    pB = pitchesB(i, :);

    % Remove NaN
    maskA = ~isnan(pA);
    maskB = ~isnan(pB);
    pA_valid = sort(pA(maskA));
    pB_valid = sort(pB(maskB));

    % Check minimum pitch count
    if numel(pA_valid) < r || numel(pB_valid) < r
        continue;
    end

    % Pad to fixed width with NaN (sorted pitches left-aligned)
    keyA_p = NaN(1, nA);
    keyA_p(1:numel(pA_valid)) = pA_valid;
    keyB_p = NaN(1, nB);
    keyB_p(1:numel(pB_valid)) = pB_valid;

    key = [keyA_p, keyB_p];

    % Include weights in key if provided (different weights = different key)
    if useWeightsA
        wA_row = NaN(1, nA);
        wA_valid = weightsA(i, maskA);
        [~, sortIdx] = sort(pA(maskA));
        wA_row(1:numel(wA_valid)) = wA_valid(sortIdx);
        key = [key, wA_row]; %#ok<AGROW>
    end
    if useWeightsB
        wB_row = NaN(1, nB);
        wB_valid = weightsB(i, maskB);
        [~, sortIdx] = sort(pB(maskB));
        wB_row(1:numel(wB_valid)) = wB_valid(sortIdx);
        key = [key, wB_row]; %#ok<AGROW>
    end

    keys(i, :) = key;
    valid(i)   = true;
end

% === Deduplicate ===

validIdx  = find(valid);
validKeys = keys(validIdx, :);

% unique with 'rows' treats NaN as equal to NaN, which is what we want
% (NaN-padded columns should match)
[uniqueKeys, ~, keyMap] = unique(validKeys, 'rows');
nUnique = size(uniqueKeys, 1);

if verbose
    fprintf('batchCosSimExpTens: %d rows, %d valid, %d unique pairs.\n', ...
        nRows, numel(validIdx), nUnique);
end

% === Compute similarity for each unique pair ===

uniqueS = NaN(nUnique, 1);
for ui = 1:nUnique
    % Extract pitches from key (strip NaN padding)
    keyA_p = uniqueKeys(ui, 1:nA);
    keyB_p = uniqueKeys(ui, nA+1:nA+nB);
    pA = keyA_p(~isnan(keyA_p));
    pB = keyB_p(~isnan(keyB_p));
    pA = pA(:);
    pB = pB(:);

    % Extract weights from key (or use empty for uniform)
    if useWeightsA
        wA_key = uniqueKeys(ui, nA+nB+1 : nA+nB+nA);
        wA = wA_key(~isnan(wA_key));
        wA = wA(:);
    else
        wA = [];
    end
    if useWeightsB
        offset = nA + nB;
        if useWeightsA, offset = offset + nA; end
        wB_key = uniqueKeys(ui, offset+1 : offset+nB);
        wB = wB_key(~isnan(wB_key));
        wB = wB(:);
    else
        wB = [];
    end

    % Add spectral partials if requested
    if useSpectra
        [pA, wA] = addSpectra(pA, wA, specArgs{:});
        [pB, wB] = addSpectra(pB, wB, specArgs{:});
    end

    % Compute cosine similarity
    uniqueS(ui) = cosSimExpTens(pA, wA, pB, wB, ...
                                sigma, r, isRel, isPer, period, ...
                                'verbose', false);

    % Progress
    if verbose && (mod(ui, 100) == 0 || ui == nUnique)
        fprintf('  %d / %d unique pairs computed.\n', ui, nUnique);
    end
end

% === Map results back to all valid rows ===

s(validIdx) = uniqueS(keyMap);

if verbose
    fprintf('batchCosSimExpTens: done.\n');
end

end