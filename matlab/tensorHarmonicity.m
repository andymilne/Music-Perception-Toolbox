function h = tensorHarmonicity(p, w, sigma, nvArgs)
%TENSORHARMONICITY Harmonicity via expectation tensor lookup.
%
%   h = tensorHarmonicity(p, w, sigma)
%   h = tensorHarmonicity(p, w, sigma, Name, Value)
%
%   For batched processing (v2.1+), p may also be a 2-D nRows-by-K
%   matrix with both dimensions > 1; rows are then treated as separate
%   multisets and the function returns an nRows-by-1 column vector of
%   harmonicities. NaN-padded rows are accepted; rows with fewer than
%   2 valid pitches return NaN.
%
%   Measures the harmonicity of a weighted pitch multiset by evaluating
%   the relative r-ad expectation tensor of a harmonic series at the
%   multiset's interval vector. The expectation tensor represents the
%   density of all ordered r-tuples of intervals that arise from a
%   harmonic series, smoothed by perceptual uncertainty sigma. A high
%   density at the chord's intervals indicates those intervals are
%   likely to co-occur in a harmonic series — hence the multiset is
%   "harmonic."
%
%   The procedure is:
%     1. Build a harmonic template spectrum via addSpectra.
%     2. Duplicate the template K times (see 'duplicate' below).
%     3. Build the relative r-ad expectation tensor (r = number of
%        pitches, isRel = true, isPer = false) from the template.
%     4. Compute the chord's intervals relative to its lowest pitch.
%     5. Evaluate the tensor density at that single interval point.
%
%   By default, the template spectrum is duplicated K times, where K is the
%   number of pitches in the chord. This is important because without
%   duplication, every position in an r-tuple can only be filled by a
%   different partial — a unison (two chord tones sharing the same
%   partials) cannot contribute. With K-fold duplication, each partial can
%   appear in up to K positions, correctly allowing unisons (and other
%   interval repetitions within the harmonic series) to register as
%   consonant.
%
%   The expectation tensor is evaluated at exact query points (not on a
%   grid), so no resolution parameter is needed: the density is computed
%   analytically at the precise interval values. v2.2 routes this query
%   through the orbit-Mobius point evaluator (mobius.evalOrbitRel),
%   which evaluates the relative tensor without materialising the
%   (r-1, K!/(K-r)!) centres array. For dup = 4 with the default
%   64-partial template this avoids a centres array of order 10^9
%   floats; runtime is dominated by the u-grid translation integral and
%   grows as B_r * r * K * N_u per query. This unblocks chord
%   cardinalities greater than 3, which the v2.0/v2.1 centres path
%   could not feasibly handle.
%
%   For batch processing (many chords), pass a 2-D nRows-by-K matrix as
%   p; the batched dispatch path memoises both the harmonic template
%   (per duplicate value) and the per-chord orbit-evaluated harmonicity
%   (per canonical, transposition-equivalent chord). Rows whose sorted
%   pitches differ only by transposition share a single computation.
%
%   Inputs:
%     p     — Pitch values in cents (vector of length K, where K >= 2).
%             These are absolute pitches, not pitch classes. The
%             function computes intervals internally.
%     w     — Weights (vector of length K, or empty for all ones).
%             These weight the template's partials; typically left
%             empty unless modelling unequal-amplitude tones.
%     sigma — Gaussian smoothing width in cents. Models perceptual
%             uncertainty. Values of 9-15 are typical; 12 is a good
%             default.
%
%   Name-Value Arguments:
%     'spectrum'  — Cell array of arguments to pass to addSpectra
%                   (everything after p and w). Defines the harmonic
%                   template whose expectation tensor is queried.
%                   Default: {'harmonic', 64, 'powerlaw', 1}.
%                   Examples:
%                     'spectrum', {'harmonic', 24, 'powerlaw', 1}
%                     'spectrum', {'harmonic', 12, 'geometric', 0.9}
%                     'spectrum', {'stretched', 8, 1.02, 'powerlaw', 1}
%     'duplicate' — Number of times to replicate the template pitch
%                   before adding harmonics (default: 0, meaning auto).
%                     0  — Automatically set to the number of pitches
%                          in p (the chord cardinality). This is the
%                          recommended default.
%                     K  — Any positive integer overrides the automatic
%                          setting. K = 1 disables duplication (each
%                          partial appears once; unisons cannot
%                          contribute).
%                   Computation time grows rapidly with K: the number
%                   of ordered r-tuples scales as (N*K)! / (N*K - r)!
%                   where N is the number of partials. A warning is
%                   issued when K > 3.
%     'normalize' — Normalization mode for the density value
%                   (default: 'none'):
%                     'none'     — Raw density. Suitable for comparing
%                                  chords evaluated with the same
%                                  parameters.
%                     'gaussian' — Each Gaussian integrates to 1.
%                                  Useful for comparing across
%                                  different sigma values.
%                     'pdf'      — Full probability density. Useful
%                                  for comparing across different
%                                  template sizes.
%
%   Output:
%     h     — Harmonicity (scalar, non-negative). Higher values
%             indicate greater harmonicity.
%
%   Examples:
%     % Harmonicity of a JI major triad (default: 3-fold duplication)
%     h = tensorHarmonicity([0, 386.31, 701.96], [], 12)
%
%     % Harmonicity of a 12-EDO major triad
%     h = tensorHarmonicity([0, 400, 700], [], 12)
%
%     % Without duplication (unisons do not contribute)
%     h = tensorHarmonicity([0, 400, 700], [], 12, 'duplicate', 1)
%
%     % Using convertPitch from MIDI
%     p = convertPitch([60 64 67], 'midi', 'cents');
%     h = tensorHarmonicity(p, [], 12)
%
%   References:
%     Milne, A. J. (2013). A computational model of the cognition of
%       tonality. PhD thesis, The Open University.
%     Smit, E. A., Milne, A. J., Dean, R. T., & Weidemann, G. (2019).
%       Perception of affect in unfamiliar musical chords. PLOS ONE,
%       14(6), e0218570.
%
%   See also ADDSPECTRA, BUILDEXPTENS, EVALEXPTENS, CONVERTPITCH,
%            TEMPLATEHARMONICITY, ROUGHNESS, SPECTRALENTROPY.

    arguments
        p {mustBeNumeric}
        w {mustBeNumeric} = []
        sigma (1,1) {mustBePositive} = 12
        nvArgs.spectrum = {'harmonic', 64, 'powerlaw', 1}
        nvArgs.duplicate (1,1) {mustBeNonnegative, mustBeInteger} = 0
        nvArgs.normalize (1,1) string ...
            {mustBeMember(nvArgs.normalize, {'none','gaussian','pdf'})} = 'none'
        nvArgs.verbose (1,1) logical = true
    end

    % --- Batched dispatch (v2.1+) ---
    % If p is a 2-D matrix with both dimensions > 1, treat rows as
    % paired multisets and return a column vector of harmonicities.
    % NaN-padded rows are accepted; rows with fewer than 2 valid
    % pitches return NaN.
    if size(p, 1) > 1 && size(p, 2) > 1
        h = localBatchedTensorHarmonicity(p, w, sigma, nvArgs);
        return;
    end

    % Scalar path: force column vectors for consistency below.
    p = p(:);
    if ~isempty(w)
        w = w(:);
    end

    specArgs = nvArgs.spectrum;

    if ~iscell(specArgs)
        error('tensorHarmonicity:badSpectrum', ...
              '''spectrum'' value must be a cell array of addSpectra arguments.');
    end

    nPitches = numel(p);
    if nPitches < 2
        error('tensorHarmonicity:tooFewPitches', ...
              'At least 2 pitches are required (got %d).', nPitches);
    end

    % === Determine duplication count ===
    % Default (0) uses the chord cardinality, so that each partial can
    % fill every position in an r-tuple — allowing unisons to register
    % as consonant.

    dup = nvArgs.duplicate;
    if dup == 0
        dup = nPitches;
    end

    if dup > 3
        warning('tensorHarmonicity:largeDuplicate', ...
                ['duplicate = %d: computation time grows rapidly with ' ...
                 'duplication. Consider reducing to 3 or fewer.'], dup);
    end

    % === Build harmonic template spectrum ===
    % Pass dup copies of pitch 0 to addSpectra so each harmonic partial
    % appears dup times. This allows r-tuples that reuse the same
    % harmonic in multiple positions — critical for unisons (identical
    % intervals) to contribute to the density.

    [tmpl_p, tmpl_w] = addSpectra(zeros(dup, 1), ones(dup, 1), specArgs{:});

    % === Compute chord intervals ===
    % Sort pitches and take intervals relative to the lowest pitch,
    % giving a (nPitches-1)-dimensional interval vector — the query
    % point in the relative-tensor's reduced space.

    p = sort(p);
    intervals = p(2:end) - p(1);  % (nPitches-1) x 1 column vector

    % === Evaluate the relative template tensor at the chord's intervals ===
    %
    % v2.2 rewrite: the orbit-Mobius point evaluator (mobius.evalOrbitRel)
    % bypasses buildExpTens and evalExpTens entirely, never materialising
    % the (r-1, K!/(K-r)!) centres array. For dup = 4 with the default
    % 64-partial template this avoids a centres array of order 10^9
    % floats; runtime is dominated by the u-grid translation integral
    % and grows as B_r * r * K * N_u per query. This unblocks
    % nPitches > 3 where the centres path was infeasible.

    if nvArgs.verbose
        K_tmpl = numel(tmpl_p);
        fprintf(['tensorHarmonicity: orbit-path eval at K = %d, r = %d, ' ...
                 'sigma = %g.\n'], K_tmpl, nPitches, sigma);
    end

    h_vec = localTensorHarmonicityOrbit( ...
        tmpl_p, tmpl_w, sigma, nPitches, intervals, char(nvArgs.normalize));
    h = h_vec(1);

end


% =====================================================================
%  v2.2 orbit-Mobius core (shared by scalar and batched paths)
% =====================================================================

function vals = localTensorHarmonicityOrbit(tmpl_p, tmpl_w, sigma, r, ...
                                              x_query, normalize)
%LOCALTENSORHARMONICITYORBIT  Evaluate the relative template tensor at
%query points via mobius.evalOrbitRel, then apply normalisation.
%
%   x_query is (r-1, n_q). Returns a row vector of length n_q. The
%   normalisation maths is mirrored from evalExpTens so the value is
%   numerically identical to what the centres path would have produced
%   at the same query.

    vals = mobius.evalOrbitRel(tmpl_p(:), tmpl_w(:), sigma, r, x_query, ...
        'is_per', false, 'period', 0);
    vals = vals(:).';   % standardise to row vector for downstream use

    if strcmp(normalize, 'none')
        return;
    end

    % Template tensor is rel-mode: dim = r - 1, det_M = 1/r.
    dim = r - 1;
    detM = 1.0 / r;
    gaussConst = (2 * pi * sigma^2)^(-dim/2) * sqrt(detM);
    vals = vals * gaussConst;

    if strcmp(normalize, 'pdf')
        sumW = sum(tmpl_w);
        if sumW > 0
            vals = vals / sumW;
        else
            warning('tensorHarmonicity:zeroSumWeights', ...
                'Sum of template weights is zero; cannot normalize to pdf.');
        end
    end
end

% =====================================================================
%  Batched dispatch (v2.2: per-chord and per-template caching)
% =====================================================================

function h = localBatchedTensorHarmonicity(P, W, sigma, nvArgs)
%LOCALBATCHEDTENSORHARMONICITY Per-row tensor harmonicity from a 2-D matrix.
%
%   Returns an nRows-by-1 column vector. NaN-padded rows are handled
%   (NaN entries dropped per row); rows with fewer than 2 valid
%   pitches yield NaN.
%
%   v2.2+: groups rows by (effective nP, dup), deduplicates canonical
%   chord intervals within each group, and issues a single batched
%   call to LOCALTENSORHARMONICITYORBIT (which wraps
%   MOBIUS.EVALORBITREL) per group. This replaces the previous
%   per-row loop, which paid MATLAB function-call overhead once per
%   row regardless of how trivial each per-row computation was. With
%   v2.2's u-grid vectorisation in MOBIUS.EVALORBITREL, the batched
%   call processes all unique chord queries simultaneously. For
%   uniform-cardinality batches the loop collapses to a single orbit
%   call.

    nRows = size(P, 1);
    h = nan(nRows, 1);

    haveRowWeights = ~isempty(W) && isequal(size(W), size(P));
    if ~isempty(W) && ~haveRowWeights
        if isvector(W) && numel(W) == size(P, 2)
            % broadcastable; not yet used (chord weights don't enter
            % the orbit eval, since the chord is the query side).
        else
            error('tensorHarmonicity:weightShape', ...
                ['In batched mode, w must be empty, a matrix the same size as p, ' ...
                 'or a vector matching the number of pitch columns.']);
        end
    end

    specArgs = nvArgs.spectrum;
    normalize = char(nvArgs.normalize);
    duplicateOpt = nvArgs.duplicate;

    % Pass 1: per-row metadata. Rows with fewer than 2 valid pitches
    % keep h(k) = NaN and are excluded from grouping below.
    rowNP = zeros(nRows, 1);
    rowDup = zeros(nRows, 1);
    rowKey = cell(nRows, 1);
    rowIntervals = cell(nRows, 1);
    largeDupWarned = false;
    for k = 1:nRows
        pRow = P(k, :);
        validMask = ~isnan(pRow);
        pK = pRow(validMask);
        if numel(pK) < 2
            continue;
        end
        nP = numel(pK);
        if duplicateOpt == 0
            dup = nP;
        else
            dup = duplicateOpt;
        end
        if dup > 3 && ~largeDupWarned
            warning('tensorHarmonicity:largeDuplicate', ...
                ['duplicate = %d: computation time grows rapidly ' ...
                 'with duplication. Consider reducing to 3 or fewer.'], ...
                dup);
            largeDupWarned = true;
        end
        pSorted = sort(pK(:));
        pCanon = pSorted - pSorted(1);
        intervals = pCanon(2:end);

        rowNP(k) = nP;
        rowDup(k) = dup;
        rowIntervals{k} = intervals;
        rowKey{k} = sprintf('p=%s|s=%.12g|d=%d|n=%s', ...
            mat2str(pCanon, 12), sigma, dup, normalize);
    end

    % Pass 2: group by (nP, dup); within each group dedup canonical
    % chords; one batched orbit call per group; distribute back.
    validRows = find(rowNP > 0);
    if isempty(validRows)
        return;
    end

    groupTags = arrayfun( ...
        @(k) sprintf('%d_%d', rowNP(k), rowDup(k)), ...
        validRows, 'UniformOutput', false);
    [uniqueGroups, ~, groupIdx] = unique(groupTags);

    if nvArgs.verbose
        % Gate the groups print on a row-count threshold matching the
        % 'silent for fast' semantics used by the other batched
        % functions (printBatchedEstimate's min-print threshold). The
        % threshold is deliberately rough: at >~100 rows the batched
        % orbit call is likely to exceed the 10s estimate-print
        % threshold; tiny batches stay silent.
        nValid = numel(validRows);
        if nValid >= 100
            nGroups = numel(uniqueGroups);
            if nGroups == 1
                fprintf(['tensorHarmonicity: %d valid rows in 1 ' ...
                         '(nP, dup) group.\n'], nValid);
            else
                fprintf(['tensorHarmonicity: %d valid rows across %d ' ...
                         '(nP, dup) groups.\n'], nValid, nGroups);
            end
        end
    end

    for g = 1:numel(uniqueGroups)
        rowsInGroup = validRows(groupIdx == g);
        nP = rowNP(rowsInGroup(1));
        dup = rowDup(rowsInGroup(1));
        r = nP;

        % Dedup canonical chords within the group.
        keyToIdx = containers.Map('KeyType', 'char', 'ValueType', 'int32');
        nUnique = 0;
        groupRowKey = rowKey(rowsInGroup);
        groupRowIntervals = rowIntervals(rowsInGroup);
        rowToUniqueIdx = zeros(numel(rowsInGroup), 1);
        for ii = 1:numel(rowsInGroup)
            ck = groupRowKey{ii};
            if isKey(keyToIdx, ck)
                rowToUniqueIdx(ii) = keyToIdx(ck);
            else
                nUnique = nUnique + 1;
                keyToIdx(ck) = int32(nUnique);
                rowToUniqueIdx(ii) = nUnique;
            end
        end

        % Build (r-1, nUnique) query matrix from unique intervals.
        queryMat = zeros(r - 1, nUnique);
        seen = false(nUnique, 1);
        for ii = 1:numel(rowsInGroup)
            uIdx = rowToUniqueIdx(ii);
            if ~seen(uIdx)
                queryMat(:, uIdx) = groupRowIntervals{ii};
                seen(uIdx) = true;
            end
        end

        % Build harmonic template once per group, then ONE batched call
        % to localTensorHarmonicityOrbit (which wraps mobius.evalOrbitRel
        % and applies normalisation). Same FP path as scalar mode, so
        % batched values match scalar values to machine precision.
        [tmpl_p, tmpl_w] = addSpectra(zeros(dup, 1), ones(dup, 1), ...
                                       specArgs{:});
        vals = localTensorHarmonicityOrbit( ...
            tmpl_p, tmpl_w, sigma, r, queryMat, normalize);
        vals = vals(:);

        % Distribute back to rows.
        for ii = 1:numel(rowsInGroup)
            h(rowsInGroup(ii)) = vals(rowToUniqueIdx(ii));
        end
    end
end


