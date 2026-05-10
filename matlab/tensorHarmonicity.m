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
%   v2.2: per-row chord-canonical caching of the orbit-evaluated value
%   (rows whose canonical (sorted, translation-removed) pitch sequences
%   coincide share a single computation), plus per-dup template caching
%   (chords of equal cardinality reuse one harmonic-template build).

    nRows = size(P, 1);
    h = nan(nRows, 1);

    haveRowWeights = ~isempty(W) && isequal(size(W), size(P));
    if ~isempty(W) && ~haveRowWeights
        if isvector(W) && numel(W) == size(P, 2)
            W_broadcast = W(:).';
        else
            error('tensorHarmonicity:weightShape', ...
                ['In batched mode, w must be empty, a matrix the same size as p, ' ...
                 'or a vector matching the number of pitch columns.']);
        end
    end

    % Force inner scalar calls (warmup phase) to be silent regardless of
    % nvArgs.verbose; we print one batched estimate at the top, not per-row.
    nvArgsInner = nvArgs;
    nvArgsInner.verbose = false;
    nvPairs = localPackTensorNV(nvArgsInner);
    specArgs = nvArgs.spectrum;
    normalize = char(nvArgs.normalize);
    duplicateOpt = nvArgs.duplicate;

    % Up-front time estimate (printed once). Empirical calibration via
    % a uniformly-sampled subset of K rows, with one warm-up call to
    % absorb first-call overhead. See templateHarmonicity for rationale.
    % Calibration runs scalar (no cache), so the estimate is an upper
    % bound when the result cache absorbs repeated canonical chords.
    if nvArgs.verbose && nRows > 1
        nCal = min(10, nRows);
        sampleIdx = unique(round(linspace(1, nRows, nCal)));

        % Warm-up: run the first valid sample once, untimed.
        warmupDone = false;
        for s = 1:numel(sampleIdx)
            sIdx = sampleIdx(s);
            pRowS = P(sIdx, :);
            validS = ~isnan(pRowS);
            pValidS = pRowS(validS);
            if numel(pValidS) < 2
                continue;
            end
            if haveRowWeights
                wValidS = W(sIdx, validS);
            elseif ~isempty(W)
                wValidS = W_broadcast(validS);
            else
                wValidS = [];
            end
            tensorHarmonicity(pValidS(:), wValidS(:), sigma, nvPairs{:});
            warmupDone = true;
            break;
        end

        if warmupDone
            tCalStart = tic;
            nValidCal = 0;
            for s = 1:numel(sampleIdx)
                sIdx = sampleIdx(s);
                pRowS = P(sIdx, :);
                validS = ~isnan(pRowS);
                pValidS = pRowS(validS);
                if numel(pValidS) < 2
                    continue;
                end
                if haveRowWeights
                    wValidS = W(sIdx, validS);
                elseif ~isempty(W)
                    wValidS = W_broadcast(validS);
                else
                    wValidS = [];
                end
                tensorHarmonicity(pValidS(:), wValidS(:), sigma, nvPairs{:});
                nValidCal = nValidCal + 1;
            end
            if nValidCal > 0
                tCalTotal = toc(tCalStart);
                tPerRow   = tCalTotal / nValidCal;
                estTotal  = tCalTotal + tPerRow * nRows;
                printBatchedEstimate('tensorHarmonicity', nRows, estTotal);
            end
        end
    end

    % --- Per-row main loop with caching ---
    %   resultCache: canonical-key -> harmonicity value
    %   templateCache: dup -> {tmpl_p, tmpl_w}
    % The orbit point evaluator depends only on (tmpl_p, tmpl_w, sigma,
    % r) and the chord's interval vector; the chord enters as the query
    % and not into the template, so the template is keyed by dup alone.
    resultCache   = containers.Map('KeyType', 'char', 'ValueType', 'double');
    templateCache = containers.Map('KeyType', 'int32', 'ValueType', 'any');

    for k = 1:nRows
        pRow = P(k, :);
        validMask = ~isnan(pRow);
        pK = pRow(validMask);
        if numel(pK) < 2
            h(k) = NaN;
            continue;
        end

        nP = numel(pK);
        if duplicateOpt == 0
            dup = nP;
        else
            dup = duplicateOpt;
            if dup > 3
                warning('tensorHarmonicity:largeDuplicate', ...
                    ['duplicate = %d: computation time grows rapidly ' ...
                     'with duplication. Consider reducing to 3 or fewer.'], ...
                    dup);
            end
        end

        % Canonical chord key (rel mode, non-periodic): sort, subtract
        % min. Weights are not used by the orbit eval (chord enters as
        % a query, not into the template), so they are not part of the
        % key — same canonical pitch sequence -> same result regardless
        % of any chord-side w.
        pSorted = sort(pK(:));
        pCanon = pSorted - pSorted(1);
        intervals = pCanon(2:end);   % (nP-1) x 1

        key = sprintf('p=%s|s=%.12g|r=%d|d=%d|n=%s', ...
            mat2str(pCanon, 12), sigma, nP, dup, normalize);

        if isKey(resultCache, key)
            h(k) = resultCache(key);
            continue;
        end

        % Build / fetch the harmonic-series template.
        dupKey = int32(dup);
        if isKey(templateCache, dupKey)
            tmplPair = templateCache(dupKey);
            tmpl_p = tmplPair{1};
            tmpl_w = tmplPair{2};
        else
            [tmpl_p, tmpl_w] = addSpectra(zeros(dup, 1), ones(dup, 1), ...
                                            specArgs{:});
            templateCache(dupKey) = {tmpl_p, tmpl_w};
        end

        h_vec = localTensorHarmonicityOrbit( ...
            tmpl_p, tmpl_w, sigma, nP, intervals, normalize);
        h(k) = h_vec(1);
        resultCache(key) = h(k);
    end
end


function nvPairs = localPackTensorNV(nvArgs)
    nvPairs = {};
    fns = fieldnames(nvArgs);
    for i = 1:numel(fns)
        nvPairs = [nvPairs, {fns{i}, nvArgs.(fns{i})}]; %#ok<AGROW>
    end
end
