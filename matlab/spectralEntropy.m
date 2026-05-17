function H = spectralEntropy(p, w, sigma, nvArgs)
%SPECTRALENTROPY Spectral entropy of a weighted pitch multiset.
%
%   H = spectralEntropy(p, w, sigma)
%   H = spectralEntropy(p, w, sigma, Name, Value)
%
%   Returns the Shannon entropy of the smoothed composite spectrum of
%   a weighted pitch multiset, used as a consonance measure: the
%   greater the overlap of partials (after Gaussian smoothing for
%   perceptual uncertainty), the lower the entropy. Lower entropy
%   therefore indicates greater consonance.
%
%   spectralEntropy is a thin wrapper around entropyExpTens with
%   r = 1, isRel = false, isPer = false (1-D absolute non-periodic
%   density). It applies addSpectra to enrich the pitches with
%   partials (if a 'spectrum' argument is supplied), shifts the
%   lowest pitch to 0, computes appropriate grid bounds, and
%   delegates the entropy computation.
%
%   Inputs:
%     p     — Pitch values in cents (vector for one chord; nRows-by-K
%             matrix for a batch of nRows chords). Absolute pitches
%             (e.g., MIDI 60 = 6000 cents via convertPitch). The
%             function transposes internally so the lowest pitch is 0.
%     w     — Weights (same shape as p; vector matching K for a
%             column-broadcast batch input; or empty for all ones).
%     sigma — Gaussian smoothing width in cents. Models perceptual
%             uncertainty. Values of 6-15 are typical; 12 is a good
%             default.
%
%   Name-Value Arguments:
%     'spectrum'   — Cell array of arguments to pass to addSpectra
%                    (everything after p and w). Defines the harmonic
%                    content of each tone.
%                    Default: {} (no spectral enrichment — pitches and
%                    weights are used as given). This is appropriate
%                    when p and w already represent empirical spectral
%                    peaks (e.g., from audioPeaks).
%                    Examples:
%                      'spectrum', {'harmonic', 24, 'powerlaw', 1}
%                      'spectrum', {'harmonic', 12, 'geometric', 0.9}
%     'normalize'  — Logical (default: true). If true, divides the
%                    entropy by log_base(N) to give a value in [0, 1]
%                    that is independent of grid resolution.
%     'base'       — Logarithm base for entropy (default: 2, giving
%                    bits). When 'normalize' is true, the base cancels
%                    and has no effect on the result.
%     'resolution' — Grid spacing in cents (default: 1). Finer
%                    resolution improves accuracy but increases
%                    computation time. Internally translated to a
%                    matching nPointsPerDim on the delegate call.
%     'verbose'    — Logical (default: true). If false, suppresses
%                    console output (time estimates, progress
%                    messages).
%
%   Output:
%     H     — Spectral entropy (scalar for a single chord, nRows-by-1
%             vector for a batched input). When 'normalize' is true
%             (default), H is in [0, 1]. Lower values indicate greater
%             consonance (more spectral overlap).
%
%   Examples:
%     % Spectral entropy of a JI major triad (with harmonic spectra)
%     H = spectralEntropy([0, 386.31, 701.96], [], 12, ...
%                         'spectrum', {'harmonic', 24, 'powerlaw', 1})
%
%     % Compare JI vs 12-EDO
%     spec = {'harmonic', 24, 'powerlaw', 1};
%     H_ji  = spectralEntropy([0, 386.31, 701.96], [], 12, 'spectrum', spec)
%     H_edo = spectralEntropy([0, 400, 700], [], 12, 'spectrum', spec)
%
%     % Empirical peaks (no spectral enrichment — the default)
%     [f, w] = audioPeaks('audio/piano_Cmin_open.wav');
%     p = convertPitch(f, 'hz', 'cents');
%     H = spectralEntropy(p, w, 12)
%
%     % Unnormalised entropy in bits
%     H = spectralEntropy([0, 400, 700], [], 12, 'normalize', false, ...
%                         'spectrum', {'harmonic', 24, 'powerlaw', 1})
%
%   References:
%     Milne, A. J., Bulger, D., & Herff, S. A. (2017). Exploring the
%       space of perfectly balanced rhythms and scales. Journal of
%       Mathematics and Music, 11(2-3), 101-133.
%     Smit, E. A., Milne, A. J., Dean, R. T., & Weidemann, G. (2019).
%       Perception of affect in unfamiliar musical chords. PLOS ONE,
%       14(6), e0218570.
%
%   See also ADDSPECTRA, BUILDEXPTENS, EVALEXPTENS, ENTROPYEXPTENS,
%            TEMPLATEHARMONICITY, TENSORHARMONICITY, ROUGHNESS.

    arguments
        p {mustBeNumeric}
        w {mustBeNumeric} = []
        sigma (1,1) {mustBePositive} = 12
        nvArgs.spectrum = {}
        nvArgs.normalize (1,1) logical = true
        nvArgs.base (1,1) {mustBePositive} = 2
        nvArgs.resolution (1,1) {mustBePositive} = 1
        nvArgs.verbose (1,1) logical = true
    end

    % --- Batched dispatch ---
    % If p is a 2-D matrix with both dimensions > 1, treat rows as
    % chords and return an nRows-by-1 column vector. NaN-padded rows
    % are accepted; rows with no valid pitches yield NaN.
    if size(p, 1) > 1 && size(p, 2) > 1
        H = localBatchedSpectralEntropy(p, w, sigma, nvArgs);
        return;
    end

    % Scalar path: force column vectors for consistency below.
    p = p(:);
    if ~isempty(w)
        w = w(:);
    end

    specArgs = nvArgs.spectrum;
    if ~iscell(specArgs)
        error('spectralEntropy:badSpectrum', ...
              '''spectrum'' value must be a cell array of addSpectra arguments.');
    end

    % === Weight defaults ===
    if isempty(w)
        w = ones(numel(p), 1);
    end
    if isscalar(w)
        if w == 0
            warning('All weights in w are zero.');
        end
        w = w * ones(numel(p), 1);
    end
    if numel(w) ~= numel(p)
        error('w must have the same number of entries as p (or be empty).');
    end

    % === Transpose so lowest pitch = 0 ===
    p = p - min(p);

    % === Apply addSpectra if requested ===
    % Applied here (rather than via entropyExpTens's own 'spectrum'
    % kwarg) so we can compute grid bounds from spec_p, which only
    % exists after addSpectra.
    if isempty(specArgs)
        spec_p = p;
        spec_w = w;
    else
        [spec_p, spec_w] = addSpectra(p, w, specArgs{:});
    end

    % Up-front time estimate (kernel cost in evalExpTens dominates).
    step    = nvArgs.resolution;
    margin  = 4 * sigma;
    nPoints = floor((max(spec_p) + margin) / step) + 1;
    nPairs  = double(numel(spec_p)) * double(nPoints);
    estimateCompTime(nPairs, 1, 'spectralEntropy', nvArgs.verbose);

    H = localSpectralEntropyDelegate(spec_p, spec_w, sigma, ...
                                     nvArgs.normalize, nvArgs.base, ...
                                     step, margin);
end


% =====================================================================
%  Local helper: delegate the entropy computation to entropyExpTens.
% =====================================================================

function H = localSpectralEntropyDelegate(spec_p, spec_w, sigma, ...
                                          normalize, base, step, margin)
%LOCALSPECTRALENTROPYDELEGATE  Delegate to entropyExpTens.
%
%   Used by both the scalar path and the batched per-row loop. Builds
%   no tensor itself — entropyExpTens handles buildExpTens / evalExpTens
%   / Shannon-entropy internally. We just compute the grid bounds and
%   the matching nPointsPerDim from 'resolution', so the discretisation
%   matches v2.1's `0:step:(max(spec_p) + 4*sigma)` colon-grid.
%
%   xMax is set to the actual last grid point of that colon-grid,
%   (nPoints - 1) * step, so that entropyExpTens's linspace
%   (xMin = 0, xMax, nPoints) reproduces the same grid exactly.

    nPoints     = floor((max(spec_p) + margin) / step) + 1;
    x_max_grid  = (nPoints - 1) * step;

    H = entropyExpTens(spec_p, spec_w, sigma, 1, false, false, 1200, ...
        'normalize',     normalize, ...
        'base',          base, ...
        'xMin',          0, ...
        'xMax',          x_max_grid, ...
        'nPointsPerDim', nPoints, ...
        'verbose',       false);
end


% =====================================================================
%  Unified dispatch helper: batched-raw mode.
% =====================================================================

function H = localBatchedSpectralEntropy(P, W, sigma, nvArgs)
%LOCALBATCHEDSPECTRALENTROPY Per-row spectral entropy from a 2-D matrix.
%
%   Returns an nRows-by-1 column vector. NaN-padded rows are handled
%   (NaN entries dropped per row); rows with fewer than 1 valid pitch
%   yield NaN.
%
%   Per-row dedup uses a canonical-form cache: spectralEntropy is
%   invariant under joint transposition (lowest pitch shifted to 0
%   internally) and under permutation of pitches, so structurally-
%   identical chords share a single cached result. The cache key is a
%   char-array built from (sort, subtract min, weights, sigma),
%   matching the wrapper-layer dedup style used elsewhere in v2.1
%   (cf. coherence, sameness, batchCosSimExpTens).

    nRows = size(P, 1);
    H = nan(nRows, 1);

    haveRowWeights = ~isempty(W) && isequal(size(W), size(P));
    W_broadcast = [];
    if ~isempty(W) && ~haveRowWeights
        if isvector(W) && numel(W) == size(P, 2)
            W_broadcast = W(:).';
        else
            error('spectralEntropy:weightShape', ...
                ['In batched mode, w must be empty, a matrix the same size as p, ' ...
                 'or a vector matching the number of pitch columns.']);
        end
    end

    specArgs = nvArgs.spectrum;
    if ~iscell(specArgs)
        error('spectralEntropy:badSpectrum', ...
              '''spectrum'' value must be a cell array of addSpectra arguments.');
    end

    step      = nvArgs.resolution;
    margin    = 4 * sigma;
    normalize = nvArgs.normalize;
    base      = nvArgs.base;

    % --- Up-front time estimate ---
    % Empirical calibration via the same code path the main loop pays:
    % localBatchEvalOneSE applies addSpectra and calls the delegate,
    % matching per-row work exactly. A warm-up call absorbs first-call
    % overhead (arguments-block parsing, JIT, persistent caches).
    if nvArgs.verbose && nRows > 1
        nCal = min(10, nRows);
        sampleIdx = unique(round(linspace(1, nRows, nCal)));

        warmupDone = false;
        for s = 1:numel(sampleIdx)
            sIdx = sampleIdx(s);
            pRowS = P(sIdx, :);
            validS = ~isnan(pRowS);
            pValidS = pRowS(validS);
            if numel(pValidS) < 1
                continue;
            end
            wValidS = localRowWeights(W, W_broadcast, sIdx, validS, ...
                haveRowWeights, pValidS);
            localBatchEvalOneSE(pValidS, wValidS, specArgs, sigma, ...
                step, margin, normalize, base);
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
                if numel(pValidS) < 1
                    continue;
                end
                wValidS = localRowWeights(W, W_broadcast, sIdx, validS, ...
                    haveRowWeights, pValidS);
                localBatchEvalOneSE(pValidS, wValidS, specArgs, sigma, ...
                    step, margin, normalize, base);
                nValidCal = nValidCal + 1;
            end
            if nValidCal > 0
                tCalTotal = toc(tCalStart);
                tPerRow   = tCalTotal / nValidCal;
                estTotal  = tCalTotal + tPerRow * nRows;
                printBatchedEstimate('spectralEntropy', nRows, estTotal);
            end
        end
    end

    % --- Main loop with canonical-key cache ---
    % spectralEntropy is invariant under joint transposition (lowest
    % pitch shifted to 0 inside the inner call) and under permutation
    % of pitches with their weights. We key on (sorted, transposed)
    % (p, w, sigma) — sigma included because the cache may be reused
    % across calls only within this batched invocation, but its
    % presence in the key documents the dependency.
    resultCache = containers.Map('KeyType', 'char', 'ValueType', 'any');

    for k = 1:nRows
        pRow = P(k, :);
        validMask = ~isnan(pRow);
        pK = pRow(validMask);
        if numel(pK) < 1
            continue;  % H(k) stays NaN
        end
        wK = localRowWeights(W, W_broadcast, k, validMask, ...
            haveRowWeights, pK);

        keyStr = localCanonicalKey(pK, wK, sigma);
        if isKey(resultCache, keyStr)
            H(k) = resultCache(keyStr);
        else
            Hk = localBatchEvalOneSE(pK, wK, specArgs, sigma, ...
                step, margin, normalize, base);
            resultCache(keyStr) = Hk;
            H(k) = Hk;
        end
    end
end


function w = localRowWeights(W, W_broadcast, rowIdx, validMask, haveRowWeights, pValid)
%LOCALROWWEIGHTS Resolve per-row weights from the batched W input.
    if haveRowWeights
        w = W(rowIdx, validMask);
    elseif ~isempty(W_broadcast)
        w = W_broadcast(validMask);
    else
        w = ones(1, numel(pValid));
    end
end


function H = localBatchEvalOneSE(pValid, wValid, specArgs, sigma, ...
                                  step, margin, normalize, base)
%LOCALBATCHEVALONESE Apply transposition + spectrum and delegate.
%   The per-row work the main loop pays (also used in calibration).

    pShifted = pValid(:) - min(pValid);
    if isempty(specArgs)
        spec_p = pShifted;
        spec_w = wValid(:);
    else
        [spec_p, spec_w] = addSpectra(pShifted, wValid(:), specArgs{:});
    end

    H = localSpectralEntropyDelegate(spec_p, spec_w, sigma, ...
                                     normalize, base, step, margin);
end


function keyStr = localCanonicalKey(p, w, sigma)
%LOCALCANONICALKEY Char-key for the (sort, transpose, permute)-canonical chord.
%
%   spectralEntropy shifts lowest pitch to 0 and is symmetric under
%   permutation of pitches with their weights. The canonical form is
%   the sorted, lowest-zero (p, w) pair. Sigma is included in the key
%   so caches are unambiguous if reused across configurations.
%
%   Values are formatted with %.12g, which separates entries that
%   differ in the 12th significant figure or earlier — well below any
%   musically meaningful precision but well above typical FP roundoff.

    [pSort, si] = sort(p(:).');
    pCan = pSort - pSort(1);
    if isempty(w)
        keyStr = sprintf('p:%.12g,', pCan);
        keyStr = [keyStr, sprintf('s:%.12g,', sigma)];
    else
        wSort = w(si);
        wCan  = wSort(:).';
        keyStr = sprintf('p:%.12g,', pCan);
        keyStr = [keyStr, sprintf('w:%.12g,', wCan)];
        keyStr = [keyStr, sprintf('s:%.12g,', sigma)];
    end
end
