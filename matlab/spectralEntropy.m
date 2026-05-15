function H = spectralEntropy(p, w, sigma, nvArgs)
%SPECTRALENTROPY Spectral entropy of a weighted pitch multiset.
%
%   H = spectralEntropy(p, w, sigma)
%   H = spectralEntropy(p, w, sigma, Name, Value)
%
%   Returns the entropy of the smoothed composite spectrum of a
%   weighted pitch multiset, used as a consonance measure: the greater
%   the overlap of partials (after Gaussian smoothing for perceptual
%   uncertainty), the lower the entropy. Lower entropy therefore
%   indicates greater consonance.
%
%   spectralEntropy is a thin wrapper around entropyExpTens with
%   r = 1, isRel = false, isPer = false (1-D absolute non-periodic
%   density). It applies addSpectra to enrich the pitches with
%   partials (if a 'spectrum' argument is supplied), shifts the
%   lowest pitch to 0, computes appropriate grid bounds, and
%   delegates the entropy computation. Two methods are supported:
%
%     method='shannon' (default) computes the discrete Shannon
%       entropy of the density evaluated on a regular grid, normalised
%       to [0, 1] by log_base(N) when normalize=true (the default).
%
%     method='renyi2' computes the analytical (grid-independent)
%       Rényi-2 / collision entropy via the inner-product / Möbius
%       machinery used by entropyExpTens. normalize=true is not
%       supported under renyi2 (the analytical form has no natural
%       [0, 1] reference); pass normalize=false.
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
%     'method'     — 'shannon' (default) or 'renyi2'. See above.
%     'normalize'  — Logical (default: true). Shannon only: divides
%                    the entropy by log_base(N) to give a value in
%                    [0, 1] that is independent of grid resolution.
%                    Requesting 'renyi2' with normalize=true errors.
%     'base'       — Logarithm base for entropy (default: 2, giving
%                    bits). When 'normalize' is true, the base cancels
%                    and has no effect on the result.
%
%   Grid resolution (Shannon path) is the entropyExpTens default
%   (nPointsPerDim = 1200 over [0, max(spec_p) + 4*sigma]). Users
%   needing finer control should call entropyExpTens directly with a
%   pre-built density and their own nPointsPerDim / gridLimit.
%
%   Output:
%     H     — Spectral entropy. Scalar for a single chord, nRows-by-1
%             vector for a batched input. When method='shannon' and
%             normalize=true (defaults), H is in [0, 1] with lower
%             values indicating greater consonance.
%
%   Examples:
%     % Shannon entropy of a JI major triad (with harmonic spectra)
%     H = spectralEntropy([0, 386.31, 701.96], [], 12, ...
%                         'spectrum', {'harmonic', 24, 'powerlaw', 1})
%
%     % Rényi-2 of the same chord (must pass normalize=false)
%     H = spectralEntropy([0, 386.31, 701.96], [], 12, ...
%                         'spectrum', {'harmonic', 24, 'powerlaw', 1}, ...
%                         'method', 'renyi2', 'normalize', false)
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
        nvArgs.method (1,:) char ...
            {mustBeMember(nvArgs.method, {'shannon','renyi2'})} = 'shannon'
        nvArgs.normalize (1,1) logical = true
        nvArgs.base (1,1) {mustBePositive} = 2
        nvArgs.truncationSigmas (1,1) double {mustBePositive} ...
            = mptDefaults('truncationSigmas')
        nvArgs.kernelPrecision (1,:) char ...
            {mustBeMember(nvArgs.kernelPrecision, {'double','single'})} ...
            = mptDefaults('kernelPrecision')
        nvArgs.verbose (1,1) logical = true
    end

    % Top-level call guard: see internal.dispatchScope.
    guard = internal.dispatchScope(); %#ok<NASGU>

    % renyi2 + normalize=true is not implementable (no natural [0,1]
    % reference for the analytical form). Mirror entropyExpTens's
    % constraint upfront with a spectralEntropy-specific identifier
    % so the user sees the API surface they invoked.
    if strcmp(nvArgs.method, 'renyi2') && nvArgs.normalize
        error('spectralEntropy:renyi2NormalizeNotSupported', ...
            ['method=''renyi2'' with normalize=true is not implemented. ' ...
             'The analytical Rényi-2 entropy has no natural [0, 1] ' ...
             'reference (unlike Shannon, which normalises by ' ...
             'log_b(N) on the grid). Pass normalize=false to use ' ...
             'this method.']);
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
    % We apply it here (rather than via entropyExpTens's own 'spectrum'
    % kwarg) so we can compute the grid bounds from spec_p, which only
    % exists after addSpectra. Passing 'spectrum' to entropyExpTens
    % would require us to know xMax up front, which we don't.
    if isempty(specArgs)
        spec_p = p;
        spec_w = w;
    else
        [spec_p, spec_w] = addSpectra(p, w, specArgs{:});
    end

    % Up-front time estimate: dominated by the kernel-pair work in
    % evalExpTens (Shannon path) — numel(spec_p) * numel(grid).
    % renyi2 is analytical (no grid), so the estimate is irrelevant
    % there; emit only for the shannon path. Use the entropyExpTens
    % default grid size (1200 points per dim) for the estimate.
    if strcmp(nvArgs.method, 'shannon')
        nGrid = 1200;  % matches entropyExpTens default nPointsPerDim
        nPairs = double(numel(spec_p)) * double(nGrid);
        estimateCompTime(nPairs, 1, 'spectralEntropy', nvArgs.verbose);
    end

    H = localSpectralEntropyDelegate(spec_p, spec_w, sigma, nvArgs);
end


% =====================================================================
%  Local helper: delegate the entropy computation to entropyExpTens.
% =====================================================================

function H = localSpectralEntropyDelegate(spec_p, spec_w, sigma, nvArgs)
%LOCALSPECTRALENTROPYDELEGATE  Delegate to entropyExpTens.
%
%   Used by both the scalar path (called directly after spec_p, spec_w
%   are prepared) and by the batched per-row path (called once per
%   unique canonical chord via the row loop).
%
%   For method='shannon', passes only the non-periodic grid bounds
%   (xMin = 0, xMax = max(spec_p) + 4*sigma) and lets entropyExpTens
%   use its default nPointsPerDim. Users who want finer or coarser
%   grid control should call entropyExpTens directly with a pre-built
%   density.
%
%   For method='renyi2', entropyExpTens uses the analytical inner-
%   product form — no grid involved.

    if strcmp(nvArgs.method, 'renyi2')
        H = entropyExpTens(spec_p, spec_w, sigma, 1, false, false, 1200, ...
            'method', 'renyi2', ...
            'normalize', nvArgs.normalize, ...
            'base', nvArgs.base, ...
            'truncationSigmas', nvArgs.truncationSigmas, ...
            'kernelPrecision', nvArgs.kernelPrecision, ...
            'verbose', false);
        return;
    end

    margin = 4 * sigma;
    xMax = max(spec_p) + margin;

    H = entropyExpTens(spec_p, spec_w, sigma, 1, false, false, 1200, ...
        'method', 'shannon', ...
        'normalize', nvArgs.normalize, ...
        'base', nvArgs.base, ...
        'xMin', 0, ...
        'xMax', xMax, ...
        'truncationSigmas', nvArgs.truncationSigmas, ...
        'kernelPrecision', nvArgs.kernelPrecision, ...
        'verbose', false);
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
%   Spectral entropy has no fixed template to lift (each row's
%   spec_p depends on the input), but structurally-identical canonical
%   chords (under permutation + transposition) share a single cached
%   result via the canonical key from internal.chordCacheKey. For
%   batches with repeated chord shapes the per-row cost collapses to
%   a hash lookup.

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

    % --- Up-front time estimate (shannon path only; renyi2 is analytical) ---
    % Adaptive progress-print state. Defaults: silent.
    progStride = 1;
    showProgress = false;
    if strcmp(nvArgs.method, 'shannon') && nvArgs.verbose && nRows > 1
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
            localBatchEvalOneSE(pValidS, wValidS, sigma, nvArgs);
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
                localBatchEvalOneSE(pValidS, wValidS, sigma, nvArgs);
                nValidCal = nValidCal + 1;
            end
            if nValidCal > 0
                tCalTotal = toc(tCalStart);
                tPerRow   = tCalTotal / nValidCal;
                estTotal  = tCalTotal + tPerRow * nRows;
                internal.printBatchedEstimate('spectralEntropy', nRows, estTotal);
                progStride = internal.progressStride(tPerRow);
                showProgress = estTotal >= 5;
            end
        end
    end

    % --- Main loop with canonical-key cache ----------------------
    % spectralEntropy is invariant under joint transposition (lowest
    % pitch shifted to 0 internally), so the canonical key uses
    % (isRel=true, isPer=false).
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

        key = internal.chordCacheKey(pK(:), wK(:), sigma, ...
            1, true, false, 1200);

        if isKey(resultCache, key)
            H(k) = resultCache(key);
        else
            Hk = localBatchEvalOneSE(pK, wK, sigma, nvArgs);
            H(k) = Hk;
            resultCache(key) = Hk;
        end

        if nvArgs.verbose && showProgress ...
                && (mod(k, progStride) == 0 || k == nRows)
            fprintf('  %d / %d rows computed.\n', k, nRows);
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


function H = localBatchEvalOneSE(pValid, wValid, sigma, nvArgs)
%LOCALBATCHEVALONESE Apply transposition + spectrum and delegate.

    pShifted = pValid(:) - min(pValid);
    specArgs = nvArgs.spectrum;
    if isempty(specArgs)
        spec_p = pShifted;
        spec_w = wValid(:);
    else
        [spec_p, spec_w] = addSpectra(pShifted, wValid(:), specArgs{:});
    end

    H = localSpectralEntropyDelegate(spec_p, spec_w, sigma, nvArgs);
end
