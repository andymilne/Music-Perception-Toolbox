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
%   lowest pitch to 0, computes appropriate grid bounds where needed,
%   and delegates the entropy computation. Four methods are supported:
%
%     method='differential' (default) computes the adaptive
%       differential entropy h_hat; grid-independent and the
%       principled scale-free choice. Lower h_hat -> more consonant.
%       Note: adaptive convergence (nested-grid doubling to a
%       truncation-sigma-anchored tolerance) costs several discrete
%       passes per call --- typically 10-30x the cost of method=
%       'normalized' on the same density at the default
%       truncationSigmas (~ 6). Passing 'truncationSigmas', 3 loosens
%       the convergence tolerance to exp(-9/2) ~= 1.1e-2 and brings
%       differential to comparable cost to the discrete methods, at
%       the price of fifth-decimal drift in the returned value
%       (consonance ordering is preserved). For consonance comparisons
%       across many chords, prefer 'normalized' (faster and the
%       method established in the consonance literature).
%
%     method='normalized' (alias 'normalised') computes the Pielou-
%       style ratio H / log_b(N) in [0, 1]. Reproduces the values
%       reported in Milne et al. (2017) and Smit et al. (2019).
%       Computed on an explicit grid of nPointsPerDim = 1200 over
%       [0, max(spec_p) + 4*sigma].
%
%     method='shannon' computes the raw discrete Shannon entropy
%       H = -sum q log_b q on the same grid as 'normalized'.
%
%     method='renyi2' computes the analytical (grid-independent)
%       Rényi-2 / collision entropy via the inner-product / Möbius
%       machinery used by entropyExpTens.
%
%   v2.2 breaking change: the legacy 'normalize' boolean kwarg has
%   been removed from spectralEntropy. Use method='normalized' for
%   the v2.1 default behaviour (H/log_b(N) in [0, 1]) or
%   method='shannon' for raw H. Passing 'normalize' raises a
%   migration-error exception.
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
%     'method'     — One of {'differential' (default), 'normalized',
%                    'shannon', 'renyi2'} (or the British alias
%                    'normalised'). See above.
%     'base'       — Logarithm base for entropy (default: 2, giving
%                    bits). The base cancels for method='normalized'.
%     'truncationSigmas' — Numeric scalar or []. Override the toolbox-
%                    wide mptDefaults('truncationSigmas') setting for
%                    this call. Passes through to the entropyExpTens
%                    kernel evaluator; skips Gaussian contributions
%                    whose centre-to-query distance exceeds k*sigma.
%                    For method='differential' this also anchors the
%                    convergence tolerance --- 'truncationSigmas', 3
%                    is the recommended fast-path setting (see method
%                    description above). [] (default) means use the
%                    global default (factory: Inf).
%     'kernelPrecision' — 'double', 'single', or [] for the global
%                    default. Override the toolbox-wide
%                    kernelPrecision setting for this call. Passes
%                    through to the Shannon-path kernel evaluator;
%                    'single' casts the kernel matrix to float32 for
%                    a ~2x speedup at ~7 sig fig precision.
%     'verbose'    — Logical (default: true). If false, suppresses
%                    console output (time estimates, progress
%                    messages).
%
%   Grid resolution for the discrete paths is fixed at
%   nPointsPerDim = 1200 over [0, max(spec_p) + 4*sigma]. Users
%   needing finer control should call entropyExpTens directly with a
%   pre-built density and their own nPointsPerDim / gridLimit.
%
%   Output:
%     H     — Spectral entropy. Scalar for a single chord, nRows-by-1
%             vector for a batched input. Under method='normalized'
%             (or 'shannon' divided by log_b(N) externally), lower
%             values indicate greater consonance.
%
%   Examples:
%     % JI major triad with harmonic spectra, default method
%     % (differential, scale-free).
%     H = spectralEntropy([0, 386.31, 701.96], [], 12, ...
%                         'spectrum', {'harmonic', 24, 'powerlaw', 1})
%
%     % Rényi-2 of the same chord --- closed-form, no grid.
%     H = spectralEntropy([0, 386.31, 701.96], [], 12, ...
%                         'spectrum', {'harmonic', 24, 'powerlaw', 1}, ...
%                         'method', 'renyi2')
%
%     % Differential with truncationSigmas=3 for fast batched runs.
%     H = spectralEntropy(chordBatch, [], 12, ...
%                         'spectrum', {'harmonic', 24, 'powerlaw', 1}, ...
%                         'method', 'differential', ...
%                         'truncationSigmas', 3)
%
%     % Reproduce Smit et al. (2019) / Milne et al. (2017) values.
%     H = spectralEntropy(chord, [], 12, ...
%                         'spectrum', {'harmonic', 24, 'powerlaw', 1}, ...
%                         'method', 'normalized')
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
            {mustBeMember(nvArgs.method, ...
                {'differential','shannon','normalized','normalised','renyi2'})} ...
            = 'differential'
        nvArgs.base (1,1) {mustBePositive} = 2
        nvArgs.truncationSigmas (1,1) double {mustBePositive} ...
            = mptDefaults('truncationSigmas')
        nvArgs.kernelPrecision (1,:) char ...
            {mustBeMember(nvArgs.kernelPrecision, {'double','single'})} ...
            = mptDefaults('kernelPrecision')
        nvArgs.verbose (1,1) logical = true
        nvArgs.normalize = []  % v2.2 sentinel: any value triggers migration error
    end

    % Top-level call guard: see internal.dispatchScope.
    guard = internal.dispatchScope(); %#ok<NASGU>

    % Detect the legacy 'normalize' kwarg (removed in v2.2). The empty
    % default cannot be supplied by a caller; any value here means the
    % user explicitly passed 'normalize', ...  We emit a migration
    % error pointing to the four-method API.
    if ~isempty(nvArgs.normalize)
        error('spectralEntropy:normalizeRemoved', ...
              ['spectralEntropy: the ''normalize'' kwarg has been ' ...
               'removed in v2.2. Use method=''normalized'' for ' ...
               'H/log_b(N) in [0, 1] (the v2.1 default behaviour), ' ...
               'or method=''shannon'' for raw H = -sum q log_b q. ' ...
               'method=''differential'' and method=''renyi2'' are ' ...
               'continuous-form entropies and have no [0, 1] reference.']);
    end
    % Remove the sentinel field before passing nvArgs onward, so the
    % internal helpers do not see it.
    nvArgs = rmfield(nvArgs, 'normalize');

    % Canonicalise the British 'normalised' alias to 'normalized'.
    if strcmp(nvArgs.method, 'normalised')
        nvArgs.method = 'normalized';
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
    % evalExpTens (Shannon path) — numel(spec_p) * numel(grid). The
    % grid-free methods (differential, renyi2) bypass this; emit only
    % for the discrete (shannon, normalized) paths. Use the same
    % nPointsPerDim=1200 the delegate will pass downstream.
    if any(strcmp(nvArgs.method, {'shannon', 'normalized'}))
        nGrid = 1200;
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
%   For 'shannon' and 'normalized', passes explicit non-periodic grid
%   bounds (xMin = 0, xMax = max(spec_p) + 4*sigma) and an explicit
%   nPointsPerDim = 1200 (matching the toolbox's pre-v2.2 default).
%   For 'differential', the span auto-derives from event centres
%   +/- truncationSigmas * sigma and the grid is refined adaptively.
%   For 'renyi2', no grid is constructed (analytical inner-product
%   form).
%
%   Used by both the scalar path (called directly after spec_p,
%   spec_w are prepared) and by the batched per-row path (called once
%   per unique canonical chord via the row loop).

    if strcmp(nvArgs.method, 'renyi2')
        H = entropyExpTens(spec_p, spec_w, sigma, 1, false, false, 1200, ...
            'method', 'renyi2', ...
            'base', nvArgs.base, ...
            'truncationSigmas', nvArgs.truncationSigmas, ...
            'kernelPrecision', nvArgs.kernelPrecision, ...
            'verbose', false);
        return;
    end

    if strcmp(nvArgs.method, 'differential')
        H = entropyExpTens(spec_p, spec_w, sigma, 1, false, false, 1200, ...
            'method', 'differential', ...
            'base', nvArgs.base, ...
            'truncationSigmas', nvArgs.truncationSigmas, ...
            'kernelPrecision', nvArgs.kernelPrecision, ...
            'verbose', false);
        return;
    end

    margin = 4 * sigma;
    xMax = max(spec_p) + margin;

    if strcmp(nvArgs.method, 'normalized')
        H = entropyExpTens(spec_p, spec_w, sigma, 1, false, false, 1200, ...
            'method', 'normalized', ...
            'base', nvArgs.base, ...
            'nPointsPerDim', 1200, ...
            'xMin', 0, ...
            'xMax', xMax, ...
            'truncationSigmas', nvArgs.truncationSigmas, ...
            'kernelPrecision', nvArgs.kernelPrecision, ...
            'verbose', false);
        return;
    end

    % method == 'shannon': raw discrete H = -sum q log_b q.
    H = entropyExpTens(spec_p, spec_w, sigma, 1, false, false, 1200, ...
        'method', 'shannon', ...
        'base', nvArgs.base, ...
        'nPointsPerDim', 1200, ...
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

    % --- Up-front time estimate and adaptive progress-print ---
    % Method-agnostic empirical calibration: time a small sample of
    % rows, extrapolate to estTotal, print an up-front estimate, and
    % stride the row-completion countdown to keep terminal noise
    % proportional to wall-clock time. The mechanism applies to all
    % four methods --- differential and the discrete methods benefit
    % most (renyi2 is usually fast enough that showProgress's >= 5 s
    % gate suppresses the countdown automatically).
    progStride = 1;
    showProgress = false;
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
