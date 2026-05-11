function H = spectralEntropy(p, w, sigma, nvArgs)
%SPECTRALENTROPY Spectral entropy of a weighted pitch multiset.
%
%   H = spectralEntropy(p, w, sigma)
%   H = spectralEntropy(p, w, sigma, Name, Value)
%
%   Computes the Shannon entropy of the smoothed composite spectrum of
%   a weighted pitch multiset. The spectrum is constructed by adding
%   harmonics to each pitch via addSpectra, evaluating the resulting
%   1-D absolute non-periodic expectation tensor on a fine grid,
%   normalising to a probability distribution, and computing the entropy.
%
%   Spectral entropy aggregates the spectral pitch similarities of all
%   pairs of sounds in the multiset: the greater the overlap of
%   partials (after Gaussian smoothing for perceptual uncertainty),
%   the lower the entropy. Lower entropy therefore indicates greater
%   consonance.
%
%   The procedure is:
%     1. (Optional) Add harmonics to each pitch via addSpectra, if a
%        'spectrum' argument is provided. If omitted, the pitches and
%        weights are used as-is — suitable for empirical spectral
%        peaks (e.g., from audioPeaks).
%     2. Build a 1-D absolute expectation tensor (r = 1, isRel = false,
%        isPer = false) from the (enriched) spectrum.
%     3. Evaluate the tensor on a fine grid spanning the full range of
%        partials.
%     4. Normalise to a probability distribution and compute Shannon
%        entropy.
%
%   By default, the entropy is normalised to [0, 1] by dividing by
%   log_base(N), where N is the number of grid points. This removes
%   the dependence on the arbitrary grid resolution.
%
%   Inputs:
%     p     — Pitch values in cents (vector). These are absolute
%             pitches (e.g., MIDI 60 = 6000 cents via convertPitch),
%             not pitch classes. The function transposes internally
%             so the lowest pitch is 0.
%     w     — Weights (vector same length as p, or empty for all ones).
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
%                      'spectrum', {'harmonic', 64, 'powerlaw', 1}
%                      'spectrum', {'harmonic', 12, 'geometric', 0.9}
%     'normalize'  — Logical (default: true). If true, divides the
%                    entropy by log_base(N) to give a value in [0, 1]
%                    that is independent of grid resolution.
%     'base'       — Logarithm base for entropy (default: 2, giving
%                    bits). When 'normalize' is true, the base cancels
%                    and has no effect on the result.
%     'resolution' — Grid spacing in cents (default: 1). Finer
%                    resolution improves accuracy but increases
%                    computation time.
%
%   Output:
%     H     — Spectral entropy (scalar, non-negative). When
%             'normalize' is true (default), H is in [0, 1]. Lower
%             values indicate greater consonance (more spectral
%             overlap).
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
        nvArgs.truncationSigmas (1,1) double {mustBePositive} ...
            = mptDefaults('truncationSigmas')
        nvArgs.kernelPrecision (1,:) char ...
            {mustBeMember(nvArgs.kernelPrecision, {'double','single'})} ...
            = mptDefaults('kernelPrecision')
        nvArgs.verbose (1,1) logical = true
    end

    % --- Batched dispatch (v2.1+) ---
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
    step     = nvArgs.resolution;

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

    % === Build composite spectrum ===
    % If a 'spectrum' argument was provided, enrich pitches via
    % addSpectra; otherwise use the pitches and weights as given.

    if isempty(specArgs)
        spec_p = p;
        spec_w = w;
    else
        [spec_p, spec_w] = addSpectra(p, w, specArgs{:});
    end

    % === Build 1-D absolute non-periodic expectation tensor ===

    r     = 1;
    isRel = false;
    isPer = false;
    period = 1200;  % not used for wrapping; required by buildExpTens

    T = buildExpTens(spec_p, spec_w, sigma, r, isRel, isPer, period, ...
                     'verbose', false);

    % === Evaluate on a fine grid ===

    margin = 4 * sigma;
    x = 0:step:(max(spec_p) + margin);

    % Time estimate (kernel cost only; eval_exp_tens kernel pair count
    % is the dominant work for spectral entropy at typical scales).
    nPairs = double(numel(spec_p)) * double(numel(x));
    estimateCompTime(nPairs, 1, 'spectralEntropy', nvArgs.verbose);

    t = evalExpTens(T, x, ...
        'truncationSigmas', nvArgs.truncationSigmas, ...
        'kernelPrecision', nvArgs.kernelPrecision, ...
        'verbose', false);

    % === Normalise to probability distribution ===

    q = t(:) / sum(t(:));

    N = numel(q);  % total bins (before removing zeros)

    % Apply 0 * log(0) = 0 convention
    q(q == 0) = [];

    % === Shannon entropy ===

    H = -sum(q .* (log(q) / log(nvArgs.base)));

    if nvArgs.normalize
        H = H / (log(N) / log(nvArgs.base));
    end

end


% =====================================================================
%  v2.1 unified dispatch helper: batched-raw mode.
% =====================================================================

function H = localBatchedSpectralEntropy(P, W, sigma, nvArgs)
%LOCALBATCHEDSPECTRALENTROPY Per-row spectral entropy from a 2-D matrix.
%
%   Returns an nRows-by-1 column vector. NaN-padded rows are handled
%   (NaN entries dropped per row); rows with fewer than 1 valid pitch
%   yield NaN.

    nRows = size(P, 1);
    H = nan(nRows, 1);

    haveRowWeights = ~isempty(W) && isequal(size(W), size(P));
    if ~isempty(W) && ~haveRowWeights
        if isvector(W) && numel(W) == size(P, 2)
            W_broadcast = W(:).';
        else
            error('spectralEntropy:weightShape', ...
                ['In batched mode, w must be empty, a matrix the same size as p, ' ...
                 'or a vector matching the number of pitch columns.']);
        end
    end

    % Force inner scalar calls silent; one batched estimate at top.
    nvArgsInner = nvArgs;
    nvArgsInner.verbose = false;
    nvPairs = localPackSpectralEntropyNV(nvArgsInner);

    % Up-front time estimate (printed once). Empirical calibration via
    % a uniformly-sampled subset of K rows, with one warm-up call to
    % absorb first-call overhead.
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
            if haveRowWeights
                wValidS = W(sIdx, validS);
            elseif ~isempty(W)
                wValidS = W_broadcast(validS);
            else
                wValidS = [];
            end
            spectralEntropy(pValidS(:), wValidS(:), sigma, nvPairs{:});
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
                if haveRowWeights
                    wValidS = W(sIdx, validS);
                elseif ~isempty(W)
                    wValidS = W_broadcast(validS);
                else
                    wValidS = [];
                end
                spectralEntropy(pValidS(:), wValidS(:), sigma, nvPairs{:});
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

    for k = 1:nRows
        pRow = P(k, :);
        validMask = ~isnan(pRow);
        pK = pRow(validMask);
        if haveRowWeights
            wK = W(k, validMask);
        elseif ~isempty(W)
            wK = W_broadcast(validMask);
        else
            wK = [];
        end
        if numel(pK) < 1
            H(k) = NaN;
            continue;
        end
        H(k) = spectralEntropy(pK(:), wK(:), sigma, nvPairs{:});
    end
end


function nvPairs = localPackSpectralEntropyNV(nvArgs)
    nvPairs = {};
    fns = fieldnames(nvArgs);
    for i = 1:numel(fns)
        nvPairs = [nvPairs, {fns{i}, nvArgs.(fns{i})}]; %#ok<AGROW>
    end
end