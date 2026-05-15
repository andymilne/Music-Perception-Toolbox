function [hMax, hEntropy] = templateHarmonicity(p, w, sigma, nvArgs)
%TEMPLATEHARMONICITY Harmonicity via template cross-correlation.
%
%   hMax = templateHarmonicity(p, w, sigma)
%   [hMax, hEntropy] = templateHarmonicity(p, w, sigma)
%   [hMax, hEntropy] = templateHarmonicity(p, w, sigma, Name, Value)
%
%   For batched processing , p may also be a 2-D nRows-by-K
%   matrix with both dimensions > 1; rows are then treated as separate
%   multisets and the function returns hMax and hEntropy each as an
%   nRows-by-1 column vector. NaN-padded rows are accepted; rows with
%   fewer than 1 valid pitch return NaN. See the formulation note in
%   `template_harmonicity_formulation_review.md` for limitations of
%   the hMax and hEntropy measures (interval-multiset symmetry of
%   chord inversions, single-pitch / H1 hijacking at high spectrum
%   weights, high-harmonic-density hijacking at low rho with large N).
%
%   Measures the harmonicity of a weighted pitch multiset by
%   cross-correlating its spectral expectation tensor with a harmonic
%   template (a single complex tone with nHarm harmonics). Two
%   complementary measures are returned:
%
%     hMax     — Maximum of the normalized cross-correlation (Milne
%                2013). This is the cosine similarity between the
%                chord's spectrum and the template at the best-matching
%                transposition. Values range from 0 (no match) to 1
%                (perfect harmonic series). A multiset whose partials
%                align closely with a harmonic series at some
%                transposition will score high.
%
%     hEntropy — Normalized Shannon entropy of the cross-correlation
%                treated as a probability distribution (Harrison 2020).
%                A highly harmonic multiset produces a peaked
%                cross-correlation (low entropy); an inharmonic multiset
%                produces a flatter cross-correlation (high entropy).
%                By default, the entropy is normalized to [0, 1] by
%                dividing by log_base(N), removing the dependence on
%                the arbitrary grid resolution.
%
%   The procedure is:
%     1. Transpose the multiset so the lowest pitch is 0.
%     2. Build the template: add harmonics to a single pitch at 0
%        cents using the 'spectrum' parameters.
%     3. Build the chord spectrum: if 'chordSpectrum' is provided,
%        add harmonics to each chord pitch via addSpectra; otherwise
%        use the chord's pitches and weights as given (suitable for
%        empirical spectral peaks, e.g., from audioPeaks).
%     4. Evaluate both as 1-D absolute expectation tensors (r = 1,
%        isRel = false) on a fine grid.
%     5. Cross-correlate the two density vectors.
%     6. Normalize by the geometric mean of their energies (giving
%        cosine similarity at each lag).
%     7. Return the maximum (hMax) and the entropy (hEntropy).
%
%   Inputs:
%     p     — Pitch values in cents (vector). These are absolute
%             pitches (e.g., MIDI 60 = 6000 cents via convertPitch),
%             not pitch classes. The function transposes internally
%             so the lowest pitch is 0.
%     w     — Weights (vector same length as p, or empty for all ones).
%     sigma — Gaussian smoothing width in cents. Values of 9-15 are
%             typically effective; 12 is a good default.
%
%   Name-Value Arguments:
%     'spectrum'      — Cell array of arguments to pass to addSpectra
%                       for the harmonic template (everything after p
%                       and w). Defines the reference harmonic series
%                       against which the chord is compared.
%                       Default: {'harmonic', 36, 'powerlaw', 1}.
%                       Examples:
%                         'spectrum', {'harmonic', 24, 'powerlaw', 1}
%                         'spectrum', {'harmonic', 12, 'geometric', 0.9}
%                         'spectrum', {'stretched', 8, 1.02, 'powerlaw', 1}
%     'chordSpectrum' — Cell array of arguments to pass to addSpectra
%                       for the chord (everything after p and w).
%                       Default: {} (no enrichment — the chord's
%                       pitches and weights are used as given). This
%                       is appropriate when p and w already represent
%                       empirical spectral peaks (e.g., from
%                       audioPeaks). To apply the same spectral model
%                       as the template, pass the same arguments:
%                         'chordSpectrum', {'harmonic', 36, 'powerlaw', 1}
%     'normalize'     — Logical (default: true). If true, the entropy
%                       is divided by log_base(N) to give a value in
%                       [0, 1] that is independent of grid resolution.
%                       Only affects hEntropy.
%     'base'          — Logarithm base for entropy (default: 2, giving
%                       bits). When 'normalize' is true, the base
%                       cancels and has no effect on the result. Only
%                       affects hEntropy.
%     'resolution'    — Grid spacing in cents (default: 1). Finer
%                       resolution improves accuracy but increases
%                       computation time.
%
%   Outputs:
%     hMax     — Maximum normalized cross-correlation (Milne 2013).
%                Scalar in [0, 1].
%     hEntropy — Shannon entropy of the normalized cross-correlation
%                (Harrison 2020). When 'normalize' is true (default),
%                scalar in [0, 1]; when false, in units determined by
%                'base'.
%
%   Examples:
%     % Harmonicity of a JI major triad (synthetic spectrum on chord)
%     spec = {'harmonic', 36, 'powerlaw', 1};
%     [hMax, hEnt] = templateHarmonicity([0, 386.31, 701.96], [], 12, ...
%                        'chordSpectrum', spec)
%
%     % Same triad, chord pitches treated as raw spectral peaks
%     % (default: no chord enrichment)
%     [hMax, hEnt] = templateHarmonicity([0, 386.31, 701.96], [], 12)
%
%     % Empirical audio peaks
%     [f, w] = audioPeaks('audio/piano_Cmin_open.wav');
%     p = convertPitch(f, 'hz', 'cents');
%     [hMax, hEnt] = templateHarmonicity(p, w, 12)
%
%     % Custom template spectrum
%     [hMax, hEnt] = templateHarmonicity([0, 400, 700], [], 12, ...
%                        'spectrum', {'harmonic', 64, 'powerlaw', 2})
%
%     % Unnormalized entropy
%     [~, hEnt] = templateHarmonicity([0, 400, 700], [], 12, ...
%                     'normalize', false)
%
%   References:
%     Milne, A. J. (2013). A computational model of the cognition of
%       tonality. PhD thesis, The Open University.
%     Harrison, P. M. C. & Pearce, M. T. (2020). Simultaneous
%       consonance in music perception and composition. Psychological
%       Review, 127(2), 216-244.
%
%   See also ADDSPECTRA, BUILDEXPTENS, EVALEXPTENS, CONVERTPITCH,
%            ROUGHNESS, AUDIOPEAKS.

    arguments
        p {mustBeNumeric}
        w {mustBeNumeric} = []
        sigma (1,1) {mustBePositive} = 12
        nvArgs.spectrum = {'harmonic', 36, 'powerlaw', 1}
        nvArgs.chordSpectrum = {}
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

    % --- Batched dispatch ---
    % If p is a 2-D matrix with both dimensions > 1, treat rows as
    % multisets and return per-row hMax and hEntropy as column
    % vectors. NaN-padded rows are accepted; rows with fewer than 1
    % valid pitch return NaN.
    if size(p, 1) > 1 && size(p, 2) > 1
        [hMax, hEntropy] = localBatchedTemplateHarmonicity(p, w, sigma, nvArgs);
        return;
    end

    % Scalar path: force column vectors.
    p = p(:);
    if ~isempty(w)
        w = w(:);
    end

    specArgs      = nvArgs.spectrum;
    chordSpecArgs = nvArgs.chordSpectrum;
    step          = nvArgs.resolution;

    if ~iscell(specArgs)
        error('templateHarmonicity:badSpectrum', ...
              '''spectrum'' value must be a cell array of addSpectra arguments.');
    end
    if ~iscell(chordSpecArgs)
        error('templateHarmonicity:badChordSpectrum', ...
              '''chordSpectrum'' value must be a cell array of addSpectra arguments.');
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

    % === Transpose chord so lowest pitch = 0 ===

    p = p - min(p);

    % === Build template and chord spectra ===
    % The template is a single complex tone at 0 cents with harmonics
    % defined by the 'spectrum' parameter. The chord is either used
    % as-is (default) or enriched via addSpectra if 'chordSpectrum'
    % is provided.

    [tmpl_p, tmpl_w] = addSpectra(0, 1, specArgs{:});

    if isempty(chordSpecArgs)
        chord_p = p;
        chord_w = w;
    else
        [chord_p, chord_w] = addSpectra(p, w, chordSpecArgs{:});
    end

    % === Build template tensor and evaluate on grid ===
    % r = 1, isRel = false: intrinsic to the harmonicity definition
    % (1-D absolute density of spectral components).

    margin = 4 * sigma;
    x_tmpl  = 0:step:(max(tmpl_p) + margin);
    x_chord = 0:step:(max(chord_p) + margin);

    % Time estimate (kernel cost only; conv() and other overheads not
    % included, so this is a lower bound). Pair count is the sum of
    % the two evalExpTens workloads. dim = 1 since both densities use
    % r = 1, isRel = false.
    nPairs = double(numel(chord_p)) * double(numel(x_chord)) ...
           + double(numel(tmpl_p))  * double(numel(x_tmpl));
    estimateCompTime(nPairs, 1, 'templateHarmonicity', nvArgs.verbose);

    tmpl_dens = buildExpTens(tmpl_p, tmpl_w, sigma, 1, false, ...
        false, 1200, 'verbose', false);
    tmpl_vals = evalExpTens(tmpl_dens, x_tmpl, ...
        'truncationSigmas', nvArgs.truncationSigmas, ...
        'kernelPrecision', nvArgs.kernelPrecision, ...
        'verbose', false);
    tmpl_norm_sq = sum(tmpl_vals .^ 2);

    [hMax, hEntropy] = localTemplateChordOnly( ...
        chord_p, chord_w, sigma, ...
        tmpl_vals, tmpl_norm_sq, margin, step, ...
        nvArgs.normalize, nvArgs.base, ...
        nvArgs.truncationSigmas, nvArgs.kernelPrecision, ...
        nargout);

end


% =====================================================================
%  Local helper: chord-side evaluation given a pre-built template.
% =====================================================================

function [hMax, hEntropy] = localTemplateChordOnly( ...
    chord_p, chord_w, sigma, ...
    tmpl_vals, tmpl_norm_sq, margin, step, ...
    normalize, base, truncationSigmas, kernelPrecision, ...
    requestedNargout)
%LOCALTEMPLATECHORDONLY Chord-side: cross-correlation, hMax, hEntropy.
%
%   Used by the scalar path (which builds the template first, then
%   calls this) and by the batched path (which builds the template
%   once for the entire batch and calls this per unique canonical
%   chord). Hoisting the template build out of this function is what
%   lets the batched path avoid M template rebuilds.
%
%   The build-eval-conv-normalise core is shared with virtualPitches
%   via internal.templateXcorrChordSide; this wrapper adds
%   templateHarmonicity-specific postprocessing (max, optional
%   Harrison-2020 entropy of the profile).

    xcorr_norm = internal.templateXcorrChordSide( ...
        chord_p, chord_w, sigma, ...
        tmpl_vals, tmpl_norm_sq, margin, step, ...
        truncationSigmas, kernelPrecision);

    % Milne 2013: maximum normalized cross-correlation.
    hMax = max(xcorr_norm);

    % Harrison 2020: entropy of normalized cross-correlation. (The
    % profile is treated as a probability distribution; this is a
    % discrete-Shannon computation on a vector, not on an
    % expectation-tensor density, so it does not delegate to
    % entropyExpTens.)
    if requestedNargout > 1
        q = xcorr_norm(:);
        N = numel(q);       % total bins (before removing zeros)
        q = q / sum(q);     % normalize to probability distribution
        q(q <= 0) = [];     % apply 0*log(0) = 0 convention

        hEntropy = -sum(q .* (log(q) / log(base)));

        if normalize
            hEntropy = hEntropy / (log(N) / log(base));
        end
    else
        hEntropy = [];
    end
end

% =====================================================================
%  Unified dispatch helper: batched-raw mode.
% =====================================================================

function [hMax, hEntropy] = localBatchedTemplateHarmonicity(P, W, sigma, nvArgs)
%LOCALBATCHEDTEMPLATEHARMONICITY Per-row template harmonicity from a 2-D matrix.
%
%   Returns hMax and hEntropy as nRows-by-1 column vectors. NaN-padded
%   rows are handled (NaN entries dropped per row); rows with fewer
%   than 1 valid pitch yield NaN.
%
%   Applies the "build once, evaluate once" principle that
%   batchCosSimExpTens and tensor_harmonicity_batched also use:
%     - The harmonic template is built ONCE for the whole batch (it
%       depends only on (spectrum, sigma, resolution), not on the
%       chord), saving M template rebuilds compared to a recursive
%       scalar call.
%     - Structurally-identical chords (under permutation +
%       transposition) share a single cached result via the canonical
%       key from internal.chordCacheKey. For batches with repeated
%       chord shapes (typical of scale and progression sweeps), this
%       reduces per-row cost to a hash lookup.

    nRows = size(P, 1);
    hMax = nan(nRows, 1);
    hEntropy = nan(nRows, 1);

    haveRowWeights = ~isempty(W) && isequal(size(W), size(P));
    W_broadcast = [];
    if ~isempty(W) && ~haveRowWeights
        if isvector(W) && numel(W) == size(P, 2)
            W_broadcast = W(:).';
        else
            error('templateHarmonicity:weightShape', ...
                ['In batched mode, w must be empty, a matrix the same size as p, ' ...
                 'or a vector matching the number of pitch columns.']);
        end
    end

    specArgs      = nvArgs.spectrum;
    chordSpecArgs = nvArgs.chordSpectrum;
    step          = nvArgs.resolution;
    normalize     = nvArgs.normalize;
    base          = nvArgs.base;
    truncationSigmas = nvArgs.truncationSigmas;
    kernelPrecision  = nvArgs.kernelPrecision;

    if ~iscell(specArgs)
        error('templateHarmonicity:badSpectrum', ...
              '''spectrum'' value must be a cell array of addSpectra arguments.');
    end
    if ~iscell(chordSpecArgs)
        error('templateHarmonicity:badChordSpectrum', ...
              '''chordSpectrum'' value must be a cell array of addSpectra arguments.');
    end

    % --- Build template once for the whole batch -----------------
    [tmpl_p, tmpl_w] = addSpectra(0, 1, specArgs{:});
    margin = 4 * sigma;
    x_tmpl = 0:step:(max(tmpl_p) + margin);
    tmpl_dens = buildExpTens(tmpl_p, tmpl_w, sigma, 1, false, ...
        false, 1200, 'verbose', false);
    tmpl_vals = evalExpTens(tmpl_dens, x_tmpl, ...
        'truncationSigmas', truncationSigmas, ...
        'kernelPrecision', kernelPrecision, ...
        'verbose', false);
    tmpl_norm_sq = sum(tmpl_vals .^ 2);

    % --- Up-front time estimate ----------------------------------
    % The kernel-only nPairs-based estimate underestimates the actual
    % cost of templateHarmonicity batched runs by 3-5x because it
    % omits conv, addSpectra, and per-row loop overheads. So we run
    % a small empirical calibration: pick K rows spaced uniformly
    % across the input, time them via the same code path the main
    % loop uses (chord_spectrum + chord-side eval + xcorr + entropy
    % via localTemplateChordOnly, with the pre-built template),
    % and extrapolate.
    %
    % Crucially, calibration must match what the main loop pays per
    % row. The previous version routed through the scalar
    % templateHarmonicity entry, which rebuilt the template inside
    % each call — those rebuilds appeared in the calibration timing
    % but not in the main-loop work, biasing the printed estimate
    % upward (and contributing M rebuilds to the actual cost).
    if nvArgs.verbose && nRows > 1
        nCal = min(10, nRows);
        sampleIdx = unique(round(linspace(1, nRows, nCal)));

        % Warm-up.
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
            localBatchEvalOneChord(pValidS, wValidS, ...
                chordSpecArgs, sigma, ...
                tmpl_vals, tmpl_norm_sq, margin, step, ...
                normalize, base, truncationSigmas, kernelPrecision);
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
                localBatchEvalOneChord(pValidS, wValidS, ...
                    chordSpecArgs, sigma, ...
                    tmpl_vals, tmpl_norm_sq, margin, step, ...
                    normalize, base, truncationSigmas, kernelPrecision);
                nValidCal = nValidCal + 1;
            end
            if nValidCal > 0
                tCalTotal = toc(tCalStart);
                tPerRow   = tCalTotal / nValidCal;
                estTotal  = tCalTotal + tPerRow * nRows;
                printBatchedEstimate('templateHarmonicity', nRows, estTotal);
            end
        end
    end

    % --- Main loop with canonical-key cache ----------------------
    % Template-harmonicity is invariant under joint transposition
    % (lowest pitch is shifted to 0 internally), so the canonical key
    % uses (isRel=true, isPer=false). Structurally-identical chords
    % share one cached (hMax, hEntropy) result.
    resultCache = containers.Map('KeyType', 'char', 'ValueType', 'any');

    for k = 1:nRows
        pRow = P(k, :);
        validMask = ~isnan(pRow);
        pK = pRow(validMask);
        if numel(pK) < 1
            continue;  % hMax(k), hEntropy(k) stay NaN
        end
        wK = localRowWeights(W, W_broadcast, k, validMask, ...
            haveRowWeights, pK);

        key = internal.chordCacheKey(pK(:), wK(:), sigma, ...
            1, true, false, 1200);

        if isKey(resultCache, key)
            cached = resultCache(key);
            hMax(k) = cached(1);
            hEntropy(k) = cached(2);
        else
            [hMaxK, hEntK] = localBatchEvalOneChord( ...
                pK, wK, chordSpecArgs, sigma, ...
                tmpl_vals, tmpl_norm_sq, margin, step, ...
                normalize, base, truncationSigmas, kernelPrecision);
            hMax(k) = hMaxK;
            hEntropy(k) = hEntK;
            resultCache(key) = [hMaxK, hEntK];
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


function [hMaxK, hEntK] = localBatchEvalOneChord( ...
        pValid, wValid, chordSpecArgs, sigma, ...
        tmpl_vals, tmpl_norm_sq, margin, step, ...
        normalize, base, truncationSigmas, kernelPrecision)
%LOCALBATCHEVALONECHORD Apply chord_spectrum and call chord-only.
%   Used by both the calibration pass and the main loop so the timed
%   per-row work matches the per-row work the main loop pays.

    pShifted = pValid(:) - min(pValid);
    if isempty(chordSpecArgs)
        chord_p = pShifted;
        chord_w = wValid(:);
    else
        [chord_p, chord_w] = addSpectra(pShifted, wValid(:), chordSpecArgs{:});
    end

    [hMaxK, hEntK] = localTemplateChordOnly( ...
        chord_p, chord_w, sigma, ...
        tmpl_vals, tmpl_norm_sq, margin, step, ...
        normalize, base, truncationSigmas, kernelPrecision, ...
        2);  % always compute both outputs in batched mode
end
