function [vp_p, vp_w] = virtualPitches(p, w, sigma, nvArgs)
%VIRTUALPITCHES Virtual pitch salience profile via template cross-correlation.
%
%   [vp_p, vp_w] = virtualPitches(p, w, sigma)
%   [vp_p, vp_w] = virtualPitches(p, w, sigma, Name, Value)
%
%   For batched processing , p may also be a 2-D nRows-by-K
%   matrix with both dimensions > 1; rows are then treated as separate
%   multisets and the function returns vp_p and vp_w each as a
%   1-by-nRows cell array of column vectors. Profile lengths vary per
%   row (cross-correlation extent depends on the chord's pitch range
%   plus margin), so cell-of-arrays output is used rather than NaN-
%   padding to a common length. NaN-padded input rows are accepted;
%   rows with fewer than 1 valid pitch yield empty cell entries.
%
%   Computes the virtual pitch (fundamental) salience profile for a
%   weighted pitch multiset by cross-correlating its spectral expectation
%   tensor with a harmonic template. The result is a pitch-indexed vector
%   weights indicating how strongly each candidate fundamental is
%   supported by the input spectrum.
%
%   The normalized cross-correlation at each lag gives the cosine
%   similarity between the chord's spectrum and the harmonic template
%   at that transposition. A peak at pitch p0 means that a harmonic
%   series rooted at p0 matches the chord's spectrum well — i.e., p0
%   is a strong virtual pitch (fundamental) of the chord.
%
%   This function returns the full cross-correlation profile from
%   which templateHarmonicity extracts summary statistics (hMax is the
%   maximum of vp_w; hEntropy is the entropy of vp_w treated as a
%   probability distribution).
%
%   The procedure is:
%     1. Transpose the multiset so the lowest pitch is 0 (internal).
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
%     7. Map each lag to a pitch value in the input coordinate system.
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
%                       audioPeaks). To model each chord tone as a
%                       complex tone, pass the same (or different)
%                       spectral arguments:
%                         'chordSpectrum', {'harmonic', 36, 'powerlaw', 1}
%     'resolution'    — Grid spacing in cents (default: 1). Finer
%                       resolution improves pitch accuracy but
%                       increases computation time and output length.
%     'truncationSigmas' — Numeric scalar or []. Override the toolbox-
%                       wide mptDefaults('truncationSigmas') setting
%                       for this call. Passes through to the kernel
%                       evaluator on the centres path; skips Gaussian
%                       contributions whose centre-to-query distance
%                       exceeds k*sigma. [] (default) means use the
%                       global default (factory: Inf).
%     'kernelPrecision' — 'double', 'single', or [] for the global
%                       default. Override the toolbox-wide
%                       kernelPrecision setting for this call. 'single'
%                       casts the kernel matrix to float32 for a ~2x
%                       speedup at ~7 sig fig precision.
%     'verbose'       — Logical (default: true). If false, suppresses
%                       console output (time estimates, progress
%                       messages).
%
%   Outputs:
%     vp_p — Pitch values in cents (column vector), in the same
%            absolute coordinate system as the input p. Each value is
%            the candidate fundamental pitch for the corresponding
%            element of vp_w.
%     vp_w — Virtual pitch weights (column vector, same length as
%            vp_p). These are the normalized cross-correlation values
%            (cosine similarity at each lag), non-negative. The
%            maximum of vp_w equals templateHarmonicity's hMax output.
%
%   Examples:
%     % Virtual pitches of a JI major triad (synthetic spectrum)
%     spec = {'harmonic', 36, 'powerlaw', 1};
%     [vp_p, vp_w] = virtualPitches([0, 386.31, 701.96], [], 12, ...
%                         'chordSpectrum', spec);
%     plot(vp_p, vp_w)
%     xlabel('Pitch (cents)')
%     ylabel('Salience')
%
%     % Pitches as raw spectral peaks (no chord enrichment)
%     [vp_p, vp_w] = virtualPitches([0, 400, 700], [], 12)
%
%     % Empirical audio peaks
%     [f, a] = audioPeaks('audio/piano_Cmin_open.wav');
%     p_cents = convertPitch(f, 'hz', 'cents');
%     [vp_p, vp_w] = virtualPitches(p_cents, a, 12);
%
%     % MIDI input via convertPitch
%     p = convertPitch([60 64 67], 'midi', 'cents');
%     spec = {'harmonic', 36, 'powerlaw', 1};
%     [vp_p, vp_w] = virtualPitches(p, [], 12, 'chordSpectrum', spec);
%     % Plot with MIDI pitch axis
%     plot(convertPitch(vp_p, 'cents', 'midi'), vp_w)
%
%     % Verify consistency with templateHarmonicity
%     hMax = templateHarmonicity([0, 400, 700], [], 12, ...
%                'chordSpectrum', {'harmonic', 36, 'powerlaw', 1});
%     [~, vp_w] = virtualPitches([0, 400, 700], [], 12, ...
%                     'chordSpectrum', {'harmonic', 36, 'powerlaw', 1});
%     assert(abs(max(vp_w) - hMax) < 1e-10)
%
%   References:
%     Milne, A. J. (2013). A computational model of the cognition of
%       tonality. PhD thesis, The Open University.
%     Milne, A. J., Laney, R., & Sharp, D. B. (2016). Testing a
%       spectral model of tonal affinity with microtonal melodies and
%       inharmonic spectra. Musicae Scientiae, 20(4), 465-494.
%
%   See also TEMPLATEHARMONICITY, ADDSPECTRA, BUILDEXPTENS,
%            EVALEXPTENS, CONVERTPITCH, AUDIOPEAKS.

    arguments
        p {mustBeNumeric}
        w {mustBeNumeric} = []
        sigma (1,1) {mustBePositive} = 12
        nvArgs.spectrum = {'harmonic', 36, 'powerlaw', 1}
        nvArgs.chordSpectrum = {}
        nvArgs.resolution (1,1) {mustBePositive} = 1
        nvArgs.truncationSigmas (1,1) double {mustBePositive} ...
            = mptDefaults('truncationSigmas')
        nvArgs.kernelPrecision (1,:) char ...
            {mustBeMember(nvArgs.kernelPrecision, {'double','single'})} ...
            = mptDefaults('kernelPrecision')
        nvArgs.verbose (1,1) logical = true
    end

    % Top-level call guard: dispatch throttle + kernelChunkBytes pin. See internal.callGuard.
    guard = internal.callGuard(); %#ok<NASGU>

    % --- Batched dispatch ---
    % If p is a 2-D matrix with both dimensions > 1, treat rows as
    % multisets and return per-row vp_p and vp_w as 1-by-nRows cell
    % arrays of column vectors. Cell-of-arrays output is used because
    % the cross-correlation profile length varies per row (depending
    % on the chord's pitch range plus margin), so a numeric matrix
    % alignment is not natural. NaN-padded rows are accepted; rows
    % with fewer than 1 valid pitch yield empty entries.
    if size(p, 1) > 1 && size(p, 2) > 1
        [vp_p, vp_w] = localBatchedVirtualPitches(p, w, sigma, nvArgs);
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
        error('virtualPitches:badSpectrum', ...
              '''spectrum'' value must be a cell array of addSpectra arguments.');
    end
    if ~iscell(chordSpecArgs)
        error('virtualPitches:badChordSpectrum', ...
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
    % Record the offset to convert output pitches back to the input
    % coordinate system.

    pOffset = min(p);
    p = p - pOffset;

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
    % r = 1, isRel = false: intrinsic to the virtual-pitch definition
    % (1-D absolute density of spectral components).

    margin = 4 * sigma;
    x_tmpl  = 0:step:(max(tmpl_p) + margin);
    x_chord = 0:step:(max(chord_p) + margin);

    % Time estimate (kernel cost only; conv() and other overheads
    % not included, so this is a lower bound). Pair count is the sum
    % of the two evalExpTens workloads.
    nPairs = double(numel(chord_p)) * double(numel(x_chord)) ...
           + double(numel(tmpl_p))  * double(numel(x_tmpl));
    estimateCompTime(nPairs, 1, 'virtualPitches', nvArgs.verbose);

    tmpl_dens = buildExpTens(tmpl_p, tmpl_w, sigma, 1, false, ...
        false, 1200, 'verbose', false);
    tmpl_vals = evalExpTens(tmpl_dens, x_tmpl, ...
        'truncationSigmas', nvArgs.truncationSigmas, ...
        'kernelPrecision', nvArgs.kernelPrecision, ...
        'verbose', false);
    tmpl_norm_sq = sum(tmpl_vals .^ 2);
    N_tmpl = numel(tmpl_vals);

    [vp_w, N_xcorr] = localVPChordOnly( ...
        chord_p, chord_w, sigma, ...
        tmpl_vals, tmpl_norm_sq, margin, step, ...
        nvArgs.truncationSigmas, nvArgs.kernelPrecision);

    % Map lag indices to pitch values in the input coordinate system.
    lag_indices = (0:N_xcorr - 1)' - (N_tmpl - 1);
    vp_p = lag_indices * step + pOffset;

end


% =====================================================================
%  Local helper: chord-side evaluation given a pre-built template.
% =====================================================================

function [vp_w, N_xcorr] = localVPChordOnly( ...
    chord_p, chord_w, sigma, ...
    tmpl_vals, tmpl_norm_sq, margin, step, ...
    truncationSigmas, kernelPrecision)
%LOCALVPCHORDONLY Chord-side normalized cross-correlation.
%
%   Returns the offset-independent half-cosine-similarity profile vp_w
%   and the cross-correlation length N_xcorr. The caller is
%   responsible for reconstructing vp_p = (0:N_xcorr-1) - (N_tmpl-1)
%   in step units plus the per-row pitch offset; that arithmetic is
%   row-dependent and so is not part of what gets cached when this
%   helper is called from the batched path.
%
%   The build-eval-conv-normalise core is shared with
%   templateHarmonicity via internal.templateXcorrChordSide; this
%   wrapper exists only to extract N_xcorr alongside vp_w (the caller
%   needs the length to reconstruct vp_p).

    vp_w = internal.templateXcorrChordSide( ...
        chord_p, chord_w, sigma, ...
        tmpl_vals, tmpl_norm_sq, margin, step, ...
        truncationSigmas, kernelPrecision);
    N_xcorr = numel(vp_w);
end

% =====================================================================
%  Unified dispatch helper: batched-raw mode.
% =====================================================================

function [vp_p, vp_w] = localBatchedVirtualPitches(P, W, sigma, nvArgs)
%LOCALBATCHEDVIRTUALPITCHES Per-row virtual pitch profiles from a 2-D matrix.
%
%   Returns vp_p and vp_w each as 1-by-nRows cell arrays of column
%   vectors. Profile lengths vary per row (cross-correlation extent
%   depends on the chord's pitch range), so cell-of-arrays output is
%   used rather than NaN-padding to a common length. NaN-padded rows
%   are accepted; rows with fewer than 1 valid pitch yield empty
%   cell entries.
%
%   Applies the "build once, evaluate once" principle:
%     - The harmonic template is built ONCE for the whole batch
%       (depends only on (spectrum, sigma, resolution), not on the
%       chord), saving M template rebuilds.
%     - Structurally-identical canonical chords (under permutation +
%       transposition) share a cached cross-correlation profile; only
%       the per-row pitch offset is row-dependent and is reconstructed
%       outside the cache.

    nRows = size(P, 1);
    vp_p = cell(1, nRows);
    vp_w = cell(1, nRows);

    haveRowWeights = ~isempty(W) && isequal(size(W), size(P));
    W_broadcast = [];
    if ~isempty(W) && ~haveRowWeights
        if isvector(W) && numel(W) == size(P, 2)
            W_broadcast = W(:).';
        else
            error('virtualPitches:weightShape', ...
                ['In batched mode, w must be empty, a matrix the same size as p, ' ...
                 'or a vector matching the number of pitch columns.']);
        end
    end

    specArgs      = nvArgs.spectrum;
    chordSpecArgs = nvArgs.chordSpectrum;
    step          = nvArgs.resolution;
    truncationSigmas = nvArgs.truncationSigmas;
    kernelPrecision  = nvArgs.kernelPrecision;

    if ~iscell(specArgs)
        error('virtualPitches:badSpectrum', ...
              '''spectrum'' value must be a cell array of addSpectra arguments.');
    end
    if ~iscell(chordSpecArgs)
        error('virtualPitches:badChordSpectrum', ...
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
    N_tmpl = numel(tmpl_vals);

    % --- Up-front time estimate (matches main-loop cost) ---------
    % Adaptive progress-print state. Defaults: silent.
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
            localBatchEvalOneVP(pValidS, wValidS, ...
                chordSpecArgs, sigma, ...
                tmpl_vals, tmpl_norm_sq, margin, step, ...
                truncationSigmas, kernelPrecision);
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
                localBatchEvalOneVP(pValidS, wValidS, ...
                    chordSpecArgs, sigma, ...
                    tmpl_vals, tmpl_norm_sq, margin, step, ...
                    truncationSigmas, kernelPrecision);
                nValidCal = nValidCal + 1;
            end
            if nValidCal > 0
                tCalTotal = toc(tCalStart);
                tPerRow   = tCalTotal / nValidCal;
                estTotal  = tCalTotal + tPerRow * nRows;
                internal.printBatchedEstimate('virtualPitches', nRows, estTotal);
                progStride = internal.progressStride(tPerRow);
                showProgress = estTotal >= 5;
            end
        end
    end

    % --- Main loop with canonical-key cache ----------------------
    % virtualPitches is invariant under joint transposition up to a
    % shift of the output vp_p coordinate (rebuilt per row). The
    % offset-independent profile (vp_w_internal, N_xcorr) is cached;
    % vp_p reconstruction uses the per-row pOffset.
    resultCache = containers.Map('KeyType', 'char', 'ValueType', 'any');

    for k = 1:nRows
        pRow = P(k, :);
        validMask = ~isnan(pRow);
        pK = pRow(validMask);
        if numel(pK) < 1
            vp_p{k} = [];
            vp_w{k} = [];
            continue;
        end
        wK = localRowWeights(W, W_broadcast, k, validMask, ...
            haveRowWeights, pK);

        pOffset = min(pK);
        key = internal.chordCacheKey(pK(:), wK(:), sigma, ...
            1, true, false, 1200);

        if isKey(resultCache, key)
            cached = resultCache(key);
            vp_w_k = cached.vp_w;
            N_xcorr_k = cached.N_xcorr;
        else
            [vp_w_k, N_xcorr_k] = localBatchEvalOneVP( ...
                pK, wK, chordSpecArgs, sigma, ...
                tmpl_vals, tmpl_norm_sq, margin, step, ...
                truncationSigmas, kernelPrecision);
            resultCache(key) = struct('vp_w', vp_w_k, ...
                'N_xcorr', N_xcorr_k);
        end

        % Reconstruct vp_p in the input coordinate system.
        lag_indices = (0:N_xcorr_k - 1)' - (N_tmpl - 1);
        vp_p{k} = lag_indices * step + pOffset;
        vp_w{k} = vp_w_k;

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


function [vp_w, N_xcorr] = localBatchEvalOneVP( ...
        pValid, wValid, chordSpecArgs, sigma, ...
        tmpl_vals, tmpl_norm_sq, margin, step, ...
        truncationSigmas, kernelPrecision)
%LOCALBATCHEVALONEVP Apply chord_spectrum and call chord-only.

    pShifted = pValid(:) - min(pValid);
    if isempty(chordSpecArgs)
        chord_p = pShifted;
        chord_w = wValid(:);
    else
        [chord_p, chord_w] = addSpectra(pShifted, wValid(:), chordSpecArgs{:});
    end

    [vp_w, N_xcorr] = localVPChordOnly( ...
        chord_p, chord_w, sigma, ...
        tmpl_vals, tmpl_norm_sq, margin, step, ...
        truncationSigmas, kernelPrecision);
end
