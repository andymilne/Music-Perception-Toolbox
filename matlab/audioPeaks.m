function [f, w, detail] = audioPeaks(audioFile, nvArgs)
%AUDIOPEAKS Extract spectral peaks from an audio file.
%
%   [f, w] = audioPeaks(audioFile)
%   [f, w] = audioPeaks(audioFile, Name, Value)
%   [f, w, detail] = audioPeaks(...)
%
%   Reads an audio file, computes the magnitude spectrum, and extracts
%   peaks. The output frequencies (in Hz) and normalised amplitudes
%   (in [0, 1]) are designed for direct use with the Music Perception
%   Toolbox after conversion to the desired pitch scale via
%   convertPitch.
%
%   By default, peaks are picked directly from the FFT magnitude
%   spectrum (no smoothing). An optional Gaussian smoothing step is
%   available via the 'sigma' parameter; this is useful when the audio
%   contains vibrato or frequency jitter, which can cause a single
%   perceptual pitch to produce multiple closely spaced spectral peaks.
%   Smoothing with a sigma comparable to the vibrato width (typically
%   9-12 cents) collapses these into a single peak. However, if the
%   audio contains steady-state tones (synthesised sounds, keyboard
%   instruments, etc.), smoothing is unnecessary — and the downstream
%   toolbox functions already apply their own Gaussian smoothing via
%   the expectation tensor framework. When sigma > 0, the magnitude
%   spectrum is resampled onto a uniform log-frequency (cents) grid
%   before smoothing, so that the Gaussian kernel has a fixed
%   perceptual width at all frequencies. Peak positions are then
%   converted back to Hz.
%
%   The pipeline is:
%     1. Read audio and mix to mono.
%     2. Apply onset/offset ramps (if rampDuration > 0) to reduce
%        spectral artefacts from abrupt onsets or offsets.
%     3. Compute the single-sided magnitude spectrum via FFT.
%     4. If sigma > 0: resample onto a uniform cents grid via
%        convertPitch, smooth with a Gaussian kernel, and find peaks
%        on the smoothed spectrum. Otherwise: find peaks directly on
%        the magnitude spectrum.
%     5. Normalise peak amplitudes to [0, 1] (tallest peak = 1).
%
%   Note on frequency resolution: The FFT bin spacing is Fs / N Hz
%   (where N is the number of samples). At frequency f, this
%   corresponds to approximately 1200 / (log(2) * f * N / Fs) cents.
%   At low frequencies (below ~100 Hz), individual partials may not
%   be fully resolved unless the audio is several seconds long.
%
%   Typical workflow: Extract peaks in Hz, convert to the desired
%   pitch scale, then pass to toolbox functions. For example:
%
%     [f, w] = audioPeaks('audio/piano_Cmin_open.wav');
%     p = convertPitch(f, 'hz', 'cents');   % absolute MIDI cents
%     H = spectralEntropy(p, w, 12);        % no addSpectra needed
%
%   The Hz output also feeds directly into roughness, which requires
%   frequencies in Hz:
%
%     [f, w] = audioPeaks('audio/piano_Cmin_open.wav');
%     r = roughness(f, w);
%
%   Input:
%     audioFile — Path to an audio file (string or char). Any format
%                 supported by MATLAB's audioread (WAV, FLAC, OGG,
%                 MP3, M4A, etc.).
%
%   Name-Value Arguments:
%     'sigma'          — Gaussian smoothing width in cents
%                        (default: 0, no smoothing). When set to a
%                        positive value, the magnitude spectrum is
%                        resampled onto a uniform log-frequency (cents)
%                        grid and convolved with a Gaussian kernel of
%                        this width before peak picking. This is useful
%                        for audio with vibrato or frequency jitter:
%                        a single pitch with vibrato can produce
%                        multiple closely spaced spectral peaks, and
%                        smoothing collapses these into one. Values of
%                        9-12 cents are typically effective. Partials
%                        separated by more than about 2 * sigma cents
%                        are individually resolved; closer partials
%                        are merged. Unnecessary for steady-state tones.
%     'resolution'     — Cents-grid spacing when sigma > 0 (default: 1).
%                        Ignored when sigma = 0.
%     'rampDuration'   — Onset/offset ramp duration in seconds
%                        (default: 0). If the audio starts or ends
%                        abruptly, a ramp of 0.01-0.05 s can reduce
%                        spectral leakage artefacts. The ramp is a
%                        raised-cosine (Hann) fade.
%     'fMin'           — Minimum frequency in Hz (default: 27.5, A0).
%                        Frequencies below this are excluded.
%     'fMax'           — Maximum frequency in Hz (default: Nyquist).
%                        Frequencies above this are excluded.
%     'minProminence'  — Minimum peak prominence as a fraction of the
%                        maximum spectrum amplitude (default: 0.01).
%                        Peaks whose prominence is below this threshold
%                        are discarded.
%     'noiseFactor'    — Noise-floor threshold as a multiple of the
%                        median spectrum amplitude (default: 0). Set
%                        to a positive value (e.g., 3) to discard
%                        peaks below this noise floor.
%     'plot'           — Logical (default: false). If true, plots the
%                        spectrum with peaks marked.
%     'verbose'        — Logical (default: true). If false, suppresses
%                        all console output.
%
%   Outputs:
%     f      — Frequencies of spectral peaks in Hz (column vector),
%              sorted in ascending order.
%     w      — Normalised amplitudes of spectral peaks (column
%              vector, same order as f), in [0, 1] with the largest
%              peak at 1.
%     detail — (Optional) Struct containing intermediate data:
%                .freqSpectrum   — Single-sided magnitude spectrum
%                                  (column vector).
%                .freqAxis       — Frequency axis in Hz (column
%                                  vector).
%                .audio          — Mono time-domain signal (column
%                                  vector).
%                .Fs             — Sample rate in Hz (scalar).
%              When sigma > 0, the following are also populated:
%                .smoothSpectrum — Smoothed log-frequency spectrum
%                                  (column vector).
%                .rawSpectrum    — Unsmoothed log-frequency spectrum
%                                  (column vector).
%                .centsAxis      — Cents grid for the above spectra
%                                  (column vector, MIDI cents).
%
%   Examples:
%     % Basic peak extraction (no smoothing)
%     [f, w] = audioPeaks('audio/piano_C4.wav');
%
%     % With smoothing for vibrato-rich audio
%     [f, w] = audioPeaks('audio/violin_A4.wav', 'sigma', 12);
%
%     % Onset/offset ramp for an abruptly cut sample
%     [f, w] = audioPeaks('audio/music_sample.wav', 'rampDuration', 0.02);
%
%     % Inspect the spectrum via plot
%     [f, w, detail] = audioPeaks('audio/oboe_A4.wav', 'plot', true);
%
%     % Spectral pitch similarity of two audio files
%     [fA, wA] = audioPeaks('audio/piano_Emin.wav');
%     [fB, wB] = audioPeaks('audio/piano_G7_3rd_inversion.wav');
%     pA = convertPitch(fA, 'hz', 'cents');
%     pB = convertPitch(fB, 'hz', 'cents');
%     s = cosSimExpTens(pA, wA, pB, wB, 12, 2, true, true, 1200);
%
%     % Spectral entropy (no addSpectra needed)
%     [f, w] = audioPeaks('audio/piano_Cmin_open.wav');
%     p = convertPitch(f, 'hz', 'cents');
%     H = spectralEntropy(p, w, 12);
%
%     % Roughness (Hz input — no conversion needed)
%     [f, w] = audioPeaks('audio/piano_Cmin_open.wav');
%     r = roughness(f, w);
%
%   See also COSSIMEXPTENS, BATCHCOSSIMEXPTENS, ADDSPECTRA,
%            SPECTRALENTROPY, TEMPLATEHARMONICITY, ROUGHNESS,
%            CONVERTPITCH.

    arguments
        audioFile (1,:) {mustBeText}
        nvArgs.sigma (1,1) {mustBeNonnegative} = 0
        nvArgs.resolution (1,1) {mustBePositive} = 1
        nvArgs.rampDuration (1,1) {mustBeNonnegative} = 0
        nvArgs.fMin (1,1) {mustBePositive} = 27.5
        nvArgs.fMax (1,1) {mustBePositive} = Inf
        nvArgs.minProminence (1,1) {mustBeNonnegative} = 0.01
        nvArgs.noiseFactor (1,1) {mustBeNonnegative} = 0
        nvArgs.plot (1,1) logical = false
        nvArgs.verbose (1,1) logical = true
    end

    sigma      = nvArgs.sigma;
    step       = nvArgs.resolution;
    rampDur    = nvArgs.rampDuration;
    fMin       = nvArgs.fMin;
    fMax       = nvArgs.fMax;
    minProm    = nvArgs.minProminence;
    noiseFac   = nvArgs.noiseFactor;
    doPlot     = nvArgs.plot;
    verbose    = nvArgs.verbose;

    % === Validate file ===

    audioFile = char(audioFile);
    if ~isfile(audioFile)
        error('audioPeaks:fileNotFound', ...
              'Audio file not found: %s', audioFile);
    end

    % === Read audio ===

    [x, Fs] = audioread(audioFile);
    nChannels = size(x, 2);

    % Mix to mono (average across channels)
    if nChannels > 1
        x = mean(x, 2);
    end

    N = numel(x);
    [~, fname, fext] = fileparts(audioFile);

    if verbose
        fprintf('audioPeaks: %s%s  (%d samples, %.1f s, Fs = %d Hz)\n', ...
                fname, fext, N, N / Fs, Fs);
    end

    % === Apply onset/offset ramp ===

    if rampDur > 0
        rampSamples = round(rampDur * Fs);
        if rampSamples > 0 && rampSamples <= floor(N / 2)
            % Raised-cosine (Hann) fade
            ramp = 0.5 * (1 - cos(pi * (0:rampSamples-1)' / rampSamples));
            x(1:rampSamples) = x(1:rampSamples) .* ramp;
            x(end-rampSamples+1:end) = x(end-rampSamples+1:end) .* flipud(ramp);
        elseif rampSamples > floor(N / 2)
            warning('audioPeaks:rampTooLong', ...
                    ['Ramp duration (%.3f s, %d samples) exceeds half ' ...
                     'the signal length (%d samples). Ramp not applied.'], ...
                    rampDur, rampSamples, N);
        end
    end

    % === Compute single-sided magnitude spectrum ===

    X = fft(x);
    nFFT = floor(N / 2) + 1;          % number of unique bins
    magSpec = abs(X(1:nFFT)) / N;      % normalise by signal length
    magSpec(2:end-1) = 2 * magSpec(2:end-1);  % account for negative freqs

    fAxis = (0:nFFT-1)' * (Fs / N);   % frequency axis in Hz

    % === Clamp fMax to Nyquist ===

    nyquist = Fs / 2;
    if isinf(fMax) || fMax > nyquist
        fMax = nyquist;
    end

    if fMin >= fMax
        error('audioPeaks:badFreqRange', ...
              'fMin (%.1f Hz) must be less than fMax (%.1f Hz).', fMin, fMax);
    end

    % === Select bins within frequency range ===

    validIdx = fAxis >= fMin & fAxis <= fMax;
    fValid   = fAxis(validIdx);
    magValid = magSpec(validIdx);

    if numel(fValid) < 2
        warning('audioPeaks:tooFewBins', ...
                'Fewer than 2 FFT bins in the frequency range [%.1f, %.1f] Hz.', ...
                fMin, fMax);
        f = zeros(0, 1);
        w = zeros(0, 1);
        if nargout >= 3
            detail = struct('freqSpectrum', magSpec, 'freqAxis', fAxis, ...
                            'audio', x, 'Fs', Fs, ...
                            'smoothSpectrum', [], 'rawSpectrum', [], ...
                            'centsAxis', []);
        end
        return;
    end

    % === Peak picking ===
    % Two paths: when sigma > 0, the spectrum is resampled onto a
    % uniform cents grid and Gaussian-smoothed before peak picking.
    % This is useful for audio with vibrato or frequency jitter, where
    % a single perceptual pitch can produce multiple closely spaced
    % spectral peaks that should be merged into one. When sigma = 0
    % (the default), peaks are picked directly from the magnitude
    % spectrum in Hz — simpler and appropriate for steady-state tones
    % or any signal without substantial frequency modulation.

    if sigma > 0
        % --- Smoothed peak picking in log-frequency (cents) space ---
        % Resampling onto a uniform cents grid ensures that the
        % Gaussian kernel has a fixed perceptual width at all
        % frequencies: sigma cents at 100 Hz spans the same
        % musical interval as sigma cents at 5000 Hz. Without this
        % conversion, a fixed-Hz kernel would over-smooth high
        % partials and under-smooth low ones.

        cValid = convertPitch(fValid, 'hz', 'cents');

        % Build uniform cents grid
        cMin = ceil(min(cValid) / step) * step;
        cMax = floor(max(cValid) / step) * step;
        cGrid = (cMin:step:cMax)';

        if numel(cGrid) < 2
            warning('audioPeaks:narrowRange', ...
                    'Cents range is too narrow for the requested resolution.');
            f = zeros(0, 1);
            w = zeros(0, 1);
            if nargout >= 3
                detail = struct('freqSpectrum', magSpec, 'freqAxis', fAxis, ...
                                'audio', x, 'Fs', Fs, ...
                                'smoothSpectrum', [], 'rawSpectrum', [], ...
                                'centsAxis', []);
            end
            return;
        end

        % Interpolate magnitude spectrum onto cents grid (pchip
        % preserves peak shapes better than linear interpolation)
        rawLogSpec = interp1(cValid, magValid, cGrid, 'pchip', 0);
        rawLogSpec = max(rawLogSpec, 0);  % pchip can undershoot

        % Gaussian smoothing (truncated at 4 sigma)
        halfWidth = ceil(4 * sigma / step);
        kx = (-halfWidth:halfWidth)' * step;
        kernel = exp(-kx.^2 / (2 * sigma^2));
        kernel = kernel / sum(kernel);
        smoothLogSpec = conv(rawLogSpec, kernel, 'same');

        % Find peaks on the smoothed spectrum
        specForPeaks = smoothLogSpec;

        if verbose
            fprintf('  Smoothed log-f spectrum: %.1f to %.1f MIDI cents, %d bins, sigma = %g cents\n', ...
                    cMin, cMax, numel(cGrid), sigma);
        end

    else
        % --- Direct peak picking in Hz ---
        specForPeaks = magValid;

        % These are not populated in the unsmoothed path
        smoothLogSpec = [];
        rawLogSpec    = [];
        cGrid         = [];

        if verbose
            fprintf('  Frequency range: %.1f to %.1f Hz, %d bins (no smoothing)\n', ...
                    fMin, fMax, numel(fValid));
        end
    end

    % === Apply prominence and noise-floor thresholds ===

    if max(specForPeaks) == 0
        if verbose
            fprintf('  No spectral energy in the specified range.\n');
        end
        f = zeros(0, 1);
        w = zeros(0, 1);
        if nargout >= 3
            detail = struct('freqSpectrum', magSpec, 'freqAxis', fAxis, ...
                            'audio', x, 'Fs', Fs, ...
                            'smoothSpectrum', smoothLogSpec, ...
                            'rawSpectrum', rawLogSpec, ...
                            'centsAxis', cGrid);
        end
        return;
    end

    peakProm = minProm * max(specForPeaks);
    [pkAmps, pkLocs] = findpeaks(specForPeaks, 'MinPeakProminence', peakProm);

    % Noise-floor threshold
    if noiseFac > 0
        noiseFloor = noiseFac * median(specForPeaks);
        keep = pkAmps > noiseFloor;
        pkAmps = pkAmps(keep);
        pkLocs = pkLocs(keep);
    end

    if isempty(pkAmps)
        if verbose
            fprintf('  No peaks above threshold.\n');
        end
        f = zeros(0, 1);
        w = zeros(0, 1);
        if nargout >= 3
            detail = struct('freqSpectrum', magSpec, 'freqAxis', fAxis, ...
                            'audio', x, 'Fs', Fs, ...
                            'smoothSpectrum', smoothLogSpec, ...
                            'rawSpectrum', rawLogSpec, ...
                            'centsAxis', cGrid);
        end
        return;
    end

    % === Convert peak positions to Hz ===

    if sigma > 0
        pkCents = cGrid(pkLocs);
        f = convertPitch(pkCents, 'cents', 'hz');
    else
        f = fValid(pkLocs);
    end

    w = pkAmps / max(pkAmps);

    % Ensure column vectors
    f = f(:);
    w = w(:);

    if verbose
        fprintf('  Found %d peaks (prominence threshold: %.1f%% of max).\n', ...
                numel(f), minProm * 100);
    end

    % === Optional detail struct ===

    if nargout >= 3
        detail = struct( ...
            'freqSpectrum',   magSpec, ...
            'freqAxis',       fAxis, ...
            'audio',          x, ...
            'Fs',             Fs, ...
            'smoothSpectrum', smoothLogSpec, ...
            'rawSpectrum',    rawLogSpec, ...
            'centsAxis',      cGrid);
    end

    % === Optional plot ===

    if doPlot
        figure('Name', 'audioPeaks', ...
               'Position', [100 100 900 400]);

        if sigma > 0
            % Plot in cents space
            plot(cGrid, smoothLogSpec, 'Color', [0.2 0.2 0.6], ...
                 'LineWidth', 0.8);
            hold on;
            plot(cGrid, rawLogSpec, 'Color', [0.7 0.7 0.7], ...
                 'LineWidth', 0.4);
            plot(pkCents, pkAmps, 'r.', 'MarkerSize', 12);

            % Label the top peaks (up to 20) with Hz values
            [~, sortIdx] = sort(pkAmps, 'descend');
            nLabel = min(20, numel(sortIdx));
            for i = 1:nLabel
                idx = sortIdx(i);
                text(pkCents(idx), pkAmps(idx), ...
                     sprintf(' %.1f Hz', f(idx)), ...
                     'FontSize', 7, 'Color', [0.8 0.1 0.1], ...
                     'VerticalAlignment', 'bottom');
            end

            xlabel('Pitch (MIDI cents)');
            ylabel('Amplitude');
            title(sprintf('audioPeaks: %s%s  (\\sigma = %g cents)', ...
                          fname, fext, sigma));
            legend('Smoothed', 'Raw', 'Peaks', 'Location', 'best');

        else
            % Plot in Hz
            plot(fValid, magValid, 'Color', [0.2 0.2 0.6], ...
                 'LineWidth', 0.8);
            hold on;
            plot(f, pkAmps, 'r.', 'MarkerSize', 12);

            % Label the top peaks (up to 20)
            [~, sortIdx] = sort(pkAmps, 'descend');
            nLabel = min(20, numel(sortIdx));
            for i = 1:nLabel
                idx = sortIdx(i);
                text(f(idx), pkAmps(idx), ...
                     sprintf(' %.1f', f(idx)), ...
                     'FontSize', 7, 'Color', [0.8 0.1 0.1], ...
                     'VerticalAlignment', 'bottom');
            end

            xlabel('Frequency (Hz)');
            ylabel('Amplitude');
            title(sprintf('audioPeaks: %s%s  (no smoothing)', ...
                          fname, fext));
            legend('Spectrum', 'Peaks', 'Location', 'best');
        end

        hold off;
    end

end