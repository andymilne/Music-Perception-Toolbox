function [hMax, hEntropy] = templateHarmonicity(p, w, sigma, nvArgs)
%TEMPLATEHARMONICITY Harmonicity via template cross-correlation.
%
%   hMax = templateHarmonicity(p, w, sigma)
%   [hMax, hEntropy] = templateHarmonicity(p, w, sigma)
%   [hMax, hEntropy] = templateHarmonicity(p, w, sigma, Name, Value)
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
%                dividing by log_base(N), N the number of
%                transpositions in the cross-correlation (see
%                'normalize' below).
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
%        isRel = false) on a fine grid: with 'per' false (the
%        default), each from 6 sigma below its lowest partial to 6
%        sigma above its highest; with 'per' true, over one period.
%     5. Cross-correlate the two density vectors (circularly, over
%        pitch class, when 'per' is true).
%     6. Normalize by the geometric mean of their energies (giving
%        cosine similarity at each lag).
%     7. Return the maximum (hMax) and the entropy (hEntropy).
%
%   Inputs:
%     p     — Pitch values in cents (vector). These are absolute
%             pitches (e.g., MIDI 60 = 6000 cents via convertPitch),
%             not pitch classes. The function transposes internally
%             so the lowest pitch is 0.
%     w     — Weights (vector same length as p, or empty for uniform).
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
%                       is divided by log_base(N), N the number of
%                       transpositions in the cross-correlation, to
%                       give a value in [0, 1]. With 'per' false, N
%                       grows with the span of the chord's spectrum,
%                       so the normalized value is an entropy relative
%                       to that span. For values that do not depend on
%                       the span, set 'normalize' to false or 'per' to
%                       true (where N is fixed by the period). Only
%                       affects hEntropy.
%     'base'          — Logarithm base for entropy (default: 2, giving
%                       bits). When 'normalize' is true, the base
%                       cancels and has no effect on the result. Only
%                       affects hEntropy.
%     'resolution'    — Grid spacing in cents (default: 1). Finer
%                       resolution improves accuracy but increases
%                       computation time. With 'per' true it must
%                       divide 'period'.
%     'per'           — Logical (default: false). If true, chord and
%                       template are folded into one period and the
%                       cross-correlation is circular, over pitch class.
%     'period'        — Period in cents when 'per' is true (default:
%                       1200).
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
        p (:,1) {mustBeNumeric}
        w (:,1) {mustBeNumeric} = []
        sigma (1,1) {mustBePositive} = 12
        nvArgs.spectrum = {'harmonic', 36, 'powerlaw', 1}
        nvArgs.chordSpectrum = {}
        nvArgs.normalize (1,1) logical = true
        nvArgs.base (1,1) {mustBePositive} = 2
        nvArgs.resolution (1,1) {mustBePositive} = 1
        nvArgs.per (1,1) logical = false
        nvArgs.period (1,1) {mustBePositive} = 1200
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

    % === Cross-correlate chord and template ===
    % r = 1, isRel = false: these are intrinsic to the harmonicity
    % definition (1-D absolute density of spectral components).

    isPer = nvArgs.per;
    if isPer
        period = nvArgs.period;
    else
        period = 1200;  % not used for wrapping; required by buildExpTens
    end

    xcorr_norm = localTemplateXcorr(chord_p, chord_w, tmpl_p, tmpl_w, ...
                                    sigma, step, isPer, period, 0);

    % === Milne 2013: maximum normalized cross-correlation ===

    hMax = max(xcorr_norm);

    % === Harrison 2020: entropy of normalized cross-correlation ===

    if nargout > 1
        q = xcorr_norm(:);
        N = numel(q);       % total bins (before removing zeros)
        q = q / sum(q);     % normalize to probability distribution
        q(q <= 0) = [];     % apply 0*log(0) = 0 convention

        hEntropy = -sum(q .* (log(q) / log(nvArgs.base)));

        if nvArgs.normalize
            hEntropy = hEntropy / (log(N) / log(nvArgs.base));
        end
    end

end

function [xcorrNorm, vpP] = localTemplateXcorr(chord_p, chord_w, ...
        tmpl_p, tmpl_w, sigma, step, isPer, period, pOffset)
%LOCALTEMPLATEXCORR Normalised cross-correlation of a chord's spectrum
%   with a harmonic template, and the pitch at which each transposition
%   places the template's fundamental. Non-periodic: a full linear
%   cross-correlation over both grids. Periodic: a circular
%   cross-correlation over one period, sorted by pitch class.
    tmpl_dens  = buildExpTens(tmpl_p, tmpl_w, sigma, 1, false, ...
                              isPer, period, 'verbose', false);
    chord_dens = buildExpTens(chord_p, chord_w, sigma, 1, false, ...
                              isPer, period, 'verbose', false);

    x_tmpl  = localGrid(max(tmpl_p), sigma, step, isPer, period);
    x_chord = localGrid(max(chord_p), sigma, step, isPer, period);

    tmpl_vals  = reshape(evalExpTens(tmpl_dens, x_tmpl, 'verbose', false), [], 1);
    chord_vals = reshape(evalExpTens(chord_dens, x_chord, 'verbose', false), [], 1);

    if isPer
        % xcorr(l+1) = sum_i chord(i + l) * tmpl(i): the template shifted
        % up by l steps, its fundamental at l * step.
        xc = real(ifft(fft(chord_vals) .* conj(fft(tmpl_vals))));
        xc = max(xc, 0);   % remove FFT round-off below 0
        vpP = mod((0:numel(xc) - 1)' * step + pOffset, period);
        [vpP, order] = sort(vpP);
        xc = xc(order);
    else
        % Both grids start at -6*sigma, so lag index k (0-based) of the
        % full correlation places the template's fundamental
        % (k - (N_tmpl - 1)) * step above the chord's lowest pitch.
        xc = conv(chord_vals, flipud(tmpl_vals), 'full');
        nTmpl = numel(tmpl_vals);
        vpP = ((0:numel(xc) - 1)' - (nTmpl - 1)) * step + pOffset;
    end

    normFactor = sqrt(sum(chord_vals.^2) * sum(tmpl_vals.^2));
    xcorrNorm = xc / normFactor;
end


function x = localGrid(specMax, sigma, step, isPer, period)
%LOCALGRID Evaluation grid for a spectrum whose lowest partial is at 0.
%   Periodic: one period, 0:step:(period - step), step dividing period.
%   Non-periodic: from -m to specMax + m (to within one step),
%   m = 6 * sigma, so that every partial's Gaussian lies whole on the
%   grid (the omitted tail of each is below 1e-8 of its peak).
    if isPer
        n = round(period / step);
        if n < 2 || abs(n * step - period) > 1e-9 * period
            error('templateHarmonicity:badResolution', ...
                  ['''resolution'' (%g) must divide the period (%g) ' ...
                   'when ''per'' is true.'], step, period);
        end
        x = (0:n - 1) * step;
    else
        m = 6 * sigma;
        n = floor((specMax + 2 * m) / step + 1e-9) + 1;
        x = -m + (0:n - 1) * step;
    end
end
