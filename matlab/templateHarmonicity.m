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
        p (:,1) {mustBeNumeric}
        w (:,1) {mustBeNumeric} = []
        sigma (1,1) {mustBePositive} = 12
        nvArgs.spectrum = {'harmonic', 36, 'powerlaw', 1}
        nvArgs.chordSpectrum = {}
        nvArgs.normalize (1,1) logical = true
        nvArgs.base (1,1) {mustBePositive} = 2
        nvArgs.resolution (1,1) {mustBePositive} = 1
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

    % === Build expectation tensors ===
    % r = 1, isRel = false: these are intrinsic to the harmonicity
    % definition (1-D absolute density of spectral components).

    r     = 1;
    isRel = false;
    isPer = false;
    period = 1200;  % not used for wrapping; required by buildExpTens

    tmpl_dens  = buildExpTens(tmpl_p, tmpl_w, sigma, r, isRel, ...
                              isPer, period, 'verbose', false);
    chord_dens = buildExpTens(chord_p, chord_w, sigma, r, isRel, ...
                              isPer, period, 'verbose', false);

    % === Evaluate on grids ===
    % Both grids start at 0 with the same spacing. The margin captures
    % Gaussian tails beyond the outermost partials.

    margin = 4 * sigma;

    x_tmpl  = 0:step:(max(tmpl_p) + margin);
    x_chord = 0:step:(max(chord_p) + margin);

    tmpl_vals  = evalExpTens(tmpl_dens, x_tmpl, 'verbose', false);
    chord_vals = evalExpTens(chord_dens, x_chord, 'verbose', false);

    % === Cross-correlation ===

    xcorr_vals = conv(chord_vals, fliplr(tmpl_vals), 'full');

    % === Normalize (cosine similarity at each lag) ===

    norm_factor = sqrt(sum(chord_vals.^2) * sum(tmpl_vals.^2));
    xcorr_norm = xcorr_vals / norm_factor;

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