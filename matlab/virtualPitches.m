function [vp_p, vp_w] = virtualPitches(p, w, sigma, nvArgs)
%VIRTUALPITCHES Virtual pitch salience profile via template cross-correlation.
%
%   [vp_p, vp_w] = virtualPitches(p, w, sigma)
%   [vp_p, vp_w] = virtualPitches(p, w, sigma, Name, Value)
%
%   Computes the virtual pitch (fundamental) salience profile for a
%   pitch set by cross-correlating its spectral expectation tensor with
%   a harmonic template. The result is a pitch-indexed vector of
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
%     1. Transpose the pitch set so the lowest pitch is 0 (internal).
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
%                       audioPeaks). To model each chord tone as a
%                       complex tone, pass the same (or different)
%                       spectral arguments:
%                         'chordSpectrum', {'harmonic', 36, 'powerlaw', 1}
%     'resolution'    — Grid spacing in cents (default: 1). Finer
%                       resolution improves pitch accuracy but
%                       increases computation time and output length.
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
        p (:,1) {mustBeNumeric}
        w (:,1) {mustBeNumeric} = []
        sigma (1,1) {mustBePositive} = 12
        nvArgs.spectrum = {'harmonic', 36, 'powerlaw', 1}
        nvArgs.chordSpectrum = {}
        nvArgs.resolution (1,1) {mustBePositive} = 1
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

    % === Build expectation tensors ===
    % r = 1, isRel = false: these are intrinsic to the virtual pitch
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

    % === Map lag indices to pitch values ===
    % The 'full' convolution output has length N_chord + N_tmpl - 1.
    % At lag index k (0-based), the template's root (pitch 0) aligns
    % with chord-grid position (k - N_tmpl + 1) * step. Adding back
    % pOffset converts to the input's absolute pitch coordinate system.

    N_tmpl  = numel(tmpl_vals);
    N_xcorr = numel(xcorr_norm);

    lag_indices = (0:N_xcorr - 1)' - (N_tmpl - 1);
    vp_p = lag_indices * step + pOffset;
    vp_w = xcorr_norm(:);

end