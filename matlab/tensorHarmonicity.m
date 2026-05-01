function h = tensorHarmonicity(p, w, sigma, nvArgs)
%TENSORHARMONICITY Harmonicity via expectation tensor lookup.
%
%   h = tensorHarmonicity(p, w, sigma)
%   h = tensorHarmonicity(p, w, sigma, Name, Value)
%
%   Measures the harmonicity of a weighted pitch multiset by evaluating
%   the relative r-ad expectation tensor of a harmonic series at the
%   multiset's interval vector. The expectation tensor represents the
%   density of all ordered r-tuples of intervals that arise from a
%   harmonic series, smoothed by perceptual uncertainty sigma. A high
%   density at the chord's intervals indicates those intervals are
%   likely to co-occur in a harmonic series — hence the multiset is
%   "harmonic."
%
%   The procedure is:
%     1. Build a harmonic template spectrum via addSpectra.
%     2. Duplicate the template K times (see 'duplicate' below).
%     3. Build the relative r-ad expectation tensor (r = number of
%        pitches, isRel = true, isPer = false) from the template.
%     4. Compute the chord's intervals relative to its lowest pitch.
%     5. Evaluate the tensor density at that single interval point.
%
%   By default, the template spectrum is duplicated K times, where K is the
%   number of pitches in the chord. This is important because without
%   duplication, every position in an r-tuple can only be filled by a
%   different partial — a unison (two chord tones sharing the same
%   partials) cannot contribute. With K-fold duplication, each partial can
%   appear in up to K positions, correctly allowing unisons (and other
%   interval repetitions within the harmonic series) to register as
%   consonant.
%
%   Because evalExpTens evaluates at exact query points (not on a
%   grid), no resolution parameter is needed — the density is computed
%   analytically at the precise interval values. This eliminates the
%   grid resolution trade-offs present in earlier implementations.
%
%   For batch processing (many chords of the same cardinality), it is
%   more efficient to precompute the tensor once and query it
%   repeatedly using the lower-level functions:
%
%     K = 3;  % triad
%     [tp, tw] = addSpectra(zeros(K,1), ones(K,1), ...
%                           'harmonic', 64, 'powerlaw', 1);
%     T = buildExpTens(tp, tw, 12, K, true, false, 1200);
%     for i = 1:nChords
%         ints = sort(chords(i,:));
%         ints = ints(2:end) - ints(1);
%         h(i) = evalExpTens(T, ints(:));
%     end
%
%   Inputs:
%     p     — Pitch values in cents (vector of length K, where K >= 2).
%             These are absolute pitches, not pitch classes. The
%             function computes intervals internally.
%     w     — Weights (vector of length K, or empty for all ones).
%             These weight the template's partials; typically left
%             empty unless modelling unequal-amplitude tones.
%     sigma — Gaussian smoothing width in cents. Models perceptual
%             uncertainty. Values of 9-15 are typical; 12 is a good
%             default.
%
%   Name-Value Arguments:
%     'spectrum'  — Cell array of arguments to pass to addSpectra
%                   (everything after p and w). Defines the harmonic
%                   template whose expectation tensor is queried.
%                   Default: {'harmonic', 64, 'powerlaw', 1}.
%                   Examples:
%                     'spectrum', {'harmonic', 24, 'powerlaw', 1}
%                     'spectrum', {'harmonic', 12, 'geometric', 0.9}
%                     'spectrum', {'stretched', 8, 1.02, 'powerlaw', 1}
%     'duplicate' — Number of times to replicate the template pitch
%                   before adding harmonics (default: 0, meaning auto).
%                     0  — Automatically set to the number of pitches
%                          in p (the chord cardinality). This is the
%                          recommended default.
%                     K  — Any positive integer overrides the automatic
%                          setting. K = 1 disables duplication (each
%                          partial appears once; unisons cannot
%                          contribute).
%                   Computation time grows rapidly with K: the number
%                   of ordered r-tuples scales as (N*K)! / (N*K - r)!
%                   where N is the number of partials. A warning is
%                   issued when K > 3.
%     'normalize' — Normalization mode for the density value
%                   (default: 'none'):
%                     'none'     — Raw density. Suitable for comparing
%                                  chords evaluated with the same
%                                  parameters.
%                     'gaussian' — Each Gaussian integrates to 1.
%                                  Useful for comparing across
%                                  different sigma values.
%                     'pdf'      — Full probability density. Useful
%                                  for comparing across different
%                                  template sizes.
%
%   Output:
%     h     — Harmonicity (scalar, non-negative). Higher values
%             indicate greater harmonicity.
%
%   Examples:
%     % Harmonicity of a JI major triad (default: 3-fold duplication)
%     h = tensorHarmonicity([0, 386.31, 701.96], [], 12)
%
%     % Harmonicity of a 12-EDO major triad
%     h = tensorHarmonicity([0, 400, 700], [], 12)
%
%     % Without duplication (unisons do not contribute)
%     h = tensorHarmonicity([0, 400, 700], [], 12, 'duplicate', 1)
%
%     % Using convertPitch from MIDI
%     p = convertPitch([60 64 67], 'midi', 'cents');
%     h = tensorHarmonicity(p, [], 12)
%
%   References:
%     Milne, A. J. (2013). A computational model of the cognition of
%       tonality. PhD thesis, The Open University.
%     Smit, E. A., Milne, A. J., Dean, R. T., & Weidemann, G. (2019).
%       Perception of affect in unfamiliar musical chords. PLOS ONE,
%       14(6), e0218570.
%
%   See also ADDSPECTRA, BUILDEXPTENS, EVALEXPTENS, CONVERTPITCH,
%            TEMPLATEHARMONICITY, ROUGHNESS, SPECTRALENTROPY.

    arguments
        p (:,1) {mustBeNumeric}
        w (:,1) {mustBeNumeric} = []
        sigma (1,1) {mustBePositive} = 12
        nvArgs.spectrum = {'harmonic', 64, 'powerlaw', 1}
        nvArgs.duplicate (1,1) {mustBeNonnegative, mustBeInteger} = 0
        nvArgs.normalize (1,1) string ...
            {mustBeMember(nvArgs.normalize, {'none','gaussian','pdf'})} = 'none'
    end

    specArgs = nvArgs.spectrum;

    if ~iscell(specArgs)
        error('tensorHarmonicity:badSpectrum', ...
              '''spectrum'' value must be a cell array of addSpectra arguments.');
    end

    nPitches = numel(p);
    if nPitches < 2
        error('tensorHarmonicity:tooFewPitches', ...
              'At least 2 pitches are required (got %d).', nPitches);
    end

    % === Determine duplication count ===
    % Default (0) uses the chord cardinality, so that each partial can
    % fill every position in an r-tuple — allowing unisons to register
    % as consonant.

    dup = nvArgs.duplicate;
    if dup == 0
        dup = nPitches;
    end

    if dup > 3
        warning('tensorHarmonicity:largeDuplicate', ...
                ['duplicate = %d: computation time grows rapidly with ' ...
                 'duplication. Consider reducing to 3 or fewer.'], dup);
    end

    % === Build harmonic template spectrum ===
    % Pass K copies of pitch 0 to addSpectra so each harmonic partial
    % appears K times. This allows r-tuples that reuse the same
    % harmonic in multiple positions — critical for unisons (identical
    % intervals) to contribute to the density.

    [tmpl_p, tmpl_w] = addSpectra(zeros(dup, 1), ones(dup, 1), specArgs{:});

    % === Build the r-ad expectation tensor ===
    % r = nPitches (number of pitches in the chord), isRel = true
    % (intervals), isPer = false (non-periodic). The effective
    % dimensionality is dim = r - 1 = nPitches - 1.

    r     = nPitches;
    isRel = true;
    isPer = false;
    period = 1200;  % not used for wrapping; required by buildExpTens

    T = buildExpTens(tmpl_p, tmpl_w, sigma, r, isRel, isPer, period, ...
                     'verbose', false);

    % === Compute chord intervals ===
    % Sort pitches and take intervals relative to the lowest pitch,
    % giving a (nPitches-1)-dimensional interval vector.

    p = sort(p);
    intervals = p(2:end) - p(1);  % (nPitches-1) x 1 column vector

    % === Evaluate the tensor at the chord's interval point ===

    h = evalExpTens(T, intervals, nvArgs.normalize, 'verbose', false);

end