function out = convertPitch(values, fromScale, toScale)
%CONVERTPITCH Convert between pitch and frequency scales.
%
%   out = convertPitch(values, fromScale, toScale) converts the input
%   values from one pitch/frequency scale to another.
%
%   All conversions are routed through Hz as an intermediate
%   representation. The function is vectorized: values may be a scalar,
%   vector, or matrix, and the output has the same shape.
%
%   Supported scales (case-insensitive):
%
%     'hz'        Frequency in hertz.
%
%     'midi'      MIDI note number. A4 = 69, middle C (C4) = 60, each
%                 unit = one 12-EDO semitone.
%                   midi = 69 + 12 * log2(f / 440)
%
%     'cents'     Absolute cents: 100 * MIDI note number. A4 = 6900,
%                 middle C (C4) = 6000. This is an absolute pitch
%                 measure, not a relative interval measure.
%                   cents = 6900 + 1200 * log2(f / 440)
%
%     'mel'       Mel scale (O'Shaughnessy 1987). Perceptual pitch
%                 scale derived from subjective pitch bisection
%                 experiments (Stevens, Volkmann & Newman 1937/1940).
%                 Linear below ~500 Hz, logarithmic above.
%                   mel = 2595 * log10(1 + f / 700)
%
%     'bark'      Bark scale (Zwicker 1961). Critical-band-rate scale
%                 based on auditory masking experiments. Values range
%                 from 0 to ~24 Bark over the audible range. Uses
%                 Traunmuller's (1990) invertible approximation:
%                   bark = 26.81 / (1 + 1960 / f) - 0.53
%                 Valid for f > 0 Hz.
%
%     'erb'       ERB-rate scale (Glasberg & Moore 1990). Based on
%                 equivalent rectangular bandwidths measured with the
%                 notched-noise method. Generally considered the most
%                 rigorous psychoacoustic scale and has largely
%                 superseded Bark in modern auditory modelling.
%                   erb = 21.4 * log10(0.00437 * f + 1)
%
%     'greenwood' Greenwood cochlear position (Greenwood 1961/1990).
%                 Fractional distance along the basilar membrane from
%                 the apex (0 = apex, 1 = base). Maps frequency to
%                 cochlear place using the empirically derived function:
%                   f = 165.4 * (10^(2.1 * x) - 0.88)
%                 Primarily used in cochlear implant research.
%
%   Inputs:
%     values    — Numeric array of values in the source scale.
%     fromScale — Source scale (string).
%     toScale   — Target scale (string).
%
%   Output:
%     out       — Numeric array of converted values (same shape as
%                 values).
%
%   Examples:
%     % Concert A in various scales
%     convertPitch(440, 'hz', 'midi')       % 69
%     convertPitch(440, 'hz', 'cents')      % 6900
%     convertPitch(440, 'hz', 'mel')        % ~549.6
%     convertPitch(440, 'hz', 'bark')       % ~3.97
%     convertPitch(440, 'hz', 'erb')        % ~14.38
%     convertPitch(440, 'hz', 'greenwood')  % ~0.616
%
%     % MIDI to Hz
%     convertPitch(60, 'midi', 'hz')        % ~261.63 (middle C)
%
%     % Mel to ERB-rate
%     convertPitch(1000, 'mel', 'erb')      % converts via Hz
%
%     % Vectorized
%     convertPitch([261.63 440 880], 'hz', 'midi')  % [60 69 81]
%
%   References:
%     Glasberg, B. R. & Moore, B. C. J. (1990). Derivation of auditory
%       filter shapes from notched-noise data. Hearing Research, 47,
%       103-138.
%     Greenwood, D. D. (1990). A cochlear frequency-position function
%       for several species — 29 years later. Journal of the Acoustical
%       Society of America, 87(6), 2592-2605.
%     O'Shaughnessy, D. (1987). Speech Communication: Human and
%       Machine. Reading, MA: Addison-Wesley.
%     Stevens, S. S., Volkmann, J. & Newman, E. B. (1937). A scale for
%       the measurement of the psychological magnitude of pitch. Journal
%       of the Acoustical Society of America, 8, 185-190.
%     Traunmuller, H. (1990). Analytical expressions for the tonotopic
%       sensory scale. Journal of the Acoustical Society of America, 88,
%       97-100.
%     Zwicker, E. (1961). Subdivision of the audible frequency range
%       into critical bands (Frequenzgruppen). Journal of the Acoustical
%       Society of America, 33, 248.
%
%   See also ADDSPECTRA, BUILDEXPTENS.

    arguments
        values {mustBeNumeric}
        fromScale (1,1) string
        toScale   (1,1) string
    end

    fromScale = lower(fromScale);
    toScale   = lower(toScale);

    validScales = {'hz', 'midi', 'cents', 'mel', 'bark', 'erb', 'greenwood'};
    if ~ismember(fromScale, validScales)
        error('convertPitch:unknownScale', ...
              'Unknown source scale ''%s''. Valid scales: %s.', ...
              fromScale, strjoin(validScales, ', '));
    end
    if ~ismember(toScale, validScales)
        error('convertPitch:unknownScale', ...
              'Unknown target scale ''%s''. Valid scales: %s.', ...
              toScale, strjoin(validScales, ', '));
    end

    % Short-circuit identity conversion.
    if strcmp(fromScale, toScale)
        out = values;
        return;
    end

    % === Convert to Hz ===
    f = toHz(values, fromScale);

    % === Convert from Hz to target ===
    out = fromHz(f, toScale);

end


% =====================================================================
%  Local functions
% =====================================================================

function f = toHz(values, scale)
%TOHZ Convert from the named scale to Hz.

    switch scale
        case 'hz'
            f = values;

        case 'midi'
            f = 440 * 2 .^ ((values - 69) / 12);

        case 'cents'
            f = 440 * 2 .^ ((values - 6900) / 1200);

        case 'mel'
            f = 700 * (10 .^ (values / 2595) - 1);

        case 'bark'
            % Inverse of Traunmuller: z = 26.81/(1 + 1960/f) - 0.53
            %   z + 0.53 = 26.81*f / (f + 1960)
            %   f = 1960*(z + 0.53) / (26.28 - z)
            f = 1960 * (values + 0.53) ./ (26.28 - values);

        case 'erb'
            % Inverse of E = 21.4 * log10(0.00437*f + 1)
            f = (10 .^ (values / 21.4) - 1) / 0.00437;

        case 'greenwood'
            % f = 165.4 * (10^(2.1*x) - 0.88)
            f = 165.4 * (10 .^ (2.1 * values) - 0.88);
    end
end

function out = fromHz(f, scale)
%FROMHZ Convert from Hz to the named scale.

    switch scale
        case 'hz'
            out = f;

        case 'midi'
            out = 69 + 12 * log2(f / 440);

        case 'cents'
            out = 6900 + 1200 * log2(f / 440);

        case 'mel'
            out = 2595 * log10(1 + f / 700);

        case 'bark'
            out = 26.81 ./ (1 + 1960 ./ f) - 0.53;

        case 'erb'
            out = 21.4 * log10(0.00437 * f + 1);

        case 'greenwood'
            % x = log10(f/165.4 + 0.88) / 2.1
            out = log10(f / 165.4 + 0.88) / 2.1;
    end
end
