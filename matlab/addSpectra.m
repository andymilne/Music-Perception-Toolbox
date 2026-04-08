function [p, w] = addSpectra(p, w, mode, varargin)
%ADDSPECTRA Add spectral partials to a set of pitches.
%
%   Five modes determine the partial positions; for the first four, a
%   sub-option selects the weight decay law.
%
%   === Harmonic mode ===
%
%   [p, w] = addSpectra(p, w, 'harmonic', N, 'powerlaw', rho):
%   [p, w] = addSpectra(p, w, 'harmonic', N, 'geometric', tau):
%
%   Adds N harmonic partials to each pitch. The n-th partial (n = 1, 2,
%   ..., N) has a frequency ratio of n relative to the fundamental, giving
%   a pitch offset of units * log2(n). The first partial (n = 1) is the
%   fundamental itself.
%
%   This is a special case of 'stretched' with beta = 1, kept as a
%   separate mode for convenience.
%
%   === Stretched mode (log-frequency stretch) ===
%
%   [p, w] = addSpectra(p, w, 'stretched', N, beta, 'powerlaw', rho):
%   [p, w] = addSpectra(p, w, 'stretched', N, beta, 'geometric', tau):
%
%   Generalizes 'harmonic' by placing the n-th partial at a frequency
%   ratio of n^beta, giving a pitch offset of beta * units * log2(n).
%   This uniformly scales all partial spacings in log-frequency by the
%   factor beta.
%
%     beta = 1: harmonic series (identical to 'harmonic' mode)
%     beta > 1: stretched — partials wider apart than harmonic
%     beta < 1: compressed — partials closer together than harmonic
%
%   This mode is the simplest one-parameter continuous deformation of the
%   harmonic series in log-frequency space: every partial spacing is
%   uniformly scaled by the same factor beta. It does not correspond to a
%   specific physical vibrating system (unlike 'stiff', which models real
%   string inharmonicity), but its mathematical simplicity makes it useful
%   for systematic exploration of how the expectation tensor landscape
%   changes as spectra depart from harmonicity.
%
%   === Freq-linear mode (frequency-domain stretch) ===
%
%   [p, w] = addSpectra(p, w, 'freqlinear', N, alpha, 'powerlaw', rho):
%   [p, w] = addSpectra(p, w, 'freqlinear', N, alpha, 'geometric', tau):
%
%   Places the n-th partial at a frequency ratio of (alpha + n) / (alpha + 1)
%   relative to the fundamental, giving a pitch offset of:
%     units * log2((alpha + n) / (alpha + 1))
%
%   The single parameter alpha controls the departure from harmonicity in
%   the frequency domain:
%
%     alpha = 0: harmonic series (ratios 1, 2, 3, ...)
%     alpha < 0: stretched — partials wider apart than harmonic (the
%       spacing in log-frequency increases for higher partials). E.g.,
%       alpha = -1/2 gives ratios 1, 3, 5, ... scaled by a constant.
%     alpha > 0: compressed — partials closer together than harmonic
%       (the spacing in log-frequency decreases for higher partials).
%
%   Note: all frequency ratios (alpha + n) must be positive, so alpha
%   must be greater than -1.
%
%   This type of stretching/compression is commonly used in psychoacoustic
%   experiments. The stretching is non-uniform in log-frequency (unlike
%   'stretched' mode where it is uniform), because it arises from a linear
%   perturbation in the frequency domain.
%
%   === Stiff mode (stiff-string inharmonicity) ===
%
%   [p, w] = addSpectra(p, w, 'stiff', N, B, 'powerlaw', rho):
%   [p, w] = addSpectra(p, w, 'stiff', N, B, 'geometric', tau):
%
%   Models the inharmonicity of stiff strings (e.g., piano strings). The
%   n-th partial has a frequency ratio of n * sqrt(1 + B * n^2), giving
%   a pitch offset of:
%     units * log2(n * sqrt(1 + B * n^2))
%
%   normalized so the fundamental (n = 1) is at offset 0:
%     units * log2((n * sqrt(1 + B * n^2)) / sqrt(1 + B))
%
%   The parameter B is the inharmonicity coefficient:
%     B = 0: harmonic series
%     B > 0: partials progressively sharper than harmonic (stretch grows
%       approximately as n^2 for small B)
%     Typical piano values: B ~ 10^-5 (bass strings) to 10^-3 (treble)
%
%   Physically derived from the wave equation for a stiff vibrating string,
%   where B depends on the string's Young's modulus, cross-sectional
%   moment of inertia, tension, and length.
%
%   === Custom mode ===
%
%   [p, w] = addSpectra(p, w, 'custom', offsets, spec_w):
%
%   Fully user-specified partials. 'offsets' is a vector of pitch offsets
%   (in the same units as p) relative to each fundamental. 'spec_w' is a
%   vector of corresponding spectral weights. To include the fundamental,
%   include 0 in offsets. This mode can represent:
%     - Harmonic spectra with arbitrary weights
%     - Inharmonic spectra (e.g., bells, metallophones)
%     - Any combination thereof
%
%   === Weight decay options (harmonic, stretched, freqlinear, stiff) ===
%
%     'powerlaw', rho  — weight(n) = 1 / n^rho
%       rho = 0: flat spectrum (equal amplitude)
%       rho = 1: 1/n rolloff (sawtooth-like)
%       Higher rho gives steeper rolloff (more emphasis on lower harmonics)
%     'geometric', tau — weight(n) = tau^(n-1)
%       tau = 1: flat spectrum
%       tau = 0.5: 6 dB per partial rolloff
%       tau = 0: pure tone (only fundamental)
%
%   === Optional name-value pair (all modes) ===
%
%   [...] = addSpectra(..., 'units', U):
%   Specifies the number of pitch units per octave (default: 1200, i.e.,
%   cents). Used in all modes except 'custom' to compute partial positions
%   in log-frequency space. Has no effect in 'custom' mode.
%
%   Inputs:
%     p       — Pitch values (vector of length M)
%     w       — Weights (vector of length M, or empty for uniform)
%     mode    — 'harmonic', 'stretched', 'freqlinear', 'stiff', or 'custom'
%
%     For 'harmonic' mode:
%       N          — Number of partials (positive integer, incl. fundamental)
%       weightType — 'powerlaw' or 'geometric'
%       param      — rho (for powerlaw) or tau (for geometric)
%
%     For 'stretched' mode:
%       N          — Number of partials
%       beta       — Log-frequency stretch factor (positive scalar)
%       weightType — 'powerlaw' or 'geometric'
%       param      — rho or tau
%
%     For 'freqlinear' mode:
%       N          — Number of partials
%       alpha      — Frequency-domain stretch parameter (scalar > -1)
%       weightType — 'powerlaw' or 'geometric'
%       param      — rho or tau
%
%     For 'stiff' mode:
%       N          — Number of partials
%       B          — Inharmonicity coefficient (non-negative scalar)
%       weightType — 'powerlaw' or 'geometric'
%       param      — rho or tau
%
%     For 'custom' mode:
%       offsets    — Vector of pitch offsets relative to each fundamental
%       spec_w     — Vector of spectral weights (same length as offsets)
%
%   Optional:
%     'units', U — Pitch units per octave (default: 1200)
%
%   Outputs:
%     p   — Column vector of all pitches with partials added
%               (length = M * K, where K is the number of partials)
%     w   — Column vector of corresponding weights (each weight is
%               the product of the original pitch weight and the spectral
%               weight of that partial)
%
%   Examples:
%     % 8 harmonic partials with 1/n power-law rolloff (cents)
%     [p, w] = addSpectra([0; 400; 700], [], 'harmonic', 8, 'powerlaw', 1);
%
%     % 12 harmonic partials with geometric decay tau = 0.7
%     [p, w] = addSpectra([0; 400; 700], [], 'harmonic', 12, 'geometric', 0.7);
%
%     % Log-stretched partials (beta = 1.02, slightly wider than harmonic)
%     [p, w] = addSpectra([0; 400; 700], [], 'stretched', 8, 1.02, ...
%                            'powerlaw', 1);
%
%     % Log-compressed partials (beta = 0.9)
%     [p, w] = addSpectra([0; 400; 700], [], 'stretched', 8, 0.9, ...
%                            'geometric', 0.7);
%
%     % Freq-linear stretch (alpha = -0.3, non-uniform widening)
%     [p, w] = addSpectra([0; 400; 700], [], 'freqlinear', 8, -0.3, ...
%                            'powerlaw', 1);
%
%     % Freq-linear compression (alpha = 0.5, partials converge)
%     [p, w] = addSpectra([0; 400; 700], [], 'freqlinear', 8, 0.5, ...
%                            'powerlaw', 1);
%
%     % Piano-like stiff string (B = 0.0003)
%     [p, w] = addSpectra([0; 400; 700], [], 'stiff', 12, 0.0003, ...
%                            'powerlaw', 1);
%
%     % Custom inharmonic partials (e.g., bell-like)
%     offsets = [0, 1202, 1904, 2808];
%     spec_w  = [1, 0.5, 0.25, 0.125];
%     [p, w] = addSpectra([0; 400; 700], [], 'custom', offsets, spec_w);
%
%     % Harmonic partials in semitones (12 per octave)
%     [p, w] = addSpectra([0; 4; 7], [], 'harmonic', 6, 'powerlaw', 1, ...
%                            'units', 12);
%
%   The output can be passed directly to buildExpTens, evalExpTens, or
%   cosSimExpTens as the pitch and weight arguments.
%
%   See also buildExpTens, evalExpTens, cosSimExpTens.

% === Input validation ===

p = p(:);

if isempty(w)
    w = ones(numel(p), 1);
end
if isscalar(w)
    w = w * ones(numel(p), 1);
end
w = w(:);

if numel(p) ~= numel(w)
    error('w must have the same number of entries as p (or be empty).');
end

if ~ischar(mode)
    error(['mode must be a string: ''harmonic'', ''stretched'', ' ...
        '''freqlinear'', ''stiff'', or ''custom''.']);
end

M = numel(p);

% === Extract optional 'units' name-value pair from varargin ===

units = 1200;  % default: cents
unitsIdx = [];
for i = 1:numel(varargin)
    if (ischar(varargin{i}) || isstring(varargin{i})) && strcmpi(varargin{i}, 'units')
        if i + 1 > numel(varargin)
            error('''units'' must be followed by a numeric value.');
        end
        units = varargin{i + 1};
        if ~isscalar(units) || units <= 0
            error('''units'' must be a positive scalar.');
        end
        unitsIdx = [i, i + 1]; %#ok<AGROW>
        break;
    end
end

% Remove 'units' pair from varargin for cleaner mode-specific parsing
if ~isempty(unitsIdx)
    varargin(unitsIdx) = [];
end

% === Parse mode-specific arguments ===

switch lower(mode)

    % =================================================================
    %  Harmonic: ratio(n) = n
    %  Usage: addSpectra(p, w, 'harmonic', N, weightType, param)
    % =================================================================
    case 'harmonic'
        if numel(varargin) < 3
            error(['For ''harmonic'' mode, usage is:\n' ...
                '  addSpectra(p, w, ''harmonic'', N, ''powerlaw'', rho)\n' ...
                '  addSpectra(p, w, ''harmonic'', N, ''geometric'', tau)']);
        end

        N          = varargin{1};
        weightType = varargin{2};
        param      = varargin{3};

        validateN(N);
        n = (1:N)';

        % Harmonic partial positions: offset = units * log2(n)
        offsets = units * log2(n);

        % Spectral weights
        spec_w = parseWeights(n, weightType, param);

    % =================================================================
    %  Stretched: ratio(n) = n^beta
    %  Usage: addSpectra(p, w, 'stretched', N, beta, weightType, param)
    % =================================================================
    case 'stretched'
        if numel(varargin) < 4
            error(['For ''stretched'' mode, usage is:\n' ...
                '  addSpectra(p, w, ''stretched'', N, beta, ' ...
                '''powerlaw'', rho)\n' ...
                '  addSpectra(p, w, ''stretched'', N, beta, ' ...
                '''geometric'', tau)']);
        end

        N          = varargin{1};
        beta       = varargin{2};
        weightType = varargin{3};
        param      = varargin{4};

        validateN(N);
        if ~isscalar(beta) || beta <= 0
            error('beta must be a positive scalar.');
        end

        n = (1:N)';

        % Log-stretched partial positions: offset = beta * units * log2(n)
        % Equivalently, ratio(n) = n^beta
        offsets = beta * units * log2(n);

        spec_w = parseWeights(n, weightType, param);

    % =================================================================
    %  Freq-linear: ratio(n) = (alpha + n) / (alpha + 1)
    %  Usage: addSpectra(p, w, 'freqlinear', N, alpha, weightType, param)
    % =================================================================
    case 'freqlinear'
        if numel(varargin) < 4
            error(['For ''freqlinear'' mode, usage is:\n' ...
                '  addSpectra(p, w, ''freqlinear'', N, alpha, ' ...
                '''powerlaw'', rho)\n' ...
                '  addSpectra(p, w, ''freqlinear'', N, alpha, ' ...
                '''geometric'', tau)']);
        end

        N          = varargin{1};
        alpha      = varargin{2};
        weightType = varargin{3};
        param      = varargin{4};

        validateN(N);
        if ~isscalar(alpha)
            error('alpha must be a scalar.');
        end
        if alpha <= -1
            error(['alpha must be greater than -1 (to ensure all ' ...
                'frequency ratios are positive).']);
        end

        n = (1:N)';

        % Frequency ratios: (alpha + n) for n = 1, 2, ..., N
        % Normalized to the fundamental (n = 1): ratio = (alpha + n) / (alpha + 1)
        ratios = (alpha + n) / (alpha + 1);

        % Validate all ratios are positive
        if any(ratios <= 0)
            badN = find(ratios <= 0, 1);
            error(['All frequency ratios (alpha + n) / (alpha + 1) must ' ...
                'be positive. With alpha = %.4g, the ratio for n = %d ' ...
                'is %.4g.'], alpha, badN, ratios(badN));
        end

        offsets = units * log2(ratios);

        spec_w = parseWeights(n, weightType, param);

    % =================================================================
    %  Stiff: ratio(n) = n * sqrt(1 + B * n^2), normalized to n = 1
    %  Usage: addSpectra(p, w, 'stiff', N, B, weightType, param)
    % =================================================================
    case 'stiff'
        if numel(varargin) < 4
            error(['For ''stiff'' mode, usage is:\n' ...
                '  addSpectra(p, w, ''stiff'', N, B, ''powerlaw'', rho)\n' ...
                '  addSpectra(p, w, ''stiff'', N, B, ''geometric'', tau)']);
        end

        N          = varargin{1};
        B          = varargin{2};
        weightType = varargin{3};
        param      = varargin{4};

        validateN(N);
        if ~isscalar(B) || B < 0
            error('B must be a non-negative scalar.');
        end

        n = (1:N)';

        % Stiff-string frequency ratios: n * sqrt(1 + B * n^2)
        % Normalized to the fundamental (n = 1): divide by sqrt(1 + B)
        ratios = (n .* sqrt(1 + B * n.^2)) / sqrt(1 + B);

        offsets = units * log2(ratios);

        spec_w = parseWeights(n, weightType, param);

    % =================================================================
    %  Custom: user-specified offsets and weights
    %  Usage: addSpectra(p, w, 'custom', offsets, spec_w)
    % =================================================================
    case 'custom'
        if numel(varargin) < 2
            error(['For ''custom'' mode, usage is:\n' ...
                '  addSpectra(p, w, ''custom'', offsets, spec_w)']);
        end

        offsets = varargin{1};
        spec_w  = varargin{2};

        offsets = offsets(:);
        spec_w  = spec_w(:);

        if numel(offsets) ~= numel(spec_w)
            error('offsets and spec_w must have the same length.');
        end

    otherwise
        error(['Unknown mode ''%s''. Use ''harmonic'', ''stretched'', ' ...
            '''freqlinear'', ''stiff'', or ''custom''.'], mode);
end

offsets = offsets(:);
spec_w  = spec_w(:);

% === Build output ===
% For each pitch p(i) with weight w(i), and each partial k with offset
% offsets(k) and spectral weight spec_w(k), the output contains:
%   pitch:  p(i) + offsets(k)
%   weight: w(i) * spec_w(k)
%
% Using implicit expansion: (M x 1) + (1 x K) -> (M x K)

p_matrix = p + offsets';      % M x K
w_matrix = w .* spec_w';     % M x K

% Flatten to column vectors
p = p_matrix(:);
w = w_matrix(:);

end


% =====================================================================
%  LOCAL HELPER FUNCTIONS
% =====================================================================

function validateN(N)
%VALIDATEN Check that N is a positive integer.
    if ~isscalar(N) || N < 1 || rem(N, 1)
        error('N must be a positive integer.');
    end
end

function spec_w = parseWeights(n, weightType, param)
%PARSEWEIGHTS Compute spectral weights for partial numbers n.
%
%   weightType:
%     'powerlaw'  — spec_w = 1 ./ n.^param  (param = rho >= 0)
%     'geometric' — spec_w = param.^(n-1)    (param = tau in [0, 1])

    if ~ischar(weightType)
        error('weightType must be ''powerlaw'' or ''geometric''.');
    end

    switch lower(weightType)

        case 'powerlaw'
            rho = param;
            if ~isscalar(rho) || rho < 0
                error('rho must be a non-negative scalar.');
            end
            spec_w = 1 ./ (n .^ rho);

        case 'geometric'
            tau = param;
            if ~isscalar(tau) || tau < 0 || tau > 1
                error('tau must be a scalar in [0, 1].');
            end
            spec_w = tau .^ (n - 1);

        otherwise
            error(['Unknown weight type ''%s''. ' ...
                'Use ''powerlaw'' or ''geometric''.'], weightType);
    end

end