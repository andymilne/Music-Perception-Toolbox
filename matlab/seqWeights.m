function v = seqWeights(w, spec, nvArgs)
%SEQWEIGHTS Apply a position-weighting profile to an existing weight
%vector.
%
%   v = seqWeights(w, spec)
%   v = seqWeights(w, spec, 'N', N)
%   v = seqWeights(..., Name, Value)
%
%   Constructs a length-N profile from the named, callable, or explicit
%   specification and returns its pointwise product with w.
%
%   The length N of the output is inferred from w when w is a
%   non-empty, non-scalar numeric vector. When w is [] or scalar,
%   N must be supplied explicitly via the 'N' name-value argument.
%
%   Inputs:
%     w          — Length-N vector of per-position weights, [] for
%                  all ones — requires 'N' —, or a scalar
%                  broadcast to length N — requires 'N'.
%     spec       — Named specification: 'flat', 'primacy', 'recency',
%                  'exponentialFromStart', 'exponentialFromEnd',
%                  'uShape', 'uAsym'; a function handle f(t) -> profile
%                  applied to the (possibly user-supplied) time vector;
%                  or an explicit length-N numeric vector (passthrough
%                  with length validation).
%
%   Name-Value arguments:
%     'N'              — Output length. Required when w is [] or scalar;
%                        otherwise inferred from numel(w) and validated
%                        if also supplied.
%     'decayRate'      — Non-negative decay rate for 'exponentialFromStart',
%                        'exponentialFromEnd', and 'uShape'. Used as a
%                        default for 'uAsym' when 'decayRateStart' or
%                        'decayRateEnd' are not supplied. Default 1.
%     'decayRateStart' — Decay rate for the primacy component of 'uAsym'.
%                        Falls back to 'decayRate' when []. Default [].
%     'decayRateEnd'   — Decay rate for the recency component of 'uAsym'.
%                        Falls back to 'decayRate' when []. Default [].
%     'alpha'          — Mixing parameter in [0, 1] for 'uShape' and
%                        'uAsym'. alpha = 1 ≡ pure primacy;
%                        alpha = 0 ≡ pure recency; alpha = 0.5 gives a
%                        balanced mix. Default 0.5.
%     't'              — Strictly increasing time index of length N.
%                        When supplied, decay operates over elapsed time
%                        from the relevant endpoint rather than over
%                        position index. Default [] (unit spacing).
%
%   Output:
%     v — Length-N non-negative weight vector (column vector),
%         equal to profile(spec) .* w(:).
%
%   For 'uAsym', decayRateStart has no effect when alpha = 0 (pure
%   recency) and decayRateEnd has no effect when alpha = 1 (pure
%   primacy).
%
%   Examples:
%     v = seqWeights([], 'recency', 'N', 5);         % [0;0;0;0;1]
%     v = seqWeights([], 'exponentialFromEnd', ...
%                    'N', 5, 'decayRate', 0.5);
%     v = seqWeights([], 'uAsym', 'N', 21, ...
%                    'decayRateStart', 0.05, ...
%                    'decayRateEnd',   0.20, 'alpha', 0.4);
%     v = seqWeights([], @(t) 1 ./ (1 + 0.1 * t), 'N', 10);
%     % Apply recency to pre-existing event salience:
%     s = [0.8; 0.5; 1.0; 0.3; 0.9];
%     v = seqWeights(s, 'recency');                  % [0;0;0;0;0.9]
%
%   See also continuity, buildExpTens, addSpectra.

    arguments
        w
        spec
        nvArgs.N = []
        nvArgs.decayRate (1,1) {mustBeNonnegative} = 1
        nvArgs.decayRateStart = []
        nvArgs.decayRateEnd   = []
        nvArgs.alpha (1,1) ...
            {mustBeInRange(nvArgs.alpha, 0, 1)} = 0.5
        nvArgs.t = []
    end

    N = nvArgs.N;

    % Determine N (output length) and normalise w to a column vector
    % of length N.
    if isempty(w)
        if isempty(N)
            error('seqWeights:missingN', ...
                  ['N must be supplied as the ''N'' name-value ' ...
                   'argument when w is empty (all ones).']);
        end
        validateN_(N);
        wCol = ones(N, 1);
    elseif isscalar(w)
        if isempty(N)
            error('seqWeights:missingN', ...
                  ['N must be supplied as the ''N'' name-value ' ...
                   'argument when w is a scalar.']);
        end
        validateN_(N);
        wCol = w * ones(N, 1);
    elseif isnumeric(w) && isvector(w)
        inferredN = numel(w);
        if isempty(N)
            N = inferredN;
        else
            validateN_(N);
            if N ~= inferredN
                error('seqWeights:nMismatch', ...
                      ['N = %d does not match length of w (%d). ' ...
                       'Either omit ''N'' or supply a consistent ' ...
                       'value.'], N, inferredN);
            end
        end
        wCol = w(:);
    else
        error('seqWeights:wInvalid', ...
              ['w must be [], a scalar, or a numeric vector ' ...
               '(got %s).'], mat2str(size(w)));
    end

    % Time index
    if isempty(nvArgs.t)
        tArr = (0:N-1)';
    else
        tArr = nvArgs.t(:);
        if numel(tArr) ~= N
            error('seqWeights:tLengthMismatch', ...
                  't must have length %d (got %d).', ...
                  N, numel(tArr));
        end
        if any(diff(tArr) <= 0)
            error('seqWeights:tNotIncreasing', ...
                  't must be strictly increasing.');
        end
        tArr = tArr - tArr(1);
    end

    % Callable spec
    if isa(spec, 'function_handle')
        profile = spec(tArr);
        profile = profile(:);
        if numel(profile) ~= N
            error('seqWeights:callableLengthMismatch', ...
                  'Callable spec returned length %d; expected %d.', ...
                  numel(profile), N);
        end
        v = profile .* wCol;
        return;
    end

    % Explicit vector passthrough for spec
    if isnumeric(spec)
        profile = spec(:);
        if numel(profile) ~= N
            error('seqWeights:lengthMismatch', ...
                  'Profile vector length must be %d (got %d).', ...
                  N, numel(profile));
        end
        v = profile .* wCol;
        return;
    end

    spec = string(spec);

    % Resolve uAsym decay rates with fallback to decayRate
    if spec == "uAsym"
        if isempty(nvArgs.decayRateStart)
            dStart = nvArgs.decayRate;
        else
            dStart = nvArgs.decayRateStart;
        end
        if isempty(nvArgs.decayRateEnd)
            dEnd = nvArgs.decayRate;
        else
            dEnd = nvArgs.decayRateEnd;
        end
        if ~isscalar(dStart) || dStart < 0
            error('seqWeights:badDecayRateStart', ...
                  'decayRateStart must be a non-negative scalar.');
        end
        if ~isscalar(dEnd) || dEnd < 0
            error('seqWeights:badDecayRateEnd', ...
                  'decayRateEnd must be a non-negative scalar.');
        end
    end

    switch spec
        case "flat"
            profile = ones(N, 1);
        case "primacy"
            profile = zeros(N, 1);
            profile(1) = 1;
        case "recency"
            profile = zeros(N, 1);
            profile(end) = 1;
        case "exponentialFromStart"
            profile = exp(-nvArgs.decayRate * tArr);
        case "exponentialFromEnd"
            profile = exp(-nvArgs.decayRate * (tArr(end) - tArr));
        case "uShape"
            vS = exp(-nvArgs.decayRate * tArr);
            vE = exp(-nvArgs.decayRate * (tArr(end) - tArr));
            profile = nvArgs.alpha * vS + (1 - nvArgs.alpha) * vE;
        case "uAsym"
            vS = exp(-dStart * tArr);
            vE = exp(-dEnd * (tArr(end) - tArr));
            profile = nvArgs.alpha * vS + (1 - nvArgs.alpha) * vE;
        otherwise
            error('seqWeights:unknownSpec', ...
                  ['Unknown weight specification "%s". Expected ' ...
                   '''flat'', ''primacy'', ''recency'', ' ...
                   '''exponentialFromStart'', ''exponentialFromEnd'', ' ...
                   '''uShape'', ''uAsym'', a function handle, ' ...
                   'or an explicit vector.'], spec);
    end

    v = profile .* wCol;
end


function validateN_(N)
    if ~(isnumeric(N) && isscalar(N) && N > 0 && N == round(N))
        error('seqWeights:invalidN', ...
              'N must be a positive integer (got %s).', mat2str(N));
    end
end
