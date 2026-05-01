function v = seqWeights(w, spec, nvArgs)
%SEQWEIGHTS Apply a position-weighting profile to an existing weight
%vector.
%
%   v = seqWeights(w, spec)
%   v = seqWeights(w, spec, 'N', N)
%   v = seqWeights(..., Name, Value)
%
%   Constructs a length-N profile from the named or explicit
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
%                  'uShape'; or an explicit length-N numeric vector
%                  (passthrough with length validation).
%
%   Name-Value arguments:
%     'N'         — Output length. Required when w is [] or scalar;
%                   otherwise inferred from numel(w) and validated
%                   if also supplied.
%     'decayRate' — Non-negative decay rate for exponential and
%                   uShape specs. Zero decay gives uniform profile.
%                   Default 1.
%     'alpha'     — Mixing parameter in [0, 1] for 'uShape'.
%                   alpha = 1 ≡ exponentialFromStart;
%                   alpha = 0 ≡ exponentialFromEnd;
%                   alpha = 0.5 gives the symmetric U. Default 0.5.
%     't'         — Strictly increasing time index of length N.
%                   When supplied, decay operates over elapsed time
%                   from the relevant endpoint rather than over
%                   position index. Default [] (unit spacing).
%
%   Output:
%     v — Length-N non-negative weight vector (column vector),
%         equal to profile(spec) .* w(:).
%
%   Examples:
%     v = seqWeights([], 'recency', 'N', 5);         % [0;0;0;0;1]
%     v = seqWeights([], 'exponentialFromEnd', ...
%                    'N', 5, 'decayRate', 0.5);
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
        otherwise
            error('seqWeights:unknownSpec', ...
                  ['Unknown weight specification "%s". Expected ' ...
                   '''flat'', ''primacy'', ''recency'', ' ...
                   '''exponentialFromStart'', ''exponentialFromEnd'', ' ...
                   '''uShape'', or an explicit vector.'], spec);
    end

    v = profile .* wCol;
end


function validateN_(N)
    if ~(isnumeric(N) && isscalar(N) && N > 0 && N == round(N))
        error('seqWeights:invalidN', ...
              'N must be a positive integer (got %s).', mat2str(N));
    end
end
