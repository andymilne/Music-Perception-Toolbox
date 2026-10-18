function [count, magnitude] = continuity(seq, x, sigma, nvArgs)
%CONTINUITY Expected backward same-direction run leading up to each
%query, under Gaussian pitch uncertainty, with optional per-event
%salience weights.
%
%   [count, magnitude] = continuity(seq, x, sigma)
%   [count, magnitude] = continuity(..., Name, Value)
%
%   [count, magnitude] = continuity(SEQ, X, sigma, ...) with SEQ a
%   matrix (more than one row and more than one column) is the batched
%   form: each row of SEQ is one sequence, NaN-padded where the
%   sequences differ in length (NaN entries are dropped). X is either a
%   matrix with one row per sequence (a column for one query per
%   sequence), or a scalar or row vector of Q queries shared by every
%   sequence. count and magnitude are nRows x Q; a NaN query, or a row
%   with no valid entry, gives NaN. The results depend only on the
%   sequence's intervals and each query's interval from the sequence's
%   last value, so rows that are transpositions of one another, with
%   their queries transposed with them, are computed once.
%
%   Inputs:
%     seq   — 1-D ordered sequence of pitches or positions (length N);
%             in the batched form, a matrix whose rows are sequences.
%     x     — Query points (scalar or vector of length M); in the
%             batched form, also a matrix with one row per sequence.
%     sigma — Gaussian pitch uncertainty. Use 0 for the discrete limit.
%
%   Name-Value arguments:
%     'w'     — Per-event salience weights. [] (default) broadcasts
%               weight 1 to every event. A scalar broadcasts to a
%               length-N uniform vector. A length-N vector provides
%               per-event salience (in the batched form, shared by
%               every row; a matrix the same size as SEQ gives each
%               row its own). All weights must be non-negative.
%               The salience of difference event k — the interval
%               from seq(k) to seq(k+1) — is the product
%               w(k)·w(k+1) (rolling product of width 2, matching
%               differenceEvents at order 1), interpretable as the
%               probability that both endpoints are perceived. The
%               count and magnitude contributions from interval k
%               are scaled by this salience; the directional-break
%               threshold θ acts on the unweighted sign-product, so
%               weights modulate contribution size without affecting
%               when the backward walk halts.
%     'mode'  — 'strict' (default, θ = 0) or 'lenient' (θ = −1).
%     'theta' — Explicit break threshold in [−1, +1]; overrides mode.
%
%   Outputs:
%     count     — Length-M non-negative vector: expected run length
%                 (nRows x Q in the batched form).
%     magnitude — Length-M signed vector: weighted sum of counted
%                 intervals (nRows x Q in the batched form). Positive for ascending trends, negative
%                 for descending. The ratio magnitude/count gives a
%                 trend-slope measure.
%
%   Notes:
%     Defined only on linearly ordered domains — those where the
%     ordering is inherited from the real line. For pitch, these
%     include pitch heights, pitch intervals (differences), signed
%     interval changes (second differences), and higher-order
%     differences. For time, the analogous sequence starts one level
%     higher — IOIs (differences between successive event times),
%     signed IOI changes, and so on — because event times are by
%     convention monotonically increasing, so direction on raw time
%     stamps is trivially always positive and carries no
%     information relevant to continuity; only from IOIs onward can
%     the sequence change direction. Not applicable to periodic
%     pitch-class data, for which direction is inherently ambiguous
%     on a cycle.
%
%   The smoothed directional-agreement score is
%       a_k = erf(i_k/(2σ)) · erf(i_N/(2σ))
%   where i_k = seq(k+1)−seq(k) and i_N = x − seq(end). Under
%   independent Gaussian pitch uncertainty, a_k is the expected
%   product of the context interval's sign and the query interval's
%   sign. The backward walk accumulates max(a_k, 0) (count) and
%   max(a_k, 0) · i_k (magnitude), breaking when a_k ≤ θ; each
%   per-interval contribution is further scaled by the difference-
%   event weight w(k)·w(k+1) when weights are supplied.
%
%   The difference-event representation and its rolling-product
%   weight propagation are shared with the MAET-with-differencing
%   pipeline: continuity consumes the same (pAttrDiff, wDiff) stream
%   that differenceEvents produces at order 1, but reads it as an
%   *ordered* sequence with a directional gate and a break condition,
%   rather than aggregating it order-free into a tensor.
%
%   Examples:
%     % Simple ascending trend
%     [c, m] = continuity([3 5 7 7 9], 11, 0, 'mode', 'lenient');
%     % c = 3, m = 6 (three ascending steps totalling +6)
%
%     % Recency zero-out: only the most-recent interval contributes
%     [c, m] = continuity([3 5 7 7 9], 11, 0, 'mode', 'lenient', ...
%                         'w', [0 0 0 1 1]);
%     % c = 1, m = 2 (the 7->9 interval is the only one with non-
%     % zero difference-event salience)
%
%   See also DIFFERENCEEVENTS, SEQWEIGHTS.

    arguments
        seq {mustBeNumeric}
        x {mustBeNumeric}
        sigma (1,1) {mustBeNonnegative}
        nvArgs.w = []
        nvArgs.mode (1,1) string ...
            {mustBeMember(nvArgs.mode, ["strict", "lenient"])} = "strict"
        nvArgs.theta = []
    end

    if isempty(nvArgs.theta)
        if nvArgs.mode == "strict"
            theta = 0;
        else
            theta = -1;
        end
    else
        theta = double(nvArgs.theta);
        if theta < -1 - 1e-12 || theta > 1 + 1e-12
            error('continuity:thetaOutOfRange', ...
                  'theta must be in [-1, +1] (got %g).', theta);
        end
    end

    if size(seq, 1) > 1 && size(seq, 2) > 1
        [count, magnitude] = localBatchedContinuity(seq, x, sigma, ...
                                                    nvArgs.w, theta);
        return;
    end

    [count, magnitude] = localContinuity(seq(:), x(:), sigma, ...
                                         nvArgs.w, theta);
end


function [count, magnitude] = localContinuity(seq, x, sigma, w, theta)
%LOCALCONTINUITY continuity for one sequence (column) and a column of
%   queries.

    N = numel(seq);
    M = numel(x);
    count = zeros(M, 1);
    magnitude = zeros(M, 1);
    if N < 2
        return;
    end

    % --- Normalise weights to [] or a length-N column vector ---
    wVec = localNormaliseWeights(w, N);

    % --- Compute difference events and their weights via
    %     differenceEvents, so the per-event-salience reading and
    %     rolling-product rule match the MAET preprocessing helper
    %     exactly. ---
    seqRow = seq(:).';
    if isempty(wVec)
        [pD, ~, ~] = unpackPreMaet(differenceEvents({seqRow}, [], 1));
        diffWeights = [];
    else
        [pD, wD, ~] = unpackPreMaet(differenceEvents({seqRow}, {wVec(:).'}, 1));
        diffWeights = wD{1}(:);  % (N-1) x 1
    end
    ctxIntervals = pD{1}(:);  % (N-1) x 1

    d_ctx = localErf(ctxIntervals, sigma);

    for i = 1:M
        i_N = x(i) - seq(end);
        d_N = localErf(i_N, sigma);
        a = d_ctx * d_N;
        c = 0;
        mg = 0;
        for k = N - 1 : -1 : 1
            a_k = a(k);
            if a_k <= theta
                break;
            end
            contrib = max(a_k, 0);
            if ~isempty(diffWeights)
                contrib = contrib * diffWeights(k);
            end
            c = c + contrib;
            mg = mg + contrib * ctxIntervals(k);
        end
        count(i) = c;
        magnitude(i) = mg;
    end
end


function [count, magnitude] = localBatchedContinuity(S, x, sigma, w, theta)
%LOCALBATCHEDCONTINUITY continuity for each row of a sequence matrix.
%
%   Rows are keyed by their intervals, their weights, and their
%   queries' intervals from the last value (rounded to 9 decimal places
%   to absorb the noise of the subtraction), so transposed copies,
%   queries included, share one computation.

    [nRows, N] = size(S);

    if size(x, 1) == nRows
        X = x;
        perRow = true;
    elseif size(x, 1) == 1
        X = x;
        perRow = false;
    else
        error('continuity:queryShape', ...
              ['In the batched form, x must be a scalar, a row vector ' ...
               'of queries shared by every sequence, or a matrix with ' ...
               'one row per sequence (%d).'], nRows);
    end
    Q = size(X, 2);

    W = w;
    haveRowWeights = ~isempty(W) && isequal(size(W), size(S));
    if ~isempty(W) && ~haveRowWeights
        if isscalar(W)
            wBroadcast = repmat(double(W), 1, N);
        elseif isvector(W) && numel(W) == N
            wBroadcast = double(W(:).');
        else
            error('continuity:badWeightLength', ...
                  ['In the batched form, w must be empty, a scalar, a ' ...
                   'vector of length N = %d shared by every row, or a ' ...
                   'matrix the same size as seq.'], N);
        end
    end

    count = NaN(nRows, Q);
    magnitude = NaN(nRows, Q);
    cache = containers.Map('KeyType', 'char', 'ValueType', 'any');
    for k = 1:nRows
        sRow  = S(k, :);
        valid = ~isnan(sRow);
        sK    = sRow(valid);
        if isempty(sK)
            continue;
        end
        if haveRowWeights
            wK = W(k, valid);
        elseif ~isempty(W)
            wK = wBroadcast(valid);
        else
            wK = [];
        end
        if perRow
            xK = X(k, :);
        else
            xK = X;
        end
        ok  = ~isnan(xK);
        ref = sK(end);

        key = [sprintf('%.17g,', round(sK - ref, 9)), '|', ...
               sprintf('%.17g,', wK), '|', ...
               sprintf('%.17g,', round(xK - ref, 9))];
        if isKey(cache, key)
            stored = cache(key);
        else
            cK = NaN(1, Q);
            mK = NaN(1, Q);
            if any(ok)
                [cOk, mOk] = localContinuity(sK(:), xK(ok).', sigma, ...
                                             wK, theta);
                cK(ok) = cOk;
                mK(ok) = mOk;
            end
            stored = {cK, mK};
            cache(key) = stored;
        end
        count(k, :) = stored{1};
        magnitude(k, :) = stored{2};
    end
end


function y = localErf(z, sigma)
%LOCALERF Smoothed sign under Gaussian pitch uncertainty.
    if sigma <= 0
        y = sign(z);
    else
        y = erf(z ./ (2 * sigma));
    end
end


function wVec = localNormaliseWeights(w, N)
%LOCALNORMALISEWEIGHTS Coerce the w name-value input to [] or an
%N x 1 non-negative column vector.
%
%  [] (or empty)       -> [] (downstream treats as all ones)
%  non-negative scalar -> N x 1 of that scalar (rolling product of
%                         width 2 then gives difference-event
%                         salience scalar^2 at every position)
%  N-element vector    -> N x 1 as-is, non-negativity enforced
    if isempty(w)
        wVec = [];
        return;
    end
    if ~isnumeric(w)
        error('continuity:badWeightType', ...
              'w must be numeric.');
    end
    if any(w(:) < 0)
        error('continuity:negativeWeights', ...
              'w must be non-negative.');
    end
    if isscalar(w)
        wVec = double(w) * ones(N, 1);
        return;
    end
    if numel(w) ~= N
        error('continuity:badWeightLength', ...
              'w must have length N = %d (got length %d).', ...
              N, numel(w));
    end
    wVec = double(w(:));
end
