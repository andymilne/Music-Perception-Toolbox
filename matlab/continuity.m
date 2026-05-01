function [count, magnitude] = continuity(seq, x, sigma, nvArgs)
%CONTINUITY Expected backward same-direction run leading up to each
%query, under Gaussian pitch uncertainty, with optional per-event
%salience weights.
%
%   [count, magnitude] = continuity(seq, x, sigma)
%   [count, magnitude] = continuity(..., Name, Value)
%
%   Inputs:
%     seq   — 1-D ordered sequence of pitches or positions (length N).
%     x     — Query points (scalar or vector of length M).
%     sigma — Gaussian pitch uncertainty. Use 0 for the discrete limit.
%
%   Name-Value arguments:
%     'w'     — Per-event salience weights. [] (default) broadcasts
%               weight 1 to every event. A scalar broadcasts to a
%               length-N uniform vector. A length-N vector provides
%               per-event salience. All weights must be non-negative.
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
%     count     — Length-M non-negative vector: expected run length.
%     magnitude — Length-M signed vector: weighted sum of counted
%                 intervals. Positive for ascending trends, negative
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
        seq (:,1) {mustBeNumeric}
        x {mustBeNumeric}
        sigma (1,1) {mustBeNonnegative}
        nvArgs.w = []
        nvArgs.mode (1,1) string ...
            {mustBeMember(nvArgs.mode, ["strict", "lenient"])} = "strict"
        nvArgs.theta = []
    end

    x = x(:);

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

    N = numel(seq);
    M = numel(x);
    count = zeros(M, 1);
    magnitude = zeros(M, 1);
    if N < 2
        return;
    end

    % --- Normalise weights to [] or a length-N column vector ---
    wVec = localNormaliseWeights(nvArgs.w, N);

    % --- Compute difference events and their weights via
    %     differenceEvents, so the per-event-salience reading and
    %     rolling-product rule match the MAET preprocessing helper
    %     exactly. ---
    seqRow = seq(:).';
    if isempty(wVec)
        [pD, ~] = differenceEvents({seqRow}, [], [], 1, 0);
        diffWeights = [];
    else
        [pD, wD] = differenceEvents({seqRow}, {wVec(:).'}, [], 1, 0);
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
