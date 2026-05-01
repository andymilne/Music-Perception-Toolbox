function [R, rPhase, rLag] = circApm(p, w, period, nvArgs)
%CIRCAPM Circular autocorrelation phase matrix.
%
%   R = circApm(p, w, period) returns the circular autocorrelation
%   phase matrix (APM) of the weighted multiset p within a cycle of
%   length period (p represents pitches or positions). The APM
%   decomposes the circular autocorrelation into contributions at
%   each combination of lag and phase, adapting
%   the method described by Eck (2006) for non-circular sequences to
%   the circular (periodic) case.
%
%   R is a period x period matrix. R(l+1, phi+1) is the contribution
%   to the autocorrelation from lag l and phase phi (both 0-indexed;
%   MATLAB's 1-based indexing means row 1 is lag 0, column 1 is
%   phase 0).
%
%   For each lag l and phase phi, the function samples the weighted
%   indicator vector at an arithmetic progression with step l starting
%   from phi:
%
%     pos(n) = mod(l*n + phi, period)    for n = 0, 1, ..., period-1
%
%   and computes:
%
%     R(l+1, phi+1) = sum_n s(pos(n)) * s(pos(n+1))
%
%   where s is the weighted indicator vector (s(j) = w_k if event k
%   is at position j, otherwise 0), and pos(n+1) = mod(l*(n+1) + phi,
%   period).
%
%   [R, rPhase] = circApm(...) also returns the column sum of the
%   APM (a 1 x period row vector). This can be used as a model of
%   metrical weight (Parncutt 1994): positions with high rPhase
%   values are strong candidates for perceived beat locations.
%
%   [R, rPhase, rLag] = circApm(...) also returns the row sum of the
%   APM (a period x 1 column vector). This equals the circular
%   autocorrelation of the weighted indicator vector.
%
%   Inputs:
%     p      — Pitch or position values (vector of length K). Must be
%              non-negative integers less than period.
%     w      — Weights (vector of length K, or empty for all ones).
%     period — Length of the cycle (positive integer, e.g., 16 for a
%              16-step rhythmic loop).
%
%   Name-Value Arguments:
%     'decay' — Exponential decay rate (default: 0, no decay). When
%               decay > 0, each sample in the arithmetic progression
%               is weighted by exp(-decay * idx), where idx is the
%               unwrapped (non-circular) index. This models a
%               listener who attends more to recent events: early
%               positions in the progression contribute more than
%               later ones.
%
%   Outputs:
%     R      — Autocorrelation phase matrix (period x period).
%              Row l+1 corresponds to lag l; column phi+1 corresponds
%              to phase phi.
%     rPhase — Column sum of R (1 x period). Model of metrical
%              weight.
%     rLag   — Row sum of R (period x 1). Circular autocorrelation.
%
%   Examples:
%     % APM of a son clave pattern (16-step cycle)
%     R = circApm([0, 3, 6, 10, 12], [], 16);
%     imagesc(0:15, 0:15, R);
%     xlabel('Phase'); ylabel('Lag');
%     title('Autocorrelation phase matrix');
%
%     % Metrical weight profile
%     [~, rPhase] = circApm([0, 3, 6, 10, 12], [], 16);
%     bar(0:15, rPhase);
%     xlabel('Phase'); ylabel('Metrical weight');
%
%     % With exponential decay
%     R = circApm([0, 3, 6, 10, 12], [], 16, 'decay', 0.1);
%
%   References:
%     Eck, D. (2006). Beat tracking using an autocorrelation phase
%       matrix. In Proceedings of the International Computer Music
%       Conference (ICMC).
%     Milne, A. J., Dean, R. T., & Bulger, D. (2023). The effects of
%       rhythmic structure on tapping accuracy. Attention, Perception,
%       & Psychophysics, 85, 2673-2699.
%       (Generalized Eck's non-circular APM to circular (periodic)
%       sequences.)
%     Parncutt, R. (1994). A perceptual model of pulse salience and
%       metrical accent in musical rhythms. Music Perception, 11(4),
%       409-464.
%
%   See also edges, markovS, meanOffset, projCentroid.

    arguments
        p (:,1) {mustBeNonnegative, mustBeInteger}
        w (:,1) {mustBeNumeric}
        period (1,1) {mustBePositive, mustBeInteger}
        nvArgs.decay (1,1) {mustBeNonnegative} = 0
    end

    % === Build weighted indicator vector ===

    K = numel(p);

    if isempty(w)
        w = ones(K, 1);
    end
    if isscalar(w)
        w = w * ones(K, 1);
    end

    if numel(w) ~= K
        error('w must have the same number of entries as p (or be empty).');
    end

    if any(p >= period)
        error('All positions in p must be less than period.');
    end

    N = period;
    s = zeros(1, N);
    for i = 1:K
        s(p(i) + 1) = s(p(i) + 1) + w(i);
    end

    decay = nvArgs.decay;

    % === Compute the APM ===
    % For each lag l and phase phi, sample the indicator at an
    % arithmetic progression with step l starting from phi, and
    % compute the dot product of consecutive pairs.

    R = zeros(N, N);
    steps = 0:N-1;

    for l = 0:N-1
        for phi = 0:N-1
            % Positions in the arithmetic progression
            pos1 = mod(l * steps + phi, N) + 1;
            pos2 = mod(l * (steps + 1) + phi, N) + 1;

            % Sample the weighted indicator
            s1 = s(pos1);
            s2 = s(pos2);

            % Apply exponential decay if requested
            if decay > 0
                idx1 = l * steps + phi;
                idx2 = l * (steps + 1) + phi;
                s1 = s1 .* exp(-decay * idx1);
                s2 = s2 .* exp(-decay * idx2);
            end

            R(l + 1, phi + 1) = s1 * s2';
        end
    end

    % === Summary statistics ===

    if nargout > 1
        rPhase = sum(R, 1);   % column sum: metrical weight
    end
    if nargout > 2
        rLag = sum(R, 2);     % row sum: autocorrelation
    end

end