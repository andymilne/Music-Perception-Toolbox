function h = meanOffset(p, w, period, x)
%MEANOFFSET Mean offset (net upward arc) of a weighted circular multiset.
%
%   h = meanOffset(p, w, period) computes, at each query point, the
%   sum of upward arc lengths to all events minus the sum of downward
%   arc lengths, with each arc normalized by the period. By default,
%   query points are 0, 1, ..., period-1:
%
%     h(x_n) = sum_k w_k * [mod(p_k - x_n, period)
%                          - mod(x_n - p_k, period)] / period
%
%   Each event's contribution lies in (-1, 1); the total scales with
%   the sum of weights (i.e., with the number of events for all-ones
%   weights). Positive values indicate that the events are, on balance,
%   concentrated in the upper (clockwise) half relative to x_n;
%   negative values indicate concentration in the lower half.
%
%   In a pitch-class context, this measure formalizes and
%   generalizes what Huron (2008) calls "average pitch height" —
%   the mean pitch of a set of tones, which Huron applies to both
%   melodies and scales (e.g., comparing the average pitch height
%   of major and minor modes). Huron's measure is implicitly
%   mode-dependent (the same pitch-class set yields different
%   averages relative to different modal "tonics"). meanOffset
%   makes this position-dependence explicit: it returns a value
%   for every position around the circle, including non-scale-tone
%   positions. The term "mode height" for a closely related concept
%   in the pitch-class domain is used by Hearne (2020) and
%   Tymoczko (2023).
%
%   The measure captures the "brightness" or "darkness" of a mode as
%   seen from each chromatic position: modes whose tones lie
%   predominantly in the upper half of the circle (larger intervals
%   above the reference) have higher mean offset. For example, among
%   the seven modes of the diatonic scale, Lydian (starting on F in
%   the white-note set) is the brightest — all its intervals are as
%   large as possible — while Locrian (starting on B) is the darkest.
%
%   h = meanOffset(p, w, period, x) evaluates the mean offset at
%   the query points specified in the vector x (in the same units as p
%   and period) instead of at integer positions 0:period-1.
%
%   Batched (v2.1+):
%   hCell = meanOffset(P, W, period) with P an nRows-by-K matrix
%   returns a 1-by-nRows cell array of per-row results. NaN-padded
%   rows are accepted (NaN entries dropped per row); rows with no
%   valid pitches give empty cell entries. Per-row dedup over
%   permutation + period symmetries: structurally-identical
%   canonical inputs share one cached result. Transposition dedup
%   is *not* applied — the mean-offset profile is transposition-
%   equivariant rather than invariant, and post-transform via
%   query-point shifting would not save work for user-supplied x.
%
%   Inputs:
%     p      — Pitch or position values (vector of length K, or
%              nRows-by-K matrix in batched mode).
%              Values are interpreted modulo 'period'.
%     w      — Weights (vector of length K, matrix the same size as p
%              in batched mode, or empty for all ones).
%     period — Period of the circular domain.
%     x      — (Optional) Query points at which to evaluate the mean
%              offset (vector, shared across all rows in batched
%              mode). Default: 0:period-1.
%
%   Output:
%     h      — Mean offset values (row vector in scalar mode,
%              1-by-nRows cell of row vectors in batched mode).
%
%   Examples:
%     % Mean offset of a diatonic scale (12 chromatic positions)
%     h = meanOffset([0, 2, 4, 5, 7, 9, 11], [], 12);
%     bar(0:11, h);
%     xlabel('Pitch class'); ylabel('Mean offset');
%
%     % Fine grid in cents (0.1-cent resolution)
%     x = 0:0.1:1199.9;
%     h = meanOffset([0, 200, 400, 500, 700, 900, 1100], [], 1200, x);
%     plot(x, h);
%
%     % Son clave rhythm (16-step cycle)
%     h = meanOffset([0, 3, 6, 10, 12], [], 16);
%     bar(0:15, h);
%     xlabel('Pulse'); ylabel('Mean offset');
%
%   References:
%     Hearne, L. M. (2020). The Cognition of Harmonic Tonality in
%       Microtonal Scales. PhD thesis, Western Sydney University.
%       Retrieved from http://hdl.handle.net/1959.7/uws:58606
%       (Uses the term "mode height".)
%     Huron, D. (2008). A comparison of average pitch height and
%       interval size in major- and minor-key themes: Evidence
%       consistent with affect-related pitch prosody. Empirical
%       Musicology Review, 3, 59-63.
%       (Established the concept of "average pitch height" for
%       characterizing the pitch register of melodies and scales.)
%     Milne, A. J., Dean, R. T., & Bulger, D. (2023). The effects of
%       rhythmic structure on tapping accuracy. Attention, Perception,
%       & Psychophysics, 85, 2673-2699.
%       (Introduced this predictor as mean_offset — the normalized
%       mean temporal offset between the rhythm's cues and each
%       pulse position.)
%     Tymoczko, D. (2023). Tonality: An Owner's Manual. Oxford
%       University Press.
%       (Uses the term "mode height".)
%
%   See also projCentroid, edges, dftCircular.

% --- Batched dispatch (v2.1+) ---
% If p is a 2-D matrix with both dimensions > 1, treat rows as
% multisets and return a cell of per-row results.
if size(p, 1) > 1 && size(p, 2) > 1
    if nargin < 4
        x = [];
    end
    if nargin < 2
        w = [];
    end
    h = localBatchedMeanOffset(p, w, period, x);
    return;
end

% === Input validation ===

p = p(:).';
K = numel(p);

if nargin < 2 || isempty(w)
    w = ones(1, K);
end
if isscalar(w)
    w = w * ones(1, K);
end
w = w(:).';

if numel(w) ~= K
    error('w must have the same number of entries as p (or be empty).');
end

if nargin < 4 || isempty(x)
    x = 0:period-1;
end
x = x(:).';

% === Compute mean offset ===
% For each query point x_n and each event p_k:
%   upward arc   = mod(p_k - x_n, period)
%   downward arc = mod(x_n - p_k, period)
%   net arc      = upward - downward
%
% Vectorized: p is 1 x K, x is 1 x nQ.
% D(q, k) = mod(p(k) - x(q), period) is the upward arc.

nQ = numel(x);

% Upward arcs: nQ x K
upward = mod(p - x(:), period);

% Downward arcs: nQ x K
downward = mod(x(:) - p, period);

% Weighted sum of (upward - downward), normalized by period
h = ((upward - downward) * w(:)) ./ period;

h = h(:).';

end


% =====================================================================
%  v2.1 unified dispatch helper: batched-raw mode.
% =====================================================================

function hCell = localBatchedMeanOffset(P, W, period, x)
%LOCALBATCHEDMEANOFFSET Per-row mean offset from a 2-D pitch matrix.
%
%   Returns 1-by-nRows cell. Per-row dedup over permutation + period
%   symmetries via a sorted-modular canonical key.

    nRows = size(P, 1);
    hCell = cell(1, nRows);

    haveRowWeights = ~isempty(W) && isequal(size(W), size(P));
    if ~isempty(W) && ~haveRowWeights
        if isvector(W) && numel(W) == size(P, 2)
            W_broadcast = W(:).';
        else
            error('meanOffset:weightShape', ...
                ['In batched mode, w must be empty, a matrix the same size as p, ' ...
                 'or a vector matching the number of pitch columns.']);
        end
    end

    cache = containers.Map('KeyType', 'char', 'ValueType', 'any');

    for k = 1:nRows
        pRow = P(k, :);
        validMask = ~isnan(pRow);
        pK = pRow(validMask);
        if haveRowWeights
            wK = W(k, validMask);
        elseif ~isempty(W)
            wK = W_broadcast(validMask);
        else
            wK = [];
        end
        if isempty(pK)
            hCell{k} = [];
            continue;
        end

        % Canonical key: sort(mod(p, period)) plus matching weights.
        if isempty(wK)
            wKcol = ones(numel(pK), 1);
        else
            wKcol = wK(:);
        end
        pMod = mod(pK(:), period);
        [pSorted, sortIdx] = sort(pMod);
        wSorted = wKcol(sortIdx);
        keyStr = sprintf('%.12g,', pSorted, wSorted);

        if isKey(cache, keyStr)
            hCell{k} = cache(keyStr);
            continue;
        end

        if isempty(x)
            hk = meanOffset(pK(:).', wKcol(:).', period);
        else
            hk = meanOffset(pK(:).', wKcol(:).', period, x);
        end
        hCell{k} = hk;
        cache(keyStr) = hk;
    end
end