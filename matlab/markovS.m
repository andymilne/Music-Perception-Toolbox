function y = markovS(p, w, period, S)
%MARKOVS Optimal S-step Markov predictor for a periodic weighted multiset.
%
%   y = markovS(p, w, period) returns the predicted event weight at
%   each integer position 0, 1, ..., period-1 of a cycle of length
%   period, using a 3-step Markov context (the default).
%
%   y = markovS(p, w, period, S) uses an S-step context.
%
%   Y = markovS(P, W, period, ...) with P a matrix (more than one row and
%   more than one column) is the batched form: each row of P is one
%   multiset, NaN-padded where the multisets differ in size, and Y is
%   nRows x period, one prediction per row (a row of NaN for a row with
%   no valid position). The predictor is rotation-equivariant -- rotating
%   a pattern rotates its prediction -- so rows that are rotations of one
%   another are computed once. W is empty, a matrix the same size as P,
%   or a vector with one entry per column of P, shared by every row.
%
%   For each position j in the cycle, the predictor finds all
%   positions i whose S-step future context — the binary pattern of
%   events and non-events over the next S positions — is identical
%   to that of j. The predicted value at j is the average of the
%   weights at all such matching positions. Positions without events
%   have weight 0.
%
%   This is the optimal predictor for a stationary S-th order Markov
%   process observed over one period: given knowledge of the next S
%   positions, it returns the conditional expected value at the
%   current position.
%
%   Context matching is based on binary event/non-event status
%   (whether a position has any event, regardless of weight), while
%   the prediction averages actual weights. This means two positions
%   with different non-zero weights can still share a context.
%
%   Inputs:
%     p      — Pitch or position values (vector of length K). Must be
%              non-negative integers less than period.
%     w      — Weights (vector of length K, or empty for all ones).
%     period — Length of the cycle (positive integer).
%     S      — (Optional) Number of lookahead steps for context
%              matching (positive integer, default: 3). Larger values
%              use more context and give more specific predictions;
%              smaller values average over more positions.
%
%   Output:
%     y      — Predicted event weights (1 x period row vector). y(j)
%              is the predicted weight at position j-1 (0-indexed). In
%              the batched form, nRows x period.
%
%   Examples:
%     % Markov predictor for a son clave pattern (16-step cycle)
%     y = markovS([0, 3, 6, 10, 12], [], 16);
%     bar(0:15, y);
%     xlabel('Position'); ylabel('Predicted weight');
%
%     % With a longer context (S = 5)
%     y = markovS([0, 3, 6, 10, 12], [], 16, 5);
%
%     % With non-uniform weights
%     y = markovS([0, 3, 6, 10, 12], [1, 0.5, 0.8, 0.7, 1], 16);
%
%   Originally by David Bulger, Macquarie University.
%   Adapted for the Music Perception Toolbox v2 by Andrew J. Milne.
%
%   References:
%     Milne, A. J., Dean, R. T., & Bulger, D. (2023). The effects of
%       rhythmic structure on tapping accuracy. Attention, Perception,
%       & Psychophysics, 85, 2673-2699.
%       (Introduced the S-step Markov predictor for rhythmic cue
%       sequences.)
%
%   See also circApm, edges.

% === Input defaults ===

if nargin < 4 || isempty(S)
    S = 3;
end

% === Batched form: a matrix, one multiset per row ===
if size(p, 1) > 1 && size(p, 2) > 1
    y = localBatchedMarkovS(p, w, period, S);
    return;
end

K = numel(p);
p = p(:).';

if isempty(w)
    w = ones(1, K);
end
if isscalar(w)
    w = w * ones(1, K);
end
w = w(:).';

if numel(w) ~= K
    error('w must have the same number of entries as p (or be empty).');
end

if any(p < 0) || any(p >= period) || any(p ~= round(p))
    error('All positions in p must be non-negative integers less than period.');
end

% === Build indicator vectors ===
% Weighted indicator for prediction; binary indicator for context matching.

N = period;
wCycle = zeros(1, N);
for i = 1:K
    wCycle(p(i) + 1) = wCycle(p(i) + 1) + w(i);
end

y = localMarkovFromCycle(wCycle, S);

end


function y = localMarkovFromCycle(wCycle, S)
%LOCALMARKOVFROMCYCLE The S-step Markov prediction from a cycle's
%   weighted indicator (1 x N).

N = numel(wCycle);
binCycle = (wCycle ~= 0);  % binary: event (1) or non-event (0)

% === S-step Markov predictor ===
% E(i, j) = 1 iff positions i and j have the same binary status.
% T(i, j) = 1 iff the S-step future contexts of i and j are identical.

E = binCycle == binCycle';
T = true(N);
for k = 1:S
    T = T & circshift(circshift(E, k, 2), k, 1);
end

% Predict: average the weighted indicator over matching positions.
y = (wCycle * double(T)) ./ sum(T);

end


function Y = localBatchedMarkovS(P, W, period, S)
%LOCALBATCHEDMARKOVS One prediction per row of a position matrix.
%
%   Each row's weighted indicator is brought to its lexicographically
%   smallest rotation; rows sharing that canonical cycle share one
%   prediction, rotated back into place.

nRows = size(P, 1);
N = period;

haveRowWeights = ~isempty(W) && isequal(size(W), size(P));
if ~isempty(W) && ~haveRowWeights
    if isvector(W) && numel(W) == size(P, 2)
        wBroadcast = W(:).';
    else
        error('markovS:weightShape', ...
              ['In the batched form, w must be empty, a matrix the same ' ...
               'size as p, or a vector with one entry per column of p.']);
    end
end

% Row r+1 of a cycle's rotation matrix is the cycle rotated left by r:
% rots(r+1, j+1) = c(mod(j + r, N) + 1).
rotIdx = mod((0:N-1).' + (0:N-1), N) + 1;

Y = NaN(nRows, N);
cache = containers.Map('KeyType', 'char', 'ValueType', 'any');
for k = 1:nRows
    pRow  = P(k, :);
    valid = ~isnan(pRow);
    pK    = pRow(valid);
    if isempty(pK)
        continue;
    end
    if any(pK < 0) || any(pK >= period) || any(pK ~= round(pK))
        error('markovS:badPositions', ...
              ['All positions must be non-negative integers less than ' ...
               'period; row %d is not.'], k);
    end
    if haveRowWeights
        wK = W(k, valid);
    elseif ~isempty(W)
        wK = wBroadcast(valid);
    else
        wK = ones(size(pK));
    end

    wCycle = accumarray(pK(:) + 1, wK(:), [N 1]).';
    rots = wCycle(rotIdx);
    [~, order] = sortrows(round(rots, 12));
    r = order(1) - 1;
    canon = rots(order(1), :);
    key = sprintf('%.17g,', round(canon, 12));
    if isKey(cache, key)
        yC = cache(key);
    else
        yC = localMarkovFromCycle(canon, S);
        cache(key) = yC;
    end
    % wCycle is canon rotated right by r, and so is its prediction.
    Y(k, :) = circshift(yC, r, 2);
end

end
