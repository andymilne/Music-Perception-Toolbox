function [F, mag] = dftCircular(p, w, period)
%DFTCIRCULAR Discrete Fourier transform of a weighted circular multiset.
%
%   [F, mag] = dftCircular(p, w, period):
%
%   Computes the DFT of a weighted multiset of K points distributed
%   around a circle of circumference 'period' (p represents pitches or
%   positions).
%
%   The procedure is:
%     1. Sort p ascending (reordering w to match).
%     2. Map each element to the unit circle and scale by its weight:
%          z(j) = w(j) * exp(2*pi*1i*p(j)/period).
%     3. Compute the DFT of z, normalized by the sum of weights:
%          F(k) = (1/sum(w)) * sum_j z(j) * exp(-2*pi*1i * (j-1) * k / K)
%        for k = 0, 1, ..., K-1.
%        (For uniform weights, sum(w) = K.)
%
%   The magnitudes of the coefficients have well-established music-
%   theoretical interpretations (see balanceCircular and evennessCircular):
%     |F(0)|: imbalance — distance of the centre of gravity from the
%             origin. When |F(0)| = 0, the multiset is perfectly balanced.
%     |F(1)|: evenness — closeness to a maximally even (equal-step)
%             distribution. When |F(1)| = 1, the multiset is maximally
%             even.
%   Higher coefficients capture additional distributional properties.
%
%   For further information, see:
%     Milne, A. J., Bulger, D., & Herff, S. A. (2017). Exploring the
%       space of perfectly balanced rhythms and scales. Journal of
%       Mathematics and Music, 11(2-3), 101-133.
%     Milne, A. J. & Herff, S. A. (2020). The perceptual relevance of
%       balance, evenness, and entropy in musical rhythms. Cognition,
%       203, 104233.
%
%   Inputs:
%     p      — Pitch or position values (vector of length K, or
%              nRows-by-K matrix in batched mode). Values are
%              interpreted modulo 'period'. The function sorts p
%              internally; the caller does not need to pre-sort.
%     w      — Weights (vector of length K, matrix the same size as p
%              in batched mode, or empty for all ones).
%     period — Period of the circular domain (e.g., 1200 for one octave
%              in cents, or the cycle length for rhythmic patterns).
%
%   Outputs:
%     F      — Complex Fourier coefficients (1-by-K row vector in
%              scalar mode; 1-by-nRows cell in batched mode).
%     mag    — Magnitudes |F(k)|, same shape as F.
%
%   Examples:
%     % DFT of a 12-EDO diatonic scale (in cents)
%     [F, mag] = dftCircular([0, 200, 400, 500, 700, 900, 1100], [], 1200);
%     fprintf('Balance = %.3f, Evenness = %.3f\n', 1 - mag(1), mag(2));
%
%     % DFT of a rhythmic pattern (onsets in a 16-step cycle)
%     [F, mag] = dftCircular([0, 3, 6, 8, 10, 12, 14], [], 16);
%
%     % Batched: DFT of multiple scales
%     P = [0, 200, 400, 500, 700, 900, 1100;       % major
%          0, 200, 300, 500, 700, 800, 1000];      % natural minor
%     [Fcell, magCell] = dftCircular(P, [], 1200);
%
%   Batched:
%   When p is a 2-D nRows-by-K matrix, each row is treated as a
%   separate multiset and the function returns 1-by-nRows cell
%   arrays. NaN-padded rows are accepted (NaN entries dropped per
%   row); rows with no valid pitches give empty cell entries. Per-row
%   canonical-form dedup over permutation + period symmetries:
%   structurally-identical canonical inputs (sorted modular pitches
%   plus sorted matching weights) share one cached pair. Dedup over
%   transposition is *not* applied — the DFT is transposition-
%   equivariant rather than invariant, so transposed inputs would
%   need a phase post-transform; this is left as a future
%   optimisation.
%
%   See also balanceCircular, evennessCircular.

% --- Batched dispatch ---
% If p is a 2-D matrix with both dimensions > 1, treat rows as
% multisets and return cell arrays of per-row results.
if size(p, 1) > 1 && size(p, 2) > 1
    [F, mag] = localBatchedDftCircular(p, w, period);
    return;
end

% === Input validation ===

p = p(:);
K = numel(p);

if isempty(w)
    w = ones(K, 1);
end
if isscalar(w)
    w = w * ones(K, 1);
end
w = w(:);

if numel(w) ~= K
    error('w must have the same number of entries as p (or be empty).');
end

% === Sort by pitch class ===

[p, sortIdx] = sort(p);
w = w(sortIdx);

% === Map to unit circle and compute DFT ===

z = w .* exp(2 * pi * 1i * p / period);  % K x 1, weighted
F = fft(z).' / sum(w);                    % 1 x K row vector
                                           % (sum(w) = K for all-ones weights)

% === Magnitudes ===

mag = abs(F);

end


% =====================================================================
%  Unified dispatch helper: batched-raw mode.
% =====================================================================

function [Fcell, magCell] = localBatchedDftCircular(P, W, period)
%LOCALBATCHEDDFTCIRCULAR Per-row DFT from a 2-D pitch matrix.
%
%   Returns 1-by-nRows cell arrays. Per-row dedup over permutation +
%   period symmetries via a sorted-modular canonical key.

    nRows = size(P, 1);
    Fcell = cell(1, nRows);
    magCell = cell(1, nRows);

    haveRowWeights = ~isempty(W) && isequal(size(W), size(P));
    if ~isempty(W) && ~haveRowWeights
        if isvector(W) && numel(W) == size(P, 2)
            W_broadcast = W(:).';
        else
            error('dftCircular:weightShape', ...
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
            Fcell{k} = [];
            magCell{k} = [];
            continue;
        end

        % Canonical key: sort(mod(p, period)) plus matching weights.
        % This collapses permutations and period-equivalent inputs
        % onto one representative (no transposition dedup — see
        % function help).
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
            stored = cache(keyStr);
            Fcell{k}   = stored{1};
            magCell{k} = stored{2};
            continue;
        end

        [Fk, magk] = dftCircular(pK(:), wKcol, period);
        Fcell{k}   = Fk;
        magCell{k} = magk;
        cache(keyStr) = {Fk, magk};
    end
end
