function r = roughness(f, w, nvArgs)
%ROUGHNESS Sensory roughness of a weighted multiset of partials.
%
%   r = roughness(f, w) returns the total sensory roughness of a weighted
%   multiset of partials with frequencies f (in Hz) and amplitudes/weights
%   roughness is computed by summing the pairwise roughness contributions
%   of all partial pairs, using Sethares' (1993) parameterization of
%   Plomp and Levelt's (1965) empirical dissonance curve.
%
%   r = roughness(f, w, Name, Value) specifies additional options using
%   one or more name-value arguments.
%
%   R = roughness(F, W, ...) with F a matrix (more than one row and more
%   than one column) is the batched form: each row of F is one multiset,
%   NaN-padded where the multisets differ in size, and R is a column
%   with one roughness per row (NaN for a row with no valid frequency).
%   Rows holding the same frequencies and weights, in any order, are
%   computed once. Roughness depends on absolute frequency, so
%   transpositions are not merged.
%
%   The pairwise roughness between two partials at frequencies f1 < f2
%   with amplitudes a1, a2 is:
%     d = min(a1,a2) * (C1*exp(A1*s*(f2-f1)) + C2*exp(A2*s*(f2-f1)))
%   where s = Dstar / (S1*f1 + S2) scales the frequency difference by
%   the critical bandwidth at the lower frequency. The total roughness
%   is the p-norm of all pairwise roughnesses.
%
%   IMPORTANT: Frequencies must be in Hz (not cents, MIDI, or other
%   pitch scales). Use transformAttributes to convert from other scales:
%     f_hz = transformAttributes(f_midi, [], {'midi', 'hz'});
%     r = roughness(f_hz, w);
%
%   Inputs:
%     f — Frequencies in Hz (vector of length K). Must be positive. In
%         the batched form, a matrix whose rows are multisets.
%     w — Amplitudes/weights (vector of length K, or empty for
%         uniform). The minimum of each pair's weights scales the
%         pairwise roughness, so these are typically linear amplitudes
%         (not dB). In the batched form, empty, a matrix the same size
%         as F, or a vector with one entry per column of F, shared by
%         every row.
%
%   Name-Value Arguments:
%     'spectrum' — Cell of addSpectra arguments (for example
%                 {'harmonic', 16, 'powerlaw', 1}). Each frequency is
%                 then the fundamental of a complex tone with these
%                 partials, and the roughness is that of all the
%                 partials together. {} (default): the frequencies are
%                 the partials themselves.
%     'pNorm'   — Norm exponent for combining pairwise roughnesses
%                 (default: 1). With pNorm = 1, the total roughness is
%                 a simple sum. Higher values give greater weight to the
%                 roughest pairs.
%     'average' — Logical (default: false). If true, divides the total
%                 roughness by the number of partial pairs C(K, 2).
%                 When pNorm = 1, this gives the expected roughness of
%                 a randomly chosen partial pair.
%
%   Output:
%     r — Total (or average) roughness (scalar, non-negative); in the
%         batched form, a column with one value per row of F.
%
%   Examples:
%     % Roughness of a dyad at 300 and 330 Hz (equal amplitude)
%     r = roughness([300 330], [1 1]);
%
%     % Roughness of a harmonic complex tone (6 harmonics of 220 Hz)
%     f = 220 * (1:6);
%     w = 1 ./ (1:6);  % amplitude rolloff
%     r = roughness(f, w);
%
%     % Average pairwise roughness
%     r = roughness(f, w, 'average', true);
%
%     % Using transformAttributes to convert from MIDI
%     f = transformAttributes([60 64 67], [], {'midi', 'hz'});  % C major triad
%     r = roughness(f, [1 1 1]);
%
%     % The same triad as three harmonic complex tones
%     r = roughness(f, [], 'spectrum', {'harmonic', 16, 'powerlaw', 1});
%
%     % A batch: one chord per row, NaN-padded
%     F = transformAttributes([60 64 67 NaN; 60 63 67 NaN; 60 64 67 70], ...
%                             [], {'midi', 'hz'});
%     R = roughness(F, [], 'spectrum', {'harmonic', 16, 'powerlaw', 1});
%
%   References:
%     Plomp, R. & Levelt, W. J. M. (1965). Tonal consonance and
%       critical bandwidth. Journal of the Acoustical Society of
%       America, 38(4), 548-560.
%     Sethares, W. A. (1993). Local consonance and the relationship
%       between timbre and scale. Journal of the Acoustical Society of
%       America, 94(3), 1218-1228.
%
%   See also ADDSPECTRA, CONVERTPITCH.

    arguments
        f {mustBeNumeric}
        w {mustBeNumeric} = []
        nvArgs.spectrum = {}
        nvArgs.pNorm (1,1) {mustBePositive} = 1
        nvArgs.average (1,1) logical = false
    end

    spec = nvArgs.spectrum;
    if ~iscell(spec)
        error('roughness:badSpectrum', ...
              'spectrum must be a cell of addSpectra arguments, or {}.');
    end

    % === Batched form: a matrix, one multiset per row ===
    if size(f, 1) > 1 && size(f, 2) > 1
        r = localBatchedRoughness(f, w, spec, nvArgs.pNorm, ...
                                  nvArgs.average);
        return;
    end

    % === Single multiset ===
    f = f(:);
    w = w(:);
    if any(~(f > 0))
        error('roughness:nonPositive', 'Frequencies must be positive.');
    end

    K = numel(f);

    if isempty(w)
        w = ones(K, 1);
    end
    if isscalar(w)
        if w == 0
            warning('All weights in w are zero.');
        end
        w = w * ones(K, 1);
    end

    if numel(w) ~= K
        error('w must have the same number of entries as f (or be empty).');
    end

    if ~isempty(spec)
        [f, w] = localSpectralPartials(f, w, spec);
    end
    r = localRoughnessCore(f, w, nvArgs.pNorm, nvArgs.average);

end


function [f, w] = localSpectralPartials(f, w, spec)
%LOCALSPECTRALPARTIALS Partials of complex tones on fundamentals f (Hz),
%   via addSpectra applied in cents; returned in Hz.
    [c, w] = addSpectra(1200 * log2(f(:)), w(:), spec{:});
    f = 2 .^ (c(:) / 1200);
    w = w(:);
end


function r = localRoughnessCore(f, w, pNorm, average)
%LOCALROUGHNESSCORE Roughness of one multiset of partials (Hz, columns).

    K = numel(f);

    % === Sethares (1993) parameters ===

    Dstar = 0.24;
    S1    = 0.0207;
    S2    = 18.96;
    C1    =  5;
    C2    = -5;
    A1    = -3.51;
    A2    = -5.75;

    % === All pairwise differences and minima ===
    % For each ordered pair (i, j) with f(i) < f(j), we need:
    %   fDiff = f(j) - f(i)          (frequency difference)
    %   fMin  = f(i)                  (lower frequency)
    %   wMin  = min(w(i), w(j))       (minimum amplitude)

    fDiff = f - f';     % K x K matrix
    fMin  = min(f, f'); % K x K: element-wise min
    wMin  = min(w, w'); % K x K: element-wise min

    % Keep only upper triangle (pairs where fDiff > 0)
    mask  = fDiff(:) > 0;
    fDiff = fDiff(mask);
    fMin  = fMin(mask);
    wMin  = wMin(mask);

    % === Pairwise roughnesses ===
    % The critical-bandwidth scaling factor s decreases with frequency,
    % so the same Hz difference produces less roughness at higher
    % frequencies (wider critical bands).

    s = Dstar ./ (S1 * fMin + S2);
    pairRough = wMin .* (C1 * exp(A1 * s .* fDiff) ...
                       + C2 * exp(A2 * s .* fDiff));

    % === Combine via p-norm ===

    r = sum(pairRough .^ pNorm) ^ (1 / pNorm);

    if average
        nPairs = nchoosek(K, 2);
        r = r / nPairs;
    end

end


function r = localBatchedRoughness(F, W, spec, pNorm, average)
%LOCALBATCHEDROUGHNESS One roughness per row of a frequency matrix.
%
%   Rows are keyed by their sorted (frequency, weight) pairs, so
%   repeated rows -- in any order -- share one computation. NaN-padded
%   rows are accepted; a row with no valid frequency gives NaN.

    nRows = size(F, 1);

    haveRowWeights = ~isempty(W) && isequal(size(W), size(F));
    if ~isempty(W) && ~haveRowWeights
        if isvector(W) && numel(W) == size(F, 2)
            wBroadcast = W(:).';
        else
            error('roughness:weightShape', ...
                  ['In the batched form, w must be empty, a matrix the ' ...
                   'same size as f, or a vector with one entry per ' ...
                   'column of f.']);
        end
    end

    r = NaN(nRows, 1);
    cache = containers.Map('KeyType', 'char', 'ValueType', 'double');
    for k = 1:nRows
        fRow  = F(k, :);
        valid = ~isnan(fRow);
        fK    = fRow(valid);
        if isempty(fK)
            continue;
        end
        if any(fK <= 0)
            error('roughness:nonPositive', ...
                  'Frequencies must be positive; row %d is not.', k);
        end
        if haveRowWeights
            wK = W(k, valid);
        elseif ~isempty(W)
            wK = wBroadcast(valid);
        else
            wK = ones(size(fK));
        end

        fw  = sortrows([fK(:), wK(:)]);
        key = [sprintf('%.17g,', fw(:, 1)), ';', sprintf('%.17g,', fw(:, 2))];
        if isKey(cache, key)
            r(k) = cache(key);
            continue;
        end

        fC = fw(:, 1);
        wC = fw(:, 2);
        if ~isempty(spec)
            [fC, wC] = localSpectralPartials(fC, wC, spec);
        end
        r(k) = localRoughnessCore(fC, wC, pNorm, average);
        cache(key) = r(k);
    end
end
