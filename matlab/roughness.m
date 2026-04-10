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
%   The pairwise roughness between two partials at frequencies f1 < f2
%   with amplitudes a1, a2 is:
%     d = min(a1,a2) * (C1*exp(A1*s*(f2-f1)) + C2*exp(A2*s*(f2-f1)))
%   where s = Dstar / (S1*f1 + S2) scales the frequency difference by
%   the critical bandwidth at the lower frequency. The total roughness
%   is the p-norm of all pairwise roughnesses.
%
%   IMPORTANT: Frequencies must be in Hz (not cents, MIDI, or other
%   pitch scales). Use convertPitch to convert from other scales:
%     f_hz = convertPitch(f_midi, 'midi', 'hz');
%     r = roughness(f_hz, w);
%
%   Inputs:
%     f — Frequencies in Hz (vector of length K). Must be positive.
%     w — Amplitudes/weights (vector of length K, or empty for
%         uniform). The minimum of each pair's weights scales the
%         pairwise roughness, so these are typically linear amplitudes
%         (not dB).
%
%   Name-Value Arguments:
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
%     r — Total (or average) roughness (scalar, non-negative).
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
%     % Using convertPitch to convert from MIDI
%     f = convertPitch([60 64 67], 'midi', 'hz');  % C major triad
%     r = roughness(f, [1 1 1]);
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
        f (:,1) {mustBeNumeric, mustBePositive}
        w (:,1) {mustBeNumeric} = []
        nvArgs.pNorm (1,1) {mustBePositive} = 1
        nvArgs.average (1,1) logical = false
    end

    % === Weight defaults ===

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

    p = nvArgs.pNorm;
    r = sum(pairRough .^ p) ^ (1 / p);

    if nvArgs.average
        nPairs = nchoosek(K, 2);
        r = r / nPairs;
    end

end
