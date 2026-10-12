function [key, pCanon, wCanon] = chordCanonicalKey(p, w, sigma, r, rel, per, period, truncationSigmas)
%CHORDCANONICALKEY Canonical hashable key for a single weighted multiset.
%   [key, pCanon, wCanon] = internal.chordCanonicalKey(p, w, sigma, r, ...
%                                              rel, per, period)
%   [...] = internal.chordCanonicalKey(..., truncationSigmas)
%
%   This is the MATLAB twin of the Python _chord_canonical_key. Two chords
%   (p1, w1) and (p2, w2) produce the same KEY iff their density
%   object is structurally identical for the given (sigma, r, rel,
%   per, period) — i.e., regardless of input-side permutation or
%   (in relative modes) transposition / rotation around the period.
%
%   Used as the dictionary key for consumer-level deduplication in
%   the harmony-batched dispatchers (templateHarmonicity,
%   virtualPitches, spectralEntropy). For a batch of M rows, this
%   makes the per-row work scale with the number of structurally
%   distinct chords, not with M.
%
%   Inputs
%     p, w               : pitch and weight vectors (w may be []).
%     sigma, r           : density parameters baked into the key.
%     rel, per           : mode flags (drive canonicalisation).
%     period             : period in cents (used iff per).
%     truncationSigmas   : optional; the caller's truncation setting ([]
%                          for the toolbox default), which sets the
%                          rounding of the canonical form
%                          (internal.keyRoundDigits).
%
%   Outputs
%     key      : char row vector usable as a containers.Map key.
%     pCanon   : canonical pitches (1-by-n row).
%     wCanon   : canonical weights (1-by-n row), or [].

    if nargin < 8
        truncationSigmas = [];   % the toolbox default
    end
    % The canonical values are rounded to the places that keep rounding
    % within the accuracy floor set by truncationSigmas.
    roundDigits = internal.keyRoundDigits(sigma, r, truncationSigmas);
    [pCanon, wCanon] = internal.canonicalizeSet(p, w, rel, per, period, ...
                                                roundDigits);

    % The key holds every value at full precision (%.17g), so it
    % encodes exactly the rounded canonical form: two keys agree only
    % where the forms do. (mat2str at 12 significant digits, used
    % before, could merge forms differing by more than the rounding.)
    key = sprintf('p=%s|w=%s|s=%.17g|r=%d|rel=%d|per=%d|P=%.17g', ...
        sprintf('%.17g,', pCanon), sprintf('%.17g,', wCanon), sigma, r, ...
        double(logical(rel)), double(logical(per)), period);
end
