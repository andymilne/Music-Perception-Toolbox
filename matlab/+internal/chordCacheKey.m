function [key, pCanon, wCanon] = chordCacheKey(p, w, sigma, r, isRel, isPer, period)
%CHORDCACHEKEY Canonical hashable key for a single weighted multiset.
%   [key, pCanon, wCanon] = internal.chordCacheKey(p, w, sigma, r, ...
%                                              isRel, isPer, period)
%
%   Mirrors python/mpt/tensor.py:_chord_canonical_key. Two chords
%   (p1, w1) and (p2, w2) produce the same KEY iff their density
%   object is structurally identical for the given (sigma, r, isRel,
%   isPer, period) — i.e., regardless of input-side permutation or
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
%     isRel, isPer       : mode flags (drive canonicalisation).
%     period             : period in cents (used iff isPer).
%
%   Outputs
%     key      : char row vector usable as a containers.Map key.
%     pCanon   : canonical pitches (1-by-n row).
%     wCanon   : canonical weights (1-by-n row), or [].

    [pCanon, wCanon] = internal.canonicalizeSet(p, w, isRel, isPer, period);

    % mat2str at 12 decimal places is well below FP precision but
    % above the 9-decimal rounding inside cyclicCanonical, so the key
    % faithfully encodes the canonical form.
    if isempty(wCanon)
        wStr = '[]';
    else
        wStr = mat2str(wCanon(:)', 12);
    end
    key = sprintf('p=%s|w=%s|s=%.12g|r=%d|rel=%d|per=%d|P=%.12g', ...
        mat2str(pCanon(:)', 12), wStr, sigma, r, ...
        double(logical(isRel)), double(logical(isPer)), period);
end
