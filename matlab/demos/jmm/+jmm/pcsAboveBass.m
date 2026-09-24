function pcs = pcsAboveBass(son)
%PCSABOVEBASS  Sorted distinct pitch classes above the bass of a sonority.
%
%   pcs = jmm.pcsAboveBass(son)
%
%   son is a vector of MIDI pitches (NaN entries ignored); the result is
%   the sorted set of round(p - min(p)) mod 12.
    p = son(isfinite(son));
    pcs = unique(mod(round(p - min(p)), 12));
    pcs = pcs(:).';
end
