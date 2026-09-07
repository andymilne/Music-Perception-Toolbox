function dens = queryDensity(chords, flagged, rInner)
%QUERYDENSITY  Nested density of a three-chord (or any L-chord) prototype query.
%
%   dens = jmm.queryDensity(chords, flagged, rInner)
%
%   chords is a 1 x L cell of MIDI pitch vectors (each chord at unit
%   weights); flagged (logical) adds the root-position flag (rootYes) as
%   the inversion attribute; rInner is the inner tuple size. The density
%   is built by jmm.boundDensity.
%
%   Twin of _query_density in demo_jmm_1_4_cadence_nesting.py.
%
%   See also JMM.BOUNDDENSITY, JMM.PROTOTYPESWEEP.
    S = jmm.bwvWindowState();
    aggs = struct('p', {}, 'w', {});
    for j = 1:numel(chords)
        aggs(j).p = double(chords{j});
        aggs(j).w = ones(size(chords{j}));
    end
    if flagged, flag = S.rootYes; else, flag = []; end
    dens = jmm.boundDensity(aggs, flag, rInner);
end
