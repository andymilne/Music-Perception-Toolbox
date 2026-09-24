function pm = prototypeQuery(chords, flagged, rInner)
%PROTOTYPEQUERY  A three-chord (or any L-chord) prototype as a bound query.
%
%   pm = jmm.prototypeQuery(chords, flagged, rInner)
%
%   chords is a 1 x L cell of MIDI pitch vectors, converted by jmm.query at
%   unit weights; flagged (logical) adds the root-position flag (rootYes) as
%   the inversion attribute; rInner is the inner tuple size. The one thing
%   it adds over jmm.query is the mapping from the logical flagged to the
%   flag value.
%
%   See also JMM.QUERY, JMM.BOUNDCONTEXT.
    S = jmm.bwvWindowState();
    if flagged, flag = S.rootYes; else, flag = []; end
    pm = jmm.query(chords, flag, rInner);
end
