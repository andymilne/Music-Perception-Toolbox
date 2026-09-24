function pm = dyadQuery(flag, rInner)
%DYADQUERY  Dyad-skeleton query density (with optional root-position flag).
%
%   pm = jmm.dyadQuery()
%   pm = jmm.dyadQuery(flag)
%   pm = jmm.dyadQuery(flag, rInner)
%
%   The minimal cadential prototype B-F -> C-E (the tritone-to-major-third
%   dyad skeleton), each chord at unit weights, read exactly as a two-beat
%   window is (jmm.query); flag ([] for none) adds the inversion attribute.
%
%   See also JMM.QUERY, JMM.BOUNDCONTEXT, JMM.BWVWINDOWSTATE.
    if nargin < 1, flag = []; end
    if nargin < 2 || isempty(rInner), rInner = 1; end
    S = jmm.bwvWindowState();
    pm = jmm.query(S.dyadChords, flag, rInner);
end
