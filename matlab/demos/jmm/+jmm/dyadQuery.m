function dens = dyadQuery(flag, rInner)
%DYADQUERY  Dyad-skeleton query density (with optional root-position flag).
%
%   dens = jmm.dyadQuery()
%   dens = jmm.dyadQuery(flag)
%   dens = jmm.dyadQuery(flag, rInner)
%
%   The minimal cadential prototype B-F -> C-E (the tritone-to-major-third
%   dyad skeleton), each chord at unit weights, bound and nested by
%   jmm.boundDensity; flag ([] for none) adds the inversion attribute.
%
%   Twin of bwv_window.query in the Python demos.
%
%   See also JMM.BOUNDDENSITY, JMM.BWVWINDOWSTATE.
    if nargin < 1, flag = []; end
    if nargin < 2 || isempty(rInner), rInner = 1; end
    S = jmm.bwvWindowState();
    aggs = struct('p', {}, 'w', {});
    for j = 1:numel(S.dyadChords)
        aggs(j).p = S.dyadChords{j};
        aggs(j).w = ones(size(S.dyadChords{j}));
    end
    dens = jmm.boundDensity(aggs, flag, rInner);
end
