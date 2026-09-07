function dens = buildPair(c1, c2, flag, rInner)
%BUILDPAIR  Context density: two windowed beat aggregates, bound and nested.
%
%   dens = jmm.buildPair(c1, c2)
%   dens = jmm.buildPair(c1, c2, flag)
%   dens = jmm.buildPair(c1, c2, flag, rInner)
%
%   c1 and c2 are window structs from jmm.winEvents (the approach and the
%   resolution beat); flag ([] for none) and rInner (default 1) are passed
%   to jmm.boundDensity.
%
%   Twin of bwv_window.build_pair in the Python demos.
%
%   See also JMM.BOUNDDENSITY, JMM.AGGREGATE, JMM.WINEVENTS.
    if nargin < 3, flag = []; end
    if nargin < 4 || isempty(rInner), rInner = 1; end
    aggs = [jmm.aggregate(c1), jmm.aggregate(c2)];
    dens = jmm.boundDensity(aggs, flag, rInner);
end
