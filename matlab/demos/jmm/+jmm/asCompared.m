function pm = asCompared(pm)
%ASCOMPARED  A bound pre-MAET without its placement axis.
%
%   pm = jmm.asCompared(pm)
%
%   What the cosine actually receives, once the sweep has used the time to
%   place the window.
%
%   See also SELECTPREMAET, WINDOWEDSIMILARITY.
    [~, ~, specs] = unpackPreMaet(pm);
    names = cellfun(@(sp) sp.name, specs, 'UniformOutput', false);
    pm = selectPreMaet(pm, 'attributes', names(~strcmp(names, 'onset')));
end
