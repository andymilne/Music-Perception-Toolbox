function pm = asCompared(pm)
%ASCOMPARED  A bound pre-MAET without its time attribute, the window attribute.
%
%   pm = jmm.asCompared(pm)
%
%   What the comparison actually receives, once the sweep has used the time to
%   place the window.
%
%   See also SELECTPREMAET, WINDOWEDSIMILARITY.
    [~, ~, specs] = unpackPreMaet(pm);
    names = cellfun(@(sp) sp.name, specs, 'UniformOutput', false);
    pm = selectPreMaet(pm, 'attributes', names(~strcmp(names, 'onset')));
end
