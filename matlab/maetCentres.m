function C = maetCentres(dens)
%MAETCENTRES Tuple centres of an expectation tensor density.
%
%   C = maetCentres(dens) returns the points at which the density places
%   its Gaussians: a 1 x A cell whose a-th entry is a
%   (r_a - isRel_a) x nJ matrix of tuple centres in that attribute's own
%   coordinates, unwrapped on a periodic attribute.
%
%   buildMaet defaults to 'lazy', true and returns a density whose
%   per-tuple fields are not yet built, so this call materialises them
%   when they are absent and passes them through when they are not.
%
%   Inputs
%       dens - Density struct from buildMaet (tag 'MaetDensity').
%
%   Output
%       C    - 1 x A cell of tuple-centre matrices.
%
%   Example
%       dens = buildMaet([0 400 700], [], 10, 2, false, false, 1200);
%       C = maetCentres(dens);       % 2 x 6: the six ordered pairs
%
%   The Python mirror is mpt.maet_centres.
%
%   See also BUILDMAET, EVALMAET, SIMMAET.

if ~isstruct(dens) || ~isfield(dens, 'tag') ...
        || ~strcmp(dens.tag, 'MaetDensity')
    error('maetCentres:badInput', ...
          'Input must be a density struct from buildMaet.');
end

dens = internal.ensureMaetExpensive(dens);
C = dens.Centres;

end
