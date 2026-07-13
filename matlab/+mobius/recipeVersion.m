function v = recipeVersion()
%MOBIUS.RECIPEVERSION  Version of the contraction-recipe builder.
%
%   V = MOBIUS.RECIPEVERSION() returns the version number embedded in
%   recipes built by MOBIUS.BUILDCONTRACTRECIPE. Cached orbit tables
%   (.mat files, in-memory) carry recipes from whichever builder
%   produced them; MOBIUS.GETORBITTABLE rebuilds recipes whose version
%   predates this value on first load, so improvements to the
%   builder's contraction ordering take effect without regenerating
%   the shipped table files. Increment when the builder's ordering
%   algorithm or the recipe struct layout changes.
%
%   Version 2: two-phase pair picker (subset absorption, then
%   minimal-result-rank greedy), bounding every intermediate to
%   non-free rank 2 for all orbits up to r = 6.
%
%   See also MOBIUS.BUILDCONTRACTRECIPE, MOBIUS.GETORBITTABLE.

    v = 2;
end
