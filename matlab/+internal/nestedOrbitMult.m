function mult = nestedOrbitMult(rLevels, symLevels)
%INTERNAL.NESTEDORBITMULT  Order of a nested attribute's tuple-symmetry group.
%
%   MULT = INTERNAL.NESTEDORBITMULT(RLEVELS, SYMLEVELS) returns |G|, the
%   order of the group the build's nested enumeration acts with on the
%   leaf positions of one nested tuple.
%
%   A nested tuple is a tree: RLEVELS(l) nodes of level l - 1 under every
%   level-l node, innermost-outward, so level l carries
%   prod(RLEVELS(l+1:end)) nodes. INTERNAL.NESTEDENUMINDICES symmetrises
%   level l independently at each of those nodes when SYMLEVELS(l) is
%   set, so the group is the iterated wreath product
%
%       G = prod_{l : sym} (S_{RLEVELS(l)}) ^ (prod_{m > l} RLEVELS(m))
%
%   of order prod_{l : sym} RLEVELS(l)!^(prod(RLEVELS(l+1:end))).
%   Ordered levels contribute no factor, so an all-ordered attribute
%   gives 1.
%
%   |G| is the factor by which the build's perm side tiles its comb side
%   (checked structurally as nJ == |G| * nK wherever it is used), which
%   is what makes Bulger's X-side restriction exact on a nested
%   attribute; it is also the factor by which that restriction lowers
%   the centres route's price, so the nested dispatch's cost race must
%   know it.
%
%   Mirror of Python _mobius_inner._nested_orbit_mult.
%
%   See also MOBIUS.CLOSEDFORMATTRCENTRES, INTERNAL.NESTEDCONTRACT.

    rLevels   = double(rLevels(:)).';
    symLevels = logical(symLevels(:)).';
    mult = 1;
    for l = 1:numel(symLevels)
        if ~symLevels(l)
            continue;
        end
        % prod of an empty slice is 1, so the outermost symmetric level
        % contributes r!^1 as it should.
        nodes = prod(rLevels(l + 1:end));
        mult = mult * factorial(rLevels(l)) ^ nodes;
    end
end
