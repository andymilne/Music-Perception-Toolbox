function tf = hasOrderedAttr(dens)
%HASORDEREDATTR  True if any flat attribute is ordered ([sym] = 0) at r > 1.
%
%   Such an attribute carries no position-permutation symmetry, so the Möbius
%   orbit decomposition does not apply to it: the partition sum realises
%   the symmetrised tuple set, which is a *different* density rather than
%   the same one computed faster. r = 1 is exempt ([sym] is vacuous at a
%   single value), as are nested attributes, whose per-attribute density is
%   built by contraction rather than by a single Möbius sum.
%
%   Twin of python mpt._tensor.dispatch._has_ordered_attr.
%
%   See also INTERNAL.SELECTMAEVAL, MOBIUS.EVALMAORBIT.

    tf = false;
    A = double(dens.nAttrs);

    if isfield(dens, 'isSym') && ~isempty(dens.isSym)
        isSym = logical(dens.isSym(:).');
    else
        isSym = true(1, A);          % default: symmetric
    end
    if numel(isSym) == 1 && A > 1
        isSym = repmat(isSym, 1, A);
    end

    rVec = double(dens.r(:).');
    if numel(rVec) == 1 && A > 1
        rVec = repmat(rVec, 1, A);
    end

    hasNested = isfield(dens, 'nested') && ~isempty(dens.nested);

    for a = 1:A
        isFlat = ~hasNested || isempty(dens.nested{a});
        if isFlat && ~isSym(a) && rVec(a) > 1
            tf = true;
            return;
        end
    end
end
