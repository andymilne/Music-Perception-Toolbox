function tf = isSingleMultiset(dens)
%INTERNAL.ISSINGLEMULTISET  True when a density is the single-multiset corner (A = N = 1).
%
%   A single flat (non-nested) attribute whose values are one weighted
%   multiset (the ET of Milne 2011), for which the single-multiset fast
%   kernels apply. The A == 1, r == 1 case with N > 1 never reaches here
%   as such: localBuildMA collapses it into one pooled event at build (a
%   tuple is a lone value at r = 1), so every downstream consumer only
%   ever meets the canonical N == 1 form. The evaluation strategy for
%   this corner is shape-gated, not type-gated.
%
%   Twin of Python mpt._tensor.density.is_single_multiset.

    tf = false;
    if ~(isstruct(dens) && isfield(dens, 'tag') ...
            && strcmp(dens.tag, 'MaetDensity'))
        return;
    end
    if ~(dens.nAttrs == 1 && dens.N == 1)
        return;
    end
    % A nested first attribute is not the flat single-multiset corner.
    if isfield(dens, 'nested') && ~isempty(dens.nested) ...
            && ~isempty(dens.nested{1})
        return;
    end
    tf = true;
end
