function tf = isSingleMultiset(dens)
%INTERNAL.ISSINGLEMULTISET  True when a density is the single-multiset corner.
%
%   A MaetDensity at the A = N = 1 case of the multi-attribute
%   formulation --- one event carrying one flat (non-nested) attribute
%   whose values are a single weighted multiset (the expectation tensor
%   of Milne 2011). The evaluation strategy for this corner is
%   shape-gated, not type-gated: there is one density type, and the
%   single-multiset fast kernels are an optimisation dispatched on this
%   shape.
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
