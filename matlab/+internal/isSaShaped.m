function tf = isSaShaped(dens)
%ISSASHAPED  True when a density is the single-collection corner.
%
%   TF = INTERNAL.ISSASHAPED(DENS) is true for a MaetDensity at the
%   A = N = 1 case of the multi-attribute formulation --- one event
%   carrying one flat (non-nested) attribute whose multiset is the whole
%   collection. This is the single-collection ("single-attribute")
%   corner. Evaluation strategy for it is shape-gated, not type-gated:
%   there is one density type, and the corner is recognised by shape so
%   the specialized single-collection pipeline (probe dispatch,
%   validation-with-fallback) can serve it via internal.saView.
%
%   A legacy ExpTensDensity struct (should no longer be produced now that
%   the vector build returns a MaetDensity) is also accepted, so mixed
%   states during migration do not error.
%
%   Twin of python mpt._tensor.density.is_sa_shaped.
%
%   See also INTERNAL.SAVIEW, BUILDEXPTENS.

    tf = false;
    if ~isstruct(dens) || ~isfield(dens, 'tag')
        return;
    end
    if strcmp(dens.tag, 'ExpTensDensity')
        tf = true;
        return;
    end
    if ~strcmp(dens.tag, 'MaetDensity')
        return;
    end
    if double(dens.nAttrs) ~= 1 || double(dens.N) ~= 1
        return;
    end
    % Flat (non-nested) single attribute.
    if isfield(dens, 'nested') && iscell(dens.nested) ...
            && ~isempty(dens.nested) && ~isempty(dens.nested{1})
        return;
    end
    tf = true;
end
