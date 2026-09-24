function [pAttr, wAttr, specs] = unpackPreMaet(pm)
%UNPACKPREMAET Split a pre-MAET into its three parts.
%
%   [PATTR, WATTR, SPECS] = UNPACKPREMAET(PM) returns the parts of PM in
%   the order the loose-triple signatures take them. WATTR
%   and SPECS are [] where unset.
%
%   Input
%       pm - Pre-MAET, as built by packPreMaet or returned by any pre-MAET
%            operator.
%
%   Outputs
%       pAttr - 1 x A cell of per-attribute value matrices.
%       wAttr - [] , a scalar, or a 1 x A cell of per-attribute weights.
%       specs - [] or a 1 x A cell of per-attribute spec structs.
%
%   unpackPreMaet and packPreMaet are inverses: packPreMaet undoes this
%   split, holding the three parts in one pre-MAET again.
%
%   See also PACKPREMAET, SHOWPREMAET.

if ~internal.isPreMaet(pm)
    error('unpackPreMaet:notPreMaet', ...
          ['Expected a pre-MAET: a struct with the fields pAttr, ' ...
           'wAttr, and specs.']);
end
pAttr = pm.pAttr;
wAttr = pm.wAttr;
specs = pm.specs;
end
