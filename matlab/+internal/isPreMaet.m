function tf = isPreMaet(x)
%ISPREMAET  True for a pre-MAET struct.
%   A pre-MAET is a scalar struct carrying all three of pAttr, wAttr, and
%   specs -- what preMaet builds and what every pre-MAET operator returns.
%   Requiring all three keeps a density struct, which also has a pAttr
%   field, from being read as one. preMaet performs the real validation.
    tf = isstruct(x) && isscalar(x) && isfield(x, 'pAttr') ...
         && isfield(x, 'wAttr') && isfield(x, 'specs');
end
