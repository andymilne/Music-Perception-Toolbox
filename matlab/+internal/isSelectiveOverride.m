function tf = isSelectiveOverride(v, A)
%ISSELECTIVEOVERRIDE  True for a 1 x A cell with at least one empty entry.
%   The selective form of an override: the named attributes are set and
%   the rest keep what the spec carries. A cell with no empty entry is
%   the full form and overrides every attribute; a numeric array is never
%   selective, having no way to say "leave this one alone".
    tf = iscell(v) && numel(v) == A && any(cellfun(@isempty, v));
end
