function symC = subSymArgs(isSym, keep)
%SUBSYMARGS  Reduce a per-attribute isSym vector to the kept axes.
%
%   Returns {} when ISSYM is empty (the buildExpTens/cosSimExpTens
%   default, symmetric), else a one-element cell holding the subset,
%   ready to splat as the trailing positional isSym of the raw forms.

    if isempty(isSym)
        symC = {};
        return;
    end
    if numel(isSym) >= max(keep)
        symC = {isSym(keep)};
    else
        symC = {isSym};
    end
end
