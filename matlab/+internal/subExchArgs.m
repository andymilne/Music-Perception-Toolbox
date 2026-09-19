function exchC = subExchArgs(isExch, keep)
%SUBEXCHARGS  Reduce a per-attribute isExch vector to the kept axes.
%
%   Returns {} when ISSYM is empty (the buildMaet/simMaet
%   default, symmetric), else a one-element cell holding the subset,
%   ready to splat as the trailing positional isExch of the raw forms.

    if isempty(isExch)
        exchC = {};
        return;
    end
    if numel(isExch) >= max(keep)
        exchC = {isExch(keep)};
    else
        exchC = {isExch};
    end
end
