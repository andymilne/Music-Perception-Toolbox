function [sg, rr, rl, pr, pd] = subGeom(sigma, r, isRel, isPer, period, keep)
%SUBGEOM  Reduce the five per-attribute geometry sequences to the kept axes.
    sg = subOne(sigma, keep); rr = subOne(r, keep); rl = subOne(isRel, keep);
    pr = subOne(isPer, keep); pd = subOne(period, keep);
end

function s = subOne(seq, keep)
    if (isnumeric(seq) || islogical(seq) || iscell(seq)) && numel(seq) >= max(keep)
        s = seq(keep);
    else
        s = seq;
    end
end
