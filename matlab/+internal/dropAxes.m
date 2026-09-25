function [p2, w2, sp2] = dropAxes(p, w, sp, drop, A)
%DROPAXES  Remove a set of attributes from the positions, weights, and specs
%   (marginalization by removal).
    keep = setdiff(1:A, drop);
    p2 = p(keep);
    if iscell(w) && numel(w) == A, w2 = w(keep); else, w2 = w; end
    if isempty(sp), sp2 = sp; else, sp2 = sp(keep); end
end
