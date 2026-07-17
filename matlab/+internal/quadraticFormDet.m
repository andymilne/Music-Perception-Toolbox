function detM = quadraticFormDet(r, innerR, isRel)
%INTERNAL.QUADRATICFORMDET  Determinant of the relative-mode quadratic form M for one attribute.
%
%   A single flat attribute of tuple order r in relative mode carries
%   the metric M = I - e e' / r (the all-ones removed once), whose
%   determinant is 1 / r. A nested attribute whose active
%   co-transposition unit has block size s_u removes an all-ones within
%   each of its r / s_u block-diagonal blocks, giving
%   (1 / s_u) ^ (r / s_u). Absolute mode (and the vacuous r = 1
%   relative case) has M = I, determinant 1.
%
%   Parameters mirror the three-way branch every normalisation site
%   used to inline: pass the attribute's tuple order R, its block
%   size INNERR (0 when flat), and its ISREL flag.
%
%   Twin of Python mpt._tensor.dispatch._quadratic_form_det: same
%   three-way branch, same return value, so the two languages route
%   every consumer through a single algebraic identity.

    r = double(r);
    innerR = double(innerR);
    if innerR >= 2
        detM = (1 / innerR) ^ (r / innerR);
    elseif logical(isRel) && r >= 2
        detM = 1 / r;
    else
        detM = 1;
    end
end
