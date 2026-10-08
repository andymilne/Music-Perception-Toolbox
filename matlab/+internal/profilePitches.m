function [vp_p, order] = profilePitches(nXcorr, nTmpl, step, pOffset, per, period)
%PROFILEPITCHES  Candidate fundamentals of a cross-correlation profile.
%
%   [vp_p, order] = internal.profilePitches(nXcorr, nTmpl, step, ...
%       pOffset, per, period)
%
%   Non-periodic: lag k (0-based) of the full correlation places the
%   template's fundamental at (k - (nTmpl - 1)) * step above the
%   chord's lowest pitch, pOffset; order is [] (the profile is already
%   in ascending order). Periodic: circular lag l places it at
%   l * step + pOffset; the values are taken modulo the period and
%   sorted, order being the permutation to apply to the profile.
%   vp_p is a column vector.
%
%   See also VIRTUALPITCHES, INTERNAL.TEMPLATEXCORRCHORDSIDE.

    if per
        vp = mod((0:nXcorr-1)' * step + pOffset, period);
        [vp_p, order] = sort(vp);
    else
        vp_p = ((0:nXcorr-1)' - (nTmpl - 1)) * step + pOffset;
        order = [];
    end
end
