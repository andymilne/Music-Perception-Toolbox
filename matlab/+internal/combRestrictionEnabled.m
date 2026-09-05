function tf = combRestrictionEnabled(value)
%INTERNAL.COMBRESTRICTIONENABLED  Master switch for Bulger's X-side
%   restriction in the centres route.
%
%   TF = INTERNAL.COMBRESTRICTIONENABLED() returns whether
%   MOBIUS.CLOSEDFORMATTRCENTRES attaches the comb-side bundle that
%   MOBIUS.CLOSEDFORMATTRMATRIXFROM uses to restrict the X side of the
%   centre-overlap array to one representative per permutation orbit.
%   INTERNAL.COMBRESTRICTIONENABLED(VALUE) sets it and returns the new
%   state.
%
%   Exists so tests can force the unrestricted perm-vs-perm form and
%   compare the two, exactly as the Python tests toggle
%   _mobius_inner._COMB_RESTRICTION_ENABLED. The restriction is an exact
%   identity, so the switch changes cost, not value.
%
%   Mirror of Python _mobius_inner._COMB_RESTRICTION_ENABLED.
    persistent enabled
    if isempty(enabled)
        enabled = true;
    end
    if nargin > 0
        enabled = logical(value);
    end
    tf = enabled;
end
