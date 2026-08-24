function tf = spectralIpForce(value)
%INTERNAL.SPECTRALIPFORCE  Bypass the spectral branch's cost gate.
%
%   TF = INTERNAL.SPECTRALIPFORCE() returns whether the cost gate in
%   MOBIUS.SPECTRALRELINNERMATRIX is bypassed.
%   INTERNAL.SPECTRALIPFORCE(VALUE) sets it and returns the new state.
%
%   The memory guard (MAX_POINTS) is never bypassed; only the cost
%   comparison is. Both routes compute the full-image measure, so this
%   changes cost, not value.
%
%   It exists because the cost gate is known to misroute outside the
%   shapes it was calibrated on. The gate declines the branch when
%   gridSize > COST_C * K^2 * nPairs, which models the translation-grid
%   route as costing K^2 per event pair -- omitting the node count N_u,
%   which scales with span/sigma. Where the data span many sigmas the
%   grid route is far dearer than the gate believes, and the branch is
%   declined where it would have won: at r = 3, K = 80, sigma = 10 with
%   positions over three octaves, taking the grid costs ~980 ms against
%   ~105 ms for the branch, and the gate flips to the branch only at
%   K >= 86. Benchmarking the decomposition rather than the routing
%   therefore pins this, as it pins the method and the relative-attribute
%   route.
%
%   Mirror of Python cosine._SPECTRAL_IP_FORCE.
    persistent forced
    if isempty(forced)
        forced = false;
    end
    if nargin > 0
        forced = logical(value);
    end
    tf = forced;
end
