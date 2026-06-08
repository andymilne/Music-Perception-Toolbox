function tf = orbitSafeForPrecision(rVec, kVec)
%ORBITSAFEFORPRECISION  True if every attribute satisfies K_a >= r_a + margin.
%   Mirror of Python dispatch._orbit_safe_for_precision. Refuses the
%   Möbius method when its alternating-sum cancellation could swamp the
%   answer (K_a too close to r_a). Used by the MA dispatcher and the
%   per-level nested orbit choice.
    K_MINUS_R_MIN = 2;          % match Python _ORBIT_K_MINUS_R_MIN
    r = rVec(:);
    k = kVec(:);
    tf = all((k - r) >= K_MINUS_R_MIN);
end
