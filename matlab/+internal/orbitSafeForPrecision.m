function tf = orbitSafeForPrecision(rVec, kVec)
%ORBITSAFEFORPRECISION  True if every r_a >= 2 attribute has K_a >= r_a + margin.
%   Mirror of Python dispatch._orbit_safe_for_precision. Refuses the
%   Möbius method when its alternating-sum cancellation could swamp the
%   answer (K_a too close to r_a). Used by the MA dispatcher and the
%   per-level nested orbit choice.
%
%   The margin applies only to attributes with r_a >= 2: the Möbius
%   alternating sum over set partitions is trivial at r_a = 1 (a single
%   partition, no signs), so an r_a = 1 attribute carries no
%   cancellation risk regardless of its K_a. Scalar attributes
%   (K_a = 1, r_a = 1) are the canonical multi-attribute pattern -- an
%   onset or duration alongside pitch content -- and must not veto the
%   Möbius method for the density.
    K_MINUS_R_MIN = 2;          % match Python _ORBIT_K_MINUS_R_MIN
    r = rVec(:);
    k = kVec(:);
    mask = r >= 2;
    tf = all((k(mask) - r(mask)) >= K_MINUS_R_MIN);
end
