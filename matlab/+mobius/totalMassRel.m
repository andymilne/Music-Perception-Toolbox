function Z = totalMassRel(p, w, sigma, r)
%MOBIUS.TOTALMASSREL  Total mass of the relative-mode tensor.
%
%   Z = MOBIUS.TOTALMASSREL(P, W, SIGMA, R) returns ∫T_rel(Δ)dΔ for the
%   relative, non-periodic tensor, via the relation
%
%       Z_rel = Z_abs / (sigma * sqrt(2*pi/r)).
%
%   The first argument is unused (kept for signature symmetry with
%   orbit-IP routines).
%
%   Inputs:
%     P      Ignored.
%     W      Weights vector.
%     SIGMA  Scalar bandwidth.
%     R      Integer tensor order, R >= 1.
%
%   See also MOBIUS.TOTALMASSABS.

    arguments
        p
        w (:,1) double
        sigma (1,1) double {mustBePositive}
        r (1,1) {mustBeInteger, mustBePositive}
    end

    Zabs = mobius.totalMassAbs(p, w, sigma, r);
    Z = Zabs / (sigma * sqrt(2 * pi / r));
end
