function Z = totalMassAbs(p_unused, w, sigma, r) %#ok<INUSL>
%MOBIUS.TOTALMASSABS  Total mass of the absolute-mode tensor via Möbius.
%
%   Z = MOBIUS.TOTALMASSABS(P, W, SIGMA, R) returns ∫T_abs(x)dx for the
%   absolute, non-periodic tensor with weights W and isotropic sigma.
%   The first argument is unused (positions do not enter the integral —
%   each tuple's r-dimensional Gaussian integrates to (sigma * sqrt(2*pi))^r
%   independent of position) and is included for signature symmetry with
%   the orbit-IP routines.
%
%   The total mass is
%
%     Z_abs = (sigma * sqrt(2*pi))^r * sum_m N(m) * mu(m) * prod_l (sum_i w_i^{m_l})
%
%   where the sum is over integer partitions m of r, N(m) is the number
%   of set partitions of [r] with block-size profile m (= r! / (prod_l
%   m_l! * aut(m))), and mu(m) is the Möbius coefficient.
%
%   Inputs:
%     P      Ignored (kept for signature symmetry).
%     W      Weights vector.
%     SIGMA  Scalar bandwidth.
%     R      Integer tensor order, R >= 1.
%
%   See also MOBIUS.TOTALMASSREL.

    arguments
        p_unused
        w (:,1) double
        sigma (1,1) double {mustBePositive}
        r (1,1) {mustBeInteger, mustBePositive}
    end

    sigmaFactor = (sigma * sqrt(2 * pi)) ^ r;
    moebiusSum = 0.0;
    partitions = mobius.integerPartitions(r);
    rFact = factorial(r);
    for ip = 1:numel(partitions)
        m = partitions{ip};
        % Number of set partitions of [r] with block-size profile m.
        nWithProfile = rFact;
        for j = 1:numel(m)
            nWithProfile = nWithProfile / factorial(m(j));
        end
        nWithProfile = nWithProfile / mobius.autSize(m);

        muM = mobius.mobiusForBlocksizes(m);

        perPartition = 1.0;
        for j = 1:numel(m)
            perPartition = perPartition * sum(w .^ m(j));
        end

        moebiusSum = moebiusSum + nWithProfile * muM * perPartition;
    end

    Z = sigmaFactor * moebiusSum;
end
