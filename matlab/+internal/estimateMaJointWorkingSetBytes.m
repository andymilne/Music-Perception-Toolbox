function bytes = estimateMaJointWorkingSetBytes(rVec, kVec, isRel)
%ESTIMATEMAJOINTWORKINGSETBYTES  Multi-attribute joint-centres working set.
%
%   BYTES = INTERNAL.ESTIMATEMAJOINTWORKINGSETBYTES(RVEC, KVEC, ISREL)
%   estimates, in bytes, the working set of the multi-attribute centres
%   path, which materialises the *joint* tuple set: the product across
%   attributes of each attribute's ordered-tuple count r_a! * C(K_a, r_a).
%   The stored joint centres array is (D, nJoint) with D = sum_a (r_a -
%   isRel_a), plus per-attribute index bookkeeping of the same nJoint
%   length; a row factor of 2*D over-counts honestly for a memory guard.
%
%   Used only to detect when a convention- or precision-forced centres
%   pick would be infeasible, so an over-count is the right bias.
%
%   Twin of python _estimate_ma_joint_working_set_bytes.

    A = numel(rVec);
    nJoint = 1;
    D = 0;
    CAP = 2^60;
    for a = 1:A
        r_a = double(rVec(a));
        K_a = double(kVec(a));
        if K_a < r_a
            bytes = 0;
            return;
        end
        % ordered-tuple count r_a! * C(K_a, r_a) = K_a! / (K_a - r_a)!
        cnt = 1;
        for k = (K_a - r_a + 1):K_a
            cnt = cnt * k;
        end
        nJoint = nJoint * max(cnt, 1);
        if isRel(a)
            D = D + r_a - 1;
        else
            D = D + r_a;
        end
        % Cap to avoid unbounded growth; anything past budget is infeasible.
        if nJoint * max(D, 1) * 8 > CAP
            bytes = CAP;
            return;
        end
    end
    bytes = nJoint * max(D, 1) * 2 * 8;
end
