function bytes = estimateMaJointWorkingSetBytes(rVec, kVec, isRel, isSym)
%ESTIMATEMAJOINTWORKINGSETBYTES  Multi-attribute joint-centres working set.
%
%   BYTES = INTERNAL.ESTIMATEMAJOINTWORKINGSETBYTES(RVEC, KVEC, ISREL,
%   ISSYM) estimates, in bytes, the working set of the multi-attribute
%   centres path, which materialises the *joint* tuple set: the product
%   across attributes of each attribute's enumerated tuple count ---
%   r_a! * C(K_a, r_a) on an unordered attribute, C(K_a, r_a) on an
%   ordered one (the perm side is the comb side; see the enumeration in
%   buildExpTens). The stored joint centres array is (D, nJoint) with
%   D = sum_a (r_a - isRel_a), plus per-attribute index bookkeeping of
%   the same nJoint length; a row factor of 2*D over-counts honestly
%   for a memory guard.
%
%   Used only to detect when a convention- or precision-forced centres
%   pick would be infeasible, so an over-count is the right bias. ISSYM
%   omitted treats every attribute as unordered, the conservative
%   (larger) count.
%
%   Twin of python _estimate_ma_joint_working_set_bytes.

    A = numel(rVec);
    if nargin < 4 || isempty(isSym)
        isSym = true(1, A);
    end
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
        % enumerated tuple count: r_a! * C(K_a, r_a) unordered,
        % C(K_a, r_a) ordered (perm side = comb side)
        cnt = 1;
        for k = (K_a - r_a + 1):K_a
            cnt = cnt * k;
        end
        if ~isSym(a)
            cnt = cnt / factorial(r_a);
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
