function bytes = estimateMaJointWorkingSetBytes(rVec, kVec, isRel, isExch, ...
                                                nEvents)
%ESTIMATEMAJOINTWORKINGSETBYTES  Multi-attribute joint-centres working set.
%
%   BYTES = INTERNAL.ESTIMATEMAJOINTWORKINGSETBYTES(RVEC, KVEC, ISREL,
%   ISEXCH, NEVENTS) estimates, in bytes, the working set of the multi-attribute
%   centres path, which materialises the *joint* tuple set: the product
%   across attributes of each attribute's enumerated tuple count ---
%   r_a! * C(K_a, r_a) on an unordered attribute, C(K_a, r_a) on an
%   ordered one (the perm side is the comb side; see the enumeration in
%   buildMaet). The stored joint centres array is (D, nJoint) with
%   D = sum_a (r_a - isRel_a), plus per-attribute index bookkeeping of
%   the same nJoint length; a row factor of 2*D over-counts honestly
%   for a memory guard.
%
%   Used only to detect when a convention- or precision-forced centres
%   pick would be infeasible, so an over-count is the right bias. ISEXCH
%   omitted treats every attribute as unordered, the conservative
%   (larger) count. NEVENTS (default 1) is the number of events whose
%   joint tuple sets are held at once: N on the joint-centres path,
%   which materialises every event's joint tuple set together (taken
%   where an attribute is at r <= 1 or carries a kernel covariance), and
%   1 on the factored routes, which take a density event by event.
%
%   Twin of python _estimate_ma_joint_working_set_bytes.

    A = numel(rVec);
    if nargin < 4 || isempty(isExch)
        isExch = true(1, A);
    end
    if nargin < 5 || isempty(nEvents)
        nEvents = 1;
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
        if ~isExch(a)
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
    bytes = min(nJoint * max(D, 1) * 2 * 8 * max(double(nEvents), 1), CAP);
end
