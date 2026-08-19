function guardForcedBulgerFeasible(kVec, rVec, Nx, Ny, reason, kVecY, symVec)
%GUARDFORCEDBULGERFEASIBLE  Raise if a forced Bulger inner product is infeasible.
%
%   INTERNAL.GUARDFORCEDBULGERFEASIBLE(KVEC, RVEC, NX, NY, REASON,
%   KVECY, SYMVEC) raises mpt:dispatch:singleImageInfeasible when the
%   single-image Bulger route is *forced* (the Möbius method is
%   unavailable --- r above the shipped orbit order) and its tuple-pair
%   kernel would be too large to materialise.
%
%   Bulger's MA inner product materialises each side's joint perm-side
%   working set nJ = N * prod_a nj_a, where nj_a is the attribute's
%   enumerated tuple count: K_a! / (K_a - r_a)! on an unordered
%   attribute (every ordering of every combination) and C(K_a, r_a) on
%   an ordered one (the perm side is the comb side; see the enumeration
%   in buildExpTens). The tuple-pair kernel is nJ_x * nJ_y float64
%   entries. When Bulger is forced there is no cheaper all-image
%   substitute to fall back to, so rather than let the product exhaust
%   memory and crash, this raises a clear error naming the shape.
%   Explicit method='bulger' overrides do not reach here (they are
%   honoured earlier), so this guards only auto-dispatch.
%
%   Each side is sized from its own density's per-attribute value
%   counts, since the two need not agree; KVECY omitted (or empty)
%   means they do. SYMVEC omitted (or empty) treats every attribute as
%   unordered, the conservative (larger) count.
%
%   Twin of python _guard_forced_bulger_feasible_ma /
%   SingleImageInfeasibleError (identifier
%   mpt:dispatch:singleImageInfeasible).

    A = numel(rVec);
    if nargin < 6 || isempty(kVecY)
        kVecY = kVec;
    end
    if nargin < 7 || isempty(symVec)
        symVec = true(1, A);
    end
    budget = internal.dispatchMemBudget();
    njx = localNjSide(Nx, kVec, rVec, symVec);
    njy = localNjSide(Ny, kVecY, rVec, symVec);
    pairBytes = njx * njy * 8;   % nJ_x * nJ_y float64 entries
    if pairBytes > budget
        error('mpt:dispatch:singleImageInfeasible', ...
            ['The inner product requires the single-image Bulger route ' ...
             '(%s, so the Möbius method is not available), but its ' ...
             'tuple-pair kernel would need ~%.1f GB (nJ_x = %.2e, ' ...
             'nJ_y = %.2e). Reduce the tuple order r or the collection ' ...
             'sizes.'], reason, pairBytes / 1024^3, njx, njy);
    end
end

function nJ = localNjSide(N, kSide, rVec, symVec)
%LOCALNJSIDE  One side's joint working set N * prod_a nj_a, saturating
%   past hopelessness. nj_a is K!/(K-r)! unordered, C(K, r) ordered.
    nJ = double(N);
    for a = 1:numel(rVec)
        K_a = double(kSide(a));
        r_a = double(rVec(a));
        if K_a < r_a
            nJ = 0;
            return;
        end
        fac = 1;
        for k = (K_a - r_a + 1):K_a
            fac = fac * k;
        end
        if ~symVec(a)
            fac = fac / factorial(r_a);
        end
        nJ = nJ * fac;
        if nJ > 1e18   % already hopeless; stop growing
            nJ = 1e18;
            return;
        end
    end
end
