function guardForcedBulgerFeasible(K_x, K_y, r, reason)
%GUARDFORCEDBULGERFEASIBLE  Raise if a forced Bulger inner product is infeasible.
%
%   INTERNAL.GUARDFORCEDBULGERFEASIBLE(K_X, K_Y, R, REASON) raises
%   mpt:dispatch:singleImageInfeasible when the single-image Bulger
%   route is *forced* (the Möbius method is unavailable --- r above the
%   shipped orbit order) and its
%   tuple-pair kernel would be too large to materialise.
%
%   Bulger materialises a kernel whose size is the product of the two
%   sides' ordered-tuple counts, n_j_x * n_j_y with n_j = K! / (K - r)!.
%   When Bulger is forced there is no cheaper all-image substitute to
%   fall back to, so rather than let the product exhaust memory and
%   crash, this raises a clear error naming the shape. Explicit
%   method='bulger' overrides do not reach here (they are honoured
%   earlier), so this guards only auto-dispatch.
%
%   Twin of python _guard_forced_bulger_feasible /
%   SingleImageInfeasibleError (identifier
%   mpt:dispatch:singleImageInfeasible).

    budget = internal.dispatchMemBudget();
    njx = localNj(K_x, r);
    njy = localNj(K_y, r);
    pairBytes = njx * njy * 8;   % n_j_x * n_j_y float64 entries
    if pairBytes > budget
        error('mpt:dispatch:singleImageInfeasible', ...
            ['The inner product requires the single-image Bulger route ' ...
             '(%s, so the Möbius method is not available), but its ' ...
             'tuple-pair kernel would need ~%.1f GB (n_j_x = %.2e, ' ...
             'n_j_y = %.2e). Reduce the tuple order r or the collection ' ...
             'sizes.'], reason, pairBytes / 1024^3, njx, njy);
    end
end

function n = localNj(K, r)
%LOCALNJ  Ordered-tuple count K!/(K-r)!, saturating past hopelessness.
    if K < r
        n = 0;
        return;
    end
    n = 1;
    for k = (K - r + 1):K
        n = n * k;
        if n * n > 2^62   % already hopeless; stop growing
            n = 2^31;
            return;
        end
    end
end
