function M = closedFormAttrMatrixFrom(cx, cy, wrapA, truncationSigmas)
%CLOSEDFORMATTRMATRIXFROM  (N_x, N_y) per-attribute inner matrix from centres.
%
%   M = MOBIUS.CLOSEDFORMATTRMATRIXFROM(CX, CY) uses the toolbox
%   default wrap and truncation width; CX, CY are bundles from
%   MOBIUS.CLOSEDFORMATTRCENTRES. WRAPA ('full-image' by default) is
%   the attribute's declared wrap and TRUNCATIONSIGMAS an optional
%   per-call accuracy width ([] takes the default). The Python twin
%   orders its two optional arguments the other way round
%   (truncation_sigmas then wrap_a); the MATLAB order is fixed by the
%   existing call sites, which pass the wrap alone.
%
%   Mirror of Python _mobius_inner._closed_form_attr_matrix_from, for
%   flat attributes and for the nested attributes
%   MOBIUS.CLOSEDFORMATTRCENTRES accepts. The attribute's expectation
%   tensor is a finite Gaussian mixture over its materialised
%   tuple-centres, so its inner matrix is the pairwise Gaussian overlap
%   of the two centre sets, aggregated to events by incidence sums. The
%   constant per-attribute Gaussian prefactor is dropped: it multiplies
%   the cross and both self matrices of the attribute identically, so
%   it cancels in every supported normalisation.
%
%   Absolute-periodic uses the full-image (torus) measure: the r-tuple
%   kernel is the product across coordinates of the 1-D wrapped Gaussian
%   theta(d) = sum_n exp(-(d + n P)^2 / (4 sigma^2)). Because Q factors
%   across coordinates in absolute mode, the product-of-theta form
%   (r * (2L+1) or r * M per pair) is the cheaper representation of
%   the all-image kernel; the image sum switches on only when the
%   accuracy floor requires it. When the user has opted this attribute
%   into wrapA = 'single-image' the code falls back to the nearest-
%   image kernel unchanged.
%
%   Relative-periodic uses the minimum-image pairwise-wrap quadratic
%   (exactly period-shift invariant) — measure (A), *not* the toolbox's
%   defined relative-periodic measure. The measure is declared by the
%   attribute's wrap: the v3+ default 'full-image' declares (C), the
%   all-image transposition average over the torus, which the
%   translation-grid routes compute. The two coincide for sigma << P and
%   diverge as sigma approaches it, so this route may serve a full-image
%   attribute only below INTERNAL.RELPERSIGMAOVERPTHRESHOLD, where the
%   difference sits inside the truncation floor; above that threshold
%   only wrap = 'single-image', which declares (A), reaches here. The
%   two predicates that enforce this are
%   MOBIUS.MARELATTRPREFERSCENTRES on the flat path and the nested
%   route rule inside INTERNAL.NESTEDCONTRACT.
%   Relative-non-periodic uses the exact relative quadratic.
%
%   A co-transposition unit at an inner or intermediate level carries a
%   block-diagonal metric that this flat quadratic form does not. No
%   caller can deliver one: INTERNAL.NESTEDCONTRACT declines an inner
%   unit before any centres bundle is built, and the flat MA
%   orchestrator sends only flat attributes, so CX.innerBlockSize is 0
%   at every call site in the toolbox. A bundle carrying one is refused
%   (mobius:closedFormAttrMatrixFrom:innerUnit) rather than computed
%   with a silently wrong flat metric.
%
%   The X-tuple axis is chunked so the pairwise overlap block never
%   exceeds a fixed element budget, mirroring the Python chunk rule.
%
%   The X side is restricted to one representative per tuple-symmetry
%   orbit and the sum scaled by the orbit size |G| — r_a! for a flat
%   symmetric attribute, the nested wreath-product order of
%   INTERNAL.NESTEDORBITMULT for a nested one. Bulger's restriction,
%   exact per event pair. See LOCALCOMBRESTRICTION inside
%   MOBIUS.CLOSEDFORMATTRCENTRES for the identity and for the cases
%   (ordered flat attribute, a trivial orbit, an unexpected tiling)
%   where it is declined; there cx.comb is empty and this is the
%   unrestricted (nJx, nJy) perm-vs-perm form as before.

    if nargin < 3 || isempty(wrapA)
        wrapA = 'full-image';
    end
    if nargin < 4
        truncationSigmas = [];
    end

    Cx = cx.Centres;   wx = cx.wJ(:);   Ex = cx.eventOfJ(:);
    Cy = cy.Centres;   wy = cy.wJ(:);   Ey = cy.eventOfJ(:);
    Nx = cx.N;  Ny = cy.N;

    % Bulger's restriction on the X side. The identity needs the *Y*
    % perm side to be stable under the same permutation group, i.e. the
    % two densities to carry the same tuple symmetry. Guaranteed by
    % every caller (the two densities share r_a and isSym), and checked
    % structurally: the Y side must be the same orbit tiling of its own
    % comb side. Twin of the Python guard.
    combX = localCombOf(cx);
    combY = localCombOf(cy);
    if ~isempty(combX)
        if isempty(combY) || combY.mult ~= combX.mult ...
                || size(Cy, 2) ~= combX.mult * size(combY.Centres, 2)
            combX = [];
        end
    end
    if ~isempty(combX)
        Cx = combX.Centres;
        wx = combX.wJ(:) * double(combX.mult);
        Ex = combX.eventOfJ(:);
    end
    sigma = cx.sigma;  r_a = cx.r;
    isRel = cx.isRel;  isPer = cx.isPer;  period = cx.period;
    % Co-transposition block size (0 = flat metric). Bundles assembled
    % before the field existed read as 0, which is the flat path they
    % were built for.
    if isfield(cx, 'innerBlockSize') && ~isempty(cx.innerBlockSize)
        bs = double(cx.innerBlockSize);
    else
        bs = 0;
    end
    if bs >= 2
        error('mobius:closedFormAttrMatrixFrom:innerUnit', ...
              ['the tuple-centres closed form does not carry an inner ' ...
               '[rel] unit; the nested plan should have declined this ' ...
               'attribute']);
    end

    d = size(Cx, 1);
    njx = size(Cx, 2);
    njy = size(Cy, 2);

    GY = sparse(1:njy, Ey, 1, njy, Ny);      % (njy, Ny) incidence
    inv4s2 = 1 / (4 * sigma^2);

    if isempty(truncationSigmas)
        truncationSigmas = mptDefaults('truncationSigmas');
    end
    % Single-image short-circuit. Under truncation the image budget can
    % admit no image beyond the nearest one (L = 0, which at the
    % 6-sigma default holds for sigma/P <= 0.059 in this convention).
    % theta(d) is then exactly the nearest-image Gaussian
    % exp(-d^2 / (4 sigma^2)), so the per-coordinate product below
    % computes the same number as the joint Q-form path -- but pays one
    % exp per coordinate of the (d, nc, njy) difference array instead
    % of one on the summed (nc, njy) form. Fall through to
    % localComputeQFlat there (which applies the nearest-image
    % reduction itself, as the wrap = 'single-image' opt-in does); the
    % two agree to ~3e-16. wrappedGaussian1d cannot prefer Fourier at
    % L = 0 (M >= 1 fails the M < 2L + 1 test), so the gate is exact.
    absPerFullImage = isPer && ~isRel && strcmp(wrapA, 'full-image') ...
        && internal.wrappedKernelImageCount(sigma, period, ...
                                            truncationSigmas, 4) > 0;

    chunk = max(1, min(njx, floor(16e6 / max(njy * max(d, 1), 1))));
    M = zeros(Nx, Ny);
    for s = 1:chunk:njx
        e = min(s + chunk - 1, njx);
        idx = s:e;
        nc = numel(idx);
        D = reshape(Cx(:, idx), [d, nc, 1]) - reshape(Cy, [d, 1, njy]);
        if absPerFullImage
            % Per-coordinate theta then product across coordinates. Overlap-kernel
            % convention (exponent_denominator = 4). Shape of theta:
            % (d, nc, njy); product over d = axis 1.
            theta = internal.wrappedGaussian1d(D, sigma, period, ...
                                                truncationSigmas, 4);
            kernelBlock = reshape(prod(theta, 1), [nc, njy]);
        else
            Q = localComputeQFlat(D, r_a, isRel, isPer, period);
            kernelBlock = reshape(exp(-Q * inv4s2), [nc, njy]);
        end
        ov = (wx(idx) * wy.') .* kernelBlock;
        GXc = sparse(Ex(idx), 1:nc, 1, Nx, nc);   % (Nx, nc) incidence
        M = M + GXc * (ov * GY);
    end
end


function comb = localCombOf(c)
%LOCALCOMBOF  The bundle's comb-side restriction, or [] when absent.
%
%   Tolerates a bundle built before the field existed (or one a caller
%   assembled by hand), which simply leaves the route unrestricted.

    if isfield(c, 'comb')
        comb = c.comb;
    else
        comb = [];
    end
end


function Q = localComputeQFlat(D, r, isRel, isPer, period)
%LOCALCOMPUTEQFLAT  Quadratic form from centre differences (flat attrs).
%
%   Mirror of Python dispatch._compute_Q with reduced=isRel: relative
%   centres arrive in the first-coordinate reduced convention (r - 1 rows),
%   absolute centres as full r-tuples.
%
%   - abs: Q = sum(D.^2, 1), with component-wise principal wrap first
%     when periodic.
%   - rel non-periodic: Q = sum(D.^2, 1) - sum(D, 1).^2 / r (the same
%     algebraic identity serves the full and reduced conventions).
%   - rel periodic (pairwise wrap, Eq 6 form): the implicit coordinate 0
%     contributes pairs (0, k+1) -> wrap(D(k))^2, and reduced rows
%     contribute pairs (i+1, j+1) -> wrap(D(i) - D(j))^2; the sum is
%     divided by r. Pairwise wrapping (not a component-wise outer
%     wrap) is what preserves exact transposition invariance on the
%     circle.

    if isRel
        if isPer
            position0Wrapped = D - period * floor(D / period + 0.5);
            Q = sum(position0Wrapped.^2, 1);
            dRows = size(D, 1);
            for i = 1:dRows - 1
                for j = i + 1:dRows
                    delta = D(i, :, :) - D(j, :, :);
                    delta = delta - period * floor(delta / period + 0.5);
                    Q = Q + delta.^2;
                end
            end
            Q = Q / r;
        else
            Q = sum(D.^2, 1) - sum(D, 1).^2 / r;
        end
    else
        if isPer
            D = D - period * floor(D / period + 0.5);
        end
        Q = sum(D.^2, 1);
    end
end
