function M = closedFormAttrMatrixFrom(cx, cy, wrapA)
%CLOSEDFORMATTRMATRIXFROM  (N_x, N_y) per-attribute inner matrix from centres.
%
%   Mirror of Python cosine._closed_form_attr_matrix_from (flat
%   attributes). The attribute's expectation tensor is a finite
%   Gaussian mixture over its materialised tuple-centres, so its inner
%   matrix is the pairwise Gaussian overlap of the two centre sets,
%   aggregated to events by incidence sums. The constant per-attribute
%   Gaussian prefactor is dropped: it multiplies the cross and both
%   self matrices of the attribute identically, so it cancels in every
%   supported normalisation.
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
%   (exactly period-shift invariant) — the toolbox's defined
%   relative-periodic measure, which coincides with the all-image
%   translation-grid reading below the sigma/P threshold
%   (MOBIUS.MARELATTRPREFERSCENTRES enforces that condition).
%   Relative-non-periodic uses the exact relative quadratic.
%
%   The X-tuple axis is chunked so the pairwise overlap block never
%   exceeds a fixed element budget, mirroring the Python chunk rule.

    if nargin < 3 || isempty(wrapA)
        wrapA = 'full-image';
    end

    Cx = cx.Centres;   wx = cx.wJ(:);   Ex = cx.eventOfJ(:);
    Cy = cy.Centres;   wy = cy.wJ(:);   Ey = cy.eventOfJ(:);
    Nx = cx.N;  Ny = cy.N;
    sigma = cx.sigma;  r_a = cx.r;
    isRel = cx.isRel;  isPer = cx.isPer;  period = cx.period;

    d = size(Cx, 1);
    njx = size(Cx, 2);
    njy = size(Cy, 2);

    GY = sparse(1:njy, Ey, 1, njy, Ny);      % (njy, Ny) incidence
    inv4s2 = 1 / (4 * sigma^2);

    truncationSigmas = mptDefaults('truncationSigmas');
    absPerFullImage = isPer && ~isRel && strcmp(wrapA, 'full-image');

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
            slot0 = D - period * floor(D / period + 0.5);
            Q = sum(slot0.^2, 1);
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
