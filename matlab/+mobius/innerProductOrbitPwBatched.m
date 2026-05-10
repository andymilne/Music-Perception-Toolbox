function [vals, ratios] = innerProductOrbitPwBatched(K_g, w_A_g, w_B_g, r, opts)
%MOBIUS.INNERPRODUCTORBITPWBATCHED  Per-grid-point weights orbit IP.
%
%   VALS = MOBIUS.INNERPRODUCTORBITPWBATCHED(K_G, W_A_G, W_B_G, R)
%   evaluates the orbit-Möbius inner product at each batch index,
%   allowing the source weights to differ per batch. This is the
%   v2.2 multi-attribute path: each grid point represents one
%   (event_X, event_Y) pair and per-attribute weights vary per event.
%
%   The standard MOBIUS.INNERPRODUCTORBITGRID requires shared W_A and
%   W_B across the grid axis; this variant lifts that restriction.
%
%   VALS = MOBIUS.INNERPRODUCTORBITPWBATCHED(..., 'prefactor', PF)
%   multiplies the raw orbit sum by PF before returning. Default: 1.
%
%   [VALS, RATIOS] = MOBIUS.INNERPRODUCTORBITPWBATCHED(..., 'returnCancellationRatio', true)
%   additionally returns per-batch cancellation ratios.
%
%   Inputs:
%     K_G    (N, n_A, n_B) double — stack of kernels per batch.
%     W_A_G  (N, n_A)       double — per-batch A-side weights.
%     W_B_G  (N, n_B)       double — per-batch B-side weights.
%     R      integer        — tensor order, 2 <= R <= 12.
%
%   See also MOBIUS.INNERPRODUCTORBITGRID.

    arguments
        K_g (:,:,:) double
        w_A_g (:,:) double
        w_B_g (:,:) double
        r (1,1) {mustBeInteger}
        opts.prefactor (1,1) double = 1.0
        opts.returnCancellationRatio (1,1) logical = false
    end

    N = size(K_g, 1);
    if size(w_A_g, 1) ~= N || size(w_B_g, 1) ~= N
        error('mobius:innerProductOrbitPwBatched:batchMismatch', ...
            'Batch dimensions of K_g, w_A_g, w_B_g must match.');
    end

    table = mobius.getOrbitTable(r);
    total = zeros(N, 1);
    maxAbsTerm = zeros(N, 1);

    % U_LABEL = 1000 is the consumer-side convention; baked into
    % orb.recipeBatched at table-build time (see buildOrbitRecipes in
    % buildOrbitTable.m).

    for k = 1:numel(table)
        orb = table(k);
        nE = size(orb.edges, 1);
        operands = cell(1, orb.qA + orb.qB + nE);
        idx = 1;

        % Per-A-block weight vectors carry the batch axis (shared u).
        for alpha = 1:orb.qA
            operands{idx} = w_A_g .^ orb.m_A(alpha);
            idx = idx + 1;
        end
        for beta = 1:orb.qB
            operands{idx} = w_B_g .^ orb.m_B(beta);
            idx = idx + 1;
        end
        for e = 1:nE
            m = orb.edges(e, 3);
            if m == 1
                operands{idx} = K_g;
            else
                operands{idx} = K_g .^ m;
            end
            idx = idx + 1;
        end

        contribution = mobius.executeRecipe(operands, orb.recipeBatched);
        contribution = contribution(:);
        term = orb.weight * orb.mu * contribution;
        total = total + term;
        maxAbsTerm = max(maxAbsTerm, abs(term));
    end

    vals = opts.prefactor * total;
    if opts.returnCancellationRatio
        ratios = ones(N, 1);
        nz = maxAbsTerm > 0;
        ratios(nz) = abs(total(nz)) ./ maxAbsTerm(nz);
    else
        ratios = [];
    end
end
