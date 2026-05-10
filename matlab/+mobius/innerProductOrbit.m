function [val, ratio] = innerProductOrbit(K, w_A, w_B, r, opts)
%MOBIUS.INNERPRODUCTORBIT  Distinct-index inner product via Möbius/orbit.
%
%   VAL = MOBIUS.INNERPRODUCTORBIT(K, W_A, W_B, R) evaluates
%   <T_A, T_B> as an alternating sum over partition orbits, exact and
%   independent of n.
%
%   VAL = MOBIUS.INNERPRODUCTORBIT(..., 'prefactor', PF) multiplies the
%   raw orbit sum by PF before returning. For the single-attribute case,
%   PF is conventionally (sigma * sqrt(pi))^R; pass 1 for the bare orbit
%   quantity. Default: 1.
%
%   [VAL, RATIO] = MOBIUS.INNERPRODUCTORBIT(..., 'returnCancellationRatio', true)
%   additionally returns the alternating-sum cancellation ratio
%   |sum| / max_orb(|term_orb|). A value near 1 indicates no
%   cancellation; a value much smaller than 1 indicates digits of
%   precision lost. ~1e-10 corresponds to roughly 6 surviving decimal
%   digits; below that, callers should fall back to a non-cancelling
%   method.
%
%   Inputs:
%     K    (n_A, n_B) double — pairwise kernel.
%     W_A  (n_A, 1)   double — A-side weights.
%     W_B  (n_B, 1)   double — B-side weights.
%     R    integer    — tensor order, 2 <= R <= 12.
%
%   Per-call cost: O(|Omega_r| * contraction_cost) with the contraction
%   typically O(R * n^2) per orbit, dominated by the few highest-rank
%   orbits.
%
%   See also MOBIUS.INNERPRODUCTORBITGRID, MOBIUS.GETORBITTABLE.

    arguments
        K (:,:) double
        w_A (:,1) double
        w_B (:,1) double
        r (1,1) {mustBeInteger}
        opts.prefactor (1,1) double = 1.0
        opts.returnCancellationRatio (1,1) logical = false
    end

    table = mobius.getOrbitTable(r);
    total = 0.0;
    maxAbsTerm = 0.0;
    for k = 1:numel(table)
        orb = table(k);
        nE = size(orb.edges, 1);
        operands = cell(1, orb.qA + orb.qB + nE);
        idx = 1;

        % Per-A-block weight vectors (axis label = alpha, 1..qA).
        for alpha = 1:orb.qA
            operands{idx} = w_A .^ orb.m_A(alpha);
            idx = idx + 1;
        end
        % Per-B-block weight vectors (axis label = qA + beta, beta in 1..qB).
        for beta = 1:orb.qB
            operands{idx} = w_B .^ orb.m_B(beta);
            idx = idx + 1;
        end
        % Per-edge kernel powers (axes = [alpha, qA + beta]).
        for e = 1:nE
            m = orb.edges(e, 3);
            if m == 1
                operands{idx} = K;
            else
                operands{idx} = K .^ m;
            end
            idx = idx + 1;
        end

        contribution = mobius.executeRecipe(operands, orb.recipeIP);
        term = orb.weight * orb.mu * contribution;
        total = total + term;
        absTerm = abs(term);
        if absTerm > maxAbsTerm
            maxAbsTerm = absTerm;
        end
    end

    val = opts.prefactor * total;
    if opts.returnCancellationRatio
        if maxAbsTerm > 0
            ratio = abs(total) / maxAbsTerm;
        else
            ratio = 1.0;
        end
    else
        ratio = [];
    end
end
