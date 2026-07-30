function [vals, ratios, termMass, termMassSum] = innerProductOrbitGrid(K_u, w_A, w_B, r, opts)
%MOBIUS.INNERPRODUCTORBITGRID  Distinct-index IP on a grid of u-shifts.
%
%   VALS = MOBIUS.INNERPRODUCTORBITGRID(K_U, W_A, W_B, R) evaluates the
%   absolute-mode inner product at each u-grid point in K_U, returning a
%   length-N_u row vector. Used by the relative-mode integration
%   approach: the resulting profile is trapezoidally integrated and
%   divided by an appropriate prefactor to recover <T_A, T_B>_rel.
%
%   VALS = MOBIUS.INNERPRODUCTORBITGRID(..., 'prefactor', PF) multiplies
%   the raw orbit sum by PF before returning. Default: 1.
%
%   [VALS, RATIOS] = MOBIUS.INNERPRODUCTORBITGRID(..., 'returnCancellationRatio', true)
%   additionally returns per-grid-point cancellation ratios. See
%   MOBIUS.INNERPRODUCTORBIT for interpretation.
%
%   [VALS, RATIOS, TERMMASS] = MOBIUS.INNERPRODUCTORBITGRID(...,
%   'returnCancellationRatio', true) additionally returns the
%   per-grid-point worst-term magnitudes prefactor * max_orb|term|,
%   length-N_u. Callers that integrate VALS over the grid use this to
%   form a mass-aware global cancellation diagnostic
%   |sum(VALS)| / sum(TERMMASS): a grid point where the alternating sum
%   cancels exactly to a true zero contributes nothing to the numerator
%   or the integral, so — unlike the pointwise minimum of RATIOS — the
%   global diagnostic is not driven to zero by zero-mass points.
%
%   Inputs:
%     K_U  (N_u, n_A, n_B) double — stack of kernels per u-grid point.
%     W_A  (n_A, 1)        double — A-side weights (shared across u).
%     W_B  (n_B, 1)        double — B-side weights (shared across u).
%     R    integer         — tensor order, 2 <= R <= 12.
%
%   See also MOBIUS.INNERPRODUCTORBIT, MOBIUS.INNERPRODUCTORBITPWBATCHED.

    arguments
        K_u (:,:,:) double
        w_A (:,1) double
        w_B (:,1) double
        r (1,1) {mustBeInteger}
        opts.prefactor (1,1) double = 1.0
        opts.returnCancellationRatio (1,1) logical = false
    end

    N_u = size(K_u, 1);
    table = mobius.getOrbitTable(r);
    total = zeros(N_u, 1);
    maxAbsTerm = zeros(N_u, 1);
    % Sum of the terms' magnitudes. Adding n numbers carries a forward
    % error bounded by eps times the sum of their magnitudes, so this is
    % what an error bound on the alternating sum needs. Bounding it by
    % n * max|term| instead assumes every term is as large as the
    % largest; the terms decay, so that over-states the sum severalfold
    % and the over-statement grows with r.
    sumAbsTerm = zeros(N_u, 1);

    % Reserve a label for the u-axis distinct from any A/B label.
    % A labels run 1..qA, B labels run qA+1..qA+qB; use a high value.
    % U_LABEL = 1000 is the consumer-side convention; baked into
    % orb.recipeGrid at table-build time (see buildOrbitRecipes in
    % buildOrbitTable.m).

    for k = 1:numel(table)
        orb = table(k);
        nE = size(orb.edges, 1);
        operands = cell(1, orb.qA + orb.qB + nE);
        idx = 1;

        % Weight vectors: 1-D, shared across u.
        for alpha = 1:orb.qA
            operands{idx} = w_A .^ orb.m_A(alpha);
            idx = idx + 1;
        end
        for beta = 1:orb.qB
            operands{idx} = w_B .^ orb.m_B(beta);
            idx = idx + 1;
        end
        % Kernel powers carry the u-axis as their first dimension.
        for e = 1:nE
            m = orb.edges(e, 3);
            if m == 1
                operands{idx} = K_u;
            else
                operands{idx} = K_u .^ m;
            end
            idx = idx + 1;
        end

        contribution = mobius.executeRecipe(operands, orb.recipeGrid);
        % contribution is a length-N_u vector.
        contribution = contribution(:);
        term = orb.weight * orb.mu * contribution;
        total = total + term;
        absTerm    = abs(term);
        maxAbsTerm = max(maxAbsTerm, absTerm);
        sumAbsTerm = sumAbsTerm + absTerm;
    end

    vals = opts.prefactor * total;
    if opts.returnCancellationRatio
        ratios = ones(N_u, 1);
        nz = maxAbsTerm > 0;
        ratios(nz) = abs(total(nz)) ./ maxAbsTerm(nz);
        termMass    = opts.prefactor * maxAbsTerm;
        termMassSum = opts.prefactor * sumAbsTerm;
    else
        ratios = [];
        termMass    = [];
        termMassSum = [];
    end
end
