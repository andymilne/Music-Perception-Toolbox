function [recipeIP, recipeGrid, recipeBatched] = buildOrbitRecipes(orb)
%MOBIUS.BUILDORBITRECIPES  Per-orbit precomputed contraction recipes.
%
%   [RECIPE_IP, RECIPE_GRID, RECIPE_BATCHED] = MOBIUS.BUILDORBITRECIPES(ORB)
%   builds three precomputed contraction recipes for the orbit ORB,
%   corresponding to the three IP consumer call patterns:
%
%     - recipeIP       : MOBIUS.INNERPRODUCTORBIT (no batch, no free axes)
%     - recipeGrid     : MOBIUS.INNERPRODUCTORBITGRID (kernels carry U axis)
%     - recipeBatched  : MOBIUS.INNERPRODUCTORBITPWBATCHED (everything carries U)
%
%   Called once per orbit by MOBIUS.BUILDORBITTABLE; called on
%   first-load by MOBIUS.GETORBITTABLE for old pre-built .mat files
%   that predate the recipe fields. The U_LABEL value (1000) matches
%   the consumers' constant.
%
%   See also MOBIUS.BUILDCONTRACTRECIPE, MOBIUS.EXECUTERECIPE,
%            MOBIUS.BUILDORBITTABLE.

    qA = orb.qA;
    qB = orb.qB;
    nE = size(orb.edges, 1);
    nOps = qA + qB + nE;
    U_LABEL = 1000;

    % --- IP: weights have 1 axis, kernels have 2 axes; freeAxes = []
    opAxesIP = cell(1, nOps);
    idx = 1;
    for alpha = 1:qA
        opAxesIP{idx} = alpha;
        idx = idx + 1;
    end
    for beta = 1:qB
        opAxesIP{idx} = qA + beta;
        idx = idx + 1;
    end
    for e = 1:nE
        opAxesIP{idx} = [orb.edges(e, 1), qA + orb.edges(e, 2)];
        idx = idx + 1;
    end
    recipeIP = mobius.buildContractRecipe(opAxesIP, []);

    % --- Grid: weights as IP, kernels gain leading U axis; freeAxes = U
    opAxesGrid = cell(1, nOps);
    idx = 1;
    for alpha = 1:qA
        opAxesGrid{idx} = alpha;
        idx = idx + 1;
    end
    for beta = 1:qB
        opAxesGrid{idx} = qA + beta;
        idx = idx + 1;
    end
    for e = 1:nE
        opAxesGrid{idx} = [U_LABEL, orb.edges(e, 1), qA + orb.edges(e, 2)];
        idx = idx + 1;
    end
    recipeGrid = mobius.buildContractRecipe(opAxesGrid, U_LABEL);

    % --- PwBatched: every operand carries leading U axis; freeAxes = U
    opAxesBatched = cell(1, nOps);
    idx = 1;
    for alpha = 1:qA
        opAxesBatched{idx} = [U_LABEL, alpha];
        idx = idx + 1;
    end
    for beta = 1:qB
        opAxesBatched{idx} = [U_LABEL, qA + beta];
        idx = idx + 1;
    end
    for e = 1:nE
        opAxesBatched{idx} = [U_LABEL, orb.edges(e, 1), qA + orb.edges(e, 2)];
        idx = idx + 1;
    end
    recipeBatched = mobius.buildContractRecipe(opAxesBatched, U_LABEL);
end
