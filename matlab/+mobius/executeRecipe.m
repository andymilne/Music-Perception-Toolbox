function R = executeRecipe(operands, recipe)
%MOBIUS.EXECUTERECIPE  Execute a precomputed contraction recipe.
%
%   R = MOBIUS.EXECUTERECIPE(OPERANDS, RECIPE) consumes the cell
%   array of tensors OPERANDS together with a recipe produced by
%   MOBIUS.BUILDCONTRACTRECIPE and returns the contracted tensor R.
%
%   Equivalent to MOBIUS.CONTRACT(OPERANDS, OPAXES, FREEAXES) where
%   OPAXES and FREEAXES were the inputs originally passed to
%   MOBIUS.BUILDCONTRACTRECIPE, but executed without any
%   intersect / setdiff / unique / ismember calls. This eliminates
%   the per-merge dispatch overhead that dominates MOBIUS.CONTRACT's
%   wall time for the orbit-IP use case.
%
%   See also MOBIUS.BUILDCONTRACTRECIPE, MOBIUS.CONTRACT.

    nOps = numel(operands);
    if nOps == 0
        R = 1;
        return;
    end
    if nOps == 1
        R = finalReduce(operands{1}, recipe);
        return;
    end

    ops = operands;
    nSteps = numel(recipe.steps);
    for s = 1:nSteps
        step = recipe.steps(s);
        A = ops{step.lo};
        B = ops{step.hi};

        % Apply permutations, padding for trailing singletons.
        nDimA = ndims(A);
        permLen = numel(step.lhsPerm);
        if nDimA > permLen
            permA = [step.lhsPerm, (permLen + 1):nDimA];
        else
            permA = step.lhsPerm;
        end
        if ~isequal(permA, 1:numel(permA))
            A = permute(A, permA);
        end

        nDimB = ndims(B);
        permLen = numel(step.rhsPerm);
        if nDimB > permLen
            permB = [step.rhsPerm, (permLen + 1):nDimB];
        else
            permB = step.rhsPerm;
        end
        if ~isequal(permB, 1:numel(permB))
            B = permute(B, permB);
        end

        % Compute aggregated sizes from the actual operands.
        szA = size(A);
        szB = size(B);
        totalDimsA = step.nKeep + step.nContract + step.nLhsSolo;
        totalDimsB = step.nKeep + step.nContract + step.nRhsSolo;
        if numel(szA) < totalDimsA
            szA = [szA, ones(1, totalDimsA - numel(szA))];
        end
        if numel(szB) < totalDimsB
            szB = [szB, ones(1, totalDimsB - numel(szB))];
        end

        szKeep   = szA(1:step.nKeep);
        szContrA = szA(step.nKeep + (1:step.nContract));
        szLSolo  = szA(step.nKeep + step.nContract + (1:step.nLhsSolo));
        szRSolo  = szB(step.nKeep + step.nContract + (1:step.nRhsSolo));

        pKeep     = max(prod(szKeep), 1);
        pContract = max(prod(szContrA), 1);
        pLSolo    = max(prod(szLSolo), 1);
        pRSolo    = max(prod(szRSolo), 1);

        A_re = reshape(A, [pKeep, pContract, pLSolo]);
        B_re = reshape(B, [pKeep, pContract, pRSolo]);

        if step.nContract == 0 || pContract == 1
            % No contraction: outer product per keep-page.
            A_x = reshape(A_re, [pKeep, pLSolo, 1]);
            B_x = reshape(B_re, [pKeep, 1, pRSolo]);
            prod3 = A_x .* B_x;
        else
            % Per-page matmul.
            A_M = permute(A_re, [3, 2, 1]);
            B_M = permute(B_re, [2, 3, 1]);
            prod3pg = pagemtimes(A_M, B_M);
            prod3 = permute(prod3pg, [3, 1, 2]);
        end

        finalSize = [szKeep, szLSolo, szRSolo];
        if isempty(finalSize)
            finalSize = [1, 1];
        elseif numel(finalSize) == 1
            finalSize = [finalSize, 1];
        end
        merged = reshape(prod3, finalSize);

        ops{step.lo} = merged;
        ops(step.hi) = [];
    end

    R = finalReduce(ops{1}, recipe);
end


function R = finalReduce(data, recipe)
    for d = recipe.finalSumDims
        data = sum(data, d);
    end
    if recipe.isScalar
        R = double(data(:));
        if numel(R) == 1
            R = R(1);
        end
        return;
    end
    if ~isempty(recipe.finalPerm)
        nDimsActual = max(ndims(data), max(recipe.finalPerm));
        permLen = numel(recipe.finalPerm);
        if nDimsActual > permLen
            permFull = [recipe.finalPerm, (permLen + 1):nDimsActual];
        else
            permFull = recipe.finalPerm;
        end
        if ~isequal(permFull, 1:numel(permFull))
            data = permute(data, permFull);
        end
    end
    R = data;
end
