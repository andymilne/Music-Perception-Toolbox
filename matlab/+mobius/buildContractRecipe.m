function recipe = buildContractRecipe(opAxes, freeAxes)
%MOBIUS.BUILDCONTRACTRECIPE  Precompute a contraction recipe for executeRecipe.
%
%   RECIPE = MOBIUS.BUILDCONTRACTRECIPE(OPAXES, FREEAXES) precomputes
%   the sequence of merge steps that MOBIUS.CONTRACT would have
%   performed dynamically, given a list of operand axis labels and a
%   set of free (output) axis labels. The recipe records, per step,
%   the operand-list indices to merge, the permutations to apply, and
%   the dimension classifications (kept-shared, contracted, A-solo,
%   B-solo). The result is consumed by MOBIUS.EXECUTERECIPE at
%   runtime, which does not invoke any set operations
%   (intersect/setdiff/unique/ismember) and therefore avoids the bulk
%   of MOBIUS.CONTRACT's per-call dispatch overhead.
%
%   This is intended to be called once per orbit at table-build time;
%   the recipe is then embedded in the orbit table struct. The cost
%   of building the recipe is paid once when the orbit table is
%   constructed, not per inner-product call.
%
%   Inputs:
%     OPAXES    Cell array of integer-vector axis labels;
%               OPAXES{k}(d) names dimension d of operand k. Must be
%               consistent with the corresponding operands at runtime.
%     FREEAXES  Integer vector of axis labels to KEEP in the output.
%
%   Output:
%     RECIPE    Struct with fields:
%       .steps         struct array of merge steps (length = N-1
%                      where N = numel(OPAXES))
%       .finalSumDims  dim indices to sum out at the end
%                      (descending order)
%       .finalPerm     permutation to apply to final result, or []
%                      if no permute needed
%       .isScalar      true iff freeAxes is empty (output is a scalar)
%
%   Each step has fields:
%     .lo / .hi      indices in the current operand list to merge;
%                    operand at .lo receives the merged result, .hi
%                    is dropped. Subsequent step indices are
%                    interpreted in the post-drop list.
%     .lhsPerm /     permutations to apply to operand at .lo / .hi
%     .rhsPerm       before the reshape-and-contract step.
%     .nKeep         number of kept-shared axes (page dims).
%     .nContract     number of axes contracted in this step.
%     .nLhsSolo      number of axes unique to operand at .lo.
%     .nRhsSolo      number of axes unique to operand at .hi.
%
%   Algorithm: identical to MOBIUS.CONTRACT's greedy pair-picker
%   (most-shared-axes first, tiebreak by smaller estimated merge size)
%   but using abstract-axis information only. Operand "size" for the
%   tiebreak proxy is the count of unique axis labels (since actual
%   sizes are not known at table-build time).
%
%   See also MOBIUS.EXECUTERECIPE, MOBIUS.CONTRACT, MOBIUS.BUILDORBITTABLE.

    arguments
        opAxes cell
        freeAxes (1,:) double {mustBeInteger}
    end

    N = numel(opAxes);
    steps = struct('lo', {}, 'hi', {}, ...
                    'lhsPerm', {}, 'rhsPerm', {}, ...
                    'nKeep', {}, 'nContract', {}, ...
                    'nLhsSolo', {}, 'nRhsSolo', {});

    if N == 0
        recipe = struct('steps', steps, ...
                         'finalSumDims', [], ...
                         'finalPerm', [], ...
                         'isScalar', isempty(freeAxes));
        return;
    end
    if N == 1
        % No merges; just final reduction on the lone operand.
        finalAxes = opAxes{1};
        recipe = makeFinalRecipe(steps, finalAxes, freeAxes);
        return;
    end

    % Track current operand list as a cell of axis-label vectors.
    curAxes = opAxes(:);

    while numel(curAxes) > 1
        [iBest, jBest] = pickBestPairAbstract(curAxes);

        % Other operands' axes (excluding the pair).
        others = setdiff(1:numel(curAxes), [iBest, jBest]);
        otherAxes = [];
        for kk = others
            otherAxes = unique([otherAxes, curAxes{kk}]);
        end

        % Classify axes for this merge.
        axA = curAxes{iBest};
        axB = curAxes{jBest};
        sharedAxes = intersect(axA, axB);
        contractAxes = setdiff(sharedAxes, [otherAxes, freeAxes]);
        keepShared = setdiff(sharedAxes, contractAxes);
        aSolo = setdiff(axA, sharedAxes, 'stable');
        bSolo = setdiff(axB, sharedAxes, 'stable');

        targetA = [keepShared, contractAxes, aSolo];
        targetB = [keepShared, contractAxes, bSolo];

        permA = arrayfun(@(a) find(axA == a, 1), targetA);
        permB = arrayfun(@(a) find(axB == a, 1), targetB);

        step = struct( ...
            'lo', min(iBest, jBest), ...
            'hi', max(iBest, jBest), ...
            'lhsPerm', permA, ...
            'rhsPerm', permB, ...
            'nKeep', numel(keepShared), ...
            'nContract', numel(contractAxes), ...
            'nLhsSolo', numel(aSolo), ...
            'nRhsSolo', numel(bSolo));

        % If iBest > jBest, we've swapped roles: lhsPerm / rhsPerm need
        % to follow the lo/hi convention (lo holds the merged result).
        if iBest > jBest
            step.lhsPerm = permB;
            step.rhsPerm = permA;
            step.nLhsSolo = numel(bSolo);
            step.nRhsSolo = numel(aSolo);
        end

        steps(end+1) = step; %#ok<AGROW>

        % Update tracker. The merged-axis order must match what
        % executeRecipe will produce: it does L-solo then R-solo,
        % where L is at .lo and R is at .hi. With iBest > jBest,
        % .lo = jBest and .hi = iBest, so L was originally B and
        % R was originally A.
        if iBest <= jBest
            mergedAxes = [keepShared, aSolo, bSolo];
        else
            mergedAxes = [keepShared, bSolo, aSolo];
        end
        hi = max(iBest, jBest);
        lo = min(iBest, jBest);
        curAxes(hi) = [];
        curAxes{lo} = mergedAxes;
    end

    finalAxes = curAxes{1};
    recipe = makeFinalRecipe(steps, finalAxes, freeAxes);
end


% =========================================================================
%  Helpers
% =========================================================================

function recipe = makeFinalRecipe(steps, finalAxes, freeAxes)
    contractFinal = setdiff(finalAxes, freeAxes);
    contractDims = arrayfun(@(a) find(finalAxes == a, 1), contractFinal);
    keepAx = setdiff(finalAxes, contractFinal, 'stable');

    if isempty(freeAxes) || isempty(keepAx)
        finalPerm = [];
    else
        finalPerm = arrayfun(@(a) find(keepAx == a, 1), freeAxes);
        if isequal(finalPerm, 1:numel(finalPerm))
            finalPerm = [];
        end
    end

    recipe = struct( ...
        'steps', steps, ...
        'finalSumDims', sort(contractDims, 'descend'), ...
        'finalPerm', finalPerm, ...
        'isScalar', isempty(freeAxes));
end


function [iBest, jBest] = pickBestPairAbstract(curAxes)
    N = numel(curAxes);
    iBest = 1; jBest = 2;
    bestSharedCount = -1;
    bestMergeSize = inf;
    for i = 1:N - 1
        for j = i + 1:N
            shared = intersect(curAxes{i}, curAxes{j});
            ns = numel(shared);
            if ns > bestSharedCount
                bestSharedCount = ns;
                bestMergeSize = numel(curAxes{i}) + numel(curAxes{j});
                iBest = i; jBest = j;
            elseif ns == bestSharedCount
                ms = numel(curAxes{i}) + numel(curAxes{j});
                if ms < bestMergeSize
                    bestMergeSize = ms;
                    iBest = i; jBest = j;
                end
            end
        end
    end
end
