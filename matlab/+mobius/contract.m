function R = contract(operands, opAxes, freeAxes)
%MOBIUS.CONTRACT  Generic tensor contraction over labelled axes.
%
%   R = MOBIUS.CONTRACT(OPERANDS, OPAXES, FREEAXES) contracts a list of
%   tensors into a single tensor by summing over all axis labels that
%   are not in FREEAXES. The contraction is performed greedily by
%   pairwise merges, mirroring what np.einsum(..., optimize=True) does
%   on the Python side; correctness does not depend on the merge order.
%
%   Inputs:
%     OPERANDS  Cell array of N numerical tensors (each may be 1-D, 2-D,
%               or higher).
%     OPAXES    Cell array of N integer-vector axis labels;
%               OPAXES{k}(d) names dimension d of OPERANDS{k}. Numel
%               must match ndims(OPERANDS{k}) (or ndims when OPERANDS{k}
%               has trailing singleton dims explicitly populated).
%     FREEAXES  Integer vector of axis labels to KEEP in the output, in
%               the order they should appear in R's dimensions. Pass []
%               for a scalar result.
%
%   Output:
%     R         Tensor whose dimensions correspond to FREEAXES, in the
%               given order. Scalar (Python-style) when FREEAXES is empty.
%
%   The algorithm at each step picks the operand pair with the largest
%   shared-axis count and merges them, contracting any shared axis that
%   no other operand still uses (and is not free). Shared axes that
%   must persist are carried through via paged matrix multiplication —
%   they become batch axes in the merged tensor.
%
%   See also TENSORPROD, PAGEMTIMES.

    arguments
        operands cell
        opAxes cell
        freeAxes (1,:) double {mustBeInteger}
    end

    N = numel(operands);
    if N ~= numel(opAxes)
        error('mobius:contract:countMismatch', ...
            'numel(operands) must equal numel(opAxes).');
    end

    % Trivial cases.
    if N == 0
        R = 1;
        return
    end
    if N == 1
        R = singleOperandReduce(operands{1}, opAxes{1}, freeAxes);
        return
    end

    % Greedy pairwise contraction.
    while numel(operands) > 1
        [iBest, jBest] = pickBestPair(operands, opAxes);

        % Other operands' axes (excluding the pair).
        others = setdiff(1:numel(operands), [iBest, jBest]);
        otherAxes = [];
        for kk = others
            otherAxes = unique([otherAxes, opAxes{kk}]);
        end

        [merged, mergedAxes] = mergePair( ...
            operands{iBest}, opAxes{iBest}, ...
            operands{jBest}, opAxes{jBest}, ...
            otherAxes, freeAxes);

        % Replace iBest with merged, drop jBest.
        hi = max(iBest, jBest);
        lo = min(iBest, jBest);
        operands(hi) = []; opAxes(hi) = [];
        operands{lo} = merged; opAxes{lo} = mergedAxes;
    end

    % Final reduction + permutation. The remaining operand may still
    % carry axes that were never paired off (axes unique to a single
    % operand and not in free); sum those out before reordering.
    R = singleOperandReduce(operands{1}, opAxes{1}, freeAxes);
end


% ---------------------------------------------------------------------
% Single operand: sum out non-free axes, optionally permute.
% ---------------------------------------------------------------------
function R = singleOperandReduce(data, ax, freeAxes)
    contractAx = setdiff(ax, freeAxes);
    contractDims = arrayfun(@(a) find(ax == a, 1), contractAx);
    for d = sort(contractDims, 'descend')
        data = sum(data, d);
    end
    keepAx = setdiff(ax, contractAx, 'stable');
    R = reorderToFree(data, keepAx, freeAxes);
end


% ---------------------------------------------------------------------
% Final reorder to match the FREEAXES order; handle scalar / 1-D cases.
% ---------------------------------------------------------------------
function R = reorderToFree(data, dataAxes, freeAxes)
    if isempty(freeAxes)
        R = double(data(:));
        if numel(R) == 1
            R = R(1);
        end
        return
    end
    perm = arrayfun(@(a) find(dataAxes == a, 1), freeAxes);
    if ~isequal(perm, 1:numel(perm))
        % If data has trailing singletons, permute may need a
        % fully-populated permutation list.
        nDimsActual = max(ndims(data), max(perm));
        permFull = perm;
        for k = 1:nDimsActual
            if ~ismember(k, permFull)
                permFull(end+1) = k; %#ok<AGROW>
            end
        end
        data = permute(data, permFull);
    end
    R = data;
end


% ---------------------------------------------------------------------
% Pick a pair to merge: prefer the most shared labels (cuts the most
% in one step); tiebreak by smaller combined size of the merged tensor.
% ---------------------------------------------------------------------
function [iBest, jBest] = pickBestPair(operands, opAxes)
    N = numel(operands);
    iBest = 1; jBest = 2;
    bestSharedCount = -1;
    bestMergeSize = inf;
    for i = 1:N - 1
        for j = i + 1:N
            shared = intersect(opAxes{i}, opAxes{j});
            ns = numel(shared);
            if ns > bestSharedCount
                bestSharedCount = ns;
                bestMergeSize = mergeSizeEstimate( ...
                    operands{i}, opAxes{i}, operands{j}, opAxes{j});
                iBest = i; jBest = j;
            elseif ns == bestSharedCount
                ms = mergeSizeEstimate( ...
                    operands{i}, opAxes{i}, operands{j}, opAxes{j});
                if ms < bestMergeSize
                    bestMergeSize = ms;
                    iBest = i; jBest = j;
                end
            end
        end
    end
end


function s = mergeSizeEstimate(A, axA, B, axB)
    % Outer product upper bound; fine as a proxy.
    s = numel(A) * numel(B);
end


% ---------------------------------------------------------------------
% Merge two operands, contracting fully-resolved shared axes.
%
% Layout strategy:
%   permute A to (keepShared, contractAxes, soloA)
%   permute B to (keepShared, contractAxes, soloB)
%   reshape both to 3-D arrays (keep, contract, solo) where keep is the
%   product of keepShared sizes and so on.
%   pagemtimes(A_T, B_) computes the per-page matmul, contracting
%   contractAxes; A_T has been transposed page-wise so contract becomes
%   the inner dim.
% ---------------------------------------------------------------------
function [merged, mergedAxes] = mergePair(A, axA, B, axB, otherAxes, freeAxes)
    sharedAxes = intersect(axA, axB);
    contractAxes = setdiff(sharedAxes, [otherAxes, freeAxes]);
    keepShared = setdiff(sharedAxes, contractAxes);
    aSolo = setdiff(axA, sharedAxes, 'stable');
    bSolo = setdiff(axB, sharedAxes, 'stable');

    % Permute each operand to [keepShared, contractAxes, solo] order.
    A = permuteToOrder(A, axA, [keepShared, contractAxes, aSolo]);
    B = permuteToOrder(B, axB, [keepShared, contractAxes, bSolo]);

    % Compute aggregated sizes along each group.
    nKeep = numel(keepShared);
    nContract = numel(contractAxes);
    nASolo = numel(aSolo);
    nBSolo = numel(bSolo);

    [szKeepA, szContractA, szASolo] = sliceSize(A, [nKeep, nContract, nASolo], axA, [keepShared, contractAxes, aSolo]);
    [szKeepB, szContractB, szBSolo] = sliceSize(B, [nKeep, nContract, nBSolo], axB, [keepShared, contractAxes, bSolo]);

    % Sanity: keep and contract sizes must match.
    if ~isequal(szKeepA, szKeepB)
        error('mobius:contract:keepShapeMismatch', ...
            'keep-shared axis sizes differ between operands.');
    end
    if ~isequal(szContractA, szContractB)
        error('mobius:contract:contractShapeMismatch', ...
            'contraction axis sizes differ between operands.');
    end

    pKeep = max(prod(szKeepA), 1);
    pContract = max(prod(szContractA), 1);
    pASolo = max(prod(szASolo), 1);
    pBSolo = max(prod(szBSolo), 1);

    % Reshape to (pKeep, pContract, pASolo) and (pKeep, pContract, pBSolo).
    A_re = reshape(A, [pKeep, pContract, pASolo]);
    B_re = reshape(B, [pKeep, pContract, pBSolo]);

    if pContract == 1
        % No contraction: outer product per keep-page.
        % Reshape A to (pKeep, pASolo, 1), B to (pKeep, 1, pBSolo) and
        % broadcast-multiply via implicit expansion.
        A_x = reshape(A_re, [pKeep, pASolo, 1]);
        B_x = reshape(B_re, [pKeep, 1, pBSolo]);
        prod3 = A_x .* B_x;  % (pKeep, pASolo, pBSolo)
    else
        % Per-page matmul. pagemtimes(A, B) pages along dim 3+, with
        % matmul on dims 1-2. Permute so:
        %   A : (pASolo, pContract, pKeep)  -- M, K, page
        %   B : (pContract, pBSolo, pKeep)  -- K, N, page
        %   pagemtimes → (pASolo, pBSolo, pKeep)
        A_M = permute(A_re, [3, 2, 1]);
        B_M = permute(B_re, [2, 3, 1]);
        prod3pg = localPageMatMul(A_M, B_M);  % (pASolo, pBSolo, pKeep)
        % Bring keep-axis back to the front.
        prod3 = permute(prod3pg, [3, 1, 2]);  % (pKeep, pASolo, pBSolo)
    end

    % Reshape back to full multi-axis layout.
    finalSize = [szKeepA, szASolo, szBSolo];
    if isempty(finalSize)
        finalSize = [1, 1];
    elseif numel(finalSize) == 1
        finalSize = [finalSize, 1];
    end
    merged = reshape(prod3, finalSize);
    mergedAxes = [keepShared, aSolo, bSolo];
end


% ---------------------------------------------------------------------
% Permute a tensor so its axes appear in the specified order.
% Handles mismatches between numel(axis labels) and ndims(data) by
% padding with trailing singleton-dim labels.
% ---------------------------------------------------------------------
function out = permuteToOrder(data, currentAxes, targetAxes)
    if isequal(currentAxes, targetAxes)
        out = data;
        return
    end
    % Build perm: for each target label, find its position in current.
    perm = arrayfun(@(a) find(currentAxes == a, 1), targetAxes);
    % Ensure perm covers all dimensions of data (pad with any unused).
    unused = setdiff(1:max(ndims(data), numel(currentAxes)), perm);
    perm = [perm, unused];
    out = permute(data, perm);
end


% ---------------------------------------------------------------------
% Compute size vectors for the three groups (keep, contract, solo)
% within a permuted operand. Returns row vectors.
% ---------------------------------------------------------------------
function [szKeep, szContract, szSolo] = sliceSize(data, groupSizes, ~, ~)
    % groupSizes: 1x3 [nKeep, nContract, nSolo]
    sz = size(data);
    sz = [sz, ones(1, max(0, sum(groupSizes) - numel(sz)))];
    nKeep = groupSizes(1);
    nContract = groupSizes(2);
    nSolo = groupSizes(3);
    szKeep = sz(1:nKeep);
    szContract = sz(nKeep + (1:nContract));
    szSolo = sz(nKeep + nContract + (1:nSolo));
end


% ---------------------------------------------------------------------
% Page-wise matrix multiplication. Mirrors PAGEMTIMES (R2020b+); falls
% back to a for-loop on environments that don't ship it (Octave).
% ---------------------------------------------------------------------
function C = localPageMatMul(A, B)
    if exist('pagemtimes', 'builtin') == 5 || exist('pagemtimes', 'file')
        C = pagemtimes(A, B);
        return
    end
    sA = size(A);
    sB = size(B);
    M = sA(1);
    K = sA(2);
    N = sB(2);
    pageDimsA = sA(3:end);
    pageDimsB = sB(3:end);
    if isempty(pageDimsA); pageDimsA = 1; end
    if isempty(pageDimsB); pageDimsB = 1; end
    pageDims = max([pageDimsA, ones(1, max(0, numel(pageDimsB) - numel(pageDimsA)))], ...
                   [pageDimsB, ones(1, max(0, numel(pageDimsA) - numel(pageDimsB)))]);
    nPages = prod(pageDims);
    A3 = reshape(A, M, K, []);
    B3 = reshape(B, K, N, []);
    nA = size(A3, 3);
    nB = size(B3, 3);
    C3 = zeros(M, N, nPages);
    for p = 1:nPages
        pA = min(p, nA);
        pB = min(p, nB);
        C3(:, :, p) = A3(:, :, pA) * B3(:, :, pB);
    end
    if numel(pageDims) <= 1
        C = reshape(C3, [M, N, pageDims]);
    else
        C = reshape(C3, [M, N, pageDims]);
    end
end
