function [rsCanon, csCanon, MCanon] = canonicalForm(M, rowSums, colSums)
%MOBIUS.CANONICALFORM  Exact canonical form of a contingency table under
%   independent row and column permutations.
%
%   [RS_CANON, CS_CANON, M_CANON] = MOBIUS.CANONICALFORM(M, ROWSUMS, COLSUMS)
%   orders the columns by an orbit-invariant key (column sum, then the
%   sorted multiset of (row sum, entry) pairs); columns sharing a key
%   form a group. The canonical form is the lexicographic minimum, over
%   all products of within-group column permutations, of the matrix
%   with its rows sorted by (row sum, content). Two tables are in the
%   same orbit if and only if their canonical forms coincide, so the
%   orbit table has exactly |Omega_r| entries (OEIS A007716: 4, 10, 33,
%   91, 298, 910, 3017 for r = 2..8). Mirrors the Python
%   ``canonical_form`` step for step, so the two languages build
%   identical tables.
%
%   Inputs:
%     M       — qA-by-qB non-negative integer matrix.
%     ROWSUMS — 1-by-qA vector of row block sizes (matches sum(M,2)').
%     COLSUMS — 1-by-qB vector of column block sizes (matches sum(M,1)).
%
%   Outputs:
%     RS_CANON — Row block sizes in canonical row order (1-by-qA).
%     CS_CANON — Column block sizes in canonical column order (1-by-qB).
%     M_CANON  — Canonical matrix.
%
%   See also MOBIUS.GREEDYFORM, MOBIUS.ENUMERATECONTINGENCYTABLES.

    arguments
        M (:,:) {mustBeInteger, mustBeNonnegative}
        rowSums (1,:) {mustBeInteger, mustBeNonnegative}
        colSums (1,:) {mustBeInteger, mustBeNonnegative}
    end

    rs = double(rowSums(:)');
    cs = double(colSums(:)');
    M = double(M);
    qA = numel(rs);
    qB = numel(cs);

    % Column invariant: [colSum, sorted (rowSum, entry) pairs, flattened].
    inv = zeros(qB, 1 + 2 * qA);
    for j = 1:qB
        pairs = sortrows([rs(:), M(:, j)]);
        inv(j, :) = [cs(j), reshape(pairs', 1, [])];
    end
    [invSorted, order] = sortrows(inv);
    order = order(:)';

    % Groups: runs of equal invariant in sorted order.
    if qB > 1
        breaks = find(any(diff(invSorted, 1, 1) ~= 0, 2))';
    else
        breaks = [];
    end
    grpStart = [1, breaks + 1];
    grpEnd = [breaks, qB];
    nG = numel(grpStart);
    groupPerms = cell(1, nG);
    for g = 1:nG
        idx = order(grpStart(g):grpEnd(g));
        groupPerms{g} = perms(idx);          % one permutation per row
    end
    nPerms = cellfun(@(P) size(P, 1), groupPerms);

    % Odometer over the Cartesian product of within-group permutations.
    bestKey = [];
    counter = ones(1, nG);
    while true
        cols = zeros(1, qB);
        pos = 1;
        for g = 1:nG
            P = groupPerms{g}(counter(g), :);
            cols(pos:pos + numel(P) - 1) = P;
            pos = pos + numel(P);
        end
        Mc = M(:, cols);
        sortedRows = sortrows([rs(:), Mc]);
        key = reshape(sortedRows(:, 2:end)', 1, []);   % row-major content
        if isempty(bestKey) || localLexLess(key, bestKey)
            bestKey = key;
            rsCanon = sortedRows(:, 1)';
            csCanon = cs(cols);
            MCanon = sortedRows(:, 2:end);
        end
        g = nG;
        while g >= 1
            counter(g) = counter(g) + 1;
            if counter(g) <= nPerms(g)
                break
            end
            counter(g) = 1;
            g = g - 1;
        end
        if g < 1
            break
        end
    end
end

function tf = localLexLess(a, b)
    d = find(a ~= b, 1);
    tf = ~isempty(d) && a(d) < b(d);
end
