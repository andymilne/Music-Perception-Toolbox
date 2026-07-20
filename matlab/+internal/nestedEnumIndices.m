function [permIdx, combIdx] = nestedEnumIndices( ...
        validSlots, tagsValid, rLevels, symLevels)
%NESTEDENUMINDICES  Tag-scoped nested r-tuple enumeration (rep. B).
%   Generalises the two-level enumeration to arbitrary nesting depth by
%   recursing outermost-inward through the grouping columns of the tag
%   matrix. At L = 2 it reproduces the two-level result exactly.
%   validSlots : 1 x Kv slot indices (ascending) non-NaN for this event.
%   tagsValid  : Kv x (L-1) per-slot group ids, innermost-grouping first
%                (column 1 the finest grouping above the leaf slots,
%                column L-1 the outermost). A Kv-vector is the single-
%                column (L = 2) case.
%   rLevels    : 1 x L read-arities, innermost-outward (rLevels(1) leaf).
%   symLevels  : 1 x L per-level symmetrisation.
%   Returns permIdx, combIdx: D x M slot-index arrays, D = prod(rLevels).
%   permIdx is the symmetrised deposit (each level permuted into its
%   orbit when that level's sym is set, else listed order); combIdx is
%   the canonical one-per-combination side (combinations at every level)
%   used for inner-product pairing. Columns concatenate outermost-group-
%   major, innermost-slot-minor. Shared by buildExpTens's nested fill
%   loop and evalExpTens's factored centres path.
    if isvector(tagsValid)
        tagsValid = tagsValid(:);            % Kv x 1 (L = 2 single column)
    end
    rLevels = rLevels(:).';
    symLevels = logical(symLevels(:).');
    L = numel(rLevels);
    D = prod(rLevels);
    Kv = numel(validSlots);
    permCols = localEnumSide(1:Kv, L, validSlots(:).', tagsValid, ...
                             rLevels, symLevels);
    combCols = localEnumSide(1:Kv, L, validSlots(:).', tagsValid, ...
                             rLevels, false(1, L));
    if isempty(permCols), permIdx = zeros(D, 0); else, permIdx = [permCols{:}]; end
    if isempty(combCols), combIdx = zeros(D, 0); else, combIdx = [combCols{:}]; end
end


function cols = localEnumSide(rowset, level, validSlots, tagsValid, ...
                              rLevels, symFlags)
    %LOCALENUMSIDE  Recursive enumeration. `rowset` are row indices into
    %   validSlots/tagsValid. Returns a cell row of column vectors, each of
    %   length prod(rLevels(1:level)) holding emitted slot indices.
    rowset = rowset(:).';
    if level == 1
        r0 = rLevels(1);
        k = numel(rowset);
        if r0 > k
            cols = {};
            return
        elseif r0 == k
            combRows = rowset;                        % single combination
        else
            % nchoosek(1:k, r0) is (nCk x r0); map positions to slot row
            % indices via rowset. reshape guards the r0 = 1 case: there the
            % index is a column vector and plain v(idx) would follow the row
            % vector rowset's orientation, collapsing nCk combinations of one
            % into a single combination of nCk. Forcing the (nCk x r0) shape
            % keeps each row a distinct combination.
            combPos = nchoosek(1:k, r0);              % nCk x r0 position rows
            combRows = reshape(rowset(combPos), size(combPos));
        end
        if symFlags(1)
            combRows = localExpandPerms(combRows);
        end
        nC = size(combRows, 1);
        cols = cell(1, nC);
        for i = 1:nC
            cols{i} = validSlots(combRows(i, :)).';   % r0 x 1 slot column
        end
        return
    end
    col = level - 1;
    gids = tagsValid(rowset, col).';                  % 1 x k group ids
    ug = unique(gids);                                % ascending
    ng = numel(ug);
    rg = rLevels(level);
    if rg > ng
        cols = {};
        return
    elseif rg == ng
        gsel = 1:ng;
    else
        gsel = nchoosek(1:ng, rg);                    % nG x rg positions
    end
    if symFlags(level)
        gsel = localExpandPerms(gsel);
    end
    cols = {};
    for s = 1:size(gsel, 1)
        pickPos = gsel(s, :);
        perGroup = cell(1, rg);
        ok = true;
        for j = 1:rg
            g = ug(pickPos(j));
            subrows = rowset(gids == g);
            perGroup{j} = localEnumSide(subrows, level - 1, validSlots, ...
                                        tagsValid, rLevels, symFlags);
            if isempty(perGroup{j})
                ok = false;
                break
            end
        end
        if ~ok
            continue
        end
        counts = cellfun(@numel, perGroup);
        total = prod(counts);
        for c = 0:total - 1
            choice = zeros(1, rg);
            rem = c;
            for j = rg:-1:1            % last group varies fastest
                choice(j) = mod(rem, counts(j)) + 1;
                rem = floor(rem / counts(j));
            end
            seg = [];
            for j = 1:rg
                seg = [seg; perGroup{j}{choice(j)}];  %#ok<AGROW>
            end
            cols{end + 1} = seg;       %#ok<AGROW>
        end
    end
end


function out = localExpandPerms(combs)
    %LOCALEXPANDPERMS  Expand each row (a combination) into its full
    %   permutation orbit; stacks rows of width size(combs, 2).
    if isempty(combs)
        out = combs;
        return
    end
    r = size(combs, 2);
    P = perms(1:r);
    nP = size(P, 1);
    nC = size(combs, 1);
    out = zeros(nC * nP, r);
    row = 0;
    for i = 1:nC
        for pp = 1:nP
            row = row + 1;
            out(row, :) = combs(i, P(pp, :));
        end
    end
end
