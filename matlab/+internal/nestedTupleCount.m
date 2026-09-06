function n = nestedTupleCount(tags, rLevels, symLevels, weights)
%NESTEDTUPLECOUNT  Ordered tuple centres a nested attribute enumerates.
%
%   n = internal.nestedTupleCount(tags, rLevels, symLevels) is the number
%   of tuple centres the centres route materialises per event for a
%   nested attribute with the given tag tree: at each symmetric level
%   every ordered selection of r children (n! / (n - r)!), at each
%   ordered level every selection in listed order (C(n, r)), multiplied
%   down the tree. For ragged groups the per-level sum over selections
%   is the elementary symmetric polynomial of the children's counts.
%   tags is K x 1 or K x (L - 1) as in the spec; NaN values are the
%   caller's to drop. With WEIGHTS (one per value) the same sum is taken
%   over the products of the selected values' weights, which is the
%   density's total tuple weight --- its mass up to the kernel's volume.
%   Twin of the Python mpt._tensor.dispatch.nested_tuple_count.

    if nargin < 4; weights = []; end
    tags = double(tags);
    if isvector(tags)
        tags = tags(:);
    end
    K = size(tags, 1);
    rLevels = double(rLevels(:)).';
    symLevels = logical(symLevels(:)).';
    L = numel(rLevels);
    n = localCount(L, (1:K).', tags, rLevels, symLevels, double(weights(:)).');
end


function c = localCount(level, idx, tags, rLevels, symLevels, weights)
    r = rLevels(level);
    if level == 1
        m = numel(idx);
        if r > m
            c = 0;
            return;
        end
        if isempty(weights)
            sel = nchoosek(m, r);
        else
            sel = localEsp(weights(idx), r);
        end
    else
        keys = tags(idx, level - 1);
        u = unique(keys);
        counts = zeros(1, numel(u));
        for i = 1:numel(u)
            counts(i) = localCount(level - 1, idx(keys == u(i)), tags, rLevels, symLevels, weights);
        end
        if r > numel(counts)
            c = 0;
            return;
        end
        sel = localEsp(counts, r);
    end
    if symLevels(level)
        c = sel * factorial(r);
    else
        c = sel;
    end
end


function e = localEsp(counts, r)
    % Elementary symmetric polynomial e_r of the children's counts.
    ev = [1, zeros(1, r)];
    for i = 1:numel(counts)
        for j = r:-1:1
            ev(j + 1) = ev(j + 1) + ev(j) * counts(i);
        end
    end
    e = ev(r + 1);
end
