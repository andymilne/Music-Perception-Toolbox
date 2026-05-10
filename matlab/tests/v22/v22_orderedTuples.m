function P = v22_orderedTuples(n, k)
%V22_ORDEREDTUPLES  All ordered k-tuples of distinct elements from 1..n.
%
%   P = V22_ORDEREDTUPLES(N, K) returns a matrix whose rows enumerate
%   the n!/(n-k)! distinct ordered k-tuples of [1..N]. Test helper for
%   v22 brute-force ground-truth checks; not intended for production
%   call paths.

    if k == 0
        P = zeros(1, 0);
        return
    end
    if k == 1
        P = (1:n)';
        return
    end
    sub = v22_orderedTuples(n, k - 1);
    P = zeros(0, k);
    for j = 1:n
        keep = true(size(sub, 1), 1);
        for col = 1:k - 1
            keep = keep & (sub(:, col) ~= j);
        end
        kept = sub(keep, :);
        P = [P; [j * ones(size(kept, 1), 1), kept]]; %#ok<AGROW>
    end
end
