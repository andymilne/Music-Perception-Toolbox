function subs = lin2sub(sizes, li)
%LIN2SUB  Column-major linear index -> subscripts for a size vector.
    K = numel(sizes); subs = zeros(1, K); rem = li - 1;
    for k = 1:K
        subs(k) = mod(rem, sizes(k)) + 1;
        rem = floor(rem / sizes(k));
    end
end
