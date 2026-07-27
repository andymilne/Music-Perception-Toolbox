function [permMat, combMat, permW, combW] = ...
        enumFlatAttr(valCol, valid, r_a, isSym, wColOrig)
%ENUMFLATATTR  Per-(event, attribute) r-ad enumeration for one flat
%   attribute. Returns the perm/comb value-index matrices and their
%   per-tuple weight products for the non-NaN values `valid` of value
%   column `valCol` at tuple size `r_a`. Applies the r = 1 equal-value
%   collapse (summing weights). Shared by buildExpTens's per-(n, a) fill
%   loop and evalExpTens's factored centres path so both produce
%   identical tuples. Caller guarantees numel(valid) >= r_a. Twin of
%   Python _enum_flat_attr.
    K_na = numel(valid);
    collapsed = false;
    if r_a == 1 && K_na > 1
        valsValid = valCol(valid);
        [uniqueVals, firstIdx, inverse] = unique(valsValid, 'first');
        if numel(firstIdx) < K_na
            wColLocal = wColOrig;
            summed = accumarray(inverse, wColOrig(valid), ...
                                 [numel(uniqueVals), 1]);
            wColLocal(valid(firstIdx)) = summed;
            valid = valid(firstIdx);
            K_na  = numel(valid);
            collapsed = true;
        end
    end
    if ~collapsed
        wColLocal = wColOrig;
    end

    % Combinations: r_a x C(K_na, r_a)
    if K_na == r_a
        combMat = valid(:);
    else
        combMat = nchoosek(valid, r_a).';
    end

    % Permutations: r_a x (r_a! * C) when symmetric; ordered (isSym = 0)
    % keeps each combination in listed order (perm side == comb side);
    % r_a = 1 has no order to symmetrise either way.
    if r_a == 1 || ~isSym
        permMat = combMat;
    else
        Pm = perms(1:r_a).';
        nC = size(combMat, 2);
        nP = size(Pm, 2);
        permMat = zeros(r_a, nC * nP);
        for pp = 1:nP
            permMat(:, (pp - 1) * nC + 1 : pp * nC) = combMat(Pm(:, pp), :);
        end
    end

    % Slot-weight products (per-tuple).
    wCol = wColLocal;
    if r_a == 1
        permW = reshape(wCol(permMat), 1, []);
        combW = reshape(wCol(combMat), 1, []);
    else
        permW = prod(reshape(wCol(permMat), r_a, []), 1);
        combW = prod(reshape(wCol(combMat), r_a, []), 1);
    end
end
