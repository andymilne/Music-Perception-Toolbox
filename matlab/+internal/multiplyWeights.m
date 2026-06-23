function wNew = multiplyWeights(wExisting, factor, K_target)
%LOCALMULTIPLYWEIGHTS  Multiply per-attribute weight by factor ((1, N) row).
%
%   factor is (1, N); the target attribute's existing weight may be
%   [], a scalar, a (1, N) row, or a (K_target, N) matrix. The factor
%   broadcasts across K_target slots (every slot of every event sees
%   the same factor).
    if isempty(wExisting)
        % factor broadcast to (K_target, N).
        wNew = repmat(factor, K_target, 1);
        return;
    end
    if isnumeric(wExisting) && isscalar(wExisting)
        wNew = repmat(double(wExisting) * factor, K_target, 1);
        return;
    end
    arr = double(wExisting);
    if isequal(size(arr), [1, size(factor, 2)])
        % (1, N) row: broadcast to K_target after multiplying.
        wNew = repmat(arr .* factor, K_target, 1);
        return;
    end
    if isequal(size(arr), [K_target, size(factor, 2)])
        % (K_target, N): per-slot weights, broadcast factor across rows.
        wNew = arr .* factor;
        return;
    end
    error('mptWindowing:badExistingWeightShape', ...
          ['Existing weight shape [%s] is incompatible with target ' ...
           'shape [%d, %d] (factor is (1, %d)).'], ...
          num2str(size(arr)), K_target, size(factor, 2), size(factor, 2));
end
