function [total, maxAbs] = mobiusPartitionCombine(blockContrib, partBlockIdx, mus, trackMax)
%MOBIUS.MOBIUSPARTITIONCOMBINE  Möbius set-partition sum from block factors.
%
%   Computes sum_pi mu(pi) prod_{B in pi} blockContrib{B}, where blockContrib
%   is indexed by the distinct-block ordering of
%   mobius.getPartitionBlockStructure and partBlockIdx / mus are its reuse
%   map and Möbius weights. All contributions share a shape (the query axis,
%   of any size), so this is mode-agnostic: (n_q, 1) in absolute mode,
%   (N_u, nQc) per chunk in the relative factored strategy.
%
%   Returns total and, when trackMax is true, maxAbs --- the per-query
%   maximum |mu(pi) prod ...| over partitions for the alternating-sum
%   cancellation diagnostic (otherwise []).

    if nargin < 4
        trackMax = true;
    end
    sz = size(blockContrib{1});
    total = zeros(sz);
    if trackMax
        maxAbs = zeros(sz);
    else
        maxAbs = [];
    end
    for iPart = 1:numel(partBlockIdx)
        blockProd = ones(sz);
        bidx = partBlockIdx{iPart};
        for b = 1:numel(bidx)
            blockProd = blockProd .* blockContrib{bidx(b)};
        end
        term = mus(iPart) * blockProd;
        total = total + term;
        if trackMax
            maxAbs = max(maxAbs, abs(term));
        end
    end
end
