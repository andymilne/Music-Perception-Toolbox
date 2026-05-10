function mu = mobiusForBlocksizes(mTuple)
%MOBIUS.MOBIUSFORBLOCKSIZES  Mobius coefficient mu(0_hat, pi) for a
%   partition with the given block sizes.
%
%   MU = MOBIUS.MOBIUSFORBLOCKSIZES(MTUPLE) returns the standard Mobius
%   coefficient on the partition lattice (Rota 1964):
%
%       mu(0_hat, pi) = prod_l (-1)^(m_l - 1) * (m_l - 1)!
%
%   where MTUPLE = (m_1, ..., m_q) lists the block sizes. Depends only on
%   the multiset of block sizes, not the labelling.
%
%   The empty partition returns 1 (empty product convention).
%
%   Examples:
%     mobius.mobiusForBlocksizes([1])      == 1
%     mobius.mobiusForBlocksizes([2])      == -1
%     mobius.mobiusForBlocksizes([3])      == 2
%     mobius.mobiusForBlocksizes([2 1])    == -1
%     mobius.mobiusForBlocksizes([3 2 1])  == -2
%
%   See also MOBIUS.AUTSIZE, MOBIUS.INTEGERPARTITIONS.

    arguments
        mTuple (1,:) {mustBeInteger, mustBePositive}
    end

    if isempty(mTuple)
        mu = 1;
        return
    end

    mu = prod((-1).^(mTuple - 1) .* factorial(mTuple - 1));
end
