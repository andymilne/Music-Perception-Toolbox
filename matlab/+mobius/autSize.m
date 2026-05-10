function s = autSize(mTuple)
%MOBIUS.AUTSIZE  Size of the automorphism group of an integer partition.
%
%   S = MOBIUS.AUTSIZE(MTUPLE) returns the size of the group of
%   permutations of the blocks of an integer partition that preserve
%   block sizes. For a partition with k_j blocks of each distinct size,
%   this is prod(factorial(k_j)).
%
%   Examples:
%     mobius.autSize([])              == 1
%     mobius.autSize([3])             == 1
%     mobius.autSize([2 2])           == 2
%     mobius.autSize([3 3 2 1 1])     == 4   % 2! * 1! * 2!
%     mobius.autSize([1 1 1 1])       == 24  % 4!
%
%   See also MOBIUS.INTEGERPARTITIONS, MOBIUS.MOBIUSFORBLOCKSIZES.

    arguments
        mTuple (1,:) {mustBeInteger, mustBeNonnegative}
    end

    if isempty(mTuple)
        s = 1;
        return
    end

    [~, ~, ic] = unique(mTuple);
    counts = accumarray(ic, 1);
    s = prod(factorial(counts));
end
