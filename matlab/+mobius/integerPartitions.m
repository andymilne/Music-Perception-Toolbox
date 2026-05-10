function P = integerPartitions(r, maxPart)
%MOBIUS.INTEGERPARTITIONS  All integer partitions of r as decreasing rows.
%
%   P = MOBIUS.INTEGERPARTITIONS(R) returns all integer partitions of R
%   as a cell array of weakly decreasing row vectors.
%
%   P = MOBIUS.INTEGERPARTITIONS(R, MAXPART) restricts the largest part
%   to MAXPART (used internally by the recursion; rarely needed by
%   external callers).
%
%   The empty partition of zero is returned as a 1-by-1 cell containing
%   the empty row vector [], so numel(integerPartitions(0)) == 1.
%
%   Examples:
%     P = mobius.integerPartitions(4)
%     %   {[4]} {[3,1]} {[2,2]} {[2,1,1]} {[1,1,1,1]}
%
%   See also MOBIUS.AUTSIZE, MOBIUS.MOBIUSFORBLOCKSIZES.

    arguments
        r (1,1) {mustBeInteger, mustBeNonnegative}
        maxPart (1,1) {mustBeInteger, mustBeNonnegative} = r
    end

    if r == 0
        P = {[]};
        return
    end

    P = {};
    for p = min(r, maxPart):-1:1
        rest = mobius.integerPartitions(r - p, p);
        for k = 1:numel(rest)
            P{end+1} = [p, rest{k}]; %#ok<AGROW>
        end
    end
end
