function [p, w] = roundedSorted(p, w, period, roundDigits)
%ROUNDEDSORTED Round canonical values, fold into the period, re-sort.
%   [p, w] = internal.roundedSorted(p, w, period, roundDigits)
%
%   Rounds the values to roundDigits decimal places (default 9; as
%   internal.cyclicCanonical does), folds them back into [0, period) when period is non-empty,
%   and re-sorts them with their weights. Rounding absorbs the
%   floating-point noise left by mod-reduction and subtraction, which
%   would otherwise give equivalent sets different keys (e.g. 22-EDO
%   pitches reached by different arithmetic). Returns row vectors; w
%   may be [].
%
%   Twin of _rounded_sorted in python/mpt/_tensor/canonical.py.

    if nargin < 4 || isempty(roundDigits)
        roundDigits = 9;
    end
    p = round(p(:)', roundDigits);
    if ~isempty(period)
        p = mod(p, period);
    end
    [p, si] = sort(p);
    if ~isempty(w)
        w = w(:)';
        w = w(si);
    end
end
