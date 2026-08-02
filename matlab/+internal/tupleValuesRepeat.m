function tf = tupleValuesRepeat(C, nQ, minQueries, minEntries, minSaving)
%TUPLEVALUESREPEAT  Does tabulating the kernel on distinct values pay?
%
%   TF = INTERNAL.TUPLEVALUESREPEAT(C, NQ) is true when evaluating the
%   kernel once per distinct value in C and reading the tuple layout off
%   that table costs less than evaluating it once per tuple.
%
%   Every coordinate of an r-tuple is a value of the same multiset, so
%   the distinct arguments number the multiset size rather than the tuple
%   count, and the saving per query grows with the tuple count. Against
%   that stands the sort that finds the distinct values, which is paid
%   once per call however many queries follow. The trade is therefore
%   settled by the query count NQ: measured on the calibration grid, the
%   sort costs 0.17 ms at 4512 centre entries rising to 59.8 ms at
%   1020096, against a per-query saving of 0.005 ms to 0.77 ms over the
%   same range, putting the break-even between 35 and 78 queries. The
%   default MINQUERIES of 100 sits above that across the range.
%
%   MINENTRIES (default 256) is the smallest centre array worth
%   tabulating at all. MINSAVING (default 4) is the factor by which the
%   distinct values must be fewer than the entries; an arbitrary centre
%   array with no repeats fails it and the caller takes the direct route.
%
%   Twin of the Python _tuple_values_repeat in mpt/_tensor/eval.py.
%
%   See also INTERNAL.GAUSSIANKERNELSUM, INTERNAL.WRAPPEDGAUSSIAN1D.

    if nargin < 3 || isempty(minQueries), minQueries = 100; end
    if nargin < 4 || isempty(minEntries), minEntries = 256; end
    if nargin < 5 || isempty(minSaving),  minSaving  = 4;   end
    nEntries = numel(C);
    % The query-count and size tests precede the sort: the predicate must
    % not cost what it is deciding whether to spend.
    if nQ < minQueries || nEntries < minEntries
        tf = false;
        return;
    end
    tf = numel(unique(C(:))) * minSaving <= nEntries;
end
