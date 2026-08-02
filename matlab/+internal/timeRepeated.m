function [t, nTimed] = timeRepeated(fn, opts)
%TIMEREPEATED  Median wall time of FN, after discarding warm-up runs.
%
%   T = internal.timeRepeated(FN) calls FN several times, throws the
%   first few away, and returns the median of the rest.
%
%   Timings do not settle until a few calls have been made: the first
%   pays for JIT compilation, the next for cache and allocator warm-up.
%   Averaging over three calls including those is not enough to see
%   through it, and a mean lets a single slow call dominate. So: discard
%   the first NDISCARD, time up to NMAX more, and report the median.
%
%   Slow routes are handled by a time budget rather than a fixed count.
%   A cell whose single call already takes a second does not need ten
%   repeats -- its relative noise is small and ten would make the sweep
%   unusable -- so timing stops once BUDGETSEC of measured time has
%   accumulated, provided NMIN timed runs are in hand. The warm-up runs
%   answer to the same budget, so NDISCARD is a maximum rather than a
%   fixed cost.
%
%   [T, NTIMED] = ... also returns how many runs were actually timed, so
%   a caller can report it and a reader can tell a well-sampled cell
%   from a thinly sampled one.
%
%   Options (name-value):
%     nDiscard   most warm-up runs, not timed     (default 3)
%     nMax       most timed runs                  (default 10)
%     nMin       fewest timed runs                (default 3)
%     budgetSec  stop after this much timed time  (default 2.0)
%
%   Example:
%       t = internal.timeRepeated(@() myRoute(M, r));
%
%   See also tic, toc.
    arguments
        fn (1,1) function_handle
        opts.nDiscard (1,1) double {mustBeNonnegative} = 3
        opts.nMax (1,1) double {mustBePositive} = 10
        opts.nMin (1,1) double {mustBePositive} = 3
        opts.budgetSec (1,1) double {mustBePositive} = 2.0
    end

    %   The warm-up runs answer to the same budget as the timed ones. A
    %   route whose single call already costs the whole budget has paid
    %   the JIT and allocator costs the discards exist to absorb, so
    %   three of them buy nothing and treble the cost of the cell.
    tWarm = 0;
    for i = 1:opts.nDiscard
        wTic = tic;
        fn();
        tWarm = tWarm + toc(wTic);
        if tWarm >= opts.budgetSec
            break;
        end
    end

    ts = zeros(1, opts.nMax);
    nTimed = 0;
    spent = 0;
    for i = 1:opts.nMax
        t0 = tic;
        fn();
        ts(i) = toc(t0);
        nTimed = i;
        spent = spent + ts(i);
        if nTimed >= opts.nMin && spent >= opts.budgetSec
            break;
        end
    end

    t = median(ts(1:nTimed));
end
