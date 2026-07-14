function bytes = dispatchMemBudget()
%DISPATCHMEMBUDGET  Memory budget (bytes) for the single-image feasibility guard.
%
%   BYTES = INTERNAL.DISPATCHMEMBUDGET() returns the budget against which
%   a forced single-image (centres / Bulger) working set is judged
%   feasible. Above it, auto-dispatch raises rather than risk an
%   out-of-memory materialisation.
%
%   The Python twin uses a fixed 4 GB constant. MATLAB can do better by
%   consulting internal.availableMemory: the budget is half the currently
%   available physical memory, floored at 1 GB and capped at 4 GB so it
%   never exceeds the Python guard on a memory-rich machine (keeping
%   cross-language behaviour comparable) yet never over-commits on a
%   constrained one. The 4 GB cap is the point of parity with Python's
%   _CENTRES_PROBE_MEM_BUDGET.

    MIN_BUDGET = 1 * 1024^3;   % 1 GB floor
    MAX_BUDGET = 4 * 1024^3;   % 4 GB cap (Python parity point)
    half = internal.availableMemory() / 2;
    bytes = min(max(half, MIN_BUDGET), MAX_BUDGET);
end
