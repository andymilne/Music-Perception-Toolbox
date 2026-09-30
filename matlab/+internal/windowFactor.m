function factor = windowFactor(loc, at, win)
%WINDOWFACTOR  Per-event factor of window WIN aligned at AT, over the
%   located values LOC (1 x N): the event weighting of weightEvents, through
%   the same implementation (internal.weightFactor). WIN is a struct with
%   fields prof (internal.resolveProfile), closed, isPer, period, and
%   optionally shift, the displacement of the window's centre from its
%   reference value (a rectangle aligned at its start or end). Twin of the
%   Python _window_factor.
    if isfield(win, 'shift'), at = at + win.shift; end
    factor = internal.weightFactor(loc, at, win.prof, win.isPer, ...
        win.period, win.closed);
end
