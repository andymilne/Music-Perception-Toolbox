function [pc, wc, sc] = applyWindows(p, w, specs, axes, at, wins, locates, target)
%APPLYWINDOWS  Compose each window's factor onto the target's weights, then
%   prune. The window WINS{k} on attribute axes(k) is aligned at at(k): its
%   reference value (displacement zero) is placed there, and each event is
%   weighted by the window's value at its located value's displacement from
%   it (internal.windowFactor, the implementation weightEvents uses). The
%   located values are reduced as a separate array (the inputs are never
%   mutated), so the target's value count is read from the real attribute
%   and a windowed-and-compared bundle is handled correctly. Mirrors the
%   Python _apply_windows.
    A = numel(p);
    wAcc = internal.normaliseWeightsToCell(w, A);
    Kt = size(p{target}, 1);
    for k = 1:numel(axes)
        a = axes(k);
        loc = internal.locateRow(p{a}, locates{k});      % (1, N), separate array
        factor = internal.windowFactor(loc, at(k), wins{k});
        wAcc{target} = internal.multiplyWeights(wAcc{target}, factor, Kt);
    end
    [pc, wc, sc] = internal.pruneDeadEvents(p, wAcc, specs);
end
