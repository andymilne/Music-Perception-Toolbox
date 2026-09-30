function [idxs, at] = windowsInPiece(mus, L)
%WINDOWSINPIECE  The resolution moments a sweep visits, and their positions.
%
%   [idxs, at] = jmm.windowsInPiece(mus, L)
%
%   A window of L beats resolving at mus(k) spans [mus(k) - (L - 1),
%   mus(k) + 1) and must lie inside the piece. Returns the positions of the
%   mus that qualify and the sweep values at which the window is aligned
%   to read them. Binding is end-aligned, so each bound window is timed at
%   its resolution beat and the sweep values are the qualifying mus
%   themselves.
%
%   See also JMM.BOUNDCONTEXT, SWEPTSIMILARITY.
    S = jmm.bwvWindowState();
    keep = mus - (L - 1) >= S.T0 - 1e-9 & mus + 1.0 <= S.T1 + 1e-9;
    idxs = find(keep);
    at = mus(keep);
end
