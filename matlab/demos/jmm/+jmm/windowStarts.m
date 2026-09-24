function [idxs, centres] = windowStarts(mus, lead)
%WINDOWSTARTS  The window start times a sweep visits, and whose mu they are.
%
%   [idxs, centres] = jmm.windowStarts(mus, lead)
%
%   A window resolving at mus(k) starts LEAD beats before it, and must lie
%   inside the piece. Returns the positions of the mus that qualify and the
%   start times of their windows.
%
%   See also JMM.BOUNDCONTEXT, WINDOWEDSIMILARITY.
    S = jmm.bwvWindowState();
    keep = mus - lead >= S.T0 - 1e-9 & mus + 1.0 <= S.T1 + 1e-9;
    idxs = find(keep);
    centres = mus(keep) - lead;
end
