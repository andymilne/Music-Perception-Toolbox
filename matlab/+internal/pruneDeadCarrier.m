% =========================================================================
%  pruneDeadCarrier — drop window-zeroed events from a carrier (internal)
% =========================================================================

function [pAttr, w, specs] = pruneDeadCarrier(pAttr, w, specs)
%PRUNEDEADCARRIER  Drop events the window hard-zeroed, before the build.
%
%   [pAttr, w, specs] = internal.pruneDeadCarrier(pAttr, w, specs)
%
%   A windowed carrier carries out-of-window / beyond-truncation events
%   at weight zero (weightEvents writes its factor as such). Those events
%   contribute nothing to any inner product or to the density an entropy
%   integrates, so dropping them here -- at the windowing seam, before
%   buildExpTens runs its eager feasibility scan and r-ad enumeration over
%   every column -- is exact and saves the bulk of a sliding sweep's cost,
%   without touching the build itself or the density-level prunedExpTens
%   path. The liveness rule is the shared internal.weightIsLive (also used
%   by internal.prunedExpTens): an event is live iff every weighted
%   attribute has a finite, nonzero slot in its column. Returned unchanged when nothing is dead (the un-windowed
%   common case pays only a mask scan) or when everything is dead (an
%   empty window keeps its existing path). The specs (per-slot tags) are
%   untouched: event-column pruning leaves the slot layout intact.

    if isempty(pAttr) || ~iscell(w)
        return
    end
    N = size(pAttr{1}, 2);
    if N == 0
        return
    end

    live = true(1, N);
    for a = 1:numel(w)
        Wa = w{a};
        if isempty(Wa)
            continue
        end
        if ~ismatrix(Wa) || size(Wa, 2) ~= N
            continue    % per-slot / scalar: cannot kill an event on its own
        end
        live = live & any(internal.weightIsLive(Wa), 1);
    end

    nLive = nnz(live);
    if nLive == N || nLive == 0
        return          % nothing dead, or everything dead -> leave as is
    end

    keep = find(live);
    for a = 1:numel(pAttr)
        pAttr{a} = pAttr{a}(:, keep);
    end
    for a = 1:numel(w)
        Wa = w{a};
        if isempty(Wa)
            continue
        end
        if ismatrix(Wa) && size(Wa, 2) == N
            w{a} = Wa(:, keep);
        end
    end
end
