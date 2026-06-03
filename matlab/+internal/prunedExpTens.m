function dens = prunedExpTens(dens)
%PRUNEDEXPTENS  Density restricted to its live events (performance view).
%
%   DENS = internal.prunedExpTens(DENS) returns a density equivalent to
%   DENS but carrying only its live events/elements. A dead event
%   contributes nothing to any inner product or total mass, so dropping
%   such events leaves every such quantity unchanged while shrinking the
%   O(n) / O(n^2) work. Returns DENS unchanged when nothing is dead, so
%   the common (un-windowed) path pays only one scan.
%
%   Liveness rule (single definition; see LOCALWEIGHTISLIVE):
%     * SA (ExpTensDensity): an element is live iff its weight is finite
%       and of nonzero magnitude.
%     * MA (MaetDensity): an event is live iff EVERY attribute has at
%       least one finite, nonzero weight slot in that event's column. An
%       all-zero or all-NaN column kills the event (the per-attribute
%       factors multiply); a partly-zero column does not.
%
%   The source build (buildExpTens) stays faithful — it keeps every
%   event. This view is the opt-in performance form consumed by the
%   inner-product / total-mass paths (Rényi-2 entropy, cosine
%   similarity). It is the MATLAB counterpart of the Python density
%   `pruned()` method.

    if ~isstruct(dens) || ~isfield(dens, 'tag')
        return;   % not a density struct; nothing to prune
    end

    switch dens.tag
        case 'ExpTensDensity'
            live = localWeightIsLive(dens.w);
            if all(live(:))
                return;
            end
            out        = struct();
            out.tag    = 'ExpTensDensity';
            out.p      = dens.p(live);
            out.w      = dens.w(live);
            out.sigma  = dens.sigma;
            out.r      = dens.r;
            out.isRel  = dens.isRel;
            out.isPer  = dens.isPer;
            out.period = dens.period;
            if isfield(dens, 'isSym'); out.isSym = dens.isSym; end
            out.dim    = dens.dim;
            dens       = out;

        case 'MaetDensity'
            live = true(1, dens.N);
            for a = 1:dens.nAttrs
                live = live & any(localWeightIsLive(dens.w{a}), 1);
            end
            if all(live)
                return;
            end
            out              = struct();
            out.tag          = 'MaetDensity';
            out.nAttrs       = dens.nAttrs;
            out.N            = nnz(live);
            out.r            = dens.r;
            out.K            = dens.K;
            out.pAttr        = cellfun(@(P) P(:, live), dens.pAttr, ...
                                       'UniformOutput', false);
            out.w            = cellfun(@(W) W(:, live), dens.w, ...
                                       'UniformOutput', false);
            out.sigma        = dens.sigma;
            out.isRel        = dens.isRel;
            out.isPer        = dens.isPer;
            out.period       = dens.period;
            if isfield(dens, 'isSym'); out.isSym = dens.isSym; end
            out.dim          = dens.dim;
            out.dimPerAttr   = dens.dimPerAttr;
            % Per-slot nesting spec (representation B): tags are
            % row-indexed, so event (column) pruning leaves them intact.
            % Must be carried, or the rebuild flattens the attribute.
            if isfield(dens, 'nested'); out.nested = dens.nested; end
            dens             = out;

        otherwise
            % WindowedMaetDensity or unknown tag: no event-level prune.
            return;
    end
end


function tf = localWeightIsLive(w)
%LOCALWEIGHTISLIVE  Mask of weights that contribute (finite and nonzero).
%   NaN (a structurally absent slot) and 0 (present but zero-weighted,
%   e.g. hard-zeroed outside a window's truncation support) both fail.
%   This is the single definition of a "live" weight used by the
%   event-level prune above.
    tf = isfinite(w) & abs(w) > 0;
end
