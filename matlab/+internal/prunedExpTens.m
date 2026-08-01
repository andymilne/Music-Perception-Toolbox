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
%   Liveness rule (single definition; see internal.weightIsLive):
%     * Single-multiset (MaetDensity at A = N = 1): an element is live
%       iff its weight is finite and of nonzero magnitude. Dead elements
%       are dropped from the multiset (element-level prune).
%     * General MA (MaetDensity): an event is live iff EVERY attribute
%       has at least one finite, nonzero weight in that event's
%       column. An all-zero or all-NaN column kills the event (the
%       per-attribute factors multiply); a partly-zero column does not.
%
%   The source build (buildExpTens) stays faithful --- it keeps every
%   event. This reduced form is the opt-in performance density consumed
%   by the inner-product / total-mass paths (Rényi-2 entropy, cosine
%   similarity). It is the MATLAB counterpart of the Python density
%   `pruned()` method (and, for the single-multiset corner, of
%   _SingleMultisetView.pruned()).

    if ~isstruct(dens) || ~isfield(dens, 'tag')
        return;   % not a density struct; nothing to prune
    end

    switch dens.tag
        case 'MaetDensity'
            if internal.isSingleMultiset(dens)
                % Single-multiset corner (A = N = 1): the one flat
                % attribute's values are a single multiset. Prune at the
                % value level --- drop zero-/NaN-weight values (the build
                % collapse has already pooled any r = 1 events into this
                % one, so there is nothing left to pool here). Mirrors
                % Python _SingleMultisetView.pruned() and the value-level
                % pruning rule. No-op under a matrix-valued kernel
                % (dropping a value would change the single tuple's
                % dimension; the dead value already zeroes its weight).
                w1 = dens.w{1};
                live = internal.weightIsLive(w1);
                live = live(:).';
                % Nothing dead: the density is already its own pruned
                % form. Otherwise rebuild on the live subset, dropping
                % zero-/NaN-weight values (element-level pruning). The
                % build collapse guarantees the single-multiset corner is
                % N = 1 here, so a pooled build and a pre-pruned build
                % reduce to the identical multiset (auto-prune ==
                % manual-prune, bit for bit).
                if all(live)
                    return;
                end
                if internal.densityHasKernelCov(dens)
                    return;
                end
                p1 = dens.pAttr{1};
                symArg = {};
                if isfield(dens, 'isSym') && ~isempty(dens.isSym)
                    symArg = {dens.isSym(1)};
                end
                wrapArg = {};
                if isfield(dens, 'wrap') && ~isempty(dens.wrap)
                    wrapArg = {'wrap', dens.wrap{1}};
                end
                dens = buildExpTens( ...
                    p1(live), w1(live), dens.sigma(1), dens.r(1), ...
                    dens.isRel(1), dens.isPer(1), dens.period(1), ...
                    symArg{:}, wrapArg{:}, 'lazy', true, 'verbose', false);
                return;
            end

            live = true(1, dens.N);
            for a = 1:dens.nAttrs
                live = live & any(internal.weightIsLive(dens.w{a}), 1);
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
            % Per-value nesting spec (representation B): tags are
            % row-indexed, so event (column) pruning leaves them intact.
            % Must be carried, or the rebuild flattens the attribute.
            if isfield(dens, 'nested'); out.nested = dens.nested; end
            if isfield(dens, 'names'); out.names = dens.names; end
            % Per-attribute wrap: must be carried, or a density built with
            % 'single-image' silently reverts to full-image, changing the
            % measure rather than the speed. Only shows up above the
            % sigma/period threshold, where the two forms diverge.
            if isfield(dens, 'wrap'); out.wrap = dens.wrap; end
            if isfield(dens, 'kernelCov')
                out.kernelCov = dens.kernelCov;
                out.kernelChol = dens.kernelChol;
            end
            dens             = out;

        otherwise
            % WindowedMaetDensity or unknown tag: no event-level prune.
            return;
    end
end
