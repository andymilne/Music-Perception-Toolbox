function [pc, wc, sc] = applyWindows(p, w, specs, axes, centres, gammas, sds, locates, target)
%APPLYWINDOWS  Compose each swept axis's window factor onto the target's
%   weights, then prune. The locating value is reduced as a separate array
%   (the carrier is never mutated), so the target's slot count is read from
%   the real attribute and a windowed-and-compared bundle is handled
%   correctly. Mirrors the Python seam (factor via evaluateShape, multiply
%   via multiplyWeights), with the window-factor truncation reading the
%   global mptDefaults setting.
    A = numel(p);
    wAcc = internal.normaliseWeightsToCell(w, A);
    truncSig = mptDefaults('truncationSigmas');
    Kt = size(p{target}, 1);
    for k = 1:numel(axes)
        a = axes(k);
        loc = internal.locateRow(p{a}, locates{k});      % (1, N), separate array
        delta = loc - centres(k);
        factor = internal.evaluateShape(delta, sds(k), gammas(k));
        if isfinite(truncSig)
            factor(abs(delta) > truncSig * sds(k)) = 0;
        end
        wAcc{target} = internal.multiplyWeights(wAcc{target}, factor, Kt);
    end
    [pc, wc, sc] = internal.pruneDeadCarrier(p, wAcc, specs);
end
