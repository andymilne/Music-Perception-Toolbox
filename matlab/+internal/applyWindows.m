function [pc, wc, sc] = applyWindows(p, w, specs, axes, centres, gammas, sds, locates, target)
%APPLYWINDOWS  Compose each swept axis's window factor onto the target's
%   weights, then prune. The locating value is reduced as a separate array
%   (the inputs are never mutated), so the target's value count is read from
%   the real attribute and a windowed-and-compared bundle is handled
%   correctly. Mirrors the Python seam (factor via evaluateShape, multiply
%   via multiplyWeights). Per the truncation contract, the window-factor
%   cutoff is resolved through internal.accuracyFloor so a user Inf
%   resolves to the finite accuracy-floor width (~7.43 sigma, the 1e-12
%   floor); truncation always applies at least at that width.
    A = numel(p);
    wAcc = internal.normaliseWeightsToCell(w, A);
    truncSig = internal.accuracyFloor('resolve', ...
        mptDefaults('truncationSigmas'));
    Kt = size(p{target}, 1);
    for k = 1:numel(axes)
        a = axes(k);
        loc = internal.locateRow(p{a}, locates{k});      % (1, N), separate array
        delta = loc - centres(k);
        factor = internal.evaluateShape(delta, sds(k), gammas(k));
        factor(abs(delta) > truncSig * sds(k)) = 0;
        wAcc{target} = internal.multiplyWeights(wAcc{target}, factor, Kt);
    end
    [pc, wc, sc] = internal.pruneDeadEvents(p, wAcc, specs);
end
