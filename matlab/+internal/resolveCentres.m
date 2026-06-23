function c = resolveCentres(pAttr, axisIdx, centres, startV, stopV, stepV, defStep)
%RESOLVECENTRES  Explicit centres, or a generative start/stop/step sweep
%   (mutually exclusive). step defaults to defStep; start/stop to the data
%   extent on the window axis.
    if ~isempty(centres)
        if ~isempty(startV) || ~isempty(stopV) || ~isempty(stepV)
            error('mptWindowing:sweepArgs', 'Pass either centres or start/stop/step, not both.');
        end
        c = centres(:).'; return;
    end
    v = axisValuesLocal(pAttr, axisIdx);
    if isempty(v)
        error('mptWindowing:noRange', 'Cannot derive a sweep range: window axis has no finite values.');
    end
    if isempty(startV), lo = min(v); else, lo = startV; end
    if isempty(stopV),  hi = max(v); else, hi = stopV;  end
    if isempty(stepV),  st = defStep; else, st = stepV; end
    if ~(st > 0), error('mptWindowing:badStep', 'step must be positive.'); end
    nn = floor((hi - lo) / st + 1e-9) + 1;
    c = lo + st * (0:max(nn - 1, 0));
end

function v = axisValuesLocal(pAttr, axisIdx)
    M = pAttr{axisIdx}; v = M(:); v = v(isfinite(v));
end
