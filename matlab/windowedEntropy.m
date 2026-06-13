function H = windowedEntropy(pAttr, w, sigma, r, isRel, isPer, period, centres, nv)
%WINDOWEDENTROPY  Sliding pre-MAET entropy profile.
%
%   H = windowedEntropy(pAttr, w, sigma, r, isRel, isPer, period, centres, ...)
%   slides a window along one attribute axis of the *pre-MAET* carrier and
%   records, at each sweep centre, the entropy of the resulting density.
%   The carrier is a 1-by-A cell of K_a-by-N value matrices, as passed to
%   buildExpTens; the window acts on event weights via weightEvents before
%   the tensor is built.
%
%   Window. 'window' is {shape, width}: shape in [0,1] (0 Gaussian, 1
%   rectangular) or 'gaussian'/'rect'; width is the rectangular full
%   support (variance-matched for other shapes). Unlike windowedSimilarity
%   there is no query to default the width from, so a width MUST be given.
%
%   Marginalisation. Name axes in 'marginalise' (default []) to integrate
%   them out before the entropy is taken. Only the window axis may be
%   marginalised at present, and it must be an r = 1 attribute (absolute or
%   periodic), for which deletion from the carrier equals marginalisation
%   of the density; an r >= 2 axis errors.
%
%   Sweep geometry. Pass an explicit 'centres' vector, OR generative
%   'start'/'stop'/'step' (mutually exclusive). 'step' defaults to the
%   window width; 'start'/'stop' default to the data extent on the axis.
%
%   Name-value arguments:
%     'start','stop','step'  - generative sweep.
%     'window'               - {shape, width}; width required.
%                              Default {1, []} (rectangle; width must be set).
%     'method'               - entropy method passed to entropyExpTens
%                              (default 'differential').
%     'base'                 - logarithm base (default 2).
%     'targetAttr'           - 1-based attribute carrying the window
%                              factor (must differ from the window axis).
%                              Default: first non-axis attribute.
%     'windowAttr'           - 1-based window axis. Default: last attribute.
%     'marginalise'          - axes to integrate out (only the window axis
%                              supported; r = 1 required). Default [].
%     'verbose'              - logical, default false.
%
%   See also WINDOWEDSIMILARITY, WEIGHTEVENTS, BUILDEXPTENS, ENTROPYEXPTENS.

arguments
    pAttr (1,:) cell
    w
    sigma
    r
    isRel
    isPer
    period
    centres = []
    nv.start = []
    nv.stop = []
    nv.step = []
    nv.window = {1.0, []}
    nv.method (1,:) char = 'differential'
    nv.base (1,1) double = 2
    nv.targetAttr = []
    nv.windowAttr = []
    nv.marginalise = []
    nv.verbose (1,1) logical = false
end

A = numel(pAttr);
if isempty(nv.windowAttr), axisIdx = A; else, axisIdx = nv.windowAttr; end
if axisIdx < 1 || axisIdx > A
    error('windowedEntropy:badWindowAttr', ...
        'windowAttr %d out of range for %d attributes.', axisIdx, A);
end
if isempty(nv.targetAttr)
    if axisIdx ~= 1
        target = 1;
    elseif A > 1
        target = 2;
    else
        target = 1;
    end
else
    target = nv.targetAttr;
end
if target == axisIdx
    error('windowedEntropy:badTarget', ...
        'targetAttr must differ from windowAttr.');
end

wShape = local_resolve_shape(nv.window{1});
wWidth = nv.window{2};
if isempty(wWidth)
    error('windowedEntropy:noWidth', ...
        ['windowedEntropy requires an explicit window width ' ...
         '(window = {shape, width}); there is no query to default it from.']);
end

% --- marginalisation: only the window axis, and only if r = 1 ------------
marg = nv.marginalise(:).';
extra = setdiff(marg, axisIdx);
if ~isempty(extra)
    error('windowedEntropy:marginaliseUnsupported', ...
        ['windowedEntropy currently marginalises only the window axis; ' ...
         'marginalising other axes is not yet supported.']);
end
deleteAxis = any(marg == axisIdx);
if deleteAxis && r(axisIdx) ~= 1
    error('windowedEntropy:marginaliseR', ...
        ['marginalising the window axis requires it to be an r = 1 ' ...
         'attribute; for r >= 2 deletion does not equal marginalisation.']);
end

% kept-attribute specs for the density built after the window
if deleteAxis
    keep = setdiff(1:A, axisIdx);
else
    keep = 1:A;
end
sigK = sigma(keep); rK = r(keep);
relK = isRel(keep); perK = isPer(keep); pdK = period(keep);

ctr = local_resolve_centres(pAttr, axisIdx, centres, ...
    nv.start, nv.stop, nv.step, wWidth);

H = zeros(1, numel(ctr));
for i = 1:numel(ctr)
    [pw, ww] = weightEvents(pAttr, w, axisIdx, target, ctr(i), wShape, ...
        'width', wWidth, 'deleteInput', deleteAxis);
    dens = buildExpTens(pw, ww, sigK, rK, relK, perK, pdK, 'verbose', false);
    H(i) = entropyExpTens(dens, 'method', nv.method, 'base', nv.base, ...
        'verbose', false);
end
end


% ======================================================================
%  local helpers
% ======================================================================
function g = local_resolve_shape(shape)
    if ischar(shape) || isstring(shape)
        key = lower(char(shape));
        switch key
            case {'rect','rectangular','box'}, g = 1.0;
            case {'gaussian','gauss','normal'}, g = 0.0;
            otherwise
                error('windowedEntropy:badShape', ...
                    ['Unknown window shape ''%s''; pass a number in [0,1] ' ...
                     '(0 Gaussian, 1 rectangular) or ''gaussian''/''rect''.'], key);
        end
    else
        g = double(shape);
        if ~(g >= 0 && g <= 1)
            error('windowedEntropy:badShape', ...
                'Window shape must be in [0,1]; got %g.', g);
        end
    end
end

function v = local_axis_values(pAttr, axisIdx)
    M = pAttr{axisIdx};
    v = M(:);
    v = v(isfinite(v));
end

function c = local_resolve_centres(pAttr, axisIdx, centres, startV, stopV, stepV, defStep)
    if ~isempty(centres)
        if ~isempty(startV) || ~isempty(stopV) || ~isempty(stepV)
            error('windowedEntropy:sweepArgs', ...
                'Pass either centres or start/stop/step, not both.');
        end
        c = centres(:).';
        return;
    end
    v = local_axis_values(pAttr, axisIdx);
    if isempty(v)
        error('windowedEntropy:noRange', ...
            'Cannot derive a sweep range: window axis has no finite values.');
    end
    if isempty(startV), lo = min(v); else, lo = startV; end
    if isempty(stopV),  hi = max(v); else, hi = stopV;  end
    if isempty(stepV),  st = defStep; else, st = stepV; end
    if ~(st > 0)
        error('windowedEntropy:badStep', 'step must be positive.');
    end
    n = floor((hi - lo) / st + 1e-9) + 1;
    c = lo + st * (0:max(n - 1, 0));
end
