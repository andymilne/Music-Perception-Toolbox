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
%   Window axis role. 'dropWindowAttr' is REQUIRED and fixes the structural
%   role of the window axis: true removes it from the density (placement
%   coordinate only; it must then be an r = 1 attribute, for which deletion
%   equals marginalisation), false retains it as a compared dimension whose
%   entropy is taken. 'marginalise' (default []) is the separate, general
%   operation of integrating a *retained* axis out of the density before the
%   entropy is taken; it is not yet implemented, and naming the dropped axis
%   in it is an error.
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
%     'dropWindowAttr'       - logical, REQUIRED (no default). true drops the
%                              window axis from the density (r = 1 only);
%                              false retains it as a compared dimension.
%     'marginalise'          - axes to integrate out of a retained density
%                              (general operation; not yet implemented).
%                              Default [].
%     'specs'                - [] (flat carrier) or a 1-by-A cell of carrier
%                              specs (as returned by bindEvents). When given,
%                              the per-attribute geometry is read from specs
%                              and the positional r/isRel supply only the
%                              window-axis order used by the deletion
%                              guard; sigma/isPer/period still supply kernel
%                              widths and periodicity. Default [].
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
    nv.specs = []
    nv.verbose (1,1) logical = false
    nv.dropWindowAttr (1,1) logical
end

% dropWindowAttr has no default: omitting it errors below when first read,
% matching weightEvents. The window axis is either dropped (placement only,
% removed from the density) or retained as a compared dimension.
dropWindowAttr = nv.dropWindowAttr;

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

% --- drop vs marginalise -------------------------------------------------
% Dropping equals marginalisation only at r = 1.
if dropWindowAttr && r(axisIdx) ~= 1
    error('windowedEntropy:dropWindowAttrR', ...
        ['dropWindowAttr=true requires the window axis to be an r = 1 ' ...
         'attribute; for r >= 2 deletion does not equal marginalisation.']);
end

% 'marginalise' is the separate, general integrate-out operation over a
% retained axis (not yet implemented). A dropped axis is already gone, so it
% cannot also be marginalised.
marg = nv.marginalise(:).';
if dropWindowAttr && any(marg == axisIdx)
    error('windowedEntropy:dropAndMarginalise', ...
        ['the window axis is dropped (dropWindowAttr=true), so it cannot ' ...
         'also appear in marginalise.']);
end
if ~isempty(marg)
    error('windowedEntropy:marginaliseNotImplemented', ...
        ['marginalise (integrating a retained axis out of the density) is ' ...
         'not yet implemented.']);
end

% kept-attribute specs for the density built after the window
if dropWindowAttr
    keep = setdiff(1:A, axisIdx);
else
    keep = 1:A;
end
sigK = sigma(keep); rK = r(keep);
relK = isRel(keep); perK = isPer(keep); pdK = period(keep);

ctr = local_resolve_centres(pAttr, axisIdx, centres, ...
    nv.start, nv.stop, nv.step, wWidth);

nested = ~isempty(nv.specs);
H = zeros(1, numel(ctr));
for i = 1:numel(ctr)
    [pw, ww, sw] = weightEvents(pAttr, w, axisIdx, target, ctr(i), wShape, ...
        'width', wWidth, 'dropInputAttr', dropWindowAttr, 'specs', nv.specs);
    % Drop events the window hard-zeroed before the (heavy) build.
    [pw, ww, sw] = internal.pruneDeadCarrier(pw, ww, sw);
    if nested
        % geometry rides in the (axis-pruned) specs; r/isRel unused
        dens = buildExpTens(pw, ww, 'sigma', sigK, 'isPer', perK, ...
            'period', pdK, 'specs', sw, 'verbose', false);
    else
        dens = buildExpTens(pw, ww, sigK, rK, relK, perK, pdK, 'verbose', false);
    end
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
