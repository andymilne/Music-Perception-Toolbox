function out = windowedSimilarity(pContext, wContext, pQuery, wQuery, ...
        sigma, r, isRel, isPer, period, centres, nv)
%WINDOWEDSIMILARITY  Sliding pre-MAET similarity profile (cross-correlation).
%
%   out = windowedSimilarity(pContext, wContext, pQuery, wQuery, ...
%             sigma, r, isRel, isPer, period, centres, ...)
%   slides a window along one attribute axis of the *pre-MAET* context
%   carrier and scores it, at each sweep centre, against the query. Both
%   operands are carriers (1-by-A cells of K_a-by-N value matrices), as
%   passed to buildExpTens; the window acts on event weights via
%   weightEvents before any tensor is built. This is the pre-MAET
%   companion of windowedTensorSimilarity (which windows the built tensor).
%
%   At each sweep centre the context is placed by 'contextWindow' and the
%   query by 'queryWindow'. With the defaults the context is windowed by a
%   rectangle of the query's extent and the query is translated to the same
%   centre (the locked template sweep). A window of [] translates that
%   operand whole; {shape, width} windows it (shape in [0,1], 0 Gaussian,
%   1 rectangular, or 'gaussian'/'rect'; width the rectangular full
%   support, variance-matched for other shapes).
%
%   Sweep geometry. Pass an explicit 'centres' vector, OR the generative
%   'start'/'stop'/'step' (mutually exclusive; both errors). 'step'
%   defaults to the context-window width; 'start'/'stop' default to the
%   data extent on the window axis.
%
%   Decoupling and output shape. The output shape follows 'queryCentres',
%   with the context broadcast along its trailing axis:
%     queryCentres = []            -> locked, 1-by-A row (query at each
%                                     context centre).
%     queryCentres a length-A vec  -> element-wise paired, 1-by-A row.
%     queryCentres an A-by-T matrix-> grid out(a,t) with the context at
%                                     ctxCentres(a) and the query at
%                                     queryCentres(a,t) (e.g. a lag sweep:
%                                     queryCentres = ctx(:) - tau).
%
%   Name-value arguments:
%     'start','stop','step'  - generative sweep (see above).
%     'queryCentres'         - [] | length-A vector | A-by-T matrix.
%     'contextWindow'        - {shape, width}; [] shape translates the
%                              context; width [] defaults to query extent.
%                              Default {1, []} (rectangle of query extent).
%     'queryWindow'          - [] (translate the query) | {shape, width}.
%                              Default [].
%     'targetAttr'           - 1-based attribute carrying the window
%                              factor (must differ from the window axis).
%                              Default: first non-axis attribute.
%     'normalize'            - 'oneSidedDenom' (default) | 'cosine'.
%     'windowAttr'           - 1-based window axis. Default: last attribute.
%     'verbose'              - logical, default false.
%
%   The query-translation path batches all trailing-axis placements into a
%   single translateAttributes call and one cosSimExpTens scalar-vs-list
%   call per context centre, matching the Python implementation.
%
%   See also WINDOWEDENTROPY, WINDOWEDTENSORSIMILARITY, WEIGHTEVENTS,
%   TRANSLATEATTRIBUTES, COSSIMEXPTENS.

arguments
    pContext (1,:) cell
    wContext
    pQuery   (1,:) cell
    wQuery
    sigma
    r
    isRel
    isPer
    period
    centres = []
    nv.start = []
    nv.stop = []
    nv.step = []
    nv.queryCentres = []
    nv.contextWindow = {1.0, []}
    nv.queryWindow = []
    nv.targetAttr = []
    nv.normalize (1,:) char = 'oneSidedDenom'
    nv.windowAttr = []
    nv.verbose (1,1) logical = false
end

A = numel(pContext);
if isempty(nv.windowAttr), axisIdx = A; else, axisIdx = nv.windowAttr; end
if axisIdx < 1 || axisIdx > A
    error('windowedSimilarity:badWindowAttr', ...
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
    error('windowedSimilarity:badTarget', ...
        'targetAttr must differ from windowAttr.');
end

% --- context window spec -------------------------------------------------
cwShapeRaw = nv.contextWindow{1};
cwWidth    = nv.contextWindow{2};
translateContext = isempty(cwShapeRaw);
if ~translateContext
    cwShape = local_resolve_shape(cwShapeRaw);
end
if isempty(cwWidth)
    qv = local_axis_values(pQuery, axisIdx);
    if isempty(qv), cwWidth = 0; else, cwWidth = max(qv) - min(qv); end
end

% --- sweep centres (context, leading axis) -------------------------------
ctxCentres = local_resolve_centres(pContext, axisIdx, centres, ...
    nv.start, nv.stop, nv.step, cwWidth);
nA = numel(ctxCentres);

% --- per-row query centres and output shape ------------------------------
if isempty(nv.queryCentres)
    qMat = ctxCentres(:);          % nA x 1, locked
    twoD = false;
else
    qc = nv.queryCentres;
    if isvector(qc)
        if numel(qc) ~= nA
            error('windowedSimilarity:badQueryCentres', ...
                ['1-D queryCentres must match the context length A=%d; ' ...
                 'got %d. For a grid pass an A-by-T matrix.'], nA, numel(qc));
        end
        qMat = qc(:);
        twoD = false;
    else
        if size(qc, 1) ~= nA
            error('windowedSimilarity:badQueryCentres', ...
                ['2-D queryCentres must have first dimension == context ' ...
                 'length A=%d; got %dx%d.'], nA, size(qc,1), size(qc,2));
        end
        qMat = qc;
        twoD = true;
    end
end
T = size(qMat, 2);

% --- query placement set-up ----------------------------------------------
translateQuery = isempty(nv.queryWindow);
if translateQuery
    muQ = mean(pQuery{axisIdx}(:));
else
    qwShape = local_resolve_shape(nv.queryWindow{1});
    qwWidth = nv.queryWindow{2};
end
if translateContext
    muC = mean(pContext{axisIdx}(:));
end

out = zeros(nA, T);
for a = 1:nA
    % place the context once for this row
    if translateContext
        offs = cell(1, A);
        offs{axisIdx} = ctxCentres(a) - muC;
        [pc, wc] = translateAttributes(pContext, wContext, offs);
    else
        [pc, wc] = weightEvents(pContext, wContext, axisIdx, target, ...
            ctxCentres(a), cwShape, 'width', cwWidth, 'deleteInput', false);
    end

    row = qMat(a, :);              % 1 x T query centres for this context

    if translateQuery
        % one translate produces all T shifted copies; one cosSim scores
        % the single context against the batch.
        offs = cell(1, A);
        offs{axisIdx} = row - muQ;             % 1 x T row -> M copies
        qSwept = translateAttributes(pQuery, wQuery, offs);
        if T == 1
            out(a, 1) = cosSimExpTens(pc, wc, qSwept, wQuery, ...
                sigma, r, isRel, isPer, period, ...
                'normalize', nv.normalize, 'verbose', false);
        else
            sCell = cosSimExpTens(pc, wc, qSwept, wQuery, ...
                sigma, r, isRel, isPer, period, ...
                'normalize', nv.normalize, 'verbose', false);
            out(a, :) = cell2mat(sCell(:).');
        end
    else
        for t = 1:T
            [pq, wq] = weightEvents(pQuery, wQuery, axisIdx, target, ...
                row(t), qwShape, 'width', qwWidth, 'deleteInput', false);
            out(a, t) = cosSimExpTens(pc, wc, pq, wq, ...
                sigma, r, isRel, isPer, period, ...
                'normalize', nv.normalize, 'verbose', false);
        end
    end
end

if ~twoD
    out = out(:).';                % 1 x nA row for locked / paired sweeps
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
                error('windowedSimilarity:badShape', ...
                    ['Unknown window shape ''%s''; pass a number in [0,1] ' ...
                     '(0 Gaussian, 1 rectangular) or ''gaussian''/''rect''.'], key);
        end
    else
        g = double(shape);
        if ~(g >= 0 && g <= 1)
            error('windowedSimilarity:badShape', ...
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
            error('windowedSimilarity:sweepArgs', ...
                'Pass either centres or start/stop/step, not both.');
        end
        c = centres(:).';
        return;
    end
    v = local_axis_values(pAttr, axisIdx);
    if isempty(v)
        error('windowedSimilarity:noRange', ...
            'Cannot derive a sweep range: window axis has no finite values.');
    end
    if isempty(startV), lo = min(v); else, lo = startV; end
    if isempty(stopV),  hi = max(v); else, hi = stopV;  end
    if isempty(stepV),  st = defStep; else, st = stepV; end
    if ~(st > 0)
        error('windowedSimilarity:badStep', 'step must be positive.');
    end
    n = floor((hi - lo) / st + 1e-9) + 1;
    c = lo + st * (0:max(n - 1, 0));
end
