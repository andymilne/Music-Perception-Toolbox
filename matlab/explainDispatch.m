function report = explainDispatch(dens, varargin)
%EXPLAINDISPATCH  Report how a call would be routed, and why.
%
%   EXPLAINDISPATCH(DENS, NQ) explains an evaluation of DENS at NQ query
%   points. EXPLAINDISPATCH(DENS_X, DENS_Y) explains a cosine similarity
%   between two densities. With no output argument the report is
%   printed; with one, it is returned as a struct.
%
%   Name-value arguments:
%     'method'            'auto' (default), or a forced route.
%     'truncationSigmas'  accuracy width; [] takes the default.
%
%   The toolbox chooses between two ways of computing the same quantity
%   --- materialising the joint tuple set, or the Mobius decomposition
%   that never does --- and the choice follows three tests applied in
%   order: is the route feasible, is it accurate enough for the accuracy
%   asked for, and if more than one survives, which is fastest.
%
%   That rule is easy to state and hard to inspect. The selectors return
%   a short reason but not the quantities behind it: the predicted time
%   for each route, the accuracy floor in force, the sigma/P limit that
%   follows from it, or where the call sits relative to any of them.
%   This reports all of it without running the call.
%
%   It is a diagnostic, not a decision: it calls the same selector the
%   toolbox uses, so what it reports is what would happen rather than a
%   reconstruction of it.
%
%   Example:
%       d = buildExpTens(p, w, 60, 3, true, true, 1200);
%       explainDispatch(d, 200);
%
%   Twin of the Python mpt.explain_dispatch.
%
%   See also INTERNAL.SELECTMAEVAL, INTERNAL.RELPERSIGMAOVERPTHRESHOLD.

    nQ = 200;
    other = [];
    method = 'auto';
    truncationSigmas = [];

    args = varargin;
    if ~isempty(args) && isnumeric(args{1}) && isscalar(args{1})
        nQ = args{1};  args(1) = [];
    elseif ~isempty(args) && isstruct(args{1})
        other = args{1};  args(1) = [];
    end
    for ii = 1:2:numel(args)
        switch lower(args{ii})
            case 'method',           method = args{ii + 1};
            case 'truncationsigmas', truncationSigmas = args{ii + 1};
            otherwise
                error('explainDispatch:badArg', ...
                      'Unknown argument ''%s''.', args{ii});
        end
    end

    ts    = internal.accuracyFloor('resolve', truncationSigmas);
    floorV = internal.truncationFloor(truncationSigmas);
    [sop, limit, limitSetBy] = localPeriodicity(dens, truncationSigmas);

    if ~isempty(other)
        report = localCosine(dens, other, ts, floorV, sop, limit, ...
                             limitSetBy, method);
    else
        report = localEval(dens, nQ, ts, floorV, sop, limit, ...
                           limitSetBy, method);
    end

    if nargout == 0
        localPrint(report);
        clear report;
    end
end


function [sop, limit, setBy] = localPeriodicity(dens, truncationSigmas)
%LOCALPERIODICITY  Largest sigma/P over relative-periodic attributes,
%   with the limit in force and which test set it.
    sop = [];
    sig = double(dens.sigma(:).');
    per = double(dens.period(:).');
    isRel = logical(dens.isRel(:).');
    isPer = logical(dens.isPer(:).');
    for a = 1:numel(sig)
        if a <= numel(per) && per(a) > 0 && isPer(a) && isRel(a)
            sop = max([sop, sig(a) / per(a)]);
        end
    end
    limit = internal.relPerSigmaOverPThreshold(truncationSigmas);
    % Which of the two tests bound it. The accuracy limit moves with
    % truncationSigmas; the positive-definiteness ceiling does not, so
    % saying which one applies tells the reader whether asking for less
    % accuracy would move it.
    PD_CEILING = 0.05;
    if limit >= PD_CEILING
        setBy = 'positive-definiteness';
    else
        setBy = 'accuracy';
    end
end


function report = localEval(dens, nQ, ts, floorV, sop, limit, setBy, method)
    [chosen, reason, centresMs, mobiusMs] = ...
        internal.selectMaEval(dens, nQ, false, ts);
    if ~strcmpi(method, 'auto')
        chosen = method;  reason = 'user override';
    end
    priced = ~isempty(strfind(reason, 'cost model')); %#ok<STREMP>
    report = struct( ...
        'call', 'evalExpTens', ...
        'shape', sprintf('%s; nQ=%d', localShape(dens), nQ), ...
        'truncationSigmas', ts, 'floor', floorV, ...
        'sigmaOverP', sop, 'sigmaOverPLimit', limit, ...
        'limitSetBy', setBy, 'measure', localMeasure(sop, chosen), ...
        'routeNames', {{'centres', 'mobius'}}, ...
        'routeMs', [centresMs, mobiusMs], ...
        'chosen', chosen, 'decidedBy', reason, 'priced', priced);
end


function report = localCosine(dX, dY, ts, floorV, sop, limit, setBy, method)
    rVec = double(dX.r(:).');
    kVec = double(dX.K(:).');
    kVecY = double(dY.K(:).');
    isRel = logical(dX.isRel(:).');
    isPer = logical(dX.isPer(:).');
    nX = double(dX.N);  nY = double(dY.N);
    if isempty(sop), sopArg = 0; else, sopArg = sop; end
    [chosen, pwMs, orbMs] = internal.selectMaInnerProductMethod( ...
        rVec, kVec, numel(rVec), nX, nY, any(isPer), ...
        any(isRel & ~isPer), any(isRel & isPer), sopArg, method, ...
        false, isRel, [], kVecY, {}, ts);
    priced = all(isfinite([pwMs, orbMs]));
    report = struct( ...
        'call', 'cosSimExpTens', ...
        'shape', sprintf('%s against K=%s', localShape(dX), ...
                         mat2str(kVecY)), ...
        'truncationSigmas', ts, 'floor', floorV, ...
        'sigmaOverP', sop, 'sigmaOverPLimit', limit, ...
        'limitSetBy', setBy, 'measure', localMeasure(sop, chosen), ...
        'routeNames', {{'bulger', 'mobius'}}, ...
        'routeMs', [pwMs, orbMs], ...
        'chosen', chosen, ...
        'decidedBy', localTernary(priced, 'cost model', ...
                                  'structural rule'), ...
        'priced', priced);
end


function s = localShape(dens)
    rVec = double(dens.r(:).');
    kVec = double(dens.K(:).');
    isRel = logical(dens.isRel(:).');
    isPer = logical(dens.isPer(:).');
    parts = cell(1, numel(rVec));
    for a = 1:numel(rVec)
        if isRel(a), rel = 'rel'; else, rel = 'abs'; end
        if isPer(a), pe  = 'per'; else, pe  = 'np';  end
        parts{a} = sprintf('r=%d K=%d %s%s', rVec(a), kVec(a), rel, pe);
    end
    s = sprintf('%d attribute(s), %d event(s); %s', numel(rVec), ...
                double(dens.N), strjoin(parts, '; '));
end


function m = localMeasure(sop, chosen)
    if isempty(sop)
        m = '';
    elseif strcmp(chosen, 'mobius')
        m = 'transposition average (the definition)';
    else
        m = 'wrapped-difference approximation';
    end
end


function out = localTernary(cond, a, b)
    if cond, out = a; else, out = b; end
end


function localPrint(rep)
    fprintf('%s: %s\n', rep.call, rep.shape);
    fprintf('%s\n', repmat('-', 1, 62));
    fprintf('accuracy      truncationSigmas = %g  ->  floor %.2e\n', ...
            rep.truncationSigmas, rep.floor);
    if ~isempty(rep.sigmaOverP)
        if rep.sigmaOverP <= rep.sigmaOverPLimit
            within = 'within';
        else
            within = 'beyond';
        end
        fprintf('periodicity   sigma/P = %.4f, %s the limit of %g\n', ...
                rep.sigmaOverP, within, rep.sigmaOverPLimit);
        fprintf('              the limit is set by %s\n', rep.limitSetBy);
    end
    if ~isempty(rep.measure)
        fprintf('measure       %s\n', rep.measure);
    end
    fprintf('\n%-12s%-10s%12s   %s\n', 'route', 'feasible', ...
            'predicted', 'why');
    for a = 1:numel(rep.routeNames)
        if isnan(rep.routeMs(a))
            pred = '--';
        else
            pred = sprintf('%.3f ms', rep.routeMs(a));
        end
        if strcmp(rep.routeNames{a}, rep.chosen)
            mark = '*';
            why  = rep.decidedBy;
        else
            mark = ' ';
            why  = localTernary(rep.priced, 'priced', 'not selected');
        end
        fprintf('%s%-11s%-10s%12s   %s\n', mark, rep.routeNames{a}, ...
                'true', pred, why);
    end
    fprintf('\nchosen        %s  (%s)\n', rep.chosen, rep.decidedBy);
end
