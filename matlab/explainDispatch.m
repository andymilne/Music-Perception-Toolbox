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
                             limitSetBy, method, truncationSigmas);
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
    [chosen, reason] = internal.selectMaEval(dens, nQ, ts);
    % Price both routes unconditionally, not just where the selector
    % consulted the cost model. A hard rule returns before pricing, so the
    % selector's own CENTRESMS/MOBIUSMS are NaN there --- but the reader
    % wants to know what the rule cost or saved. Guarded: this is a
    % report, so a cost model that cannot price this shape must leave the
    % column blank rather than fail the call. Mirrors the Python
    % explain_dispatch, which calls _ma_eval_costs_ms in a try/except.
    try
        [centresMs, mobiusMs] = internal.maEvalCostsMs(dens, nQ);
        [cN, mN] = internal.nestedEvalCostsMs(dens, nQ);
        centresMs = centresMs + cN;
        mobiusMs = mobiusMs + mN;
    catch
        centresMs = NaN;  mobiusMs = NaN;
    end
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


function report = localCosine(dX, dY, ts, floorV, sop, limit, setBy, ...
                              method, truncationSigmas)
%LOCALCOSINE  Route report for a flat cosine, built from the call's own
%   inputs.
%
%   The densities are pruned and the empty-operand rule applied first,
%   as cosSimExpTens's localCosSimMA does; the flat selector then
%   receives exactly the inputs the call gives it --- the wrap vector,
%   the per-attribute grid node counts, and the memo flags read from
%   the structs' self-IP caches (INTERNAL.FLATSELECTORINPUTS) --- and
%   the ordered-attribute override is applied after it. The report
%   therefore names the route the call takes, including the rel-per
%   wrap rule above the sigma/P threshold, which decides without
%   pricing. Twin of the Python explain._explain_cosine.
    if localAnyNested(dX) || localAnyNested(dY)
        report = localCosineNested(dX, dY, ts, floorV, sop, limit, ...
                                   setBy, method);
        return;
    end
    dX = internal.prunedExpTens(dX);
    dY = internal.prunedExpTens(dY);
    kVecY = double(dY.K(:).');
    shape = sprintf('%s against K=%s', localShape(dX), mat2str(kVecY));
    if dX.N == 0 || dY.N == 0
        % No events to overlap: the call returns 0 before any selector
        % runs, so there is no route to report.
        report = struct( ...
            'call', 'cosSimExpTens', 'shape', shape, ...
            'truncationSigmas', ts, 'floor', floorV, ...
            'sigmaOverP', sop, 'sigmaOverPLimit', limit, ...
            'limitSetBy', setBy, 'measure', '', ...
            'routeNames', {{'bulger', 'mobius'}}, ...
            'routeMs', [NaN, NaN], 'chosen', '', ...
            'decidedBy', 'empty operand (the similarity is 0 without a route)', ...
            'priced', false);
        return;
    end
    % The raw per-call width goes in, as on the real call; the selector
    % resolves it where it needs a number (ts is the resolved value the
    % report prints).
    [selIn, orderedAny, ~] = internal.flatSelectorInputs( ...
        dX, dY, 'cosine', truncationSigmas);
    [chosen, pwMs, orbMs] = internal.selectMaInnerProductMethod( ...
        selIn.rVec, selIn.kVec, selIn.A, selIn.Nx, selIn.Ny, ...
        selIn.anyPer, selIn.anyRelNonper, selIn.anyRelPer, ...
        selIn.sigmaOverPMax, method, false, selIn.relVec, selIn.nuVec, ...
        selIn.kVecY, selIn.wrapVec, selIn.truncationSigmas, ...
        selIn.skipXX, selIn.skipYY, selIn.symVec, ...
        selIn.guardForcedBulger, selIn.perVec);
    priced = all(isfinite([pwMs, orbMs]));
    if orderedAny
        % An ordered ([sym]=0) attribute has no orbit, so the call takes
        % Bulger's method whatever the selector said (and whatever the
        % user asked for).
        chosen = 'bulger';
        decidedBy = 'ordered ([sym]=0) attribute (no orbit to collapse)';
    elseif ~strcmp(method, 'auto')
        decidedBy = 'user method';
    elseif priced
        decidedBy = 'cost model';
    else
        decidedBy = 'structural rule';
    end
    report = struct( ...
        'call', 'cosSimExpTens', ...
        'shape', shape, ...
        'truncationSigmas', ts, 'floor', floorV, ...
        'sigmaOverP', sop, 'sigmaOverPLimit', limit, ...
        'limitSetBy', setBy, 'measure', localMeasure(sop, chosen), ...
        'routeNames', {{'bulger', 'mobius'}}, ...
        'routeMs', [pwMs, orbMs], ...
        'chosen', chosen, ...
        'decidedBy', decidedBy, ...
        'priced', priced);
end


function tf = localAnyNested(dens)
%LOCALANYNESTED  Whether the density carries a nested attribute.
    tf = isfield(dens, 'nested') && iscell(dens.nested) ...
        && any(~cellfun(@isempty, dens.nested));
end


function report = localCosineNested(dX, dY, ts, floorV, sop, limit, ...
                                    setBy, method)
%LOCALCOSINENESTED  Route report for a cosine on a nested density.
%
%   A nested attribute never reaches the flat orbit (Möbius) entry
%   point, so the two routes the flat report prices are not the two on
%   offer here. The candidates are the hierarchical contraction plan ---
%   which itself picks a route per nested attribute --- and the
%   joint-tuple enumeration.
%
%   Both are priced, as the flat report prices Bulger's method against
%   the Moebius method: each nested attribute's admissible routes are
%   costed in milliseconds by INTERNAL.NESTEDCOST, the chosen ones summed
%   with the flat companions' cost to give the plan's price, and that
%   compared with the enumeration's. The per-attribute prices are
%   reported as the contraction route's reason, an Inf there marking a
%   centres route diverted by the working-set guard. Where the measure
%   rule leaves a nested attribute one admissible route the price is
%   still shown, but the decision was not a cost decision and the report
%   says so. Twin of the Python explain._explain_cosine_nested.

    ncOpts = struct('routesOnly', true, 'methodName', char(method));
    if strcmpi(method, 'centres')
        ncOpts.forceRoute = 'centres';
    end
    blocked = '';
    routes = {};
    try
        [~, routes] = internal.nestedContract(dX, dY, 'cosine', ts, ...
                                              false, ncOpts);
    catch err
        % The measure rule refusing a forced route is a report-worthy
        % answer, not a failure of the report. Mirrors the Python
        % explain path, which catches the same ValueError.
        blocked = err.message;
    end
    parts = {};
    anyTaugrid = false;
    A = double(dX.nAttrs);
    admByAttr = cell(1, A);
    for a = 1:numel(routes)
        if strcmp(routes{a}, '-')
            continue;   % neither a nested nor an ordered-flat route
        end
        anyTaugrid = anyTaugrid || strcmp(routes{a}, 'taugrid');
        if localIsNestedAttr(dX, a) || localIsNestedAttr(dY, a)
            admByAttr{a} = routes(a);
        end
    end

    planMs = NaN;
    enumMs = NaN;
    priced = false;
    enumWhy = 'not selected';
    if isempty(blocked) && any(~cellfun(@isempty, admByAttr))
        % Price the plan as it would run, and the enumeration against it.
        % Guarded: this is a report, so a cost model that cannot price
        % this shape must leave the column blank rather than fail the
        % call, as the eval report's pricing is guarded.
        try
            enumOk = localNestedEnumOk(dX, ts);
            [~, planMs, enumMs] = internal.nestedCost( ...
                'selectNestedMethod', dX, dY, admByAttr, enumOk, ts);
            priced = true;
        catch
            planMs = NaN;  enumMs = NaN;
        end
    end
    if priced
        for a = 1:numel(routes)
            if isempty(admByAttr{a})
                continue;
            end
            % Quote every route the measure rule admits, not only the one
            % taken: an attribute with a single admissible route was not
            % a cost decision, and the reader should see that it had no
            % alternative to price.
            adm = localNestedAdmissible(dX, a, ts);
            [~, ~, prices] = internal.nestedCost('priceNestedAttr', ...
                dX, dY, a, adm, ts);
            quoted = cell(1, numel(adm));
            for k = 1:numel(adm)
                quoted{k} = sprintf('%s %.3f ms', adm{k}, prices.(adm{k}));
            end
            parts{end + 1} = sprintf('attr %d: %s (%s)', a, routes{a}, ...
                                     strjoin(quoted, ', ')); %#ok<AGROW>
        end
    else
        for a = 1:numel(routes)
            if strcmp(routes{a}, '-')
                continue;
            end
            parts{end + 1} = sprintf('attr %d: %s', a, routes{a}); %#ok<AGROW>
        end
    end

    if ~isempty(blocked)
        reason = blocked;
    elseif isempty(parts)
        reason = 'not covered by the contraction plan';
    else
        reason = strjoin(parts, '; ');
    end
    if strcmpi(method, 'bulger')
        chosen = 'bulger';
        enumWhy = 'forced';
    elseif ~strcmpi(method, 'auto') || ~priced
        chosen = 'contract';
        if isempty(blocked) && isempty(parts)
            chosen = 'bulger';
        end
    elseif enumMs * internal.nestedCost('enumSafety') < planMs
        chosen = 'bulger';
    else
        chosen = 'contract';
    end
    if strcmpi(method, 'bulger')
        % already set
    elseif priced && ~isfinite(enumMs)
        enumWhy = ['inadmissible: it computes the minimum-image ' ...
                   'measure, which is not the declared one above the ' ...
                   'sigma/P limit'];
        enumMs = NaN;
    elseif priced
        enumWhy = 'priced';
    end
    if isempty(sop)
        measure = '';
    elseif anyTaugrid
        measure = 'transposition average (the definition)';
    else
        measure = 'wrapped-difference approximation';
    end
    if ~strcmpi(method, 'auto')
        decidedBy = 'user method';
    elseif ~isempty(blocked)
        decidedBy = 'measure rule';
    else
        decidedBy = 'measure rule, then the nested cost model';
    end
    report = struct( ...
        'call', 'cosSimExpTens', ...
        'shape', [localShape(dX) ' (nested)'], ...
        'truncationSigmas', ts, 'floor', floorV, ...
        'sigmaOverP', sop, 'sigmaOverPLimit', limit, ...
        'limitSetBy', setBy, 'measure', measure, ...
        'routeNames', {{'contract', 'bulger'}}, ...
        'routeMs', [planMs, enumMs], ...
        'routeWhy', {{reason, enumWhy}}, ...
        'chosen', chosen, 'decidedBy', decidedBy, 'priced', priced);
end


function tf = localIsNestedAttr(dens, a)
%LOCALISNESTEDATTR  Whether attribute A of DENS carries a nested spec.
    tf = isfield(dens, 'nested') && iscell(dens.nested) ...
        && numel(dens.nested) >= a && ~isempty(dens.nested{a});
end


function adm = localNestedAdmissible(dens, a, ts)
%LOCALNESTEDADMISSIBLE  The routes the measure rule admits for nested
%   attribute A. Report-side twin of NESTEDADMISSIBLEROUTES inside
%   INTERNAL.NESTEDCONTRACT, which is a local function there; the rule is
%   three lines and stating it twice is cheaper than widening that file's
%   interface for a diagnostic. TESTS/TEST_NESTED_COST_MODEL checks the
%   two agree.
    isRel = logical(dens.isRel(a));
    isPer = logical(dens.isPer(a));
    if ~isRel
        adm = {'contract'};
        return;
    end
    if ~isPer
        adm = {'centres', 'contract_relnonper'};
        return;
    end
    period = double(dens.period(a));
    if period > 0 && double(dens.sigma(a)) / period ...
            > internal.relPerSigmaOverPThreshold(ts)
        if isfield(dens, 'wrap') && iscell(dens.wrap) ...
                && numel(dens.wrap) >= a ...
                && strcmp(char(dens.wrap{a}), 'single-image')
            adm = {'centres'};
        else
            adm = {'taugrid'};
        end
        return;
    end
    adm = {'centres', 'taugrid'};
end


function tf = localNestedEnumOk(dens, ts)
%LOCALNESTEDENUMOK  Whether the joint-tuple enumeration carries the
%   declared measure: it computes the minimum-image reading, so a
%   wrap = 'full-image' relative-periodic attribute above the sigma/P
%   threshold rules it out. Report-side twin of
%   NESTEDENUMERATIONADMISSIBLE inside INTERNAL.NESTEDCONTRACT.
    tf = true;
    limit = internal.relPerSigmaOverPThreshold(ts);
    for a = 1:double(dens.nAttrs)
        if ~(logical(dens.isRel(a)) && logical(dens.isPer(a)))
            continue;
        end
        if isfield(dens, 'wrap') && iscell(dens.wrap) ...
                && numel(dens.wrap) >= a ...
                && strcmp(char(dens.wrap{a}), 'single-image')
            continue;
        end
        period = double(dens.period(a));
        if period > 0 && double(dens.sigma(a)) / period > limit
            tf = false;
            return;
        end
    end
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
    fprintf('\n%-12s%12s   %s\n', 'route', 'predicted', 'why');
    for a = 1:numel(rep.routeNames)
        if isnan(rep.routeMs(a))
            pred = '--';
        else
            pred = sprintf('%.3f ms', rep.routeMs(a));
        end
        if strcmp(rep.routeNames{a}, rep.chosen)
            mark = '*';
        else
            mark = ' ';
        end
        % A report that carries a per-route reason (the nested cosine
        % does: each route has its own, as in the Python Route records)
        % shows it for every row. The others keep the older shape, where
        % only the chosen row carries the decision.
        if isfield(rep, 'routeWhy') && numel(rep.routeWhy) >= a ...
                && ~isempty(rep.routeWhy{a})
            why = rep.routeWhy{a};
        elseif mark == '*'
            why = rep.decidedBy;
        else
            why = localTernary(rep.priced, 'priced', 'not selected');
        end
        fprintf('%s%-11s%12s   %s\n', mark, rep.routeNames{a}, pred, why);
    end
    fprintf('\nchosen        %s  (%s)\n', ...
            localTernary(isempty(rep.chosen), 'none', rep.chosen), ...
            rep.decidedBy);
end
