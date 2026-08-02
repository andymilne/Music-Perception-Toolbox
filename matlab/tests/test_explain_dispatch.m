%% test_explain_dispatch.m — explainDispatch agrees with the dispatch
%
%  The report must agree with the dispatch it describes: if it says a
%  call takes a route, that is the route the selector picks. These
%  checks assert that agreement rather than the wording, so they survive
%  rephrasing but fail if the report and the toolbox part company.
%
%  Python twin: tests/test_explain_dispatch.py. The one check that does
%  not transliterate is the Python file's monkeypatched proof that
%  explaining a call does not evaluate it; the MATLAB analogue is
%  check 10, which explains a shape whose joint-centres route the cost
%  model prices in minutes and asserts a report comes back anyway.
%
%  Standalone-runnable; appends to `results` when called from test_mpt.m.

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    clear cleanupDefaults
    cleanupDefaults = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end

xdP = 1200;

% Deterministic values (no RNG dependence across languages): K points
% spread unevenly over the period so the collection is not degenerate.
xdPts = @(K) xdP * (mod((1:K) * 0.37, 1)).';

% --- 1. The reported route is the route the selector picks ------------
xdCases = { ...
    2, 12, false, false, 15; ...
    3, 12, true,  true,  60; ...
    3, 24, true,  false, 15; ...
    4, 12, false, true,  15};
xdOk = true;
for xdI = 1:size(xdCases, 1)
    [xdR, xdK, xdRel, xdPer, xdS] = xdCases{xdI, :};
    if xdPer, xdPeriod = xdP; else, xdPeriod = 0; end
    xdD = buildExpTens(xdPts(xdK), [], xdS, xdR, xdRel, xdPer, ...
                       xdPeriod, 'verbose', false);
    xdSel = internal.selectMaEval(xdD, 200, false);
    xdRep = explainDispatch(xdD, 200);
    xdOk = xdOk && strcmp(xdRep.chosen, xdSel);
end
results{end+1, 1} = 'explainDispatch: reported route is the selected route'; %#ok<*AGROW>
results{end, 2}   = xdOk;

% --- 2. A forced method is reported as chosen -------------------------
xdD = buildExpTens(xdPts(12), [], 60, 3, true, true, xdP, 'verbose', false);
xdRepC = explainDispatch(xdD, 200, 'method', 'centres');
xdRepM = explainDispatch(xdD, 200, 'method', 'mobius');
xdOk = strcmp(xdRepC.chosen, 'centres') && strcmp(xdRepM.chosen, 'mobius');
results{end+1, 1} = 'explainDispatch: a forced method is reported as chosen';
results{end, 2}   = xdOk;

% --- 3. The floor is the one the accuracy width implies ---------------
xdOk = true;
for xdTs = [4, 6, 8, Inf]
    xdRep = explainDispatch(xdD, 200, 'truncationSigmas', xdTs);
    xdOk = xdOk && abs(xdRep.floor - internal.truncationFloor(xdTs)) <= ...
                   1e-15 * max(1, internal.truncationFloor(xdTs));
end
results{end+1, 1} = 'explainDispatch: floor matches the accuracy asked for';
results{end, 2}   = xdOk;

% --- 4. The sigma/P limit moves with the accuracy width ---------------
%  The limit is a function of truncationSigmas, so the report must not
%  cache one value: a tighter setting gives a smaller limit, which is
%  the whole reason the constant became a function.
xdLoose = explainDispatch(xdD, 200, 'truncationSigmas', 4);
xdTight = explainDispatch(xdD, 200, 'truncationSigmas', 8);
results{end+1, 1} = 'explainDispatch: sigma/P limit tracks the accuracy asked for';
results{end, 2}   = xdLoose.sigmaOverPLimit > xdTight.sigmaOverPLimit && ...
                    xdTight.sigmaOverPLimit == ...
                        internal.relPerSigmaOverPThreshold(8);

% --- 5. sigma/P is absent when nothing is relative-periodic -----------
xdDnp = buildExpTens(xdPts(12), [], 60, 3, false, false, 0, ...
                     'verbose', false);
results{end+1, 1} = 'explainDispatch: sigma/P absent when not relative-periodic';
xdRepNp = explainDispatch(xdDnp, 200);
results{end, 2}   = isempty(xdRepNp.sigmaOverP);

% --- 6. Both routes are priced when the cost model decides ------------
%  Non-periodic, so no measure rule pre-empts the pricing.
xdRep = explainDispatch( ...
    buildExpTens(xdPts(12), [], 15, 3, true, false, 0, 'verbose', false), ...
    200);
results{end+1, 1} = 'explainDispatch: both routes priced when the cost model decides';
results{end, 2}   = all(isfinite(xdRep.routeMs)) && all(xdRep.routeMs > 0);

% --- 7. Exactly one route is marked chosen ----------------------------
xdRep = explainDispatch(xdD, 200);
results{end+1, 1} = 'explainDispatch: exactly one route is chosen';
results{end, 2}   = sum(strcmp(xdRep.routeNames, xdRep.chosen)) == 1;

% --- 8. A density pair explains a cosine ------------------------------
xdX = buildExpTens(xdPts(12), [], 60, 3, true, true, xdP, 'verbose', false);
xdY = buildExpTens(xdP * mod((1:12) * 0.61, 1).', [], 60, 3, true, true, ...
                   xdP, 'verbose', false);
xdRep = explainDispatch(xdX, xdY);
results{end+1, 1} = 'explainDispatch: a density pair explains a cosine';
results{end, 2}   = strcmp(xdRep.call, 'cosSimExpTens') && ...
                    any(strcmp(xdRep.chosen, {'bulger', 'mobius'}));

% --- 9. The printed report names the quantities it turns on -----------
xdText = evalc('explainDispatch(xdD, 200);');
results{end+1, 1} = 'explainDispatch: printed report names chosen, truncationSigmas, sigma/P';
results{end, 2}   = contains(xdText, 'chosen') && ...
                    contains(xdText, 'truncationSigmas') && ...
                    contains(xdText, 'sigma/P');

% --- 10. Explaining a call does not evaluate it -----------------------
%  r = 7 over K = 60 would materialise C(60, 7) = 386 million tuples on
%  the joint-centres route; the cost model prices that route in minutes.
%  A report still comes back, so the explanation is not the call.
xdBig = buildExpTens(xdPts(60), [], 20, 7, false, false, 0, ...
                     'verbose', false);
xdRep = explainDispatch(xdBig, 200);
xdCentres = xdRep.routeMs(strcmp(xdRep.routeNames, 'centres'));
results{end+1, 1} = 'explainDispatch: explaining a prohibitive call still returns a report';
results{end, 2}   = ischar(xdRep.chosen) && ~isempty(xdRep.chosen) && ...
                    xdCentres > 6e4;

% --- 11. Past the sigma/P limit the measure decides, not the price ----
%  At sigma = 60 over P = 1200 the wrapped-difference form is
%  inadmissible, so the transposition average is computed whatever the
%  relative cost of the two routes. The decision is structural: the
%  selector returns before pricing either route, so the report marks it
%  unpriced rather than quoting times the decision did not rest on.
xdRep = explainDispatch(xdD, 200);
results{end+1, 1} = 'explainDispatch: past the sigma/P limit the measure decides the route';
results{end, 2}   = xdRep.sigmaOverP > xdRep.sigmaOverPLimit && ...
                    strcmp(xdRep.chosen, 'mobius') && ...
                    strcmp(xdRep.measure, ...
                           'transposition average (the definition)') && ...
                    ~xdRep.priced;

% --- 12. Within the sigma/P limit the wrapped-difference form is used -
xdRep = explainDispatch( ...
    buildExpTens(xdPts(12), [], 24, 3, true, true, xdP, 'verbose', false), ...
    200);
results{end+1, 1} = 'explainDispatch: within the sigma/P limit the approximation is used';
results{end, 2}   = xdRep.sigmaOverP <= xdRep.sigmaOverPLimit && ...
                    strcmp(xdRep.measure, 'wrapped-difference approximation');

if standalone
    nPass = sum(cellfun(@(x) isequal(x, true), results(:,2)));
    nFail = numel(results(:,1)) - nPass;
    fprintf('\n=== test_explain_dispatch: %d passed, %d failed (of %d) ===\n', ...
        nPass, nFail, numel(results(:,1)));
    clear cleanupDefaults
    if nFail > 0
        for ii = 1:size(results, 1)
            if ~isequal(results{ii, 2}, true)
                fprintf('  FAIL  %s\n', results{ii, 1});
            end
        end
    end
end
