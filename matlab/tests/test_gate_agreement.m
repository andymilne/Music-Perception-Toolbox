%% test_gate_agreement.m
%  Cross-language gate agreement, driven from a shared fixture. Twin of
%  the Python tests/test_gate_agreement.py.
%
%  The existing gate tests pin decisions independently in each language
%  against hand-written expectations. That catches a regression in one
%  language but not a shared misunderstanding: if both sides were wrong
%  in the same way, both would pass. This reads the same fixture the
%  Python twin reads, so a decision can only be marked correct once.
%
%  Why gate agreement is worth a dedicated test. These gates choose
%  between routes, and before the full-image work those routes computed
%  different measures -- the centres route the pairwise-wrap
%  (single-image) form, the grid route a transposition average. A gate
%  disagreement would then have surfaced as a value difference looking
%  like a numerical bug rather than a routing mismatch. The routes now
%  agree in value, so a disagreement costs only time; the constants are
%  still deliberately matched, and this test is what holds them matched.
%
%  The Python twin additionally checks that the decisions are monotone in
%  each value count within a shape; that check reads the same fixture and
%  is not duplicated here.
%
%  No local functions: this file is executed as a script from test_mpt.m.
%
%  Standalone-runnable; appends to `results` when called from test_mpt.m.

if ~exist('results', 'var')
    results = {};
    standalone = true;
else
    standalone = false;
end

gaJson = fullfile(fileparts(mfilename('fullpath')), ...
                  'gate_agreement_parity.json');
if exist(gaJson, 'file') ~= 2
    results(end+1, :) = {'gate agreement: fixture present', false}; %#ok<*SAGROW>
else
    gaFix = jsondecode(fileread(gaJson));
    % jsondecode yields a cell array rather than a struct array when the
    % cases are non-uniform (the position vectors differ in length across
    % K), so accept either shape.
    if iscell(gaFix.cases)
        gaGet = @(k) gaFix.cases{k};
        gaN   = numel(gaFix.cases);
    else
        gaGet = @(k) gaFix.cases(k);
        gaN   = numel(gaFix.cases);
    end

    % A fixture that is all-centres or all-grid would pass trivially
    % while testing nothing about where the boundary sits.
    gaNTrue = 0;
    for ii = 1:gaN
        if gaGet(ii).decision
            gaNTrue = gaNTrue + 1;
        end
    end
    gaFrac = gaNTrue / max(gaN, 1);
    results(end+1, :) = { ...
        sprintf('gate agreement: fixture not degenerate (%d/%d centres)', ...
                gaNTrue, gaN), ...
        gaN >= 100 && gaFrac > 0.05 && gaFrac < 0.95};

    % Every decision must reproduce exactly.
    gaBad = 0;
    gaFirst = '';
    for ii = 1:gaN
        c = gaGet(ii);
        px = c.px(:);
        py = c.py(:);
        got = mobius.maRelAttrPrefersCentres(px, py, c.sigma, c.r, ...
                                             true, logical(c.isPer), ...
                                             c.period);
        if ~isequal(logical(got), logical(c.decision))
            gaBad = gaBad + 1;
            if isempty(gaFirst)
                gaFirst = sprintf(['r=%d K=%d sigma=%g isPer=%d ' ...
                                   'period=%g: expected %d, got %d'], ...
                                  c.r, c.K, c.sigma, c.isPer, c.period, ...
                                  c.decision, got);
            end
        end
    end
    if gaBad == 0
        results(end+1, :) = { ...
            sprintf('gate agreement: all %d decisions reproduce', gaN), true};
    else
        results(end+1, :) = { ...
            sprintf('gate agreement: %d/%d decisions differ (first: %s)', ...
                    gaBad, gaN, gaFirst), false};
    end

    % A chord against a scale, or a reference tuning against an equal
    % division, gives the two densities different numbers of values. The
    % fit the gate constants come from used equal counts throughout, so
    % the fixture has to carry the unequal case or nothing holds the two
    % languages matched on it. K is the first density's count, Ky the
    % second's.
    gaNUneq = 0;
    gaNUneqCentres = 0;
    for ii = 1:gaN
        c = gaGet(ii);
        if c.Ky ~= c.K
            gaNUneq = gaNUneq + 1;
            if c.decision
                gaNUneqCentres = gaNUneqCentres + 1;
            end
        end
    end
    results(end+1, :) = { ...
        sprintf(['gate agreement: unequal value counts covered ' ...
                 '(%d cases, %d centres)'], gaNUneq, gaNUneqCentres), ...
        gaNUneq >= 40 && gaNUneqCentres > 0 && gaNUneqCentres < gaNUneq};
end


if standalone
    nPass = sum(cellfun(@(x) isequal(x, true), results(:, 2)));
    nFail = size(results, 1) - nPass;
    fprintf('test_gate_agreement: %d passed, %d failed\n', nPass, nFail);
    for ii = 1:size(results, 1)
        if ~isequal(results{ii, 2}, true)
            fprintf('  FAIL  %s\n', results{ii, 1});
        end
    end
end
