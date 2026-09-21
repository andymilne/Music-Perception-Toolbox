%% test_empty_events.m — events empty on some attributes, populated on others
%
%  A grid point with nothing sounding is an event that carries a time and
%  no pitch. It has to be keepable: the populated attributes are doing work
%  even where one is empty, and dropping it would make the event index no
%  longer a uniform time index. Such an event admits no tuple and so
%  contributes nothing, to the density, the inner product, or any entropy
%  taken from them. A partly filled event -- fewer than r values but more
%  than none -- is still an error.
%
%  Mirror of Python tests/test_empty_events.py.
%
%  Standalone-runnable; appends to `results` when called from test_mpt.m.

if ~exist('results', 'var')
    results = {};
    standalone = true;
else
    standalone = false;
end
ee_tol = 1e-12;

ee_two = @(p) buildMaet(p, [], [1 0.1], [1 1], [false false], ...
                        [false false], [0 0], 'verbose', false);
ee_one = @(p, r) buildMaet({p}, [], 1, r, false, false, 0, 'verbose', false);

% One live event and one empty on pitch, against the live one alone.
withEmpty = ee_two({[60 NaN], [0 1]});
alone     = ee_two({60, 0});

results(end+1, :) = {'empty events: an event empty on one attribute builds', ...
    withEmpty.N == 2}; %#ok<*SAGROW>

ee_query = {60, 0};
results(end+1, :) = {'empty events: it contributes nothing to the density', ...
    abs(evalMaet(withEmpty, ee_query, 'verbose', false) ...
        - evalMaet(alone, ee_query, 'verbose', false)) < ee_tol};

results(end+1, :) = {'empty events: it contributes nothing to the inner product', ...
    abs(simMaet(withEmpty, alone, 'verbose', false) - 1) < ee_tol ...
    && abs(simMaet(withEmpty, withEmpty, 'verbose', false) - 1) < ee_tol};

% Renyi-2 is closed form, so the two agree exactly rather than to a
% quadrature tolerance.
results(end+1, :) = {'empty events: it contributes nothing to the entropy', ...
    abs(entropyMaet(withEmpty, 'method', 'renyi2', 'verbose', false) ...
        - entropyMaet(alone, 'method', 'renyi2', 'verbose', false)) < ee_tol};

% A silent slice beside a chord, at r = 2.
ee_chord = ee_one([60 NaN; 64 NaN], 2);
ee_only  = ee_one([60; 64], 2);
results(end+1, :) = {'empty events: a silent slice beside a chord at r = 2', ...
    abs(simMaet(ee_chord, ee_only, 'verbose', false) - 1) < ee_tol};

% A nested attribute tolerates an empty event.
ee_spec = struct('r', [2 2], 'exch', [0 1], 'tags', [0; 0; 1; 1]);
ee_nested = buildMaet({[60 NaN; 64 NaN; 62 NaN; 65 NaN]}, [], 1, 1, ...
                      false, false, 0, 'nested', {ee_spec}, 'verbose', false);
results(end+1, :) = {'empty events: a nested attribute tolerates an empty event', ...
    abs(simMaet(ee_nested, ee_nested, 'verbose', false) - 1) < ee_tol};

% Every event empty gives a zero-mass density.
ee_allEmpty = ee_one([NaN NaN], 1);
results(end+1, :) = {'empty events: every event empty gives a zero-mass density', ...
    abs(evalMaet(ee_allEmpty, {60}, 'verbose', false)) < ee_tol};

% Two values asked of an event that has one is a mistake, not a rest.
results(end+1, :) = {'empty events: a partly filled event is still an error', ...
    throwsErrorWithId(@() ee_one([60 62; 64 NaN], 2), ...
                      'buildMaet:insufficientValues')};

if standalone
    nPass = sum([results{:, 2}]);
    fprintf('\n=== test_empty_events: %d passed, %d failed (of %d) ===\n', ...
            nPass, size(results, 1) - nPass, size(results, 1));
end
