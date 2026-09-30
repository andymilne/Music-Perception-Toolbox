%% test_per_event.m — the per-event form of values and weights
%
%  An attribute may be given as a 1 x N cell with one entry per event, each
%  entry the event's values (a scalar, a vector, or []). It is read exactly
%  as the NaN-padded K x N matrix it stands for. Per-event weights are a
%  1 x N cell, each entry a scalar weighting all the event's values or a
%  vector with one weight per value.
%
%  Mirror of Python tests/test_per_event.py.
%
%  Standalone-runnable; appends to `results` when called from test_mpt.m.

if ~exist('results', 'var')
    results = {};
    standalone = true;
else
    standalone = false;
end
pe_tol = 1e-12;

pe_E = {{[60 64 67], 62, 64, 65}, [0 1 1.5 2]};
pe_M = {[60 62 64 65; 64 NaN NaN NaN; 67 NaN NaN NaN], [0 1 1.5 2]};
pe_geom = {[0.5 0.25], [1 1], [false false], [true false], [12 0]};

pm_e = packPreMaet(pe_E);
pm_m = packPreMaet(pe_M);
results(end+1, :) = {'per event: packed values match the NaN-padded matrix', ...
    isequaln(pm_e.pAttr, pm_m.pAttr)}; %#ok<*SAGROW>

pm = packPreMaet({{[60 64], [], 62}});
results(end+1, :) = {'per event: an empty entry holds no value', ...
    isequaln(pm.pAttr{1}, [60 NaN 62; 64 NaN NaN])};

results(end+1, :) = {'per event: a matrix entry is refused', ...
    throwsErrorWithId(@() packPreMaet({{[60 64; 62 65], 1}}), ...
                      'mpt:perEvent:badValues')};

pm = packPreMaet(pe_E, {{[1 0.5 0.25], 1, 1, 1}, []});
results(end+1, :) = {'per event: a vector of weights, one per value', ...
    isequal(pm.wAttr{1}, [1 1 1 1; 0.5 0 0 0; 0.25 0 0 0])};

pm = packPreMaet(pe_E, {{2, 1, 1, 1}, []});
results(end+1, :) = {'per event: a scalar weight covers the event''s values', ...
    isequal(pm.wAttr{1}, [2 1 1 1; 2 0 0 0; 2 0 0 0])};

results(end+1, :) = {'per event: a weight count mismatch is refused', ...
    throwsErrorWithId(@() packPreMaet(pe_E, {{[1 0.5], 1, 1, 1}, []}), ...
                      'mpt:perEvent:badWeights')};

d_e = buildMaet(pe_E, [], pe_geom{:}, 'verbose', false);
d_m = buildMaet(pe_M, [], pe_geom{:}, 'verbose', false);
pe_q = {61, 1};
results(end+1, :) = {'per event: the built density matches the matrix form', ...
    abs(evalMaet(d_e, pe_q, 'verbose', false) ...
        - evalMaet(d_m, pe_q, 'verbose', false)) < pe_tol ...
    && abs(entropyMaet(d_e, 'method', 'renyi2', 'verbose', false) ...
        - entropyMaet(d_m, 'method', 'renyi2', 'verbose', false)) < pe_tol};

results(end+1, :) = {'per event: raw evalMaet matches the matrix form', ...
    abs(evalMaet(pe_E, [], pe_geom{:}, pe_q, 'verbose', false) ...
        - evalMaet(pe_M, [], pe_geom{:}, pe_q, 'verbose', false)) < pe_tol};

pe_qE = {{[60 63], 62, [64 67], 65}, [0 1 1.5 2]};
pe_qM = {[60 62 64 65; 63 NaN 67 NaN], [0 1 1.5 2]};
results(end+1, :) = {'per event: raw simMaet matches the matrix form', ...
    abs(simMaet(pe_E, [], pe_qE, [], pe_geom{:}, 'verbose', false) ...
        - simMaet(pe_M, [], pe_qM, [], pe_geom{:}, 'verbose', false)) < pe_tol};

pm = transformAttributes({{[440 880], 220}, [1 2]}, [], {{'hz', 'midi'}, 'log'});
results(end+1, :) = {'per event: transformAttributes passes absent values through', ...
    max(max(abs(pm.pAttr{1}(~isnan(pm.pAttr{1})) - [69; 81; 57]))) < 1e-9 ...
    && isnan(pm.pAttr{1}(2, 2))};

if standalone
    nPass = sum([results{:, 2}]);
    fprintf('\n=== test_per_event: %d passed, %d failed (of %d) ===\n', ...
            nPass, size(results, 1) - nPass, size(results, 1));
end
