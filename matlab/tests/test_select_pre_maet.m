%% test_select_pre_maet.m — keeping a selection of a pre-MAET
%
%  Mirror of Python tests/test_select_pre_maet.py.
%
%  Standalone-runnable; appends to `results` when called from test_mpt.m.

if ~exist('results', 'var')
    results = {};
    standalone = true;
else
    standalone = false;
end

sp_p = {[60 64 67; 72 76 79], [0 1 2]};
sp_w = {[1 2 3; 1 1 1], ones(1, 3)};
sp_specs = flatSpecs(sp_p, 'r', [2 1], 'exch', [false true], ...
                     'name', {'pitch', 'onset'});
sp_pm = @() preMaet(sp_p, sp_w, sp_specs);

[p, ~, sp] = unpackPreMaet(selectPreMaet(sp_pm(), 'attributes', {'onset'}));
results(end+1, :) = {'select: attributes by name', ...
    isequal(cellfun(@(s) s.name, sp, 'UniformOutput', false), {'onset'}) ...
    && isequal(size(p{1}), [1 3])}; %#ok<*SAGROW>

[~, ~, sp] = unpackPreMaet(selectPreMaet(sp_pm(), 'attributes', [2 1]));
results(end+1, :) = {'select: attributes by index, in the order given', ...
    isequal(cellfun(@(s) s.name, sp, 'UniformOutput', false), {'onset', 'pitch'})};

[~, ~, sp] = unpackPreMaet(selectPreMaet(sp_pm(), 'attributes', [true false]));
results(end+1, :) = {'select: attributes by mask', ...
    isequal(cellfun(@(s) s.name, sp, 'UniformOutput', false), {'pitch'})};

[p, w, ~] = unpackPreMaet(selectPreMaet(sp_pm(), 'events', [1 3]));
results(end+1, :) = {'select: events by index', ...
    isequal(size(p{1}), [2 2]) && isequal(p{2}, [0 2]) ...
    && isequal(w{1}(1, :), [1 3])};

[p, ~, ~] = unpackPreMaet(selectPreMaet(sp_pm(), 'events', [false true false]));
results(end+1, :) = {'select: events by mask', isequal(p{2}, 1)};

[p, ~, sp] = unpackPreMaet(selectPreMaet(sp_pm(), ...
    'attributes', {'pitch'}, 'events', 3));
results(end+1, :) = {'select: both at once', ...
    isequal(cellfun(@(s) s.name, sp, 'UniformOutput', false), {'pitch'}) ...
    && isequal(p{1}, [67; 79])};

% A selection cannot change what an attribute means.
[~, ~, sp] = unpackPreMaet(selectPreMaet(sp_pm(), 'events', [1 2]));
results(end+1, :) = {'select: an attribute keeps its tuple size and flags', ...
    sp{1}.r == 2 && sp{1}.exch == false};

% A level's simplex coordinates are one attribute, not several, so no
% selection over attributes can take part of one.
sp_vert = simplexVertices(4);
sp_pmv = preMaet({[60 64], sp_vert(1:2, :).'}, [], ...
    flatSpecs({[60 64], zeros(3, 2)}, 'r', [1 3], 'exch', [true false], ...
              'name', {'pitch', 'voice'}));
[p, ~, sp] = unpackPreMaet(selectPreMaet(sp_pmv, 'attributes', {'voice'}));
results(end+1, :) = {'select: a multi-coordinate attribute moves whole', ...
    isequal(size(p{1}), [3 2]) && sp{1}.r == 3};

results(end+1, :) = {'select: keeping no attribute is refused', ...
    throwsErrorWithId(@() selectPreMaet(sp_pm(), 'attributes', false(1, 2)), ...
                      'selectPreMaet:emptySelection')};

results(end+1, :) = {'select: an unknown name is refused', ...
    throwsErrorWithId(@() selectPreMaet(sp_pm(), 'attributes', {'velocity'}), ...
                      'selectPreMaet:unknownName')};

results(end+1, :) = {'select: an out-of-range index is refused', ...
    throwsErrorWithId(@() selectPreMaet(sp_pm(), 'events', 5), ...
                      'selectPreMaet:outOfRange')};

results(end+1, :) = {'select: a wrong-length mask is refused', ...
    throwsErrorWithId(@() selectPreMaet(sp_pm(), 'events', [true false]), ...
                      'selectPreMaet:maskLength')};

% Allowed, and it means what it says: the event contributes nothing on
% that attribute while keeping its place.
sp_gap = preMaet({[60 NaN], [0 1]}, {[1 0], ones(1, 2)}, ...
                 flatSpecs({[60 NaN], [0 1]}, 'name', {'pitch', 'onset'}));
[p, ~, ~] = unpackPreMaet(selectPreMaet(sp_gap, 'events', 2));
results(end+1, :) = {'select: selecting may leave an event with no value', ...
    isnan(p{1}(1, 1))};

sp_out = selectPreMaet(sp_pm(), 'attributes', {'pitch'}, 'events', [1 2]);
sp_d = buildMaet(sp_out.pAttr, sp_out.wAttr, 1, 2, false, false, 0, ...
                 'verbose', false);
results(end+1, :) = {'select: the result is a pre-MAET that builds', ...
    abs(simMaet(sp_d, sp_d, 'verbose', false) - 1) < 1e-12};

if standalone
    nPass = sum([results{:, 2}]);
    fprintf('\n=== test_select_pre_maet: %d passed, %d failed (of %d) ===\n', ...
            nPass, size(results, 1) - nPass, size(results, 1));
end
