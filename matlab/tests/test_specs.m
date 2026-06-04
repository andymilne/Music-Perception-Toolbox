%% test_specs.m — canonical specs= form of buildExpTens (3c-i)
%
%  specs is the single home for level-structured geometry (toolbox spec
%  §6.4): a cell of per-attribute structs. A flat attribute is a one-level
%  spec struct('r',.,'rel',.,'sym',.,'name',.) (scalar r, bool rel/sym); a
%  nested attribute carries a 'tags' field plus per-level vectors. Scalar
%  geometry not structured by nesting (sigma, isPer, period) stays outside
%  the spec, supplied as name-value kwargs. The old positional form is
%  retained unchanged as a shim.

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    clear cleanupDefaults
    cleanupDefaults = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end

P2 = {[0 4; 7 11], [100 140]};


% --- Flat specs match old positional ---
d_old = buildExpTens(P2, [], [50 30], [2 1], [true false], ...
                     [false false], [0 0], 'verbose', false);
specs = {struct('r', 2, 'rel', true, 'sym', true), struct('r', 1, 'rel', false)};
d_spec = buildExpTens(P2, [], 'specs', specs, 'sigma', [50 30], ...
                      'isPer', [false false], 'period', [0 0], 'verbose', false);
okFlat = (d_old.dim == d_spec.dim);
for q = {[5; 100], [7; 140], [3; 120]}
    okFlat = okFlat && abs( ...
        evalExpTens(d_old,  q{1}, 'verbose', false) ...
      - evalExpTens(d_spec, q{1}, 'verbose', false)) < 1e-12;
end
results{end+1,1} = 'specs: flat specs match old positional';
results{end,2}   = okFlat;


% --- Flat-spec defaults (rel=false, sym=true) ---
d_min = buildExpTens(P2, [], 'specs', {struct('r', 2), struct('r', 1)}, ...
                     'sigma', [50 30], 'isPer', [false false], ...
                     'period', [0 0], 'verbose', false);
results{end+1,1} = 'specs: flat defaults rel=false sym=true';
results{end,2}   = (d_min.isRel(1) == false) && (d_min.isSym(1) == true);


% --- Nested spec matches nested= kwarg ---
nsp = struct('tags', [0 0 1 1], 'r', [2 2], 'sym', [true false], 'rel', 'outermost');
d_kw = buildExpTens({[0; 4; 7; 11]}, [], [50], [1], [false], [false], [0], ...
                    'nested', {nsp}, 'verbose', false);
d_ns = buildExpTens({[0; 4; 7; 11]}, [], 'specs', {nsp}, 'sigma', 50, ...
                    'isPer', false, 'period', 0, 'verbose', false);
results{end+1,1} = 'specs: nested spec matches nested= kwarg';
results{end,2}   = (d_kw.dim == d_ns.dim) && (d_ns.dim == 3) ...
                   && (abs(cosSimExpTens(d_kw, d_ns, 'verbose', false) - 1) < 1e-9);


% --- Attribute names stored and prune-invariant ---
dN = buildExpTens(P2, [], 'specs', ...
        {struct('r', 2, 'rel', true, 'name', 'pitch'), ...
         struct('r', 1, 'name', 'time')}, ...
        'sigma', [50 30], 'isPer', [false false], 'period', [0 0], ...
        'verbose', false);
dNp = internal.prunedExpTens(dN);
okNames = isfield(dN, 'names') ...
    && strcmp(dN.names{1}, 'pitch') && strcmp(dN.names{2}, 'time') ...
    && strcmp(dNp.names{1}, 'pitch');
results{end+1,1} = 'specs: attribute names stored and prune-invariant';
results{end,2}   = okNames;


% --- Level names carried in nested spec ---
nspNamed = struct('tags', [0 0 1 1], 'r', [2 2], 'sym', [true false], ...
                  'rel', 'innermost', 'name', 'chordprog');
nspNamed.names = {'note', 'chord'};
dL = buildExpTens({[0; 4; 7; 11]}, [], 'specs', {nspNamed}, 'sigma', 50, ...
                  'isPer', false, 'period', 0, 'verbose', false);
results{end+1,1} = 'specs: level names carried in nested spec';
results{end,2}   = strcmp(dL.names{1}, 'chordprog') ...
                   && isfield(dL.nested{1}, 'names') ...
                   && strcmp(dL.nested{1}.names{2}, 'chord');


% --- Guards ---
okG1 = false; okG2 = false; okG3 = false; okG4 = false; okG5 = false;
try
    buildExpTens(P2, [], [50 30], 'specs', specs, 'sigma', [1 1], ...
        'isPer', [false false], 'period', [0 0], 'verbose', false);
catch; okG1 = true; end
try
    buildExpTens(P2, [], 'specs', specs, 'isPer', [false false], ...
        'period', [0 0], 'verbose', false);
catch; okG2 = true; end
try
    buildExpTens(P2, [], [50 30], [2 1], [true false], [false false], ...
        [0 0], 'sigma', [1 1], 'verbose', false);
catch; okG3 = true; end
try
    buildExpTens(P2, [], 'specs', {struct('r', 2)}, 'sigma', [1 1], ...
        'isPer', [false false], 'period', [0 0], 'verbose', false);
catch; okG4 = true; end
try
    buildExpTens(P2, [], 'specs', {struct('rel', true), struct('r', 1)}, ...
        'sigma', [1 1], 'isPer', [false false], 'period', [0 0], 'verbose', false);
catch; okG5 = true; end
results{end+1,1} = 'specs: guards (positional/missing/length/flat-r)';
results{end,2}   = okG1 && okG2 && okG3 && okG4 && okG5;


% ===================================================================
%  3c-iv-a: flatSpecs constructor + partial hand-edit tolerance
% ===================================================================

% --- flatSpecs constructor matches positional ---
sFS = flatSpecs(P2, 'r', [2 1], 'rel', [true false], 'name', {'pitch', 'time'});
dFS = buildExpTens(P2, [], 'specs', sFS, 'sigma', [50 30], ...
                   'isPer', [false false], 'period', [0 0], 'verbose', false);
dPos = buildExpTens(P2, [], [50 30], [2 1], [true false], ...
                    [false false], [0 0], 'verbose', false);
okFS = (dFS.dim == dPos.dim) ...
       && strcmp(sFS{1}.name, 'pitch') && sFS{1}.r == 2 && sFS{1}.rel == true;
for q = {[5; 100], [7; 140]}
    okFS = okFS && abs(evalExpTens(dFS,  q{1}, 'verbose', false) ...
                     - evalExpTens(dPos, q{1}, 'verbose', false)) < 1e-12;
end
results{end+1,1} = 'specs: flatSpecs constructor matches positional';
results{end,2}   = okFS;


% --- flatSpecs scalar broadcast ---
sSB = flatSpecs({zeros(1,3), zeros(1,3), zeros(1,3)}, 'r', 2);
results{end+1,1} = 'specs: flatSpecs scalar broadcast';
results{end,2}   = numel(sSB) == 3 && sSB{2}.r == 2 ...
                   && sSB{3}.rel == false && sSB{1}.sym == true;


% --- Nested omitted sym defaults to all-True ---
pvT = {[0; 4; 7; 11]};
full    = struct('tags', [0 0 1 1], 'r', [2 2], 'sym', [true true], 'rel', 'innermost');
partial = struct('tags', [0 0 1 1], 'r', [2 2], 'rel', 'innermost');   % no sym
dFull = buildExpTens(pvT, [], 'specs', {full},    'sigma', 50, ...
                     'isPer', false, 'period', 0, 'verbose', false);
dPart = buildExpTens(pvT, [], 'specs', {partial}, 'sigma', 50, ...
                     'isPer', false, 'period', 0, 'verbose', false);
results{end+1,1} = 'specs: nested omitted sym defaults all-True';
results{end,2}   = abs(cosSimExpTens(dFull, dPart, 'verbose', false) - 1) < 1e-9;


% --- Nested carries unknown fields (name, names, stray key) ---
edited = struct('tags', [0 0 1 1], 'r', [2 2], 'rel', 'outermost', 'name', 'cp');
edited.names = {'n', 'c'};
edited.myNote = 'hand-edit';
dE = buildExpTens(pvT, [], 'specs', {edited}, 'sigma', 50, ...
                  'isPer', false, 'period', 0, 'verbose', false);
results{end+1,1} = 'specs: nested carries unknown fields';
results{end,2}   = strcmp(dE.names{1}, 'cp') ...
                   && isfield(dE.nested{1}, 'names') ...
                   && strcmp(dE.nested{1}.names{2}, 'c') ...
                   && isfield(dE.nested{1}, 'myNote') ...
                   && strcmp(dE.nested{1}.myNote, 'hand-edit');


% --- Structural fields required (r, tags) ---
okSR1 = false; okSR2 = false;
try
    buildExpTens(pvT, [], 'specs', {struct('tags', [0 0 1 1], 'sym', [true false])}, ...
        'sigma', 50, 'isPer', false, 'period', 0, 'verbose', false);
catch; okSR1 = true; end
try
    buildExpTens(pvT, [], 'specs', {struct('r', [2 2], 'sym', [true false])}, ...
        'sigma', 50, 'isPer', false, 'period', 0, 'verbose', false);
catch; okSR2 = true; end
results{end+1,1} = 'specs: structural fields r/tags required';
results{end,2}   = okSR1 && okSR2;


% --- Standalone summary ---
if standalone
    nPass = sum([results{:, 2}]);
    nFail = size(results, 1) - nPass;
    for i = 1:size(results, 1)
        if results{i, 2}
            fprintf('  PASS  %s\n', results{i, 1});
        else
            fprintf('  FAIL  %s\n', results{i, 1});
        end
    end
    fprintf('\n=== test_specs: %d passed, %d failed (of %d) ===\n\n', ...
        nPass, nFail, nPass + nFail);
    clear cleanupDefaults
    if nFail > 0
        error('test_specs:failed', '%d test(s) failed.', nFail);
    end
end
