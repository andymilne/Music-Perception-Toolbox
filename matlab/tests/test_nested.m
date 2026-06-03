%% test_nested.m — nested binding (representation B) in the MAET build core
%
%  Mirror of the Python tests/test_nested.py. Covers the toolbox
%  specification §6/§7.1/§8/§10 for the two-level nested attribute. The
%  nested spec uses per-level vectors (innermost-outward):
%  struct('tags',..., 'r',[ri ro], 'sym',[si so], 'rel',<...>), where rel
%  is the co-transposition-unit selector (a per-level vector or the
%  depth-proof strings 'innermost'/'outermost'; a bare scalar/bool is
%  rejected for a nested attribute).
%
%   * Outer r = K reproduces old separate-attribute binding (tensor join).
%   * Partial symmetry (sym inner=1, outer=0): inner slots orbit within
%     each source event; bound events keep order (no cross-tag interleave).
%   * Pooled within-source reading (outer r < K) -> shared lower-dim space.
%   * Outer/whole [rel] unit: dim 4->3 and exact global-transposition
%     invariance (centres shift-invariant; cosine of a progression and its
%     transpose == 1); absolute is not invariant.
%   * Guards: scalar/bool rel rejected for nested; user isRelVec on a
%     nested attribute rejected; inner rel not yet wired; insufficient
%     distinct source events errors.
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

setOf = @(C) sortrows(unique(C.', 'rows'));   % column set of a centres matrix


% --- Outer r = K reproduces old separate-attribute binding ---
d_old = buildExpTens({[0], [7]}, [], [50 50], [1 1], ...
                     [false false], [false false], [0 0], 'verbose', false);
nsp1  = struct('tags', [0 1], 'r', [1 2], 'sym', [true false]);
d_nest = buildExpTens({[0; 7]}, [], 50, 1, false, false, 0, ...
                      'nested', {nsp1}, 'verbose', false);
okp = (d_old.dim == 2) && (d_nest.dim == 2);
queries = {[0; 7], [0; 0], [3; 7], [-5; 12]};
for qi = 1:numel(queries)
    vo = evalExpTens(d_old,  queries{qi}, 'verbose', false);
    vn = evalExpTens(d_nest, queries{qi}, 'verbose', false);
    okp = okp && abs(vo - vn) < 1e-12;
end
results{end+1,1} = 'nested: outer r=K reproduces old separate-attr binding';
results{end,2}   = okp;


% --- Inner S_2 orbit, outer order preserved (no cross-tag interleaving) ---
nsp2 = struct('tags', [0 0 1 1], 'r', [2 2], 'sym', [true false]);
d2 = internal.ensureExpTensExpensive( ...
    buildExpTens({[0; 4; 7; 11]}, [], 50, 1, false, false, 0, ...
                 'nested', {nsp2}, 'verbose', false));
cset = setOf(d2.Centres{1});
expected = sortrows([0 4 7 11; 0 4 11 7; 4 0 7 11; 4 0 11 7]);
okset = isequal(cset, expected);
nointer = true;
for rr = 1:size(cset, 1)
    nointer = nointer ...
        && isequal(sort(cset(rr, 1:2)), [0 4]) ...
        && isequal(sort(cset(rr, 3:4)), [7 11]);
end
results{end+1,1} = 'nested: inner S2 orbit, outer order preserved (no interleave)';
results{end,2}   = okset && nointer && (d2.dim == 4);


% --- Pooled within-source reading (outer r < K) ---
nsp3 = struct('tags', [0 0 1 1], 'r', [2 1], 'sym', [false false]);
d3 = internal.ensureExpTensExpensive( ...
    buildExpTens({[0; 4; 7; 11]}, [], 50, 1, false, false, 0, ...
                 'nested', {nsp3}, 'verbose', false));
results{end+1,1} = 'nested: pooled outer r<K (shared 2-D space)';
results{end,2}   = isequal(setOf(d3.Centres{1}), sortrows([0 4; 7 11])) ...
                   && (d3.dim == 2);


% --- Outer/whole [rel] unit: dim 4->3 and global-transposition invariance ---
nspOut = struct('tags', [0 0 1 1], 'r', [2 2], 'sym', [true false], ...
                'rel', 'outermost');
dO  = buildExpTens({[0; 4; 7; 11]},  [], 50, 1, false, false, 0, ...
                   'nested', {nspOut}, 'verbose', false);
dOS = buildExpTens({[5; 9; 12; 16]}, [], 50, 1, false, false, 0, ...
                   'nested', {nspOut}, 'verbose', false);
dOe  = internal.ensureExpTensExpensive(dO);
dOSe = internal.ensureExpTensExpensive(dOS);
okOut = (dO.dim == 3) ...
    && isequal(sort(dOe.Centres{1}, 2), sort(dOSe.Centres{1}, 2)) ...
    && abs(cosSimExpTens(dO, dOS, 'verbose', false) - 1) < 1e-9;
results{end+1,1} = 'nested: outer [rel] dim 4->3 and transposition-invariant';
results{end,2}   = okOut;


% --- Absolute nested density is NOT transposition invariant ---
nspAbs = struct('tags', [0 0 1 1], 'r', [2 2], 'sym', [true false]);
dA  = buildExpTens({[0; 4; 7; 11]},  [], 50, 1, false, false, 0, ...
                   'nested', {nspAbs}, 'verbose', false);
dAS = buildExpTens({[5; 9; 12; 16]}, [], 50, 1, false, false, 0, ...
                   'nested', {nspAbs}, 'verbose', false);
results{end+1,1} = 'nested: absolute is not transposition invariant';
results{end,2}   = (dA.dim == 4) ...
                   && (cosSimExpTens(dA, dAS, 'verbose', false) < 0.999);


% --- rel=[0 1] (outermost level) equals rel='outermost' ---
nspV = struct('tags', [0 0 1 1], 'r', [2 2], 'sym', [true false], 'rel', [0 1]);
dV = internal.ensureExpTensExpensive( ...
    buildExpTens({[0; 4; 7; 11]}, [], 50, 1, false, false, 0, ...
                 'nested', {nspV}, 'verbose', false));
results{end+1,1} = 'nested: rel=[0 1] equals rel=''outermost''';
results{end,2}   = (dV.dim == 3) ...
                   && isequal(sort(dV.Centres{1}, 2), sort(dOe.Centres{1}, 2));


% --- §6.3 projection-rank dims 4/2/3/2 ---
perA = [1 1 0 0]; perB = [0 0 1 1]; glob = [1 1 1 1];
dims = [4, 4 - rank([perA; perB]), 4 - rank(glob), 4 - rank([perA; perB; glob])];
results{end+1,1} = 'nested: [rel] projection dims 4/2/3/2';
results{end,2}   = isequal(dims, [4 2 3 2]);


% --- Inner/per-event [rel] unit: dim 2 and per-event invariance ---
nspIn = struct('tags', [0 0 1 1], 'r', [2 2], 'sym', [true false], ...
               'rel', 'innermost');
dIn = buildExpTens({[0; 4; 7; 11]}, [], 50, 1, false, false, 0, ...
                   'nested', {nspIn}, 'verbose', false);
dInPE = buildExpTens({[10; 14; 107; 111]}, [], 50, 1, false, false, 0, ...
                     'nested', {nspIn}, 'verbose', false);  % per-event shift
% Interval change must register at a resolving sigma.
dInA = buildExpTens({[0; 4; 7; 11]}, [], 1, 1, false, false, 0, ...
                    'nested', {nspIn}, 'verbose', false);
dInB = buildExpTens({[0; 5; 7; 11]}, [], 1, 1, false, false, 0, ...
                    'nested', {nspIn}, 'verbose', false);
okInner = (dIn.dim == 2) ...
    && (abs(cosSimExpTens(dIn, dInPE, 'verbose', false) - 1) < 1e-9) ...
    && (cosSimExpTens(dInA, dInB, 'verbose', false) < 0.95);
results{end+1,1} = 'nested: inner [rel] dim 2 and per-event-invariant';
results{end,2}   = okInner;


% --- Inner equals tensor join of two is_rel dyads (eval parity) ---
dRef = buildExpTens({[0; 4], [7; 11]}, [], [50 50], [2 2], ...
                    [true true], [false false], [0 0], 'verbose', false);
okJoin = (dIn.dim == dRef.dim);
for q = {[4; 4], [4; -4], [0; 0], [3; 5], [-4; 4]}
    okJoin = okJoin && abs( ...
        evalExpTens(dIn,  q{1}, 'verbose', false) ...
      - evalExpTens(dRef, q{1}, 'verbose', false)) < 1e-11;
end
results{end+1,1} = 'nested: inner equals tensor-join of two is_rel dyads';
results{end,2}   = okJoin;


% --- rel=[1 1] subsumes to inner (warns; dim 2) ---
ws = warning('off', 'buildExpTens:nestedRelSubsumption');
dSub = buildExpTens({[0; 4; 7; 11]}, [], 50, 1, false, false, 0, ...
        'nested', {struct('tags', [0 0 1 1], 'r', [2 2], 'sym', [true false], ...
                          'rel', [1 1])}, 'verbose', false);
warning(ws);
results{end+1,1} = 'nested: rel=[1 1] subsumes to inner (dim 2)';
results{end,2}   = (dSub.dim == 2) ...
                   && (abs(cosSimExpTens(dIn, dSub, 'verbose', false) - 1) < 1e-9);


% --- Guards ---
okScalar = false;
try
    buildExpTens({[0; 7]}, [], 50, 1, false, false, 0, ...
        'nested', {struct('tags', [0 1], 'r', [1 2], 'sym', [true false], ...
                          'rel', true)}, 'verbose', false);
catch
    okScalar = true;
end
results{end+1,1} = 'nested: scalar/bool rel rejected for nested';
results{end,2}   = okScalar;


okUserRel = false;
try
    buildExpTens({[0; 7]}, [], 50, 1, true, false, 0, ...
        'nested', {struct('tags', [0 1], 'r', [1 2], 'sym', [true false])}, ...
        'verbose', false);
catch
    okUserRel = true;
end
results{end+1,1} = 'nested: user isRelVec on nested attribute rejected';
results{end,2}   = okUserRel;

okTags = false;
try
    buildExpTens({[0; 4]}, [], 50, 1, false, false, 0, ...
        'nested', {struct('tags', [0 0], 'r', [1 2], 'sym', [true false])}, ...
        'verbose', false);
catch
    okTags = true;
end
results{end+1,1} = 'nested: insufficient distinct source events errors';
results{end,2}   = okTags;


% --- Empty nested cell equals flat default ---
pAttr  = {[0 4; 7 11]};
d_def  = buildExpTens(pAttr, [], 50, 2, false, false, 0, 'verbose', false);
d_none = buildExpTens(pAttr, [], 50, 2, false, false, 0, ...
                      'nested', {[]}, 'verbose', false);
okflat = (d_def.dim == d_none.dim);
for qi = 1:numel(queries)
    if numel(queries{qi}) == 2
        okflat = okflat && abs( ...
            evalExpTens(d_def,  queries{qi}, 'verbose', false) ...
          - evalExpTens(d_none, queries{qi}, 'verbose', false)) < 1e-14;
    end
end
results{end+1,1} = 'nested: empty nested cell equals flat default';
results{end,2}   = okflat;


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
    fprintf('\n=== test_nested: %d passed, %d failed (of %d) ===\n\n', ...
        nPass, nFail, nPass + nFail);
    clear cleanupDefaults
    if nFail > 0
        error('test_nested:failed', '%d test(s) failed.', nFail);
    end
end
