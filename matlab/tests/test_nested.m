%% test_nested.m — nested binding (representation B) in the MAET build core
%
%  Mirror of the Python tests/test_nested.py. Covers the toolbox
%  specification §6/§7.1/§10 for the two-level nested attribute,
%  isRel = absolute only (the [rel] co-transposition-unit selector is a
%  later step):
%
%   * Outer r = K reproduces the old separate-attribute binding (tensor
%     join): a single nested attribute read at the whole-tuple level
%     evaluates identically to the equivalent flat tensor-joined density.
%   * Within-event / across-event partial symmetry with symOuter = 0:
%     inner slots orbit within each source event, but bound events keep
%     sequence order (no cross-tag interleaving).
%   * Pooled within-source reading (outer r < K) sums per-event
%     sub-tuples into one shared lower-dimensional space.
%   * The §6.3 inner/outer [rel] projection-rank dims (4/2/3/2 for two
%     dyads) — the geometric target the [rel] selector step must hit.
%   * isRel on a nested attribute errors until the selector lands;
%     insufficient distinct source events errors.
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


% ---------------------------------------------------------------------
%  Outer r = K reproduces old separate-attribute binding
% ---------------------------------------------------------------------

d_old = buildExpTens({[0], [7]}, [], [50 50], [1 1], ...
                     [false false], [false false], [0 0], 'verbose', false);
nspec1 = struct('rInner', 1, 'rOuter', 2, ...
                'symInner', true, 'symOuter', false, 'tags', [0 1]);
d_nest = buildExpTens({[0; 7]}, [], 50, 1, false, false, 0, ...
                      'nested', {nspec1}, 'verbose', false);
okp = (d_old.dim == 2) && (d_nest.dim == 2);
queries = {[0; 7], [0; 0], [3; 7], [-5; 12]};
for qi = 1:numel(queries)
    vo = evalExpTens(d_old,  queries{qi}, 'verbose', false);
    vn = evalExpTens(d_nest, queries{qi}, 'verbose', false);
    okp = okp && abs(vo - vn) < 1e-12;
end
results{end+1,1} = 'nested: outer r=K reproduces old separate-attr binding';
results{end,2}   = okp;


% ---------------------------------------------------------------------
%  Inner S_2 orbit, outer order preserved (no cross-tag interleaving)
% ---------------------------------------------------------------------

nspec2 = struct('rInner', 2, 'rOuter', 2, ...
                'symInner', true, 'symOuter', false, 'tags', [0 0 1 1]);
d2 = internal.ensureExpTensExpensive( ...
    buildExpTens({[0; 4; 7; 11]}, [], 50, 1, false, false, 0, ...
                 'nested', {nspec2}, 'verbose', false));
C2   = d2.Centres{1};                 % 4 x M
cset = unique(C2.', 'rows');          % each row a centre
expected = sortrows([0 4 7 11; 0 4 11 7; 4 0 7 11; 4 0 11 7]);
okset = isequal(sortrows(cset), expected);
nointer = true;
for rr = 1:size(cset, 1)
    nointer = nointer ...
        && isequal(sort(cset(rr, 1:2)), [0 4]) ...
        && isequal(sort(cset(rr, 3:4)), [7 11]);
end
results{end+1,1} = 'nested: inner S2 orbit, outer order preserved (no interleave)';
results{end,2}   = okset && nointer && (d2.dim == 4);


% ---------------------------------------------------------------------
%  Pooled within-source reading (outer r < K)
% ---------------------------------------------------------------------

nspec3 = struct('rInner', 2, 'rOuter', 1, ...
                'symInner', false, 'symOuter', false, 'tags', [0 0 1 1]);
d3 = internal.ensureExpTensExpensive( ...
    buildExpTens({[0; 4; 7; 11]}, [], 50, 1, false, false, 0, ...
                 'nested', {nspec3}, 'verbose', false));
cset3 = sortrows(unique(d3.Centres{1}.', 'rows'));
results{end+1,1} = 'nested: pooled outer r<K (shared 2-D space)';
results{end,2}   = isequal(cset3, sortrows([0 4; 7 11])) && (d3.dim == 2);


% ---------------------------------------------------------------------
%  §6.3 inner/outer [rel] projection-rank dims (geometric target)
% ---------------------------------------------------------------------

perA = [1 1 0 0]; perB = [0 0 1 1]; glob = [1 1 1 1];
dimAbs   = 4;
dimInner = 4 - rank([perA; perB]);
dimOuter = 4 - rank(glob);
dimBoth  = 4 - rank([perA; perB; glob]);
results{end+1,1} = 'nested: [rel] projection dims 4/2/3/2';
results{end,2}   = isequal([dimAbs dimInner dimOuter dimBoth], [4 2 3 2]);


% ---------------------------------------------------------------------
%  Guards: isRel deferred; insufficient distinct tags errors
% ---------------------------------------------------------------------

okRel = false;
try
    buildExpTens({[0; 7]}, [], 50, 1, true, false, 0, ...
                 'nested', {nspec1}, 'verbose', false);
catch
    okRel = true;
end
results{end+1,1} = 'nested: isRel on nested attribute errors (deferred)';
results{end,2}   = okRel;

nspecBad = struct('rInner', 1, 'rOuter', 2, ...
                  'symInner', true, 'symOuter', false, 'tags', [0 0]);
okErr = false;
try
    buildExpTens({[0; 4]}, [], 50, 1, false, false, 0, ...
                 'nested', {nspecBad}, 'verbose', false);
catch
    okErr = true;
end
results{end+1,1} = 'nested: insufficient distinct source events errors';
results{end,2}   = okErr;


% ---------------------------------------------------------------------
%  Empty nested cell equals no nested (flat path untouched)
% ---------------------------------------------------------------------

pAttr = {[0 4; 7 11]};
d_def  = buildExpTens(pAttr, [], 50, 2, false, false, 0, 'verbose', false);
d_none = buildExpTens(pAttr, [], 50, 2, false, false, 0, ...
                      'nested', {[]}, 'verbose', false);
okflat = (d_def.dim == d_none.dim);
for qi = 1:numel(queries)
    if numel(queries{qi}) == 2
        okflat = okflat && abs( ...
            evalExpTens(d_def, queries{qi}, 'verbose', false) ...
            - evalExpTens(d_none, queries{qi}, 'verbose', false)) < 1e-14;
    end
end
results{end+1,1} = 'nested: empty nested cell equals flat default';
results{end,2}   = okflat;


% ---------------------------------------------------------------------
%  Standalone summary
% ---------------------------------------------------------------------

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
