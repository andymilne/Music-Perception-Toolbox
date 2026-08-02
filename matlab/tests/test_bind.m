%% test_bind.m — triple-form bindEvents (3c-ii / 3c-iv-c)
%
%  bindEvents nests sliding windows of consecutive events into a single
%  nested attribute per input attribute (toolbox spec §6.1/§6.5): the bound
%  events form an ordered outer level (symOuter = 0 by default), each
%  event's own multiset is the inner level. The inner level's geometry
%  (r/rel/sym) is read from the incoming triple's specs (flatSpecs defaults
%  when specs is []). It returns {pAttrBound, wBound, specs} ready for
%  buildExpTens(..., 'specs', specs). L = 1 is a flat passthrough. Outer
%  r = L with rel = [relIn, 0] reproduces the old separate-attribute tensor
%  join (§6.5).

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    clear cleanupDefaults
    cleanupDefaults = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end


% --- Returns three-tuple with specs (nested has tags) ---
[~, ~, sp] = bindEvents({[0 4 7 11 2]}, [], 2);
results{end+1,1} = 'bind: returns specs (nested has tags)';
results{end,2}   = iscell(sp) && numel(sp) == 1 && isfield(sp{1}, 'tags');


% --- L = 1 passes the incoming flat spec through (no tags) ---
p1 = {[0 4 7 11 2]};
[pb1, ~, sp1] = bindEvents(p1, [], 1, 'specs', flatSpecs(p1, 'r', 3, ...
                                                         'rel', true, 'sym', true));
okFlat = ~isfield(sp1{1}, 'tags') && sp1{1}.r == 3 && sp1{1}.rel == true ...
         && sp1{1}.sym == true && isequal(pb1{1}, p1{1});
results{end+1,1} = 'bind: L=1 flat passthrough';
results{end,2}   = okFlat;


% --- Inner geometry read from incoming spec (inner inherits; outer r=L) ---
pN = {[0 4 7; 10 12 14]};   % K=2, N=3
[pbN, ~, spN] = bindEvents(pN, [], 2, 'specs', flatSpecs(pN, 'r', 2, ...
                                                         'rel', true, 'sym', true));
s = spN{1};
okNest = isequal(s.r, [2 2]) && isequal(s.sym, [true false]) ...
         && isequal(logical(s.rel), [true false]) ...
         && isequal(s.tags, [0 0 1 1]) && isequal(size(pbN{1}), [4 2]);
results{end+1,1} = 'bind: inner geometry read from specs';
results{end,2}   = okNest;


% --- Synthesised specs (specs []) give flat-default inner geometry ---
pR = {[0 4 7 11]};
[~, ~, sSyn] = bindEvents(pR, [], 2);
results{end+1,1} = 'bind: synthesised specs default inner geometry';
results{end,2}   = isequal(sSyn{1}.r, [1 2]) ...
                   && isequal(sSyn{1}.sym, [true false]) ...
                   && isequal(logical(sSyn{1}.rel), [false false]);


% --- rel default [relIn, 0]; absolute -> [0 0] ---
[~, ~, sAbs] = bindEvents(pR, [], 2);
[~, ~, sRel] = bindEvents(pR, [], 2, 'specs', flatSpecs(pR, 'rel', true));
results{end+1,1} = 'bind: rel default [relIn,0]';
results{end,2}   = isequal(logical(sAbs{1}.rel), [false false]) ...
                   && isequal(logical(sRel{1}.rel), [true false]);


% --- relOuter -> [0 1] (global-transposition quotient) ---
[~, ~, sRO] = bindEvents(pR, [], 2, 'relOuter', true);
results{end+1,1} = 'bind: relOuter gives [0 1]';
results{end,2}   = isequal(logical(sRO{1}.rel), [false true]);


% --- symOuter adjustable ---
[~, ~, sS0] = bindEvents(pR, [], 2);
[~, ~, sS1] = bindEvents(pR, [], 2, 'symOuter', true);
results{end+1,1} = 'bind: symOuter adjustable';
results{end,2}   = isequal(sS0{1}.sym, [true false]) ...
                   && isequal(sS1{1}.sym, [true true]);


% --- Reproduces old tensor join (eval parity, §6.5) ---
diffs = [2 -1 3 0 -2 1 4];
n = 3;
[pb, wb, specs] = bindEvents({diffs}, [], n, 'circular', true);
dNew = buildExpTens(pb, wb, 'specs', specs, 'sigma', 10, 'isPer', true, ...
                    'period', 12, 'verbose', false);
pOld = cell(1, n);
for ell = 0:(n - 1)
    idx = mod((0:6) + ell, 7) + 1;
    pOld{ell + 1} = diffs(idx);
end
dOld = buildExpTens(pOld, [], 10 * ones(1, n), ones(1, n), false(1, n), ...
                    true(1, n), 12 * ones(1, n), 'verbose', false);
Q = [1 2 -3; 0 -1 2; 3 1 -2];   % n x 3 queries
vNew = evalExpTens(dNew, Q, 'verbose', false);
vOld = evalExpTens(dOld, Q, 'verbose', false);
results{end+1,1} = 'bind: reproduces old tensor join (eval parity)';
results{end,2}   = (dNew.dim == dOld.dim) && (dNew.dim == n) ...
                   && max(abs(vNew(:) - vOld(:))) < 1e-12;


% --- circular vs non-circular sizes ---
pC = {[0 4 7 11 2]};   % N=5
[pbNC, ~, ~] = bindEvents(pC, [], 2, 'circular', false);
[pbC,  ~, ~] = bindEvents(pC, [], 2, 'circular', true);
results{end+1,1} = 'bind: circular/non-circular N''';
results{end,2}   = isequal(size(pbNC{1}), [2 4]) && isequal(size(pbC{1}), [2 5]);


% --- Per-attribute orders + alignment (smaller L keeps leading N') ---
pPA = {[0 1 2 3 4], [10 11 12 13 14]};
[pbPA, ~, spPA] = bindEvents(pPA, [], [1 3]);
nPrime = 5 - 3 + 1;
results{end+1,1} = 'bind: per-attribute orders + alignment';
results{end,2}   = ~isfield(spPA{1}, 'tags') && isfield(spPA{2}, 'tags') ...
                   && isequal(size(pbPA{1}), [1 nPrime]) ...
                   && isequal(size(pbPA{2}), [3 nPrime]) ...
                   && isequal(pbPA{1}, pPA{1}(:, 1:nPrime));


% --- K_a > 1: tags repeat per event block ---
pK = {[0 7; 4 11]};   % K=2, N=2
[pbK, ~, spK] = bindEvents(pK, [], 2, 'specs', flatSpecs(pK, 'r', 2));
results{end+1,1} = 'bind: K_a>1 tags repeat per block';
results{end,2}   = isequal(spK{1}.tags, [0 0 1 1]) && isequal(size(pbK{1}), [4 1]);


% --- Event-dependent weight stacks ---
pW = {[0 4 7 11]};
wW = {[1 2 3 4]};
[~, wbW, ~] = bindEvents(pW, wW, 2);
results{end+1,1} = 'bind: event-dependent weight stacks';
results{end,2}   = isequal(wbW{1}, [1 2 3; 2 3 4]);


% --- Scalar weight passes through ---
[~, wbS, ~] = bindEvents(pW, 0.7, 2);
results{end+1,1} = 'bind: scalar weight passes through';
results{end,2}   = isequal(wbS, 0.7);


% --- Nested input spec rejected (L>=3 deep nesting not yet supported) ---
[~, ~, spNested] = bindEvents(pW, [], 2);
results{end+1,1} = 'bind: nested input spec rejected';
results{end,2}   = throwsError(@() bindEvents(pW, [], 2, 'specs', spNested));


% --- name from kwarg and inherited from incoming spec ---
[~, ~, spNm] = bindEvents(pW, [], 2, ...
                          'name', 'steps', 'levelNames', {'step', 'ngram'});
[~, ~, spInh] = bindEvents(pW, [], 2, 'specs', flatSpecs(pW, 'name', 'pitch'));
results{end+1,1} = 'bind: name from kwarg and inherited';
results{end,2}   = strcmp(spNm{1}.name, 'steps') ...
                   && isequal(spNm{1}.names, {'step', 'ngram'}) ...
                   && strcmp(spInh{1}.name, 'pitch');


% --- Invalid orders error ---
okErr = false(1, 3);
try; bindEvents(pW, [], 0);   catch; okErr(1) = true; end
try; bindEvents(pW, [], 1.5); catch; okErr(2) = true; end
try; bindEvents(pW, [], 5);   catch; okErr(3) = true; end
results{end+1,1} = 'bind: invalid orders error';
results{end,2}   = all(okErr);


% --- specs wrong length errors ---
pTwo = {[0 4 7 11], [1 2 3 4]};
results{end+1,1} = 'bind: specs wrong length errors';
results{end,2}   = throwsError(@() bindEvents(pTwo, [], 2, ...
                       'specs', {flatSpecs({[0 4 7 11]})}));


% --- nTupleEntropy still works (migrated caller) ---
pE = [0 2 5 7 9 11 4 6];
h1 = nTupleEntropy(pE, 12, 1, 'sigma', 20, 'method', 'shannon');
h2 = nTupleEntropy(pE, 12, 2, 'sigma', 20, 'method', 'shannon');
results{end+1,1} = 'bind: nTupleEntropy still works (h2 > h1, finite)';
results{end,2}   = isfinite(h1) && isfinite(h2) && (h2 > h1);


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
    fprintf('\n=== test_bind: %d passed, %d failed (of %d) ===\n\n', ...
        nPass, nFail, nPass + nFail);
    clear cleanupDefaults
    if nFail > 0
        error('test_bind:failed', '%d test(s) failed.', nFail);
    end
end
