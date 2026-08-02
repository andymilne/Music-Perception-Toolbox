%% test_nesting_L3.m — deep nesting (L >= 3): matrix tags + L-general build
%
%  A nested attribute is a flat K_total x N value matrix plus a
%  (K_total, L-1) integer tags matrix (one grouping column per level,
%  innermost-grouping first) and per-level r/sym/rel vectors. These
%  exercise the L-general enumeration and the absolute and outer
%  (global-transposition) projections at three levels; the per-group
%  inner/intermediate reductions at L >= 3 are a later step and must defer.

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    clear cleanupDefaults
    cleanupDefaults = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end

% Three-level attribute: 2 bars x 2 chords/bar x 2 notes/chord = 8 values.
% tags columns: 1 = chord (finest grouping above leaves), 2 = bar (outermost).
tags3 = [0 0; 0 0; 1 0; 1 0; 2 1; 2 1; 3 1; 3 1];     % 8 x 2
P3    = {[0 4 7 11 12 16 19 23].'};                   % K_total = 8, N = 1

specAbs = struct('tags', tags3, 'r', [2 2 2], 'sym', [true true false]);
specOut = struct('tags', tags3, 'r', [2 2 2], 'sym', [true true false], ...
                 'rel', [0 0 1]);
specOutS = struct('tags', tags3, 'r', [2 2 2], 'sym', [true true false], ...
                  'rel', 'outermost');

bkw = {'sigma', 30, 'isPer', false, 'period', 0, 'verbose', false};

dA  = buildExpTens(P3, [], 'specs', {specAbs},  bkw{:});
dO  = buildExpTens(P3, [], 'specs', {specOut},  bkw{:});
dOS = buildExpTens(P3, [], 'specs', {specOutS}, bkw{:});


% --- Dimensions ---
results{end+1,1} = 'L3: absolute dim = full D (8)';
results{end,2}   = dA.dim == 8;

results{end+1,1} = 'L3: outer dim = D - 1 (7)';
results{end,2}   = dO.dim == 7;

results{end+1,1} = 'L3: outermost string matches rel vector';
results{end,2}   = dOS.dim == dO.dim && dO.dim == 7;


% --- Cosine self-match ---
results{end+1,1} = 'L3: absolute cosine self-match = 1';
results{end,2}   = abs(cosSimExpTens(dA, dA, 'verbose', false) - 1) < 1e-9;

results{end+1,1} = 'L3: outer cosine self-match = 1';
results{end,2}   = abs(cosSimExpTens(dO, dO, 'verbose', false) - 1) < 1e-9;


% --- Global-transposition invariance (outermost unit) ---
dOT = buildExpTens({P3{1} + 5}, [], 'specs', {specOut}, bkw{:});
results{end+1,1} = 'L3: outer is global-transposition invariant';
results{end,2}   = abs(cosSimExpTens(dO, dOT, 'verbose', false) - 1) < 1e-9;

dAT = buildExpTens({P3{1} + 5}, [], 'specs', {specAbs}, bkw{:});
results{end+1,1} = 'L3: absolute is NOT transposition invariant';
results{end,2}   = cosSimExpTens(dA, dAT, 'verbose', false) < 0.999;


% --- inner / intermediate per-group reduction at L = 3 ---
% Co-transposition at unit u removes each level-u sub-tuple's all-ones:
% inner (u=1) per-chord, intermediate (u=2) per-bar, outer (u=3) global.
% dim = D - G_u with G_u = prod(r(u+1:end)). Each unit is invariant to
% transposition at its own level and coarser, not finer.

% Per-position offsets: per-chord (tags col 1), per-bar (col 2), global.
perChord = P3{1} + [0; 0; 60; 60; 0; 0; 60; 60];
perBar   = P3{1} + [10; 10; 10; 10; 20; 20; 20; 20];
glob     = P3{1} + 5;

specIn  = struct('tags', tags3, 'r', [2 2 2], 'sym', [true true false], ...
                 'rel', [1 0 0]);
specMid = struct('tags', tags3, 'r', [2 2 2], 'sym', [true true false], ...
                 'rel', [0 1 0]);

dIn  = buildExpTens(P3, [], 'specs', {specIn},  bkw{:});
dMid = buildExpTens(P3, [], 'specs', {specMid}, bkw{:});

% inner: dim 4, self-match, per-chord transposition invariant
dInC = buildExpTens({perChord}, [], 'specs', {specIn}, bkw{:});
results{end+1,1} = 'L3: inner dim = D - G0 (4)';
results{end,2}   = dIn.dim == 4;
results{end+1,1} = 'L3: inner cosine self-match = 1';
results{end,2}   = abs(cosSimExpTens(dIn, dIn, 'verbose', false) - 1) < 1e-9;
results{end+1,1} = 'L3: inner is per-chord transposition invariant';
results{end,2}   = abs(cosSimExpTens(dIn, dInC, 'verbose', false) - 1) < 1e-6;

% intermediate: dim 6, self-match, per-bar invariant, per-chord NOT
dMidB = buildExpTens({perBar},   [], 'specs', {specMid}, bkw{:});
dMidC = buildExpTens({perChord}, [], 'specs', {specMid}, bkw{:});
results{end+1,1} = 'L3: intermediate dim = D - G1 (6)';
results{end,2}   = dMid.dim == 6;
results{end+1,1} = 'L3: intermediate cosine self-match = 1';
results{end,2}   = abs(cosSimExpTens(dMid, dMid, 'verbose', false) - 1) < 1e-9;
results{end+1,1} = 'L3: intermediate is per-bar transposition invariant';
results{end,2}   = abs(cosSimExpTens(dMid, dMidB, 'verbose', false) - 1) < 1e-6;
results{end+1,1} = 'L3: intermediate NOT invariant to finer (per-chord)';
results{end,2}   = cosSimExpTens(dMid, dMidC, 'verbose', false) < 0.5;

% Unit dims strictly increase inner -> intermediate -> outer -> absolute
results{end+1,1} = 'L3: unit dims increase 4 < 6 < 7 < 8';
results{end,2}   = isequal([dIn.dim, dMid.dim, dO.dim, dA.dim], [4 6 7 8]);


% --- renyi2 entropy at L = 3 (nested block-metric inner matrix) ---
% Was a MemoryError via the flat S_8 = 40320 orbit; now the per-attribute
% inner matrix is built numerically from the nested tuples and block
% metric. Values are golden against the Python implementation (base 2).
hIn  = entropyExpTens(dIn,  'method', 'renyi2', 'verbose', false);
hMid = entropyExpTens(dMid, 'method', 'renyi2', 'verbose', false);
hOut = entropyExpTens(dO,   'method', 'renyi2', 'verbose', false);
hAbs = entropyExpTens(dA,   'method', 'renyi2', 'verbose', false);

results{end+1,1} = 'L3: renyi2 finite for all four projections';
results{end,2}   = all(isfinite([hIn, hMid, hOut, hAbs]));

results{end+1,1} = 'L3: renyi2 decreases with finer quotient';
results{end,2}   = (hIn < hMid) && (hMid < hOut) && (hOut < hAbs);

results{end+1,1} = 'L3: renyi2 matches Python golden (inner)';
results{end,2}   = abs(hIn  - 28.956146) < 1e-3;
results{end+1,1} = 'L3: renyi2 matches Python golden (intermediate)';
results{end,2}   = abs(hMid - 42.498901) < 1e-3;
results{end+1,1} = 'L3: renyi2 matches Python golden (outer)';
results{end,2}   = abs(hOut - 48.731539) < 1e-3;
results{end+1,1} = 'L3: renyi2 matches Python golden (absolute)';
results{end,2}   = abs(hAbs - 53.964178) < 1e-3;


% --- Representation validation ---
specVecTags = struct('tags', zeros(1, 8), 'r', [2 2 2], ...
                     'sym', [true true false]);
results{end+1,1} = 'L3: 1-D tags rejected for deep nesting';
results{end,2}   = throwsErrorWithId( ...
    @() buildExpTens(P3, [], 'specs', {specVecTags}, bkw{:}), ...
    'buildExpTens:nestedTags');

specBadShape = struct('tags', [tags3, tags3(:, 1)], 'r', [2 2 2], ...
                      'sym', [true true false]);    % 8 x 3, need 8 x 2
results{end+1,1} = 'L3: tag matrix wrong shape rejected';
results{end,2}   = throwsErrorWithId( ...
    @() buildExpTens(P3, [], 'specs', {specBadShape}, bkw{:}), ...
    'buildExpTens:nestedTags');

specScalarRel = struct('tags', tags3, 'r', [2 2 2], ...
                       'sym', [true true false], 'rel', true);
results{end+1,1} = 'L3: scalar rel rejected when nested';
results{end,2}   = throwsErrorWithId( ...
    @() buildExpTens(P3, [], 'specs', {specScalarRel}, bkw{:}), ...
    'buildExpTens:nestedRelScalar');

specInfeasible = struct('tags', tags3, 'r', [2 2 3], ...
                        'sym', [true true false]);  % 3 bars, only 2 present
results{end+1,1} = 'L3: infeasible read errors';
results{end,2}   = throwsErrorWithId( ...
    @() buildExpTens(P3, [], 'specs', {specInfeasible}, bkw{:}), ...
    'buildExpTens:nestedInfeasible');


% --- bind_events deepening: flat -> L=2 -> L=3 ---
% bindEvents deepens an already-nested attribute by tiling the existing
% tag columns and appending a new outermost grouping level; r/sym/rel
% extend by the bound outer level.
pf = {[0 2 4 5 7 9]};                        % flat, K=1, N=6
[p1, w1, s1] = bindEvents(pf, [], 2);        % -> L=2
[p2, w2, s2] = bindEvents(p1, w1, 2, 'specs', s1);   % -> L=3
sp = s2{1};
results{end+1,1} = 'L3 bind: deepened tags is (4,2)';
results{end,2}   = isequal(size(sp.tags), [4 2]);
results{end+1,1} = 'L3 bind: deepened r = [1 2 2]';
results{end,2}   = isequal(sp.r(:).', [1 2 2]) && numel(sp.sym) == 3 ...
                   && numel(sp.rel) == 3;
results{end+1,1} = 'L3 bind: new outermost column = [0 0 1 1]';
results{end,2}   = isequal(sp.tags(:, 2).', [0 0 1 1]) ...
                   && isequal(sp.tags(:, 1).', [0 1 0 1]);

dB = buildExpTens(p2, w2, 'specs', s2, bkw{:});
results{end+1,1} = 'L3 bind: deepened density builds (dim 4)';
results{end,2}   = dB.dim == 4;
results{end+1,1} = 'L3 bind: deepened cosine self-match = 1';
results{end,2}   = abs(cosSimExpTens(dB, dB, 'verbose', false) - 1) < 1e-9;

[p2r, w2r, s2r] = bindEvents(p1, w1, 2, 'specs', s1, 'relOuter', true);
dR  = buildExpTens(p2r, w2r, 'specs', s2r, bkw{:});
dRT = buildExpTens({p2r{1} + 5}, w2r, 'specs', s2r, bkw{:});
results{end+1,1} = 'L3 bind: relOuter deepen -> rel [0 0 1], outer dim 3';
results{end,2}   = isequal(s2r{1}.rel(:).', [0 0 1]) && dR.dim == 3;
results{end+1,1} = 'L3 bind: relOuter deepen is global-transposition invariant';
results{end,2}   = abs(cosSimExpTens(dR, dRT, 'verbose', false) - 1) < 1e-6;

results{end+1,1} = 'L3 bind: levelNames rejected when deepening';
results{end,2}   = throwsErrorWithId( ...
    @() bindEvents(p1, w1, 2, 'specs', s1, 'levelNames', {'x','y','z'}), ...
    'bindEvents:levelNamesNested');


if standalone
    nPass = sum(cell2mat(results(:, 2)));
    fprintf('test_nesting_L3: %d/%d passed\n', nPass, size(results, 1));
end
