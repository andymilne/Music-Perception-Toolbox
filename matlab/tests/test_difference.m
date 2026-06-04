%% test_difference.m — differenceEvents on the (pAttr, w, specs) carrier (3c-iv-b)
%
%  differenceEvents applies the k-th finite difference along the event axis,
%  slot-wise. It is well-defined exactly when slots have stable identity ---
%  an ordered attribute ([sym]=0) or a singleton (K=1) --- so a symmetric
%  multiset (K>1) errors, and the rule extends per level for a nested
%  attribute. The spec passes through unchanged (values change, structure
%  does not); NaN propagates (absent slot). With order = L the differenced-
%  then-bound and bound-then-differenced routes coincide (B o D == D o B).

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    clear cleanupDefaults
    cleanupDefaults = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end


% --- Carrier basics -------------------------------------------------

% Returns three-tuple with specs (flat: no tags field).
[pd, ~, sd] = differenceEvents({[0 2 5 9]}, [], 1);
results{end+1,1} = 'diff: returns specs (flat, no tags)';
results{end,2}   = iscell(sd) && numel(sd) == 1 && ~isfield(sd{1}, 'tags') ...
                   && isequal(pd{1}, [2 3 4]);

% Specs synthesised when none supplied (flatSpecs defaults).
[~, ~, sd] = differenceEvents({[0 2 5]}, [], 1);
results{end+1,1} = 'diff: synthesised flat spec has r=1 rel=false sym=true';
results{end,2}   = isequal(sd{1}.r, 1) && isequal(logical(sd{1}.rel), false) ...
                   && isequal(logical(sd{1}.sym), true);

% Spec passes through unchanged.
sIn = flatSpecs({zeros(1, 4)}, 'r', 2, 'rel', true, 'sym', false);
[~, ~, sOut] = differenceEvents({[0 2 5 9]}, [], 1, 'specs', sIn);
results{end+1,1} = 'diff: spec passes through unchanged';
results{end,2}   = isequal(sOut, sIn);

% Order 0 identity.
[pd, ~, ~] = differenceEvents({[60 62 64]}, [], 0);
results{end+1,1} = 'diff: order 0 identity';
results{end,2}   = isequal(pd{1}, [60 62 64]);

% Order 2 and leading-drop alignment (orders [2 0] -> N' = 3).
[pd, ~, ~] = differenceEvents({[0 1 4 9 16], [10 20 30 40 50]}, [], [2 0]);
results{end+1,1} = 'diff: order 2 + alignment';
results{end,2}   = isequal(size(pd{1}), [1 3]) && isequal(size(pd{2}), [1 3]) ...
                   && isequal(pd{1}, [2 2 2]) && isequal(pd{2}, [30 40 50]);

% Scalar order broadcasts to all attributes.
[pd, ~, ~] = differenceEvents({[0 2 5], [1 4 9]}, [], 1);
results{end+1,1} = 'diff: scalar order broadcasts';
results{end,2}   = isequal(size(pd{1}), [1 2]) && isequal(size(pd{2}), [1 2]);

% Weight rolling product (width 2).
[~, wd, ~] = differenceEvents({[0 1 2 3]}, {[1 2 3 4]}, 1);
results{end+1,1} = 'diff: weight rolling product';
results{end,2}   = isequal(wd{1}, [2 6 12]);


% --- K generalisation: ordered any-K, symmetric rejected ------------

% K>1 ordered attribute differences slot-wise (the lifted K=1 rule).
M = [0 2 5; 10 13 17];
[pd, ~, ~] = differenceEvents({M}, [], 1, 'specs', flatSpecs({M}, 'sym', false));
results{end+1,1} = 'diff: ordered K>1 differences slot-wise';
results{end,2}   = isequal(pd{1}, [2 3; 3 4]);

% Symmetric K>1 rejected.
results{end+1,1} = 'diff: symmetric K>1 rejected';
results{end,2}   = throwsError(@() differenceEvents({M}, [], 1, ...
                       'specs', flatSpecs({M}, 'sym', true)));

% Symmetric K>1 with order 0 never triggers the guard (identity).
[pd, ~, ~] = differenceEvents({M}, [], 0, 'specs', flatSpecs({M}, 'sym', true));
results{end+1,1} = 'diff: symmetric K>1 order 0 ok (identity)';
results{end,2}   = isequal(pd{1}, M);

% Singleton (K=1) is always differenceable regardless of sym.
[pd, ~, ~] = differenceEvents({[0 2 5]}, [], 1, ...
                 'specs', flatSpecs({[0 2 5]}, 'sym', true));
results{end+1,1} = 'diff: K=1 differences regardless of sym';
results{end,2}   = isequal(pd{1}, [2 3]);

% NaN propagates as an absent slot.
Mn = [0 2 5; 10 NaN 17];
[pd, ~, ~] = differenceEvents({Mn}, [], 1, 'specs', flatSpecs({Mn}, 'sym', false));
out = pd{1};
results{end+1,1} = 'diff: NaN propagates as absent slot';
results{end,2}   = isequal(out(1, :), [2 3]) && all(isnan(out(2, :)));


% --- Nested-D: difference a bound attribute -------------------------

raw = [0 2 5 9 14];
[pb, wb, specs] = bindEvents({raw}, [], 2, 1, false, true);   % nested, N'=4
[pnd, ~, snd] = differenceEvents(pb, wb, 1, 'specs', specs);
results{end+1,1} = 'diff: nested-D slot-wise, spec passthrough';
results{end,2}   = isequal(size(pnd{1}), [2 3]) ...
                   && isequal(snd{1}.tags(:).', [0 1]) ...
                   && isequal(snd{1}.r, specs{1}.r);

% A bag outer level (symOuter = true) cannot be differenced.
[pb2, wb2, specs2] = bindEvents({raw}, [], 2, 1, false, true, 'symOuter', true);
results{end+1,1} = 'diff: nested symmetric-outer rejected';
results{end,2}   = throwsError(@() differenceEvents(pb2, wb2, 1, 'specs', specs2));


% --- Commutation B o D == D o B -------------------------------------

P = [0 3 7 12 18];
L = 2;
[pD, wD, sD]    = differenceEvents({P}, [], 1);
[pDB, wDB, sDB] = bindEvents(pD, wD, L, 1, false, true);
[pB, wB, sB]    = bindEvents({P}, [], L, 1, false, true);
[pBD, wBD, sBD] = differenceEvents(pB, wB, 1, 'specs', sB);
results{end+1,1} = 'diff: B o D == D o B (values + specs)';
results{end,2}   = isequal(pDB{1}, pBD{1}) ...
                   && isequal(sDB{1}.tags, sBD{1}.tags) ...
                   && isequal(sDB{1}.r, sBD{1}.r) ...
                   && isequal(sDB{1}.sym, sBD{1}.sym) ...
                   && isequal(sDB{1}.rel, sBD{1}.rel);

% The two routes build eval-identical densities.
dDB = buildExpTens(pDB, wDB, 'specs', sDB, 'sigma', 30, ...
                   'isPer', false, 'period', 0, 'verbose', false);
dBD = buildExpTens(pBD, wBD, 'specs', sBD, 'sigma', 30, ...
                   'isPer', false, 'period', 0, 'verbose', false);
Qd  = (reshape(1:(dDB.dim * 4), dDB.dim, 4) - 6) / 2;
vDB = evalExpTens(dDB, Qd, 'verbose', false);
vBD = evalExpTens(dBD, Qd, 'verbose', false);
results{end+1,1} = 'diff: B o D == D o B (density eval-identical)';
results{end,2}   = (dDB.dim == dBD.dim) && max(abs(vDB(:) - vBD(:))) < 1e-12;


% --- Circular + errors + entropy parity -----------------------------

% Circular mode keeps N events; first difference wraps at the boundary.
[pd, ~, ~] = differenceEvents({[0 2 5 9]}, [], 1, 'circular', true);
results{end+1,1} = 'diff: circular keeps N';
results{end,2}   = isequal(size(pd{1}), [1 4]) && isequal(pd{1}, [0 - 9, 2, 3, 4]);

results{end+1,1} = 'diff: too-high order errors';
results{end,2}   = throwsError(@() differenceEvents({[1 2 3]}, [], 5));

results{end+1,1} = 'diff: circular order >= N errors';
results{end,2}   = throwsError(@() differenceEvents({[1 2 3]}, [], 3, ...
                       'circular', true));

results{end+1,1} = 'diff: wrong-length orders errors';
results{end,2}   = throwsError(@() differenceEvents({[1 2 3], [4 5 6]}, ...
                       [], [1 1 1]));

results{end+1,1} = 'diff: empty attribute errors';
results{end,2}   = throwsError(@() differenceEvents({zeros(0, 3)}, [], 1));

% Whole-tone scale: every cyclic 2-tuple of step sizes is (2,2), H = 0.
Hwt = nTupleEntropy([0 2 4 6 8 10], 12, 2, 'method', 'shannon');
results{end+1,1} = 'diff: n_tuple_entropy circular parity (whole-tone n=2 = 0)';
results{end,2}   = abs(Hwt) < 1e-12;


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
    fprintf('\n=== test_difference: %d passed, %d failed (of %d) ===\n\n', ...
        nPass, nFail, nPass + nFail);
    clear cleanupDefaults
    if nFail > 0
        error('test_difference:failed', '%d test(s) failed.', nFail);
    end
end
