%% test_translate.m — translateAttributes on the (pAttr, w, specs) carrier (3c-iv-d)
%
%  translateAttributes shifts attribute values by per-slot offsets. Everything
%  hangs off the slot axis (the rows of the value matrix, length K_total): the
%  offset is a value per slot held constant across the event axis (which makes
%  D(T(p)) == D(p)). A scalar broadcasts to all slots; a column is per-slot; a
%  row is a per-sweep global shift; a K_total x M matrix is per-slot x sweep.
%  is_rel is read per-attribute from specs: a uniform shift on an outermost-
%  relative attribute is a structural no-op (warns); a non-uniform offset
%  applies.

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

% Returns triple; single translation.
[pOut, wOut, sOut] = translateAttributes({[0 4 7]}, [], {5});
results{end+1,1} = 'translate: returns triple, single translation';
results{end,2}   = iscell(pOut) && numel(pOut) == 1 && isempty(wOut) ...
                   && iscell(sOut) && numel(sOut) == 1 ...
                   && isequal(pOut{1}, [5 9 12]);

% Scalar broadcasts to all slots.
[pOut, ~, ~] = translateAttributes({[0 4; 7 11]}, [], {10});
results{end+1,1} = 'translate: scalar broadcasts to all slots';
results{end,2}   = isequal(pOut{1}, [10 14; 17 21]);

% Per-slot offset (column).
[pOut, ~, ~] = translateAttributes({[0 4; 7 11]}, [], {[10; 20]});
results{end+1,1} = 'translate: per-slot offset (column)';
results{end,2}   = isequal(pOut{1}, [10 14; 27 31]);

% [] skips an attribute.
[pOut, ~, ~] = translateAttributes({[0 4], [1 2]}, [], {5, []});
results{end+1,1} = 'translate: [] skips attribute';
results{end,2}   = isequal(pOut{1}, [5 9]) && isequal(pOut{2}, [1 2]);

% NaN skips a slot.
[pOut, ~, ~] = translateAttributes({[0 4; 7 11]}, [], {[10; NaN]});
results{end+1,1} = 'translate: NaN skips a slot';
results{end,2}   = isequal(pOut{1}, [10 14; 7 11]);

% Weights pass through.
wIn = {[1 2 3]};
[~, wOut, ~] = translateAttributes({[0 4 7]}, wIn, {5});
results{end+1,1} = 'translate: weights pass through';
results{end,2}   = isequal(wOut, wIn);

% Does not mutate input.
pIn = {[0 4 7]}; orig = pIn{1};
translateAttributes(pIn, [], {5});
results{end+1,1} = 'translate: does not mutate input';
results{end,2}   = isequal(pIn{1}, orig);


% --- Sweep ----------------------------------------------------------

% (1 x M) row -> M copies, each a global shift broadcast across slots.
[pSweep, ~, ~] = translateAttributes({[0 4]}, [], {[0 5 12]});
results{end+1,1} = 'translate: sweep per-sweep scalar (row)';
results{end,2}   = iscell(pSweep) && numel(pSweep) == 3 ...
                   && isequal(pSweep{1}{1}, [0 4]) ...
                   && isequal(pSweep{2}{1}, [5 9]) ...
                   && isequal(pSweep{3}{1}, [12 16]);

% (K_total x M) matrix: slots down, sweep index across.
offs = [0 10; 0 20];                         % slot 0 then slot 1, over M=2
[pSweep, ~, ~] = translateAttributes({[0 4; 7 11]}, [], {offs});
results{end+1,1} = 'translate: sweep per-slot x sweep (matrix)';
results{end,2}   = numel(pSweep) == 2 ...
                   && isequal(pSweep{1}{1}, [0 4; 7 11]) ...
                   && isequal(pSweep{2}{1}, [10 14; 27 31]);

% Scalar/row entry broadcasts across the call's M (set by another attr).
[pSweep, ~, ~] = translateAttributes({[0 4], [1 2]}, [], {[0 10 20], 100});
okBroad = numel(pSweep) == 3;
for m = 1:3
    okBroad = okBroad && isequal(pSweep{m}{2}, [101 102]);
end
results{end+1,1} = 'translate: scalar broadcasts across sweep M';
results{end,2}   = okBroad;


% --- is_rel from specs: relative no-op ------------------------------

% Relative + uniform -> no-op + warn.
pR = {[0 4 7 11]};
lastwarn('', '');
[pOut, ~, ~] = translateAttributes(pR, [], {5}, 'specs', flatSpecs(pR, 'rel', true));
[~, wid] = lastwarn();
results{end+1,1} = 'translate: relative uniform is no-op + warns';
results{end,2}   = strcmp(wid, 'translateAttributes:noOp') && isequal(pOut{1}, pR{1});

% Relative + non-uniform -> applies, no warn.
pR2 = {[0 4; 7 11]};
lastwarn('', '');
[pOut, ~, ~] = translateAttributes(pR2, [], {[10; 20]}, 'specs', flatSpecs(pR2, 'rel', true));
[~, wid] = lastwarn();
results{end+1,1} = 'translate: relative non-uniform applies (no warn)';
results{end,2}   = isempty(wid) && isequal(pOut{1}, [10 14; 27 31]);

% Nested outermost-relative (relOuter) + uniform -> no-op + warn.
pN = {[0 4 7 11 2]};
[pb, ~, sb] = bindEvents(pN, [], 2, 'relOuter', true);   % rel = [0 1]
lastwarn('', '');
[pOut, ~, ~] = translateAttributes(pb, [], {3}, 'specs', sb);
[~, wid] = lastwarn();
results{end+1,1} = 'translate: nested outermost-relative uniform no-op';
results{end,2}   = strcmp(wid, 'translateAttributes:noOp') && isequal(pOut{1}, pb{1});

% Absolute attribute applies a uniform shift (no warn).
lastwarn('', '');
[pOut, ~, ~] = translateAttributes({[0 4 7]}, [], {5});   % specs [] -> absolute
[~, wid] = lastwarn();
results{end+1,1} = 'translate: absolute applies uniform (no warn)';
results{end,2}   = isempty(wid) && isequal(pOut{1}, [5 9 12]);


% --- Composition: D o T == D ----------------------------------------

P = [0 3 7 12];
[pT, ~, ~] = translateAttributes({P}, [], {100});
[dT, ~, ~]  = differenceEvents(pT, [], 1);
[d0, ~, ~]  = differenceEvents({P}, [], 1);
results{end+1,1} = 'translate: difference absorbs uniform translation (D o T == D)';
results{end,2}   = isequal(dT{1}, d0{1});


% --- Nested per-slot via the tag structure --------------------------

pIn2 = {[0 4; 7 11]};                          % K_inner=2, N=2
[pb2, ~, sb2] = bindEvents(pIn2, [], 2);       % L=2 -> K_total=4
[pOut, ~, ~] = translateAttributes(pb2, [], {[1; 2; 3; 4]}, 'specs', sb2);
results{end+1,1} = 'translate: nested per-slot addresses stacked slots';
results{end,2}   = isequal(pOut{1}(:, 1), pb2{1}(:, 1) + [1; 2; 3; 4]);


% --- Errors ---------------------------------------------------------

results{end+1,1} = 'translate: offsets wrong list length errors';
results{end,2}   = throwsError(@() translateAttributes({[0 4]}, [], {1, 2}));

results{end+1,1} = 'translate: per-slot wrong length errors';
results{end,2}   = throwsError(@() translateAttributes({[0 4; 7 11]}, [], {[1; 2; 3]}));

results{end+1,1} = 'translate: Inf rejected';
results{end,2}   = throwsError(@() translateAttributes({[0 4]}, [], {Inf}));

results{end+1,1} = 'translate: sweep M mismatch errors';
results{end,2}   = throwsError(@() translateAttributes({[0 4], [1 2]}, [], ...
                       {[0 1 2], [0 1]}));


% --- Integration: sweep -> build -> cosine self-match ---------------

chord = [0; 4; 7];                             % K=3 chord, N=1
grid  = [-200 -100 0 100 200];                 % row, M=5
[sweep, ~, specs] = translateAttributes({chord}, [], {grid});
ref = buildExpTens({chord}, [], 'specs', specs, 'sigma', 30, ...
                   'isPer', false, 'period', 0, 'verbose', false);
sims = zeros(1, numel(sweep));
for m = 1:numel(sweep)
    d = buildExpTens(sweep{m}, [], 'specs', specs, 'sigma', 30, ...
                     'isPer', false, 'period', 0, 'verbose', false);
    sims(m) = cosSimExpTens(ref, d, 'verbose', false);
end
[~, peakIdx] = max(sims);
results{end+1,1} = 'translate: sweep feeds cosine self-match peak';
results{end,2}   = peakIdx == 3 && abs(sims(3) - 1) < 1e-9;   % offset 0 at index 3


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
    fprintf('\n=== test_translate: %d passed, %d failed (of %d) ===\n\n', ...
        nPass, nFail, nPass + nFail);
    clear cleanupDefaults
    if nFail > 0
        error('test_translate:failed', '%d test(s) failed.', nFail);
    end
end
