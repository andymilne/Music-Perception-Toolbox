%% test_entropy.m — entropyExpTens — single multiset Shannon and Rényi-2
%
%  Tests for single multiset Shannon and Rényi-2.
%
%  Standalone-runnable; appends to `results` when called from
%  test_mpt.m.

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    clear cleanupDefaults
    cleanupDefaults = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end



H = nTupleEntropy([0, 2, 4, 6, 8, 10], 12);
results{end+1,1} = 'nTupleEntropy: whole-tone = 0';
results{end,2}   = abs(H) < 1e-10;

H = nTupleEntropy([0, 2, 4, 5, 7, 9, 11], 12, 2, 'method', 'shannon');
results{end+1,1} = 'nTupleEntropy: diatonic 2-tuple ≈ 1.56';
results{end,2}   = abs(H - 1.56) < 0.01;

H_raw = nTupleEntropy([0, 2, 4, 5, 7, 9, 11], 12, 1);
H_smooth = nTupleEntropy([0, 2, 4, 5, 7, 9, 11], 12, 1, 'sigma', 0.2);
results{end+1,1} = 'nTupleEntropy: smoothing increases H';
results{end,2}   = H_smooth > H_raw;

H = entropyExpTens(0:11, ones(1,12), 100, 1, false, true, 12, ...
    'method', 'normalized', 'nPointsPerDim', 1200, 'verbose', false);
results{end+1,1} = 'entropyExpTens normalized: uniform ≈ 1';
results{end,2}   = H > 0.95;

% --- v3 unified dispatch: entropyExpTens list and batched-raw modes ----

% List mode: cell of density structs returns cell of entropy values
de1 = buildExpTens([0, 4, 7], [], 50, 1, false, true, 1200, 'verbose', false);
de2 = buildExpTens([0, 3, 7], [], 50, 1, false, true, 1200, 'verbose', false);
HCell = entropyExpTens({de1, de2}, 'nPointsPerDim', 1200, 'verbose', false);
results{end+1,1} = 'entropyExpTens list: returns cell of correct length';
results{end,2}   = iscell(HCell) && numel(HCell) == 2;
H1 = entropyExpTens(de1, 'nPointsPerDim', 1200, 'verbose', false);
H2 = entropyExpTens(de2, 'nPointsPerDim', 1200, 'verbose', false);
results{end+1,1} = 'entropyExpTens list: matches scalar dispatch element-wise';
results{end,2}   = abs(HCell{1} - H1) < 1e-14 ...
                   && abs(HCell{2} - H2) < 1e-14;

% List mode: Option II (length-1 stays length-1)
HCell1 = entropyExpTens({de1}, 'nPointsPerDim', 1200, 'verbose', false);
results{end+1,1} = 'entropyExpTens list: length-1 returns length-1 cell';
results{end,2}   = iscell(HCell1) && numel(HCell1) == 1;

% List mode: name-value pairs forwarded
HCellNorm = entropyExpTens({de1, de2}, 'method', 'shannon', 'base', exp(1), 'nPointsPerDim', 1200, 'verbose', false);
H1_nat = entropyExpTens(de1, 'method', 'shannon', 'base', exp(1), 'nPointsPerDim', 1200, 'verbose', false);
results{end+1,1} = 'entropyExpTens list: name-value pairs forwarded';
results{end,2}   = abs(HCellNorm{1} - H1_nat) < 1e-12;

% List mode: non-struct entry errors
results{end+1,1} = 'entropyExpTens list: non-struct entry errors';
results{end,2}   = throwsErrorWithId( ...
    @() entropyExpTens({de1, [1, 2, 3]}, 'verbose', false), ...
    'entropyExpTens:listNonStruct');

% Batched-raw mode: 2-D pitch matrix returns vector of entropies
P_h = [0, 4, 7; 0, 3, 7];
H_batched = entropyExpTens(P_h, [], 50, 1, false, true, 1200, 'nPointsPerDim', 1200, 'verbose', false);
results{end+1,1} = 'entropyExpTens batched: returns vector of correct length';
results{end,2}   = isnumeric(H_batched) && numel(H_batched) == 2;

% Batched-raw matches scalar dispatch row-by-row
H_row1 = entropyExpTens(P_h(1, :), [], 50, 1, false, true, 1200, 'nPointsPerDim', 1200, 'verbose', false);
H_row2 = entropyExpTens(P_h(2, :), [], 50, 1, false, true, 1200, 'nPointsPerDim', 1200, 'verbose', false);
results{end+1,1} = 'entropyExpTens batched: matches scalar dispatch row-by-row';
results{end,2}   = abs(H_batched(1) - H_row1) < 1e-12 ...
                   && abs(H_batched(2) - H_row2) < 1e-12;

% Batched-raw: NaN-padded rows
P_h_nan = [0, 4, 7, NaN; 0, 3, 7, NaN];
H_batched_nan = entropyExpTens(P_h_nan, [], 50, 1, false, true, 1200, 'nPointsPerDim', 1200, 'verbose', false);
results{end+1,1} = 'entropyExpTens batched: NaN-padded rows match unpadded';
results{end,2}   = max(abs(H_batched_nan - H_batched)) < 1e-14;

% Batched-raw: insufficient pitches in a row gives NaN
P_h_short = [0, 4, 7; 0, NaN, NaN];   % second row has only 1 valid pitch
H_short = entropyExpTens(P_h_short, [], 50, 2, false, true, 1200, 'nPointsPerDim', 1200, 'verbose', false);
results{end+1,1} = 'entropyExpTens batched: row with too few pitches returns NaN';
results{end,2}   = ~isnan(H_short(1)) && isnan(H_short(2));

% --- v3 fix: single multiset entropy with dim > 1 (previously errored) -----------

% r = 2, isRel = false: dim = 2. Build a periodic dyad density and
% compute its entropy via the new Cartesian grid path.
H_dim2_per = entropyExpTens([0, 4, 7], [], 100, 2, false, true, 1200, ...
    'method', 'normalized', 'nPointsPerDim', 60, 'verbose', false);
results{end+1,1} = 'entropyExpTens single multiset dim=2 periodic: returns finite value';
results{end,2}   = isfinite(H_dim2_per) && H_dim2_per > 0 && H_dim2_per <= 1;

% Non-periodic with explicit bounds
H_dim2_nonper = entropyExpTens([0, 400, 700], [], 12, 2, false, false, 1200, ...
    'method', 'normalized', ...
    'xMin', -100, 'xMax', 800, 'nPointsPerDim', 60, 'verbose', false);
results{end+1,1} = 'entropyExpTens single multiset dim=2 non-periodic: returns finite value';
results{end,2}   = isfinite(H_dim2_nonper) && H_dim2_nonper > 0 && H_dim2_nonper <= 1;

% gridLimit guard
results{end+1,1} = 'entropyExpTens single multiset dim=2: gridLimit guard fires';
results{end,2}   = throwsErrorWithId( ...
    @() entropyExpTens([0, 400, 700], [], 12, 2, false, true, 1200, ...
        'nPointsPerDim', 1200, 'gridLimit', 1e3, 'verbose', false), ...
    'entropyExpTens:gridLimitExceeded');

% Empty weights treated as uniform (same as buildExpTens convention)
H_uni = entropyExpTens(0:11, [], 100, 1, false, true, 12, 'nPointsPerDim', 1200, 'verbose', false);
H_ones = entropyExpTens(0:11, ones(1, 12), 100, 1, false, true, 12, 'nPointsPerDim', 1200, 'verbose', false);
results{end+1,1} = 'entropyExpTens: w=[] equivalent to ones(1,N)';
results{end,2}   = abs(H_uni - H_ones) < 1e-14;


% ---- Differential entropy: tightest accuracy in >= 2-D refuses ----
% At the tightest accuracy (truncationSigmas = Inf resolves to the accuracy
% floor) the 2-D grid needed to certify convergence exceeds the
% memory-derived feasibility budget, so the routine refuses and directs the
% user to a coarser accuracy or the closed-form estimator, rather than
% degrading silently or exhausting memory. A small kernelChunkBytes pins
% the budget low so the refusal is deterministic regardless of the
% machine's available memory. Twin of the Python
% test_2d_periodic_tightest_accuracy_refuses_with_guidance.
P2 = [1; 2; 4; 5];
W2 = [1; 1; 1; 1];
sigEff = sqrt(2);
densRefuse = buildExpTens({P2; P2}, {W2; W2}, [sigEff sigEff], [1 1], ...
    [false false], [true true], [12 12], 'verbose', false);
prevKcb = mptDefaults('kernelChunkBytes');
mptDefaults('kernelChunkBytes', 8 * 1024 * 1024);
refused = false;
try
    entropyExpTens(densRefuse, 'method', 'differential', ...
        'truncationSigmas', Inf, 'verbose', false);
catch ME
    refused = strcmp(ME.identifier, 'entropyExpTens:differentialGridLimit');
end
mptDefaults('kernelChunkBytes', prevKcb);
results{end+1,1} = 'entropyExpTens differential: tightest 2-D accuracy refuses with guidance';
results{end,2}   = refused;


%% ---- Standalone summary ----

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
    fprintf('\n=== test_entropy: %d passed, %d failed (of %d) ===\n\n', ...
        nPass, nFail, nPass + nFail);
    clear cleanupDefaults
    if nFail > 0
        error('test_entropy:failed', '%d test(s) failed.', nFail);
    end
end
