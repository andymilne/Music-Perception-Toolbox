%% test_serial_seq_weights.m — serial-position feature: seqWeights
%
%  Tests for serial-position feature: seqWeights.
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



v = seqWeights([], 'primacy', 'N', 5);
results{end+1,1} = 'seqWeights: primacy -> [1;0;0;0;0]';
results{end,2}   = isequal(v, [1;0;0;0;0]);

v = seqWeights([], 'recency', 'N', 5);
results{end+1,1} = 'seqWeights: recency -> [0;0;0;0;1]';
results{end,2}   = isequal(v, [0;0;0;0;1]);

v = seqWeights([], 'exponentialFromEnd', 'N', 5, 'decayRate', 0);
results{end+1,1} = 'seqWeights: zero decay gives uniform';
results{end,2}   = all(abs(v - 1) < 1e-10);

v = seqWeights([], 'uShape', 'N', 5, 'decayRate', 0.5, 'alpha', 0.5);
results{end+1,1} = 'seqWeights: uShape alpha=0.5 symmetric';
results{end,2}   = max(abs(v - flipud(v))) < 1e-10;

v = seqWeights([], [0.1;0.2;0.4;0.2;0.1], 'N', 5);
results{end+1,1} = 'seqWeights: numeric vector passthrough';
results{end,2}   = isequal(v, [0.1;0.2;0.4;0.2;0.1]);

results{end+1,1} = 'seqWeights: profile length mismatch errors';
results{end,2}   = throwsError(@() seqWeights([], [0.1;0.2;0.3], 'N', 5));

results{end+1,1} = 'seqWeights: unknown spec errors';
results{end,2}   = throwsError(@() seqWeights([], 'wibble', 'N', 5));

% --- w as [] (uniform) matches explicit ones ---
v_empty = seqWeights([], 'exponentialFromEnd', 'N', 5, 'decayRate', 0.5);
v_ones  = seqWeights(ones(5,1), 'exponentialFromEnd', 'decayRate', 0.5);
results{end+1,1} = 'seqWeights: [] for w matches ones';
results{end,2}   = max(abs(v_empty - v_ones)) < 1e-10;

% --- w multiplies profile pointwise ---
w = [0.8; 0.5; 1.0; 0.3; 0.9];
v = seqWeights(w, 'recency');
results{end+1,1} = 'seqWeights: w multiplies profile (recency picks w(end))';
results{end,2}   = isequal(v, [0;0;0;0;0.9]);

% --- w multiplies explicit profile vector ---
w = [2;2;2];
profile = [0.1; 0.5; 0.4];
v = seqWeights(w, profile);
results{end+1,1} = 'seqWeights: w multiplies explicit profile';
results{end,2}   = max(abs(v - 2*profile)) < 1e-10;

% --- N vs length(w) mismatch errors ---
results{end+1,1} = 'seqWeights: N vs length(w) mismatch errors';
results{end,2}   = throwsError(@() seqWeights([1;2;3], 'flat', 'N', 5));

% --- N required when w is [] ---
results{end+1,1} = 'seqWeights: missing N with [] w errors';
results{end,2}   = throwsError(@() seqWeights([], 'flat'));

% --- N required when w is scalar ---
results{end+1,1} = 'seqWeights: missing N with scalar w errors';
results{end,2}   = throwsError(@() seqWeights(0.5, 'flat'));

% --- Scalar w broadcasts to length N ---
v = seqWeights(0.5, 'flat', 'N', 4);
results{end+1,1} = 'seqWeights: scalar w broadcasts';
results{end,2}   = max(abs(v - 0.5*ones(4,1))) < 1e-10;

% --- N inferred from w matches explicit N ---
w = [0.2; 0.8; 0.5];
v_inf = seqWeights(w, 'recency');
v_exp = seqWeights(w, 'recency', 'N', 3);
results{end+1,1} = 'seqWeights: inferred N matches explicit N';
results{end,2}   = max(abs(v_inf - v_exp)) < 1e-10;


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
    fprintf('\n=== test_serial_seq_weights: %d passed, %d failed (of %d) ===\n\n', ...
        nPass, nFail, nPass + nFail);
    clear cleanupDefaults
    if nFail > 0
        error('test_serial_seq_weights:failed', '%d test(s) failed.', nFail);
    end
end
