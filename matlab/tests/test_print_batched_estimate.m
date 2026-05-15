%% test_print_batched_estimate.m — printBatchedEstimate
%
%  Tests for printBatchedEstimate.
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



% Default threshold (10 s): short estimate suppressed
outShortPB = evalc('internal.printBatchedEstimate(''foo'', 100, 5.0);');
results{end+1,1} = 'printBatchedEstimate: 5 s < 10 s default threshold is silent';
results{end,2}   = isempty(strtrim(outShortPB));

% Long estimate prints
outLongPB = evalc('internal.printBatchedEstimate(''foo'', 100, 30.0);');
results{end+1,1} = 'printBatchedEstimate: 30 s > 10 s threshold prints';
results{end,2}   = contains(outLongPB, 'foo') ...
                && contains(outLongPB, 'batched, 100 rows') ...
                && contains(outLongPB, 'estimated time') ...
                && contains(outLongPB, '(Ctrl+C to cancel)');

% verbose=false silences regardless
outFalsePB = evalc('internal.printBatchedEstimate(''foo'', 100, 30.0, false);');
results{end+1,1} = 'printBatchedEstimate: verbose=false silences large estimates';
results{end,2}   = isempty(strtrim(outFalsePB));

% Explicit minPrintSec=0 prints sub-10s
outZeroPB = evalc('internal.printBatchedEstimate(''foo'', 100, 0.05, true, 0);');
results{end+1,1} = 'printBatchedEstimate: minPrintSec=0 prints regardless';
results{end,2}   = contains(outZeroPB, 'foo') && contains(outZeroPB, '50 ms');


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
    fprintf('\n=== test_print_batched_estimate: %d passed, %d failed (of %d) ===\n\n', ...
        nPass, nFail, nPass + nFail);
    clear cleanupDefaults
    if nFail > 0
        error('test_print_batched_estimate:failed', '%d test(s) failed.', nFail);
    end
end
