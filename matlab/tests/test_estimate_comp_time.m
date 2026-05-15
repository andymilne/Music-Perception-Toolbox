%% test_estimate_comp_time.m — estimateCompTime
%
%  Tests for estimateCompTime.
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



est = estimateCompTime(1000, 2, 'test');
results{end+1,1} = 'estimateCompTime: positive';
results{end,2}   = est > 0;

% Default threshold (10 s): tiny work doesn't print
outTinyEC = evalc('estimateCompTime(100, 1, ''tinywork'', true);');
results{end+1,1} = 'estimateCompTime: default threshold suppresses sub-10s estimates';
results{end,2}   = isempty(strtrim(outTinyEC));

% Pass minPrintSec=0 to recover always-print behaviour
outZeroEC = evalc('estimateCompTime(100, 1, ''tinywork'', true, 0);');
results{end+1,1} = 'estimateCompTime: minPrintSec=0 prints regardless of size';
results{end,2}   = contains(outZeroEC, 'tinywork');

% Every printed estimate carries the Ctrl+C suffix
outPrintedEC = evalc('estimateCompTime(100, 1, ''tinywork'', true, 0);');
results{end+1,1} = 'estimateCompTime: printed estimate includes Ctrl+C suffix';
results{end,2}   = contains(outPrintedEC, '(Ctrl+C to cancel)');


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
    fprintf('\n=== test_estimate_comp_time: %d passed, %d failed (of %d) ===\n\n', ...
        nPass, nFail, nPass + nFail);
    clear cleanupDefaults
    if nFail > 0
        error('test_estimate_comp_time:failed', '%d test(s) failed.', nFail);
    end
end
