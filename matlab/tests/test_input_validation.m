%% test_input_validation.m — cross-cutting input-validation paths
%
%  Tests for cross-cutting input-validation paths.
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



results{end+1,1} = 'buildExpTens: r too large errors';
results{end,2}   = throwsError(@() buildExpTens([0, 4], [], 10, 3, ...
    false, true, 12, 'verbose', false));

% v2.2: SA isRel + r=1 was a hard error in v2.0/v2.1; relaxed to a
% degenerate warning that parallels the MA path's behaviour. The build
% itself succeeds (dim = 0); the warning flags the unusual regime.
w_state_sa1rel = warning('on', 'buildExpTens:isRelDegenerate');
lastwarn('');
dens_sa1rel = buildExpTens([0, 4, 7], [], 10, 1, true, true, 12, ...
    'verbose', false);
[~, lastID_sa1rel] = lastwarn;
warning(w_state_sa1rel);
results{end+1,1} = 'buildExpTens: SA isRel + r=1 emits degenerate warning (was error in v2.1)';
results{end,2}   = strcmp(lastID_sa1rel, 'buildExpTens:isRelDegenerate') ...
                   && isstruct(dens_sa1rel) && dens_sa1rel.dim == 0;

results{end+1,1} = 'coherence: duplicates error';
results{end,2}   = throwsError(@() coherence([0, 0, 4, 7], 12));

results{end+1,1} = 'nTupleEntropy: n too large errors';
results{end,2}   = throwsError(@() nTupleEntropy([0, 2, 4], 12, 3));


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
    fprintf('\n=== test_input_validation: %d passed, %d failed (of %d) ===\n\n', ...
        nPass, nFail, nPass + nFail);
    clear cleanupDefaults
    if nFail > 0
        error('test_input_validation:failed', '%d test(s) failed.', nFail);
    end
end
