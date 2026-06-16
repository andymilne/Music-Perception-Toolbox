%% test_mpt.m — Test orchestrator for the Music Perception Toolbox (MATLAB)
%
%  Runs every standalone-runnable test file in matlab/tests/ in
%  feature-grouped order, accumulating their results into a single
%  `results` cell, and prints a unified pass/fail summary at the
%  end. Each individual test file can also be run on its own (via
%  `run('test_X')` or by simply typing `test_X` with matlab/tests/
%  on the path); when it detects no caller-provided `results`
%  variable, it prints its own summary and otherwise behaves
%  identically. See mptTestIsolateDefaults.m for the suite-entry
%  defaults-isolation convention.

results = {};  % accumulate {'name', true/false}

% --- Defaults isolation -------------------------------------------------
% Several sections of the suite compare numerical outputs at 1e-10 /
% 1e-12 tolerance, against either analytical goldens, the un-truncated
% direct enumeration, or different internal arithmetic orderings of the
% same computation. Those tolerances assume the factory defaults
% (un-truncated kernels at double precision); under non-default
% mptDefaults state they fail by construction. mptTestIsolateDefaults
% saves the caller's defaults, resets to factory, and returns an
% onCleanup token that restores the saved struct when this script
% exits (normally, on error, or via Ctrl-C). Adding this folder to
% the path ensures sub-runs via `run(fullfile(...))` can find the
% helper too.
%
% IMPORTANT: clear any stale `cleanupDefaults` from a previous
% test_mpt run BEFORE calling the helper. If we let the natural
% reassignment below release the old binding, the stale token's
% destructor would fire AFTER the helper has captured prev and
% reset to factory — meaning the stale destructor's restoration
% (captured at some prior moment) would corrupt the state the
% suite sees. Symptom: tests that depend on kernel precision /
% truncation fail on the first run after a session in which those
% knobs were set, but pass on the second run (because by then the
% chain has drained itself). Clearing first means the stale
% destructor fires BEFORE prev is captured — its effect is harmless
% and we end up at factory defaults regardless.
clear cleanupDefaults
addpath(fileparts(mfilename('fullpath')));
cleanupDefaults = mptTestIsolateDefaults(); %#ok<NASGU>

% Silence informational hints and dispatch-decision announces for the
% suite. Matches the Python conftest.py autouse fixture
% (set_default(show_hints=False)). Tests that verify announce content
% explicitly opt back in via mptDefaults('showHints', true) with their
% own onCleanup token, mirroring the pattern in test_dispatch_scope.m
% and test_dispatch_msg_throttle.m. Function-based tests
% (TEST_DISPATCHER_PROBE, etc.) reset to factory in their per-test
% setup, which restores showHints=true within those tests
% automatically.
mptDefaults('showHints', false);

fprintf('\n=== Music Perception Toolbox — Test Suite ===\n\n');

testsDir = fileparts(mfilename('fullpath'));

% Build the file list in feature-grouped order. We construct it as a
% series of per-group cell arrays and concatenate, rather than as one
% multi-line literal with embedded section comments. The latter
% pattern, e.g.
%
%     testFiles = { ...
%         % --- group label ---
%         'foo.m', ...
%         'bar.m'  };
%
% looks tidy but is a trap: the `...` on the `{` line escapes only its
% own newline; the comment line's trailing newline then acts as a
% cell-row separator, leaving an empty first row followed by content
% rows. The literal becomes ragged and the parse fails with
% "Dimensions of arrays being concatenated are not consistent".
% Grouped concatenation sidesteps this entirely and is unambiguous.

% Core (pitch, spectra, circular measures)
core = {'test_convert_pitch.m', 'test_add_spectra.m', 'test_circular.m'};

% Tier-1/2/4 batched dispatch (v2.1+)
batched = {'test_tier1_batched.m', 'test_tier2_batched.m', 'test_tier4_batched.m'};

% Expectation tensors and entropy
expTens = {'test_exp_tens.m', 'test_entropy.m', 'test_dft_montecarlo.m', ...
           'test_sigma_space.m'};

% Harmony wrappers
harmony = {'test_harmony.m'};

% Cost estimators and validation
cost = {'test_estimate_comp_time.m', 'test_print_batched_estimate.m', ...
        'test_input_validation.m'};

% Serial module (v2.1.0)
serial = {'test_serial_continuity.m', 'test_serial_seq_weights.m'};

% Multi-Attribute Expectation Tensor (MAET, v2.1.0)
maet = {'test_maet.m', 'test_windowed_similarity_offset.m', 'test_windowed_premaet.m', 'test_windowed_nested.m', 'test_sym.m', 'test_nested.m', 'test_nesting_L3.m', 'test_nested_unequal_cardinality.m', 'test_nested_ma_contraction.m', 'test_nested_spectral_factor.m', 'test_specs.m', 'test_bind.m', 'test_bind_by_attribute.m', 'test_difference.m', 'test_translate.m'};

% Geometry helpers
geom = {'test_simplex_vertices.m'};

% v2.2 features
v22 = {'test_mobius_combinatorics.m', 'test_mobius_orbit_table.m', ...
       'test_mobius_ip.m', 'test_mobius_eval.m', ...
       'test_dispatch_sa_cossim.m', 'test_dispatch_sa_eval.m', ...
       'test_dispatch_ma_cossim.m', 'test_tensor_harmonicity_orbit.m', ...
       'test_entropy_renyi2.m', 'test_entropy_four_method_api.m', ...
       'test_cell_mass_zero_weight_prune.m', ...
       'test_event_prune.m', ...
       'test_ma_per_attr_hybrid.m', ...
       'test_cross_language_golden.m', 'test_recipe_equivalence.m', ...
       'test_orbit_vectorisation.m', 'test_kernel_truncation.m', ...
       'test_eval_routing.m', 'test_wrapper_routing.m', ...
       'test_cossim_centres_routing.m', 'test_dispatcher_probe.m', ...
       'test_dispatcher_ip_probe.m', 'test_centres_chunking.m', ...
       'test_dispatch_scope.m'};

testFiles = [core, batched, expTens, harmony, cost, serial, maet, geom, v22];
for ki = 1:numel(testFiles)
    run(fullfile(testsDir, testFiles{ki}));
end

%% ---- Print results ----

nPass = sum([results{:,2}]);
nFail = size(results, 1) - nPass;

for i = 1:size(results, 1)
    if results{i,2}
        fprintf('  PASS  %s\n', results{i,1});
    else
        fprintf('  FAIL  %s\n', results{i,1});
    end
end

fprintf('\n=== Results: %d passed, %d failed (of %d) ===\n\n', ...
    nPass, nFail, nPass + nFail);

% Restore the caller's pre-test defaults eagerly. Clearing the
% onCleanup token fires its destructor immediately, so the user
% sees their original mptDefaults state restored on script exit
% rather than the factory state we ran the suite at. (Without this,
% the token sits in base workspace until the user clears it or the
% session ends.) Done before the conditional error below so the
% restoration also fires when some tests failed.
clear cleanupDefaults

if nFail > 0
    error('test_mpt:failed', '%d test(s) failed.', nFail);
end
