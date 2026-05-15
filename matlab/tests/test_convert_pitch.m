%% test_convert_pitch.m — convertPitch unit/round-trip/error tests
%
%  Tests for convertPitch unit/round-trip/error tests.
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



results{end+1,1} = 'convertPitch: Hz→MIDI';
results{end,2}   = abs(convertPitch(440, 'hz', 'midi') - 69) < 1e-10;

results{end+1,1} = 'convertPitch: MIDI→Hz';
results{end,2}   = abs(convertPitch(60, 'midi', 'hz') - 261.6256) / 261.6256 < 1e-4;

results{end+1,1} = 'convertPitch: Hz→cents';
results{end,2}   = abs(convertPitch(440, 'hz', 'cents') - 6900) < 1e-10;

results{end+1,1} = 'convertPitch: identity';
results{end,2}   = isequal(convertPitch([100, 200, 300], 'hz', 'hz'), [100, 200, 300]);

scales = {'midi', 'cents', 'mel', 'bark', 'erb', 'greenwood'};
for i = 1:numel(scales)
    rt = convertPitch(convertPitch(440, 'hz', scales{i}), scales{i}, 'hz');
    results{end+1,1} = ['convertPitch: roundtrip ' scales{i}]; %#ok<SAGROW>
    results{end,2}   = abs(rt - 440) / 440 < 1e-8;
end

out = convertPitch([261.63, 440, 880], 'hz', 'midi');
results{end+1,1} = 'convertPitch: vectorised';
results{end,2}   = all(abs(out - [60, 69, 81]) < 0.01);

results{end+1,1} = 'convertPitch: unknown scale errors';
results{end,2}   = throwsError(@() convertPitch(440, 'hz', 'bogus'));


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
    fprintf('\n=== test_convert_pitch: %d passed, %d failed (of %d) ===\n\n', ...
        nPass, nFail, nPass + nFail);
    clear cleanupDefaults
    if nFail > 0
        error('test_convert_pitch:failed', '%d test(s) failed.', nFail);
    end
end
