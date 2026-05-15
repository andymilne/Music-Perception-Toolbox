%% test_add_spectra.m — addSpectra spectrum-generation tests
%
%  Tests for addSpectra spectrum-generation tests.
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



[p, ~] = addSpectra([0, 400, 700], [], 'harmonic', 8, 'powerlaw', 1);
results{end+1,1} = 'addSpectra: harmonic count';
results{end,2}   = numel(p) == 24;

[p, w] = addSpectra(0, 1, 'harmonic', 4, 'powerlaw', 0);
expected_p = 1200 * log2((1:4)');
results{end+1,1} = 'addSpectra: harmonic positions';
results{end,2}   = all(abs(p - expected_p) < 1e-10);
results{end+1,1} = 'addSpectra: flat weights';
results{end,2}   = all(abs(w - 1) < 1e-10);

[p, ~] = addSpectra(0, 1, 'stretched', 3, 1.02, 'powerlaw', 1);
[pHarm, ~] = addSpectra(0, 1, 'harmonic', 3, 'powerlaw', 1);
results{end+1,1} = 'addSpectra: stretched wider than harmonic';
results{end,2}   = p(3) > pHarm(3);

[p, ~] = addSpectra(0, 1, 'stiff', 4, 0.0003, 'powerlaw', 1);
[pHarm, ~] = addSpectra(0, 1, 'harmonic', 4, 'powerlaw', 1);
results{end+1,1} = 'addSpectra: stiff sharper than harmonic';
results{end,2}   = p(4) > pHarm(4);

[p, w] = addSpectra([0, 700], [], 'custom', [0, 1200], [1, 0.5]);
results{end+1,1} = 'addSpectra: custom positions';
results{end,2}   = all(abs(p - [0; 700; 1200; 1900]) < 1e-10);
results{end+1,1} = 'addSpectra: custom weights';
results{end,2}   = all(abs(w - [1; 1; 0.5; 0.5]) < 1e-10);

[~, w] = addSpectra(0, 1, 'harmonic', 4, 'geometric', 0.5);
results{end+1,1} = 'addSpectra: geometric weights';
results{end,2}   = all(abs(w - [1; 0.5; 0.25; 0.125]) < 1e-10);

% -- freqlinear: alpha = 0 reproduces harmonic (ratio(n) = n) --
[p_lin, w_lin] = addSpectra(0, 1, 'freqlinear', 4, 0.0, 'powerlaw', 0);
[p_har, w_har] = addSpectra(0, 1, 'harmonic', 4, 'powerlaw', 0);
results{end+1,1} = 'addSpectra: freqlinear alpha=0 equals harmonic';
results{end,2}   = max(abs(p_lin - p_har)) < 1e-10 ...
                && max(abs(w_lin - w_har)) < 1e-10;

% -- freqlinear: alpha = 1 gives ratios [1, 1.5, 2, 2.5] --
[p_fl, ~] = addSpectra(0, 1, 'freqlinear', 4, 1.0, 'powerlaw', 0);
expected_fl = 1200 * log2([1; 1.5; 2; 2.5]);
results{end+1,1} = 'addSpectra: freqlinear alpha=1 partial ratios';
results{end,2}   = max(abs(p_fl - expected_fl)) < 1e-10;

% -- freqlinear: alpha <= -1 errors (ratio non-positive) --
results{end+1,1} = 'addSpectra: freqlinear alpha<=-1 errors';
results{end,2}   = throwsError(@() addSpectra(0, 1, 'freqlinear', 4, -1.0, 'powerlaw', 0));


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
    fprintf('\n=== test_add_spectra: %d passed, %d failed (of %d) ===\n\n', ...
        nPass, nFail, nPass + nFail);
    clear cleanupDefaults
    if nFail > 0
        error('test_add_spectra:failed', '%d test(s) failed.', nFail);
    end
end
