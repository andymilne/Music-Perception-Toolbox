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


% --- the pre-MAET form ---------------------------------------------------
% One attribute of every event at once. Event by event it gives what the
% single-multiset primitive gives: K becomes K * P, N and the spec stand.
as_spec = {'harmonic', 3, 'powerlaw', 1};
as_values = [0, 100; 1200, 1300];
as_base = struct('name', 'pitch', 'sigma', 20, 'r', 1, 'exch', true, ...
                 'rel', false, 'isPer', false, 'period', 0);
as_pm = packPreMaet({as_values}, [], {as_base});
[as_p, as_w, as_specs] = unpackPreMaet( ...
    addSpectra(as_pm, as_spec{:}, 'attribute', 'pitch'));
as_ok = isequal(size(as_p{1}), [6, 2]);
for as_n = 1:2
    [as_pe, as_we] = addSpectra(as_values(:, as_n), [], as_spec{:});
    as_ok = as_ok && max(abs(as_p{1}(:, as_n) - as_pe)) < 1e-12 ...
            && max(abs(as_w{1}(:, as_n) - as_we)) < 1e-12;
end
results(end+1, :) = {'addSpectra: the pre-MAET form expands every event', ...
    as_ok && as_specs{1}.r == 1 && as_specs{1}.sigma == 20}; %#ok<*SAGROW>

% A missing value has no spectrum, so its partials are missing too and
% weigh nothing.
as_pm = packPreMaet({[0, 100; 1200, NaN]}, {[1, 1; 1, 0]}, {as_base});
[as_p, as_w] = unpackPreMaet(addSpectra(as_pm, 'harmonic', 2, ...
                                        'powerlaw', 1, 'attribute', 1));
% The padded value is the second of two, and the rows run partial by
% partial, so its partials are rows 2 and 4.
results(end+1, :) = {'addSpectra: padded slots stay padded', ...
    all(isnan(as_p{1}([2 4], 2))) && all(as_w{1}([2 4], 2) == 0) ...
    && all(isfinite(as_p{1}([1 3], 2))) ...
    && all(isfinite(as_p{1}(:, 1)))};

% Partials of one value differ in weight, so weights become necessary and
% every attribute gets them.
as_two = packPreMaet({[0], [7]}, [], ...
    {as_base, struct('name', 'b', 'sigma', 1, 'r', 1, 'exch', true, ...
                     'rel', false, 'isPer', false, 'period', 0)});
[~, as_w0] = unpackPreMaet(as_two);
[~, as_w] = unpackPreMaet(addSpectra(as_two, as_spec{:}, 'attribute', 'pitch'));
results(end+1, :) = {'addSpectra: a weightless pre-MAET comes back weighted', ...
    isempty(as_w0) && numel(as_w) == 2 && isequal(as_w{2}, 1)};

% Where the positions carry meaning, expanding them would make a tuple
% take one value's partials rather than one value per position.
as_ordered = packPreMaet({[0; 700]}, [], ...
    {struct('name', 'voicing', 'sigma', 20, 'r', 2, 'exch', false, ...
            'rel', false, 'isPer', false, 'period', 0)});
as_ids = cell(1, 4);
as_calls = {@() addSpectra(as_ordered, as_spec{:}, 'attribute', 1), ...
            @() addSpectra(as_pm), ...
            @() addSpectra(as_pm, as_spec{:}), ...
            @() addSpectra(as_pm, as_spec{:}, 'attribute', 'nope')};
for as_k = 1:4
    try
        as_calls{as_k}();
        as_ids{as_k} = '';
    catch as_err
        as_ids{as_k} = as_err.identifier;
    end
end
results(end+1, :) = {'addSpectra: the pre-MAET form refuses what it cannot do', ...
    isequal(as_ids, {'addSpectra:orderedAttribute', 'addSpectra:noMode', ...
                     'addSpectra:noAttribute', 'addSpectra:unknownName'})};

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
