%% test_rect_window_and_renyi2_degenerate.m — two v3 out-of-band fixes
%
%  #20  weightEvents' rectangular window uses a half-open support
%       [c - W/2, c + W/2): a regular pulse grid yields exactly N pulses
%       for full support N*IOI at every N. A closed interval over-counts
%       even widths (widths 1..5 collapse to pulse counts 1, 3, 3, 5, 5)
%       and, at a between-pulse centre, can drop both flanking edge
%       pulses and leave a zero-mass density.
%
%  #21  entropyExpTens with method='renyi2' returns NaN for a zero-mass
%       density (for instance a windowed sweep centre with no event in
%       support) rather than raising. Sweep callers already want NaN
%       outside the data.
%
%  Python twin: tests/test_rect_window_and_renyi2_degenerate.py, whose
%  eight checks this file mirrors one for one.
%
%  Standalone-runnable; appends to `results` when called from test_mpt.m.

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    clear cleanupDefaults
    cleanupDefaults = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end

% Unit pulse grid: pitch and time both run 0, 1, ..., 8. The window is
% driven by time (attribute 2) and written to pitch (attribute 1).
rwPitch9 = 0:8;
rwTime9  = 0:8;

% --- 1. On-pulse centre: width N keeps exactly N pulses ---------------
rwOk = true;
for rwW = 1:5
    [~, rwWc, ~] = unpackPreMaet(weightEvents({rwPitch9, rwTime9}, [], 2, 1, 4, 1, ...
                                'width', rwW, 'dropInputAttr', false));
    rwOk = rwOk && (nnz(rwWc{1}) == rwW);
end
results{end+1, 1} = 'rectWindow: width N keeps exactly N pulses (on-pulse centre)'; %#ok<*AGROW>
results{end, 2}   = rwOk;

% --- 2. Between-pulse centre is not empty -----------------------------
%  Midway between pulses 3 and 4; width 1 keeps the lower-edge pulse,
%  not zero (the closed-interval degeneracy) and not two.
[~, rwWb, ~] = unpackPreMaet(weightEvents({rwPitch9, rwTime9}, [], 2, 1, 3.5, 1, ...
                            'width', 1, 'dropInputAttr', false));
results{end+1, 1} = 'rectWindow: between-pulse centre keeps one pulse';
results{end, 2}   = nnz(rwWb{1}) == 1;

% --- 3. Lower edge included, upper edge excluded ----------------------
%  width 2 at centre 4 spans [3, 5): pulses 3 and 4, not pulse 5.
[~, rwWe, ~] = unpackPreMaet(weightEvents({rwPitch9, rwTime9}, [], 2, 1, 4, 1, ...
                            'width', 2, 'dropInputAttr', false));
rwFactor = rwWe{1};
results{end+1, 1} = 'rectWindow: lower edge included, upper edge excluded';
results{end, 2}   = rwFactor(4) > 0 && rwFactor(5) > 0 && rwFactor(6) == 0;

% --- 4. Non-integer inter-onset interval ------------------------------
%  Pulses spaced 0.25; full support 1.0 = 4 * IOI keeps 4 pulses.
rwPitch = 0:8;
rwTime  = 0.25 * (0:8);
[~, rwW4, ~] = unpackPreMaet(weightEvents({rwPitch, rwTime}, [], 2, 1, rwTime(5), 1, ...
                            'width', 1, 'dropInputAttr', false));
results{end+1, 1} = 'rectWindow: fractional IOI grid keeps width/IOI pulses';
results{end, 2}   = nnz(rwW4{1}) == 4;

% --- 5. The Gaussian shape is untouched by the half-open rule ---------
rwPitch5 = 0:4;
rwTime5  = 0:4;
[~, rwW5, ~] = unpackPreMaet(weightEvents({rwPitch5, rwTime5}, [], 2, 1, 2, 0, ...
                            'sd', 1, 'dropInputAttr', false));
rwF5 = rwW5{1};
results{end+1, 1} = 'rectWindow: Gaussian shape peak-normalised with no hard edge';
results{end, 2}   = abs(rwF5(3) - 1) < 1e-12 && all(rwF5 > 0);

% --- 6. renyi2 on an all-zero-weight single multiset returns NaN ------
rwH1 = entropyExpTens([0 4 7], [0 0 0], 1, 1, false, false, 0, ...
                      'method', 'renyi2', 'verbose', false);
results{end+1, 1} = 'renyi2: all-zero weights return NaN, not an error';
results{end, 2}   = isnan(rwH1);

% --- 7. renyi2 on an out-of-support windowed density returns NaN ------
rwPitch7 = [60 62 64];
rwTime7  = [0 1 2];
[rwPa, rwWa, rwSp] = unpackPreMaet(weightEvents({rwPitch7, rwTime7}, [], 2, 1, 100, 1, ...
                                  'width', 1, 'dropInputAttr', false));
rwDens = buildExpTens(rwPa, rwWa, 'specs', rwSp, 'sigma', [1 1], ...
                      'isPer', [false false], 'period', [0 0], ...
                      'verbose', false);
rwH2 = entropyExpTens(rwDens, 'method', 'renyi2', 'verbose', false);
results{end+1, 1} = 'renyi2: out-of-support window returns NaN, not an error';
results{end, 2}   = isnan(rwH2);

% --- 8. A healthy density still returns a finite value ----------------
%  Guards against the NaN paths above swallowing every case.
rwH3 = entropyExpTens([0 4 7], [], 30, 2, false, false, 0, ...
                      'method', 'renyi2', 'verbose', false);
results{end+1, 1} = 'renyi2: a healthy density still returns a finite value';
results{end, 2}   = isfinite(rwH3);

if standalone
    nPass = sum(cellfun(@(x) isequal(x, true), results(:,2)));
    nFail = numel(results(:,1)) - nPass;
    fprintf('\n=== test_rect_window_and_renyi2_degenerate: %d passed, %d failed (of %d) ===\n', ...
        nPass, nFail, numel(results(:,1)));
    clear cleanupDefaults
    if nFail > 0
        for ii = 1:size(results, 1)
            if ~isequal(results{ii, 2}, true)
                fprintf('  FAIL  %s\n', results{ii, 1});
            end
        end
    end
end
