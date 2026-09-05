%% test_serial_continuity.m — serial-position feature: continuity
%
%  Tests for serial-position feature: continuity.
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



% --- Discrete-limit examples ---
[c, m] = continuity([3;5;7;7;9], 11, 0, 'mode', 'strict');
results{end+1,1} = 'continuity: strict count = 1, mag = 2';
results{end,2}   = abs(c - 1) < 1e-10 && abs(m - 2) < 1e-10;

[c, m] = continuity([3;5;7;7;9], 11, 0, 'mode', 'lenient');
results{end+1,1} = 'continuity: lenient count = 3, mag = 6';
results{end,2}   = abs(c - 3) < 1e-10 && abs(m - 6) < 1e-10;

% --- Query = seq(end) gives 0 ---
[c, m] = continuity([3;5;7;7;9], 9, 0);
results{end+1,1} = 'continuity: query = seq(end) gives 0';
results{end,2}   = abs(c) < 1e-10 && abs(m) < 1e-10;

% --- Multi-query shape ---
[c, m] = continuity([3;5;7;7;9], [10;11;12], 0, 'mode', 'lenient');
results{end+1,1} = 'continuity: multi-query output shape';
results{end,2}   = isequal(size(c), [3, 1]) && isequal(size(m), [3, 1]);

% --- N < 2 ---
[c, m] = continuity(5, [11;12;13], 0);
results{end+1,1} = 'continuity: N<2 gives zero vectors';
results{end,2}   = all(c == 0) && all(m == 0);

% --- Descending query on ascending seq ---
[c, m] = continuity([1;2;3;4;5], 4, 0, 'mode', 'lenient');
results{end+1,1} = 'continuity: descending query on ascending seq = 0';
results{end,2}   = abs(c) < 1e-10 && abs(m) < 1e-10;

% --- Slope via magnitude/count ---
[c, m] = continuity([1;4;7;10;13], 16, 0, 'mode', 'lenient');
results{end+1,1} = 'continuity: slope = magnitude / count = 3';
results{end,2}   = abs(c - 4) < 1e-10 && abs(m - 12) < 1e-10 && ...
                   abs(m/c - 3) < 1e-10;

% --- Signed magnitude for descending ---
[c, m] = continuity([10;8;6;4], 2, 0, 'mode', 'strict');
results{end+1,1} = 'continuity: descending seq gives negative magnitude';
results{end,2}   = abs(c - 3) < 1e-10 && abs(m + 6) < 1e-10;

% --- Smoothing monotone ---
[c_small, ~] = continuity([3;5;7;7;9], 11, 0.01, 'mode', 'lenient');
[c_large, ~] = continuity([3;5;7;7;9], 11, 1.0,  'mode', 'lenient');
results{end+1,1} = 'continuity: smoothing lowers count when sigma large';
results{end,2}   = c_large < c_small;

% --- Explicit theta ---
[c_theta, ~] = continuity([3;5;7;7;9], 11, 0, 'theta', -1);
results{end+1,1} = 'continuity: explicit theta = -1 matches lenient';
results{end,2}   = abs(c_theta - 3) < 1e-10;

% --- theta out of range ---
results{end+1,1} = 'continuity: theta out of range errors';
results{end,2}   = throwsError(@() continuity([3;5;7], 8, 0, ...
    'theta', 2));

% --- Weight argument (v3) ---

% w = [] matches default
[c1, m1] = continuity([3;5;7;7;9], 11, 0, 'mode', 'lenient');
[c2, m2] = continuity([3;5;7;7;9], 11, 0, 'mode', 'lenient', 'w', []);
results{end+1,1} = 'continuity: w=[] matches default';
results{end,2}   = abs(c1 - c2) < 1e-12 && abs(m1 - m2) < 1e-12;

% Scalar w = 1 equals unweighted
[c_un, m_un] = continuity([3;5;7;7;9], 11, 0, 'mode', 'lenient');
[c_w,  m_w ] = continuity([3;5;7;7;9], 11, 0, 'mode', 'lenient', ...
    'w', 1);
results{end+1,1} = 'continuity: scalar w=1 equals unweighted';
results{end,2}   = abs(c_un - c_w) < 1e-12 && abs(m_un - m_w) < 1e-12;

% Scalar w = 0.5 scales outputs by 0.25 (rolling-product gives c^2)
[c_un, m_un] = continuity([3;5;7;7;9], 11, 0, 'mode', 'lenient');
[c_w,  m_w ] = continuity([3;5;7;7;9], 11, 0, 'mode', 'lenient', ...
    'w', 0.5);
results{end+1,1} = 'continuity: scalar w=0.5 scales by 0.25';
results{end,2}   = abs(c_w - 0.25 * c_un) < 1e-12 && ...
                   abs(m_w - 0.25 * m_un) < 1e-12;

% Recency zero-out: only the most-recent interval contributes
[c, m] = continuity([3;5;7;7;9], 11, 0, 'mode', 'lenient', ...
    'w', [0 0 0 1 1]);
results{end+1,1} = 'continuity: recency w truncates to last interval';
results{end,2}   = abs(c - 1) < 1e-12 && abs(m - 2) < 1e-12;

% Per-event w = [1 1 0.5 1 1] hand-calculation:
%   diff events: (3->5)=2, (5->7)=2, (7->7)=0, (7->9)=2
%   diff weights: 1*1=1, 1*0.5=0.5, 0.5*1=0.5, 1*1=1
%   lenient walk: 1*1 + 0.5*1 + 0 + 1*1 = 2.5 count; 2+1+0+2 = 5 mag
[c, m] = continuity([3;5;7;7;9], 11, 0, 'mode', 'lenient', ...
    'w', [1 1 0.5 1 1]);
results{end+1,1} = 'continuity: per-event w matches rolling product';
results{end,2}   = abs(c - 2.5) < 1e-12 && abs(m - 5) < 1e-12;

% w composes with sigma smoothing: uniform scalar c -> scale by c^2
[c_un, m_un] = continuity([3;5;7;7;9], 11, 0.3, 'mode', 'lenient');
[c_w,  m_w ] = continuity([3;5;7;7;9], 11, 0.3, 'mode', 'lenient', ...
    'w', 0.7);
results{end+1,1} = 'continuity: w composes with sigma smoothing';
results{end,2}   = abs(c_w - 0.49 * c_un) < 1e-12 && ...
                   abs(m_w - 0.49 * m_un) < 1e-12;

% Wrong-length w errors
results{end+1,1} = 'continuity: wrong-length w errors';
results{end,2}   = throwsError(@() continuity([3;5;7;7;9], 11, 0, ...
    'w', [1 1 1]));

% Negative weights error
results{end+1,1} = 'continuity: negative weights error';
results{end,2}   = throwsError(@() continuity([3;5;7;7;9], 11, 0, ...
    'w', [1 1 -0.1 1 1]));

% Negative scalar weight errors
results{end+1,1} = 'continuity: negative scalar w errors';
results{end,2}   = throwsError(@() continuity([3;5;7;7;9], 11, 0, ...
    'w', -0.5));


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
    fprintf('\n=== test_serial_continuity: %d passed, %d failed (of %d) ===\n\n', ...
        nPass, nFail, nPass + nFail);
    clear cleanupDefaults
    if nFail > 0
        error('test_serial_continuity:failed', '%d test(s) failed.', nFail);
    end
end
