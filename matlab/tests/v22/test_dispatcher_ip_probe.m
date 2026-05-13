%% test_dispatcher_ip_probe.m — v2.2.x probe-based SA IP dispatcher
%
%  Mirrors python/tests/v22/test_dispatcher_ip_probe.py. The
%  dispatcher (localSelectAndEstimateSAIP inside cosSimExpTens) decides
%  between the Möbius method and the Bulger's method. Hard rules decide
%  first (correctness / feasibility); then an analytical pre-screen
%  catches clear-winner cases without probe overhead; otherwise both
%  paths are timed on a small subset and the faster is picked.
%
%  Standalone-runnable.

if ~exist('results', 'var')
    results = {};
end

% --- Semantic equivalence: explicit orbit matches explicit Bulger --
rng(0, 'twister');
ipprobe_px = sort(rand(20, 1) * 100);
rng(1, 'twister');
ipprobe_py = sort(rand(20, 1) * 100);
ipprobe_dx = buildExpTens(ipprobe_px, ones(20, 1), 1, 3, false, false, 1200, ...
    'verbose', false);
ipprobe_dy = buildExpTens(ipprobe_py, ones(20, 1), 1, 3, false, false, 1200, ...
    'verbose', false);
ipprobe_simOrbit = cosSimExpTens(ipprobe_dx, ipprobe_dy, ...
    'method', 'mobius', 'verbose', false);
ipprobe_simPair  = cosSimExpTens(ipprobe_dx, ipprobe_dy, ...
    'method', 'bulger', 'verbose', false);
results{end+1, 1} = 'ip_probe: explicit orbit matches explicit Bulger';
results{end, 2}   = abs(ipprobe_simOrbit - ipprobe_simPair) < 1e-10;

% --- Auto matches explicit paths --------------------------------------
ipprobe_simAuto = cosSimExpTens(ipprobe_dx, ipprobe_dy, 'verbose', false);
results{end+1, 1} = 'ip_probe: auto routing matches explicit orbit';
results{end, 2}   = abs(ipprobe_simAuto - ipprobe_simOrbit) < 1e-10;

% --- Hard rule: r=1 routes pairwise (no orbit machinery) -------------
ipprobe_dx1 = buildExpTens(ipprobe_px, ones(20, 1), 1, 1, false, false, 1200, ...
    'verbose', false);
ipprobe_dy1 = buildExpTens(ipprobe_py, ones(20, 1), 1, 1, false, false, 1200, ...
    'verbose', false);
ipprobe_simR1 = cosSimExpTens(ipprobe_dx1, ipprobe_dy1, ...
    'method', 'auto', 'verbose', false);
results{end+1, 1} = 'ip_probe: r=1 hard rule routes pairwise (call ok)';
results{end, 2}   = isfinite(ipprobe_simR1);

% --- Hard rule: K_y too small for orbit safety (n_min - r < 2) -------
rng(1, 'twister');
ipprobe_pyS = sort(rand(4, 1) * 100);
ipprobe_dyS = buildExpTens(ipprobe_pyS, ones(4, 1), 1, 3, false, false, 1200, ...
    'verbose', false);
ipprobe_simSafety = cosSimExpTens(ipprobe_dx, ipprobe_dyS, ...
    'method', 'auto', 'verbose', false);
results{end+1, 1} = 'ip_probe: n_min-r<2 hard rule routes pairwise';
results{end, 2}   = isfinite(ipprobe_simSafety);

% --- Verbose dispatch message: probe-firing region (r=2, K=5) --------
rng(0, 'twister');
ipprobe_pxP = sort(rand(5, 1) * 100);
rng(1, 'twister');
ipprobe_pyP = sort(rand(5, 1) * 100);
ipprobe_dxP = buildExpTens(ipprobe_pxP, ones(5, 1), 1, 2, false, false, 1200, ...
    'verbose', false);
ipprobe_dyP = buildExpTens(ipprobe_pyP, ones(5, 1), 1, 2, false, false, 1200, ...
    'verbose', false);
ipprobe_evalStr = evalc( ...
    'cosSimExpTens(ipprobe_dxP, ipprobe_dyP, ''verbose'', true);');
results{end+1, 1} = 'ip_probe: verbose message printed when probe fires';
results{end, 2}   = contains(ipprobe_evalStr, 'cosSimExpTens') ...
                 && contains(ipprobe_evalStr, 'chose');

% --- Verbose dispatch message: absent when hard rule decides ---------
ipprobe_evalStrR1 = evalc( ...
    'cosSimExpTens(ipprobe_dx1, ipprobe_dy1, ''verbose'', true);');
results{end+1, 1} = 'ip_probe: verbose message absent at r=1 (hard rule)';
results{end, 2}   = ~contains(ipprobe_evalStrR1, 'cosSimExpTens: chose');

% --- Verbose dispatch message: absent when user method set -----------
ipprobe_evalStrUser = evalc( ...
    ['cosSimExpTens(ipprobe_dx, ipprobe_dy, ''method'', ''bulger'', ' ...
     '''verbose'', true);']);
results{end+1, 1} = 'ip_probe: verbose message absent when method set';
results{end, 2}   = ~contains(ipprobe_evalStrUser, 'cosSimExpTens: chose');

clear ipprobe_px ipprobe_py ipprobe_dx ipprobe_dy ipprobe_simOrbit ...
      ipprobe_simPair ipprobe_simAuto ipprobe_dx1 ipprobe_dy1 ipprobe_simR1 ...
      ipprobe_pyS ipprobe_dyS ipprobe_simSafety ipprobe_pxP ipprobe_pyP ...
      ipprobe_dxP ipprobe_dyP ipprobe_evalStr ipprobe_evalStrR1 ...
      ipprobe_evalStrUser
