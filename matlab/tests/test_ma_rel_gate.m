%% test_ma_rel_gate.m
%  Regression tests for the centres-vs-grid gate on the multi-attribute
%  relative path (MOBIUS.MARELATTRPREFERSCENTRES). Mirrors the Python
%  tests/test_ma_rel_gate.py coverage cell-for-cell.
%
%  The gate compares predicted wall time on the two paths. Its
%  calibration notes (see the constants at the top of the .m file)
%  explain the empirical fit. The cells below pin its decisions on the
%  labelled truth set from that fit, including the r = 2 band where the
%  previous raw-op-count gate over-selected centres by one to two orders
%  of magnitude in wall time.
%
%  Each case is one (K, r_a, sigma, span_or_period, isPer, expectCentres)
%  tuple. The MATLAB and Python cost-model constants match exactly so
%  that the two languages route the same cells to the same path.
%
%  Standalone-runnable; appends to `results` when called from test_mpt.m.

if ~exist('results', 'var')
    results = {};
    standalone = true;
else
    standalone = false;
end

rng(0);

% --------------------------------------------------------------------
% Non-periodic r = 2 band -- previously the gate over-selected centres
% here; the corrected gate must return grid.
% --------------------------------------------------------------------
nonperR2Grid = { ...
    % K, sigma, spanPerSide
    20, 15.0, 1800.0; ...
    30, 15.0, 1800.0; ...
    50, 15.0, 1800.0; ...
    70, 15.0, 1800.0; ...
    30, 30.0, 1800.0; ...
    40,  5.0, 1800.0; ...
    50,  5.0, 1800.0; ...
};
for ii = 1:size(nonperR2Grid, 1)
    K = nonperR2Grid{ii, 1};
    sigma = nonperR2Grid{ii, 2};
    span = nonperR2Grid{ii, 3};
    Px = sort(rand(K, 1) * span);
    Py = sort(rand(K, 1) * span);
    tf = mobius.maRelAttrPrefersCentres(Px, Py, sigma, 2, true, false, 0);
    name = sprintf('gate non-per r=2: K=%d sigma=%.0f -> grid', K, sigma);
    results(end+1, :) = {name, ~tf}; %#ok<*SAGROW>
end

% --------------------------------------------------------------------
% Non-periodic cells where centres is genuinely faster.
% --------------------------------------------------------------------
nonperCentres = { ...
    % K, r, sigma, spanPerSide
     5, 2, 15.0, 1800.0; ...
     8, 2, 15.0, 1800.0; ...
    12, 2, 15.0, 1800.0; ...
     6, 3, 15.0, 1800.0; ...
     8, 3, 15.0, 1800.0; ...
     5, 4, 15.0, 1800.0; ...
     6, 4, 15.0, 1800.0; ...
};
for ii = 1:size(nonperCentres, 1)
    K = nonperCentres{ii, 1};
    r_a = nonperCentres{ii, 2};
    sigma = nonperCentres{ii, 3};
    span = nonperCentres{ii, 4};
    Px = sort(rand(K, 1) * span);
    Py = sort(rand(K, 1) * span);
    tf = mobius.maRelAttrPrefersCentres(Px, Py, sigma, r_a, true, false, 0);
    name = sprintf('gate non-per: K=%d r=%d -> centres', K, r_a);
    results(end+1, :) = {name, tf};
end

% --------------------------------------------------------------------
% Periodic cells where the gate must pick grid.
% --------------------------------------------------------------------
perGrid = { ...
    % K, r, sigma, period
    30, 2, 15.0, 3600.0; ...
    50, 2, 15.0, 3600.0; ...
    12, 3, 15.0, 3600.0; ...
    10, 3,  5.0, 1200.0; ...
     8, 4, 15.0, 3600.0; ...
};
for ii = 1:size(perGrid, 1)
    K = perGrid{ii, 1};
    r_a = perGrid{ii, 2};
    sigma = perGrid{ii, 3};
    P = perGrid{ii, 4};
    Px = sort(rand(K, 1) * P);
    Py = sort(rand(K, 1) * P);
    tf = mobius.maRelAttrPrefersCentres(Px, Py, sigma, r_a, true, true, P);
    name = sprintf('gate per: K=%d r=%d sigma=%.0f -> grid', K, r_a, sigma);
    results(end+1, :) = {name, ~tf};
end

% --------------------------------------------------------------------
% Periodic cells where centres is cheaper.
% --------------------------------------------------------------------
perCentres = { ...
    % K, r, sigma, period
     5, 2, 15.0, 3600.0; ...
     8, 2, 15.0, 3600.0; ...
     6, 3, 15.0, 3600.0; ...
     8, 3, 15.0, 3600.0; ...
     5, 4, 15.0, 3600.0; ...
     6, 4, 15.0, 3600.0; ...
};
for ii = 1:size(perCentres, 1)
    K = perCentres{ii, 1};
    r_a = perCentres{ii, 2};
    sigma = perCentres{ii, 3};
    P = perCentres{ii, 4};
    Px = sort(rand(K, 1) * P);
    Py = sort(rand(K, 1) * P);
    tf = mobius.maRelAttrPrefersCentres(Px, Py, sigma, r_a, true, true, P);
    name = sprintf('gate per: K=%d r=%d -> centres', K, r_a);
    results(end+1, :) = {name, tf};
end

% --------------------------------------------------------------------
% Trivial early-exit cases (invariant under any calibration).
% --------------------------------------------------------------------
PxAny = sort(rand(20, 1) * 1800);
PyAny = sort(rand(20, 1) * 1800);
results(end+1, :) = {'gate: absolute attribute -> false', ...
    ~mobius.maRelAttrPrefersCentres(PxAny, PyAny, 15, 2, false, false, 0)};
results(end+1, :) = {'gate: r_a < 2 -> false', ...
    ~mobius.maRelAttrPrefersCentres(PxAny, PyAny, 15, 1, true, false, 0)};

Px2 = sort(rand(2, 1) * 1800);
Py2 = sort(rand(2, 1) * 1800);
results(end+1, :) = {'gate: K < r_a (K=2,r=3) -> false', ...
    ~mobius.maRelAttrPrefersCentres(Px2, Py2, 15, 3, true, false, 0)};

Px3 = sort(rand(3, 1) * 1800);
Py3 = sort(rand(3, 1) * 1800);
results(end+1, :) = {'gate: K < r_a (K=3,r=4) -> false', ...
    ~mobius.maRelAttrPrefersCentres(Px3, Py3, 15, 4, true, false, 0)};

Px8 = sort(rand(8, 1) * 100);
Py8 = sort(rand(8, 1) * 100);
results(end+1, :) = {'gate: above sigma/P threshold -> false', ...
    ~mobius.maRelAttrPrefersCentres(Px8, Py8, 10, 2, true, true, 100)};


if standalone
    nPass = sum(cellfun(@(x) isequal(x, true), results(:, 2)));
    nFail = size(results, 1) - nPass;
    fprintf('test_ma_rel_gate: %d passed, %d failed\n', nPass, nFail);
    if nFail > 0
        for ii = 1:size(results, 1)
            if ~isequal(results{ii, 2}, true)
                fprintf('  FAIL  %s\n', results{ii, 1});
            end
        end
    end
end
