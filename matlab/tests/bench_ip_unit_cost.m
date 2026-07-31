%% bench_ip_unit_cost.m
%  Diagnostic for cosSimExpTens on the EDO-approximation workload: a
%  five-partial just-intonation reference against an n-tone equal
%  division, r = 2, relative and periodic.
%
%  Answers three questions.
%
%  1. Do the two routes agree to within the accuracy truncationSigmas
%     states? An explicit 'method','mobius' call returns the Möbius
%     value, which differs from the 'bulger' value by a small fraction
%     of the truncation floor. A cosine has value scale 1, so the floor
%     applies to the difference directly. A difference ABOVE the floor
%     warrants a report; exact equality is unexpected on this workload
%     and also warrants one.
%
%  2. What is the per-operation cost of a translation-grid kernel
%     operation relative to a pairwise kernel operation on this
%     machine? This is ORBIT_GRID_OP_UNIT_COST in
%     localOrbitIPGridFactors (cosSimExpTens.m).
%
%  3. Does the Möbius side's operation count have the right FORM? The
%     count charged to the translation-grid route is
%     B_r * N_u * (K_x*n + K_x^2 + n^2), quadratic in the second value
%     count. If the measured time grows far more slowly than that count,
%     the exponent is wrong, and no choice of unit-cost constant can
%     repair a count of the wrong form. Section 2 measures the Möbius
%     route alone across a wide range of n and fits the growth.
%
%  Section 1 runs both routes and so is limited to small n: Bulger's
%  tuple-pair kernel is n_J*n_K entries, which at n = 200 is 7.9e8
%  doubles (6.3 GB) and at n = 400 is 1.3e10 (100 GB).
%
%  Section 2 runs the Möbius route only. Its working set is far smaller,
%  but the translation-grid branch is not bounded by kernelChunkBytes
%  here (that chunking is over the event count, which is 1 on this
%  workload), so the largest n may still exhaust memory on some
%  machines. Each n is therefore attempted separately and the sweep
%  stops at the first failure, keeping the points already measured.
%
%  Run from anywhere with the toolbox on the path. Takes a few
%  minutes, most of it in Section 3 at r = 4.

prevHints = mptDefaults('showHints');
mptDefaults('showHints', false);

try

refPitches = [0, log2(3), log2(5), log2(7), log2(11)] * 1200;
sigma  = 6;
r      = 2;
isRel  = 1;
isPer  = 1;
period = 1200;

K_x = numel(refPitches);
N_u = internal.autoNtauDefault(period, sigma);
B_r = 2;   % Bell number, r = 2

% Falling factorial K!/(K-k)!, the permutation-side tuple count.
ff = @(K, k) prod(K:-1:(K - k + 1)) * (K >= k);

% Operation counts charged to the two routes.
pwOpsOf  = @(n) ff(K_x, r) * ff(n, r) + ff(K_x, r)^2 + ff(n, r)^2;
orbSqOf  = @(n) B_r * N_u * (K_x * n + K_x^2 + n^2);   % shipped form
orbLinOf = @(n) B_r * N_u * (K_x + n);                 % linear alternative

fprintf('bench_ip_unit_cost: sigma = %g, N_u = %d, K_x = %d\n\n', ...
    sigma, N_u, K_x);

%% ---- Section 1: both routes, agreement and relative per-operation cost ----

nPair = [40, 60, 80, 100];

fprintf('Section 1 -- both routes (n limited by Bulger memory)\n');
fprintf('%6s %13s %13s %14s %11s %11s %9s\n', ...
    'n-EDO', 't_bulger(ms)', 't_mobius(ms)', '|s_mob-s_bul|', ...
    'c_pw(ns)', 'c_orb(ns)', 'ratio');

ratios = zeros(1, numel(nPair));
diffs  = zeros(1, numel(nPair));
for i = 1:numel(nPair)
    n = nPair(i);
    edoPitches = (0:n-1) * (1200 / n);

    call = @(m) cosSimExpTens(refPitches, [], edoPitches, [], ...
        sigma, r, isRel, isPer, period, 'method', m);

    sBul = call('bulger');
    sMob = call('mobius');
    tBul = internal.timeRepeated(@() call('bulger'));
    tMob = internal.timeRepeated(@() call('mobius'));

    c_pw  = tBul / pwOpsOf(n) * 1e9;
    c_orb = tMob / orbSqOf(n) * 1e9;
    ratios(i) = c_orb / c_pw;
    diffs(i)  = abs(sMob - sBul);

    fprintf('%6d %13.3f %13.3f %14.3e %11.3f %11.4f %9.4f\n', ...
        n, tBul * 1e3, tMob * 1e3, diffs(i), c_pw, c_orb, ratios(i));
end

%% ---- Section 2: Möbius route alone, does the operation count scale? ----

nSolo = [40, 60, 80, 100, 150, 200, 300, 400, 600, 800];

fprintf('\nSection 2 -- Mobius route alone\n');
fprintf('%6s %13s %16s %16s\n', ...
    'n-EDO', 't_mobius(ms)', 'ops (N_u K^2)', 'ops (N_u K)');

tSolo  = [];
opsSq  = [];
opsLin = [];
nDone  = [];
for i = 1:numel(nSolo)
    n = nSolo(i);
    edoPitches = (0:n-1) * (1200 / n);
    call = @() cosSimExpTens(refPitches, [], edoPitches, [], ...
        sigma, r, isRel, isPer, period, 'method', 'mobius');
    try
        call();                                  % warm and prove feasible
        t = internal.timeRepeated(call);
    catch ME
        fprintf(['%6d  stopped: %s\n' ...
                 '        The sweep ends here; the fit below uses the ' ...
                 '%d points already measured.\n'], n, ME.message, numel(nDone));
        break;
    end
    nDone(end+1)  = n;      %#ok<SAGROW>
    tSolo(end+1)  = t;      %#ok<SAGROW>
    opsSq(end+1)  = orbSqOf(n);   %#ok<SAGROW>
    opsLin(end+1) = orbLinOf(n);  %#ok<SAGROW>
    fprintf('%6d %13.3f %16.4g %16.4g\n', n, t * 1e3, opsSq(end), opsLin(end));
end

%% ---- Section 3: how the grid route scales with the value count ----

% Section 2 tests r = 2 only. The shipped count charges the
% translation-grid route a term in n^2 at every tuple order, but the
% exponent is not the same at every order: the grid contraction at order
% r takes r-fold products over the value set, so the dependence on the
% value count steepens as r rises. This section forces the grid route,
% sweeps the value count at r = 2, 3 and 4, and reports the exponent
% directly as the slope of log(time) against log(value count).
%
% Read it as: slope near 1 means the cost is linear in the value count,
% near 2 means quadratic, and the shipped count is right only where the
% slope is near 2.
%
% Python measurements on this sweep give slope 0.65 at r = 2 and 1.79 at
% r = 4, so the shipped n^2 is far too steep at r = 2 and about right at
% r = 4. But the Python per-value slope at r = 2 is some seventeen times
% steeper than the MATLAB slope Section 2 measures, so those numbers do
% not carry across and the fit has to be made here before any constant
% in the cost model is changed.

% The value counts are capped per order because the grid route steepens
% with r: at r = 4 a single call at K = 80 already runs for seconds in
% Python, and timeRepeated times at least three runs whatever its budget
% says. Four points are enough to read an exponent.
rOrders  = [2, 3, 4];
KPerOrder = {[5, 10, 20, 40, 80], [5, 10, 20, 40, 80], [5, 10, 20, 40]};

fprintf('\nSection 3 -- grid route scaling by tuple order\n');
fprintf('%3s %6s %13s\n', 'r', 'K', 't_grid(ms)');

for ri = 1:numel(rOrders)
    ra = rOrders(ri);
    Ks = KPerOrder{ri};
    Ks = Ks(Ks >= ra);
    tg = nan(1, numel(Ks));
    for ki = 1:numel(Ks)
        K = Ks(ki);
        rs = RandStream('twister', 'Seed', 1000 * K + ra);
        px = sort(rand(rs, 1, K) * period);
        py = sort(rand(rs, 1, K) * period);
        call = @() cosSimExpTens(px, [], py, [], sigma, ra, isRel, ...
            isPer, period, 'method', 'mobius');
        try
            call();
            tg(ki) = internal.timeRepeated(call);
        catch ME
            fprintf('%3d %6d  stopped: %s\n', ra, K, ME.message);
            break;
        end
    end
    ok = ~isnan(tg);
    Kok = Ks(ok); tok = tg(ok) * 1e3;          % ms
    for ki = 1:numel(Kok)
        fprintf('%3d %6d %13.3f\n', ra, Kok(ki), tok(ki));
    end
    if numel(Kok) < 3
        fprintf('    r = %d: too few points to fit\n', ra);
        continue;
    end
    slope = localLogSlope(Kok, tok);
    % Both candidate forms, for a reader who wants the constants. The
    % exponent above is the primary reading; these say how well each
    % fixed form does.
    Alin = [ones(numel(Kok), 1), 2 * Kok(:)];
    Asq  = [ones(numel(Kok), 1), Kok(:).^2];
    clin = Alin \ tok(:);
    csq  = Asq  \ tok(:);
    fprintf(['    r = %d: exponent in the value count = %.2f\n' ...
             '           setup+linear R^2 = %.4f (setup %.4f ms, ' ...
             '%.4e ms per value)\n' ...
             '           setup+K^2    R^2 = %.4f\n'], ...
        ra, slope, localR2(Alin, clin, tok(:)), clin(1), clin(2), ...
        localR2(Asq, csq, tok(:)));
    if clin(1) < 0
        fprintf(['           The linear fit wants a negative setup, so ' ...
                 'it is the wrong form here.\n']);
    end
    if slope < 1.4
        fprintf(['           -> the shipped n^2 charge is too steep at ' ...
                 'this order.\n']);
    elseif slope > 1.7
        fprintf(['           -> consistent with the shipped n^2 charge ' ...
                 'at this order.\n']);
    else
        fprintf(['           -> between the two; widen KList before ' ...
                 'drawing a conclusion.\n']);
    end
end

%% ---- Verdicts ----

fprintf('\n--- Verdicts ---\n');
floorv = internal.truncationFloor([]);
if max(diffs) == 0
    fprintf(['[1] mobius output is EXACTLY equal to bulger output.\n' ...
             '    The two routes are numerically distinct on this\n' ...
             '    workload, so exact equality means one of them was not\n' ...
             '    exercised. Report this: the measurements below are\n' ...
             '    meaningless if both timings ran the same path.\n']);
elseif max(diffs) > floorv
    fprintf(['[1] mobius and bulger outputs differ by ~%.1e, ABOVE the\n' ...
             '    truncation floor %.1e. The routes should agree to\n' ...
             '    within the accuracy truncationSigmas states; report\n' ...
             '    this.\n'], max(diffs), floorv);
else
    fprintf(['[1] mobius and bulger outputs differ by ~%.1e, within the\n' ...
             '    truncation floor %.1e: both routes are being used and\n' ...
             '    agree to the stated accuracy.\n'], max(diffs), floorv);

    % Reported across n rather than as a single median. A genuine
    % per-operation constant is flat in n; a ratio that declines
    % monotonically means the count it divides grows faster than the
    % work does, which no constant can fix.
    fprintf('[2] Measured per-operation cost ratio across n:\n    ');
    fprintf('%.4f ', ratios);
    fprintf('\n');
    if numel(ratios) > 1 && all(diff(ratios) < 0)
        fprintf(['    The ratio DECLINES monotonically (%.4f to %.4f).\n' ...
                 '    A per-operation constant would be flat, so this is\n' ...
                 '    evidence that the Mobius operation count grows\n' ...
                 '    faster than the measured work. See verdict [3]: a\n' ...
                 '    single constant cannot repair a count of the wrong\n' ...
                 '    form, so ORBIT_GRID_OP_UNIT_COST should not be\n' ...
                 '    reset from these numbers.\n'], ratios(1), ratios(end));
    else
        fprintf(['    The ratio is not monotone in n. Median %.4f is a\n' ...
                 '    usable value for ORBIT_GRID_OP_UNIT_COST in\n' ...
                 '    localOrbitIPGridFactors (cosSimExpTens.m); the\n' ...
                 '    shipped value is 1.6.\n'], median(ratios));
    end
end

if numel(nDone) < 4
    fprintf(['[3] Only %d Mobius points were measured; at least 4 are\n' ...
             '    needed to fit the growth. Reduce the top of nSolo and\n' ...
             '    re-run.\n'], numel(nDone));
else
    % A count of the right form gives a slope near 1 when the measured
    % time is regressed on it in logs; a count whose exponent in the
    % value count is too high gives a slope well below 1.
    slopeSq  = localLogSlope(opsSq,  tSolo);
    slopeLin = localLogSlope(opsLin, tSolo);
    slopeN   = localLogSlope(nDone,  tSolo);

    % Setup plus per-operation cost under the shipped count, by least
    % squares, with the count in millions to keep the two columns
    % comparable in scale. The setup share at the largest n says how
    % much of that reading is fixed overhead rather than growing work.
    A = [ones(numel(nDone), 1), opsSq(:) / 1e6];
    coef = A \ tSolo(:);
    setupMs   = coef(1) * 1e3;
    perMopMs  = coef(2) * 1e3;
    setupShare = 100 * coef(1) / tSolo(end);

    fprintf(['[3] Mobius route growth, n = %d to %d (%.0fx in n):\n' ...
             '    measured time            %.3f ms -> %.3f ms  (%.1fx)\n' ...
             '    count B_r N_u (.. + n^2) rises %.0fx, log-log slope %.3f\n' ...
             '    count B_r N_u (K_x + n)  rises %.0fx, log-log slope %.3f\n' ...
             '    time against n directly, log-log slope %.3f\n'], ...
        nDone(1), nDone(end), nDone(end) / nDone(1), ...
        tSolo(1) * 1e3, tSolo(end) * 1e3, tSolo(end) / tSolo(1), ...
        opsSq(end) / opsSq(1), slopeSq, ...
        opsLin(end) / opsLin(1), slopeLin, slopeN);
    fprintf(['    Least squares on the shipped count: setup %.3f ms plus\n' ...
             '    %.4f ms per million operations; setup is %.0f%% of the\n' ...
             '    reading at n = %d.\n'], ...
        setupMs, perMopMs, setupShare, nDone(end));
    if slopeSq < 0.6
        fprintf(['    A slope near 1 is what a count of the right form\n' ...
                 '    gives. %.3f means the n^2 term charges work the\n' ...
                 '    route does not do, so the exponent in the value\n' ...
                 '    count, not the unit-cost constant, is what needs\n' ...
                 '    changing.\n'], slopeSq);
    elseif slopeSq > 0.85
        fprintf(['    A slope of %.3f is consistent with the shipped\n' ...
                 '    count being the right form, leaving only the\n' ...
                 '    constant to set.\n'], slopeSq);
    else
        fprintf(['    A slope of %.3f sits between the two readings;\n' ...
                 '    extend nSolo before drawing a conclusion.\n'], slopeSq);
    end
end

catch benchErr
    mptDefaults('showHints', prevHints);
    rethrow(benchErr);
end
mptDefaults('showHints', prevHints);


function s = localLogSlope(x, y)
%LOCALLOGSLOPE  Slope of log(y) regressed on log(x), least squares.
    lx = log(double(x(:)));
    ly = log(double(y(:)));
    p = [ones(numel(lx), 1), lx] \ ly;
    s = p(2);
end


function v = localR2(A, c, y)
%LOCALR2  Coefficient of determination for the fit A*c against y.
    e = y - A * c;
    d = y - mean(y);
    v = 1 - (e' * e) / (d' * d);
end
