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
%     machine? Section 4 fits this as the 'grid' law in relRouteCostMs
%     (+internal/selectMaInnerProductMethod.m); the ratio measured here
%     is the quantity that law has to reproduce.
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
        sigma, r, isRel, isPer, period, 'method', m, 'verbose', false);

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
        sigma, r, isRel, isPer, period, 'method', 'mobius', ...
        'verbose', false);
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

%% ---- Section 3: how each route scales with the value count ----

% The shipped count charges the translation-grid route a term in n^2 at
% every tuple order. This section measures whether that is the right
% shape, and does so for each route separately.
%
% Three arms, over the same sweep:
%
%   bulger           Bulger's method, selected by 'method'.
%   mobius/centres   the Mobius method on the materialised tuple-centres
%                    route.
%   mobius/grid      the Mobius method on the translation-grid route.
%
% The second choice is not reachable through 'method', which selects only
% between Bulger's method and the Mobius method; within the latter, a
% relative attribute takes the centres or the grid route according to a
% cost estimate. Left to itself that estimate switches partway through a
% sweep, and the resulting curve is a mixture of two routes rather than
% the scaling of either. mptDefaults('relAttrRoute', ...) pins it.
%
% The value counts are set per arm as well as per order. Bulger's method
% and the centres route both grow as K^(2r) -- at r = 3 the centres route
% already takes tens of seconds at K = 20 -- and their shape is settled
% long before the top of the range. The grid route is the one the n^2
% question is about, and it is cheap, so it gets the wide sweep. A wall
% budget stops any arm whose single call exceeds ARMBUDGETSEC, so a
% machine slower than the one these caps were chosen on abandons the arm
% rather than grinding through it.

rOrders = [2, 3, 4];
armNames = {'bulger', 'mobius/centres', 'mobius/grid'};
armRoute = {'auto', 'centres', 'grid'};
armMethod = {'bulger', 'mobius', 'mobius'};
% Rows: arm. Columns: r = 2, 3, 4.
% The caps come from measurement: at r = 3 the centres route runs 1.4 ms
% at K = 5, 90 ms at K = 10 and 39 s at K = 20, so its row stops well
% below the counts the grid row uses. Four points are enough to fit.
KPerArmOrder = { ...
    [5, 10, 20, 40, 80],      [5, 8, 12, 16],  [5, 7, 9, 11]; ...  % bulger
    [5, 10, 20, 40, 80],      [5, 8, 12, 16],  [5, 7, 9, 11]; ...  % centres
    [5, 10, 20, 40, 80, 160], [5, 10, 20, 40, 80, 160], ...
                                               [5, 10, 20, 40]};   % grid
ARMBUDGETSEC = 3;

prevRoute = mptDefaults('relAttrRoute');

fprintf('\nSection 3 -- scaling by route and tuple order\n');

for ri = 1:numel(rOrders)
    ra = rOrders(ri);
    for arm = 1:3
        Ks = KPerArmOrder{arm, ri};
        Ks = Ks(Ks >= ra);
        tok = [];
        Kok = [];
        mptDefaults('relAttrRoute', armRoute{arm});
        for ki = 1:numel(Ks)
            K = Ks(ki);
            rs = RandStream('twister', 'Seed', 1000 * K + ra);
            px = sort(rand(rs, 1, K) * period);
            py = sort(rand(rs, 1, K) * period);
            call = @() cosSimExpTens(px, [], py, [], sigma, ra, isRel, ...
                isPer, period, 'method', armMethod{arm}, 'verbose', false);
            try
                tic; call(); tWarm = toc;
            catch ME
                fprintf('    r = %d, %-14s: K = %d stopped: %s\n', ...
                    ra, armNames{arm}, K, ME.message);
                break;
            end
            if tWarm > ARMBUDGETSEC
                fprintf(['    r = %d, %-14s: K = %d took %.1f s on a ' ...
                         'single call, over the %g s budget; arm ' ...
                         'stopped here.\n'], ...
                    ra, armNames{arm}, K, tWarm, ARMBUDGETSEC);
                break;
            end
            Kok(end+1) = K;                          %#ok<SAGROW>
            tok(end+1) = internal.timeRepeated(call) * 1e3;  %#ok<SAGROW>
        end
        mptDefaults('relAttrRoute', 'auto');

        fprintf('  r = %d, %s\n', ra, armNames{arm});
        for ki = 1:numel(Kok)
            fprintf('%8d %14.3f ms\n', Kok(ki), tok(ki));
        end
        if numel(Kok) < 3
            fprintf('           too few points to fit\n');
            continue;
        end
        slope = localLogSlope(Kok, tok);
        Alin  = [ones(numel(Kok), 1), 2 * Kok(:)];
        Asq   = [ones(numel(Kok), 1), Kok(:).^2];
        clin  = Alin \ tok(:);
        csq   = Asq  \ tok(:);
        r2lin = localR2(Alin, clin, tok(:));
        r2sq  = localR2(Asq,  csq,  tok(:));
        fprintf(['           exponent %.2f | setup+linear R^2 %.4f ' ...
                 '(setup %.3f ms, %.4e ms per value) | setup+K^2 ' ...
                 'R^2 %.4f\n'], ...
            slope, r2lin, clin(1), clin(2), r2sq);
        if arm == 3
            if max(r2lin, r2sq) < 0.5
                fprintf(['           -> neither form fits; the time is ' ...
                         'dominated by fixed setup over this range, so ' ...
                         'the n^2 charge is unsupported but the ' ...
                         'alternative is not yet measurable.\n']);
            elseif r2lin > r2sq + 0.02 && clin(1) >= 0
                fprintf(['           -> setup plus a term linear in the ' ...
                         'value count; the shipped n^2 charge is too ' ...
                         'steep at this order.\n']);
            elseif r2sq > r2lin + 0.02
                fprintf(['           -> consistent with the shipped n^2 ' ...
                         'charge at this order.\n']);
            else
                fprintf(['           -> the two forms fit equally well; ' ...
                         'the range is too narrow to separate them.\n']);
            end
        end
    end
end

mptDefaults('relAttrRoute', prevRoute);

%% ---- Section 4: calibrating the relative-mode cost model ----

% Where the auto-dispatched time exceeds the better of the two forced
% times, the gap is a routing decision rather than a missing technique:
% both methods reach mobius.relInnerBatched, and the question is only
% which of them the selector picks. In relative mode the Mobius method
% can run 15 to 300 times faster than Bulger's on the same cell, so a
% misroute there is expensive.
%
% This section supplies what a fit needs: for each cell, both methods
% timed, and beside them the two wall times the selector's comparison
% actually rests on. The ratio of predicted to measured is the quantity
% to calibrate against; the sign of the disagreement says which side is
% mispriced.
%
% Relative mode only, both periodicities, since that is where the
% misprediction lives. The node count is reconstructed the way
% relInnerBatched sets it, so the prediction seen here is the one the
% dispatcher forms.

fprintf('\nSection 4 -- relative-mode cost model against measurement\n');
fprintf('%-20s %9s %9s %8s %10s %10s %8s %8s\n', ...
    'case', 'bulger', 'mobius', 'picked', 'pred bul', 'pred mob', ...
    'bul p/m', 'mob p/m');

s6ts = mptDefaults('truncationSigmas');
s6margin = internal.relWindowMargin(s6ts);
s6cells = {2, 20; 2, 40; 2, 80; 3, 10; 3, 20; 4, 8; 4, 10};
s6bad = 0;
for ci = 1:size(s6cells, 1)
    for s6per = [false, true]
        s6r = s6cells{ci, 1};
        s6K = s6cells{ci, 2};
        if s6per, s6P = period; else, s6P = 0; end
        rs = RandStream('twister', 'Seed', 613 * s6K + s6r);
        px = sort(rand(rs, 1, s6K) * period);
        py = sort(rand(rs, 1, s6K) * period);
        wx = 0.5 + rand(rs, 1, s6K);
        wy = 0.5 + rand(rs, 1, s6K);

        callB = @() cosSimExpTens(px, wx, py, wy, sigma, s6r, 1, s6per, ...
            s6P, 'method', 'bulger', 'verbose', false);
        callM = @() cosSimExpTens(px, wx, py, wy, sigma, s6r, 1, s6per, ...
            s6P, 'method', 'mobius', 'verbose', false);
        try
            callB(); tB = internal.timeRepeated(callB) * 1e3;
        catch
            tB = NaN;
        end
        try
            callM(); tM = internal.timeRepeated(callM) * 1e3;
        catch
            tM = NaN;
        end

        % Node count as relInnerBatched sets it.
        if s6per
            s6nu = internal.autoNtauDefault(period, sigma);
            s6sop = sigma / period;
        else
            % The node density is tied to the accuracy floor and to the
            % tuple order, so it is resolved per cell.
            s6sps = internal.resolveSamplesPerSigma([], s6r, s6ts);
            s6span = (max(px) - min(px)) + (max(py) - min(py)) ...
                     + 2 * s6margin * sigma;
            s6nu = max(64, ceil(max(s6span, 1.0) / sigma * s6sps));
            s6sop = 0;
        end
        [s6pick, s6pw, s6orb] = internal.selectMaInnerProductMethod( ...
            s6r, s6K, 1, 1, 1, s6per, ~s6per, s6per, s6sop, 'auto', ...
            false, true, s6nu, s6K);

        s6faster = 'bulger';
        if tM < tB, s6faster = 'mobius'; end
        s6flag = '';
        if ~strcmp(s6pick, s6faster)
            s6flag = '  <- mispredicted';
            s6bad = s6bad + 1;
        end
        fprintf('%-20s %9.2f %9.2f %8s %10.1f %10.1f %8.2f %8.2f%s\n', ...
            sprintf('r=%d K=%d per=%d', s6r, s6K, s6per), ...
            tB, tM, s6pick, s6pw, s6orb, s6pw / tB, s6orb / tM, s6flag);
    end
end

fprintf(['\n    %d of %d cells routed to the slower method.\n' ...
         '    The last two columns are predicted over measured: 1 is a ' ...
         'calibrated\n    model, above 1 over-prices that side, below 1 ' ...
         'under-prices it. A\n    correction has to bring both near 1 ' ...
         'across the grid, not just flip\n    the verdict on these ' ...
         'cells.\n'], s6bad, 2 * size(s6cells, 1));

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
                 '    form, so the grid law in relRouteCostMs should\n' ...
                 '    not be refitted from these numbers.\n'], ...
                 ratios(1), ratios(end));
    else
        fprintf(['    The ratio is not monotone in n. Median %.4f is a\n' ...
                 '    usable per-operation reading for the grid route,\n' ...
                 '    against which the fitted law in relRouteCostMs\n' ...
                 '    (+internal/selectMaInnerProductMethod.m) can be\n' ...
                 '    sanity-checked.\n'], median(ratios));
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
