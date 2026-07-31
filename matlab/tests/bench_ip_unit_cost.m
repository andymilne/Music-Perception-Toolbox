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

%% ---- Section 4: does the dedicated single-multiset stack earn its place? ----

% A single multiset (A = N = 1) is handled in MATLAB by a stack of
% kernels, cost model and dispatcher separate from the multi-attribute
% one: localSelectSingleMultisetMethod, localOrbitGridUnitCost,
% localOrbitIPGridFactors, localCosSimSingleMultisetOrbit and
% localOrbitIPsCorrupted have no counterpart in Python, which routes the
% corner through its general path and keeps only a deduplication cache
% for repeated collections.
%
% The duplication has already cost correctness: the centres-versus-grid
% gate, the relAttrRoute lever and three cost-model corrections all live
% in the multi-attribute path, so none of them reaches this workload in
% MATLAB while all of them reach it in Python.
%
% This section runs identical work down both MATLAB paths and reports
% the ratio, so the decision to keep or remove the dedicated stack rests
% on measurement. Values are compared first: a timing comparison between
% paths that disagree would be meaningless.
%
% The batched and list entry forms deduplicate and then call back into
% cosSimExpTens once per unique pair, so they reach the same redirect;
% they are covered here because the flip changes what those inner calls
% do, and because dedup means a batched result is assembled from cached
% values whose provenance the scalar cells cannot exercise.

fprintf(['\nSection 4 -- dedicated single-multiset stack against the ' ...
         'MA path\n']);

prevPath = mptDefaults('singleMultisetPath');

% Cases are assembled first, then run in one loop: a script cannot carry
% nested functions sharing a workspace, so the comparison is written out
% once rather than wrapped in a helper.
s4cases = {};   % {label, function handle}

% --- 4a. Scalar grid: both modes, both periodicities, r = 2 to 4,
%         non-uniform weights throughout. ---
for s4r = [2, 3, 4]
    for s4K = [10, 20, 40, 80]
        if s4K < s4r, continue; end
        if s4r == 4 && s4K > 20, continue; end   % K^8; keep it tractable
        for s4mode = 1:4
            s4isRel = (s4mode == 2 || s4mode == 4);
            s4isPer = (s4mode == 3 || s4mode == 4);
            if s4isPer, s4P = period; else, s4P = 0; end
            rs = RandStream('twister', 'Seed', 77 * s4K + s4r + 13 * s4mode);
            px = sort(rand(rs, 1, s4K) * period);
            py = sort(rand(rs, 1, s4K) * period);
            wx = 0.5 + rand(rs, 1, s4K);
            wy = 0.5 + rand(rs, 1, s4K);
            s4cases(end+1, :) = { ...
                sprintf('4a r=%d K=%d rel=%d per=%d', s4r, s4K, ...
                        s4isRel, s4isPer), ...
                @() cosSimExpTens(px, wx, py, wy, sigma, s4r, ...
                    s4isRel, s4isPer, s4P, 'verbose', false)}; %#ok<SAGROW>
        end
    end
end

% --- 4b. Unweighted, to separate the weighted path from the plain one. ---
rs = RandStream('twister', 'Seed', 991);
s4bx = sort(rand(rs, 1, 30) * period);
s4by = sort(rand(rs, 1, 30) * period);
s4cases(end+1, :) = {'4b unweighted r=2 rel-per', ...
    @() cosSimExpTens(s4bx, [], s4by, [], sigma, 2, 1, 1, period, ...
        'verbose', false)};

% --- 4c. Spectral augmentation: the option whose plumbing differs most
%         between entry forms. MATLAB accepts 'spectrum' only where at
%         least one operand is a 2-D matrix with both dimensions above
%         one, so the case is presented in that form. (Python accepts it
%         on scalar input as well; that divergence is reported
%         separately and is not this benchmark's business.) ---
rs = RandStream('twister', 'Seed', 992);
s4cx = sort(rand(rs, 1, 6) * 1200);
s4cy = zeros(5, 6);
for ii = 1:5
    s4cy(ii, :) = sort(rand(rs, 1, 6) * 1200);
end
s4spec = {'harmonic', 10, 'powerlaw', 0.75};
s4cases(end+1, :) = {'4c spectrum batched 5x6', ...
    @() cosSimExpTens(s4cx, [], s4cy, [], sigma, 2, 1, 1, period, ...
        'spectrum', s4spec, 'verbose', false)};

% --- 4d. Batched raw against a reference row, dedup on and off. The
%         batch deduplicates and calls back per unique pair, so the flip
%         changes what each inner call does; the whole vector is
%         compared, not one entry. Repeated rows exercise the cache. ---
rs = RandStream('twister', 'Seed', 993);
s4ref = sort(rand(rs, 1, 8) * period);
s4M = 60;
s4batch = zeros(s4M, 8);
for ii = 1:s4M
    s4batch(ii, :) = sort(rand(rs, 1, 8) * period);
end
s4batch(2:3:end, :) = repmat(s4batch(1, :), numel(2:3:s4M), 1);
s4cases(end+1, :) = {'4d batched 60x8 dedup on', ...
    @() cosSimExpTens(s4ref, [], s4batch, [], sigma, 2, 1, 1, period, ...
        'verbose', false)};
% 'dedup', false is a documented no-op on the batched-raw path and warns
% on every call, including every repeat inside timeRepeated, so it is not
% exercised here. The dedup cache is covered by the repeated rows above.
s4cases(end+1, :) = {'4d batched 60x8 r=3', ...
    @() cosSimExpTens(s4ref, [], s4batch, [], sigma, 3, 1, 1, period, ...
        'verbose', false)};

% --- 4e. Density-list input, cell against cell and scalar broadcast. ---
s4listA = cell(1, 12);
s4listB = cell(1, 12);
for ii = 1:12
    pa = sort(rand(rs, 1, 7) * period);
    pb = sort(rand(rs, 1, 7) * period);
    s4listA{ii} = buildExpTens(pa, [], sigma, 2, 1, 1, period, ...
        'verbose', false);
    s4listB{ii} = buildExpTens(pb, [], sigma, 2, 1, 1, period, ...
        'verbose', false);
end
s4cases(end+1, :) = {'4e list 12 vs 12', ...
    @() cell2mat(cosSimExpTens(s4listA, s4listB, 'verbose', false))};
s4cases(end+1, :) = {'4e list scalar broadcast', ...
    @() cell2mat(cosSimExpTens(s4listA{1}, s4listB, 'verbose', false))};

fprintf('%-34s %10s %10s %7s %13s\n', ...
        'case', 'ded(ms)', 'MA(ms)', 'MA/ded', 'worst diff');

s4worstDiff = 0;
s4worstWhere = '';
s4ratios = [];
for ci = 1:size(s4cases, 1)
    s4label = s4cases{ci, 1};
    s4fn    = s4cases{ci, 2};
    mptDefaults('singleMultisetPath', 'auto');
    s4failed = false;
    try
        s4vD = s4fn();
        s4tD = internal.timeRepeated(s4fn) * 1e3;
    catch s4err
        s4failed = true;
        fprintf(['%-34s  rejected by the dedicated stack: %s\n' ...
                 '%34s  The case is malformed, not the path; fix the ' ...
                 'case.\n'], s4label, s4err.message, '');
    end
    if s4failed
        mptDefaults('singleMultisetPath', 'auto');
        continue;
    end
    mptDefaults('singleMultisetPath', 'ma');
    try
        s4vM = s4fn();
        s4tM = internal.timeRepeated(s4fn) * 1e3;
    catch s4err
        s4failed = true;
        fprintf(['%-34s  the MA path rejected this shape: %s\n' ...
                 '%34s  The dedicated stack cannot simply be deleted; ' ...
                 'report this.\n'], s4label, s4err.message, '');
    end
    mptDefaults('singleMultisetPath', 'auto');
    if s4failed, continue; end
    if ~isequal(size(s4vM), size(s4vD))
        fprintf('%-34s  shape differs between paths; report this.\n', ...
                s4label);
        continue;
    end
    s4d = max(abs(s4vM(:) - s4vD(:)));
    if s4d > s4worstDiff
        s4worstDiff = s4d;
        s4worstWhere = s4label;
    end
    s4ratios(end+1) = s4tM / s4tD;   %#ok<SAGROW>
    fprintf('%-34s %10.3f %10.3f %7.2f %13.3e\n', ...
            s4label, s4tD, s4tM, s4tM / s4tD, s4d);
end

mptDefaults('singleMultisetPath', prevPath);

fprintf('\n');
if isempty(s4ratios)
    fprintf('    No case completed; the MA path is not a drop-in.\n');
else
    fprintf(['    Cases compared: %d. Worst difference: %.3e (%s).\n' ...
             '    MA/dedicated ratio: median %.2f, worst %.2f.\n'], ...
        numel(s4ratios), s4worstDiff, s4worstWhere, ...
        median(s4ratios), max(s4ratios));
    if s4worstDiff > 1.5e-8
        fprintf(['    That exceeds the truncation floor: the two paths ' ...
                 'do not agree, and\n    the timings are moot. Report ' ...
                 'this before anything is deleted.\n']);
    elseif max(s4ratios) < 1.3
        fprintf(['    Values agree to within the floor and the MA path ' ...
                 'is nowhere more\n    than 30%% slower: the dedicated ' ...
                 'stack does not earn its keep.\n']);
    else
        fprintf(['    Values agree, but the MA path is materially ' ...
                 'slower in at least one\n    case. Report which, ' ...
                 'before anything is deleted.\n']);
    end
end

%% ---- Section 5: where the MA path is slower, is it route or cost? ----

% Section 4 finds the two paths agreeing on value everywhere but the MA
% path far slower in a few cells, all of them relative and most of them
% non-periodic. That is either a route the MA path chooses badly or a
% cost it cannot avoid, and the two call for different remedies: the
% first is a gate correction, the second means the dedicated stack is
% buying something real.
%
% relAttrRoute pins the route inside the MA path, so timing each cell
% under 'centres' and 'grid' separates the two. In Python the same cells
% show 'auto' tracking 'grid' closely while 'centres' runs hundreds of
% times slower, so if a forced route here comes in near the dedicated
% figure the fault is the gate, not the path.

fprintf('\nSection 5 -- outlier diagnosis (relative mode)\n');
fprintf('%-26s %9s %9s %9s %9s\n', ...
        'case', 'ded(ms)', 'MA auto', 'MA centr', 'MA grid');

s5cells = { ...
    2, 40, false; ...
    2, 80, false; ...
    2, 80, true;  ...
    4, 10, false; ...
    4, 10, true};

prevPath5 = mptDefaults('singleMultisetPath');
prevRoute5 = mptDefaults('relAttrRoute');
for ci = 1:size(s5cells, 1)
    s5r = s5cells{ci, 1};
    s5K = s5cells{ci, 2};
    s5isPer = s5cells{ci, 3};
    if s5isPer, s5P = period; else, s5P = 0; end
    rs = RandStream('twister', 'Seed', 77 * s5K + s5r + 13 * (2 + 2 * s5isPer));
    px = sort(rand(rs, 1, s5K) * period);
    py = sort(rand(rs, 1, s5K) * period);
    wx = 0.5 + rand(rs, 1, s5K);
    wy = 0.5 + rand(rs, 1, s5K);
    call = @() cosSimExpTens(px, wx, py, wy, sigma, s5r, 1, s5isPer, ...
        s5P, 'verbose', false);
    mptDefaults('singleMultisetPath', 'auto');
    mptDefaults('relAttrRoute', 'auto');
    call();
    s5ded = internal.timeRepeated(call) * 1e3;
    s5t = nan(1, 3);
    s5routes = {'auto', 'centres', 'grid'};
    mptDefaults('singleMultisetPath', 'ma');
    for ri = 1:3
        mptDefaults('relAttrRoute', s5routes{ri});
        try
            call();
            s5t(ri) = internal.timeRepeated(call) * 1e3;
        catch
            s5t(ri) = NaN;   % route inadmissible here
        end
    end
    mptDefaults('relAttrRoute', 'auto');
    mptDefaults('singleMultisetPath', 'auto');
    fprintf('%-26s %9.3f %9.3f %9.3f %9.3f\n', ...
        sprintf('r=%d K=%d per=%d', s5r, s5K, s5isPer), ...
        s5ded, s5t(1), s5t(2), s5t(3));
end
mptDefaults('relAttrRoute', prevRoute5);
mptDefaults('singleMultisetPath', prevPath5);

fprintf(['    If a forced route comes in near the dedicated figure, the ' ...
         'gate is\n    choosing badly and the fix is the gate. If every ' ...
         'route is far slower,\n    the MA path cannot reach the ' ...
         'dedicated stack''s cost and the stack\n    is buying ' ...
         'something real.\n']);

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
