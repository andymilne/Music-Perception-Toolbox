function bench_mobius(varargin)
%BENCH_MOBIUS  MATLAB twin of bench_mobius.py.
%
%   Wall-time comparison of the toolbox's alternative routes, across
%   tuple size r, multiset size K, and the four modes.
%
%   Two tasks, selected with the 'task' parameter:
%
%     'ip'    the inner product (cosSimExpTens): 'centres' (unrestricted
%             enumeration of the tuple centres, the O(K^(2r)) baseline),
%             'bulger' (the within-r-ad decomposition), and 'mobius'
%             (the Moebius-orbit decomposition).
%     'eval'  point evaluation (evalExpTens): 'centres' (the centres
%             array) and 'mobius' (the Moebius point evaluator).
%             Bulger's identity has no analogue here, there being no
%             two-sided pairing to exploit.
%     'both'  (default) runs each in turn.
%
%   The comparison of interest is the RELATIVE timing and its scaling
%   with K; absolute times establish only feasibility, and are not
%   comparable across machines or languages.
%
%   Each method is forced explicitly rather than routed by the toolbox's
%   automatic dispatch: the aim is the real-world cost and feasibility of
%   the three routes in each regime, not the routing policy.
%
%   'centres' is the toolbox's own unrestricted enumeration -- the
%   O(K^(2r)) baseline the Supplement's cost accounting starts from.
%   'referenceR' is the SAME enumeration written here in plain MATLAB
%   with the X side restricted to combinations and multiplied by r!;
%   timed against a local unrestricted twin it isolates Bulger's saving,
%   the two differing only in the restriction. (Before v2.2 the toolbox
%   had no unrestricted route: 'direct' named Bulger's method.)
%
%   The relative modes' orbit route is pinned with the 'relRoute'
%   parameter. The toolbox otherwise chooses between a materialised
%   tuple-centres route and a translation-grid route on a cost estimate,
%   and the two have very different profiles: on 'auto' the mobius column
%   becomes a mixture of two implementations, visible as non-monotonic
%   timings in K. Run once per route and report them separately.
%
%   Bulger's method is the accuracy reference: it restricts a sum of
%   non-negative terms, so it does not suffer the cancellation the
%   alternating Moebius sums can. Each method's relative deviation from
%   it is recorded per cell (column rel_dev), so the cancellation
%   regimes -- K close to r, and relative-periodic at large sigma/P --
%   appear as loss of agreement rather than a pass/fail verdict.
%
%   Forcing works as in Python: an explicit 'method' returns from
%   internal.selectMaInnerProductMethod before the structural
%   pre-screen, so 'mobius' really does run the orbit route.
%
%   Densities are built once per cell, outside the timed region: the
%   timed unit is the similarity call alone, so construction is not
%   charged to any method. Informational hints and the post-hoc guards
%   are switched off for the duration and restored on exit.
%
%   Protocol matches the Python script cell for cell: same sweep, same
%   sigma and period, same seeded positions and weights (a plain LCG is
%   used in both languages so the inputs are identical), correctness
%   asserted before timing with truncation disabled, repetitions
%   auto-scaled to ~150 ms per timed unit after a warm-up call.
%
%   Usage:
%     bench_mobius                 % standard sweep
%     bench_mobius('quick', true)  % small sweep, smoke test
%     bench_mobius('truncation', true)   % add the 6-sigma pass
%     bench_mobius('out', 'results.csv')
%
%   Writes a tidy CSV with the same columns as the Python script, so the
%   two can be concatenated and compared directly.

ip = inputParser;
ip.addParameter('quick', false, @islogical);
ip.addParameter('truncation', false, @islogical);
ip.addParameter('task', 'both', @(s) any(strcmp(s, {'ip','eval','both'})));
ip.addParameter('relRoute', 'auto', @(s) any(strcmp(s, {'auto','centres','grid'})));
ip.addParameter('out', '', @ischar);
ip.parse(varargin{:});
opt = ip.Results;
if isempty(opt.out)
    opt.out = sprintf('bench_mobius_matlab_%s.csv', opt.relRoute);
end

% Defaults chosen so a full run finishes in a few minutes. The orbit
% method's advantage is unambiguous by r = 4, K = 34; larger cells cost
% minutes each and add only confirmation. Widen with 'rs' and 'ks'.
RS_FULL = [2 3 4];
KS_FULL = [6 12 20 34];
MODES = { false false 'absolute non-periodic'
          false true  'absolute periodic'
          true  false 'relative non-periodic'
          true  true  'relative periodic' };
IP_METHODS = {'centres', 'reference', 'referenceR', 'bulger', 'mobius'};
EVAL_METHODS = {'centres', 'mobius'};
% 'centres' is the toolbox's own unrestricted enumeration. 'reference'
% and 'referenceR' are a local pair -- the same enumeration unrestricted
% and with the X side restricted to combinations -- whose ratio isolates
% Bulger's r! saving, the two differing only in the restriction.
N_QUERIES = 64;   % query points per evaluation cell, held fixed
METHODS = IP_METHODS;

SIGMA  = 30.0;
PERIOD = 1200.0;
TOL    = 1e-9;
TARGET_MS = 150.0;
SEED = 20260822;

% --- work limits ------------------------------------------------------
% Both enumerating methods cost about one unit per tuple PAIR, and the
% count explodes: at r = 5, K = 48 it is 3.5e14. Cells whose predicted
% cost exceeds CELL_BUDGET_S are skipped. Periodic wrapping multiplies
% the per-pair cost by roughly 40 and the relative quadrature by about 4;
% both multipliers deliberately over-estimate, so the budget errs toward
% skipping rather than running for minutes.
CELL_BUDGET_S = 2.0;
PER_SCALE = 40.0;
REL_SCALE = 4.0;
ORBIT_S = containers.Map( ...
    {'00', '01', '10', '11'}, ...
    {[6e-4 1.3e-3 3.8e-3 1.2e-2 4e-2], ...
     [6e-4 1.2e-3 3.6e-3 1.2e-2 4e-2], ...
     [1.3e-3 1.6e-2 8.5e-2 3.4e-1 1.4e0], ...
     [4e-4 9e-4 2.5e-2 9.1e-2 3.6e-1]});   % index by r-1, r = 2..6

if opt.quick
    RS = [2 3]; KS = [6 12 20]; modes = MODES(1, :);
else
    RS = RS_FULL; KS = KS_FULL; modes = MODES;
end

% Silence one-time informational hints and switch off the post-hoc
% guards: with guards on, a route that diverts pays for both routes, so
% the measured cost would not be the cost of the route being forced.
% Both are restored on exit, including on error.
prevDefaults = mptDefaults('showHints', false, 'postHocGuards', false, ...
                          'relAttrRoute', opt.relRoute);
cleanup = onCleanup(@() mptDefaults(prevDefaults));

rows = {};
failures = {};
nSkipped = 0;
state = SEED;                                   % shared LCG state
rate = calibrateRate(SIGMA, PERIOD);

fprintf('MATLAB %s\n%s\n', version, computer);
fprintf('calibrated at %.1f ns per tuple pair; cell budget %g s; rel route %s\n', ...
        rate * 1e9, CELL_BUDGET_S, opt.relRoute);
if strcmp(opt.relRoute, 'auto')
    fprintf(['note: the relative modes mix two orbit routes on auto; ' ...
             'rerun with relRoute ''centres'' and ''grid''\n\n']);
else
    fprintf('\n');
end
fprintf('%-22s%3s%5s  ', 'mode', 'r', 'K');
fprintf('%12s', METHODS{:}); fprintf('   winner\n');
fprintf('%s\n', repmat('-', 1, 22 + 8 + 12 * numel(METHODS) + 9));

if any(strcmp(opt.task, {'ip', 'both'}))
for mi = 1:size(modes, 1)
    isRel = modes{mi, 1}; isPer = modes{mi, 2}; mlabel = modes{mi, 3};
    for r = RS
        for K = KS
            if K < r + 1, continue; end

            [p, wp, state] = draw(K, PERIOD, state);
            [q, wq, state] = draw(K, PERIOD, state);
            [densA, densB] = buildPair(p, wp, q, wq, SIGMA, r, isRel, isPer, PERIOD);

            % --- correctness first, untruncated ------------------------
            vals = containers.Map();
            for k = 1:numel(METHODS)
                m = METHODS{k};
                if strcmp(m, 'referenceR') && (isRel || isPer)
                    continue;          % local twin defined for the plain case
                end
                pS = predictS(m, r, K, rate, isRel, isPer, ORBIT_S, PER_SCALE, REL_SCALE);
                if pS > CELL_BUDGET_S
                    nSkipped = nSkipped + 1;
                    continue;
                end
                try
                    if strncmp(m, 'reference', 9)
                        vals(m) = referenceIp(p, wp, q, wq, r, SIGMA, isRel, ...
                                              isPer, strcmp(m, 'referenceR'));
                    else
                        vals(m) = callSim(densA, densB, m, Inf);
                    end
                catch err
                    failures{end+1} = sprintf('%s r=%d K=%d %s: %s', ...
                        mlabel, r, K, m, err.message); %#ok<AGROW>
                end
            end
            ks = vals.keys();
            devs = containers.Map('KeyType', 'char', 'ValueType', 'double');
            if ~isempty(ks)
                % Bulger is the accuracy reference; where it was skipped as
                % over budget there is none, and the deviation is NaN rather
                % than a spurious zero.
                if vals.isKey('bulger'), ref = vals('bulger'); else, ref = NaN; end
                for k = 1:numel(ks)
                    v = vals(ks{k});
                    if strcmp(ks{k}, 'bulger') || ~isfinite(v) || ~isfinite(ref)
                        devs(ks{k}) = NaN;
                    else
                        devs(ks{k}) = abs(v - ref) / max(1, abs(ref));
                    end
                    if ~isfinite(v) || devs(ks{k}) > TOL
                        failures{end+1} = sprintf('%s r=%d K=%d %s deviates from bulger by %.2e', ...
                            mlabel, r, K, ks{k}, devs(ks{k})); %#ok<AGROW>
                    end
                end
            end

            % --- timing -------------------------------------------------
            times = containers.Map('KeyType', 'char', 'ValueType', 'double');
            for k = 1:numel(ks)
                m = ks{k};
                if strncmp(m, 'reference', 9)
                    fn = @() referenceIp(p, wp, q, wq, r, SIGMA, isRel, ...
                                         isPer, strcmp(m, 'referenceR'));
                else
                    fn = @() callSim(densA, densB, m, Inf);
                end
                try
                    [t, n] = timeIt(fn, TARGET_MS);
                    times(m) = t;
                    rows{end+1} = {'matlab', mlabel, isRel, isPer, r, K, m, ...
                                   'inf', t, n, vals(m), devs(m), opt.relRoute}; %#ok<AGROW>
                catch err
                    failures{end+1} = sprintf('%s r=%d K=%d %s timing: %s', ...
                        mlabel, r, K, m, err.message); %#ok<AGROW>
                end
            end

            if opt.truncation
                for k = 1:numel(ks)
                    m = ks{k};
                    fn = @() callSim(densA, densB, m, 6.0);
                    try
                        [t, n] = timeIt(fn, TARGET_MS);
                        rows{end+1} = {'matlab', mlabel, isRel, isPer, r, K, m, ...
                                       '6', t, n, fn(), NaN, opt.relRoute}; %#ok<AGROW>
                    catch
                    end
                end
            end

            % --- report the cell ----------------------------------------
            fprintf('%-22s%3d%5d  ', mlabel, r, K);
            best = ''; bestT = Inf;
            for k = 1:numel(METHODS)
                m = METHODS{k};
                if times.isKey(m)
                    fprintf('%11.3fm', times(m) * 1e3);
                    if times(m) < bestT
                        bestT = times(m); best = m;
                    end
                else
                    fprintf('%12s', '-');
                end
            end
            fprintf('   %s\n', best);
        end
    end
end

end   % task 'ip'

% --- point evaluation ---------------------------------------------------
if any(strcmp(opt.task, {'eval', 'both'}))
fprintf('\npoint evaluation, %d query points per cell\n\n', N_QUERIES);
fprintf('%-22s%3s%5s  ', 'mode', 'r', 'K');
fprintf('%12s', EVAL_METHODS{:}); fprintf('   winner\n');
fprintf('%s\n', repmat('-', 1, 22 + 8 + 12 * numel(EVAL_METHODS) + 9));
for mi = 1:size(modes, 1)
    isRel = modes{mi, 1}; isPer = modes{mi, 2}; mlabel = modes{mi, 3};
    for r = RS
        for K = KS
            if K < r + 1, continue; end
            [p, wp, state] = draw(K, PERIOD, state);
            dens = buildExpTens(p, wp, SIGMA, r, isRel, isPer, PERIOD, ...
                                'verbose', false);
            dim = max(r - double(isRel), 1);
            [u, state] = lcg(dim * N_QUERIES, state);
            pts = reshape(u, dim, N_QUERIES) * PERIOD;

            vals = containers.Map(); tms = containers.Map();
            for k = 1:numel(EVAL_METHODS)
                mm = EVAL_METHODS{k};
                try
                    vals(mm) = sum(sum(evalExpTens(dens, pts, 'method', mm, ...
                        'truncationSigmas', Inf, 'verbose', false)));
                    fn = @() evalExpTens(dens, pts, 'method', mm, ...
                        'truncationSigmas', Inf, 'verbose', false);
                    [t, n] = timeIt(fn, TARGET_MS);
                    tms(mm) = t;
                    rows{end+1} = {'matlab', mlabel, isRel, isPer, r, K, mm, ...
                                   'inf', t, n, vals(mm), NaN, opt.relRoute, ...
                                   'eval', N_QUERIES}; %#ok<AGROW>
                catch err
                    failures{end+1} = sprintf('eval %s r=%d K=%d %s: %s', ...
                        mlabel, r, K, mm, err.message); %#ok<AGROW>
                end
            end
            % centres is the reference for evaluation. The comparison
            % needs an absolute floor as well as a relative one: at query
            % points far from every centre the centres route returns
            % exactly 0 while the alternating Moebius sum returns dust of
            % order 1e-16, and a purely relative measure divides by zero.
            if vals.isKey('centres') && vals.isKey('mobius')
                dev = abs(vals('mobius') - vals('centres')) / ...
                      max(abs(vals('centres')), 1);
                if dev > TOL
                    failures{end+1} = sprintf( ...
                        'eval %s r=%d K=%d mobius deviates from centres by %.2e', ...
                        mlabel, r, K, dev); %#ok<AGROW>
                end
            end
            fprintf('%-22s%3d%5d  ', mlabel, r, K);
            bestE = ''; bestT = Inf;
            for k = 1:numel(EVAL_METHODS)
                mm = EVAL_METHODS{k};
                if tms.isKey(mm)
                    fprintf('%11.3fm', tms(mm) * 1e3);
                    if tms(mm) < bestT, bestT = tms(mm); bestE = mm; end
                else
                    fprintf('%12s', '-');
                end
            end
            fprintf('   %s\n', bestE);
        end
    end
end
end   % task 'eval'

% --- scaling fits -------------------------------------------------------
fprintf('\nfitted exponent of K over the three largest K, untruncated:\n');
fprintf('  %-22s%3s   %10s%10s%10s     predicted\n', 'mode', 'r', ...
        'direct', 'bulger', 'mobius');
for mi = 1:size(modes, 1)
    mlabel = modes{mi, 3};
    for r = RS
        fprintf('  %-22s%3d   ', mlabel, r);
        for m = METHODS
            [kk, tt] = pick(rows, mlabel, r, m{1});
            e = fitExponent(kk, tt);
            if isfinite(e), fprintf('%10.2f', e); else, fprintf('%10s', '-'); end
        end
        fprintf('     %d / %d / 2\n', 2 * r, 2 * r);
    end
end

fprintf('\nmonotonicity of the orbit method in K (a fall signals a route change, not noise):\n');
for mi = 1:size(modes, 1)
    mlabel = modes{mi, 3};
    for r = RS
        [kk, tt] = pick(rows, mlabel, r, 'mobius');
        if numel(kk) < 3, continue; end
        [kk, ord] = sort(kk); tt = tt(ord);
        drops = kk(find(tt(2:end) < 0.85 * tt(1:end-1)) + 1);
        if ~isempty(drops)
            fprintf('   %-22s r=%d   falls at K = %s\n', mlabel, r, mat2str(drops));
        end
    end
end

fprintf('\nagreement with Bulger (relative deviation, untruncated);\n');
fprintf('large values indicate cancellation, not error:\n');
devRows = {};
for i = 1:numel(rows)
    if strcmp(rows{i}{7}, 'mobius') && strcmp(rows{i}{8}, 'inf') && isfinite(rows{i}{12})
        devRows{end+1} = rows{i}; %#ok<AGROW>
    end
end
if ~isempty(devRows)
    dv = cellfun(@(x) x{12}, devRows);
    [~, ord] = sort(dv, 'descend');
    for i = 1:min(8, numel(ord))
        x = devRows{ord(i)};
        fprintf('   %-22s r=%d  K=%3d   %.2e\n', x{2}, x{5}, x{6}, x{12});
    end
end

fprintf('\ncells skipped as over budget: %d (raise CELL_BUDGET_S to include)\n', nSkipped);

if isempty(failures)
    fprintf('\nall available methods agreed to within %g on every cell\n', TOL);
else
    fprintf('\nFAILURES (%d):\n', numel(failures));
    for i = 1:min(25, numel(failures)), fprintf('   %s\n', failures{i}); end
end

writeCsv(opt.out, rows);
fprintf('\nwrote %s  (%d rows)\n', opt.out, numel(rows));
end

% =======================================================================

function s = referenceIp(p, w, q, wq, r, sigma, isRel, isPer, restrict)
%REFERENCEIP  Unrestricted double enumeration over D x D, in plain MATLAB.
%
%   The O(K^(2r)) baseline the Supplement's cost accounting starts from:
%   every ordered r-tuple of distinct indices on each side against every
%   such tuple on the other, with no restriction to combinations and no
%   partition algebra. The toolbox has no unrestricted route ('direct'
%   and 'bulger' both enter the same core, which is always called
%   permutation-side against combination-side), so without this the
%   factor r! is asserted rather than measured. It also shares no code
%   with the toolbox, making agreement with it a stronger check than
%   Bulger against Moebius, which share a core.
%
%   Absolute non-periodic only, and small K only.
%   With `restrict` set, the X side is reduced to increasing tuples and
%   the result multiplied by r! (Supplement Eq. S8). That is the ONLY
%   difference from the unrestricted form, so the timing ratio between
%   the two isolates Bulger's saving rather than an implementation gap.
s = NaN;
if isRel || isPer, return; end
if nargin < 9, restrict = false; end
fac = 1; if restrict, fac = factorial(r); end
[cxR, wxR] = localTuples(p(:), w(:), r, restrict);
[cyR, wyR] = localTuples(q(:), wq(:), r, restrict);
[cx,  wx ] = localTuples(p(:), w(:), r, false);
[cy,  wy ] = localTuples(q(:), wq(:), r, false);
xy = fac * localIp(cxR, wxR, cy, wy, sigma);
xx = fac * localIp(cxR, wxR, cx, wx, sigma);
yy = fac * localIp(cyR, wyR, cy, wy, sigma);
s = xy / sqrt(xx * yy);
end

function [C, W] = localTuples(pos, wt, r, restrict)
if restrict
    idx = nchoosek(1:numel(pos), r);        % increasing tuples
else
    idx = localOrderedTuples(numel(pos), r);
end
C = pos(idx);
W = prod(wt(idx), 2);
if r == 1, C = C(:); W = W(:); end
end

function idx = localOrderedTuples(K, r)
idx = (1:K)';
for d = 2:r
    n = size(idx, 1);
    idx = [repmat(idx, K, 1), reshape(repmat(1:K, n, 1), [], 1)]; %#ok<AGROW>
    keep = true(size(idx, 1), 1);
    for c = 1:(d - 1)
        keep = keep & (idx(:, c) ~= idx(:, d));
    end
    idx = idx(keep, :);
end
end

function v = localIp(ca, wa, cb, wb, sigma)
d2 = zeros(size(ca, 1), size(cb, 1));
for c = 1:size(ca, 2)
    d2 = d2 + (ca(:, c) - cb(:, c)').^2;
end
v = sum(sum((wa * wb') .* exp(-d2 / (4 * sigma^2))));
end

function [p, w, state] = draw(K, period, state)
%DRAW  Positions and weights from a plain LCG, identical across languages.
[u, state] = lcg(2 * K, state);
p = sort(u(1:K) * 3 * period);
w = 0.4 + 0.6 * u(K+1:2*K);
p = p(:); w = w(:);
end

function [u, state] = lcg(n, state)
%LCG  Numerical Recipes ranqd1 constants; values in [0, 1).
u = zeros(n, 1);
for i = 1:n
    state = mod(1664525 * state + 1013904223, 2^32);
    u(i) = state / 2^32;
end
end

function [A, B] = buildPair(p, wp, q, wq, sigma, r, isRel, isPer, period)
%BUILDPAIR  Construct both densities once, outside any timed region.
A = buildExpTens(p, wp, sigma, r, isRel, isPer, period, 'verbose', false);
B = buildExpTens(q, wq, sigma, r, isRel, isPer, period, 'verbose', false);
end

function v = callSim(A, B, method, trunc)
%CALLSIM  The timed unit: the similarity alone, on pre-built densities.
v = cosSimExpTens(A, B, 'method', method, 'truncationSigmas', trunc, ...
                  'verbose', false);
end

function s = predictS(method, r, K, rate, isRel, isPer, orbitS, perScale, relScale)
%PREDICTS  Predicted seconds for ONE call.
if K < r, s = 0; return; end
ordered = factorial(r) * nchoosek(K, r);
switch method
    case {'centres', 'reference'}
        n = ordered * ordered;         % unrestricted: both sides enumerated
    case 'referenceR'
        n = nchoosek(K, r) * ordered;  % same code, X side restricted
    case {'direct', 'bulger'}
        % Both are priced at the restricted count: in the single-multiset
        % path they route through the same core. ('direct' is unavailable
        % in MATLAB -- see METHODS above -- but the case is kept so the
        % two scripts price identically.)
        n = nchoosek(K, r) * ordered;
    otherwise                                    % mobius: K-independent
        key = sprintf('%d%d', isRel, isPer);
        tbl = orbitS(key);
        idx = min(max(r - 1, 1), numel(tbl));
        s = tbl(idx);
        return;
end
scale = 1.0;
if isPer, scale = scale * perScale; end
if isRel, scale = scale * relScale; end
s = n * rate * scale;
end

function rate = calibrateRate(sigma, period)
%CALIBRATERATE  Seconds per tuple pair on this machine, from a safe cell.
state = 20260822;
[p, w, state] = draw(12, period, state);
[q, wq, ~] = draw(12, period, state);
[A, B] = buildPair(p, w, q, wq, sigma, 3, false, false, period);
fn = @() cosSimExpTens(A, B, 'method', 'bulger', ...
                       'truncationSigmas', Inf, 'verbose', false);
[t, ~] = timeIt(fn, 60);
rate = t / (nchoosek(12, 3) * factorial(3) * nchoosek(12, 3));
end

function [t, n] = timeIt(fn, targetMs)
fn();                                       % warm-up
n = 1;
while true
    tic; for i = 1:n, fn(); end; el = toc;
    if el * 1e3 >= targetMs || n >= 2^20, break; end
    n = max(2 * n, ceil(n * targetMs / max(el * 1e3, 1e-6)));
end
ts = zeros(5, 1);
for b = 1:5
    tic; for i = 1:n, fn(); end; ts(b) = toc / n;
end
t = median(ts);
end

function [kk, tt] = pick(rows, mlabel, r, method)
kk = []; tt = [];
for i = 1:numel(rows)
    if strcmp(rows{i}{2}, mlabel) && rows{i}{5} == r && ...
       strcmp(rows{i}{7}, method) && strcmp(rows{i}{8}, 'inf')
        kk(end+1) = rows{i}{6}; %#ok<AGROW>
        tt(end+1) = rows{i}{9}; %#ok<AGROW>
    end
end
end

function e = fitExponent(kk, tt)
ok = kk > 0 & tt > 0;
kk = kk(ok); tt = tt(ok);
if numel(kk) < 3, e = NaN; return; end
[kk, ord] = sort(kk); tt = tt(ord);
kk = kk(end-2:end); tt = tt(end-2:end);
c = polyfit(log(kk), log(tt), 1);
e = c(1);
end

function writeCsv(path, rows)
fid = fopen(path, 'w');
fprintf(fid, 'language,task,mode,is_rel,is_per,r,K,n_queries,method,rel_route,truncation,seconds,reps,value,rel_dev\n');
for i = 1:numel(rows)
    x = rows{i};
    if numel(x) >= 15, task = x{14}; nq = x{15}; else, task = 'ip'; nq = 0; end
    fprintf(fid, '%s,%s,%s,%d,%d,%d,%d,%d,%s,%s,%s,%.9g,%d,%.15g,%.3e\n', ...
            x{1}, task, x{2}, x{3}, x{4}, x{5}, x{6}, nq, x{7}, x{13}, ...
            x{8}, x{9}, x{10}, x{11}, x{12});
end
fclose(fid);
end
