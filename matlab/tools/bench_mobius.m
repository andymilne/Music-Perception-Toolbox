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
%   the two differing only in the restriction. (Before v3 the toolbox
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
% extend: false (default) runs the main grid only; true appends the
% Moebius-only large-K pass; 'only' runs that pass alone, for topping up
% a run whose main grid is already in hand.
ip.addParameter('extend', false, ...
    @(x) islogical(x) || (ischar(x) && strcmp(x, 'only')));
ip.addParameter('extKs', [], @isnumeric);
% resume: paths to earlier CSVs. Any (task, mode, r, K, method) cell
% found there is skipped, so a run fills only what is missing.
ip.addParameter('resume', {}, @(x) ischar(x) || iscellstr(x));
% Explicit r and K grids, as in the Python twin. Empty keeps the defaults.
ip.addParameter('rs', [], @isnumeric);
ip.addParameter('ks', [], @isnumeric);
% spectralGate: false (default) bypasses the spectral branch's cost
% gate, because that gate misroutes outside its calibration range and
% these figures are about the decompositions, not the routing. Set true
% to measure the shipped routing instead.
ip.addParameter('spectralGate', false, @islogical);
ip.addParameter('sigma', 10.0, @(x) isnumeric(x) && isscalar(x) && x > 0);
ip.addParameter('task', 'both', @(s) any(strcmp(s, {'ip','eval','both'})));
% Relative-attribute route inside the orbit method. Has no effect on
% these timings: every method here is forced, and an explicit
% method='mobius' pins the sub-route ahead of this lever. Retained only
% to measure the lever itself. 'grid' is its retired name, accepted as
% the toolbox accepts it and normalised by mptDefaults.
ip.addParameter('relRoute', 'auto', ...
    @(s) any(strcmp(s, {'auto','centres','mobius','grid'})));
ip.addParameter('out', '', @ischar);
ip.parse(varargin{:});
opt = ip.Results;

% Resolved once, here, because both the pass construction and the header
% below depend on them.
extOnly = ischar(opt.extend) && strcmp(opt.extend, 'only');
doExtend = extOnly || (islogical(opt.extend) && opt.extend);
if isempty(opt.out)
    % Never overwrite an existing results file. A run given 'resume'
    % reads the earlier output and then writes only what it measured;
    % writing back to the same name would delete the very cells it
    % skipped, which is how an extension run once destroyed the data it
    % had just been told to preserve.
    tag = '';
    if extOnly, tag = '_ext'; end
    stem = sprintf('bench_mobius_matlab_%s_sig%g%s', ...
                   opt.relRoute, opt.sigma, tag);
    opt.out = [stem '.csv'];
    nOut = 2;
    while exist(opt.out, 'file') == 2
        opt.out = sprintf('%s_%d.csv', stem, nOut);
        nOut = nOut + 1;
    end
end

% K grid: a geometric progression at ratio 1.5, six points spanning a
% factor of eight (6, 9, 14, 21, 32, 48). Equal spacing on a log axis is
% what a power-law slope fit wants, since the fit is a straight line in
% log K against log t and evenly spaced points weight it evenly; the
% earlier 6, 12, 20, 34 had ratios 2.0, 1.67, 1.70 and no stated basis.
% The lower end starts at 6 so that r = 4 still has K > r; the upper end
% stops at 48 because the enumerating routes are already far past
% feasibility there and the budget will skip them.
RS_FULL = [2 3 4];
KS_FULL = [6 9 14 21 32 48];

% Extension grid, Moebius only. The enumerating routes are infeasible
% beyond K ~ 48, but the Moebius route is nearly flat there and its
% predicted O(|Omega_r| K^2) has not begun to bite: measured exponents on
% 6..48 come out at 0.1-0.5 against a predicted 2, because the fixed
% per-call cost still dominates. It does bite further out -- the fitted
% exponent reaches 1.96 by K = 1200 at r = 2 -- so the asymptote is
% reached by extending K for that route alone, at a few hundred
% milliseconds per cell.
%
% This continues the main grid's geometric progression at ratio 1.5,
% rounded to two significant figures, rather than starting a new one:
% 6, 9, 14, 21, 32, 48 | 72, 110, 160, 240, 360, 540, 820, 1200. One
% progression across the whole range keeps the points evenly spaced on
% the log axis a slope fit reads, with no discontinuity at the join.
EXT_KS = [72 110 160 240 360 540 820 1200];
% The extension pass runs every route, not the Moebius one alone. Which
% routes can reach a given K is a question for the per-cell budget, not
% for a hard-coded list: in the relative modes at low r the centres
% array is the cheaper route and was previously cut off at K = 48, while
% still below the Moebius evaluator, so the crossing went unobserved.
% Cells beyond reach are skipped by the budget and the block abandoned,
% as everywhere else. Empty means 'use the task's full route list'.
EXT_METHODS = {};
MODES = { false false 'absolute non-periodic'
          false true  'absolute periodic'
          true  false 'relative non-periodic'
          true  true  'relative periodic' };
% Every route timed is one the toolbox exposes and a user can call, so
% the numbers are comparable: all carry the same per-call machinery
% (density structs, dispatch, self-inner-product caching, truncation).
% Local re-implementations were timed in an earlier version and removed
% -- bypassing that machinery, they measured the harness rather than the
% algorithm, and at small K the difference was the whole result.
IP_METHODS = {'centres', 'bulger', 'mobius'};
EVAL_METHODS = {'centres', 'mobius'};
N_QUERIES = 64;   % query points per evaluation cell, held fixed
METHODS = IP_METHODS;

% Kernel width, in cents, with the octave as period. The default follows
% the manuscript's worked examples (sigma_pitch = 10 cents; equivalently
% sigma = 0.1 with P = 12 in semitone units), giving sigma/P = 0.0083.
% This is not a neutral choice: sigma/P governs every threshold in the
% toolbox, so it decides which regime is being timed. Below sigma/P
% ~0.03 the relative-periodic routes agree to floating point and the
% periodic kernel sums a single image; above it they diverge and the
% image count grows. Override with the 'sigma' parameter.
SIGMA  = opt.sigma;
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
CELL_BUDGET_S = 20.0;
% Per-pair cost multipliers, relative to absolute non-periodic Bulger,
% fitted from measured runs rather than guessed. The earlier guesses
% (40x for periodic wrapping, 4x for the relative quadrature) were an
% order of magnitude too pessimistic and rejected cells that complete in
% well under a second: the slowest cell that survived a 2 s budget
% actually took 347 ms.
MODE_SCALE = containers.Map({'00','01','10','11'}, {1.0, 3.2, 1.5, 5.5});
% The unrestricted route carries a further factor over Bulger at the same
% pair count, largest in the relative periodic mode.
CENTRES_SCALE = containers.Map({'00','01','10','11'}, {1.8, 1.3, 1.5, 2.9});
ORBIT_S = containers.Map( ...
    {'00', '01', '10', '11'}, ...
    {[6e-4 1.3e-3 3.8e-3 1.2e-2 4e-2], ...
     [6e-4 1.2e-3 3.6e-3 1.2e-2 4e-2], ...
     [1.3e-3 1.6e-2 8.5e-2 3.4e-1 1.4e0], ...
     [4e-4 9e-4 2.5e-2 9.1e-2 3.6e-1]});   % index by r-1, r = 2..6

if opt.quick
    RS = [2 3]; KS = [6 9 14]; modes = MODES(1, :);
else
    RS = RS_FULL; KS = KS_FULL; modes = MODES;
end
if ~isempty(opt.rs), RS = opt.rs(:)'; end
if ~isempty(opt.ks), KS = opt.ks(:)'; end

% Silence one-time informational hints and switch off the post-hoc
% guards: with guards on, a route that diverts pays for both routes, so
% the measured cost would not be the cost of the route being forced.
% Both are restored on exit, including on error.
prevDefaults = mptDefaults('showHints', false, 'postHocGuards', false, ...
                          'relAttrRoute', opt.relRoute);
if strcmp(opt.relRoute, 'auto')
    relRouteTag = 'forced-by-method';
else
    relRouteTag = opt.relRoute;
end
prevSpectral = internal.spectralIpForce(~opt.spectralGate);
cleanupSpectral = onCleanup(@() internal.spectralIpForce(prevSpectral));

% 'centres' on the inner product requires a toolbox that exposes it. An
% older build rejects it, which would silently drop the baseline column.
if any(strcmp(opt.task, {'ip', 'both'})) && ~methodAccepted('centres', SIGMA, PERIOD)
    error('bench_mobius:centresUnavailable', ...
          ['this toolbox build does not accept method=''centres'' on ' ...
           'cosSimExpTens, so the O(K^(2r)) baseline cannot be timed. ' ...
           'Apply the method-vocabulary update first, or run with ' ...
           '''task'', ''eval''.']);
end
cleanup = onCleanup(@() mptDefaults(prevDefaults));

resumePaths = opt.resume;
if ischar(resumePaths), resumePaths = {resumePaths}; end
doneCells = loadDone(resumePaths);
if ~isempty(doneCells)
    fprintf(['resuming: %d cells already measured will be skipped. Rows ' ...
             'with nothing left to measure are not printed, and a ' ...
             'starred ''fastest'' ranks only the routes this run ' ...
             'timed.\n'], numel(doneCells));
end

rows = {};
failures = {};
nSkipped = 0;
nAbandoned = 0;
state = SEED;                                   % shared LCG state
rate = calibrateRate(SIGMA, PERIOD);

fprintf('MATLAB %s\n%s\n', version, computer);
fprintf('calibrated at %.1f ns per tuple pair; cell budget %g s; rel route %s\n', ...
        rate * 1e9, CELL_BUDGET_S, opt.relRoute);
fprintf('sigma %g cents, period %g (sigma/P = %.4f)\n', ...
        SIGMA, PERIOD, SIGMA / PERIOD);
if opt.spectralGate
    fprintf('spectral branch cost gate: in force (shipped routing)\n');
else
    fprintf('spectral branch cost gate: bypassed (measuring the decomposition)\n');
end
fprintf(['relative-attribute sub-route: pinned by the forced method ' ...
         '(relAttrRoute has no effect here)\n\n']);
if ~extOnly
    fprintf('%-22s%3s%5s  ', 'mode', 'r', 'K');
    fprintf('%12s', IP_METHODS{:}); fprintf('   fastest\n');
    fprintf('%s\n', repmat('-', 1, 22 + 8 + 12 * numel(IP_METHODS) + 9));
end

% Passes: the main grid over all routes, then optionally a Moebius-only
% pass at large K where that route's K^2 asymptote appears.
if extOnly
    passKs = {}; passMethods = {}; passEvalMethods = {};
else
    passKs = {KS};
    passMethods = {IP_METHODS};
    passEvalMethods = {EVAL_METHODS};
end
if doExtend
    if isempty(opt.extKs), extKs = EXT_KS; else, extKs = opt.extKs; end
    passKs{end+1} = extKs;
    if isempty(EXT_METHODS)
        passMethods{end+1} = IP_METHODS;
        passEvalMethods{end+1} = EVAL_METHODS;
    else
        passMethods{end+1} = EXT_METHODS;
        passEvalMethods{end+1} = EXT_METHODS;
    end
end

for pass = 1:numel(passKs)
KS = passKs{pass};
sweepMethods = passMethods{pass};
sweepEvalMethods = passEvalMethods{pass};
if pass > 1 || extOnly
    fprintf('\nextension pass, K = %s\n\n', mat2str(KS));
    fprintf('%-22s%3s%5s  ', 'mode', 'r', 'K');
    fprintf('%12s', sweepMethods{:}); fprintf('   fastest\n');
    fprintf('%s\n', repmat('-', 1, 22 + 8 + 12 * numel(sweepMethods) + 9));
end

if any(strcmp(opt.task, {'ip', 'both'}))
for mi = 1:size(modes, 1)
    isRel = modes{mi, 1}; isPer = modes{mi, 2}; mlabel = modes{mi, 3};
    for r = RS
        % Cost rises monotonically with K for each route separately, so a
        % route that has blown the budget will blow it at every larger K
        % and is dropped from the rest of the block. The bookkeeping is
        % per route, not per block: a single flag for the whole block let
        % one expensive route abandon the others, so the cheaper route in
        % a panel stopped early while still below the route it was being
        % compared against.
        doneMethods = {};
        for K = KS
            if numel(doneMethods) >= numel(sweepMethods)
                nAbandoned = nAbandoned + 1;
                continue;
            end
            if K < r + 1, continue; end

            [p, wp, state] = draw(K, PERIOD, state);
            [q, wq, state] = draw(K, PERIOD, state);
            [densA, densB] = buildPair(p, wp, q, wq, SIGMA, r, isRel, isPer, PERIOD);

            % --- correctness first, untruncated ------------------------
            vals = containers.Map();
            for k = 1:numel(sweepMethods)
                m = sweepMethods{k};
                if any(strcmp(doneMethods, m)), continue; end
                if isKeyCell(doneCells, 'ip', mlabel, r, K, m), continue; end
                pS = predictS(m, r, K, rate, isRel, isPer, ORBIT_S, MODE_SCALE, CENTRES_SCALE);
                if pS > CELL_BUDGET_S
                    nSkipped = nSkipped + 1;
                    doneMethods{end+1} = m; %#ok<AGROW>
                    continue;
                end
                try
                    vals(m) = callSim(densA, densB, m, Inf);
                catch err
                    failures{end+1} = sprintf('%s r=%d K=%d %s: %s', ...
                        mlabel, r, K, m, err.message); %#ok<AGROW>
                end
            end
            ks = vals.keys();
            devs = containers.Map('KeyType', 'char', 'ValueType', 'double');
            if ~isempty(ks)
                % centres is the accuracy reference: it enumerates the
                % definition directly and has no alternating sum. Where it
                % was skipped as over budget there is none, and the
                % deviation is NaN rather than a spurious zero.
                if vals.isKey('centres'), ref = vals('centres'); else, ref = NaN; end
                for k = 1:numel(ks)
                    v = vals(ks{k});
                    if strcmp(ks{k}, 'centres') || ~isfinite(v) || ~isfinite(ref)
                        devs(ks{k}) = NaN;
                    else
                        devs(ks{k}) = abs(v - ref) / max(1, abs(ref));
                    end
                    if ~isfinite(v) || devs(ks{k}) > TOL
                        failures{end+1} = sprintf('%s r=%d K=%d %s deviates from centres by %.2e', ...
                            mlabel, r, K, ks{k}, devs(ks{k})); %#ok<AGROW>
                    end
                end
            end

            % --- timing -------------------------------------------------
            times = containers.Map('KeyType', 'char', 'ValueType', 'double');
            for k = 1:numel(ks)
                m = ks{k};
                fn = @() callSim(densA, densB, m, Inf);
                try
                    [t, n, guarded] = timeIt(fn, TARGET_MS, CELL_BUDGET_S);
                    if guarded, doneMethods{end+1} = m; end %#ok<AGROW>
                    times(m) = t;
                    rows{end+1} = {'matlab', mlabel, isRel, isPer, r, K, m, ...
                                   'inf', t, n, vals(m), devs(m), opt.relRoute, guarded}; %#ok<AGROW>
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
                        [t, n, guarded] = timeIt(fn, TARGET_MS, CELL_BUDGET_S);
                    if guarded, doneMethods{end+1} = m; end %#ok<AGROW>
                        rows{end+1} = {'matlab', mlabel, isRel, isPer, r, K, m, ...
                                       '6', t, n, fn(), NaN, relRouteTag}; %#ok<AGROW>
                    catch
                    end
                end
            end

            % --- report the cell ----------------------------------------
            % Nothing measured here -- every route resumed, over budget,
            % or abandoned -- so print no line: a table of dashes buries
            % the rows that do carry a measurement.
            if times.Count == 0, continue; end
            fprintf('%-22s%3d%5d  ', mlabel, r, K);
            best = ''; bestT = Inf;
            for k = 1:numel(sweepMethods)
                m = sweepMethods{k};
                if times.isKey(m)
                    fprintf('%11.3fm', times(m) * 1e3);
                    if times(m) < bestT
                        bestT = times(m); best = m;
                    end
                else
                    fprintf('%12s', '-');
                end
            end
            % 'fastest' ranks only what this run measured. With resume
            % the absent routes may well be quicker, so a partial ranking
            % is starred rather than left to imply otherwise.
            if times.Count < numel(sweepMethods), best = [best '*']; end
            fprintf('   %s\n', best);
        end
    end
end

end   % task 'ip'

% --- point evaluation ---------------------------------------------------
if any(strcmp(opt.task, {'eval', 'both'}))
fprintf('\npoint evaluation, %d query points per cell\n\n', N_QUERIES);
fprintf('%-22s%3s%5s  ', 'mode', 'r', 'K');
fprintf('%12s', sweepEvalMethods{:}); fprintf('   fastest\n');
fprintf('%s\n', repmat('-', 1, 22 + 8 + 12 * numel(sweepEvalMethods) + 9));
for mi = 1:size(modes, 1)
    isRel = modes{mi, 1}; isPer = modes{mi, 2}; mlabel = modes{mi, 3};
    for r = RS
        doneMethods = {};           % per route; see the inner-product sweep
        for K = KS
            if numel(doneMethods) >= numel(sweepEvalMethods)
                nAbandoned = nAbandoned + 1;
                continue;
            end
            if K < r + 1, continue; end
            [p, wp, state] = draw(K, PERIOD, state);
            dens = buildExpTens(p, wp, SIGMA, r, isRel, isPer, PERIOD, ...
                                'verbose', false);
            dim = max(r - double(isRel), 1);
            [u, state] = lcg(dim * N_QUERIES, state);
            pts = reshape(u, dim, N_QUERIES) * PERIOD;

            vals = containers.Map(); tms = containers.Map();
            for k = 1:numel(sweepEvalMethods)
                mm = sweepEvalMethods{k};
                if any(strcmp(doneMethods, mm)), continue; end
                if isKeyCell(doneCells, 'eval', mlabel, r, K, mm), continue; end
                try
                    vals(mm) = sum(sum(evalExpTens(dens, pts, 'method', mm, ...
                        'truncationSigmas', Inf, 'verbose', false)));
                    fn = @() evalExpTens(dens, pts, 'method', mm, ...
                        'truncationSigmas', Inf, 'verbose', false);
                    [t, n, guarded] = timeIt(fn, TARGET_MS, CELL_BUDGET_S);
                    if guarded, doneMethods{end+1} = mm; end %#ok<AGROW>
                    tms(mm) = t;
                    rows{end+1} = {'matlab', mlabel, isRel, isPer, r, K, mm, ...
                                   'inf', t, n, vals(mm), NaN, opt.relRoute, guarded, ...
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
            if tms.Count == 0, continue; end      % see the note above
            fprintf('%-22s%3d%5d  ', mlabel, r, K);
            bestE = ''; bestT = Inf;
            for k = 1:numel(sweepEvalMethods)
                mm = sweepEvalMethods{k};
                if tms.isKey(mm)
                    fprintf('%11.3fm', tms(mm) * 1e3);
                    if tms(mm) < bestT, bestT = tms(mm); bestE = mm; end
                else
                    fprintf('%12s', '-');
                end
            end
            if tms.Count < numel(sweepEvalMethods), bestE = [bestE '*']; end
            fprintf('   %s\n', bestE);
        end
    end
end
end   % task 'eval'

end   % passes

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
        % Distinct K only: repeated K differ by measurement noise, which
        % would otherwise be reported as a fall.
        if numel(unique(kk)) < 3, continue; end
        [kk, ord] = sort(kk); tt = tt(ord);
        drops = kk(find(tt(2:end) < 0.85 * tt(1:end-1)) + 1);
        if ~isempty(drops)
            fprintf('   %-22s r=%d   falls at K = %s\n', mlabel, r, mat2str(drops));
        end
    end
end

nGuarded = 0;
for i = 1:numel(rows)
    if numel(rows{i}) >= 14 && islogical(rows{i}{14}) && rows{i}{14}
        nGuarded = nGuarded + 1;
    end
end
if nGuarded > 0
    fprintf(['\nguarded cells (%d): one call already exceeded the %g s ' ...
             'budget, so repetitions were skipped; these timings are ' ...
             'noisier than the rest.\n'], nGuarded, CELL_BUDGET_S);
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
if nAbandoned > 0
    fprintf(['cells abandoned: %d (a smaller K in the same (mode, r) ' ...
             'block already exceeded the %g s budget, and cost rises ' ...
             'with K)\n'], nAbandoned, CELL_BUDGET_S);
end

if isempty(failures)
    fprintf('\nall available methods agreed to within %g on every cell\n', TOL);
else
    fprintf('\nFAILURES (%d):\n', numel(failures));
    for i = 1:min(25, numel(failures)), fprintf('   %s\n', failures{i}); end
end

writeCsv(opt.out, rows, SIGMA, PERIOD);
fprintf('\nwrote %s  (%d rows)\n', opt.out, numel(rows));
end

% =======================================================================

function keys = loadDone(paths)
%LOADDONE  Cell keys already measured, from earlier CSV output.
keys = {};
for i = 1:numel(paths)
    p = paths{i};
    if exist(p, 'file') ~= 2
        fprintf('warning: cannot read %s\n', p);
        continue;
    end
    fid = fopen(p, 'r');
    hdr = strsplit(strtrim(fgetl(fid)), ',');
    iTask = find(strcmp(hdr, 'task'));    iMode = find(strcmp(hdr, 'mode'));
    iR    = find(strcmp(hdr, 'r'));       iK    = find(strcmp(hdr, 'K'));
    iMeth = find(strcmp(hdr, 'method'));
    if isempty(iTask) || isempty(iMode) || isempty(iR) || isempty(iK) ...
            || isempty(iMeth)
        fclose(fid);
        fprintf('warning: %s lacks the columns needed to resume\n', p);
        continue;
    end
    while true
        ln = fgetl(fid);
        if ~ischar(ln), break; end
        f = strsplit(strtrim(ln), ',');
        if numel(f) < max([iTask iMode iR iK iMeth]), continue; end
        keys{end+1} = sprintf('%s|%s|%s|%s|%s', f{iTask}, f{iMode}, ...
                              f{iR}, f{iK}, f{iMeth}); %#ok<AGROW>
    end
    fclose(fid);
end
keys = unique(keys);
end

function tf = isKeyCell(keys, task, mlabel, r, K, method)
%ISKEYCELL  True when this cell is already present in the resume set.
if isempty(keys), tf = false; return; end
tf = any(strcmp(keys, sprintf('%s|%s|%d|%d|%s', task, mlabel, r, K, method)));
end

function tf = methodAccepted(method, sigma, period)
%METHODACCEPTED  True when the installed cosSimExpTens accepts a method.
tf = true;
try
    A = buildExpTens((0:5)' * 100, [], sigma, 2, false, false, period, ...
                     'verbose', false);
    B = buildExpTens((2:7)' * 100, [], sigma, 2, false, false, period, ...
                     'verbose', false);
    cosSimExpTens(A, B, 'method', method, 'verbose', false);
catch err
    if strcmp(err.identifier, 'cosSimExpTens:badMethod')
        tf = false;
    else
        rethrow(err);
    end
end
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
%
%   The memoised self inner products are NOT carried between calls: they
%   travel on the optional second and third outputs (densXOut, densYOut),
%   which are deliberately not captured, so each timed call pays for the
%   full triple <X,Y>, <X,X>, <Y,Y> that a similarity requires. The
%   Python twin clears its densities' caches explicitly to match, since
%   there the density is an object whose cache would otherwise persist
%   and the measured time would fall to the cross term alone -- about a
%   third of the work.
v = cosSimExpTens(A, B, 'method', method, 'truncationSigmas', trunc, ...
                  'verbose', false);
end

function s = predictS(method, r, K, rate, isRel, isPer, orbitS, modeScale, centresScale)
%PREDICTS  Predicted seconds for ONE call.
if K < r, s = 0; return; end
ordered = factorial(r) * nchoosek(K, r);
switch method
    case 'centres'
        n = ordered * ordered;         % unrestricted: both sides enumerated
    case 'bulger'
        % One side restricted to combinations (Supplement Eq. S8).
        n = nchoosek(K, r) * ordered;
    otherwise                                    % mobius: K-independent
        key = sprintf('%d%d', isRel, isPer);
        tbl = orbitS(key);
        idx = min(max(r - 1, 1), numel(tbl));
        s = tbl(idx);
        return;
end
key = sprintf('%d%d', isRel, isPer);
scale = modeScale(key);
if strcmp(method, 'centres')
    scale = scale * centresScale(key);
end
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

function [t, n, guarded] = timeIt(fn, targetMs, budgetS)
%TIMEIT  Median seconds per call, batch auto-sized to ~targetMs.
%
%   The warm-up is timed, and if it alone exceeds budgetS the cell is
%   reported from that single call with the repetitions skipped. A
%   predicted-cost budget can only skip what its model prices correctly;
%   this guard is model-free, so a cell the predictor underestimates
%   cannot run away. Such cells carry reps = 1 in the CSV, marking them
%   as single-shot and therefore noisier than the rest.
if nargin < 3 || isempty(budgetS), budgetS = Inf; end
tWarm = tic;
fn();                                       % warm-up, timed
warm = toc(tWarm);
guarded = false;
if warm > budgetS
    t = warm; n = 1; guarded = true; return;
end
n = 1;
while true
    tic; for i = 1:n, fn(); end; el = toc;
    if el * 1e3 >= targetMs || n >= 2^20, break; end
    n = max(2 * n, ceil(n * targetMs / max(el * 1e3, 1e-6)));
end
guarded = false;
nB = 3; if n == 1, nB = 2; end   % a single slow call needs no repeats
ts = zeros(nB, 1);
for b = 1:nB
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
%FITEXPONENT  Slope of log t against log K over the three largest K.
%   Returns NaN unless at least three DISTINCT K are present: a repeated
%   K makes the fit singular, and polyfit then returns a meaningless
%   value with a conditioning warning rather than failing.
ok = kk > 0 & tt > 0;
kk = kk(ok); tt = tt(ok);
if numel(unique(kk)) < 3, e = NaN; return; end
[kk, ord] = sort(kk); tt = tt(ord);
kk = kk(end-2:end); tt = tt(end-2:end);
c = polyfit(log(kk), log(tt), 1);
e = c(1);
end

function writeCsv(path, rows, sigmaOut, periodOut)
fid = fopen(path, 'w');
fprintf(fid, ['language,task,sigma,period,mode,is_rel,is_per,r,K,' ...
              'n_queries,method,rel_route,truncation,seconds,reps,value,' ...
              'rel_dev,guarded\n']);
for i = 1:numel(rows)
    x = rows{i};
    % Row layout: 1..13 common, 14 guarded, then 15..16 (task, n_queries)
    % on evaluation rows only. Inner-product rows carry 14 elements.
    guarded = false;
    if numel(x) >= 14, guarded = x{14}; end
    if numel(x) >= 16, task = x{15}; nq = x{16}; else, task = 'ip'; nq = 0; end
    fprintf(fid, '%s,%s,%g,%g,%s,%d,%d,%d,%d,%d,%s,%s,%s,%.9g,%d,%.15g,%.3e,%d\n', ...
            x{1}, task, sigmaOut, periodOut, x{2}, x{3}, x{4}, x{5}, x{6}, ...
            nq, x{7}, x{13}, x{8}, x{9}, x{10}, x{11}, x{12}, guarded);
end
fclose(fid);
end
