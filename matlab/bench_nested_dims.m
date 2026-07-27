%% bench_nested_dims.m
%
%  Where does the nested contraction path win?
%
%  The companion script bench_nested_contract.m scaled the event axis and
%  found 'contract' losing to 'bulger' by a widening margin. That axis is
%  not where the contraction helps: its advantage is within the event,
%  where the Mobius decomposition replaces enumeration over tuples with a
%  sum over set partitions. This script scales the within-event dimensions
%  with the event count held fixed.
%
%  Cost control. The leaf-tuple count per super-event is known in advance:
%  a symmetric inner level of tuple size r over K values gives r!*C(K,r)
%  tuples per bound event, and L bound events multiply, so
%
%      nTuple = (r! * C(K,r))^L
%
%  and pairwise cost grows roughly as nTuple^2 per event pair. That climbs
%  violently -- at K = 6, L = 3, going from r = 1 to r = 2 multiplies the
%  tuple count by 125 and the work by about 15000. Every configuration is
%  therefore costed and skipped before it runs if predicted to be too slow,
%  rather than interrupted afterwards. The constant is calibrated at run
%  time from a cheap configuration, so it adapts to the machine.
%
%  The whole script is bounded by TOTAL_BUDGET (default 4 minutes).
%
%  Run from the matlab directory:  clear all; rehash; bench_nested_dims
%
%  A measurement tool, not part of the toolbox. Delete when done.

clear functions %#ok<CLFUNC>

SIG    = 0.15;
PERIOD = 12.0;
NSUPER = 4;       % super-events; small, so the pair loop does not mask
                  % the within-event effect this script looks for
TOTAL_BUDGET = 240;   % seconds for the whole run
PER_CONFIG   = 20;    % seconds a single configuration may be predicted to take
REPS         = 2;

runClock = tic;

fprintf('\n');
fprintf('Nested contraction: where does the contract path win?\n');
fprintf('sigma = %.3g, period = %.4g, super-events = %d\n', SIG, PERIOD, NSUPER);
fprintf('budget %.0f s total, %.0f s per configuration\n', TOTAL_BUDGET, PER_CONFIG);
fprintf('\n');

% --- Calibrate the cost model ----------------------------------------
[cx, cy] = local_pair(NSUPER, 2, 4, 1, SIG, PERIOD);
cosSimExpTens(cx, cy, 'method', 'bulger', 'verbose', false);   % warm-up
t0 = tic;
cosSimExpTens(cx, cy, 'method', 'bulger', 'verbose', false);
tCal = toc(t0);
costK = max(tCal, 1e-4) / local_work(4, 1, 2, NSUPER);
fprintf('calibration: %d leaf tuples, %.4f s  ->  %.3e s per unit work\n\n', ...
        local_ntuple(4, 1, 2), tCal, costK);

state = struct('costK', costK, 'clock', runClock, 'total', TOTAL_BUDGET, ...
               'perCfg', PER_CONFIG, 'reps', REPS, 'nsuper', NSUPER);

% --- Sweep A: inner tuple size ---------------------------------------
fprintf('A. inner tuple size r      (K = 6, bind width L = 2)\n');
state = local_sweep('r', (1:5).', [6 2], 'r', state, SIG, PERIOD);

% --- Sweep B: multiset size ------------------------------------------
fprintf('\nB. multiset size K         (inner r = 2, bind width L = 2)\n');
state = local_sweep('K', (3:12).', [2 2], 'K', state, SIG, PERIOD);

% --- Sweep C: bind width, which sets the outer tuple size ------------
fprintf('\nC. bind width L            (K = 4, inner r = 2)\n');
state = local_sweep('L', (2:6).', [4 2], 'L', state, SIG, PERIOD);

fprintf('\nelapsed: %.1f s\n', toc(runClock));
fprintf('\n');
fprintf('Reading the result. Ratio below 1 is a configuration where the\n');
fprintf('contraction already beats enumeration despite the per-pair loop.\n');
fprintf('The trend across a sweep matters more than any single value: a\n');
fprintf('ratio falling steadily as the tuple count grows means the\n');
fprintf('contraction is doing its job and the loop is what holds it back.\n');
fprintf('\n');


% ---------------------------------------------------------------------
function state = local_sweep(label, vals, fixed, which, state, sigma, period)
    fprintf('  %-6s %9s %10s %10s %8s %10s\n', ...
            label, 'nTuple', 'contract', 'bulger', 'ratio', 'agree');
    for ii = 1:numel(vals)
        v = vals(ii);
        switch which
            case 'r'
                K = fixed(1); L = fixed(2); r = v;
            case 'K'
                r = fixed(1); L = fixed(2); K = v;
            case 'L'
                K = fixed(1); r = fixed(2); L = v;
        end
        if r > K
            fprintf('  %-6g %9s   r > K, skipped\n', v, '-');
            continue
        end

        nT   = local_ntuple(K, r, L);
        pred = state.costK * local_work(K, r, L, state.nsuper);

        if pred > state.perCfg
            fprintf('  %-6g %9d   predicted %.0f s, skipped\n', v, nT, pred);
            continue
        end
        if toc(state.clock) + 3 * pred > state.total
            fprintf('  %-6g %9d   would exceed the total budget, stopping sweep\n', ...
                    v, nT);
            break
        end

        try
            [X, Y] = local_pair(state.nsuper, L, K, r, sigma, period);
        catch err
            fprintf('  %-6g %9d   build failed: %s\n', v, nT, err.message);
            continue
        end

        tC = local_time(X, Y, 'contract', state.reps);
        tB = local_time(X, Y, 'bulger',   state.reps);
        if isnan(tC) || isnan(tB)
            fprintf('  %-6g %9d   a route errored\n', v, nT);
            continue
        end

        sC = cosSimExpTens(X, Y, 'method', 'contract', 'verbose', false);
        sB = cosSimExpTens(X, Y, 'method', 'bulger',   'verbose', false);
        ok = abs(sC - sB) < 1e-9;

        fprintf('  %-6g %9d %9.4fs %9.4fs %8.2f %10s\n', ...
                v, nT, tC, tB, tC / max(tB, eps), local_yn(ok));

        % Refine the constant from what actually happened, so later
        % predictions track this machine rather than the calibration point.
        state.costK = 0.5 * state.costK ...
                    + 0.5 * (tB / local_work(K, r, L, state.nsuper));
    end
end


function n = local_ntuple(K, r, L)
    % Leaf tuples per super-event: symmetric inner level, L bound events.
    n = (factorial(r) * nchoosek(K, r))^L;
end


function w = local_work(K, r, L, nSuper)
    % Pairwise cost is about nTuple^2 per event pair, over the XY grid plus
    % the two symmetric self-terms.
    nT = double(local_ntuple(K, r, L));
    nPairs = nSuper^2 + nSuper * (nSuper + 1);
    w = nT^2 * nPairs;
end


function t = local_time(X, Y, method, reps)
    t = Inf;
    for i = 1:reps
        try
            t0 = tic;
            cosSimExpTens(X, Y, 'method', method, 'verbose', false);
            t = min(t, toc(t0));
        catch
            t = NaN;
            return
        end
    end
end


function s = local_yn(ok)
    if ok
        s = 'yes';
    else
        s = '*** NO ***';
    end
end


% ---------------------------------------------------------------------
function [X, Y] = local_pair(nSuper, L, K, rIn, sigma, period)
    % Two nested densities with nSuper super-events, each binding L events
    % of K values, read at inner tuple size rIn. Events must be multivalued:
    % singleton inner groups make the nesting degenerate and it is flattened
    % away. Inner absolute, outer relative.
    N  = nSuper + L - 1;
    rs = RandStream('mt19937ar', 'Seed', 11);

    rootsA = mod(cumsum(randi(rs, [-4 4], 1, N)), period);
    rootsB = mod(rootsA + 1 + randi(rs, [0 2], 1, N), period);
    stack  = (0:K-1)' * (period / K);

    A = mod(repmat(rootsA, K, 1) + repmat(stack, 1, N), period);
    B = mod(repmat(rootsB, K, 1) + repmat(stack, 1, N), period);

    spA = flatSpecs({A}, 'r', rIn, 'rel', false, 'sym', true);
    spB = flatSpecs({B}, 'r', rIn, 'rel', false, 'sym', true);
    [pa, ~, spa] = bindEvents({A}, [], L, 'specs', spA, 'relOuter', true);
    [pb, ~, spb] = bindEvents({B}, [], L, 'specs', spB, 'relOuter', true);

    Pa = pa{1}; Pb = pb{1};
    X = buildExpTens({Pa}, {ones(size(Pa))}, 'specs', {spa{1}}, ...
                     'sigma', sigma, 'isPer', true, 'period', period, ...
                     'verbose', false);
    Y = buildExpTens({Pb}, {ones(size(Pb))}, 'specs', {spb{1}}, ...
                     'sigma', sigma, 'isPer', true, 'period', period, ...
                     'verbose', false);

    if ~(isfield(X, 'nested') && iscell(X.nested) ...
         && any(~cellfun(@isempty, X.nested)))
        error('bench:notNested', ...
              'Density is not nested, so method=''contract'' does not apply.');
    end
end
