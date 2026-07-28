%% bench_nested_dims_relnonper.m
%
%  Where does the contraction win on the line?
%
%  bench_nested_dims.m answered this for the periodic case: contraction
%  overtakes enumeration at about K = 5 and wins by 12x at K = 8. The
%  relative-non-periodic mode is a different cost regime -- its quadrature
%  spans the value range rather than a period, so the tau count is set by
%  the data -- and the crossover has not been measured there. This is the
%  regime a sequence of tensor harmonicities runs in.
%
%  Three sweeps with the event count held fixed, so the pair grid
%  contributes a constant and only within-event work varies:
%
%    A. inner tuple size r, at fixed multiset size
%    B. multiset size K, at fixed inner tuple size
%    C. bind width L, which sets the outer tuple size
%
%  Each row reports the leaf-tuple count (which drives enumeration) and the
%  tau-window width (which drives the contraction), so the two costs can be
%  read against each other rather than inferred from the ratio alone.
%
%  Configurations are costed before they run and skipped when predicted to
%  be too slow: at K = 6, L = 3 the tuple count rises 125-fold from r = 1 to
%  r = 2, and enumeration with it.
%
%  Run from the matlab directory:  clear all; rehash; bench_nested_dims_relnonper
%
%  A measurement tool, not part of the toolbox. Delete when done.

clear functions %#ok<CLFUNC>

SIG    = 0.30;    % as in bench_nested_relnonper: the line grid steps at
                  % sigma/4 across the whole spread, so a small sigma makes
                  % the tau count enormous
NSUPER = 4;
TOTAL_BUDGET = 240;
PER_CONFIG   = 20;
REPS         = 2;

runClock = tic;

fprintf('\n');
fprintf('Relative non-periodic nested contraction: where does contract win?\n');
fprintf('sigma = %.3g, super-events = %d, values on the line\n', SIG, NSUPER);
fprintf('budget %.0f s total, %.0f s per configuration\n\n', TOTAL_BUDGET, PER_CONFIG);

% --- Calibrate ---------------------------------------------------------
[cx, cy] = local_pair(NSUPER, 2, 4, 1, SIG);
cosSimExpTens(cx, cy, 'method', 'bulger', 'verbose', false);   % warm-up
t0 = tic;
cosSimExpTens(cx, cy, 'method', 'bulger', 'verbose', false);
tCal = toc(t0);
costK = max(tCal, 1e-4) / local_work(4, 1, 2, NSUPER);
fprintf('calibration: %d leaf tuples, %.4f s  ->  %.3e s per unit work\n\n', ...
        local_ntuple(4, 1, 2), tCal, costK);

state = struct('costK', costK, 'clock', runClock, 'total', TOTAL_BUDGET, ...
               'perCfg', PER_CONFIG, 'reps', REPS, 'nsuper', NSUPER, 'sig', SIG);

fprintf('A. inner tuple size r      (K = 6, bind width L = 2)\n');
state = local_sweep('r', (1:5).', [6 2], 'r', state);

fprintf('\nB. multiset size K         (inner r = 2, bind width L = 2)\n');
state = local_sweep('K', (3:12).', [2 2], 'K', state);

fprintf('\nC. bind width L            (K = 4, inner r = 2)\n');
state = local_sweep('L', (2:6).', [4 2], 'L', state);

fprintf('\nelapsed: %.1f s\n\n', toc(runClock));
fprintf('Reading the result. Ratio below 1 is a configuration where the\n');
fprintf('contraction beats enumeration on the line. Watch the two cost\n');
fprintf('drivers: nTuple rises steeply with r, K and L and is what\n');
fprintf('enumeration pays, while the tau window is nearly flat and is what\n');
fprintf('the contraction pays. Where the first overtakes the second is the\n');
fprintf('crossover.\n\n');


% ---------------------------------------------------------------------
function state = local_sweep(label, vals, fixed, which, state)
    fprintf('  %-6s %9s %8s %10s %10s %8s %10s\n', ...
            label, 'nTuple', 'taus', 'contract', 'bulger', 'ratio', 'agree');
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
            fprintf('  %-6g %9d %8s   predicted %.0f s, skipped\n', v, nT, '-', pred);
            continue
        end
        if toc(state.clock) + 3 * pred > state.total
            fprintf('  %-6g %9d %8s   would exceed the total budget, stopping\n', ...
                    v, nT, '-');
            break
        end

        try
            [X, Y] = local_pair(state.nsuper, L, K, r, state.sig);
        catch err
            fprintf('  %-6g %9d   build failed: %s\n', v, nT, err.message);
            continue
        end

        % The tau grid the line quadrature will lay down for these data.
        spread = max(max(X.pAttr{1}(:)), max(Y.pAttr{1}(:))) ...
               - min(min(X.pAttr{1}(:)), min(Y.pAttr{1}(:)));
        hi = spread + (6 + 0.5 * 12) * state.sig;
        nTau = max(64, ceil(2 * hi / (state.sig / 4)));

        tC = local_time(X, Y, 'contract', state.reps);
        tB = local_time(X, Y, 'bulger',   state.reps);
        if isnan(tC) || isnan(tB)
            fprintf('  %-6g %9d %8d   a route errored\n', v, nT, nTau);
            continue
        end

        sC = cosSimExpTens(X, Y, 'method', 'contract', 'verbose', false);
        sB = cosSimExpTens(X, Y, 'method', 'bulger',   'verbose', false);
        ok = abs(sC - sB) < 1e-7;      % 6-sigma truncation error is ~1.5e-8

        fprintf('  %-6g %9d %8d %9.4fs %9.4fs %8.2f %10s\n', ...
                v, nT, nTau, tC, tB, tC / max(tB, eps), local_yn(ok));

        state.costK = 0.5 * state.costK ...
                    + 0.5 * (tB / local_work(K, r, L, state.nsuper));
    end
end


function n = local_ntuple(K, r, L)
    n = (factorial(r) * nchoosek(K, r))^L;
end


function w = local_work(K, r, L, nSuper)
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
function [X, Y] = local_pair(nSuper, L, K, rIn, sigma)
    % Chord passages on the line: relative, non-periodic. Values are not
    % wrapped, so the quadrature runs over the line. Multivalued events keep
    % the nesting from flattening away. The roots are reflected into a two-
    % octave band: the tau grid spans the whole value spread, so an
    % unbounded walk would make the quadrature enormous.
    N  = nSuper + L - 1;
    rs = RandStream('mt19937ar', 'Seed', 11);
    stack = (0:K-1)' * (12 / K);

    step = randi(rs, [-4 4], 1, N);
    rootsA = zeros(1, N); cur = 60;
    for i = 1:N
        cur = cur + step(i);
        if cur > 84; cur = 84 - (cur - 84); end
        if cur < 60; cur = 60 + (60 - cur); end
        rootsA(i) = cur;
    end
    rootsB = rootsA + 1 + randi(rs, [0 2], 1, N);

    A = repmat(rootsA, K, 1) + repmat(stack, 1, N);
    B = repmat(rootsB, K, 1) + repmat(stack, 1, N);

    spA = flatSpecs({A}, 'r', rIn, 'rel', false, 'sym', true);
    spB = flatSpecs({B}, 'r', rIn, 'rel', false, 'sym', true);
    [pa, ~, spa] = bindEvents({A}, [], L, 'specs', spA, 'relOuter', true);
    [pb, ~, spb] = bindEvents({B}, [], L, 'specs', spB, 'relOuter', true);

    Pa = pa{1}; Pb = pb{1};
    % With 'specs', buildExpTens requires sigma, isPer and period together;
    % period is inert because isPer is false.
    X = buildExpTens({Pa}, {ones(size(Pa))}, 'specs', {spa{1}}, ...
                     'sigma', sigma, 'isPer', false, 'period', 12, ...
                     'verbose', false);
    Y = buildExpTens({Pb}, {ones(size(Pb))}, 'specs', {spb{1}}, ...
                     'sigma', sigma, 'isPer', false, 'period', 12, ...
                     'verbose', false);

    if ~(isfield(X, 'nested') && iscell(X.nested) ...
         && any(~cellfun(@isempty, X.nested)))
        error('bench:notNested', ...
              'Density is not nested, so method=''contract'' does not apply.');
    end
end
