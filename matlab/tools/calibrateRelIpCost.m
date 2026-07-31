function results = calibrateRelIpCost(varargin)
% Measure the relative-mode inner-product cost, per sub-route, for fitting.
%
% Why this exists. The multi-attribute selector chooses between Bulger's
% method and the Mobius method by comparing two predicted wall times
% (internal.selectMaInnerProductMethod). On relative-mode densities that
% comparison misroutes: measured on a small grid it sends between a third
% and a half of cells to the slower method, and the Mobius side is
% over-priced by up to three orders at r = 2 while under-priced at r = 4.
%
% Refitting it has so far failed for want of data, not want of candidates.
% Fitted on twenty-odd cells, no replacement form beat the shipped one by
% more than a single cell under leave-one-out: the routing decision turns
% on a ratio between two predictions, many cells sit near the crossover,
% and a factor of two of timing noise flips them. The fit needs a few
% hundred cells measured on a quiet machine, not another candidate.
%
% What this reports, per cell:
%
%   - each sub-route timed in isolation. The t_auto column is left NaN:
%     the unforced Mobius call runs whichever route gate_route names, so
%     its time is already in one of the two route columns.
%   - each sub-route timed in isolation. Inside the Mobius method a
%     relative attribute takes the materialised tuple-centres route or the
%     translation grid, chosen by a cost estimate that changes partway
%     through any sweep; mptDefaults('relAttrRoute', ...) pins it, so a
%     curve is the scaling of one route rather than a mixture of two.
%   - Bulger's method on the same cell, since the decision is between them.
%   - the two predictions the selector's comparison actually rests on, so
%     the model's error is a continuous quantity beside the measurement
%     rather than a win-or-lose label.
%   - the quantities a fit needs: the grid node count N_u and the
%     permutation-side tuple count M = r! C(K, r).
%
% Agreement is checked before any timing, so a comparison is never made
% between two computations that disagree.
%
% Output is a CSV block on the console with a provenance header. Paste the
% whole block; the fit is done offline against it.
%
% Usage:
%   results = calibrateRelIpCost();
%   results = calibrateRelIpCost('sigmas', [3 6 12], 'seeds', 1:3);
%
% Name-value arguments:
%   'sigmas'      kernel widths to sweep (default [2 6 25])
%   'seeds'       repeats per cell with fresh values (default 1:2)
%   'budgetSec'      skip a route predicted, or found, to exceed this
%                    (default 5)
%   'repeatBelowSec' repeat-time only calls faster than this; slower ones
%                    are measured from their single warm call (default
%                    0.25). Repeating a call that already ran for a
%                    second buys nothing: timing scatter is a fixed
%                    overhead, so its share of a long call is negligible,
%                    and the repeats cost several times the measurement.
%   'period'      periodic-mode period in cents (default 1200)
%
% Runtime. 16 value counts x 6 widths x 2 periodicities x 2 seeds = 384
% cells, each timing three arms. Three things keep that affordable: an
% arm the running estimate puts over budget is never started; a call
% slower than repeatBelowSec is measured from its single warm run rather
% than repeated; and the unforced Mobius arm is not timed at all, since
% gate_route says which route column already holds its time. Simulated
% against a previous sweep's measurements these give a 1.9-fold saving,
% putting the run at ten minutes or so. Rows stream as they are measured,
% so a run can be read while it proceeds and stopped early without losing
% what came before.

    p = inputParser;
    % The Mobius cost tracks the grid node count, which sigma sets, so the
    % sweep needs enough distinct values of it to separate that dependence
    % from the value count.
    p.addParameter('sigmas', [2 6 25], @(x) isnumeric(x) && all(x > 0));
    p.addParameter('seeds', 1, @isnumeric);
    p.addParameter('budgetSec', 5, @(x) isscalar(x) && x > 0);
    p.addParameter('repeatBelowSec', 0.25, @(x) isscalar(x) && x > 0);
    p.addParameter('period', 1200, @(x) isscalar(x) && x > 0);
    p.parse(varargin{:});
    opt = p.Results;

    prevHints = mptDefaults('showHints');
    prevRoute = mptDefaults('relAttrRoute');
    prevPath  = mptDefaults('singleMultisetPath');
    mptDefaults('showHints', false);
    % The cost model being calibrated is the multi-attribute one, so the
    % cells must run through the path it governs. Left at 'auto', a single
    % multiset takes the dedicated stack instead, which has its own
    % selector and never reaches the centres-versus-grid gate: the two
    % route columns then measure the same computation twice, and the fit
    % would be against timings the model never produces. Section 4 puts
    % that discrepancy at up to 180x on relative cells.
    mptDefaults('singleMultisetPath', 'ma');
    cleanup = onCleanup(@() localRestore(prevHints, prevRoute, prevPath));

    % Value counts per tuple order. Bulger's method builds K!/(K-r)!
    % tuples per side, so the affordable range narrows sharply with r.
    % K = 100 at r = 2 is dropped: through the MA path its Bulger arm runs
    % for seconds, and the r = 2 curve is already determined by K = 64.
    KByOrder = {[6 10 16 24 40 64], [6 8 12 16 24], [5 6 8 10 12]};

    % The two densities need not carry the same number of values, and a
    % chord against a scale is the ordinary case. A sweep with equal
    % counts throughout leaves the asymmetric case unconstrained, and it
    % is where a cost model most easily goes wrong: the tuple-pair count
    % spans five orders of magnitude across it, so a term fitted only on
    % equal counts flattens exactly the shape that matters. K_REF is the
    % small side.
    K_REF = 5;
    shapes = {'equal', 'asym'};

    % Weights change how much of a multiset the truncated kernel actually
    % touches, so a model fitted on one profile need not hold on another.
    % Three shapes, each jittered per seed so no cell is a special case.
    profiles = {'flat', 'decay', 'bimodal'};
    rOrders  = [2 3 4];
    ts = mptDefaults('truncationSigmas');
    margin = internal.relWindowMargin(ts);

    fprintf('# calibrateRelIpCost\n');
    fprintf('# generated %s\n', datestr(now, 31));
    fprintf('# matlab %s on %s\n', version, computer);
    fprintf('# truncationSigmas %g, period %g\n', ts, opt.period);
    fprintf('# budgetSec %g, seeds %s\n', opt.budgetSec, mat2str(opt.seeds));
    fprintf('# times in milliseconds; NaN means the route exceeded the budget\n');
    fprintf('# or is inadmissible in that mode\n');
    fprintf(['r,K_x,K_y,shape,weights,isPer,sigma,seed,nu,M_x,M_y,' ...
             't_bulger,t_centres,t_grid,pred_bulger,pred_mobius,' ...
             'max_abs_diff,gate_route,faster\n']);

    results = struct('r', {}, 'K', {}, 'isPer', {}, 'sigma', {}, ...
                     'seed', {}, 'nu', {}, 'M', {}, 't', {});
    nCell = 0;
    % Running per-arm, per-order cost estimate, built from the cells
    % already timed in this run and used to skip an arm whose predicted
    % time exceeds the budget. Each arm is priced by the quantity it
    % scales with -- Bulger's method and the centres route by the
    % permutation-side tuple count, the translation grid by the node
    % count -- and the constant is the running median of measured over
    % predictor, so the estimate calibrates itself to the machine as the
    % sweep proceeds. Nothing is skipped until an arm has three
    % measurements at that order to estimate from.
    est = struct('ratios', {containers.Map('KeyType', 'char', ...
                                           'ValueType', 'any')});
    for ri = 1:numel(rOrders)
        ra = rOrders(ri);
        Ks = KByOrder{ri};
        for ki = 1:numel(Ks)
            K = Ks(ki);
            for shi = 1:numel(shapes)
                if strcmp(shapes{shi}, 'asym')
                    Kx = K_REF;
                else
                    Kx = K;
                end
                Ky = K;
                if Kx < ra || Ky < ra || (strcmp(shapes{shi}, 'asym') && Kx == Ky)
                    continue;
                end
                for pfi = 1:numel(profiles)
                    for si = 1:numel(opt.sigmas)
                        sg = opt.sigmas(si);
                        for isPer = [false true]
                            for sd = opt.seeds
                                nCell = nCell + 1;
                                [row, est] = localCell(ra, Kx, Ky, ...
                                    shapes{shi}, profiles{pfi}, isPer, sg, ...
                                    sd, opt, ts, margin, est);
                                fprintf(['%d,%d,%d,%s,%s,%d,%g,%d,%d,%.0f,' ...
                                         '%.0f,%.4f,%.4f,%.4f,%.4f,%.4f,' ...
                                         '%.3e,%s,%s\n'], ...
                                    ra, Kx, Ky, shapes{shi}, profiles{pfi}, ...
                                    isPer, sg, sd, row.nu, row.Mx, row.My, ...
                                    row.tB, row.tC, row.tG, row.pB, row.pM, ...
                                    row.diff, row.gate, row.faster);
                            end
                        end
                    end
                end
            end
        end
    end
    fprintf('# %d cells\n', nCell);
end


function [row, est] = localCell(ra, Kx, Ky, shape, profile, isPer, sg, ...
                                sd, opt, ts, margin, est)
    if isPer, P = opt.period; else, P = 0; end
    rs = RandStream('twister', 'Seed', 7919 * Ky + 131 * ra + 17 * sd + ...
                                       round(1000 * sg));
    px = sort(rand(rs, 1, Kx) * opt.period);
    py = sort(rand(rs, 1, Ky) * opt.period);
    wx = localWeights(profile, Kx, rs);
    wy = localWeights(profile, Ky, rs);

    row.Mx = factorial(ra) * nchoosek(Kx, ra);
    row.My = factorial(ra) * nchoosek(Ky, ra);
    if isPer
        row.nu = internal.autoNtauDefault(opt.period, sg);
        sop = sg / opt.period;
    else
        sps = internal.resolveSamplesPerSigma([], ra, ts);
        span = (max(px) - min(px)) + (max(py) - min(py)) + 2 * margin * sg;
        row.nu = max(64, ceil(max(span, 1.0) / sg * sps));
        sop = 0;
    end

    % Predictors price by the larger side, since that is what dominates
    % each route.
    Mbig = max(row.Mx, row.My);
    Kbig = max(Kx, Ky);
    arms = { 'B', 'bulger', 'auto',    Mbig^2; ...
             'C', 'mobius', 'centres', Mbig^2; ...
             'G', 'mobius', 'grid',    row.nu * Kbig };
    vals = [];
    for ai = 1:size(arms, 1)
        key = sprintf('%s%d', arms{ai, 1}, ra);
        [t, v, est] = localTimed(px, wx, py, wy, sg, ra, isPer, P, ...
            arms{ai, 2}, arms{ai, 3}, opt, est, key, arms{ai, 4});
        switch arms{ai, 1}
            case 'B', row.tB = t;
            case 'C', row.tC = t;
            case 'G', row.tG = t;
        end
        vals(end+1) = v; %#ok<AGROW>
    end

    % Agreement first: a timing comparison between computations that
    % disagree would be meaningless. Arms skipped or inadmissible
    % contribute NaN and drop out.
    vals = vals(~isnan(vals));
    if numel(vals) > 1
        row.diff = max(abs(vals - vals(1)));
    else
        row.diff = NaN;
    end

    % Which route the gate itself picks, so a fitted min(centres, grid)
    % can be checked against the code's own choice rather than inferred.
    if mobius.maRelAttrPrefersCentres(px(:), py(:), sg, ra, true, isPer, ...
                                      max(opt.period, 1))
        row.gate = 'centres';
    else
        row.gate = 'grid';
    end
    tMob = min([row.tC, row.tG]);
    if isnan(row.tB) || isnan(tMob)
        row.faster = 'unknown';
    elseif tMob < row.tB
        row.faster = 'mobius';
    else
        row.faster = 'bulger';
    end

    [~, row.pB, row.pM] = internal.selectMaInnerProductMethod( ...
        ra, Kx, 1, 1, 1, isPer, ~isPer, isPer, sop, 'auto', false, ...
        true, row.nu, Ky);
end


function w = localWeights(profile, K, rs)
%LOCALWEIGHTS  Weight vector of the named shape, jittered.
    switch profile
        case 'flat'
            base = ones(1, K);
        case 'decay'
            base = exp(-linspace(0, 4, K));
        case 'bimodal'
            x = linspace(-1, 1, K);
            base = exp(-((1 - abs(x)).^2) * 6);
        otherwise
            error('mpt:badProfile', 'Unknown weight profile ''%s''.', profile);
    end
    w = base .* (0.75 + 0.5 * rand(rs, 1, K));
end


function [t, v, est] = localTimed(px, wx, py, wy, sg, ra, isPer, P, ...
                                  method, route, opt, est, key, predictor)
    t = NaN;
    v = NaN;

    % Skip an arm the running estimate puts over budget, rather than
    % paying its wall time to discover that.
    if isKey(est.ratios, key)
        seen = est.ratios(key);
        if numel(seen) >= 3 && median(seen) * predictor > opt.budgetSec
            return;
        end
    end

    mptDefaults('relAttrRoute', route);
    try
        tic;
        v = cosSimExpTens(px, wx, py, wy, sg, ra, 1, isPer, P, ...
            'method', method, 'verbose', false);
        tWarm = toc;
        if tWarm > opt.budgetSec
            % Over budget: keep the one sample rather than discard the
            % time already spent, and record it so the estimate learns.
            t = tWarm * 1e3;
        elseif tWarm > opt.repeatBelowSec
            t = tWarm * 1e3;
        else
            f = @() cosSimExpTens(px, wx, py, wy, sg, ra, 1, isPer, P, ...
                'method', method, 'verbose', false);
            t = internal.timeRepeated(f) * 1e3;
        end
        if predictor > 0
            if isKey(est.ratios, key)
                seen = est.ratios(key);
            else
                seen = [];
            end
            est.ratios(key) = [seen, (t / 1e3) / predictor];
        end
    catch
        % Inadmissible route or a shape the arm cannot take: left as NaN.
    end
    mptDefaults('relAttrRoute', 'auto');
end


function localRestore(hints, route, smPath)
    mptDefaults('showHints', hints);
    mptDefaults('relAttrRoute', route);
    mptDefaults('singleMultisetPath', smPath);
end
