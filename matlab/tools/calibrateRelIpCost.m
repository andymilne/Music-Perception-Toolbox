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
%   'sigmas'      kernel widths to sweep (default [3 25])
%   'seeds'       repeats per cell with fresh values (default 1)
%   'budgetSec'      skip a route predicted, or found, to exceed this
%                    (default 5)
%   'repeatBelowSec' repeat-time only calls faster than this; slower ones
%                    are measured from their single warm call (default
%                    0.25). Repeating a call that already ran for a
%                    second buys nothing: timing scatter is a fixed
%                    overhead, so its share of a long call is negligible,
%                    and the repeats cost several times the measurement.
%   'period'      periodic-mode period in cents (default 1200)
%   'check'       run one cell per swept axis, asserting each does what it
%                 claims, and return without sweeping (default false)
%   'outFile'     where the measurements are written (default
%                 'relIpCost.csv'). The file is written directly rather
%                 than captured from the console: the progress notes go
%                 to the console too, and DIARY records both, splicing
%                 notes into data rows. A third of one run was lost that
%                 way.
%
% Runtime. 171 combinations of tuple order, value counts, weight profile
% and event count x 2 widths x 2 periodicities x 1 seed = 684 cells, each
% timing three arms. The same 684 as the twin sweep in
% tools/calibrate_rel_ip_cost.py. Three things keep that affordable: an
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
    p.addParameter('sigmas', [3 25], @(x) isnumeric(x) && all(x > 0));
    p.addParameter('seeds', 1, @isnumeric);
    p.addParameter('budgetSec', 5, @(x) isscalar(x) && x > 0);
    p.addParameter('repeatBelowSec', 0.25, @(x) isscalar(x) && x > 0);
    p.addParameter('period', 1200, @(x) isscalar(x) && x > 0);
    p.addParameter('check', false, @(x) islogical(x) || isnumeric(x));
    p.addParameter('outFile', 'relIpCost.csv', @(x) ischar(x) || isstring(x));
    p.parse(varargin{:});
    opt = p.Results;

    if opt.check
        results = localCheck(opt.period);
        return;
    end

    prevHints = mptDefaults('showHints');
    prevRoute = mptDefaults('relAttrRoute');
    mptDefaults('showHints', false);
    cleanup = onCleanup(@() localRestore(prevHints, prevRoute));

    % Value counts per tuple order. Bulger's method builds K!/(K-r)!
    % tuples per side, so the affordable range narrows sharply with r.
    % K = 100 at r = 2 is dropped: through the MA path its Bulger arm runs
    % for seconds, and the r = 2 curve is already determined by K = 64.
    % Three value counts per order rather than six: the event count is
    % now a swept axis too, and a full factorial over both would run for
    % hours. Geometric spacing separates a power law as well as a dense
    % grid does.
    KByOrder = {[6 16 40], [6 12 24], [5 8 12]};

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

    % Event counts. Both methods price per event pair, but they do not
    % scale with the pair count the same way: Bulger's method builds one
    % joint tuple-pair kernel over all events at once, so its working set
    % grows with the product, while the Mobius method repeats a per-pair
    % cost. A sweep at one event per density leaves that unconstrained,
    % and it is a large effect -- at twelve events against twelve values,
    % relative periodic at sigma/period = 0.05, Bulger's method takes
    % 50.3 s where the Mobius method takes 5.5 ms, a factor of 9100. A
    % model fitted on single-event cells alone put that on the wrong side.
    % The range runs well past where Bulger's method can be timed:
    % measured at K = 8, it reaches 5.4 s by twelve events, while the
    % Mobius method is still 43 ms at 128. Those cells are not wasted. An
    % arm that exceeds the budget is recorded as Inf rather than dropped,
    % which is a censored observation -- its time is unknown but bounded
    % below -- and that is enough to settle which method is faster, which
    % is what the routing fit is scored on. Dropping them would discard
    % exactly the cells where the decision is most consequential.
    %
    % Capped by tuple order. Cost grows with the tuple order, the value
    % count and the event count together, and the budget can only decline
    % to START an arm --- neither language can interrupt one already
    % running --- so the worst cell has to be bounded by construction.
    % Sixty-four events at r = 4 is not a workload anyone runs;
    % sixty-four at r = 2 is, and that is where the range is wanted.
    NByOrder = {[1 4 16 64], [1 4 16], [1 4 8]};
    rOrders  = [2 3 4];
    ts = mptDefaults('truncationSigmas');
    margin = internal.relWindowMargin(ts);

    fid = fopen(char(opt.outFile), 'w');
    if fid < 0
        error('mpt:cannotWrite', 'Cannot open %s for writing.', ...
              char(opt.outFile));
    end
    closeFile = onCleanup(@() fclose(fid));

    fprintf(fid, '# calibrateRelIpCost\n');
    fprintf(fid, '# generated %s\n', datestr(now, 31));
    fprintf(fid, '# matlab %s on %s\n', version, computer);
    fprintf(fid, '# truncationSigmas %g, period %g\n', ts, opt.period);
    fprintf(fid, '# budgetSec %g, seeds %s\n', opt.budgetSec, mat2str(opt.seeds));
    fprintf(fid, ['# times in milliseconds. Inf means the arm exceeded the ' ...
             'budget: its time is\n']);
    fprintf(fid, ['# unknown but at least the budget, which still settles ' ...
             'the comparison. NaN\n']);
    fprintf(fid, ['# means the arm is inadmissible in that mode, or ' ...
                  'errored. A declined_ column\n']);
    fprintf(fid, '# reads - when the arm was measured normally.\n');
    fprintf(fid, '%s\n', localHeader());

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
                  Ns = NByOrder{ri};
                  for ni = 1:numel(Ns)
                    for si = 1:numel(opt.sigmas)
                        sg = opt.sigmas(si);
                        for isPer = [false true]
                            for sd = opt.seeds
                                nCell = nCell + 1;
                                fprintf(2, ['  r=%d K=%d/%d N=%d %s %s ' ...
                                            'per=%d sigma=%g\n'], ra, Kx, ...
                                    Ky, Ns(ni), shapes{shi}, ...
                                    profiles{pfi}, isPer, sg);
                                [row, est] = localCell(ra, Kx, Ky, ...
                                    Ns(ni), shapes{shi}, ...
                                    profiles{pfi}, isPer, sg, sd, opt, ts, ...
                                    margin, est);
                                localAssertScalars(row);
                                fprintf(fid, ['%d,%d,%d,%d,%s,%s,%d,%g,%d,%d,' ...
                                         '%.0f,%.0f,%.4f,%.4f,%.4f,%.4f,' ...
                                         '%.4f,%.3e,%s,%s,%s,%s,%s\n'], ...
                                    ra, Kx, Ky, Ns(ni), shapes{shi}, ...
                                    profiles{pfi}, ...
                                    isPer, sg, sd, row.nu, row.Mx, row.My, ...
                                    row.tB, row.tC, row.tG, row.pB, row.pM, ...
                                    row.diff, row.gate, row.faster, ...
                                    row.whyB, row.whyC, row.whyG);
                            end
                        end
                    end
                  end
                end
            end
        end
    end
    fprintf(fid, '# %d cells\n', nCell);
end


function [row, est] = localCell(ra, Kx, Ky, N, shape, profile, isPer, sg, ...
                                sd, opt, ts, margin, est)
    if isPer, P = opt.period; else, P = 0; end
    rs = RandStream('twister', 'Seed', 7919 * Ky + 131 * ra + 17 * sd + ...
                                       round(1000 * sg));
    px = sort(rand(rs, Kx, N) * opt.period, 1);
    py = sort(rand(rs, Ky, N) * opt.period, 1);
    wx = zeros(Kx, N);
    wy = zeros(Ky, N);
    for nn = 1:N
        wx(:, nn) = localWeights(profile, Kx, rs).';
        wy(:, nn) = localWeights(profile, Ky, rs).';
    end

    row.Mx = factorial(ra) * nchoosek(Kx, ra);
    row.My = factorial(ra) * nchoosek(Ky, ra);
    if isPer
        row.nu = internal.autoNtauDefault(opt.period, sg);
        sop = sg / opt.period;
    else
        sps = internal.resolveSamplesPerSigma([], ra, ts);
        % Colon-subscripted: px is K x N, and MAX over a matrix reduces
        % down columns, giving one value per event rather than one for
        % the whole density. That made row.nu a vector, which shifted
        % every field after it in the output row and corrupted 306 of
        % 816 measurements.
        span = (max(px(:)) - min(px(:))) + (max(py(:)) - min(py(:))) ...
               + 2 * margin * sg;
        row.nu = max(64, ceil(max(span, 1.0) / sg * sps));
        sop = 0;
    end

    % Predictors price by the larger side, since that is what dominates
    % each route.
    % Both sides scale per event pair, so the predictors carry it.
    Mbig = max(row.Mx, row.My);
    Kbig = max(Kx, Ky);
    pairs = N * N;
    % Bulger's joint tuple-pair kernel is held at once, and grows with the
    % event count as well as the value count: at r = 2, K = 40 and 64
    % events it is 4.98e9 entries, 40 GB. Such a cell cannot be timed --
    % the call raises -- and an arm that raises was recorded as missing,
    % losing the cell. It is not missing: a method that cannot run is
    % decisively the slower one, so it is censored instead.
    nJ = N * factorial(ra) * nchoosek(Kx, ra);
    nK = N * nchoosek(Ky, ra);
    tooBig = (nJ * nK) > 2.0e8;
    % Densities are built explicitly. Passing 2-D arrays to the raw entry
    % would be read as a *batch of collections*, one per column, not as
    % one density carrying N events -- a different computation, and one
    % that cannot even broadcast when the two sides carry different value
    % counts.
    % The multi-attribute signature, selected by passing cells, is the one
    % that carries events: it keeps a (K, N) matrix as K values across N
    % events. The single-multiset signature flattens the same array into
    % one event of K*N values, which is a different density entirely.
    densX = buildExpTens({px}, {wx}, sg, ra, 1, isPer, P, 'verbose', false);
    densY = buildExpTens({py}, {wy}, sg, ra, 1, isPer, P, 'verbose', false);
    arms = { 'B', 'bulger', 'auto',    pairs * Mbig^2; ...
             'C', 'mobius', 'centres', pairs * Mbig^2; ...
             'G', 'mobius', 'grid',    pairs * row.nu * Kbig };
    vals = {};
    for ai = 1:size(arms, 1)
        key = sprintf('%s%d', arms{ai, 1}, ra);
        infeasible = strcmp(arms{ai, 1}, 'B') && tooBig;
        [t, v, est, why] = localTimed(densX, densY, arms{ai, 2}, ...
            arms{ai, 3}, opt, est, key, arms{ai, 4}, infeasible);
        switch arms{ai, 1}
            case 'B', row.tB = t; row.whyB = why;
            case 'C', row.tC = t; row.whyC = why;
            case 'G', row.tG = t; row.whyG = why;
        end
        vals{end+1} = v; %#ok<AGROW>
    end

    % Agreement first: a timing comparison between computations that
    % disagree would be meaningless. Arms skipped or inadmissible
    % contribute NaN and drop out.
    % Multi-event calls return one cosine per event pair, so the
    % agreement check is elementwise.
    keep = ~cellfun(@(v) isempty(v) || any(~isfinite(v(:))), vals);
    vals = vals(keep);
    if numel(vals) > 1
        row.diff = max(cellfun(@(v) max(abs(v(:) - vals{1}(:))), vals));
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
    % A censored arm still settles the comparison whenever the other side
    % is measured and below the bound.
    tMob = min([row.tC, row.tG]);
    if isnan(row.tB) || isnan(tMob) || (isinf(row.tB) && isinf(tMob))
        row.faster = 'unknown';
    elseif tMob < row.tB
        row.faster = 'mobius';
    else
        row.faster = 'bulger';
    end

    [~, row.pB, row.pM] = internal.selectMaInnerProductMethod( ...
        ra, Kx, 1, N, N, isPer, ~isPer, isPer, sop, 'auto', false, ...
        true, row.nu, Ky);
end


function nFail = localCheck(period)
%LOCALCHECK  Assert each swept axis does what it claims, on one cell each.
%
%   Every defect this sweep has carried was an axis quietly not varying
%   what it was supposed to: the route lever pinned a choice the workload
%   never reached, the cells ran through a different code path, and a
%   (K, N) array was flattened into one event of K*N values rather than
%   kept as K values across N events. Each showed up only after a full
%   run, or not at all. These take seconds.
%
%   Twin of the --check mode in tools/calibrate_rel_ip_cost.py.

    prevHints = mptDefaults('showHints');
    prevRoute = mptDefaults('relAttrRoute');
    mptDefaults('showHints', false);
    cleanup = onCleanup(@() localRestore(prevHints, prevRoute));

    nFail = 0;
    K = 8; N = 6; sigma = 25; r = 2;
    rs = RandStream('twister', 'Seed', 0);
    px = sort(rand(rs, K, N) * period, 1);
    wx = ones(K, N);

    % The multi-attribute signature (cells) carries events; the
    % single-multiset signature flattens the same array into one event.
    densMA = buildExpTens({px}, {wx}, sigma, r, 1, true, period, ...
                          'verbose', false);
    densFlat = buildExpTens(px, wx, sigma, r, 1, true, period, ...
                            'verbose', false);
    nEv = size(densMA.pAttr{1}, 2);
    nFail = nFail + localOk('event count reaches the density', ...
        nEv == N, sprintf('built %d, wanted %d', nEv, N));
    nFail = nFail + localOk('the flattening signature is not the one used', ...
        size(densFlat.pAttr{1}, 2) == 1, ...
        sprintf('flattened to %dx%d', size(densFlat.pAttr{1}, 1), ...
                size(densFlat.pAttr{1}, 2)));

    % A multi-event density is one density, so its cosine is one number.
    v = cosSimExpTens(densMA, densMA, 'method', 'mobius', 'verbose', false);
    nFail = nFail + localOk('multi-event call returns one cosine', ...
        isscalar(v), sprintf('numel %d', numel(v)));
    nFail = nFail + localOk('self-similarity is 1', ...
        abs(v - 1) < 1e-9, sprintf('%.12f', v));

    % The route lever must change which route runs, and not the answer.
    % Asserted on the gate rather than on a stopwatch: the gate is what
    % the lever exists to override, and a timing threshold would be a
    % machine-dependent assertion in a check whose whole purpose is to be
    % reliable. The gate is queried on a shape where it prefers grid, so
    % forcing centres has something to overturn.
    bigP = sort(rand(rs, 40, 1) * period, 1);
    routes = {'auto', 'centres', 'grid'};
    picks = false(1, 3);
    for ii = 1:3
        mptDefaults('relAttrRoute', routes{ii});
        picks(ii) = mobius.maRelAttrPrefersCentres(bigP, bigP, sigma, r, ...
            true, true, period);
    end
    mptDefaults('relAttrRoute', 'auto');
    nFail = nFail + localOk('route lever overrides the gate', ...
        picks(2) && ~picks(3), ...
        sprintf('auto=%d, centres=%d, grid=%d', picks(1), picks(2), picks(3)));

    % And the two routes must agree on the answer, which is not
    % machine-dependent at all.
    vv = zeros(1, 2);
    for ii = 1:2
        mptDefaults('relAttrRoute', routes{ii + 1});
        vv(ii) = cosSimExpTens(densMA, densMA, 'method', 'mobius', ...
                               'verbose', false);
    end
    mptDefaults('relAttrRoute', 'auto');
    nFail = nFail + localOk('routes agree on the value', ...
        abs(vv(1) - vv(2)) < 1.5e-8, sprintf('gap %.2e', abs(vv(1) - vv(2))));

    % Unequal value counts must reach both sides.
    py = sort(rand(rs, 3 * K, N) * period, 1);
    densY = buildExpTens({py}, {ones(3 * K, N)}, sigma, r, 1, true, ...
                         period, 'verbose', false);
    v2 = cosSimExpTens(densMA, densY, 'method', 'mobius', 'verbose', false);
    nFail = nFail + localOk('unequal value counts run', all(isfinite(v2(:))));

    % Weight profiles must actually differ.
    rs2 = RandStream('twister', 'Seed', 1);
    spreads = zeros(1, 3);
    names = {'flat', 'decay', 'bimodal'};
    for ii = 1:3
        w = localWeights(names{ii}, 12, rs2);
        spreads(ii) = max(w) / min(w);
    end
    nFail = nFail + localOk('weight profiles differ', ...
        numel(unique(round(spreads, 3))) == 3, ...
        sprintf('%s %.1fx, %s %.1fx, %s %.1fx', names{1}, spreads(1), ...
                names{2}, spreads(2), names{3}, spreads(3)));

    % The memory rule must decline the cell that cannot run and admit one
    % that can.
    big = 64 * factorial(2) * nchoosek(40, 2) * 64 * nchoosek(40, 2);
    small = 4 * factorial(2) * nchoosek(6, 2) * 4 * nchoosek(6, 2);
    nFail = nFail + localOk('memory rule declines a 40 GB cell', big > 2.0e8);
    nFail = nFail + localOk('memory rule admits a small cell', small < 2.0e8);

    % A formatted row must carry as many fields as the header, and must
    % end in a newline. Checked by formatting one, because the failure
    % mode is silent: fprintf drops empty arguments, runs out of format
    % before the newline, and the next row continues on the same line.
    probe = sprintf(['%d,%d,%d,%d,%s,%s,%d,%g,%d,%d,' ...
                     '%.0f,%.0f,%.4f,%.4f,%.4f,%.4f,' ...
                     '%.4f,%.3e,%s,%s,%s,%s,%s\n'], ...
                    2, 6, 6, 1, 'equal', 'flat', 0, 3, 1, 100, 30, 30, ...
                    1, 1, 1, 1, 1, 1e-13, 'centres', 'mobius', '-', '-', '-');
    nFail = nFail + localOk('a formatted row ends in a newline', ...
        ~isempty(probe) && probe(end) == newline);
    nFail = nFail + localOk('a formatted row has the header''s field count', ...
        numel(strsplit(strtrim(probe), ',')) == localRowFields(), ...
        sprintf('row %d, want %d', numel(strsplit(strtrim(probe), ',')), ...
                localRowFields()));

    % Every value in a row must be a scalar. A multi-event density makes
    % MAX over a K x N matrix return a row vector, and one non-scalar
    % field shifts every field after it.
    rs3 = RandStream('twister', 'Seed', 3);
    pxm = sort(rand(rs3, 8, 5) * period, 1);
    spanScalar = (max(pxm(:)) - min(pxm(:)));
    nFail = nFail + localOk('span reduces over the whole density', ...
        isscalar(spanScalar), sprintf('numel %d', numel(spanScalar)));
    nFail = nFail + localOk('the column-wise form would not be scalar', ...
        ~isscalar(max(pxm) - min(pxm)), ...
        sprintf('numel %d', numel(max(pxm) - min(pxm))));

    nHeader = numel(strsplit(localHeader(), ','));
    nFail = nFail + localOk('header and rows have the same field count', ...
        nHeader == localRowFields(), ...
        sprintf('header %d, row %d', nHeader, localRowFields()));

    if nFail == 0
        fprintf('\nall checks passed\n');
    else
        fprintf('\n%d failed\n', nFail);
    end
end


function bad = localOk(name, cond, detail)
%LOCALOK  Print one check result; return 1 if it failed.
    if nargin < 3, detail = ''; end
    if cond
        tag = 'pass';
    else
        tag = 'FAIL';
    end
    if isempty(detail)
        fprintf('  %s  %s\n', tag, name);
    else
        fprintf('  %s  %s   %s\n', tag, name, detail);
    end
    bad = double(~cond);
end


function localAssertScalars(row)
%LOCALASSERTSCALARS  Every numeric field of a row must be a scalar.
%
%   One non-scalar shifts every field after it in the output line, which
%   is silent in the file and only shows up when the data is parsed. It
%   has happened twice; the cost is one comparison per row.
    names = {'nu', 'Mx', 'My', 'tB', 'tC', 'tG', 'pB', 'pM', 'diff'};
    for ii = 1:numel(names)
        v = row.(names{ii});
        if ~isscalar(v)
            error('mpt:nonScalarField', ...
                  ['Row field ''%s'' has %d elements, not 1. One ' ...
                   'non-scalar shifts every field after it.'], ...
                  names{ii}, numel(v));
        end
    end
end


function h = localHeader()
%LOCALHEADER  The one place the column names live, so the header cannot
%   drift from the rows. Its width is asserted against LOCALROWFIELDS by
%   the check mode: an edit has already added columns to one and not the
%   other, and the mismatch was only noticed after a full sweep had been
%   run.
    h = ['r,K_x,K_y,N,shape,weights,isPer,sigma,seed,nu,M_x,M_y,' ...
         't_bulger,t_centres,t_grid,pred_bulger,pred_mobius,' ...
         'max_abs_diff,gate_route,faster,declined_bulger,' ...
         'declined_centres,declined_grid'];
end


function n = localRowFields()
%LOCALROWFIELDS  Number of values the row format string emits.
    n = 23;
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


function [t, v, est, why] = localTimed(densX, densY, method, route, opt, ...
                                       est, key, predictor, infeasible)
    t = NaN;
    v = NaN;

    % Skip an arm the running estimate puts over budget, rather than
    % paying its wall time to discover that.
    % Never empty: MATLAB flattens fprintf's arguments into one list and
    % an empty array vanishes from it, so three empty reasons would leave
    % 20 values for 23 conversions, the format would run out before the
    % newline, and consecutive rows would run together on one line. That
    % cost 306 of 816 measurements before it was noticed.
    why = '-';
    if nargin >= 9 && infeasible
        t = Inf; why = 'memory';   % exact arithmetic: does not fit
        return;
    end
    % Prior cost per predictor unit, used before any measurement exists to
    % estimate from, so the first cells of a sweep cannot each cost
    % minutes. The running median replaces it after three samples.
    priors = struct('B', 2.0e-8, 'C', 2.0e-8, 'G', 1.0e-8);
    learned = isKey(est.ratios, key) && numel(est.ratios(key)) >= 3;
    if learned
        rate = median(est.ratios(key));
    else
        rate = priors.(key(1));
    end
    if rate * predictor > opt.budgetSec
        % Censored, not missing. Which rate made the call is recorded: a
        % prior-based skip is a guess, a learned one a measurement.
        t = Inf;
        if learned, why = 'rate'; else, why = 'prior'; end
        return;
    end

    mptDefaults('relAttrRoute', route);
    try
        tic;
        v = cosSimExpTens(densX, densY, 'method', method, ...
            'verbose', false);
        tWarm = toc;
        if tWarm > opt.budgetSec
            t = Inf; why = 'budget';       % ran, but over
        elseif tWarm > opt.repeatBelowSec
            t = tWarm * 1e3;
        else
            f = @() cosSimExpTens(densX, densY, 'method', method, ...
                'verbose', false);
            t = internal.timeRepeated(f) * 1e3;
        end
        if predictor > 0 && isfinite(t)
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


function localRestore(hints, route)
    mptDefaults('showHints', hints);
    mptDefaults('relAttrRoute', route);
end
