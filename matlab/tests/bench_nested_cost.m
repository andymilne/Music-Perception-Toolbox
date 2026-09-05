function bench_nested_cost(varargin)
%BENCH_NESTED_COST  Timing grid for the nested-attribute IP cost model.
%
%  MATLAB twin of python/tools/calibrate_nested_cost.py: the same cells in
%  the same order, the same per-cell seeding convention, the same timing
%  discipline and the same CSV columns, so cell index i here and cell
%  index i there measure the same shape and the two grids can be fitted
%  side by side. The constants absorb per-language and per-machine factors
%  --- interpreter overhead, array layout, BLAS, JIT --- so each language
%  and each machine must be calibrated on its own measurements. This
%  harness MEASURES; python/tools/fit_nested_cost.py FITS.
%
%  HOW TO RUN
%  ----------
%  With the toolbox on the path, from anywhere:
%
%      >> bench_nested_cost
%      >> bench_nested_cost('out', '/path/to/nested_cost_matlab.csv')
%
%  The default output file is <repo>/_to_delete/nested_cost_matlab.csv
%  (the directory is created if missing). Rows are appended as each cell
%  completes and the file is closed after every row, so a run that is
%  interrupted --- by the user, by the machine, by an error --- keeps
%  every cell it finished, and a reader can watch the file while the run
%  proceeds. Re-running with the same 'out' SKIPS the cells already in the
%  file, so an interrupted run is resumed simply by running it again.
%
%  Each cell is wrapped in try/catch: a cell that raises is written as an
%  error row (all timings -1) with its message reported on stderr, and the
%  run continues. Nothing is threaded between cells --- the densities are
%  rebuilt inside every cell and every timed repeat, so no memo is warm
%  across arms --- which is the state the per-attribute laws predict.
%
%  OPTIONS (name-value)
%    'out'      CSV path (see above); '' to write no file (stdout only).
%    'start'    0-based index of the first cell to run (default 0).
%    'limit'    stop after this many cells (0 = no limit).
%    'seconds'  stop once this much wall time has been spent (0 = none).
%    'section'  restrict to these section letters, e.g. 'ABC'.
%    'list'     true to print the cell list and return without timing.
%
%  WHAT TO SEND BACK
%  -----------------
%  The CSV file, or everything between the BEGIN_CSV and END_CSV markers
%  (inclusive of the header row). Its columns fully determine the features
%  the laws are fitted against as well as the analytic terms themselves;
%  the terms are computed by INTERNAL.NESTEDCOSTTERMS, i.e. by the same
%  tupleCounts / recipeWork / quadNodes the dispatch itself uses.
%
%  WHAT IS TIMED
%  -------------
%  Per cell, each route that carries the cell's measure is forced by name
%  (INTERNAL.NESTEDCONTRACT's opts.forceRoute) and timed on the three
%  inner matrices the cosine needs (xy, xx, yy) with cold densities. The
%  joint-tuple enumeration is timed on the same pair through
%  COSSIMEXPTENS(..., 'method', 'bulger'), so the two sides are measured
%  over the same work. The route arms run on a density carrying the nested
%  attribute ALONE, mirroring Python's per-attribute _nested_attr_matrix
%  timing; the enumeration arm runs on the full density (section E adds a
%  flat companion attribute), which is what it must enumerate.
%
%  One arm is timed in Python but not here: the centres arm of a
%  relative-periodic cell above the sigma/period threshold, where that
%  route no longer carries the declared full-image measure and the
%  measure rule refuses to force it. Python patches the route hook and so
%  reaches past the rule; this harness records -1 rather than defeating a
%  correctness rule inside the toolbox. Only section C's sop = 0.05 cells
%  are affected, and the centres term does not move with sigma, so the
%  centres law loses no information.
%
%  A route whose analytic term exceeds its skip bound is not timed and is
%  recorded as -1. The bounds are Python's, so the two grids are skipped
%  on the same cells; a cell over the bound is one whose routing is
%  decided by orders of magnitude rather than by the fitted constants.
%
%  SECTIONS (identical to the Python harness)
%  A  shape sweep, B  event sweep, C  sigma sweep on relative-periodic
%  shapes, D  symmetry patterns, E  multi-attribute cells. See the Python
%  module docstring for what each constrains.

    % ---- options -----------------------------------------------------
    thisDir  = fileparts(mfilename('fullpath'));
    mlDir    = fileparts(thisDir);
    repoRoot = fileparts(mlDir);
    if exist('mptDefaults', 'file') ~= 2
        addpath(mlDir);
    end
    defOut = fullfile(repoRoot, '_to_delete', 'nested_cost_matlab.csv');

    opt = struct('out', defOut, 'start', 0, 'limit', 0, 'seconds', 0, ...
                 'section', '', 'list', false);
    if mod(numel(varargin), 2) ~= 0
        error('bench_nested_cost:badNV', 'Options are name-value pairs.');
    end
    for k = 1:2:numel(varargin)
        name = lower(char(varargin{k}));
        if ~isfield(opt, name)
            error('bench_nested_cost:unknownOption', ...
                  'Unknown option ''%s''.', name);
        end
        opt.(name) = varargin{k + 1};
    end

    cellsAll = localCells();
    if ~isempty(opt.section)
        keep = false(1, numel(cellsAll));
        for i = 1:numel(cellsAll)
            keep(i) = any(upper(opt.section) == cellsAll(i).section);
        end
        cellsAll = cellsAll(keep);
    end
    nCells = numel(cellsAll);

    if opt.list
        for i = 1:nCells
            fprintf('%d %s\n', i - 1, localHuman(cellsAll(i)));
        end
        fprintf('%d cells.\n', nCells);
        return;
    end

    % ---- output file -------------------------------------------------
    done = containers.Map('KeyType', 'char', 'ValueType', 'logical');
    outPath = char(opt.out);
    if ~isempty(outPath)
        outDir = fileparts(outPath);
        if ~isempty(outDir) && exist(outDir, 'dir') ~= 7
            mkdir(outDir);
        end
        if exist(outPath, 'file') == 2
            done = localReadDone(outPath);
        else
            fid = fopen(outPath, 'a');
            if fid < 0
                error('bench_nested_cost:openFailed', ...
                      'Could not open ''%s'' for writing.', outPath);
            end
            fprintf(fid, '%s\n', localHeader());
            fclose(fid);
        end
        fprintf('Writing rows to %s (%d already present).\n', ...
                outPath, done.Count);
    end

    % ---- defaults, restored on exit (including on error/Ctrl-C) ------
    prevHints = mptDefaults('showHints');
    hintsCleanup = onCleanup(@() mptDefaults('showHints', prevHints)); %#ok<NASGU>
    mptDefaults('showHints', false);

    fprintf('\n=== Nested-attribute cost-model calibration grid ===\n');
    fprintf(['%d cells; every cell rebuilds its densities, so no memo ' ...
             'is warm across arms.\n'], nCells);
    localWarmUp();

    rows = {};
    tStart = tic;
    i = double(opt.start);          % 0-based, as in Python
    while i < nCells
        if opt.limit > 0 && (i - double(opt.start)) >= opt.limit
            break;
        end
        if opt.seconds > 0 && toc(tStart) > opt.seconds
            break;
        end
        c = cellsAll(i + 1);
        key = localCellKey(c);
        if isKey(done, key)
            fprintf('[%d] skipped (already in file)\n', i);
            i = i + 1;
            continue;
        end
        try
            [row, human] = localRunCell(c, 1000 + i);
            fprintf('[%d] %s\n', i, human);
        catch err
            row = localErrorRow(c);
            fprintf(2, '[%d] ERROR %s: %s\n', i, err.identifier, err.message);
        end
        rows{end + 1} = row; %#ok<AGROW>
        if ~isempty(outPath)
            fid = fopen(outPath, 'a');
            if fid >= 0
                fprintf(fid, '%s\n', row);
                fclose(fid);        % close per row: the file is complete
            else                    % and flushed after every cell
                fprintf(2, 'WARNING: could not append row %d to %s\n', ...
                        i, outPath);
            end
        end
        i = i + 1;
    end

    fprintf('\nBEGIN_CSV\n');
    fprintf('%s\n', localHeader());
    for k = 1:numel(rows)
        fprintf('%s\n', rows{k});
    end
    fprintf('END_CSV\n');
    fprintf('# %d cells this run; next index %d of %d; %.1f s\n', ...
            numel(rows), i, nCells, toc(tStart));
    if i < nCells
        fprintf('# resume by running again with the same ''out'' file\n');
    end
end


% ======================================================================
%  Cell list --- the same cells, in the same order, as the Python
%  harness's cells(), so index i names the same shape in both languages.
% ======================================================================
function out = localCells()
    levelShapes = {[1 2], [2 2], [1 3], [2 3], [3 2]};
    symPatterns = {[1 1], [0 1], [1 0]};
    modes = [false false; false true; true false; true true];  % (rel, per)

    out = localCell('A', [1 1], [1 1], 2, 2, 2, false, false, 0.02, 0, 0);
    out(1) = [];    % empty struct array with the right fields

    % A: shape sweep.
    for s = 1:numel(levelShapes)
        rL = levelShapes{s};
        for chord = 2:4
            for ngroup = 2:4
                if chord < rL(1) || ngroup < rL(2)
                    continue;
                end
                for m = 1:size(modes, 1)
                    out(end + 1) = localCell('A', rL, [1 1], chord, ngroup, ...
                        2, modes(m, 1), modes(m, 2), 0.02, 0, 0); %#ok<AGROW>
                end
            end
        end
    end
    % B: event sweep on a reduced shape set.
    for s = 1:numel(levelShapes)
        rL = levelShapes{s};
        chord = max(2, rL(1));
        ngroup = max(2, rL(2));
        for N = [1 4]
            for m = 1:size(modes, 1)
                out(end + 1) = localCell('B', rL, [1 1], chord, ngroup, ...
                    N, modes(m, 1), modes(m, 2), 0.02, 0, 0); %#ok<AGROW>
            end
        end
    end
    % C: sigma sweep, relative-periodic only (the tau-grid node count).
    for s = 1:numel(levelShapes)
        rL = levelShapes{s};
        chord = max(2, rL(1));
        ngroup = max(2, rL(2));
        for sop = [0.005 0.05]
            out(end + 1) = localCell('C', rL, [1 1], chord, ngroup, 2, ...
                true, true, sop, 0, 0); %#ok<AGROW>
        end
    end
    % D: symmetry patterns.
    for s = 1:numel(levelShapes)
        rL = levelShapes{s};
        chord = max(2, rL(1)) + 1;
        ngroup = max(2, rL(2));
        for q = 2:numel(symPatterns)
            for m = 1:size(modes, 1)
                out(end + 1) = localCell('D', rL, symPatterns{q}, chord, ...
                    ngroup, 2, modes(m, 1), modes(m, 2), 0.02, 0, 0); %#ok<AGROW>
            end
        end
    end
    % E: multi-attribute cells.
    eShapes = {[1 2], [2 2], [2 3]};
    eFlat   = [1 3; 2 5];
    for s = 1:numel(eShapes)
        for f = 1:size(eFlat, 1)
            for m = 1:size(modes, 1)
                out(end + 1) = localCell('E', eShapes{s}, [1 1], 3, 3, 2, ...
                    modes(m, 1), modes(m, 2), 0.02, ...
                    eFlat(f, 1), eFlat(f, 2)); %#ok<AGROW>
            end
        end
    end
end


function c = localCell(section, rLevels, sym, chord, ngroup, N, rel, per, ...
                       sop, flatR, flatK)
    if per
        sigma = sop * localPeriod();
    else
        sigma = localSigmaNonPer();
        sop = 0.0;
    end
    c = struct('section', section, 'rLevels', double(rLevels(:)).', ...
               'sym', double(sym(:)).', 'chord', double(chord), ...
               'ngroup', double(ngroup), 'N', double(N), ...
               'rel', logical(rel), 'per', logical(per), ...
               'sop', double(sop), 'sigma', double(sigma), ...
               'flatR', double(flatR), 'flatK', double(flatK), ...
               'wrap', 'full-image');
end


function P = localPeriod()
    P = 12.0;
end


function S = localSpan()
    S = 10.0;
end


function S = localSigmaNonPer()
    S = 0.24;
end


% ======================================================================
%  Densities. One RandStream per (cell, seed), consumed in the same
%  order every time, so every repeat of a cell measures identical
%  densities --- cold, but the same.
% ======================================================================
function pr = localPair(c, seed)
    st = RandStream('mt19937ar', 'Seed', seed);
    [nx, fx] = localDraw(c, st, 0.0);
    [ny, fy] = localDraw(c, st, 0.13);
    pr = struct();
    pr.dxN = localBuildNested(c, nx);
    pr.dyN = localBuildNested(c, ny);
    if c.flatR > 0
        pr.dxF = localBuildFull(c, nx, fx);
        pr.dyF = localBuildFull(c, ny, fy);
    else
        pr.dxF = pr.dxN;
        pr.dyF = pr.dyN;
    end
end


function [v, f] = localDraw(c, st, jitter)
    K = c.chord * c.ngroup;
    if c.per
        hi = localPeriod();
    else
        hi = localSpan();
    end
    v = sort(rand(st, K, c.N) * hi, 1) + jitter;
    if c.per
        v = sort(mod(v, localPeriod()), 1);
    end
    f = [];
    if c.flatR > 0
        f = rand(st, c.flatK, c.N) * 5.0 + jitter;
    end
end


function spec = localNestedSpec(c)
    spec = struct('tags', repelem(0:(c.ngroup - 1), c.chord), ...
                  'r', c.rLevels, 'sym', logical(c.sym));
    if c.rel
        L = numel(c.rLevels);
        spec.rel = [zeros(1, L - 1), 1];
    end
end


function d = localBuildNested(c, v)
    if c.per
        period = localPeriod();
    else
        period = 0.0;
    end
    d = buildExpTens({v}, {[]}, 'specs', {localNestedSpec(c)}, ...
                     'sigma', c.sigma, 'isPer', c.per, ...
                     'period', period, 'verbose', false);
end


function d = localBuildFull(c, v, f)
    if c.per
        period = localPeriod();
    else
        period = 0.0;
    end
    sp1 = struct('r', c.flatR, 'rel', false, 'sym', true);
    d = buildExpTens({v, f}, {[], []}, ...
                     'specs', {localNestedSpec(c), sp1}, ...
                     'sigma', [c.sigma 0.5], 'isPer', [c.per false], ...
                     'period', [period 0.0], 'verbose', false);
end


% ======================================================================
%  Timing
% ======================================================================
function ms = localTimeMs(setupFn, runFn)
%LOCALTIMEMS  Median of several timed repeats, each on fresh state.
%   SETUPFN returns the arguments RUNFN consumes and is not timed; it is
%   what makes each repeat cold, which is the state a real call finds. A
%   first call slower than LONG_CALL_SEC is reported from that call
%   alone: repeating a call that already ran for a third of a second
%   removes scatter whose share of the total is negligible and costs the
%   rest of the budget. Mirror of the Python harness's _time_ms.
    BUDGET_SEC = 1.0;
    LONG_CALL_SEC = 0.3;
    N_TIMED = 5;

    args = setupFn();
    t0 = tic; runFn(args); first = toc(t0);
    if first > LONG_CALL_SEC
        ms = first * 1000.0;
        return;
    end
    samples = zeros(1, 0);
    spent = 0.0;
    for k = 1:N_TIMED
        args = setupFn();
        t0 = tic; runFn(args); dt = toc(t0);
        samples(end + 1) = dt * 1000.0; %#ok<AGROW>
        spent = spent + dt;
        if spent > BUDGET_SEC && numel(samples) >= 3
            break;
        end
    end
    s = sort(samples);
    ms = s(floor(numel(s) / 2) + 1);
end


function localRunRoute(pr, route)
%LOCALRUNROUTE  The three inner matrices, on the forced route. force =
%   true so an uncovered case raises here rather than silently returning
%   an empty triple that would be timed as "fast".
    o = struct('forceRoute', route);
    internal.nestedContract(pr.dxN, pr.dyN, 'cosine', [], true, o);
end


function localRunBulger(pr)
    cosSimExpTens(pr.dxF, pr.dyF, 'method', 'bulger', 'verbose', false);
end


function routes = localRoutesFor(c)
%LOCALROUTESFOR  The routes to time on this cell: those that carry its
%   measure (all wraps here are the default full-image).
    if ~c.rel
        routes = {'centres', 'contract'};
    elseif c.per
        routes = {'centres', 'taugrid'};
    else
        routes = {'centres', 'contract_relnonper'};
    end
end


function [row, human] = localRunCell(c, seed)
    CENTRES_TERM_SKIP = 3.0e6;
    CONTRACT_TERM_SKIP = 4.0e8;
    BULGER_TERM_SKIP = 6.0e6;

    pr = localPair(c, seed);
    % Terms come from the FULL pair, as in Python: attribute 1's terms do
    % not depend on the companion attribute, while the enumeration term
    % (bulger) is a property of the whole density.
    [terms, termBulger, info] = internal.nestedCostTerms(pr.dxF, pr.dyF, 1, []);

    ms = struct('centres', -1.0, 'taugrid', -1.0, ...
                'contract_relnonper', -1.0, 'contract', -1.0, ...
                'bulger', -1.0);
    parts = {};
    rts = localRoutesFor(c);
    for k = 1:numel(rts)
        route = rts{k};
        if strcmp(route, 'centres')
            bound = CENTRES_TERM_SKIP;
        else
            bound = CONTRACT_TERM_SKIP;
        end
        if terms.(route) > bound
            continue;               % stays -1
        end
        if strcmp(route, 'centres') && c.rel && c.per && ...
                c.sigma / localPeriod() > internal.relPerSigmaOverPThreshold([])
            % Above the sigma/period threshold the minimum-image centres
            % route no longer computes this attribute's declared
            % full-image measure, so the measure rule refuses to force it
            % and the arm is recorded as -1. (The Python harness reaches
            % past the rule by patching the route hook and does time it.
            % Nothing is lost for the fit: the centres term does not move
            % with sigma, so those cells duplicate the same-shape cells at
            % the lower sigma of section C.) Only section C's sop = 0.05
            % cells are affected; every other cell sits at 0.02.
            continue;
        end
        % Per-arm try/catch, inside the harness's per-cell one: an arm
        % that raises is recorded as -1 and the cell keeps its other
        % arms, which is what makes an unattended run informative rather
        % than merely finished.
        try
            setupFn = @() localPair(c, seed);
            runFn = @(a) localRunRoute(a, route);
            ms.(route) = localTimeMs(setupFn, runFn);
            parts{end + 1} = sprintf('%s=%.3f', route, ms.(route)); %#ok<AGROW>
        catch err
            fprintf(2, '    route %s failed (%s): %s\n', route, ...
                    err.identifier, err.message);
        end
    end
    if termBulger <= BULGER_TERM_SKIP
        try
            ms.bulger = localTimeMs(@() localPair(c, seed), ...
                                    @(a) localRunBulger(a));
            parts{end + 1} = sprintf('bulger=%.3f', ms.bulger); %#ok<AGROW>
        catch err
            fprintf(2, '    route bulger failed (%s): %s\n', ...
                    err.identifier, err.message);
        end
    end

    row = localRow(c, terms, termBulger, info, ms);
    human = sprintf('%s %s', localHuman(c), strjoin(parts, ' '));
end


% ======================================================================
%  CSV
% ======================================================================
function h = localHeader()
    h = ['section,r_levels,sym,chord,n_chords,K,N,rel,per,sigma,span,wrap,' ...
         'flat_r,flat_K,total_order,m_perm_x,m_comb_x,m_perm_y,restricted,' ...
         'work,n_tau,n_line,term_centres,term_taugrid,term_relnonper,' ...
         'term_contract,term_bulger,ms_centres,ms_taugrid,ms_relnonper,' ...
         'ms_contract,ms_bulger'];
end


function k = localCellKey(c)
%LOCALCELLKEY  The row's first fourteen fields, which identify the cell.
%   Used to skip cells already present in the output file.
    if c.per
        span = localPeriod();
    else
        span = localSpan();
    end
    k = sprintf('%s,%s,%s,%d,%d,%d,%d,%d,%d,%g,%g,%s,%d,%d', ...
        c.section, localJoin(c.rLevels), localJoin(c.sym), c.chord, ...
        c.ngroup, c.chord * c.ngroup, c.N, double(c.rel), double(c.per), ...
        c.sigma, span, c.wrap, c.flatR, c.flatK);
end


function row = localRow(c, terms, termBulger, info, ms)
%LOCALROW  One CSV row, one field per header column, in the header order.
    row = sprintf(['%s,%d,%g,%g,%g,%d,%g,%g,%g,' ...
                   '%.6g,%.6g,%.6g,%.6g,%.6g,' ...
                   '%.4f,%.4f,%.4f,%.4f,%.4f'], ...
        localCellKey(c), round(info.totalOrder), ...
        info.mPermX, info.mCombX, info.mPermY, double(info.restrictedX), ...
        info.workX, info.nTau, info.nLine, ...
        terms.centres, terms.taugrid, terms.contract_relnonper, ...
        terms.contract, termBulger, ...
        ms.centres, ms.taugrid, ms.contract_relnonper, ms.contract, ...
        ms.bulger);
end


function row = localErrorRow(c)
%LOCALERRORROW  A cell that raised: every measured and derived quantity
%   is -1, so the fitter drops the row and the run keeps its place.
    row = sprintf(['%s,-1,-1,-1,-1,-1,-1,-1,-1,' ...
                   '-1,-1,-1,-1,-1,' ...
                   '-1.0000,-1.0000,-1.0000,-1.0000,-1.0000'], ...
                  localCellKey(c));
end


function s = localJoin(v)
    parts = cell(1, numel(v));
    for k = 1:numel(v)
        parts{k} = sprintf('%d', round(v(k)));
    end
    s = strjoin(parts, '|');
end


function done = localReadDone(path)
%LOCALREADDONE  Keys of the cells already present in the output file.
    done = containers.Map('KeyType', 'char', 'ValueType', 'logical');
    fid = fopen(path, 'r');
    if fid < 0
        return;
    end
    cleaner = onCleanup(@() fclose(fid)); %#ok<NASGU>
    while true
        ln = fgetl(fid);
        if ~ischar(ln)
            break;
        end
        if isempty(ln) || strncmp(ln, 'section,', 8)
            continue;
        end
        f = strsplit(ln, ',');
        if numel(f) < 14
            continue;
        end
        done(strjoin(f(1:14), ',')) = true;
    end
end


function s = localHuman(c)
    if c.rel
        relS = 'rel';
    else
        relS = 'abs';
    end
    if c.per
        perS = 'per';
    else
        perS = 'np';
    end
    s = sprintf('%s r=[%s] sym=[%s] %dx%d N=%d %s-%s sigma=%g flat=%d |', ...
        c.section, localJoin(c.rLevels), localJoin(c.sym), c.chord, ...
        c.ngroup, c.N, relS, perS, c.sigma, c.flatR);
end


% ======================================================================
%  Warm-up: MATLAB pays one-time costs on first use of each path
%  (function compilation, +mobius package resolution, orbit and
%  contraction tables into the in-memory cache). Paying them here keeps
%  them out of cell 0. Failures are ignored: a warm-up is not a test.
% ======================================================================
function localWarmUp()
    fprintf('Warming the routes (untimed) ...');
    try
        warmCells = [localCell('A', [2 2], [1 1], 2, 2, 2, false, false, 0.02, 0, 0), ...
                     localCell('A', [2 2], [1 1], 2, 2, 2, true, true, 0.02, 0, 0), ...
                     localCell('A', [2 2], [1 1], 2, 2, 2, true, false, 0.02, 0, 0), ...
                     localCell('E', [2 2], [1 1], 3, 3, 2, false, false, 0.02, 1, 3)];
        for k = 1:numel(warmCells)
            c = warmCells(k);
            pr = localPair(c, 1);
            internal.nestedCostTerms(pr.dxF, pr.dyF, 1, []);
            rts = localRoutesFor(c);
            for q = 1:numel(rts)
                localRunRoute(pr, rts{q});
            end
            localRunBulger(pr);
        end
        fprintf(' done.\n\n');
    catch err
        fprintf(2, ' warm-up skipped (%s: %s)\n\n', err.identifier, err.message);
    end
end
