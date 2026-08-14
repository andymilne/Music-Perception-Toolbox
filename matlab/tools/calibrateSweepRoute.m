function calibrateSweepRoute(outFile)
%CALIBRATESWEEPROUTE  Measure the mixture-vs-orbit crossover on this machine.
%
%   CALIBRATESWEEPROUTE() times both sweep routes over a grid of tuple
%   sizes and prints the two constants sweepCosSimExpTens should use on
%   this machine. CALIBRATESWEEPROUTE(OUTFILE) also writes the raw
%   per-cell timings to OUTFILE as CSV.
%
%   Why this exists. sweepCosSimExpTens picks between two decompositions
%   whose costs are counted in different units --- tuple pairs for the
%   mixture, orbit contractions over the value kernel for the orbit
%   route. Converting between those units needs one constant, and the
%   orbit route's fixed per-call overhead needs a second (a floor below
%   which the mixture always wins). Both are machine-specific in the
%   same way as mptDefaults('orbitCostIntercept').
%
%   Correctness does not depend on them: either route computes the same
%   quantity, and away from the crossover they differ by 10x or more, so
%   a misplaced constant costs nothing there. Only near-crossover
%   routing is affected.
%
%   Reading the output. The script prints, for each (K, r) cell, the
%   measured time for each route and the ratio orbitTotal / nPairs that
%   the chooser would compute. The chooser takes the orbit route when
%   that ratio is BELOW ORBIT_WORK_RATIO, so any threshold strictly
%   between the largest ratio at which the orbit route won and the
%   smallest at which the mixture won reproduces every measured winner.
%   The script reports that interval and suggests its geometric
%   midpoint, which is the most robust point within it. A threshold set
%   at either end misroutes the cell that end came from.
%
%   ORBIT_MIN_PAIRS should be set above the largest nPairs at which the
%   mixture won despite a favourable ratio --- that is the regime where
%   the orbit route's fixed overhead dominates.
%
%   Edit the two constants at the top of localChooseRoute in
%   sweepCosSimExpTens.m with the printed values.
%
%   Runtime is a few minutes; the largest cells dominate.
%
%   Example:
%       cd matlab/tools
%       calibrateSweepRoute('sweep_route_calibration.csv')

    if nargin < 1
        outFile = '';
    end

    % (K, r, N_context, N_query, M). Chosen to straddle the crossover:
    % the first rows should favour the mixture and the last the orbit
    % route. Single attribute keeps the cells interpretable.
    grid = [ 3 2  40 4 180
             4 2  40 4 180
             6 3  40 4 180
             6 4  40 4 180
             8 3  40 4 180
             8 4  40 4 180
             9 4  20 3 120
            10 4  20 3 120 ];

    reps = 3;
    rows = zeros(0, 7);

    % The default-truncation notice fires once per session, and would
    % otherwise land in the middle of the table. Draw it here.
    mptDefaults('truncationSigmas');

    fprintf('\n%6s %8s %12s %12s %12s %10s %9s\n', ...
            'K,r', 'nPairs', 'mixture (s)', 'orbit (s)', 'ratio', ...
            'faster', 'speedup');
    fprintf('%s\n', repmat('-', 1, 76));

    for g = 1:size(grid, 1)
        K = grid(g, 1); r = grid(g, 2);
        N = grid(g, 3); Ny = grid(g, 4); M = grid(g, 5);

        rng(11);
        pX = {sort(rand(K, N) * 80, 1)};
        pY = {sort(rand(K, Ny) * 20, 1)};
        off = linspace(0, 60, M);

        dX = buildExpTens(pX, [], 0.9, r, false, false, NaN, true, ...
                          'verbose', false);
        dY = buildExpTens(pY, [], 0.9, r, false, false, NaN, true, ...
                          'verbose', false);

        % nPairs and the chooser's work estimate, computed the same way
        % localChooseRoute computes them.
        cX = nchoosek(K, r); cY = nchoosek(K, r);
        nPairs = N * factorial(r) * cX * Ny * cY;
        if r >= 2
            nOrb = numel(mobius.getOrbitTable(r));
            orbitTotal = M * N * Ny * nOrb * K * K * r;
        else
            nOrb = NaN; orbitTotal = NaN;
        end
        ratio = orbitTotal / nPairs;

        tMix = localTime(@() sweepCosSimExpTens(dX, dY, off, ...
            'method', 'mixture', 'verbose', false), reps);
        tOrb = localTime(@() sweepCosSimExpTens(dX, dY, off, ...
            'method', 'orbit', 'verbose', false), reps);

        if tOrb < tMix
            faster = 'orbit';
        else
            faster = 'mixture';
        end
        fprintf('%3d,%-2d %8.3g %12.3f %12.3f %12.1f %10s %8.1fx\n', ...
                K, r, nPairs, tMix, tOrb, ratio, faster, ...
                max(tMix, tOrb) / min(tMix, tOrb));

        rows(end + 1, :) = [K, r, nPairs, tMix, tOrb, ratio, ...
                            double(tOrb < tMix)]; %#ok<AGROW>
    end

    % --- Recommended constants -----------------------------------------
    orbitWon  = rows(rows(:, 7) == 1, :);
    mixtureWon = rows(rows(:, 7) == 0, :);

    fprintf('\n');
    if isempty(orbitWon)
        fprintf(['No cell favoured the orbit route. Either the grid does ' ...
                 'not reach\nhigh enough tuple order on this machine, or ' ...
                 'the mixture is uniformly\nfaster here; leave the ' ...
                 'constants as shipped.\n']);
        return;
    end
    ratioLo = max(orbitWon(:, 6));
    fprintf('ORBIT_WORK_RATIO : orbit won up to ratio %.1f\n', ratioLo);
    if isempty(mixtureWon)
        fprintf(['                   no cell favoured the mixture; set ' ...
                 'above %.0f\n'], ratioLo);
    else
        ratioHi = min(mixtureWon(:, 6));
        fprintf('                   mixture won from ratio %.1f\n', ratioHi);
        if ratioHi <= ratioLo
            fprintf(['                   the two overlap: no single ' ...
                     'threshold reproduces every cell,\n' ...
                     '                   which means the cost model ' ...
                     'itself misranks this grid.\n']);
        else
            % Strictly inside the interval: a threshold AT either end
            % misroutes the cell that end came from.
            fprintf(['                   -> set strictly between %.1f ' ...
                     'and %.1f; suggested %.0f\n'], ...
                    ratioLo, ratioHi, sqrt(ratioLo * ratioHi));
            localReportChoice(rows, sqrt(ratioLo * ratioHi));
        end
    end

    % Cells where the ratio favoured orbit but the mixture still won are
    % the ones the floor exists to catch.
    overhead = rows(rows(:, 7) == 0 & rows(:, 6) <= ratioCut, :);
    if isempty(overhead)
        fprintf(['ORBIT_MIN_PAIRS  : no cell where a favourable ratio ' ...
                 'still lost;\n                   the shipped 1e6 is ' ...
                 'not contradicted here.\n']);
    else
        fprintf('ORBIT_MIN_PAIRS  : -> set above %.3g\n', ...
                max(overhead(:, 3)));
    end

    if ~isempty(outFile)
        fid = fopen(outFile, 'w');
        fprintf(fid, 'K,r,nPairs,tMixture,tOrbit,ratio,orbitWon\n');
        fprintf(fid, '%d,%d,%.0f,%.6f,%.6f,%.4f,%d\n', rows.');
        fclose(fid);
        fprintf('\nRaw timings written to %s\n', outFile);
    end
end


function localReportChoice(rows, threshold)
%LOCALREPORTCHOICE  Check a candidate threshold against every measured cell.
    nBad = 0;
    for i = 1:size(rows, 1)
        picksOrbit = rows(i, 6) < threshold;
        orbitWonHere = rows(i, 7) == 1;
        if picksOrbit ~= orbitWonHere
            if nBad == 0
                fprintf('                   cells this would misroute:\n');
            end
            nBad = nBad + 1;
            cost = max(rows(i, 4), rows(i, 5)) / min(rows(i, 4), rows(i, 5));
            fprintf(['                     K=%d r=%d ratio %.1f ' ...
                     '(%.1fx slower)\n'], rows(i, 1), rows(i, 2), ...
                    rows(i, 6), cost);
        end
    end
    if nBad == 0
        fprintf('                   reproduces every measured winner\n');
    end
end

function t = localTime(fn, reps)
%LOCALTIME  Minimum of REPS runs, after one warm-up.
%
%   The minimum rather than the mean: a single unrepeated timing is not
%   evidence, and the mean is pulled by scheduling noise that has
%   nothing to do with the routes.
    fn();                       % warm up caches and JIT
    ts = zeros(1, reps);
    for i = 1:reps
        tic; fn(); ts(i) = toc;
    end
    t = min(ts);
end
