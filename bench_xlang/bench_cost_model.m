%% bench_cost_model.m — skip-flag pricing audit and Möbius per-matrix costs
%
%  MATLAB counterpart of bench_cost_model.py. Three parts:
%
%  1. Selector audit (deterministic): internal.selectMaInnerProductMethod
%     across a (r, K, N) grid with the self-matrix skip flags off (a
%     first pair: full triple) and on (a later broadcast pair: cross
%     only); rows where the routing flips are the cells the flags exist
%     for.
%  2. Möbius per-matrix timings: method = 'mobius' on absolute
%     densities under cosine-fresh (three matrices), oneSidedDenom-fresh
%     (cross + one self), and cosine-warm-selves (cross only; the memo
%     threaded via the cache-carrying outputs), giving measured
%     per-matrix costs to hold against the pricing's nMatrices/3
%     scaling of the fitted whole-triple constants.
%  3. Flip-cell behavioural check: both forced routes timed in the
%     warm-selves regime; the flags-on choice should be the faster.
%
%  Writes bench_cost_model_matlab.csv. Requires MATLAB
%  (internal.timeRepeated and +mobius use arguments blocks; not
%  Octave-runnable).


function bench_cost_model()

% Dispatch hints would print inside the timed closures and distort
% the measurements; silence them for the run (session-scoped).
mptDefaults('showHints', false);
internal.maybeShowTruncationNotice('suppress');
outPath = fullfile(fileparts(mfilename('fullpath')), ...
    'bench_cost_model_matlab.csv');
fid = fopen(outPath, 'w');
fprintf(fid, ['language,part,r,K,N,regime,chosen_off,chosen_on,' ...
              'pw_off,orbit_off,pw_on,orbit_on,ms\n']);

%% ---- Part 1: selector audit ----
fprintf('== selector skip-flag audit (abs, A=1, N_x=N_y) ==\n');
flips = zeros(0, 3);
flipRoutes = cell(0, 2);
for r = [2 3]
    for N = [4 16]
        for K = [4 6 8 12 16 24 40]
            [cOff, pwOff, orbOff] = internal.selectMaInnerProductMethod( ...
                r, K, 1, N, N, false, false, false, 0.0, 'auto', ...
                false, false, [], K, {}, [], ...
                false, false, false, false);
            [cOn, pwOn, orbOn] = internal.selectMaInnerProductMethod( ...
                r, K, 1, N, N, false, false, false, 0.0, 'auto', ...
                false, false, [], K, {}, [], ...
                true, true, true, true);
            isFlip = ~strcmp(cOff, cOn);
            if isFlip
                flips(end+1, :) = [r, K, N];              %#ok<SAGROW>
                flipRoutes(end+1, :) = {cOff, cOn};       %#ok<SAGROW>
            end
            fprintf(fid, ['matlab,selector,%d,%d,%d,,%s,%s,' ...
                          '%.4g,%.4g,%.4g,%.4g,\n'], ...
                r, K, N, cOff, cOn, pwOff, orbOff, pwOn, orbOn);
            if isFlip
                mark = '  <-- flip';
            else
                mark = '';
            end
            fprintf('  r=%d K=%3d N=%3d: off=%-6s on=%-6s%s\n', ...
                r, K, N, cOff, cOn, mark);
        end
    end
end

%% ---- Part 1b: selector audit, asymmetric (the broadcast regime) ----
% A large shared X against a small fixed query (K_y = 4, N_y = 3): the
% shared self matrix dominates the full-triple Bulger price, so
% skipping it (a later broadcast pair) is where the routing genuinely
% moves.
fprintf('\n== selector audit, asymmetric (K_y=4, N_y=3) ==\n');
for r = [2 3]
    for Nx = [8 32]
        for Kx = [6 8 12 16 24 40 64]
            [cOff, pwOff, orbOff] = internal.selectMaInnerProductMethod( ...
                r, Kx, 1, Nx, 3, false, false, false, 0.0, 'auto', ...
                false, false, [], 4, {}, [], ...
                false, false, false, false);
            [cOn, pwOn, orbOn] = internal.selectMaInnerProductMethod( ...
                r, Kx, 1, Nx, 3, false, false, false, 0.0, 'auto', ...
                false, false, [], 4, {}, [], ...
                true, true, true, true);
            isFlip = ~strcmp(cOff, cOn);
            if isFlip
                flips(end+1, :) = [r, Kx, Nx];            %#ok<SAGROW>
                flipRoutes(end+1, :) = {cOff, cOn};       %#ok<SAGROW>
            end
            fprintf(fid, ['matlab,selector_asym,%d,%d,%d,Ky4_Ny3,%s,%s,' ...
                          '%.4g,%.4g,%.4g,%.4g,\n'], ...
                r, Kx, Nx, cOff, cOn, pwOff, orbOff, pwOn, orbOn);
            if isFlip
                mark = '  <-- flip';
            else
                mark = '';
            end
            fprintf('  r=%d Kx=%3d Nx=%3d: off=%-6s on=%-6s%s\n', ...
                r, Kx, Nx, cOff, cOn, mark);
        end
    end
end

%% ---- Part 2: Möbius per-matrix timings ----
fprintf('\n== Möbius per-matrix timings (abs, A=1, K=8) ==\n');
for r = [2 3]
    for N = [20 60]
        dX = iBuildAbs(8, N, r, 0);
        dY = iBuildAbs(8, N, r, 137);

        fnCosFresh = @() cosSimExpTens(dX, dY, ...
            'method', 'mobius', 'verbose', false);
        fnOsdFresh = @() cosSimExpTens(dX, dY, ...
            'method', 'mobius', 'normalize', 'oneSidedDenom', ...
            'verbose', false);
        % Warm regime: seed both self terms once via the cache-carrying
        % outputs, then time calls on the seeded structs — the timed
        % closure computes the cross matrix only (matching the Python
        % bench, whose memo persists on the objects across calls).
        [~, dXw, dYw] = cosSimExpTens(dX, dY, 'method', 'mobius', ...
            'verbose', false);
        fnCosWarm = @() cosSimExpTens(dXw, dYw, 'method', 'mobius', ...
            'verbose', false);

        t3 = internal.timeRepeated(fnCosFresh) * 1000;
        t2 = internal.timeRepeated(fnOsdFresh) * 1000;
        t1 = internal.timeRepeated(fnCosWarm) * 1000;
        fprintf(fid, 'matlab,mobius_matrices,%d,8,%d,cos_fresh_3mat,,,,,,,%.5f\n', r, N, t3);
        fprintf(fid, 'matlab,mobius_matrices,%d,8,%d,osd_fresh_2mat,,,,,,,%.5f\n', r, N, t2);
        fprintf(fid, 'matlab,mobius_matrices,%d,8,%d,cos_warm_1mat,,,,,,,%.5f\n', r, N, t1);
        fprintf(['  r=%d N=%3d: 3mat %.3f  2mat %.3f  1mat %.3f ms' ...
                 '  ->  xy~%.3f  yy~%.3f  xx~%.3f\n'], ...
            r, N, t3, t2, t1, t1, max(t2 - t1, 0), max(t3 - t2, 0));
    end
end

%% ---- Part 3: flip-cell forced-route timings (warm selves) ----
fprintf('\n== flip-cell forced-route timings (warm selves) ==\n');
nCheck = min(size(flips, 1), 4);
for fi = 1:nCheck
    r = flips(fi, 1); K = flips(fi, 2); N = flips(fi, 3);
    % Flip cells arise on the asymmetric grid: shared X (K, N) against
    % the small fixed query (mirrors bench_cost_model.py).
    dX = iBuildAbs(K, N, r, 0);
    dY = iBuildAbs(4, 3, r, 137);
    tForced = struct();
    for methC = {'bulger', 'mobius'}
        meth = methC{1};
        % Seed both self terms under this route once, outside the timed
        % closure (see the part-2 warm-regime note).
        [~, dXw, dYw] = cosSimExpTens(dX, dY, 'method', meth, ...
            'verbose', false);
        fn = @() cosSimExpTens(dXw, dYw, 'method', meth, ...
            'verbose', false);
        t = internal.timeRepeated(fn) * 1000;
        tForced.(meth) = t;
        fprintf(fid, 'matlab,flip_check,%d,%d,%d,forced_%s_warm,%s,%s,,,,,%.5f\n', ...
            r, K, N, meth, flipRoutes{fi, 1}, flipRoutes{fi, 2}, t);
    end
    if tForced.bulger <= tForced.mobius
        faster = 'bulger';
    else
        faster = 'mobius';
    end
    if strcmp(faster, flipRoutes{fi, 2})
        verdict = 'OK';
    else
        verdict = 'MISPICK';
    end
    fprintf(['  r=%d K=%d N=%d: off->%s, on->%s; warm bulger %.3f ms, ' ...
             'mobius %.3f ms; faster=%s [%s]\n'], ...
        r, K, N, flipRoutes{fi, 1}, flipRoutes{fi, 2}, ...
        tForced.bulger, tForced.mobius, faster, verdict);
end

fclose(fid);
fprintf('\nwrote %s\n', outPath);

end


function d = iBuildAbs(K, N, r, seedShift)
%IBUILDABS  Deterministic absolute density (mirrors bench_cost_model.py).
    j = (1:(K * N)) + seedShift;
    vals = reshape(2000 * mod(11 * j.^2 + 5 * j, 397) / 397, K, N);
    d = buildExpTens({vals}, [], 30.0, r, false, false, 0, ...
        'verbose', false);
end

