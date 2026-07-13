%% bench_ma_dispatch.m
%  Audits the multi-attribute (MAET) cosine dispatcher: times 'bulger',
%  'mobius', and 'auto' on MA densities and checks that 'auto' tracks
%  the faster method and that the two methods agree numerically.
%
%  Workload: N events, each with two attributes — a scalar onset
%  (K = 1, r = 1, absolute, non-periodic, sigma = 15) and a K-pitch
%  chord (r = 2 or 3, sigma = 6, period 1200). The pitch attribute's
%  rel/per flags sweep the four mode combinations. The scalar r = 1
%  attribute is deliberate: it is the canonical MAET pattern, and it
%  exercises the K-vs-r precision guard, which must exempt r_a = 1
%  attributes (no alternating sum, no cancellation risk) rather than
%  veto the Möbius method for the whole density.
%
%  Columns:
%    * t_bul / t_mob / t_auto — forced and automatic timings.
%    * auto~  — which forced method the auto timing sits closer to;
%      should match whichever of t_bul/t_mob is smaller (near-ties of
%      a few ms either way are noise).
%    * |ds|   — |s_mobius - s_bulger|; ~1e-9 or below. A LARGER value,
%      or exact equality of the two methods' outputs at sizes where
%      they should differ in the last digits, both warrant a report.
%    * ovh    — t_auto minus the faster forced method. The MA
%      dispatcher is a pure cost model (no probes), so this measures
%      only mispick cost and should sit near zero.
%
%  The relative-mode Möbius rows use small N and a TIME_CAP: the
%  batched translation-grid contraction is slab-bounded (no memory
%  blowups) but still costs N_u * K^2 kernel ops per event pair, so
%  at small K Bulger's per-pair closed form generally wins and 'auto'
%  is expected to route those rows to Bulger's method. Möbius wins
%  the relative modes at large K, where Bulger's r-tuple enumeration
%  compounds, or when the joint working set would exhaust memory.
%
%  Run from anywhere with the toolbox on the path.

sigOnset  = 15;   sigPitch = 6;
perOnset  = 4000; perPitch = 1200;
nReps     = 3;
TIME_CAP  = 60;

modes = { ...
    struct('rel', 0, 'per', 1, 'name', 'abs+per', ...
           'cells', {{[20, 4, 2]; [60, 4, 2]; [120, 4, 2]; [20, 5, 3]; [60, 5, 3]}}), ...
    struct('rel', 0, 'per', 0, 'name', 'abs+nonper', ...
           'cells', {{[20, 4, 2]; [60, 4, 2]; [120, 4, 2]; [20, 5, 3]; [60, 5, 3]}}), ...
    struct('rel', 1, 'per', 1, 'name', 'rel+per', ...
           'cells', {{[10, 4, 2]; [20, 4, 2]; [40, 4, 2]}}), ...
    struct('rel', 1, 'per', 0, 'name', 'rel+nonper', ...
           'cells', {{[10, 4, 2]; [20, 4, 2]; [40, 4, 2]}}) ...
    };

fprintf('bench_ma_dispatch: A = 2 (scalar onset r=1 + K-pitch chord)\n');

for mi = 1:numel(modes)
    md = modes{mi};
    fprintf('\n--- %s ---\n', md.name);
    fprintf('%14s %10s %10s %10s %8s %10s %8s\n', ...
        'N / K / r', 't_bul(s)', 't_mob(s)', 't_auto(s)', ...
        'auto~', '|ds|', 'ovh');

    skipMob = false;
    for ci = 1:numel(md.cells)
        cfg = md.cells{ci};
        N = cfg(1); K = cfg(2); r = cfg(3);

        rng(10 + ci);
        pitchesX = rand(K, N) * 1200;
        onsetsX  = (0:N-1) * 250 + randn(1, N) * 10;
        rng(20 + ci);
        pitchesY = rand(K, N) * 1200;
        onsetsY  = (0:N-1) * 250 + randn(1, N) * 10;

        dx = buildExpTens({onsetsX; pitchesX}, {[]; []}, ...
            [sigOnset, sigPitch], [1, r], [false, logical(md.rel)], ...
            [false, logical(md.per)], [perOnset, perPitch], ...
            'verbose', false);
        dy = buildExpTens({onsetsY; pitchesY}, {[]; []}, ...
            [sigOnset, sigPitch], [1, r], [false, logical(md.rel)], ...
            [false, logical(md.per)], [perOnset, perPitch], ...
            'verbose', false);

        % --- forced bulger ---
        tStart = tic;
        sBul = cosSimExpTens(dx, dy, 'method', 'bulger', 'verbose', false);
        tWarm = toc(tStart);
        if tWarm > TIME_CAP
            tBul = tWarm;
        else
            tB = zeros(1, nReps);
            for k = 1:nReps
                tStart = tic;
                sBul = cosSimExpTens(dx, dy, ...
                    'method', 'bulger', 'verbose', false);
                tB(k) = toc(tStart);
            end
            tBul = median(tB);
        end

        % --- forced mobius (capped, try/catch) ---
        if skipMob
            tMob = NaN; sMob = NaN;
        else
            try
                tStart = tic;
                sMob = cosSimExpTens(dx, dy, ...
                    'method', 'mobius', 'verbose', false);
                tWarm = toc(tStart);
                if tWarm > TIME_CAP
                    tMob = tWarm;
                    skipMob = true;
                else
                    tM = zeros(1, nReps);
                    for k = 1:nReps
                        tStart = tic;
                        sMob = cosSimExpTens(dx, dy, ...
                            'method', 'mobius', 'verbose', false);
                        tM(k) = toc(tStart);
                    end
                    tMob = median(tM);
                end
            catch err
                fprintf('  mobius unavailable: %s\n', err.message);
                tMob = NaN; sMob = NaN;
                skipMob = true;
            end
        end

        % --- auto ---
        cosSimExpTens(dx, dy, 'verbose', false);   % warm
        tStart = tic;
        cosSimExpTens(dx, dy, 'verbose', false);
        tAuto = toc(tStart);

        if isnan(tMob)
            autoPick = '?';
        elseif abs(tAuto - tBul) <= abs(tAuto - tMob)
            autoPick = 'bulger';
        else
            autoPick = 'mobius';
        end
        ds  = abs(sMob - sBul);
        ovh = tAuto - min(tBul, tMob);

        fprintf('%4d /%2d /%2d %12.3f %10.3f %10.3f %8s %10.1e %8.3f\n', ...
            N, K, r, tBul, tMob, tAuto, autoPick, ds, ovh);
    end
end

fprintf(['\nPass criteria: auto~ matches the smaller of t_bul/t_mob in\n' ...
         'every row (few-ms near-ties excepted); |ds| <= ~1e-9; ovh ~ 0.\n' ...
         'At these small-K sizes auto is expected to choose bulger in\n' ...
         'the rel modes and mobius in the abs modes at moderate N.\n' ...
         'Per-r medians from the abs rows calibrate the MATLAB entries\n' ...
         'in internal.selectMaInnerProductMethod.\n']);
