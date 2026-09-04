%% bench_spectral_ip_gate.m
%  Timing-grid harness for the spectral IP gate constant
%  (COST_C in mobius.spectralRelInnerMatrix).
%
%  WHY THIS EXISTS
%  ---------------
%  The gate decides whether the spectral (Fourier) branch of the
%  relative-mode inner product is worth its mode grid against the
%  translation grid's K^2 per event pair. It declines when
%
%      gridSize > COST_C * K^2 * nPairs .
%
%  COST_C absorbs the per-op cost ratio between the two paths, which is
%  a per-language, per-machine quantity: BLAS, JIT, column-major layout
%  and copy-on-write all differ from NumPy, so the constant is fitted on
%  MATLAB timings rather than transcribed from Python.
%
%  This is now purely a SPEED question. Before the full-image work the
%  two routes computed different measures, so a gate that fired on
%  different shapes in the two languages changed the answer; the routes
%  now agree in value (see test_spectral_ip_branch.m), so a mis-set
%  COST_C costs time only. That is why this constant may be recalibrated
%  per machine while the centres/grid gate constants stay matched.
%
%  This harness MEASURES the grid; it does not fit the constant. Run it
%  warm, then send the printed CSV block back so COST_C can be fitted
%  and validated against these exact timings before shipping.
%
%  HOW TO RUN
%  ----------
%  With the toolbox on the path, from anywhere:
%
%      >> bench_spectral_ip_gate
%
%  Run it on an otherwise idle machine. The first call in each cell is
%  discarded to absorb JIT; the reported figure is the median of the
%  rest. Expect a few minutes.
%
%  WHAT IT PRINTS
%  --------------
%  One CSV row per (r, K, sigma/period, isPer) cell:
%
%      r, K, N, sigmaOverP, isPer, gridSize, msSpectral, msGrid, ratio
%
%  ratio = msGrid / msSpectral. The branch is worth taking where
%  ratio > 1. The spectral arm is timed with the cost gate bypassed
%  (internal.spectralIpForce), so every cell measures the branch itself
%  against the grid; the constant is then fitted by routing regret over
%  these timings. A cell whose two arms agree to within noise is one the
%  gate can route either way at no cost.
%
%  This grid stops at K = 30. bench_spectral_ip_gate_ext covers larger K
%  at the shape where the shipped constant was found to misroute.

fprintf('\n=== bench_spectral_ip_gate ===\n');
fprintf('Measuring the spectral branch against the translation grid.\n');
fprintf('Send the CSV block below back for fitting.\n\n');

% The routing announcements ("cos_sim_exp_tens: chose ... path") are
% gated by mptDefaults('showHints'), not by the per-call verbose flag,
% so this is the switch that keeps them out of the CSV. Restored via
% onCleanup so an early exit or Ctrl+C cannot leave the session muted,
% and again explicitly at the end because an onCleanup object created
% by a top-level script survives until its variable is cleared.
bsg_prevShowHints = mptDefaults('showHints', false);
bsg_hintsCleanup  = onCleanup(@() mptDefaults(bsg_prevShowHints)); %#ok<NASGU>

bsg_P      = 1200;
bsg_rs     = [2, 3, 4];
bsg_Ks     = [4, 8, 16, 30];
bsg_Ns     = [1, 2, 4, 8, 16];
bsg_sops   = [0.002, 0.005, 0.0125, 0.05, 0.20];
bsg_isPers = [true, false];
bsg_reps   = 5;

fprintf('CSV_BEGIN\n');
fprintf('r,K,N,sigmaOverP,isPer,gridSize,msSpectral,msGrid,ratio\n');

for bsg_isPer = bsg_isPers
    for bsg_r = bsg_rs
        for bsg_K = bsg_Ks
            if bsg_K < bsg_r
                continue;
            end
            for bsg_N = bsg_Ns
            for bsg_sop = bsg_sops
                bsg_sigma = bsg_sop * bsg_P;

                rng(1000 * bsg_r + 10 * bsg_K + bsg_N);
                % N events per side: the branch's Gram term scales as
                % N_x*N_y while the grid's contraction scales as
                % N_u*N_x*N_y*K^2, so N is the axis the old gate handled
                % worst and the one this sweep exists to cover.
                bsg_p = sort(rand(bsg_K, bsg_N) * bsg_P, 1);
                bsg_q = sort(rand(bsg_K, bsg_N) * bsg_P, 1);
                bsg_w = ones(bsg_K, bsg_N);

                if bsg_isPer
                    bsg_periodArg = bsg_P;
                else
                    bsg_periodArg = 0;
                end

                % Mode-grid size, mirroring the branch's own sizing.
                if bsg_isPer
                    bsg_L = bsg_P;
                else
                    bsg_L = 2 * bsg_P + 2 * (8.6 + 2) * bsg_sigma;
                end
                bsg_M = ceil(8.6 / sqrt(2) * bsg_L ...
                             / (2 * pi * bsg_sigma)) + 2;
                bsg_gridSize = (2 * bsg_M + 1)^(bsg_r - 1);

                % When the mode grid exceeds the memory guard the branch
                % always declines, so both toggle states run the grid and
                % the ratio is 1 by construction. Timing that says nothing
                % about the cost gate and these are the slowest cells, so
                % record the decline without paying for a multi-second
                % grid contraction (matches the Python harness).
                if bsg_gridSize > 4e6
                    fprintf('%d,%d,%d,%.4f,%d,%d,nan,nan,nan\n', ...
                            bsg_r, bsg_K, bsg_N, bsg_sop, bsg_isPer, ...
                            bsg_gridSize);
                    continue;
                end

                % Time the per-attribute inner matrix directly: it is the
                % object the branch replaces, and it takes N events per
                % side without going through a density build.
                bsg_call = @() mobius.relInnerBatched(bsg_p, bsg_w, ...
                    bsg_q, bsg_w, bsg_sigma, bsg_r, bsg_isPer, ...
                    bsg_periodArg, 'truncationSigmas', Inf);

                % Cost-aware repetition: one timed call gauges the cost,
                % then cheap cells get the full bsg_reps for a stable
                % median while cells already costing seconds are timed
                % fewer times (the extended low-sigma/P grid has a few
                % very heavy ones and their magnitude dwarfs jitter).
                % The spectral arm must BYPASS the cost gate, not merely
                % enable the branch: spectralIpEnabled(true) leaves the
                % gate in force, so on every cell the shipped constant
                % declines both arms would run the grid and the ratio
                % would read 1.0 by construction -- which is what an
                % earlier run of this harness reported on all 80 such
                % cells. spectralIpForce bypasses the comparison only,
                % never the MAX_POINTS memory guard.
                internal.spectralIpEnabled(true);
                internal.spectralIpForce(true);
                bsg_t0 = tic; bsg_call(); bsg_first = toc(bsg_t0);
                if bsg_first > 2.0
                    bsg_nrep = 1;
                elseif bsg_first > 0.2
                    bsg_nrep = 2;
                else
                    bsg_nrep = bsg_reps;
                end

                % --- spectral branch ---
                bsg_tS = zeros(1, bsg_nrep);
                for bsg_i = 1:bsg_nrep
                    bsg_t0 = tic;  bsg_call();  bsg_tS(bsg_i) = toc(bsg_t0);
                end

                % --- translation grid ---
                internal.spectralIpForce(false);
                internal.spectralIpEnabled(false);
                bsg_call();
                bsg_tG = zeros(1, bsg_nrep);
                for bsg_i = 1:bsg_nrep
                    bsg_t0 = tic;  bsg_call();  bsg_tG(bsg_i) = toc(bsg_t0);
                end
                internal.spectralIpEnabled(true);

                bsg_msS = 1e3 * median(bsg_tS);
                bsg_msG = 1e3 * median(bsg_tG);
                fprintf('%d,%d,%d,%.4f,%d,%d,%.4f,%.4f,%.4f\n', ...
                        bsg_r, bsg_K, bsg_N, bsg_sop, bsg_isPer, ...
                        bsg_gridSize, bsg_msS, bsg_msG, ...
                        bsg_msG / max(bsg_msS, eps));
            end
            end
        end
    end
end

fprintf('CSV_END\n\n');

mptDefaults(bsg_prevShowHints);
clear bsg_hintsCleanup;
fprintf(['Done. The branch pays off where ratio > 1; COST_C should ' ...
         'place\nthe decline boundary on that contour. Send the block ' ...
         'between\nCSV_BEGIN and CSV_END.\n\n']);
