%% bench_spectral_ip_gate_ext.m
%  Extension of bench_spectral_ip_gate to the regime the K <= 30 grid
%  cannot see: larger K at a fixed, musically ordinary shape.
%
%  On the K <= 30 grid the routing regret of the shipped gate form is
%  flat in COST_C from ~250 upward (geometric mean 1.008 at 282, 1.013
%  at any C above ~400, 1.016 with the gate removed), because every cell
%  there with gridSize above ~300 K^2 nPairs is one where the two routes
%  are within a factor of two of each other. The grid therefore cannot
%  place the constant; it only bounds it from below. The one cell that
%  can is the shape the audit found: r = 3, sigma = 10 cents, positions
%  over three octaves, where the grid route measured ~980 ms at K = 80
%  against ~105 ms for the branch at K = 140. At that shape gridSize is
%  ~2.08e6 and nPairs = 1, so the branch is taken only when
%  COST_C >= gridSize / K^2: 325 at K = 80, 901 at K = 48, 1298 at
%  K = 40, 2028 at K = 32.
%
%  This sweep measures that shape across K, for N = 1 and N = 2, so the
%  constant can be set where the crossover actually is. Same call, same
%  toggling and same CSV columns as bench_spectral_ip_gate; positions
%  span 3 * bsx_P rather than one period, and the mode is non-periodic.
%
%  HOW TO RUN
%  ----------
%      >> bench_spectral_ip_gate_ext
%
%  Send the CSV block back. Expect a couple of minutes.

fprintf('\n=== bench_spectral_ip_gate_ext ===\n');

bsx_prevShowHints = mptDefaults('showHints', false);
bsx_hintsCleanup  = onCleanup(@() mptDefaults(bsx_prevShowHints)); %#ok<NASGU>

bsx_P      = 1200;
bsx_span   = 3 * bsx_P;
bsx_sigma  = 10;
bsx_r      = 3;
bsx_Ks     = [24, 32, 40, 48, 64, 80, 100, 140];
bsx_Ns     = [1, 2];
bsx_reps   = 5;

fprintf('CSV_BEGIN\n');
fprintf('r,K,N,sigmaOverP,isPer,gridSize,msSpectral,msGrid,ratio\n');

for bsx_N = bsx_Ns
    for bsx_K = bsx_Ks
        rng(1000 * bsx_r + 10 * bsx_K + bsx_N);
        bsx_p = sort(rand(bsx_K, bsx_N) * bsx_span, 1);
        bsx_q = sort(rand(bsx_K, bsx_N) * bsx_span, 1);
        bsx_w = ones(bsx_K, bsx_N);

        % Mode-grid size, mirroring the branch's own sizing for the
        % non-periodic mode (span plus the truncation padding).
        bsx_L = 2 * bsx_span + 2 * (8.6 + 2) * bsx_sigma;
        bsx_M = ceil(8.6 / sqrt(2) * bsx_L / (2 * pi * bsx_sigma)) + 2;
        bsx_gridSize = (2 * bsx_M + 1)^(bsx_r - 1);
        if bsx_gridSize > 4e6
            fprintf('%d,%d,%d,%.4f,%d,%d,nan,nan,nan\n', ...
                    bsx_r, bsx_K, bsx_N, bsx_sigma / bsx_P, 0, bsx_gridSize);
            continue;
        end

        bsx_call = @() mobius.relInnerBatched(bsx_p, bsx_w, ...
            bsx_q, bsx_w, bsx_sigma, bsx_r, false, 0, ...
            'truncationSigmas', Inf);

        bsx_ms = zeros(1, 2);
        for bsx_toggle = [true, false]
            % Bypass the cost gate on the spectral arm (see the note in
            % bench_spectral_ip_gate): enabling alone leaves it in force.
            internal.spectralIpEnabled(bsx_toggle);
            internal.spectralIpForce(bsx_toggle);
            bsx_call();                                  % discard: JIT
            bsx_t = zeros(1, bsx_reps);
            for bsx_i = 1:bsx_reps
                bsx_t0 = tic; bsx_call(); bsx_t(bsx_i) = toc(bsx_t0);
            end
            bsx_ms(1 + ~bsx_toggle) = 1e3 * median(bsx_t);
        end
        internal.spectralIpForce(false);
        internal.spectralIpEnabled(true);

        fprintf('%d,%d,%d,%.4f,%d,%d,%.4f,%.4f,%.4f\n', ...
                bsx_r, bsx_K, bsx_N, bsx_sigma / bsx_P, 0, bsx_gridSize, ...
                bsx_ms(1), bsx_ms(2), bsx_ms(2) / bsx_ms(1));
    end
end
fprintf('CSV_END\n');

mptDefaults(bsx_prevShowHints);
clear bsx_hintsCleanup
