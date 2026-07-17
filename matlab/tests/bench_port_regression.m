function bench_port_regression
%BENCH_PORT_REGRESSION  Speed baseline for the MATLAB SA-merge port.
%
%   Times the exact code paths the port touches --- the truncation and
%   normalisation call sites in evalExpTens, nestedContract, and the
%   orbit cosine --- at configurations where MATLAB currently beats the
%   Python version. Capture this table ONCE on the known-good baseline
%   (the tree that passes test_mpt with 1145/1145), then re-run it after
%   each port increment. Any row whose median time regresses beyond
%   run-to-run noise is a stop-and-investigate signal, treated exactly
%   like a 1e-12 parity break: it means the port overwrote a MATLAB
%   optimisation rather than adapting to it.
%
%   Prints CSV lines: stage,bench,r,K,extra,t_median_ms
%   'stage' is 'baseline' here; after a port increment, re-run and label
%   the capture with the increment name so the two tables diff cleanly.
%
%   The configs are deliberately the heavy ones:
%     * eval_rel_dim3   --- r=4 relative non-periodic, dim=3, ~1.77M
%       query points: the single configuration that dominates
%       demo_expTensorPlots and is ~4x slower in Python. Exercises the
%       relative-mode centres/truncation eval path.
%     * eval_abs_dim4   --- r=4 absolute, dim=4: the absolute centres
%       path, a normalisation call site.
%     * nested_contract --- rel-periodic nested IP: the nestedContract
%       truncation floor site (the inf-handling fix lands here).
%     * cossim_orbit    --- rel-periodic orbit cosine across r=2..4:
%       the factored path whose dispatch calibration must be preserved.
%
%   Run from the matlab/ directory (with the toolbox on the path):
%     >> bench_port_regression
%   then copy the CSV block for comparison.

    N_REPS   = 5;
    N_WARMUP = 2;

    fprintf('stage,bench,r,K,extra,t_median_ms\n');

    % --- eval: relative, dim = 3 (the Python bottleneck) ---
    t = timed(@() evalRelDim3(), N_REPS, N_WARMUP) * 1000;
    fprintf('baseline,eval_rel_dim3,4,6,res121,%.4f\n', t);

    % --- eval: absolute, dim = 4 ---
    t = timed(@() evalAbsDim4(), N_REPS, N_WARMUP) * 1000;
    fprintf('baseline,eval_abs_dim4,4,6,res41,%.4f\n', t);

    % --- nested contraction: rel-periodic IP (truncation floor site) ---
    % These per-call workloads run at 1-10 ms, below MATLAB's reliable
    % timing floor, so each is repeated innerReps(r) times per timed unit
    % (~150 ms) and the median is divided back down to per-call ms. This
    % makes the row a trustworthy regression tripwire rather than noise.
    for r = [2, 3, 4]
        ni = innerReps(r);
        t = timed(@() nestedContractRelPer(r, ni), N_REPS, N_WARMUP) * 1000 / ni;
        fprintf('baseline,nested_contract_relper,%d,8,-,%.4f\n', r, t);
    end

    % --- orbit cosine: rel-periodic across r (dispatch calibration) ---
    for r = [2, 3, 4]
        ni = innerReps(r);
        t = timed(@() cosSimOrbitRelPer(r, ni), N_REPS, N_WARMUP) * 1000 / ni;
        fprintf('baseline,cossim_orbit_relper,%d,8,-,%.4f\n', r, t);
    end
end


% =========================================================================
%  Workload builders (each returns nothing; called for its wall time)
% =========================================================================

function evalRelDim3()
    rng(0, 'twister');
    p = sort(1200 * rand(1, 6));
    dens = buildExpTens(p, [], 60, 4, 1, 0, 0, 'verbose', false);   % r=4 rel non-per -> dim 3
    ax = linspace(0, 1200, 121);
    [G1, G2, G3] = ndgrid(ax, ax, ax);
    X = [G1(:), G2(:), G3(:)].';                  % 3 x 1.77M
    evalExpTens(dens, X, 'gaussian', 'verbose', false);
end


function evalAbsDim4()
    rng(1, 'twister');
    p = sort(1200 * rand(1, 6));
    dens = buildExpTens(p, [], 60, 4, 0, 0, 0, 'verbose', false);   % r=4 abs -> dim 4
    ax = linspace(0, 1200, 41);
    [G1, G2, G3, G4] = ndgrid(ax, ax, ax, ax);
    X = [G1(:), G2(:), G3(:), G4(:)].';           % 4 x 2.83M
    evalExpTens(dens, X, 'gaussian', 'verbose', false);
end


function nestedContractRelPer(r, nInner)
    % Build the two collections once, outside the timed inner loop, so the
    % measurement is dominated by the repeated cosine call rather than the
    % (cheap) rand/sort/mod setup. The inner loop runs the call nInner
    % times so the timed unit is ~150 ms even at r = 2 (~1 ms/call),
    % averaging out MATLAB's sub-10-ms timing noise (JIT, cache, frequency
    % scaling). The caller divides the measured time by nInner to report
    % per-call ms, keeping the table comparable to single-call baselines.
    rng(2 + r, 'twister');
    K = 8;
    pA = sort(1200 * rand(1, K));
    pB = mod(pA + 37.3, 1200);
    % rel-periodic cosine routes through the nested/orbit contraction;
    % force the pairwise path so the nestedContract truncation site is hit.
    for i = 1:nInner
        cosSimExpTens(pA, [], pB, [], 30, r, 1, 1, 1200, ...
                      'method', 'bulger', 'verbose', false);
    end
end


function cosSimOrbitRelPer(r, nInner)
    rng(20 + r, 'twister');
    K = 8;
    pA = sort(1200 * rand(1, K));
    pB = mod(pA + 41.7, 1200);
    % default dispatch: exercises selectMaInnerProductMethod calibration.
    for i = 1:nInner
        cosSimExpTens(pA, [], pB, [], 30, r, 1, 1, 1200, 'verbose', false);
    end
end


function n = innerReps(r)
    %INNERREPS  Repetitions to bring a sub-10-ms workload up to ~150 ms.
    %   Sized from observed per-call times (~1 ms at r=2, ~4 ms at r=3,
    %   ~10 ms at r=4). Only the affected nested/orbit rows use this; the
    %   seconds-scale eval rows are already stable and are timed as single
    %   calls.
    switch r
        case 2
            n = 150;
        case 3
            n = 40;
        otherwise
            n = 15;
    end
end


% =========================================================================
%  Timing helper (matches bench_orbit_xlang idiom)
% =========================================================================

function t = timed(fn, nReps, nWarmup)
    for w = 1:nWarmup
        fn();
    end
    times = zeros(nReps, 1);
    for i = 1:nReps
        t0 = tic;
        fn();
        times(i) = toc(t0);
    end
    t = median(times);
end
