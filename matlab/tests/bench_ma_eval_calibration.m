%% bench_ma_eval_calibration.m
%  Timing-grid harness for calibrating the multi-attribute EVAL cost
%  model (internal.selectMaEval, routing evalExpTens between the
%  joint-centres accumulator and the factored Möbius evaluator).
%
%  WHY THIS EXISTS
%  ---------------
%  The cost model estimates each path's per-call time as a per-call
%  setup term plus per-query work scaled by nQ, with a small group of
%  calibration constants absorbing per-language constant factors (BLAS,
%  JIT, column-major layout, copy-on-write --- all differ from NumPy).
%  The constants are settled once per language by measurement on the
%  target machine. This harness MEASURES the grid; it does not fit the
%  constants. Run it warm (see below), then send the printed CSV block
%  back so the constants can be fitted and validated against these exact
%  timings before shipping.
%
%  HOW TO RUN
%  ----------
%  With the toolbox on the path, from anywhere:
%
%      >> bench_ma_eval_calibration
%
%  Run it TWICE in the same session and read the SECOND run. MATLAB pays
%  one-time costs on first use of each path (function compilation,
%  +mobius package resolution, loading the orbit / contraction tables
%  into the in-memory cache). The harness warms both paths before timing,
%  but a second whole-script run is the safest way to reach the
%  steady-state timings a real user's session sees after first use. Do
%  NOT `clear all` between the two runs.
%
%  WHAT TO SEND BACK
%  -----------------
%  Everything between the BEGIN_CSV and END_CSV markers (inclusive of the
%  header row). That block alone is sufficient to fit the constants.
%
%  Single-attribute cells only: the multi-attribute behaviour is the
%  product-vs-sum contrast, structurally Möbius-dominated and insensitive
%  to the constants; the crossover that the constants must place lives in
%  the single-attribute grid, especially the relative modes where the
%  u-grid quadrature makes centres win far past the absolute crossover.

fprintf('\n=== MA eval cost-model calibration grid ===\n');
fprintf('Run twice in one session (no clear all); read the SECOND run.\n\n');

rng(0, 'twister');
nReps = 3;

% Centres is skipped (not timed) above this joint tuple count: there it
% is decisively the wrong pick (the model must choose Möbius) and timing
% it wastes seconds. Such cells still appear in the CSV with cen_ms = -1.
JOINT_SKIP = 3e5;

sigma = 15.0;   % cents; small enough that the relative u-grid genuinely costs
spanNP = 3600.0;   % non-periodic value span
periodP = 1200.0;  % periodic period (one octave)

% ---- Global warm-up: exercise every path once, off the clock. ----
warmAbs = buildExpTens(spanNP * rand(8, 1), 0.2 + 0.8 * rand(8, 1), ...
    sigma, 3, false, false, 0, 'verbose', false);
evalExpTens(warmAbs, spanNP * rand(3, 50), 'method', 'centres', 'verbose', false);
evalExpTens(warmAbs, spanNP * rand(3, 50), 'method', 'mobius',  'verbose', false);
warmRel = buildExpTens(spanNP * rand(8, 1), 0.2 + 0.8 * rand(8, 1), ...
    sigma, 3, true, false, 0, 'verbose', false);
evalExpTens(warmRel, spanNP * rand(2, 50), 'method', 'centres', 'verbose', false);
evalExpTens(warmRel, spanNP * rand(2, 50), 'method', 'mobius',  'verbose', false);
warmPer = buildExpTens(periodP * rand(8, 1), 0.2 + 0.8 * rand(8, 1), ...
    sigma, 3, true, true, periodP, 'verbose', false);
evalExpTens(warmPer, periodP * rand(2, 50), 'method', 'mobius',  'verbose', false);

% ---- Grid: (isRel, isPer) x r x K x nQ, single attribute. ----
relPer = {[false false], [true false], [false true], [true true]};
rVals  = [2 3 4];
KVals  = [6 12 24 48];
nQVals = [1 200];

rows = {};  % accumulate CSV rows

fprintf('%-4s %-4s %-3s %-4s %-5s %10s %10s %10s %-8s\n', ...
    'rel', 'per', 'r', 'K', 'nQ', 'joint', 'cen_ms', 'mob_ms', 'faster');
fprintf('%s\n', repmat('-', 1, 66));

for rpi = 1:numel(relPer)
    rp = relPer{rpi};
    isRel = rp(1); isPer = rp(2);
    P = periodP * double(isPer);
    span = periodP * double(isPer) + spanNP * double(~isPer);
    for r = rVals
        for K = KVals
            joint = factorial(r) * nchoosek(K, r);
            p = span * rand(K, 1);
            w = 0.2 + 0.8 * rand(K, 1);
            dens = buildExpTens(p, w, sigma, r, isRel, isPer, P, ...
                'verbose', false);
            d = dens.dim;   % r for abs, r-1 for rel
            for nQ = nQVals
                xq = span * rand(d, nQ);

                if joint <= JOINT_SKIP
                    tCen = timeMethod(@() evalExpTens(dens, xq, ...
                        'method', 'centres', 'verbose', false), nReps);
                else
                    tCen = -1;   % skipped: decisively a Möbius cell
                end
                tMob = timeMethod(@() evalExpTens(dens, xq, ...
                    'method', 'mobius', 'verbose', false), nReps);

                if tCen < 0
                    faster = 'mobius';
                elseif tCen < tMob
                    faster = 'centres';
                else
                    faster = 'mobius';
                end

                if tCen < 0, cenMs = -1; else, cenMs = tCen * 1e3; end
                mobMs = tMob * 1e3;

                fprintf('%-4d %-4d %-3d %-4d %-5d %10d %10.3f %10.3f %-8s\n', ...
                    isRel, isPer, r, K, nQ, joint, cenMs, mobMs, faster);
                rows{end + 1} = sprintf('%d,%d,%d,%d,%d,%d,%.4f,%.4f', ...
                    isRel, isPer, r, K, nQ, joint, cenMs, mobMs); %#ok<AGROW>
            end
        end
    end
end

% ---- Parseable block to send back ----
fprintf('\nBEGIN_CSV\n');
fprintf('rel,per,r,K,nQ,joint,cen_ms,mob_ms\n');
for i = 1:numel(rows)
    fprintf('%s\n', rows{i});
end
fprintf('END_CSV\n');
fprintf(['\ncen_ms = -1 marks a cell where centres was not timed ' ...
         '(joint > %g, decisively Möbius).\n'], JOINT_SKIP);
fprintf('Reminder: this is the SECOND-run output that matters.\n\n');

% ---- helper ----
function t = timeMethod(fn, nReps)
    fn();                       % warm this specific shape/method
    ts = zeros(1, nReps);
    for i = 1:nReps
        tic; fn(); ts(i) = toc;
    end
    t = min(ts);
end
