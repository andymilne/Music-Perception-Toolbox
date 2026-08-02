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
%  header row). That block alone is sufficient to fit the constants. The
%  row format now carries sigma and span alongside the shape, so earlier
%  CSV blocks (which do not) are not interchangeable with this one.
%
%  TWO SECTIONS
%  ------------
%  Section A sweeps shape (mode, r, K, nQ) at one reference geometry,
%  sigma = 15 cents over a 3600-cent span. It is unchanged, so its rows
%  remain comparable with earlier runs.
%
%  Section B sweeps the geometry --- sigma and, non-periodically, the
%  value span --- over a reduced shape set. The centres per-query cost
%  is culled by a factor set by sigma against the source spread, so a
%  grid at one sigma and one span cannot constrain that factor at all:
%  it fixes the very quantity the term is a function of. Section B
%  supplies the variation. Periodically the span is the period, so only
%  sigma varies there, and sigma/P reaches 0.05 at the widest kernel,
%  which is past the measure threshold --- the arms are still timed,
%  since both are forced by name.
%
%  Section C traces the relative-mode node count alone: one shape at a
%  time, sigma walked over six values, which is what separates the
%  node-count exponent from everything else moving with sigma.
%
%  Section D exercises the spectral strategy inside the Mobius relative
%  evaluator, whose per-mode constants are per tuple order. The branch
%  engages only above its thresholds and only where the mode grid fits
%  under the memory guard, so it needs geometries picked for it.
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

% Centres is not timed where it would cost too much to be worth the wall
% time. Two bounds, whichever bites first: the joint tuple count, beyond
% which centres is decisively the wrong pick; and the work the centres
% arm actually does, which is the joint count times the query count.
% Measured on the spectral-branch shapes, one centres call runs 35 ms at
% 1e5 units of work, 1.6 s at 5e6 and 10.6 s at 4e7, so the work bound is
% what keeps the run finite where the joint bound alone would not. Such
% cells appear in the CSV with cen_ms = -1.
JOINT_SKIP = 3e5;
WORK_SKIP  = 4e6;

sigmaRef = 15.0;   % cents; small enough that the relative u-grid genuinely costs
spanRef  = 3600.0; % non-periodic value span
periodP  = 1200.0; % periodic period (one octave)

% Section B geometry axes. Sigma spans a factor of twelve, the span a
% factor of eight, so the cull factor sigma/spread ranges over roughly
% two orders of magnitude.
sigmaVals = [5.0 15.0 60.0];
spanVals  = [1200.0 3600.0 9600.0];
% Reduced shape set for Section B: the geometry terms are identified by
% varying sigma and span, not by adding shapes, and the full shape grid
% at every geometry would run for a long time on the largest cells.
rValsB = [3 4];
KValsB = [12 24];

% Section C: the relative-mode node-count trace. The Mobius relative
% evaluator integrates over a grid whose node count is set by the period
% (or the source span) over sigma, and the cost model prices its
% per-query work linearly in that count. Holding the shape fixed and
% walking sigma widely is what separates the node-count exponent from
% everything else that moves with sigma; three sigma values cannot do
% it, which is why Section B leaves the exponent unidentified.
% Chosen to avoid Section B's 5, 15 and 60, so no cell is measured twice.
sigmaValsC = [3.0 7.0 10.0 20.0 40.0 80.0];
rValsC     = [3 4];
KValsC     = [12 24];

% Section D: the spectral (Fourier) strategy inside the Mobius relative
% evaluator. Its per-mode constants are per tuple order, and the branch
% engages only above its query and value-count thresholds and only where
% the mode grid fits under the memory guard --- which at r = 4 needs
% sigma at least P/78 periodically, or a short span with a wide kernel
% otherwise. Sections A and B satisfy that in one cell per periodicity
% at r = 4, so the r = 4 constant rested on two measurements; these
% geometries are chosen so every tuple order gets a proper sample.
sigmaValsDPer = [20.0 40.0 80.0];
geomValsDNP   = [40.0 600.0; 80.0 600.0; 60.0 1200.0; 120.0 1200.0];
rValsD  = [2 3 4];
KValsD  = [16 24 48];
nQValsD = [100 400];

% ---- Global warm-up: exercise every path once, off the clock. ----
warmAbs = buildExpTens(spanRef * rand(8, 1), 0.2 + 0.8 * rand(8, 1), ...
    sigmaRef, 3, false, false, 0, 'verbose', false);
evalExpTens(warmAbs, spanRef * rand(3, 50), 'method', 'centres', 'verbose', false);
evalExpTens(warmAbs, spanRef * rand(3, 50), 'method', 'mobius',  'verbose', false);
warmRel = buildExpTens(spanRef * rand(8, 1), 0.2 + 0.8 * rand(8, 1), ...
    sigmaRef, 3, true, false, 0, 'verbose', false);
evalExpTens(warmRel, spanRef * rand(2, 50), 'method', 'centres', 'verbose', false);
evalExpTens(warmRel, spanRef * rand(2, 50), 'method', 'mobius',  'verbose', false);
warmPer = buildExpTens(periodP * rand(8, 1), 0.2 + 0.8 * rand(8, 1), ...
    sigmaRef, 3, true, true, periodP, 'verbose', false);
evalExpTens(warmPer, periodP * rand(2, 50), 'method', 'mobius',  'verbose', false);

% ---- Grid: (isRel, isPer) x r x K x nQ, single attribute. ----
relPer = {[false false], [true false], [false true], [true true]};
rVals  = [2 3 4];
KVals  = [6 12 24 48];
nQVals = [1 200];

rows = {};  % accumulate CSV rows

fprintf('\n--- Section A: shape sweep at sigma = %g over a span of %g ---\n', ...
        sigmaRef, spanRef);
fprintf('%-4s %-4s %-3s %-4s %-5s %10s %10s %10s %-8s\n', ...
    'rel', 'per', 'r', 'K', 'nQ', 'joint', 'cen_ms', 'mob_ms', 'faster');
fprintf('%s\n', repmat('-', 1, 66));

for rpi = 1:numel(relPer)
    rp = relPer{rpi};
    for r = rVals
        for K = KVals
            rows = sweepCell(rows, rp(1), rp(2), r, K, nQVals, ...
                sigmaRef, spanRef, periodP, JOINT_SKIP, nReps, WORK_SKIP);
        end
    end
end

fprintf('\n--- Section B: geometry sweep (sigma, span) ---\n');
fprintf('%-4s %-4s %-3s %-4s %-5s %10s %10s %10s %-8s\n', ...
    'rel', 'per', 'r', 'K', 'nQ', 'joint', 'cen_ms', 'mob_ms', 'faster');
fprintf('%s\n', repmat('-', 1, 66));

for rpi = 1:numel(relPer)
    rp = relPer{rpi};
    isRelB = rp(1); isPerB = rp(2);
    for sg = sigmaVals
        % Periodically the span is the period, so the span axis is
        % vacuous there and only sigma varies.
        if isPerB
            spansHere = periodP;
        else
            spansHere = spanVals;
        end
        for sp = spansHere
            if isPerB && sg == sigmaRef
                continue;   % already measured in Section A
            end
            if ~isPerB && sg == sigmaRef && sp == spanRef
                continue;   % already measured in Section A
            end
            for r = rValsB
                for K = KValsB
                    rows = sweepCell(rows, isRelB, isPerB, r, K, ...
                        nQVals, sg, sp, periodP, JOINT_SKIP, nReps, WORK_SKIP);
                end
            end
        end
    end
end

fprintf('\n--- Section C: node-count trace (relative modes) ---\n');
fprintf('%-4s %-4s %-3s %-4s %-5s %10s %10s %10s %-8s\n', ...
    'rel', 'per', 'r', 'K', 'nQ', 'joint', 'cen_ms', 'mob_ms', 'faster');
fprintf('%s\n', repmat('-', 1, 66));

for isPerC = [true false]
    for sg = sigmaValsC
        if isPerC
            sp = periodP;
        else
            sp = spanRef;
        end
        for r = rValsC
            for K = KValsC
                rows = sweepCell(rows, true, isPerC, r, K, 200, ...
                    sg, sp, periodP, JOINT_SKIP, nReps, WORK_SKIP);
            end
        end
    end
end

fprintf('\n--- Section D: spectral branch (relative modes) ---\n');
fprintf('%-4s %-4s %-3s %-4s %-5s %10s %10s %10s %-8s\n', ...
    'rel', 'per', 'r', 'K', 'nQ', 'joint', 'cen_ms', 'mob_ms', 'faster');
fprintf('%s\n', repmat('-', 1, 66));

for gi = 1:numel(sigmaValsDPer)
    for r = rValsD
        for K = KValsD
            rows = sweepCell(rows, true, true, r, K, nQValsD, ...
                sigmaValsDPer(gi), periodP, periodP, JOINT_SKIP, nReps, ...
                WORK_SKIP);
        end
    end
end
for gi = 1:size(geomValsDNP, 1)
    for r = rValsD
        for K = KValsD
            rows = sweepCell(rows, true, false, r, K, nQValsD, ...
                geomValsDNP(gi, 1), geomValsDNP(gi, 2), periodP, ...
                JOINT_SKIP, nReps, WORK_SKIP);
        end
    end
end

% ---- Parseable block to send back ----
fprintf('\nBEGIN_CSV\n');
fprintf('rel,per,r,K,nQ,sigma,span,joint,cen_ms,mob_ms\n');
for i = 1:numel(rows)
    fprintf('%s\n', rows{i});
end
fprintf('END_CSV\n');
fprintf(['\ncen_ms = -1 marks a cell where centres was not timed: ' ...
         'joint > %g,\nor joint x nQ > %g. The faster column reads ' ...
         '''-'' there, since only one arm ran.\n'], JOINT_SKIP, WORK_SKIP);
fprintf(['Section A is 96 cells at the reference geometry, Section B ' ...
         'adds 160 over\nsigma and span, Section C 48 tracing the node ' ...
         'count, and Section D 126\nexercising the spectral branch: 430 ' ...
         'in all. Expect several minutes.\n']);
fprintf('Reminder: this is the SECOND-run output that matters.\n\n');

% ---- helpers ----
function rows = sweepCell(rows, isRel, isPer, r, K, nQVals, sg, sp, ...
                          periodP, JOINT_SKIP, nReps, WORK_SKIP)
%SWEEPCELL  Time both arms for one (mode, r, K, geometry) cell.
%
%   Appends one CSV row per query count in NQVALS. SP is the value span
%   the source multiset is drawn over; periodically it is the period.
%
%   WORK_SKIP bounds the centres arm's work, the joint tuple count times
%   the query count, above which it is not timed.
    if nargin < 12 || isempty(WORK_SKIP), WORK_SKIP = Inf; end
    if isPer
        P = periodP;
        span = periodP;
    else
        P = 0;
        span = sp;
    end
    joint = factorial(r) * nchoosek(K, r);
    p = span * rand(K, 1);
    w = 0.2 + 0.8 * rand(K, 1);
    dens = buildExpTens(p, w, sg, r, isRel, isPer, P, 'verbose', false);
    d = dens.dim;   % r for abs, r-1 for rel
    for nQ = nQVals
        xq = span * rand(d, nQ);

        if joint <= JOINT_SKIP && joint * double(nQ) <= WORK_SKIP
            tCen = timeMethod(@() evalExpTens(dens, xq, ...
                'method', 'centres', 'verbose', false), nReps);
        else
            tCen = -1;   % skipped: decisively a Möbius cell
        end
        tMob = timeMethod(@() evalExpTens(dens, xq, ...
            'method', 'mobius', 'verbose', false), nReps);

        % Where centres was not timed there is no comparison to report:
        % the cell says which arm was faster only when both were run.
        if tCen < 0
            faster = '-';
        elseif tCen < tMob
            faster = 'centres';
        else
            faster = 'mobius';
        end

        if tCen < 0, cenMs = -1; else, cenMs = tCen * 1e3; end
        mobMs = tMob * 1e3;

        fprintf('%-4d %-4d %-3d %-4d %-5d %10d %10.3f %10.3f %-8s\n', ...
            isRel, isPer, r, K, nQ, joint, cenMs, mobMs, faster);
        rows{end + 1} = sprintf('%d,%d,%d,%d,%d,%g,%g,%d,%.4f,%.4f', ...
            isRel, isPer, r, K, nQ, sg, span, joint, cenMs, mobMs); %#ok<AGROW>
    end
end


function t = timeMethod(fn, nReps) %#ok<INUSD>
    % Timing policy lives in internal.timeRepeated: discard the first
    % few runs, then take the median of several more. One warm-up and
    % three timed runs was not enough -- timings do not settle until a
    % few calls have been made -- and min() reports the luckiest run
    % rather than the typical one. nReps is retained for call
    % compatibility and is no longer used.
    t = internal.timeRepeated(fn);
end
