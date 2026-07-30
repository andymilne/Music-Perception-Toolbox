function results = calibrateOrbitCrossover(varargin)
% Measure the orbit-vs-enumeration cost crossover on the MATLAB side.
%
% Why this exists. The per-level route choice in internal.nestedContract has
% two criteria. For r <= 6 it uses internal.orbitBeatsPairwisePerAttr, whose
% K thresholds were measured; for r >= 7 it uses the two routes'
% complexities, orbit being cheaper exactly when
%
%     |Omega_r| Kx Ky  <  C(Kx,r) C(Ky,r) r!
%
% That comparison is blind to mode, where the measured thresholds are not:
% the relative-periodic u-grid overhead raises the K at which the orbit
% route pays off. It is also blind to per-call overhead, which dominates at
% small batch extent and differs between the two languages -- the Python
% harness that derived the criterion cannot settle the MATLAB offsets. This
% script measures them here.
%
% What it reports, per (mode, r, K, batch extent B):
%
%   - agreement between the two routes, checked BEFORE any timing, so a
%     timing comparison is never made between two different computations;
%   - wall time for each route and the measured winner;
%   - what orbitEligible would choose, and whether the two agree.
%
% Read the disagreements by K margin, not in aggregate. The criterion is
% expected to be least reliable at K = r and K = r+1, where enumeration's
% C(K,r) collapses to 1 and the orbit route's fixed |Omega_r| call count
% dominates.
%
% Terminology follows section 3 of the manuscript. K is a level's member
% count: the multiset size K_{a,n} at the innermost level, the count of
% sub-multisets at an outer one. B is the batch extent a combine receives,
% which is gx*gy*Q -- sibling pairs times shift-quadrature nodes -- not Q
% alone.
%
% Usage:
%   results = calibrateOrbitCrossover();
%   results = calibrateOrbitCrossover('rVals', 5:8, 'BVals', [1 64 4096]);
%
% Name-value options:
%   'rVals'    tuple sizes to sweep            (default 2:8)
%   'margins'  K - r values to sweep           (default 0:3)
%   'BVals'    batch extents to sweep          (default [1 32 256 2048])
%   'maxElems' peak (B, Tx, Ty) element cap    (default 2e7, ~160 MB)
%   'reps'     timed repetitions per cell      (default 3)
%   'tol'      route-disagreement guard          (default 1e-6)
%
% Error is measured the way truncationSigmas states it: as an ABSOLUTE
% perturbation on the value scale, never as a ratio to the returned value.
% A ratio explodes wherever the result is legitimately near zero and says
% nothing about accuracy. The tolerance guards against a transcription error
% in this file -- the two local routes computing different things, which
% would make their timings meaningless -- and sits well above the orbit
% route's own worst absolute error, which the shipped guard manages.

    p = inputParser;
    p.addParameter('rVals', 2:8);
    p.addParameter('margins', 0:3);
    p.addParameter('BVals', [1 32 256 2048]);
    p.addParameter('maxElems', 2e7);
    p.addParameter('reps', 3);
    p.addParameter('tol', 1e-6);
    p.parse(varargin{:});
    opt = p.Results;

    % Provenance. Two runs of this harness on different trees are not
    % comparable, and nothing in the output previously said which tree
    % produced it. Fingerprint the files whose behaviour is measured.
    fprintf('\ncalibrateOrbitCrossover  %s\n', ...
            datestr(now, 'yyyy-mm-dd HH:MM:SS')); %#ok<TNOW1,DATST>
    fprintf('  postHocGuards = %d   (switch off for timing: the nested\n', ...
            logical(mptDefaults('postHocGuards')));
    fprintf('  guard pays for both routes when it diverts)\n');
    provFiles = {'+internal/nestedContract.m', ...
                 '+mobius/innerProductOrbitGrid.m', ...
                 '+internal/orbitBeatsPairwisePerAttr.m'};
    for pf = 1:numel(provFiles)
        fprintf('  %-42s %s\n', provFiles{pf}, fileFingerprint(provFiles{pf}));
    end

    modes = {'absolute', 'relPeriodic'};
    results = struct('mode', {}, 'r', {}, 'K', {}, 'B', {}, ...
                     'absErr', {}, 'tOrbit', {}, 'tEnum', {}, ...
                     'workRatio', {}, 'timeRatio', {}, ...
                     'measured', {}, 'predicted', {}, 'agree', {});

    fprintf(['\n%-12s %3s %3s %6s %10s %10s %10s %9s %9s  ' ...
             '%-6s %-6s %s\n'], ...
            'mode', 'r', 'K', 'B', 'absErr', 'orbit ms', 'enum ms', ...
            'predRatio', 'measRatio', 'measrd', 'predct', 'ok');
    fprintf(['predRatio is the work criterion''s orbit-over-enum ratio, ' ...
             'measRatio the measured\ntime ratio; 1 is the crossover for ' ...
             'both, so the gap between them is the model''s\nerror. At ' ...
             'r <= 6 the shipped choice comes from the measured table ' ...
             '(orbitBeatsPairwisePerAttr),\nnot from predRatio, so the ' ...
             'two can disagree there by design.\n\n']);
    fprintf('%s\n', repmat('-', 1, 104));

    for mi = 1:numel(modes)
        mode = modes{mi};
        isRel = strcmp(mode, 'relPeriodic');
        isPer = isRel;
        for r = opt.rVals
            for m = opt.margins
                K = r + m;
                for B = opt.BVals
                    [Tx, Ty] = tupleCounts(K, r);
                    if B * Tx * Ty > opt.maxElems
                        continue;   % enumeration not materialisable here
                    end

                    M = makeBlock(B, K, K, r);

                    % Agreement first. Timing two routes that do not compute
                    % the same thing is the error this guards against.
                    vOrb = combineOrbitLocal(M, r);
                    vEnu = combineEnumLocal(M, r);
                    absErr = max(abs(vOrb - vEnu));
                    if ~(isnan(absErr)) && absErr > opt.tol
                        warning('mpt:calibrateOrbitCrossover:disagree', ...
                            ['Routes disagree at r = %d, K = %d, B = %d ' ...
                             '(absolute difference %.2e), beyond what the ' ...
                             'orbit route''s cancellation explains. Suspect ' ...
                             'a transcription error in this file rather ' ...
                             'than a toolbox fault; skipping the cell.'], ...
                            r, K, B, absErr);
                        continue;
                    end

                    tOrb = timeRoute(@() combineOrbitLocal(M, r), opt.reps);
                    tEnu = timeRoute(@() combineEnumLocal(M, r), opt.reps);

                    if tOrb < tEnu
                        measured = 'orbit';
                    else
                        measured = 'enum';
                    end
                    if predictOrbit(K, r, isRel, isPer)
                        predicted = 'orbit';
                    else
                        predicted = 'enum';
                    end
                    ok = strcmp(measured, predicted);
                    if ok; okStr = 'yes'; else; okStr = 'NO'; end

                    % Predicted work ratio against measured time ratio:
                    % both are orbit-over-enum, so 1 is the crossover and
                    % the gap between them is the model's error on a
                    % continuous scale.
                    [oW, eW] = predictWork(K, r);
                    workRatio = oW / eW;
                    timeRatio = tOrb / tEnu;

                    fprintf(['%-12s %3d %3d %6d %10.2e %10.3f %10.3f ' ...
                             '%9.2f %9.2f  %-6s %-6s %s\n'], ...
                            mode, r, K, B, absErr, tOrb*1e3, tEnu*1e3, ...
                            workRatio, timeRatio, measured, predicted, okStr);

                    results(end+1) = struct('mode', mode, 'r', r, 'K', K, ...
                        'B', B, 'absErr', absErr, 'tOrbit', tOrb, ...
                        'tEnum', tEnu, 'workRatio', workRatio, ...
                        'timeRatio', timeRatio, 'measured', measured, ...
                        'predicted', predicted, 'agree', ok); %#ok<AGROW>
                end
            end
        end
    end

    if isempty(results)
        fprintf('\nNo cells measured.\n');
        return;
    end
    nBad = sum(~[results.agree]);
    fprintf('\n%d of %d cells disagree with orbitEligible.\n', ...
            nBad, numel(results));

    % Break the disagreements down by batch extent. orbitFlopsBeatEnum
    % holds only once the batch is large enough that both routes are
    % flop-bound rather than call-overhead-bound, so a headline count
    % that weights B = 1 equally with the largest B overstates the
    % failure. Read the largest-B row.
    fprintf('\n%-8s %8s %8s %12s\n', 'B', 'cells', 'disagree', 'median t_o/t_e');
    fprintf('%s\n', repmat('-', 1, 40));
    allB = unique([results.B]);
    for bi = 1:numel(allB)
        sel = [results.B] == allB(bi);
        fprintf('%-8d %8d %8d %12.1f\n', allB(bi), sum(sel), ...
                sum(~[results(sel).agree]), median([results(sel).timeRatio]));
    end

    % Raw block, so a later run can be compared against this one rather
    % than against a remembered conclusion.
    fprintf('\nBEGIN_CSV\n');
    fprintf('mode,r,K,B,absErr,tOrbit_ms,tEnum_ms,workRatio,timeRatio\n');
    for i = 1:numel(results)
        fprintf('%s,%d,%d,%d,%.6e,%.6f,%.6f,%.6f,%.6f\n', ...
                results(i).mode, results(i).r, results(i).K, results(i).B, ...
                results(i).absErr, results(i).tOrbit*1e3, ...
                results(i).tEnum*1e3, results(i).workRatio, ...
                results(i).timeRatio);
    end
    fprintf('END_CSV\n');
end


% ----------------------------------------------------------------------
function tag = fileFingerprint(relPath)
    % Modification time and byte count of a toolbox file, so a run can be
    % matched to the tree that produced it.
    full = which(regexprep(relPath, '^\+\w+[/\\]', ''));
    if isempty(full)
        p = fileparts(fileparts(mfilename('fullpath')));
        full = fullfile(p, relPath);
    end
    d = dir(full);
    if isempty(d)
        tag = '(not found)';
    else
        tag = sprintf('%s  %d bytes', ...
                      datestr(d.datenum, 'yyyy-mm-dd HH:MM'), d.bytes); %#ok<DATST>
    end
end


function [Tx, Ty] = tupleCounts(K, r)
    % X side takes ordered tuples, Y side unordered -- the r!-cancelled
    % perm x comb form the enumerated combine uses.
    if r > K
        Tx = 0; Ty = 0;
        return;
    end
    Tx = nchoosek(K, r) * factorial(r);
    Ty = nchoosek(K, r);
end


% ----------------------------------------------------------------------
function M = makeBlock(B, Kx, Ky, r)
    % Gaussian overlap block exp(-(vx - vy)^2 / (4 sigma^2)) with per-batch
    % jitter, so the batch is not B identical copies (which would let the
    % cache flatter one route over the other).
    rng(1000 * r + Kx, 'twister');
    vx = sort(12 * rand(Kx, 1));
    vy = sort(12 * rand(1, Ky));
    base = exp(-((vx - vy).^2) / 4);
    jit = 1 + 0.05 * randn(B, 1);
    M = reshape(jit, [B 1 1]) .* reshape(base, [1 Kx Ky]);
end


% ----------------------------------------------------------------------
function v = combineOrbitLocal(M, r)
    % Orbit route, on the scale the enumerated combine returns. K_u is
    % (batch, n_A, n_B), which is the layout M already carries -- no
    % permutation.
    B = size(M, 1);
    Kx = size(M, 2);
    Ky = size(M, 3);
    w_A = ones(Kx, 1);
    w_B = ones(Ky, 1);
    vals = mobius.innerProductOrbitGrid(M, w_A, w_B, r);
    v = reshape(vals, [B 1]) / factorial(r);
end


% ----------------------------------------------------------------------
function v = combineEnumLocal(M, r)
    % Enumerated route. This mirrors internal.nestedContract/combine
    % verbatim -- the (B, Tx, Ty) materialisation and r element-wise
    % multiplies -- because a loop-based transcription would time this file
    % rather than the shipped route, and would report the orbit route as the
    % winner nearly everywhere.
    [xt, yt] = tupleIndicesLocal(size(M, 2), r);
    [~, ytY] = tupleIndicesLocal(size(M, 3), r);
    yt = ytY;
    Tx = size(xt, 1);
    Ty = size(yt, 1);
    if Tx == 0 || Ty == 0
        v = zeros(size(M, 1), 1);
        return;
    end
    P = M(:, xt(:, 1), yt(:, 1));                   % B x Tx x Ty
    for t = 2:r
        P = P .* M(:, xt(:, t), yt(:, t));
    end
    v = sum(sum(P, 3), 2);
    v = v(:);
end


% ----------------------------------------------------------------------
function [xt, yt] = tupleIndicesLocal(n, r)
    % Mirror of internal.nestedContract/tupleIndices at sym = true, which is
    % the only case an orbit-vs-enumeration comparison arises in. X side
    % takes permutations of r-combinations, Y side the combinations.
    if r > n
        xt = zeros(0, r);
        yt = zeros(0, r);
        return;
    end
    if r == 1
        C = (1:n).';
    else
        C = nchoosek(1:n, r);
    end
    yt = C;
    if r > 1
        P = perms(1:r);
        nC = size(C, 1);
        nP = size(P, 1);
        xt = zeros(nC * nP, r);
        idx = 1;
        for q = 1:nP
            xt(idx:idx + nC - 1, :) = C(:, P(q, :));
            idx = idx + nC;
        end
    else
        xt = C;
    end
end


% ----------------------------------------------------------------------
function t = timeRoute(fn, reps)
    fn();                      % warm any caches, as in steady state
    t0 = tic;
    for i = 1:reps
        fn();
    end
    t = toc(t0) / reps;
end


% ----------------------------------------------------------------------
function [orbWork, enumWork] = predictWork(K, r)
    % The two work counts the r >= 7 criterion compares. Reporting them
    % rather than only the winner they imply makes the model's error a
    % continuous quantity: predicted work ratio against measured time
    % ratio. A label discards the magnitude, so a model wrong by 40x and
    % one wrong by 1.01x score the same.
    orbWork = numel(mobius.getOrbitTable(r)) * K * K;
    if r > K
        enumWork = Inf;
    else
        enumWork = nchoosek(K, r)^2 * factorial(r);
    end
end


function tf = predictOrbit(K, r, isRel, isPer)
    % Mirror of internal.nestedContract/orbitEligible. That function is
    % local to its file and cannot be called from here, so the policy is
    % restated; if it changes there, change it here too.
    ORBIT_R_MAX_SHIPPED = 8;
    tf = false;
    if r < 2 || r > ORBIT_R_MAX_SHIPPED
        return;
    end
    if r <= 6
        tf = internal.orbitBeatsPairwisePerAttr(r, K, isRel, isPer);
    else
        if r > K
            tf = false;
            return;
        end
        n = numel(mobius.getOrbitTable(r));
        tf = n * K * K < nchoosek(K, r)^2 * factorial(r);
    end
end
