%% bench_sweep.m — cross-language sweep benchmark (point-set query shape)
%
%  MATLAB counterpart of bench_sweep.py: identical deterministic
%  inputs, identical variant definitions, one CSV row per
%  (N, normalize, variant). Join against the Python CSV with
%  compare_sweep.py; the checksum column doubles as a cross-language
%  value-parity check.
%
%  Variants:
%    broadcast   — one cosSimExpTens(dX, {d1, ..., dM}) call (the
%                  batched kernel pass; self terms memoised in-call).
%    loop_memo   — M scalar calls threading the cache-carrying output
%                  ([s, dRef] = cosSimExpTens(dRef, dQ{k}, ...)), so
%                  the context's self term is paid once.
%    loop_fresh  — M plain scalar calls (value semantics: no memo
%                  crosses calls; each pair pays its own self terms).
%
%  Run from anywhere with the toolbox on the path. Requires MATLAB
%  (internal.timeRepeated uses an arguments block; not Octave-runnable).


function bench_sweep()

% Dispatch hints would print inside the timed closures and distort
% the measurements; silence them for the run (session-scoped).
mptDefaults('showHints', false);
internal.maybeShowTruncationNotice('suppress');
SIG_P = 0.25;  SIG_T = 0.08;
M_QUERIES = 100;
GRID_N = [300, 1200, 5000];
NORMS = {'cosine', 'oneSidedDenom'};

outPath = fullfile(fileparts(mfilename('fullpath')), ...
    'bench_sweep_matlab.csv');
fid = fopen(outPath, 'w');
fprintf(fid, ['language,N,M,normalize,variant,' ...
              'ms_per_offset,n_inner,checksum\n']);

for ni = 1:numel(GRID_N)
    N = GRID_N(ni);
    % Deterministic context (mirrors bench_sweep.py context_arrays).
    j = 1:N;
    P = 40 + 50 * mod(7 * j.^2 + 3 * j, 997) / 997;
    T = cumsum(0.05 + 0.2 * mod(3 * j, 11) / 11);
    dCtx = buildExpTens({P, T}, [], [SIG_P SIG_T], [1 1], ...
        [false false], [false false], [0 0], 'verbose', false);
    dQs = cell(1, M_QUERIES);
    for k = 1:M_QUERIES
        kk = k - 1;   % zero-based sweep index, matching Python
        qP = [60 63.5 68.25] + 0.7 * kk;
        qT = [0.5 1.25 2.0] + 1.3 * kk;
        dQs{k} = buildExpTens({qP, qT}, [], [SIG_P SIG_T], [1 1], ...
            [false false], [false false], [0 0], 'verbose', false);
    end

    for nrmI = 1:numel(NORMS)
        nrm = NORMS{nrmI};

        fnBroadcast = @() cell2mat(cosSimExpTens(dCtx, dQs, ...
            'normalize', nrm, 'verbose', false));
        fnLoopMemo  = @() iLoopMemo(dCtx, dQs, nrm);
        fnLoopFresh = @() iLoopFresh(dCtx, dQs, nrm);

        variants = {'broadcast', fnBroadcast; ...
                    'loop_memo', fnLoopMemo; ...
                    'loop_fresh', fnLoopFresh};
        for vi = 1:size(variants, 1)
            name = variants{vi, 1};
            fn = variants{vi, 2};
            s = fn();                                  %#ok<NASGU> warm
            [t, nTimed] = internal.timeRepeated(fn);
            s = fn();
            checksum = sum(s);
            msPerOffset = t / M_QUERIES * 1000;
            fprintf(fid, 'matlab,%d,%d,%s,%s,%.6f,%d,%.12e\n', ...
                N, M_QUERIES, nrm, name, msPerOffset, nTimed, checksum);
            fprintf('N=%5d %-14s %-11s: %8.4f ms/offset  checksum %.12e\n', ...
                N, nrm, name, msPerOffset, checksum);
        end
    end
end
fclose(fid);
fprintf('\nwrote %s\n', outPath);

end


function s = iLoopMemo(dCtx, dQs, nrm)
%ILOOPMEMO  Scalar loop threading the cache-carrying output.
    M = numel(dQs);
    s = zeros(1, M);
    dRef = dCtx;
    for k = 1:M
        [s(k), dRef] = cosSimExpTens(dRef, dQs{k}, ...
            'normalize', nrm, 'verbose', false);
    end
end


function s = iLoopFresh(dCtx, dQs, nrm)
%ILOOPFRESH  Plain scalar loop: no memo crosses calls.
    M = numel(dQs);
    s = zeros(1, M);
    for k = 1:M
        s(k) = cosSimExpTens(dCtx, dQs{k}, ...
            'normalize', nrm, 'verbose', false);
    end
end
