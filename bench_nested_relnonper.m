%% bench_nested_relnonper.m
%
%  Is the per-event-pair loop a real cost in the relative-non-periodic mode?
%
%  The batched contraction covers the absolute and relative-periodic modes.
%  Relative-non-periodic kept the per-pair route because it carries a
%  closed-form shortcut, ipRelNonperFactored, chosen per pair and with no
%  batched form. Two of that shortcut's three gates are structural (ordered
%  cells, and the whole cell read as one tuple); the third asks whether the
%  cell's leaves share a template, which is a property of how the density
%  was built rather than of an individual pair. So for a plain chord
%  passage the shortcut never applies and every pair takes the generic
%  route -- exactly the form that batches.
%
%  This script measures that case: plain chords, relative, non-periodic,
%  scaled along the event axis, which is where a per-pair loop hurts.
%
%  Run from the matlab directory:  clear all; rehash; bench_nested_relnonper
%
%  A measurement tool, not part of the toolbox. Delete when done.

clear functions %#ok<CLFUNC>

SIG   = 0.30;                 % wider than elsewhere: the line quadrature
                              % steps at sigma/4 across the whole value
                              % spread, so a small sigma makes T enormous
L     = 3;                    % bind width, as in the chorale cadence case
SIZES = [8 16 32];            % passage lengths (events before binding)
REPS  = 3;

fprintf('\n');
fprintf('Relative non-periodic nested contraction: cost of the pair loop\n');
fprintf('sigma = %.3g, bind width L = %d, values on the line (no period)\n', SIG, L);
fprintf('\n');

% Warm-up, so the smallest size is not charged with parse cost.
[wx, wy] = local_pair(8, L, SIG);
cosSimExpTens(wx, wy, 'method', 'contract', 'verbose', false);
cosSimExpTens(wx, wy, 'method', 'bulger',   'verbose', false);

n = numel(SIZES);
nEv = zeros(1, n); tC = zeros(1, n); tB = zeros(1, n);
sC = zeros(1, n);  sB = zeros(1, n);

for k = 1:n
    N = SIZES(k);
    [X, Y] = local_pair(N, L, SIG);
    nEv(k) = size(X.pAttr{1}, 2);

    best = Inf; val = NaN;
    for rep = 1:REPS
        t0 = tic;
        val = cosSimExpTens(X, Y, 'method', 'contract', 'verbose', false);
        best = min(best, toc(t0));
    end
    tC(k) = best; sC(k) = val;

    best = Inf; val = NaN;
    for rep = 1:REPS
        t0 = tic;
        val = cosSimExpTens(X, Y, 'method', 'bulger', 'verbose', false);
        best = min(best, toc(t0));
    end
    tB(k) = best; sB(k) = val;

    spread = max(max(X.pAttr{1}(:)), max(Y.pAttr{1}(:))) ...
           - min(min(X.pAttr{1}(:)), min(Y.pAttr{1}(:)));
    hi = spread + (6 + 0.5 * 12) * SIG;
    nTau = max(64, ceil(2 * hi / (SIG / 4)));
    fprintf('  N = %4d  -> %4d events, ~%5d taus   contract %8.4f s   bulger %8.4f s   (sim %.6f / %.6f)\n', ...
            N, nEv(k), nTau, tC(k), tB(k), sC(k), sB(k));
end

ok = nEv > 0 & tC > 0;
if nnz(ok) >= 2
    pC = polyfit(log(nEv(ok)), log(tC(ok)), 1);
    pB = polyfit(log(nEv(ok)), log(tB(ok)), 1);
    fprintf('\n  growth exponent, contract : %.2f\n', pC(1));
    fprintf('  growth exponent, bulger   : %.2f\n', pB(1));
end

d = max(abs(sC - sB));
fprintf('\n  max |contract - bulger|: %.3e', d);
if d < 1e-8
    fprintf('   (agree)\n');
else
    fprintf('   *** ROUTES DISAGREE ***\n');
end

% --- Profiler breakdown ----------------------------------------------
fprintf('\nProfiler breakdown at N = %d\n', SIZES(end));
[X, Y] = local_pair(SIZES(end), L, SIG);
profile off; profile clear; profile on;
cosSimExpTens(X, Y, 'method', 'contract', 'verbose', false);
profile off;
info = profile('info');

total = 0;
for i = 1:numel(info.FunctionTable)
    if strcmp(info.FunctionTable(i).FunctionName, 'cosSimExpTens')
        total = info.FunctionTable(i).TotalTime;
    end
end
if total <= 0
    total = sum([info.FunctionTable.TotalTime]);
end

want = {'tripSum', 'nestedIp', 'ipRelNonperFactored', 'sharedLeafTemplate', ...
        'pairValuesBatched', 'contractNode'};
fprintf('  %-38s %10s %10s %8s\n', 'function', 'time (s)', 'calls', 'share');
for w = 1:numel(want)
    for i = 1:numel(info.FunctionTable)
        nm = info.FunctionTable(i).FunctionName;
        if ~isempty(strfind(nm, want{w})) %#ok<STREMP>
            e = info.FunctionTable(i);
            fprintf('  %-38s %10.4f %10d %7.1f%%\n', nm, e.TotalTime, ...
                    e.NumCalls, 100 * e.TotalTime / max(total, eps));
        end
    end
end
fprintf('  %-38s %10.4f\n', 'cosSimExpTens (total)', total);

fprintf('\n');
fprintf('Reading the result. If ipRelNonperFactored shows many calls and\n');
fprintf('little time, the shortcut is declining and the generic per-pair\n');
fprintf('route is carrying the cost -- which is the batchable form. If the\n');
fprintf('shortcut is doing the work, the loop is not the problem here.\n');
fprintf('\n');


% ---------------------------------------------------------------------
function [X, Y] = local_pair(N, L, sigma)
    % Plain triad passages on the line: relative, non-periodic. Values are
    % not wrapped, so the quadrature runs over the line rather than the
    % period. Multivalued events keep the nesting from flattening away.
    rs = RandStream('mt19937ar', 'Seed', 11);
    K  = 3;
    triad = [0; 4; 7];

    % A bounded walk: the tau grid spans the whole value spread at sigma/4,
    % so an unbounded walk would make the quadrature enormous. Reflecting
    % the roots into a two-octave band keeps the spread musical and the
    % grid tractable.
    step = randi(rs, [-4 4], 1, N);
    rootsA = zeros(1, N); cur = 60;
    for i = 1:N
        cur = cur + step(i);
        if cur > 84; cur = 84 - (cur - 84); end
        if cur < 60; cur = 60 + (60 - cur); end
        rootsA(i) = cur;
    end
    rootsB = rootsA + 1 + randi(rs, [0 2], 1, N);
    A = repmat(rootsA, K, 1) + repmat(triad, 1, N);
    B = repmat(rootsB, K, 1) + repmat(triad, 1, N);

    spA = flatSpecs({A}, 'r', 2, 'rel', false, 'sym', true);
    spB = flatSpecs({B}, 'r', 2, 'rel', false, 'sym', true);
    [pa, ~, spa] = bindEvents({A}, [], L, 'specs', spA, 'relOuter', true);
    [pb, ~, spb] = bindEvents({B}, [], L, 'specs', spB, 'relOuter', true);

    Pa = pa{1}; Pb = pb{1};
    % With 'specs', buildExpTens requires sigma, isPer and period together.
    % period is inert here because isPer is false: the quadrature runs over
    % the line, and no wrapping is applied.
    X = buildExpTens({Pa}, {ones(size(Pa))}, 'specs', {spa{1}}, ...
                     'sigma', sigma, 'isPer', false, 'period', 12, ...
                     'verbose', false);
    Y = buildExpTens({Pb}, {ones(size(Pb))}, 'specs', {spb{1}}, ...
                     'sigma', sigma, 'isPer', false, 'period', 12, ...
                     'verbose', false);

    if ~(isfield(X, 'nested') && iscell(X.nested) ...
         && any(~cellfun(@isempty, X.nested)))
        error('bench:notNested', ...
              'Density is not nested, so method=''contract'' does not apply.');
    end
end
