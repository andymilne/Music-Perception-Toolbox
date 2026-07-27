%% bench_nested_contract.m
%
%  Measures whether the per-event-pair scalar loop in
%  internal.nestedContract (tripSum -> nestedIp) is a material cost, and
%  how it scales with the number of events.
%
%  Background. Both languages agree numerically on the nested contraction,
%  but they reach the answer differently. The Python side folds the whole
%  event-pair grid into a leading batch axis and reduces it in one pass.
%  The MATLAB side walks the grid with a doubly-nested loop, calling
%  nestedIp once per event pair, and does so three times per comparison
%  (the XY, XX, and YY terms). This script establishes whether that
%  difference matters before any porting work is considered.
%
%  What it reports.
%    1. Wall time for 'contract' against passage length, with the implied
%       growth exponent. A loop over the event-pair grid predicts an
%       exponent near 2.
%    2. Wall time for 'bulger' on the same inputs, so the two routes can be
%       compared. If 'contract' is already the slower of the two at the
%       sizes of interest, that is itself the finding.
%    3. A profiler breakdown at the largest size, attributing time to
%       tripSum, nestedIp, and makeQuadrature as a share of the total.
%
%  Run from the matlab directory:  clear all; rehash; bench_nested_contract
%
%  This script is a measurement tool, not part of the toolbox or its test
%  suite. Delete it once the question is settled.

clear functions %#ok<CLFUNC>

SIG    = 0.15;    % kernel width
PERIOD = 12.0;    % octave equivalence
L      = 3;       % bind width: 3-grams, as in the chorale cadence example
SIZES  = [8 16 32 64];        % passage lengths (events before binding)
REPS   = 3;       % repeats per size; the minimum is reported

fprintf('\n');
fprintf('Nested contraction: cost of the per-event-pair loop\n');
fprintf('sigma = %.3g, period = %.4g, bind width L = %d\n', SIG, PERIOD, L);
fprintf('\n');

% --- Warm-up ---------------------------------------------------------
% The first call carries parse and JIT overhead that would otherwise be
% charged to the smallest size.
[wx, wy] = local_pair(8, L, SIG, PERIOD);
cosSimExpTens(wx, wy, 'method', 'contract', 'verbose', false);
cosSimExpTens(wx, wy, 'method', 'bulger',   'verbose', false);

% --- Scaling sweep ---------------------------------------------------
nSizes  = numel(SIZES);
nEvents = zeros(1, nSizes);
tCon    = zeros(1, nSizes);
tBul    = zeros(1, nSizes);
sCon    = zeros(1, nSizes);
sBul    = zeros(1, nSizes);

for k = 1:nSizes
    N = SIZES(k);
    [X, Y] = local_pair(N, L, SIG, PERIOD);
    nEvents(k) = size(X.pAttr{1}, 2);

    best = Inf; val = NaN;
    for rep = 1:REPS
        t0 = tic;
        val = cosSimExpTens(X, Y, 'method', 'contract', 'verbose', false);
        best = min(best, toc(t0));
    end
    tCon(k) = best; sCon(k) = val;

    best = Inf; val = NaN;
    for rep = 1:REPS
        t0 = tic;
        val = cosSimExpTens(X, Y, 'method', 'bulger', 'verbose', false);
        best = min(best, toc(t0));
    end
    tBul(k) = best; sBul(k) = val;

    fprintf('  N = %4d  ->  %4d events   contract %8.4f s   bulger %8.4f s   (sim %.6f / %.6f)\n', ...
            N, nEvents(k), tCon(k), tBul(k), sCon(k), sBul(k));
end

% --- Growth exponent -------------------------------------------------
% Fit log(time) against log(events). A per-pair loop over the event grid
% predicts a slope near 2; a batched reduction should sit well below it.
fprintf('\n');
ok = nEvents > 0 & tCon > 0;
if nnz(ok) >= 2
    pCon = polyfit(log(nEvents(ok)), log(tCon(ok)), 1);
    pBul = polyfit(log(nEvents(ok)), log(tBul(ok)), 1);
    fprintf('  growth exponent, contract : %.2f\n', pCon(1));
    fprintf('  growth exponent, bulger   : %.2f\n', pBul(1));
    fprintf('  (a per-pair loop over the event grid implies about 2)\n');
end

% --- Agreement check -------------------------------------------------
% A timing result is only meaningful if both routes are computing the same
% quantity on these inputs.
d = max(abs(sCon - sBul));
fprintf('\n  max |contract - bulger| across sizes: %.3e', d);
if d < 1e-9
    fprintf('   (agree)\n');
else
    fprintf('   *** ROUTES DISAGREE - timings below are not comparable ***\n');
end

% --- Profiler breakdown at the largest size --------------------------
fprintf('\n');
fprintf('Profiler breakdown at N = %d\n', SIZES(end));

[X, Y] = local_pair(SIZES(end), L, SIG, PERIOD);
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

% Local functions may be reported bare or package-qualified depending on
% the MATLAB version, so match on the tail of the name rather than the
% whole of it.
want = {'tripSum', 'nestedIp', 'makeQuadrature', 'nestedContract'};
fprintf('  %-38s %10s %10s %8s\n', 'function', 'time (s)', 'calls', 'share');
shown = false;
for w = 1:numel(want)
    for i = 1:numel(info.FunctionTable)
        nm = info.FunctionTable(i).FunctionName;
        if ~isempty(strfind(nm, want{w})) %#ok<STREMP>
            e = info.FunctionTable(i);
            fprintf('  %-38s %10.4f %10d %7.1f%%\n', ...
                    nm, e.TotalTime, e.NumCalls, ...
                    100 * e.TotalTime / max(total, eps));
            shown = true;
        end
    end
end
if ~shown
    fprintf('  (no matching entries; the ten heaviest functions instead)\n');
    [~, ord] = sort([info.FunctionTable.TotalTime], 'descend');
    for i = ord(1:min(10, numel(ord)))
        e = info.FunctionTable(i);
        fprintf('  %-38s %10.4f %10d\n', e.FunctionName, e.TotalTime, e.NumCalls);
    end
end
fprintf('  %-38s %10.4f\n', 'cosSimExpTens (total)', total);

fprintf('\n');
fprintf('Reading the result. If tripSum and nestedIp together account for a\n');
fprintf('small share of the total, the loop is not where the time goes and no\n');
fprintf('porting is warranted. If they dominate and the exponent is near 2,\n');
fprintf('the batched form used on the Python side is worth carrying across.\n');
fprintf('\n');


% ---------------------------------------------------------------------
function [X, Y] = local_pair(N, L, sigma, period)
    % Two nested densities from N-event chord passages bound into L-grams.
    %
    % The events must be multivalued. Binding single-valued events yields a
    % degenerate nesting -- every inner group a singleton -- which the build
    % flattens back to a plain flat attribute, leaving nothing for the
    % contraction path to do. Chords of K = 3 pitches keep the inner level
    % real, so the outer level genuinely nests.
    %
    % Inner absolute, outer relative: the transposition-invariant
    % progression, and the case the contraction path exists to serve.
    rs = RandStream('mt19937ar', 'Seed', 11);
    K  = 3;
    triad = [0; 4; 7];

    rootsA = mod(cumsum(randi(rs, [-4 4], 1, N)), period);
    rootsB = mod(rootsA + 1 + randi(rs, [0 2], 1, N), period);
    A = mod(repmat(rootsA, K, 1) + repmat(triad, 1, N), period);
    B = mod(repmat(rootsB, K, 1) + repmat(triad, 1, N), period);

    spA = flatSpecs({A}, 'r', 2, 'rel', false, 'sym', true);
    spB = flatSpecs({B}, 'r', 2, 'rel', false, 'sym', true);
    [pa, ~, spa] = bindEvents({A}, [], L, 'specs', spA, 'relOuter', true);
    [pb, ~, spb] = bindEvents({B}, [], L, 'specs', spB, 'relOuter', true);

    Pa = pa{1}; Pb = pb{1};
    X = buildExpTens({Pa}, {ones(size(Pa))}, 'specs', {spa{1}}, ...
                     'sigma', sigma, 'isPer', true, 'period', period, ...
                     'verbose', false);
    Y = buildExpTens({Pb}, {ones(size(Pb))}, 'specs', {spb{1}}, ...
                     'sigma', sigma, 'isPer', true, 'period', period, ...
                     'verbose', false);

    % Fail loudly rather than silently timing the wrong path.
    if ~(isfield(X, 'nested') && iscell(X.nested) ...
         && any(~cellfun(@isempty, X.nested)))
        error('bench:notNested', ...
              ['The constructed density is not nested, so method=''contract'' ' ...
               'does not apply. Check the inner spec: singleton inner groups ' ...
               'flatten the nesting away.']);
    end
end
