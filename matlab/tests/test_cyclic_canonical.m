%% test_cyclic_canonical.m — narrowed cyclic canonical form
%
%  internal.cyclicCanonical narrows the candidate rotations before
%  comparing them exactly (it was a rotation-by-rotation search, whose
%  O(n^2 log n) cost dominated canonicalisation of spectrally-expanded
%  sets). These tests pin that it returns exactly what a
%  rotation-by-rotation search returns (localRef below: every rotation,
%  re-sorted stably, rounded to 9 decimals, lex-smallest pitches then
%  weights, earliest rotation on ties), on sets chosen to stress the
%  narrowing: symmetric sets, runs of equal values with weights, values
%  just below the period, spectrally-expanded triads, and large sets.
%
%  The last block pins that pairCanonicalKey gives EDO pairs (pitches that
%  are not whole cents) one key exactly when they are equivalent.
%
%  Mirror of Python tests/test_cyclic_canonical.py.
%
%  Standalone-runnable; appends to `results` when called from test_mpt.m.

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    addpath(fileparts(fileparts(mfilename('fullpath'))));
    clear cleanupDefaults
    cleanupDefaults = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end

rng(7, 'twister');
PERIODS = [1200, 12, 1, 2*pi, 7.3];

% --- symmetric sets (many tied rotations), with and without weights ------

ok = true;
for period = PERIODS
    for edo = [12 19 31 53 100]
        base = (0:edo-1) * period / edo;
        sub = randperm(edo, max(7, floor(edo/2)));
        ok = ok && localSame(base, [], period) ...
                && localSame(base, randi(2, 1, edo), period) ...
                && localSame([base base], rand(1, 2*edo), period) ...
                && localSame([base(sub) base(sub) + period/2], [], period);
    end
end
results{end+1,1} = ['cyclicCanonical: symmetric sets match the ' ...
                    'rotation-by-rotation search'];
results{end,2}   = ok;

% --- random sets with ties, near-period values, and chains ---------------

ok = true;
for period = PERIODS
    for t = 1:100
        n = randi([2 80]);
        near = period - 1e-9 * rand(1, n) .* (rand(1, n) > 0.5) ...
               + rand(1, n) * period .* (rand(1, n) > 0.5);
        if rand < 0.5, w = []; else, w = rand(1, n); end
        q = round(rand(1, n) * period, 3);
        q = [q q(1:floor(n/3))];
        chain = mod((0:n-1) * rand * period, period);
        ok = ok && localSame(near, w, period) ...
                && localSame(q, randi(3, 1, numel(q)), period) ...
                && localSame(chain, [], period);
    end
end
results{end+1,1} = ['cyclicCanonical: random sets with ties and ' ...
                    'near-period values match'];
results{end,2}   = ok;

% --- spectrally-expanded triads, and large sets --------------------------

ok = true;
for t = 1:40
    x = sort(randi(24, 1, 4)) * 100 + 3600;
    [p, w] = addSpectra(x, [], 'harmonic', 64, 'powerlaw', 1);
    ok = ok && localSame(p, w, 1200);
end
for n = [40 500]
    base = (0:n-1) * 1200 / n;
    ok = ok && localSame(base, [], 1200) ...
            && localSame(base, randi(2, 1, n), 1200) ...
            && localSame(rand(1, n) * 1200, rand(1, n), 1200);
end
results{end+1,1} = 'cyclicCanonical: spectral triads and large sets match';
results{end,2}   = ok;

% --- column inputs give the same values ----------------------------------

[p, w] = addSpectra([3600 4000 4300 4800], [], 'harmonic', 16, ...
                    'powerlaw', 1);
p = mod(p(:)', 1200); [p, si] = sort(p); w = w(:)'; w = w(si);
[a1, a2, a3] = internal.cyclicCanonical(p, w, true, 1200);
[b1, b2, b3] = internal.cyclicCanonical(p(:), w(:), true, 1200);
results{end+1,1} = 'cyclicCanonical: row and column inputs agree';
results{end,2}   = isequal(a1(:), b1(:)) && isequal(a2(:), b2(:)) ...
                   && a3 == b3;

% --- EDO pairs: equivalent pairs share a key, others never do -----------
% (the keys' rounding is set by internal.keyRoundDigits)
% Pitches of 12- to 53-EDO reached by different arithmetic carry
% different floating-point noise; symmetric A sets have several rotations
% giving A's canonical form, for which B's shift is chosen jointly.

ok = true;
for nEdo = [12 19 22 24 31 53]
    divs = find(mod(nEdo, 2:nEdo-1) == 0) + 1;
    for per = [false true]
        for rel = [false true]
            keys = {}; cls = {};
            for i = 1:20
                if mod(i, 3) == 1 && ~isempty(divs)
                    dd = divs(randi(numel(divs))); g = nEdo / dd;
                    cell0 = randperm(g, randi(min(g, 3))) - 1;
                    x = sort(reshape(cell0(:) + g * (0:dd-1), 1, []));
                else
                    x = sort(randperm(3 * nEdo, randi([2 5])) - 1);
                end
                y = sort(randperm(3 * nEdo, randi([2 5])) - 1);
                for c = 1:4
                    t = randi([-2 * nEdo, 3 * nEdo]);
                    xs = x + t; ys = y + t;
                    if per
                        xs = xs + randi([-1 1], size(xs)) * nEdo;
                        ys = ys + randi([-1 1], size(ys)) * nEdo;
                    end
                    how = randi(4) - 1;
                    % At the factory accuracy (truncationSigmas = 6); the
                    % suite pins Inf, whose 1e-12 floor needs rounding as
                    % fine as the noise in pitches built by repeated
                    % addition, so copies may then keep separate keys.
                    [pa, ~, pb] = internal.pairCanonicalKey( ...
                        localEdoCents(xs, nEdo, how), [], ...
                        localEdoCents(ys, nEdo, how), [], ...
                        rel, per, 1200, [], ...
                        internal.keyRoundDigits(10, 1 + rel, 6));
                    keys{end+1} = [sprintf('%.17g,', pa), '|', ...
                                   sprintf('%.17g,', pb)]; %#ok<SAGROW>
                    cls{end+1} = localEdoTruth(xs, ys, nEdo, rel, per); %#ok<SAGROW>
                end
            end
            [uk, ~, jk] = unique(keys);
            nMerged = 0;
            for k = 1:numel(uk)
                nMerged = nMerged + (numel(unique(cls(jk == k))) > 1);
            end
            ok = ok && nMerged == 0 && numel(uk) == numel(unique(cls));
        end
    end
end
results{end+1,1} = ['pairCanonicalKey: EDO pairs share a key exactly ' ...
                    'when equivalent'];
results{end,2}   = ok;

if standalone
    nPass = 0; nFail = 0;
    for i = 1:size(results, 1)
        if results{i,2}
            nPass = nPass + 1;
            fprintf('  PASS  %s\n', results{i,1});
        else
            nFail = nFail + 1;
            fprintf('  FAIL  %s\n', results{i,1});
        end
    end
    fprintf(['\n=== test_cyclic_canonical: %d passed, %d failed ' ...
             '(of %d) ===\n\n'], nPass, nFail, nPass + nFail);
    clear cleanupDefaults
    if nFail > 0
        error('test_cyclic_canonical:failed', '%d test(s) failed.', nFail);
    end
end


function ok = localSame(p, w, period)
    p = mod(p(:)', period);
    [p, si] = sort(p);
    hasW = ~isempty(w);
    if hasW
        w = w(:)';
        w = w(si);
    end
    [rp, rw, rs] = localRef(p, w, hasW, period);
    [gp, gw, gs] = internal.cyclicCanonical(p, w, hasW, period);
    ok = isequal(gp(:), rp(:)) && isequal(gw(:), rw(:)) && gs == rs;
end


function [pBest, wBest, bestShift] = localRef(p, w, hasW, period)
    n = numel(p);
    pBest = round(p - p(1), 9);
    wBest = w;
    bestShift = p(1);
    for rot = 2:n
        [s, si] = sort(mod(p - p(rot), period));
        s = round(s, 9);
        cmp = internal.lexCompare(s, pBest);
        if cmp < 0 || (cmp == 0 && hasW ...
                       && internal.lexCompare(w(si), wBest) < 0)
            pBest = s;
            if hasW, wBest = w(si); end
            bestShift = p(rot);
        end
    end
end


function c = localEdoCents(steps, nEdo, how)
%   The same EDO pitches by different arithmetic.
    step = 1200 / nEdo;
    switch how
        case 0, c = steps * 1200 / nEdo;
        case 1, c = steps * step;
        case 2, c = steps / nEdo * 1200;
        otherwise
            c = arrayfun(@(k) sign(k) * sum(repmat(step, 1, abs(k))), steps);
    end
end


function key = localEdoTruth(x, y, nEdo, rel, per)
%   The pair's equivalence class, computed exactly in EDO steps.
    if per
        x = mod(x, nEdo); y = mod(y, nEdo);
        if rel
            key = [localMinRot(x, nEdo), '|', localMinRot(y, nEdo)];
        else
            best = '';
            for t = 0:nEdo-1
                k = [mat2str(sort(mod(x - t, nEdo))), '|', ...
                     mat2str(sort(mod(y - t, nEdo)))];
                if isempty(best) || localStrLess(k, x, y, t, nEdo, best)
                    best = k;
                end
            end
            key = best;
        end
    elseif rel
        key = [mat2str(sort(x - min(x))), '|', mat2str(sort(y - min(y)))];
    else
        key = [mat2str(sort(x - min(x))), '|', mat2str(sort(y - min(x)))];
    end
end


function k = localMinRot(z, nEdo)
    best = [];
    for t = 0:nEdo-1
        r = sort(mod(z - t, nEdo));
        if isempty(best) || internal.lexCompare(r, best) < 0
            best = r;
        end
    end
    k = mat2str(best);
end


function tf = localStrLess(~, x, y, t, nEdo, best)
%   Lex comparison of the numeric pair form at shift t with best.
    parts = strsplit(best, '|');
    bx = eval(parts{1}); by = eval(parts{2});
    ax = sort(mod(x - t, nEdo)); ay = sort(mod(y - t, nEdo));
    c = internal.lexCompare(ax, bx);
    if c == 0
        c = internal.lexCompare(ay, by);
    end
    tf = c < 0;
end
