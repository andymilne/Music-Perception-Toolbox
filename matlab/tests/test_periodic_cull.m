%% test_periodic_cull.m — the spatial cull on the relative periodic path
%
%  Relative periodic was the one evaluation path with no cull: every
%  tuple was evaluated against every query. It admits one, because the
%  quadratic form bounds each wrapped coordinate. Q is the pairwise-wrap
%  sum over r positions divided by r, and the inner terms are
%  non-negative, so a pair inside the truncation threshold has every
%  |wrap(D_k)| <= sqrt(r) k sigma; on that region no inner pair can
%  wrap, so Q there is exactly w' (I - J/r) w, whose inverse has
%  diagonal 2, tightening the box to sqrt(2) k sigma whatever r is. The
%  lattice is then pitched on the circle and its neighbour indices taken
%  modulo the bucket count.
%
%  The claim the tests have to catch failing is that the box is wide
%  enough: too narrow a box silently drops contributions, by amounts far
%  below any tolerance chosen for other reasons. So the comparisons are
%  against the untruncated evaluation, where a dropped term shows up.
%
%  Mirror of Python tests/test_periodic_cull.py.
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

PERIOD = 1200;
SCALE  = [0 200 400 500 700 900 1100];
SIGMA  = 15;

rng(0);

% --- the culled result against the untruncated one ------------------------
% truncationSigmas = Inf resolves to the width at which the kernel is
% below the toolbox's 1e-12 parity floor, so the two should agree there
% to that floor. A box too narrow to hold the truncation region shows up
% here and nowhere else.

for r = [2 3 4 5]
    for exchFlag = [true false]
        specs = flatSpecs({SCALE(:)}, 'r', r, 'rel', true, 'exch', exchFlag);
        dens  = buildMaet({SCALE(:)}, [], 'specs', specs, ...
                          'sigma', SIGMA, 'isPer', true, ...
                          'period', PERIOD, 'verbose', false);
        dim = r - 1;
        X = rand(dim, 4000) * PERIOD;

        got  = evalMaet(dens, X, 'verbose', false);
        want = evalMaet(dens, X, 'truncationSigmas', Inf, 'verbose', false);

        results{end+1,1} = sprintf(['periodic cull: r = %d, %s, agrees ' ...
                                    'with the untruncated evaluation'], ...
                                   r, ternaryLabel(exchFlag));
        results{end,2}   = max(abs(got(:) - want(:))) < 1e-7;
    end
end

% --- the density at its own centres ---------------------------------------
% Each centre carries a kernel, so the density there cannot be below that
% kernel's own weight. Catches a cull that misses the query's own bucket.

specs = flatSpecs({SCALE(:)}, 'r', 4, 'rel', true, 'exch', true);
dens  = buildMaet({SCALE(:)}, [], 'specs', specs, 'sigma', SIGMA, ...
                  'isPer', true, 'period', PERIOD, 'verbose', false);
Cc = maetCentres(dens);
vals = evalMaet(dens, Cc{1}, 'verbose', false);
results{end+1,1} = 'periodic cull: the density at its own centres is at least 1';
results{end,2}   = all(vals(:) >= 1 - 1e-9);

% --- queries outside the principal period ---------------------------------
% A query is reduced onto the circle, so an unreduced one gives the same
% value as its reduced image.

X = rand(3, 2000) * PERIOD;
shifted = X + PERIOD * randi([-3 3], size(X));
a = evalMaet(dens, X, 'verbose', false);
b = evalMaet(dens, shifted, 'verbose', false);
results{end+1,1} = 'periodic cull: an unreduced query matches its reduced image';
results{end,2}   = max(abs(a(:) - b(:))) < 1e-9;

% --- the gate stands down where the cull cannot apply ---------------------
% A kernel wide relative to the period cannot be culled: the three-bucket
% span would wrap onto itself. The fallback is the exact path, which must
% still be right.

WIDE = 400;
specsW = flatSpecs({SCALE(:)}, 'r', 4, 'rel', true, 'exch', true);
densW  = buildMaet({SCALE(:)}, [], 'specs', specsW, 'sigma', WIDE, ...
                   'isPer', true, 'period', PERIOD, 'verbose', false);
Xw = rand(3, 500) * PERIOD;
gotW  = evalMaet(densW, Xw, 'verbose', false);
wantW = evalMaet(densW, Xw, 'truncationSigmas', Inf, 'verbose', false);
results{end+1,1} = 'periodic cull: a wide kernel falls back and stays correct';
results{end,2}   = max(abs(gotW(:) - wantW(:))) < 1e-7;

% --- the bound contains the region ----------------------------------------
% The property the half-width exists to guarantee, checked by sampling:
% nothing inside the truncation threshold lies outside the box.

KSIG = 6;
SIG  = 10;
allInside = true;
for r = [2 3 4 5]
    dim  = r - 1;
    half = localHalfWidth(r, SIG, KSIG, PERIOD);
    W = (rand(dim, 200000) - 0.5) * PERIOD;
    Q = sum(W.^2, 1);
    for i = 1:dim
        for j = i+1:dim
            delta = W(i, :) - W(j, :);
            delta = delta - PERIOD * floor(delta / PERIOD + 0.5);
            Q = Q + delta.^2;
        end
    end
    Q = Q / r;
    inside = Q <= (KSIG * SIG)^2;
    if any(inside) && max(max(abs(W(:, inside)))) > half + 1e-9
        allInside = false;
    end
end
results{end+1,1} = 'periodic cull: the box contains the truncation region';
results{end,2}   = allInside;


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
    fprintf(['\n=== test_periodic_cull: %d passed, %d failed ' ...
             '(of %d) ===\n\n'], nPass, nFail, nPass + nFail);
    clear cleanupDefaults
    if nFail > 0
        error('test_periodic_cull:failed', '%d test(s) failed.', nFail);
    end
end


function label = ternaryLabel(exchFlag)
    if exchFlag
        label = 'unordered';
    else
        label = 'ordered';
    end
end


function halfWidth = localHalfWidth(r, sigma, kSigma, period)
%LOCALHALFWIDTH  The same rule as internal.gaussianKernelSum's
%   localRelPerHalfWidth, restated here so the test pins the property
%   rather than the implementation.
    rho   = kSigma * sigma;
    loose = sqrt(r) * rho;
    if 4 * loose < period
        halfWidth = sqrt(2) * rho;
    else
        halfWidth = loose;
    end
end
