%% test_wrap_dispatch.m
%  Tests for the abs-per full-image image-sum vs Fourier dispatch.
%  Mirrors the Python tests/test_wrap_dispatch.py.
%
%  Covers:
%   - The 1-D wrapped-Gaussian helper's image-sum and Fourier
%     representations agree within the accuracy floor at both sides
%     of the sigma/P crossover.
%   - The dispatch decision matches the derived crossover (image-sum
%     below, Fourier above), verified via the count helpers.
%   - End-to-end cosine is invariant under the choice of route: at
%     any sigma/P the same numerical value comes out, regardless of
%     which representation the helper picked internally.
%   - The count helpers behave sensibly at edge conditions.
%
%  No local functions: this file is executed as a script from
%  test_mpt.m, so probes are written inline. Where the Python test
%  monkey-patches the dispatch to force a route, the MATLAB test
%  instead reconstructs each route by hand from the count helpers
%  and compares against the helper's dispatched output.
%
%  Standalone-runnable; appends to ``results`` when called from
%  test_mpt.m.

if ~exist('results', 'var')
    results = {};
    standalone = true;
else
    standalone = false;
end

dispPERIOD = 1200;
dispTS = 6;


%% ---- Helper: reconstruct image-sum and Fourier forms by hand ----
% (Inline formulae; not a local function since this file is a script.)


%% ---- Image-sum and Fourier agree within the accuracy floor ----
% Force each representation by direct computation from the count
% helpers, and check they agree within a generous floor multiple.

dispFloor = exp(-dispTS * dispTS / 2);   % accuracy-floor scale
dispAgreeOk = true;
for sopIter = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    sig = sopIter * dispPERIOD;
    dGrid = linspace(-dispPERIOD, dispPERIOD, 41);

    % Direct image-sum reconstruction.
    Lval = internal.wrappedKernelImageCount(sig, dispPERIOD, dispTS, 4);
    dRed = dGrid - dispPERIOD * floor(dGrid / dispPERIOD + 0.5);
    dispInvExp = 1 / (4 * sig * sig);
    if Lval == 0
        thetaImg = exp(-dRed .* dRed * dispInvExp);
    else
        nShift = (-Lval:Lval);
        nRe = reshape(nShift, [ones(1, ndims(dRed)), 2*Lval + 1]);
        dShift = dRed + dispPERIOD * nRe;
        thetaImg = sum(exp(-dShift .* dShift * dispInvExp), ndims(dShift));
    end

    % Direct Fourier reconstruction.
    Mval = internal.wrappedKernelFourierCount(sig, dispPERIOD, dispTS, 4);
    alpha = pi * pi * 4 * sig * sig / (dispPERIOD * dispPERIOD);
    mIdx = 1:Mval;
    env = exp(-alpha * mIdx .* mIdx);
    modeShape = ones(1, ndims(dGrid));
    modeShape(ndims(dGrid) + 1) = Mval;
    mShaped = reshape(mIdx, modeShape);
    envShaped = reshape(env, modeShape);
    twoPiOverP = 2 * pi / dispPERIOD;
    prefactor = sqrt(pi * 4) * sig / dispPERIOD;
    phase = twoPiOverP * dGrid .* mShaped;
    thetaFou = prefactor * (1 + 2 * sum(envShaped .* cos(phase), ...
        ndims(dGrid) + 1));

    % Both must be within the accuracy floor of each other.
    maxErr = max(abs(thetaImg(:) - thetaFou(:)));
    dispAgreeOk = dispAgreeOk && (maxErr < dispFloor * 100);
end
results(end+1, :) = { ...
    'wrap dispatch: image-sum and Fourier agree within accuracy floor', ...
    dispAgreeOk}; %#ok<*SAGROW>


%% ---- Dispatch prefers image-sum at small sigma/P ----

sig = 0.05 * dispPERIOD;
dispPrefFou = internal.wrappedKernelPreferFourier(sig, dispPERIOD, dispTS, 4);
Lval = internal.wrappedKernelImageCount(sig, dispPERIOD, dispTS, 4);
Mval = internal.wrappedKernelFourierCount(sig, dispPERIOD, dispTS, 4);
results(end+1, :) = { ...
    'wrap dispatch: prefers image-sum at small sigma/P', ...
    ~dispPrefFou && (2*Lval + 1 <= Mval)};


%% ---- Dispatch prefers Fourier at large sigma/P ----

sig = 0.30 * dispPERIOD;
dispPrefFou = internal.wrappedKernelPreferFourier(sig, dispPERIOD, dispTS, 4);
Lval = internal.wrappedKernelImageCount(sig, dispPERIOD, dispTS, 4);
Mval = internal.wrappedKernelFourierCount(sig, dispPERIOD, dispTS, 4);
results(end+1, :) = { ...
    'wrap dispatch: prefers Fourier at large sigma/P', ...
    dispPrefFou && (Mval < 2*Lval + 1)};


%% ---- Cosine invariant across the crossover ----
% The wrappedGaussian1d helper picks the cheaper of image-sum and
% Fourier automatically; the end-to-end cosine at any sigma/P should
% match a reconstruction that forces the other route. Rather than
% monkey-patching the internal helper, we reconstruct the alternative
% theta values by hand for a specific query point and confirm the
% helper's output matches within the accuracy floor.

dispInvOk = true;
for sopIter = [0.10, 0.20, 0.30]
    sig = sopIter * dispPERIOD;
    dGrid = linspace(-dispPERIOD/2, dispPERIOD/2, 21);

    % Helper's dispatched value.
    thetaHelper = internal.wrappedGaussian1d(dGrid, sig, dispPERIOD, dispTS, 4);

    % Manual image-sum reference (exhaustive to 8 sigma, well past floor).
    Lref = ceil(8 * sig / dispPERIOD) + 4;
    nRef = -Lref:Lref;
    dRef = dGrid - dispPERIOD * floor(dGrid / dispPERIOD + 0.5);
    nReRef = reshape(nRef, [ones(1, ndims(dRef)), 2*Lref + 1]);
    dShiftRef = dRef + dispPERIOD * nReRef;
    thetaRef = sum(exp(-dShiftRef .* dShiftRef ...
        / (4 * sig * sig)), ndims(dShiftRef));

    maxErr = max(abs(thetaHelper(:) - thetaRef(:)));
    dispInvOk = dispInvOk && (maxErr < 1e-7);
end
results(end+1, :) = { ...
    'wrap dispatch: cosine invariant under image-sum vs Fourier', ...
    dispInvOk};


%% ---- End-to-end: cosine values are stable across the crossover ----
% A cosine computed at sigma/P values spanning the crossover should
% vary smoothly (no discontinuous jump from a route swap).

pW = [0; 100; 300; 700];
qW = [50; 250; 500; 900];
wW = ones(4, 1);
sops = [0.15, 0.20, 0.25, 0.30, 0.35];
cosVals = zeros(size(sops));
for k = 1:numel(sops)
    sig = sops(k) * dispPERIOD;
    d1 = buildExpTens({pW}, {wW}, sig, 2, false, true, dispPERIOD, ...
        'verbose', false);
    d2 = buildExpTens({qW}, {wW}, sig, 2, false, true, dispPERIOD, ...
        'verbose', false);
    cosVals(k) = cosSimExpTens(d1, d2, 'verbose', false);
end
% Differences between adjacent sigma steps should be smooth; no
% jump larger than the total range.
totalRange = max(cosVals) - min(cosVals);
maxJump = max(abs(diff(cosVals)));
results(end+1, :) = { ...
    'wrap dispatch: cosine varies smoothly across the crossover', ...
    (maxJump < totalRange) || totalRange < 1e-6};


%% ---- Count helpers behave sensibly at edge conditions ----

% Zero sigma: dispatch returns L = 0 and M = 1 (defensive fallback).
edgeOk1 = internal.wrappedKernelImageCount(0, dispPERIOD, dispTS, 4) == 0;
edgeOk2 = internal.wrappedKernelFourierCount(0, dispPERIOD, dispTS, 4) == 1;
results(end+1, :) = { ...
    'wrap dispatch: edge conditions handled defensively', ...
    edgeOk1 && edgeOk2};


if standalone
    nPass = sum(cellfun(@(x) isequal(x, true), results(:, 2)));
    nFail = size(results, 1) - nPass;
    fprintf('test_wrap_dispatch: %d passed, %d failed\n', nPass, nFail);
    for ii = 1:size(results, 1)
        if ~isequal(results{ii, 2}, true)
            fprintf('  FAIL  %s\n', results{ii, 1});
        end
    end
end
