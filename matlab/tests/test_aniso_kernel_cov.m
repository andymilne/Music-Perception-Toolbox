%% test_aniso_kernel_cov.m — Matrix-valued (anisotropic) kernel covariances
%
%  Mirror of Python tests/test_aniso_kernel_cov.py. Covers
%  kernelCov (the constructor), the whitening implementation in
%  buildMaet / evalMaet / simMaet / entropyMaet, the raw
%  sliding-comparison path (windowedSimilarity), the mode-constraint
%  error paths, and cross-language golden values generated from the
%  Python implementation.
%
%  Standalone-runnable; appends to `results` when called from
%  test_mpt.m.

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    clear cleanupDefaults
    cleanupDefaults = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end

TOL = 1e-12;

% =====================================================================
%  kernelCov: algebraic structure and error paths
% =====================================================================

rK = 4; sp = 0.3; si = 0.7; ss = 1.9;
Sigma4 = kernelCov(rK, 'differenced', true, 'sdValue', sp, 'sdInterval', si, ...
    'sdShift', ss);
ddt = 2 * eye(rK) - diag(ones(rK - 1, 1), 1) - diag(ones(rK - 1, 1), -1);
expected4 = sp^2 * ddt + si^2 * eye(rK) + ss^2 * ones(rK);
results{end+1,1} = 'aniso: kernelCov structure (DDt + I + ones)';
results{end,2}   = max(abs(Sigma4 - expected4), [], 'all') == 0;

% The tridiagonal term is literally D * D' for the first-differencing map.
r5 = 5;
D = zeros(r5, r5 + 1);
for i = 1:r5, D(i, i) = -1; D(i, i + 1) = 1; end
results{end+1,1} = 'aniso: DDt term equals differencing-map Gram matrix';
results{end,2}   = max(abs(kernelCov(r5, 'differenced', true, 'sdValue', 1.0) ...
    - D * D'), [], 'all') < 1e-14;

results{end+1,1} = 'aniso: kernelCov rejects r = 0';
results{end,2}   = errorMessageContains( ...
    @() kernelCov(0, 'differenced', true, 'sdInterval', 1.0), 'positive integer');

results{end+1,1} = 'aniso: kernelCov rejects r = 1 (scalar parity)';
results{end,2}   = errorMessageContains( ...
    @() kernelCov(1, 'differenced', true, 'sdShift', 2.0), 'indistinguishable');

results{end+1,1} = 'aniso: kernelCov rejects negative sd';
results{end,2}   = errorMessageContains( ...
    @() kernelCov(3, 'differenced', true, 'sdInterval', -1.0), 'non-negative');

results{end+1,1} = 'aniso: kernelCov rejects infinite sdShift';
results{end,2}   = errorMessageContains( ...
    @() kernelCov(3, 'differenced', true, 'sdShift', Inf), 'isRel');

results{end+1,1} = 'aniso: kernelCov rejects rank-one (shift alone)';
results{end,2}   = errorMessageContains( ...
    @() kernelCov(3, 'differenced', true, 'sdShift', 1.0), 'singular');


% --- the undifferenced case ---
results{end+1,1} = 'aniso: kernelCov requires the differenced flag';
results{end,2}   = errorMessageContains( ...
    @() kernelCov(3, 'sdInterval', 1.0), 'differenced');

% Positions: sdValue^2 I + sdInterval^2 P S S' P + sdShift^2 J, the
% walk centred on the tuple's mean.
S4 = tril(ones(rK, rK - 1), -1);
P4 = eye(rK) - ones(rK) / rK;
expectedU = sp^2 * eye(rK) + si^2 * (P4 * (S4 * S4') * P4) + ss^2 * ones(rK);
SigmaU = kernelCov(rK, 'differenced', false, 'sdValue', sp, ...
    'sdInterval', si, 'sdShift', ss);
results{end+1,1} = 'aniso: kernelCov undifferenced structure (I + PSStP + ones)';
results{end,2}   = max(abs(SigmaU - expectedU), [], 'all') < 1e-14;

walk = kernelCov(rK, 'differenced', false, 'sdInterval', 1.0, ...
    'sdShift', 1.0) - ones(rK);
results{end+1,1} = 'aniso: undifferenced walk term is centred (annihilates ones)';
results{end,2}   = max(abs(walk * ones(rK, 1))) < 1e-14;

% D Sigma_pos D' = sdValue^2 D D' + sdInterval^2 I: one model, the
% ridge falling away under D.
D5 = zeros(r5 - 1, r5);
for i = 1:r5 - 1, D5(i, i) = -1; D5(i, i + 1) = 1; end
Sp5 = kernelCov(r5, 'differenced', false, 'sdValue', 0.4, ...
    'sdInterval', 0.9, 'sdShift', 3.0);
Sd4 = kernelCov(r5 - 1, 'differenced', true, 'sdValue', 0.4, ...
    'sdInterval', 0.9);
results{end+1,1} = 'aniso: differenced covariance is undifferenced pushed through D';
results{end,2}   = max(abs(D5 * Sp5 * D5' - Sd4), [], 'all') < 1e-12;

results{end+1,1} = 'aniso: kernelCov undifferenced rejects interval noise alone';
results{end,2}   = errorMessageContains( ...
    @() kernelCov(3, 'differenced', false, 'sdInterval', 1.0), 'singular');
okAlone = true;
try
    kernelCov(3, 'differenced', false, 'sdValue', 1.0);
    kernelCov(3, 'differenced', false, 'sdInterval', 1.0, 'sdShift', 0.1);
catch
    okAlone = false;
end
results{end+1,1} = 'aniso: kernelCov undifferenced accepts position alone, interval+shift';
results{end,2}   = okAlone;

% =====================================================================
%  Scalar reduction: Sigma = sigma^2 I reproduces scalar-sigma machinery
% =====================================================================

pS = [0; 3; -3]; wS = [1; 0.8; 0.6]; sgS = 1.3;
densMat = buildMaet(pS, wS, sgS^2 * eye(3), 3, false, false, 0, false, ...
    'verbose', false);
densSca = buildMaet(pS, wS, sgS, 3, false, false, 0, false, ...
    'verbose', false);
rng(20260709);
Xq = randn(3, 12) * 3;
ok = true;
for nrm = {'none', 'gaussian', 'pdf'}
    vM = evalMaet(densMat, Xq, nrm{1});
    vS = evalMaet(densSca, Xq, nrm{1});
    ok = ok && max(abs(vM - vS)) <= TOL * max(1, max(abs(vS)));
end
results{end+1,1} = 'aniso: eval reduces to scalar sigma at Sigma = s^2 I';
results{end,2}   = ok;

qS = [0.2; 3.4; -2.9];
vM = simMaet(pS, wS, qS, wS, sgS^2 * eye(3), 3, false, false, 0, ...
    false, 'verbose', false);
vS = simMaet(pS, wS, qS, wS, sgS, 3, false, false, 0, false, ...
    'verbose', false);
results{end+1,1} = 'aniso: cosine reduces to scalar sigma at Sigma = s^2 I';
results{end,2}   = abs(vM - vS) <= TOL;

hM = entropyMaet(densMat, 'method', 'renyi2', 'base', exp(1), ...
    'verbose', false);
hS = entropyMaet(densSca, 'method', 'renyi2', 'base', exp(1), ...
    'verbose', false);
results{end+1,1} = 'aniso: renyi2 reduces to scalar sigma at Sigma = s^2 I';
results{end,2}   = abs(hM - hS) <= 1e-9;

% D==3 differential: pin a feasible accuracy on both sides (the tightest
% accuracy would need an infeasible 3-D grid). Both use the same pin, so
% the anisotropic-vs-scalar equivalence is compared like with like.
hM = entropyMaet(densMat, 'method', 'differential', 'base', exp(1), ...
    'truncationSigmas', 4, 'verbose', false);
hS = entropyMaet(densSca, 'method', 'differential', 'base', exp(1), ...
    'truncationSigmas', 4, 'verbose', false);
results{end+1,1} = 'aniso: differential reduces to scalar sigma at Sigma = s^2 I';
results{end,2}   = abs(hM - hS) <= 1e-6 * max(1, abs(hS));

% =====================================================================
%  Whitened machinery vs direct anisotropic computation
% =====================================================================

SigmaW = [0.5, 0.1, -0.05; 0.1, 0.4, 0.08; -0.05, 0.08, 0.6];
pW = [1.0; -0.5; 2.0]; wW = [1.0; 0.7; 0.9];
densW = buildMaet(pW, wW, SigmaW, 3, false, false, 0, false, ...
    'verbose', false);
Xw = randn(3, 20);
got = evalMaet(densW, Xw, 'none');
SinvW = inv(SigmaW);
dW = Xw - pW;
want = prod(wW) * exp(-0.5 * sum(dW .* (SinvW * dW), 1));
results{end+1,1} = 'aniso: eval equals direct anisotropic kernel (none)';
results{end,2}   = max(abs(got(:) - want(:))) <= TOL * max(1, max(abs(want)));

got = evalMaet(densW, Xw, 'gaussian');
constW = (2 * pi)^(-3/2) * det(SigmaW)^(-1/2);
want = constW * prod(wW) * exp(-0.5 * sum(dW .* (SinvW * dW), 1));
results{end+1,1} = 'aniso: eval gaussian normalization carries det(Sigma)^{-1/2}';
results{end,2}   = max(abs(got(:) - want(:))) <= TOL * max(1, max(abs(want)));

% 'pdf' integrates to 1 in original coordinates (2-D grid check).
Sigma2 = [0.09, 0.05; 0.05, 0.16];
dens2 = buildMaet([0.3; -0.2], [1; 1], Sigma2, 2, false, false, 0, ...
    false, 'verbose', false);
g = linspace(-3, 3, 301);
[GX, GY] = ndgrid(g, g);
vals2 = evalMaet(dens2, [GX(:).'; GY(:).'], 'pdf');
results{end+1,1} = 'aniso: pdf normalization integrates to 1';
results{end,2}   = abs(sum(vals2) * (g(2) - g(1))^2 - 1) < 1e-6;

% =====================================================================
%  Cross-language golden values (generated by the Python implementation)
% =====================================================================

SigmaG = kernelCov(3, 'differenced', true, 'sdValue', 0.4, 'sdInterval', 0.2, ...
    'sdShift', 0.6);
PXg = [0.0, 0.2, -1.0; 1.0, 1.1, 0.0; 0.5, 0.4, 2.0];
PYg = [0.1, 2.0; 0.9, -1.0; 0.55, 0.3];
gotG = simMaet({PXg}, {ones(3, 3)}, {PYg}, {ones(3, 2)}, ...
    {SigmaG}, 3, false, false, 0, false, 'verbose', false);
results{end+1,1} = 'aniso: golden MA cosine matches Python (<= 1e-12)';
results{end,2}   = abs(gotG - 0.63538792128952482) <= 1e-12;

pG = [1.0; -0.5; 2.0]; wG = [1.0; 0.7; 0.9];
densG = buildMaet(pG, wG, SigmaG, 3, false, false, 0, false, ...
    'verbose', false);
XG = [1.1, 0.0; -0.4, 0.5; 1.9, -1.0];
gotE = evalMaet(densG, XG, 'gaussian');
goldE = [0.077035468315033231, 1.1710081311248932e-05];
results{end+1,1} = 'aniso: golden eval matches Python (<= 1e-12 rel)';
results{end,2}   = max(abs(gotE(:).' - goldE) ./ max(abs(goldE), eps)) <= 1e-12;

gotH = entropyMaet(densG, 'method', 'renyi2', 'base', exp(1), ...
    'verbose', false);
results{end+1,1} = 'aniso: golden renyi2 matches Python (<= 1e-12)';
results{end,2}   = abs(gotH - 3.1056560434942622) <= 1e-12;

% =====================================================================
%  Shift-ridge limits: convergence to exact relative mode
% =====================================================================

P1 = [0.0; 0.35; 0.15];   % log-IOI triple
P2 = P1 + 0.9;            % same shape, common shift 0.9
P3 = [0.0; 0.30; 0.35];   % different shape
w3 = ones(3, 1);
relTarget = simMaet(P1, w3, P3, w3, 0.1, 3, true, false, 0, false, ...
    'verbose', false);
results{end+1,1} = 'aniso: rel-mode target matches Python golden';
results{end,2}   = abs(relTarget - 0.4168620196785085) <= 1e-12;

prevErr = Inf; monotone = true;
for ssv = [1, 10, 100]
    Sg = kernelCov(3, 'differenced', true, 'sdInterval', 0.1, 'sdShift', ssv);
    v = simMaet(P1, w3, P3, w3, Sg, 3, false, false, 0, false, ...
        'verbose', false);
    err = abs(v - relTarget);
    monotone = monotone && (err < prevErr);
    prevErr = err;
end
results{end+1,1} = 'aniso: ridge cosine converges monotonically to rel mode';
results{end,2}   = monotone && prevErr < 1e-3;

SgNo = kernelCov(3, 'differenced', true, 'sdInterval', 0.1, 'sdShift', 0);
SgHi = kernelCov(3, 'differenced', true, 'sdInterval', 0.1, 'sdShift', 5);
vNo = simMaet(P1, w3, P2, w3, SgNo, 3, false, false, 0, false, ...
    'verbose', false);
vHi = simMaet(P1, w3, P2, w3, SgHi, 3, false, false, 0, false, ...
    'verbose', false);
results{end+1,1} = 'aniso: ridge grades a common shift (0 penalizes, large forgives)';
results{end,2}   = vNo < 1e-6 && vHi > 0.9;

% =====================================================================
%  Entropy closed forms (single Gaussian, log det term)
% =====================================================================

SigmaH = [0.3, 0.05, -0.02; 0.05, 0.25, 0.04; -0.02, 0.04, 0.35];
densH = buildMaet([0; 1; -1], ones(3, 1), SigmaH, 3, false, false, 0, ...
    false, 'verbose', false);
gotH2 = entropyMaet(densH, 'method', 'renyi2', 'base', exp(1), ...
    'verbose', false);
wantH2 = 1.5 * log(4 * pi) + 0.5 * log(det(SigmaH));
results{end+1,1} = 'aniso: renyi2 single-Gaussian closed form incl. log det';
results{end,2}   = abs(gotH2 - wantH2) <= 1e-10;

SigmaD = [0.04, -0.01; -0.01, 0.09];
densD = buildMaet([0; 0.5], ones(2, 1), SigmaD, 2, false, false, 0, ...
    false, 'verbose', false);
% D==2 differential: this entropy is near zero, so the closed-form check
% is sensitive and needs a tight (but still feasible) accuracy.
gotHd = entropyMaet(densD, 'method', 'differential', 'base', exp(1), ...
    'truncationSigmas', 6, 'verbose', false);
wantHd = log(2 * pi * exp(1)) + 0.5 * log(det(SigmaD));
results{end+1,1} = 'aniso: differential single-Gaussian closed form incl. log det';
results{end,2}   = abs(gotHd - wantHd) <= 1e-4 * max(1, abs(wantHd));

% =====================================================================
%  windowedSimilarity end-to-end (ordered aniso attribute, time sweep)
% =====================================================================

rP = 2;
SigmaP = kernelCov(rP, 'differenced', true, 'sdInterval', 0.05, 'sdShift', 5.0);
shapes = [0.00, 0.30, 0.20, 0.50, 0.05; ...
          0.40, 0.10, 0.20, 0.00, 0.45];
shapes(:, 3) = shapes(:, 3) + 0.9;   % query shape at a common shift
onsets = 0:4;
pCtx = {shapes, onsets};
wCtx = {ones(rP, 5), ones(1, 5)};
pQry = {[-0.70; -0.70], 0};
wQry = {ones(rP, 1), 1};
prof = windowedSimilarity(pCtx, wCtx, pQry, wQry, ...
    {SigmaP, 0.25}, [rP, 1], [false, false], [false, false], [0, 0], ...
    onsets, 'isExch', [false, true], 'windowAttr', 2, ...
    'dropWindowAttr', true, 'contextWindow', {1.0, 0.5}, ...
    'normalize', 'oneSidedDenom', 'verbose', false);
[~, peakIdx] = max(prof);
results{end+1,1} = 'aniso: windowed sweep peaks at the shifted match';
results{end,2}   = numel(prof) == 5 && peakIdx == 3 && prof(3) > 0.9;

% Manual off-peak check: with the time axis dropped, step n is the
% plain one-sided single multiset kernel of the anisotropic pairs.
dOff = shapes(:, 1) - pQry{1};
wantOff = exp(-0.25 * dOff' * (SigmaP \ dOff));
results{end+1,1} = 'aniso: windowed off-peak value matches direct kernel';
results{end,2}   = abs(prof(1) - wantOff) <= 1e-10;

% =====================================================================
%  Guards and constraint error paths
% =====================================================================

pC = [0; 1; 2]; wC = ones(3, 1);

results{end+1,1} = 'aniso: rejects symmetric multiset (isExch = true)';
results{end,2}   = errorMessageContains(@() buildMaet(pC, wC, ...
    eye(3), 3, false, false, 0, true, 'verbose', false), 'ordered multiset');

results{end+1,1} = 'aniso: rejects isRel = true';
results{end,2}   = errorMessageContains(@() buildMaet(pC, wC, ...
    eye(3), 3, true, false, 0, false, 'verbose', false), 'isRel = false');

results{end+1,1} = 'aniso: rejects isPer = true';
results{end,2}   = errorMessageContains(@() buildMaet(pC, wC, ...
    eye(3), 3, false, true, 12, false, 'verbose', false), 'isPer = false');

results{end+1,1} = 'aniso: rejects r < K';
results{end,2}   = errorMessageContains(@() buildMaet(pC, wC, ...
    eye(2), 2, false, false, 0, false, 'verbose', false), 'r == K');

results{end+1,1} = 'aniso: rejects wrong covariance size';
results{end,2}   = errorMessageContains(@() buildMaet(pC, wC, ...
    eye(4), 3, false, false, 0, false, 'verbose', false), 'tuple dimension');

Sasym = eye(3); Sasym(1, 2) = 0.5;
results{end+1,1} = 'aniso: rejects asymmetric covariance';
results{end,2}   = errorMessageContains(@() buildMaet(pC, wC, ...
    Sasym, 3, false, false, 0, false, 'verbose', false), 'symmetric');

Sneg = eye(3); Sneg(1, 1) = -1;
results{end+1,1} = 'aniso: rejects indefinite covariance';
results{end,2}   = errorMessageContains(@() buildMaet(pC, wC, ...
    Sneg, 3, false, false, 0, false, 'verbose', false), 'positive definite');

Sinf = eye(3); Sinf(2, 2) = Inf;
results{end+1,1} = 'aniso: rejects non-finite covariance';
results{end,2}   = errorMessageContains(@() buildMaet(pC, wC, ...
    Sinf, 3, false, false, 0, false, 'verbose', false), 'finite');

results{end+1,1} = 'aniso: rejects NaN values under matrix sigma';
results{end,2}   = errorMessageContains(@() buildMaet([0; NaN; 2], wC, ...
    eye(3), 3, false, false, 0, false, 'verbose', false), 'NaN');

d1 = buildMaet(pC, wC, eye(3), 3, false, false, 0, false, 'verbose', false);
d2 = buildMaet(pC, wC, 2 * eye(3), 3, false, false, 0, false, ...
    'verbose', false);
results{end+1,1} = 'aniso: cosine rejects mismatched covariances';
results{end,2}   = errorMessageContains(@() simMaet(d1, d2, ...
    'verbose', false), 'kernel');

d3 = buildMaet(pC, wC, 1.0, 3, false, false, 0, false, 'verbose', false);
results{end+1,1} = 'aniso: cosine rejects cov-vs-scalar density pair';
results{end,2}   = errorMessageContains(@() simMaet(d1, d3, ...
    'verbose', false), 'kernel');

% Ordered K-tuple with diagonal covariance == bound singleton attributes.
s1 = 0.4; s2 = 0.9;
PX2 = [0.0, 1.0; 2.0, 3.0];
PY2 = [0.1, 0.8; 2.2, 2.9];
vA = simMaet({PX2}, {ones(2, 2)}, {PY2}, {ones(2, 2)}, ...
    {diag([s1^2, s2^2])}, 2, false, false, 0, false, 'verbose', false);
vT = simMaet({PX2(1, :), PX2(2, :)}, {ones(1, 2), ones(1, 2)}, ...
    {PY2(1, :), PY2(2, :)}, {ones(1, 2), ones(1, 2)}, ...
    [s1, s2], [1, 1], [false, false], [false, false], [0, 0], ...
    [true, true], 'verbose', false);
results{end+1,1} = 'aniso: diagonal cov ordered pair == bound singleton attrs';
results{end,2}   = abs(vA - vT) <= 1e-12;

% =====================================================================

% =====================================================================
%  Degenerate nested flattening (bindEvents over flat single-value
%  events + matrix sigma; v3+)
% =====================================================================

rngSeed = RandStream('mt19937ar', 'Seed', 7);
xDeg = rand(rngSeed, 1, 12);
[pbD, ~, spD] = unpackPreMaet(bindEvents({xDeg}, [], 3));
PDeg = pbD{1};
nDeg = size(PDeg, 2);
flatDeg = struct('r', 3, 'exch', false, 'rel', false);
SigDeg = kernelCov(3, 'differenced', true, 'sdValue', 0.07, 'sdShift', 0.2);

densNest = buildMaet({PDeg}, {ones(3, nDeg)}, 'specs', {spD{1}}, ...
    'sigma', {SigDeg}, 'isPer', false, 'period', 0, 'verbose', false);
densFlat = buildMaet({PDeg}, {ones(3, nDeg)}, 'specs', {flatDeg}, ...
    'sigma', {SigDeg}, 'isPer', false, 'period', 0, 'verbose', false);
ptsDeg = rand(rngSeed, 3, 6) - 0.5;
evNest = evalMaet(densNest, ptsDeg, 'verbose', false);
evFlat = evalMaet(densFlat, ptsDeg, 'verbose', false);
results{end+1,1} = 'aniso: degenerate bound == flat, eval, matrix sigma';
results{end,2}   = max(abs(evNest(:) - evFlat(:))) <= TOL;

cosNF = simMaet(densNest, densFlat, 'verbose', false);
results{end+1,1} = 'aniso: degenerate bound == flat, cosine, matrix sigma';
results{end,2}   = abs(cosNF - 1) <= TOL;

densNestS = buildMaet({PDeg}, {ones(3, nDeg)}, 'specs', {spD{1}}, ...
    'sigma', {0.3}, 'isPer', false, 'period', 0, 'verbose', false);
densFlatS = buildMaet({PDeg}, {ones(3, nDeg)}, 'specs', {flatDeg}, ...
    'sigma', {0.3}, 'isPer', false, 'period', 0, 'verbose', false);
evNestS = evalMaet(densNestS, ptsDeg, 'verbose', false);
evFlatS = evalMaet(densFlatS, ptsDeg, 'verbose', false);
results{end+1,1} = 'aniso: degenerate bound == flat, scalar-sigma baseline';
results{end,2}   = max(abs(evNestS(:) - evFlatS(:))) <= TOL;

% Demo pipeline: difference -> log -> bind, swept via specs, against
% the manually stacked flat surface via isExch.
onsW = [0, 0.5, 0.75, 1.0, 2.0, 2.5, 2.75, 3.0, 4.0, 4.4, 4.6, 4.8];
[pDf, wDf, spDf] = unpackPreMaet(differenceEvents({onsW}, [], 1));
pDf{1} = log(pDf{1});
[pBf, wBf, spBf] = unpackPreMaet(bindEvents(pDf, wDf, 3, 'specs', spDf));
nTriW = size(pBf{1}, 2);
triTimesW = onsW(1:nTriW);
liW = log(diff(onsW));
triManualW = [liW(1:nTriW); liW(2:nTriW+1); liW(3:nTriW+2)];
results{end+1,1} = 'aniso: bound pipeline values equal manual stacking';
results{end,2}   = isequal(pBf{1}, triManualW);

qW = log([0.5; 0.25; 0.25]);
wCtxW = {ones(3, nTriW), ones(1, nTriW)};
pQW = {qW, 0};
wQW = {ones(3, 1), 1};
tspW = struct('r', 1, 'exch', true, 'rel', false);
profSpecs = windowedSimilarity({pBf{1}, triTimesW}, wCtxW, pQW, wQW, ...
    {SigDeg, 0.25}, [3, 1], [false, false], [false, false], [0, 0], ...
    triTimesW, 'specs', {spBf{1}, tspW}, 'windowAttr', 2, ...
    'dropWindowAttr', true, 'contextWindow', {'rect', 0.1}, ...
    'normalize', 'oneSidedDenom', 'verbose', false);
profFlatW = windowedSimilarity({triManualW, triTimesW}, wCtxW, pQW, wQW, ...
    {SigDeg, 0.25}, [3, 1], [false, false], [false, false], [0, 0], ...
    triTimesW, 'isExch', [false, true], 'windowAttr', 2, ...
    'dropWindowAttr', true, 'contextWindow', {'rect', 0.1}, ...
    'normalize', 'oneSidedDenom', 'verbose', false);
results{end+1,1} = 'aniso: windowed bound-specs equals flat-isExch profile';
results{end,2}   = max(abs(profSpecs(:) - profFlatW(:))) <= TOL;

% specs form with a matrix sigma on a *flat* spec (previously
% blanket-rejected) must match the positional form.
densPos = buildMaet({PDeg}, {ones(3, nDeg)}, {SigDeg}, 3, false, ...
    false, 0, false, 'verbose', false);
cosSP = simMaet(densFlat, densPos, 'verbose', false);
results{end+1,1} = 'aniso: flat spec + matrix sigma via specs == positional';
results{end,2}   = abs(cosSP - 1) <= TOL;

% Non-degenerate nesting (K = 2 constituents) is rejected.
x2Deg = rand(rngSeed, 2, 12);
[pb2D, ~, sp2D] = unpackPreMaet(bindEvents({x2Deg}, [], 3));
results{end+1,1} = 'aniso: non-degenerate nested rejected';
results{end,2}   = errorMessageContains(@() buildMaet({pb2D{1}}, [], ...
    'specs', {sp2D{1}}, 'sigma', {0.01 * eye(6)}, 'isPer', false, ...
    'period', 0, 'verbose', false), 'not degenerate');

% Outer-level exch/rel on a degenerate spec hit the canonical messages.
[pbSy, ~, spSy] = unpackPreMaet(bindEvents({xDeg}, [], 3, 'exchOuter', true));
results{end+1,1} = 'aniso: degenerate spec with exchOuter rejected canonically';
results{end,2}   = errorMessageContains(@() buildMaet({pbSy{1}}, [], ...
    'specs', {spSy{1}}, 'sigma', {0.01 * eye(3)}, 'isPer', false, ...
    'period', 0, 'verbose', false), 'ordered multiset');

[pbRl, ~, spRl] = unpackPreMaet(bindEvents({xDeg}, [], 3, 'relOuter', true));
results{end+1,1} = 'aniso: degenerate spec with relOuter rejected canonically';
results{end,2}   = errorMessageContains(@() buildMaet({pbRl{1}}, [], ...
    'specs', {spRl{1}}, 'sigma', {0.01 * eye(3)}, 'isPer', false, ...
    'period', 0, 'verbose', false), 'isRel = false');


% =====================================================================
%  Truncation parity: the kernel-covariance route truncates exactly as
%  the isotropic one (mirror of Python TestTruncationParity)
% =====================================================================
% A matrix-valued covariance is evaluated by whitening onto the
% isotropic unit-sigma kernel, and the truncation rule (drop a kernel
% contribution whose value falls below exp(-k^2/2) of its peak) then
% holds in the Mahalanobis metric. Regression: the whitened tuple
% (dim = r >= 2, a handful of centres) always reached the exhaustive
% branch of internal.gaussianKernelSum, which skipped the cutoff, so
% kernelCov similarities were untruncated while the equivalent paired
% (relative + absolute) attributes were truncated.

tpQ = [6000; 6200; 6400; 6700];
tpX = tpQ + 700;
tpSdRel = 30;
tpPm = @(v, sig, rel) packPreMaet(repmat({v(:)}, 1, numel(rel)), [], ...
    flatSpecs(repmat({v(:)}, 1, numel(rel)), 'r', numel(v), 'rel', rel, ...
    'exch', false(1, numel(rel)), 'sigma', sig, ...
    'isPer', false(1, numel(rel)), 'period', zeros(1, numel(rel))));

% {s, truncationSigmas, expectZero}: 3e-9 < exp(-18) at (158, 6);
% 3e-9 > 1e-12 at (158, Inf); 5e-22 and 2e-54 < 1e-12; 4e-3 above both.
tpCases = {158, 6, true; 158, Inf, false; 100, Inf, true; ...
           63, Inf, true; 300, 6, false};
for iCase = 1:size(tpCases, 1)
    s = tpCases{iCase, 1}; ts = tpCases{iCase, 2};
    expectZero = tpCases{iCase, 3};
    vs = tpSdRel^2 + s^2;
    Ctp = kernelCov(4, 'differenced', false, ...
        'sdValue', tpSdRel * s / sqrt(vs), 'sdShift', s^2 / sqrt(4 * vs));
    a = simMaet(tpPm(tpQ, {tpSdRel, s}, [true false]), ...
                tpPm(tpX, {tpSdRel, s}, [true false]), ...
                'truncationSigmas', ts, 'verbose', false);
    b = simMaet(tpPm(tpQ, {Ctp}, false), tpPm(tpX, {Ctp}, false), ...
                'truncationSigmas', ts, 'verbose', false);
    if expectZero
        ok = (a == 0) && (b == 0);
    else
        ok = (a > 0) && abs(b - a) <= 1e-12 * abs(a);
    end
    results{end+1,1} = sprintf(['aniso: pairing equals kernelCov under ' ...
        'truncation (s = %g, truncationSigmas = %g)'], s, ts); %#ok<SAGROW>
    results{end,2}   = ok;
end

% Sigma = s^2 I against scalar sigma, under default-width truncation.
for nrm = {'cosine', 'oneSidedDenom', 'none'}
    for sCase = {158, true; 300, false}.'
        s = sCase{1}; expectZero = sCase{2};
        a = simMaet(tpPm(tpQ, {s}, false), tpPm(tpX, {s}, false), ...
            'normalize', nrm{1}, 'truncationSigmas', 6, 'verbose', false);
        b = simMaet(tpPm(tpQ, {s^2 * eye(4)}, false), ...
            tpPm(tpX, {s^2 * eye(4)}, false), ...
            'normalize', nrm{1}, 'truncationSigmas', 6, 'verbose', false);
        if expectZero
            ok = (a == 0) && (b == 0);
        else
            ok = abs(b - a) <= 1e-12 * abs(a);
        end
        results{end+1,1} = sprintf(['aniso: s^2 I matches scalar sigma ' ...
            'under truncation (sim, %s, s = %g)'], nrm{1}, s); %#ok<SAGROW>
        results{end,2}   = ok;
    end
end

% Evaluation: offsets 0 and 300 inside the 6-sigma ball, 600 and 900
% outside it.
s = 158;
dTpI = buildMaet(tpPm(tpQ, {s}, false), 'verbose', false);
dTpC = buildMaet(tpPm(tpQ, {s^2 * eye(4)}, false), 'verbose', false);
XTp = tpQ + [0, 300, 600, 900];
ok = true;
for nrm = {'none', 'gaussian'}
    vI = evalMaet(dTpI, XTp, nrm{1}, 'truncationSigmas', 6, 'verbose', false);
    vC = evalMaet(dTpC, XTp, nrm{1}, 'truncationSigmas', 6, 'verbose', false);
    ok = ok && all(abs(vC - vI) <= 1e-12 * abs(vI)) ...
        && vI(3) == 0 && vI(4) == 0 && vI(2) > 0;
end
results{end+1,1} = 'aniso: s^2 I matches scalar sigma under truncation (eval)';
results{end,2}   = ok;

% The exhaustive branch of internal.gaussianKernelSum (dim = 4, one
% centre: the bucket index is not worthwhile) applies the same Q-ball
% cutoff as the bucketed one.
kTp = 6; uTp = ones(4, 1) / 2;
vTp = internal.gaussianKernelSum(zeros(4, 1), 1, ...
    [uTp * (kTp - 1e-6), uTp * (kTp + 1e-6)], 1, 'truncationSigmas', kTp);
results{end+1,1} = 'aniso: exhaustive kernel-sum branch truncates at k sigma';
results{end,2}   = abs(vTp(1) - exp(-0.5 * (kTp - 1e-6)^2)) ...
    <= 1e-12 * exp(-0.5 * (kTp - 1e-6)^2) && vTp(2) == 0;


% =====================================================================
%  Bare inner product (normalize = 'none') in the original coordinates
% =====================================================================
% Whitening x = R y carries the anisotropic kernel to the isotropic
% unit-sigma one, but the inner product is an integral, so the change of
% variables contributes the Jacobian det(Sigma)^(1/2) per attribute.
% Regression: the bare value omitted it, so Sigma = s^2 I of dimension d
% returned s^-d times the scalar-sigma value. Mirror of the Python
% TestBareInnerProductScale.

sgB = 1.7;
pB = [0; 3; -3]; qB = [0.2; 3.4; -2.9]; wB = [1; 0.8; 0.6];
ok = true;
for rB = [2, 3]
    vM = simMaet(pB(1:rB), wB(1:rB), qB(1:rB), wB(1:rB), sgB^2 * eye(rB), ...
        rB, false, false, 0, false, 'normalize', 'none', 'verbose', false);
    vS = simMaet(pB(1:rB), wB(1:rB), qB(1:rB), wB(1:rB), sgB, ...
        rB, false, false, 0, false, 'normalize', 'none', 'verbose', false);
    ok = ok && abs(vM - vS) <= TOL * abs(vS);
end
results{end+1,1} = 'aniso: none reduces to scalar sigma at Sigma = s^2 I (raw, r = 2, 3)';
results{end,2}   = ok;

dPM = buildMaet(pB, wB, sgB^2 * eye(3), 3, false, false, 0, false, 'verbose', false);
dQM = buildMaet(qB, wB, sgB^2 * eye(3), 3, false, false, 0, false, 'verbose', false);
dPS = buildMaet(pB, wB, sgB, 3, false, false, 0, false, 'verbose', false);
dQS = buildMaet(qB, wB, sgB, 3, false, false, 0, false, 'verbose', false);
vM = simMaet(dPM, dQM, 'normalize', 'none', 'verbose', false);
vS = simMaet(dPS, dQS, 'normalize', 'none', 'verbose', false);
lM = simMaet(dPM, {dQM, dPM}, 'normalize', 'none', 'verbose', false);
lS = simMaet(dPS, {dQS, dPS}, 'normalize', 'none', 'verbose', false);
if iscell(lM), lM = cell2mat(lM); lS = cell2mat(lS); end
results{end+1,1} = 'aniso: none reduces to scalar sigma at Sigma = s^2 I (density, list)';
results{end,2}   = abs(vM - vS) <= TOL * abs(vS) ...
    && all(abs(lM(:) - lS(:)) <= TOL * abs(lS(:)));

% Multi-attribute: two matrix-sigma attributes (r = 3 and r = 2) tensored
% with a scalar one; raw, raw broadcast, and density forms, plus Renyi-2.
P1x = [0.3, -1.2, 0.8; 1.1, 0.4, -0.6; -0.5, 0.9, 0.2]; P1y = [0.1; 0.7; -0.3];
P2x = [0.6, -0.2, 1.4; -0.9, 0.5, 0.0];                  P2y = [0.4; -0.1];
Tx = [0, 0.5, 1.0];                                        Ty = 0.3;
s1 = 0.9; s2 = 1.4; st = 0.4;
pxB = {P1x, P2x, Tx}; wxB = {ones(3, 3), ones(2, 3), ones(1, 3)};
pyB = {P1y, P2y, Ty}; wyB = {ones(3, 1), ones(2, 1), 1};
geomB = {[3, 2, 1], [false false false], [false false false], [0 0 0], ...
    [false false true]};
sigMat = {s1^2 * eye(3), s2^2 * eye(2), st};
sigSca = {s1, s2, st};
vM = simMaet(pxB, wxB, pyB, wyB, sigMat, geomB{:}, 'normalize', 'none', ...
    'verbose', false);
vS = simMaet(pxB, wxB, pyB, wyB, sigSca, geomB{:}, 'normalize', 'none', ...
    'verbose', false);
bM = simMaet(pxB, wxB, {pyB, pyB}, wyB, sigMat, geomB{:}, ...
    'normalize', 'none', 'verbose', false);
bS = simMaet(pxB, wxB, {pyB, pyB}, wyB, sigSca, geomB{:}, ...
    'normalize', 'none', 'verbose', false);
if iscell(bM), bM = cell2mat(bM); bS = cell2mat(bS); end
dxM = buildMaet(pxB, wxB, sigMat, geomB{:}, 'verbose', false);
dyM = buildMaet(pyB, wyB, sigMat, geomB{:}, 'verbose', false);
dxS = buildMaet(pxB, wxB, sigSca, geomB{:}, 'verbose', false);
dyS = buildMaet(pyB, wyB, sigSca, geomB{:}, 'verbose', false);
dM = simMaet(dxM, dyM, 'normalize', 'none', 'verbose', false);
dS = simMaet(dxS, dyS, 'normalize', 'none', 'verbose', false);
hM = entropyMaet(dxM, 'method', 'renyi2', 'verbose', false);
hS = entropyMaet(dxS, 'method', 'renyi2', 'verbose', false);
results{end+1,1} = 'aniso: none reduces to scalar sigma at Sigma = s^2 I (multi-attribute)';
results{end,2}   = abs(vM - vS) <= TOL * abs(vS) ...
    && all(abs(bM(:) - bS(:)) <= TOL * abs(bS(:))) ...
    && abs(dM - dS) <= TOL * abs(dS);
results{end+1,1} = 'aniso: renyi2 unchanged by the none scale (multi-attribute)';
results{end,2}   = abs(hM - hS) <= TOL * max(1, abs(hS));

% A relative r = 1 attribute takes the Renyi-2 sub-density branch; the
% log det term must still be added exactly once.
P1r = [0.3, -1.2, 0.8, 0.1; 1.1, 0.4, -0.6, 0.9];
Tr = [0, 0.3, 0.9, 1.4];
geomR = {[2, 1], [false true], [false false], [0 0], [false true]};
wR = {ones(2, 4), ones(1, 4)};
dRM = buildMaet({P1r, Tr}, wR, {0.36 * eye(2), 0.2}, geomR{:}, 'verbose', false);
dRS = buildMaet({P1r, Tr}, wR, {0.6, 0.2}, geomR{:}, 'verbose', false);
hM = entropyMaet(dRM, 'method', 'renyi2', 'verbose', false);
hS = entropyMaet(dRS, 'method', 'renyi2', 'verbose', false);
results{end+1,1} = 'aniso: renyi2 unchanged by the none scale (relative r = 1 sub-density)';
results{end,2}   = abs(hM - hS) <= TOL * max(1, abs(hS));

% Non-isotropic covariances against the direct canonical-scale inner
% product pi^(d/2) det(Sigma)^(1/2) sum W W' exp(-d' Sigma^-1 d / 4).
SigmaN = [0.5, 0.1, -0.05; 0.1, 0.4, 0.08; -0.05, 0.08, 0.6];
cxN = [0.0, -1.0; 1.0, 0.0; 0.5, 2.0];
cyN = [0.1, 2.0; 0.9, -1.0; 0.55, 0.3];
SinvN = inv(SigmaN);
wantN = 0;
for i = 1:size(cxN, 2)
    for j = 1:size(cyN, 2)
        dd = cxN(:, i) - cyN(:, j);
        wantN = wantN + exp(-0.25 * (dd' * SinvN * dd));
    end
end
wantN = pi^(3/2) * sqrt(det(SigmaN)) * wantN;
gotN = simMaet({cxN}, {ones(3, 2)}, {cyN}, {ones(3, 2)}, {SigmaN}, 3, ...
    false, false, 0, false, 'normalize', 'none', 'verbose', false);
DN = diag([0.3, 1.1, 2.5]);
ddN = cxN(:, 1) - cyN(:, 1);
wantD = pi^(3/2) * sqrt(det(DN)) * exp(-0.25 * (ddN' * (DN \ ddN)));
gotD = simMaet(cxN(:, 1), ones(3, 1), cyN(:, 1), ones(3, 1), DN, 3, ...
    false, false, 0, false, 'normalize', 'none', 'verbose', false);
results{end+1,1} = 'aniso: none equals direct anisotropic inner product (full and diagonal)';
results{end,2}   = abs(gotN - wantN) <= TOL * abs(wantN) ...
    && abs(gotD - wantD) <= TOL * abs(wantD);

% windowedSimilarity under 'none', with and without the window attribute.
shapesB = [0.0, 0.3, 0.2, 0.5; 0.4, 0.1, 0.2, 0.0];
onsetsB = [0, 1, 2, 3];
ok = true;
for dropB = [true, false]
    profB = cell(1, 2);
    sigsB = {0.09 * eye(2), 0.3};
    for k = 1:2
        profB{k} = windowedSimilarity({shapesB, onsetsB}, ...
            {ones(2, 4), ones(1, 4)}, {[0.2; 0.25], 0}, {ones(2, 1), 1}, ...
            {sigsB{k}, 0.25}, [2, 1], [false, false], [false, false], [0, 0], ...
            onsetsB, 'isExch', [false, true], 'windowAttr', 2, ...
            'dropWindowAttr', dropB, 'contextWindow', {'rect', 0.5}, ...
            'normalize', 'none', 'verbose', false);
    end
    ok = ok && all(abs(profB{1}(:) - profB{2}(:)) <= TOL * abs(profB{2}(:)));
end
results{end+1,1} = 'aniso: windowedSimilarity none reduces to scalar sigma at Sigma = s^2 I';
results{end,2}   = ok;


if standalone
    nFail = sum(~[results{:,2}]);
    fprintf('\ntest_aniso_kernel_cov: %d/%d passed.\n', ...
        size(results, 1) - nFail, size(results, 1));
    for i = 1:size(results, 1)
        if ~results{i,2}, fprintf('  FAILED: %s\n', results{i,1}); end
    end
    if nFail > 0
        error('test_aniso_kernel_cov:failed', '%d test(s) failed.', nFail);
    end
end
