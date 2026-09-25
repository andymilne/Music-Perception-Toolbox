%% test_sweep_kernel_cov.m — translation sweeps with a kernel covariance
%
%  A kernel covariance is carried on whitened values at sigma = 1, so an
%  attribute that is not swept enters the mixture's fixed weight
%  exactly, while a swept one is refused (a uniform translation of the
%  original values is not uniform in the whitened coordinates). Every
%  test compares the mixture route against the per-offset path it
%  replaces: translate the query explicitly, call simMaet, and require
%  agreement. Mirrors python/tests/test_sweep_kernel_cov.py.

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    clear cleanupDefaults
    cleanupDefaults = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end

tol = 1e-12;
baseOff = [-3.1, -1.4, -0.35, 0.0, 0.8, 2.2, 5.0];
off = [baseOff; zeros(1, numel(baseOff))];

Sig2 = [1.0 0.4; 0.4 0.7];
Sig3 = [1.0 0.3 0.1; 0.3 0.9 0.2; 0.1 0.2 1.2];
% {K0, r0, isExch0, Sigma}: attribute 1 isotropic and swept; attribute 2
% carries the covariance and is not swept.
cases = {{3, 2, true, Sig2}, {2, 2, false, Sig2}, {2, 2, false, Sig3}};
norms = {'cosine', 'oneSidedDenom'};
widths = {Inf, []};

for ci = 1:numel(cases)
    K0 = cases{ci}{1}; r0 = cases{ci}{2}; ex0 = cases{ci}{3};
    Sig = cases{ci}{4}; d1 = size(Sig, 1);
    rng(200 + 10 * K0 + d1);
    pX = {randn(K0, 6) * 3, randn(d1, 6) * 1.5};
    pY = {randn(K0, 4) * 3, randn(d1, 4) * 1.5};
    sigma = {0.9, Sig};
    geom = {[r0 d1], [false false], [false false], [0 0], [ex0 false]};
    dX = buildMaet(pX, [], sigma, geom{:}, 'verbose', false);
    dY = buildMaet(pY, [], sigma, geom{:}, 'verbose', false);
    for ni = 1:numel(norms)
        for wi = 1:numel(widths)
            ts = widths{wi};
            ref = zeros(1, size(off, 2));
            for m = 1:size(off, 2)
                dYm = buildMaet({pY{1} + off(1, m), pY{2}}, [], sigma, ...
                    geom{:}, 'verbose', false);
                ref(m) = simMaet(dX, dYm, 'method', 'bulger', ...
                    'normalize', norms{ni}, 'truncationSigmas', ts, ...
                    'verbose', false);
            end
            for meth = {'mixture', 'auto'}
                got = sweepSimMaet(dX, dY, off, 'method', meth{1}, ...
                    'normalize', norms{ni}, 'truncationSigmas', ts, ...
                    'verbose', false);
                dev = max(abs(got - ref)) / max(abs(ref));
                results{end+1,1} = sprintf( ...
                    ['sweep.kernelCov: unswept %dx%d covariance, K=%d ' ...
                     'r=%d exch=%d, %s, ts=%s, %s matches per-offset'], ...
                    d1, d1, K0, r0, ex0, norms{ni}, mat2str(ts), meth{1}); %#ok<*SAGROW>
                results{end,2} = dev <= tol;
            end
        end
    end
end

% --- sigma^2 I matches the scalar sigma --------------------------------
rng(4);
pX = {randn(3, 6) * 3, randn(2, 6) * 1.5};
pY = {randn(3, 4) * 3, randn(2, 4) * 1.5};
geom = {[3 2], [false false], [false false], [0 0], [true false]};
s = 0.9;
vals = cell(1, 2);
sigs = {{0.9, s^2 * eye(2)}, {0.9, s}};
for k = 1:2
    dX = buildMaet(pX, [], sigs{k}, geom{:}, 'verbose', false);
    dY = buildMaet(pY, [], sigs{k}, geom{:}, 'verbose', false);
    vals{k} = sweepSimMaet(dX, dY, off, 'method', 'mixture', ...
        'truncationSigmas', Inf, 'verbose', false);
end
results{end+1,1} = 'sweep.kernelCov: sigma^2 I equals scalar sigma';
results{end,2} = max(abs(vals{1} - vals{2})) / max(abs(vals{2})) <= tol;

% --- A swept attribute with a covariance is refused ---------------------
rng(5);
pX = {randn(3, 6) * 3, randn(2, 6) * 1.5};
pY = {randn(3, 4) * 3, randn(2, 4) * 1.5};
geom = {[2 2], [false false], [false false], [0 0], [true false]};
dX = buildMaet(pX, [], {0.9, Sig2}, geom{:}, 'verbose', false);
dY = buildMaet(pY, [], {0.9, Sig2}, geom{:}, 'verbose', false);
offSwept = [zeros(1, numel(baseOff)); baseOff];
results{end+1,1} = 'sweep.kernelCov: swept covariance attribute refused by the mixture';
results{end,2} = throwsErrorWithId(@() sweepSimMaet(dX, dY, offSwept, ...
    'method', 'mixture', 'verbose', false), 'sweepSimMaet:anisotropicKernel');

% --- Mismatched covariances are refused ---------------------------------
dY2 = buildMaet(pY, [], {0.9, 2 * Sig2}, geom{:}, 'verbose', false);
okMis = true;
for meth = {'auto', 'mixture'}
    okMis = okMis && throwsErrorWithId(@() sweepSimMaet(dX, dY2, off, ...
        'method', meth{1}, 'verbose', false), 'sweepSimMaet:kernelCovMismatch');
end
results{end+1,1} = 'sweep.kernelCov: mismatched covariances refused';
results{end,2} = okMis;

%% ---- Standalone summary ----

if standalone
    nPass = sum(cellfun(@(x) isequal(x, true), results(:,2)));
    nFail = numel(results(:,1)) - nPass;
    fprintf('\n=== test_sweep_kernel_cov: %d passed, %d failed (of %d) ===\n', ...
        nPass, nFail, numel(results(:,1)));
    clear cleanupDefaults
    if nFail > 0
        for ii = 1:size(results, 1)
            if ~isequal(results{ii, 2}, true)
                fprintf('  FAIL  %s\n', results{ii, 1});
            end
        end
    end
end
