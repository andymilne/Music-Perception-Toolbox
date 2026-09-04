%% test_nested_rel_per_full_image.m — the nested tau-grid contraction averages the wrapped Gaussian
%
%  Mirror of the Python tests/test_nested_rel_per_full_image.py.
%
%  The relative-periodic contraction is the all-image transposition
%  average: the lattice-sum measure the flat Moebius integrator computes,
%  which sums periodic images before averaging over tau. Both the batched
%  contraction and the per-pair reference nestedIp used to average the
%  *nearest-image* Gaussian instead -- a different measure once the accuracy
%  floor asks for more than one image (1.4e-5 in the cosine at
%  sigma/P = 0.1, 1.8e-3 at 0.2), and one whose kink at |d| = P/2 also cost
%  the trapezoidal rule its spectral convergence.
%
%  The reference is independent of the contraction: the nested tuple set
%  is enumerated explicitly and the transposition average taken on a
%  6000-node grid, once with the wrapped Gaussian per coordinate and once
%  with the nearest-image Gaussian, so the test says which measure the
%  route computes rather than only that it changed. method = 'contract'
%  forces the contraction, which for a relative-periodic attribute is the
%  tau-grid.
%
%  Standalone-runnable; appends to `results` when called from test_mpt.m.
%  mptTestIsolateDefaults sets truncationSigmas = Inf (accuracy floor).

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    addpath(fileparts(fileparts(mfilename('fullpath'))));
    clear cleanupDefaults_nrpf
    cleanupDefaults_nrpf = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end

nrpf_P = 12.0;
nrpf_chord = 2; nrpf_nCh = 2; nrpf_N = 2;
nrpf_tags = repelem(0:nrpf_nCh - 1, nrpf_chord);
nrpf_spec = struct('tags', nrpf_tags, 'r', [1 nrpf_nCh], ...
                   'sym', [true true], 'rel', [0 1]);

nrpf_pts = @(seed) nrpfPts(seed, nrpf_P, nrpf_chord * nrpf_nCh, nrpf_N);
nrpf_dens = @(p, sigma) buildExpTens({p}, {[]}, 'specs', {nrpf_spec}, ...
    'sigma', sigma, 'isPer', true, 'period', nrpf_P, 'verbose', false);

% --- contract == all-image brute force at every sigma/P ---
for nrpf_sop = [0.02 0.05 0.1 0.2 0.3]
    nrpf_sigma = nrpf_sop * nrpf_P;
    px = nrpf_pts(1); py = nrpf_pts(2);
    c = cosSimExpTens(nrpf_dens(px, nrpf_sigma), nrpf_dens(py, nrpf_sigma), ...
                      'method', 'contract', 'verbose', false);
    r = nrpfRefCos(px, py, nrpf_sigma, nrpf_P, nrpf_tags, nrpf_nCh, true);
    results{end+1, 1} = sprintf( ...
        'nested rel-per taugrid == all-image transposition average at sigma/P=%.2f', ...
        nrpf_sop); %#ok<*SAGROW>
    results{end, 2} = abs(c - r) <= 1e-11 * max(abs(r), 1) + 1e-12;
end

% --- the two measures differ at sigma/P = 0.2 ---
nrpf_sigma = 0.2 * nrpf_P;
px = nrpf_pts(1); py = nrpf_pts(2);
rFull = nrpfRefCos(px, py, nrpf_sigma, nrpf_P, nrpf_tags, nrpf_nCh, true);
rNear = nrpfRefCos(px, py, nrpf_sigma, nrpf_P, nrpf_tags, nrpf_nCh, false);
results{end+1, 1} = 'nested rel-per: all-image and nearest-image averages differ at sigma/P=0.2';
results{end, 2}   = abs(rFull - rNear) > 1e-4;

if standalone
    nPass = sum([results{:, 2}]);
    nFail = size(results, 1) - nPass;
    for ii = 1:size(results, 1)
        if results{ii, 2}
            fprintf('  PASS  %s\n', results{ii, 1});
        else
            fprintf('  FAIL  %s\n', results{ii, 1});
        end
    end
    fprintf('\n=== test_nested_rel_per_full_image: %d passed, %d failed (of %d) ===\n\n', ...
        nPass, nFail, nPass + nFail);
    clear cleanupDefaults_nrpf
    if nFail > 0
        error('test_nested_rel_per_full_image:failed', '%d test(s) failed.', nFail);
    end
end


function p = nrpfPts(seed, P, nVal, N)
    rng(seed, 'twister');
    p = sort(P * rand(nVal, N), 1);
end


function T = nrpfTuples(col, tags, nCh)
    % One value per chord, every ordering (the outer level is symmetric).
    chords = cell(1, nCh);
    for k = 1:nCh
        chords{k} = col(tags == k - 1);
    end
    picks = chords{1}(:);
    for k = 2:nCh
        nxt = chords{k}(:);
        picks = [kron(picks, ones(numel(nxt), 1)), repmat(nxt, size(picks, 1), 1)];
    end
    perms_ = perms(1:nCh);
    T = zeros(size(picks, 1) * size(perms_, 1), nCh);
    row = 0;
    for i = 1:size(picks, 1)
        for q = 1:size(perms_, 1)
            row = row + 1;
            T(row, :) = picks(i, perms_(q, :));
        end
    end
end


function v = nrpfRefIp(px, py, sigma, P, tags, nCh, full)
    ntau = 6000;
    taus = (0:ntau - 1) * (P / ntau);
    N = size(px, 2);
    v = 0;
    for i = 1:N
        for j = 1:N
            X = nrpfTuples(px(:, i), tags, nCh);      % (Tx, r)
            Y = nrpfTuples(py(:, j), tags, nCh);      % (Ty, r)
            d = reshape(X, [size(X, 1), 1, nCh, 1]) ...
              - reshape(Y, [1, size(Y, 1), nCh, 1]) ...
              - reshape(taus, [1, 1, 1, ntau]);
            if full
                K = internal.wrappedGaussian1d(d, sigma, P, Inf, 4);
            else
                dw = d - P * floor(d / P + 0.5);
                K = exp(-dw.^2 / (4 * sigma^2));
            end
            v = v + mean(sum(sum(prod(K, 3), 1), 2), 4);
        end
    end
end


function c = nrpfRefCos(px, py, sigma, P, tags, nCh, full)
    c = nrpfRefIp(px, py, sigma, P, tags, nCh, full) / sqrt( ...
        nrpfRefIp(px, px, sigma, P, tags, nCh, full) * ...
        nrpfRefIp(py, py, sigma, P, tags, nCh, full));
end
