%% test_nested_spectral_factor.m — diagonal-exact spectral factorisation
%
%  The relative-non-periodic nested inner product for spectrally-augmented
%  ordered cells reduces the inner partial index analytically into the partial
%  template cross-correlation, evaluating only the per-position carrier
%  overlaps (ipRelNonperFactored in +internal/nestedContract.m, mirror of the
%  Python _ip_rel_nonper_factored). It is the default path for relative
%  spectral cells, so this exercises it end-to-end through cosSimExpTens and
%  checks the values against the Python build. Reference values are the
%  cross-language goldens at the factory defaults (untruncated kernels, double
%  precision), sigma = [0.15 0.125].
%
%  Standalone-runnable; appends to `results` when called from test_mpt.m.

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    clear cleanupDefaults
    cleanupDefaults = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end

tol = 1e-9;
SIG = [0.15, 0.125];

% {label, X, KpX, rhoX, Y, KpY, rhoY, reference cosine}
cases = {
    'ALS self',           [60 63 60 65], 12, 1.0, [60 63 60 65], 12, 1.0, 1.0000000000
    'ALS vs +7 transp',   [60 63 60 65], 12, 1.0, [67 70 67 72], 12, 1.0, 1.0000000000
    'ALS vs octave-fold', [60 63 60 65], 12, 1.0, [70 73 70 63], 12, 1.0, 0.5541184880
    'ALS vs fifth-disp',  [60 63 60 65], 12, 1.0, [60 70 60 65], 12, 1.0, 0.1883087300
    'diff templates',     [60 63 60 65], 12, 1.0, [70 73 70 63],  7, 0.6, 0.5315144553
};

for ci = 1:size(cases, 1)
    lbl  = cases{ci, 1};
    X    = cases{ci, 2};  KpX = cases{ci, 3};  rhoX = cases{ci, 4};
    Y    = cases{ci, 5};  KpY = cases{ci, 6};  rhoY = cases{ci, 7};
    ref  = cases{ci, 8};

    NX = numel(X);
    [ppX, wpX] = addSpectra(X, [], 'harmonic', KpX, 'powerlaw', rhoX, ...
        'units', 12);
    PITX = reshape(ppX, NX, KpX).';        % Kp x N carrier (partial-major)
    WPX  = reshape(wpX, NX, KpX).';
    [pbX, wbX, sbX] = bindEvents({PITX, 0:(NX - 1)}, {WPX, []}, [NX 1], ...
        'step', 1, 'relOuter', true);
    dX = buildExpTens(pbX, wbX, 'sigma', SIG, 'isPer', [false false], ...
        'period', [0 0], 'specs', sbX, 'verbose', false);

    NY = numel(Y);
    [ppY, wpY] = addSpectra(Y, [], 'harmonic', KpY, 'powerlaw', rhoY, ...
        'units', 12);
    PITY = reshape(ppY, NY, KpY).';
    WPY  = reshape(wpY, NY, KpY).';
    [pbY, wbY, sbY] = bindEvents({PITY, 0:(NY - 1)}, {WPY, []}, [NY 1], ...
        'step', 1, 'relOuter', true);
    dY = buildExpTens(pbY, wbY, 'sigma', SIG, 'isPer', [false false], ...
        'period', [0 0], 'specs', sbY, 'verbose', false);

    got = cosSimExpTens(dX, dY, 'verbose', false);
    results = [results; {sprintf('factored cos: %s', lbl), ...
        abs(got - ref) < tol}]; %#ok<AGROW>
end

if standalone
    nPass = sum([results{:, 2}]);
    fprintf('test_nested_spectral_factor: %d/%d passed\n', nPass, ...
        size(results, 1));
    for i = 1:size(results, 1)
        if ~results{i, 2}
            fprintf('  FAIL: %s\n', results{i, 1});
        end
    end
    clear cleanupDefaults
end
