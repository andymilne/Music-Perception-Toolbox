function results = test_nested_spectral_factor()
%TEST_NESTED_SPECTRAL_FACTOR Diagonal-exact factorisation of the relative-non-
%   periodic nested inner product for spectrally-augmented ordered cells.
%
%   The factorisation reduces the inner partial index analytically into the
%   partial-template cross-correlation and evaluates only the per-position
%   carrier overlaps. It is the default path for relative spectral cells, so
%   this test exercises it end-to-end through cosSimExpTens and checks the
%   values against the Python reference (mpt/_tensor/_nested_contraction.py,
%   _ip_rel_nonper_factored). Reference values produced by the Python build at
%   sigma = 0.15, truncationSigmas = 4, double precision.

results = {};
tol = 1e-9;

mpt.setDefault('showHints', false, 'truncationSigmas', 4.0, ...
    'kernelPrecision', 'double');

% --- end-to-end cosine values vs Python reference ----------------------
cases = {
    'ALS self',           [60 63 60 65], 12, 1.0, [60 63 60 65], 12, 1.0, 1.0000000000
    'ALS vs +7 transp',   [60 63 60 65], 12, 1.0, [67 70 67 72], 12, 1.0, 1.0000000635
    'ALS vs octave-fold', [60 63 60 65], 12, 1.0, [70 73 70 63], 12, 1.0, 0.5541160219
    'ALS vs fifth-disp',  [60 63 60 65], 12, 1.0, [60 70 60 65], 12, 1.0, 0.1882538286
    'diff templates',     [60 63 60 65], 12, 1.0, [70 73 70 63],  7, 0.6, 0.5315128244
};
for i = 1:size(cases, 1)
    [lbl, X, KpX, rhoX, Y, KpY, rhoY, ref] = cases{i, :};
    dX = makeCell(X, KpX, rhoX);
    dY = makeCell(Y, KpY, rhoY);
    got = cosSimExpTens(dX, dY, 'verbose', false);
    results = [results; {sprintf('factored cos: %s', lbl), ...
        abs(got - ref) < tol}]; %#ok<AGROW>
end

% --- exact restatement and whole-cell transposition both -> 1 ----------
als = makeCell([60 63 60 65], 12, 1.0);
selfsim = cosSimExpTens(als, makeCell([60 63 60 65], 12, 1.0), 'verbose', false);
results = [results; {'self similarity == 1', abs(selfsim - 1.0) < tol}];

% --- octave-displaced one note gives graded partial credit -------------
oct = cosSimExpTens(als, makeCell([70 73 70 63], 12, 1.0), 'verbose', false);
results = [results; {'octave-fold in (0.50, 0.60)', oct > 0.50 && oct < 0.60}];

end


function d = makeCell(pitches, Kp, rho)
    p = pitches(:).';
    N = numel(p);
    [pp, wp] = addSpectra(p, [], 'harmonic', Kp, 'powerlaw', rho, 'units', 12);
    PIT = reshape(pp, N, Kp).';          % Kp x N carrier (partial-major flatten)
    WP  = reshape(wp, N, Kp).';
    idx = 0:(N - 1);                     % index attribute, mirrors Python arange
    [pb, wb, sb] = bindEvents({PIT, idx}, {WP, []}, [N 1], ...
        'step', 1, 'relOuter', true);
    d = buildExpTens(pb, wb, 'sigma', [0.15 0.125], ...
        'isPer', [false false], 'period', [0 0], 'specs', sb, 'verbose', false);
end
