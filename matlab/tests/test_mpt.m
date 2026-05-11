%% test_mpt.m — Test script for the Music Perception Toolbox (MATLAB)
%
%  Run from the MATLAB command window:
%    run('test_mpt')
%
%  The script runs a series of checks and reports pass/fail for each.
%  A summary is printed at the end.

results = {};  % accumulate {'name', true/false}

fprintf('\n=== Music Perception Toolbox — Test Suite ===\n\n');

%% ---- convertPitch ----

results{end+1,1} = 'convertPitch: Hz→MIDI';
results{end,2}   = abs(convertPitch(440, 'hz', 'midi') - 69) < 1e-10;

results{end+1,1} = 'convertPitch: MIDI→Hz';
results{end,2}   = abs(convertPitch(60, 'midi', 'hz') - 261.6256) / 261.6256 < 1e-4;

results{end+1,1} = 'convertPitch: Hz→cents';
results{end,2}   = abs(convertPitch(440, 'hz', 'cents') - 6900) < 1e-10;

results{end+1,1} = 'convertPitch: identity';
results{end,2}   = isequal(convertPitch([100, 200, 300], 'hz', 'hz'), [100, 200, 300]);

scales = {'midi', 'cents', 'mel', 'bark', 'erb', 'greenwood'};
for i = 1:numel(scales)
    rt = convertPitch(convertPitch(440, 'hz', scales{i}), scales{i}, 'hz');
    results{end+1,1} = ['convertPitch: roundtrip ' scales{i}]; %#ok<SAGROW>
    results{end,2}   = abs(rt - 440) / 440 < 1e-8;
end

out = convertPitch([261.63, 440, 880], 'hz', 'midi');
results{end+1,1} = 'convertPitch: vectorised';
results{end,2}   = all(abs(out - [60, 69, 81]) < 0.01);

results{end+1,1} = 'convertPitch: unknown scale errors';
results{end,2}   = throwsError(@() convertPitch(440, 'hz', 'bogus'));

%% ---- addSpectra ----

[p, ~] = addSpectra([0, 400, 700], [], 'harmonic', 8, 'powerlaw', 1);
results{end+1,1} = 'addSpectra: harmonic count';
results{end,2}   = numel(p) == 24;

[p, w] = addSpectra(0, 1, 'harmonic', 4, 'powerlaw', 0);
expected_p = 1200 * log2((1:4)');
results{end+1,1} = 'addSpectra: harmonic positions';
results{end,2}   = all(abs(p - expected_p) < 1e-10);
results{end+1,1} = 'addSpectra: flat weights';
results{end,2}   = all(abs(w - 1) < 1e-10);

[p, ~] = addSpectra(0, 1, 'stretched', 3, 1.02, 'powerlaw', 1);
[pHarm, ~] = addSpectra(0, 1, 'harmonic', 3, 'powerlaw', 1);
results{end+1,1} = 'addSpectra: stretched wider than harmonic';
results{end,2}   = p(3) > pHarm(3);

[p, ~] = addSpectra(0, 1, 'stiff', 4, 0.0003, 'powerlaw', 1);
[pHarm, ~] = addSpectra(0, 1, 'harmonic', 4, 'powerlaw', 1);
results{end+1,1} = 'addSpectra: stiff sharper than harmonic';
results{end,2}   = p(4) > pHarm(4);

[p, w] = addSpectra([0, 700], [], 'custom', [0, 1200], [1, 0.5]);
results{end+1,1} = 'addSpectra: custom positions';
results{end,2}   = all(abs(p - [0; 700; 1200; 1900]) < 1e-10);
results{end+1,1} = 'addSpectra: custom weights';
results{end,2}   = all(abs(w - [1; 1; 0.5; 0.5]) < 1e-10);

[~, w] = addSpectra(0, 1, 'harmonic', 4, 'geometric', 0.5);
results{end+1,1} = 'addSpectra: geometric weights';
results{end,2}   = all(abs(w - [1; 0.5; 0.25; 0.125]) < 1e-10);

% -- freqlinear: alpha = 0 reproduces harmonic (ratio(n) = n) --
[p_lin, w_lin] = addSpectra(0, 1, 'freqlinear', 4, 0.0, 'powerlaw', 0);
[p_har, w_har] = addSpectra(0, 1, 'harmonic', 4, 'powerlaw', 0);
results{end+1,1} = 'addSpectra: freqlinear alpha=0 equals harmonic';
results{end,2}   = max(abs(p_lin - p_har)) < 1e-10 ...
                && max(abs(w_lin - w_har)) < 1e-10;

% -- freqlinear: alpha = 1 gives ratios [1, 1.5, 2, 2.5] --
[p_fl, ~] = addSpectra(0, 1, 'freqlinear', 4, 1.0, 'powerlaw', 0);
expected_fl = 1200 * log2([1; 1.5; 2; 2.5]);
results{end+1,1} = 'addSpectra: freqlinear alpha=1 partial ratios';
results{end,2}   = max(abs(p_fl - expected_fl)) < 1e-10;

% -- freqlinear: alpha <= -1 errors (ratio non-positive) --
results{end+1,1} = 'addSpectra: freqlinear alpha<=-1 errors';
results{end,2}   = throwsError(@() addSpectra(0, 1, 'freqlinear', 4, -1.0, 'powerlaw', 0));

%% ---- Circular measures ----

b = balanceCircular([0, 400, 800], [], 1200);
results{end+1,1} = 'balanceCircular: augmented triad';
results{end,2}   = abs(b - 1) < 1e-10;

b = balanceCircular([0, 100, 200], [], 1200);
results{end+1,1} = 'balanceCircular: cluster unbalanced';
results{end,2}   = b < 0.5;

e = evennessCircular([0, 200, 400, 600, 800, 1000], 1200);
results{end+1,1} = 'evennessCircular: whole-tone';
results{end,2}   = abs(e - 1) < 1e-10;

[F, mag] = dftCircular([0, 4, 7], [], 12);
results{end+1,1} = 'dftCircular: output length';
results{end,2}   = numel(F) == 3 && numel(mag) == 3;

% -- dftCircular: |F[0]| = 1 for unison --
[~, mag_uni] = dftCircular([100, 100, 100], [], 1200);
results{end+1,1} = 'dftCircular: |F[0]| = 1 for unison';
results{end,2}   = abs(mag_uni(1) - 1) < 1e-10;

% -- dftCircular: |F[0]| = 0 for augmented triad (cube roots of unity) --
[~, mag_aug] = dftCircular([0, 400, 800], [], 1200);
results{end+1,1} = 'dftCircular: |F[0]| = 0 for augmented triad';
results{end,2}   = abs(mag_aug(1)) < 1e-10;

[c, nc] = coherence([0, 2, 4, 5, 7, 9, 11], 12);
results{end+1,1} = 'coherence: diatonic nc = 1';
results{end,2}   = nc == 1;
results{end+1,1} = 'coherence: diatonic c > 0.99';
results{end,2}   = c > 0.99;

[c, nc] = coherence([0, 2, 4, 6, 8, 10], 12);
results{end+1,1} = 'coherence: whole-tone perfect';
results{end,2}   = abs(c - 1) < 1e-10 && nc == 0;

[sq, nd] = sameness([0, 2, 4, 5, 7, 9, 11], 12);
results{end+1,1} = 'sameness: diatonic nd = 1';
results{end,2}   = nd == 1;
results{end+1,1} = 'sameness: diatonic sq > 0.99';
results{end,2}   = sq > 0.99;

[sq, nd] = sameness([0, 2, 4, 6, 8, 10], 12);
results{end+1,1} = 'sameness: whole-tone perfect';
results{end,2}   = abs(sq - 1) < 1e-10 && nd == 0;

[e_out, ~] = edges([0, 2, 4, 5, 7, 9, 11], [], 12);
results{end+1,1} = 'edges: output shape and non-negative';
results{end,2}   = numel(e_out) == 12 && all(e_out >= 0);

% -- edges: zero at event positions of an even scale (rotational symmetry) --
[e_even, ~] = edges([0, 200, 400, 600, 800, 1000], [], 1200);
% MATLAB 1-indexed: positions 0, 200, 400, 600, 800, 1000 → indices 1, 201, 401, 601, 801, 1001
e_at_events = e_even([1, 201, 401, 601, 801, 1001]);
results{end+1,1} = 'edges: zero at events of even scale';
results{end,2}   = max(abs(e_at_events)) < 1e-10;

% -- edges: signed antisymmetric for a contiguous block --
% Six events filling positions 0..5 of a 12-slot circle: rising edge
% just before position 0 (index 12, position 11), falling edge just
% after position 5 (index 7, position 6).
[~, e_signed] = edges([0, 1, 2, 3, 4, 5], [], 12);
results{end+1,1} = 'edges: signed > 0 at rising-edge boundary';
results{end,2}   = e_signed(12) > 0;
results{end+1,1} = 'edges: signed < 0 at falling-edge boundary';
results{end,2}   = e_signed(7) < 0;

[y, cm, ~] = projCentroid([0, 400, 800], [], 1200);
results{end+1,1} = 'projCentroid: balanced magnitude = 0';
results{end,2}   = abs(cm) < 1e-10;
results{end+1,1} = 'projCentroid: balanced projections = 0';
results{end,2}   = all(abs(y) < 1e-10);

h = meanOffset([0, 2, 4, 5, 7, 9, 11], [], 12);
results{end+1,1} = 'meanOffset: output length';
results{end,2}   = numel(h) == 12;

% -- meanOffset: zero at event positions of an even scale --
h_even = meanOffset([0, 200, 400, 600, 800, 1000], [], 1200);
h_at_events = h_even([1, 201, 401, 601, 801, 1001]);
results{end+1,1} = 'meanOffset: zero at events of even scale';
results{end,2}   = max(abs(h_at_events)) < 1e-10;

[R, rp, rl] = circApm([0, 3, 6, 10, 12], [], 16);
results{end+1,1} = 'circApm: R shape';
results{end,2}   = isequal(size(R), [16, 16]);
results{end+1,1} = 'circApm: rPhase length';
results{end,2}   = numel(rp) == 16;
results{end+1,1} = 'circApm: rLag length';
results{end,2}   = numel(rl) == 16;

% -- circApm: rLag is symmetric about lag 0
% (real-valued circular autocorrelation: r_lag(k) == r_lag(P - k))
% MATLAB 1-indexed: r_lag(k+1) == r_lag(P-k+1) for k = 1, ..., P/2 - 1.
sym_diff = max(arrayfun(@(k) abs(rl(k+1) - rl(17-k)), 1:7));
results{end+1,1} = 'circApm: rLag symmetric about lag 0';
results{end,2}   = sym_diff < 1e-10;

y = markovS([0, 3, 6, 10, 12], [], 16);
results{end+1,1} = 'markovS: output length';
results{end,2}   = numel(y) == 16;
results{end+1,1} = 'markovS: positive at events';
results{end,2}   = y(1) > 0 && y(4) > 0;

% -- markovS: equal weights for events of a 4-periodic pattern --
% In a period-16 cycle with events at 0, 4, 8, 12 (a 4-step pattern),
% the four event positions share the same S-step look-ahead context,
% so predicted weights are equal there. Same for the non-events
% within a period.
y_per = markovS([0, 4, 8, 12], [], 16);
results{end+1,1} = 'markovS: equal at events of 4-periodic pattern';
results{end,2}   = max(abs(y_per([1, 5, 9, 13]) - y_per(1))) < 1e-10;
results{end+1,1} = 'markovS: equal at non-events of 4-periodic pattern';
results{end,2}   = max(abs(y_per([2, 6, 10, 14]) - y_per(2))) < 1e-10;
results{end+1,1} = 'markovS: events outweigh non-events in periodic pattern';
results{end,2}   = y_per(1) > y_per(2);

%% ---- Tier-1 batched dispatch (v2.1+): dftCircular, meanOffset, edges, projCentroid, circApm ----
%
% Each accepts an nRows-by-K matrix in addition to the original 1-D
% form, and returns 1-by-nRows cell arrays. Per-row dedup is over
% permutation + period symmetries (not transposition).

% --- dftCircular ---
P_dft = [0, 200, 400, 500, 700, 900, 1100;     % major
         0, 100, 300, 500, 700, 900, 1000;     % something else
         0, 200, 400, 500, 700, 900, 1100];    % = row 1

[FCell, magCell] = dftCircular(P_dft, [], 1200);
results{end+1,1} = 'dftCircular batched: returns 1-by-nRows cells';
results{end,2}   = iscell(FCell) && iscell(magCell) ...
                && isequal(size(FCell), [1, 3]) ...
                && isequal(size(magCell), [1, 3]);

% Per-row matches scalar
[F_scalar, mag_scalar] = dftCircular(P_dft(2, :), [], 1200);
results{end+1,1} = 'dftCircular batched: matches scalar dispatch row-by-row';
results{end,2}   = max(abs(FCell{2} - F_scalar)) < 1e-12 ...
                && max(abs(magCell{2} - mag_scalar)) < 1e-12;

% Permutation dedup
P_perm = [0, 200, 400; 400, 0, 200];
[Fperm, magPerm] = dftCircular(P_perm, [], 1200);
results{end+1,1} = 'dftCircular batched: permutation dedup';
results{end,2}   = isequal(Fperm{1}, Fperm{2});

% NaN-padded variable cardinality
P_nan = [0, 200, 400, NaN; 0, 100, 200, 300; NaN, NaN, NaN, NaN];
[Fnan, magNan] = dftCircular(P_nan, [], 1200);
results{end+1,1} = 'dftCircular batched: NaN-padded cardinality OK';
results{end,2}   = numel(Fnan{1}) == 3 && numel(Fnan{2}) == 4 && isempty(Fnan{3});

% --- meanOffset ---
P_mo = [0, 4, 7; 0, 3, 7; 0, 4, 7];
hMOCell = meanOffset(P_mo, [], 12);
results{end+1,1} = 'meanOffset batched: returns 1-by-nRows cell';
results{end,2}   = iscell(hMOCell) && isequal(size(hMOCell), [1, 3]);

hMOscalar = meanOffset(P_mo(2, :), [], 12);
results{end+1,1} = 'meanOffset batched: matches scalar dispatch';
results{end,2}   = max(abs(hMOCell{2} - hMOscalar)) < 1e-12;

results{end+1,1} = 'meanOffset batched: dedup row 1 = row 3';
results{end,2}   = isequal(hMOCell{1}, hMOCell{3});

% --- edges ---
P_e = [0, 4, 7; 0, 3, 7; 0, 4, 7];
[eCell, esCell] = edges(P_e, [], 12);
results{end+1,1} = 'edges batched: returns two 1-by-nRows cells';
results{end,2}   = iscell(eCell) && iscell(esCell) ...
                && isequal(size(eCell), [1, 3]);

[e_scalar, es_scalar] = edges(P_e(2, :)', [], 12);
results{end+1,1} = 'edges batched: matches scalar dispatch';
results{end,2}   = max(abs(eCell{2} - e_scalar)) < 1e-12 ...
                && max(abs(esCell{2} - es_scalar)) < 1e-12;

results{end+1,1} = 'edges batched: dedup row 1 = row 3';
results{end,2}   = isequal(eCell{1}, eCell{3}) && isequal(esCell{1}, esCell{3});

% --- projCentroid ---
P_pc = [0, 4, 7; 0, 3, 7; 0, 4, 7];
[yCell, cmCell, cpCell] = projCentroid(P_pc, [], 12);
results{end+1,1} = 'projCentroid batched: returns three 1-by-nRows cells';
results{end,2}   = iscell(yCell) && iscell(cmCell) && iscell(cpCell) ...
                && isequal(size(yCell), [1, 3]);

[y_scalar, cm_scalar, cp_scalar] = projCentroid(P_pc(2, :)', [], 12);
results{end+1,1} = 'projCentroid batched: matches scalar dispatch';
results{end,2}   = max(abs(yCell{2} - y_scalar)) < 1e-12 ...
                && abs(cmCell{2} - cm_scalar) < 1e-12 ...
                && abs(cpCell{2} - cp_scalar) < 1e-12;

results{end+1,1} = 'projCentroid batched: dedup centroid magnitude row 1 = row 3';
results{end,2}   = isequal(cmCell{1}, cmCell{3});

% --- circApm ---
P_apm = [0, 3, 6, 8, 10, 12, 14;
         0, 2, 4, 6, 8, 10, 12;
         0, 3, 6, 8, 10, 12, 14];
[Rcell, rPhaseCell, rLagCell] = circApm(P_apm, [], 16);
results{end+1,1} = 'circApm batched: returns three 1-by-nRows cells';
results{end,2}   = iscell(Rcell) && iscell(rPhaseCell) && iscell(rLagCell) ...
                && isequal(size(Rcell), [1, 3]) ...
                && isequal(size(Rcell{1}), [16, 16]);

% Scalar-equivalent: pre-sort to canonical form (batched dedups via
% sorted modular form before computing).
[R_scalar, ~, ~] = circApm(sort(mod(P_apm(2, :), 16))', [], 16);
results{end+1,1} = 'circApm batched: matches scalar dispatch (canonical form)';
results{end,2}   = isequal(Rcell{2}, R_scalar);

results{end+1,1} = 'circApm batched: dedup row 1 = row 3';
results{end,2}   = isequal(Rcell{1}, Rcell{3});

% Period reduction in canonical key
P_apm_pred = [0, 3, 6, 8, 10, 12, 14;
              0, 19, 6, 8, 10, 12, 14];   % 19 mod 16 = 3
[Rred, ~, ~] = circApm(P_apm_pred, [], 16);
results{end+1,1} = 'circApm batched: dedup over period reduction';
results{end,2}   = isequal(Rred{1}, Rred{2});

% Non-integer pitch errors clearly
results{end+1,1} = 'circApm batched: non-integer pitches error';
results{end,2}   = throwsErrorWithId( ...
    @() circApm([0.5, 1.0, 2.0; 0.5, 1.0, 2.0], [], 16), ...
    'circApm:nonIntegerPitch');

%% ---- Tier-2 batched dispatch (v2.1+): coherence, sameness, nTupleEntropy ----
%
% These set-based functions accept an nRows-by-K matrix and return
% per-row results: nRows-by-1 vectors for scalar outputs and 1-by-nRows
% cell arrays for variable-shape outputs (nTupleEntropy's ``tuples``).
% Per-row dedup uses a sorted-modular canonical key (permutation +
% period symmetries; not transposition).

% --- coherence ---
P_coh = [0, 4, 7; 0, 3, 7; 0, 4, 7];
[cVec, ncVec] = coherence(P_coh, 12);
results{end+1,1} = 'coherence batched: returns nRows-by-1 vectors';
results{end,2}   = isnumeric(cVec) && isequal(size(cVec), [3, 1]) ...
                && isequal(size(ncVec), [3, 1]);

[c_scalar, nc_scalar] = coherence(P_coh(2, :)', 12);
results{end+1,1} = 'coherence batched: matches scalar dispatch';
results{end,2}   = abs(cVec(2) - c_scalar) < 1e-12 ...
                && abs(ncVec(2) - nc_scalar) < 1e-12;

results{end+1,1} = 'coherence batched: dedup row 1 = row 3';
results{end,2}   = cVec(1) == cVec(3) && ncVec(1) == ncVec(3);

% Permutation dedup
P_coh_perm = [0, 4, 7; 4, 0, 7];
[cPerm, ~] = coherence(P_coh_perm, 12);
results{end+1,1} = 'coherence batched: permutation dedup';
results{end,2}   = cPerm(1) == cPerm(2);

% NaN-padded
P_coh_pad = [0, 4, 7, NaN; 0, 1, 5, 6; NaN, NaN, NaN, NaN];
[cPad, ~] = coherence(P_coh_pad, 12);
results{end+1,1} = 'coherence batched: NaN-padded variable cardinality';
results{end,2}   = ~isnan(cPad(1)) && ~isnan(cPad(2)) && isnan(cPad(3));

% --- sameness ---
P_sm = [0, 4, 7; 0, 3, 7; 0, 4, 7];
[sqVec, ndVec] = sameness(P_sm, 12);
results{end+1,1} = 'sameness batched: returns nRows-by-1 vectors';
results{end,2}   = isequal(size(sqVec), [3, 1]);

[sq_s, nd_s] = sameness(P_sm(2, :)', 12);
results{end+1,1} = 'sameness batched: matches scalar dispatch';
results{end,2}   = abs(sqVec(2) - sq_s) < 1e-12 ...
                && abs(ndVec(2) - nd_s) < 1e-12;

results{end+1,1} = 'sameness batched: dedup row 1 = row 3';
results{end,2}   = sqVec(1) == sqVec(3);

% --- nTupleEntropy ---
P_nte = [0, 4, 7; 0, 3, 7; 0, 4, 7];
[HVec, tuplesCell] = nTupleEntropy(P_nte, 12, 1);
results{end+1,1} = 'nTupleEntropy batched: returns vector + cell';
results{end,2}   = isnumeric(HVec) && isequal(size(HVec), [3, 1]) ...
                && iscell(tuplesCell) && isequal(size(tuplesCell), [1, 3]);

[H_s, t_s] = nTupleEntropy(P_nte(2, :)', 12, 1);
results{end+1,1} = 'nTupleEntropy batched: matches scalar dispatch';
results{end,2}   = abs(HVec(2) - H_s) < 1e-12 ...
                && isequal(tuplesCell{2}, t_s);

results{end+1,1} = 'nTupleEntropy batched: dedup row 1 = row 3';
results{end,2}   = HVec(1) == HVec(3);

% NaN-padded all-NaN row gives NaN H and empty tuples
P_nte_pad = [0, 4, 7, NaN; NaN, NaN, NaN, NaN];
[Hpad, tpad] = nTupleEntropy(P_nte_pad, 12, 1);
results{end+1,1} = 'nTupleEntropy batched: all-NaN row returns NaN';
results{end,2}   = ~isnan(Hpad(1)) && isnan(Hpad(2)) && isempty(tpad{2});

% --- Transposition dedup (necklace canonical) for coherence and sameness ---
% All 12 transpositions of the major triad share one necklace key, so
% values are identical across rows.
shifts = (0:11)';
P_majorTrans = mod([0, 4, 7] + shifts, 12);  % 12-by-3
[cTrans, ncTrans] = coherence(P_majorTrans, 12);
results{end+1,1} = 'coherence batched: all 12 major-triad transpositions agree';
results{end,2}   = all(cTrans == cTrans(1)) && all(ncTrans == ncTrans(1));

% Coherence with failures: {0, 1, 5, 6} and 12 transpositions
P_aug = mod([0, 1, 5, 6] + shifts, 12);
[cAug, ncAug] = coherence(P_aug, 12);
results{end+1,1} = 'coherence batched: transposition dedup with non-trivial nc';
results{end,2}   = all(cAug == cAug(1)) && ncAug(1) == 5.0;

% Sameness: 12 major-triad transpositions agree
[sqTrans, ndTrans] = sameness(P_majorTrans, 12);
results{end+1,1} = 'sameness batched: all 12 major-triad transpositions agree';
results{end,2}   = all(sqTrans == sqTrans(1)) && all(ndTrans == ndTrans(1));

% nTupleEntropy: H is transposition-invariant; tuples is NOT (in general).
% Pin this down to make the dedup contract explicit.
P_nteTrans = [0, 4, 7; 2, 7, 11];   % {0,4,7} and its shift-by-7 transposition
[HnteTrans, tnteTrans] = nTupleEntropy(P_nteTrans, 12, 1);
results{end+1,1} = 'nTupleEntropy batched: H is transposition-invariant';
results{end,2}   = abs(HnteTrans(1) - HnteTrans(2)) < 1e-12;
results{end+1,1} = 'nTupleEntropy batched: tuples differ across transpositions';
results{end,2}   = ~isequal(tnteTrans{1}, tnteTrans{2});

%% ---- Tier-4 batched dispatch (v2.1+): balanceCircular, evennessCircular ----
%
% These Monte-Carlo functions accept an nRows-by-K matrix and return
% per-row column vectors. The new ``rngScope`` NV pair controls how
% each row's RNG seed is derived from the base ``rngSeed``:
%   'canonical' (default): derived from canonical-form key, so
%      transposition-equivalent rows share an MC realisation; dedup
%      works for sigma > 0.
%   'row': derived from row index, so identical rows get distinct
%      reproducible realisations; dedup is disabled.

% --- balanceCircular: sigma = 0 (deterministic) ---
P_b0 = [0, 4, 7; 0, 3, 7; 0, 4, 7];
bVec0 = balanceCircular(P_b0, [], 12, 0);
results{end+1,1} = 'balanceCircular batched sigma=0: returns nRows-by-1 vector';
results{end,2}   = isnumeric(bVec0) && isequal(size(bVec0), [3, 1]);

results{end+1,1} = 'balanceCircular batched sigma=0: matches scalar dispatch';
results{end,2}   = abs(bVec0(2) - balanceCircular(P_b0(2, :)', [], 12, 0)) < 1e-12;

results{end+1,1} = 'balanceCircular batched sigma=0: dedup (rows 1, 3 identical)';
results{end,2}   = bVec0(1) == bVec0(3);

[bVec0_, bStd0_] = balanceCircular(P_b0, [], 12, 0);
results{end+1,1} = 'balanceCircular batched sigma=0: bStd is zero with two outputs';
results{end,2}   = isequal(size(bStd0_), [3, 1]) && all(bStd0_ == 0);

% NaN-padded
P_b0_nan = [0, 4, 7, NaN; 0, 1, 5, 6; NaN, NaN, NaN, NaN];
bNan = balanceCircular(P_b0_nan, [], 12, 0);
results{end+1,1} = 'balanceCircular batched sigma=0: NaN-padded rows give NaN';
results{end,2}   = ~isnan(bNan(1)) && ~isnan(bNan(2)) && isnan(bNan(3));

% --- balanceCircular: sigma > 0 (Monte Carlo) ---

% Canonical scope: identical inputs give identical MC results
P_b1 = [0, 4, 7; 4, 0, 7; 0, 4, 7];
bCanon = balanceCircular(P_b1, [], 12, 0.3, ...
    'rngSeed', 42, 'rngScope', 'canonical');
results{end+1,1} = 'balanceCircular batched MC canonical: identical inputs identical results';
results{end,2}   = bCanon(1) == bCanon(2) && bCanon(1) == bCanon(3);

% Different canonicals -> different results
P_b2 = [0, 4, 7; 0, 3, 7];
bDiff = balanceCircular(P_b2, [], 12, 0.3, ...
    'rngSeed', 42, 'rngScope', 'canonical');
results{end+1,1} = 'balanceCircular batched MC canonical: different scales give different results';
results{end,2}   = bDiff(1) ~= bDiff(2);

% Reproducibility across calls
b_call_a = balanceCircular(P_b1, [], 12, 0.3, 'rngSeed', 42);
b_call_b = balanceCircular(P_b1, [], 12, 0.3, 'rngSeed', 42);
results{end+1,1} = 'balanceCircular batched MC: reproducible with same rngSeed';
results{end,2}   = isequal(b_call_a, b_call_b);

% Row scope: identical inputs give DIFFERENT realisations
bRow = balanceCircular(P_b1, [], 12, 0.3, ...
    'rngSeed', 42, 'rngScope', 'row');
results{end+1,1} = 'balanceCircular batched MC row scope: identical inputs differ';
results{end,2}   = bRow(1) ~= bRow(2) && bRow(2) ~= bRow(3) && bRow(1) ~= bRow(3);

% Empty rngSeed in batched: within-call dedup still works
bDedup = balanceCircular(P_b1, [], 12, 0.3);
results{end+1,1} = 'balanceCircular batched MC: within-call dedup with empty rngSeed';
results{end,2}   = bDedup(1) == bDedup(2) && bDedup(1) == bDedup(3);

% --- evennessCircular: brief MC sanity ---
P_e = [0, 4, 7; 4, 0, 7];
eDedup = evennessCircular(P_e, 12, 0.3, 'rngSeed', 42, 'rngScope', 'canonical');
results{end+1,1} = 'evennessCircular batched MC canonical: dedup works';
results{end,2}   = eDedup(1) == eDedup(2);

eRow = evennessCircular([0, 4, 7; 0, 4, 7], 12, 0.3, ...
    'rngSeed', 42, 'rngScope', 'row');
results{end+1,1} = 'evennessCircular batched MC row scope: identical inputs differ';
results{end,2}   = eRow(1) ~= eRow(2);

%% ---- Expectation tensors ----

dens = buildExpTens([0, 4, 7], [], 0.5, 1, false, true, 12, ...
    'verbose', false);
vals = evalExpTens(dens, 0:11, 'verbose', false);
peaks = find(vals > 0.5) - 1;
results{end+1,1} = 'buildExpTens/evalExpTens: peaks at 0, 4, 7';
results{end,2}   = isequal(peaks, [0, 4, 7]);

s = cosSimExpTens([0, 4, 7], [], [0, 4, 7], [], ...
    10, 1, false, true, 1200, 'verbose', false);
results{end+1,1} = 'cosSimExpTens: identical = 1';
results{end,2}   = abs(s - 1) < 1e-10;

s = cosSimExpTens( ...
    [0, 200, 400, 500, 700, 900, 1100], [], ...
    [0, 400, 700], [], ...
    10, 1, false, true, 1200, 'verbose', false);
results{end+1,1} = 'cosSimExpTens: 0 < s < 1';
results{end,2}   = s > 0 && s < 1;

dens = buildExpTens([0, 4, 7], [], 0.5, 2, true, true, 12, ...
    'verbose', false);
results{end+1,1} = 'buildExpTens: relative tensor dim = 1';
results{end,2}   = dens.dim == 1;

A = [0, 200, 400, 500, 700, 900, 1100;
     0, 200, 400, 500, 700, 900, 1100];
B = [0, 400, 700, NaN, NaN, NaN, NaN;
     0, 300, 700, NaN, NaN, NaN, NaN];
s = batchCosSimExpTens(A, B, 10, 1, false, true, 1200, ...
    'verbose', false);
results{end+1,1} = 'batchCosSimExpTens: output length';
results{end,2}   = numel(s) == 2;
results{end+1,1} = 'batchCosSimExpTens: no NaN';
results{end,2}   = all(~isnan(s));
results{end+1,1} = 'batchCosSimExpTens: major > minor fit';
results{end,2}   = s(1) > s(2);

% --- v2.1 unified dispatch: list mode and batched-raw mode ----

% List mode (SA): cell of density structs in, cell of values out
d1 = buildExpTens([0, 4, 7], [], 0.5, 1, false, true, 12, 'verbose', false);
d2 = buildExpTens([0, 3, 7], [], 0.5, 1, false, true, 12, 'verbose', false);
d3 = buildExpTens([0, 4, 7], [], 0.5, 1, false, true, 12, 'verbose', false);
sCell = cosSimExpTens({d1, d2}, {d3, d3}, 'verbose', false);
results{end+1,1} = 'cosSimExpTens list: returns cell of correct length';
results{end,2}   = iscell(sCell) && numel(sCell) == 2;
sScalar1 = cosSimExpTens(d1, d3, 'verbose', false);
sScalar2 = cosSimExpTens(d2, d3, 'verbose', false);
results{end+1,1} = 'cosSimExpTens list: matches scalar dispatch element-wise';
results{end,2}   = abs(sCell{1} - sScalar1) < 1e-14 ...
                   && abs(sCell{2} - sScalar2) < 1e-14;

% List mode: Option II shape rule (length-1 stays length-1)
sCell1 = cosSimExpTens({d1}, {d3}, 'verbose', false);
results{end+1,1} = 'cosSimExpTens list: length-1 returns length-1 cell (Option II)';
results{end,2}   = iscell(sCell1) && numel(sCell1) == 1 ...
                   && abs(sCell1{1} - sScalar1) < 1e-14;

% List mode: length mismatch errors
results{end+1,1} = 'cosSimExpTens list: length mismatch errors';
results{end,2}   = throwsErrorWithId( ...
    @() cosSimExpTens({d1, d2}, {d3}, 'verbose', false), ...
    'MPT:CosSimList:LengthMismatch');

% List mode: non-struct entry errors
results{end+1,1} = 'cosSimExpTens list: non-struct entry errors';
results{end,2}   = throwsErrorWithId( ...
    @() cosSimExpTens({d1, [1, 2, 3]}, {d3, d3}, 'verbose', false), ...
    'MPT:CosSimList:NonStruct');

% Batched-raw mode: 2-D matrix dispatch returns vector
A2 = [0, 200, 400, 500, 700, 900, 1100;
      0, 200, 400, 500, 700, 900, 1100];
B2 = [0, 400, 700, NaN, NaN, NaN, NaN;
      0, 300, 700, NaN, NaN, NaN, NaN];
sBatched = cosSimExpTens(A2, [], B2, [], 10, 1, false, true, 1200, ...
    'verbose', false);
results{end+1,1} = 'cosSimExpTens batched: returns vector of correct length';
results{end,2}   = isnumeric(sBatched) && numel(sBatched) == 2;

% Batched-raw mode: numerically equivalent to batchCosSimExpTens
% (suppress the v2.1 deprecation warning while we make the comparison)
warnState = warning('off', 'MPT:DeprecatedAPI');
sBatchOld = batchCosSimExpTens(A2, B2, 10, 1, false, true, 1200, ...
    'verbose', false);
warning(warnState);
results{end+1,1} = 'cosSimExpTens batched: matches batchCosSimExpTens exactly';
results{end,2}   = max(abs(sBatched(:) - sBatchOld(:))) < 1e-14;

% Batched-raw mode: matches scalar dispatch row-by-row
sScalar1 = cosSimExpTens(A2(1, :), [], B2(1, ~isnan(B2(1, :))), [], ...
    10, 1, false, true, 1200, 'verbose', false);
sScalar2 = cosSimExpTens(A2(2, :), [], B2(2, ~isnan(B2(2, :))), [], ...
    10, 1, false, true, 1200, 'verbose', false);
results{end+1,1} = 'cosSimExpTens batched: matches scalar dispatch row-by-row';
results{end,2}   = abs(sBatched(1) - sScalar1) < 1e-12 ...
                   && abs(sBatched(2) - sScalar2) < 1e-12;

% Batched-raw mode: row mismatch errors
A3 = [0, 4, 7; 0, 3, 7; 0, 5, 9];   % 3 rows
results{end+1,1} = 'cosSimExpTens batched: row mismatch errors';
results{end,2}   = throwsErrorWithId( ...
    @() cosSimExpTens(A3, [], B2, [], 10, 1, false, true, 1200, ...
        'verbose', false), ...
    'MPT:CosSimBatched:RowMismatch');

% Row vector still uses scalar SA raw path (backward compatibility)
% Despite being a 1-by-3 matrix, [0 4 7] is a vector and dispatches to
% the existing scalar form, returning a scalar.
sScalarFromRow = cosSimExpTens([0, 4, 7], [], [0, 4, 7], [], ...
    10, 1, false, true, 1200, 'verbose', false);
results{end+1,1} = 'cosSimExpTens batched: row vector falls through to scalar';
results{end,2}   = isnumeric(sScalarFromRow) && isscalar(sScalarFromRow) ...
                   && abs(sScalarFromRow - 1) < 1e-10;

% Direct batchCosSimExpTens call now emits a deprecation warning
prevWarnState = warning('on', 'MPT:DeprecatedAPI');
lastwarn('', '');  % reset lastwarn so we capture only this call's warning
sDummy = batchCosSimExpTens(A2, B2, 10, 1, false, true, 1200, ...
    'verbose', false); %#ok<NASGU>
[~, lastWarnId] = lastwarn;
warning(prevWarnState);
results{end+1,1} = 'batchCosSimExpTens: emits MPT:DeprecatedAPI warning';
results{end,2}   = strcmp(lastWarnId, 'MPT:DeprecatedAPI');

% cosSimExpTens batched-raw delegation does NOT re-emit the warning
prevWarnState = warning('on', 'MPT:DeprecatedAPI');
lastwarn('', '');
sDummy = cosSimExpTens(A2, [], B2, [], 10, 1, false, true, 1200, ...
    'verbose', false); %#ok<NASGU>
[~, internalWarnId] = lastwarn;
warning(prevWarnState);
results{end+1,1} = 'cosSimExpTens batched: internal call suppresses deprecation';
results{end,2}   = ~strcmp(internalWarnId, 'MPT:DeprecatedAPI');

% spectrum/precision/dedup forwarding (batched-raw only)
spec_fwd = {'harmonic', 12, 'powerlaw', 1};
sBatched_spec = cosSimExpTens(A2, [], B2, [], 10, 1, false, true, 1200, ...
    'spectrum', spec_fwd, 'verbose', false);
warnState = warning('off', 'MPT:DeprecatedAPI');
sBatchOld_spec = batchCosSimExpTens(A2, B2, 10, 1, false, true, 1200, ...
    'spectrum', spec_fwd, 'verbose', false);
warning(warnState);
results{end+1,1} = 'cosSimExpTens batched: ''spectrum'' forwards to batchCosSimExpTens';
results{end,2}   = max(abs(sBatched_spec(:) - sBatchOld_spec(:))) < 1e-14;

% spectrum kwarg rejected in non-batched modes
results{end+1,1} = 'cosSimExpTens scalar: ''spectrum'' kwarg errors';
results{end,2}   = throwsErrorWithId( ...
    @() cosSimExpTens([0, 4, 7], [], [0, 4, 7], [], 10, 1, false, true, 1200, ...
        'spectrum', spec_fwd, 'verbose', false), ...
    'cosSimExpTens:spectrumNotApplicable');

results{end+1,1} = 'cosSimExpTens MA struct: ''spectrum'' kwarg errors';
% Build small MA densities just for this test
densMA_x = buildExpTens({[0; 4; 7]}, [], 0.5, 1, [], false, true, 12, 'verbose', false);
densMA_y = buildExpTens({[0; 4; 7]}, [], 0.5, 1, [], false, true, 12, 'verbose', false);
results{end,2}   = throwsErrorWithId( ...
    @() cosSimExpTens(densMA_x, densMA_y, 'spectrum', spec_fwd, 'verbose', false), ...
    'cosSimExpTens:spectrumNotApplicable');

% --- Broadcasting in batched-raw mode (v2.1.1+) ---
% Reference multiset broadcast against M candidate rows: should match
% the explicit repmat formulation row-by-row.
ref_pitches = [0, 386.31, 701.96];
candidates = [0, 400, 700;
              0, 300, 700;
              0, 300, 600;
              0, 400, 800];
sims_explicit = cosSimExpTens(repmat(ref_pitches, 4, 1), [], candidates, [], ...
    10, 1, false, true, 1200, 'verbose', false);

% (a) 1×K row reference broadcast as P1
sims_bcast_row = cosSimExpTens(ref_pitches, [], candidates, [], ...
    10, 1, false, true, 1200, 'verbose', false);
results{end+1,1} = 'cosSimExpTens batched: 1xK row P1 broadcasts against MxK P2';
results{end,2}   = isequal(size(sims_bcast_row), [4, 1]) && ...
                   max(abs(sims_bcast_row - sims_explicit)) < 1e-12;

% (b) K-by-1 column reference broadcast as P1
sims_bcast_col = cosSimExpTens(ref_pitches.', [], candidates, [], ...
    10, 1, false, true, 1200, 'verbose', false);
results{end+1,1} = 'cosSimExpTens batched: Kx1 column P1 broadcasts against MxK P2';
results{end,2}   = max(abs(sims_bcast_col - sims_explicit)) < 1e-12;

% (c) Symmetric: P1 matrix, P2 vector reference
sims_bcast_p2 = cosSimExpTens(candidates, [], ref_pitches, [], ...
    10, 1, false, true, 1200, 'verbose', false);
% cos sim is symmetric in P1 vs P2 swap, so should equal sims_explicit
results{end+1,1} = 'cosSimExpTens batched: P2 vector broadcasts against MxK P1';
results{end,2}   = max(abs(sims_bcast_p2 - sims_explicit)) < 1e-12;

% (d) Broadcast with non-empty weights: W1 vector broadcast in lockstep
ref_w = [1.0, 0.8, 0.6];
sims_w_bcast = cosSimExpTens(ref_pitches, ref_w, candidates, [], ...
    10, 1, false, true, 1200, 'verbose', false);
sims_w_explicit = cosSimExpTens(repmat(ref_pitches, 4, 1), repmat(ref_w, 4, 1), ...
    candidates, [], 10, 1, false, true, 1200, 'verbose', false);
results{end+1,1} = 'cosSimExpTens batched: W1 vector broadcasts alongside P1';
results{end,2}   = max(abs(sims_w_bcast - sims_w_explicit)) < 1e-12;

% (e) Broadcast composes with 'spectrum' kwarg
sims_bcast_spec = cosSimExpTens(ref_pitches, [], candidates, [], ...
    10, 1, false, true, 1200, 'spectrum', spec_fwd, 'verbose', false);
sims_explicit_spec = cosSimExpTens(repmat(ref_pitches, 4, 1), [], candidates, [], ...
    10, 1, false, true, 1200, 'spectrum', spec_fwd, 'verbose', false);
results{end+1,1} = 'cosSimExpTens batched: broadcast composes with ''spectrum''';
results{end,2}   = max(abs(sims_bcast_spec - sims_explicit_spec)) < 1e-12;

% (f) Mismatched row counts (no broadcast possible) errors clearly
results{end+1,1} = 'cosSimExpTens batched: mismatched row counts errors';
results{end,2}   = throwsErrorWithId( ...
    @() cosSimExpTens(rand(4, 3), [], rand(5, 3), [], ...
        10, 1, false, true, 1200, 'verbose', false), ...
    'MPT:CosSimBatched:RowMismatch');

% --- List-mode broadcasting (v2.1.1+) ---
% Build a small population of density structs.
dRef = buildExpTens([0, 4, 7], [], 0.5, 1, false, true, 12, 'verbose', false);
dC1  = buildExpTens([0, 4, 7], [], 0.5, 1, false, true, 12, 'verbose', false);
dC2  = buildExpTens([0, 3, 7], [], 0.5, 1, false, true, 12, 'verbose', false);
dC3  = buildExpTens([0, 3, 6], [], 0.5, 1, false, true, 12, 'verbose', false);

simExplicit = cosSimExpTens({dRef, dRef, dRef}, {dC1, dC2, dC3}, 'verbose', false);

% (a) Right-broadcast: scalar struct vs cell
simBcastR = cosSimExpTens(dRef, {dC1, dC2, dC3}, 'verbose', false);
results{end+1,1} = 'cosSimExpTens list: scalar vs cell broadcasts (right)';
results{end,2}   = iscell(simBcastR) && numel(simBcastR) == 3 && ...
    abs(simBcastR{1} - simExplicit{1}) < 1e-12 && ...
    abs(simBcastR{2} - simExplicit{2}) < 1e-12 && ...
    abs(simBcastR{3} - simExplicit{3}) < 1e-12;

% (b) Left-broadcast: cell vs scalar struct (symmetric: cosine is symmetric)
simBcastL = cosSimExpTens({dC1, dC2, dC3}, dRef, 'verbose', false);
results{end+1,1} = 'cosSimExpTens list: cell vs scalar broadcasts (left)';
results{end,2}   = iscell(simBcastL) && numel(simBcastL) == 3 && ...
    abs(simBcastL{1} - simExplicit{1}) < 1e-12 && ...
    abs(simBcastL{2} - simExplicit{2}) < 1e-12 && ...
    abs(simBcastL{3} - simExplicit{3}) < 1e-12;

% (c) Length-1 cell still returns length-1 cell (Option II preserved)
simBcastOne = cosSimExpTens(dRef, {dC1}, 'verbose', false);
results{end+1,1} = 'cosSimExpTens list: scalar vs length-1 cell returns length-1 cell';
results{end,2}   = iscell(simBcastOne) && numel(simBcastOne) == 1 && ...
    abs(simBcastOne{1} - simExplicit{1}) < 1e-12;

% (d) Cell + cell with mismatched length still errors clearly
results{end+1,1} = 'cosSimExpTens list: cell+cell length mismatch errors';
results{end,2}   = throwsErrorWithId( ...
    @() cosSimExpTens({dRef, dRef}, {dC1, dC2, dC3}, 'verbose', false), ...
    'MPT:CosSimList:LengthMismatch');

% (e) Cell + non-struct, non-cell (e.g. numeric) errors with bad-broadcast id
results{end+1,1} = 'cosSimExpTens list: cell vs non-struct other-arg errors';
results{end,2}   = throwsErrorWithId( ...
    @() cosSimExpTens({dC1, dC2}, 42, 'verbose', false), ...
    'MPT:CosSimList:BadBroadcast');

% --- v2.1 unified dispatch: evalExpTens list and batched-raw modes ----

% List mode: cell of density structs returns cell of value vectors
de1 = buildExpTens([0, 4, 7], [], 0.5, 1, false, true, 12, 'verbose', false);
de2 = buildExpTens([0, 3, 7], [], 0.5, 1, false, true, 12, 'verbose', false);
xGrid = 0:11;
valsCell = evalExpTens({de1, de2}, xGrid, 'verbose', false);
results{end+1,1} = 'evalExpTens list: returns cell of correct length';
results{end,2}   = iscell(valsCell) && numel(valsCell) == 2;
vals1 = evalExpTens(de1, xGrid, 'verbose', false);
vals2 = evalExpTens(de2, xGrid, 'verbose', false);
results{end+1,1} = 'evalExpTens list: matches scalar dispatch element-wise';
results{end,2}   = max(abs(valsCell{1}(:) - vals1(:))) < 1e-14 ...
                   && max(abs(valsCell{2}(:) - vals2(:))) < 1e-14;

% List mode: per-density X (cell of vectors of length matching density count)
xCell = {0:11, 0:23};
valsCellPerDens = evalExpTens({de1, de2}, xCell, 'verbose', false);
vals1b = evalExpTens(de1, xCell{1}, 'verbose', false);
vals2b = evalExpTens(de2, xCell{2}, 'verbose', false);
results{end+1,1} = 'evalExpTens list: per-density X (cell broadcast disambiguation)';
results{end,2}   = max(abs(valsCellPerDens{1}(:) - vals1b(:))) < 1e-14 ...
                   && max(abs(valsCellPerDens{2}(:) - vals2b(:))) < 1e-14;

% List mode: Option II (length-1 stays length-1)
valsCell1 = evalExpTens({de1}, xGrid, 'verbose', false);
results{end+1,1} = 'evalExpTens list: length-1 returns length-1 cell';
results{end,2}   = iscell(valsCell1) && numel(valsCell1) == 1;

% List mode: non-struct entry errors
results{end+1,1} = 'evalExpTens list: non-struct entry errors';
results{end,2}   = throwsErrorWithId( ...
    @() evalExpTens({de1, [1, 2, 3]}, xGrid, 'verbose', false), ...
    'MPT:EvalList:NonStruct');

% Batched-raw mode: 2-D matrix dispatch returns matrix of values
P_e = [0, 4, 7; 0, 3, 7];   % 2 x 3 matrix (major and minor triads)
xq = 0:11;
valsBatched = evalExpTens(P_e, [], 0.5, 1, false, true, 12, xq, ...
    'verbose', false);
results{end+1,1} = 'evalExpTens batched: returns matrix of correct shape';
results{end,2}   = isnumeric(valsBatched) && isequal(size(valsBatched), [2, 12]);

% Batched-raw matches scalar dispatch row-by-row
vals_row1 = evalExpTens(P_e(1, :), [], 0.5, 1, false, true, 12, xq, ...
    'verbose', false);
vals_row2 = evalExpTens(P_e(2, :), [], 0.5, 1, false, true, 12, xq, ...
    'verbose', false);
results{end+1,1} = 'evalExpTens batched: matches scalar dispatch row-by-row';
results{end,2}   = max(abs(valsBatched(1, :) - vals_row1(:).')) < 1e-12 ...
                   && max(abs(valsBatched(2, :) - vals_row2(:).')) < 1e-12;

% Batched-raw: NaN-padded rows handled (consistent with batchCosSim convention)
P_e_nan = [0, 4, 7, NaN; 0, 3, 7, NaN];
valsNan = evalExpTens(P_e_nan, [], 0.5, 1, false, true, 12, xq, ...
    'verbose', false);
results{end+1,1} = 'evalExpTens batched: NaN-padded rows match unpadded rows';
results{end,2}   = max(abs(valsNan(:) - valsBatched(:))) < 1e-14;

% Row vector falls through to scalar SA raw path (backward compatibility)
vals_row_compat = evalExpTens([0, 4, 7], [], 0.5, 1, false, true, 12, xq, ...
    'verbose', false);
results{end+1,1} = 'evalExpTens batched: row vector falls through to scalar';
results{end,2}   = isnumeric(vals_row_compat) && isvector(vals_row_compat);

% --- Transposition invariance (cosSimExpTens fix) ---

B_diat = [0, 200, 400, 500, 700, 900, 1100];

s0 = cosSimExpTens([0, 400, 700], [], B_diat, [], ...
    10, 2, true, false, 1200, 'verbose', false);
s1 = cosSimExpTens([100, 500, 800], [], B_diat, [], ...
    10, 2, true, false, 1200, 'verbose', false);
results{end+1,1} = 'cosSimExpTens: isRel transposition (non-periodic)';
results{end,2}   = abs(s0 - s1) < 1e-14;

s0 = cosSimExpTens([0, 400, 700], [], B_diat, [], ...
    10, 2, true, true, 1200, 'verbose', false);
s1 = cosSimExpTens([100, 500, 800], [], B_diat, [], ...
    10, 2, true, true, 1200, 'verbose', false);
results{end+1,1} = 'cosSimExpTens: isRel transposition (periodic)';
results{end,2}   = abs(s0 - s1) < 1e-14;

s0 = cosSimExpTens([0, 400, 700], [], B_diat, [], ...
    10, 3, true, true, 1200, 'verbose', false);
s1 = cosSimExpTens([500, 900, 1200], [], B_diat, [], ...
    10, 3, true, true, 1200, 'verbose', false);
results{end+1,1} = 'cosSimExpTens: isRel transposition (periodic, r=3)';
results{end,2}   = abs(s0 - s1) < 1e-14;

shifts = [100, 300, 500, 700, 1100];
s_ref = cosSimExpTens([0, 400, 700], [], B_diat, [], ...
    10, 2, true, true, 1200, 'verbose', false);
allMatch = true;
for c = shifts
    sc = cosSimExpTens([0, 400, 700] + c, [], B_diat, [], ...
        10, 2, true, true, 1200, 'verbose', false);
    if abs(sc - s_ref) >= 1e-14
        allMatch = false;
    end
end
results{end+1,1} = 'cosSimExpTens: isRel all shifts (periodic)';
results{end,2}   = allMatch;

s0 = cosSimExpTens([0, 400, 700], [], B_diat, [], ...
    10, 1, false, true, 1200, 'verbose', false);
s1 = cosSimExpTens([1200, 1600, 1900], [], B_diat, [], ...
    10, 1, false, true, 1200, 'verbose', false);
results{end+1,1} = 'cosSimExpTens: isPer octave equivalence';
results{end,2}   = abs(s0 - s1) < 1e-14;

w = [1.0, 0.8, 0.6];
s0 = cosSimExpTens([0, 400, 700], w, B_diat, [], ...
    10, 2, true, true, 1200, 'verbose', false);
s1 = cosSimExpTens([100, 500, 800], w, B_diat, [], ...
    10, 2, true, true, 1200, 'verbose', false);
results{end+1,1} = 'cosSimExpTens: isRel+isPer with weights';
results{end,2}   = abs(s0 - s1) < 1e-14;

A3 = [0, 400, 700; 1200, 1600, 1900; 0, 400, 700];
B3 = repmat(B_diat, 3, 1);
sb = batchCosSimExpTens(A3, B3, 10, 1, false, true, 1200, ...
    'verbose', false);
results{end+1,1} = 'batchCosSimExpTens: octave deduplication';
results{end,2}   = abs(sb(1) - sb(2)) < 1e-14 && ...
                    abs(sb(1) - sb(3)) < 1e-14;

%% ---- Entropy ----

H = nTupleEntropy([0, 2, 4, 6, 8, 10], 12);
results{end+1,1} = 'nTupleEntropy: whole-tone = 0';
results{end,2}   = abs(H) < 1e-10;

H = nTupleEntropy([0, 2, 4, 5, 7, 9, 11], 12, 2, 'normalize', false);
results{end+1,1} = 'nTupleEntropy: diatonic 2-tuple ≈ 1.56';
results{end,2}   = abs(H - 1.56) < 0.01;

H_raw = nTupleEntropy([0, 2, 4, 5, 7, 9, 11], 12, 1);
H_smooth = nTupleEntropy([0, 2, 4, 5, 7, 9, 11], 12, 1, 'sigma', 0.2);
results{end+1,1} = 'nTupleEntropy: smoothing increases H';
results{end,2}   = H_smooth > H_raw;

H = entropyExpTens(0:11, ones(1,12), 100, 1, false, true, 12, 'verbose', false);
results{end+1,1} = 'entropyExpTens: uniform ≈ 1';
results{end,2}   = H > 0.95;

% --- v2.1 unified dispatch: entropyExpTens list and batched-raw modes ----

% List mode: cell of density structs returns cell of entropy values
de1 = buildExpTens([0, 4, 7], [], 50, 1, false, true, 1200, 'verbose', false);
de2 = buildExpTens([0, 3, 7], [], 50, 1, false, true, 1200, 'verbose', false);
HCell = entropyExpTens({de1, de2}, 'verbose', false);
results{end+1,1} = 'entropyExpTens list: returns cell of correct length';
results{end,2}   = iscell(HCell) && numel(HCell) == 2;
H1 = entropyExpTens(de1, 'verbose', false);
H2 = entropyExpTens(de2, 'verbose', false);
results{end+1,1} = 'entropyExpTens list: matches scalar dispatch element-wise';
results{end,2}   = abs(HCell{1} - H1) < 1e-14 ...
                   && abs(HCell{2} - H2) < 1e-14;

% List mode: Option II (length-1 stays length-1)
HCell1 = entropyExpTens({de1}, 'verbose', false);
results{end+1,1} = 'entropyExpTens list: length-1 returns length-1 cell';
results{end,2}   = iscell(HCell1) && numel(HCell1) == 1;

% List mode: name-value pairs forwarded
HCellNorm = entropyExpTens({de1, de2}, 'normalize', false, 'base', exp(1), 'verbose', false);
H1_nat = entropyExpTens(de1, 'normalize', false, 'base', exp(1), 'verbose', false);
results{end+1,1} = 'entropyExpTens list: name-value pairs forwarded';
results{end,2}   = abs(HCellNorm{1} - H1_nat) < 1e-12;

% List mode: non-struct entry errors
results{end+1,1} = 'entropyExpTens list: non-struct entry errors';
results{end,2}   = throwsErrorWithId( ...
    @() entropyExpTens({de1, [1, 2, 3]}, 'verbose', false), ...
    'MPT:EntropyList:NonStruct');

% Batched-raw mode: 2-D pitch matrix returns vector of entropies
P_h = [0, 4, 7; 0, 3, 7];
H_batched = entropyExpTens(P_h, [], 50, 1, false, true, 1200, 'verbose', false);
results{end+1,1} = 'entropyExpTens batched: returns vector of correct length';
results{end,2}   = isnumeric(H_batched) && numel(H_batched) == 2;

% Batched-raw matches scalar dispatch row-by-row
H_row1 = entropyExpTens(P_h(1, :), [], 50, 1, false, true, 1200, 'verbose', false);
H_row2 = entropyExpTens(P_h(2, :), [], 50, 1, false, true, 1200, 'verbose', false);
results{end+1,1} = 'entropyExpTens batched: matches scalar dispatch row-by-row';
results{end,2}   = abs(H_batched(1) - H_row1) < 1e-12 ...
                   && abs(H_batched(2) - H_row2) < 1e-12;

% Batched-raw: NaN-padded rows
P_h_nan = [0, 4, 7, NaN; 0, 3, 7, NaN];
H_batched_nan = entropyExpTens(P_h_nan, [], 50, 1, false, true, 1200, 'verbose', false);
results{end+1,1} = 'entropyExpTens batched: NaN-padded rows match unpadded';
results{end,2}   = max(abs(H_batched_nan - H_batched)) < 1e-14;

% Batched-raw: insufficient pitches in a row gives NaN
P_h_short = [0, 4, 7; 0, NaN, NaN];   % second row has only 1 valid pitch
H_short = entropyExpTens(P_h_short, [], 50, 2, false, true, 1200, 'verbose', false);
results{end+1,1} = 'entropyExpTens batched: row with too few pitches returns NaN';
results{end,2}   = ~isnan(H_short(1)) && isnan(H_short(2));

% --- v2.1 fix: SA entropy with dim > 1 (previously errored) -----------

% r = 2, isRel = false: dim = 2. Build a periodic dyad density and
% compute its entropy via the new Cartesian grid path.
H_dim2_per = entropyExpTens([0, 4, 7], [], 100, 2, false, true, 1200, ...
    'nPointsPerDim', 60, 'verbose', false);
results{end+1,1} = 'entropyExpTens SA dim=2 periodic: returns finite value';
results{end,2}   = isfinite(H_dim2_per) && H_dim2_per > 0 && H_dim2_per <= 1;

% Non-periodic with explicit bounds
H_dim2_nonper = entropyExpTens([0, 400, 700], [], 12, 2, false, false, 1200, ...
    'xMin', -100, 'xMax', 800, 'nPointsPerDim', 60, 'verbose', false);
results{end+1,1} = 'entropyExpTens SA dim=2 non-periodic: returns finite value';
results{end,2}   = isfinite(H_dim2_nonper) && H_dim2_nonper > 0 && H_dim2_nonper <= 1;

% gridLimit guard
results{end+1,1} = 'entropyExpTens SA dim=2: gridLimit guard fires';
results{end,2}   = throwsErrorWithId( ...
    @() entropyExpTens([0, 400, 700], [], 12, 2, false, true, 1200, ...
        'nPointsPerDim', 1200, 'gridLimit', 1e3, 'verbose', false), ...
    'entropyExpTens:gridLimitExceeded');

% Empty weights treated as uniform (same as buildExpTens convention)
H_uni = entropyExpTens(0:11, [], 100, 1, false, true, 12, 'verbose', false);
H_ones = entropyExpTens(0:11, ones(1, 12), 100, 1, false, true, 12, 'verbose', false);
results{end+1,1} = 'entropyExpTens: w=[] equivalent to ones(1,N)';
results{end,2}   = abs(H_uni - H_ones) < 1e-14;

%% ---- DFT Monte Carlo: dftCircularSimulate, balance/evenness/projCentroid sigma (v2.1) ----
%
%  Tests for the MC sigma additions to balance, evenness, projCentroid,
%  and the new dftCircularSimulate function. Insert after the existing
%  Circular measures section (before sigmaSpace section, or just after
%  the existing dftCircular tests).

% --- dftCircularSimulate: sigma=0 exact -----------------------------

[m, s] = dftCircularSimulate([0, 200, 400, 500, 700, 900, 1100], [], ...
                              1200, 0, 'nDraws', 100, 'rngSeed', 42);
[~, magDet] = dftCircular([0, 200, 400, 500, 700, 900, 1100], [], 1200);
results{end+1,1} = 'dftCircularSimulate: sigma=0 mean = deterministic';
results{end,2}   = max(abs(m - magDet)) < 1e-12;
results{end+1,1} = 'dftCircularSimulate: sigma=0 SD = 0';
results{end,2}   = max(abs(s)) < 1e-12;

% --- dftCircularSimulate: small sigma -> deterministic --------------

[m, s] = dftCircularSimulate([0, 200, 400, 500, 700, 900, 1100], [], ...
                              1200, 1e-3, 'nDraws', 2000, 'rngSeed', 42);
results{end+1,1} = 'dftCircularSimulate: small sigma close to deterministic';
results{end,2}   = max(abs(m - magDet)) < 1e-3 && max(s) < 1e-3;

% --- dftCircularSimulate: closed-form E[|F(0)|^2] for augmented triad ---
%   Augmented triad has F_det(0) = 0, so:
%   E[|F(0)|^2] = (1 - alpha_1^2) * sum(w^2) / sum(w)^2 = (1 - alpha_1^2) / K
period = 1200; sigma = 50; K = 3;
alpha1 = exp(-2 * pi^2 * sigma^2 / period^2);
expectedF0sq = (1 - alpha1^2) / K;
[~, ~, samples] = dftCircularSimulate([0, 400, 800], [], period, sigma, ...
                                      'nDraws', 50000, 'rngSeed', 42);
mcF0sq = mean(samples(:, 1).^2);
results{end+1,1} = 'dftCircularSimulate: closed-form E[|F(0)|^2] (aug triad)';
results{end,2}   = abs(mcF0sq - expectedF0sq) < 5e-3;

% --- dftCircularSimulate: rngSeed reproducibility -------------------

m1 = dftCircularSimulate([0, 200, 400, 500, 700, 900, 1100], [], 1200, ...
                         50, 'nDraws', 1000, 'rngSeed', 42);
m2 = dftCircularSimulate([0, 200, 400, 500, 700, 900, 1100], [], 1200, ...
                         50, 'nDraws', 1000, 'rngSeed', 42);
results{end+1,1} = 'dftCircularSimulate: rngSeed reproducible';
results{end,2}   = isequal(m1, m2);

% --- dftCircularSimulate: returnSamples shape -----------------------

[~, ~, samples] = dftCircularSimulate([0, 200, 400, 500, 700, 900, 1100], ...
                                      [], 1200, 50, ...
                                      'nDraws', 500, 'rngSeed', 42);
results{end+1,1} = 'dftCircularSimulate: samples shape [nDraws x K]';
results{end,2}   = isequal(size(samples), [500, 7]);

% --- balanceCircular: sigma=0 backward compatibility ----------------

b = balanceCircular([0, 400, 800], [], 1200);
results{end+1,1} = 'balanceCircular: sigma=0 default scalar = 1 (aug triad)';
results{end,2}   = abs(b - 1) < 1e-10;

% --- balanceCircular: sigma=0 with explicit sigma argument ----------

b = balanceCircular([0, 400, 800], [], 1200, 0);
results{end+1,1} = 'balanceCircular: sigma=0 explicit = 1 (aug triad)';
results{end,2}   = abs(b - 1) < 1e-10;

% --- balanceCircular: sigma>0 returns Rayleigh bias ------------------

[b, bs] = balanceCircular([0, 400, 800], [], 1200, 50, ...
                          'nDraws', 50000, 'rngSeed', 42);
expectedRayleigh = sqrt((1 - alpha1^2) * pi / (4 * K));
results{end+1,1} = 'balanceCircular: sigma>0 reveals Rayleigh bias';
results{end,2}   = b < 1 && bs > 0 && ...
                   abs((1 - b) - expectedRayleigh) < 5e-3;

% --- balanceCircular: nDraws name-value works -----------------------

b = balanceCircular([0, 400, 800], [], 1200, 25, 'nDraws', 5000, 'rngSeed', 7);
results{end+1,1} = 'balanceCircular: accepts nDraws/rngSeed name-value args';
results{end,2}   = isfinite(b) && b >= 0 && b <= 1;

% --- evennessCircular: sigma=0 backward compatibility ---------------

e = evennessCircular([0, 200, 400, 600, 800, 1000], 1200);
results{end+1,1} = 'evennessCircular: sigma=0 default = 1 (whole-tone)';
results{end,2}   = abs(e - 1) < 1e-10;

% --- evennessCircular: sigma>0 returns scalar -----------------------

e = evennessCircular([0, 200, 400, 600, 800, 1000], 1200, 50, ...
                     'nDraws', 5000, 'rngSeed', 42);
results{end+1,1} = 'evennessCircular: sigma>0 returns valid scalar';
results{end,2}   = isfinite(e) && e >= 0 && e <= 1;

% --- evennessCircular: smoothing reduces evenness for irregular pattern ---

eDet = evennessCircular([0, 200, 400, 500, 700, 900, 1100], 1200);
eSmooth = evennessCircular([0, 200, 400, 500, 700, 900, 1100], 1200, 100, ...
                           'nDraws', 20000, 'rngSeed', 42);
results{end+1,1} = 'evennessCircular: smoothing reduces |F(1)| for diatonic';
results{end,2}   = eSmooth < eDet;

% --- evennessCircular: SD output ------------------------------------

[e, es] = evennessCircular([0, 200, 400, 500, 700, 900, 1100], 1200, 50, ...
                           'nDraws', 5000, 'rngSeed', 42);
results{end+1,1} = 'evennessCircular: SD output positive when sigma>0';
results{end,2}   = es > 0;

% --- projCentroid: alpha_1 damping is exact (closed-form) -----------

period = 12; sigma = 0.5;
alpha1 = exp(-2 * pi^2 * sigma^2 / period^2);
yDet = projCentroid([0, 4, 7], [], period);
ySmooth = projCentroid([0, 4, 7], [], period, [], sigma);
results{end+1,1} = 'projCentroid: y_smooth = alpha_1 * y_det (exact)';
results{end,2}   = max(abs(ySmooth - alpha1 * yDet)) < 1e-12;

% --- projCentroid: phase preserved ----------------------------------

[~, ~, cpDet] = projCentroid([0, 4, 7], [], 12);
[~, ~, cpSmooth] = projCentroid([0, 4, 7], [], 12, [], 1.0);
results{end+1,1} = 'projCentroid: phase preserved under sigma';
results{end,2}   = abs(cpDet - cpSmooth) < 1e-12;

% --- projCentroid: cent_mag damped by alpha_1 -----------------------

period = 1200; sigma = 100;
alpha1 = exp(-2 * pi^2 * sigma^2 / period^2);
[~, cmDet] = projCentroid([0, 200, 400, 500, 700, 900, 1100], [], period);
[~, cmSmooth] = projCentroid([0, 200, 400, 500, 700, 900, 1100], [], period, ...
                              [], sigma);
results{end+1,1} = 'projCentroid: cent_mag damped by alpha_1';
results{end,2}   = abs(cmSmooth - alpha1 * cmDet) < 1e-12;

% --- projCentroid: sigma=0 recovers v2 ------------------------------

[y0, cm0, cp0] = projCentroid([0, 4, 7], [], 12, [], 0);
[y1, cm1, cp1] = projCentroid([0, 4, 7], [], 12);
results{end+1,1} = 'projCentroid: sigma=0 explicit = v2';
results{end,2}   = isequal(y0, y1) && cm0 == cm1 && cp0 == cp1;


%% ---- sigmaSpace: position-aware soft measures (v2.1) ----
%
%  Tests for the sigma + sigmaSpace additions to sameness,
%  coherence, and nTupleEntropy (v2.1.0). Insert after the existing
%  Circular measures and Entropy sections.

% --- positionVariance helper: signed-coefficient cases --------------

V = positionVariance([1, 2, 3, 4], [+1, -1, -1, +1], 1.0);
results{end+1,1} = 'positionVariance: disjoint endpoints -> 4 sigma^2';
results{end,2}   = abs(V - 4) < 1e-12;

V = positionVariance([1, 2, 1, 3], [+1, -1, -1, +1], 1.0);
results{end+1,1} = 'positionVariance: shared cancelling -> 2 sigma^2';
results{end,2}   = abs(V - 2) < 1e-12;

V = positionVariance([2, 1, 1, 2], [+1, -1, -1, +1], 1.0);
results{end+1,1} = 'positionVariance: shared reinforcing -> 8 sigma^2';
results{end,2}   = abs(V - 8) < 1e-12;

V = positionVariance([1, 2], [+1, -1], 0.5);
results{end+1,1} = 'positionVariance: scales with sigma^2';
results{end,2}   = abs(V - 0.5) < 1e-12;

% --- sameness: sigma = 0 byte-equivalence with v2 -------------------

[sq0, nd0] = sameness([0, 2, 4, 5, 7, 9, 11], 12, 0);
[sq1, nd1] = sameness([0, 2, 4, 5, 7, 9, 11], 12);     % default sigma = 0
results{end+1,1} = 'sameness: sigma=0 vs default agree (diatonic)';
results{end,2}   = sq0 == sq1 && nd0 == nd1;

[sqW0, ndW0] = sameness([0, 2, 4, 6, 8, 10], 12, 0);
results{end+1,1} = 'sameness: sigma=0 whole-tone perfect';
results{end,2}   = abs(sqW0 - 1) < 1e-12 && ndW0 == 0;

% --- sameness: sigma=0 flags coincide -------------------------------

[sqP, ~] = sameness([0, 2, 4, 5, 7, 9, 11], 12, 0, 'sigmaSpace', 'position');
[sqI, ~] = sameness([0, 2, 4, 5, 7, 9, 11], 12, 0, 'sigmaSpace', 'interval');
results{end+1,1} = 'sameness: sigma=0 flags coincide';
results{end,2}   = abs(sqP - sqI) < 1e-12;

% --- sameness: sigma>0 numerical regression -------------------------
%
%  Reference numbers from the Python verification of the same
%  implementation (computed at the same sigma values) — see
%  the v2.1 design discussion. These pin down the soft-path
%  computation against future regressions.

[sqP, ~] = sameness([0, 2, 4, 5, 7, 9, 11], 12, 0.5, ...
                    'sigmaSpace', 'position');
[sqI, ~] = sameness([0, 2, 4, 5, 7, 9, 11], 12, 0.5, ...
                    'sigmaSpace', 'interval');
results{end+1,1} = 'sameness: diatonic sigma=0.5 position ~ 0.4224';
results{end,2}   = abs(sqP - 0.422420) < 1e-4;
results{end+1,1} = 'sameness: diatonic sigma=0.5 interval ~ 0.7144';
results{end,2}   = abs(sqI - 0.714369) < 1e-4;
results{end+1,1} = 'sameness: position more aggressive than interval';
results{end,2}   = sqP < sqI;

% --- sameness: float positions accepted when sigma > 0 --------------

ji = [0, 203.91, 386.31, 498.04, 701.96, 884.36, 1088.27];
[sqJI, ~] = sameness(ji, 1200, 25);
results{end+1,1} = 'sameness: JI diatonic accepts float p (sigma>0)';
results{end,2}   = isfinite(sqJI) && sqJI > 0 && sqJI <= 1.0;

% --- sameness: integer required at sigma = 0 ------------------------

results{end+1,1} = 'sameness: float p errors at sigma=0';
results{end,2}   = throwsErrorWithId( ...
    @() sameness([0.5, 2, 4, 7], 12, 0), ...
    'sameness:nonIntegerPositions');

% --- sameness: invalid sigmaSpace errors ----------------------------

results{end+1,1} = 'sameness: invalid sigmaSpace errors';
results{end,2}   = throwsError( ...
    @() sameness([0, 2, 4, 5, 7, 9, 11], 12, 0.5, 'sigmaSpace', 'bogus'));

% --- coherence: sigma = 0 byte-equivalence with v2 ------------------

[c0, nc0] = coherence([0, 2, 4, 5, 7, 9, 11], 12, 0);
[c1, nc1] = coherence([0, 2, 4, 5, 7, 9, 11], 12);   % default sigma = 0
results{end+1,1} = 'coherence: sigma=0 vs default agree';
results{end,2}   = c0 == c1 && nc0 == nc1;

[c0, nc0] = coherence([0, 2, 4, 5, 7, 9, 11], 12, 0, 'strict', false);
results{end+1,1} = 'coherence: sigma=0 strict=false (diatonic) -> nc=0';
results{end,2}   = nc0 == 0 && abs(c0 - 1.0) < 1e-12;

% --- coherence: sigma>0 numerical regression -------------------------

[cP, ~] = coherence([0, 2, 4, 5, 7, 9, 11], 12, 0.5, ...
                    'sigmaSpace', 'position');
[cI, ~] = coherence([0, 2, 4, 5, 7, 9, 11], 12, 0.5, ...
                    'sigmaSpace', 'interval');
results{end+1,1} = 'coherence: diatonic sigma=0.5 position ~ 0.8735';
results{end,2}   = abs(cP - 0.873485) < 1e-4;
results{end+1,1} = 'coherence: diatonic sigma=0.5 interval ~ 0.9446';
results{end,2}   = abs(cI - 0.944610) < 1e-4;

% --- coherence: tritone gives 0.5 contribution under position -------
%
%  The diatonic tritone is a fourth (F-B) and a fifth (B-F) sharing
%  endpoints with reinforcing signs. Var(D2 - D1) = 8 sigma^2 and
%  the means coincide exactly (both = 6 chromatic steps), so
%  P(D2 <= D1) = 0.5 at every sigma > 0. The soft-path nc as
%  sigma -> 0 therefore approaches 0.5 (one tritone, contributing
%  half a failure), giving c -> 1 - 0.5/140 = 0.99643.
[cLim, ~] = coherence([0, 2, 4, 5, 7, 9, 11], 12, 1e-6);
results{end+1,1} = 'coherence: sigma->0+ limit handles tritone tie';
results{end,2}   = abs(cLim - (1 - 0.5/140)) < 1e-3;

% --- coherence: invalid sigmaSpace errors ---------------------------

results{end+1,1} = 'coherence: invalid sigmaSpace errors';
results{end,2}   = throwsError( ...
    @() coherence([0, 2, 4, 5, 7, 9, 11], 12, 0.5, 'sigmaSpace', 'bogus'));

% --- nTupleEntropy: sigma=0 byte-equivalence with v2 ----------------

H0a = nTupleEntropy([0, 2, 4, 5, 7, 9, 11], 12);
H0b = nTupleEntropy([0, 2, 4, 5, 7, 9, 11], 12, 1, 'sigma', 0);
results{end+1,1} = 'nTupleEntropy: sigma=0 equals default';
results{end,2}   = abs(H0a - H0b) < 1e-12;

% --- nTupleEntropy: n=1 exactness (position == interval with sigma*sqrt(2)) ---

sigmaTest = 0.5;
HposN1 = nTupleEntropy([0, 2, 4, 5, 7, 9, 11], 12, 1, ...
                       'sigma', sigmaTest, 'sigmaSpace', 'position');
HintN1 = nTupleEntropy([0, 2, 4, 5, 7, 9, 11], 12, 1, ...
                       'sigma', sigmaTest * sqrt(2), 'sigmaSpace', 'interval');
results{end+1,1} = 'nTupleEntropy: n=1 position(sigma) = interval(sigma*sqrt(2))';
results{end,2}   = abs(HposN1 - HintN1) < 1e-10;

% --- nTupleEntropy: smoothing increases entropy ---------------------

H_raw    = nTupleEntropy([0, 2, 4, 5, 7, 9, 11], 12, 1);
H_smooth = nTupleEntropy([0, 2, 4, 5, 7, 9, 11], 12, 1, 'sigma', 0.2);
results{end+1,1} = 'nTupleEntropy: smoothing increases H (n=1, position)';
results{end,2}   = H_smooth > H_raw;

% --- nTupleEntropy: position with same sigma > interval (because sqrt(2) wider) ---

Hpos = nTupleEntropy([0, 2, 4, 5, 7, 9, 11], 12, 1, ...
                     'sigma', 0.3, 'sigmaSpace', 'position');
Hint = nTupleEntropy([0, 2, 4, 5, 7, 9, 11], 12, 1, ...
                     'sigma', 0.3, 'sigmaSpace', 'interval');
results{end+1,1} = 'nTupleEntropy: position smoother than interval at same sigma';
results{end,2}   = Hpos > Hint;

% --- nTupleEntropy: float positions accepted when sigma > 0 ---------

H_ji = nTupleEntropy(ji, 1200, 1, 'sigma', 25);
results{end+1,1} = 'nTupleEntropy: JI diatonic accepts float p (sigma>0)';
results{end,2}   = isfinite(H_ji) && H_ji > 0;

% --- nTupleEntropy: integer required at sigma = 0 -------------------

results{end+1,1} = 'nTupleEntropy: float p errors at sigma=0';
results{end,2}   = throwsErrorWithId( ...
    @() nTupleEntropy([0.5, 2, 4, 5, 7, 9, 11], 12, 1), ...
    'nTupleEntropy:nonIntegerPositions');

% --- nTupleEntropy: warning at n>=2 with sigmaSpace=position --------
%
%  At n>=2 with sigmaSpace='position', the marginal-matched
%  approximation triggers a warning (suppressible via the
%  warning ID).

origState = warning('off', 'nTupleEntropy:positionApprox');
cleanupObj = onCleanup(@() warning(origState));
lastwarn('');
H_n2_pos = nTupleEntropy([0, 2, 4, 5, 7, 9, 11], 12, 2, ...
                         'sigma', 0.3, 'sigmaSpace', 'position');
results{end+1,1} = 'nTupleEntropy: n=2 position computes (warning suppressed)';
results{end,2}   = isfinite(H_n2_pos) && H_n2_pos > 0;
clear cleanupObj;

% Verify the warning fires when not suppressed:
warning('on', 'nTupleEntropy:positionApprox');
lastwarn('');
H_n2_pos = nTupleEntropy([0, 2, 4, 5, 7, 9, 11], 12, 2, ...
                         'sigma', 0.3, 'sigmaSpace', 'position');
[~, warnId] = lastwarn;
results{end+1,1} = 'nTupleEntropy: n>=2 position issues approxApprox warning';
results{end,2}   = strcmp(warnId, 'nTupleEntropy:positionApprox');


%% ---- Harmony ----

r = roughness(440, 1);
results{end+1,1} = 'roughness: unison = 0';
results{end,2}   = abs(r) < 1e-10;

r = roughness([300, 330], [1, 1]);
results{end+1,1} = 'roughness: positive for nearby freqs';
results{end,2}   = r > 0;

spec = {'harmonic', 24, 'powerlaw', 1};
H_ji = spectralEntropy([0, 386.31, 701.96], [], 12, 'spectrum', spec, 'verbose', false);
H_edo = spectralEntropy([0, 400, 700], [], 12, 'spectrum', spec, 'verbose', false);
results{end+1,1} = 'spectralEntropy: JI < EDO';
results{end,2}   = H_ji < H_edo;

% --- spectralEntropy 2-D batched dispatch (Bundle 2, v2.1+) ---
P_se = [0, 386.31, 701.96; 0, 400, 700; 0, 100, 200];
H_se_batch = spectralEntropy(P_se, [], 12, 'spectrum', spec, 'verbose', false);
results{end+1,1} = 'spectralEntropy batched: returns column vector of length nRows';
results{end,2}   = isequal(size(H_se_batch), [3, 1]);

% Per-row matches scalar
H_se_row1 = spectralEntropy(P_se(1, :), [], 12, 'spectrum', spec, 'verbose', false);
H_se_row2 = spectralEntropy(P_se(2, :), [], 12, 'spectrum', spec, 'verbose', false);
results{end+1,1} = 'spectralEntropy batched: matches scalar dispatch row-by-row';
results{end,2}   = abs(H_se_batch(1) - H_se_row1) < 1e-12 ...
                && abs(H_se_batch(2) - H_se_row2) < 1e-12;

% Dedup: two rows that are transpositions of each other give identical entropy
P_se_dup = [0, 400, 700; 100, 500, 800];   % row 2 = row 1 + 100
H_se_dup = spectralEntropy(P_se_dup, [], 12, 'spectrum', spec, 'verbose', false);
results{end+1,1} = 'spectralEntropy batched: transposition dedup';
results{end,2}   = abs(H_se_dup(1) - H_se_dup(2)) < 1e-12;

% NaN-padded variable cardinality
P_se_pad = [0, 400, 700, NaN; 0, 1200, NaN, NaN; 0, 300, 600, 900];
H_se_pad = spectralEntropy(P_se_pad, [], 12, 'spectrum', spec, 'verbose', false);
results{end+1,1} = 'spectralEntropy batched: NaN-padded cardinality OK';
results{end,2}   = all(~isnan(H_se_pad)) && isequal(size(H_se_pad), [3, 1]);

% All-NaN row returns NaN
P_se_allnan = [0, 400, 700; NaN, NaN, NaN];
H_se_allnan = spectralEntropy(P_se_allnan, [], 12, 'spectrum', spec, 'verbose', false);
results{end+1,1} = 'spectralEntropy batched: all-NaN row returns NaN';
results{end,2}   = ~isnan(H_se_allnan(1)) && isnan(H_se_allnan(2));

% Verbose printing tests
% Note (commit 14+): estimateCompTime default minPrintSec is 10,
% so verbose=true
% for typical fast inputs is silent. The print path itself is exercised
% via the batched-mode tests below and via direct estimateCompTime tests.
outSEScalar = evalc('spectralEntropy([0, 400, 700], [], 12, ''spectrum'', spec, ''verbose'', true);');
results{end+1,1} = 'spectralEntropy scalar: verbose=true silent for fast call';
results{end,2}   = isempty(strtrim(outSEScalar));

outSEScalarSilent = evalc('spectralEntropy([0, 400, 700], [], 12, ''spectrum'', spec, ''verbose'', false);');
results{end+1,1} = 'spectralEntropy scalar: verbose=false silent';
results{end,2}   = isempty(strtrim(outSEScalarSilent));

outSEBatch = evalc('spectralEntropy(P_se, [], 12, ''spectrum'', spec, ''verbose'', true);');
results{end+1,1} = 'spectralEntropy batched: verbose=true silent for fast call';
results{end,2}   = isempty(strtrim(outSEBatch));

% --- entropyExpTens batched verbose (Bundle 2) ---
P_ee = [0, 100, 200, 300; 0, 200, 400, 600; 0, 100, 200, 300];
outEEBatch = evalc(['entropyExpTens(P_ee, [], 12, 1, false, false, 1200, ' ...
    '''xMin'', 0, ''xMax'', 600, ''verbose'', true);']);
results{end+1,1} = 'entropyExpTens batched: verbose=true silent for fast call';
results{end,2}   = isempty(strtrim(outEEBatch));

outEEBatchSilent = evalc(['entropyExpTens(P_ee, [], 12, 1, false, false, 1200, ' ...
    '''xMin'', 0, ''xMax'', 600, ''verbose'', false);']);
results{end+1,1} = 'entropyExpTens batched: verbose=false silent';
results{end,2}   = isempty(strtrim(outEEBatchSilent));

% Numerical results unchanged by verbose flag
H_ee_v = entropyExpTens(P_ee, [], 12, 1, false, false, 1200, ...
    'xMin', 0, 'xMax', 600, 'verbose', true);
H_ee_q = entropyExpTens(P_ee, [], 12, 1, false, false, 1200, ...
    'xMin', 0, 'xMax', 600, 'verbose', false);
results{end+1,1} = 'entropyExpTens batched: verbose flag does not affect outputs';
results{end,2}   = isequaln(H_ee_v, H_ee_q);

[hMax, hEnt] = templateHarmonicity([0, 400, 700], [], 12, 'verbose', false);
results{end+1,1} = 'templateHarmonicity: hMax in (0,1]';
results{end,2}   = hMax > 0 && hMax <= 1;
results{end+1,1} = 'templateHarmonicity: hEntropy in (0,1]';
results{end,2}   = hEnt > 0 && hEnt <= 1;

% -- templateHarmonicity: hEntropy lower for octave than for cluster --
[~, hEnt_oct] = templateHarmonicity([0, 1200], [], 12, 'verbose', false);
[~, hEnt_clu] = templateHarmonicity([0, 100, 200], [], 12, 'verbose', false);
results{end+1,1} = 'templateHarmonicity: hEntropy octave < cluster';
results{end,2}   = hEnt_oct < hEnt_clu;

% -- templateHarmonicity: hEntropy lower for major triad than for cluster
% (3-note vs 3-note, controlling for cardinality) --
[~, hEnt_maj] = templateHarmonicity([0, 400, 700], [], 12, 'verbose', false);
[~, hEnt_clu] = templateHarmonicity([0, 100, 200], [], 12, 'verbose', false);
results{end+1,1} = 'templateHarmonicity: hEntropy major triad < cluster';
results{end,2}   = hEnt_maj < hEnt_clu;

spec = {'harmonic', 12, 'powerlaw', 1};
h_uni = tensorHarmonicity([0, 0], [], 12, 'spectrum', spec, 'verbose', false);
h_tri = tensorHarmonicity([0, 600], [], 12, 'spectrum', spec, 'verbose', false);
results{end+1,1} = 'tensorHarmonicity: unison > tritone';
results{end,2}   = h_uni > h_tri;

% -- tensorHarmonicity: ordered ranking
% octave (2:1) > perfect 5th (3:2) > major triad (4:5:6) > minor triad --
h_oct  = tensorHarmonicity([0, 1200],     [], 12, 'spectrum', spec, 'verbose', false);
h_p5   = tensorHarmonicity([0, 700],      [], 12, 'spectrum', spec, 'verbose', false);
h_maj  = tensorHarmonicity([0, 400, 700], [], 12, 'spectrum', spec, 'verbose', false);
h_min  = tensorHarmonicity([0, 300, 700], [], 12, 'spectrum', spec, 'verbose', false);
results{end+1,1} = 'tensorHarmonicity: octave > P5 > major > minor';
results{end,2}   = (h_oct > h_p5) && (h_p5 > h_maj) && (h_maj > h_min);

[vp_p, vp_w] = virtualPitches([0, 400, 700], [], 12, 'verbose', false);
results{end+1,1} = 'virtualPitches: non-empty';
results{end,2}   = numel(vp_p) > 0;
results{end+1,1} = 'virtualPitches: lengths match';
results{end,2}   = numel(vp_p) == numel(vp_w);

% -- virtualPitches: peak of a single pitch sits at the pitch itself --
[vp_p1, vp_w1] = virtualPitches(400, [], 12, 'verbose', false);
[~, i_max1] = max(vp_w1);
results{end+1,1} = 'virtualPitches: single pitch peak at the pitch';
results{end,2}   = abs(vp_p1(i_max1) - 400) < 5;

% -- virtualPitches: peak of an octave dyad at the lower note
% (partials 1 and 2 of a template at 0 align with both chord notes) --
[vp_p2, vp_w2] = virtualPitches([0, 1200], [], 12, 'verbose', false);
[~, i_max2] = max(vp_w2);
results{end+1,1} = 'virtualPitches: octave peak at lower note';
results{end,2}   = abs(vp_p2(i_max2)) < 5;

% --- v2.1 unified dispatch: harmony wrappers batched mode ---

% tensorHarmonicity: 2-D matrix dispatch returns column vector
P_h = [0, 1200, 0, 0;     % unison-with-octave (4-pitch); cardinality 4
       0, 700, 0, 0;       % open-fifth-doubled (4-pitch); cardinality 4
       0, 400, 700, 0];    % major triad (3-pitch with NaN pad — wait, 0 is a pitch)
% The above is ambiguous because 0 is a valid pitch. Use NaN padding instead.
P_th = [0, 1200,  NaN;
        0, 400,   700;
        0, 300,   700];
spec_th = {'harmonic', 12, 'powerlaw', 1};
h_th = tensorHarmonicity(P_th, [], 12, 'spectrum', spec_th, 'verbose', false);
results{end+1,1} = 'tensorHarmonicity batched: returns column vector of correct length';
results{end,2}   = isnumeric(h_th) && isequal(size(h_th), [3, 1]);

% Matches scalar dispatch row-by-row (NaN dropped per row)
h_oct  = tensorHarmonicity([0, 1200],     [], 12, 'spectrum', spec_th, 'verbose', false);
h_maj3 = tensorHarmonicity([0, 400, 700], [], 12, 'spectrum', spec_th, 'verbose', false);
h_min3 = tensorHarmonicity([0, 300, 700], [], 12, 'spectrum', spec_th, 'verbose', false);
results{end+1,1} = 'tensorHarmonicity batched: matches scalar dispatch row-by-row';
results{end,2}   = abs(h_th(1) - h_oct)  < 1e-12 ...
                   && abs(h_th(2) - h_maj3) < 1e-12 ...
                   && abs(h_th(3) - h_min3) < 1e-12;

% Major > minor preserved across batched rows (existing scalar property)
results{end+1,1} = 'tensorHarmonicity batched: major > minor (octave > both)';
results{end,2}   = h_th(1) > h_th(2) && h_th(2) > h_th(3);

% Row with fewer than 2 valid pitches returns NaN
P_th_short = [0, 400, 700;  NaN, NaN, NaN; 0, NaN, NaN];   % row 2 empty, row 3 has 1
h_th_short = tensorHarmonicity(P_th_short, [], 12, 'spectrum', spec_th, 'verbose', false);
results{end+1,1} = 'tensorHarmonicity batched: insufficient pitches return NaN';
results{end,2}   = ~isnan(h_th_short(1)) && isnan(h_th_short(2)) && isnan(h_th_short(3));

% --- tensorHarmonicity verbose / estimateCompTime integration (v2.1.1+) ---
% Scalar verbose=true forwards to buildExpTens, which prints its own estimate.
outScalarVerb = evalc(['tensorHarmonicity([0, 400, 700], [], 12, ' ...
    '''spectrum'', spec, ''verbose'', true);']);
results{end+1,1} = 'tensorHarmonicity scalar: verbose=true prints something';
results{end,2}   = ~isempty(strtrim(outScalarVerb));

outScalarSilent = evalc(['tensorHarmonicity([0, 400, 700], [], 12, ' ...
    '''spectrum'', spec, ''verbose'', false);']);
results{end+1,1} = 'tensorHarmonicity scalar: verbose=false silent';
results{end,2}   = isempty(strtrim(outScalarSilent));

outBatchVerb = evalc(['tensorHarmonicity(P_th, [], 12, ''spectrum'', spec_th, ' ...
    '''verbose'', true);']);
results{end+1,1} = 'tensorHarmonicity batched: verbose=true silent for fast call';
results{end,2}   = isempty(strtrim(outBatchVerb));

outBatchSilent = evalc(['tensorHarmonicity(P_th, [], 12, ''spectrum'', spec_th, ' ...
    '''verbose'', false);']);
results{end+1,1} = 'tensorHarmonicity batched: verbose=false silent';
results{end,2}   = isempty(strtrim(outBatchSilent));

[hA] = tensorHarmonicity([0, 400, 700], [], 12, 'spectrum', spec, 'verbose', true);
[hB] = tensorHarmonicity([0, 400, 700], [], 12, 'spectrum', spec, 'verbose', false);
results{end+1,1} = 'tensorHarmonicity: verbose flag does not affect outputs';
results{end,2}   = (hA == hB);

% templateHarmonicity batched
[hMax_b, hEnt_b] = templateHarmonicity(P_th, [], 12, 'verbose', false);
results{end+1,1} = 'templateHarmonicity batched: hMax shape';
results{end,2}   = isequal(size(hMax_b), [3, 1]);
results{end+1,1} = 'templateHarmonicity batched: hEntropy shape';
results{end,2}   = isequal(size(hEnt_b), [3, 1]);

% Matches scalar row-by-row
[hMax_oct,  hEnt_oct]  = templateHarmonicity([0, 1200],     [], 12, 'verbose', false);
[hMax_maj3, hEnt_maj3] = templateHarmonicity([0, 400, 700], [], 12, 'verbose', false);
results{end+1,1} = 'templateHarmonicity batched: matches scalar dispatch row-by-row';
results{end,2}   = abs(hMax_b(1) - hMax_oct)  < 1e-12 ...
                   && abs(hEnt_b(1) - hEnt_oct)  < 1e-12 ...
                   && abs(hMax_b(2) - hMax_maj3) < 1e-12 ...
                   && abs(hEnt_b(2) - hEnt_maj3) < 1e-12;

% --- templateHarmonicity batched: chord-side dedup (v2.2+) ---
% Two rows that are transpositions of each other must produce
% identical hMax and hEntropy (template-harmonicity transposes
% internally so any shift cancels). Same for permutations.
P_th_trans = [0, 400, 700; 100, 500, 800; 1200, 1600, 1900];
[hMaxT_th, hEntT_th] = templateHarmonicity(P_th_trans, [], 12, 'verbose', false);
results{end+1,1} = 'templateHarmonicity batched: transposition dedup (hMax)';
results{end,2}   = abs(hMaxT_th(1) - hMaxT_th(2)) < 1e-12 ...
                && abs(hMaxT_th(1) - hMaxT_th(3)) < 1e-12;
results{end+1,1} = 'templateHarmonicity batched: transposition dedup (hEntropy)';
results{end,2}   = abs(hEntT_th(1) - hEntT_th(2)) < 1e-12 ...
                && abs(hEntT_th(1) - hEntT_th(3)) < 1e-12;

P_th_perm = [0, 400, 700; 700, 0, 400; 400, 700, 0];
[hMaxP_th, hEntP_th] = templateHarmonicity(P_th_perm, [], 12, 'verbose', false);
results{end+1,1} = 'templateHarmonicity batched: permutation dedup (hMax)';
results{end,2}   = abs(hMaxP_th(1) - hMaxP_th(2)) < 1e-12 ...
                && abs(hMaxP_th(1) - hMaxP_th(3)) < 1e-12;
results{end+1,1} = 'templateHarmonicity batched: permutation dedup (hEntropy)';
results{end,2}   = abs(hEntP_th(1) - hEntP_th(2)) < 1e-12 ...
                && abs(hEntP_th(1) - hEntP_th(3)) < 1e-12;
clear P_th_trans P_th_perm hMaxT_th hEntT_th hMaxP_th hEntP_th

% --- templateHarmonicity verbose / estimateCompTime integration (v2.1.1+) ---
% Note (commit 14+): estimateCompTime default minPrintSec is 10,
% so verbose=true
% for fast inputs is silent.

% Scalar verbose=true is silent for typical fast inputs (sub-half-sec)
outScalarVerb = evalc('templateHarmonicity([0, 400, 700], [], 12, ''verbose'', true);');
results{end+1,1} = 'templateHarmonicity scalar: verbose=true silent for fast call';
results{end,2}   = isempty(strtrim(outScalarVerb));

% Scalar verbose=false suppresses output entirely
outScalarSilent = evalc('templateHarmonicity([0, 400, 700], [], 12, ''verbose'', false);');
results{end+1,1} = 'templateHarmonicity scalar: verbose=false silent';
results{end,2}   = isempty(strtrim(outScalarSilent));

% Default verbose is true but fast scalar still silent under threshold
outDefault = evalc('templateHarmonicity([0, 400, 700], [], 12);');
results{end+1,1} = 'templateHarmonicity scalar: default verbose=true silent for fast call';
results{end,2}   = isempty(strtrim(outDefault));

% Batched verbose=true silent for fast call (new contract: 10s threshold)
outBatchVerb = evalc('templateHarmonicity([0, 400, 700; 0, 300, 700; 0, 300, 600], [], 12, ''verbose'', true);');
results{end+1,1} = 'templateHarmonicity batched: verbose=true silent for fast call';
results{end,2}   = isempty(strtrim(outBatchVerb));

% Batched verbose=false silent
outBatchSilent = evalc('templateHarmonicity([0, 400, 700; 0, 300, 700], [], 12, ''verbose'', false);');
results{end+1,1} = 'templateHarmonicity batched: verbose=false silent';
results{end,2}   = isempty(strtrim(outBatchSilent));

% Numerical results unaffected by verbose flag
[hMaxA, hEntA] = templateHarmonicity([0, 400, 700], [], 12, 'verbose', true);
[hMaxB, hEntB] = templateHarmonicity([0, 400, 700], [], 12, 'verbose', false);
results{end+1,1} = 'templateHarmonicity: verbose flag does not affect outputs';
results{end,2}   = (hMaxA == hMaxB) && (hEntA == hEntB);

% All-NaN batched input does not crash and gives NaN output
P_allnan = NaN(3, 3);
outAllNaN = evalc('[hMax_n, hEnt_n] = templateHarmonicity(P_allnan, [], 12, ''verbose'', true);');
results{end+1,1} = 'templateHarmonicity batched: all-NaN rows yield NaN, no crash';
% Need to capture outputs — re-run without evalc to get them
[hMax_n, hEnt_n] = templateHarmonicity(P_allnan, [], 12, 'verbose', false);
results{end,2}   = isequal(size(hMax_n), [3, 1]) && all(isnan(hMax_n)) ...
                && isequal(size(hEnt_n), [3, 1]) && all(isnan(hEnt_n));

% virtualPitches batched: cell-of-arrays output
[vp_pcell, vp_wcell] = virtualPitches(P_th, [], 12, 'verbose', false);
results{end+1,1} = 'virtualPitches batched: vp_p is 1-by-nRows cell';
results{end,2}   = iscell(vp_pcell) && isequal(size(vp_pcell), [1, 3]);
results{end+1,1} = 'virtualPitches batched: vp_w is 1-by-nRows cell';
results{end,2}   = iscell(vp_wcell) && isequal(size(vp_wcell), [1, 3]);

% Each cell entry matches scalar dispatch
[vp_p_oct,  vp_w_oct]  = virtualPitches([0, 1200],     [], 12, 'verbose', false);
[vp_p_maj,  vp_w_maj]  = virtualPitches([0, 400, 700], [], 12, 'verbose', false);
results{end+1,1} = 'virtualPitches batched: cell entries match scalar dispatch';
results{end,2}   = numel(vp_pcell{1}) == numel(vp_p_oct) ...
                   && max(abs(vp_pcell{1} - vp_p_oct)) < 1e-12 ...
                   && numel(vp_pcell{2}) == numel(vp_p_maj) ...
                   && max(abs(vp_pcell{2} - vp_p_maj)) < 1e-12;

% NaN-padded rows produce same outputs as unpadded
P_clean = [0, 1200; 0, 400; 0, 300];   % no NaN padding (cardinality 2)
P_padded = [0, 1200, NaN; 0, 400, NaN; 0, 300, NaN];
spec_simple = {'harmonic', 12, 'powerlaw', 1};
h_clean  = tensorHarmonicity(P_clean,  [], 12, 'spectrum', spec_simple, 'verbose', false);
h_padded = tensorHarmonicity(P_padded, [], 12, 'spectrum', spec_simple, 'verbose', false);
results{end+1,1} = 'tensorHarmonicity batched: NaN-padded rows match unpadded';
results{end,2}   = max(abs(h_clean - h_padded)) < 1e-12;

% Scalar (vector) input still dispatches to scalar path
h_scalar_check = tensorHarmonicity([0, 400, 700], [], 12, 'spectrum', spec_th, 'verbose', false);
results{end+1,1} = 'tensorHarmonicity scalar: row vector still works (backward compat)';
results{end,2}   = isscalar(h_scalar_check) && abs(h_scalar_check - h_maj3) < 1e-14;

% --- virtualPitches verbose / estimateCompTime integration (v2.1.1+) ---
% Note (commit 14+): estimateCompTime default minPrintSec is 10,
% so verbose=true
% for fast inputs is silent.
outVPScalarVerb = evalc('virtualPitches([0, 400, 700], [], 12, ''verbose'', true);');
results{end+1,1} = 'virtualPitches scalar: verbose=true silent for fast call';
results{end,2}   = isempty(strtrim(outVPScalarVerb));

outVPScalarSilent = evalc('virtualPitches([0, 400, 700], [], 12, ''verbose'', false);');
results{end+1,1} = 'virtualPitches scalar: verbose=false silent';
results{end,2}   = isempty(strtrim(outVPScalarSilent));

outVPDefault = evalc('virtualPitches([0, 400, 700], [], 12);');
results{end+1,1} = 'virtualPitches scalar: default verbose=true silent for fast call';
results{end,2}   = isempty(strtrim(outVPDefault));

outVPBatchVerb = evalc('virtualPitches([0, 400, 700; 0, 300, 700], [], 12, ''verbose'', true);');
results{end+1,1} = 'virtualPitches batched: verbose=true silent for fast call';
results{end,2}   = isempty(strtrim(outVPBatchVerb));

outVPBatchSilent = evalc('virtualPitches([0, 400, 700; 0, 300, 700], [], 12, ''verbose'', false);');
results{end+1,1} = 'virtualPitches batched: verbose=false silent';
results{end,2}   = isempty(strtrim(outVPBatchSilent));

[vp_pA, vp_wA] = virtualPitches([0, 400, 700], [], 12, 'verbose', true);
[vp_pB, vp_wB] = virtualPitches([0, 400, 700], [], 12, 'verbose', false);
results{end+1,1} = 'virtualPitches: verbose flag does not affect outputs';
results{end,2}   = isequal(vp_pA, vp_pB) && isequal(vp_wA, vp_wB);

%% ---- estimateCompTime ----

est = estimateCompTime(1000, 2, 'test');
results{end+1,1} = 'estimateCompTime: positive';
results{end,2}   = est > 0;

% Default threshold (10 s): tiny work doesn't print
outTinyEC = evalc('estimateCompTime(100, 1, ''tinywork'', true);');
results{end+1,1} = 'estimateCompTime: default threshold suppresses sub-10s estimates';
results{end,2}   = isempty(strtrim(outTinyEC));

% Pass minPrintSec=0 to recover always-print behaviour
outZeroEC = evalc('estimateCompTime(100, 1, ''tinywork'', true, 0);');
results{end+1,1} = 'estimateCompTime: minPrintSec=0 prints regardless of size';
results{end,2}   = contains(outZeroEC, 'tinywork');

% Every printed estimate carries the Ctrl+C suffix
outPrintedEC = evalc('estimateCompTime(100, 1, ''tinywork'', true, 0);');
results{end+1,1} = 'estimateCompTime: printed estimate includes Ctrl+C suffix';
results{end,2}   = contains(outPrintedEC, '(Ctrl+C to cancel)');

%% ---- printBatchedEstimate ----

% Default threshold (10 s): short estimate suppressed
outShortPB = evalc('printBatchedEstimate(''foo'', 100, 5.0);');
results{end+1,1} = 'printBatchedEstimate: 5 s < 10 s default threshold is silent';
results{end,2}   = isempty(strtrim(outShortPB));

% Long estimate prints
outLongPB = evalc('printBatchedEstimate(''foo'', 100, 30.0);');
results{end+1,1} = 'printBatchedEstimate: 30 s > 10 s threshold prints';
results{end,2}   = contains(outLongPB, 'foo') ...
                && contains(outLongPB, 'batched, 100 rows') ...
                && contains(outLongPB, 'estimated time') ...
                && contains(outLongPB, '(Ctrl+C to cancel)');

% verbose=false silences regardless
outFalsePB = evalc('printBatchedEstimate(''foo'', 100, 30.0, false);');
results{end+1,1} = 'printBatchedEstimate: verbose=false silences large estimates';
results{end,2}   = isempty(strtrim(outFalsePB));

% Explicit minPrintSec=0 prints sub-10s
outZeroPB = evalc('printBatchedEstimate(''foo'', 100, 0.05, true, 0);');
results{end+1,1} = 'printBatchedEstimate: minPrintSec=0 prints regardless';
results{end,2}   = contains(outZeroPB, 'foo') && contains(outZeroPB, '50 ms');

%% ---- Input validation ----

results{end+1,1} = 'buildExpTens: r too large errors';
results{end,2}   = throwsError(@() buildExpTens([0, 4], [], 10, 3, ...
    false, true, 12, 'verbose', false));

% v2.2: SA isRel + r=1 was a hard error in v2.0/v2.1; relaxed to a
% degenerate warning that parallels the MA path's behaviour. The build
% itself succeeds (dim = 0); the warning flags the unusual regime.
w_state_sa1rel = warning('on', 'buildExpTens:isRelDegenerate');
lastwarn('');
dens_sa1rel = buildExpTens([0, 4, 7], [], 10, 1, true, true, 12, ...
    'verbose', false);
[~, lastID_sa1rel] = lastwarn;
warning(w_state_sa1rel);
results{end+1,1} = 'buildExpTens: SA isRel + r=1 emits degenerate warning (was error in v2.1)';
results{end,2}   = strcmp(lastID_sa1rel, 'buildExpTens:isRelDegenerate') ...
                   && isstruct(dens_sa1rel) && dens_sa1rel.dim == 0;

results{end+1,1} = 'coherence: duplicates error';
results{end,2}   = throwsError(@() coherence([0, 0, 4, 7], 12));

results{end+1,1} = 'nTupleEntropy: n too large errors';
results{end,2}   = throwsError(@() nTupleEntropy([0, 2, 4], 12, 3));

%% ---- Serial-position features: continuity ----

% --- Discrete-limit examples ---
[c, m] = continuity([3;5;7;7;9], 11, 0, 'mode', 'strict');
results{end+1,1} = 'continuity: strict count = 1, mag = 2';
results{end,2}   = abs(c - 1) < 1e-10 && abs(m - 2) < 1e-10;

[c, m] = continuity([3;5;7;7;9], 11, 0, 'mode', 'lenient');
results{end+1,1} = 'continuity: lenient count = 3, mag = 6';
results{end,2}   = abs(c - 3) < 1e-10 && abs(m - 6) < 1e-10;

% --- Query = seq(end) gives 0 ---
[c, m] = continuity([3;5;7;7;9], 9, 0);
results{end+1,1} = 'continuity: query = seq(end) gives 0';
results{end,2}   = abs(c) < 1e-10 && abs(m) < 1e-10;

% --- Multi-query shape ---
[c, m] = continuity([3;5;7;7;9], [10;11;12], 0, 'mode', 'lenient');
results{end+1,1} = 'continuity: multi-query output shape';
results{end,2}   = isequal(size(c), [3, 1]) && isequal(size(m), [3, 1]);

% --- N < 2 ---
[c, m] = continuity(5, [11;12;13], 0);
results{end+1,1} = 'continuity: N<2 gives zero vectors';
results{end,2}   = all(c == 0) && all(m == 0);

% --- Descending query on ascending seq ---
[c, m] = continuity([1;2;3;4;5], 4, 0, 'mode', 'lenient');
results{end+1,1} = 'continuity: descending query on ascending seq = 0';
results{end,2}   = abs(c) < 1e-10 && abs(m) < 1e-10;

% --- Slope via magnitude/count ---
[c, m] = continuity([1;4;7;10;13], 16, 0, 'mode', 'lenient');
results{end+1,1} = 'continuity: slope = magnitude / count = 3';
results{end,2}   = abs(c - 4) < 1e-10 && abs(m - 12) < 1e-10 && ...
                   abs(m/c - 3) < 1e-10;

% --- Signed magnitude for descending ---
[c, m] = continuity([10;8;6;4], 2, 0, 'mode', 'strict');
results{end+1,1} = 'continuity: descending seq gives negative magnitude';
results{end,2}   = abs(c - 3) < 1e-10 && abs(m + 6) < 1e-10;

% --- Smoothing monotone ---
[c_small, ~] = continuity([3;5;7;7;9], 11, 0.01, 'mode', 'lenient');
[c_large, ~] = continuity([3;5;7;7;9], 11, 1.0,  'mode', 'lenient');
results{end+1,1} = 'continuity: smoothing lowers count when sigma large';
results{end,2}   = c_large < c_small;

% --- Explicit theta ---
[c_theta, ~] = continuity([3;5;7;7;9], 11, 0, 'theta', -1);
results{end+1,1} = 'continuity: explicit theta = -1 matches lenient';
results{end,2}   = abs(c_theta - 3) < 1e-10;

% --- theta out of range ---
results{end+1,1} = 'continuity: theta out of range errors';
results{end,2}   = throwsError(@() continuity([3;5;7], 8, 0, ...
    'theta', 2));

% --- Weight argument (v2.1.0) ---

% w = [] matches default
[c1, m1] = continuity([3;5;7;7;9], 11, 0, 'mode', 'lenient');
[c2, m2] = continuity([3;5;7;7;9], 11, 0, 'mode', 'lenient', 'w', []);
results{end+1,1} = 'continuity: w=[] matches default';
results{end,2}   = abs(c1 - c2) < 1e-12 && abs(m1 - m2) < 1e-12;

% Scalar w = 1 equals unweighted
[c_un, m_un] = continuity([3;5;7;7;9], 11, 0, 'mode', 'lenient');
[c_w,  m_w ] = continuity([3;5;7;7;9], 11, 0, 'mode', 'lenient', ...
    'w', 1);
results{end+1,1} = 'continuity: scalar w=1 equals unweighted';
results{end,2}   = abs(c_un - c_w) < 1e-12 && abs(m_un - m_w) < 1e-12;

% Scalar w = 0.5 scales outputs by 0.25 (rolling-product gives c^2)
[c_un, m_un] = continuity([3;5;7;7;9], 11, 0, 'mode', 'lenient');
[c_w,  m_w ] = continuity([3;5;7;7;9], 11, 0, 'mode', 'lenient', ...
    'w', 0.5);
results{end+1,1} = 'continuity: scalar w=0.5 scales by 0.25';
results{end,2}   = abs(c_w - 0.25 * c_un) < 1e-12 && ...
                   abs(m_w - 0.25 * m_un) < 1e-12;

% Recency zero-out: only the most-recent interval contributes
[c, m] = continuity([3;5;7;7;9], 11, 0, 'mode', 'lenient', ...
    'w', [0 0 0 1 1]);
results{end+1,1} = 'continuity: recency w truncates to last interval';
results{end,2}   = abs(c - 1) < 1e-12 && abs(m - 2) < 1e-12;

% Per-event w = [1 1 0.5 1 1] hand-calculation:
%   diff events: (3->5)=2, (5->7)=2, (7->7)=0, (7->9)=2
%   diff weights: 1*1=1, 1*0.5=0.5, 0.5*1=0.5, 1*1=1
%   lenient walk: 1*1 + 0.5*1 + 0 + 1*1 = 2.5 count; 2+1+0+2 = 5 mag
[c, m] = continuity([3;5;7;7;9], 11, 0, 'mode', 'lenient', ...
    'w', [1 1 0.5 1 1]);
results{end+1,1} = 'continuity: per-event w matches rolling product';
results{end,2}   = abs(c - 2.5) < 1e-12 && abs(m - 5) < 1e-12;

% w composes with sigma smoothing: uniform scalar c -> scale by c^2
[c_un, m_un] = continuity([3;5;7;7;9], 11, 0.3, 'mode', 'lenient');
[c_w,  m_w ] = continuity([3;5;7;7;9], 11, 0.3, 'mode', 'lenient', ...
    'w', 0.7);
results{end+1,1} = 'continuity: w composes with sigma smoothing';
results{end,2}   = abs(c_w - 0.49 * c_un) < 1e-12 && ...
                   abs(m_w - 0.49 * m_un) < 1e-12;

% Wrong-length w errors
results{end+1,1} = 'continuity: wrong-length w errors';
results{end,2}   = throwsError(@() continuity([3;5;7;7;9], 11, 0, ...
    'w', [1 1 1]));

% Negative weights error
results{end+1,1} = 'continuity: negative weights error';
results{end,2}   = throwsError(@() continuity([3;5;7;7;9], 11, 0, ...
    'w', [1 1 -0.1 1 1]));

% Negative scalar weight errors
results{end+1,1} = 'continuity: negative scalar w errors';
results{end,2}   = throwsError(@() continuity([3;5;7;7;9], 11, 0, ...
    'w', -0.5));

%% ---- Serial-position features: seqWeights ----

v = seqWeights([], 'primacy', 'N', 5);
results{end+1,1} = 'seqWeights: primacy -> [1;0;0;0;0]';
results{end,2}   = isequal(v, [1;0;0;0;0]);

v = seqWeights([], 'recency', 'N', 5);
results{end+1,1} = 'seqWeights: recency -> [0;0;0;0;1]';
results{end,2}   = isequal(v, [0;0;0;0;1]);

v = seqWeights([], 'exponentialFromEnd', 'N', 5, 'decayRate', 0);
results{end+1,1} = 'seqWeights: zero decay gives uniform';
results{end,2}   = all(abs(v - 1) < 1e-10);

v = seqWeights([], 'uShape', 'N', 5, 'decayRate', 0.5, 'alpha', 0.5);
results{end+1,1} = 'seqWeights: uShape alpha=0.5 symmetric';
results{end,2}   = max(abs(v - flipud(v))) < 1e-10;

v = seqWeights([], [0.1;0.2;0.4;0.2;0.1], 'N', 5);
results{end+1,1} = 'seqWeights: numeric vector passthrough';
results{end,2}   = isequal(v, [0.1;0.2;0.4;0.2;0.1]);

results{end+1,1} = 'seqWeights: profile length mismatch errors';
results{end,2}   = throwsError(@() seqWeights([], [0.1;0.2;0.3], 'N', 5));

results{end+1,1} = 'seqWeights: unknown spec errors';
results{end,2}   = throwsError(@() seqWeights([], 'wibble', 'N', 5));

% --- w as [] (uniform) matches explicit ones ---
v_empty = seqWeights([], 'exponentialFromEnd', 'N', 5, 'decayRate', 0.5);
v_ones  = seqWeights(ones(5,1), 'exponentialFromEnd', 'decayRate', 0.5);
results{end+1,1} = 'seqWeights: [] for w matches ones';
results{end,2}   = max(abs(v_empty - v_ones)) < 1e-10;

% --- w multiplies profile pointwise ---
w = [0.8; 0.5; 1.0; 0.3; 0.9];
v = seqWeights(w, 'recency');
results{end+1,1} = 'seqWeights: w multiplies profile (recency picks w(end))';
results{end,2}   = isequal(v, [0;0;0;0;0.9]);

% --- w multiplies explicit profile vector ---
w = [2;2;2];
profile = [0.1; 0.5; 0.4];
v = seqWeights(w, profile);
results{end+1,1} = 'seqWeights: w multiplies explicit profile';
results{end,2}   = max(abs(v - 2*profile)) < 1e-10;

% --- N vs length(w) mismatch errors ---
results{end+1,1} = 'seqWeights: N vs length(w) mismatch errors';
results{end,2}   = throwsError(@() seqWeights([1;2;3], 'flat', 'N', 5));

% --- N required when w is [] ---
results{end+1,1} = 'seqWeights: missing N with [] w errors';
results{end,2}   = throwsError(@() seqWeights([], 'flat'));

% --- N required when w is scalar ---
results{end+1,1} = 'seqWeights: missing N with scalar w errors';
results{end,2}   = throwsError(@() seqWeights(0.5, 'flat'));

% --- Scalar w broadcasts to length N ---
v = seqWeights(0.5, 'flat', 'N', 4);
results{end+1,1} = 'seqWeights: scalar w broadcasts';
results{end,2}   = max(abs(v - 0.5*ones(4,1))) < 1e-10;

% --- N inferred from w matches explicit N ---
w = [0.2; 0.8; 0.5];
v_inf = seqWeights(w, 'recency');
v_exp = seqWeights(w, 'recency', 'N', 3);
results{end+1,1} = 'seqWeights: inferred N matches explicit N';
results{end,2}   = max(abs(v_inf - v_exp)) < 1e-10;

%% ---- MAET (multi-attribute expectation tensor, v2.1.0) ----
% Tests the multi-attribute path of buildExpTens. The SA path is covered
% by the Expectation tensors section above; these tests focus on
% MAET-specific behaviours: SA-equivalence under the degenerate
% (N=1, A=1) mapping, per-attribute enumeration, weight broadcasting,
% group canonicalisation, NaN padding, and the new error paths.

% -- SA-equivalence: MA with (N=1, A=1, K x 1 column w) reproduces SA --

p_sa = [0; 400; 700];
w_sa = [1; 0.7; 0.5];
sigma = 10; r_ = 2; isPer_ = true; period_ = 1200;

% --- Lazy/eager parity (v2.2) ---
%
% buildExpTens defaults to skinny (lazy=true); ensureExpTensExpensive
% populates the per-tuple fields on demand. The eager and ensured-lazy
% paths must produce structurally identical structs.

dens_eager_sa  = buildExpTens(p_sa, w_sa, sigma, r_, true, isPer_, period_, ...
    'lazy', false, 'verbose', false);
dens_skinny_sa = buildExpTens(p_sa, w_sa, sigma, r_, true, isPer_, period_, ...
    'verbose', false);
results{end+1,1} = 'lazy: SA skinny default has only cheap fields';
results{end,2}   = ~isfield(dens_skinny_sa, 'Centres') ...
                && ~isfield(dens_skinny_sa, 'U_perm') ...
                && ~isfield(dens_skinny_sa, 'nJ');
results{end+1,1} = 'lazy: SA skinny exposes dim';
results{end,2}   = isfield(dens_skinny_sa, 'dim') ...
                && dens_skinny_sa.dim == dens_eager_sa.dim;
dens_filled_sa = ensureExpTensExpensive(dens_skinny_sa);
results{end+1,1} = 'lazy: SA ensure -> matches eager Centres';
results{end,2}   = isequal(dens_filled_sa.Centres, dens_eager_sa.Centres);
results{end+1,1} = 'lazy: SA ensure -> matches eager wJ';
results{end,2}   = isequal(dens_filled_sa.wJ, dens_eager_sa.wJ);
results{end+1,1} = 'lazy: SA ensure -> matches eager U_perm';
results{end,2}   = isequal(dens_filled_sa.U_perm, dens_eager_sa.U_perm);
results{end+1,1} = 'lazy: SA ensure idempotent';
dens_twice_sa = ensureExpTensExpensive(dens_filled_sa);
results{end,2}   = isequal(dens_twice_sa, dens_filled_sa);

dens_eager_ma  = buildExpTens({p_sa}, {w_sa}, sigma, r_, [], true, isPer_, period_, ...
    'lazy', false, 'verbose', false);
dens_skinny_ma = buildExpTens({p_sa}, {w_sa}, sigma, r_, [], true, isPer_, period_, ...
    'verbose', false);
results{end+1,1} = 'lazy: MA skinny default has only cheap fields';
results{end,2}   = ~isfield(dens_skinny_ma, 'Centres') ...
                && ~isfield(dens_skinny_ma, 'U_perm') ...
                && ~isfield(dens_skinny_ma, 'nJ');
results{end+1,1} = 'lazy: MA skinny exposes dim and dimPerAttr';
results{end,2}   = isfield(dens_skinny_ma, 'dim') ...
                && isfield(dens_skinny_ma, 'dimPerAttr') ...
                && isequal(dens_skinny_ma.dim, dens_eager_ma.dim) ...
                && isequal(dens_skinny_ma.dimPerAttr, dens_eager_ma.dimPerAttr);
dens_filled_ma = ensureExpTensExpensive(dens_skinny_ma);
results{end+1,1} = 'lazy: MA ensure -> matches eager Centres';
results{end,2}   = isequal(dens_filled_ma.Centres, dens_eager_ma.Centres);
results{end+1,1} = 'lazy: MA ensure -> matches eager wJ and wv_comb';
results{end,2}   = isequal(dens_filled_ma.wJ, dens_eager_ma.wJ) ...
                && isequal(dens_filled_ma.wv_comb, dens_eager_ma.wv_comb);
results{end+1,1} = 'lazy: MA ensure idempotent';
dens_twice_ma = ensureExpTensExpensive(dens_filled_ma);
results{end,2}   = isequal(dens_twice_ma, dens_filled_ma);

% Consumers transparently handle skinny input (cosSimExpTens, evalExpTens).
results{end+1,1} = 'lazy: cosSimExpTens accepts skinny dens (SA self-similarity = 1)';
s_self = cosSimExpTens(dens_skinny_sa, dens_skinny_sa, 'verbose', false);
results{end,2}   = abs(s_self - 1) < 1e-12;
results{end+1,1} = 'lazy: evalExpTens accepts skinny dens';
v_skinny = evalExpTens(dens_skinny_sa, [0 100 350], 'verbose', false);
v_eager  = evalExpTens(dens_eager_sa,  [0 100 350], 'verbose', false);
results{end,2}   = max(abs(v_skinny - v_eager)) < 1e-12;

for isRel_ = [false, true]
    dens_sa = buildExpTens(p_sa, w_sa, sigma, r_, isRel_, isPer_, period_, ...
        'lazy', false, 'verbose', false);
    dens_ma = buildExpTens({p_sa}, {w_sa}, sigma, r_, [], isRel_, isPer_, period_, ...
        'lazy', false, 'verbose', false);

    relTag = sprintf(' (isRel=%d)', isRel_);
    results{end+1,1} = ['MAET: SA-equivalence tag' relTag];
    results{end,2}   = strcmp(dens_ma.tag, 'MaetDensity'); %#ok<*SAGROW>

    results{end+1,1} = ['MAET: SA-equivalence nJ' relTag];
    results{end,2}   = dens_ma.nJ == dens_sa.nJ;

    results{end+1,1} = ['MAET: SA-equivalence U_perm' relTag];
    results{end,2}   = isequal(dens_ma.U_perm{1}, dens_sa.U_perm);

    results{end+1,1} = ['MAET: SA-equivalence V_comb' relTag];
    results{end,2}   = isequal(dens_ma.V_comb{1}, dens_sa.V_comb);

    results{end+1,1} = ['MAET: SA-equivalence Centres' relTag];
    results{end,2}   = isequal(dens_ma.Centres{1}, dens_sa.Centres);

    results{end+1,1} = ['MAET: SA-equivalence wJ' relTag];
    results{end,2}   = max(abs(dens_ma.wJ - dens_sa.wJ)) < 1e-12;

    results{end+1,1} = ['MAET: SA-equivalence wv_comb' relTag];
    results{end,2}   = max(abs(dens_ma.wv_comb - dens_sa.wv_comb)) < 1e-12;
end

% Dimensionality reduction under isRel=true
results{end+1,1} = 'MAET: Centres dim reduction (isRel=true, r=2)';
dens_ma = buildExpTens({p_sa}, {w_sa}, sigma, 2, [], true, true, 1200, ...
    'lazy', false, 'verbose', false);
results{end,2}   = isequal(size(dens_ma.Centres{1}), [1, dens_ma.nJ]);

% -- Struct basics for pitch + time --

pitchMat = [0 12; 4 15; 7 19];       % 3 x 2
timeMat  = [0 1];                     % 1 x 2
dens = buildExpTens({pitchMat, timeMat}, [], ...
    [10, 0.1], [3, 1], [], [true, false], [true, false], [1200, 0], ...
    'verbose', false);

results{end+1,1} = 'MAET: struct nAttrs';
results{end,2}   = dens.nAttrs == 2;
results{end+1,1} = 'MAET: struct nGroups';
results{end,2}   = dens.nGroups == 2;
results{end+1,1} = 'MAET: struct N';
results{end,2}   = dens.N == 2;
results{end+1,1} = 'MAET: struct groupOfAttr default';
results{end,2}   = isequal(dens.groupOfAttr, [1 2]);
results{end+1,1} = 'MAET: struct r';
results{end,2}   = isequal(dens.r, [3 1]);
results{end+1,1} = 'MAET: struct K';
results{end,2}   = isequal(dens.K, [3 1]);
results{end+1,1} = 'MAET: struct dim';
results{end,2}   = dens.dim == 3;
results{end+1,1} = 'MAET: struct dimPerAttr';
results{end,2}   = isequal(dens.dimPerAttr, [2 1]);

% -- Cartesian product count and event bookkeeping --

pitchMat = [0 12 5; 4 15 9; 7 19 12];   % 3 x 3
timeMat  = [0 1 2];                      % 1 x 3
dens = buildExpTens({pitchMat, timeMat}, [], ...
    [10, 0.1], [3, 1], [], [true, false], [true, false], [1200, 0], ...
    'lazy', false, 'verbose', false);

results{end+1,1} = 'MAET: nJ = sum of per-event Cartesian products';
results{end,2}   = dens.nJ == 18;   % 3 events * P(3,3)=6 perms * 1 time = 18
results{end+1,1} = 'MAET: nK = sum of per-event Cartesian product (comb)';
results{end,2}   = dens.nK == 3;    % 3 events * C(3,3)=1 comb * 1 time = 3
results{end+1,1} = 'MAET: eventOfJ';
results{end,2}   = isequal(dens.eventOfJ, repelem(1:3, 6));
results{end+1,1} = 'MAET: eventOfK';
results{end,2}   = isequal(dens.eventOfK, 1:3);

% -- Weight broadcasting --

pitchMat = [0 4; 4 8];                   % K=2, N=2
dens = buildExpTens({pitchMat}, [], 10, 2, [], false, true, 1200, 'verbose', false);
results{end+1,1} = 'MAET: weight [] -> ones';
results{end,2}   = isequal(dens.w{1}, ones(2, 2));

dens = buildExpTens({pitchMat}, 0.5, 10, 2, [], false, true, 1200, 'verbose', false);
results{end+1,1} = 'MAET: weight scalar top-level';
results{end,2}   = isequal(dens.w{1}, 0.5 * ones(2, 2));

pitchMat = [0 4 5; 4 8 6];               % K=2, N=3
wRow = [0.5, 1.0, 2.0];                  % 1 x N
dens = buildExpTens({pitchMat}, {wRow}, 10, 2, [], false, true, 1200, 'verbose', false);
results{end+1,1} = 'MAET: weight 1 x N row broadcast';
results{end,2}   = isequal(dens.w{1}, [0.5 1.0 2.0; 0.5 1.0 2.0]);

pitchMat = [0 4 5; 4 8 6; 7 10 9];       % K=3, N=3
wCol = [0.5; 1.0; 2.0];                  % K x 1
dens = buildExpTens({pitchMat}, {wCol}, 10, 2, [], false, true, 1200, 'verbose', false);
results{end+1,1} = 'MAET: weight K x 1 column broadcast';
results{end,2}   = isequal(dens.w{1}, repmat([0.5; 1.0; 2.0], 1, 3));

pitchMat = [0 4; 4 8];
W = [0.1 0.2; 0.3 0.4];
dens = buildExpTens({pitchMat}, {W}, 10, 2, [], false, true, 1200, 'verbose', false);
results{end+1,1} = 'MAET: weight K x N full matrix';
results{end,2}   = isequal(dens.w{1}, W);

% -- Groups: vector form and cell form agree --

pitchMat = [0; 4];   % K=2, N=1
timeMat  = 0;         % 1 x 1
xMat     = 0; yMat = 0; zMat = 0;
sigV     = [10, 0.1, 0.2];  rV = [1 1 1 1 1];
isRelV   = [false false false];
isPerV   = [true false false];
perV     = [1200, 0, 0];

dens_v = buildExpTens({pitchMat, timeMat, xMat, yMat, zMat}, [], ...
    sigV, rV, [1 2 3 3 3], isRelV, isPerV, perV, 'verbose', false);
dens_c = buildExpTens({pitchMat, timeMat, xMat, yMat, zMat}, [], ...
    sigV, rV, {1, 2, [3 4 5]}, isRelV, isPerV, perV, 'verbose', false);

results{end+1,1} = 'MAET: groups vector vs cell (groupOfAttr)';
results{end,2}   = isequal(dens_v.groupOfAttr, dens_c.groupOfAttr);
results{end+1,1} = 'MAET: groups vector vs cell (nGroups)';
results{end,2}   = dens_v.nGroups == dens_c.nGroups;

agree = true;
for gg = 1:dens_v.nGroups
    if ~isequal(sort(dens_v.attrsOfGroup{gg}), sort(dens_c.attrsOfGroup{gg}))
        agree = false; break;
    end
end
results{end+1,1} = 'MAET: groups vector vs cell (attrsOfGroup)';
results{end,2}   = agree;

results{end+1,1} = 'MAET: groups non-contiguous errors';
results{end,2}   = throwsError(@() buildExpTens({pitchMat, pitchMat}, [], ...
    [10 10], [1 1], [1 3], [false false], [true true], [1200 1200], 'verbose', false));

results{end+1,1} = 'MAET: groups cell duplicate attr errors';
results{end,2}   = throwsError(@() buildExpTens({pitchMat, pitchMat}, [], ...
    10, [1 1], {[1 2], 2}, false, true, 1200, 'verbose', false));

% -- NaN-padded variable-size events --

pitchMat = [0 0; 4 4; 7 NaN];
timeMat  = [0 1];
dens = buildExpTens({pitchMat, timeMat}, [], ...
    [10, 0.1], [2, 1], [], [false false], [true false], [1200, 0], ...
    'lazy', false, 'verbose', false);
% Event 1: P(3,2)=6 perms, C(3,2)=3 combs. Event 2: P(2,2)=2, C(2,2)=1.
results{end+1,1} = 'MAET: NaN-padded nJ';
results{end,2}   = dens.nJ == 8;
results{end+1,1} = 'MAET: NaN-padded nK';
results{end,2}   = dens.nK == 4;

% -- Per-tuple weight factorisation --

pitch1 = [0; 4];
time1  = 1.5;
wPitch = [2.0; 3.0];
wTime  = 5.0;
dens = buildExpTens({pitch1, time1}, {wPitch, wTime}, ...
    [10, 0.1], [2, 1], [], [false false], [true false], [1200, 0], ...
    'lazy', false, 'verbose', false);
% 2 pitch perms, each with weight 2 * 3 * 5 = 30
results{end+1,1} = 'MAET: per-tuple weight factorisation (wJ)';
results{end,2}   = all(abs(dens.wJ - 30) < 1e-12);
results{end+1,1} = 'MAET: per-tuple weight factorisation (wv_comb)';
results{end,2}   = all(abs(dens.wv_comb - 30) < 1e-12);

% -- Error paths --

pitchMat = [0 4];
results{end+1,1} = 'MAET: insufficient slots errors';
results{end,2}   = throwsError(@() buildExpTens({[0 0; 4 NaN; NaN NaN]}, [], ...
    10, 2, [], false, true, 1200, 'verbose', false));

results{end+1,1} = 'MAET: wrong r length errors';
results{end,2}   = throwsError(@() buildExpTens({pitchMat, pitchMat}, [], ...
    [10 10], 1, [], [false false], [true true], [1200 1200], 'verbose', false));

results{end+1,1} = 'MAET: wrong sigma length errors';
results{end,2}   = throwsError(@() buildExpTens({pitchMat, pitchMat}, [], ...
    10, [1 1], [], [false false], [true true], [1200 1200], 'verbose', false));

results{end+1,1} = 'MAET: mismatched N errors';
results{end,2}   = throwsError(@() buildExpTens({[0 4], [0 1 2]}, [], ...
    [10 0.1], [1 1], [], [false false], [true false], [1200 0], 'verbose', false));

results{end+1,1} = 'MAET: wrong positional count errors';
results{end,2}   = throwsError(@() buildExpTens({pitchMat}, [], ...
    10, 1, false, true, 1200, 'verbose', false));

% -- isRel + r=1 degenerate warning --

lastwarn('');   % clear the warning buffer
buildExpTens({pitchMat}, [], 10, 1, [], true, true, 1200, 'verbose', false);
warnMsg = lastwarn;
results{end+1,1} = 'MAET: isRel + r=1 emits degenerate warning';
results{end,2}   = ~isempty(warnMsg) && contains(warnMsg, 'degenerate');

% -- evalExpTens MA path: SA-equivalence (isRel=false) --

p_sa_v  = [0; 400; 700];
w_sa_v  = [1; 0.7; 0.5];
sigma_v = 10; r_v = 2; isPer_v = true; period_v = 1200;
xSA_abs = [100 500; 300 600];   % dim=2, nQ=2 (absolute r=2)

dens_sa = buildExpTens(p_sa_v, w_sa_v, sigma_v, r_v, false, isPer_v, period_v, ...
    'verbose', false);
vals_sa = evalExpTens(dens_sa, xSA_abs, 'verbose', false);

dens_ma = buildExpTens({p_sa_v}, {w_sa_v}, sigma_v, r_v, [], false, isPer_v, ...
    period_v, 'verbose', false);
vals_ma_cell = evalExpTens(dens_ma, {xSA_abs}, 'verbose', false);
vals_ma_mat  = evalExpTens(dens_ma,  xSA_abs,  'verbose', false);

results{end+1,1} = 'evalExpTens MA: SA-equivalence abs (cell form)';
results{end,2}   = max(abs(vals_ma_cell - vals_sa)) < 1e-12;
results{end+1,1} = 'evalExpTens MA: SA-equivalence abs (matrix form)';
results{end,2}   = max(abs(vals_ma_mat - vals_sa)) < 1e-12;

% -- evalExpTens MA path: SA-equivalence (isRel=true, r=3) --

r_v = 3;
xSA_rel = [400 200; 700 500];    % dim = r-1 = 2, nQ = 2
dens_sa = buildExpTens(p_sa_v, w_sa_v, sigma_v, r_v, true, isPer_v, period_v, ...
    'verbose', false);
vals_sa = evalExpTens(dens_sa, xSA_rel, 'verbose', false);

dens_ma = buildExpTens({p_sa_v}, {w_sa_v}, sigma_v, r_v, [], true, isPer_v, ...
    period_v, 'verbose', false);
vals_ma = evalExpTens(dens_ma, {xSA_rel}, 'verbose', false);

results{end+1,1} = 'evalExpTens MA: SA-equivalence rel';
results{end,2}   = max(abs(vals_ma - vals_sa)) < 1e-12;

% Normalisation modes
for modeCell = {'gaussian', 'pdf'}
    mode = modeCell{1};
    vals_sa_n = evalExpTens(dens_sa, xSA_rel, mode, 'verbose', false);
    vals_ma_n = evalExpTens(dens_ma, {xSA_rel}, mode, 'verbose', false);
    results{end+1,1} = ['evalExpTens MA: SA-equivalence normalize=' mode]; %#ok<SAGROW>
    results{end,2}   = max(abs(vals_ma_n - vals_sa_n)) < 1e-12;
end

% -- evalExpTens MA: cell form vs matrix form agree --

pitchMat = [0; 4; 7];    % K=3, N=1
timeMat  = 1.0;           % 1 x 1
dens = buildExpTens({pitchMat, timeMat}, [], ...
    [10, 0.1], [2, 1], [], [false false], [true false], [1200, 0], ...
    'verbose', false);

x_pitch = [0 4; 4 7];    % 2 x 2
x_time  = [1 2];          % 1 x 2
vals_cell = evalExpTens(dens, {x_pitch, x_time}, 'verbose', false);
vals_mat  = evalExpTens(dens, [x_pitch; x_time], 'verbose', false);
results{end+1,1} = 'evalExpTens MA: cell form == matrix form';
results{end,2}   = isequal(vals_cell, vals_mat);

% -- evalExpTens MA: per-group isPer --

pitch1 = 0;   % K=1, N=1
time1  = 0;
dens = buildExpTens({pitch1, time1}, [], ...
    [20, 20], [1, 1], [], [false false], [true false], [1200, 0], ...
    'verbose', false);
% Pitch periodic: value at pitch=0 vs pitch=1200 should be equal
v_p0    = evalExpTens(dens, {0,    0}, 'verbose', false);
v_p1200 = evalExpTens(dens, {1200, 0}, 'verbose', false);
results{end+1,1} = 'evalExpTens MA: periodic pitch wraps';
results{end,2}   = abs(v_p0 - v_p1200) < 1e-12;
% Time nonperiodic: value at time=0 > time=1200
v_t0    = evalExpTens(dens, {0, 0},    'verbose', false);
v_t1200 = evalExpTens(dens, {0, 1200}, 'verbose', false);
results{end+1,1} = 'evalExpTens MA: nonperiodic time does not wrap';
results{end,2}   = v_t1200 < v_t0;

% -- evalExpTens MA: density positive at a tuple centre --

pitchMat = [0; 4; 7];
timeMat  = 1.0;
dens = buildExpTens({pitchMat, timeMat}, [], ...
    [10, 0.1], [2, 1], [], [false false], [true false], [1200, 0], ...
    'verbose', false);
v_centre = evalExpTens(dens, {[0; 4], 1.0}, 'verbose', false);
v_far    = evalExpTens(dens, {[600; 800], 50.0}, 'verbose', false);
results{end+1,1} = 'evalExpTens MA: density is positive at tuple centre';
results{end,2}   = v_centre > 0 && v_centre > v_far;

% -- evalExpTens MA: error paths --

results{end+1,1} = 'evalExpTens MA: wrong cell length errors';
results{end,2}   = throwsError(@() evalExpTens(dens, {[0; 4]}, 'verbose', false));

results{end+1,1} = 'evalExpTens MA: wrong per-attr rows errors';
results{end,2}   = throwsError(@() evalExpTens(dens, ...
    {zeros(3,1), zeros(1,1)}, 'verbose', false));

results{end+1,1} = 'evalExpTens MA: wrong total rows (matrix form) errors';
results{end,2}   = throwsError(@() evalExpTens(dens, zeros(5,1), 'verbose', false));

% -- cosSimExpTens MA path: SA-equivalence --

p_a_v  = [0; 400; 700];
p_b_v  = [0; 300; 700];
w_a_v  = [1; 0.7; 0.5];
w_b_v  = [1; 0.6; 0.8];

% Absolute (isRel=false), periodic
s_sa = cosSimExpTens(p_a_v, w_a_v, p_b_v, w_b_v, 10, 2, false, true, 1200, ...
    'verbose', false);
da = buildExpTens({p_a_v}, {w_a_v}, 10, 2, [], false, true, 1200, 'verbose', false);
db = buildExpTens({p_b_v}, {w_b_v}, 10, 2, [], false, true, 1200, 'verbose', false);
s_ma = cosSimExpTens(da, db, 'verbose', false);
results{end+1,1} = 'cosSimExpTens MA: SA-equivalence abs periodic';
results{end,2}   = abs(s_ma - s_sa) < 1e-12;

% Relative + periodic (uses pairwise-differences formula per attribute)
for r_v = [2, 3]
    s_sa = cosSimExpTens(p_a_v, w_a_v, p_b_v, w_b_v, 10, r_v, true, true, 1200, ...
        'verbose', false);
    da = buildExpTens({p_a_v}, {w_a_v}, 10, r_v, [], true, true, 1200, 'verbose', false);
    db = buildExpTens({p_b_v}, {w_b_v}, 10, r_v, [], true, true, 1200, 'verbose', false);
    s_ma = cosSimExpTens(da, db, 'verbose', false);
    results{end+1,1} = sprintf('cosSimExpTens MA: SA-equivalence rel periodic r=%d', r_v); %#ok<SAGROW>
    results{end,2}   = abs(s_ma - s_sa) < 1e-12;
end

% Relative + non-periodic
s_sa = cosSimExpTens(p_a_v, w_a_v, p_b_v, w_b_v, 10, 3, true, false, 0, ...
    'verbose', false);
da = buildExpTens({p_a_v}, {w_a_v}, 10, 3, [], true, false, 0, 'verbose', false);
db = buildExpTens({p_b_v}, {w_b_v}, 10, 3, [], true, false, 0, 'verbose', false);
s_ma = cosSimExpTens(da, db, 'verbose', false);
results{end+1,1} = 'cosSimExpTens MA: SA-equivalence rel non-periodic';
results{end,2}   = abs(s_ma - s_sa) < 1e-12;

% -- cosSimExpTens MA: self-similarity = 1 --

pitchMA = [0 12; 4 15; 7 19];    % 3 x 2
timeMA  = [0 1];                  % 1 x 2
d = buildExpTens({pitchMA, timeMA}, [], ...
    [10, 0.1], [3, 1], [], [true, false], [true, false], [1200, 0], ...
    'verbose', false);
s_self = cosSimExpTens(d, d, 'verbose', false);
results{end+1,1} = 'cosSimExpTens MA: self-similarity = 1';
results{end,2}   = abs(s_self - 1) < 1e-12;

% -- cosSimExpTens MA: symmetry --

pitchA = [0 12; 4 15; 7 19];
timeA  = [0 1];
pitchB = [0 10; 4 13; 7 17];
timeB  = [0 1.2];
da = buildExpTens({pitchA, timeA}, [], ...
    [10, 0.1], [3, 1], [], [true, false], [true, false], [1200, 0], ...
    'verbose', false);
db = buildExpTens({pitchB, timeB}, [], ...
    [10, 0.1], [3, 1], [], [true, false], [true, false], [1200, 0], ...
    'verbose', false);
s_ab = cosSimExpTens(da, db, 'verbose', false);
s_ba = cosSimExpTens(db, da, 'verbose', false);
results{end+1,1} = 'cosSimExpTens MA: symmetry (a,b) == (b,a)';
results{end,2}   = abs(s_ab - s_ba) < 1e-12;

% -- cosSimExpTens MA: isRel transposition invariance --

pitchT  = [0; 400; 700];
pitchTs = pitchT + 137;
timeT   = 1;
d1 = buildExpTens({pitchT,  timeT}, [], ...
    [10, 0.1], [3, 1], [], [true, false], [true, false], [1200, 0], ...
    'verbose', false);
d2 = buildExpTens({pitchTs, timeT}, [], ...
    [10, 0.1], [3, 1], [], [true, false], [true, false], [1200, 0], ...
    'verbose', false);
s_trans = cosSimExpTens(d1, d2, 'verbose', false);
results{end+1,1} = 'cosSimExpTens MA: isRel transposition invariance';
results{end,2}   = abs(s_trans - 1) < 1e-10;

% -- cosSimExpTens MA: raw-args matches struct form --

s_raw = cosSimExpTens({pitchA, timeA}, [], {pitchB, timeB}, [], ...
    [10, 0.1], [3, 1], [], [true, false], [true, false], [1200, 0], ...
    'verbose', false);
results{end+1,1} = 'cosSimExpTens MA: raw-args == struct form';
results{end,2}   = abs(s_raw - s_ab) < 1e-12;

% -- cosSimExpTens: SA raw-args still works (backward compat check) --

s_sa_raw = cosSimExpTens([0 4 7], [], [0 4 7], [], 10, 2, true, true, 1200, ...
    'verbose', false);
results{end+1,1} = 'cosSimExpTens: SA raw-args identical = 1';
results{end,2}   = abs(s_sa_raw - 1) < 1e-12;

% -- cosSimExpTens MA: mismatched raw-args kinds error --

results{end+1,1} = 'cosSimExpTens MA: mismatched raw-args kinds error';
results{end,2}   = throwsError(@() cosSimExpTens( ...
    {pitchA, timeA}, [], [0 4 7], [], 10, 2, true, true, 1200, 'verbose', false));

% -- cosSimExpTens MA: mixed struct types error --

d_sa = buildExpTens([0 4 7], [], 10, 2, false, true, 1200, 'verbose', false);
d_ma = buildExpTens({[0; 4; 7]}, [], 10, 2, [], false, true, 1200, 'verbose', false);
results{end+1,1} = 'cosSimExpTens: mixed SA/MA structs error';
results{end,2}   = throwsError(@() cosSimExpTens(d_sa, d_ma, 'verbose', false));

% -- cosSimExpTens MA: parameter-mismatch errors --

d_ref = buildExpTens({pitchA}, [], 10, 2, [], false, true, 1200, 'verbose', false);
% different r
d_r = buildExpTens({pitchA}, [], 10, 3, [], false, true, 1200, 'verbose', false);
results{end+1,1} = 'cosSimExpTens MA: mismatched r error';
results{end,2}   = throwsError(@() cosSimExpTens(d_ref, d_r, 'verbose', false));
% different sigma
d_s = buildExpTens({pitchA}, [], 20, 2, [], false, true, 1200, 'verbose', false);
results{end+1,1} = 'cosSimExpTens MA: mismatched sigma error';
results{end,2}   = throwsError(@() cosSimExpTens(d_ref, d_s, 'verbose', false));
% different isRel
d_rel = buildExpTens({pitchA}, [], 10, 2, [], true, true, 1200, 'verbose', false);
results{end+1,1} = 'cosSimExpTens MA: mismatched isRel error';
results{end,2}   = throwsError(@() cosSimExpTens(d_ref, d_rel, 'verbose', false));
% different period on periodic group
d_p = buildExpTens({pitchA}, [], 10, 2, [], false, true, 2400, 'verbose', false);
results{end+1,1} = 'cosSimExpTens MA: mismatched period error';
results{end,2}   = throwsError(@() cosSimExpTens(d_ref, d_p, 'verbose', false));

% -- entropyExpTens MA: SA-equivalence periodic --

p_e = [0; 4; 7];
w_e = [1; 1; 1];
H_sa = entropyExpTens(p_e.', w_e.', 10, 1, false, true, 12, ...
    'nPointsPerDim', 400, 'verbose', false);
H_ma = entropyExpTens({p_e}, {w_e}, 10, 1, [], false, true, 12, ...
    'nPointsPerDim', 400, 'verbose', false);
results{end+1,1} = 'entropyExpTens MA: SA-equivalence periodic';
results{end,2}   = abs(H_ma - H_sa) < 1e-10;

% -- entropyExpTens MA: SA-equivalence non-periodic --

H_sa = entropyExpTens(p_e.', w_e.', 10, 1, false, false, 0, ...
    'xMin', -3, 'xMax', 10, 'nPointsPerDim', 400, 'verbose', false);
H_ma = entropyExpTens({p_e}, {w_e}, 10, 1, [], false, false, 0, ...
    'xMin', -3, 'xMax', 10, 'nPointsPerDim', 400, 'verbose', false);
results{end+1,1} = 'entropyExpTens MA: SA-equivalence non-periodic';
results{end,2}   = abs(H_ma - H_sa) < 1e-10;

% -- entropyExpTens MA: uniform pitch near 1 --

p_uniform = (0:11).';
H_u = entropyExpTens({p_uniform}, [], 100, 1, [], false, true, 12, ...
    'nPointsPerDim', 400, 'verbose', false);
results{end+1,1} = 'entropyExpTens MA: uniform chromatic near 1';
results{end,2}   = H_u > 0.95;

% -- entropyExpTens MA: concentrated below uniform --

H_one = entropyExpTens({5}, [], 20, 1, [], false, true, 12, ...
    'nPointsPerDim', 400, 'verbose', false);
H_all = entropyExpTens({p_uniform}, [], 20, 1, [], false, true, 12, ...
    'nPointsPerDim', 400, 'verbose', false);
results{end+1,1} = 'entropyExpTens MA: concentrated < uniform';
results{end,2}   = H_one < H_all;

% -- entropyExpTens MA: pitch + time runs (dim = 2) --

pitchE = [0 12; 4 15; 7 19];   % 3 x 2
timeE  = [0 1];                 % 1 x 2
densE = buildExpTens({pitchE, timeE}, [], ...
    [20, 0.1], [2, 1], [], [true, false], [true, false], [1200, 0], ...
    'verbose', false);
results{end+1,1} = 'entropyExpTens MA: dim == 2 (r=2 pitch + r=1 time)';
results{end,2}   = densE.dim == 2;
H_pt = entropyExpTens(densE, ...
    'xMin', -0.5, 'xMax', 1.5, 'nPointsPerDim', 80, 'verbose', false);
results{end+1,1} = 'entropyExpTens MA: pitch+time H in (0,1)';
results{end,2}   = H_pt > 0 && H_pt < 1;

% -- entropyExpTens MA: grid-limit guard --

results{end+1,1} = 'entropyExpTens MA: grid-limit exceeded errors';
results{end,2}   = throwsError(@() entropyExpTens(densE, ...
    'xMin', 0, 'xMax', 2, 'nPointsPerDim', 20000, 'gridLimit', 1e6, 'verbose', false));

% -- entropyExpTens MA: missing bounds error --

results{end+1,1} = 'entropyExpTens MA: missing non-periodic bounds errors';
results{end,2}   = throwsError(@() entropyExpTens({p_e}, [], 10, 1, [], ...
    false, false, 0, 'nPointsPerDim', 100, 'verbose', false));

% -- entropyExpTens MA: per-group bounds vector matches scalar --

H_scalar = entropyExpTens(densE, ...
    'xMin', -0.5, 'xMax', 1.5, 'nPointsPerDim', 60, 'verbose', false);
H_vec = entropyExpTens(densE, ...
    'xMin', [NaN, -0.5], 'xMax', [NaN, 1.5], 'nPointsPerDim', 60, 'verbose', false);
results{end+1,1} = 'entropyExpTens MA: per-group bounds vector == scalar';
results{end,2}   = abs(H_scalar - H_vec) < 1e-12;

% -- differenceEvents: order 0 identity --

p_d = {[0 2 5 7]};
[pd, wd] = differenceEvents(p_d, [], [], 0, 12);
results{end+1,1} = 'differenceEvents: order 0 values unchanged';
results{end,2}   = isequal(pd{1}, p_d{1});
results{end+1,1} = 'differenceEvents: order 0 weight stays []';
results{end,2}   = isempty(wd);

% -- differenceEvents: order 1 non-periodic --

[pd, ~] = differenceEvents({[0 2 5 7]}, [], [], 1, 0);
results{end+1,1} = 'differenceEvents: order 1 non-periodic';
results{end,2}   = isequal(pd{1}, [2 3 2]);

% -- differenceEvents: order 1 periodic wrap --

[pd, ~] = differenceEvents({[0 11]}, [], [], 1, 12);
results{end+1,1} = 'differenceEvents: order 1 periodic wrap (11 -> -1)';
results{end,2}   = isequal(pd{1}, -1);

% -- differenceEvents: order 2 --

[pd, ~] = differenceEvents({[0 2 5 7]}, [], [], 2, 0);
results{end+1,1} = 'differenceEvents: order 2';
results{end,2}   = isequal(pd{1}, [1 -1]);

% -- differenceEvents: order 1 weight rolling product --

[~, wd] = differenceEvents({[0 2 5 7]}, {[0.5 0.8 1.0 0.2]}, [], 1, 0);
expected = [0.5*0.8, 0.8*1.0, 1.0*0.2];
results{end+1,1} = 'differenceEvents: order 1 rolling-product weights';
results{end,2}   = max(abs(wd{1} - expected)) < 1e-12;

% -- differenceEvents: order 2 weight rolling product (width 3) --

[~, wd] = differenceEvents({[0 1 3 6]}, {[0.5 0.8 1.0 0.2]}, [], 2, 0);
expected = [0.5*0.8*1.0, 0.8*1.0*0.2];
results{end+1,1} = 'differenceEvents: order 2 rolling-product weights';
results{end,2}   = max(abs(wd{1} - expected)) < 1e-12;

% -- differenceEvents: scalar top-level weight raised to power --
% A top-level scalar c with uniform order k returns scalar c^(k+1),
% so scalar and vector-of-c inputs produce equivalent downstream
% densities.

[~, wd] = differenceEvents({[0 2 5]}, 0.7, [], 1, 0);
results{end+1,1} = 'differenceEvents: scalar weight raised to power';
results{end,2}   = abs(wd - 0.7^2) < 1e-12;

% -- differenceEvents: mixed orders alignment --

p_d = {[0 2 5 7], [0 1 2 3.5]};
[pd, ~] = differenceEvents(p_d, [], [], [0 1], [0 0]);
results{end+1,1} = 'differenceEvents: mixed orders — k=0 group drops leading';
results{end,2}   = isequal(pd{1}, [2 5 7]);
results{end+1,1} = 'differenceEvents: mixed orders — k=1 group differenced';
results{end,2}   = isequal(pd{2}, [1 1 1.5]);

% -- differenceEvents: grouped attributes share order --

p_d = {[0 1 3], [10 12 16]};
[pd, ~] = differenceEvents(p_d, [], [1 1], 1, 0);
results{end+1,1} = 'differenceEvents: grouped attrs both differenced';
results{end,2}   = isequal(pd{1}, [1 2]) && isequal(pd{2}, [2 4]);

% -- differenceEvents: output feeds buildExpTens --

p_d = {[0 2 5 7], [0 0.5 1.2 1.7]};
[pd, wd] = differenceEvents(p_d, [], [], [0 1], [1200 0]);
dens_d = buildExpTens(pd, wd, [10 0.05], [1 1], [], ...
    [false false], [true false], [1200 0], 'verbose', false);
results{end+1,1} = 'differenceEvents: output feeds buildExpTens';
results{end,2}   = strcmp(dens_d.tag, 'MaetDensity') && dens_d.N == 3;

% -- differenceEvents: too-high order errors --

results{end+1,1} = 'differenceEvents: too-high order errors';
results{end,2}   = throwsError(@() differenceEvents({[0 2 5]}, [], [], 3, 0));

% -- differenceEvents: negative order errors --

results{end+1,1} = 'differenceEvents: negative order errors';
results{end,2}   = throwsError(@() differenceEvents({[0 2 5]}, [], [], -1, 0));

% -- differenceEvents: diffOrders length mismatch errors --

results{end+1,1} = 'differenceEvents: diffOrders length mismatch errors';
results{end,2}   = throwsError(@() differenceEvents({[0 2 5]}, [], [], [1 1], [0 0]));

% -- differenceEvents: mismatched event counts error --

results{end+1,1} = 'differenceEvents: mismatched event counts error';
results{end,2}   = throwsError(@() differenceEvents({[0 2 5], [0 1]}, [], [], [0 0], [0 0]));

% -- differenceEvents: multi-slot attribute errors --
% A single K_a = 2 attribute must raise differenceEvents:multiSlotAttribute.
% Column-wise differencing would impose a cross-event slot correspondence
% that within-event slot exchangeability does not license.

p_ms = {[60 62 64; 67 69 71]};   % K_a = 2, N = 3
results{end+1,1} = 'differenceEvents: K_a = 2 attribute errors (multiSlotAttribute id)';
results{end,2}   = throwsErrorWithId( ...
    @() differenceEvents(p_ms, [], [], 1, 0), ...
    'differenceEvents:multiSlotAttribute');

% -- differenceEvents: empty attribute (K_a = 0) errors --
% K_a = 0 is likewise rejected by the K_a = 1 check; there is nothing
% to difference in an empty attribute.

p_empty = {zeros(0, 3)};          % K_a = 0, N = 3
results{end+1,1} = 'differenceEvents: K_a = 0 attribute errors (multiSlotAttribute id)';
results{end,2}   = throwsErrorWithId( ...
    @() differenceEvents(p_empty, [], [], 1, 0), ...
    'differenceEvents:multiSlotAttribute');

% -- differenceEvents: mixed K_a input errors on the offending attribute --
% Attribute 1 has K_a = 1, attribute 2 has K_a = 2 — the error must fire
% and its message must name the offending attribute index.

p_mixed = {[60 62 64], [60 62 64; 67 69 71]};
results{end+1,1} = 'differenceEvents: mixed K_a input errors (multiSlotAttribute id)';
results{end,2}   = throwsErrorWithId( ...
    @() differenceEvents(p_mixed, [], [], [1 1], [0 0]), ...
    'differenceEvents:multiSlotAttribute');

results{end+1,1} = 'differenceEvents: mixed K_a error message names attribute 2';
results{end,2}   = errorMessageContains( ...
    @() differenceEvents(p_mixed, [], [], [1 1], [0 0]), ...
    'Attribute 2');

% -- differenceEvents: voices-as-attributes pipeline round trip --
% Four voices, each K_a = 1 in a shared group, differenced, then stacked
% into a single multi-slot attribute before buildExpTens. The resulting
% MaetDensity should have the expected shape and evalExpTens should
% return finite non-negative values at a few query points.

pS = [72 74 76 77];     % soprano
pA = [67 69 71 72];     % alto
pT = [60 62 64 65];     % tenor
pB = [48 50 52 53];     % bass
pAttr  = {pS, pA, pT, pB};
groupsV = [1 1 1 1];
[pDiff, ~] = differenceEvents(pAttr, [], groupsV, 1, 0);

results{end+1,1} = 'differenceEvents: voices-as-attrs — each differenced attribute is 1 x 3';
results{end,2}   = all(cellfun(@(M) isequal(size(M), [1 3]), pDiff));

pBundled = { vertcat(pDiff{:}) };    % 4 x 3 multi-slot bundle
results{end+1,1} = 'differenceEvents: voices-as-attrs — bundle is 4 x 3';
results{end,2}   = isequal(size(pBundled{1}), [4 3]);

dens_v = buildExpTens(pBundled, [], 10, 1, [], false, false, 0, ...
    'verbose', false);
results{end+1,1} = 'differenceEvents: voices-as-attrs — buildExpTens returns MaetDensity';
results{end,2}   = strcmp(dens_v.tag, 'MaetDensity');

% Evaluate at a handful of query points; expect finite non-negative
% output everywhere.
x_query = [-3 0 2 4 7];
vals_v = evalExpTens(dens_v, x_query);
results{end+1,1} = 'differenceEvents: voices-as-attrs — evalExpTens returns finite non-negative values';
results{end,2}   = all(isfinite(vals_v)) && all(vals_v >= 0);

% -- windowTensor: basic construction --

pitch_w = [60 62 64 65];    % 1 x 4 events
time_w  = [0  1  2  3];
dens_w = buildExpTens({pitch_w, time_w}, [], ...
    [10 0.1], [1 1], [], ...
    [false false], [true false], [1200 0], ...
    'lazy', false, 'verbose', false);

spec_w = struct();
spec_w.size = [Inf, 1];
spec_w.mix  = [0, 0];
spec_w.centre = {zeros(1, 1), 0.5};
wmd_w = windowTensor(dens_w, spec_w);
results{end+1,1} = 'windowTensor: returns tagged WindowedMaetDensity';
results{end,2}   = strcmp(wmd_w.tag, 'WindowedMaetDensity');

% -- windowTensor: wide window, centred at context mean, gives
%    cos_sim ~= 1 --
% Under v2.1.0 cross-correlation semantics the query is translated so
% that its effective-space mean moves onto the window centre, so a
% centred window at the context's own mean is the correct analogue of
% the no-window case.

t_mean_w = mean(time_w);
spec_wide = struct('size', [Inf, 1e6], 'mix', [0, 0], ...
                   'centre', {{zeros(1, 1), t_mean_w}});
wmd_wide = windowTensor(dens_w, spec_wide);
s_wide = cosSimExpTens(dens_w, wmd_wide, 'verbose', false);
s_self = cosSimExpTens(dens_w, dens_w, 'verbose', false);
results{end+1,1} = 'windowTensor: wide centred window == unwindowed self-sim';
results{end,2}   = abs(s_wide - s_self) < 1e-3;

% -- windowTensor: infinite size on all groups == identity --

spec_inf = struct('size', [Inf, Inf], 'mix', [0, 0]);
wmd_inf = windowTensor(dens_w, spec_inf);
s_inf = cosSimExpTens(dens_w, wmd_inf, 'verbose', false);
results{end+1,1} = 'windowTensor: all-Inf size == identity (s ~= 1)';
results{end,2}   = abs(s_inf - 1) < 1e-6;

% -- windowTensor: narrow window reduces cos_sim --

spec_narrow = struct('size', [Inf, 0.2], 'mix', [0, 0], ...
                     'centre', {{zeros(1, 1), 0}});
wmd_narrow = windowTensor(dens_w, spec_narrow);
s_narrow = cosSimExpTens(dens_w, wmd_narrow, 'verbose', false);
results{end+1,1} = 'windowTensor: narrow window reduces cos_sim';
results{end,2}   = s_narrow < 0.5;

% -- windowTensor: rectangular window on 1-D time works --

spec_rect = struct('size', [Inf, 0.5], 'mix', [0, 1], ...
                   'centre', {{zeros(1, 1), 1.0}});
wmd_rect = windowTensor(dens_w, spec_rect);
s_rect = cosSimExpTens(dens_w, wmd_rect, 'verbose', false);
results{end+1,1} = 'windowTensor: rectangular 1-D time works (finite, 0<s<1)';
results{end,2}   = isfinite(s_rect) && s_rect > 0 && s_rect < 1;

% -- windowTensor: raised-rectangular window on 1-D time works --

spec_raised = struct('size', [Inf, 0.5], 'mix', [0, 0.5], ...
                     'centre', {{zeros(1, 1), 1.0}});
wmd_raised = windowTensor(dens_w, spec_raised);
s_raised = cosSimExpTens(dens_w, wmd_raised, 'verbose', false);
results{end+1,1} = 'windowTensor: raised-rectangular 1-D time works';
results{end,2}   = isfinite(s_raised) && s_raised > 0 && s_raised < 1;

% -- windowTensor: multi-D relative Gaussian works --

pitchMR = [60 62; 64 65; 67 69];   % 3 slots, 2 events
dens_mr = buildExpTens({pitchMR}, [], 10, 3, [], ...
    true, true, 1200, 'verbose', false);
spec_mr_gauss = struct('size', 1, 'mix', 0, ...
                       'centre', {{[50; 100]}});
wmd_mr = windowTensor(dens_mr, spec_mr_gauss);
s_mr = cosSimExpTens(dens_mr, wmd_mr, 'verbose', false);
results{end+1,1} = 'windowTensor: multi-D rel Gaussian window works';
results{end,2}   = isfinite(s_mr) && s_mr >= 0 && s_mr <= 1;

% -- windowTensor: multi-D relative rectangular raises --

spec_mr_rect = struct('size', 1, 'mix', 1, 'centre', {{[50; 100]}});
wmd_mr_rect = windowTensor(dens_mr, spec_mr_rect);
results{end+1,1} = 'windowTensor: multi-D rel rectangular errors';
results{end,2}   = throwsError(@() cosSimExpTens(dens_mr, wmd_mr_rect, 'verbose', false));

% -- windowTensor: multi-D relative raised-rect raises --

spec_mr_rr = struct('size', 1, 'mix', 0.5, 'centre', {{[50; 100]}});
wmd_mr_rr = windowTensor(dens_mr, spec_mr_rr);
results{end+1,1} = 'windowTensor: multi-D rel raised-rect errors';
results{end,2}   = throwsError(@() cosSimExpTens(dens_mr, wmd_mr_rr, 'verbose', false));

% -- windowTensor: entropy on rect-windowed multi-D rel works --

H_mr_rect = entropyExpTens(wmd_mr_rect, 'nPointsPerDim', 20, 'verbose', false);
results{end+1,1} = 'windowTensor: entropy on rect-windowed multi-D rel runs';
results{end,2}   = isfinite(H_mr_rect);

% -- windowTensor: narrower window yields lower entropy --

H_base = entropyExpTens(dens_w, 'xMin', [0, -1], 'xMax', [1200, 4], ...
                         'nPointsPerDim', 60, 'verbose', false);
spec_ew = struct('size', [Inf, 0.3], 'mix', [0, 0], ...
                 'centre', {{zeros(1, 1), 1.0}});
wmd_ew = windowTensor(dens_w, spec_ew);
H_narrow = entropyExpTens(wmd_ew, 'xMin', [0, -1], 'xMax', [1200, 4], ...
                           'nPointsPerDim', 60, 'verbose', false);
results{end+1,1} = 'windowTensor: narrower window => lower entropy';
results{end,2}   = H_narrow < H_base;

% -- windowedSimilarity: profile peaks at matching event offset --

pitch_narrow = [60 62 64 65];
time_narrow  = [0  1  2  3];
ctx_narrow = buildExpTens({pitch_narrow, time_narrow}, [], ...
    [0.5 0.1], [1 1], [], [false false], [true false], [1200 0], ...
    'verbose', false);

% Fixed single-event query at pitch 62, time 0 (centroid at t=0).
q_sw = buildExpTens({62, 0}, [], ...
    [0.5 0.1], [1 1], [], [false false], [true false], [1200 0], ...
    'verbose', false);

M_sweep = 21;
offs_sw = linspace(-0.5, 3.5, M_sweep);
offsets_sw = zeros(2, M_sweep);
offsets_sw(2, :) = offs_sw;
spec_sw = struct('size', [Inf, 0.3], 'mix', [0, 0]);
profile = windowedSimilarity(q_sw, ctx_narrow, spec_sw, offsets_sw, ...
    'verbose', false);
[~, peak_idx] = max(profile);
peak_off = offs_sw(peak_idx);
% Query centroid is at t=0, so offset 1 corresponds to the pitch-62
% context event at absolute t=1.
results{end+1,1} = 'windowedSimilarity: profile peaks at matching event offset';
results{end,2}   = abs(peak_off - 1.0) < 0.3;

% -- windowedSimilarity: returns length-M profile --

offsets_vec = zeros(2, 7);
offsets_vec(2, :) = linspace(0, 1, 7);
spec_lm = struct('size', [Inf, 0.5], 'mix', [0, 0]);
prof_lm = windowedSimilarity(dens_w, dens_w, spec_lm, offsets_vec, 'verbose', false);
results{end+1,1} = 'windowedSimilarity: output is 1 x M';
results{end,2}   = isequal(size(prof_lm), [1, 7]);

% -- windowedSimilarity: reference=[] (default) matches omitted reference --
%
% Explicit empty reference must reproduce the default path byte-for-byte.
q_ref     = dens_w;
ctx_ref   = dens_w;
offs_ref  = zeros(2, 11);
offs_ref(2, :) = linspace(-0.5, 1.5, 11);
spec_ref  = struct('size', [Inf, 0.3], 'mix', [0, 0]);
prof_default  = windowedSimilarity(q_ref, ctx_ref, spec_ref, offs_ref, ...
                               'verbose', false);
prof_explicit = windowedSimilarity(q_ref, ctx_ref, spec_ref, offs_ref, ...
                               'reference', [], 'verbose', false);
results{end+1,1} = 'windowedSimilarity: reference=[] == default';
results{end,2}   = max(abs(prof_default - prof_explicit)) < 1e-12;

% -- windowedSimilarity: supplied reference shifts the profile --
%
% Set the time-attribute reference to (default + 0.2 s); the resulting
% profile at offset o must equal the default profile at offset o + 0.2
% (for offsets where both fall on the sweep grid).
muA_pitch = mean(q_ref.Centres{1}, 2);
muA_time  = mean(q_ref.Centres{2}, 2);
ref_shift = { muA_pitch, muA_time + 0.2 };
M_sh      = 21;
offs_sh   = zeros(2, M_sh);
off_t_sh  = linspace(-1.0, 3.0, M_sh);
offs_sh(2, :) = off_t_sh;
prof_d  = windowedSimilarity(q_ref, ctx_ref, spec_ref, offs_sh, ...
                         'verbose', false);
prof_sh = windowedSimilarity(q_ref, ctx_ref, spec_ref, offs_sh, ...
                         'reference', ref_shift, 'verbose', false);
% Check: prof_sh(m) should equal prof_d at offset off_t_sh(m) + 0.2
ok_shift = true;
for m_idx = 1:M_sh
    target = off_t_sh(m_idx) + 0.2;
    [dmin, jj] = min(abs(off_t_sh - target));
    if dmin < 1e-9
        if abs(prof_sh(m_idx) - prof_d(jj)) > 1e-10
            ok_shift = false;
            break;
        end
    end
end
results{end+1,1} = 'windowedSimilarity: reference shifts profile by offset';
results{end,2}   = ok_shift;

% -- windowedSimilarity: bad reference shape errors --
results{end+1,1} = 'windowedSimilarity: reference wrong cell count errors';
results{end,2}   = throwsError(@() windowedSimilarity(q_ref, ctx_ref, ...
    spec_ref, offs_ref, 'reference', {muA_pitch}, 'verbose', false));

results{end+1,1} = 'windowedSimilarity: reference wrong length errors';
results{end,2}   = throwsError(@() windowedSimilarity(q_ref, ctx_ref, ...
    spec_ref, offs_ref, 'reference', {muA_pitch, [0; 0]}, 'verbose', false));

% -- windowedSimilarity: periodic-window warning --
%
% The line-case closed form used downstream is exact for non-periodic
% groups and only approximate for periodic groups (it retains only
% the leading periodic image of the window). A single warning,
% windowedSimilarity:periodicWindowApprox, is emitted on every call
% involving a windowed periodic group, with two message forms:
%
%   - Within the recommended bound (lambda*sigma <= P/(2*sqrt(3))):
%     a brief informational form.
%   - Past the bound: a stronger form with phi reported and per-mix
%     behaviour described.
%
% dens_w has pitch sigma = 10 cents and period = 1200 cents on a
% periodic group, so the bound lambda*sigma > P/(2*sqrt(3)) ~=
% 346.4 cents corresponds to size > 34.64. See manuscript §5.2
% Remark 5.2 and User Guide §3.1 "Post-tensor windowing".

offs_off = [zeros(1, 5); linspace(0, 1, 5)];

% (a) The warning fires on every call involving a windowed periodic
% group, regardless of window size. Tested with a tiny window
% (size = 5) well within the bound.
spec_small = struct('size', [5, 0.3], 'mix', [0, 0]);
W = warning('error', 'windowedSimilarity:periodicWindowApprox');
fired_small = false;
try
    windowedSimilarity(dens_w, dens_w, spec_small, offs_off, 'verbose', false);
catch ME
    fired_small = strcmp(ME.identifier, ...
        'windowedSimilarity:periodicWindowApprox');
end
warning(W);
results{end+1,1} = 'windowedSimilarity: periodic warning fires on every call';
results{end,2}   = fired_small;

% (b) Within the bound, the message takes the brief informational
% form. Detected by absence of the past-bound marker phrase.
spec_small = struct('size', [5, 0.3], 'mix', [0, 0]);
W = warning('off', 'windowedSimilarity:periodicWindowApprox');
% lastwarn does not capture warnings that have been turned off, so
% set to 'on' for capture but route through evalc to suppress the
% on-screen print.
warning('on', 'windowedSimilarity:periodicWindowApprox');
lastwarn('');
evalc(['windowedSimilarity(dens_w, dens_w, spec_small, offs_off, ' ...
       '''verbose'', false);']);
[msg_within, id_within] = lastwarn;
warning(W);
results{end+1,1} = 'windowedSimilarity: within-bound message is the informational form';
results{end,2}   = strcmp(id_within, ...
    'windowedSimilarity:periodicWindowApprox') && ...
    isempty(strfind(msg_within, 'exceeds the recommended bound')) && ...
    ~isempty(strfind(msg_within, 'approximation is sub-percent'));

% (c) Past the bound, the message takes the stronger form with phi
% and per-mix behaviour. size = 40 -> lambda*sigma = 400 > 346.4.
spec_off = struct('size', [40, 0.3], 'mix', [0, 0]);
W = warning('on', 'windowedSimilarity:periodicWindowApprox');
lastwarn('');
evalc(['windowedSimilarity(dens_w, dens_w, spec_off, offs_off, ' ...
       '''verbose'', false);']);
[msg_past, id_past] = lastwarn;
warning(W);
results{end+1,1} = 'windowedSimilarity: past-bound message is the stronger form';
results{end,2}   = strcmp(id_past, ...
    'windowedSimilarity:periodicWindowApprox') && ...
    ~isempty(strfind(msg_past, 'exceeds the recommended bound')) && ...
    ~isempty(strfind(msg_past, 'phi (rect half-width)'));

% (d) Aperiodic case: a non-periodic group never triggers the
% warning, even under a very wide window. The time group has
% isPer=false in dens_w, so windowing only the time group with
% size = 1e6 must not warn.
spec_time_only = struct('size', [Inf, 1e6], 'mix', [0, 0]);
W = warning('error', 'windowedSimilarity:periodicWindowApprox');
silent_aper = true;
try
    windowedSimilarity(dens_w, dens_w, spec_time_only, offs_off, 'verbose', false);
catch ME
    silent_aper = ~strcmp(ME.identifier, ...
        'windowedSimilarity:periodicWindowApprox');
end
warning(W);
results{end+1,1} = 'windowedSimilarity: periodic warning silent on aperiodic group';
results{end,2}   = silent_aper;

% (e) The warning fires on every offending call (matching MATLAB's
% default warning behaviour, which we rely on here).
W = warning('error', 'windowedSimilarity:periodicWindowApprox');
n_fired = 0;
for k_call = 1:2
    try
        windowedSimilarity(dens_w, dens_w, spec_off, offs_off, 'verbose', false);
    catch ME
        if strcmp(ME.identifier, 'windowedSimilarity:periodicWindowApprox')
            n_fired = n_fired + 1;
        end
    end
end
warning(W);
results{end+1,1} = 'windowedSimilarity: periodic warning fires every call';
results{end,2}   = (n_fired >= 2);

% -- windowTensor: shape-validation errors --

results{end+1,1} = 'windowTensor: bad size length errors';
results{end,2}   = throwsError(@() windowTensor(dens_w, ...
    struct('size', [1 1 1], 'mix', [0 0])));

results{end+1,1} = 'windowTensor: mix out of range errors';
results{end,2}   = throwsError(@() windowTensor(dens_w, ...
    struct('size', [1 1], 'mix', [0 1.5])));

results{end+1,1} = 'windowTensor: wrong centre length errors';
bad_spec = struct('size', [1 1], 'mix', [0 0]);
bad_spec.centre = {zeros(1,1)};   % length 1 cell, need A = 2
results{end,2}   = throwsError(@() windowTensor(dens_w, bad_spec));

% -- windowTensor: scalar centre broadcasting --
% A size-1 centre input (numeric scalar, 1x1 array, or single-element
% cell containing a scalar) broadcasts to fill every per-attribute
% slot uniformly. Mirrors the Python window_tensor behaviour.

% Build a small MA density: A=2, both attributes r=1, separate groups.
% dim_per_attr = [1, 1], dim_total = 2.
% For these tests we use the existing dens_w (pitch r=1 + time r=1).
ref_centre_cell = {3.0, 3.0};   % equivalent uniform centre per attribute
spec_ref = struct('size', [1, 0.5], 'mix', [0, 0], 'centre', {ref_centre_cell});
wmd_ref = windowTensor(dens_w, spec_ref);
cos_ref = cosSimExpTens(dens_w, wmd_ref, 'verbose', false);

% Numeric scalar.
spec_scalar = struct('size', [1, 0.5], 'mix', [0, 0], 'centre', 3.0);
wmd_s = windowTensor(dens_w, spec_scalar);
cos_s  = cosSimExpTens(dens_w, wmd_s, 'verbose', false);
results{end+1,1} = 'windowTensor: numeric scalar centre broadcasts';
results{end,2}   = isequal(wmd_s.centre, wmd_ref.centre) && cos_s == cos_ref;

% 1x1 array.
spec_1x1 = struct('size', [1, 0.5], 'mix', [0, 0], 'centre', [3.0]);
wmd_1x1 = windowTensor(dens_w, spec_1x1);
cos_1x1 = cosSimExpTens(dens_w, wmd_1x1, 'verbose', false);
results{end+1,1} = 'windowTensor: 1x1 array centre broadcasts';
results{end,2}   = isequal(wmd_1x1.centre, wmd_ref.centre) && cos_1x1 == cos_ref;

% Length-1 cell does NOT broadcast — preserves the pre-fix contract
% that a wrong-length cell raises. This complements the existing test
% 'windowTensor: wrong centre length errors' above.
results{end+1,1} = 'windowTensor: length-1 cell on A>1 still errors (no broadcast)';
results{end,2}   = throwsError(@() windowTensor(dens_w, ...
    struct('size', [1, 0.5], 'mix', [0, 0], 'centre', {{3.0}})));

% Zero scalar.
spec_zero = struct('size', [1, 0.5], 'mix', [0, 0], 'centre', 0);
wmd_z = windowTensor(dens_w, spec_zero);
results{end+1,1} = 'windowTensor: zero scalar centre broadcasts';
results{end,2}   = isequal(wmd_z.centre{1}, 0) && isequal(wmd_z.centre{2}, 0);

% Negative scalar.
spec_neg = struct('size', [1, 0.5], 'mix', [0, 0], 'centre', -7.5);
wmd_n = windowTensor(dens_w, spec_neg);
results{end+1,1} = 'windowTensor: negative scalar centre broadcasts';
results{end,2}   = isequal(wmd_n.centre{1}, -7.5) && isequal(wmd_n.centre{2}, -7.5);

% Multi-D scalar broadcasting: r=3 absolute single attribute, d_a = 3.
% A scalar should fill all three slots.
pitchMA3 = [60 62 64; 67 69 71; 72 74 76];   % 3 slots, 3 events
dens_ma3 = buildExpTens({pitchMA3}, [], 10, 3, [], ...
    false, false, 0, 'verbose', false);
spec_sc3 = struct('size', 1.5, 'mix', 0.5, 'centre', 5.0);
wmd_sc3 = windowTensor(dens_ma3, spec_sc3);
results{end+1,1} = 'windowTensor: scalar broadcast on r=3 absolute (d_a=3)';
results{end,2}   = isequal(wmd_sc3.centre{1}, [5.0; 5.0; 5.0]);

% Compare scalar broadcast vs explicit uniform cell — must be byte-equal.
spec_ref3 = struct('size', 1.5, 'mix', 0.5, 'centre', {{[5.0; 5.0; 5.0]}});
wmd_ref3 = windowTensor(dens_ma3, spec_ref3);
results{end+1,1} = 'windowTensor: scalar broadcast == explicit uniform cell';
results{end,2}   = isequal(wmd_sc3.centre, wmd_ref3.centre) && ...
    cosSimExpTens(dens_ma3, wmd_sc3, 'verbose', false) == ...
    cosSimExpTens(dens_ma3, wmd_ref3, 'verbose', false);

% Per-attribute scalar list NOT broadcast. With A=2 and dim_total=2, the
% length-2 numeric vector [5; 10] is a valid flat-form input — and is
% accepted as such, NOT as per-attribute scalar broadcast. This is the
% intended behaviour: the scalar-broadcast bypass is reserved for
% size-1 inputs only.
spec_flat2 = struct('size', [1, 0.5], 'mix', [0, 0], 'centre', [5; 10]);
wmd_f2 = windowTensor(dens_w, spec_flat2);
results{end+1,1} = 'windowTensor: length-2 vector parsed as flat form, not broadcast';
results{end,2}   = isequal(wmd_f2.centre{1}, 5) && isequal(wmd_f2.centre{2}, 10);

% Wrong-length still errors informatively.
results{end+1,1} = 'windowTensor: wrong-length flat input errors';
results{end,2}   = throwsError(@() windowTensor(dens_ma3, ...
    struct('size', 1.5, 'mix', 0.5, 'centre', [1; 2; 3; 4])));

% -- cosSimExpTens windowed: within-attribute centre symmetrisation --
% The MAET density is symmetric under permutations of components within
% each attribute's effective coordinates (JMM windowing-theorem remark
% on within-attribute symmetry of the windowed integral). The integral
% therefore depends on the within-attribute centre components only
% through their multiset. The toolbox uses a perm-comb summation form
% internally; the v2.1 fix detects non-uniform within-attribute centres
% and averages the IP over their within-attribute permutations to
% restore framework-correct behaviour.
%
% This block mirrors the Python tests/test_windowed_within_attr_
% symmetrisation.py: 5 logical tests, expanded by parametrisation to
% 42 result rows, sharing the local helper functions
% directWindowedCosineSA and toolboxWindowedCosineSA defined at the
% end of this file.

sigma_sym = 30.0;
K_sym = 6;
sm_grid = {[5.0, 0.0], [3.0, 1.0], [4.0, 0.5]};
cp_grid = {'uniform', 'non_uniform_small_spread', 'non_uniform_large_spread'};
ORTH_FLOOR_SYM = 1e-6;
RTOL_SYM = 1e-9;

% --- 1. Toolbox cosine matches direct enumeration (perm-perm) ---
% Mirrors test_toolbox_matches_direct_for_non_uniform_centre. 2 × 3 × 3 = 18 rows.
for r_sym = [2, 3]
    for sm_idx = 1:numel(sm_grid)
        size_v = sm_grid{sm_idx}(1);
        mix_v  = sm_grid{sm_idx}(2);
        for cp_idx = 1:numel(cp_grid)
            cp = cp_grid{cp_idx};
            rng(7919*r_sym + 31*sm_idx + cp_idx);
            p_a = (rand(K_sym, 1) * 2 - 1) * 200;
            w_a = 0.5 + rand(K_sym, 1);
            p_b = (rand(K_sym, 1) * 2 - 1) * 200;
            w_b = 0.5 + rand(K_sym, 1);
            switch cp
                case 'uniform'
                    offset_vec = repmat(50.0, r_sym, 1);
                case 'non_uniform_small_spread'
                    offset_vec = linspace(-30.0, 30.0, r_sym).';
                case 'non_uniform_large_spread'
                    offset_vec = linspace(-100.0, 100.0, r_sym).';
            end
            cos_d = directWindowedCosineSA(p_a, w_a, p_b, w_b, ...
                sigma_sym, r_sym, offset_vec, size_v, mix_v);
            cos_t = toolboxWindowedCosineSA(p_a, w_a, p_b, w_b, ...
                sigma_sym, r_sym, offset_vec, size_v, mix_v);
            results{end+1,1} = sprintf( ...
                'symmetrisation: toolbox==direct r=%d (s,m)=(%g,%g) %s', ...
                r_sym, size_v, mix_v, cp);
            if abs(cos_d) < ORTH_FLOOR_SYM && abs(cos_t) < ORTH_FLOOR_SYM
                results{end,2} = true;
            else
                results{end,2} = abs(cos_t - cos_d) <= ...
                    RTOL_SYM * max(abs(cos_d), abs(cos_t));
            end
        end
    end
end

% --- 2. Cosine invariant under context-side input reordering ---
% Mirrors test_windowed_cosine_invariant_under_context_reorder.
% 2 × 3 × 3 = 18 rows; each row checks reversed AND shuffled orderings.
for r_sym = [2, 3]
    for sm_idx = 1:numel(sm_grid)
        size_v = sm_grid{sm_idx}(1);
        mix_v  = sm_grid{sm_idx}(2);
        for cp_idx = 1:numel(cp_grid)
            cp = cp_grid{cp_idx};
            rng(13591*r_sym + 41*sm_idx + 17*cp_idx + 23);
            p_a = (rand(K_sym, 1) * 2 - 1) * 200;
            w_a = 0.5 + rand(K_sym, 1);
            p_b = (rand(K_sym, 1) * 2 - 1) * 200;
            w_b = 0.5 + rand(K_sym, 1);
            switch cp
                case 'uniform'
                    offset_vec = repmat(30.0, r_sym, 1);
                case 'non_uniform_small_spread'
                    offset_vec = linspace(-25.0, 25.0, r_sym).';
                case 'non_uniform_large_spread'
                    offset_vec = linspace(-100.0, 100.0, r_sym).';
            end
            cos_orig = toolboxWindowedCosineSA(p_a, w_a, p_b, w_b, ...
                sigma_sym, r_sym, offset_vec, size_v, mix_v);
            % Reverse ordering of the context-side values.
            cos_rev = toolboxWindowedCosineSA(p_a, w_a, ...
                p_b(end:-1:1), w_b(end:-1:1), ...
                sigma_sym, r_sym, offset_vec, size_v, mix_v);
            % Random shuffle.
            perm_b = randperm(K_sym);
            cos_shuf = toolboxWindowedCosineSA(p_a, w_a, ...
                p_b(perm_b), w_b(perm_b), ...
                sigma_sym, r_sym, offset_vec, size_v, mix_v);
            results{end+1,1} = sprintf( ...
                'symmetrisation: context reorder invariant r=%d (s,m)=(%g,%g) %s', ...
                r_sym, size_v, mix_v, cp);
            if abs(cos_orig) < ORTH_FLOOR_SYM
                results{end,2} = abs(cos_rev) < ORTH_FLOOR_SYM ...
                    && abs(cos_shuf) < ORTH_FLOOR_SYM;
            else
                ok_rev = abs(cos_rev - cos_orig) <= ...
                    RTOL_SYM * max(abs(cos_orig), abs(cos_rev));
                ok_shuf = abs(cos_shuf - cos_orig) <= ...
                    RTOL_SYM * max(abs(cos_orig), abs(cos_shuf));
                results{end,2} = ok_rev && ok_shuf;
            end
        end
    end
end

% --- 3. Cosine invariant under permutation of centre entries ---
% Mirrors test_windowed_cosine_invariant_under_centre_permutation.
% 2 × 2 = 4 rows; each row checks all r_sym! permutations.
sm_grid_3 = {[5.0, 0.0], [3.0, 1.0]};
for r_sym = [2, 3]
    for sm_idx = 1:numel(sm_grid_3)
        size_v = sm_grid_3{sm_idx}(1);
        mix_v  = sm_grid_3{sm_idx}(2);
        rng(50261*r_sym + 91*sm_idx + 7);
        p_a = (rand(K_sym, 1) * 2 - 1) * 200;
        w_a = 0.5 + rand(K_sym, 1);
        p_b = (rand(K_sym, 1) * 2 - 1) * 200;
        w_b = 0.5 + rand(K_sym, 1);
        offset_vec = linspace(-50.0, 50.0, r_sym).';
        cos_orig = toolboxWindowedCosineSA(p_a, w_a, p_b, w_b, ...
            sigma_sym, r_sym, offset_vec, size_v, mix_v);
        all_perms_3 = perms(1:r_sym);
        ok_all = true;
        for ip_row = 1:size(all_perms_3, 1)
            offset_perm = offset_vec(all_perms_3(ip_row, :));
            cos_pi = toolboxWindowedCosineSA(p_a, w_a, p_b, w_b, ...
                sigma_sym, r_sym, offset_perm, size_v, mix_v);
            if abs(cos_orig) < ORTH_FLOOR_SYM
                if abs(cos_pi) >= ORTH_FLOOR_SYM
                    ok_all = false; break;
                end
            else
                if abs(cos_pi - cos_orig) > ...
                        RTOL_SYM * max(abs(cos_orig), abs(cos_pi))
                    ok_all = false; break;
                end
            end
        end
        results{end+1,1} = sprintf( ...
            'symmetrisation: centre permutation invariant r=%d (s,m)=(%g,%g)', ...
            r_sym, size_v, mix_v);
        results{end,2} = ok_all;
    end
end

% --- 4. Multi-attribute case: per-attribute symmetrisation independence ---
% Mirrors test_multi_attribute_within_attribute_symmetrisation. 1 row,
% inner check covering attr-0-reversed AND attr-1-reversed.
rng(0);
K_a4 = 4; K_b4 = 4;
p0_x = (rand(K_a4, 1) * 2 - 1) * 200; w0_x = 0.5 + rand(K_a4, 1);
p1_x = (rand(K_b4, 1) * 2 - 1) * 200; w1_x = 0.5 + rand(K_b4, 1);
p0_y = (rand(K_a4, 1) * 2 - 1) * 200; w0_y = 0.5 + rand(K_a4, 1);
p1_y = (rand(K_b4, 1) * 2 - 1) * 200; w1_y = 0.5 + rand(K_b4, 1);
dens_x_4 = buildExpTens({p0_x, p1_x}, {w0_x, w1_x}, 30.0, [2 2], [1 1], ...
    false, false, 0.0, 'verbose', false);
dens_y_4 = buildExpTens({p0_y, p1_y}, {w0_y, w1_y}, 30.0, [2 2], [1 1], ...
    false, false, 0.0, 'verbose', false);
centre_a0 = [10; 50];   centre_a1 = [-30; 20];
spec_orig_4 = struct('size', 5.0, 'mix', 0.0, ...
    'centre', {{centre_a0, centre_a1}});
spec_a0_rev_4 = struct('size', 5.0, 'mix', 0.0, ...
    'centre', {{centre_a0(end:-1:1), centre_a1}});
spec_a1_rev_4 = struct('size', 5.0, 'mix', 0.0, ...
    'centre', {{centre_a0, centre_a1(end:-1:1)}});
cos_orig_4 = cosSimExpTens(dens_x_4, ...
    windowTensor(dens_y_4, spec_orig_4), 'verbose', false);
cos_a0_rev = cosSimExpTens(dens_x_4, ...
    windowTensor(dens_y_4, spec_a0_rev_4), 'verbose', false);
cos_a1_rev = cosSimExpTens(dens_x_4, ...
    windowTensor(dens_y_4, spec_a1_rev_4), 'verbose', false);
if abs(cos_orig_4) < ORTH_FLOOR_SYM
    ok_a0 = abs(cos_a0_rev) < ORTH_FLOOR_SYM;
    ok_a1 = abs(cos_a1_rev) < ORTH_FLOOR_SYM;
else
    ok_a0 = abs(cos_a0_rev - cos_orig_4) <= ...
        RTOL_SYM * max(abs(cos_orig_4), abs(cos_a0_rev));
    ok_a1 = abs(cos_a1_rev - cos_orig_4) <= ...
        RTOL_SYM * max(abs(cos_orig_4), abs(cos_a1_rev));
end
results{end+1,1} = 'symmetrisation: multi-attribute per-attribute symmetry';
results{end,2} = ok_a0 && ok_a1;

% --- 5. Uniform-c regression: byte-identical to v2.1 fast path ---
% Mirrors test_uniform_centre_byte_identical_to_old_path. 1 row.
rng(0);
K_5 = 5;
p_a5 = (rand(K_5, 1) * 2 - 1) * 200; w_a5 = 0.5 + rand(K_5, 1);
p_b5 = (rand(K_5, 1) * 2 - 1) * 200; w_b5 = 0.5 + rand(K_5, 1);
offset_unif = repmat(25.0, 2, 1);
cos_t_5 = toolboxWindowedCosineSA(p_a5, w_a5, p_b5, w_b5, ...
    30.0, 2, offset_unif, 5.0, 0.0);
cos_d_5 = directWindowedCosineSA(p_a5, w_a5, p_b5, w_b5, ...
    30.0, 2, offset_unif, 5.0, 0.0);
results{end+1,1} = 'symmetrisation: uniform centre toolbox==direct (regression)';
results{end,2} = abs(cos_t_5 - cos_d_5) <= 1e-12 * max(abs(cos_d_5), abs(cos_t_5));

%% ---- windowedSimilarity: cross-correlation semantics (offset API) ----
% Mirror of the Python TestWindowedCrossCorrelation class. Verifies
% that at offset o the query's effective-space centroid is aligned
% with the window centre, so a peak at offset o means the query
% pattern is present in the context displaced by o from its centroid.

% Helper: build a pitch/time MAET from row vectors of (pitches, times).
% Pitch is absolute, non-periodic, r=1; time is absolute, non-periodic,
% r=1. sigmas: pitch 1.0, time 0.2.
mkPT = @(pvec, tvec) buildExpTens({pvec, tvec}, [], ...
    [1.0, 0.2], [1, 1], [], ...
    [false, false], [false, false], [0.0, 0.0], 'verbose', false);

% Helper: sweep a time offset with spec(size_time, mix_time).
function prof = cc_sweep(q, c, s, m, omin, omax, n)
    offs = linspace(omin, omax, n);
    offsets = zeros(2, n);
    offsets(2, :) = offs;   % pitch offset = 0 (pitch group unwindowed)
    spec = struct('size', [Inf, s], 'mix', [0, m]);
    prof = windowedSimilarity(q, c, spec, offsets, 'verbose', false);
end %#ok<DEFNU>

% 1. Peak at offset equal to single-event context time (mu_q = 0).
q  = mkPT(60, 0);
ct = mkPT(60, 5);
offs_cc = linspace(0, 10, 41);
prof1 = cc_sweep(q, ct, 2.0, 0.0, 0, 10, 41);
[~, ipk] = max(prof1);
results{end+1,1} = 'windowedSimilarity cross-corr: peak at single-event offset';
results{end,2}   = abs(offs_cc(ipk) - 5) < 0.3 && max(prof1) > 0.5;

% 2. Peak invariance across Gaussian window sizes.
szs = [1 2 4 8];
peak_off = zeros(size(szs));
for k = 1:numel(szs)
    pf = cc_sweep(q, ct, szs(k), 0.0, 0, 10, 41);
    [~, ip] = max(pf);
    peak_off(k) = offs_cc(ip);
end
results{end+1,1} = 'windowedSimilarity cross-corr: peak invariant over size (Gaussian)';
results{end,2}   = all(abs(peak_off - 5) < 0.3);

% 3. Peak invariance across (mix) range.
mixes = [0.0 0.25 0.5 0.75 1.0];
peak_off = zeros(size(mixes));
for k = 1:numel(mixes)
    pf = cc_sweep(q, ct, 2.0, mixes(k), 0, 10, 41);
    [~, ip] = max(pf);
    peak_off(k) = offs_cc(ip);
end
results{end+1,1} = 'windowedSimilarity cross-corr: peak invariant over mix';
results{end,2}   = all(abs(peak_off - 5) < 0.3);

% 4. Multi-event query peak at centroid offset.
% Query centroid at t=0.5; context motif centroid at t=5.5; offset 5.0.
q2 = mkPT([60 64], [0 1]);
c2 = mkPT([60 64], [5 6]);
offs_fine = linspace(0, 10, 101);
offsets_fine = zeros(2, 101);
offsets_fine(2, :) = offs_fine;
prof4 = windowedSimilarity(q2, c2, ...
    struct('size', [Inf 2], 'mix', [0 0]), ...
    offsets_fine, 'verbose', false);
[~, ipk] = max(prof4);
results{end+1,1} = 'windowedSimilarity cross-corr: multi-event peak at centroid offset';
results{end,2}   = abs(offs_fine(ipk) - 5.0) < 0.2;

% 5. Two recurrences of the motif produce two equal peaks.
% Motif centroids at 2.5 and 5.5; query centroid at 0.5; peaks at 2.0 and 5.0.
c2b = mkPT([60 64 60 64], [2 3 5 6]);
offs_finer = linspace(0, 10, 201);
offsets_finer = zeros(2, 201);
offsets_finer(2, :) = offs_finer;
prof5 = windowedSimilarity(q2, c2b, ...
    struct('size', [Inf 2], 'mix', [0 0]), ...
    offsets_finer, 'verbose', false);
lm = (prof5(2:end-1) > prof5(1:end-2)) & ...
     (prof5(2:end-1) > prof5(3:end))  & ...
     (prof5(2:end-1) > 0.5 * max(prof5));
pk_idx = find(lm) + 1;
results{end+1,1} = 'windowedSimilarity cross-corr: two motif recurrences -> two peaks';
results{end,2}   = (numel(pk_idx) == 2) ...
                   && all(abs(sort(offs_finer(pk_idx)) - [2.0 5.0]) < 0.15) ...
                   && abs(prof5(pk_idx(1)) - prof5(pk_idx(2))) < 1e-3;

% 6. isRel=true concurrent dyad: time peak invariant under pitch
%    translation of the context.
q_dy  = buildExpTens({[60; 64], 0}, [], ...
                     [1.0, 0.2], [2, 1], [], ...
                     [true, false], [false, false], [0, 0], ...
                     'verbose', false);
c_dy1 = buildExpTens({[60; 64], 5}, [], ...
                     [1.0, 0.2], [2, 1], [], ...
                     [true, false], [false, false], [0, 0], ...
                     'verbose', false);
c_dy2 = buildExpTens({[70; 74], 5}, [], ...
                     [1.0, 0.2], [2, 1], [], ...
                     [true, false], [false, false], [0, 0], ...
                     'verbose', false);
spec_dy = struct('size', [Inf 2], 'mix', [0 0]);
offs_dy = linspace(0, 10, 101);
% dim = (r_pitch - isRel_pitch) + (r_time - isRel_time)
%     = (2 - 1) + (1 - 0) = 2 rows.
offsets_dy = zeros(2, 101);
offsets_dy(2, :) = offs_dy;
p_d1 = windowedSimilarity(q_dy, c_dy1, spec_dy, offsets_dy, 'verbose', false);
p_d2 = windowedSimilarity(q_dy, c_dy2, spec_dy, offsets_dy, 'verbose', false);
[~, i1] = max(p_d1);
[~, i2] = max(p_d2);
results{end+1,1} = ['windowedSimilarity cross-corr: isRel dyad peak ' ...
                    'invariant under pitch translation'];
results{end,2}   = (i1 == i2) && abs(offs_dy(i1) - 5) < 0.3;

% 7. Unwindowed cosSimExpTens on identical densities == 1 (unwindowed
%    path untouched).
d_id = mkPT([60 64 67], [0 1 2]);
s_id = cosSimExpTens(d_id, d_id, 'verbose', false);
results{end+1,1} = 'windowedSimilarity cross-corr: unwindowed cos_sim identical == 1';
results{end,2}   = abs(s_id - 1) < 1e-10;

% 8. Unwindowed cos_sim on distinct MA densities stays in (0, 1).
d_a = mkPT([60 64 67], [0 1 2]);
d_b = mkPT([60 65 67], [0 1 2]);
s_ab = cosSimExpTens(d_a, d_b, 'verbose', false);
results{end+1,1} = 'windowedSimilarity cross-corr: unwindowed cos_sim distinct in (0,1)';
results{end,2}   = s_ab > 0 && s_ab < 1;

% 9. Pitch mismatch suppresses the matched-offset peak.
c_miss  = mkPT(72, 5);
c_match = mkPT(60, 5);
p_miss  = cc_sweep(q, c_miss, 2.0, 0.0, 0, 10, 41);
p_match = cc_sweep(q, c_match, 2.0, 0.0, 0, 10, 41);
results{end+1,1} = 'windowedSimilarity cross-corr: pitch mismatch suppresses peak';
results{end,2}   = max(p_miss) < 0.01 * max(p_match);

% 10. Peak height increases monotonically with window size and approaches
%     1 as size -> Inf.
szs_h = [1 2 8 100];
pk_h  = zeros(size(szs_h));
for k = 1:numel(szs_h)
    pf = cc_sweep(q, ct, szs_h(k), 0.0, 4, 6, 201);
    pk_h(k) = max(pf);
end
results{end+1,1} = 'windowedSimilarity cross-corr: peak height increases with size';
results{end,2}   = all(diff(pk_h) > 0) && pk_h(end) > 0.99;

% --- v2.1 unified dispatch: windowedSimilarity list mode ----

% Build a tiny pair of MaetDensity queries and contexts for list-mode
% testing. We reuse the shape from the earlier section: a single-event
% query against a 4-event context with periodic pitch + non-periodic
% time, sweeping along time only.
qList_a = buildExpTens({62, 0}, [], [0.5 0.1], [1 1], [], ...
    [false false], [true false], [1200 0], 'verbose', false);
qList_b = buildExpTens({64, 0}, [], [0.5 0.1], [1 1], [], ...
    [false false], [true false], [1200 0], 'verbose', false);
cList_a = buildExpTens({[60 62 64 65], [0 1 2 3]}, [], ...
    [0.5 0.1], [1 1], [], [false false], [true false], [1200 0], ...
    'verbose', false);
cList_b = buildExpTens({[60 64 67 71], [0 1 2 3]}, [], ...
    [0.5 0.1], [1 1], [], [false false], [true false], [1200 0], ...
    'verbose', false);
M_list = 9;
offs_list = zeros(2, M_list);
offs_list(2, :) = linspace(-0.5, 3.5, M_list);
spec_list = struct('size', [Inf, 0.3], 'mix', [0, 0]);

% Pairwise mode: equal-length lists give cell of profiles
prof_pair = windowedSimilarity({qList_a, qList_b}, {cList_a, cList_b}, ...
    spec_list, offs_list, 'verbose', false);
results{end+1,1} = 'windowedSimilarity list pairwise: returns 1-by-n cell';
results{end,2}   = iscell(prof_pair) && isequal(size(prof_pair), [1, 2]);

prof_a_scalar = windowedSimilarity(qList_a, cList_a, spec_list, offs_list, ...
    'verbose', false);
prof_b_scalar = windowedSimilarity(qList_b, cList_b, spec_list, offs_list, ...
    'verbose', false);
results{end+1,1} = 'windowedSimilarity list pairwise: matches scalar dispatch element-wise';
results{end,2}   = max(abs(prof_pair{1} - prof_a_scalar)) < 1e-12 ...
                   && max(abs(prof_pair{2} - prof_b_scalar)) < 1e-12;

% Cartesian mode: different-length lists or explicit 'mode' = 'cartesian'
prof_cart = windowedSimilarity({qList_a, qList_b}, {cList_a, cList_b}, ...
    spec_list, offs_list, 'verbose', false, 'mode', 'cartesian');
results{end+1,1} = 'windowedSimilarity list cartesian: returns nQ-by-nC cell';
results{end,2}   = iscell(prof_cart) && isequal(size(prof_cart), [2, 2]);

prof_ab_scalar = windowedSimilarity(qList_a, cList_b, spec_list, offs_list, ...
    'verbose', false);
results{end+1,1} = 'windowedSimilarity list cartesian: matches scalar dispatch (i,j)';
results{end,2}   = max(abs(prof_cart{1, 2} - prof_ab_scalar)) < 1e-12;

% Auto mode: equal lengths -> pairwise
prof_auto_p = windowedSimilarity({qList_a, qList_b}, {cList_a, cList_b}, ...
    spec_list, offs_list, 'verbose', false, 'mode', 'auto');
results{end+1,1} = 'windowedSimilarity list auto: equal lengths -> pairwise cell';
results{end,2}   = iscell(prof_auto_p) && isequal(size(prof_auto_p), [1, 2]);

% Auto mode: unequal lengths -> cartesian
prof_auto_c = windowedSimilarity({qList_a}, {cList_a, cList_b}, ...
    spec_list, offs_list, 'verbose', false, 'mode', 'auto');
results{end+1,1} = 'windowedSimilarity list auto: unequal lengths -> cartesian';
results{end,2}   = iscell(prof_auto_c) && isequal(size(prof_auto_c), [1, 2]);

% Pairwise with mismatched lengths errors
results{end+1,1} = 'windowedSimilarity list pairwise: length mismatch errors';
results{end,2}   = throwsErrorWithId( ...
    @() windowedSimilarity({qList_a, qList_b}, {cList_a}, spec_list, offs_list, ...
        'verbose', false, 'mode', 'pairwise'), ...
    'windowedSimilarity:listLengthMismatch');

% Length-1 list returns length-1 cell (Option II)
prof_one = windowedSimilarity({qList_a}, {cList_a}, spec_list, offs_list, ...
    'verbose', false);
results{end+1,1} = 'windowedSimilarity list: length-1 returns length-1 cell';
results{end,2}   = iscell(prof_one) && isequal(size(prof_one), [1, 1]);

% Scalar query + list context (broadcast query)
prof_qScalar = windowedSimilarity(qList_a, {cList_a, cList_b}, ...
    spec_list, offs_list, 'verbose', false);
results{end+1,1} = 'windowedSimilarity list: scalar query + list context';
results{end,2}   = iscell(prof_qScalar) && numel(prof_qScalar) == 2;

% List query + scalar context (broadcast context)
prof_cScalar = windowedSimilarity({qList_a, qList_b}, cList_a, ...
    spec_list, offs_list, 'verbose', false);
results{end+1,1} = 'windowedSimilarity list: list query + scalar context';
results{end,2}   = iscell(prof_cScalar) && numel(prof_cScalar) == 2;

% Per-query reference (cell-of-cells form)
% qList_a has 2 attributes (pitch, time), so each per-query reference
% is a 1-by-2 cell of attribute-dimension vectors.
ref_per_a = {{[62], [0]}, {[64], [0]}};   % length-2 cell of length-2 cells
prof_perRef = windowedSimilarity({qList_a, qList_b}, {cList_a, cList_b}, ...
    spec_list, offs_list, 'verbose', false, ...
    'reference', ref_per_a, 'mode', 'pairwise');
prof_a_perRef_scalar = windowedSimilarity(qList_a, cList_a, spec_list, offs_list, ...
    'verbose', false, 'reference', {[62], [0]});
results{end+1,1} = 'windowedSimilarity list: per-query reference forwarded correctly';
results{end,2}   = max(abs(prof_perRef{1} - prof_a_perRef_scalar)) < 1e-12;

% Shared reference (single length-A cell broadcast to all queries)
ref_shared = {[62], [0]};   % length-2 cell of vectors -> shared
prof_sharedRef = windowedSimilarity({qList_a, qList_b}, {cList_a, cList_b}, ...
    spec_list, offs_list, 'verbose', false, ...
    'reference', ref_shared, 'mode', 'pairwise');
prof_b_sharedRef_scalar = windowedSimilarity(qList_b, cList_b, spec_list, offs_list, ...
    'verbose', false, 'reference', ref_shared);
results{end+1,1} = 'windowedSimilarity list: shared reference broadcast';
results{end,2}   = max(abs(prof_sharedRef{2} - prof_b_sharedRef_scalar)) < 1e-12;

% Bad mode errors
results{end+1,1} = 'windowedSimilarity: bad mode value errors';
results{end,2}   = throwsErrorWithId( ...
    @() windowedSimilarity(qList_a, cList_a, spec_list, offs_list, ...
        'verbose', false, 'mode', 'bogus'), ...
    'windowedSimilarity:badMode');

%% ---- simplexVertices ----

% (Pairwise distances via local helper pairwiseDistances at end of file,
%  avoiding pdist / Statistics Toolbox.)

% Shapes: simplexVertices(N) returns N x (N-1).
for N = [2 3 4 5 10]
    V = simplexVertices(N);
    results{end+1,1} = sprintf('simplexVertices: shape (N=%d)', N); %#ok<SAGROW>
    results{end,2}   = isequal(size(V), [N, N-1]);
end

% Centroid is at the origin.
for N = [2 3 4 5 7]
    V = simplexVertices(N);
    results{end+1,1} = sprintf('simplexVertices: centroid at origin (N=%d)', N); %#ok<SAGROW>
    results{end,2}   = max(abs(mean(V, 1))) < 1e-12;
end

% Default edge length is 1: all pairwise distances equal 1.
for N = [2 3 4 5 7]
    V = simplexVertices(N);
    D = pairwiseDistances(V);
    results{end+1,1} = sprintf('simplexVertices: unit edge length (N=%d)', N); %#ok<SAGROW>
    results{end,2}   = max(abs(D - 1)) < 1e-12;
end

% Custom edge length scales correctly.
for L = [0.5 2.0 100.0]
    V = simplexVertices(4, L);
    D = pairwiseDistances(V);
    results{end+1,1} = sprintf('simplexVertices: edge length %g', L); %#ok<SAGROW>
    results{end,2}   = max(abs(D - L)) < 1e-10;
end

% N = 2 collapses to a 1-D pair, distance 1, centred at origin.
V = simplexVertices(2);
results{end+1,1} = 'simplexVertices: N=2 collapses to 1-D';
results{end,2}   = isequal(size(V), [2, 1]) ...
                   && abs(V(1) + V(2)) < 1e-12 ...
                   && abs(abs(V(1) - V(2)) - 1) < 1e-12;

% Error paths: N < 2, non-positive edge length.
results{end+1,1} = 'simplexVertices: N=1 errors';
results{end,2}   = throwsError(@() simplexVertices(1));

results{end+1,1} = 'simplexVertices: N=0 errors';
results{end,2}   = throwsError(@() simplexVertices(0));

results{end+1,1} = 'simplexVertices: negative edge length errors';
results{end,2}   = throwsError(@() simplexVertices(3, -1));

results{end+1,1} = 'simplexVertices: zero edge length errors';
results{end,2}   = throwsError(@() simplexVertices(3, 0));

%% ---- v2.2 tests (matlab/tests/v22/) ----
% v2.2-dev tests live in tests/v22/ and follow the same `results = {...}`
% accumulation idiom. Each script appends to the existing `results` cell
% when invoked from here; when run alone, each prints its own summary.

v22Dir = fullfile(fileparts(mfilename('fullpath')), 'v22');
v22Files = { ...
    'test_mobius_combinatorics.m', ...
    'test_mobius_orbit_table.m', ...
    'test_mobius_ip.m', ...
    'test_mobius_eval.m', ...
    'test_dispatch_sa_cossim.m', ...
    'test_dispatch_sa_eval.m', ...
    'test_dispatch_ma_cossim.m', ...
    'test_tensor_harmonicity_orbit.m', ...
    'test_entropy_renyi2.m', ...
    'test_ma_per_attr_hybrid.m', ...
    'test_cross_language_golden.m', ...
    'test_recipe_equivalence.m', ...
    'test_orbit_vectorisation.m', ...
    'test_kernel_truncation.m', ...
    'test_eval_routing.m', ...
    'test_wrapper_routing.m', ...
    'test_cossim_centres_routing.m', ...
    'test_dispatcher_probe.m', ...
};
for ki = 1:numel(v22Files)
    run(fullfile(v22Dir, v22Files{ki}));
end

%% ---- Print results ----

nPass = sum([results{:,2}]);
nFail = size(results, 1) - nPass;

for i = 1:size(results, 1)
    if results{i,2}
        fprintf('  PASS  %s\n', results{i,1});
    else
        fprintf('  FAIL  %s\n', results{i,1});
    end
end

fprintf('\n=== Results: %d passed, %d failed (of %d) ===\n\n', ...
    nPass, nFail, nPass + nFail);
if nFail > 0
    error('test_mpt:failed', '%d test(s) failed.', nFail);
end

%% ---- Helper ----

function tf = throwsError(fn)
    try
        fn();
        tf = false;
    catch
        tf = true;
    end
end

function tf = throwsErrorWithId(fn, expectedId)
    try
        fn();
        tf = false;
    catch ME
        tf = strcmp(ME.identifier, expectedId);
    end
end

function tf = errorMessageContains(fn, expectedSubstr)
    try
        fn();
        tf = false;
    catch ME
        tf = ~isempty(strfind(ME.message, expectedSubstr));
    end
end

function D = pairwiseDistances(V)
%PAIRWISEDISTANCES Pairwise Euclidean distances between rows of V.
%
%   D = pairwiseDistances(V) returns a column vector of length
%   N*(N-1)/2 containing the pairwise Euclidean distances between
%   the N rows of V, in the order
%   (1,2), (1,3), (2,3), (1,4), (2,4), (3,4), ...
%
%   Computed via the Gram matrix:
%       D2(i,j) = ||v_i||^2 + ||v_j||^2 - 2 * v_i * v_j'
%   so no Statistics Toolbox is required.
    n = size(V, 1);
    sqNorm = sum(V .* V, 2);
    D2 = sqNorm + sqNorm' - 2 * (V * V');
    D2 = max(D2, 0);                       % clamp tiny negatives from FP error
    mask = triu(true(n), 1);               % strict upper triangle
    D = sqrt(D2(mask));
end

% =====================================================================
% Helpers for the within-attribute centre symmetrisation tests.
% Mirror the helper structure of Python's tests/test_windowed_within_
% attr_symmetrisation.py: a direct (perm-perm) enumeration reference
% and a toolbox-API wrapper, both for the SA case.
% =====================================================================

function P = orderedPermutations(K, r)
%ORDEREDPERMUTATIONS  All ordered r-tuples drawn without replacement
%from {1, ..., K}. Output is M-by-r where M = K!/(K-r)!.
    if r == 0
        P = zeros(1, 0);
        return;
    end
    if r > K
        P = zeros(0, r);
        return;
    end
    combos = nchoosek(1:K, r);                 % nchoosek(K,r) x r
    nC = size(combos, 1);
    P = zeros(nC * factorial(r), r);
    row = 1;
    for i = 1:nC
        ps = perms(combos(i, :));               % r! x r
        nP = size(ps, 1);
        P(row : row + nP - 1, :) = ps;
        row = row + nP;
    end
end

function f = perAxisF(mu_shift, a_rect, b_conv, sigma_pair)
%PERAXISF  Closed-form per-axis windowed factor for the rect-conv-
%Gaussian family. Mirrors localWindowedFactorisable's per-axis form.
    sigma_t = sqrt(sigma_pair^2 + b_conv^2);
    if a_rect == 0 && b_conv > 0
        f = (b_conv / sigma_t) * exp(-mu_shift^2 / (2 * sigma_t^2));
    elseif b_conv == 0 && a_rect > 0
        d = sigma_t * sqrt(2);
        f = 0.5 * (erf((mu_shift + a_rect) / d) ...
                  - erf((mu_shift - a_rect) / d));
    else
        d = sigma_t * sqrt(2);
        num = erf((mu_shift + a_rect) / d) ...
            - erf((mu_shift - a_rect) / d);
        f = num / (2 * erf(a_rect / (b_conv * sqrt(2))));
    end
end

function ip = directWindowedCrossCorr(p_a, w_a, p_b, w_b, sigma, r, ...
        offset_vec, mu_q, a_rect, b_conv)
%DIRECTWINDOWEDCROSSCORR  Framework-correct (perm-perm) reference for
%the windowed cross-correlation IP, in the toolbox's translated form
%(kernel_shift = offset - mu_q, F-centre = (mu_q + offset)/2).
    K_a = numel(p_a); K_b = numel(p_b);
    sigma_pair = sigma / sqrt(2);
    Tperm_a = orderedPermutations(K_a, r);
    Tperm_b = orderedPermutations(K_b, r);
    total = 0;
    for ia = 1:size(Tperm_a, 1)
        ta = Tperm_a(ia, :);
        for ib = 1:size(Tperm_b, 1)
            tb = Tperm_b(ib, :);
            contrib = 1;
            for l = 1:r
                d = (p_a(ta(l)) - p_b(tb(l))) ...
                  + (offset_vec(l) - mu_q);
                K_l = exp(-d^2 / (4 * sigma^2));
                f_centre = (mu_q + offset_vec(l)) / 2;
                mid = (p_a(ta(l)) + p_b(tb(l))) / 2;
                F_l = perAxisF(mid - f_centre, a_rect, b_conv, sigma_pair);
                contrib = contrib * K_l * F_l ...
                        * w_a(ta(l)) * w_b(tb(l));
            end
            total = total + contrib;
        end
    end
    ip = total * (sigma * sqrt(pi))^r;
end

function ip = directUnwindowedAbs(p, w, sigma, r)
%DIRECTUNWINDOWEDABS  Direct (perm-perm) enumeration of the unwindowed
%absolute non-periodic IP, used for self-norms in the cosine.
    K = numel(p);
    Tperm = orderedPermutations(K, r);
    total = 0;
    for ia = 1:size(Tperm, 1)
        ta = Tperm(ia, :);
        for ib = 1:size(Tperm, 1)
            tb = Tperm(ib, :);
            contrib = 1;
            for l = 1:r
                d = p(ta(l)) - p(tb(l));
                K_l = exp(-d^2 / (4 * sigma^2));
                contrib = contrib * K_l * w(ta(l)) * w(tb(l));
            end
            total = total + contrib;
        end
    end
    ip = total * (sigma * sqrt(pi))^r;
end

function c = directWindowedCosineSA(p_a, w_a, p_b, w_b, sigma, r, ...
        offset_vec, size_v, mix_v)
%DIRECTWINDOWEDCOSINESA  Framework-correct windowed cosine for the SA
%case, via direct perm-perm enumeration of cross-correlation IP and
%unwindowed self-norms.
    a_rect = size_v * sigma * sqrt(3 * mix_v);
    b_conv = size_v * sigma * sqrt(1 - mix_v);
    mu_q = mean(p_a);
    ip_xy = directWindowedCrossCorr(p_a, w_a, p_b, w_b, sigma, r, ...
        offset_vec, mu_q, a_rect, b_conv);
    ip_xx = directUnwindowedAbs(p_a, w_a, sigma, r);
    ip_yy = directUnwindowedAbs(p_b, w_b, sigma, r);
    c = ip_xy / sqrt(ip_xx * ip_yy);
end

function c = toolboxWindowedCosineSA(p_a, w_a, p_b, w_b, sigma, r, ...
        offset_vec, size_v, mix_v)
%TOOLBOXWINDOWEDCOSINESA  Toolbox-API windowed cosine for the SA
%case, used as the path under test in the symmetrisation suite.
    Pa = p_a(:);  Wa = w_a(:);
    Pb = p_b(:);  Wb = w_b(:);
    dens_q = buildExpTens({Pa}, {Wa}, sigma, r, [], ...
        false, false, 0, 'verbose', false);
    dens_c = buildExpTens({Pb}, {Wb}, sigma, r, [], ...
        false, false, 0, 'verbose', false);
    spec = struct('size', size_v, 'mix', mix_v, ...
        'centre', {{offset_vec(:)}});
    wmd = windowTensor(dens_c, spec);
    c = cosSimExpTens(dens_q, wmd, 'verbose', false);
end
