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

H = entropyExpTens(0:11, ones(1,12), 100, 1, false, true, 12);
results{end+1,1} = 'entropyExpTens: uniform ≈ 1';
results{end,2}   = H > 0.95;

%% ---- Harmony ----

r = roughness(440, 1);
results{end+1,1} = 'roughness: unison = 0';
results{end,2}   = abs(r) < 1e-10;

r = roughness([300, 330], [1, 1]);
results{end+1,1} = 'roughness: positive for nearby freqs';
results{end,2}   = r > 0;

spec = {'harmonic', 24, 'powerlaw', 1};
H_ji = spectralEntropy([0, 386.31, 701.96], [], 12, 'spectrum', spec);
H_edo = spectralEntropy([0, 400, 700], [], 12, 'spectrum', spec);
results{end+1,1} = 'spectralEntropy: JI < EDO';
results{end,2}   = H_ji < H_edo;

[hMax, hEnt] = templateHarmonicity([0, 400, 700], [], 12);
results{end+1,1} = 'templateHarmonicity: hMax in (0,1]';
results{end,2}   = hMax > 0 && hMax <= 1;
results{end+1,1} = 'templateHarmonicity: hEntropy in (0,1]';
results{end,2}   = hEnt > 0 && hEnt <= 1;

% -- templateHarmonicity: hEntropy lower for octave than for cluster --
[~, hEnt_oct] = templateHarmonicity([0, 1200], [], 12);
[~, hEnt_clu] = templateHarmonicity([0, 100, 200], [], 12);
results{end+1,1} = 'templateHarmonicity: hEntropy octave < cluster';
results{end,2}   = hEnt_oct < hEnt_clu;

% -- templateHarmonicity: hEntropy lower for major triad than for cluster
% (3-note vs 3-note, controlling for cardinality) --
[~, hEnt_maj] = templateHarmonicity([0, 400, 700], [], 12);
[~, hEnt_clu] = templateHarmonicity([0, 100, 200], [], 12);
results{end+1,1} = 'templateHarmonicity: hEntropy major triad < cluster';
results{end,2}   = hEnt_maj < hEnt_clu;

spec = {'harmonic', 12, 'powerlaw', 1};
h_uni = tensorHarmonicity([0, 0], [], 12, 'spectrum', spec);
h_tri = tensorHarmonicity([0, 600], [], 12, 'spectrum', spec);
results{end+1,1} = 'tensorHarmonicity: unison > tritone';
results{end,2}   = h_uni > h_tri;

% -- tensorHarmonicity: ordered ranking
% octave (2:1) > perfect 5th (3:2) > major triad (4:5:6) > minor triad --
h_oct  = tensorHarmonicity([0, 1200],     [], 12, 'spectrum', spec);
h_p5   = tensorHarmonicity([0, 700],      [], 12, 'spectrum', spec);
h_maj  = tensorHarmonicity([0, 400, 700], [], 12, 'spectrum', spec);
h_min  = tensorHarmonicity([0, 300, 700], [], 12, 'spectrum', spec);
results{end+1,1} = 'tensorHarmonicity: octave > P5 > major > minor';
results{end,2}   = (h_oct > h_p5) && (h_p5 > h_maj) && (h_maj > h_min);

[vp_p, vp_w] = virtualPitches([0, 400, 700], [], 12);
results{end+1,1} = 'virtualPitches: non-empty';
results{end,2}   = numel(vp_p) > 0;
results{end+1,1} = 'virtualPitches: lengths match';
results{end,2}   = numel(vp_p) == numel(vp_w);

% -- virtualPitches: peak of a single pitch sits at the pitch itself --
[vp_p1, vp_w1] = virtualPitches(400, [], 12);
[~, i_max1] = max(vp_w1);
results{end+1,1} = 'virtualPitches: single pitch peak at the pitch';
results{end,2}   = abs(vp_p1(i_max1) - 400) < 5;

% -- virtualPitches: peak of an octave dyad at the lower note
% (partials 1 and 2 of a template at 0 align with both chord notes) --
[vp_p2, vp_w2] = virtualPitches([0, 1200], [], 12);
[~, i_max2] = max(vp_w2);
results{end+1,1} = 'virtualPitches: octave peak at lower note';
results{end,2}   = abs(vp_p2(i_max2)) < 5;

%% ---- estimateCompTime ----

est = estimateCompTime(1000, 2, 'test');
results{end+1,1} = 'estimateCompTime: positive';
results{end,2}   = est > 0;

%% ---- Input validation ----

results{end+1,1} = 'buildExpTens: r too large errors';
results{end,2}   = throwsError(@() buildExpTens([0, 4], [], 10, 3, ...
    false, true, 12, 'verbose', false));

results{end+1,1} = 'buildExpTens: isRel with r=1 errors';
results{end,2}   = throwsError(@() buildExpTens([0, 4, 7], [], 10, 1, ...
    true, true, 12, 'verbose', false));

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

for isRel_ = [false, true]
    dens_sa = buildExpTens(p_sa, w_sa, sigma, r_, isRel_, isPer_, period_, ...
        'verbose', false);
    dens_ma = buildExpTens({p_sa}, {w_sa}, sigma, r_, [], isRel_, isPer_, period_, ...
        'verbose', false);

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
dens_ma = buildExpTens({p_sa}, {w_sa}, sigma, 2, [], true, true, 1200, 'verbose', false);
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
    'verbose', false);

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
    'verbose', false);
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
    'verbose', false);
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
    'nPointsPerDim', 400);
H_ma = entropyExpTens({p_e}, {w_e}, 10, 1, [], false, true, 12, ...
    'nPointsPerDim', 400);
results{end+1,1} = 'entropyExpTens MA: SA-equivalence periodic';
results{end,2}   = abs(H_ma - H_sa) < 1e-10;

% -- entropyExpTens MA: SA-equivalence non-periodic --

H_sa = entropyExpTens(p_e.', w_e.', 10, 1, false, false, 0, ...
    'xMin', -3, 'xMax', 10, 'nPointsPerDim', 400);
H_ma = entropyExpTens({p_e}, {w_e}, 10, 1, [], false, false, 0, ...
    'xMin', -3, 'xMax', 10, 'nPointsPerDim', 400);
results{end+1,1} = 'entropyExpTens MA: SA-equivalence non-periodic';
results{end,2}   = abs(H_ma - H_sa) < 1e-10;

% -- entropyExpTens MA: uniform pitch near 1 --

p_uniform = (0:11).';
H_u = entropyExpTens({p_uniform}, [], 100, 1, [], false, true, 12, ...
    'nPointsPerDim', 400);
results{end+1,1} = 'entropyExpTens MA: uniform chromatic near 1';
results{end,2}   = H_u > 0.95;

% -- entropyExpTens MA: concentrated below uniform --

H_one = entropyExpTens({5}, [], 20, 1, [], false, true, 12, ...
    'nPointsPerDim', 400);
H_all = entropyExpTens({p_uniform}, [], 20, 1, [], false, true, 12, ...
    'nPointsPerDim', 400);
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
    'xMin', -0.5, 'xMax', 1.5, 'nPointsPerDim', 80);
results{end+1,1} = 'entropyExpTens MA: pitch+time H in (0,1)';
results{end,2}   = H_pt > 0 && H_pt < 1;

% -- entropyExpTens MA: grid-limit guard --

results{end+1,1} = 'entropyExpTens MA: grid-limit exceeded errors';
results{end,2}   = throwsError(@() entropyExpTens(densE, ...
    'xMin', 0, 'xMax', 2, 'nPointsPerDim', 20000, 'gridLimit', 1e6));

% -- entropyExpTens MA: missing bounds error --

results{end+1,1} = 'entropyExpTens MA: missing non-periodic bounds errors';
results{end,2}   = throwsError(@() entropyExpTens({p_e}, [], 10, 1, [], ...
    false, false, 0, 'nPointsPerDim', 100));

% -- entropyExpTens MA: per-group bounds vector matches scalar --

H_scalar = entropyExpTens(densE, ...
    'xMin', -0.5, 'xMax', 1.5, 'nPointsPerDim', 60);
H_vec = entropyExpTens(densE, ...
    'xMin', [NaN, -0.5], 'xMax', [NaN, 1.5], 'nPointsPerDim', 60);
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
    'verbose', false);

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

H_mr_rect = entropyExpTens(wmd_mr_rect, 'nPointsPerDim', 20);
results{end+1,1} = 'windowTensor: entropy on rect-windowed multi-D rel runs';
results{end,2}   = isfinite(H_mr_rect);

% -- windowTensor: narrower window yields lower entropy --

H_base = entropyExpTens(dens_w, 'xMin', [0, -1], 'xMax', [1200, 4], ...
                         'nPointsPerDim', 60);
spec_ew = struct('size', [Inf, 0.3], 'mix', [0, 0], ...
                 'centre', {{zeros(1, 1), 1.0}});
wmd_ew = windowTensor(dens_w, spec_ew);
H_narrow = entropyExpTens(wmd_ew, 'xMin', [0, -1], 'xMax', [1200, 4], ...
                           'nPointsPerDim', 60);
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
