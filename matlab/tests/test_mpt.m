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

[y, cm, ~] = projCentroid([0, 400, 800], [], 1200);
results{end+1,1} = 'projCentroid: balanced magnitude = 0';
results{end,2}   = abs(cm) < 1e-10;
results{end+1,1} = 'projCentroid: balanced projections = 0';
results{end,2}   = all(abs(y) < 1e-10);

h = meanOffset([0, 2, 4, 5, 7, 9, 11], [], 12);
results{end+1,1} = 'meanOffset: output length';
results{end,2}   = numel(h) == 12;

[R, rp, rl] = circApm([0, 3, 6, 10, 12], [], 16);
results{end+1,1} = 'circApm: R shape';
results{end,2}   = isequal(size(R), [16, 16]);
results{end+1,1} = 'circApm: rPhase length';
results{end,2}   = numel(rp) == 16;
results{end+1,1} = 'circApm: rLag length';
results{end,2}   = numel(rl) == 16;

y = markovS([0, 3, 6, 10, 12], [], 16);
results{end+1,1} = 'markovS: output length';
results{end,2}   = numel(y) == 16;
results{end+1,1} = 'markovS: positive at events';
results{end,2}   = y(1) > 0 && y(4) > 0;

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

spec = {'harmonic', 12, 'powerlaw', 1};
h_uni = tensorHarmonicity([0, 0], [], 12, 'spectrum', spec);
h_tri = tensorHarmonicity([0, 600], [], 12, 'spectrum', spec);
results{end+1,1} = 'tensorHarmonicity: unison > tritone';
results{end,2}   = h_uni > h_tri;

[vp_p, vp_w] = virtualPitches([0, 400, 700], [], 12);
results{end+1,1} = 'virtualPitches: non-empty';
results{end,2}   = numel(vp_p) > 0;
results{end+1,1} = 'virtualPitches: lengths match';
results{end,2}   = numel(vp_p) == numel(vp_w);

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