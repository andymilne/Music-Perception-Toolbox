%% test_circular.m — circular measures (rPhase, edges, projCentroid, meanOffset, etc.)
%
%  Tests for circular measures (rPhase, edges, projCentroid, meanOffset, etc.).
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
% Six events filling positions 0..5 of a 12-position circle: rising edge
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


%% ---- Standalone summary ----

if standalone
    nPass = sum([results{:, 2}]);
    nFail = size(results, 1) - nPass;
    for i = 1:size(results, 1)
        if results{i, 2}
            fprintf('  PASS  %s\n', results{i, 1});
        else
            fprintf('  FAIL  %s\n', results{i, 1});
        end
    end
    fprintf('\n=== test_circular: %d passed, %d failed (of %d) ===\n\n', ...
        nPass, nFail, nPass + nFail);
    clear cleanupDefaults
    if nFail > 0
        error('test_circular:failed', '%d test(s) failed.', nFail);
    end
end
