%% demo_rhythmTensors.m
%  Expectation tensors of rhythms: the analyses the other demos apply to
%  pitch, applied to time.
%
%  A rhythm here is a multiset of onsets, one element per onset, in a
%  cycle of 16 pulses (the equally spaced time points of the cycle),
%  with time periodic at the cycle length: cycle equivalence, the
%  counterpart of octave equivalence. Only the period changes; every
%  function is the one used for pitch. The running example is the son
%  clave [0, 3, 6, 10, 12], compared with five other well-known 16-pulse
%  timelines. Sigma is 0.5 pulses, except in the limit taken in 4a.
%
%  Sections:
%    1. Similarity: absolute (r = 1) against relative (r = 2)
%    2. Density: the r = 1 and r = 2 relative densities (figure)
%    3. Complexity: entropy of the r = 2 relative tensor
%    4. Circular differencing and binding, and n-tuple entropy
%
%  Uses: simMaet, buildMaet, plotMaet, entropyMaet, flatSpecs,
%        packPreMaet, differenceEvents, bindEvents, nTupleEntropy,
%        mptDefaults
%  (from the Music Perception Toolbox).
%
%  The Python mirror is demo_rhythm_tensors.py.

clear; clc;

% Informational hints are switched off, and restored at the end.
prevDefaults = mptDefaults('showHints', false);

PERIOD = 16;
SIGMA  = 0.5;
son    = [0, 3, 6, 10, 12];
names  = {'son clave 2-3', 'rumba clave', 'bossa nova', 'gahu', ...
          'shiko', 'soukous'};
others = [2, 4,  8, 11, 14;      % the son clave with its two halves swapped
          0, 3,  7, 10, 12;
          0, 3,  6, 10, 13;
          0, 3,  6, 10, 14;
          0, 4,  6, 10, 12;
          0, 3,  6, 10, 11];

%% === 1. Similarity: absolute (r = 1) against relative (r = 2) ===

% At r = 1, absolute, the density records where in the cycle the onsets
% fall, so similarity depends on phase. At r = 2, relative, it records
% the interval between every pair of onsets (an inter-onset interval,
% IOI, taken between any two onsets, not only successive ones): the
% rhythmic counterpart of the interval vector, unchanged when the
% rhythm is rotated (every onset shifted by the same number of pulses).
fprintf('\n=== 1. Similarity to the son clave (sigma = 0.5 pulses) ===\n');
sAbs = simMaet(son, [], others, [], SIGMA, 1, false, true, PERIOD, ...
               'verbose', false);
sRel = simMaet(son, [], others, [], SIGMA, 2, true, true, PERIOD, ...
               'verbose', false);
fprintf('  %-15s r = 1 absolute   r = 2 relative\n', '');
for k = 1:numel(names)
    fprintf('  %-15s %10.3f %16.3f\n', names{k}, sAbs(k), sRel(k));
end

rotations = mod(son + (0:PERIOD - 1)', PERIOD);
sRot = simMaet(son, [], rotations, [], SIGMA, 1, false, true, PERIOD, ...
               'verbose', false);
fprintf('  r = 1, against its rotations by 0-15 pulses:\n    %s\n', ...
        sprintf('%.3f ', sRot));
% The 2-3 son clave is the rotation by 8 pulses: absolute similarity
% 0.314, the lowest of the 16 rotations, relative 1. Each of the other
% five timelines is the son clave with one onset displaced, so they
% share most of its IOIs (relative 0.955-0.984); absolute similarity
% separates them more.

%% === 2. Density: the r = 1 and r = 2 relative densities ===

% Evaluated at a point, the r = 1 density says how strongly an onset is
% expected there; the r = 2 relative density, how many pairs of onsets
% lie that far apart. The latter is symmetric about 8 pulses, since an
% IOI of d pulses read backwards around the cycle is one of 16 - d.
fprintf('\n=== 2. Densities of the son clave (figure) ===\n');
figure('Name', 'Son clave densities', 'Position', [100 100 900 550]);
cfg = {1, false, 'onset (pulse)', 'Son clave: r = 1, absolute';
       2, true,  'IOI (pulses)',  'Son clave: r = 2, relative'};
for k = 1:2
    ax = subplot(2, 1, k);
    dens = buildMaet(son, [], SIGMA, cfg{k, 1}, cfg{k, 2}, true, PERIOD, ...
                     'verbose', false);
    plotMaet(dens, 'method', 'density', 'axes', ax);
    set(ax, 'XTick', 0:PERIOD);
    xlabel(ax, cfg{k, 3});
    ylabel(ax, 'density');
    title(ax, cfg{k, 4});
end
fprintf('  Drawn: r = 1 absolute and r = 2 relative.\n');

%% === 3. Complexity: entropy of the r = 2 relative tensor ===

% The Renyi-2 entropy (minus the logarithm of the integral of the
% squared density, here in bits, computed in closed form) of the r = 2
% relative density is high when a rhythm's IOIs take many different
% sizes in similar numbers, and low when few sizes recur.
fprintf('\n=== 3. IOI-content entropy (Renyi-2, bits) ===\n');
rhythms = {'four on the floor', [0, 4, 8, 12];
           'bossa nova',        [0, 3, 6, 10, 13];
           'son clave',         son;
           'shiko',             [0, 4, 6, 10, 12];
           'clustered',         [0, 1, 2, 3, 9]};
for k = 1:size(rhythms, 1)
    H = entropyMaet(rhythms{k, 2}, [], SIGMA, 2, true, true, PERIOD, ...
                    'method', 'renyi2', 'verbose', false);
    fprintf('  %-17s: %.3f\n', rhythms{k, 1}, H);
end
% Four evenly spaced onsets have IOIs of only 4, 8, and 12 pulses, so
% the lowest entropy. Among the five-onset rhythms, the bossa nova, the
% most evenly spread, repeats IOIs most and scores lowest; the clustered
% rhythm is irregular, yet its IOIs crowd into two groups of
% neighbouring sizes (1-3 and 6-8 pulses), which the smoothing partly
% merges, so it scores below the son clave and the shiko. The measure
% reads the variety of IOIs, not irregularity as such.

%% === 4. Circular differencing and binding, and n-tuple entropy ===

% nTupleEntropy (Milne & Dean, 2016) is the entropy of the n-tuples of
% successive IOIs, the cycle wrapping from the last onset to the first.
% In a pre-MAET each onset is an event holding one element. Circular
% differencing (differenceEvents, 'circular' true) replaces each onset
% by the IOI from the onset before it, and circular binding
% (bindEvents) nests n successive IOIs into one super-event, whose
% MAET's entropy is n-tuple entropy.
fprintf('\n=== 4. Differencing and binding against nTupleEntropy ===\n');

% 4a. The sigma -> 0 limit, on the grid of 16 integer pulses: the
% Shannon entropy of the IOI n-tuple histogram, normalized to [0, 1].
% entropyMaet needs sigma > 0, so a vanishing width of 1e-12 stands in
% for 0, as inside nTupleEntropy.
fprintf('  4a. sigma -> 0, normalized entropy on the 16-pulse grid\n');
steps = differenceEvents(rhythmPm(son, [], PERIOD), 1, 'circular', true);
for n = 1:2
    pmN = bindEvents(steps, n, 'circular', true);
    dens = buildMaet(pmN, 'sigma', 1e-12, 'verbose', false);
    hPipe = entropyMaet(dens, 'method', 'normalized', ...
                        'nPointsPerDim', PERIOD, 'verbose', false);
    hNte = nTupleEntropy(son, PERIOD, n, 'verbose', false);
    fprintf('    n = %d: pipeline %.12f, nTupleEntropy %.12f\n', ...
            n, hPipe, hNte);
end

% 4b. sigma = 0.5 pulses on each onset, Renyi-2 entropy. A difference of
% two onsets each of width sigma has width sigma * sqrt(2), and
% differenceEvents rescales the IOIs' sigma accordingly (it announces
% this). Differencing then binding treats successive IOIs as
% independent, which is nTupleEntropy's 'sigmaSpace' 'interval' at the
% rescaled width. Its default, 'sigmaSpace' 'position', keeps the
% dependence: successive IOIs share an onset, so each pair covaries by
% -sigma^2. Binding n + 1 onsets and taking the super-event relative
% ('relOuter' true), in place of differencing, captures it exactly.
fprintf('  4b. sigma = 0.5 on each onset, Renyi-2 entropy (bits)\n');
steps = differenceEvents(rhythmPm(son, SIGMA, PERIOD), 1, 'circular', true);
fprintf('    IOI sigma after differencing: %.4f\n', steps.specs{1}.sigma);
for n = 1:2
    hDb = entropyMaet(bindEvents(steps, n, 'circular', true), ...
                      'method', 'renyi2', 'verbose', false);
    hBr = entropyMaet(bindEvents(rhythmPm(son, SIGMA, PERIOD), n + 1, ...
                                 'circular', true, 'relOuter', true), ...
                      'method', 'renyi2', 'verbose', false);
    hInt = nTupleEntropy(son, PERIOD, n, 'sigma', SIGMA * sqrt(2), ...
                         'sigmaSpace', 'interval', 'method', 'renyi2', ...
                         'verbose', false);
    hPos = nTupleEntropy(son, PERIOD, n, 'sigma', SIGMA, ...
                         'method', 'renyi2', 'verbose', false);
    fprintf(['    n = %d: difference-bind %.6f = ''interval'' %.6f;  ' ...
             'bind-relative %.6f = ''position'' %.6f\n'], ...
            n, hDb, hInt, hBr, hPos);
end
% Each route equals its nTupleEntropy mode to machine precision. At
% n = 1 there is no neighbouring IOI to covary with, so the two models
% agree; at n = 2 they differ, so "n-tuple entropy" at sigma > 0 needs
% its 'sigmaSpace' stated. At sigma -> 0 (4a) both reduce to the
% histogram.

% 4c. The same pre-MAET goes where nTupleEntropy cannot. A second
% pre-MAET is a query, and simMaet compares the two rhythms' densities
% of successive-IOI pairs: rotation-invariant, like r = 2 relative in
% section 1, but sensitive to the order of the IOIs. (The pre-MAET
% would equally carry per-onset weights, such as accents, for which
% nTupleEntropy has no argument.)
fprintf('  4c. Similarity of successive-IOI pairs (bind-relative, n = 2)\n');
ctx = bindEvents(rhythmPm(son, SIGMA, PERIOD), 3, 'circular', true, ...
                 'relOuter', true);
for k = 2:numel(names)
    q = bindEvents(rhythmPm(others(k, :), SIGMA, PERIOD), 3, ...
                   'circular', true, 'relOuter', true);
    fprintf('    son clave vs %-12s: %.3f\n', names{k}, ...
            simMaet(ctx, q, 'verbose', false));
end
% Each value is lower than its r = 2 relative counterpart in section 1:
% the IOI content of all pairs of onsets discards which IOIs are
% successive and in what order, and that order separates the rhythms
% further.

mptDefaults(prevDefaults);
fprintf('\nDone.\n');

% -------------------------------------------------------------------
function pm = rhythmPm(onsets, sigma, period)
%RHYTHMPM A rhythm as a pre-MAET: one onset per event.
    p = {onsets(:)'};
    specs = flatSpecs(p, 'names', {'onset'}, 'sigma', sigma, ...
                      'per', true, 'period', period);
    pm = packPreMaet(p, [], specs);
end
